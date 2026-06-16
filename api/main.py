# api/main.py

import pickle
import numpy as np
import torch
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from api.schemas import PredictRequest, PredictResponse, InfoResponse, HealthResponse

# -------------------------------------------------------
# Rutas a artefactos (relativas a la raíz del proyecto)
# -------------------------------------------------------
BASE_DIR      = Path(__file__).parent.parent
SCALER_X_PATH = BASE_DIR / "artifacts" / "scaler_X.pkl"
SCALER_Y_PATH = BASE_DIR / "artifacts" / "scaler_y.pkl"
CKPT_PATH     = BASE_DIR / "models"    / "lstm_p3_best.ckpt"
METRICS_PATH  = BASE_DIR / "artifacts" / "LSTM_p3_w144_h64_l1_test_results.npz"

# -------------------------------------------------------
# Estado global del servicio
# -------------------------------------------------------
state = {}

# -------------------------------------------------------
# Definición del modelo (debe coincidir con train_lstm.py)
# -------------------------------------------------------
import torch.nn as nn
import pytorch_lightning as pl

class LSTMForecaster(pl.LightningModule):
    def __init__(self, input_dim, hidden_size=64, num_layers=1, dropout=0.0, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size =input_dim,
            hidden_size=hidden_size,
            num_layers =num_layers,
            dropout    =dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.output    = nn.Linear(hidden_size, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.output(last_hidden).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# -------------------------------------------------------
# Lifespan: carga de artefactos al arrancar el servicio
# -------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar scalers
    with open(SCALER_X_PATH, 'rb') as f:
        state['scaler_X'] = pickle.load(f)
    with open(SCALER_Y_PATH, 'rb') as f:
        state['scaler_y'] = pickle.load(f)

    # Cargar modelo
    model = LSTMForecaster.load_from_checkpoint(CKPT_PATH, map_location='cpu')
    model.eval()
    state['model'] = model

    # Cargar métricas de test
    metrics = np.load(METRICS_PATH)
    state['test_mae']  = float(metrics['mae'])
    state['test_rmse'] = float(metrics['rmse'])

    yield

    state.clear()


# -------------------------------------------------------
# Aplicación
# -------------------------------------------------------
app = FastAPI(
    title      ="Weather Temperature Forecasting API",
    description="Predice la temperatura en el siguiente paso temporal (t+10 min) a partir de una ventana de 144 observaciones meteorológicas (últimas 24h).",
    version    ="1.0.0",
    lifespan   =lifespan,
)


# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Verifica que el servicio está activo y el modelo cargado."""
    if 'model' not in state:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return HealthResponse(status="ok")


@app.get("/info", response_model=InfoResponse, tags=["Monitoring"])
def info():
    """Devuelve metadata del modelo en producción."""
    return InfoResponse(
        model        ="LSTM_p3_w144_h64_l1",
        window_size  =144,
        num_features =19,
        horizon      =1,
        units        ="°C",
        test_mae     =state['test_mae'],
        test_rmse    =state['test_rmse'],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """
    Recibe una ventana de 144 pasos temporales con 19 features cada uno
    y devuelve la temperatura predicha en °C para t+10 minutos.
    """
    try:
        # Convertir a numpy y normalizar
        X = np.array(request.window, dtype=np.float32)        # (144, 19)
        X_scaled = state['scaler_X'].transform(X)              # (144, 19)

        # Inferencia
        tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 144, 19)
        with torch.no_grad():
            y_scaled = state['model'](tensor).item()

        # Desnormalizar
        y_pred = state['scaler_y'].inverse_transform(
            np.array([[y_scaled]])
        )[0, 0]

        return PredictResponse(
            predicted_temperature_celsius=round(float(y_pred), 4),
            model                        ="LSTM_p3_w144_h64_l1",
            horizon_minutes              =10,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
