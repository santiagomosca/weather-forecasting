#!/usr/bin/env python3
# train_lstm.py

import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import Dataset, DataLoader
import wandb

# -------------------------------------------------------
# Reproducibilidad
# -------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
pl.seed_everything(SEED, workers=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

# -------------------------------------------------------
# 1. Cargar datos y preprocesado
# -------------------------------------------------------
df = pd.read_csv('data/cleaned_weather.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

df = df.replace(-9999, np.nan)
df = df.interpolate(method='time')
df = df.ffill().bfill()

for col in ['rain', 'raining']:
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=upper)

# -------------------------------------------------------
# 2. Features y split
# -------------------------------------------------------
feature_cols = [col for col in df.columns if col != 'T' and df[col].dtype in ['float64', 'int64']]
target_col   = 'T'
HORIZON      = 1
NUM_WORKERS  = 4

n         = len(df)
train_end = int(n * 0.70)
val_end   = train_end + int(n * 0.15)

train_df = df.iloc[:train_end]
val_df   = df.iloc[train_end:val_end]
test_df  = df.iloc[val_end:]

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# -------------------------------------------------------
# 3. Normalización con scalers del baseline
# -------------------------------------------------------
with open('artifacts/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('artifacts/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

X_train_scaled = scaler_X.transform(train_df[feature_cols].values)
y_train_scaled = scaler_y.transform(train_df[target_col].values.reshape(-1, 1)).flatten()

X_val_scaled  = scaler_X.transform(val_df[feature_cols].values)
y_val_scaled  = scaler_y.transform(val_df[target_col].values.reshape(-1, 1)).flatten()

X_test_scaled = scaler_X.transform(test_df[feature_cols].values)
y_test_scaled = scaler_y.transform(test_df[target_col].values.reshape(-1, 1)).flatten()

# -------------------------------------------------------
# 4. Funciones auxiliares
# -------------------------------------------------------
def create_sequences(X, y, w, h):
    X_seq, y_seq = [], []
    for i in range(len(X) - w - h + 1):
        X_seq.append(X[i:i+w])
        y_seq.append(y[i+w+h-1])
    return np.array(X_seq), np.array(y_seq)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker,
        generator=g, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        generator=g, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        generator=g, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    return train_loader, val_loader, test_loader

# -------------------------------------------------------
# 5. Modelo
# -------------------------------------------------------
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
# 6. Configuraciones de las tres propuestas
# -------------------------------------------------------
lstm_configs = [
    {
        'nombre'       : 'LSTM_p1_w36_h64_l1',
        'window_size'  : 36,
        'hidden_size'  : 64,
        'num_layers'   : 1,
        'dropout'      : 0.0,
        'learning_rate': 0.001,
        'batch_size'   : 32,
    },
    {
        'nombre'       : 'LSTM_p2_w144_h128_l2',
        'window_size'  : 144,
        'hidden_size'  : 128,
        'num_layers'   : 2,
        'dropout'      : 0.2,
        'learning_rate': 0.001,
        'batch_size'   : 64,
    },
    {
        'nombre'       : 'LSTM_p3_w144_h64_l1',
        'window_size'  : 144,
        'hidden_size'  : 64,
        'num_layers'   : 1,
        'dropout'      : 0.0,
        'learning_rate': 0.001,
        'batch_size'   : 32,
    },
]

os.makedirs('artifacts', exist_ok=True)

# -------------------------------------------------------
# 7. Loop de entrenamiento — un run de W&B por configuración
# -------------------------------------------------------
for config in lstm_configs:
    nombre = config['nombre']
    w      = config['window_size']

    print(f"\n{'='*50}")
    print(f"Configuración: {nombre}")
    print(f"{'='*50}")

    # Secuencias
    X_tr_seq, y_tr_seq = create_sequences(X_train_scaled, y_train_scaled, w, HORIZON)
    X_va_seq, y_va_seq = create_sequences(X_val_scaled,   y_val_scaled,   w, HORIZON)
    X_te_seq, y_te_seq = create_sequences(X_test_scaled,  y_test_scaled,  w, HORIZON)

    print(f"Secuencias — Train: {X_tr_seq.shape} | Val: {X_va_seq.shape} | Test: {X_te_seq.shape}")

    train_loader, val_loader, test_loader = get_dataloaders(
        TimeSeriesDataset(X_tr_seq, y_tr_seq),
        TimeSeriesDataset(X_va_seq, y_va_seq),
        TimeSeriesDataset(X_te_seq, y_te_seq),
        batch_size  =config['batch_size'],
        num_workers =NUM_WORKERS
    )

    # W&B: un run por configuración
    wandb_logger = WandbLogger(
        project='weather-forecasting',
        name   =nombre,
        config ={
            'model'        : 'LSTM',
            'window_size'  : config['window_size'],
            'hidden_size'  : config['hidden_size'],
            'num_layers'   : config['num_layers'],
            'dropout'      : config['dropout'],
            'learning_rate': config['learning_rate'],
            'batch_size'   : config['batch_size'],
            'optimizer'    : 'Adam',
        }
    )

    # Modelo
    model = LSTMForecaster(
        input_dim    =X_tr_seq.shape[2],
        hidden_size  =config['hidden_size'],
        num_layers   =config['num_layers'],
        dropout      =config['dropout'],
        learning_rate=config['learning_rate']
    )

    checkpoint = ModelCheckpoint(
        monitor   ='val_loss',
        mode      ='min',
        save_top_k=1,
        filename  =f'{nombre}_best'
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=10, mode='min')

    trainer = pl.Trainer(
        max_epochs        =100,
        callbacks         =[checkpoint, early_stop],
        logger            =wandb_logger,
        accelerator       ='gpu',
        devices           =1,
        log_every_n_steps =10,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Mejor val_loss: {checkpoint.best_model_score:.6f}")

    # Evaluación en test
    print(f"\n--- Evaluando {nombre} en test ---")
    best_model = LSTMForecaster.load_from_checkpoint(checkpoint.best_model_path)
    best_model.eval()
    best_model.to('cpu')

    y_preds, y_trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            y_hat = best_model(x)
            y_preds.append(y_hat.numpy())
            y_trues.append(y.numpy())

    y_pred_scaled = np.concatenate(y_preds)
    y_true_scaled = np.concatenate(y_trues)

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"MAE  : {mae:.4f} °C")
    print(f"RMSE : {rmse:.4f} °C")

    # Guardar resultados en disco
    npz_path = f'artifacts/{nombre}_test_results.npz'
    np.savez(
        npz_path,
        y_true =y_true,
        y_pred =y_pred,
        mae    =np.array(mae),
        rmse   =np.array(rmse),
        config =np.array(str(config))
    )
    print(f"Resultados guardados en {npz_path}")

    # -------------------------------------------------------
    # Log de métricas y artefactos en W&B
    # -------------------------------------------------------
    run = wandb.run

    # Métricas finales de test
    wandb.log({'test_mae': mae, 'test_rmse': rmse})

    # Artefacto: checkpoint del modelo
    model_artifact = wandb.Artifact(
        name       =nombre,
        type       ='model',
        description=f'LSTM {nombre} — w={config["window_size"]}, hidden={config["hidden_size"]}, layers={config["num_layers"]}'
    )
    model_artifact.add_file(checkpoint.best_model_path)
    run.log_artifact(model_artifact)

    # Artefacto: resultados de test
    eval_artifact = wandb.Artifact(
        name       =f'{nombre}_test_results',
        type       ='evaluation',
        description=f'Predicciones y métricas de {nombre} sobre test set'
    )
    eval_artifact.add_file(npz_path)
    run.log_artifact(eval_artifact)

    wandb.finish()
