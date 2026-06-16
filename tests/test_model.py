# tests/test_model.py

import numpy as np
import torch
import pytest
from pathlib import Path

# Ruta al checkpoint relativa a la raíz del proyecto
CKPT_PATH = Path(__file__).parent.parent / "models" / "lstm_p3_best.ckpt"

WINDOW_SIZE  = 144
NUM_FEATURES = 19


# -------------------------------------------------------
# Importar la clase del modelo
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
# Fixture: modelo cargado una sola vez por sesión
# -------------------------------------------------------
@pytest.fixture(scope="session")
def model():
    m = LSTMForecaster.load_from_checkpoint(CKPT_PATH, map_location="cpu")
    m.eval()
    return m


# -------------------------------------------------------
# Tests
# -------------------------------------------------------
def test_checkpoint_exists():
    assert CKPT_PATH.exists(), f"Checkpoint no encontrado en {CKPT_PATH}"

def test_model_loads(model):
    assert model is not None

def test_model_output_is_scalar(model):
    x = torch.zeros(1, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert y.shape == torch.Size([]), f"Se esperaba escalar, se obtuvo shape {y.shape}"

def test_model_output_is_float(model):
    x = torch.zeros(1, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y.item(), float)

def test_model_is_deterministic(model):
    x = torch.zeros(1, WINDOW_SIZE, NUM_FEATURES)
    with torch.no_grad():
        y1 = model(x).item()
        y2 = model(x).item()
    assert y1 == y2
