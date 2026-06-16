#!/usr/bin/env python3
# train_baseline.py

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
from sklearn.preprocessing import StandardScaler
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
# 1. Cargar datos
# -------------------------------------------------------
df = pd.read_csv('data/cleaned_weather.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# -------------------------------------------------------
# 2. Preprocesado: tratamiento de -9999 y clipping
# -------------------------------------------------------
ERROR_VALUE = -9999
n_errors = (df == ERROR_VALUE).sum()
print("Valores -9999 por columna:\n", n_errors[n_errors > 0])

df = df.replace(ERROR_VALUE, np.nan)
df = df.interpolate(method='time')
df = df.ffill().bfill()

cols_to_clip = ['rain', 'raining']
for col in cols_to_clip:
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=upper)
    print(f"Clipping {col}: percentil 99 = {upper:.4f}")

# -------------------------------------------------------
# 3. Features y parámetros
# -------------------------------------------------------
feature_cols = [col for col in df.columns if col != 'T' and df[col].dtype in ['float64', 'int64']]
target_col = 'T'

WINDOW_SIZE   = 4
HORIZON       = 1
BATCH_SIZE    = 8
NUM_WORKERS   = 4
LEARNING_RATE = 0.001

print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")

# -------------------------------------------------------
# 4. Split temporal
# -------------------------------------------------------
n         = len(df)
train_end = int(n * 0.70)
val_end   = train_end + int(n * 0.15)

train_df = df.iloc[:train_end]
val_df   = df.iloc[train_end:val_end]
test_df  = df.iloc[val_end:]

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# -------------------------------------------------------
# 5. Normalización (scaler ajustado solo con train)
# -------------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(train_df[feature_cols].values)
y_train = scaler_y.fit_transform(train_df[target_col].values.reshape(-1, 1)).flatten()

X_val = scaler_X.transform(val_df[feature_cols].values)
y_val = scaler_y.transform(val_df[target_col].values.reshape(-1, 1)).flatten()

X_test = scaler_X.transform(test_df[feature_cols].values)
y_test = scaler_y.transform(test_df[target_col].values.reshape(-1, 1)).flatten()

os.makedirs('artifacts', exist_ok=True)
with open('artifacts/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open('artifacts/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

# -------------------------------------------------------
# 6. Crear secuencias
# -------------------------------------------------------
def create_sequences(X, y, w, h):
    X_seq, y_seq = [], []
    for i in range(len(X) - w - h + 1):
        X_seq.append(X[i:i+w])
        y_seq.append(y[i+w+h-1])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, WINDOW_SIZE, HORIZON)
X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   WINDOW_SIZE, HORIZON)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  WINDOW_SIZE, HORIZON)

print(f"\nSecuencias — Train: {X_train_seq.shape} | Val: {X_val_seq.shape} | Test: {X_test_seq.shape}")

# -------------------------------------------------------
# 7. Dataset y DataLoaders
# -------------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=8, num_workers=4):
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

train_loader, val_loader, test_loader = get_dataloaders(
    TimeSeriesDataset(X_train_seq, y_train_seq),
    TimeSeriesDataset(X_val_seq,   y_val_seq),
    TimeSeriesDataset(X_test_seq,  y_test_seq),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# -------------------------------------------------------
# 8. Modelo
# -------------------------------------------------------
class MLPBaseline(pl.LightningModule):
    def __init__(self, input_dim, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.hidden    = nn.Linear(input_dim, 100)
        self.output    = nn.Linear(100, 1)
        self.relu      = nn.ReLU()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x.squeeze()

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
        return torch.optim.SGD(self.parameters(),
                               lr=self.hparams.learning_rate,
                               momentum=0.9)

# -------------------------------------------------------
# 9. Entrenamiento con WandbLogger
# -------------------------------------------------------
wandb_logger = WandbLogger(
    project='weather-forecasting',
    name   ='MLP_baseline_w4_lr0.001',
    config ={
        'model'        : 'MLP',
        'window_size'  : WINDOW_SIZE,
        'hidden_size'  : 100,
        'learning_rate': LEARNING_RATE,
        'batch_size'   : BATCH_SIZE,
        'optimizer'    : 'SGD',
        'momentum'     : 0.9,
    }
)

checkpoint = ModelCheckpoint(
    monitor   ='val_loss',
    mode      ='min',
    save_top_k=1,
    filename  ='mlp_baseline_best'
)
early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

trainer = pl.Trainer(
    max_epochs       =100,
    callbacks        =[checkpoint, early_stop],
    logger           =wandb_logger,
    accelerator      ='gpu',
    devices          =1,
    log_every_n_steps=50,
)

model = MLPBaseline(input_dim=WINDOW_SIZE * len(feature_cols), learning_rate=LEARNING_RATE)
trainer.fit(model, train_loader, val_loader)

print(f"\nMejor val_loss: {checkpoint.best_model_score:.6f}")
print(f"Checkpoint    : {checkpoint.best_model_path}")

# -------------------------------------------------------
# 10. Evaluación en test
# -------------------------------------------------------
print("\n--- Evaluando en test ---")

best_model = MLPBaseline.load_from_checkpoint(checkpoint.best_model_path)
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
npz_path = 'artifacts/baseline_test_results.npz'
np.savez(
    npz_path,
    y_true=y_true,
    y_pred=y_pred,
    mae   =np.array(mae),
    rmse  =np.array(rmse),
)
print(f"\nResultados guardados en {npz_path}")

# -------------------------------------------------------
# 11. Log de métricas y artefactos en W&B
# -------------------------------------------------------
run = wandb.run

# Métricas finales de test
wandb.log({'test_mae': mae, 'test_rmse': rmse})

# Artefacto: scalers (preprocessing)
scaler_artifact = wandb.Artifact(
    name       ='scalers',
    type       ='preprocessing',
    description='StandardScaler ajustado sobre train set (features y target)'
)
scaler_artifact.add_file('artifacts/scaler_X.pkl')
scaler_artifact.add_file('artifacts/scaler_y.pkl')
run.log_artifact(scaler_artifact)

# Artefacto: checkpoint del modelo
model_artifact = wandb.Artifact(
    name       ='MLP_baseline',
    type       ='model',
    description='MLP baseline entrenado con w=4, lr=0.001, SGD+momentum'
)
model_artifact.add_file(checkpoint.best_model_path)
run.log_artifact(model_artifact)

# Artefacto: resultados de test
eval_artifact = wandb.Artifact(
    name       ='baseline_test_results',
    type       ='evaluation',
    description='Predicciones y métricas del MLP baseline sobre test set'
)
eval_artifact.add_file(npz_path)
run.log_artifact(eval_artifact)

wandb.finish()
