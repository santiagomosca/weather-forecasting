# Weather Temperature Forecasting — MLOps Project

**Autor:** Santiago Mosca  
**Máster en Deep Learning — Asignatura: MLOps**  
**Universidad Politécnica de Madrid**

---

## Descripción

Proyecto de MLOps sobre predicción de series temporales meteorológicas. Dado un histórico de las últimas 24 horas de observaciones atmosféricas (144 pasos de 10 minutos, 19 variables), el modelo predice la temperatura en el siguiente paso temporal (t+10 min).

El modelo seleccionado es una **LSTM** (1 capa, 64 unidades ocultas, ventana de 144 pasos), entrenada sobre el dataset [Weather Long-term Time Series Forecasting](https://www.kaggle.com/datasets/mnassrib/jena-climate). Se compara contra un baseline MLP.

| Modelo         | MAE (°C) | RMSE (°C) |
|----------------|----------|-----------|
| MLP Baseline   | 0.1262   | 0.1742    |
| LSTM P3 (prod) | 0.1032   | 0.1416    |

---

## Enlaces

- **GitHub:** _pendiente_
- **W&B Project:** _pendiente_
- **Endpoint en producción:** _pendiente_

---

## Estructura del proyecto

```
weather-forecasting/
├── api/
│   ├── main.py          # Aplicación FastAPI
│   └── schemas.py       # Modelos Pydantic
├── artifacts/
│   ├── scaler_X.pkl     # Scaler de features
│   ├── scaler_y.pkl     # Scaler del target
│   └── LSTM_p3_w144_h64_l1_test_results.npz
├── data/
│   └── cleaned_weather.csv
├── models/
│   └── lstm_p3_best.ckpt
├── notebooks/
│   └── PRACTICA_DL_MOSCA_SANTIAGO.ipynb
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── training/
│   ├── train_baseline.py
│   └── train_lstm.py
├── conftest.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Requisitos

- Python 3.11+
- Entorno conda o virtualenv con las dependencias del proyecto

---

## Configuración local

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd weather-forecasting
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** `requirements.txt` usa PyTorch CPU. Para entrenamiento con GPU, instalar manualmente `torch==2.12.0+cu130` desde el índice de CUDA correspondiente.

### 3. Configurar W&B (solo para entrenamiento)

```bash
wandb login
```

---

## Entrenamiento

Correr siempre desde la raíz del proyecto:

```bash
# 1. Baseline MLP (genera los scalers en artifacts/)
python training/train_baseline.py

# 2. Propuestas LSTM (P1, P2, P3)
python training/train_lstm.py
```

Cada run queda registrado automáticamente en el proyecto `weather-forecasting` de W&B.

---

## API

### Lanzar el servicio

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8080
```

### Endpoints

| Método | Endpoint   | Descripción                        |
|--------|------------|------------------------------------|
| GET    | `/health`  | Estado del servicio                |
| GET    | `/info`    | Metadata del modelo en producción  |
| POST   | `/predict` | Predicción de temperatura          |

### Ejemplo de predicción

```bash
python -c "
import json, numpy as np
window = np.zeros((144, 19)).tolist()
print(json.dumps({'window': window}))
" | curl -s -X POST http://localhost:8080/predict \
  -H 'Content-Type: application/json' \
  -d @-
```

Respuesta:
```json
{
  "predicted_temperature_celsius": 0.44,
  "model": "LSTM_p3_w144_h64_l1",
  "horizon_minutes": 10
}
```

La documentación interactiva está disponible en `http://localhost:8080/docs`.

---

## Docker

```bash
# Build
docker build -t weather-forecasting .

# Run
docker run -p 8080:8000 weather-forecasting
```

---

## Tests

```bash
pytest tests/ -v
```

16 tests en total: 11 de API y 5 de modelo.
