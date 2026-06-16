# Dockerfile

FROM python:3.11-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar requirements e instalar dependencias primero
# (aprovecha el cache de Docker si el requirements no cambia)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar artefactos necesarios para el servicio
COPY artifacts/scaler_X.pkl                        artifacts/scaler_X.pkl
COPY artifacts/scaler_y.pkl                        artifacts/scaler_y.pkl
COPY artifacts/LSTM_p3_w144_h64_l1_test_results.npz artifacts/LSTM_p3_w144_h64_l1_test_results.npz
COPY models/lstm_p3_best.ckpt                      models/lstm_p3_best.ckpt

# Copiar código de la API
COPY api/ api/

# Puerto que expone el servicio
EXPOSE 8000

# Arrancar uvicorn apuntando a api/main.py
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
