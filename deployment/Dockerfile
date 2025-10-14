# ──────────────────────────────────────────────
# Dockerfile para VISION_ARTIFICIAL
# ──────────────────────────────────────────────

# Usar Python 3.10 como base
FROM python:3.10-slim

# Establecer working directory
WORKDIR /app

# Evitar bytecode y buffering de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar aplicación
COPY . .

# Variables de entorno por defecto
ENV FLASK_ENV=production
ENV APP_PORT=5000

# Exponer puerto
EXPOSE 5000

# Comando para producción con Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
