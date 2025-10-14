#!/bin/bash
# ──────────────────────────────────────────────
# Script para ejecutar la app Flask en desarrollo
# ──────────────────────────────────────────────

echo "===== Iniciando VISION_ARTIFICIAL (modo desarrollo) ====="

# Activar entorno virtual si existe
if [ -d "venv" ]; then
  echo "Activando entorno virtual..."
  source venv/bin/activate
fi

# Exportar variables de entorno
export FLASK_APP=app.py
export FLASK_ENV=development
export FLASK_DEBUG=1

# Ejecutar Flask
flask run --host=0.0.0.0 --port=5000
