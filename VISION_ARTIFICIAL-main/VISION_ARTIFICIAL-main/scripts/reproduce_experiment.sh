#!/usr/bin/env bash
#
# reproduce_experiment.sh
# Reproduce todo el pipeline de experimento: venv, install, train, tuning, convert, export metrics, optional firebase init.
#
# Uso:
#   ./reproduce_experiment.sh --all
#   ./reproduce_experiment.sh --train --seed 42 --epochs 30
#   ./reproduce_experiment.sh --tune --trials 30
#   ./reproduce_experiment.sh --convert --quantization int8 --num-representative 300
#
set -euo pipefail

# ---------- Configurable defaults ----------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"

# paths to scripts (ajusta si moviste archivos)
TRAIN_SCRIPT="${PROJECT_ROOT}/src/train.py"
TUNING_SCRIPT="${PROJECT_ROOT}/src/tuning.py"
CONVERT_SCRIPT="${PROJECT_ROOT}/scripts/convert_to_tflite.py"
EXPORT_SCRIPT="${PROJECT_ROOT}/scripts/export_metrics.py"
INIT_FIREBASE_SCRIPT="${PROJECT_ROOT}/scripts/init_firebase.py"

EXPERIMENTS_DIR="${PROJECT_ROOT}/experiments"
MODELS_DIR="${PROJECT_ROOT}/models"
TFLITE_OUT_DIR="${MODELS_DIR}/tflite"

# defaults for conversions and training
DEFAULT_SEED=42
DEFAULT_EPOCHS=30
DEFAULT_BATCH=64
DEFAULT_QUANT="dynamic"
DEFAULT_REPRESENTATIVE=200
DEFAULT_TRIALS=20

# ---------- Helpers ----------
log(){ printf "\n[%s] %s\n" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"; }
err_exit(){ echo "ERROR: $*" >&2; exit 1; }

print_help() {
  cat <<EOF

reproduce_experiment.sh - reproducir pipeline de experimento VISION_ARTIFICIAL

Opciones:
  --all                     Ejecuta todo el pipeline (install, train, tune, convert, export)
  --install                 Crear venv y pip install -r requirements.txt
  --train                   Ejecutar script de entrenamiento (${TRAIN_SCRIPT})
  --tune                    Ejecutar búsqueda de hiperparámetros (${TUNING_SCRIPT})
  --convert                 Convertir modelos .keras a .tflite (${CONVERT_SCRIPT})
  --export                  Exportar métricas y generar exported_metrics.json/csv (${EXPORT_SCRIPT})
  --firebase-init           Ejecutar scripts/init_firebase.py (requiere --firebase-cred)
  --quantization <type>     Tipo de cuantización (none|dynamic|float16|int8). Default: ${DEFAULT_QUANT}
  --num-representative N    Número muestras representativas para int8. Default: ${DEFAULT_REPRESENTATIVE}
  --seed N                  Semilla reproducible. Default: ${DEFAULT_SEED}
  --epochs N                Epochs para training. Default: ${DEFAULT_EPOCHS}
  --trials N                Número de trials para tuning. Default: ${DEFAULT_TRIALS}
  --overwrite               Forzar overwrite en conversiones / export
  --firebase-cred PATH      Ruta al serviceAccountKey JSON (para --firebase-init)
  -h, --help                Mostrar este mensaje

EOF
}

# ---------- Parse args ----------
DO_INSTALL=0
DO_TRAIN=0
DO_TUNE=0
DO_CONVERT=0
DO_EXPORT=0
DO_ALL=0
DO_FIREBASE_INIT=0
OVERWRITE=0

QUANT=${DEFAULT_QUANT}
NUM_REP=${DEFAULT_REPRESENTATIVE}
SEED=${DEFAULT_SEED}
EPOCHS=${DEFAULT_EPOCHS}
TRIALS=${DEFAULT_TRIALS}
FIREBASE_CRED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) DO_ALL=1; DO_INSTALL=1; DO_TRAIN=1; DO_TUNE=1; DO_CONVERT=1; DO_EXPORT=1; shift ;;
    --install) DO_INSTALL=1; shift ;;
    --train) DO_TRAIN=1; shift ;;
    --tune) DO_TUNE=1; shift ;;
    --convert) DO_CONVERT=1; shift ;;
    --export) DO_EXPORT=1; shift ;;
    --firebase-init) DO_FIREBASE_INIT=1; shift ;;
    --quantization) QUANT="${2:-$QUANT}"; shift 2 ;;
    --num-representative) NUM_REP="${2:-$NUM_REP}"; shift 2 ;;
    --seed) SEED="${2:-$SEED}"; shift 2 ;;
    --epochs) EPOCHS="${2:-$EPOCHS}"; shift 2 ;;
    --trials) TRIALS="${2:-$TRIALS}"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --firebase-cred) FIREBASE_CRED="${2:-}"; shift 2 ;;
    -h|--help) print_help; exit 0 ;;
    *) err_exit "Argumento desconocido: $1" ;;
  esac
done

# ---------- Sanity checks ----------
if [[ $DO_FIREBASE_INIT -eq 1 && -z "$FIREBASE_CRED" ]]; then
  log "Advertencia: --firebase-init usado pero no se pasó --firebase-cred. Saltando firebase init."
  DO_FIREBASE_INIT=0
fi

# ---------- Step: install venv + deps ----------
if [[ $DO_INSTALL -eq 1 ]]; then
  log "=== INSTALL: creando/actualizando virtualenv y dependencias ==="
  if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    err_exit "No se encontró ${REQUIREMENTS_FILE}"
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creando virtualenv en ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
  fi

  # Activate venv for the rest of the script
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"

  log "Upgrade pip y install requirements..."
  "${PIP}" install --upgrade pip setuptools wheel
  "${PIP}" install -r "${REQUIREMENTS_FILE}"
  log "Dependencias instaladas."
else
  # if not installing, try to use existing venv if available
  if [[ -x "${PYTHON}" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
    log "Virtualenv activado desde ${VENV_DIR}."
  else
    log "Virtualenv no activado (ejecuta --install para crear). Asumimos que 'python' del sistema es válido."
  fi
fi

# Ensure experiments and models folders exist
mkdir -p "${EXPERIMENTS_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${TFLITE_OUT_DIR}"

# Set deterministic seeds for reproducibility (env + message)
export PYTHONHASHSEED="${SEED}"
export VISION_EXPERIMENT_SEED="${SEED}"
log "Usando seed=${SEED} (PYTHONHASHSEED and VISION_EXPERIMENT_SEED)"

# ---------- Step: training ----------
if [[ $DO_TRAIN -eq 1 ]]; then
  log "=== TRAIN: ejecutando entrenamiento ==="
  if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    log "Aviso: ${TRAIN_SCRIPT} no encontrado. Asegúrate de tener src/train.py"
  else
    log "Lanzando: ${PYTHON} ${TRAIN_SCRIPT} --seed ${SEED} --epochs ${EPOCHS} --batch ${DEFAULT_BATCH}"
    "${PYTHON}" "${TRAIN_SCRIPT}" --seed "${SEED}" --epochs "${EPOCHS}" --batch "${DEFAULT_BATCH}"
    log "Train finalizado."
  fi
fi

# ---------- Step: hyperparam tuning ----------
if [[ $DO_TUNE -eq 1 ]]; then
  log "=== TUNING: ejecutando búsqueda de hiperparámetros ==="
  if [[ ! -f "${TUNING_SCRIPT}" ]]; then
    log "Aviso: ${TUNING_SCRIPT} no encontrado. Asegúrate de tener src/tuning.py"
  else
    log "Lanzando: ${PYTHON} ${TUNING_SCRIPT} --trials ${TRIALS} --seed ${SEED}"
    "${PYTHON}" "${TUNING_SCRIPT}" --trials "${TRIALS}" --seed "${SEED}"
    log "Tuning finalizado."
  fi
fi

# ---------- Step: convert to tflite ----------
if [[ $DO_CONVERT -eq 1 ]]; then
  log "=== CONVERT: convertir modelos a TFLite (quant=${QUANT}) ==="
  if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    log "Aviso: ${CONVERT_SCRIPT} no encontrado. Asegúrate de tener scripts/convert_to_tflite.py"
  else
    # decide input: models/ directory
    INPUT_MODELS_DIR="${MODELS_DIR}"
    CMD=( "${PYTHON}" "${CONVERT_SCRIPT}" --input "${INPUT_MODELS_DIR}" --output_dir "${TFLITE_OUT_DIR}" --quantization "${QUANT}" --num_representative "${NUM_REP}" --recursive )
    if [[ "${OVERWRITE}" -eq 1 ]]; then
      CMD+=( --overwrite )
    fi
    log "Lanzando: ${CMD[*]}"
    "${CMD[@]}"
    log "Conversiones completadas. Archivos en ${TFLITE_OUT_DIR}"
  fi
fi

# ---------- Step: export metrics ----------
if [[ $DO_EXPORT -eq 1 ]]; then
  log "=== EXPORT: exportar métricas y resumen ==="
  if [[ ! -f "${EXPORT_SCRIPT}" ]]; then
    log "Aviso: ${EXPORT_SCRIPT} no encontrado. Asegúrate de tener scripts/export_metrics.py"
  else
    CMD=( "${PYTHON}" "${EXPORT_SCRIPT}" --experiments-dir "${EXPERIMENTS_DIR}" )
    if [[ "${OVERWRITE}" -eq 1 ]]; then
      CMD+=( --force )
    fi
    log "Lanzando: ${CMD[*]}"
    "${CMD[@]}"
    log "Export de métricas finalizado."
  fi
fi

# ---------- Step: firebase init (optional) ----------
if [[ $DO_FIREBASE_INIT -eq 1 ]]; then
  log "=== FIREBASE INIT: inicialización y seed (opcional) ==="
  if [[ ! -f "${INIT_FIREBASE_SCRIPT}" ]]; then
    log "Aviso: ${INIT_FIREBASE_SCRIPT} no encontrado. Asegúrate de tener scripts/init_firebase.py"
  else
    if [[ -z "${FIREBASE_CRED}" ]]; then
      err_exit "FIREBASE_CRED no definida. Pasa --firebase-cred path/to/creds.json"
    fi
    CMD=( "${PYTHON}" "${INIT_FIREBASE_SCRIPT}" --cred "${FIREBASE_CRED}" --create-collections )
    log "Lanzando: ${CMD[*]}"
    "${CMD[@]}"
    log "Firebase init completado."
  fi
fi

log "=== PIPELINE completado ==="
log "Artefactos generados:"
ls -la "${EXPERIMENTS_DIR}" || true
ls -la "${MODELS_DIR}" || true
ls -la "${TFLITE_OUT_DIR}" || true

log "Fin. Revisa los logs y los archivos en ${EXPERIMENTS_DIR} y ${MODELS_DIR}."

exit 0
