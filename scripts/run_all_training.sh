#!/usr/bin/env bash
#
# run_all_training.sh
#
# Script para entrenamientos completos:
#  - mlp baseline
#  - cnn baseline
#  - tuning (Optuna / búsqueda bayesiana)
#  - reentrenar top trials / ensemble
#  - guarda modelos en models/ y historiales en experiments/
#
# Uso:
#   ./run_all_training.sh --all
#   ./run_all_training.sh --cnn --epochs 50 --batch 128
#   ./run_all_training.sh --tune --trials 40
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_DIR="${PROJECT_ROOT}/venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

REQ_FILE="${PROJECT_ROOT}/requirements.txt"

SRC_DIR="${PROJECT_ROOT}/src"
TRAIN_SCRIPT="${SRC_DIR}/train.py"          # debe aceptar args --model {mlp,cnn} --epochs --batch --seed --out_model
TUNING_SCRIPT="${SRC_DIR}/tuning.py"        # debe aceptar args --trials --seed
ENSEMBLE_SCRIPT="${SRC_DIR}/ensemble.py"    # script para crear ensemble desde modelos guardados (opcional)

MODELS_DIR="${PROJECT_ROOT}/models"
EXPERIMENTS_DIR="${PROJECT_ROOT}/experiments"
LOGS_DIR="${PROJECT_ROOT}/logs"

# Defaults
DO_MLP=0
DO_CNN=0
DO_TUNE=0
DO_ENSEMBLE=0
DO_ALL=0
INSTALL_DEPS=0

EPOCHS=30
BATCH=64
SEED=42
TRIALS=20
GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-}"  # preserve if user set env var
TF_DETERMINISM=1
LOG_SUFFIX="$(date -u +"%Y%m%dT%H%M%SZ")"
VERBOSE=0

print_help() {
  cat <<EOF

run_all_training.sh - Entrena modelos del proyecto VISION_ARTIFICIAL

Opciones:
  --all                   Ejecuta MLP + CNN + tuning + ensemble
  --mlp                   Entrena solo el MLP baseline
  --cnn                   Entrena solo la CNN baseline
  --tune                  Ejecuta búsqueda de hiperparámetros (tuning)
  --ensemble              Construye / entrena ensemble (si existe ensemble.py)
  --install-deps          Crea venv e instala requirements.txt
  --epochs N              Número de epochs (default: ${EPOCHS})
  --batch N               Tamaño de batch (default: ${BATCH})
  --trials N              Número de trials (tuning) (default: ${TRIALS})
  --seed N                Semilla reproducible (default: ${SEED})
  --python PATH           Ruta a binario Python (usa venv si no se pasa)
  --no-venv               No usar virtualenv, ejecutar con python del sistema o --python
  --log-dir PATH          Directorio para logs (default: ${LOGS_DIR})
  --no-tf-determinism     No activar determinismo en TF (por defecto activado)
  -h, --help              Mostrar ayuda

Ejemplos:
  ./run_all_training.sh --all --epochs 40 --batch 128
  ./run_all_training.sh --cnn --epochs 50 --install-deps
  ./run_all_training.sh --tune --trials 50 --seed 123

EOF
}

# parse args
POSITIONAL=()
USE_VENV=1
LOG_DIR="${LOGS_DIR}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --all) DO_ALL=1; DO_MLP=1; DO_CNN=1; DO_TUNE=1; DO_ENSEMBLE=1; shift ;;
    --mlp) DO_MLP=1; shift ;;
    --cnn) DO_CNN=1; shift ;;
    --tune) DO_TUNE=1; shift ;;
    --ensemble) DO_ENSEMBLE=1; shift ;;
    --install-deps) INSTALL_DEPS=1; shift ;;
    --epochs) EPOCHS="${2}"; shift 2 ;;
    --batch) BATCH="${2}"; shift 2 ;;
    --trials) TRIALS="${2}"; shift 2 ;;
    --seed) SEED="${2}"; shift 2 ;;
    --python) PYTHON_BIN="${2}"; USE_VENV=0; shift 2 ;;
    --no-venv) USE_VENV=0; shift ;;
    --log-dir) LOG_DIR="${2}"; shift 2 ;;
    --no-tf-determinism) TF_DETERMINISM=0; shift ;;
    -h|--help) print_help; exit 0 ;;
    --verbose) VERBOSE=1; shift ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

# ensure directories exist
mkdir -p "${MODELS_DIR}"
mkdir -p "${EXPERIMENTS_DIR}"
mkdir -p "${LOG_DIR}"

log() { echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"; }

# create/activate virtualenv and install deps if requested
if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creando virtualenv en ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
  fi
  log "Activando virtualenv..."
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  log "Instalando dependencias desde ${REQ_FILE}..."
  "${PIP_BIN}" install --upgrade pip setuptools wheel
  if [[ -f "${REQ_FILE}" ]]; then
    "${PIP_BIN}" install -r "${REQ_FILE}"
  else
    log "WARNING: ${REQ_FILE} no encontrado. Saltando instalación de requirements."
  fi
else
  if [[ "${USE_VENV}" -eq 1 && -x "${PYTHON_BIN}" ]]; then
    # activate if exists
    log "Activando virtualenv existente en ${VENV_DIR}..."
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
  else
    log "Usando Python del sistema: ${PYTHON_BIN}"
  fi
fi

# final python executable
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

# Set reproducibility env vars
export PYTHONHASHSEED="${SEED}"
export VISION_EXPERIMENT_SEED="${SEED}"
if [[ "${TF_DETERMINISM}" -eq 1 ]]; then
  # TF deterministic env vars (best-effort; exact API depends on TF version)
  export TF_DETERMINISTIC_OPS=1
  export TF_CUDNN_DETERMINISM=1
  log "Activado determinismo TF (env vars): TF_DETERMINISTIC_OPS=1 TF_CUDNN_DETERMINISM=1"
else
  log "Determinismo TF desactivado por opción"
fi

# If user specified GPU devices, export; else keep existing
if [[ -n "${GPU_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
  log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

# helper to run a command and tee its output to a timestamped logfile
run_and_log() {
  local name="$1"; shift
  local logfile="${LOG_DIR}/${name}_${LOG_SUFFIX}.log"
  log "=== RUN ${name} -> ${logfile} ==="
  # run using python wrapper if command starts with python path
  if [[ "$VERBOSE" -eq 1 ]]; then
    "$@" 2>&1 | tee "${logfile}"
  else
    # non-verbose: save output but show progress header only
    ( "$@" ) &> "${logfile}" || { echo "Command failed, see ${logfile}"; return 1; }
  fi
  log "=== DONE ${name} (logs: ${logfile}) ==="
}

# ensure scripts exist and are executable where expected; warn otherwise
warn_if_missing() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    log "WARNING: script not found: ${path}"
  fi
}

warn_if_missing "${TRAIN_SCRIPT}"
warn_if_missing "${TUNING_SCRIPT}"
warn_if_missing "${ENSEMBLE_SCRIPT}"

# Step: MLP baseline
if [[ "${DO_MLP}" -eq 1 ]]; then
  log ">>> Entrenando MLP baseline"
  MLP_MODEL_OUT="${MODELS_DIR}/mlp_baseline.keras"
  MLP_HISTORY_OUT="${EXPERIMENTS_DIR}/history_mlp.json"
  # call train.py --model mlp --epochs ... --batch ... --seed ... --out_model ...
  if [[ -f "${TRAIN_SCRIPT}" ]]; then
    run_and_log "train_mlp" "${PYTHON_BIN}" "${TRAIN_SCRIPT}" --model mlp --epochs "${EPOCHS}" --batch "${BATCH}" --seed "${SEED}" --out_model "${MLP_MODEL_OUT}" --history_out "${MLP_HISTORY_OUT}"
  else
    log "No se encontró ${TRAIN_SCRIPT}, se omite MLP."
  fi
fi

# Step: CNN baseline
if [[ "${DO_CNN}" -eq 1 ]]; then
  log ">>> Entrenando CNN baseline"
  CNN_MODEL_OUT="${MODELS_DIR}/cnn_best.keras"
  CNN_HISTORY_OUT="${EXPERIMENTS_DIR}/history_cnn.json"
  if [[ -f "${TRAIN_SCRIPT}" ]]; then
    run_and_log "train_cnn" "${PYTHON_BIN}" "${TRAIN_SCRIPT}" --model cnn --epochs "${EPOCHS}" --batch "${BATCH}" --seed "${SEED}" --out_model "${CNN_MODEL_OUT}" --history_out "${CNN_HISTORY_OUT}"
  else
    log "No se encontró ${TRAIN_SCRIPT}, se omite CNN."
  fi
fi

# Step: Hyperparameter tuning
if [[ "${DO_TUNE}" -eq 1 ]]; then
  log ">>> Ejecutando tuning (búsqueda de hiperparámetros)"
  HYPER_TRIALS_OUT="${EXPERIMENTS_DIR}/hyperparam_trials.json"
  if [[ -f "${TUNING_SCRIPT}" ]]; then
    run_and_log "tuning" "${PYTHON_BIN}" "${TUNING_SCRIPT}" --trials "${TRIALS}" --seed "${SEED}" --out_json "${HYPER_TRIALS_OUT}"
  else
    log "No se encontró ${TUNING_SCRIPT}, se omite tuning."
  fi
fi

# Step: Ensemble (retrain / combine)
if [[ "${DO_ENSEMBLE}" -eq 1 ]]; then
  log ">>> Creando / entrenando ensemble"
  ENSEMBLE_OUTPUT="${MODELS_DIR}/ensemble/ensemble_model.keras"
  mkdir -p "${MODELS_DIR}/ensemble"
  if [[ -f "${ENSEMBLE_SCRIPT}" ]]; then
    run_and_log "ensemble" "${PYTHON_BIN}" "${ENSEMBLE_SCRIPT}" --models_dir "${MODELS_DIR}/ensemble" --out "${ENSEMBLE_OUTPUT}"
  else
    log "No se encontró ${ENSEMBLE_SCRIPT}. Si quieres crear ensemble, implementa src/ensemble.py o usa herramientas manuales."
  fi
fi

log "=== Entrenamientos completados ==="
log "Modelos guardados en: ${MODELS_DIR}"
log "Historias en: ${EXPERIMENTS_DIR}"
log "Logs en: ${LOG_DIR}"

exit 0
