"""
src/utils/helpers.py
Funciones auxiliares genéricas para VISION_ARTIFICIAL.
Incluye control de semillas, guardado/carga de JSON, métricas, logging, etc.
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------
# 🔁 Control de reproducibilidad
# ---------------------------------------------------------
def set_global_seed(seed: int = 42):
    """
    Fija una semilla global para reproducibilidad total en NumPy, random y TensorFlow.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[helpers] Semilla global fijada en {seed}")


# ---------------------------------------------------------
# 🧾 Manejo de JSON y archivos
# ---------------------------------------------------------
def save_json(obj, path: str | Path, indent: int = 2):
    """
    Guarda un diccionario o lista en un archivo JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    print(f"[helpers] JSON guardado en: {path}")


def load_json(path: str | Path):
    """
    Carga un archivo JSON en memoria. Devuelve None si no existe.
    """
    path = Path(path)
    if not path.exists():
        print(f"[helpers] ⚠️ Archivo no encontrado: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# ---------------------------------------------------------
# ⏱️ Control de tiempo y logs
# ---------------------------------------------------------
def timestamp(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """Devuelve una cadena de tiempo formateada."""
    return datetime.now().strftime(fmt)


def log(msg: str, tag: str = "INFO"):
    """Imprime mensaje con timestamp y tag."""
    print(f"[{timestamp()}][{tag}] {msg}")


# ---------------------------------------------------------
# 📊 Métricas de entrenamiento
# ---------------------------------------------------------
def summarize_history(history, metrics=("loss", "accuracy")) -> dict:
    """
    Resume el historial de entrenamiento Keras.
    """
    result = {}
    for m in metrics:
        val_key = f"val_{m}"
        result[m] = history.history.get(m, [])
        result[val_key] = history.history.get(val_key, [])
        if result[m]:
            log(f"{m}: {result[m][-1]:.4f} | {val_key}: {result[val_key][-1]:.4f}", tag="METRIC")
    return result


def save_training_summary(history, model_name: str, output_dir: str | Path):
    """
    Guarda resumen de entrenamiento (loss y accuracy) en JSON.
    """
    summary = summarize_history(history)
    summary["model_name"] = model_name
    summary["timestamp"] = timestamp()
    path = Path(output_dir) / f"{model_name}_history.json"
    save_json(summary, path)
    return summary


# ---------------------------------------------------------
# 🧠 Utilidades TensorFlow/Keras
# ---------------------------------------------------------
def count_params(model: tf.keras.Model):
    """
    Devuelve el número total de parámetros entrenables y no entrenables.
    """
    trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total = trainable + non_trainable
    log(f"Parámetros totales: {total:,} (entrenables: {trainable:,}, no entrenables: {non_trainable:,})")
    return {"trainable": int(trainable), "non_trainable": int(non_trainable), "total": int(total)}


def print_model_summary(model: tf.keras.Model):
    """
    Imprime el resumen del modelo con conteo de parámetros.
    """
    model.summary()
    count_params(model)


# ---------------------------------------------------------
# 🧩 Utilidades varias
# ---------------------------------------------------------
def ensure_dir(path: str | Path):
    """Crea un directorio si no existe."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(directory: str | Path, extension: str | None = None):
    """
    Lista todos los archivos dentro de un directorio, opcionalmente filtrando por extensión.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    if extension:
        return sorted(directory.glob(f"*.{extension}"))
    return sorted(directory.glob("*"))


def readable_size(num_bytes: int) -> str:
    """
    Convierte bytes en formato legible (KB, MB, GB).
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"
