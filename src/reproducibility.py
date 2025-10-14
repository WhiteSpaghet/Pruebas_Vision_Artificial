"""
src/reproducibility.py

Funciones para asegurar reproducibilidad en experimentos:
- Semillas globales
- Configuración de TensorFlow determinista
- Control de cuántos hilos usar
- Guardado de estado de random, numpy y TF
"""

import os
import random
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from src.utils.config import cfg


# ---------------------------
# Seed global
# ---------------------------
def set_global_seed(seed: int = None, tf_deterministic: bool = None):
    """
    Fija seeds para Python, NumPy y TensorFlow.

    Args:
        seed (int): seed numérica. Default: cfg.random_seed
        tf_deterministic (bool): fuerza TensorFlow determinista. Default: cfg.tf_enable_determinism
    """
    if seed is None:
        seed = cfg.random_seed
    if tf_deterministic is None:
        tf_deterministic = cfg.tf_enable_determinism

    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # Hash
    os.environ["PYTHONHASHSEED"] = str(seed)

    # TensorFlow
    if tf is not None:
        tf.random.set_seed(seed)

        if tf_deterministic:
            try:
                # TF >= 2.9
                tf.config.experimental.enable_op_determinism()
                print("[reproducibility] TensorFlow op determinism enabled")
            except Exception:
                os.environ["TF_DETERMINISTIC_OPS"] = "1"
                print("[reproducibility] fallback: TF_DETERMINISTIC_OPS set")

        # limitar memoria GPU
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    print(f"[reproducibility] Global seed set: {seed}")


# ---------------------------
# Funciones para reproducir entrenamiento
# ---------------------------
def reproducible_session():
    """
    Devuelve un contexto/session compatible con reproducibilidad.
    """
    set_global_seed()
    if tf is not None:
        # opcional: reinicia graph de TF 1.x si se usa
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass


# ---------------------------
# Guardar y cargar estado de RNG
# ---------------------------
def save_rng_state(path: str):
    """
    Guarda el estado de random, numpy y TF en un archivo npz
    """
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
    }

    if tf is not None:
        try:
            # TF no tiene dumpable state completo, solo seed
            state["tf_seed"] = cfg.random_seed
        except Exception:
            pass

    np.savez(path, **state)
    print(f"[reproducibility] RNG state saved: {path}")


def load_rng_state(path: str):
    """
    Carga estado de RNG desde un archivo npz
    """
    data = np.load(path, allow_pickle=True)
    random.setstate(data["python_random"].item())
    np.random.set_state(data["numpy_random"])
    if tf is not None:
        try:
            tf.random.set_seed(int(data.get("tf_seed", cfg.random_seed)))
        except Exception:
            pass
    print(f"[reproducibility] RNG state restored: {path}")


# ---------------------------
# Auto-setup al importar
# ---------------------------
try:
    reproducible_session()
except Exception:
    pass
