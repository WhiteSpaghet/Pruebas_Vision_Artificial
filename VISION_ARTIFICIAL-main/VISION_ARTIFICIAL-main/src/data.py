"""
src/data.py

Gestión y carga de datos para VISION_ARTIFICIAL (CIFAR-10 u otros).

Incluye:
- Carga del dataset CIFAR-10 (Keras)
- Normalización y preprocesamiento
- División train/val/test
- Soporte para guardar versiones procesadas o aumentadas
- Utilidades para convertir entre arrays e imágenes

Uso:
    from src.data import load_cifar10, get_data_loaders
"""

from __future__ import annotations
import os
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.image import random_flip_left_right, random_brightness, random_contrast
from PIL import Image

from src.utils.config import cfg, ensure_dirs, CIFAR10_LABELS

# --------------------------------------------------------
# 🔧 Helpers básicos
# --------------------------------------------------------
def normalize_images(x: np.ndarray) -> np.ndarray:
    """Normaliza imágenes de [0,255] a [0,1]."""
    return x.astype("float32") / 255.0


def augment_image(img: np.ndarray) -> np.ndarray:
    """Aplica aumento de datos simple con TensorFlow ops."""
    import tensorflow as tf
    img_tf = tf.convert_to_tensor(img)
    img_tf = random_flip_left_right(img_tf)
    img_tf = random_brightness(img_tf, max_delta=0.15)
    img_tf = random_contrast(img_tf, 0.8, 1.2)
    return img_tf.numpy()


def array_to_pil(img: np.ndarray) -> Image.Image:
    """Convierte array (float o uint8) a PIL.Image."""
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


# --------------------------------------------------------
# 📦 Carga CIFAR-10
# --------------------------------------------------------
def load_cifar10(
    normalize: bool = True,
    one_hot: bool = True,
    validation_split: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Carga el dataset CIFAR-10 con opción de normalizar y crear conjunto de validación.
    Retorna un diccionario con train/val/test.
    """
    ensure_dirs(cfg, verbose=False)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if normalize:
        x_train = normalize_images(x_train)
        x_test = normalize_images(x_test)

    if one_hot:
        y_train = to_categorical(y_train, num_classes=10)
        y_test = to_categorical(y_test, num_classes=10)

    # División train/val
    np.random.seed(seed)
    n_val = int(len(x_train) * validation_split)
    idx = np.random.permutation(len(x_train))

    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "labels": CIFAR10_LABELS
    }


# --------------------------------------------------------
# 💾 Guardar datos procesados o aumentados
# --------------------------------------------------------
def save_dataset_split(x: np.ndarray, y: np.ndarray, name: str, folder: Path = cfg.data_processed) -> None:
    """Guarda arrays numpy procesados (para reproducibilidad o inspección)."""
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"{name}.npz", x=x, y=y)
    print(f"[data] ✅ Guardado {name}.npz en {folder}")


def save_sample_images(x: np.ndarray, y: np.ndarray, out_dir: Path, n: int = 20) -> None:
    """Guarda n muestras como imágenes PNG (útil para debugging o informes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, len(x))):
        img = array_to_pil(x[i])
        label = np.argmax(y[i]) if y.ndim > 1 else int(y[i])
        img.save(out_dir / f"sample_{i:03d}_{CIFAR10_LABELS[label]}.png")
    print(f"[data] 🖼️ Guardadas {min(n, len(x))} muestras en {out_dir}")


# --------------------------------------------------------
# 🧠 DataLoaders (para entrenamiento)
# --------------------------------------------------------
def get_data_loaders(
    batch_size: int = 64,
    augment: bool = False,
) -> Tuple[tuple, tuple, tuple]:
    """
    Devuelve (train, val, test) en formato adecuado para entrenamiento con TensorFlow.
    Si augment=True, aplica aumento de datos en el conjunto de entrenamiento.
    """
    import tensorflow as tf

    data = load_cifar10()
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]

    if augment:
        def aug_fn(x, y):
            x = tf.numpy_function(augment_image, [x], tf.float32)
            return x, y
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    ds_train = ds_train.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return ds_train, ds_val, ds_test


# --------------------------------------------------------
# 🔍 Prueba rápida
# --------------------------------------------------------
if __name__ == "__main__":
    print("[data] Cargando CIFAR-10…")
    d = load_cifar10()
    print(f"Train: {d['x_train'].shape}, Val: {d['x_val'].shape}, Test: {d['x_test'].shape}")

    save_dataset_split(d["x_train"], d["y_train"], "train_sample")
    save_sample_images(d["x_train"], d["y_train"], cfg.data_processed / "samples")
