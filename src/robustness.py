"""
src/robustness.py

Funciones para evaluar la robustez de modelos CNN:
- Añadir ruido a imágenes
- Transformaciones simples (rotación, zoom, desplazamiento)
- Métricas de robustez y evaluación batch
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf

from src.utils.config import CIFAR10_LABELS

# ---------------------------
# Transformaciones / Perturbaciones
# ---------------------------

def add_gaussian_noise(img_array: np.ndarray, mean: float = 0.0, std: float = 0.05) -> np.ndarray:
    """
    Añade ruido Gaussiano a una imagen normalizada (0-1)
    """
    noise = np.random.normal(mean, std, img_array.shape).astype(np.float32)
    noisy = img_array + noise
    return np.clip(noisy, 0.0, 1.0)


def random_brightness(img: Image.Image, factor_range: Tuple[float, float] = (0.7, 1.3)) -> Image.Image:
    """
    Ajusta brillo de la imagen con factor aleatorio
    """
    factor = np.random.uniform(*factor_range)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def random_rotation(img: Image.Image, max_deg: float = 20.0) -> Image.Image:
    """
    Rota la imagen aleatoriamente en el rango [-max_deg, max_deg]
    """
    angle = np.random.uniform(-max_deg, max_deg)
    return img.rotate(angle)


def random_blur(img: Image.Image, radius_range: Tuple[float, float] = (0.0, 1.5)) -> Image.Image:
    """
    Aplica desenfoque Gaussiano aleatorio
    """
    radius = np.random.uniform(*radius_range)
    return img.filter(ImageFilter.GaussianBlur(radius))


# ---------------------------
# Funciones de evaluación robusta
# ---------------------------

def preprocess_for_model(img: Image.Image, target_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """
    Convierte PIL.Image a np.ndarray normalizado (0-1) para modelo
    """
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)


def evaluate_model_robustness(model: tf.keras.Model,
                              images: List[Image.Image],
                              labels: List[int],
                              perturb_fn_list: Optional[List] = None) -> Dict[str, float]:
    """
    Evalúa precisión de un modelo bajo perturbaciones.

    Args:
        model: tf.keras.Model entrenado
        images: lista de PIL.Image
        labels: lista de enteros (0-9)
        perturb_fn_list: lista de funciones que reciben PIL.Image y devuelven PIL.Image
    Returns:
        dict con métricas: {perturb_name: accuracy}
    """
    if perturb_fn_list is None:
        perturb_fn_list = []

    results = {}
    for fn in [lambda x: x] + perturb_fn_list:  # incluimos original
        correct = 0
        for img, lbl in zip(images, labels):
            img_pert = fn(img)
            arr = preprocess_for_model(img_pert)
            preds = model.predict(arr, verbose=0)
            pred_idx = int(np.argmax(preds[0]))
            if pred_idx == lbl:
                correct += 1
        key = fn.__name__ if hasattr(fn, "__name__") else str(fn)
        results[key] = correct / len(images)
    return results


# ---------------------------
# Funciones helpers
# ---------------------------

def perturb_batch(images: List[Image.Image],
                  perturb_fn_list: Optional[List] = None) -> List[List[np.ndarray]]:
    """
    Aplica varias perturbaciones a un batch y devuelve arrays listos para modelo
    """
    if perturb_fn_list is None:
        perturb_fn_list = []

    batch_results = []
    for img in images:
        row = []
        for fn in [lambda x: x] + perturb_fn_list:
            img_pert = fn(img)
            arr = preprocess_for_model(img_pert)
            row.append(arr)
        batch_results.append(row)
    return batch_results
