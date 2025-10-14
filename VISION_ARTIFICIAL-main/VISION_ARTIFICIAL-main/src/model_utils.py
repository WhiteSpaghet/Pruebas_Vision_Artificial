"""
src/model_utils.py

Funciones auxiliares para manejo de modelos:
- Guardar modelos en formato Keras (.keras)
- Guardar en TFLite (.tflite)
- Cargar modelos con seguridad
- Evaluar modelos sobre dataset
- Exportar métricas
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import json

# ----------------------------------------
# Guardado y carga de modelos
# ----------------------------------------
def save_model_keras(model, path):
    """
    Guarda un modelo Keras en disco.

    Args:
        model (tf.keras.Model): modelo a guardar
        path (str | Path): ruta de destino (.keras)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print(f"✅ Modelo guardado en {path}")


def convert_to_tflite(model, path_tflite):
    """
    Convierte un modelo Keras a TFLite.

    Args:
        model (tf.keras.Model): modelo a convertir
        path_tflite (str | Path): ruta destino (.tflite)
    """
    path_tflite = Path(path_tflite)
    path_tflite.parent.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(path_tflite, "wb") as f:
        f.write(tflite_model)

    print(f"✅ Modelo convertido a TFLite en {path_tflite}")


def load_model_safe(path):
    """
    Carga un modelo Keras desde disco, con manejo de errores.

    Args:
        path (str | Path): ruta del modelo (.keras)

    Returns:
        tf.keras.Model | None
    """
    path = Path(path)
    try:
        model = tf.keras.models.load_model(str(path))
        print(f"✅ Modelo cargado desde {path}")
        return model
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None


# ----------------------------------------
# Evaluación
# ----------------------------------------
def evaluate_model(model, x, y, batch_size=32, verbose=1):
    """
    Evalúa un modelo sobre un dataset y retorna loss y accuracy.

    Args:
        model (tf.keras.Model): modelo a evaluar
        x (np.array): datos de entrada
        y (np.array): etiquetas one-hot
        batch_size (int)
        verbose (int)

    Returns:
        dict: {'loss': float, 'accuracy': float}
    """
    loss, acc = model.evaluate(x, y, batch_size=batch_size, verbose=verbose)
    return {"loss": float(loss), "accuracy": float(acc)}


# ----------------------------------------
# Exportar métricas
# ----------------------------------------
def save_metrics(metrics: dict, path_json):
    """
    Guarda un diccionario de métricas en JSON.

    Args:
        metrics (dict)
        path_json (str | Path)
    """
    path_json = Path(path_json)
    path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Métricas guardadas en {path_json}")


# ----------------------------------------
# Predicción simple
# ----------------------------------------
def predict_batch(model, x, class_names=None):
    """
    Predice sobre un batch de imágenes.

    Args:
        model (tf.keras.Model)
        x (np.array)
        class_names (list[str]): nombres de clases, si None usa índices

    Returns:
        list[dict]: [{'label': str, 'confidence': float}, ...]
    """
    preds = model.predict(x)
    results = []
    for p in preds:
        idx = int(np.argmax(p))
        label = class_names[idx] if class_names else str(idx)
        conf = float(np.max(p))
        results.append({"label": label, "confidence": conf})
    return results


# ----------------------------------------
# Uso rápido de demo
# ----------------------------------------
if __name__ == "__main__":
    print("Módulo model_utils listo para usarse con CNN o MLP.")
