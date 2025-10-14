"""
src/ensemble.py

Módulo de Ensemble Learning para VISION_ARTIFICIAL.

Combina múltiples modelos (por ejemplo, CNNs entrenadas con distintas seeds o arquitecturas)
para obtener predicciones más robustas y precisas.

Características:
- Promedio de probabilidades ("soft voting")
- Votación por mayoría ("hard voting")
- Evaluación automática con métricas estándar
- Guardado de resultados CSV y gráficas opcionales

Uso:
    from src.ensemble import load_models, predict_ensemble, evaluate_ensemble
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict

import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils.config import cfg, ensure_dirs
from src.utils.visualization import plot_confusion_matrix

# ---------------------------------------------------------------------
# 🔧 Funciones de carga de modelos
# ---------------------------------------------------------------------
def load_models(model_paths: List[str] | None = None) -> List[tf.keras.Model]:
    """
    Carga múltiples modelos Keras (.keras o .h5) para ensemble.
    """
    ensure_dirs(cfg)
    models_dir = cfg.models_ensemble_dir

    if model_paths is None:
        model_paths = [str(p) for p in models_dir.glob("*.keras")]

    models = []
    for p in model_paths:
        try:
            m = tf.keras.models.load_model(p)
            models.append(m)
            print(f"[ensemble] ✅ Modelo cargado: {p}")
        except Exception as e:
            print(f"[ensemble] ⚠️ No se pudo cargar {p}: {e}")

    return models


# ---------------------------------------------------------------------
# 🔮 Predicción de ensemble
# ---------------------------------------------------------------------
def predict_ensemble(models: List[tf.keras.Model], x: np.ndarray, mode: str = "soft") -> np.ndarray:
    """
    Realiza predicciones combinadas de varios modelos.

    Args:
        models: lista de modelos Keras cargados
        x: batch o dataset numpy (imágenes preprocesadas)
        mode: 'soft' (promedio de probabilidades) o 'hard' (votación por mayoría)
    """
    preds = [m.predict(x, verbose=0) for m in models]
    preds = np.array(preds)  # shape = (n_models, n_samples, n_classes)

    if mode == "soft":
        # Promedio de probabilidades
        return np.mean(preds, axis=0)
    elif mode == "hard":
        # Votación por mayoría
        votes = np.argmax(preds, axis=2)
        maj = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=0, arr=votes)
        # Convertir a one-hot
        return tf.keras.utils.to_categorical(maj, num_classes=preds.shape[-1])
    else:
        raise ValueError("mode debe ser 'soft' o 'hard'.")


# ---------------------------------------------------------------------
# 📊 Evaluación del ensemble
# ---------------------------------------------------------------------
def evaluate_ensemble(models: List[tf.keras.Model], x_test: np.ndarray, y_test: np.ndarray,
                      mode: str = "soft", save_csv: bool = True) -> Dict[str, float]:
    """
    Evalúa el ensemble y devuelve métricas de rendimiento.
    """
    print(f"[ensemble] Evaluando ensemble con modo '{mode}'...")

    y_pred_prob = predict_ensemble(models, x_test, mode=mode)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=None, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print(f"[ensemble] ✅ Accuracy: {acc:.4f}")

    if save_csv:
        ensure_dirs(cfg)
        df_report = pd.DataFrame(report).transpose()
        out_csv = cfg.experiments_dir / "results_ensemble.csv"
        df_report.to_csv(out_csv, index=True)
        print(f"[ensemble] 📁 Reporte guardado en {out_csv}")

        # opcional: guardar matriz de confusión como figura
        try:
            fig_path = cfg.reports_figures_dir / "ensemble_confusion.png"
            plot_confusion_matrix(cm, title=f"Ensemble ({mode})", save_path=fig_path)
            print(f"[ensemble] 📊 Matriz de confusión guardada en {fig_path}")
        except Exception as e:
            print(f"[ensemble] ⚠️ No se pudo generar gráfica de confusión: {e}")

    return {
        "accuracy": acc,
        "n_models": len(models),
        "mode": mode
    }


# ---------------------------------------------------------------------
# 🧪 Ejemplo rápido de uso (modo standalone)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from src.data import get_data_loaders

    print("[ensemble] 🔍 Cargando dataset de prueba...")
    _, _, (x_test, y_test) = get_data_loaders(batch_size=128, as_numpy=True)

    print("[ensemble] 🔍 Cargando modelos...")
    models = load_models()  # carga todos los .keras del ensemble

    if len(models) < 2:
        print("[ensemble] ⚠️ Se requieren al menos 2 modelos para el ensemble.")
        exit()

    results = evaluate_ensemble(models, x_test, y_test, mode="soft", save_csv=True)
    print("[ensemble] 🎯 Resultados finales:", results)
