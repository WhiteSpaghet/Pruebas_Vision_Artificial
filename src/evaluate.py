"""
src/evaluate.py

Evaluación de modelos para VISION_ARTIFICIAL.

Funciones:
- Cargar modelo entrenado (.keras o .h5)
- Evaluar sobre el set de test (CIFAR-10)
- Calcular métricas (accuracy, precision, recall, F1)
- Generar matriz de confusión y reporte CSV
- Visualizar ejemplos de aciertos y errores

Uso:
    from src.evaluate import evaluate_model

    metrics = evaluate_model("models/cnn_best.keras")
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.utils.config import cfg, ensure_dirs, CIFAR10_LABELS
from src.utils.visualization import (
    plot_confusion_matrix,
    show_predictions_grid
)
from src.data import get_data_loaders


# ---------------------------------------------------------------------
# 🔧 Cargar modelo y dataset
# ---------------------------------------------------------------------
def load_trained_model(model_path: str | Path) -> tf.keras.Model:
    """
    Carga un modelo Keras guardado (.keras o .h5)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    print(f"[evaluate] 🔍 Cargando modelo desde: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


# ---------------------------------------------------------------------
# 📊 Evaluar modelo
# ---------------------------------------------------------------------
def evaluate_model(
    model_path: str | Path,
    batch_size: int = 128,
    save_results: bool = True,
    show_examples: bool = False
) -> Dict[str, float]:
    """
    Evalúa un modelo entrenado sobre el test set.

    Args:
        model_path: ruta del modelo (.keras)
        batch_size: tamaño del batch para inferencia
        save_results: si True, guarda los resultados en /experiments
        show_examples: muestra ejemplos correctos e incorrectos
    Returns:
        dict con accuracy, precision_media, recall_media, f1_media
    """
    ensure_dirs(cfg)
    print("[evaluate] 🚀 Iniciando evaluación...")

    # 1️⃣ Cargar modelo
    model = load_trained_model(model_path)

    # 2️⃣ Cargar dataset de test
    _, _, (x_test, y_test) = get_data_loaders(batch_size=batch_size, as_numpy=True)

    # 3️⃣ Predicciones
    preds = model.predict(x_test, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 4️⃣ Métricas
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CIFAR10_LABELS, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print(f"[evaluate] ✅ Accuracy total: {acc:.4f}")

    # 5️⃣ Guardar resultados
    if save_results:
        df_report = pd.DataFrame(report).transpose()
        out_csv = cfg.experiments_dir / "evaluation_metrics.csv"
        df_report.to_csv(out_csv, index=True)
        print(f"[evaluate] 📁 Resultados guardados en {out_csv}")

        # Gráfica de matriz de confusión
        fig_path = cfg.reports_figures_dir / "confusion_matrix.png"
        plot_confusion_matrix(cm, labels=CIFAR10_LABELS, title="Matriz de confusión (Test)", save_path=fig_path)
        print(f"[evaluate] 📊 Matriz de confusión guardada en {fig_path}")

    # 6️⃣ Visualización de ejemplos opcional
    if show_examples:
        print("[evaluate] 🎨 Mostrando predicciones...")
        show_predictions_grid(x_test, y_true, y_pred, class_names=CIFAR10_LABELS, n=25)

    # 7️⃣ Resumen numérico
    metrics_summary = {
        "accuracy": acc,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"]
    }

    print("[evaluate] 🧾 Métricas principales:")
    for k, v in metrics_summary.items():
        print(f"  {k}: {v:.4f}")

    return metrics_summary


# ---------------------------------------------------------------------
# 🧪 Uso directo desde terminal
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluar modelo entrenado (VISION_ARTIFICIAL)")
    parser.add_argument("--model", type=str, default="models/cnn_best.keras", help="Ruta al modelo .keras")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--show", action="store_true", help="Mostrar ejemplos visuales")

    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        batch_size=args.batch,
        show_examples=args.show,
        save_results=True
    )

    print("\n[evaluate] 🎯 Evaluación completada con éxito:")
    print(results)
