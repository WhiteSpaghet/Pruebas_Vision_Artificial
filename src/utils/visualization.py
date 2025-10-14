"""
src/utils/visualization.py
Funciones de visualización para métricas, resultados y explicabilidad del modelo CNN.
Guarda automáticamente las figuras en reports/figures/.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime

# Carpeta destino para guardar las figuras
FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 🧭 Utilidades generales
# ---------------------------------------------------------
def _timestamp():
    """Devuelve timestamp para nombres de archivo."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_fig(fig, name: str, tight=True):
    """Guarda una figura en FIGURES_DIR."""
    filename = f"{name}_{_timestamp()}.png"
    path = FIGURES_DIR / filename
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[visualization] Figura guardada en {path}")
    plt.close(fig)
    return path


# ---------------------------------------------------------
# 📉 Curvas de entrenamiento
# ---------------------------------------------------------
def plot_training_curves(history, title="Entrenamiento CNN", show=False):
    """
    Genera una figura con curvas de loss y accuracy para entrenamiento y validación.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Pérdida
    ax[0].plot(history.history["loss"], label="train_loss", color="tab:blue")
    if "val_loss" in history.history:
        ax[0].plot(history.history["val_loss"], label="val_loss", color="tab:orange")
    ax[0].set_title("Evolución de la pérdida")
    ax[0].set_xlabel("Época")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Precisión
    acc_key = "accuracy" if "accuracy" in history.history else "acc"
    val_acc_key = f"val_{acc_key}"
    ax[1].plot(history.history[acc_key], label="train_acc", color="tab:green")
    if val_acc_key in history.history:
        ax[1].plot(history.history[val_acc_key], label="val_acc", color="tab:red")
    ax[1].set_title("Evolución de la exactitud")
    ax[1].set_xlabel("Época")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    fig.suptitle(title)
    path = _save_fig(fig, "loss_curves")
    if show:
        plt.show()
    return path


# ---------------------------------------------------------
# 🧩 Matriz de confusión
# ---------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True, title="Matriz de confusión", show=False):
    """
    Dibuja una matriz de confusión con etiquetas.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)

    path = _save_fig(fig, "confusion_matrix")
    if show:
        plt.show()
    return path


# ---------------------------------------------------------
# 🧠 Grad-CAM visualization
# ---------------------------------------------------------
def plot_gradcam(model, img_array, class_idx=None, layer_name=None, class_names=None, title="Grad-CAM", show=False):
    """
    Genera visualización Grad-CAM para una imagen de entrada.
    """
    # Seleccionar capa convolucional
    if layer_name is None:
        layer_name = [l.name for l in model.layers if "conv" in l.name][-1]

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Aplicar pesos
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    # Superposición con imagen original
    img = (img_array[0] * 255).astype("uint8")
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.stack([heatmap] * 3, axis=-1)
    heatmap = tf.image.resize(heatmap, (img.shape[0], img.shape[1])).numpy()

    overlay = np.uint8(0.6 * img + 0.4 * heatmap)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Mapa de calor")
    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM superpuesto")

    for ax in axes:
        ax.axis("off")

    if class_names is not None:
        pred_label = class_names[int(class_idx)]
        fig.suptitle(f"{title} - Clase predicha: {pred_label}")
    else:
        fig.suptitle(title)

    path = _save_fig(fig, "gradcam_example")
    if show:
        plt.show()
    return path


# ---------------------------------------------------------
# 📈 Reporte de métricas por clase
# ---------------------------------------------------------
def plot_classification_report(y_true, y_pred, class_names, show=False):
    """
    Genera un heatmap del classification_report de sklearn.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_data = np.array([[v["precision"], v["recall"], v["f1-score"]] for k, v in report.items() if k in class_names])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(report_data, annot=True, cmap="Greens", fmt=".2f",
                xticklabels=["Precision", "Recall", "F1-Score"],
                yticklabels=class_names)
    ax.set_title("Reporte de clasificación")
    path = _save_fig(fig, "classification_report")
    if show:
        plt.show()
    return path
