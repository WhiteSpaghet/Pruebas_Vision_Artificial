"""
src/explainability.py

Explicabilidad del modelo VISION_ARTIFICIAL (Grad-CAM y mapas de saliencia).

Permite:
- Calcular Grad-CAM sobre capas convolucionales
- Generar saliency maps basados en gradientes
- Guardar y visualizar mapas de calor superpuestos a la imagen original
- Evaluar interpretabilidad en imágenes del test set

Uso:
    from src.explainability import generate_gradcam, generate_saliency_map
"""

from __future__ import annotations
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.config import cfg, ensure_dirs, CIFAR10_LABELS


# ---------------------------------------------------------------------
# 🔍 Grad-CAM
# ---------------------------------------------------------------------
def generate_gradcam(model: tf.keras.Model, image: np.ndarray,
                     layer_name: Optional[str] = None, class_index: Optional[int] = None) -> np.ndarray:
    """
    Genera un mapa Grad-CAM para una imagen dada y una capa convolucional.

    Args:
        model: modelo Keras entrenado
        image: np.array (H, W, 3) normalizada [0,1]
        layer_name: nombre de la capa objetivo (si None, busca la última Conv2D)
        class_index: índice de clase objetivo (si None, usa la predicha)
    Returns:
        heatmap (H, W) normalizado [0,1]
    """
    if image.ndim == 3:
        img_tensor = np.expand_dims(image, axis=0)
    else:
        img_tensor = image

    # Buscar capa conv final
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
    if layer_name is None:
        raise ValueError("No se encontró capa convolucional para Grad-CAM")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)

    return heatmap.numpy()


# ---------------------------------------------------------------------
# 🧠 Saliency Map
# ---------------------------------------------------------------------
def generate_saliency_map(model: tf.keras.Model, image: np.ndarray,
                          class_index: Optional[int] = None) -> np.ndarray:
    """
    Calcula mapa de saliencia (gradiente de la clase sobre la entrada).
    """
    if image.ndim == 3:
        img_tensor = np.expand_dims(image, axis=0)
    else:
        img_tensor = image

    img_tensor = tf.convert_to_tensor(img_tensor)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    gradient = tape.gradient(loss, img_tensor)[0]
    saliency = tf.reduce_max(tf.abs(gradient), axis=-1)
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) + 1e-8)
    return saliency.numpy()


# ---------------------------------------------------------------------
# 🎨 Superposición visual
# ---------------------------------------------------------------------
def overlay_heatmap_on_image(img: np.ndarray, heatmap: np.ndarray,
                             alpha: float = 0.4, cmap: str = "jet") -> np.ndarray:
    """
    Superpone el mapa de calor sobre la imagen original.
    """
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap(cmap)
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = Image.fromarray((jet_heatmap * 255).astype(np.uint8))
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = np.array(jet_heatmap) / 255.0

    overlay = (jet_heatmap * alpha + img * (1 - alpha))
    overlay = np.clip(overlay, 0, 1)
    return overlay


# ---------------------------------------------------------------------
# 📊 Pipeline de explicabilidad
# ---------------------------------------------------------------------
def explain_single_image(model: tf.keras.Model, image: np.ndarray,
                         label_true: Optional[int] = None, save_path: Optional[Path] = None) -> None:
    """
    Genera y guarda Grad-CAM y Saliency map para una sola imagen.
    """
    ensure_dirs(cfg)

    # Grad-CAM
    heatmap = generate_gradcam(model, image)
    gradcam_overlay = overlay_heatmap_on_image(image, heatmap)

    # Saliency
    saliency = generate_saliency_map(model, image)
    saliency_rgb = np.stack([saliency]*3, axis=-1)

    # Predicción
    preds = model.predict(np.expand_dims(image, 0))
    pred_class = int(np.argmax(preds[0]))
    pred_label = CIFAR10_LABELS[pred_class]

    # Plot final
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title(f"Original\n(True: {CIFAR10_LABELS[label_true] if label_true is not None else '?'})")
    axes[1].imshow(gradcam_overlay)
    axes[1].set_title(f"Grad-CAM ({pred_label})")
    axes[2].imshow(saliency_rgb, cmap="gray")
    axes[2].set_title("Saliency Map")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[explainability] 💾 Guardado en {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------
# 🧪 Uso desde terminal
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from src.data import get_data_loaders

    parser = argparse.ArgumentParser(description="Generar Grad-CAM y Saliency Maps")
    parser.add_argument("--model", type=str, default="models/cnn_best.keras")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    idx = args.index
    img = x_test[idx].astype("float32") / 255.0
    label = int(y_test[idx][0])

    out_path = cfg.reports_figures_dir / f"explain_img_{idx}.png" if args.save else None
    explain_single_image(model, img, label_true=label, save_path=out_path)
