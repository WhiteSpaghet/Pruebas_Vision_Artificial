import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2


# ----------------- CARGA DE MODELO ------------------ #

MODEL_PATH = os.path.join("models", "cnn_best.keras")

try:
    model = load_model(MODEL_PATH)
    print("[INFO] Modelo cargado correctamente:", MODEL_PATH)
except Exception as e:
    print("[ERROR] No se pudo cargar el modelo:", e)
    model = None


# ----------------- PREPROCESAR IMAGEN ------------------ #

def preprocess_image(img_path, target_size=(32, 32)):
    """
    Preprocesa una imagen para la CNN
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ----------------- PREDICCIONES ------------------ #

def predict_image(img_path, labels):
    """
    Predice una única imagen
    """
    img = preprocess_image(img_path)
    preds = model.predict(img)
    class_index = np.argmax(preds)
    return labels[class_index], preds[0][class_index]


def predict_batch(image_paths, labels):
    """
    Predicciones por lote (batch)
    """
    results = []
    for path in image_paths:
        label, confidence = predict_image(path, labels)
        results.append((path, label, confidence))
    return results


# ----------------- GRAD-CAM ------------------ #

def generate_gradcam(img_path, last_conv_layer_name="conv2d"):
    """
    Genera mapa de calor Grad-CAM para explicar la predicción de la CNN
    """
    img = preprocess_image(img_path)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        pred_class = tf.argmax(predictions[0])
        loss = predictions[:, pred_class]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    img_original = cv2.imread(img_path)
    img_original = cv2.resize(img_original, (32, 32))
    heatmap = cv2.resize(heatmap.numpy(), (img_original.shape[1], img_original.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    super_imposed = cv2.addWeighted(img_original, 0.6, heatmap, 0.4, 0)

    return super_imposed


# ----------------- VALIDACIÓN ------------------ #

def allowed_file(filename):
    """
    Comprueba si el archivo es una imagen válida
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}
