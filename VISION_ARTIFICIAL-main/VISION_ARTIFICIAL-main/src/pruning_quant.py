"""
src/pruning_quant.py

Funciones para:
- Poda de redes neuronales (Weight Pruning)
- Cuantización de modelos (TFLite)

Requisitos:
- tensorflow>=2.9
- tensorflow-model-optimization
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from pathlib import Path
from src.model_utils import convert_to_tflite, save_model_keras

# ---------------------------
# Pruning (poda de pesos)
# ---------------------------
def prune_model(model, pruning_percent=0.5, input_shape=(32, 32, 3)):
    """
    Aplica pruning a un modelo Keras.

    Args:
        model (tf.keras.Model): modelo a podar
        pruning_percent (float): porcentaje de pesos a podar (0-1)
        input_shape (tuple): shape de entrada (para reconstrucción si es secuencial)

    Returns:
        tf.keras.Model podado
    """
    # política de poda gradual
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=pruning_percent, begin_step=0, frequency=100
        )
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    print(f"✅ Modelo preparado para pruning ({pruning_percent*100:.1f}% de sparsity)")

    return model_for_pruning


def strip_pruning(model):
    """
    Quita las capas de pruning y deja un modelo normal.
    """
    model_stripped = tfmot.sparsity.keras.strip_pruning(model)
    print("✅ Capas de pruning eliminadas, modelo listo para inferencia")
    return model_stripped


# ---------------------------
# Quantization (TFLite)
# ---------------------------
def quantize_model_tflite(model, tflite_path, representative_data=None, int8=False):
    """
    Convierte modelo a TFLite con cuantización opcional.

    Args:
        model (tf.keras.Model)
        tflite_path (str | Path)
        representative_data (generator): para int8 calibración
        int8 (bool): True para INT8 quantization, False float32
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if int8 and representative_data is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        print("⚡ Cuantización INT8 aplicada")
    else:
        print("⚡ Cuantización FLOAT32 (default)")

    tflite_model = converter.convert()
    tflite_path = Path(tflite_path)
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ Modelo TFLite guardado en {tflite_path}")


# ---------------------------
# Ejemplo de uso
# ---------------------------
if __name__ == "__main__":
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

    # modelo de ejemplo
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(32,32,3)),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])
    print("Modelo base creado")

    # aplicar pruning
    pruned_model = prune_model(model, pruning_percent=0.5)
    pruned_model = strip_pruning(pruned_model)

    # guardar modelo podado
    save_model_keras(pruned_model, "models/cnn_pruned.keras")

    # convertir a TFLite
    quantize_model_tflite(pruned_model, "models/cnn_pruned.tflite", int8=False)
