"""
src/model_mlp.py

Definición de un modelo MLP (Multi-Layer Perceptron) para clasificación de CIFAR-10.
Incluye:
- build_mlp_model(): construye el modelo MLP parametrizable.
- compile_model(): aplica compilación con optimizer y loss.
- load_model_safe(): carga un modelo guardado y maneja errores.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# CIFAR-10 clases
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def build_mlp_model(input_shape=(32, 32, 3),
                    num_classes=10,
                    hidden_units=[512, 256],
                    dropout_rate=0.3):
    """
    Construye un modelo MLP simple para CIFAR-10.

    Args:
        input_shape (tuple): forma de las imágenes de entrada (H, W, C)
        num_classes (int): número de clases
        hidden_units (list[int]): lista con el número de neuronas por capa oculta
        dropout_rate (float): porcentaje de dropout entre capas

    Returns:
        tf.keras.Model: modelo MLP compilado
    """
    model = Sequential(name="mlp_cortex_vision")

    # Aplanamos la entrada (32x32x3 -> 3072)
    model.add(Flatten(input_shape=input_shape))

    # Capas ocultas
    for units in hidden_units:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))

    # Capa de salida
    model.add(Dense(num_classes, activation='softmax'))

    return model


def compile_model(model, learning_rate=1e-3):
    """
    Compila un modelo MLP para clasificación multi-clase.

    Args:
        model (tf.keras.Model): modelo a compilar
        learning_rate (float): tasa de aprendizaje del optimizador Adam

    Returns:
        tf.keras.Model: modelo compilado
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_model_safe(path):
    """
    Carga un modelo Keras desde archivo, maneja errores.

    Args:
        path (str/Path): ruta al modelo (.keras)

    Returns:
        tf.keras.Model | None
    """
    try:
        model = tf.keras.models.load_model(str(path))
        print("✅ Modelo MLP cargado desde:", path)
        return model
    except Exception as e:
        print("❌ Error cargando modelo:", e)
        return None


# Demo de uso
if __name__ == "__main__":
    model = build_mlp_model()
    model = compile_model(model)
    model.summary()
