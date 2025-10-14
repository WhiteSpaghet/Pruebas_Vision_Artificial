"""
src/model_cnn.py

Definición de la CNN para clasificación de imágenes CIFAR-10.
Incluye:
- build_cnn_model(): crea el modelo con parámetros configurables.
- compile_model(): aplica compilación con optimizer y loss.
- load_model_safe(): función helper para cargar pesos guardados.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# CIFAR-10 clases
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def build_cnn_model(input_shape=(32, 32, 3),
                    num_classes=10,
                    base_filters=32,
                    dropout_rate=0.25):
    """
    Construye un modelo CNN secuencial simple para CIFAR-10.

    Args:
        input_shape (tuple): forma de las imágenes de entrada (H, W, C)
        num_classes (int): número de clases a predecir
        base_filters (int): cantidad de filtros de la primera capa Conv2D
        dropout_rate (float): porcentaje de dropout antes de las Dense layers

    Returns:
        tf.keras.Model: modelo CNN compilado
    """

    model = Sequential(name="cnn_cortex_vision")

    # BLOQUE CONVOLUCIONAL 1
    model.add(Conv2D(base_filters, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # BLOQUE CONVOLUCIONAL 2
    model.add(Conv2D(base_filters*2, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # BLOQUE CONVOLUCIONAL 3 opcional
    model.add(Conv2D(base_filters*4, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Clasificador
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def compile_model(model, learning_rate=1e-3):
    """
    Compila un modelo CNN para clasificación multi-clase.

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
        print("✅ Modelo CNN cargado desde:", path)
        return model
    except Exception as e:
        print("❌ Error cargando modelo:", e)
        return None


# Demo de uso
if __name__ == "__main__":
    model = build_cnn_model()
    model = compile_model(model)
    model.summary()
