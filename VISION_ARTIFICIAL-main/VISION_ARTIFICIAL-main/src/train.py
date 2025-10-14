"""
src/train.py

Entrenamiento de modelos CNN para VISION_ARTIFICIAL.

Funcionalidades:
- Carga y preprocesamiento del dataset CIFAR-10
- Construcción de modelo CNN desde model_cnn.py
- Entrenamiento con callbacks (EarlyStopping, ModelCheckpoint)
- Guardado de métricas y modelo
- Reproducibilidad usando cfg y set_global_seed
"""

import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from src.utils.config import cfg, set_global_seed
from src.model_cnn import build_cnn_model

# ---------------------------
# Configuración y reproducibilidad
# ---------------------------
set_global_seed(cfg.random_seed)
MODEL_SAVE_PATH = cfg.models_dir / "cnn_best.keras"
HISTORY_SAVE_PATH = cfg.experiments_dir / "history_cnn.json"
BATCH_SIZE = cfg.default_batch_size
EPOCHS = cfg.default_epochs
LR = cfg.default_learning_rate

# ---------------------------
# Carga de datos CIFAR-10
# ---------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# ---------------------------
# Construcción del modelo
# ---------------------------
model = build_cnn_model(input_shape=x_train.shape[1:], num_classes=10)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# Callbacks
# ---------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(str(MODEL_SAVE_PATH), monitor="val_loss", save_best_only=True)
]

# ---------------------------
# Entrenamiento
# ---------------------------
history = model.fit(
    x_train, y_train_cat,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2
)

# ---------------------------
# Guardar historia de entrenamiento
# ---------------------------
history_dict = history.history
with open(HISTORY_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(history_dict, f, indent=2)

print(f"✅ Entrenamiento finalizado. Modelo guardado en: {MODEL_SAVE_PATH}")
print(f"✅ Historia guardada en: {HISTORY_SAVE_PATH}")
