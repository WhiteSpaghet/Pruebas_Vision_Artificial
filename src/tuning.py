"""
src/tuning.py

Búsqueda de hiperparámetros para CNN en VISION_ARTIFICIAL usando Optuna.

Optimiza:
- Número de filtros por capa convolucional
- Tamaño del kernel
- Número de neuronas en la capa densa
- Learning rate
"""

import json
from pathlib import Path

import optuna
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from src.utils.config import cfg, set_global_seed
from src.model_cnn import build_cnn_model

# ---------------------------
# Configuración y reproducibilidad
# ---------------------------
set_global_seed(cfg.random_seed)
HYPERPARAM_TRIALS_PATH = cfg.experiments_dir / "hyperparam_trials.json"
PARAMS_USED_PATH = cfg.experiments_dir / "params_used.json"

# ---------------------------
# Carga de datos CIFAR-10
# ---------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# ---------------------------
# Función objetivo de Optuna
# ---------------------------
def objective(trial):
    # Hiperparámetros
    filters1 = trial.suggest_int("filters1", 16, 64, step=16)
    filters2 = trial.suggest_int("filters2", 32, 128, step=16)
    kernel_size = trial.suggest_int("kernel_size", 3, 5)
    dense_units = trial.suggest_int("dense_units", 32, 128, step=32)
    lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)

    model = build_cnn_model(
        input_shape=x_train.shape[1:],
        num_classes=10,
        conv1_filters=filters1,
        conv2_filters=filters2,
        kernel_size=(kernel_size, kernel_size),
        dense_units=dense_units
    )

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train_cat,
        validation_split=0.1,
        epochs=10,
        batch_size=cfg.default_batch_size,
        verbose=0
    )

    # Optimizamos sobre accuracy de validación
    val_acc = history.history["val_accuracy"][-1]
    return val_acc

# ---------------------------
# Crear estudio y ejecutar búsqueda
# ---------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, timeout=3600)  # timeout en segundos

# ---------------------------
# Guardar resultados de trials
# ---------------------------
trials_data = [trial.params for trial in study.trials]
with open(HYPERPARAM_TRIALS_PATH, "w", encoding="utf-8") as f:
    json.dump(trials_data, f, indent=2)

# Guardar mejores hiperparámetros
best_params = study.best_params
with open(PARAMS_USED_PATH, "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)

print("✅ Hiperparámetros optimizados guardados en:")
print(" - Trials:", HYPERPARAM_TRIALS_PATH)
print(" - Mejor:", PARAMS_USED_PATH)
