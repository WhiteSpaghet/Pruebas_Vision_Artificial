"""
src/distillation.py

Implementación de Knowledge Distillation para VISION_ARTIFICIAL.

Permite transferir conocimiento de un modelo "teacher" (grande y preciso)
a un modelo "student" (más pequeño o eficiente).

Características:
- Soporte para entrenamiento combinado (distillation loss + student loss)
- Guarda métricas y pesos del student
- Compatible con modelos Keras / TensorFlow
"""

from __future__ import annotations
import os
import json
import numpy as np
from typing import Dict, Any, Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from src.utils.config import cfg, ensure_dirs

# ------------------------------------------------------------
# 🔧 Distillation Model Wrapper
# ------------------------------------------------------------
class Distiller(Model):
    """
    Keras Model que implementa el proceso de distillation:
    combina la pérdida del estudiante con la pérdida KL-divergence entre
    las predicciones del teacher y del student.
    """

    def __init__(self, student: Model, teacher: Model, temperature: float = 5.0, alpha: float = 0.5):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.student_loss_fn = CategoricalCrossentropy()
        self.distill_loss_fn = CategoricalCrossentropy(from_logits=True)
        self.student_accuracy = CategoricalAccuracy(name="student_acc")

    def compile(self, optimizer, metrics=None):
        super().compile()
        self.optimizer = optimizer
        self.student_metrics = metrics or [self.student_accuracy]

    def train_step(self, data):
        x, y = data
        # Forward pass del teacher (no entrenable)
        teacher_logits = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            # pérdidas
            student_loss = self.student_loss_fn(y, student_logits)
            distill_loss = self.distill_loss_fn(
                tf.nn.softmax(teacher_logits / self.temperature, axis=1),
                tf.nn.softmax(student_logits / self.temperature, axis=1)
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distill_loss

        # backprop
        trainable_vars = self.student.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        # métricas
        self.student_accuracy.update_state(y, student_logits)

        return {
            "loss": loss,
            "student_loss": student_loss,
            "distill_loss": distill_loss,
            "student_acc": self.student_accuracy.result(),
        }

    def test_step(self, data):
        x, y = data
        y_pred = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)
        self.student_accuracy.update_state(y, y_pred)
        return {"student_loss": student_loss, "student_acc": self.student_accuracy.result()}


# ------------------------------------------------------------
# 🧠 Función auxiliar de entrenamiento
# ------------------------------------------------------------
def train_distillation(
    teacher_model: Model,
    student_model: Model,
    train_data,
    val_data,
    epochs: int = 10,
    temperature: float = 5.0,
    alpha: float = 0.5,
    learning_rate: float = 1e-4,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Entrena el modelo de distillation y devuelve métricas.
    Guarda el student model en save_path si se especifica.
    """

    print(f"[distillation] 🔥 Iniciando entrenamiento (α={alpha}, T={temperature})")

    distiller = Distiller(student=student_model, teacher=teacher_model,
                          temperature=temperature, alpha=alpha)
    distiller.compile(optimizer=Adam(learning_rate=learning_rate))

    history = distiller.fit(train_data, validation_data=val_data, epochs=epochs, verbose=1)

    if save_path:
        ensure_dirs(cfg)
        out_path = cfg.models_dir / f"{save_path}_student.keras"
        student_model.save(out_path)
        print(f"[distillation] ✅ Student model guardado en {out_path}")

        # guardar historial de métricas
        hist_path = cfg.experiments_dir / f"distill_history_{save_path}.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history.history, f, indent=2)
        print(f"[distillation] 📊 Historial guardado en {hist_path}")

    results = {
        "final_student_acc": float(history.history["student_acc"][-1]),
        "final_val_acc": float(history.history["val_student_acc"][-1]),
    }

    return results


# ------------------------------------------------------------
# 🔍 Prueba rápida
# ------------------------------------------------------------
if __name__ == "__main__":
    from src.data import get_data_loaders
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten

    ds_train, ds_val, _ = get_data_loaders(batch_size=128, augment=False)

    print("[distillation] Cargando modelos...")

    teacher = MobileNetV2(weights=None, classes=10, input_shape=(32, 32, 3))
    student = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])

    results = train_distillation(
        teacher_model=teacher,
        student_model=student,
        train_data=ds_train,
        val_data=ds_val,
        epochs=3,
        temperature=4.0,
        alpha=0.6,
        learning_rate=1e-4,
        save_path="mobilenetv2_to_dense"
    )

    print("[distillation] Resultados finales:", results)
