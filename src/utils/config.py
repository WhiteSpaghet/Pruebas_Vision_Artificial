"""
src/utils/config.py
Configuración central del proyecto CNN-CORTEX-VISION.

Provee:
- Rutas relativas al repositorio
- Parámetros globales (batch, epochs, learning rate)
- CIFAR-10 labels (EN/ES)
- Helpers: ensure_dirs(), set_global_seed(), pretty_print_config()

Uso:
    from src.utils.config import cfg, ensure_dirs, set_global_seed
"""

from __future__ import annotations
import os
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False


# --------------------------------------------------------
# 🧭 Paths base del proyecto
# --------------------------------------------------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]  # cnn-cortex-vision/

# Cargar variables de entorno (.env)
ENV_PATH = PROJECT_ROOT / ".env"
if _HAS_DOTENV and ENV_PATH.exists():
    load_dotenv(dotenv_path=str(ENV_PATH))


# --------------------------------------------------------
# 📚 Etiquetas CIFAR-10
# --------------------------------------------------------
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_LABELS_ES = [
    "Avión", "Coche", "Pájaro", "Gato", "Ciervo",
    "Perro", "Rana", "Caballo", "Barco", "Camión"
]


# --------------------------------------------------------
# ⚙️ Dataclass principal
# --------------------------------------------------------
@dataclass
class Config:
    project_root: Path
    data_dir: Path
    data_raw: Path
    data_processed: Path
    data_augmented: Path

    src_dir: Path
    models_dir: Path
    models_ensemble_dir: Path

    experiments_dir: Path
    reports_dir: Path
    reports_figures_dir: Path

    deployment_dir: Path
    flask_app_dir: Path
    static_dir: Path
    templates_dir: Path

    firebase_config: Optional[Path]

    # training
    random_seed: int
    default_batch_size: int
    default_epochs: int
    default_learning_rate: float

    tf_enable_determinism: bool

    extra: Dict[str, Any]

    def as_dict(self):
        """Convierte a dict plano (Paths -> str)."""
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d


# --------------------------------------------------------
# 🧩 Constructor de configuración
# --------------------------------------------------------
def _path(p: Optional[str]) -> Optional[Path]:
    return Path(p) if p else None


def load_config_from_env() -> Config:
    """Carga configuración con valores de entorno o defaults."""
    data_dir = PROJECT_ROOT / "data"
    src_dir = PROJECT_ROOT / "src"
    models_dir = PROJECT_ROOT / "models"
    experiments_dir = PROJECT_ROOT / "experiments"
    reports_dir = PROJECT_ROOT / "reports"
    deployment_dir = PROJECT_ROOT / "deployment"
    flask_app_dir = deployment_dir / "flask_app"

    firebase_json = (
        os.getenv("FIREBASE_CREDENTIALS")
        or os.getenv("FIREBASE_CONFIG")
        or str(flask_app_dir / "firebase_config.json")
    )

    return Config(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        data_raw=data_dir / "raw",
        data_processed=data_dir / "processed",
        data_augmented=data_dir / "augmented",

        src_dir=src_dir,
        models_dir=models_dir,
        models_ensemble_dir=models_dir / "ensemble",

        experiments_dir=experiments_dir,
        reports_dir=reports_dir,
        reports_figures_dir=reports_dir / "figures",

        deployment_dir=deployment_dir,
        flask_app_dir=flask_app_dir,
        static_dir=flask_app_dir / "static",
        templates_dir=flask_app_dir / "templates",

        firebase_config=_path(firebase_json),

        random_seed=int(os.getenv("VISION_SEED", 42)),
        default_batch_size=int(os.getenv("VISION_BATCH", 64)),
        default_epochs=int(os.getenv("VISION_EPOCHS", 30)),
        default_learning_rate=float(os.getenv("VISION_LR", 1e-3)),

        tf_enable_determinism=os.getenv("TF_DETERMINISM", "1") in ("1", "true", "True", "yes", "on"),

        extra={}
    )


# Singleton global
cfg: Config = load_config_from_env()


# --------------------------------------------------------
# 🛠️ Utilidades globales
# --------------------------------------------------------
def ensure_dirs(config: Optional[Config] = None, verbose: bool = True) -> None:
    """Crea las carpetas básicas necesarias para el proyecto."""
    config = config or cfg
    dirs = [
        config.data_dir,
        config.data_raw,
        config.data_processed,
        config.data_augmented,
        config.models_dir,
        config.models_ensemble_dir,
        config.experiments_dir,
        config.reports_dir,
        config.reports_figures_dir,
        config.deployment_dir,
        config.flask_app_dir,
        config.static_dir,
        config.templates_dir,
    ]
    for d in dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[config] ⚠️ No se pudo crear {d}: {e}")
    if verbose:
        print(f"[config] ✅ Directorios verificados en {config.project_root}")


def set_global_seed(seed: Optional[int] = None, enable_tf_determinism: Optional[bool] = None) -> None:
    """Fija semillas globales (random, numpy, tensorflow) para reproducibilidad."""
    if seed is None:
        seed = cfg.random_seed
    if enable_tf_determinism is None:
        enable_tf_determinism = cfg.tf_enable_determinism

    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if enable_tf_determinism:
            try:
                tf.config.experimental.enable_op_determinism()
            except Exception:
                os.environ["TF_DETERMINISTIC_OPS"] = "1"

            try:
                gpus = tf.config.experimental.list_physical_devices("GPU")
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
    except Exception:
        pass


def pretty_print_config(conf: Optional[Config] = None) -> None:
    """Imprime configuración en formato JSON."""
    conf = conf or cfg
    print(json.dumps(conf.as_dict(), indent=2, ensure_ascii=False))


# --------------------------------------------------------
# Auto-init: asegurar carpetas
# --------------------------------------------------------
try:
    ensure_dirs(cfg, verbose=False)
except Exception:
    pass
