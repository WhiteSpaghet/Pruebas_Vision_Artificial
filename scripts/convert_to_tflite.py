#!/usr/bin/env python3
"""
convert_to_tflite.py

Convierte modelos Keras (.keras / .h5) a TensorFlow Lite (.tflite) con opciones:
  - quantization: none | dynamic | float16 | int8
  - para int8, soporta generador representativo automático usando CIFAR-10
  - puede convertir un archivo o todos los .keras en un directorio

Ejemplos:
  python convert_to_tflite.py --input models/cnn_best.keras --output_dir models/tflite --quantization dynamic
  python convert_to_tflite.py --input models/ --quantization int8 --num_representative 200

Requisitos:
  tensorflow instalado (compatible con el modelo)
"""

import os
import argparse
import logging
from pathlib import Path
import sys

try:
    import tensorflow as tf
    from tensorflow import lite as tflite
except Exception as e:
    print("ERROR: TensorFlow no está disponible. Instala tensorflow antes de usar este script.")
    raise

# ----------------------------
# Helpers: representative dataset
# ----------------------------
def representative_dataset_from_cifar(num_samples=100):
    """
    Generador representativo usando CIFAR-10 (normalizado a [0,1]) adaptado al tamaño 32x32x3.
    Devuelve float32 arrays shape (1, h, w, c).
    """
    try:
        (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    except Exception as e:
        logging.error("No se pudo cargar CIFAR-10 automáticamente: %s", e)
        raise

    x_train = x_train.astype("float32") / 255.0
    count = min(num_samples, x_train.shape[0])

    def gen():
        for i in range(count):
            img = x_train[i]
            img = img.reshape((1,) + img.shape)  # (1,h,w,c)
            yield [img]
    return gen

def representative_dataset_from_folder(folder_path, target_size=(32,32), num_samples=100):
    """
    Generador representativo desde un folder con imágenes. Usa PIL para abrir y redimensionar.
    Retorna función generadora.
    """
    from PIL import Image
    import numpy as np

    folder = Path(folder_path)
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")]
    if len(imgs) == 0:
        raise ValueError(f"No se encontraron imágenes en {folder_path}")

    def gen():
        for i, p in enumerate(imgs):
            if i >= num_samples:
                break
            im = Image.open(p).convert("RGB")
            im = im.resize(target_size, Image.BILINEAR)
            arr = (np.asarray(im).astype("float32") / 255.0).reshape((1,) + (target_size[0], target_size[1], 3))
            yield [arr]
    return gen

# ----------------------------
# Conversion logic
# ----------------------------
def convert_model_to_tflite(input_path, output_path, quantization=None, representative_gen=None, optimizations=None):
    """
    input_path: path to keras model (.keras or .h5) or SavedModel directory
    output_path: where to write .tflite
    quantization: one of (None, 'dynamic', 'float16', 'int8')
    representative_gen: callable that returns generator for int8 quant (each yield -> [input])
    optimizations: optional list of optimizations to set on converter
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model (use converter.from_keras_model if possible or from_saved_model)
    try:
        # prefer to load model into memory (works for .keras/.h5)
        keras_model = tf.keras.models.load_model(str(input_path))
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        logging.info("Usando TFLiteConverter.from_keras_model para %s", input_path)
    except Exception as e:
        logging.warning("No se pudo cargar con load_model(), intentando desde SavedModel: %s", e)
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(str(input_path))
            logging.info("Usando TFLiteConverter.from_saved_model para %s", input_path)
        except Exception as e2:
            logging.exception("No se pudo crear converter desde el modelo: %s", e2)
            raise

    # Apply optimizations
    if optimizations is None:
        optimizations = [tf.lite.Optimize.DEFAULT]
    converter.optimizations = optimizations

    # Configure by quantization type
    quantization = (quantization or "").lower()
    if quantization in ("", "none", "no", None):
        logging.info("Sin cuantización (float32) para %s", input_path)
        # default converter settings suffice
    elif quantization == "dynamic":
        logging.info("Cuantización dinámica para %s", input_path)
        # dynamic quantization: only weights quantized, activations float
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # no further config required
    elif quantization == "float16":
        logging.info("Cuantización float16 para %s", input_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        logging.info("Cuantización INT8 (full integer) para %s", input_path)
        if representative_gen is None:
            raise ValueError("Para int8 se requiere un 'representative_gen' (generador representativo)")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # ensure we set target ops for full integer
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set representative dataset
        converter.representative_dataset = representative_gen()
        # ensure input/output types as int8
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    else:
        raise ValueError(f"Tipo de cuantización desconocido: {quantization}")

    # Convert
    logging.info("Convirtiendo modelo...")
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    logging.info("Modelo guardado en %s (size: %.2f KB)", output_path, output_path.stat().st_size / 1024.0)
    return output_path

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(prog="convert_to_tflite.py", description="Convert Keras model(s) to TFLite with quantization options")
    p.add_argument("--input", "-i", required=True, help="File (.keras/.h5) or directory containing models to convert")
    p.add_argument("--output_dir", "-o", default="models/tflite", help="Output directory for .tflite files")
    p.add_argument("--quantization", "-q", choices=["none","dynamic","float16","int8"], default="dynamic", help="Type of quantization")
    p.add_argument("--representative_folder", "-r", default=None, help="Folder with images to use as representative dataset (optional). If not provided and quantization=int8, uses CIFAR-10 by default.")
    p.add_argument("--num_representative", "-n", type=int, default=200, help="Number of representative samples to use for int8 quant")
    p.add_argument("--target_size", type=int, nargs=2, metavar=("H","W"), default=(32,32), help="Target H W for representative images (default 32x32)")
    p.add_argument("--recursive", action="store_true", help="When input is directory, search recursively for .keras/.h5 files")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .tflite files")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return p.parse_args()

def main():
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # build list of model files
    model_files = []
    if input_path.is_dir():
        pattern = "**/*.keras" if args.recursive else "*.keras"
        model_files.extend(list(input_path.glob(pattern)))
        # also accept .h5
        if args.recursive:
            model_files.extend(list(input_path.glob("**/*.h5")))
        else:
            model_files.extend(list(input_path.glob("*.h5")))
        model_files = sorted(set(model_files))
        if len(model_files) == 0:
            logging.error("No se encontraron archivos .keras o .h5 en %s", input_path)
            sys.exit(1)
    elif input_path.is_file():
        model_files = [input_path]
    else:
        logging.error("Input path no existe: %s", input_path)
        sys.exit(1)

    # prepare representative generator if needed
    rep_gen_factory = None
    if args.quantization == "int8":
        if args.representative_folder:
            try:
                rep_gen_factory = lambda : representative_dataset_from_folder(args.representative_folder, target_size=tuple(args.target_size), num_samples=args.num_representative)
                # test constructing
                _ = rep_gen_factory()
            except Exception as e:
                logging.error("Error preparando representante desde carpeta: %s", e)
                sys.exit(1)
        else:
            # use CIFAR-10 generator
            try:
                rep_gen_factory = lambda : representative_dataset_from_cifar(num_samples=args.num_representative)
            except Exception as e:
                logging.error("No se pudo usar CIFAR-10 para representante: %s", e)
                sys.exit(1)

    # convert each model
    for model_path in model_files:
        out_name = model_path.stem + ".tflite"
        out_path = output_dir / out_name
        if out_path.exists() and not args.overwrite:
            logging.info("Saltando %s (ya existe). Usa --overwrite para forzar", out_path)
            continue
        try:
            rep = rep_gen_factory if args.quantization == "int8" else None
            convert_model_to_tflite(model_path, out_path, quantization=(args.quantization if args.quantization != "none" else None), representative_gen=rep)
        except Exception as e:
            logging.exception("Error convirtiendo %s : %s", model_path, e)

    logging.info("Conversión completada.")

if __name__ == "__main__":
    main()
