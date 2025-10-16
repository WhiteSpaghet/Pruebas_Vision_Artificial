# Pruebas_Vision_Artificial (MLP / CNN)

Resumen
- Proyecto de ejemplo para clasificación/experimentos de visión artificial usando MLP y/o CNN.
- Incluye scripts para preprocesado, entrenamiento, evaluación e inferencia.

Índice
- Requisitos
- Estructura del proyecto
- Preparar datos
- Instalación
- Uso (entrenar / evaluar / predecir)
- Problemas comunes y soluciones
- Reproducibilidad y notas finales
- Licencia

Requisitos
- Python 3.8+ (recomendado)
- pip
- GPU opcional (CUDA compatible) para acelerar entrenamiento
- Dependencias típicas (ejemplos): torch, torchvision, numpy, pillow, scikit-learn, matplotlib
  - Se recomienda usar un virtualenv o conda.
  - Si existe, instalar desde requirements.txt: `pip install -r requirements.txt`

Estructura típica
- src/                 → código fuente (train.py, evaluate.py, predict.py, utils.py)
- data/                → dataset organizado por carpetas o CSV
  - data/train/<clase>/*, data/val/<clase>/*, data/test/<clase>/*
- models/              → checkpoints guardados (.pth)
- outputs/             → métricas, figuras, logs
- notebooks/           → experimentos exploratorios

Preparar datos
- Formato esperado (opción A - carpetas):
  - data/train/classA/*.jpg
  - data/train/classB/*.jpg
  - data/val/...
  - data/test/...
- Opción B: CSV con columnas (image_path,label) y función de lectura en src/utils.py
- Normalización: seguir transformaciones usadas por los modelos (resize, crop, normalize)

Instalación (rápida)
1. Crear entorno:
   - python -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)
2. Instalar dependencias:
   - pip install -r requirements.txt
   - Si no hay requirements.txt: pip install torch torchvision numpy pillow scikit-learn matplotlib

Uso (comandos ejemplos)
- Entrenamiento:
  - python src/train.py --data data/ --save-dir models/ --epochs 30 --batch-size 32 --lr 1e-3
  - O con config YAML: python src/train.py --config configs/train.yaml
- Evaluación:
  - python src/evaluate.py --model models/best.pth --data data/test/
- Inferencia:
  - python src/predict.py --model models/best.pth --image path/to/image.jpg --output outputs/pred.txt
- Ajusta flags dependiendo de la implementación (path, batch size, dispositivo, semilla).

Parámetros recomendados
- batch-size: 16–64 (ajustar según memoria GPU)
- learning rate: 1e-3 (optim. Adam) o 1e-4–1e-2 según modelo
- epochs: 20–100 (depende de dataset)
- guardar checkpoints por época si mejora la métrica de validación

Problemas comunes y soluciones
- Error: módulo no encontrado (ModuleNotFoundError)
  - Solución: activar entorno virtual e instalar dependencias; verificar PYTHONPATH si el proyecto usa imports relativos.
- CUDA no disponible o versión incompatible
  - Solución: comprobar `torch.cuda.is_available()`; instalar la versión de PyTorch compatible con la versión de CUDA instalada o usar la versión CPU de torch.
- Out of memory (OOM) en GPU
  - Solución: reducir batch size, usar mixed precision (AMP), reducir resolución de entrada, cerrar otros procesos que usan GPU.
- Ruta de datos incorrecta / FileNotFoundError
  - Solución: usar rutas absolutas o verificar la estructura de carpetas; OneDrive puede cambiar permisos o rutas, desactivar sincronización temporalmente.
- Checkpoint no encontrado o formato incompatible
  - Solución: comprobar que se guarda `state_dict()` o el checkpoint completo; adaptar el script de carga al formato guardado.
- Resultados inesperados (accuracy muy baja)
  - Solución: verificar preprocesado (normalización), etiquetas, balance de clases, overfitting/underfitting; probar un subset y sobreajustar para validar pipeline.
- Diferencias en versiones de librerías
  - Solución: fijar versiones en requirements.txt y usar un entorno limpio.
- Permisos (Windows/OneDrive)
  - Solución: ejecutar el terminal con permisos adecuados y evitar rutas con caracteres especiales; mover proyecto fuera de OneDrive si hay conflictos.

Consejos rápidos
- Empieza con un subset pequeño para validar pipeline.
- Guarda logs y métricas (tensorboard o CSV) para comparar experimentos.
- Fija seed para reproducibilidad (numpy, torch.manual_seed, torch.backends.cudnn.deterministic = True / benchmark = False).
- Documenta configuraciones usadas en cada experimento.

¿Dónde mirar primero si algo falla?
1. Mensaje de error completo.
2. Paths y permisos.
3. Disponibilidad de GPU y versión de PyTorch.
4. Estructura y etiquetas del dataset.

Contribuciones
- Abrir issues con reproducible example.
- Pull requests con tests o notebooks explicativos.

Licencia
- MIT (o la que prefieras). Ajustar según proyecto.

Contacto
- Añadir correo o referencia de repo si se requiere.
