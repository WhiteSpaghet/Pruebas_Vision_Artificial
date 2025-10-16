#!/usr/bin/env python3
# deployment/flask_app/app.py
# VISION_ARTIFICIAL - Flask backend con carga robusta de credenciales Firebase

import os
import io
import json
import pathlib
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify, send_file,
    session, redirect, url_for, flash
)
from werkzeug.security import generate_password_hash, check_password_hash

# ML libs
import numpy as np
from PIL import Image
import tensorflow as tf

# Firebase admin (opcional)
FIREBASE_ENABLED = True
firebase_app = None
firebase_auth = None
firebase_db = None
firebase_storage_bucket = None

try:
    import firebase_admin
    from firebase_admin import credentials, auth, firestore, storage
except Exception:
    FIREBASE_ENABLED = False
    firebase_admin = None
    credentials = None
    auth = None
    firestore = None
    storage = None
    print("[firebase] firebase-admin no disponible (import failed). Continuando sin Firebase.")


# ---------------------------
# Paths y creación de carpetas
# ---------------------------
HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]  # si app.py está en deployment/flask_app -> project root es 2 niveles arriba
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_best.keras"
UPLOADS_DIR = HERE / "static" / "uploads"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

for d in (UPLOADS_DIR, EXPERIMENTS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")  # cambia en producción

# ---------------------------
# Búsqueda robusta del service account
# ---------------------------
def _find_service_account():
    """
    Busca el fichero de credenciales en varias ubicaciones:
      1) variable de entorno FIREBASE_CREDENTIALS_PATH
      2) junto a app.py -> deployment/flask_app/firebase_service_account.json
      3) desde cwd: deployment/flask_app/firebase_service_account.json
      4) raíz del repo: firebase_service_account.json
    """
    candidates = []
    env_path = os.getenv("FIREBASE_CREDENTIALS_PATH") or os.getenv("FIREBASE_CRED_PATH")
    if env_path:
        candidates.append(pathlib.Path(env_path))

    # junto a app.py
    candidates.append(HERE / "firebase_service_account.json")
    # ruta habitual desde cwd
    candidates.append(pathlib.Path.cwd() / "deployment" / "flask_app" / "firebase_service_account.json")
    candidates.append(pathlib.Path.cwd() / "firebase_service_account.json")

    for c in candidates:
        if c and c.exists():
            return c.resolve()
    return None

# ---------------------------
# Inicialización de Firebase (best-effort)
# ---------------------------
if FIREBASE_ENABLED:
    try:
        sa_path = _find_service_account()
        if sa_path is None:
            FIREBASE_ENABLED = False
            print("[firebase] service account NO encontrado. Buscados en variables y rutas habituales.")
            print("           Si quieres forzar la ruta define la variable FIREBASE_CREDENTIALS_PATH.")
        else:
            print(f"[firebase] usando credenciales en: {sa_path}")
            cred = credentials.Certificate(str(sa_path))
            firebase_app = firebase_admin.initialize_app(cred, {
                "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET")  # opcional: "mi-bucket.appspot.com"
            } if os.getenv("FIREBASE_STORAGE_BUCKET") else None)
            firebase_auth = auth
            try:
                firebase_db = firestore.client()
            except Exception as e:
                print("[firebase] Firestore no inicializado:", e)
                firebase_db = None
            try:
                firebase_storage_bucket = storage.bucket()
            except Exception as e:
                print("[firebase] Storage bucket no inicializado:", e)
                firebase_storage_bucket = None
            print("[firebase] inicializado OK")
    except Exception as e:
        FIREBASE_ENABLED = False
        print("[firebase] fallo en init:", e)

# ---------------------------
# Modelo Keras (opcional)
# ---------------------------
MODEL = None
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_model_safe(path):
    global MODEL
    try:
        print("Cargando modelo desde:", path)
        MODEL = tf.keras.models.load_model(str(path))
        print("✅ Modelo cargado.")
    except Exception as e:
        print("❌ Error cargando modelo:", e)
        MODEL = None

if MODEL_PATH.exists():
    load_model_safe(MODEL_PATH)
else:
    print("⚠️ Modelo no encontrado en:", MODEL_PATH)

# ---------------------------
# Auxiliares
# ---------------------------
def preprocess_image_bytes(image_bytes, target_size=(32,32)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def save_local_file(file_storage, filename=None):
    if filename is None:
        filename = datetime.utcnow().strftime("%Y%m%d%H%M%S%f_") + file_storage.filename
    safe_name = filename.replace(" ", "_")
    dest = UPLOADS_DIR / safe_name
    # ensure parent exists
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_storage.save(str(dest))
    return f"/static/uploads/{safe_name}", str(dest)

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Inicia sesión para acceder a esta sección", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

# ---------------------------
# Rutas web
# ---------------------------
@app.route("/")
def index():
    theme = session.get("theme", "theme_light_green.css")
    return render_template("index.html", theme=theme)

@app.route("/dashboard")
@login_required
def dashboard():
    theme = session.get("theme", "theme_light_green.css")
    history = session.get("history", [])
    return render_template("dashboard.html", theme=theme, history=history)

@app.route("/profile")
@login_required
def profile():
    theme = session.get("theme", "theme_light_green.css")
    user = session.get("user", {})
    return render_template("profile.html", theme=theme, user=user)

@app.route("/change_theme", methods=["POST"])
def change_theme():
    theme = request.form.get("theme", "theme_light_green.css")
    session["theme"] = theme
    return jsonify({"theme": theme})

# ---------------------------
# Usuarios locales (simple)
# ---------------------------
USERS_LOCAL_FILE = EXPERIMENTS_DIR / "users_local.json"

def load_local_users():
    if USERS_LOCAL_FILE.exists():
        with open(USERS_LOCAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_local_users(users):
    with open(USERS_LOCAL_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def create_local_user(email, name, password):
    users = load_local_users()
    if email in users:
        return False, "El usuario ya existe."
    users[email] = {
        "name": name,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    save_local_users(users)
    return True, None

def authenticate_local_user(email, password):
    users = load_local_users()
    user = users.get(email)
    if not user:
        return False, None
    if check_password_hash(user["password_hash"], password):
        return True, user
    return False, None

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    email = request.form.get("email")
    password = request.form.get("password")
    ok, user = authenticate_local_user(email, password)
    if not ok:
        flash("Credenciales incorrectas", "error")
        return redirect(url_for("login"))
    session["user"] = {"email": email, "name": user["name"]}
    flash("Sesión iniciada", "success")
    return redirect(url_for("dashboard"))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    confirm = request.form.get("password_confirm")
    if password != confirm:
        flash("Las contraseñas no coinciden", "error")
        return redirect(url_for("register"))
    ok, msg = create_local_user(email, name, password)
    if not ok:
        flash(msg, "error")
        return redirect(url_for("register"))
    session["user"] = {"email": email, "name": name}
    flash("Cuenta creada correctamente", "success")
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Sesión cerrada", "info")
    return redirect(url_for("index"))

# ---------------------------
# API de predicción
# ---------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if MODEL is None:
        return jsonify({"error": "Modelo no cargado"}), 500
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No se subieron archivos"}), 400
    results = []
    for f in files:
        try:
            arr = preprocess_image_bytes(f.read())
            preds = MODEL.predict(arr)
            idx = int(np.argmax(preds[0]))
            label = CLASS_NAMES[idx]
            conf = float(np.max(preds[0]))
        except Exception as e:
            label = None
            conf = 0.0
        rel_url, local_path = save_local_file(f)
        results.append({
            "prediction": label,
            "confidence": conf,
            "image_url": rel_url
        })
        # guardar historial en session si usuario logueado
        session.setdefault("history", []).append({
            "file": rel_url,
            "prediction": label,
            "confidence": round(conf * 100, 2),
            "timestamp": datetime.utcnow().isoformat()
        })
    return jsonify(results), 200

# ---------------------------
# API historial
# ---------------------------
@app.route("/api/history")
def api_history():
    limit = int(request.args.get("limit", 10))
    history = session.get("history", [])
    # devolver los últimos 'limit' en orden inverso (más recientes primero)
    return jsonify(list(reversed(history))[:limit])

# ---------------------------
# API: listar notebooks en Firebase Storage (si Firebase activo)
# ---------------------------
@app.route("/api/notebooks")
def api_notebooks():
    if not FIREBASE_ENABLED or firebase_storage_bucket is None:
        return jsonify({"error": "Firebase Storage no disponible"}), 503
    try:
        blobs = firebase_storage_bucket.list_blobs(prefix="notebooks/")
        out = []
        for b in blobs:
            if not b.name.endswith(".ipynb"):
                continue
            # generar signed url temporal (vencimiento 1 hora)
            try:
                signed = b.generate_signed_url(expiration=timedelta(hours=1), method="GET")
            except Exception:
                # fallback: usar gs:// path si no se puede generar signed url
                signed = f"gs://{firebase_storage_bucket.name}/{b.name}"
            out.append({"name": b.name.split("/")[-1], "path": b.name, "url": signed})
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------
# API: descargar notebook (proxy) - sirve si quieres forzar autorización
# ---------------------------
@app.route("/api/notebook/<path:name>")
@login_required
def api_download_notebook(name):
    if not FIREBASE_ENABLED or firebase_storage_bucket is None:
        return "Storage no disponible", 503
    blob = firebase_storage_bucket.blob(f"notebooks/{name}")
    if not blob.exists():
        return "No encontrado", 404
    tmp_path = pathlib.Path.cwd() / f"tmp_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}_{name}"
    try:
        blob.download_to_filename(str(tmp_path))
        return send_file(str(tmp_path), as_attachment=True, download_name=name)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

# ---------------------------
# Error handlers
# ---------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    # evitar logs de TF demasiado verbosos
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if MODEL is None and MODEL_PATH.exists():
        load_model_safe(MODEL_PATH)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
