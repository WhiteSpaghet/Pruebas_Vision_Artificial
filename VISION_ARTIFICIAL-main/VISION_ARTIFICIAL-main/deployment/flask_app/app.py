"""
deployment/flask_app/app.py
VISION_ARTIFICIAL - Flask backend (predicción, API, Firebase optional)

Notas:
 - Maneja Firebase admin de forma robusta (si el json no es válido se desactiva).
 - Carga modelo Keras de forma tolerante: si no está presente la app sigue corriendo.
 - Endpoints útiles: /api/predict (anónimo posible), /api/history, /api/whoami, /api/metrics
 - Ruta /reset_password añadida (placeholder).
"""

import os
import io
import json
import pathlib
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, jsonify, redirect, url_for, flash, session
)
from werkzeug.security import generate_password_hash, check_password_hash

# optional ML libs (fall back gracefully if TF missing)
TF_AVAILABLE = True
try:
    import numpy as np
    from PIL import Image
    import tensorflow as tf
except Exception as e:
    TF_AVAILABLE = False
    # keep running the app even if TF is not available
    print("[warning] TensorFlow (or other ML deps) not available:", e)

# Optional Firebase admin (robust import)
FIREBASE_ENABLED = True
FIREBASE_CRED_PATH = "firebase_config.json"  # server side service account (NOT the client web config)
firebase_app = None
firebase_auth = None
firebase_db = None
firebase_storage_bucket = None

try:
    import firebase_admin
    from firebase_admin import credentials, auth, firestore, storage
except Exception:
    FIREBASE_ENABLED = False
    print("[firebase] firebase_admin not installed or import failed; running in local mode")

# Project paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_best.keras"
UPLOADS_DIR = pathlib.Path(__file__).resolve().parent / "static" / "uploads"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

# ensure folders exist
for d in (UPLOADS_DIR, EXPERIMENTS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

# -----------------------
# Firebase initialization (robust)
# -----------------------
if FIREBASE_ENABLED:
    try:
        cred_path = pathlib.Path(__file__).resolve().parent / FIREBASE_CRED_PATH
        if cred_path.exists():
            # quick validation: JSON must include "type":"service_account"
            raw = json.loads(cred_path.read_text(encoding="utf-8"))
            if raw.get("type") != "service_account":
                raise ValueError("Invalid service account JSON: 'type' != 'service_account'")
            cred = credentials.Certificate(str(cred_path))
            firebase_app = firebase_admin.initialize_app(cred)
            firebase_auth = auth
            firebase_db = firestore.client()
            try:
                firebase_storage_bucket = storage.bucket()
            except Exception:
                firebase_storage_bucket = None
            print("[firebase] initialized")
        else:
            FIREBASE_ENABLED = False
            print("[firebase] firebase_config.json not found; running without Firebase admin")
    except Exception as e:
        FIREBASE_ENABLED = False
        print("[firebase] init failed:", e)

# -----------------------
# Context processor: inject current year for footer
# -----------------------
@app.context_processor
def inject_current_year():
    return {"current_year": datetime.utcnow().year}

# -----------------------
# Model loading (tolerant)
# -----------------------
MODEL = None
if TF_AVAILABLE:
    def try_load_model(path):
        try:
            print(f"[model] trying to load: {path}")
            m = tf.keras.models.load_model(str(path))
            print("[model] loaded from", path)
            return m
        except Exception as e:
            print(f"[model] cannot load {path}: {e}")
            return None

    # try common candidate paths
    candidates = [
        MODEL_PATH,
        PROJECT_ROOT / "models" / "cnn_best.h5",
        PROJECT_ROOT / "models" / "cnn_v1.keras",
        PROJECT_ROOT / "models" / "cnn_v1.h5"
    ]
    for p in candidates:
        if p and p.exists():
            MODEL = try_load_model(p)
            if MODEL is not None:
                break

    if MODEL is None:
        print("⚠️ No Keras model loaded. /api/predict seguirá funcionando pero devolverá error 500 si se intenta predecir.")
else:
    print("⚠️ TensorFlow no está disponible; API de predicción deshabilitada.")

# CIFAR-10 labels
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# -----------------------
# Helpers
# -----------------------
def is_api_request(req):
    """Detectar llamada AJAX/API para devolver 401 JSON en vez de redirect."""
    if req.path.startswith("/api"):
        return True
    # XHR header or fetch
    if req.headers.get("X-Requested-With") == "XMLHttpRequest":
        return True
    # Accept header prefers json
    if "application/json" in (req.headers.get("Accept", "")):
        return True
    return False

def preprocess_image_bytes(image_bytes, target_size=(32, 32)):
    """Bytes -> normalized numpy batch (1, h, w, c). Requires PIL + numpy."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow / PIL / numpy not available.")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def save_local_file(file_storage, filename=None):
    """Guarda la imagen subida en static/uploads y devuelve la ruta relativa y path local."""
    if filename is None:
        filename = datetime.utcnow().strftime("%Y%m%d%H%M%S%f_") + file_storage.filename
    safe_name = filename.replace(" ", "_")
    dest = UPLOADS_DIR / safe_name
    file_storage.save(str(dest))
    return f"/static/uploads/{safe_name}", str(dest)

# -----------------------
# Simple local user store (file-based) - useful si no usas Firebase Auth
# -----------------------
USERS_LOCAL_FILE = EXPERIMENTS_DIR / "users_local.json"

def load_local_users():
    if USERS_LOCAL_FILE.exists():
        with open(USERS_LOCAL_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
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

# -----------------------
# Decorator login_required (mejor comportamiento para APIs)
# -----------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            if is_api_request(request):
                return jsonify({"error": "authentication required"}), 401
            flash("Inicia sesión para acceder a esta sección", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

# -----------------------
# Web routes
# -----------------------
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

# -----------------------
# Authentication routes (local fallback)
# -----------------------
@app.route("/login", methods=["GET", "POST"])
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
    flash("Sesión iniciada correctamente", "success")
    return redirect(url_for("dashboard"))

@app.route("/register", methods=["GET", "POST"])
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
    flash("Sesión cerrada correctamente", "info")
    return redirect(url_for("index"))

# -----------------------
# Reset password (placeholder)
# -----------------------
@app.route("/reset_password", methods=["GET", "POST"])
def reset_password():
    if request.method == "POST":
        email = request.form.get("email")
        # aquí podrías conectar con Firebase Auth / tu sistema de envío de emails
        # por ahora simulamos: guardamos un flash y redirigimos al login
        flash("Si ese correo está registrado, recibirás instrucciones (simulado).", "info")
        return redirect(url_for("login"))
    return render_template("reset_password.html")

# -----------------------
# API: prediction (public allowed; saves history only si user está logueado)
# -----------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not TF_AVAILABLE or MODEL is None:
        return jsonify({"error": "model not available"}), 500

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "no files uploaded"}), 400

    results = []
    for f in files:
        try:
            arr = preprocess_image_bytes(f.read())
            preds = MODEL.predict(arr)
            idx = int(np.argmax(preds[0]))
            label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
            conf = float(np.max(preds[0]))
        except Exception as e:
            return jsonify({"error": f"prediction failed: {e}"}), 500

        rel_url, local_path = save_local_file(f)

        results.append({
            "prediction": label,
            "confidence": conf,
            "image_url": rel_url
        })

        # save to session history if logged in (simple local history)
        session.setdefault("history", []).append({
            "file": rel_url,
            "prediction": label,
            "confidence": round(conf * 100, 2),
            "timestamp": datetime.utcnow().isoformat()
        })

    return jsonify(results), 200

# -----------------------
# API: whoami, metrics, history
# -----------------------
@app.route("/api/whoami")
def api_whoami():
    return jsonify({"user": session.get("user")})

@app.route("/api/metrics")
def api_metrics():
    # Try to read a summary from experiments/history_cnn.json
    try:
        histf = EXPERIMENTS_DIR / "history_cnn.json"
        if histf.exists():
            with open(histf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # expected structure may vary; attempt to extract last epoch metrics
            if isinstance(data, dict) and "history" in data:
                last = data["history"][-1] if data["history"] else {}
                return jsonify({"accuracy": last.get("accuracy"), "loss": last.get("loss")})
            # fallback: try keys 'final'
            if isinstance(data, dict) and "final" in data:
                return jsonify(data["final"])
    except Exception:
        pass
    return jsonify({"accuracy": None, "loss": None})

@app.route("/api/history")
def api_history():
    try:
        limit = int(request.args.get("limit", 50))
    except Exception:
        limit = 50
    history = session.get("history", [])
    return jsonify(list(reversed(history[-limit:])))

# -----------------------
# Error handlers
# -----------------------
@app.errorhandler(404)
def not_found(e):
    return render_template("index.html"), 404

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    # reduce TF log spam if available
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # If model exists but not loaded (rare), try load again
    if TF_AVAILABLE and MODEL is None:
        try:
            if MODEL_PATH.exists():
                MODEL = tf.keras.models.load_model(str(MODEL_PATH))
                print("[model] loaded on startup retry")
        except Exception as e:
            print("[model] startup retry failed:", e)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
