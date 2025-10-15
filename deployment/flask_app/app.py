"""
deployment/flask_app/app.py
VISION_ARTIFICIAL - Flask backend (predicción + API + Firebase optional)

Pega este archivo en deployment/flask_app/app.py
Requisitos:
  - tensorflow (si vas a predecir localmente)
  - pillow, numpy
  - firebase-admin (opcional, si quieres usar Firestore / token verify)
"""

import os
import io
import json
import base64
import pathlib
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, storage

# Inicializar Firebase Admin
cred = credentials.Certificate("firebase_service_account.json")  # tu clave descargada
firebase_admin.initialize_app(cred, {
    "storageBucket": "TU_BUCKET.appspot.com"
})

bucket = storage.bucket()  # referencia al bucket


from flask import (
    Flask, render_template, request, jsonify, session, redirect, url_for, flash
)
from werkzeug.security import generate_password_hash, check_password_hash

# ML libs
try:
    import numpy as np
    from PIL import Image
    import tensorflow as tf
except Exception:
    # If TF not installed, app will still run pages but /api/predict will return error.
    np = None
    Image = None
    tf = None

# Optional firebase_admin
FIREBASE_ENABLED = True
firebase_admin = None
firebase_auth = None
firebase_db = None
firebase_storage = None

try:
    import firebase_admin
    from firebase_admin import credentials, auth, firestore, storage
    firebase_admin  # silence linter
except Exception:
    FIREBASE_ENABLED = False

# ---------------------------
# Paths & basic setup
# ---------------------------
HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "cnn_best.keras"
UPLOADS_DIR = HERE / "static" / "uploads"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

for d in (UPLOADS_DIR, EXPERIMENTS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "replace-this-with-a-secure-key")

# Make current year available in templates (fixes now() issues)
app.jinja_env.globals['current_year'] = datetime.utcnow().year

# ---------------------------
# Firebase initialization helpers
# ---------------------------
def init_firebase_from_env() -> bool:
    """
    Intenta inicializar firebase-admin desde:
      1) FIREBASE_CREDENTIALS_PATH -> ruta a JSON
      2) FIREBASE_SERVICE_ACCOUNT_JSON -> JSON crudo en env
      3) FIREBASE_SA_B64 -> base64 del JSON
    Devuelve True si inicializa correctamente.
    """
    global FIREBASE_ENABLED, firebase_admin, firebase_auth, firebase_db, firebase_storage
    if not FIREBASE_ENABLED:
        return False
    try:
        # prefer path
        sa_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
        if sa_path:
            p = pathlib.Path(sa_path)
            if p.exists():
                cred = credentials.Certificate(str(p))
                firebase_admin.initialize_app(cred)
                firebase_auth = auth
                firebase_db = firestore.client()
                try:
                    firebase_storage = storage.bucket()
                except Exception:
                    firebase_storage = None
                print("[firebase] initialized from file path")
                return True

        # raw json
        sa_raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        if sa_raw:
            cred = credentials.Certificate(json.loads(sa_raw))
            firebase_admin.initialize_app(cred)
            firebase_auth = auth
            firebase_db = firestore.client()
            try:
                firebase_storage = storage.bucket()
            except Exception:
                firebase_storage = None
            print("[firebase] initialized from FIREBASE_SERVICE_ACCOUNT_JSON")
            return True

        # base64 encoded
        sa_b64 = os.environ.get("FIREBASE_SA_B64")
        if sa_b64:
            decoded = base64.b64decode(sa_b64).decode("utf-8")
            cred = credentials.Certificate(json.loads(decoded))
            firebase_admin.initialize_app(cred)
            firebase_auth = auth
            firebase_db = firestore.client()
            try:
                firebase_storage = storage.bucket()
            except Exception:
                firebase_storage = None
            print("[firebase] initialized from FIREBASE_SA_B64")
            return True

        # attempt to find firebase_config file in static (service account not the same as client config)
        print("[firebase] no server credentials provided via env; firebase-admin disabled (client SDK still OK on frontend)")
        FIREBASE_ENABLED = False
        return False
    except Exception as e:
        print("[firebase] init failed:", e)
        FIREBASE_ENABLED = False
        return False

# run firebase init attempt
init_firebase_from_env()

# ---------------------------
# Model loading
# ---------------------------
MODEL = None
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def load_model_safe(path: pathlib.Path):
    global MODEL
    if tf is None:
        print("[model] tensorflow not installed; skipping model load.")
        MODEL = None
        return
    try:
        print(f"[model] loading model from {path}")
        MODEL = tf.keras.models.load_model(str(path))
        print("[model] model loaded.")
    except Exception as e:
        print("[model] load error:", e)
        MODEL = None


if MODEL_PATH.exists():
    load_model_safe(MODEL_PATH)
else:
    print("[model] model file not found at", MODEL_PATH)

# ---------------------------
# Utilities
# ---------------------------
def preprocess_image_bytes(image_bytes: bytes, target_size=(32, 32)):
    """Bytes -> normalized numpy array [1,H,W,3]."""
    if Image is None or np is None:
        raise RuntimeError("Pillow / numpy required for preprocessing")
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def save_local_file(file_storage, filename=None):
    """Save uploaded FileStorage to static/uploads and return relative url and absolute path."""
    fname = filename or (datetime.utcnow().strftime("%Y%m%d%H%M%S%f_") + file_storage.filename)
    safe = fname.replace(" ", "_")
    dest = UPLOADS_DIR / safe
    file_storage.save(str(dest))
    return f"/static/uploads/{safe}", str(dest)


# ---------------------------
# User store (local fallback)
# ---------------------------
USERS_LOCAL_FILE = EXPERIMENTS_DIR / "users_local.json"


def load_local_users() -> Dict[str, Any]:
    if USERS_LOCAL_FILE.exists():
        with open(USERS_LOCAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_local_users(users: Dict[str, Any]) -> None:
    with open(USERS_LOCAL_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)


def create_local_user(email: str, name: str, password: str):
    users = load_local_users()
    if email in users:
        return False, "Usuario ya existe"
    users[email] = {
        "name": name,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    save_local_users(users)
    return True, None


def authenticate_local_user(email: str, password: str):
    users = load_local_users()
    u = users.get(email)
    if not u:
        return False, None
    if check_password_hash(u["password_hash"], password):
        return True, u
    return False, None


# ---------------------------
# Auth helpers (session + token)
# ---------------------------
def get_token_from_request(req) -> Optional[str]:
    auth_header = req.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1].strip()
    # JSON body token
    try:
        if req.is_json:
            j = req.get_json(silent=True) or {}
            if "idToken" in j:
                return j.get("idToken")
    except Exception:
        pass
    return req.form.get("idToken") or req.args.get("idToken") or req.headers.get("X-ID-TOKEN")


def verify_firebase_token(id_token: str) -> Optional[Dict[str, Any]]:
    """Verifica idToken con firebase_admin.auth y devuelve dict decodificado o None."""
    if not FIREBASE_ENABLED or firebase_auth is None:
        return None
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        app.logger.debug("verify_firebase_token failed: %s", e)
        return None


def get_current_user_from_request(req) -> Optional[Dict[str, Any]]:
    # session first
    if "user" in session:
        return session.get("user")
    # token
    token = get_token_from_request(req)
    if token:
        decoded = verify_firebase_token(token)
        if decoded:
            return {
                "uid": decoded.get("uid"),
                "email": decoded.get("email"),
                "name": decoded.get("name") or decoded.get("displayName")
            }
    return None


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Inicia sesión para acceder a esta sección", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


# ---------------------------
# Routes - pages
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
# Auth - local register / login / logout
# ---------------------------
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


# ---------------------------
# API: Firebase login (client -> server)
# ---------------------------
@app.route("/api/login_firebase", methods=["POST"])
def api_login_firebase():
    """
    Recibe JSON { idToken } desde el cliente. Verifica token con Firebase Admin,
    y crea session['user'] con uid/email/name para uso en rutas que usan sesiones Flask.
    """
    data = {}
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict() or {}
    id_token = data.get("idToken")
    if not id_token:
        return jsonify({"error": "idToken required"}), 400
    decoded = verify_firebase_token(id_token)
    if decoded is None:
        return jsonify({"error": "invalid token"}), 401
    session["user"] = {
        "uid": decoded.get("uid"),
        "email": decoded.get("email"),
        "name": decoded.get("name") or decoded.get("displayName")
    }
    return jsonify({"ok": True, "user": session["user"]}), 200


# ---------------------------
# API: predict and history
# ---------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Acepta multipart/form-data con campo 'files' (uno o varios).
    Si el usuario está autenticado por sesión o token, se guarda en session history.
    Retorna lista de predicciones.
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    user = get_current_user_from_request(request)
    results = []
    for f in files:
        try:
            arr = preprocess_image_bytes(f.read())
            preds = MODEL.predict(arr)
            idx = int(np.argmax(preds[0]))
            label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
            conf = float(np.max(preds[0]))
        except Exception as e:
            # prediction error
            app.logger.exception("prediction error")
            return jsonify({"error": "prediction failed", "detail": str(e)}), 500

        rel_url, local_path = save_local_file(f)
        out = {"prediction": label, "confidence": conf, "image_url": rel_url}
        results.append(out)

        # Save to local session history
        entry = {
            "file": rel_url,
            "prediction": label,
            "confidence": round(conf * 100, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        session.setdefault("history", []).append(entry)

        # Optionally, save to Firestore if configured and user is available
        try:
            if FIREBASE_ENABLED and firebase_db and user:
                doc = {
                    "user": user.get("email") or user.get("uid"),
                    "prediction": label,
                    "confidence": conf,
                    "image_url": rel_url,
                    "timestamp": datetime.utcnow()
                }
                firebase_db.collection("predictions").add(doc)
        except Exception as e:
            app.logger.debug("failed saving to firestore: %s", e)

    return jsonify(results), 200


@app.route("/api/history")
def api_history():
    """
    Devuelve historial del usuario desde session (limit opcional).
    Si Firestore está habilitado y quieres, se puede implementar lectura desde ahí.
    """
    try:
        limit = int(request.args.get("limit", 20))
    except Exception:
        limit = 20
    history = session.get("history", [])
    # devolver los últimos `limit` (más recientes al final)
    return jsonify(history[-limit:][::-1])


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
    # reduce TF logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # attempt to load model if not loaded and model file exists
    if MODEL is None and MODEL_PATH.exists() and tf is not None:
        load_model_safe(MODEL_PATH)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
