#!/usr/bin/env python3
"""
scripts/init_firebase.py

Inicializa Firebase Admin SDK usando una cuenta de servicio (JSON).
Opciones:
  --cred PATH                Ruta al serviceAccountKey JSON (obligatorio si no está en default)
  --project PROJECT_ID       (opcional) Project ID, para verificar coincidencia
  --storage-bucket BUCKET    (opcional) nombre del storage bucket (ej: my-project.appspot.com)
  --seed-demo-user EMAIL:PASS  (opcional) crear usuario demo en Firebase Auth
  --create-collections       (flag) crea documentos iniciales en Firestore (config/ensemble, metrics)
  --upload-test-file PATH    (opcional) sube archivo al bucket y lo hace público (prueba)
  --no-public-upload         (flag) no llamar blob.make_public() (si no quieres la URL pública)
  --verbose / -v             Más logging

Ejemplos:
  python scripts/init_firebase.py --cred deployment/flask_app/firebase_config.json --create-collections
  python scripts/init_firebase.py --cred deployment/flask_app/firebase_config.json --seed-demo-user demo@example.com:password123 --upload-test-file tests/sample.png

IMPORTANTE:
 - Este script usa las librerías 'firebase_admin' (Admin SDK) y 'google-cloud-storage' (instalada normalmente con firebase_admin).
 - Proporciona la ruta correcta al JSON de la cuenta de servicio descargada desde Firebase Console.
 - No modifica reglas de seguridad del bucket ni configura IAM; sólo sube un archivo de prueba si se solicita.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth, storage
except Exception as e:
    print("ERROR: firebase_admin no está instalado. Instala firebase-admin (`pip install firebase-admin`).")
    raise

def setup_logging(verbose=False):
    lvl = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=lvl, format="[%(levelname)s] %(message)s")

def init_app(cred_path: Path, storage_bucket: str = None):
    """
    Inicializa firebase_admin y devuelve la app inicializada.
    """
    try:
        # Si ya está inicializado, retornamos la app existente
        existing = None
        try:
            existing = firebase_admin.get_app()
            logging.info("firebase_admin ya inicializado como app: %s", existing.name)
            return existing
        except ValueError:
            pass

        cred = credentials.Certificate(str(cred_path))
        options = {}
        if storage_bucket:
            options['storageBucket'] = storage_bucket

        app = firebase_admin.initialize_app(cred, options=options)
        logging.info("firebase_admin inicializado correctamente.")
        return app
    except Exception as e:
        logging.exception("Fallo al inicializar firebase_admin: %s", e)
        raise

def create_collections_seed(db):
    """
    Crea documentos iniciales en Firestore: config/ensemble, metrics/sample
    """
    try:
        # config/ensemble doc
        config_ref = db.collection("config").document("ensemble")
        config_ref.set({
            "enabled": False,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "notes": "Estado inicial del ensemble. Cambiar via backend si necesario."
        }, merge=True)
        logging.info("Documento 'config/ensemble' creado/actualizado.")

        # metrics sample doc (empty)
        metrics_ref = db.collection("metrics").document("latest")
        metrics_ref.set({
            "accuracy": None,
            "loss": None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "source": "init_script"
        }, merge=True)
        logging.info("Documento 'metrics/latest' creado/actualizado.")

        # optionally create an index-like placeholder collection to avoid empty collection confusion
        history_ref = db.collection("history").document("_placeholder")
        history_ref.set({
            "note": "placeholder doc to ensure collection exists",
            "created_at": datetime.utcnow().isoformat() + "Z"
        })
        logging.info("Documento placeholder en 'history' creado (puedes eliminarlo luego).")
        return True
    except Exception as e:
        logging.exception("Error al crear documentos iniciales en Firestore: %s", e)
        return False

def create_demo_user(email, password):
    """
    Crea un usuario demo en Firebase Auth. Si ya existe, lo retorna.
    Devuelve objeto user_record o None en error.
    """
    try:
        # comprobar existencia
        try:
            existing = auth.get_user_by_email(email)
            logging.info("Usuario ya existe: %s (uid=%s)", email, existing.uid)
            return existing
        except auth.UserNotFoundError:
            pass

        user = auth.create_user(email=email, password=password, email_verified=False, disabled=False)
        logging.info("Usuario creado: %s (uid=%s)", email, user.uid)
        # opcional: añadir custom claims / metadata
        # auth.set_custom_user_claims(user.uid, {'role': 'demo'})
        return user
    except Exception as e:
        logging.exception("Error al crear usuario demo: %s", e)
        return None

def upload_test_file(bucket, local_path: Path, dest_path: str = None, make_public: bool = True):
    """
    Sube un archivo al bucket y devuelve la public_url o gs:// path.
    """
    try:
        if not local_path.exists():
            logging.error("Archivo local no existe: %s", local_path)
            return None
        blob_name = dest_path or f"init_test_uploads/{local_path.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        logging.info("Archivo subido a bucket: %s", blob_name)
        public_url = None
        if make_public:
            try:
                blob.make_public()
                public_url = blob.public_url
                logging.info("Archivo hecho público: %s", public_url)
            except Exception as e:
                # puede fallar si las credenciales no tienen permisos para cambiar ACLs
                logging.warning("No se pudo hacer público el archivo (se requiere permiso): %s", e)
                public_url = f"gs://{bucket.name}/{blob_name}"
        else:
            public_url = f"gs://{bucket.name}/{blob_name}"
        return public_url
    except Exception as e:
        logging.exception("Error al subir archivo al bucket: %s", e)
        return None

def main():
    parser = argparse.ArgumentParser(description="Inicializar Firebase (Admin SDK) y crear artefactos iniciales")
    parser.add_argument("--cred", "-c", required=True, help="Ruta al JSON de la cuenta de servicio (serviceAccountKey)")
    parser.add_argument("--project", "-p", default=None, help="Project ID esperado (comprobar coincidencia)")
    parser.add_argument("--storage-bucket", "-b", default=None, help="Storage bucket name (opcional)")
    parser.add_argument("--seed-demo-user", "-s", default=None, help="Crear demo user: email:password")
    parser.add_argument("--create-collections", action="store_true", help="Crear documentos iniciales en Firestore")
    parser.add_argument("--upload-test-file", "-u", default=None, help="Subir archivo de prueba al bucket")
    parser.add_argument("--no-public-upload", action="store_true", help="No llamar blob.make_public() al subir")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    cred_path = Path(args.cred)
    if not cred_path.exists():
        logging.error("El archivo de credenciales no existe: %s", cred_path)
        sys.exit(2)

    # initialize firebase app
    try:
        app = init_app(cred_path, storage_bucket=args.storage_bucket)
    except Exception:
        logging.error("No se pudo inicializar Firebase Admin SDK.")
        sys.exit(3)

    # check project id in credentials (best-effort)
    try:
        with open(cred_path, "r", encoding="utf-8") as f:
            cred_json = json.load(f)
        cred_project_id = cred_json.get("project_id")
        if args.project and cred_project_id and args.project != cred_project_id:
            logging.warning("Project ID pasado (%s) no coincide con cred JSON (%s).", args.project, cred_project_id)
        else:
            logging.info("Project ID (cred): %s", cred_project_id)
    except Exception:
        logging.debug("No se pudo leer project_id del JSON de credenciales.")

    # Firestore client
    db = None
    try:
        db = firestore.client()
        logging.info("Conectado a Firestore.")
    except Exception as e:
        logging.warning("No se pudo conectar a Firestore: %s", e)

    # Storage bucket object
    bucket = None
    if args.storage_bucket:
        try:
            bucket = storage.bucket()
            logging.info("Storage bucket listo: %s", bucket.name)
        except Exception as e:
            logging.warning("No se pudo obtener bucket: %s", e)
    else:
        # try to get default bucket if app options set
        try:
            bucket = storage.bucket()
            logging.info("Storage bucket (auto detectado): %s", bucket.name)
        except Exception:
            bucket = None

    # create collections/docs if requested
    if args.create_collections:
        if db is None:
            logging.error("Firestore no disponible, no se pueden crear colecciones.")
        else:
            ok = create_collections_seed(db)
            if not ok:
                logging.error("Error creando documentos iniciales.")

    # create demo user if requested
    if args.seed_demo_user:
        try:
            parts = args.seed_demo_user.split(":", 1)
            if len(parts) != 2:
                logging.error("--seed-demo-user debe estar en formato email:password")
            else:
                email = parts[0]
                password = parts[1]
                user = create_demo_user(email, password)
                if user:
                    logging.info("Usuario demo: uid=%s email=%s", user.uid, user.email)
        except Exception as e:
            logging.exception("Error creando usuario demo: %s", e)

    # upload test file if requested
    if args.upload_test_file:
        if bucket is None:
            logging.error("No hay bucket configurado/obtenido. Proporciona --storage-bucket o configura storageBucket en la app.")
        else:
            local_path = Path(args.upload_test_file)
            if not local_path.exists():
                logging.error("Archivo de prueba no encontrado: %s", local_path)
            else:
                url = upload_test_file(bucket, local_path, make_public=(not args.no_public_upload))
                if url:
                    logging.info("Archivo de prueba subido correctamente: %s", url)
                else:
                    logging.error("La subida del archivo de prueba falló.")

    logging.info("init_firebase.py completado.")

if __name__ == "__main__":
    main()
