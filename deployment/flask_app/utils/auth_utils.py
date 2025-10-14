import firebase_admin
from firebase_admin import auth, credentials
from flask import session
import pyrebase

# Inicializar Pyrebase para autenticación con email y contraseña
firebase_config = {
    "apiKey": "YOUR_API_KEY",
    "authDomain": "YOUR_AUTH_DOMAIN",
    "projectId": "YOUR_PROJECT_ID",
    "storageBucket": "YOUR_STORAGE_BUCKET",
    "messagingSenderId": "YOUR_SENDER_ID",
    "appId": "YOUR_APP_ID",
    "databaseURL": ""
}

firebase = pyrebase.initialize_app(firebase_config)
pyre_auth = firebase.auth()

# Inicializar firebase-admin para endpoints seguros
try:
    firebase_admin.get_app()
except ValueError:
    cred = credentials.Certificate("firebase_config.json")
    firebase_admin.initialize_app(cred)


# ---------------- AUTH FUNCTIONS ---------------- #

def register_user(email, password):
    """
    Registra un usuario en Firebase Authentication
    """
    try:
        user = pyre_auth.create_user_with_email_and_password(email, password)
        return user
    except Exception as e:
        raise Exception(f"Error al registrar usuario: {e}")


def login_user(email, password):
    """
    Inicia sesión y guarda la sesión en cookies de Flask
    """
    try:
        user = pyre_auth.sign_in_with_email_and_password(email, password)
        session['user'] = user
        return user
    except Exception as e:
        raise Exception("Email o contraseña incorrectos")


def logout_user():
    """
    Cierra sesión eliminando la sesión de Flask
    """
    session.pop('user', None)


def get_current_user():
    """
    Devuelve el usuario autenticado actual
    """
    return session.get('user')
