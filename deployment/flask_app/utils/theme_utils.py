from flask import request, make_response

# Temas disponibles en la app
AVAILABLE_THEMES = {
    "light_green": "theme_light_green.css",
    "light_dark": "theme_light_dark.css",
    "dark_gray": "theme_dark_gray.css"
}

DEFAULT_THEME = "light_green"


def get_user_theme():
    """
    Obtiene el tema actual del usuario desde cookies,
    o el tema por defecto si no hay uno guardado
    """
    theme = request.cookies.get("theme")
    if theme in AVAILABLE_THEMES:
        return theme
    return DEFAULT_THEME


def set_user_theme(theme_name, response=None):
    """
    Guarda el tema: crea una cookie para recordar el tema seleccionado.
    """
    if theme_name not in AVAILABLE_THEMES:
        theme_name = DEFAULT_THEME

    if response is None:
        response = make_response()

    response.set_cookie("theme", theme_name, max_age=60 * 60 * 24 * 30)  # Dura 30 días
    return response


def get_theme_stylesheet(theme_name):
    """
    Devuelve el archivo CSS asociado al nombre del tema
    """
    return AVAILABLE_THEMES.get(theme_name, AVAILABLE_THEMES[DEFAULT_THEME])
