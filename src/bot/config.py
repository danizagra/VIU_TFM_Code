"""
Bot configuration and constants.
"""

# Maximum message length for Telegram
MAX_MESSAGE_LENGTH = 4096

# Default pagination
DEFAULT_NEWS_LIMIT = 5
MAX_NEWS_LIMIT = 20

# Timeout for API calls (seconds)
API_TIMEOUT = 30

# Search thresholds for vector similarity
# These control the trade-off between precision and recall in news search
VECTOR_SEARCH_THRESHOLD = 0.40  # Initial wide net for vector search
MIN_RELEVANCE_THRESHOLD = 0.45  # Minimum after keyword filtering
HIGH_SIMILARITY_THRESHOLD = 0.80  # Very high confidence semantic match (no keyword needed)
# Note: 0.80 is strict - only truly relevant articles pass without keywords

# Minimum articles to consider "sufficient coverage"
# If fewer than this, also search external sources for fresher/more content
MIN_ARTICLES_FOR_LOCAL_ONLY = 3

# Messages
MESSAGES = {
    "welcome": (
        "Hola! Soy tu asistente de noticias.\n\n"
        "Puedo ayudarte a:\n"
        "- Ver las ultimas noticias\n"
        "- Buscar noticias por tema\n"
        "- Responder preguntas sobre las noticias\n\n"
        "Usa /help para ver los comandos disponibles."
    ),
    "help": (
        "📋 *COMANDOS DISPONIBLES*\n\n"
        "🔹 /ultimas [N] - Ver las últimas N noticias (default: 5)\n"
        "🔹 /buscar [tema] - Buscar noticias por tema\n"
        "🔹 /digest - Resumen del día\n"
        "🔹 /categorias - Ver categorías disponibles\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "💬 *PREGUNTAS LIBRES (RAG)*\n\n"
        "También puedes escribir cualquier pregunta SIN usar comandos:\n\n"
        "📝 Ejemplos:\n"
        "• ¿Qué está pasando con la economía?\n"
        "• Cuéntame sobre las noticias de tecnología\n"
        "• ¿Qué decisiones ha tomado Petro?\n\n"
        "El sistema buscará artículos relevantes y generará una respuesta "
        "con fuentes citadas [1], [2], etc.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "ℹ️ *DIFERENCIA*\n"
        "• /buscar → Lista de artículos\n"
        "• Pregunta libre → Respuesta generada por IA"
    ),
    "no_results": "No encontre noticias relevantes para tu consulta.",
    "error": "Ocurrio un error procesando tu solicitud. Intenta de nuevo.",
    "thinking": "Buscando informacion...",
    "no_token": (
        "El bot no esta configurado. "
        "Configura TELEGRAM_BOT_TOKEN en el archivo .env"
    ),
}

# Emoji for formatting
EMOJI = {
    "news": "\U0001F4F0",      # newspaper
    "search": "\U0001F50D",    # magnifying glass
    "robot": "\U0001F916",     # robot
    "check": "\u2705",         # check mark
    "warning": "\u26A0\uFE0F", # warning
    "link": "\U0001F517",      # link
    "calendar": "\U0001F4C5",  # calendar
    "source": "\U0001F4DA",    # books
}
