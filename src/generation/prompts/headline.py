"""
Prompts for news headline generation.

Generates multiple headline variations with different styles
(informative, engaging, SEO-optimized).
"""

# System prompt for headline generation
HEADLINE_SYSTEM_PROMPT = """Eres un editor de noticias experto en crear titulares efectivos para medios digitales.

Tu tarea es generar titulares alternativos para artículos de noticias siguiendo estas reglas:

1. LONGITUD: Entre 8-15 palabras (máximo 100 caracteres).
2. CLARIDAD: El titular debe comunicar la noticia principal de forma clara.
3. ENGAGEMENT: Debe captar la atención del lector sin ser sensacionalista.
4. PRECISIÓN: No exageres ni distorsiones los hechos.
5. ACCIÓN: Usa verbos activos cuando sea posible.
6. IDIOMA: Todos los titulares en español.

Genera exactamente 3 titulares con diferentes enfoques:
1. INFORMATIVO: Objetivo y directo, estilo periodístico tradicional.
2. ENGAGEMENT: Más atractivo, genera curiosidad (sin clickbait).
3. SEO: Optimizado para búsquedas, incluye palabras clave relevantes.

Formato de respuesta:
INFORMATIVO: [titular]
ENGAGEMENT: [titular]
SEO: [titular]"""

# User prompt template
HEADLINE_USER_TEMPLATE = """Genera 3 titulares alternativos para este artículo:

TÍTULO ORIGINAL: {title}

RESUMEN: {summary}

Responde con exactamente 3 titulares en el formato especificado."""

# Few-shot examples
HEADLINE_FEW_SHOT_EXAMPLES = [
    {
        "title": "El gobierno anuncia nuevo plan de infraestructura",
        "summary": "El presidente presentó un plan de $5.000 millones para construir carreteras y modernizar aeropuertos, generando 100.000 empleos.",
        "headlines": """INFORMATIVO: Gobierno invertirá $5.000 millones en carreteras y aeropuertos
ENGAGEMENT: El ambicioso plan que promete transformar la infraestructura del país
SEO: Plan infraestructura Colombia 2026: inversión de $5.000 millones en carreteras"""
    },
    {
        "title": "Descubren nueva especie en la Amazonía",
        "summary": "Científicos de la Universidad Nacional hallaron una rana de 2 cm de color azul brillante en el Amazonas colombiano.",
        "headlines": """INFORMATIVO: Científicos colombianos descubren nueva especie de rana en el Amazonas
ENGAGEMENT: La diminuta rana azul que sorprendió a los científicos en la selva
SEO: Nueva especie rana Amazonas Colombia: Pristimantis amazoniensis descubierta"""
    }
]


def get_headline_prompt(
    title: str,
    summary: str,
    use_few_shot: bool = False
) -> list[dict]:
    """
    Build the complete prompt for headline generation.

    Args:
        title: Original article title.
        summary: Article summary or description.
        use_few_shot: Whether to include few-shot examples.

    Returns:
        List of message dicts for the LLM.
    """
    messages = [
        {"role": "system", "content": HEADLINE_SYSTEM_PROMPT}
    ]

    if use_few_shot:
        for example in HEADLINE_FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": HEADLINE_USER_TEMPLATE.format(
                    title=example["title"],
                    summary=example["summary"]
                )
            })
            messages.append({
                "role": "assistant",
                "content": example["headlines"]
            })

    messages.append({
        "role": "user",
        "content": HEADLINE_USER_TEMPLATE.format(title=title, summary=summary)
    })

    return messages


def parse_headlines(response: str) -> dict[str, str]:
    """
    Parse the LLM response into individual headlines.

    Args:
        response: Raw LLM response with headlines.

    Returns:
        Dict with 'informativo', 'engagement', 'seo' keys.
    """
    headlines = {
        "informativo": "",
        "engagement": "",
        "seo": ""
    }

    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.upper().startswith("INFORMATIVO:"):
            headlines["informativo"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("ENGAGEMENT:"):
            headlines["engagement"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("SEO:"):
            headlines["seo"] = line.split(":", 1)[1].strip()

    return headlines


# Single headline prompt (simpler)
SINGLE_HEADLINE_PROMPT = """Genera un titular periodístico conciso (máximo 15 palabras) para esta noticia:

{content}

TITULAR:"""
