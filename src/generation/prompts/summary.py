"""
Prompts for news article summarization.

Includes system prompts, user templates, and few-shot examples
optimized for Spanish-language news content.
"""

# System prompt for summarization
SUMMARY_SYSTEM_PROMPT = """Eres un asistente de redacción periodística especializado en crear resúmenes concisos y precisos de noticias.

Tu tarea es resumir artículos de noticias siguiendo estas reglas:

1. LONGITUD: El resumen debe tener entre 2-4 oraciones (50-100 palabras).
2. CONTENIDO: Incluye los elementos clave: quién, qué, cuándo, dónde y por qué.
3. OBJETIVIDAD: Mantén un tono neutral y objetivo, sin opiniones personales.
4. FIDELIDAD: Solo incluye información presente en el artículo original. NO inventes datos.
5. CLARIDAD: Usa lenguaje claro y directo, evitando jerga innecesaria.
6. IDIOMA: Responde siempre en español.

IMPORTANTE: Siempre genera un resumen con la información disponible, aunque sea parcial.
El título y la descripción proporcionan contexto suficiente para crear un resumen útil.
NUNCA respondas diciendo que no puedes generar un resumen o que necesitas más información.

NO incluyas:
- Frases como "El artículo habla de..." o "En resumen..."
- Opiniones o valoraciones personales
- Información que no esté en el texto original
- Repeticiones innecesarias
- Frases indicando que falta información o contenido"""

# User prompt template
SUMMARY_USER_TEMPLATE = """Resume el siguiente artículo de noticias:

TÍTULO: {title}

CONTENIDO:
{content}

Genera un resumen conciso de 2-4 oraciones."""

# Few-shot examples for better quality
SUMMARY_FEW_SHOT_EXAMPLES = [
    {
        "title": "Colombia anuncia inversión de $5.000 millones en infraestructura",
        "content": """El presidente Gustavo Petro anunció hoy un ambicioso plan de infraestructura
        valorado en $5.000 millones de dólares. El plan incluye la construcción de 500 kilómetros
        de nuevas carreteras y la modernización de 20 aeropuertos en todo el país. Según el mandatario,
        esta inversión generará aproximadamente 100.000 nuevos empleos en los próximos cinco años.
        La oposición ha cuestionado las fuentes de financiamiento del proyecto.""",
        "summary": """El gobierno colombiano presentó un plan de infraestructura de $5.000 millones
        que contempla 500 km de carreteras y la modernización de 20 aeropuertos. El presidente Petro
        estima que el proyecto creará 100.000 empleos en cinco años, aunque la oposición cuestiona
        su financiamiento."""
    },
    {
        "title": "Científicos descubren nueva especie de rana en la Amazonía colombiana",
        "content": """Un equipo de biólogos de la Universidad Nacional de Colombia descubrió una
        nueva especie de rana en el departamento del Amazonas. La especie, bautizada como
        'Pristimantis amazoniensis', mide apenas 2 centímetros y tiene un distintivo color
        azul brillante. Los investigadores señalan que el hallazgo resalta la importancia de
        conservar los ecosistemas amazónicos, que siguen albergando especies desconocidas para
        la ciencia. El estudio fue publicado en la revista Nature.""",
        "summary": """Biólogos de la Universidad Nacional descubrieron una nueva especie de rana
        en el Amazonas colombiano, denominada 'Pristimantis amazoniensis'. El anfibio de 2 cm
        y color azul brillante fue documentado en un estudio publicado en Nature, destacando
        la biodiversidad aún inexplorada de la región."""
    }
]


def get_summary_prompt(title: str, content: str, use_few_shot: bool = False) -> list[dict]:
    """
    Build the complete prompt for summarization.

    Args:
        title: Article title.
        content: Article content/description.
        use_few_shot: Whether to include few-shot examples.

    Returns:
        List of message dicts for the LLM.
    """
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT}
    ]

    # Add few-shot examples if requested
    if use_few_shot:
        for example in SUMMARY_FEW_SHOT_EXAMPLES:
            # User message with article
            messages.append({
                "role": "user",
                "content": SUMMARY_USER_TEMPLATE.format(
                    title=example["title"],
                    content=example["content"]
                )
            })
            # Assistant response with summary
            messages.append({
                "role": "assistant",
                "content": example["summary"]
            })

    # Add the actual article to summarize
    messages.append({
        "role": "user",
        "content": SUMMARY_USER_TEMPLATE.format(title=title, content=content)
    })

    return messages


# Alternative: Shorter prompt for faster inference
SUMMARY_CONCISE_PROMPT = """Resume esta noticia en 2-3 oraciones, incluyendo los hechos principales (quién, qué, cuándo, dónde):

TÍTULO: {title}

{content}

RESUMEN:"""
