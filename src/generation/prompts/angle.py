"""
Prompts for generating journalistic angles.

Suggests different perspectives and approaches
for covering a news story or topic cluster.
"""

# System prompt for angle generation
ANGLE_SYSTEM_PROMPT = """Eres un editor de redacción periodística con experiencia en identificar ángulos de cobertura únicos y relevantes.

Tu tarea es analizar noticias y sugerir diferentes ángulos periodísticos para profundizar la cobertura. Cada ángulo debe:

1. RELEVANCIA: Ser relevante para la audiencia y el contexto actual.
2. ORIGINALIDAD: Ofrecer una perspectiva diferente a la cobertura estándar.
3. VIABILIDAD: Ser realizable con fuentes accesibles.
4. PROFUNDIDAD: Permitir investigación más allá del hecho básico.
5. IMPACTO: Conectar con temas de interés público o social.

Tipos de ángulos a considerar:
- HUMANO: Historias de personas afectadas o involucradas.
- EXPLICATIVO: Contexto, causas y consecuencias.
- DATOS: Análisis con cifras, tendencias o comparaciones.
- INVESTIGATIVO: Aspectos no revelados o poco explorados.
- LOCAL: Impacto en comunidades específicas.
- PROSPECTIVO: Qué pasará después, escenarios futuros.

Formato de respuesta:
ÁNGULO 1:
- Tipo: [tipo]
- Enfoque: [descripción breve del ángulo]
- Pregunta clave: [la pregunta que busca responder]
- Fuentes sugeridas: [tipos de fuentes a consultar]

ÁNGULO 2:
[mismo formato]

ÁNGULO 3:
[mismo formato]"""

# User prompt template
ANGLE_USER_TEMPLATE = """Analiza esta noticia y sugiere 3 ángulos periodísticos para profundizar la cobertura:

TÍTULO: {title}

RESUMEN: {summary}

{context_section}

Genera exactamente 3 ángulos diferentes y complementarios."""

# Template for context section when multiple related articles exist
CONTEXT_TEMPLATE = """ARTÍCULOS RELACIONADOS EN EL CLUSTER:
{related_articles}"""

# Few-shot examples
ANGLE_FEW_SHOT_EXAMPLES = [
    {
        "title": "Gobierno anuncia aumento del salario mínimo del 12%",
        "summary": "El Ministerio de Trabajo anunció un incremento del 12% en el salario mínimo para 2026, pasando de $1.300.000 a $1.456.000 mensuales. La decisión se tomó sin consenso entre empresarios y sindicatos.",
        "context": "",
        "angles": """ÁNGULO 1:
- Tipo: HUMANO
- Enfoque: Cómo el aumento afecta a familias de trabajadores de diferentes sectores
- Pregunta clave: ¿Alcanzará este aumento para cubrir la canasta familiar básica?
- Fuentes sugeridas: Trabajadores de sectores informales, economistas del hogar, familias beneficiarias

ÁNGULO 2:
- Tipo: DATOS
- Enfoque: Comparación histórica del poder adquisitivo del salario mínimo vs inflación
- Pregunta clave: ¿Ha ganado o perdido poder adquisitivo el salario mínimo en la última década?
- Fuentes sugeridas: DANE, Banco de la República, estudios de universidades económicas

ÁNGULO 3:
- Tipo: PROSPECTIVO
- Enfoque: Impacto en el empleo formal e informal durante 2026
- Pregunta clave: ¿Cuántos empleos formales podrían perderse o crearse con este ajuste?
- Fuentes sugeridas: Gremios empresariales, ANIF, expertos laborales, empresarios PYME"""
    },
    {
        "title": "Sequía afecta a 15 departamentos del país",
        "summary": "El IDEAM declaró alerta roja por sequía en 15 departamentos. Los embalses están al 40% de capacidad y se prevén racionamientos de agua y energía en las próximas semanas.",
        "context": "",
        "angles": """ÁNGULO 1:
- Tipo: LOCAL
- Enfoque: Comunidades rurales sin acceso a agua potable durante la emergencia
- Pregunta clave: ¿Cómo sobreviven las familias en zonas sin acueducto durante la sequía?
- Fuentes sugeridas: Líderes comunitarios, alcaldías municipales, familias afectadas, ONG de ayuda humanitaria

ÁNGULO 2:
- Tipo: INVESTIGATIVO
- Enfoque: Estado real de la infraestructura de embalses y gestión del agua
- Pregunta clave: ¿Por qué Colombia sigue siendo vulnerable a sequías a pesar de inversiones en infraestructura?
- Fuentes sugeridas: Contraloría, empresas de servicios públicos, ingenieros hidráulicos, documentos de planificación

ÁNGULO 3:
- Tipo: EXPLICATIVO
- Enfoque: Conexión entre el fenómeno de El Niño, cambio climático y sequías recurrentes
- Pregunta clave: ¿Se están volviendo más frecuentes e intensas las sequías en Colombia?
- Fuentes sugeridas: IDEAM, científicos del clima, estudios de universidades, informes del IPCC"""
    }
]


def get_angle_prompt(
    title: str,
    summary: str,
    related_articles: list[str] | None = None,
    use_few_shot: bool = False
) -> list[dict]:
    """
    Build the complete prompt for angle generation.

    Args:
        title: Main article title.
        summary: Article summary.
        related_articles: List of related article summaries (for cluster context).
        use_few_shot: Whether to include few-shot examples.

    Returns:
        List of message dicts for the LLM.
    """
    messages = [
        {"role": "system", "content": ANGLE_SYSTEM_PROMPT}
    ]

    if use_few_shot:
        for example in ANGLE_FEW_SHOT_EXAMPLES:
            # Build context section
            context_section = ""
            if example["context"]:
                context_section = CONTEXT_TEMPLATE.format(
                    related_articles=example["context"]
                )

            messages.append({
                "role": "user",
                "content": ANGLE_USER_TEMPLATE.format(
                    title=example["title"],
                    summary=example["summary"],
                    context_section=context_section
                )
            })
            messages.append({
                "role": "assistant",
                "content": example["angles"]
            })

    # Build context section for actual request
    context_section = ""
    if related_articles:
        articles_text = "\n".join(
            f"- {article}" for article in related_articles
        )
        context_section = CONTEXT_TEMPLATE.format(
            related_articles=articles_text
        )

    messages.append({
        "role": "user",
        "content": ANGLE_USER_TEMPLATE.format(
            title=title,
            summary=summary,
            context_section=context_section
        )
    })

    return messages


def parse_angles(response: str) -> list[dict]:
    """
    Parse the LLM response into structured angle data.

    Args:
        response: Raw LLM response with angles.

    Returns:
        List of dicts with angle information.
    """
    angles = []
    current_angle = {}

    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()

        if line.upper().startswith("ÁNGULO") or line.upper().startswith("ANGULO"):
            # Save previous angle if exists
            if current_angle:
                angles.append(current_angle)
            current_angle = {}

        elif line.startswith("- Tipo:"):
            current_angle["tipo"] = line.split(":", 1)[1].strip()

        elif line.startswith("- Enfoque:"):
            current_angle["enfoque"] = line.split(":", 1)[1].strip()

        elif line.startswith("- Pregunta clave:"):
            current_angle["pregunta_clave"] = line.split(":", 1)[1].strip()

        elif line.startswith("- Fuentes sugeridas:"):
            current_angle["fuentes"] = line.split(":", 1)[1].strip()

    # Don't forget the last angle
    if current_angle:
        angles.append(current_angle)

    return angles


# Simple angle prompt for quick suggestions
SIMPLE_ANGLE_PROMPT = """Sugiere 3 ángulos periodísticos breves para profundizar esta noticia:

{content}

Formato: Una línea por ángulo, comenzando con el tipo (HUMANO/DATOS/INVESTIGATIVO/LOCAL/EXPLICATIVO/PROSPECTIVO):

1.
2.
3."""
