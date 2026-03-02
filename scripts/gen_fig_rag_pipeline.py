"""Generate figure: RAG pipeline - Recuperación, Aumento, Generación."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np


fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")

# ============================================================
# Colors
# ============================================================
c_query = "#5C6BC0"      # Indigo - query
c_retrieval = "#1E88E5"  # Blue - retrieval
c_augment = "#FF9800"    # Orange - augmentation
c_generate = "#43A047"   # Green - generation
c_response = "#5C6BC0"   # Indigo - response
c_bg_ret = "#E3F2FD"
c_bg_aug = "#FFF3E0"
c_bg_gen = "#E8F5E9"

# ============================================================
# Stage boxes - 3 main stages
# ============================================================
box_y = 2.2
box_h = 2.8
box_w = 3.0
gap = 0.8
start_x = 1.8

stages = [
    (start_x,                    "1. Recuperación",  c_retrieval, c_bg_ret,
     "Búsqueda vectorial\n(embeddings + coseno)\n\nDense Passage\nRetrieval (DPR)"),
    (start_x + box_w + gap,      "2. Aumento",       c_augment, c_bg_aug,
     "Reranking de\ndocumentos\n\nFiltrado por\nrelevancia"),
    (start_x + 2*(box_w + gap),  "3. Generación",    c_generate, c_bg_gen,
     "LLM genera respuesta\ncon contexto\n\nCitas [1], [2]\nfundamentadas"),
]

for x, title, color, bg, desc in stages:
    # Background box
    rect = mpatches.FancyBboxPatch(
        (x, box_y), box_w, box_h,
        boxstyle="round,pad=0.15",
        facecolor=bg, edgecolor=color, linewidth=2.5,
    )
    ax.add_patch(rect)

    # Title bar at top of box
    title_rect = mpatches.FancyBboxPatch(
        (x + 0.05, box_y + box_h - 0.6), box_w - 0.1, 0.55,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=color, linewidth=0,
    )
    ax.add_patch(title_rect)
    ax.text(x + box_w / 2, box_y + box_h - 0.32, title,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="white")

    # Description
    ax.text(x + box_w / 2, box_y + (box_h - 0.6) / 2 + 0.1, desc,
            ha="center", va="center", fontsize=9, color="#333",
            linespacing=1.4)

# ============================================================
# Arrows between stages
# ============================================================
arrow_y = box_y + box_h / 2

for i in range(2):
    x_start = start_x + (i + 1) * box_w + i * gap + 0.1
    x_end = x_start + gap - 0.2
    ax.annotate("", xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color="#555",
                                lw=2.5, mutation_scale=22))

# ============================================================
# Query input (left)
# ============================================================
q_x = 0.5
q_y = arrow_y
ax.annotate("", xy=(start_x - 0.15, q_y), xytext=(q_x + 0.8, q_y),
            arrowprops=dict(arrowstyle="-|>", color=c_query,
                            lw=2.5, mutation_scale=22))

# Query icon + text
query_box = mpatches.FancyBboxPatch(
    (q_x - 0.05, q_y - 0.5), 0.9, 1.0,
    boxstyle="round,pad=0.1",
    facecolor="#E8EAF6", edgecolor=c_query, linewidth=2,
)
ax.add_patch(query_box)
ax.text(q_x + 0.4, q_y + 0.15, "?", ha="center", va="center",
        fontsize=24, fontweight="bold", color=c_query)
ax.text(q_x + 0.4, q_y - 0.25, "Query", ha="center", va="center",
        fontsize=9, fontweight="bold", color=c_query)

# ============================================================
# Response output (right)
# ============================================================
r_x = start_x + 3 * box_w + 2 * gap + 0.15
ax.annotate("", xy=(r_x + 0.3, q_y), xytext=(r_x - 0.3, q_y),
            arrowprops=dict(arrowstyle="-|>", color=c_response,
                            lw=2.5, mutation_scale=22))

resp_box = mpatches.FancyBboxPatch(
    (r_x + 0.25, q_y - 0.5), 1.0, 1.0,
    boxstyle="round,pad=0.1",
    facecolor="#E8EAF6", edgecolor=c_response, linewidth=2,
)
ax.add_patch(resp_box)
ax.text(r_x + 0.75, q_y + 0.15, "R", ha="center", va="center",
        fontsize=22, fontweight="bold", color=c_response)
ax.text(r_x + 0.75, q_y - 0.25, "Respuesta", ha="center", va="center",
        fontsize=9, fontweight="bold", color=c_response)

# ============================================================
# Bottom: data source icons
# ============================================================
# Documents below retrieval
doc_y = 1.2
doc_x = start_x + box_w / 2

ax.annotate("", xy=(doc_x, box_y - 0.05), xytext=(doc_x, doc_y + 0.55),
            arrowprops=dict(arrowstyle="-|>", color=c_retrieval,
                            lw=1.8, mutation_scale=16, ls="--"))

# Document icons
for dx in [-0.6, 0, 0.6]:
    doc_rect = mpatches.FancyBboxPatch(
        (doc_x + dx - 0.22, doc_y - 0.35), 0.44, 0.55,
        boxstyle="round,pad=0.05",
        facecolor="#BBDEFB", edgecolor=c_retrieval, linewidth=1.2,
    )
    ax.add_patch(doc_rect)
    for ly in [0.1, 0.0, -0.1]:
        ax.plot([doc_x + dx - 0.13, doc_x + dx + 0.13],
                [doc_y + ly, doc_y + ly],
                color=c_retrieval, lw=0.8, alpha=0.5)

ax.text(doc_x, doc_y - 0.6, "Base de conocimiento\n(PostgreSQL + pgvector)",
        ha="center", va="top", fontsize=9, color="#555", fontstyle="italic")

# LLM below generation
llm_x = start_x + 2 * (box_w + gap) + box_w / 2
llm_y = doc_y

ax.annotate("", xy=(llm_x, box_y - 0.05), xytext=(llm_x, llm_y + 0.55),
            arrowprops=dict(arrowstyle="-|>", color=c_generate,
                            lw=1.8, mutation_scale=16, ls="--"))

# LLM icon
llm_circle = mpatches.Circle(
    (llm_x, llm_y), 0.4,
    facecolor="#C8E6C9", edgecolor=c_generate, linewidth=1.5,
)
ax.add_patch(llm_circle)
ax.text(llm_x, llm_y + 0.05, "LLM", ha="center", va="center",
        fontsize=12, fontweight="bold", color=c_generate)

ax.text(llm_x, llm_y - 0.6, "Conocimiento previo\n+ contexto recuperado",
        ha="center", va="top", fontsize=9, color="#555", fontstyle="italic")

# ============================================================
# Title
# ============================================================
ax.text(7, 6.3, "Pipeline RAG (Retrieval-Augmented Generation)",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#333")

# --- Save ---
output = "images/tfm_figures/fig_rag_pipeline.png"
plt.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output}")
