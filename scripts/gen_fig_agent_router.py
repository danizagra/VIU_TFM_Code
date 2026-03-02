"""Generate figure: Agent Router - simple B&W flowchart."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")

# ============================================================
# Colors - B&W palette
# ============================================================
c_border = "#333333"
c_text = "#333333"
c_light = "#F5F5F5"
c_mid = "#E0E0E0"

# ============================================================
# Helper: rounded box with text
# ============================================================
def draw_box(x, y, w, h, text, fontsize=10, bold=True, bg=c_light,
             border=c_border, lw=1.8, subtext=None, subsize=8.5):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=bg, edgecolor=border, linewidth=lw,
    )
    ax.add_patch(rect)
    y_offset = 0.12 if subtext else 0
    ax.text(x + w / 2, y + h / 2 + y_offset, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", color=c_text)
    if subtext:
        ax.text(x + w / 2, y + h / 2 - 0.22, subtext,
                ha="center", va="center", fontsize=subsize,
                color="#666", fontstyle="italic")

def draw_arrow(x1, y1, x2, y2, lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c_border,
                                lw=lw, mutation_scale=20))

# ============================================================
# 1. User Query (top)
# ============================================================
draw_box(3.5, 7.0, 3.0, 0.7, "Consulta del usuario")

# Arrow down
draw_arrow(5, 6.95, 5, 6.45)

# ============================================================
# 2. Router (diamond shape - using rotated square)
# ============================================================
# Draw diamond manually
diamond_cx, diamond_cy = 5, 5.7
diamond_w, diamond_h = 1.6, 0.7
diamond = plt.Polygon([
    (diamond_cx, diamond_cy + diamond_h),
    (diamond_cx + diamond_w, diamond_cy),
    (diamond_cx, diamond_cy - diamond_h),
    (diamond_cx - diamond_w, diamond_cy),
], closed=True, facecolor=c_mid, edgecolor=c_border, linewidth=2.0)
ax.add_patch(diamond)
ax.text(diamond_cx, diamond_cy + 0.12, "Agent Router",
        ha="center", va="center", fontsize=10, fontweight="bold", color=c_text)
ax.text(diamond_cx, diamond_cy - 0.18, "(LLM clasifica)",
        ha="center", va="center", fontsize=8, color="#555", fontstyle="italic")

# ============================================================
# 3. Three branches
# ============================================================
# Arrow positions from diamond bottom-left, bottom, bottom-right
branch_y_start = diamond_cy - diamond_h  # bottom of diamond

box_w = 2.6
box_h = 1.6
box_y = 1.8
box_centers = [1.7, 5.0, 8.3]

# Arrows from diamond to boxes
for cx in box_centers:
    # Start from diamond edge
    if cx < diamond_cx:
        start_x = diamond_cx - diamond_w * 0.5
        start_y = diamond_cy - diamond_h * 0.5
    elif cx > diamond_cx:
        start_x = diamond_cx + diamond_w * 0.5
        start_y = diamond_cy - diamond_h * 0.5
    else:
        start_x = diamond_cx
        start_y = branch_y_start

    ax.annotate("",
                xy=(cx, box_y + box_h + 0.05),
                xytext=(start_x, start_y),
                arrowprops=dict(arrowstyle="-|>", color=c_border,
                                lw=1.8, mutation_scale=18))

# Route labels on arrows
ax.text(2.6, 4.25, "LOCAL_RAG", ha="center", va="center",
        fontsize=8, fontweight="bold", color=c_text, family="monospace",
        rotation=30,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="#CCC", linewidth=0.8))
ax.text(5, 4.35, "EXTERNAL\nSEARCH", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color=c_text, family="monospace",
        linespacing=1.1,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="#CCC", linewidth=0.8))
ax.text(7.4, 4.25, "COMBINED", ha="center", va="center",
        fontsize=8, fontweight="bold", color=c_text, family="monospace",
        rotation=-30,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor="#CCC", linewidth=0.8))

# ============================================================
# Three route boxes
# ============================================================
routes = [
    ("Búsqueda local\n(RAG)", "pgvector + reranking\n~1 segundo"),
    ("Búsqueda externa\n(Agente)", "NewsAPI, GNews\n~166 segundos"),
    ("Combinada\n(Local → Externo)", "RAG primero, si falla\nbusca en fuentes externas"),
]

for cx, (title, desc) in zip(box_centers, routes):
    x = cx - box_w / 2
    rect = mpatches.FancyBboxPatch(
        (x, box_y), box_w, box_h,
        boxstyle="round,pad=0.12",
        facecolor=c_light, edgecolor=c_border, linewidth=1.8,
    )
    ax.add_patch(rect)
    ax.text(cx, box_y + box_h / 2 + 0.25, title,
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=c_text, linespacing=1.3)
    # Separator
    ax.plot([x + 0.3, x + box_w - 0.3],
            [box_y + box_h / 2 - 0.05, box_y + box_h / 2 - 0.05],
            color="#CCC", lw=0.8)
    ax.text(cx, box_y + box_h / 2 - 0.4, desc,
            ha="center", va="center", fontsize=8.5,
            color="#555", fontstyle="italic", linespacing=1.3)

# ============================================================
# Converging arrows to Response
# ============================================================
resp_w, resp_h = 3.0, 0.65
resp_x = 5 - resp_w / 2
resp_y = 0.7

for cx in box_centers:
    target_x = 5 + (cx - 5) * 0.25
    ax.annotate("",
                xy=(target_x, resp_y + resp_h + 0.05),
                xytext=(cx, box_y - 0.05),
                arrowprops=dict(arrowstyle="-|>", color=c_border,
                                lw=1.5, mutation_scale=16, alpha=0.5))

draw_box(resp_x, resp_y, resp_w, resp_h,
         "Respuesta con citas [1], [2]", fontsize=10, bg=c_mid)

# ============================================================
# Fallback note
# ============================================================
ax.text(5, 0.18,
        "Fallback: si el LLM falla → se usa COMBINED por defecto",
        ha="center", va="center", fontsize=8, color="#999",
        fontstyle="italic")

# --- Save ---
output = "images/tfm_figures/fig_agent_router.png"
plt.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output}")
