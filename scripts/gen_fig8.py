"""Generate Figure 8: Cosine similarity + threshold interpretation (with color)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig = plt.figure(figsize=(7, 10))

# Two panels stacked vertically - tighter layout
ax_vectors = fig.add_axes([0.08, 0.52, 0.84, 0.42])   # Top: vector diagram
ax_bar = fig.add_axes([0.08, 0.03, 0.84, 0.38])       # Bottom: threshold bar

# ============================================================
# Panel 1: Cosine similarity between vectors
# ============================================================
ax = ax_vectors
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(-0.15, 1.1)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Similitud coseno entre vectores", fontsize=14, fontweight="bold", pad=12)

# Document A (reference vector along x-axis)
ax.annotate("", xy=(1.0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#333333", lw=2.2))
ax.text(1.03, -0.05, "Documento A", fontsize=10, fontweight="bold", va="top")

# Document B (similar - small angle ~18°)
angle_b = np.radians(18)
bx, by = 0.85 * np.cos(angle_b), 0.85 * np.sin(angle_b)
ax.annotate("", xy=(bx, by), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#4CAF50", lw=2.2))
ax.text(bx + 0.03, by + 0.04, "Documento B\n(similar)", fontsize=10,
        fontweight="bold", color="#2E7D32")

# Document C (different - large angle ~75°)
angle_c = np.radians(75)
cx, cy = 0.75 * np.cos(angle_c), 0.75 * np.sin(angle_c)
ax.annotate("", xy=(cx, cy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="#F44336", lw=2.2))
ax.text(cx - 0.18, cy + 0.03, "Documento C\n(diferente)", fontsize=10,
        fontweight="bold", color="#C62828")

# Angle arc for B (small angle)
theta_b = np.linspace(0, angle_b, 30)
r_small = 0.3
ax.plot(r_small * np.cos(theta_b), r_small * np.sin(theta_b),
        color="#4CAF50", lw=1.5, ls="--")
ax.text(0.33, 0.06, "θ pequeño\n≈ 0.95", fontsize=9, fontstyle="italic",
        color="#2E7D32")

# Angle arc for C (large angle)
theta_c = np.linspace(0, angle_c, 50)
r_large = 0.2
ax.plot(r_large * np.cos(theta_c), r_large * np.sin(theta_c),
        color="#F44336", lw=1.5, ls="--")
ax.text(0.05, 0.22, "θ grande\n≈ 0.50", fontsize=9, fontstyle="italic",
        color="#C62828")

# Arrow transition - centered, prominent, with label to the right
fig.patches.append(mpatches.FancyArrowPatch(
    (0.42, 0.51), (0.42, 0.455),
    transform=fig.transFigure,
    arrowstyle="-|>", mutation_scale=30,
    color="#1565C0", linewidth=3,
))
fig.text(0.58, 0.483, "Se interpreta como", ha="center", va="center", fontsize=11,
         fontweight="bold", color="#1565C0")

# ============================================================
# Panel 2: Threshold interpretation bar
# ============================================================
ax2 = ax_bar
ax2.set_xlim(-0.05, 1.18)
ax2.set_ylim(-1.3, 1.2)
ax2.axis("off")
ax2.set_title("Umbrales de interpretación", fontsize=14, fontweight="bold", pad=12)

# Define segments: (start, end, label_inside, bg_color, border_color)
segments = [
    (0.0,  0.50, "Sin relación",              "#FFCDD2", "#B71C1C"),
    (0.50, 0.65, "Relación\nparcial",          "#FFF9C4", "#F57F17"),
    (0.65, 0.95, "Solapamiento\ntemático",      "#C8E6C9", "#1B5E20"),
    (0.95, 1.00, None,                          "#BBDEFB", "#0D47A1"),  # label outside
]

bar_y = 0.2
bar_h = 0.55

for start, end, label, bg_color, border_color in segments:
    width = end - start
    # Use plain Rectangle for exact alignment with tick marks
    rect = plt.Rectangle(
        (start, bar_y), width, bar_h,
        facecolor=bg_color, edgecolor=border_color, linewidth=1.5,
    )
    ax2.add_patch(rect)

    # Label centered in segment (only if it fits)
    if label:
        mid_x = (start + end) / 2
        ax2.text(mid_x, bar_y + bar_h / 2, label,
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color=border_color)

# "Misma noticia" label outside the bar (segment too narrow)
ax2.annotate(
    "Misma\nnoticia",
    xy=(0.975, bar_y + bar_h / 2),
    xytext=(1.12, bar_y + bar_h / 2),
    ha="center", va="center", fontsize=10, fontweight="bold", color="#0D47A1",
    arrowprops=dict(arrowstyle="-", color="#0D47A1", lw=1.2),
)

# Threshold numbers directly below the bar with tick marks
thresholds = [0.0, 0.50, 0.65, 0.95, 1.0]
for t in thresholds:
    ax2.plot([t, t], [bar_y, bar_y - 0.1], color="#333", lw=1.2)
    ax2.text(t, bar_y - 0.15, f"{t:.2f}", ha="center", va="top",
             fontsize=10, fontweight="bold", color="#333")

# Examples below the bar - each points to the CENTER of its segment
# Two rows (staggered) to avoid overlap
row1_y = bar_y - 0.52
row2_y = bar_y - 0.82

examples = [
    (0.25,  0.25,  '"Crisis en Gaza"\nvs "Receta de pasta"',    row1_y, "#B71C1C"),
    (0.575, 0.575, '"Economía UE"\nvs "PIB España"',            row2_y, "#F57F17"),
    (0.80,  0.80,  '"Crisis hídrica Asia"\nvs "Sequía India"',   row1_y, "#1B5E20"),
    (0.975, 0.975, '"Reuters" vs "AP"\n(misma noticia)',          row2_y, "#0D47A1"),
]

for x_target, x_text, text, y_text, color in examples:
    ax2.annotate(
        text,
        xy=(x_target, bar_y),  # arrow tip at exact bottom edge of bar
        xytext=(x_text, y_text),
        ha="center", va="top", fontsize=8.5, fontstyle="italic", color=color,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, alpha=0.6),
    )

# --- Save ---
output = "images/tfm_figures/fig8_similitud_coseno.png"
plt.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output}")
