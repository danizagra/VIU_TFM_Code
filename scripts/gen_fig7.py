"""Generate Figure 7: UMAP + HDBSCAN visualization (vertical, with color)."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

np.random.seed(42)

# --- Simulated data ---
# Panel 1: High-dimensional (random scatter)
n_points = 30
x1 = np.random.uniform(0.1, 0.9, n_points)
y1 = np.random.uniform(0.1, 0.9, n_points)

# Panel 2: After UMAP reduction - points start to form groups
# 4 loose clusters + some noise
clusters_umap = {
    "Política": (np.random.normal(0.2, 0.06, 6), np.random.normal(0.8, 0.06, 6)),
    "Deportes": (np.random.normal(0.8, 0.07, 7), np.random.normal(0.8, 0.05, 7)),
    "Tecnología": (np.random.normal(0.3, 0.07, 7), np.random.normal(0.25, 0.06, 7)),
    "Economía": (np.random.normal(0.78, 0.06, 6), np.random.normal(0.22, 0.06, 6)),
}
# Noise points
noise_x = np.random.uniform(0.3, 0.7, 4)
noise_y = np.random.uniform(0.4, 0.6, 4)

# Colors per cluster
colors = {
    "Política": "#2196F3",    # Blue
    "Deportes": "#4CAF50",    # Green
    "Tecnología": "#9C27B0",  # Purple
    "Economía": "#FF9800",    # Orange
}
noise_color = "#9E9E9E"  # Gray

# --- Figure setup ---
fig, axes = plt.subplots(3, 1, figsize=(6, 16))
fig.subplots_adjust(hspace=0.45)

# ============================================================
# Panel 1: Original embeddings (768 dimensions)
# ============================================================
ax1 = axes[0]
ax1.scatter(x1, y1, c="#9E9E9E", s=50, alpha=0.6, edgecolors="#666666", linewidths=0.5)
ax1.set_title("Embeddings originales\n(768 dimensiones)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("")  # Label moved into the plot area
ax1.text(0.5, 0.03, "Alta dimensionalidad (difícil de visualizar)",
         transform=ax1.transAxes, ha="center", fontsize=9, fontstyle="italic", color="#666")
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines["top"].set_visible(True)
ax1.spines["right"].set_visible(True)

# Arrow 1: Panel 1 → Panel 2
arrow1_y = 0.675
fig.patches.append(mpatches.FancyArrowPatch(
    (0.35, arrow1_y + 0.012), (0.35, arrow1_y - 0.012),
    transform=fig.transFigure,
    arrowstyle="-|>", mutation_scale=25,
    color="#1565C0", linewidth=3,
))
fig.text(0.55, arrow1_y, "Reducción UMAP", ha="center", va="center", fontsize=11,
         fontweight="bold", color="#1565C0")

# ============================================================
# Panel 2: UMAP reduction (2 dimensions)
# ============================================================
ax2 = axes[1]

# Plot each cluster with soft/desaturated color (groups emerging but not yet identified)
# Use lighter, pastel versions of the final cluster colors
colors_soft = {
    "Política": "#90CAF9",    # Light blue
    "Deportes": "#A5D6A7",    # Light green
    "Tecnología": "#CE93D8",  # Light purple
    "Economía": "#FFCC80",    # Light orange
}
all_x, all_y = [], []
for name, (cx, cy) in clusters_umap.items():
    ax2.scatter(cx, cy, c=colors_soft[name], s=55, alpha=0.7,
                edgecolors=colors[name], linewidths=0.6)
    all_x.extend(cx)
    all_y.extend(cy)

# Noise
ax2.scatter(noise_x, noise_y, c="#D5D5D5", s=40, alpha=0.5, edgecolors="#999", linewidths=0.5)

ax2.set_title("Reducción UMAP\n(2 dimensiones)", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("UMAP Dimensión 1", fontsize=10)
ax2.set_ylabel("UMAP Dimensión 2", fontsize=10)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

# Arrow 2: Panel 2 → Panel 3
arrow2_y = 0.36
fig.patches.append(mpatches.FancyArrowPatch(
    (0.35, arrow2_y + 0.012), (0.35, arrow2_y - 0.012),
    transform=fig.transFigure,
    arrowstyle="-|>", mutation_scale=25,
    color="#E65100", linewidth=3,
))
fig.text(0.55, arrow2_y, "Clustering HDBSCAN", ha="center", va="center", fontsize=11,
         fontweight="bold", color="#E65100")

# ============================================================
# Panel 3: HDBSCAN clusters identified
# ============================================================
ax3 = axes[2]

for name, (cx, cy) in clusters_umap.items():
    color = colors[name]
    ax3.scatter(cx, cy, c=color, s=60, alpha=0.8, edgecolors="white", linewidths=0.8, zorder=3)

    # Draw cluster boundary (ellipse)
    center_x, center_y = np.mean(cx), np.mean(cy)
    radius_x = max(np.std(cx) * 2.8, 0.12)
    radius_y = max(np.std(cy) * 2.8, 0.12)

    ellipse = mpatches.Ellipse(
        (center_x, center_y), radius_x * 2, radius_y * 2,
        fill=True, facecolor=color, alpha=0.1,
        edgecolor=color, linewidth=1.8, linestyle="--", zorder=1,
    )
    ax3.add_patch(ellipse)

    # Cluster label
    ax3.annotate(
        name, (center_x, center_y + radius_y + 0.04),
        ha="center", fontsize=10, fontweight="bold", color=color,
        fontstyle="italic",
    )

# Noise points (not assigned to any cluster)
ax3.scatter(noise_x, noise_y, c=noise_color, s=40, alpha=0.5,
            marker="x", linewidths=1.2, zorder=2)

ax3.set_title("Clusters identificados\n(HDBSCAN)", fontsize=13, fontweight="bold", pad=12)
ax3.set_xlabel("UMAP Dimensión 1", fontsize=10)
ax3.set_ylabel("UMAP Dimensión 2", fontsize=10)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

# Legend
legend_handles = [
    mpatches.Patch(color=c, label=n, alpha=0.8) for n, c in colors.items()
]
legend_handles.append(
    plt.Line2D([0], [0], marker="x", color=noise_color, linestyle="None",
               markersize=8, label="Ruido (sin cluster)")
)
ax3.legend(handles=legend_handles, loc="lower right", fontsize=9,
           framealpha=0.9, edgecolor="#BDBDBD")

# --- Save ---
output = "images/tfm_figures/fig7_umap_hdbscan.png"
plt.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output}")
