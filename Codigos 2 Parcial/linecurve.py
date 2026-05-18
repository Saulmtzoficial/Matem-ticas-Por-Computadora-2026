import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#000000",
    "axes.facecolor":   "#000000",
    "grid.color":       "#1a1a2e",
    "grid.linewidth":   0.4,
    "xtick.color":      "#555577",
    "ytick.color":      "#555577",
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "axes.labelcolor":  "#888899",
    "axes.labelsize":   8,
    "text.color":       "#ccccee",
})

NEON_CYAN    = "#00f5ff"
NEON_MAGENTA = "#ff00ff"

# ── Function ───────────────────────────────────────────────────────────────
def f(x, y):
    r = np.sqrt(x**2 + y**2) + 1e-9
    return np.sin(r) / r + 0.3 * np.cos(x * 0.8)

x = np.linspace(-6, 6, 200)
y = np.linspace(-6, 6, 200)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 5), facecolor="black")

# ── Left: 3D surface ───────────────────────────────────────────────────────
ax3d = fig.add_subplot(121, projection="3d", facecolor="black")
ax3d.set_facecolor("black")

surf = ax3d.plot_surface(
    X, Y, Z,
    cmap="plasma",
    linewidth=0,
    antialiased=True,
    alpha=0.85,
    rcount=80, ccount=80,
)

# level curve on the plane z = 0.4
ax3d.contour(X, Y, Z, levels=[0.4], zdir="z", offset=0.4,
             colors=[NEON_CYAN], linewidths=1.6, alpha=0.9)

# semi-transparent horizontal plane at z = 0.4
xx = np.array([[-6, 6], [-6, 6]])
yy = np.array([[-6, -6], [6, 6]])
zz = np.full_like(xx, 0.4)
ax3d.plot_surface(xx, yy, zz, alpha=0.08, color=NEON_CYAN, linewidth=0)

# level curve projected on the floor
ax3d.contour(X, Y, Z, levels=[0.4], zdir="z", offset=Z.min() - 0.15,
             colors=[NEON_MAGENTA], linewidths=0.8, alpha=0.5, linestyles="dashed")

ax3d.set_xlabel("x", labelpad=2)
ax3d.set_ylabel("y", labelpad=2)
ax3d.set_zlabel("f(x,y)", labelpad=2)
ax3d.set_title("surface  +  level plane", color=NEON_CYAN,
               fontsize=9, pad=6, fontfamily="monospace")

for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#111122")

ax3d.tick_params(colors="#444466")
ax3d.view_init(elev=28, azim=-55)

cbar = fig.colorbar(surf, ax=ax3d, shrink=0.45, pad=0.08, orientation="vertical")
cbar.ax.yaxis.set_tick_params(color="#555577")
cbar.outline.set_edgecolor("#222233")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#888899", fontsize=6)

# ── Right: 2D level curves ─────────────────────────────────────────────────
ax2d = fig.add_subplot(122, facecolor="black")

ax2d.contourf(X, Y, Z, levels=30, cmap="inferno", alpha=0.35)

levels_all = np.linspace(Z.min(), Z.max(), 16)
ax2d.contour(X, Y, Z, levels=levels_all,
             colors=[NEON_MAGENTA], linewidths=0.5, alpha=0.35)

cs_hi = ax2d.contour(X, Y, Z, levels=[0.4],
                     colors=[NEON_CYAN], linewidths=2.2, alpha=1.0)
ax2d.clabel(cs_hi, fmt="f = 0.40", fontsize=7,
            colors=NEON_CYAN, inline=True, inline_spacing=4)

ax2d.set_xlabel("x")
ax2d.set_ylabel("y")
ax2d.set_title("level curves  (cyan: f = 0.40)", color=NEON_CYAN,
               fontsize=9, pad=6, fontfamily="monospace")
ax2d.tick_params(colors="#444466", labelsize=7)
for spine in ax2d.spines.values():
    spine.set_edgecolor("#1a1a2e")

ax2d.set_aspect("equal")
ax2d.grid(True, linewidth=0.3, color="#111122")

# ── Final ──────────────────────────────────────────────────────────────────
fig.suptitle("f(x,y) = sin(r)/r + 0.3·cos(0.8x)",
             color="#666688", fontsize=8, fontfamily="monospace", y=0.02)
plt.tight_layout(rect=[0, 0.04, 1, 1])

plt.savefig("neon_3d_plot.png", dpi=180, bbox_inches="tight", facecolor="black")
plt.show()
print("Done! Saved as neon_3d_plot.png in the same folder as this script.")