import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ── Paleta (UASLP Mechatronics Style) ──────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"
CLR_TRUE  = "#66BB6A" # Real function
CLR_UNIF  = "#FF7043" # Uniform knots
CLR_OPT   = "#CE93D8" # Strategic knots

# ── Datos de APP9 ──────────────────────────────────────────────────────────
def target_func(x):
    # A function with high curvature at the start and flat at the end
    return np.sin(x**2 / 5)

x_fine = np.linspace(0, 10, 500)
y_fine = target_func(x_fine)

# Case 1: 4 Uniformly spaced points
x_unif = np.linspace(0, 10, 4)
y_unif = target_func(x_unif)
spl_unif = CubicSpline(x_unif, y_unif, bc_type='natural')

# Case 2: 4 Strategically placed points (near peaks/curvature changes)
x_opt  = np.array([0, 2.2, 4.5, 10]) 
y_opt  = target_func(x_opt)
spl_opt  = CubicSpline(x_opt, y_opt, bc_type='natural')

# ── Interfaz de Visualización ──────────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG_FIG)
ax.set_facecolor(BG_AX)
ax.grid(True, color=GRID_CLR, lw=0.5)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)

# Plot Real Function
ax.plot(x_fine, y_fine, color=CLR_TRUE, lw=3, label="Función Real", alpha=0.5)

# Plot Uniform Spline
ax.plot(x_fine, spl_unif(x_fine), color=CLR_UNIF, lw=2, label="Nodos Uniformes (Mal ajuste)")
ax.scatter(x_unif, y_unif, color=CLR_UNIF, s=60, edgecolors="white", zorder=5)

# Plot Optimized Spline
ax.plot(x_fine, spl_opt(x_fine), color=CLR_OPT, lw=2, ls="--", label="Nodos Estratégicos (Mejor ajuste)")
ax.scatter(x_opt, y_opt, color=CLR_OPT, s=100, marker='*', edgecolors="white", zorder=6)

ax.set_title("APP9: Impacto de la Posición de los Nodos en Splines", color=ACCENT, fontsize=14)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend(facecolor=BG_AX, edgecolor=SPINE_CLR)

# Residual calculation for console
res_unif = np.mean((y_fine - spl_unif(x_fine))**2)
res_opt  = np.mean((y_fine - spl_opt(x_fine))**2)

print(f"Error Cuadrático Medio (Uniforme):    {res_unif:.6f}")
print(f"Error Cuadrático Medio (Estratégico): {res_opt:.6f}")

plt.show()