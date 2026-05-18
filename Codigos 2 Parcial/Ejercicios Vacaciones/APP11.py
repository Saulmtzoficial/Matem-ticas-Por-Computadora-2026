import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator, CubicSpline

# ── Datos de APP11 ─────────────────────────────────────────────────────────
t_pts = np.array([-1.0, -0.96, -0.86, -0.79, 0.22, 0.5, 0.930])
y_pts = np.array([-1.0, -0.151, 0.894, 0.986, 0.895, 0.5, -0.306])

# ── Interpolación ──────────────────────────────────────────────────────────
# 1. Polinomio de Lagrange (Grado 6 para 7 puntos)
poly_lagrange = BarycentricInterpolator(t_pts, y_pts)

# 2. Spline Cúbico (Segmentos de grado 3 con suavidad continua)
spline_cubic = CubicSpline(t_pts, y_pts, bc_type='natural')

# ── Generación de Curvas para Graficar ─────────────────────────────────────
t_fine = np.linspace(-1.0, 1.0, 500)
y_lagrange = poly_lagrange(t_fine)
y_spline = spline_cubic(t_fine)

# ── Visualización ──────────────────────────────────────────────────────────
# Paleta de colores UASLP Mechatronics (Dark Mode)
BG_FIG = "#1A1A2E"; BG_AX = "#16213E"; ACCENT = "#4FC3F7"

plt.rcParams.update({"text.color": "white", "axes.labelcolor": "lightgray", "font.size": 10})
fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_FIG)
ax.set_facecolor(BG_AX)

ax.plot(t_fine, y_spline, color=ACCENT, lw=2.5, label="Spline Cúbico (Suave)")
ax.plot(t_fine, y_lagrange, color="#FF7043", lw=1.5, ls="--", label="Polinomio Lagrange (Oscilatorio)")
ax.scatter(t_pts, y_pts, color="white", s=60, edgecolors="black", zorder=5, label="Datos Reales")

ax.set_title("APP11: Comparativa de Métodos de Interpolación", color=ACCENT, fontsize=14)
ax.set_xlabel("Tiempo (t)"); ax.set_ylabel("Respuesta (y)")
ax.set_ylim(-1.5, 1.5); ax.grid(True, color="#2A2A4A")
ax.legend(facecolor=BG_AX, edgecolor="#3A3A5C")

plt.show()