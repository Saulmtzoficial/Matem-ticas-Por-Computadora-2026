import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ── Paleta (UASLP Dark Mode) ──────────────────────────────────────────────
BG_FIG    = "#1A1A2E"; BG_AX     = "#16213E"
TXT_PRI   = "#E2E2F0"; TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"; CLR_DATA2  = "#FF7043"

# ── Datos del Problema (APP10) ─────────────────────────────────────────────
# Set 1: Nodos para construir el spline
# Nota: Para 'periodic' en Scipy, y[0] debe ser igual a y[-1]. 
# Como el periodo es 120, asumimos el cierre del ciclo.
x1 = np.array([-110, -80, -40, -10, 30, 80, 110, 130]) # Agregamos un punto para cerrar el ciclo
y1 = np.array([7.98, 8.95, 10.71, 11.70, 10.01, 8.23, 7.86, 7.98]) 

# Set 2: Observaciones para validación
x_test = np.array([-100, -60, -20, 20, 60, 100])
y_actual = np.array([8.37, 9.40, 11.39, 10.84, 8.53, 7.89])

# ── Construcción del Spline Periódico ──────────────────────────────────────
# bc_type='periodic' obliga a que S'(a)=S'(b) y S''(a)=S''(b)
star_spline = CubicSpline(x1, y1, bc_type='periodic')

# ── Evaluación y Errores ───────────────────────────────────────────────────
y_pred = star_spline(x_test)
errors = np.abs(y_pred - y_actual)

# ── Visualización ──────────────────────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig = plt.figure(figsize=(14, 7), facecolor=BG_FIG)
fig.text(0.5, 0.95, "APP10: Curva de Luz de Estrella Variable S (Spline Periódico)", 
         ha="center", fontsize=14, color=ACCENT, fontweight='bold')

ax = fig.add_subplot(1, 2, 1)
ax.set_facecolor(BG_AX)

x_fine = np.linspace(-120, 120, 500)
ax.plot(x_fine, star_spline(x_fine), color=ACCENT, lw=2, label="Spline Periódico")
ax.scatter(x1[:-1], y1[:-1], color=ACCENT, s=50, label="Set 1 (Nodos)", zorder=5)
ax.scatter(x_test, y_actual, color=CLR_DATA2, s=50, marker='X', label="Set 2 (Validación)", zorder=5)

ax.set_xlabel("Phase"); ax.set_ylabel("Magnitude")
ax.invert_yaxis() # Magnitud astronómica: menor valor es más brillo
ax.legend(facecolor=BG_AX, edgecolor="#3A3A5C")

# Tabla de Errores
axt = fig.add_subplot(1, 2, 2)
axt.set_axis_off()
y_start = 0.85
axt.text(0.5, 0.95, "Validación vs. Segundo Set", ha="center", color=ACCENT, weight='bold', fontsize=12)

for i, (p, act, pre, err) in enumerate(zip(x_test, y_actual, y_pred, errors)):
    row_y = y_start - (i * 0.1)
    axt.text(0.1, row_y, f"Fase {p}:", color=TXT_SEC)
    axt.text(0.4, row_y, f"Real: {act}", color=TXT_PRI)
    axt.text(0.65, row_y, f"Spline: {pre:.2f}", color=ACCENT)
    axt.text(0.85, row_y, f"Δ: {err:.3f}", color=CLR_DATA2)

plt.show()