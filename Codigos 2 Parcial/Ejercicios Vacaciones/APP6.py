import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ── Paleta (UASLP Mechatronics Style - Fixed) ─────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"  # <── This was the missing line!
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"
CLR_CURVE = "#66BB6A"
CLR_DOTS  = "#FF7043"

# ── Datos de APP6 ──────────────────────────────────────────────────────────
# N: Normalidad | D: Difusividad
ns = np.array([0.0521, 0.1028, 0.2036, 0.4946, 0.9863, 1.9739, 2.443, 5.06])
ds = np.array([1.65, 2.10, 2.27, 2.76, 3.12, 3.06, 2.92, 2.07])
target_ns = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

# ── Cálculo de Interpolación ───────────────────────────────────────────────
spline = CubicSpline(ns, ds, bc_type='natural')
target_ds = spline(target_ns)

# ── Salida en Consola (Backup) ─────────────────────────────────────────────
print("\n" + "="*45)
print(f"{'N (Normalidad)':<18} | {'D x 10^6 (cm²/sec)':<18}")
print("-" * 45)
for n_val, d_val in zip(target_ns, target_ds):
    print(f"{n_val:<18.1f} | {d_val:<18.4f}")
print("="*45 + "\n")

# ── Configuración del Gráfico ──────────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig = plt.figure(figsize=(14, 7), facecolor=BG_FIG)
fig.text(0.5, 0.95, "APP6: Cinética de Elución de Cobre (Resuelta)", 
         ha="center", fontsize=14, color=ACCENT, fontweight='bold')

gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], left=0.07, right=0.95, wspace=0.15)
ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1])

# Plot Izquierdo: Curva de Cinética
ax.set_facecolor(BG_AX)
ax.grid(True, color=GRID_CLR, lw=0.5)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)

x_smooth = np.linspace(0, 5.1, 200)
ax.plot(x_smooth, spline(x_smooth), color=CLR_CURVE, lw=2.5, label="Spline Cúbico Natural", alpha=0.8)
ax.scatter(ns, ds, color=CLR_DOTS, s=55, label="Datos Exp.", zorder=5, edgecolors="white", lw=0.5)
ax.scatter(target_ns, target_ds, color=ACCENT, s=90, marker='X', label="Resultados Requeridos", zorder=6)

ax.set_xlabel("Normalidad (N)")
ax.set_ylabel(r"$D \times 10^6$ (cm²/sec)")
ax.legend(facecolor=BG_AX, edgecolor=SPINE_CLR, fontsize=8)

# Plot Derecho: Tabla Visual
axt.set_axis_off()
y_start = 0.82
axt.text(0.5, 0.92, "Valores de Difusividad (D)", ha="center", color=ACCENT, weight='bold', fontsize=12)

# Encabezados
axt.text(0.15, y_start, "N", color=TXT_SEC, weight='bold', fontsize=10)
axt.text(0.55, y_start, r"$D \times 10^6$", color=TXT_SEC, weight='bold', fontsize=10)

# Dibujo de filas
for i, (nv, dv) in enumerate(zip(target_ns, target_ds)):
    row_y = y_start - 0.1 - (i * 0.1)
    
    # Fondo de fila alternado
    row_color = BG_TABLE if i % 2 == 0 else "#1A1A2E"
    rect = plt.Rectangle((0.05, row_y - 0.035), 0.9, 0.09, facecolor=row_color, 
                         alpha=0.6, transform=axt.transAxes, zorder=1)
    axt.add_patch(rect)
    
    axt.text(0.20, row_y, f"{nv:.1f}", color=TXT_PRI, fontsize=11, zorder=2)
    axt.text(0.60, row_y, f"{dv:.4f}", color=ACCENT, fontsize=11, weight='bold', zorder=2)

plt.show()