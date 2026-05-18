import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# ── Paleta (UASLP Mechatronics Style) ──────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"
CMAP_HEAT = "magma" # Standard for temperature visualization

# ── Datos de APP7 ──────────────────────────────────────────────────────────
# Ejes de la malla
x_coords = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y_coords = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

# Matriz de Temperaturas u(x, y)
# Nota: La tabla tiene x en filas e y en columnas
u_data = np.array([
    [0.0,  5.0,  10.0, 15.0, 20.0, 25.0],  # x = 0.0
    [5.0,  7.51, 10.05, 12.7, 15.67, 20.0], # x = 0.5
    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],  # x = 1.0
    [15.0, 12.51, 9.95, 7.32, 4.33, 0.0],   # x = 1.5
    [20.0, 15.0, 10.0, 5.0,  0.0, -5.0]    # x = 2.0
])

# Puntos a evaluar (x, y)
eval_points = np.array([
    [0.7,  1.2],
    [1.6,  2.4],
    [0.65, 0.82]
])

# ── Cálculo de Interpolación 2D ───────────────────────────────────────────
# Usamos RegularGridInterpolator con método 'cubic' para flujo de calor
interp_func = RegularGridInterpolator((x_coords, y_coords), u_data, method='cubic')
results = interp_func(eval_points)

# ── Salida en Consola ─────────────────────────────────────────────────────
print("\n" + "="*50)
print(f"{'Punto (x, y)':<20} | {'Temp. Estimada u(x, y)':<20}")
print("-" * 50)
for pt, val in zip(eval_points, results):
    print(f"({pt[0]:.2f}, {pt[1]:.2f}){'':<10} | {val:<20.4f}")
print("="*50 + "\n")

# ── Configuración de la Interfaz ───────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig = plt.figure(figsize=(14, 7), facecolor=BG_FIG)
fig.text(0.5, 0.95, "APP7: Interpolación 2D de Flujo de Calor en Estado Estable", 
         ha="center", fontsize=14, color=ACCENT, fontweight='bold')

gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], left=0.07, right=0.95, wspace=0.15)
ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1])

# Plot Izquierdo: Mapa de Calor
ax.set_facecolor(BG_AX)
X, Y = np.meshgrid(y_coords, x_coords) # Invertimos para el plot (y=columnas, x=filas)
im = ax.imshow(u_data, extent=[0, 2.5, 2.0, 0], cmap=CMAP_HEAT, aspect='auto', origin='upper')
plt.colorbar(im, ax=ax, label="Temperatura (u)")

# Marcadores para puntos de evaluación
ax.scatter(eval_points[:, 1], eval_points[:, 0], color=ACCENT, s=100, marker='X', 
           edgecolors="white", label="Puntos Evaluados", zorder=10)

ax.set_xlabel("Eje y (Posición)"); ax.set_ylabel("Eje x (Posición)")
ax.invert_yaxis() # Ajustar para que x=0 esté abajo si se prefiere, o arriba según tabla
ax.legend(facecolor=BG_AX, edgecolor=SPINE_CLR)

# Plot Derecho: Tabla de Resultados Visual
axt.set_axis_off()
y_start = 0.82
axt.text(0.5, 0.92, "Temperaturas Estimadas", ha="center", color=ACCENT, weight='bold', fontsize=12)

# Encabezados de tabla
axt.text(0.15, y_start, "Coordenada (x, y)", color=TXT_SEC, weight='bold')
axt.text(0.60, y_start, "Temp. u(x, y)", color=TXT_SEC, weight='bold')

# Dibujo de filas
for i, (pt, val) in enumerate(zip(eval_points, results)):
    row_y = y_start - 0.12 - (i * 0.12)
    row_color = BG_TABLE if i % 2 == 0 else "#1A1A2E"
    
    rect = plt.Rectangle((0.05, row_y - 0.04), 0.9, 0.1, facecolor=row_color, 
                         alpha=0.6, transform=axt.transAxes)
    axt.add_patch(rect)
    
    axt.text(0.18, row_y, f"({pt[0]:.2f}, {pt[1]:.2f})", color=TXT_PRI, fontsize=11)
    axt.text(0.65, row_y, f"{val:.4f}", color=ACCENT, fontsize=11, weight='bold')

plt.show()