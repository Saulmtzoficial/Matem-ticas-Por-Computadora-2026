import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline
import matplotlib.cm as cm

# ── Paleta (UASLP Mechatronics Style) ──────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_PANES  = "#0F3460"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"
NODES_CLR = "#FF7043"

# ── Datos de APP7 ──────────────────────────────────────────────────────────
x_coords = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y_coords = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

u_matrix = np.array([
    [0.0,  5.00, 10.00, 15.00, 20.00, 25.00],
    [5.00, 7.51, 10.05, 12.70, 15.67, 20.00],
    [10.00,10.00,10.00, 10.00, 10.00, 10.00],
    [15.00,12.51, 9.95, 7.32,  4.33,  0.0],
    [20.00,15.00,10.00, 5.00,  0.0,  -5.0]
])

eval_points = np.array([[0.7, 1.2], [1.6, 2.4], [0.65, 0.82]])

# ── Interpolación Bicúbica ───────────────────────────────────────────────
spline = RectBivariateSpline(x_coords, y_coords, u_matrix, kx=3, ky=3)
N_FINE = 100
xi_fine = np.linspace(x_coords[0], x_coords[-1], N_FINE)
yi_fine = np.linspace(y_coords[0], y_coords[-1], N_FINE)
Z_mesh = spline(xi_fine, yi_fine)
Y_mesh, X_mesh = np.meshgrid(yi_fine, xi_fine)
z_evals = spline(eval_points[:, 0], eval_points[:, 1], grid=False)

# ── Configuración y Gráfico 3D ─────────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.96, "Superficie de Temperatura Bicúbica (Corregida)", 
         ha="center", fontsize=15, color=ACCENT, fontweight='bold')

ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor(BG_FIG)

# --- CORRECCIÓN DE ATRIBUTOS (xaxis en lugar de w_xaxis) ---
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_tick_params(colors=TXT_SEC)
    # Configurar color de los paneles (paredes del cubo)
    axis.set_pane_color((15/255, 52/255, 96/255, 0.7)) 
    # Configurar color de las líneas de los ejes
    axis.line.set_color(SPINE_CLR)

# --- Graficar Superficie ---
surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap=cm.magma, 
                       edgecolor='none', alpha=0.85, antialiased=True)

cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
cbar.set_label("Temp u(x, y)", color=TXT_SEC)

# --- Nodos Originales ---
Y_orig, X_orig = np.meshgrid(y_coords, x_coords)
ax.scatter(X_orig.ravel(), Y_orig.ravel(), u_matrix.ravel(), 
           color=NODES_CLR, s=40, marker='o', edgecolors="black", label="Nodos Originales")

# --- Puntos Evaluados ---
ax.scatter(eval_points[:, 0], eval_points[:, 1], z_evals, 
           color=ACCENT, s=150, marker='X', edgecolors="white", linewidths=1.5, zorder=10, label="Resultados")

# Líneas de referencia y etiquetas
for i, pt in enumerate(eval_points):
    ax.plot([pt[0], pt[0]], [pt[1], pt[1]], [z_evals[i], -5], color=ACCENT, ls="--", lw=1, alpha=0.6)
    ax.text(pt[0], pt[1], z_evals[i] + 1.5, f"{z_evals[i]:.2f}", 
            color="white", fontsize=10, fontweight='bold', ha='center')

ax.set_xlabel("Eje x"); ax.set_ylabel("Eje y"); ax.set_zlabel("Temp u")
ax.legend(facecolor=BG_FIG, edgecolor=SPINE_CLR, fontsize=9)
ax.view_init(elev=28, azim=-55)

plt.show()