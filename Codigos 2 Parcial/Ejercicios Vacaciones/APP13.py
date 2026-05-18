import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# ── Paleta (UASLP Mechatronics Style) ──────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"
CLR_KNOTS = "#FF7043"

# ── Coordenadas Paramétricas (Ajustadas al Dibujo) ────────────────────────
# Definimos los puntos en orden desde la frente hasta el hombro
# X: Posición horizontal | Y: Altura
pts = np.array([
    [3.5, 10.0], # 0: Inicio Frente
    [3.3, 9.0],  # 1: Frente
    [3.1, 8.2],  # 2: Entrecejo
    [2.0, 7.2],  # 3: Punta de la Nariz
    [3.0, 6.5],  # 4: Base de la Nariz
    [2.8, 6.2],  # 5: Labio Superior
    [3.2, 5.9],  # 6: Comisura (Boca)
    [2.9, 5.6],  # 7: Labio Inferior
    [3.4, 4.8],  # 8: Mentón
    [3.8, 3.0],  # 9: Cuello
    [4.5, 0.5],  # 10: Hombro
])

x_k = pts[:, 0]
y_k = pts[:, 1]

# ── Cálculo del Spline Paramétrico ────────────────────────────────────────
# Creamos un parámetro 't' que es simplemente el índice del punto [0, 1, 2...]
t = np.arange(len(pts))
t_fine = np.linspace(0, len(pts)-1, 500)

# Interpolamos X(t) y Y(t) por separado
# Esto permite que la curva haga curvas cerradas sin errores matemáticos
spline_x = CubicSpline(t, x_k, bc_type='natural')
spline_y = CubicSpline(t, y_k, bc_type='natural')

x_fine = spline_x(t_fine)
y_fine = spline_y(t_fine)

# ── Interfaz de Visualización ──────────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.96, "APP13: Reconstrucción Paramétrica de Perfil", 
         ha="center", fontsize=14, color=ACCENT, fontweight='bold')

# Subplots
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], left=0.08, right=0.95, bottom=0.15, top=0.90)
ax = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

ax.set_facecolor(BG_AX); ax.set_xlim(1, 6); ax.set_ylim(-1, 11)
ax.grid(True, color=GRID_CLR, lw=0.5)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)

ln_profile, = ax.plot([], [], color=ACCENT, lw=3, label="Contorno Reconstruido")
sc_knots = ax.scatter([], [], color=CLR_KNOTS, s=0, edgecolors="white", zorder=5)
ax.legend(facecolor=BG_AX, edgecolor=SPINE_CLR)

status_txt = ax.text(0.05, 0.05, "Presiona ▶ para iniciar...", transform=ax.transAxes, color=ACCENT)

# ── Panel Técnico ──
axt.set_axis_off()
def draw_panel():
    axt.cla(); axt.set_axis_off()
    axt.text(0.5, 0.95, "Lógica Paramétrica vs Global", ha="center", color=ACCENT, weight='bold')
    
    txt = [
        "Para formas orgánicas, no usamos y=f(x).",
        "Usamos x=f(t) y y=f(t).",
        "",
        "ESTRATEGIA DE NODOS:",
        "• Labios (P5-P8): Nodos muy cercanos para",
        "  capturar la curvatura del gesto.",
        "• Frente/Hombros: Nodos espaciados.",
        "",
        "RESULTADO:",
        "La spline garantiza continuidad de la",
        "tangente y la curvatura (Clase C2)."
    ]
    y_p = 0.8
    for line in txt:
        axt.text(0.1, y_p, line, color=TXT_PRI if "•" not in line else CLR_KNOTS)
        y_p -= 0.06

# ── Animación ──────────────────────────────────────────────────────────────
def update(frame):
    limit = int((frame / 100) * len(x_fine))
    ln_profile.set_data(x_fine[:limit], y_fine[:limit])
    
    node_limit = int((frame / 100) * len(x_k))
    if node_limit > 0:
        sc_knots.set_offsets(pts[:node_limit])
        sc_knots.set_sizes([50] * node_limit)
    
    if frame == 99: status_txt.set_text("✓ Perfil Paramétrico Finalizado.")
    return ln_profile, sc_knots

ani_data = {"obj": None}
def start_anim(event):
    if ani_data["obj"] is None:
        draw_panel()
        ani_data["obj"] = FuncAnimation(fig, update, frames=100, interval=25, repeat=False)
    plt.draw()

ax_btn = fig.add_axes([0.44, 0.05, 0.12, 0.06])
btn = Button(ax_btn, "▶ Animar", color="#0D47A1", hovercolor="#1565C0")
btn.label.set_color("white"); btn.on_clicked(start_anim)

draw_panel()
plt.show()