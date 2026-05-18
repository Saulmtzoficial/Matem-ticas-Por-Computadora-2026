"""
Interpolación de Diferencias Finitas: Newton Forward, Backward y Stirling
Solución al IndexError: Se agregaron columnas suficientes para n=6 puntos.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# ── Paleta (Estilo Dark) ───────────────────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"

CLR_FWD   = "#FF7043" # Naranja: Forward
CLR_BWD   = "#CE93D8" # Morado: Backward
CLR_STR   = "#66BB6A" # Verde: Stirling

# ── Datos ──────────────────────────────────────────────────────────────────
xs = np.array([0., 1., 2., 3., 4., 5.])
ys = np.array([1., 4., 9., 16., 25., 36.]) # y = (x+1)^2
n = len(xs)
h = xs[1] - xs[0]
X_EVAL = 2.5

# Tabla de Diferencias
diff_table = np.zeros((n, n))
diff_table[:, 0] = ys
for j in range(1, n):
    for i in range(n - j):
        diff_table[i, j] = diff_table[i+1, j-1] - diff_table[i, j-1]

# ── Funciones de Interpolación ─────────────────────────────────────────────
def get_newton_fwd(x_val):
    p = (x_val - xs[0]) / h
    res = diff_table[0, 0]
    term = 1.0
    for i in range(1, n):
        term *= (p - (i - 1)) / i
        res += term * diff_table[0, i]
    return res

def get_newton_bwd(x_val):
    p = (x_val - xs[-1]) / h
    res = diff_table[n-1, 0]
    term = 1.0
    for i in range(1, n):
        term *= (p + (i - 1)) / i
        res += term * diff_table[n-1-i, i]
    return res

def get_stirling(x_val):
    x0_idx = 2 # Centrado en x=2
    p = (x_val - xs[x0_idx]) / h
    res = diff_table[x0_idx, 0]
    # Término 1: p * mean(delta_y)
    t1 = p * (diff_table[x0_idx, 1] + diff_table[x0_idx-1, 1]) / 2
    # Término 2: p^2/2 * delta2_y
    t2 = (p**2) * diff_table[x0_idx-1, 2] / 2
    return res + t1 + t2

# ── Configuración Figura ───────────────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig = plt.figure(figsize=(15, 8), facecolor=BG_FIG)
fig.text(0.5, 0.96, "Análisis de Trayectorias: Newton vs Stirling", 
         ha="center", fontsize=14, color=ACCENT, fontweight='bold')

gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], left=0.06, right=0.97, bottom=0.15, top=0.90, wspace=0.12)
ax = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

ax.set_facecolor(BG_AX)
ax.set_xlim(-0.5, 5.5); ax.set_ylim(0, 45)
ax.grid(True, color=GRID_CLR, lw=0.5, zorder=0)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)

ax.scatter(xs, ys, color=ACCENT, s=100, zorder=5, edgecolors="white", linewidths=1.5)
ax.axvline(X_EVAL, color="#FF5252", ls="--", alpha=0.5, label=f"Eval x={X_EVAL}")

ln_fwd, = ax.plot([], [], color=CLR_FWD, lw=2.5, label="Newton Forward", zorder=4)
ln_bwd, = ax.plot([], [], color=CLR_BWD, lw=2.5, ls="--", label="Newton Backward", zorder=3)
ln_str, = ax.plot([], [], color=CLR_STR, lw=2.5, ls=":", label="Stirling", zorder=2)

ax.legend(facecolor=BG_AX, edgecolor=SPINE_CLR, loc="upper left")
status_txt = ax.text(0.02, 0.95, "Listo para animar.", transform=ax.transAxes, color=ACCENT)

# ── Dibujo de Tabla (Corregido) ────────────────────────────────────────────
def draw_table_ui(highlight=None):
    axt.cla(); axt.set_axis_off()
    
    # Headers y Columnas (7 columnas necesarias para x, y, y 5 diferencias)
    headers = ["x", "y", "Δy", "Δ²y", "Δ³y", "Δ⁴y", "Δ⁵y"]
    cols = [0.02, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84]
    
    for h_txt, cx in zip(headers, cols):
        axt.text(cx, 0.92, h_txt, color=ACCENT, fontsize=9, fontweight='bold')
    
    for i in range(n):
        y_pos = 0.82 - i*0.12
        axt.text(cols[0], y_pos, f"{xs[i]:.0f}", color=TXT_SEC)
        
        for j in range(n - i):
            val = diff_table[i, j]
            clr = TXT_PRI
            
            # Resaltado de Caminos
            if highlight == "fwd" and i == 0: 
                clr = CLR_FWD
            elif highlight == "bwd" and (i + j) == (n - 1): 
                clr = CLR_BWD
            elif highlight == "str":
                # Resalta el patrón de Stirling centrado en x=2
                if (j == 0 and i == 2) or (j == 1 and i in [1, 2]) or (j == 2 and i == 1):
                    clr = CLR_STR
            
            # j+1 siempre será < len(cols) ahora
            axt.text(cols[j+1], y_pos - (j*0.06), f"{val:.0f}", color=clr, 
                     fontweight='bold' if clr != TXT_PRI else 'normal')

# ── Lógica de Animación ────────────────────────────────────────────────────
ani_data = {"obj": None}

def update(frame):
    x_range = np.linspace(0, 5, 100)
    step = frame % 150
    
    if step < 50: # Newton Forward
        status_txt.set_text("Caminando por la parte superior (Newton Forward)")
        draw_table_ui(highlight="fwd")
        idx = int((step/50)*100)
        ln_fwd.set_data(x_range[:idx], [get_newton_fwd(xv) for xv in x_range[:idx]])
    elif step < 100: # Newton Backward
        status_txt.set_text("Subiendo desde la base (Newton Backward)")
        draw_table_ui(highlight="bwd")
        idx = int(((step-50)/50)*100)
        ln_bwd.set_data(x_range[:idx], [get_newton_bwd(xv) for xv in x_range[:idx]])
    else: # Stirling
        status_txt.set_text("Promediando el centro (Stirling)")
        draw_table_ui(highlight="str")
        idx = int(((step-100)/50)*100)
        ln_str.set_data(x_range[:idx], [get_stirling(xv) for xv in x_range[:idx]])
    return ln_fwd, ln_bwd, ln_str

def start_anim(event):
    if ani_data["obj"] is None:
        ani_data["obj"] = FuncAnimation(fig, update, frames=450, interval=35, blit=False)
    plt.draw()

def reset_all(event):
    if ani_data["obj"]:
        ani_data["obj"].event_source.stop()
        ani_data["obj"] = None
    for ln in [ln_fwd, ln_bwd, ln_str]: ln.set_data([], [])
    draw_table_ui()
    plt.draw()

# Botones UI
ax_play = fig.add_axes([0.1, 0.04, 0.12, 0.06])
ax_reset = fig.add_axes([0.25, 0.04, 0.12, 0.06])
btn_play = Button(ax_play, "▶ Animar", color="#0D47A1", hovercolor="#1565C0")
btn_reset = Button(ax_reset, "↺ Reset", color="#311B92", hovercolor="#4527A0")
btn_play.label.set_color("white"); btn_reset.label.set_color("white")

btn_play.on_clicked(start_anim)
btn_reset.on_clicked(reset_all)

draw_table_ui()
plt.show()