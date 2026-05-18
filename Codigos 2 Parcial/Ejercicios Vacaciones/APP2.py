import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

# ── Paleta (UASLP Engineering Style) ──────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"

CLR_POLY  = "#FF7043" # Cubic Poly
CLR_LINE  = "#4FC3F7" # LS Line
CLR_QUAD  = "#CE93D8" # LS Quad
CLR_SPLI  = "#66BB6A" # Spline

# ── Datos ──────────────────────────────────────────────────────────────────
# Full Table for Reference
years_all = np.array([1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995])
exp_all   = np.array([731, 782, 833, 886, 956, 1049, 1159, 1267, 1367, 1436, 1505])

# Training set (1991-1994)
mask      = (years_all >= 1991) & (years_all <= 1994)
x_train   = years_all[mask]
y_train   = exp_all[mask]

# Evaluation Points
X_1995, Y_1995_ACT = 1995, 1505
X_2000 = 2000

# ── Modelos ───────────────────────────────────────────────────────────────
# a. Cubic Poly
p_cub = np.polyfit(x_train, y_train, 3)
# b. Least-Squares Line
p_lin = np.polyfit(x_train, y_train, 1)
# c. Least-Squares Quadratic
p_qua = np.polyfit(x_train, y_train, 2)
# d. Cubic Spline (Natural)
spline = CubicSpline(x_train, y_train, bc_type='natural')

def eval_models(x):
    return {
        "poly": np.polyval(p_cub, x),
        "line": np.polyval(p_lin, x),
        "quad": np.polyval(p_qua, x),
        "spli": spline(x)
    }

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({"text.color":TXT_PRI, "axes.labelcolor":TXT_SEC, "font.size":9})
fig = plt.figure(figsize=(15, 8), facecolor=BG_FIG)
fig.text(0.5, 0.96, "Análisis de Gastos de Bienestar (APP2) — Proyecciones 1995-2000", 
         ha="center", fontsize=13, color=ACCENT, fontweight='bold')

gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=0.06, right=0.97, bottom=0.15, top=0.90, wspace=0.1)
ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1])

ax.set_facecolor(BG_AX); ax.set_xlim(1990, 2001); ax.set_ylim(1100, 2600)
ax.grid(True, color=GRID_CLR, lw=0.5); ax.tick_params(colors=TXT_SEC)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)

# Plot actual data points
ax.scatter(years_all, exp_all, color=TXT_SEC, s=40, alpha=0.4, label="Datos Históricos")
ax.scatter(x_train, y_train, color=ACCENT, s=80, edgecolors="white", zorder=5, label="Set de Entrenamiento (91-94)")
ax.axvline(X_1995, color="#FF5252", ls="--", alpha=0.3)
ax.axvline(X_2000, color="#FF5252", ls="--", alpha=0.3)

ln_poly, = ax.plot([], [], color=CLR_POLY, lw=2, label="Polinomio Cúbico")
ln_line, = ax.plot([], [], color=CLR_LINE, lw=2, label="Línea Mín. Cuadrados")
ln_quad, = ax.plot([], [], color=CLR_QUAD, lw=2, label="Cuadrática Mín. Cuadrados")
ln_spli, = ax.plot([], [], color=CLR_SPLI, lw=2, ls="--", label="Spline Cúbico")

ax.legend(facecolor=BG_AX, edgecolor=SPINE_CLR, fontsize=8, loc="upper left")
status_txt = ax.text(0.02, 0.03, "", transform=ax.transAxes, color=ACCENT, fontsize=10)

# ── Tabla de Resultados ────────────────────────────────────────────────────
def draw_table(done=False):
    axt.cla(); axt.set_axis_off()
    if not done:
        axt.text(0.5, 0.5, "Presiona ▶ Animar\npara procesar modelos", ha="center", color=TXT_SEC)
        return

    results_95 = eval_models(X_1995)
    results_00 = eval_models(X_2000)

    y = 0.95; r_h = 0.07
    axt.text(0.5, y, "Resultados de Proyección", ha="center", color=ACCENT, weight='bold', fontsize=11)
    
    headers = ["Método", "Est. 1995", "Error %", "Proj. 2000"]
    cols = [0.05, 0.35, 0.60, 0.80]
    y -= 0.1
    for h, cx in zip(headers, cols): axt.text(cx, y, h, color=TXT_SEC, fontsize=8, weight='bold')

    methods = [
        ("Cúbico", CLR_POLY, results_95['poly'], results_00['poly']),
        ("Línea MC", CLR_LINE, results_95['line'], results_00['line']),
        ("Cuad. MC", CLR_QUAD, results_95['quad'], results_00['quad']),
        ("Spline", CLR_SPLI, results_95['spli'], results_00['spli']),
    ]

    for name, clr, v95, v00 in methods:
        y -= r_h
        err = abs(v95 - Y_1995_ACT) / Y_1995_ACT * 100
        axt.text(cols[0], y, name, color=clr, weight='bold')
        axt.text(cols[1], y, f"{v95:.1f}", color=TXT_PRI)
        axt.text(cols[2], y, f"{err:.2f}%", color="#FF5252" if err > 2 else CLR_SPLI)
        axt.text(cols[3], y, f"{v00:.1f}", color=TXT_PRI)
    
    # Footnote
    axt.text(0.05, 0.1, f"* Valor Real 1995: {Y_1995_ACT} $B", color=TXT_SEC, style='italic')

# ── Animación ──────────────────────────────────────────────────────────────
def update(frame):
    x_range = np.linspace(1991, 2000, 100)
    idx = (frame % 50) * 2
    
    if frame < 50:
        status_txt.set_text("Calculando Polinomio Cúbico...")
        ln_poly.set_data(x_range[:idx], [np.polyval(p_cub, x) for x in x_range[:idx]])
    elif frame < 100:
        status_txt.set_text("Calculando Línea de Mínimos Cuadrados...")
        ln_line.set_data(x_range[:idx], [np.polyval(p_lin, x) for x in x_range[:idx]])
    elif frame < 150:
        status_txt.set_text("Calculando Cuadrática de Mínimos Cuadrados...")
        ln_quad.set_data(x_range[:idx], [np.polyval(p_qua, x) for x in x_range[:idx]])
    elif frame < 200:
        status_txt.set_text("Calculando Spline Cúbico Natural...")
        ln_spli.set_data(x_range[:idx], [spline(x) for x in x_range[:idx]])
    
    if frame == 199:
        draw_table(done=True)
        status_txt.set_text("Proceso completado.")
    return ln_poly, ln_line, ln_quad, ln_spli

ani_data = {"obj": None}

def start_anim(event):
    if ani_data["obj"] is None:
        ani_data["obj"] = FuncAnimation(fig, update, frames=200, interval=25, repeat=False)
    plt.draw()

def reset(event):
    if ani_data["obj"]: ani_data["obj"].event_source.stop(); ani_data["obj"] = None
    for l in [ln_poly, ln_line, ln_quad, ln_spli]: l.set_data([], [])
    draw_table(done=False); status_txt.set_text(""); plt.draw()

# Botones
ax_play = fig.add_axes([0.06, 0.05, 0.12, 0.055])
ax_reset = fig.add_axes([0.20, 0.05, 0.12, 0.055])
btn_play = Button(ax_play, "▶ Animar", color="#0D47A1", hovercolor="#1565C0")
btn_reset = Button(ax_reset, "↺ Reset", color="#311B92", hovercolor="#4527A0")
btn_play.label.set_color("white"); btn_reset.label.set_color("white")

btn_play.on_clicked(start_anim); btn_reset.on_clicked(reset)

draw_table(done=False)
plt.show()