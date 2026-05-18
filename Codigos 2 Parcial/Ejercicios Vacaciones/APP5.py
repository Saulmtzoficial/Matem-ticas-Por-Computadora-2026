import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from scipy.interpolate import CubicSpline

# ── Paleta (UASLP Mechatronics Style) ──────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"

CLR_LIN   = "#FF7043" # Linear
CLR_QUAD  = "#CE93D8" # Quadratic
CLR_POLY  = "#4FC3F7" # Cubic/Higher
CLR_SPLI  = "#66BB6A" # Spline

# ── Datos de APP5 ──────────────────────────────────────────────────────────
# Position (in)
xs = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0])
# Dosage (10^5 rads/hr)
ys = np.array([1.90, 2.39, 2.71, 2.98, 3.20, 3.20, 2.98, 2.74])

X_TARGET = 2.5

# ── Modelos ───────────────────────────────────────────────────────────────
# 1. Linear (Using nearest neighbors 2.0 and 3.0)
p_lin = np.polyfit(xs[4:6], ys[4:6], 1)
# 2. Quadratic (Using 1.5, 2.0, 3.0)
p_qua = np.polyfit(xs[3:6], ys[3:6], 2)
# 3. Cubic (Using 1.5, 2.0, 3.0, 3.5)
p_cub = np.polyfit(xs[3:7], ys[3:7], 3)
# 4. Cubic Spline
spline = CubicSpline(xs, ys, bc_type='natural')

def eval_dosage(x):
    return {
        "lin":  np.polyval(p_lin, x),
        "qua":  np.polyval(p_qua, x),
        "cub":  np.polyval(p_cub, x),
        "spli": float(spline(x))
    }

# ── Interfaz ───────────────────────────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig = plt.figure(figsize=(15, 8), facecolor=BG_FIG)
fig.text(0.5, 0.96, "APP5: Estimación de Dosis de Radiación en x = 2.5 in.", 
         ha="center", fontsize=13, color=ACCENT, fontweight='bold')

gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=0.06, right=0.97, bottom=0.15, top=0.90, wspace=0.1)
ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1])

ax.set_facecolor(BG_AX); ax.set_xlim(-0.2, 4.2); ax.set_ylim(1.5, 3.5)
ax.grid(True, color=GRID_CLR, lw=0.5); ax.tick_params(colors=TXT_SEC)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)

ax.scatter(xs, ys, color=ACCENT, s=70, edgecolors="white", zorder=5, label="Lecturas Registradas")
ax.axvline(X_TARGET, color="#FF5252", ls="--", alpha=0.3, label="Punto Faltante (2.5)")

ln_lin, = ax.plot([], [], color=CLR_LIN,  lw=1.5, label="Lineal (Vecinos)")
ln_qua, = ax.plot([], [], color=CLR_QUAD, lw=2, label="Quadrática (Local)")
ln_cub, = ax.plot([], [], color=CLR_POLY, lw=2, label="Cúbica (Sétrica)")
ln_spl, = ax.plot([], [], color=CLR_SPLI, lw=1.5, ls="--", label="Spline Cúbico")

ax.legend(facecolor=BG_AX, edgecolor=SPINE_CLR, fontsize=8, loc="lower center")
status_txt = ax.text(0.02, 0.03, "Analizando curvatura de dosis...", transform=ax.transAxes, color=ACCENT)

def draw_table(done=False):
    axt.cla(); axt.set_axis_off()
    if not done: return
    res = eval_dosage(X_TARGET)
    y = 0.95; r_h = 0.08
    axt.text(0.5, y, "Resultados para x = 2.5", ha="center", color=ACCENT, weight='bold', fontsize=11)
    y -= 0.12
    for h, cx in zip(["Grado/Método", "Dosis Est. (10^5 rads/hr)"], [0.05, 0.55]):
        axt.text(cx, y, h, color=TXT_SEC, fontsize=8, weight='bold')

    methods = [("Lineal (Deg 1)", CLR_LIN, res['lin']), ("Quadrática (Deg 2)", CLR_QUAD, res['qua']),
               ("Cúbica (Deg 3)", CLR_POLY, res['cub']), ("Cubic Spline", CLR_SPLI, res['spli'])]

    for name, clr, val in methods:
        y -= r_h
        axt.text(0.05, y, name, color=clr, weight='bold')
        axt.text(0.55, y, f"{val:.4f}", color=TXT_PRI)
    
    # Conclusion box
    rect = mpatches.FancyBboxPatch((0.05, 0.15), 0.9, 0.2, boxstyle="round,pad=0.02", 
                                  facecolor="#0A2A50", edgecolor=ACCENT, transform=axt.transAxes)
    axt.add_patch(rect)
    axt.text(0.5, 0.28, "MEJOR ESTIMACIÓN", ha="center", color=ACCENT, weight='bold', transform=axt.transAxes)
    axt.text(0.5, 0.20, f"≈ {res['qua']:.3f} x 10^5 rads/hr", ha="center", color=TXT_PRI, fontsize=12, transform=axt.transAxes)

def update(frame):
    x_range = np.linspace(0, 4, 100)
    idx = (frame % 50) * 2
    if frame < 50:
        ln_lin.set_data(x_range[:idx], [np.polyval(p_lin, x) for x in x_range[:idx]])
    elif frame < 100:
        ln_qua.set_data(x_range[:idx], [np.polyval(p_qua, x) for x in x_range[:idx]])
    elif frame < 150:
        ln_cub.set_data(x_range[:idx], [np.polyval(p_cub, x) for x in x_range[:idx]])
    elif frame < 200:
        ln_spl.set_data(x_range[:idx], [spline(x) for x in x_range[:idx]])
    if frame == 199: draw_table(done=True); status_txt.set_text("Cálculo finalizado.")
    return ln_lin, ln_qua, ln_cub, ln_spl

ani_data = {"obj": None}
def start_anim(event):
    if ani_data["obj"] is None:
        ani_data["obj"] = FuncAnimation(fig, update, frames=200, interval=20, repeat=False)
    plt.draw()

ax_play = fig.add_axes([0.1, 0.05, 0.12, 0.055])
btn_play = Button(ax_play, "▶ Animar", color="#0D47A1", hovercolor="#1565C0")
btn_play.label.set_color("white"); btn_play.on_clicked(start_anim)

draw_table(done=False)
plt.show()