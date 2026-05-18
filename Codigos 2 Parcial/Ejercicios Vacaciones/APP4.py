import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# ── Paleta (UASLP Engineering Style) ──────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"

CLR_LIN2  = "#FF7043" # Linear (2 pts)
CLR_CUB   = "#CE93D8" # Cubic Poly (4 pts)
CLR_LRS   = "#4FC3F7" # Linear Regression (All)
CLR_QRS   = "#66BB6A" # Quadratic Regression (All)

# ── Datos de APP2 ──────────────────────────────────────────────────────────
years_all = np.array([1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995])
exp_all   = np.array([731, 782, 833, 886, 956, 1049, 1159, 1267, 1367, 1436, 1505])

X_TARGET = 1980
Y_ACTUAL = 492

# ── Modelos de Extrapolación Hacia Atrás ───────────────────────────────────
p_lin2  = np.polyfit(years_all[:2], exp_all[:2], 1)  # Linear (85-86)
p_cub4  = np.polyfit(years_all[:4], exp_all[:4], 3)  # Cubic (85-88)
p_reg_l = np.polyfit(years_all, exp_all, 1)          # Linear Regression (All)
p_reg_q = np.polyfit(years_all, exp_all, 2)          # Quadratic Regression (All)

def eval_back(x):
    return {
        "lin2": np.polyval(p_lin2, x),
        "cub4": np.polyval(p_cub4, x),
        "regl": np.polyval(p_reg_l, x),
        "regq": np.polyval(p_reg_q, x)
    }

# ── Interfaz ───────────────────────────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig = plt.figure(figsize=(15, 8), facecolor=BG_FIG)
fig.text(0.5, 0.96, "APP4: Extrapolación Hacia Atrás (1985 → 1980)", 
         ha="center", fontsize=13, color=ACCENT, fontweight='bold')

gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=0.06, right=0.97, bottom=0.15, top=0.90, wspace=0.1)
ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1])

ax.set_facecolor(BG_AX); ax.set_xlim(1979, 1996); ax.set_ylim(200, 1600)
ax.grid(True, color=GRID_CLR, lw=0.5); ax.tick_params(colors=TXT_SEC)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)

ax.scatter(years_all, exp_all, color=ACCENT, s=60, edgecolors="white", zorder=5, label="Datos APP2")
ax.scatter(X_TARGET, Y_ACTUAL, color="#FF5252", s=120, marker="*", zorder=6, label="Valor Real 1980 (492)")

ln_lin2, = ax.plot([], [], color=CLR_LIN2, lw=2, label="Lineal (85-86)")
ln_cub4, = ax.plot([], [], color=CLR_CUB, lw=2, label="Cúbico (85-88)")
ln_regl, = ax.plot([], [], color=CLR_LRS, lw=2, ls="--", label="Reg. Lineal (85-95)")
ln_regq, = ax.plot([], [], color=CLR_QRS, lw=2, ls="--", label="Reg. Cuadrática (85-95)")

ax.legend(facecolor=BG_AX, edgecolor=SPINE_CLR, fontsize=8, loc="upper left")
status_txt = ax.text(0.02, 0.03, "Presiona Animar para retroceder en el tiempo...", transform=ax.transAxes, color=ACCENT)

def draw_table(done=False):
    axt.cla(); axt.set_axis_off()
    if not done: return
    res = eval_back(X_TARGET)
    y = 0.95; r_h = 0.08
    axt.text(0.5, y, "Resultados vs. Objetivo (492 $B)", ha="center", color=ACCENT, weight='bold', fontsize=11)
    y -= 0.12
    cols = [0.05, 0.40, 0.70]
    for h, cx in zip(["Método", "Est. 1980", "Error %"], cols):
        axt.text(cx, y, h, color=TXT_SEC, fontsize=8, weight='bold')

    methods = [("Lineal (2pt)", CLR_LIN2, res['lin2']), ("Cúbico (4pt)", CLR_CUB, res['cub4']),
               ("Reg. Lineal", CLR_LRS, res['regl']), ("Reg. Cuad.", CLR_QRS, res['regq'])]

    for name, clr, val in methods:
        y -= r_h
        err = abs(val - Y_ACTUAL) / Y_ACTUAL * 100
        axt.text(cols[0], y, name, color=clr, weight='bold')
        axt.text(cols[1], y, f"{val:.1f}", color=TXT_PRI)
        axt.text(cols[2], y, f"{err:.1f}%", color="#FF5252" if err > 10 else CLR_QRS)

def update(frame):
    x_range = np.linspace(1985, 1980, 50)
    idx = (frame % 50) + 1
    if frame < 50:
        ln_lin2.set_data(x_range[:idx], [np.polyval(p_lin2, x) for x in x_range[:idx]])
    elif frame < 100:
        ln_cub4.set_data(x_range[:idx], [np.polyval(p_cub4, x) for x in x_range[:idx]])
    elif frame < 150:
        ln_regl.set_data(x_range[:idx], [np.polyval(p_reg_l, x) for x in x_range[:idx]])
    elif frame < 200:
        ln_regq.set_data(x_range[:idx], [np.polyval(p_reg_q, x) for x in x_range[:idx]])
    if frame == 199: draw_table(done=True)
    return ln_lin2, ln_cub4, ln_regl, ln_regq

ani_data = {"obj": None}
def start_anim(event):
    if ani_data["obj"] is None:
        ani_data["obj"] = FuncAnimation(fig, update, frames=200, interval=25, repeat=False)
    plt.draw()

ax_play = fig.add_axes([0.1, 0.05, 0.12, 0.055])
btn_play = Button(ax_play, "▶ Animar", color="#0D47A1", hovercolor="#1565C0")
btn_play.label.set_color("white"); btn_play.on_clicked(start_anim)

draw_table(done=False)
plt.show()