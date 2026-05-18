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
# 1. Lineal (Usando solo los primeros dos puntos 85-86)
p_lin2 = np.polyfit(years_all[:2], exp_all[:2], 1)

# 2. Polinomio Cúbico (Usando los primeros cuatro puntos 85-88)
p_cub4 = np.polyfit(years_all[:4], exp_all[:4], 3)

# 3. Regresión Lineal (Todos los puntos 85-95)
p_reg_l = np.polyfit(years_all, exp_all, 1)

# 4. Regresión Cuadrática (Todos los puntos 85-95)
p_reg_q = np.polyfit(years_all, exp_all, 2)

def eval_back(x):
    return {
        "lin2": np.polyval(p_lin2, x),
        "cub4": np.polyval(p_cub4, x),
        "regl": np.polyval(p_reg_l, x),
        "regq": np.polyval(p_reg_q, x)
    }

# ── Configuración de la Interfaz ───────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig = plt.figure(figsize=(15, 8), facecolor=BG_FIG)
fig.text(0.5, 0.96, "APP3: Extrapolación Hacia Atrás (1985 → 1980)", 
         ha="center", fontsize=13, color=ACCENT, fontweight='bold')

gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=0.06, right=0.97, bottom=0.15, top=0.90, wspace=0.1)
ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1])

ax.set_facecolor(BG_AX); ax.set_xlim(1979, 1996); ax.set_ylim(200, 1600)
ax.grid(True, color=GRID_CLR, lw=0.5); ax.tick_params(colors=TXT_SEC)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)

# Puntos de datos y objetivo
ax.scatter(years_all, exp_all, color=ACCENT, s=60, edgecolors="white", zorder=5, label="Datos APP2")
ax.scatter(X_TARGET, Y_ACTUAL, color="#FF5252", s=120, marker="*", zorder=6, label="Valor Real 1980 (492)")
ax.axvline(X_TARGET, color="#FF5252", ls="--", alpha=0.3)

ln_lin2, = ax.plot([], [], color=CLR_LIN2, lw=2, label="Lineal (85-86)")
ln_cub4, = ax.plot([], [], color=CLR_CUB, lw=2, label="Cúbico (85-88)")
ln_regl, = ax.plot([], [], color=CLR_LRS, lw=2, ls="--", label="Reg. Lineal (85-95)")
ln_regq, = ax.plot([], [], color=CLR_QRS, lw=2, ls="--", label="Reg. Cuadrática (85-95)")

ax.legend(facecolor=BG_AX, edgecolor=SPINE_CLR, fontsize=8, loc="upper left")
status_txt = ax.text(0.02, 0.03, "Listo para extrapolar.", transform=ax.transAxes, color=ACCENT, fontsize=10)

# ── Tabla de Resultados ────────────────────────────────────────────────────
def draw_table(done=False):
    axt.cla(); axt.set_axis_off()
    if not done:
        axt.text(0.5, 0.5, "Presiona ▶ Animar\npara ver la proyección hacia 1980", ha="center", color=TXT_SEC)
        return

    res = eval_back(X_TARGET)
    y = 0.95; r_h = 0.08
    axt.text(0.5, y, "Comparación vs. 492 $B (Real)", ha="center", color=ACCENT, weight='bold', fontsize=11)
    
    headers = ["Método", "Est. 1980", "Error Abs", "Error %"]
    cols = [0.05, 0.35, 0.60, 0.80]
    y -= 0.12
    for h, cx in zip(headers, cols): axt.text(cx, y, h, color=TXT_SEC, fontsize=8, weight='bold')

    methods = [
        ("Lineal (2pt)", CLR_LIN2, res['lin2']),
        ("Cúbico (4pt)", CLR_CUB, res['cub4']),
        ("Reg. Lineal", CLR_LRS, res['regl']),
        ("Reg. Cuad.", CLR_QRS, res['regq']),
    ]

    for name, clr, val in methods:
        y -= r_h
        err_abs = abs(val - Y_ACTUAL)
        err_pct = (err_abs / Y_ACTUAL) * 100
        axt.text(cols[0], y, name, color=clr, weight='bold')
        axt.text(cols[1], y, f"{val:.1f}", color=TXT_PRI)
        axt.text(cols[2], y, f"{err_abs:.1f}", color=TXT_PRI)
        axt.text(cols[3], y, f"{err_pct:.1f}%", color="#FF5252" if err_pct > 15 else CLR_QRS)

# ── Animación ──────────────────────────────────────────────────────────────
def update(frame):
    # Rango de visualización de 1985 hacia atrás a 1980
    x_range = np.linspace(1985, 1980, 50)
    idx = (frame % 50) + 1
    
    if frame < 50:
        status_txt.set_text("Extrapolando: Lineal (Puntos 1985-1986)...")
        ln_lin2.set_data(x_range[:idx], [np.polyval(p_lin2, x) for x in x_range[:idx]])
    elif frame < 100:
        status_txt.set_text("Extrapolando: Polinomio Cúbico (Puntos 1985-1988)...")
        idx_sub = (frame-50) + 1
        ln_cub4.set_data(x_range[:idx_sub], [np.polyval(p_cub4, x) for x in x_range[:idx_sub]])
    elif frame < 150:
        status_txt.set_text("Extrapolando: Regresión Lineal Global...")
        idx_sub = (frame-100) + 1
        ln_regl.set_data(x_range[:idx_sub], [np.polyval(p_reg_l, x) for x in x_range[:idx_sub]])
    elif frame < 200:
        status_txt.set_text("Extrapolando: Regresión Cuadrática Global...")
        idx_sub = (frame-150) + 1
        ln_regq.set_data(x_range[:idx_sub], [np.polyval(p_reg_q, x) for x in x_range[:idx_sub]])
    
    if frame == 199:
        draw_table(done=True)
        status_txt.set_text("Extrapolación finalizada.")
    return ln_lin2, ln_cub4, ln_regl, ln_regq

ani_data = {"obj": None}

def start_anim(event):
    if ani_data["obj"] is None:
        ani_data["obj"] = FuncAnimation(fig, update, frames=200, interval=30, repeat=False)
    plt.draw()

def reset(event):
    if ani_data["obj"]: ani_data["obj"].event_source.stop(); ani_data["obj"] = None
    for l in [ln_lin2, ln_cub4, ln_regl, ln_regq]: l.set_data([], [])
    draw_table(done=False); status_txt.set_text("Reset."); plt.draw()

# Botones
ax_play = fig.add_axes([0.06, 0.05, 0.12, 0.055])
ax_reset = fig.add_axes([0.20, 0.05, 0.12, 0.055])
btn_play = Button(ax_play, "▶ Animar", color="#0D47A1", hovercolor="#1565C0")
btn_reset = Button(ax_reset, "↺ Reset", color="#311B92", hovercolor="#4527A0")
btn_play.label.set_color("white"); btn_reset.label.set_color("white")

btn_play.on_clicked(start_anim); btn_reset.on_clicked(reset)

draw_table(done=False)
plt.show()