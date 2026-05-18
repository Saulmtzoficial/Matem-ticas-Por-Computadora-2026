"""
Encontrar el cero de y(x) usando Interpolación de Lagrange
Datos: x = [0, 0.5, 1, 1.5, 2, 2.5, 3]
       y = [1.8421, 2.4694, 2.4921, 1.9047, 0.8509, -0.4112, -1.5727]

(a) 3 puntos más cercanos al cero
(b) 4 puntos más cercanos al cero
Estrategia: construir P(x), hallar raíz con bisección sobre el polinomio interpolante.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
from matplotlib.animation import FuncAnimation
from scipy.optimize import brentq

# ── Paleta oscura ──────────────────────────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"

C = {
    "pts":    "#4FC3F7",
    "used3":  "#FFB74D",   # puntos usados en (a)
    "used4":  "#CE93D8",   # puntos usados en (b)
    "l0":     "#FF7043",
    "l1":     "#66BB6A",
    "l2":     "#CE93D8",
    "l3":     "#FFB74D",
    "poly3":  "#FFB74D",
    "poly4":  "#CE93D8",
    "zero":   "#FF5252",
    "vline":  "#546E7A",
    "yaxis":  "#3A3A5C",
}

# ── Datos ──────────────────────────────────────────────────────────────────
xs_all = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
ys_all = np.array([1.8421, 2.4694, 2.4921, 1.9047, 0.8509, -0.4112, -1.5727])

# El cambio de signo ocurre entre x=2 (y=0.8509) y x=2.5 (y=-0.4112)
# Vecinos más cercanos al cero:
#   (a) 3 puntos: índices 3,4,5 → x=[1.5,2.0,2.5]
#   (b) 4 puntos: índices 3,4,5,6 → x=[1.5,2.0,2.5,3.0]
IDX3 = [3, 4, 5]
IDX4 = [3, 4, 5, 6]

xr = np.linspace(1.0, 3.2, 600)

# ── Funciones de interpolación ─────────────────────────────────────────────
def lagrange_subset(idx, x):
    """Polinomio de Lagrange con los puntos dados por idx."""
    xs_s = xs_all[idx]
    ys_s = ys_all[idx]
    x = np.atleast_1d(x).astype(float)
    result = np.zeros_like(x)
    n = len(idx)
    for i in range(n):
        Li = np.ones_like(x)
        for j in range(n):
            if i != j:
                Li *= (x - xs_s[j]) / (xs_s[i] - xs_s[j])
        result += ys_s[i] * Li
    return result

def find_zero(idx):
    """Encuentra el cero del polinomio de Lagrange en [1, 3]."""
    fn = lambda x: float(lagrange_subset(idx, np.array([x]))[0])
    return brentq(fn, 1.8, 2.9)

zero3 = find_zero(IDX3)
zero4 = find_zero(IDX4)

# ── Bases de Lagrange individuales (para animación) ────────────────────────
def Li_contrib(idx, i, x):
    """yi * Li(x) para el i-ésimo punto del subconjunto idx."""
    xs_s = xs_all[idx]
    ys_s = ys_all[idx]
    x = np.atleast_1d(x).astype(float)
    Li = np.ones_like(x)
    for j in range(len(idx)):
        if j != i:
            Li *= (x - xs_s[j]) / (xs_s[i] - xs_s[j])
    return ys_s[i] * Li

# ── Resultados numéricos ───────────────────────────────────────────────────
def calc_table(idx, zero_x):
    xs_s = xs_all[idx]
    ys_s = ys_all[idx]
    n = len(idx)
    Lvals = []
    for i in range(n):
        Lv = float(lagrange_subset([idx[i]] if False else idx,
                   np.array([zero_x]))[0])
        # Calcular Li(zero_x) individualmente
        Li = 1.0
        for j in range(n):
            if j != i:
                Li *= (zero_x - xs_s[j]) / (xs_s[i] - xs_s[j])
        Lvals.append(Li)

    header = ["i", "xᵢ", "yᵢ", "Lᵢ(x*)", "yᵢ·Lᵢ(x*)"]
    data = []
    total = 0.0
    for i in range(n):
        contrib = ys_s[i] * Lvals[i]
        total += contrib
        data.append([
            str(i),
            f"{xs_s[i]:.1f}",
            f"{ys_s[i]:+.4f}",
            f"{Lvals[i]:+.5f}",
            f"{contrib:+.5f}",
        ])
    data.append(["", "", "", "P(x*) →", f"{total:+.5f}"])
    return header, data, total, zero_x

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color":      TXT_PRI,
    "axes.labelcolor": TXT_SEC,
    "xtick.color":     TXT_SEC,
    "ytick.color":     TXT_SEC,
    "font.size":       9,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.975,
         "Cero de y(x) por Interpolación de Lagrange  —  3 y 4 puntos vecinos",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1],
                      left=0.06, right=0.97,
                      bottom=0.20, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje gráfica
ax.set_facecolor(BG_AX)
ax.set_xlim(-0.2, 3.4); ax.set_ylim(-2.5, 3.2)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y(x)", fontsize=11)
for sp in ax.spines.values():
    sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color="#5A5A8A", lw=1.0, zorder=1)
ax.axvline(0, color=SPINE_CLR, lw=0.6, zorder=1)

# Todos los puntos de datos (fondo, gris)
ax.scatter(xs_all, ys_all, zorder=4, s=55, color=TXT_SEC,
           edgecolors=BG_AX, linewidths=1.2)
for xi, yi in zip(xs_all, ys_all):
    ax.annotate(f"({xi:.1f},{yi:.4f})", (xi, yi),
                textcoords="offset points", xytext=(5, 6),
                fontsize=7, color=TXT_SEC)

# Scatter de puntos "activos" (cambian según modo)
scat_active = ax.scatter([], [], zorder=6, s=90, color=C["used3"],
                         edgecolors="white", linewidths=1.5)

# Punto del cero
scat_zero = ax.scatter([], [], zorder=8, s=140, color=C["zero"],
                       edgecolors="white", linewidths=2, marker="*")
vline_zero = ax.axvline(x=0, color=C["zero"], lw=1.0, ls=":", alpha=0, zorder=2)
zero_txt   = ax.text(0, -2.3, "", fontsize=9, color=C["zero"], ha="center")

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

# Líneas animadas (bases + polinomio final)
BASE_COLORS = [C["l0"], C["l1"], C["l2"], C["l3"]]
base_lines = [ax.plot([], [], color=BASE_COLORS[i], lw=1.4, ls="--", alpha=0.85)[0]
              for i in range(4)]
poly_line, = ax.plot([], [], lw=2.6, color=C["poly3"])

all_anim_lines = base_lines + [poly_line]

ax.legend(
    handles=[
        mpatches.Patch(color=C["l0"],   label="Base L₀(x)·y₀"),
        mpatches.Patch(color=C["l1"],   label="Base L₁(x)·y₁"),
        mpatches.Patch(color=C["l2"],   label="Base L₂(x)·y₂"),
        mpatches.Patch(color=C["l3"],   label="Base L₃(x)·y₃"),
        mpatches.Patch(color=C["poly3"],label="P₃(x)  — 3 pts"),
        mpatches.Patch(color=C["poly4"],label="P₄(x)  — 4 pts"),
        mpatches.Patch(color=C["zero"], label="Cero x*"),
    ],
    loc="upper right", fontsize=7.5, framealpha=0.35,
    facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI
)

# Estilo eje tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values():
    sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(mode):
    axt.cla()
    axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values():
        sp.set_color(SPINE_CLR)
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.set_axis_off()

    if mode is None:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara ver resultados",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle()
        return

    idx    = IDX3 if mode == "3pts" else IDX4
    zero_x = zero3 if mode == "3pts" else zero4
    header, data, pval, zx = calc_table(idx, zero_x)

    n_pts  = len(idx)
    title  = (f"Lagrange  {n_pts} puntos  —  x* ≈ {zx:.5f}")
    col_w  = [0.07, 0.12, 0.16, 0.33, 0.32]

    n_data  = len(data)
    total_h = 0.76
    row_h   = total_h / (n_data + 1.2)
    y_start = 0.91

    axt.text(0.5, 0.97, title, ha="center", va="top",
             fontsize=10, color=ACCENT, transform=axt.transAxes)

    def draw_row(cells, y_center, is_header=False, is_last=False):
        bg = "#1A3A5C" if is_header else ("#0A2A50" if is_last else None)
        if bg:
            rect = mpatches.FancyBboxPatch(
                (0.01, y_center - row_h * 0.47), 0.98, row_h * 0.94,
                boxstyle="round,pad=0.004",
                facecolor=bg, edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False
            )
            axt.add_patch(rect)
        x = 0.02
        for cell, cw in zip(cells, col_w):
            clr = TXT_SEC if is_header else (ACCENT if is_last else TXT_PRI)
            fw  = "bold" if is_last else "normal"
            fs  = 7.5 if is_header else (9.0 if is_last else 8.5)
            axt.text(x + cw / 2, y_center, str(cell),
                     ha="center", va="center",
                     fontsize=fs, color=clr, fontweight=fw,
                     transform=axt.transAxes, zorder=2, clip_on=False)
            x += cw

    # Encabezado
    y_h = y_start - row_h * 0.5
    draw_row(header, y_h, is_header=True)
    axt.plot([0.02, 0.98], [y_h - row_h * 0.52] * 2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=3)

    for ri, row in enumerate(data):
        y_r = y_start - row_h * (ri + 1.5)
        draw_row(row, y_r, is_last=(ri == n_data - 1))

    # Caja resultado final
    y_box = y_start - row_h * (n_data + 2.1)
    box = mpatches.FancyBboxPatch(
        (0.05, y_box - 0.035), 0.90, 0.065,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False
    )
    axt.add_patch(box)
    axt.text(0.5, y_box - 0.002,
             f"Cero  x*  =  {zx:.6f}",
             ha="center", va="center", fontsize=11,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Estado ─────────────────────────────────────────────────────────────────
state = {"mode": "3pts", "running": False, "anim": None}
N_FRAMES = 80

def get_phases(mode):
    idx      = IDX3 if mode == "3pts" else IDX4
    poly_clr = C["poly3"] if mode == "3pts" else C["poly4"]
    poly_line.set_color(poly_clr)
    n = len(idx)
    phases = []
    labels = ["L₀(x)·y₀", "L₁(x)·y₁", "L₂(x)·y₂", "L₃(x)·y₃"]
    for i in range(n):
        ii = i  # capture
        phases.append((
            f"Trazando  {labels[i]}",
            base_lines[i],
            (lambda ii: lambda x: Li_contrib(idx, ii, x))(ii)
        ))
    phases.append((
        f"Sumando  →  P{'₃' if mode=='3pts' else '₄'}(x)",
        poly_line,
        lambda x: lagrange_subset(idx, x)
    ))
    return phases, idx

def clear_all():
    for ln in all_anim_lines:
        ln.set_data([], [])
    scat_active.set_offsets(np.empty((0, 2)))
    scat_zero.set_offsets(np.empty((0, 2)))
    vline_zero.set_alpha(0)
    zero_txt.set_text("")
    status_txt.set_text("")
    draw_table(None)

# ── Animación ──────────────────────────────────────────────────────────────
def run_animation(event=None):
    if state["running"]:
        return
    clear_all()
    state["running"] = True
    mode = state["mode"]
    phases, idx = get_phases(mode)
    total = len(phases) * N_FRAMES

    # Mostrar puntos activos
    col = C["used3"] if mode == "3pts" else C["used4"]
    scat_active.set_color(col)
    scat_active.set_offsets(np.column_stack([xs_all[idx], ys_all[idx]]))

    zero_x = zero3 if mode == "3pts" else zero4

    def update(frame):
        ph_idx   = min(frame // N_FRAMES, len(phases) - 1)
        ph_frame = frame % N_FRAMES
        name, line, fn = phases[ph_idx]
        status_txt.set_text(name)
        n_pts = max(2, int((ph_frame + 1) / N_FRAMES * len(xr)))
        xp = xr[:n_pts]
        line.set_data(xp, fn(xp))

        if frame == total - 1:
            scat_zero.set_offsets([[zero_x, 0.0]])
            vline_zero.set_xdata([zero_x, zero_x])
            vline_zero.set_alpha(0.7)
            zero_txt.set_text(f"x* ≈ {zero_x:.5f}")
            zero_txt.set_x(zero_x)
            status_txt.set_text(f"✓  Cero encontrado: x* = {zero_x:.5f}")
            draw_table(mode)
            state["running"] = False
        return (line,)

    state["anim"] = FuncAnimation(
        fig, update, frames=total,
        interval=16, blit=False, repeat=False
    )
    fig.canvas.draw_idle()

def reset_animation(event=None):
    if state["anim"] is not None:
        state["anim"].event_source.stop()
        state["anim"] = None
    state["running"] = False
    clear_all()
    fig.canvas.draw_idle()

def switch_mode(label):
    reset_animation()
    state["mode"] = "3pts" if "3" in label else "4pts"

# ── Widgets ────────────────────────────────────────────────────────────────
ax_play  = fig.add_axes([0.06, 0.09, 0.15, 0.07])
ax_reset = fig.add_axes([0.23, 0.09, 0.15, 0.07])
ax_radio = fig.add_axes([0.44, 0.03, 0.24, 0.14])

for wax in [ax_play, ax_reset, ax_radio]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values():
        sp.set_color(SPINE_CLR)

btn_play  = Button(ax_play,  "▶   Animar",
                   color="#0D2B55", hovercolor="#1A3A6E")
btn_reset = Button(ax_reset, "↺   Reset",
                   color="#1A1A3E", hovercolor="#2A2A5E")
radio     = RadioButtons(ax_radio,
                         ("3 puntos vecinos (a)", "4 puntos vecinos (b)"),
                         activecolor=ACCENT)

for b in [btn_play, btn_reset]:
    b.label.set_color(TXT_PRI)
    b.label.set_fontsize(10)
for lbl in radio.labels:
    lbl.set_color(TXT_PRI)
    lbl.set_fontsize(9.5)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)
radio.on_clicked(switch_mode)

draw_table(None)
plt.show()