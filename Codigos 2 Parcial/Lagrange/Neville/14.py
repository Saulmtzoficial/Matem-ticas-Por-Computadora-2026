"""
Programa General de Interpolación de Neville
Datos: 8 puntos dispersos
x = [-2.0, -0.1, -1.5, 0.5, -0.6, 2.2, 1.0, 1.8]
y = [2.2796, 1.0025, 1.6467, 1.0635, 1.0920, 2.6291, 1.2661, 1.9896]

Evaluar en x = 1.1, 1.2, 1.3
Respuestas: y = 1.3262, 1.3938, 1.4693
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, TextBox, Slider
from matplotlib.animation import FuncAnimation

# ── Paleta ─────────────────────────────────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"
G_COLORS  = ["#FF7043","#66BB6A","#CE93D8","#4FC3F7",
              "#FFB74D","#80CBC4","#F48FB1","#A5D6A7"]

# ── Datos ──────────────────────────────────────────────────────────────────
xs_all = np.array([-2.0, -0.1, -1.5, 0.5, -0.6, 2.2, 1.0, 1.8])
ys_all = np.array([2.2796, 1.0025, 1.6467, 1.0635, 1.0920, 2.6291, 1.2661, 1.9896])
TARGETS  = [1.1, 1.2, 1.3]
EXPECTED = [1.3262, 1.3938, 1.4693]

# ── Núcleo de Neville ──────────────────────────────────────────────────────
def neville_table(xs, ys, x):
    n = len(xs)
    Q = np.zeros((n, n))
    Q[:, 0] = ys.copy()
    for j in range(1, n):
        for i in range(j, n):
            Q[i, j] = ((x - xs[i-j])*Q[i, j-1] -
                       (x - xs[i])*Q[i-1, j-1]) / (xs[i] - xs[i-j])
    return Q

def neville_eval(xs, ys, x):
    return neville_table(xs, ys, x)[-1, -1]

# Precompute full tables for targets
tables = {xv: neville_table(xs_all, ys_all, xv) for xv in TARGETS}

# Interpolating curve (using all 8 pts on a dense grid)
xr = np.linspace(-2.3, 2.5, 500)
yr = np.array([neville_eval(xs_all, ys_all, xi) for xi in xr])

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color": TXT_PRI, "axes.labelcolor": TXT_SEC,
    "xtick.color": TXT_SEC, "ytick.color": TXT_SEC, "font.size": 9,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.975,
         "Programa Neville — Interpolación en múltiples puntos",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1],
                      left=0.06, right=0.97,
                      bottom=0.20, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo gráfica
ax.set_facecolor(BG_AX)
ax.set_xlim(-2.5, 2.7); ax.set_ylim(0.0, 3.2)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y(x)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.6, zorder=1)
ax.axvline(0, color=SPINE_CLR, lw=0.5, zorder=1)

# Puntos de datos
ax.scatter(xs_all, ys_all, zorder=6, s=85, color=ACCENT,
           edgecolors="white", linewidths=1.5)
for xi, yi in zip(xs_all, ys_all):
    off = (5, 7) if yi < 2.5 else (5, -14)
    ax.annotate(f"({xi},{yi})", (xi, yi),
                textcoords="offset points", xytext=off,
                fontsize=7.5, color=ACCENT)

# Curva interpolante (animada)
poly_line, = ax.plot([], [], color="#FFB74D", lw=2.5, zorder=5,
                     label="P(x)  Neville (8 pts)")

# Marcadores de targets
TARGET_COLORS = ["#FF5252", "#CE93D8", "#80CBC4"]
target_dots  = [ax.scatter([], [], s=150, color=tc, zorder=8,
                            edgecolors="white", linewidths=2, marker="*")
                for tc in TARGET_COLORS]
target_vlines = [ax.axvline(xv, color=tc, lw=0.9, ls="--", alpha=0.7, zorder=2)
                 for xv, tc in zip(TARGETS, TARGET_COLORS)]
target_txts  = [ax.text(xv+0.04, 0.12, f"x={xv}", fontsize=8, color=tc)
                for xv, tc in zip(TARGETS, TARGET_COLORS)]
# Inicialmente ocultar los vlines
for vl in target_vlines: vl.set_alpha(0)

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

ax.legend(loc="upper left", fontsize=8.5, framealpha=0.35,
          facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(mode=None):
    axt.cla(); axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
    axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.set_axis_off()

    if mode is None:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\no selecciona un x objetivo",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle(); return

    # mode = target x value
    xv   = mode
    Q    = tables[xv]
    n    = len(xs_all)
    tc   = TARGET_COLORS[TARGETS.index(xv)]
    res  = Q[-1, -1]

    axt.text(0.5, 0.975, f"Tabla Neville  en  x = {xv}",
             ha="center", va="top", fontsize=10,
             color=tc, transform=axt.transAxes)

    # Columnas: xᵢ | Q[i,0] | Q[i,1] | ... | Q[i,7]
    # Solo mostrar hasta grado 4 (diagonal visible)
    MAX_COL = min(n, 5)   # mostrar cols 0..4
    col_labels = ["xᵢ"] + [f"ord {j}" for j in range(MAX_COL)]
    col_w = [0.10] + [0.18]*MAX_COL

    n_rows  = n + 1
    total_h = 0.60
    row_h   = total_h / (n_rows + 0.4)
    y_start = 0.89

    def draw_row(cells, y_c, is_hdr=False, hi_col=-1):
        if is_hdr:
            rect = mpatches.FancyBboxPatch(
                (0.01, y_c - row_h*0.47), 0.98, row_h*0.94,
                boxstyle="round,pad=0.003", facecolor="#1A3A5C",
                edgecolor="none", transform=axt.transAxes,
                zorder=1, clip_on=False)
            axt.add_patch(rect)
        x = 0.015
        for ci, (cell, cw) in enumerate(zip(cells, col_w)):
            val = str(cell)
            if val == "—":
                clr = SPINE_CLR
            elif is_hdr:
                clr = TXT_SEC
            elif ci == hi_col and not is_hdr:
                box = mpatches.FancyBboxPatch(
                    (x+0.004, y_c-row_h*0.42), cw-0.008, row_h*0.84,
                    boxstyle="round,pad=0.002",
                    facecolor="#0D47A1", edgecolor=tc, linewidth=0.8,
                    transform=axt.transAxes, zorder=2, clip_on=False)
                axt.add_patch(box)
                clr = "#FFFFFF"
            elif not is_hdr and ci >= 2:
                clr = G_COLORS[min(ci-1, len(G_COLORS)-1)]
            else:
                clr = TXT_PRI

            fs = 6.8 if is_hdr else 7.8
            fw = "bold" if ci == hi_col and not is_hdr else "normal"
            axt.text(x + cw/2, y_c, val,
                     ha="center", va="center", fontsize=fs,
                     color=clr, fontweight=fw,
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    y_h = y_start - row_h*0.5
    draw_row(col_labels, y_h, is_hdr=True)
    axt.plot([0.01, 0.99], [y_h - row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    for i in range(n):
        y_r = y_start - row_h*(i + 1.5)
        hi_col = MAX_COL if i == n-1 else -1  # resaltar última diagonal visible
        cells = [f"{xs_all[i]:+.1f}"]
        for j in range(MAX_COL):
            if i >= j:
                cells.append(f"{Q[i,j]:.4f}")
            else:
                cells.append("—")
        draw_row(cells, y_r, hi_col=hi_col)

    axt.plot([0.01, 0.99], [y_start - row_h*(n+1.0)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # Bloque resultados de los 3 targets
    y_res = y_start - row_h*(n+1.5)
    r_h2  = 0.068
    axt.text(0.5, y_res, "Resultados de los 3 objetivos",
             ha="center", va="center", fontsize=7.8, color=TXT_SEC,
             transform=axt.transAxes, zorder=3, clip_on=False)
    y_res -= r_h2*0.9

    for ti, (xti, exp, tc2) in enumerate(zip(TARGETS, EXPECTED, TARGET_COLORS)):
        val_c = neville_eval(xs_all, ys_all, xti)
        is_cur = (xti == xv)
        bg = "#0A2A50" if is_cur else None
        if bg:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_res-r_h2*0.46), 0.96, r_h2*0.92,
                boxstyle="round,pad=0.003", facecolor=bg,
                edgecolor="none", transform=axt.transAxes,
                zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.04, y_res, f"y({xti}) =",
                 ha="left", va="center", fontsize=8,
                 color=tc2, transform=axt.transAxes,
                 zorder=3, clip_on=False)
        axt.text(0.60, y_res, f"{val_c:.4f}",
                 ha="left", va="center", fontsize=8.5 if is_cur else 8,
                 color=tc2, fontweight="bold" if is_cur else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.80, y_res, f"({exp:.4f})",
                 ha="left", va="center", fontsize=7.5,
                 color=TXT_SEC, transform=axt.transAxes,
                 zorder=3, clip_on=False)
        y_res -= r_h2

    # Caja final
    y_box = y_res - r_h2*0.4
    box = mpatches.FancyBboxPatch(
        (0.03, y_box-0.038), 0.94, 0.068,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=tc, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box-0.004,
             f"y({xv})  =  {res:.6f}",
             ha="center", va="center", fontsize=11,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
N_FRAMES   = 100
MSG_FRAMES = 25
state = {"running": False, "anim": None, "target_idx": 0}

BUILD = [
    "Cargando 8 puntos de datos...",
    "Construyendo tabla de Neville completa (8×8)...",
    "Trazando polinomio interpolante P(x)...",
    "Evaluando en x = 1.1...",
    "Evaluando en x = 1.2...",
    "Evaluando en x = 1.3...",
]

def clear_all():
    poly_line.set_data([], [])
    for d in target_dots: d.set_offsets(np.empty((0,2)))
    for vl in target_vlines: vl.set_alpha(0)
    status_txt.set_text("")
    draw_table(None)

def run_animation(event=None):
    if state["running"]: return
    clear_all(); state["running"] = True

    n_msg = 3
    n_dot = 3
    total = n_msg*MSG_FRAMES + N_FRAMES + n_dot*MSG_FRAMES

    def update(frame):
        if frame < n_msg*MSG_FRAMES:
            ph = frame // MSG_FRAMES
            status_txt.set_text(BUILD[ph])
        elif frame < n_msg*MSG_FRAMES + N_FRAMES:
            cf = frame - n_msg*MSG_FRAMES
            status_txt.set_text(BUILD[2])
            n_pts = max(2, int((cf+1)/N_FRAMES * len(xr)))
            poly_line.set_data(xr[:n_pts], yr[:n_pts])
        else:
            f2  = frame - n_msg*MSG_FRAMES - N_FRAMES
            idx = f2 // MSG_FRAMES
            if idx < n_dot:
                status_txt.set_text(BUILD[3+idx])
                xv = TARGETS[idx]; yv = neville_eval(xs_all, ys_all, xv)
                target_dots[idx].set_offsets([[xv, yv]])
                target_vlines[idx].set_alpha(0.75)

        if frame == total - 1:
            status_txt.set_text(
                "✓  y(1.1)=1.3262   y(1.2)=1.3938   y(1.3)=1.4693")
            draw_table(TARGETS[0])
            state["running"] = False
        return (poly_line,)

    state["anim"] = FuncAnimation(
        fig, update, frames=total,
        interval=18, blit=False, repeat=False)
    fig.canvas.draw_idle()

def reset_animation(event=None):
    if state["anim"]:
        state["anim"].event_source.stop(); state["anim"] = None
    state["running"] = False
    clear_all(); fig.canvas.draw_idle()

# Botones de target
def show_t1(e): draw_table(TARGETS[0])
def show_t2(e): draw_table(TARGETS[1])
def show_t3(e): draw_table(TARGETS[2])

# ── Widgets ────────────────────────────────────────────────────────────────
ax_play  = fig.add_axes([0.06, 0.075, 0.14, 0.07])
ax_reset = fig.add_axes([0.22, 0.075, 0.11, 0.07])
ax_t1    = fig.add_axes([0.36, 0.075, 0.10, 0.07])
ax_t2    = fig.add_axes([0.48, 0.075, 0.10, 0.07])
ax_t3    = fig.add_axes([0.60, 0.075, 0.10, 0.07])

for wax in [ax_play, ax_reset, ax_t1, ax_t2, ax_t3]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values(): sp.set_color(SPINE_CLR)

btn_play  = Button(ax_play,  "▶  Animar",
                   color="#0D2B55", hovercolor="#1A3A6E")
btn_reset = Button(ax_reset, "↺ Reset",
                   color="#1A1A3E", hovercolor="#2A2A5E")
btn_t1    = Button(ax_t1, "x=1.1",
                   color="#4A1500", hovercolor="#7A2500")
btn_t2    = Button(ax_t2, "x=1.2",
                   color="#260A3A", hovercolor="#4A1570")
btn_t3    = Button(ax_t3, "x=1.3",
                   color="#003A3A", hovercolor="#005555")

for b, tc in zip([btn_t1, btn_t2, btn_t3], TARGET_COLORS):
    b.label.set_color(tc); b.label.set_fontsize(9.5)
for b in [btn_play, btn_reset]:
    b.label.set_color(TXT_PRI); b.label.set_fontsize(10)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)
btn_t1.on_clicked(show_t1)
btn_t2.on_clicked(show_t2)
btn_t3.on_clicked(show_t3)

# Instrucción
fig.text(0.36, 0.015,
         "Botones x=... → ver tabla de Neville para ese objetivo",
         ha="left", va="bottom", fontsize=8, color=TXT_SEC)

draw_table(None)
plt.show()