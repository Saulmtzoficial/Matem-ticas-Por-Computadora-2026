"""
Interpolación de Lagrange y Neville — Animación interactiva (tema oscuro)
Datos: x = [-1.2, 0.3, 1.1], y = [-5.76, -5.61, -3.69]
Estimar y en x = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
from matplotlib.animation import FuncAnimation

# ── Paleta oscura ──────────────────────────────────────────────────────────
BG_FIG   = "#1A1A2E"
BG_AX    = "#16213E"
BG_TABLE = "#0F3460"
GRID_CLR = "#2A2A4A"
SPINE_CLR= "#3A3A5C"
TXT_PRI  = "#E2E2F0"
TXT_SEC  = "#8888AA"
ACCENT   = "#4FC3F7"

C = {
    "pts":   "#4FC3F7",
    "l0":    "#FF7043",
    "l1":    "#66BB6A",
    "l2":    "#CE93D8",
    "poly":  "#4FC3F7",
    "q01":   "#FF7043",
    "q12":   "#66BB6A",
    "q02":   "#4FC3F7",
    "res":   "#FF5252",
    "vline": "#546E7A",
}

# ── Datos ──────────────────────────────────────────────────────────────────
xs = np.array([-1.2, 0.3, 1.1])
ys = np.array([-5.76, -5.61, -3.69])
x_target = 0.0
xr = np.linspace(-2, 2, 400)

# ── Funciones de interpolación ─────────────────────────────────────────────
def L(j, x):
    x = np.atleast_1d(x).astype(float)
    n = np.ones_like(x)
    d = 1.0
    for i in range(3):
        if i != j:
            n *= (x - xs[i])
            d *= (xs[j] - xs[i])
    return n / d

def lagrange(x):
    x = np.atleast_1d(x).astype(float)
    return sum(ys[i] * L(i, x) for i in range(3))

def Q01(x):
    return ((x - xs[0])*ys[1] - (x - xs[1])*ys[0]) / (xs[1] - xs[0])

def Q12(x):
    return ((x - xs[1])*ys[2] - (x - xs[2])*ys[1]) / (xs[2] - xs[1])

def neville(x):
    q01 = Q01(x); q12 = Q12(x)
    return ((x - xs[0])*q12 - (x - xs[2])*q01) / (xs[2] - xs[0])

# ── Resultados numéricos ───────────────────────────────────────────────────
def calc_lag():
    l0v = float(L(0, np.array([x_target]))[0])
    l1v = float(L(1, np.array([x_target]))[0])
    l2v = float(L(2, np.array([x_target]))[0])
    t0, t1, t2 = ys[0]*l0v, ys[1]*l1v, ys[2]*l2v
    pval = t0 + t1 + t2
    header = ["Base",   "Numerador",    "Denominador",  "Lᵢ(0)",        "yᵢ · Lᵢ(0)"]
    data = [
        ["L₀(0)", "(-0.3)(-1.1)", "(-1.5)(-2.3)", f"{l0v:+.5f}", f"{t0:+.5f}"],
        ["L₁(0)", "(1.2)(-1.1)",  "(1.5)(-0.8)",  f"{l1v:+.5f}", f"{t1:+.5f}"],
        ["L₂(0)", "(1.2)(-0.3)",  "(2.3)(0.8)",   f"{l2v:+.5f}", f"{t2:+.5f}"],
        ["P(0)",  "",             "",             "suma →",      f"{pval:+.5f}"],
    ]
    return header, data, pval

def calc_nev():
    q01v = float(Q01(x_target))
    q12v = float(Q12(x_target))
    q02v = float(neville(x_target))
    header = ["Valor",        "xᵢ",   "Resultado",     "Nota"]
    data = [
        ["Q₀,₀",       "-1.2", f"{ys[0]:+.4f}",  "dato"],
        ["Q₁,₀",       " 0.3", f"{ys[1]:+.4f}",  "dato"],
        ["Q₂,₀",       " 1.1", f"{ys[2]:+.4f}",  "dato"],
        ["Q₀,₁(0)",    "",     f"{q01v:+.5f}",   "grado 1"],
        ["Q₁,₁(0)",    "",     f"{q12v:+.5f}",   "grado 1"],
        ["Q₀,₂ = P(0)","",     f"{q02v:+.5f}",   "← final"],
    ]
    return header, data, q02v

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
         "Interpolación polinomial  —  Lagrange & Neville",
         ha="center", va="top", fontsize=13, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1],
                      left=0.06, right=0.97,
                      bottom=0.20, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje gráfica
ax.set_facecolor(BG_AX)
ax.set_xlim(-2, 2); ax.set_ylim(-13, 3)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y", fontsize=11)
for sp in ax.spines.values():
    sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.8, zorder=1)
ax.axvline(x_target, color=C["vline"], lw=1.2, ls="--", zorder=1)
ax.text(x_target + 0.07, -12.2, "x = 0", fontsize=9, color=TXT_SEC)

ax.scatter(xs, ys, zorder=5, s=80, color=C["pts"],
           edgecolors=BG_AX, linewidths=1.5)
for xi, yi in zip(xs, ys):
    ax.annotate(f"({xi}, {yi})", (xi, yi),
                textcoords="offset points", xytext=(7, 6),
                fontsize=8, color=C["pts"])

# Estilo eje tabla (inicial)
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values():
    sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Líneas animadas ────────────────────────────────────────────────────────
line_l0,   = ax.plot([], [], color=C["l0"],  lw=1.5, ls="--", alpha=0.9, label="L₀·y₀ / Q₀,₁")
line_l1,   = ax.plot([], [], color=C["l1"],  lw=1.5, ls="--", alpha=0.9, label="L₁·y₁ / Q₁,₁")
line_l2,   = ax.plot([], [], color=C["l2"],  lw=1.5, ls="--", alpha=0.9, label="L₂·y₂")
line_poly, = ax.plot([], [], color=C["poly"],lw=2.6,                      label="P(x) final")
line_q01,  = ax.plot([], [], color=C["q01"], lw=1.5, ls="--", alpha=0.9)
line_q12,  = ax.plot([], [], color=C["q12"], lw=1.5, ls="--", alpha=0.9)
line_q02,  = ax.plot([], [], color=C["q02"], lw=2.6)

point_res = ax.scatter([], [], s=120, color=C["res"], zorder=7,
                       edgecolors="white", linewidths=1.5, label="P(0)")
status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

all_lines = [line_l0, line_l1, line_l2, line_poly,
             line_q01, line_q12, line_q02]

ax.legend(loc="upper left", fontsize=8, framealpha=0.35,
          facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI)

# ── Tabla de resultados ────────────────────────────────────────────────────
def draw_table(mode):
    """Redibuja el panel derecho con la tabla de resultados."""
    # Limpiar completamente
    axt.cla()
    axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values():
        sp.set_color(SPINE_CLR)
    axt.set_xlim(0, 1)
    axt.set_ylim(0, 1)
    axt.set_axis_off()

    if mode is None:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara ver resultados",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2,
                 transform=axt.transAxes)
        fig.canvas.draw_idle()
        return

    if mode == "lagrange":
        header, data, pval = calc_lag()
        title = f"Lagrange  —  P(0) ≈ {pval:.5f}"
        col_w = [0.12, 0.24, 0.24, 0.21, 0.19]
    else:
        header, data, pval = calc_nev()
        title = f"Neville  —  P(0) ≈ {pval:.5f}"
        col_w = [0.26, 0.12, 0.32, 0.30]

    n_data = len(data)
    total_h = 0.76
    row_h   = total_h / (n_data + 1.2)
    y_start = 0.91

    # Título
    axt.text(0.5, 0.97, title,
             ha="center", va="top", fontsize=10, color=ACCENT,
             transform=axt.transAxes)

    def draw_row(cells, y_center, is_header=False, is_last=False):
        if is_header:
            bg = "#1A3A5C"
        elif is_last:
            bg = "#0A2A50"
        else:
            bg = None
        if bg:
            rect = mpatches.FancyBboxPatch(
                (0.01, y_center - row_h * 0.47),
                0.98, row_h * 0.94,
                boxstyle="round,pad=0.004",
                facecolor=bg, edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False
            )
            axt.add_patch(rect)
        x = 0.02
        for cell, cw in zip(cells, col_w):
            clr = TXT_SEC if is_header else (ACCENT if is_last else TXT_PRI)
            fw  = "bold" if is_last else "normal"
            fs  = 7.5 if is_header else (9.5 if is_last else 8.5)
            axt.text(x + cw / 2, y_center, str(cell),
                     ha="center", va="center",
                     fontsize=fs, color=clr, fontweight=fw,
                     transform=axt.transAxes, zorder=2, clip_on=False)
            x += cw

    # Encabezado
    y_h = y_start - row_h * 0.5
    draw_row(header, y_h, is_header=True)

    # Línea divisoria bajo encabezado
    axt.plot([0.02, 0.98], [y_h - row_h * 0.52] * 2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=3)

    # Filas de datos
    for ri, row in enumerate(data):
        y_r = y_start - row_h * (ri + 1.5)
        draw_row(row, y_r, is_last=(ri == n_data - 1))

    # Caja resultado final
    y_box = y_start - row_h * (n_data + 2.1)
    box = mpatches.FancyBboxPatch(
        (0.06, y_box - 0.033), 0.88, 0.062,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False
    )
    axt.add_patch(box)
    axt.text(0.5, y_box - 0.002,
             f"P(0)  =  {pval:.6f}",
             ha="center", va="center", fontsize=11,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Estado ─────────────────────────────────────────────────────────────────
state = {"mode": "lagrange", "running": False, "anim": None}
N_FRAMES = 80

PHASES_LAG = [
    ("Trazando  L₀(x) · y₀",   line_l0,   lambda x: ys[0] * L(0, x)),
    ("Trazando  L₁(x) · y₁",   line_l1,   lambda x: ys[1] * L(1, x)),
    ("Trazando  L₂(x) · y₂",   line_l2,   lambda x: ys[2] * L(2, x)),
    ("Sumando curvas  →  P(x)", line_poly, lagrange),
]
PHASES_NEV = [
    ("Trazando  Q₀,₁(x)",    line_q01, Q01),
    ("Trazando  Q₁,₁(x)",    line_q12, Q12),
    ("Curva final  Q₀,₂(x)", line_q02, neville),
]

def clear_all():
    for ln in all_lines:
        ln.set_data([], [])
    point_res.set_offsets(np.empty((0, 2)))
    status_txt.set_text("")
    draw_table(None)

# ── Animación ──────────────────────────────────────────────────────────────
def run_animation(event=None):
    if state["running"]:
        return
    clear_all()
    state["running"] = True
    mode   = state["mode"]
    phases = PHASES_LAG if mode == "lagrange" else PHASES_NEV
    total  = len(phases) * N_FRAMES

    def update(frame):
        ph_idx   = min(frame // N_FRAMES, len(phases) - 1)
        ph_frame = frame % N_FRAMES
        name, line, fn = phases[ph_idx]
        status_txt.set_text(name)
        n_pts = max(2, int((ph_frame + 1) / N_FRAMES * len(xr)))
        xp = xr[:n_pts]
        line.set_data(xp, fn(xp))

        if frame == total - 1:
            y_res = float(np.atleast_1d(fn(np.array([x_target])))[0])
            point_res.set_offsets([[x_target, y_res]])
            status_txt.set_text(f"✓  P(0) = {y_res:.5f}")
            draw_table(mode)          # <-- tabla aparece al terminar
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
    state["mode"] = "lagrange" if "Lagrange" in label else "neville"

# ── Widgets ────────────────────────────────────────────────────────────────
ax_play  = fig.add_axes([0.06, 0.09, 0.15, 0.07])
ax_reset = fig.add_axes([0.23, 0.09, 0.15, 0.07])
ax_radio = fig.add_axes([0.44, 0.03, 0.20, 0.14])

for wax in [ax_play, ax_reset, ax_radio]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values():
        sp.set_color(SPINE_CLR)

btn_play  = Button(ax_play,  "▶   Animar",
                   color="#0D2B55", hovercolor="#1A3A6E")
btn_reset = Button(ax_reset, "↺   Reset",
                   color="#1A1A3E", hovercolor="#2A2A5E")
radio     = RadioButtons(ax_radio, ("Lagrange", "Neville"),
                         activecolor=ACCENT)

for b in [btn_play, btn_reset]:
    b.label.set_color(TXT_PRI)
    b.label.set_fontsize(10)
for lbl in radio.labels:
    lbl.set_color(TXT_PRI)
    lbl.set_fontsize(10)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)
radio.on_clicked(switch_mode)

draw_table(None)
plt.show()