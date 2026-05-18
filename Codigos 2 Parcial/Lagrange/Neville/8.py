"""
Método de Neville — Ecuación de la cuadrática
Datos: x = [-1, 1, 3],  y = [17, -7, -15]

Tabla de Neville (simbólica):
  Q[0,0] = 17
  Q[1,0] = -7          Q[1,1] = -12x + 5
  Q[2,0] = -15         Q[2,1] = -4x - 3       Q[2,2] = 2x² - 12x + 3

Resultado: P(x) = 2x² − 12x + 3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
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

C = {
    "pts":   "#FFB74D",
    "q0":    TXT_SEC,
    "q1":    "#FF7043",
    "q2":    "#66BB6A",
    "poly":  "#4FC3F7",
    "roots": "#EF5350",
    "vert":  "#CE93D8",
}

# ── Datos ──────────────────────────────────────────────────────────────────
xs = np.array([-1., 1., 3.])
ys = np.array([17., -7., -15.])
n  = len(xs)

# ── Polinomios de Neville (como funciones) ─────────────────────────────────
def Q00(x): return np.full_like(np.atleast_1d(x), 17.0, dtype=float)
def Q10(x): return np.full_like(np.atleast_1d(x), -7.0, dtype=float)
def Q20(x): return np.full_like(np.atleast_1d(x), -15.0, dtype=float)
def Q11(x): return -12*np.atleast_1d(x) + 5          # lineal por x0,x1
def Q21(x): return -4*np.atleast_1d(x) - 3           # lineal por x1,x2
def Q22(x): return 2*np.atleast_1d(x)**2 - 12*np.atleast_1d(x) + 3  # cuadrática

# Raíces y vértice de P(x) = 2x²-12x+3
a_c, b_c, c_c = 2.0, -12.0, 3.0
x_vert = -b_c / (2*a_c)                   # = 3.0
y_vert = Q22(np.array([x_vert]))[0]       # = -15.0
disc   = b_c**2 - 4*a_c*c_c
x_root1 = (-b_c + np.sqrt(disc)) / (2*a_c)   # ≈ 5.7386
x_root2 = (-b_c - np.sqrt(disc)) / (2*a_c)   # ≈ 0.2614

xr = np.linspace(-2.5, 7.5, 600)

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
         "Neville  —  Cuadrática que pasa por 3 puntos",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.18, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje
ax.set_facecolor(BG_AX)
ax.set_xlim(-2.6, 7.6); ax.set_ylim(-22, 28)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("P(x)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.9, zorder=1)
ax.axvline(0, color=SPINE_CLR, lw=0.6, zorder=1)

# Puntos de datos
ax.scatter(xs, ys, zorder=6, s=100, color=C["pts"],
           edgecolors="white", linewidths=1.8)
for xi, yi in zip(xs, ys):
    ax.annotate(f"({xi:.0f}, {yi:.0f})", (xi, yi),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, color=C["pts"])

# Marcadores raíces y vértice (ocultos al inicio)
scat_roots = ax.scatter([], [], zorder=6, s=100, color=C["roots"],
                        edgecolors="white", linewidths=1.5, marker="D")
scat_vert  = ax.scatter([], [], zorder=6, s=110, color=C["vert"],
                        edgecolors="white", linewidths=1.5, marker="^")
txt_r1  = ax.text(0, 0, "", fontsize=8, color=C["roots"])
txt_r2  = ax.text(0, 0, "", fontsize=8, color=C["roots"])
txt_vrt = ax.text(0, 0, "", fontsize=8, color=C["vert"])

# Líneas animadas
line_q11,  = ax.plot([], [], color=C["q1"],  lw=1.6, ls="--", alpha=0.9,
                     label="Q[1,1] = −12x + 5")
line_q21,  = ax.plot([], [], color=C["q2"],  lw=1.6, ls="--", alpha=0.9,
                     label="Q[2,1] = −4x − 3")
line_poly, = ax.plot([], [], color=C["poly"], lw=2.8, zorder=5,
                     label="P(x) = 2x² − 12x + 3")

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

ax.legend(loc="upper right", fontsize=8.5, framealpha=0.35,
          facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(done=False):
    axt.cla()
    axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.set_axis_off()

    if not done:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara construir la tabla",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle()
        return

    # ── Título ────────────────────────────────────────────────────────────
    axt.text(0.5, 0.975, "Tabla de Neville  (simbólica)",
             ha="center", va="top", fontsize=10,
             color=ACCENT, transform=axt.transAxes)

    # ── Tabla simbólica ───────────────────────────────────────────────────
    #   i | xᵢ | Q[i,0] | Q[i,1]       | Q[i,2]
    col_h = ["i", "xᵢ", "Q[i,0]", "Q[i,1]", "Q[i,2]"]
    col_w = [0.06, 0.08, 0.16, 0.32, 0.38]

    sym_table = [
        ["0", "−1", "17",   "—",          "—"],
        ["1", " 1", "−7",   "−12x + 5",   "—"],
        ["2", " 3", "−15",  "−4x − 3",    "2x² − 12x + 3"],
    ]
    row_colors = [
        [None,   None,   TXT_PRI, None,    None],
        [None,   None,   TXT_PRI, C["q1"], None],
        [None,   None,   TXT_PRI, C["q2"], C["poly"]],
    ]

    n_rows  = len(sym_table) + 1
    total_h = 0.32
    row_h   = total_h / (n_rows + 0.3)
    y_start = 0.88

    def draw_row(cells, y_c, is_hdr=False, clrs=None):
        if is_hdr:
            rect = mpatches.FancyBboxPatch(
                (0.01, y_c - row_h*0.47), 0.98, row_h*0.94,
                boxstyle="round,pad=0.003",
                facecolor="#1A3A5C", edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        x = 0.015
        for ci, (cell, cw) in enumerate(zip(cells, col_w)):
            if cell == "—":
                clr = SPINE_CLR
            elif is_hdr:
                clr = TXT_SEC
            elif clrs and clrs[ci]:
                clr = clrs[ci]
                # caja de acento para fórmulas
                if ci >= 3 and cell != "—":
                    box = mpatches.FancyBboxPatch(
                        (x+0.004, y_c-row_h*0.43), cw-0.008, row_h*0.86,
                        boxstyle="round,pad=0.003",
                        facecolor=BG_AX, edgecolor=clr,
                        linewidth=0.8, transform=axt.transAxes,
                        zorder=2, clip_on=False)
                    axt.add_patch(box)
            else:
                clr = TXT_PRI
            fs = 7.0 if is_hdr else (8.5 if ci <= 2 else 8.0)
            fw = "bold" if (ci == 4 and cell != "—" and not is_hdr) else "normal"
            axt.text(x + cw/2, y_c, cell,
                     ha="center", va="center",
                     fontsize=fs, color=clr, fontweight=fw,
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    y_h = y_start - row_h*0.5
    draw_row(col_h, y_h, is_hdr=True)
    axt.plot([0.01, 0.99], [y_h - row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    for ri, (row, clrs) in enumerate(zip(sym_table, row_colors)):
        y_r = y_start - row_h*(ri + 1.5)
        draw_row(row, y_r, clrs=clrs)

    axt.plot([0.01, 0.99], [y_start - row_h*(len(sym_table) + 1.0)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # ── Desarrollo del polinomio ──────────────────────────────────────────
    y_dev = y_start - row_h*(len(sym_table) + 1.6)
    steps = [
        ("Forma de Newton:",
         "0 = Q[0,0]",    TXT_SEC),
        ("Paso 1  →",
         "Q[1,1] = −12x + 5",   C["q1"]),
        ("Paso 2  →",
         "Q[2,2] = 2x² − 12x + 3",  C["poly"]),
    ]
    r_h3 = 0.072
    for ri, (lbl, val, clr) in enumerate(steps):
        y_r = y_dev - r_h3*ri
        axt.text(0.05, y_r, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_SEC,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.95, y_r, val, ha="right", va="center",
                 fontsize=8.5, color=clr, fontweight="normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    # ── Propiedades de la parábola ────────────────────────────────────────
    y_prop = y_dev - r_h3*(len(steps) + 0.6)
    props = [
        ("Vértice:",       f"({x_vert:.1f},  {y_vert:.1f})",  C["vert"]),
        ("Raíces:",        f"x ≈ {x_root2:.4f}  y  x ≈ {x_root1:.4f}", C["roots"]),
        ("Abre hacia:",    "arriba  (a = 2 > 0)",  TXT_PRI),
    ]
    r_h4 = 0.068
    for ri, (lbl, val, clr) in enumerate(props):
        y_r = y_prop - r_h4*ri
        is_key = ri <= 1
        if is_key:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_r - r_h4*0.46), 0.96, r_h4*0.92,
                boxstyle="round,pad=0.003",
                facecolor="#0A2A50", edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.05, y_r, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_SEC,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.95, y_r, val, ha="right", va="center",
                 fontsize=8.5 if is_key else 8,
                 color=clr, fontweight="bold" if is_key else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    # ── Caja final ────────────────────────────────────────────────────────
    y_box = y_prop - r_h4*(len(props) + 0.6)
    box = mpatches.FancyBboxPatch(
        (0.04, y_box - 0.038), 0.92, 0.068,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box - 0.004,
             "P(x)  =  2x²  −  12x  +  3",
             ha="center", va="center", fontsize=11,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Fases de animación ─────────────────────────────────────────────────────
#  Fase 0: mensaje inicial
#  Fase 1: trazar Q[1,1] = -12x+5   (recta por x0,x1)
#  Fase 2: trazar Q[2,1] = -4x-3    (recta por x1,x2)
#  Fase 3: trazar Q[2,2] = parábola final
#  Fase 4: marcar raíces y vértice

N_FRAMES = 80
MSG_FRAMES = 25
state = {"running": False, "anim": None, "phase": 0}

PHASES = [
    ("Columna 0:  Q[i,0] = yᵢ  (valores dados)",
     None, None),
    ("Grado 1:  Q[1,1](x) = −12x + 5  (por x₀, x₁)",
     line_q11, Q11),
    ("Grado 1:  Q[2,1](x) = −4x − 3  (por x₁, x₂)",
     line_q21, Q21),
    ("Grado 2:  Q[2,2](x) = 2x² − 12x + 3  (por x₀, x₁, x₂)",
     line_poly, Q22),
]

def clear_all():
    line_q11.set_data([], [])
    line_q21.set_data([], [])
    line_poly.set_data([], [])
    scat_roots.set_offsets(np.empty((0, 2)))
    scat_vert.set_offsets(np.empty((0, 2)))
    txt_r1.set_text(""); txt_r2.set_text(""); txt_vrt.set_text("")
    status_txt.set_text("")
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]: return
    clear_all()
    state["running"] = True

    # Fases: msg_only + curva + msg_only + curva + msg_only + curva + reveal
    n_curve_phases = len(PHASES)  # incluye la fase 0 (solo msg)
    frames_per_phase = MSG_FRAMES + N_FRAMES
    total = n_curve_phases * frames_per_phase + MSG_FRAMES  # extra para marcar raíces

    def update(frame):
        ph  = frame // frames_per_phase
        loc = frame % frames_per_phase

        if ph < n_curve_phases:
            name, line, fn = PHASES[ph]
            if loc < MSG_FRAMES:
                status_txt.set_text(name)
            else:
                cf = loc - MSG_FRAMES
                if fn is not None:
                    n_pts = max(2, int((cf + 1) / N_FRAMES * len(xr)))
                    line.set_data(xr[:n_pts], fn(xr[:n_pts]))
        else:
            # Última fase: marcar raíces y vértice
            cf = frame - n_curve_phases * frames_per_phase
            if cf == 0:
                scat_roots.set_offsets([[x_root1, 0], [x_root2, 0]])
                scat_vert.set_offsets([[x_vert, y_vert]])
                txt_r1.set_text(f" x≈{x_root1:.3f}")
                txt_r1.set_x(x_root1 + 0.1); txt_r1.set_y(-2.5)
                txt_r2.set_text(f" x≈{x_root2:.3f}")
                txt_r2.set_x(x_root2 + 0.1); txt_r2.set_y(-2.5)
                txt_vrt.set_text(f"  vértice\n  ({x_vert:.0f},{y_vert:.0f})")
                txt_vrt.set_x(x_vert + 0.15); txt_vrt.set_y(y_vert + 1.0)
                status_txt.set_text(
                    f"✓  P(x) = 2x² − 12x + 3   vértice ({x_vert:.0f},{y_vert:.0f})")
            if cf == MSG_FRAMES - 1:
                draw_table(done=True)
                state["running"] = False

        return (line_q11, line_q21, line_poly)

    state["anim"] = FuncAnimation(
        fig, update, frames=total,
        interval=18, blit=False, repeat=False)
    fig.canvas.draw_idle()

def reset_animation(event=None):
    if state["anim"]:
        state["anim"].event_source.stop()
        state["anim"] = None
    state["running"] = False
    clear_all()
    fig.canvas.draw_idle()

# ── Widgets ────────────────────────────────────────────────────────────────
ax_play  = fig.add_axes([0.06, 0.07, 0.16, 0.07])
ax_reset = fig.add_axes([0.24, 0.07, 0.16, 0.07])

for wax in [ax_play, ax_reset]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values(): sp.set_color(SPINE_CLR)

btn_play  = Button(ax_play,  "▶   Animar",
                   color="#0D2B55", hovercolor="#1A3A6E")
btn_reset = Button(ax_reset, "↺   Reset",
                   color="#1A1A3E", hovercolor="#2A2A5E")
for b in [btn_play, btn_reset]:
    b.label.set_color(TXT_PRI); b.label.set_fontsize(10)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)

draw_table(done=False)
plt.show()