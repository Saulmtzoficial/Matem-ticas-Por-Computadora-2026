"""
Spline Cúbico Natural — 3 puntos: (0,0), (1,2), (2,1)
Condición natural: S''(0) = S''(2) = 0

Sistema de momentos:  2(h0+h1)·M1 = 6[(y2−y1)/h1 − (y1−y0)/h0]
  → M0=0, M1=−4.5, M2=0

Tramo 0  [0,1]:  S0(x) = −0.75x³ + 2.75x
Tramo 1  [1,2]:  S1(x) =  0.75x³ − 4.5x² + 7.25x − 1.5

Verificaciones en x=1:
  S0(1) = S1(1) = 2         (continuidad de valor)
  S0'(1)= S1'(1)= 0.5       (continuidad de primera derivada)
  S0''(1)= S1''(1)= −4.5    (continuidad de segunda derivada)
  S0''(0)= 0, S1''(2)= 0    (condición natural)
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
    "pts":    "#FFB74D",
    "s0":     "#FF7043",
    "s1":     "#66BB6A",
    "deriv1": "#CE93D8",
    "deriv2": "#80CBC4",
    "join":   "#FF5252",
    "nat":    "#FFF176",
}

# ── Splines ────────────────────────────────────────────────────────────────
# S0(x) = -0.75x³ + 2.75x                 para x in [0, 1]
# S1(x) =  0.75x³ - 4.5x² + 7.25x - 1.5  para x in [1, 2]
def S0(x):
    x = np.atleast_1d(x).astype(float)
    return -0.75*x**3 + 2.75*x

def S1(x):
    x = np.atleast_1d(x).astype(float)
    return 0.75*x**3 - 4.5*x**2 + 7.25*x - 1.5

def S0p(x):  return -2.25*x**2 + 2.75          # primera derivada S0
def S1p(x):  return  2.25*x**2 - 9.0*x + 7.25  # primera derivada S1
def S0pp(x): return -4.5*x                       # segunda derivada S0
def S1pp(x): return  4.5*x - 9.0                # segunda derivada S1

xs_data = np.array([0., 1., 2.])
ys_data = np.array([0., 2., 1.])

xr0 = np.linspace(0, 1, 300)
xr1 = np.linspace(1, 2, 300)
xr_all = np.linspace(-0.1, 2.1, 600)

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
         "Spline Cúbico Natural  —  3 puntos: (0,0)  (1,2)  (2,1)",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.18, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje
ax.set_facecolor(BG_AX)
ax.set_xlim(-0.3, 2.4); ax.set_ylim(-0.5, 2.8)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("S(x)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.8, zorder=1)
ax.axvline(0, color=SPINE_CLR, lw=0.5, zorder=1)

# Línea de unión en x=1
ax.axvline(1, color=SPINE_CLR, lw=0.8, ls=":", zorder=2, alpha=0.6)
ax.text(1.03, -0.38, "x=1", fontsize=8, color=TXT_SEC)

# Puntos de datos
ax.scatter(xs_data, ys_data, zorder=7, s=100, color=C["pts"],
           edgecolors="white", linewidths=1.8)
for xi, yi in zip(xs_data, ys_data):
    ax.annotate(f"({xi:.0f}, {yi:.0f})", (xi, yi),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, color=C["pts"])

# Punto de unión en x=1 (resaltado)
join_dot = ax.scatter([], [], s=140, color=C["join"], zorder=8,
                      edgecolors="white", linewidths=2, marker="o")

# Tangente en x=1
tang_line, = ax.plot([], [], color=C["deriv1"], lw=1.3, ls="-.", alpha=0,
                     zorder=3)

# Líneas animadas
line_s0,  = ax.plot([], [], color=C["s0"],  lw=2.8, zorder=5,
                    label="S₀(x) = −0.75x³ + 2.75x")
line_s1,  = ax.plot([], [], color=C["s1"],  lw=2.8, zorder=5,
                    label="S₁(x) = 0.75x³ − 4.5x² + 7.25x − 1.5")

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
                 "Presiona  ▶  Animar\npara construir el spline",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle()
        return

    # Título
    axt.text(0.5, 0.975, "Derivación del spline natural",
             ha="center", va="top", fontsize=10,
             color=ACCENT, transform=axt.transAxes)

    # ── Sección 1: sistema de momentos ─────────────────────────────────
    y_cur = 0.91
    r_h = 0.055

    def line_txt(y, txt, clr=TXT_PRI, fs=8.0, bold=False, indent=0.04):
        axt.text(indent, y, txt, ha="left", va="center",
                 fontsize=fs, color=clr, fontweight="bold" if bold else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    def section_title(y, txt):
        rect = mpatches.FancyBboxPatch(
            (0.01, y - r_h*0.46), 0.98, r_h*0.92,
            boxstyle="round,pad=0.003",
            facecolor="#1A3A5C", edgecolor="none",
            transform=axt.transAxes, zorder=1, clip_on=False)
        axt.add_patch(rect)
        axt.text(0.5, y, txt, ha="center", va="center",
                 fontsize=8, color=ACCENT, fontweight="normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    section_title(y_cur, "1 · Condición natural  →  M₀ = 0,  M₂ = 0")
    y_cur -= r_h*1.3

    items_sys = [
        ("Sistema (h₀=h₁=1):",          TXT_SEC),
        ("2(h₀+h₁)M₁ = 6[(y₂−y₁)/h₁ − (y₁−y₀)/h₀]", TXT_PRI),
        ("4 M₁  =  6[(1−2)/1 − (2−0)/1]",  TXT_PRI),
        ("4 M₁  =  6[−1 − 2]  =  −18",    TXT_PRI),
        ("M₁  =  −18/4  =  −4.5",          C["deriv2"]),
    ]
    for txt, clr in items_sys:
        line_txt(y_cur, txt, clr)
        y_cur -= r_h

    axt.plot([0.02, 0.98], [y_cur - r_h*0.1]*2,
             color=SPINE_CLR, lw=0.5, ls="--", transform=axt.transAxes, zorder=4)
    y_cur -= r_h*0.5

    # ── Sección 2: coeficientes ─────────────────────────────────────────
    section_title(y_cur, "2 · Coeficientes de cada tramo")
    y_cur -= r_h*1.3

    coeff_data = [
        ("Tramo 0  [0, 1]:", C["s0"], True),
        ("S₀(x) = −0.75x³  +  2.75x",              C["s0"], False),
        ("Tramo 1  [1, 2]:", C["s1"], True),
        ("S₁(x) = 0.75x³ − 4.5x² + 7.25x − 1.5",  C["s1"], False),
    ]
    for txt, clr, bold in coeff_data:
        line_txt(y_cur, txt, clr, bold=bold)
        y_cur -= r_h

    axt.plot([0.02, 0.98], [y_cur - r_h*0.1]*2,
             color=SPINE_CLR, lw=0.5, ls="--", transform=axt.transAxes, zorder=4)
    y_cur -= r_h*0.5

    # ── Sección 3: verificaciones en x=1 ───────────────────────────────
    section_title(y_cur, "3 · Verificaciones en x = 1")
    y_cur -= r_h*1.3

    verif = [
        ("Valor:",       f"S₀(1)= {S0(1.0)[0]:.2f}  =  S₁(1)= {S1(1.0)[0]:.2f}",
         C["pts"]),
        ("S' (1ª der.):", f"S₀'(1)= {S0p(1.0):.4f}  =  S₁'(1)= {S1p(1.0):.4f}",
         C["deriv1"]),
        ("S'' (2ª der.):", f"S₀''(1)= {S0pp(1.0):.2f}  =  S₁''(1)= {S1pp(1.0):.2f}",
         C["deriv2"]),
        ("Nat. S₀''(0):", f"{S0pp(0.0):.2f}",     C["nat"]),
        ("Nat. S₁''(2):", f"{S1pp(2.0):.2f}",     C["nat"]),
    ]
    for lbl, val, clr in verif:
        is_key = "der" in lbl
        if is_key:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_cur - r_h*0.46), 0.96, r_h*0.92,
                boxstyle="round,pad=0.003",
                facecolor="#0A2A50", edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.04, y_cur, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_SEC,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.96, y_cur, val, ha="right", va="center",
                 fontsize=8.0, color=clr, fontweight="bold" if is_key else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)
        y_cur -= r_h

    # ── Caja final ────────────────────────────────────────────────────
    y_box = y_cur - r_h*0.5
    box = mpatches.FancyBboxPatch(
        (0.03, y_box - 0.048), 0.94, 0.082,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box + 0.014,
             "S₀(x) = −0.75x³ + 2.75x",
             ha="center", va="center", fontsize=9.5,
             color=C["s0"], fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)
    axt.text(0.5, y_box - 0.022,
             "S₁(x) = 0.75x³ − 4.5x² + 7.25x − 1.5",
             ha="center", va="center", fontsize=9.5,
             color=C["s1"], fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
BUILD_MSGS = [
    "Condición natural: M₀ = M₂ = 0...",
    "Resolviendo sistema: 4·M₁ = 6[(1−2) − (2−0)]  →  M₁ = −4.5",
    "Coeficientes tramo 0:  S₀(x) = −0.75x³ + 2.75x",
    "Trazando S₀(x) en [0, 1]...",
    "Coeficientes tramo 1:  S₁(x) = 0.75x³ − 4.5x² + 7.25x − 1.5",
    "Trazando S₁(x) en [1, 2]...",
    "Verificando continuidad C² en x = 1...",
]

N_FRAMES   = 90
MSG_FRAMES = 32
state = {"running": False, "anim": None}

def clear_all():
    line_s0.set_data([], [])
    line_s1.set_data([], [])
    join_dot.set_offsets(np.empty((0, 2)))
    tang_line.set_data([], []); tang_line.set_alpha(0)
    status_txt.set_text("")
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]: return
    clear_all()
    state["running"] = True

    # Secuencia:
    #  msg0 → msg1 → [msg2 + curve s0] → [msg3 + curve s1] → msg4 + reveal
    total = 4 * MSG_FRAMES + 2 * N_FRAMES + MSG_FRAMES   # 5 bloques msg + 2 curvas

    def update(frame):
        # Bloque 0: msg "condición natural"
        # Bloque 1: msg "resolver M1"
        # Bloque 2: msg "coeff s0"
        # Bloque 3: curve s0
        # Bloque 4: msg "coeff s1"
        # Bloque 5: curve s1
        # Bloque 6: msg "verificar"

        thresholds = [
            MSG_FRAMES,              # 0 fin de bloque 0
            2*MSG_FRAMES,            # 1 fin bloque 1
            3*MSG_FRAMES,            # 2 fin bloque 2
            3*MSG_FRAMES+N_FRAMES,   # 3 fin curva s0
            4*MSG_FRAMES+N_FRAMES,   # 4 fin bloque 4
            4*MSG_FRAMES+2*N_FRAMES, # 5 fin curva s1
            5*MSG_FRAMES+2*N_FRAMES, # 6 fin bloque 6
        ]

        if frame < thresholds[0]:
            status_txt.set_text(BUILD_MSGS[0])
        elif frame < thresholds[1]:
            status_txt.set_text(BUILD_MSGS[1])
        elif frame < thresholds[2]:
            status_txt.set_text(BUILD_MSGS[2])
        elif frame < thresholds[3]:
            cf = frame - thresholds[2]
            status_txt.set_text(BUILD_MSGS[3])
            n_pts = max(2, int((cf+1)/N_FRAMES * len(xr0)))
            line_s0.set_data(xr0[:n_pts], S0(xr0[:n_pts]))
        elif frame < thresholds[4]:
            status_txt.set_text(BUILD_MSGS[4])
        elif frame < thresholds[5]:
            cf = frame - thresholds[4]
            status_txt.set_text(BUILD_MSGS[5])
            n_pts = max(2, int((cf+1)/N_FRAMES * len(xr1)))
            line_s1.set_data(xr1[:n_pts], S1(xr1[:n_pts]))
        elif frame < thresholds[6]:
            status_txt.set_text(BUILD_MSGS[6])
            # mostrar punto de unión y tangente
            join_dot.set_offsets([[1.0, 2.0]])
            tx = np.array([0.4, 1.6])
            ty = 2.0 + S0p(1.0)*(tx - 1.0)
            tang_line.set_data(tx, ty)
            tang_line.set_alpha(0.75)

        if frame == thresholds[6] - 1:
            status_txt.set_text(
                "✓  S₀'(1) = S₁'(1) = 0.5   |   S₀''(1) = S₁''(1) = −4.5")
            draw_table(done=True)
            state["running"] = False

        return (line_s0, line_s1)

    state["anim"] = FuncAnimation(
        fig, update, frames=thresholds[-1],
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