"""
Densidad relativa del aire ρ vs altitud h — Interpolación de Neville
Datos: h(km) = [0, 1.525, 3.050, 4.575, 6.10, 7.625, 9.150]
       ρ     = [1, 0.8617, 0.7385, 0.6292, 0.5328, 0.4481, 0.3741]

Objetivo: ρ en h = 10.5 km  (EXTRAPOLACIÓN — más allá del rango)

Métodos:
  Neville 7 pts (todos):  ρ(10.5) ≈ 0.31778
  Neville 4 pts vecinos:  ρ(10.5) ≈ 0.31672
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
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

C_GLOB  = "#FFB74D"
C_4PT   = "#66BB6A"
C_EXTR  = "#FF5252"
C_WARN  = "#FFF176"
G_COLORS= ["#FF7043","#66BB6A","#CE93D8","#4FC3F7"]

# ── Datos ──────────────────────────────────────────────────────────────────
hs   = np.array([0., 1.525, 3.050, 4.575, 6.10, 7.625, 9.150])
rhos = np.array([1., 0.8617, 0.7385, 0.6292, 0.5328, 0.4481, 0.3741])
n    = len(hs)
H_TARGET = 10.5

# ── Neville ────────────────────────────────────────────────────────────────
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

Q_full   = neville_table(hs, rhos, H_TARGET)
RES_GLOB = Q_full[-1, -1]   # 0.317776

IDX4     = np.array([3, 4, 5, 6])
XS4, YS4 = hs[IDX4], rhos[IDX4]
Q4       = neville_table(XS4, YS4, H_TARGET)
RES_4PT  = Q4[-1, -1]       # 0.316718

# Curvas suaves
hr_inside  = np.linspace(0, 9.15, 500)
hr_extrap  = np.linspace(9.15, 11.0, 100)

def eval_glob(h_arr):
    return np.array([neville_eval(hs, rhos, hv) for hv in np.atleast_1d(h_arr)])

def eval_4pt(h_arr):
    return np.array([neville_eval(XS4, YS4, hv) for hv in np.atleast_1d(h_arr)])

rho_inside_glob = eval_glob(hr_inside)
rho_extrap_glob = eval_glob(hr_extrap)
rho_inside_4pt  = eval_4pt(hr_inside[250:])  # solo desde ~4.5 km
rho_extrap_4pt  = eval_4pt(hr_extrap)

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color": TXT_PRI, "axes.labelcolor": TXT_SEC,
    "xtick.color": TXT_SEC, "ytick.color": TXT_SEC, "font.size": 9,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.975,
         "Densidad relativa del aire  ρ(h)  —  Extrapolación Neville en h = 10.5 km",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.20, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje
ax.set_facecolor(BG_AX)
ax.set_xlim(-0.3, 11.5); ax.set_ylim(0.20, 1.12)
ax.set_xlabel("h  (km)", fontsize=11)
ax.set_ylabel("ρ  (densidad relativa)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.5, zorder=1)

# Zona de extrapolación (fondo sombreado)
ax.axvspan(9.15, 11.5, color="#3A1A1A", alpha=0.5, zorder=0)
ax.text(9.5, 1.08, "Extrapolación", fontsize=8.5, color=C_WARN, style="italic")
ax.axvline(9.15, color=C_WARN, lw=0.8, ls="--", alpha=0.7, zorder=2)
ax.text(9.2, 0.26, "h_max\n9.15 km", fontsize=7.5, color=C_WARN, ha="left")

# Línea de target
ax.axvline(H_TARGET, color=C_EXTR, lw=1.1, ls="--", alpha=0.85, zorder=2)
ax.text(H_TARGET+0.1, 0.26, f"h={H_TARGET}", fontsize=8.5, color=C_EXTR)

# Puntos de datos
ax.scatter(hs, rhos, zorder=7, s=90, color=ACCENT,
           edgecolors="white", linewidths=1.8)
ax.scatter(hs[IDX4], rhos[IDX4], zorder=8, s=60, color=C_4PT,
           edgecolors="white", linewidths=1.2, marker="D",
           label="4 vecinos más cercanos")
for hi, ri in zip(hs, rhos):
    ax.annotate(f"({hi:.3f},{ri:.4f})", (hi, ri),
                textcoords="offset points", xytext=(5, 7),
                fontsize=7, color=ACCENT)

# Curvas animadas
glob_in,  = ax.plot([], [], color=C_GLOB, lw=2.5, zorder=5,
                    label="Neville 7 pts (interpolación)")
glob_ex,  = ax.plot([], [], color=C_GLOB, lw=2.2, zorder=5,
                    ls="--", alpha=0.9)
pt4_in,   = ax.plot([], [], color=C_4PT, lw=2.0, zorder=4,
                    label="Neville 4 pts vecinos")
pt4_ex,   = ax.plot([], [], color=C_4PT, lw=2.0, zorder=4,
                    ls="--", alpha=0.9)

# Puntos de evaluación
dot_glob  = ax.scatter([], [], s=160, color=C_GLOB, zorder=9,
                        edgecolors="white", linewidths=2, marker="*")
dot_4pt   = ax.scatter([], [], s=120, color=C_4PT, zorder=9,
                        edgecolors="white", linewidths=2, marker="*")
txt_glob  = ax.text(H_TARGET+0.1, 0, "", fontsize=9, color=C_GLOB)
txt_4pt   = ax.text(H_TARGET+0.1, 0, "", fontsize=9, color=C_4PT)

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

ax.legend(loc="upper right", fontsize=8.5, framealpha=0.35,
          facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(mode="4pt"):
    axt.cla(); axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
    axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.set_axis_off()

    if mode is None:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara ver la extrapolación",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle(); return

    is_4pt = (mode == "4pt")
    Q      = Q4 if is_4pt else Q_full
    xs_use = XS4 if is_4pt else hs
    ys_use = YS4 if is_4pt else rhos
    res    = RES_4PT if is_4pt else RES_GLOB
    n_use  = len(xs_use)
    clr    = C_4PT if is_4pt else C_GLOB
    title  = f"Neville {'4 pts' if is_4pt else '7 pts'}  —  h = {H_TARGET} km"

    axt.text(0.5, 0.975, title, ha="center", va="top",
             fontsize=9.5, color=clr, transform=axt.transAxes)

    # Advertencia de extrapolación
    rect = mpatches.FancyBboxPatch(
        (0.02, 0.89), 0.96, 0.052,
        boxstyle="round,pad=0.004", facecolor="#3A2000",
        edgecolor=C_WARN, linewidth=0.8,
        transform=axt.transAxes, zorder=1, clip_on=False)
    axt.add_patch(rect)
    axt.text(0.5, 0.916,
             f"⚠  Extrapolación  —  h=10.5 km > h_max=9.15 km",
             ha="center", va="center", fontsize=7.8,
             color=C_WARN, transform=axt.transAxes, zorder=3, clip_on=False)

    max_col = min(n_use, 5)
    col_labels = ["hᵢ (km)", "ρᵢ"] + [f"ord {j}" for j in range(1, max_col)]
    col_w = [0.16, 0.16] + [0.17]*(max_col - 1)

    total_h = 0.38
    row_h   = total_h / (n_use + 1.2)
    y_start = 0.86

    def draw_row(cells, y_c, is_hdr=False, hi_col=-1):
        if is_hdr:
            rect = mpatches.FancyBboxPatch(
                (0.01, y_c-row_h*0.47), 0.98, row_h*0.94,
                boxstyle="round,pad=0.003", facecolor="#1A3A5C",
                edgecolor="none", transform=axt.transAxes,
                zorder=1, clip_on=False)
            axt.add_patch(rect)
        x = 0.015
        for ci, (cell, cw) in enumerate(zip(cells, col_w)):
            val = str(cell)
            if val == "—": c = SPINE_CLR
            elif is_hdr:   c = TXT_SEC
            elif ci == hi_col:
                box = mpatches.FancyBboxPatch(
                    (x+0.004, y_c-row_h*0.42), cw-0.008, row_h*0.84,
                    boxstyle="round,pad=0.002",
                    facecolor="#0D47A1", edgecolor=clr, linewidth=0.8,
                    transform=axt.transAxes, zorder=2, clip_on=False)
                axt.add_patch(box); c = "#FFFFFF"
            elif not is_hdr and ci >= 2:
                c = G_COLORS[min(ci-2, len(G_COLORS)-1)]
            else: c = TXT_PRI
            fs = 6.5 if is_hdr else 7.8
            fw = "bold" if ci == hi_col and not is_hdr else "normal"
            axt.text(x+cw/2, y_c, val, ha="center", va="center",
                     fontsize=fs, color=c, fontweight=fw,
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    y_h = y_start - row_h*0.5
    draw_row(col_labels, y_h, is_hdr=True)
    axt.plot([0.01, 0.99], [y_h-row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    for i in range(n_use):
        y_r = y_start - row_h*(i+1.5)
        hi  = max_col if i == n_use-1 else -1
        cells = [f"{xs_use[i]:.3f}", f"{ys_use[i]:.4f}"]
        for j in range(1, max_col):
            cells.append(f"{Q[i,j]:.5f}" if i >= j else "—")
        draw_row(cells, y_r, hi_col=hi)

    axt.plot([0.01, 0.99], [y_start-row_h*(n_use+1.0)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # Comparación
    y_c = y_start - row_h*(n_use+1.6)
    r_h2 = 0.068
    items = [
        ("Neville 4 pts:", f"{RES_4PT:.6f}", C_4PT),
        ("Neville 7 pts:", f"{RES_GLOB:.6f}", C_GLOB),
        ("Δ diferencia:", f"{abs(RES_4PT-RES_GLOB):.6f}", TXT_SEC),
    ]
    for lbl, val, c in items:
        is_k = lbl.startswith("Neville")
        if is_k:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_c-r_h2*0.46), 0.96, r_h2*0.92,
                boxstyle="round,pad=0.003", facecolor="#0A2A50",
                edgecolor="none", transform=axt.transAxes,
                zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.04, y_c, lbl, ha="left", va="center",
                 fontsize=8, color=TXT_SEC,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.96, y_c, val, ha="right", va="center",
                 fontsize=9 if is_k else 8, color=c,
                 fontweight="bold" if is_k else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)
        y_c -= r_h2

    # Caja final
    y_box = y_c - r_h2*0.5
    box = mpatches.FancyBboxPatch(
        (0.03, y_box-0.052), 0.94, 0.088,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box+0.016,
             f"ρ(10.5 km) ≈ {RES_4PT:.5f}  [4 pts]",
             ha="center", va="center", fontsize=10,
             color=C_4PT, fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)
    axt.text(0.5, y_box-0.022,
             f"ρ(10.5 km) ≈ {RES_GLOB:.5f}  [7 pts]",
             ha="center", va="center", fontsize=10,
             color=C_GLOB, fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
N_FRAMES   = 85
MSG_FRAMES = 28
state = {"running": False, "anim": None}

BUILD = [
    "7 puntos con espaciado uniforme  Δh = 1.525 km...",
    "Advertencia: h=10.5 km está FUERA del rango [0, 9.15]...",
    "Trazando Neville 7 pts (interpolación + extrapolación)...",
    "Trazando Neville 4 pts vecinos (extrapolación)...",
    f"Evaluando en h = {H_TARGET} km...",
]

def clear_all():
    for ln in [glob_in, glob_ex, pt4_in, pt4_ex]:
        ln.set_data([], [])
    dot_glob.set_offsets(np.empty((0,2)))
    dot_4pt.set_offsets(np.empty((0,2)))
    txt_glob.set_text(""); txt_4pt.set_text("")
    status_txt.set_text("")
    draw_table(None)

def run_animation(event=None):
    if state["running"]: return
    clear_all(); state["running"] = True

    bk = [MSG_FRAMES, 2*MSG_FRAMES,
          2*MSG_FRAMES+N_FRAMES, 2*MSG_FRAMES+2*N_FRAMES,
          2*MSG_FRAMES+2*N_FRAMES+MSG_FRAMES]

    def update(frame):
        if frame < bk[0]:
            status_txt.set_text(BUILD[0])
        elif frame < bk[1]:
            status_txt.set_text(BUILD[1])
        elif frame < bk[2]:
            status_txt.set_text(BUILD[2])
            cf = frame - bk[1]
            f1 = int((cf+1)/N_FRAMES * len(hr_inside))
            f2 = int((cf+1)/N_FRAMES * len(hr_extrap))
            glob_in.set_data(hr_inside[:f1], rho_inside_glob[:f1])
            glob_ex.set_data(hr_extrap[:f2], rho_extrap_glob[:f2])
        elif frame < bk[3]:
            status_txt.set_text(BUILD[3])
            cf = frame - bk[2]
            f1 = int((cf+1)/N_FRAMES * len(hr_inside[250:]))
            f2 = int((cf+1)/N_FRAMES * len(hr_extrap))
            pt4_in.set_data(hr_inside[250:250+f1], rho_inside_4pt[:f1])
            pt4_ex.set_data(hr_extrap[:f2], rho_extrap_4pt[:f2])
        else:
            status_txt.set_text(BUILD[4])
            dot_glob.set_offsets([[H_TARGET, RES_GLOB]])
            dot_4pt.set_offsets([[H_TARGET, RES_4PT]])
            txt_glob.set_text(f"  {RES_GLOB:.4f}")
            txt_glob.set_x(H_TARGET+0.12); txt_glob.set_y(RES_GLOB+0.014)
            txt_4pt.set_text(f"  {RES_4PT:.4f}")
            txt_4pt.set_x(H_TARGET+0.12); txt_4pt.set_y(RES_4PT-0.022)

        if frame == bk[4] - 1:
            status_txt.set_text(
                f"✓  ρ(10.5 km) = {RES_4PT:.5f} [4pts]   "
                f"{RES_GLOB:.5f} [7pts]")
            draw_table("4pt")
            state["running"] = False
        return (glob_in, glob_ex, pt4_in, pt4_ex)

    state["anim"] = FuncAnimation(
        fig, update, frames=bk[4],
        interval=18, blit=False, repeat=False)
    fig.canvas.draw_idle()

def reset_animation(event=None):
    if state["anim"]:
        state["anim"].event_source.stop(); state["anim"] = None
    state["running"] = False; clear_all(); fig.canvas.draw_idle()

def show_4pt(e): draw_table("4pt")
def show_7pt(e): draw_table("7pt")

# ── Widgets ────────────────────────────────────────────────────────────────
ax_play  = fig.add_axes([0.06, 0.08, 0.14, 0.07])
ax_reset = fig.add_axes([0.22, 0.08, 0.12, 0.07])
ax_4pt   = fig.add_axes([0.37, 0.08, 0.14, 0.07])
ax_7pt   = fig.add_axes([0.53, 0.08, 0.14, 0.07])

for wax in [ax_play, ax_reset, ax_4pt, ax_7pt]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values(): sp.set_color(SPINE_CLR)

btn_play  = Button(ax_play,  "▶  Animar",
                   color="#0D2B55", hovercolor="#1A3A6E")
btn_reset = Button(ax_reset, "↺ Reset",
                   color="#1A1A3E", hovercolor="#2A2A5E")
btn_4pt   = Button(ax_4pt, "Tabla 4 pts",
                   color="#003A00", hovercolor="#005500")
btn_7pt   = Button(ax_7pt, "Tabla 7 pts",
                   color="#3A2A00", hovercolor="#5A4000")

btn_4pt.label.set_color(C_4PT);  btn_4pt.label.set_fontsize(9)
btn_7pt.label.set_color(C_GLOB); btn_7pt.label.set_fontsize(9)
for b in [btn_play, btn_reset]:
    b.label.set_color(TXT_PRI); b.label.set_fontsize(10)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)
btn_4pt.on_clicked(show_4pt)
btn_7pt.on_clicked(show_7pt)

draw_table(None)
plt.show()