"""
Calor específico cp del aluminio — Interpolación de Neville
Datos: T(°C) = [-250, -200, -100, 0, 100, 300]
       cp(kJ/kg·K) = [0.0163, 0.318, 0.699, 0.870, 0.941, 1.04]

Determinar cp en T = 200°C y T = 400°C

cp(200°C) ≈ 0.9933 kJ/(kg·K)
cp(400°C) ≈ 0.9860 kJ/(kg·K)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons, Slider
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
G_COLORS  = ["#FF7043","#66BB6A","#CE93D8","#FFB74D","#80CBC4","#F48FB1"]

# ── Datos ──────────────────────────────────────────────────────────────────
Ts  = np.array([-250., -200., -100., 0., 100., 300.])
cps = np.array([0.0163, 0.318, 0.699, 0.870, 0.941, 1.04])
n   = len(Ts)

TARGETS   = [200., 400.]
TC        = ["#FF5252", "#80CBC4"]   # colores de cada objetivo

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

def neville_eval(x):
    return neville_table(Ts, cps, x)[-1, -1]

TABLES = {xv: neville_table(Ts, cps, xv) for xv in TARGETS}
RESULTS= {xv: TABLES[xv][-1,-1] for xv in TARGETS}

# Curva interpolante
Tr  = np.linspace(-260, 420, 600)
cpr = np.array([neville_eval(ti) for ti in Tr])

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color": TXT_PRI, "axes.labelcolor": TXT_SEC,
    "xtick.color": TXT_SEC, "ytick.color": TXT_SEC, "font.size": 9,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.975,
         "Calor específico del aluminio  cₚ(T)  —  Interpolación de Neville",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.21, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje
ax.set_facecolor(BG_AX)
ax.set_xlim(-280, 440); ax.set_ylim(-0.05, 1.35)
ax.set_xlabel("T  (°C)", fontsize=11)
ax.set_ylabel("cₚ  (kJ/kg·K)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.6, zorder=1)
ax.axvline(0, color=SPINE_CLR, lw=0.5, zorder=1)

# Líneas de targets
for xv, tc in zip(TARGETS, TC):
    ax.axvline(xv, color=tc, lw=1.0, ls="--", alpha=0.75, zorder=2)
    ax.text(xv+5, 0.05, f"T={xv:.0f}°C", fontsize=8, color=tc)

# Puntos de datos
ax.scatter(Ts, cps, zorder=6, s=90, color=ACCENT,
           edgecolors="white", linewidths=1.8)
for ti, ci in zip(Ts, cps):
    ax.annotate(f"({ti:.0f}, {ci})", (ti, ci),
                textcoords="offset points", xytext=(6, 7),
                fontsize=7.5, color=ACCENT)

# Curva animada
poly_line, = ax.plot([], [], color="#FFB74D", lw=2.6, zorder=5,
                     label="cₚ(T)  Neville (6 pts)")

# Puntos de evaluación
dots  = [ax.scatter([], [], s=150, color=tc, zorder=8,
                    edgecolors="white", linewidths=2, marker="*")
         for tc in TC]
htxt  = [ax.text(0, 0, "", fontsize=9, color=tc) for tc in TC]

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
    axt.set_xlim(0,1); axt.set_ylim(0,1); axt.set_axis_off()

    if mode is None:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara construir la interpolación",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle(); return

    xv  = mode
    Q   = TABLES[xv]
    tc  = TC[TARGETS.index(xv)]
    res = RESULTS[xv]

    axt.text(0.5, 0.975, f"Tabla Neville  —  T = {xv:.0f} °C",
             ha="center", va="top", fontsize=9.5, color=tc,
             transform=axt.transAxes)

    # Columnas: Tᵢ | cₚᵢ | ord1 | ord2 | ord3 | ord4 | ord5
    col_labels = ["Tᵢ","cₚᵢ","ord 1","ord 2","ord 3","ord 4","ord 5"]
    col_w      = [0.10, 0.10, 0.14, 0.14, 0.14, 0.14, 0.14]

    n_rows  = n + 1
    total_h = 0.52
    row_h   = total_h / (n_rows + 0.4)
    y_start = 0.89

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
            if val == "—": clr = SPINE_CLR
            elif is_hdr:   clr = TXT_SEC
            elif ci == hi_col:
                box = mpatches.FancyBboxPatch(
                    (x+0.004, y_c-row_h*0.42), cw-0.008, row_h*0.84,
                    boxstyle="round,pad=0.002",
                    facecolor="#0D47A1", edgecolor=tc, linewidth=0.8,
                    transform=axt.transAxes, zorder=2, clip_on=False)
                axt.add_patch(box); clr = "#FFFFFF"
            elif not is_hdr and ci >= 2:
                clr = G_COLORS[min(ci-2, len(G_COLORS)-1)]
            else: clr = TXT_PRI
            fs = 6.5 if is_hdr else 7.5
            fw = "bold" if ci == hi_col and not is_hdr else "normal"
            axt.text(x+cw/2, y_c, val, ha="center", va="center",
                     fontsize=fs, color=clr, fontweight=fw,
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    y_h = y_start - row_h*0.5
    draw_row(col_labels, y_h, is_hdr=True)
    axt.plot([0.01,0.99],[y_h-row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    for i in range(n):
        y_r = y_start - row_h*(i+1.5)
        hi = len(col_labels)-1 if i==n-1 else -1
        cells = [f"{Ts[i]:.0f}", f"{cps[i]:.4f}"]
        for j in range(1, n):
            cells.append(f"{Q[i,j]:.5f}" if i>=j else "—")
        draw_row(cells, y_r, hi_col=hi)

    axt.plot([0.01,0.99],[y_start-row_h*(n+1.0)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # Ambos resultados
    y_res = y_start - row_h*(n+1.5)
    r_h2  = 0.072
    for ti, tc2 in zip(TARGETS, TC):
        rv = RESULTS[ti]
        is_cur = (ti == xv)
        if is_cur:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_res-r_h2*0.46), 0.96, r_h2*0.92,
                boxstyle="round,pad=0.003", facecolor="#0A2A50",
                edgecolor="none", transform=axt.transAxes,
                zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.04, y_res, f"cₚ({ti:.0f}°C) =",
                 ha="left", va="center", fontsize=8, color=tc2,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.96, y_res, f"{rv:.5f}  kJ/(kg·K)",
                 ha="right", va="center",
                 fontsize=9 if is_cur else 8, color=tc2,
                 fontweight="bold" if is_cur else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)
        y_res -= r_h2

    # Caja final
    y_box = y_res - r_h2*0.5
    box = mpatches.FancyBboxPatch(
        (0.03, y_box-0.052), 0.94, 0.088,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=tc, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box+0.016,
             f"cₚ(200°C) = {RESULTS[200.]:.5f}  kJ/(kg·K)",
             ha="center", va="center", fontsize=9.5,
             color=TC[0], fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)
    axt.text(0.5, y_box-0.022,
             f"cₚ(400°C) = {RESULTS[400.]:.5f}  kJ/(kg·K)",
             ha="center", va="center", fontsize=9.5,
             color=TC[1], fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
N_FRAMES   = 100
MSG_FRAMES = 28
state = {"running": False, "anim": None}

BUILD = [
    "Datos: cₚ del aluminio a 6 temperaturas...",
    "Construyendo tabla de Neville 6×6...",
    "Trazando polinomio interpolante cₚ(T)...",
    "Evaluando en T = 200 °C...",
    "Evaluando en T = 400 °C...",
]

def clear_all():
    poly_line.set_data([], [])
    for d in dots: d.set_offsets(np.empty((0,2)))
    for t in htxt: t.set_text("")
    status_txt.set_text("")
    draw_table(None)

def run_animation(event=None):
    if state["running"]: return
    clear_all(); state["running"] = True

    n_msg = 2; total = n_msg*MSG_FRAMES + N_FRAMES + 2*MSG_FRAMES

    def update(frame):
        if frame < n_msg*MSG_FRAMES:
            ph = frame // MSG_FRAMES
            status_txt.set_text(BUILD[ph])
        elif frame < n_msg*MSG_FRAMES + N_FRAMES:
            cf = frame - n_msg*MSG_FRAMES
            status_txt.set_text(BUILD[2])
            n_pts = max(2, int((cf+1)/N_FRAMES * len(Tr)))
            poly_line.set_data(Tr[:n_pts], cpr[:n_pts])
        else:
            f2  = frame - n_msg*MSG_FRAMES - N_FRAMES
            idx = f2 // MSG_FRAMES
            if idx < 2:
                status_txt.set_text(BUILD[3+idx])
                xv = TARGETS[idx]; yv = RESULTS[xv]
                dots[idx].set_offsets([[xv, yv]])
                htxt[idx].set_text(f"  {yv:.4f}")
                htxt[idx].set_x(xv+8); htxt[idx].set_y(yv+0.03)

        if frame == total - 1:
            status_txt.set_text(
                f"✓  cₚ(200°C)={RESULTS[200.]:.4f}   "
                f"cₚ(400°C)={RESULTS[400.]:.4f}  kJ/(kg·K)")
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

def show_200(e): draw_table(200.)
def show_400(e): draw_table(400.)

# ── Widgets ────────────────────────────────────────────────────────────────
ax_play  = fig.add_axes([0.06, 0.08, 0.15, 0.07])
ax_reset = fig.add_axes([0.23, 0.08, 0.13, 0.07])
ax_t1    = fig.add_axes([0.39, 0.08, 0.13, 0.07])
ax_t2    = fig.add_axes([0.54, 0.08, 0.13, 0.07])

for wax in [ax_play, ax_reset, ax_t1, ax_t2]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values(): sp.set_color(SPINE_CLR)

btn_play  = Button(ax_play,  "▶   Animar",
                   color="#0D2B55", hovercolor="#1A3A6E")
btn_reset = Button(ax_reset, "↺  Reset",
                   color="#1A1A3E", hovercolor="#2A2A5E")
btn_t1    = Button(ax_t1, "T = 200 °C",
                   color="#3A0000", hovercolor="#5A1000")
btn_t2    = Button(ax_t2, "T = 400 °C",
                   color="#003030", hovercolor="#005050")

btn_t1.label.set_color(TC[0]); btn_t1.label.set_fontsize(9.5)
btn_t2.label.set_color(TC[1]); btn_t2.label.set_fontsize(9.5)
for b in [btn_play, btn_reset]:
    b.label.set_color(TXT_PRI); b.label.set_fontsize(10)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)
btn_t1.on_clicked(show_200)
btn_t2.on_clicked(show_400)

draw_table(None)
plt.show()