"""
Viscosidad cinemática del agua μk vs temperatura T
Interpolación de Neville con 4 vecinos más cercanos

Datos: T(°C)        = [0, 21.1, 37.8, 54.4, 71.1, 87.8, 100]
       μk(10⁻³m²/s) = [1.79, 1.13, 0.696, 0.519, 0.338, 0.321, 0.296]

Resultados:
  μk(10°)  ≈ 1.4798 × 10⁻³ m²/s
  μk(30°)  ≈ 0.8767 × 10⁻³ m²/s
  μk(60°)  ≈ 0.4504 × 10⁻³ m²/s
  μk(90°)  ≈ 0.3203 × 10⁻³ m²/s
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
TC        = ["#FF5252", "#FFB74D", "#66BB6A", "#CE93D8"]

# ── Datos ──────────────────────────────────────────────────────────────────
T_data  = np.array([0., 21.1, 37.8, 54.4, 71.1, 87.8, 100.])
mu_data = np.array([1.79, 1.13, 0.696, 0.519, 0.338, 0.321, 0.296])
n       = len(T_data)
TARGETS = [10., 30., 60., 90.]

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

def get_4nn(T_t):
    dists = np.abs(T_data - T_t)
    idx4  = np.sort(np.argsort(dists)[:4])
    return idx4, T_data[idx4], mu_data[idx4]

# Precompute
INFO = {}
for T_t in TARGETS:
    idx4, xs4, ys4 = get_4nn(T_t)
    Q    = neville_table(xs4, ys4, T_t)
    res  = Q[-1, -1]
    INFO[T_t] = {"idx4": idx4, "xs4": xs4, "ys4": ys4, "Q": Q, "res": res}

# Curva global Neville (7 pts)
def neville_global(T_val):
    return neville_table(T_data, mu_data, T_val)[-1, -1]

Tr      = np.linspace(0, 100, 500)
mu_glob = np.array([neville_global(ti) for ti in Tr])

# Curva Neville 4pt por target
def nev4_curve(T_t, T_arr):
    inf = INFO[T_t]
    xs4, ys4 = inf["xs4"], inf["ys4"]
    return np.array([neville_table(xs4, ys4, tv)[-1,-1] for tv in np.atleast_1d(T_arr)])

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color": TXT_PRI, "axes.labelcolor": TXT_SEC,
    "xtick.color": TXT_SEC, "ytick.color": TXT_SEC, "font.size": 9,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.975,
         "Viscosidad cinemática del agua  μk(T)  —  Neville 4 vecinos",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.21, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje
ax.set_facecolor(BG_AX)
ax.set_xlim(-3, 105); ax.set_ylim(0.15, 2.05)
ax.set_xlabel("T  (°C)", fontsize=11)
ax.set_ylabel("μk  (×10⁻³ m²/s)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.5, zorder=1)

# Líneas de targets
for T_t, tc in zip(TARGETS, TC):
    ax.axvline(T_t, color=tc, lw=0.9, ls="--", alpha=0.75, zorder=2)
    ax.text(T_t+0.8, 0.20, f"{T_t:.0f}°C", fontsize=8, color=tc)

# Puntos de datos
ax.scatter(T_data, mu_data, zorder=7, s=95, color=ACCENT,
           edgecolors="white", linewidths=1.8)
for Ti, mi in zip(T_data, mu_data):
    ax.annotate(f"({Ti:.0f},{mi})", (Ti, mi),
                textcoords="offset points", xytext=(5, 7),
                fontsize=7.5, color=ACCENT)

# Curva global (fondo, punteada)
glob_line, = ax.plot([], [], color=TXT_SEC, lw=1.2, ls=":", alpha=0, zorder=3,
                     label="Neville global (7 pts)")

# Curvas 4pt por target
nev4_lines = [ax.plot([], [], color=tc, lw=2.6, zorder=5)[0] for tc in TC]

# Puntos usados y evaluados
used_scats = [ax.scatter([], [], s=100, color=tc, zorder=8,
                          edgecolors="white", linewidths=1.5, marker="D")
              for tc in TC]
eval_dots  = [ax.scatter([], [], s=160, color=tc, zorder=9,
                          edgecolors="white", linewidths=2, marker="*")
              for tc in TC]
eval_txts  = [ax.text(0, 0, "", fontsize=9, color=tc) for tc in TC]

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

handles = ([mpatches.Patch(color=ACCENT, label="Datos")] +
           [mpatches.Patch(color=tc, label=f"μk({T_t:.0f}°C)")
            for T_t, tc in zip(TARGETS, TC)])
ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.35,
          facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(target_idx=None):
    axt.cla(); axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
    axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.set_axis_off()

    if target_idx is None:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\no selecciona temperatura",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle(); return

    T_t  = TARGETS[target_idx]
    tc   = TC[target_idx]
    inf  = INFO[T_t]
    Q    = inf["Q"]; xs4 = inf["xs4"]; ys4 = inf["ys4"]
    res  = inf["res"]; idx4 = inf["idx4"]

    axt.text(0.5, 0.975, f"Tabla Neville  —  T = {T_t:.0f} °C",
             ha="center", va="top", fontsize=9.5, color=tc,
             transform=axt.transAxes)

    col_labels = ["Tᵢ (°C)", "μkᵢ", "ord 1", "ord 2", "ord 3"]
    col_w      = [0.18, 0.18, 0.20, 0.20, 0.24]

    n4      = 4
    total_h = 0.30
    row_h   = total_h / (n4 + 1.2)
    y_start = 0.90

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
                cols = ["#FF7043","#66BB6A","#CE93D8"]
                clr = cols[min(ci-2, 2)]
            else: clr = TXT_PRI
            fs = 6.8 if is_hdr else 8.2
            fw = "bold" if ci == hi_col and not is_hdr else "normal"
            axt.text(x+cw/2, y_c, val, ha="center", va="center",
                     fontsize=fs, color=clr, fontweight=fw,
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    y_h = y_start - row_h*0.5
    draw_row(col_labels, y_h, is_hdr=True)
    axt.plot([0.01, 0.99], [y_h-row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    for i in range(n4):
        y_r = y_start - row_h*(i+1.5)
        hi  = n4 if i == n4-1 else -1
        cells = [f"{xs4[i]:.1f}", f"{ys4[i]:.5f}"]
        for j in range(1, n4):
            cells.append(f"{Q[i,j]:.5f}" if i >= j else "—")
        draw_row(cells, y_r, hi_col=hi)

    axt.plot([0.01, 0.99], [y_start-row_h*(n4+1.0)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # Resumen de todos los resultados
    y_res = y_start - row_h*(n4+1.6)
    r_h2  = 0.070
    axt.text(0.5, y_res, "Resumen de los 4 objetivos:",
             ha="center", va="center", fontsize=8, color=TXT_SEC,
             transform=axt.transAxes, zorder=3, clip_on=False)
    y_res -= r_h2*0.85

    for Ti_t, tc2 in zip(TARGETS, TC):
        rv   = INFO[Ti_t]["res"]
        is_c = (Ti_t == T_t)
        if is_c:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_res-r_h2*0.46), 0.96, r_h2*0.92,
                boxstyle="round,pad=0.003", facecolor="#0A2A50",
                edgecolor="none", transform=axt.transAxes,
                zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.05, y_res, f"μk({Ti_t:.0f}°C)  =",
                 ha="left", va="center", fontsize=8, color=tc2,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.95, y_res, f"{rv:.5f}  ×10⁻³ m²/s",
                 ha="right", va="center",
                 fontsize=9 if is_c else 8, color=tc2,
                 fontweight="bold" if is_c else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)
        y_res -= r_h2

    # Caja final
    y_box = y_res - r_h2*0.5
    box = mpatches.FancyBboxPatch(
        (0.03, y_box-0.042), 0.94, 0.075,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=tc, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box-0.004,
             f"μk({T_t:.0f}°C)  =  {res:.5f}  ×10⁻³ m²/s",
             ha="center", va="center", fontsize=11,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
N_FRAMES   = 80
MSG_FRAMES = 28
state = {"running": False, "anim": None}

BUILD = [
    "Datos: μk del agua en 7 temperaturas...",
    "Curva de referencia Neville (7 pts)...",
    "Neville 4pts  T=10°C  (vecinos: 0, 21.1, 37.8, 54.4)...",
    "Neville 4pts  T=30°C  (vecinos: 0, 21.1, 37.8, 54.4)...",
    "Neville 4pts  T=60°C  (vecinos: 37.8, 54.4, 71.1, 87.8)...",
    "Neville 4pts  T=90°C  (vecinos: 54.4, 71.1, 87.8, 100)...",
    "Marcando puntos de evaluación...",
]

# Rangos de curva 4pt por target
nev4_xr = [np.linspace(INFO[T_t]["xs4"][0], INFO[T_t]["xs4"][-1], 200)
            for T_t in TARGETS]

def clear_all():
    for ln in nev4_lines: ln.set_data([], [])
    glob_line.set_alpha(0); glob_line.set_data([], [])
    for d in used_scats+eval_dots: d.set_offsets(np.empty((0,2)))
    for t in eval_txts: t.set_text("")
    status_txt.set_text("")
    draw_table(None)

def run_animation(event=None):
    if state["running"]: return
    clear_all(); state["running"] = True

    n_msg = 1; n_glob = 1; n_cur = 4; n_fin = 1
    ph_len = MSG_FRAMES + N_FRAMES
    total  = n_msg*MSG_FRAMES + n_glob*ph_len + n_cur*ph_len + n_fin*MSG_FRAMES

    def update(frame):
        bk0 = n_msg*MSG_FRAMES
        bk1 = bk0 + ph_len          # fin curva global
        bk2 = bk1 + n_cur*ph_len    # fin curvas 4pt

        if frame < bk0:
            status_txt.set_text(BUILD[0])
        elif frame < bk1:
            loc = frame - bk0
            status_txt.set_text(BUILD[1])
            if loc >= MSG_FRAMES:
                cf = loc - MSG_FRAMES
                n_pts = max(2, int((cf+1)/N_FRAMES * len(Tr)))
                glob_line.set_data(Tr[:n_pts], mu_glob[:n_pts])
                glob_line.set_alpha(0.4)
        elif frame < bk2:
            f2  = frame - bk1
            ph2 = f2 // ph_len; loc = f2 % ph_len
            if ph2 < n_cur:
                T_t = TARGETS[ph2]
                status_txt.set_text(BUILD[2+ph2])
                used_scats[ph2].set_offsets(
                    np.column_stack([INFO[T_t]["xs4"], INFO[T_t]["ys4"]]))
                if loc >= MSG_FRAMES:
                    cf    = loc - MSG_FRAMES
                    n_pts = max(2, int((cf+1)/N_FRAMES * len(nev4_xr[ph2])))
                    xp    = nev4_xr[ph2][:n_pts]
                    nev4_lines[ph2].set_data(xp, nev4_curve(T_t, xp))
        else:
            status_txt.set_text(BUILD[-1])
            for ti, T_t in enumerate(TARGETS):
                rv = INFO[T_t]["res"]
                eval_dots[ti].set_offsets([[T_t, rv]])
                eval_txts[ti].set_text(f"  {rv:.4f}")
                eval_txts[ti].set_x(T_t+1.0); eval_txts[ti].set_y(rv+0.04)

        if frame == total - 1:
            r = INFO
            status_txt.set_text(
                f"✓  μk: {r[10.]['res']:.4f} | {r[30.]['res']:.4f} | "
                f"{r[60.]['res']:.4f} | {r[90.]['res']:.4f}  (×10⁻³ m²/s)")
            draw_table(0); state["running"] = False
        return tuple(nev4_lines) + (glob_line,)

    state["anim"] = FuncAnimation(
        fig, update, frames=total,
        interval=18, blit=False, repeat=False)
    fig.canvas.draw_idle()

def reset_animation(event=None):
    if state["anim"]:
        state["anim"].event_source.stop(); state["anim"] = None
    state["running"] = False; clear_all(); fig.canvas.draw_idle()

def show_tab(label):
    idx = {"T=10°C": 0, "T=30°C": 1, "T=60°C": 2, "T=90°C": 3}.get(label, 0)
    draw_table(idx)

# ── Widgets ────────────────────────────────────────────────────────────────
ax_play  = fig.add_axes([0.06, 0.08, 0.14, 0.07])
ax_reset = fig.add_axes([0.22, 0.08, 0.12, 0.07])
ax_radio = fig.add_axes([0.38, 0.03, 0.28, 0.14])

for wax in [ax_play, ax_reset, ax_radio]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values(): sp.set_color(SPINE_CLR)

btn_play  = Button(ax_play,  "▶  Animar",
                   color="#0D2B55", hovercolor="#1A3A6E")
btn_reset = Button(ax_reset, "↺ Reset",
                   color="#1A1A3E", hovercolor="#2A2A5E")
radio     = RadioButtons(ax_radio,
                         ("T=10°C", "T=30°C", "T=60°C", "T=90°C"),
                         activecolor=ACCENT)

for b in [btn_play, btn_reset]:
    b.label.set_color(TXT_PRI); b.label.set_fontsize(10)
for lbl in radio.labels:
    lbl.set_color(TXT_PRI); lbl.set_fontsize(9.5)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)
radio.on_clicked(show_tab)

draw_table(None)
plt.show()