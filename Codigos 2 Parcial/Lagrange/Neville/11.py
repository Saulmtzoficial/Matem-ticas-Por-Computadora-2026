"""
Spline Cúbico Natural — 5 puntos, evaluar en x = 3.4
Datos: x = [1, 2, 3, 4, 5], y = [13, 15, 12, 9, 13]
Condición natural: M0 = M4 = 0

Sistema tridiagonal (h=1 uniforme):
  [4 1 0] [M1]   [-30]
  [1 4 1] [M2] =  [ 0]
  [0 1 4] [M3]   [ 42]

Momentos: M = [0, -7.2857, -0.8571, 10.7143, 0]
S(3.4) ≈ 10.2549
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, Slider
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
SEG_COLORS = ["#FF7043", "#66BB6A", "#CE93D8", "#FFB74D"]  # 4 tramos

# ── Datos y momentos ──────────────────────────────────────────────────────
xs  = np.array([1., 2., 3., 4., 5.])
ys  = np.array([13., 15., 12., 9., 13.])
n   = len(xs)
h   = np.diff(xs)   # [1, 1, 1, 1]
M   = np.array([0., -7.28571429, -0.85714286, 10.71428571, 0.])
X_EVAL = 3.4

# ── Evaluación del spline ─────────────────────────────────────────────────
def spline_eval(x_val):
    i = min(max(np.searchsorted(xs, x_val, side='right') - 1, 0), n - 2)
    hi  = h[i]
    xi, xi1 = xs[i], xs[i+1]
    Mi, Mi1 = M[i], M[i+1]
    yi, yi1 = ys[i], ys[i+1]
    return (Mi  / (6*hi) * (xi1 - x_val)**3
          + Mi1 / (6*hi) * (x_val - xi)**3
          + (yi / hi - Mi  * hi / 6) * (xi1 - x_val)
          + (yi1/ hi - Mi1 * hi / 6) * (x_val - xi))

# Curvas por tramo
def spline_segment(i, x_arr):
    hi  = h[i]; xi, xi1 = xs[i], xs[i+1]
    Mi, Mi1 = M[i], M[i+1]; yi, yi1 = ys[i], ys[i+1]
    x = np.atleast_1d(x_arr)
    return (Mi  / (6*hi) * (xi1 - x)**3
          + Mi1 / (6*hi) * (x  - xi)**3
          + (yi / hi - Mi  * hi / 6) * (xi1 - x)
          + (yi1/ hi - Mi1 * hi / 6) * (x  - xi))

# Rangos por tramo
segments_xr = [np.linspace(xs[i], xs[i+1], 200) for i in range(n-1)]
result_val   = spline_eval(X_EVAL)

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
         "Spline Cúbico Natural  —  x = [1…5],  evaluar en  x = 3.4",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.21, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje gráfica
ax.set_facecolor(BG_AX)
ax.set_xlim(0.6, 5.5); ax.set_ylim(6, 18)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("S(x)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.5, zorder=1)

# Líneas de nodos
for xi in xs:
    ax.axvline(xi, color=SPINE_CLR, lw=0.6, ls=":", alpha=0.5, zorder=1)

# Línea de evaluación
ax.axvline(X_EVAL, color="#FF5252", lw=1.1, ls="--", alpha=0.8, zorder=2)
ax.text(X_EVAL+0.05, 6.4, f"x = {X_EVAL}", fontsize=8.5, color="#FF5252")

# Puntos de datos
ax.scatter(xs, ys, zorder=7, s=95, color=ACCENT,
           edgecolors="white", linewidths=1.8)
for xi, yi in zip(xs, ys):
    ax.annotate(f"({xi:.0f},{yi:.0f})", (xi, yi),
                textcoords="offset points", xytext=(6, 7),
                fontsize=8.5, color=ACCENT)

# Líneas animadas (una por tramo)
seg_lines = [ax.plot([], [], color=SEG_COLORS[i], lw=2.6, zorder=5,
                     label=f"S{i}(x)  [{xs[i]:.0f},{xs[i+1]:.0f}]")[0]
             for i in range(n-1)]

# Punto de evaluación
eval_dot = ax.scatter([], [], s=150, color="#FF5252", zorder=8,
                      edgecolors="white", linewidths=2, marker="*")
eval_hline, = ax.plot([], [], color="#FF5252", lw=0.9, ls=":", alpha=0, zorder=3)

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

ax.legend(loc="upper right", fontsize=7.8, framealpha=0.35,
          facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# Slider
ax_sl = fig.add_axes([0.44, 0.085, 0.38, 0.04])
ax_sl.set_facecolor(BG_FIG)
for sp in ax_sl.spines.values(): sp.set_color(SPINE_CLR)
slider = Slider(ax_sl, "x eval", 1.0, 5.0, valinit=X_EVAL,
                valstep=0.05, color="#185FA5", initcolor="none")
slider.label.set_color(TXT_PRI); slider.valtext.set_color(ACCENT)
probe_active = [False]

def update_probe(val):
    if not probe_active[0]: return
    xv = slider.val; yv = float(spline_eval(xv))
    eval_dot.set_offsets([[xv, yv]])
    eval_hline.set_data([0.6, xv], [yv, yv])
    eval_hline.set_alpha(0.55)
    # find which segment
    seg = min(max(np.searchsorted(xs, xv, 'right')-1, 0), n-2)
    ax.axvline(xv, color="#FF5252", lw=0, alpha=0)  # just trigger draw
    status_txt.set_text(
        f"S({xv:.2f}) = {yv:.5f}   [tramo {seg}: S{seg}]")
    fig.canvas.draw_idle()

slider.on_changed(update_probe)

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

    axt.text(0.5, 0.975, "Spline natural  —  momentos y evaluación",
             ha="center", va="top", fontsize=9.5,
             color=ACCENT, transform=axt.transAxes)

    r_h = 0.054
    y = 0.91

    def sec_hdr(y_c, txt):
        rect = mpatches.FancyBboxPatch(
            (0.01, y_c - r_h*0.46), 0.98, r_h*0.92,
            boxstyle="round,pad=0.003", facecolor="#1A3A5C",
            edgecolor="none", transform=axt.transAxes, zorder=1, clip_on=False)
        axt.add_patch(rect)
        axt.text(0.5, y_c, txt, ha="center", va="center",
                 fontsize=7.8, color=ACCENT,
                 transform=axt.transAxes, zorder=3, clip_on=False)

    def row(y_c, lbl, val, clr=TXT_PRI, bold=False, bg=None):
        if bg:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_c - r_h*0.46), 0.96, r_h*0.92,
                boxstyle="round,pad=0.003", facecolor=bg,
                edgecolor="none", transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.04, y_c, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_SEC,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.96, y_c, val, ha="right", va="center",
                 fontsize=8.0 if bold else 7.8,
                 color=clr, fontweight="bold" if bold else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    # 1. Sistema
    sec_hdr(y, "1 · Sistema tridiagonal  (h = 1)");  y -= r_h*1.25
    row(y, "Ecuación general:",
        "2(hᵢ₋₁+hᵢ)Mᵢ = 6[(yᵢ₊₁−yᵢ)/hᵢ−(yᵢ−yᵢ₋₁)/hᵢ₋₁]"); y -= r_h
    row(y, "Nodo i=1:", "4M₁ + M₂  = −30"); y -= r_h
    row(y, "Nodo i=2:", "M₁ + 4M₂ + M₃ = 0"); y -= r_h
    row(y, "Nodo i=3:", "M₂ + 4M₃  = 42"); y -= r_h

    axt.plot([0.02,0.98],[y-r_h*0.2]*2, color=SPINE_CLR, lw=0.5,
             ls="--", transform=axt.transAxes, zorder=4); y -= r_h*0.6

    # 2. Momentos
    sec_hdr(y, "2 · Momentos  M = S''(xᵢ)");  y -= r_h*1.25
    mom_lbl = ["M₀ (natural)", "M₁", "M₂", "M₃", "M₄ (natural)"]
    for mi, (lbl, mv) in enumerate(zip(mom_lbl, M)):
        is_nat = (mi == 0 or mi == 4)
        clr = ACCENT if not is_nat else "#FFF176"
        row(y, lbl, f"{mv:.5f}",
            clr=clr, bold=not is_nat,
            bg="#0A2A50" if not is_nat else None)
        y -= r_h

    axt.plot([0.02,0.98],[y-r_h*0.2]*2, color=SPINE_CLR, lw=0.5,
             ls="--", transform=axt.transAxes, zorder=4); y -= r_h*0.6

    # 3. Tramo activo [3,4]
    sec_hdr(y, "3 · Evaluación en  x = 3.4  →  tramo [3, 4]");  y -= r_h*1.25
    Mi2, Mi3 = M[2], M[3]
    t1 = Mi2/6*(4-3.4)**3; t2 = Mi3/6*(3.4-3)**3
    t3 = (12 - Mi2/6)*(4-3.4); t4 = (9 - Mi3/6)*(3.4-3)
    step_data = [
        ("t₁ = M₂/6·(4−3.4)³",  f"{Mi2:.5f}/6·0.216  = {t1:.5f}", SEG_COLORS[2]),
        ("t₂ = M₃/6·(3.4−3)³",  f"{Mi3:.5f}/6·0.064  = {t2:.5f}", SEG_COLORS[2]),
        ("t₃ = (y₂−M₂/6)·0.6",  f"({12:.4f}+{abs(Mi2/6):.4f})·0.6 = {t3:.5f}", SEG_COLORS[2]),
        ("t₄ = (y₃−M₃/6)·0.4",  f"({9:.4f}−{Mi3/6:.4f})·0.4  = {t4:.5f}", SEG_COLORS[2]),
    ]
    for lbl, val, clr in step_data:
        row(y, lbl, val, clr=clr); y -= r_h

    axt.plot([0.02,0.98],[y-r_h*0.2]*2, color=SPINE_CLR, lw=0.5,
             ls="--", transform=axt.transAxes, zorder=4); y -= r_h*0.6

    # 4. Resultado
    y_box = y - r_h*0.2
    box = mpatches.FancyBboxPatch(
        (0.03, y_box - 0.052), 0.94, 0.088,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box + 0.012,
             f"S(3.4)  =  t₁ + t₂ + t₃ + t₄",
             ha="center", va="center", fontsize=9,
             color="#B0BEC5", transform=axt.transAxes,
             zorder=5, clip_on=False)
    axt.text(0.5, y_box - 0.024,
             f"S(3.4)  ≈  {result_val:.6f}",
             ha="center", va="center", fontsize=11,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
N_FRAMES   = 75
MSG_FRAMES = 28
state = {"running": False, "anim": None}

BUILD = [
    "Condición natural: M₀ = M₄ = 0   (extremos libres)...",
    "Resolviendo sistema 3×3 tridiagonal...",
    "M = [0,  −7.2857,  −0.8571,  10.7143,  0]",
    f"Tramo S₀  [1,2]  →  naranja...",
    f"Tramo S₁  [2,3]  →  verde...",
    f"Tramo S₂  [3,4]  →  morado  (contiene x=3.4)...",
    f"Tramo S₃  [4,5]  →  ámbar...",
    f"Evaluando S₂(3.4)  =  {result_val:.6f}...",
]

CURVE_PHASES = [3, 4, 5, 6]   # índices en BUILD que corresponden a tramos

def clear_all():
    for ln in seg_lines: ln.set_data([], [])
    eval_dot.set_offsets(np.empty((0, 2)))
    eval_hline.set_data([], []); eval_hline.set_alpha(0)
    status_txt.set_text("")
    probe_active[0] = False
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]: return
    clear_all()
    state["running"] = True

    # sequence: msg0, msg1, msg2, [msg3+curve0], [msg4+curve1],
    #           [msg5+curve2], [msg6+curve3], msg7+dot
    n_msg_only = 3
    n_curve    = 4
    n_final    = 1
    total = n_msg_only*MSG_FRAMES + n_curve*(MSG_FRAMES+N_FRAMES) + n_final*MSG_FRAMES

    def update(frame):
        if frame < n_msg_only*MSG_FRAMES:
            ph = frame // MSG_FRAMES
            status_txt.set_text(BUILD[ph])
        else:
            f2 = frame - n_msg_only*MSG_FRAMES
            phase_len = MSG_FRAMES + N_FRAMES
            ph2 = f2 // phase_len
            loc = f2 % phase_len
            if ph2 < n_curve:
                status_txt.set_text(BUILD[n_msg_only + ph2])
                if loc >= MSG_FRAMES:
                    cf    = loc - MSG_FRAMES
                    n_pts = max(2, int((cf+1)/N_FRAMES * len(segments_xr[ph2])))
                    xp    = segments_xr[ph2][:n_pts]
                    seg_lines[ph2].set_data(xp, spline_segment(ph2, xp))
            else:
                # Final: mostrar punto
                status_txt.set_text(BUILD[-1])
                eval_dot.set_offsets([[X_EVAL, result_val]])
                eval_hline.set_data([0.6, X_EVAL], [result_val, result_val])
                eval_hline.set_alpha(0.6)

        if frame == total - 1:
            status_txt.set_text(
                f"✓  S(3.4) = {result_val:.6f}   "
                "— desliza para evaluar en otros puntos")
            probe_active[0] = True
            draw_table(done=True)
            state["running"] = False

        return tuple(seg_lines)

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
ax_play  = fig.add_axes([0.06, 0.085, 0.15, 0.07])
ax_reset = fig.add_axes([0.23, 0.085, 0.15, 0.07])
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