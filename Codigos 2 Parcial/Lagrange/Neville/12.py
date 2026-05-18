"""
Interpolación Inversa con Spline Cúbico Natural — Cero de y(x)
Datos: x = [0.2, 0.4, 0.6, 0.8, 1.0]
       y = [1.150, 0.855, 0.377, -0.266, -1.049]

Estrategia: intercambiar roles x↔y, reordenar y ascendente,
construir spline S_inv(y), evaluar S_inv(0) → x donde y=0.

Momentos M = [0, 1.338896, -0.448826, -0.158614, 0]
Cero: x* = S_inv(0) ≈ 0.6924
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
SEG_COLORS = ["#FF7043", "#66BB6A", "#CE93D8", "#FFB74D"]

# ── Datos originales ───────────────────────────────────────────────────────
xs_orig = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
ys_orig = np.array([1.150, 0.855, 0.377, -0.266, -1.049])

# Intercambio x↔y, reordenar ascendente
idx    = np.argsort(ys_orig)
xs_inv = ys_orig[idx]        # nuevo x = y original (ascendente)
ys_inv = xs_orig[idx]        # nuevo y = x original
n      = len(xs_inv)
h      = np.diff(xs_inv)

# ── Momentos (sistema tridiagonal) ────────────────────────────────────────
A_mat = np.zeros((n-2, n-2))
rhs_v = np.zeros(n-2)
for k in range(1, n-1):
    ii = k - 1
    if ii > 0:   A_mat[ii, ii-1] = h[k-2]
    A_mat[ii, ii] = 2*(h[k-2] + h[k-1])
    if ii < n-3: A_mat[ii, ii+1] = h[k-1]
    rhs_v[ii] = 6*((ys_inv[k+1]-ys_inv[k])/h[k-1]
                  -(ys_inv[k]-ys_inv[k-1])/h[k-2])

M_inner = np.linalg.solve(A_mat, rhs_v)
M       = np.concatenate([[0.], M_inner, [0.]])

# ── Evaluación ────────────────────────────────────────────────────────────
def spline_eval(x_val):
    i = min(max(np.searchsorted(xs_inv, x_val, side='right')-1, 0), n-2)
    hi = h[i]; xi,xi1 = xs_inv[i],xs_inv[i+1]
    Mi,Mi1 = M[i],M[i+1]; yi,yi1 = ys_inv[i],ys_inv[i+1]
    return (Mi/(6*hi)*(xi1-x_val)**3 + Mi1/(6*hi)*(x_val-xi)**3
           +(yi/hi - Mi*hi/6)*(xi1-x_val)
           +(yi1/hi - Mi1*hi/6)*(x_val-xi))

def spline_segment(i, x_arr):
    x = np.atleast_1d(x_arr)
    hi = h[i]; xi,xi1 = xs_inv[i],xs_inv[i+1]
    Mi,Mi1 = M[i],M[i+1]; yi,yi1 = ys_inv[i],ys_inv[i+1]
    return (Mi/(6*hi)*(xi1-x)**3 + Mi1/(6*hi)*(x-xi)**3
           +(yi/hi - Mi*hi/6)*(xi1-x)
           +(yi1/hi - Mi1*hi/6)*(x-xi))

zero_x = spline_eval(0.0)          # ≈ 0.6924
segments_xr = [np.linspace(xs_inv[i], xs_inv[i+1], 200) for i in range(n-1)]

# ── Figura: DOS paneles (original + invertido) ────────────────────────────
plt.rcParams.update({
    "text.color": TXT_PRI, "axes.labelcolor": TXT_SEC,
    "xtick.color": TXT_SEC, "ytick.color": TXT_SEC, "font.size": 9,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.975,
         "Interpolación Inversa (Spline Natural)  —  Cero de y(x)",
         ha="center", va="top", fontsize=12, color=ACCENT)

# Layout: ax_orig | ax_inv | axt (tabla)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9],
                      left=0.05, right=0.97,
                      bottom=0.18, top=0.93, wspace=0.08)
ax_o = fig.add_subplot(gs[0])   # datos originales
ax_i = fig.add_subplot(gs[1])   # datos invertidos + spline
axt  = fig.add_subplot(gs[2])   # tabla

def style_ax(a, xl, yl, xlab, ylab):
    a.set_facecolor(BG_AX)
    a.set_xlim(*xl); a.set_ylim(*yl)
    a.set_xlabel(xlab, fontsize=10); a.set_ylabel(ylab, fontsize=10)
    for sp in a.spines.values(): sp.set_color(SPINE_CLR)
    a.tick_params(colors=TXT_SEC, labelsize=8.5)
    a.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
    a.axhline(0, color=SPINE_CLR, lw=0.9, zorder=1)
    a.axvline(0, color=SPINE_CLR, lw=0.6, zorder=1)

style_ax(ax_o, (-0.1, 1.2), (-1.4, 1.5), "x", "y(x)")
style_ax(ax_i, (-1.3, 1.4), (0.1, 1.1),  "y  (nuevo x)", "x  (nuevo y)")

# ── Panel izquierdo: datos originales ─────────────────────────────────────
ax_o.set_title("Datos originales", fontsize=9, color=TXT_SEC, pad=4)
ax_o.scatter(xs_orig, ys_orig, s=80, color=ACCENT,
             edgecolors="white", linewidths=1.5, zorder=6)
for xi,yi in zip(xs_orig, ys_orig):
    ax_o.annotate(f"({xi},{yi})", (xi,yi),
                  textcoords="offset points", xytext=(5,6),
                  fontsize=7.5, color=ACCENT)

# Flecha de intercambio
ax_o.text(0.55, -0.6, "x↔y\nreordenar ↑", ha="center",
          fontsize=9, color=TXT_SEC,
          bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_AX,
                    edgecolor=SPINE_CLR, linewidth=0.8))

# Línea vertical donde cruza cero (aproximada)
ax_o.axvline(zero_x, color="#FF5252", lw=1.1, ls="--", alpha=0.8, zorder=2)
ax_o.scatter([zero_x], [0], s=120, color="#FF5252", zorder=8,
             edgecolors="white", linewidths=1.5, marker="*")
ax_o.text(zero_x+0.03, 0.15, f"x*≈{zero_x:.4f}", fontsize=8, color="#FF5252")

# ── Panel derecho: spline inverso ─────────────────────────────────────────
ax_i.set_title("Spline inverso  S(y) = x", fontsize=9, color=TXT_SEC, pad=4)
ax_i.scatter(xs_inv, ys_inv, s=80, color="#FFB74D",
             edgecolors="white", linewidths=1.5, zorder=6)
for xi,yi in zip(xs_inv, ys_inv):
    ax_i.annotate(f"({xi:.3f},{yi:.1f})", (xi,yi),
                  textcoords="offset points", xytext=(5,6),
                  fontsize=7, color="#FFB74D")

# Línea vertical en y_nuevo=0
ax_i.axvline(0, color="#FF5252", lw=1.1, ls="--", alpha=0.8, zorder=2)
ax_i.text(0.03, 0.15, "y=0", fontsize=8, color="#FF5252")

# Líneas animadas (4 tramos)
seg_lines = [ax_i.plot([], [], color=SEG_COLORS[i], lw=2.4, zorder=5)[0]
             for i in range(n-1)]

# Punto de evaluación en ax_i
eval_dot_i = ax_i.scatter([], [], s=150, color="#FF5252", zorder=8,
                           edgecolors="white", linewidths=2, marker="*")
eval_hline, = ax_i.plot([], [], color="#FF5252", lw=0.9, ls=":", alpha=0, zorder=3)

status_txt = fig.text(0.36, 0.01, "", ha="center", va="bottom",
                      fontsize=9.5, color=ACCENT)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(done=False):
    axt.cla()
    axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
    axt.set_xlim(0,1); axt.set_ylim(0,1)
    axt.set_axis_off()

    if not done:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara construir el spline\ninverso",
                 ha="center", va="center", fontsize=10.5,
                 color=TXT_SEC, linespacing=2.0, transform=axt.transAxes)
        fig.canvas.draw_idle()
        return

    axt.text(0.5, 0.975, "Interpolación inversa",
             ha="center", va="top", fontsize=10, color=ACCENT,
             transform=axt.transAxes)

    r_h = 0.056
    y   = 0.90

    def sec_hdr(y_c, txt):
        rect = mpatches.FancyBboxPatch(
            (0.01, y_c-r_h*0.46), 0.98, r_h*0.92,
            boxstyle="round,pad=0.003", facecolor="#1A3A5C",
            edgecolor="none", transform=axt.transAxes, zorder=1, clip_on=False)
        axt.add_patch(rect)
        axt.text(0.5, y_c, txt, ha="center", va="center",
                 fontsize=7.8, color=ACCENT,
                 transform=axt.transAxes, zorder=3, clip_on=False)

    def row(y_c, lbl, val, clr=TXT_PRI, bold=False, bg=None):
        if bg:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_c-r_h*0.46), 0.96, r_h*0.92,
                boxstyle="round,pad=0.003", facecolor=bg,
                edgecolor="none", transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.04, y_c, lbl, ha="left", va="center",
                 fontsize=7.2, color=TXT_SEC,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.96, y_c, val, ha="right", va="center",
                 fontsize=8.0 if bold else 7.5, color=clr,
                 fontweight="bold" if bold else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    def hline(y_c):
        axt.plot([0.02,0.98],[y_c]*2, color=SPINE_CLR, lw=0.5,
                 ls="--", transform=axt.transAxes, zorder=4)

    # 1. Reordenamiento
    sec_hdr(y, "1 · Reordenar: x_inv = y_orig  ↑");  y -= r_h*1.2
    for xi,yi in zip(xs_inv, ys_inv):
        row(y, f"x_inv = {xi:+.3f}", f"y_inv = {yi:.1f}")
        y -= r_h
    hline(y-r_h*0.2); y -= r_h*0.7

    # 2. Momentos
    sec_hdr(y, "2 · Momentos  M  (condición natural)");  y -= r_h*1.2
    m_lbl = ["M₀ (nat.)", "M₁", "M₂", "M₃", "M₄ (nat.)"]
    m_clr = ["#FFF176", ACCENT, ACCENT, ACCENT, "#FFF176"]
    for lbl, mv, mc in zip(m_lbl, M, m_clr):
        is_key = mc == ACCENT
        row(y, lbl, f"{mv:.5f}", clr=mc, bold=is_key,
            bg="#0A2A50" if is_key else None)
        y -= r_h
    hline(y-r_h*0.2); y -= r_h*0.7

    # 3. Evaluación en y=0
    sec_hdr(y, "3 · S_inv(0)  →  tramo [−0.266, 0.377]");  y -= r_h*1.2
    i = 1
    hi = h[i]; xi,xi1 = xs_inv[i],xs_inv[i+1]
    Mi,Mi1 = M[i],M[i+1]; yi_v,yi1_v = ys_inv[i],ys_inv[i+1]
    t1 = Mi/(6*hi)*(xi1)**3
    t2 = Mi1/(6*hi)*(-xi)**3
    t3 = (yi_v/hi - Mi*hi/6)*xi1
    t4 = (yi1_v/hi - Mi1*hi/6)*(-xi)
    steps = [
        ("t₁ = M₁/6h·(x₁)³",   f"{t1:+.6f}",  SEG_COLORS[1]),
        ("t₂ = M₂/6h·(−x₀)³",  f"{t2:+.6f}",  SEG_COLORS[1]),
        ("t₃ = (y₀/h−M₁h/6)x₁",f"{t3:+.6f}",  SEG_COLORS[1]),
        ("t₄ = (y₁/h−M₂h/6)(−x₀)", f"{t4:+.6f}", SEG_COLORS[1]),
    ]
    for lbl,val,clr in steps:
        row(y, lbl, val, clr=clr); y -= r_h
    hline(y-r_h*0.2); y -= r_h*0.7

    # 4. Resultado
    y_box = y - r_h*0.2
    box = mpatches.FancyBboxPatch(
        (0.03, y_box-0.06), 0.94, 0.10,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box+0.018,
             "S_inv(0)  =  t₁+t₂+t₃+t₄",
             ha="center", va="center", fontsize=8.5,
             color="#B0BEC5", transform=axt.transAxes,
             zorder=5, clip_on=False)
    axt.text(0.5, y_box-0.024,
             f"x*  =  {zero_x:.6f}",
             ha="center", va="center", fontsize=12,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
N_FRAMES   = 70
MSG_FRAMES = 30
state = {"running": False, "anim": None}

BUILD = [
    "Datos originales: cambio de signo entre x=0.6 y x=0.8...",
    "Intercambiando roles: x_inv = y_orig,  y_inv = x_orig...",
    "Reordenando x_inv en orden ascendente...",
    "Resolviendo sistema tridiagonal para momentos M...",
    "Trazando tramo S₀  [−1.049, −0.266]...",
    "Trazando tramo S₁  [−0.266,  0.377]  ← contiene y=0...",
    "Trazando tramo S₂  [ 0.377,  0.855]...",
    "Trazando tramo S₃  [ 0.855,  1.150]...",
    f"Evaluando S_inv(0) ≈ {zero_x:.5f}...",
]

def clear_all():
    for ln in seg_lines: ln.set_data([], [])
    eval_dot_i.set_offsets(np.empty((0,2)))
    eval_hline.set_data([], []); eval_hline.set_alpha(0)
    status_txt.set_text("")
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]: return
    clear_all()
    state["running"] = True

    n_msg  = 4   # mensajes iniciales
    n_curv = 4   # tramos
    n_fin  = 1
    total  = n_msg*MSG_FRAMES + n_curv*(MSG_FRAMES+N_FRAMES) + n_fin*MSG_FRAMES

    def update(frame):
        if frame < n_msg*MSG_FRAMES:
            ph = frame // MSG_FRAMES
            status_txt.set_text(BUILD[ph])
        else:
            f2 = frame - n_msg*MSG_FRAMES
            ph_len = MSG_FRAMES + N_FRAMES
            ph2 = f2 // ph_len
            loc = f2 % ph_len
            if ph2 < n_curv:
                status_txt.set_text(BUILD[n_msg + ph2])
                if loc >= MSG_FRAMES:
                    cf = loc - MSG_FRAMES
                    n_pts = max(2, int((cf+1)/N_FRAMES * len(segments_xr[ph2])))
                    xp = segments_xr[ph2][:n_pts]
                    seg_lines[ph2].set_data(xp, spline_segment(ph2, xp))
            else:
                status_txt.set_text(BUILD[-1])
                eval_dot_i.set_offsets([[0.0, zero_x]])
                eval_hline.set_data([-1.3, 0.0], [zero_x, zero_x])
                eval_hline.set_alpha(0.65)

        if frame == total - 1:
            status_txt.set_text(
                f"✓  Cero de y(x) en  x* = {zero_x:.5f}")
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
ax_play  = fig.add_axes([0.05, 0.075, 0.13, 0.07])
ax_reset = fig.add_axes([0.20, 0.075, 0.13, 0.07])
for wax in [ax_play, ax_reset]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values(): sp.set_color(SPINE_CLR)

btn_play  = Button(ax_play,  "▶  Animar",
                   color="#0D2B55", hovercolor="#1A3A6E")
btn_reset = Button(ax_reset, "↺  Reset",
                   color="#1A1A3E", hovercolor="#2A2A5E")
for b in [btn_play, btn_reset]:
    b.label.set_color(TXT_PRI); b.label.set_fontsize(10)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)

draw_table(done=False)
plt.show()