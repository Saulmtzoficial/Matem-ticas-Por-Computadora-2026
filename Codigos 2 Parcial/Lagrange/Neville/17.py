"""
Coeficiente de arrastre cD vs número de Reynolds Re
Spline Cúbico Natural en escala log-log

Datos: Re  = [0.2, 2, 20, 200, 2000, 20000]
       cD  = [103, 13.9, 2.72, 0.800, 0.401, 0.433]

Transformación: u = log10(Re),  v = log10(cD)
Intervalos uniformes h = 1 en escala log10

Momentos M = [0, 0.2003, 0.1673, 0.1925, 0.4518, 0]

Resultados:
  cD(Re=5)    ≈ 6.902
  cD(Re=50)   ≈ 1.591
  cD(Re=500)  ≈ 0.557
  cD(Re=5000) ≈ 0.387
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
SEG_COLORS= ["#FF7043","#66BB6A","#CE93D8","#FFB74D","#80CBC4"]

TC = ["#FF5252","#CE93D8","#66BB6A","#FFB74D"]   # colores por target

# ── Datos ──────────────────────────────────────────────────────────────────
Re_data = np.array([0.2, 2., 20., 200., 2000., 20000.])
cD_data = np.array([103., 13.9, 2.72, 0.800, 0.401, 0.433])
n       = len(Re_data)

lRe = np.log10(Re_data)
lcD = np.log10(cD_data)
h   = np.diff(lRe)   # all ≈ 1

# ── Sistema y momentos ────────────────────────────────────────────────────
A_mat = np.zeros((n-2, n-2))
rhs_v = np.zeros(n-2)
for k in range(1, n-1):
    ii = k-1
    if ii > 0:   A_mat[ii, ii-1] = h[k-2]
    A_mat[ii, ii] = 2*(h[k-2]+h[k-1])
    if ii < n-3: A_mat[ii, ii+1] = h[k-1]
    rhs_v[ii] = 6*((lcD[k+1]-lcD[k])/h[k-1] - (lcD[k]-lcD[k-1])/h[k-2])

M_inner = np.linalg.solve(A_mat, rhs_v)
M       = np.concatenate([[0.], M_inner, [0.]])

# ── Evaluación ────────────────────────────────────────────────────────────
def spline_log(lRe_val):
    i = min(max(np.searchsorted(lRe, lRe_val, 'right')-1, 0), n-2)
    hi = h[i]; xi,xi1 = lRe[i],lRe[i+1]
    Mi,Mi1 = M[i],M[i+1]; yi,yi1 = lcD[i],lcD[i+1]
    return (Mi/(6*hi)*(xi1-lRe_val)**3 + Mi1/(6*hi)*(lRe_val-xi)**3
           +(yi/hi-Mi*hi/6)*(xi1-lRe_val) +(yi1/hi-Mi1*hi/6)*(lRe_val-xi))

def spline_cD(Re_val):
    return 10**spline_log(np.log10(Re_val))

def seg_fn_log(i, lRe_arr):
    x  = np.atleast_1d(lRe_arr)
    hi = h[i]; xi,xi1 = lRe[i],lRe[i+1]
    Mi,Mi1 = M[i],M[i+1]; yi,yi1 = lcD[i],lcD[i+1]
    return (Mi/(6*hi)*(xi1-x)**3 + Mi1/(6*hi)*(x-xi)**3
           +(yi/hi-Mi*hi/6)*(xi1-x) +(yi1/hi-Mi1*hi/6)*(x-xi))

TARGETS    = [5., 50., 500., 5000.]
RESULTS    = {Re_t: spline_cD(Re_t) for Re_t in TARGETS}
seg_xr     = [np.linspace(lRe[i], lRe[i+1], 200) for i in range(n-1)]

# Curva continua log-log
lRe_r = np.linspace(lRe[0], lRe[-1], 600)
lcD_r = np.array([spline_log(lv) for lv in lRe_r])

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color": TXT_PRI, "axes.labelcolor": TXT_SEC,
    "xtick.color": TXT_SEC, "ytick.color": TXT_SEC, "font.size": 9,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.975,
         "Coeficiente de Arrastre  cD(Re)  —  Spline Natural (escala log-log)",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.18, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje (log-log)
ax.set_facecolor(BG_AX)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.15, 25000); ax.set_ylim(0.3, 200)
ax.set_xlabel("Re  (número de Reynolds)", fontsize=11)
ax.set_ylabel("cD  (coeficiente de arrastre)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=8.5, which="both")
ax.grid(True, color=GRID_CLR, lw=0.5, which="both", zorder=0)

# Líneas verticales de targets
for Re_t, tc in zip(TARGETS, TC):
    ax.axvline(Re_t, color=tc, lw=0.9, ls="--", alpha=0.7, zorder=2)
    ax.text(Re_t*1.05, 0.38, f"Re={Re_t:.0f}", fontsize=7.5, color=tc)

# Puntos de datos
ax.scatter(Re_data, cD_data, zorder=7, s=90, color=ACCENT,
           edgecolors="white", linewidths=1.8)
for ri, ci in zip(Re_data, cD_data):
    ax.annotate(f"({ri:.0f},{ci})", (ri, ci),
                textcoords="offset points", xytext=(6, 6),
                fontsize=7.5, color=ACCENT)

# Líneas animadas por tramo (en espacio log)
seg_lines = [ax.plot([], [], color=SEG_COLORS[i], lw=2.4, zorder=5)[0]
             for i in range(n-1)]
poly_line, = ax.plot([], [], color="#FFB74D", lw=2.8, zorder=6, alpha=0)

# Puntos de evaluación
target_dots = [ax.scatter([], [], s=150, color=tc, zorder=9,
                           edgecolors="white", linewidths=2, marker="*")
               for tc in TC]
target_txts = [ax.text(0, 0, "", fontsize=9, color=tc) for tc in TC]

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

ax.legend(
    handles=[mpatches.Patch(color=c, label=f"Tramo {i}")
             for i,c in enumerate(SEG_COLORS[:5])],
    loc="upper right", fontsize=8, framealpha=0.35,
    facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(done=False):
    axt.cla(); axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
    axt.set_xlim(0,1); axt.set_ylim(0,1); axt.set_axis_off()

    if not done:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara construir el spline",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle(); return

    axt.text(0.5, 0.975, "Spline natural en escala log-log",
             ha="center", va="top", fontsize=9.5,
             color=ACCENT, transform=axt.transAxes)

    r_h = 0.052; y = 0.915

    def sec_hdr(y_c, txt):
        rect = mpatches.FancyBboxPatch(
            (0.01, y_c-r_h*0.46), 0.98, r_h*0.92,
            boxstyle="round,pad=0.003", facecolor="#1A3A5C",
            edgecolor="none", transform=axt.transAxes,
            zorder=1, clip_on=False)
        axt.add_patch(rect)
        axt.text(0.5, y_c, txt, ha="center", va="center",
                 fontsize=7.8, color=ACCENT,
                 transform=axt.transAxes, zorder=3, clip_on=False)

    def row(y_c, lbl, val, clr=TXT_PRI, bold=False, bg=None):
        if bg:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_c-r_h*0.46), 0.96, r_h*0.92,
                boxstyle="round,pad=0.003", facecolor=bg,
                edgecolor="none", transform=axt.transAxes,
                zorder=1, clip_on=False)
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

    # 1. Transformación
    sec_hdr(y, "1 · Transformación  u=log₁₀(Re),  v=log₁₀(cD)")
    y -= r_h*1.2
    row(y, "Intervalos h:", "uniformes h = 1  en escala log₁₀"); y -= r_h
    row(y, "Condición:", "natural  M₀=M₅=0"); y -= r_h
    hline(y-r_h*0.2); y -= r_h*0.6

    # 2. Momentos
    sec_hdr(y, "2 · Momentos  Mᵢ = S''(uᵢ)"); y -= r_h*1.2
    m_lbls = ["M₀ (nat.)", "M₁", "M₂", "M₃", "M₄", "M₅ (nat.)"]
    for mi, (lbl, mv) in enumerate(zip(m_lbls, M)):
        is_nat = (mi==0 or mi==5)
        clr = "#FFF176" if is_nat else ACCENT
        bg  = "#0A2A50" if not is_nat else None
        row(y, lbl, f"{mv:.6f}", clr=clr, bold=not is_nat, bg=bg)
        y -= r_h
    hline(y-r_h*0.2); y -= r_h*0.6

    # 3. Evaluaciones
    sec_hdr(y, "3 · Resultados"); y -= r_h*1.2
    for Re_t, tc in zip(TARGETS, TC):
        cD_t = RESULTS[Re_t]
        lv   = np.log10(cD_t)
        bg   = "#0A2A50"
        rect = mpatches.FancyBboxPatch(
            (0.02, y-r_h*0.46), 0.96, r_h*0.92,
            boxstyle="round,pad=0.003", facecolor=bg,
            edgecolor="none", transform=axt.transAxes,
            zorder=1, clip_on=False)
        axt.add_patch(rect)
        axt.text(0.04, y, f"Re={Re_t:.0f}:",
                 ha="left", va="center", fontsize=7.5, color=tc,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.60, y, f"log cD={lv:.4f}",
                 ha="left", va="center", fontsize=7.5, color=TXT_PRI,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.96, y, f"cD={cD_t:.4f}",
                 ha="right", va="center", fontsize=8.5, color=tc,
                 fontweight="bold",
                 transform=axt.transAxes, zorder=3, clip_on=False)
        y -= r_h

    # Caja final
    y_box = y - r_h*0.6
    box = mpatches.FancyBboxPatch(
        (0.03, y_box-0.058), 0.94, 0.095,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    vals = [f"Re={int(Re_t):5d}: cD={RESULTS[Re_t]:.4f}" for Re_t in TARGETS]
    axt.text(0.5, y_box+0.020,
             f"cD(5) = {RESULTS[5.]:.4f}       cD(50) = {RESULTS[50.]:.4f}",
             ha="center", va="center", fontsize=9,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)
    axt.text(0.5, y_box-0.022,
             f"cD(500) = {RESULTS[500.]:.4f}     cD(5000) = {RESULTS[5000.]:.4f}",
             ha="center", va="center", fontsize=9,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
N_FRAMES   = 70
MSG_FRAMES = 28
state = {"running": False, "anim": None}

BUILD = [
    "Transformando a escala log₁₀: u=log(Re), v=log(cD)...",
    "h uniforme = 1  en escala log₁₀  (6 décadas)...",
    "Resolviendo sistema 4×4 para momentos M...",
    "Trazando tramo S₀  [Re: 0.2 → 2]...",
    "Trazando tramo S₁  [Re: 2 → 20]...",
    "Trazando tramo S₂  [Re: 20 → 200]...",
    "Trazando tramo S₃  [Re: 200 → 2000]...",
    "Trazando tramo S₄  [Re: 2000 → 20000]...",
    "Evaluando en Re = 5, 50, 500, 5000...",
]

def clear_all():
    for ln in seg_lines: ln.set_data([], [])
    poly_line.set_data([], []); poly_line.set_alpha(0)
    for d in target_dots: d.set_offsets(np.empty((0,2)))
    for t in target_txts: t.set_text("")
    status_txt.set_text("")
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]: return
    clear_all(); state["running"] = True

    n_msg = 3; n_seg = 5; n_fin = 1
    total = n_msg*MSG_FRAMES + n_seg*(MSG_FRAMES+N_FRAMES) + n_fin*MSG_FRAMES

    def update(frame):
        if frame < n_msg*MSG_FRAMES:
            ph = frame // MSG_FRAMES
            status_txt.set_text(BUILD[ph])
        else:
            f2 = frame - n_msg*MSG_FRAMES
            ph_len = MSG_FRAMES + N_FRAMES
            ph2 = f2 // ph_len
            loc = f2 % ph_len
            if ph2 < n_seg:
                status_txt.set_text(BUILD[n_msg+ph2])
                if loc >= MSG_FRAMES:
                    cf = loc - MSG_FRAMES
                    n_pts = max(2, int((cf+1)/N_FRAMES * len(seg_xr[ph2])))
                    lxp = seg_xr[ph2][:n_pts]
                    lyp = seg_fn_log(ph2, lxp)
                    seg_lines[ph2].set_data(10**lxp, 10**lyp)
            else:
                status_txt.set_text(BUILD[-1])
                for ti, (Re_t, tc, dot, ttxt) in enumerate(
                        zip(TARGETS, TC, target_dots, target_txts)):
                    cD_t = RESULTS[Re_t]
                    dot.set_offsets([[Re_t, cD_t]])
                    ttxt.set_text(f"  {cD_t:.3f}")
                    ttxt.set_x(Re_t*1.08); ttxt.set_y(cD_t*1.15)

        if frame == total - 1:
            status_txt.set_text(
                f"✓  cD: Re=5→{RESULTS[5.]:.3f}  "
                f"Re=50→{RESULTS[50.]:.3f}  "
                f"Re=500→{RESULTS[500.]:.3f}  "
                f"Re=5000→{RESULTS[5000.]:.3f}")
            draw_table(done=True)
            state["running"] = False
        return tuple(seg_lines)

    state["anim"] = FuncAnimation(
        fig, update, frames=total,
        interval=18, blit=False, repeat=False)
    fig.canvas.draw_idle()

def reset_animation(event=None):
    if state["anim"]:
        state["anim"].event_source.stop(); state["anim"] = None
    state["running"] = False
    clear_all(); fig.canvas.draw_idle()

# ── Widgets ────────────────────────────────────────────────────────────────
ax_play  = fig.add_axes([0.06, 0.07, 0.16, 0.07])
ax_reset = fig.add_axes([0.24, 0.07, 0.14, 0.07])
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