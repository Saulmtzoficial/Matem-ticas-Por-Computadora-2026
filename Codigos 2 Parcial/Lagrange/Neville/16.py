"""
Interpolación de Neville — y en x = 0.46
Datos: x = [0, 0.0204, 0.1055, 0.241, 0.582, 0.712, 0.981]
       y = [0.385, 1.04, 1.79, 2.63, 4.39, 4.99, 5.27]

Método: Neville con los 7 puntos completos
Resultado: y(0.46) ≈ 4.8785
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
G_COLORS  = ["#FF7043","#66BB6A","#CE93D8","#FFB74D","#80CBC4","#F48FB1"]

# ── Datos ──────────────────────────────────────────────────────────────────
xs      = np.array([0.0, 0.0204, 0.1055, 0.241, 0.582, 0.712, 0.981])
ys      = np.array([0.385, 1.04, 1.79, 2.63, 4.39, 4.99, 5.27])
n       = len(xs)
X_EVAL  = 0.46

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

Q_full  = neville_table(xs, ys, X_EVAL)
result  = Q_full[-1, -1]

def neville_eval(x_val):
    return neville_table(xs, ys, x_val)[-1, -1]

xr  = np.linspace(-0.02, 1.05, 600)
yr  = np.array([neville_eval(xi) for xi in xr])

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color": TXT_PRI, "axes.labelcolor": TXT_SEC,
    "xtick.color": TXT_SEC, "ytick.color": TXT_SEC, "font.size": 9,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(0.5, 0.975,
         "Neville  —  y(x = 0.46)  con 7 puntos de datos",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1],
                      left=0.06, right=0.97,
                      bottom=0.18, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje gráfica
ax.set_facecolor(BG_AX)
ax.set_xlim(-0.04, 1.08); ax.set_ylim(-0.2, 6.5)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y(x)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.6, zorder=1)
ax.axvline(0, color=SPINE_CLR, lw=0.5, zorder=1)

# Línea de evaluación
ax.axvline(X_EVAL, color="#FF5252", lw=1.1, ls="--", alpha=0.8, zorder=2)
ax.text(X_EVAL + 0.012, 0.25, f"x = {X_EVAL}", fontsize=8.5, color="#FF5252")

# Puntos de datos
ax.scatter(xs, ys, zorder=6, s=90, color=ACCENT,
           edgecolors="white", linewidths=1.8)
for xi, yi in zip(xs, ys):
    off = (6, 7) if xi < 0.5 else (-60, 7)
    ax.annotate(f"({xi:.4f},{yi:.3f})", (xi, yi),
                textcoords="offset points", xytext=off,
                fontsize=7.5, color=ACCENT)

# Curva y punto animados
poly_line, = ax.plot([], [], color="#FFB74D", lw=2.6, zorder=5,
                     label="P(x)  Neville (7 pts)")
eval_dot   = ax.scatter([], [], s=160, color="#FF5252", zorder=8,
                        edgecolors="white", linewidths=2, marker="*")
eval_hline, = ax.plot([], [], color="#FF5252", lw=0.9, ls=":", alpha=0, zorder=3)
eval_txt   = ax.text(X_EVAL + 0.012, 0, "", fontsize=9, color="#FF5252")

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

ax.legend(loc="upper left", fontsize=8.5, framealpha=0.35,
          facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(done=False):
    axt.cla(); axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
    axt.set_xlim(0, 1); axt.set_ylim(0, 1); axt.set_axis_off()

    if not done:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara construir la tabla",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle(); return

    axt.text(0.5, 0.975,
             f"Tabla Neville  en  x = {X_EVAL}",
             ha="center", va="top", fontsize=9.5,
             color="#FF5252", transform=axt.transAxes)

    # Columnas: xᵢ | Q[i,0] | Q[i,1] | ... | Q[i,6]
    MAX_COL = 5   # mostrar hasta orden 4 (tabla demasiado ancha para 7)
    col_labels = ["xᵢ", "ord 0"] + [f"ord {j}" for j in range(1, MAX_COL)]
    col_w      = [0.10, 0.14] + [0.16]*(MAX_COL - 1)

    n_rows  = n + 1
    total_h = 0.55
    row_h   = total_h / (n_rows + 0.4)
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
                    facecolor="#0D47A1", edgecolor="#FF5252", linewidth=0.8,
                    transform=axt.transAxes, zorder=2, clip_on=False)
                axt.add_patch(box); clr = "#FFFFFF"
            elif not is_hdr and ci >= 2:
                clr = G_COLORS[min(ci-2, len(G_COLORS)-1)]
            else: clr = TXT_PRI
            fs = 6.5 if is_hdr else 7.8
            fw = "bold" if ci == hi_col and not is_hdr else "normal"
            axt.text(x+cw/2, y_c, val, ha="center", va="center",
                     fontsize=fs, color=clr, fontweight=fw,
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    y_h = y_start - row_h*0.5
    draw_row(col_labels, y_h, is_hdr=True)
    axt.plot([0.01, 0.99], [y_h-row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    for i in range(n):
        y_r = y_start - row_h*(i+1.5)
        hi  = MAX_COL if i == n-1 else -1
        cells = [f"{xs[i]:.4f}"]
        for j in range(MAX_COL):
            if i >= j:
                cells.append(f"{Q_full[i,j]:.4f}")
            else:
                cells.append("—")
        draw_row(cells, y_r, hi_col=hi)

    axt.plot([0.01, 0.99], [y_start-row_h*(n+1.0)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # Nota sobre convergencia
    y_conv = y_start - row_h*(n+1.6)
    diag   = [round(Q_full[i,i],4) for i in range(n)]
    axt.text(0.5, y_conv,
             "Diagonal (convergencia):",
             ha="center", va="center", fontsize=7.5, color=TXT_SEC,
             transform=axt.transAxes, zorder=3, clip_on=False)
    y_conv -= 0.055
    axt.text(0.5, y_conv,
             f"{diag[-3]:.4f} → {diag[-2]:.4f} → {diag[-1]:.4f}",
             ha="center", va="center", fontsize=8, color="#FFB74D",
             transform=axt.transAxes, zorder=3, clip_on=False)

    # Caja final
    y_box = y_conv - 0.09
    box = mpatches.FancyBboxPatch(
        (0.04, y_box-0.042), 0.92, 0.074,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor="#FF5252", linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box-0.004,
             f"y(0.46)  =  {result:.6f}",
             ha="center", va="center", fontsize=12,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
N_FRAMES   = 110
MSG_FRAMES = 28
state = {"running": False, "anim": None}

BUILD = [
    "7 puntos con espaciado no uniforme — método de Neville...",
    "Construyendo tabla Q[i,j] de 7×7...",
    "Trazando polinomio interpolante P(x)...",
    f"Evaluando en x = {X_EVAL}  →  y ≈ {result:.4f}...",
]

def clear_all():
    poly_line.set_data([], [])
    eval_dot.set_offsets(np.empty((0,2)))
    eval_hline.set_data([], []); eval_hline.set_alpha(0)
    eval_txt.set_text("")
    status_txt.set_text("")
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]: return
    clear_all(); state["running"] = True

    n_msg = 2
    total = n_msg*MSG_FRAMES + N_FRAMES + 2*MSG_FRAMES

    def update(frame):
        if frame < n_msg*MSG_FRAMES:
            ph = frame // MSG_FRAMES
            status_txt.set_text(BUILD[ph])
        elif frame < n_msg*MSG_FRAMES + N_FRAMES:
            cf = frame - n_msg*MSG_FRAMES
            status_txt.set_text(BUILD[2])
            n_pts = max(2, int((cf+1)/N_FRAMES * len(xr)))
            poly_line.set_data(xr[:n_pts], yr[:n_pts])
        elif frame < n_msg*MSG_FRAMES + N_FRAMES + MSG_FRAMES:
            status_txt.set_text(BUILD[3])
            eval_dot.set_offsets([[X_EVAL, result]])
            eval_hline.set_data([-0.04, X_EVAL], [result, result])
            eval_hline.set_alpha(0.65)
            eval_txt.set_text(f"y ≈ {result:.4f}")
            eval_txt.set_y(result + 0.15)
        else:
            pass

        if frame == total - 1:
            status_txt.set_text(f"✓  y(0.46) = {result:.6f}")
            draw_table(done=True)
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