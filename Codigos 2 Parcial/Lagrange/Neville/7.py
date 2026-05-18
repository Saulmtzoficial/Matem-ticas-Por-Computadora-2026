"""
Método de Newton — Diferencias Divididas
Datos: x = [-3, 2, -1, 3, 1]
       y = [0, 5, -4, 12, 0]

Ordenados: x = [-3, -1, 1, 2, 3], y = [0, -4, 0, 5, 12]
Resultado: P(x) = x² + 2x − 3  (grado 2)
           = (x+3)(x−1)
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
NONZ_CLR  = "#FFB74D"
ZERO_CLR  = "#EF5350"
GREEN_CLR = "#A5D6A7"

# ── Datos (ordenados por x) ────────────────────────────────────────────────
xs_raw = np.array([-3,  2, -1,  3,  1], dtype=float)
ys_raw = np.array([ 0,  5, -4, 12,  0], dtype=float)
idx    = np.argsort(xs_raw)
xs     = xs_raw[idx]    # [-3, -1, 1, 2, 3]
ys     = ys_raw[idx]    # [ 0, -4, 0, 5, 12]
n      = len(xs)

# ── Tabla de diferencias divididas ────────────────────────────────────────
DD = np.zeros((n, n))
DD[:, 0] = ys.copy()
for j in range(1, n):
    for i in range(j, n):
        DD[i, j] = (DD[i, j-1] - DD[i-1, j-1]) / (xs[i] - xs[i-j])

coeffs = np.array([DD[j, j] for j in range(n)])
degree = int(max(k for k, c in enumerate(coeffs) if abs(c) > 1e-9))

# ── Evaluación del polinomio ───────────────────────────────────────────────
def newton_eval(x):
    r = coeffs[degree]
    for i in range(degree - 1, -1, -1):
        r = r * (x - xs[i]) + coeffs[i]
    return r

xr = np.linspace(-4.2, 4.2, 600)
yr = np.array([newton_eval(xi) for xi in xr])

# Raíces del polinomio x²+2x-3 = (x+3)(x-1)
roots = np.array([-3.0, 1.0])

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
         "Método de Newton  —  Diferencias Divididas  →  Polinomio interpolante",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.18, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje gráfica
ax.set_facecolor(BG_AX)
ax.set_xlim(-4.3, 4.3); ax.set_ylim(-8, 16)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("P(x)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.9, zorder=1)
ax.axvline(0, color=SPINE_CLR, lw=0.6, zorder=1)

# Puntos de datos
ax.scatter(xs, ys, zorder=5, s=90, color=NONZ_CLR,
           edgecolors="white", linewidths=1.5)
for xi, yi in zip(xs, ys):
    ax.annotate(f"({xi:.0f},{yi:.0f})", (xi, yi),
                textcoords="offset points", xytext=(6, 7),
                fontsize=8.5, color=NONZ_CLR)

# Marcadores de raíces
ax.scatter(roots, [0, 0], zorder=6, s=100, color=ZERO_CLR,
           edgecolors="white", linewidths=1.5, marker="D")
for r in roots:
    ax.annotate(f"x={r:.0f}", (r, 0),
                textcoords="offset points", xytext=(6, -14),
                fontsize=8, color=ZERO_CLR)

# Curva animada
poly_line, = ax.plot([], [], color=ACCENT, lw=2.6, zorder=4,
                     label=f"P(x) = x² + 2x − 3  (grado {degree})")
status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)
result_txt = ax.text(0.02, 0.04, "", transform=ax.transAxes,
                     fontsize=10, va="bottom", color=NONZ_CLR, fontweight="bold")

ax.legend(loc="upper left", fontsize=9, framealpha=0.35,
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

    axt.text(0.5, 0.975,
             "Tabla de diferencias divididas",
             ha="center", va="top", fontsize=10,
             color=ACCENT, transform=axt.transAxes)

    # Columnas: i | xᵢ | ord0 | ord1 | ord2 | ord3 | ord4
    col_h = ["i", "xᵢ", "ord 0", "ord 1", "ord 2", "ord 3", "ord 4"]
    col_w = [0.055, 0.075, 0.130, 0.130, 0.130, 0.130, 0.130]

    n_rows  = n + 1
    total_h = 0.48
    row_h   = total_h / (n_rows + 0.4)
    y_start = 0.89

    def draw_row(cells, y_c, is_hdr=False):
        if is_hdr:
            rect = mpatches.FancyBboxPatch(
                (0.01, y_c - row_h*0.47), 0.98, row_h*0.94,
                boxstyle="round,pad=0.003",
                facecolor="#1A3A5C", edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        x = 0.015
        for ci, (cell, cw) in enumerate(zip(cells, col_w)):
            val_str = str(cell)
            clr = TXT_SEC if is_hdr else TXT_PRI

            if not is_hdr and ci >= 2:
                order = ci - 2
                try:
                    row_i = int(cells[0])
                    is_diag = (order == row_i)
                    if val_str == "—":
                        clr = SPINE_CLR
                    elif is_diag:
                        try:
                            fv = float(val_str)
                            if abs(fv) < 1e-8:
                                # caja roja: coeff nulo
                                box = mpatches.FancyBboxPatch(
                                    (x+0.003, y_c-row_h*0.42), cw-0.006, row_h*0.84,
                                    boxstyle="round,pad=0.002",
                                    facecolor="#3A0000", edgecolor=ZERO_CLR,
                                    linewidth=0.7, transform=axt.transAxes,
                                    zorder=2, clip_on=False)
                                axt.add_patch(box)
                                clr = "#EF9A9A"
                            else:
                                # caja verde: coeff activo
                                box = mpatches.FancyBboxPatch(
                                    (x+0.003, y_c-row_h*0.42), cw-0.006, row_h*0.84,
                                    boxstyle="round,pad=0.002",
                                    facecolor="#1A3A00", edgecolor="#66BB6A",
                                    linewidth=0.7, transform=axt.transAxes,
                                    zorder=2, clip_on=False)
                                axt.add_patch(box)
                                clr = GREEN_CLR
                        except Exception:
                            pass
                except Exception:
                    pass

            fs = 6.8 if is_hdr else 8.0
            axt.text(x + cw/2, y_c, val_str,
                     ha="center", va="center", fontsize=fs,
                     color=clr, fontweight="normal",
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    # Encabezado
    y_h = y_start - row_h*0.5
    draw_row(col_h, y_h, is_hdr=True)
    axt.plot([0.01, 0.99], [y_h - row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    # Filas
    for i in range(n):
        y_r = y_start - row_h*(i + 1.5)
        cells = [str(i), f"{xs[i]:.0f}"]
        for j in range(n):
            if i >= j:
                v = DD[i, j]
                cells.append(f"{v:.4f}" if abs(v) >= 0.00005 else "0.0000")
            else:
                cells.append("—")
        draw_row(cells, y_r)

    axt.plot([0.01, 0.99], [y_start - row_h*(n + 1.0)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # ── Bloque de resultados ───────────────────────────────────────────────
    y_sec = y_start - row_h*(n + 1.5)

    # Forma de Newton
    axt.text(0.5, y_sec,
             "Forma de Newton:",
             ha="center", va="center", fontsize=8,
             color=TXT_SEC, transform=axt.transAxes, zorder=3, clip_on=False)
    axt.text(0.5, y_sec - 0.06,
             "P(x) = 0 − 2(x+3) + 1·(x+3)(x+1)",
             ha="center", va="center", fontsize=8.5,
             color=TXT_PRI, transform=axt.transAxes, zorder=3, clip_on=False)

    items = [
        ("Coeficientes Newton:", f"0, −2, 1, 0, 0"),
        ("Último orden ≠ 0:",    f"orden {degree}  →  grado {degree}"),
        ("Raíces:",               "x = −3  y  x = 1"),
        ("Forma factorizada:",   "P(x) = (x+3)(x−1)"),
    ]
    r_h2 = 0.070
    y_items = y_sec - 0.13
    for ri, (lbl, val) in enumerate(items):
        y_r = y_items - r_h2*ri
        is_key = ri >= 2
        if is_key:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_r - r_h2*0.46), 0.96, r_h2*0.92,
                boxstyle="round,pad=0.003",
                facecolor="#0D1A00" if ri == 3 else "#0A2A50",
                edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.05, y_r, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_SEC,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        clr = (GREEN_CLR if ri == 3 else
               ACCENT if ri == 2 else TXT_PRI)
        axt.text(0.95, y_r, val, ha="right", va="center",
                 fontsize=8.5 if is_key else 8,
                 color=clr, fontweight="bold" if is_key else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    # Caja final
    y_box = y_items - r_h2*(len(items) + 0.5)
    box = mpatches.FancyBboxPatch(
        (0.04, y_box - 0.038), 0.92, 0.068,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box - 0.004,
             "P(x)  =  x²  +  2x  −  3",
             ha="center", va="center", fontsize=11,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
BUILD_MSGS = [
    "Ordenando puntos: x = [−3, −1, 1, 2, 3]...",
    "Columna 0 — valores f[xᵢ] = yᵢ...",
    "Columna 1 — diferencias divididas orden 1...",
    "Columna 2 — diferencias divididas orden 2...",
    "Columna 3 — diferencias orden 3  →  0.0000",
    "Columna 4 — diferencias orden 4  →  0.0000",
    f"Diagonal: grado = {degree}  →  P(x) = x² + 2x − 3",
]

N_FRAMES   = 110
MSG_FRAMES = 30
state = {"running": False, "anim": None}

def clear_all():
    poly_line.set_data([], [])
    status_txt.set_text("")
    result_txt.set_text("")
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]: return
    clear_all()
    state["running"] = True

    n_msg   = len(BUILD_MSGS)
    total   = n_msg * MSG_FRAMES + N_FRAMES

    def update(frame):
        if frame < n_msg * MSG_FRAMES:
            ph = frame // MSG_FRAMES
            status_txt.set_text(BUILD_MSGS[ph])
        else:
            cf = frame - n_msg * MSG_FRAMES
            status_txt.set_text("Trazando  P(x) = x² + 2x − 3 ...")
            n_pts = max(2, int((cf + 1) / N_FRAMES * len(xr)))
            poly_line.set_data(xr[:n_pts], yr[:n_pts])

        if frame == total - 1:
            result_txt.set_text(
                f"Grado = {degree}   →   P(x) = x² + 2x − 3 = (x+3)(x−1)")
            status_txt.set_text(
                f"✓  Polinomio de grado {degree} verificado en los 5 puntos")
            draw_table(done=True)
            state["running"] = False
        return (poly_line,)

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