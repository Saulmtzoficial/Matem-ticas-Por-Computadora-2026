"""
Diferencias divididas de Newton — determinar el grado del polinomio
Datos: x = [-2, 1, 4, -1, 3, -4]
       y = [-1, 2, 59, 4, 24, -53]

Los puntos se ordenan por x antes de construir la tabla.
El grado se detecta como el mayor orden k donde f[x0..xk] ≠ 0.
Polinomio resultante: P(x) = x³ − 2x + 3  (grado 3)
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
ZERO_CLR  = "#EF5350"
NONZ_CLR  = "#FFB74D"

# ── Datos (ordenados por x) ────────────────────────────────────────────────
xs_raw = np.array([-2,  1,  4, -1,  3, -4], dtype=float)
ys_raw = np.array([ -1,  2, 59,  4, 24, -53], dtype=float)
idx    = np.argsort(xs_raw)
xs     = xs_raw[idx]          # [-4, -2, -1, 1, 3, 4]
ys     = ys_raw[idx]          # [-53, -1, 4, 2, 24, 59]
n      = len(xs)

# ── Tabla de diferencias divididas completa ────────────────────────────────
DD = np.zeros((n, n))
DD[:, 0] = ys.copy()
for j in range(1, n):
    for i in range(j, n):
        DD[i, j] = (DD[i, j-1] - DD[i-1, j-1]) / (xs[i] - xs[i-j])

# Coeficientes de Newton (diagonal superior)
coeffs = np.array([DD[j, j] for j in range(n)])
degree = int(np.max(np.where(np.abs(coeffs) > 1e-8)[0]))

# ── Evaluación del polinomio ───────────────────────────────────────────────
def newton_eval(x):
    """Horner sobre los coeficientes de Newton."""
    result = coeffs[degree]
    for i in range(degree - 1, -1, -1):
        result = result * (x - xs[i]) + coeffs[i]
    return result

xr     = np.linspace(-5, 5, 600)
yr     = np.array([newton_eval(xi) for xi in xr])

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
         "Diferencias Divididas de Newton  —  Determinación del grado",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.18, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje
ax.set_facecolor(BG_AX)
ax.set_xlim(-5, 5); ax.set_ylim(-75, 75)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("P(x)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.8, zorder=1)
ax.axvline(0, color=SPINE_CLR, lw=0.6, zorder=1)

# Puntos de datos
ax.scatter(xs, ys, zorder=5, s=90, color=NONZ_CLR,
           edgecolors="white", linewidths=1.5)
for xi, yi in zip(xs, ys):
    ax.annotate(f"({xi:.0f},{yi:.0f})", (xi, yi),
                textcoords="offset points", xytext=(6, 7),
                fontsize=8, color=NONZ_CLR)

# Curva animada del polinomio
poly_line, = ax.plot([], [], color=ACCENT, lw=2.6, zorder=4,
                     label=f"P(x) = x³ − 2x + 3  (grado {degree})")

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)
degree_txt = ax.text(0.02, 0.03, "", transform=ax.transAxes,
                     fontsize=11, va="bottom", color=NONZ_CLR, fontweight="bold")

ax.legend(loc="upper left", fontsize=9, framealpha=0.35,
          facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(done=False, highlight_order=-1):
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

    # Encabezados
    col_h = ["i", "xᵢ", "f[·]", "ord 1", "ord 2", "ord 3", "ord 4", "ord 5"]
    col_w = [0.055, 0.075, 0.110, 0.120, 0.120, 0.120, 0.120, 0.120]

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
            # Color por columna (orden de diferencia)
            if is_hdr:
                clr = TXT_SEC
            elif ci == 0 or ci == 1:
                clr = TXT_SEC
            else:
                order = ci - 2   # col 2 = orden 0, col3 = orden 1, ...
                # Diagonal: coeficiente de Newton
                is_newton_coeff = False
                row_i = None
                # buscar si esta celda es la diagonal
                # cells[0] es el índice i
                try:
                    row_i = int(cells[0])
                    if order == row_i:
                        is_newton_coeff = True
                except Exception:
                    pass

                if val_str == "—":
                    clr = SPINE_CLR
                elif val_str in ["0.0000", "0.000", " 0.0000"]:
                    clr = ZERO_CLR
                elif is_newton_coeff and order <= degree:
                    clr = NONZ_CLR
                elif is_newton_coeff:
                    clr = ZERO_CLR
                else:
                    clr = TXT_PRI

            # Caja de acento en coeficientes de Newton no nulos
            try:
                row_i = int(cells[0])
                order = ci - 2
                is_ncoeff = (order == row_i) and not is_hdr and val_str != "—"
                if is_ncoeff and abs(float(val_str)) > 1e-8 and order <= degree:
                    box = mpatches.FancyBboxPatch(
                        (x + 0.003, y_c - row_h*0.42), cw - 0.006, row_h*0.84,
                        boxstyle="round,pad=0.002",
                        facecolor="#1A3A00", edgecolor="#66BB6A", linewidth=0.7,
                        transform=axt.transAxes, zorder=2, clip_on=False)
                    axt.add_patch(box)
                    clr = "#A5D6A7"
                elif is_ncoeff and (abs(float(val_str)) < 1e-8 or order > degree):
                    box = mpatches.FancyBboxPatch(
                        (x + 0.003, y_c - row_h*0.42), cw - 0.006, row_h*0.84,
                        boxstyle="round,pad=0.002",
                        facecolor="#3A0000", edgecolor=ZERO_CLR, linewidth=0.7,
                        transform=axt.transAxes, zorder=2, clip_on=False)
                    axt.add_patch(box)
                    clr = "#EF9A9A"
            except Exception:
                pass

            fs = 6.8 if is_hdr else 7.8
            axt.text(x + cw/2, y_c, val_str,
                     ha="center", va="center",
                     fontsize=fs, color=clr, fontweight="normal",
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    y_h = y_start - row_h*0.5
    draw_row(col_h, y_h, is_hdr=True)
    axt.plot([0.01, 0.99], [y_h - row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

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

    # Bloque de resultados
    y_sec = y_start - row_h*(n + 1.5)
    items = [
        ("Coeficientes (diagonal):",
         f"{coeffs[0]:.0f}, {coeffs[1]:.0f}, {coeffs[2]:.0f}, {coeffs[3]:.0f}, 0, 0"),
        ("Último orden ≠ 0:",  f"orden {degree}"),
        ("Grado del polinomio:", f"{degree}"),
        ("Forma estándar:",    "P(x) = x³ − 2x + 3"),
    ]
    r_h2 = 0.070
    for ri, (lbl, val) in enumerate(items):
        y_r = y_sec - r_h2*ri
        is_key = ri >= 2
        if is_key:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_r - r_h2*0.46), 0.96, r_h2*0.92,
                boxstyle="round,pad=0.003",
                facecolor="#0A2A50" if ri == 2 else "#0D1A00",
                edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.05, y_r, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_SEC,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        clr = (ACCENT if ri == 2 else
               "#A5D6A7" if ri == 3 else TXT_PRI)
        fw  = "bold" if is_key else "normal"
        axt.text(0.95, y_r, val, ha="right", va="center",
                 fontsize=8.5 if is_key else 7.5,
                 color=clr, fontweight=fw,
                 transform=axt.transAxes, zorder=3, clip_on=False)

    # Caja final
    y_box = y_sec - r_h2*(len(items) + 0.4)
    box = mpatches.FancyBboxPatch(
        (0.04, y_box - 0.038), 0.92, 0.068,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box - 0.004,
             f"Grado = {degree}   →   P(x) = x³ − 2x + 3",
             ha="center", va="center", fontsize=10,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
# Fase 1-5: construir columnas de la tabla (visual en texto → ya en draw_table)
# Fase 6: trazar la curva P(x) punto a punto
# Usamos una sola fase animada para la curva + aparición de tabla al final
N_FRAMES = 120
state = {"running": False, "anim": None}

# Pasos de construcción de la tabla (mensajes)
BUILD_MSGS = [
    "Ordenando puntos por x...",
    "Columna 0: valores f[xᵢ]...",
    "Columna 1: diferencias de orden 1...",
    "Columna 2: diferencias de orden 2...",
    "Columna 3: diferencias de orden 3...",
    "Columna 4: diferencias de orden 4 → 0",
    "Columna 5: diferencias de orden 5 → 0",
    f"Diagonal: grado = {degree}  →  trazando P(x)...",
]

def clear_all():
    poly_line.set_data([], [])
    status_txt.set_text("")
    degree_txt.set_text("")
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]: return
    clear_all()
    state["running"] = True

    # Fases: mensajes de construcción (fijos, sin curva) + trazar curva
    n_msg_phases = len(BUILD_MSGS)
    msg_frames   = 28          # frames por mensaje
    curve_frames = N_FRAMES    # frames para dibujar la curva

    total_frames = n_msg_phases * msg_frames + curve_frames

    def update(frame):
        if frame < n_msg_phases * msg_frames:
            ph = frame // msg_frames
            status_txt.set_text(BUILD_MSGS[ph])
        else:
            cf = frame - n_msg_phases * msg_frames
            status_txt.set_text(f"Trazando P(x) = x³ − 2x + 3  (grado {degree})...")
            n_pts = max(2, int((cf + 1) / curve_frames * len(xr)))
            poly_line.set_data(xr[:n_pts], yr[:n_pts])

        if frame == total_frames - 1:
            degree_txt.set_text(
                f"Grado detectado: {degree}   →   f[x₀..x₃] = {coeffs[degree]:.0f} ≠ 0,   "
                f"f[x₀..x₄] = {coeffs[degree+1]:.0f}")
            status_txt.set_text(f"✓  Polinomio de grado {degree}  →  P(x) = x³ − 2x + 3")
            draw_table(done=True)
            state["running"] = False
        return (poly_line,)

    state["anim"] = FuncAnimation(
        fig, update, frames=total_frames,
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