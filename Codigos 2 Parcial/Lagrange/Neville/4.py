"""
Interpolación de Neville en x = π/4
Datos: x = [0, 0.5, 1.0, 1.5, 2.0]
       y = [-1.00, 1.75, 4.00, 5.75, 7.00]
Construye la tabla completa Q[i,j] y evalúa P(π/4).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# ── Paleta oscura ──────────────────────────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"

C = {
    "pts":   "#FFB74D",
    "g1":    "#FF7043",
    "g2":    "#66BB6A",
    "g3":    "#CE93D8",
    "g4":    "#4FC3F7",
    "poly":  "#FFB74D",
    "vline": "#FF5252",
    "res":   "#FF5252",
}

# ── Datos ──────────────────────────────────────────────────────────────────
xs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
ys = np.array([-1.00, 1.75, 4.00, 5.75, 7.00])
x_target = np.pi / 4          # ≈ 0.785398
n = len(xs)

# ── Tabla de Neville completa en x_target ─────────────────────────────────
Q_full = np.zeros((n, n))
Q_full[:, 0] = ys.copy()
for j in range(1, n):
    for i in range(j, n):
        Q_full[i, j] = (
            (x_target - xs[i - j]) * Q_full[i,     j - 1] -
            (x_target - xs[i])     * Q_full[i - 1, j - 1]
        ) / (xs[i] - xs[i - j])

result = Q_full[n - 1, n - 1]

# ── Neville como función continua de x ────────────────────────────────────
def neville_fn(x_val):
    Q = ys.copy().astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            Q[i] = (
                (x_val - xs[i - j]) * Q[i] -
                (x_val - xs[i])     * Q[i - 1]
            ) / (xs[i] - xs[i - j])
    return Q[-1]

def make_partial(i_row, j_col):
    """Evalúa Q[i_row, j_col] como función de x (construcción parcial)."""
    def fn(x_arr):
        out = []
        for xv in np.atleast_1d(x_arr):
            Qtmp = ys.copy().astype(float)
            for jj in range(1, j_col + 1):
                for ii in range(n - 1, jj - 1, -1):
                    Qtmp[ii] = (
                        (xv - xs[ii - jj]) * Qtmp[ii] -
                        (xv - xs[ii])      * Qtmp[ii - 1]
                    ) / (xs[ii] - xs[ii - jj])
            out.append(Qtmp[i_row])
        return np.array(out)
    return fn

xr = np.linspace(-0.1, 2.2, 600)

# ── Figura ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.color":      TXT_PRI,
    "axes.labelcolor": TXT_SEC,
    "xtick.color":     TXT_SEC,
    "ytick.color":     TXT_SEC,
    "font.size":       9,
})

fig = plt.figure(figsize=(14, 8), facecolor=BG_FIG)
fig.text(
    0.5, 0.975,
    "Neville  —  P(π/4)  con 5 puntos de datos",
    ha="center", va="top", fontsize=12, color=ACCENT
)

gs = fig.add_gridspec(
    1, 2, width_ratios=[1.5, 1],
    left=0.06, right=0.97, bottom=0.18, top=0.94, wspace=0.06
)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje gráfica
ax.set_facecolor(BG_AX)
ax.set_xlim(-0.15, 2.25); ax.set_ylim(-2.5, 9.5)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y(x)", fontsize=11)
for sp in ax.spines.values():
    sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.8, zorder=1)

# Línea vertical x = π/4
ax.axvline(x_target, color=C["vline"], lw=1.1, ls=":", alpha=0.7, zorder=2)
ax.text(x_target + 0.04, -2.1, "x = π/4", fontsize=9, color=C["vline"])

# Puntos de datos
ax.scatter(xs, ys, zorder=5, s=90, color=C["pts"],
           edgecolors="white", linewidths=1.5)
for xi, yi in zip(xs, ys):
    ax.annotate(f"({xi:.1f}, {yi:.2f})", (xi, yi),
                textcoords="offset points", xytext=(6, 7),
                fontsize=8, color=C["pts"])

# Punto resultado
scat_res  = ax.scatter([], [], s=160, color=C["res"], zorder=8,
                       edgecolors="white", linewidths=2, marker="*")
res_txt   = ax.text(0, 0, "", fontsize=9, color=C["res"])
status_txt= ax.text(0.02, 0.97, "", transform=ax.transAxes,
                    fontsize=10, va="top", color=ACCENT)

# Líneas animadas: grado 1 (4 curvas) + grado 2 (3) + grado 3 (2) + grado 4 (1) + poly final
G_COLORS = [C["g1"], C["g2"], C["g3"], C["g4"]]
# Necesitamos líneas para: Q[1..4,1], Q[2..4,2], Q[3..4,3], Q[4,4]
# Total de curvas intermedias: 4+3+2+1 = 10 → + 1 poly final = 11
phase_lines = [ax.plot([], [], color="white", lw=1.3, ls="--", alpha=0.8)[0]
               for _ in range(10)]
poly_line,  = ax.plot([], [], color=C["poly"], lw=2.8, zorder=4)

ax.legend(
    handles=[
        mpatches.Patch(color=C["g1"],  label="Grado 1"),
        mpatches.Patch(color=C["g2"],  label="Grado 2"),
        mpatches.Patch(color=C["g3"],  label="Grado 3"),
        mpatches.Patch(color=C["g4"],  label="Grado 4"),
        mpatches.Patch(color=C["poly"],label="P(x) final"),
        mpatches.Patch(color=C["res"], label="P(π/4)"),
    ],
    loc="upper left", fontsize=8, framealpha=0.35,
    facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI
)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values():
    sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla de Neville ───────────────────────────────────────────────────────
def draw_table(done=False):
    axt.cla()
    axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values():
        sp.set_color(SPINE_CLR)
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.set_axis_off()

    if not done:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara ver resultados",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle()
        return

    axt.text(0.5, 0.975,
             f"Tabla Neville  en  x = π/4 ≈ {x_target:.5f}",
             ha="center", va="top", fontsize=9.5,
             color=ACCENT, transform=axt.transAxes)

    # Encabezados
    col_headers = ["i", "xᵢ", "Q i,0", "Q i,1", "Q i,2", "Q i,3", "Q i,4"]
    col_w = [0.055, 0.085, 0.135, 0.145, 0.145, 0.145, 0.145]

    n_rows  = n + 1      # encabezado + 5 filas de datos
    total_h = 0.50
    row_h   = total_h / (n_rows + 0.4)
    y_start = 0.89

    def draw_row(cells, y_c, is_hdr=False, hi_col=-1):
        if is_hdr:
            rect = mpatches.FancyBboxPatch(
                (0.01, y_c - row_h*0.47), 0.98, row_h*0.94,
                boxstyle="round,pad=0.003",
                facecolor="#1A3A5C", edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        x = 0.015
        for ci, (cell, cw) in enumerate(zip(cells, col_w)):
            if ci == hi_col and not is_hdr:
                box = mpatches.FancyBboxPatch(
                    (x + 0.004, y_c - row_h*0.42), cw - 0.008, row_h*0.84,
                    boxstyle="round,pad=0.003",
                    facecolor="#0D47A1", edgecolor=ACCENT, linewidth=0.8,
                    transform=axt.transAxes, zorder=2, clip_on=False)
                axt.add_patch(box)
                clr = "#FFFFFF"
            else:
                clr = TXT_SEC if is_hdr else TXT_PRI
            # color por grado de columna
            if not is_hdr and ci >= 3 and str(cell) != "—":
                grade_clr = [None, None, None, C["g1"], C["g2"], C["g3"], C["g4"]]
                clr = grade_clr[ci] if ci < len(grade_clr) else TXT_PRI
            if ci == hi_col and not is_hdr:
                clr = "#FFFFFF"
            fs = 7.0 if is_hdr else 8.0
            fw = "bold" if ci == hi_col and not is_hdr else "normal"
            axt.text(x + cw/2, y_c, str(cell),
                     ha="center", va="center",
                     fontsize=fs, color=clr, fontweight=fw,
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    # Encabezado
    y_h = y_start - row_h*0.5
    draw_row(col_headers, y_h, is_hdr=True)
    axt.plot([0.01, 0.99], [y_h - row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    # Filas de datos
    for i in range(n):
        y_r = y_start - row_h*(i + 1.5)
        hi_col = n + 1 if i == n - 1 else -1   # resaltar Q[4,4]
        cells = [str(i), f"{xs[i]:.1f}"]
        for j in range(n):
            cells.append(f"{Q_full[i,j]:.4f}" if i >= j else "—")
        draw_row(cells, y_r, hi_col=hi_col)

    axt.plot([0.01, 0.99], [y_start - row_h*(n + 1.0)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # Bloque de resultado
    y_sec  = y_start - row_h*(n + 1.5)
    items  = [
        ("x = π/4",         f"{x_target:.6f}"),
        ("P(π/4)",          f"{result:.6f}"),
        ("Grado polinomio",  f"{n-1}"),
    ]
    r_h2 = 0.075
    for ri, (lbl, val) in enumerate(items):
        y_r = y_sec - r_h2*ri
        is_key = ri == 1
        if is_key:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_r - r_h2*0.46), 0.96, r_h2*0.92,
                boxstyle="round,pad=0.003",
                facecolor="#0A2A50", edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.06, y_r, lbl,
                 ha="left", va="center", fontsize=8,
                 color=TXT_SEC, transform=axt.transAxes,
                 zorder=3, clip_on=False)
        axt.text(0.94, y_r, val,
                 ha="right", va="center",
                 fontsize=9 if is_key else 8,
                 color=ACCENT if is_key else TXT_PRI,
                 fontweight="bold" if is_key else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    # Caja final
    y_box = y_sec - r_h2*(len(items) + 0.4)
    box = mpatches.FancyBboxPatch(
        (0.04, y_box - 0.036), 0.92, 0.065,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box - 0.003,
             f"P(π/4)  =  {result:.6f}",
             ha="center", va="center", fontsize=11,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Fases de animación ─────────────────────────────────────────────────────
# Grado 1: Q[1,1] Q[2,1] Q[3,1] Q[4,1]
# Grado 2: Q[2,2] Q[3,2] Q[4,2]
# Grado 3: Q[3,3] Q[4,3]
# Grado 4: Q[4,4] = P(x)  → poly_line
phase_specs = []
li = 0
for j in range(1, n):
    for i in range(j, n):
        is_final = (j == n - 1 and i == n - 1)
        line = poly_line if is_final else phase_lines[li]
        if not is_final:
            line.set_color(G_COLORS[j - 1])
            li += 1
        ii, jj = i, j
        phase_specs.append((
            f"Grado {j}:  Q[{i},{j}]",
            line,
            make_partial(ii, jj),
            j
        ))

N_FRAMES = 60
state = {"running": False, "anim": None}

def clear_all():
    for ln in phase_lines:
        ln.set_data([], [])
    poly_line.set_data([], [])
    scat_res.set_offsets(np.empty((0, 2)))
    res_txt.set_text("")
    status_txt.set_text("")
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]:
        return
    clear_all()
    state["running"] = True
    total = len(phase_specs) * N_FRAMES

    def update(frame):
        ph_idx   = min(frame // N_FRAMES, len(phase_specs) - 1)
        ph_frame = frame % N_FRAMES
        name, line, fn, grade = phase_specs[ph_idx]
        status_txt.set_text(name)
        n_pts = max(2, int((ph_frame + 1) / N_FRAMES * len(xr)))
        xp = xr[:n_pts]
        line.set_data(xp, fn(xp))

        if frame == total - 1:
            y_res = float(neville_fn(x_target))
            scat_res.set_offsets([[x_target, y_res]])
            res_txt.set_text(f"  P(π/4) ≈ {y_res:.5f}")
            res_txt.set_x(x_target + 0.04)
            res_txt.set_y(y_res + 0.25)
            status_txt.set_text(f"✓  P(π/4) = {y_res:.6f}")
            draw_table(done=True)
            state["running"] = False
        return (line,)

    state["anim"] = FuncAnimation(
        fig, update, frames=total,
        interval=16, blit=False, repeat=False
    )
    fig.canvas.draw_idle()

def reset_animation(event=None):
    if state["anim"] is not None:
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
    for sp in wax.spines.values():
        sp.set_color(SPINE_CLR)

btn_play  = Button(ax_play,  "▶   Animar",
                   color="#0D2B55", hovercolor="#1A3A6E")
btn_reset = Button(ax_reset, "↺   Reset",
                   color="#1A1A3E", hovercolor="#2A2A5E")
for b in [btn_play, btn_reset]:
    b.label.set_color(TXT_PRI)
    b.label.set_fontsize(10)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)

draw_table(done=False)
plt.show()