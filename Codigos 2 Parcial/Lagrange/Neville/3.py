"""
Máximo de y(x) usando Interpolación de Neville
Datos (Prob. 2): x = [0, 0.5, 1, 1.5, 2, 2.5, 3]
                 y = [1.8421, 2.4694, 2.4921, 1.9047, 0.8509, -0.4112, -1.5727]

Máximo conocido en x ≈ 0.7679  →  4 vecinos más cercanos: x = [0, 0.5, 1.0, 1.5]
Estrategia: construir tabla de Neville completa, luego minimizar −P(x) para hallar x*.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize_scalar

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
    "pts_all":  TXT_SEC,
    "pts_used": "#FFB74D",
    "q0":       "#FF7043",
    "q1":       "#66BB6A",
    "q2":       "#CE93D8",
    "q3":       "#4FC3F7",
    "poly":     "#FFB74D",
    "max_pt":   "#FF5252",
    "vline":    "#FF5252",
    "deriv":    "#80CBC4",
}

# ── Datos ──────────────────────────────────────────────────────────────────
xs_all = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
ys_all = np.array([1.8421, 2.4694, 2.4921, 1.9047, 0.8509, -0.4112, -1.5727])

# 4 vecinos más cercanos a x=0.7679
IDX = np.array([0, 1, 2, 3])
xs_s = xs_all[IDX]
ys_s = ys_all[IDX]

# ── Neville ────────────────────────────────────────────────────────────────
def neville_eval(x):
    """Evalúa P(x) con la tabla de Neville (4 puntos)."""
    n = len(xs_s)
    Q = ys_s.copy().astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            Q[i] = ((x - xs_s[i - j]) * Q[i] - (x - xs_s[i]) * Q[i - 1]) / \
                   (xs_s[i] - xs_s[i - j])
    return Q[-1]

def neville_table(x):
    """Devuelve la tabla completa Q[i,j] en x."""
    n = len(xs_s)
    Q = np.zeros((n, n))
    Q[:, 0] = ys_s.copy()
    for j in range(1, n):
        for i in range(j, n):
            Q[i, j] = ((x - xs_s[i - j]) * Q[i, j - 1] -
                       (x - xs_s[i]) * Q[i - 1, j - 1]) / \
                      (xs_s[i] - xs_s[i - j])
    return Q

# Encontrar máximo
res = minimize_scalar(lambda x: -neville_eval(x),
                      bounds=(0.3, 1.2), method="bounded")
x_max = res.x
y_max = -res.fun
Q_table = neville_table(x_max)

print(f"Máximo en x* = {x_max:.6f},  y* = {y_max:.6f}")

xr = np.linspace(-0.1, 1.8, 600)
yr = np.array([neville_eval(xi) for xi in xr])

# Derivada numérica (para mostrar tangente horizontal)
h = 1e-5
dyr = np.array([(neville_eval(xi + h) - neville_eval(xi - h)) / (2 * h) for xi in xr])

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
         "Máximo de y(x) por Interpolación de Neville  —  4 puntos vecinos",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1],
                      left=0.06, right=0.97,
                      bottom=0.18, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje gráfica
ax.set_facecolor(BG_AX)
ax.set_xlim(-0.2, 1.9); ax.set_ylim(1.4, 2.9)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y(x)", fontsize=11)
for sp in ax.spines.values():
    sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.6, zorder=1)

# Todos los datos (fondo gris, solo los 4 relevantes se resaltan después)
ax.scatter(xs_all, ys_all, zorder=3, s=55, color=C["pts_all"],
           edgecolors=BG_AX, linewidths=1.0)
for xi, yi in zip(xs_all, ys_all):
    if xi <= 1.8:
        ax.annotate(f"({xi:.1f},{yi:.4f})", (xi, yi),
                    textcoords="offset points", xytext=(5, 7),
                    fontsize=7.5, color=TXT_SEC)

# Puntos activos (4 vecinos)
scat_used = ax.scatter(xs_s, ys_s, zorder=5, s=90,
                       color=C["pts_used"],
                       edgecolors="white", linewidths=1.5)

# Líneas animadas
PHASE_COLORS = [C["q0"], C["q1"], C["q2"], C["q3"]]
# Q[:,0] son los puntos base (no se trazan como líneas)
# Fases: Q₀,₁  Q₁,₁  Q₂,₁ → Q₀,₂  Q₁,₂ → Q₀,₃ → Poly final
phase_lines = [ax.plot([], [], color=PHASE_COLORS[i % 4],
                       lw=1.5, ls="--", alpha=0.85)[0] for i in range(6)]
poly_line, = ax.plot([], [], color=C["poly"], lw=2.6, zorder=4)

# Punto del máximo
scat_max  = ax.scatter([], [], s=160, color=C["max_pt"], zorder=8,
                       edgecolors="white", linewidths=2, marker="*")
vline_max = ax.axvline(x=x_max, color=C["vline"], lw=1.0, ls=":", alpha=0, zorder=2)

# Tangente horizontal en el máximo
tang_line, = ax.plot([], [], color=C["deriv"], lw=1.2, ls="-.", alpha=0, zorder=3)

max_txt    = ax.text(0, 2.5, "", fontsize=9, color=C["max_pt"], ha="left")
status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

ax.legend(
    handles=[
        mpatches.Patch(color=C["pts_used"], label="4 puntos vecinos"),
        mpatches.Patch(color=C["q0"],       label="Q grado 1"),
        mpatches.Patch(color=C["q2"],       label="Q grado 2"),
        mpatches.Patch(color=C["q3"],       label="Q grado 3"),
        mpatches.Patch(color=C["poly"],     label="P(x) final"),
        mpatches.Patch(color=C["max_pt"],   label="Máximo x*"),
    ],
    loc="lower left", fontsize=8, framealpha=0.35,
    facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI
)

# Estilo panel tabla
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

    title = f"Tabla de Neville  en  x* = {x_max:.5f}"
    axt.text(0.5, 0.975, title, ha="center", va="top",
             fontsize=9.5, color=ACCENT, transform=axt.transAxes)

    # ── Tabla Q[i,j] ──────────────────────────────────────────────────────
    # Encabezados de columna
    col_headers = ["i", "xᵢ", "Q i,0", "Q i,1", "Q i,2", "Q i,3"]
    col_w = [0.06, 0.10, 0.16, 0.18, 0.18, 0.18]  # suma ~0.86; margen 0.07 c/lado
    n = 4
    n_rows = n + 1  # encabezado + 4 filas
    total_h = 0.50
    row_h   = total_h / (n_rows + 0.5)
    y_start = 0.88

    def draw_row(cells, y_c, is_hdr=False, highlight_col=-1):
        if is_hdr:
            bg = "#1A3A5C"
            rect = mpatches.FancyBboxPatch(
                (0.02, y_c - row_h*0.47), 0.96, row_h*0.94,
                boxstyle="round,pad=0.004",
                facecolor=bg, edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        x = 0.03
        for ci, (cell, cw) in enumerate(zip(cells, col_w)):
            if ci == highlight_col and not is_hdr:
                # caja de acento en la celda final
                box = mpatches.FancyBboxPatch(
                    (x + 0.005, y_c - row_h*0.42), cw - 0.01, row_h*0.84,
                    boxstyle="round,pad=0.003",
                    facecolor="#0D47A1", edgecolor=ACCENT, linewidth=0.8,
                    transform=axt.transAxes, zorder=2, clip_on=False)
                axt.add_patch(box)
                clr = "#FFFFFF"
            else:
                clr = TXT_SEC if is_hdr else TXT_PRI
            fw = "normal"
            fs = 7.5 if is_hdr else 8.5
            axt.text(x + cw/2, y_c, str(cell),
                     ha="center", va="center",
                     fontsize=fs, color=clr, fontweight=fw,
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    # Encabezado
    y_h = y_start - row_h*0.5
    draw_row(col_headers, y_h, is_hdr=True)
    axt.plot([0.02, 0.98], [y_h - row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    # Filas de datos
    for i in range(n):
        y_r = y_start - row_h*(i + 1.5)
        cells = [str(i), f"{xs_s[i]:.1f}"]
        for j in range(n):
            if j == 0:
                cells.append(f"{Q_table[i,0]:.4f}")
            elif i >= j:
                cells.append(f"{Q_table[i,j]:.4f}")
            else:
                cells.append("—")
        hi_col = n + 1 if i == n - 1 else -1  # última celda = resultado final
        draw_row(cells, y_r, highlight_col=hi_col)

    axt.plot([0.02, 0.98], [y_start - row_h*(n + 1.02)]*2,
             color=SPINE_CLR, lw=0.5, ls="--", transform=axt.transAxes, zorder=4)

    # ── Resultado numérico ─────────────────────────────────────────────────
    y_sec = y_start - row_h*(n + 1.4)
    sec_data = [
        ("x máximo",    f"x*  =  {x_max:.6f}"),
        ("y máximo",    f"y*  =  {y_max:.6f}"),
        ("P(x*) check", f"{neville_eval(x_max):.6f}"),
        ("P'(x*) ≈",    "0  (tangente horizontal)"),
    ]
    r_h2 = 0.085
    for ri, (lbl, val) in enumerate(sec_data):
        y_r = y_sec - r_h2*ri
        is_last = ri <= 1
        bg = "#0A2A50" if is_last else None
        if bg:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_r - r_h2*0.46), 0.96, r_h2*0.92,
                boxstyle="round,pad=0.004",
                facecolor=bg, edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        clr = ACCENT if is_last else TXT_SEC
        axt.text(0.05, y_r, lbl,
                 ha="left", va="center", fontsize=8,
                 color=TXT_SEC, transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.95, y_r, val,
                 ha="right", va="center", fontsize=8.5,
                 color=clr, fontweight="bold" if is_last else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    # Caja final
    y_box = y_sec - r_h2*(len(sec_data) + 0.3)
    box = mpatches.FancyBboxPatch(
        (0.05, y_box - 0.036), 0.90, 0.065,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box - 0.003,
             f"y_max  =  {y_max:.6f}   en   x* = {x_max:.5f}",
             ha="center", va="center", fontsize=10,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Fases de animación ─────────────────────────────────────────────────────
# Construiremos las curvas Q de Neville progresivamente:
# Fase 0-2: los 3 polinomios de grado 1  (Q[1,1], Q[2,1], Q[3,1])
# Fase 3-4: los 2 polinomios de grado 2  (Q[2,2], Q[3,2])
# Fase 5:   el polinomio de grado 3      (Q[3,3])  — curva final

def make_nev_curve(i_row, j_col):
    """Genera función que evalúa Q[i,j] como función de x (interpolación parcial)."""
    def fn(x_arr):
        result = []
        for xv in np.atleast_1d(x_arr):
            Q = ys_s.copy().astype(float)
            for jj in range(1, j_col + 1):
                for ii in range(len(xs_s) - 1, jj - 1, -1):
                    Q[ii] = ((xv - xs_s[ii - jj]) * Q[ii] -
                             (xv - xs_s[ii]) * Q[ii - 1]) / \
                            (xs_s[ii] - xs_s[ii - jj])
            result.append(Q[i_row])
        return np.array(result)
    return fn

PHASES = [
    ("Grado 1:  Q₁,₁(x)",  phase_lines[0], make_nev_curve(1, 1)),
    ("Grado 1:  Q₂,₁(x)",  phase_lines[1], make_nev_curve(2, 1)),
    ("Grado 1:  Q₃,₁(x)",  phase_lines[2], make_nev_curve(3, 1)),
    ("Grado 2:  Q₂,₂(x)",  phase_lines[3], make_nev_curve(2, 2)),
    ("Grado 2:  Q₃,₂(x)",  phase_lines[4], make_nev_curve(3, 2)),
    ("Grado 3:  Q₃,₃ = P(x)", poly_line,   lambda x: np.array([neville_eval(xi) for xi in np.atleast_1d(x)])),
]

N_FRAMES = 75
state = {"running": False, "anim": None}

def clear_all():
    for ln in phase_lines:
        ln.set_data([], [])
    poly_line.set_data([], [])
    scat_max.set_offsets(np.empty((0, 2)))
    vline_max.set_alpha(0)
    tang_line.set_data([], [])
    tang_line.set_alpha(0)
    max_txt.set_text("")
    status_txt.set_text("")
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]:
        return
    clear_all()
    state["running"] = True
    total = len(PHASES) * N_FRAMES

    def update(frame):
        ph_idx   = min(frame // N_FRAMES, len(PHASES) - 1)
        ph_frame = frame % N_FRAMES
        name, line, fn = PHASES[ph_idx]
        status_txt.set_text(name)
        n_pts = max(2, int((ph_frame + 1) / N_FRAMES * len(xr)))
        xp = xr[:n_pts]
        line.set_data(xp, fn(xp))

        if frame == total - 1:
            # Marcar máximo
            scat_max.set_offsets([[x_max, y_max]])
            vline_max.set_alpha(0.65)
            # Tangente horizontal
            tang_x = np.array([x_max - 0.25, x_max + 0.25])
            tang_line.set_data(tang_x, [y_max, y_max])
            tang_line.set_alpha(0.8)
            max_txt.set_text(f"  x* ≈ {x_max:.5f}\n  y* ≈ {y_max:.5f}")
            max_txt.set_x(x_max + 0.03)
            max_txt.set_y(y_max - 0.12)
            status_txt.set_text(f"✓  Máximo: y* = {y_max:.6f} en x* = {x_max:.5f}")
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