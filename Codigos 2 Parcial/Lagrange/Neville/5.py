"""
Interpolación de Neville — y en x = π/4  y  x = π/2
Datos: x = [0, 0.5, 1.0, 1.5, 2.0]
       y = [-0.7854, 0.6529, 1.7390, 2.2071, 1.9425]
Método elegido: Neville (tabla de diferencias divididas iterativa)
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

C = {
    "pts":  "#FFB74D",
    "g1":   "#FF7043",
    "g2":   "#66BB6A",
    "g3":   "#CE93D8",
    "g4":   "#4FC3F7",
    "poly": "#FFB74D",
    "t1":   "#FF5252",   # marcador π/4
    "t2":   "#80CBC4",   # marcador π/2
}

# ── Datos ──────────────────────────────────────────────────────────────────
xs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
ys = np.array([-0.7854, 0.6529, 1.7390, 2.2071, 1.9425])
n  = len(xs)

X_PI4 = np.pi / 4    # ≈ 0.7854
X_PI2 = np.pi / 2    # ≈ 1.5708

TARGETS = {
    "pi4": {"x": X_PI4, "label": "π/4", "color": C["t1"]},
    "pi2": {"x": X_PI2, "label": "π/2", "color": C["t2"]},
}

# ── Neville ────────────────────────────────────────────────────────────────
def build_Q(x_val):
    Q = np.zeros((n, n))
    Q[:, 0] = ys.copy()
    for j in range(1, n):
        for i in range(j, n):
            Q[i, j] = (
                (x_val - xs[i - j]) * Q[i,     j - 1] -
                (x_val - xs[i])     * Q[i - 1, j - 1]
            ) / (xs[i] - xs[i - j])
    return Q

Q_PI4 = build_Q(X_PI4)
Q_PI2 = build_Q(X_PI2)
RES   = {"pi4": Q_PI4[-1, -1], "pi2": Q_PI2[-1, -1]}

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
            return_val = Qtmp[i_row]
            out.append(return_val)
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
fig.text(0.5, 0.975,
         "Neville  —  y(π/4)  y  y(π/2)  con 5 puntos",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1],
                      left=0.06, right=0.97,
                      bottom=0.19, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje
ax.set_facecolor(BG_AX)
ax.set_xlim(-0.15, 2.25); ax.set_ylim(-1.6, 3.2)
ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y(x)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.8, zorder=1)

# Líneas verticales de los dos objetivos
vl_pi4 = ax.axvline(X_PI4, color=C["t1"], lw=1.0, ls=":", alpha=0.7, zorder=2)
vl_pi2 = ax.axvline(X_PI2, color=C["t2"], lw=1.0, ls=":", alpha=0.7, zorder=2)
ax.text(X_PI4 + 0.03, -1.45, "π/4", fontsize=8.5, color=C["t1"])
ax.text(X_PI2 + 0.03, -1.45, "π/2", fontsize=8.5, color=C["t2"])

# Puntos de datos
ax.scatter(xs, ys, zorder=5, s=90, color=C["pts"],
           edgecolors="white", linewidths=1.5)
for xi, yi in zip(xs, ys):
    ax.annotate(f"({xi:.1f},{yi:.4f})", (xi, yi),
                textcoords="offset points", xytext=(6, 7),
                fontsize=7.5, color=C["pts"])

# Puntos resultado (uno por target)
scat_pi4 = ax.scatter([], [], s=160, color=C["t1"], zorder=8,
                      edgecolors="white", linewidths=2, marker="*")
scat_pi2 = ax.scatter([], [], s=160, color=C["t2"], zorder=8,
                      edgecolors="white", linewidths=2, marker="*")
txt_pi4  = ax.text(0, 0, "", fontsize=8.5, color=C["t1"])
txt_pi2  = ax.text(0, 0, "", fontsize=8.5, color=C["t2"])
status_t = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                   fontsize=10, va="top", color=ACCENT)

# Líneas animadas (10 fases + poly)
G_COLORS = [C["g1"], C["g2"], C["g3"], C["g4"]]
phase_lines = [ax.plot([], [], color="white", lw=1.3, ls="--", alpha=0.8)[0]
               for _ in range(10)]
poly_line,  = ax.plot([], [], color=C["poly"], lw=2.8, zorder=4)

ax.legend(
    handles=[
        mpatches.Patch(color=C["g1"],  label="Grado 1"),
        mpatches.Patch(color=C["g2"],  label="Grado 2"),
        mpatches.Patch(color=C["g3"],  label="Grado 3"),
        mpatches.Patch(color=C["g4"],  label="Grado 4  = P(x)"),
        mpatches.Patch(color=C["t1"],  label="y(π/4)"),
        mpatches.Patch(color=C["t2"],  label="y(π/2)"),
    ],
    loc="upper left", fontsize=8, framealpha=0.35,
    facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI
)

axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Tabla ──────────────────────────────────────────────────────────────────
def draw_table(mode):
    axt.cla()
    axt.set_facecolor(BG_TABLE)
    for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
    axt.set_xlim(0, 1); axt.set_ylim(0, 1)
    axt.set_axis_off()

    if mode is None:
        axt.text(0.5, 0.52,
                 "Presiona  ▶  Animar\npara ver resultados",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle()
        return

    tgt   = TARGETS[mode]
    Q     = Q_PI4 if mode == "pi4" else Q_PI2
    res_v = RES[mode]

    axt.text(0.5, 0.975,
             f"Tabla Neville  —  x = {tgt['label']} ≈ {tgt['x']:.5f}",
             ha="center", va="top", fontsize=9.5,
             color=tgt["color"], transform=axt.transAxes)

    col_h = ["i", "xᵢ", "Q i,0", "Q i,1", "Q i,2", "Q i,3", "Q i,4"]
    col_w = [0.055, 0.085, 0.135, 0.145, 0.145, 0.145, 0.145]

    n_rows  = n + 1
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
                    facecolor="#0D47A1", edgecolor=tgt["color"], linewidth=0.8,
                    transform=axt.transAxes, zorder=2, clip_on=False)
                axt.add_patch(box)
                clr = "#FFFFFF"
            elif not is_hdr and ci >= 3 and str(cell) != "—":
                clr = G_COLORS[ci - 3] if ci - 3 < len(G_COLORS) else TXT_PRI
            else:
                clr = TXT_SEC if is_hdr else TXT_PRI
            if ci == hi_col and not is_hdr:
                clr = "#FFFFFF"
            fs = 7.0 if is_hdr else 8.0
            fw = "bold" if ci == hi_col and not is_hdr else "normal"
            axt.text(x + cw/2, y_c, str(cell),
                     ha="center", va="center",
                     fontsize=fs, color=clr, fontweight=fw,
                     transform=axt.transAxes, zorder=3, clip_on=False)
            x += cw

    y_h = y_start - row_h*0.5
    draw_row(col_h, y_h, is_hdr=True)
    axt.plot([0.01, 0.99], [y_h - row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    for i in range(n):
        y_r   = y_start - row_h*(i + 1.5)
        hi    = n + 1 if i == n - 1 else -1
        cells = [str(i), f"{xs[i]:.1f}"]
        for j in range(n):
            cells.append(f"{Q[i,j]:.4f}" if i >= j else "—")
        draw_row(cells, y_r, hi_col=hi)

    axt.plot([0.01, 0.99], [y_start - row_h*(n + 1.0)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # Bloque resultado
    y_sec = y_start - row_h*(n + 1.5)
    items = [
        ("Método",      "Neville (grado 4)"),
        (f"x = {tgt['label']}",  f"{tgt['x']:.6f}"),
        ("P(x)",        f"{res_v:.6f}"),
    ]
    r_h2 = 0.075
    for ri, (lbl, val) in enumerate(items):
        y_r   = y_sec - r_h2*ri
        is_key = ri == 2
        if is_key:
            rect = mpatches.FancyBboxPatch(
                (0.02, y_r - r_h2*0.46), 0.96, r_h2*0.92,
                boxstyle="round,pad=0.003",
                facecolor="#0A2A50", edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        axt.text(0.06, y_r, lbl, ha="left", va="center",
                 fontsize=8, color=TXT_SEC,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.94, y_r, val, ha="right", va="center",
                 fontsize=9 if is_key else 8,
                 color=tgt["color"] if is_key else TXT_PRI,
                 fontweight="bold" if is_key else "normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    # Ambos resultados al final
    y_box = y_sec - r_h2*(len(items) + 0.4)
    box = mpatches.FancyBboxPatch(
        (0.04, y_box - 0.07), 0.92, 0.11,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box + 0.012,
             f"y(π/4)  =  {RES['pi4']:.6f}",
             ha="center", va="center", fontsize=9.5,
             color="#FFB74D", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)
    axt.text(0.5, y_box - 0.028,
             f"y(π/2)  =  {RES['pi2']:.6f}",
             ha="center", va="center", fontsize=9.5,
             color="#80CBC4", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Fases ──────────────────────────────────────────────────────────────────
def build_phases(x_val):
    specs = []
    li = 0
    for j in range(1, n):
        for i in range(j, n):
            is_final = (j == n - 1 and i == n - 1)
            line = poly_line if is_final else phase_lines[li]
            if not is_final:
                line.set_color(G_COLORS[j - 1])
                li += 1
            ii, jj = i, j
            specs.append((f"Grado {j}:  Q[{i},{j}]", line, make_partial(ii, jj)))
    return specs

N_FRAMES = 55
state = {"mode": "pi4", "running": False, "anim": None}

def clear_all():
    for ln in phase_lines: ln.set_data([], [])
    poly_line.set_data([], [])
    scat_pi4.set_offsets(np.empty((0, 2)))
    scat_pi2.set_offsets(np.empty((0, 2)))
    txt_pi4.set_text(""); txt_pi2.set_text("")
    status_t.set_text("")
    draw_table(None)

def run_animation(event=None):
    if state["running"]: return
    clear_all()
    state["running"] = True
    mode   = state["mode"]
    x_val  = TARGETS[mode]["x"]
    phases = build_phases(x_val)
    total  = len(phases) * N_FRAMES

    def update(frame):
        ph_idx   = min(frame // N_FRAMES, len(phases) - 1)
        ph_frame = frame % N_FRAMES
        name, line, fn = phases[ph_idx]
        status_t.set_text(name)
        n_pts = max(2, int((ph_frame + 1) / N_FRAMES * len(xr)))
        line.set_data(xr[:n_pts], fn(xr[:n_pts]))

        if frame == total - 1:
            y_pi4 = float(neville_fn(X_PI4))
            y_pi2 = float(neville_fn(X_PI2))
            scat_pi4.set_offsets([[X_PI4, y_pi4]])
            scat_pi2.set_offsets([[X_PI2, y_pi2]])
            txt_pi4.set_text(f"  {y_pi4:.5f}")
            txt_pi4.set_x(X_PI4 + 0.04); txt_pi4.set_y(y_pi4 + 0.1)
            txt_pi2.set_text(f"  {y_pi2:.5f}")
            txt_pi2.set_x(X_PI2 + 0.04); txt_pi2.set_y(y_pi2 + 0.1)
            status_t.set_text(
                f"✓  y(π/4)={y_pi4:.5f}   y(π/2)={y_pi2:.5f}")
            draw_table(mode)
            state["running"] = False
        return (line,)

    state["anim"] = FuncAnimation(
        fig, update, frames=total,
        interval=16, blit=False, repeat=False)
    fig.canvas.draw_idle()

def reset_animation(event=None):
    if state["anim"]:
        state["anim"].event_source.stop()
        state["anim"] = None
    state["running"] = False
    clear_all()
    fig.canvas.draw_idle()

def switch_mode(label):
    reset_animation()
    state["mode"] = "pi4" if "4" in label else "pi2"

# ── Widgets ────────────────────────────────────────────────────────────────
ax_play  = fig.add_axes([0.06, 0.08, 0.15, 0.07])
ax_reset = fig.add_axes([0.23, 0.08, 0.15, 0.07])
ax_radio = fig.add_axes([0.42, 0.03, 0.26, 0.14])

for wax in [ax_play, ax_reset, ax_radio]:
    wax.set_facecolor(BG_FIG)
    for sp in wax.spines.values(): sp.set_color(SPINE_CLR)

btn_play  = Button(ax_play,  "▶   Animar",
                   color="#0D2B55", hovercolor="#1A3A6E")
btn_reset = Button(ax_reset, "↺   Reset",
                   color="#1A1A3E", hovercolor="#2A2A5E")
radio     = RadioButtons(ax_radio,
                         ("Tabla para x = π/4", "Tabla para x = π/2"),
                         activecolor=ACCENT)

for b in [btn_play, btn_reset]:
    b.label.set_color(TXT_PRI); b.label.set_fontsize(10)
for lbl in radio.labels:
    lbl.set_color(TXT_PRI); lbl.set_fontsize(9.5)

btn_play.on_clicked(run_animation)
btn_reset.on_clicked(reset_animation)
radio.on_clicked(switch_mode)

draw_table(None)
plt.show()