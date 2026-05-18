"""
Densidad del aire ρ(h) como función cuadrática — Interpolación de Lagrange
Datos: h (km) = [0, 3, 6]
       ρ (kg/m³) = [1.225, 0.905, 0.652]

Bases de Lagrange:
  L₀(h) = (h−3)(h−6) / (0−3)(0−6) = (h²−9h+18) / 18
  L₁(h) = (h−0)(h−6) / (3−0)(3−6) = h(h−6) / (−9)
  L₂(h) = (h−0)(h−3) / (6−0)(6−3) = h(h−3) / 18

Resultado: ρ(h) ≈ 0.003722 h² − 0.117833 h + 1.225  [kg/m³]
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

C = {
    "pts":   "#FFB74D",
    "l0":    "#FF7043",
    "l1":    "#66BB6A",
    "l2":    "#CE93D8",
    "poly":  "#4FC3F7",
    "probe": "#FF5252",
}

# ── Datos ──────────────────────────────────────────────────────────────────
hs   = np.array([0., 3., 6.])           # elevación en km
rhos = np.array([1.225, 0.905, 0.652])  # densidad en kg/m³
n    = len(hs)

# Coeficientes exactos del polinomio
a_coef =  1.225/18 + 0.905*(-1)/9 + 0.652/18    # ≈  0.003722
b_coef = -1.225*9/18 + 0.905*6/9 + 0.652*(-3)/18  # ≈ -0.117833
c_coef =  1.225                                   # = 1.225

def rho_poly(h):
    return a_coef*h**2 + b_coef*h + c_coef

# ── Bases de Lagrange ──────────────────────────────────────────────────────
def L(j, h):
    h = np.atleast_1d(h).astype(float)
    num = np.ones_like(h); den = 1.0
    for k in range(n):
        if k != j:
            num *= (h - hs[k]); den *= (hs[j] - hs[k])
    return num / den

def term(j, h): return rhos[j] * L(j, h)

hr = np.linspace(-0.5, 8.5, 600)

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
         "Lagrange  —  Densidad del aire  ρ(h)  como cuadrática",
         ha="center", va="top", fontsize=12, color=ACCENT)

gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1],
                      left=0.06, right=0.97,
                      bottom=0.22, top=0.94, wspace=0.06)
ax  = fig.add_subplot(gs[0])
axt = fig.add_subplot(gs[1])

# Estilo eje gráfica
ax.set_facecolor(BG_AX)
ax.set_xlim(-0.5, 8.5); ax.set_ylim(-0.05, 1.55)
ax.set_xlabel("h  (km)", fontsize=11)
ax.set_ylabel("ρ  (kg/m³)", fontsize=11)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)
ax.tick_params(colors=TXT_SEC, labelsize=9)
ax.grid(True, color=GRID_CLR, lw=0.6, zorder=0)
ax.axhline(0, color=SPINE_CLR, lw=0.8, zorder=1)

# Puntos de datos
ax.scatter(hs, rhos, zorder=6, s=100, color=C["pts"],
           edgecolors="white", linewidths=1.8)
for hi, ri in zip(hs, rhos):
    ax.annotate(f"({hi:.0f} km, {ri:.3f})", (hi, ri),
                textcoords="offset points", xytext=(8, 8),
                fontsize=8.5, color=C["pts"])

# Etiquetas de unidades en ejes
ax.text(8.3, -0.04, "km", fontsize=8, color=TXT_SEC, ha="right")
ax.text(-0.3, 1.52,  "kg/m³", fontsize=8, color=TXT_SEC)

# Líneas animadas
line_t0,  = ax.plot([], [], color=C["l0"],  lw=1.6, ls="--", alpha=0.9,
                    label="ρ₀·L₀(h) = 1.225·L₀")
line_t1,  = ax.plot([], [], color=C["l1"],  lw=1.6, ls="--", alpha=0.9,
                    label="ρ₁·L₁(h) = 0.905·L₁")
line_t2,  = ax.plot([], [], color=C["l2"],  lw=1.6, ls="--", alpha=0.9,
                    label="ρ₂·L₂(h) = 0.652·L₂")
line_poly,= ax.plot([], [], color=C["poly"], lw=2.8, zorder=5,
                    label="ρ(h) = 0.003722h² − 0.117833h + 1.225")

# Sonda interactiva (aparece tras animación)
probe_vline = ax.axvline(x=0, color=C["probe"], lw=1.0, ls=":", alpha=0)
probe_dot   = ax.scatter([], [], s=120, color=C["probe"], zorder=8,
                         edgecolors="white", linewidths=1.5)
probe_txt   = ax.text(0, 0, "", fontsize=9, color=C["probe"])

status_txt = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     fontsize=10, va="top", color=ACCENT)

ax.legend(loc="upper right", fontsize=8, framealpha=0.35,
          facecolor=BG_AX, edgecolor=SPINE_CLR, labelcolor=TXT_PRI)

# Panel tabla
axt.set_facecolor(BG_TABLE)
for sp in axt.spines.values(): sp.set_color(SPINE_CLR)
axt.set_axis_off()

# ── Slider de elevación ────────────────────────────────────────────────────
ax_slider = fig.add_axes([0.44, 0.075, 0.35, 0.04])
ax_slider.set_facecolor(BG_FIG)
for sp in ax_slider.spines.values(): sp.set_color(SPINE_CLR)
slider = Slider(ax_slider, "h (km)", 0.0, 8.0,
                valinit=4.0, valstep=0.1,
                color="#185FA5", initcolor="none")
slider.label.set_color(TXT_PRI)
slider.valtext.set_color(ACCENT)
probe_active = [False]

def update_probe(val):
    if not probe_active[0]: return
    h_val = slider.val
    rho_val = float(rho_poly(h_val))
    probe_vline.set_xdata([h_val, h_val])
    probe_dot.set_offsets([[h_val, rho_val]])
    probe_txt.set_text(f"  ρ({h_val:.1f}) = {rho_val:.4f}")
    probe_txt.set_x(h_val + 0.1)
    probe_txt.set_y(rho_val + 0.04)
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
                 "Presiona  ▶  Animar\npara construir ρ(h)",
                 ha="center", va="center", fontsize=11,
                 color=TXT_SEC, linespacing=2.2, transform=axt.transAxes)
        fig.canvas.draw_idle()
        return

    # ── Título ────────────────────────────────────────────────────────────
    axt.text(0.5, 0.975, "Bases de Lagrange",
             ha="center", va="top", fontsize=10,
             color=ACCENT, transform=axt.transAxes)

    # ── Tabla de bases ─────────────────────────────────────────────────────
    col_h = ["j", "hⱼ", "ρⱼ", "Lⱼ(h)  (forma factorizada)"]
    col_w = [0.06, 0.09, 0.12, 0.73]

    rows = [
        ["0", "0", "1.225",
         "(h−3)(h−6) / 18"],
        ["1", "3", "0.905",
         "h(h−6) / (−9)"],
        ["2", "6", "0.652",
         "h(h−3) / 18"],
    ]
    row_clrs = [C["l0"], C["l1"], C["l2"]]

    n_rows  = len(rows) + 1
    total_h = 0.22
    row_h   = total_h / (n_rows + 0.3)
    y_start = 0.89

    def draw_row(cells, y_c, is_hdr=False, row_clr=None):
        if is_hdr:
            rect = mpatches.FancyBboxPatch(
                (0.01, y_c - row_h*0.47), 0.98, row_h*0.94,
                boxstyle="round,pad=0.003",
                facecolor="#1A3A5C", edgecolor="none",
                transform=axt.transAxes, zorder=1, clip_on=False)
            axt.add_patch(rect)
        x = 0.015
        for ci, (cell, cw) in enumerate(zip(cells, col_w)):
            if is_hdr:
                clr = TXT_SEC
            elif ci == 3 and row_clr:
                clr = row_clr
            elif ci == 0 and row_clr:
                clr = row_clr
            else:
                clr = TXT_PRI
            fs = 7.0 if is_hdr else (8.5 if ci < 3 else 8.0)
            axt.text(x + cw/2, y_c, cell,
                     ha="center", va="center", fontsize=fs,
                     color=clr, fontweight="normal",
                     transform=axt.transAxes, zorder=2, clip_on=False)
            x += cw

    y_h = y_start - row_h*0.5
    draw_row(col_h, y_h, is_hdr=True)
    axt.plot([0.01, 0.99], [y_h - row_h*0.52]*2,
             color=SPINE_CLR, lw=0.8, transform=axt.transAxes, zorder=4)

    for ri, (row, rc) in enumerate(zip(rows, row_clrs)):
        y_r = y_start - row_h*(ri + 1.5)
        draw_row(row, y_r, row_clr=rc)

    axt.plot([0.01, 0.99], [y_start - row_h*(len(rows)+1.0)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # ── Desarrollo ────────────────────────────────────────────────────────
    y_dev = y_start - row_h*(len(rows) + 1.6)

    dev_items = [
        ("Término 0:", "1.225 · (h²−9h+18)/18", C["l0"]),
        ("",           "= 0.06806h² − 0.6125h + 1.225", TXT_PRI),
        ("Término 1:", "0.905 · h(h−6)/(−9)", C["l1"]),
        ("",           "= −0.10056h² + 0.60333h", TXT_PRI),
        ("Término 2:", "0.652 · h(h−3)/18", C["l2"]),
        ("",           "= 0.03622h² − 0.10867h", TXT_PRI),
    ]
    r_h2 = 0.058
    for ri, (lbl, val, clr) in enumerate(dev_items):
        y_r = y_dev - r_h2*ri
        if lbl:
            axt.text(0.04, y_r, lbl, ha="left", va="center",
                     fontsize=7.5, color=clr,
                     transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.96, y_r, val, ha="right", va="center",
                 fontsize=7.8, color=TXT_PRI,
                 transform=axt.transAxes, zorder=3, clip_on=False)

    axt.plot([0.01, 0.99], [y_dev - r_h2*(len(dev_items)+0.2)]*2,
             color=SPINE_CLR, lw=0.5, ls="--",
             transform=axt.transAxes, zorder=4)

    # ── Resultado sumado ──────────────────────────────────────────────────
    y_sum = y_dev - r_h2*(len(dev_items) + 0.8)
    sum_items = [
        ("Suma  a:",  f"{a_coef:.6f}  h²",  ACCENT),
        ("Suma  b:",  f"{b_coef:.6f}  h",   ACCENT),
        ("Suma  c:",  f"{c_coef:.4f}",      ACCENT),
    ]
    r_h3 = 0.062
    for ri, (lbl, val, clr) in enumerate(sum_items):
        y_r = y_sum - r_h3*ri
        axt.text(0.04, y_r, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_SEC,
                 transform=axt.transAxes, zorder=3, clip_on=False)
        axt.text(0.96, y_r, val, ha="right", va="center",
                 fontsize=8, color=clr, fontweight="normal",
                 transform=axt.transAxes, zorder=3, clip_on=False)

    # ── Caja final ────────────────────────────────────────────────────────
    y_box = y_sum - r_h3*(len(sum_items) + 0.6)
    box = mpatches.FancyBboxPatch(
        (0.03, y_box - 0.052), 0.94, 0.088,
        boxstyle="round,pad=0.01",
        facecolor="#0D47A1", edgecolor=ACCENT, linewidth=1.2,
        transform=axt.transAxes, zorder=4, clip_on=False)
    axt.add_patch(box)
    axt.text(0.5, y_box + 0.010,
             "ρ(h)  =  0.003722 h²  −  0.117833 h  +  1.225",
             ha="center", va="center", fontsize=9.5,
             color="#FFFFFF", fontweight="bold",
             transform=axt.transAxes, zorder=5, clip_on=False)
    axt.text(0.5, y_box - 0.028,
             "  h en km  ,  ρ en kg/m³  ",
             ha="center", va="center", fontsize=8,
             color="#B0BEC5",
             transform=axt.transAxes, zorder=5, clip_on=False)

    fig.canvas.draw_idle()

# ── Animación ──────────────────────────────────────────────────────────────
PHASES = [
    ("Trazando  ρ₀·L₀(h) = 1.225·(h−3)(h−6)/18 ...",
     line_t0, lambda h: term(0, h)),
    ("Trazando  ρ₁·L₁(h) = 0.905·h(h−6)/(−9) ...",
     line_t1, lambda h: term(1, h)),
    ("Trazando  ρ₂·L₂(h) = 0.652·h(h−3)/18 ...",
     line_t2, lambda h: term(2, h)),
    ("Sumando términos  →  ρ(h) cuadrática ...",
     line_poly, rho_poly),
]

N_FRAMES   = 85
MSG_FRAMES = 28
state = {"running": False, "anim": None}

def clear_all():
    for ln in [line_t0, line_t1, line_t2, line_poly]:
        ln.set_data([], [])
    probe_vline.set_alpha(0)
    probe_dot.set_offsets(np.empty((0, 2)))
    probe_txt.set_text("")
    status_txt.set_text("")
    probe_active[0] = False
    draw_table(done=False)

def run_animation(event=None):
    if state["running"]: return
    clear_all()
    state["running"] = True
    total = len(PHASES) * (MSG_FRAMES + N_FRAMES)

    def update(frame):
        ph_total = MSG_FRAMES + N_FRAMES
        ph       = frame // ph_total
        loc      = frame % ph_total
        if ph >= len(PHASES): return

        name, line, fn = PHASES[ph]
        if loc < MSG_FRAMES:
            status_txt.set_text(name)
        else:
            cf    = loc - MSG_FRAMES
            n_pts = max(2, int((cf + 1) / N_FRAMES * len(hr)))
            line.set_data(hr[:n_pts], fn(hr[:n_pts]))

        if frame == total - 1:
            status_txt.set_text(
                "✓  ρ(h) = 0.003722h² − 0.117833h + 1.225   "
                "— desliza para evaluar")
            probe_vline.set_alpha(0.7)
            probe_active[0] = True
            update_probe(slider.val)
            draw_table(done=True)
            state["running"] = False
        return (line_t0, line_t1, line_t2, line_poly)

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
ax_play  = fig.add_axes([0.06, 0.075, 0.15, 0.07])
ax_reset = fig.add_axes([0.23, 0.075, 0.15, 0.07])

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