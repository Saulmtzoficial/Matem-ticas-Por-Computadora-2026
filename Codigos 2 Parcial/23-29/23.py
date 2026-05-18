"""
Problema 23 - Intersección de dos círculos  (figura estática + animación)
==========================================================================
Círculo 1: (x - 2)² + y²       = 4
Círculo 2:  x²      + (y - 3)² = 4

Salidas:
  interseccion_tabla.png       — círculos, convergencia, trayectorias, tablas
  interseccion_animacion.gif   — animación suavizada con easing cúbico
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # sin ventana interactiva; quita esta línea si quieres plt.show()
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.gridspec import GridSpec
import os

# ─────────────────────────────────────────────
#  PALETA OSCURA
# ─────────────────────────────────────────────

BG     = '#050810'
PANEL  = '#0b0e1a'
GRID   = '#0f1528'
CYAN   = '#00d4ff'
TEAL   = '#00ffb3'
ORANGE = '#ff9500'
RED    = '#ff3c5f'
DIM    = '#1e3358'
WHITE  = '#c8dff5'
GOLD   = '#ffd166'

def apply_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  BG,
        'axes.facecolor':    PANEL,
        'axes.edgecolor':    DIM,
        'axes.labelcolor':   WHITE,
        'xtick.color':       DIM,
        'ytick.color':       DIM,
        'text.color':        WHITE,
        'grid.color':        GRID,
        'grid.linewidth':    0.8,
        'font.family':       'monospace',
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })

# ─────────────────────────────────────────────
# 1. SISTEMA Y JACOBIANO
# ─────────────────────────────────────────────

def F(v):
    """Vector de funciones F(x, y)."""
    x, y = v
    f1 = (x - 2)**2 + y**2       - 4
    f2 =  x**2      + (y - 3)**2 - 4
    return np.array([f1, f2])

def J(v):
    """Jacobiana 2×2."""
    x, y = v
    return np.array([
        [2*(x - 2),  2*y    ],
        [2*x,        2*(y-3)]
    ])

# ─────────────────────────────────────────────
# 2. NEWTON-RAPHSON
# ─────────────────────────────────────────────

def newton_raphson(x0, tol=1e-10, max_iter=50):
    x = np.array(x0, dtype=float)
    history = []
    for i in range(max_iter):
        Fval = F(x)
        Jval = J(x)
        detJ = np.linalg.det(Jval)
        err  = np.linalg.norm(Fval)
        history.append({'k': i, 'x': x[0], 'y': x[1],
                         'f1': Fval[0], 'f2': Fval[1],
                         'err': err, 'det': detJ})
        if err < tol:
            break
        delta_x = np.linalg.solve(Jval, -Fval)
        x = x + delta_x
    return x, history

# ─────────────────────────────────────────────
# 3. RESOLVER
# ─────────────────────────────────────────────

x0_A = [0.5, 2.0]
x0_B = [2.5, 1.0]

sol_A, hist_A = newton_raphson(x0_A)
sol_B, hist_B = newton_raphson(x0_B)

print(f"Punto A: ({sol_A[0]:.10f}, {sol_A[1]:.10f})  |  iters: {len(hist_A)}")
print(f"Punto B: ({sol_B[0]:.10f}, {sol_B[1]:.10f})  |  iters: {len(hist_B)}")
print(f"Verificación F(A) = {F(sol_A)}")
print(f"Verificación F(B) = {F(sol_B)}")

# ─────────────────────────────────────────────
# 4. FIGURA ESTÁTICA
# ─────────────────────────────────────────────

apply_dark_style()

fig = plt.figure(figsize=(18, 10), dpi=130)
fig.patch.set_facecolor(BG)

gs = GridSpec(2, 3, figure=fig,
              left=0.04, right=0.98,
              top=0.90, bottom=0.06,
              hspace=0.52, wspace=0.35)

fig.text(0.5, 0.960,
         'INTERSECCIÓN DE DOS CÍRCULOS  ·  MÉTODO DE NEWTON-RAPHSON',
         ha='center', fontsize=13, color=CYAN,
         fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.938,
         r'$(x-2)^2 + y^2 = 4$   ∩   $x^2 + (y-3)^2 = 4$',
         ha='center', fontsize=9, color=WHITE, alpha=0.55)

# ── Círculos ────────────────────────────────────────────────────────────────

ax_circ = fig.add_subplot(gs[:, 0])
ax_circ.set_facecolor(PANEL)

theta = np.linspace(0, 2*np.pi, 500)
c1x = 2 + 2*np.cos(theta);  c1y = 0 + 2*np.sin(theta)
c2x = 0 + 2*np.cos(theta);  c2y = 3 + 2*np.sin(theta)

ax_circ.plot(c1x, c1y, color=CYAN,   lw=2.5, label=r'$(x-2)^2+y^2=4$')
ax_circ.plot(c2x, c2y, color=ORANGE, lw=2.5, label=r'$x^2+(y-3)^2=4$')
ax_circ.scatter([2], [0], color=CYAN,   s=60, zorder=5, edgecolors=BG, lw=1.5)
ax_circ.scatter([0], [3], color=ORANGE, s=60, zorder=5, edgecolors=BG, lw=1.5)
ax_circ.annotate('C₁(2, 0)', (2, 0), xytext=(8, -14),
                 textcoords='offset points', color=CYAN,   fontsize=8.5)
ax_circ.annotate('C₂(0, 3)', (0, 3), xytext=(8,  6),
                 textcoords='offset points', color=ORANGE, fontsize=8.5)

for sol, lbl, col in zip([sol_A, sol_B], ['A', 'B'], [TEAL, GOLD]):
    ax_circ.scatter(*sol, color=col, s=200, marker='*', zorder=7,
                    edgecolors=BG, lw=0.5)
    ax_circ.annotate(
        f'  P{lbl}\n  ({sol[0]:.4f},\n  {sol[1]:.4f})',
        sol, color=col, fontsize=8.5,
        textcoords='offset points', xytext=(10, 4)
    )

ax_circ.axhline(0, color=DIM, lw=0.5, alpha=0.5)
ax_circ.axvline(0, color=DIM, lw=0.5, alpha=0.5)
ax_circ.set_xlim(-2.8, 5.2);  ax_circ.set_ylim(-2.8, 5.8)
ax_circ.set_aspect('equal')
ax_circ.legend(loc='lower right', facecolor=PANEL, edgecolor=DIM,
               labelcolor=WHITE, fontsize=8.5)
ax_circ.set_title('Intersección de los Círculos', color=WHITE, fontsize=10, pad=7)
ax_circ.tick_params(labelsize=7)
ax_circ.grid(True, alpha=0.3)
ax_circ.set_xlabel('x', color=WHITE, fontsize=8)
ax_circ.set_ylabel('y', color=WHITE, fontsize=8)

# ── Convergencia ─────────────────────────────────────────────────────────────

ax_err = fig.add_subplot(gs[0, 1])
ax_err.set_facecolor(PANEL)

for hist, col, lbl in [(hist_A, TEAL, 'Punto A'), (hist_B, GOLD, 'Punto B')]:
    ks   = [h['k']   for h in hist]
    errs = [h['err'] for h in hist]
    ax_err.semilogy(ks, errs, 'o-', color=col, lw=2, ms=5,
                    markerfacecolor=BG, markeredgecolor=col, mew=1.2, label=lbl)
    ax_err.fill_between(ks, errs, alpha=0.07, color=col)
    ax_err.annotate(f'  ε={errs[-1]:.1e}', xy=(ks[-1], errs[-1]),
                    fontsize=7, color=col, alpha=0.9)

ax_err.set_title('Convergencia  ‖F(x,y)‖  vs Iteración', color=WHITE, fontsize=9, pad=5)
ax_err.set_xlabel('Iteración k', fontsize=7.5)
ax_err.set_ylabel('‖F‖', fontsize=7.5)
ax_err.tick_params(labelsize=7)
ax_err.legend(fontsize=7.5, framealpha=0.2, facecolor=PANEL,
              edgecolor=DIM, labelcolor=WHITE)
ax_err.grid(True, which='both', alpha=0.3)

# ── Trayectoria (x, y) ───────────────────────────────────────────────────────

ax_xy = fig.add_subplot(gs[1, 1])
ax_xy.set_facecolor(PANEL)

for hist, col, sol, lbl in [
        (hist_A, TEAL, sol_A, 'Punto A'),
        (hist_B, GOLD, sol_B, 'Punto B')]:
    xs = [h['x'] for h in hist]
    ys = [h['y'] for h in hist]
    ax_xy.plot(xs, ys, 'o-', color=col, lw=1.8, ms=4,
               markerfacecolor=BG, label=lbl)
    ax_xy.scatter(xs[0], ys[0], color=col, s=60, marker='s',
                  zorder=5, alpha=0.7)
    ax_xy.scatter(*sol, color=col, s=120, marker='*', zorder=6)

ax_xy.set_title('Trayectoria de las Iteraciones (x, y)', color=WHITE, fontsize=9, pad=5)
ax_xy.set_xlabel('x', fontsize=7.5);  ax_xy.set_ylabel('y', fontsize=7.5)
ax_xy.tick_params(labelsize=7)
ax_xy.legend(fontsize=7.5, framealpha=0.2, facecolor=PANEL,
             edgecolor=DIM, labelcolor=WHITE)
ax_xy.grid(True, alpha=0.3)

# ── Tablas ───────────────────────────────────────────────────────────────────

col_labels = ['k', 'x', 'y', 'f₁', 'f₂', '‖F‖', 'det(J)']

def make_rows(hist):
    rows = []
    show = hist if len(hist) <= 10 else hist[:5] + [None] + hist[-3:]
    for h in show:
        if h is None:
            rows.append(['⋮'] * len(col_labels))
            continue
        rows.append([
            f"{h['k']}",
            f"{h['x']:+.5f}",
            f"{h['y']:+.5f}",
            f"{h['f1']:+.3e}",
            f"{h['f2']:+.3e}",
            f"{h['err']:.3e}",
            f"{h['det']:+.4f}",
        ])
    return rows

def render_table(ax, rows, title, accent_col):
    ax.axis('off')
    ax.set_title(title, color=accent_col, fontsize=8.5, pad=5)
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(DIM)
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#0d1e38')
            cell.set_text_props(color=accent_col, fontweight='bold')
        elif rows[r-1][0] == '⋮':
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=DIM)
        else:
            idx = r - 1
            try:
                err_val = float(rows[idx][5])
            except ValueError:
                err_val = 1.0
            if err_val < 1e-6:
                cell.set_facecolor('#051a10')
                cell.set_text_props(color=TEAL)
            elif idx % 2 == 0:
                cell.set_facecolor('#0b0f1e')
                cell.set_text_props(color=WHITE)
            else:
                cell.set_facecolor(PANEL)
                cell.set_text_props(color=WHITE)

ax_tblA = fig.add_subplot(gs[0, 2])
render_table(ax_tblA, make_rows(hist_A),
             f'Tabla — Punto A  (x₀={x0_A})', TEAL)

ax_tblB = fig.add_subplot(gs[1, 2])
render_table(ax_tblB, make_rows(hist_B),
             f'Tabla — Punto B  (x₀={x0_B})', GOLD)

# ── Cuadro de soluciones finales ─────────────────────────────────────────────

sol_text = (
    f"SOLUCIONES ENCONTRADAS\n"
    f"────────────────────────────────────────\n"
    f"Punto A:  x = {sol_A[0]:+.8f}\n"
    f"          y = {sol_A[1]:+.8f}\n"
    f"  ‖F(A)‖ = {np.linalg.norm(F(sol_A)):.3e}  |  iters: {len(hist_A)}\n\n"
    f"Punto B:  x = {sol_B[0]:+.8f}\n"
    f"          y = {sol_B[1]:+.8f}\n"
    f"  ‖F(B)‖ = {np.linalg.norm(F(sol_B)):.3e}  |  iters: {len(hist_B)}"
)
fig.text(0.015, 0.01, sol_text,
         fontsize=7.5, color=TEAL, fontfamily='monospace', va='bottom',
         bbox=dict(facecolor='#050f08', edgecolor='#00aa77',
                   boxstyle='round,pad=0.5', lw=0.8))

out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'interseccion_tabla.png')
plt.savefig(out_png, dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"✔  Imagen guardada → {out_png}")

# ─────────────────────────────────────────────
# 5. ANIMACIÓN SUAVIZADA (GIF)
# ─────────────────────────────────────────────

N_FRAMES = 100
FPS      = 25

def ease_in_out_cubic(t):
    if t < 0.5:
        return 4*t*t*t
    p = 2*t - 2
    return 0.5*p*p*p + 1

def interpolate_trajectory(hist, n_frames):
    pts   = np.array([[h['x'], h['y']] for h in hist])
    n     = len(pts)
    t_raw = np.linspace(0, 1, n)
    eased = np.array([ease_in_out_cubic(t) for t in np.linspace(0, 1, n_frames)])
    x_an  = np.interp(eased, t_raw, pts[:, 0])
    y_an  = np.interp(eased, t_raw, pts[:, 1])
    return x_an, y_an

xA_anim, yA_anim = interpolate_trajectory(hist_A, N_FRAMES)
xB_anim, yB_anim = interpolate_trajectory(hist_B, N_FRAMES)

def err_along_anim(x_anim, y_anim):
    return [np.linalg.norm(F([x, y])) for x, y in zip(x_anim, y_anim)]

errA_anim = err_along_anim(xA_anim, yA_anim)
errB_anim = err_along_anim(xB_anim, yB_anim)

t_frames = np.linspace(0, 1, N_FRAMES)
theta_c  = np.linspace(0, 2*np.pi, 500)
C1x = 2 + 2*np.cos(theta_c);  C1y = 0 + 2*np.sin(theta_c)
C2x = 0 + 2*np.cos(theta_c);  C2y = 3 + 2*np.sin(theta_c)

apply_dark_style()
fig_ani = plt.figure(figsize=(13, 7), dpi=100)
fig_ani.patch.set_facecolor(BG)

gs2 = GridSpec(2, 3, figure=fig_ani,
               left=0.05, right=0.97,
               top=0.88, bottom=0.08,
               hspace=0.55, wspace=0.38)

fig_ani.text(0.5, 0.955,
             'INTERSECCIÓN DE CÍRCULOS  ·  ANIMACIÓN NEWTON-RAPHSON',
             ha='center', fontsize=11, color=CYAN,
             fontweight='bold', fontfamily='monospace')
fig_ani.text(0.5, 0.935,
             r'$(x-2)^2+y^2=4$  ∩  $x^2+(y-3)^2=4$',
             ha='center', fontsize=8, color=WHITE, alpha=0.45)

ax_main = fig_ani.add_subplot(gs2[:, 0])
ax_errA = fig_ani.add_subplot(gs2[0, 1])
ax_errB = fig_ani.add_subplot(gs2[1, 1])
ax_xyA  = fig_ani.add_subplot(gs2[0, 2])
ax_xyB  = fig_ani.add_subplot(gs2[1, 2])

all_xA = [h['x'] for h in hist_A];  all_yA = [h['y'] for h in hist_A]
all_xB = [h['x'] for h in hist_B];  all_yB = [h['y'] for h in hist_B]

def update(frame):
    prog = frame / (N_FRAMES - 1)
    for ax in [ax_main, ax_errA, ax_errB, ax_xyA, ax_xyB]:
        ax.cla()

    # ── Círculos + iteraciones ───────────────────────────────────────────────
    ax_main.set_facecolor(PANEL)
    ax_main.set_xlim(-2.8, 5.2);  ax_main.set_ylim(-2.8, 5.8)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(0, color=DIM, lw=0.4, alpha=0.5)
    ax_main.axvline(0, color=DIM, lw=0.4, alpha=0.5)

    ax_main.plot(C1x, C1y, color=CYAN,   lw=2.0, alpha=0.85)
    ax_main.plot(C2x, C2y, color=ORANGE, lw=2.0, alpha=0.85)
    ax_main.scatter([2], [0], color=CYAN,   s=40, zorder=5,
                    edgecolors=BG, lw=1.2)
    ax_main.scatter([0], [3], color=ORANGE, s=40, zorder=5,
                    edgecolors=BG, lw=1.2)
    ax_main.text(2.05, -0.25, 'C₁', color=CYAN,   fontsize=7.5, alpha=0.7)
    ax_main.text(0.05,  3.18, 'C₂', color=ORANGE, fontsize=7.5, alpha=0.7)

    # Trails
    trail = max(1, frame)
    for i in range(1, trail):
        a = (i / trail) ** 1.5 * 0.75
        ax_main.plot([xA_anim[i-1], xA_anim[i]],
                     [yA_anim[i-1], yA_anim[i]],
                     color=TEAL, lw=1.2, alpha=a, solid_capstyle='round')
        ax_main.plot([xB_anim[i-1], xB_anim[i]],
                     [yB_anim[i-1], yB_anim[i]],
                     color=GOLD, lw=1.2, alpha=a, solid_capstyle='round')

    ax_main.scatter(xA_anim[frame], yA_anim[frame],
                    color=TEAL, s=70, zorder=8, edgecolors=BG, lw=1.2)
    ax_main.scatter(xB_anim[frame], yB_anim[frame],
                    color=GOLD, s=70, zorder=8, edgecolors=BG, lw=1.2)

    # Objetivos (estrella)
    ax_main.scatter(*sol_A, color=TEAL, s=160, marker='*',
                    zorder=9, edgecolors=BG, lw=0.5, alpha=0.6)
    ax_main.scatter(*sol_B, color=GOLD, s=160, marker='*',
                    zorder=9, edgecolors=BG, lw=0.5, alpha=0.6)

    # Puntos iniciales (fantasma cuadrado)
    ax_main.scatter(*x0_A, color=TEAL, s=50, marker='s',
                    zorder=4, alpha=0.35, edgecolors=BG, lw=1)
    ax_main.scatter(*x0_B, color=GOLD, s=50, marker='s',
                    zorder=4, alpha=0.35, edgecolors=BG, lw=1)

    # Barra de progreso
    bar_y = 5.55
    ax_main.plot([-2.5, 4.8],            [bar_y, bar_y],
                 color=DIM, lw=3, solid_capstyle='round', zorder=6)
    ax_main.plot([-2.5, -2.5 + 7.3*prog], [bar_y, bar_y],
                 color=CYAN, lw=3, solid_capstyle='round', zorder=7, alpha=0.8)

    ax_main.set_title(
        f'Frame {frame+1}/{N_FRAMES}\n'
        f'A=({xA_anim[frame]:.4f}, {yA_anim[frame]:.4f})  '
        f'B=({xB_anim[frame]:.4f}, {yB_anim[frame]:.4f})',
        color=WHITE, fontsize=7.8, pad=4
    )
    ax_main.tick_params(labelsize=6.5)
    ax_main.set_xlabel('x', fontsize=7.5)
    ax_main.set_ylabel('y', fontsize=7.5)

    # ── Convergencia A ───────────────────────────────────────────────────────
    ax_errA.set_facecolor(PANEL)
    ax_errA.semilogy(t_frames, errA_anim, color=TEAL, lw=0.8, alpha=0.25)
    ax_errA.semilogy(t_frames[:frame+1], errA_anim[:frame+1], color=TEAL, lw=1.8)
    ax_errA.scatter(t_frames[frame], errA_anim[frame], color=TEAL, s=40, zorder=5)
    ax_errA.fill_between(t_frames[:frame+1], errA_anim[:frame+1],
                         alpha=0.08, color=TEAL)
    ax_errA.set_title('‖F‖ — Punto A', color=TEAL, fontsize=8, pad=3)
    ax_errA.set_xlabel('Progreso', fontsize=6.5)
    ax_errA.tick_params(labelsize=6)
    ax_errA.grid(True, alpha=0.25, which='both')

    # ── Convergencia B ───────────────────────────────────────────────────────
    ax_errB.set_facecolor(PANEL)
    ax_errB.semilogy(t_frames, errB_anim, color=GOLD, lw=0.8, alpha=0.25)
    ax_errB.semilogy(t_frames[:frame+1], errB_anim[:frame+1], color=GOLD, lw=1.8)
    ax_errB.scatter(t_frames[frame], errB_anim[frame], color=GOLD, s=40, zorder=5)
    ax_errB.fill_between(t_frames[:frame+1], errB_anim[:frame+1],
                         alpha=0.08, color=GOLD)
    ax_errB.set_title('‖F‖ — Punto B', color=GOLD, fontsize=8, pad=3)
    ax_errB.set_xlabel('Progreso', fontsize=6.5)
    ax_errB.tick_params(labelsize=6)
    ax_errB.grid(True, alpha=0.25, which='both')

    # ── Camino (x, y) — A ────────────────────────────────────────────────────
    ax_xyA.set_facecolor(PANEL)
    ax_xyA.plot(all_xA, all_yA, color=TEAL, lw=0.6, alpha=0.2)
    if frame > 0:
        ax_xyA.plot(xA_anim[:frame+1], yA_anim[:frame+1], color=TEAL, lw=1.8)
    ax_xyA.scatter(xA_anim[0], yA_anim[0], color=TEAL,
                   s=45, marker='s', alpha=0.5, zorder=4)
    ax_xyA.scatter(xA_anim[frame], yA_anim[frame],
                   color=TEAL, s=55, zorder=6)
    ax_xyA.scatter(*sol_A, color='white', s=80, marker='*', zorder=7, alpha=0.6)
    ax_xyA.set_title('Camino (x,y) — A', color=TEAL, fontsize=8, pad=3)
    ax_xyA.set_xlabel('x', fontsize=6.5);  ax_xyA.set_ylabel('y', fontsize=6.5)
    ax_xyA.tick_params(labelsize=6)
    ax_xyA.grid(True, alpha=0.25)

    # ── Camino (x, y) — B ────────────────────────────────────────────────────
    ax_xyB.set_facecolor(PANEL)
    ax_xyB.plot(all_xB, all_yB, color=GOLD, lw=0.6, alpha=0.2)
    if frame > 0:
        ax_xyB.plot(xB_anim[:frame+1], yB_anim[:frame+1], color=GOLD, lw=1.8)
    ax_xyB.scatter(xB_anim[0], yB_anim[0], color=GOLD,
                   s=45, marker='s', alpha=0.5, zorder=4)
    ax_xyB.scatter(xB_anim[frame], yB_anim[frame],
                   color=GOLD, s=55, zorder=6)
    ax_xyB.scatter(*sol_B, color='white', s=80, marker='*', zorder=7, alpha=0.6)
    ax_xyB.set_title('Camino (x,y) — B', color=GOLD, fontsize=8, pad=3)
    ax_xyB.set_xlabel('x', fontsize=6.5);  ax_xyB.set_ylabel('y', fontsize=6.5)
    ax_xyB.tick_params(labelsize=6)
    ax_xyB.grid(True, alpha=0.25)

    return []

ani = animation.FuncAnimation(
    fig_ani, update,
    frames=N_FRAMES,
    interval=1000 // FPS,
    blit=False
)

out_gif = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'interseccion_animacion.gif')
print("  Renderizando animación… (puede tardar ~15 s)")
ani.save(out_gif, writer=PillowWriter(fps=FPS),
         savefig_kwargs={'facecolor': BG})
plt.close(fig_ani)
print(f"✔  Animación guardada → {out_gif}")
print("\n  Archivos generados en la misma carpeta del script:")
print("    → interseccion_tabla.png")
print("    → interseccion_animacion.gif")