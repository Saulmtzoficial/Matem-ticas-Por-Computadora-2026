"""
Problema 25 — Todas las soluciones reales en 0 < x < 1.5  (v2 — tabla + animación)
=====================================================================================
  f1(x, y) = tan(x) − y − 1       = 0
  f2(x, y) = cos(x) − 3·sin(y)    = 0

Jacobiano J(x, y):
  J = | sec²(x)      −1      |
      | −sin(x)   −3cos(y)   |

Salida:
  → problema25_tabla.png
  → problema25_animacion.gif
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from itertools import product
import math

# ─────────────────────────────────────────────
#  PALETA OSCURA
# ─────────────────────────────────────────────

BG     = '#050810'
PANEL  = '#0b0e1a'
GRID   = '#0f1528'
DIM    = '#1e3358'
WHITE  = '#c8dff5'
GOLD   = '#facc15'

COLORS = ['#38bdf8', '#fb923c', '#a78bfa', '#34d399',
          '#f472b6', '#60a5fa', '#fbbf24', '#4ade80']

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
    x, y = v
    return np.array([
        np.tan(x) - y - 1,
        np.cos(x) - 3*np.sin(y)
    ])

def J(v):
    x, y = v
    return np.array([
        [1 / np.cos(x)**2,  -1.0         ],
        [-np.sin(x),        -3*np.cos(y) ]
    ])

# ─────────────────────────────────────────────
# 2. NEWTON-RAPHSON MATRICIAL
# ─────────────────────────────────────────────

def newton_raphson(x0, tol=1e-10, max_iter=60):
    x = np.array(x0, dtype=float)
    history = []
    for _ in range(max_iter):
        Fval = F(x)
        Jval = J(x)
        detJ = np.linalg.det(Jval)
        err  = np.linalg.norm(Fval)

        history.append({
            'x': x[0], 'y': x[1],
            'f1': Fval[0], 'f2': Fval[1],
            'err': err, 'det': detJ
        })

        if err < tol:
            break
        if abs(detJ) < 1e-14:
            return None, []
        delta = np.linalg.solve(Jval, -Fval)
        x = x + delta
        if not (0 < x[0] < 1.5):
            return None, []

    if np.linalg.norm(F(x)) < tol and 0 < x[0] < 1.5:
        return x, history
    return None, []

# ─────────────────────────────────────────────
# 3. ESCANEO Y DEDUPLICACIÓN
# ─────────────────────────────────────────────

xs = np.linspace(0.05, 1.45, 40)
ys = np.linspace(-3.0,  3.0, 40)

solutions      = []
solution_hists = []

print("Buscando soluciones en 0 < x < 1.5 ...\n")

for x0, y0 in product(xs, ys):
    if abs(x0 - np.pi/2) < 0.05:
        continue
    sol, hist = newton_raphson([x0, y0])
    if sol is None:
        continue
    duplicate = any(np.linalg.norm(sol - s) < 1e-6 for s in solutions)
    if not duplicate:
        solutions.append(sol)
        solution_hists.append(hist)

print(f"  Soluciones encontradas: {len(solutions)}\n")
print(f"{'#':>3}  {'x':>14}  {'y':>14}  {'f1(sol)':>12}  {'f2(sol)':>12}")
print("─" * 60)
for i, s in enumerate(solutions):
    fv = F(s)
    print(f"{i+1:>3}  {s[0]:>14.10f}  {s[1]:>14.10f}  "
          f"{fv[0]:>12.2e}  {fv[1]:>12.2e}")

# Re-correr desde puntos cercanos para convergencias más vistosas
detailed_hists = []
for i, s in enumerate(solutions):
    x0_near = s + np.array([0.15, 0.15]) * ((-1)**i)
    x0_near[0] = np.clip(x0_near[0], 0.05, 1.45)
    _, hist = newton_raphson(x0_near)
    detailed_hists.append(hist if hist else solution_hists[i])
    print(f"  S{i+1}: x₀=({x0_near[0]:.3f}, {x0_near[1]:.3f})  →  "
          f"({s[0]:.8f}, {s[1]:.8f})  en {len(detailed_hists[-1])-1} iters")

# ─────────────────────────────────────────────
# 4. REJILLA PARA CURVAS Y MAPA DE CALOR
# ─────────────────────────────────────────────

xg = np.linspace(0.01, 1.499, 700)
yg = np.linspace(-3.5,  3.5,  700)
Xg, Yg = np.meshgrid(xg, yg)

mask = np.abs(Xg - np.pi/2) < 0.04
Z1   = np.where(mask, np.nan, np.tan(Xg) - Yg - 1)
Z2   = np.cos(Xg) - 3*np.sin(Yg)
normF = np.sqrt(np.nan_to_num(Z1)**2 + Z2**2)
normF_clip = np.clip(normF, 0, 5)

cmap_heat = LinearSegmentedColormap.from_list(
    'heat', [BG, '#0f2a3f', '#0e4a6e', '#1d7fa8', '#38bdf8', GOLD], N=256)

# ─────────────────────────────────────────────
# 5. FIGURA ESTÁTICA
# ─────────────────────────────────────────────

apply_dark_style()

n_sol = len(solutions)
fig = plt.figure(figsize=(22, 10), dpi=130)
fig.patch.set_facecolor(BG)

gs = GridSpec(2, 4, figure=fig,
              left=0.04, right=0.98,
              top=0.90, bottom=0.07,
              hspace=0.52, wspace=0.36)

# ── Título ──────────────────────────────────
fig.text(0.5, 0.962,
         'PROBLEMA 25  ·  TODAS LAS RAÍCES  0 < x < 1.5  ·  NEWTON-RAPHSON MATRICIAL',
         ha='center', fontsize=13, color=COLORS[0],
         fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.940,
         r'$f_1: \tan(x)-y-1=0$     $f_2: \cos(x)-3\sin(y)=0$',
         ha='center', fontsize=9, color=WHITE, alpha=0.55)

# ── Panel 1: Curvas de nivel ─────────────────
ax_cnt = fig.add_subplot(gs[:, 0])
ax_cnt.set_facecolor(PANEL)

ax_cnt.contour(Xg, Yg, Z1, levels=[0], colors=[COLORS[0]], linewidths=2.2)
ax_cnt.contour(Xg, Yg, Z2, levels=[0], colors=[COLORS[1]], linewidths=2.2)
ax_cnt.text(0.06,  0.3, 'tan x − y = 1',      color=COLORS[0], fontsize=8.5)
ax_cnt.text(0.50, -3.0, 'cos x − 3 sin y = 0', color=COLORS[1], fontsize=8.5)

for i, (s, hist) in enumerate(zip(solutions, detailed_hists)):
    c  = COLORS[i % len(COLORS)]
    hx = [h['x'] for h in hist]
    hy = [h['y'] for h in hist]
    ax_cnt.plot(hx, hy, 'o--', color=c, alpha=0.5, ms=3.5, lw=1.2)
    ax_cnt.scatter(*s, color=c, s=200, marker='*', zorder=6,
                   edgecolors=BG, lw=0.5,
                   label=f'S{i+1}: ({s[0]:.4f}, {s[1]:.4f})')

ax_cnt.axvline(np.pi/2, color='#444', lw=0.8, ls=':')
ax_cnt.text(np.pi/2 + 0.02, 3.1, 'π/2', color='#666', fontsize=7.5)
ax_cnt.axhline(0, color=DIM, lw=0.4, alpha=0.5)
ax_cnt.axvline(0, color=DIM, lw=0.4, alpha=0.5)
ax_cnt.set_xlim(0, 1.5);  ax_cnt.set_ylim(-3.5, 3.5)
ax_cnt.set_title('Curvas de nivel  (0 < x < 1.5)', color=WHITE, fontsize=10, pad=7)
ax_cnt.set_xlabel('x', fontsize=8);  ax_cnt.set_ylabel('y', fontsize=8)
ax_cnt.tick_params(labelsize=7)
ax_cnt.grid(True, alpha=0.3)
ax_cnt.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE,
              fontsize=7.5, loc='upper left')

# ── Panel 2: Mapa de calor ───────────────────
ax_heat = fig.add_subplot(gs[0, 1])
ax_heat.set_facecolor(PANEL)

im = ax_heat.pcolormesh(Xg, Yg, normF_clip, cmap=cmap_heat, shading='auto')
cb = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
cb.set_label('‖F(x,y)‖', color='gray', fontsize=8)
cb.ax.yaxis.set_tick_params(color='gray')
plt.setp(cb.ax.yaxis.get_ticklabels(), color='gray', fontsize=6.5)

ax_heat.contour(Xg, Yg, Z1, levels=[0], colors=[COLORS[0]], linewidths=1.3, alpha=0.7)
ax_heat.contour(Xg, Yg, Z2, levels=[0], colors=[COLORS[1]], linewidths=1.3, alpha=0.7)

for i, s in enumerate(solutions):
    ax_heat.scatter(*s, color=COLORS[i % len(COLORS)], s=130, marker='*',
                    zorder=5, edgecolors=BG, lw=0.5)
    ax_heat.annotate(f'S{i+1}', s, color=COLORS[i % len(COLORS)],
                     fontsize=7.5, textcoords='offset points', xytext=(5, 3))

ax_heat.set_xlim(0, 1.5);  ax_heat.set_ylim(-3.5, 3.5)
ax_heat.set_title('Mapa de calor  ‖F(x,y)‖', color=WHITE, fontsize=9, pad=5)
ax_heat.set_xlabel('x', fontsize=7.5);  ax_heat.set_ylabel('y', fontsize=7.5)
ax_heat.tick_params(labelsize=6.5)

# ── Panel 3: Convergencia ‖F‖ ───────────────
ax_err = fig.add_subplot(gs[1, 1])
ax_err.set_facecolor(PANEL)

for i, (s, hist) in enumerate(zip(solutions, detailed_hists)):
    c     = COLORS[i % len(COLORS)]
    norms = [h['err'] for h in hist]
    ks    = range(len(norms))
    ax_err.semilogy(ks, norms, 'o-', color=c, lw=1.8, ms=5,
                    markerfacecolor=BG, markeredgecolor=c, mew=1.1,
                    label=f'S{i+1}')
    ax_err.fill_between(ks, norms, alpha=0.06, color=c)

ax_err.set_title('Convergencia  ‖F(xₙ)‖', color=WHITE, fontsize=9, pad=5)
ax_err.set_xlabel('Iteración n', fontsize=7.5)
ax_err.set_ylabel('‖F‖  (log)', fontsize=7.5)
ax_err.tick_params(labelsize=6.5)
ax_err.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE,
              fontsize=7.5, ncol=2)
ax_err.grid(True, which='both', alpha=0.3)

# ── Panel 4 (2 filas): TABLA ─────────────────
ax_tbl = fig.add_subplot(gs[:, 2:])
ax_tbl.axis('off')
ax_tbl.set_title('Tabla de Iteraciones por Solución', color=GOLD,
                 fontsize=10, pad=6)

col_labels = ['Sol.', 'k', 'x', 'y', 'f₁(x,y)', 'f₂(x,y)', '‖F‖', 'det(J)']

all_rows   = []
row_colors = []

for i, (s, hist) in enumerate(zip(solutions, detailed_hists)):
    c    = COLORS[i % len(COLORS)]
    show = hist if len(hist) <= 8 else hist[:4] + [None] + hist[-3:]
    for h in show:
        if h is None:
            all_rows.append(['', '⋮', '⋮', '⋮', '⋮', '⋮', '⋮', '⋮'])
            row_colors.append('ellipsis')
        else:
            all_rows.append([
                f'S{i+1}',
                f"{hist.index(h)}",
                f"{h['x']:+.7f}",
                f"{h['y']:+.7f}",
                f"{h['f1']:+.3e}",
                f"{h['f2']:+.3e}",
                f"{h['err']:.3e}",
                f"{h['det']:+.5f}",
            ])
            row_colors.append((i, h['err']))

tbl = ax_tbl.table(
    cellText=all_rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0.01, 0.01, 0.98, 0.95]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.8)

for (r, c_), cell in tbl.get_celld().items():
    cell.set_edgecolor(DIM)
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor('#0d1e38')
        cell.set_text_props(color=GOLD, fontweight='bold')
    else:
        rc = row_colors[r - 1]
        if rc == 'ellipsis':
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=DIM)
        else:
            sol_idx, err_val = rc
            base_color = COLORS[sol_idx % len(COLORS)]
            if err_val < 1e-6:
                cell.set_facecolor('#081408')
                cell.set_text_props(color='#34d399')
            elif (r % 2) == 0:
                cell.set_facecolor('#0b0f1e')
                cell.set_text_props(color=WHITE)
            else:
                cell.set_facecolor(PANEL)
                cell.set_text_props(color=WHITE)
            # Columna "Sol." con color de cada solución
            if c_ == 0:
                cell.set_text_props(color=base_color, fontweight='bold')

# ── Cuadro resumen final ─────────────────────
lines = [f"SOLUCIONES ENCONTRADAS: {n_sol}", "─" * 44]
for i, s in enumerate(solutions):
    fv = F(s)
    lines.append(
        f"S{i+1}: x={s[0]:+.8f}  y={s[1]:+.8f}  "
        f"‖F‖={np.linalg.norm(fv):.2e}"
    )
fig.text(0.015, 0.01, '\n'.join(lines),
         fontsize=7.0, color='#34d399', fontfamily='monospace',
         va='bottom', alpha=0.88,
         bbox=dict(facecolor='#050f08', edgecolor='#00aa77',
                   boxstyle='round,pad=0.5', lw=0.8))

plt.savefig('problema25_tabla.png',
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.show()
plt.close()
print("✔  Figura estática guardada → problema25_tabla.png")

# ─────────────────────────────────────────────
# 6. ANIMACIÓN SUAVIZADA
# ─────────────────────────────────────────────

def ease_in_out_cubic(t):
    if t < 0.5:
        return 4*t*t*t
    p = 2*t - 2
    return 0.5*p*p*p + 1

N_FRAMES = 120
FPS      = 25

def interpolate_traj(hist, n_frames):
    pts   = np.array([[h['x'], h['y']] for h in hist])
    n     = len(pts)
    t_raw = np.linspace(0, 1, n)
    t_ani = np.linspace(0, 1, n_frames)
    eased = np.array([ease_in_out_cubic(t) for t in t_ani])
    return (np.interp(eased, t_raw, pts[:, 0]),
            np.interp(eased, t_raw, pts[:, 1]))

anim_trajs = [interpolate_traj(h, N_FRAMES) for h in detailed_hists]
err_trajs  = [[np.linalg.norm(F([x, y]))
               for x, y in zip(tx, ty)]
              for tx, ty in anim_trajs]

t_frames = np.linspace(0, 1, N_FRAMES)

# Layout: panel principal + convergencia + trayectoria xy
apply_dark_style()
fig_ani = plt.figure(figsize=(14, 7), dpi=100)
fig_ani.patch.set_facecolor(BG)

gs2 = GridSpec(2, 3, figure=fig_ani,
               left=0.05, right=0.97,
               top=0.88, bottom=0.08,
               hspace=0.55, wspace=0.40)

fig_ani.text(0.5, 0.955,
             'PROBLEMA 25  ·  TODAS LAS RAÍCES  ·  ANIMACIÓN NEWTON-RAPHSON',
             ha='center', fontsize=11, color=COLORS[0],
             fontweight='bold', fontfamily='monospace')
fig_ani.text(0.5, 0.932,
             r'$f_1: \tan(x)-y-1=0$     $f_2: \cos(x)-3\sin(y)=0$     $0 < x < 1.5$',
             ha='center', fontsize=8, color=WHITE, alpha=0.50)

ax_main = fig_ani.add_subplot(gs2[:, 0])
ax_err  = fig_ani.add_subplot(gs2[0, 1])
ax_traj = fig_ani.add_subplot(gs2[1, 1])
ax_f1   = fig_ani.add_subplot(gs2[0, 2])
ax_f2   = fig_ani.add_subplot(gs2[1, 2])

# Pre-calcular residuos f1 y f2 por solución
f1_trajs = [[F([x, y])[0] for x, y in zip(tx, ty)] for tx, ty in anim_trajs]
f2_trajs = [[F([x, y])[1] for x, y in zip(tx, ty)] for tx, ty in anim_trajs]

def update(frame):
    prog = frame / (N_FRAMES - 1)

    for ax in [ax_main, ax_err, ax_traj, ax_f1, ax_f2]:
        ax.cla()

    # ── Panel principal ────────────────────────
    ax_main.set_facecolor(PANEL)
    ax_main.set_xlim(0, 1.5);  ax_main.set_ylim(-3.5, 3.5)
    ax_main.grid(True, alpha=0.25)
    ax_main.axhline(0, color=DIM, lw=0.4, alpha=0.5)
    ax_main.axvline(0, color=DIM, lw=0.4, alpha=0.5)
    ax_main.axvline(np.pi/2, color='#444', lw=0.7, ls=':')
    ax_main.text(np.pi/2 + 0.02, 3.15, 'π/2', color='#555', fontsize=7)

    ax_main.contour(Xg, Yg, Z1, levels=[0], colors=[COLORS[0]],
                    linewidths=1.8, alpha=0.8)
    ax_main.contour(Xg, Yg, Z2, levels=[0], colors=[COLORS[1]],
                    linewidths=1.8, alpha=0.8)

    trail = max(1, frame)
    for i, ((tx, ty), s) in enumerate(zip(anim_trajs, solutions)):
        c = COLORS[i % len(COLORS)]

        # Trail
        for j in range(1, trail):
            a = (j / trail) ** 1.5 * 0.8
            ax_main.plot([tx[j-1], tx[j]], [ty[j-1], ty[j]],
                         color=c, lw=1.3, alpha=a, solid_capstyle='round')

        # Punto actual
        ax_main.scatter(tx[frame], ty[frame],
                        color=c, s=70, zorder=7,
                        edgecolors=BG, lw=1.0)

        # Solución objetivo (tenue)
        ax_main.scatter(*s, color=c, s=140, marker='*', zorder=8,
                        edgecolors=BG, lw=0.5, alpha=0.5)

    # Barra de progreso
    bar_y = 3.3
    ax_main.plot([0.02, 1.48], [bar_y, bar_y],
                 color=DIM, lw=3, solid_capstyle='round', zorder=6)
    ax_main.plot([0.02, 0.02 + 1.46*prog], [bar_y, bar_y],
                 color=COLORS[0], lw=3, solid_capstyle='round', zorder=7, alpha=0.85)

    ax_main.set_title(
        f'Frame {frame+1}/{N_FRAMES}  —  {n_sol} soluciones',
        color=WHITE, fontsize=8, pad=4
    )
    ax_main.tick_params(labelsize=6.5)
    ax_main.set_xlabel('x', fontsize=7.5)
    ax_main.set_ylabel('y', fontsize=7.5)

    # ── Convergencia ‖F‖ ──────────────────────
    ax_err.set_facecolor(PANEL)
    for i, (errs, s) in enumerate(zip(err_trajs, solutions)):
        c = COLORS[i % len(COLORS)]
        ax_err.semilogy(t_frames, errs, color=c, lw=0.7, alpha=0.2)
        ax_err.semilogy(t_frames[:frame+1], errs[:frame+1], color=c, lw=1.8, label=f'S{i+1}')
        ax_err.scatter(t_frames[frame], errs[frame], color=c, s=35, zorder=5)
    ax_err.set_title('‖F(xₙ)‖', color=WHITE, fontsize=8.5, pad=3)
    ax_err.set_xlabel('Progreso', fontsize=6.5)
    ax_err.tick_params(labelsize=6)
    ax_err.legend(fontsize=6, framealpha=0.15, facecolor=PANEL,
                  edgecolor=DIM, labelcolor=WHITE, ncol=2)
    ax_err.grid(True, alpha=0.25, which='both')

    # ── Trayectorias (x,y) ────────────────────
    ax_traj.set_facecolor(PANEL)
    for i, ((tx, ty), s) in enumerate(zip(anim_trajs, solutions)):
        c = COLORS[i % len(COLORS)]
        ax_traj.plot(tx, ty, color=c, lw=0.6, alpha=0.18)
        if frame > 0:
            ax_traj.plot(tx[:frame+1], ty[:frame+1], color=c, lw=1.6)
        ax_traj.scatter(tx[frame], ty[frame], color=c, s=35, zorder=5)
        ax_traj.scatter(*s, color=c, s=80, marker='*', zorder=6, alpha=0.5)
    ax_traj.set_title('Camino (x, y)', color=WHITE, fontsize=8.5, pad=3)
    ax_traj.set_xlabel('x', fontsize=6.5);  ax_traj.set_ylabel('y', fontsize=6.5)
    ax_traj.tick_params(labelsize=6)
    ax_traj.grid(True, alpha=0.25)

    # ── Residuos f₁ ───────────────────────────
    ax_f1.set_facecolor(PANEL)
    for i, f1v in enumerate(f1_trajs):
        c = COLORS[i % len(COLORS)]
        ax_f1.plot(t_frames, f1v, color=c, lw=0.7, alpha=0.2)
        ax_f1.plot(t_frames[:frame+1], f1v[:frame+1], color=c, lw=1.6, label=f'S{i+1}')
        ax_f1.scatter(t_frames[frame], f1v[frame], color=c, s=30, zorder=5)
    ax_f1.axhline(0, color=WHITE, lw=0.6, ls='--', alpha=0.35)
    ax_f1.set_title('Residuo  f₁', color=COLORS[0], fontsize=8.5, pad=3)
    ax_f1.set_xlabel('Progreso', fontsize=6.5)
    ax_f1.tick_params(labelsize=6)
    ax_f1.legend(fontsize=6, framealpha=0.15, facecolor=PANEL,
                 edgecolor=DIM, labelcolor=WHITE, ncol=2)
    ax_f1.grid(True, alpha=0.25)

    # ── Residuos f₂ ───────────────────────────
    ax_f2.set_facecolor(PANEL)
    for i, f2v in enumerate(f2_trajs):
        c = COLORS[i % len(COLORS)]
        ax_f2.plot(t_frames, f2v, color=c, lw=0.7, alpha=0.2)
        ax_f2.plot(t_frames[:frame+1], f2v[:frame+1], color=c, lw=1.6, label=f'S{i+1}')
        ax_f2.scatter(t_frames[frame], f2v[frame], color=c, s=30, zorder=5)
    ax_f2.axhline(0, color=WHITE, lw=0.6, ls='--', alpha=0.35)
    ax_f2.set_title('Residuo  f₂', color=COLORS[1], fontsize=8.5, pad=3)
    ax_f2.set_xlabel('Progreso', fontsize=6.5)
    ax_f2.tick_params(labelsize=6)
    ax_f2.legend(fontsize=6, framealpha=0.15, facecolor=PANEL,
                 edgecolor=DIM, labelcolor=WHITE, ncol=2)
    ax_f2.grid(True, alpha=0.25)

    return []

ani = animation.FuncAnimation(
    fig_ani, update, frames=N_FRAMES,
    interval=1000 // FPS, blit=False
)

print("  Renderizando animación (puede tardar ~30 s)…")
ani.save('problema25_animacion.gif',
         writer=PillowWriter(fps=FPS),
         savefig_kwargs={'facecolor': BG})
plt.close(fig_ani)
print("✔  Animación guardada → problema25_animacion.gif")
print("\n  Archivos generados:")
print("    → problema25_tabla.png")
print("    → problema25_animacion.gif")