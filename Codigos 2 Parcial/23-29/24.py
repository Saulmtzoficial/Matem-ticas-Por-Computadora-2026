"""
Problema 24 - Sistema trigonométrico  (v2 — tabla + animación)
===============================================================
  f1(x, y) = sin(x) + 3·cos(x) − 2       = 0
  f2(x, y) = cos(x) − sin(y)  + 0.2      = 0
Punto inicial: (1, 1)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.gridspec import GridSpec
import math

# ─────────────────────────────────────────────
#  PALETA OSCURA
# ─────────────────────────────────────────────

BG     = '#050810'
PANEL  = '#0b0e1a'
GRID   = '#0f1528'
CYAN   = '#38bdf8'
ORANGE = '#fb923c'
TEAL   = '#00ffb3'
RED    = '#ff3c5f'
DIM    = '#1e3358'
WHITE  = '#c8dff5'
GOLD   = '#facc15'
PURPLE = '#c084fc'

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
# 1. SISTEMA F y JACOBIANO J
# ─────────────────────────────────────────────

def F(v):
    x, y = v
    return np.array([
        np.sin(x) + 3*np.cos(x) - 2,
        np.cos(x) - np.sin(y) + 0.2
    ])

def J(v):
    x, y = v
    return np.array([
        [np.cos(x) - 3*np.sin(x),  0.0      ],
        [-np.sin(x),               -np.cos(y)]
    ])

# ─────────────────────────────────────────────
# 2. NEWTON-RAPHSON (guarda historial completo)
# ─────────────────────────────────────────────

def newton_raphson(x0, tol=1e-10, max_iter=50):
    x = np.array(x0, dtype=float)
    history = []

    for i in range(max_iter):
        Fval = F(x)
        Jval = J(x)
        detJ = np.linalg.det(Jval)
        err  = np.linalg.norm(Fval)

        history.append({
            'k':   i,
            'x':   x[0],
            'y':   x[1],
            'f1':  Fval[0],
            'f2':  Fval[1],
            'err': err,
            'det': detJ,
        })

        if err < tol:
            break

        delta = np.linalg.solve(Jval, -Fval)
        x = x + delta

    return x, history

# ─────────────────────────────────────────────
# 3. RESOLVER
# ─────────────────────────────────────────────

x0  = [1.0, 1.0]
sol, hist = newton_raphson(x0)

print(f"Solución: x = {sol[0]:.10f},  y = {sol[1]:.10f}")
print(f"Verificación: F(sol) = {F(sol)}")

# ─────────────────────────────────────────────
# 4. FIGURA ESTÁTICA: curvas + tabla + convergencia
# ─────────────────────────────────────────────

apply_dark_style()

fig = plt.figure(figsize=(20, 9), dpi=130)
fig.patch.set_facecolor(BG)

gs = GridSpec(2, 4, figure=fig,
              left=0.04, right=0.98,
              top=0.90, bottom=0.07,
              hspace=0.52, wspace=0.38)

# ── Título ──────────────────────────────────
fig.text(0.5, 0.962,
         'PROBLEMA 24  ·  SISTEMA TRIGONOMÉTRICO  ·  NEWTON-RAPHSON MATRICIAL',
         ha='center', fontsize=13, color=CYAN,
         fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.940,
         r'$f_1: \sin(x)+3\cos(x)-2=0$     $f_2: \cos(x)-\sin(y)+0.2=0$     $x_0=(1,1)$',
         ha='center', fontsize=9, color=WHITE, alpha=0.55)

# ── Panel 1: Curvas de nivel ─────────────────
ax_cnt = fig.add_subplot(gs[:, 0])
ax_cnt.set_facecolor(PANEL)

xg = np.linspace(-2, 3, 800)
yg = np.linspace(-2, 3, 800)
Xg, Yg = np.meshgrid(xg, yg)
Z1 = np.sin(Xg) + 3*np.cos(Xg) - 2
Z2 = np.cos(Xg) - np.sin(Yg)   + 0.2

ax_cnt.contour(Xg, Yg, Z1, levels=[0], colors=[CYAN],   linewidths=2.2)
ax_cnt.contour(Xg, Yg, Z2, levels=[0], colors=[ORANGE], linewidths=2.2)

hx = [h['x'] for h in hist]
hy = [h['y'] for h in hist]
ax_cnt.plot(hx, hy, 'o--', color='white', alpha=0.45, ms=4, lw=1.2, zorder=3)
ax_cnt.scatter(hx[0], hy[0], color='white', s=70, marker='s',
               zorder=5, label=f'x₀ = ({x0[0]}, {x0[1]})')
ax_cnt.scatter(*sol, color=GOLD, s=220, marker='*', zorder=6,
               edgecolors=BG, lw=0.5,
               label=f'Solución\n({sol[0]:.4f}, {sol[1]:.4f})')

ax_cnt.axhline(0, color=DIM, lw=0.5, alpha=0.5)
ax_cnt.axvline(0, color=DIM, lw=0.5, alpha=0.5)
ax_cnt.set_xlim(-2, 3);  ax_cnt.set_ylim(-2, 3)
ax_cnt.set_aspect('equal')
ax_cnt.set_title('Curvas de nivel  f₁=0  y  f₂=0', color=WHITE, fontsize=10, pad=7)
ax_cnt.set_xlabel('x', fontsize=8);  ax_cnt.set_ylabel('y', fontsize=8)
ax_cnt.tick_params(labelsize=7)
ax_cnt.grid(True, alpha=0.3)
ax_cnt.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE,
              fontsize=8, loc='lower right')
ax_cnt.text(1.8,  1.2, 'f₁=0', color=CYAN,   fontsize=8.5, rotation=-30)
ax_cnt.text(0.4, -1.5, 'f₂=0', color=ORANGE, fontsize=8.5)

# ── Panel 2: Convergencia ‖F‖ ───────────────
ax_err = fig.add_subplot(gs[0, 1])
ax_err.set_facecolor(PANEL)

ks   = [h['k']   for h in hist]
errs = [h['err'] for h in hist]
ax_err.semilogy(ks, errs, 'o-', color=GOLD, lw=2, ms=6,
                markerfacecolor=BG, markeredgecolor=GOLD, mew=1.2)
ax_err.fill_between(ks, errs, alpha=0.08, color=GOLD)
for i, r in enumerate(errs):
    if r > 1e-13:
        ax_err.annotate(f'{r:.1e}', (i, r),
                        textcoords='offset points', xytext=(5, 3),
                        color=WHITE, fontsize=6.5, alpha=0.8)
ax_err.set_title('Convergencia  ‖F(xₙ)‖', color=WHITE, fontsize=9, pad=5)
ax_err.set_xlabel('Iteración n', fontsize=7.5)
ax_err.set_ylabel('‖F‖  (log)', fontsize=7.5)
ax_err.tick_params(labelsize=7)
ax_err.set_xticks(ks)
ax_err.grid(True, which='both', alpha=0.3)

# ── Panel 3: Evolución de x e y ─────────────
ax_xy = fig.add_subplot(gs[1, 1])
ax_xy.set_facecolor(PANEL)

ax_xy.plot(ks, hx, 'o-', color=CYAN,   lw=2, ms=6,
           markerfacecolor=BG, label='x')
ax_xy.plot(ks, hy, 's-', color=ORANGE, lw=2, ms=6,
           markerfacecolor=BG, label='y')
ax_xy.axhline(sol[0], color=CYAN,   lw=0.8, ls='--', alpha=0.5)
ax_xy.axhline(sol[1], color=ORANGE, lw=0.8, ls='--', alpha=0.5)
ax_xy.annotate(f'x*={sol[0]:.5f}',
               xy=(ks[-1], sol[0]), xytext=(0.35, 0.22),
               textcoords='axes fraction', color=CYAN, fontsize=7.5,
               arrowprops=dict(arrowstyle='->', color=CYAN, lw=0.8))
ax_xy.annotate(f'y*={sol[1]:.5f}',
               xy=(ks[-1], sol[1]), xytext=(0.35, 0.70),
               textcoords='axes fraction', color=ORANGE, fontsize=7.5,
               arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8))
ax_xy.set_title('Evolución de x  e  y', color=WHITE, fontsize=9, pad=5)
ax_xy.set_xlabel('Iteración n', fontsize=7.5)
ax_xy.set_ylabel('Valor', fontsize=7.5)
ax_xy.tick_params(labelsize=7)
ax_xy.set_xticks(ks)
ax_xy.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE, fontsize=8)
ax_xy.grid(True, alpha=0.3)

# ── Panel 4 (ocupa 2 filas): TABLA ──────────
ax_tbl = fig.add_subplot(gs[:, 2:])
ax_tbl.axis('off')
ax_tbl.set_facecolor(PANEL)
ax_tbl.set_title('Tabla de Iteraciones — Newton-Raphson',
                 color=GOLD, fontsize=10, pad=6)

col_labels = ['k', 'x', 'y', 'f₁(x,y)', 'f₂(x,y)', '‖F‖', 'det(J)']

def make_rows(history):
    rows = []
    show = history if len(history) <= 12 else history[:6] + [None] + history[-4:]
    for h in show:
        if h is None:
            rows.append(['⋮'] * len(col_labels))
            continue
        rows.append([
            f"{h['k']}",
            f"{h['x']:+.8f}",
            f"{h['y']:+.8f}",
            f"{h['f1']:+.4e}",
            f"{h['f2']:+.4e}",
            f"{h['err']:.4e}",
            f"{h['det']:+.6f}",
        ])
    return rows

rows = make_rows(hist)

tbl = ax_tbl.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0.02, 0.02, 0.96, 0.92]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(DIM)
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor('#0d1e38')
        cell.set_text_props(color=GOLD, fontweight='bold')
    elif rows[r-1][0] == '⋮':
        cell.set_facecolor(PANEL)
        cell.set_text_props(color=DIM)
    else:
        idx = r - 1
        try:
            err_val = float(rows[idx][5])
        except Exception:
            err_val = 1.0
        if err_val < 1e-6:
            cell.set_facecolor('#0d1a10')
            cell.set_text_props(color=TEAL)
        elif idx % 2 == 0:
            cell.set_facecolor('#0b0f1e')
            cell.set_text_props(color=WHITE)
        else:
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=WHITE)

# ── Cuadro de solución final ──────────────────
sol_text = (
    f"SOLUCIÓN ENCONTRADA  |  Iteraciones: {len(hist)}\n"
    f"─────────────────────────────────────────\n"
    f"  x* = {sol[0]:+.10f}\n"
    f"  y* = {sol[1]:+.10f}\n"
    f"  f₁(x*,y*) = {F(sol)[0]:.3e}\n"
    f"  f₂(x*,y*) = {F(sol)[1]:.3e}\n"
    f"  ‖F‖ = {np.linalg.norm(F(sol)):.3e}"
)
fig.text(0.015, 0.01, sol_text,
         fontsize=7.5, color=TEAL, fontfamily='monospace',
         va='bottom', alpha=0.90,
         bbox=dict(facecolor='#050f08', edgecolor='#00aa77',
                   boxstyle='round,pad=0.5', lw=0.8))

plt.savefig('sistema_trigonometrico_tabla.png',
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.show()
plt.close()
print("✔  Figura estática guardada → sistema_trigonometrico_tabla.png")

# ─────────────────────────────────────────────
# 5. ANIMACIÓN SUAVIZADA (GIF)
# ─────────────────────────────────────────────

def ease_in_out_cubic(t):
    if t < 0.5:
        return 4*t*t*t
    p = 2*t - 2
    return 0.5*p*p*p + 1

N_FRAMES = 110
FPS      = 25

# Interpolación suavizada sobre el historial
pts    = np.array([[h['x'], h['y']] for h in hist])
n_pts  = len(pts)
t_raw  = np.linspace(0, 1, n_pts)
t_anim = np.linspace(0, 1, N_FRAMES)
eased  = np.array([ease_in_out_cubic(t) for t in t_anim])

x_anim = np.interp(eased, t_raw, pts[:, 0])
y_anim = np.interp(eased, t_raw, pts[:, 1])
err_anim = [np.linalg.norm(F([x, y])) for x, y in zip(x_anim, y_anim)]

# Pre-calcular curvas de nivel (estáticas)
xg = np.linspace(-2, 3, 600)
yg = np.linspace(-2, 3, 600)
Xg, Yg = np.meshgrid(xg, yg)
Z1 = np.sin(Xg) + 3*np.cos(Xg) - 2
Z2 = np.cos(Xg) - np.sin(Yg)   + 0.2

t_frames = np.linspace(0, 1, N_FRAMES)

apply_dark_style()
fig_ani = plt.figure(figsize=(14, 7), dpi=100)
fig_ani.patch.set_facecolor(BG)

gs2 = GridSpec(2, 3, figure=fig_ani,
               left=0.05, right=0.97,
               top=0.88, bottom=0.08,
               hspace=0.55, wspace=0.40)

fig_ani.text(0.5, 0.955,
             'SISTEMA TRIGONOMÉTRICO  ·  ANIMACIÓN NEWTON-RAPHSON',
             ha='center', fontsize=11, color=CYAN,
             fontweight='bold', fontfamily='monospace')
fig_ani.text(0.5, 0.933,
             r'$f_1: \sin(x)+3\cos(x)-2=0$     $f_2: \cos(x)-\sin(y)+0.2=0$     $x_0=(1,1)$',
             ha='center', fontsize=8, color=WHITE, alpha=0.5)

ax_main = fig_ani.add_subplot(gs2[:, 0])
ax_err  = fig_ani.add_subplot(gs2[0, 1])
ax_xy   = fig_ani.add_subplot(gs2[1, 1])
ax_f1   = fig_ani.add_subplot(gs2[0, 2])
ax_f2   = fig_ani.add_subplot(gs2[1, 2])

f1_anim = [F([x, y])[0] for x, y in zip(x_anim, y_anim)]
f2_anim = [F([x, y])[1] for x, y in zip(x_anim, y_anim)]

def update(frame):
    prog = frame / (N_FRAMES - 1)

    for ax in [ax_main, ax_err, ax_xy, ax_f1, ax_f2]:
        ax.cla()

    # ── Panel principal: curvas de nivel + camino ──
    ax_main.set_facecolor(PANEL)
    ax_main.set_xlim(-2, 3);  ax_main.set_ylim(-2, 3)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.25)
    ax_main.axhline(0, color=DIM, lw=0.4, alpha=0.5)
    ax_main.axvline(0, color=DIM, lw=0.4, alpha=0.5)

    # Curvas de nivel (siempre visibles)
    ax_main.contour(Xg, Yg, Z1, levels=[0], colors=[CYAN],   linewidths=2.0, alpha=0.85)
    ax_main.contour(Xg, Yg, Z2, levels=[0], colors=[ORANGE], linewidths=2.0, alpha=0.85)
    ax_main.text(1.7,  1.1, 'f₁=0', color=CYAN,   fontsize=8, rotation=-30, alpha=0.7)
    ax_main.text(0.3, -1.4, 'f₂=0', color=ORANGE, fontsize=8, alpha=0.7)

    # Punto inicial fantasma
    ax_main.scatter(x0[0], x0[1], color='white', s=55, marker='s',
                    zorder=4, alpha=0.4, edgecolors=BG, lw=1)

    # Trail suavizado
    trail = max(1, frame)
    for i in range(1, trail):
        a = (i / trail) ** 1.5 * 0.8
        ax_main.plot([x_anim[i-1], x_anim[i]],
                     [y_anim[i-1], y_anim[i]],
                     color=GOLD, lw=1.4, alpha=a, solid_capstyle='round')

    # Punto actual
    ax_main.scatter(x_anim[frame], y_anim[frame],
                    color=GOLD, s=90, zorder=8,
                    edgecolors=BG, lw=1.2)

    # Línea punteada hasta la solución
    err_now = math.sqrt((x_anim[frame]-sol[0])**2 + (y_anim[frame]-sol[1])**2)
    if err_now > 0.005:
        ax_main.plot([x_anim[frame], sol[0]],
                     [y_anim[frame], sol[1]],
                     color=RED, lw=0.9, ls=':', alpha=0.5)

    # Solución final (objetivo)
    ax_main.scatter(*sol, color=GOLD, s=180, marker='*', zorder=9,
                    edgecolors=BG, lw=0.5, alpha=0.55)

    # Barra de progreso
    bar_y = 2.85
    ax_main.plot([-1.8, 2.8], [bar_y, bar_y],
                 color=DIM, lw=3, solid_capstyle='round', zorder=6)
    ax_main.plot([-1.8, -1.8 + 4.6*prog], [bar_y, bar_y],
                 color=CYAN, lw=3, solid_capstyle='round', zorder=7, alpha=0.8)

    ax_main.set_title(
        f'Iter. frame {frame+1}/{N_FRAMES}\n'
        f'x={x_anim[frame]:.5f}  y={y_anim[frame]:.5f}  ‖F‖={err_anim[frame]:.3e}',
        color=WHITE, fontsize=7.8, pad=4
    )
    ax_main.tick_params(labelsize=6.5)
    ax_main.set_xlabel('x', fontsize=7.5)
    ax_main.set_ylabel('y', fontsize=7.5)

    # ── Convergencia ‖F‖ ──────────────────────
    ax_err.set_facecolor(PANEL)
    ax_err.semilogy(t_frames, err_anim, color=GOLD, lw=0.8, alpha=0.2)
    ax_err.semilogy(t_frames[:frame+1], err_anim[:frame+1],
                    color=GOLD, lw=1.8)
    ax_err.scatter(t_frames[frame], err_anim[frame],
                   color=GOLD, s=40, zorder=5)
    ax_err.fill_between(t_frames[:frame+1], err_anim[:frame+1],
                        alpha=0.08, color=GOLD)
    ax_err.set_title('‖F(x,y)‖', color=GOLD, fontsize=8.5, pad=3)
    ax_err.set_xlabel('Progreso', fontsize=6.5)
    ax_err.tick_params(labelsize=6)
    ax_err.grid(True, alpha=0.25, which='both')

    # ── Evolución x e y ───────────────────────
    ax_xy.set_facecolor(PANEL)
    ax_xy.plot(t_frames, x_anim, color=CYAN,   lw=0.8, alpha=0.2)
    ax_xy.plot(t_frames, y_anim, color=ORANGE, lw=0.8, alpha=0.2)
    ax_xy.plot(t_frames[:frame+1], x_anim[:frame+1], color=CYAN,   lw=1.8, label='x')
    ax_xy.plot(t_frames[:frame+1], y_anim[:frame+1], color=ORANGE, lw=1.8, label='y')
    ax_xy.axvline(prog, color=WHITE, lw=0.5, alpha=0.35)
    ax_xy.axhline(sol[0], color=CYAN,   lw=0.6, ls='--', alpha=0.4)
    ax_xy.axhline(sol[1], color=ORANGE, lw=0.6, ls='--', alpha=0.4)
    ax_xy.set_title('Evolución x, y', color=WHITE, fontsize=8.5, pad=3)
    ax_xy.set_xlabel('Progreso', fontsize=6.5)
    ax_xy.tick_params(labelsize=6)
    ax_xy.legend(fontsize=6.5, framealpha=0.15,
                 facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE)
    ax_xy.grid(True, alpha=0.25)

    # ── Residuo f₁ ────────────────────────────
    ax_f1.set_facecolor(PANEL)
    ax_f1.plot(t_frames, f1_anim, color=CYAN, lw=0.8, alpha=0.2)
    ax_f1.plot(t_frames[:frame+1], f1_anim[:frame+1], color=CYAN, lw=1.8)
    ax_f1.scatter(t_frames[frame], f1_anim[frame], color=CYAN, s=40, zorder=5)
    ax_f1.axhline(0, color=WHITE, lw=0.6, ls='--', alpha=0.4)
    ax_f1.fill_between(t_frames[:frame+1], f1_anim[:frame+1],
                       alpha=0.07, color=CYAN)
    ax_f1.set_title('Residuo f₁', color=CYAN, fontsize=8.5, pad=3)
    ax_f1.set_xlabel('Progreso', fontsize=6.5)
    ax_f1.tick_params(labelsize=6)
    ax_f1.grid(True, alpha=0.25)

    # ── Residuo f₂ ────────────────────────────
    ax_f2.set_facecolor(PANEL)
    ax_f2.plot(t_frames, f2_anim, color=ORANGE, lw=0.8, alpha=0.2)
    ax_f2.plot(t_frames[:frame+1], f2_anim[:frame+1], color=ORANGE, lw=1.8)
    ax_f2.scatter(t_frames[frame], f2_anim[frame], color=ORANGE, s=40, zorder=5)
    ax_f2.axhline(0, color=WHITE, lw=0.6, ls='--', alpha=0.4)
    ax_f2.fill_between(t_frames[:frame+1], f2_anim[:frame+1],
                       alpha=0.07, color=ORANGE)
    ax_f2.set_title('Residuo f₂', color=ORANGE, fontsize=8.5, pad=3)
    ax_f2.set_xlabel('Progreso', fontsize=6.5)
    ax_f2.tick_params(labelsize=6)
    ax_f2.grid(True, alpha=0.25)

    return []

ani = animation.FuncAnimation(
    fig_ani, update, frames=N_FRAMES,
    interval=1000//FPS, blit=False
)

print("  Renderizando animación…")
ani.save('sistema_trigonometrico_animacion.gif',
         writer=PillowWriter(fps=FPS),
         savefig_kwargs={'facecolor': BG})
plt.close(fig_ani)
print("✔  Animación guardada → sistema_trigonometrico_animacion.gif")
print("\n  Archivos generados:")
print("    → sistema_trigonometrico_tabla.png")
print("    → sistema_trigonometrico_animacion.gif")