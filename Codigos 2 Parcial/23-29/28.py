"""
Problema 28 — Proyectil: determinar v, θ y tiempo de vuelo  (v2)
=================================================================
  f1 = v·cosθ·t − 300                        = 0
  f2 = −½g·t² + v·sinθ·t − 61               = 0
  f3 = −g·t + v·sinθ + v·cosθ               = 0
Sistema no lineal 3×3  →  Newton-Raphson matricial
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.gridspec import GridSpec
import math
import os

# Guardar en la misma carpeta que este script
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
def out(filename):
    return os.path.join(OUT_DIR, filename)

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
GREEN  = '#4ade80'

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

g = 9.81  # m/s²

def F(q):
    v, theta, t = q
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        v*c*t - 300,
        -0.5*g*t**2 + v*s*t - 61,
        -g*t + v*s + v*c
    ])

def J(q):
    v, theta, t = q
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c*t,      -v*s*t,       v*c        ],
        [s*t,       v*c*t,      -g*t + v*s  ],
        [s+c,       v*(c-s),    -g          ]
    ])

# ─────────────────────────────────────────────
# 2. NEWTON-RAPHSON (guarda historial completo)
# ─────────────────────────────────────────────

def newton_raphson(q0, tol=1e-10, max_iter=50):
    q = np.array(q0, dtype=float)
    history = []

    for i in range(max_iter):
        Fval = F(q)
        Jval = J(q)
        detJ = np.linalg.det(Jval)
        err  = np.linalg.norm(Fval)

        history.append({
            'k':    i,
            'v':    q[0],
            'th':   math.degrees(q[1]),
            't':    q[2],
            'f1':   Fval[0],
            'f2':   Fval[1],
            'f3':   Fval[2],
            'err':  err,
            'det':  detJ,
        })

        if err < tol:
            break

        delta = np.linalg.solve(Jval, -Fval)
        q = q + delta

    return q, history

# ─────────────────────────────────────────────
# 3. RESOLVER
# ─────────────────────────────────────────────

q0  = [50.0, math.radians(40.0), 8.0]
sol, hist = newton_raphson(q0)

v_sol  = sol[0]
th_sol = math.degrees(sol[1])
t_sol  = sol[2]

# Verificación
x_chk  = v_sol * math.cos(sol[1]) * t_sol
y_chk  = -0.5*g*t_sol**2 + v_sol*math.sin(sol[1])*t_sol
vx_imp = v_sol * math.cos(sol[1])
vy_imp = -g*t_sol + v_sol*math.sin(sol[1])
ang_imp = math.degrees(math.atan2(vy_imp, vx_imp))

print(f"v  = {v_sol:.8f} m/s")
print(f"θ  = {th_sol:.8f} °")
print(f"t  = {t_sol:.8f} s")
print(f"x(t)={x_chk:.4f} m   y(t)={y_chk:.4f} m   ángulo impacto={ang_imp:.4f}°")

# Trayectoria completa
t_arr = np.linspace(0, t_sol, 600)
x_arr = v_sol*math.cos(sol[1])*t_arr
y_arr = -0.5*g*t_arr**2 + v_sol*math.sin(sol[1])*t_arr
y_max = float(np.max(y_arr))
x_apex = float(x_arr[np.argmax(y_arr)])

# ─────────────────────────────────────────────
# 4. FIGURA ESTÁTICA — Trayectoria + Tabla + Análisis
# ─────────────────────────────────────────────

apply_dark_style()

fig = plt.figure(figsize=(22, 10), dpi=130)
fig.patch.set_facecolor(BG)

gs = GridSpec(2, 4, figure=fig,
              left=0.04, right=0.98,
              top=0.90, bottom=0.07,
              hspace=0.52, wspace=0.40)

# ── Título ──────────────────────────────────
fig.text(0.5, 0.962,
         'PROBLEMA 28  ·  PROYECTIL: v, θ, t  ·  NEWTON-RAPHSON 3×3',
         ha='center', fontsize=13, color=CYAN,
         fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.940,
         'f₁: v·cosθ·t=300     f₂: −½gt²+v·sinθ·t=61     f₃: ángulo de impacto 45°',
         ha='center', fontsize=9, color=WHITE, alpha=0.55)

# ── Panel 1: Trayectoria ─────────────────────
ax_traj = fig.add_subplot(gs[:, 0])
ax_traj.set_facecolor(PANEL)

ax_traj.plot(x_arr, y_arr, color=CYAN, lw=2.5, label='Trayectoria', zorder=3)

# Suelo
ax_traj.axhline(0, color=DIM, lw=0.8, alpha=0.6)

# Lanzamiento
ax_traj.scatter([0], [0], color=GREEN, s=110, zorder=7,
                edgecolors=BG, lw=1.3, label='Lanzamiento')

# Vértice
ax_traj.scatter([x_apex], [y_max], color=TEAL, s=80, zorder=6,
                edgecolors=BG, lw=1.2,
                label=f'Vértice ({x_apex:.1f}, {y_max:.1f}) m')
ax_traj.plot([x_apex, x_apex], [0, y_max],
             color=TEAL, lw=0.8, ls='--', alpha=0.4)

# Blanco (300, 61)
ax_traj.scatter([300], [61], color=GOLD, s=220, marker='*', zorder=8,
                edgecolors=BG, lw=0.5, label='Blanco (300, 61) m')

# Muro vertical
ax_traj.axvline(300, color=WHITE, lw=1.2, ls='--', alpha=0.35)

# Flecha de impacto a 45°
ax_traj.annotate('', xy=(300+20, 61-20), xytext=(300, 61),
                 arrowprops=dict(arrowstyle='-|>', color=ORANGE,
                                 lw=2.0, mutation_scale=14))
ax_traj.text(304, 56, '45°\nimpacto', color=ORANGE, fontsize=8)

# Vector velocidad inicial
scale = 15 / v_sol
vx0 = v_sol*math.cos(sol[1]); vy0 = v_sol*math.sin(sol[1])
ax_traj.annotate('', xy=(vx0*scale, vy0*scale), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='-|>', color=GREEN,
                                 lw=2.0, mutation_scale=12))
ax_traj.text(3, vy0*scale*0.55,
             f'v={v_sol:.2f} m/s\nθ={th_sol:.2f}°',
             color=GREEN, fontsize=8)

ax_traj.set_xlim(-15, 340)
ax_traj.set_ylim(-8, y_max * 1.20)
ax_traj.set_title('Trayectoria del Proyectil', color=WHITE, fontsize=10, pad=7)
ax_traj.set_xlabel('x (m)', fontsize=8)
ax_traj.set_ylabel('y (m)', fontsize=8)
ax_traj.tick_params(labelsize=7)
ax_traj.grid(True, alpha=0.3)
ax_traj.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE,
               fontsize=8, loc='upper left')

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
ax_err.set_title('Convergencia  ‖F(q)‖', color=WHITE, fontsize=9, pad=5)
ax_err.set_xlabel('Iteración n', fontsize=7.5)
ax_err.set_ylabel('‖F‖  (log)', fontsize=7.5)
ax_err.tick_params(labelsize=7)
ax_err.set_xticks(ks)
ax_err.grid(True, which='both', alpha=0.3)

# ── Panel 3: Evolución de v, θ, t ───────────
ax_evo = fig.add_subplot(gs[1, 1])
ax_evo.set_facecolor(PANEL)

v_h  = [h['v']  for h in hist]
th_h = [h['th'] for h in hist]
t_h  = [h['t']  for h in hist]

ax_evo.plot(ks, v_h,  'o-', color=CYAN,   lw=2, ms=5,
            markerfacecolor=BG, label='v (m/s)')
ax_evo.plot(ks, th_h, 's-', color=ORANGE, lw=2, ms=5,
            markerfacecolor=BG, label='θ (°)')

ax_evo2 = ax_evo.twinx()
ax_evo2.plot(ks, t_h, '^-', color=PURPLE, lw=2, ms=5,
             markerfacecolor=BG, label='t (s)')
ax_evo2.set_ylabel('t (s)', color=PURPLE, fontsize=7.5)
ax_evo2.tick_params(colors=PURPLE, labelsize=7)
for sp in ax_evo2.spines.values():
    sp.set_edgecolor(DIM)

ax_evo.axhline(v_sol,  color=CYAN,   lw=0.7, ls='--', alpha=0.4)
ax_evo.axhline(th_sol, color=ORANGE, lw=0.7, ls='--', alpha=0.4)
ax_evo2.axhline(t_sol, color=PURPLE, lw=0.7, ls='--', alpha=0.4)
ax_evo.set_title('Evolución  v, θ, t', color=WHITE, fontsize=9, pad=5)
ax_evo.set_xlabel('Iteración n', fontsize=7.5)
ax_evo.set_ylabel('v (m/s)  /  θ (°)', fontsize=7.5)
ax_evo.tick_params(labelsize=7)
ax_evo.set_xticks(ks)
ax_evo.grid(True, alpha=0.3)
lines1, labs1 = ax_evo.get_legend_handles_labels()
lines2, labs2 = ax_evo2.get_legend_handles_labels()
ax_evo.legend(lines1+lines2, labs1+labs2,
              facecolor=PANEL, edgecolor=DIM,
              labelcolor=WHITE, fontsize=7.5)

# ── Panel 4 (col 2-3, ambas filas): TABLA ───
ax_tbl = fig.add_subplot(gs[:, 2:])
ax_tbl.axis('off')
ax_tbl.set_title('Tabla de Iteraciones — Newton-Raphson 3×3',
                 color=GOLD, fontsize=10, pad=6)

col_labels = ['k', 'v (m/s)', 'θ (°)', 't (s)',
              'f₁', 'f₂', 'f₃', '‖F‖', 'det(J)']

def make_rows(history):
    rows = []
    show = history if len(history) <= 12 else history[:6] + [None] + history[-4:]
    for h in show:
        if h is None:
            rows.append(['⋮'] * len(col_labels))
            continue
        rows.append([
            f"{h['k']}",
            f"{h['v']:+.6f}",
            f"{h['th']:+.6f}",
            f"{h['t']:+.6f}",
            f"{h['f1']:+.3e}",
            f"{h['f2']:+.3e}",
            f"{h['f3']:+.3e}",
            f"{h['err']:.3e}",
            f"{h['det']:+.4f}",
        ])
    return rows

rows = make_rows(hist)
tbl = ax_tbl.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0.01, 0.08, 0.98, 0.88]
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
            err_val = float(rows[idx][7])
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
    f"SOLUCIÓN FINAL  |  Iteraciones: {len(hist)}\n"
    f"──────────────────────────────────────────────────\n"
    f"  v  = {v_sol:+.8f}  m/s\n"
    f"  θ  = {th_sol:+.8f}  °\n"
    f"  t  = {t_sol:+.8f}  s\n"
    f"  x(t) = {x_chk:.6f} m   (debe ser 300)\n"
    f"  y(t) = {y_chk:.6f} m   (debe ser  61)\n"
    f"  ángulo impacto = {ang_imp:.6f}°  (debe ser −45°)"
)
fig.text(0.015, 0.01, sol_text,
         fontsize=7.5, color=TEAL, fontfamily='monospace',
         va='bottom', alpha=0.90,
         bbox=dict(facecolor='#050f08', edgecolor='#00aa77',
                   boxstyle='round,pad=0.5', lw=0.8))

plt.savefig(out('proyectil_tabla.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()
print("✔  Figura estática guardada.")

# ─────────────────────────────────────────────
# 5. ANIMACIÓN SUAVIZADA (GIF)
# ─────────────────────────────────────────────

def ease_in_out_cubic(t):
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4*t*t*t
    p = 2*t - 2
    return 0.5*p*p*p + 1

def interp_val(prog, v0, v1, t0=0.0, t1=1.0):
    tt = ease_in_out_cubic((prog - t0) / max(1e-9, t1 - t0))
    return v0 + (v1 - v0) * tt

N_FRAMES = 130
FPS      = 25

# Fases:
#   0.00–0.50  El proyectil viaja a lo largo de la trayectoria
#   0.40–0.75  v, θ, t convergen via Newton-Raphson (paneles laterales)
#   0.65–1.00  Tabla de verificación y ángulo de impacto

t_frames = np.linspace(0, 1, N_FRAMES)

# Interpolación de los parámetros NR a lo largo de la animación
v_path  = np.array([interp_val(tt, q0[0], v_sol,            0.0, 0.50) for tt in t_frames])
th_path = np.array([interp_val(tt, math.degrees(q0[1]), th_sol, 0.0, 0.50) for tt in t_frames])
t_path  = np.array([interp_val(tt, q0[2], t_sol,            0.0, 0.50) for tt in t_frames])

# Posición del proyectil en cada frame
def projectile_pos(v_now, th_now_deg, t_now, frac):
    """Posición a lo largo de la trayectoria según fracción 0-1."""
    th_r  = math.radians(th_now_deg)
    t_fly = frac * t_now
    x = v_now * math.cos(th_r) * t_fly
    y = -0.5*g*t_fly**2 + v_now*math.sin(th_r)*t_fly
    return x, y

def traj_array(v_now, th_now_deg, t_now, frac):
    th_r  = math.radians(th_now_deg)
    t_end = frac * t_now
    ts = np.linspace(0, t_end, 300)
    xs = v_now * math.cos(th_r) * ts
    ys = -0.5*g*ts**2 + v_now*math.sin(th_r)*ts
    return xs, ys

apply_dark_style()
fig_ani = plt.figure(figsize=(15, 7), dpi=100)
fig_ani.patch.set_facecolor(BG)

gs2 = GridSpec(2, 3, figure=fig_ani,
               left=0.05, right=0.97,
               top=0.88, bottom=0.08,
               hspace=0.55, wspace=0.42)

fig_ani.text(0.5, 0.955,
             'PROYECTIL  ·  NEWTON-RAPHSON 3×3  ·  ANIMACIÓN',
             ha='center', fontsize=11, color=CYAN,
             fontweight='bold', fontfamily='monospace')
fig_ani.text(0.5, 0.933,
             'f₁: x=300 m     f₂: y=61 m     f₃: ángulo impacto 45°',
             ha='center', fontsize=8, color=WHITE, alpha=0.5)

ax_main  = fig_ani.add_subplot(gs2[:, 0])
ax_err2  = fig_ani.add_subplot(gs2[0, 1])
ax_evo2  = fig_ani.add_subplot(gs2[1, 1])
ax_f123  = fig_ani.add_subplot(gs2[0, 2])
ax_verif = fig_ani.add_subplot(gs2[1, 2])

# Pre-calcular residuos animados
err_anim = []
f1_anim  = []
f2_anim  = []
f3_anim  = []
for v_n, th_n, t_n in zip(v_path, th_path, t_path):
    Fv = F([v_n, math.radians(th_n), t_n])
    err_anim.append(np.linalg.norm(Fv))
    f1_anim.append(Fv[0])
    f2_anim.append(Fv[1])
    f3_anim.append(Fv[2])

def update(frame):
    prog = frame / (N_FRAMES - 1)

    v_now  = v_path[frame]
    th_now = th_path[frame]
    t_now  = t_path[frame]

    # Fracción de vuelo del proyectil (0→1 mientras prog va 0→0.55)
    frac_fly = ease_in_out_cubic(min(1.0, prog / 0.55))
    frac_nr  = ease_in_out_cubic(min(1.0, max(0.0, (prog - 0.25) / 0.50)))
    frac_vrf = ease_in_out_cubic(min(1.0, max(0.0, (prog - 0.65) / 0.30)))

    for ax in [ax_main, ax_err2, ax_evo2, ax_f123, ax_verif]:
        ax.cla()

    # ─────────────────────────────────────
    # Panel principal: proyectil en vuelo
    # ─────────────────────────────────────
    ax_main.set_facecolor(PANEL)
    ax_main.set_xlim(-15, 340)
    ax_main.set_ylim(-8, y_max * 1.22)
    ax_main.grid(True, alpha=0.25)
    ax_main.axhline(0, color=DIM, lw=0.8, alpha=0.5)
    ax_main.axvline(300, color=WHITE, lw=0.8, ls='--', alpha=0.25)

    # Trayectoria completa (guía tenue)
    ax_main.plot(x_arr, y_arr, color=CYAN, lw=0.7, alpha=0.15)

    # Trail hasta la posición actual
    xs_now, ys_now = traj_array(v_now, th_now, t_now, frac_fly)
    if len(xs_now) > 1:
        # Trail con gradiente de opacidad
        n_seg = len(xs_now) - 1
        for i in range(max(0, n_seg-50), n_seg):
            a = ((i - max(0, n_seg-50)) / min(50, n_seg)) ** 1.4 * 0.85
            ax_main.plot([xs_now[i], xs_now[i+1]],
                         [ys_now[i], ys_now[i+1]],
                         color=CYAN, lw=2.0, alpha=a,
                         solid_capstyle='round')

    # Proyectil (cabeza)
    if len(xs_now) > 0:
        px, py = xs_now[-1], ys_now[-1]
        ax_main.scatter([px], [py], color=CYAN, s=90,
                        zorder=8, edgecolors=BG, lw=1.2)

    # Lanzamiento
    ax_main.scatter([0], [0], color=GREEN, s=90, zorder=7,
                    edgecolors=BG, lw=1.2)

    # Blanco
    ax_main.scatter([300], [61], color=GOLD, s=200, marker='*',
                    zorder=9, edgecolors=BG, lw=0.5, alpha=0.7)

    # Vector velocidad inicial (aparece al inicio)
    if frac_fly < 0.15:
        scale = 15 / max(1, v_now)
        vx0_n = v_now*math.cos(math.radians(th_now))
        vy0_n = v_now*math.sin(math.radians(th_now))
        ax_main.annotate('', xy=(vx0_n*scale, vy0_n*scale), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='-|>', color=GREEN,
                                         lw=1.8, mutation_scale=12))

    # Flecha de impacto (aparece en fase final)
    if frac_fly > 0.85 and frac_vrf > 0.1:
        ax_main.annotate('', xy=(300+18, 61-18), xytext=(300, 61),
                         arrowprops=dict(arrowstyle='-|>', color=ORANGE,
                                         lw=2.0, mutation_scale=12),
                         alpha=min(1.0, frac_vrf))
        ax_main.text(305, 54, '45°', color=ORANGE, fontsize=8,
                     alpha=min(1.0, frac_vrf))

    # Barra de progreso
    ax_main.plot([-10, 320], [y_max*1.17]*2,
                 color=DIM, lw=3, solid_capstyle='round', zorder=6)
    ax_main.plot([-10, -10 + 330*prog], [y_max*1.17]*2,
                 color=CYAN, lw=3, solid_capstyle='round', zorder=7, alpha=0.8)

    ax_main.set_title(
        f'v={v_now:.2f} m/s   θ={th_now:.2f}°   t={t_now:.3f} s\n'
        f'frame {frame+1}/{N_FRAMES}',
        color=WHITE, fontsize=7.8, pad=4
    )
    ax_main.tick_params(labelsize=6.5)
    ax_main.set_xlabel('x (m)', fontsize=7.5)
    ax_main.set_ylabel('y (m)', fontsize=7.5)

    # ─────────────────────────────────────
    # Convergencia ‖F‖
    # ─────────────────────────────────────
    ax_err2.set_facecolor(PANEL)
    ax_err2.semilogy(t_frames, err_anim, color=GOLD, lw=0.8, alpha=0.2)
    ax_err2.semilogy(t_frames[:frame+1], err_anim[:frame+1],
                     color=GOLD, lw=1.8)
    ax_err2.scatter(t_frames[frame], err_anim[frame],
                    color=GOLD, s=40, zorder=5)
    ax_err2.fill_between(t_frames[:frame+1], err_anim[:frame+1],
                         alpha=0.08, color=GOLD)
    ax_err2.set_title('‖F(v,θ,t)‖', color=GOLD, fontsize=8.5, pad=3)
    ax_err2.set_xlabel('Progreso', fontsize=6.5)
    ax_err2.tick_params(labelsize=6)
    ax_err2.grid(True, alpha=0.25, which='both')

    # ─────────────────────────────────────
    # Evolución v, θ, t
    # ─────────────────────────────────────
    ax_evo2.set_facecolor(PANEL)
    v_n  = (v_path  - q0[0])  / max(1e-6, v_sol  - q0[0])
    th_n = (th_path - math.degrees(q0[1])) / max(1e-6, th_sol - math.degrees(q0[1]))
    t_n  = (t_path  - q0[2])  / max(1e-6, t_sol  - q0[2])

    ax_evo2.plot(t_frames, v_n,  color=CYAN,   lw=0.8, alpha=0.2)
    ax_evo2.plot(t_frames, th_n, color=ORANGE, lw=0.8, alpha=0.2)
    ax_evo2.plot(t_frames, t_n,  color=PURPLE, lw=0.8, alpha=0.2)
    ax_evo2.plot(t_frames[:frame+1], v_n[:frame+1],  color=CYAN,   lw=1.8, label='v')
    ax_evo2.plot(t_frames[:frame+1], th_n[:frame+1], color=ORANGE, lw=1.8, label='θ')
    ax_evo2.plot(t_frames[:frame+1], t_n[:frame+1],  color=PURPLE, lw=1.8, label='t')
    ax_evo2.axvline(prog, color=WHITE, lw=0.5, alpha=0.3)
    ax_evo2.axhline(1.0, color=WHITE, lw=0.6, ls='--', alpha=0.2)
    ax_evo2.set_title('Convergencia v, θ, t', color=WHITE, fontsize=8.5, pad=3)
    ax_evo2.set_xlabel('Progreso', fontsize=6.5)
    ax_evo2.set_ylabel('Valor normalizado', fontsize=6.5)
    ax_evo2.tick_params(labelsize=6)
    ax_evo2.legend(fontsize=6.5, framealpha=0.15,
                   facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE)
    ax_evo2.grid(True, alpha=0.25)
    ax_evo2.set_ylim(-0.15, 1.25)

    # ─────────────────────────────────────
    # Residuos f₁, f₂, f₃
    # ─────────────────────────────────────
    ax_f123.set_facecolor(PANEL)
    ax_f123.plot(t_frames, f1_anim, color=CYAN,   lw=0.8, alpha=0.2)
    ax_f123.plot(t_frames, f2_anim, color=ORANGE, lw=0.8, alpha=0.2)
    ax_f123.plot(t_frames, f3_anim, color=TEAL,   lw=0.8, alpha=0.2)
    ax_f123.plot(t_frames[:frame+1], f1_anim[:frame+1], color=CYAN,   lw=1.8, label='f₁')
    ax_f123.plot(t_frames[:frame+1], f2_anim[:frame+1], color=ORANGE, lw=1.8, label='f₂')
    ax_f123.plot(t_frames[:frame+1], f3_anim[:frame+1], color=TEAL,   lw=1.8, label='f₃')
    ax_f123.axhline(0, color=WHITE, lw=0.6, ls='--', alpha=0.35)
    ax_f123.axvline(prog, color=WHITE, lw=0.5, alpha=0.3)
    ax_f123.set_title('Residuos f₁, f₂, f₃', color=WHITE, fontsize=8.5, pad=3)
    ax_f123.set_xlabel('Progreso', fontsize=6.5)
    ax_f123.tick_params(labelsize=6)
    ax_f123.legend(fontsize=6.5, framealpha=0.15,
                   facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE)
    ax_f123.grid(True, alpha=0.25)

    # ─────────────────────────────────────
    # Verificación: x(t), y(t), ángulo
    # ─────────────────────────────────────
    ax_verif.set_facecolor(PANEL)

    x_now = interp_val(prog, 0.0,    x_chk,  0.55, 0.85)
    y_now = interp_val(prog, 0.0,    y_chk,  0.55, 0.85)
    a_now = interp_val(prog, 0.0,    ang_imp,0.55, 0.85)

    targets = [300.0, 61.0, -45.0]
    vals    = [x_now, y_now, a_now]
    labels  = ['x(t) [m]', 'y(t) [m]', 'ang [°]']
    cols    = [CYAN, ORANGE, PURPLE]

    x_pos = np.arange(3)
    ax_verif.bar(x_pos, targets, width=0.4, color=GOLD, alpha=0.3,
                 edgecolor=DIM, lw=0.7, label='Objetivo')
    ax_verif.bar(x_pos, vals,    width=0.4, color=cols, alpha=0.80,
                 edgecolor=DIM, lw=0.7, label='Calculado')
    ax_verif.set_xticks(x_pos)
    ax_verif.set_xticklabels(labels, fontsize=7)

    if frac_vrf > 0.5:
        for i, (v_, t_) in enumerate(zip(vals, targets)):
            ax_verif.text(i, max(abs(v_), abs(t_))*1.02 + 2,
                          f'{v_:.2f}', ha='center', va='bottom',
                          color=WHITE, fontsize=7,
                          alpha=min(1.0, (frac_vrf-0.5)/0.5))

    ax_verif.set_title('Verificación final', color=WHITE, fontsize=8.5, pad=3)
    ax_verif.tick_params(labelsize=6)
    ax_verif.legend(fontsize=6.5, framealpha=0.15,
                    facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE)
    ax_verif.grid(True, axis='y', alpha=0.25)

    return []

ani = animation.FuncAnimation(
    fig_ani, update, frames=N_FRAMES,
    interval=1000//FPS, blit=False
)

print("  Renderizando animación…")
ani.save(out('proyectil_animacion.gif'),
         writer=PillowWriter(fps=FPS),
         savefig_kwargs={'facecolor': BG})
plt.close(fig_ani)
print("✔  Animación guardada.")
print(f"\n  Archivos generados en: {OUT_DIR}")
print("    → proyectil_tabla.png")
print("    → proyectil_animacion.gif")