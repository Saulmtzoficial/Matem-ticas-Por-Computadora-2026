"""
Problema 29 — Cuadrilátero articulado (four-bar linkage)
=========================================================
Eslabones: L1 = 150 mm, L2 = 180 mm, L3 = 200 mm, base = 200 mm
θ₃ = 75° (dado)

Sistema no lineal 2×2 en (θ₁, θ₂):
  f1 = 150·cosθ₁ + 180·cosθ₂ − 200·cosθ₃ − 200 = 0
  f2 = 150·sinθ₁ + 180·sinθ₂ − 200·sinθ₃       = 0

Jacobiano J(θ₁, θ₂):
  J = | −150·sinθ₁   −180·sinθ₂ |
      |  150·cosθ₁    180·cosθ₂ |

Existen DOS soluciones → se resuelve con dos puntos iniciales distintos.
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
# 1. PARÁMETROS Y SISTEMA
# ─────────────────────────────────────────────

L1, L2, L3 = 150.0, 180.0, 200.0
BASE = 200.0
TH3_DEG = 75.0
th3 = math.radians(TH3_DEG)

C3 = math.cos(th3)
S3 = math.sin(th3)

def F(v):
    t1, t2 = v
    f1 = L1*math.cos(t1) + L2*math.cos(t2) - L3*C3 - BASE
    f2 = L1*math.sin(t1) + L2*math.sin(t2) - L3*S3
    return np.array([f1, f2])

def J(v):
    t1, t2 = v
    return np.array([
        [-L1*math.sin(t1), -L2*math.sin(t2)],
        [ L1*math.cos(t1),  L2*math.cos(t2)]
    ])

# ─────────────────────────────────────────────
# 2. NEWTON-RAPHSON
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
            'k':   i,
            'th1': math.degrees(q[0]),
            'th2': math.degrees(q[1]),
            'f1':  Fval[0],
            'f2':  Fval[1],
            'err': err,
            'det': detJ,
        })
        if err < tol:
            break
        delta = np.linalg.solve(Jval, -Fval)
        q = q + delta
    return q, history

# ─────────────────────────────────────────────
# 3. DOS SOLUCIONES
# ─────────────────────────────────────────────

q0_A = [math.radians(80.0), math.radians(20.0)]   # Solución A
q0_B = [math.radians(30.0), math.radians(120.0)]  # Solución B

sol_A, hist_A = newton_raphson(q0_A)
sol_B, hist_B = newton_raphson(q0_B)

th1_A, th2_A = math.degrees(sol_A[0]), math.degrees(sol_A[1])
th1_B, th2_B = math.degrees(sol_B[0]), math.degrees(sol_B[1])

print(f"Solución A:  θ₁ = {th1_A:.8f}°   θ₂ = {th2_A:.8f}°")
print(f"Verificación: F(A) = {F(sol_A)}")
print(f"Solución B:  θ₁ = {th1_B:.8f}°   θ₂ = {th2_B:.8f}°")
print(f"Verificación: F(B) = {F(sol_B)}")

# ─────────────────────────────────────────────
# 4. POSICIONES DEL MECANISMO
# ─────────────────────────────────────────────

def linkage_coords(th1_r, th2_r):
    """Devuelve los 4 vértices del cuadrilátero: O, A, B, C."""
    O  = np.array([0.0, 0.0])
    A  = O + L1 * np.array([math.cos(th1_r), math.sin(th1_r)])
    B  = A + L2 * np.array([math.cos(th2_r), math.sin(th2_r)])
    C  = np.array([BASE + L3*C3, L3*S3])   # = (BASE + 200·cos75°, 200·sin75°)
    return O, A, B, C

O_A, A_A, B_A, C_A = linkage_coords(sol_A[0], sol_A[1])
O_B, A_B, B_B, C_B = linkage_coords(sol_B[0], sol_B[1])

# Punto de anclaje derecho (base + eslabón L3)
PIVOT_R = np.array([BASE + L3*C3, L3*S3])

print(f"\nCoordenadas Solución A:  O={O_A}  A={A_A}  B={B_A}")
print(f"Coordenadas Solución B:  O={O_B}  A={A_B}  B={B_B}")
print(f"Pivote derecho (C): {PIVOT_R}")

# ─────────────────────────────────────────────
# 5. FIGURA ESTÁTICA
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
         'PROBLEMA 29  ·  CUADRILÁTERO ARTICULADO  ·  NEWTON-RAPHSON 2×2',
         ha='center', fontsize=13, color=CYAN,
         fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.940,
         f'150·cosθ₁ + 180·cosθ₂ − 200·cosθ₃ = 200     '
         f'150·sinθ₁ + 180·sinθ₂ − 200·sinθ₃ = 0     θ₃ = 75°',
         ha='center', fontsize=9, color=WHITE, alpha=0.55)

# ─── Helper para dibujar el mecanismo ────────
def draw_linkage(ax, th1_r, th2_r, col1, col2, col3, alpha=1.0,
                 show_labels=True, suffix=''):
    O, A, B, C = linkage_coords(th1_r, th2_r)
    base_end   = np.array([BASE, 0.0])

    # Base (suelo)
    ax.plot([O[0], BASE], [O[1], 0], color=DIM, lw=2.5,
            solid_capstyle='round', alpha=alpha*0.6)

    # Eslabón L1 (θ₁)
    ax.plot([O[0], A[0]], [O[1], A[1]], color=col1, lw=3.0,
            solid_capstyle='round', alpha=alpha, zorder=4)

    # Eslabón L2 (θ₂)
    ax.plot([A[0], B[0]], [A[1], B[1]], color=col2, lw=3.0,
            solid_capstyle='round', alpha=alpha, zorder=4)

    # Eslabón L3 (θ₃, fijo)
    ax.plot([BASE, PIVOT_R[0]], [0, PIVOT_R[1]], color=col3, lw=3.0,
            solid_capstyle='round', alpha=alpha, zorder=4)

    # Articulaciones
    ax.scatter(*O,        color='white', s=70, zorder=7,
               edgecolors=BG, lw=1.3, alpha=alpha)
    ax.scatter(*A,        color=col1,   s=60, zorder=7,
               edgecolors=BG, lw=1.3, alpha=alpha)
    ax.scatter(*B,        color=col2,   s=60, zorder=7,
               edgecolors=BG, lw=1.3, alpha=alpha)
    ax.scatter(*PIVOT_R,  color=col3,   s=70, zorder=7,
               edgecolors=BG, lw=1.3, alpha=alpha)

    if show_labels and alpha > 0.5:
        ax.text(-8, -18, 'O', color='white',  fontsize=8.5, alpha=0.8)
        ax.text(A[0]+5, A[1]+5, f'A{suffix}', color=col1, fontsize=8.5)
        ax.text(B[0]+5, B[1]+5, f'B{suffix}', color=col2, fontsize=8.5)
        ax.text(PIVOT_R[0]+5, PIVOT_R[1]+3, 'C', color=col3, fontsize=8.5)

# ── Panel 1: Solución A ──────────────────────
ax_A = fig.add_subplot(gs[0, 0])
ax_A.set_facecolor(PANEL)
ax_A.set_aspect('equal')

draw_linkage(ax_A, sol_A[0], sol_A[1],
             CYAN, TEAL, ORANGE, suffix='_A')

# Arco θ₁
arc_t = np.linspace(0, sol_A[0], 50)
ax_A.plot(30*np.cos(arc_t), 30*np.sin(arc_t), color=CYAN, lw=1.2, alpha=0.7)
ax_A.text(35, 10, f'θ₁={th1_A:.2f}°', color=CYAN, fontsize=8)

arc_t2 = np.linspace(0, sol_A[1], 50)
ax_A.plot(A_A[0]+25*np.cos(arc_t2), A_A[1]+25*np.sin(arc_t2),
          color=TEAL, lw=1.2, alpha=0.7)
ax_A.text(A_A[0]-60, A_A[1]+15, f'θ₂={th2_A:.2f}°', color=TEAL, fontsize=8)

ax_A.set_xlim(-30, 420); ax_A.set_ylim(-35, 250)
ax_A.set_title(f'Solución A  (θ₁={th1_A:.3f}°, θ₂={th2_A:.3f}°)',
               color=CYAN, fontsize=9, pad=5)
ax_A.set_xlabel('x (mm)', fontsize=7.5); ax_A.set_ylabel('y (mm)', fontsize=7.5)
ax_A.tick_params(labelsize=7)
ax_A.grid(True, alpha=0.3)
ax_A.axhline(0, color=DIM, lw=0.5, alpha=0.5)
ax_A.axvline(0, color=DIM, lw=0.5, alpha=0.5)

# Marcas de apoyo (suelo)
for xg in np.linspace(-10, BASE+10, 12):
    ax_A.plot([xg, xg-6], [0, -7], color=DIM, lw=0.8, alpha=0.5)

# ── Panel 2: Solución B ──────────────────────
ax_B = fig.add_subplot(gs[1, 0])
ax_B.set_facecolor(PANEL)
ax_B.set_aspect('equal')

draw_linkage(ax_B, sol_B[0], sol_B[1],
             PURPLE, ORANGE, TEAL, suffix='_B')

arc_t = np.linspace(0, sol_B[0], 50)
ax_B.plot(30*np.cos(arc_t), 30*np.sin(arc_t), color=PURPLE, lw=1.2, alpha=0.7)
ax_B.text(35, 10, f'θ₁={th1_B:.2f}°', color=PURPLE, fontsize=8)

arc_t2 = np.linspace(0, sol_B[1], 50)
ax_B.plot(A_B[0]+25*np.cos(arc_t2), A_B[1]+25*np.sin(arc_t2),
          color=ORANGE, lw=1.2, alpha=0.7)
ax_B.text(A_B[0]-90, A_B[1]+12, f'θ₂={th2_B:.2f}°', color=ORANGE, fontsize=8)

ax_B.set_xlim(-30, 420); ax_B.set_ylim(-35, 250)
ax_B.set_title(f'Solución B  (θ₁={th1_B:.3f}°, θ₂={th2_B:.3f}°)',
               color=PURPLE, fontsize=9, pad=5)
ax_B.set_xlabel('x (mm)', fontsize=7.5); ax_B.set_ylabel('y (mm)', fontsize=7.5)
ax_B.tick_params(labelsize=7)
ax_B.grid(True, alpha=0.3)
ax_B.axhline(0, color=DIM, lw=0.5, alpha=0.5)
ax_B.axvline(0, color=DIM, lw=0.5, alpha=0.5)

for xg in np.linspace(-10, BASE+10, 12):
    ax_B.plot([xg, xg-6], [0, -7], color=DIM, lw=0.8, alpha=0.5)

# ── Panel 3: Convergencia ‖F‖ ───────────────
ax_err = fig.add_subplot(gs[0, 1])
ax_err.set_facecolor(PANEL)

for hist, col, lbl in [(hist_A, CYAN, 'Sol. A'), (hist_B, PURPLE, 'Sol. B')]:
    ks   = [h['k']   for h in hist]
    errs = [h['err'] for h in hist]
    ax_err.semilogy(ks, errs, 'o-', color=col, lw=2, ms=6,
                    markerfacecolor=BG, markeredgecolor=col, mew=1.2,
                    label=lbl)
    ax_err.fill_between(ks, errs, alpha=0.07, color=col)
    ax_err.annotate(f'ε={errs[-1]:.1e}', (ks[-1], errs[-1]),
                    textcoords='offset points', xytext=(5, 3),
                    color=col, fontsize=7, alpha=0.9)

ax_err.set_title('Convergencia  ‖F(θ₁,θ₂)‖', color=WHITE, fontsize=9, pad=5)
ax_err.set_xlabel('Iteración k', fontsize=7.5)
ax_err.set_ylabel('‖F‖  (log)', fontsize=7.5)
ax_err.tick_params(labelsize=7)
ax_err.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE, fontsize=8)
ax_err.grid(True, which='both', alpha=0.3)

# ── Panel 4: Trayectoria θ₁, θ₂ ────────────
ax_ev = fig.add_subplot(gs[1, 1])
ax_ev.set_facecolor(PANEL)

for hist, col1, col2, sfx in [
        (hist_A, CYAN,   TEAL,   'A'),
        (hist_B, PURPLE, ORANGE, 'B')]:
    ks   = [h['k']   for h in hist]
    th1s = [h['th1'] for h in hist]
    th2s = [h['th2'] for h in hist]
    ax_ev.plot(ks, th1s, 'o-', color=col1, lw=1.8, ms=5,
               markerfacecolor=BG, label=f'θ₁ ({sfx})')
    ax_ev.plot(ks, th2s, 's--', color=col2, lw=1.8, ms=5,
               markerfacecolor=BG, label=f'θ₂ ({sfx})')

ax_ev.set_title('Evolución de θ₁ y θ₂', color=WHITE, fontsize=9, pad=5)
ax_ev.set_xlabel('Iteración k', fontsize=7.5)
ax_ev.set_ylabel('Ángulo (°)', fontsize=7.5)
ax_ev.tick_params(labelsize=7)
ax_ev.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE,
             fontsize=7.5, ncol=2)
ax_ev.grid(True, alpha=0.3)

# ── Panel 5 (col 2-3, ambas filas): TABLA ───
ax_tbl = fig.add_subplot(gs[:, 2:])
ax_tbl.axis('off')
ax_tbl.set_title('Tabla de Iteraciones — Newton-Raphson (ambas soluciones)',
                 color=GOLD, fontsize=10, pad=6)

col_labels = ['k', 'θ₁ (°)', 'θ₂ (°)', 'f₁', 'f₂', '‖F‖', 'det(J)']

def make_rows(hist, label):
    rows = []
    rows.append([f'── {label} ──'] + [''] * (len(col_labels)-1))
    show = hist if len(hist) <= 10 else hist[:5] + [None] + hist[-3:]
    for h in show:
        if h is None:
            rows.append(['⋮'] * len(col_labels))
            continue
        rows.append([
            f"{h['k']}",
            f"{h['th1']:+.6f}",
            f"{h['th2']:+.6f}",
            f"{h['f1']:+.4e}",
            f"{h['f2']:+.4e}",
            f"{h['err']:.4e}",
            f"{h['det']:+.4f}",
        ])
    return rows

rows = make_rows(hist_A, 'Solución A') + [['—']*len(col_labels)] + \
       make_rows(hist_B, 'Solución B')

tbl = ax_tbl.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0.01, 0.04, 0.98, 0.92]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.2)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(DIM)
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor('#0d1e38')
        cell.set_text_props(color=GOLD, fontweight='bold')
    else:
        idx = r - 1
        val0 = rows[idx][0]
        if '─' in str(val0) or 'Sol' in str(val0):
            bg = '#0d2030' if 'A' in str(val0) else '#1a0d30'
            fc = CYAN    if 'A' in str(val0) else PURPLE
            cell.set_facecolor(bg)
            cell.set_text_props(color=fc, fontweight='bold')
        elif val0 == '—' or val0 == '⋮':
            cell.set_facecolor('#0a0a14')
            cell.set_text_props(color=DIM)
        else:
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

# ── Cuadro de soluciones finales ──────────────
sol_text = (
    f"SOLUCIONES FINALES  (θ₃ = {TH3_DEG}°)\n"
    f"─────────────────────────────────────────────────\n"
    f"  Sol. A:  θ₁ = {th1_A:+.8f}°\n"
    f"           θ₂ = {th2_A:+.8f}°\n"
    f"  ‖F(A)‖ = {np.linalg.norm(F(sol_A)):.3e}   iters: {len(hist_A)}\n\n"
    f"  Sol. B:  θ₁ = {th1_B:+.8f}°\n"
    f"           θ₂ = {th2_B:+.8f}°\n"
    f"  ‖F(B)‖ = {np.linalg.norm(F(sol_B)):.3e}   iters: {len(hist_B)}"
)
fig.text(0.015, 0.01, sol_text,
         fontsize=7.5, color=TEAL, fontfamily='monospace',
         va='bottom', alpha=0.90,
         bbox=dict(facecolor='#050f08', edgecolor='#00aa77',
                   boxstyle='round,pad=0.5', lw=0.8))

plt.savefig(out('cuadrilatero_tabla.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()
print("✔  Figura estática guardada.")

# ─────────────────────────────────────────────
# 6. ANIMACIÓN SUAVIZADA (GIF)
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
t_frames = np.linspace(0, 1, N_FRAMES)

# Trayectorias de θ₁, θ₂ para cada solución
th1A_path = np.array([interp_val(tt, math.radians(80), sol_A[0], 0.0, 0.50)
                       for tt in t_frames])
th2A_path = np.array([interp_val(tt, math.radians(20), sol_A[1], 0.0, 0.50)
                       for tt in t_frames])
th1B_path = np.array([interp_val(tt, math.radians(30), sol_B[0], 0.0, 0.50)
                       for tt in t_frames])
th2B_path = np.array([interp_val(tt, math.radians(120), sol_B[1], 0.0, 0.50)
                       for tt in t_frames])

# Errores animados
errA_anim = [np.linalg.norm(F([t1, t2]))
             for t1, t2 in zip(th1A_path, th2A_path)]
errB_anim = [np.linalg.norm(F([t1, t2]))
             for t1, t2 in zip(th1B_path, th2B_path)]

apply_dark_style()
fig_ani = plt.figure(figsize=(16, 7), dpi=100)
fig_ani.patch.set_facecolor(BG)

gs2 = GridSpec(2, 3, figure=fig_ani,
               left=0.05, right=0.97,
               top=0.88, bottom=0.08,
               hspace=0.55, wspace=0.42)

fig_ani.text(0.5, 0.955,
             'CUADRILÁTERO ARTICULADO  ·  DOS SOLUCIONES  ·  NEWTON-RAPHSON',
             ha='center', fontsize=11, color=CYAN,
             fontweight='bold', fontfamily='monospace')
fig_ani.text(0.5, 0.933,
             f'L₁=150 mm   L₂=180 mm   L₃=200 mm   θ₃={TH3_DEG}°   base=200 mm',
             ha='center', fontsize=8, color=WHITE, alpha=0.5)

ax_mec  = fig_ani.add_subplot(gs2[:, 0])
ax_errA = fig_ani.add_subplot(gs2[0, 1])
ax_errB = fig_ani.add_subplot(gs2[1, 1])
ax_thA  = fig_ani.add_subplot(gs2[0, 2])
ax_thB  = fig_ani.add_subplot(gs2[1, 2])

def draw_linkage_anim(ax, th1_r, th2_r, col1, col2, col3,
                       alpha=1.0, suffix=''):
    O, A, B, C = linkage_coords(th1_r, th2_r)
    ax.plot([O[0], BASE], [0, 0], color=DIM, lw=2.5, alpha=0.5,
            solid_capstyle='round')
    ax.plot([O[0], A[0]], [O[1], A[1]], color=col1, lw=3.5,
            solid_capstyle='round', alpha=alpha, zorder=4)
    ax.plot([A[0], B[0]], [A[1], B[1]], color=col2, lw=3.5,
            solid_capstyle='round', alpha=alpha, zorder=4)
    ax.plot([BASE, PIVOT_R[0]], [0, PIVOT_R[1]], color=col3, lw=3.5,
            solid_capstyle='round', alpha=alpha, zorder=4)
    # Marcas de suelo
    for xg in np.linspace(-10, BASE+10, 10):
        ax.plot([xg, xg-5], [0, -6], color=DIM, lw=0.7, alpha=0.4)
    # Articulaciones
    ax.scatter(*O,       color='white', s=65, zorder=7, edgecolors=BG, lw=1.2, alpha=alpha)
    ax.scatter(*A,       color=col1,   s=55, zorder=7, edgecolors=BG, lw=1.2, alpha=alpha)
    ax.scatter(*B,       color=col2,   s=55, zorder=7, edgecolors=BG, lw=1.2, alpha=alpha)
    ax.scatter(*PIVOT_R, color=col3,   s=65, zorder=7, edgecolors=BG, lw=1.2, alpha=alpha)

def update(frame):
    prog = frame / (N_FRAMES - 1)
    frac_label = ease_in_out_cubic(min(1.0, max(0.0, (prog-0.45)/0.20)))

    for ax in [ax_mec, ax_errA, ax_errB, ax_thA, ax_thB]:
        ax.cla()

    # ─────────────────────────────────────
    # Panel principal: ambas configuraciones
    # ─────────────────────────────────────
    ax_mec.set_facecolor(PANEL)
    ax_mec.set_aspect('equal')
    ax_mec.set_xlim(-30, 430)
    ax_mec.set_ylim(-40, 260)
    ax_mec.grid(True, alpha=0.22)
    ax_mec.axhline(0, color=DIM, lw=0.5, alpha=0.4)
    ax_mec.axvline(0, color=DIM, lw=0.5, alpha=0.4)

    th1a = th1A_path[frame]
    th2a = th2A_path[frame]
    th1b = th1B_path[frame]
    th2b = th2B_path[frame]

    # Solución B (detrás, más tenue)
    draw_linkage_anim(ax_mec, th1b, th2b,
                      PURPLE, ORANGE, TEAL, alpha=0.50, suffix='B')

    # Solución A (delante)
    draw_linkage_anim(ax_mec, th1a, th2a,
                      CYAN, TEAL, ORANGE, alpha=0.92, suffix='A')

    # Etiquetas finales
    if frac_label > 0.1:
        _, Aa, _, _ = linkage_coords(th1a, th2a)
        _, Ab, _, _ = linkage_coords(th1b, th2b)
        ax_mec.text(Aa[0]+5, Aa[1]+6,
                    f'A  θ₁={math.degrees(th1a):.1f}°',
                    color=CYAN, fontsize=7.5,
                    alpha=min(1.0, frac_label))
        ax_mec.text(Ab[0]+5, Ab[1]+6,
                    f'B  θ₁={math.degrees(th1b):.1f}°',
                    color=PURPLE, fontsize=7.5,
                    alpha=min(1.0, frac_label))

    # Barra de progreso
    ax_mec.plot([-20, 410], [248]*2,
                color=DIM, lw=3, solid_capstyle='round', zorder=9)
    ax_mec.plot([-20, -20+430*prog], [248]*2,
                color=CYAN, lw=3, solid_capstyle='round', zorder=10, alpha=0.8)

    ax_mec.set_title(
        f'Configuraciones A y B  —  frame {frame+1}/{N_FRAMES}\n'
        f'θ₁A={math.degrees(th1a):.2f}°  θ₂A={math.degrees(th2a):.2f}°   '
        f'θ₁B={math.degrees(th1b):.2f}°  θ₂B={math.degrees(th2b):.2f}°',
        color=WHITE, fontsize=7.5, pad=4
    )
    ax_mec.tick_params(labelsize=6.5)
    ax_mec.set_xlabel('x (mm)', fontsize=7.5)
    ax_mec.set_ylabel('y (mm)', fontsize=7.5)

    # ─────────────────────────────────────
    # Convergencia ‖F‖ — Solución A
    # ─────────────────────────────────────
    ax_errA.set_facecolor(PANEL)
    ax_errA.semilogy(t_frames, errA_anim, color=CYAN, lw=0.8, alpha=0.2)
    ax_errA.semilogy(t_frames[:frame+1], errA_anim[:frame+1],
                     color=CYAN, lw=1.8)
    ax_errA.scatter(t_frames[frame], errA_anim[frame],
                    color=CYAN, s=40, zorder=5)
    ax_errA.fill_between(t_frames[:frame+1], errA_anim[:frame+1],
                         alpha=0.08, color=CYAN)
    ax_errA.set_title('‖F‖  —  Sol. A', color=CYAN, fontsize=8.5, pad=3)
    ax_errA.set_xlabel('Progreso', fontsize=6.5)
    ax_errA.tick_params(labelsize=6)
    ax_errA.grid(True, alpha=0.25, which='both')

    # ─────────────────────────────────────
    # Convergencia ‖F‖ — Solución B
    # ─────────────────────────────────────
    ax_errB.set_facecolor(PANEL)
    ax_errB.semilogy(t_frames, errB_anim, color=PURPLE, lw=0.8, alpha=0.2)
    ax_errB.semilogy(t_frames[:frame+1], errB_anim[:frame+1],
                     color=PURPLE, lw=1.8)
    ax_errB.scatter(t_frames[frame], errB_anim[frame],
                    color=PURPLE, s=40, zorder=5)
    ax_errB.fill_between(t_frames[:frame+1], errB_anim[:frame+1],
                         alpha=0.08, color=PURPLE)
    ax_errB.set_title('‖F‖  —  Sol. B', color=PURPLE, fontsize=8.5, pad=3)
    ax_errB.set_xlabel('Progreso', fontsize=6.5)
    ax_errB.tick_params(labelsize=6)
    ax_errB.grid(True, alpha=0.25, which='both')

    # ─────────────────────────────────────
    # Evolución θ₁, θ₂ — Solución A
    # ─────────────────────────────────────
    ax_thA.set_facecolor(PANEL)
    th1A_deg = np.degrees(th1A_path)
    th2A_deg = np.degrees(th2A_path)
    ax_thA.plot(t_frames, th1A_deg, color=CYAN,   lw=0.8, alpha=0.2)
    ax_thA.plot(t_frames, th2A_deg, color=TEAL,   lw=0.8, alpha=0.2)
    ax_thA.plot(t_frames[:frame+1], th1A_deg[:frame+1],
                color=CYAN,   lw=1.8, label='θ₁')
    ax_thA.plot(t_frames[:frame+1], th2A_deg[:frame+1],
                color=TEAL,   lw=1.8, label='θ₂')
    ax_thA.axhline(th1_A, color=CYAN,   lw=0.7, ls='--', alpha=0.4)
    ax_thA.axhline(th2_A, color=TEAL,   lw=0.7, ls='--', alpha=0.4)
    ax_thA.axvline(prog, color=WHITE, lw=0.5, alpha=0.3)
    ax_thA.set_title('θ₁, θ₂  —  Sol. A (°)', color=CYAN, fontsize=8.5, pad=3)
    ax_thA.set_xlabel('Progreso', fontsize=6.5)
    ax_thA.set_ylabel('Grados (°)', fontsize=6.5)
    ax_thA.tick_params(labelsize=6)
    ax_thA.legend(fontsize=6.5, framealpha=0.15,
                  facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE)
    ax_thA.grid(True, alpha=0.25)

    # ─────────────────────────────────────
    # Evolución θ₁, θ₂ — Solución B
    # ─────────────────────────────────────
    ax_thB.set_facecolor(PANEL)
    th1B_deg = np.degrees(th1B_path)
    th2B_deg = np.degrees(th2B_path)
    ax_thB.plot(t_frames, th1B_deg, color=PURPLE, lw=0.8, alpha=0.2)
    ax_thB.plot(t_frames, th2B_deg, color=ORANGE, lw=0.8, alpha=0.2)
    ax_thB.plot(t_frames[:frame+1], th1B_deg[:frame+1],
                color=PURPLE, lw=1.8, label='θ₁')
    ax_thB.plot(t_frames[:frame+1], th2B_deg[:frame+1],
                color=ORANGE, lw=1.8, label='θ₂')
    ax_thB.axhline(th1_B, color=PURPLE, lw=0.7, ls='--', alpha=0.4)
    ax_thB.axhline(th2_B, color=ORANGE, lw=0.7, ls='--', alpha=0.4)
    ax_thB.axvline(prog, color=WHITE, lw=0.5, alpha=0.3)
    ax_thB.set_title('θ₁, θ₂  —  Sol. B (°)', color=PURPLE, fontsize=8.5, pad=3)
    ax_thB.set_xlabel('Progreso', fontsize=6.5)
    ax_thB.set_ylabel('Grados (°)', fontsize=6.5)
    ax_thB.tick_params(labelsize=6)
    ax_thB.legend(fontsize=6.5, framealpha=0.15,
                  facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE)
    ax_thB.grid(True, alpha=0.25)

    return []

ani = animation.FuncAnimation(
    fig_ani, update, frames=N_FRAMES,
    interval=1000//FPS, blit=False
)

print("  Renderizando animación…")
ani.save(out('cuadrilatero_animacion.gif'),
         writer=PillowWriter(fps=FPS),
         savefig_kwargs={'facecolor': BG})
plt.close(fig_ani)
print("✔  Animación guardada.")
print(f"\n  Archivos generados en: {OUT_DIR}")
print("    → cuadrilatero_tabla.png")
print("    → cuadrilatero_animacion.gif")