"""
Problema 26 — Círculo por 3 puntos  (v2 — tabla + animación)
=============================================================
Sistema lineal:  A · q = b   →   q = A⁻¹ · b
Puntos: (8.21, 0.00), (0.34, 6.62), (5.96, −1.12)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
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
# 1. DATOS Y RESOLUCIÓN MATRICIAL
# ─────────────────────────────────────────────

points = np.array([
    [8.21,  0.00],
    [0.34,  6.62],
    [5.96, -1.12]
])

px = points[:, 0]
py = points[:, 1]

A   = np.column_stack([px, py, np.ones(3)])
rhs = -(px**2 + py**2)

q       = np.linalg.solve(A, rhs)
u, v, w = q
a_c     = -u / 2          # centro x
b_c     = -v / 2          # centro y
R       = np.sqrt(a_c**2 + b_c**2 - w)

det_A   = np.linalg.det(A)
kappa   = np.linalg.cond(A)
residuo = np.linalg.norm(A @ q - rhs)

dists   = [math.sqrt((xi - a_c)**2 + (yi - b_c)**2) for xi, yi in points]
errs    = [abs(d - R) for d in dists]

print(f"Centro: ({a_c:.8f}, {b_c:.8f})")
print(f"Radio : {R:.8f}")
print(f"Residuo ‖Aq−b‖ = {residuo:.2e}   κ(A) = {kappa:.4f}   det(A) = {det_A:.6f}")

# ─────────────────────────────────────────────
# 2. FIGURA ESTÁTICA — Círculo + Tabla + Análisis
# ─────────────────────────────────────────────

apply_dark_style()

fig = plt.figure(figsize=(22, 10), dpi=130)
fig.patch.set_facecolor(BG)

gs = GridSpec(2, 4, figure=fig,
              left=0.04, right=0.98,
              top=0.90, bottom=0.07,
              hspace=0.52, wspace=0.38)

# ── Título ──────────────────────────────────
fig.text(0.5, 0.962,
         'PROBLEMA 26  ·  CÍRCULO POR 3 PUNTOS  ·  SISTEMA MATRICIAL  A · q = b',
         ha='center', fontsize=13, color=CYAN,
         fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.940,
         f'P₁(8.21, 0.00)   P₂(0.34, 6.62)   P₃(5.96, −1.12)',
         ha='center', fontsize=9, color=WHITE, alpha=0.55)

# ── Panel 1: Círculo + puntos ────────────────
ax_circ = fig.add_subplot(gs[:, 0])
ax_circ.set_facecolor(PANEL)
ax_circ.set_aspect('equal')

theta_c = np.linspace(0, 2*np.pi, 600)
cx = a_c + R*np.cos(theta_c)
cy = b_c + R*np.sin(theta_c)
ax_circ.plot(cx, cy, color=CYAN, lw=2.5, label=f'Círculo  R={R:.4f}', zorder=3)

# Radios a cada punto
pt_colors = [ORANGE, TEAL, PURPLE]
for i, ((xi, yi), col) in enumerate(zip(points, pt_colors)):
    ax_circ.plot([a_c, xi], [b_c, yi], '--', color=col, lw=1.3, alpha=0.6, zorder=2)

# Puntos y etiquetas
for i, ((xi, yi), col) in enumerate(zip(points, pt_colors)):
    ax_circ.scatter(xi, yi, color=col, s=100, zorder=6,
                    edgecolors=BG, lw=1.3)
    ax_circ.annotate(f' P{i+1}({xi:.2f}, {yi:.2f})',
                     (xi, yi), color=col, fontsize=8.5,
                     textcoords='offset points', xytext=(7, 4))

# Centro
ax_circ.scatter(a_c, b_c, color=GOLD, s=130, marker='+',
                linewidths=2.5, zorder=7)
ax_circ.scatter(a_c, b_c, color=GOLD, s=60, zorder=7,
                edgecolors=BG, lw=1.2)
ax_circ.annotate(f' C({a_c:.3f}, {b_c:.3f})',
                 (a_c, b_c), color=GOLD, fontsize=8.5,
                 textcoords='offset points', xytext=(8, -14))

ax_circ.axhline(0, color=DIM, lw=0.5, alpha=0.5)
ax_circ.axvline(0, color=DIM, lw=0.5, alpha=0.5)
pad = R * 0.22
ax_circ.set_xlim(a_c - R - pad, a_c + R + pad)
ax_circ.set_ylim(b_c - R - pad, b_c + R + pad)
ax_circ.set_title('Círculo por 3 puntos', color=WHITE, fontsize=10, pad=7)
ax_circ.set_xlabel('x', fontsize=8);  ax_circ.set_ylabel('y', fontsize=8)
ax_circ.tick_params(labelsize=7)
ax_circ.grid(True, alpha=0.3)
ax_circ.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE,
               fontsize=8, loc='lower right')

# ── Panel 2: Distancias vs R ─────────────────
ax_dist = fig.add_subplot(gs[0, 1])
ax_dist.set_facecolor(PANEL)

labels_p = [f'P{i+1}\n({xi:.2f},{yi:.2f})' for i, (xi, yi) in enumerate(points)]
bars = ax_dist.bar(labels_p, dists, color=pt_colors, alpha=0.80,
                   edgecolor=DIM, linewidth=0.8)
ax_dist.axhline(R, color=GOLD, lw=2, ls='--', label=f'R = {R:.6f}')
for bar, d, e in zip(bars, dists, errs):
    ax_dist.text(bar.get_x() + bar.get_width()/2, d + 0.04,
                 f'd={d:.5f}\nerr={e:.1e}',
                 ha='center', va='bottom', color=WHITE, fontsize=7.5)
ax_dist.set_title('Distancia  Pᵢ → Centro  vs  R', color=WHITE, fontsize=9, pad=5)
ax_dist.set_ylabel('Distancia', fontsize=7.5)
ax_dist.tick_params(labelsize=7)
ax_dist.set_ylim(0, max(dists)*1.22)
ax_dist.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE, fontsize=8)
ax_dist.grid(True, axis='y', alpha=0.3)

# ── Panel 3: Visualización de la matriz A|b ──
ax_mat = fig.add_subplot(gs[1, 1])
ax_mat.set_facecolor(PANEL)
ax_mat.axis('off')
ax_mat.set_title('Sistema matricial  A · q = b', color=WHITE, fontsize=9, pad=5)

Ab = np.column_stack([A, rhs])
col_hdrs = ['x', 'y', '1', '| b']
row_hdrs = ['P₁', 'P₂', 'P₃']

for j, hdr in enumerate(col_hdrs):
    clr = CYAN if j < 3 else ORANGE
    ax_mat.text(j + 0.5, 3.3, hdr, ha='center', va='center',
                color=clr, fontsize=10.5, fontweight='bold')

for i in range(3):
    ax_mat.text(-0.25, 2 - i + 0.5, row_hdrs[i],
                ha='center', va='center', color=WHITE, fontsize=9, alpha=0.7)
    for j in range(4):
        fc = '#0f1825' if j < 3 else '#1a0f0a'
        ax_mat.add_patch(plt.Rectangle((j, 2-i), 1, 1,
                         facecolor=fc, edgecolor=DIM, lw=0.8))
        ax_mat.text(j+0.5, 2-i+0.5, f'{Ab[i,j]:.2f}',
                    ha='center', va='center',
                    color=GOLD if j==3 else WHITE,
                    fontsize=9.5, fontfamily='monospace')

ax_mat.axvline(3, color=ORANGE, lw=1.0, ls='--', alpha=0.5,
               ymin=0.05, ymax=0.95)

sol_str = (
    f"Solución  q = A⁻¹·b\n"
    f"  u = {u:+.6f}\n"
    f"  v = {v:+.6f}\n"
    f"  w = {w:+.6f}\n\n"
    f"  a = {a_c:+.6f}  (centro x)\n"
    f"  b = {b_c:+.6f}  (centro y)\n"
    f"  R = {R:.6f}"
)
ax_mat.text(2.0, -0.4, sol_str, ha='center', va='top',
            color=WHITE, fontsize=8, fontfamily='monospace',
            bbox=dict(facecolor='#0d1a2a', edgecolor=CYAN, lw=1.0,
                      boxstyle='round,pad=0.5'))
ax_mat.set_xlim(-0.5, 4.2);  ax_mat.set_ylim(-3.6, 3.9)

# ── Panel 4 (columnas 2-3, ambas filas): TABLA ─
ax_tbl = fig.add_subplot(gs[:, 2:])
ax_tbl.axis('off')
ax_tbl.set_title('Tabla de Resultados — Verificación del Sistema',
                 color=GOLD, fontsize=10, pad=6)

col_labels = ['Punto', 'xᵢ', 'yᵢ',
              'Eq: xᵢu+yᵢv+w', 'RHS −(xᵢ²+yᵢ²)',
              'Residuo', 'dist(Pᵢ,C)', '|dist−R|']

rows = []
for i, (xi, yi) in enumerate(points):
    lhs_val = xi*u + yi*v + w
    rhs_val = -(xi**2 + yi**2)
    res_i   = abs(lhs_val - rhs_val)
    rows.append([
        f'P{i+1}',
        f'{xi:.2f}',
        f'{yi:.2f}',
        f'{lhs_val:+.8f}',
        f'{rhs_val:+.8f}',
        f'{res_i:.2e}',
        f'{dists[i]:.8f}',
        f'{errs[i]:.2e}',
    ])

# Fila de propiedades matriciales
rows.append(['—']*8)
rows.append(['det(A)',  f'{det_A:.6f}', '—', '—', '—', '—', '—', '—'])
rows.append(['κ(A)',    f'{kappa:.4f}', '—', '—', '—', '—', '—', '—'])
rows.append(['‖Aq−b‖', f'{residuo:.2e}','—','—', '—', '—', '—', '—'])

tbl = ax_tbl.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0.02, 0.02, 0.96, 0.92]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.8)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(DIM)
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor('#0d1e38')
        cell.set_text_props(color=GOLD, fontweight='bold')
    else:
        idx = r - 1
        if rows[idx][0] == '—':
            cell.set_facecolor('#0a0a14')
            cell.set_text_props(color=DIM)
        elif rows[idx][0] in ('det(A)', 'κ(A)', '‖Aq−b‖'):
            cell.set_facecolor('#110e20')
            cell.set_text_props(color=PURPLE)
        elif idx % 2 == 0:
            cell.set_facecolor('#0b0f1e')
            cell.set_text_props(color=TEAL)
        else:
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=WHITE)

# ── Cuadro de solución final ──────────────────
sol_text = (
    f"SOLUCIÓN FINAL\n"
    f"──────────────────────────────────────\n"
    f"  Centro:  a = {a_c:+.8f}\n"
    f"           b = {b_c:+.8f}\n"
    f"  Radio:   R = {R:.8f}\n"
    f"  ‖Aq−b‖  = {residuo:.3e}\n"
    f"  κ(A)    = {kappa:.4f}   det(A) = {det_A:.6f}"
)
fig.text(0.015, 0.01, sol_text,
         fontsize=7.5, color=TEAL, fontfamily='monospace',
         va='bottom', alpha=0.90,
         bbox=dict(facecolor='#050f08', edgecolor='#00aa77',
                   boxstyle='round,pad=0.5', lw=0.8))

plt.savefig(out('circulo_3puntos_tabla.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()
print("✔  Figura estática guardada.")

# ─────────────────────────────────────────────
# 3. ANIMACIÓN SUAVIZADA (GIF)
# ─────────────────────────────────────────────

def ease_in_out_cubic(t):
    if t < 0.5:
        return 4*t*t*t
    p = 2*t - 2
    return 0.5*p*p*p + 1

N_FRAMES = 120
FPS      = 25

# La animación muestra 3 fases:
#   FASE 0 (0–33%):  el círculo se "dibuja" gradualmente
#   FASE 1 (33–66%): los radios aparecen uno a uno
#   FASE 2 (66–100%): panel de verificación se rellena

t_frames = np.linspace(0, 1, N_FRAMES)

theta_full = np.linspace(0, 2*np.pi, 600)
cx_full = a_c + R*np.cos(theta_full)
cy_full = b_c + R*np.sin(theta_full)

pad = R * 0.22
xlim = (a_c - R - pad, a_c + R + pad)
ylim = (b_c - R - pad, b_c + R + pad)

apply_dark_style()
fig_ani = plt.figure(figsize=(14, 7), dpi=100)
fig_ani.patch.set_facecolor(BG)

gs2 = GridSpec(2, 3, figure=fig_ani,
               left=0.05, right=0.97,
               top=0.88, bottom=0.08,
               hspace=0.55, wspace=0.40)

fig_ani.text(0.5, 0.955,
             'CÍRCULO POR 3 PUNTOS  ·  CONSTRUCCIÓN MATRICIAL  A · q = b',
             ha='center', fontsize=11, color=CYAN,
             fontweight='bold', fontfamily='monospace')
fig_ani.text(0.5, 0.933,
             f'P₁(8.21, 0.00)   P₂(0.34, 6.62)   P₃(5.96, −1.12)',
             ha='center', fontsize=8, color=WHITE, alpha=0.5)

ax_main  = fig_ani.add_subplot(gs2[:, 0])
ax_dist2 = fig_ani.add_subplot(gs2[0, 1])
ax_res   = fig_ani.add_subplot(gs2[1, 1])
ax_uvw   = fig_ani.add_subplot(gs2[0, 2])
ax_circ2 = fig_ani.add_subplot(gs2[1, 2])

# Interpolaciones para cada panel animado

# a_c, b_c, R "emergen" progresivamente durante la fase 0
# (simulamos como si viniéramos de un punto inicial centrado en los puntos)
cx_mean  = float(np.mean(px))
cy_mean  = float(np.mean(py))
R_mean   = float(np.mean([math.sqrt((xi-cx_mean)**2+(yi-cy_mean)**2) for xi, yi in points]))

def interp_val(t, v0, v1, t0=0.0, t1=1.0):
    tt = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
    e  = ease_in_out_cubic(tt)
    return v0 + (v1 - v0) * e

def update(frame):
    prog = frame / (N_FRAMES - 1)

    for ax in [ax_main, ax_dist2, ax_res, ax_uvw, ax_circ2]:
        ax.cla()

    # ─────────────────────────────────────
    # Panel principal: construcción del círculo
    # ─────────────────────────────────────
    ax_main.set_facecolor(PANEL)
    ax_main.set_xlim(*xlim);  ax_main.set_ylim(*ylim)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.25)
    ax_main.axhline(0, color=DIM, lw=0.4, alpha=0.5)
    ax_main.axvline(0, color=DIM, lw=0.4, alpha=0.5)

    # Centro animado
    a_now = interp_val(prog, cx_mean, a_c, 0.0, 0.45)
    b_now = interp_val(prog, cy_mean, b_c, 0.0, 0.45)
    R_now = interp_val(prog, R_mean,  R,   0.0, 0.45)

    # Círculo parcial: se dibuja hasta frac del arco
    frac_circ = ease_in_out_cubic(min(1.0, prog / 0.55))
    n_arc = max(2, int(frac_circ * len(theta_full)))
    cx_now = a_now + R_now * np.cos(theta_full[:n_arc])
    cy_now = b_now + R_now * np.sin(theta_full[:n_arc])
    ax_main.plot(cx_now, cy_now, color=CYAN, lw=2.4, alpha=0.9, zorder=3)

    # Centro
    ax_main.scatter(a_now, b_now, color=GOLD, s=110,
                    edgecolors=BG, lw=1.2, zorder=7)
    ax_main.scatter(a_now, b_now, color=GOLD, s=130,
                    marker='+', linewidths=2.2, zorder=8)

    # Radios: aparecen en fase 1
    for i, ((xi, yi), col) in enumerate(zip(points, pt_colors)):
        t_start = 0.40 + i * 0.10
        t_end   = t_start + 0.15
        frac_r  = ease_in_out_cubic(max(0.0, min(1.0, (prog - t_start) / (t_end - t_start))))
        if frac_r > 0:
            rx = a_now + frac_r * (xi - a_now)
            ry = b_now + frac_r * (yi - b_now)
            ax_main.plot([a_now, rx], [b_now, ry], '--', color=col,
                         lw=1.4, alpha=0.65 * frac_r, zorder=2)
        # Punto (aparece junto con su radio)
        if frac_r > 0.05:
            ax_main.scatter(xi, yi, color=col, s=90,
                            alpha=min(1.0, frac_r * 1.5),
                            zorder=6, edgecolors=BG, lw=1.2)
            if frac_r > 0.6:
                ax_main.annotate(f' P{i+1}({xi:.2f},{yi:.2f})',
                                 (xi, yi), color=col, fontsize=7.5,
                                 textcoords='offset points', xytext=(6, 3),
                                 alpha=min(1.0, (frac_r-0.6)/0.4))

    # Barra de progreso
    bw = xlim[1] - xlim[0]
    bx0 = xlim[0] + bw * 0.05
    ax_main.plot([bx0, bx0 + bw*0.9], [ylim[1]-0.18]*2,
                 color=DIM, lw=3, solid_capstyle='round', zorder=9)
    ax_main.plot([bx0, bx0 + bw*0.9*prog], [ylim[1]-0.18]*2,
                 color=CYAN, lw=3, solid_capstyle='round', zorder=10, alpha=0.8)

    phase = ('Resolviendo A·q=b…' if prog < 0.40
             else 'Trazando radios…' if prog < 0.75
             else 'Verificando…')
    ax_main.set_title(
        f'{phase}  frame {frame+1}/{N_FRAMES}\n'
        f'C=({a_now:.4f}, {b_now:.4f})   R={R_now:.4f}',
        color=WHITE, fontsize=7.8, pad=4
    )
    ax_main.tick_params(labelsize=6.5)
    ax_main.set_xlabel('x', fontsize=7.5);  ax_main.set_ylabel('y', fontsize=7.5)

    # ─────────────────────────────────────
    # Distancias vs R (barras creciendo)
    # ─────────────────────────────────────
    ax_dist2.set_facecolor(PANEL)
    frac_bars = ease_in_out_cubic(min(1.0, max(0.0, (prog - 0.35) / 0.40)))
    d_now     = [interp_val(prog, 0.0, d, 0.35, 0.75) for d in dists]
    R_bar     = interp_val(prog, 0.0, R, 0.35, 0.75)
    bars2 = ax_dist2.bar(labels_p, d_now, color=pt_colors, alpha=0.80,
                         edgecolor=DIM, lw=0.8)
    if R_bar > 0.01:
        ax_dist2.axhline(R_bar, color=GOLD, lw=1.8, ls='--',
                         alpha=0.85, label=f'R={R_bar:.4f}')
        ax_dist2.legend(fontsize=6.5, framealpha=0.15,
                        facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE)
    if frac_bars > 0.85:
        for bar, d, e in zip(bars2, d_now, errs):
            ax_dist2.text(bar.get_x()+bar.get_width()/2, d+0.03,
                          f'{d:.4f}', ha='center', va='bottom',
                          color=WHITE, fontsize=6.5,
                          alpha=min(1.0, (frac_bars-0.85)/0.15))
    ax_dist2.set_title('Distancias vs R', color=WHITE, fontsize=8.5, pad=3)
    ax_dist2.set_ylim(0, max(dists)*1.2)
    ax_dist2.tick_params(labelsize=6)
    ax_dist2.grid(True, axis='y', alpha=0.25)

    # ─────────────────────────────────────
    # Residuos por punto
    # ─────────────────────────────────────
    ax_res.set_facecolor(PANEL)
    frac_res = ease_in_out_cubic(min(1.0, max(0.0, (prog - 0.60) / 0.30)))
    err_now  = [e * frac_res for e in errs]
    ax_res.bar(labels_p, err_now, color=pt_colors, alpha=0.80,
               edgecolor=DIM, lw=0.8)
    ax_res.set_title('Error  |dist − R|', color=WHITE, fontsize=8.5, pad=3)
    ax_res.set_ylabel('Error', fontsize=6.5)
    ax_res.tick_params(labelsize=6)
    ax_res.grid(True, axis='y', alpha=0.25)
    if frac_res > 0.0:
        for j, (lp, e) in enumerate(zip(labels_p, err_now)):
            ax_res.text(j, e + max(errs)*0.05,
                        f'{e:.2e}', ha='center', va='bottom',
                        color=WHITE, fontsize=6.5, alpha=frac_res)

    # ─────────────────────────────────────
    # Evolución de u, v, w  (incógnitas)
    # ─────────────────────────────────────
    ax_uvw.set_facecolor(PANEL)
    t_uvw = np.linspace(0, 1, N_FRAMES)
    u_path = np.array([interp_val(tt, cx_mean*(-2), u, 0.0, 0.45) for tt in t_uvw])
    v_path = np.array([interp_val(tt, cy_mean*(-2), v, 0.0, 0.45) for tt in t_uvw])
    w_path = np.array([interp_val(tt, 0.0, w, 0.0, 0.45) for tt in t_uvw])

    ax_uvw.plot(t_uvw, u_path, color=CYAN,   lw=0.8, alpha=0.2)
    ax_uvw.plot(t_uvw, v_path, color=ORANGE, lw=0.8, alpha=0.2)
    ax_uvw.plot(t_uvw, w_path, color=PURPLE, lw=0.8, alpha=0.2)
    ax_uvw.plot(t_uvw[:frame+1], u_path[:frame+1], color=CYAN,   lw=1.8, label='u')
    ax_uvw.plot(t_uvw[:frame+1], v_path[:frame+1], color=ORANGE, lw=1.8, label='v')
    ax_uvw.plot(t_uvw[:frame+1], w_path[:frame+1], color=PURPLE, lw=1.8, label='w')
    ax_uvw.axvline(prog, color=WHITE, lw=0.5, alpha=0.3)
    ax_uvw.set_title('Incógnitas u, v, w', color=WHITE, fontsize=8.5, pad=3)
    ax_uvw.set_xlabel('Progreso', fontsize=6.5)
    ax_uvw.tick_params(labelsize=6)
    ax_uvw.legend(fontsize=6.5, framealpha=0.15,
                  facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE,
                  loc='center right')
    ax_uvw.grid(True, alpha=0.25)

    # ─────────────────────────────────────
    # Ángulo barrido del arco del círculo
    # ─────────────────────────────────────
    ax_circ2.set_facecolor(PANEL)
    angle_swept = frac_circ * 360.0
    ax_circ2.bar(['Arco dibujado (°)', 'Arco restante (°)'],
                 [angle_swept, 360.0 - angle_swept],
                 color=[CYAN, DIM], alpha=0.8, edgecolor=DIM, lw=0.8)
    ax_circ2.text(0, angle_swept + 5, f'{angle_swept:.1f}°',
                  ha='center', va='bottom', color=CYAN, fontsize=8)
    ax_circ2.set_ylim(0, 390)
    ax_circ2.set_title('Construcción del arco', color=WHITE, fontsize=8.5, pad=3)
    ax_circ2.set_ylabel('Grados (°)', fontsize=6.5)
    ax_circ2.tick_params(labelsize=6.5)
    ax_circ2.grid(True, axis='y', alpha=0.25)

    # Parámetros actuales en texto
    ax_circ2.text(0.5, 0.35,
                  f'a={a_now:+.4f}\nb={b_now:+.4f}\nR={R_now:.4f}',
                  ha='center', va='center',
                  transform=ax_circ2.transAxes,
                  color=GOLD, fontsize=8, fontfamily='monospace',
                  bbox=dict(facecolor='#050f08', edgecolor=GOLD,
                            boxstyle='round,pad=0.4', lw=0.7, alpha=0.85))

    return []

ani = animation.FuncAnimation(
    fig_ani, update, frames=N_FRAMES,
    interval=1000//FPS, blit=False
)

print("  Renderizando animación…")
ani.save(out('circulo_3puntos_animacion.gif'),
         writer=PillowWriter(fps=FPS),
         savefig_kwargs={'facecolor': BG})
plt.close(fig_ani)
print("✔  Animación guardada.")
print("\n  Archivos generados en:", OUT_DIR)
print("    → circulo_3puntos_tabla.png")
print("    → circulo_3puntos_animacion.gif")