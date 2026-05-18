"""
Problema 27 — Trayectoria de satélite en órbita  (v2 — tabla + animación)
==========================================================================
        C
  R = ─────────────────
      1 + e·sin(θ + α)

Sistema lineal:  M · q = b   →   q = [C, A, B]ᵀ
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
# 1. DATOS Y RESOLUCIÓN MATRICIAL
# ─────────────────────────────────────────────

theta_deg = np.array([-30.0, 0.0, 30.0])
R_obs     = np.array([6870.0, 6728.0, 6615.0])
theta_rad = np.radians(theta_deg)

M = np.column_stack([
    1.0 / R_obs,
    -np.sin(theta_rad),
    -np.cos(theta_rad)
])
b_vec = np.ones(3)

q           = np.linalg.solve(M, b_vec)
C_val, A_val, B_val = q
e           = math.sqrt(A_val**2 + B_val**2)
alpha       = math.degrees(math.atan2(B_val, A_val))
alpha_rad   = math.radians(alpha)

theta_min_deg = 90.0 - alpha
theta_min_rad = math.radians(theta_min_deg)
R_min = C_val / (1 + e * math.sin(theta_min_rad + alpha_rad))

det_M   = np.linalg.det(M)
kappa   = np.linalg.cond(M)
residuo = np.linalg.norm(M @ q - b_vec)

def R_orbit(th_rad):
    return C_val / (1 + e * math.sin(th_rad + alpha_rad))

print(f"C = {C_val:.6f} km")
print(f"e = {e:.8f}   α = {alpha:.6f}°")
print(f"R_min = {R_min:.4f} km  @ θ = {theta_min_deg:.4f}°")
print(f"Residuo ‖Mq−b‖ = {residuo:.2e}   κ(M) = {kappa:.4f}")

# ─────────────────────────────────────────────
# 2. FIGURA ESTÁTICA — Órbita + Tabla + Análisis
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
         'PROBLEMA 27  ·  TRAYECTORIA ORBITAL DEL SATÉLITE  ·  SISTEMA MATRICIAL  M · q = b',
         ha='center', fontsize=13, color=CYAN,
         fontweight='bold', fontfamily='monospace')
fig.text(0.5, 0.940,
         'θ=−30° → R=6870 km     θ=0° → R=6728 km     θ=30° → R=6615 km',
         ha='center', fontsize=9, color=WHITE, alpha=0.55)

# ── Panel 1: Órbita polar ────────────────────
ax_pol = fig.add_subplot(gs[:, 0], projection='polar')
ax_pol.set_facecolor(PANEL)

theta_full = np.linspace(0, 2*np.pi, 1000)
R_full     = np.array([R_orbit(t) for t in theta_full])
valid      = R_full > 0
ax_pol.plot(theta_full[valid], R_full[valid],
            color=CYAN, lw=2.5, label='Órbita', zorder=3)

# Tierra
ax_pol.scatter([0], [0], color=GREEN, s=120, zorder=7,
               edgecolors=BG, lw=1.5, label='Tierra')

# Puntos observados
pt_colors = [ORANGE, TEAL, PURPLE]
for th, Ro, col in zip(theta_rad, R_obs, pt_colors):
    ax_pol.scatter([th], [Ro], color=col, s=90, zorder=6,
                   edgecolors=BG, lw=1.2)
    ax_pol.annotate(f'R={Ro:.0f}', (th, Ro),
                    textcoords='offset points', xytext=(6, 3),
                    color=col, fontsize=7.5)

# Perigeo
ax_pol.scatter([theta_min_rad], [R_min], color=GOLD, s=200,
               marker='*', zorder=8, edgecolors=BG, lw=0.5,
               label=f'Perigeo\nR={R_min:.1f} km\nθ={theta_min_deg:.2f}°')

ax_pol.set_title('Órbita del Satélite\n(coordenadas polares)',
                 color=WHITE, fontsize=10, pad=18)
ax_pol.tick_params(colors='gray', labelsize=7)
ax_pol.grid(color=DIM, lw=0.6, alpha=0.5)
ax_pol.legend(loc='lower left', facecolor=PANEL, edgecolor=DIM,
              labelcolor=WHITE, fontsize=7.5,
              bbox_to_anchor=(-0.05, -0.22))

# ── Panel 2: R(θ) cartesiano ─────────────────
ax_r = fig.add_subplot(gs[0, 1])
ax_r.set_facecolor(PANEL)

theta_plot = np.linspace(-180, 180, 800)
R_plot = np.array([R_orbit(math.radians(t)) for t in theta_plot])
ax_r.plot(theta_plot, R_plot, color=CYAN, lw=2.2, label='R(θ)')
for th, Ro, col in zip(theta_deg, R_obs, pt_colors):
    ax_r.scatter([th], [Ro], color=col, s=70, zorder=5)
    ax_r.annotate(f'({th:.0f}°, {Ro})', (th, Ro),
                  textcoords='offset points', xytext=(5, 5),
                  color=col, fontsize=7)
ax_r.scatter([theta_min_deg], [R_min], color=GOLD, s=160,
             marker='*', zorder=6,
             label=f'R_min={R_min:.1f} km')
ax_r.axvline(theta_min_deg, color=GOLD, lw=0.8, ls='--', alpha=0.5)
ax_r.axhline(R_min,         color=GOLD, lw=0.8, ls='--', alpha=0.5)
ax_r.set_title('R(θ) — Radio orbital vs ángulo', color=WHITE, fontsize=9, pad=5)
ax_r.set_xlabel('θ (grados)', fontsize=7.5)
ax_r.set_ylabel('R (km)', fontsize=7.5)
ax_r.tick_params(labelsize=7)
ax_r.legend(facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE, fontsize=8)
ax_r.grid(True, alpha=0.3)

# ── Panel 3: Visualización de la matriz M|b ──
ax_mat = fig.add_subplot(gs[1, 1])
ax_mat.set_facecolor(PANEL)
ax_mat.axis('off')
ax_mat.set_title('Sistema  M · q = b', color=WHITE, fontsize=9, pad=5)

Mb      = np.column_stack([M, b_vec])
col_hdr = ['1/Rᵢ', '−sinθᵢ', '−cosθᵢ', '| b']
row_hdr = ['θ=−30°', 'θ=0°', 'θ=30°']
col_w   = 1.18

for j, hdr in enumerate(col_hdr):
    clr = CYAN if j < 3 else ORANGE
    ax_mat.text(j*col_w + 0.55, 3.25, hdr,
                ha='center', va='center',
                color=clr, fontsize=9.5, fontweight='bold')

for i in range(3):
    ax_mat.text(-0.3, 2-i+0.42, row_hdr[i],
                ha='center', va='center', color=WHITE, fontsize=8, alpha=0.7)
    for j in range(4):
        fc = '#0f1d2e' if j < 3 else '#1e1208'
        ax_mat.add_patch(plt.Rectangle(
            (j*col_w, 2-i), col_w-0.05, 0.82,
            facecolor=fc, edgecolor=DIM, lw=0.8))
        ax_mat.text(j*col_w + 0.55, 2-i+0.41,
                    f'{Mb[i,j]:.6f}',
                    ha='center', va='center',
                    color=GOLD if j == 3 else WHITE,
                    fontsize=8.5, fontfamily='monospace')

ax_mat.axvline(3*col_w, color=ORANGE, lw=1.0, ls='--', alpha=0.5,
               ymin=0.08, ymax=0.88)

sol_str = (
    f"q = M⁻¹·b\n"
    f"  C = {C_val:.4f} km\n"
    f"  A = {A_val:.8f}\n"
    f"  B = {B_val:.8f}\n\n"
    f"  e = √(A²+B²) = {e:.6f}\n"
    f"  α = atan2(B,A) = {alpha:.4f}°\n"
    f"  R_min = {R_min:.4f} km\n"
    f"  θ_min = {theta_min_deg:.4f}°"
)
ax_mat.text(2.35, -0.3, sol_str, ha='center', va='top',
            color=WHITE, fontsize=8, fontfamily='monospace',
            bbox=dict(facecolor='#0d1a2a', edgecolor=CYAN,
                      lw=1.0, boxstyle='round,pad=0.5'))
ax_mat.set_xlim(-0.6, 4.9);  ax_mat.set_ylim(-4.0, 3.8)

# ── Panel 4 (columnas 2-3, ambas filas): TABLA ─
ax_tbl = fig.add_subplot(gs[:, 2:])
ax_tbl.axis('off')
ax_tbl.set_title('Tabla de Resultados — Verificación del Sistema Orbital',
                 color=GOLD, fontsize=10, pad=6)

col_labels = ['θᵢ (°)', 'R_obs (km)', 'Ecuación M·q',
              'b = 1', 'Residuo fila', 'R_calc (km)', '|R_calc−R_obs|']

rows = []
for th_d, th_r, Ro in zip(theta_deg, theta_rad, R_obs):
    eq_val  = (1/Ro)*C_val - math.sin(th_r)*A_val - math.cos(th_r)*B_val
    R_calc  = R_orbit(th_r)
    res_row = abs(eq_val - 1.0)
    err_R   = abs(R_calc - Ro)
    rows.append([
        f'{th_d:.1f}°',
        f'{Ro:.2f}',
        f'{eq_val:.8f}',
        f'1.00000000',
        f'{res_row:.2e}',
        f'{R_calc:.6f}',
        f'{err_R:.4e}',
    ])

rows.append(['—'] * 7)

# Propiedades matriciales
rows.append(['det(M)',    f'{det_M:.6e}', '—', '—', '—', '—', '—'])
rows.append(['κ(M)',      f'{kappa:.4f}', '—', '—', '—', '—', '—'])
rows.append(['‖Mq−b‖',   f'{residuo:.2e}','—','—', '—', '—', '—'])
rows.append(['—'] * 7)

# Parámetros orbitales
rows.append(['C (km)',    f'{C_val:.6f}',  '—', '—', '—', '—', '—'])
rows.append(['e',         f'{e:.8f}',       '—', '—', '—', '—', '—'])
rows.append(['α (°)',     f'{alpha:.6f}',   '—', '—', '—', '—', '—'])
rows.append(['θ_min (°)', f'{theta_min_deg:.6f}', '—', '—', '—', '—', '—'])
rows.append(['R_min (km)',f'{R_min:.6f}',  '—', '—', '—', '—', '—'])

tbl = ax_tbl.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0.02, 0.02, 0.96, 0.94]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(DIM)
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor('#0d1e38')
        cell.set_text_props(color=GOLD, fontweight='bold')
    else:
        idx = r - 1
        val0 = rows[idx][0]
        if val0 == '—':
            cell.set_facecolor('#0a0a14')
            cell.set_text_props(color=DIM)
        elif val0 in ('det(M)', 'κ(M)', '‖Mq−b‖'):
            cell.set_facecolor('#110e20')
            cell.set_text_props(color=PURPLE)
        elif val0 in ('C (km)', 'e', 'α (°)', 'θ_min (°)', 'R_min (km)'):
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
    f"SOLUCIÓN FINAL\n"
    f"──────────────────────────────────────\n"
    f"  C      = {C_val:.6f} km\n"
    f"  e      = {e:.8f}   (elíptica)\n"
    f"  α      = {alpha:.6f}°\n"
    f"  θ_min  = {theta_min_deg:.6f}°  (perigeo)\n"
    f"  R_min  = {R_min:.6f} km\n"
    f"  ‖Mq−b‖ = {residuo:.3e}   κ(M) = {kappa:.4f}"
)
fig.text(0.015, 0.01, sol_text,
         fontsize=7.5, color=TEAL, fontfamily='monospace',
         va='bottom', alpha=0.90,
         bbox=dict(facecolor='#050f08', edgecolor='#00aa77',
                   boxstyle='round,pad=0.5', lw=0.8))

plt.savefig(out('orbita_satelite_tabla.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close()
print("✔  Figura estática guardada.")

# ─────────────────────────────────────────────
# 3. ANIMACIÓN SUAVIZADA (GIF)
# ─────────────────────────────────────────────

def ease_in_out_cubic(t):
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4*t*t*t
    p = 2*t - 2
    return 0.5*p*p*p + 1

def interp_val(t, v0, v1, t0=0.0, t1=1.0):
    tt = ease_in_out_cubic((t - t0) / max(1e-9, t1 - t0))
    return v0 + (v1 - v0) * tt

N_FRAMES = 130
FPS      = 25

# La animación tiene 4 fases:
#   0.00–0.40  La órbita se traza gradualmente (C, e, α emergen)
#   0.35–0.65  Los 3 puntos observados aparecen
#   0.55–0.80  El perigeo aparece y se resalta
#   0.70–1.00  Los paneles de verificación se rellenan

theta_full_anim = np.linspace(0, 2*np.pi, 600)

# Pre-calcular R(θ) cartesiano para toda la animación
theta_cart = np.linspace(-180, 180, 800)
R_cart_full = np.array([R_orbit(math.radians(t)) for t in theta_cart])

t_frames = np.linspace(0, 1, N_FRAMES)

# Trayectorias de los parámetros orbitales
C_init  = np.mean(R_obs)   # valor inicial "neutro"
e_init  = 0.0
alpha_init = 0.0

apply_dark_style()
fig_ani = plt.figure(figsize=(15, 7), dpi=100)
fig_ani.patch.set_facecolor(BG)

gs2 = GridSpec(2, 3, figure=fig_ani,
               left=0.05, right=0.97,
               top=0.88, bottom=0.08,
               hspace=0.55, wspace=0.42)

fig_ani.text(0.5, 0.955,
             'TRAYECTORIA ORBITAL DEL SATÉLITE  ·  ANIMACIÓN MATRICIAL',
             ha='center', fontsize=11, color=CYAN,
             fontweight='bold', fontfamily='monospace')
fig_ani.text(0.5, 0.933,
             'θ=−30° → R=6870   θ=0° → R=6728   θ=30° → R=6615  (km)',
             ha='center', fontsize=8, color=WHITE, alpha=0.5)

ax_pol2  = fig_ani.add_subplot(gs2[:, 0], projection='polar')
ax_cart  = fig_ani.add_subplot(gs2[0, 1])
ax_param = fig_ani.add_subplot(gs2[1, 1])
ax_verif = fig_ani.add_subplot(gs2[0, 2])
ax_rmin  = fig_ani.add_subplot(gs2[1, 2])

def update(frame):
    prog = frame / (N_FRAMES - 1)

    # Parámetros animados
    C_now     = interp_val(prog, C_init, C_val,    0.00, 0.40)
    e_now     = interp_val(prog, e_init, e,         0.00, 0.40)
    alpha_now = interp_val(prog, alpha_init, alpha, 0.00, 0.40)
    alpha_now_rad = math.radians(alpha_now)

    def R_now(th_r):
        denom = 1 + e_now * math.sin(th_r + alpha_now_rad)
        return C_now / denom if abs(denom) > 1e-9 else np.nan

    # Fracción del arco dibujado
    frac_arc = ease_in_out_cubic(min(1.0, prog / 0.50))
    n_arc    = max(2, int(frac_arc * len(theta_full_anim)))

    # Fracciones de aparición de cada punto observado
    frac_pts = [ease_in_out_cubic(max(0.0, min(1.0, (prog - 0.30 - i*0.08) / 0.15)))
                for i in range(3)]
    frac_perigeo = ease_in_out_cubic(max(0.0, min(1.0, (prog - 0.60) / 0.18)))
    frac_verif   = ease_in_out_cubic(max(0.0, min(1.0, (prog - 0.72) / 0.25)))

    # ─────────────────────────────────────
    # Panel polar: órbita en construcción
    # ─────────────────────────────────────
    ax_pol2.cla()
    ax_pol2.set_facecolor(PANEL)

    # Arco parcial
    th_arc = theta_full_anim[:n_arc]
    R_arc  = np.array([R_now(t) for t in th_arc])
    valid  = np.isfinite(R_arc) & (R_arc > 0)
    if valid.sum() > 1:
        ax_pol2.plot(th_arc[valid], R_arc[valid],
                     color=CYAN, lw=2.3, alpha=0.9, zorder=3)

    # Tierra
    ax_pol2.scatter([0], [0], color=GREEN, s=110, zorder=7,
                    edgecolors=BG, lw=1.3)

    # Puntos observados con fade-in
    for i, (th_r, Ro, col, fp) in enumerate(zip(theta_rad, R_obs, pt_colors, frac_pts)):
        if fp > 0.02:
            ax_pol2.scatter([th_r], [Ro], color=col, s=80,
                            alpha=min(1.0, fp), zorder=6,
                            edgecolors=BG, lw=1.2)
            if fp > 0.6:
                ax_pol2.annotate(f'R={Ro:.0f}', (th_r, Ro),
                                 textcoords='offset points', xytext=(5, 3),
                                 color=col, fontsize=7,
                                 alpha=min(1.0, (fp-0.6)/0.4))

    # Perigeo con fade-in
    if frac_perigeo > 0.02:
        R_min_now = C_now / (1 + e_now)  # aproximación para animación
        ax_pol2.scatter([theta_min_rad], [R_min_now], color=GOLD,
                        s=180, marker='*', zorder=8,
                        edgecolors=BG, lw=0.5,
                        alpha=min(1.0, frac_perigeo))

    # Barra de progreso radial (en el borde exterior)
    R_max_est = C_now / max(1e-6, 1 - e_now) if e_now < 1 else C_now
    bar_R = R_max_est * 1.08
    th_bar = np.linspace(0, 2*np.pi * prog, 120)
    ax_pol2.plot(th_bar, [bar_R]*len(th_bar),
                 color=CYAN, lw=2.5, alpha=0.6, zorder=9)

    ax_pol2.set_title(
        f'Órbita — frame {frame+1}/{N_FRAMES}\n'
        f'C={C_now:.1f}  e={e_now:.5f}  α={alpha_now:.3f}°',
        color=WHITE, fontsize=7.5, pad=12
    )
    ax_pol2.tick_params(colors='gray', labelsize=6.5)
    ax_pol2.grid(color=DIM, lw=0.5, alpha=0.4)

    # ─────────────────────────────────────
    # R(θ) cartesiano (se dibuja progresivamente)
    # ─────────────────────────────────────
    ax_cart.cla()
    ax_cart.set_facecolor(PANEL)

    n_cart = max(2, int(frac_arc * len(theta_cart)))
    R_cart_now = np.array([C_now / (1 + e_now * math.sin(
        math.radians(t) + alpha_now_rad)) for t in theta_cart[:n_cart]])
    valid_c = np.isfinite(R_cart_now) & (R_cart_now > 0) & (R_cart_now < 1.5e5)
    if valid_c.sum() > 1:
        ax_cart.plot(theta_cart[:n_cart][valid_c], R_cart_now[valid_c],
                     color=CYAN, lw=1.8, alpha=0.9)

    # Curva final (guía tenue)
    ax_cart.plot(theta_cart, R_cart_full, color=CYAN, lw=0.6, alpha=0.15)

    for i, (th_d, Ro, col, fp) in enumerate(zip(theta_deg, R_obs, pt_colors, frac_pts)):
        if fp > 0.05:
            ax_cart.scatter([th_d], [Ro], color=col, s=60,
                            alpha=min(1.0, fp), zorder=5)

    if frac_perigeo > 0.05:
        ax_cart.scatter([theta_min_deg], [R_min], color=GOLD,
                        s=130, marker='*', zorder=6,
                        alpha=min(1.0, frac_perigeo))
        ax_cart.axvline(theta_min_deg, color=GOLD, lw=0.7,
                        ls='--', alpha=0.4 * frac_perigeo)
        ax_cart.axhline(R_min, color=GOLD, lw=0.7,
                        ls='--', alpha=0.4 * frac_perigeo)

    ax_cart.set_title('R(θ) vs ángulo', color=WHITE, fontsize=8.5, pad=3)
    ax_cart.set_xlabel('θ (°)', fontsize=6.5)
    ax_cart.set_ylabel('R (km)', fontsize=6.5)
    ax_cart.tick_params(labelsize=6)
    ax_cart.grid(True, alpha=0.25)

    # ─────────────────────────────────────
    # Evolución de parámetros C, e, α
    # ─────────────────────────────────────
    ax_param.cla()
    ax_param.set_facecolor(PANEL)

    t_arr = t_frames
    C_arr = np.array([interp_val(t, C_init, C_val, 0.0, 0.40) for t in t_arr])
    e_arr = np.array([interp_val(t, e_init, e,      0.0, 0.40) for t in t_arr])

    # Normalizar para graficar juntos
    C_norm = (C_arr - C_init) / max(1e-6, C_val - C_init)
    e_norm = e_arr / max(1e-6, e)

    ax_param.plot(t_arr, C_norm, color=CYAN,   lw=0.8, alpha=0.2)
    ax_param.plot(t_arr, e_norm, color=ORANGE, lw=0.8, alpha=0.2)
    ax_param.plot(t_arr[:frame+1], C_norm[:frame+1], color=CYAN,
                  lw=1.8, label='C (norm.)')
    ax_param.plot(t_arr[:frame+1], e_norm[:frame+1], color=ORANGE,
                  lw=1.8, label='e (norm.)')
    ax_param.axvline(prog, color=WHITE, lw=0.5, alpha=0.3)
    ax_param.set_title('Convergencia C y e', color=WHITE, fontsize=8.5, pad=3)
    ax_param.set_xlabel('Progreso', fontsize=6.5)
    ax_param.set_ylabel('Valor normalizado', fontsize=6.5)
    ax_param.tick_params(labelsize=6)
    ax_param.legend(fontsize=6.5, framealpha=0.15,
                    facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE)
    ax_param.grid(True, alpha=0.25)
    ax_param.set_ylim(-0.05, 1.15)

    # ─────────────────────────────────────
    # Verificación: R_calc vs R_obs por punto
    # ─────────────────────────────────────
    ax_verif.cla()
    ax_verif.set_facecolor(PANEL)

    R_calc_list = [interp_val(prog, Ro, R_orbit(th_r), 0.35, 0.75)
                   for Ro, th_r in zip(R_obs, theta_rad)]

    x_pos = np.arange(3)
    w = 0.32
    ax_verif.bar(x_pos - w/2, R_obs, width=w, color=ORANGE,
                 alpha=0.75, edgecolor=DIM, lw=0.7, label='R_obs')
    ax_verif.bar(x_pos + w/2, R_calc_list, width=w, color=CYAN,
                 alpha=0.75, edgecolor=DIM, lw=0.7, label='R_calc')
    ax_verif.set_xticks(x_pos)
    ax_verif.set_xticklabels(['θ=−30°', 'θ=0°', 'θ=30°'], fontsize=7)
    ax_verif.set_title('R_obs vs R_calc', color=WHITE, fontsize=8.5, pad=3)
    ax_verif.set_ylabel('km', fontsize=6.5)
    ax_verif.tick_params(labelsize=6)
    ax_verif.legend(fontsize=6.5, framealpha=0.15,
                    facecolor=PANEL, edgecolor=DIM, labelcolor=WHITE)
    ax_verif.grid(True, axis='y', alpha=0.25)
    y_min = min(min(R_obs), min(R_calc_list)) * 0.998
    y_max = max(max(R_obs), max(R_calc_list)) * 1.002
    ax_verif.set_ylim(y_min, y_max)

    # ─────────────────────────────────────
    # R_min evolucionando + texto de perigeo
    # ─────────────────────────────────────
    ax_rmin.cla()
    ax_rmin.set_facecolor(PANEL)

    R_min_arr = np.array([
        interp_val(t, C_init, R_min, 0.0, 0.65)
        for t in t_arr
    ])
    ax_rmin.plot(t_arr, R_min_arr, color=GOLD, lw=0.8, alpha=0.2)
    ax_rmin.plot(t_arr[:frame+1], R_min_arr[:frame+1],
                 color=GOLD, lw=1.8)
    ax_rmin.scatter(t_arr[frame], R_min_arr[frame],
                    color=GOLD, s=45, zorder=5)
    ax_rmin.fill_between(t_arr[:frame+1], R_min_arr[:frame+1],
                         alpha=0.08, color=GOLD)
    ax_rmin.axhline(R_min, color=GOLD, lw=0.8, ls='--', alpha=0.45)

    ax_rmin.set_title('R_min (Perigeo)', color=GOLD, fontsize=8.5, pad=3)
    ax_rmin.set_xlabel('Progreso', fontsize=6.5)
    ax_rmin.set_ylabel('km', fontsize=6.5)
    ax_rmin.tick_params(labelsize=6)
    ax_rmin.grid(True, alpha=0.25)

    if frac_perigeo > 0.1:
        ax_rmin.text(0.5, 0.30,
                     f'R_min={R_min:.2f} km\nθ={theta_min_deg:.2f}°',
                     ha='center', va='center',
                     transform=ax_rmin.transAxes,
                     color=GOLD, fontsize=8, fontfamily='monospace',
                     alpha=min(1.0, frac_perigeo),
                     bbox=dict(facecolor='#0a0a08', edgecolor=GOLD,
                               boxstyle='round,pad=0.4', lw=0.7))

    return []

ani = animation.FuncAnimation(
    fig_ani, update, frames=N_FRAMES,
    interval=1000//FPS, blit=False
)

print("  Renderizando animación…")
ani.save(out('orbita_satelite_animacion.gif'),
         writer=PillowWriter(fps=FPS),
         savefig_kwargs={'facecolor': BG})
plt.close(fig_ani)
print("✔  Animación guardada.")
print(f"\n  Archivos generados en: {OUT_DIR}")
print("    → orbita_satelite_tabla.png")
print("    → orbita_satelite_animacion.gif")