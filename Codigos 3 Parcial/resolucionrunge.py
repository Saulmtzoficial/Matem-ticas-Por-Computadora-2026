import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


plt.rcParams.update({
    'font.family': 'serif',       # Letra clásica de paper
    'font.size': 12,              # Tamaño base legible
    'axes.labelsize': 14,         # Títulos de los ejes más grandes
    'axes.titlesize': 14,         # Títulos de las gráficas
    'legend.fontsize': 11,        # Letra de la leyenda
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300             
})


def deriv_circuito(y, t):
    i1, i3 = y
    di1_dt = -20*i1 + 10*i3 + 100
    di3_dt = 10*i1  - 20*i3
    return [di1_dt, di3_dt]

# ====================================================================
# PARTE 2: Generar datos de Simulación Suave
# ====================================================================
t_sim = np.linspace(0, 5, 500) 
y0 = [0.0, 0.0]
sol_sim = odeint(deriv_circuito, y0, t_sim)
i1_sim = sol_sim[:, 0]
i3_sim = sol_sim[:, 1]
i2_sim = i1_sim - i3_sim

# ====================================================================
# PARTE 3: El Método RK4 (h=0.01)
# ====================================================================
h = 0.01          
t_max = 5.0      

t_rk = np.arange(0, t_max + h, h)
n_steps = len(t_rk)
i1_rk = np.zeros(n_steps)
i3_rk = np.zeros(n_steps)
i1_rk[0] = 0.0
i3_rk[0] = 0.0

for k in range(n_steps - 1):
    y_k = [i1_rk[k], i3_rk[k]]
    k1 = h * np.array(deriv_circuito(y_k, t_rk[k]))
    k2 = h * np.array(deriv_circuito(y_k + 0.5*k1, t_rk[k] + 0.5*h))
    k3 = h * np.array(deriv_circuito(y_k + 0.5*k2, t_rk[k] + 0.5*h))
    k4 = h * np.array(deriv_circuito(y_k + k3, t_rk[k] + h))
    
    y_next = y_k + (k1 + 2*k2 + 2*k3 + k4) / 6.0
    i1_rk[k+1] = y_next[0]
    i3_rk[k+1] = y_next[1]

i2_rk = i1_rk - i3_rk

# ====================================================================
# PARTE 4: Graficar solo el estado transitorio
# ====================================================================
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(t_sim, i1_sim, color='#1f77b4', linewidth=2, label='Sim. $i_1(t)$', alpha=0.7)
ax1.plot(t_sim, i3_sim, color='#d62728', linewidth=2, label='Sim. $i_3(t)$', alpha=0.7)
ax1.plot(t_sim, i2_sim, color='#2ca02c', linewidth=2, label='Sim. $i_2(t)$', alpha=0.7)

ax1.plot(t_rk[:6], i1_rk[:6], marker='o', linestyle='', color='#1f77b4', markersize=7, markeredgecolor='black', label='RK4 $i_1(t)$')
ax1.plot(t_rk[:6], i3_rk[:6], marker='s', linestyle='', color='#d62728', markersize=7, markeredgecolor='black', label='RK4 $i_3(t)$')
ax1.plot(t_rk[:6], i2_rk[:6], marker='^', linestyle='', color='#2ca02c', markersize=7, markeredgecolor='black', label='RK4 $i_2(t)$')

ax1.set_title('Detalle Transitorio ($t \in [0, 0.5]$ s)', pad=10)
ax1.set_xlabel('Tiempo $t$ (s)')
ax1.set_ylabel('Corriente $i$ (A)')
ax1.set_xlim(0, 0.5)
ax1.set_ylim(-1, 7)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right', framealpha=1.0, edgecolor='black')

plt.tight_layout()

# Guardar imagen en alta calidad
plt.savefig('transitorio_rk4_paper.png', dpi=300, bbox_inches='tight')

plt.show()