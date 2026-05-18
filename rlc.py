import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# --- ACTIVAR MODO OSCURO ---
plt.style.use('dark_background')

# ==========================================
# 1. PARÁMETROS Y RESOLUCIÓN DE LA EDO
# ==========================================
# --- Configuración del circuito ---
R = 2.0      # Resistencia
L = 1.0      # Inductancia
C = 0.1      # Capacitancia
V_in = 10.0  # Voltaje de entrada escalón

# --- Ecuaciones Diferenciales ---
def rlc_circuit(t, state, R, L, C, V_in):
    i, vc = state
    di_dt = (V_in - (R * i) - vc) / L
    dvc_dt = i / C
    return [di_dt, dvc_dt]

# --- Configuración de tiempo y resolución ---
# Aumentamos un poco los puntos para que la animación sea muy fluida
t_span = (0, 15) 
t_eval = np.linspace(t_span[0], t_span[1], 1500) 
initial_state = [0.0, 0.0]

print("Calculando solución de la EDO...")
solution = solve_ivp(
    fun=lambda t, y: rlc_circuit(t, y, R, L, C, V_in),
    t_span=t_span, y0=initial_state, t_eval=t_eval, method='RK45'
)
print("Cálculo finalizado. Iniciando animación...")

# Extraer los datos completos
time_data = solution.t
current_data = solution.y[0]
vc_data = solution.y[1]
vl_data = V_in - (current_data * R) - vc_data

# ==========================================
# 2. PREPARACIÓN DEL "ESCENARIO" GRÁFICO
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle(f'Simulación Dinámica RLC Serie (R={R}$\Omega$, L={L}H, C={C}F)', color='white')

# --- Configuración Subplot 1: Voltajes ---
ax1.set_xlim(t_span[0], t_span[1])
# Calculamos límites verticales dinámicamente con un margen del 10%
max_v = max(max(vc_data), max(vl_data), V_in)
min_v = min(min(vc_data), min(vl_data), 0)
margin_v = (max_v - min_v) * 0.1
ax1.set_ylim(min_v - margin_v, max_v + margin_v)

ax1.set_ylabel('Voltaje (V)')
ax1.grid(True, which='both', linestyle='--', alpha=0.3)
# Dibujamos la referencia de entrada (fija)
ax1.axhline(y=V_in, color='red', linestyle=':', alpha=0.6, label=r'$V_{in}$ (Fuente)')

# Creámos líneas VACÍAS que llenaremos en la animación
line_vc, = ax1.plot([], [], lw=2, color='cyan', label=r'$V_C$')
line_vl, = ax1.plot([], [], lw=2, color='lime', linestyle='--', label=r'$V_L$')
ax1.legend(loc='upper right')

# --- Configuración Subplot 2: Corriente ---
ax2.set_xlim(t_span[0], t_span[1])
# Límites verticales dinámicos para corriente
max_i = max(current_data)
min_i = min(current_data)
margin_i = (max_i - min_i) * 0.1 if max_i != min_i else 0.1
ax2.set_ylim(min_i - margin_i, max_i + margin_i)

ax2.set_xlabel('Tiempo (s)')
ax2.set_ylabel('Corriente (A)')
ax2.grid(True, which='both', linestyle='--', alpha=0.3)

# Línea VACÍA para corriente
line_i, = ax2.plot([], [], lw=2, color='yellow', label=r'$i(t)$')
ax2.legend(loc='upper right')


# ==========================================
# 3. FUNCIONES DE ANIMACIÓN
# ==========================================

def init_anim():
    """Función inicial para limpiar el gráfico antes de empezar."""
    line_vc.set_data([], [])
    line_vl.set_data([], [])
    line_i.set_data([], [])
    return line_vc, line_vl, line_i

def update_anim(frame_idx):
    """Esta función se llama en cada frame.
    frame_idx es el índice del punto en el tiempo que toca dibujar."""
    
    # Cortamos los datos desde el inicio hasta el índice actual (frame_idx)
    x_now = time_data[:frame_idx]
    vc_now = vc_data[:frame_idx]
    vl_now = vl_data[:frame_idx]
    i_now = current_data[:frame_idx]
    
    # Actualizamos los datos de las líneas
    line_vc.set_data(x_now, vc_now)
    line_vl.set_data(x_now, vl_now)
    line_i.set_data(x_now, i_now)
    
    return line_vc, line_vl, line_i

# Creación del objeto de animación
ani = FuncAnimation(
    fig, 
    update_anim, 
    init_func=init_anim,
    frames=len(time_data),  # Número total de frames = número de puntos calculados
    interval=20,            # Milisegundos entre frames (menor = más rápido)
    blit=True,              # Optimización para que vaya más fluido
    repeat=False            # No repetir al terminar
)

plt.tight_layout()
plt.show()