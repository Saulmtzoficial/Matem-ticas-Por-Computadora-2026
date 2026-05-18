import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """Nuestra ecuación diferencial: y' = x^2 - 4y"""
    return x**2 - 4*y

def runge_kutta_2(x0, y0, h, pasos):
    """Simulación usando Runge-Kutta de 2do Orden (RK2)"""
    x_vals = [x0]
    y_vals = [y0]
    
    for i in range(pasos):
        x = x_vals[-1]
        y = y_vals[-1]
        
        k1 = f(x, y)
        k2 = f(x + h, y + h * k1)
        
        y_next = y + (h / 2) * (k1 + k2)
        x_next = x + h
        
        x_vals.append(x_next)
        y_vals.append(y_next)
        
    return x_vals, y_vals

def runge_kutta_4(x0, y0, h, pasos):
    """Simulación usando Runge-Kutta de 4to Orden (RK4)"""
    x_vals = [x0]
    y_vals = [y0]
    
    for i in range(pasos):
        x = x_vals[-1]
        y = y_vals[-1]
        
        k1 = f(x, y)
        k2 = f(x + h/2, y + (h/2) * k1)
        k3 = f(x + h/2, y + (h/2) * k2)
        k4 = f(x + h, y + h * k3)
        
        y_next = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        x_next = x + h
        
        x_vals.append(x_next)
        y_vals.append(y_next)
        
    return x_vals, y_vals

# 🎒 Preparando el terreno
x_inicial = 0.0
y_inicial = 1.0
h_paso = 0.1
num_pasos = 5  # Haremos 5 pasos para que la gráfica luzca chula, pero nos enfocaremos en el primero.

# ¡A darle que es mole de olla!
x_rk2, y_rk2 = runge_kutta_2(x_inicial, y_inicial, h_paso, num_pasos)
x_rk4, y_rk4 = runge_kutta_4(x_inicial, y_inicial, h_paso, num_pasos)

print("🚀 Resultados para un solo paso (x = 0.1):")
print("-" * 50)
print(f"RK2 (Orden 2) -> y(0.1) = {y_rk2[1]:.5f}")
print(f"RK4 (Orden 4) -> y(0.1) = {y_rk4[1]:.5f}")
print("-" * 50)

# --- SECCIÓN DE VISUALIZACIÓN CYBERPUNK ---
bg_color = '#464646'   
fg_color = '#f2f1fb'   
cyan_neon = '#00ffff'  
magenta_neon = '#ff00ff' 

plt.figure(figsize=(9, 6), facecolor=bg_color)
ax = plt.axes()
ax.set_facecolor(bg_color)

# Estilizando ejes
for spine in ax.spines.values():
    spine.set_color(fg_color)
ax.tick_params(axis='x', colors=fg_color)
ax.tick_params(axis='y', colors=fg_color)
ax.yaxis.label.set_color(fg_color)
ax.xaxis.label.set_color(fg_color)
ax.title.set_color(fg_color)

# Ploteamos ambas aproximaciones
plt.plot(x_rk2, y_rk2, color=cyan_neon, marker='o', linestyle='dashed', 
         linewidth=2, markersize=7, label='RK2 (Orden 2)')
plt.plot(x_rk4, y_rk4, color=magenta_neon, marker='s', linestyle='-', 
         linewidth=2, markersize=7, label='RK4 (Orden 4)', alpha=0.8)

# Resaltar el primer paso (el que pide el problema)
plt.axvline(x=0.1, color=fg_color, linestyle=':', alpha=0.5)
plt.text(0.12, 0.9, 'Meta del\nproblema (x=0.1)', color=fg_color, fontsize=10)

# Detalles de la gráfica
plt.title('Comparativa Runge-Kutta - Simulación Numérica', fontsize=15, pad=15)
plt.xlabel('Eje x', fontsize=12)
plt.ylabel('Eje y', fontsize=12)
plt.grid(color=fg_color, linestyle=':', alpha=0.2)
plt.legend(facecolor=bg_color, edgecolor=fg_color, labelcolor=fg_color)

# ¡Mostramos la joyita!
plt.show()