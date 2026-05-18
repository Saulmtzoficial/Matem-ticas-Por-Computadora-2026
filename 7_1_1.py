import numpy as np
import matplotlib.pyplot as plt

def dy(x, y):
    """Primera derivada: y' = x^2 - 4y"""
    return x**2 - 4*y

def d2y(x, y):
    """Segunda derivada: y'' = 2x - 4y'"""
    return 2*x - 4*dy(x, y)

def taylor_orden_2(x0, y0, h, pasos):
    """
    Resuelve la EDO usando el método de Series de Taylor de orden 2.
    """
    x_vals = [x0]
    y_vals = [y0]
    
    print("🚀 Iniciando simulación numérica...")
    print("-" * 50)
    print(f"Paso 0: x = {x0:.2f}, y = {y0:.6f}")
    
    for i in range(1, pasos + 1):
        x = x_vals[-1]
        y = y_vals[-1]
        
        # Calculamos las derivadas en el punto actual
        y_prima = dy(x, y)
        y_biprima = d2y(x, y)
        
        # Aplicamos nuestra fórmula chula de Taylor
        y_next = y + (h * y_prima) + ((h**2 / 2) * y_biprima)
        x_next = x + h
        
        # Guardamos los nuevos valores
        x_vals.append(x_next)
        y_vals.append(y_next)
        
        print(f"Paso {i}: x = {x_next:.2f}, y = {y_next:.7f}")
        
    print("-" * 50)
    print(f"💥 Valor final aproximado y({x_vals[-1]:.2f}) = {y_vals[-1]:.7f}")
    
    return x_vals, y_vals

# 🎒 Preparando el terreno (Nuestros datos iniciales)
x_inicial = 0.0
y_inicial = 1.0
h_paso = 0.05
num_pasos = 2

# ¡Que empiece la magia!
x_data, y_data = taylor_orden_2(x_inicial, y_inicial, h_paso, num_pasos)


# --- SECCIÓN DE VISUALIZACIÓN ---
# Configuración visual con paleta minimalista y blueprint
bg_color = '#464646'   # Fondo oscuro elegante
fg_color = '#f2f1fb'   # Texto y líneas claras
cyan_neon = '#00ffff'  # El toque visual

plt.figure(figsize=(8, 5), facecolor=bg_color)
ax = plt.axes()
ax.set_facecolor(bg_color)

# Ajustando los colores de los ejes para que combinen al cien
for spine in ax.spines.values():
    spine.set_color(fg_color)
ax.tick_params(axis='x', colors=fg_color)
ax.tick_params(axis='y', colors=fg_color)
ax.yaxis.label.set_color(fg_color)
ax.xaxis.label.set_color(fg_color)
ax.title.set_color(fg_color)

# Ploteamos nuestra aproximación
plt.plot(x_data, y_data, color=cyan_neon, marker='o', linestyle='dashed', 
         linewidth=2, markersize=8, label='Taylor Orden 2')

# Detalles de la gráfica
plt.title('Aproximación Numérica - Serie de Taylor', fontsize=14, pad=15)
plt.xlabel('Eje x', fontsize=12)
plt.ylabel('Eje y', fontsize=12)
plt.grid(color=fg_color, linestyle=':', alpha=0.3)
plt.legend(facecolor=bg_color, edgecolor=fg_color, labelcolor=fg_color)

# ¡Mostramos la joyita!
plt.show()