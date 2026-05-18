import numpy as np
import matplotlib.pyplot as plt

# Parámetros (¡vaya datos más exactos!)
V = 12   # Voltaje en volts
L = 0.5  # Inductancia en henries
R = 10   # Resistencia en ohms

# Creamos el tiempo de 0 a 1 segundo
t = np.linspace(0, 1, 1000)

# La fórmula mágica de la corriente i(t) para un circuito RL
# Basada en la ecuación diferencial: L(di/dt) + Ri = V
i = (V / R) * (1 - np.exp(-R * t / L))

# Calculamos el valor específico en t = 0.15 para que no haya duda
t_especifico = 0.1
i_especifico = (V / R) * (1 - np.exp(-R * t_especifico / L))
print(f"La corriente en t = 0.15s es aproximadamente: {i_especifico:.4f} A")

# Graficamos toda la cosa
plt.figure(figsize=(10, 5))
plt.plot(t, i, label='Corriente i(t)', color='blue', linewidth=2)
plt.title('Corriente i(t) en un Circuito RL (¡Súper interesante!)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Corriente (A)')

# Aquí corregí la línea roja que nos estaba dando lata
plt.axvline(x=0.15, color='red', linestyle='--', label='Punto t = 0.15s')

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()