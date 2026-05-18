import numpy as np

# Datos de la tabla (Vaya que son precisos)
x_puntos = np.array([2.36, 2.37, 2.38, 2.39])
f_puntos = np.array([0.85866, 0.86289, 0.86710, 0.87129])

# El tamaño del paso (h)
h = x_puntos[1] - x_puntos[0]

def calcular_derivadas():
    # Para f'(2.36) con O(h^2) hacia adelante usamos:
    # f'(x) ≈ [-3f(x) + 4f(x+h) - f(x+2h)] / (2h)
    f_prime = (-3*f_puntos[0] + 4*f_puntos[1] - f_puntos[2]) / (2 * h)
    
    # Para f''(2.36) con O(h^2) hacia adelante usamos:
    # f''(x) ≈ [2f(x) - 5f(x+h) + 4f(x+2h) - f(x+3h)] / h^2
    f_double_prime = (2*f_puntos[0] - 5*f_puntos[1] + 4*f_puntos[2] - f_puntos[3]) / (h**2)
    
    return f_prime, f_double_prime

# Ejecutamos la magia
derivada1, derivada2 = calcular_derivadas()

print(f"--- Resultados de CRACK ---")
print(f"Paso (h): {h}")
print(f"f'(2.36) ≈ {derivada1:.5f}")
print(f"f''(2.36) ≈ {derivada2:.5f}")