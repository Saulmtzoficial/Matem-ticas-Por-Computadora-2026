import numpy as np

# Datos de la tabla
x_puntos = np.array([0.84, 0.92, 1.00, 1.08, 1.16])
f_puntos = np.array([0.431711, 0.398519, 0.367879, 0.339596, 0.313486])

# El tamaño del paso constante (h)
h = 0.08

def calcular_segunda_derivada_pro():
    # Usaremos la fórmula de diferencias centradas de orden O(h^4)
    # Para f''(x) en el punto central (x2 = 1.00):
    # f''(x) ≈ [-f(x0) + 16f(x1) - 30f(x2) + 16f(x3) - f(x4)] / (12 * h^2)
    
    f0, f1, f2, f3, f4 = f_puntos
    
    f_double_prime = (-f0 + 16*f1 - 30*f2 + 16*f3 - f4) / (12 * (h**2))
    
    return f_double_prime

# Ejecutamos para ver qué sale
resultado = calcular_segunda_derivada_pro()

print(f"--- Cálculo de Alta Precisión ---")
print(f"Paso (h): {h}")
print(f"f''(1.00) ≈ {resultado:.7f}")