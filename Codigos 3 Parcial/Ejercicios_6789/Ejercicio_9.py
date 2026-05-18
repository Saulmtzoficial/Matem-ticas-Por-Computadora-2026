import numpy as np

# Datos de la tabla (muy precisos, por cierto)
x_puntos = np.array([0, 0.1, 0.2, 0.3, 0.4])
f_puntos = np.array([0.000000, 0.078348, 0.138910, 0.192916, 0.244981])

# El paso constante
h = 0.1

def calcular_derivada_max_precision():
    # Usamos la fórmula de diferencias centradas de orden O(h^4)
    # f'(x) ≈ [f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)] / (12h)
    
    f_m2 = f_puntos[0] # f(0)
    f_m1 = f_puntos[1] # f(0.1)
    # f_0 = f_puntos[2]  (no se usa en esta fórmula específica)
    f_p1 = f_puntos[3] # f(0.3)
    f_p2 = f_puntos[4] # f(0.4)
    
    f_prime = (f_m2 - 8*f_m1 + 8*f_p1 - f_p2) / (12 * h)
    
    return f_prime

# Ejecutamos la magia
resultado = calcular_derivada_max_precision()

print(f"--- Resultado Final ---")
print(f"Paso (h): {h}")
print(f"f'(0.2) ≈ {resultado:.7f}")