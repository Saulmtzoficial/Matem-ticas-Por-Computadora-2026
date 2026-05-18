import numpy as np

# Datos de la tabla
x = np.array([0.97, 1.00, 1.05])
y = np.array([0.85040, 0.84147, 0.82612])

def estimar_derivadas_no_equidistantes(x_val, x_puntos, y_puntos):
    """
    Calcula derivadas usando el polinomio de Lagrange de segundo grado
    porque los puntos no están a la misma distancia.
    """
    # Definimos los puntos para que sea más fácil leer la fórmula
    x0, x1, x2 = x_puntos
    f0, f1, f2 = y_puntos
    
    # Derivada de Lagrange L'(x) evaluada en x1 = 1.00
    # La fórmula general para la primera derivada en x=x1 es:
    f_prime = (f0 * (2*x_val - x1 - x2) / ((x0 - x1) * (x0 - x2)) +
               f1 * (2*x_val - x0 - x2) / ((x1 - x0) * (x1 - x2)) +
               f2 * (2*x_val - x0 - x1) / ((x2 - x0) * (x2 - x1)))
    
    # La segunda derivada para un polinomio de grado 2 es constante:
    # f''(x) ≈ 2 * [ f0/((x0-x1)(x0-x2)) + f1/((x1-x0)(x1-x2)) + f2/((x2-x0)(x2-x1)) ]
    f_double_prime = 2 * (
        f0 / ((x0 - x1) * (x0 - x2)) +
        f1 / ((x1 - x0) * (x1 - x2)) +
        f2 / ((x2 - x0) * (x2 - x1))
    )
    
    return f_prime, f_double_prime

# Punto donde queremos estimar (x = 1.00)
punto_objetivo = 1.00
d1, d2 = estimar_derivadas_no_equidistantes(punto_objetivo, x, y)

print(f"--- Resultados para x = {punto_objetivo} ---")
print(f"f'(1) ≈ {d1:.5f}")
print(f"f''(1) ≈ {d2:.5f}")