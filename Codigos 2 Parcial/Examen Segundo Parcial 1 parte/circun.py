import numpy as np

# Puntos de datos registrados por el sistema de adquisición
x_puntos = np.array([-0.2047, 4.7297, 3.5377])
y_puntos = np.array([5.2202, 1.2313, 6.5454])


A = np.column_stack((x_puntos, y_puntos, np.ones(3)))
B = -(x_puntos**2 + y_puntos**2)

# Resolver el sistema de ecuaciones lineales
try:
    D, E, F = np.linalg.solve(A, B)

    # Calcular el centro (a, b) y el radio r
    a = -D / 2
    b = -E / 2
    r = np.sqrt(a**2 + b**2 - F)

    print(f"--- Resultados ---")
    print(f"Centro (a, b): ({a:.5f}, {b:.5f})")
    print(f"Radio r:      {r:.5f}")

    # Apartado de comprobación usando el primer punto (-0.2047, 5.2202)
    x1, y1 = x_puntos[0], y_puntos[0]
    lado_izquierdo = (x1 - a)**2 + (y1 - b)**2
    lado_derecho = r**2
    diferencia = abs(lado_izquierdo - lado_derecho)

    print(f"\n--- Comprobación (Primer Punto) ---")
    print(f"Lado Izquierdo (x-a)^2 + (y-b)^2: {lado_izquierdo:.6f}")
    print(f"Lado Derecho r^2:                {lado_derecho:.6f}")
    print(f"Diferencia:                      {diferencia:.2e}")

    # Verificación de la tolerancia 
    if diferencia < 1e-4:
        print("Verificación Exitosa")
    else:
        print("Verificación Fallida")

except np.linalg.LinAlgError:
    print("Error")