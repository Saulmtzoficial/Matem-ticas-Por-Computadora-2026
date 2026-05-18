#Ejercicio 2
#Diseñar un codigo con el metodo de la secante determinar las dos raices de la ecuacion: sinx+3cosx-2=0, dentro del intervalo [-2,2] 

import numpy as np
def f(x):   
    return np.sin(x) + 3*np.cos(x) - 2          
def secant_method(x0, x1, tol=1e-8, max_iter=100):
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        if fx1 - fx0 == 0:  # Evitar división por cero
            print("Función no cambia entre x0 y x1. No se puede continuar.")
            return None
        # Calcular el siguiente punto
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        if abs(f(x2)) < tol:  # Verificar la convergencia
            print(f"Raíz aproximada: {x2} encontrada en la iteración {i+1}")
            return x2
        # Actualizar para la siguiente iteración
        x0, x1 = x1, x2
    print("No se encontró una raíz dentro del número máximo de iteraciones.")
    return None
# Valores iniciales dentro del intervalo [-2, 2]
x0 = -2  
x1 = -0.1  
# Encontrar la primera raíz
raiz1 = secant_method(x0, x1)
# Para encontrar la segunda raíz, podemos usar un nuevo intervalo que no incluya la primera raíz encontrada
if raiz1 is not None:
    if raiz1 < 1.0:
        x0 = raiz1 + 0.1  # Mover ligeramente el intervalo hacia la derecha
        x1 = 1
    else:
        x0 = 0
        x1 = raiz1 - 0.1  # Mover ligeramente el intervalo hacia la izquierda
    raiz2 = secant_method(x0, x1)
    print(f"Segunda raíz aproximada: {raiz2}")


