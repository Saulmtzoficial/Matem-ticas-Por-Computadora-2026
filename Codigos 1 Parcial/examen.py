#Se requiere elaborar un programa que permita la resolucion de raices por el metodo de Biseccion, para la funcion f(x) = 1/sqrt(x) - 4.5, con un error de 10^-8 y un maximo de iteraciones de 100. El programa debe mostrar el valor de la raiz, el numero de iteraciones y el error final.

import numpy as np

def bisection(f, a, b, tol=1e-8, max_iter=100):
    if f(a) * f(b) >= 0:
        print("La función no cambia de signo en el intervalo [a, b].")
        return None, None, None

    iter_count = 0
    error = float('inf')
    c = a

    while iter_count < max_iter and error > tol:
        c = (a + b) / 2  # Punto medio
        fc = f(c)

        if fc == 0:  # Encontramos la raíz exacta
            break
        elif f(a) * fc < 0:  # La raíz está en [a, c]
            b = c
        else:  # La raíz está en [c, b]
            a = c

        iter_count += 1
        error = abs(fc)

    return c, iter_count, error
# Definir la función f(x) = 1/sqrt(x) - 4.5
def f(x):
    return 1/np.sqrt(x) - 4.5
# Definir el intervalo [a, b]
a = 0.01  # No puede ser 0 porque f(x) no está
b = 1.0   # Un valor razonable para b
# Ejecutar el método de bisección
raiz, iteraciones, error_final = bisection(f, a, b)
# Mostrar los resultados
print(f"Raíz aproximada: {raiz}")
print(f"Número de iteraciones: {iteraciones}")
print(f"Error final: {error_final}")
