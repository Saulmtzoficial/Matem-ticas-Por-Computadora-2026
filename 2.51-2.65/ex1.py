#Ejercicio 1
#La ecuacion x**3-1.2x**2-8.19x+13.23 tiene una raiz doble cerca de x=2.Diseña un codigo que utilice el metodo de Newthon Raphsonpara encontrar estas raices, la funcion y su derivada deben definirse dentro del codigo, la precision debe de ser de 10ê-4 con 50 iteraciones maximas, la terminal debe mostrar una tabla con las columas i(numero de iteracion), x_i(valor actual de x), f(x_i)(evalucion del valor actual de x).

import numpy as np
def f(x):
    return x**3 - 1.2*x**2 - 8.19*x + 13.23
def df(x):
    return 3*x**2 - 2.4*x - 8.19
def newton_raphson(x0, tol=1e-4, max_iter=50):
    print(f"{'i':<5} {'x_i':<15} {'f(x_i)':<15}")
    for i in range(max_iter):
        fx = f(x0)
        dfx = df(x0)
        if dfx == 0:  # Evitar división por cero
            print("Derivada es cero. No se puede continuar.")
            return None
        x1 = x0 - fx / dfx  # Actualización de x
        print(f"{i:<5} {x0:<15.6f} {fx:<15.6f}")
        if abs(fx) < tol:  # Verificar la convergencia
            break
        x0 = x1  # Preparar para la siguiente iteración
    return x0

# Valor inicial cerca de la raíz doble
x0 = 2.0
raiz = newton_raphson(x0)
print(f"Raíz aproximada: {raiz}")

#Imprimir tabla de resultados
