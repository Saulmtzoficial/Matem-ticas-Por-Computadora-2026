#Ejercicio 3
#Un servomotor controla la posiicon angular de un brazo mecanico unido a un resorte torsional no lineal. El equilibro del sistema se alcanza cuando el par del motor es igual al par del resorte.El modelo del sistema se describe por la ecuacion teta=M/(K+sin(teta)), donde teta es el angulo de equilibrio en radianes, M=3N*m=par aplicado por el servomotor, K=1.5N*m=constante del resorte. 

#Diseñar un codigo que determine el angulo de equilibro usando el metodo de  punto fijo , con una precision de 10ê-4, 50 iteraciones como valor maximo, imprimir una tabla con las siguientes columnas: i(numero de iteracion), tetha_i(valor actual de teta), f(theta_i)(evaluacion del valor actual de teta).

import numpy as np
def f(theta):
    M = 3.0  # Par aplicado por el servomotor
    K = 1.5  # Constante del resorte
    return M / (K + np.sin(theta))
def fixed_point_iteration(theta0, tol=1e-4, max_iter=50):

    print(f"{'i':<5} {'theta_i':<15} {'f(theta_i)':<15}")
    for i in range(max_iter):
        theta1 = f(theta0)  # Actualización de theta
        print(f"{i:<5} {theta0:<15.6f} {theta1:<15.6f}")
        if abs(theta1 - theta0) < tol:  # Verificar la convergencia
            break
        theta0 = theta1  # Preparar para la siguiente iteración
    return theta0
# Valor inicial para theta
theta0 = 0.5  # Un valor inicial razonable
# Ejecutar el método de punto fijo
theta_equilibrio = fixed_point_iteration(theta0)
print(f"Ángulo de equilibrio aproximado: {theta_equilibrio} radianes")

