#Resuelve la siguiente  matriz 11x11 que representa un sistema de ecuaciones, resolver por el metodo de la transpuesta de la matriz de coeficientes
import numpy as np
# Matriz de coeficientes
A = np.array([[2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],  
                [0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 2 ,1],
                [1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2]])

# Vector de términos independientes
b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
# Transpuesta de la matriz de coeficientes
A_transpuesta = A.T
# Resolviendo el sistema de ecuaciones
x = np.linalg.solve(A_transpuesta, b)
print("La solución del sistema de ecuaciones es:")
print(x)