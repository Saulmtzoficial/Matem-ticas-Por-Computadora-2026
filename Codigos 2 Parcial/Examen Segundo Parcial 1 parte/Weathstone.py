import numpy as np

# Valores de las resistencias
R1, R2, R3, R4, R5, R6, Rx = 2, 15, 6, 14, 5, 0.1, 7
V1, V6 = 5, 0
# Orden de las variables: [V2, V3, V4, V5, I4, I1, I3, I6, Ix, I2, I5]
A = np.zeros((11, 11))
B = np.zeros(11)

# --- Ecuaciones de la Ley de Ohm ---
# Ec 0: V2 + R4*I4 = V1
A[0, 0] = 1; A[0, 4] = R4; B[0] = V1
# Ec 1: V2 - V3 - R1*I1 = 0
A[1, 0] = 1; A[1, 1] = -1; A[1, 5] = -R1
# Ec 2: V2 - V4 - R3*I3 = 0
A[2, 0] = 1; A[2, 2] = -1; A[2, 6] = -R3
# Ec 3: V3 - V4 - R6*I6 = 0
A[3, 1] = 1; A[3, 2] = -1; A[3, 7] = -R6
# Ec 4: V3 - V5 - Rx*Ix = 0
A[4, 1] = 1; A[4, 3] = -1; A[4, 8] = -Rx
# Ec 5: V4 - V5 - R2*I2 = 0
A[5, 2] = 1; A[5, 3] = -1; A[5, 9] = -R2
# Ec 6: V5 - R5*I5 = V6
A[6, 3] = 1; A[6, 10] = -R5; B[6] = V6

# --- Ecuaciones de la Ley de Corrientes de Kirchhoff (LCK) ---
# Ec 7: I4 - I1 - I3 = 0
A[7, 4] = 1; A[7, 5] = -1; A[7, 6] = -1
# Ec 8: I1 - I6 - Ix = 0
A[8, 5] = 1; A[8, 7] = -1; A[8, 8] = -1
# Ec 9: I3 + I6 - I2 = 0
A[9, 6] = 1; A[9, 7] = 1; A[9, 9] = -1
# Ec 10: Ix + I2 - I5 = 0
A[10, 8] = 1; A[10, 9] = 1; A[10, 10] = -1

# Resolver el sistema de ecuaciones
solucion = np.linalg.solve(A, B)

# Mostrar resultados
etiquetas = ["V2", "V3", "V4", "V5", "I4", "I1", "I3", "I6", "Ix", "I2", "I5"]
print(f"{'Variable':<10} | {'Valor':<10}")
print("-" * 25)
for etiqueta, val in zip(etiquetas, solucion):
    unidad = "V" if etiqueta.startswith("V") else "A"
    print(f"{etiqueta:<10} | {val:>8.4f} {unidad}")