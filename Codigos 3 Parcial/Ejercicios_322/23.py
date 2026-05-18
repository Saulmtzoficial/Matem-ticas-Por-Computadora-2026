import numpy as np

# Definimos la función real para calcular el error exacto
def f_real(x):
    return x + np.sin(x)/3

# Derivadas analíticas para comparar
def d2f_real(x): return -np.sin(x)/3
def d3f_real(x): return -np.cos(x)/3
def d4f_real(x): return np.sin(x)/3

# Datos de la tabla en x = 0.90 (i=3)
x0 = 0.90
h = 0.20

# Diferencias finitas desde la tabla para x=0.90
# Usaremos diferencias centradas para mayor precisión
f_0 = 1.1611
# Para derivadas de orden superior centradas en i=3, necesitamos puntos vecinos
# i=1(0.50), i=2(0.70), i=3(0.90), i=4(1.10), i=5(1.30)
f_vals = {
    0.50: 0.6598,
    0.70: 0.9147,
    0.90: 1.1611,
    1.10: 1.3971,
    1.30: 1.6212
}

def calcular_estimaciones():
    # 2da Derivada (Centrada O(h^2)): [f(x-h) - 2f(x) + f(x+h)] / h^2
    d2 = (f_vals[0.70] - 2*f_vals[0.90] + f_vals[1.10]) / (h**2)
    
    # 3ra Derivada (Centrada O(h^2)): [f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)] / (2h^3)
    d3 = (f_vals[1.30] - 2*f_vals[1.10] + 2*f_vals[0.70] - f_vals[0.50]) / (2 * (h**3))
    
    # 4ta Derivada (Centrada O(h^2)): [f(x+2h) - 4f(x+h) + 6f(x) - 4f(x-h) + f(x-2h)] / h^4
    d4 = (f_vals[1.30] - 4*f_vals[1.10] + 6*f_vals[0.90] - 4*f_vals[0.70] + f_vals[0.50]) / (h**4)
    
    return d2, d3, d4

# Cálculos
est_d2, est_d3, est_d4 = calcular_estimaciones()

# Resultados y Errores
print(f"--- Estimaciones en x = {x0} ---")
print(f"f''(0.90):  Estimado = {est_d2:.5f} | Real = {d2f_real(x0):.5f} | Error = {abs(est_d2 - d2f_real(x0)):.5f}")
print(f"f'''(0.90): Estimado = {est_d3:.5f} | Real = {d3f_real(x0):.5f} | Error = {abs(est_d3 - d3f_real(x0)):.5f}")
print(f"f^(4)(0.90):Estimado = {est_d4:.5f} | Real = {d4f_real(x0):.5f} | Error = {abs(est_d4 - d4f_real(x0)):.5f}")