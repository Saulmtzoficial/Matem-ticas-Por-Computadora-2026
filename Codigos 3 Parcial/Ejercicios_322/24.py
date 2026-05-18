import numpy as np

# Datos de la tabla del Ejercicio 19
# x: [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
# f: [0.3985, 0.6598, 0.9147, 1.1611, 1.3971, 1.6212, 1.8325]
f = {
    0.3: 0.3985, 0.5: 0.6598, 0.7: 0.9147, 
    0.9: 1.1611, 1.1: 1.3971, 1.3: 1.6212, 1.5: 1.8325
}

def f_real_2da(x):
    return -np.sin(x)/3

def richardson_extrapolation():
    x0 = 0.9
    h1 = 0.2
    h2 = 0.4 # El doble del paso original
    
    # 1. Estimación con h = 0.2 (Orden h^2)
    # D(h) = [f(x+h) - 2f(x) + f(x-h)] / h^2
    D_h1 = (f[0.7] - 2*f[0.9] + f[1.1]) / (h1**2)
    
    # 2. Estimación con 2h = 0.4 (Orden h^2)
    # D(2h) = [f(x+2h) - 2f(x) + f(x-2h)] / (2h)^2
    D_h2 = (f[0.5] - 2*f[0.9] + f[1.3]) / (h2**2)
    
    # 3. Extrapolación de Richardson para eliminar el error O(h^2)
    # Esto nos da un resultado de orden O(h^4)
    # Formula: R = [4 * D(h) - D(2h)] / 3
    resultado_extrapolado = (4 * D_h1 - D_h2) / 3
    
    return D_h1, D_h2, resultado_extrapolado

d_h, d_2h, final = richardson_extrapolation()
valor_real = f_real_2da(0.9)

print(f"--- Extrapolación de Richardson en x = 0.90 ---")
print(f"Estimación D(h=0.2): {d_h:.6f}")
print(f"Estimación D(h=0.4): {d_2h:.6f}")
print(f"Resultado Extrapolado: {final:.6f}")
print(f"Valor Real Analítico: {valor_real:.6f}")
print(f"Error Final: {abs(final - valor_real):.6f}")