import numpy as np

def f(x):
    return np.sin(x/2)**2

def f_prime_analitica(x):
    # Usando regla de la cadena: 2*sin(x/2)*cos(x/2)*(1/2) = sin(x/2)cos(x/2)
    # Por identidad: (1/2)sin(x)
    return 0.5 * np.sin(x)

def richardson_iterativo(x0, h_inicial):
    # Lista para guardar nuestras estimaciones de nivel 1 (Diferencias centradas)
    # D(h, 0) son diferencias centradas de orden O(h^2)
    R = []
    
    print(f"{'h':<10} | {'Estimación R(i,0)':<18} | {'Nivel Superior (Richardson)':<20}")
    print("-" * 60)

    for i in range(4):  # Haremos 4 niveles de refinamiento
        h = h_inicial / (2**i)
        # Diferencia centrada básica: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
        estimacion_base = (f(x0 + h) - f(x0 - h)) / (2 * h)
        R.append([estimacion_base])
        
        # Aplicamos las extrapolaciones para subir de orden
        for j in range(1, i + 1):
            # Fórmula general de Richardson: 
            # R(i,j) = R(i, j-1) + [R(i, j-1) - R(i-1, j-1)] / (4^j - 1)
            valor = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (4**j - 1)
            R[i].append(valor)
        
        fila_str = " | ".join([f"{v:.8f}" for v in R[i]])
        print(f"{h:<10.4f} | {fila_str}")

    return R

x_target = 0.32
h_start = 0.1

tabla_resultados = richardson_iterativo(x_target, h_start)
final_val = tabla_resultados[-1][-1]
real_val = f_prime_analitica(x_target)

print("-" * 60)
print(f"Resultado final extrapolado: {final_val:.7f}")
print(f"Valor real analítico:       {real_val:.7f}")
print(f"Error absoluto:             {abs(final_val - real_val):.2e}")