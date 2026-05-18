import numpy as np

def f_exacta(x):
    """Función exacta f(x) = 1 + log10(x)"""
    return 1 + np.log10(x)

def derivada_exacta(x):
    """Derivada exacta f'(x) = 1 / (x * ln(10))"""
    return 1 / (x * np.log(10))

def diferencias_divididas_newton(x, y):
    """Calcula la tabla de diferencias divididas de Newton."""
    n = len(y)
    tabla = np.zeros([n, n])
    tabla[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            tabla[i][j] = (tabla[i+1][j-1] - tabla[i][j-1]) / (x[i+j] - x[i])
    return tabla

def evaluar_derivada_newton(puntos_x, tabla_diff_div, x_objetivo):
    """
    Evalúa la derivada de un polinomio de Newton cuadrático.
    P2(x) = f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1)
    P2'(x) = f[x0,x1] + f[x0,x1,x2] * (2*x - x0 - x1)
    """
    f_x0_x1 = tabla_diff_div[0, 1]
    f_x0_x1_x2 = tabla_diff_div[0, 2]
    
    x0, x1 = puntos_x[0], puntos_x[1]
    
    derivada = f_x0_x1 + f_x0_x1_x2 * (2 * x_objetivo - x0 - x1)
    return derivada

# ==========================================
# CONFIGURACIÓN DE PUNTOS Y OBJETIVO
# ==========================================
# Usando los 3 mejores puntos típicos alrededor de 0.268 (asumiendo paso h=0.1)
puntos_x = np.array([0.200, 0.300, 0.400])
objetivo = 0.268

# ==========================================
# 1. ALTA PRECISIÓN (Sin redondeo significativo)
# ==========================================
y_preciso = f_exacta(puntos_x)
tabla_diff_precisa = diferencias_divididas_newton(puntos_x, y_preciso)
derivada_precisa = evaluar_derivada_newton(puntos_x, tabla_diff_precisa, objetivo)

# ==========================================
# 2. VALORES REDONDEADOS (4 decimales)
# ==========================================
y_redondeado = np.round(y_preciso, 4)
tabla_diff_redondeada = diferencias_divididas_newton(puntos_x, y_redondeado)
derivada_redondeada = evaluar_derivada_newton(puntos_x, tabla_diff_redondeada, objetivo)

# ==========================================
# ANÁLISIS DE ERROR
# ==========================================
valor_real = derivada_exacta(objetivo)

error_preciso = abs(valor_real - derivada_precisa)
error_redondeado = abs(valor_real - derivada_redondeada)

print("--- Datos Utilizados ---")
print(f"Valores Y precisos: {y_preciso}")
print(f"Valores Y redondeados: {y_redondeado}\n")

print("--- Estimaciones de la Derivada en x = 0.268 ---")
print(f"f'(x) Analítica Exacta:         {valor_real:.6f}")
print(f"Usando Valores Precisos:        {derivada_precisa:.6f}")
print(f"Usando Valores Redond. (4-dec): {derivada_redondeada:.6f}\n")

print("--- Comparación de Errores ---")
print(f"Solo Error de Truncamiento (Preciso): {error_preciso:.6f}")
print(f"Error Total (Datos Redondeados):      {error_redondeado:.6f}")
print(f"\nDiferencia causada por el redondeo:   {abs(error_redondeado - error_preciso):.6f}")

# Responder a la pregunta conceptual
print("\n--- Conclusión ---")
if error_redondeado > error_preciso * 2:
    print("El redondeo impacta significativamente el error. En la diferenciación numérica,")
    print("restar valores redondeados cercanos lleva a una cancelación catastrófica,")
    print("haciendo que el error de redondeo a menudo sea MÁS importante que el de truncamiento.")
else:
    print("En este caso específico, el error de truncamiento sigue siendo el factor dominante.")