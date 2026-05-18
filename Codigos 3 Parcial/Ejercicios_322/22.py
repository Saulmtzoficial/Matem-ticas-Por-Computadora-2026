# Versión "Sin SymPy" para que no sufras, pillo
import numpy as np

# Estos son los coeficientes que sacamos con el método de coeficientes indeterminados
# Para f''(x0), f'''(x0) y f(4)(x0) con h constante
def mostrar_formulas_directas():
    print("--- Fórmulas para 5 puntos equidistantes (x-2 a x2) ---")
    
    print("\n1. Segunda Derivada f''(x0):")
    print("f''(x0) ≈ [-f(x-2h) + 16f(x-h) - 30f(x0) + 16f(x+h) - f(x+2h)] / (12*h^2)")
    
    print("\n2. Tercera Derivada f'''(x0):")
    print("f'''(x0) ≈ [-f(x-2h) + 2f(x-h) - 2f(x+h) + f(x+2h)] / (2*h^3)")
    
    print("\n3. Cuarta Derivada f^(4)(x0):")
    print("f^(4)(x0) ≈ [f(x-2h) - 4f(x-h) + 6f(x0) - 4f(x+h) + f(x+2h)] / h^4")

mostrar_formulas_directas()