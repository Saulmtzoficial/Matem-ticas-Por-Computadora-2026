"""
Exercise 2.51: Bisection Method

PROBLEMA:
Sea f(x) una función continua en el intervalo [a,b] donde f(a)⋅f(b)<0.
Dar claramente todos los detalles matemáticos de cómo el Método de Bisección
aproxima la raíz de la función f(x) en el intervalo [a,b].

DETALLES MATEMÁTICOS DEL MÉTODO DE BISECCIÓN:

1. TEOREMA DEL VALOR INTERMEDIO:
   Si f es continua en [a,b] y f(a)⋅f(b) < 0, entonces existe al menos
   un c ∈ (a,b) tal que f(c) = 0.

2. ALGORITMO:
   Paso 0: Verificar que f(a)⋅f(b) < 0
   
   Para cada iteración n = 1, 2, 3, ...:
   
   Paso 1: Calcular el punto medio
           c_n = (a_n + b_n) / 2
   
   Paso 2: Evaluar f(c_n)
   
   Paso 3: Determinar el nuevo intervalo:
           - Si f(c_n) = 0: ¡Encontramos la raíz exacta! c_n es la raíz.
           - Si f(a_n)⋅f(c_n) < 0: La raíz está en [a_n, c_n]
             Entonces: a_{n+1} = a_n, b_{n+1} = c_n
           - Si f(c_n)⋅f(b_n) < 0: La raíz está en [c_n, b_n]
             Entonces: a_{n+1} = c_n, b_{n+1} = b_n
   
   Paso 4: Verificar criterio de parada:
           - |b_n - a_n| < tolerancia, o
           - |f(c_n)| < tolerancia, o
           - número máximo de iteraciones alcanzado

3. PROPIEDADES DE CONVERGENCIA:
   
   a) El error máximo en la iteración n está acotado por:
      |x_root - c_n| ≤ (b - a) / 2^(n+1)
   
   b) El método converge linealmente con razón 1/2
   
   c) Después de n iteraciones, el intervalo se reduce a:
      longitud = (b - a) / 2^n
   
   d) Para alcanzar una tolerancia ε, se necesitan al menos:
      n ≥ log₂((b - a) / ε)  iteraciones

4. VENTAJAS:
   - Siempre converge (garantizado por el Teorema del Valor Intermedio)
   - Robusto y simple de implementar
   - No requiere cálculo de derivadas

5. DESVENTAJAS:
   - Convergencia lenta (lineal)
   - Requiere que f(a)⋅f(b) < 0
   - No funciona con raíces de multiplicidad par
"""

import numpy as np
import matplotlib.pyplot as plt

def bisection_method(f, a, b, tol=1e-6, max_iter=100, verbose=True):
    """
    Implementación del Método de Bisección
    
    Parámetros:
    -----------
    f : función
        La función f(x) para la cual buscamos la raíz
    a : float
        Extremo izquierdo del intervalo
    b : float
        Extremo derecho del intervalo
    tol : float
        Tolerancia para el criterio de parada
    max_iter : int
        Número máximo de iteraciones
    verbose : bool
        Si True, imprime información de cada iteración
    
    Retorna:
    --------
    root : float
        Aproximación de la raíz
    iterations : int
        Número de iteraciones realizadas
    history : list
        Historia de aproximaciones en cada iteración
    """
    
    # Verificar condición inicial
    if f(a) * f(b) >= 0:
        raise ValueError(f"f(a)⋅f(b) debe ser < 0. Valores: f({a}) = {f(a)}, f({b}) = {f(b)}")
    
    # Inicialización
    history = []
    intervals = []
    
    if verbose:
        print("=" * 80)
        print("MÉTODO DE BISECCIÓN")
        print("=" * 80)
        print(f"Intervalo inicial: [{a}, {b}]")
        print(f"f(a) = {f(a):.6f}, f(b) = {f(b):.6f}")
        print(f"Longitud inicial: {b - a:.6f}")
        print(f"Tolerancia: {tol}")
        print("\n" + "-" * 80)
        print(f"{'Iter':<6} {'a':<12} {'b':<12} {'c':<12} {'f(c)':<12} {'|b-a|':<12}")
        print("-" * 80)
    
    # Iteraciones
    for n in range(max_iter):
        # Paso 1: Calcular punto medio
        c = (a + b) / 2.0
        fc = f(c)
        
        history.append(c)
        intervals.append([a, b])
        
        if verbose:
            print(f"{n+1:<6} {a:<12.6f} {b:<12.6f} {c:<12.6f} {fc:<12.6e} {abs(b-a):<12.6e}")
        
        # Paso 2: Verificar criterios de parada
        if abs(fc) < tol or abs(b - a) < tol:
            if verbose:
                print("-" * 80)
                print(f"\n¡Convergencia alcanzada en {n+1} iteraciones!")
                print(f"Raíz aproximada: {c:.10f}")
                print(f"|f(c)| = {abs(fc):.6e}")
                print(f"Longitud del intervalo final: {abs(b-a):.6e}")
            return c, n+1, history, intervals
        
        # Paso 3: Determinar nuevo intervalo
        if f(a) * fc < 0:
            # La raíz está en [a, c]
            b = c
        else:
            # La raíz está en [c, b]
            a = c
    
    # Si llegamos aquí, alcanzamos el máximo de iteraciones
    c = (a + b) / 2.0
    if verbose:
        print("-" * 80)
        print(f"\nAlcanzado el número máximo de iteraciones ({max_iter})")
        print(f"Mejor aproximación: {c:.10f}")
    
    return c, max_iter, history, intervals


def plot_bisection(f, a, b, history, intervals, true_root=None):
    """
    Grafica el proceso del Método de Bisección
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Gráfica 1: Función y convergencia de las aproximaciones
    x = np.linspace(a - 0.5, b + 0.5, 1000)
    y = [f(xi) for xi in x]
    
    ax1.plot(x, y, 'b-', linewidth=2, label='f(x)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # Marcar las aproximaciones
    colors = plt.cm.viridis(np.linspace(0, 1, len(history)))
    for i, (approx, color) in enumerate(zip(history, colors)):
        ax1.plot(approx, f(approx), 'o', color=color, markersize=8, 
                label=f'Iteración {i+1}' if i < 5 or i == len(history)-1 else '')
    
    if true_root is not None:
        ax1.axvline(x=true_root, color='r', linestyle='--', linewidth=2, 
                   label=f'Raíz verdadera: {true_root:.6f}')
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Método de Bisección: Función y Aproximaciones', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    
    # Gráfica 2: Reducción del intervalo
    interval_lengths = [b - a for a, b in intervals]
    ax2.semilogy(range(1, len(interval_lengths) + 1), interval_lengths, 'go-', linewidth=2, markersize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Iteración', fontsize=12)
    ax2.set_ylabel('Longitud del intervalo (escala log)', fontsize=12)
    ax2.set_title('Reducción del Intervalo: |b_n - a_n|', fontsize=14, fontweight='bold')
    
    # Añadir línea teórica
    theoretical = [(b - a) / (2**n) for n in range(1, len(interval_lengths) + 1)]
    ax2.semilogy(range(1, len(theoretical) + 1), theoretical, 'r--', linewidth=2, 
                label='Teórico: (b-a)/2^n')
    ax2.legend()
    
    # Gráfica 3: Error absoluto (si conocemos la raíz verdadera)
    if true_root is not None:
        errors = [abs(approx - true_root) for approx in history]
        ax3.semilogy(range(1, len(errors) + 1), errors, 'mo-', linewidth=2, markersize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Iteración', fontsize=12)
        ax3.set_ylabel('Error absoluto (escala log)', fontsize=12)
        ax3.set_title('Convergencia: |x_n - x_root|', fontsize=14, fontweight='bold')
        
        # Línea de cota teórica del error
        theoretical_error = [(b - a) / (2**(n+1)) for n in range(1, len(errors) + 1)]
        ax3.semilogy(range(1, len(theoretical_error) + 1), theoretical_error, 'r--', 
                    linewidth=2, label='Cota: (b-a)/2^(n+1)')
        ax3.legend()
    else:
        # Si no conocemos la raíz, graficar |f(c_n)|
        f_values = [abs(f(approx)) for approx in history]
        ax3.semilogy(range(1, len(f_values) + 1), f_values, 'co-', linewidth=2, markersize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Iteración', fontsize=12)
        ax3.set_ylabel('|f(c_n)| (escala log)', fontsize=12)
        ax3.set_title('Convergencia: |f(c_n)|', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# EJEMPLOS DE APLICACIÓN
# ============================================================================

print("\n" + "="*80)
print("EJEMPLO 1: f(x) = x³ - x - 2")
print("="*80)

# Definir función
def f1(x):
    return x**3 - x - 2

# Aplicar método de bisección
a1, b1 = 1, 2
root1, iters1, history1, intervals1 = bisection_method(f1, a1, b1, tol=1e-8)

# La raíz verdadera se puede calcular numéricamente con mayor precisión
true_root1 = 1.5213797068045676  # x³ - x - 2 = 0

# Graficar
plot_bisection(f1, a1, b1, history1, intervals1, true_root1)


print("\n" + "="*80)
print("EJEMPLO 2: f(x) = cos(x) - x")
print("="*80)

# Definir función
def f2(x):
    return np.cos(x) - x

# Aplicar método de bisección
a2, b2 = 0, 1
root2, iters2, history2, intervals2 = bisection_method(f2, a2, b2, tol=1e-8)

# Raíz verdadera (punto fijo de coseno)
true_root2 = 0.7390851332151607

# Graficar
plot_bisection(f2, a2, b2, history2, intervals2, true_root2)


print("\n" + "="*80)
print("EJEMPLO 3: f(x) = e^x - 3x²")
print("="*80)

# Definir función
def f3(x):
    return np.exp(x) - 3*x**2

# Aplicar método de bisección (una de las raíces está cerca de x = 0.9)
a3, b3 = 0.5, 1.5
root3, iters3, history3, intervals3 = bisection_method(f3, a3, b3, tol=1e-8)

# Graficar
plot_bisection(f3, a3, b3, history3, intervals3, root3)


# ============================================================================
# ANÁLISIS DE CONVERGENCIA
# ============================================================================

print("\n" + "="*80)
print("ANÁLISIS DE CONVERGENCIA")
print("="*80)

def analyze_convergence(b, a, tol):
    """
    Calcula el número de iteraciones necesarias para alcanzar una tolerancia dada
    """
    n = np.ceil(np.log2((b - a) / tol))
    return int(n)

# Para el primer ejemplo
n_required = analyze_convergence(b1, a1, 1e-8)
print(f"\nPara alcanzar tolerancia de 1e-8 en el intervalo [{a1}, {b1}]:")
print(f"Número de iteraciones teóricas requeridas: {n_required}")
print(f"Número de iteraciones realizadas: {iters1}")
print(f"\nCota del error después de {iters1} iteraciones:")
print(f"Error máximo ≤ (b-a)/2^(n+1) = {(b1-a1)/(2**(iters1+1)):.6e}")
print(f"Error real: {abs(root1 - true_root1):.6e}")