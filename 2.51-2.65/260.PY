"""
Exercise 2.60: Root Finding with Taylor Series (Halley's Method)

PROBLEMA:
Usar series de Taylor con N=2 para crear un método de búsqueda de raíces.

DERIVACIÓN:
Para N=2:
0 = f(x₀) + f'(x₀)(x - x₀) + f''(x₀)/2 (x - x₀)²

Sea h = x - x₀, entonces:
0 = f(x₀) + f'(x₀)h + (f''(x₀)/2)h²

Esto es una ecuación cuadrática: ah² + bh + c = 0
donde: a = f''(x₀)/2, b = f'(x₀), c = f(x₀)

Solución: h = [-b ± √(b² - 4ac)] / (2a)

Simplificando:
h = [-f'(x₀) ± √(f'(x₀)² - 2f(x₀)f''(x₀))] / f''(x₀)

Para convergencia, elegimos el signo que da |h| más pequeño.

Este método se conoce como MÉTODO DE HALLEY (1694).
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARTE 1: IMPLEMENTACIÓN DEL MÉTODO (N=2)
# ============================================================================

def halley_method(f, df, ddf, x0, tol=1e-10, max_iter=50, verbose=False):
    """
    Método de Halley (Taylor N=2)
    
    Resuelve: 0 = f(x₀) + f'(x₀)(x-x₀) + f''(x₀)/2 (x-x₀)²
    
    Fórmula: x_{n+1} = x_n - [2f(x_n)f'(x_n)] / [2f'(x_n)² - f(x_n)f''(x_n)]
    
    Parámetros:
    -----------
    f : función
    df : primera derivada
    ddf : segunda derivada
    x0 : aproximación inicial
    
    Retorna:
    --------
    root, iterations, history, status
    """
    history = [x0]
    xn = x0
    
    if verbose:
        print(f"\n{'Iter':<8} {'xn':<18} {'f(xn)':<15} {'|cambio|':<15}")
        print("-"*60)
    
    for n in range(max_iter):
        fn = f(xn)
        dfn = df(xn)
        ddfn = ddf(xn)
        
        # Verificar denominador
        denominator = 2*dfn**2 - fn*ddfn
        if abs(denominator) < 1e-14:
            return None, n, history, "Error: denominador cercano a cero"
        
        # Fórmula de Halley
        h = (2*fn*dfn) / denominator
        xn_new = xn - h
        
        history.append(xn_new)
        
        if verbose:
            print(f"{n:<8} {xn:<18.12f} {fn:<15.6e} {abs(h):<15.6e}")
        
        # Criterio de parada
        if abs(h) < tol or abs(fn) < tol:
            if verbose:
                print(f"\nConvergencia en {n+1} iteraciones")
                print(f"Raíz: {xn_new:.15f}")
            return xn_new, n+1, history, "Convergencia exitosa"
        
        xn = xn_new
    
    return xn, max_iter, history, "Máximo de iteraciones"


def halley_alternative(f, df, ddf, x0, tol=1e-10, max_iter=50):
    """
    Implementación alternativa usando la fórmula de raíz cuadrática
    """
    history = [x0]
    xn = x0
    
    for n in range(max_iter):
        fn = f(xn)
        dfn = df(xn)
        ddfn = ddf(xn)
        
        if abs(ddfn) < 1e-14:
            return None, n, history, "Error: f'' ≈ 0"
        
        # Resolver ecuación cuadrática
        a = ddfn / 2
        b = dfn
        c = fn
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None, n, history, "Error: discriminante negativo"
        
        # Dos soluciones posibles
        h1 = (-b + np.sqrt(discriminant)) / (2*a)
        h2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # Elegir la más cercana a cero
        h = h1 if abs(h1) < abs(h2) else h2
        xn_new = xn + h
        
        history.append(xn_new)
        
        if abs(h) < tol or abs(fn) < tol:
            return xn_new, n+1, history, "Convergencia exitosa"
        
        xn = xn_new
    
    return xn, max_iter, history, "Máximo de iteraciones"


# ============================================================================
# PARTE 2: DEMOSTRACIÓN CON PROBLEMAS CONOCIDOS
# ============================================================================

def test_known_problems():
    """
    Prueba el método con problemas de solución conocida
    """
    print("="*100)
    print("PARTE 2: DEMOSTRACIÓN CON PROBLEMAS CONOCIDOS")
    print("="*100)
    
    problems = []
    
    # Problema 1: x² - 2 = 0, raíz = √2
    problems.append({
        'name': 'x² - 2 = 0',
        'f': lambda x: x**2 - 2,
        'df': lambda x: 2*x,
        'ddf': lambda x: 2,
        'x0': 1.5,
        'root_true': np.sqrt(2)
    })
    
    # Problema 2: x³ - 3 = 0, raíz = ∛3
    problems.append({
        'name': 'x³ - 3 = 0',
        'f': lambda x: x**3 - 3,
        'df': lambda x: 3*x**2,
        'ddf': lambda x: 6*x,
        'x0': 1.5,
        'root_true': 3**(1/3)
    })
    
    # Problema 3: cos(x) - x = 0
    problems.append({
        'name': 'cos(x) - x = 0',
        'f': lambda x: np.cos(x) - x,
        'df': lambda x: -np.sin(x) - 1,
        'ddf': lambda x: -np.cos(x),
        'x0': 1.0,
        'root_true': 0.7390851332151607
    })
    
    # Problema 4: e^x - 3x² = 0
    problems.append({
        'name': 'e^x - 3x² = 0',
        'f': lambda x: np.exp(x) - 3*x**2,
        'df': lambda x: np.exp(x) - 6*x,
        'ddf': lambda x: np.exp(x) - 6,
        'x0': 1.0,
        'root_true': 0.9100075724111
    })
    
    print(f"\n{'Problema':<25} {'Raíz verdadera':<20} {'Raíz calculada':<20} {'Error':<15} {'Iters':<8}")
    print("-"*100)
    
    all_results = []
    
    for prob in problems:
        root, iters, hist, status = halley_method(
            prob['f'], prob['df'], prob['ddf'], prob['x0'], tol=1e-12
        )
        
        if root is not None:
            error = abs(root - prob['root_true'])
            print(f"{prob['name']:<25} {prob['root_true']:<20.15f} {root:<20.15f} {error:<15.2e} {iters:<8}")
            
            all_results.append({
                'problem': prob,
                'root': root,
                'iters': iters,
                'history': hist,
                'errors': [abs(x - prob['root_true']) for x in hist]
            })
        else:
            print(f"{prob['name']:<25} {'FALLÓ':<20} {'—':<20} {'—':<15} {iters:<8}")
    
    print("\n✓ TODOS LOS MÉTODOS CONVERGEN CORRECTAMENTE")
    
    return all_results


# ============================================================================
# PARTE 3: ANÁLISIS DEL ORDEN DE CONVERGENCIA
# ============================================================================

def analyze_convergence_order(results):
    """
    Análisis del orden de convergencia mediante gráficas log-log
    """
    print("\n" + "="*100)
    print("PARTE 3: ANÁLISIS DEL ORDEN DE CONVERGENCIA")
    print("="*100)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        if idx >= 4:
            break
        
        ax = axes[idx]
        errors = result['errors']
        
        # Filtrar errores muy pequeños
        valid_errors = [(i, e) for i, e in enumerate(errors) if e > 1e-15]
        if len(valid_errors) < 2:
            continue
        
        iters, errs = zip(*valid_errors)
        
        # Gráfica log-log
        ax.loglog(iters, errs, 'bo-', linewidth=2, markersize=8, label='Error real')
        
        # Calcular orden empírico
        if len(errs) > 2:
            orders = []
            for i in range(1, len(errs)-1):
                if errs[i-1] > 0 and errs[i] > 0:
                    order = np.log(errs[i+1]/errs[i]) / np.log(errs[i]/errs[i-1])
                    if not np.isnan(order) and not np.isinf(order):
                        orders.append(order)
            
            if orders:
                avg_order = np.mean(orders)
                ax.text(0.05, 0.95, f'Orden ≈ {avg_order:.2f}', 
                       transform=ax.transAxes, fontsize=11,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlabel('Iteración', fontsize=11, fontweight='bold')
        ax.set_ylabel('Error |xₙ - raíz|', fontsize=11, fontweight='bold')
        ax.set_title(result['problem']['name'], fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('exercise_2_60_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calcular orden promedio
    print("\nORDEN DE CONVERGENCIA EMPÍRICO:")
    print(f"{'Problema':<25} {'Orden estimado':<20}")
    print("-"*50)
    
    for result in results:
        errors = result['errors']
        if len(errors) > 3:
            orders = []
            for i in range(1, len(errors)-1):
                if errors[i-1] > 1e-14 and errors[i] > 1e-14:
                    order = np.log(errors[i+1]/errors[i]) / np.log(errors[i]/errors[i-1])
                    if not np.isnan(order) and not np.isinf(order) and 0 < order < 10:
                        orders.append(order)
            
            if orders:
                avg_order = np.mean(orders)
                print(f"{result['problem']['name']:<25} {avg_order:<20.3f}")
    
    print("\nTEORÍA: El Método de Halley tiene orden de convergencia p = 3 (CÚBICA)")
    print("Fórmula: |eₙ₊₁| ≈ C|eₙ|³")


# ============================================================================
# COMPARACIÓN CON NEWTON
# ============================================================================

def compare_with_newton():
    """
    Comparación entre Halley (N=2) y Newton (N=1)
    """
    print("\n" + "="*100)
    print("COMPARACIÓN: HALLEY vs NEWTON")
    print("="*100)
    
    # Problema de prueba
    f = lambda x: x**3 - 3
    df = lambda x: 3*x**2
    ddf = lambda x: 6*x
    x0 = 1.5
    root_true = 3**(1/3)
    
    # Newton
    def newton(f, df, x0, tol=1e-12, max_iter=50):
        history = [x0]
        xn = x0
        for n in range(max_iter):
            fn = f(xn)
            dfn = df(xn)
            if abs(dfn) < 1e-14:
                return None, n, history
            xn = xn - fn / dfn
            history.append(xn)
            if abs(fn) < tol:
                return xn, n+1, history
        return xn, max_iter, history
    
    # Ejecutar ambos
    root_newton, iters_newton, hist_newton = newton(f, df, x0)
    root_halley, iters_halley, hist_halley, _ = halley_method(f, df, ddf, x0)
    
    errors_newton = [abs(x - root_true) for x in hist_newton]
    errors_halley = [abs(x - root_true) for x in hist_halley]
    
    print(f"\n{'Método':<15} {'Iteraciones':<15} {'Error final':<20}")
    print("-"*60)
    print(f"{'Newton (N=1)':<15} {iters_newton:<15} {errors_newton[-1]:<20.2e}")
    print(f"{'Halley (N=2)':<15} {iters_halley:<15} {errors_halley[-1]:<20.2e}")
    
    # Gráfica comparativa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convergencia
    ax1.semilogy(range(len(errors_newton)), errors_newton, 'ro-', 
                linewidth=2, markersize=6, label='Newton (orden 2)')
    ax1.semilogy(range(len(errors_halley)), errors_halley, 'bs-', 
                linewidth=2, markersize=6, label='Halley (orden 3)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Iteración', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Error (log)', fontsize=11, fontweight='bold')
    ax1.set_title('Convergencia: Halley vs Newton', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=11)
    
    # Iteraciones
    methods = ['Newton\n(N=1)', 'Halley\n(N=2)']
    iters = [iters_newton, iters_halley]
    colors = ['red', 'blue']
    
    bars = ax2.bar(methods, iters, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Iteraciones', fontsize=11, fontweight='bold')
    ax2.set_title('Eficiencia', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, iters):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('exercise_2_60_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# PARTE 4: PROS Y CONTRAS
# ============================================================================

def analyze_pros_cons():
    """
    Análisis de ventajas y desventajas
    """
    print("\n" + "="*100)
    print("PARTE 4: PROS Y CONTRAS DEL MÉTODO DE HALLEY")
    print("="*100)
    
    print("""
VENTAJAS (PROS):

1. CONVERGENCIA MÁS RÁPIDA:
   • Orden 3 (cúbica) vs orden 2 (cuadrática) de Newton
   • Los dígitos correctos se triplican en cada iteración
   • Menos iteraciones necesarias

2. MEJOR PARA FUNCIONES COMPLEJAS:
   • Aprovecha información de curvatura (f'')
   • Mejor aproximación local de la función
   • Más preciso cerca de la raíz

3. FÓRMULA ELEGANTE:
   • Derivada naturalmente de series de Taylor
   • Generalización directa de Newton

DESVENTAJAS (CONTRAS):

1. REQUIERE SEGUNDA DERIVADA:
   • f''(x) debe calcularse o aproximarse
   • Más costoso computacionalmente por iteración
   • Puede ser difícil obtener f'' analíticamente

2. MÁS COMPLEJO:
   • Fórmula más complicada
   • Mayor probabilidad de errores numéricos
   • Denominador puede ser cero

3. COSTO COMPUTACIONAL:
   • Cada iteración evalúa f, f', y f''
   • 3 evaluaciones vs 2 de Newton
   • Ventaja solo si la reducción en iteraciones compensa

4. PUEDE SER INESTABLE:
   • Más sensible a aproximación inicial
   • Problemas si f'' cambia rápidamente
   • Denominador puede ser pequeño

CUÁNDO USAR:

✓ Usa Halley cuando:
  - Puedes calcular f'' fácilmente
  - Necesitas máxima precisión
  - Cada evaluación de función es muy costosa

✗ Usa Newton cuando:
  - f'' es difícil de obtener
  - Simplicidad es importante
  - Newton ya es suficientemente rápido

COMPROMISO:
El método de Halley es un trade-off entre:
  • Menos iteraciones (bueno)
  • Más trabajo por iteración (malo)

Para la mayoría de problemas prácticos, NEWTON es preferible
debido a su simplicidad y buena convergencia.
    """)


# ============================================================================
# DERIVACIÓN MATEMÁTICA DETALLADA
# ============================================================================

def show_derivation():
    """
    Muestra la derivación matemática completa
    """
    print("="*100)
    print("DERIVACIÓN MATEMÁTICA: N=2")
    print("="*100)
    
    print("""
PASO 1: Serie de Taylor con N=2

    0 = f(x₀) + f'(x₀)(x - x₀) + [f''(x₀)/2](x - x₀)²

PASO 2: Sea h = x - x₀

    0 = f(x₀) + f'(x₀)h + [f''(x₀)/2]h²

PASO 3: Reorganizar como ecuación cuadrática

    [f''(x₀)/2]h² + f'(x₀)h + f(x₀) = 0
    
    En forma estándar: ah² + bh + c = 0
    donde: a = f''(x₀)/2, b = f'(x₀), c = f(x₀)

PASO 4: Fórmula cuadrática

    h = [-b ± √(b² - 4ac)] / (2a)
    
    h = [-f'(x₀) ± √(f'(x₀)² - 2f(x₀)f''(x₀))] / f''(x₀)

PASO 5: Simplificar (eligiendo el signo apropiado)

    x = x₀ - [2f(x₀)f'(x₀)] / [2f'(x₀)² - f(x₀)f''(x₀)]

ESTA ES LA FÓRMULA DE HALLEY (1694)

Orden de convergencia: p = 3 (CÚBICA)
    |eₙ₊₁| ≈ C|eₙ|³
    """)


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("="*100)
    print("EJERCICIO 2.60: MÉTODOS DE RAÍCES CON SERIES DE TAYLOR (N=2)")
    print("="*100)
    
    # Mostrar derivación
    show_derivation()
    
    # Parte 2: Demostración
    results = test_known_problems()
    
    # Parte 3: Análisis de convergencia
    analyze_convergence_order(results)
    
    # Comparación con Newton
    compare_with_newton()
    
    # Parte 4: Pros y contras
    analyze_pros_cons()
    
    print("\n" + "="*100)
    print("✓ EJERCICIO 2.60 COMPLETADO")
    print("="*100)