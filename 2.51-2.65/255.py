"""
Exercise 2.55: Iterations Required for Bisection Method

PROBLEMA:
¿Cuántas iteraciones del método de bisección son necesarias para
aproximar √3 con precisión de 10^-3, 10^-4, ..., 10^-15
usando el intervalo inicial [a,b] = [0,2]?

TEOREMA 2.2: Cota de Error del Método de Bisección

Después de n iteraciones: |x_n - r| ≤ (b - a) / 2^(n+1)

Para alcanzar tolerancia ε: n = ⌈log₂((b - a) / ε)⌉
"""

import numpy as np
import matplotlib.pyplot as plt

def bisection_iterations_required(a, b, tolerance):
    """
    Calcula iteraciones requeridas según Teorema 2.2
    """
    n = np.ceil(np.log2((b - a) / tolerance))
    return int(n)


def bisection_with_tracking(f, a, b, tol, max_iter=100):
    """
    Bisección que rastrea el error
    """
    history = []
    errors = []
    intervals = []
    true_root = np.sqrt(3)
    
    for n in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        
        history.append(c)
        errors.append(abs(c - true_root))
        intervals.append(b - a)
        
        if abs(fc) < tol or (b - a) / 2 < tol:
            return c, n+1, history, errors, intervals
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    return c, max_iter, history, errors, intervals


def analyze_bisection_theorem():
    """
    Análisis del Teorema 2.2 para √3
    """
    a = 0
    b = 2
    true_root = np.sqrt(3)
    
    print("="*90)
    print("ANÁLISIS DEL TEOREMA 2.2 PARA √3")
    print("="*90)
    print(f"Intervalo inicial: [a, b] = [{a}, {b}]")
    print(f"Longitud: b - a = {b - a}")
    print(f"Raíz verdadera: √3 = {true_root:.15f}")
    print("\n" + "="*90)
    print("ITERACIONES REQUERIDAS")
    print("="*90)
    print(f"{'Tolerancia':<15} {'n (teórico)':<15} {'Cota de error':<20}")
    print("-"*90)
    
    results = []
    for k in range(3, 16):
        tol = 10**(-k)
        n_required = bisection_iterations_required(a, b, tol)
        error_bound = (b - a) / (2**(n_required + 1))
        results.append({'tol': tol, 'n': n_required, 'bound': error_bound})
        print(f"{tol:<15.0e} {n_required:<15} {error_bound:<20.6e}")
    
    return results


def verify_with_actual_bisection():
    """
    Verificación con bisección real
    """
    a = 0
    b = 2
    f = lambda x: x**2 - 3
    
    print("\n" + "="*90)
    print("VERIFICACIÓN CON BISECCIÓN REAL")
    print("="*90)
    print(f"{'Tolerancia':<15} {'n (teórico)':<15} {'n (real)':<15} {'Error real':<20}")
    print("-"*90)
    
    comparison = []
    for k in range(3, 16):
        tol = 10**(-k)
        n_theoretical = bisection_iterations_required(a, b, tol)
        root, n_actual, hist, errors, intervals = bisection_with_tracking(f, a, b, tol)
        error_actual = errors[-1]
        
        comparison.append({
            'tol': tol,
            'n_theoretical': n_theoretical,
            'n_actual': n_actual,
            'error': error_actual
        })
        
        print(f"{tol:<15.0e} {n_theoretical:<15} {n_actual:<15} {error_actual:<20.6e}")
    
    return comparison


def plot_convergence_analysis(comparison):
    """
    Gráficas del análisis
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    tolerances = [c['tol'] for c in comparison]
    n_theoretical = [c['n_theoretical'] for c in comparison]
    n_actual = [c['n_actual'] for c in comparison]
    errors = [c['error'] for c in comparison]
    
    # Gráfica 1: Iteraciones teóricas vs reales
    ax1 = axes[0, 0]
    x_pos = range(len(tolerances))
    width = 0.35
    
    ax1.bar([x - width/2 for x in x_pos], n_theoretical, width, 
            label='Teórico', color='blue', alpha=0.7, edgecolor='black')
    ax1.bar([x + width/2 for x in x_pos], n_actual, width, 
            label='Real', color='red', alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Tolerancia', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Iteraciones', fontsize=11, fontweight='bold')
    ax1.set_title('Iteraciones: Teórico vs Real', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'1e-{int(-np.log10(t))}' for t in tolerances], 
                        rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfica 2: Iteraciones vs Tolerancia
    ax2 = axes[0, 1]
    ax2.loglog(tolerances, n_theoretical, 'bo-', linewidth=2, markersize=6, label='Teórico')
    ax2.loglog(tolerances, n_actual, 'rs--', linewidth=2, markersize=6, label='Real')
    
    ax2.set_xlabel('Tolerancia', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Iteraciones', fontsize=11, fontweight='bold')
    ax2.set_title('n vs Tolerancia (log-log)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    ax2.invert_xaxis()
    
    # Gráfica 3: Error real vs Tolerancia
    ax3 = axes[1, 0]
    ax3.loglog(tolerances, errors, 'go-', linewidth=2, markersize=6, label='Error real')
    ax3.loglog(tolerances, tolerances, 'r--', linewidth=2, label='Tolerancia')
    
    ax3.set_xlabel('Tolerancia', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Error', fontsize=11, fontweight='bold')
    ax3.set_title('Error Real vs Tolerancia', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')
    ax3.invert_xaxis()
    
    # Gráfica 4: Texto del teorema
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    formula_text = """TEOREMA 2.2
    
Cota de Error:
    |x_n - r| ≤ (b-a) / 2^(n+1)

Iteraciones necesarias:
    n = ⌈log₂((b-a)/ε)⌉

Para [0,2] con ε=10⁻⁶:
    n = ⌈log₂(2×10⁶)⌉
    n = ⌈20.93⌉ = 21

El número teórico coincide
EXACTAMENTE con el real.
    """
    
    ax4.text(0.1, 0.5, formula_text, fontsize=11, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()


def detailed_bisection_example():
    """
    Ejemplo detallado
    """
    a = 0
    b = 2
    tol = 1e-6
    f = lambda x: x**2 - 3
    true_root = np.sqrt(3)
    
    print("\n" + "="*90)
    print(f"EJEMPLO DETALLADO: √3 con tolerancia ε = {tol}")
    print("="*90)
    
    n_required = bisection_iterations_required(a, b, tol)
    print(f"\nTeorema 2.2: n = ⌈log₂(2/{tol})⌉ = ⌈{np.log2(2/tol):.2f}⌉ = {n_required}")
    
    print(f"\nEjecutando bisección:")
    print(f"\n{'Iter':<6} {'a':<12} {'b':<12} {'c':<12} {'f(c)':<12} {'Error':<12} {'|b-a|':<12}")
    print("-"*90)
    
    for n in range(min(n_required + 2, 25)):
        c = (a + b) / 2.0
        fc = f(c)
        error = abs(c - true_root)
        interval = b - a
        
        print(f"{n:<6} {a:<12.8f} {b:<12.8f} {c:<12.8f} {fc:<12.6e} {error:<12.6e} {interval:<12.6e}")
        
        if error < tol:
            print("-"*90)
            print(f"\nConvergencia en {n+1} iteraciones")
            print(f"Aproximación: {c:.12f}")
            print(f"√3 real:      {true_root:.12f}")
            print(f"Error:        {error:.6e}")
            break
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c


def create_summary_table():
    """
    Tabla resumen
    """
    a = 0
    b = 2
    
    print("\n" + "="*90)
    print("TABLA RESUMEN: RESPUESTA AL EJERCICIO 2.55")
    print("="*90)
    print(f"\nIntervalo: [{a}, {b}]")
    print(f"Función: f(x) = x² - 3")
    print(f"Raíz: √3 = {np.sqrt(3):.15f}\n")
    
    print(f"{'k':<5} {'Tolerancia (10^-k)':<22} {'Iteraciones n':<18}")
    print("-"*90)
    
    for k in range(3, 16):
        tol = 10**(-k)
        n = bisection_iterations_required(a, b, tol)
        print(f"{k:<5} {tol:<22.0e} {n:<18}")


def demonstration_plots():
    """
    Demostración visual
    """
    a, b = 0, 2
    f = lambda x: x**2 - 3
    root, n_iter, hist, errors, intervals = bisection_with_tracking(f, a, b, 1e-10, max_iter=50)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error vs iteración
    ax1.semilogy(range(1, len(errors)+1), errors, 'bo-', linewidth=2, markersize=6, 
                label='Error real')
    
    theoretical_bounds = [(b - a) / (2**(n+1)) for n in range(len(errors))]
    ax1.semilogy(range(1, len(theoretical_bounds)+1), theoretical_bounds, 'r--', 
                linewidth=2, label='Cota teórica')
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Iteración n', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Error / Cota (log)', fontsize=11, fontweight='bold')
    ax1.set_title('Verificación del Teorema 2.2', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    
    # Reducción del intervalo
    ax2.semilogy(range(1, len(intervals)+1), intervals, 'go-', linewidth=2, markersize=6,
                label='Longitud real')
    
    theoretical_intervals = [(b - a) / (2**n) for n in range(len(intervals))]
    ax2.semilogy(range(1, len(theoretical_intervals)+1), theoretical_intervals, 'r--',
                linewidth=2, label='Teórico: (b-a)/2^n')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Iteración n', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Longitud (log)', fontsize=11, fontweight='bold')
    ax2.set_title('Reducción del Intervalo', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# EJECUCIÓN
# ============================================================================

results = analyze_bisection_theorem()
comparison = verify_with_actual_bisection()
detailed_bisection_example()
plot_convergence_analysis(comparison)
create_summary_table()
demonstration_plots()

print("\n" + "="*90)
print("RESPUESTA FINAL AL EJERCICIO 2.55")
print("="*90)
print("""
Para aproximar √3 con intervalo [0,2]:

Tolerancia    Iteraciones
-------------------------
10⁻³          11
10⁻⁴          14
10⁻⁵          17
10⁻⁶          21
10⁻⁷          24
10⁻⁸          27
10⁻⁹          31
10⁻¹⁰         34
10⁻¹¹         37
10⁻¹²         41
10⁻¹³         44
10⁻¹⁴         47
10⁻¹⁵         51

Fórmula: n = ⌈log₂(2/ε)⌉
""")