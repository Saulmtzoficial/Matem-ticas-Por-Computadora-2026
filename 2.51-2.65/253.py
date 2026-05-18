"""
Exercise 2.53: Newton's Method (Newton-Raphson Method)

PROBLEMA:
Sea f(x) una función diferenciable con una raíz cerca de x=x0.
Dar todos los detalles matemáticos de cómo el Método de Newton
aproxima la raíz de la función f(x).

MÉTODO DE NEWTON-RAPHSON:

FÓRMULA: x_(n+1) = x_n - f(x_n)/f'(x_n)

GEOMETRÍA:
- Usa la tangente a f(x) en x_n para aproximar la raíz
- La tangente cruza el eje x en x_(n+1)

CONVERGENCIA:
- Cuadrática: |e_(n+1)| ≈ M|e_n|²
- Los dígitos correctos se duplican en cada iteración

VENTAJAS:
- Convergencia muy rápida
- Pocas iteraciones necesarias

DESVENTAJAS:
- Requiere calcular f'(x)
- Sensible a x0 inicial
- Falla si f'(x) = 0
"""

import numpy as np
import matplotlib.pyplot as plt

def newton_method(f, df, x0, tol=1e-8, max_iter=50, verbose=True):
    """
    Método de Newton-Raphson
    """
    history = []
    f_values = []
    xn = x0
    
    if verbose:
        print("="*90)
        print("MÉTODO DE NEWTON-RAPHSON")
        print("="*90)
        print(f"Aproximación inicial: x0 = {x0}")
        print(f"Tolerancia: {tol}\n")
        print(f"{'Iter':<6} {'xn':<15} {'f(xn)':<15} {'df(xn)':<15} {'|cambio|':<15}")
        print("-"*90)
    
    for n in range(max_iter):
        fn = f(xn)
        dfn = df(xn)
        
        history.append(xn)
        f_values.append(fn)
        
        if abs(dfn) < 1e-14:
            if verbose:
                print(f"\nERROR: f'(x_{n}) ≈ 0. Tangente horizontal.")
            return None, n, history, f_values
        
        xn_new = xn - fn / dfn
        change = abs(xn_new - xn)
        
        if verbose:
            print(f"{n:<6} {xn:<15.10f} {fn:<15.6e} {dfn:<15.6e} {change:<15.6e}")
        
        if change < tol or abs(fn) < tol:
            history.append(xn_new)
            f_values.append(f(xn_new))
            
            if verbose:
                print("-"*90)
                print(f"\nConvergencia en {n+1} iteraciones!")
                print(f"Raíz: x = {xn_new:.12f}")
                print(f"|f(x)| = {abs(f(xn_new)):.6e}")
            
            return xn_new, n+1, history, f_values
        
        xn = xn_new
    
    if verbose:
        print(f"\nMáximo de iteraciones alcanzado ({max_iter})")
    
    return xn, max_iter, history, f_values


def plot_newton(f, df, history, f_values, true_root=None):
    """
    Visualización del Método de Newton
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rango para graficar
    all_x = history + ([true_root] if true_root else [])
    x_min = min(all_x) - 0.5
    x_max = max(all_x) + 0.5
    
    x = np.linspace(x_min, x_max, 1000)
    y = [f(xi) for xi in x]
    
    # Gráfica 1: Geometría - Tangentes
    ax1 = axes[0, 0]
    ax1.plot(x, y, 'b-', linewidth=2, label='f(x)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    num_show = min(5, len(history) - 1)
    
    for i in range(num_show):
        xi = history[i]
        fi = f_values[i]
        dfi = df(xi)
        
        ax1.plot(xi, fi, 'o', color=colors[i], markersize=8, label=f'x{i}={xi:.3f}')
        
        # Tangente
        x_tang = np.array([xi - 0.8, xi + 0.8])
        y_tang = fi + dfi * (x_tang - xi)
        ax1.plot(x_tang, y_tang, '--', color=colors[i], linewidth=1.5, alpha=0.7)
        
        if i < len(history) - 1:
            xi_next = history[i + 1]
            ax1.plot(xi_next, 0, 'X', color=colors[i], markersize=10)
    
    if true_root:
        ax1.axvline(x=true_root, color='darkred', linestyle='--', 
                   linewidth=2, label=f'Raíz={true_root:.4f}')
    
    ax1.set_xlabel('x', fontsize=11, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=11, fontweight='bold')
    ax1.set_title('Geometría: Tangentes Sucesivas', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    
    # Gráfica 2: Convergencia
    ax2 = axes[0, 1]
    iterations = range(len(history))
    ax2.plot(iterations, history, 'go-', linewidth=2, markersize=6)
    
    if true_root:
        ax2.axhline(y=true_root, color='r', linestyle='--', linewidth=2)
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Iteración', fontsize=11, fontweight='bold')
    ax2.set_ylabel('x_n', fontsize=11, fontweight='bold')
    ax2.set_title('Convergencia de Aproximaciones', fontsize=12, fontweight='bold')
    
    # Gráfica 3: Error (si conocemos la raíz)
    ax3 = axes[1, 0]
    
    if true_root:
        errors = [abs(xi - true_root) for xi in history if abs(xi - true_root) > 1e-16]
        if errors:
            ax3.semilogy(range(len(errors)), errors, 'mo-', linewidth=2, markersize=6)
            ax3.set_ylabel('Error |x_n - raíz| (log)', fontsize=11, fontweight='bold')
            ax3.set_title('Error Absoluto (Convergencia Cuadrática)', fontsize=12, fontweight='bold')
    else:
        f_abs = [abs(fi) for fi in f_values]
        ax3.semilogy(range(len(f_abs)), f_abs, 'co-', linewidth=2, markersize=6)
        ax3.set_ylabel('|f(x_n)| (log)', fontsize=11, fontweight='bold')
        ax3.set_title('|f(x_n)| vs Iteración', fontsize=12, fontweight='bold')
    
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Iteración', fontsize=11, fontweight='bold')
    
    # Gráfica 4: Cambios sucesivos
    ax4 = axes[1, 1]
    changes = [abs(history[i+1] - history[i]) for i in range(len(history)-1)]
    ax4.semilogy(range(1, len(changes)+1), changes, 'rs-', linewidth=2, markersize=6)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Iteración', fontsize=11, fontweight='bold')
    ax4.set_ylabel('|x_(n+1) - x_n| (log)', fontsize=11, fontweight='bold')
    ax4.set_title('Cambio entre Iteraciones', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def compare_methods(f, df, a, b, x0, true_root):
    """
    Compara Newton, Secante y Bisección
    """
    print("\n" + "="*90)
    print("COMPARACIÓN DE MÉTODOS")
    print("="*90)
    
    # Newton
    root_newton, iters_newton, hist_newton, f_newton = newton_method(
        f, df, x0, tol=1e-10, verbose=False)
    error_newton = abs(root_newton - true_root)
    print(f"\nNewton:     {iters_newton} iteraciones, error = {error_newton:.2e}")
    
    # Bisección
    from scipy.optimize import bisect
    root_bisect = bisect(f, a, b, xtol=1e-10)
    iters_bisect = int(np.ceil(np.log2((b - a) / 1e-10)))
    error_bisect = abs(root_bisect - true_root)
    print(f"Bisección:  {iters_bisect} iteraciones, error = {error_bisect:.2e}")
    
    # Gráfica comparativa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['Bisección', 'Newton']
    iterations = [iters_bisect, iters_newton]
    colors = ['blue', 'red']
    
    bars = ax1.bar(methods, iterations, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Iteraciones', fontsize=11, fontweight='bold')
    ax1.set_title('Eficiencia', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, iterations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Convergencia
    errors_newton = [abs(x - true_root) for x in hist_newton]
    bisect_errors = [(b - a) / (2**(n+1)) for n in range(iters_bisect)]
    
    ax2.semilogy(range(1, len(errors_newton)+1), errors_newton, 
                'r-o', linewidth=2, markersize=5, label='Newton (cuadrática)')
    ax2.semilogy(range(1, len(bisect_errors)+1), bisect_errors, 
                'b-^', linewidth=2, markersize=5, label='Bisección (lineal)')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Iteración', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Error (log)', fontsize=11, fontweight='bold')
    ax2.set_title('Convergencia', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# EJEMPLOS
# ============================================================================

print("\n" + "="*90)
print("EJEMPLO 1: f(x) = x² - 2  (encontrar √2)")
print("="*90)

def f1(x):
    return x**2 - 2

def df1(x):
    return 2*x

x0_1 = 1.0
true_root1 = np.sqrt(2)

root1, iters1, hist1, f1_vals = newton_method(f1, df1, x0_1, tol=1e-10)
plot_newton(f1, df1, hist1, f1_vals, true_root1)


print("\n" + "="*90)
print("EJEMPLO 2: f(x) = x³ - 2x - 5")
print("="*90)

def f2(x):
    return x**3 - 2*x - 5

def df2(x):
    return 3*x**2 - 2

x0_2 = 2.5
true_root2 = 2.0945514815423265

root2, iters2, hist2, f2_vals = newton_method(f2, df2, x0_2, tol=1e-10)
plot_newton(f2, df2, hist2, f2_vals, true_root2)
compare_methods(f2, df2, 2.0, 3.0, x0_2, true_root2)


print("\n" + "="*90)
print("EJEMPLO 3: f(x) = cos(x) - x")
print("="*90)

def f3(x):
    return np.cos(x) - x

def df3(x):
    return -np.sin(x) - 1

x0_3 = 1.0
true_root3 = 0.7390851332151607

root3, iters3, hist3, f3_vals = newton_method(f3, df3, x0_3, tol=1e-10)
plot_newton(f3, df3, hist3, f3_vals, true_root3)


print("\n" + "="*90)
print("RESUMEN")
print("="*90)
print("""
MÉTODO DE NEWTON: x_(n+1) = x_n - f(x_n)/f'(x_n)

✓ Convergencia CUADRÁTICA (muy rápida)
✓ Pocas iteraciones
✗ Requiere f'(x)
✗ Sensible a x0
✗ Falla si f'(x) = 0
""")