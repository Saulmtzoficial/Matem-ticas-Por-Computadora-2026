"""
Exercise 2.54: Secant Method

PROBLEMA:
Sea f(x) una función continua con una raíz cerca de x=x0.
Dar todos los detalles matemáticos de cómo el Método de la Secante
aproxima la raíz de la función f(x).

MÉTODO DE LA SECANTE:

FÓRMULA: x_(n+1) = x_n - f(x_n) * (x_n - x_(n-1)) / (f(x_n) - f(x_(n-1)))

GEOMETRÍA:
- Similar a Newton pero NO requiere derivada
- Usa una secante (línea recta entre dos puntos) en lugar de tangente
- Aproxima f'(x_n) ≈ (f(x_n) - f(x_(n-1))) / (x_n - x_(n-1))

RELACIÓN CON NEWTON:
- Newton: x_(n+1) = x_n - f(x_n)/f'(x_n)
- Secante: Reemplaza f'(x_n) con diferencia finita

CONVERGENCIA:
- Orden: p ≈ 1.618 (número áureo φ) - superlineal
- Más lenta que Newton (p=2) pero más rápida que bisección (p=1)
- Fórmula: |e_(n+1)| ≈ C|e_n|^1.618

VENTAJAS:
- NO requiere calcular derivada
- Solo necesita evaluar f(x)
- Convergencia superlineal

DESVENTAJAS:
- Requiere DOS puntos iniciales
- Más lenta que Newton
- Puede fallar si f(x_n) ≈ f(x_(n-1))
"""

import numpy as np
import matplotlib.pyplot as plt

def secant_method(f, x0, x1, tol=1e-8, max_iter=50, verbose=True):
    """
    Método de la Secante
    """
    history = [x0, x1]
    f_values = [f(x0), f(x1)]
    
    if verbose:
        print("="*90)
        print("MÉTODO DE LA SECANTE")
        print("="*90)
        print(f"Aproximaciones iniciales: x0 = {x0}, x1 = {x1}")
        print(f"Tolerancia: {tol}\n")
        print(f"{'Iter':<6} {'xn':<15} {'f(xn)':<15} {'|cambio|':<15}")
        print("-"*90)
    
    for n in range(2, max_iter):
        fn = f_values[-1]
        fn_prev = f_values[-2]
        xn = history[-1]
        xn_prev = history[-2]
        
        # Verificar denominador
        if abs(fn - fn_prev) < 1e-14:
            if verbose:
                print(f"\nERROR: f(x_n) ≈ f(x_(n-1)). División por cero.")
            return None, n, history, f_values
        
        # Fórmula de la secante
        xn_new = xn - fn * (xn - xn_prev) / (fn - fn_prev)
        fn_new = f(xn_new)
        
        history.append(xn_new)
        f_values.append(fn_new)
        
        change = abs(xn_new - xn)
        
        if verbose:
            print(f"{n:<6} {xn:<15.10f} {fn:<15.6e} {change:<15.6e}")
        
        if change < tol or abs(fn_new) < tol:
            if verbose:
                print("-"*90)
                print(f"\nConvergencia en {n} iteraciones!")
                print(f"Raíz: x = {xn_new:.12f}")
                print(f"|f(x)| = {abs(fn_new):.6e}")
            
            return xn_new, n, history, f_values
        
    if verbose:
        print(f"\nMáximo de iteraciones alcanzado ({max_iter})")
    
    return history[-1], max_iter, history, f_values


def plot_secant(f, history, f_values, true_root=None):
    """
    Visualización del Método de la Secante
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rango para graficar
    all_x = history + ([true_root] if true_root else [])
    x_min = min(all_x) - 0.5
    x_max = max(all_x) + 0.5
    
    x = np.linspace(x_min, x_max, 1000)
    y = [f(xi) for xi in x]
    
    # Gráfica 1: Geometría - Secantes
    ax1 = axes[0, 0]
    ax1.plot(x, y, 'b-', linewidth=2, label='f(x)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    colors = ['red', 'green', 'orange', 'purple', 'brown']
    num_show = min(5, len(history) - 2)
    
    for i in range(num_show):
        xi = history[i]
        xi_next = history[i + 1]
        fi = f_values[i]
        fi_next = f_values[i + 1]
        
        # Puntos
        ax1.plot(xi, fi, 'o', color=colors[i], markersize=8, label=f'x{i}={xi:.3f}')
        ax1.plot(xi_next, fi_next, 's', color=colors[i], markersize=8)
        
        # Línea secante
        x_sec = np.array([xi - 0.5, xi_next + 0.5])
        slope = (fi_next - fi) / (xi_next - xi)
        y_sec = fi + slope * (x_sec - xi)
        ax1.plot(x_sec, y_sec, '--', color=colors[i], linewidth=1.5, alpha=0.7)
        
        # Próxima aproximación
        if i + 2 < len(history):
            xi_new = history[i + 2]
            ax1.plot(xi_new, 0, 'X', color=colors[i], markersize=10)
    
    if true_root:
        ax1.axvline(x=true_root, color='darkred', linestyle='--', 
                   linewidth=2, label=f'Raíz={true_root:.4f}')
    
    ax1.set_xlabel('x', fontsize=11, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=11, fontweight='bold')
    ax1.set_title('Geometría: Secantes Sucesivas', fontsize=12, fontweight='bold')
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
    
    # Gráfica 3: Error
    ax3 = axes[1, 0]
    
    if true_root:
        errors = [abs(xi - true_root) for xi in history if abs(xi - true_root) > 1e-16]
        if errors:
            ax3.semilogy(range(len(errors)), errors, 'mo-', linewidth=2, markersize=6)
            ax3.set_ylabel('Error |x_n - raíz| (log)', fontsize=11, fontweight='bold')
            ax3.set_title('Error Absoluto (Orden ≈ 1.618)', fontsize=12, fontweight='bold')
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


def compare_methods(f, df, a, b, x0, x1, true_root):
    """
    Compara Secante, Newton y Bisección
    """
    print("\n" + "="*90)
    print("COMPARACIÓN DE MÉTODOS")
    print("="*90)
    
    # Secante
    root_secant, iters_secant, hist_secant, f_secant = secant_method(
        f, x0, x1, tol=1e-10, verbose=False)
    error_secant = abs(root_secant - true_root) if root_secant else float('inf')
    print(f"\nSecante:    {iters_secant} iteraciones, error = {error_secant:.2e}")
    
    # Newton (si tenemos derivada)
    if df:
        root_newton, hist_newton = newton_simple(f, df, (x0+x1)/2, tol=1e-10)
        iters_newton = len(hist_newton) - 1
        error_newton = abs(root_newton - true_root)
        print(f"Newton:     {iters_newton} iteraciones, error = {error_newton:.2e}")
    
    # Bisección
    from scipy.optimize import bisect
    root_bisect = bisect(f, a, b, xtol=1e-10)
    iters_bisect = int(np.ceil(np.log2((b - a) / 1e-10)))
    error_bisect = abs(root_bisect - true_root)
    print(f"Bisección:  {iters_bisect} iteraciones, error = {error_bisect:.2e}")
    
    # Gráfica comparativa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if df:
        methods = ['Bisección', 'Secante', 'Newton']
        iterations = [iters_bisect, iters_secant, iters_newton]
        colors = ['blue', 'green', 'red']
    else:
        methods = ['Bisección', 'Secante']
        iterations = [iters_bisect, iters_secant]
        colors = ['blue', 'green']
    
    bars = ax1.bar(methods, iterations, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Iteraciones', fontsize=11, fontweight='bold')
    ax1.set_title('Eficiencia', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, iterations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Convergencia
    errors_secant = [abs(x - true_root) for x in hist_secant]
    bisect_errors = [(b - a) / (2**(n+1)) for n in range(iters_bisect)]
    
    ax2.semilogy(range(1, len(errors_secant)+1), errors_secant, 
                'g-s', linewidth=2, markersize=5, label='Secante (p≈1.618)')
    ax2.semilogy(range(1, len(bisect_errors)+1), bisect_errors, 
                'b-^', linewidth=2, markersize=5, label='Bisección (p=1)')
    
    if df:
        errors_newton = [abs(x - true_root) for x in hist_newton]
        ax2.semilogy(range(1, len(errors_newton)+1), errors_newton, 
                    'r-o', linewidth=2, markersize=5, label='Newton (p=2)')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Iteración', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Error (log)', fontsize=11, fontweight='bold')
    ax2.set_title('Convergencia', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


def newton_simple(f, df, x0, tol=1e-10, max_iter=50):
    """Newton simple para comparación"""
    history = [x0]
    xn = x0
    for _ in range(max_iter):
        fn = f(xn)
        dfn = df(xn)
        if abs(dfn) < 1e-14 or abs(fn) < tol:
            break
        xn = xn - fn / dfn
        history.append(xn)
        if abs(f(xn)) < tol:
            break
    return xn, history


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
x1_1 = 2.0
true_root1 = np.sqrt(2)

root1, iters1, hist1, f1_vals = secant_method(f1, x0_1, x1_1, tol=1e-10)
plot_secant(f1, hist1, f1_vals, true_root1)


print("\n" + "="*90)
print("EJEMPLO 2: f(x) = x³ - 2x - 5")
print("="*90)

def f2(x):
    return x**3 - 2*x - 5

def df2(x):
    return 3*x**2 - 2

x0_2 = 2.0
x1_2 = 3.0
true_root2 = 2.0945514815423265

root2, iters2, hist2, f2_vals = secant_method(f2, x0_2, x1_2, tol=1e-10)
plot_secant(f2, hist2, f2_vals, true_root2)
compare_methods(f2, df2, 2.0, 3.0, x0_2, x1_2, true_root2)


print("\n" + "="*90)
print("EJEMPLO 3: f(x) = cos(x) - x")
print("="*90)

def f3(x):
    return np.cos(x) - x

def df3(x):
    return -np.sin(x) - 1

x0_3 = 0.5
x1_3 = 1.0
true_root3 = 0.7390851332151607

root3, iters3, hist3, f3_vals = secant_method(f3, x0_3, x1_3, tol=1e-10)
plot_secant(f3, hist3, f3_vals, true_root3)


print("\n" + "="*90)
print("EJEMPLO 4: f(x) = e^x - 3x²  (sin derivada analítica)")
print("="*90)

def f4(x):
    return np.exp(x) - 3*x**2

x0_4 = 0.5
x1_4 = 1.5
true_root4 = 0.9100075724111

root4, iters4, hist4, f4_vals = secant_method(f4, x0_4, x1_4, tol=1e-10)
plot_secant(f4, hist4, f4_vals, true_root4)
compare_methods(f4, None, 0.5, 1.5, x0_4, x1_4, true_root4)


print("\n" + "="*90)
print("RESUMEN")
print("="*90)
print("""
MÉTODO DE LA SECANTE: x_(n+1) = x_n - f(x_n)*(x_n - x_(n-1))/(f(x_n) - f(x_(n-1)))

✓ NO requiere derivada (solo evalúa f(x))
✓ Convergencia superlineal (orden ≈ 1.618)
✓ Más rápido que bisección
✗ Requiere DOS puntos iniciales
✗ Más lento que Newton (orden 2)
✗ Puede fallar si f(x_n) ≈ f(x_(n-1))

COMPARACIÓN:
- Bisección:  Orden 1 (lineal)     - Siempre converge
- Secante:    Orden 1.618 (φ)      - No requiere derivada
- Newton:     Orden 2 (cuadrática) - Más rápido, requiere f'(x)
""")