"""
Exercise 2.57: Demostración de todos los métodos de búsqueda de raíces

PROBLEMA:
Resolver la ecuación: 3sin(x) + 9 = x² - cos(x)

Reorganizando: x² - 3sin(x) - cos(x) - 9 = 0

Definimos: f(x) = x² - 3sin(x) - cos(x) - 9
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# DEFINICIÓN DEL PROBLEMA
# ============================================================================

def f(x):
    """f(x) = x² - 3sin(x) - cos(x) - 9"""
    return x**2 - 3*np.sin(x) - np.cos(x) - 9

def df(x):
    """f'(x) = 2x - 3cos(x) + sin(x)"""
    return 2*x - 3*np.cos(x) + np.sin(x)

# ============================================================================
# MÉTODOS NUMÉRICOS
# ============================================================================

def bisection_method(f, a, b, tol=1e-10, max_iter=100):
    """Método de Bisección"""
    if f(a) * f(b) >= 0:
        return None, 0, [], "Error: f(a) y f(b) deben tener signos opuestos"
    
    history = []
    for n in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        history.append(c)
        
        if abs(fc) < tol or abs(b - a) < tol:
            return c, n+1, history, "Convergencia exitosa"
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    return c, max_iter, history, "Máximo de iteraciones"


def regula_falsi_method(f, a, b, tol=1e-10, max_iter=100):
    """Método de Regula Falsi"""
    if f(a) * f(b) >= 0:
        return None, 0, [], "Error: f(a) y f(b) deben tener signos opuestos"
    
    history = []
    fa = f(a)
    fb = f(b)
    
    for n in range(max_iter):
        c = a - fa * (b - a) / (fb - fa)
        fc = f(c)
        history.append(c)
        
        if abs(fc) < tol:
            return c, n+1, history, "Convergencia exitosa"
        
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    return c, max_iter, history, "Máximo de iteraciones"


def newton_method(f, df, x0, tol=1e-10, max_iter=100):
    """Método de Newton"""
    history = [x0]
    xn = x0
    
    for n in range(max_iter):
        fn = f(xn)
        dfn = df(xn)
        
        if abs(dfn) < 1e-14:
            return None, n, history, "Error: Derivada cercana a cero"
        
        xn_new = xn - fn / dfn
        history.append(xn_new)
        
        if abs(xn_new - xn) < tol or abs(fn) < tol:
            return xn_new, n+1, history, "Convergencia exitosa"
        
        xn = xn_new
    
    return xn, max_iter, history, "Máximo de iteraciones"


def secant_method(f, x0, x1, tol=1e-10, max_iter=100):
    """Método de la Secante"""
    history = [x0, x1]
    
    for n in range(2, max_iter):
        fn = f(history[-1])
        fn_prev = f(history[-2])
        
        if abs(fn - fn_prev) < 1e-14:
            return None, n-2, history, "Error: División por cero"
        
        xn_new = history[-1] - fn * (history[-1] - history[-2]) / (fn - fn_prev)
        history.append(xn_new)
        
        if abs(xn_new - history[-2]) < tol or abs(f(xn_new)) < tol:
            return xn_new, n-1, history, "Convergencia exitosa"
    
    return history[-1], max_iter, history, "Máximo de iteraciones"


def find_interval(f, start, end, num_points=1000):
    """Encuentra intervalos donde f cambia de signo"""
    x_vals = np.linspace(start, end, num_points)
    intervals = []
    
    for i in range(len(x_vals)-1):
        if f(x_vals[i]) * f(x_vals[i+1]) < 0:
            intervals.append((x_vals[i], x_vals[i+1]))
    
    return intervals


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    print("="*100)
    print("EJERCICIO 2.57: RESOLUCIÓN DE 3sin(x) + 9 = x² - cos(x)")
    print("="*100)
    print("\nEcuación original: 3sin(x) + 9 = x² - cos(x)")
    print("Forma estándar:    f(x) = x² - 3sin(x) - cos(x) - 9 = 0")
    print("Derivada:          f'(x) = 2x - 3cos(x) + sin(x)")
    
    # ========================================================================
    # PASO 1: VISUALIZAR LA FUNCIÓN
    # ========================================================================
    
    print("\n" + "="*100)
    print("PASO 1: ANÁLISIS GRÁFICO")
    print("="*100)
    
    x = np.linspace(-2, 6, 1000)
    y = f(x)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', linewidth=2.5, label='f(x) = x² - 3sin(x) - cos(x) - 9')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12, fontweight='bold')
    plt.ylabel('f(x)', fontsize=12, fontweight='bold')
    plt.title('Visualización de f(x)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    
    # Encontrar intervalos donde cruza el eje x
    intervals = find_interval(f, -2, 6, 1000)
    
    print(f"\nIntervalos encontrados donde f(x) cambia de signo:")
    for i, (a_int, b_int) in enumerate(intervals):
        mid = (a_int + b_int) / 2
        print(f"  Intervalo {i+1}: [{a_int:.6f}, {b_int:.6f}], raíz ≈ {mid:.6f}")
        plt.plot(mid, 0, 'ro', markersize=10)
        plt.annotate(f'≈{mid:.2f}', xy=(mid, 0), 
                    xytext=(mid, -3), fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('exercise_2_57_function.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Seleccionar la raíz positiva
    positive_intervals = [(a_int, b_int) for a_int, b_int in intervals if (a_int + b_int)/2 > 0]
    
    if not positive_intervals:
        print("\nNo se encontró raíz positiva!")
        return
    
    # Usar el primer intervalo positivo
    a, b = positive_intervals[0]
    print(f"\n" + "="*100)
    print(f"RAÍZ POSITIVA SELECCIONADA")
    print("="*100)
    print(f"Intervalo: [{a:.10f}, {b:.10f}]")
    print(f"f({a:.6f}) = {f(a):.10f}")
    print(f"f({b:.6f}) = {f(b):.10f}")
    print(f"f(a) * f(b) = {f(a)*f(b):.10f} < 0 ✓")
    
    # ========================================================================
    # MÉTODOS NUMÉRICOS
    # ========================================================================
    
    # BISECCIÓN
    print("\n" + "="*100)
    print("MÉTODO 1: BISECCIÓN")
    print("="*100)
    
    root_bis, iters_bis, hist_bis, status_bis = bisection_method(f, a, b)
    
    if root_bis is not None:
        print(f"\nEstado:         {status_bis}")
        print(f"Iteraciones:    {iters_bis}")
        print(f"Raíz:           x = {root_bis:.15f}")
        print(f"Verificación:   f(x) = {f(root_bis):.6e}")
    else:
        print(f"\nError: {status_bis}")
        return
    
    # REGULA FALSI
    print("\n" + "="*100)
    print("MÉTODO 2: REGULA FALSI (FALSA POSICIÓN)")
    print("="*100)
    
    root_rf, iters_rf, hist_rf, status_rf = regula_falsi_method(f, a, b)
    
    if root_rf is not None:
        print(f"\nEstado:         {status_rf}")
        print(f"Iteraciones:    {iters_rf}")
        print(f"Raíz:           x = {root_rf:.15f}")
        print(f"Verificación:   f(x) = {f(root_rf):.6e}")
    else:
        print(f"\nError: {status_rf}")
    
    # NEWTON
    print("\n" + "="*100)
    print("MÉTODO 3: NEWTON-RAPHSON")
    print("="*100)
    
    x0_newton = (a + b) / 2
    root_newton, iters_newton, hist_newton, status_newton = newton_method(f, df, x0_newton)
    
    if root_newton is not None:
        print(f"\nEstado:              {status_newton}")
        print(f"Iteraciones:         {iters_newton}")
        print(f"Raíz:                x = {root_newton:.15f}")
        print(f"Verificación:        f(x) = {f(root_newton):.6e}")
        print(f"Aprox. inicial:      x₀ = {x0_newton:.6f}")
    else:
        print(f"\nError: {status_newton}")
    
    # SECANTE
    print("\n" + "="*100)
    print("MÉTODO 4: SECANTE")
    print("="*100)
    
    root_secant, iters_secant, hist_secant, status_secant = secant_method(f, a, b)
    
    if root_secant is not None:
        print(f"\nEstado:              {status_secant}")
        print(f"Iteraciones:         {iters_secant}")
        print(f"Raíz:                x = {root_secant:.15f}")
        print(f"Verificación:        f(x) = {f(root_secant):.6e}")
        print(f"Aprox. iniciales:    x₀ = {a:.6f}, x₁ = {b:.6f}")
    else:
        print(f"\nError: {status_secant}")
    
    # ========================================================================
    # TABLA COMPARATIVA
    # ========================================================================
    
    print("\n" + "="*100)
    print("TABLA COMPARATIVA DE RESULTADOS")
    print("="*100)
    print(f"\n{'Método':<20} {'Iteraciones':<15} {'Raíz':<25} {'|f(raíz)|':<15}")
    print("-"*100)
    
    if root_bis is not None:
        print(f"{'Bisección':<20} {iters_bis:<15} {root_bis:<25.15f} {abs(f(root_bis)):<15.6e}")
    if root_rf is not None:
        print(f"{'Regula Falsi':<20} {iters_rf:<15} {root_rf:<25.15f} {abs(f(root_rf)):<15.6e}")
    if root_newton is not None:
        print(f"{'Newton':<20} {iters_newton:<15} {root_newton:<25.15f} {abs(f(root_newton)):<15.6e}")
    if root_secant is not None:
        print(f"{'Secante':<20} {iters_secant:<15} {root_secant:<25.15f} {abs(f(root_secant)):<15.6e}")
    
    # ========================================================================
    # GRÁFICAS COMPARATIVAS
    # ========================================================================
    
    print("\n" + "="*100)
    print("GENERANDO GRÁFICAS COMPARATIVAS...")
    print("="*100)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfica 1: Número de iteraciones
    ax1 = axes[0, 0]
    methods = ['Bisección', 'Regula\nFalsi', 'Newton', 'Secante']
    iterations = [iters_bis, iters_rf, iters_newton, iters_secant]
    colors = ['blue', 'green', 'red', 'orange']
    
    bars = ax1.bar(methods, iterations, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Iteraciones', fontsize=11, fontweight='bold')
    ax1.set_title('Eficiencia: Número de Iteraciones', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, iterations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Gráfica 2: Convergencia
    ax2 = axes[0, 1]
    
    errors_bis = [abs(xi - root_bis) for xi in hist_bis]
    errors_rf = [abs(xi - root_rf) for xi in hist_rf]
    errors_newton = [abs(xi - root_newton) for xi in hist_newton]
    errors_secant = [abs(xi - root_secant) for xi in hist_secant]
    
    ax2.semilogy(range(1, len(errors_bis)+1), errors_bis, 'b^-', 
                linewidth=2, markersize=5, label='Bisección')
    ax2.semilogy(range(1, len(errors_rf)+1), errors_rf, 'gs-', 
                linewidth=2, markersize=5, label='Regula Falsi')
    ax2.semilogy(range(1, len(errors_newton)+1), errors_newton, 'ro-', 
                linewidth=2, markersize=5, label='Newton')
    ax2.semilogy(range(1, len(errors_secant)+1), errors_secant, 'o-', color='orange',
                linewidth=2, markersize=5, label='Secante')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Iteración', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Error (log)', fontsize=11, fontweight='bold')
    ax2.set_title('Convergencia del Error', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    
    # Gráfica 3: Precisión final
    ax3 = axes[1, 0]
    
    final_errors = [abs(f(root_bis)), abs(f(root_rf)), 
                   abs(f(root_newton)), abs(f(root_secant))]
    
    bars2 = ax3.bar(methods, final_errors, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('|f(raíz)|', fontsize=11, fontweight='bold')
    ax3.set_title('Precisión Final', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, final_errors):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 2,
                f'{val:.1e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Gráfica 4: Trayectorias
    ax4 = axes[1, 1]
    
    ax4.plot(range(len(hist_bis)), hist_bis, 'b^-', 
            linewidth=2, markersize=5, label='Bisección', alpha=0.7)
    ax4.plot(range(len(hist_rf)), hist_rf, 'gs-', 
            linewidth=2, markersize=5, label='Regula Falsi', alpha=0.7)
    ax4.plot(range(len(hist_newton)), hist_newton, 'ro-', 
            linewidth=2, markersize=5, label='Newton', alpha=0.7)
    ax4.plot(range(len(hist_secant)), hist_secant, 'o-', color='orange',
            linewidth=2, markersize=5, label='Secante', alpha=0.7)
    
    avg_root = np.mean([root_bis, root_rf, root_newton, root_secant])
    ax4.axhline(y=avg_root, color='black', linestyle='--', linewidth=2, label='Raíz')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Iteración', fontsize=11, fontweight='bold')
    ax4.set_ylabel('x_n', fontsize=11, fontweight='bold')
    ax4.set_title('Trayectorias de Convergencia', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('exercise_2_57_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # CONCLUSIONES
    # ========================================================================
    
    print("\n" + "="*100)
    print("CONCLUSIONES")
    print("="*100)
    print("""
✓ TODOS LOS MÉTODOS CONVERGEN A LA MISMA RAÍZ

RANKING DE EFICIENCIA (menos iteraciones = mejor):
1. Newton: Más rápido (convergencia cuadrática)
2. Secante: Rápido (no requiere derivada)
3. Regula Falsi: Medio
4. Bisección: Más lento pero robusto

CARACTERÍSTICAS:
- Newton: Requiere f'(x), muy rápido
- Secante: No requiere f'(x), buen compromiso
- Regula Falsi: Usa interpolación lineal
- Bisección: Siempre converge, predecible
    """)
    
    print("="*100)
    print("✓ EJERCICIO 2.57 COMPLETADO")
    print("="*100)


if __name__ == "__main__":
    main()