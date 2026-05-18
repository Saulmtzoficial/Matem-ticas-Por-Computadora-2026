"""
Exercise 2.63: Finding Local Extrema using Root-Finding Methods

PROBLEMA:
Encontrar los extremos locales de f(x) = x³(x-3)(x-6)⁴
usando técnicas numéricas.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# DEFINICIÓN DE LA FUNCIÓN
# ============================================================================

def f(x):
    """f(x) = x³(x-3)(x-6)⁴"""
    return x**3 * (x - 3) * (x - 6)**4

def df(x):
    """Primera derivada (analítica)"""
    term1 = 3 * x**2 * (x - 3) * (x - 6)**4
    term2 = x**3 * (x - 6)**4
    term3 = x**3 * (x - 3) * 4 * (x - 6)**3
    return term1 + term2 + term3

def ddf(x):
    """Segunda derivada (numérica)"""
    h = 1e-6
    return (df(x + h) - df(x - h)) / (2 * h)

# ============================================================================
# MÉTODO DE NEWTON
# ============================================================================

def newton_for_derivative(f_prime, f_double_prime, x0, tol=1e-10, max_iter=100):
    """Newton para encontrar raíces de f'(x)"""
    xn = x0
    for _ in range(max_iter):
        fn = f_prime(xn)
        dfn = f_double_prime(xn)
        if abs(dfn) < 1e-14 or abs(fn) < tol:
            return xn
        xn = xn - fn / dfn
    return xn

# ============================================================================
# ENCONTRAR EXTREMOS
# ============================================================================

def find_extrema():
    print("="*90)
    print("EJERCICIO 2.63: EXTREMOS LOCALES DE f(x) = x³(x-3)(x-6)⁴")
    print("="*90)
    
    print("\nMÉTODO:")
    print("  1. Extremos locales → f'(x) = 0")
    print("  2. Usar Newton para resolver f'(x) = 0")
    print("  3. Clasificar con f''(x)")
    
    # Buscar raíces de f'(x)
    initial_guesses = [-1, 0.5, 2, 4, 5, 7]
    extrema = []
    
    print("\n" + "="*90)
    print("BÚSQUEDA DE PUNTOS CRÍTICOS")
    print("="*90)
    
    for x0 in initial_guesses:
        x_crit = newton_for_derivative(df, ddf, x0)
        
        if abs(df(x_crit)) < 1e-6:
            is_duplicate = any(abs(x_crit - e['x']) < 0.01 for e in extrema)
            if not is_duplicate:
                f_val = f(x_crit)
                f_double_val = ddf(x_crit)
                
                if abs(f_double_val) < 1e-6:
                    extremum_type = "Punto de inflexión"
                elif f_double_val > 0:
                    extremum_type = "MÍNIMO LOCAL"
                else:
                    extremum_type = "MÁXIMO LOCAL"
                
                extrema.append({
                    'x': x_crit,
                    'f': f_val,
                    'f_prime': df(x_crit),
                    'f_double': f_double_val,
                    'type': extremum_type
                })
    
    extrema.sort(key=lambda e: e['x'])
    
    # Mostrar resultados
    header1 = "x"
    header2 = "f(x)"
    header3 = "f'(x)"
    header4 = "f''(x)"
    header5 = "Tipo"
    print(f"\n{header1:<15} {header2:<15} {header3:<15} {header4:<15} {header5:<20}")
    print("-"*90)
    
    for e in extrema:
        print(f"{e['x']:<15.6f} {e['f']:<15.6e} {e['f_prime']:<15.2e} {e['f_double']:<15.6f} {e['type']:<20}")
    
    return extrema

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def visualize_extrema(extrema):
    print("\n" + "="*90)
    print("GENERANDO GRÁFICAS...")
    print("="*90)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    x = np.linspace(-1, 7, 2000)
    y_f = f(x)
    y_df = df(x)
    y_ddf = [ddf(xi) for xi in x]
    
    # Gráfica 1: Función f(x)
    ax1 = axes[0, 0]
    ax1.plot(x, y_f, 'b-', linewidth=2, label='f(x)')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    for e in extrema:
        color = 'red' if 'MÍNIMO' in e['type'] else 'green' if 'MÁXIMO' in e['type'] else 'orange'
        marker = 'v' if 'MÍNIMO' in e['type'] else '^' if 'MÁXIMO' in e['type'] else 'o'
        ax1.plot(e['x'], e['f'], marker, color=color, markersize=12, 
                label=f"{e['type']} en x={e['x']:.2f}")
    
    ax1.set_xlabel('x', fontsize=11, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=11, fontweight='bold')
    ax1.set_title('f(x) = x³(x-3)(x-6)⁴', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_ylim([min(y_f)*1.1, max(y_f)*0.3])
    
    # Gráfica 2: Primera derivada
    ax2 = axes[0, 1]
    ax2.plot(x, y_df, 'g-', linewidth=2, label="f'(x)")
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    for e in extrema:
        ax2.plot(e['x'], 0, 'ro', markersize=10)
        ax2.axvline(x=e['x'], color='r', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('x', fontsize=11, fontweight='bold')
    ax2.set_ylabel("f'(x)", fontsize=11, fontweight='bold')
    ax2.set_title("Primera Derivada (raíces = puntos críticos)", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    
    # Gráfica 3: Segunda derivada
    ax3 = axes[1, 0]
    ax3.plot(x, y_ddf, 'm-', linewidth=2, label="f''(x)")
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.5)
    ax3.grid(True, alpha=0.3)
    
    for e in extrema:
        color = 'red' if e['f_double'] > 0 else 'green' if e['f_double'] < 0 else 'orange'
        ax3.plot(e['x'], e['f_double'], 'o', color=color, markersize=10)
        ax3.axvline(x=e['x'], color=color, linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('x', fontsize=11, fontweight='bold')
    ax3.set_ylabel("f''(x)", fontsize=11, fontweight='bold')
    ax3.set_title("Segunda Derivada (clasifica extremos)", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    
    # Gráfica 4: Zoom en extremos
    ax4 = axes[1, 1]
    
    for e in extrema:
        x_zoom = np.linspace(e['x']-0.5, e['x']+0.5, 200)
        y_zoom = f(x_zoom)
        color = 'red' if 'MÍNIMO' in e['type'] else 'green' if 'MÁXIMO' in e['type'] else 'orange'
        ax4.plot(x_zoom, y_zoom, linewidth=2, color=color, alpha=0.7)
        ax4.plot(e['x'], e['f'], 'o', color=color, markersize=10)
    
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('x', fontsize=11, fontweight='bold')
    ax4.set_ylabel('f(x)', fontsize=11, fontweight='bold')
    ax4.set_title('Zoom en Extremos Locales', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('exercise_2_63_extrema.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# RESUMEN
# ============================================================================

def print_summary(extrema):
    print("\n" + "="*90)
    print("RESUMEN: EXTREMOS LOCALES DE f(x) = x³(x-3)(x-6)⁴")
    print("="*90)
    
    minimos = [e for e in extrema if 'MÍNIMO' in e['type']]
    maximos = [e for e in extrema if 'MÁXIMO' in e['type']]
    
    print(f"\nMÍNIMOS LOCALES: {len(minimos)}")
    for m in minimos:
        print(f"  • x = {m['x']:.6f}, f(x) = {m['f']:.6e}")
    
    print(f"\nMÁXIMOS LOCALES: {len(maximos)}")
    for m in maximos:
        print(f"  • x = {m['x']:.6f}, f(x) = {m['f']:.6e}")
    
    print("\nMÉTODO USADO:")
    print("  ✓ Newton-Raphson para resolver f'(x) = 0")
    print("  ✓ Prueba de segunda derivada para clasificar")
    print("  ✓ f''(x) > 0 → mínimo local")
    print("  ✓ f''(x) < 0 → máximo local")
    
    print("\n" + "="*90)
    print("✓ EJERCICIO 2.63 COMPLETADO")
    print("="*90)

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    extrema = find_extrema()
    visualize_extrema(extrema)
    print_summary(extrema)