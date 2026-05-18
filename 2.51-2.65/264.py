"""
Exercise 2.64: Finding Fixed Points using Root-Finding Methods

PROBLEMA:
Un punto fijo de f(x) es un punto donde f(x) = x.

Encontrar puntos fijos es equivalente a resolver: f(x) - x = 0

Tareas:
1. Encontrar puntos fijos de f(x) = x² - 6 en [0, ∞)
2. Encontrar puntos fijos de f(x) = √(8x + 6)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# MÉTODO DE NEWTON
# ============================================================================

def newton_method(f, df, x0, tol=1e-10, max_iter=100):
    """Newton-Raphson para encontrar raíces"""
    xn = x0
    history = [x0]
    
    for _ in range(max_iter):
        fn = f(xn)
        dfn = df(xn)
        
        if abs(dfn) < 1e-14:
            return None, history
        
        xn_new = xn - fn / dfn
        history.append(xn_new)
        
        if abs(fn) < tol or abs(xn_new - xn) < tol:
            return xn_new, history
        
        xn = xn_new
    
    return xn, history

# ============================================================================
# PARTE 1: f(x) = x² - 6
# ============================================================================

def part_1():
    print("="*90)
    print("PARTE 1: PUNTOS FIJOS DE f(x) = x² - 6 EN [0, ∞)")
    print("="*90)
    
    print("\nTEORÍA:")
    print("  Punto fijo: f(x) = x")
    print("  Para f(x) = x² - 6:")
    print("    x² - 6 = x")
    print("    x² - x - 6 = 0")
    print("    (x - 3)(x + 2) = 0")
    print("    x = 3 o x = -2")
    print("\n  En [0, ∞): x = 3 es el único punto fijo")
    
    # Definir g(x) = f(x) - x para encontrar raíces
    def f_original(x):
        return x**2 - 6
    
    def g(x):
        """g(x) = f(x) - x = x² - x - 6"""
        return x**2 - x - 6
    
    def dg(x):
        """g'(x) = 2x - 1"""
        return 2*x - 1
    
    print("\n" + "="*90)
    print("MÉTODO NUMÉRICO: Newton-Raphson para g(x) = x² - x - 6")
    print("="*90)
    
    # Buscar desde diferentes x₀ en [0, ∞)
    initial_guesses = [0.5, 1, 2, 4, 5]
    fixed_points = []
    
    print(f"\n{'x₀':<10} {'Punto fijo':<15} {'f(x)':<15} {'Verificación':<15}")
    print("-"*60)
    
    for x0 in initial_guesses:
        root, history = newton_method(g, dg, x0)
        
        if root is not None and root >= 0:
            f_val = f_original(root)
            is_fixed = abs(f_val - root) < 1e-8
            
            # Evitar duplicados
            is_duplicate = any(abs(root - fp) < 0.01 for fp in fixed_points)
            if not is_duplicate:
                fixed_points.append(root)
                check = "✓ f(x) = x" if is_fixed else "✗"
                print(f"{x0:<10.1f} {root:<15.10f} {f_val:<15.10f} {check:<15}")
    
    print(f"\nPUNTO FIJO ENCONTRADO: x = {fixed_points[0]:.10f}")
    print(f"VERIFICACIÓN: f({fixed_points[0]:.6f}) = {f_original(fixed_points[0]):.6f}")
    
    # Visualización
    visualize_fixed_point_1(f_original, fixed_points)
    
    return fixed_points

def visualize_fixed_point_1(f, fixed_points):
    """Visualización para parte 1"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.linspace(-1, 5, 1000)
    y_f = [f(xi) for xi in x]
    y_identity = x
    
    # Gráfica 1: f(x) vs x
    ax1 = axes[0]
    ax1.plot(x, y_f, 'b-', linewidth=2, label='f(x) = x² - 6')
    ax1.plot(x, y_identity, 'r--', linewidth=2, label='y = x')
    ax1.grid(True, alpha=0.3)
    
    for fp in fixed_points:
        if fp >= 0:
            ax1.plot(fp, fp, 'go', markersize=12, label=f'Punto fijo: x={fp:.2f}')
            ax1.plot([fp, fp], [0, fp], 'g:', alpha=0.5)
            ax1.plot([0, fp], [fp, fp], 'g:', alpha=0.5)
    
    ax1.set_xlabel('x', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y', fontsize=11, fontweight='bold')
    ax1.set_title('Punto Fijo: Intersección de f(x) y y=x', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim([-1, 5])
    ax1.set_ylim([-7, 10])
    
    # Gráfica 2: g(x) = f(x) - x
    ax2 = axes[1]
    g = [f(xi) - xi for xi in x]
    ax2.plot(x, g, 'b-', linewidth=2, label='g(x) = x² - x - 6')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3)
    
    for fp in fixed_points:
        if fp >= 0:
            ax2.plot(fp, 0, 'ro', markersize=12, label=f'Raíz: x={fp:.2f}')
    
    ax2.set_xlabel('x', fontsize=11, fontweight='bold')
    ax2.set_ylabel('g(x)', fontsize=11, fontweight='bold')
    ax2.set_title('Equivalente: Raíces de g(x) = f(x) - x', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('exercise_2_64_part1.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# PARTE 2: f(x) = √(8x + 6)
# ============================================================================

def part_2():
    print("\n" + "="*90)
    print("PARTE 2: PUNTOS FIJOS DE f(x) = √(8x + 6)")
    print("="*90)
    
    print("\nTEORÍA:")
    print("  Punto fijo: f(x) = x")
    print("  Para f(x) = √(8x + 6):")
    print("    √(8x + 6) = x")
    print("    8x + 6 = x²  (elevando al cuadrado)")
    print("    x² - 8x - 6 = 0")
    print("  Fórmula cuadrática:")
    print("    x = (8 ± √(64 + 24)) / 2 = (8 ± √88) / 2")
    
    # Calcular raíces exactas
    discriminant = 64 + 24
    x1 = (8 + np.sqrt(discriminant)) / 2
    x2 = (8 - np.sqrt(discriminant)) / 2
    
    print(f"    x₁ = {x1:.10f}")
    print(f"    x₂ = {x2:.10f}")
    print(f"\n  Nota: x debe ser ≥ -3/4 = {-3/4:.3f} (para que √(8x+6) esté definida)")
    print(f"        y x ≥ 0 (para que √(8x+6) = x tenga sentido)")
    
    def f_original(x):
        """f(x) = √(8x + 6)"""
        return np.sqrt(8*x + 6) if 8*x + 6 >= 0 else np.nan
    
    def g(x):
        """g(x) = f(x) - x = √(8x + 6) - x"""
        if 8*x + 6 < 0:
            return np.nan
        return np.sqrt(8*x + 6) - x
    
    def dg(x):
        """g'(x) = 4/√(8x + 6) - 1"""
        if 8*x + 6 <= 0:
            return np.nan
        return 4 / np.sqrt(8*x + 6) - 1
    
    print("\n" + "="*90)
    print("MÉTODO NUMÉRICO: Newton-Raphson para g(x) = √(8x + 6) - x")
    print("="*90)
    
    initial_guesses = [0, 1, 3, 5, 8, 10]
    fixed_points = []
    
    print(f"\n{'x₀':<10} {'Punto fijo':<15} {'f(x)':<15} {'Verificación':<15}")
    print("-"*60)
    
    for x0 in initial_guesses:
        root, history = newton_method(g, dg, x0)
        
        if root is not None and root >= -0.75:
            f_val = f_original(root)
            if not np.isnan(f_val):
                is_fixed = abs(f_val - root) < 1e-8
                
                is_duplicate = any(abs(root - fp) < 0.01 for fp in fixed_points)
                if not is_duplicate:
                    fixed_points.append(root)
                    check = "✓ f(x) = x" if is_fixed else "✗"
                    print(f"{x0:<10.1f} {root:<15.10f} {f_val:<15.10f} {check:<15}")
    
    print(f"\nPUNTOS FIJOS ENCONTRADOS:")
    for fp in sorted(fixed_points):
        if fp >= 0:
            print(f"  x = {fp:.10f}, f(x) = {f_original(fp):.10f}")
    
    # Visualización
    visualize_fixed_point_2(f_original, fixed_points)
    
    return fixed_points

def visualize_fixed_point_2(f, fixed_points):
    """Visualización para parte 2"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.linspace(-0.7, 12, 1000)
    y_f = []
    for xi in x:
        val = f(xi)
        y_f.append(val if not np.isnan(val) else None)
    
    y_identity = x
    
    # Gráfica 1: f(x) vs x
    ax1 = axes[0]
    ax1.plot(x, y_f, 'b-', linewidth=2, label='f(x) = √(8x + 6)')
    ax1.plot(x, y_identity, 'r--', linewidth=2, label='y = x')
    ax1.grid(True, alpha=0.3)
    
    for fp in fixed_points:
        if fp >= 0:
            ax1.plot(fp, fp, 'go', markersize=12, label=f'Punto fijo: x={fp:.2f}')
            ax1.plot([fp, fp], [0, fp], 'g:', alpha=0.5)
            ax1.plot([0, fp], [fp, fp], 'g:', alpha=0.5)
    
    ax1.set_xlabel('x', fontsize=11, fontweight='bold')
    ax1.set_ylabel('y', fontsize=11, fontweight='bold')
    ax1.set_title('Punto Fijo: Intersección de f(x) y y=x', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_xlim([-1, 12])
    ax1.set_ylim([-1, 12])
    
    # Gráfica 2: Iteración de punto fijo
    ax2 = axes[1]
    ax2.plot(x, y_f, 'b-', linewidth=2, label='f(x) = √(8x + 6)')
    ax2.plot(x, y_identity, 'r--', linewidth=2, label='y = x')
    ax2.grid(True, alpha=0.3)
    
    # Mostrar iteración desde x₀
    if fixed_points:
        fp = fixed_points[0] if fixed_points[0] > 0 else fixed_points[1] if len(fixed_points) > 1 else None
        if fp:
            x0_iter = 1.0
            x_curr = x0_iter
            for i in range(5):
                x_next = f(x_curr)
                if not np.isnan(x_next):
                    ax2.plot([x_curr, x_curr], [x_curr, x_next], 'g-', linewidth=1, alpha=0.7)
                    ax2.plot([x_curr, x_next], [x_next, x_next], 'g-', linewidth=1, alpha=0.7)
                    ax2.plot(x_curr, x_next, 'go', markersize=4)
                    x_curr = x_next
    
    ax2.set_xlabel('x', fontsize=11, fontweight='bold')
    ax2.set_ylabel('y', fontsize=11, fontweight='bold')
    ax2.set_title('Iteración de Punto Fijo', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 10])
    ax2.set_ylim([0, 10])
    
    plt.tight_layout()
    plt.savefig('exercise_2_64_part2.png', dpi=150, bbox_inches='tight')
    plt.show()

# ============================================================================
# RESUMEN
# ============================================================================

def print_summary(fp1, fp2):
    print("\n" + "="*90)
    print("RESUMEN: PUNTOS FIJOS")
    print("="*90)
    
    print("\nCONCEPTO CLAVE:")
    print("  Punto fijo: f(x) = x")
    print("  Encontrar puntos fijos ≡ Resolver f(x) - x = 0")
    
    print("\nPARTE 1: f(x) = x² - 6 en [0, ∞)")
    print(f"  Punto fijo: x = {fp1[0]:.10f}")
    print(f"  Verificación: f({fp1[0]:.6f}) = {fp1[0]**2 - 6:.6f} ≈ {fp1[0]:.6f} ✓")
    
    print("\nPARTE 2: f(x) = √(8x + 6)")
    valid_fp2 = [fp for fp in fp2 if fp >= 0]
    for fp in valid_fp2:
        print(f"  Punto fijo: x = {fp:.10f}")
        print(f"  Verificación: f({fp:.6f}) = {np.sqrt(8*fp + 6):.6f} ≈ {fp:.6f} ✓")
    
    print("\nMÉTODO: Newton-Raphson para resolver g(x) = f(x) - x = 0")
    
    print("\n" + "="*90)
    print("✓ EJERCICIO 2.64 COMPLETADO")
    print("="*90)

# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    print("="*90)
    print("EJERCICIO 2.64: ENCONTRAR PUNTOS FIJOS")
    print("="*90)
    
    fp1 = part_1()
    fp2 = part_2()
    print_summary(fp1, fp2)