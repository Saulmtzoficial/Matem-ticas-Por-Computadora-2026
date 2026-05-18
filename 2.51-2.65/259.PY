"""
Exercise 2.59: Error Analysis in Newton's Method

PROBLEMA:
Shelby está usando el Método de Newton. Después de una iteración,
el error absoluto es |x₁ - x*| = 0.2

¿Cuál es el error esperado en los pasos 2, 3 y 4?

TEORÍA: CONVERGENCIA CUADRÁTICA
|eₙ₊₁| ≈ M|eₙ|²  donde M = |f''(x*)| / (2|f'(x*)|)
"""

import numpy as np
import matplotlib.pyplot as plt

def analyze_quadratic_convergence():
    print("="*100)
    print("EJERCICIO 2.59: ANÁLISIS DE ERROR EN EL MÉTODO DE NEWTON")
    print("="*100)
    
    print("\n" + "="*100)
    print("DATOS DEL PROBLEMA")
    print("="*100)
    print("Error después de 1 iteración: |x₁ - x*| = 0.2")
    print("Pregunta: ¿Cuál es el error esperado en pasos 2, 3, 4?")
    
    # ========================================================================
    # TEORÍA
    # ========================================================================
    
    print("\n" + "="*100)
    print("TEORÍA: CONVERGENCIA CUADRÁTICA DE NEWTON")
    print("="*100)
    print("""
Para raíces simples, Newton tiene convergencia cuadrática:

    |eₙ₊₁| ≈ M|eₙ|²

donde:
    eₙ = xₙ - x*  (error en iteración n)
    M = |f''(x*)| / (2|f'(x*)|)  (constante asintótica)

SIGNIFICADO:
- El error se eleva al cuadrado en cada iteración
- Los dígitos correctos se duplican aproximadamente
    """)
    
    # ========================================================================
    # CASO 1: M = 1
    # ========================================================================
    
    print("\n" + "="*100)
    print("CASO 1: SUPONIENDO M ≈ 1")
    print("="*100)
    
    e1 = 0.2
    M = 1.0
    
    e2 = M * e1**2
    e3 = M * e2**2
    e4 = M * e3**2
    
    print(f"\nError en paso 1: e₁ = {e1}")
    print(f"Error en paso 2: e₂ ≈ M × e₁² = {M} × ({e1})² = {e2:.6e}")
    print(f"Error en paso 3: e₃ ≈ M × e₂² = {M} × ({e2:.6e})² = {e3:.6e}")
    print(f"Error en paso 4: e₄ ≈ M × e₃² = {M} × ({e3:.6e})² = {e4:.6e}")
    
    print("\nREDUCCIÓN DEL ERROR:")
    print(f"  • e₁ → e₂: factor de {e1/e2:.1f}×")
    print(f"  • e₂ → e₃: factor de {e2/e3:.0f}×")
    print(f"  • e₃ → e₄: factor de {e3/e4:.0f}×")
    
    # ========================================================================
    # CASO 2: DIFERENTES M
    # ========================================================================
    
    print("\n" + "="*100)
    print("CASO 2: SENSIBILIDAD A M")
    print("="*100)
    
    M_values = [0.5, 1.0, 2.0, 5.0]
    
    print(f"\n{'M':<10} {'e₂':<15} {'e₃':<15} {'e₄':<15}")
    print("-"*60)
    
    results_M = {}
    for M_val in M_values:
        e2_M = M_val * e1**2
        e3_M = M_val * e2_M**2
        e4_M = M_val * e3_M**2
        results_M[M_val] = [e2_M, e3_M, e4_M]
        print(f"{M_val:<10.1f} {e2_M:<15.6e} {e3_M:<15.6e} {e4_M:<15.6e}")
    
    # ========================================================================
    # DEMOSTRACIÓN CON EJEMPLO
    # ========================================================================
    
    print("\n" + "="*100)
    print("DEMOSTRACIÓN CON EJEMPLO: f(x) = x² - 2")
    print("="*100)
    
    def f_example(x):
        return x**2 - 2
    
    def df_example(x):
        return 2*x
    
    root_true = np.sqrt(2)
    
    # Buscar x₀ tal que |x₁ - √2| ≈ 0.2
    x0_found = None
    for x0_test in np.linspace(1.0, 2.5, 1000):
        x1 = x0_test - f_example(x0_test) / df_example(x0_test)
        if abs(abs(x1 - root_true) - 0.2) < 0.005:
            x0_found = x0_test
            break
    
    if x0_found is None:
        print("\nUsando x₀ = 2.0 como ejemplo aproximado...")
        x0_found = 2.0
    else:
        print(f"\nEncontramos x₀ = {x0_found:.10f} tal que |x₁ - √2| ≈ 0.2")
    
    # Ejecutar Newton
    history = [x0_found]
    errors = [abs(x0_found - root_true)]
    xn = x0_found
    
    for n in range(4):
        xn = xn - f_example(xn) / df_example(xn)
        history.append(xn)
        errors.append(abs(xn - root_true))
    
    print(f"\n{'Iter':<8} {'xₙ':<18} {'|xₙ - √2|':<18} {'Ratio eₙ₊₁/eₙ²':<18}")
    print("-"*70)
    for n in range(len(errors)):
        if n > 0 and errors[n-1] > 1e-14:
            ratio = errors[n]/errors[n-1]**2
        else:
            ratio = 0
        print(f"{n:<8} {history[n]:<18.12f} {errors[n]:<18.6e} {ratio:<18.6f}")
    
    # M empírico
    if errors[0] > 1e-14:
        M_empirical = errors[1] / errors[0]**2
        print(f"\nM empírico: {M_empirical:.6f}")
    
    # M teórico
    M_theoretical = 1 / (2 * np.sqrt(2))
    print(f"M teórico:  {M_theoretical:.6f}")
    
    # ========================================================================
    # VISUALIZACIÓN
    # ========================================================================
    
    print("\n" + "="*100)
    print("GENERANDO GRÁFICAS...")
    print("="*100)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfica 1: Error vs Iteración
    ax1 = axes[0, 0]
    iterations = [1, 2, 3, 4]
    errors_M1 = [e1, e2, e3, e4]
    
    ax1.semilogy(iterations, errors_M1, 'ro-', linewidth=3, markersize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Iteración n', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Error |eₙ| (log)', fontsize=11, fontweight='bold')
    ax1.set_title('Convergencia Cuadrática (M=1)', fontsize=12, fontweight='bold')
    ax1.set_xticks(iterations)
    
    for i, (it, err) in enumerate(zip(iterations, errors_M1)):
        ax1.text(it, err*1.5, f'{err:.2e}', ha='center', fontsize=9, fontweight='bold')
    
    # Gráfica 2: Diferentes M
    ax2 = axes[0, 1]
    for M_val in M_values:
        errors_temp = [e1] + results_M[M_val]
        ax2.semilogy(iterations, errors_temp, 'o-', linewidth=2, 
                    markersize=6, label=f'M={M_val}')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Iteración n', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Error |eₙ| (log)', fontsize=11, fontweight='bold')
    ax2.set_title('Efecto de M', fontsize=12, fontweight='bold')
    ax2.set_xticks(iterations)
    ax2.legend(fontsize=10)
    
    # Gráfica 3: Verificación cuadrática (ejemplo)
    ax3 = axes[1, 0]
    ratios = []
    for i in range(len(errors)-1):
        if errors[i] > 1e-14:
            ratios.append(errors[i+1]/errors[i]**2)
    
    if ratios:
        ax3.plot(range(1, len(ratios)+1), ratios, 'bs-', linewidth=2, markersize=8)
        if errors[0] > 1e-14:
            ax3.axhline(y=M_empirical, color='r', linestyle='--', 
                       linewidth=2, label=f'M empírico={M_empirical:.3f}')
        ax3.axhline(y=M_theoretical, color='g', linestyle='--', 
                   linewidth=2, label=f'M teórico={M_theoretical:.3f}')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Iteración n', fontsize=11, fontweight='bold')
        ax3.set_ylabel('eₙ₊₁/eₙ²', fontsize=11, fontweight='bold')
        ax3.set_title('Verificación Cuadrática', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
    
    # Gráfica 4: Dígitos correctos
    ax4 = axes[1, 1]
    digits = [-np.log10(e) if e > 0 else 15 for e in errors_M1]
    
    ax4.plot(iterations, digits, 'go-', linewidth=3, markersize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Iteración n', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Dígitos correctos', fontsize=11, fontweight='bold')
    ax4.set_title('Duplicación de Dígitos', fontsize=12, fontweight='bold')
    ax4.set_xticks(iterations)
    
    for i, (it, dig) in enumerate(zip(iterations, digits)):
        ax4.text(it, dig+0.3, f'{dig:.1f}', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('exercise_2_59.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ========================================================================
    # RESPUESTA FINAL
    # ========================================================================
    
    print("\n" + "="*100)
    print("RESPUESTA AL EJERCICIO 2.59")
    print("="*100)
    
    print(f"""
DADO: |x₁ - x*| = 0.2

RESPUESTAS (asumiendo M ≈ 1):

  Paso 2: |x₂ - x*| ≈ {e2} = 4.0 × 10⁻²
  
  Paso 3: |x₃ - x*| ≈ {e3:.2e} = 1.6 × 10⁻³
  
  Paso 4: |x₄ - x*| ≈ {e4:.2e} = 2.56 × 10⁻⁶

DEFENSA DEL RAZONAMIENTO:

1. CONVERGENCIA CUADRÁTICA:
   Newton tiene convergencia cuadrática: |eₙ₊₁| ≈ M|eₙ|²
   El error se ELEVA AL CUADRADO en cada iteración.

2. CÁLCULO:
   • e₂ = M × (0.2)² = 1 × 0.04 = 0.04
   • e₃ = M × (0.04)² = 1 × 0.0016 = 0.0016
   • e₄ = M × (0.0016)² = 1 × 0.0000026 ≈ 2.56×10⁻⁶

3. SUPOSICIÓN M ≈ 1:
   Sin información adicional, asumimos M ≈ 1.
   M = |f''(x*)| / (2|f'(x*)|) depende de la función específica.

4. DUPLICACIÓN DE DÍGITOS:
   Paso 1: ~0.7 dígitos correctos
   Paso 2: ~1.4 dígitos correctos  (duplica)
   Paso 3: ~2.8 dígitos correctos  (duplica)
   Paso 4: ~5.6 dígitos correctos  (duplica)

5. ACELERACIÓN:
   La convergencia se ACELERA dramáticamente:
   • e₁ → e₂: reducción de 5×
   • e₂ → e₃: reducción de 25×
   • e₃ → e₄: reducción de 625×

CONCLUSIÓN:
Estos son valores aproximados. El patrón cuadrático |eₙ₊₁| ≈ M|eₙ|²
es exacto para Newton con raíces simples.
    """)
    
    # TABLA RESUMEN
    print("\n" + "="*100)
    print("TABLA RESUMEN")
    print("="*100)
    
    print(f"\n{'Iter':<8} {'Error':<20} {'Dígitos correctos':<20} {'Reducción':<15}")
    print("-"*70)
    print(f"{1:<8} {e1:<20.6e} {-np.log10(e1):<20.1f} {'—':<15}")
    print(f"{2:<8} {e2:<20.6e} {-np.log10(e2):<20.1f} {e1/e2:<15.0f}×")
    print(f"{3:<8} {e3:<20.6e} {-np.log10(e3):<20.1f} {e2/e3:<15.0f}×")
    print(f"{4:<8} {e4:<20.6e} {-np.log10(e4):<20.1f} {e3/e4:<15.0f}×")
    
    print("\n" + "="*100)
    print("✓ EJERCICIO COMPLETADO")
    print("="*100)


if __name__ == "__main__":
    analyze_quadratic_convergence()