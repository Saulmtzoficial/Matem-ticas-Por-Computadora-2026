"""
Exercise 2.58: Convergence Rate Analysis

PROBLEMA:
Un método de búsqueda de raíces tiene tasa de convergencia de orden M si:
    |x_{k+1} - x*| ≤ C|x_k - x*|^M

Analizar:
1. Logaritmo de la ecuación
2. Pendiente en gráfica log-log
3. Orden de convergencia de 6 métodos (de la imagen)
4. Interpretación de órdenes de convergencia
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================================
# PARTE 1: DERIVACIÓN LOGARÍTMICA
# ============================================================================

def part_1_derivation():
    """
    Parte 1: Tomar logaritmo de la ecuación de convergencia
    """
    print("="*100)
    print("PARTE 1: LOGARITMO DE LA ECUACIÓN DE CONVERGENCIA")
    print("="*100)
    
    print("""
ECUACIÓN ORIGINAL:
    |x_{k+1} - x*| ≤ C|x_k - x*|^M

TOMANDO log₁₀ DE AMBOS LADOS:
    log₁₀(|x_{k+1} - x*|) ≤ log₁₀(C|x_k - x*|^M)

APLICANDO PROPIEDADES DE LOGARITMOS:
    log₁₀(|x_{k+1} - x*|) ≤ log₁₀(C) + log₁₀(|x_k - x*|^M)
    
    log₁₀(|x_{k+1} - x*|) ≤ log₁₀(C) + M·log₁₀(|x_k - x*|)

RESPUESTA:
    log(|x_{k+1} - x*|) ≤ log₁₀(C) + M·log(|x_k - x*|)
    
    Primer término: log₁₀(C)
    Segundo término: M·log(|x_k - x*|)
    """)


# ============================================================================
# PARTE 2: PENDIENTE EN GRÁFICA LOG-LOG
# ============================================================================

def part_2_slope():
    """
    Parte 2: Identificar la pendiente
    """
    print("\n" + "="*100)
    print("PARTE 2: PENDIENTE EN GRÁFICA LOG-LOG")
    print("="*100)
    
    print("""
DE LA PARTE 1:
    log(|x_{k+1} - x*|) ≤ log₁₀(C) + M·log(|x_k - x*|)

FORMA LINEAL: y ≤ b + mx
donde:
    y = log(|x_{k+1} - x*|)  (error nuevo en log)
    x = log(|x_k - x*|)      (error viejo en log)
    m = M                     (pendiente)
    b = log₁₀(C)             (intercepto)

RESPUESTA:
    La pendiente de la función lineal en gráfica log-log es: M
    
    M es el ORDEN DE CONVERGENCIA del método.
    
INTERPRETACIÓN:
    • Pendiente M = 1: Convergencia lineal
    • Pendiente M = 2: Convergencia cuadrática
    • Pendiente M = 3: Convergencia cúbica
    • etc.
    """)


# ============================================================================
# PARTE 3: ANÁLISIS DE LOS 6 MÉTODOS
# ============================================================================

def analyze_image_methods():
    """
    Parte 3: Analizar los 6 métodos de la imagen
    """
    print("\n" + "="*100)
    print("PARTE 3: ANÁLISIS DE LOS 6 MÉTODOS DE LA IMAGEN")
    print("="*100)
    
    print("""
CÓMO DETERMINAR EL ORDEN DE CONVERGENCIA:

1. En una gráfica log-log de error_{k+1} vs error_k
2. La pendiente de la línea = orden de convergencia M
3. Pendiente más empinada = convergencia más rápida

ANÁLISIS DE LA IMAGEN:
""")
    
    # Análisis basado en la imagen proporcionada
    methods = {
        'Method 1': {
            'order': 1,
            'color': 'green',
            'description': 'Convergencia LINEAL',
            'analysis': 'Pendiente ≈ 1, línea con ángulo ~45° en log-log'
        },
        'Method 2': {
            'order': 2,
            'color': 'black',
            'description': 'Convergencia CUADRÁTICA',
            'analysis': 'Pendiente ≈ 2, línea más empinada'
        },
        'Method 3': {
            'order': 2,
            'color': 'magenta',
            'description': 'Convergencia CUADRÁTICA',
            'analysis': 'Pendiente ≈ 2, similar al Method 2'
        },
        'Method 4': {
            'order': 2,
            'color': 'red',
            'description': 'Convergencia CUADRÁTICA',
            'analysis': 'Pendiente ≈ 2, comportamiento cuadrático'
        },
        'Method 5': {
            'order': 1,
            'color': 'blue',
            'description': 'Convergencia LINEAL',
            'analysis': 'Pendiente ≈ 1, similar al Method 1'
        },
        'Method 6': {
            'order': 2,
            'color': 'cyan',
            'description': 'Convergencia CUADRÁTICA',
            'analysis': 'Pendiente ≈ 2, comportamiento cuadrático'
        }
    }
    
    print(f"\n{'Método':<12} {'Orden M':<10} {'Tipo':<25} {'Observación':<50}")
    print("-"*100)
    
    for method_name, info in methods.items():
        print(f"{method_name:<12} {info['order']:<10} {info['description']:<25} {info['analysis']:<50}")
    
    print("""
OBSERVACIONES CLAVE DE LA IMAGEN:

1. MÉTODOS DE ORDEN 1 (Lineales):
   - Method 1 (verde punteado)
   - Method 5 (azul punteado)
   - Líneas con pendiente ≈ 1 en log-log
   - Convergen más lentamente
   - Ejemplos: Bisección, Regula Falsi, Secante (≈1.618)

2. MÉTODOS DE ORDEN 2 (Cuadráticos):
   - Method 2 (negro)
   - Method 3 (magenta)
   - Method 4 (rojo)
   - Method 6 (cyan)
   - Líneas con pendiente ≈ 2 en log-log
   - Convergen más rápidamente
   - Ejemplo: Newton-Raphson

3. PATRÓN EN LA GRÁFICA:
   - Todos parten del mismo error inicial (~10⁻¹²)
   - Los métodos cuadráticos convergen mucho más rápido
   - La diferencia es dramática: orden 2 >> orden 1
    """)


# ============================================================================
# DEMOSTRACIÓN CON MÉTODOS REALES
# ============================================================================

def demonstrate_convergence_orders():
    """
    Demostración práctica con métodos conocidos
    """
    print("\n" + "="*100)
    print("DEMOSTRACIÓN: ÓRDENES DE CONVERGENCIA CON MÉTODOS REALES")
    print("="*100)
    
    # Problema: f(x) = x² - 2, raíz = √2
    f = lambda x: x**2 - 2
    df = lambda x: 2*x
    ddf = lambda x: 2
    root_true = np.sqrt(2)
    x0 = 1.5
    
    # Bisección (Orden 1)
    def bisection_errors():
        a, b = 1.0, 2.0
        errors = []
        for _ in range(20):
            c = (a + b) / 2
            errors.append(abs(c - root_true))
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return errors
    
    # Newton (Orden 2)
    def newton_errors():
        xn = x0
        errors = [abs(xn - root_true)]
        for _ in range(10):
            xn = xn - f(xn) / df(xn)
            errors.append(abs(xn - root_true))
        return errors
    
    # Halley (Orden 3)
    def halley_errors():
        xn = x0
        errors = [abs(xn - root_true)]
        for _ in range(8):
            fn = f(xn)
            dfn = df(xn)
            ddfn = ddf(xn)
            denominator = 2*dfn**2 - fn*ddfn
            if abs(denominator) > 1e-14:
                xn = xn - (2*fn*dfn) / denominator
                errors.append(abs(xn - root_true))
        return errors
    
    errors_bis = bisection_errors()
    errors_newton = newton_errors()
    errors_halley = halley_errors()
    
    # Gráfica log-log
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Bisección
    ax1 = axes[0]
    if len(errors_bis) > 1:
        for i in range(len(errors_bis)-1):
            if errors_bis[i] > 1e-15 and errors_bis[i+1] > 1e-15:
                ax1.loglog(errors_bis[i], errors_bis[i+1], 'go', markersize=8)
        
        # Línea teórica con pendiente 1
        x_line = np.logspace(np.log10(min(errors_bis)), np.log10(max(errors_bis)), 100)
        y_line = x_line  # Pendiente 1
        ax1.loglog(x_line, y_line, 'r--', linewidth=2, label='Pendiente = 1')
    
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_xlabel('Error en iteración k', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Error en iteración k+1', fontsize=11, fontweight='bold')
    ax1.set_title('Bisección (Orden 1)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    
    # Newton
    ax2 = axes[1]
    if len(errors_newton) > 1:
        for i in range(len(errors_newton)-1):
            if errors_newton[i] > 1e-15 and errors_newton[i+1] > 1e-15:
                ax2.loglog(errors_newton[i], errors_newton[i+1], 'bo', markersize=8)
        
        # Línea teórica con pendiente 2
        x_line = np.logspace(np.log10(min(errors_newton[:-1])), np.log10(max(errors_newton[:-1])), 100)
        y_line = x_line**2  # Pendiente 2
        ax2.loglog(x_line, y_line, 'r--', linewidth=2, label='Pendiente = 2')
    
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xlabel('Error en iteración k', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Error en iteración k+1', fontsize=11, fontweight='bold')
    ax2.set_title('Newton (Orden 2)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    
    # Halley
    ax3 = axes[2]
    if len(errors_halley) > 1:
        for i in range(len(errors_halley)-1):
            if errors_halley[i] > 1e-15 and errors_halley[i+1] > 1e-15:
                ax3.loglog(errors_halley[i], errors_halley[i+1], 'mo', markersize=8)
        
        # Línea teórica con pendiente 3
        x_line = np.logspace(np.log10(min(errors_halley[:-1])), np.log10(max(errors_halley[:-1])), 100)
        y_line = x_line**3  # Pendiente 3
        ax3.loglog(x_line, y_line, 'r--', linewidth=2, label='Pendiente = 3')
    
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_xlabel('Error en iteración k', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Error en iteración k+1', fontsize=11, fontweight='bold')
    ax3.set_title('Halley (Orden 3)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('exercise_2_58_convergence_orders.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# PARTE 4: INTERPRETACIÓN DE ÓRDENES DE CONVERGENCIA
# ============================================================================

def part_4_interpretation():
    """
    Parte 4: ¿Qué significa cada orden de convergencia?
    """
    print("\n" + "="*100)
    print("PARTE 4: INTERPRETACIÓN DE ÓRDENES DE CONVERGENCIA")
    print("="*100)
    
    print("""
¿QUÉ SIGNIFICA CADA ORDEN DE CONVERGENCIA?

═══════════════════════════════════════════════════════════════════════════

1. CONVERGENCIA DE PRIMER ORDEN (M = 1, LINEAL):
   
   Fórmula: |e_{k+1}| ≈ C|e_k|
   
   SIGNIFICADO:
   • El error se MULTIPLICA por una constante C < 1 en cada iteración
   • Reducción constante del error
   • Ganancia CONSTANTE de dígitos correctos por iteración
   
   EJEMPLO:
   Si C = 0.5:
   - e_0 = 0.1
   - e_1 = 0.05      (reducción de 2×)
   - e_2 = 0.025     (reducción de 2×)
   - e_3 = 0.0125    (reducción de 2×)
   
   MÉTODOS:
   • Bisección (C = 0.5, cada iteración divide error a la mitad)
   • Regula Falsi (orden ≈ 1.618, superlineal)
   
   VELOCIDAD: LENTA pero predecible

═══════════════════════════════════════════════════════════════════════════

2. CONVERGENCIA DE SEGUNDO ORDEN (M = 2, CUADRÁTICA):
   
   Fórmula: |e_{k+1}| ≈ C|e_k|²
   
   SIGNIFICADO:
   • El error se ELEVA AL CUADRADO en cada iteración
   • Los dígitos correctos se DUPLICAN aproximadamente
   • Convergencia MUY rápida cerca de la raíz
   
   EJEMPLO:
   Si C = 1:
   - e_0 = 0.1        (1 dígito correcto)
   - e_1 = 0.01       (2 dígitos correctos)
   - e_2 = 0.0001     (4 dígitos correctos)
   - e_3 = 0.00000001 (8 dígitos correctos)
   
   MÉTODOS:
   • Newton-Raphson
   • Método de la secante al cuadrado
   
   VELOCIDAD: MUY RÁPIDA cerca de la raíz

═══════════════════════════════════════════════════════════════════════════

3. CONVERGENCIA DE TERCER ORDEN (M = 3, CÚBICA):
   
   Fórmula: |e_{k+1}| ≈ C|e_k|³
   
   SIGNIFICADO:
   • El error se ELEVA AL CUBO en cada iteración
   • Los dígitos correctos se TRIPLICAN aproximadamente
   • Convergencia EXTREMADAMENTE rápida
   
   EJEMPLO:
   Si C = 1:
   - e_0 = 0.1           (1 dígito)
   - e_1 = 0.001         (3 dígitos)
   - e_2 = 0.000000001   (9 dígitos)
   - e_3 = 10⁻²⁷         (27 dígitos)
   
   MÉTODOS:
   • Halley (Taylor N=2)
   • Chebyshev
   
   VELOCIDAD: EXTREMADAMENTE RÁPIDA

═══════════════════════════════════════════════════════════════════════════

4. CONVERGENCIA SUPERLINEAL (1 < M < 2):
   
   SIGNIFICADO:
   • Más rápida que lineal pero más lenta que cuadrática
   • Intermedia entre primer y segundo orden
   
   MÉTODOS:
   • Secante (M ≈ 1.618, el número áureo φ)

═══════════════════════════════════════════════════════════════════════════

COMPARACIÓN VISUAL:

Iteraciones para 10 dígitos correctos (desde e_0 = 0.1):

  Orden 1 (lineal):      ~30 iteraciones
  Orden 1.618 (secante): ~14 iteraciones
  Orden 2 (cuadrática):  ~4 iteraciones
  Orden 3 (cúbica):      ~3 iteraciones

═══════════════════════════════════════════════════════════════════════════

TRADE-OFFS:

Mayor orden NO siempre es mejor:
- Orden superior requiere más cálculos por iteración
- Puede requerir más derivadas
- Puede ser más complejo de implementar
- Puede ser menos robusto

REGLA GENERAL:
- Para problemas simples: Newton (orden 2) es óptimo
- Para funciones sin derivada: Secante (orden 1.618)
- Para máxima eficiencia: considerar costo por iteración vs número de iteraciones
    """)


# ============================================================================
# TABLA COMPARATIVA
# ============================================================================

def comparison_table():
    """
    Tabla comparativa de métodos
    """
    print("\n" + "="*100)
    print("TABLA COMPARATIVA DE MÉTODOS")
    print("="*100)
    
    data = [
        ["Bisección", "1", "Lineal", "0.5", "No", "Intervalo [a,b]", "Siempre converge"],
        ["Regula Falsi", "~1.618", "Superlineal", "Variable", "No", "Intervalo [a,b]", "Siempre converge"],
        ["Secante", "~1.618", "Superlineal", "Variable", "No", "2 puntos", "Generalmente"],
        ["Newton", "2", "Cuadrática", "Variable", "Sí (f')", "1 punto", "Si x₀ cercano"],
        ["Halley", "3", "Cúbica", "Variable", "Sí (f', f'')", "1 punto", "Si x₀ cercano"]
    ]
    
    print(f"\n{'Método':<15} {'Orden':<10} {'Tipo':<15} {'C':<10} {'Derivada':<15} {'Inicio':<15} {'Robustez':<20}")
    print("-"*100)
    for row in data:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<15} {row[3]:<10} {row[4]:<15} {row[5]:<15} {row[6]:<20}")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("="*100)
    print("EJERCICIO 2.58: ANÁLISIS DE TASAS DE CONVERGENCIA")
    print("="*100)
    
    # Parte 1
    part_1_derivation()
    
    # Parte 2
    part_2_slope()
    
    # Parte 3
    analyze_image_methods()
    
    # Demostración
    demonstrate_convergence_orders()
    
    # Parte 4
    part_4_interpretation()
    
    # Tabla comparativa
    comparison_table()
    
    print("\n" + "="*100)
    print("✓ EJERCICIO 2.58 COMPLETADO")
    print("="*100)