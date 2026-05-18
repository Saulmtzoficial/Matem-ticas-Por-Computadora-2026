"""
Exercise 2.62: Can methods find roots of f(x) = cos(x) + 1?

PROBLEMA:
¿Pueden el Método de Bisección, Regula Falsi, o Newton encontrar
las raíces de f(x) = cos(x) + 1?

Explicar por qué sí o por qué no para cada técnica.

ANÁLISIS:
f(x) = cos(x) + 1
f'(x) = -sin(x)

Raíces: cos(x) + 1 = 0 → cos(x) = -1 → x = π + 2πn, n ∈ ℤ
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# ANÁLISIS DE LA FUNCIÓN
# ============================================================================

def analyze_function():
    """
    Análisis completo de f(x) = cos(x) + 1
    """
    print("="*100)
    print("EJERCICIO 2.62: ¿PUEDEN LOS MÉTODOS ENCONTRAR RAÍCES DE f(x) = cos(x) + 1?")
    print("="*100)
    
    print("\n" + "="*100)
    print("ANÁLISIS DE LA FUNCIÓN")
    print("="*100)
    
    print("""
FUNCIÓN: f(x) = cos(x) + 1

DERIVADA: f'(x) = -sin(x)

RAÍCES:
    cos(x) + 1 = 0
    cos(x) = -1
    x = π + 2πn, donde n ∈ ℤ (enteros)

RAÍCES ESPECÍFICAS:
    x = ..., -3π, -π, π, 3π, 5π, ...
    x ≈ ..., -9.42, -3.14, 3.14, 9.42, 15.71, ...

PROPIEDADES IMPORTANTES:
    1. f(x) ≥ 0 para todo x (¡NUNCA ES NEGATIVA!)
       Rango: [0, 2]
    
    2. f(x) = 0 SOLAMENTE en las raíces
       f(x) > 0 en todos los demás puntos
    
    3. f(x) toca el eje x pero NO lo cruza
       La función es tangente al eje x en las raíces
    
    4. f'(x) = -sin(x) = 0 en las raíces
       ¡La derivada es cero en las raíces!
    """)


def visualize_function():
    """
    Visualización de la función
    """
    print("\n" + "="*100)
    print("VISUALIZACIÓN DE f(x) = cos(x) + 1")
    print("="*100)
    
    x = np.linspace(-4*np.pi, 4*np.pi, 2000)
    f = np.cos(x) + 1
    df = -np.sin(x)
    
    # Raíces exactas
    roots = [np.pi * (2*n + 1) for n in range(-2, 3)]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Gráfica 1: Función completa
    ax1 = axes[0, 0]
    ax1.plot(x, f, 'b-', linewidth=2.5, label='f(x) = cos(x) + 1')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    for root in roots:
        ax1.plot(root, 0, 'ro', markersize=10)
        ax1.annotate(f'x={root/np.pi:.0f}π', xy=(root, 0), 
                    xytext=(root, 0.3), fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x', fontsize=11, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=11, fontweight='bold')
    ax1.set_title('f(x) = cos(x) + 1 (Función completa)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim([-0.5, 2.5])
    
    # Añadir texto importante
    ax1.text(0, 1.8, '¡f(x) ≥ 0 siempre!\nNunca cruza el eje x', 
            fontsize=11, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # Gráfica 2: Zoom cerca de una raíz
    ax2 = axes[0, 1]
    x_zoom = np.linspace(np.pi - 1, np.pi + 1, 500)
    f_zoom = np.cos(x_zoom) + 1
    
    ax2.plot(x_zoom, f_zoom, 'b-', linewidth=3, label='f(x)')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.plot(np.pi, 0, 'ro', markersize=12, label=f'Raíz en x=π')
    
    # Tangente en la raíz
    tangent_x = np.linspace(np.pi - 0.5, np.pi + 0.5, 100)
    tangent_y = 0  # f'(π) = 0, tangente horizontal
    ax2.plot(tangent_x, tangent_y, 'g--', linewidth=2, label='Tangente (horizontal)')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x', fontsize=11, fontweight='bold')
    ax2.set_ylabel('f(x)', fontsize=11, fontweight='bold')
    ax2.set_title('Zoom cerca de x = π', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_ylim([-0.1, 0.5])
    
    ax2.text(np.pi + 0.3, 0.3, 'f(x) toca pero\nNO cruza eje x', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Gráfica 3: Derivada
    ax3 = axes[1, 0]
    ax3.plot(x, df, 'g-', linewidth=2.5, label="f'(x) = -sin(x)")
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    
    for root in roots:
        ax3.plot(root, 0, 'ro', markersize=10)
    
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('x', fontsize=11, fontweight='bold')
    ax3.set_ylabel("f'(x)", fontsize=11, fontweight='bold')
    ax3.set_title("Derivada f'(x) = -sin(x)", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    
    ax3.text(0, 0.7, "f'(x) = 0 en las raíces!\n¡Tangente horizontal!", 
            fontsize=11, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
    
    # Gráfica 4: Signos de f(x)
    ax4 = axes[1, 1]
    ax4.fill_between(x, 0, f, where=(f >= 0), color='blue', alpha=0.3, label='f(x) > 0')
    ax4.plot(x, f, 'b-', linewidth=2)
    ax4.axhline(y=0, color='k', linewidth=2)
    
    for root in roots:
        ax4.plot(root, 0, 'ro', markersize=10)
    
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('x', fontsize=11, fontweight='bold')
    ax4.set_ylabel('f(x)', fontsize=11, fontweight='bold')
    ax4.set_title('Análisis de Signos', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.set_ylim([-0.5, 2.5])
    
    ax4.text(0, 1.8, '¡TODA la función\nestá ARRIBA del eje x!', 
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('exercise_2_62_function.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# ANÁLISIS MÉTODO POR MÉTODO
# ============================================================================

def analyze_bisection():
    """
    ¿Puede Bisección encontrar raíces?
    """
    print("\n" + "="*100)
    print("MÉTODO 1: BISECCIÓN")
    print("="*100)
    
    print("""
REQUISITO DEL MÉTODO DE BISECCIÓN:
    Necesita un intervalo [a, b] donde f(a) · f(b) < 0
    Es decir, f(a) y f(b) deben tener SIGNOS OPUESTOS

ANÁLISIS PARA f(x) = cos(x) + 1:

    f(x) = cos(x) + 1 ≥ 0 para TODO x
    
    • Rango de f(x): [0, 2]
    • f(x) nunca es negativa
    • NO EXISTE ningún intervalo [a, b] donde f(a) · f(b) < 0

DEMOSTRACIÓN:
    Para cualquier a y b:
    - f(a) ≥ 0
    - f(b) ≥ 0
    - Por lo tanto: f(a) · f(b) ≥ 0
    
    ¡NUNCA se cumple f(a) · f(b) < 0!

CONCLUSIÓN:
    ❌ BISECCIÓN NO PUEDE USARSE
    
    Razón: La función nunca cambia de signo. No cruza el eje x,
           solo lo toca tangencialmente en las raíces.

NOTA IMPORTANTE:
    Aunque las raíces existen matemáticamente (donde f(x) = 0),
    el método de bisección requiere que la función CRUCE el eje x,
    no solo que lo toque.
    """)


def analyze_regula_falsi():
    """
    ¿Puede Regula Falsi encontrar raíces?
    """
    print("\n" + "="*100)
    print("MÉTODO 2: REGULA FALSI (FALSA POSICIÓN)")
    print("="*100)
    
    print("""
REQUISITO DEL MÉTODO DE REGULA FALSI:
    Igual que Bisección: necesita un intervalo [a, b] donde f(a) · f(b) < 0

ANÁLISIS PARA f(x) = cos(x) + 1:

    Regula Falsi usa interpolación lineal, pero TAMBIÉN requiere
    que f(a) y f(b) tengan signos opuestos.
    
    Por la misma razón que Bisección:
    • f(x) ≥ 0 siempre
    • NO hay cambio de signo
    • NO existe intervalo válido

CONCLUSIÓN:
    ❌ REGULA FALSI NO PUEDE USARSE
    
    Razón: Misma que Bisección. Requiere que f(a) · f(b) < 0,
           lo cual es imposible para esta función.

DIFERENCIA CON BISECCIÓN:
    Aunque Regula Falsi es "más inteligente" (usa pendientes),
    tiene el MISMO requisito fundamental: cambio de signo.
    """)


def analyze_newton():
    """
    ¿Puede Newton encontrar raíces?
    """
    print("\n" + "="*100)
    print("MÉTODO 3: NEWTON-RAPHSON")
    print("="*100)
    
    print("""
REQUISITO DEL MÉTODO DE NEWTON:
    xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)
    
    Necesita:
    1. Una aproximación inicial x₀
    2. Que f'(xₙ) ≠ 0

ANÁLISIS PARA f(x) = cos(x) + 1:

    f(x) = cos(x) + 1
    f'(x) = -sin(x)
    
    PROBLEMA EN LAS RAÍCES:
    Las raíces están en x = π + 2πn
    
    En las raíces:
    • f(π) = cos(π) + 1 = -1 + 1 = 0 ✓
    • f'(π) = -sin(π) = 0 ✗
    
    ¡LA DERIVADA ES CERO EN LAS RAÍCES!

¿QUÉ SUCEDE CON NEWTON?

    Caso 1: x₀ cerca pero no exactamente en la raíz
    -------
    Si x₀ está cerca de π pero no en π:
    • f(x₀) ≈ 0 (muy pequeño)
    • f'(x₀) ≈ 0 (muy pequeño)
    • x₁ = x₀ - f(x₀)/f'(x₀) → división de números pequeños
    
    Resultado: INESTABILIDAD NUMÉRICA
    El método puede:
    - Diverger
    - Dar saltos enormes
    - Converger muy lentamente
    - Fallar completamente

    Caso 2: x₀ exactamente en la raíz
    -------
    • f(π) = 0
    • f'(π) = 0
    • x₁ = π - 0/0 → ¡INDETERMINADO!
    
    Resultado: DIVISIÓN POR CERO

CONCLUSIÓN:
    ⚠️ NEWTON PUEDE FALLAR o ser MUY INESTABLE
    
    Razones:
    1. f'(x) = 0 en las raíces (tangente horizontal)
    2. Raíces de multiplicidad 2 (la función "toca" pero no cruza)
    3. División por cero o números muy pequeños
    4. Convergencia extremadamente lenta si converge

NOTA TEÓRICA:
    Las raíces donde f(x) = f'(x) = 0 se llaman "raíces múltiples"
    o "raíces de orden superior". Newton converge linealmente (lento)
    en lugar de cuadráticamente (rápido) para estas raíces.
    """)


def demonstrate_newton_failure():
    """
    Demostración práctica de por qué Newton falla
    """
    print("\n" + "="*100)
    print("DEMOSTRACIÓN: INTENTANDO NEWTON EN f(x) = cos(x) + 1")
    print("="*100)
    
    f = lambda x: np.cos(x) + 1
    df = lambda x: -np.sin(x)
    
    # Intentar desde diferentes x₀
    initial_guesses = [2.5, 3.0, 3.14, 3.5, 4.0]
    
    print("\nIntentando Newton desde diferentes aproximaciones iniciales:\n")
    print(f"{'x₀':<10} {'Resultado':<30} {'Comentario':<50}")
    print("-"*100)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, x0 in enumerate(initial_guesses):
        history = [x0]
        xn = x0
        
        converged = False
        diverged = False
        
        for n in range(20):
            fn = f(xn)
            dfn = df(xn)
            
            if abs(dfn) < 1e-10:
                print(f"{x0:<10.3f} {'FALLA':<30} {'Derivada ≈ 0, división por cero':<50}")
                diverged = True
                break
            
            xn_new = xn - fn / dfn
            history.append(xn_new)
            
            # Verificar si diverge
            if abs(xn_new - xn) > 100:
                print(f"{x0:<10.3f} {'DIVERGE':<30} {'Saltos enormes, inestable':<50}")
                diverged = True
                break
            
            if abs(fn) < 1e-8:
                print(f"{x0:<10.3f} {f'Converge en {n+1} iters':<30} {f'Raíz ≈ {xn_new:.6f}':<50}")
                converged = True
                break
            
            xn = xn_new
        
        if not converged and not diverged:
            print(f"{x0:<10.3f} {'Convergencia LENTA':<30} {'20 iteraciones, aún no converge':<50}")
        
        # Graficar
        if idx < 6:
            ax = axes[idx]
            x_plot = np.linspace(x0 - 2, x0 + 2, 1000)
            y_plot = f(x_plot)
            
            ax.plot(x_plot, y_plot, 'b-', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
            ax.axvline(x=np.pi, color='r', linestyle='--', linewidth=1, alpha=0.5)
            
            # Mostrar trayectoria
            if len(history) <= 20:
                for i, xi in enumerate(history[:10]):
                    ax.plot(xi, f(xi), 'go' if i == 0 else 'yo', markersize=8)
                    if i < len(history) - 1:
                        # Tangente
                        xi_next = history[i+1]
                        ax.plot([xi, xi_next], [f(xi), 0], 'g--', alpha=0.5, linewidth=1)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('x', fontsize=10, fontweight='bold')
            ax.set_ylabel('f(x)', fontsize=10, fontweight='bold')
            ax.set_title(f'x₀ = {x0:.2f}', fontsize=11, fontweight='bold')
            ax.set_ylim([-0.5, 2.5])
    
    # Remover subplot extra
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('exercise_2_62_newton_attempts.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# RESUMEN Y CONCLUSIONES
# ============================================================================

def final_summary():
    """
    Resumen final del ejercicio
    """
    print("\n" + "="*100)
    print("RESUMEN FINAL")
    print("="*100)
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        FUNCIÓN: f(x) = cos(x) + 1                            ║
║                        RAÍCES: x = π + 2πn (n entero)                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ MÉTODO 1: BISECCIÓN                                                          │
├──────────────────────────────────────────────────────────────────────────────┤
│ ¿PUEDE USARSE?  ❌ NO                                                        │
│                                                                              │
│ RAZÓN:                                                                       │
│   • Requiere f(a) · f(b) < 0 (cambio de signo)                             │
│   • f(x) = cos(x) + 1 ≥ 0 SIEMPRE (nunca negativa)                         │
│   • La función NO cruza el eje x, solo lo toca                             │
│   • Imposible encontrar intervalo con cambio de signo                      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ MÉTODO 2: REGULA FALSI                                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ ¿PUEDE USARSE?  ❌ NO                                                        │
│                                                                              │
│ RAZÓN:                                                                       │
│   • Mismo requisito que Bisección: f(a) · f(b) < 0                         │
│   • Por la misma razón, NO puede usarse                                    │
│   • Aunque usa interpolación lineal, necesita cambio de signo              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ MÉTODO 3: NEWTON-RAPHSON                                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│ ¿PUEDE USARSE?  ⚠️ MUY PROBLEMÁTICO / INESTABLE                             │
│                                                                              │
│ RAZÓN:                                                                       │
│   • f'(x) = -sin(x) = 0 en TODAS las raíces                                │
│   • Tangente horizontal en las raíces                                      │
│   • División por cero o números muy pequeños                               │
│   • Raíces de multiplicidad 2 (f = f' = 0)                                │
│                                                                              │
│ COMPORTAMIENTO:                                                              │
│   • Puede diverger                                                          │
│   • Puede dar saltos enormes                                               │
│   • Convergencia MUY lenta si converge                                     │
│   • Inestabilidad numérica severa                                          │
└──────────────────────────────────────────────────────────────────────────────┘

╔══════════════════════════════════════════════════════════════════════════════╗
║                          CONCLUSIÓN GENERAL                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NINGUNO de los tres métodos es apropiado para esta función.                ║
║                                                                              ║
║  El problema fundamental es que f(x) = cos(x) + 1 tiene RAÍCES MÚLTIPLES    ║
║  (de orden 2) donde tanto f(x) = 0 como f'(x) = 0.                         ║
║                                                                              ║
║  Métodos que podrían funcionar mejor:                                       ║
║    • Métodos especializados para raíces múltiples                          ║
║    • Newton modificado: xₙ₊₁ = xₙ - m·f(xₙ)/f'(xₙ) con m=2                ║
║    • Deflación de raíces                                                    ║
║    • Métodos de optimización (minimizar |f(x)|)                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


# ============================================================================
# LECCIÓN GENERAL
# ============================================================================

def general_lesson():
    """
    Lección general sobre métodos numéricos
    """
    print("\n" + "="*100)
    print("LECCIÓN IMPORTANTE")
    print("="*100)
    
    print("""
LECCIÓN IMPORTANTE DE ESTE EJERCICIO:

1. NO TODOS LOS MÉTODOS FUNCIONAN PARA TODAS LAS FUNCIONES
   Los métodos tienen REQUISITOS específicos que deben cumplirse.

2. ENTENDER LA FUNCIÓN ES CRUCIAL
   Antes de aplicar un método, analiza:
   • ¿Dónde están las raíces?
   • ¿Cómo se comporta la función cerca de las raíces?
   • ¿La función cruza o solo toca el eje x?
   • ¿Qué pasa con las derivadas en las raíces?

3. RAÍCES MÚLTIPLES SON PROBLEMÁTICAS
   Cuando f(x) = f'(x) = 0 en la raíz:
   • Bisección y Regula Falsi fallan (no hay cambio de signo)
   • Newton es inestable o muy lento
   • Se necesitan métodos especiales

4. SIEMPRE VERIFICA LOS REQUISITOS
   • Bisección/Regula Falsi: f(a)·f(b) < 0 ✓
   • Newton: f'(x) ≠ 0 en la raíz ✓
   • Secante: Similar a Newton ✓

5. LA TEORÍA IMPORTA
   Entender la matemática detrás de los métodos te ayuda a
   identificar cuándo funcionarán y cuándo no.
    """)


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("="*100)
    print("EJERCICIO 2.62: ¿PUEDEN LOS MÉTODOS ENCONTRAR RAÍCES DE f(x) = cos(x) + 1?")
    print("="*100)
    
    # Análisis de la función
    analyze_function()
    visualize_function()
    
    # Análisis método por método
    analyze_bisection()
    analyze_regula_falsi()
    analyze_newton()
    
    # Demostración práctica
    demonstrate_newton_failure()
    
    # Resumen
    final_summary()
    
    # Lección general
    general_lesson()
    
    print("\n" + "="*100)
    print("✓ EJERCICIO 2.62 COMPLETADO")
    print("="*100)