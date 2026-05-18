"""
================================================================================
MÉTODOS NUMÉRICOS - EJERCICIOS 2.51 A 2.61
Implementación en Python de métodos de búsqueda de raíces
================================================================================

Autor: Código generado para ejercicios de Métodos Numéricos
Fecha: 2026-02-13

Este archivo contiene las implementaciones de:
- Ejercicio 2.51: Método de Bisección
- Ejercicio 2.52: Método de Regula Falsi (Falsa Posición)
- Ejercicio 2.53: Método de Newton-Raphson
- Ejercicio 2.54: Método de la Secante
- Ejercicio 2.55: Cálculo de iteraciones para Bisección
- Ejercicio 2.56: Análisis de x³ - 3 = 0
- Ejercicio 2.57: Demostración de todos los métodos
- Ejercicio 2.58: Análisis de orden de convergencia
- Ejercicio 2.59: Predicción de errores con Newton
- Ejercicio 2.60: Método de Taylor de orden 2
- Ejercicio 2.61: Caída con resistencia del aire
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Optional
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# Configuración de gráficas
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['lines.linewidth'] = 2

# ============================================================================
# EJERCICIO 2.51: MÉTODO DE BISECCIÓN
# ============================================================================

def bisection(f: Callable, a: float, b: float, tol: float = 1e-6, 
              max_iter: int = 100, verbose: bool = False) -> Tuple[float, int, List]:
    """
    Método de Bisección para encontrar raíces de funciones.
    
    Parámetros:
    -----------
    f : Callable
        Función cuya raíz se busca
    a : float
        Extremo izquierdo del intervalo inicial
    b : float
        Extremo derecho del intervalo inicial
    tol : float
        Tolerancia para el criterio de paro
    max_iter : int
        Número máximo de iteraciones
    verbose : bool
        Si True, imprime información de cada iteración
        
    Retorna:
    --------
    root : float
        Aproximación de la raíz
    iterations : int
        Número de iteraciones realizadas
    history : List
        Historia de iteraciones [iteración, a, b, c, f(c), error]
    """
    
    # Verificar condiciones iniciales
    if f(a) * f(b) > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")
    
    history = []
    
    if verbose:
        print("="*80)
        print("MÉTODO DE BISECCIÓN")
        print("="*80)
        print(f"{'Iter':<6} {'a':<12} {'b':<12} {'c':<12} {'f(c)':<12} {'Error':<12}")
        print("-"*80)
    
    for k in range(max_iter):
        # Calcular punto medio
        c = (a + b) / 2
        fc = f(c)
        error = (b - a) / 2
        
        history.append([k, a, b, c, fc, error])
        
        if verbose:
            print(f"{k:<6} {a:<12.8f} {b:<12.8f} {c:<12.8f} {fc:<12.2e} {error:<12.2e}")
        
        # Criterio de paro
        if abs(fc) < tol or error < tol:
            if verbose:
                print("-"*80)
                print(f"Convergencia alcanzada en {k+1} iteraciones")
                print(f"Raíz aproximada: {c:.10f}")
                print(f"f(raíz) = {fc:.2e}")
            return c, k+1, history
        
        # Actualizar intervalo
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    print(f"Advertencia: No convergió en {max_iter} iteraciones")
    return c, max_iter, history


# ============================================================================
# EJERCICIO 2.52: MÉTODO DE REGULA FALSI
# ============================================================================

def regula_falsi(f: Callable, a: float, b: float, tol: float = 1e-6,
                 max_iter: int = 100, verbose: bool = False) -> Tuple[float, int, List]:
    """
    Método de Regula Falsi (Falsa Posición) para encontrar raíces.
    
    Parámetros:
    -----------
    f : Callable
        Función cuya raíz se busca
    a : float
        Extremo izquierdo del intervalo inicial
    b : float
        Extremo derecho del intervalo inicial
    tol : float
        Tolerancia para el criterio de paro
    max_iter : int
        Número máximo de iteraciones
    verbose : bool
        Si True, imprime información de cada iteración
        
    Retorna:
    --------
    root : float
        Aproximación de la raíz
    iterations : int
        Número de iteraciones realizadas
    history : List
        Historia de iteraciones
    """
    
    # Verificar condiciones iniciales
    if f(a) * f(b) > 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos")
    
    history = []
    
    if verbose:
        print("="*80)
        print("MÉTODO DE REGULA FALSI")
        print("="*80)
        print(f"{'Iter':<6} {'a':<12} {'b':<12} {'c':<12} {'f(c)':<12} {'Error':<12}")
        print("-"*80)
    
    c_prev = a
    
    for k in range(max_iter):
        fa = f(a)
        fb = f(b)
        
        # Calcular punto de intersección de la secante
        c = b - fb * (b - a) / (fb - fa)
        fc = f(c)
        error = abs(c - c_prev)
        
        history.append([k, a, b, c, fc, error])
        
        if verbose:
            print(f"{k:<6} {a:<12.8f} {b:<12.8f} {c:<12.8f} {fc:<12.2e} {error:<12.2e}")
        
        # Criterio de paro
        if abs(fc) < tol or error < tol:
            if verbose:
                print("-"*80)
                print(f"Convergencia alcanzada en {k+1} iteraciones")
                print(f"Raíz aproximada: {c:.10f}")
                print(f"f(raíz) = {fc:.2e}")
            return c, k+1, history
        
        # Actualizar intervalo
        if fa * fc < 0:
            b = c
        else:
            a = c
        
        c_prev = c
    
    print(f"Advertencia: No convergió en {max_iter} iteraciones")
    return c, max_iter, history


# ============================================================================
# EJERCICIO 2.53: MÉTODO DE NEWTON
# ============================================================================

def newton(f: Callable, fprime: Callable, x0: float, tol: float = 1e-6,
           max_iter: int = 100, verbose: bool = False) -> Tuple[float, int, List]:
    """
    Método de Newton-Raphson para encontrar raíces.
    
    Parámetros:
    -----------
    f : Callable
        Función cuya raíz se busca
    fprime : Callable
        Derivada de la función f
    x0 : float
        Estimación inicial
    tol : float
        Tolerancia para el criterio de paro
    max_iter : int
        Número máximo de iteraciones
    verbose : bool
        Si True, imprime información de cada iteración
        
    Retorna:
    --------
    root : float
        Aproximación de la raíz
    iterations : int
        Número de iteraciones realizadas
    history : List
        Historia de iteraciones
    """
    
    history = []
    x = x0
    
    if verbose:
        print("="*80)
        print("MÉTODO DE NEWTON-RAPHSON")
        print("="*80)
        header = "{:<6} {:<15} {:<15} {:<15} {:<15}".format('Iter', 'x', 'f(x)', "f'(x)", 'Error')
        print(header)
        print("-"*80)
    
    for k in range(max_iter):
        fx = f(x)
        fpx = fprime(x)
        
        if abs(fpx) < 1e-15:
            print(f"Advertencia: f'(x) ≈ 0 en x = {x}")
            return x, k, history
        
        x_new = x - fx / fpx
        error = abs(x_new - x)
        
        history.append([k, x, fx, fpx, error])
        
        if verbose:
            print(f"{k:<6} {x:<15.10f} {fx:<15.2e} {fpx:<15.2e} {error:<15.2e}")
        
        # Criterio de paro
        if abs(fx) < tol or error < tol:
            if verbose:
                print("-"*80)
                print(f"Convergencia alcanzada en {k+1} iteraciones")
                print(f"Raíz aproximada: {x_new:.10f}")
                print(f"f(raíz) = {f(x_new):.2e}")
            return x_new, k+1, history
        
        x = x_new
    
    print(f"Advertencia: No convergió en {max_iter} iteraciones")
    return x, max_iter, history


# ============================================================================
# EJERCICIO 2.54: MÉTODO DE LA SECANTE
# ============================================================================

def secant(f: Callable, x0: float, x1: float, tol: float = 1e-6,
           max_iter: int = 100, verbose: bool = False) -> Tuple[float, int, List]:
    """
    Método de la Secante para encontrar raíces.
    
    Parámetros:
    -----------
    f : Callable
        Función cuya raíz se busca
    x0 : float
        Primera estimación inicial
    x1 : float
        Segunda estimación inicial
    tol : float
        Tolerancia para el criterio de paro
    max_iter : int
        Número máximo de iteraciones
    verbose : bool
        Si True, imprime información de cada iteración
        
    Retorna:
    --------
    root : float
        Aproximación de la raíz
    iterations : int
        Número de iteraciones realizadas
    history : List
        Historia de iteraciones
    """
    
    history = []
    
    if verbose:
        print("="*80)
        print("MÉTODO DE LA SECANTE")
        print("="*80)
        print(f"{'Iter':<6} {'x_k-1':<15} {'x_k':<15} {'f(x_k)':<15} {'Error':<15}")
        print("-"*80)
    
    for k in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        
        if abs(fx1 - fx0) < 1e-15:
            print(f"Advertencia: f(x1) - f(x0) ≈ 0")
            return x1, k, history
        
        # Calcular siguiente aproximación
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        error = abs(x2 - x1)
        
        history.append([k, x0, x1, fx1, error])
        
        if verbose:
            print(f"{k:<6} {x0:<15.10f} {x1:<15.10f} {fx1:<15.2e} {error:<15.2e}")
        
        # Criterio de paro
        if abs(fx1) < tol or error < tol:
            if verbose:
                print("-"*80)
                print(f"Convergencia alcanzada en {k+1} iteraciones")
                print(f"Raíz aproximada: {x2:.10f}")
                print(f"f(raíz) = {f(x2):.2e}")
            return x2, k+1, history
        
        # Actualizar para siguiente iteración
        x0, x1 = x1, x2
    
    print(f"Advertencia: No convergió en {max_iter} iteraciones")
    return x1, max_iter, history


# ============================================================================
# EJERCICIO 2.55: ITERACIONES NECESARIAS PARA BISECCIÓN
# ============================================================================

def bisection_iterations_required(a: float, b: float, epsilon: float) -> int:
    """
    Calcula el número de iteraciones necesarias para alcanzar una tolerancia
    específica usando el método de bisección.
    
    Parámetros:
    -----------
    a : float
        Extremo izquierdo del intervalo
    b : float
        Extremo derecho del intervalo
    epsilon : float
        Tolerancia deseada
        
    Retorna:
    --------
    n : int
        Número de iteraciones necesarias
    """
    n = np.ceil(np.log2((b - a) / epsilon) - 1)
    return int(n)


def exercise_2_55():
    """
    Ejercicio 2.55: Calcula iteraciones necesarias para aproximar √3
    con diferentes tolerancias usando bisección en [0, 2].
    """
    print("\n" + "="*80)
    print("EJERCICIO 2.55: ITERACIONES NECESARIAS PARA BISECCIÓN")
    print("="*80)
    print(f"Aproximar √3 en el intervalo [0, 2]\n")
    
    a, b = 0, 2
    tolerances = [10**(-k) for k in range(3, 16)]
    
    results = []
    for eps in tolerances:
        n = bisection_iterations_required(a, b, eps)
        results.append([eps, n])
    
    # Crear tabla
    df = pd.DataFrame(results, columns=['Tolerancia ε', 'Iteraciones n'])
    print(df.to_string(index=False))
    
    # Crear gráfica
    plt.figure(figsize=(10, 6))
    plt.semilogy([r[1] for r in results], [r[0] for r in results], 'bo-', linewidth=2)
    plt.xlabel('Número de Iteraciones', fontsize=12)
    plt.ylabel('Tolerancia ε', fontsize=12)
    plt.title('Iteraciones Requeridas vs Tolerancia (Método de Bisección)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/claude/exercise_2_55.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfica guardada como 'exercise_2_55.png'")
    
    return results


# ============================================================================
# EJERCICIO 2.56: ANÁLISIS DE x³ - 3 = 0
# ============================================================================

def exercise_2_56():
    """
    Ejercicio 2.56: Análisis completo de la ecuación x³ - 3 = 0
    con todos los métodos y generación de gráficas.
    """
    print("\n" + "="*80)
    print("EJERCICIO 2.56: ANÁLISIS DE x³ - 3 = 0")
    print("="*80)
    
    # Definir funciones
    f = lambda x: x**3 - 3
    fprime = lambda x: 3*x**2
    root_exact = 3**(1/3)
    
    print(f"\nRaíz exacta: {root_exact:.15f}\n")
    
    # 1. Gráfica de la función
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Función
    plt.subplot(2, 2, 1)
    x = np.linspace(0, 2.5, 1000)
    y = f(x)
    plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x³ - 3')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=root_exact, color='r', linestyle='--', alpha=0.5, label=f'Raíz = {root_exact:.4f}')
    plt.plot(root_exact, 0, 'ro', markersize=10)
    plt.xlabel('x', fontsize=11)
    plt.ylabel('f(x)', fontsize=11)
    plt.title('Función f(x) = x³ - 3', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Comparación de convergencia
    methods_data = {}
    
    # Bisección
    _, _, hist_bis = bisection(f, 0, 2, tol=1e-12, verbose=False)
    errors_bis = [abs(h[3] - root_exact) for h in hist_bis]
    methods_data['Bisección'] = errors_bis
    
    # Regula Falsi
    _, _, hist_rf = regula_falsi(f, 0, 2, tol=1e-12, verbose=False)
    errors_rf = [abs(h[3] - root_exact) for h in hist_rf]
    methods_data['Regula Falsi'] = errors_rf
    
    # Newton
    _, _, hist_newton = newton(f, fprime, 1.5, tol=1e-12, verbose=False)
    errors_newton = [abs(h[1] - root_exact) for h in hist_newton]
    methods_data['Newton'] = errors_newton
    
    # Secante
    _, _, hist_secant = secant(f, 1, 2, tol=1e-12, verbose=False)
    errors_secant = [abs(h[2] - root_exact) for h in hist_secant]
    methods_data['Secante'] = errors_secant
    
    # Subplot 2: Convergencia (escala log)
    plt.subplot(2, 2, 2)
    for method, errors in methods_data.items():
        plt.semilogy(range(len(errors)), errors, 'o-', linewidth=2, label=method)
    plt.xlabel('Iteración', fontsize=11)
    plt.ylabel('Error Absoluto |xₖ - x*|', fontsize=11)
    plt.title('Convergencia de Métodos (Escala Log)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Iteraciones de Newton (visualización geométrica)
    plt.subplot(2, 2, 3)
    x_plot = np.linspace(0.5, 2.5, 1000)
    plt.plot(x_plot, f(x_plot), 'b-', linewidth=2, label='f(x)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Mostrar las primeras 4 iteraciones de Newton
    x_current = 2.0
    colors = ['red', 'green', 'orange', 'purple']
    for i in range(4):
        fx = f(x_current)
        fpx = fprime(x_current)
        x_new = x_current - fx/fpx
        
        # Dibujar tangente
        x_tangent = np.linspace(x_current - 0.5, x_current + 0.5, 100)
        y_tangent = fx + fpx * (x_tangent - x_current)
        plt.plot(x_tangent, y_tangent, '--', color=colors[i], alpha=0.6, linewidth=1.5)
        plt.plot(x_current, fx, 'o', color=colors[i], markersize=8, label=f'Iter {i}')
        plt.plot([x_new, x_new], [0, f(x_new)], ':', color=colors[i], alpha=0.5)
        
        x_current = x_new
    
    plt.plot(root_exact, 0, 'ro', markersize=10, label='Raíz exacta')
    plt.xlabel('x', fontsize=11)
    plt.ylabel('f(x)', fontsize=11)
    plt.title('Iteraciones del Método de Newton', fontsize=12, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.5, 2.5)
    
    # Subplot 4: Análisis de orden (log-log)
    plt.subplot(2, 2, 4)
    
    # Newton: log-log plot
    if len(errors_newton) > 2:
        log_old = np.log10(errors_newton[:-1])
        log_new = np.log10(errors_newton[1:])
        plt.plot(log_old, log_new, 'ro', markersize=8, label='Newton')
        
        # Ajuste lineal
        coeffs = np.polyfit(log_old, log_new, 1)
        plt.plot(log_old, np.polyval(coeffs, log_old), 'r-', alpha=0.6, 
                label=f'Newton (pendiente={coeffs[0]:.2f})')
    
    # Secante: log-log plot
    if len(errors_secant) > 2:
        log_old = np.log10(errors_secant[:-1])
        log_new = np.log10(errors_secant[1:])
        plt.plot(log_old, log_new, 'go', markersize=8, label='Secante')
        
        coeffs = np.polyfit(log_old, log_new, 1)
        plt.plot(log_old, np.polyval(coeffs, log_old), 'g-', alpha=0.6,
                label=f'Secante (pendiente={coeffs[0]:.2f})')
    
    plt.xlabel('log₁₀(error anterior)', fontsize=11)
    plt.ylabel('log₁₀(error nuevo)', fontsize=11)
    plt.title('Análisis de Orden de Convergencia', fontsize=12, fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/exercise_2_56.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nResumen de resultados:")
    print("-" * 60)
    print(f"{'Método':<20} {'Iteraciones':<15} {'Raíz encontrada':<20}")
    print("-" * 60)
    print(f"{'Bisección':<20} {len(hist_bis):<15} {hist_bis[-1][3]:.15f}")
    print(f"{'Regula Falsi':<20} {len(hist_rf):<15} {hist_rf[-1][3]:.15f}")
    print(f"{'Newton':<20} {len(hist_newton):<15} {hist_newton[-1][1]:.15f}")
    print(f"{'Secante':<20} {len(hist_secant):<15} {hist_secant[-1][2]:.15f}")
    print(f"{'Exacta':<20} {'-':<15} {root_exact:.15f}")
    print("-" * 60)
    
    print(f"\nGráficas guardadas como 'exercise_2_56.png'")


# ============================================================================
# EJERCICIO 2.57: DEMOSTRACIÓN DE TODOS LOS MÉTODOS
# ============================================================================

def exercise_2_57():
    """
    Ejercicio 2.57: Resolver 3sin(x) + 9 = x² - cos(x)
    usando todos los métodos implementados.
    """
    print("\n" + "="*80)
    print("EJERCICIO 2.57: ECUACIÓN 3sin(x) + 9 = x² - cos(x)")
    print("="*80)
    
    # Reformular como f(x) = 0
    f = lambda x: 3*np.sin(x) + np.cos(x) + 9 - x**2
    fprime = lambda x: 3*np.cos(x) - np.sin(x) - 2*x
    
    print("\nReformulación: f(x) = 3sin(x) + cos(x) + 9 - x² = 0")
    print("Buscamos la solución positiva.\n")
    
    # Análisis preliminar
    print("Análisis preliminar:")
    print(f"f(0) = {f(0):.4f} > 0")
    print(f"f(4) = {f(4):.4f} < 0")
    print("→ Existe una raíz en (0, 4)\n")
    
    results = {}
    
    # Método 1: Bisección
    print("="*80)
    print("MÉTODO DE BISECCIÓN")
    print("="*80)
    root_bis, iter_bis, _ = bisection(f, a=0, b=4, tol=1e-10, verbose=True)
    results['Bisección'] = (root_bis, iter_bis)
    
    # Método 2: Regula Falsi
    print("\n" + "="*80)
    print("MÉTODO DE REGULA FALSI")
    print("="*80)
    root_rf, iter_rf, _ = regula_falsi(f, a=0, b=4, tol=1e-10, verbose=True)
    results['Regula Falsi'] = (root_rf, iter_rf)
    
    # Método 3: Newton
    print("\n" + "="*80)
    print("MÉTODO DE NEWTON")
    print("="*80)
    root_newton, iter_newton, _ = newton(f, fprime, x0=2, tol=1e-10, verbose=True)
    results['Newton'] = (root_newton, iter_newton)
    
    # Método 4: Secante
    print("\n" + "="*80)
    print("MÉTODO DE LA SECANTE")
    print("="*80)
    root_secant, iter_secant, _ = secant(f, x0=0, x1=4, tol=1e-10, verbose=True)
    results['Secante'] = (root_secant, iter_secant)
    
    # Resumen comparativo
    print("\n" + "="*80)
    print("RESUMEN COMPARATIVO")
    print("="*80)
    print(f"{'Método':<20} {'Raíz encontrada':<20} {'Iteraciones':<12} {'f(raíz)':<15}")
    print("-"*80)
    for method, (root, iters) in results.items():
        print(f"{method:<20} {root:<20.15f} {iters:<12} {f(root):<15.2e}")
    print("="*80)
    
    # Crear gráfica de la función
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    x = np.linspace(0, 4, 1000)
    y = f(x)
    plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Marcar las raíces encontradas
    colors = {'Bisección': 'red', 'Regula Falsi': 'green', 'Newton': 'orange', 'Secante': 'purple'}
    for method, (root, _) in results.items():
        plt.plot(root, 0, 'o', color=colors[method], markersize=10, label=f'{method}: {root:.4f}')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('f(x) = 3sin(x) + cos(x) + 9 - x²', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Zoom cerca de la raíz
    plt.subplot(1, 2, 2)
    root_avg = np.mean([r[0] for r in results.values()])
    x_zoom = np.linspace(root_avg - 0.1, root_avg + 0.1, 1000)
    y_zoom = f(x_zoom)
    plt.plot(x_zoom, y_zoom, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    for method, (root, _) in results.items():
        plt.plot(root, 0, 'o', color=colors[method], markersize=10, label=method)
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('Zoom cerca de la raíz', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/exercise_2_57.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfica guardada como 'exercise_2_57.png'")
    
    return results


# ============================================================================
# EJERCICIO 2.58: ANÁLISIS DE ORDEN DE CONVERGENCIA
# ============================================================================

def exercise_2_58():
    """
    Ejercicio 2.58: Análisis de orden de convergencia.
    Crea gráficas log-log para determinar el orden de diferentes métodos.
    """
    print("\n" + "="*80)
    print("EJERCICIO 2.58: ANÁLISIS DE ORDEN DE CONVERGENCIA")
    print("="*80)
    
    # Parte (a): Transformación logarítmica
    print("\nParte (a): Transformación logarítmica")
    print("-" * 60)
    print("Dada: |x_{k+1} - x*| ≤ C|x_k - x*|^M")
    print("\nTomando logaritmo de ambos lados:")
    print("log(|x_{k+1} - x*|) ≤ log(C) + M·log(|x_k - x*|)")
    print("\nRespuesta:")
    print("log(|x_{k+1} - x*|) = M·log(|x_k - x*|) + log(C)")
    
    # Parte (b): Pendiente
    print("\nParte (b): Pendiente en gráfica log-log")
    print("-" * 60)
    print("La ecuación es lineal de la forma y = mx + b")
    print("donde la pendiente m = M (el orden de convergencia)")
    print("\nRespuesta: La pendiente es M")
    
    # Parte (c): Crear gráficas de diferentes métodos
    print("\nParte (c): Gráficas log-log para diferentes métodos")
    print("-" * 60)
    
    # Función de prueba
    f = lambda x: x**3 - 3
    fprime = lambda x: 3*x**2
    fsecond = lambda x: 6*x
    root_exact = 3**(1/3)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Análisis de Orden de Convergencia (Gráficas Log-Log)', 
                 fontsize=16, fontweight='bold')
    
    methods = [
        ('Bisección', lambda: bisection(f, 0, 2, tol=1e-14, verbose=False)),
        ('Regula Falsi', lambda: regula_falsi(f, 0, 2, tol=1e-14, verbose=False)),
        ('Newton', lambda: newton(f, fprime, 1.5, tol=1e-14, verbose=False)),
        ('Secante', lambda: secant(f, 1, 2, tol=1e-14, verbose=False)),
    ]
    
    # Añadir método de punto fijo y Taylor de orden 2
    def fixed_point_iteration(g, x0, tol=1e-14, max_iter=100):
        """Método de punto fijo: x = g(x)"""
        history = []
        x = x0
        for k in range(max_iter):
            x_new = g(x)
            error = abs(x_new - x)
            history.append([k, x, error])
            if error < tol:
                return x_new, k+1, history
            x = x_new
        return x, max_iter, history
    
    # Para x³ = 3, podemos usar g(x) = ∛3 = x^(1/3) * 3^(1/3), pero mejor usar g(x) = (3/x²)^(1/3)
    # Para simplicidad, usamos g(x) = ∛(3) como aproximación iterativa
    g = lambda x: (3 / x**2)**(1/3) if x != 0 else 1.5
    methods.append(('Punto Fijo', lambda: fixed_point_iteration(g, 1.5, tol=1e-14)))
    
    # Método de Taylor orden 2
    def taylor_order2_simple(f, fp, fpp, x0, tol=1e-14, max_iter=50):
        """Método de Taylor de orden 2"""
        history = []
        x = x0
        for k in range(max_iter):
            fx = f(x)
            fpx = fp(x)
            fppx = fpp(x)
            
            if abs(fppx) < 1e-15:
                return x, k, history
            
            disc = fpx**2 - 2*fppx*fx
            if disc < 0:
                return x, k, history
            
            sqrt_disc = np.sqrt(disc)
            if fpx >= 0:
                h = (-fpx - sqrt_disc) / fppx
            else:
                h = (-fpx + sqrt_disc) / fppx
            
            x_new = x + h
            error = abs(x_new - x)
            history.append([k, x, error])
            
            if error < tol or abs(fx) < tol:
                return x_new, k+1, history
            x = x_new
        return x, max_iter, history
    
    methods.append(('Taylor Orden 2', lambda: taylor_order2_simple(f, fprime, fsecond, 1.5, tol=1e-14)))
    
    for idx, (method_name, method_func) in enumerate(methods):
        ax = axes[idx // 3, idx % 3]
        
        try:
            _, _, history = method_func()
            
            # Calcular errores
            if method_name in ['Bisección', 'Regula Falsi']:
                errors = [abs(h[3] - root_exact) for h in history]
            elif method_name in ['Newton', 'Punto Fijo', 'Taylor Orden 2']:
                errors = [abs(h[1] - root_exact) for h in history]
            else:  # Secante
                errors = [abs(h[2] - root_exact) for h in history]
            
            # Filtrar errores muy pequeños
            errors = [e for e in errors if e > 1e-15]
            
            if len(errors) > 2:
                log_old = np.log10(errors[:-1])
                log_new = np.log10(errors[1:])
                
                # Graficar
                ax.plot(log_old, log_new, 'bo', markersize=8, label='Datos')
                
                # Ajuste lineal
                coeffs = np.polyfit(log_old, log_new, 1)
                slope = coeffs[0]
                
                ax.plot(log_old, np.polyval(coeffs, log_old), 'r-', linewidth=2,
                       label=f'Ajuste (M ≈ {slope:.3f})')
                
                ax.set_xlabel('log₁₀(|x_k - x*|)', fontsize=10)
                ax.set_ylabel('log₁₀(|x_{k+1} - x*|)', fontsize=10)
                ax.set_title(f'{method_name}\nOrden ≈ {slope:.3f}', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                
                print(f"{method_name:<20} Orden de convergencia ≈ {slope:.3f}")
        except Exception as e:
            ax.text(0.5, 0.5, f'Error en {method_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            print(f"{method_name:<20} Error: {str(e)}")
    
    # Ocultar el último subplot si hay menos de 6 métodos
    if len(methods) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/exercise_2_58_part_c.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Parte (d): Interpretación
    print("\nParte (d): Interpretación del orden de convergencia")
    print("-" * 60)
    print("\nCONVERGENCIA DE PRIMER ORDEN (Lineal):")
    print("  • El error se reduce por un factor constante en cada iteración")
    print("  • Ejemplo: Bisección reduce el error a la mitad (C = 0.5)")
    print("  • Se ganan aproximadamente -log₁₀(C) dígitos por iteración")
    
    print("\nCONVERGENCIA DE SEGUNDO ORDEN (Cuadrática):")
    print("  • El error se eleva al cuadrado en cada iteración")
    print("  • Ejemplo: Método de Newton para raíces simples")
    print("  • Los dígitos correctos aproximadamente se duplican por iteración")
    print("  • Secuencia típica: 1 → 2 → 4 → 8 → 16 dígitos correctos")
    
    print("\nCONVERGENCIA SUPERLINEAL:")
    print("  • Orden entre 1 y 2")
    print("  • Ejemplo: Método de la Secante (orden ≈ 1.618, número áureo φ)")
    print("  • Más rápido que lineal pero más lento que cuadrático")
    
    print("\nCONVERGENCIA DE ORDEN SUPERIOR (M > 2):")
    print("  • El error se reduce extremadamente rápido")
    print("  • Ejemplo: Métodos de Taylor de orden superior")
    print("  • Requieren derivadas de orden superior")
    print("  • Costo computacional alto por iteración")
    
    print(f"\nGráficas guardadas como 'exercise_2_58_part_c.png'")


# ============================================================================
# EJERCICIO 2.59: PREDICCIÓN DE ERRORES CON NEWTON
# ============================================================================

def exercise_2_59():
    """
    Ejercicio 2.59: Predicción de errores usando convergencia cuadrática.
    """
    print("\n" + "="*80)
    print("EJERCICIO 2.59: PREDICCIÓN DE ERRORES CON NEWTON")
    print("="*80)
    
    print("\nDato: Error después del paso 1: |x₁ - x*| = 0.2")
    print("\nPara el método de Newton con convergencia cuadrática:")
    print("|e_{k+1}| ≈ C·|e_k|²")
    print("\nAsumiendo C ≈ 1 (típico para muchas funciones bien comportadas):\n")
    
    e1 = 0.2
    C = 1.0
    
    errors = [e1]
    
    print(f"{'Paso':<8} {'Error e_k':<20} {'Notación Científica':<25} {'Dígitos Correctos':<20}")
    print("-" * 80)
    
    for k in range(1, 5):
        e_prev = errors[-1]
        e_new = C * e_prev**2
        errors.append(e_new)
        
        digits = -np.log10(e_new) if e_new > 0 else float('inf')
        
        print(f"{k:<8} {e_new:<20.2e} {e_new:<25.2e} {digits:<20.1f}")
    
    print("-" * 80)
    
    print("\nRESPUESTAS:")
    print(f"  Paso 2: Error ≈ {errors[1]:.2e} = 0.04")
    print(f"  Paso 3: Error ≈ {errors[2]:.2e} = 0.0016")
    print(f"  Paso 4: Error ≈ {errors[3]:.2e} ≈ 2.56×10⁻⁶")
    
    print("\nOBSERVACIÓN:")
    print("  • Los dígitos correctos aproximadamente se duplican en cada iteración")
    print("  • Paso 1: <1 dígito → Paso 2: ~1 dígito → Paso 3: ~3 dígitos → Paso 4: ~6 dígitos")
    print("  • Esta duplicación es característica de la convergencia cuadrática")
    
    # Crear gráfica
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    steps = list(range(1, 5))
    plt.semilogy(steps, errors[:-1], 'bo-', markersize=10, linewidth=2)
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Error Absoluto', fontsize=12)
    plt.title('Reducción del Error (Método de Newton)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, (step, err) in enumerate(zip(steps, errors[:-1])):
        plt.text(step, err, f'{err:.2e}', ha='right', va='bottom', fontsize=9)
    
    plt.subplot(1, 2, 2)
    digits = [-np.log10(e) for e in errors[:-1]]
    plt.plot(steps, digits, 'go-', markersize=10, linewidth=2)
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Dígitos Correctos Aproximados', fontsize=12)
    plt.title('Crecimiento de Dígitos Correctos', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    for i, (step, dig) in enumerate(zip(steps, digits)):
        plt.text(step, dig, f'{dig:.1f}', ha='left', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/home/claude/exercise_2_59.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfica guardada como 'exercise_2_59.png'")


# ============================================================================
# EJERCICIO 2.60: MÉTODO DE TAYLOR DE ORDEN 2
# ============================================================================

def taylor_order2(f: Callable, fprime: Callable, fsecond: Callable,
                  x0: float, tol: float = 1e-6, max_iter: int = 100,
                  verbose: bool = False) -> Tuple[float, int, List]:
    """
    Método de Taylor de orden 2 para encontrar raíces.
    
    Resuelve: 0 = f(x₀) + f'(x₀)(x-x₀) + (f''(x₀)/2)(x-x₀)²
    
    Parámetros:
    -----------
    f : Callable
        Función cuya raíz se busca
    fprime : Callable
        Primera derivada de f
    fsecond : Callable
        Segunda derivada de f
    x0 : float
        Estimación inicial
    tol : float
        Tolerancia
    max_iter : int
        Número máximo de iteraciones
    verbose : bool
        Imprimir detalles
        
    Retorna:
    --------
    root : float
        Aproximación de la raíz
    iterations : int
        Número de iteraciones
    history : List
        Historia de iteraciones
    """
    
    history = []
    x = x0
    
    if verbose:
        print("="*80)
        print("MÉTODO DE TAYLOR DE ORDEN 2")
        print("="*80)
        header = "{:<6} {:<15} {:<15} {:<15} {:<15} {:<15}".format('Iter', 'x', 'f(x)', "f'(x)", "f''(x)", 'Error')
        print(header)
        print("-"*80)
    
    for k in range(max_iter):
        fx = f(x)
        fpx = fprime(x)
        fppx = fsecond(x)
        
        if abs(fppx) < 1e-15:
            if verbose:
                print(f"Advertencia: f''(x) ≈ 0 en iteración {k}")
            return x, k, history
        
        # Calcular discriminante
        discriminant = fpx**2 - 2*fppx*fx
        
        if discriminant < 0:
            if verbose:
                print(f"Advertencia: Discriminante negativo en iteración {k}")
            return x, k, history
        
        sqrt_discriminant = np.sqrt(discriminant)
        
        # Elegir el signo que minimiza |h|
        # Usamos la fórmula estable numéricamente
        if fpx >= 0:
            h = (-fpx - sqrt_discriminant) / fppx
        else:
            h = (-fpx + sqrt_discriminant) / fppx
        
        x_new = x + h
        error = abs(h)
        
        history.append([k, x, fx, fpx, fppx, error])
        
        if verbose:
            print(f"{k:<6} {x:<15.10f} {fx:<15.2e} {fpx:<15.2e} {fppx:<15.2e} {error:<15.2e}")
        
        # Criterio de paro
        if abs(fx) < tol or error < tol:
            if verbose:
                print("-"*80)
                print(f"Convergencia en {k+1} iteraciones")
                print(f"Raíz: {x_new:.10f}")
                print(f"f(raíz) = {f(x_new):.2e}")
            return x_new, k+1, history
        
        x = x_new
    
    print(f"Advertencia: No convergió en {max_iter} iteraciones")
    return x, max_iter, history


def exercise_2_60():
    """
    Ejercicio 2.60: Implementación y análisis del método de Taylor orden 2.
    """
    print("\n" + "="*80)
    print("EJERCICIO 2.60: MÉTODO DE TAYLOR DE ORDEN 2")
    print("="*80)
    
    # Parte (a): Derivación
    print("\nParte (a): Derivación de la fórmula")
    print("-" * 60)
    print("Ecuación a resolver: 0 = f(x₀) + f'(x₀)(x-x₀) + (f''(x₀)/2)(x-x₀)²")
    print("\nSea h = x - x₀, entonces:")
    print("  f''(x₀)h² + 2f'(x₀)h + 2f(x₀) = 0")
    print("\nUsando la fórmula cuadrática:")
    print("  h = [-2f'(x₀) ± √(4[f'(x₀)]² - 8f''(x₀)f(x₀))] / [2f''(x₀)]")
    print("    = [-f'(x₀) ± √([f'(x₀)]² - 2f''(x₀)f(x₀))] / f''(x₀)")
    print("\nPor lo tanto:")
    print("  x = x₀ + [-f'(x₀) ± √([f'(x₀)]² - 2f''(x₀)f(x₀))] / f''(x₀)")
    
    # Parte (b): Demostración
    print("\n" + "="*80)
    print("Parte (b): Demostración del código")
    print("="*80)
    
    test_problems = [
        {
            'name': 'x² - 4 = 0',
            'f': lambda x: x**2 - 4,
            'fp': lambda x: 2*x,
            'fpp': lambda x: 2,
            'x0': 1.0,
            'exact': 2.0
        },
        {
            'name': 'x³ - 3 = 0',
            'f': lambda x: x**3 - 3,
            'fp': lambda x: 3*x**2,
            'fpp': lambda x: 6*x,
            'x0': 1.0,
            'exact': 3**(1/3)
        },
        {
            'name': 'eˣ - 2 = 0',
            'f': lambda x: np.exp(x) - 2,
            'fp': lambda x: np.exp(x),
            'fpp': lambda x: np.exp(x),
            'x0': 0.5,
            'exact': np.log(2)
        },
        {
            'name': 'cos(x) - x = 0',
            'f': lambda x: np.cos(x) - x,
            'fp': lambda x: -np.sin(x) - 1,
            'fpp': lambda x: -np.cos(x),
            'x0': 0.5,
            'exact': 0.7390851332  # Valor numérico conocido
        }
    ]
    
    results_table = []
    
    for problem in test_problems:
        print(f"\n{'-'*60}")
        print(f"Problema: {problem['name']}")
        print(f"Raíz exacta: {problem['exact']:.10f}")
        print(f"{'-'*60}")
        
        root, iters, history = taylor_order2(
            problem['f'], problem['fp'], problem['fpp'],
            problem['x0'], tol=1e-12, verbose=True
        )
        
        error = abs(root - problem['exact'])
        results_table.append([problem['name'], root, iters, error])
    
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    df = pd.DataFrame(results_table, 
                     columns=['Problema', 'Raíz encontrada', 'Iteraciones', 'Error'])
    print(df.to_string(index=False))
    
    # Parte (c): Gráficas de orden
    print("\n" + "="*80)
    print("Parte (c): Análisis de orden de convergencia")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Análisis de Orden: Método de Taylor de Orden 2', fontsize=14, fontweight='bold')
    
    for idx, problem in enumerate(test_problems):
        ax = axes[idx // 2, idx % 2]
        
        # Ejecutar método guardando errores
        x = problem['x0']
        errors = []
        
        for _ in range(15):
            error = abs(x - problem['exact'])
            if error < 1e-15:
                break
            errors.append(error)
            
            fx = problem['f'](x)
            fpx = problem['fp'](x)
            fppx = problem['fpp'](x)
            
            disc = fpx**2 - 2*fppx*fx
            if disc < 0 or abs(fppx) < 1e-15:
                break
            
            sqrt_disc = np.sqrt(disc)
            if fpx >= 0:
                h = (-fpx - sqrt_disc) / fppx
            else:
                h = (-fpx + sqrt_disc) / fppx
            
            x = x + h
        
        if len(errors) > 2:
            log_old = np.log10(errors[:-1])
            log_new = np.log10(errors[1:])
            
            ax.plot(log_old, log_new, 'bo', markersize=8)
            
            # Ajuste lineal
            coeffs = np.polyfit(log_old, log_new, 1)
            slope = coeffs[0]
            
            ax.plot(log_old, np.polyval(coeffs, log_old), 'r-', linewidth=2,
                   label=f'Pendiente = {slope:.3f}')
            
            ax.set_xlabel('log₁₀(error anterior)', fontsize=10)
            ax.set_ylabel('log₁₀(error nuevo)', fontsize=10)
            ax.set_title(f'{problem["name"]}\nOrden ≈ {slope:.3f}', fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            print(f"{problem['name']:<20} Orden observado: {slope:.3f}")
    
    plt.tight_layout()
    plt.savefig('/home/claude/exercise_2_60_part_c.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Parte (d): Pros y contras
    print("\n" + "="*80)
    print("Parte (d): Ventajas y Desventajas")
    print("="*80)
    
    print("\nVENTAJAS:")
    print("  ✓ Convergencia cúbica (orden ≈ 3) cuando funciona correctamente")
    print("  ✓ Menos iteraciones que Newton para alcanzar la misma precisión")
    print("  ✓ Utiliza información de la curvatura (segunda derivada)")
    print("  ✓ Generalización natural a órdenes superiores")
    
    print("\nDESVENTAJAS:")
    print("  ✗ Requiere calcular f''(x), que puede ser:")
    print("     • Costoso computacionalmente")
    print("     • Difícil de obtener analíticamente")
    print("     • Propenso a errores si se aproxima numéricamente")
    print("  ✗ Cada iteración requiere:")
    print("     • 3 evaluaciones (f, f', f'')")
    print("     • Resolver ecuación cuadrática (raíz cuadrada)")
    print("     • Más operaciones aritméticas")
    print("  ✗ Problemas potenciales:")
    print("     • Discriminante negativo → raíces complejas")
    print("     • f''(x) ≈ 0 → división por cero")
    print("     • Propagación de errores de redondeo")
    print("  ✗ No garantiza convergencia global")
    print("  ✗ Implementación más compleja que Newton")
    
    print("\nEFICIENCIA COMPARATIVA:")
    print("  Aunque tiene orden 3, el costo adicional puede hacer que Newton")
    print("  (orden 2) sea más eficiente en tiempo total de cómputo.")
    
    print(f"\nGráficas guardadas como 'exercise_2_60_part_c.png'")


# ============================================================================
# EJERCICIO 2.61: CAÍDA CON RESISTENCIA DEL AIRE
# ============================================================================

def exercise_2_61():
    """
    Ejercicio 2.61: Modelado de caída de objeto con resistencia del aire.
    """
    print("\n" + "="*80)
    print("EJERCICIO 2.61: CAÍDA CON RESISTENCIA DEL AIRE")
    print("="*80)
    
    # Función de posición
    def s(t, m, g, k, s0):
        """Posición del objeto en función del tiempo"""
        term1 = s0
        term2 = -(m*g/k) * t
        term3 = (m**2 * g / k**2) * (1 - np.exp(-k*t/m))
        return term1 + term2 + term3
    
    # Parte (a): Unidades de k
    print("\nParte (a): Unidades del parámetro k")
    print("-" * 60)
    print("Análisis dimensional del término exponencial:")
    print("  e^(-kt/m) debe ser adimensional")
    print("  Por lo tanto: [k]·[t] = [m]")
    print("  Donde [t] = segundos, [m] = kilogramos")
    print("\nRespuesta: [k] = kg/s (kilogramos por segundo)")
    print("\nVerificación con mg/k:")
    print("  [mg/k] = [kg]·[m/s²] / [kg/s] = m/s ✓ (velocidad)")
    print("\nVerificación con m²g/k²:")
    print("  [m²g/k²] = [kg²]·[m/s²] / [kg²/s²] = m ✓ (longitud)")
    
    # Parte (b): Tiempo de caída
    print("\n" + "="*80)
    print("Parte (b): Tiempo para tocar el suelo")
    print("="*80)
    
    # Parámetros
    m = 1.0      # kg
    g = 9.8      # m/s²
    k = 0.1      # kg/s
    s0 = 100.0   # m
    
    print(f"\nParámetros:")
    print(f"  Masa (m) = {m} kg")
    print(f"  Gravedad (g) = {g} m/s²")
    print(f"  Coeficiente de arrastre (k) = {k} kg/s")
    print(f"  Posición inicial (s₀) = {s0} m")
    
    # Definir función para búsqueda de raíces
    f = lambda t: s(t, m, g, k, s0)
    fprime = lambda t: -(m*g/k) + (m*g/k)*np.exp(-k*t/m)
    
    print("\nEcuación a resolver: s(t) = 0")
    print(f"  s(t) = {s0} - ({m*g/k:.1f})t + ({m**2*g/k**2:.1f})(1 - e^(-{k/m}t))")
    
    # Análisis preliminar
    print("\nAnálisis preliminar:")
    print(f"  s(0) = {f(0):.2f} m > 0")
    print(f"  s(20) = {f(20):.2f} m")
    
    # Resolver usando Newton
    print("\nResolviendo con el método de Newton...")
    t_impact, iters, _ = newton(f, fprime, x0=10.0, tol=0.005, verbose=False)
    
    print(f"\nResultado:")
    print(f"  Tiempo de caída: {t_impact:.2f} segundos")
    print(f"  Iteraciones necesarias: {iters}")
    print(f"  Verificación: s({t_impact:.2f}) = {f(t_impact):.6f} m")
    
    # Comparación con caída libre
    t_free_fall = np.sqrt(2 * s0 / g)
    print(f"\nComparación con caída libre (sin resistencia del aire):")
    print(f"  Tiempo en caída libre: {t_free_fall:.2f} segundos")
    print(f"  Diferencia: {t_impact - t_free_fall:.2f} segundos ({(t_impact/t_free_fall - 1)*100:.1f}% más lento)")
    
    # Parte (c): Análisis de sensibilidad
    print("\n" + "="*80)
    print("Parte (c): Análisis de Sensibilidad")
    print("="*80)
    print("\nSi k se conoce con precisión del 10%, entonces k ∈ [0.09, 0.11] kg/s")
    
    k_values = [0.09, 0.10, 0.11]
    k_labels = ['k mínimo (0.09)', 'k nominal (0.10)', 'k máximo (0.11)']
    times = []
    
    print(f"\n{'Valor de k (kg/s)':<25} {'Tiempo (s)':<15} {'Variación (s)':<15} {'Porcentaje':<15}")
    print("-" * 80)
    
    # Primero calcular todos los tiempos
    for k_val in k_values:
        f_k = lambda t, kv=k_val: s(t, m, g, kv, s0)
        fprime_k = lambda t, kv=k_val: -(m*g/kv) + (m*g/kv)*np.exp(-kv*t/m)
        
        t_k, _, _ = newton(f_k, fprime_k, x0=10.0, tol=0.005, verbose=False)
        times.append(t_k)
    
    # Encontrar el tiempo nominal (k=0.10)
    t_nominal = times[1]  # k=0.10 es el segundo valor
    
    # Ahora imprimir con las variaciones
    for idx, (k_val, t_k) in enumerate(zip(k_values, times)):
        if k_val == 0.10:
            variation = 0
            percentage = 0
        else:
            variation = t_k - t_nominal
            percentage = (variation / t_nominal) * 100
        
        print(f"{k_val:<25} {t_k:<15.2f} {variation:<15.2f} {percentage:<15.1f}%")
    
    print("-" * 80)
    print(f"\nRango de tiempos: [{min(times):.2f}, {max(times):.2f}] segundos")
    print(f"Incertidumbre: ± {(max(times) - min(times))/2:.2f} segundos")
    
    uncertainty = (max(times) - min(times))/2
    print(f"\nRespuesta final: t = {t_nominal:.2f} ± {uncertainty:.2f} segundos")
    
    # Crear gráficas
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Ejercicio 2.61: Caída con Resistencia del Aire', fontsize=14, fontweight='bold')
    
    # Subplot 1: Posición vs tiempo
    ax1 = axes[0, 0]
    t_range = np.linspace(0, 12, 1000)
    s_range = s(t_range, m, g, k, s0)
    ax1.plot(t_range, s_range, 'b-', linewidth=2, label='s(t) con resistencia')
    
    # Caída libre
    s_free_fall = s0 - 0.5*g*t_range**2
    ax1.plot(t_range, s_free_fall, 'r--', linewidth=2, label='s(t) caída libre', alpha=0.7)
    
    ax1.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax1.axvline(x=t_impact, color='b', linestyle='--', alpha=0.5, label=f't = {t_impact:.2f}s')
    ax1.axvline(x=t_free_fall, color='r', linestyle='--', alpha=0.5, label=f't = {t_free_fall:.2f}s')
    ax1.plot(t_impact, 0, 'bo', markersize=10)
    ax1.plot(t_free_fall, 0, 'ro', markersize=10)
    
    ax1.set_xlabel('Tiempo (s)', fontsize=11)
    ax1.set_ylabel('Posición (m)', fontsize=11)
    ax1.set_title('Posición vs Tiempo', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Velocidad vs tiempo
    ax2 = axes[0, 1]
    v = lambda t, k_val: -(m*g/k_val)*(1 - np.exp(-k_val*t/m))
    v_range = v(t_range, k)
    v_free_fall = -g * t_range
    
    ax2.plot(t_range, v_range, 'b-', linewidth=2, label='v(t) con resistencia')
    ax2.plot(t_range, v_free_fall, 'r--', linewidth=2, label='v(t) caída libre', alpha=0.7)
    
    # Velocidad terminal
    v_terminal = -m*g/k
    ax2.axhline(y=v_terminal, color='g', linestyle='-.', linewidth=2, 
                label=f'v_terminal = {v_terminal:.1f} m/s')
    
    ax2.set_xlabel('Tiempo (s)', fontsize=11)
    ax2.set_ylabel('Velocidad (m/s)', fontsize=11)
    ax2.set_title('Velocidad vs Tiempo', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Análisis de sensibilidad
    ax3 = axes[1, 0]
    colors_sens = ['blue', 'green', 'red']
    
    for k_val, color, label in zip(k_values, colors_sens, k_labels):
        t_plot = np.linspace(0, 12, 1000)
        s_plot = s(t_plot, m, g, k_val, s0)
        ax3.plot(t_plot, s_plot, color=color, linewidth=2, label=label)
    
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Tiempo (s)', fontsize=11)
    ax3.set_ylabel('Posición (m)', fontsize=11)
    ax3.set_title('Análisis de Sensibilidad (variación de k)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Comparación de tiempos de impacto
    ax4 = axes[1, 1]
    methods = ['k = 0.09', 'k = 0.10\n(nominal)', 'k = 0.11']
    colors_bar = ['blue', 'green', 'red']
    
    bars = ax4.bar(methods, times, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4.axhline(y=t_nominal, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax4.set_ylabel('Tiempo de caída (s)', fontsize=11)
    ax4.set_title('Tiempo de Impacto vs Valor de k', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/claude/exercise_2_61.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráficas guardadas como 'exercise_2_61.png'")


# ============================================================================
# FUNCIÓN PRINCIPAL PARA EJECUTAR TODOS LOS EJERCICIOS
# ============================================================================

def run_all_exercises():
    """
    Ejecuta todos los ejercicios en secuencia.
    """
    print("\n" + "="*80)
    print(" "*20 + "MÉTODOS NUMÉRICOS - EJERCICIOS 2.51 A 2.61")
    print("="*80)
    
    # Ejercicio 2.55
    exercise_2_55()
    
    # Ejercicio 2.56
    exercise_2_56()
    
    # Ejercicio 2.57
    exercise_2_57()
    
    # Ejercicio 2.58
    exercise_2_58()
    
    # Ejercicio 2.59
    exercise_2_59()
    
    # Ejercicio 2.60
    exercise_2_60()
    
    # Ejercicio 2.61
    exercise_2_61()
    
    print("\n" + "="*80)
    print("TODOS LOS EJERCICIOS COMPLETADOS")
    print("="*80)
    print("\nArchivos generados:")
    print("  • exercise_2_55.png - Iteraciones vs Tolerancia (Bisección)")
    print("  • exercise_2_56.png - Análisis completo de x³ - 3 = 0")
    print("  • exercise_2_57.png - Ecuación 3sin(x) + 9 = x² - cos(x)")
    print("  • exercise_2_58_part_c.png - Análisis de orden de convergencia")
    print("  • exercise_2_59.png - Predicción de errores (Newton)")
    print("  • exercise_2_60_part_c.png - Método de Taylor orden 2")
    print("  • exercise_2_61.png - Caída con resistencia del aire")


# ============================================================================
# EJEMPLOS DE USO INDIVIDUAL
# ============================================================================

def example_usage():
    """
    Ejemplos de cómo usar cada función individualmente.
    """
    print("\n" + "="*80)
    print("EJEMPLOS DE USO")
    print("="*80)
    
    # Ejemplo 1: Método de Bisección
    print("\nEjemplo 1: Bisección para x² - 2 = 0")
    print("-" * 60)
    f = lambda x: x**2 - 2
    root, iters, _ = bisection(f, a=0, b=2, tol=1e-6, verbose=True)
    
    # Ejemplo 2: Método de Newton
    print("\n\nEjemplo 2: Newton para x³ - 3 = 0")
    print("-" * 60)
    f = lambda x: x**3 - 3
    fprime = lambda x: 3*x**2
    root, iters, _ = newton(f, fprime, x0=1.5, tol=1e-10, verbose=True)
    
    # Ejemplo 3: Método de la Secante
    print("\n\nEjemplo 3: Secante para cos(x) - x = 0")
    print("-" * 60)
    f = lambda x: np.cos(x) - x
    root, iters, _ = secant(f, x0=0, x1=1, tol=1e-10, verbose=True)


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Ejecutar todos los ejercicios
    run_all_exercises()
    
    # O ejecutar ejercicios individuales:
    # exercise_2_55()
    # exercise_2_56()
    # exercise_2_57()
    # exercise_2_58()
    # exercise_2_59()
    # exercise_2_60()
    # exercise_2_61()
    
    # O ver ejemplos de uso:
    # example_usage()