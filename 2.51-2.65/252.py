"""
Exercise 2.52: Regula Falsi Method (False Position Method)

PROBLEMA:
Sea f(x) una función continua en el intervalo [a,b] donde f(a)⋅f(b)<0.
Dar claramente todos los detalles matemáticos de cómo el Método de Regula Falsi
aproxima la raíz de la función f(x) en el intervalo [a,b].

DETALLES MATEMÁTICOS DEL MÉTODO DE REGULA FALSI:

1. FUNDAMENTO TEÓRICO:
   
   El método de Regula Falsi es una variación del método de bisección que utiliza
   interpolación lineal en lugar de simplemente dividir el intervalo a la mitad.
   
   Teorema del Valor Intermedio:
   Si f es continua en [a,b] y f(a)⋅f(b) < 0, entonces existe al menos
   un c ∈ (a,b) tal que f(c) = 0.

2. GEOMETRÍA DEL MÉTODO:
   
   En lugar de tomar el punto medio, Regula Falsi encuentra el punto donde
   la línea recta que conecta (a, f(a)) y (b, f(b)) cruza el eje x.
   
   Esta línea recta es: y - f(a) = [(f(b) - f(a))/(b - a)] × (x - a)
   
   Para encontrar donde y = 0:
   0 - f(a) = [(f(b) - f(a))/(b - a)] × (x - a)
   -f(a) × (b - a) = (f(b) - f(a)) × (x - a)
   x - a = -f(a) × (b - a) / (f(b) - f(a))

3. FÓRMULA DE ITERACIÓN:
   
   El punto de intersección c_n se calcula como:
   
   c_n = a_n - f(a_n) × (b_n - a_n) / (f(b_n) - f(a_n))
   
   O equivalentemente:
   
   c_n = [a_n × f(b_n) - b_n × f(a_n)] / [f(b_n) - f(a_n)]
   
   Esta fórmula viene de la interpolación lineal y garantiza que c_n ∈ (a_n, b_n).

4. ALGORITMO:
   
   Paso 0: Verificar que f(a)⋅f(b) < 0
   
   Para cada iteración n = 1, 2, 3, ...:
   
   Paso 1: Calcular el punto de falsa posición usando interpolación lineal:
           c_n = a_n - f(a_n) × (b_n - a_n) / (f(b_n) - f(a_n))
   
   Paso 2: Evaluar f(c_n)
   
   Paso 3: Determinar el nuevo intervalo:
           - Si f(c_n) = 0: ¡Raíz exacta encontrada!
           - Si f(a_n)⋅f(c_n) < 0: La raíz está en [a_n, c_n]
             Entonces: a_{n+1} = a_n, b_{n+1} = c_n
           - Si f(c_n)⋅f(b_n) < 0: La raíz está en [c_n, b_n]
             Entonces: a_{n+1} = c_n, b_{n+1} = b_n
   
   Paso 4: Verificar criterio de parada:
           - |c_n - c_{n-1}| < tolerancia, o
           - |f(c_n)| < tolerancia, o
           - número máximo de iteraciones alcanzado

5. PROPIEDADES DE CONVERGENCIA:
   
   a) Convergencia superlineal (más rápida que bisección, pero típicamente
      más lenta que Newton-Raphson)
   
   b) Orden de convergencia: aproximadamente 1.618 (número áureo φ) en casos
      favorables, pero puede degradarse a convergencia lineal
   
   c) El método SIEMPRE converge (garantizado por el Teorema del Valor Intermedio)
   
   d) Puede sufrir de "convergencia lenta en un extremo": si la función es
      muy curvada, uno de los extremos puede permanecer fijo durante muchas
      iteraciones

6. COMPARACIÓN CON BISECCIÓN:
   
   BISECCIÓN:
   - Siempre divide el intervalo a la mitad
   - Convergencia lineal con tasa 1/2
   - No considera la forma de la función
   
   REGULA FALSI:
   - Usa la pendiente de la función
   - Convergencia típicamente más rápida
   - Aprovecha información geométrica
   - Puede ser más eficiente en funciones "bien comportadas"

7. VENTAJAS:
   - Converge más rápido que bisección en la mayoría de los casos
   - Robusto (siempre converge)
   - No requiere derivadas
   - Utiliza información geométrica de la función

8. DESVENTAJAS:
   - Puede converger lentamente si la función es muy curvada
   - Un extremo puede quedarse "atascado"
   - Más complejo que bisección
   - Puede requerir más evaluaciones de función que Newton-Raphson

9. MODIFICACIÓN: Illinois Method
   
   Para evitar el problema de "extremo atascado", se puede usar la modificación
   de Illinois que reduce el valor de f en el extremo estacionario:
   
   Si el mismo extremo persiste por 2 iteraciones consecutivas:
   f(extremo_fijo) = f(extremo_fijo) / 2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Polygon

def regula_falsi(f, a, b, tol=1e-6, max_iter=100, verbose=True, method='standard'):
    """
    Implementación del Método de Regula Falsi (Falsa Posición)
    
    Parámetros:
    -----------
    f : función
        La función f(x) para la cual buscamos la raíz
    a : float
        Extremo izquierdo del intervalo
    b : float
        Extremo derecho del intervalo
    tol : float
        Tolerancia para el criterio de parada
    max_iter : int
        Número máximo de iteraciones
    verbose : bool
        Si True, imprime información de cada iteración
    method : str
        'standard' o 'illinois' (modificación para evitar convergencia lenta)
    
    Retorna:
    --------
    root : float
        Aproximación de la raíz
    iterations : int
        Número de iteraciones realizadas
    history : list
        Historia de aproximaciones en cada iteración
    intervals : list
        Historia de intervalos
    """
    
    # Verificar condición inicial
    fa = f(a)
    fb = f(b)
    
    if fa * fb >= 0:
        raise ValueError(f"f(a)⋅f(b) debe ser < 0. Valores: f({a}) = {fa}, f({b}) = {fb}")
    
    # Inicialización
    history = []
    intervals = []
    f_values = []
    c_prev = None
    
    # Para el método de Illinois
    side = 0  # Contador para verificar si un lado se queda fijo
    
    if verbose:
        print("=" * 95)
        print("MÉTODO DE REGULA FALSI (FALSA POSICIÓN)" + 
              (" - MODIFICACIÓN ILLINOIS" if method == 'illinois' else ""))
        print("=" * 95)
        print(f"Intervalo inicial: [{a}, {b}]")
        print(f"f(a) = {fa:.6f}, f(b) = {fb:.6f}")
        print(f"Tolerancia: {tol}")
        print("\n" + "-" * 95)
        print(f"{'Iter':<6} {'a':<12} {'b':<12} {'c':<12} {'f(c)':<12} {'|c-c_prev|':<12} {'|b-a|':<12}")
        print("-" * 95)
    
    # Iteraciones
    for n in range(max_iter):
        # Paso 1: Calcular punto de falsa posición usando interpolación lineal
        # Fórmula: c = a - f(a) * (b - a) / (f(b) - f(a))
        c = a - fa * (b - a) / (fb - fa)
        fc = f(c)
        
        history.append(c)
        intervals.append([a, b])
        f_values.append(fc)
        
        # Calcular cambio desde iteración anterior
        change = abs(c - c_prev) if c_prev is not None else float('inf')
        
        if verbose:
            print(f"{n+1:<6} {a:<12.6f} {b:<12.6f} {c:<12.6f} {fc:<12.6e} {change:<12.6e} {abs(b-a):<12.6e}")
        
        # Paso 2: Verificar criterios de parada
        if abs(fc) < tol or (c_prev is not None and change < tol):
            if verbose:
                print("-" * 95)
                print(f"\n¡Convergencia alcanzada en {n+1} iteraciones!")
                print(f"Raíz aproximada: {c:.10f}")
                print(f"|f(c)| = {abs(fc):.6e}")
                print(f"Cambio en última iteración: {change:.6e}")
            return c, n+1, history, intervals, f_values
        
        # Paso 3: Determinar nuevo intervalo
        c_prev_old = c_prev
        c_prev = c
        
        if fa * fc < 0:
            # La raíz está en [a, c]
            b = c
            fb = fc
            
            # Modificación Illinois
            if method == 'illinois':
                if side == -1:  # El lado izquierdo se quedó fijo
                    fa = fa / 2.0
                side = -1
        else:
            # La raíz está en [c, b]
            a = c
            fa = fc
            
            # Modificación Illinois
            if method == 'illinois':
                if side == 1:  # El lado derecho se quedó fijo
                    fb = fb / 2.0
                side = 1
    
    # Si llegamos aquí, alcanzamos el máximo de iteraciones
    if verbose:
        print("-" * 95)
        print(f"\nAlcanzado el número máximo de iteraciones ({max_iter})")
        print(f"Mejor aproximación: {c:.10f}")
    
    return c, max_iter, history, intervals, f_values


def bisection_for_comparison(f, a, b, tol=1e-6, max_iter=100):
    """
    Implementación simple de bisección para comparación
    """
    history = []
    
    for n in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        history.append(c)
        
        if abs(fc) < tol or abs(b - a) < tol:
            return c, n+1, history
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    return c, max_iter, history


def plot_regula_falsi_detailed(f, a, b, history, intervals, f_values, true_root=None):
    """
    Grafica detallada del proceso del Método de Regula Falsi
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Gráfica 1: Visualización geométrica del método (primeras 4 iteraciones)
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.linspace(a - 0.2, b + 0.2, 1000)
    y = [f(xi) for xi in x]
    
    ax1.plot(x, y, 'b-', linewidth=2, label='f(x)', zorder=1)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3, zorder=1)
    ax1.grid(True, alpha=0.3)
    
    # Mostrar las primeras iteraciones con líneas de interpolación
    colors = ['red', 'green', 'orange', 'purple']
    num_show = min(4, len(history))
    
    for i in range(num_show):
        a_i, b_i = intervals[i]
        c_i = history[i]
        
        # Dibujar línea de interpolación
        x_line = np.array([a_i, b_i])
        y_line = np.array([f(a_i), f(b_i)])
        ax1.plot(x_line, y_line, '--', color=colors[i], linewidth=1.5, 
                alpha=0.7, label=f'Iteración {i+1}: secante', zorder=2)
        
        # Marcar puntos
        ax1.plot(a_i, f(a_i), 'o', color=colors[i], markersize=8, zorder=3)
        ax1.plot(b_i, f(b_i), 's', color=colors[i], markersize=8, zorder=3)
        ax1.plot(c_i, 0, 'X', color=colors[i], markersize=12, 
                label=f'c_{i+1} = {c_i:.4f}', zorder=4)
        
        # Línea vertical desde c hasta f(c)
        ax1.plot([c_i, c_i], [0, f(c_i)], ':', color=colors[i], 
                linewidth=1.5, alpha=0.5, zorder=2)
    
    if true_root is not None:
        ax1.axvline(x=true_root, color='darkred', linestyle='--', linewidth=2, 
                   label=f'Raíz verdadera: {true_root:.6f}', zorder=5)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('f(x)', fontsize=12)
    ax1.set_title('Regula Falsi: Interpolación Lineal (Primeras 4 Iteraciones)', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.set_xlim([a - 0.1, b + 0.1])
    
    # Gráfica 2: Convergencia de aproximaciones
    ax2 = fig.add_subplot(gs[1, 0])
    
    iterations = range(1, len(history) + 1)
    ax2.plot(iterations, history, 'go-', linewidth=2, markersize=6, label='Aproximaciones c_n')
    
    if true_root is not None:
        ax2.axhline(y=true_root, color='r', linestyle='--', linewidth=2, 
                   label=f'Raíz verdadera')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Iteración', fontsize=12)
    ax2.set_ylabel('Valor de c_n', fontsize=12)
    ax2.set_title('Convergencia de las Aproximaciones', fontsize=13, fontweight='bold')
    ax2.legend()
    
    # Gráfica 3: Error absoluto
    ax3 = fig.add_subplot(gs[1, 1])
    
    if true_root is not None:
        errors = [abs(c - true_root) for c in history]
        ax3.semilogy(iterations, errors, 'mo-', linewidth=2, markersize=6)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Iteración', fontsize=12)
        ax3.set_ylabel('Error absoluto (escala log)', fontsize=12)
        ax3.set_title('Error Absoluto: |c_n - raíz|', fontsize=13, fontweight='bold')
    else:
        # Si no conocemos la raíz, graficar |f(c_n)|
        ax3.semilogy(iterations, [abs(fv) for fv in f_values], 'co-', 
                    linewidth=2, markersize=6)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Iteración', fontsize=12)
        ax3.set_ylabel('|f(c_n)| (escala log)', fontsize=12)
        ax3.set_title('Convergencia: |f(c_n)|', fontsize=13, fontweight='bold')
    
    # Gráfica 4: Longitud del intervalo
    ax4 = fig.add_subplot(gs[2, 0])
    
    interval_lengths = [b - a for a, b in intervals]
    ax4.semilogy(iterations, interval_lengths, 'bs-', linewidth=2, markersize=6)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Iteración', fontsize=12)
    ax4.set_ylabel('Longitud del intervalo (escala log)', fontsize=12)
    ax4.set_title('Reducción del Intervalo: |b_n - a_n|', fontsize=13, fontweight='bold')
    
    # Gráfica 5: Cambio entre iteraciones
    ax5 = fig.add_subplot(gs[2, 1])
    
    changes = [abs(history[i] - history[i-1]) for i in range(1, len(history))]
    ax5.semilogy(range(2, len(history) + 1), changes, 'rs-', linewidth=2, markersize=6)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlabel('Iteración', fontsize=12)
    ax5.set_ylabel('|c_n - c_{n-1}| (escala log)', fontsize=12)
    ax5.set_title('Cambio entre Iteraciones Consecutivas', fontsize=13, fontweight='bold')
    
    plt.suptitle('Análisis Completo del Método de Regula Falsi', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.show()


def compare_methods(f, a, b, true_root, tol=1e-8):
    """
    Compara Bisección, Regula Falsi estándar y Regula Falsi Illinois
    """
    print("\n" + "="*80)
    print("COMPARACIÓN DE MÉTODOS")
    print("="*80)
    
    # Bisección
    print("\n--- MÉTODO DE BISECCIÓN ---")
    root_bis, iters_bis, history_bis = bisection_for_comparison(f, a, b, tol, verbose=False)
    error_bis = abs(root_bis - true_root)
    print(f"Iteraciones: {iters_bis}")
    print(f"Raíz aproximada: {root_bis:.10f}")
    print(f"Error: {error_bis:.6e}")
    
    # Regula Falsi estándar
    print("\n--- REGULA FALSI ESTÁNDAR ---")
    root_rf, iters_rf, history_rf, intervals_rf, f_rf = regula_falsi(
        f, a, b, tol, max_iter=200, verbose=False, method='standard')
    error_rf = abs(root_rf - true_root)
    print(f"Iteraciones: {iters_rf}")
    print(f"Raíz aproximada: {root_rf:.10f}")
    print(f"Error: {error_rf:.6e}")
    
    # Regula Falsi Illinois
    print("\n--- REGULA FALSI ILLINOIS ---")
    root_ill, iters_ill, history_ill, intervals_ill, f_ill = regula_falsi(
        f, a, b, tol, max_iter=200, verbose=False, method='illinois')
    error_ill = abs(root_ill - true_root)
    print(f"Iteraciones: {iters_ill}")
    print(f"Raíz aproximada: {root_ill:.10f}")
    print(f"Error: {error_ill:.6e}")
    
    # Gráfica comparativa
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Número de iteraciones
    methods = ['Bisección', 'Regula Falsi\nEstándar', 'Regula Falsi\nIllinois']
    iterations = [iters_bis, iters_rf, iters_ill]
    colors_bars = ['blue', 'green', 'orange']
    
    bars = ax1.bar(methods, iterations, color=colors_bars, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Número de Iteraciones', fontsize=12)
    ax1.set_title('Comparación de Eficiencia', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores encima de las barras
    for bar, val in zip(bars, iterations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Convergencia comparativa
    max_len = max(len(history_bis), len(history_rf), len(history_ill))
    
    errors_bis = [abs(c - true_root) for c in history_bis]
    errors_rf = [abs(c - true_root) for c in history_rf]
    errors_ill = [abs(c - true_root) for c in history_ill]
    
    ax2.semilogy(range(1, len(errors_bis) + 1), errors_bis, 'b-o', 
                linewidth=2, markersize=5, label='Bisección')
    ax2.semilogy(range(1, len(errors_rf) + 1), errors_rf, 'g-s', 
                linewidth=2, markersize=5, label='Regula Falsi Estándar')
    ax2.semilogy(range(1, len(errors_ill) + 1), errors_ill, 'r-^', 
                linewidth=2, markersize=5, label='Regula Falsi Illinois')
    
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Iteración', fontsize=12)
    ax2.set_ylabel('Error Absoluto (escala log)', fontsize=12)
    ax2.set_title('Comparación de Convergencia', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# EJEMPLOS DE APLICACIÓN
# ============================================================================

print("\n" + "="*95)
print("EJEMPLO 1: f(x) = x³ - x - 2")
print("="*95)

def f1(x):
    return x**3 - x - 2

a1, b1 = 1, 2
true_root1 = 1.5213797068045676

root1, iters1, history1, intervals1, f_vals1 = regula_falsi(
    f1, a1, b1, tol=1e-8, method='standard')

plot_regula_falsi_detailed(f1, a1, b1, history1, intervals1, f_vals1, true_root1)

# Comparación de métodos
compare_methods(f1, a1, b1, true_root1)


print("\n" + "="*95)
print("EJEMPLO 2: f(x) = e^x - 3x² (función con curvatura pronunciada)")
print("="*95)

def f2(x):
    return np.exp(x) - 3*x**2

a2, b2 = 0.5, 1.5

# Encontrar raíz verdadera con alta precisión
from scipy.optimize import fsolve
true_root2 = fsolve(f2, 1.0)[0]

root2, iters2, history2, intervals2, f_vals2 = regula_falsi(
    f2, a2, b2, tol=1e-8, method='standard')

plot_regula_falsi_detailed(f2, a2, b2, history2, intervals2, f_vals2, true_root2)

# Comparación de métodos
compare_methods(f2, a2, b2, true_root2)


print("\n" + "="*95)
print("EJEMPLO 3: f(x) = x³ - 2x - 5 (caso donde Illinois mejora mucho)")
print("="*95)

def f3(x):
    return x**3 - 2*x - 5

a3, b3 = 2, 3
true_root3 = 2.0945514815423265

print("\n--- Regula Falsi ESTÁNDAR ---")
root3_std, iters3_std, history3_std, intervals3_std, f_vals3_std = regula_falsi(
    f3, a3, b3, tol=1e-8, method='standard', max_iter=50)

print("\n--- Regula Falsi ILLINOIS ---")
root3_ill, iters3_ill, history3_ill, intervals3_ill, f_vals3_ill = regula_falsi(
    f3, a3, b3, tol=1e-8, method='illinois', max_iter=50)

# Graficar ambos
plot_regula_falsi_detailed(f3, a3, b3, history3_std, intervals3_std, f_vals3_std, true_root3)
plot_regula_falsi_detailed(f3, a3, b3, history3_ill, intervals3_ill, f_vals3_ill, true_root3)

# Comparación
compare_methods(f3, a3, b3, true_root3)


# ============================================================================
# ANÁLISIS MATEMÁTICO ADICIONAL
# ============================================================================

print("\n" + "="*95)
print("ANÁLISIS MATEMÁTICO DE LA CONVERGENCIA")
print("="*95)

def analyze_convergence_rate(history, true_root):
    """
    Analiza la tasa de convergencia empírica
    """
    errors = [abs(c - true_root) for c in history]
    
    # Calcular orden de convergencia aproximado
    if len(errors) > 3:
        # p ≈ ln(e_{n+1}/e_n) / ln(e_n/e_{n-1})
        ratios = []
        for i in range(2, len(errors) - 1):
            if errors[i-1] > 0 and errors[i] > 0 and errors[i-2] > 0:
                ratio = np.log(errors[i+1] / errors[i]) / np.log(errors[i] / errors[i-1])
                if not np.isnan(ratio) and not np.isinf(ratio):
                    ratios.append(ratio)
        
        if ratios:
            avg_order = np.mean(ratios)
            print(f"\nOrden de convergencia empírico: {avg_order:.4f}")
            print("(Valor teórico para Regula Falsi: ~1.618 en casos favorables)")
    
    # Mostrar últimas 5 iteraciones
    print("\nÚltimas 5 iteraciones:")
    print(f"{'Iter':<8} {'Aproximación':<15} {'Error':<15} {'Reducción':<15}")
    print("-" * 60)
    start = max(0, len(history) - 5)
    for i in range(start, len(history)):
        reduction = errors[i-1] / errors[i] if i > 0 and errors[i] > 0 else 0
        print(f"{i+1:<8} {history[i]:<15.10f} {errors[i]:<15.6e} {reduction:<15.4f}")

analyze_convergence_rate(history1, true_root1)

print("\n" + "="*95)
print("CONCLUSIÓN")
print("="*95)
print("""
El Método de Regula Falsi mejora sobre el método de bisección al utilizar
la información geométrica de la función. En lugar de simplemente dividir
el intervalo a la mitad, utiliza interpolación lineal para obtener una
mejor aproximación de dónde está la raíz.

La modificación de Illinois resuelve el problema de convergencia lenta
cuando un extremo se queda "atascado" al reducir el peso del valor de la
función en ese extremo.

En general, Regula Falsi converge más rápido que bisección pero puede
ser más lento que métodos como Newton-Raphson que utilizan derivadas.
""")