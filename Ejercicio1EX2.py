"""
Ejercicio 1 - Examen 2
Spline Cúbico Natural para respuesta de circuito RLC subamortiguado

Función teórica: y_real(t) = 10[1 - e^(-2.25t)(cos(2.222t) + 1.0126·sin(2.222t))]

Datos experimentales obtenidos de mediciones del circuito.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Datos experimentales (mediciones del circuito RLC)
# ============================================================
t_datos = np.array([0, 0.1, 0.25, 0.5, 0.75, 1, 1.41, 2.12, 2.83, 4, 5])
y_datos = np.array([0, 0.4292, 2.1162, 5.6133, 8.3122, 9.79, 10.4153, 10.086, 9.9827, 10.0004, 10.0001])

n = len(t_datos)  # Número de puntos


# ============================================================
# 2. Función teórica del circuito RLC subamortiguado
# ============================================================
def y_real(t):
    """Respuesta teórica del circuito RLC subamortiguado."""
    return 10 * (1 - np.exp(-2.25 * t) * (np.cos(2.222 * t) + 1.0126 * np.sin(2.222 * t)))


# ============================================================
# 3. Implementación del Spline Cúbico Natural
# ============================================================
def spline_cubico_natural(x, y):
    """
    Calcula los coeficientes del spline cúbico natural.

    Para cada intervalo [x_i, x_{i+1}], el spline es:
        S_i(t) = a_i + b_i*(t - x_i) + c_i*(t - x_i)^2 + d_i*(t - x_i)^3

    Condiciones de frontera naturales: S''(x_0) = 0 y S''(x_n) = 0

    Parámetros:
        x : array de nodos (puntos de interpolación en t)
        y : array de valores en los nodos

    Retorna:
        a, b, c, d : arrays de coeficientes para cada tramo del spline
    """
    n = len(x)
    m = n - 1  # Número de intervalos (tramos)

    # --- Paso 1: Calcular h_i (anchos de los intervalos) ---
    h = np.zeros(m)
    for i in range(m):
        h[i] = x[i + 1] - x[i]

    # --- Paso 2: Coeficientes a_i = y_i ---
    a = y.copy()

    # --- Paso 3: Construir el sistema tridiagonal para c_i ---
    # El sistema Ac = alpha resuelve los coeficientes c_i
    # Condiciones naturales: c_0 = 0, c_{n-1} = 0

    # Vector del lado derecho (alpha)
    alpha = np.zeros(n)
    for i in range(1, m):
        alpha[i] = (3.0 / h[i]) * (a[i + 1] - a[i]) - (3.0 / h[i - 1]) * (a[i] - a[i - 1])

    # --- Paso 4: Resolver el sistema tridiagonal con el algoritmo de Thomas ---
    # Matrices del sistema tridiagonal
    l = np.zeros(n)
    mu = np.zeros(n)
    z = np.zeros(n)

    # Condición de frontera natural izquierda: c_0 = 0
    l[0] = 1.0
    mu[0] = 0.0
    z[0] = 0.0

    # Eliminación hacia adelante
    for i in range(1, m):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    # Condición de frontera natural derecha: c_{n-1} = 0
    l[n - 1] = 1.0
    z[n - 1] = 0.0

    # Sustitución hacia atrás para obtener c_i
    c = np.zeros(n)
    c[n - 1] = 0.0

    for j in range(m - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]

    # --- Paso 5: Calcular b_i y d_i ---
    b = np.zeros(m)
    d = np.zeros(m)

    for i in range(m):
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2.0 * c[i]) / 3.0
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    return a[:m], b, c[:m], d


def evaluar_spline(x_nodos, a, b, c, d, t_eval):
    """
    Evalúa el spline cúbico en los puntos t_eval.

    Para cada punto t en t_eval, encuentra el intervalo correspondiente
    y evalúa: S_i(t) = a_i + b_i*(t-x_i) + c_i*(t-x_i)^2 + d_i*(t-x_i)^3

    Parámetros:
        x_nodos : array de nodos originales
        a, b, c, d : coeficientes del spline
        t_eval : array de puntos donde evaluar el spline

    Retorna:
        y_eval : array de valores interpolados
    """
    m = len(a)
    y_eval = np.zeros(len(t_eval))

    for k, t in enumerate(t_eval):
        # Encontrar el intervalo correcto [x_i, x_{i+1}]
        # Si t está fuera del rango, usar el tramo más cercano
        if t <= x_nodos[0]:
            i = 0
        elif t >= x_nodos[-1]:
            i = m - 1
        else:
            # Búsqueda del intervalo
            i = 0
            for j in range(m):
                if x_nodos[j] <= t <= x_nodos[j + 1]:
                    i = j
                    break

        # Evaluar el polinomio cúbico en el tramo i
        dt = t - x_nodos[i]
        y_eval[k] = a[i] + b[i] * dt + c[i] * dt**2 + d[i] * dt**3

    return y_eval


# ============================================================
# 4. Calcular el Spline Cúbico Natural
# ============================================================
a, b, c, d = spline_cubico_natural(t_datos, y_datos)

# Mostrar los coeficientes del spline
print("=" * 70)
print("  SPLINE CÚBICO NATURAL - Circuito RLC Subamortiguado")
print("=" * 70)
print(f"\n  Número de puntos (nodos): {n}")
print(f"  Número de tramos:         {n - 1}")
print(f"  Rango de interpolación:   [{t_datos[0]}, {t_datos[-1]}] s\n")

print("-" * 70)
print(f"  {'Tramo':<8} {'a_i':<12} {'b_i':<12} {'c_i':<12} {'d_i':<12}")
print("-" * 70)
for i in range(len(a)):
    print(f"  S_{i:<4}   {a[i]:<12.6f} {b[i]:<12.6f} {c[i]:<12.6f} {d[i]:<12.6f}")
print("-" * 70)

# Mostrar los polinomios de cada tramo
print("\n  Polinomios por tramo:")
print("-" * 70)
for i in range(len(a)):
    print(f"  S_{i}(t) = {a[i]:.4f} + {b[i]:.4f}·(t-{t_datos[i]:.2f})"
          f" + {c[i]:.4f}·(t-{t_datos[i]:.2f})² + {d[i]:.4f}·(t-{t_datos[i]:.2f})³")
    print(f"         para t en [{t_datos[i]:.2f}, {t_datos[i + 1]:.2f}]")
print("-" * 70)


# ============================================================
# 5. Evaluar el spline y la función real en puntos densos
# ============================================================
t_fino = np.linspace(t_datos[0], t_datos[-1], 500)
y_spline = evaluar_spline(t_datos, a, b, c, d, t_fino)
y_teorica = y_real(t_fino)

# Calcular el error
error = np.abs(y_spline - y_teorica)
error_max = np.max(error)
error_promedio = np.mean(error)

print(f"\n  Error máximo absoluto:    {error_max:.6f}")
print(f"  Error promedio absoluto:  {error_promedio:.6f}")
print("=" * 70)

# Comparar en los nodos
print("\n  Comparación en los nodos de medición:")
print("-" * 70)
print(f"  {'t (s)':<10} {'y_medido':<14} {'y_real(t)':<14} {'|Error|':<12}")
print("-" * 70)
for i in range(n):
    yr = y_real(t_datos[i])
    err = abs(y_datos[i] - yr)
    print(f"  {t_datos[i]:<10.2f} {y_datos[i]:<14.4f} {yr:<14.4f} {err:<12.6f}")
print("-" * 70)


# ============================================================
# 6. Gráficas de comparación
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('Spline Cúbico Natural vs Respuesta Real\nCircuito RLC Subamortiguado',
             fontsize=14, fontweight='bold')

# --- Gráfica principal: Spline vs Función Real ---
ax1 = axes[0]
ax1.plot(t_fino, y_teorica, 'b-', linewidth=2, label=r'$y_{real}(t) = 10[1 - e^{-2.25t}(\cos(2.222t) + 1.0126\sin(2.222t))]$')
ax1.plot(t_fino, y_spline, 'r--', linewidth=2, label='Spline Cúbico Natural')
ax1.plot(t_datos, y_datos, 'ko', markersize=8, markerfacecolor='gold',
         markeredgewidth=1.5, label='Datos experimentales', zorder=5)

# Línea del valor de estado estable
ax1.axhline(y=10, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Estado estable (y = 10)')

ax1.set_xlabel('Tiempo t (s)', fontsize=12)
ax1.set_ylabel('y(t)', fontsize=12)
ax1.set_title('Comparación de la interpolación', fontsize=12)
ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_xlim([t_datos[0], t_datos[-1]])

# --- Gráfica del error absoluto ---
ax2 = axes[1]
ax2.plot(t_fino, error, 'm-', linewidth=1.5, label='|Error| = |Spline - y_real|')
ax2.fill_between(t_fino, 0, error, color='magenta', alpha=0.2)
ax2.axhline(y=error_promedio, color='green', linestyle='--', linewidth=1,
            label=f'Error promedio = {error_promedio:.4f}')

ax2.set_xlabel('Tiempo t (s)', fontsize=12)
ax2.set_ylabel('|Error|', fontsize=12)
ax2.set_title(f'Error absoluto (máximo = {error_max:.4f})', fontsize=12)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_xlim([t_datos[0], t_datos[-1]])

plt.tight_layout()
plt.savefig(r'c:\Users\soysa\OneDrive\Documentos\python codes\spline_cubico_RLC.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n  Grafica guardada como 'spline_cubico_RLC.png'")


# ============================================================
# 7. Datos faltantes de la tabla
# ============================================================
print("\n" + "=" * 70)
print("  DATOS FALTANTES DE LA TABLA")
print("=" * 70)

# --- 7a. Interpolar y(1.3) y y(1.2) ---
t_interp = [1.2, 1.3]
print("\n  a) Interpolacion directa (encontrar y dado t):")
print("-" * 70)
print(f"  {'t (s)':<10} {'y_spline':<14} {'y_real':<14} {'Error Abs':<14} {'Error Rel':<14}")
print("-" * 70)

for t_val in t_interp:
    y_spl = evaluar_spline(t_datos, a, b, c, d, np.array([t_val]))[0]
    y_r = y_real(t_val)
    e_abs = abs(y_r - y_spl)
    e_rel = e_abs / abs(y_r)
    print(f"  {t_val:<10.2f} {y_spl:<14.6f} {y_r:<14.6f} {e_abs:<14.6f} {e_rel:<14.6f}")

print("-" * 70)

# --- 7b. Interpolacion inversa: encontrar t dado y = 5 y y = 8.4671 ---
y_buscar = [5.0, 8.4671]
print("\n  b) Interpolacion inversa (encontrar t dado y):")
print("-" * 70)

for y_obj in y_buscar:
    # Buscar en que tramo el spline cruza y_obj
    # Evaluar el spline en muchos puntos y encontrar el cruce
    t_busqueda = np.linspace(t_datos[0], t_datos[-1], 10000)
    y_busqueda = evaluar_spline(t_datos, a, b, c, d, t_busqueda)

    # Encontrar el primer cruce (donde y pasa por y_obj)
    t_encontrado = None
    for i in range(len(t_busqueda) - 1):
        if (y_busqueda[i] - y_obj) * (y_busqueda[i + 1] - y_obj) <= 0:
            # Interpolacion lineal para refinar
            frac = (y_obj - y_busqueda[i]) / (y_busqueda[i + 1] - y_busqueda[i])
            t_encontrado = t_busqueda[i] + frac * (t_busqueda[i + 1] - t_busqueda[i])
            break

    if t_encontrado is not None:
        y_spl = evaluar_spline(t_datos, a, b, c, d, np.array([t_encontrado]))[0]
        y_r = y_real(t_encontrado)
        e_abs = abs(y_r - y_spl)
        e_rel = e_abs / abs(y_r) if abs(y_r) > 0 else 0
        print(f"  y = {y_obj:<10.4f} -> t_spline = {t_encontrado:.6f} s")
        print(f"                    y_real({t_encontrado:.4f}) = {y_r:.6f}")
        print(f"                    Error absoluto  = {e_abs:.6f}")
        print(f"                    Error relativo  = {e_rel:.6f} ({e_rel*100:.4f}%)")
        print()

print("-" * 70)

# --- 7c. Tabla resumen con datos completos ---
print("\n  TABLA COMPLETA:")
print("-" * 70)
print(f"  {'t (s)':<12} {'y(t) Spline':<16} {'y(t) Real':<16} {'Error Abs':<14} {'Error Rel':<14}")
print("-" * 70)

# Encontrar t para y=5
t_busqueda = np.linspace(t_datos[0], t_datos[-1], 10000)
y_busqueda = evaluar_spline(t_datos, a, b, c, d, t_busqueda)
for y_obj in [5.0, 8.4671]:
    for i in range(len(t_busqueda) - 1):
        if (y_busqueda[i] - y_obj) * (y_busqueda[i + 1] - y_obj) <= 0:
            frac = (y_obj - y_busqueda[i]) / (y_busqueda[i + 1] - y_busqueda[i])
            t_enc = t_busqueda[i] + frac * (t_busqueda[i + 1] - t_busqueda[i])
            yr = y_real(t_enc)
            ea = abs(yr - y_obj)
            er = ea / abs(yr)
            print(f"  {t_enc:<12.6f} {y_obj:<16.4f} {yr:<16.6f} {ea:<14.6f} {er:<14.6f}")
            break

for t_val in [1.2, 1.3]:
    y_spl = evaluar_spline(t_datos, a, b, c, d, np.array([t_val]))[0]
    yr = y_real(t_val)
    ea = abs(yr - y_spl)
    er = ea / abs(yr)
    print(f"  {t_val:<12.2f} {y_spl:<16.6f} {yr:<16.6f} {ea:<14.6f} {er:<14.6f}")

print("-" * 70)
print("=" * 70)


# ============================================================
# 8. Error Global: Error Cuadratico Medio (ECM)
# ============================================================
print("\n" + "=" * 70)
print("  ERROR GLOBAL - Error Cuadratico Medio (ECM)")
print("=" * 70)

# Evaluar en N puntos uniformemente distribuidos
N = 1000
t_ecm = np.linspace(t_datos[0], t_datos[-1], N)
y_spline_ecm = evaluar_spline(t_datos, a, b, c, d, t_ecm)
y_real_ecm = y_real(t_ecm)

# Error Cuadratico Medio: ECM = (1/N) * sum((y_real - y_spline)^2)
ecm = np.mean((y_real_ecm - y_spline_ecm) ** 2)

# Raiz del Error Cuadratico Medio (RECM)
recm = np.sqrt(ecm)

print(f"\n  Puntos de evaluacion (N):              {N}")
print(f"  Rango de evaluacion:                   [{t_datos[0]}, {t_datos[-1]}] s")
print(f"\n  Error Cuadratico Medio (ECM):           {ecm:.10f}")
print(f"  Raiz del Error Cuadratico Medio (RECM): {recm:.10f}")
print(f"\n  Error maximo absoluto:                  {error_max:.10f}")
print(f"  Error promedio absoluto:                {error_promedio:.10f}")
print("=" * 70)
