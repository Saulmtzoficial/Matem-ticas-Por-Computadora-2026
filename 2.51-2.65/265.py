"""
Exercise 2.65: Using scipy.optimize.fsolve()

PROBLEMA:
Explorar y usar scipy.optimize.fsolve() para resolver ecuaciones y sistemas.

1. Familiarizarse con fsolve
2. Resolver x*sin(x) - ln(x) = 0 desde x₀=3
3. Demostrar uso con ecuación no trivial
4. Resolver sistema de ecuaciones
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

print("="*100)
print("EJERCICIO 2.65: scipy.optimize.fsolve()")
print("="*100)

# ============================================================================
# PARTE 1: DOCUMENTACIÓN
# ============================================================================

print("\n" + "="*100)
print("PARTE 1: DOCUMENTACIÓN DE fsolve")
print("="*100)

print("""
SINTAXIS:
    fsolve(func, x0, args=(), fprime=None, full_output=0, ...)

PARÁMETROS PRINCIPALES:
    func:         Función a resolver (debe retornar 0 en la raíz)
    x0:           Aproximación inicial
    args:         Argumentos adicionales para func
    fprime:       Jacobiano (opcional, calculado numéricamente si no se da)
    full_output:  Si es 1, retorna información diagnóstica completa

RETORNA:
    Si full_output=0: solo la raíz
    Si full_output=1: (raíz, infodict, ier, mesg)
        - raíz: solución encontrada
        - infodict: diccionario con info diagnóstica
        - ier: código de estado (1=éxito)
        - mesg: mensaje describiendo la causa de terminación
""")

# ============================================================================
# PARTE 2: RESOLVER x*sin(x) - ln(x) = 0
# ============================================================================

print("\n" + "="*100)
print("PARTE 2: RESOLVER x*sin(x) - ln(x) = 0")
print("="*100)

# 2a) Graficar la función
print("\n2a) VISUALIZACIÓN DE LA FUNCIÓN")
print("-"*100)

def f_part2(x):
    """f(x) = x*sin(x) - ln(x)"""
    return x * np.sin(x) - np.log(x)

x = np.linspace(0.1, 5, 1000)
y = f_part2(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = x·sin(x) - ln(x)')
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12, fontweight='bold')
plt.ylabel('f(x)', fontsize=12, fontweight='bold')
plt.title('Gráfica de f(x) = x·sin(x) - ln(x)', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)

# Marcar aproximadamente donde cruza cero
zero_crossings = []
for i in range(len(y)-1):
    if y[i] * y[i+1] < 0:
        zero_crossings.append(x[i])
        plt.plot(x[i], 0, 'ro', markersize=10)
        plt.annotate(f'Raíz ≈ {x[i]:.2f}', xy=(x[i], 0), 
                    xytext=(x[i], -1), fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('exercise_2_65_part2a.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Observación: La función cruza el eje x aproximadamente en x ≈ {zero_crossings[0]:.3f}")

# 2b) Usar fsolve
print("\n2b) USANDO fsolve() CON full_output=1")
print("-"*100)

x0 = 3.0
print(f"Aproximación inicial: x₀ = {x0}")

# Resolver con full_output=1
solution, infodict, ier, mesg = fsolve(f_part2, x0, full_output=1)

print(f"\nSOLUCIÓN ENCONTRADA:")
print(f"  x = {solution[0]:.15f}")
print(f"  f(x) = {f_part2(solution[0]):.6e}")

# 2c) Explicar outputs
print("\n2c) EXPLICACIÓN COMPLETA DE OUTPUTS")
print("-"*100)

print(f"\n1. SOLUCIÓN (solution):")
print(f"   Valor: {solution}")
print(f"   Tipo: {type(solution)}")
print(f"   Explicación: Array con la(s) raíz(raíces) encontrada(s)")

print(f"\n2. DICCIONARIO DE INFORMACIÓN (infodict):")
print(f"   Tipo: {type(infodict)}")
print(f"   Claves: {list(infodict.keys())}")

print(f"\n   'nfev': {infodict['nfev']}")
print(f"      → Número de evaluaciones de función")

print(f"\n   'fvec': {infodict['fvec']}")
print(f"      → Vector de valores de función en la solución")
print(f"      → Debe estar cercano a cero")

print(f"\n   'fjac': (matriz {infodict['fjac'].shape})")
print(f"      → Matriz Jacobiana en la solución")
print(f"      → Usada internamente por el algoritmo")

print(f"\n   'r': (array de longitud {len(infodict['r'])})")
print(f"      → Factorización QR del Jacobiano")

print(f"\n   'qtf': {infodict['qtf']}")
print(f"      → Q^T · f(x) donde Q es de la factorización QR")

print(f"\n3. CÓDIGO DE ESTADO (ier):")
print(f"   Valor: {ier}")
print(f"   Explicación:")
if ier == 1:
    print(f"      ✓ ier = 1: ÉXITO - Solución encontrada correctamente")
else:
    print(f"      ✗ ier ≠ 1: Hubo un problema")

print(f"\n4. MENSAJE (mesg):")
print(f"   '{mesg}'")
print(f"   Explicación: Descripción detallada del resultado")

print(f"\nVERIFICACIÓN:")
print(f"  f({solution[0]:.10f}) = {f_part2(solution[0]):.15e}")
print(f"  {'✓ Cercano a cero!' if abs(f_part2(solution[0])) < 1e-10 else '✗ No es raíz'}")

# ============================================================================
# PARTE 3: ECUACIÓN NO TRIVIAL
# ============================================================================

print("\n" + "="*100)
print("PARTE 3: ECUACIÓN NO TRIVIAL - OPCIONES DE fsolve")
print("="*100)

print("\nECUACIÓN: e^x - 3x² = 0")

def f_part3(x):
    """f(x) = e^x - 3x²"""
    return np.exp(x) - 3*x**2

# Graficar
x = np.linspace(-1, 4, 1000)
y = f_part3(x)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='f(x) = e^x - 3x²')
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=11, fontweight='bold')
plt.ylabel('f(x)', fontsize=11, fontweight='bold')
plt.title('Función No Trivial', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)

# Opción 1: Sin opciones especiales
print("\nOPCIÓN 1: Uso básico")
sol1 = fsolve(f_part3, 0.5)
print(f"  x₀ = 0.5 → solución: {sol1[0]:.10f}")

# Opción 2: Con Jacobiano
print("\nOPCIÓN 2: Proporcionando el Jacobiano (fprime)")
def df_part3(x):
    """Derivada: f'(x) = e^x - 6x"""
    return np.exp(x) - 6*x

sol2 = fsolve(f_part3, 0.5, fprime=df_part3)
print(f"  x₀ = 0.5 con Jacobiano → solución: {sol2[0]:.10f}")
print(f"  (Puede ser más rápido con Jacobiano analítico)")

# Opción 3: Con argumentos adicionales
print("\nOPCIÓN 3: Con argumentos adicionales (args)")
def f_with_param(x, a):
    """f(x) = e^x - a·x²"""
    return np.exp(x) - a*x**2

sol3 = fsolve(f_with_param, 0.5, args=(3,))
print(f"  Resolviendo e^x - 3x² = 0 con args=(3,)")
print(f"  Solución: {sol3[0]:.10f}")

# Opción 4: Diferentes aproximaciones iniciales
print("\nOPCIÓN 4: Sensibilidad a x₀")
x0_values = [0.1, 0.5, 1.0, 3.0, 4.0]
solutions = []

plt.subplot(1, 2, 2)
plt.plot(x, y, 'b-', linewidth=2, label='f(x)')
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)

for x0 in x0_values:
    sol = fsolve(f_part3, x0)
    solutions.append(sol[0])
    plt.plot(x0, f_part3(x0), 'o', markersize=8, label=f'x₀={x0}→x={sol[0]:.2f}')
    plt.plot(sol[0], 0, 'X', markersize=12)
    print(f"  x₀ = {x0:.1f} → x = {sol[0]:.10f}")

plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=11, fontweight='bold')
plt.ylabel('f(x)', fontsize=11, fontweight='bold')
plt.title('Diferentes Aproximaciones Iniciales', fontsize=12, fontweight='bold')
plt.legend(fontsize=8)

plt.tight_layout()
plt.savefig('exercise_2_65_part3.png', dpi=150, bbox_inches='tight')
plt.show()

# Identificar raíces únicas
unique_sols = []
for sol in solutions:
    if not any(abs(sol - us) < 0.01 for us in unique_sols):
        unique_sols.append(sol)

print(f"\nRAÍCES ENCONTRADAS: {len(unique_sols)}")
for i, sol in enumerate(sorted(unique_sols), 1):
    print(f"  Raíz {i}: x = {sol:.10f}")

# ============================================================================
# PARTE 4: SISTEMA DE ECUACIONES (EJEMPLO DADO)
# ============================================================================

print("\n" + "="*100)
print("PARTE 4: SISTEMA DE ECUACIONES")
print("="*100)

print("\nSISTEMA:")
print("  x₀·cos(x₁) = 4")
print("  x₀·x₁ - x₁ = 5")

def equations_part4(x):
    """
    Sistema de ecuaciones:
    x[0]*cos(x[1]) - 4 = 0
    x[0]*x[1] - x[1] - 5 = 0
    """
    eq1 = x[0] * np.cos(x[1]) - 4
    eq2 = x[0] * x[1] - x[1] - 5
    return [eq1, eq2]

# Aproximación inicial
x0_system = [1, 1]
print(f"\nAproximación inicial: x₀ = {x0_system}")

# Resolver
solution_sys = fsolve(equations_part4, x0_system)

print(f"\nSOLUCIÓN:")
print(f"  x₀ = {solution_sys[0]:.10f}")
print(f"  x₁ = {solution_sys[1]:.10f}")

print(f"\nVERIFICACIÓN:")
print(f"  x₀·cos(x₁) = {solution_sys[0] * np.cos(solution_sys[1]):.10f} (debe ser 4)")
print(f"  x₀·x₁ - x₁ = {solution_sys[0]*solution_sys[1] - solution_sys[1]:.10f} (debe ser 5)")

residuals = equations_part4(solution_sys)
print(f"  Residuos: {residuals}")

# ============================================================================
# PARTE 5: NUEVO SISTEMA
# ============================================================================

print("\n" + "="*100)
print("PARTE 5: RESOLVER NUEVO SISTEMA")
print("="*100)

print("\nSISTEMA:")
print("  x² - xy² = 2")
print("  xy = 2")

def equations_part5(vars):
    """
    Sistema:
    x² - xy² = 2
    xy = 2
    """
    x, y = vars
    eq1 = x**2 - x*y**2 - 2
    eq2 = x*y - 2
    return [eq1, eq2]

# Probar diferentes aproximaciones iniciales
print("\nPROBANDO DIFERENTES APROXIMACIONES INICIALES:")

initial_guesses = [
    [1, 1],
    [2, 1],
    [-2, -1],
    [3, 0.5]
]

solutions_part5 = []

for x0 in initial_guesses:
    sol = fsolve(equations_part5, x0)
    x_sol, y_sol = sol
    
    # Verificar
    eq1_check = x_sol**2 - x_sol*y_sol**2
    eq2_check = x_sol*y_sol
    
    print(f"\nx₀ = {x0}")
    print(f"  Solución: x = {x_sol:.10f}, y = {y_sol:.10f}")
    print(f"  Verificación:")
    print(f"    x² - xy² = {eq1_check:.10f} (debe ser 2)")
    print(f"    xy = {eq2_check:.10f} (debe ser 2)")
    
    # Guardar si es única
    is_unique = True
    for prev_sol in solutions_part5:
        if abs(x_sol - prev_sol[0]) < 0.01 and abs(y_sol - prev_sol[1]) < 0.01:
            is_unique = False
            break
    
    if is_unique:
        solutions_part5.append([x_sol, y_sol])

print(f"\nSOLUCIONES ÚNICAS ENCONTRADAS: {len(solutions_part5)}")
for i, sol in enumerate(solutions_part5, 1):
    print(f"  Solución {i}: x = {sol[0]:.10f}, y = {sol[1]:.10f}")

# Visualización del sistema
x_range = np.linspace(-3, 3, 400)
y_range = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_range, y_range)

# Ecuación 1: x² - xy² = 2
Z1 = X**2 - X*Y**2 - 2

# Ecuación 2: xy = 2
Z2 = X*Y - 2

plt.figure(figsize=(10, 8))
plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2, label='x² - xy² = 2')
plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, label='xy = 2')

# Marcar soluciones
for sol in solutions_part5:
    plt.plot(sol[0], sol[1], 'go', markersize=12, markeredgecolor='black', markeredgewidth=2)
    plt.annotate(f'({sol[0]:.2f}, {sol[1]:.2f})', 
                xy=(sol[0], sol[1]), xytext=(sol[0]+0.3, sol[1]+0.3),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.grid(True, alpha=0.3)
plt.xlabel('x', fontsize=12, fontweight='bold')
plt.ylabel('y', fontsize=12, fontweight='bold')
plt.title('Sistema: x² - xy² = 2, xy = 2', fontsize=13, fontweight='bold')
plt.legend(['x² - xy² = 2', 'xy = 2'], fontsize=11)
plt.axis('equal')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

plt.tight_layout()
plt.savefig('exercise_2_65_part5.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "="*100)
print("RESUMEN")
print("="*100)

print("""
scipy.optimize.fsolve() es una herramienta poderosa que:

1. RESUELVE ECUACIONES: Encuentra raíces de f(x) = 0
2. RESUELVE SISTEMAS: Puede manejar múltiples ecuaciones simultáneas
3. OPCIONES FLEXIBLES: 
   - full_output para diagnósticos
   - fprime para Jacobiano
   - args para parámetros adicionales
4. BASADO EN HYBRD: Usa algoritmo híbrido (Powell)
5. ROBUSTO: Generalmente converge si x₀ es razonable

VENTAJAS vs NUESTROS MÉTODOS:
  ✓ Optimizado y probado
  ✓ Maneja sistemas de ecuaciones
  ✓ Diagnósticos detallados
  ✓ No requiere programación del método

DESVENTAJAS:
  ✗ "Caja negra" - menos comprensión
  ✗ Sensible a x₀
  ✗ Puede encontrar solo una raíz
""")

print("\n" + "="*100)
print("✓ EJERCICIO 2.65 COMPLETADO")
print("="*100)