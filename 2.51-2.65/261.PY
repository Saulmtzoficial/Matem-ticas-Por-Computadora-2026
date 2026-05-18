"""
Exercise 2.61: Falling Object with Air Resistance

PROBLEMA:
Un objeto cayendo verticalmente está sujeto a fricción del aire y gravedad.

Posición: s(t) = s₀ - (mg/k)t + (m²g/k²)(1 - e^(-kt/m))

donde:
- m: masa (kg)
- g: gravedad (m/s²)
- s₀: posición inicial (m)
- k: coeficiente de resistencia del aire
- t: tiempo (s)

Preguntas:
1. ¿Cuáles son las unidades de k?
2. Con m=1kg, g=9.8m/s², k=0.1, s₀=100m, ¿cuánto tarda en tocar el suelo?
3. Si k tiene 10% de incertidumbre, ¿cómo afecta el tiempo?
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARTE 1: ANÁLISIS DIMENSIONAL
# ============================================================================

def analyze_units():
    """
    Análisis dimensional para determinar unidades de k
    """
    print("="*100)
    print("PARTE 1: ANÁLISIS DIMENSIONAL DE k")
    print("="*100)
    
    print("""
ECUACIÓN: s(t) = s₀ - (mg/k)t + (m²g/k²)(1 - e^(-kt/m))

ANÁLISIS DIMENSIONAL:

1. Término: s₀
   Unidades: [m] (metros)

2. Término: (mg/k)t
   - [s] debe tener unidades de metros [m]
   - mg tiene unidades: [kg][m/s²] = [kg·m/s²]
   - t tiene unidades: [s]
   - Entonces: [mg/k][t] = [m]
   - [kg·m/s²]/[k] × [s] = [m]
   - [kg·m/s]/[k] = [m]
   - [k] = [kg/s]

3. Término: (m²g/k²)(1 - e^(-kt/m))
   - El exponente debe ser adimensional
   - kt/m debe ser adimensional
   - [k][t]/[m] = adimensional
   - [k][s]/[kg] = adimensional
   - [k] = [kg/s]

4. Verificación del término completo:
   - m²g/k² debe tener unidades de [m]
   - [kg²][m/s²]/[k²] = [m]
   - [kg²·m/s²]/[k²] = [m]
   - [k²] = [kg²/s²]
   - [k] = [kg/s]

CONCLUSIÓN:
    k tiene unidades de [kg/s]
    
INTERPRETACIÓN FÍSICA:
    k representa una fuerza de resistencia por unidad de velocidad
    Fuerza de arrastre ∝ k × velocidad
    """)


# ============================================================================
# PARTE 2: RESOLVER PARA TIEMPO DE CAÍDA
# ============================================================================

def falling_object_position(t, m, g, k, s0):
    """
    Calcula la posición del objeto en el tiempo t
    
    s(t) = s₀ - (mg/k)t + (m²g/k²)(1 - e^(-kt/m))
    """
    term1 = s0
    term2 = -(m*g/k) * t
    term3 = (m**2 * g / k**2) * (1 - np.exp(-k*t/m))
    return term1 + term2 + term3


def find_impact_time(m, g, k, s0, tol=0.01):
    """
    Encuentra el tiempo cuando s(t) = 0 (toca el suelo)
    
    Usa el método de bisección para encontrar la raíz
    """
    # Función a resolver: s(t) = 0
    f = lambda t: falling_object_position(t, m, g, k, s0)
    
    # Encontrar intervalo apropiado
    # El objeto debe tocar el suelo eventualmente
    t_max = 2 * s0 / (m*g/k)  # Estimación conservadora
    
    # Verificar que hay cambio de signo
    if f(0) * f(t_max) > 0:
        t_max *= 2
    
    # Bisección
    a, b = 0, t_max
    history = []
    
    while (b - a) > tol:
        c = (a + b) / 2
        fc = f(c)
        history.append(c)
        
        if abs(fc) < 1e-10:
            return c, history
        
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    
    return (a + b) / 2, history


def solve_part_b():
    """
    Parte (b): Tiempo de caída con parámetros dados
    """
    print("\n" + "="*100)
    print("PARTE 2: TIEMPO DE CAÍDA")
    print("="*100)
    
    # Parámetros
    m = 1.0    # kg
    g = 9.8    # m/s²
    k = 0.1    # kg/s
    s0 = 100.0 # m
    
    print("\nPARÁMETROS:")
    print(f"  Masa:                m = {m} kg")
    print(f"  Gravedad:            g = {g} m/s²")
    print(f"  Resistencia:         k = {k} kg/s")
    print(f"  Altura inicial:      s₀ = {s0} m")
    print(f"  Precisión deseada:   {0.01} segundos")
    
    # Resolver
    t_impact, history = find_impact_time(m, g, k, s0, tol=0.01)
    
    print(f"\nRESULTADO:")
    print(f"  Tiempo de impacto:   t = {t_impact:.3f} segundos")
    print(f"  Iteraciones:         {len(history)}")
    print(f"  Verificación s(t):   {falling_object_position(t_impact, m, g, k, s0):.6f} m")
    
    # Gráfica de la trayectoria
    t_vals = np.linspace(0, t_impact*1.2, 1000)
    s_vals = [falling_object_position(t, m, g, k, s0) for t in t_vals]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Posición vs tiempo
    ax1.plot(t_vals, s_vals, 'b-', linewidth=2.5, label='Posición s(t)')
    ax1.axhline(y=0, color='brown', linestyle='--', linewidth=2, label='Suelo')
    ax1.axvline(x=t_impact, color='red', linestyle='--', linewidth=2, 
               label=f'Impacto t={t_impact:.3f}s')
    ax1.plot(t_impact, 0, 'ro', markersize=12, label='Punto de impacto')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Altura (m)', fontsize=11, fontweight='bold')
    ax1.set_title('Trayectoria del Objeto', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.set_ylim([-10, 110])
    
    # Velocidad vs tiempo
    # v(t) = ds/dt = -(mg/k) + (mg/k)e^(-kt/m)
    def velocity(t, m, g, k):
        return -(m*g/k) * (1 - np.exp(-k*t/m))
    
    v_vals = [velocity(t, m, g, k) for t in t_vals]
    v_terminal = -(m*g/k)
    
    ax2.plot(t_vals, v_vals, 'g-', linewidth=2.5, label='Velocidad v(t)')
    ax2.axhline(y=v_terminal, color='orange', linestyle='--', linewidth=2,
               label=f'Velocidad terminal = {v_terminal:.2f} m/s')
    ax2.axvline(x=t_impact, color='red', linestyle='--', linewidth=2,
               label=f'Impacto')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Velocidad (m/s)', fontsize=11, fontweight='bold')
    ax2.set_title('Velocidad del Objeto', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('exercise_2_61_trajectory.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVELOCIDAD TERMINAL:")
    print(f"  v_terminal = -mg/k = -{m*g/k:.3f} m/s")
    print(f"  Velocidad al impacto: {velocity(t_impact, m, g, k):.3f} m/s")
    
    return t_impact, m, g, k, s0


# ============================================================================
# PARTE 3: ANÁLISIS DE SENSIBILIDAD
# ============================================================================

def sensitivity_analysis(t_nominal, m, g, k_nominal, s0):
    """
    Análisis de sensibilidad: k con ±10% de incertidumbre
    """
    print("\n" + "="*100)
    print("PARTE 3: ANÁLISIS DE SENSIBILIDAD")
    print("="*100)
    
    print(f"\nVALOR NOMINAL:")
    print(f"  k = {k_nominal} kg/s")
    print(f"  t = {t_nominal:.3f} s")
    
    print(f"\nINCERTIDUMBRE:")
    print(f"  k puede variar ±10%")
    print(f"  Rango: [{k_nominal*0.9:.4f}, {k_nominal*1.1:.4f}] kg/s")
    
    # Calcular tiempos para diferentes k
    k_variations = np.linspace(k_nominal*0.9, k_nominal*1.1, 21)
    t_variations = []
    
    print(f"\n{'k (kg/s)':<15} {'Δk (%)':<12} {'t (s)':<12} {'Δt (s)':<12}")
    print("-"*60)
    
    for k_val in k_variations:
        t_val, _ = find_impact_time(m, g, k_val, s0, tol=0.001)
        t_variations.append(t_val)
        
        delta_k_percent = (k_val - k_nominal) / k_nominal * 100
        delta_t = t_val - t_nominal
        
        if abs(delta_k_percent) < 0.1 or abs(delta_k_percent - 10) < 0.1 or abs(delta_k_percent + 10) < 0.1:
            print(f"{k_val:<15.4f} {delta_k_percent:<12.1f} {t_val:<12.3f} {delta_t:<12.3f}")
    
    t_min = min(t_variations)
    t_max = max(t_variations)
    
    print("\n" + "="*60)
    print(f"RESULTADOS DEL ANÁLISIS DE SENSIBILIDAD:")
    print("="*60)
    print(f"  k mínimo (k - 10%):  {k_nominal*0.9:.4f} kg/s → t = {t_max:.3f} s")
    print(f"  k nominal:           {k_nominal:.4f} kg/s → t = {t_nominal:.3f} s")
    print(f"  k máximo (k + 10%):  {k_nominal*1.1:.4f} kg/s → t = {t_min:.3f} s")
    print(f"\n  RANGO DE TIEMPOS: [{t_min:.3f}, {t_max:.3f}] segundos")
    print(f"  VARIACIÓN EN t: ±{(t_max - t_min)/2:.3f} segundos")
    print(f"  VARIACIÓN RELATIVA: ±{(t_max - t_nominal)/t_nominal * 100:.2f}%")
    
    # Gráficas de sensibilidad
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfica 1: t vs k
    ax1 = axes[0, 0]
    ax1.plot(k_variations, t_variations, 'b-', linewidth=2.5)
    ax1.plot(k_nominal, t_nominal, 'ro', markersize=12, label='Valor nominal')
    ax1.axvline(x=k_nominal*0.9, color='green', linestyle='--', alpha=0.5, label='k - 10%')
    ax1.axvline(x=k_nominal*1.1, color='green', linestyle='--', alpha=0.5, label='k + 10%')
    ax1.fill_between(k_variations, t_variations, alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('k (kg/s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Tiempo de impacto (s)', fontsize=11, fontweight='bold')
    ax1.set_title('Sensibilidad: Tiempo vs k', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    
    # Gráfica 2: Δt vs Δk
    ax2 = axes[0, 1]
    delta_k = (k_variations - k_nominal) / k_nominal * 100
    delta_t = np.array(t_variations) - t_nominal
    
    ax2.plot(delta_k, delta_t, 'g-', linewidth=2.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Variación en k (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Variación en t (s)', fontsize=11, fontweight='bold')
    ax2.set_title('Cambio Relativo', fontsize=12, fontweight='bold')
    
    # Gráfica 3: Trayectorias para diferentes k
    ax3 = axes[1, 0]
    
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    k_samples = [k_nominal*0.9, k_nominal*0.95, k_nominal, k_nominal*1.05, k_nominal*1.1]
    
    for k_val, color in zip(k_samples, colors):
        t_impact_k, _ = find_impact_time(m, g, k_val, s0, tol=0.001)
        t_plot = np.linspace(0, t_impact_k, 500)
        s_plot = [falling_object_position(t, m, g, k_val, s0) for t in t_plot]
        
        label = f'k={k_val:.3f} (t={t_impact_k:.2f}s)'
        ax3.plot(t_plot, s_plot, linewidth=2, color=color, label=label)
    
    ax3.axhline(y=0, color='brown', linestyle='--', linewidth=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Altura (m)', fontsize=11, fontweight='bold')
    ax3.set_title('Trayectorias para Diferentes k', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    
    # Gráfica 4: Derivada dt/dk
    ax4 = axes[1, 1]
    
    # Calcular derivada numérica dt/dk
    dt_dk = np.gradient(t_variations, k_variations)
    
    ax4.plot(k_variations, dt_dk, 'm-', linewidth=2.5)
    ax4.axvline(x=k_nominal, color='r', linestyle='--', linewidth=2, label='k nominal')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('k (kg/s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('dt/dk (s·s/kg)', fontsize=11, fontweight='bold')
    ax4.set_title('Sensibilidad: Derivada dt/dk', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('exercise_2_61_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calcular sensibilidad relativa
    sensitivity = (delta_t[-1] - delta_t[0]) / (delta_k[-1] - delta_k[0]) * 100
    print(f"\nSENSIBILIDAD RELATIVA:")
    print(f"  (Δt/t) / (Δk/k) ≈ {sensitivity:.3f}")
    print(f"  Un cambio de 1% en k causa un cambio de ~{sensitivity:.2f}% en t")
    
    return k_variations, t_variations


# ============================================================================
# ANÁLISIS FÍSICO ADICIONAL
# ============================================================================

def physical_analysis():
    """
    Análisis físico del problema
    """
    print("\n" + "="*100)
    print("ANÁLISIS FÍSICO")
    print("="*100)
    
    print("""
FÍSICA DEL PROBLEMA:

1. ECUACIÓN DE MOVIMIENTO:
   ma = mg - kv
   donde v es la velocidad (negativa al caer)

2. SOLUCIÓN:
   s(t) = s₀ - (mg/k)t + (m²g/k²)(1 - e^(-kt/m))

3. VELOCIDAD TERMINAL:
   Cuando a → 0: mg = k|v_terminal|
   v_terminal = -mg/k
   
   Para los valores dados: v_terminal = -98 m/s

4. TIEMPO CARACTERÍSTICO:
   τ = m/k
   El objeto alcanza ~63% de la velocidad terminal en tiempo τ

5. LÍMITES:
   • k → 0 (sin resistencia): s(t) = s₀ - ½gt²  (caída libre)
   • k → ∞ (mucha resistencia): caída muy lenta

6. SENSIBILIDAD:
   • k más pequeño → más resistencia → caída más lenta
   • La relación NO es lineal (exponencial en la solución)
    """)


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("="*100)
    print("EJERCICIO 2.61: OBJETO EN CAÍDA CON RESISTENCIA DEL AIRE")
    print("="*100)
    
    # Parte 1: Unidades
    analyze_units()
    
    # Parte 2: Tiempo de caída
    t_impact, m, g, k, s0 = solve_part_b()
    
    # Parte 3: Sensibilidad
    k_vars, t_vars = sensitivity_analysis(t_impact, m, g, k, s0)
    
    # Análisis físico
    physical_analysis()
    
    print("\n" + "="*100)
    print("RESUMEN FINAL")
    print("="*100)
    print(f"""
1. UNIDADES DE k: kg/s (kilogramos por segundo)

2. TIEMPO DE IMPACTO: {t_impact:.3f} segundos
   (con precisión de 0.01 segundos)

3. ANÁLISIS DE SENSIBILIDAD (k ± 10%):
   • Rango de tiempos: [{min(t_vars):.3f}, {max(t_vars):.3f}] segundos
   • Variación: ±{(max(t_vars) - min(t_vars))/2:.3f} segundos
   • k menor → tiempo mayor (más resistencia → caída más lenta)
    """)
    
    print("="*100)
    print("✓ EJERCICIO 2.61 COMPLETADO")
    print("="*100)