import numpy as np
# --- Parámetros del sistema ---
a = 500  # Distancia entre estaciones A y B [m]
h = 1    # Intervalo de tiempo [s]

# --- Mediciones angulares (en grados) ---
t = np.array([9, 10, 11])                          # Instantes de tiempo [s]
alpha_deg = np.array([54.80, 54.06, 53.34])         # Ángulo α [°]
beta_deg  = np.array([65.59, 64.59, 63.62])         # Ángulo β [°]

# Conversión a radianes
alpha = np.radians(alpha_deg)
beta  = np.radians(beta_deg)

tan_alpha = np.tan(alpha)
tan_beta  = np.tan(beta)

x = a * tan_beta / (tan_beta - tan_alpha)
y = a * (tan_alpha * tan_beta) / (tan_beta - tan_alpha)

print("=" * 60)
print("1. COORDENADAS DE POSICION")
print("=" * 60)
print(f"{'t [s]':>8} {'alfa [deg]':>12} {'beta [deg]':>12} {'x [m]':>12} {'y [m]':>12}")
print("-" * 60)
for i in range(len(t)):
    print(f"{t[i]:>8.0f} {alpha_deg[i]:>12.2f} {beta_deg[i]:>12.2f} {x[i]:>12.4f} {y[i]:>12.4f}")


Vx = (x[2] - x[0]) / (2 * h)
Vy = (y[2] - y[0]) / (2 * h)

print()
print("=" * 60)
print("2. COMPONENTES DE VELOCIDAD EN t = 10 s")
print("   (Diferencias centrales de orden O(h^2))")
print("=" * 60)
print(f"   Vx = (x(11) - x(9)) / (2*h)")
print(f"   Vx = ({x[2]:.4f} - {x[0]:.4f}) / (2*{h})")
print(f"   Vx = {Vx:.4f} m/s")
print()
print(f"   Vy = (y(11) - y(9)) / (2*h)")
print(f"   Vy = ({y[2]:.4f} - {y[0]:.4f}) / (2*{h})")
print(f"   Vy = {Vy:.4f} m/s")

v = np.sqrt(Vx**2 + Vy**2)
gamma_rad = np.arctan2(Vy, Vx)
gamma_deg = np.degrees(gamma_rad)

print()
print("=" * 60)
print("3. METRICAS DE VUELO EN t = 10 s")
print("=" * 60)
print(f"   Rapidez total:     v = sqrt(Vx^2 + Vy^2)")
print(f"                      v = sqrt({Vx:.4f}^2 + {Vy:.4f}^2)")
print(f"                      v = {v:.4f} m/s")
print()
print(f"   Angulo de ascenso: gamma = arctan(Vy / Vx)")
print(f"                      gamma = arctan({Vy:.4f} / {Vx:.4f})")
print(f"                      gamma = {gamma_rad:.6f} rad")
print(f"                      gamma = {gamma_deg:.4f} grados")
