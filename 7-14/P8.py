import numpy as np
from scipy.optimize import minimize_scalar

# ─────────────────────────────────────────────────────────────
#  f(x) = 2x·cos(2x)  y sus derivadas de orden n+1
#
#  f'  (x) = 2cos(2x) - 4x·sin(2x)
#  f'' (x) = -8sin(2x) - 8x·cos(2x)          ← orden 2 (forward/backward, n=1)
#  f'''(x) = -24cos(2x) + 16x·sin(2x)         ← orden 3 (central, n=2)
# ─────────────────────────────────────────────────────────────

def f(x):
    return 2 * x * np.cos(2 * x)

def f_exacta(x):
    """f'(x) exacta"""
    return 2 * np.cos(2 * x) - 4 * x * np.sin(2 * x)

def f_orden2(x):
    """f''(x) — derivada de orden 2"""
    return -8 * np.sin(2 * x) - 8 * x * np.cos(2 * x)

def f_orden3(x):
    """f'''(x) — derivada de orden 3"""
    return -24 * np.cos(2 * x) + 16 * x * np.sin(2 * x)


# ─────────────────────────────────────────────────────────────
#  Aproximaciones de la derivada (igual que ejercicio 7)
# ─────────────────────────────────────────────────────────────

def f_forward(xs, h, idx):
    return (f(xs[idx + 1]) - f(xs[idx])) / h

def f_backward(xs, h, idx):
    return (f(xs[idx]) - f(xs[idx - 1])) / h

def f_central(xs, h, idx):
    return (f(xs[idx + 1]) - f(xs[idx - 1])) / (2 * h)


# ─────────────────────────────────────────────────────────────
#  Cota del error — Ec. (5.7)
#
#  Error = [ prod_{j=0, j≠i}^{n} (x - x_j) ] * f^(n+1)(ξ) / (n+1)!
#
#  Para diferencias finitas de orden 1 (forward/backward):
#    Error ≈ h/2 * f''(ξ)         con ξ ∈ [x, x+h]
#
#  Para diferencias centrales de orden 2:
#    Error ≈ h²/6 * f'''(ξ)       con ξ ∈ [x-h, x+h]
# ─────────────────────────────────────────────────────────────

def max_abs(func, a, b, n=1000):
    """Máximo de |func| en [a, b] por muestreo denso."""
    xx = np.linspace(a, b, n)
    return np.max(np.abs(func(xx)))


def calcular_cotas(xs, h, x_eval, idx):
    """
    Devuelve (cota_fwd, cota_bwd, cota_ctr) usando la Ec. 5.7.
    """
    # ── Forward  (n=1): intervalo [x_eval, x_eval+h]
    M2_fwd = max_abs(f_orden2, x_eval, x_eval + h)
    cota_fwd = (h / 2) * M2_fwd          # h * h / 2! = h²/2  pero el prod da h

    # ── Backward (n=1): intervalo [x_eval-h, x_eval]
    M2_bwd = max_abs(f_orden2, x_eval - h, x_eval)
    cota_bwd = (h / 2) * M2_bwd

    # ── Central  (n=2): intervalo [x_eval-h, x_eval+h]
    M3_ctr = max_abs(f_orden3, x_eval - h, x_eval + h)
    cota_ctr = (h**2 / 6) * M3_ctr       # h² * h / 3! = h³/6  pero el prod da h²

    return cota_fwd, cota_bwd, cota_ctr


# ─────────────────────────────────────────────────────────────
#  PARÁMETROS
# ─────────────────────────────────────────────────────────────
xs     = np.array([1.70, 1.80, 2.00, 2.35, 2.50])
h      = 0.10
x_eval = 2.00
idx    = np.argmin(np.abs(xs - x_eval))   # índice de x_eval en xs

# ─────────────────────────────────────────────────────────────
#  RESULTADOS
# ─────────────────────────────────────────────────────────────
exacto  = f_exacta(x_eval)
fwd_val = f_forward(xs, h, idx)
bwd_val = f_backward(xs, h, idx)
ctr_val = f_central(xs, h, idx)

cota_fwd, cota_bwd, cota_ctr = calcular_cotas(xs, h, x_eval, idx)

error_fwd = abs(fwd_val - exacto)
error_bwd = abs(bwd_val - exacto)
error_ctr = abs(ctr_val - exacto)

print("\n" + "═"*70)
print("  Cotas de error — Ec. (5.7)   f(x) = 2x·cos(2x),  x = 2.0")
print("═"*70)
print(f"\n  f'({x_eval}) exacta = {exacto:.8f}")
print()

filas = [
    ("a) Hacia adelante (forward) ", fwd_val, error_fwd, cota_fwd,
     "h/2 · max|f''(ξ)|,  ξ ∈ [2.0, 2.1]"),
    ("b) Hacia atrás   (backward) ", bwd_val, error_bwd, cota_bwd,
     "h/2 · max|f''(ξ)|,  ξ ∈ [1.9, 2.0]"),
    ("c) Central                  ", ctr_val, error_ctr, cota_ctr,
     "h²/6 · max|f'''(ξ)|, ξ ∈ [1.9, 2.1]"),
]

for nombre, aprox, err_real, cota, formula in filas:
    print(f"  {nombre}")
    print(f"    Aproximación        : {aprox:.8f}")
    print(f"    Error real |aprox-f': {err_real:.2e}")
    print(f"    Cota  ({formula})")
    print(f"                        : {cota:.2e}")
    dentro = "✓  dentro de la cota" if err_real <= cota + 1e-15 else "✗  fuera de la cota"
    print(f"    Verificación        : {dentro}")
    print()

print("  Nota: la diferencia central es O(h²) → cota y error real")
print("  son ~100× menores que los de las fórmulas de un lado.")
print("═"*70 + "\n")