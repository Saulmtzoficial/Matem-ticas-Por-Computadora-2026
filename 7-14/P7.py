import numpy as np

# ─────────────────────────────────────────────
#  f(x) = 2x * cos(2x)   y   f'(x) exacta
# ─────────────────────────────────────────────
def f(x):
    return 2 * x * np.cos(2 * x)

def f_exacta(x):
    return 2 * np.cos(2 * x) - 4 * x * np.sin(2 * x)


# ─────────────────────────────────────────────
#  Construcción de la tabla de diferencias
#  divididas (igual que Tabla 5.3 del libro)
# ─────────────────────────────────────────────
def tabla_diferencias_divididas(xs):
    n = len(xs)
    # F[j][i] = diferencia dividida de orden j a partir de x_i
    F = [list(f(xs))]                          # orden 0
    for j in range(1, n):
        col = []
        for i in range(n - j):
            val = (F[j-1][i+1] - F[j-1][i]) / (xs[i+j] - xs[i])
            col.append(val)
        F.append(col)
    return F


def imprimir_tabla(xs, F):
    n = len(xs)
    # Encabezado
    encabezado = f"{'i':>3}  {'x_i':>6}  {'f_i':>10}"
    for k in range(1, n):
        encabezado += f"  {'f[xi..xi+'+str(k)+']':>14}"
    print(encabezado)
    print("─" * len(encabezado))

    for i in range(n):
        fila = f"{i:>3}  {xs[i]:>6.2f}  {F[0][i]:>10.4f}"
        for j in range(1, n):
            if i < n - j:
                fila += f"  {F[j][i]:>14.4f}"
            else:
                fila += f"  {'':>14}"
        print(fila)


# ─────────────────────────────────────────────
#  Aproximaciones de la derivada en x = x_eval
# ─────────────────────────────────────────────
def aproximaciones(xs, h, x_eval):
    idx = np.argmin(np.abs(xs - x_eval))   # índice más cercano a x_eval

    # Diferencia hacia adelante
    if idx + 1 < len(xs):
        fwd = (f(xs[idx + 1]) - f(xs[idx])) / h
    else:
        fwd = None

    # Diferencia hacia atrás
    if idx - 1 >= 0:
        bwd = (f(xs[idx]) - f(xs[idx - 1])) / h
    else:
        bwd = None

    # Diferencia central
    if idx - 1 >= 0 and idx + 1 < len(xs):
        ctr = (f(xs[idx + 1]) - f(xs[idx - 1])) / (2 * h)
    else:
        ctr = None

    return fwd, bwd, ctr


# ─────────────────────────────────────────────
#  PARÁMETROS  (mismos puntos que Tabla 5.3)
# ─────────────────────────────────────────────
xs    = np.array([1.70, 1.80, 2.00, 2.35, 2.50])
h     = 0.10          # paso entre puntos uniformes
x_eval = 2.00         # punto donde se evalúa f'


# ─────────────────────────────────────────────
#  EJECUCIÓN
# ─────────────────────────────────────────────
print("\n" + "═"*65)
print("  Tabla de diferencias divididas  —  f(x) = 2x·cos(2x)")
print("═"*65)

F = tabla_diferencias_divididas(xs)
imprimir_tabla(xs, F)

print("\n" + "═"*65)
print(f"  Aproximaciones de f'({x_eval})")
print("═"*65)

fwd, bwd, ctr = aproximaciones(xs, h, x_eval)
exacto = f_exacta(x_eval)

etiquetas = [
    ("a) Diferencia hacia adelante (forward) ", fwd),
    ("b) Diferencia hacia atrás   (backward) ", bwd),
    ("c) Diferencia central       (central)  ", ctr),
]

for nombre, val in etiquetas:
    if val is not None:
        error = abs(val - exacto) / abs(exacto) * 100
        print(f"  {nombre}: {val:>12.6f}   (error rel. = {error:.4f}%)")
    else:
        print(f"  {nombre}: no disponible (puntos insuficientes)")

print(f"\n  Valor exacto f'({x_eval})           : {exacto:>12.6f}")
print("═"*65 + "\n")