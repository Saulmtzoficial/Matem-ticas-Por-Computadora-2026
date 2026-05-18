import numpy as np

# ── Funciones ──────────────────────────────────────────────────────────────────
def F(x, y):
    f1 = x**2 + x*y - 10
    f2 = y + 3*x*y**2 - 57
    return np.array([f1, f2])

# ── Jacobiana ──────────────────────────────────────────────────────────────────
def J(x, y):
    df1dx = 2*x + y
    df1dy = x
    df2dx = 3*y**2
    df2dy = 1 + 6*x*y
    return np.array([[df1dx, df1dy],
                     [df2dx, df2dy]])

# ── Newton-Raphson ─────────────────────────────────────────────────────────────
def newton_raphson(x0, y0, max_iter=7, tol=1e-10):
    x, y = x0, y0
    resultados = []

    for i in range(1, max_iter + 1):
        Fval = F(x, y)
        Jval = J(x, y)

        # Resolver J·Δ = -F
        delta = np.linalg.solve(Jval, -Fval)

        x_new = x + delta[0]
        y_new = y + delta[1]

        error = np.sqrt(delta[0]**2 + delta[1]**2)
        resultados.append((i, x_new, y_new, error))

        x, y = x_new, y_new

        if error < tol:
            break

    return resultados, x, y

# ── Ejecutar ───────────────────────────────────────────────────────────────────
x0, y0 = 4.5, -2.27
iters, x_sol, y_sol = newton_raphson(x0, y0, max_iter=7)

# ── Tabla de iteraciones ───────────────────────────────────────────────────────
sep  = "+" + "-"*6 + "+" + "-"*20 + "+" + "-"*20 + "+" + "-"*20 + "+"
head = f"| {'Iter':^4} | {'Xn':^18} | {'Yn':^18} | {'Error':^18} |"

print("\n" + "═"*70)
print("  MÉTODO DE NEWTON-RAPHSON — Sistema No Lineal")
print("  f1(x,y) = x² + xy − 10 = 0")
print("  f2(x,y) = y + 3xy² − 57 = 0")
print("  x₀ = 1.5 ,  y₀ = 3.5")
print("═"*70)
print(sep)
print(head)
print(sep)

for (it, xn, yn, err) in iters:
    print(f"| {it:^4d} | {xn:^18.10f} | {yn:^18.10f} | {err:^18.2e} |")

print(sep)



print(f"\n  SOLUCIÓN FINAL")
print(f"  x = {x_sol:.10f}")
print(f"  y = {y_sol:.10f}")
