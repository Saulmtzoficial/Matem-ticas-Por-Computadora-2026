import numpy as np
import matplotlib.pyplot as plt

# Función dada
def f(x):
    return (2**x) * np.cos(2*x)

# Construcción de la tabla de diferencias divididas
def divided_diff(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)

    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])

    return coef

# Evaluar polinomio de Newton
def newton_poly(coef, x_data, x):
    n = len(coef)
    p = coef[n-1]

    for k in range(1, n):
        p = coef[n-k-1] + (x - x_data[n-k-1]) * p

    return p

# Derivada numérica del polinomio de Newton (diferencia central)
def derivative_newton(coef, x_data, x, h=1e-5):
    return (newton_poly(coef, x_data, x + h) - newton_poly(coef, x_data, x - h)) / (2*h)

# -----------------------------
# Datos (puedes ajustarlos)
# -----------------------------
x_data = np.array([1.6, 1.8, 2.0, 2.2, 2.4])
y_data = f(x_data)

# Tabla de diferencias divididas
coef = divided_diff(x_data, y_data)

print("Coeficientes de Newton:")
print(coef)

# Aproximación de la derivada en x = 2.0
x_eval = 2.0
f_prime_approx = derivative_newton(coef, x_data, x_eval)

print(f"\nAproximación de f'(2.0): {f_prime_approx}")

# -----------------------------
# Gráfica
# -----------------------------
x_plot = np.linspace(1.5, 2.7, 200)
y_real = f(x_plot)
y_interp = [newton_poly(coef, x_data, xi) for xi in x_plot]

plt.figure()
plt.plot(x_plot, y_real, label="f(x) real")
plt.plot(x_plot, y_interp, '--', label="Interpolación (Newton)")
plt.scatter(x_data, y_data, color='red', label="Datos")
plt.axvline(2.0, linestyle=':', label="x = 2.0")

plt.legend()
plt.title("Interpolación y aproximación de derivada")
plt.grid()
plt.show()