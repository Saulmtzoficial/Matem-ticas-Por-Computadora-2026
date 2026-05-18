import numpy as np
import matplotlib.pyplot as plt

def exact_derivative(x):
    """Computes the exact derivative of f(x) = 1 + log10(x)"""
    return 1 / (x * np.log(10))

# Reconstructing the table from previous exercises (assuming x = 0.1 to 0.5)
table_x = np.array([0.100, 0.200, 0.300, 0.400, 0.500])
table_y = 1 + np.log10(table_x) # Using exact y-values for the base table

# The target x values requested in Exercise 14
targets = np.array([0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27])

estimates = []
true_values = []
errors = []

print("--- Calculating Estimates ---")
for target in targets:
    # 1. Find the 3 closest points in the table
    diffs = np.abs(table_x - target)
    closest_idx = np.argsort(diffs)[:3]
    x_pts = table_x[closest_idx]
    y_pts = table_y[closest_idx]
    
    # 2. Fit a quadratic polynomial and find its derivative
    poly_coeffs = np.polyfit(x_pts, y_pts, 2)
    deriv_coeffs = np.polyder(poly_coeffs)
    est = np.polyval(deriv_coeffs, target)
    estimates.append(est)
    
    # 3. Calculate true value and absolute error
    true_val = exact_derivative(target)
    true_values.append(true_val)
    
    error = np.abs(true_val - est)
    errors.append(error)
    
    print(f"x = {target:.2f} | Est: {est:.6f} | True: {true_val:.6f} | Error: {error:.6f}")

# Find where the error is smallest
min_error_idx = np.argmin(errors)
best_x = targets[min_error_idx]
min_error_val = errors[min_error_idx]

print("\n--- Conclusion ---")
print(f"The error is smallest at x = {best_x} with an error of {min_error_val:.8f}")

# ==========================================
# PLOTTING
# ==========================================

# Plot 1: Estimates vs True Values
plt.figure(figsize=(10, 5))
plt.plot(targets, true_values, 'b-', linewidth=2, label='True Derivative')
plt.plot(targets, estimates, 'ro--', markersize=6, label='Quadratic Estimate')
plt.title("Derivative Estimates vs True Values")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot 2: Errors vs x
plt.figure(figsize=(10, 5))
plt.plot(targets, errors, 'g^-', markersize=8, label='Absolute Error')
plt.axvline(best_x, color='r', linestyle=':', label=f'Smallest Error at x={best_x}')
plt.title("Interpolation Error versus x")
plt.xlabel("x")
plt.ylabel("Absolute Error |True - Estimate|")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()