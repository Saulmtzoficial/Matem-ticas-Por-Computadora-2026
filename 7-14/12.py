import numpy as np

def exact_derivative(x):
    """
    Computes the exact analytical derivative of f(x) = 1 + log10(x).
    The derivative is f'(x) = 1 / (x * ln(10)).
    """
    return 1 / (x * np.log(10))

def divided_difference(x, y):
    """Calculates the Newton divided differences."""
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1] - coef[i][j-1]) / (x[i+j] - x[i])
    return coef[0, :]

def analyze_points(x_points, y_points, target_x, next_x=None, next_y=None):
    """
    Computes the derivative using a 3-point quadratic fit, 
    calculates the true error, and estimates the next-term error.
    """
    x_array = np.array(x_points)
    y_array = np.array(y_points)
    
    # 1. Compute approximate derivative using quadratic interpolation
    poly_coeffs = np.polyfit(x_array, y_array, 2)
    deriv_coeffs = np.polyder(poly_coeffs)
    approx_deriv = np.polyval(deriv_coeffs, target_x)
    
    # 2. Compute exact derivative and True Error (Part a)
    true_deriv = exact_derivative(target_x)
    true_error = true_deriv - approx_deriv
    
    print(f"Points used: {x_points}")
    print(f"  Approximate f'({target_x}): {approx_deriv:.6f}")
    print(f"  Exact f'({target_x}):       {true_deriv:.6f}")
    print(f"  True Error:             {true_error:.6f}")
    
    # 3. Next-Term Rule Error Estimation (Part b)
    # The error of a quadratic fit is estimated using the 4th point
    if next_x is not None and next_y is not None:
        x_4pts = np.append(x_array, next_x)
        y_4pts = np.append(y_array, next_y)
        
        # Calculate divided differences for 4 points
        div_diffs = divided_difference(x_4pts, y_4pts)
        f_bracket_4 = div_diffs[3] # f[x0, x1, x2, x3]
        
        # The derivative of the next term: d/dx [ (x-x0)(x-x1)(x-x2) ] at target_x
        # d/dx = (x-x1)(x-x2) + (x-x0)(x-x2) + (x-x0)(x-x1)
        term_deriv = ((target_x - x_array[1]) * (target_x - x_array[2]) + 
                      (target_x - x_array[0]) * (target_x - x_array[2]) + 
                      (target_x - x_array[0]) * (target_x - x_array[1]))
        
        estimated_error = f_bracket_4 * term_deriv
        print(f"  Estimated Error (Next-Term Rule): {estimated_error:.6f}")
    else:
        print("  Estimated Error: Provide a 4th point for the next-term rule.")
    print("-" * 40)

# ==========================================
# ENTER YOUR EXERCISE 11 DATA HERE
# ==========================================
target = 0.268

# Example Set 1 (From Exercise 11)
# Replace with the 3 points you actually used, plus the 4th point for the next-term rule
set1_x = [0.200, 0.300, 0.400] 
set1_y = [1.221, 1.350, 1.492] # Replace with actual 1 + log10(x) values if needed
next_pt1_x = 0.500
next_pt1_y = 1.649

# Example Set 2 (Other sets of points for Part c)
set2_x = [0.100, 0.200, 0.300]
set2_y = [1.105, 1.221, 1.350]
next_pt2_x = 0.400
next_pt2_y = 1.492

print("--- Analysis for Exercise 12 ---")
# Run Part A and B for the original set
analyze_points(set1_x, set1_y, target, next_pt1_x, next_pt1_y)

# Run Part C (Repeat for other sets of points)
analyze_points(set2_x, set2_y, target, next_pt2_x, next_pt2_y)