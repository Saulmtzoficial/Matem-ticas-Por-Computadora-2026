import numpy as np

def compute_derivative(x_table, y_table, target_x):
    """
    Computes the derivative of a quadratic interpolating polynomial
    at a target_x using the 3 closest points from a given table.
    """
    x_array = np.array(x_table)
    y_array = np.array(y_table)
    
    # 1. Find the 3 points whose x-values are closest to the target_x
    # Calculate the absolute difference between each x and the target
    differences = np.abs(x_array - target_x)
    
    # Get the indices of the 3 smallest differences
    closest_indices = np.argsort(differences)[:3]
    
    # Extract the x and y values for these 3 points
    x_points = x_array[closest_indices]
    y_points = y_array[closest_indices]
    
    print("Which points are these?")
    print("To get the most accurate answer, we use the 3 points closest to the target:")
    for x, y in zip(x_points, y_points):
        print(f"  (x = {x}, y = {y})")
        
    # 2. Fit a quadratic interpolating polynomial (degree = 2)
    # This returns the coefficients [a, b, c] for ax^2 + bx + c
    poly_coeffs = np.polyfit(x_points, y_points, 2)
    
    # 3. Compute the derivative of the polynomial
    # This returns the coefficients for 2ax + b
    deriv_coeffs = np.polyder(poly_coeffs)
    
    # 4. Evaluate the derivative at the target_x
    derivative_value = np.polyval(deriv_coeffs, target_x)
    
    return derivative_value

# ==========================================
# ENTER YOUR TABLE DATA HERE
# ==========================================
# Replace these dummy values with the actual numbers from your problem
table_x = [0.100, 0.200, 0.300, 0.400, 0.500] 
table_y = [1.105, 1.221, 1.350, 1.492, 1.649] 

target = 0.268

# Run the calculation
print("--- Results ---")
result = compute_derivative(table_x, table_y, target)
print(f"\nf'({target}) ≈ {result:.5f}")