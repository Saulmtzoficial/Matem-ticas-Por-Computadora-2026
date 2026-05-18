import numpy as np

def solve_systems():
    # --- Exercise 1: Simple Cascade System ---
    # This matrix represents the first 5x5 system provided earlier.
    A1 = np.array([
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [1, 0, 0, 0, 1]
    ])
    B1 = np.array([3, 5, 7, 9, 6])

    # --- Exercise 2: The "Messy" Dense System ---
    # This matrix represents the more complex system with fractions.
    A2 = np.array([
        [2, 3, -1, 4, -2],
        [1, 2, 2, -1,  1],
        [-1, 1, 3, 2,  4],
        [4, -1, 1, 5, -1],
        [2, 4, -2, 1,  3]
    ])
    B2 = np.array([8, 10, 20, 12, 18])

    # Solving using numpy.linalg.solve
    # This function is equivalent to performing Gauss-Jordan until reaching the identity matrix.
    sol1 = np.linalg.solve(A1, B1)
    sol2 = np.linalg.solve(A2, B2)

    print("=== Results for Exercise 1 ===")
    for i, val in enumerate(sol1, 1):
        print(f"x{i} = {val:7.2f}")

    print("\n=== Results for Exercise 2 ===")
    for i, val in enumerate(sol2, 1):
        print(f"x{i} = {val:7.2f}")

if __name__ == "__main__":
    solve_systems()