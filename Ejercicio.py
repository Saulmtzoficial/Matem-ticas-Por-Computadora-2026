import math

def biseccion():
    # 1. Inputs - Converting strings to float/int
    f_str = input("Ingresa la funcion f(x): ")
    # We create a lambda function to evaluate the input string
    f = lambda x: eval(f_str)
    
    a = float(input("Ingresa el inicio a: "))
    b = float(input("Ingresa el final b: "))
    tol = float(input("Ingresa la tolerancia: "))
    i_max = int(input("Ingrsa la cantidad maxima de iteraciones: "))

    i = 1
    # Evaluate f(a) using the lambda
    fa = f(a)
    
    # Check if a root exists in the interval (Bolzano's Theorem)
    if fa * f(b) > 0:
        print("Esta funcion no cambia de signo en el intervalo [a, b].")
        return None

    while i <= i_max:
        p = a + (b - a) / 2
        fp = f(p)
        
        # Convergence check
        if fp == 0 or (b - a) / 2 < tol:
            return p
        
        i += 1 # Corrected increment
        
        # Update interval
        if fa * fp > 0:
            a = p
            fa = fp
        else:
            b = p
            
    print("Hachiko, deten la simulacion, no va a converger.")
    return None

# Run the function
resultado = biseccion()
if resultado is not None:
    print(f"La solucion aproximada es: {resultado}")               