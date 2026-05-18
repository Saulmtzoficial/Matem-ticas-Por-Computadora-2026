import numpy as np
import matplotlib.pyplot asplt
import math
import cmath

def newtonRaphson (f,df,x, i_max,tol=1e-8):
    k=0
    cumple = False
    print("{:^10} | {:^20} | {:^20} | {:^20}".format("Iteración", "x_n", "f(x_n)", "df(x_n)"))

    while(not cumple and k<i_max):
        if df(x) !=0:
            x= x - f(x)/df(x)
        else:
            x=x+tol
        print("{:^10} | {:^20.10f} | {:^20.10f} | {:^20.10f}".format(k, x, f(x), df(x)))
        cumple = abs(f(x)) <= tol
        k += 1
    if k<i_max
        return x
    else:
        raise ValueError("El método no converge en el número máximo de iteraciones")

#Funcion a evaluar
def f(x):
    e=0.0015
    D=4
    Re=13743
    return 1/np.sqr
#DErivada
def df(x):
    e=0.0015
    D=4
    Re=13743
    return -1/(2*x**(3/2))-

def main():
    #Valores iniciales
    x0=0.01
    #llamada al algoritmo
    raiz=newtonRaphson(f,df,x0)
print())