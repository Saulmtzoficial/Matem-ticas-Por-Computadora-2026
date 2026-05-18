#>Resolver sistema ded ecuaciones
import numpy as np

np.set_printoptions(precision=3, suppress=1, floatmode='fixed')

A=np.array([[1,1,1,6],[2,-3,1,-1],[1,2,-3,-4]])

print("Nuestro sistema de ecuaciones es: ")
print(A)

#Hacemos operaciones con renglones
#-2R1+R2 ->
print("-2R1+R2")
A[1,:] = -2*A[0,:]+A[1,:]
print("A= ")
#R3-R1
print("R3-R1")
print(A)
A[2,:] = -1*A[0,:]+A[2,:]
print("A= ")
print(A)
print("R2*-0.2")
A[1,:] = -0.2*A[1,:]
print("A= ")
print(A)
print("R0-R1")
A[0,:]= A[0,:]-A[1,:]
print("A= ")
print(A)
print("R2-R1")
A[2,:]= A[2,:]-A[1,:]
print("A= ")
print(A)
print("R2*-0.25")
A[2,:] = -0.25*A[2,:]
print("A= ")
print(A)
print("R2-R1")
A[0,:]= A[0,:]-A[2,:]
print("A= ")
print(A)

