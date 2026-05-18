#Multiplicando un vector A por la matriz identidad 

I3=np.array([[1,0,0],[0,1,0],[0,0,1]])

I2=np.array([[1,0],[0,1]])

c=.dot(I3,A)
print("El resultado de multiplicar el vector A por la matriz identidad I3 es: ",c)

d=np.dot(A,I3)
print("El resultado de multiplicar la matriz identidad I3 por el vector A es: ",d)


#Calculando la inversa de una matriz 3x3

MAT=np.array([[1,2,3],[-1,2,-3],[0,2,5]])
print("La matriz es:")
print(MAT)

MAT_INV=np.linalg.inv(MAT)
print("La inversa de la matriz es:")
print(MAT_INV)

#Verificar que la matriz sea correcta
MAT2=np.dot(MAT,MAT_INV)
print("El resultado de multiplicar la matriz por su inversa es:")
print(MAT2)

#Verificar que el producto de la matriz por su inversa es la matriz identidad
Identi= np.dot(MAT,MAT_INV)
print("El resultado de multiplicar la matriz por su inversa es:")
print(Identi)