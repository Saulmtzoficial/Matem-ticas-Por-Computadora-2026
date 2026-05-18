#cuantos grados son en 1.22radianes?Para convertir radianes a grados, puedes usar la siguiente fórmula:
#grados = radianes * (180 / π)
import math
def radianes_a_grados(radianes):
    grados = radianes * (180 / math.pi)
    return grados
radianes = 1.22
grados = radianes_a_grados(radianes)
print(f"{radianes} radianes son aproximadamente {grados:.2f} grados.")
