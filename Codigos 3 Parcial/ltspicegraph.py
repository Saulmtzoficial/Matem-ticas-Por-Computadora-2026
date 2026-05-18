import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',       # Letra clásica de paper
    'font.size': 12,              # Tamaño base legible
    'axes.labelsize': 14,         # Títulos de los ejes más grandes
    'axes.titlesize': 14,         # Títulos de las gráficas
    'legend.fontsize': 11,        # Letra de la leyenda
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300             
})

# ====================================================================
# PARTE 1: Datos de la simulación
# ====================================================================
time = [0.0, 0.012207031, 0.024414063, 0.036621094, 0.048828125, 0.061035156,
        0.073242188, 0.085449219, 0.103585379, 0.12172154, 0.139857701, 
        0.157993862, 0.176130022, 0.194266183, 0.212402344, 0.244489397, 
        0.276576451, 0.308663504, 0.340750558, 0.372837612, 0.404924665, 
        0.437011719, 0.586983817, 0.736955915, 0.886928013, 1.036900112, 
        1.18687221, 1.336844308, 1.486816406, 4.9]

I_L1 = [4.744856, 4.972031, 5.17742, 5.361025, 5.522845, 5.662881, 
        5.781131, 5.877597, 6.002528, 6.112896, 6.208701, 6.289943, 
        6.356622, 6.408738, 6.44629, 6.49926, 6.543935, 6.580314, 
        6.608399, 6.628188, 6.639682, 6.642882, 6.652685, 6.660327, 
        6.665807, 6.669125, 6.670282, 6.669277, 6.666111, 0.0]

I_L2 = [1.576745, 1.760601, 1.929852, 2.084499, 2.224541, 2.349979, 
        2.460811, 2.557039, 2.677867, 2.784909, 2.878166, 2.957639, 
        3.023327, 3.07523, 3.113348, 3.166208, 3.210796, 3.247112, 
        3.275157, 3.29493, 3.306431, 3.30966, 3.319463, 3.327104, 
        3.332584, 3.335903, 3.337059, 3.336055, 3.332888, 0.0]

# Tomamos el valor absoluto para que la línea se vea como en la imagen
I_R1_originales = [-3.168112, -3.21143, -3.247568, -3.276526, -3.298304, 
                   -3.312902, -3.32032, -3.320558, -3.324662, -3.327987, 
                   -3.330535, -3.332304, -3.333295, -3.333508, -3.332943, 
                   -3.333053, -3.333139, -3.333202, -3.333242, -3.333258, 
                   -3.333252, -3.333222, -3.333222, -3.333222, -3.333222, 
                   -3.333222, -3.333222, -3.333222, -3.333222, 0.0]
I_R1 = [abs(x) for x in I_R1_originales]

# ====================================================================
# PARTE 2: Graficar la fase transitoria
# ====================================================================
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.plot(time, I_L1, color='#1f77b4', linewidth=2, marker='o', markersize=5,
         markeredgecolor='black', alpha=0.8, label='$i_1(t)$ (Corriente en $L_1$)')
ax1.plot(time, I_L2, color='#d62728', linewidth=2, marker='s', markersize=5,
         markeredgecolor='black', alpha=0.8, label='$i_3(t)$ (Corriente en $L_2$)')
ax1.plot(time, I_R1, color='#2ca02c', linewidth=2, linestyle='--', marker='^', markersize=5,
         markeredgecolor='black', alpha=0.8, label='$i_R(t)$ (Corriente en $R_1, i_2$)')

ax1.set_title('Detalle de la Fase Transitoria ($t \in [0, 0.5]$ s)', pad=10)
ax1.set_xlabel('Tiempo $t$ (s)')
ax1.set_ylabel('Corriente $i$ (A)')
ax1.set_xlim(0, 0.5)
ax1.set_ylim(0, 7)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', framealpha=1.0, edgecolor='black')

plt.tight_layout()

# Guardar imagen en alta calidad
plt.savefig('transitorio_simulacion_paper.png', dpi=300, bbox_inches='tight')

plt.show()