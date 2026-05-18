import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BarycentricInterpolator

# ── Paleta (UASLP Mechatronics Style) ──────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"
CLR_TRUE  = "#66BB6A" # True Function (Green)
CLR_HERM  = "#FF7043" # Hermite (Orange)
CLR_CUB   = "#CE93D8" # Standard Cubic (Purple)

# ── Datos de APP8 ──────────────────────────────────────────────────────────
x1, f1, df1 = 1.0, 2.71828, 0.0
x2, f3, df3 = 3.0, 6.69518, 4.46345
eval_pts = np.array([1.5, 2.0, 2.5])

# True Function
def true_f(x): return np.exp(x) / x

# 1. Hermite Divided Differences
# z = [1, 1, 3, 3]
z = np.array([x1, x1, x2, x2])
fz = np.array([f1, f1, f3, f3])
# Table construction
f_1 = np.array([df1, (f3-f1)/(x2-x1), df3])
f_2 = np.array([(f_1[1]-f_1[0])/(z[2]-z[0]), (f_1[2]-f_1[1])/(z[3]-z[1])])
f_3 = (f_2[1]-f_2[0])/(z[3]-z[0])

def hermite_eval(x):
    return (f1 + f_1[0]*(x-z[0]) + f_2[0]*(x-z[0])*(x-z[1]) + 
            f_3*(x-z[0])*(x-z[1])*(x-z[2]))

# 2. Standard Cubic Interpolant (Nodes: 1, 1.5, 2.4, 3)
x_std = np.array([1.0, 1.5, 2.4, 3.0])
y_std = true_f(x_std)
std_cubic = BarycentricInterpolator(x_std, y_std)

# ── Interfaz de Visualización ──────────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 9})
fig = plt.figure(figsize=(15, 8), facecolor=BG_FIG)
fig.text(0.5, 0.95, "APP8: Polinomio de Hermite vs. Interpolación Estándar", 
         ha="center", fontsize=14, color=ACCENT, fontweight='bold')

gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1], left=0.07, right=0.95, wspace=0.15)
ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1])

ax.set_facecolor(BG_AX); ax.grid(True, color=GRID_CLR, lw=0.5)
for sp in ax.spines.values(): sp.set_color(SPINE_CLR)

x_fine = np.linspace(1, 3, 200)
ax.plot(x_fine, true_f(x_fine), color=CLR_TRUE, lw=3, label="Real: $e^x/x$", alpha=0.6)
ax.plot(x_fine, hermite_eval(x_fine), color=CLR_HERM, lw=2, ls="--", label="Hermite $H_3(x)$")
ax.plot(x_fine, std_cubic(x_fine), color=CLR_CUB, lw=1.5, ls=":", label="Cúbica Estándar")

ax.scatter([1, 3], [f1, f3], color=CLR_HERM, s=80, edgecolors="white", label="Nodos Hermite")
ax.scatter(x_std, y_std, color=CLR_CUB, s=40, marker='s', label="Nodos Estándar")

ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.legend(facecolor=BG_AX, edgecolor=SPINE_CLR)

# --- Tabla de Errores ---
axt.set_axis_off()
y_pos = 0.9
axt.text(0.5, 0.95, "Comparativa de Error", ha="center", color=ACCENT, weight='bold', fontsize=12)

headers = ["x", "Error Hermite", "Error Cúbico"]
cols = [0.1, 0.4, 0.75]
for h, cx in zip(headers, cols): axt.text(cx, y_pos, h, color=TXT_SEC, weight='bold')

for i, x in enumerate(eval_pts):
    y_pos -= 0.1
    val_true = true_f(x)
    err_h = abs(hermite_eval(x) - val_true)
    err_c = abs(std_cubic(x) - val_true)
    
    rect = plt.Rectangle((0.05, y_pos-0.03), 0.9, 0.08, facecolor=BG_TABLE if i%2==0 else BG_FIG, alpha=0.5, transform=axt.transAxes)
    axt.add_patch(rect)
    
    axt.text(cols[0], y_pos, f"{x:.1f}", color=TXT_PRI)
    axt.text(cols[1], y_pos, f"{err_h:.6f}", color=CLR_HERM)
    axt.text(cols[2], y_pos, f"{err_c:.6f}", color=CLR_CUB)

plt.show()