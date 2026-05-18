import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation

# ── Paleta (UASLP Mechatronics Style) ──────────────────────────────────────
BG_FIG    = "#1A1A2E"
BG_AX     = "#16213E"
BG_TABLE  = "#0F3460"
GRID_CLR  = "#2A2A4A"
SPINE_CLR = "#3A3A5C"
TXT_PRI   = "#E2E2F0"
TXT_SEC   = "#8888AA"
ACCENT    = "#4FC3F7"
CLR_ERR   = "#FF5252"

# ── Datos de APP12 ──────────────────────────────────────────────────────────
xs = np.array([0.12, 0.24, 0.36, 0.48, 0.60, 0.72])
fs_orig = np.array([0.79168, 0.77334, 0.74371, 0.70413, 0.65632, 0.60228])

# Error ε en f(0.36)
epsilon = 0.01
fs_err = fs_orig.copy()
fs_err[2] += epsilon 

def calc_diffs(data):
    n = len(data)
    table = np.zeros((n, n))
    table[:, 0] = data
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i+1, j-1] - table[i, j-1]
    return table

table_orig = calc_diffs(fs_orig)
table_err  = calc_diffs(fs_err)
table_delta = table_err - table_orig

# ── Interfaz ───────────────────────────────────────────────────────────────
plt.rcParams.update({"text.color": TXT_PRI, "axes.labelcolor": TXT_SEC, "font.size": 8})
fig = plt.figure(figsize=(15, 8), facecolor=BG_FIG)
fig.text(0.5, 0.96, "APP12: Propagación de Errores y Coeficientes Binomiales", 
         ha="center", fontsize=14, color=ACCENT, fontweight='bold')

gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], left=0.05, right=0.95, bottom=0.15, top=0.90, wspace=0.1)
ax_o = fig.add_subplot(gs[0])
ax_d = fig.add_subplot(gs[1])

def draw_table_grid(ax, title, table_data, highlight_err=False):
    ax.cla(); ax.set_axis_off()
    ax.text(0.5, 0.98, title, ha="center", color=ACCENT, weight='bold', transform=ax.transAxes)
    
    # AJUSTE: Ahora 'headers' y 'cols' tienen la misma longitud (7)
    headers = ["x", "f(x)", "Δf", "Δ²f", "Δ³f", "Δ⁴f", "Δ⁵f"]
    cols = [0.03, 0.14, 0.28, 0.42, 0.56, 0.70, 0.84]
    
    for h, cx in zip(headers, cols):
        ax.text(cx, 0.92, h, color=TXT_SEC, fontweight='bold', transform=ax.transAxes)

    for i in range(len(xs)):
        y_pos = 0.82 - i*0.12
        ax.text(cols[0], y_pos, f"{xs[i]:.2f}", color=TXT_SEC, transform=ax.transAxes)
        # Dibujamos las diferencias j (de 0 a 5)
        for j in range(len(xs) - i):
            val = table_data[i, j]
            is_affected = highlight_err and abs(val) > 1e-9
            clr = CLR_ERR if is_affected else TXT_PRI
            
            if is_affected:
                rect = plt.Rectangle((cols[j+1]-0.02, y_pos-0.035), 0.11, 0.08, 
                                     facecolor=CLR_ERR, alpha=0.15, transform=ax.transAxes)
                ax.add_patch(rect)
                
            ax.text(cols[j+1], y_pos - (j*0.06), f"{val:+.4f}" if highlight_err else f"{val:.5f}", 
                    color=clr, transform=ax.transAxes)

# ── Control ───────────────────────────────────────────────────────────────
def update(event):
    draw_table_grid(ax_o, "Tabla Original (Sin Error)", table_orig, False)
    draw_table_grid(ax_d, f"Propagación del Error (ε = {epsilon})", table_delta, True)
    plt.draw()

ax_btn = fig.add_axes([0.44, 0.04, 0.12, 0.06])
btn = Button(ax_btn, "▶ Mostrar", color="#0D47A1", hovercolor="#1565C0")
btn.label.set_color("white"); btn.on_clicked(update)

update(None)
plt.show()