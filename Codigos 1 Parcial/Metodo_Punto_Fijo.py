import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

# ==========================================
#       LÓGICA DE SIMULACIÓN (MATPLOTLIB)
# ==========================================

def run_simulation(g_str, tolerancia, inicio_x):
    """
    Ejecuta la simulación de Iteración de Punto Fijo basada en la entrada del usuario.
    """
    
    # --- 1. Definir Lógica Matemática ---
    try:
        # Evaluamos la cadena del usuario usando el espacio de nombres de numpy
        gfunc = lambda x: eval(g_str, {"np": np, "x": x, "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "sqrt": np.sqrt, "log": np.log})
        # Prueba rápida para detectar errores de sintaxis
        gfunc(inicio_x)
    except Exception as e:
        messagebox.showerror("Error Matemático", f"Expresión de función inválida: {e}")
        return

    # --- 2. Cálculo de Iteraciones de Punto Fijo ---
    pasos = [inicio_x]
    max_iteraciones = 50
    
    for _ in range(max_iteraciones):
        actual_x = pasos[-1]
        try:
            siguiente_x = gfunc(actual_x)
            pasos.append(siguiente_x)
            if abs(siguiente_x - actual_x) < tolerancia:
                break
        except Exception:
            break

    # --- 3. Configuración Visual (Modo Oscuro) ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 7))
    plt.subplots_adjust(bottom=0.15) 

    # Dominio dinámico basado en los pasos calculados
    margen = max(abs(pasos[0] - pasos[-1]), 1.0)
    dominio = (min(pasos) - margen, max(pasos) + margen)
    
    x_vals = np.linspace(dominio[0], dominio[1], 500)
    try:
        g_vals = np.array([gfunc(val) for val in x_vals])
    except:
        g_vals = x_vals 

    ax.plot(x_vals, g_vals, color='#00FFFF', linewidth=2.5, label=f'y = g(x)')
    ax.plot(x_vals, x_vals, color='#00FF00', linewidth=2, linestyle='--', label='y = x')
    ax.axhline(0, color='white', linewidth=0.8, linestyle=':', alpha=0.3)
    ax.axvline(0, color='white', linewidth=0.8, linestyle=':', alpha=0.3)
    
    # Elementos Animados
    punto_curva, = ax.plot([], [], 'o', color='white', markersize=6, zorder=5)
    punto_linea, = ax.plot([], [], 'o', color='#FFFF00', markersize=6, zorder=5)
    linea_horizontal, = ax.plot([], [], color='#FF0044', linewidth=1.5, alpha=0.8)
    linea_vertical, = ax.plot([], [], color='#FF0044', linewidth=1.5, alpha=0.8)
    texto_info = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12, verticalalignment='top')

    ax.set_title(f"Iteración de Punto Fijo: g(x) = {g_str}", color='white', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.2)

    # Botón de Menú en Matplotlib
    ax_btn = plt.axes([0.8, 0.02, 0.1, 0.05])
    btn = Button(ax_btn, 'Menú', color='#333333', hovercolor='#555555')
    btn.label.set_color('white')
    btn.on_clicked(lambda e: plt.close(fig))

    def update(frame):
        idx = frame // 2 
        fase = frame % 2 
        if idx >= len(pasos) - 1:
            return punto_curva, punto_linea, linea_horizontal, linea_vertical, texto_info

        x_actual = pasos[idx]
        x_sig = pasos[idx + 1]
        y_actual = gfunc(x_actual)
        
        texto_info.set_text(f'Iteración: {idx}\nx_actual: {x_actual:.6f}\nx_siguiente: {x_sig:.6f}')
        
        if fase == 0:
            punto_linea.set_data([x_actual], [x_actual])
            punto_curva.set_data([x_actual], [y_actual])
            linea_horizontal.set_data([x_actual, x_actual], [x_actual, y_actual])
            linea_vertical.set_data([], [])
        else:
            punto_curva.set_data([x_actual], [y_actual])
            punto_linea.set_data([x_sig], [x_sig])
            linea_vertical.set_data([x_actual, x_sig], [y_actual, y_actual])

        return punto_curva, punto_linea, linea_horizontal, linea_vertical, texto_info

    ani = FuncAnimation(fig, update, frames=(len(pasos)-1)*2, interval=800, blit=True)
    plt.show()

# ==========================================
#          MENÚ GUI (TKINTER)
# ==========================================

class MenuPuntoFijo:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora de Punto Fijo")
        self.root.geometry("450x500")
        self.root.configure(bg="#1E1E1E")
        self.root.resizable(False, False)

        fuente_titulo = tkfont.Font(family="Helvetica", size=16, weight="bold")
        fuente_label = tkfont.Font(family="Helvetica", size=11)

        tk.Label(root, text="Parámetros de Punto Fijo", font=fuente_titulo, bg="#1E1E1E", fg="#00FFFF", pady=25).pack()

        # Campos de Entrada
        self.g_entry = self.crear_input("Función de Iteración g(x):", "np.cos(x)")
        self.tol_entry = self.crear_input("Tolerancia (ej. 0.0001):", "0.0001")
        self.inicio_entry = self.crear_input("Punto Inicial (x0):", "1.0")

        # Botón de Ejecución
        btn_run = tk.Button(
            root, text="INICIAR SIMULACIÓN", command=self.validar_y_ejecutar,
            font=fuente_label, bg="#0088CC", fg="white", relief="flat", width=25, pady=10, cursor="hand2"
        )
        btn_run.pack(pady=30)

        # Botón Salir
        tk.Button(root, text="SALIR", command=root.quit, bg="#444444", fg="white", relief="flat", width=10, cursor="hand2").pack()

    def crear_input(self, texto_label, valor_defecto):
        frame = tk.Frame(self.root, bg="#1E1E1E")
        frame.pack(fill="x", padx=40, pady=10)
        
        tk.Label(frame, text=texto_label, bg="#1E1E1E", fg="white", anchor="w", font=("Helvetica", 10)).pack(fill="x")
        entry = tk.Entry(frame, bg="#333333", fg="white", insertbackground="white", relief="flat", font=("Consolas", 12))
        entry.insert(0, valor_defecto)
        entry.pack(fill="x", ipady=5)
        return entry

    def validar_y_ejecutar(self):
        try:
            g_str = self.g_entry.get()
            tol = float(self.tol_entry.get())
            x0 = float(self.inicio_entry.get())
            
            self.root.withdraw()
            run_simulation(g_str, tol, x0)
            self.root.deiconify()
        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor, asegúrate de que la Tolerancia y el Punto Inicial sean números válidos.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MenuPuntoFijo(root)
    root.mainloop()