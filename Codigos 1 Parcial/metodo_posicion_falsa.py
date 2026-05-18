import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

# ==========================================
#        SIMULATION LOGIC (MATPLOTLIB)
# ==========================================

def run_simulation(f_str, start_a, start_b, tolerance):
    """
    Runs the False Position Method simulation based on user input.
    Requiere dos puntos iniciales a y b donde f(a) y f(b) tienen signos opuestos.
    """
    
    # --- 1. Define Math Logic from Input ---
    try:
        # Define the function using eval
        safe_dict = {
            "np": np, "x": 0, 
            "sin": np.sin, "cos": np.cos, "tan": np.tan, 
            "exp": np.exp, "sqrt": np.sqrt, "log": np.log, "pi": np.pi,
            "abs": np.abs, "arcsin": np.arcsin, "arccos": np.arccos,
            "arctan": np.arctan, "sinh": np.sinh, "cosh": np.cosh,
            "tanh": np.tanh
        }
        
        func = lambda x: eval(f_str, {**safe_dict, "x": x})
        
        # Test if the function works at both starting points
        f_a = func(start_a)
        f_b = func(start_b)
        
        # Verificar que f(a) y f(b) tengan signos opuestos
        if f_a * f_b >= 0:
            messagebox.showerror("Input Error", 
                "f(a) and f(b) must have opposite signs.\n"
                f"f({start_a}) = {f_a:.4f}\n"
                f"f({start_b}) = {f_b:.4f}")
            return
            
    except Exception as e:
        messagebox.showerror("Math Error", f"Invalid function expression: {e}")
        return

    # --- 2. False Position Method Calculation ---
    intervals = [(start_a, start_b)]  # Almacena los intervalos [a, b]
    steps = []  # Almacena los puntos c calculados
    iterations = 20  # Maximum steps for visualization
    
    a, b = start_a, start_b
    
    for _ in range(iterations):
        try:
            f_a = func(a)
            f_b = func(b)
            
            # Fórmula del método de la posición falsa
            c = b - f_b * (b - a) / (f_b - f_a)
            f_c = func(c)
            
            steps.append(c)
            
            # Criterio de convergencia
            if abs(f_c) < tolerance:
                intervals.append((a, b))
                break
            
            # Actualizar el intervalo
            if f_a * f_c < 0:
                b = c
            else:
                a = c
            
            intervals.append((a, b))
            
            # Verificar convergencia en el tamaño del intervalo
            if abs(b - a) < tolerance:
                break
                
        except Exception:
            break

    if len(steps) == 0:
        messagebox.showerror("Error", "No convergence achieved.")
        return

    # --- 3. Visual Setup (Dark Mode) ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 7))
    plt.subplots_adjust(bottom=0.15) 

    # Dynamic Domain based on initial interval
    margin = abs(start_b - start_a) * 0.3
    domain = (min(start_a, start_b) - margin, max(start_a, start_b) + margin)
    
    x_vals = np.linspace(domain[0], domain[1], 500)
    try:
        y_vals = np.array([func(v) for v in x_vals])
    except:
        y_vals = x_vals * 0  # Fallback

    ax.plot(x_vals, y_vals, color='#00FFFF', linewidth=2.5, label=f'f(x) = {f_str}')
    ax.axhline(0, color='white', linewidth=0.8, linestyle='--')
    ax.set_xlim(domain[0], domain[1])
    
    # Set Y limits with a buffer
    y_min, y_max = min(y_vals), max(y_vals)
    ax.set_ylim(y_min - abs(y_min)*0.1 - 1, y_max + abs(y_max)*0.1 + 1)

    # Animated Elements
    point_a, = ax.plot([], [], 'o', color='#FF6B6B', markersize=8, zorder=5, label='Point a')
    point_b, = ax.plot([], [], 'o', color='#4ECDC4', markersize=8, zorder=5, label='Point b')
    point_c, = ax.plot([], [], 'o', color='#FFE66D', markersize=8, zorder=5, label='Point c')
    point_on_axis, = ax.plot([], [], 'x', color='#FFFF00', markersize=10, markeredgewidth=2, zorder=5)
    secant_line, = ax.plot([], [], color='#FF0044', linewidth=2, label='Secant', linestyle='--')
    interval_line, = ax.plot([], [], color='#95E1D3', linewidth=3, alpha=0.5, zorder=3)
    vertical_line, = ax.plot([], [], color='white', linestyle=':', linewidth=1, alpha=0.6)
    text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', 
                       fontsize=11, verticalalignment='top', family='monospace')

    ax.set_title(f"False Position Method: {f_str}", color='white', fontsize=14, pad=20)
    ax.legend(loc='upper right', frameon=False, fontsize=9)

    # --- 4. Menu Button ---
    ax_btn = plt.axes([0.85, 0.02, 0.1, 0.05])
    btn = Button(ax_btn, 'Menu', color='#333333', hovercolor='#555555')
    btn.label.set_color('white')
    btn.on_clicked(lambda e: plt.close(fig))

    # --- 5. Animation Logic ---
    def update(frame):
        step_index = frame // 3 
        phase = frame % 3 
        
        if step_index >= len(steps):
            return point_a, point_b, point_c, secant_line, vertical_line, point_on_axis, interval_line, text_info

        a_curr, b_curr = intervals[step_index]
        c_curr = steps[step_index]
        
        y_a = func(a_curr)
        y_b = func(b_curr)
        y_c = func(c_curr)
        
        # Mostrar información
        text_info.set_text(
            f'Iteration: {step_index}\n'
            f'Interval: [{a_curr:.6f}, {b_curr:.6f}]\n'
            f'c ≈ {c_curr:.6f}\n'
            f'f(c) ≈ {y_c:.6f}'
        )
        
        # Mostrar intervalo actual
        interval_line.set_data([a_curr, b_curr], [0, 0])
        
        if phase == 0:
            # Mostrar solo los puntos a y b
            point_a.set_data([a_curr], [y_a])
            point_b.set_data([b_curr], [y_b])
            point_c.set_data([], [])
            secant_line.set_data([], [])
            vertical_line.set_data([], [])
            point_on_axis.set_data([], [])
            
        elif phase == 1:
            # Mostrar la línea secante
            point_a.set_data([a_curr], [y_a])
            point_b.set_data([b_curr], [y_b])
            point_c.set_data([], [])
            
            # Dibujar la secante extendida
            t = np.linspace(min(a_curr, b_curr) - 0.5, max(a_curr, b_curr) + 0.5, 100)
            if abs(b_curr - a_curr) > 1e-10:
                slope = (y_b - y_a) / (b_curr - a_curr)
                secant_y = y_a + slope * (t - a_curr)
                secant_line.set_data(t, secant_y)
            
            point_on_axis.set_data([c_curr], [0])
            vertical_line.set_data([], [])
            
        else:  # phase == 2
            # Mostrar el punto c y la línea vertical
            point_a.set_data([a_curr], [y_a])
            point_b.set_data([b_curr], [y_b])
            point_c.set_data([c_curr], [y_c])
            vertical_line.set_data([c_curr, c_curr], [0, y_c])
            point_on_axis.set_data([c_curr], [0])

        return point_a, point_b, point_c, secant_line, vertical_line, point_on_axis, interval_line, text_info

    ani = FuncAnimation(fig, update, frames=(len(steps))*3, interval=800, blit=True)
    plt.show()

# ==========================================
#           GUI MENU (TKINTER)
# ==========================================

class FalsePositionMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("False Position Method Simulator")
        self.root.geometry("500x450")
        self.root.configure(bg="#1E1E1E")
        self.root.resizable(False, False)

        title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        
        tk.Label(root, text="False Position Method", font=title_font, 
                 bg="#1E1E1E", fg="#00FFFF", pady=20).pack()

        # Input Fields
        self.f_entry = self.create_input("Function f(x):", "x**3 - 2*x**2 - 5")
        self.a_entry = self.create_input("Interval start (a):", "2.0")
        self.b_entry = self.create_input("Interval end (b):", "4.0")
        self.tol_entry = self.create_input("Tolerance:", "0.0001")

        # Info Label
        info_label = tk.Label(
            root, 
            text="Note: f(a) and f(b) must have opposite signs", 
            bg="#1E1E1E", 
            fg="#FFE66D", 
            font=("Helvetica", 9, "italic"),
            pady=5
        )
        info_label.pack()

        # Start Button
        btn_run = tk.Button(
            root, text="START ANIMATION", command=self.validate_and_run,
            font=("Helvetica", 12, "bold"), bg="#0088CC", fg="white", 
            relief="flat", width=25, pady=10, cursor="hand2"
        )
        btn_run.pack(pady=15)

        # Exit Button
        tk.Button(root, text="EXIT", command=root.quit, bg="#444444", 
                  fg="white", relief="flat", width=10, cursor="hand2").pack()

    def create_input(self, label_text, default_val):
        frame = tk.Frame(self.root, bg="#1E1E1E")
        frame.pack(fill="x", padx=50, pady=5)
        
        tk.Label(frame, text=label_text, bg="#1E1E1E", fg="white", anchor="w").pack(fill="x")
        entry = tk.Entry(frame, bg="#333333", fg="white", insertbackground="white", 
                         relief="flat", font=("Consolas", 11))
        entry.insert(0, default_val)
        entry.pack(fill="x", ipady=5)
        return entry

    def validate_and_run(self):
        try:
            f_s = self.f_entry.get()
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            tol = float(self.tol_entry.get())
            
            if abs(b - a) < 1e-10:
                messagebox.showerror("Input Error", "a and b must be different values.")
                return
            
            if a > b:
                messagebox.showwarning("Input Warning", "Swapping a and b so that a < b")
                a, b = b, a
            
            self.root.withdraw()
            run_simulation(f_s, a, b, tol)
            self.root.deiconify()
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure a, b, and Tolerance are valid numbers.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FalsePositionMenu(root)
    root.mainloop()