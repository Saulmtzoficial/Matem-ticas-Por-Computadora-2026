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

def run_simulation(f_str, start_x0, start_x1, tolerance):
    """
    Runs the Secant Method simulation based on user input.
    Requiere dos puntos iniciales x0 y x1.
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
        func(start_x0)
        func(start_x1)
    except Exception as e:
        messagebox.showerror("Math Error", f"Invalid function expression: {e}")
        return

    # --- 2. Secant Method Calculation ---
    steps = [start_x0, start_x1]
    iterations = 15  # Maximum steps for visualization
    
    for _ in range(iterations):
        x_prev = steps[-2]
        x_curr = steps[-1]
        
        try:
            f_prev = func(x_prev)
            f_curr = func(x_curr)
            
            # Evitar división por cero
            if abs(f_curr - f_prev) < 1e-10:
                break
            
            # Fórmula del método de la secante
            x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            steps.append(x_next)
            
            # Criterio de convergencia
            if abs(x_next - x_curr) < tolerance:
                break
        except Exception:
            break

    # --- 3. Visual Setup (Dark Mode) ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(11, 7))
    plt.subplots_adjust(bottom=0.15) 

    # Dynamic Domain based on steps
    margin = max(abs(steps[0] - steps[-1]) * 0.5, 2.0)
    domain = (min(steps) - margin, max(steps) + margin)
    
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
    point_prev, = ax.plot([], [], 'o', color='#FF6B6B', markersize=6, zorder=5, label='x_{n-1}')
    point_curr, = ax.plot([], [], 'o', color='white', markersize=6, zorder=5, label='x_n')
    point_on_axis, = ax.plot([], [], 'x', color='#FFFF00', markersize=8, markeredgewidth=2, zorder=5)
    secant_line, = ax.plot([], [], color='#FF0044', linewidth=2, label='Secant')
    vertical_line, = ax.plot([], [], color='white', linestyle=':', linewidth=1, alpha=0.6)
    text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12, verticalalignment='top')

    ax.set_title(f"Secant Method: {f_str}", color='white', fontsize=14, pad=20)
    ax.legend(loc='upper right', frameon=False)

    # --- 4. Menu Button ---
    ax_btn = plt.axes([0.85, 0.02, 0.1, 0.05])
    btn = Button(ax_btn, 'Menu', color='#333333', hovercolor='#555555')
    btn.label.set_color('white')
    btn.on_clicked(lambda e: plt.close(fig))

    # --- 5. Animation Logic ---
    def update(frame):
        step_index = frame // 2 
        phase = frame % 2 
        
        if step_index >= len(steps) - 1:
            return point_prev, point_curr, secant_line, vertical_line, point_on_axis, text_info

        x_prev = steps[step_index]
        x_curr = steps[step_index + 1]
        y_prev = func(x_prev)
        y_curr = func(x_curr)
        
        text_info.set_text(f'Iteration: {step_index}\nx_{step_index} ≈ {x_prev:.6f}\nx_{step_index+1} ≈ {x_curr:.6f}')
        point_prev.set_data([x_prev], [y_prev])
        point_curr.set_data([x_curr], [y_curr])
        
        if phase == 0:
            # Mostrar puntos y líneas verticales
            vertical_line.set_data([x_curr, x_curr], [0, y_curr])
            secant_line.set_data([], []) 
            point_on_axis.set_data([], [])
        else:
            # Mostrar la línea secante
            if step_index + 2 < len(steps):
                x_next = steps[step_index + 2]
                
                # Extender la línea secante para mejor visualización
                t = np.linspace(min(x_prev, x_next) - 0.5, max(x_prev, x_next) + 0.5, 100)
                # Ecuación de la recta que pasa por (x_prev, y_prev) y (x_curr, y_curr)
                if abs(x_curr - x_prev) > 1e-10:
                    slope = (y_curr - y_prev) / (x_curr - x_prev)
                    secant_y = y_prev + slope * (t - x_prev)
                    secant_line.set_data(t, secant_y)
                
                point_on_axis.set_data([x_next], [0])
            else:
                secant_line.set_data([], [])
                point_on_axis.set_data([], [])

        return point_prev, point_curr, secant_line, vertical_line, point_on_axis, text_info

    ani = FuncAnimation(fig, update, frames=(len(steps)-1)*2, interval=1000, blit=True)
    plt.show()

# ==========================================
#           GUI MENU (TKINTER)
# ==========================================

class SecantMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Secant Method Simulator")
        self.root.geometry("500x450")
        self.root.configure(bg="#1E1E1E")
        self.root.resizable(False, False)

        title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        
        tk.Label(root, text="Secant Method", font=title_font, 
                 bg="#1E1E1E", fg="#00FFFF", pady=20).pack()

        # Input Fields
        self.f_entry = self.create_input("Function f(x):", "x**3 - 2*x**2 - 5")
        self.x0_entry = self.create_input("Initial Point x0:", "2.0")
        self.x1_entry = self.create_input("Initial Point x1:", "3.0")
        self.tol_entry = self.create_input("Tolerance:", "0.0001")

        # Start Button
        btn_run = tk.Button(
            root, text="START ANIMATION", command=self.validate_and_run,
            font=("Helvetica", 12, "bold"), bg="#0088CC", fg="white", 
            relief="flat", width=25, pady=10, cursor="hand2"
        )
        btn_run.pack(pady=25)

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
            x0 = float(self.x0_entry.get())
            x1 = float(self.x1_entry.get())
            tol = float(self.tol_entry.get())
            
            if abs(x1 - x0) < 1e-10:
                messagebox.showerror("Input Error", "x0 and x1 must be different values.")
                return
            
            self.root.withdraw()
            run_simulation(f_s, x0, x1, tol)
            self.root.deiconify()
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure x0, x1, and Tolerance are valid numbers.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SecantMenu(root)
    root.mainloop()