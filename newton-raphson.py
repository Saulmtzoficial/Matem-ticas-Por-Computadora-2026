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

def numerical_derivative(func, x, h=1e-7):
    """
    Calculates numerical derivative using central difference method.
    """
    return (func(x + h) - func(x - h)) / (2 * h)

def run_simulation(f_str, start_x, tolerance):
    """
    Runs the Newton-Raphson simulation and displays a table of iterations.
    """
    
    # --- 1. Define Math Logic from Input ---
    try:
        # Safe dictionary for eval
        safe_dict = {
            "np": np, "x": 0, 
            "sin": np.sin, "cos": np.cos, "tan": np.tan, 
            "exp": np.exp, "sqrt": np.sqrt, 
            "log": np.log, "ln": np.log, "pi": np.pi,
            "abs": np.abs, "arcsin": np.arcsin, "arccos": np.arccos,
            "arctan": np.arctan, "sinh": np.sinh, "cosh": np.cosh,
            "tanh": np.tanh, "e": np.e
        }
        
        func = lambda x: eval(f_str, {**safe_dict, "x": x})
        func(start_x) # Test run
    except Exception as e:
        messagebox.showerror("Math Error", f"Invalid function expression: {e}")
        return

    # --- 2. Newton-Raphson Calculation & Data Collection ---
    steps = [start_x]
    table_data = [] # List to store row data for the table
    iterations = 20 
    
    for i in range(iterations):
        current_x = steps[-1]
        try:
            f_val = func(current_x)
            d_val = numerical_derivative(func, current_x)
            
            if abs(d_val) < 1e-10: break 
            
            next_x = current_x - f_val / d_val
            error = abs(next_x - current_x)
            
            # Store data: Iteration, x_n, f(x_n), f'(x_n), x_{n+1}, Error
            # Formatting numbers to strings for the table
            table_data.append([
                str(i + 1),
                f"{current_x:.5f}",
                f"{f_val:.5f}",
                f"{d_val:.5f}",
                f"{next_x:.5f}",
                f"{error:.5e}"
            ])

            steps.append(next_x)
            
            if error < tolerance:
                break
        except Exception:
            break

    # --- 3. Visual Setup (Dark Mode) ---
    plt.style.use('dark_background')
    
    # Create a figure with 2 subplots (Plot on top, Table on bottom)
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1]) # 3 parts graph, 1 part table
    
    ax = fig.add_subplot(gs[0])       # Main Graph Axis
    ax_table = fig.add_subplot(gs[1]) # Table Axis
    ax_table.axis('off')              # Hide axis lines for the table area

    plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.2) 

    # --- GRAPH PLOTTING ---
    margin = max(abs(steps[0] - steps[-1]) * 0.5, 2.0)
    domain = (min(steps) - margin, max(steps) + margin)
    
    x_vals = np.linspace(domain[0], domain[1], 500)
    try:
        y_vals = np.array([func(v) for v in x_vals])
    except:
        y_vals = x_vals * 0 

    ax.plot(x_vals, y_vals, color='#00FFFF', linewidth=2.5, label=f'f(x) = {f_str}')
    ax.axhline(0, color='white', linewidth=0.8, linestyle='--')
    ax.set_xlim(domain[0], domain[1])
    
    y_min, y_max = min(y_vals), max(y_vals)
    ax.set_ylim(y_min - abs(y_min)*0.1 - 1, y_max + abs(y_max)*0.1 + 1)

    # Animated Elements
    point_on_curve, = ax.plot([], [], 'o', color='white', markersize=6, zorder=5)
    point_on_axis, = ax.plot([], [], 'x', color='#FFFF00', markersize=8, markeredgewidth=2, zorder=5)
    tangent_line, = ax.plot([], [], color='#FF0044', linewidth=2, label='Tangent')
    vertical_line, = ax.plot([], [], color='white', linestyle=':', linewidth=1, alpha=0.6)
    text_info = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', fontsize=12, verticalalignment='top')

    ax.set_title(f"Newton-Raphson: {f_str}", color='white', fontsize=14, pad=10)
    ax.legend(loc='upper right', frameon=False)

    # --- TABLE CREATION ---
    columns = ("Iter", "x_n", "f(x_n)", "f'(x_n)", "x_n+1", "Error")
    
    # If no steps were taken (error or immediate finish), provide a placeholder
    if not table_data:
        table_data = [["-", "-", "-", "-", "-", "-"]]

    the_table = ax_table.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table for Dark Mode
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.3) # Adjust row height

    for (row, col), cell in the_table.get_celld().items():
        cell.set_edgecolor('#555555')
        if row == 0:
            # Header Styling
            cell.set_facecolor('#333333')
            cell.set_text_props(weight='bold', color='#00FFFF')
        else:
            # Data Styling
            cell.set_facecolor('#1E1E1E')
            cell.set_text_props(color='white')

    # --- 4. Menu Button ---
    ax_btn = plt.axes([0.85, 0.02, 0.1, 0.04]) # Positioned at very bottom right
    btn = Button(ax_btn, 'Menu', color='#333333', hovercolor='#555555')
    btn.label.set_color('white')
    btn.on_clicked(lambda e: plt.close(fig))

    # --- 5. Animation Logic ---
    def update(frame):
        step_index = frame // 2 
        phase = frame % 2 
        
        if step_index >= len(steps) - 1:
            return point_on_curve, tangent_line, vertical_line, point_on_axis, text_info

        curr_x = steps[step_index]
        curr_y = func(curr_x)
        
        text_info.set_text(f'Iteration: {step_index}\nx ≈ {curr_x:.6f}')
        point_on_curve.set_data([curr_x], [curr_y])
        
        if phase == 0:
            vertical_line.set_data([curr_x, curr_x], [0, curr_y])
            tangent_line.set_data([], []) 
            point_on_axis.set_data([], [])
        else:
            next_x = steps[step_index + 1]
            t_x = np.array([curr_x, next_x])
            t_y = np.array([curr_y, 0])
            
            tangent_line.set_data(t_x, t_y)
            point_on_axis.set_data([next_x], [0])

        return point_on_curve, tangent_line, vertical_line, point_on_axis, text_info

    ani = FuncAnimation(fig, update, frames=(len(steps)-1)*2, interval=1000, blit=True)
    plt.show()

# ==========================================
#           GUI MENU (TKINTER)
# ==========================================

class NewtonMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Newton-Raphson Simulator")
        self.root.geometry("500x400")
        self.root.configure(bg="#1E1E1E")
        self.root.resizable(False, False)

        title_font = tkfont.Font(family="Helvetica", size=16, weight="bold")
        
        tk.Label(root, text="Newton-Raphson Method", font=title_font, 
                 bg="#1E1E1E", fg="#00FFFF", pady=20).pack()

        # Input Fields
        self.f_entry = self.create_input("Function f(x):", "x**3 - 2*x**2 - 5")
        self.x0_entry = self.create_input("Initial Guess (x0):", "3.0")
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
            tol = float(self.tol_entry.get())
            
            self.root.withdraw()
            run_simulation(f_s, x0, tol)
            self.root.deiconify()
        except ValueError:
            messagebox.showerror("Input Error", "Please ensure x0 and Tolerance are valid numbers.")

if __name__ == "__main__":
    root = tk.Tk()
    app = NewtonMenu(root)
    root.mainloop()