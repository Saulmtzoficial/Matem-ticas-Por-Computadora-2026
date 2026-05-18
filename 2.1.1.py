import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import math

# --- CONFIGURACIÓN DE COLORES ---
COLOR_BG = "#121212"
COLOR_ACCENT = "#ff073a"  # Rojo Neón
COLOR_A = "#00ffff"       # Cyan (Límite A)
COLOR_B = "#00ff00"       # Verde (Límite B)
COLOR_P = "#ffff00"       # Amarillo (Punto P)
TEXT_COLOR = "#ffffff"

class BisectionSolver:
    def __init__(self):
        self.history = [] 

    def solve(self, func_str, a, b, tol, max_iter):
        self.history = []
        
        # Diccionario seguro
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({k: v for k, v in np.__dict__.items() if not k.startswith("__")})
        
        def f(x):
            allowed_names['x'] = x
            try:
                val = eval(func_str, {"__builtins__": {}}, allowed_names)
                return val
            except:
                return float('nan')

        try:
            fa = f(a)
            fb = f(b)
        except:
            return None, "Error evaluando f(a) o f(b)"

        if np.isnan(fa) or np.isnan(fb):
             return None, "La función no es válida en los puntos a o b."

        if fa * fb > 0:
            return None, "Fallo Teorema Bolzano (mismo signo)."

        # Guardar estado inicial
        self.history.append((0, a, b, a, fa))

        iteration = 0
        p = a 
        
        while iteration < max_iter:
            iteration += 1
            p = (a + b) / 2
            fp = f(p)
            
            self.history.append((iteration, a, b, p, fp))

            if fp == 0 or (b - a) / 2 < tol:
                return p, f"Convergencia: {p:.6f} (Iter {iteration})"
            
            if fa * fp > 0:
                a = p
                fa = fp
            else:
                b = p
        
        return p, f"Iteraciones agotadas. Aprox: {p:.6f}"

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neon Bisection Solver")
        self.geometry("600x750")
        self.configure(bg=COLOR_BG)
        self.solver = BisectionSolver()
        self.style_gui()
        self.create_widgets()

    def style_gui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background=COLOR_BG)
        style.configure("TLabel", background=COLOR_BG, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TButton", background=COLOR_BG, foreground=COLOR_ACCENT, font=("Segoe UI", 10, "bold"), borderwidth=1)
        style.map("TButton", background=[("active", COLOR_ACCENT)], foreground=[("active", "white")])
        style.configure("TEntry", fieldbackground="#1e1e1e", foreground=COLOR_ACCENT, insertcolor="white", borderwidth=0)

    def create_widgets(self):
        tk.Label(self, text="MÉTODO DE BISECCIÓN", font=("Segoe UI", 18, "bold"), bg=COLOR_BG, fg=COLOR_ACCENT, pady=20).pack()
        
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=10, padx=30, fill="x")

        self.create_input(input_frame, "FUNCIÓN", "sqrt(x) - cos(x)")
        self.create_input(input_frame, "A (INICIO)", "0")
        self.create_input(input_frame, "B (FINAL)", "1")
        self.create_input(input_frame, "TOLERANCIA", "0.0001")
        self.create_input(input_frame, "ITERACIONES", "15")

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=20)
        
        ttk.Button(btn_frame, text="CALCULAR", command=self.calculate).pack(fill="x", ipady=8, pady=5)
        self.graph_btn = ttk.Button(btn_frame, text="VER GRÁFICA", command=self.show_graph, state="disabled")
        self.graph_btn.pack(fill="x", ipady=8, pady=5)

        self.result_var = tk.StringVar(value="...")
        tk.Label(self, textvariable=self.result_var, font=("Consolas", 11), bg="#000000", fg="#888", pady=10).pack(fill="x", padx=30)

    def create_input(self, parent, label, default):
        c = ttk.Frame(parent)
        c.pack(fill="x", pady=5)
        ttk.Label(c, text=label, font=("Segoe UI", 8, "bold"), foreground="#666").pack(anchor="w")
        e = ttk.Entry(c, font=("Consolas", 12))
        e.insert(0, default)
        e.pack(fill="x", ipady=5)
        setattr(self, f"entry_{label.split()[0].lower()}", e)

    def calculate(self):
        try:
            f_str = self.entry_función.get()
            a = float(self.entry_a.get())
            b = float(self.entry_b.get())
            tol = float(self.entry_tolerancia.get())
            mx = int(self.entry_iteraciones.get())

            res, msg = self.solver.solve(f_str, a, b, tol, mx)
            
            if res is not None:
                self.result_var.set(f"ÉXITO: {msg}")
                self.graph_btn.config(state="normal")
                if len(self.solver.history) >= 3:
                     p3 = self.solver.history[2][3]
                     self.result_var.set(f"{msg}\n[Tarea p3]: {p3}")
            else:
                self.result_var.set(f"ERROR: {msg}")
                self.graph_btn.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_graph(self):
        top = tk.Toplevel(self)
        top.title("Solución Gráfica")
        top.geometry("900x700")
        top.configure(bg=COLOR_BG)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor(COLOR_BG)

        # Preparar Datos
        f_str = self.entry_función.get()
        a_start = float(self.entry_a.get())
        b_start = float(self.entry_b.get())
        
        margin = (b_start - a_start) * 0.5
        x_min = a_start - margin
        x_max = b_start + margin
        x = np.linspace(x_min, x_max, 500)
        
        # Evaluar función
        allowed = {k: v for k, v in np.__dict__.items() if not k.startswith("__")}
        allowed.update({k: v for k, v in math.__dict__.items() if not k.startswith("__")})
        allowed['x'] = x
        try:
            y = eval(f_str, {"__builtins__": {}}, allowed)
            if np.iscomplexobj(y):
                y = np.real(y)
        except:
            y = np.zeros_like(x)

        # Calcular limites Y
        valid_y = y[~np.isnan(y)]
        if len(valid_y) == 0: valid_y = [0, 1]
        y_bottom, y_top = min(valid_y), max(valid_y)
        pad = (y_top - y_bottom) * 0.1
        if pad == 0: pad = 1
        y_bottom -= pad
        y_top += pad
        ax.set_ylim(y_bottom, y_top)

        # Función evaluadora
        def eval_func(x_val):
            allowed_eval = {k: v for k, v in np.__dict__.items() if not k.startswith("__")}
            allowed_eval.update({k: v for k, v in math.__dict__.items() if not k.startswith("__")})
            allowed_eval['x'] = x_val
            try:
                return eval(f_str, {"__builtins__": {}}, allowed_eval)
            except:
                return 0
        
        # Obtener solución final (último elemento del historial)
        final_iter, final_a, final_b, final_p, final_fp = self.solver.history[-1]
        
        # Calcular valores de la función en los puntos finales
        final_fa = eval_func(final_a)
        final_fb = eval_func(final_b)
        
        # Dibujar curva principal
        ax.plot(x, y, color=COLOR_ACCENT, lw=2.5, label='f(x)')
        ax.axhline(0, color='white', lw=0.8, linestyle='--', alpha=0.5)
        ax.axvline(final_p, color=COLOR_P, lw=1, linestyle=':', alpha=0.5)
        ax.grid(True, color='#333', linestyle=':', alpha=0.3)
        
        # Dibujar intervalo inicial (semi-transparente)
        ax.axvline(a_start, color=COLOR_A, lw=1, linestyle='--', alpha=0.3, label='Intervalo inicial')
        ax.axvline(b_start, color=COLOR_B, lw=1, linestyle='--', alpha=0.3)
        
        # Dibujar líneas verticales en límites finales
        ax.plot([final_a, final_a], [y_bottom, final_fa], color=COLOR_A, lw=2, alpha=0.7)
        ax.plot([final_b, final_b], [y_bottom, final_fb], color=COLOR_B, lw=2, alpha=0.7)
        
        # Sombreado del intervalo final
        ax.fill_between([final_a, final_b], y_bottom, y_top, color=COLOR_ACCENT, alpha=0.15)
        
        # Puntos en la curva
        ax.plot(final_a, final_fa, 'o', color=COLOR_A, ms=12, zorder=5, 
                markeredgecolor='white', markeredgewidth=2, label=f'a = {final_a:.5f}')
        ax.plot(final_b, final_fb, 'o', color=COLOR_B, ms=12, zorder=5, 
                markeredgecolor='white', markeredgewidth=2, label=f'b = {final_b:.5f}')
        ax.plot(final_p, final_fp, 'o', color=COLOR_P, ms=15, zorder=6, 
                markeredgecolor='white', markeredgewidth=2.5, label=f'Raíz ≈ {final_p:.6f}')
        
        # Etiquetas en la parte superior
        ax.text(final_a, y_top - pad/2, 'a', color=COLOR_A, ha='center', 
                fontweight='bold', fontsize=14, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='black', edgecolor=COLOR_A, linewidth=2))
        ax.text(final_b, y_top - pad/2, 'b', color=COLOR_B, ha='center', 
                fontweight='bold', fontsize=14, bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='black', edgecolor=COLOR_B, linewidth=2))
        ax.text(final_p, y_top - pad*1.5, 'RAÍZ', color=COLOR_P, ha='center', 
                fontweight='bold', fontsize=12, bbox=dict(boxstyle='round,pad=0.4', 
                facecolor='black', edgecolor=COLOR_P, linewidth=2))
        
        # Cuadro de información
        info_text = f"""SOLUCIÓN FINAL
{'='*30}
Iteraciones: {final_iter}
Intervalo final: [{final_a:.6f}, {final_b:.6f}]
Raíz aproximada: {final_p:.8f}
f(raíz) = {final_fp:.2e}
Ancho intervalo: {(final_b - final_a):.2e}"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, color='white', 
                fontfamily='Consolas', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, 
                         edgecolor=COLOR_ACCENT, linewidth=2))
        
        # Configuración final
        ax.set_xlabel('x', color='white', fontsize=12, fontweight='bold')
        ax.set_ylabel('f(x)', color='white', fontsize=12, fontweight='bold')
        ax.set_title('MÉTODO DE BISECCIÓN - SOLUCIÓN FINAL', 
                    color=COLOR_ACCENT, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', framealpha=0.9, edgecolor='white')
        
        # Crear canvas
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()

if __name__ == "__main__":
    app = App()
    app.mainloop()