import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Physics (Undamped LC) ---
# Alpha = 0 (No damping), pure oscillation
t = np.linspace(0, 10, 500)
omega0 = 3.0  # Frequency of oscillation

# Initial Conditions: Starts at 0, oscillates forever
A = 3
v = A * np.sin(omega0 * t)

# --- Visual Configuration ---
plt.style.use('dark_background')

# 9:16 Aspect Ratio
fig, ax = plt.subplots(figsize=(6, 10.6)) 

# Colors
bg_color = '#000000'       
line_color = '#ff1a1a'     
glow_color = '#800000'     
text_color = '#ffcccc'     

fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Remove standard axes
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#330000') 
ax.spines['left'].set_color('#330000')

# Hide ticks
ax.tick_params(axis='x', colors=text_color, labelsize=8)
ax.tick_params(axis='y', colors=text_color, labelsize=8)
ax.set_ylim(-4, 4)  # Adjusted limits for the wave
ax.set_xlim(0, 10)

# --- Plot Objects ---
# 1. Wide Glow
glow1, = ax.plot([], [], color='red', linewidth=15, alpha=0.05)
# 2. Medium Glow
glow2, = ax.plot([], [], color=glow_color, linewidth=6, alpha=0.3)
# 3. Core Line
line, = ax.plot([], [], color=line_color, linewidth=2.5, alpha=1.0)
# 4. Scanner Head
head, = ax.plot([], [], 'o', color='white', markersize=6, markeredgecolor='red')

# Text Labels
title = ax.text(0.5, 0.85, "UNDAMPED", transform=ax.transAxes, 
                ha='center', color=line_color, fontsize=24, 
                fontweight='bold') 

subtitle = ax.text(0.5, 0.81, "PURE OSCILLATION", transform=ax.transAxes, 
                   ha='center', color='white', fontsize=10, 
                   fontfamily='monospace', alpha=0.7)

# Equation for Undamped: Simple Harmonic Motion
eq_text = ax.text(0.5, 0.1, r"$v(t) = A \sin(\omega_0 t)$", 
                  transform=ax.transAxes, ha='center', color='gray', fontsize=14, alpha=0.5)

def init():
    line.set_data([], [])
    glow1.set_data([], [])
    glow2.set_data([], [])
    head.set_data([], [])
    return line, glow1, glow2, head

def update(frame):
    x_data = t[:frame]
    y_data = v[:frame]
    
    line.set_data(x_data, y_data)
    glow1.set_data(x_data, y_data)
    glow2.set_data(x_data, y_data)
    
    if frame > 0:
        head.set_data([t[frame-1]], [v[frame-1]])
    
    return line, glow1, glow2, head

# --- Run Animation ---
# interval=20 means roughly 50fps. Increase to 30 for a slower, lazier wave.
ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, 
                              interval=20, blit=True)

plt.tight_layout()
plt.show()