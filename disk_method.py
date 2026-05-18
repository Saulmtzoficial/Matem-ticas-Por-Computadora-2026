import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
# Function to rotate: f(x) = sqrt(x) + 0.5
# (Added 0.5 to give the object some thickness at the base)
def func(x):
    return np.sqrt(x) + 0.3

# Domain
x_min, x_max = 0, 4
resolution = 100

# Visual Style
plt.style.use('dark_background') # Minimalistic dark theme
color_map = 'cyan'               # "Tron" like aesthetic
line_alpha = 0.6

# --- Setup Data ---
x = np.linspace(x_min, x_max, resolution)
y = func(x)

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Remove axes/ticks for a cleaner, sophisticated look
ax.set_axis_off()
ax.set_facecolor('black') 
fig.patch.set_facecolor('black')

# Text placeholder for calculus notation
title_text = ax.text2D(0.5, 0.95, "", transform=ax.transAxes, 
                       ha="center", color="white", fontsize=14, fontfamily='monospace')

# Placeholders for the plot objects
# We use a wireframe for a "mathematical mesh" look rather than a solid surface
plot_surface = None

def update(frame):
    global plot_surface
    
    # Clean up previous frame
    if plot_surface:
        plot_surface.remove()
        
    # Current angle (from 0 to 2*pi)
    # We animate 0 to 360 degrees (2pi radians)
    theta_max = (frame / 100.0) * 2 * np.pi
    
    # If the revolution is complete, keep spinning the camera
    if theta_max > 2 * np.pi:
        theta_max = 2 * np.pi
        ax.view_init(elev=20, azim=frame) # Gentle rotation after formation
    
    # Create the meshgrid for the surface up to the current angle
    # We use fewer points in theta for a stylistic "grid" look
    theta = np.linspace(0, theta_max, 60) 
    X_grid, Theta_grid = np.meshgrid(x, theta)
    
    # Parametric equations for revolution around x-axis
    # X stays x
    # Y = f(x) * cos(theta)
    # Z = f(x) * sin(theta)
    Y_grid = func(X_grid) * np.cos(Theta_grid)
    Z_grid = func(X_grid) * np.sin(Theta_grid)
    
    # Plot the surface
    # 'rcount' and 'ccount' control the wireframe density
    plot_surface = ax.plot_surface(X_grid, Y_grid, Z_grid, 
                                   color=color_map, 
                                   alpha=0.3, 
                                   edgecolor='white', 
                                   linewidth=0.2,
                                   rcount=20, ccount=40,
                                   antialiased=True)
    
    # Plot the initial 2D curve (the "rib") to show origin
    ax.plot(x, func(x), np.zeros_like(x), color='red', linewidth=2, linestyle='--')

    # Dynamic Title
    percent = int((theta_max / (2*np.pi)) * 100)
    if percent >= 100:
        title_text.set_text(r"$V = \int_{0}^{4} \pi (\sqrt{x} + 0.3)^2 dx$")
    else:
        title_text.set_text(f"Revolving... {percent}%")
        
    return plot_surface, title_text

# --- Run Animation ---
# Frames: 100 frames to build, plus 100 frames to rotate the finished solid
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.tight_layout()
plt.show()