import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Algorithm: Regla Falsa (Your Original Logic)
def reglafalsa(f, x0, x1, tol=1e-8):
    if f(x0) * f(x1) > 0:
        raise ValueError('La funcion no cambia de signo en este rango')
    
    sigue = False
    print('{:^10s} {:^10s} {:^10s} {:^10s} {:^10s} {:^10s}'.format('x0','x','x1','f(x0)','f(x)','f(x1)'))
    
    while (not sigue):
        # False position formula
        x = ( x0*f(x1) - x1*f(x0) ) / ( f(x1) - f(x0) )
        print('{:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(x0,x,x1,f(x0),f(x),f(x1)))
        
        if f(x0) * f(x) < 0:
            x1 = x
        else:
            x0 = x
        sigue = abs(f(x)) < tol
    return x

# Physics Function: Volume of spherical segment - target volume
def f(h):
    R = 10
    target_v = 800
    return (np.pi * h**2 * (3*R - h) / 3) - target_v

def main():
    # 1. Solve for the height
    x0, x1 = 0, 15  # Range where the solution lies
    tol = 1e-4
    raiz = reglafalsa(f, x0, x1, tol)
    
    print(f'\nCalculated Height: {raiz:.4f} units')

    # 2. Setup the Pop-up Animation Window
    R = 10
    fig, ax = plt.subplots(figsize=(6, 7))
    plt.subplots_adjust(bottom=0.2)
    
    ax.set_xlim(-12, 12)
    ax.set_ylim(-2, 22)
    ax.set_aspect('equal')
    ax.set_title("Spherical Tank Filling Simulation", fontsize=12, fontweight='bold')
    ax.set_ylabel("Height (h)")

    # Drawing the Tank (Circle)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R*np.cos(theta), R + R*np.sin(theta), color='black', lw=3, label='Tank Wall')

    # Water setup
    # We use a Polygon to represent the filled area
    water_poly = ax.fill_between([], [], color='#3498db', alpha=0.7, label='Water')[0]
    
    # Text annotation for real-time height
    txt = ax.text(-9, 19, '', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    def update(frame):
        # Progressively increase height to the calculated root
        h_current = (frame / 100) * raiz
        
        # Calculate X boundaries for each Y slice using circle equation: x^2 + (y-R)^2 = R^2
        y_slice = np.linspace(0, h_current, 100)
        x_slice = np.sqrt(np.maximum(0, R**2 - (y_slice - R)**2))
        
        # Create vertices for the fill_between (mirroring x for the left side)
        verts = np.column_stack([np.concatenate([-x_slice, x_slice[::-1]]), 
                                 np.concatenate([y_slice, y_slice[::-1]])])
        
        water_poly.set_xy(verts)
        txt.set_text(f'Current Height: {h_current:.2f}\nTarget Volume: 800')
        return water_poly, txt

    # Create Animation
    # frames=100 controls the duration/smoothness of the fill
    ani = FuncAnimation(fig, update, frames=101, interval=30, blit=True, repeat=False)

    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # This triggers the pop-up window
    plt.show()

if __name__ == "__main__":
    main()