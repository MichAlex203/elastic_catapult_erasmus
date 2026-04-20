# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:19:13 2026

@author: Michaela Alexandridi

BIP Sorbone Code - Main file
Final Project: Elastic Catapult
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from NR_Solver import custom_newton_raphson
from ArcLengthSolver import arc_length_step
from newPostProcessor import deformed_shape, plot_moment_rotation, merged_plot_NR_AL, plot_AL_with_dashed_lines

# -------------
# Parameters
# -------------
N = 100                 # Number of nodes for discretization
h = 1.0 / (N - 1)       # Discretization step size
l_square = np.pi**2     # Dimensionless load parameter (λ^2 = P*L^2/B = π^2)
num_steps = 300         # Number of steps for the continuation method sweeps

def calculate_m0(theta_sol):
    """Calculates the dimensionless Bending Moment at the clamp M0 = θ'(0)"""
    # 2nd-order forward difference for higher accuracy at x=0
    return -((-3*theta_sol[0] + 4*theta_sol[1] - theta_sol[2]) / (2*h))


# ==========================================
# EXECUTION: Running the Arc-Length Continuation
# ==========================================

ds = 0.1    # Arc-length step size 

theta_curr = 0.01 * np.linspace(0, 1, N) # Small initial guess so it doesn't balance perfectly at 0
alpha_curr = 0.0
m0_vals = []
alpha_vals = []

t_theta_curr = np.zeros(N)
t_alpha_curr = 1.0

max_steps = 5000
for step in range(max_steps):
    theta_curr, alpha_curr, success, t_theta_curr, t_alpha_curr = arc_length_step(
        N, h, l_square, theta_curr, alpha_curr, t_theta_curr, t_alpha_curr, ds, l_square
    )
    
    if not success:
        print(f"Solver failed to converge at step {step}. Try reducing 'ds'.")
        break
        
    m0_vals.append(calculate_m0(theta_curr)) 
    alpha_vals.append(alpha_curr)
    
    if alpha_curr > 2 * np.pi:
        break
    
# =====================
# Deformed Shape for α = [0,2π]
# ======================

# -------------
# Continuation Method & Shape Collection
# -------------
alphas_forward = np.linspace(0, 2*np.pi, num_steps)
theta_guess = 0.1 * np.linspace(0, 1, N)

stored_X = []
stored_Y = []
stored_alphas = []

plot_interval = max(1, num_steps // 15)

for step, a in enumerate(alphas_forward):
    theta_sol, success = custom_newton_raphson(N, h, l_square, theta_guess, a)
    if success:
        # Save geometry at specific intervals
        if step % plot_interval == 0:
            X, Y = deformed_shape(N, h, theta_sol, a)
            
            stored_X.append(X)
            stored_Y.append(Y)
            stored_alphas.append(a)
            
        theta_guess = theta_sol 

# -------------
# Plotting All Shapes
# -------------
plt.figure(figsize=(10, 8))

cmap = cm.viridis
colors = [cmap(i) for i in np.linspace(0, 1, len(stored_alphas))]

for i in range(len(stored_alphas)):
    alpha_deg = stored_alphas[i] #* 180 / np.pi
    plt.plot(stored_X[i], stored_Y[i], color=colors[i], linewidth=2, alpha=0.8,
             label=f'$\\alpha \\approx {alpha_deg:.3f}$')
    plt.plot(stored_X[i][-1], stored_Y[i][-1], 'o', color=colors[i], markersize=5)

plt.plot(0, 0, 'ks', markersize=10, label='Clamp (0,0)')

plt.title('Evolution of the Deformed Shape for varying $\\alpha$ (Global Frame)')
plt.xlabel('X / L')
plt.ylabel('Y / L')
plt.axis('equal')
plt.grid(True, linestyle=':', alpha=0.7)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
    
# ==========================================
# Continuation Method & Snapping
# ==========================================

# 1. Forward Sweep (Increasing alpha: 0 -> 2π)
alphas_forward = np.linspace(0, 2*np.pi, num_steps)
m0_fwd = []
alpha_fwd_valid = []

theta_guess = 0.1 * np.linspace(0, 1, N) # Initial guess for the very first step

for a in alphas_forward:
    theta_sol, success = custom_newton_raphson(N, h, l_square, theta_guess, a)
    if success:
        m0_fwd.append(calculate_m0(theta_sol))
        alpha_fwd_valid.append(a)
        theta_guess = theta_sol # Continuation step

# 2. Backward Sweep (Decreasing alpha: 2π -> 0)
alphas_backward = np.linspace(2*np.pi, 0, num_steps)
m0_bwd = []
alpha_bwd_valid = []

for a in alphas_backward:
    theta_sol, success = custom_newton_raphson(N, h, l_square, theta_guess, a)
    if success:
        m0_bwd.append(calculate_m0(theta_sol))
        alpha_bwd_valid.append(a)
        theta_guess = theta_sol

# Find the maximum absolute bending moment from the forward sweep
max_m0_val_fwd = np.max(np.abs(m0_fwd))
max_m0_val_bwd = np.max(np.abs(m0_bwd))


print(f"The maximum calculated M0 is: {max_m0_val_fwd:.3f}")
print(f"The maximum calculated M0 is: {max_m0_val_bwd:.3f}")

# -------------
# Snapping Detection (Jumps in M0)
# -------------
dm0_fwd = np.abs(np.diff(m0_fwd))
snap_idx_fwd = np.argmax(dm0_fwd)         
alpha_snap_fwd = alpha_fwd_valid[snap_idx_fwd]

dm0_bwd = np.abs(np.diff(m0_bwd))
snap_idx_bwd = np.argmax(dm0_bwd)
alpha_snap_bwd = alpha_bwd_valid[snap_idx_bwd]

print(f"Forward Loading Snapping occurs at: {alpha_snap_fwd:.3f} rad ({alpha_snap_fwd * 180/np.pi:.1f} deg)")
print(f"Backward Unloading Snapping occurs at: {alpha_snap_bwd:.3f} rad ({alpha_snap_bwd * 180/np.pi:.1f} deg)")

# -------------
# Plotting M0-Alpha
# -------------
plot_moment_rotation(alpha_fwd_valid, m0_fwd, alpha_bwd_valid, m0_bwd, alpha_snap_fwd, alpha_snap_bwd)

# ==========================================
# MERGED PLOTTING: Newton-Raphson vs Arc-Length
# ==========================================
merged_plot_NR_AL(alpha_bwd_valid, alpha_fwd_valid, m0_bwd, m0_fwd, alpha_snap_bwd, alpha_snap_fwd, alpha_vals, m0_vals)

# -------------
# Plotting: Arc-Length with Dashed Unstable Regions
# -------------
plot_AL_with_dashed_lines(alpha_vals, m0_vals, l_square)

