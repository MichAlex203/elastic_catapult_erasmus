# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:21:37 2026

@author: Micha
"""

import numpy as np
import matplotlib.pyplot as plt

def deformed_shape(N, h, theta_sol_fixed, alpha_fixed):
    X = np.zeros(N)
    Y = np.zeros(N)
    for i in range(1, N):
        # Correct kinematics based on the paper's coordinate system (gravity is -y)
        # Tangent angle w.r.t vertical is (α + θ)
        dX_ds = np.sin(theta_sol_fixed[i] + alpha_fixed)
        dY_ds = -np.cos(theta_sol_fixed[i] + alpha_fixed)
        dX_ds_prev = np.sin(theta_sol_fixed[i-1] + alpha_fixed)
        dY_ds_prev = -np.cos(theta_sol_fixed[i-1] + alpha_fixed)
        
        X[i] = X[i-1] + 0.5 * h * (dX_ds + dX_ds_prev)
        Y[i] = Y[i-1] + 0.5 * h * (dY_ds + dY_ds_prev)
        
    return X, Y
    

def plot_moment_rotation(alpha_fwd_valid, m0_fwd, alpha_bwd_valid, m0_bwd, alpha_snap_fwd, alpha_snap_bwd):
    """
    Plotting M0-Alpha
    """
    plt.figure(figsize=(10, 6))

    plt.plot(alpha_fwd_valid, m0_fwd, label='Forward Sweep (Increasing $\\alpha$)', color='blue', lw=2)
    plt.plot(alpha_bwd_valid, m0_bwd, label='Backward Sweep (Decreasing $\\alpha$)', color='red', lw=2)

    plt.axvline(alpha_snap_fwd, color='blue', linestyle=':', alpha=0.5, label=f'Forward Snapping at ~{alpha_snap_fwd:.2f} rad')
    plt.axvline(alpha_snap_bwd, color='red', linestyle=':', alpha=0.5, label=f'Backward Snapping at ~{alpha_snap_bwd:.2f} rad')

    plt.title('Moment-Rotation Diagram ($M_0$ vs $\\alpha$) - Elastica Catapult')
    plt.xlabel('Clamp Inclination $\\alpha$ (rad)')
    plt.ylabel('Dimensionless Bending Moment $M_0 = \\theta\'(0)$ (rad)')

    # Format x-axis with exact Pi multiples
    pi_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    pi_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    plt.xticks(pi_ticks, pi_labels)
    
    plt.grid(True, linestyle=':', alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def merged_plot_NR_AL(alpha_bwd_valid, alpha_fwd_valid, m0_bwd, m0_fwd, alpha_snap_bwd, alpha_snap_fwd, alpha_vals, m0_vals):
    plt.figure(figsize=(10, 6))

    # 1. Plot Newton-Raphson Sweeps
    plt.plot(alpha_fwd_valid, m0_fwd, label='Newton-Raphson: Forward', color='blue', lw=2)
    plt.plot(alpha_bwd_valid, m0_bwd, label='Newton-Raphson: Backward', color='red', lw=2)

    # 2. Plot Snapping Lines
    plt.axvline(alpha_snap_fwd, color='blue', linestyle=':', alpha=0.5, label='Forward Snapping')
    plt.axvline(alpha_snap_bwd, color='red', linestyle=':', alpha=0.5, label='Backward Snapping')

    # 3. Plot Arc-Length Path
    # We make this slightly thicker and dashed so it stands out while perfectly overlapping the others!
    plt.plot(alpha_vals, m0_vals, label='Arc-Length Path (True Equilibrium)', color='green', lw=3, linestyle='--')

    plt.title('Moment-Rotation Diagram: Newton-Raphson vs Arc-Length')
    plt.xlabel(r'Clamp Inclination $\alpha$ (rad)')
    plt.ylabel(r'Dimensionless Bending Moment $M_0$')

    # Format x-axis with exact Pi multiples
    pi_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    pi_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    plt.xticks(pi_ticks, pi_labels)

    plt.grid(True, linestyle=':', alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_AL_with_dashed_lines(alpha_vals, m0_vals, l_square):
    """
    Plots the Moment-Rotation Diagram using the Arc-Length Method,
    highlighting stable (solid) and unstable/snapping (dashed) regions.
    """
    plt.figure(figsize=(10, 6))

    # Convert lists to numpy arrays for easier math
    a_arr = np.array(alpha_vals)
    m_arr = np.array(m0_vals)

    # Find the exact indices where the curve changes direction (turns backward or forward)
    # We do this by looking at the sign change of the difference between consecutive alpha values
    d_alpha = np.diff(a_arr)
    turning_points = np.where(np.diff(np.sign(d_alpha)) != 0)[0] + 1

    start_idx = 0
    added_solid_label = False
    added_dashed_label = False

    # Loop through each segment of the curve between the turning points
    for i in range(len(turning_points) + 1):
        end_idx = turning_points[i] + 1 if i < len(turning_points) else len(a_arr)
        
        segment_alpha = a_arr[start_idx:end_idx]
        segment_m0 = m_arr[start_idx:end_idx]
        
        # Check if the segment is moving forward (stable) or backward (unstable)
        if segment_alpha[-1] >= segment_alpha[0]:
            line_style = '-'  # Solid for stable
            label = 'Stable Path' if not added_solid_label else None
            added_solid_label = True
        else:
            line_style = '--' # Dashed for unstable snapping region
            label = 'Unstable Path (Snapping)' if not added_dashed_label else None
            added_dashed_label = True

        # Plot the specific segment
        plt.plot(segment_alpha, segment_m0, color='green', lw=2, linestyle=line_style, label=label)
        
        # Start the next segment one point back so the lines connect seamlessly
        start_idx = end_idx - 1

    # Formatting (Added 'fr' so f-strings play nicely with LaTeX symbols)
    plt.title(fr'Moment-Rotation Diagram (Arc-Length Method, $\lambda^2 = {l_square:.3f}$)')
    plt.xlabel(r'Clamp Inclination $\alpha$ (rad)')
    plt.ylabel(r'Dimensionless Bending Moment $M_0$')

    pi_ticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    pi_labels = ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
    plt.xticks(pi_ticks, pi_labels)

    plt.grid(True, linestyle=':', alpha=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()