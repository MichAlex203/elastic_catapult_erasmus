# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:20:19 2026

@author: Micha

Custom Newton-Raphson Solver
"""

import numpy as np

def custom_newton_raphson(N, h, l_square, theta_guess, current_alpha, tol=1e-8, max_iter=50):
    """
    Solves the non-linear discretized ODE using the Newton-Raphson method.
    """
    theta = np.copy(theta_guess)
    
    for iteration in range(max_iter):
        R = np.zeros(N)        # Residuals vector
        J = np.zeros((N, N))   # Jacobian matrix
        
        # 1. Boundary condition for s=0: θ(0) = 0
        R[0] = theta[0]
        J[0, 0] = 1.0
        
        # 2. Internal nodes (1 to N-2)
        for i in range(1, N - 1):
            # Finite differences for: θ'' - λ^2 * sin(θ + α) = 0
            # (Note the MINUS sign to match the system's potential energy definition)
            R[i] = (theta[i+1] - 2*theta[i] + theta[i-1]) / (h**2) - l_square * np.sin(theta[i] + current_alpha)
            
            # Derivatives for Jacobian
            J[i, i-1] = 1.0 / (h**2)                                                # w.r.t θ_{i-1}
            J[i, i]   = -2.0 / (h**2) - l_square * np.cos(theta[i] + current_alpha) # w.r.t θ_i
            J[i, i+1] = 1.0 / (h**2)                                                # w.r.t θ_{i+1}
            
        # 3. Boundary condition for node N-1: θ'(1) = 0 (Backward difference)
        R[-1] = theta[-1] - theta[-2]
        J[-1, -1] = 1.0
        J[-1, -2] = -1.0
        
        # Convergence check
        max_error = np.max(np.abs(R))
        if max_error < tol:
            return theta, True
            
        # Solving linear system J * Δθ = -R
        try:
            delta_theta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            return theta, False # Jacobian became singular (bifurcation/limit point)
            
        # Update Solution
        theta = theta + delta_theta
        
    return theta, False


