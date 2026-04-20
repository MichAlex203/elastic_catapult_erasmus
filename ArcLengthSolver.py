# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:57:13 2026

@author: Micha

Arc Lenghth Method Solver 
"""

import numpy as np

def R_J(N, h, l_square, theta, alpha, current_l_sq):
    """Calculates Residuals, Jacobian, and the derivative of R w.r.t alpha."""
    R = np.zeros(N)
    J = np.zeros((N, N))
    dR_da = np.zeros(N)
    
    R[0] = theta[0]
    J[0, 0] = 1.0
    
    for i in range(1, N - 1):
        R[i] = (theta[i+1] - 2*theta[i] + theta[i-1]) / (h**2) - current_l_sq * np.sin(theta[i] + alpha)
        
        J[i, i-1] = 1.0 / (h**2)
        J[i, i]   = -2.0 / (h**2) - current_l_sq * np.cos(theta[i] + alpha)
        J[i, i+1] = 1.0 / (h**2)
        
        # New: Derivative of the internal node residuals with respect to alpha
        dR_da[i]  = -current_l_sq * np.cos(theta[i] + alpha)
        
    R[-1] = theta[-1] - theta[-2]
    J[-1, -1] = 1.0
    J[-1, -2] = -1.0
    
    return R, J, dR_da

def arc_length_step(N, h, l_square, theta_prev, alpha_prev, t_theta_prev, t_alpha_prev, ds, current_l_sq, tol=1e-6, max_iter=20):
    """Performs one predictor-corrector step along the equilibrium path."""
    
    # --- 1. PREDICTOR STEP ---
    theta = theta_prev + ds * t_theta_prev
    alpha = alpha_prev + ds * t_alpha_prev
    
    # --- 2. CORRECTOR STEP ---
    for _ in range(max_iter):
        R, J, dR_da = R_J(N, h, l_square, theta, alpha, current_l_sq)
        
        if np.max(np.abs(R)) < tol:
            break
            
        try:
            delta_theta_R = np.linalg.solve(J, -R)
            delta_theta_a = np.linalg.solve(J, -dR_da)
        except np.linalg.LinAlgError:
            return theta, alpha, False, t_theta_prev, t_alpha_prev
            
        # Arc-length constraint
        denominator = t_alpha_prev + np.dot(t_theta_prev, delta_theta_a)
        if abs(denominator) < 1e-12: # Prevent division by zero
            return theta, alpha, False, t_theta_prev, t_alpha_prev
            
        delta_alpha = -(np.dot(t_theta_prev, delta_theta_R)) / denominator
        delta_theta = delta_theta_R + delta_alpha * delta_theta_a
        
        theta += delta_theta
        alpha += delta_alpha
    else:
        return theta, alpha, False, t_theta_prev, t_alpha_prev

    # --- 3. CALCULATE NEW TANGENT ---
    _, J_new, dR_da_new = R_J(N, h, l_square, theta, alpha, current_l_sq)
    try:
        t_theta_new = np.linalg.solve(J_new, -dR_da_new)
        t_alpha_new = 1.0
    except np.linalg.LinAlgError:
        return theta, alpha, False, t_theta_prev, t_alpha_prev
        
    norm = np.sqrt(np.dot(t_theta_new, t_theta_new) + t_alpha_new**2)
    t_theta_new /= norm
    t_alpha_new /= norm
    
    # Keep tangent pointing forward
    if np.dot(t_theta_prev, t_theta_new) + t_alpha_prev * t_alpha_new < 0:
        t_theta_new *= -1.0
        t_alpha_new *= -1.0

    return theta, alpha, True, t_theta_new, t_alpha_new