import numpy as np
from cmaes import CMA
from scipy.optimize import minimize

# SGD optimizer
def sgd_optimization(func, start_point, lr=0.05, iterations=50):
    reach_min = False
    opt_steps = iterations
    path = [start_point]
    x = start_point.copy()
    for _ in range(iterations):
        grad = np.array([func(x + np.eye(1, len(x), i)[0] * 1e-8) - func(x) for i in range(len(x))]) / 1e-8
        x = x - lr * grad
        path.append(x.copy())
        if np.allclose(np.round(np.abs(x), 2), np.round(np.abs(func.global_min), 2)):
            reach_min = True
            opt_steps = _
            break
    return np.array(path), reach_min, opt_steps, x

# Adam optimizer
def adam_optimization(func, start_point, lr=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8, iterations=50):
    reach_min = False
    path = [start_point]
    opt_steps = iterations
    x = start_point.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    t = 0
    for _ in range(iterations):
        t += 1
        grad = np.array([func(x + np.eye(1, len(x), i)[0] * 1e-8) - func(x) for i in range(len(x))]) / 1e-8
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
        path.append(x.copy())
        if np.allclose(np.round(np.abs(x), 2), np.round(np.abs(func.global_min), 2)):
            reach_min = True
            opt_steps = t
            break
    return np.array(path), reach_min, opt_steps, x

# CMA-ES
def cmaes_optimization(func, start_point, sigma=1.3, iterations=50):
    reach_min = False
    opt_steps = iterations
    optimizer = CMA(mean=start_point, sigma=sigma)
    path = [start_point.copy()]
    for generation in range(iterations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = func(x)
            solutions.append((x, value))
        optimizer.tell(solutions)
        path.append(optimizer._mean.copy())
        if np.allclose(np.round(np.abs(optimizer._mean), 2), np.round(np.abs(func.global_min), 2)):
            reach_min = True
            opt_steps = generation
            break
    return np.array(path), reach_min, opt_steps, optimizer._mean

# LRA-CMA
def lra_cma_optimization(func, start_point, sigma=1.3, iterations=50):
    reach_min = False
    opt_steps = iterations
    optimizer = CMA(mean=start_point, sigma=sigma, lr_adapt=True)
    path = [start_point.copy()]
    for generation in range(iterations):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = func(x)
            solutions.append((x, value))
        optimizer.tell(solutions)
        path.append(optimizer._mean.copy())
        if np.allclose(np.round(np.abs(optimizer._mean), 2), np.round(np.abs(func.global_min), 2)):
            reach_min = True
            opt_steps = generation
            break
    return np.array(path), reach_min, opt_steps, optimizer._mean

class StopOptimization(Exception):
    pass

# BFGS optimizer
def bfgs_optimization(func, start_point, iterations=50):
    reach_min = False
    iteration_count = 0
    path = [start_point]

    def callback(xk):
        nonlocal reach_min, iteration_count
        path.append(np.copy(xk))
        iteration_count += 1
        if np.allclose(np.round(np.abs(xk), 2), np.round(np.abs(func.global_min), 2)):
            reach_min = True
            raise StopOptimization

    try:
        result = minimize(func, start_point, method='BFGS', options={'maxiter': iterations}, callback=callback)
    except StopOptimization:
        result = None 
    
    return np.array(path), reach_min, iteration_count, path[-1]

# Quasi-Newton (L-BFGS-B) optimizer
def lbfgsb_optimization(func, start_point, iterations=50):
    reach_min = False
    iteration_count = 0
    path = [start_point]

    def callback(xk):
        nonlocal reach_min, iteration_count
        path.append(np.copy(xk))
        iteration_count += 1
        if np.allclose(np.round(np.abs(xk), 2), np.round(np.abs(func.global_min), 2)):
            reach_min = True
            raise StopOptimization

    try:
        result = minimize(func, start_point, method='L-BFGS-B', options={'maxiter': iterations}, callback=callback)
    except StopOptimization:
        result = None  
    
    return np.array(path), reach_min, iteration_count, path[-1]