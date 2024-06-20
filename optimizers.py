import numpy as np
from cmaes import CMA

# SGD optimizer
def sgd_optimization(func, start_point, lr=0.05, iterations=100):
    path = [start_point]
    x = start_point.copy()
    for _ in range(iterations):
        grad = np.array([func(x + np.eye(1, len(x), i)[0] * 1e-8) - func(x) for i in range(len(x))]) / 1e-8
        x = x - lr * grad
        path.append(x.copy())
    return np.array(path)

# Adam optimizer
def adam_optimization(func, start_point, lr=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8, iterations=100):
    path = [start_point]
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
    return np.array(path)

# CMA-ES
def cmaes_optimization(func, start_point, sigma=1.3, iterations=50):
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
    return np.array(path)

# LRA-CMA
def lra_cma_optimization(func, start_point, sigma=1.3, iterations=50):
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
    return np.array(path)