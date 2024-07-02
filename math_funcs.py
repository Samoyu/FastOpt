import numpy as np
import math
from utils.utils_funcs import check_dim, easy_bounds

class Ackley:

    b = 3
    global_min = [0.0, 0.0]

    def __init__(self, dim):

        check_dim(dim, 1)

        self.x_best = np.zeros(dim)
        self.f_best = 0
        self.bounds = easy_bounds(Ackley.b)
        self.bias = 20 + math.e
        self.pi2 = 2 * math.pi

    def __call__(self, vec):

        s1 = sum((x*x for x in vec))/vec.size
        s2 = sum((math.cos(self.pi2 * x) for x in vec))/vec.size


        return self.bias - 20*math.exp(-0.2*s1) - math.exp(s2)
    
class Rastrigin:

    b = 5.12
    global_min = [0.0, 0.0]

    def __init__(self, dim):

        check_dim(dim, 1)

        self.x_best = np.zeros(dim)
        self.f_best = 0
        self.pi2 = math.pi*2
        self.bias = 10*dim
        self.bounds = easy_bounds(Rastrigin.b)


    def __call__(self, vec):

        s = sum(( x*x - math.cos(self.pi2*x)*10 for x in vec))

        return self.bias + s
    
class Rosenbrock:
    
    b = 2.048
    global_min = [1.0, 1.0]

    def __init__(self, dim):

        check_dim(dim, 2)

        self.x_best = np.ones(dim)
        self.f_best = 0

        self.bounds = easy_bounds(Rosenbrock.b)

    
    def __call__(self, vec):

        s = sum(( 100 * (vec[i+1] - vec[i]**2) ** 2 + (vec[i] - 1)**2 for i in range(vec.size-1)))

        return s

class Fletcher:
    
    b = math.pi
    global_min = [0.0, 0.0]

    def __init__(self, dim, seed = None):

        if seed != None:
            np.random.seed(seed)

        check_dim(dim, 1)

        self.x_best = np.random.uniform(-np.pi, np.pi, dim)
        self.f_best = 0
        self.bounds = easy_bounds(Fletcher.b)

        self.a = np.random.uniform(-100, 100, (dim, dim))
        self.b = np.random.uniform(-100, 100, (dim, dim))

        self.A = np.sum(self.a * np.sin(self.x_best) + self.b * np.cos(self.x_best), axis = 0)
    
    def __call__(self, vec):

        B = np.sum(self.a * np.sin(vec) + self.b * np.cos(vec) ,axis = 0)
        #raise Exception()
        return  sum((a-b)**2 for a, b in zip(self.A, B))

class Michalewicz:
    
    global_min = [2.20, 1.57]

    def __init__(self, m = 10):

        self.x_best = None
        self.f_best = None

        self.bounds = (0, math.pi, 0, math.pi)

        self.m = m*2

    def __call__(self, vec):

        return -sum(( math.sin(x)*math.sin((i+1)*x**2/math.pi)**self.m for i, x in enumerate(vec)))