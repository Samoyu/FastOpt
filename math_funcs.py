import numpy as np
import math
from utils_funcs import check_dim, easy_bounds

class Ackley:

    b = 3

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