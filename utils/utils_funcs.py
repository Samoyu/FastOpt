import numpy as np
from OppOpPopInit import OppositionOperators

def easy_bounds(bound):
    return (-bound, bound, -bound, bound)

def check_dim(dim, min = 1):
    assert (type(dim) == int and dim >=min), f"Dimension should be int and not less than {min} for this function (got {dim})"

def get_good_arrow_place(optimum, bounds):
    opt = np.array(optimum)
    minimums = np.array([bounds[0], bounds[2]])
    maximums = np.array([bounds[1], bounds[3]])

    t1 = OppositionOperators.Continual.over(minimums, maximums)(opt)
    t2 = OppositionOperators.Continual.quasi_reflect(minimums, maximums)(t1)

    return tuple(t2)