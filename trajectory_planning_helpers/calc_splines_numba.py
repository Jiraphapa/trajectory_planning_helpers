import numpy as np
import math
from timeit import Timer
from numba import jit, vectorize, float64

@jit(nopython=True, cache=True)
def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    x, y = np.asarray(a), np.asarray(b)
    # check if arrays are element-wise equal within a tolerance.
    result = np.less_equal(np.abs(x-y), atol + rtol * np.abs(y))   
    return result

@jit(nopython=True, cache=True)
def calc_splines(path: np.ndarray,
                 el_lengths: np.ndarray = None,
                 psi_s: float = None,
                 psi_e: float = None,
                 use_dist_scaling: bool = True) -> tuple:

    # check if path is closed
    if np.all(isclose(path[0], path[-1])):
        closed = True
    else:
        closed = False

   
    return 1,2,3,4


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass


path = np.ones((15,2))
t = Timer(lambda: calc_splines(path))
print("Execution time for calc_splines with numba (with compilation):",t.timeit(number=1))

path = np.ones((15,2))
t = Timer(lambda: calc_splines(path))
print("Execution time for calc_splines with numba (after compilation):",t.timeit(number=1))
