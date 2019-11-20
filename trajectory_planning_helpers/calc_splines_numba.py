import numpy as np
import math
from timeit import Timer
from numba import jit, vectorize, float64

@jit(nopython=True, cache=True)
def isclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    x, y = np.asarray(a), np.asarray(b)
    # check if arrays are element-wise equal within a tolerance (assume that both arrays are of valid format)
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

    # check inputs
    if not closed and (psi_s is None or psi_e is None):
        raise ValueError("Headings must be provided for unclosed spline calculation!")

    if el_lengths is not None and path.shape[0] != el_lengths.size + 1:
        raise ValueError("el_lengths input must be one element smaller than path input!")

    # if distances between path coordinates are not provided but required, calculate euclidean distances as el_lengths
    if use_dist_scaling and el_lengths is None:
        el_lengths = np.sqrt(np.sum(np.power(np.diff(path, 0), 2), axis=1))

    # if closed and use_dist_scaling active append element length in order to obtain overlapping scaling
    if use_dist_scaling and closed:
        el_lengths = np.hstack((np.copy(el_lengths), np.array([el_lengths[0]])))

        

    # get number of splines
    no_splines = path.shape[0] - 1

    # calculate scaling factors between every pair of splines
    if use_dist_scaling:
        scaling = el_lengths[:-1] / el_lengths[1:]
    else:
        scaling = np.ones(no_splines - 1)

    # ------------------------------------------------------------------------------------------------------------------
    # DEFINE LINEAR EQUATION SYSTEM ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # M_{x,y} * a_{x,y} = b_{x,y}) with a_{x,y} being the desired spline param
    # *4 because of 4 parameters in cubic spline
    M = np.zeros((no_splines * 4, no_splines * 4))
    b_x = np.zeros((no_splines * 4, 1))
    b_y = np.zeros((no_splines * 4, 1))

    # create template for M array entries
    template_M = np.array(                          # current time step           | next time step          | bounds
                [[1,  0,  0,  0,  0,  0,  0,  0],   # a_0i                                                  = {x,y}_i
                 [1,  1,  1,  1,  0,  0,  0,  0],   # a_0i + a_1i +  a_2i +  a_3i                           = {x,y}_i+1
                 [0,  1,  2,  3,  0, -1,  0,  0],   # _      a_1i + 2a_2i + 3a_3i      - a_1i+1             = 0
                 [0,  0,  2,  6,  0,  0, -2,  0]])  # _             2a_2i + 6a_3i               - 2a_2i+1   = 0

   
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
