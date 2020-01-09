import numpy as np
import math
from timeit import Timer
from numba.pycc import CC
from numba import jit

# Module name
cc = CC('calc_splines_numba')

@cc.export('isclose', 'boolean[:](float64[:], float64[:])')
@jit(nopython=True, cache=True)
def isclose(a, b):  # implementation for np.isclose function
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    rtol, atol = 1.e-5, 1.e-8
    x, y = np.asarray(a), np.asarray(b)
    # check if arrays are element-wise equal within a tolerance (assume that both arrays are of valid format)
    result = np.less_equal(np.abs(x-y), atol + rtol * np.abs(y))   
    return result 

@cc.export('diff', 'float64[:,:](float64[:,:], optional(uint8))')
@jit(nopython=True, cache=True)
def diff(arr, axis=1):   # implementation for np.diff function with axis parameter supported
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        col_diff_arr = np.empty((arr.shape[1], arr.shape[0]-1))
        for i in range(arr.shape[1]):   # loop through columns
            col_arr = np.copy(arr)[:, i].flatten()
            col_diff_arr[i] = np.diff(col_arr)
        return np.transpose(col_diff_arr)
    else:
        return np.diff(np.copy(arr))    # default is the last axis (axis=1)

@cc.export('calc_splines', 'UniTuple(float64[:,:],4)(float64[:,:], optional(float64[:]), optional(float64), optional(float64), optional(boolean))')
@jit(nopython=True, cache=True)
def calc_splines(path: np.ndarray,
                 el_lengths: np.ndarray = None,
                 psi_s: float = None,
                 psi_e: float = None,
                 use_dist_scaling: bool = None) -> tuple:
    """
    Author:
    Tim Stahl & Alexander Heilmeier

    Description:
    Solve for a curvature continous cubic spline between given poses

                    P_{x,y}   = a3 *t³ + a2 *t² + a1*t + a0
                    P_{x,y}'  = 3a3*t² + 2a2*t  + a1
                    P_{x,y}'' = 6a3*t² + 2a2

                    a * {x; y} = {b_x; b_y}

    Inputs:
    path:                   x and y coordinates as the basis for the spline construction (closed or unclosed).
    el_lengths:             distances between path points (closed or unclosed).
    psi_{s,e}:              orientation of the {start, end} point.
    use_dist_scaling:       bool flag to indicate if heading and curvature scaling should be performed. This is required
                            if the distances between the points in the path are not equal.

    path and el_lengths inputs can either be closed or unclosed, but must be consistent! The function detects
    automatically if the path was inserted closed.

    Outputs:
    x_coeff:                spline coefficients of the x-component.
    y_coeff:                spline coefficients of the y-component.
    M:                      LES coefficients.
    normvec_normalized:     normalized normal vectors.

    Coefficient matrices have the form a_i, b_i * t, c_i * t^2, d_i * t^3.
    """
    
    if use_dist_scaling is None:
        use_dist_scaling = True

    # check if path is closed
    if np.all(isclose(path[0], path[-1])):      # Numba 0.46.0 does not support NumPy function 'numpy.isclose'
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
        el_lengths = np.sqrt(np.sum(np.power(diff(path, axis=0), 2), axis=1))

    # if closed and use_dist_scaling active append element length in order to obtain overlapping scaling
    if use_dist_scaling and closed:
        # (Numba 0.46.0) np.hstack is supported but only for arguments passed in a tuple of arrays  
        # and the parameter also cannot be Nonetype so it's required to initialize (e.g. np.copy(el_lengths)) somehow
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

    for i in range(no_splines):
        j = i * 4

        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M

            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= math.pow(scaling[i], 2)

        else:
            M[j: j + 2, j: j + 4] = np.array([[1,  0,  0,  0],  # no curvature and heading bounds on last element
                                     [1,  1,  1,  1]])

        b_x[j: j + 2] = np.array([[path[i,     0]],      # NOTE: the bounds of the two last equations remain zero
                         [path[i + 1, 0]]])
        b_y[j: j + 2] = np.array([[path[i,     1]],
                         [path[i + 1, 1]]])
    
    # ------------------------------------------------------------------------------------------------------------------
    # SET BOUNDARY CONDITIONS FOR FIRST AND LAST POINT -----------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if not closed:
        # We want to fix heading at the start and end point (heading and curvature at start gets unbound at spline end)
        # heading and curvature boundary condition
        M[-2, 1] = 1                  # heading start
        M[-1, -4:] = [0,  1,  2,  3]  # heading end
        
        # heading start
        if el_lengths is None:
            el_length_s = 1.0
        else:
            el_length_s = el_lengths[0]

        if psi_s is not None:   # (Numba 0.46.0) built-in function 'add' does not work with argument(s) of type: none
            b_x[-2] = math.cos(psi_s + math.pi / 2) * el_length_s
            b_y[-2] = math.sin(psi_s + math.pi / 2) * el_length_s

        # heading end
        if el_lengths is None:
            el_length_e = 1.0
        else:
            el_length_e = el_lengths[-1]

        if psi_e is not None:  # (Numba 0.46.0) built-in function 'add' does not work with argument(s) of type: none
            b_x[-1] = math.cos(psi_e + math.pi / 2) * el_length_e
            b_y[-1] = math.sin(psi_e + math.pi / 2) * el_length_e

    else:
        # gradient boundary conditions (for a closed spline)
        M[-2, 1] = scaling[-1]
        M[-2, -3:] = [-1, -2, -3]
        # b_x[-2] = 0
        # b_y[-2] = 0

        # curvature boundary conditions (for a closed spline)
        M[-1, 2] = 2 * math.pow(scaling[-1], 2)
        M[-1, -2:] = [-2, -6]
        # b_x[-1] = 0
        # b_y[-1] = 0

    # ------------------------------------------------------------------------------------------------------------------
    # SOLVE ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    x_les = np.linalg.solve(M, b_x).flatten()    #  flatten/collapsed into one dimension.
    y_les = np.linalg.solve(M, b_y).flatten()

    # get coefficients of every piece into one row -> reshape
    coeffs_x = np.reshape(x_les, (no_splines, 4))
    coeffs_y = np.reshape(y_les, (no_splines, 4))

    # get normal vector (second coefficient of cubic splines is relevant for the gradient)
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)

    # normalize normal vectors
    norm_factors = 1.0 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
    norm_factors_2 = np.expand_dims(norm_factors, axis=1)  # second dimension must be inserted for next step
    normvec_normalized = norm_factors_2 * normvec

    return coeffs_x, coeffs_y, M, normvec_normalized


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    cc.compile()

path = np.ones((15,2))
el_lengths = np.ones((14))
t = Timer(lambda: calc_splines(path,el_lengths))
print("Execution time for calc_splines with numba (with compilation):",t.timeit(number=1))

print("Execution time for calc_splines with numba (after compilation):",t.timeit(number=1))

