import numpy as np
from timeit import Timer

# cython: import special compile-time information
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t

#TODO: move to helper function
cdef extern from "math.h":
    const double pi "M_PI"  # as in Python's math module
    double pow(double base, double power)
    double sin(double x)
    double cos(double x)
    double sqrt(double x, double power)
    

def wrap_calc_splines_cy(path: np.ndarray,
                 el_lengths: np.ndarray = None,
                 psi_s: float = None,
                 psi_e: float = None,
                 use_dist_scaling: bool = True) -> tuple:

    cdef float *cdef_psi_s, *cdef_psi_e
    if psi_s is None: cdef_psi_s = NULL
    if psi_e is None: cdef_psi_e = NULL
    #return calc_splines_cy(path, el_lengths, psi_s, psi_e, use_dist_scaling)
    print(calc_splines_cy(path, el_lengths, cdef_psi_s, cdef_psi_e, use_dist_scaling))

cdef tuple calc_splines_cy(np.ndarray[DTYPE_t, ndim = 2] path,
                 np.ndarray[DTYPE_t, ndim = 1] el_lengths,
                 float *psi_s,  
                 float *psi_e,
                 bint use_dist_scaling):
     
     # Typed memoryviews both use and support the buffer protocol, so they do not copy memory unnecessarily. 
    cdef double[:,:] path_mv = memoryview(path)  
    cdef double[:] el_lengths_mv = memoryview(el_lengths) if el_lengths is not None else None
    
      # check if path is closed
    cdef bint closed        
    if np.all(np.isclose(path[0], path[-1])):
        closed = True
    else:
        closed = False

     # check inputs
    if not closed and (psi_s is NULL or psi_e is NULL):
        raise ValueError("Headings must be provided for unclosed spline calculation!")

    if el_lengths_mv is not None and path_mv.shape[0] != el_lengths_mv.size + 1:
        raise ValueError("el_lengths input must be one element smaller than path input!")

     # if distances between path coordinates are not provided but required, calculate euclidean distances as el_lengths
    if use_dist_scaling and el_lengths_mv is None:
        el_lengths_mv = memoryview(np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1)))

    # if closed and use_dist_scaling active append element length in order to obtain overlapping scaling
    if use_dist_scaling and closed:
        el_lengths_mv = memoryview(np.hstack((el_lengths_mv, el_lengths_mv[0])))

     # get number of splines
    cdef int no_splines = path_mv.shape[0] - 1

    # calculate scaling factors between every pair of splines
    cdef double[:] scaling_mv
    if use_dist_scaling:
        scaling_mv = memoryview(np.asarray(el_lengths_mv[:-1]) / np.asarray(el_lengths_mv[1:]))
    else:
        scaling_mv = memoryview(np.ones(no_splines - 1))

    # ------------------------------------------------------------------------------------------------------------------
    # DEFINE LINEAR EQUATION SYSTEM ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # M_{x,y} * a_{x,y} = b_{x,y}) with a_{x,y} being the desired spline param
    # *4 because of 4 parameters in cubic spline
    cdef double[:, :] M_mv, b_x_mv, b_y_mv
    M_mv = memoryview(np.zeros((no_splines * 4, no_splines * 4)))
    b_x_mv = memoryview(np.zeros((no_splines * 4, 1)))
    b_y_mv = memoryview(np.zeros((no_splines * 4, 1)))

    # create template for M array entries
    cdef double[:,:] template_M_mv
    template_M_mv = memoryview(np.array(                          # current time step           | next time step          | bounds
                [[1,  0,  0,  0,  0,  0,  0,  0],   # a_0i                                                  = {x,y}_i
                 [1,  1,  1,  1,  0,  0,  0,  0],   # a_0i + a_1i +  a_2i +  a_3i                           = {x,y}_i+1
                 [0,  1,  2,  3,  0, -1,  0,  0],   # _      a_1i + 2a_2i + 3a_3i      - a_1i+1             = 0
                 [0,  0,  2,  6,  0,  0, -2,  0]]).astype('double'))  # _             2a_2i + 6a_3i               - 2a_2i+1   = 0
    
    cdef double[:,:] no_curvature_mv = memoryview(np.array(
                [[1,  0,  0,  0],                   # no curvature and heading bounds on last element
                 [1,  1,  1,  1]]).astype('double'))
    
    cdef double[:,:] zero_bound_b_x_mv = memoryview(np.array( 
                [[path_mv[i,     0]],      # NOTE: the bounds of the two last equations remain zero
                 [path_mv[i + 1, 0]]]).astype('double'))
    
    cdef double[:,:] zero_bound_b_y_mv = memoryview(np.array( 
                [[path_mv[i,     1]],      # NOTE: the bounds of the two last equations remain zero
                 [path_mv[i + 1, 1]]]).astype('double'))

    cdef double[:] heading_end_mv, value
    cdef double[:] x_les_mv, y_les_mv
    cdef double[:] coeffs_x_mv, coeffs_y_mv
    cdef double[:,:] normvec_mv, norm_factors_mv_2, normvec_normalized_mv
    cdef double[:] norm_factors_mv


    cdef int i,j
    cdef double el_length_s, el_length_e


    for i in range(no_splines):
         j = i * 4

         if i < no_splines - 1:
            M_mv[j: j + 4, j: j + 8] = template_M_mv

            M_mv[j + 2, j + 5] *= scaling_mv[i]
            M_mv[j + 3, j + 6] *= pow(scaling_mv[i], 2)

         else:
            M_mv[j: j + 2, j: j + 4] = no_curvature_mv    # no curvature and heading bounds on last element

         b_x_mv[j: j + 2] = zero_bound_b_x_mv
         b_y_mv[j: j + 2] = zero_bound_b_y_mv
                   
    # # ------------------------------------------------------------------------------------------------------------------
    # # SET BOUNDARY CONDITIONS FOR FIRST AND LAST POINT -----------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------

    if not closed:
        # We want to fix heading at the start and end point (heading and curvature at start gets unbound at spline end)
        # heading and curvature boundary condition
        M_mv[-2, 1] = 1                  # heading start
        heading_end_mv = memoryview((np.array([0,  1,  2,  3]).astype('double')))
        M_mv[-1, -4:] = heading_end_mv  # heading end

        # heading start
        if el_lengths_mv is None:
            el_length_s = 1.0
        else:
            el_length_s = el_lengths_mv[0]

        b_x_mv[-2] = cos(psi_s[0] + pi / 2) * el_length_s
        b_y_mv[-2] = sin(psi_s[0] + pi / 2) * el_length_s

        # heading end
        if el_lengths_mv is None:
            el_length_e = 1.0
        else:
            el_length_e = el_lengths_mv[-1]

        b_x_mv[-1] = cos(psi_e[0] + pi / 2) * el_length_e
        b_y_mv[-1] = sin(psi_e[0] + pi / 2) * el_length_e

    else:
        # gradient boundary conditions (for a closed spline)
        M_mv[-2, 1] = scaling_mv[-1]
        value = memoryview((np.array([-1, -2, -3]).astype('double')))
        M_mv[-2, -3:] = value
        # b_x[-2] = 0
        # b_y[-2] = 0

        # curvature boundary conditions (for a closed spline)
        M_mv[-1, 2] = 2 * pow(scaling_mv[-1], 2)
        value = memoryview((np.array([-2, -6]).astype('double')))
        M_mv[-1, -2:] = value
        # b_x[-1] = 0
        # b_y[-1] = 0
  
    # ------------------------------------------------------------------------------------------------------------------
    # SOLVE ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    
    x_les_mv = memoryview(np.squeeze(np.linalg.solve(M_mv, b_x_mv)))  # squeeze removes single-dimensional entries
    y_les_mv = memoryview(np.squeeze(np.linalg.solve(M_mv, b_y_mv)))
 
    # get coefficients of every piece into one row -> reshape
    cdef double[:,:] coeffs_x, coeffs_y
    coeffs_x = memoryview(np.reshape(x_les_mv, (no_splines, 4)))
    coeffs_y = memoryview(np.reshape(y_les_mv, (no_splines, 4)))

    # get normal vector (second coefficient of cubic splines is relevant for the gradient)
    normvec_mv = memoryview(np.stack((coeffs_y[:, 1], -np.asarray(coeffs_x[:, 1])), axis=1))

    # normalize normal vectors
    norm_factors_mv = memoryview(1.0 / np.sqrt(np.sum(np.power(normvec_mv, 2), axis=1)))
    norm_factors_mv_2 = memoryview(np.expand_dims(norm_factors_mv, axis=1))  # second dimension must be inserted for next step
    normvec_normalized_mv = memoryview(np.asarray(norm_factors_mv_2) * np.asarray(normvec_mv))
   
    return coeffs_x, coeffs_y, M_mv, normvec_normalized_mv


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass

path = np.ones((15,2))
t = Timer(lambda: wrap_calc_splines_cy(path))
print("Execution time for wrap_calc_splines_cy:",t.timeit(number=1))

#import cProfile
#cProfile.run('main()', sort='time')


    
    
    
    



