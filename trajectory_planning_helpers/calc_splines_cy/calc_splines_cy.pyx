import numpy as np
import math 
from timeit import Timer

# import special compile-time information
cimport numpy as np
cimport cython

print("calc_splines with cython...")
# DTYPE is assigned to the NumPy runtime type info object.
DTYPE = np.float64
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float64_t DTYPE_t


def wrap_calc_splines_cy(path: np.ndarray,
                 el_lengths: np.ndarray = None,
                 psi_s: float = None,
                 psi_e: float = None,
                 use_dist_scaling: bool = True) -> tuple:

    cdef float *cdef_psi_s, *cdef_psi_e
    if psi_s is None:
        cdef_psi_s = NULL
    if psi_e is None:
        cdef_psi_e = NULL
    #return calc_splines_cy(path, el_lengths, psi_s, psi_e, use_dist_scaling)
    print(calc_splines_cy(path, el_lengths, cdef_psi_s, cdef_psi_e, use_dist_scaling))

cdef calc_splines_cy(np.ndarray path,
                 np.ndarray el_lengths,
                 float *psi_s,  
                 float *psi_e,
                 bint use_dist_scaling):
     
     # Typed memoryviews both use and support the buffer protocol, so they do not copy memory unnecessarily. 
    cdef double[:,:] path_mv = memoryview(path)  
    cdef double[:,:] el_lengths_mv = memoryview(el_lengths) if el_lengths is not None else None
    
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
     # TODO:
    if use_dist_scaling and el_lengths_mv is None:
        el_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))

    # if closed and use_dist_scaling active append element length in order to obtain overlapping scaling
    if use_dist_scaling and closed:
        el_lengths = np.hstack((el_lengths, el_lengths[0]))


    return 1, 2, 3, 4


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass

path = np.ones((15,2))
t = Timer(lambda: wrap_calc_splines_cy(path))
print("Execution time for calc_splines:",t.timeit(number=1))
    
    
    
    



