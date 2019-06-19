import numpy as np
import trajectory_planning_helpers.calc_spline_lengths


def interp_splines(coeffs_x: np.ndarray,
                   coeffs_y: np.ndarray,
                   spline_lengths: np.ndarray = None,
                   incl_last_point: bool = False,
                   stepsize_approx: float = None,
                   stepnum_fixed: list = None) -> tuple:
    """
    Created by:
    Alexander Heilmeier & Tim Stahl

    Documentation:
    Interpolate points on one or more splines with third order. The last point (i.e. t = 0)
    can be included if option is set accordingly. The algorithm keeps stepsize_approx as good as possible.
    ws_track can be inserted optionally and should contain [w_tr_right, w_tr_left].

    Inputs:
    coeffs_x:           coefficient matrix of the x splines with size no_splines x 4.
    coeffs_y:           coefficient matrix of the y splines with size no_splines x 4.
    spline_lengths:     array containing the lengths of the inserted splines with size no_splines x 1.
    incl_last_point:    flag to set if last point should be kept or removed before return.
    stepsize_approx:    desired stepsize of the points after interpolation.                          \\ Provide only one
    stepnum_fixed:      return a fixed number of coordinates per spline, list of length no_splines.  \\ of these two!
    """

    # ------------------------------------------------------------------------------------------------------------------
    # INPUT CHECKS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check if coeffs_x and coeffs_y have exactly two dimensions and raise error otherwise
    if not (coeffs_x.ndim == 2 and coeffs_y.ndim == 2):
        raise ValueError("Coefficient matrices do not have two dimensions!")

    # check if step size specification is valid
    if (stepsize_approx is None and stepnum_fixed is None) \
            or (stepsize_approx is not None and stepnum_fixed is not None):
        raise ValueError("Provide one of 'stepsize_approx' and 'stepnum_fixed' and set the other to 'None'!")

    if stepnum_fixed is not None and len(stepnum_fixed) != coeffs_x.shape[0]:
        raise ValueError("The provided list 'stepnum_fixed' must hold an entry for every spline!")

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE NUMBER OF INTERPOLATION POINTS AND ACCORDING DISTANCES -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if stepsize_approx is not None:
        # get the total distance up to the end of every spline (i.e. cumulated distances)
        if spline_lengths is None:
            spline_lengths = trajectory_planning_helpers.calc_spline_lengths.calc_spline_lengths(coeffs_x=coeffs_x,
                                                                                                 coeffs_y=coeffs_y,
                                                                                                 quickndirty=False)

        dists_cum = np.cumsum(spline_lengths)

        # calculate number of interpolation points and distances (+1 because last point is included at first)
        no_interp_points = (np.ceil(dists_cum[-1] / stepsize_approx)).astype(int) + 1
        dists_interp = np.linspace(0.0, dists_cum[-1], no_interp_points)

    else:
        # get total number of points to be sampled (subtract overlapping points)
        no_interp_points = sum(stepnum_fixed) - (len(stepnum_fixed) - 1)
        dists_interp = None

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE INTERMEDIATE STEPS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create arrays to save the values
    path_interp = np.zeros((no_interp_points, 2))           # raceline coords (x, y) array
    spline_inds = np.zeros(no_interp_points, dtype=int)     # save the spline index to which a point belongs
    t_values = np.zeros(no_interp_points)                   # save t values

    j = 0

    if stepsize_approx is not None:

        # --------------------------------------------------------------------------------------------------------------
        # APPROX. EQUAL STEP SIZE ALONG PATH OF ADJACENT SPLINES -------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # loop through all the elements and create steps with stepsize_approx
        for i in range(no_interp_points - 1):
            # find the spline that hosts the current interpolation point
            j = np.argmax(dists_interp[i] < dists_cum)
            spline_inds[i] = j

            # get spline t value depending on the progress within the current element
            if j > 0:
                t_values[i] = (dists_interp[i] - dists_cum[j - 1]) / spline_lengths[j]
            else:
                if spline_lengths.ndim == 0:
                    t_values[i] = dists_interp[i] / spline_lengths
                else:
                    t_values[i] = dists_interp[i] / spline_lengths[0]

            # calculate coords
            path_interp[i, 0] = coeffs_x[j, 0]\
                                    + coeffs_x[j, 1] * t_values[i]\
                                    + coeffs_x[j, 2] * np.power(t_values[i], 2) \
                                    + coeffs_x[j, 3] * np.power(t_values[i], 3)

            path_interp[i, 1] = coeffs_y[j, 0]\
                                    + coeffs_y[j, 1] * t_values[i]\
                                    + coeffs_y[j, 2] * np.power(t_values[i], 2) \
                                    + coeffs_y[j, 3] * np.power(t_values[i], 3)

    else:

        # --------------------------------------------------------------------------------------------------------------
        # FIXED STEP SIZE FOR EVERY SPLINE SEGMENT ---------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        for i in range(len(stepnum_fixed)):
            # skip last point except for last segment
            if i < len(stepnum_fixed) - 1:
                t_values[j:(j + stepnum_fixed[i] - 1)] = np.linspace(0, 1, stepnum_fixed[i])[:-1]
                spline_inds[j:(j + stepnum_fixed[i] - 1)] = i
                j += stepnum_fixed[i] - 1
            else:
                t_values[j:(j + stepnum_fixed[i])] = np.linspace(0, 1, stepnum_fixed[i])
                spline_inds[j:(j + stepnum_fixed[i])] = i
                j += stepnum_fixed[i]

        t_set = np.column_stack(([1] * len(t_values), t_values, np.power(t_values, 2), np.power(t_values, 3)))

        # remove overlapping samples
        n_samples = np.array(stepnum_fixed)
        n_samples[:-1] -= 1

        path_interp[:, 0] = np.sum(np.multiply(np.repeat(coeffs_x, n_samples, axis=0), t_set), axis=1)
        path_interp[:, 1] = np.sum(np.multiply(np.repeat(coeffs_y, n_samples, axis=0), t_set), axis=1)

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE LAST POINT IF REQUIRED (t = 1.0) -----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if incl_last_point:
        t_values[-1] = 1.0

        path_interp[-1, 0] = coeffs_x[-1, 0] + coeffs_x[-1, 1] + coeffs_x[-1, 2] + coeffs_x[-1, 3]
        path_interp[-1, 1] = coeffs_y[-1, 0] + coeffs_y[-1, 1] + coeffs_y[-1, 2] + coeffs_y[-1, 3]

    else:
        path_interp = path_interp[:-1]
        spline_inds = spline_inds[:-1]
        t_values = t_values[:-1]

        if dists_interp is not None:
            dists_interp = dists_interp[:-1]

    # NOTE: dists_interp is None, when using a fixed step size
    return path_interp, spline_inds, t_values, dists_interp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass