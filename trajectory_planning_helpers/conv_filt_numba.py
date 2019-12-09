import numpy as np
from numba import jit

@jit(nopython=True)
def conv_filt(signal: np.ndarray,
              filt_window: int,
              closed: bool) -> np.ndarray:
    """
    Author:
    Alexander Heilmeier

    Modified by:
    Tim Stahl

    Description:
    Filter a given temporal signal using a convolution (moving average) filter.

    Inputs:
    signal:         temporal signal that should be filtered (always unclosed).
    filt_window:    filter window size for moving average filter (must be odd).
    closed:         flag showing if the signal can be considered as closable, e.g. for velocity profiles.

    signal input is always unclosed!

    Outputs:
    signal_filt:    filtered input signal (always unclosed).

    len(signal) = len(signal_filt)
    """

    # check if window width is odd
    if not filt_window % 2 == 1:
        raise ValueError("Window width of moving average filter must be odd!")

    # calculate half window width - 1
    w_window_half = int((filt_window - 1) / 2)

    # apply filter
    if closed:
        # temporarily add points in front of and behind signal
        signal_tmp = np.concatenate((signal[-w_window_half:], signal, signal[:w_window_half]), axis=0)

        # apply convolution filter used as a moving average filter and remove temporary points
        signal_filt = np.convolve(signal_tmp,
                                  np.ones(filt_window) / float(filt_window))[w_window_half:-w_window_half]
        n1 = signal_tmp.shape[0];
        n2 = filt_window;
        if n1 > n2:
            temp = n1
            n1 = n2
            n2 = temp
        length = n1
        n = n2
        n_left = n//2
        n_right = n - n_left - 1;
        signal_filt = signal_filt[n_left:-n_right]

    else:
        # implementation 1: include boundaries during filtering
        # no_points = signal.size
        # signal_filt = np.zeros(no_points)
        #
        # for i in range(no_points):
        #     if i < w_window_half:
        #         signal_filt[i] = np.average(signal[:i + w_window_half + 1])
        #
        #     elif i < no_points - w_window_half:
        #         signal_filt[i] = np.average(signal[i - w_window_half:i + w_window_half + 1])
        #
        #     else:
        #         signal_filt[i] = np.average(signal[i - w_window_half:])

        # implementation 2: start filtering at w_window_half and stop at -w_window_half
        signal_filt = np.copy(signal)
        signal_filt_tmp = np.convolve(signal,
                                                                np.ones(filt_window) / float(filt_window),
                                                                )[w_window_half:-w_window_half]
        n1 = signal.shape[0];
        n2 = filt_window;
        if n1 > n2:
            temp = n1
            n1 = n2
            n2 = temp
        length = n1
        n = n2
        n_left = n//2
        n_right = n - n_left - 1;

        signal_filt[w_window_half:-w_window_half] = signal_filt_tmp[n_left:-n_right]

    return signal_filt


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
