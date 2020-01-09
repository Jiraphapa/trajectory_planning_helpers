import unittest
import conv_filt_numba 
import conv_filt 
import numpy as np
import pickle
import os, glob

class ConvFiltNumbaTest(unittest.TestCase):
    def setUp(self):
        path = 'unittest/calc_vel_profile_inputs/'
        self.inputs = list()
        for filename in glob.glob(os.path.join(path, '*.pkl')):
            with open(filename, 'rb') as fh:
                data = pickle.load(fh)
                self.inputs.append(data)

    def test_conv_filt(self):
        for input in self.inputs:
            kappa, el_lengths, mu, closed = input['kappa'], input['el_lengths'], input.get('mu',None), input['closed']
            radii = np.abs(np.divide(1.0, kappa, np.full(kappa.size, np.inf)))
            if mu is None:
                mu = np.ones(kappa.size)

            for item in [kappa, el_lengths, mu, radii]:
                for filt_window in range(1,1000,2):  # filter window size for moving average filter (must be odd)
                    conv_filt_numba_result = conv_filt_numba.conv_filt(item, filt_window, closed)
                    conv_filt_result = conv_filt.conv_filt(item, filt_window, closed)
                    self.assertTrue(str(conv_filt_numba_result) == str(conv_filt_result))

if __name__ == '__main__':
    unittest.main()
