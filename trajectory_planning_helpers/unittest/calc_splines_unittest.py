import unittest
import calc_splines_numba as cspn
import calc_splines as csp
import numpy as np
import pickle
import os, glob

class CalcSplinesTest(unittest.TestCase):
    def setUp(self):
        path = 'unittest/calc_splines_inputs/'
        self.inputs = list()
        for filename in glob.glob(os.path.join(path, '*.pkl')):
            with open(filename, 'rb') as fh:
                data = pickle.load(fh)
                self.inputs.append(data)

    def test_calc_splines(self):
        for input in self.inputs:
            path, el_lengths, psi_s, psi_e, use_dist_scaling = input['path'], input.get('el_lengths',None), input.get('psi_s',None), input.get('psi_e',None), input.get('use_dist_scaling',True)
            calc_splines_numba_result = cspn.calc_splines(path, el_lengths, psi_s, psi_e, use_dist_scaling)
            calc_splines_result = csp.calc_splines(path, el_lengths, psi_s, psi_e, use_dist_scaling)
            self.assertTrue(str(calc_splines_numba_result) == str(calc_splines_result))

    def test_diff(self):
        for input in self.inputs:
            path = input['path']
            numba_diff_result = cspn.diff(path, 0)
            numpy_diff_result = np.diff(path, axis=0)
            self.assertTrue(str(numba_diff_result) == str(numpy_diff_result))
    
    def test_isclose(self):
        for input in self.inputs:
            path = input['path']
            numba_isclose_result = cspn.isclose(path[0], path[-1])
            numpy_isclose_result = np.isclose(path[0], path[-1])
            self.assertTrue(str(numba_isclose_result) == str(numpy_isclose_result))

if __name__ == '__main__':
    unittest.main()

    