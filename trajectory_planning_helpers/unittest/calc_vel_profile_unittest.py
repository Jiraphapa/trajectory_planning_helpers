import unittest
import calc_vel_profile_numba as cvpn
import calc_vel_profile as cvp
import numpy as np
import pickle
import os, glob
import math

class CalcVelProfileTest(unittest.TestCase):
    def setUp(self):
        path = 'unittest/calc_vel_profile_inputs/'
        self.inputs = list()
        for filename in glob.glob(os.path.join(path, '*.pkl')):
            with open(filename, 'rb') as fh:
                data = pickle.load(fh)
                self.inputs.append(data)
    
    def test_calc_vel_profile(self):
        for input in self.inputs:
            ax_max_machines, kappa, el_lengths, closed, drag_coeff, m_veh, ggv, loc_gg, v_max, dyn_model_exp, mu, v_start, v_end, filt_window = input.get('ax_max_machines', None), input['kappa'], input['el_lengths'], input['closed'], input.get('drag_coeff',0.85), input.get('m_veh',1160.0), input.get('ggv', None), input.get('loc_gg', None), input.get('v_max', None), input.get('dyn_model_exp',None), input.get('mu',None), input.get('v_start',None), input.get('v_end',None), input.get('filt_window',None)
            if ggv is not None:
                ggv = ggv[:,[0,1,2]] 
            if ax_max_machines is None:
                ax_max_machines = ggv[:,[0,1]]

            calc_vel_profile_numba_result = cvpn.calc_vel_profile(ax_max_machines, kappa, el_lengths, closed, drag_coeff, m_veh, ggv, loc_gg, v_max, dyn_model_exp, mu, v_start, v_end, filt_window)
            calc_vel_profile_result = cvp.calc_vel_profile(ax_max_machines, kappa, el_lengths, closed, drag_coeff, m_veh, ggv, loc_gg, v_max, dyn_model_exp, mu, v_start, v_end, filt_window)
            self.assertTrue(str(calc_vel_profile_numba_result) == str(calc_vel_profile_result))

    def test_flipud(self):
        for input in self.inputs:
            kappa, el_lengths, mu = input['kappa'], input['el_lengths'], input.get('mu',None)
            radii = np.abs(np.divide(1.0, kappa, np.full(kappa.size, np.inf)))
            if mu is None:
                mu = np.ones(kappa.size)

            for item in [kappa, el_lengths, mu, radii]:
                flipud_numba_result = cvpn.flipud(item)
                flipud_numpy_result = np.flipud(item)
                self.assertTrue(str(flipud_numba_result) == str(flipud_numpy_result))
                
    def test_insert(self):
        for input in self.inputs:
            kappa, el_lengths, mu = input['kappa'], input['el_lengths'], input.get('mu',None)
            radii = np.abs(np.divide(1.0, kappa, np.full(kappa.size, np.inf)))
            if mu is None:
                mu = np.ones(kappa.size)

            for item in [kappa, el_lengths, mu, radii]:
                for index in (0, len(item)):
                    for value in np.arange(-5, 5, 0.05):
                        insert_numba_result = cvpn.insert(item,index,value)
                        insert_numpy_result = np.insert(item,index,value)
                        self.assertTrue(str(insert_numba_result) == str(insert_numpy_result))


if __name__ == '__main__':
    unittest.main()