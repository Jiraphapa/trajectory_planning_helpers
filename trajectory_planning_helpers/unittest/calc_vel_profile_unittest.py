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
            ggv, kappa, el_lengths, closed, dyn_model_exp, drag_coeff, m_veh, mu, v_start, v_end, filt_window = input['ggv'], input['kappa'], input['el_lengths'], input['closed'], input.get('dyn_model_exp',1.0), input.get('drag_coeff',0.85), input.get('m_veh',1160.0), input.get('mu',None), input.get('v_start',None), input.get('v_end',None), input.get('filt_window',None)
            calc_vel_profile_numba_result = cvpn.calc_vel_profile(ggv, kappa, el_lengths, closed, dyn_model_exp, drag_coeff, m_veh, mu, v_start, v_end, 1)
            calc_vel_profile_result = cvp.calc_vel_profile(ggv, kappa, el_lengths, closed, dyn_model_exp, drag_coeff, m_veh, mu, v_start, v_end, 1)
            self.assertTrue(str(calc_vel_profile_numba_result) == str(calc_vel_profile_result))

    def test_calc_ax_poss(self):
        for input in self.inputs:
            ggv, kappa, el_lengths, closed, dyn_model_exp, drag_coeff, m_veh, mu, v_start, v_end, filt_window = input['ggv'], input['kappa'], input['el_lengths'], input['closed'], input.get('dyn_model_exp',1.0), input.get('drag_coeff',0.85), input.get('m_veh',1160.0), input.get('mu',None), input.get('v_start',None), input.get('v_end',None), input.get('filt_window',None)    
            if mu is None or kappa.size == mu.size:
                if (not closed or kappa.size == el_lengths.size) and (closed or kappa.size == el_lengths.size + 1):
                    if closed or v_start is not None:
                        if v_start is not None and v_start < 0.0:
                            v_start = 0.0

                        if v_end is not None and v_end < 0.0:
                            v_end = 0.0

                        # transform curvature kappa into corresponding radii
                        radii = np.abs(np.divide(1.0, kappa, np.full(kappa.size, np.inf)))

                        # set mu to one in case it is not set
                        if mu is None:
                            mu = np.ones(kappa.size)

                        if closed:
                            # run through all the points and check for possible lateral acceleration
                            mu_mean = np.mean(mu)
                            ay_max_global = mu_mean * np.amin(np.abs(ggv[:, 4]))    # get first lateral acceleration estimate
                            vx_profile = np.sqrt(ay_max_global * radii)             # get first velocity profile estimate
                            
                             # do it two times to improve accuracy (because of velocity-dependent accelerations)
                            for i in range(2):
                                ay_max_curr = mu * np.interp(vx_profile, ggv[:, 0], ggv[:, 4])
                                vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

                            ay_max_curr = mu * np.interp(vx_profile, ggv[:, 0], ggv[:, 4])
                            vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

                            # cut vx_profile to car's top speed
                            vx_max = ggv[-1, 0]
                            vx_profile[vx_profile > vx_max] = vx_max

                            # double arrays
                            vx_profile_double = np.concatenate((vx_profile, vx_profile), axis=0)
                            radii_double = np.concatenate((radii, radii), axis=0)
                            el_lengths_double = np.concatenate((el_lengths, el_lengths), axis=0)
                            mu_double = np.concatenate((mu, mu), axis=0)

                            # calculate acceleration profile
                            vx_max = ggv[-1, 0]
                            no_points = vx_profile.size

                            # Test on forward
                            col = np.asarray([0, 1, 2, 4])
                            ggv_mod = ggv[:, col]
                            radii_mod = radii_double
                            el_lengths_mod = el_lengths_double
                            mu_mod = mu_double
                            mode = 'accel_forw'

                            # ------------------------------------------------------------------------------------------------------------------
                            # SEARCH START POINTS FOR ACCELERATION PHASES ----------------------------------------------------------------------
                            # ------------------------------------------------------------------------------------------------------------------
                            
                            vx_diffs = np.diff(np.copy(vx_profile_double))
                            acc_inds = np.where(vx_diffs > 0.0)[0]    
                            
                            if acc_inds.size != 0:
                                # check index diffs -> we only need the first point of every acceleration phase
                                acc_inds_diffs = np.diff(acc_inds)
                                acc_inds_diffs = cvpn.insert(acc_inds_diffs, 0, 2)       # Notes: Numba 0.46.0 currently not support numpy.insert
                                acc_inds_rel = list(acc_inds[acc_inds_diffs > 1])   # starting point indices for acceleration phases
                            else:
                                # define empty list, instruct the type np.int64(x) to avoid Numba untyped list problem
                                acc_inds_rel = [np.int64(x) for x in range(0)] 

                            # ------------------------------------------------------------------------------------------------------------------
                            # CALCULATE VELOCITY PROFILE ---------------------------------------------------------------------------------------
                            # ------------------------------------------------------------------------------------------------------------------

                            acc_inds_rel = list(acc_inds_rel)

                            # while we have indices remaining in the list
                            while acc_inds_rel:
                                # set index to first list element
                                i = acc_inds_rel.pop(0)

                                # start from current index and run until either the end of the lap or a termination criterion are reached
                                while i < no_points - 1:
                                    calc_ax_poss_numba_result = cvpn.calc_ax_poss(vx_profile_double[i],
                                           radii_mod[i],
                                           ggv_mod,
                                           mu_mod[i],
                                           dyn_model_exp,
                                           drag_coeff,
                                           m_veh,
                                           mode)

                                    ax_possible_cur = cvp.calc_ax_poss(vx_profile_double[i],
                                           radii_mod[i],
                                           ggv_mod,
                                           mu_mod[i],
                                           dyn_model_exp,
                                           drag_coeff,
                                           m_veh,
                                           mode)

                                    self.assertTrue(str(calc_ax_poss_numba_result) == str(ax_possible_cur))

                                    vx_possible_next = math.sqrt(math.pow(vx_profile_double[i], 2) + 2 * ax_possible_cur * el_lengths_mod[i])

                                     # save possible next velocity if it is smaller than the current value
                                    if vx_possible_next < vx_profile_double[i + 1]:
                                        vx_profile_double[i + 1] = vx_possible_next

                                    i += 1

                                    # break current acceleration phase if next speed would be higher than the maximum vehicle velocity or if we
                                    # are at the next acceleration phase start index
                                    if vx_possible_next > vx_max or (acc_inds_rel and i >= acc_inds_rel[0]):
                                        break

                        else:
                            # run through all the points and check for possible lateral acceleration
                            mu_mean = np.mean(mu)
                            ay_max_global = mu_mean * np.amin(np.abs(ggv[:, 4]))    # get first lateral acceleration estimate
                            vx_profile = np.sqrt(ay_max_global * radii)             # get first velocity profile estimate

                            ay_max_curr = mu * np.interp(vx_profile, ggv[:, 0], ggv[:, 4])
                            vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

                            # cut vx_profile to car's top speed
                            vx_max = ggv[-1, 0]
                            vx_profile[vx_profile > vx_max] = vx_max

                            # consider v_start
                            if vx_profile[0] > v_start:
                                vx_profile[0] = v_start

                            # calculate acceleration profile
                            vx_max = ggv[-1, 0]
                            no_points = vx_profile.size

                            # Test on forward
                            col = np.asarray([0, 1, 2, 4])
                            ggv_mod = ggv[:, col]
                            radii_mod = radii
                            el_lengths_mod = el_lengths
                            mu_mod = mu
                            mode = 'accel_forw'

                            # ------------------------------------------------------------------------------------------------------------------
                            # SEARCH START POINTS FOR ACCELERATION PHASES ----------------------------------------------------------------------
                            # ------------------------------------------------------------------------------------------------------------------
    
                            vx_diffs = np.diff(np.copy(vx_profile))
                            acc_inds = np.where(vx_diffs > 0.0)[0]                  # indices of points with positive acceleration
                            if acc_inds.size != 0:
                                # check index diffs -> we only need the first point of every acceleration phase
                                acc_inds_diffs = np.diff(acc_inds)
                                acc_inds_diffs = insert(acc_inds_diffs, 0, 2)       # Notes: Numba 0.46.0 currently not support numpy.insert
                                acc_inds_rel = list(acc_inds[acc_inds_diffs > 1])   # starting point indices for acceleration phases
                            else:
                                # define empty list, instruct the type np.int64(x) to avoid Numba untyped list problem
                                acc_inds_rel = [np.int64(x) for x in range(0)]      # if vmax is low and can be driven all the time
                                
                            # ------------------------------------------------------------------------------------------------------------------
                            # CALCULATE VELOCITY PROFILE ---------------------------------------------------------------------------------------
                            # ------------------------------------------------------------------------------------------------------------------

                            # cast np.array as a list
                            acc_inds_rel = list(acc_inds_rel)

                            # while we have indices remaining in the list
                            while acc_inds_rel:
                                # set index to first list element
                                i = acc_inds_rel.pop(0)

                                # start from current index and run until either the end of the lap or a termination criterion are reached
                                while i < no_points - 1:

                                    calc_ax_poss_numba_result = cvpn.calc_ax_poss(vx_profile[i],
                                                                radii_mod[i],
                                                                ggv_mod,
                                                                mu_mod[i],
                                                                dyn_model_exp,
                                                                drag_coeff,
                                                                m_veh,
                                                                mode)

                                    ax_possible_cur = cvp.calc_ax_poss(vx_profile[i],
                                                                radii_mod[i],
                                                                ggv_mod,
                                                                mu_mod[i],
                                                                dyn_model_exp,
                                                                drag_coeff,
                                                                m_veh,
                                                                mode)

                                    self.assertTrue(str(calc_ax_poss_numba_result) == str(ax_possible_cur))

                                    vx_possible_next = math.sqrt(math.pow(vx_profile[i], 2) + 2 * ax_possible_cur * el_lengths_mod[i])

                                    # save possible next velocity if it is smaller than the current value
                                    if vx_possible_next < vx_profile[i + 1]:
                                        vx_profile[i + 1] = vx_possible_next

                                    i += 1

                                    # break current acceleration phase if next speed would be higher than the maximum vehicle velocity or if we
                                    # are at the next acceleration phase start index
                                    if vx_possible_next > vx_max or (acc_inds_rel and i >= acc_inds_rel[0]):
                                        break



                            
         
    def test_flipud(self):
        pass

    def test_insert(self):
        pass

if __name__ == '__main__':
    unittest.main()