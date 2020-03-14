""" RPE configuration for X(pi/2), Y(pi/2) single qubit model """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from . import RPEconfig as _RPEconfig

rpeconfig_gxpi2_gypi2_updn_dict = {}
rpeconfig_gxpi2_gypi2_updn_dict['fixed_axis_gate_label'] = 'Gx'
rpeconfig_gxpi2_gypi2_updn_dict['fixed_axis_label'] = 'X'
rpeconfig_gxpi2_gypi2_updn_dict['fixed_axis_target'] = [0, 1, 0, 0]
rpeconfig_gxpi2_gypi2_updn_dict['loose_axis_gate_label'] = 'Gy'
rpeconfig_gxpi2_gypi2_updn_dict['loose_axis_label'] = 'Y'
rpeconfig_gxpi2_gypi2_updn_dict['loose_axis_target'] = [0, 0, 1, 0]
rpeconfig_gxpi2_gypi2_updn_dict['auxiliary_axis_gate_label'] = 'Gz'
rpeconfig_gxpi2_gypi2_updn_dict['auxiliary_axis_label'] = 'Z'
rpeconfig_gxpi2_gypi2_updn_dict['rhoExpressions'] = ["0"]
rpeconfig_gxpi2_gypi2_updn_dict['ELabels'] = ["0", "1"]
rpeconfig_gxpi2_gypi2_updn_dict['EExpressions'] = ["0", "1"]
#rpeconfig_gxpi2_gypi2_updn_dict['spamLabelDict'] = {'plus': (0,0), 'minus': (0,-1) }
rpeconfig_gxpi2_gypi2_updn_dict['spamLabelDict'] = {'plus': ('rho0', 'E0'), 'minus': ('rho0', 'remainder')}
rpeconfig_gxpi2_gypi2_updn_dict['dn_labels'] = ['1']
rpeconfig_gxpi2_gypi2_updn_dict['up_labels'] = ['0']
rpeconfig_gxpi2_gypi2_updn_dict['alpha'] = _np.pi / 2
rpeconfig_gxpi2_gypi2_updn_dict['epsilon'] = _np.pi / 2
rpeconfig_gxpi2_gypi2_updn_dict['theta'] = 0  # This should always be 0.
rpeconfig_gxpi2_gypi2_updn_dict['new_epsilon_func'] = lambda epsilon: (epsilon / (_np.pi / 2)) - 1
rpeconfig_gxpi2_gypi2_updn_dict['alpha_hat_func'] = lambda xhat, yhat, Nx, Ny: _np.arctan2(
    (xhat - Nx / 2.) / Nx, -(yhat - Ny / 2.) / Ny)
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_hat_func'] = lambda xhat, yhat, Nx, Ny: _np.arctan2(
    (xhat - Nx / 2.) / Nx, -(yhat - Ny / 2.) / Ny)
rpeconfig_gxpi2_gypi2_updn_dict['Phi_hat_func'] = lambda xhat, yhat, Nx, Ny: _np.arctan2(
    (xhat - Nx / 2.) / Nx, -(yhat - Ny / 2.) / Ny)
rpeconfig_gxpi2_gypi2_updn_dict['alpha_cos_prep_tuple'] = ()
rpeconfig_gxpi2_gypi2_updn_dict['alpha_cos_prep_str'] = ''
rpeconfig_gxpi2_gypi2_updn_dict['alpha_cos_germ_tuple'] = ('Gx',)
rpeconfig_gxpi2_gypi2_updn_dict['alpha_cos_germ_str'] = 'Gx'
rpeconfig_gxpi2_gypi2_updn_dict['alpha_cos_meas_tuple'] = ()
rpeconfig_gxpi2_gypi2_updn_dict['alpha_cos_meas_str'] = ''
rpeconfig_gxpi2_gypi2_updn_dict['alpha_sin_prep_tuple'] = ()
rpeconfig_gxpi2_gypi2_updn_dict['alpha_sin_prep_str'] = ''
rpeconfig_gxpi2_gypi2_updn_dict['alpha_sin_germ_tuple'] = ('Gx',)
rpeconfig_gxpi2_gypi2_updn_dict['alpha_sin_germ_str'] = 'Gx'
rpeconfig_gxpi2_gypi2_updn_dict['alpha_sin_meas_tuple'] = ('Gx',)
rpeconfig_gxpi2_gypi2_updn_dict['alpha_sin_meas_str'] = 'Gx'
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_cos_prep_tuple'] = ()
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_cos_prep_str'] = ''
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_cos_germ_tuple'] = ('Gy',)
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_cos_germ_str'] = 'Gy'
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_cos_meas_tuple'] = ()
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_cos_meas_str'] = ''
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_sin_prep_tuple'] = ()
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_sin_prep_str'] = ''
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_sin_germ_tuple'] = ('Gy',)
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_sin_germ_str'] = 'Gy'
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_sin_meas_tuple'] = ('Gy',)
rpeconfig_gxpi2_gypi2_updn_dict['epsilon_sin_meas_str'] = 'Gy'
rpeconfig_gxpi2_gypi2_updn_dict['theta_cos_prep_tuple'] = ()
rpeconfig_gxpi2_gypi2_updn_dict['theta_cos_prep_str'] = ''
rpeconfig_gxpi2_gypi2_updn_dict['theta_cos_germ_tuple'] = ('Gx', 'Gy', 'Gy', 'Gx', 'Gx', 'Gy', 'Gy', 'Gx')
rpeconfig_gxpi2_gypi2_updn_dict['theta_cos_germ_str'] = 'GxGyGyGxGxGyGyGx'
rpeconfig_gxpi2_gypi2_updn_dict['theta_cos_meas_tuple'] = ()
rpeconfig_gxpi2_gypi2_updn_dict['theta_cos_meas_str'] = ''
rpeconfig_gxpi2_gypi2_updn_dict['theta_sin_prep_tuple'] = ()
rpeconfig_gxpi2_gypi2_updn_dict['theta_sin_prep_str'] = ''
rpeconfig_gxpi2_gypi2_updn_dict['theta_sin_germ_tuple'] = ('Gx', 'Gy', 'Gy', 'Gx', 'Gx', 'Gy', 'Gy', 'Gx')
rpeconfig_gxpi2_gypi2_updn_dict['theta_sin_germ_str'] = 'GxGyGyGxGxGyGyGx'
rpeconfig_gxpi2_gypi2_updn_dict['theta_sin_meas_tuple'] = ('Gy',)
rpeconfig_gxpi2_gypi2_updn_dict['theta_sin_meas_str'] = 'Gy'

import sys
sys.modules[__name__] = _RPEconfig(rpeconfig_gxpi2_gypi2_updn_dict)
