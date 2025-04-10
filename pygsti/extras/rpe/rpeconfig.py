""" Defines the RPEconfig class and supporting functionality."""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


class RPEconfig:
    """
    Encapsulates a collection of settings for an RPE run.  Provides full specifications,
    including target gates, SPAM, sine and cosine strings for alpha, epsilon, and theta.
    """

    def __init__(self, input_dict):
        unspecified_keys = []
        for nm in ('fixed_axis_gate_label', 'fixed_axis_label', 'fixed_axis_target',
                   'loose_axis_gate_label', 'loose_axis_label', 'loose_axis_target',
                   'auxiliary_axis_gate_label', 'auxiliary_axis_label',
                   'rhoExpressions', 'ELabels', 'EExpressions', 'spamLabelDict', 'dn_labels', 'up_labels',
                   'alpha', 'epsilon', 'theta',
                   'new_epsilon_func', 'alpha_hat_func', 'epsilon_hat_func', 'Phi_hat_func',
                   'alpha_cos_prep_tuple', 'alpha_cos_prep_str', 'alpha_cos_germ_tuple',
                   'alpha_cos_germ_str', 'alpha_cos_meas_tuple', 'alpha_cos_meas_str',
                   'alpha_sin_prep_tuple', 'alpha_sin_prep_str', 'alpha_sin_germ_tuple',
                   'alpha_sin_germ_str', 'alpha_sin_meas_tuple', 'alpha_sin_meas_str',
                   'epsilon_cos_prep_tuple', 'epsilon_cos_prep_str', 'epsilon_cos_germ_tuple',
                   'epsilon_cos_germ_str', 'epsilon_cos_meas_tuple', 'epsilon_cos_meas_str',
                   'epsilon_sin_prep_tuple', 'epsilon_sin_prep_str', 'epsilon_sin_germ_tuple',
                   'epsilon_sin_germ_str', 'epsilon_sin_meas_tuple', 'epsilon_sin_meas_str',
                   'theta_cos_prep_tuple', 'theta_cos_prep_str', 'theta_cos_germ_tuple',
                   'theta_cos_germ_str', 'theta_cos_meas_tuple', 'theta_cos_meas_str',
                   'theta_sin_prep_tuple', 'theta_sin_prep_str', 'theta_sin_germ_tuple',
                   'theta_sin_germ_str', 'theta_sin_meas_tuple', 'theta_sin_meas_str'):
            try:
                self.__dict__[nm] = input_dict[nm]
            except:
                unspecified_keys.append(nm)

        if unspecified_keys == []:
            #print("Fully specified RPE configuration.")
            pass  # no error
        else:
            raise ValueError(("RPE configuration not fully specified.  "
                              "Missing following keys:\n")
                             + '\n'.join(unspecified_keys))
