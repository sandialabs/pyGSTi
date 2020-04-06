""" Utilities for defining advanced low-level parameterizations for various pyGSTi operations """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


class AdvancedOptions(dict):
    valid_keys = ()

    def __init__(self, items=None):
        super().__init__()
        self.update(items or {})

    def __setitem__(self, key, val):
        if key not in self.valid_keys:
            raise ValueError("Invalid key '%s'. Valid keys are: '%s'" % (str(key), "', '".join(self.valid_keys)))
        super().__setitem__(key, val)

    def update(self, d):
        invalid_keys = [k for k in d.keys() if k not in self.valid_keys]
        if invalid_keys:
            raise ValueError("Invalid keys '%s'. Valid keys are: '%s'" % ("', '".join(invalid_keys),
                             "', '".join(self.valid_keys)))


class GSTAdvancedOptions(AdvancedOptions):
    valid_keys = ('germ_length_limits', 'include_lgst', 'nested_circuit_lists',
                  'string_manipulation_rules', 'op_label_aliases', 'circuit_weights',
                  'profile', 'record_output', 'distribute_method',
                  'objective', 'use_freq_weighted_chi2', 'prob_clip_interval', 'min_prob_clip',
                  'min_prob_clip_for_weighting', 'radius', 'cptp_penalty_factor', 'spam_penalty_factor',
                  'bad_fit_threshold', 'on_bad_fit',
                  'starting_point', 'depolarize_start', 'randomize_start', 'lgst_gaugeopt_tol',
                  'contract_start_to_cptp',
                  'always_perform_mle', 'only_perform_mle',
                  'max_iterations', 'tolerance', 'finitediff_iterations', 'extra_lm_opts',
                  'set trivial_gauge_group', 'op_labels', 'unreliable_ops')
