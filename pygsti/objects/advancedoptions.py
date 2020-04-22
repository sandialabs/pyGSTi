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
            raise ValueError("Invalid key '%s'. Valid keys are: '%s'" %
                             (str(key), "', '".join(sorted(self.valid_keys))))
        super().__setitem__(key, val)

    def update(self, d):
        invalid_keys = [k for k in d.keys() if k not in self.valid_keys]
        if invalid_keys:
            raise ValueError("Invalid keys '%s'. Valid keys are: '%s'" % ("', '".join(invalid_keys),
                                                                          "', '".join(sorted(self.valid_keys))))
        super().update(d)


class GSTAdvancedOptions(AdvancedOptions):
    valid_keys = (
        'always_perform_mle',
        'bad_fit_threshold',
        'circuit_weights',
        'contract_start_to_cptp',
        'cptp_penalty_factor',
        'depolarize_start',
        'distribute_method',
        'estimate_label'
        'extra_lm_opts',
        'finitediff_iterations',
        'germ_length_limits',
        'include_lgst',
        'lgst_gaugeopt_tol',
        'max_iterations',
        'min_prob_clip',
        'min_prob_clip_for_weighting',
        'nested_circuit_lists',
        'objective',
        'on_bad_fit',
        'only_perform_mle',
        'op_label_aliases',
        'op_labels',
        'prob_clip_interval',
        'profile',
        'radius',
        'randomize_start',
        'record_output',
        'set trivial_gauge_group',
        'spam_penalty_factor',
        'starting_point',
        'string_manipulation_rules',
        'tolerance',
        'unreliable_ops',
        'use_freq_weighted_chi2',
    )
