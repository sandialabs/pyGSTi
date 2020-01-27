"""
Variables for working with the a partial model the X(pi/2)
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti import objects as _obj
import numpy as _np
from ..protocols import rpe as _rpe


def get_rpe_experiment_design(max_max_length, qubit_labels=None, req_counts=None):
    max_log_lengths = _np.log2(max_max_length)
    if not (int(max_log_lengths) - max_log_lengths == 0):
        raise ValueError('Only integer powers of two accepted for max_max_length.')

    assert(qubit_labels is None or qubit_labels == (0,)), "Only qubit_labels=(0,) is supported so far"
    return _rpe.RobustPhaseEstimationDesign(
        _obj.Circuit([('Gxpi2', 0)], line_labels=(0,)),
        [2**i for i in range(int(max_log_lengths) + 1)],
        _obj.Circuit([], line_labels=(0,)),
        _obj.Circuit([('Gxpi2', 0)], line_labels=(0,)),
        ['1'],
        ['0'],
        _obj.Circuit([], line_labels=(0,)),
        _obj.Circuit([], line_labels=(0,)),
        ['0'],
        ['1'],
        qubit_labels=qubit_labels,
        req_counts=req_counts)
