"""
An RPE gate set module.

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

import numpy as _np

from ..circuits.circuit import Circuit as _Circuit
from ..protocols import rpe as _rpe


def get_rpe_experiment_design(max_max_length, qubit_labels=None, req_counts=None):
    """
    Create a RPE experiment design based on this model-pack's gate set.

    Parameters
    ----------
    max_max_length : int
        The maximum number of gate repetitions to use.

    qubit_labels : tuple, optional
        If not None, a tuple of the qubit labels to use in the returned circuits. If None,
        then the default labels are used, which are often the integers beginning with 0.

    req_counts : int, optional
        <TODO description>
    """
    max_log_lengths = _np.log2(max_max_length)
    if not (int(max_log_lengths) - max_log_lengths == 0):
        raise ValueError('Only integer powers of two accepted for max_max_length.')

    assert(qubit_labels is None or qubit_labels == (0,)), "Only qubit_labels=(0,) is supported so far"
    return _rpe.RobustPhaseEstimationDesign(
        _Circuit([('Gxpi2', 0)], line_labels=(0,)),
        [2**i for i in range(int(max_log_lengths) + 1)],
        _Circuit([], line_labels=(0,)),
        _Circuit([('Gxpi2', 0)], line_labels=(0,)),
        ['1'],
        ['0'],
        _Circuit([], line_labels=(0,)),
        _Circuit([], line_labels=(0,)),
        ['0'],
        ['1'],
        qubit_labels=qubit_labels,
        req_counts=req_counts)
