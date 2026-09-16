#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Utilities that take an ExperimentDesign and tell you something about it, or hand
you a modified one: `calculate_edesign_estimated_runtime` for wall-clock
estimates, the `calculate_fisher_information_*` family for how much a design
tells you about a model.

`DesignReducer` is the interface for cutting a design down to a budget; write
one to plug your own selection rule into `design.reduce_with(...)`.
The `blockdopt` submodule holds the greedy block D-optimal selection kernel,
whose three public functions are re-exported here.

New code should import from here.
"""
from ._fisher import (calculate_fisher_information_matrices_by_L, calculate_fisher_information_matrix,
                      calculate_fisher_information_per_circuit)
from ._reduction import CallableReducer, CircuitSelection, DesignReducer
from ._runtime import calculate_edesign_estimated_runtime
from .blockdopt import block_linear_dopt, greedy_candidate_scores, greedy_path_log_volumes

__all__ = [
    "CallableReducer",
    "CircuitSelection",
    "DesignReducer",
    "block_linear_dopt",
    "calculate_edesign_estimated_runtime",
    "calculate_fisher_information_matrices_by_L",
    "calculate_fisher_information_matrix",
    "calculate_fisher_information_per_circuit",
    "greedy_candidate_scores",
    "greedy_path_log_volumes",
]
