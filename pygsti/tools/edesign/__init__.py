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
tells you about a model, and `pad_edesign_with_idle_lines` for widening a design
onto more qubits.

`DesignReducer` is the interface for cutting a design down to a budget; write
one to plug your own selection rule into `design.reduce_with(...)`.
`blockdopt` supplies the reference implementation, which ranks candidates by
how much information each adds about a model's parameters.

`pygsti.tools.edesigntools` is a compatibility alias for the runtime, Fisher
and padding names below.  New code should import from here.
"""
from ._fisher import (calculate_fisher_information_matrices_by_L, calculate_fisher_information_matrix,
                      calculate_fisher_information_per_circuit)
from ._padding import pad_edesign_with_idle_lines
from ._reduction import CallableReducer, CircuitSelection, DesignReducer
from ._runtime import calculate_edesign_estimated_runtime
from .blockdopt import (BlockDoptReducer, block_linear_dopt, greedy_candidate_scores,
                        greedy_path_log_volumes, jacobian_dict_to_array, perturb_errorgen_rates,
                        rank_circuits_by_dopt, reduce_design_by_dopt)

__all__ = [
    "BlockDoptReducer",
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
    "jacobian_dict_to_array",
    "pad_edesign_with_idle_lines",
    "perturb_errorgen_rates",
    "rank_circuits_by_dopt",
    "reduce_design_by_dopt",
]
