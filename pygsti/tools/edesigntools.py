"""
Tools for working with ExperimentDesigns
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

# This module's contents now live in the pygsti.tools.edesign subpackage.  It is
# kept as a compatibility alias so that `pygsti.tools.edesigntools.<name>` keeps
# working; it re-exports the same objects, not copies of them.
from pygsti.tools.edesign import (calculate_edesign_estimated_runtime,
                                  calculate_fisher_information_matrices_by_L,
                                  calculate_fisher_information_matrix,
                                  calculate_fisher_information_per_circuit,
                                  pad_edesign_with_idle_lines)

__all__ = [
    "calculate_edesign_estimated_runtime",
    "calculate_fisher_information_matrices_by_L",
    "calculate_fisher_information_matrix",
    "calculate_fisher_information_per_circuit",
    "pad_edesign_with_idle_lines",
]
