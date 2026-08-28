#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""
Edge-coloring toolkit: a name-based switchboard (`switchboard_find_edge_coloring`)
over several edge-coloring algorithms, plus a verification helper
(`check_valid_edge_coloring`) to validate the resulting coloring.
"""
from ._dispatch import switchboard_find_edge_coloring
from ._definitions import check_valid_edge_coloring

__all__ = [
    "switchboard_find_edge_coloring",
    "check_valid_edge_coloring",
]
