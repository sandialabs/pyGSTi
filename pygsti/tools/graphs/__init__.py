#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

"""Graph utilities: shared definitions, connectivity/representation coercion, and coloring algorithms."""

from ._common import order, canonical_edges, find_neighbors, max_degree
from ._connectivity import (
    qubit_graph_to_networkx,
    qubit_graph_from_edges,
    qubit_graph_adjacency_matrix,
    within_hops_matrix,
    qubits_within_hops,
)
from . import coloring

__all__ = [
    "order",
    "canonical_edges",
    "find_neighbors",
    "max_degree",
    "qubit_graph_to_networkx",
    "qubit_graph_from_edges",
    "qubit_graph_adjacency_matrix",
    "within_hops_matrix",
    "qubits_within_hops",
    "coloring",
]
