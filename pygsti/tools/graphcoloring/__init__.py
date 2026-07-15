"""
Edge-coloring toolkit: a name-based switchboard (`switchboard_find_edge_coloring`)
over several edge-coloring algorithms, plus topology-aware closed-form
colorings (`auto_edge_coloring`) for the canonical topologies produced by
`ProcessorSpec(geometry=...)`.

This is a package (split by algorithm family for readability); everything
below is re-exported here so `from pygsti.tools.graphcoloring import X`
keeps working exactly as it did when this was a single module.
"""
from ._common import Vertex, Color, Edge, NeighborMap, Coloring, order, check_valid_edge_coloring
from ._dispatch import find_edge_coloring, switchboard_find_edge_coloring
from ._topology import detect_topology, auto_edge_coloring
from ._vizing import misra_gries_edge_coloring, vizing_edge_coloring, new_bipartite_edge_coloring
from ._sinnamon import (
    _eulerian_partition,
    sinnamon_2d_minus_1_edge_coloring,
    sinnamon_euler_color_edge_coloring,
)
from ._line_graph import ColoringSolver, moser_tardos_edge_coloring, assadi_oct25_edge_coloring

__all__ = [
    "Vertex", "Color", "Edge", "NeighborMap", "Coloring",
    "order", "check_valid_edge_coloring",
    "find_edge_coloring", "switchboard_find_edge_coloring",
    "detect_topology", "auto_edge_coloring",
    "misra_gries_edge_coloring", "vizing_edge_coloring", "new_bipartite_edge_coloring",
    "sinnamon_2d_minus_1_edge_coloring", "sinnamon_euler_color_edge_coloring",
    "ColoringSolver", "moser_tardos_edge_coloring", "assadi_oct25_edge_coloring",
    "_eulerian_partition",
]
