"""
Edge-coloring toolkit: a name-based switchboard (`switchboard_find_edge_coloring`)
over several edge-coloring algorithms, plus a verification helper
(`check_valid_edge_coloring`) to validate the resulting coloring.
"""
from ._dispatch import switchboard_find_edge_coloring
from ._common import check_valid_edge_coloring

__all__ = [
    "switchboard_find_edge_coloring",
    "check_valid_edge_coloring",
]
