# ***************************************************************************************************
# Copyright 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************
import copy as _copy
import numpy as _np
import itertools as _itertools
import pathlib as _pathlib
import warnings as _warnings

from pygsti.protocols.treenode import TreeNode as _TreeNode
from pygsti import io as _io
from collections.abc import Container as _Container
from pygsti.circuits import (
    Circuit as _Circuit, 
    CircuitList as _CircuitList,
    CircuitPlaquette as _CircuitPlaquette
)
from pygsti import data as _data
from pygsti.tools import NamedDict as _NamedDict
from pygsti.tools import listtools as _lt
from pygsti.tools.dataframetools import _process_dataframe
from pygsti.baseobjs.label import (
    LabelStr as _LabelStr
)
from pygsti.baseobjs.mongoserializable import MongoSerializable as _MongoSerializable
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.protocols.protocol import Protocol as _Protocol, CircuitListsDesign as _CircuitListsDesign

from typing import Union, Literal

BinningStrategy = Union[
    int, Literal['auto-int'], Literal['auto'], Literal['fd'], Literal['doane'], Literal['scott'], Literal['stone'], Literal['sturges']
]
LengthTransformer = Union[
    Literal['log'], Union[Literal['none'], None], _np.ufunc
]

def histonested_circuitlists(
        circuits: Union[_CircuitList, list[_Circuit]],
        bins: BinningStrategy = 'auto-int',
        trans: LengthTransformer = 'log'
    ) -> list[_Circuit]:
    """
    This is a helper function for building CircuitListsDesign objects with certain
    nested structures. If `clists` is the output of this function, then the induced
    design is canonically

        d = CircuitListsDesign(clists nested=True, remove_duplicates=True).

    If `circuits` contained no duplicates, then we'll have
     
         set(circuits) == set(d.all_circuits_needing_data).
    """
    assert len(circuits) > 0
    lengths = _np.array([len(c) + 1 for c in circuits])
    if isinstance(bins, str) and 'auto' in bins and 'int' in bins: # type: ignore
        bins = int(_np.log2(_np.max(lengths)))
    if isinstance(trans, _np.ufunc):
        lengths = trans(lengths)
    elif trans == 'log':
        lengths = _np.log2(lengths)
    elif (trans != 'none') and (trans is not None):
        raise ValueError(f'Argument `trans` had unsupported value, {trans}.')
    # get bin edges from numpy, then drop empty bins.
    counts, edges = _np.histogram(lengths, bins)
    edges = _np.concatenate([[edges[0]], edges[1:][counts > 0]])
    assignments = _np.digitize(lengths, edges)
    assignments -= 1
    # edges[ assignments[j] ] <= lengths[j] < edges[ assignments[j]+1 ]
    num_bins = edges.size - 1
    circuit_lists = [list() for _ in range(num_bins)]
    for j, c in zip(assignments, circuits):
        for i in range(j, num_bins):
            circuit_lists[i].append(c)
    """
    # The following approch to building circuit_lists is (in theory) less efficient than
    # the approach above, but it has the advantage of avoiding the nested for-loop.
    circuit_lists = []
    last_size = 0
    edges = _np.histogram_bin_edges(lengths, bins)
    for upperbound in edges[1:]:
        cur_inds = _np.where(lengths < upperbound)[0]
        if cur_inds.size == last_size:
            continue  # empty bin
        last_size = cur_inds.size
        circuit_lists.append([circuits[j] for j in cur_inds])
    """
    return circuit_lists # type: ignore


def logspaced_prefix_circuits(
    c: _Circuit,
    povms_to_keep: _Container[_LabelStr] = (_LabelStr('Mdefault'),),
    base: Union[int, float]=2,
    editable = False
    ) -> list[_Circuit]:

    if len(c) > 0 and c[-1] in povms_to_keep:
        M = c[-1]
        if not isinstance(M, _LabelStr):
            M = _LabelStr(M)
        c = c[:-1]
        circuits = logspaced_prefix_circuits(c, tuple(), base, editable=True)
        for c in circuits:
            c._labels.append(M) # type: ignore
            c.done_editing()
        return circuits

    if editable and not c._static:
        c = c.copy(editable=True) # type: ignore
    
    circuits = [c]
    assert base > 1
    next_len = int(len(c) // base)
    while next_len > 0:
        c = c[:next_len]
        circuits.append(c)
        next_len = int(len(c) // base)

    if not editable:
        for c in circuits:
            c.done_editing()

    return circuits

