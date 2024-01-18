"""
Defines the TorchForwardSimulator class
"""
#***************************************************************************************************
# Copyright 2024, National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as warnings
from typing import Tuple, Optional, TypeVar

import numpy as np
import scipy.linalg as la
try:
    import torch
    ENABLED = True
except ImportError:
    ENABLED = False

from pygsti.forwardsims.forwardsim import ForwardSimulator
from pygsti.forwardsims.forwardsim import _bytes_for_array_types
from pygsti.layouts.maplayout import MapCOPALayout
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter
from pygsti.tools import sharedmemtools as _smt
from pygsti.tools import slicetools as _slct
from pygsti.tools.matrixtools import _fas

# Below: imports only needed for typehints
from pygsti.circuits import Circuit
from pygsti.baseobjs.resourceallocation import ResourceAllocation
ExplicitOpModel = TypeVar('ExplicitOpModel')
# ^ declare to avoid circular reference (pygsti.models.explicitmodel.ExplicitOpModel)



def propagate_staterep(staterep, operationreps):
    ret = staterep.actionable_staterep()
    for oprep in operationreps:
        ret = oprep.acton(ret)
    return ret


class TorchForwardSimulator(ForwardSimulator):
    """
    A forward simulator that leverages automatic differentiation in PyTorch.
    (The current work-in-progress implementation has no Torch functionality whatsoever.)
    """

    def __init__(self, model: Optional[ExplicitOpModel] = None):
        if model is not None:
            assert isinstance(model, ExplicitOpModel)
        super().__init__(model)

    def _compute_circuit_outcome_probabilities(
            self, array_to_fill: np.ndarray, circuit: Circuit,
            outcomes: Tuple[Tuple[str]], resource_alloc: ResourceAllocation, time=None
        ):
        expanded_circuit_outcomes = circuit.expand_instruments_and_separate_povm(self.model, outcomes)
        outcome_to_index = {outc: i for i, outc in enumerate(outcomes)}
        if time is not None:
            raise NotImplementedError()
        for spc, spc_outcomes in expanded_circuit_outcomes.items():
            # ^ spc is a SeparatePOVMCircuit
            # Note: `spc.circuit_without_povm` *always* begins with a prep label.
            prep_label = spc.circuit_without_povm[0]
            op_labels  = spc.circuit_without_povm[1:]
            povm_label = spc.povm_label

            # function calls that eventually reach
            #   ExplicitLayerRules.prep_layer_operator,
            #   ExplicitLayerRules.povm_layer_operator,
            #   ExplicitLayerRules.operation_layer_operator
            # for self.model._layer_rules as the ExplicitLayerRules object.
            rho = self.model.circuit_layer_operator(prep_label, typ='prep')
            povm = self.model.circuit_layer_operator(povm_label, typ='povm')
            ops = [self.model.circuit_layer_operator(ol, 'op') for ol in op_labels]

            rhorep  = rho._rep
            povmrep = povm._rep
            opreps = [op._rep for op in ops]
            
            rhorep = propagate_staterep(rhorep, opreps)

            indices = [outcome_to_index[o] for o in spc_outcomes]
            if povmrep is None:
                ereps = [self.model.circuit_layer_operator(elabel, 'povm')._rep for elabel in spc.full_effect_labels]
                array_to_fill[indices] = [erep.probability(rhorep) for erep in ereps]  # outcome probabilities
            else:
                raise NotImplementedError()
        pass

