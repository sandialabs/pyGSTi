"""
POVM representation classes for the `qibo` evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import os as _os
import re as _re
import subprocess as _sp
import tempfile as _tf
import numpy as _np

from .. import basereps as _basereps
from . import _get_densitymx_mode, _get_nshots
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from pygsti.baseobjs.outcomelabeldict import OutcomeLabelDict as _OutcomeLabelDict

try:
    import qibo as _qibo
except ImportError:
    _qibo = None


class POVMRep(_basereps.POVMRep):
    def __init__(self):
        super(POVMRep, self).__init__()


class ComputationalPOVMRep(POVMRep):
    def __init__(self, nqubits, qubit_filter):
        self.nqubits = nqubits
        self.qubit_filter = qubit_filter
        super(ComputationalPOVMRep, self).__init__()

    #REMOVE
    #def sample_outcome(self, state, rand_state):
    #    chp_ops = state.chp_ops
    #
    #    povm_qubits = _np.array(range(self.nqubits))
    #    for iqubit in povm_qubits:
    #        if self.qubit_filter is None or iqubit in self.qubit_filter:
    #            chp_ops.append(f'm {iqubit}')
    #
    #    # TODO: Make sure this handles intermediate measurements
    #    outcomes, _ = self._run_chp_ops(chp_ops)
    #    outcome = ''.join(outcomes)
    #    outcome_label = _OutcomeLabelDict.to_outcome(outcome)
    #    return outcome_label

    def probabilities(self, state, rand_state, effect_labels):
        qibo_circuit = state.qibo_circuit
        initial_state = state.qibo_state
        # TODO: change below to: sole_tensor_product_block_labels
        qubits_to_measure = state.state_space.tensor_product_block_labels(0) \
            if (self.qubit_filter is None) else self.qubit_filter

        gatetypes_requiring_shots = set(('UnitaryChannel', 'PauliNoiseChannel',
                                         'ResetChannel', 'ThermalRelaxationChannel'))
        circuit_requires_shots = len(gatetypes_requiring_shots.intersection(set(qibo_circuit.gate_types.keys()))) > 0
        if _get_densitymx_mode() or circuit_requires_shots is False:
            #then we can use QIBO's exact .probabilities call:
            results = qibo_circuit(initial_state)
            prob_tensor = results.probabilities(qubits_to_measure)

            probs = [prob_tensor[tuple(map(int, effect_lbl))] for effect_lbl in effect_labels]
            # Above map & int converts, e.g., '01' -> (0,1)
        else:
            #we must use built-in weak fwdsim
            qibo_circuit.add(_qibo.gates.M(*qubits_to_measure))
            nshots = _get_nshots()
            results = qibo_circuit(initial_state, nshots=nshots)
            freqs = results.frequencies(binary=True)
            probs = [freqs[effect_lbl] / nshots for effect_lbl in effect_labels]
    
        return probs


class ComposedPOVMRep(POVMRep):
    def __init__(self, errmap_rep, base_povm_rep, state_space):
        self.errmap_rep = errmap_rep
        self.base_povm_rep = base_povm_rep
        self.state_space = state_space
        super(ComposedPOVMRep, self).__init__()

#REMOVE
#    def sample_outcome(self, state, rand_state):
#        state = self.errmap_rep.acton_random(state, rand_state)
#        return self.base_povm_rep.sample_outcome(state)

    def probabilities(self, state, rand_state, effect_labels):
        state = self.errmap_rep.acton_random(state, rand_state)
        return self.base_povm.probabilities(state, rand_state, effect_labels)
