"""
POVM representation classes for the `chp` evolution type.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _cp
import os as _os
import re as _re
import subprocess as _sp
import tempfile as _tf
import numpy as _np

from .. import basereps as _basereps
from pygsti.evotypes.chp import chpexe_path as _chpexe_path
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from pygsti.baseobjs.outcomelabeldict import OutcomeLabelDict as _OutcomeLabelDict


class POVMRep(_basereps.POVMRep):
    def __init__(self):
        super(POVMRep, self).__init__()

    def _run_chp_ops(self, chp_ops):
        chp_program = '\n'.join(chp_ops)
        if len(chp_program) > 0: chp_program += '\n'
        chpexe = _chpexe_path()
        #print("CHP program input (debug):\n", chp_program)

        fd, path = _tf.mkstemp()
        try:
            with _os.fdopen(fd, 'w') as tmp:
                tmp.write('#\n')
                tmp.write(chp_program)

            # Run CHP
            process = _sp.Popen([f'{chpexe.resolve()}', f'{path}'], stdout=_sp.PIPE, stderr=_sp.PIPE)
            # TODO: Handle errors?
            out, err = process.communicate()
        finally:
            #with open(path) as f:  # for debugging REMOVE
            #    print("CHP program dump (debug):")
            #    print(f.read())
            _os.remove(path)

        # Extract outputs
        #print("CHP program out (debug): ", out.decode('utf-8'))
        pattern = _re.compile('Outcome of measuring qubit (\d+): (\d)( ?\S*)')
        matched_values = []  # elements = (qubit_index, outcome, '(random)' or '') tuples
        for match in pattern.finditer(out.decode('utf-8')):
            matched_values.append((int(match.group(1)), match.group(2), match.group(3)))

        # Sorting orders matches by qubit index.
        # TODO: Could have multiple measurements for each qubit...
        assert(len(set([mv[0] for mv in matched_values])) == len(matched_values)), \
            "Cannot currently handle more than one measurement per qubit"
        outcomes = [mv[1] for mv in sorted(matched_values)]
        random_flags = [bool(mv[2] == ' (random)') for mv in sorted(matched_values)]
        return outcomes, random_flags


class ComputationalPOVMRep(POVMRep):
    def __init__(self, nqubits, qubit_filter):
        self.nqubits = nqubits
        self.qubit_filter = qubit_filter
        super(ComputationalPOVMRep, self).__init__()

    def sample_outcome(self, state, rand_state):
        chp_ops = _cp.copy(state.chp_ops)

        povm_qubits = _np.array(range(self.nqubits))
        for iqubit in povm_qubits:
            if self.qubit_filter is None or iqubit in self.qubit_filter:
                chp_ops.append(f'm {iqubit}')

        # TODO: Make sure this handles intermediate measurements
        outcomes, _ = self._run_chp_ops(chp_ops)
        outcome = ''.join(outcomes)
        outcome_label = _OutcomeLabelDict.to_outcome(outcome)
        return outcome_label

    def probabilities(self, state, rand_state, effect_labels):
        chp_ops = _cp.copy(state.chp_ops)

        povm_qubits = _np.array(range(self.nqubits))
        for iqubit in povm_qubits:
            if self.qubit_filter is None or iqubit in self.qubit_filter:
                chp_ops.append(f'm {iqubit}')

        outcomes, random_flags = self._run_chp_ops(chp_ops)

        probs = []
        for effect_lbl in effect_labels:
            assert(len(effect_lbl) == len(povm_qubits))
            p = 1.0
            for ebit, outbit, israndom in zip(effect_lbl, outcomes, random_flags):
                if israndom: p *= 0.5
                elif ebit != outbit:
                    p = 0.0; break
            probs.append(p)
        return probs


class ComposedPOVMRep(POVMRep):
    def __init__(self, errmap_rep, base_povm_rep, state_space):
        self.errmap_rep = errmap_rep
        self.base_povm_rep = base_povm_rep
        self.state_space = state_space
        super(ComposedPOVMRep, self).__init__()

    def sample_outcome(self, state, rand_state):
        state = self.errmap_rep.acton_random(state, rand_state)
        return self.base_povm_rep.sample_outcome(state)

    def probabilities(self, state, rand_state, effect_labels):
        state = self.errmap_rep.acton_random(state, rand_state)
        return self.base_povm.probabilities(state, rand_state, effect_labels)
