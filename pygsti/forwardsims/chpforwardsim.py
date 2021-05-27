"""
Defines the CHPForwardSimulator calculator class
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
from pathlib import Path as _Path
import re as _re
import subprocess as _sp
import tempfile as _tf

#from . import povm as _povm
from ..objects.label import Label as _Label
from ..models.labeldicts import OutcomeLabelDict as _OutcomeLabelDict
from .weakforwardsim import WeakForwardSimulator as _WeakForwardSimulator


class CHPForwardSimulator(_WeakForwardSimulator):
    """
    A WeakForwardSimulator returning probabilities with Scott Aaronson's CHP code
    """
    def __init__(self, chpexe, shots, model=None):
        """
        Construct a new CHPForwardSimulator.

        Parameters
        ----------
        chpexe: str or Path
            Path to CHP executable
        shots: int
            Number of times to run each circuit to obtain an approximate probability
        model : Model
            Optional parent Model to be stored with the Simulator
        """
        self.chpexe = _Path(chpexe)
        assert self.chpexe.is_file(), "A valid CHP executable must be passed to CHPForwardSimulator"

        super().__init__(shots, model)

    def _compute_circuit_outcome_for_shot(self, circuit, resource_alloc, time=None):
        assert(time is None), "CHPForwardSimulator cannot be used to simulate time-dependent circuits yet"

        complete_circuit = self.model.complete_circuit(circuit)
        
        # Use temporary file as per https://stackoverflow.com/a/8577225
        fd, path = _tf.mkstemp()
        try:
            with _os.fdopen(fd, 'w') as tmp:
                tmp.write('#\n')

                # Prep
                # TODO: Make sure this works with SPAMVec
                rho = self.model.circuit_layer_operator(complete_circuit[0], 'prep')
                tmp.write(rho.chp_str())

                # Op layers
                for op_label in complete_circuit[1:-1]:
                    op = self.model.circuit_layer_operator(op_label, 'op')
                    tmp.write(op.chp_str())
                
                # POVM (sort of, actually using it more like a straight PVM)
                povm_label = complete_circuit[-1]
                povm = self.model.circuit_layer_operator(_Label(povm_label.name), 'povm')
                assert(isinstance(povm, _povm.ComputationalBasisPOVM)), "CHP POVM must be a ComputationalBasisPOVM"

                # Handle marginalization (not through MarginalizedPOVM,
                # where most logic is based on simplify_effects and therefore expensive for many qubits)
                if povm_label.sslbls is not None:
                    assert(self.model.state_space.num_tensor_product_blocks == 1), \
                        "Only single-TPB state spaces are supported with CHPForwardSimulator"
                    flat_sslbls = self.model.state_space.tensor_product_block_labels(0)
                    qubit_indices = [flat_sslbls.index(q) for q in povm_label.sslbls]
                else:
                    qubit_indices = range(povm.nqubits)
                    
                for qind in qubit_indices:
                    tmp.write(f'm {qind}\n')

            # Run CHP
            process = _sp.Popen([f'{self.chpexe.resolve()}', f'{path}'], stdout=_sp.PIPE, stderr=_sp.PIPE)
            # TODO: Handle errors?
            out, err = process.communicate()
        finally:
            _os.remove(path)

        # Extract outputs
        pattern = _re.compile('Outcome of measuring qubit (\d): (\d)')
        qubit_outcomes = []
        for match in pattern.finditer(out.decode('utf-8')):
            qubit_outcomes.append((int(match.group(1)), match.group(2)))

        # TODO: Make sure this handles intermediate measurements
        outcome = ''.join([qo[1] for qo in sorted(qubit_outcomes)])
        outcome_label = _OutcomeLabelDict.to_outcome(outcome)
        
        return outcome_label
