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
import re as _re
import subprocess as _sp
import tempfile as _tf
import numpy as _np
from pathlib import Path as _Path

from ..baseobjs.label import Label as _Label
from pygsti.forwardsims.weakforwardsim import WeakForwardSimulator as _WeakForwardSimulator
from pygsti.modelmembers import states as _state
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.baseobjs.outcomelabeldict import OutcomeLabelDict as _OutcomeLabelDict
# from . import povm as _povm


class CHPForwardSimulator(_WeakForwardSimulator):
    """
    A WeakForwardSimulator returning probabilities with Scott Aaronson's CHP code
    """
    def __init__(self, chpexe, shots, model=None, base_seed=None):
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
        base_seed: int, optional
            Base seed for RNG of probabilitic operations during circuit simulation.
            Incremented for every shot such that deterministic seeding behavior can be
            carried out with both serial or MPI execution.
            If not provided, falls back to using time.time() to get a valid seed.
        """
        self.chpexe = _Path(chpexe)
        assert self.chpexe.is_file(), "A valid CHP executable must be passed to CHPForwardSimulator"

        super().__init__(shots, model, base_seed)

    def _compute_circuit_outcome_for_shot(self, circuit, resource_alloc, time=None, rand_state=None):
        assert(time is None), "CHPForwardSimulator cannot be used to simulate time-dependent circuits yet"
        assert(resource_alloc is None), "CHPForwardSimulator cannot use a resource_alloc for one shot."

        prep_label, op_labels, povm_label = self.model.split_circuit(circuit)

        # Use temporary file as per https://stackoverflow.com/a/8577225
        fd, path = _tf.mkstemp()
        try:
            with _os.fdopen(fd, 'w') as tmp:
                tmp.write('#\n')

                # Prep
                rho = self.model.circuit_layer_operator(prep_label, 'prep')
                tmp.write(rho._rep.chp_str(seed_or_state=rand_state))

                # Op layers
                for op_label in op_labels:
                    op = self.model.circuit_layer_operator(op_label, 'op')
                    tmp.write(op._rep.chp_str(seed_or_state=rand_state))

                # POVM (sort of, actually using it more like a straight PVM)
                povm = self.model.circuit_layer_operator(_Label(povm_label.name), 'povm')
                self._process_povm(povm, povm_label, tmp, rand_state)

            # Run CHP
            process = _sp.Popen([f'{self.chpexe.resolve()}', f'{path}'], stdout=_sp.PIPE, stderr=_sp.PIPE)
            # TODO: Handle errors?
            out, err = process.communicate()
        finally:
            _os.remove(path)

        # Extract outputs
        pattern = _re.compile('Outcome of measuring qubit (\d+): (\d)')
        qubit_outcomes = []
        for match in pattern.finditer(out.decode('utf-8')):
            qubit_outcomes.append((int(match.group(1)), match.group(2)))

        # TODO: Make sure this handles intermediate measurements
        outcome = ''.join([qo[1] for qo in sorted(qubit_outcomes)])
        outcome_label = _OutcomeLabelDict.to_outcome(outcome)

        return outcome_label

    # TODO: This should be somehow part of the evotype reps, but effect/povm is weird for CHP
    def _process_povm(self, povm, povm_label, file_handle, rand_state):
        """Helper function to process measurement for CHP circuits.

        Recursively handles ComposedPOVM > ComputationalBasisPOVM objects
        (e.g. those created by create_crosstalk_free_model).

        Parameters
        ----------
        povm: POVM
            Unmarginalized POVM to process

        povm_label: Label
            POVM label, which may include StateSpaceLabels that result
            in POVM marginalization

        file_handle: TextIOWrapper
            Open file handle for dumping CHP strings
        """
        # Handle marginalization (not through MarginalizedPOVM,
        # where most logic is based on simplify_effects and therefore expensive for many qubits)
        qubit_indices = None
        if povm_label.sslbls is not None:
            flat_sslbls = [lbl for i in range(self.model.state_space.num_tensor_product_blocks)
                           for lbl in self.model.state_space.tensor_product_block_labels(i)]
            qubit_indices = [flat_sslbls.index(q) for q in povm_label.sslbls]

        # Handle ComputationalBasisPOVM
        def process_computational_povm(povm, qubit_indices):
            assert isinstance(povm, _povm.ComputationalBasisPOVM), \
                "CHP povm must be ComputationalPOVM (may be inside ComposedPOVM/TensorProdPOVM)"

            povm_qubits = _np.array(range(povm.nqubits))
            for target in povm_qubits:
                if qubit_indices is None or target in qubit_indices:
                    file_handle.write(f'm {target}\n')

        # Handle ComposedPOVM of ComputationalBasisPOVM + noise op with chp evotype
        def process_composed_povm(povm, qubit_indices):
            if isinstance(povm, _povm.ComposedPOVM):
                assert povm._evotype == 'chp', \
                    "ComposedPOVM must have `chp` evotype for noise op"

                file_handle.write(povm.error_map._rep.chp_str(seed_or_state=rand_state))

                process_computational_povm(povm.base_povm, qubit_indices)
            else:
                process_computational_povm(povm, qubit_indices)
        
        process_composed_povm(povm, qubit_indices)
