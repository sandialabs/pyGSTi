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

#from . import povm as _povm
from ..baseobjs.label import Label as _Label
# ***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************
import os as _os
import re as _re
import subprocess as _sp
import tempfile as _tf
import numpy as _np
from pathlib import Path as _Path

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

        # Don't error on POVM, in case it's just an issue of marginalization
        prep_label, op_labels, povm_label = self.model.split_circuit(circuit, erroron=('prep',))
        # Try to get unmarginalized POVM
        if povm_label is None:
            #OLD way (perhaps to avoid constructing a MarginalizedPOVM on many qubits?):
            #default_povm_label = self.model._default_primitive_povm_layer_lbl(None)
            #povm_label = _Label(default_povm_label.name, state_space_labels=circuit.line_labels)
            povm_label = self.model._default_primitive_povm_layer_lbl(circuit.line_labels)
        assert (povm_label is not None), \
            "Unable to get default POVM for %s" % str(circuit)

        # Use temporary file as per https://stackoverflow.com/a/8577225
        fd, path = _tf.mkstemp()
        try:
            with _os.fdopen(fd, 'w') as tmp:
                tmp.write('#\n')

                # Prep
                rho = self.model.circuit_layer_operator(prep_label, 'prep')
                self._process_state(rho, tmp)

                # Op layers
                for op_label in op_labels:
                    op = self.model.circuit_layer_operator(op_label, 'op')
                    tmp.write(op._rep.chp_str())

                # POVM (sort of, actually using it more like a straight PVM)
                povm = self.model.circuit_layer_operator(_Label(povm_label.name), 'povm')
                self._process_povm(povm, povm_label, tmp)

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

    def _process_state(self, rho, file_handle):
        """Helper function to process state prep for CHP circuits.

        Recursively handles TensorProd > Composed > Computational
        SPAMVec objects (e.g. those created by create_crosstalk_free_model).

        Parameters
        ----------
        rho: State
            State vector to process

        file_handle: TextIOWrapper
            Open file handle for dumping CHP strings
        """
        # Handle ComputationalBasisState, applying bitflips as needed
        def process_computational_state(rho, target_offset=0):
            assert isinstance(rho, _state.ComputationalBasisState), \
                "CHP prep must be ComputationalBasisState (may be inside ComposedState/TensorProductState)"

            bitflip = _op.StaticStandardOp('Gxpi', 'pp', evotype='chp', state_space=None)
            for i, zval in enumerate(rho._zvals):
                if zval:
                    file_handle.write(bitflip.chp_str)
                    #OLD: file_handle.write(bitflip.get_chp_str([target_offset + i]))

        # Handle ComposedState of ComputationalBasisState + noise op with chp evotype
        def process_composed_state(rho, target_offset=0):
            if isinstance(rho, _state.ComposedState):
                assert(rho._evotype == 'chp'), "ComposedState must have `chp` evotype for noise op"
                process_computational_state(rho.state_vec, target_offset)
                #nqubits = rho.state_space.num_qubits
                #targets = _np.array(range(nqubits)) + target_offset  # Stefan - this used to be an arg to chp_str (?)
                file_handle.write(rho.error_map.chp_str)
            else:
                process_computational_state(rho, target_offset)

        # Handle TensorProductSTate made of ComposedState or ComputationalBasisStates
        target_offset = 0
        if isinstance(rho, _state.TensorProductState):
            for rho_factor in rho.factors:
                nqubits = (rho_factor.noise_op.dim - 1).bit_length()
                process_composed_state(rho_factor, target_offset)
                target_offset += nqubits
        else:
            process_composed_state(rho, target_offset)

    def _process_povm(self, povm, povm_label, file_handle):
        """Helper function to process measurement for CHP circuits.

        Recursively handles TensorProd > Composed > ComputationalBasis
        POVM objects (e.g. those created by create_crosstalk_free_model).

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
        def process_computational_povm(povm, qubit_indices, target_offset=0):
            assert isinstance(povm, _povm.ComputationalBasisPOVM), \
                "CHP povm must be ComputationalPOVM (may be inside ComposedPOVM/TensorProdPOVM)"

            povm_qubits = _np.array(range(povm.nqubits)) + target_offset
            for target in povm_qubits:
                if qubit_indices is None or target in qubit_indices:
                    file_handle.write(f'm {target}\n')

        # Handle ComposedPOVM of ComputationalBasisPOVM + noise op with chp evotype
        def process_composed_povm(povm, qubit_indices, target_offset=0):
            if isinstance(povm, _povm.ComposedPOVM):
                assert povm._evotype == 'chp', \
                    "ComposedPOVM must have `chp` evotype for noise op"

                #OLD REMOVE: nqubits = povm.state_space.num_qubits
                #OLD REMOVE: targets = _np.array(range(nqubits)) + target_offset
                #OLD REMOVE: file_handle.write(povm.error_map.get_chp_str(targets))
                file_handle.write(povm.error_map.chp_str)

                process_computational_povm(povm.base_povm, qubit_indices, target_offset)
            else:
                process_computational_povm(povm, qubit_indices, target_offset)

        # Handle TensorProductPOVM made of ComposedPOVM or ComputationalBasisPOVMs
        target_offset = 0
        if isinstance(povm, _povm.TensorProductPOVM):
            for povm_factor in povm.factors:
                nqubits = povm_factor.error_map.state_space.num_qubits
                process_composed_povm(povm_factor, qubit_indices, target_offset)
                target_offset += nqubits
        else:
            process_composed_povm(povm, qubit_indices, target_offset)
