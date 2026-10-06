"""
DFE Protocol objects
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from collections import defaultdict
import numpy as _np

from pygsti.protocols import protocol as _proto
from pygsti.protocols import vb as _vb
from pygsti import tools as _tools
from pygsti.algorithms import dfe as _dfe
from pygsti.protocols import rb as _rb
from pygsti.baseobjs.label import Label as _Label


class DFEDesign(_vb.BenchmarkingDesign):
    """
    An experiment design for direct fidelity estimation of a set of Clifford circuits.

    For each circuit, `num_samples` DFE circuits are sampled with
    :func:`~pygsti.algorithms.dfe.sample_dfe_circuit`.

    This currently only works if `circuits` are all on the same qubits
    b/c it is built on BenchmarkingDesign. Code will likely silently
    break if this isn't true

    Parameters
    ----------
    pspec : QubitProcessorSpec
        The processor specification that the circuits are defined on.

    circuits : iterable of Circuit
        The Clifford target circuits whose fidelities are to be estimated.

    clifford_compilations : CompilationRules
        The absolute Clifford compilation rules used to compile the DFE preparation and
        measurement layers into the native gates of `pspec`.

    num_samples : int
        The number of DFE circuits sampled for each target circuit.

    descriptor : str, optional
        A string describing the experiment design.

    add_default_protocol : bool, optional
        Currently ignored: the design always attaches a default `RB` protocol.

    seed : int, optional
        A seed for the random number generator, for reproducible sampling.

    fixed_identity_locations : None, 'idle' or sequence, optional
        Qubits on which the sampled Pauli is pinned to the identity. If None, a uniformly
        random non-identity Pauli is sampled over all qubits for every circuit. If `'idle'`,
        each circuit pins the qubits that no gate acts on: empty layers are ignored, and an
        explicit idle gate such as `Gi` counts as a gate. Otherwise, a sequence with one
        entry per circuit, each None or a list of bools over that circuit's `line_labels`
        (see :func:`~pygsti.algorithms.dfe.sample_dfe_circuit`). Pinned sampling has fixed
        support and generally gives a biased process fidelity, for every circuit with a
        mask, including an `'idle'` circuit with no idle qubit.

    create_same_measurement_reference : bool, optional
        Not implemented. Passing True raises NotImplementedError.
    """
    def __init__(self, pspec, circuits, clifford_compilations, num_samples,
                 descriptor='A DFE experiment', add_default_protocol=False, seed=None,
                 fixed_identity_locations=None, create_same_measurement_reference=False):

        circuits = list(circuits)
        identity_masks = self._resolve_identity_masks(circuits, fixed_identity_locations)

        circuit_lists = []
        measurements = []
        signs = []

        rand_state = _np.random.RandomState(seed)  # Ok if seed is None

        for c, mask in zip(circuits, identity_masks):
            dfe_circuits_for_c = []
            measurements_for_c = []
            signs_for_c = []
            for sample_num in range(num_samples):
                dfe_circ, meas, sign = _dfe.sample_dfe_circuit(
                    pspec, c, clifford_compilations,
                    create_same_measurement_reference=create_same_measurement_reference,
                    seed=rand_state.randint(0, 2**31), fixed_identity_locations=mask)
                dfe_circuits_for_c.append(dfe_circ)
                measurements_for_c.append(meas)
                signs_for_c.append(int(sign))

            circuit_lists.append(dfe_circuits_for_c)
            measurements.append(measurements_for_c)
            signs.append(signs_for_c)

        self._init_foundation(circuit_lists, measurements, signs, num_samples, descriptor,
                              add_default_protocol)

    @staticmethod
    def _resolve_identity_masks(circuits, fixed_identity_locations):
        # Returns one fixed_identity_locations entry (None or a list of bools over the
        # circuit's line labels) per circuit. Consumes no randomness; the per-mask checks
        # (length, all-True) are left to sample_dfe_circuit.
        if fixed_identity_locations is None:
            return [None] * len(circuits)
        if isinstance(fixed_identity_locations, str):
            if fixed_identity_locations != 'idle':
                raise ValueError("fixed_identity_locations must be None, 'idle' or a sequence with one "
                                 "entry per circuit, not %r." % fixed_identity_locations)
            masks = []
            for c in circuits:
                idle = c.idling_lines(idle_layer_labels=[_Label(())])
                masks.append([q in idle for q in c.line_labels])
            return masks
        if len(fixed_identity_locations) != len(circuits):
            raise ValueError("fixed_identity_locations has %d entries, but there are %d circuits."
                             % (len(fixed_identity_locations), len(circuits)))
        for entry in fixed_identity_locations:
            if not (entry is None or isinstance(entry, (list, tuple))
                    or (isinstance(entry, _np.ndarray) and entry.ndim == 1)):
                raise ValueError("fixed_identity_locations takes one mask (or None) per circuit, but got "
                                 "the entry %r." % (entry,))
        return list(fixed_identity_locations)

    def _init_foundation(self, circuit_lists, measurements, signs, num_samples, descriptor,
                         add_default_protocol):
        # Pair these attributes with circuit data so that we serialize/truncate properly
        self.paired_with_circuit_attrs = ["measurements", "signs"]

        dummy_depths = list(range(len(circuit_lists)))
        qubit_labels = circuit_lists[0][0].line_labels
        super().__init__(dummy_depths, circuit_lists, signs, qubit_labels, remove_duplicates=False)
        self.measurements = measurements
        self.signs = signs
        self.circuits_per_depth = num_samples
        self.descriptor = descriptor

        # This is a hack to do the data processing automatically
        defaultfit = 'A-fixed'
        self.add_default_protocol(_rb.RB(name='RB', defaultfit=defaultfit))
