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
from pygsti.tools import mcfetools as _mcfe
from pygsti.baseobjs.label import Label as _Label


class DFEDesign(_vb.BenchmarkingDesign):
    """
    An experiment design for direct fidelity estimation of a set of Clifford circuits.

    For each circuit, `num_samples` DFE circuits are sampled with
    :func:`~pygsti.algorithms.dfe.sample_dfe_circuit`.

    The design's `qubit_labels` are the first circuit's `line_labels`. The :class:`DFE`
    protocol handles circuits on different qubits, but other machinery inherited from
    BenchmarkingDesign may assume that every circuit shares those labels.

    Whether any circuit was sampled with a mask is recorded in the boolean attribute
    `has_fixed_identity_locations`.

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
        If True, a :class:`DFE` protocol named `'DFE'` is attached as the default protocol.
        Combining this with `fixed_identity_locations` raises ValueError.

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
        mask, including an `'idle'` circuit with no idle qubit. The :class:`DFE` protocol
        refuses designs with any mask.

    create_same_measurement_reference : bool, optional
        Not implemented. Passing True raises NotImplementedError.
    """
    def __init__(self, pspec, circuits, clifford_compilations, num_samples,
                 descriptor='A DFE experiment', add_default_protocol=False, seed=None,
                 fixed_identity_locations=None, create_same_measurement_reference=False):

        circuits = list(circuits)
        identity_masks = self._resolve_identity_masks(circuits, fixed_identity_locations)
        pinned = any(mask is not None for mask in identity_masks)
        if pinned and add_default_protocol:
            raise ValueError("The DFE protocol refuses designs with fixed_identity_locations, because their "
                             "fixed-support Paulis bias its process-fidelity estimate. Use "
                             "add_default_protocol=False to build this design for other analyses.")

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
                              add_default_protocol, pinned)

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
                         add_default_protocol, has_fixed_identity_locations):
        # Pair these attributes with circuit data so that we serialize/truncate properly
        self.paired_with_circuit_attrs = ["measurements", "signs"]

        dummy_depths = list(range(len(circuit_lists)))
        qubit_labels = circuit_lists[0][0].line_labels
        super().__init__(dummy_depths, circuit_lists, signs, qubit_labels, remove_duplicates=False)
        self.measurements = measurements
        self.signs = signs
        self.circuits_per_depth = num_samples
        self.descriptor = descriptor
        self.has_fixed_identity_locations = bool(has_fixed_identity_locations)

        if add_default_protocol:
            self.add_default_protocol(DFE(name='DFE'))


class DFE(_vb.SummaryStatistics):
    """
    Direct fidelity estimation of the process fidelity of each target circuit in a DFEDesign.

    For a Clifford target on n qubits (its own `line_labels`) under stochastic Pauli errors, the energy of a dressed
    circuit is an unbiased estimate of the Pauli fidelity of its sampled Pauli. The default
    sampling is uniform over all 4^n - 1 non-identity Paulis, so the mean energy `p` over a
    target's dressed circuits gives the process fidelity `1 - (4^n - 1)(1 - p)/4^n`, computed by
    :func:`~pygsti.tools.mcfetools.polarization_to_fidelity`. For n >= 20 that function returns
    `p` itself.

    Non-Pauli noise can bias the estimate: qubits where the sampled Pauli is the identity start
    in |0> rather than in a random state, so non-unital errors such as amplitude damping shift
    the estimate by an amount that depends on the target circuit. Under T1 decay toward |0> the
    fidelity is overstated.

    The standard error is `(1 - 4^-n)` times the sample standard deviation of the energies,
    divided by the square root of the number of dressed circuits K. It folds shot noise and
    Pauli sampling into one empirical spread. It is zero when every energy is equal, as on
    noiseless data, and NaN when K is 1. With few samples, or when the Pauli fidelities take few
    distinct values, it can underestimate the true error.

    The estimate covers the whole dressed circuit: errors in the preparation and
    measurement-basis layers that dressing adds, and in state preparation and readout, are
    included.

    Designs that aren't :class:`DFEDesign` objects are refused with TypeError, since other
    benchmarking designs (e.g. `BinaryRBDesign`) carry similar attributes but would give
    meaningless numbers. Designs built with `fixed_identity_locations` are refused with
    ValueError, because their sampled Paulis have fixed support and the process-fidelity
    estimate would be biased.

    Parameters
    ----------
    name : str, optional
        The name of this protocol, also used to (by default) name the
        results produced by this protocol. If None, the class name will
        be used.
    """

    def __init__(self, name=None):
        super().__init__(name)

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data, whose experiment design must be a :class:`DFEDesign`.

        memlimit : int, optional
            A rough per-processor memory limit in bytes. Unused.

        comm : mpi4py.MPI.Comm, optional
            An MPI communicator. Unused.

        Returns
        -------
        DFEResults
        """
        design = data.edesign
        if not isinstance(design, DFEDesign):
            raise TypeError("The DFE protocol needs a DFEDesign, not a %s." % type(design).__name__)
        if getattr(design, 'has_fixed_identity_locations', False):
            raise ValueError("This design pins identity locations (fixed_identity_locations), so its sampled "
                             "Paulis have fixed support and the process-fidelity estimate would be biased. "
                             "Rebuild it with fixed_identity_locations=None.")

        summary = self._compute_summary_statistics(data, energy=True)

        fidelities = []
        standard_errors = []
        for i, d in enumerate(design.depths):
            e = _np.asarray(summary['energies'][d], dtype=float)
            K = len(e)
            if K == 0:
                fidelities.append(float('nan'))
                standard_errors.append(float('nan'))
                continue
            # Each target's Paulis are drawn over its own line_labels, which can be narrower than
            # design.qubit_labels; a measurement has one entry per qubit of its target.
            n = len(design.measurements[i][0])
            fidelities.append(float(_mcfe.polarization_to_fidelity(e.mean(), n)))
            if K >= 2:
                standard_errors.append(float((1 - 4.0**-n) * e.std(ddof=1) / _np.sqrt(K)))
            else:
                standard_errors.append(float('nan'))

        return DFEResults(data, self, fidelities, standard_errors)


class DFEResults(_proto.ProtocolResults):
    """
    The process-fidelity estimates produced by running a :class:`DFE` protocol.

    Entry `i` of `fidelities` and of `standard_errors` belongs to target circuit
    `data.edesign.depths[i]`, which is that circuit's index in the `circuits` given to
    :class:`DFEDesign`. These indices stay correct after the design is truncated.

    Parameters
    ----------
    data : ProtocolData
        The experimental data these results are generated from.

    protocol_instance : Protocol
        The protocol that generated these results.

    fidelities : list of float
        The estimated process fidelity of each target circuit.

    standard_errors : list of float
        The standard error of each estimate (NaN when it can't be computed).
    """

    def __init__(self, data, protocol_instance, fidelities, standard_errors):
        super().__init__(data, protocol_instance)
        self.fidelities = [float(f) for f in fidelities]
        self.standard_errors = [float(s) for s in standard_errors]
