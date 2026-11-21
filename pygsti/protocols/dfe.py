"""
DFE Protocol objects
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numbers as _numbers
import warnings as _warnings

import numpy as _np

from pygsti.protocols import protocol as _proto
from pygsti.protocols import vb as _vb
from pygsti.algorithms import dfe as _dfe
from pygsti.tools import mcfetools as _mcfe


class DFEDesign(_vb.BenchmarkingDesign):
    """
    An experiment design for direct fidelity estimation of a set of Clifford circuits.

    For each circuit, `num_samples` DFE circuits are sampled with
    :func:`~pygsti.algorithms.dfe.sample_dfe_circuit`.

    The design's `qubit_labels` are the first circuit's `line_labels`. The :class:`DFE`
    protocol handles circuits on different qubits, but other machinery inherited from
    BenchmarkingDesign may assume that every circuit shares those labels. Building a
    dataframe from results on such a design warns, because its `Qubits` entry shows
    `qubit_labels` for every target.

    The reference circuits, if any, are stored in the attribute `reference_circuits` (None
    otherwise), and their kind in `reference`.

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

    seed : int, optional
        A seed for the random number generator, for reproducible sampling. If None, a seed
        is drawn from NumPy's global random state, as the RB designs do. The seed used is
        stored in the `seed` attribute, so the design can be regenerated from its saved
        metadata.

    reference : None, 'paired' or 'readout', optional
        `'paired'` adds, for each dressed circuit, a random eigenstate of the image of its Pauli,
        then the dressed circuit's own final layer; `'readout'` adds a uniformly random bitstring
        with the measured Z string's parity, then the measurement. Either is stored in
        `reference_circuits` (one list per target, paired with `circuit_lists`), recorded in
        `reference`, and doubles the number of circuits; :class:`DFE` then also reports a
        reference-corrected fidelity. Any other value raises ValueError.
    """
    def __init__(self, pspec, circuits, clifford_compilations, num_samples,
                 descriptor='A DFE experiment', add_default_protocol=False, seed=None,
                 reference=None):

        circuits = list(circuits)

        circuit_lists = []
        measurements = []
        signs = []
        reference_circuits = [] if reference is not None else None

        if seed is None:
            self.seed = _np.random.randint(1, 1e6)  # Pick a random seed
        elif isinstance(seed, _numbers.Integral):
            self.seed = int(seed)  # A NumPy integer can't be written to the design's JSON metadata
        else:
            self.seed = seed
        rand_state = _np.random.RandomState(self.seed)

        for c in circuits:
            dfe_circuits_for_c = []
            measurements_for_c = []
            signs_for_c = []
            references_for_c = []
            for sample_num in range(num_samples):
                sampled = _dfe.sample_dfe_circuit(
                    pspec, c, clifford_compilations,
                    reference=reference, seed=rand_state.randint(0, 2**31))
                if reference is not None:
                    dfe_circ, meas, sign, ref_circ = sampled
                    references_for_c.append(ref_circ)
                else:
                    dfe_circ, meas, sign = sampled
                dfe_circuits_for_c.append(dfe_circ)
                measurements_for_c.append(meas)
                signs_for_c.append(int(sign))

            circuit_lists.append(dfe_circuits_for_c)
            measurements.append(measurements_for_c)
            signs.append(signs_for_c)
            if reference is not None:
                reference_circuits.append(references_for_c)

        self._init_foundation(circuit_lists, measurements, signs, num_samples, descriptor,
                              add_default_protocol, reference=reference,
                              reference_circuits=reference_circuits)

    def _init_foundation(self, circuit_lists, measurements, signs, num_samples, descriptor,
                         add_default_protocol, depths=None, qubit_labels=None,
                         reference=None, reference_circuits=None):
        # Pair these attributes with circuit data so that we serialize/truncate properly
        self.paired_with_circuit_attrs = ["measurements", "signs"]
        if reference_circuits is not None:
            self.paired_with_circuit_attrs.append("reference_circuits")

        # By default the "depths" are the target indices 0..len-1, and the qubit labels are the
        # first circuit's line labels; map_qubit_labels passes both explicitly.
        if depths is None:
            depths = list(range(len(circuit_lists)))
        if qubit_labels is None:
            qubit_labels = circuit_lists[0][0].line_labels
        super().__init__(depths, circuit_lists, signs, qubit_labels, remove_duplicates=False)
        self.measurements = measurements
        self.signs = signs
        self.circuits_per_depth = num_samples
        self.descriptor = descriptor
        self.reference = reference
        self.reference_circuits = reference_circuits
        if reference_circuits is not None:
            # BenchmarkingDesign's 'json' type can't serialize circuits.
            self.auxfile_types['reference_circuits'] = 'list:text-circuit-list'
            # Like the dressed circuits, the references aren't deduplicated.
            self.all_circuits_needing_data = list(self.all_circuits_needing_data) \
                + [c for ref_list in reference_circuits for c in ref_list]

        if add_default_protocol:
            self.add_default_protocol(DFE(name='DFE'))

    def map_qubit_labels(self, mapper):
        """
        Creates a new experiment design whose circuits' qubit labels are updated according to a given mapping.

        The new design keeps the depths (target indices), measurements, signs,
        `seed`, `reference` and reference circuits (mapped) of this one.
        Like the RB designs, it has no default protocols: use `add_default_protocol(DFE(name='DFE'))` to get one.

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing self.qubit_labels values
            and whose value are the new labels, or a function which takes a
            single (existing qubit-label) argument and returns a new qubit-label.

        Returns
        -------
        DFEDesign
        """
        mapped_circuit_lists = [[c.map_state_space_labels(mapper) for c in circuit_list]
                                for circuit_list in self.circuit_lists]
        references = getattr(self, 'reference_circuits', None)
        if references is not None:
            references = [[c.map_state_space_labels(mapper) for c in ref_list] for ref_list in references]
        # Measurements are indexed by position in each circuit's line labels, so they stay valid.
        # The getattrs let designs saved before these attributes existed be mapped too.
        ret = self.__class__.__new__(self.__class__)
        ret._init_foundation(mapped_circuit_lists,
                             [list(m) for m in self.measurements],
                             [list(s) for s in self.signs],
                             self.circuits_per_depth, self.descriptor,
                             add_default_protocol=False,
                             depths=list(self.depths),
                             qubit_labels=self._mapped_qubit_labels(mapper),
                             reference=getattr(self, 'reference', None),
                             reference_circuits=references)
        ret.seed = getattr(self, 'seed', None)
        return ret

    def setup_nameddict(self, final_dict):
        """
        Initializes nested :class:`NamedDict` dictionaries describing this design.

        Warns when some targets' line labels differ from the design's `qubit_labels`,
        since the `Qubits` entry then shows `qubit_labels` for every target.

        Parameters
        ----------
        final_dict : NamedDict
            The dictionary that becomes the innermost level of the nested dictionaries.

        Returns
        -------
        NamedDict
        """
        design_labels = tuple(self.qubit_labels)
        mixed = []
        for circuit_list in self.circuit_lists:
            for c in circuit_list:
                labels = tuple(c.line_labels)
                if labels != design_labels and labels not in mixed:
                    mixed.append(labels)
        if mixed:
            _warnings.warn(
                f"Some targets' line labels {mixed} differ from the design's qubit_labels "
                f"{design_labels}, which are the first target's line labels. The `Qubits` entry "
                "of a dataframe or nested dict built from these results labels every target "
                "with the design's qubit_labels. The fidelities themselves are correct, one "
                "per target in target order.", UserWarning)
        return super().setup_nameddict(final_dict)


def _repeat_covariance(dataset, entries):
    """
    The shot covariance between different entries that read the same dataset row.

    A DataSet pools the shots of a repeated circuit into one row, so the sample variance of
    the entries' values misses the covariance that their shared shots induce. `entries` has
    one list per entry of `(circuit, coefficient, measurement, sign)` terms, and an entry's
    value is the sum of `coefficient * energy` over its terms. For each row that two or more
    entries read, this adds the variance of all the terms on the row, less the variance of
    each entry's own terms on it, each estimated from the row's counts as
    `(E[y^2] - E[y]^2) / (M - 1)` for `M` shots. Rows read by a single entry are skipped.
    The row must have at least 2 shots.
    """
    readers = {}
    for k, terms in enumerate(entries):
        for term in terms:
            readers.setdefault(term[0], {}).setdefault(k, []).append(term[1:])

    def variance(terms, row):
        m1 = 0.0
        m2 = 0.0
        for outcome, count in row.counts.items():
            y = sum(coef * _vb._outcome_energy(outcome[0], meas, sign) for coef, meas, sign in terms)
            m1 += count * y
            m2 += count * y * y
        total = row.total
        return (m2 / total - (m1 / total)**2) / (total - 1)

    covariance = 0.0
    for circuit, by_entry in readers.items():
        if len(by_entry) < 2:
            continue
        row = dataset[circuit]
        covariance += variance([t for terms in by_entry.values() for t in terms], row)
        covariance -= sum(variance(terms, row) for terms in by_entry.values())
    return covariance


class DFE(_vb.SummaryStatistics):
    """
    Direct fidelity estimation of the process fidelity of each target circuit in a DFEDesign.

    For a Clifford target on n qubits (its own `line_labels`), every qubit of a dressed circuit
    starts in a uniformly random eigenstate bit, so averaged over the bits the input is
    `(I + sign P) / 2^n`. Averaged over the bits and the Pauli's uniformly random sign, the mean
    energy of a dressed circuit is the diagonal entry for its Pauli of the target error's Pauli
    transfer matrix (for one sign, a non-unital error adds a term that flips with the sign). The
    default sampling is uniform over all 4^n - 1 non-identity Paulis, and the process fidelity
    is the mean of those entries, so the estimate is unbiased for any trace-preserving error on
    the target, not only stochastic Pauli errors (amplitude damping included). The mean energy
    `p` over a target's dressed circuits gives the process fidelity `1 - (4^n - 1)(1 - p)/4^n`,
    computed by :func:`~pygsti.tools.mcfetools.polarization_to_fidelity`. For n >= 20 that
    function returns `p` itself.

    The standard error is `(1 - 4^-n)` times the sample standard deviation of the energies,
    divided by the square root of the number of dressed circuits K. It folds shot noise and
    Pauli sampling into one empirical spread. A DataSet pools a repeated circuit's shots into one
    row, which hides their shot covariance from the sample variance, so the variance adds it
    back, estimated from those rows; repeats don't shrink it. It is zero when every energy is
    equal, as on noiseless data, and NaN when K is 1. With few samples, or when the Pauli
    fidelities take few distinct values, it can underestimate the true error.

    The estimate covers the whole dressed circuit: errors in the preparation and
    measurement-basis layers that dressing adds, and in state preparation and readout, are
    included.

    For a design with reference circuits (`reference='paired'` or `'readout'`), the results
    also hold `reference_corrected_fidelities` and `reference_corrected_standard_errors`. With
    `E` the dressed energies and `R` the energies of their paired references, the estimate is
    the process fidelity of `p = mean(E) / mean(R)`, with no square root. The vanilla fields
    ignore the references. Both kinds use this estimator, and readout error divides out on
    average, asymmetric readout included, because a dressed circuit and its reference both end,
    ideally and averaged over the draws, in the uniform distribution over bitstrings whose
    parity gives the sign. A paired reference also divides out its shared final layer fully,
    and the errors of state preparation and of the preparation layer in part, because it
    prepares a stabilizer state of the image of the sampled Pauli rather than that Pauli's own
    state. A readout reference shares only the measurement, so it leaves the errors of the
    dressing's preparation and final layers in the corrected estimate, and its circuits are
    shallow. The ratio of means weights each pair by its reference energy, which overstates the
    fidelity slightly under large readout error. Values above 1 or below 0 aren't clamped, and a
    non-positive mean reference energy gives NaN with a warning. The standard error is the
    delta method's for the ratio of paired means,
    `(1 - 4^-n) std(E - p R) / (sqrt(K) mean(R))`, with the shot covariance of repeated dressed
    or reference circuits added back to the variance of the paired differences, and has the
    same caveats as the vanilla one.

    Designs that aren't :class:`DFEDesign` objects are refused with TypeError, since other
    benchmarking designs (e.g. `BinaryRBDesign`) carry similar attributes but would give
    meaningless numbers. ValueError is raised for data with no dataset, for a dataset
    that lacks counts for some dressed or reference circuit (missing or zero-count rows), for
    a target whose repeated dressed or reference circuit has fewer than 2 shots, and for
    a design whose `measurements`, `signs` or reference circuits don't line up with its circuit
    lists.

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
        self._check_alignment(design)
        self._check_data_complete(design, data.dataset)
        self._check_repeat_shots(design, data.dataset)

        summary = self._compute_summary_statistics(data, energy=True)

        references = getattr(design, 'reference_circuits', None)
        fidelities = []
        standard_errors = []
        ref_fidelities = [] if references is not None else None
        ref_standard_errors = [] if references is not None else None
        nonpositive = []
        for i, d in enumerate(design.depths):
            e = _np.asarray(summary['energies'][d], dtype=float)
            K = len(e)
            if K == 0:
                fidelities.append(float('nan'))
                standard_errors.append(float('nan'))
                if references is not None:
                    ref_fidelities.append(float('nan'))
                    ref_standard_errors.append(float('nan'))
                continue
            # Each target's Paulis are drawn over its own line_labels, which can be narrower than
            # design.qubit_labels; a measurement has one entry per qubit of its target.
            n = len(design.measurements[i][0])
            fidelities.append(float(_mcfe.polarization_to_fidelity(e.mean(), n)))
            if K >= 2:
                repeats = _repeat_covariance(
                    data.dataset, [[(c, 1.0, m, s)] for c, m, s in
                                   zip(design.circuit_lists[i], design.measurements[i], design.signs[i])])
                var = (_np.sum((e - e.mean())**2) + repeats) / (K - 1)
                standard_errors.append(float((1 - 4.0**-n) * _np.sqrt(var / K)))
            else:
                standard_errors.append(float('nan'))

            if references is not None:
                r = _np.array([_vb._avg_energy(data.dataset[c], m, s)
                               for c, m, s in zip(references[i], design.measurements[i], design.signs[i])],
                              dtype=float)
                if not r.mean() > 0:
                    ref_fidelities.append(float('nan'))
                    ref_standard_errors.append(float('nan'))
                    nonpositive.append(d)
                    continue
                p = e.mean() / r.mean()
                ref_fidelities.append(float(_mcfe.polarization_to_fidelity(p, n)))
                if K >= 2:
                    d_ = e - p * r
                    repeats = _repeat_covariance(
                        data.dataset, [[(c, 1.0, m, s), (ref, -p, m, s)] for c, ref, m, s in
                                       zip(design.circuit_lists[i], references[i],
                                           design.measurements[i], design.signs[i])])
                    var = (_np.sum((d_ - d_.mean())**2) + repeats) / (K - 1)
                    ref_standard_errors.append(float((1 - 4.0**-n) * _np.sqrt(var / K) / r.mean()))
                else:
                    ref_standard_errors.append(float('nan'))

        if nonpositive:
            _warnings.warn(f"Targets {nonpositive} have a non-positive mean reference energy, so their "
                           "reference-corrected fidelities and standard errors are NaN.", UserWarning)

        return DFEResults(data, self, fidelities, standard_errors, ref_fidelities, ref_standard_errors)

    @staticmethod
    def _check_alignment(design):
        # The paired measurements and signs must match the circuit lists entry for entry, or the
        # energies would be computed against the wrong measurements.
        n_lists = len(design.circuit_lists)
        references = getattr(design, 'reference_circuits', None)
        refs_aligned = references is None or (
            len(references) == n_lists and all(len(r) == len(cl) for cl, r in zip(design.circuit_lists, references)))
        aligned = (refs_aligned and len(design.depths) == n_lists and len(design.measurements) == n_lists
                   and len(design.signs) == n_lists
                   and all(len(m) == len(cl) and len(s) == len(cl)
                           for cl, m, s in zip(design.circuit_lists, design.measurements, design.signs)))
        if not aligned:
            raise ValueError("This design's depths, measurements, signs or reference circuits don't line up "
                             "with its circuit lists. "
                             "A truncation that removes every dressed circuit of some target causes this (a known "
                             "bug in BenchmarkingDesign's truncation). Whole targets can be dropped safely with "
                             "truncate_to_lists.")

    @staticmethod
    def _check_repeat_shots(design, ds):
        # The repeat covariance divides by (shots - 1), so a repeated circuit needs 2 or more.
        references = getattr(design, 'reference_circuits', None)
        for i, circuit_list in enumerate(design.circuit_lists):
            counts = {}
            for c in circuit_list:
                counts[c] = counts.get(c, 0) + 1
            if references is not None:
                for c in references[i]:
                    counts[c] = counts.get(c, 0) + 1
            if any(n >= 2 and ds[c].total < 2 for c, n in counts.items()):
                raise ValueError(
                    f"Target {design.depths[i]} repeats a dressed or reference circuit whose data has fewer "
                    "than 2 shots, so its standard error can't account for the repeat. Use 2 or more shots "
                    "per circuit.")

    @staticmethod
    def _check_data_complete(design, ds):
        if ds is None:
            raise ValueError("The data has no dataset; DFE needs counts for every dressed circuit of the design.")
        distinct = set(c for circuit_list in design.circuit_lists for c in circuit_list)
        references = getattr(design, 'reference_circuits', None)
        if references is not None:
            distinct.update(c for ref_list in references for c in ref_list)
        n_missing = 0
        n_zero = 0
        for c in distinct:
            if c not in ds:
                n_missing += 1
            elif ds[c].total == 0:
                n_zero += 1
        if n_missing + n_zero > 0:
            msg = (f"The dataset lacks counts for {n_missing + n_zero} of the design's {len(distinct)} "
                   f"distinct circuits ({n_missing} missing, {n_zero} with zero counts). To "
                   "analyze the available data, remove any zero-count rows from the dataset, then "
                   "truncate the design with design.truncate_to_available_data(dataset). Truncation "
                   "alone doesn't drop zero-count rows, because the dataset's keys include them.")
            if references is not None:
                msg += (" Truncation filters on the dressed circuits, so it keeps a dressed circuit whose "
                        "reference lacks counts: remove that dressed circuit's row from the dataset "
                        "before truncating.")
            raise ValueError(msg)


class DFEResults(_proto.ProtocolResults):
    """
    The process-fidelity estimates produced by running a :class:`DFE` protocol.

    Entry `i` of `fidelities` and of `standard_errors` belongs to target circuit
    `data.edesign.depths[i]`, which is that circuit's index in the `circuits` given to
    :class:`DFEDesign`. The indices stay correct after `truncate_to_lists`, and after a
    truncation to circuits or data that leaves every target at least one dressed circuit. A
    truncation that removes all of a target's dressed circuits misaligns the design's
    `measurements` and `signs` (a bug in `BenchmarkingDesign`), and :class:`DFE` refuses such
    a design.

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

    reference_corrected_fidelities : list of float, optional
        The reference-corrected process fidelity of each target circuit, entry `i` belonging
        to target `depths[i]` like `fidelities`. None when the design has no reference circuits.

    reference_corrected_standard_errors : list of float, optional
        The delta-method standard error of each reference-corrected fidelity (NaN when it
        can't be computed). None when the design has no reference circuits.
    """

    def __init__(self, data, protocol_instance, fidelities, standard_errors,
                 reference_corrected_fidelities=None, reference_corrected_standard_errors=None):
        super().__init__(data, protocol_instance)
        self.fidelities = [float(f) for f in fidelities]
        self.standard_errors = [float(s) for s in standard_errors]
        self.reference_corrected_fidelities = (
            None if reference_corrected_fidelities is None
            else [float(f) for f in reference_corrected_fidelities])
        self.reference_corrected_standard_errors = (
            None if reference_corrected_standard_errors is None
            else [float(s) for s in reference_corrected_standard_errors])
