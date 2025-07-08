"""
Volumetric Benchmarking Protocol objects
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import copy as _copy

from pygsti import tools as _tools
from pygsti.algorithms import randomcircuit as _rc
from pygsti.protocols import protocol as _proto
from pygsti.models.oplessmodel import SuccessFailModel as _SuccessFailModel


class ByDepthDesign(_proto.CircuitListsDesign):
    """
    Experiment design that holds circuits organized by depth.

    Parameters
    ----------
    depths : list or tuple
        A sequence of integers specifying the circuit depth associated with each
        element of `circuit_lists`.

    circuit_lists : list or tuple
        The circuits to include in this experiment design.  Each element is a
        list of :class:`Circuits` specifying the circuits at the corresponding depth.

    qubit_labels : tuple, optional
        The qubits that this experiment design applies to. If None, the
        line labels of the first circuit is used.

    remove_duplicates : bool, optional
        Whether to remove duplicates when automatically creating
        all the circuits that need data.
    """

    def __init__(self, depths, circuit_lists, qubit_labels=None, remove_duplicates=True):
        assert(len(depths) == len(circuit_lists)), \
            "Number of depths must equal the number of circuit lists!"
        super().__init__(circuit_lists, qubit_labels=qubit_labels, remove_duplicates=remove_duplicates)
        self.depths = depths

    def map_qubit_labels(self, mapper):
        """
        Creates a new experiment design whose circuits' qubit labels are updated according to a given mapping.

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing self.qubit_labels values
            and whose value are the new labels, or a function which takes a
            single (existing qubit-label) argument and returns a new qubit-label.

        Returns
        -------
        ByDepthDesign
        """
        mapped_circuit_lists = [[c.map_state_space_labels(mapper) for c in circuit_list]
                                for circuit_list in self.circuit_lists]
        mapped_qubit_labels = self._mapped_qubit_labels(mapper)
        return ByDepthDesign(self.depths, mapped_circuit_lists, mapped_qubit_labels, remove_duplicates=False)

    def truncate_to_lists(self, list_indices_to_keep):
        """
        Truncates this experiment design by only keeping a subset of its circuit lists.

        Parameters
        ----------
        list_indices_to_keep : iterable
            A list of the (integer) list indices to keep.

        Returns
        -------
        ByDepthDesign
            The truncated experiment design.
        """
        ret = _copy.deepcopy(self) # Works for derived classes too
        ret.depths = [self.depths[i] for i in list_indices_to_keep]
        ret.circuit_lists = [self.circuit_lists[i] for i in list_indices_to_keep]
        return ret

    def merge_with(self, other_edesign, remove_duplicates=True, sort_depths=False):
        """
        Merge this experiment design with another one and return the result.

        Parameters
        ----------
        other_edesign: ByDepthDesign
            The other experiment design to merge

        Returns
        -------
        ByDepthDesign
        """
        assert self.qubit_labels == other_edesign.qubit_labels, "To merge, qubit_labels must be equal"

        depths = []
        circuits_by_depth = {}
        for obj in (self, other_edesign):
            for d, cl in zip(obj.depths, obj.circuit_lists):
                if d in circuits_by_depth:
                    circuits_by_depth[d].extend(cl)
                else:
                    depths.append(d)
                    circuits_by_depth[d] = list(cl)

        if sort_depths:
            depths = sorted(depths)
        
        circuit_lists = [circuits_by_depth[d] for d in depths]
        return ByDepthDesign(depths, circuit_lists, self.qubit_labels, remove_duplicates=remove_duplicates)


class BenchmarkingDesign(ByDepthDesign):
    """
    Experiment design that holds benchmarking data.

    By "benchmarking data" we mean definite-outcome circuits organized
    by depth along with their corresponding ideal outcomes.

    Parameters
    ----------
    depths : list or tuple
        A sequence of integers specifying the circuit depth associated with each
        element of `circuit_lists`.

    circuit_lists : list or tuple
        The circuits to include in this experiment design.  Each element is a
        list of :class:`Circuits` specifying the circuits at the corresponding depth.

    ideal_outs : list or tuple
        The ideal circuit outcomes corresponding to the circuits in `circuit_lists`.
        Each element of `ideal_outs` is a list (with the same length as the corresponding
        `circuits_lists` element) of outcome labels.

    qubit_labels : tuple, optional
        The qubits that this experiment design applies to. If None, the
        line labels of the first circuit is used.

    remove_duplicates : bool, optional
        Whether to remove duplicates when automatically creating
        all the circuits that need data.
    """
    
    paired_with_circuit_attrs = None
    """List of attributes which are paired up with circuit lists

    These will be saved as external files during serialization,
    and are truncated when circuit lists are truncated.
    """

    def __init__(self, depths, circuit_lists, ideal_outs, qubit_labels=None, remove_duplicates=False):
        assert(len(depths) == len(ideal_outs))
        super().__init__(depths, circuit_lists, qubit_labels, remove_duplicates)
        
        self.idealout_lists = ideal_outs

        if self.paired_with_circuit_attrs is None:
            self.paired_with_circuit_attrs = ['idealout_lists']
        else:
            self.paired_with_circuit_attrs.insert(0, 'idealout_lists')
        
        for paired_attr in self.paired_with_circuit_attrs:
            self.auxfile_types[paired_attr] = 'json'

    def _mapped_circuits_and_idealouts_by_depth(self, mapper):
        """ Used in derived classes """
        mapped_circuits_and_idealouts_by_depth = {}
        for depth, circuit_list, idealout_list in zip(self.depths, self.circuit_lists, self.idealout_lists):
            mapped_circuits_and_idealouts_by_depth[depth] = \
                [(c.map_state_space_labels(mapper), iout) for c, iout in zip(circuit_list, idealout_list)]
        return mapped_circuits_and_idealouts_by_depth

    def map_qubit_labels(self, mapper):
        """
        Creates a new experiment design whose circuits' qubit labels are updated according to a given mapping.

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing self.qubit_labels values
            and whose value are the new labels, or a function which takes a
            single (existing qubit-label) argument and returns a new qubit-label.

        Returns
        -------
        ByDepthDesign
        """
        mapped_circuit_lists = [[c.map_state_space_labels(mapper) for c in circuit_list]
                                for circuit_list in self.circuit_lists]
        mapped_qubit_labels = self._mapped_qubit_labels(mapper)
        return BenchmarkingDesign(self.depths, mapped_circuit_lists, list(self.idealout_lists),
                                  mapped_qubit_labels, remove_duplicates=False)
    
    def truncate_to_lists(self, list_indices_to_keep):
        """
        Truncates this experiment design by only keeping a subset of its circuit lists.

        Parameters
        ----------
        list_indices_to_keep : iterable
            A list of the (integer) list indices to keep.

        Returns
        -------
        BenchmarkingDesign
            The truncated experiment design.
        """
        ret = _copy.deepcopy(self) # Works for derived classes too
        ret.depths = [self.depths[i] for i in list_indices_to_keep]
        ret.circuit_lists = [self.circuit_lists[i] for i in list_indices_to_keep]
        for paired_attr in self.paired_with_circuit_attrs:
            val = getattr(self, paired_attr)
            new_val = [val[i] for i in list_indices_to_keep]
            setattr(ret, paired_attr, new_val)
        return ret

    def _truncate_to_circuits_inplace(self, circuits_to_keep):
        truncated_circuit_lists = []
        paired_attr_lists_list = [getattr(self, paired_attr) for paired_attr in self.paired_with_circuit_attrs]
        truncated_paired_attr_lists_list = [[] for _ in range(len(self.paired_with_circuit_attrs))]
        for list_idx, circuits in enumerate(self.circuit_lists):
            paired_attrs = [pal[list_idx] for pal in paired_attr_lists_list]
            # Do the same filtering as CircuitList.truncate, but drag along any paired attributes
            new_data = list(zip(*filter(lambda ci: ci[0] in set(circuits_to_keep), zip(circuits, *paired_attrs))))
            if len(new_data):
                truncated_circuit_lists.append(new_data[0])
                for i, attr_data in enumerate(new_data[1:]):
                    truncated_paired_attr_lists_list[i].append(attr_data)
            else:
                # If we have truncated all circuits, append empty lists
                truncated_circuit_lists.append([])
                truncated_paired_attr_lists_list.append([[] for _ in range(len(self.paired_with_circuit_attrs))])

        self.circuit_lists = truncated_circuit_lists
        for paired_attr, paired_attr_lists in zip(self.paired_with_circuit_attrs, truncated_paired_attr_lists_list):
            setattr(self, paired_attr, paired_attr_lists)
        super()._truncate_to_circuits_inplace(circuits_to_keep)

    def _truncate_to_design_inplace(self, other_design):
        truncated_circuit_lists = []
        paired_attr_lists_list = [getattr(self, paired_attr) for paired_attr in self.paired_with_circuit_attrs]
        truncated_paired_attr_lists_list = [[] for _ in range(len(self.paired_with_circuit_attrs))]
        for list_idx, circuits in enumerate(self.circuit_lists):
            paired_attrs = [pal[list_idx] for pal in paired_attr_lists_list]
            # Do the same filtering as CircuitList.truncate, but drag along any paired attributes
            new_data = list(zip(*filter(lambda ci: ci[0] in set(other_design.circuit_lists[list_idx]), zip(circuits, *paired_attrs))))
            if len(new_data):
                truncated_circuit_lists.append(new_data[0])
                for i, attr_data in enumerate(new_data[1:]):
                    truncated_paired_attr_lists_list[i].append(attr_data)
            else:
                # If we have truncated all circuits, append empty lists
                truncated_circuit_lists.append([])
                truncated_paired_attr_lists_list.append([[] for _ in range(len(self.paired_with_circuit_attrs))])

        self.circuit_lists = truncated_circuit_lists
        for paired_attr, paired_attr_lists in zip(self.paired_with_circuit_attrs, truncated_paired_attr_lists_list):
            setattr(self, paired_attr, paired_attr_lists)
        super()._truncate_to_design_inplace(other_design)

    def _truncate_to_available_data_inplace(self, dataset):
        self._truncate_to_circuits_inplace(set(dataset.keys()))

    def _merge_data(self, other_edesign, sort_depths):
        assert self.paired_with_circuit_attrs == other_edesign.paired_with_circuit_attrs, \
            "To merge, `paired_with_circuit_attrs` must be equal"

        depths = []
        circuits_by_depth = {}
        paired_attrs_by_depth = {nm: {} for nm in self.paired_with_circuit_attrs}

        for obj in (self, other_edesign):
            for i, (d, cl) in enumerate(zip(obj.depths, obj.circuit_lists)):
                if d in circuits_by_depth:
                    circuits_by_depth[d].extend(cl)
                    for paired_attr in self.paired_with_circuit_attrs:
                        val = getattr(obj, paired_attr)  # e.g. self.idealout_lists
                        paired_attrs_by_depth[paired_attr][d].extend(val[i])

                else:
                    depths.append(d)
                    circuits_by_depth[d] = list(cl)
                    for paired_attr in self.paired_with_circuit_attrs:
                        val = getattr(obj, paired_attr)  # e.g. self.idealout_lists
                        paired_attrs_by_depth[paired_attr][d] = list(val[i])

        if sort_depths:
            depths = sorted(depths)
        return depths, circuits_by_depth, paired_attrs_by_depth

    def merge_with(self, other_edesign, remove_duplicates=True, sort_depths=False):
        """
        Merge this experiment design with another one and return the result.

        The returned design will contain the union of the circuits in the two
        experiment designs being merged, and will contain depth, ideal-output,
        etc. metadata that is updated appropriately.

        Parameters
        ----------
        other_edesign: BenchmarkingDesign
            The other experiment design to merge

        Returns
        -------
        BenchmarkingDesign
        """
        assert self.qubit_labels == other_edesign.qubit_labels, "To merge, qubit_labels must be equal"
        depths, circuits_by_depth, paired_attrs_by_depth = self._merge_data(other_edesign, sort_depths)

        circuit_lists = [circuits_by_depth[d] for d in depths]
        idealouts_per_depth = paired_attrs_by_depth.pop('idealout_lists')
        ideal_outs = [idealouts_per_depth[d] for d in depths]
        return BenchmarkingDesign(depths, circuit_lists, ideal_outs, self.qubit_labels,
                                  remove_duplicates=remove_duplicates)



class PeriodicMirrorCircuitDesign(BenchmarkingDesign):
    """
    Experiment design for periodic mirror-circuit benchmarking.

    THIS METHOD IS IN DEVELOPEMENT. DO NOT EXPECT THAT THIS FUNCTION WILL BEHAVE THE SAME IN FUTURE RELEASES
    OF PYGSTI! THE DOCSTRINGS SHOULD ALSO NOT BE TRUSTED -- MANY (MAYBE ALL) OF THEM ARE COPIED FROM THE
    MIRRORBDESIGN OBJECT AND SO SOME BITS ARE WRONG OR NOT APPLICABLE.

    Parameters
    ----------
    pspec : QubitProcessorSpec
       The QubitProcessorSpec for the device that the experiment is being generated for. The `pspec` is always
       handed to the sampler, as the first argument of the sampler function.

    depths : list of ints
        The "mirror RB depths" of the circuits, which is closely related to the circuit depth. A MRB
        length must be an even integer, and can be zero.

        * If `localclifford` and `paulirandomize` are False, the depth of a sampled circuit = the MRB length.
          The first length/2 layers are all sampled independently according to the sampler specified by
          `sampler`. The remaining half of the circuit is the "inversion" circuit that is determined
          by the first half.
        * If `paulirandomize` is True and `localclifford` is False, the depth of a circuit is
          `2*length+1` with odd-indexed layers sampled according to the sampler specified by `sampler`, and
          the the zeroth layer + the even-indexed layers consisting of random 1-qubit Pauli gates.
        * If `paulirandomize` and `localclifford` are True, the depth of a circuit is
          `2*length+1 + X` where X is a random variable (between 0 and normally `<= ~12-16`) that accounts for
          the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.
        * If `paulirandomize` is False and `localclifford` is True, the depth of a circuit is
          length + X where X is a random variable (between 0 and normally `<= ~12-16`) that accounts for
          the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

    circuits_per_depth : int
        The number of (possibly) different MRB circuits sampled at each length.

    qubit_labels : list, optional
        If not None, a list of the qubits that the RB circuit is to be sampled for. This should
        be all or a subset of the qubits in the device specified by the QubitProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the "wires" in the returned circuit, but is otherwise
        irrelevant.

    sampler : str or function, optional
        If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function
        in rb.sampler named `circuit_layer_by*` (with `*` replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
        a valid option for n-qubit MRB -- it results in sim. 1-qubit MRB -- but it is not explicitly
        forbidden by this function]. If `sampler` is a function, it should be a function that takes
        as the first argument a QubitProcessorSpec, and returns a random circuit layer as a list of gate
        Label objects. Note that the default 'Qelimination' is not necessarily the most useful
        in-built sampler, but it is the only sampler that requires no parameters beyond the QubitProcessorSpec
        *and* works for arbitrary connectivity devices. See the docstrings for each of these samplers
        for more information.

    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec` and `samplerargs` lists the
        remaining arguments handed to the sampler.

    localclifford : bool, optional
        Whether to start the circuit with uniformly random 1-qubit Cliffords and all of the
        qubits (compiled into the native gates of the device).

    paulirandomize : bool, optional
        Whether to have uniformly random Pauli operators on all of the qubits before and
        after all of the layers in the "out" and "back" random circuits. At length 0 there
        is a single layer of random Pauli operators (in between two layers of 1-qubit Clifford
        gates if `localclifford` is True); at length l there are 2l+1 Pauli layers as there
        are

    fixed_versus_depth : bool, optional
        <TODO description>

    descriptor : str, optional
        A string describing the generated experiment. Stored in the returned dictionary.
    """
    @classmethod
    def from_existing_circuits(cls, circuits_and_idealouts_by_depth, qubit_labels=None,
                               sampler='edgegrab', samplerargs=(0.125,), localclifford=True,
                               paulirandomize=True, fixed_versus_depth=False,
                               descriptor='A random germ mirror circuit experiment'):
        """
        Create a :class:`PeriodicMirrorCircuitDesign` from an existing set of sampled RB circuits.

        This function serves as an alternative to the usual method of creating a mirror
        RB experiment design by sampling a number of circuits randomly.  This function
        takes a list of previously-sampled random circuits and does not sampling internally.

        Parameters
        ----------
        circuits_and_idealouts_by_depth : dict
            A dictionary whose keys are integer depths and whose values are lists
            of `(circuit, ideal_outcome)` 2-tuples giving each RB circuit and its
            ideal (correct) outcome.

        qubit_labels : list, optional
            If not None, a list of the qubits that the RB circuit is to be sampled for. This should
            be all or a subset of the qubits in the device specified by the QubitProcessorSpec `pspec`.
            If None, it is assumed that the RB circuit should be over all the qubits. Note that the
            ordering of this list is the order of the "wires" in the returned circuit, but is otherwise
            irrelevant.

        sampler : str or function, optional
            If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
            Except for 'local', this corresponds to sampling layers according to the sampling function
            in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
            corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
            a valid option for n-qubit MRB -- it results in sim. 1-qubit MRB -- but it is not explicitly
            forbidden by this function]. If `sampler` is a function, it should be a function that takes
            as the first argument a QubitProcessorSpec, and returns a random circuit layer as a list of gate
            Label objects. Note that the default 'Qelimination' is not necessarily the most useful
            in-built sampler, but it is the only sampler that requires no parameters beyond the QubitProcessorSpec
            *and* works for arbitrary connectivity devices. See the docstrings for each of these samplers
            for more information.

        samplerargs : list, optional
            A list of arguments that are handed to the sampler function, specified by `sampler`.
            The first argument handed to the sampler is `pspec` and `samplerargs` lists the
            remaining arguments handed to the sampler.

        localclifford : bool, optional
            Whether to start the circuit with uniformly random 1-qubit Cliffords and all of the
            qubits (compiled into the native gates of the device).

        paulirandomize : bool, optional
            Whether to have uniformly random Pauli operators on all of the qubits before and
            after all of the layers in the "out" and "back" random circuits. At length 0 there
            is a single layer of random Pauli operators (in between two layers of 1-qubit Clifford
            gates if `localclifford` is True); at length l there are 2l+1 Pauli layers as there
            are

        fixed_versus_depth : bool, optional
            <TODO description>

        descriptor : str, optional
            A string describing the generated experiment. Stored in the returned dictionary.

        Returns
        -------
        PeriodicMirrorCircuitDesign
        """
        depths = sorted(list(circuits_and_idealouts_by_depth.keys()))
        circuit_lists = [[x[0] for x in circuits_and_idealouts_by_depth[d]] for d in depths]
        ideal_outs = [[x[1] for x in circuits_and_idealouts_by_depth[d]] for d in depths]
        circuits_per_depth = [len(circuits_and_idealouts_by_depth[d]) for d in depths]
        self = cls.__new__(cls)
        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              sampler, samplerargs, localclifford, paulirandomize, fixed_versus_depth, descriptor)
        return self

    def __init__(self, pspec, depths, circuits_per_depth, qubit_labels=None, clifford_compilations=None,
                 sampler='edgegrab', samplerargs=(0.125,),
                 localclifford=True, paulirandomize=True, fixed_versus_depth=False,
                 descriptor='A random germ mirror circuit experiment', seed=None):

        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = [[] for d in depths]
        ideal_outs = [[] for d in depths]

        assert(clifford_compilations is not None)
        abs_clifford_compilations = clifford_compilations['absolute']

        if seed is None:
            self.seed = _np.random.randint(1, 1e6)  # Pick a random seed
        else:
            self.seed = seed

        for j in range(circuits_per_depth):
            circtemp, outtemp, junk = _rc.create_random_germpower_mirror_circuits(
                pspec, abs_clifford_compilations, depths, qubit_labels=qubit_labels, localclifford=localclifford,
                paulirandomize=paulirandomize, interacting_qs_density=samplerargs[0],
                fixed_versus_depth=fixed_versus_depth, seed=seed)
            for ind in range(len(depths)):
                circuit_lists[ind].append(circtemp[ind])
                ideal_outs[ind].append((''.join(map(str, outtemp[ind])),))

        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              sampler, samplerargs, localclifford, paulirandomize, fixed_versus_depth, descriptor, seed=seed)

    def _init_foundation(self, depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                         sampler, samplerargs, localclifford, paulirandomize, fixed_versus_depth, descriptor, seed=None):
        super().__init__(depths, circuit_lists, ideal_outs, qubit_labels, remove_duplicates=False)
        self.circuits_per_depth = circuits_per_depth
        self.descriptor = descriptor
        self.sampler = sampler
        self.samplerargs = samplerargs
        self.localclifford = localclifford
        self.paulirandomize = paulirandomize
        self.fixed_versus_depth = fixed_versus_depth
        self.seed = seed

    def map_qubit_labels(self, mapper):
        """
        Creates a new experiment design whose circuits' qubit labels are updated according to a given mapping.

        Parameters
        ----------
        mapper : dict or function
            A dictionary whose keys are the existing self.qubit_labels values
            and whose value are the new labels, or a function which takes a
            single (existing qubit-label) argument and returns a new qubit-label.

        Returns
        -------
        PeriodicMirrorCircuitDesign
        """
        mapped_circuits_and_idealouts_by_depth = self._mapped_circuits_and_idealouts_by_depth(mapper)
        mapped_qubit_labels = self._mapped_qubit_labels(mapper)
        return PeriodicMirrorCircuitDesign.from_existing_circuits(mapped_circuits_and_idealouts_by_depth,
                                                                  mapped_qubit_labels,
                                                                  self.sampler, self.samplerargs, self.localclifford,
                                                                  self.paulirandomize, self.fixed_versus_depth,
                                                                  self.descriptor)


class SummaryStatistics(_proto.Protocol):
    """
    A protocol that can construct "summary" quantities from raw data.

    Parameters
    ----------
    name : str
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.

    Attributes
    ----------
    summary_statistics : tuple
        Static list of the categories of summary information this protocol can compute.

    circuit_statistics : tuple
        Static list of the categories of circuit information this protocol can compute.
    """
    summary_statistics = ('success_counts', 'total_counts', 'hamming_distance_counts',
                          'success_probabilities', 'polarization', 'adjusted_success_probabilities', 'energies')
    circuit_statistics = ('two_q_gate_count', 'depth', 'idealout', 'circuit_index', 'width')
    # dscmp_statistics = ('tvds', 'pvals', 'jsds', 'llrs', 'sstvds')

    def __init__(self, name):
        super().__init__(name)

    def _compute_summary_statistics(self, data, energy = False):
        """
        Computes all summary statistics for the given data.

        Parameters
        ----------
        data : ProtocolData
            The data to operate on.

        Returns
        -------
        NamedDict
        """
        
        def outcome_energy(outcome, measurement, sign):
            """
            Computes the result of a Pauli measurement from a computational basis outcome
            Parameters
            ----------
            outcome: str
                A string of '0's and '1's, representing a measurement outcome

            measurement: str
                A string of 'I's and 'Z's, representing the target Pauli measurement, 

            sign: int
                The sign of the target measurement (either 1 or -1)

            Returns
            -------
            int
                The Pauli measurement result
            """
            energy = 1
            for i,j in zip(outcome,measurement):
                if i == '1' and j == 'Z':
                    energy = -1*energy
            return sign*energy

        def avg_energy(dsrow, measurement, sign):
            """Computes the result of a Pauli measurement from counts of computational basis measurements
            Parameters
            ----------
            dsrow: DataSetRow
                The data to operate on

            measurement: str
                A string of 'I's and 'Z's, representing the target Pauli measurement, 

            sign: int
                The sign of the target measurement (either 1 or -1)

            Returns
            -------
            float
                The Pauli measurement result
            """
            energy = 0
            for i in dsrow.counts:
                out_eng = outcome_energy(i[0],measurement,sign)
                energy += dsrow.counts[i] * out_eng    
            return energy / dsrow.total
        
        def success_counts(dsrow, circ, idealout):
            if dsrow.total == 0: return 0  # shortcut?
            return dsrow.get(tuple(idealout), 0.)

        def hamming_distance_counts(dsrow, circ, idealout):
            nQ = len(circ.line_labels)  # number of qubits
            assert(nQ == len(idealout[-1]))
            hamming_distance_counts = _np.zeros(nQ + 1, float)
            if dsrow.total > 0:
                for outcome_lbl, counts in dsrow.counts.items():
                    outbitstring = outcome_lbl[-1]
                    hamming_distance_counts[_tools.rbtools.hamming_distance(outbitstring, idealout[-1])] += counts
            return hamming_distance_counts

        def adjusted_success_probability(hamming_distance_counts):

            """ A scaled success probability that is useful for mirror circuit benchmarks """
            if _np.sum(hamming_distance_counts) == 0.:
                return 0.
            else:
                hamming_distance_pdf = _np.array(hamming_distance_counts) / _np.sum(hamming_distance_counts)
                adjSP = _np.sum([(-1 / 2)**n * hamming_distance_pdf[n] for n in range(len(hamming_distance_pdf))])
                return adjSP
            
        def _get_energies(icirc, circ, dsrow, measurement, sign):
            eng = avg_energy(dsrow, measurement, sign)
            ret = {'energies': eng, 'total_counts': dsrow.total}
            return ret

        def _get_summary_values(icirc, circ, dsrow, idealout):
            sc = success_counts(dsrow, circ, idealout)
            tc = dsrow.total
            hdc = hamming_distance_counts(dsrow, circ, idealout)
            sp = _np.nan if tc == 0 else sc / tc
            nQ = len(circ.line_labels)
            pol = (sp - 1 / 2**nQ) / (1 - 1 / 2**nQ)
            ret = {'success_counts': sc,
                   'total_counts': tc,
                   'success_probabilities': sp,
                   'polarization': pol,
                   'hamming_distance_counts': hdc,
                   'adjusted_success_probabilities': adjusted_success_probability(hdc)}
            return ret
        
        if energy is False:
            return self._compute_dict(data, self.summary_statistics,
                                  _get_summary_values, for_passes='all')
        
        else:
            return self._compute_dict(data, ['energies',  'total_counts'],
                                     _get_energies, for_passes = 'all', energy = True)
    
    def _compute_circuit_statistics(self, data):
        """
        Computes all circuit statistics for the given data.

        Parameters
        ----------
        data : ProtocolData
            The data to operate on.

        Returns
        -------
        NamedDict
        """
        def _get_circuit_values(icirc, circ, dsrow, idealout):
            ret = {'two_q_gate_count': circ.two_q_gate_count(),
                   'depth': circ.depth,
                   'idealout': idealout,
                   'circuit_index': icirc,
                   'width': len(circ.line_labels)}
            ret.update(dsrow.aux)  # note: will only get aux data from *first* pass in multi-pass data
            return ret

        return self._compute_dict(data, self.circuit_statistics, _get_circuit_values, for_passes="first")

    def _compute_predicted_probs(self, data, model):
        """
        Compute the predicted success probabilities of `model` given `data`.

        Parameters
        ----------
        data : ProtocolData
            The data.

        model : SuccessFailModel
            The model.

        Returns
        -------
        NamedDict
        """
        def _get_success_prob(icirc, circ, dsrow, idealout):
            #if set(circ.line_labels) != set(qubits):
            #    trimmedcirc = circ.copy(editable=True)
            #    for q in circ.line_labels:
            #        if q not in qubits:
            #            trimmedcirc.delete_lines(q)
            #else:
            #    trimmedcirc = circ
            return {'success_probabilities': model.probabilities(circ)[('success',)]}

        return self._compute_dict(data, ('success_probabilities',),
                                  _get_success_prob, for_passes="none")

    def _compute_dict(self, data, component_names, compute_fn, for_passes="all", energy = False):
        """
        Executes a computation function row-by-row on the data in `data` and packages the results.

        Parameters
        ----------
        data : ProtocolData
            The data.

        component_names : list or tuple
            A sequence of string-valued component names which must be the keys of the dictionary
            returned by `compute_fn`.

        compute_fn : function
            A function that computes values for each item in `component_names` for each row of data.
            This function should have signature:
            `compute_fn(icirc : int, circ : Circuit, dsrow : _DataSetRow, idealout : OutcomeLabel)`
            and should return a dictionary whose keys are the same as `component_names`.

        for_passes : {'all', 'none', 'first'}
            UNUSED.  What passes within `data` values are computed for.

        Returns
        -------
        NamedDict
            A nested dictionary with indices: component-name, depth, circuit-index
            (the last level is a *list*, not a dict).
        """
        design = data.edesign
        ds = data.dataset

        depths = design.depths
        qty_data = _tools.NamedDict('Datatype', 'category', None, None,
                                    {comp: _tools.NamedDict('Depth', 'int', 'Value', 'float',
                                                            {depth: [] for depth in depths})
                                     for comp in component_names})

        #loop over all circuits
        if energy is False:
            for depth, circuits_at_depth, idealouts_at_depth in zip(depths, design.circuit_lists, design.idealout_lists):
                for icirc, (circ, idealout) in enumerate(zip(circuits_at_depth, idealouts_at_depth)):
                    dsrow = ds[circ] if (ds is not None) else None  # stripOccurrenceTags=True ??
                # -- this is where Tim thinks there's a bottleneck, as these loops will be called for each
                # member of a simultaneous experiment separately instead of having an inner-more iteration
                # that loops over the "structure", i.e. the simultaneous qubit sectors.
                #TODO: <print percentage>

                    for component_name, val in compute_fn(icirc, circ, dsrow, idealout).items():
                        qty_data[component_name][depth].append(val)  # maybe use a pandas dataframe here?
        else:
            for depth, circuits_at_depth, measurements_at_depth, signs_at_depth in zip(depths, design.circuit_lists, design.measurements, design.signs):
                for icirc, (circ, measurement, sign) in enumerate(zip(circuits_at_depth, measurements_at_depth, signs_at_depth)):
                    dsrow = ds[circ] if (ds is not None) else None
                    
                    for component_name, val in compute_fn(icirc, circ, dsrow, measurement, sign).items():
                        qty_data[component_name][depth].append(val)
    
        return qty_data

    def _create_depthwidth_dict(self, depths, widths, fillfn, seriestype):
        """
        Create a nested :class:`NamedDict` with depht and width indices.

        Parameters
        ----------
        depths : list or tuple
            The (integer) depths to use.

        widths : list or tuple
            The (integer) widths to use.

        fillfn : function
            A function with no arguments that is called to return a default value
            for each (depth, width).

        seriestype : {"float", "int"}
            The type of values held by this nested dict.

        Returns
        -------
        NamedDict
        """
        return _tools.NamedDict(
            'Depth', 'int', None, None, {depth: _tools.NamedDict(
                'Width', 'int', 'Value', seriestype, {width: fillfn() for width in widths}) for depth in depths})

    def _add_bootstrap_qtys(self, data_cache, num_qtys, finitecounts=True):
        """
        Adds bootstrapped "summary data".

        The bootstrap is over both the finite counts of each circuit and
        over the circuits at each length.

        Note: only adds quantities if they're needed.

        Parameters
        ----------
        data_cache : dict
            A cache of already-existing bootstraps.

        num_qtys : int, optional
            The number of bootstrapped data to construct.

        finitecounts : bool, optional
            Whether finite counts should be used, i.e. whether the bootstrap samples
            include finite sample error with the same number of counts as the sampled
            data, or whether they have no finite sample error (just probabilities).

        Returns
        -------
        None
        """
        key = 'bootstraps' if finitecounts else 'infbootstraps'
        if key in data_cache:
            num_existing = len(data_cache['bootstraps'])
        else:
            data_cache[key] = []
            num_existing = 0

        #extract "base" values from cache, to base boostrap off of
        # Wonky try statements aren't working...
        try:
            success_probabilities = data_cache['success_probabilities']
            depths = list(success_probabilities.keys())
            exists_sps = True
        except:
            exists_sps = False
        try:
            total_counts = data_cache['total_counts']
        except:
            pass
        try:
            hamming_distance_counts = data_cache['hamming_distance_counts']
            depths = list(success_probabilities.keys())
            exists_hds = True
        except:
            exists_hds = False
        try:
            energies = data_cache['energies']

            depths = list(energies.keys())
            exists_energies = True
        except:
            exists_energies = False


        for i in range(num_existing, num_qtys):

            component_names = self.summary_statistics
            bcache = _tools.NamedDict(
                'Datatype', 'category', None, None,
                {comp: _tools.NamedDict('Depth', 'int', 'Value', 'float', {depth: [] for depth in depths})
                 for comp in component_names})  # ~= "RB summary dataset"

            if exists_sps:
                for depth, SPs in success_probabilities.items():
                    numcircuits = len(SPs)
                    for k in range(numcircuits):
                        ind = _np.random.randint(numcircuits)
                        sampledSP = SPs[ind]
                        totalcounts = total_counts[depth][ind] if finitecounts else None
                        bcache['success_probabilities'][depth].append(sampledSP)
                        if finitecounts:
                            if not _np.isnan(sampledSP):
                                bcache['success_counts'][depth].append(_np.random.binomial(totalcounts, sampledSP))
                            else:
                                bcache['success_probabilities'][depth].append(sampledSP)
                            bcache['total_counts'][depth].append(totalcounts)
                        else:
                            bcache['success_counts'][depth].append(sampledSP)

                        ind = _np.random.randint(numcircuits)  # note: old code picked different random ints
                        totalcounts = int(total_counts[depth][ind]) if finitecounts else None  # need this if a new randint
                        #sampledE = (energies[depth][ind]+1)/2
                        sampledHDcounts = hamming_distance_counts[depth][ind]
                        sampledHDpdf = _np.array(sampledHDcounts) / _np.sum(sampledHDcounts)
                        if exists_hds:
                            if finitecounts:
                                if not _np.isnan(sampledSP):
                                    bcache['hamming_distance_counts'][depth].append(
                                        list(_np.random.multinomial(totalcounts, sampledHDpdf)))

                                else:
                                    bcache['hamming_distance_counts'][depth].append(sampledHDpdf)
                            else:
                                bcache['hamming_distance_counts'][depth].append(sampledHDpdf)

                            # replicates adjusted_success_probability function above
                            adjSP = _np.sum([(-1 / 2)**n * sampledHDpdf[n] for n in range(len(sampledHDpdf))])
                            bcache['adjusted_success_probabilities'][depth].append(adjSP)
                    
            #ENERGIES BOOTSTRAP#
            if exists_energies:
                for depth, Es in energies.items():
                    #print(energies)
                    numcircuits = len(Es)
                    for k in range(numcircuits):
                        ind = _np.random.randint(numcircuits)

                        sampledSP = (Es[ind]+1)/2 #energies[depth][ind]
                        totalcounts = total_counts[depth][ind]
                        bcache['total_counts'][depth].append(totalcounts)
            
            #resample at each depth
                        if finitecounts:
                            if not _np.isnan(sampledSP):
                                new_sp = _np.random.binomial(totalcounts, sampledSP)/totalcounts    
                                #reconvert to energy
                                e = 2*new_sp-1
                                bcache['energies'][depth].append(e)
                            else:
                                #return original
                                bcache['energies'][depth].append(2*sampledSP-1)
                        else:
                            bcache['energies'][depth].append(2*sampledSP-1)


            data_cache[key].append(bcache)


class ByDepthSummaryStatistics(SummaryStatistics):
    """
    A protocol that computes summary statistics for data organized into by-depth circuit lists.
    Parameters
    ----------
    depths : list or "all", optional
        A sequence of the depths to compute summary statistics for or the special `"all"`
        value which means "all the depths in the data".  If data being processed does not
        contain a given value in `depths`, it is just ignored.
    statistics_to_compute : tuple, optional
        A sequence of the statistic names to compute. Allowed names are:
       'success_counts', 'total_counts', 'hamming_distance_counts', 'success_probabilities', 'polarization',
       'adjusted_success_probabilities', 'two_q_gate_count', 'depth', 'idealout', 'circuit_index',
       and 'width'.
    names_to_compute : tuple, optional
        A sequence of user-defined names for the statistics in `statistics_to_compute`.  If `None`, then
        the statistic names themselves are used.  These names are the column names produced by calling
        `to_dataframe` on this protocol's results, so can be useful to name the computed statistics differently
        from the statistic name itself to distinguish it from the same statistic run on other data, when you
        want to combine data frames generated from multiple :class:`ProtocolData` objects.
    custom_data_src : SuccessFailModel, optional
        An alternate source of the data counts used to compute the desired summary statistics.  Currently
        this can only be a :class:`SuccessFailModel`.
    name : str, optional
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.
    """
    def __init__(self, depths='all', statistics_to_compute=('polarization',), names_to_compute=None,
                 custom_data_src=None, name=None):
        super().__init__(name)
        self.depths = depths
        self.statistics_to_compute = statistics_to_compute
        self.names_to_compute = statistics_to_compute if (names_to_compute is None) else names_to_compute
        self.custom_data_src = custom_data_src
        # because this *could* be a model or a qty dict (or just a string?)
        self.auxfile_types['custom_data_src'] = 'serialized-object'

    def _get_statistic_per_depth(self, statistic, data):
        design = data.edesign

        if self.custom_data_src is None:  # then use the data in `data`
            #Note: can only take/put things ("results") in data.cache that *only* depend on the exp. design
            # and dataset (i.e. the DataProtocol object).  Protocols must be careful of this in their implementation!
            if statistic in self.summary_statistics:
                if statistic not in data.cache:
                    summary_data_dict = self._compute_summary_statistics(data)
                    data.cache.update(summary_data_dict)
            # Code currently doesn't work with a dscmp, so commented out.
            # elif statistic in self.dscmp_statistics:
            #     if statistic not in data.cache:
            #         dscmp_data = self.compute_dscmp_data(data, dscomparator)
            #         data.cache.update(dscmp_data)
            elif statistic in self.circuit_statistics:
                if statistic not in data.cache:
                    circuit_data = self._compute_circuit_statistics(data)
                    data.cache.update(circuit_data)
            else:
                raise ValueError("Invalid statistic: %s" % statistic)

            statistic_per_depth = data.cache[statistic]

        elif isinstance(self.custom_data_src, _SuccessFailModel):  # then simulate all the circuits in `data`
            assert(statistic in ('success_probabilities', 'polarization')), \
                "Only success probabilities or polarizations can be simulated!"
            sfmodel = self.custom_data_src
            depths = design.depths if self.depths == 'all' else self.depths
            statistic_per_depth = _tools.NamedDict('Depth', 'int', 'Value', 'float', {depth: [] for depth in depths})
            circuit_lists_for_depths = {depth: lst for depth, lst in zip(design.depths, design.circuit_lists)}

            for depth in depths:
                for circ in circuit_lists_for_depths[depth]:
                    predicted_success_prob = sfmodel.probabilities(circ)[('success',)]
                    if statistic == 'success_probabilities':
                        statistic_per_depth[depth].append(predicted_success_prob)
                    elif statistic == 'polarization':
                        nQ = len(circ.line_labels)
                        pol = (predicted_success_prob - 1 / 2**nQ) / (1 - 1 / 2**nQ)
                        statistic_per_depth[depth].append(pol)

        # Note sure this is used anywhere, so commented out for now.
        #elif isinstance(custom_data_src, dict):  # Assume this is a "summary dataset"
        #    summary_data = self.custom_data_src
        #    statistic_per_depth = summary_data['success_probabilities']

        else:
            raise ValueError("Invalid 'custom_data_src' of type: %s" % str(type(self.custom_data_src)))

        return statistic_per_depth

    def run(self, data, memlimit=None, comm=None, dscomparator=None):
        """
        Run this protocol on `data`.
        Parameters
        ----------
        results : ProtocolResults or ProtocolResultsDir
            The input results.
        memlimit : int, optional
            A rough per-processor memory limit in bytes.
        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.
        dscomparator : DataComparator
            Special additional comparator object for
            comparing data sets.
        Returns
        -------
        SummaryStatisticsResults
        """
        design = data.edesign
        width = len(design.qubit_labels)
        results = SummaryStatisticsResults(data, self)

        for statistic, statistic_nm in zip(self.statistics_to_compute, self.names_to_compute):
            statistic_per_depth = self._get_statistic_per_depth(statistic, data)
            depths = statistic_per_depth.keys() if self.depths == 'all' else \
                filter(lambda d: d in statistic_per_depth, self.depths)
            statistic_per_dwc = self._create_depthwidth_dict(depths, (width,), lambda: None, 'float')
            # a nested NamedDict with indices: depth, width, circuit_index (width only has single value though)

            for depth in depths:
                percircuitdata = statistic_per_depth[depth]
                statistic_per_dwc[depth][width] = \
                    _tools.NamedDict('CircuitIndex', 'int', 'Value', 'float',
                                     {i: j for i, j in enumerate(percircuitdata)})
            results.statistics[statistic_nm] = statistic_per_dwc
        return results


class SummaryStatisticsResults(_proto.ProtocolResults):
    """
    Summary statistics computed for a set of data.

    Usually the result of running a :class:`SummaryStatistics` (or derived) protocol.

    Parameters
    ----------
    data : ProtocolData
        The experimental data these results are generated from.

    protocol_instance : Protocol
        The protocol that generated these results.
    """
    def __init__(self, data, protocol_instance):
        """
        Initialize an empty SummaryStatisticsResults object.
        """
        super().__init__(data, protocol_instance)
        self.statistics = {}
        self.auxfile_types['statistics'] = 'dict:serialized-object'  # dict of NamedDicts

    def _my_attributes_as_nameddict(self):
        """Overrides base class behavior so elements of self.statistics form top-level NamedDict"""
        stats = _tools.NamedDict('ValueName', 'category')
        for k, v in self.statistics.items():
            assert(isinstance(v, _tools.NamedDict)), \
                "SummaryStatisticsResults.statistics dict should be populated with NamedDicts, not %s" % str(type(v))
            stats[k] = v
        return stats
