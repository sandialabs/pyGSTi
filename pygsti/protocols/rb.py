"""
RB Protocol objects
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
from pygsti.algorithms import randomcircuit as _rc
from pygsti.algorithms import rbfit as _rbfit
from pygsti.algorithms import mirroring as _mirroring


class CliffordRBDesign(_vb.BenchmarkingDesign):
    """
    Experiment design for Clifford randomized benchmarking.

    This encapsulates a "Clifford randomized benchmarking" (CRB) experiment. CRB is the RB protocol defined 
    in "Scalable and robust randomized benchmarking of quantum processes", Magesan et al. PRL 106 180504 (2011).
    The circuits created by this function will respect the connectivity and gate-set of the device encoded by 
    `pspec` (see the :class:`QubitProcessorSpec` object docstring for how to construct the relevant `pspec` 
    for a device).

    Note that this function uses the convention that a depth "l" CRB circuit  consists of "l"+2 Clifford gates 
    before compilation.

    Parameters
    ----------
    pspec : QubitProcessorSpec
       The QubitProcessorSpec for the device that the CRB experiment is being generated for, which defines the 
       "native" gate-set and the connectivity of the device. The returned CRB circuits will be over the gates in 
       `pspec`, and will respect the connectivity encoded by `pspec`.

    clifford_compilations : dict
        A dictionary with the potential keys `'absolute'` and `'paulieq'` and corresponding class:`CompilationRules` values. 
        These compilation rules specify how to compile the "native" gates of `pspec` into Clifford gates.

    depths : list of ints
        The "CRB depths" of the circuit; a list of integers >= 0. The CRB length is the number of Cliffords in the 
        circuit - 2 *before* each Clifford is compiled into the native gate-set.

    circuits_per_depth : int
        The number of (possibly) different CRB circuits sampled at each length.

    qubit_labels : list, optional
        If not None, a list of the qubits that the RB circuits are to be sampled for. This should
        be all or a subset of the qubits in the device specified by the QubitProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the "wires" in the returned circuit, but is otherwise
        irrelevant. If desired, a circuit that explicitly idles on the other qubits can be obtained
        by using methods of the Circuit object.

    randomizeout : bool, optional
        If False, the ideal output of the circuits (the "success" or "survival" outcome) is always
        the all-zeros bit string. This is probably considered to be the "standard" in CRB. If True,
        the ideal output a circuit is randomized to a uniformly random bit-string. This setting is
        useful for, e.g., detecting leakage/loss/measurement-bias etc.

    interleaved_circuit : Circuit, optional (default None)
        Circuit to use in the constuction of an interleaved CRB experiment. When specified each
        random clifford operation is interleaved with the specified circuit.

    citerations : int, optional
        Some of the Clifford compilation algorithms in pyGSTi (including the default algorithm) are
        randomized, and the lowest-cost circuit is chosen from all the circuit generated in the
        iterations of the algorithm. This is the number of iterations used. The time required to
        generate a CRB circuit is linear in `citerations * (CRB length + 2)`. Lower-depth / lower 2-qubit
        gate count compilations of the Cliffords are important in order to successfully implement
        CRB on more qubits.

    compilerargs : list, optional
        A list of arguments that are handed to compile_clifford() function, which includes all the
        optional arguments of compile_clifford() *after* the `iterations` option (set by `citerations`).
        In order, this list should be values for:
        
        * algorithm : str. A string that specifies the compilation algorithm. The default in
          compile_clifford() will always be whatever we consider to be the 'best' all-round
          algorithm.
        * aargs : list. A list of optional arguments for the particular compilation algorithm.
        * costfunction : 'str' or function. The cost-function from which the "best" compilation
          for a Clifford is chosen from all `citerations` compilations. The default costs a
          circuit as 10x the num. of 2-qubit gates in the circuit + 1x the depth of the circuit.
        * prefixpaulis : bool. Whether to prefix or append the Paulis on each Clifford.
        * paulirandomize : bool. Whether to follow each layer in the Clifford circuit with a
          random Pauli on each qubit (compiled into native gates). I.e., if this is True the
          native gates are Pauli-randomized. When True, this prevents any coherent errors adding
          (on average) inside the layers of each compiled Clifford, at the cost of increased
          circuit depth. Defaults to False.
        
        For more information on these options, see the compile_clifford() docstring.

    descriptor : str, optional
        A string describing the experiment generated, which will be stored in the returned
        dictionary.

    add_default_protocol : bool, optional
        Whether to add a default RB protocol to the experiment design, which can be run
        later (once data is taken) by using a :class:`DefaultProtocolRunner` object.

    seed : int, optional
        A seed to initialize the random number generator used for creating random clifford
        circuits.

    verbosity : int, optional
        If > 0 the number of circuits generated so far is shown.
    """

    @classmethod
    def from_existing_circuits(cls, data_by_depth, qubit_labels=None,
                               randomizeout=False, citerations=20, compilerargs=(), interleaved_circuit=None,
                               descriptor='A Clifford RB experiment', add_default_protocol=False):
        """
        Create a :class:`CliffordRBDesign` from an existing set of sampled RB circuits.

        This function serves as an alternative to the usual method of creating a Clifford
        RB experiment design by sampling a number of circuits randomly.  This function
        takes a list of previously-sampled random circuits and does not sampling internally.

        Parameters
        ----------
        data_by_depth : dict
            A dictionary whose keys are integer depths and whose values are lists of
            `(circuit, ideal_outcome, num_native_gates)` tuples giving each RB circuit, its 
            ideal (correct) outcome, and (optionally) the number of native gates in the compiled Cliffords.
            If only a 2-tuple is passed, i.e. number of native gates is not included,
            the :meth:`average_gates_per_clifford()` function will not work.

        qubit_labels : list, optional
            If not None, a list of the qubits that the RB circuits are to be sampled for. This should
            be all or a subset of the qubits in the device specified by the QubitProcessorSpec `pspec`.
            If None, it is assumed that the RB circuit should be over all the qubits. Note that the
            ordering of this list is the order of the "wires" in the returned circuit, but is otherwise
            irrelevant. If desired, a circuit that explicitly idles on the other qubits can be obtained
            by using methods of the Circuit object.

        randomizeout : bool, optional
            If False, the ideal output of the circuits (the "success" or "survival" outcome) is always
            the all-zeros bit string. This is probably considered to be the "standard" in CRB. If True,
            the ideal output a circuit is randomized to a uniformly random bit-string. This setting is
            useful for, e.g., detecting leakage/loss/measurement-bias etc.

        citerations : int, optional
            Some of the Clifford compilation algorithms in pyGSTi (including the default algorithm) are
            randomized, and the lowest-cost circuit is chosen from all the circuit generated in the
            iterations of the algorithm. This is the number of iterations used. The time required to
            generate a CRB circuit is linear in `citerations` * (CRB length + 2). Lower-depth / lower 2-qubit
            gate count compilations of the Cliffords are important in order to successfully implement
            CRB on more qubits.

        compilerargs : list, optional
            A list of arguments that are handed to compile_clifford() function, which includes all the
            optional arguments of compile_clifford() *after* the `iterations` option (set by `citerations`).
            In order, this list should be values for:
            
            * algorithm : str. A string that specifies the compilation algorithm. The default in
              compile_clifford() will always be whatever we consider to be the 'best' all-round
              algorithm.
            * aargs : list. A list of optional arguments for the particular compilation algorithm.
            * costfunction : 'str' or function. The cost-function from which the "best" compilation
              for a Clifford is chosen from all `citerations` compilations. The default costs a
              circuit as 10x the num. of 2-qubit gates in the circuit + 1x the depth of the circuit.
            * prefixpaulis : bool. Whether to prefix or append the Paulis on each Clifford.
            * paulirandomize : bool. Whether to follow each layer in the Clifford circuit with a
              random Pauli on each qubit (compiled into native gates). I.e., if this is True the
              native gates are Pauli-randomized. When True, this prevents any coherent errors adding
              (on average) inside the layers of each compiled Clifford, at the cost of increased
              circuit depth. Defaults to False.
                    
            For more information on these options, see the compile_clifford() docstring.

        descriptor : str, optional
            A string describing the experiment generated, which will be stored in the returned
            dictionary.

        add_default_protocol : bool, optional
            Whether to add a default RB protocol to the experiment design, which can be run
            later (once data is taken) by using a :class:`DefaultProtocolRunner` object.

        Returns
        -------
        CliffordRBDesign
        """
        depths = sorted(list(data_by_depth.keys()))
        circuit_lists = [[x[0] for x in data_by_depth[d]] for d in depths]
        ideal_outs = [[x[1] for x in data_by_depth[d]] for d in depths]
        try:
            native_gate_counts = [[x[2] for x in data_by_depth[d]] for d in depths]
        except IndexError:
            native_gate_counts = None
        circuits_per_depth = [len(data_by_depth[d]) for d in depths]
        self = cls.__new__(cls)
        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              randomizeout, citerations, compilerargs, descriptor, add_default_protocol,
                              interleaved_circuit, native_gate_counts=native_gate_counts)
        return self

    def __init__(self, pspec, clifford_compilations, depths, circuits_per_depth, qubit_labels=None, randomizeout=False,
                 interleaved_circuit=None, citerations=20, compilerargs=(), exact_compilation_key=None,
                 descriptor='A Clifford RB experiment', add_default_protocol=False, seed=None, verbosity=1, num_processes=1):
        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        ideal_outs = []
        native_gate_counts = []

        if seed is None:
            self.seed = _np.random.randint(1, 1e6)  # Pick a random seed
        else:
            self.seed = seed

        for lnum, l in enumerate(depths):
            lseed = self.seed + lnum * circuits_per_depth
            if verbosity > 0:
                print('- Sampling {} circuits at CRB length {} ({} of {} depths) with seed {}'.format(
                    circuits_per_depth, l, lnum + 1, len(depths), lseed))

            args_list = [(pspec, clifford_compilations, l)] * circuits_per_depth
            kwargs_list = [dict(qubit_labels=qubit_labels, randomizeout=randomizeout, citerations=citerations,
                                compilerargs=compilerargs, interleaved_circuit=interleaved_circuit,
                                seed=lseed + i, return_native_gate_counts=True, exact_compilation_key=exact_compilation_key)
                                for i in range(circuits_per_depth)]
            results = _tools.mptools.starmap_with_kwargs(_rc.create_clifford_rb_circuit, circuits_per_depth,
                                                         num_processes, args_list, kwargs_list)

            circuits_at_depth = []
            idealouts_at_depth = []
            native_gate_counts_at_depth = []
            for c, iout, nng in results:
                circuits_at_depth.append(c)
                idealouts_at_depth.append((''.join(map(str, iout)),))
                native_gate_counts_at_depth.append(nng)

            circuit_lists.append(circuits_at_depth)
            ideal_outs.append(idealouts_at_depth)
            native_gate_counts.append(native_gate_counts_at_depth)

        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              randomizeout, citerations, compilerargs, descriptor, add_default_protocol,
                              interleaved_circuit, native_gate_counts=native_gate_counts)

    def _init_foundation(self, depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                         randomizeout, citerations, compilerargs, descriptor, add_default_protocol,
                         interleaved_circuit, native_gate_counts=None, exact_compilation_key=None):
        self.native_gate_count_lists = native_gate_counts
        if self.native_gate_count_lists is not None:
            # If we have native gate information, pair this with circuit data so that we serialize/truncate properly
            self.paired_with_circuit_attrs = ["native_gate_count_lists"]

        super().__init__(depths, circuit_lists, ideal_outs, qubit_labels, remove_duplicates=False)
        self.circuits_per_depth = circuits_per_depth
        self.randomizeout = randomizeout
        self.citerations = citerations
        self.compilerargs = compilerargs
        self.descriptor = descriptor
        self.interleaved_circuit = interleaved_circuit
        self.exact_compilation_key = exact_compilation_key
        if add_default_protocol:
            if randomizeout:
                defaultfit = 'A-fixed'
            else:
                defaultfit = 'full'
            self.add_default_protocol(RB(name='RB', defaultfit=defaultfit))
        
        #set some auxfile information for interleaved_circuit
        self.auxfile_types['interleaved_circuit'] = 'circuit-str-json'

    def average_native_gates_per_clifford_for_circuit(self, list_idx, circ_idx):
        """The average number of native gates per Clifford for a specific circuit

        Parameters
        ----------
        list_idx: int
            The index of the circuit list (for a given depth)

        circ_idx: int
            The index of the circuit within the circuit list
        
        Returns
        -------
        avg_gate_counts: dict
            The average number of native gates, native 2Q gates, and native size
            per Clifford as values with respective label keys
        """
        if self.native_gate_count_lists is None:
            raise ValueError("Native gate counts not available, cannot compute average gates per Clifford")
        
        num_clifford_gates = self.depths[list_idx] + 1
        avg_gate_counts = {}
        for key, native_gate_count in self.native_gate_count_lists[list_idx][circ_idx].items():
            avg_gate_counts[key.replace('native', 'avg_native_per_clifford')] = native_gate_count / num_clifford_gates
        
        return avg_gate_counts

    def average_native_gates_per_clifford_for_circuit_list(self, list_idx):
        """The average number of gates per Clifford for a circuit list

        This essentially gives the average number of native gates per Clifford
        for a given depth (indexed by list index, not depth).

        Parameters
        ----------
        list_idx: int
            The index of the circuit list (for a given depth)

        circ_idx: int
            The index of the circuit within the circuit list
        
        Returns
        -------
        float
            The average number of native gates per Clifford
        """
        if self.native_gate_count_lists is None:
            raise ValueError("Native gate counts not available, cannot compute average gates per Clifford")
        
        gate_counts = defaultdict(int)
        for native_gate_counts in self.native_gate_count_lists[list_idx]:
            for k, v in native_gate_counts.items():
                gate_counts[k] += v
        
        num_clifford_gates = len(self.native_gate_count_lists[list_idx]) * (self.depths[list_idx] + 1)
        avg_gate_counts = {}
        for key, total_native_gate_counts in gate_counts.items():
            avg_gate_counts[key.replace('native', 'avg_native_per_clifford')] = total_native_gate_counts / num_clifford_gates
        
        return avg_gate_counts

    def average_native_gates_per_clifford(self):
        """The average number of native gates per Clifford for all circuits

        Returns
        -------
        float
            The average number of native gates per Clifford
        """
        if self.native_gate_count_lists is None:
            raise ValueError("Number of native gates not available, cannot compute average gates per Clifford")
        
        gate_counts = defaultdict(int)
        num_clifford_gates = 0
        for list_idx in range(len(self.depths)):
            for native_gate_counts in self.native_gate_count_lists[list_idx]:
                for k, v in native_gate_counts.items():
                    gate_counts[k] += v
            num_clifford_gates += len(self.native_gate_count_lists[list_idx]) * (self.depths[list_idx] + 1)
            
        avg_gate_counts = {}
        for key, total_native_gate_counts in gate_counts.items():
            avg_gate_counts[key.replace('native', 'avg_native_per_clifford')] = total_native_gate_counts / num_clifford_gates

        return avg_gate_counts

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
        CliffordRBDesign
        """
        mapped_circuits_and_idealouts_by_depth = self._mapped_circuits_and_idealouts_by_depth(mapper)
        mapped_qubit_labels = self._mapped_qubit_labels(mapper)
        if self.interleaved_circuit is not None:
            raise NotImplementedError("TODO: figure out whether `interleaved_circuit` needs to be mapped!")
        return CliffordRBDesign.from_existing_circuits(mapped_circuits_and_idealouts_by_depth,
                                                       mapped_qubit_labels,
                                                       self.randomizeout, self.citerations, self.compilerargs,
                                                       self.interleaved_circuit, self.descriptor,
                                                       add_default_protocol=False)


class DirectRBDesign(_vb.BenchmarkingDesign):
    """
    Experiment design for Direct randomized benchmarking.

    This encapsulates a "direct randomized benchmarking" (DRB) experiments.  DRB was a protocol
    introduced in arXiv:1807.07975 (2018).

    An n-qubit DRB circuit consists of (1) a circuit the prepares a uniformly random stabilizer state;
    (2) a length-l circuit (specified by `length`) consisting of circuit layers sampled according to
    some user-specified distribution (specified by `sampler`), (3) a circuit that maps the output of
    the preceeding circuit to a computational basis state. See arXiv:1807.07975 (2018) for further
    details.

    Parameters
    ----------
    pspec : QubitProcessorSpec
       The QubitProcessorSpec for the device that the circuit is being sampled for, which defines the
       "native" gate-set and the connectivity of the device. The returned DRB circuit will be over
       the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
       is always handed to the sampler, as the first argument of the sampler function (this is only
       of importance when not using an in-built sampler for the "core" of the DRB circuit). Unless
       `qubit_labels` is not None, the circuit is sampled over all the qubits in `pspec`.

    clifford_compilations : dict
        A dictionary with the potential keys `'absolute'` and `'paulieq'` and corresponding
        :class:`CompilationRules` values.  These compilation rules specify how to compile the
        "native" gates of `pspec` into Clifford gates.

    depths : int
        The set of "direct RB depths" for the circuits. The DRB depths must be integers >= 0.
        Unless `addlocal` is True, the DRB length is the depth of the "core" random circuit,
        sampled according to `sampler`, specified in step (2) above. If `addlocal` is True,
        each layer in the "core" circuit sampled according to "sampler` is followed by a layer of
        1-qubit gates, with sampling specified by `lsargs` (and the first layer is proceeded by a
        layer of 1-qubit gates), and so the circuit of step (2) is length 2*`length` + 1.

    circuits_per_depth : int
        The number of (possibly) different DRB circuits sampled at each length.

    qubit_labels : list, optional
        If not None, a list of the qubits to sample the circuit for. This is a subset of
        `pspec.qubit_labels`. If None, the circuit is sampled to act on all the qubits
        in `pspec`.

    sampler : str or function, optional
        If a string, this should be one of: {'edgegrab', pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function
        in rb.sampler named `circuit_layer_by_*` (with `*` replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
        a valid form of sampling for n-qubit DRB, but is not explicitly forbidden in this function].
        If `sampler` is a function, it should be a function that takes as the first argument a
        QubitProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
        the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is the
        only sampler that requires no parameters beyond the QubitProcessorSpec *and* works for arbitrary
        connectivity devices. See the docstrings for each of these samplers for more information.

    samplerargs : list, optional
        A list of arguments that are handed to the sampler function, specified by `sampler`.
        The first argument handed to the sampler is `pspec`, the second argument is `qubit_labels`,
        and `samplerargs` lists the remaining arguments handed to the sampler. This is not
        optional for some choices of `sampler`.

    addlocal : bool, optional
        Whether to follow each layer in the "core" circuits, sampled according to `sampler` with
        a layer of 1-qubit gates.

    lsargs : list, optional
        Only used if addlocal is True. A list of optional arguments handed to the 1Q gate
        layer sampler circuit_layer_by_oneQgate(). Specifies how to sample 1Q-gate layers.

    randomizeout : bool, optional
        If False, the ideal output of the circuits (the "success" or "survival" outcome) is the all-zeros
        bit string. If True, the ideal output of each circuit is randomized to a uniformly random bit-string.
        This setting is useful for, e.g., detecting leakage/loss/measurement-bias etc.

    cliffordtwirl : bool, optional
        Wether to begin the circuits with a sequence that generates a random stabilizer state. For
        standard DRB this should be set to True. There are a variety of reasons why it is better
        to have this set to True.

    conditionaltwirl : bool, optional
        DRB only requires that the initial/final sequences of step (1) and (3) create/measure
        a uniformly random / particular stabilizer state, rather than implement a particular unitary.
        step (1) and (3) can be achieved by implementing a uniformly random Clifford gate and the
        unique inversion Clifford, respectively. This is implemented if `conditionaltwirl` is False.
        However, steps (1) and (3) can be implemented much more efficiently than this: the sequences
        of (1) and (3) only need to map a particular input state to a particular output state,
        if `conditionaltwirl` is True this more efficient option is chosen -- this is option corresponds
        to "standard" DRB. (the term "conditional" refers to the fact that in this case we essentially
        implementing a particular Clifford conditional on a known input).

    citerations : int, optional
        Some of the stabilizer state / Clifford compilation algorithms in pyGSTi (including the default
        algorithms) are  randomized, and the lowest-cost circuit is chosen from all the circuits generated
        in the iterations of the algorithm. This is the number of iterations used. The time required to
        generate a DRB circuit is linear in `citerations`. Lower-depth / lower 2-qubit gate count
        compilations of steps (1) and (3) are important in order to successfully implement DRB on as many
        qubits as possible.

    compilerargs : list, optional
        A list of arguments that are handed to the compile_stabilier_state/measurement()functions (or the
        compile_clifford() function if `conditionaltwirl `is False). This includes all the optional
        arguments of these functions *after* the `iterations` option (set by `citerations`). For most
        purposes the default options will be suitable (or at least near-optimal from the compilation methods
        in-built into pyGSTi). See the docstrings of these functions for more information.

    partitioned : bool, optional
        If False, each circuit is returned as a single full circuit. If True, each circuit is returned as
        a list of three circuits consisting of: (1) the stabilizer-prep circuit, (2) the core random circuit,
        (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1)
        and then (3) to (1).

    descriptor : str, optional
        A description of the experiment being generated. Stored in the output dictionary.

    add_default_protocol : bool, optional
        Whether to add a default RB protocol to the experiment design, which can be run
        later (once data is taken) by using a :class:`DefaultProtocolRunner` object.

    seed : int, optional
        A seed to initialize the random number generator used for creating random clifford
        circuits.

    verbosity : int, optional
        If > 0 the number of circuits generated so far is shown.
    """

    @classmethod
    def from_existing_circuits(cls, circuits_and_idealouts_by_depth, qubit_labels=None,
                               sampler='edgegrab', samplerargs=None, addlocal=False,
                               lsargs=(), randomizeout=False, cliffordtwirl=True, conditionaltwirl=True,
                               citerations=20, compilerargs=(), partitioned=False,
                               descriptor='A DRB experiment', add_default_protocol=False):
        """
        Create a :class:`DirectRBDesign` from an existing set of sampled RB circuits.

        This function serves as an alternative to the usual method of creating a direct
        RB experiment design by sampling a number of circuits randomly.  This function
        takes a list of previously-sampled random circuits and does not sampling internally.

        Parameters
        ----------
        circuits_and_idealouts_by_depth : dict
            A dictionary whose keys are integer depths and whose values are lists
            of `(circuit, ideal_outcome)` 2-tuples giving each RB circuit and its
            ideal (correct) outcome.

        qubit_labels : list, optional
            If not None, a list of the qubits to sample the circuit for. This is a subset of
            `pspec.qubit_labels`. If None, the circuit is sampled to act on all the qubits
            in `pspec`.

        sampler : str or function, optional
            If a string, this should be one of: {'edgegrab', pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
            Except for 'local', this corresponds to sampling layers according to the sampling function
            in rb.sampler named circuit_layer_by_* (with * replaced by 'sampler'). For 'local', this
            corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
            a valid form of sampling for n-qubit DRB, but is not explicitly forbidden in this function].
            If `sampler` is a function, it should be a function that takes as the first argument a
            QubitProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
            the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is the
            only sampler that requires no parameters beyond the QubitProcessorSpec *and* works for arbitrary
            connectivity devices. See the docstrings for each of these samplers for more information.

        samplerargs : list, optional
            A list of arguments that are handed to the sampler function, specified by `sampler`.
            Defaults to [0.25, ].
            The first argument handed to the sampler is `pspec`, the second argument is `qubit_labels`,
            and `samplerargs` lists the remaining arguments handed to the sampler. This is not
            optional for some choices of `sampler`.

        addlocal : bool, optional
            Whether to follow each layer in the "core" circuits, sampled according to `sampler` with
            a layer of 1-qubit gates.

        lsargs : list, optional
            Only used if addlocal is True. A list of optional arguments handed to the 1Q gate
            layer sampler circuit_layer_by_oneQgate(). Specifies how to sample 1Q-gate layers.

        randomizeout : bool, optional
            If False, the ideal output of the circuits (the "success" or "survival" outcome) is the all-zeros
            bit string. If True, the ideal output of each circuit is randomized to a uniformly random bit-string.
            This setting is useful for, e.g., detecting leakage/loss/measurement-bias etc.

        cliffordtwirl : bool, optional
            Wether to begin the circuits with a sequence that generates a random stabilizer state. For
            standard DRB this should be set to True. There are a variety of reasons why it is better
            to have this set to True.

        conditionaltwirl : bool, optional
            DRB only requires that the initial/final sequences of step (1) and (3) create/measure
            a uniformly random / particular stabilizer state, rather than implement a particular unitary.
            step (1) and (3) can be achieved by implementing a uniformly random Clifford gate and the
            unique inversion Clifford, respectively. This is implemented if `conditionaltwirl` is False.
            However, steps (1) and (3) can be implemented much more efficiently than this: the sequences
            of (1) and (3) only need to map a particular input state to a particular output state,
            if `conditionaltwirl` is True this more efficient option is chosen -- this is option corresponds
            to "standard" DRB. (the term "conditional" refers to the fact that in this case we essentially
            implementing a particular Clifford conditional on a known input).

        citerations : int, optional
            Some of the stabilizer state / Clifford compilation algorithms in pyGSTi (including the default
            algorithms) are  randomized, and the lowest-cost circuit is chosen from all the circuits generated
            in the iterations of the algorithm. This is the number of iterations used. The time required to
            generate a DRB circuit is linear in `citerations`. Lower-depth / lower 2-qubit gate count
            compilations of steps (1) and (3) are important in order to successfully implement DRB on as many
            qubits as possible.

        compilerargs : list, optional
            A list of arguments that are handed to the compile_stabilier_state/measurement()functions (or the
            compile_clifford() function if `conditionaltwirl `is False). This includes all the optional
            arguments of these functions *after* the `iterations` option (set by `citerations`). For most
            purposes the default options will be suitable (or at least near-optimal from the compilation methods
            in-built into pyGSTi). See the docstrings of these functions for more information.

        partitioned : bool, optional
            If False, each circuit is returned as a single full circuit. If True, each circuit is returned as
            a list of three circuits consisting of: (1) the stabilizer-prep circuit, (2) the core random circuit,
            (3) the pre-measurement circuit. In that case the full circuit is obtained by appended (2) to (1)
            and then (3) to (1).

        descriptor : str, optional
            A description of the experiment being generated. Stored in the output dictionary.

        add_default_protocol : bool, optional
            Whether to add a default RB protocol to the experiment design, which can be run
            later (once data is taken) by using a :class:`DefaultProtocolRunner` object.

        Returns
        -------
        DirectRBDesign
        """
        if samplerargs is None:
            samplerargs = [0.25, ]
        depths = sorted(list(circuits_and_idealouts_by_depth.keys()))
        circuit_lists = [[x[0] for x in circuits_and_idealouts_by_depth[d]] for d in depths]
        ideal_outs = [[x[1] for x in circuits_and_idealouts_by_depth[d]] for d in depths]
        circuits_per_depth = [len(circuits_and_idealouts_by_depth[d]) for d in depths]
        self = cls.__new__(cls)
        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              sampler, samplerargs, addlocal, lsargs, randomizeout, cliffordtwirl,
                              conditionaltwirl, citerations, compilerargs, partitioned, descriptor,
                              add_default_protocol)
        return self

    def __init__(self, pspec, clifford_compilations, depths, circuits_per_depth, qubit_labels=None,
                 sampler='edgegrab', samplerargs=None,
                 addlocal=False, lsargs=(), randomizeout=False, cliffordtwirl=True, conditionaltwirl=True,
                 citerations=20, compilerargs=(), partitioned=False, descriptor='A DRB experiment',
                 add_default_protocol=False, seed=None, verbosity=1, num_processes=1):

        if samplerargs is None:
            samplerargs = [0.25, ]
        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        ideal_outs = []

        if seed is None:
            self.seed = _np.random.randint(1, 1e6)  # Pick a random seed
        else:
            self.seed = seed

        for lnum, l in enumerate(depths):
            lseed = self.seed + lnum * circuits_per_depth
            if verbosity > 0:
                print('- Sampling {} circuits at DRB length {} ({} of {} depths) with seed {}'.format(
                    circuits_per_depth, l, lnum + 1, len(depths), lseed))

            args_list = [(pspec, clifford_compilations, l)] * circuits_per_depth
            kwargs_list = [dict(qubit_labels=qubit_labels, sampler=sampler, samplerargs=samplerargs,
                                addlocal=addlocal, lsargs=lsargs, randomizeout=randomizeout,
                                cliffordtwirl=cliffordtwirl, conditionaltwirl=conditionaltwirl,
                                citerations=citerations, compilerargs=compilerargs,
                                partitioned=partitioned,
                                seed=lseed + i) for i in range(circuits_per_depth)]
            #results = [_rc.create_direct_rb_circuit(*(args_list[0]), **(kwargs_list[0]))]  # num_processes == 1 case
            results = _tools.mptools.starmap_with_kwargs(_rc.create_direct_rb_circuit, circuits_per_depth,
                                                         num_processes, args_list, kwargs_list)

            circuits_at_depth = []
            idealouts_at_depth = []
            for c, iout in results:
                circuits_at_depth.append(c)
                idealouts_at_depth.append((''.join(map(str, iout)),))

            circuit_lists.append(circuits_at_depth)
            ideal_outs.append(idealouts_at_depth)

        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              sampler, samplerargs, addlocal, lsargs, randomizeout, cliffordtwirl,
                              conditionaltwirl, citerations, compilerargs, partitioned, descriptor,
                              add_default_protocol)

    def _init_foundation(self, depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                         sampler, samplerargs, addlocal, lsargs, randomizeout, cliffordtwirl,
                         conditionaltwirl, citerations, compilerargs, partitioned, descriptor,
                         add_default_protocol):
        super().__init__(depths, circuit_lists, ideal_outs, qubit_labels, remove_duplicates=False)
        self.circuits_per_depth = circuits_per_depth
        self.randomizeout = randomizeout
        self.citerations = citerations
        self.compilerargs = compilerargs
        self.descriptor = descriptor
        if isinstance(sampler, str):
            self.sampler = sampler
        else:
            self.sampler = 'function'
        self.samplerargs = samplerargs
        self.addlocal = addlocal
        self.lsargs = lsargs
        self.cliffordtwirl = cliffordtwirl
        self.conditionaltwirl = conditionaltwirl
        self.partitioned = partitioned

        if add_default_protocol:
            if randomizeout:
                defaultfit = 'A-fixed'
            else:
                defaultfit = 'full'
            self.add_default_protocol(RB(name='RB', defaultfit=defaultfit))

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
        DirectRBDesign
        """
        mapped_circuits_and_idealouts_by_depth = self._mapped_circuits_and_idealouts_by_depth(mapper)
        mapped_qubit_labels = self._mapped_qubit_labels(mapper)
        return DirectRBDesign.from_existing_circuits(mapped_circuits_and_idealouts_by_depth,
                                                     mapped_qubit_labels,
                                                     self.sampler, self.samplerargs, self.addlocal,
                                                     self.lsargs, self.randomizeout, self.cliffordtwirl,
                                                     self.conditionaltwirl, self.citerations, self.compilerargs,
                                                     self.partitioned, self.descriptor, add_default_protocol=False)


class MirrorRBDesign(_vb.BenchmarkingDesign):
    """
    Experiment design for mirror randomized benchmarking.

    Encapsulates a "mirror randomized benchmarking" (MRB) experiment, for the case of Clifford gates and with
    the option of Pauli randomization and local Clifford twirling. To implement mirror RB it is necessary
    for U^(-1) to in the gate set for every gate U in the gate set.

    **THIS METHOD IS IN DEVELOPEMENT. DO NOT EXPECT THAT THIS FUNCTION WILL BEHAVE THE SAME IN FUTURE RELEASES OF PYGSTI!**

    Parameters
    ----------
    pspec : QubitProcessorSpec
       The QubitProcessorSpec for the device that the experiment is being generated for. The `pspec` is always
       handed to the sampler, as the first argument of the sampler function.

    clifford_compilations : dict
        A dictionary with the potential keys `'absolute'` and `'paulieq'` and corresponding
        :class:`CompilationRules` values.  These compilation rules specify how to compile the
        "native" gates of `pspec` into Clifford gates.

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
          `2*length+1 + X` where X is a random variable (between 0 and normally <= ~12-16) that accounts for
          the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.
        * If `paulirandomize` is False and `localclifford` is True, the depth of a circuit is
          length + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for
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
        If a string, this should be one of: {'edgegrab', 'Qelimination', 'co2Qgates', 'local'}.
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

    descriptor : str, optional
        A string describing the generated experiment. Stored in the returned dictionary.

    add_default_protocol : bool, optional
        Whether to add a default RB protocol to the experiment design, which can be run
        later (once data is taken) by using a :class:`DefaultProtocolRunner` object.
    """

    @classmethod
    def from_existing_circuits(cls, circuits_and_idealouts_by_depth, qubit_labels=None,
                               circuit_type='clifford',
                               sampler='edgegrab', samplerargs=(0.25, ), localclifford=True,
                               paulirandomize=True, descriptor='A mirror RB experiment',
                               add_default_protocol=False):
        """
        Create a :class:`MirrorRBDesign` from an existing set of sampled RB circuits.

        This function serves as an alternative to the usual method of creating a mirror
        RB experiment design by sampling a number of circuits randomly.  This function
        takes a list of previously-sampled random circuits and does not sampling internally.

        Parameters
        ----------
        circuits_and_idealouts_by_depth : dict
            A dictionary whose keys are integer depths and whose values are lists
            of `(circuit, ideal_outcome)` 2-tuples giving each RB circuit and its
            ideal (correct) outcome.


        See init docstring for details on all other parameters.

        Returns
        -------
        MirrorRBDesign
        """
        depths = sorted(list(circuits_and_idealouts_by_depth.keys()))
        circuit_lists = [[x[0] for x in circuits_and_idealouts_by_depth[d]] for d in depths]
        ideal_outs = [[x[1] for x in circuits_and_idealouts_by_depth[d]] for d in depths]
        circuits_per_depth = [len(circuits_and_idealouts_by_depth[d]) for d in depths]
        self = cls.__new__(cls)
        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              circuit_type,
                              sampler, samplerargs, localclifford, paulirandomize, descriptor,
                              add_default_protocol)
        return self

    def __init__(self, pspec, depths, circuits_per_depth, qubit_labels=None, circuit_type='clifford',
                 clifford_compilations=None, sampler='edgegrab', samplerargs=(0.25, ),
                 localclifford=True, paulirandomize=True, descriptor='A mirror RB experiment',
                 add_default_protocol=False, seed=None, num_processes=1, verbosity=1):

        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        ideal_outs = []

        if seed is None:
            self.seed = _np.random.randint(1, 1e6)  # Pick a random seed
        else:
            self.seed = seed

        for lnum, l in enumerate(depths):
            lseed = self.seed + lnum * circuits_per_depth
            if verbosity > 0:
                print('- Sampling {} circuits at MRB length {} ({} of {} depths) with seed {}'.format(
                    circuits_per_depth, l, lnum + 1, len(depths), lseed))

            # future: port the starmap functionality to the non-clifford case and merge the two methods
            # by just callling `create_mirror_rb_circuit` but with a different argument.
            if circuit_type == 'clifford':
                args_list = [(pspec, clifford_compilations['absolute'], l)] * circuits_per_depth
                kwargs_list = [dict(qubit_labels=qubit_labels, sampler=sampler,
                                    samplerargs=samplerargs, localclifford=localclifford,
                                    paulirandomize=paulirandomize,
                                    seed=lseed + i) for i in range(circuits_per_depth)]
                results = _tools.mptools.starmap_with_kwargs(_rc.create_mirror_rb_circuit, circuits_per_depth,
                                                             num_processes, args_list, kwargs_list)

            elif circuit_type in ('cz+zxzxz-clifford', 'clifford+zxzxz-haar', 'clifford+zxzxz-clifford',
                                  'cz(theta)+zxzxz-haar'):
                assert(sampler == 'edgegrab'), "Unless circuit_type = 'clifford' the only valid sampler is 'edgegrab'."
                two_q_gate_density = samplerargs[0]
                if len(samplerargs) >= 2:
                    two_q_gate_args_lists = samplerargs[1]
                else:
                    # Default sampler arguments.
                    two_q_gate_args_lists = {'Gczr': [(str(_np.pi / 2),), (str(-_np.pi / 2),)]}

                one_q_gate_type = circuit_type.split('-')[-1]

                circs = [_rc.sample_random_cz_zxzxz_circuit(pspec, l // 2, qubit_labels=qubit_labels,
                                                            two_q_gate_density=two_q_gate_density,
                                                            one_q_gate_type=one_q_gate_type,
                                                            two_q_gate_args_lists=two_q_gate_args_lists)
                         for _ in range(circuits_per_depth)]

                mirroring_type = circuit_type.split('-')[0]
                if mirroring_type == 'cz+zxzxz':
                    mirroring_type = 'clifford+zxzxz'
                results = [(a, [b]) for a, b in [_mirroring.create_mirror_circuit(c, pspec, circ_type=mirroring_type)
                                                 for c in circs]]

            else:
                raise ValueError('Invalid option for `circuit_type`!')

            circuits_at_depth = []
            idealouts_at_depth = []
            for c, iout in results:
                circuits_at_depth.append(c)
                idealouts_at_depth.append((''.join(map(str, iout)),))

            circuit_lists.append(circuits_at_depth)
            ideal_outs.append(idealouts_at_depth)

        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              circuit_type, sampler, samplerargs, localclifford, paulirandomize, descriptor,
                              add_default_protocol, seed=seed)

    def _init_foundation(self, depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                         circuit_type, sampler, samplerargs, localclifford, paulirandomize, descriptor,
                         add_default_protocol, seed=None):
        super().__init__(depths, circuit_lists, ideal_outs, qubit_labels, remove_duplicates=False)
        self.circuits_per_depth = circuits_per_depth
        self.descriptor = descriptor
        self.circuit_type = circuit_type
        self.sampler = sampler
        self.samplerargs = samplerargs
        self.localclifford = localclifford
        self.paulirandomize = paulirandomize
        self.seed = seed

        if add_default_protocol:
            self.add_default_protocol(RB(name='RB', datatype='adjusted_success_probabilities', defaultfit='A-fixed'))

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
        MirrorRBDesign
        """
        mapped_circuits_and_idealouts_by_depth = self._mapped_circuits_and_idealouts_by_depth(mapper)
        mapped_qubit_labels = self._mapped_qubit_labels(mapper)
        return DirectRBDesign.from_existing_circuits(mapped_circuits_and_idealouts_by_depth,
                                                     mapped_qubit_labels,
                                                     self.circuit_type, self.sampler,
                                                     self.samplerargs, self.localclifford,
                                                     self.paulirandomize, self.descriptor,
                                                     add_default_protocol=False)



class BinaryRBDesign(_vb.BenchmarkingDesign):
    """
    Experiment design for binary randomized benchmarking.

    Encapsulates a "binary randomized benchmarking" (BiRB) experiment.

    Parameters
    ----------
    pspec : QubitProcessorSpec
       The QubitProcessorSpec for the device that the experiment is being generated for. The `pspec` is always
       handed to the sampler, as the first argument of the sampler function.

    clifford_compilation: CompilationRules
        Rules for exactly (absolutely) compiling the "native" gates of `pspec` into Clifford gates.

    depths : list of ints
        The "benchmark depth" of the circuit, which is the number of randomly sampled layers of gates in 
        the core circuit. The full BiRB circuit has depth=length+2. 

    circuits_per_depth : int
        The number of (possibly) different MRB circuits sampled at each length.

    qubit_labels : list, optional
        If not None, a list of the qubits that the RB circuit is to be sampled for. This should
        be all or a subset of the qubits in the device specified by the QubitProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
        irrelevant.

    layer_sampling: str, optional
        Determines the structure of the randomly sampled layers of gates:
            1. 'mixed1q2q': Layers contain radomly-sampled two-qubit gates and randomly-sampled 
            single-qubit gates on all remaining qubits. 
            2. 'alternating1q2q': Each layer consists of radomly-sampled two-qubit gates, with 
            all other qubits idling, followed by randomly sampled single-qubit gates on all qubits. 

    sampler : str or function, optional
        If a string, this should be one of: {'edgegrab', 'Qelimination', 'co2Qgates', 'local'}.
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


    descriptor : str, optional
        A string describing the generated experiment. Stored in the returned dictionary.

    add_default_protocol : bool, optional
        Whether to add a default RB protocol to the experiment design, which can be run
        later (once data is taken) by using a :class:`DefaultProtocolRunner` object.
    """
    def __init__(self, pspec, clifford_compilations, depths, circuits_per_depth, qubit_labels=None, layer_sampling='mixed1q2q',
                 sampler='edgegrab', samplerargs=None,
                 addlocal=False, lsargs=(),
                 descriptor='A BiRB experiment',
                 add_default_protocol=False, seed=None, verbosity=1, num_processes=1):

        if samplerargs is None:
            samplerargs = [0.25, ]
        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        measurements = []
        signs = []

        if seed is None:
            self.seed = _np.random.randint(1, 1e6)  # Pick a random seed
        else:
            self.seed = seed

        for lnum, l in enumerate(depths):
            lseed = self.seed + lnum * circuits_per_depth
            if verbosity > 0:
                print('- Sampling {} circuits at DRB length {} ({} of {} depths) with seed {}'.format(
                    circuits_per_depth, l, lnum + 1, len(depths), lseed))

            args_list = [(pspec, clifford_compilations, l)] * circuits_per_depth
            kwargs_list = [dict(qubit_labels=qubit_labels, layer_sampling=layer_sampling, sampler=sampler, samplerargs=samplerargs,
                                addlocal=addlocal, lsargs=lsargs,
                                seed=lseed + i) for i in range(circuits_per_depth)]
            #results = [_rc.create_direct_rb_circuit(*(args_list[0]), **(kwargs_list[0]))]  # num_processes == 1 case
            results = _tools.mptools.starmap_with_kwargs(_rc.create_binary_rb_circuit, circuits_per_depth,
                                                         num_processes, args_list, kwargs_list)

            circuits_at_depth = []
            measurements_at_depth = []
            signs_at_depth = []
            for c, meas, sign in results:
                circuits_at_depth.append(c)
                measurements_at_depth.append(meas)
                signs_at_depth.append(int(sign))

            circuit_lists.append(circuits_at_depth)
            measurements.append(measurements_at_depth)
            signs.append(signs_at_depth)

        self._init_foundation(depths, circuit_lists, measurements, signs, circuits_per_depth, qubit_labels, layer_sampling,
                              sampler, samplerargs, addlocal, lsargs, descriptor,
                              add_default_protocol)

    def _init_foundation(self, depths, circuit_lists, measurements, signs, circuits_per_depth, qubit_labels, layer_sampling,
                         sampler, samplerargs,  addlocal, lsargs, descriptor,
                         add_default_protocol):
        # Pair these attributes with circuit data so that we serialize/truncate properly
        self.paired_with_circuit_attrs = ["measurements", "signs"]

        super().__init__(depths, circuit_lists, signs, qubit_labels, remove_duplicates=False)
        self.measurements = measurements
        self.signs = signs
        self.circuits_per_depth = circuits_per_depth
        self.descriptor = descriptor
        self.layer_sampling = layer_sampling
        if isinstance(sampler, str):
            self.sampler = sampler
        else:
            self.sampler = 'function'
        self.samplerargs = samplerargs
        self.addlocal = addlocal
        self.lsargs = lsargs

        defaultfit = 'A-fixed'
        self.add_default_protocol(RB(name='RB', defaultfit=defaultfit))


class InterleavedRBDesign(_proto.CombinedExperimentDesign):
    """
    Experiment design for interleaved randomized benchmarking (IRB).

    IRB encapsulates a pair of "Clifford randomized benchmarking" (CRB) experiments.
    One of these CRB designs is a 'standard' one, but the other interleaves some
    clifford gate of interest between each random clifford operation. 
    The circuits created by this function will respect the connectivity and gate-set of the device encoded by 
    `pspec` (see the :class:`QubitProcessorSpec` object docstring for how to construct the relevant `pspec` 
    for a device).

    Parameters
    ----------
    pspec : QubitProcessorSpec
       The QubitProcessorSpec for the device that the CRB experiment is being generated for, which defines the 
       "native" gate-set and the connectivity of the device. The returned CRB circuits will be over the gates in 
       `pspec`, and will respect the connectivity encoded by `pspec`.

    clifford_compilations : dict
        A dictionary with the potential keys `'absolute'` and `'paulieq'` and corresponding class:`CompilationRules` values. 
        These compilation rules specify how to compile the "native" gates of `pspec` into Clifford gates.

    depths : list of ints
        The "CRB depths" of the circuit; a list of integers >= 0. The CRB length is the number of Cliffords in the 
        circuit - 2 *before* each Clifford is compiled into the native gate-set.

    circuits_per_depth : int
        The number of (possibly) different CRB circuits sampled at each length.

    interleaved_circuit : Circuit
        Circuit to use in the constuction of the interleaved CRB experiment. This is the circuit
        whose error rate is to be estimated by the IRB experiment.

    qubit_labels : list, optional
        If not None, a list of the qubits that the RB circuits are to be sampled for. This should
        be all or a subset of the qubits in the device specified by the QubitProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the "wires" in the returned circuit, but is otherwise
        irrelevant. If desired, a circuit that explicitly idles on the other qubits can be obtained
        by using methods of the Circuit object.

    randomizeout : bool, optional
        If False, the ideal output of the circuits (the "success" or "survival" outcome) is always
        the all-zeros bit string. This is probably considered to be the "standard" in CRB. If True,
        the ideal output a circuit is randomized to a uniformly random bit-string. This setting is
        useful for, e.g., detecting leakage/loss/measurement-bias etc.

    citerations : int, optional
        Some of the Clifford compilation algorithms in pyGSTi (including the default algorithm) are
        randomized, and the lowest-cost circuit is chosen from all the circuit generated in the
        iterations of the algorithm. This is the number of iterations used. The time required to
        generate a CRB circuit is linear in `citerations * (CRB length + 2)`. Lower-depth / lower 2-qubit
        gate count compilations of the Cliffords are important in order to successfully implement
        CRB on more qubits.

    compilerargs : list, optional
        A list of arguments that are handed to compile_clifford() function, which includes all the
        optional arguments of compile_clifford() *after* the `iterations` option (set by `citerations`).
        In order, this list should be values for:
        
        * algorithm : str. A string that specifies the compilation algorithm. The default in
          compile_clifford() will always be whatever we consider to be the 'best' all-round
          algorithm.
        * aargs : list. A list of optional arguments for the particular compilation algorithm.
        * costfunction : 'str' or function. The cost-function from which the "best" compilation
          for a Clifford is chosen from all `citerations` compilations. The default costs a
          circuit as 10x the num. of 2-qubit gates in the circuit + 1x the depth of the circuit.
        * prefixpaulis : bool. Whether to prefix or append the Paulis on each Clifford.
        * paulirandomize : bool. Whether to follow each layer in the Clifford circuit with a
          random Pauli on each qubit (compiled into native gates). I.e., if this is True the
          native gates are Pauli-randomized. When True, this prevents any coherent errors adding
          (on average) inside the layers of each compiled Clifford, at the cost of increased
          circuit depth. Defaults to False.
        
        For more information on these options, see the compile_clifford() docstring.

    descriptor : str, optional
        A string describing the experiment generated, which will be stored in the returned
        dictionary.

    add_default_protocol : bool, optional
        Whether to add a default RB protocol to the experiment design, which can be run
        later (once data is taken) by using a :class:`DefaultProtocolRunner` object.

    seed : int, optional
        A seed to initialize the random number generator used for creating random clifford
        circuits. The first of the two subdesigns will use the specified seed directly,
        while the second will use seed+1.

    verbosity : int, optional
        If > 0 the number of circuits generated so far is shown.

    interleave : bool, optional
        Whether the circuits of the standard CRB and IRB sub designs should be interleaved to
        form the circuit ordering of this experiment design. E.g. when calling the `all_circuits_needing_data`
        attribute.
    """

    def __init__(self, pspec, clifford_compilations, depths, circuits_per_depth, interleaved_circuit, qubit_labels=None, randomizeout=False,
                 citerations=20, compilerargs=(), exact_compilation_key=None,
                 descriptor='An Interleaved RB experiment', add_default_protocol=False, seed=None, verbosity=1, num_processes=1,
                 interleave = False):
        #Farm out the construction of the experiment designs to CliffordRBDesign:
        print('Constructing Standard CRB Subdesign:')
        crb_subdesign = CliffordRBDesign(pspec, clifford_compilations, depths, circuits_per_depth, qubit_labels, randomizeout,
                                              None, citerations, compilerargs, exact_compilation_key,
                                              descriptor + ' (Standard)', add_default_protocol, seed, verbosity, num_processes)
        print('Constructing Interleaved CRB Subdesign:')
        icrb_subdesign = CliffordRBDesign(pspec, clifford_compilations, depths, circuits_per_depth, qubit_labels, randomizeout,
                                              interleaved_circuit, citerations, compilerargs, exact_compilation_key,
                                              descriptor + ' (Interleaved)', add_default_protocol, seed+1 if seed is not None else None, 
                                              verbosity, num_processes)

        self._init_foundation(crb_subdesign, icrb_subdesign, circuits_per_depth, interleaved_circuit, randomizeout,
                              citerations, compilerargs, exact_compilation_key, interleave)

    @classmethod
    def from_existing_designs(cls, crb_subdesign, icrb_subdesign, circuits_per_depth, interleaved_circuit, randomizeout=False,
                              citerations=20, compilerargs=(), exact_compilation_key=None, interleave=False):        
        self = cls.__new__(cls)
        self._init_foundation(self, crb_subdesign, icrb_subdesign, circuits_per_depth, interleaved_circuit, randomizeout,
                              citerations, compilerargs, exact_compilation_key, interleave)

    #helper method for reducing code duplication on different class constructors.
    def _init_foundation(self, crb_subdesign, icrb_subdesign, circuits_per_depth, interleaved_circuit, randomizeout,
                              citerations, compilerargs, exact_compilation_key, interleave):
        super().__init__({'crb':crb_subdesign, 
                          'icrb':icrb_subdesign}, interleave=interleave)
        self.circuits_per_depth = circuits_per_depth
        self.randomizeout = randomizeout
        self.citerations = citerations
        self.compilerargs = compilerargs
        self.interleaved_circuit = interleaved_circuit
        self.exact_compilation_key = exact_compilation_key

        #set some auxfile information for serializing interleaved_circuit
        self.auxfile_types['interleaved_circuit'] = 'circuit-str-json'

    def average_native_gates_per_clifford(self):
        """
        The average number of native gates per Clifford for all circuits

        Returns
        -------
        tuple of floats
            A tuple of the average number of native gates per Clifford
            for the contained standard CRB design, and interleaved CRB design,
            respectively.
        """
        avg_gate_counts_crb = self['crb'].average_native_gates_per_clifford()
        avg_gate_counts_icrb = self['icrb'].average_native_gates_per_clifford()
        
        return (avg_gate_counts_crb, avg_gate_counts_icrb)

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
        InterleavedRBDesign
        """

        mapped_crb_design = self['crb'].map_qubit_labels(mapper)
        mapped_icrb_design = self['icrb'].map_qubit_labels(mapper)

        return InterleavedRBDesign.from_existing_designs(mapped_crb_design, mapped_icrb_design, self.circuits_per_depth, 
                                                         self.randomizeout, self.citerations, self.compilerargs, self.interleaved_circuit,
                                                         self.exact_compilation_key)

class RandomizedBenchmarking(_vb.SummaryStatistics):
    """
    The randomized benchmarking protocol.

    This same analysis protocol is used for Clifford, Direct and Mirror RB.
    The standard Mirror RB analysis is obtained by setting
    `datatype` = `adjusted_success_probabilities`.
    """

    def __init__(self, datatype='success_probabilities', defaultfit='full', asymptote='std', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', name=None):
        """
        Initialize an RB protocol for analyzing RB data.

        Parameters
        ----------
        datatype: 'success_probabilities', 'adjusted_success_probabilities', or 'energies', optional
            The type of summary data to extract, average, and the fit to an exponential decay. If
            'success_probabilities' then the summary data for a circuit is the frequency that
            the target bitstring is observed, i.e., the success probability of the circuit. If
            'adjusted_success_probabilties' then the summary data for a circuit is
            S = sum_{k = 0}^n (-1/2)^k h_k where h_k is the frequency at which the output bitstring is
            a Hamming distance of k from the target bitstring, and n is the number of qubits.
            This datatype is used in Mirror RB, but can also be used in Clifford and Direct RB. 
            If 'energies',  then the summary data is Pauli operator measurement results. This datatype is
            only used for Binary RB. 
            

        defaultfit: 'A-fixed' or 'full'
            The summary data is fit to A + Bp^m with A fixed and with A as a fit parameter.
            If 'A-fixed' then the default results displayed are those from fitting with A
            fixed, and if 'full' then the default results displayed are those where A is a
            fit parameter.

        asymptote : 'std' or float, optional
            The summary data is fit to A + Bp^m with A fixed and with A has a fit parameter,
            with the default results returned set by `defaultfit`. This argument specifies the
            value used when 'A' is fixed. If left as 'std', then 'A' defaults to 1/2^n if
            `datatype` is `success_probabilities` and to 1/4^n if `datatype` is
            `adjusted_success_probabilities`.

        rtype : 'EI' or 'AGI', optional
            The RB error rate definition convention. 'EI' results in RB error rates that are associated
            with the entanglement infidelity, which is the error probability with stochastic Pauli errors.
            'AGI' results in RB error rates that are associated with the average gate infidelity.

        seed : list, optional
            Seeds for the fit of B and p (A is seeded to the asymptote defined by `asympote`).

        bootstrap_samples : float, optional
            The number of samples for generating bootstrapped error bars.

        depths: list or 'all'
            If not 'all', a list of depths to use (data at other depths is discarded).

        name : str, optional
            The name of this protocol, also used to (by default) name the
            results produced by this protocol.  If None, the class name will
            be used.
        """
        super().__init__(name)

        assert(datatype in self.summary_statistics), "Unknown data type: %s!" % str(datatype)
        assert(datatype in ('success_probabilities', 'adjusted_success_probabilities', 'energies')), \
            "Data type '%s' must be 'success_probabilities', 'adjusted_success_probabilities', or 'energies'!" % str(datatype)

        self.seed = seed
        self.depths = depths
        self.bootstrap_samples = bootstrap_samples
        self.asymptote = asymptote
        self.rtype = rtype
        self.datatype = datatype
        self.defaultfit = defaultfit
        if self.datatype == 'energies':
            self.energies = True
        else:
            self.energies = False

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        RandomizedBenchmarkingResults
        """
        design = data.edesign

        if isinstance(design, BinaryRBDesign):
            assert(self.datatype=='energies'),  'dataype=energies is required for Binary RB data!'

        if self.datatype not in data.cache:
            summary_data_dict = self._compute_summary_statistics(data, energy = self.energies)
            data.cache.update(summary_data_dict)
        src_data = data.cache[self.datatype]
        data_per_depth = src_data

        if self.depths == 'all':
            depths = list(data_per_depth.keys())
        else:
            depths = filter(lambda d: d in data_per_depth, self.depths)

        nqubits = len(design.qubit_labels)

        if isinstance(self.asymptote, str):
            assert(self.asymptote == 'std'), "If `asymptote` is a string it must be 'std'!"
            if self.datatype == 'success_probabilities':
                asymptote = 1 / 2**nqubits
            elif self.datatype == 'adjusted_success_probabilities':
                asymptote = 1 / 4**nqubits
            elif self.datatype == 'energies':
                asymptote = 0
            else:
                raise ValueError("No 'std' asymptote for %s datatype!" % self.asymptote)

        def _get_rb_fits(circuitdata_per_depth):
            adj_sps = []
            for depth in depths:
                percircuitdata = circuitdata_per_depth[depth]
                adj_sps.append(_np.nanmean(percircuitdata))  # average [adjusted] success probabilities or energies

            # Don't think this needs changed
            full_fit_results, fixed_asym_fit_results = _rbfit.std_least_squares_fit(
                depths, adj_sps, nqubits, seed=self.seed, asymptote=asymptote,
                ftype='full+FA', rtype=self.rtype)

            return full_fit_results, fixed_asym_fit_results

        #do RB fit on actual data
        # Think this works just fine
        ff_results, faf_results = _get_rb_fits(data_per_depth)

        if self.bootstrap_samples > 0:

            parameters = ['a', 'b', 'p', 'r']
            bootstraps_ff = {p: [] for p in parameters}
            bootstraps_faf = {p: [] for p in parameters}
            failcount_ff = 0
            failcount_faf = 0

            #Store bootstrap "cache" dicts (containing summary keys) as a list under data.cache
            if 'bootstraps' not in data.cache or len(data.cache['bootstraps']) < self.bootstrap_samples:
                # TIM - finite counts always True here?
                self._add_bootstrap_qtys(data.cache, self.bootstrap_samples, finitecounts=True)
            bootstrap_caches = data.cache['bootstraps']  # if finitecounts else 'infbootstraps'

            for bootstrap_cache in bootstrap_caches:
                bs_ff_results, bs_faf_results = _get_rb_fits(bootstrap_cache[self.datatype])

                if bs_ff_results['success']:
                    for p in parameters:
                        bootstraps_ff[p].append(bs_ff_results['estimates'][p])
                else:
                    failcount_ff += 1
                if bs_faf_results['success']:
                    for p in parameters:
                        bootstraps_faf[p].append(bs_faf_results['estimates'][p])
                else:
                    failcount_faf += 1

            failrate_ff = failcount_ff / self.bootstrap_samples
            failrate_faf = failcount_faf / self.bootstrap_samples

            std_ff = {p: _np.std(_np.array(bootstraps_ff[p])) for p in parameters}
            std_faf = {p: _np.std(_np.array(bootstraps_faf[p])) for p in parameters}

        else:
            bootstraps_ff = None
            std_ff = None
            failrate_ff = None

            bootstraps_faf = None
            std_faf = None
            failrate_faf = None
        # we are here
        fits = _tools.NamedDict('FitType', 'category')
        fits['full'] = _rbfit.FitResults(
            'LS', ff_results['seed'], self.rtype, ff_results['success'], ff_results['estimates'],
            ff_results['variable'], stds=std_ff, bootstraps=bootstraps_ff,
            bootstraps_failrate=failrate_ff)

        fits['A-fixed'] = _rbfit.FitResults(
            'LS', faf_results['seed'], self.rtype, faf_results['success'],
            faf_results['estimates'], faf_results['variable'], stds=std_faf,
            bootstraps=bootstraps_faf, bootstraps_failrate=failrate_faf)

        return RandomizedBenchmarkingResults(data, self, fits, depths, self.defaultfit)


class RandomizedBenchmarkingResults(_proto.ProtocolResults):
    """
    The results of running randomized benchmarking.

    Parameters
    ----------
    data : ProtocolData
        The experimental data these results are generated from.

    protocol_instance : Protocol
        The protocol that generated these results.

    fits : dict
        A dictionary of RB fit parameters.

    depths : list or tuple
        A sequence of the depths used in the RB experiment. The x-values
        of the RB fit curve.

    defaultfit : str
        The default key within `fits` to plot when calling :meth:`plot`.
    """

    def __init__(self, data, protocol_instance, fits, depths, defaultfit):
        """
        Initialize an empty RandomizedBenchmarkingResults object.
        """
        super().__init__(data, protocol_instance)

        self.depths = depths  # Note: can be different from protocol_instance.depths (which can be 'all')
        self.rtype = protocol_instance.rtype  # replicated for convenience?
        self.fits = fits
        self.defaultfit = defaultfit
        self.auxfile_types['fits'] = 'dict:serialized-object'  # b/c NamedDict don't json

    def plot(self, fitkey=None, decay=True, success_probabilities=True, size=(8, 5), ylim=None, xlim=None,
             legend=True, title=None, figpath=None):
        """
        Plots RB data and, optionally, a fitted exponential decay.

        Parameters
        ----------
        fitkey : dict key, optional
            The key of the self.fits dictionary to plot the fit for. If None, will
            look for a 'full' key (the key for a full fit to A + Bp^m if the standard
            analysis functions are used) and plot this if possible. It otherwise checks
            that there is only one key in the dict and defaults to this. If there are
            multiple keys and none of them are 'full', `fitkey` must be specified when
            `decay` is True.

        decay : bool, optional
            Whether to plot a fit, or just the data.

        success_probabilities : bool, optional
            Whether to plot the success probabilities distribution, as a violin plot. (as well
            as the *average* success probabilities at each length).

        size : tuple, optional
            The figure size

        ylim : tuple, optional
            The y-axis range.

        xlim : tuple, optional
            The x-axis range.

        legend : bool, optional
            Whether to show a legend.

        title : str, optional
            A title to put on the figure.

        figpath : str, optional
            If specified, the figure is saved with this filename.

        Returns
        -------
        None
        """

        # Future : change to a plotly plot.
        try: import matplotlib.pyplot as _plt
        except ImportError: raise ValueError("This function requires you to install matplotlib!")

        if decay and fitkey is None:
            if self.defaultfit is not None:
                fitkey = self.defaultfit
            else:
                allfitkeys = list(self.fits.keys())
                if 'full' in allfitkeys: fitkey = 'full'
                else:
                    assert(len(allfitkeys) == 1), \
                        ("There are multiple fits, there is no defaultfit and none have the key "
                         "'full'. Please specify the fit to plot!")
                    fitkey = allfitkeys[0]

        adj_sps = []
        data_per_depth = self.data.cache[self.protocol.datatype]
        for depth in self.depths:
            percircuitdata = data_per_depth[depth]
            adj_sps.append(_np.mean(percircuitdata))  # average [adjusted] success probabilities

        _plt.figure(figsize=size)
        _plt.plot(self.depths, adj_sps, 'o', label='Average success probabilities')

        if decay:
            lengths = _np.linspace(0, max(self.depths), 200)
            a = self.fits[fitkey].estimates['a']
            b = self.fits[fitkey].estimates['b']
            p = self.fits[fitkey].estimates['p']
            _plt.plot(lengths, a + b * p**lengths,
                      label='Fit, r = {:.2} +/- {:.1}'.format(self.fits[fitkey].estimates['r'],
                                                              self.fits[fitkey].stds['r']))
            #_plt.plot(lengths, a + b * p**lengths,
            #          label='Fit, r = {:.2}'.format(self.fits[fitkey].estimates['r']))

        if success_probabilities:
            all_success_probs_by_depth = [data_per_depth[depth] for depth in self.depths]
            _plt.violinplot(all_success_probs_by_depth, self.depths, points=10, widths=1.,
                            showmeans=False, showextrema=False, showmedians=False)  # , label='Success probabilities')

        if title is not None: _plt.title(title)
        _plt.ylabel("Success probability")
        _plt.xlabel("RB depth $(m)$")
        _plt.ylim(ylim)
        _plt.xlim(xlim)

        if legend: _plt.legend()

        if figpath is not None: _plt.savefig(figpath, dpi=1000)
        else: _plt.show()

        return
    

    def copy(self):
        """
        Creates a copy of this :class:`RandomizedBenchmarkingResults` object.

        Returns
        -------
        RandomizedBenchmarkingResults
        """
        #TODO: check whether this deep copies (if we want it to...) - I expect it doesn't currently
        data = _proto.ProtocolData(self.data.edesign, self.data.dataset)
        cpy = RandomizedBenchmarkingResults(data, self.protocol, self.fits, self.depths, self.defaultfit)
        return cpy
    

class InterleavedRandomizedBenchmarking(_proto.Protocol):
    """
    The interleaved randomized benchmarking protocol.

    This object itself utilizes the RandomizedBenchmarking protocol to
    perform the analysis for the standard CRB and interleaved RB subexperiments
    that constitute the IRB process. As such, this class takes as input
    the subset of RandomizedBenchmarking's arguments relevant for CRB.
    """

    def __init__(self, defaultfit='full', asymptote='std', rtype='EI', seed=(0.8, 0.95), 
                 bootstrap_samples=200, depths='all', name=None):
        """
        Initialize an RB protocol for analyzing RB data.

        Parameters
        ----------
        defaultfit: 'A-fixed' or 'full'
            The summary data is fit to A + Bp^m with A fixed and with A as a fit parameter.
            If 'A-fixed' then the default results displayed are those from fitting with A
            fixed, and if 'full' then the default results displayed are those where A is a
            fit parameter.

        asymptote : 'std' or float, optional
            The summary data is fit to A + Bp^m with A fixed and with A has a fit parameter,
            with the default results returned set by `defaultfit`. This argument specifies the
            value used when 'A' is fixed. If left as 'std', then 'A' defaults to 1/2^n if
            `datatype` is `success_probabilities` and to 1/4^n if `datatype` is
            `adjusted_success_probabilities`.

        rtype : 'EI' or 'AGI', optional
            The RB error rate definition convention. 'EI' results in RB error rates that are associated
            with the entanglement infidelity, which is the error probability with stochastic Pauli errors.
            'AGI' results in RB error rates that are associated with the average gate infidelity.

        seed : list, optional
            Seeds for the fit of B and p (A is seeded to the asymptote defined by `asympote`).

        bootstrap_samples : float, optional
            The number of samples for generating bootstrapped error bars.

        depths: list or 'all'
            If not 'all', a list of depths to use (data at other depths is discarded).

        name : str, optional
            The name of this protocol, also used to (by default) name the
            results produced by this protocol.  If None, the class name will
            be used.
        """
        super().__init__(name)
        self.seed = seed
        self.depths = depths
        self.bootstrap_samples = bootstrap_samples
        self.asymptote = asymptote
        self.rtype = rtype
        self.datatype = 'success_probabilities'
        self.defaultfit = defaultfit

    def run(self, data, memlimit=None, comm=None):
        """
        Run this protocol on `data`.

        Parameters
        ----------
        data : ProtocolData
            The input data.

        memlimit : int, optional
            A rough per-processor memory limit in bytes.

        comm : mpi4py.MPI.Comm, optional
            When not ``None``, an MPI communicator used to run this protocol
            in parallel.

        Returns
        -------
        RandomizedBenchmarkingResults
        """
        design = data.edesign
        assert(isinstance(design, InterleavedRBDesign)), 'This protocol can only be run on InterleavedRBDesign.' 
        #initialize a RandomizedBenchmarking protocol to use as a helper
        #for performing analysis on the two subexperiments.
        rb_protocol = RandomizedBenchmarking('success_probabilities', self.defaultfit, self.asymptote, self.rtype,
                                             self.seed, self.bootstrap_samples, self.depths, name=None)

        #run the RB protocol on both subdesigns.
        crb_results = rb_protocol.run(data['crb'])
        icrb_results = rb_protocol.run(data['icrb'])

        nqubits = len(design.qubit_labels)
        #let the dimension depend on the value of rtype.
        dim = 2**nqubits
        if self.rtype == 'EI':
            dim_prefactor = (dim**2 -1)/(dim**2)
        elif self.rtype == 'AGI':
            dim_prefactor = (dim -1)/dim
        else:
            raise ValueError('Only EI and AGI type IRB numbers are currently implemented.')

        irb_numbers = dict()
        irb_bounds = dict()
        #use the crb and icrb results to get the irb number and the bounds.
        for fit_key in crb_results.fits.keys():
            p_crb = crb_results.fits[fit_key].estimates['p']
            p_icrb = icrb_results.fits[fit_key].estimates['p']
            irb_numbers[fit_key] = dim_prefactor*(1-(p_icrb/p_crb))
            #Magesan paper gives the bounds as the minimum of two quantities.
            possible_bound_1 = dim_prefactor * (abs(p_crb - (p_icrb/p_crb)) + (1 - p_crb))
            possible_bound_2 = (2*(dim**2-1)*(1-p_crb))/(p_crb*dim**2) + (4*_np.sqrt(1-p_crb)*_np.sqrt(dim**2 - 1))/p_crb
            #The value of the possible_bound_2 coming directly from the Magesan paper should be in units of AGI.
            #So if we want EI use the standard dimensional conversion factor.
            if self.rtype == 'EI':
                possible_bound_2 = ((dim + 1)/dim)*possible_bound_2
            irb_bounds[fit_key] = min(possible_bound_1, possible_bound_2)
        
        children = {'crb': _proto.ProtocolResultsDir(data['crb'], crb_results),
                    'icrb': _proto.ProtocolResultsDir(data['icrb'], icrb_results)}

        irb_top_results = InterleavedRandomizedBenchmarkingResults(data, self, irb_numbers, irb_bounds)

        return _proto.ProtocolResultsDir(data, irb_top_results, children = children)

class InterleavedRandomizedBenchmarkingResults(_proto.ProtocolResults):
    """
    Class for storing the results of an interleaved randomized benchmarking experiment.
    This subclasses off of ProtocolResultsDir as this class acts primarily as both a container
    class for holding the two subexperiment's results, as well as containing some specialized
    information regarding the IRB number estimates.
    """

    def __init__(self, data, protocol, irb_numbers, irb_bounds):
        #msg = 'rb_subexperiment_results should be a dictionary with values corresponding to the'\
        #      +' standard CRB and interleaved CRB subexperiments used in performing IRB.'
        #assert(isinstance(rb_subexperiment_results, dict)), msg
        #super().__init__(data, rb_subexperiment_results)
        super().__init__(data, protocol)

        self.irb_numbers = irb_numbers
        self.irb_bounds = irb_bounds


RB = RandomizedBenchmarking
RBResults = RandomizedBenchmarkingResults  # shorthand
