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

import numpy as _np

from . import protocol as _proto
from . import vb as _vb
from .. import tools as _tools
from ..algorithms import randomcircuit as _rc
from ..algorithms import rbfit as _rbfit


class CliffordRBDesign(_vb.BenchmarkingDesign):
    """
    Experiment design for Clifford randomized benchmarking.

    This encapsulates a "Clifford randomized benchmarking" (CRB) experiment.  CRB is the RB protocol defined
    in "Scalable and robust randomized benchmarking of quantum processes", Magesan et al. PRL 106 180504 (2011).
    The circuits created by this function will respect the connectivity and gate-set of the device encoded
    by `pspec` (see the :class:`ProcessorSpec` object docstring for how to construct the relevant `pspec` for a device).

    Note that this function uses the convention that a depth "l" CRB circuit  consists of "l"+2 Clifford gates
    before compilation.

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the CRB experiment is being generated for, which defines the
       "native" gate-set and the connectivity of the device. The returned CRB circuits will be over
       the gates in `pspec`, and will respect the connectivity encoded by `pspec`.

    depths : list of ints
        The "CRB depths" of the circuit; a list of integers >= 0. The CRB length is the number of Cliffords
        in the circuit - 2 *before* each Clifford is compiled into the native gate-set.

    circuits_per_depth : int
        The number of (possibly) different CRB circuits sampled at each length.

    qubit_labels : list, optional
        If not None, a list of the qubits that the RB circuits are to be sampled for. This should
        be all or a subset of the qubits in the device specified by the ProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
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
            - algorithm : str. A string that specifies the compilation algorithm. The default in
                compile_clifford() will always be whatever we consider to be the 'best' all-round
                algorith,
            - aargs : list. A list of optional arguments for the particular compilation algorithm.
            - costfunction : 'str' or function. The cost-function from which the "best" compilation
                for a Clifford is chosen from all `citerations` compilations. The default costs a
                circuit as 10x the num. of 2-qubit gates in the circuit + 1x the depth of the circuit.
            - prefixpaulis : bool. Whether to prefix or append the Paulis on each Clifford.
            - paulirandomize : bool. Whether to follow each layer in the Clifford circuit with a
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
    def from_existing_circuits(cls, circuits_and_idealouts_by_depth, qubit_labels=None,
                               randomizeout=False, citerations=20, compilerargs=(), interleaved_circuit=None,
                               descriptor='A Clifford RB experiment', add_default_protocol=False):
        """
        Create a :class:`CliffordRBDesign` from an existing set of sampled RB circuits.

        This function serves as an alternative to the usual method of creating a Clifford
        RB experiment design by sampling a number of circuits randomly.  This function
        takes a list of previously-sampled random circuits and does not sampling internally.

        Parameters
        ----------
        circuits_and_idealouts_by_depth : dict
            A dictionary whose keys are integer depths and whose values are lists
            of `(circuit, ideal_outcome)` 2-tuples giving each RB circuit and its
            ideal (correct) outcome.

        qubit_labels : list, optional
            If not None, a list of the qubits that the RB circuits are to be sampled for. This should
            be all or a subset of the qubits in the device specified by the ProcessorSpec `pspec`.
            If None, it is assumed that the RB circuit should be over all the qubits. Note that the
            ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
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
                - algorithm : str. A string that specifies the compilation algorithm. The default in
                    compile_clifford() will always be whatever we consider to be the 'best' all-round
                    algorith,
                - aargs : list. A list of optional arguments for the particular compilation algorithm.
                - costfunction : 'str' or function. The cost-function from which the "best" compilation
                    for a Clifford is chosen from all `citerations` compilations. The default costs a
                    circuit as 10x the num. of 2-qubit gates in the circuit + 1x the depth of the circuit.
                - prefixpaulis : bool. Whether to prefix or append the Paulis on each Clifford.
                - paulirandomize : bool. Whether to follow each layer in the Clifford circuit with a
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
        depths = sorted(list(circuits_and_idealouts_by_depth.keys()))
        circuit_lists = [[x[0] for x in circuits_and_idealouts_by_depth[d]] for d in depths]
        ideal_outs = [[x[1] for x in circuits_and_idealouts_by_depth[d]] for d in depths]
        circuits_per_depth = [len(circuits_and_idealouts_by_depth[d]) for d in depths]
        self = cls.__new__(cls)
        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              randomizeout, citerations, compilerargs, descriptor, add_default_protocol,
                              interleaved_circuit)
        return self

    def __init__(self, pspec, depths, circuits_per_depth, qubit_labels=None, randomizeout=False,
                 interleaved_circuit=None,
                 citerations=20, compilerargs=(), descriptor='A Clifford RB experiment',
                 add_default_protocol=False, seed=None, verbosity=1, num_processes=1):
        """
        Generates a "Clifford randomized benchmarking" (CRB) experiment, which is the RB protocol defined
        in "Scalable and robust randomized benchmarking of quantum processes", Magesan et al. PRL 106 180504 (2011).
        The circuits created by this function will respect the connectivity and gate-set of the device encoded
        by `pspec` (see the ProcessorSpec object docstring for how to construct the relevant `pspec` for a device).

        Note that this function uses the convention that a depth "l" CRB circuit  consists of "l"+2 Clifford gates
        before compilation.

        Parameters
        ----------
        pspec : ProcessorSpec
           The ProcessorSpec for the device that the CRB experiment is being generated for, which defines the
           "native" gate-set and the connectivity of the device. The returned CRB circuits will be over
           the gates in `pspec`, and will respect the connectivity encoded by `pspec`.

        depths : list of ints
            The "CRB depths" of the circuit; a list of integers >= 0. The CRB length is the number of Cliffords
            in the circuit - 2 *before* each Clifford is compiled into the native gate-set.

        circuits_per_depth : int
            The number of (possibly) different CRB circuits sampled at each length.

        qubit_labels : list, optional
            If not None, a list of the qubits that the RB circuits are to be sampled for. This should
            be all or a subset of the qubits in the device specified by the ProcessorSpec `pspec`.
            If None, it is assumed that the RB circuit should be over all the qubits. Note that the
            ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
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
                - algorithm : str. A string that specifies the compilation algorithm. The default in
                    compile_clifford() will always be whatever we consider to be the 'best' all-round
                    algorith,
                - aargs : list. A list of optional arguments for the particular compilation algorithm.
                - costfunction : 'str' or function. The cost-function from which the "best" compilation
                    for a Clifford is chosen from all `citerations` compilations. The default costs a
                    circuit as 10x the num. of 2-qubit gates in the circuit + 1x the depth of the circuit.
                - prefixpaulis : bool. Whether to prefix or append the Paulis on each Clifford.
                - paulirandomize : bool. Whether to follow each layer in the Clifford circuit with a
                    random Pauli on each qubit (compiled into native gates). I.e., if this is True the
                    native gates are Pauli-randomized. When True, this prevents any coherent errors adding
                    (on average) inside the layers of each compiled Clifford, at the cost of increased
                    circuit depth. Defaults to False.
            For more information on these options, see the compile_clifford() docstring.

        decscriptor : str, optional
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

        num_processes : int, optional
            Number of processes to parallelize circuit creation over. Defaults to 1

        Returns
        -------
        CliffordRBDesign
        """
        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        ideal_outs = []
        if seed is not None:
            _np.random.seed(seed)

        for lnum, l in enumerate(depths):
            if verbosity > 0:
                print('- Sampling {} circuits at CRB length {} ({} of {} depths)'.format(circuits_per_depth, l,
                                                                                         lnum + 1, len(depths)))

            results = _tools.mptools.starmap_with_kwargs(_rc.create_clifford_rb_circuit, circuits_per_depth,
                                                         num_processes, pspec, l, qubit_labels=qubit_labels,
                                                         randomizeout=randomizeout, citerations=citerations,
                                                         compilerargs=compilerargs,
                                                         interleaved_circuit=interleaved_circuit)

            circuits_at_depth = []
            idealouts_at_depth = []
            for c, iout in results:
                circuits_at_depth.append(c)
                idealouts_at_depth.append((''.join(map(str, iout)),))

            circuit_lists.append(circuits_at_depth)
            ideal_outs.append(idealouts_at_depth)

        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              randomizeout, citerations, compilerargs, descriptor, add_default_protocol,
                              interleaved_circuit)

    def _init_foundation(self, depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                         randomizeout, citerations, compilerargs, descriptor, add_default_protocol,
                         interleaved_circuit):
        super().__init__(depths, circuit_lists, ideal_outs, qubit_labels, remove_duplicates=False)
        self.circuits_per_depth = circuits_per_depth
        self.randomizeout = randomizeout
        self.citerations = citerations
        self.compilerargs = compilerargs
        self.descriptor = descriptor
        self.interleaved_circuit = interleaved_circuit
        if add_default_protocol:
            if randomizeout:
                defaultfit = 'A-fixed'
            else:
                defaultfit = 'full'
            self.add_default_protocol(RB(name='RB', defaultfit=defaultfit))


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
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the circuit is being sampled for, which defines the
       "native" gate-set and the connectivity of the device. The returned DRB circuit will be over
       the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
       is always handed to the sampler, as the first argument of the sampler function (this is only
       of importance when not using an in-built sampler for the "core" of the DRB circuit). Unless
       `qubit_labels` is not None, the circuit is sampled over all the qubits in `pspec`.

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
        If a string, this should be one of:
            {'edgegrab', pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function
        in rb.sampler named circuit_layer_by_* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
        a valid form of sampling for n-qubit DRB, but is not explicitly forbidden in this function].
        If `sampler` is a function, it should be a function that takes as the first argument a
        ProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
        the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is the
        only sampler that requires no parameters beyond the ProcessorSpec *and* works for arbitrary
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
                               sampler='Qelimination', samplerargs=[], addlocal=False,
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
            If a string, this should be one of:
                {'edgegrab', pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
            Except for 'local', this corresponds to sampling layers according to the sampling function
            in rb.sampler named circuit_layer_by_* (with * replaced by 'sampler'). For 'local', this
            corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
            a valid form of sampling for n-qubit DRB, but is not explicitly forbidden in this function].
            If `sampler` is a function, it should be a function that takes as the first argument a
            ProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
            the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is the
            only sampler that requires no parameters beyond the ProcessorSpec *and* works for arbitrary
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

        Returns
        -------
        DirectRBDesign
        """
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

    def __init__(self, pspec, depths, circuits_per_depth, qubit_labels=None, sampler='Qelimination', samplerargs=[],
                 addlocal=False, lsargs=(), randomizeout=False, cliffordtwirl=True, conditionaltwirl=True,
                 citerations=20, compilerargs=(), partitioned=False, descriptor='A DRB experiment',
                 add_default_protocol=False, seed=None, verbosity=1, num_processes=1):
        """
        Generates a "direct randomized benchmarking" (DRB) experiments, which is the protocol introduced in
        arXiv:1807.07975 (2018).

        An n-qubit DRB circuit consists of (1) a circuit the prepares a uniformly random stabilizer state;
        (2) a length-l circuit (specified by `length`) consisting of circuit layers sampled according to
        some user-specified distribution (specified by `sampler`), (3) a circuit that maps the output of
        the preceeding circuit to a computational basis state. See arXiv:1807.07975 (2018) for further
        details.

        Parameters
        ----------
        pspec : ProcessorSpec
           The ProcessorSpec for the device that the circuit is being sampled for, which defines the
           "native" gate-set and the connectivity of the device. The returned DRB circuit will be over
           the gates in `pspec`, and will respect the connectivity encoded by `pspec`. Note that `pspec`
           is always handed to the sampler, as the first argument of the sampler function (this is only
           of importance when not using an in-built sampler for the "core" of the DRB circuit). Unless
           `qubit_labels` is not None, the circuit is sampled over all the qubits in `pspec`.

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
            If a string, this should be one of:
                {'edgegrab', pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
            Except for 'local', this corresponds to sampling layers according to the sampling function
            in rb.sampler named circuit_layer_by_* (with * replaced by 'sampler'). For 'local', this
            corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
            a valid form of sampling for n-qubit DRB, but is not explicitly forbidden in this function].
            If `sampler` is a function, it should be a function that takes as the first argument a
            ProcessorSpec, and returns a random circuit layer as a list of gate Label objects. Note that
            the default 'Qelimination' is not necessarily the most useful in-built sampler, but it is the
            only sampler that requires no parameters beyond the ProcessorSpec *and* works for arbitrary
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

        num_processes : int, optional
            Number of processes to parallelize circuit creation over. Defaults to 1

        Returns
        -------
        DirectRBDesign
        """

        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        ideal_outs = []
        if seed is not None:
            _np.random.seed(seed)

        for lnum, l in enumerate(depths):
            if verbosity > 0:
                print('- Sampling {} circuits at DRB length {} ({} of {} depths)'.format(circuits_per_depth, l,
                                                                                         lnum + 1, len(depths)))

            results = _tools.mptools.starmap_with_kwargs(_rc.create_direct_rb_circuit, circuits_per_depth,
                                                         num_processes, pspec, l, qubit_labels=qubit_labels,
                                                         sampler=sampler, samplerargs=samplerargs, addlocal=addlocal,
                                                         lsargs=lsargs, randomizeout=randomizeout,
                                                         cliffordtwirl=cliffordtwirl, conditionaltwirl=conditionaltwirl,
                                                         citerations=citerations, compilerargs=compilerargs,
                                                         partitioned=partitioned)

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


class MirrorRBDesign(_vb.BenchmarkingDesign):
    """
    Experiment design for mirror randomized benchmarking.

    Encapsulates a "mirror randomized benchmarking" (MRB) experiment, for the case of Clifford gates and with
    the option of Pauli randomization and local Clifford twirling. To implement mirror RB it is necessary
    for U^(-1) to in the gate set for every gate U in the gate set.

    **THIS METHOD IS IN DEVELOPEMENT. DO NOT EXPECT THAT THIS FUNCTION WILL BEHAVE THE SAME IN FUTURE RELEASES
    OF PYGSTI!**

    Parameters
    ----------
    pspec : ProcessorSpec
       The ProcessorSpec for the device that the experiment is being generated for. The `pspec` is always
       handed to the sampler, as the first argument of the sampler function.

    depths : list of ints
        The "mirror RB depths" of the circuits, which is closely related to the circuit depth. A MRB
        length must be an even integer, and can be zero.

        - If `localclifford` and `paulirandomize` are False, the depth of a sampled circuit = the MRB length.
          The first length/2 layers are all sampled independently according to the sampler specified by
          `sampler`. The remaining half of the circuit is the "inversion" circuit that is determined
          by the first half.

        - If `paulirandomize` is True and `localclifford` is False, the depth of a circuit is
          2*length+1 with odd-indexed layers sampled according to the sampler specified by `sampler, and
          the the zeroth layer + the even-indexed layers consisting of random 1-qubit Pauli gates.

        - If `paulirandomize` and `localclifford` are True, the depth of a circuit is
          2*length+1 + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for
          the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

        - If `paulirandomize` is False and `localclifford` is True, the depth of a circuit is
          length + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for
          the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

    circuits_per_depth : int
        The number of (possibly) different MRB circuits sampled at each length.

    qubit_labels : list, optional
        If not None, a list of the qubits that the RB circuit is to be sampled for. This should
        be all or a subset of the qubits in the device specified by the ProcessorSpec `pspec`.
        If None, it is assumed that the RB circuit should be over all the qubits. Note that the
        ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
        irrelevant.

    sampler : str or function, optional
        If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
        Except for 'local', this corresponds to sampling layers according to the sampling function
        in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
        corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
        a valid option for n-qubit MRB -- it results in sim. 1-qubit MRB -- but it is not explicitly
        forbidden by this function]. If `sampler` is a function, it should be a function that takes
        as the first argument a ProcessorSpec, and returns a random circuit layer as a list of gate
        Label objects. Note that the default 'Qelimination' is not necessarily the most useful
        in-built sampler, but it is the only sampler that requires no parameters beyond the ProcessorSpec
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
                               sampler='Qelimination', samplerargs=(), localclifford=True,
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

        qubit_labels : list, optional
            If not None, a list of the qubits that the RB circuit is to be sampled for. This should
            be all or a subset of the qubits in the device specified by the ProcessorSpec `pspec`.
            If None, it is assumed that the RB circuit should be over all the qubits. Note that the
            ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
            irrelevant.

        sampler : str or function, optional
            If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
            Except for 'local', this corresponds to sampling layers according to the sampling function
            in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
            corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
            a valid option for n-qubit MRB -- it results in sim. 1-qubit MRB -- but it is not explicitly
            forbidden by this function]. If `sampler` is a function, it should be a function that takes
            as the first argument a ProcessorSpec, and returns a random circuit layer as a list of gate
            Label objects. Note that the default 'Qelimination' is not necessarily the most useful
            in-built sampler, but it is the only sampler that requires no parameters beyond the ProcessorSpec
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
                              sampler, samplerargs, localclifford, paulirandomize, descriptor,
                              add_default_protocol)
        return self

    def __init__(self, pspec, depths, circuits_per_depth, qubit_labels=None, sampler='Qelimination', samplerargs=(),
                 localclifford=True, paulirandomize=True, descriptor='A mirror RB experiment',
                 add_default_protocol=False, num_processes=1, verbosity=1):
        """
        Generates a "mirror randomized benchmarking" (MRB) experiment, for the case of Clifford gates and with
        the option of Pauli randomization and local Clifford twirling. To implement mirror RB it is necessary
        for U^(-1) to in the gate set for every gate U in the gate set.

        THIS METHOD IS IN DEVELOPEMENT. DO NOT EXPECT THAT THIS FUNCTION WILL BEHAVE THE SAME IN FUTURE RELEASES
        OF PYGSTI!

        Parameters
        ----------
        pspec : ProcessorSpec
           The ProcessorSpec for the device that the experiment is being generated for. The `pspec` is always
           handed to the sampler, as the first argument of the sampler function.

        depths : list of ints
            The "mirror RB depths" of the circuits, which is closely related to the circuit depth. A MRB
            length must be an even integer, and can be zero.

            - If `localclifford` and `paulirandomize` are False, the depth of a sampled circuit = the MRB length.
              The first length/2 layers are all sampled independently according to the sampler specified by
              `sampler`. The remaining half of the circuit is the "inversion" circuit that is determined
              by the first half.

            - If `paulirandomize` is True and `localclifford` is False, the depth of a circuit is
              2*length+1 with odd-indexed layers sampled according to the sampler specified by `sampler, and
              the the zeroth layer + the even-indexed layers consisting of random 1-qubit Pauli gates.

            - If `paulirandomize` and `localclifford` are True, the depth of a circuit is
              2*length+1 + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for
              the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

            - If `paulirandomize` is False and `localclifford` is True, the depth of a circuit is
              length + X where X is a random variable (between 0 and normally <= ~12-16) that accounts for
              the depth from the layer of random 1-qubit Cliffords at the start and end of the circuit.

        circuits_per_depth : int
            The number of (possibly) different MRB circuits sampled at each length.

        qubit_labels : list, optional
            If not None, a list of the qubits that the RB circuit is to be sampled for. This should
            be all or a subset of the qubits in the device specified by the ProcessorSpec `pspec`.
            If None, it is assumed that the RB circuit should be over all the qubits. Note that the
            ordering of this list is the order of the ``wires'' in the returned circuit, but is otherwise
            irrelevant.

        sampler : str or function, optional
            If a string, this should be one of: {'pairingQs', 'Qelimination', 'co2Qgates', 'local'}.
            Except for 'local', this corresponds to sampling layers according to the sampling function
            in rb.sampler named circuit_layer_by* (with * replaced by 'sampler'). For 'local', this
            corresponds to sampling according to rb.sampler.circuit_layer_of_oneQgates [which is not
            a valid option for n-qubit MRB -- it results in sim. 1-qubit MRB -- but it is not explicitly
            forbidden by this function]. If `sampler` is a function, it should be a function that takes
            as the first argument a ProcessorSpec, and returns a random circuit layer as a list of gate
            Label objects. Note that the default 'Qelimination' is not necessarily the most useful
            in-built sampler, but it is the only sampler that requires no parameters beyond the ProcessorSpec
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

        num_processes : int, optional
            Number of processes to parallelize circuit creation over. Defaults to 1

        verbosity : int, optional
            If > 0 the number of depths for which circuits have been generated so far.

        Returns
        -------
        MirrorRBDesign
        """
        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        ideal_outs = []

        for lnum, l in enumerate(depths):
            if verbosity > 0:
                print('- Sampling {} circuits at MRB length {} ({} of {} depths)'.format(circuits_per_depth, l,
                                                                                         lnum + 1, len(depths)))

            results = _tools.mptools.starmap_with_kwargs(_rc.create_mirror_rb_circuit, circuits_per_depth,
                                                         num_processes, pspec, l, qubit_labels=qubit_labels,
                                                         sampler=sampler, samplerargs=samplerargs,
                                                         localclifford=localclifford, paulirandomize=paulirandomize)

            circuits_at_depth = []
            idealouts_at_depth = []
            for c, iout in results:
                circuits_at_depth.append(c)
                idealouts_at_depth.append((''.join(map(str, iout)),))

            circuit_lists.append(circuits_at_depth)
            ideal_outs.append(idealouts_at_depth)

        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              sampler, samplerargs, localclifford, paulirandomize, descriptor,
                              add_default_protocol)

    def _init_foundation(self, depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                         sampler, samplerargs, localclifford, paulirandomize, descriptor,
                         add_default_protocol):
        super().__init__(depths, circuit_lists, ideal_outs, qubit_labels, remove_duplicates=False)
        self.circuits_per_depth = circuits_per_depth
        self.descriptor = descriptor
        self.sampler = sampler
        self.samplerargs = samplerargs
        self.localclifford = localclifford
        self.paulirandomize = paulirandomize

        if add_default_protocol:
            self.add_default_protocol(RB(name='RB', datatype='adjusted_success_probabilities', defaultfit='A-fixed'))


class RandomizedBenchmarking(_vb.SummaryStatistics):
    """
    The randomized benchmarking protocol.

    This same analysis protocol is used for Clifford, Direct and Mirror RB.
    The standard Mirror RB analysis is obtained by setting
    `datatype` = `adjusted_success_probabilities`.

    Parameters
    ----------
    datatype: 'success_probabilities' or 'adjusted_success_probabilities', optional
        The type of summary data to extract, average, and the fit to an exponential decay. If
        'success_probabilities' then the summary data for a circuit is the frequency that
        the target bitstring is observed, i.e., the success probability of the circuit. If
        'adjusted_success_probabilties' then the summary data for a circuit is
        S = sum_{k = 0}^n (-1/2)^k h_k where h_k is the frequency at which the output bitstring is
        a Hamming distance of k from the target bitstring, and n is the number of qubits.
        This datatype is used in Mirror RB, but can also be used in Clifford and Direct RB.

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

    def __init__(self, datatype='success_probabilities', defaultfit='full', asymptote='std', rtype='EI',
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', square_mean_root=False, name=None):
        """
        Initialize an RB protocol for analyzing RB data.

        Parameters
        ----------
        datatype: 'success_probabilities' or 'adjusted_success_probabilities', optional
            The type of summary data to extract, average, and the fit to an exponential decay. If
            'success_probabilities' then the summary data for a circuit is the frequency that
            the target bitstring is observed, i.e., the success probability of the circuit. If
            'adjusted_success_probabilties' then the summary data for a circuit is
            S = sum_{k = 0}^n (-1/2)^k h_k where h_k is the frequency at which the output bitstring is
            a Hamming distance of k from the target bitstring, and n is the number of qubits.
            This datatype is used in Mirror RB, but can also be used in Clifford and Direct RB.

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
        assert(datatype in ('success_probabilities', 'adjusted_success_probabilities')), \
            "Data type '%s' must be 'success_probabilities' or 'adjusted_success_probabilities'!" % str(datatype)

        self.seed = seed
        self.depths = depths
        self.bootstrap_samples = bootstrap_samples
        self.asymptote = asymptote
        self.rtype = rtype
        self.datatype = datatype
        self.defaultfit = defaultfit
        self.square_mean_root = square_mean_root

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

        if self.datatype not in data.cache:
            summary_data_dict = self._compute_summary_statistics(data)
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
            else:
                raise ValueError("No 'std' asymptote for %s datatype!" % self.asymptote)

        def get_rb_fits(circuitdata_per_depth):
            adj_sps = []
            for depth in depths:
                percircuitdata = circuitdata_per_depth[depth]
                #print(percircuitdata)
                if self.square_mean_root:
                    #print(percircuitdata)
                    adj_sps.append(_np.nanmean(_np.sqrt(percircuitdata))**2)
                    #print(adj_sps)
                else:
                    adj_sps.append(_np.nanmean(percircuitdata))  # average [adjusted] success probabilities

            #print(adj_sps)

            full_fit_results, fixed_asym_fit_results = _rbfit.std_least_squares_fit(
                depths, adj_sps, nqubits, seed=self.seed, asymptote=asymptote,
                ftype='full+FA', rtype=self.rtype)

            return full_fit_results, fixed_asym_fit_results

        #do RB fit on actual data
        ff_results, faf_results = get_rb_fits(data_per_depth)

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
                bs_ff_results, bs_faf_results = get_rb_fits(bootstrap_cache[self.datatype])

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
        The default key within `fits` to plot when calling :method:`plot`.
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
        self.auxfile_types['fits'] = 'pickle'  # b/c NamedDict don't json

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


RB = RandomizedBenchmarking
RBResults = RandomizedBenchmarkingResults  # shorthand
