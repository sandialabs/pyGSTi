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

import time as _time
import os as _os
import numpy as _np
import pickle as _pickle
import collections as _collections
import warnings as _warnings
import copy as _copy
import scipy.optimize as _spo
from scipy.stats import chi2 as _chi2

from . import protocol as _proto
from .modeltest import ModelTest as _ModelTest
from .. import objects as _objs
from .. import algorithms as _alg
from .. import construction as _construction
from .. import io as _io
from .. import tools as _tools

from ..objects import wildcardbudget as _wild
from ..objects.profiler import DummyProfiler as _DummyProfiler
from ..objects import objectivefns as _objfns
from ..algorithms import randomcircuit as _rc
from ..algorithms import rbfit as _rbfit


class ByDepthDesign(_proto.CircuitListsDesign):
    """
    Experiment design that holds circuits organized by depth

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

    def __init__(self, depths, circuit_lists, ideal_outs, qubit_labels=None, remove_duplicates=False):
        assert(len(depths) == len(ideal_outs))
        super().__init__(depths, circuit_lists, qubit_labels, remove_duplicates)
        self.idealout_lists = ideal_outs
        self.auxfile_types['idealout_lists'] = 'json'


class CliffordRBDesign(BenchmarkingDesign):
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
                               randomizeout=False, citerations=20, compilerargs=(),
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
                              randomizeout, citerations, compilerargs, descriptor, add_default_protocol)
        return self

    def __init__(self, pspec, depths, circuits_per_depth, qubit_labels=None, randomizeout=False,
                 citerations=20, compilerargs=(), descriptor='A Clifford RB experiment',
                 add_default_protocol=False, seed=None, verbosity=1):
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
                print('  - Number of circuits sampled = ', end='')
            circuits_at_depth = []
            idealouts_at_depth = []
            for j in range(circuits_per_depth):
                c, iout = _rc.clifford_rb_circuit(pspec, l, qubit_labels=qubit_labels, randomizeout=randomizeout,
                                                  citerations=citerations, compilerargs=compilerargs)
                circuits_at_depth.append(c)
                idealouts_at_depth.append((''.join(map(str, iout)),))
                if verbosity > 0: print(j + 1, end=',')
            circuit_lists.append(circuits_at_depth)
            ideal_outs.append(idealouts_at_depth)
            if verbosity > 0: print('')

        self._init_foundation(depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                              randomizeout, citerations, compilerargs, descriptor, add_default_protocol)

    def _init_foundation(self, depths, circuit_lists, ideal_outs, circuits_per_depth, qubit_labels,
                         randomizeout, citerations, compilerargs, descriptor, add_default_protocol):
        super().__init__(depths, circuit_lists, ideal_outs, qubit_labels, remove_duplicates=False)
        self.circuits_per_depth = circuits_per_depth
        self.randomizeout = randomizeout
        self.citerations = citerations
        self.compilerargs = compilerargs
        self.descriptor = descriptor
        if add_default_protocol:
            if randomizeout:
                defaultfit = 'A-fixed'
            else:
                defaultfit = 'full'
            self.add_default_protocol(RB(name='RB', defaultfit=defaultfit))


class DirectRBDesign(BenchmarkingDesign):
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
                 add_default_protocol=False, seed=None, verbosity=1):
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
                print('  - Number of circuits sampled = ', end='')
            circuits_at_depth = []
            idealouts_at_depth = []
            for j in range(circuits_per_depth):
                c, iout = _rc.direct_rb_circuit(
                    pspec, l, qubit_labels=qubit_labels, sampler=sampler, samplerargs=samplerargs,
                    addlocal=addlocal, lsargs=lsargs, randomizeout=randomizeout,
                    cliffordtwirl=cliffordtwirl, conditionaltwirl=conditionaltwirl,
                    citerations=citerations, compilerargs=compilerargs,
                    partitioned=partitioned)
                circuits_at_depth.append(c)
                idealouts_at_depth.append((''.join(map(str, iout)),))
                if verbosity > 0: print(j + 1, end=',')
            circuit_lists.append(circuits_at_depth)
            ideal_outs.append(idealouts_at_depth)
            if verbosity > 0: print('')

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


class MirrorRBDesign(BenchmarkingDesign):
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
                 add_default_protocol=False):
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

        Returns
        -------
        MirrorRBDesign
        """
        if qubit_labels is None: qubit_labels = tuple(pspec.qubit_labels)
        circuit_lists = []
        ideal_outs = []

        for lnum, l in enumerate(depths):
            circuits_at_depth = []
            idealouts_at_depth = []
            for j in range(circuits_per_depth):
                c, iout = _rc.mirror_rb_circuit(pspec, l, qubit_labels=qubit_labels, sampler=sampler,
                                                samplerargs=samplerargs, localclifford=localclifford,
                                                paulirandomize=paulirandomize)
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


class Benchmark(_proto.Protocol):
    """
    A benchmarking protocol that can construct "summary" quantities from the raw data.

    Parameters
    ----------
    name : str
        The name of this protocol, also used to (by default) name the
        results produced by this protocol.  If None, the class name will
        be used.

    Attributes
    ----------
    summary_datatypes : tuple
        Static list of the categories of summary information this protocol can compute.

    circuit_datatypes : tuple
        Static list of the categories of circuit information this protocol can compute.

    dscmp_datatypes : tuple
        Static list of the categories of dataset-comparison information this protocol can compute.
    """

    summary_datatypes = ('success_counts', 'total_counts', 'hamming_distance_counts',
                         'success_probabilities', 'adjusted_success_probabilities')
    circuit_datatypes = ('twoQgate_count', 'circuit_depth', 'idealout', 'circuit_index', 'circuit_width')
    dscmp_datatypes = ('tvds', 'pvals', 'jsds', 'llrs', 'sstvds')

    def __init__(self, name):
        super().__init__(name)

    #PRIVATE
    def compute_summary_data(self, data):
        """
        Compute (all of) the summary information for `data`.

        Parameters
        ----------
        data : ProtocolData
            The data.

        Returns
        -------
        NamedDict
        """
        def success_counts(dsrow, circ, idealout):
            if dsrow.total == 0: return 0  # shortcut?
            return dsrow[idealout]

        def hamming_distance_counts(dsrow, circ, idealout):
            nqubits = len(circ.line_labels)  # number of qubits
            assert(nqubits == len(idealout[-1]))
            hamming_distance_counts = _np.zeros(nqubits + 1, float)
            if dsrow.total > 0:
                for outcome_lbl, counts in dsrow.counts.items():
                    outbitstring = outcome_lbl[-1]
                    hamming_distance_counts[_tools.rbtools.hamming_distance(outbitstring, idealout[-1])] += counts
            return list(hamming_distance_counts)  # why a list?

        def adjusted_success_probability(hamming_distance_counts):
            """ TODO: docstring """
            hamming_distance_pdf = _np.array(hamming_distance_counts) / _np.sum(hamming_distance_counts)
            #adjSP = _np.sum([(-1 / 2)**n * hamming_distance_counts[n] for n in range(numqubits + 1)]) / total_counts
            adj_sp = _np.sum([(-1 / 2)**n * hamming_distance_pdf[n] for n in range(len(hamming_distance_pdf))])
            return adj_sp

        def get_summary_values(icirc, circ, dsrow, idealout):
            sc = success_counts(dsrow, circ, idealout)
            tc = dsrow.total
            hdc = hamming_distance_counts(dsrow, circ, idealout)

            ret = {'success_counts': sc,
                   'total_counts': tc,
                   'success_probabilities': _np.nan if tc == 0 else sc / tc,
                   'hamming_distance_counts': hdc,
                   'adjusted_success_probabilities': adjusted_success_probability(hdc)}
            return ret

        return self.compute_dict(data, self.summary_datatypes,
                                 get_summary_values, for_passes='all')

    #PRIVATE
    def compute_circuit_data(self, data):
        """
        Compute (all of) the circuit information for `data`.

        Parameters
        ----------
        data : ProtocolData
            The data.

        Returns
        -------
        NamedDict
        """
        def get_circuit_values(icirc, circ, dsrow, idealout):
            ret = {'twoQgate_count': circ.two_q_gate_count(),
                   'circuit_depth': circ.depth(),
                   'idealout': idealout,
                   'circuit_index': icirc,
                   'circuit_width': len(circ.line_labels)}
            ret.update(dsrow.aux)  # note: will only get aux data from *first* pass in multi-pass data
            return ret

        return self.compute_dict(data, self.circuit_datatypes, get_circuit_values, for_passes="first")

    #PRIVATE
    def compute_dscmp_data(self, data, dscomparator):
        """
        Compute (all of) the dataset comparison information for `data`.

        Parameters
        ----------
        data : ProtocolData
            The data.

        dscomparator : DataComparator
            A comparator object used to compare data sets.

        Returns
        -------
        NamedDict
        """
        def get_dscmp_values(icirc, circ, dsrow, idealout):
            ret = {'tvds': dscomparator.tvds.get(circ, _np.nan),
                   'pvals': dscomparator.pVals.get(circ, _np.nan),
                   'jsds': dscomparator.jsds.get(circ, _np.nan),
                   'llrs': dscomparator.llrs.get(circ, _np.nan)}
            return ret

        return self.compute_dict(data, "dscmpdata", self.dsmp_datatypes, get_dscmp_values, for_passes="none")

    #PRIVATE
    def compute_predicted_probs(self, data, model):
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
        def get_success_prob(icirc, circ, dsrow, idealout):
            #if set(circ.line_labels) != set(qubits):
            #    trimmedcirc = circ.copy(editable=True)
            #    for q in circ.line_labels:
            #        if q not in qubits:
            #            trimmedcirc.delete_lines(q)
            #else:
            #    trimmedcirc = circ
            return {'success_probabilities': model.probs(circ)[('success',)]}

        return self.compute_dict(data, ('success_probabilities',),
                                 get_success_prob, for_passes="none")

    #PRIVATE
    def compute_dict(self, data, component_names, compute_fn, for_passes="all"):
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
            `compute_fn(icirc : int, circ : Circuit, dsrow : DataSetRow, idealout : OutcomeLabel)`
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
        qty_data = _tools.NamedDict('Datatype', 'category', None,
                                    {comp: _tools.NamedDict('Depth', 'int', 'float', {depth: [] for depth in depths})
                                     for comp in component_names})

        #loop over all circuits
        for depth, circuits_at_depth, idealouts_at_depth in zip(depths, design.circuit_lists, design.idealout_lists):
            for icirc, (circ, idealout) in enumerate(zip(circuits_at_depth, idealouts_at_depth)):
                dsrow = ds[circ] if (ds is not None) else None  # stripOccurrenceTags=True ??
                # -- this is where Tim thinks there's a bottleneck, as these loops will be called for each
                # member of a simultaneous experiment separately instead of having an inner-more iteration
                # that loops over the "structure", i.e. the simultaneous qubit sectors.
                #TODO: <print percentage>

                for component_name, val in compute_fn(icirc, circ, dsrow, idealout).items():
                    qty_data[component_name][depth].append(val)  # maybe use a pandas dataframe here?

        return qty_data

    #PRIVATE - or just a tool function - doesn't use self
    def create_depthwidth_dict(self, depths, widths, fillfn, seriestype):
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
            'Depth', 'int', seriestype, {depth: _tools.NamedDict(
                'Width', 'int', seriestype, {width: fillfn() for width in widths}) for depth in depths})

    #PRIVATE
    def add_bootstrap_qtys(self, data_cache, num_qtys, finitecounts=True):
        """
        Adds bootstrapped "summary datasets".

        The bootstrap is over both the finite counts of each circuit and
        over the circuits at each length.

        Note: only adds quantities if they're needed.

        Parameters
        ----------
        data_cache : dict
            A cache of already-existing bootstraps.

        num_qtys : int, optional
            The number of bootstrapped datasets to construct.

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
        success_probabilities = data_cache['success_probabilities']
        total_counts = data_cache['total_counts']
        hamming_distance_counts = data_cache['hamming_distance_counts']
        depths = list(success_probabilities.keys())

        for i in range(num_existing, num_qtys):

            component_names = self.summary_datatypes
            bcache = _tools.NamedDict(
                'Datatype', 'category', None,
                {comp: _tools.NamedDict('Depth', 'int', 'float', {depth: [] for depth in depths})
                 for comp in component_names})  # ~= "RB summary dataset"

            for depth, sprobs in success_probabilities.items():
                numcircuits = len(sprobs)
                for k in range(numcircuits):
                    ind = _np.random.randint(numcircuits)
                    sampled_sprob = sprobs[ind]
                    totalcounts = total_counts[depth][ind] if finitecounts else None
                    bcache['success_probabilities'][depth].append(sampled_sprob)
                    if finitecounts:
                        bcache['success_counts'][depth].append(_np.random.binomial(totalcounts, sampled_sprob))
                        bcache['total_counts'][depth].append(totalcounts)
                    else:
                        bcache['success_counts'][depth].append(sampled_sprob)

                    #ind = _np.random.randint(numcircuits)  # note: old code picked different random ints
                    #totalcounts = total_counts[depth][ind] if finitecounts else None  # need this if a new randint
                    sampled_hamdist_counts = hamming_distance_counts[depth][ind]
                    sampled_hamdist_pdf = _np.array(sampled_hamdist_counts) / _np.sum(sampled_hamdist_counts)

                    if finitecounts:
                        bcache['hamming_distance_counts'][depth].append(
                            list(_np.random.multinomial(totalcounts, sampled_hamdist_pdf)))
                    else:
                        bcache['hamming_distance_counts'][depth].append(sampled_hamdist_pdf)

                    # replicates adjusted_success_probability function above
                    adj_sp = _np.sum([(-1 / 2)**n * sampled_hamdist_pdf[n] for n in range(len(sampled_hamdist_pdf))])
                    bcache['adjusted_success_probabilities'][depth].append(adj_sp)

            data_cache[key].append(bcache)


#Add something like this?
#class PassStabilityTest(_proto.Protocol):
#    pass


#TODO: docstrings -- do we need this class?
class VolumetricBenchmarkGrid(Benchmark):
    """
    A protocol that creates an entire depth vs. width grid of volumetric benchmark values

    Parameters
    ----------
    depths : <TODO typ>, optional
        <TODO description>

    widths : <TODO typ>, optional
        <TODO description>

    datatype : <TODO typ>, optional
        <TODO description>

    paths : <TODO typ>, optional
        <TODO description>

    statistic : <TODO typ>, optional
        <TODO description>

    aggregate : <TODO typ>, optional
        <TODO description>

    rescaler : <TODO typ>, optional
        <TODO description>

    dscomparator : <TODO typ>, optional
        <TODO description>

    name : <TODO typ>, optional
        <TODO description>
    """

    def __init__(self, depths='all', widths='all', datatype='success_probabilities',
                 paths='all', statistic='mean', aggregate=True, rescaler='auto',
                 dscomparator=None, name=None):

        super().__init__(name)
        self.postproc = VolumetricBenchmarkGridPP(depths, widths, datatype, paths, statistic, aggregate, self.name)
        self.dscomparator = dscomparator
        self.rescaler = rescaler

        self.auxfile_types['postproc'] = 'protocolobj'
        self.auxfile_types['dscomparator'] = 'pickle'
        self.auxfile_types['rescaler'] = 'reset'  # punt for now - fix later

    def run(self, data, memlimit=None, comm=None):
        #Since we know that VolumetricBenchmark protocol objects Create a single results just fill
        # in data under the result object's 'volumetric_benchmarks' and 'failure_counts'
        # keys, and these are indexed by width and depth (even though each VolumetricBenchmark
        # only contains data for a single width), we can just "merge" the VB results of all
        # the underlying by-depth datas, so long as they're all for different widths.

        #Then run resulting data normally, giving a results object
        # with "top level" dicts correpsonding to different paths
        """
        <TODO summary>

        Parameters
        ----------
        data : <TODO typ>
            <TODO description>

        memlimit : <TODO typ>, optional
            <TODO description>

        comm : <TODO typ>, optional
            <TODO description>
        """
        vb = VolumetricBenchmark(self.postproc.depths, self.postproc.datatype, self.postproc.statistic,
                                 self.rescaler, self.dscomparator, name=self.name)
        separate_results = _proto.SimpleRunner(vb).run(data, memlimit, comm)
        pp_results = self.postproc.run(separate_results, memlimit, comm)
        pp_results.protocol = self
        return pp_results


#TODO: docstrings -- do we need this class?
class VolumetricBenchmarkGridPP(_proto.ProtocolPostProcessor):
    """
    A postprocesor that constructs a grid of volumetric benchmarks from existing results.
    """

    def __init__(self, depths='all', widths='all', datatype='success_probabilities',
                 paths='all', statistic='mean', aggregate=True, name=None):

        super().__init__(name)
        self.depths = depths
        self.widths = widths
        self.datatype = datatype
        self.paths = paths if paths == 'all' else sorted(paths)  # need to ensure paths are grouped by common prefix
        self.statistic = statistic
        self.aggregate = aggregate

    def run(self, results, memlimit=None, comm=None):
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

        Returns
        -------
        VolumetricBenchmarkingResults
        """
        data = results.data
        paths = results.get_tree_paths() if self.paths == 'all' else self.paths
        #Note: above won't work if given just a results object - needs a dir

        #Process results
        #Merge/flatten the data from different paths into one depth vs width grid
        passnames = list(data.passes.keys()) if data.is_multipass() else [None]
        passresults = []
        for passname in passnames:
            vb = _tools.NamedDict('Depth', 'int', None)
            fails = _tools.NamedDict('Depth', 'int', None)
            path_for_gridloc = {}
            for path in paths:
                #TODO: need to be able to filter based on widths... - maybe replace .update calls
                # with something more complicated when width != 'all'
                #print("Aggregating path = ", path)  #TODO - show progress something like this later?

                #Traverse path to get to root of VB data
                root = results
                for key in path:
                    root = root[key]
                root = root.for_protocol.get(self.name, None)
                if root is None: continue

                if passname:  # then we expect final Results are MultiPassResults
                    root = root.passes[passname]  # now root should be a BenchmarkingResults
                assert(isinstance(root, VolumetricBenchmarkingResults))
                assert(isinstance(root.data.edesign, ByDepthDesign)), \
                    "All paths must lead to by-depth exp. design, not %s!" % str(type(root.data.edesign))

                #Get the list of depths we'll extract from this (`root`) sub-results
                depths = root.data.edesign.depths if (self.depths == 'all') else \
                    filter(lambda d: d in self.depths, root.data.edesign.depths)
                width = len(root.data.edesign.qubit_labels)  # sub-results contains only a single width
                if self.widths != 'all' and width not in self.widths: continue  # skip this one

                for depth in depths:
                    if depth not in vb:  # and depth not in fails
                        vb[depth] = _tools.NamedDict('Width', 'int', 'float')
                        fails[depth] = _tools.NamedDict('Width', 'int', None)
                        path_for_gridloc[depth] = {}  # just used for meaningful error message

                    if width in path_for_gridloc[depth]:
                        raise ValueError(("Paths %s and %s both give data for depth=%d, width=%d!  Set the `paths`"
                                          " argument of this VolumetricBenchmarkGrid to avoid this.") %
                                         (str(path_for_gridloc[depth][width]), str(path), depth, width))

                    vb[depth][width] = root.volumetric_benchmarks[depth][width]
                    fails[depth][width] = root.failure_counts[depth][width]
                    path_for_gridloc[depth][width] = path

            if self.statistic in ('minmin', 'maxmax') and not self.aggregate:
                self._update_vb_minmin_maxmax(vb)   # aggregate now since we won't aggregate over passes

            #Create Results
            results = VolumetricBenchmarkingResults(data, self)
            results.volumetric_benchmarks = vb
            results.failure_counts = fails
            passresults.append(results)

        if self.aggregate and len(passnames) > 1:  # aggregate pass data into a single set of qty dicts
            agg_vb = _tools.NamedDict('Depth', 'int', None)
            agg_fails = _tools.NamedDict('Depth', 'int', None)
            template = passresults[0].volumetric_benchmarks  # to get widths and depths

            #Get function to aggregate the different per-circuit datatype values
            if self.statistic == 'max' or self.statistic == 'maxmax':
                np_fn = _np.nanmax
            elif self.statistic == 'mean':
                np_fn = _np.nanmean
            elif self.statistic == 'min' or self.statistic == 'minmin':
                np_fn = _np.nanmin
            elif self.statistic == 'dist':
                def np_fn(v): return v  # identity
            else: raise ValueError("Invalid statistic '%s'!" % self.statistic)

            def agg_fn(percircuitdata, width, rescale=True):
                """ Aggregates datatype-data for all circuits at same depth """
                rescaled = self.rescale_function(percircuitdata, width) if rescale else percircuitdata
                if _np.isnan(rescaled).all():
                    return _np.nan
                else:
                    return np_fn(rescaled)

            for depth, template_by_width_data in template.items():
                agg_vb[depth] = _tools.NamedDict('Width', 'int', 'float')
                agg_fails[depth] = _tools.NamedDict('Width', 'int', None)

                for width in template_by_width_data.keys():
                    # ppd = "per pass data"
                    vb_ppd = [r.volumetric_benchmarks[depth][width] for r in passresults]
                    fail_ppd = [r.failure_counts[depth][width] for r in passresults]

                    successcount = 0
                    failcount = 0
                    for (successcountpass, failcountpass) in fail_ppd:
                        successcount += successcountpass
                        failcount += failcountpass
                    agg_fails[depth][width] = (successcount, failcount)

                    if self.statistic == 'dist':
                        agg_vb[depth][width] = [item for sublist in vb_ppd for item in sublist]
                    else:
                        agg_vb[depth][width] = agg_fn(vb_ppd, width, rescale=False)

            aggregated_results = VolumetricBenchmarkingResults(data, self)
            aggregated_results.volumetric_benchmarks = agg_vb
            aggregated_results.failure_counts = agg_fails

            if self.statistic in ('minmin', 'maxmax'):
                self._update_vb_minmin_maxmax(aggregated_results.qtys['volumetric_benchmarks'])
            return aggregated_results  # replace per-pass results with aggregated results
        elif len(passnames) > 1:
            multipass_results = _proto.MultiPassResults(data, self)
            multipass_results.passes.update({passname: r for passname, r in zip(passnames, passresults)})
            return multipass_results
        else:
            return passresults[0]

    def _update_vb_minmin_maxmax(self, vb):
        for d in vb.keys():
            for w in vb[d].keys():
                for d2 in vb.keys():
                    for w2 in vb[d2].keys():
                        if self.statistic == 'minmin' and d2 <= d and w2 <= w and vb[d2][w2] < vb[d][w]:
                            vb[d][w] = vb[d2][w2]
                        if self.statistic == 'maxmax' and d2 >= d and w2 >= w and vb[d2][w2] > vb[d][w]:
                            vb[d][w] = vb[d2][w2]


class VolumetricBenchmark(Benchmark):
    """
    A volumetric benchmark protocol
    """

    def __init__(self, depths='all', datatype='success_probabilities',
                 statistic='mean', rescaler='auto', dscomparator=None,
                 custom_data_src=None, name=None):

        assert(statistic in ('max', 'mean', 'min', 'dist', 'maxmax', 'minmin', 'sum'))

        super().__init__(name)
        self.depths = depths
        #self.widths = widths  # widths='all',
        self.datatype = datatype
        self.statistic = statistic
        self.dscomparator = dscomparator
        self.rescaler = rescaler
        self.custom_data_src = custom_data_src

        self.auxfile_types['dscomparator'] = 'pickle'
        # because this *could* be a model or a qty dict (or just a string?)
        self.auxfile_types['custom_data_src'] = 'pickle'
        self.auxfile_types['rescale_function'] = 'none'  # don't serialize this, so need _set_rescale_function
        self._set_rescale_function()
        self._nameddict_attributes += (('datatype', 'Data Type', 'category'),
                                       ('statistic', 'Statistic', 'category'))

    def _init_unserialized_attributes(self):
        self._set_rescale_function()

    def _set_rescale_function(self):
        rescaler = self.rescaler
        if isinstance(rescaler, str):
            if rescaler == 'auto':
                if self.datatype == 'success_probabilities':
                    def rescale_function(data, width):
                        return list((_np.array(data) - 1 / 2**width) / (1 - 1 / 2**width))
                else:
                    def rescale_function(data, width):
                        return data
            elif rescaler == 'none':
                def rescale_function(data, width):
                    return data
            else:
                raise ValueError("Unknown rescaling option!")
        else:
            rescale_function = rescaler
        self.rescale_function = rescale_function

    def run(self, data, memlimit=None, comm=None):
        design = data.edesign

        if self.custom_data_src is None:  # then use the data in `data`
            #Note: can only take/put things ("results") in data.cache that *only* depend on the exp. design
            # and dataset (i.e. the DataProtocol object).  Protocols must be careful of this in their implementation!
            if self.datatype in self.summary_datatypes:
                if self.datatype not in data.cache:
                    summary_data_dict = self.compute_summary_data(data)
                    data.cache.update(summary_data_dict)
            elif self.datatype in self.dscmp_datatypes:
                if self.datatype not in data.cache:
                    dscmp_data = self.compute_dscmp_data(data, self.dscomparator)
                    data.cache.update(dscmp_data)
            elif self.datatype in self.circuit_datatypes:
                if self.datatype not in data.cache:
                    circuit_data = self.compute_circuit_data(data)
                    data.cache.update(circuit_data)
            else:
                raise ValueError("Invalid datatype: %s" % self.datatype)

            src_data = data.cache[self.datatype]

        elif isinstance(self.custom_data_src, _objs.SuccessFailModel):  # then simulate all the circuits in `data`
            assert(self.datatype == 'success_probabilities'), "Only success probabailities can be simulated."
            sfmodel = self.custom_data_src
            depths = data.edesign.depths if self.depths == 'all' else self.depths
            src_data = _tools.NamedDict('Depth', 'int', 'float', {depth: [] for depth in depths})
            circuit_lists_for_depths = {depth: lst for depth, lst in zip(design.depths, design.circuit_lists)}

            for depth in depths:
                for circ in circuit_lists_for_depths[depth]:
                    predicted_success_prob = sfmodel.probs(circ)[('success',)]
                    src_data[depth].append(predicted_success_prob)

        elif isinstance(self.custom_data_src, dict):  # Assume this is a "summary dataset"
            summary_data = self.custom_data_src
            src_data = summary_data['success_probabilities']

        else:
            raise ValueError("Invalid 'custom_data_src' of type: %s" % str(type(self.custom_data_src)))

        #Get function to aggregate the different per-circuit datatype values
        if self.statistic == 'max' or self.statistic == 'maxmax':
            np_fn = _np.nanmax
        elif self.statistic == 'mean':
            np_fn = _np.nanmean
        elif self.statistic == 'min' or self.statistic == 'minmin':
            np_fn = _np.nanmin
        elif self.statistic == 'dist':
            def np_fn(v): return v  # identity
        elif self.statistic == 'sum':
            np_fn = _np.sum
        else: raise ValueError("Invalid statistic '%s'!" % self.statistic)

        def agg_fn(percircuitdata, width, rescale=True):
            """ Aggregates datatype-data for all circuits at same depth """
            rescaled = self.rescale_function(percircuitdata, width) if rescale else percircuitdata
            if _np.isnan(rescaled).all():
                return _np.nan
            else:
                return np_fn(rescaled)

        def failcnt_fn(percircuitdata):
            """ Returns (nSucceeded, nFailed) for all circuits at same depth """
            num_circuits = len(percircuitdata)
            failcount = int(_np.sum(_np.isnan(percircuitdata)))
            return (num_circuits - failcount, failcount)

        data_per_depth = src_data
        if self.depths == 'all':
            depths = data_per_depth.keys()
        else:
            depths = filter(lambda d: d in data_per_depth, self.depths)
        width = len(design.qubit_labels)

        vb = self.create_depthwidth_dict(depths, (width,), lambda: None, 'float')
        successes = self.create_depthwidth_dict(depths, (width,), lambda: None, 'int')
        fails = self.create_depthwidth_dict(depths, (width,), lambda: None, 'int')

        for depth in depths:
            percircuitdata = data_per_depth[depth]
            successes[depth][width], fails[depth][width] = failcnt_fn(percircuitdata)
            vb[depth][width] = agg_fn(percircuitdata, width)

        results = VolumetricBenchmarkingResults(data, self)  # 'Qty', 'category'
        results.volumetric_benchmarks = vb
        results.success_counts = successes
        results.failure_counts = fails
        return results


class PredictedVolumetricBenchmark(VolumetricBenchmark):
    """
    Runs a volumetric benchmark on success/fail data predicted from a model
    """

    def __init__(self, model_or_summary_data, depths='all', statistic='mean',
                 rescaler='auto', dscomparator=None, name=None):
        super().__init__(depths, 'success_probabilities', statistic, rescaler,
                         dscomparator, model_or_summary_data, name)


class RandomizedBenchmarking(Benchmark):
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
                 seed=(0.8, 0.95), bootstrap_samples=200, depths='all', name=None):
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

        assert(datatype in self.summary_datatypes), "Unknown data type: %s!" % str(datatype)
        assert(datatype in ('success_probabilities', 'adjusted_success_probabilities')), \
            "Data type '%s' must be 'success_probabilities' or 'adjusted_success_probabilities'!" % str(datatype)

        self.seed = seed
        self.depths = depths
        self.bootstrap_samples = bootstrap_samples
        self.asymptote = asymptote
        self.rtype = rtype
        self.datatype = datatype
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

        if self.datatype not in data.cache:
            summary_data_dict = self.compute_summary_data(data)
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
                adj_sps.append(_np.mean(percircuitdata))  # average [adjusted] success probabilities

            full_fit_results, fixed_asym_fit_results = _rbfit.std_least_squares_data_fitting(
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
                self.add_bootstrap_qtys(data.cache, self.bootstrap_samples, finitecounts=True)
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
    The results of running a randomized benchmarking
    """

    def __init__(self, data, protocol_instance, fits, depths, defaultfit):
        """
        Initialize an empty Results object.
        TODO: docstring
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

        ylim : <TODO typ>, optional
            <TODO description>

        xlim : <TODO typ>, optional
            <TODO description>

        legend : bool, optional
            Whether to show a legend.

        title : str, optional
            A title to put on the figure.

        figpath : str, optional
            If specified, the figure is saved with this filename.

        Returns
        -------
        <TODO typ>
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


class VolumetricBenchmarkingResults(_proto.ProtocolResults):
    """
    The results from running a volumetric benchmark protocol
    """

    def __init__(self, data, protocol_instance):
        """
        Initialize an empty Results object.
        TODO: docstring
        """
        super().__init__(data, protocol_instance)

        self.volumetric_benchmarks = {}
        self.success_counts = {}
        self.failure_counts = {}

        self.auxfile_types['volumetric_benchmarks'] = 'pickle'  # b/c NamedDicts don't json
        self.auxfile_types['success_counts'] = 'pickle'  # b/c NamedDict don't json
        self.auxfile_types['failure_counts'] = 'pickle'  # b/c NamedDict don't json


RB = RandomizedBenchmarking
VB = VolumetricBenchmark
VBGrid = VolumetricBenchmarkGrid
RBResults = RandomizedBenchmarkingResults  # shorthand
VBResults = VolumetricBenchmarkingResults  # shorthand
