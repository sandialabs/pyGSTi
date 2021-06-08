"""
Defines the CloudNoiseModel class and supporting functions
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
import copy as _copy
import itertools as _itertools
import collections as _collections
import scipy.sparse as _sps
import warnings as _warnings

from . import statespace as _statespace
from ..modelmembers import operations as _op
from ..modelmembers import states as _state
from ..modelmembers import povms as _povm
from ..modelmembers.operations import opfactory as _opfactory
from ..evotypes import Evotype as _Evotype
from ..objects import qubitgraph as _qgraph
from . import labeldicts as _ld
from ..tools import optools as _gt
from ..tools import basistools as _bt
from ..tools import internalgates as _itgs
from .implicitmodel import ImplicitOpModel as _ImplicitOpModel
from .layerrules import LayerRules as _LayerRules
from ..forwardsims.forwardsim import ForwardSimulator as _FSim
from ..forwardsims.matrixforwardsim import MatrixForwardSimulator as _MatrixFSim
from ..forwardsims.mapforwardsim import MapForwardSimulator as _MapFSim
from ..forwardsims.termforwardsim import TermForwardSimulator as _TermFSim

from ..objects.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from ..objects.basis import BuiltinBasis as _BuiltinBasis, ExplicitBasis as _ExplicitBasis
from ..objects.label import Label as _Lbl, CircuitLabel as _CircuitLabel

from ..tools.basisconstructors import sqrt2, id2x2, sigmax, sigmay, sigmaz


def _iter_basis_inds(weight):
    """ Iterate over product of `weight` non-identity Pauli 1Q basis indices """
    basisIndList = [[1, 2, 3]] * weight  # assume pauli 1Q basis, and only iterate over non-identity els
    for basisInds in _itertools.product(*basisIndList):
        yield basisInds


def basis_product_matrix(sigma_inds, sparse):
    """
    Construct the Pauli product matrix from the given `sigma_inds`

    Parameters
    ----------
    sigma_inds : iterable
        A sequence of integers in the range [0,3] corresponding to the
        I, X, Y, Z Pauli basis matrices.

    sparse : bool
        Whether to return a sparse matrix or not.

    Returns
    -------
    numpy.ndarray or scipy.sparse.csr_matrix
    """
    sigmaVec = (id2x2 / sqrt2, sigmax / sqrt2, sigmay / sqrt2, sigmaz / sqrt2)
    M = _np.identity(1, 'complex')
    for i in sigma_inds:
        M = _np.kron(M, sigmaVec[i])
    return _sps.csr_matrix(M) if sparse else M


class CloudNoiseModel(_ImplicitOpModel):
    """
    A n-qubit model using a low-weight and geometrically local error model with a common "global idle" operation.

    Parameters
    ----------
    num_qubits : int
        The number of qubits

    gatedict : dict
        A dictionary (an `OrderedDict` if you care about insertion order) that
        associates with string-type gate names (e.g. `"Gx"`) :class:`LinearOperator`,
        `numpy.ndarray`, or :class:`OpFactory` objects. When the objects may act on
        fewer than the total number of qubits (determined by their dimension/shape) then
        they are repeatedly embedded into `num_qubits`-qubit gates as specified by their
        `availability`.  These operations represent the ideal target operations, and
        thus, any `LinearOperator` or `OpFactory` objects must be *static*, i.e., have
        zero parameters.

    availability : dict, optional
        A dictionary whose keys are the same gate names as in
        `gatedict` and whose values are lists of qubit-label-tuples.  Each
        qubit-label-tuple must have length equal to the number of qubits
        the corresponding gate acts upon, and causes that gate to be
        embedded to act on the specified qubits.  For example,
        `{ 'Gx': [(0,),(1,),(2,)], 'Gcnot': [(0,1),(1,2)] }` would cause
        the `1-qubit `'Gx'`-gate to be embedded three times, acting on qubits
        0, 1, and 2, and the 2-qubit `'Gcnot'`-gate to be embedded twice,
        acting on qubits 0 & 1 and 1 & 2.  Instead of a list of tuples,
        values of `availability` may take the special values:

        - `"all-permutations"` and `"all-combinations"` equate to all possible
        permutations and combinations of the appropriate number of qubit labels
        (deterined by the gate's dimension).
        - `"all-edges"` equates to all the vertices, for 1Q gates, and all the
        edges, for 2Q gates of the geometry.
        - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
        on any target qubits via an :class:`EmbeddingOpFactory` (uses less
        memory but slower than `"all-permutations"`.

        If a gate name (a key of `gatedict`) is not present in `availability`,
        the default is `"all-edges"`.

    qubit_labels : tuple, optional
        The circuit-line labels for each of the qubits, which can be integers
        and/or strings.  Must be of length `num_qubits`.  If None, then the
        integers from 0 to `num_qubits-1` are used.

    geometry : {"line","ring","grid","torus"} or QubitGraph
        The type of connectivity among the qubits, specifying a
        graph used to define neighbor relationships.  Alternatively,
        a :class:`QubitGraph` object with node labels equal to
        `qubit_labels` may be passed directly.

    global_idle_layer : LinearOperator
        A global idle operation which acts on all the qubits and
        is, if `add_idle_noise_to_all_gates=True`, composed with the
        actions of specific gates to form the layer operation of
        any circuit layer.

    prep_layers, povm_layers : None or operator or dict or list, optional
        The SPAM operations as n-qubit layer operations.  If `None`, then
        no preps (or POVMs) are created.  If a dict, then the keys are
        labels and the values are layer operators.  If a list, then the
        elements are layer operators and the labels will be assigned as
        "rhoX" and "MX" where X is an integer starting at 0.  If a single
        layer operation is given, then this is used as the sole prep or
        POVM and is assigned the label "rho0" or "Mdefault" respectively.

    build_cloudnoise_fn : function, optional
        A function which takes a single :class:`Label` as an argument and
        returns the cloud-noise operation for that primitive layer
        operation.  Note that if `errcomp_type="gates"` the returned
        operator should be a superoperator whereas if
        `errcomp_type="errorgens"` then the returned operator should be
        an error generator (not yet exponentiated).

    build_cloudkey_fn : function, optional
        An function which takes a single :class:`Label` as an argument and
        returns a "cloud key" for that primitive layer.  The "cloud" is the
        set of qubits that the error (the operator returned from
        `build_cloudnoise_fn`) touches -- and the "key" returned from this
        function is meant to identify that cloud.  This is used to keep track
        of which primitive layer-labels correspond to the same cloud - e.g.
        the cloud-key for ("Gx",2) and ("Gy",2) might be the same and could
        be processed together when selecing sequences that amplify the parameters
        in the cloud-noise operations for these two labels.  The return value
        should be something hashable with the property that two noise
        which act on the same qubits should have the same cloud key.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The circuit simulator used to compute any
        requested probabilities, e.g. from :method:`probs` or
        :method:`bulk_probs`.  The default value of `"auto"` automatically
        selects the simulation type, and is usually what you want. Other
        special allowed values are:

        - "matrix" : op_matrix-op_matrix products are computed and
          cached to get composite gates which can then quickly simulate
          a circuit for any preparation and outcome.  High memory demand;
          best for a small number of (1 or 2) qubits.
        - "map" : op_matrix-state_vector products are repeatedly computed
          to simulate circuits.  Slower for a small number of qubits, but
          faster and more memory efficient for higher numbers of qubits (3+).

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    errcomp_type : {"gates","errorgens"}
        How errors are composed when creating layer operations in the created
        model.  `"gates"` means that the errors on multiple gates in a single
        layer are composed as separate and subsequent processes.  Specifically,
        the layer operation has the form `Composed(target,idleErr,cloudErr)`
        where `target` is a composition of all the ideal gate operations in the
        layer, `idleErr` is idle error (`.operation_blks['layers']['globalIdle']`),
        and `cloudErr` is the composition (ordered as layer-label) of cloud-
        noise contributions, i.e. a map that acts as the product of exponentiated
        error-generator matrices.  `"errorgens"` means that layer operations
        have the form `Composed(target, error)` where `target` is as above and
        `error` results from composing the idle and cloud-noise error
        *generators*, i.e. a map that acts as the exponentiated sum of error
        generators (ordering is irrelevant in this case).

    add_idle_noise_to_all_gates: bool, optional
        Whether the global idle should be added as a factor following the
        ideal action of each of the non-idle gates.

    sparse_lindblad_basis : bool, optional
        Whether embedded Lindblad-parameterized gates within the constructed
        `num_qubits`-qubit gates are sparse or not.

    verbosity : int, optional
        An integer >= 0 dictating how must output to send to stdout.
    """

    @classmethod
    def from_hops_and_weights(cls, num_qubits, gate_names, nonstd_gate_unitaries=None,
                              custom_gates=None, availability=None,
                              qubit_labels=None, geometry="line",
                              max_idle_weight=1, max_spam_weight=1, maxhops=0,
                              extra_weight_1_hops=0, extra_gate_weight=0,
                              simulator="auto", parameterization="H+S",
                              evotype='default', spamtype="lindblad", add_idle_noise_to_all_gates=True,
                              errcomp_type="gates", independent_clouds=True,
                              sparse_lindblad_basis=False, sparse_lindblad_reps=False,
                              verbosity=0):
        """
        Create a :class:`CloudNoiseModel` from hopping rules.

        Parameters
        ----------
        num_qubits : int
            The number of qubits

        gate_names : list
            A list of string-type gate names (e.g. `"Gx"`) either taken from
            the list of builtin "standard" gate names given above or from the
            keys of `nonstd_gate_unitaries`.  These are the typically 1- and 2-qubit
            gates that are repeatedly embedded (based on `availability`) to form
            the resulting model.

        nonstd_gate_unitaries : dict, optional
            A dictionary of numpy arrays which specifies the unitary gate action
            of the gate names given by the dictionary's keys.  As an advanced
            behavior, a unitary-matrix-returning function which takes a single
            argument - a tuple of label arguments - may be given instead of a
            single matrix to create an operation *factory* which allows
            continuously-parameterized gates.  This function must also return
            an empty/dummy unitary when `None` is given as it's argument.

        custom_gates : dict
            A dictionary that associates with gate labels
            :class:`LinearOperator`, :class:`OpFactory`, or `numpy.ndarray`
            objects.  These objects describe the full action of the gate or
            primitive-layer they're labeled by (so if the model represents
            states by density matrices these objects are superoperators, not
            unitaries), and override any standard construction based on builtin
            gate names or `nonstd_gate_unitaries`.  Keys of this dictionary must
            be string-type gate *names* -- they cannot include state space labels
            -- and they must be *static* (have zero parameters) because they
            represent only the ideal behavior of each gate -- the cloudnoise
            operations represent the parameterized noise.  To fine-tune how this
            noise is parameterized, call the :class:`CloudNoiseModel` constructor
            directly.

        availability : dict, optional
            A dictionary whose keys are the same gate names as in
            `gatedict` and whose values are lists of qubit-label-tuples.  Each
            qubit-label-tuple must have length equal to the number of qubits
            the corresponding gate acts upon, and causes that gate to be
            embedded to act on the specified qubits.  For example,
            `{ 'Gx': [(0,),(1,),(2,)], 'Gcnot': [(0,1),(1,2)] }` would cause
            the `1-qubit `'Gx'`-gate to be embedded three times, acting on qubits
            0, 1, and 2, and the 2-qubit `'Gcnot'`-gate to be embedded twice,
            acting on qubits 0 & 1 and 1 & 2.  Instead of a list of tuples,
            values of `availability` may take the special values:

            - `"all-permutations"` and `"all-combinations"` equate to all possible
            permutations and combinations of the appropriate number of qubit labels
            (deterined by the gate's dimension).
            - `"all-edges"` equates to all the vertices, for 1Q gates, and all the
            edges, for 2Q gates of the geometry.
            - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
            on any target qubits via an :class:`EmbeddingOpFactory` (uses less
            memory but slower than `"all-permutations"`.

            If a gate name (a key of `gatedict`) is not present in `availability`,
            the default is `"all-edges"`.

        qubit_labels : tuple, optional
            The circuit-line labels for each of the qubits, which can be integers
            and/or strings.  Must be of length `num_qubits`.  If None, then the
            integers from 0 to `num_qubits-1` are used.

        geometry : {"line","ring","grid","torus"} or QubitGraph
            The type of connectivity among the qubits, specifying a
            graph used to define neighbor relationships.  Alternatively,
            a :class:`QubitGraph` object with node labels equal to
            `qubit_labels` may be passed directly.

        max_idle_weight : int, optional
            The maximum-weight for errors on the global idle gate.

        max_spam_weight : int, optional
            The maximum-weight for SPAM errors when `spamtype == "linblad"`.

        maxhops : int
            The locality constraint: for a gate, errors (of weight up to the
            maximum weight for the gate) are allowed to occur on the gate's
            target qubits and those reachable by hopping at most `maxhops` times
            from a target qubit along nearest-neighbor links (defined by the
            `geometry`).

        extra_weight_1_hops : int, optional
            Additional hops (adds to `maxhops`) for weight-1 errors.  A value > 0
            can be useful for allowing just weight-1 errors (of which there are
            relatively few) to be dispersed farther from a gate's target qubits.
            For example, a crosstalk-detecting model might use this.

        extra_gate_weight : int, optional
            Addtional weight, beyond the number of target qubits (taken as a "base
            weight" - i.e. weight 2 for a 2Q gate), allowed for gate errors.  If
            this equals 1, for instance, then 1-qubit gates can have up to weight-2
            errors and 2-qubit gates can have up to weight-3 errors.

        sparse_lindblad_basis : bool, optional
            Whether the embedded Lindblad-parameterized gates within the constructed
            `num_qubits`-qubit gates are sparse or not.  (This is determied by whether
            they are constructed using sparse basis matrices.)  When sparse, these
            Lindblad gates take up less memory, but their action is slightly slower.
            Usually it's fine to leave this as the default (False), except when
            considering particularly high-weight terms (b/c then the Lindblad gates
            are higher dimensional and sparsity has a significant impact).

        simulator : ForwardSimulator or {"auto", "matrix", "map"}
            The circuit simulator used to compute any
            requested probabilities, e.g. from :method:`probs` or
            :method:`bulk_probs`.  Using `"auto"` selects `"matrix"` when there
            are 2 qubits or less, and otherwise selects `"map"`.

        parameterization : str, optional
            Can be any Lindblad parameterization base type (e.g. CPTP,
            H+S+A, H+S, S, D, etc.) This is the type of parameterizaton to use in
            the constructed model.

        evotype : Evotype or str, optional
            The evolution type of this model, describing how states are
            represented.  The special value `"default"` is equivalent
            to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

        spamtype : { "static", "lindblad", "tensorproduct" }
            Specifies how the SPAM elements of the returned `Model` are formed.
            Static elements are ideal (perfect) operations with no parameters, i.e.
            no possibility for noise.  Lindblad SPAM operations are the "normal"
            way to allow SPAM noise, in which case error terms up to weight
            `max_spam_weight` are included.  Tensor-product operations require that
            the state prep and POVM effects have a tensor-product structure; the
            "tensorproduct" mode exists for historical reasons and is *deprecated*
            in favor of `"lindblad"`; use it only if you know what you're doing.

        add_idle_noise_to_all_gates : bool, optional
            Whether the global idle should be added as a factor following the
            ideal action of each of the non-idle gates.

        errcomp_type : {"gates","errorgens"}
            How errors are composed when creating layer operations in the created
            model.  `"gates"` means that the errors on multiple gates in a single
            layer are composed as separate and subsequent processes.  Specifically,
            the layer operation has the form `Composed(target,idleErr,cloudErr)`
            where `target` is a composition of all the ideal gate operations in the
            layer, `idleErr` is idle error (`.operation_blks['layers']['globalIdle']`),
            and `cloudErr` is the composition (ordered as layer-label) of cloud-
            noise contributions, i.e. a map that acts as the product of exponentiated
            error-generator matrices.  `"errorgens"` means that layer operations
            have the form `Composed(target, error)` where `target` is as above and
            `error` results from composing the idle and cloud-noise error
            *generators*, i.e. a map that acts as the exponentiated sum of error
            generators (ordering is irrelevant in this case).

        independent_clouds : bool, optional
            Currently this must be set to True.  In a future version, setting to
            true will allow all the clouds of a given gate name to have a similar
            cloud-noise process, mapped to the full qubit graph via a stencil.

        sparse_lindblad_reps : bool, optional
            Whether created Lindblad operations use sparse (more memory efficient but
            slower action) or dense representations.

        verbosity : int, optional
            An integer >= 0 dictating how must output to send to stdout.

        Returns
        -------
        CloudNoiseModel
        """
        printer = _VerbosityPrinter.create_printer(verbosity)

        if custom_gates is None: custom_gates = {}
        if nonstd_gate_unitaries is None: nonstd_gate_unitaries = {}
        std_unitaries = _itgs.standard_gatename_unitaries()

        gatedict = _collections.OrderedDict()
        for name in gate_names:
            if name in custom_gates:
                gatedict[name] = custom_gates[name]
            else:
                U = nonstd_gate_unitaries.get(name, std_unitaries.get(name, None))
                if U is None:
                    raise KeyError("'%s' gate unitary needs to be provided by `nonstd_gate_unitaries` arg" % name)
                if callable(U):  # then assume a function: args -> unitary
                    U0 = U(None)  # U fns must return a sample unitary when passed None to get size.
                    local_state_space = _statespace.default_space_for_udim(U0.shape[0])
                    gatedict[name] = _opfactory.UnitaryOpFactory(U, local_state_space, 'pp', evotype=evotype)
                else:
                    gatedict[name] = _bt.change_basis(_gt.unitary_to_process_mx(U), "std", 'pp')
                    # assume evotype is a densitymx or term type

        #Add anything from custom_gates directly if it wasn't added already
        for lbl, gate in custom_gates.items():
            if lbl not in gate_names: gatedict[lbl] = gate

        if not independent_clouds:
            raise NotImplementedError("Non-independent noise clounds are not supported yet!")

        if isinstance(geometry, _qgraph.QubitGraph):
            qubitGraph = geometry
            if qubit_labels is None:
                qubit_labels = qubitGraph.node_names
        else:
            if qubit_labels is None:
                qubit_labels = tuple(range(num_qubits))
            qubitGraph = _qgraph.QubitGraph.common_graph(num_qubits, geometry, directed=False,
                                                         qubit_labels=qubit_labels)
            printer.log("Created qubit graph:\n" + str(qubitGraph))

        state_space = _statespace.QubitSpace(qubit_labels)
        assert(state_space.num_qubits == num_qubits), "Number of qubit labels != `num_qubits`!"

        #Process "auto" simulator
        if simulator == "auto":
            simulator = _MapFSim() if num_qubits > 2 else _MatrixFSim()
        elif simulator == "map":
            simulator = _MapFSim()
        elif simulator == "matrix":
            simulator = _MatrixFSim()
        assert(isinstance(simulator, _FSim)), "`simulator` must be a ForwardSimulator instance!"

        #Global Idle
        if max_idle_weight > 0:
            printer.log("Creating Idle:")
            global_idle_layer = _build_nqn_global_noise(
                qubitGraph, max_idle_weight, sparse_lindblad_basis, sparse_lindblad_reps,
                simulator, parameterization, evotype, errcomp_type, printer - 1)
        else:
            global_idle_layer = None

        #SPAM
        if spamtype == "static" or max_spam_weight == 0:
            if max_spam_weight > 0:
                _warnings.warn(("`spamtype == 'static'` ignores the supplied "
                                "`max_spam_weight=%d > 0`") % max_spam_weight)
            prep_layers = [_state.ComputationalBasisState([0] * num_qubits, 'pp', evotype, state_space)]
            povm_layers = {'Mdefault': _povm.ComputationalBasisPOVM(num_qubits, evotype, state_space=state_space)}

        elif spamtype == "tensorproduct":

            _warnings.warn("`spamtype == 'tensorproduct'` is deprecated!")
            basis1Q = _BuiltinBasis("pp", 4)
            prep_factors = []; povm_factors = []

            from ..construction.modelconstruction import _basis_create_spam_vector

            v0 = _basis_create_spam_vector("0", basis1Q)
            v1 = _basis_create_spam_vector("1", basis1Q)

            # Historical use of TP for non-term-based cases?
            #  - seems we could remove this. FUTURE REMOVE?
            povmtyp = rtyp = "TP" if parameterization in \
                             ("CPTP", "H+S", "S", "H+S+A", "S+A", "H+D+A", "D+A", "D") \
                             else parameterization

            for i in range(num_qubits):
                prep_factors.append(
                    _state.convert(_state.StaticState(v0, evotype, state_space=None), rtyp, basis1Q))
                povm_factors.append(
                    _povm.convert(_povm.UnconstrainedPOVM(([
                        ('0', _povm.StaticPOVMEffect(v0, evotype, state_space=None)),
                        ('1', _povm.StaticPOVMEffect(v1, evotype, state_space=None))])),
                        povmtyp, basis1Q))

            prep_layers = [_state.TensorProductState(prep_factors, state_space)]
            povm_layers = {'Mdefault': _povm.TensorProductPOVM(povm_factors, evotype, state_space)}

        elif spamtype == "lindblad":

            prepPure = _state.ComputationalBasisState([0] * num_qubits, 'pp', evotype, state_space)
            prepNoiseMap = _build_nqn_global_noise(
                qubitGraph, max_spam_weight, sparse_lindblad_basis, sparse_lindblad_reps, simulator,
                parameterization, evotype, errcomp_type, printer - 1)
            prep_layers = [_state.ComposedState(prepPure, prepNoiseMap)]

            povmNoiseMap = _build_nqn_global_noise(
                qubitGraph, max_spam_weight, sparse_lindblad_basis, sparse_lindblad_reps, simulator,
                parameterization, evotype, errcomp_type, printer - 1)
            povm_layers = {'Mdefault': _povm.ComposedPOVM(povmNoiseMap, None, "pp")}

        else:
            raise ValueError("Invalid `spamtype` argument: %s" % spamtype)

        weight_maxhops_tuples_1Q = [(1, maxhops + extra_weight_1_hops)] + \
                                   [(1 + x, maxhops) for x in range(1, extra_gate_weight + 1)]
        cloud_maxhops_1Q = max([mx for wt, mx in weight_maxhops_tuples_1Q])  # max of max-hops

        weight_maxhops_tuples_2Q = [(1, maxhops + extra_weight_1_hops), (2, maxhops)] + \
                                   [(2 + x, maxhops) for x in range(1, extra_gate_weight + 1)]
        cloud_maxhops_2Q = max([mx for wt, mx in weight_maxhops_tuples_2Q])  # max of max-hops

        def build_cloudnoise_fn(lbl):
            gate_nQubits = len(lbl.sslbls)
            if gate_nQubits not in (1, 2):
                raise ValueError("Only 1- and 2-qubit gates are supported.  %s acts on %d qubits!"
                                 % (str(lbl.name), gate_nQubits))
            weight_maxhops_tuples = weight_maxhops_tuples_1Q if len(lbl.sslbls) == 1 else weight_maxhops_tuples_2Q
            return _build_nqn_cloud_noise(
                [qubitGraph.node_names.index(nn) for nn in lbl.sslbls], qubitGraph, weight_maxhops_tuples,
                errcomp_type=errcomp_type, sparse_lindblad_basis=sparse_lindblad_basis,
                sparse_lindblad_reps=sparse_lindblad_reps, simulator=simulator, parameterization=parameterization,
                evotype=evotype, verbosity=printer - 1)

        def build_cloudkey_fn(lbl):
            cloud_maxhops = cloud_maxhops_1Q if len(lbl.sslbls) == 1 else cloud_maxhops_2Q
            cloud_inds = tuple(qubitGraph.radius(lbl.sslbls, cloud_maxhops))
            cloud_key = (tuple(lbl.sslbls), tuple(sorted(cloud_inds)))  # (sets are unhashable)
            return cloud_key

        return cls(num_qubits, gatedict, availability, qubit_labels, geometry,
                   global_idle_layer, prep_layers, povm_layers,
                   build_cloudnoise_fn, build_cloudkey_fn,
                   simulator, evotype, errcomp_type,
                   add_idle_noise_to_all_gates, sparse_lindblad_reps, printer)

    def __init__(self, num_qubits, gatedict, availability=None,
                 qubit_labels=None, geometry="line",
                 global_idle_layer=None, prep_layers=None, povm_layers=None,
                 build_cloudnoise_fn=None, build_cloudkey_fn=None,
                 simulator="map", evotype="default", errcomp_type="gates",
                 add_idle_noise_to_all_gates=True, sparse_lindblad_reps=False, verbosity=0):
        """
        Creates a CloudNoiseModel.

        A CloudNoiseModle is a n-qubit model using a low-weight and geometrically local
        error model with a common "global idle" operation.

        This constructor relies on factory functions being passed to it
        which generate the cloud-noise operators - noise thtat is specific
        to a gate but may act on a neighborhood or cloud around the gate's
        target qubits.

        Parameters
        ----------
        num_qubits : int
            The number of qubits

        gatedict : dict
            A dictionary (an `OrderedDict` if you care about insertion order) that
            associates with string-type gate names (e.g. `"Gx"`) :class:`LinearOperator`,
            `numpy.ndarray`, or :class:`OpFactory` objects. When the objects may act on
            fewer than the total number of qubits (determined by their dimension/shape) then
            they are repeatedly embedded into `num_qubits`-qubit gates as specified by their
            `availability`.  These operations represent the ideal target operations, and
            thus, any `LinearOperator` or `OpFactory` objects must be *static*, i.e., have
            zero parameters.

        availability : dict, optional
            A dictionary whose keys are the same gate names as in
            `gatedict` and whose values are lists of qubit-label-tuples.  Each
            qubit-label-tuple must have length equal to the number of qubits
            the corresponding gate acts upon, and causes that gate to be
            embedded to act on the specified qubits.  For example,
            `{ 'Gx': [(0,),(1,),(2,)], 'Gcnot': [(0,1),(1,2)] }` would cause
            the `1-qubit `'Gx'`-gate to be embedded three times, acting on qubits
            0, 1, and 2, and the 2-qubit `'Gcnot'`-gate to be embedded twice,
            acting on qubits 0 & 1 and 1 & 2.  Instead of a list of tuples,
            values of `availability` may take the special values:

            - `"all-permutations"` and `"all-combinations"` equate to all possible
            permutations and combinations of the appropriate number of qubit labels
            (deterined by the gate's dimension).
            - `"all-edges"` equates to all the vertices, for 1Q gates, and all the
            edges, for 2Q gates of the geometry.
            - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
            on any target qubits via an :class:`EmbeddingOpFactory` (uses less
            memory but slower than `"all-permutations"`.

            If a gate name (a key of `gatedict`) is not present in `availability`,
            the default is `"all-edges"`.

        qubit_labels : tuple, optional
            The circuit-line labels for each of the qubits, which can be integers
            and/or strings.  Must be of length `num_qubits`.  If None, then the
            integers from 0 to `num_qubits-1` are used.

        geometry : {"line","ring","grid","torus"} or QubitGraph
            The type of connectivity among the qubits, specifying a
            graph used to define neighbor relationships.  Alternatively,
            a :class:`QubitGraph` object with node labels equal to
            `qubit_labels` may be passed directly.

        global_idle_layer : LinearOperator
            A global idle operation which acts on all the qubits and
            is, if `add_idle_noise_to_all_gates=True`, composed with the
            actions of specific gates to form the layer operation of
            any circuit layer.

        prep_layers, povm_layers : None or operator or dict or list, optional
            The SPAM operations as n-qubit layer operations.  If `None`, then
            no preps (or POVMs) are created.  If a dict, then the keys are
            labels and the values are layer operators.  If a list, then the
            elements are layer operators and the labels will be assigned as
            "rhoX" and "MX" where X is an integer starting at 0.  If a single
            layer operation is given, then this is used as the sole prep or
            POVM and is assigned the label "rho0" or "Mdefault" respectively.

        build_cloudnoise_fn : function, optional
            A function which takes a single :class:`Label` as an argument and
            returns the cloud-noise operation for that primitive layer
            operation.  Note that if `errcomp_type="gates"` the returned
            operator should be a superoperator whereas if
            `errcomp_type="errorgens"` then the returned operator should be
            an error generator (not yet exponentiated).

        build_cloudkey_fn : function, optional
            An function which takes a single :class:`Label` as an argument and
            returns a "cloud key" for that primitive layer.  The "cloud" is the
            set of qubits that the error (the operator returned from
            `build_cloudnoise_fn`) touches -- and the "key" returned from this
            function is meant to identify that cloud.  This is used to keep track
            of which primitive layer-labels correspond to the same cloud - e.g.
            the cloud-key for ("Gx",2) and ("Gy",2) might be the same and could
            be processed together when selecing sequences that amplify the parameters
            in the cloud-noise operations for these two labels.  The return value
            should be something hashable with the property that two noise
            which act on the same qubits should have the same cloud key.

        simulator : ForwardSimulator or {"auto", "matrix", "map"}
            The simulator used to compute predicted probabilities for this
            :class:`Model`.  Using `"auto"` selects `"matrix"` when there
            are 2 qubits or less, and otherwise selects `"map"`.

        evotype : Evotype or str, optional
            The evolution type of this model, describing how states are
            represented.  The special value `"default"` is equivalent
            to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

        errcomp_type : {"gates","errorgens"}
            How errors are composed when creating layer operations in the created
            model.  `"gates"` means that the errors on multiple gates in a single
            layer are composed as separate and subsequent processes.  Specifically,
            the layer operation has the form `Composed(target,idleErr,cloudErr)`
            where `target` is a composition of all the ideal gate operations in the
            layer, `idleErr` is idle error (`.operation_blks['layers']['globalIdle']`),
            and `cloudErr` is the composition (ordered as layer-label) of cloud-
            noise contributions, i.e. a map that acts as the product of exponentiated
            error-generator matrices.  `"errorgens"` means that layer operations
            have the form `Composed(target, error)` where `target` is as above and
            `error` results from composing the idle and cloud-noise error
            *generators*, i.e. a map that acts as the exponentiated sum of error
            generators (ordering is irrelevant in this case).

        add_idle_noise_to_all_gates: bool, optional
            Whether the global idle should be added as a factor following the
            ideal action of each of the non-idle gates.

        sparse_lindblad_reps : bool, optional
            Whether created Lindblad operations use sparse (more memory efficient but
            slower action) or dense representations.

        verbosity : int, optional
            An integer >= 0 dictating how must output to send to stdout.
        """
        if qubit_labels is None:
            qubit_labels = tuple(range(num_qubits))
        if availability is None:
            availability = {}

        state_space = _statespace.QubitSpace(qubit_labels)
        assert(state_space.num_qubits == num_qubits), "Number of qubit labels != `num_qubits`!"

        prefer_dense_reps = isinstance(simulator, _MatrixFSim)
        evotype = _Evotype.cast(evotype, prefer_dense_reps)

        # Build gate dictionaries. A value of `gatedict` can be an array, a LinearOperator, or an OpFactory.
        # For later processing, we'll create mm_gatedict to contain each item as a ModelMember.  For cloud-
        # noise models, these gate operations should be *static* (no parameters) as they represent the target
        # operations and all noise (and parameters) are assumed to enter through the cloudnoise members.
        mm_gatedict = _collections.OrderedDict()  # static *target* ops as ModelMembers
        #REMOVE self.gatedict = _collections.OrderedDict()  # static *target* ops (unused) as numpy arrays
        for gn, gate in gatedict.items():
            if isinstance(gate, _op.LinearOperator):
                assert(gate.num_params == 0), "Only *static* ideal operators are allowed in `gatedict`!"
                #REMOVE self.gatedict[gn] = gate.to_dense()
                #if gate._evotype != evotype and isinstance(gate, _op.StaticDenseOp):
                #    # special case: we'll convert static ops to the right evotype (convenient)
                #    mm_gatedict[gn] = StaticDenseOp(gate, "pp")
                #else:
                mm_gatedict[gn] = gate
            elif isinstance(gate, _opfactory.OpFactory):
                assert(gate.num_params == 0), "Only *static* ideal factories are allowed in `gatedict`!"
                # don't store factories in self.gatedict for now (no good dense representation)
                mm_gatedict[gn] = gate
            else:  # presumably a numpy array or something like it:
                #REMOVE self.gatedict[gn] = _np.array(gate)
                mm_gatedict[gn] = _op.StaticDenseOp(gate, evotype, state_space=None)  # use default state space
            assert(mm_gatedict[gn]._evotype == evotype)

        #Set other members
        self.availability = availability.copy()  # create a local copy because we may update it below
        self.geometry = geometry
        #TODO REMOVE unneeded members
        #self.qubit_labels = qubit_labels
        #self.nQubits = num_qubits
        #self.maxIdleWeight = maxIdleWeight
        #self.maxSpamWeight = maxSpamWeight
        #self.maxhops = maxhops
        #self.extraWeight1Hops = extraWeight1Hops
        #self.extraGateWeight = extraGateWeight
        #self.parameterization = parameterization
        #self.spamtype = spamtype
        self.addIdleNoiseToAllGates = add_idle_noise_to_all_gates
        self.errcomp_type = errcomp_type

        #Process "auto" simulator
        if simulator == "auto":
            simulator = _MapFSim() if num_qubits > 2 else _MatrixFSim()
        elif simulator == "map":
            simulator = _MapFSim()
        elif simulator == "matrix":
            simulator = _MatrixFSim()
        assert(isinstance(simulator, _FSim)), "`simulator` must be a ForwardSimulator instance!"

        if global_idle_layer is None:
            self.addIdleNoiseToAllGates = False  # there is no idle noise to add!
        layer_rules = CloudNoiseLayerRules(self.addIdleNoiseToAllGates, errcomp_type, sparse_lindblad_reps)
        super(CloudNoiseModel, self).__init__(state_space, layer_rules, "pp", simulator=simulator, evotype=evotype)

        flags = {'auto_embed': False, 'match_parent_statespace': False,
                 'match_parent_evotype': True, 'cast_to_type': None}
        self.prep_blks['layers'] = _ld.OrderedMemberDict(self, None, None, flags)
        self.povm_blks['layers'] = _ld.OrderedMemberDict(self, None, None, flags)
        self.operation_blks['layers'] = _ld.OrderedMemberDict(self, None, None, flags)
        self.operation_blks['gates'] = _ld.OrderedMemberDict(self, None, None, flags)
        self.operation_blks['cloudnoise'] = _ld.OrderedMemberDict(self, None, None, flags)
        self.instrument_blks['layers'] = _ld.OrderedMemberDict(self, None, None, flags)
        self.factories['layers'] = _ld.OrderedMemberDict(self, None, None, flags)
        self.factories['gates'] = _ld.OrderedMemberDict(self, None, None, flags)
        self.factories['cloudnoise'] = _ld.OrderedMemberDict(self, None, None, flags)

        printer = _VerbosityPrinter.create_printer(verbosity)
        geometry_name = "custom" if isinstance(geometry, _qgraph.QubitGraph) else geometry
        printer.log("Creating a %d-qubit cloud-noise %s model" % (num_qubits, geometry_name))

        if isinstance(geometry, _qgraph.QubitGraph):
            qubitGraph = geometry
        else:
            qubitGraph = _qgraph.QubitGraph.common_graph(num_qubits, geometry, directed=False,
                                                         qubit_labels=qubit_labels)
            printer.log("Created qubit graph:\n" + str(qubitGraph))

        if global_idle_layer is None:
            pass
        elif callable(global_idle_layer):
            self.operation_blks['layers'][_Lbl('globalIdle')] = global_idle_layer()
        else:
            self.operation_blks['layers'][_Lbl('globalIdle')] = global_idle_layer

        # a dictionary of "cloud" objects
        # keys = cloud identifiers, e.g. (target_qubit_indices, cloud_qubit_indices) tuples
        # values = list of gate-labels giving the gates (primitive layers?) associated with that cloud (necessary?)
        self._clouds = _collections.OrderedDict()

        #Get gates availability
        gates_and_avail = _collections.OrderedDict()
        for gateName, gate in mm_gatedict.items():  # gate is a static ModelMember (op or factory)
            gate_nQubits = gate.state_space.num_qubits

            availList = self.availability.get(gateName, 'all-edges')
            if availList == 'all-combinations':
                availList = list(_itertools.combinations(qubit_labels, gate_nQubits))
            elif availList == 'all-permutations':
                availList = list(_itertools.permutations(qubit_labels, gate_nQubits))
            elif availList == 'all-edges':
                if gate_nQubits == 1:
                    availList = [(i,) for i in qubit_labels]
                elif gate_nQubits == 2:
                    availList = qubitGraph.edges(double_for_undirected=True)
                else:
                    raise NotImplementedError(("I don't know how to place a %d-qubit gate "
                                               "on graph edges yet") % gate_nQubits)
            elif availList in ('arbitrary', '*'):
                availList = [('*', gate_nQubits)]  # let a factory determine what's "available"

            self.availability[gateName] = tuple(availList)
            gates_and_avail[gateName] = (gate, availList)

        for gn, (gate, availList) in gates_and_avail.items():
            #Note: gate was taken from mm_gatedict, and so is a static op or factory
            gate_is_factory = isinstance(gate, _opfactory.OpFactory)

            if gate_is_factory:
                self.factories['gates'][_Lbl(gn)] = gate
            else:
                self.operation_blks['gates'][_Lbl(gn)] = gate

            for inds in availList:  # inds are target qubit labels

                #Target operation
                if inds[0] == '*':
                    printer.log("Creating %dQ %s gate on arbitrary qubits!!" % (inds[1], gn))

                    self.factories['layers'][_Lbl(gn)] = _opfactory.EmbeddingOpFactory(
                        state_space, gate, num_target_labels=inds[1])
                    # add any primitive ops for this embedding factory?
                else:
                    printer.log("Creating %dQ %s gate on qubits %s!!" % (len(inds), gn, inds))
                    assert(_Lbl(gn, inds) not in gatedict), \
                        ("Cloudnoise models do not accept primitive-op labels, e.g. %s, in `gatedict` as this dict "
                         "specfies the ideal target gates. Perhaps make the cloudnoise depend on the target qubits "
                         "of the %s gate?") % (str(_Lbl(gn, inds)), gn)

                    if gate_is_factory:
                        self.factories['layers'][_Lbl(gn, inds)] = _opfactory.EmbeddedOpFactory(
                            state_space, inds, gate)
                        # add any primitive ops for this factory?
                    else:
                        self.operation_blks['layers'][_Lbl(gn, inds)] = _op.EmbeddedOp(
                            state_space, inds, gate)

                #Cloudnoise operation
                if build_cloudnoise_fn is not None:
                    if inds[0] == '*':
                        cloudnoise = build_cloudnoise_fn(_Lbl(gn))
                        assert(isinstance(cloudnoise, _opfactory.EmbeddingOpFactory)), \
                            ("`build_cloudnoise_fn` must return an EmbeddingOpFactory for gate %s"
                             " with arbitrary availability") % gn
                        self.factories['cloudnoise'][_Lbl(gn)] = cloudnoise
                    else:
                        cloudnoise = build_cloudnoise_fn(_Lbl(gn, inds))
                        if isinstance(cloudnoise, _opfactory.OpFactory):
                            self.factories['cloudnoise'][_Lbl(gn, inds)] = cloudnoise
                        else:
                            self.operation_blks['cloudnoise'][_Lbl(gn, inds)] = cloudnoise

                #REMOVE
                #_build_nqn_cloud_noise(
                #    (i,), qubitGraph, weight_maxhops_tuples_1Q,
                #    errcomp_type=errcomp_type, sparse=sparse, sim_type=sim_type,
                #    parameterization=parameterization, verbosity=printer - 1)
                #cloud_inds = tuple(qubitGraph.radius((i,), cloud_maxhops))
                #cloud_key = ((i,), tuple(sorted(cloud_inds)))  # (sets are unhashable)

                if inds[0] != '*' and build_cloudkey_fn is not None:
                    # TODO: is there any way to get a default "key", e.g. the
                    # qubits touched by the corresponding cloudnoise op?
                    # need a way to identify a clound (e.g. Gx and Gy gates on some qubit will have the *same* cloud)
                    cloud_key = build_cloudkey_fn(_Lbl(gn, inds))
                    if cloud_key not in self.clouds: self.clouds[cloud_key] = []
                    self.clouds[cloud_key].append(_Lbl(gn, inds))
                #keep track of the primitive-layer labels in each cloud,
                # used to specify which gate parameters should be amplifiable by germs for a given cloud (?) TODO CHECK

        #SPAM (same as for local noise model)
        if prep_layers is None:
            pass  # no prep layers
        elif isinstance(prep_layers, dict):
            for rhoname, layerop in prep_layers.items():
                self.prep_blks['layers'][_Lbl(rhoname)] = layerop
        elif isinstance(prep_layers, _op.LinearOperator):  # just a single layer op
            self.prep_blks['layers'][_Lbl('rho0')] = prep_layers
        else:  # assume prep_layers is an iterable of layers, e.g. isinstance(prep_layers, (list,tuple)):
            for i, layerop in enumerate(prep_layers):
                self.prep_blks['layers'][_Lbl("rho%d" % i)] = layerop

        if povm_layers is None:
            pass  # no povms
        elif isinstance(povm_layers, _povm.POVM):  # just a single povm - must precede 'dict' test!
            self.povm_blks['layers'][_Lbl('Mdefault')] = povm_layers
        elif isinstance(povm_layers, dict):
            for povmname, layerop in povm_layers.items():
                self.povm_blks['layers'][_Lbl(povmname)] = layerop
        else:  # assume povm_layers is an iterable of layers, e.g. isinstance(povm_layers, (list,tuple)):
            for i, layerop in enumerate(povm_layers):
                self.povm_blks['layers'][_Lbl("M%d" % i)] = layerop

        printer.log("DONE! - created Model with nqubits=%d and op-blks=" % self.state_space.num_qubits)
        for op_blk_lbl, op_blk in self.operation_blks.items():
            printer.log("  %s: %s" % (op_blk_lbl, ', '.join(map(str, op_blk.keys()))))

    @property
    def clouds(self):
        """
        Returns the set of cloud-sets used when creating sequences which amplify the parameters of this model.

        Returns
        -------
        dict
        """
        return self._clouds


def _get_experrgen_factory(simulator, parameterization, errcomp_type, evotype):
    """ Returns a function that creates a ExpErrorgen-type gate appropriate
        given the simulation type and parameterization """

    if errcomp_type == "gates":
        ExpErrorgenOp = _op.ExpErrorgenOp

        #Just call from_operation_matrix with appropriate evotype
        def _f(op_matrix, proj_basis="pp", mx_basis="pp", relative=False):
            p = parameterization
            if relative:
                if parameterization == "CPTP": p = "GLND"
                elif "S" in parameterization: p = parameterization.replace("S", "s")
                elif "D" in parameterization: p = parameterization.replace("D", "d")

            nonham_mode, param_mode, use_ham_basis, use_nonham_basis = \
                _op.LindbladErrorgen.decomp_paramtype(p)
            ham_basis = proj_basis if use_ham_basis else None
            nonham_basis = proj_basis if use_nonham_basis else None
            errorgen = _op.LindbladErrorgen.from_operation_matrix(op_matrix, ham_basis, nonham_basis, param_mode,
                                                                  nonham_mode, True, mx_basis, evotype)  # truncate=True
            return ExpErrorgenOp(errorgen)
        return _f

    elif errcomp_type == "errorgens":
        def _f(error_gen, proj_basis="pp", mx_basis="pp", relative=False):
            p = parameterization
            if relative:
                if parameterization == "CPTP": p = "GLND"
                elif "S" in parameterization: p = parameterization.replace("S", "s")
                elif "D" in parameterization: p = parameterization.replace("D", "d")

            nonham_mode, param_mode, use_ham_basis, use_nonham_basis = \
                _op.LindbladErrorgen.decomp_paramtype(p)
            ham_basis = proj_basis if use_ham_basis else None
            nonham_basis = proj_basis if use_nonham_basis else None

            return _op.LindbladErrorgen.from_error_generator(error_gen, ham_basis, nonham_basis,
                                                             param_mode, nonham_mode, mx_basis,
                                                             truncate=True, evotype=evotype)
        return _f

    else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)


#REMOVE
#def _get_static_factory(simulator, evotype):
#    """ Returns a function that creates a static-type gate appropriate
#        given the simulation and parameterization """
#    if evotype == "densitymx":
#        if isinstance(simulator, _MatrixFSim):
#            return lambda g, b: _op.StaticDenseOp(g, evotype)
#        else:  # e.g. "map"-type forward simes
#            return lambda g, b: _op.StaticDenseOp(g, evotype)  # TODO: create StaticGateMap?
#
#    elif evotype in ("svterm", "cterm"):
#        assert(isinstance(simulator, _TermFSim))
#
#        def _f(op_matrix, mx_basis="pp"):
#            errorgen = _op.LindbladErrorgen.from_operation_matrix(
#                op_matrix, None, None, mx_basis=mx_basis, evotype=evotype)
#            # a LindbladErrorgen with None as ham_basis and nonham_basis => no parameters
#            return _op.ExpErrorgenOp(errorgen)
#
#        return _f
#    raise ValueError("Cannot create Static gate factory for simtype=%s evotype=%s" %
#                     (str(type(simulator)), evotype))


def _build_nqn_global_noise(qubit_graph, max_weight, sparse_lindblad_basis=False, sparse_lindblad_reps=False,
                            simulator=None, parameterization="H+S", evotype='default', errcomp_type="gates",
                            verbosity=0):
    """
    Create a "global" idle gate, meaning one that acts on all the qubits in
    `qubit_graph`.  The gate will have up to `max_weight` errors on *connected*
    (via the graph) sets of qubits.

    Parameters
    ----------
    qubit_graph : QubitGraph
        A graph giving the geometry (nearest-neighbor relations) of the qubits.

    max_weight : int
        The maximum weight errors to include in the resulting gate.

    sparse_lindblad_basis : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        gate are represented as sparse or dense matrices.  (This is determied by
        whether they are constructed using sparse basis matrices.)

    sparse_lindblad_reps : bool, optional
        Whether created Lindblad operations use sparse (more memory efficient but
        slower action) or dense representations.

    simulator : ForwardSimulator
        The forward simulation (probability computation) being used by
        the model this gate is destined for. `None` means a :class:`MatrixForwardSimulator`.

    parameterization : str
        The type of parameterizaton for the constructed gate. E.g. "H+S",
        "CPTP", etc.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    errcomp_type : {"gates","errorgens"}
        How errors are composed when creating layer operations in the associated
        model.  See :method:`CloudnoiseModel.__init__` for details.

    verbosity : int, optional
        An integer >= 0 dictating how must output to send to stdout.

    Returns
    -------
    LinearOperator
    """
    assert(max_weight <= 2), "Only `max_weight` equal to 0, 1, or 2 is supported"
    if simulator is None: simulator = _MatrixFSim()

    prefer_dense_reps = isinstance(simulator, _MatrixFSim)
    evotype = _Evotype.cast(evotype, prefer_dense_reps)

    if errcomp_type == "gates":
        Composed = _op.ComposedOp
        Embedded = _op.EmbeddedOp
    elif errcomp_type == "errorgens":
        Composed = _op.ComposedErrorgen
        Embedded = _op.EmbeddedErrorgen
    else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
    ExpErrorgen = _get_experrgen_factory(simulator, parameterization, errcomp_type, evotype)
    #constructs a gate or errorgen based on value of errcomp_type

    printer = _VerbosityPrinter.create_printer(verbosity)
    printer.log("*** Creating global idle ***")

    termops = []  # gates or error generators to compose
    qubit_labels = qubit_graph.node_names
    state_space = _statespace.QubitSpace(qubit_labels)

    nQubits = qubit_graph.nqubits
    possible_err_qubit_inds = _np.arange(nQubits)
    nPossible = nQubits
    for wt in range(1, max_weight + 1):
        printer.log("Weight %d: %d possible qubits" % (wt, nPossible), 2)
        basisEl_Id = basis_product_matrix(_np.zeros(wt, _np.int64), sparse_lindblad_basis)
        if errcomp_type == "gates":
            wtNoErr = _sps.identity(4**wt, 'd', 'csr') if sparse_lindblad_basis else _np.identity(4**wt, 'd')
        elif errcomp_type == "errorgens":
            wtNoErr = _sps.csr_matrix((4**wt, 4**wt)) if sparse_lindblad_basis else _np.zeros((4**wt, 4**wt), 'd')
        else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
        wtBasis = _BuiltinBasis('pp', 4**wt, sparse=sparse_lindblad_basis)

        for err_qubit_inds in _itertools.combinations(possible_err_qubit_inds, wt):
            if len(err_qubit_inds) == 2 and not qubit_graph.is_directly_connected(qubit_labels[err_qubit_inds[0]],
                                                                                  qubit_labels[err_qubit_inds[1]]):
                continue  # TO UPDATE - check whether all wt indices are a connected subgraph

            errbasis = [basisEl_Id]
            errbasis_lbls = ['I']
            for err_basis_inds in _iter_basis_inds(wt):
                error = _np.array(err_basis_inds, _np.int64)  # length == wt
                basisEl = basis_product_matrix(error, sparse_lindblad_basis)
                errbasis.append(basisEl)
                errbasis_lbls.append(''.join(["IXYZ"[i] for i in err_basis_inds]))

            printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_inds, len(errbasis)), 3)
            errbasis = _ExplicitBasis(errbasis, errbasis_lbls, real=True, sparse=sparse_lindblad_basis)
            termErr = ExpErrorgen(wtNoErr, proj_basis=errbasis, mx_basis=wtBasis)

            err_qubit_global_inds = err_qubit_inds
            fullTermErr = Embedded(state_space, [qubit_labels[i] for i in err_qubit_global_inds], termErr)
            assert(fullTermErr.num_params == termErr.num_params)
            printer.log("Exp(errgen) gate w/nqubits=%d and %d params -> embedded to gate w/nqubits=%d" %
                        (termErr.state_space.num_qubits, termErr.num_params, fullTermErr.state_space.num_qubits))

            termops.append(fullTermErr)

    if errcomp_type == "gates":
        return Composed(termops)
    elif errcomp_type == "errorgens":
        errgen = Composed(termops)
        #assert(not(sparse_lindblad_reps and isinstance(simulator, _MatrixFSim))), \
        #    "Cannot use sparse ExpErrorgen-op reps with a MatrixForwardSimulator!"
        return _op.ExpErrorgenOp(None, errgen)
    else: assert(False)


def _build_nqn_cloud_noise(target_qubit_inds, qubit_graph, weight_maxhops_tuples,
                           errcomp_type="gates", sparse_lindblad_basis=False, sparse_lindblad_reps=False,
                           simulator=None, parameterization="H+S", evotype='default', verbosity=0):
    """
    Create an n-qubit gate that is a composition of:

    `target_op(target_qubits) -> idle_noise(all_qubits) -> loc_noise(local_qubits)`

    where `idle_noise` is given by the `idle_noise` argument and `loc_noise` is
    given by the rest of the arguments.  `loc_noise` can be implemented either
    by a single (n-qubit) embedded exp(errorgen) gate with all relevant error
    generators, or as a composition of embedded single-errorgenerator exp(errorgen) gates
    (see param `errcomp_type`).

    The local noise consists terms up to a maximum weight acting on the qubits
    given reachable by a given maximum number of hops (along the neareset-
    neighbor edges of `qubit_graph`) from the target qubits.


    Parameters
    ----------
    target_qubit_inds : list
        The indices of the target qubits.

    qubit_graph : QubitGraph
        A graph giving the geometry (nearest-neighbor relations) of the qubits.

    weight_maxhops_tuples : iterable
        A list of `(weight,maxhops)` 2-tuples specifying which error weights
        should be included and what region of the graph (as a `maxhops` from
        the set of target qubits) should have errors of the given weight applied
        to it.

    errcomp_type : {"gates","errorgens"}
        How errors are composed when creating layer operations in the associated
        model.  See :method:`CloudnoiseModel.__init__` for details.

    sparse_lindblad_basis : bool, optional
        TODO - update docstring and probabaly rename this and arg below
        Whether the embedded Lindblad-parameterized gates within the constructed
        gate are represented as sparse or dense matrices.  (This is determied by
        whether they are constructed using sparse basis matrices.)

    sparse_lindblad_reps : bool, optional
        Whether created Lindblad operations use sparse (more memory efficient but
        slower action) or dense representations.

    simulator : ForwardSimulator
        The forward simulation (probability computation) being used by
        the model this gate is destined for. `None` means a :class:`MatrixForwardSimulator`.

    parameterization : str
        The type of parameterizaton for the constructed gate. E.g. "H+S",
        "CPTP", etc.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    verbosity : int, optional
        An integer >= 0 dictating how must output to send to stdout.

    Returns
    -------
    LinearOperator
    """
    if simulator is None: simulator = _MatrixFSim()

    prefer_dense_reps = isinstance(simulator, _MatrixFSim)
    evotype = _Evotype.cast(evotype, prefer_dense_reps)

    if errcomp_type == "gates":
        Composed = _op.ComposedOp
        Embedded = _op.EmbeddedOp
    elif errcomp_type == "errorgens":
        Composed = _op.ComposedErrorgen
        Embedded = _op.EmbeddedErrorgen
    else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
    ExpErrorgen = _get_experrgen_factory(simulator, parameterization, errcomp_type, evotype)
    #constructs a gate or errorgen based on value of errcomp_type

    printer = _VerbosityPrinter.create_printer(verbosity)
    printer.log("Creating local-noise error factor (%s)" % errcomp_type)

    # make a composed-gate of embedded single-elementary-errogen exp(errogen)-gates or -errorgens,
    #  one for each specified error term

    loc_noise_termops = []  # list of gates to compose
    qubit_labels = qubit_graph.node_names
    state_space = _statespace.QubitSpace(qubit_labels)

    for wt, maxHops in weight_maxhops_tuples:

        ## loc_noise_errinds = [] # list of basis indices for all local-error terms
        radius_nodes = qubit_graph.radius([qubit_labels[i] for i in target_qubit_inds], maxHops)
        possible_err_qubit_inds = _np.array([qubit_labels.index(nn) for nn in radius_nodes], _np.int64)
        nPossible = len(possible_err_qubit_inds)  # also == "nLocal" in this case
        basisEl_Id = basis_product_matrix(_np.zeros(wt, _np.int64), sparse_lindblad_basis)  # identity basis el

        if errcomp_type == "gates":
            wtNoErr = _sps.identity(4**wt, 'd', 'csr') if sparse_lindblad_basis else _np.identity(4**wt, 'd')
        elif errcomp_type == "errorgens":
            wtNoErr = _sps.csr_matrix((4**wt, 4**wt)) if sparse_lindblad_basis else _np.zeros((4**wt, 4**wt), 'd')
        else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
        wtBasis = _BuiltinBasis('pp', 4**wt, sparse=sparse_lindblad_basis)

        printer.log("Weight %d, max-hops %d: %d possible qubits" % (wt, maxHops, nPossible), 3)
        # print("DB: possible qubits = ", possible_err_qubit_inds,
        #       " (radius of %d around %s)" % (maxHops,str(target_qubit_inds)))

        for err_qubit_local_inds in _itertools.combinations(list(range(nPossible)), wt):
            # err_qubit_inds are in range [0,nPossible-1] qubit indices
            #Future: check that err_qubit_inds marks qubits that are connected

            errbasis = [basisEl_Id]
            errbasis_lbls = ['I']
            for err_basis_inds in _iter_basis_inds(wt):
                error = _np.array(err_basis_inds, _np.int64)  # length == wt
                basisEl = basis_product_matrix(error, sparse_lindblad_basis)
                errbasis.append(basisEl)
                errbasis_lbls.append(''.join(["IXYZ"[i] for i in err_basis_inds]))

            err_qubit_global_inds = possible_err_qubit_inds[list(err_qubit_local_inds)]
            printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_global_inds, len(errbasis)), 4)
            errbasis = _ExplicitBasis(errbasis, errbasis_lbls, real=True, sparse=sparse_lindblad_basis)
            termErr = ExpErrorgen(wtNoErr, proj_basis=errbasis, mx_basis=wtBasis, relative=True)

            fullTermErr = Embedded(state_space, [qubit_labels[i] for i in err_qubit_global_inds], termErr)
            assert(fullTermErr.num_params == termErr.num_params)
            printer.log("Exp(errorgen) gate w/nqubits=%d and %d params -> embedded to gate w/nqubits=%d" %
                        (termErr.state_space.num_qubits, termErr.num_params, fullTermErr.state_space.num_qubits))

            loc_noise_termops.append(fullTermErr)

    fullCloudErr = Composed(loc_noise_termops)
    return fullCloudErr


class CloudNoiseLayerRules(_LayerRules):

    def __init__(self, add_idle_noise, errcomp_type, sparse_lindblad_reps):
        self.add_idle_noise = add_idle_noise
        self.errcomp_type = errcomp_type
        self.sparse_lindblad_reps = sparse_lindblad_reps

    def prep_layer_operator(self, model, layerlbl, caches):
        """
        Create the operator corresponding to `layerlbl`.

        Parameters
        ----------
        layerlbl : Label
            A circuit layer label.

        Returns
        -------
        POVM or SPAMVec
        """
        #No cache for preps
        return model.prep_blks['layers'][layerlbl]  # prep_blks['layer'] are full prep ops

    def povm_layer_operator(self, model, layerlbl, caches):
        """
        Create the operator corresponding to `layerlbl`.

        Parameters
        ----------
        layerlbl : Label
            A circuit layer label.

        Returns
        -------
        POVM or SPAMVec
        """
        # caches['povm-layers'] *are* just complete layers
        if layerlbl in caches['povm-layers']: return caches['povm-layers'][layerlbl]
        if layerlbl in model.povm_blks['layers']:
            return model.povm_blks['layers'][layerlbl]
        else:
            # See if this effect label could correspond to a *marginalized* POVM, and
            # if so, create the marginalized POVM and add its effects to model.effect_blks['layers']
            assert(isinstance(layerlbl, _Lbl))  # Sanity check (REMOVE?)
            povmName = _gt.effect_label_to_povm(layerlbl)
            if povmName in model.povm_blks['layers']:
                # implicit creation of marginalized POVMs whereby an existing POVM name is used with sslbls that
                # are not present in the stored POVM's label.
                mpovm = _povm.MarginalizedPOVM(model.povm_blks['layers'][povmName],
                                               model.state_space, layerlbl.sslbls)  # cache in FUTURE
                mpovm_lbl = _Lbl(povmName, layerlbl.sslbls)
                caches['povm-layers'].update(mpovm.simplify_effects(mpovm_lbl))
                assert(layerlbl in caches['povm-layers']), "Failed to create marginalized effect!"
                return caches['povm-layers'][layerlbl]
            else:
                raise KeyError("Could not build povm/effect for %s!" % str(layerlbl))

    def operation_layer_operator(self, model, layerlbl, caches):
        """
        Create the operator corresponding to `layerlbl`.

        Parameters
        ----------
        layerlbl : Label
            A circuit layer label.

        Returns
        -------
        LinearOperator
        """
        #Note: cache uses 'op-layers' for *simple target* layers, not complete ones
        if layerlbl in caches['complete-layers']: return caches['complete-layers'][layerlbl]
        #assert(not(self.sparse_lindblad_reps and isinstance(model._sim, _MatrixFSim))), \
        #    "Cannot use sparse ExpErrorgen-op reps with a MatrixForwardSimulator!"
        #dense_lindblad_reps = not self.sparse_lindblad_reps
        #dense_composed_reps = dense_lindblad_reps  # whether dense matrix gates should be created

        if isinstance(layerlbl, _CircuitLabel):
            op = self._create_op_for_circuitlabel(model, layerlbl)
            caches['complete-layers'][layerlbl] = op
            return op

        Composed = _op.ComposedOp
        ExpErrorgen = _op.ExpErrorgenOp
        Sum = _op.ComposedErrorgen
        #print("DB: CloudNoiseLayerLizard building gate %s for %s w/comp-type %s" %
        #      (('matrix' if dense else 'map'), str(oplabel), self.errcomp_type) )

        components = layerlbl.components
        if len(components) == 0:  # or layerlbl == 'Gi': # OLD: special case: 'Gi' acts as global idle!
            return model.operation_blks['layers']['globalIdle']  # idle!

        #Compose target operation from layer's component labels, which correspond
        # to the perfect (embedded) target ops in op_blks
        if len(components) > 1:
            targetOp = Composed([self._layer_component_targetop(model, l, caches['op-layers']) for l in components],
                                evotype=model.evotype, state_space=model.state_space)
        else: targetOp = self._layer_component_targetop(model, components[0], caches['op-layers'])
        ops_to_compose = [targetOp]

        if self.errcomp_type == "gates":
            if self.add_idle_noise: ops_to_compose.append(model.operation_blks['layers']['globalIdle'])
            component_cloudnoise_ops = self._layer_component_cloudnoises(model, components, caches['op-cloudnoise'])
            if len(component_cloudnoise_ops) > 0:
                if len(component_cloudnoise_ops) > 1:
                    localErr = Composed(component_cloudnoise_ops,
                                        evotype=model.evotype, state_space=model.state_space)
                else:
                    localErr = component_cloudnoise_ops[0]
                ops_to_compose.append(localErr)

        elif self.errcomp_type == "errorgens":
            #We compose the target operations to create a
            # final target op, and compose this with a *single* ExpErrorgen operation which has as
            # its error generator the composition (sum) of all the factors' error gens.
            errorGens = [model.operation_blks['layers']['globalIdle'].errorgen] if self.add_idle_noise else []
            errorGens.extend(self._layer_component_cloudnoises(model, components, caches['op-cloudnoise']))
            if len(errorGens) > 0:
                if len(errorGens) > 1:
                    error = ExpErrorgen(Sum(errorGens, state_space=model.state_space, evotype=model.evotype))
                else:
                    error = ExpErrorgen(errorGens[0])
                ops_to_compose.append(error)
        else:
            raise ValueError("Invalid errcomp_type in CloudNoiseLayerRules: %s" % str(self.errcomp_type))

        ret = Composed(ops_to_compose, evotype=model.evotype, state_space=model.state_space)
        model._init_virtual_obj(ret)  # so ret's gpindices get set
        caches['complete-layers'][layerlbl] = ret  # cache the final label value
        return ret

    def _layer_component_targetop(self, model, complbl, cache):
        """
        Retrieves the target- or ideal-operation portion of one component of a layer operation.

        Parameters
        ----------
        complbl : Label
            A component label of a larger layer label.

        Returns
        -------
        LinearOperator
        """
        if complbl in cache:
            return cache[complbl]  # caches['op-layers'] would hold "simplified" instrument members

        if isinstance(complbl, _CircuitLabel):
            raise NotImplementedError("Cloud noise models cannot simulate circuits with partial-layer subcircuits.")
            # In the FUTURE, could easily implement this for errcomp_type == "gates", but it's unclear what to
            #  do for the "errorgens" case - how do we gate an error generator of an entire (mulit-layer) sub-circuit?
            # Maybe we just need to expand the label and create a composition of those layers?
        elif complbl in model.operation_blks['layers']:
            return model.operation_blks['layers'][complbl]
        else:
            return _opfactory.op_from_factories(model.factories['layers'], complbl)

    def _layer_component_cloudnoises(self, model, complbl_list, cache):
        """
        Retrieves cloud-noise portion of the components of a layer operation.

        Get any present cloudnoise ops from a list of components.  This function processes
        a list rather than an item because it's OK if some components don't have
        corresponding cloudnoise ops - we just leave those off.

        Parameters
        ----------
        complbl_list : list
            A list of circuit-layer component labels.

        Returns
        -------
        list
        """
        ret = []
        for complbl in complbl_list:
            if complbl in cache:
                ret.append(cache[complbl])  # caches['cloudnoise-layers'] would hold "simplified" instrument members
            elif complbl in model.operation_blks['cloudnoise']:
                ret.append(model.operation_blks['cloudnoise'][complbl])
            else:
                try:
                    ret.append(_opfactory.op_from_factories(model.factories['cloudnoise'], complbl))
                except KeyError: pass  # OK if cloudnoise doesn't exist (means no noise)
        return ret
