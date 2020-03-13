""" Defines the CloudNoiseModel class and supporting functions """
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

from . import operation as _op
from . import spamvec as _sv
from . import povm as _povm
from . import qubitgraph as _qgraph
from . import labeldicts as _ld
from . import opfactory as _opfactory
from ..tools import optools as _gt
from ..tools import basistools as _bt
from ..tools import internalgates as _itgs
from .implicitmodel import ImplicitOpModel as _ImplicitOpModel
from .layerlizard import ImplicitLayerLizard as _ImplicitLayerLizard

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .basis import BuiltinBasis as _BuiltinBasis, ExplicitBasis as _ExplicitBasis
from .label import Label as _Lbl, CircuitLabel as _CircuitLabel

from ..tools.basisconstructors import sqrt2, id2x2, sigmax, sigmay, sigmaz


def _iter_basis_inds(weight):
    """ Iterate over product of `weight` non-identity Pauli 1Q basis indices """
    basisIndList = [[1, 2, 3]] * weight  # assume pauli 1Q basis, and only iterate over non-identity els
    for basisInds in _itertools.product(*basisIndList):
        yield basisInds


def basis_product_matrix(sigma_inds, sparse):
    """ Construct the Pauli product matrix from the given `sigma_inds` """
    sigmaVec = (id2x2 / sqrt2, sigmax / sqrt2, sigmay / sqrt2, sigmaz / sqrt2)
    M = _np.identity(1, 'complex')
    for i in sigma_inds:
        M = _np.kron(M, sigmaVec[i])
    return _sps.csr_matrix(M) if sparse else M


class CloudNoiseModel(_ImplicitOpModel):
    """
    A noisy n-qubit model using a low-weight and geometrically local
    error model with a common "global idle" operation.
    """

    @classmethod
    def build_from_hops_and_weights(cls, n_qubits, gate_names, nonstd_gate_unitaries=None,
                                    custom_gates=None, availability=None,
                                    qubit_labels=None, geometry="line",
                                    max_idle_weight=1, max_spam_weight=1, maxhops=0,
                                    extra_weight_1_hops=0, extra_gate_weight=0, sparse=False,
                                    sim_type="auto", parameterization="H+S",
                                    spamtype="lindblad", add_idle_noise_to_all_gates=True,
                                    errcomp_type="gates", independent_clouds=True, verbosity=0):
        """
        Create a n-qubit model using a low-weight and geometrically local
        error model with a common "global idle" operation.

        This type of model is referred to as a "cloud noise" model because
        noise specific to a gate may act on a neighborhood or cloud around
        the gate's target qubits.  This type of model is generally useful
        for performing GST on a multi-qubit system.


        Parameters
        ----------
        n_qubits : int
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
            and/or strings.  Must be of length `n_qubits`.  If None, then the
            integers from 0 to `n_qubits-1` are used.

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

        sparse : bool, optional
            Whether the embedded Lindblad-parameterized gates within the constructed
            `n_qubits`-qubit gates are sparse or not.  (This is determied by whether
            they are constructed using sparse basis matrices.)  When sparse, these
            Lindblad gates take up less memory, but their action is slightly slower.
            Usually it's fine to leave this as the default (False), except when
            considering particularly high-weight terms (b/c then the Lindblad gates
            are higher dimensional and sparsity has a significant impact).

        sim_type : {"auto","matrix","map","termorder:<N>"}
            The type of forward simulation (probability computation) to use for the
            returned :class:`Model`.  That is, how should the model compute
            operation sequence/circuit probabilities when requested.  `"matrix"` is better
            for small numbers of qubits, `"map"` is better for larger numbers. The
            `"termorder"` option is designed for even larger numbers.  Usually,
            the default of `"auto"` is what you want.

        parameterization : {"P", "P terms", "P clifford terms"}
            Where *P* can be any Lindblad parameterization base type (e.g. CPTP,
            H+S+A, H+S, S, D, etc.) This is the type of parameterizaton to use in
            the constructed model.  Types without any "terms" suffix perform
            usual density-matrix evolution to compute circuit probabilities.  The
            other "terms" options compute probabilities using a path-integral
            approach designed for larger numbers of qubits (experts only).

        spamtype : { "static", "lindblad", "tensorproduct" }
            Specifies how the SPAM elements of the returned `Model` are formed.
            Static elements are ideal (perfect) operations with no parameters, i.e.
            no possibility for noise.  Lindblad SPAM operations are the "normal"
            way to allow SPAM noise, in which case error terms up to weight
            `max_spam_weight` are included.  Tensor-product operations require that
            the state prep and POVM effects have a tensor-product structure; the
            "tensorproduct" mode exists for historical reasons and is *deprecated*
            in favor of `"lindblad"`; use it only if you know what you're doing.

        add_idle_noise_to_all_gates: bool, optional
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

        verbosity : int, optional
            An integer >= 0 dictating how must output to send to stdout.
        """
        printer = _VerbosityPrinter.build_printer(verbosity)

        if custom_gates is None: custom_gates = {}
        if nonstd_gate_unitaries is None: nonstd_gate_unitaries = {}
        std_unitaries = _itgs.get_standard_gatename_unitaries()

        #Get evotype
        _, evotype = _gt.split_lindblad_paramtype(parameterization)
        assert(evotype in ("densitymx", "svterm", "cterm")), "State-vector evolution types not allowed."

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
                    gatedict[name] = _opfactory.UnitaryOpFactory(U, U0.shape[0], evotype=evotype)
                else:
                    gatedict[name] = _bt.change_basis(_gt.unitary_to_process_mx(U), "std", "pp")
                    # assume evotype is a densitymx or term type

        #Add anything from custom_gates directly if it wasn't added already
        for lbl, gate in custom_gates.items():
            if lbl not in gate_names: gatedict[lbl] = gate

        if qubit_labels is None:
            qubit_labels = tuple(range(n_qubits))

        if not independent_clouds:
            raise NotImplementedError("Non-independent noise clounds are not supported yet!")

        if isinstance(geometry, _qgraph.QubitGraph):
            qubitGraph = geometry
        else:
            qubitGraph = _qgraph.QubitGraph.common_graph(n_qubits, geometry, directed=False,
                                                         qubit_labels=qubit_labels)
            printer.log("Created qubit graph:\n" + str(qubitGraph))

        #Process "auto" sim_type
        if sim_type == "auto":
            if evotype in ("svterm", "cterm"): sim_type = "termorder:1"
            else: sim_type = "map" if n_qubits > 2 else "matrix"
        assert(sim_type in ("matrix", "map") or sim_type.startswith("termorder") or sim_type.startswith("termgap"))

        #Global Idle
        if max_idle_weight > 0:
            printer.log("Creating Idle:")
            global_idle_layer = _build_nqn_global_noise(
                qubitGraph, max_idle_weight, sparse,
                sim_type, parameterization, errcomp_type, printer - 1)
        else:
            global_idle_layer = None

        #SPAM
        if spamtype == "static" or max_spam_weight == 0:
            if max_spam_weight > 0:
                _warnings.warn(("`spamtype == 'static'` ignores the supplied "
                                "`max_spam_weight=%d > 0`") % max_spam_weight)
            prep_layers = [_sv.ComputationalSPAMVec([0] * n_qubits, evotype)]
            povm_layers = {'Mdefault': _povm.ComputationalBasisPOVM(n_qubits, evotype)}

        elif spamtype == "tensorproduct":

            _warnings.warn("`spamtype == 'tensorproduct'` is deprecated!")
            basis1Q = _BuiltinBasis("pp", 4)
            prep_factors = []; povm_factors = []

            from ..construction import basis_build_vector

            v0 = basis_build_vector("0", basis1Q)
            v1 = basis_build_vector("1", basis1Q)

            # Historical use of TP for non-term-based cases?
            #  - seems we could remove this. FUTURE REMOVE?
            povmtyp = rtyp = "TP" if parameterization in \
                             ("CPTP", "H+S", "S", "H+S+A", "S+A", "H+D+A", "D+A", "D") \
                             else parameterization

            for i in range(n_qubits):
                prep_factors.append(
                    _sv.convert(_sv.StaticSPAMVec(v0), rtyp, basis1Q))
                povm_factors.append(
                    _povm.convert(_povm.UnconstrainedPOVM(([
                        ('0', _sv.StaticSPAMVec(v0)),
                        ('1', _sv.StaticSPAMVec(v1))])), povmtyp, basis1Q))

            prep_layers = [_sv.TensorProdSPAMVec('prep', prep_factors)]
            povm_layers = {'Mdefault': _povm.TensorProdPOVM(povm_factors)}

        elif spamtype == "lindblad":

            prepPure = _sv.ComputationalSPAMVec([0] * n_qubits, evotype)
            prepNoiseMap = _build_nqn_global_noise(
                qubitGraph, max_spam_weight, sparse, sim_type, parameterization, errcomp_type, printer - 1)
            prep_layers = [_sv.LindbladSPAMVec(prepPure, prepNoiseMap, "prep")]

            povmNoiseMap = _build_nqn_global_noise(
                qubitGraph, max_spam_weight, sparse, sim_type, parameterization, errcomp_type, printer - 1)
            povm_layers = {'Mdefault': _povm.LindbladPOVM(povmNoiseMap, None, "pp")}

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
                lbl.sslbls, qubitGraph, weight_maxhops_tuples,
                errcomp_type=errcomp_type, sparse=sparse, sim_type=sim_type,
                parameterization=parameterization, verbosity=printer - 1)

        def build_cloudkey_fn(lbl):
            cloud_maxhops = cloud_maxhops_1Q if len(lbl.sslbls) == 1 else cloud_maxhops_2Q
            cloud_inds = tuple(qubitGraph.radius(lbl.sslbls, cloud_maxhops))
            cloud_key = (tuple(lbl.sslbls), tuple(sorted(cloud_inds)))  # (sets are unhashable)
            return cloud_key

        return cls(n_qubits, gatedict, availability, qubit_labels, geometry,
                   global_idle_layer, prep_layers, povm_layers,
                   build_cloudnoise_fn, build_cloudkey_fn,
                   sim_type, evotype, errcomp_type,
                   add_idle_noise_to_all_gates, sparse, printer)

    def __init__(self, n_qubits, gatedict, availability=None,
                 qubit_labels=None, geometry="line",
                 global_idle_layer=None, prep_layers=None, povm_layers=None,
                 build_cloudnoise_fn=None, build_cloudkey_fn=None,
                 sim_type="map", evotype="densitymx", errcomp_type="gates",
                 add_idle_noise_to_all_gates=True, sparse=False, verbosity=0):
        """
        Create a n-qubit model using a low-weight and geometrically local
        error model with a common "global idle" operation.

        This constructor relies on factory functions being passed to it
        which generate the cloud-noise operators - noise thtat is specific
        to a gate but may act on a neighborhood or cloud around the gate's
        target qubits.

        Parameters
        ----------
        n_qubits : int
            The number of qubits

        gatedict : dict
            A dictionary (an `OrderedDict` if you care about insertion order) that
            associates with string-type gate names (e.g. `"Gx"`) :class:`LinearOperator`,
            `numpy.ndarray`, or :class:`OpFactory` objects. When the objects may act on
            fewer than the total number of qubits (determined by their dimension/shape) then
            they are repeatedly embedded into `n_qubits`-qubit gates as specified by their
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
            and/or strings.  Must be of length `n_qubits`.  If None, then the
            integers from 0 to `n_qubits-1` are used.

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

        sim_type : {"matrix","map","termorder:<N>"}
            The type of forward simulation (probability computation) to use for the
            returned :class:`Model`.  That is, how should the model compute
            operation sequence/circuit probabilities when requested.  `"matrix"` is better
            for small numbers of qubits, `"map"` is better for larger numbers. The
            `"termorder"` option is designed for even larger numbers.  Usually,
            the default of `"auto"` is what you want.

        evotype : {"densitymx","statevec","stabilizer","svterm","cterm"}
            The evolution type.

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

        sparse : bool, optional
            Whether embedded Lindblad-parameterized gates within the constructed
            `n_qubits`-qubit gates are sparse or not.

        verbosity : int, optional
            An integer >= 0 dictating how must output to send to stdout.
        """
        if qubit_labels is None:
            qubit_labels = tuple(range(n_qubits))
        if availability is None:
            availability = {}

        # Build gate dictionaries. A value of `gatedict` can be an array, a LinearOperator, or an OpFactory.
        # For later processing, we'll create mm_gatedict to contain each item as a ModelMember.  For cloud-
        # noise models, these gate operations should be *static* (no parameters) as they represent the target
        # operations and all noise (and parameters) are assumed to enter through the cloudnoise members.
        StaticDenseOp = _get_static_factory(sim_type, evotype)  # always a *gate*
        mm_gatedict = _collections.OrderedDict()  # static *target* ops as ModelMembers
        #REMOVE self.gatedict = _collections.OrderedDict()  # static *target* ops (unused) as numpy arrays
        for gn, gate in gatedict.items():
            if isinstance(gate, _op.LinearOperator):
                assert(gate.num_params() == 0), "Only *static* ideal operators are allowed in `gatedict`!"
                #REMOVE self.gatedict[gn] = gate.todense()
                if gate._evotype != evotype and isinstance(gate, _op.StaticDenseOp):
                    # special case: we'll convert static ops to the right evotype (convenient)
                    mm_gatedict[gn] = StaticDenseOp(gate, "pp")
                else:
                    mm_gatedict[gn] = gate
            elif isinstance(gate, _opfactory.OpFactory):
                assert(gate.num_params() == 0), "Only *static* ideal factories are allowed in `gatedict`!"
                # don't store factories in self.gatedict for now (no good dense representation)
                mm_gatedict[gn] = gate
            else:  # presumably a numpy array or something like it:
                #REMOVE self.gatedict[gn] = _np.array(gate)
                mm_gatedict[gn] = StaticDenseOp(gate, "pp")
            assert(mm_gatedict[gn]._evotype == evotype)

        #Set other members
        self.nQubits = n_qubits
        self.availability = availability
        self.qubit_labels = qubit_labels
        self.geometry = geometry
        #TODO REMOVE unneeded members
        #self.maxIdleWeight = maxIdleWeight
        #self.maxSpamWeight = maxSpamWeight
        #self.maxhops = maxhops
        #self.extraWeight1Hops = extraWeight1Hops
        #self.extraGateWeight = extraGateWeight
        self.sparse = sparse
        #self.parameterization = parameterization
        #self.spamtype = spamtype
        self.addIdleNoiseToAllGates = add_idle_noise_to_all_gates
        self.errcomp_type = errcomp_type

        #REMOVE
        ##Process "auto" sim_type
        #_, evotype = _gt.split_lindblad_paramtype(parameterization)
        #assert(evotype in ("densitymx", "svterm", "cterm")), "State-vector evolution types not allowed."
        #if sim_type == "auto":
        #    if evotype in ("svterm", "cterm"): sim_type = "termorder:1"
        #    else: sim_type = "map" if n_qubits > 2 else "matrix"

        assert(sim_type in ("matrix", "map") or sim_type.startswith("termorder") or sim_type.startswith("termgap"))

        qubit_dim = 2 if evotype in ('statevec', 'stabilizer') else 4
        if not isinstance(qubit_labels, _ld.StateSpaceLabels):  # allow user to specify a StateSpaceLabels object
            qubit_sslbls = _ld.StateSpaceLabels(qubit_labels, (qubit_dim,) * len(qubit_labels), evotype=evotype)
        else:
            qubit_sslbls = qubit_labels
            qubit_labels = [lbl for lbl in qubit_sslbls.labels[0] if qubit_sslbls.labeldims[lbl] == qubit_dim]
            #Only extract qubit labels from the first tensor-product block...

        if global_idle_layer is None:
            self.addIdleNoiseToAllGates = False  # there is no idle noise to add!
        lizardArgs = {'add_idle_noise': self.addIdleNoiseToAllGates,
                      'errcomp_type': errcomp_type, 'dense_rep': not sparse}
        super(CloudNoiseModel, self).__init__(qubit_sslbls, "pp", {}, CloudNoiseLayerLizard,
                                              lizardArgs, sim_type=sim_type, evotype=evotype)

        flags = {'auto_embed': False, 'match_parent_dim': False,
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

        printer = _VerbosityPrinter.build_printer(verbosity)
        geometry_name = "custom" if isinstance(geometry, _qgraph.QubitGraph) else geometry
        printer.log("Creating a %d-qubit local-noise %s model" % (n_qubits, geometry_name))

        if isinstance(geometry, _qgraph.QubitGraph):
            qubitGraph = geometry
        else:
            qubitGraph = _qgraph.QubitGraph.common_graph(n_qubits, geometry, directed=False,
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
        self.clouds = _collections.OrderedDict()

        #Get gates availability
        primitive_ops = []
        gates_and_avail = _collections.OrderedDict()
        for gateName, gate in mm_gatedict.items():  # gate is a static ModelMember (op or factory)
            gate_nQubits = int(round(_np.log2(gate.dim) / 2)) if (evotype in ("densitymx", "svterm", "cterm")) \
                else int(round(_np.log2(gate.dim)))  # evotype in ("statevec","stabilizer")

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

        ssAllQ = qubit_sslbls  # labls should also be node-names of qubitGraph
        EmbeddedDenseOp = _op.EmbeddedDenseOp if sim_type == "matrix" else _op.EmbeddedOp

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
                        ssAllQ, gate, dense=bool(sim_type == "matrix"), num_target_labels=inds[1])
                    # add any primitive ops for this embedding factory?
                else:
                    printer.log("Creating %dQ %s gate on qubits %s!!" % (len(inds), gn, inds))
                    assert(_Lbl(gn, inds) not in gatedict), \
                        ("Cloudnoise models do not accept primitive-op labels, e.g. %s, in `gatedict` as this dict "
                         "specfies the ideal target gates. Perhaps make the cloudnoise depend on the target qubits "
                         "of the %s gate?") % (str(_Lbl(gn, inds)), gn)

                    if gate_is_factory:
                        self.factories['layers'][_Lbl(gn, inds)] = _opfactory.EmbeddedOpFactory(
                            ssAllQ, inds, gate, dense=bool(sim_type == "matrix"))
                        # add any primitive ops for this factory?
                    else:
                        self.operation_blks['layers'][_Lbl(gn, inds)] = EmbeddedDenseOp(
                            ssAllQ, inds, gate)
                        primitive_ops.append(_Lbl(gn, inds))

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

        #REMOVE
        #if spamtype == "static" or maxSpamWeight == 0:
        #    if maxSpamWeight > 0:
        #        _warnings.warn(("`spamtype == 'static'` ignores the supplied "
        #                        "`maxSpamWeight=%d > 0`") % maxSpamWeight)
        #    self.prep_blks['layers'][_Lbl('rho0')] = _sv.ComputationalSPAMVec([0] * n_qubits, evotype)
        #    self.povm_blks['layers'][_Lbl('Mdefault')] = _povm.ComputationalBasisPOVM(n_qubits, evotype)
        #
        #elif spamtype == "tensorproduct":
        #
        #    _warnings.warn("`spamtype == 'tensorproduct'` is deprecated!")
        #    basis1Q = _BuiltinBasis("pp", 4)
        #    prep_factors = []; povm_factors = []
        #
        #    from ..construction import basis_build_vector
        #
        #    v0 = basis_build_vector("0", basis1Q)
        #    v1 = basis_build_vector("1", basis1Q)
        #
        #    # Historical use of TP for non-term-based cases?
        #    #  - seems we could remove this. FUTURE REMOVE?
        #    povmtyp = rtyp = "TP" if parameterization in \
        #                     ("CPTP", "H+S", "S", "H+S+A", "S+A", "H+D+A", "D+A", "D") \
        #                     else parameterization
        #
        #    for i in range(n_qubits):
        #        prep_factors.append(
        #            _sv.convert(_sv.StaticSPAMVec(v0), rtyp, basis1Q))
        #        povm_factors.append(
        #            _povm.convert(_povm.UnconstrainedPOVM(([
        #                ('0', _sv.StaticSPAMVec(v0)),
        #                ('1', _sv.StaticSPAMVec(v1))])), povmtyp, basis1Q))
        #
        #    # # Noise logic refactored from construction.nqnoiseconstruction.build_nqnoise_model
        #    # if prepNoise is not None:
        #    #     if isinstance(prepNoise,tuple): # use as (seed, strength)
        #    #         seed,strength = prepNoise
        #    #         rndm = _np.random.RandomState(seed)
        #    #         depolAmts = _np.abs(rndm.random_sample(n_qubits)*strength)
        #    #     else:
        #    #         depolAmts = prepNoise[0:n_qubits]
        #    #     for amt,vec in zip(depolAmts,prep_factors): vec.depolarize(amt)
        #
        #    # if povmNoise is not None:
        #    #     if isinstance(povmNoise,tuple): # use as (seed, strength)
        #    #         seed,strength = povmNoise
        #    #         rndm = _np.random.RandomState(seed)
        #    #         depolAmts = _np.abs(rndm.random_sample(n_qubits)*strength)
        #    #     else:
        #    #         depolAmts = povmNoise[0:n_qubits]
        #    #     for amt,povm in zip(depolAmts,povm_factors): povm.depolarize(amt)
        #
        #    self.prep_blks['layers'][_Lbl('rho0')] = _sv.TensorProdSPAMVec('prep', prep_factors)
        #    self.povm_blks['layers'][_Lbl('Mdefault')] = _povm.TensorProdPOVM(povm_factors)
        #
        #elif spamtype == "lindblad":
        #
        #    prepPure = _sv.ComputationalSPAMVec([0] * n_qubits, evotype)
        #    prepNoiseMap = _build_nqn_global_noise(
        #        qubitGraph, maxSpamWeight, sparse, sim_type, parameterization, errcomp_type, printer - 1)
        #    self.prep_blks['layers'][_Lbl('rho0')] = _sv.LindbladSPAMVec(prepPure, prepNoiseMap, "prep")
        #
        #    povmNoiseMap = _build_nqn_global_noise(
        #        qubitGraph, maxSpamWeight, sparse, sim_type, parameterization, errcomp_type, printer - 1)
        #    self.povm_blks['layers'][_Lbl('Mdefault')] = _povm.LindbladPOVM(povmNoiseMap, None, "pp")
        #
        #else:
        #    raise ValueError("Invalid `spamtype` argument: %s" % spamtype)

        self.set_primitive_op_labels(primitive_ops)
        self.set_primitive_prep_labels(tuple(self.prep_blks['layers'].keys()))
        self.set_primitive_povm_labels(tuple(self.povm_blks['layers'].keys()))
        #(no instruments)

        printer.log("DONE! - created Model with dim=%d and op-blks=" % self.dim)
        for op_blk_lbl, op_blk in self.operation_blks.items():
            printer.log("  %s: %s" % (op_blk_lbl, ', '.join(map(str, op_blk.keys()))))

    def get_clouds(self):
        """
        Returns the set of cloud-sets used when creating sequences which
        amplify the parameters of this model.
        """
        return self.clouds


def _get_lindblad_factory(sim_type, parameterization, errcomp_type):
    """ Returns a function that creates a Lindblad-type gate appropriate
        given the simulation type and parameterization """
    _, evotype = _gt.split_lindblad_paramtype(parameterization)
    if errcomp_type == "gates":
        if evotype == "densitymx":
            cls = _op.LindbladDenseOp if sim_type == "matrix" \
                else _op.LindbladOp
        elif evotype in ("svterm", "cterm"):
            assert(sim_type.startswith("termorder"))
            cls = _op.LindbladOp
        else:
            raise ValueError("Cannot create Lindblad gate factory for ", sim_type, parameterization)

        #Just call cls.from_operation_matrix with appropriate evotype
        def _f(op_matrix,  # unitaryPostfactor=None,
               proj_basis="pp", mx_basis="pp", relative=False):
            unitaryPostfactor = None  # we never use this in gate construction
            p = parameterization
            if relative:
                if parameterization == "CPTP": p = "GLND"
                elif "S" in parameterization: p = parameterization.replace("S", "s")
                elif "D" in parameterization: p = parameterization.replace("D", "d")
            return cls.from_operation_obj(op_matrix, p, unitaryPostfactor,
                                          proj_basis, mx_basis, truncate=True)
        return _f

    elif errcomp_type == "errorgens":
        def _f(error_gen,
               proj_basis="pp", mx_basis="pp", relative=False):
            p = parameterization
            if relative:
                if parameterization == "CPTP": p = "GLND"
                elif "S" in parameterization: p = parameterization.replace("S", "s")
                elif "D" in parameterization: p = parameterization.replace("D", "d")
            _, evotype, nonham_mode, param_mode = _op.LindbladOp.decomp_paramtype(p)
            return _op.LindbladErrorgen.from_error_generator(error_gen, proj_basis, proj_basis,
                                                             param_mode, nonham_mode, mx_basis,
                                                             truncate=True, evotype=evotype)
        return _f

    else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)


def _get_static_factory(sim_type, evotype):
    """ Returns a function that creates a static-type gate appropriate
        given the simulation and parameterization """
    if evotype == "densitymx":
        if sim_type == "matrix":
            return lambda g, b: _op.StaticDenseOp(g, evotype)
        elif sim_type == "map":
            return lambda g, b: _op.StaticDenseOp(g, evotype)  # TODO: create StaticGateMap?

    elif evotype in ("svterm", "cterm"):
        assert(sim_type.startswith("termorder") or sim_type.startswith("termgap"))

        def _f(op_matrix, mx_basis="pp"):
            return _op.LindbladOp.from_operation_matrix(
                None, op_matrix, None, None, mx_basis=mx_basis, evotype=evotype)
            # a LindbladDenseOp with None as ham_basis and nonham_basis => no parameters

        return _f
    raise ValueError("Cannot create Static gate factory for ", sim_type, evotype)


def _build_nqn_global_noise(qubit_graph, max_weight, sparse=False, sim_type="matrix",
                            parameterization="H+S", errcomp_type="gates", verbosity=0):
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

    sparse : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        gate are represented as sparse or dense matrices.  (This is determied by
        whether they are constructed using sparse basis matrices.)

    sim_type : {"matrix","map","termorder:<N>"}
        The type of forward simulation (probability computation) being used by
        the model this gate is destined for.  This affects what type of
        gate objects (e.g. `ComposedDenseOp` vs `ComposedOp`) are created.

    parameterization : str
        The type of parameterizaton for the constructed gate. E.g. "H+S",
        "H+S terms", "H+S clifford terms", "CPTP", etc.

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

    if errcomp_type == "gates":
        if sim_type == "matrix":
            Composed = _op.ComposedDenseOp
            Embedded = _op.EmbeddedDenseOp
        else:
            Composed = _op.ComposedOp
            Embedded = _op.EmbeddedOp
    elif errcomp_type == "errorgens":
        Composed = _op.ComposedErrorgen
        Embedded = _op.EmbeddedErrorgen
    else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
    Lindblad = _get_lindblad_factory(sim_type, parameterization, errcomp_type)
    #constructs a gate or errorgen based on value of errcomp_type

    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("*** Creating global idle ***")

    termops = []  # gates or error generators to compose
    qubit_labels = qubit_graph.get_node_names()
    qubit_dim = 4  # cloud noise models always use density matrices, so not '2' here
    ssAllQ = _ld.StateSpaceLabels(qubit_labels, (qubit_dim,) * len(qubit_labels))

    nQubits = qubit_graph.nqubits
    possible_err_qubit_inds = _np.arange(nQubits)
    nPossible = nQubits
    for wt in range(1, max_weight + 1):
        printer.log("Weight %d: %d possible qubits" % (wt, nPossible), 2)
        basisEl_Id = basis_product_matrix(_np.zeros(wt, _np.int64), sparse)
        if errcomp_type == "gates":
            wtNoErr = _sps.identity(4**wt, 'd', 'csr') if sparse else _np.identity(4**wt, 'd')
        elif errcomp_type == "errorgens":
            wtNoErr = _sps.csr_matrix((4**wt, 4**wt)) if sparse else _np.zeros((4**wt, 4**wt), 'd')
        else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
        wtBasis = _BuiltinBasis('pp', 4**wt, sparse=sparse)

        for err_qubit_inds in _itertools.combinations(possible_err_qubit_inds, wt):
            if len(err_qubit_inds) == 2 and not qubit_graph.is_directly_connected(err_qubit_inds[0], err_qubit_inds[1]):
                continue  # TO UPDATE - check whether all wt indices are a connected subgraph

            errbasis = [basisEl_Id]
            errbasis_lbls = ['I']
            for err_basis_inds in _iter_basis_inds(wt):
                error = _np.array(err_basis_inds, _np.int64)  # length == wt
                basisEl = basis_product_matrix(error, sparse)
                errbasis.append(basisEl)
                errbasis_lbls.append(''.join(["IXYZ"[i] for i in err_basis_inds]))

            printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_inds, len(errbasis)), 3)
            errbasis = _ExplicitBasis(errbasis, errbasis_lbls, real=True, sparse=sparse)
            termErr = Lindblad(wtNoErr, proj_basis=errbasis, mx_basis=wtBasis)

            err_qubit_global_inds = err_qubit_inds
            fullTermErr = Embedded(ssAllQ, [qubit_labels[i] for i in err_qubit_global_inds], termErr)
            assert(fullTermErr.num_params() == termErr.num_params())
            printer.log("Lindblad gate w/dim=%d and %d params -> embedded to gate w/dim=%d" %
                        (termErr.dim, termErr.num_params(), fullTermErr.dim))

            termops.append(fullTermErr)

    if errcomp_type == "gates":
        return Composed(termops)
    elif errcomp_type == "errorgens":
        errgen = Composed(termops)
        LindbladOp = _op.LindbladDenseOp if sim_type == "matrix" \
            else _op.LindbladOp
        return LindbladOp(None, errgen, dense_rep=not sparse)
    else: assert(False)


def _build_nqn_cloud_noise(target_qubit_inds, qubit_graph, weight_maxhops_tuples,
                           errcomp_type="gates", sparse=False, sim_type="matrix",
                           parameterization="H+S", verbosity=0):
    """
    Create an n-qubit gate that is a composition of:

    `target_op(target_qubits) -> idle_noise(all_qubits) -> loc_noise(local_qubits)`

    where `idle_noise` is given by the `idle_noise` argument and `loc_noise` is
    given by the rest of the arguments.  `loc_noise` can be implemented either
    by a single (n-qubit) embedded Lindblad gate with all relevant error
    generators, or as a composition of embedded single-error-term Lindblad gates
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

    sparse : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        gate are represented as sparse or dense matrices.  (This is determied by
        whether they are constructed using sparse basis matrices.)

    sim_type : {"matrix","map","termorder:<N>"}
        The type of forward simulation (probability computation) being used by
        the model this gate is destined for.  This affects what type of
        gate objects (e.g. `ComposedDenseOp` vs `ComposedOp`) are created.

    parameterization : str
        The type of parameterizaton for the constructed gate. E.g. "H+S",
        "H+S terms", "H+S clifford terms", "CPTP", etc.

    verbosity : int, optional
        An integer >= 0 dictating how must output to send to stdout.

    Returns
    -------
    LinearOperator
    """
    if sim_type == "matrix":
        ComposedDenseOp = _op.ComposedDenseOp
        EmbeddedDenseOp = _op.EmbeddedDenseOp
    else:
        ComposedDenseOp = _op.ComposedOp
        EmbeddedDenseOp = _op.EmbeddedOp

    if errcomp_type == "gates":
        Composed = ComposedDenseOp
        Embedded = EmbeddedDenseOp
    elif errcomp_type == "errorgens":
        Composed = _op.ComposedErrorgen
        Embedded = _op.EmbeddedErrorgen
    else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
    Lindblad = _get_lindblad_factory(sim_type, parameterization, errcomp_type)
    #constructs a gate or errorgen based on value of errcomp_type

    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("Creating local-noise error factor (%s)" % errcomp_type)

    # make a composed-gate of embedded single-basis-element Lindblad-gates or -errorgens,
    #  one for each specified error term

    loc_noise_termops = []  # list of gates to compose
    qubit_labels = qubit_graph.get_node_names()
    qubit_dim = 4  # cloud noise models always use density matrices, so not '2' here
    ssAllQ = _ld.StateSpaceLabels(qubit_labels, (qubit_dim,) * len(qubit_labels))

    for wt, maxHops in weight_maxhops_tuples:

        ## loc_noise_errinds = [] # list of basis indices for all local-error terms
        possible_err_qubit_inds = _np.array(qubit_graph.radius(target_qubit_inds, maxHops),
                                            _np.int64)  # we know node labels are integers
        nPossible = len(possible_err_qubit_inds)  # also == "nLocal" in this case
        basisEl_Id = basis_product_matrix(_np.zeros(wt, _np.int64), sparse)  # identity basis el

        if errcomp_type == "gates":
            wtNoErr = _sps.identity(4**wt, 'd', 'csr') if sparse else _np.identity(4**wt, 'd')
        elif errcomp_type == "errorgens":
            wtNoErr = _sps.csr_matrix((4**wt, 4**wt)) if sparse else _np.zeros((4**wt, 4**wt), 'd')
        else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
        wtBasis = _BuiltinBasis('pp', 4**wt, sparse=sparse)

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
                basisEl = basis_product_matrix(error, sparse)
                errbasis.append(basisEl)
                errbasis_lbls.append(''.join(["IXYZ"[i] for i in err_basis_inds]))

            err_qubit_global_inds = possible_err_qubit_inds[list(err_qubit_local_inds)]
            printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_global_inds, len(errbasis)), 4)
            errbasis = _ExplicitBasis(errbasis, errbasis_lbls, real=True, sparse=sparse)
            termErr = Lindblad(wtNoErr, proj_basis=errbasis, mx_basis=wtBasis, relative=True)

            fullTermErr = Embedded(ssAllQ, [qubit_labels[i] for i in err_qubit_global_inds], termErr)
            assert(fullTermErr.num_params() == termErr.num_params())
            printer.log("Lindblad gate w/dim=%d and %d params -> embedded to gate w/dim=%d" %
                        (termErr.dim, termErr.num_params(), fullTermErr.dim))

            loc_noise_termops.append(fullTermErr)

    fullCloudErr = Composed(loc_noise_termops)
    return fullCloudErr


class CloudNoiseLayerLizard(_ImplicitLayerLizard):
    """
    The layer lizard class for a :class:`CloudNoiseModel`, which
    creates layers by composing perfect target gates, global idle error,
    and local "cloud" errors.

    The value of `model._lizardArgs['errcomp_type']` determines which of two
    composition strategies are employed.  When the errcomp_type is `"gates"`,
    the errors on multiple gates in a single layer are composed as separate
    and subsequent processes.  Specifically, the layer operation has the form
    `Composed(target,idleErr,cloudErr)` where `target` is a composition of all
    the ideal gate operations in the layer, `idleErr` is idle error
    (`.operation_blks['layers']['globalIdle']`), and `cloudErr` is the
    composition (ordered as layer-label) of cloud-noise contributions, i.e. a
    map that acts as the product of exponentiated error-generator matrices.
    `"errorgens"`, on the other hand, means that layer operations have the form
    `Composed(target, error)` where `target` is as above and `error` results
    from composing the idle and cloud-noise error *generators*, i.e. a map that
    acts as the exponentiated sum of error generators (ordering is irrelevant in
    this case).
    """

    def get_prep(self, layerlbl):
        return self.prep_blks['layers'][layerlbl]  # prep_blks['layers'] are full prep ops

    def get_effect(self, layerlbl):
        if layerlbl in self.effect_blks['layers']:
            return self.effect_blks['layers'][layerlbl]  # effect_blks['layer'] are full effect ops
        else:
            # See if this effect label could correspond to a *marginalized* POVM, and
            # if so, create the marginalized POVM and add its effects to self.effect_blks['layers']
            if isinstance(layerlbl, _Lbl):  # this should always be the case...
                povmName = _gt.e_label_to_povm(layerlbl)
                if povmName in self.povm_blks['layers']:
                    # implicit creation of marginalized POVMs whereby an existing POVM name is used with sslbls that
                    # are not present in the stored POVM's label.
                    mpovm = _povm.MarginalizedPOVM(self.povm_blks['layers'][povmName],
                                                   self.model.state_space_labels, layerlbl.sslbls)  # cache in FUTURE?
                    mpovm_lbl = _Lbl(povmName, layerlbl.sslbls)
                    self.effect_blks['layers'].update(mpovm.simplify_effects(mpovm_lbl))
                    assert(layerlbl in self.effect_blks['layers']), "Failed to create marginalized effect!"
                    return self.effect_blks['layers'][layerlbl]
        raise KeyError("Could not build effect for '%s' label!" % str(layerlbl))

    def get_operation(self, layerlbl):
        dense = bool(self.model._sim_type == "matrix")  # whether dense matrix gates should be created

        if isinstance(layerlbl, _CircuitLabel):
            return self.get_circuitlabel_op(layerlbl, dense)

        add_idle_noise = self.model._lizardArgs['add_idle_noise']
        errcomp_type = self.model._lizardArgs['errcomp_type']
        dense_rep = self.model._lizardArgs['dense_rep'] or dense
        # can't create dense-rep LindbladOps with dense_rep=False

        Composed = _op.ComposedDenseOp if dense else _op.ComposedOp
        Lindblad = _op.LindbladDenseOp if dense else _op.LindbladOp
        Sum = _op.ComposedErrorgen
        #print("DB: CloudNoiseLayerLizard building gate %s for %s w/comp-type %s" %
        #      (('matrix' if dense else 'map'), str(oplabel), self.errcomp_type) )

        components = layerlbl.components
        if len(components) == 0:  # or layerlbl == 'Gi': # OLD: special case: 'Gi' acts as global idle!
            return self.simpleop_blks['layers']['globalIdle']  # idle!

        #Compose target operation from layer's component labels, which correspond
        # to the perfect (embedded) target ops in op_blks
        if len(components) > 1:
            targetOp = Composed([self.get_layer_component_targetop(l) for l in components], dim=self.model.dim,
                                evotype=self.model._evotype)
        else: targetOp = self.get_layer_component_targetop(components[0])
        ops_to_compose = [targetOp]

        if errcomp_type == "gates":
            if add_idle_noise: ops_to_compose.append(self.simpleop_blks['layers']['globalIdle'])
            component_cloudnoise_ops = self.get_layer_component_cloudnoises(components)
            if len(component_cloudnoise_ops) > 0:
                if len(component_cloudnoise_ops) > 1:
                    localErr = Composed(component_cloudnoise_ops,
                                        dim=self.model.dim, evotype=self.model._evotype)
                else:
                    localErr = component_cloudnoise_ops[0]
                ops_to_compose.append(localErr)

        elif errcomp_type == "errorgens":
            #We compose the target operations to create a
            # final target op, and compose this with a *singe* Lindblad gate which has as
            # its error generator the composition (sum) of all the factors' error gens.
            errorGens = [self.simpleop_blks['layers']['globalIdle'].errorgen] if add_idle_noise else []
            errorGens.extend(self.get_layer_component_cloudnoises(components))
            if len(errorGens) > 0:
                if len(errorGens) > 1:
                    error = Lindblad(None, Sum(errorGens, dim=self.model.dim,
                                               evotype=self.model._evotype),
                                     dense_rep=dense_rep)
                else:
                    error = Lindblad(None, errorGens[0], dense_rep=dense_rep)
                ops_to_compose.append(error)
        else:
            raise ValueError("Invalid errcomp_type in CloudNoiseLayerLizard: %s" % errcomp_type)

        ret = Composed(ops_to_compose, dim=self.model.dim,
                       evotype=self.model._evotype)
        self.model._init_virtual_obj(ret)  # so ret's gpindices get set
        return ret

    def get_layer_component_targetop(self, complbl):
        if isinstance(complbl, _CircuitLabel):
            raise NotImplementedError("Cloud noise models cannot simulate circuits with partial-layer subcircuits.")
            # In the FUTURE, could easily implement this for errcomp_type == "gates", but it's unclear what to
            #  do for the "errorgens" case - how do we gate an error generator of an entire (mulit-layer) sub-circuit?
            # Maybe we just need to expand the label and create a composition of those layers?
        elif complbl in self.simpleop_blks['layers']:
            return self.simpleop_blks['layers'][complbl]
        else:
            return _opfactory.op_from_factories(self.model.factories['layers'], complbl)

    def get_layer_component_cloudnoises(self, complbl_list):
        """
        Get any present cloudnoise ops from a list of components.  This function processes
        a list rather than an item because it's OK if some components don't have
        corresponding cloudnoise ops - we just leave those off.
        """
        ret = []
        for complbl in complbl_list:
            if complbl in self.simpleop_blks['cloudnoise']:
                ret.append(self.simpleop_blks['cloudnoise'][complbl])
            else:
                try:
                    ret.append(_opfactory.op_from_factories(self.model.factories['cloudnoise'], complbl))
                except KeyError: pass  # OK if cloudnoise doesn't exist (means no noise)
        return ret
