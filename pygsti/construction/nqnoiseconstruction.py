"""
Defines classes which represent gates, as well as supporting functions
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import itertools as _itertools
import numpy as _np
import scipy as _scipy
import scipy.sparse as _sps
import warnings as _warnings

from .. import objects as _objs
from ..tools import basistools as _bt
from ..tools import matrixtools as _mt
from ..tools import optools as _gt
from ..tools import slicetools as _slct
from ..tools import listtools as _lt
from ..tools import internalgates as _itgs
from ..tools import mpitools as _mpit
from ..tools.legacytools import deprecate as _deprecated_fn
from ..models import model as _mdl
from ..models import labeldicts as _ld
from ..models.labeldicts import StateSpaceLabels as _StateSpaceLabels
from ..models.cloudnoisemodel import CloudNoiseModel as _CloudNoiseModel
from ..modelmembers import operations as _op
from ..modelmembers import states as _state
from ..modelmembers import povms as _povm
from ..modelmembers.operations import opfactory as _opfactory

from ..forwardsims.matrixforwardsim import MatrixForwardSimulator as _MatrixFSim
from ..forwardsims.mapforwardsim import MapForwardSimulator as _MapFSim
from ..forwardsims.termforwardsim import TermForwardSimulator as _TermFSim

from ..objects import qubitgraph as _qgraph
from ..objects.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from ..objects.basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from ..objects.label import Label as _Lbl
from ..objects.polynomial import Polynomial as _Polynomial
from ..objects.resourceallocation import ResourceAllocation as _ResourceAllocation
from ..objects.circuitstructure import GermFiducialPairPlaquette as _GermFiducialPairPlaquette
from ..io import CircuitParser as _CircuitParser

from . import circuitconstruction as _gsc
from .modelconstruction import _basis_create_spam_vector as _basis_build_vector
from .modelconstruction import _parameterization_from_errgendict

RANK_TOL = 1e-9


@_deprecated_fn("This function is overly specific and will be removed soon.")
def _nparams_xycnot_cloudnoise_model(num_qubits, geometry="line", max_idle_weight=1, maxhops=0,
                                     extra_weight_1_hops=0, extra_gate_weight=0, require_connected=False,
                                     independent_1q_gates=True, zz_only=False, bidirectional_cnots=True, verbosity=0):
    """
    Compute the number of parameters in a particular :class:`CloudNoiseModel`.

    Returns the number of parameters in the :class:`CloudNoiseModel` containing
    X(pi/2), Y(pi/2) and CNOT gates using the specified arguments without
    actually constructing the model (useful for considering parameter-count
    scaling).

    Parameters
    ----------
    num_qubits : int
        The total number of qubits.

    geometry : {"line","ring","grid","torus"} or QubitGraph
        The type of connectivity among the qubits, specifying a
        graph used to define neighbor relationships.  Alternatively,
        a :class:`QubitGraph` object may be passed directly.

    max_idle_weight : int, optional
        The maximum-weight for errors on the global idle gate.

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

    require_connected : bool, optional
        If True, then high-weight errors only occur on connected (via `geometry`) qubits.
        For example in a line of qubits there would not be weight-2 errors on qubits 1 and 3.

    independent_1q_gates : bool, optional
        If True, 1Q gates on different qubits have separate (distinct) parameters.  If
        False, the 1Q gates of each type (e.g. an pi/2 X gate) for different qubits share
        the same set of parameters.

    zz_only : bool, optional
        If True, the only high-weight errors allowed are of "Z^n" type.

    bidirectional_cnots : bool
        Whether CNOT gates can be performed in either direction (and each direction should
        be treated as an indepedent gate)

    verbosity : int, optional
        An integer >= 0 dictating how much output to send to stdout.

    Returns
    -------
    int
    """
    # noise can be either a seed or a random array that is long enough to use

    printer = _VerbosityPrinter.create_printer(verbosity)
    printer.log("Computing parameters for a %d-qubit %s model" % (num_qubits, geometry))

    qubitGraph = _objs.QubitGraph.common_graph(num_qubits, geometry, directed=True, all_directions=True)
    #printer.log("Created qubit graph:\n"+str(qubitGraph))

    def idle_count_nparams(max_weight):
        """Parameter count of a `build_nqn_global_idle`-constructed gate"""
        ret = 0
        possible_err_qubit_inds = _np.arange(num_qubits)
        for wt in range(1, max_weight + 1):
            nErrTargetLocations = qubitGraph.connected_combos(possible_err_qubit_inds, wt)
            if zz_only and wt > 1: basisSizeWoutId = 1**wt  # ( == 1)
            else: basisSizeWoutId = 3**wt  # (X,Y,Z)^wt
            nErrParams = 2 * basisSizeWoutId  # H+S terms
            ret += nErrTargetLocations * nErrParams
        return ret

    def op_count_nparams(target_qubit_inds, weight_maxhops_tuples, debug=False):
        """Parameter count of a `build_nqn_composed_gate`-constructed gate"""
        ret = 0
        #Note: no contrib from idle noise (already parameterized)
        for wt, maxHops in weight_maxhops_tuples:
            possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops), _np.int64)
            if require_connected:
                nErrTargetLocations = qubitGraph.connected_combos(possible_err_qubit_inds, wt)
            else:
                nErrTargetLocations = _scipy.special.comb(len(possible_err_qubit_inds), wt)
            if zz_only and wt > 1: basisSizeWoutId = 1**wt  # ( == 1)
            else: basisSizeWoutId = 3**wt  # (X,Y,Z)^wt
            nErrParams = 2 * basisSizeWoutId  # H+S terms
            if debug:
                print(" -- wt%d, hops%d: inds=%s locs = %d, eparams=%d, total contrib = %d" %
                      (wt, maxHops, str(possible_err_qubit_inds), nErrTargetLocations,
                       nErrParams, nErrTargetLocations * nErrParams))
            ret += nErrTargetLocations * nErrParams
        return ret

    nParams = _collections.OrderedDict()

    printer.log("Creating Idle:")
    nParams[_Lbl('Gi')] = idle_count_nparams(max_idle_weight)

    #1Q gates: X(pi/2) & Y(pi/2) on each qubit
    weight_maxhops_tuples_1Q = [(1, maxhops + extra_weight_1_hops)] + \
                               [(1 + x, maxhops) for x in range(1, extra_gate_weight + 1)]

    if independent_1q_gates:
        for i in range(num_qubits):
            printer.log("Creating 1Q X(pi/2) and Y(pi/2) gates on qubit %d!!" % i)
            nParams[_Lbl("Gx", i)] = op_count_nparams((i,), weight_maxhops_tuples_1Q)
            nParams[_Lbl("Gy", i)] = op_count_nparams((i,), weight_maxhops_tuples_1Q)
    else:
        printer.log("Creating common 1Q X(pi/2) and Y(pi/2) gates")
        rep = int(num_qubits / 2)
        nParams[_Lbl("Gxrep")] = op_count_nparams((rep,), weight_maxhops_tuples_1Q)
        nParams[_Lbl("Gyrep")] = op_count_nparams((rep,), weight_maxhops_tuples_1Q)

    #2Q gates: CNOT gates along each graph edge
    weight_maxhops_tuples_2Q = [(1, maxhops + extra_weight_1_hops), (2, maxhops)] + \
                               [(2 + x, maxhops) for x in range(1, extra_gate_weight + 1)]
    seen_pairs = set()
    for i, j in qubitGraph.edges():  # note: all edges have i<j so "control" of CNOT is always lower index (arbitrary)
        if bidirectional_cnots is False:
            ordered_tup = (i, j) if i <= j else (j, i)
            if ordered_tup in seen_pairs: continue
            else: seen_pairs.add(ordered_tup)

        printer.log("Creating CNOT gate between qubits %d and %d!!" % (i, j))
        nParams[_Lbl("Gcnot", (i, j))] = op_count_nparams((i, j), weight_maxhops_tuples_2Q)

    #SPAM
    nPOVM_1Q = 4  # params for a single 1Q POVM
    nParams[_Lbl('rho0')] = 3 * num_qubits  # 3 b/c each component is TP
    nParams[_Lbl('Mdefault')] = nPOVM_1Q * num_qubits  # num_qubits 1Q-POVMs

    return nParams, sum(nParams.values())


def create_cloudnoise_model_from_hops_and_weights(
        num_qubits, gate_names, nonstd_gate_unitaries=None, custom_gates=None,
        availability=None, qubit_labels=None, geometry="line",
        max_idle_weight=1, max_spam_weight=1, maxhops=0,
        extra_weight_1_hops=0, extra_gate_weight=0,
        rough_noise=None, sparse_lindblad_basis=False, sparse_lindblad_reps=False,
        simulator="auto", parameterization="H+S",
        spamtype="lindblad", add_idle_noise_to_all_gates=True,
        errcomp_type="gates", independent_clouds=True,
        return_clouds=False, verbosity=0):  # , debug=False):
    """
    Create a n-qubit model using a low-weight and geometrically local error model.

    The model contains a common "global idle" operation which is applied within
    every circuit layer to model a potentially high-weight background noise process.

    This type of model is referred to as a "cloud noise" model because
    noise specific to a gate may act on a neighborhood or cloud around
    the gate's target qubits.  This type of model is generally useful
    for performing GST on a multi-qubit system, whereas local-noise
    models (:class:`LocalNoiseModel` objects, created by, e.g.,
    :function:`create_standard localnoise_model`) are more useful for
    representing static (non-parameterized) models.

    The following standard gate names may be specified as elements to
    `gate_names` without the need to supply their corresponding unitaries
    (as one must when calling the constructor directly):

    - 'Gi' : the 1Q idle operation
    - 'Gx','Gy','Gz' : 1Q pi/2 rotations
    - 'Gxpi','Gypi','Gzpi' : 1Q pi rotations
    - 'Gh' : Hadamard
    - 'Gp' : phase
    - 'Gcphase','Gcnot','Gswap' : standard 2Q gates

    Furthermore, if additional "non-standard" gates are needed,
    they are specified by their *unitary* gate action, even if
    the final model propagates density matrices (as opposed
    to state vectors).

    Parameters
    ----------
    num_qubits : int
        The total number of qubits.

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
        `gate_names` and whose values are lists of qubit-label-tuples.  Each
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
        edges, for 2Q gates of the graphy given by `geometry`.
        - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
        on any target qubits via an :class:`EmbeddingOpFactory` (uses less
        memory but slower than `"all-permutations"`.

        If a gate name (element of `gate_names`) is not present in `availability`,
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

    rough_noise : tuple or numpy.ndarray, optional
        If not None, noise to place on the gates, the state prep and the povm.
        This can either be a `(seed,strength)` 2-tuple, or a long enough numpy
        array (longer than what is needed is OK).  These values specify random
        `gate.from_vector` initialization for the model, and as such applies an
        often unstructured and unmeaningful type of noise.

    sparse_lindblad_basis : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        `num_qubits`-qubit gates use sparse bases or not.  When sparse, these
        Lindblad gates - especially those with high-weight action or errors - take
        less memory, but their simulation is slightly slower.  Usually it's fine to
        leave this as the default (False), except when considering unusually
        high-weight errors.

    sparse_lindblad_reps : bool, optional
        Whether created Lindblad operations use sparse (more memory efficient but
        slower action) or dense representations.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The simulator used to compute predicted probabilities for the
        resulting :class:`Model`.  Usually `"auto"` is fine, the default for
        each `evotype` is usually what you want.  Setting this to something
        else is expert-level tuning.

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

    add_idle_noise_to_all_gates : bool, optional
        Whether the global idle should be added as a factor following the
        ideal action of each of the non-idle gates.

    errcomp_type : {"gates","errorgens"}
        How errors are composed when creating layer operations in the returned
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

    return_clouds : bool, optional
        Whether to return a dictionary of "cloud" objects, used for constructing
        the circuits necessary for probing the returned Model's
        parameters.  Used primarily internally within pyGSTi.

    verbosity : int, optional
        An integer >= 0 dictating how much output to send to stdout.

    Returns
    -------
    Model
    """
    mdl = _CloudNoiseModel.from_hops_and_weights(
        num_qubits, gate_names, nonstd_gate_unitaries, custom_gates,
        availability, qubit_labels, geometry,
        max_idle_weight, max_spam_weight, maxhops,
        extra_weight_1_hops, extra_gate_weight,
        simulator, parameterization, spamtype,
        add_idle_noise_to_all_gates, errcomp_type,
        independent_clouds, sparse_lindblad_basis,
        sparse_lindblad_reps, verbosity)

    #Insert noise on everything using rough_noise (really shouldn't be used!)
    if rough_noise is not None:
        vec = mdl.to_vector()
        assert(spamtype == "lindblad"), "Can only apply rough noise when spamtype == lindblad"
        assert(_np.linalg.norm(vec) / len(vec) < 1e-6)  # make sure our base is zero
        if isinstance(rough_noise, tuple):  # use as (seed, strength)
            seed, strength = rough_noise
            rndm = _np.random.RandomState(seed)
            vec += _np.abs(rndm.random_sample(len(vec)) * strength)  # abs b/c some params need to be positive
        else:  # use as a vector
            vec += rough_noise[0:len(vec)]
        mdl.from_vector(vec)

    if return_clouds:
        #FUTURE - just return cloud *keys*? (operation label values are never used
        # downstream, but may still be useful for debugging, so keep for now)
        return mdl, mdl.clouds
    else:
        return mdl


def create_cloud_crosstalk_model(num_qubits, gate_names, nonstd_gate_unitaries={}, custom_gates={},
                                 depolarization_strengths={}, stochastic_error_probs={}, lindblad_error_coeffs={},
                                 depolarization_parameterization='depolarize', stochastic_parameterization='stochastic',
                                 lindblad_parameterization='auto', availability=None, qubit_labels=None,
                                 geometry="line", evotype="auto", simulator="auto", independent_gates=False,
                                 sparse_lindblad_basis=False, sparse_lindblad_reps=False, errcomp_type="errorgens",
                                 add_idle_noise_to_all_gates=True, verbosity=0):
    """
    Create a n-qubit model that may contain crosstalk errors.

    This function constructs a :class:`CloudNoiseModel` that may place noise on
    a gate that affects arbitrary qubits, i.e. qubits in addition to the target
    qubits of the gate.  These errors are specified uing a dictionary of error
    rates.

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

    error_rates : dict
        A dictionary whose keys are primitive-layer and gate labels (e.g.
        `("Gx",0)` or `"Gx"`) and whose values are "error-dictionaries"
        that determine the type and amount of error placed on each item.
        Error-dictionary keys are `(termType, basisLabel)` tuples, where
        `termType` can be `"H"` (Hamiltonian), `"S"` (Stochastic), or `"A"`
        (Affine), and `basisLabel` is a string of I, X, Y, or Z to describe a
        Pauli basis element appropriate for the gate (i.e. having the same
        number of letters as there are qubits in the gate).  For example, you
        could specify a 0.01-radian Z-rotation error and 0.05 rate of Pauli-
        stochastic X errors on a 1-qubit gate by using the error dictionary:
        `{('H','Z'): 0.01, ('S','X'): 0.05}`.  Furthermore, basis elements
        may be directed at specific qubits using a color followed by a comma-
        separated qubit-label-list.  For example, `('S',"XX:0,1")` would
        mean a weight-2 XX stochastic error on qubits 0 and 1, and this term
        could be placed in the error dictionary for a gate that is only
        supposed to target qubit 0, for instance.  In addition to the primitive
        label names, the special values `"prep"`, `"povm"`, and `"idle"` may be
        used as keys of `error_rates` to specify the error on the state
        preparation, measurement, and global idle, respectively.

    nonstd_gate_unitaries : dict, optional
        A dictionary of numpy arrays which specifies the unitary gate action
        of the gate names given by the dictionary's keys.  As an advanced
        behavior, a unitary-matrix-returning function which takes a single
        argument - a tuple of label arguments - may be given instead of a
        single matrix to create an operation *factory* which allows
        continuously-parameterized gates.  This function must also return
        an empty/dummy unitary when `None` is given as it's argument.

    custom_gates : dict, optional
        A dictionary that associates with gate labels
        :class:`LinearOperator`, :class:`OpFactory`, or `numpy.ndarray`
        objects.  These objects override any other behavior for constructing
        their designated operations (e.g. from `error_rates` or
        `nonstd_gate_unitaries`).  Note: currently these objects must
        be *static*, and keys of this dictionary must by strings - there's
        no way to specify the "cloudnoise" part of a gate via this dict
        yet, only the "target" part.

    depolarization_strengths : dict, optional
        A dictionary whose keys are gate names (e.g. `"Gx"`) and whose values
        are floats that specify the strength of uniform depolarization.

    stochastic_error_probs : dict, optional
        A dictionary whose keys are gate names (e.g. `"Gx"`) and whose values
        are tuples that specify Pauli-stochastic rates for each of the non-trivial
        Paulis (so a 3-tuple would be expected for a 1Q gate and a 15-tuple for a 2Q gate).

    lindblad_error_coeffs : dict, optional
        A dictionary whose keys are gate names (e.g. `"Gx"`) and whose values
        are dictionaries corresponding to the `lindblad_term_dict` kwarg taken
        by `LindbladErrorgen`. Keys are `(termType, basisLabel1, <basisLabel2>)`
        tuples, where `termType` can be `"H"` (Hamiltonian), `"S"`
        (Stochastic), or `"A"` (Affine).  Hamiltonian and Affine terms always
        have a single basis label (so key is a 2-tuple) whereas Stochastic
        tuples with 1 basis label indicate a *diagonal* term, and are the
        only types of terms allowed when `nonham_mode != "all"`.  Otherwise,
        Stochastic term tuples can include 2 basis labels to specify
        "off-diagonal" non-Hamiltonian Lindblad terms.  Basis labels can be
        strings or integers.  Values are complex coefficients.

    depolarization_parameterization : str of {"depolarize", "stochastic", or "lindblad"}
        Determines whether a DepolarizeOp, StochasticNoiseOp, or LindbladOp
        is used to parameterize the depolarization noise, respectively.
        When "depolarize" (the default), a DepolarizeOp is created with the strength given
        in `depolarization_strengths`. When "stochastic", the depolarization strength is split
        evenly among the stochastic channels of a StochasticOp. When "lindblad", the depolarization
        strength is split evenly among the coefficients of the stochastic error generators
        (which are exponentiated to form a LindbladOp with the "depol" parameterization).

    stochastic_parameterization : str of {"stochastic", or "lindblad"}
        Determines whether a StochasticNoiseOp or LindbladOp is used to parameterize the
        stochastic noise, respectively. When "stochastic", elements of `stochastic_error_probs`
        are used as coefficients in a linear combination of stochastic channels (the default).
        When "lindblad", the elements of `stochastic_error_probs` are coefficients of
        stochastic error generators (which are exponentiated to form a LindbladOp with the
        "cptp" parameterization).

    lindblad_parameterization : "auto" or a LindbladOp paramtype
        Determines the parameterization of the LindbladOp. When "auto" (the default), the parameterization
        is inferred from the types of error generators specified in the `lindblad_error_coeffs` dictionaries.
        When not "auto", the parameterization type is passed through to the LindbladOp.

    availability : dict, optional
        A dictionary whose keys are the same gate names as in
        `gate_names` and whose values are lists of qubit-label-tuples.  Each
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
        edges, for 2Q gates of the graphy given by `geometry`.
        - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
        on any target qubits via an :class:`EmbeddingOpFactory` (uses less
        memory but slower than `"all-permutations"`.

        If a gate name (element of `gate_names`) is not present in `availability`,
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

    evotype : {"auto","densitymx","statevec","stabilizer","svterm","cterm"}
        The evolution type.  If "auto" is specified, "densitymx" is used.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The simulator used to compute predicted probabilities for the
        resulting :class:`Model`.  Usually `"auto"` is fine, the default for
        each `evotype` is usually what you want.  Setting this to something
        else is expert-level tuning.

    independent_gates : bool, optional
        Whether gates are allowed independent cloud noise or not.  If False,
        then all gates with the same name (e.g. "Gx") will have the *same*
        noise.  If True, then gates with the same name acting on different
        qubits may have different noise.

    sparse_lindblad_basis : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        `num_qubits`-qubit gates use sparse bases or not.  When sparse, these
        Lindblad gates - especially those with high-weight action or errors - take
        less memory, but their simulation is slightly slower.  Usually it's fine to
        leave this as the default (False), except when considering unusually
        high-weight errors.

    sparse_lindblad_reps : bool, optional
        Whether created Lindblad operations use sparse (more memory efficient but
        slower action) or dense representations.

    errcomp_type : {"gates","errorgens"}
        How errors are composed when creating layer operations in the returned
        model.  `"gates"` means that the errors on multiple gates in a single
        layer are composed as separate and subsequent processes.  `"errorgens"`
        means that layer operations have the form `Composed(target, error)`
        where `target` is as above and `error` results from composing the idle
        and cloud-noise error *generators*, i.e. a map that acts as the
        exponentiated sum of error generators (ordering is irrelevant in
        this case).

    add_idle_noise_to_all_gates : bool, optional
        Whether the global idle should be added as a factor following the
        ideal action of each of the non-idle gates when constructing layer
        operations.

    verbosity : int, optional
        An integer >= 0 dictating how much output to send to stdout.

    Returns
    -------
    CloudNoiseModel
    """
    # E.g. error_rates could == {'Gx': {('H','X'): 0.1, ('S','Y'): 0.2} } # Lindblad, b/c val is dict
    #                        or {'Gx': 0.1 } # Depolarization b/c val is a float
    #                        or {'Gx': (0.1,0.2,0.2) } # Pauli-Stochastic b/c val is a tuple
    # (same as those of a crosstalk-free model) PLUS additional ones which specify which
    # qubits the error operates (not necessarily the target qubits of the gate in question)
    # for example: { 'Gx:Q0': { ('H','X:Q1'): 0.01, ('S','XX:Q0,Q1'): 0.01} }

    #NOTE: to have "independent_gates=False" and specify error rates for "Gx" vs "Gx:Q0", we
    # need to have some ability to stencil a gate's cloud based on different target-qubits in
    # the qubit graph.
    printer = _VerbosityPrinter.create_printer(verbosity)

    if depolarization_strengths or stochastic_error_probs:
        raise NotImplementedError("Cloud noise models can currently only be built with `lindbad_error_coeffs`")

    if evotype == "auto":
        evotype = "densitymx"  # FUTURE: do something more sophisticated?

    if qubit_labels is None:
        qubit_labels = tuple(range(num_qubits))

    qubit_dim = 2 if evotype in ('statevec', 'stabilizer') else 4
    if not isinstance(qubit_labels, _ld.StateSpaceLabels):  # allow user to specify a StateSpaceLabels object
        all_sslbls = _ld.StateSpaceLabels(qubit_labels, (qubit_dim,) * len(qubit_labels), evotype=evotype)
    else:
        all_sslbls = qubit_labels
        qubit_labels = [lbl for lbl in all_sslbls.labels[0] if all_sslbls.labeldims[lbl] == qubit_dim]
        #Only extract qubit labels from the first tensor-product block...

    if isinstance(geometry, _qgraph.QubitGraph):
        qubitGraph = geometry
    else:
        qubitGraph = _qgraph.QubitGraph.common_graph(num_qubits, geometry, directed=True,
                                                     qubit_labels=qubit_labels, all_directions=True)
        printer.log("Created qubit graph:\n" + str(qubitGraph))

    nQubit_dim = 2**num_qubits if evotype in ('statevec', 'stabilizer') else 4**num_qubits

    orig_lindblad_error_coeffs = lindblad_error_coeffs.copy()
    cparser = _CircuitParser()
    cparser.lookup = None  # lookup - functionality removed as it wasn't used
    for k, v in orig_lindblad_error_coeffs.items():
        if isinstance(k, str) and ":" in k:  # then parse this to get a label, allowing, e.g. "Gx:0"
            lbls, _, _ = cparser.parse(k)
            assert(len(lbls) == 1), "Only single primitive-gate labels allowed as keys! (not %s)" % str(k)
            assert(all([sslbl in qubitGraph.node_names for sslbl in lbls[0].sslbls])), \
                "One or more invalid qubit names in: %s" % k
            del lindblad_error_coeffs[k]
            lindblad_error_coeffs[lbls[0]] = v
        elif isinstance(k, _Lbl):
            if k.sslbls is not None:
                assert(all([sslbl in qubitGraph.node_names for sslbl in k.sslbls])), \
                    "One or more invalid qubit names in the label: %s" % str(k)

    def _map_stencil_sslbls(stencil_sslbls, target_lbls):  # deals with graph directions
        ret = [qubitGraph.resolve_relative_nodelabel(s, target_lbls) for s in stencil_sslbls]
        if any([x is None for x in ret]): return None  # signals there is a non-present dirs, e.g. end of chain
        return ret

    def create_error(target_labels, errs=None, stencil=None, return_what="auto"):  # err = an error rates dict
        """
        Create an error generator or error superoperator based on the error dictionary
        `errs`.  This function is used to construct error for SPAM and gate layer operations.

        Parameters
        ----------
        target_labels : tuple
            The target labels of the gate/primitive layer we're constructing an
            error for.  This is needed for knowing the size of the target op and
            for using "@" syntax within stencils.

        errs : dict
            A error-dictionary specifying what types of errors we need to construct.

        stencil : None or OrderedDict
            Instead of specifying `errs`, one can specify `stencil` to tell us how
            and *with what* to construct an error -- `stencil` will contain keys
            that are tuples of "stencil labels" and values which are error generators,
            specifying errors that occur on certain "real" qubits by mapping the
            stencil qubits to real qubits using `target_labels` as an anchor.

        return_what : {"auto", "stencil", "errmap", "errgen"}, optional
            What type of object should be returned.  "auto" causes either an
            "errmap" (a superoperator) or "errgen" (an error generator) to
            be selected based on the outside-scope value of `errcomp_type`.

        Returns
        -------
        LinearOperator or OrderedDict
            The former in the "errmap" and "errgen" cases, the latter in the
            "stencil" case.
        """
        target_nQubits = len(target_labels)

        if return_what == "auto":  # then just base return type on errcomp_type
            return_what == "errgen" if errcomp_type == "errorgens" else "errmap"

        assert(stencil is None or errs is None), "Cannot specify both `errs` and `stencil`!"

        if errs is None:
            if stencil is None:
                if return_what == "stencil":
                    new_stencil = _collections.OrderedDict()  # return an empty stencil
                    return new_stencil
                errgen = _op.ComposedErrorgen([], nQubit_dim, evotype)
            else:
                # stencil is valid: apply it to create errgen
                embedded_errgens = []
                for stencil_sslbls, lind_errgen in stencil.items():
                    # Note: stencil_sslbls should contain directions like "up" or integer indices of target qubits.
                    error_sslbls = _map_stencil_sslbls(stencil_sslbls, target_labels)  # deals with graph directions
                    if error_sslbls is None: continue  # signals not all direction were present => skip this term
                    op_to_embed = lind_errgen.copy() if independent_gates else lind_errgen  # copy for independent gates
                    #REMOVE print("DB: Applying stencil: ",all_sslbls, error_sslbls,op_to_embed.dim)
                    embedded_errgen = _op.EmbeddedErrorgen(all_sslbls, error_sslbls, op_to_embed)
                    embedded_errgens.append(embedded_errgen)
                errgen = _op.ComposedErrorgen(embedded_errgens, nQubit_dim, evotype)
        else:
            #We need to build a stencil (which may contain QubitGraph directions) or an effective stencil
            assert(stencil is None)  # checked by above assert too

            # distinct sets of qubits upon which a single (high-weight) error term acts:
            distinct_errorqubits = _collections.OrderedDict()
            if isinstance(errs, dict):  # either for creating a stencil or an error
                for nm, val in errs.items():
                    #REMOVE print("DB: Processing: ",nm, val)
                    if isinstance(nm, str): nm = (nm[0], nm[1:])  # e.g. "HXX" => ('H','XX')
                    err_typ, basisEls = nm[0], nm[1:]
                    sslbls = None
                    local_nm = [err_typ]
                    for bel in basisEls:  # e.g. bel could be "X:Q0" or "XX:Q0,Q1"
                        #REMOVE print("Basis el: ",bel)
                        # OR "X:<n>" where n indexes a target qubit or "X:<dir>" where dir indicates
                        # a graph *direction*, e.g. "up"
                        if ':' in bel:
                            bel_name, bel_sslbls = bel.split(':')  # should have form <name>:<comma-separated-sslbls>
                            bel_sslbls = bel_sslbls.split(',')  # e.g. ('Q0','Q1')
                            integerized_sslbls = []
                            for ssl in bel_sslbls:
                                try: integerized_sslbls.append(int(ssl))
                                except: integerized_sslbls.append(ssl)
                            bel_sslbls = tuple(integerized_sslbls)
                        else:
                            bel_name = bel
                            bel_sslbls = target_labels
                        #REMOVE print("DB: Nm + sslbls: ",bel_name,bel_sslbls)

                        if sslbls is None:
                            sslbls = bel_sslbls
                        else:
                            #Note: sslbls should always be the same if there are multiple basisEls,
                            #  i.e for nm == ('S',bel1,bel2)
                            assert(sslbls == bel_sslbls), \
                                "All basis elements of the same error term must operate on the *same* state!"
                        local_nm.append(bel_name)  # drop the state space labels, e.g. "XY:Q0,Q1" => "XY"

                    # keep track of errors by the qubits they act on, as only each such
                    # set will have it's own LindbladErrorgen
                    sslbls = tuple(sorted(sslbls))
                    local_nm = tuple(local_nm)  # so it's hashable
                    if sslbls not in distinct_errorqubits:
                        distinct_errorqubits[sslbls] = _collections.OrderedDict()
                    if local_nm in distinct_errorqubits[sslbls]:
                        distinct_errorqubits[sslbls][local_nm] += val
                    else:
                        distinct_errorqubits[sslbls][local_nm] = val

            elif isinstance(errs, float):  # depolarization, action on only target qubits
                sslbls = tuple(range(target_nQubits)) if return_what == "stencil" else target_labels
                # Note: we use relative target indices in a stencil
                basis = _BuiltinBasis('pp', 4**target_nQubits, sparse=sparse_lindblad_basis)  # assume Pauli basis?
                distinct_errorqubits[sslbls] = _collections.OrderedDict()
                perPauliRate = errs / len(basis.labels)
                for bl in basis.labels:
                    distinct_errorqubits[sslbls][('S', bl)] = perPauliRate
            else:
                raise ValueError("Invalid `error_rates` value: %s (type %s)" % (str(errs), type(errs)))

            new_stencil = _collections.OrderedDict()
            for error_sslbls, local_errs_for_these_sslbls in distinct_errorqubits.items():
                local_nQubits = len(error_sslbls)  # weight of this group of errors which act on the same qubits
                local_dim = 4**local_nQubits
                basis = _BuiltinBasis('pp', local_dim, sparse=sparse_lindblad_basis)
                # assume we're always given els in a Pauli basis?

                #Sanity check to catch user errors that would be hard to interpret if they get caught further down
                for nm in local_errs_for_these_sslbls:
                    for bel in nm[1:]:  # bel should be a *local* (bare) basis el name, e.g. "XX" but not "XX:Q0,Q1"
                        if bel not in basis.labels:
                            raise ValueError("In %s: invalid basis element label `%s` where one of {%s} was expected" %
                                             (str(errs), str(bel), ', '.join(basis.labels)))

                parameterization = _parameterization_from_errgendict(local_errs_for_these_sslbls)
                #REMOVE print("DB: Param from ", local_errs_for_these_sslbls, " = ",parameterization)
                _, _, nonham_mode, param_mode = _op.ExpErrorgenOp.decomp_paramtype(parameterization)      ## TODO; check if this exists??
                lind_errgen = _op.LindbladErrorgen(local_dim, local_errs_for_these_sslbls, basis, param_mode,
                                                   nonham_mode, truncate=False, mx_basis="pp", evotype=evotype)
                #REMOVE print("DB: Adding to stencil: ",error_sslbls,lind_errgen.dim,local_dim)
                new_stencil[error_sslbls] = lind_errgen

            if return_what == "stencil":  # then we just return the stencil, not the error map or generator
                return new_stencil

            #Use stencil to create error map or generator.  Here `new_stencil` is not a "true" stencil
            # in that it should contain only absolute labels (it wasn't created in stencil="create" mode)
            embedded_errgens = []
            for error_sslbls, lind_errgen in new_stencil.items():
                #Then use the stencils for these steps later (if independent errgens is False especially?)
                #REMOVE print("DB: Creating from stencil: ",all_sslbls, error_sslbls)
                embedded_errgen = _op.EmbeddedErrorgen(all_sslbls, error_sslbls, lind_errgen)
                embedded_errgens.append(embedded_errgen)
            errgen = _op.ComposedErrorgen(embedded_errgens, nQubit_dim, evotype)

        #If we get here, we've created errgen, which we either return or package into a map:
        if return_what == "errmap":
            return _op.LindbladOp(None, errgen, dense_rep=not sparse_lindblad_reps)
        else:
            return errgen

    #Process "auto" simulator
    _, evotype = _gt.split_lindblad_paramtype(lindblad_parameterization)  # what about "auto" parameterization?
    assert(evotype in ("densitymx", "svterm", "cterm")), "State-vector evolution types not allowed."
    if simulator == "auto":
        if evotype in ("svterm", "cterm"): simulator = _TermFSim()
        else: simulator = _MapFSim() if num_qubits > 2 else _MatrixFSim()

    #Global Idle
    if 'idle' in lindblad_error_coeffs:
        printer.log("Creating Idle:")
        global_idle_layer = create_error(qubit_labels, lindblad_error_coeffs['idle'], return_what="errmap")
    else:
        global_idle_layer = None

    #SPAM
    if 'prep' in lindblad_error_coeffs:
        prepPure = _state.ComputationalBasisState([0] * num_qubits, evotype)
        prepNoiseMap = create_error(qubit_labels, lindblad_error_coeffs['prep'], return_what="errmap")
        prep_layers = [_state.ComposedState(prepPure, prepNoiseMap)]
    else:
        prep_layers = [_state.ComputationalBasisState([0] * num_qubits, evotype)]

    if 'povm' in lindblad_error_coeffs:
        povmNoiseMap = create_error(qubit_labels, lindblad_error_coeffs['povm'], return_what="errmap")
        povm_layers = [_povm.ExpErrorgenPOVM(povmNoiseMap, None, "pp")]
    else:
        povm_layers = [_povm.ComputationalBasisPOVM(num_qubits, evotype)]

    stencils = _collections.OrderedDict()

    def build_cloudnoise_fn(lbl):
        # lbl will be for a particular gate and target qubits.  If we have error rates for this specific gate
        # and target qubits (i.e this primitive layer op) then we should build it directly (and independently,
        # regardless of the value of `independent_gates`) using these rates.  Otherwise, if we have a stencil
        # for this gate, then we should use it to construct the output, using a copy when gates are independent
        # and a reference to the *same* stencil operations when `independent_gates==False`.
        if lbl in lindblad_error_coeffs:
            # specific instructions for this primitive layer
            return create_error(lbl.sslbls, errs=lindblad_error_coeffs[lbl])
        elif lbl.name in stencils:
            return create_error(lbl.sslbls, stencil=stencils[lbl.name])  # use existing stencil
        elif lbl.name in lindblad_error_coeffs:
            stencils[lbl.name] = create_error(lbl.sslbls, lindblad_error_coeffs[lbl.name],
                                              return_what='stencil')  # create stencil
            return create_error(lbl.sslbls, stencil=stencils[lbl.name])  # and then use it
        else:
            return create_error(lbl, None)

    def build_cloudkey_fn(lbl):
        #FUTURE: Get a list of all the qubit labels `lbl`'s cloudnoise error touches and form this into a key
        # For now, we just punt and return a key based on the target labels
        cloud_key = tuple(lbl.sslbls)
        return cloud_key

    # gate_names => gatedict
    if custom_gates is None: custom_gates = {}
    if nonstd_gate_unitaries is None: nonstd_gate_unitaries = {}
    std_unitaries = _itgs.standard_gatename_unitaries()

    gatedict = _collections.OrderedDict()
    for name in gate_names:
        if name in custom_gates:
            gatedict[name] = custom_gates[name]
        else:
            U = nonstd_gate_unitaries.get(name, std_unitaries.get(name, None))
            if U is None: raise KeyError("'%s' gate unitary needs to be provided by `nonstd_gate_unitaries` arg" % name)
            if callable(U):  # then assume a function: args -> unitary
                U0 = U(None)  # U fns must return a sample unitary when passed None to get size.
                gatedict[name] = _opfactory.UnitaryOpFactory(U, U0.shape[0], evotype=evotype)
            else:
                gatedict[name] = _bt.change_basis(_gt.unitary_to_process_mx(U), "std", "pp")
                # assume evotype is a densitymx or term type

    #Add anything from custom_gates directly if it wasn't added already
    for lbl, gate in custom_gates.items():
        if lbl not in gate_names: gatedict[lbl] = gate

    return _CloudNoiseModel(num_qubits, gatedict, availability, qubit_labels, geometry,
                            global_idle_layer, prep_layers, povm_layers,
                            build_cloudnoise_fn, build_cloudkey_fn,
                            simulator, evotype, errcomp_type,
                            add_idle_noise_to_all_gates, sparse_lindblad_reps, printer)


# -----------------------------------------------------------------------------------
#  nqnoise gate sequence construction methods
# -----------------------------------------------------------------------------------

#Note: these methods assume a Model with:
# Gx and Gy gates on each qubit that are pi/2 rotations
# a prep labeled "rho0"
# a povm labeled "Mdefault" - so effects labeled "Mdefault_N" for N=0->2^nQubits-1


def _onqubit(s, i_qubit):
    """ Takes `s`, a tuple of gate *names* and creates a Circuit
        where those names act on the `i_qubit`-th qubit """
    return _objs.Circuit([_Lbl(nm, i_qubit) for nm in s], line_labels=(i_qubit,))  # set line labels in case s is empty


def _find_amped_polynomials_for_syntheticidle(qubit_filter, idle_str, model, single_q_fiducials=None,
                                              prep_lbl=None, effect_lbls=None, init_j=None, init_j_rank=None,
                                              wrt_params=None, algorithm="greedy", require_all_amped=True,
                                              idt_pauli_dicts=None, comm=None, verbosity=0):
    """
    Find fiducial pairs which amplify the parameters of a synthetic idle gate.

    This routine is primarily used internally within higher-level n-qubit
    sequence selection routines.

    Parameters
    ----------
    qubit_filter : list
        A list specifying which qubits fiducial pairs should be placed upon.
        Typically this is a subset of all the qubits, as the synthetic idle
        is composed of nontrivial gates acting on a localized set of qubits
        and noise/errors are localized around these.

    idle_str : Circuit
        The circuit specifying the idle operation to consider.  This may
        just be a single idle gate, or it could be multiple non-idle gates
        which together act as an idle.

    model : Model
        The model used to compute the polynomial expressions of probabilities
        to first-order.  Thus, this model should always have (simulation)
        type "termorder".

    single_q_fiducials : list, optional
        A list of gate-name tuples (e.g. `('Gx',)`) which specify a set of single-
        qubit fiducials to use when trying to amplify gate parameters.  Note that
        no qubit "state-space" label is required here (i.e. *not* `(('Gx',1),)`);
        the tuples just contain single-qubit gate *names*.  If None, then
        `[(), ('Gx',), ('Gy',)]` is used by default.

    prep_lbl : Label, optional
        The state preparation label to use.  If None, then the first (and
        usually the only) state prep label of `model` is used, so it's
        usually fine to leave this as None.

    effect_lbls : list, optional
        The list of POVM effect labels to use, as a list of `Label` objects.
        These are *simplified* POVM effect labels, so something like "Mdefault_0",
        and if None the default is all the effect labels of the first POVM of
        `model`, which is usually what you want.

    init_j : numpy.ndarray, optional
        An initial Jacobian giving the derivatives of some other polynomials
        with respect to the same `wrt_params` that this function is called with.
        This acts as a starting point, and essentially informs the fiducial-pair
        selection algorithm that some parameters (or linear combos of them) are
        *already* amplified (e.g. by some other germ that's already been
        selected) and for which fiducial pairs are not needed.

    init_j_rank : int, optional
        The rank of `init_j`.  The function could compute this from `init_j`
        but in practice one usually has the rank of `init_j` lying around and
        so this saves a call to `np.linalg.matrix_rank`.

    wrt_params : slice, optional
        The parameters to consider for amplification.  (This function seeks
        fiducial pairs that amplify these parameters.)  If None, then pairs
        which amplify all of `model`'s parameters are searched for.

    algorithm : {"greedy","sequential"}
        Which algorithm is used internally to find fiducial pairs.  "greedy"
        will give smaller sets of fiducial pairs (better) but takes longer.
        Usually it's worth the wait and you should use the default ("greedy").

    require_all_amped : bool, optional
        If True and AssertionError is raised when fewer than all of the
        requested parameters (in `wrt_params`) are amplifed by the final set of
        fiducial pairs.

    idt_pauli_dicts : tuple, optional
        A (prepDict,measDict) tuple of dicts that maps a 1-qubit Pauli basis
        string (e.g. 'X' or '-Y') to a sequence of gate *names*.  If given,
        the idle-germ fiducial pairs chosen by this function are restricted
        to those where either 1) each qubit is prepared and measured in the
        same basis or 2) each qubits is prepared and measured in different
        bases (note: '-X' and 'X" are considered the *same* basis).  This
        restriction makes the resulting sequences more like the "standard"
        ones of idle tomography, and thereby easier to interpret.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.

    Returns
    -------
    J : numpy.ndarray
        The final jacobian with rows equal to the number of chosen amplified
        polynomials (note there is one row per fiducial pair *including* the
        outcome - so there will be two different rows for two different
        outcomes) and one column for each parameter specified by `wrt_params`.
    Jrank : int
        The rank of the jacobian `J`, equal to the number of amplified
        parameters (at most the number requested).
    fidpair_lists : list
        The selected fiducial pairs, each in "gatename-fidpair-list" format.
        Elements of `fidpair_lists` are themselves lists, all of length=#qubits.
        Each element of these lists is a (prep1Qnames, meas1Qnames) 2-tuple
        specifying the 1-qubit gates (by *name* only) on the corresponding qubit.
        For example, the single fiducial pair prep=Gx:1Gy:2, meas=Gx:0Gy:0 in a
        3-qubit system would have `fidpair_lists` equal to:
        `[ [ [(),('Gx','Gy')], [('Gx',), ()   ], [('Gy',), ()   ] ] ]`
        `    < Q0 prep,meas >, < Q1 prep,meas >, < Q2 prep,meas >`
    """
    #Note: "useful" fiducial pairs are identified by looking at the rank of a
    # Jacobian matrix.  Each row of this Jacobian is the derivative of the
    # "amplified polynomial" - the L=1 polynomial for a fiducial pair (i.e.
    # pr_poly(F1*(germ)*F2) ) minus the L=0 polynomial (i.e. pr_poly(F1*F2) ).
    # When the model only gives probability polynomials to first order in
    # the error rates this gives the L-dependent and hence amplified part
    # of the polynomial expression for the probability of F1*(germ^L)*F2.
    # This derivative of an amplified polynomial, taken with respect to
    # all the parameters we care about (i.e. wrt_params) would ideally be
    # kept as a polynomial and the "rank" of J would be the number of
    # linearly independent polynomials within the rows of J (each poly
    # would be a vector in the space of polynomials).  We currently take
    # a cheap/HACK way out and evaluate the derivative-polynomial at a
    # random dummy value which should yield linearly dependent vectors
    # in R^n whenever the polynomials are linearly indepdendent - then
    # we can use the usual scipy/numpy routines for computing a matrix
    # rank, etc.

    # Assert that model uses termorder, as doing L1-L0 to extract the "amplified" part
    # relies on only expanding to *first* order.
    assert(isinstance(model.sim, _TermFSim) and model.sim.max_order == 1), \
        '`model` must use a 1-st order Term-type forward simulator!'

    printer = _VerbosityPrinter.create_printer(verbosity, comm)
    polynomial_vindices_per_int = _Polynomial._vindices_per_int(model.num_params)
    resource_alloc = _ResourceAllocation()  # don't use comm here, since it's not used for prs_as_polynomials

    if prep_lbl is None:
        prep_lbl = model._default_primitive_prep_layer_lbl()
    if effect_lbls is None:
        povmLbl = model._default_primitive_povm_layer_lbl(sslbls=None)
        effect_lbls = [_Lbl("%s_%s" % (povmLbl, l))
                       for l in model._effect_labels_for_povm(povmLbl)]
    if single_q_fiducials is None:
        # TODO: assert model has Gx and Gy gates?
        single_q_fiducials = [(), ('Gx',), ('Gy',)]  # ('Gx','Gx')

    #dummy = 0.05*_np.ones(model.num_params,'d') # for evaluating derivs...
    #dummy = 0.05*_np.arange(1,model.num_params+1) # for evaluating derivs...
    #dummy = 0.05*_np.random.random(model.num_params)
    dummy = 5.0 * _np.random.random(model.num_params) + 0.5 * _np.ones(model.num_params, 'd')
    # expect terms to be either coeff*x or coeff*x^2 - (b/c of latter case don't eval at zero)

    print("DB gpindices = ")
    model._print_gpindices()

    #amped_polys = []
    selected_gatename_fidpair_lists = []
    if wrt_params is None: wrt_params = slice(0, model.num_params)
    Np = _slct.length(wrt_params)
    if init_j is None:
        J = _np.empty((0, Np), 'complex'); Jrank = 0
    else:
        J = init_j; Jrank = init_j_rank

    if algorithm == "greedy":
        Jrows = _np.empty((len(effect_lbls), Np), 'complex')

    #Outer iteration
    while Jrank < Np:

        if algorithm == "sequential":
            printer.log("Sequential _find_amped_polynomials_for_syntheticidle started. Target rank=%d" % Np)
            assert(comm is None), "No MPI support for algorithm='sequential' case!"

        elif algorithm == "greedy":
            maxRankInc = 0
            bestJrows = None
            printer.log("Greedy _find_amped_polynomials_for_syntheticidle started. Target rank=%d" % Np)

        else: raise ValueError("Invalid `algorithm` argument: %s" % algorithm)

        # loop over all possible (remaining) fiducial pairs
        nQubits = len(qubit_filter)
        loc_Indices, _, _ = _mpit.distribute_indices(
            list(range(len(single_q_fiducials)**nQubits)), comm, False)
        loc_itr = 0; nLocIters = len(loc_Indices)
        #print("DB: Rank %d indices = " % comm.Get_rank(), loc_Indices)

        with printer.progress_logging(2):
            for itr, prep in enumerate(_itertools.product(*([single_q_fiducials] * nQubits))):
                # There's probably a cleaner way to do this,
                if loc_itr < len(loc_Indices) and itr == loc_Indices[loc_itr]:
                    loc_itr += 1  # but this limits us to this processor's local indices
                else:
                    continue
                #print("DB: Rank %d: running itr=%d" % (comm.Get_rank(), itr))

                printer.show_progress(loc_itr - 1, nLocIters, prefix='--- Finding amped-polys for idle: ')
                prepFid = _objs.Circuit((), line_labels=idle_str.line_labels)
                for i, el in enumerate(prep):
                    prepFid = prepFid + _onqubit(el, qubit_filter[i])

                for meas in _itertools.product(*([single_q_fiducials] * nQubits)):

                    if idt_pauli_dicts is not None:
                        # For idle tomography compatibility, only consider fiducial pairs with either
                        # all-the-same or all-different prep & measure basis (basis is determined
                        # by the *last* letter in the value, e.g. ignore '-' sign in '-X').
                        prepDict, measDict = idt_pauli_dicts
                        rev_prepDict = {v[-1]: k for k, v in prepDict.items()}  # could do this once above,
                        rev_measDict = {v[-1]: k for k, v in measDict.items()}  # but this isn't the bottleneck.
                        cmp = [(rev_prepDict[prep[kk]] == rev_measDict[meas[kk]]) for kk in range(nQubits)]
                        # if all are not the same or all are not different, skip
                        if not (all(cmp) or not any(cmp)): continue

                    measFid = _objs.Circuit((), line_labels=idle_str.line_labels)
                    for i, el in enumerate(meas):
                        measFid = measFid + _onqubit(el, qubit_filter[i])

                    gatename_fidpair_list = [(prep[i], meas[i]) for i in range(nQubits)]
                    if gatename_fidpair_list in selected_gatename_fidpair_lists:
                        continue  # we've already chosen this pair in a previous iteration

                    gstr_L0 = prepFid + measFid            # should be a Circuit
                    gstr_L1 = prepFid + idle_str + measFid  # should be a Circuit
                    ps = model.sim._prs_as_polynomials(prep_lbl, effect_lbls, gstr_L1,
                                                       polynomial_vindices_per_int, resource_alloc)
                    qs = model.sim._prs_as_polynomials(prep_lbl, effect_lbls, gstr_L0,
                                                       polynomial_vindices_per_int, resource_alloc)

                    if algorithm == "sequential":
                        added = False
                        for elbl, p, q in zip(effect_lbls, ps, qs):
                            amped = p + -1 * q  # the amplified poly
                            Jrow = _np.array([[amped.deriv(iParam).evaluate(dummy)
                                               for iParam in _slct.to_array(wrt_params)]])
                            if _np.linalg.norm(Jrow) < 1e-8: continue  # row of zeros can fool matrix_rank

                            Jtest = _np.concatenate((J, Jrow), axis=0)
                            testRank = _np.linalg.matrix_rank(Jtest, tol=RANK_TOL)
                            if testRank > Jrank:
                                printer.log("fidpair: %s,%s (%s) increases rank => %d" %
                                            (str(prep), str(meas), str(elbl), testRank), 4)
                                J = Jtest
                                Jrank = testRank
                                if not added:
                                    selected_gatename_fidpair_lists.append(gatename_fidpair_list)
                                    added = True  # only add fidpair once per elabel loop!
                                if Jrank == Np: break  # this is the largest rank J can take!

                    elif algorithm == "greedy":
                        #test adding all effect labels - get the overall increase in rank due to this fidpair
                        for k, (elbl, p, q) in enumerate(zip(effect_lbls, ps, qs)):
                            amped = p + -1 * q  # the amplified poly
                            Jrows[k, :] = _np.array([[amped.deriv(iParam).evaluate(dummy)
                                                      for iParam in _slct.to_array(wrt_params)]])
                        Jtest = _np.concatenate((J, Jrows), axis=0)
                        testRank = _np.linalg.matrix_rank(Jtest, tol=RANK_TOL)
                        rankInc = testRank - Jrank
                        if rankInc > maxRankInc:
                            maxRankInc = rankInc
                            bestJrows = Jrows.copy()
                            bestFidpair = gatename_fidpair_list
                            if testRank == Np: break  # this is the largest rank we can get!

        if algorithm == "greedy":
            # get the best of the bestJrows, bestFidpair, and maxRankInc
            if comm is not None:
                maxRankIncs_per_rank = comm.allgather(maxRankInc)
                iWinningRank = maxRankIncs_per_rank.index(max(maxRankIncs_per_rank))
                maxRankInc = maxRankIncs_per_rank[iWinningRank]
                if comm.Get_rank() == iWinningRank:
                    comm.bcast(bestJrows, root=iWinningRank)
                    comm.bcast(bestFidpair, root=iWinningRank)
                else:
                    bestJrows = comm.bcast(None, root=iWinningRank)
                    bestFidpair = comm.bcast(None, root=iWinningRank)

            if require_all_amped:
                assert(maxRankInc > 0), "No fiducial pair increased the Jacobian rank!"
            Jrank += maxRankInc
            J = _np.concatenate((J, bestJrows), axis=0)
            selected_gatename_fidpair_lists.append(bestFidpair)
            printer.log("%d fidpairs => rank %d (Np=%d)" %
                        (len(selected_gatename_fidpair_lists), Jrank, Np))

    #DEBUG
    #print("DB: J = ")
    #_gt.print_mx(J)
    #print("DB: svals of J for synthetic idle: ", _np.linalg.svd(J, compute_uv=False))

    return J, Jrank, selected_gatename_fidpair_lists


def _test_amped_polynomials_for_syntheticidle(fidpairs, idle_str, model, prep_lbl=None, effect_lbls=None,
                                              wrt_params=None, verbosity=0):
    """
    Compute the number of model parameters amplified by a given (synthetic) idle sequence.

    Parameters
    ----------
    fidpairs : list
        A list of `(prep,meas)` 2-tuples, where `prep` and `meas` are
        :class:`Circuit` objects, specifying the fiducial pairs to test.

    idle_str : Circuit
        The circuit specifying the idle operation to consider.  This may
        just be a single idle gate, or it could be multiple non-idle gates
        which together act as an idle.

    model : Model
        The model used to compute the polynomial expressions of probabilities
        to first-order.  Thus, this model should always have (simulation)
        type "termorder".

    prep_lbl : Label, optional
        The state preparation label to use.  If None, then the first (and
        usually the only) state prep label of `model` is used, so it's
        usually fine to leave this as None.

    effect_lbls : list, optional
        The list of POVM effect labels to use, as a list of `Label` objects.
        These are *simplified* POVM effect labels, so something like "Mdefault_0",
        and if None the default is all the effect labels of the first POVM of
        `model`, which is usually what you want.

    wrt_params : slice, optional
        The parameters to consider for amplification.  If None, then pairs
        which amplify all of `model`'s parameters are searched for.

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.

    Returns
    -------
    nAmplified : int
        The number of parameters amplified.
    nTotal : int
        The total number of parameters considered for amplification.
    """
    #Assert that model uses termorder:1, as doing L1-L0 to extract the "amplified" part
    # relies on only expanding to *first* order.
    assert(isinstance(model.sim, _TermFSim) and model.sim.max_order == 1), \
        '`model` must use a 1-st order Term-type forward simulator!'

    # printer = _VerbosityPrinter.create_printer(verbosity)
    polynomial_vindices_per_int = _Polynomial._vindices_per_int(model.num_params)
    resource_alloc = _ResourceAllocation()

    if prep_lbl is None:
        prep_lbl = model._default_primitive_prep_layer_lbl()
    if effect_lbls is None:
        povmLbl = model._default_primitive_povm_layer_lbl()
        effect_lbls = [_Lbl("%s_%s" % (povmLbl, l)) for l in model._effect_labels_for_povm(povmLbl)]
    dummy = 5.0 * _np.random.random(model.num_params) + 0.5 * _np.ones(model.num_params, 'd')

    if wrt_params is None: wrt_params = slice(0, model.num_params)
    Np = _slct.length(wrt_params)
    nEffectLbls = len(effect_lbls)
    nRows = len(fidpairs) * nEffectLbls  # number of jacobian rows
    J = _np.empty((nRows, Np), 'complex')

    for i, (prepFid, measFid) in enumerate(fidpairs):
        gstr_L0 = prepFid + measFid            # should be a Circuit
        gstr_L1 = prepFid + idle_str + measFid  # should be a Circuit
        ps = model.sim._prs_as_polynomials(prep_lbl, effect_lbls, gstr_L1,
                                           polynomial_vindices_per_int, resource_alloc)
        qs = model.sim._prs_as_polynomials(prep_lbl, effect_lbls, gstr_L0,
                                           polynomial_vindices_per_int, resource_alloc)

        for k, (elbl, p, q) in enumerate(zip(effect_lbls, ps, qs)):
            amped = p + -1 * q  # the amplified poly
            Jrow = _np.array([[amped.deriv(iParam).evaluate(dummy) for iParam in _slct.to_array(wrt_params)]])
            J[i * nEffectLbls + k, :] = Jrow

    rank = _np.linalg.matrix_rank(J, tol=RANK_TOL)
    #print("Rank = %d, num params = %d" % (rank, Np))
    return rank, Np


def _find_amped_polynomials_for_clifford_syntheticidle(qubit_filter, core_filter, true_idle_pairs, idle_str, max_weight,
                                                       model, single_q_fiducials=None,
                                                       prep_lbl=None, effect_lbls=None, init_j=None, init_j_rank=None,
                                                       wrt_params=None, verbosity=0):
    """
    A specialized version of :function:`_find_amped_polynomials_for_syntheticidle`.

    Similar to :function:`_find_amped_polynomials_for_syntheticidle` but
    specialized to "qubit cloud" processing case used in higher-level
    functions and assumes that `idle_str` is composed of Clifford gates only
    which act on a "core" of qubits (given by `core_filter`).

    In particular, we assume that we already know the fiducial pairs needed
    to amplify all the errors of a "true" (non-synthetic) idle on various
    number of qubits (i.e. max-weights of idle error).  Furthermore, we
    assume that the errors found by these true-idle fiducial pairs are
    of the same kind as those afflicting the synthetic idle, so that
    by restricting our search to just certain true-idle pairs we're able
    to amplify all the parameters of the synthetic idle.

    Because of these assumptions and pre-computed information, this
    function often takes considerably less time to run than
    :function:`_find_amped_polynomials_for_syntheticidle`.

    Parameters
    ----------
    qubit_filter : list
        A list specifying which qubits fiducial pairs should be placed upon.
        Typically this is a subset of all the qubits, as the synthetic idle
        is composed of nontrivial gates acting on a localized set of qubits
        and noise/errors are localized around these.  Within the "cloud"
        picture, `qubit_filter` specifies *all* the qubits in the cloud, not
        just the "core".

    core_filter : list
        A list specifying the "core" qubits - those which the non-idle
        gates within `idle_str` ideally act upon.  This is often a proper subset
        of `qubit_filter` since errors are allowed on qubits which neighbor
        the core qubits in addition to the core qubits themselves.

    true_idle_pairs : dict
        A dictionary whose keys are integer max-weight values and whose values
        are lists of fiducial pairs, each in "gatename-fidpair-list" format,
        whcih give the fiducial pairs needed to amplify all the parameters of
        a non-synthetic idle gate on max-weight qubits.

    idle_str : Circuit
        The circuit specifying the idle operation to consider.  This may
        just be a single idle gate, or it could be multiple non-idle gates
        which together act as an idle.

    max_weight : int
        The maximum weight such that the pairs given by `true_idle_pairs[max_weight]`
        will amplify all the possible errors on `idle_str`.  This must account
        for the fact that the nontrivial comprising `idle_str` may increase the
        weight of errors.  For instance if `idle_str` contains CNOT gates
        on qubits 0 and 1 (the "core") and the noise model allows insertion of
        up to weight-2 errors at any location, then a single weight-2 error
        (recall termorder:1 means there can be only 1 error per circuit) on
        qubits 1 and 2 followed by a CNOT on 0 and 1 could yield an weight-3
        error on qubits 0,1, and 2.

    model : Model
        The model used to compute the polynomial expressions of probabilities
        to first-order.  Thus, this model should always have (simulation)
        type "termorder".

    single_q_fiducials : list, optional
        A list of gate-name tuples (e.g. `('Gx',)`) which specify a set of single-
        qubit fiducials to use when trying to amplify gate parameters.  Note that
        no qubit "state-space" label is required here (i.e. *not* `(('Gx',1),)`);
        the tuples just contain single-qubit gate *names*.  If None, then
        `[(), ('Gx',), ('Gy',)]` is used by default.

    prep_lbl : Label, optional
        The state preparation label to use.  If None, then the first (and
        usually the only) state prep label of `model` is used, so it's
        usually fine to leave this as None.

    effect_lbls : list, optional
        The list of POVM effect labels to use, as a list of `Label` objects.
        These are *simplified* POVM effect labels, so something like "Mdefault_0",
        and if None the default is all the effect labels of the first POVM of
        `model`, which is usually what you want.

    init_j : numpy.ndarray, optional
        An initial Jacobian giving the derivatives of some other polynomials
        with respect to the same `wrt_params` that this function is called with.
        This acts as a starting point, and essentially informs the fiducial-pair
        selection algorithm that some parameters (or linear combos of them) are
        *already* amplified (e.g. by some other germ that's already been
        selected) and for which fiducial pairs are not needed.

    init_j_rank : int, optional
        The rank of `init_j`.  The function could compute this from `init_j`
        but in practice one usually has the rank of `init_j` lying around and
        so this saves a call to `np.linalg.matrix_rank`.

    wrt_params : slice, optional
        The parameters to consider for amplification.  (This function seeks
        fiducial pairs that amplify these parameters.)  If None, then pairs
        which amplify all of `model`'s parameters are searched for.

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.

    Returns
    -------
    J : numpy.ndarray
        The final jacobian with rows equal to the number of chosen amplified
        polynomials (note there is one row per fiducial pair *including* the
        outcome - so there will be two different rows for two different
        outcomes) and one column for each parameter specified by `wrt_params`.
    Jrank : int
        The rank of the jacobian `J`, equal to the number of amplified
        parameters (at most the number requested).
    fidpair_lists : list
        The selected fiducial pairs, each in "gatename-fidpair-list" format.
        See :function:`_find_amped_polynomials_for_syntheticidle` for details.
    """

    #Assert that model uses termorder:1, as doing L1-L0 to extract the "amplified" part
    # relies on only expanding to *first* order.
    assert(isinstance(model.sim, _TermFSim) and model.sim.max_order == 1), \
        '`model` must use a 1-st order Term-type forward simulator!'
    polynomial_vindices_per_int = _Polynomial._vindices_per_int(model.num_params)
    resource_alloc = _ResourceAllocation()

    printer = _VerbosityPrinter.create_printer(verbosity)

    if prep_lbl is None:
        prep_lbl = model._default_primitive_prep_layer_lbl()
    if effect_lbls is None:
        povmLbl = model._default_primitive_povm_layer_lbl()
        effect_lbls = [_Lbl("%s_%s" % (povmLbl, l)) for l in model._effect_labels_for_povm(povmLbl)]
    if single_q_fiducials is None:
        # TODO: assert model has Gx and Gy gates?
        single_q_fiducials = [(), ('Gx',), ('Gy',)]  # ('Gx','Gx')

    #dummy = 0.05*_np.ones(model.num_params,'d') # for evaluating derivs...
    #dummy = 0.05*_np.arange(1,model.num_params+1) # for evaluating derivs...
    #dummy = 0.05*_np.random.random(model.num_params)
    dummy = 5.0 * _np.random.random(model.num_params) + 0.5 * _np.ones(model.num_params, 'd')
    # expect terms to be either coeff*x or coeff*x^2 - (b/c of latter case don't eval at zero)

    #amped_polys = []
    selected_gatename_fidpair_lists = []
    if wrt_params is None: wrt_params = slice(0, model.num_params)
    Np = _slct.length(wrt_params)
    if init_j is None:
        J = _np.empty((0, Np), 'complex'); Jrank = 0
    else:
        J = init_j; Jrank = init_j_rank

    # We presume that we know the fiducial pairs
    #  needed to amplify all "true-idle" errors *of the same
    #  type that are on this synthetic idle* (i.e. H+S
    #  or full LND) up to some weight.  If we also assume
    #  the core-action is Clifford (i.e. maps Paulis->Paulis)
    #  then these same fiducial pairs that find the amplifiable
    #  params of a true idle with up to weight-max_weight terms will
    #  also find all the  amplifiable parameters of the synthetic
    #  idle, with the caveat that the max_weight must account for the
    #  weight-increasing potential of the non-trivial Clifford
    #  action.

    nQubits = len(qubit_filter)
    # nCore = len(core_filter)

    #Tile idle_fidpairs for max_weight onto nQubits
    # (similar to _tile_idle_fidpairs(...) but don't need to convert to circuits?)
    tmpl = get_kcoverage_template(nQubits, max_weight)
    idle_gatename_fidpair_lists = true_idle_pairs[max_weight]
    #print("IDLE GFP LISTS = ",idle_gatename_fidpair_lists)

    gatename_fidpair_lists = []
    for gatename_fidpair_list in idle_gatename_fidpair_lists:
        # replace 0..(k-1) in each template string with the corresponding
        # gatename_fidpair (acts on the single qubit identified by the
        # its index within the template string), then convert to a Circuit/Circuit
        gfp = []
        for tmpl_row in tmpl:
            #mod_tmpl_row = tmpl_row[:]
            #for ql in core_filter: mod_tmpl_row[qubit_filter.index(ql)] = 0 # zero out to remove duplicates on non-core
            instance_row = [gatename_fidpair_list[i] for i in tmpl_row]

            gfp.append(tuple(instance_row))

        gatename_fidpair_lists.extend(gfp)
        # tuple so it can be hashed in remove_duplicates
    _lt.remove_duplicates_in_place(gatename_fidpair_lists)
    ##print("GFP LISTS (nQ=%d) = " % nQubits,gatename_fidpair_lists)
    #printer.log("Testing %d fidpairs for %d-wt idle -> %d after %dQ tiling -> %d w/free %d core (vs %d)"
    #            % (len(idle_gatename_fidpair_lists), max_weight, len(gatename_fidpair_lists),
    #               nQubits, len(gatename_fidpair_lists)*(3**(2*nCore)), nCore, 3**(2*nQubits)))
    #print("DB: over %d qubits -> template w/%d els" % (nQubits, len(tmpl)))
    printer.log("Testing %d fidpairs for %d-wt idle -> %d fidpairs after tiling onto %d qubits"
                % (len(idle_gatename_fidpair_lists), max_weight, len(gatename_fidpair_lists), nQubits))

    for gfp_list in gatename_fidpair_lists:
        # # replace 0..(k-1) in each template string with the corresponding
        # # gatename_fidpair (acts on the single qubit identified by the
        # # its index within the template string), then convert to a Circuit
        # tmpl_instance = [ [gatename_fidpair_list[i] for i in tmpl_row]  for tmpl_row in tmpl ]
        # for gfp_list in tmpl_instance: # circuit-fiducialpair list: one (gn-prepstr,gn-measstr) per qubit

        prep = tuple((gfp_list[i][0] for i in range(nQubits)))  # just the prep-part (OLD prep_noncore)
        meas = tuple((gfp_list[i][1] for i in range(nQubits)))  # just the meas-part (OLD meas_noncore)

        #OLD: back when we tried iterating over *all* core fiducial pairs
        # (now we think/know this is unnecessary - the "true idle" fidpairs suffice)
        #for prep_core in _itertools.product(*([single_q_fiducials]*nCore) ):
        #
        #    #construct prep, a gatename-string, from prep_noncore and prep_core
        #    prep = list(prep_noncore)
        #    for i,core_ql in enumerate(core_filter):
        #        prep[ qubit_filter.index(core_ql) ] = prep_core[i]
        #    prep = tuple(prep)

        prepFid = _objs.Circuit(())
        for i, el in enumerate(prep):
            prepFid = prepFid + _onqubit(el, qubit_filter[i])

        #OLD: back when we tried iterating over *all* core fiducial pairs
        # (now we think/know this is unnecessary - the "true idle" fidpairs suffice)
        #    for meas_core in [0]: # DEBUG _itertools.product(*([single_q_fiducials]*nCore) ):
        #
        #        #construct meas, a gatename-string, from meas_noncore and meas_core
        #        meas = list(meas_noncore)
        #        #for i,core_ql in enumerate(core_filter):
        #        #    meas[ qubit_filter.index(core_ql) ] = meas_core[i]
        #        meas = tuple(meas)

        measFid = _objs.Circuit(())
        for i, el in enumerate(meas):
            measFid = measFid + _onqubit(el, qubit_filter[i])

        #print("PREPMEAS = ",prepFid,measFid)

        gstr_L0 = prepFid + measFid            # should be a Circuit
        gstr_L1 = prepFid + idle_str + measFid  # should be a Circuit
        ps = model.sim._prs_as_polynomials(prep_lbl, effect_lbls, gstr_L1,
                                           polynomial_vindices_per_int, resource_alloc)
        qs = model.sim._prs_as_polynomials(prep_lbl, effect_lbls, gstr_L0,
                                           polynomial_vindices_per_int, resource_alloc)
        added = False
        for elbl, p, q in zip(effect_lbls, ps, qs):
            amped = p + -1 * q  # the amplified poly
            Jrow = _np.array([[amped.deriv(iParam).evaluate(dummy) for iParam in _slct.to_array(wrt_params)]])
            if _np.linalg.norm(Jrow) < 1e-8: continue  # row of zeros can fool matrix_rank

            Jtest = _np.concatenate((J, Jrow), axis=0)
            testRank = _np.linalg.matrix_rank(Jtest, tol=RANK_TOL)
            #print("_find_amped_polynomials_for_syntheticidle: ",prep,meas,elbl," => rank ",testRank, " (Np=",Np,")")
            if testRank > Jrank:
                J = Jtest
                Jrank = testRank
                if not added:
                    gatename_fidpair_list = [(prep[i], meas[i]) for i in range(nQubits)]
                    selected_gatename_fidpair_lists.append(gatename_fidpair_list)
                    added = True  # only add fidpair once per elabel loop!
                if Jrank == Np: break  # this is the largest rank J can take!

    #DEBUG
    #print("DB: J = (wrt = ",wrt_params,")")
    #_mt.print_mx(J,width=4,prec=1)
    #print("DB: svals of J for synthetic idle: ", _np.linalg.svd(J, compute_uv=False))

    return J, Jrank, selected_gatename_fidpair_lists


def _get_fidpairs_needed_to_access_amped_polynomials(qubit_filter, core_filter, germ_power_str, amped_poly_j,
                                                     idle_gatename_fidpair_lists, model,
                                                     single_q_fiducials=None, prep_lbl=None, effect_lbls=None,
                                                     wrt_params=None, verbosity=0):
    """
    Finds a set of fiducial pairs that probe `germ_power_str` so that the underlying germ is effective.

    More specifically, fiducial pairs must be found that probe or access the given
    germ-power such that probabilities from the `prep_fiducial + germ_power + meas_fiducial`
    circuits are sensitive to changes in the known-amplifiable directions in parameter space
    for the germ.

    This function works within the "cloud" picture of a core of qubits where
    there is nontrivial *ideal* action and a larger set of qubits upon which
    errors may exist.

    This function is used to find, after we know which directions in parameter
    -space are amplifiable by a germ (via analyzing its synthetic idle
    counterpart), which fiducial pairs are needed to amplify these directions
    when a non-synthetic-idle power of the germ is used.

    Parameters
    ----------
    qubit_filter : list
        A list specifying which qubits fiducial pairs should be placed upon.
        Typically this is a subset of all the qubits, and a "cloud" around
        the qubits being ideally acted upon.

    core_filter : list
        A list specifying the "core" qubits - those which the gates in
        `germ_power_str` ideally act upon.  This is often a proper subset
        of `qubit_filter` since errors are allowed on qubits which neighbor
        the core qubits in addition to the core qubits themselves.

    germ_power_str : Circuit
        The (non-synthetic-idle) germ power string under consideration.

    amped_poly_j : numpy.ndarray
        A jacobian matrix whose rowspace gives the space of amplifiable
        parameters.  The shape of this matrix is `(Namplified, Np)`, where
        `Namplified` is the number of independent amplified parameters and
        `Np` is the total number of parameters under consideration (the
        length of `wrt_params`).  This function seeks to find fiducial pairs
        which amplify this same space of parameters.

    idle_gatename_fidpair_lists : list
        A list of the fiducial pairs which amplify the entire space given
        by `amped_poly_j` for the germ when it is repeated enough to be a
        synthetic idle.  The strategy for finding fiducial pairs in the
        present case it to just monkey with the *core-qubit* parts of the
        *measurement* idle fiducials (non-core qubits are ideally the idle,
        and one can either modify the prep or the measure to "catch" what
        the non-idle `germ_power_str` does to the amplified portion of the
        state space).

    model : Model
        The model used to compute the polynomial expressions of probabilities
        to first-order.  Thus, this model should always have (simulation)
        type "termorder:1".

    single_q_fiducials : list, optional
        A list of gate-name tuples (e.g. `('Gx',)`) which specify a set of single-
        qubit fiducials to use when trying to amplify gate parameters.  Note that
        no qubit "state-space" label is required here (i.e. *not* `(('Gx',1),)`);
        the tuples just contain single-qubit gate *names*.  If None, then
        `[(), ('Gx',), ('Gy',)]` is used by default.

    prep_lbl : Label, optional
        The state preparation label to use.  If None, then the first (and
        usually the only) state prep label of `model` is used, so it's
        usually fine to leave this as None.

    effect_lbls : list, optional
        The list of POVM effect labels to use, as a list of `Label` objects.
        These are *simplified* POVM effect labels, so something like "Mdefault_0",
        and if None the default is all the effect labels of the first POVM of
        `model`, which is usually what you want.

    wrt_params : slice, optional
        The parameters being considered for amplification.  (This should be
        the same as that used to produce `idle_gatename_fidpair_lists`).

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.

    Returns
    -------
    fidpair_lists : list
        The selected fiducial pairs, each in "gatename-fidpair-list" format.
        See :function:`_find_amped_polynomials_for_syntheticidle` for details.
    """
    printer = _VerbosityPrinter.create_printer(verbosity)
    polynomial_vindices_per_int = _Polynomial._vindices_per_int(model.num_params)
    resource_alloc = _ResourceAllocation()

    if prep_lbl is None:
        prep_lbl = model._default_primitive_prep_layer_lbl()
    if effect_lbls is None:
        povmLbl = model._default_primitive_povm_layer_lbl(sslbls=None)
        effect_lbls = model._effect_labels_for_povm(povmLbl)
    if single_q_fiducials is None:
        # TODO: assert model has Gx and Gy gates?
        single_q_fiducials = [(), ('Gx',), ('Gy',)]  # ('Gx','Gx')

    #dummy = 0.05*_np.ones(model.num_params,'d') # for evaluating derivs...
    #dummy = 0.05*_np.arange(1,model.num_params+1) # for evaluating derivs...
    dummy = 5.0 * _np.random.random(model.num_params) + 0.5 * _np.ones(model.num_params, 'd')
    # expect terms to be either coeff*x or coeff*x^2 - (b/c of latter case don't eval at zero)

    #OLD: selected_fidpairs = []
    gatename_fidpair_lists = []
    if wrt_params is None: wrt_params = slice(0, model.num_params)
    Np = _slct.length(wrt_params)
    Namped = amped_poly_j.shape[0]; assert(amped_poly_j.shape[1] == Np)
    J = _np.empty((0, Namped), 'complex'); Jrank = 0

    #loop over all possible fiducial pairs
    nQubits = len(qubit_filter)
    nCore = len(core_filter)

    # we already know the idle fidpair preps are almost sufficient
    # - we just *may* need to modify the measure (or prep, but we choose
    #   the measure) fiducial on *core* qubits (with nontrivial base action)

    #OLD
    #idle_preps = [ tuple( (gfp_list[i][0] for i in range(nQubits)) )
    #          for gfp_list in idle_gatename_fidpair_lists ] # just the prep-part
    #_lt.remove_duplicates_in_place(idle_preps)

    printer.log("Testing %d fidpairs for idle -> %d seqs w/free %d core (vs %d)"
                % (len(idle_gatename_fidpair_lists),
                   len(idle_gatename_fidpair_lists) * (3**(nCore)), nCore,
                   3**(2 * nQubits)))

    already_tried = set()
    cores = [None] + list(_itertools.product(*([single_q_fiducials] * nCore)))
    # try *no* core insertion at first - leave as idle - before going through them...

    for prep_core in cores:  # weird loop order b/c we don't expect to need this one
        if prep_core is not None:  # I don't think this *should* happen
            _warnings.warn(("Idle's prep fiducials only amplify %d of %d"
                            " directions!  Falling back to vary prep on core")
                           % (Jrank, Namped))

        for gfp_list in idle_gatename_fidpair_lists:
            #print("GFP list = ",gfp_list)
            prep_noncore = tuple((gfp_list[i][0] for i in range(nQubits)))  # just the prep-part
            meas_noncore = tuple((gfp_list[i][1] for i in range(nQubits)))  # just the meas-part

            if prep_core is None:
                prep = prep_noncore  # special case where we try to leave it unchanged.
            else:
                # construct prep, a gatename-string, from prep_noncore and prep_core
                prep = list(prep_noncore)
                for i, core_ql in enumerate(core_filter):
                    prep[qubit_filter.index(core_ql)] = prep_core[i]
                prep = tuple(prep)

            prepFid = _objs.Circuit(())
            for i, el in enumerate(prep):
                prepFid = prepFid + _onqubit(el, qubit_filter[i])

            #for meas in _itertools.product(*([single_q_fiducials]*nQubits) ):
            #for meas_core in _itertools.product(*([single_q_fiducials]*nCore) ):
            for meas_core in cores:

                if meas_core is None:
                    meas = meas_noncore
                else:
                    #construct meas, a gatename-string, from meas_noncore and meas_core
                    meas = list(meas_noncore)
                    for i, core_ql in enumerate(core_filter):
                        meas[qubit_filter.index(core_ql)] = meas_core[i]
                    meas = tuple(meas)

                measFid = _objs.Circuit(())
                for i, el in enumerate(meas):
                    measFid = measFid + _onqubit(el, qubit_filter[i])
                #print("CONSIDER: ",prep,"-",meas)

                opstr = prepFid + germ_power_str + measFid  # should be a Circuit
                if opstr in already_tried: continue
                else: already_tried.add(opstr)

                ps = model.sim._prs_as_polynomials(prep_lbl, effect_lbls, opstr,
                                                   polynomial_vindices_per_int, resource_alloc)
                #OLD: Jtest = J
                added = False
                for elbl, p in zip(effect_lbls, ps):
                    #print(" POLY = ",p)
                    #For each fiducial pair (included pre/effect), determine how the
                    # (polynomial) probability relates to the *amplified* directions
                    # (also polynomials - now encoded by a "Jac" row/vec)
                    prow = _np.array([p.deriv(iParam).evaluate(dummy)
                                      for iParam in _slct.to_array(wrt_params)])  # complex
                    Jrow = _np.array([[_np.vdot(prow, amped_row) for amped_row in amped_poly_j]])  # complex
                    if _np.linalg.norm(Jrow) < 1e-8: continue  # row of zeros can fool matrix_rank

                    Jtest = _np.concatenate((J, Jrow), axis=0)
                    testRank = _np.linalg.matrix_rank(Jtest, tol=RANK_TOL)
                    if testRank > Jrank:
                        #print("ACCESS")
                        #print("ACCESS: ",prep,meas,testRank, _np.linalg.svd(Jtest, compute_uv=False))
                        J = Jtest
                        Jrank = testRank
                        if not added:
                            gatename_fidpair_lists.append([(prep[i], meas[i]) for i in range(nQubits)])
                            added = True
                        #OLD selected_fidpairs.append( (prepFid, measFid) )
                        if Jrank == Namped:
                            # then we've selected enough pairs to access all of the amplified directions
                            return gatename_fidpair_lists  # (i.e. the rows of `amped_poly_j`)

    #DEBUG
    #print("DEBUG: J = ")
    #_mt.print_mx(J)
    #print("SVals = ",_np.linalg.svd(J, compute_uv=False))
    #print("Nullspace = ")
    #_gt.print_mx(pygsti.tools.nullspace(J))

    raise ValueError(("Could not find sufficient fiducial pairs to access "
                      "all the amplified directions - only %d of %d were accessible")
                     % (Jrank, Namped))
    #_warnings.warn(("Could not find sufficient fiducial pairs to access "
    #                  "all the amplified directions - only %d of %d were accessible")
    #                 % (Jrank,Namped))
    #return gatename_fidpair_lists # (i.e. the rows of `amped_poly_j`)


def _tile_idle_fidpairs(qubit_labels, idle_gatename_fidpair_lists, max_idle_weight):
    """
    Tile a set of fiducial pairs that amplify idle errors.

    Tile a set of fiducial pairs that are sufficient for amplifying all the
    true-idle errors on `max_idle_weight` qubits (so with weight up to `max_idle_weight`
    onto `nQubits` qubits.

    This function essentaily converts fiducial pairs that amplify all
    up-to-weight-k errors on k qubits to fiducial pairs that amplify all
    up-to-weight-k errors on `nQubits` qubits (where `k = max_idle_weight`).

    Parameters
    ----------
    qubit_labels : int
        The labels of the final qubits.  These are the line labels of the
        returned circuits.

    idle_gatename_fidpair_lists : list
        A list of the fiducial pairs which amplify the errors on
        `max_idle_weight` qubits (so with weight up to `max_idle_weight`).
        Each element of this list is a fiducial pair in
        "gatename-fidpair-list" format.  These are the fiducial pairs
        to "tile".

    max_idle_weight : int
        The number of qubits and maximum amplified error weight for
        the fiducial pairs given by `idle_gatename_fidpair_lists`.

    Returns
    -------
    fidpairs : list
        A list of `(prep,meas)` 2-tuples, where `prep` and `meas` are
        :class:`Circuit` objects, giving the tiled fiducial pairs.
    """

    # "Tile w/overlap" the fidpairs for a k-qubit subset (where k == max_idle_weight)

    # we want to create a k-coverage set of length-nQubits strings/lists containing
    # the elements 012..(k-1)(giving the "fiducial" - possible a gate sequence - for
    # each qubit) such that for any k qubits the set includes string where these qubits
    # take on all the fiducial pairs given in the idle fiducial pairs

    # Each element of idle_gatename_fidpair_lists is a "gatename_fidpair_list".
    # Each "gatename_fidpair_list" is a list of k (prep-gate-name-str, meas-gate-name-str)
    # tuples, one per *qubit*, giving the gate names to perform on *that* qubit.

    #OLD - we don't need this conversion since we can take the gatename_fidpair_lists as an arg.
    # XX idle_fidpairs elements are (prepStr, measStr) on qubits 0->(k-1); to convert each
    # XX element to a list of k (prep-gate-name-str, meas-gate-name-str) tuples one per *qubit*.

    nQubits = len(qubit_labels)
    tmpl = get_kcoverage_template(nQubits, max_idle_weight)
    final_fidpairs = []

    def merge_into_1q(g_str, gate_names, qubit_label):
        """ Add gate_names, all acting on qubit_label, to g_str """
        while len(g_str) < len(gate_names): g_str.append([])  # make sure g_str is long enough
        for iLayer, name in enumerate(gate_names):
            # only 1 op per qubit per layer!
            assert(qubit_label not in set(_itertools.chain(*[l.sslbls for l in g_str[iLayer]])))
            g_str[iLayer].append(_Lbl(name, qubit_label))  # g_str[i] is a list of i-th layer labels
            if iLayer > 0: assert(qubit_label in set(_itertools.chain(
                *[l.sslbls for l in g_str[iLayer - 1]])))  # just to be safe

    for gatename_fidpair_list in idle_gatename_fidpair_lists:
        # replace 0..(k-1) in each template string with the corresponding
        # gatename_fidpair (acts on the single qubit identified by the
        # its index within the template string), then convert to a Circuit
        tmpl_instance = [[gatename_fidpair_list[i] for i in tmpl_row] for tmpl_row in tmpl]
        for tmpl_instance_row in tmpl_instance:
            # tmpl_instance_row row is nQubits long; elements give the
            # gate *names* to perform on that qubit.
            prep_gates = []
            meas_gates = []
            for iQubit, gatename_fidpair in enumerate(tmpl_instance_row):
                prep_gatenames, meas_gatenames = gatename_fidpair
                #prep_gates.extend( [_Lbl(gatename,iQubit) for gatename in prep_gatenames ]) #OLD: SERIAL strs
                #meas_gates.extend( [_Lbl(gatename,iQubit) for gatename in meas_gatenames ]) #OLD: SERIAL strs
                merge_into_1q(prep_gates, prep_gatenames, qubit_labels[iQubit])
                merge_into_1q(meas_gates, meas_gatenames, qubit_labels[iQubit])

            final_fidpairs.append((_objs.Circuit(prep_gates, line_labels=qubit_labels),
                                   _objs.Circuit(meas_gates, line_labels=qubit_labels)))

    _lt.remove_duplicates_in_place(final_fidpairs)
    return final_fidpairs


def _tile_cloud_fidpairs(template_gatename_fidpair_lists, template_germpower, max_len, template_germ,
                         clouds, qubit_labels):
    """
    Tile fiducial pairs that amplify "cloud" errors.

    Take a "cloud template", giving the fiducial pairs for a germ power acting
    on qubits labeled 0 to `cloudsize-1`, and map those fiducial pairs into
    fiducial pairs for all the qubits by placing in parallel the pairs for
    as many non-overlapping clouds as possible.  This function performs a
    function analogous to :function:`_tile_idle_fidpairs` except here we tile
    fiducial pairs for non-idle operations.

    Parameters
    ----------
    template_gatename_fidpair_lists : list
        A list of the fiducial pairs for the given template - that is, the
        pairs with which amplify all the desired errors for `template_germpower`
        (acting on qubits labeled by the integers 0 to the cloud size minus one).

    template_germpower : Circuit
        The germ power string under consideration.  This gives the action on
        the "core" qubits of the clouds, and is needed to construct the
        final fiducial + germPower + fiducial sequences returned by this
        function.

    max_len : int
        The maximum length used to construct template_germpower.  This is only
        needed to tag elements of the returned `sequences` list.

    template_germ : Circuit
        The germ string under consideration.  This is only needed to tag
        elements of the returned `sequences` list and place elements in
        the returned `germs` list.

    clouds : list
        A list of `(cloud_dict, template_to_cloud_map)` tuples specifying the
        set of equivalent clouds corresponding to the template.

    qubit_labels : list
        A list of the final qubit labels, which are the line labels of
        the returned circuits.

    Returns
    -------
    sequences : list
        A list of (Circuit, max_len, germ, prepFid, measFid) tuples specifying the
        final "tiled" fiducial pairs sandwiching `germPowerStr` for as many
        clouds in parallel as possible.  Actual qubit labels (not the always-
        integer labels used in templates) are used in these strings.  There are
        no duplicates in this list.
    germs : list
        A list of Circuit objects giving all the germs (with appropriate
        qubit labels).
    """
    unused_clouds = list(clouds)
    sequences = []
    germs = []

    while(len(unused_clouds) > 0):

        #figure out what clouds can be processed in parallel
        first_unused = unused_clouds[0]  # a cloud_dict, template_to_cloud_map tuple
        parallel_clouds = [first_unused]
        parallel_qubits = set(first_unused[0]['qubits'])  # qubits used by parallel_clouds
        del unused_clouds[0]

        to_delete = []
        for i, cloud in enumerate(unused_clouds):
            if len(parallel_qubits.intersection(cloud[0]['qubits'])) == 0:
                parallel_qubits.update(cloud[0]['qubits'])
                parallel_clouds.append(cloud)
                to_delete.append(i)
        for i in reversed(to_delete):
            del unused_clouds[i]

        #Create gate sequence "info-tuples" by processing in parallel the
        # list of parallel_clouds

        def merge_into_1q(g_str, gate_names, qubit_label):
            """ Add gate_names, all acting on qubit_label, to g_str """
            while len(g_str) < len(gate_names): g_str.append([])  # make sure prepStr is long enough
            for iLayer, name in enumerate(gate_names):
                # only 1 op per qubit per layer!
                assert(qubit_label not in set(_itertools.chain(*[l.sslbls for l in g_str[iLayer]])))
                g_str[iLayer].append(_Lbl(name, qubit_label))  # g_str[i] is a list of i-th layer labels
                if iLayer > 0: assert(qubit_label in set(_itertools.chain(
                    *[l.sslbls for l in g_str[iLayer - 1]])))  # only 1 op per qubit per layer!

        def merge_into(g_str, g_str_qubits, op_labels):
            """ Add op_labels to g_str using g_str_qubits to keep track of available qubits """
            for lbl in op_labels:
                iLayer = 0
                while True:  # find a layer that can accomodate lbl
                    if len(g_str_qubits) < iLayer + 1:
                        g_str.append([]); g_str_qubits.append(set())
                    if len(g_str_qubits[iLayer].intersection(lbl.sslbls)) == 0:
                        break
                    iLayer += 1
                g_str[iLayer].append(lbl)
                g_str_qubits[iLayer].update(lbl.sslbls)

        for template_gatename_fidpair_list in template_gatename_fidpair_lists:
            prepStr = []
            measStr = []
            germStr = []; germStr_qubits = []
            germPowerStr = []; germPowerStr_qubits = []
            for cloud in parallel_clouds:
                cloud_dict, template_to_cloud_map = cloud
                cloud_to_template_map = {c: t for t, c in template_to_cloud_map.items()}

                germ = template_germ.map_state_space_labels(template_to_cloud_map)
                germPower = template_germpower.map_state_space_labels(template_to_cloud_map)

                for cloud_ql in cloud_dict['qubits']:
                    prep, meas = template_gatename_fidpair_list[cloud_to_template_map[cloud_ql]]  # gate-name lists
                    #prepStr.extend( [_Lbl(name,cloud_ql) for name in prep] ) #OLD: SERIAL strs
                    #measStr.extend( [_Lbl(name,cloud_ql) for name in meas] ) #OLD: SERIAL strs
                    merge_into_1q(prepStr, prep, cloud_ql)
                    merge_into_1q(measStr, meas, cloud_ql)

                #germStr.extend( list(germ) ) #OLD: SERIAL strs
                #germPowerStr.extend( list(germPower) ) #OLD: SERIAL strs
                merge_into(germStr, germStr_qubits, germ)
                merge_into(germPowerStr, germPowerStr_qubits, germPower)

            germs.append(_objs.Circuit(germStr, line_labels=qubit_labels))
            sequences.append((_objs.Circuit(prepStr + germPowerStr + measStr, line_labels=qubit_labels),
                              max_len, germs[-1],
                              _objs.Circuit(prepStr, line_labels=qubit_labels),
                              _objs.Circuit(measStr, line_labels=qubit_labels)))
            # circuit, max_len, germ, prepFidIndex, measFidIndex??

    # return a list of circuits (duplicates removed)
    return _lt.remove_duplicates(sequences), _lt.remove_duplicates(germs)


def _compute_reps_for_synthetic_idle(model, germ_str, nqubits, core_qubits):
    """
    Return the number of times `germ_str` must be repeated to form a synthetic idle gate.

    Parameters
    ----------
    model : Model
        A model containing matrix representations of all the gates
        in `germ_str`.

    germ_str : Circuit
        The germ circuit to repeat.

    nqubits : int
        The total number of qubits that `model` acts on.  This
        is used primarily for sanity checks.

    core_qubits : list
        A list of the qubit labels upon which `germ_str` ideally acts
        nontrivially.  This could be inferred from `germ_str` but serves
        as a sanity check and more concrete specification of what
        state space the gate action takes place within.

    Returns
    -------
    int
    """
    # First, get a dense representation of germ_str on core_qubits
    # Note: only works with one level of embedding...
    def extract_gate(g):
        """ Get the gate action as a dense gate on core_qubits """
        if isinstance(g, _objs.EmbeddedOp):
            assert(len(g.state_space_labels.labels) == 1)  # 1 tensor product block
            assert(len(g.state_space_labels.labels[0]) == nqubits)  # expected qubit count
            qubit_labels = g.state_space_labels.labels[0]

            new_qubit_labels = []
            for core_ql in core_qubits:
                if core_ql in qubit_labels: new_qubit_labels.append(core_ql)  # same convention!
                #elif ("Q%d" % core_ql) in qubit_labels: new_qubit_labels.append("Q%d" % core_ql)  # HACK!
            ssl = _StateSpaceLabels(new_qubit_labels)
            assert(all([(tgt in new_qubit_labels) for tgt in g.targetLabels]))  # all target qubits should be kept!
            if len(new_qubit_labels) == len(g.targetLabels):
                # embedded gate acts on entire core-qubit space:
                return g.embedded_op
            else:
                return _objs.EmbeddedDenseOp(ssl, g.targetLabels, g.embedded_op)

        elif isinstance(g, _objs.ComposedOp):
            return _objs.ComposedDenseOp([extract_gate(f) for f in g.factorops])
        else:
            raise ValueError("Cannot extract core contrib from %s" % str(type(g)))

    core_dim = 4**len(core_qubits)
    product = _np.identity(core_dim, 'd')
    core_gates = {}
    for gl in germ_str:
        if gl not in core_gates:
            core_gates[gl] = extract_gate(model.operation_blks['layers'][gl])
        product = _np.dot(core_gates[gl], product)

    # Then just do matrix products until we hit the identity (or a large order)
    reps = 1; target = _np.identity(core_dim, 'd')
    repeated = product
    while(_np.linalg.norm(repeated - target) > 1e-6 and reps < 20):  # HARDCODED MAX_REPS
        repeated = _np.dot(repeated, product); reps += 1

    return reps


def _get_candidates_for_core(model, core_qubits, candidate_counts, seed_start):
    """
    Returns a list of candidate germs which act on a given set of "core" qubits.

    This function figures out what gates within `model` are available to act
    (only) on `core_qubits` and then randomly selects a set of them based on
    `candidate_counts`.  In each candidate germ, at least one gate will act
    on *all* of the core qubits (if the core is 2 qubits then this function
    won't return a germ consisting of just 1-qubit gates).

    This list serves as the inital candidate list when a new cloud template is
    created within create_cloudnoise_circuits.

    Parameters
    ----------
    model : Model
        The model specifying the gates allowed to be in the germs.

    core_qubits : list
        A list of the qubit labels.  All returned candidate germs (ideally) act
        nontrivially only on these qubits.

    candidate_counts : dict
        A dictionary specifying how many germs of each length to include in the
        returned set.  Thus both keys and values are integers (key specifies
        germ length, value specifies number).  The special value `"all upto"`
        means that all possible candidate germs up to the corresponding key's
        value should be included.  A typical value for this argument might be
        `{4: 'all upto', 5: 10, 6: 10 }`.

    seed_start : int
        A *initial* random number generator seed value to use.  Incrementally
        greater seeds are used for the different keys of `candidate_counts`.

    Returns
    -------
    list : candidate_germs
        A list of Circuit objects.
    """
    # or should this be ...for_cloudbank - so then we can check that gates for all "equivalent" clouds exist?

    # collect gates that only act on core_qubits.
    oplabel_list = []; full_core_list = []
    for gl in model.primitive_op_labels:
        if gl.sslbls is None: continue  # gates that act on everything (usually just the identity Gi gate)
        if set(gl.sslbls).issubset(core_qubits):
            oplabel_list.append(gl)
        if set(gl.sslbls) == set(core_qubits):
            full_core_list.append(gl)

    # form all low-length strings out of these gates.
    candidate_germs = []
    for i, (germLength, count) in enumerate(candidate_counts.items()):
        if count == "all upto":
            candidate_germs.extend(_gsc.list_all_circuits_without_powers_and_cycles(
                oplabel_list, max_length=germLength))
        else:
            candidate_germs.extend(_gsc.list_random_circuits_onelen(
                oplabel_list, germLength, count, seed=seed_start + i))

    #filter: make sure there's at least one gate in each germ that acts on the *entire* core
    candidate_germs = [g for g in candidate_germs if any([(gl in g) for gl in full_core_list])]  # filter?

    return candidate_germs


@_deprecated_fn("Use pygsti.construction.create_standard_cloudnoise_circuits(...).")
def _create_xycnot_cloudnoise_circuits(num_qubits, max_lengths, geometry, cnot_edges, max_idle_weight=1, maxhops=0,
                                       extra_weight_1_hops=0, extra_gate_weight=0, paramroot="H+S",
                                       sparse=False, verbosity=0, cache=None, idle_only=False,
                                       idt_pauli_dicts=None, algorithm="greedy", comm=None):

    """
    Compute circuits which amplify the parameters of a particular :class:`CloudNoiseModel`.

    Returns circuits that amplify the parameters of a :class:`CloudNoiseModel` containing
    X(pi/2), Y(pi/2) and CNOT gates using the specified arguments.

    Parameters
    ----------
    num_qubits : int
        The total number of qubits.

    max_lengths : list
        A list of integers specifying the different maximum lengths for germ
        powers.  Typically these values start a 1 and increase by powers of
        2, e.g. `[1,2,4,8,16]`.

    geometry : {"line","ring","grid","torus"} or QubitGraph
        The type of connectivity among the qubits, specifying a
        graph used to define neighbor relationships.  Alternatively,
        a :class:`QubitGraph` object may be passed directly.

    cnot_edges : list
        A list of 2-tuples giving the pairs of qubits where CNOT gates
        exist (i.e. are available).

    max_idle_weight : int, optional
        The maximum-weight for errors on the global idle gate.

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

    paramroot : {"CPTP", "H+S+A", "H+S", "S", "H+D+A", "D+A", "D"}
        The "root" (no trailing " terms", etc.) parameterization used for the
        cloud noise model (which specifies what needs to be amplified).

    sparse : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        `num_qubits`-qubit gates use sparse bases or not.  When sparse, these
        Lindblad gates - especially those with high-weight action or errors - take
        less memory, but their simulation is slightly slower.  Usually it's fine to
        leave this as the default (False), except when considering unusually
        high-weight errors.

    verbosity : int, optional
        An integer >= 0 dictating how much output to send to stdout.

    cache : dict, optional
        A cache dictionary which holds template information so that repeated
        calls to `_create_xycnot_cloudnoise_circuits` can draw on the same
        pool of templates.

    idle_only : bool, optional
        If True, only sequences for the idle germ are returned.  This is useful
        for idle tomography in particular.

    idt_pauli_dicts : tuple, optional
        A (prepDict,measDict) tuple of dicts that maps a 1-qubit Pauli basis
        string (e.g. 'X' or '-Y') to a sequence of gate *names*.  If given,
        the idle-germ fiducial pairs chosen by this function are restricted
        to those where either 1) each qubit is prepared and measured in the
        same basis or 2) each qubits is prepared and measured in different
        bases (note: '-X' and 'X" are considered the *same* basis).  This
        restriction makes the resulting sequences more like the "standard"
        ones of idle tomography, and thereby easier to interpret.

    algorithm : {"greedy","sequential"}
        The algorithm is used internall by
        :function:`_find_amped_polynomials_for_syntheticidle`.  You should leave this
        as the default unless you know what you're doing.

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    list
        A list of :class:`Circuit`s
    """
    from pygsti.modelpacks.legacy import std1Q_XY  # the base model for 1Q gates
    from pygsti.modelpacks.legacy import std2Q_XYICNOT  # the base model for 2Q (CNOT) gate

    tgt1Q = std1Q_XY.target_model("static")
    tgt2Q = std2Q_XYICNOT.target_model("static")
    Gx = tgt1Q.operations['Gx']
    Gy = tgt1Q.operations['Gy']
    Gcnot = tgt2Q.operations['Gcnot']
    gatedict = _collections.OrderedDict([('Gx', Gx), ('Gy', Gy), ('Gcnot', Gcnot)])
    availability = {}
    if cnot_edges is not None: availability['Gcnot'] = cnot_edges

    if paramroot in ("H+S", "S", "H+D", "D",
                     "H+s", "s", "H+d", "d"):  # no affine - can get away w/1 fewer fiducials
        singleQfiducials = [(), ('Gx',), ('Gy',)]
    else:
        singleQfiducials = [(), ('Gx',), ('Gy',), ('Gx', 'Gx')]

    return create_cloudnoise_circuits(num_qubits, max_lengths, singleQfiducials,
                                      gatedict, availability, geometry, max_idle_weight, maxhops,
                                      extra_weight_1_hops, extra_gate_weight, paramroot,
                                      sparse, True, verbosity, cache, idle_only,
                                      idt_pauli_dicts, algorithm, comm=comm)


def create_standard_localnoise_circuits(num_qubits, max_lengths, single_q_fiducials,
                                        gate_names, nonstd_gate_unitaries=None,
                                        availability=None, geometry="line",
                                        paramroot="H+S", sparse_lindblad_basis=False, sparse_lindblad_reps=True,
                                        verbosity=0, cache=None, idle_only=False, idt_pauli_dicts=None,
                                        algorithm="greedy", idle_op_str=((),), comm=None):
    """
    Find a set of circuits that amplifies all the parameters of a local noise model.

    Create a set of `fiducial1+germ^power+fiducial2` sequences which amplify
    all of the parameters of a :class:`LocalNoiseModel` created by passing the
    arguments of this function to :function:`create_localnoise_model`

    Note that this function essentialy performs fiducial selection, germ
    selection, and fiducial-pair reduction simultaneously.  It is used to
    generate a short (ideally minimal) list of sequences needed for multi-
    qubit GST.

    This function allows the local noise model to be created by specifing
    standard gate names or additional gates as *unitary* operators.  Some
    example gate names are:

        - 'Gx','Gy','Gz' : 1Q pi/2 rotations
        - 'Gxpi','Gypi','Gzpi' : 1Q pi rotations
        - 'Gh' : Hadamard
        - 'Gp' : phase
        - 'Gcphase','Gcnot','Gswap' : standard 2Q gates

    Parameters
    ----------
    num_qubits : int
        The number of qubits

    max_lengths : list
        A list of integers specifying the different maximum lengths for germ
        powers.  Typically these values start a 1 and increase by powers of
        2, e.g. `[1,2,4,8,16]`.

    single_q_fiducials : list
        A list of gate-name-tuples, e.g. `[(), ('Gx',), ('Gy',), ('Gx','Gx')]`,
        which form a set of 1-qubit fiducials for the given model (compatible
        with both the gates it posseses and their parameterizations - for
        instance, only `[(), ('Gx',), ('Gy',)]` is needed for just Hamiltonian
        and Stochastic errors.

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

    availability : dict, optional
        A dictionary whose keys are the same gate names as in
        `gate_names` and whose values are lists of qubit-label-tuples.  Each
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
        edges, for 2Q gates of the graphy given by `geometry`.
        - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
        on any target qubits via an :class:`EmbeddingOpFactory` (uses less
        memory but slower than `"all-permutations"`.

        If a gate name (element of `gate_names`) is not present in `availability`,
        the default is `"all-edges"`.

    geometry : {"line","ring","grid","torus"} or QubitGraph
        The type of connectivity among the qubits, specifying a
        graph used to define neighbor relationships.  Alternatively,
        a :class:`QubitGraph` object may be passed directly.

    paramroot : {"CPTP", "H+S+A", "H+S", "S", "H+D+A", "D+A", "D"}
        The "root" (no trailing " terms", etc.) parameterization used for the
        cloud noise model (which specifies what needs to be amplified).

    sparse_lindblad_basis : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        `num_qubits`-qubit gates use sparse bases or not.  When sparse, these
        Lindblad gates - especially those with high-weight action or errors - take
        less memory, but their simulation is slightly slower.  Usually it's fine to
        leave this as the default (False), except when considering unusually
        high-weight errors.

    sparse_lindblad_reps : bool, optional
        Whether created Lindblad operations use sparse (more memory efficient but
        slower action) or dense representations.

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.

    cache : dict, optional
        A cache dictionary which holds template information so that repeated
        calls to `create_standard_cloudnoise_circuits` can draw on the same
        pool of templates.

    idle_only : bool, optional
        If True, only sequences for the idle germ are returned.  This is useful
        for idle tomography in particular.

    idt_pauli_dicts : tuple, optional
        A (prepDict,measDict) tuple of dicts that maps a 1-qubit Pauli basis
        string (e.g. 'X' or '-Y') to a sequence of gate *names*.  If given,
        the idle-germ fiducial pairs chosen by this function are restricted
        to those where either 1) each qubit is prepared and measured in the
        same basis or 2) each qubits is prepared and measured in different
        bases (note: '-X' and 'X" are considered the *same* basis).  This
        restriction makes the resulting sequences more like the "standard"
        ones of idle tomography, and thereby easier to interpret.

    algorithm : {"greedy","sequential"}
        The algorithm is used internall by
        :function:`_find_amped_polynomials_for_syntheticidle`.  You should leave this
        as the default unless you know what you're doing.

    idle_op_str : Circuit or tuple, optional
        The circuit or label that is used to indicate a completely
        idle layer (all qubits idle).

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    PlaquetteGridCircuitStructure
        An object holding a structured (using germ and fiducial sub-sequences)
        list of circuits.
    """
    #Same as cloudnoise but no hopping. -- should max_idle_weight == 0?
    return create_standard_cloudnoise_circuits(num_qubits, max_lengths, single_q_fiducials,
                                               gate_names, nonstd_gate_unitaries,
                                               availability, geometry,
                                               max_idle_weight=1, maxhops=0, extra_weight_1_hops=0,
                                               extra_gate_weight=0,
                                               paramroot=paramroot,
                                               sparse_lindblad_basis=sparse_lindblad_basis,
                                               sparse_lindblad_reps=sparse_lindblad_reps,
                                               verbosity=verbosity,
                                               cache=cache, idle_only=idle_only,
                                               idt_pauli_dicts=idt_pauli_dicts, algorithm=algorithm,
                                               idle_op_str=idle_op_str, comm=comm)


def create_standard_cloudnoise_circuits(num_qubits, max_lengths, single_q_fiducials,
                                        gate_names, nonstd_gate_unitaries=None,
                                        availability=None, geometry="line",
                                        max_idle_weight=1, maxhops=0, extra_weight_1_hops=0, extra_gate_weight=0,
                                        paramroot="H+S", sparse_lindblad_basis=False, sparse_lindblad_reps=True,
                                        verbosity=0, cache=None, idle_only=False, idt_pauli_dicts=None,
                                        algorithm="greedy", idle_op_str=((),), comm=None):
    """
    Finds a set of circuits that amplifies all the parameters of a cloud-noise model.

    Create a set of `fiducial1+germ^power+fiducial2` sequences which amplify
    all of the parameters of a :class:`CloudNoiseModel` created by passing the
    arguments of this function to :function:`create_cloudnoise_model_from_hops_and_weights`.

    Note that this function essentialy performs fiducial selection, germ
    selection, and fiducial-pair reduction simultaneously.  It is used to
    generate a short (ideally minimal) list of sequences needed for multi-
    qubit GST.

    This function allows the cloud noise model to be created by specifing
    standard gate names or additional gates as *unitary* operators.  Some
    example gate names are:

        - 'Gx','Gy','Gz' : 1Q pi/2 rotations
        - 'Gxpi','Gypi','Gzpi' : 1Q pi rotations
        - 'Gh' : Hadamard
        - 'Gp' : phase
        - 'Gcphase','Gcnot','Gswap' : standard 2Q gates

    Parameters
    ----------
    num_qubits : int
        The number of qubits

    max_lengths : list
        A list of integers specifying the different maximum lengths for germ
        powers.  Typically these values start a 1 and increase by powers of
        2, e.g. `[1,2,4,8,16]`.

    single_q_fiducials : list
        A list of gate-name-tuples, e.g. `[(), ('Gx',), ('Gy',), ('Gx','Gx')]`,
        which form a set of 1-qubit fiducials for the given model (compatible
        with both the gates it posseses and their parameterizations - for
        instance, only `[(), ('Gx',), ('Gy',)]` is needed for just Hamiltonian
        and Stochastic errors.

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

    availability : dict, optional
        A dictionary whose keys are the same gate names as in
        `gate_names` and whose values are lists of qubit-label-tuples.  Each
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
        edges, for 2Q gates of the graphy given by `geometry`.
        - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
        on any target qubits via an :class:`EmbeddingOpFactory` (uses less
        memory but slower than `"all-permutations"`.

        If a gate name (element of `gate_names`) is not present in `availability`,
        the default is `"all-edges"`.

    geometry : {"line","ring","grid","torus"} or QubitGraph
        The type of connectivity among the qubits, specifying a
        graph used to define neighbor relationships.  Alternatively,
        a :class:`QubitGraph` object may be passed directly.

    max_idle_weight : int, optional
        The maximum-weight for errors on the global idle gate.

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

    paramroot : {"CPTP", "H+S+A", "H+S", "S", "H+D+A", "D+A", "D"}
        The "root" (no trailing " terms", etc.) parameterization used for the
        cloud noise model (which specifies what needs to be amplified).

    sparse_lindblad_basis : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        `num_qubits`-qubit gates use sparse bases or not.  When sparse, these
        Lindblad gates - especially those with high-weight action or errors - take
        less memory, but their simulation is slightly slower.  Usually it's fine to
        leave this as the default (False), except when considering unusually
        high-weight errors.

    sparse_lindblad_reps : bool, optional
        Whether created Lindblad operations use sparse (more memory efficient but
        slower action) or dense representations.

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.

    cache : dict, optional
        A cache dictionary which holds template information so that repeated
        calls to `create_standard_cloudnoise_circuits` can draw on the same
        pool of templates.

    idle_only : bool, optional
        If True, only sequences for the idle germ are returned.  This is useful
        for idle tomography in particular.

    idt_pauli_dicts : tuple, optional
        A (prepDict,measDict) tuple of dicts that maps a 1-qubit Pauli basis
        string (e.g. 'X' or '-Y') to a sequence of gate *names*.  If given,
        the idle-germ fiducial pairs chosen by this function are restricted
        to those where either 1) each qubit is prepared and measured in the
        same basis or 2) each qubits is prepared and measured in different
        bases (note: '-X' and 'X" are considered the *same* basis).  This
        restriction makes the resulting sequences more like the "standard"
        ones of idle tomography, and thereby easier to interpret.

    algorithm : {"greedy","sequential"}
        The algorithm is used internall by
        :function:`_find_amped_polynomials_for_syntheticidle`.  You should leave this
        as the default unless you know what you're doing.

    idle_op_str : Circuit or tuple, optional
        The circuit or label that is used to indicate a completely
        idle layer (all qubits idle).

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    PlaquetteGridCircuitStructure
        An object holding a structured (using germ and fiducial sub-sequences)
        list of circuits.
    """

    if nonstd_gate_unitaries is None: nonstd_gate_unitaries = {}
    std_unitaries = _itgs.standard_gatename_unitaries()

    gatedict = _collections.OrderedDict()
    for name in gate_names:
        U = nonstd_gate_unitaries.get(name, std_unitaries.get(name, None))
        if U is None: raise KeyError("'%s' gate unitary needs to be provided by `nonstd_gate_unitaries` arg" % name)
        if callable(U):  # then assume a function: args -> unitary
            raise NotImplementedError("Factories are not allowed to passed to create_standard_cloudnoise_circuits yet")
        gatedict[name] = _bt.change_basis(_gt.unitary_to_process_mx(U), "std", "pp")
        # assume evotype is a densitymx or term type

    return create_cloudnoise_circuits(num_qubits, max_lengths, single_q_fiducials,
                                      gatedict, availability, geometry, max_idle_weight, maxhops,
                                      extra_weight_1_hops, extra_gate_weight, paramroot,
                                      sparse_lindblad_basis, sparse_lindblad_reps, verbosity, cache,
                                      idle_only, idt_pauli_dicts, algorithm, idle_op_str, comm)


def create_cloudnoise_circuits(num_qubits, max_lengths, single_q_fiducials,
                               gatedict, availability, geometry, max_idle_weight=1, maxhops=0,
                               extra_weight_1_hops=0, extra_gate_weight=0, paramroot="H+S",
                               sparse_lindblad_basis=False, sparse_lindblad_reps=True, verbosity=0,
                               cache=None, idle_only=False, idt_pauli_dicts=None, algorithm="greedy",
                               idle_op_str=((),), comm=None):
    """
    Finds a set of circuits that amplify all the parameters of a clould-noise model.

    Create a set of `fiducial1+germ^power+fiducial2` sequences which amplify
    all of the parameters of a `CloudNoiseModel` created by passing the
    arguments of this function to
    function:`create_cloudnoise_model_from_hops_and_weights`.

    Note that this function essentialy performs fiducial selection, germ
    selection, and fiducial-pair reduction simultaneously.  It is used to
    generate a short (ideally minimal) list of sequences needed for multi-
    qubit GST.

    Parameters
    ----------
    num_qubits : int
        The number of qubits

    max_lengths : list
        A list of integers specifying the different maximum lengths for germ
        powers.  Typically these values start a 1 and increase by powers of
        2, e.g. `[1,2,4,8,16]`.

    single_q_fiducials : list
        A list of gate-name-tuples, e.g. `[(), ('Gx',), ('Gy',), ('Gx','Gx')]`,
        which form a set of 1-qubit fiducials for the given model (compatible
        with both the gates it posseses and their parameterizations - for
        instance, only `[(), ('Gx',), ('Gy',)]` is needed for just Hamiltonian
        and Stochastic errors.

    gatedict : dict
        A dictionary whose keys are gate labels and values are gates, specifying
        the gates of the :class:`CloudNoiseModel` this function obtains circuits for.

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
        edges, for 2Q gates of the graphy given by `geometry`.
        - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
        on any target qubits via an :class:`EmbeddingOpFactory` (uses less
        memory but slower than `"all-permutations"`.

        If a gate name (a key of `gatedict`) is not present in `availability`,
        the default is `"all-edges"`.

    geometry : {"line","ring","grid","torus"} or QubitGraph
        The type of connectivity among the qubits, specifying a
        graph used to define neighbor relationships.

    max_idle_weight : int, optional
        The maximum-weight for errors on the global idle gate.

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

    paramroot : {"CPTP", "H+S+A", "H+S", "S", "H+D+A", "D+A", "D"}
        The parameterization used to define which parameters need to be
        amplified.  Note this is only the "root", e.g. you shouldn't pass
        "H+S terms" here, since the latter is implied by "H+S" when necessary.

    sparse_lindblad_basis : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        `num_qubits`-qubit gates use sparse bases or not.  When sparse, these
        Lindblad gates - especially those with high-weight action or errors - take
        less memory, but their simulation is slightly slower.  Usually it's fine to
        leave this as the default (False), except when considering unusually
        high-weight errors.

    sparse_lindblad_reps : bool, optional
        Whether created Lindblad operations use sparse (more memory efficient but
        slower action) or dense representations.

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.

    cache : dict, optional
        A cache dictionary which holds template information so that repeated
        calls to `create_cloudnoise_circuits` can draw on the same pool of
        templates.

    idle_only : bool, optional
        If True, only sequences for the idle germ are returned.  This is useful
        for idle tomography in particular.

    idt_pauli_dicts : tuple, optional
        A (prepDict,measDict) tuple of dicts that maps a 1-qubit Pauli basis
        string (e.g. 'X' or '-Y') to a sequence of gate *names*.  If given,
        the idle-germ fiducial pairs chosen by this function are restricted
        to those where either 1) each qubit is prepared and measured in the
        same basis or 2) each qubits is prepared and measured in different
        bases (note: '-X' and 'X" are considered the *same* basis).  This
        restriction makes the resulting sequences more like the "standard"
        ones of idle tomography, and thereby easier to interpret.

    algorithm : {"greedy","sequential"}
        The algorithm is used internall by
        :function:`_find_amped_polynomials_for_syntheticidle`.  You should leave this
        as the default unless you know what you're doing.

    idle_op_str : Circuit or tuple, optional
        The circuit or label that is used to indicate a completely
        idle layer (all qubits idle).

    comm : mpi4py.MPI.Comm, optional
        When not ``None``, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    PlaquetteGridCircuitStructure
        An object holding a structured (using germ and fiducial sub-sequences)
        list of circuits.
    """

    #The algorithm here takes the following basic structure:
    # - compute idle fiducial pairs with a max-weight appropriate for
    #   the true idle gate.
    # - Add the idle germ + fiducial pairs, which amplify all the "idle
    #   parameters" (the parameters of the Gi gate)
    # - precompute other idle fiducial pairs needed for 1 & 2Q synthetic
    #   idles (with maxWeight = gate-error-weight + spreading potential)
    # - To amplify the remaining parameters iterate through the "clouds"
    #   constructed by a CloudNoiseModel (these essentially give
    #   the areas of the qubit graph where non-Gi gates should act and where
    #   they aren't supposted to act but can have errors).  For each cloud
    #   we either create a new "cloud template" for it and find a set of
    #   germs and fiducial pairs (for all requested L values) such that all
    #   the parameters of gates acting on the *entire* core of the cloud
    #   are amplified (not counting any Gi parameters which are already
    #   amplified) OR we identify that the cloud is equivalent to one
    #   we already computed sequences for and just associate the cloud
    #   with the existing cloud's template; we "add it to a cloudbank".
    #   I this latter case, we compute the template sequences for any
    #   needed additional L values not already present in the template.
    # - Once there exist templates for all the clouds which support all
    #   the needed L values, we simply iterate through the cloudbanks
    #   and "tile" the template sequences, converting them to real
    #   sequences with as many clouds in parallel as possible.

    if cache is None: cache = {}
    if 'Idle gatename fidpair lists' not in cache:
        cache['Idle gatename fidpair lists'] = {}
    if 'Cloud templates' not in cache:
        cache['Cloud templates'] = _collections.defaultdict(list)
    assert(sparse_lindblad_reps is True), "Must set sparse_lindblad_reps=True` when creating cloudnoise circuits!"
    # b/c we need to avoid dense reps when using the term simulator

    ptermstype = paramroot + " terms"
    #the parameterization type used for constructing Models
    # that will be used to construct 1st order prob polynomials.

    printer = _VerbosityPrinter.create_printer(verbosity, comm)
    printer.log("Creating full model")

    if isinstance(geometry, _objs.QubitGraph):
        qubitGraph = geometry
    else:
        qubitGraph = _objs.QubitGraph.common_graph(num_qubits, geometry, directed=False)
        printer.log("Created qubit graph:\n" + str(qubitGraph))
    all_qubit_labels = qubitGraph.node_names

    model = _CloudNoiseModel.from_hops_and_weights(
        num_qubits, tuple(gatedict.keys()), None, gatedict,
        availability, None, qubitGraph,
        max_idle_weight, 0, maxhops, extra_weight_1_hops, extra_gate_weight,
        verbosity=printer - 5,
        simulator=_TermFSim(mode="taylor-order", max_order=1),
        parameterization=ptermstype,
        errcomp_type="gates",
        sparse_lindblad_basis=sparse_lindblad_basis,
        sparse_lindblad_reps=sparse_lindblad_reps)
    clouds = model.clouds
    #Note: maxSpamWeight=0 above b/c we don't care about amplifying SPAM errors (?)
    #print("DB: GATES = ",model.operation_blks['layers'].keys())
    #print("DB: CLOUDS = ",clouds)

    # clouds is a list of (core_qubits,cloud_qubits) tuples, giving the
    # different "supports" of performing the various gates in the model
    # whose parameters we want to amplify.  The 'core' of a cloud is the
    # set of qubits that have a non-trivial ideal action applied to them.
    # The 'qubits' of a cloud are all the qubits that have any action -
    # ideal or error - except that which is the same as the Gi gate.

    ideal_model = _CloudNoiseModel.from_hops_and_weights(
        num_qubits, tuple(gatedict.keys()), None, gatedict,
        availability, None, qubitGraph,
        0, 0, 0, 0, 0,
        verbosity=printer - 5,
        simulator=_MapFSim(),
        parameterization=paramroot,
        errcomp_type="gates")
    # for testing for synthetic idles - so no " terms"

    Np = model.num_params
    idle_op_str = _objs.Circuit(idle_op_str, num_lines=num_qubits)
    prepLbl = _Lbl("rho0")
    effectLbls = [_Lbl("Mdefault_%s" % l) for l in model._effect_labels_for_povm('Mdefault')]

    # create a model with max_idle_weight qubits that includes all
    # the errors of the actual n-qubit model...
    #Note: geometry doens't matter here, since we just look at the idle gate (so just use 'line'; no CNOTs)
    # - actually better to pass qubitGraph here so we get the correct qubit labels (node labels of graph)
    # - actually *don't* pass qubitGraph as this gives the wrong # of qubits when max_idle_weight < num_qubits!
    printer.log("Creating \"idle error\" model on %d qubits" % max_idle_weight)
    idle_model = _CloudNoiseModel.from_hops_and_weights(
        max_idle_weight, tuple(gatedict.keys()), None, gatedict, {}, None, 'line',  # qubitGraph
        max_idle_weight, 0, maxhops, extra_weight_1_hops,
        extra_gate_weight, verbosity=printer - 5,
        simulator=_TermFSim(mode="taylor-order", max_order=1), parameterization=ptermstype, errcomp_type="gates",
        sparse_lindblad_basis=sparse_lindblad_basis,
        sparse_lindblad_reps=sparse_lindblad_reps)
    idle_model._clean_paramvec()  # allocates/updates .gpindices of all blocks
    # these are the params we want to amplify at first...
    idle_params = idle_model.operation_blks['layers']['globalIdle'].gpindices

    if max_idle_weight in cache['Idle gatename fidpair lists']:
        printer.log("Getting cached sequences needed for max-weight=%d errors on the idle gate" % max_idle_weight)
        idle_maxwt_gatename_fidpair_lists = cache['Idle gatename fidpair lists'][max_idle_weight]
    else:
        #First get "idle germ" sequences since the idle is special
        printer.log("Getting sequences needed for max-weight=%d errors on the idle gate" % max_idle_weight)
        ampedJ, ampedJ_rank, idle_maxwt_gatename_fidpair_lists = \
            _find_amped_polynomials_for_syntheticidle(list(range(max_idle_weight)),
                                                      idle_op_str, idle_model, single_q_fiducials,
                                                      prepLbl, None, wrt_params=idle_params,
                                                      algorithm=algorithm, idt_pauli_dicts=idt_pauli_dicts,
                                                      comm=comm, verbosity=printer - 1)
        #ampedJ, ampedJ_rank, idle_maxwt_gatename_fidpair_lists = None,0,[] # DEBUG GRAPH ISO
        cache['Idle gatename fidpair lists'][max_idle_weight] = idle_maxwt_gatename_fidpair_lists

    #Since this is the idle, these max_idle_weight-qubit fidpairs can be "tiled"
    # to the n-qubits
    printer.log("%d \"idle template pairs\".  Tiling these to all %d qubits" %
                (len(idle_maxwt_gatename_fidpair_lists), num_qubits), 2)
    idle_fidpairs = _tile_idle_fidpairs(all_qubit_labels, idle_maxwt_gatename_fidpair_lists, max_idle_weight)
    printer.log("%d idle pairs found" % len(idle_fidpairs), 2)

    # Create idle sequences by sandwiching Gi^L between all idle fiducial pairs
    sequences = []
    selected_germs = [idle_op_str]
    for L in max_lengths:
        for fidpair in idle_fidpairs:
            prepFid, measFid = fidpair
            sequences.append((prepFid + idle_op_str * L + measFid, L, idle_op_str,
                              prepFid, measFid))  # was XX
            # circuit, L, germ, prepFidIndex, measFidIndex??
    printer.log("%d idle sequences (for all max-lengths: %s)" % (len(sequences), str(max_lengths)))

    if idle_only:  # Exit now when we just wanted idle-tomography sequences
        #OLD: return sequences, selected_germs

        #Post processing: convert sequence tuples to a circuit structure
        Gi_fidpairs = _collections.defaultdict(list)  # lists of fidpairs for each L value
        for _, L, _, prepFid, measFid in sequences:
            Gi_fidpairs[L].append((prepFid, measFid))

        maxPlaqEls = max([len(fidpairs) for fidpairs in Gi_fidpairs.values()])
        nMinorRows = nMinorCols = int(_np.floor(_np.sqrt(maxPlaqEls)))
        if nMinorRows * nMinorCols < maxPlaqEls: nMinorCols += 1
        if nMinorRows * nMinorCols < maxPlaqEls: nMinorRows += 1
        assert(nMinorRows * nMinorCols >= maxPlaqEls), "Logic Error!"

        germList = [idle_op_str]
        Ls = sorted(max_lengths)

        plaquettes = {}
        serial_germ = idle_op_str.serialize()  # must serialize to get correct count
        for L, fidpairs in Gi_fidpairs.items():
            power = _gsc.repeat_count_with_max_length(serial_germ, L)
            # returns 'missing_list'; useful if using dsfilter arg
            assert((L, idle_op_str) not in plaquettes), "L-values should be different!"
            plaquettes[(L, idle_op_str)] = _GermFiducialPairPlaquette(idle_op_str, power, fidpairs, None, None)

        return _objs.PlaquetteGridCircuitStructure(plaquettes, Ls, germList, "L", "germ", name=None)

    #Compute "true-idle" fidpairs for checking synthetic idle errors for 1 & 2Q gates (HARDCODED OK?)
    # NOTE: this works when ideal gates are cliffords and Gi has same type of errors as gates...
    weights = set([len(gl.sslbls) for gl in model.primitive_op_labels if (gl.sslbls is not None)])
    for gateWt in sorted(list(weights)):
        maxSyntheticIdleWt = (gateWt + extra_gate_weight) + (gateWt - 1)  # gate-error-wt + spreading potential
        maxSyntheticIdleWt = min(maxSyntheticIdleWt, num_qubits)

        for syntheticIdleWt in range(1, maxSyntheticIdleWt + 1):
            if syntheticIdleWt not in cache['Idle gatename fidpair lists']:
                printer.log("Getting sequences needed for max-weight=%d errors" % syntheticIdleWt)
                printer.log(" on the idle gate (for %d-Q synthetic idles)" % gateWt)
                sidle_model = _CloudNoiseModel.from_hops_and_weights(
                    syntheticIdleWt, tuple(gatedict.keys()), None, gatedict, {}, None, 'line',
                    max_idle_weight, 0, maxhops, extra_weight_1_hops,
                    extra_gate_weight, verbosity=printer - 5,
                    simulator=_TermFSim(mode="taylor-order", max_order=1),
                    parameterization=ptermstype, errcomp_type="gates",
                    sparse_lindblad_basis=sparse_lindblad_basis,
                    sparse_lindblad_reps=sparse_lindblad_reps)
                sidle_model._clean_paramvec()  # allocates/updates .gpindices of all blocks
                # these are the params we want to amplify...
                idle_params = sidle_model.operation_blks['layers']['globalIdle'].gpindices

                _, _, idle_gatename_fidpair_lists = _find_amped_polynomials_for_syntheticidle(
                    list(range(syntheticIdleWt)), idle_op_str, sidle_model,
                    single_q_fiducials, prepLbl, None, wrt_params=idle_params,
                    algorithm=algorithm, comm=comm, verbosity=printer - 1)
                #idle_gatename_fidpair_lists = [] # DEBUG GRAPH ISO
                cache['Idle gatename fidpair lists'][syntheticIdleWt] = idle_gatename_fidpair_lists

    #Look for and add additional germs to amplify the *rest* of the model's parameters
    Gi_nparams = model.operation_blks['layers']['globalIdle'].num_params  # assumes nqnoise (Implicit) model
    SPAM_nparams = sum([obj.num_params for obj in _itertools.chain(model.prep_blks['layers'].values(),
                                                                   model.povm_blks['layers'].values())])
    Np_to_amplify = model.num_params - Gi_nparams - SPAM_nparams
    printer.log("Idle gate has %d (amplified) params; Spam has %d (unamplifiable) params; %d gate params left" %
                (Gi_nparams, SPAM_nparams, Np_to_amplify))

    printer.log("Beginning search for non-idle germs & fiducial pairs")

    # Cloudbanks are lists of "equivalent" clouds, such that the same template
    # can be applied to all of them given a qubit mapping.  Elements of
    # `cloudbanks` are dicts with keys "template" and "clouds":
    #   - "template" is a (template_glabels, template_graph, germ_dict) tuple, where
    #      germ_dict is where all the actual germ&fidpair selection results are kept.
    #   - "clouds" is a list of (cloud_dict, template->cloud map) tuples specifying
    #      how to map the template's sequences onto the cloud (of *actual* qubits)
    cloudbanks = _collections.OrderedDict()
    for icloud, (core_qubits, cloud_qubits) in enumerate(clouds):
        cloud_dict = {'core': core_qubits, 'qubits': cloud_qubits}  # just for clarity, label the pieces

        # Collect "pure gate" params of gates that *exactly* on (just and only) the core_qubits;
        # these are the parameters we want this cloud to amplify.  If all the gates which act on
        # the core act on the entire core (when there are no gates that only act on only a part
        # of the core), then these params will be the *only* ones the choosen germs will amplify.
        # But, if there are partial-core gates, the germs might amplify some of their parameters
        # (e.g. Gx:0 params might get amplified when processing a cloud whose core is [0,1]).
        # This is fine, but we don't demand that such params be amplified, since they *must* be
        # amplified for another cloud with core exaclty equal to the gate's target qubits (e.g. [0])
        wrtParams = set()
        # OK b/c model.num_params called above
        Gi_params = set(_slct.to_array(model.operation_blks['layers']['globalIdle'].gpindices))
        pure_op_labels = []
        for gl in model.primitive_op_labels:  # take this as the set of "base"/"serial" operations
            if gl.sslbls is None: continue  # gates that act on everything (usually just the identity Gi gate)
            if set(gl.sslbls) == set(core_qubits):
                pure_op_labels.append(gl)
                wrtParams.update(_slct.to_array(model.operation_blks['cloudnoise'][gl].gpindices))
        pure_op_params = wrtParams - Gi_params  # (Gi params don't count)
        wrtParams = _slct.list_to_slice(sorted(list(pure_op_params)), array_ok=True)
        Ngp = _slct.length(wrtParams)  # number of "pure gate" params that we want to amplify

        J = _np.empty((0, Ngp), 'complex'); Jrank = 0

        printer.log("Cloud %d of %d: qubits = %s, core = %s, nparams = %d" %
                    (icloud + 1, len(clouds), str(cloud_qubits), str(core_qubits), Ngp), 2)

        # cache struture:
        #  'Idle gatename fidpair lists' - dict w/keys = ints == max-idle-weights
        #      - values = gatename-fidpair lists (on max-idle-weight qubits)
        #  'Cloud templates' - dict w/ complex cloud-class-identifying keys (tuples)
        #      - values = list of "cloud templates": (oplabels, qubit_graph, germ_dict) tuples, where
        #        oplabels is a list/set of the operation labels for this cloud template
        #        qubit_graph is a graph giving the connectivity of the cloud template's qubits
        #        germ_dict is a dict w/keys = germs
        #           - values = (germ_order, access_cache) tuples for each germ, where
        #              germ_order is an integer
        #              access_cache is a dict w/keys = "effective germ reps" = actual_reps % germ_order
        #                 - values = gatename-fidpair lists (on cloud qubits)

        def get_cloud_key(cloud, maxhops, extra_weight_1_hops, extra_gate_weight):
            """ Get the cache key we use for a cloud """
            return (len(cloud['qubits']), len(cloud['core']), maxhops, extra_weight_1_hops, extra_gate_weight)

        def map_cloud_template(cloud, oplabels, graph, template):
            """ Attempt to map `cloud` onto the cloud template `template`"""
            template_glabels, template_graph, _ = template
            #Note: number of total & core qubits should be the same,
            # since cloud is in the same "class" as template
            nCore = len(cloud['core'])
            nQubits = len(cloud['qubits'])
            template_core_graph = template_graph.subgraph(list(range(nCore)))
            template_cloud_graph = template_graph.subgraph(list(range(nQubits)))
            core_graph = graph.subgraph(cloud['core'])
            cloud_graph = graph.subgraph(cloud['qubits'])

            #Make sure each has the same number of operation labels
            if len(template_glabels) != len(oplabels):
                return None

            # Try to match core qubit labels (via oplabels & graph)
            for possible_perm in _itertools.permutations(cloud['core']):
                # possible_perm is a permutation of cloud's core labels, e.g. ('Q1','Q0','Q2')
                # such that the ordering gives the mapping from template index/labels 0 to nCore-1
                possible_template_to_cloud_map = {i: ql for i, ql in enumerate(possible_perm)}

                gr = core_graph.copy()
                for template_edge in template_core_graph.edges():
                    edge = (possible_template_to_cloud_map[template_edge[0]],
                            possible_template_to_cloud_map[template_edge[1]])
                    if gr.has_edge(edge):  # works w/directed & undirected graphs
                        gr.remove_edge(edge[0], edge[1])
                    else:
                        break  # missing edge -> possible_perm no good
                else:  # no missing templage edges!
                    if len(gr.edges()) == 0:  # and all edges were present - a match so far!

                        #Now test operation labels
                        for template_gl in template_glabels:
                            gl = template_gl.map_state_space_labels(possible_template_to_cloud_map)
                            if gl not in oplabels:
                                break
                        else:
                            #All oplabels match (oplabels can't have extra b/c we know length are the same)
                            core_map = possible_template_to_cloud_map

                            # Try to match non-core qubit labels (via graph)
                            non_core_qubits = [ql for ql in cloud['qubits'] if (ql not in cloud['core'])]
                            for possible_perm in _itertools.permutations(non_core_qubits):
                                # possible_perm is a permutation of cloud's non-core labels, e.g. ('Q4','Q3')
                                # such that the ordering gives the mapping from template index/labels nCore to nQubits-1
                                possible_template_to_cloud_map = core_map.copy()
                                possible_template_to_cloud_map.update(
                                    {i: ql for i, ql in enumerate(possible_perm, start=nCore)})
                                # now possible_template_to_cloud_map maps *all* of the qubits

                                gr = cloud_graph.copy()
                                for template_edge in template_cloud_graph.edges():
                                    edge = (possible_template_to_cloud_map[template_edge[0]],
                                            possible_template_to_cloud_map[template_edge[1]])
                                    if gr.has_edge(edge):  # works w/directed & undirected graphs
                                        gr.remove_edge(edge[0], edge[1])
                                    else:
                                        break  # missing edge -> possible_perm no good
                                else:  # no missing templage edges!
                                    if len(gr.edges()) == 0:  # and all edges were present - a match!!!
                                        return possible_template_to_cloud_map

            return None

        def create_cloud_template(cloud, pure_op_labels, graph):
            """ Creates a new cloud template, currently a (template_glabels, template_graph, germ_dict) tuple """
            nQubits = len(cloud['qubits'])
            cloud_to_template_map = {ql: i for i, ql in enumerate(
                cloud['core'])}  # core qubits always first in template
            # then non-core
            cloud_to_template_map.update(
                {ql: i for i, ql in
                 enumerate(filter(lambda x: x not in cloud['core'], cloud['qubits']), start=len(cloud['core']))}
            )
            template_glabels = [gl.map_state_space_labels(cloud_to_template_map)
                                for gl in pure_op_labels]
            template_edges = []
            cloud_graph = graph.subgraph(cloud['qubits'])
            for edge in cloud_graph.edges():
                template_edges.append((cloud_to_template_map[edge[0]],
                                       cloud_to_template_map[edge[1]]))

            template_graph = _objs.QubitGraph(list(range(nQubits)),
                                              initial_edges=template_edges,
                                              directed=graph.directed)
            cloud_template = (template_glabels, template_graph, {})
            template_to_cloud_map = {t: c for c, t in cloud_to_template_map.items()}
            return cloud_template, template_to_cloud_map

        cloud_class_key = get_cloud_key(cloud_dict, maxhops, extra_weight_1_hops, extra_gate_weight)
        cloud_class_templates = cache['Cloud templates'][cloud_class_key]
        for cloud_template in cloud_class_templates:
            template_to_cloud_map = map_cloud_template(cloud_dict, pure_op_labels, qubitGraph, cloud_template)
            if template_to_cloud_map is not None:  # a cloud template is found!
                template_glabels, template_graph, _ = cloud_template
                printer.log("Found cached template for this cloud: %d qubits, gates: %s, map: %s" %
                            (len(cloud_qubits), template_glabels, template_to_cloud_map), 2)
                break
        else:
            cloud_template, template_to_cloud_map = create_cloud_template(cloud_dict, pure_op_labels, qubitGraph)
            cloud_class_templates.append(cloud_template)
            printer.log("Created a new template for this cloud: %d qubits, gates: %s, map: %s" %
                        (len(cloud_qubits), cloud_template[0], template_to_cloud_map), 2)

        #File this cloud under the found/created "cloud template", as these identify classes of
        # "equivalent" clouds that can be tiled together below
        if id(cloud_template) not in cloudbanks:
            printer.log("Created a new cloudbank (%d) for this cloud" % id(cloud_template), 2)
            cloudbanks[id(cloud_template)] = {'template': cloud_template,
                                              'clouds': []}  # a list of (cloud_dict, template->cloud map) tuples
        else:
            printer.log("Adding this cloud to existing cloudbank (%d)" % id(cloud_template), 2)
        cloudbanks[id(cloud_template)]['clouds'].append((cloud_dict, template_to_cloud_map))

        # *** For the rest of this loop over clouds, we just make sure the identified
        #     template supports everything we need (it has germs, and fidpairs for all needed L values)

        cloud_to_template_map = {c: t for t, c in template_to_cloud_map.items()}
        germ_dict = cloud_template[2]  # see above structure
        if len(germ_dict) > 0:  # germ_dict should always be non-None
            allLsExist = all([all([
                ((_gsc.repeat_count_with_max_length(germ, L) % germ_order) in access_cache)
                for L in max_lengths])
                for germ, (germ_order, access_cache) in germ_dict.items()])
        else: allLsExist = False

        if len(germ_dict) == 0 or not allLsExist:

            if len(germ_dict) == 0:  # we need to do the germ selection using a set of candidate germs
                candidate_counts = _collections.OrderedDict([(4, 'all upto'), (5, 10), (6, 10)])
                # above maybe should be an arg? HARDCODED! -- an ordered dict so below line behaves more predictably
                candidate_germs = _get_candidates_for_core(model, core_qubits, candidate_counts, seed_start=1234)
                # candidate_germs should only use gates with support on *core* qubits?
                germ_type = "Candidate"
            else:
                # allLsExist == False, but we have the germs already (since cloud_template is not None),
                # and maybe some L-value support
                #TODO: use qubit_map to translate germ_dict keys to candidate germs
                candidate_germs = [germ.map_state_space_labels(template_to_cloud_map)
                                   for germ in germ_dict]  # just iterate over the known-good germs
                germ_type = "Pre-computed"

            consecutive_unhelpful_germs = 0
            for candidate_germ in candidate_germs:
                template_germ = candidate_germ.map_state_space_labels(cloud_to_template_map)

                #Check if we need any new L-value support for this germ
                if template_germ in germ_dict:
                    germ_order, access_cache = germ_dict[template_germ]
                    if all([((_gsc.repeat_count_with_max_length(template_germ, L) % germ_order)
                             in access_cache) for L in max_lengths]):
                        continue  # move on to the next germ

                #Let's see if we want to add this germ
                sireps = _compute_reps_for_synthetic_idle(ideal_model, candidate_germ, num_qubits, core_qubits)
                syntheticIdle = candidate_germ * sireps
                maxWt = min((len(core_qubits) + extra_gate_weight) + (len(core_qubits) - 1),
                            len(cloud_qubits))  # gate-error-wt + spreading potential
                printer.log("%s germ: %s (synthetic idle %s)" %
                            (germ_type, candidate_germ.str, syntheticIdle.str), 3)

                old_Jrank = Jrank
                printer.log("Finding amped-polys for clifford synIdle w/max-weight = %d" % maxWt, 3)
                J, Jrank, sidle_gatename_fidpair_lists = _find_amped_polynomials_for_clifford_syntheticidle(
                    cloud_qubits, core_qubits, cache['Idle gatename fidpair lists'], syntheticIdle, maxWt, model,
                    single_q_fiducials, prepLbl, effectLbls, J, Jrank, wrtParams, printer - 2)
                #J, Jrank, sidle_gatename_fidpair_lists = None, 0, None # DEBUG GRAPH ISO

                #J, Jrank, sidle_gatename_fidpair_lists = _find_amped_polynomials_for_syntheticidle(
                #    cloud_qubits, syntheticIdle, model, single_q_fiducials, prepLbl, effectLbls, J, Jrank, wrtParams)

                nNewAmpedDirs = Jrank - old_Jrank  # OLD: not nec. equal to this: len(sidle_gatename_fidpair_lists)
                if nNewAmpedDirs > 0:
                    # then there are some "directions" that this germ amplifies that previous ones didn't...
                    # assume each cloud amplifies an independent set of params
                    printer.log("Germ amplifies %d additional parameters (so %d of %d amplified for this base cloud)" %
                                (nNewAmpedDirs, Jrank, Ngp), 3)

                    if template_germ not in germ_dict:
                        germ_dict[template_germ] = (sireps, {})  # germ_order, access_cache
                    access_fidpairs_cache = germ_dict[template_germ][1]  # see above structure
                    access_fidpairs_cache[0] = sidle_gatename_fidpair_lists  # idle: effective_reps == 0

                    amped_polyJ = J[-nNewAmpedDirs:, :]  # just the rows of the Jacobian corresponding to
                    # the directions we want the current germ to amplify
                    #print("DB: amped_polyJ = ",amped_polyJ)
                    #print("DB: amped_polyJ svals = ",_np.linalg.svd(amped_polyJ, compute_uv=False))

                    #Figure out which fiducial pairs access the amplified directions at each value of L
                    for L in max_lengths:
                        reps = _gsc.repeat_count_with_max_length(candidate_germ, L)
                        if reps == 0: continue  # don't process when we don't use the germ at all...
                        effective_reps = reps % sireps
                        germPower = candidate_germ * effective_reps  # germ^effective_reps

                        if effective_reps not in access_fidpairs_cache:
                            printer.log("Finding the fiducial pairs needed to amplify %s^%d (L=%d, effreps=%d)" %
                                        (candidate_germ.str, reps, L, effective_reps), 4)
                            gatename_fidpair_lists = _get_fidpairs_needed_to_access_amped_polynomials(
                                cloud_qubits, core_qubits, germPower, amped_polyJ, sidle_gatename_fidpair_lists,
                                model, single_q_fiducials, prepLbl, effectLbls, wrtParams, printer - 3)
                            #gatename_fidpair_lists = None # DEBUG GRAPH ISO
                            printer.log("Found %d fiducial pairs" % len(gatename_fidpair_lists), 4)

                            #Convert cloud -> template gatename fidpair lists
                            template_gatename_fidpair_lists = []
                            for gatename_fidpair_list in gatename_fidpair_lists:
                                template_gatename_fidpair_lists.append([
                                    gatename_fidpair_list[cloud_qubits.index(template_to_cloud_map[tl])]
                                    for tl in range(len(cloud_qubits))])  # tl ~= "Q0" is *label* of a template qubit
                            #E.G if template qubit labels are [0,1,2] , cloud_qubits = [Q3,Q4,Q2] and map is 0->Q4,
                            # 1->Q2, 2->Q3 then we need to know what *index* Q4,Q2,Q3 are with the template, i.e the
                            # index of template_to_cloud[0], template_to_cloud[1], ... in cloud_qubits

                            access_fidpairs_cache[effective_reps] = gatename_fidpair_lists
                        else:
                            printer.log("Already found fiducial pairs needed to amplify %s^%d (L=%d, effreps=%d)" %
                                        (candidate_germ.str, reps, L, effective_reps), 4)

                    # really this will never happen b/c we'll never amplify SPAM and gauge directions...
                    if Jrank == Np:
                        break       # instead exit after we haven't seen a germ that amplifies anything new in a while
                    consecutive_unhelpful_germs = 0
                else:
                    consecutive_unhelpful_germs += 1
                    printer.log(("No additional amplified params: %d consecutive unhelpful germs."
                                 % consecutive_unhelpful_germs), 3)
                    if consecutive_unhelpful_germs == 5:  # ??
                        break  # next cloudbank
        else:
            printer.log("Fiducials for all L-values are cached!", 3)

    for icb, cloudbank in enumerate(cloudbanks.values()):
        template_glabels, template_graph, germ_dict = cloudbank['template']

        printer.log("Tiling cloudbank %d of %d: %d clouds, template labels = %s, qubits = %s" %
                    (icb + 1, len(cloudbanks), len(cloudbank['clouds']),
                     str(template_glabels), str(template_graph.nqubits)), 2)

        # At this point, we have a cloud template w/germ_dict that
        #  supports all the L-values we need.  Now tile to this
        #  cloudbank.
        for template_germ, (germ_order, access_cache) in germ_dict.items():

            printer.log("Tiling for template germ = %s" % template_germ.str, 3)
            add_germs = True
            for L in max_lengths:
                reps = _gsc.repeat_count_with_max_length(template_germ, L)
                if reps == 0: continue  # don't process when we don't use the germ at all...
                effective_reps = reps % germ_order
                template_gatename_fidpair_lists = access_cache[effective_reps]

                template_germPower = template_germ * reps  # germ^reps
                addl_seqs, addl_germs = _tile_cloud_fidpairs(template_gatename_fidpair_lists,
                                                             template_germPower, L, template_germ,
                                                             cloudbank['clouds'], all_qubit_labels)

                sequences.extend(addl_seqs)
                if add_germs:  # addl_germs is independent of L - so just add once
                    selected_germs.extend(addl_germs)
                    add_germs = False

                printer.log("After tiling L=%d to cloudbank, have %d sequences, %d germs" %
                            (L, len(sequences), len(selected_germs)), 4)

    printer.log("Done: %d sequences, %d germs" % (len(sequences), len(selected_germs)))
    #OLD: return sequences, selected_germs
    #sequences : list
    #    A list of (Circuit, L, germ, prepFid, measFid) tuples specifying the
    #    final sequences categorized by max-length (L) and germ.
    #
    #germs : list
    #    A list of Circuit objects specifying all the germs found in
    #    `sequences`.

    #Post processing: convert sequence tuples to a circuit structure
    Ls = set()
    germs = _collections.OrderedDict()

    for opstr, L, germ, prepFid, measFid in sequences:
        Ls.add(L)
        if germ not in germs: germs[germ] = {}
        if L not in germs[germ]: germs[germ][L] = []
        germs[germ][L].append((prepFid, measFid))

    maxPlaqEls = max([len(fidpairs) for gdict in germs.values() for fidpairs in gdict.values()])
    nMinorRows = nMinorCols = int(_np.floor(_np.sqrt(maxPlaqEls)))
    if nMinorRows * nMinorCols < maxPlaqEls: nMinorCols += 1
    if nMinorRows * nMinorCols < maxPlaqEls: nMinorRows += 1
    assert(nMinorRows * nMinorCols >= maxPlaqEls), "Logic Error!"

    germList = list(germs.keys())  # ordered dict so retains nice ordering
    Ls = sorted(list(Ls))
    #gss = _objs.LsGermsSerialStructure(Ls, germList, nMinorRows, nMinorCols,
    #                                   aliases=None, sequence_rules=None)

    plaquettes = {}
    for germ, gdict in germs.items():
        serial_germ = germ.serialize()  # must serialize to get correct count
        for L, fidpairs in gdict.items():
            power = _gsc.repeat_count_with_max_length(serial_germ, L)
            plaquettes[(L, germ)] = _GermFiducialPairPlaquette(germ, power, fidpairs, None, None)

    return _objs.PlaquetteGridCircuitStructure(plaquettes, Ls, germList, "L", "germ", name=None)


def _get_kcoverage_template_k2(n):
    """ Special case where k == 2 -> use hypercube construction """
    # k = 2 implies binary strings of 0's and 1's
    def bitstr(num_qubits, bit):
        """ Returns a length-num_qubits list of the values of the bit-th bit in the integers 0->num_qubits"""
        return [((i >> bit) & 1) for i in range(num_qubits)]

    def invert(bstr):
        return [(0 if x else 1) for x in bstr]

    half = [bitstr(n, k) for k in range(int(_np.ceil(_np.math.log(n, 2))))]
    other_half = [invert(bstr) for bstr in half]
    return half + other_half


def get_kcoverage_template(n, k, verbosity=0):
    """
    Get a template for how to create a "k-coverage" set of length-`n` sequences.

    Consider a set of length-`n` words from a `k`-letter alphabet.  These words
    (sequences of letters) have the "k-coverage" property if, for any choice of
    `k` different letter positions (indexed from 0 to `n-1`), every permutation
    of the `k` distinct letters (symbols) appears in those positions for at
    least one element (word) in the set.  Such a set of sequences is returned
    by this function, namely a list length-`n` lists containing the integers
    0 to `k-1`.

    This notion has application to idle-gate fiducial pair tiling, when we have
    found a set of fiducial pairs for `k` qubits and want to find a set of
    sequences on `n > k` qubits such that any subset of `k` qubits experiences
    the entire set of (`k`-qubit) fiducial pairs.  Simply take the k-coverage
    template and replace the letters (0 to `k-1`) with the per-qubit 1Q pieces
    of each k-qubit fiducial pair.

    Parameters
    ----------
    n : int
        The sequences length (see description above).

    k : int
        The coverage number (see description above).

    verbosity : int, optional
        Amount of detail to print to stdout.

    Returns
    -------
    list
        A list of length-`n` lists containing the integers 0 to `k-1`.
        The length of the outer lists depends on the particular values
        of `n` and `k` and is not guaranteed to be minimal.
    """
    #n = total number of qubits
    #indices run 0->(k-1)
    assert(n >= k), "Total number of qubits must be >= k"

    if k == 2:
        return _get_kcoverage_template_k2(n)

    #first k cols -> k! permutations of the k indices:
    cols = [list() for i in range(k)]
    for row in _itertools.permutations(range(k), k):
        for i in range(k):
            cols[i].append(row[i])
    nRows = len(cols[0])
    if verbosity > 0: print("get_template(n=%d,k=%d):" % (n, k))

    # Now add cols k to n-1:
    for a in range(k, n):  # a is index of column we're adding
        if verbosity > 1: print(" - Adding column %d: currently %d rows" % (a, nRows))

        #We know that columns 0..(a-1) satisfy the property that
        # the values of any k of them contain every permutation
        # of the integers 0..(k-1) (perhaps multiple times).  It is
        # then also true that the values of any (k-1) columns take
        # on each Perm(k,k-1) - i.e. the length-(k-1) permutations of
        # the first k integers.
        #
        # So at this point we consider all combinations of k columns
        # that include the a-th one (so really just combinations of
        # k-1 existing colums), and fill in the a-th column values
        # so that the k-columns take on each permuations of k integers.
        #

        col_a = [None] * nRows  # the new column - start with None sentinels in all current rows

        # added heuristic step for increased efficiency:
        # preference each open element of the a-th column by taking the
        # "majority vote" among what the existing column values "want"
        # the a-th column to be.
        pref_a = []
        for m in range(nRows):
            votes = _collections.defaultdict(lambda: 0)
            for existing_cols in _itertools.combinations(range(a), k - 1):
                vals = set(range(k))  # the values the k-1 existing + a-th columns need to take
                vals = vals - set([cols[i][m] for i in existing_cols])
                if len(vals) > 1: continue  # if our chosen existing cols don't
                # even cover all but one val then don't cast a vote
                assert(len(vals) == 1)
                val = vals.pop()  # pops the *only* element
                votes[val] += 1

            majority = None; majority_cnt = 0
            for ky, val in votes.items():
                if val > majority_cnt:
                    majority, majority_cnt = ky, val
            pref_a.append(majority)

        for existing_cols in _itertools.combinations(range(a - 1, -1, -1), k - 1):  # reverse-range(a) == heuristic
            if verbosity > 2: print("  - check perms are present for cols %s" % str(existing_cols + (a,)))

            #make sure cols existing_cols + [a] take on all the needed permutations
            # Since existing_cols already takes on all permuations minus the last
            # value (which is determined as it's the only one missing from the k-1
            # existing cols) - we just need to *complete* each existing row and possibly
            # duplicate + add rows to ensure all completions exist.
            for desired_row in _itertools.permutations(range(k), k):

                matching_rows = []  # rows that match desired_row on existing_cols
                open_rows = []  # rows with a-th column open (unassigned)

                for m in range(nRows):
                    if all([cols[existing_cols[i]][m] == desired_row[i] for i in range(k - 1)]):
                        # m-th row matches desired_row on existing_cols
                        matching_rows.append(m)
                    if col_a[m] is None:
                        open_rows.append(m)

                if verbosity > 3: print("   - perm %s: %d rows, %d match perm, %d open"
                                        % (str(desired_row), nRows, len(matching_rows), len(open_rows)))
                v = {'value': desired_row[k - 1], 'alternate_rows': matching_rows}
                placed = False

                #Best: find a row that already has the value we're looking for (set via previous iteration)
                for m in matching_rows:
                    if col_a[m] and col_a[m]['value'] == desired_row[k - 1]:
                        # a perfect match! - no need to take an open slot
                        updated_alts = [i for i in col_a[m]['alternate_rows'] if i in matching_rows]
                        if verbosity > 3: print("    -> existing row (index %d) perfectly matches!" % m)
                        col_a[m]['alternate_rows'] = updated_alts; placed = True; break
                if placed: continue

                #Better: find an open row that prefers the value we want to place in it
                for m in matching_rows:
                    # slot is open & prefers the value we want to place in it - take it!
                    if col_a[m] is None and pref_a[m] == desired_row[k - 1]:
                        if verbosity > 3: print("    -> open preffered row (index %d) matches!" % m)
                        col_a[m] = v; placed = True; break
                if placed: continue

                #Good: find any open row (FUTURE: maybe try to shift for preference first?)
                for m in matching_rows:
                    if col_a[m] is None:  # slot is open - take it!
                        if verbosity > 3: print("    -> open row (index %d) matches!" % m)
                        col_a[m] = v; placed = True; break
                if placed: continue

                # no open slots
                # option1: (if there are any open rows)
                #  Look to swap an existing value in a matching row
                #   to an open row allowing us to complete the matching
                #   row using the current desired_row.
                open_rows = set(open_rows)  # b/c use intersection below
                shift_soln_found = False
                if len(open_rows) > 0:
                    for m in matching_rows:
                        # can assume col_a[m] is *not* None given above logic
                        ist = open_rows.intersection(col_a[m]['alternate_rows'])
                        if len(ist) > 0:
                            m2 = ist.pop()  # just get the first element
                            # move value in row m to m2, then put v into the now-open m-th row
                            col_a[m2] = col_a[m]
                            col_a[m] = v
                            if verbosity > 3: print("    -> row %d >> row %d, and row %d matches!" % (m, m2, m))
                            shift_soln_found = True
                            break

                if not shift_soln_found:
                    # no shifting can be performed to place v into an open row,
                    # so we just create a new row equal to desired_row on existing_cols.
                    # How do we choose the non-(existing & last) colums? For now, just
                    # replicate the first element of matching_rows:
                    if verbosity > 3: print("    -> creating NEW row.")
                    for i in range(a):
                        cols[i].append(cols[i][matching_rows[0]])
                    col_a.append(v)
                    nRows += 1

        #Check for any remaining open rows that we never needed to use.
        # (the a-th column can then be anything we want, so as heuristic
        #  choose a least-common value in the row already)
        for m in range(nRows):
            if col_a[m] is None:
                cnts = {v: 0 for v in range(k)}  # count of each possible value
                for i in range(a): cnts[cols[i][m]] += 1
                val = 0; mincnt = cnts[0]
                for v, cnt in cnts.items():  # get value with minimal count
                    if cnt < mincnt:
                        val = v; mincnt = cnt
                col_a[m] = {'value': val, 'alternate_rows': "N/A"}

        # a-th column is complete; "cement" it by replacing
        # value/alternative_rows dicts with just the values
        col_a = [d['value'] for d in col_a]
        cols.append(col_a)

    #convert cols to "strings" (rows)
    assert(len(cols) == n)
    rows = []
    for i in range(len(cols[0])):
        rows.append([cols[j][i] for j in range(n)])

    if verbosity > 0: print(" Done: %d rows total" % len(rows))
    return rows


def _check_kcoverage_template(rows, n, k, verbosity=0):
    """
    Check that k-coverage conditions are met.

    Verify that `rows` satisfies the `k`-coverage conditions for length-`n`
    sequences.  Raises an AssertionError if the check fails.

    Parameters
    ----------
    rows : list
        A list of k-coverage words.  The same as whas is returned by
        :function:`get_kcoverage_template`.

    n : int
        The sequences length.

    k : int
        The coverage number.

    verbosity : int, optional
        Amount of detail to print to stdout.

    Returns
    -------
    None
    """
    if verbosity > 0: print("check_template(n=%d,k=%d)" % (n, k))

    #for each set of k qubits (of the total n qubits)
    for cols_to_check in _itertools.combinations(range(n), k):
        if verbosity > 1: print(" - checking cols %s" % str(cols_to_check))
        for perm in _itertools.permutations(range(k), k):
            for m, row in enumerate(rows):
                if all([row[i] == perm[i] for i in range(k)]):
                    if verbosity > 2: print("  - perm %s: found at row %d" % (str(perm), m))
                    break
            else:
                assert(False), \
                    "Permutation %s on qubits (cols) %s is not present!" % (str(perm), str(cols_to_check))
    if verbosity > 0: print(" check succeeded!")


def _filter_nqubit_circuittuple(sequence_tuples, sectors_to_keep,
                                new_sectors=None, idle='Gi'):
    """
    Filters a list of n-qubit circuit tuples.

    Creates a new set of qubit sequences-tuples that is the restriction of
    `sequence_tuples` to the sectors identified by `sectors_to_keep`.

    More specifically, this function removes any operation labels which act
    specifically on sectors not in `sectors_to_keep` (e.g. an idle gate acting
    on *all* sectors because it's `.sslbls` is None will *not* be removed --
    see :function:`filter_circuit` for details).  Non-empty sequences for
    which all labels are removed in the *germ* are not included in the output
    (as these correspond to an irrelevant germ).

    A typical case is when the state-space is that of *n* qubits, and the
    state space labels the intergers 0 to *n-1*.  One may want to "rebase" the
    indices to 0 in the returned data set using `new_sectors`
    (E.g. `sectors_to_keep == [4,5,6]` and `new_sectors == [0,1,2]`).

    Parameters
    ----------
    sequence_tuples : list
        A list of (circuit, L, germ, prepfid, measfid) tuples giving the
        sequences to process.

    sectors_to_keep : list or tuple
        The state-space labels (strings or integers) of the "sectors" to keep in
        the returned tuple list.

    new_sectors : list or tuple, optional
        New sectors names to map the elements of `sectors_to_keep` onto in the
        output DataSet's circuits.  None means the labels are not renamed.
        This can be useful if, for instance, you want to run a 2-qubit protocol
        that expects the qubits to be labeled "0" and "1" on qubits "4" and "5"
        of a larger set.  Simply set `sectors_to_keep == [4,5]` and
        `new_sectors == [0,1]`.

    idle : string or Label, optional
        The operation label to be used when there are no kept components of a
        "layer" (element) of a circuit.

    Returns
    -------
    filtered_sequence_tuples : list
        A list of tuples with the same structure as `sequence tuples`.
    """
    ret = []
    for opstr, L, germ, prepfid, measfid in sequence_tuples:
        new_germ = _gsc.filter_circuit(germ, sectors_to_keep, new_sectors, idle)
        if len(new_germ) > 0 or len(opstr) == 0:
            new_prep = _gsc.filter_circuit(prepfid, sectors_to_keep, new_sectors, idle)
            new_meas = _gsc.filter_circuit(measfid, sectors_to_keep, new_sectors, idle)
            new_gstr = _gsc.filter_circuit(opstr, sectors_to_keep, new_sectors, idle)
            ret.append((new_gstr, L, new_germ, new_prep, new_meas))

    return ret


#Utility functions
def _gatename_fidpair_list_to_fidpairs(gatename_fidpair_list):
    """
    Converts a fiducial pair list between formats.

    Converts a "gatename fiducial pair list" to a standard list of 2-tuples
    of :class:`Circuit` objects.  This format is used internally for storing
    fiducial circuits containing only *single-qubit* gates.

    A "gatename fiducial pair list" is a list with one element per fiducial
    pair.  Each element is itself a list of `(prep_names, meas_names)` tuples,
    one per *qubit*.  `prep_names` and `meas_names` are tuples of simple strings
    giving the names of the (1-qubit) gates acting on the respective qubit.  The
    qubit labels for the output circuits are taken to be the integers starting
    at 0.

    For example, the input:
    `[ [ (('Gx','Gx'),('Gy',)),(('Gz','Gz'),()) ] ]`
    would result in:
    `[ ( Circuit(Gx:0Gx:0Gz:1Gz:1), Circuit(Gy:0) ) ]`

    Parameters
    ----------
    gatename_fidpair_list : list
        Each element corresponds to one (prep, meas) pair of circuits, and is
        a list of `(prep_names, meas_names)` tuples, on per qubit.

    Returns
    -------
    list
        A list of `(prep_fiducial, meas_fiducial)` pairs, where `prep_fiducial`
        and `meas_fiducial` are :class:`Circuit` objects.
    """
    fidpairs = []
    for gatenames_per_qubit in gatename_fidpair_list:
        prepStr = []
        measStr = []
        nQubits = len(gatenames_per_qubit)
        for iQubit, gatenames in enumerate(gatenames_per_qubit):
            prepnames, measnames = gatenames
            prepStr.extend([_Lbl(name, iQubit) for name in prepnames])
            measStr.extend([_Lbl(name, iQubit) for name in measnames])
        fidpair = (_objs.Circuit(prepStr, num_lines=nQubits),
                   _objs.Circuit(measStr, num_lines=nQubits))
        fidpairs.append(fidpair)
    return fidpairs


def _fidpairs_to_gatename_fidpair_list(fidpairs, num_qubits):
    """
    The inverse of :function:`_gatename_fidpair_list_to_fidpairs`.

    Converts a list of `(prep,meas)` pairs of fiducial circuits (containing
    only single-qubit gates!) to the "gatename fiducial pair list" format,
    consisting of per-qubit lists of gate names (see docstring for
    :function:`_gatename_fidpair_list_to_fidpairs` for mor details).

    Parameters
    ----------
    fidpairs : list
        A list of `(prep_fiducial, meas_fiducial)` pairs, where `prep_fiducial`
        and `meas_fiducial` are :class:`Circuit` objects.

    num_qubits : int
        The number of qubits.  Qubit labels within `fidpairs` are assumed to
        be the integers from 0 to `num_qubits-1`.

    Returns
    -------
    gatename_fidpair_list : list
        Each element corresponds to an elmeent of `fidpairs`, and is a list of
        `(prep_names, meas_names)` tuples, on per qubit.  `prep_names` and
        `meas_names` are tuples of single-qubit gate *names* (strings).
    """
    gatename_fidpair_list = []
    for fidpair in fidpairs:
        gatenames_per_qubit = [(list(), list()) for i in range(num_qubits)]  # prepnames, measnames for each qubit
        prepStr, measStr = fidpair

        for lbl in prepStr:
            assert(len(lbl.sslbls) == 1), "Can only convert strings with solely 1Q gates"
            gatename = lbl.name
            iQubit = lbl.sslbls[0]
            gatenames_per_qubit[iQubit][0].append(gatename)

        for lbl in measStr:
            assert(len(lbl.sslbls) == 1), "Can only convert strings with solely 1Q gates"
            gatename = lbl.name
            iQubit = lbl.sslbls[0]
            gatenames_per_qubit[iQubit][1].append(gatename)

        #Convert lists -> tuples
        gatenames_per_qubit = tuple([(tuple(x[0]), tuple(x[1])) for x in gatenames_per_qubit])
        gatename_fidpair_list.append(gatenames_per_qubit)
    return gatename_fidpair_list


@_deprecated_fn("the pre-build SMQ modelpacks under `pygsti.modelpacks`")
def stdmodule_to_smqmodule(std_module):
    """
    Converts a pyGSTi "standard module" to a "standard multi-qubit module".

    PyGSTi provides a number of 1- and 2-qubit models corrsponding to commonly
    used gate sets, along with related meta-information.  Each such
    model+metadata is stored in a "standard module" beneath `pygsti.construction`
    (e.g. `pygsti.construction.std1Q_XYI` is the standard module for modeling a
    single-qubit quantum processor which can perform X(pi/2), Y(pi/2) and idle
    operations).  Because they deal with just 1- and 2-qubit models, multi-qubit
    labelling conventions are not used to improve readability.  For example, a
    "X(pi/2)" gate is labelled "Gx" (in a 1Q context) or "Gix" (in a 2Q context)
    rather than "Gx:0" or "Gx:1" respectively.

    There are times, however, when you many *want* a standard module with this
    multi-qubit labelling convention (e.g. performing 1Q-GST on the 3rd qubit
    of a 5-qubit processor).  We call such a module a standard *multi-qubit*
    module, and these typically begin with `"smq"` rather than `"std"`.

    Standard multi-qubit modules are *created* by this function.  For example,
    If you want the multi-qubit version of `pygsti.modelpacks.legacy.std1Q_XYI`
    you must:

    1. import `std1Q_XYI` (`from pygsti.modelpacks.legacy import std1Q_XYI`)
    2. call this function (i.e. `stdmodule_to_smqmodule(std1Q_XYI)`)
    3. import `smq1Q_XYI` (`from pygsti.modelpacks.legacy import smq1Q_XYI`)

    The `smq1Q_XYI` module will look just like the `std1Q_XYI` module but use
    multi-qubit labelling conventions.

    .. deprecated:: v0.9.9
        `stdmodule_to_smqmodule` will be removed in future versions of
        pyGSTi. Instead, import pre-built SMQ modelpacks directly from
        `pygsti.modelpacks`.

    Parameters
    ----------
    std_module : Module
        The standard module to convert to a standard-multi-qubit module.

    Returns
    -------
    Module
        The new module, although it's better to import this using the appropriate
        "smq"-prefixed name as described above.
    """
    from types import ModuleType as _ModuleType
    import sys as _sys
    import importlib

    std_module_name_parts = std_module.__name__.split('.')
    std_module_name_parts[-1] = std_module_name_parts[-1].replace('std', 'smq')
    new_module_name = '.'.join(std_module_name_parts)

    try:
        return importlib.import_module(new_module_name)
    except ImportError:
        pass  # ok, this is what the rest of the function is for

    out_module = {}
    std_target_model = std_module.target_model()  # could use ._target_model to save a copy
    dim = std_target_model.dim
    if dim == 4:
        sslbls = [0]
        find_replace_labels = {'Gi': (), 'Gx': ('Gx', 0), 'Gy': ('Gy', 0),
                               'Gz': ('Gz', 0), 'Gn': ('Gn', 0)}
        find_replace_strs = [((oldgl,), (newgl,)) for oldgl, newgl
                             in find_replace_labels.items()]
    elif dim == 16:
        sslbls = [0, 1]
        find_replace_labels = {'Gii': (),
                               'Gxi': ('Gx', 0), 'Gyi': ('Gy', 0), 'Gzi': ('Gz', 0),
                               'Gix': ('Gx', 1), 'Giy': ('Gy', 1), 'Giz': ('Gz', 1),
                               'Gxx': ('Gxx', 0, 1), 'Gxy': ('Gxy', 0, 1),
                               'Gyx': ('Gxy', 0, 1), 'Gyy': ('Gyy', 0, 1),
                               'Gcnot': ('Gcnot', 0, 1), 'Gcphase': ('Gcphase', 0, 1)}
        find_replace_strs = [((oldgl,), (newgl,)) for oldgl, newgl
                             in find_replace_labels.items()]
        #find_replace_strs.append( (('Gxx',), (('Gx',0),('Gx',1))) )
        #find_replace_strs.append( (('Gxy',), (('Gx',0),('Gy',1))) )
        #find_replace_strs.append( (('Gyx',), (('Gy',0),('Gx',1))) )
        #find_replace_strs.append( (('Gyy',), (('Gy',0),('Gy',1))) )
    else:
        #TODO: add qutrit?
        raise ValueError("Unsupported model dimension: %d" % dim)

    def upgrade_dataset(ds):
        """
        Update DataSet `ds` in-place to use  multi-qubit style labels.
        """
        ds.process_circuits_inplace(lambda s: _gsc.manipulate_circuit(
            s, find_replace_strs, sslbls))

    out_module['find_replace_gatelabels'] = find_replace_labels
    out_module['find_replace_circuits'] = find_replace_strs
    out_module['upgrade_dataset'] = upgrade_dataset

    # gate names
    out_module['gates'] = [find_replace_labels.get(nm, nm) for nm in std_module.gates]

    #Fully-parameterized target model (update labels)
    new_target_model = _objs.ExplicitOpModel(sslbls, std_target_model.basis.copy())
    new_target_model._evotype = std_target_model._evotype
    new_target_model._default_gauge_group = std_target_model._default_gauge_group

    for lbl, obj in std_target_model.preps.items():
        new_lbl = find_replace_labels.get(lbl, lbl)
        new_target_model.preps[new_lbl] = obj.copy()
    for lbl, obj in std_target_model.povms.items():
        new_lbl = find_replace_labels.get(lbl, lbl)
        new_target_model.povms[new_lbl] = obj.copy()
    for lbl, obj in std_target_model.operations.items():
        new_lbl = find_replace_labels.get(lbl, lbl)
        new_target_model.operations[new_lbl] = obj.copy()
    for lbl, obj in std_target_model.instruments.items():
        new_lbl = find_replace_labels.get(lbl, lbl)
        new_target_model.instruments[new_lbl] = obj.copy()
    out_module['_target_model'] = new_target_model

    # _stdtarget and _gscache need to be *locals* as well so target_model(...) works
    _stdtarget = importlib.import_module('.stdtarget', 'pygsti.construction')
    _gscache = {("full", "auto"): new_target_model}
    out_module['_stdtarget'] = _stdtarget
    out_module['_gscache'] = _gscache

    def target_model(parameterization_type="full", simulator="auto"):
        """
        Returns a copy of the target model in the given parameterization.

        Parameters
        ----------
        parameterization_type : {"TP", "CPTP", "H+S", "S", ... }
            The gate and SPAM vector parameterization type. See
            :function:`Model.set_all_parameterizations` for all allowed values.

        simulator : ForwardSimulator or {"auto", "matrix", "map"}
            The simulator (or type) to be used for model calculations (leave as
            "auto" if you're not sure what this is).

        Returns
        -------
        Model
        """
        return _stdtarget._copy_target(_sys.modules[new_module_name], parameterization_type,
                                       simulator, _gscache)
    out_module['target_model'] = target_model

    # circuit lists
    circuitlist_names = ['germs', 'germs_lite', 'prepStrs', 'effectStrs', 'fiducials']
    for nm in circuitlist_names:
        if hasattr(std_module, nm):
            out_module[nm] = _gsc.manipulate_circuits(getattr(std_module, nm), find_replace_strs, sslbls)

    # clifford compilation (keys are lists of operation labels)
    if hasattr(std_module, 'clifford_compilation'):
        new_cc = _collections.OrderedDict()
        for ky, val in std_module.clifford_compilation.items():
            new_val = [find_replace_labels.get(lbl, lbl) for lbl in val]
            new_cc[ky] = new_val

    passthrough_names = ['global_fidPairs', 'pergerm_fidPairsDict', 'global_fidPairs_lite', 'pergerm_fidPairsDict_lite']
    for nm in passthrough_names:
        if hasattr(std_module, nm):
            out_module[nm] = getattr(std_module, nm)

    #Create the new module
    new_module = _ModuleType(str(new_module_name))  # str(.) converts to native string for Python 2 compatibility
    for k, v in out_module.items():
        setattr(new_module, k, v)
    _sys.modules[new_module_name] = new_module
    return new_module
