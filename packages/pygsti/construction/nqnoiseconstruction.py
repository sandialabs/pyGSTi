""" Defines classes which represent gates, as well as supporting functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import collections as _collections
import itertools as _itertools
import numpy as _np
import scipy as _scipy
import scipy.sparse as _sps
import warnings as _warnings

from .. import objects as _objs
from ..tools import basistools as _bt
from ..tools import matrixtools as _mt
from ..tools import gatetools as _gt
from ..tools import slicetools as _slct
from ..tools import listtools as _lt
from ..objects.labeldicts import StateSpaceLabels as _StateSpaceLabels

from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from ..baseobjs.basisconstructors import sqrt2, id2x2, sigmax, sigmay, sigmaz
from ..baseobjs import Basis as _Basis
from ..baseobjs import Dim as _Dim
from ..baseobjs import Label as _Lbl

from . import gatestringconstruction as _gsc
from .gatesetconstruction import basis_build_vector as _basis_build_vector

RANK_TOL = 1e-9

def _iter_basis_inds(weight):
    """ Iterate over product of `weight` non-identity Pauli 1Q basis indices """
    basisIndList = [ [1,2,3] ]*weight #assume pauli 1Q basis, and only iterate over non-identity els
    for basisInds in _itertools.product(*basisIndList):
        yield basisInds

def basisProductMatrix(sigmaInds, sparse):
    """ Construct the Pauli product matrix from the given `sigmaInds` """
    sigmaVec = (id2x2/sqrt2, sigmax/sqrt2, sigmay/sqrt2, sigmaz/sqrt2)
    M = _np.identity(1,'complex')
    for i in sigmaInds:
        M = _np.kron(M,sigmaVec[i])
    return _sps.csr_matrix(M) if sparse else M

def nparams_nqnoise_gateset(nQubits, geometry="line", maxIdleWeight=1, maxhops=0,
                            extraWeight1Hops=0, extraGateWeight=0, requireConnected=False,
                            independent1Qgates=True, ZZonly=False, verbosity=0):
    """
    Returns the number of parameters in the :class:`GateSet` that would be given
    by a call to :function:`build_nqnoise_gateset` with the same parameters, 
    without actually constructing the gate set (useful for considering
    parameter-count scaling).

    Parameters
    ----------
    Same as :function:`build_nqnoise_gateset`
    
    Returns
    -------
    int
    """
    # noise can be either a seed or a random array that is long enough to use

    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("Computing parameters for a %d-qubit %s gateset" % (nQubits,geometry))

    qubitGraph = _objs.QubitGraph.common_graph(nQubits, geometry)
    #printer.log("Created qubit graph:\n"+str(qubitGraph))

    def idle_count_nparams(maxWeight):
        """Parameter count of a `build_nqn_global_idle`-constructed gate"""
        ret = 0
        possible_err_qubit_inds = _np.arange(nQubits)
        for wt in range(1,maxWeight+1):
            nErrTargetLocations = qubitGraph.connected_combos(possible_err_qubit_inds,wt)
            if ZZonly and wt > 1: basisSizeWoutId = 1**wt # ( == 1)
            else: basisSizeWoutId = 3**wt # (X,Y,Z)^wt
            nErrParams = 2*basisSizeWoutId # H+S terms
            ret += nErrTargetLocations * nErrParams
        return ret

    def gate_count_nparams(target_qubit_inds,weight_maxhops_tuples,debug=False):
        """Parameter count of a `build_nqn_composed_gate`-constructed gate"""
        ret = 0
        #Note: no contrib from idle noise (already parameterized)
        for wt, maxHops in weight_maxhops_tuples:
            possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops),_np.int64)
            if requireConnected:
                nErrTargetLocations = qubitGraph.connected_combos(possible_err_qubit_inds,wt)
            else:
                nErrTargetLocations = _scipy.misc.comb(len(possible_err_qubit_inds),wt) #matches actual initial stud
            if ZZonly and wt > 1: basisSizeWoutId = 1**wt # ( == 1)
            else: basisSizeWoutId = 3**wt # (X,Y,Z)^wt
            nErrParams = 2*basisSizeWoutId # H+S terms
            if debug:
                print(" -- wt%d, hops%d: inds=%s locs = %d, eparams=%d, total contrib = %d" %
                      (wt,maxHops,str(possible_err_qubit_inds),nErrTargetLocations,nErrParams,nErrTargetLocations*nErrParams))
            ret += nErrTargetLocations * nErrParams
        return ret

    nParams = _collections.OrderedDict()

    printer.log("Creating Idle:")                
    nParams[_Lbl('Gi')] = idle_count_nparams(maxIdleWeight)
     
    #1Q gates: X(pi/2) & Y(pi/2) on each qubit
    weight_maxhops_tuples_1Q = [(1,maxhops+extraWeight1Hops)] + \
                               [ (1+x,maxhops) for x in range(1,extraGateWeight+1) ]

    if independent1Qgates:
        for i in range(nQubits):
            printer.log("Creating 1Q X(pi/2) and Y(pi/2) gates on qubit %d!!" % i)
            nParams[_Lbl("Gx",i)] = gate_count_nparams((i,), weight_maxhops_tuples_1Q)
            nParams[_Lbl("Gy",i)] = gate_count_nparams((i,), weight_maxhops_tuples_1Q)
    else:
        printer.log("Creating common 1Q X(pi/2) and Y(pi/2) gates")
        rep = int(nQubits / 2)
        nParams[_Lbl("Gxrep")] = gate_count_nparams((rep,), weight_maxhops_tuples_1Q)
        nParams[_Lbl("Gyrep")] = gate_count_nparams((rep,), weight_maxhops_tuples_1Q)

    #2Q gates: CNOT gates along each graph edge
    weight_maxhops_tuples_2Q = [(1,maxhops+extraWeight1Hops),(2,maxhops)] + \
                               [ (2+x,maxhops) for x in range(1,extraGateWeight+1) ]
    for i,j in qubitGraph.edges(): #note: all edges have i<j so "control" of CNOT is always lower index (arbitrary)
        printer.log("Creating CNOT gate between qubits %d and %d!!" % (i,j))
        nParams[_Lbl("Gcnot",(i,j))] = gate_count_nparams((i,j), weight_maxhops_tuples_2Q)

    #SPAM
    nPOVM_1Q = 4 # params for a single 1Q POVM
    nParams[_Lbl('rho0')] = 3*nQubits # 3 b/c each component is TP
    nParams[_Lbl('Mdefault')] = nPOVM_1Q * nQubits # nQubits 1Q-POVMs

    return nParams, sum(nParams.values())



def build_nqnoise_gateset(nQubits, geometry="line", cnot_edges=None,
                          maxIdleWeight=1, maxhops=0,
                          extraWeight1Hops=0, extraGateWeight=0, sparse=False,
                          gateNoise=None, prepNoise=None, povmNoise=None,
                          sim_type="matrix", parameterization="H+S",
                          addIdleNoiseToAllGates=True,
                          return_clouds=False, verbosity=0): #, debug=False):
    """ 
    Create a noisy n-qubit gateset using a low-weight and geometrically local
    error model with a common idle gate.  

    This type of gateset is generally useful for performing GST on a multi-
    qubit gateset, whereas functions like :function:`build_nqubit_gateset`
    are more useful for creating static (non-parameterized) gate sets.

    Parameters
    ----------
    nQubits : int
        The number of qubits
    
    geometry : {"line","ring","grid","torus"} or QubitGraph
        The type of connectivity among the qubits, specifying a
        graph used to define neighbor relationships.  Alternatively,
        a :class:`QubitGraph` object with 0-`nQubits-1` node labels
        may be passed directly.

    cnot_edges : list, optional
        A list of 2-tuples of (control,target) qubit indices for each
        CNOT gate to be included in the returned GateSet.  If None, then
        the (directed) edges of the `geometry` graph are used.

    maxIdleWeight : int, optional
        The maximum-weight for errors on the global idle gate.

    maxhops : int
        The locality constraint: for a gate, errors (of weight up to the
        maximum weight for the gate) are allowed to occur on the gate's
        target qubits and those reachable by hopping at most `maxhops` times
        from a target qubit along nearest-neighbor links (defined by the 
        `geometry`).  
    
    extraWeight1Hops : int, optional
        Additional hops (adds to `maxhops`) for weight-1 errors.  A value > 0
        can be useful for allowing just weight-1 errors (of which there are 
        relatively few) to be dispersed farther from a gate's target qubits.
        For example, a crosstalk-detecting model might use this.
    
    extraGateWeight : int, optional
        Addtional weight, beyond the number of target qubits (taken as a "base
        weight" - i.e. weight 2 for a 2Q gate), allowed for gate errors.  If
        this equals 1, for instance, then 1-qubit gates can have up to weight-2
        errors and 2-qubit gates can have up to weight-3 errors.

    sparse : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        `nQubits`-qubit gates are sparse or not.  (This is determied by whether
        they are constructed using sparse basis matrices.)  When sparse, these
        Lindblad gates take up less memory, but their action is slightly slower.
        Usually it's fine to leave this as the default (False), except when
        considering particularly high-weight terms (b/c then the Lindblad gates
        are higher dimensional and sparsity has a significant impact).

    gateNoise, prepNoise, povmNoise : tuple or numpy.ndarray, optional
        If not None, noise to place on the gates, the state prep and the povm.
        This can either be a `(seed,strength)` 2-tuple, or a long enough numpy
        array (longer than what is needed is OK).  These values specify random
        `gate.from_vector` initializatin for the gates and random depolarization
        amounts on the preparation and POVM.

    sim_type : {"auto","matrix","map","termorder:<N>"}
        The type of forward simulation (probability computation) to use for the
        returned :class:`GateSet`.  That is, how should the gate set compute
        gate string/circuit probabilities when requested.  `"matrix"` is better
        for small numbers of qubits, `"map"` is better for larger numbers. The
        `"termorder"` option is designed for even larger numbers.  Usually,
        the default of `"auto"` is what you want.
    
    parameterization : {"H+S", "H+S terms", "H+S clifford terms"}
        The type of parameterizaton to use in the constructed gate set.  The
        default of `"H+S"` performs usual density-matrix evolution to compute
        circuit probabilities.  The other "terms" options compute probabilities
        using a path-integral approach designed for larger numbers of qubits,
        and are considered expert options.

    addIdleNoiseToAllGates: bool, optional
        Whether the global idle should be added as a factor following the 
        ideal action of each of the non-idle gates.

    return_clouds : bool, optional
        Whether to return a dictionary of "cloud" objects, used for constructing
        the gate sequences necessary for probing the returned GateSet's
        parameters.  Used primarily internally within pyGSTi.

    verbosity : int, optional
        An integer >= 0 dictating how must output to send to stdout.

    Returns
    -------
    GateSet
    """

    assert(parameterization in ("H+S","H+S terms","H+S clifford terms"))
    if sim_type == "auto":
        if parameterization in ("H+S terms", "H+S clifford terms"): sim_type = "termorder:1"
        else: sim_type = "map" if nQubits > 2 else "matrix"

    assert(sim_type in ("matrix","map") or sim_type.startswith("termorder"))

    from pygsti.construction import std1Q_XY # the base gate set for 1Q gates
    from pygsti.construction import std2Q_XYICNOT # the base gate set for 2Q (CNOT) gate
    
    # noise can be either a seed or a random array that is long enough to use

    printer = _VerbosityPrinter.build_printer(verbosity)
    geometry_name = "custom" if isinstance(geometry, _objs.QubitGraph) else geometry
    printer.log("Creating a %d-qubit %s gateset" % (nQubits,geometry_name))

    gs = _objs.GateSet(sim_type=sim_type, default_param=parameterization) # no preps/POVMs
    gs.stateSpaceLabels = _StateSpaceLabels(list(range(nQubits))) #TODO: better way of setting this... (__init__?)
    
    # OLD: sim_type = "map" if sparse else "matrix") # no preps/POVMs
    
    # TODO: sparse prep & effect vecs... acton(...) analogue?

    #Full preps & povms -- maybe another option
    ##Create initial gate set with std prep & POVM
    #eLbls = []; eExprs = []
    #formatStr = '0' + str(nQubits) + 'b'
    #for i in range(2**nQubits):
    #    eLbls.append( format(i,formatStr))
    #    eExprs.append( str(i) )    
    #Qlbls = tuple( ['Q%d' % i for i in range(nQubits)] )
    #gs = pygsti.construction.build_gateset(
    #    [2**nQubits], [Qlbls], [], [], 
    #    effectLabels=eLbls, effectExpressions=eExprs)
    printer.log("Created initial gateset")

    if isinstance(geometry, _objs.QubitGraph):
        qubitGraph = geometry
    else:
        qubitGraph = _objs.QubitGraph.common_graph(nQubits, geometry, directed=False)
        printer.log("Created qubit graph:\n"+str(qubitGraph))

    if cnot_edges is None:
        cnot_edges = qubitGraph.edges(double_for_undirected=True)

    if maxIdleWeight > 0:
        printer.log("Creating Idle:")
        gs.gates[_Lbl('Gi')] = build_nqn_global_idle(qubitGraph, maxIdleWeight, sparse,
                                                     sim_type, parameterization, printer-1)
        idleOP = gs.gates['Gi']
    else:
        gs._dim = 4**nQubits # TODO: make a set_dim that does this (and is used in labeldicts.py)
        if gs._sim_type == "auto":
            gs.set_simtype("auto") # run deferred auto-simtype now that _dim is set

        idleOP = False
        
    if not addIdleNoiseToAllGates:
        idleOP = False # then never add global idle noise.

    # a dictionary of "cloud" objects
    # keys = (target_qubit_indices, cloud_qubit_indices) tuples
    # values = list of gate-labels giving the gates associated with that cloud (necessary?)
    clouds = _collections.OrderedDict()
        
    #1Q gates: X(pi/2) & Y(pi/2) on each qubit
    Gx = std1Q_XY.gs_target.gates['Gx']
    Gy = std1Q_XY.gs_target.gates['Gy'] 
    weight_maxhops_tuples_1Q = [(1,maxhops+extraWeight1Hops)] + \
                               [ (1+x,maxhops) for x in range(1,extraGateWeight+1) ]
    cloud_maxhops = max( [mx for wt,mx in weight_maxhops_tuples_1Q] ) # max of max-hops
    
    for i in range(nQubits):        
        printer.log("Creating 1Q X(pi/2) gate on qubit %d!!" % i)
        gs.gates[_Lbl("Gx",i)] = build_nqn_composed_gate(
            Gx, (i,), qubitGraph, weight_maxhops_tuples_1Q,
            idle_noise=idleOP, loc_noise_type="manylittle",
            sparse=sparse, sim_type=sim_type, parameterization=parameterization,
            verbosity=printer-1)

        printer.log("Creating 1Q Y(pi/2) gate on qubit %d!!" % i)
        gs.gates[_Lbl("Gy",i)] = build_nqn_composed_gate(
            Gy, (i,), qubitGraph, weight_maxhops_tuples_1Q,
            idle_noise=idleOP, loc_noise_type="manylittle",
            sparse=sparse, sim_type=sim_type, parameterization=parameterization,
            verbosity=printer-1)

        cloud_inds = tuple(qubitGraph.radius((i,), cloud_maxhops))
        cloud_key = ( (i,), tuple(sorted(cloud_inds)) ) # (sets are unhashable)
        if cloud_key not in clouds: clouds[cloud_key] = []
        clouds[cloud_key].extend([_Lbl("Gx",i),_Lbl("Gy",i)])
        
    #2Q gates: CNOT gates along each graph edge
    Gcnot = std2Q_XYICNOT.gs_target.gates['Gcnot']
    weight_maxhops_tuples_2Q = [(1,maxhops+extraWeight1Hops),(2,maxhops)] + \
                               [ (2+x,maxhops) for x in range(1,extraGateWeight+1) ]
    cloud_maxhops = max( [mx for wt,mx in weight_maxhops_tuples_2Q] ) # max of max-hops
    for i,j in cnot_edges:
        printer.log("Creating CNOT gate between qubits %d and %d!!" % (i,j))
        gs.gates[_Lbl("Gcnot",(i,j))] = build_nqn_composed_gate(
            Gcnot, (i,j), qubitGraph, weight_maxhops_tuples_2Q,
            idle_noise=idleOP, loc_noise_type="manylittle",
            sparse=sparse, sim_type=sim_type, parameterization=parameterization,
            verbosity=printer-1)
        cloud_inds = tuple(qubitGraph.radius((i,j), cloud_maxhops))
        cloud_key = (tuple(sorted([i,j])), tuple(sorted(cloud_inds)))
        if cloud_key not in clouds: clouds[cloud_key] = []
        clouds[cloud_key].append(_Lbl("Gcnot",(i,j)))


    #Insert noise on gates
    vecNoSpam = gs.to_vector()
    assert( _np.linalg.norm(vecNoSpam)/len(vecNoSpam) < 1e-6 )
    if gateNoise is not None:
        if isinstance(gateNoise,tuple): # use as (seed, strength)
            seed,strength = gateNoise
            rndm = _np.random.RandomState(seed)
            vecNoSpam += _np.abs(rndm.random_sample(len(vecNoSpam))*strength) #abs b/c some params need to be positive
        else: #use as a vector
            vecNoSpam += gateNoise[0:len(vecNoSpam)]
        gs.from_vector(vecNoSpam)

        
    #SPAM
    basis1Q = _Basis("pp",2)
    prep_factors = []; povm_factors = []

    v0 = _basis_build_vector("0", basis1Q)
    v1 = _basis_build_vector("1", basis1Q)

    typ = parameterization
    povmtyp = rtyp = "TP" if typ in ("CPTP","H+S","S") else typ  # use TP until we have LindbladParameterizedSPAMVecs

    #DEBUG - make spam static for now (modified convert() in spamvec.py and povm.py for terms - see "DEBUG!!!")
    #if debug:
    #    if rtyp in ("TP",): rtyp = "static"
    #    if povmtyp in ("TP",): povmtyp = "static"

    for i in range(nQubits):
        prep_factors.append(
            _objs.spamvec.convert(_objs.StaticSPAMVec(v0), rtyp, basis1Q) )
        povm_factors.append(
            _objs.povm.convert(_objs.UnconstrainedPOVM( ([
                ('0',_objs.StaticSPAMVec(v0)),
                ('1',_objs.StaticSPAMVec(v1))]) ), povmtyp, basis1Q) )

    if prepNoise is not None:
        if isinstance(prepNoise,tuple): # use as (seed, strength)
            seed,strength = prepNoise
            rndm = _np.random.RandomState(seed)
            depolAmts = _np.abs(rndm.random_sample(nQubits)*strength)
        else:
            depolAmts = prepNoise[0:nQubits]
        for amt,vec in zip(depolAmts,prep_factors): vec.depolarize(amt)

    if povmNoise is not None:
        if isinstance(povmNoise,tuple): # use as (seed, strength)
            seed,strength = povmNoise
            rndm = _np.random.RandomState(seed)
            depolAmts = _np.abs(rndm.random_sample(nQubits)*strength)
        else:
            depolAmts = povmNoise[0:nQubits]
        for amt,povm in zip(depolAmts,povm_factors): povm.depolarize(amt) 

    gs.preps[_Lbl('rho0')] = _objs.TensorProdSPAMVec('prep', prep_factors)
    gs.povms[_Lbl('Mdefault')] = _objs.TensorProdPOVM(povm_factors)

    # #DEBUG SPECIAL CASE: when prep and povm noise are None create a *fixed*
    # # (no parameters) POVM of the computational basis elements. (This is just
    # # as useful for data generation but not as useful for GST as it doesn't
    # # give the SPAM any degrees of freedom.)
    # if prepNoise is None and povmNoise is None:
    #     iterover = [(0,1)]*nQubits
    #     items = [ (''.join(map(str,outcomes)), _objs.ComputationalSPAMVec(outcomes,"densitymx"))
    #               for outcomes in _itertools.product(*iterover) ]
    #     gs.povms[_Lbl('Mdefault')] = _objs.UnconstrainedPOVM(items) # overwrite above
    # # END DEBUG

    #FUTURE - just return cloud keys? (gate label values are never used
    # downstream, but may still be useful for debugging, so keep for now)
    printer.log("DONE! - returning GateSet with dim=%d and gates=%s" % (gs.dim, list(gs.gates.keys())))    
    return (gs, clouds) if return_clouds else gs
    

def _get_Lindblad_factory(sim_type, parameterization):
    """ Returns a function that creates a Lindblad-type gate appropriate
        given the simulation type and parameterization """
    if parameterization == "H+S":
        evotype = "densitymx"
        cls = _objs.LindbladParameterizedGate if sim_type == "matrix" \
              else _objs.LindbladParameterizedGateMap

    elif parameterization in ("H+S terms","H+S clifford terms"):
        assert(sim_type.startswith("termorder"))
        evotype = "svterm" if parameterization == "H+S terms" else "cterm"
        cls = _objs.LindbladParameterizedGateMap
    else:
        raise ValueError("Cannot create Lindblad gate factory for ",sim_type, parameterization)

    #Just call cls.from_gate_matrix with appropriate evotype
    def _f(gateMatrix, unitaryPostfactor=None,
          ham_basis="pp", nonham_basis="pp", cptp=True,
          nonham_diagonal_only=False, truncate=True, mxBasis="pp"):
        return cls.from_gate_matrix(gateMatrix, unitaryPostfactor,
                                    ham_basis, nonham_basis, cptp,
                                    nonham_diagonal_only, truncate,
                                    mxBasis, evotype)
    return _f
                                    

def _get_Static_factory(sim_type, parameterization):
    """ Returns a function that creates a static-type gate appropriate 
        given the simulation and parameterization """
    if parameterization == "H+S":
        if sim_type == "matrix":
            return lambda g,b: _objs.StaticGate(g)
        elif sim_type == "map":
            return lambda g,b: _objs.StaticGate(g) # TODO: create StaticGateMap

    elif parameterization in ("H+S terms","H+S clifford terms"):
        assert(sim_type.startswith("termorder"))
        evotype = "svterm" if parameterization == "H+S terms" else "cterm"
        
        def _f(gateMatrix, mxBasis="pp"):
            return _objs.LindbladParameterizedGateMap.from_gate_matrix(
                None, gateMatrix, None, None, mxBasis=mxBasis, evotype=evotype)
                # a LindbladParameterizedGate with None as ham_basis and nonham_basis => no parameters
              
        return _f
    raise ValueError("Cannot create Static gate factory for ",sim_type, parameterization)


def build_nqn_global_idle(qubitGraph, maxWeight, sparse=False, sim_type="matrix", parameterization="H+S", verbosity=0):
    """
    Create a "global" idle gate, meaning one that acts on all the qubits in 
    `qubitGraph`.  The gate will have up to `maxWeight` errors on *connected*
    (via the graph) sets of qubits.

    Parameters
    ----------
    qubitGraph : QubitGraph
        A graph giving the geometry (nearest-neighbor relations) of the qubits.

    maxWeight : int
        The maximum weight errors to include in the resulting gate.

    sparse : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        gate are represented as sparse or dense matrices.  (This is determied by
        whether they are constructed using sparse basis matrices.)

    sim_type : {"matrix","map","termorder:<N>"}
        The type of forward simulation (probability computation) being used by 
        the gate set this gate is destined for.  This affects what type of 
        gate objects (e.g. `ComposedGate` vs `ComposedGateMap`) are created.
    
    parameterization : {"H+S", "H+S terms", "H+S clifford terms"}
        The type of parameterizaton for the constructed gate.

    verbosity : int, optional
        An integer >= 0 dictating how must output to send to stdout.

    Returns
    -------
    Gate
    """
    assert(maxWeight <= 2), "Only `maxWeight` equal to 0, 1, or 2 is supported"

    if sim_type == "matrix": 
        Composed = _objs.ComposedGate
        Embedded = _objs.EmbeddedGate
    else:
        Composed = _objs.ComposedGateMap
        Embedded = _objs.EmbeddedGateMap
    Lindblad = _get_Lindblad_factory(sim_type, parameterization)
    
    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("*** Creating global idle ***")
    
    termgates = [] # gates to compose
    ssAllQ = [tuple(['Q%d'%i for i in range(qubitGraph.nqubits)])]
    #basisAllQ = _Basis('pp', 2**qubitGraph.nqubits, sparse=sparse) # TODO: remove - all we need is its 'dim' below
    basisAllQ_dim = _Dim(2**qubitGraph.nqubits)
    
    nQubits = qubitGraph.nqubits
    possible_err_qubit_inds = _np.arange(nQubits)
    nPossible = nQubits  
    for wt in range(1,maxWeight+1):
        printer.log("Weight %d: %d possible qubits" % (wt,nPossible),2)
        basisEl_Id = basisProductMatrix(_np.zeros(wt,_np.int64),sparse)
        wtId = _sps.identity(4**wt,'d','csr') if sparse else  _np.identity(4**wt,'d')
        wtBasis = _Basis('pp', 2**wt, sparse=sparse)
        
        for err_qubit_inds in _itertools.combinations(possible_err_qubit_inds, wt):
            if len(err_qubit_inds) == 2 and not qubitGraph.is_directly_connected(err_qubit_inds[0],err_qubit_inds[1]):
                continue # TO UPDATE - check whether all wt indices are a connected subgraph

            errbasis = [basisEl_Id]
            for err_basis_inds in _iter_basis_inds(wt):        
                error = _np.array(err_basis_inds,_np.int64) #length == wt
                basisEl = basisProductMatrix(error,sparse)
                errbasis.append(basisEl)

            printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_inds,len(errbasis)), 3)
            errbasis = _Basis(matrices=errbasis, sparse=sparse) #single element basis (plus identity)
            termErr = Lindblad(wtId, ham_basis=errbasis, nonham_basis=errbasis, cptp=True,
                               nonham_diagonal_only=True, truncate=True, mxBasis=wtBasis)
        
            err_qubit_global_inds = err_qubit_inds
            fullTermErr = Embedded(ssAllQ, [('Q%d'%i) for i in err_qubit_global_inds],
                                   termErr, basisAllQ_dim)
            assert(fullTermErr.num_params() == termErr.num_params())
            printer.log("Lindblad gate w/dim=%d and %d params -> embedded to gate w/dim=%d" %
                        (termErr.dim, termErr.num_params(), fullTermErr.dim))
                    
            termgates.append( fullTermErr )
                
    return Composed(termgates)         
    
    

#def build_nqn_noncomposed_gate(targetOp, target_qubit_inds, qubitGraph, maxWeight, maxHops,
#                            spectatorMaxWeight=1, mode="embed"):
#
#    assert(spectatorMaxWeight <= 1) #only 0 and 1 are currently supported
#    
#    errinds = [] # list of basis indices for all error terms
#    possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops),_np.int64)
#    nPossible = len(possible_err_qubit_inds)
#    for wt in range(maxWeight+1):
#        if mode == "no-embedding": # make an error term for the entire gate
#            for err_qubit_inds in _itertools.combinations(possible_err_qubit_inds, wt):
#                # err_qubit_inds are global qubit indices
#                #Future: check that err_qubit_inds marks qubits that are connected
#                
#                for err_basis_inds in iter_basis_inds(wt):  
#                    error = _np.zeros(nQubits)
#                    error[ possible_err_qubit_inds[err_qubit_inds] ] = err_basis_inds
#                    errinds.append( error )
#                    
#        elif mode == "embed": # make an error term for only the "possible error" qubits
#                              # which will get embedded to form a full gate
#            for err_qubit_inds in _itertools.combinations(list(range(nPossible)), wt):
#                # err_qubit_inds are indices into possible_err_qubit_inds
#                #Future: check that err_qubit_inds marks qubits that are connected
#                
#                for err_basis_inds in iter_basis_inds(wt):  
#                    error = _np.zeros(nPossible)
#                    error[ err_qubit_inds ] = err_basis_inds
#                    errinds.append( error )
#
#    errbasis = [ basisProductMatrix(err) for err in errinds]
#    
#    ssAllQ = ['Q%d'%i for i in range(qubitGraph.nqubits)]
#    basisAllQ = pygsti.objects.Basis('pp', 2**qubitGraph.nqubits)
#    
#    if mode == "no-embedding":     
#        fullTargetOp = EmbeddedGate(ssAllQ, ['Q%d'%i for i in target_qubit_inds],
#                                    targetOp, basisAllQ) 
#        fullTargetOp = StaticGate( fullTargetOp ) #Make static
#        fullLocalErr = LindbladParameterizedGate(fullTargetOp, fullTargetOp,
#                         ham_basis=errbasis, nonham_basis=errbasis, cptp=True,
#                         nonham_diagonal_only=True, truncate=True, mxBasis=basisAllQ)
#          # gate on full qubit space that accounts for error on the "local qubits", that is,
#          # those local to the qubits being operated on
#    elif mode == "embed":
#        possible_list = list(possible_err_qubit_inds)
#        loc_target_inds = [possible_list.index(i) for i in target_qubit_inds]
#        
#        ssLocQ = ['Q%d'%i for i in range(nPossible)]
#        basisLocQ = pygsti.objects.Basis('pp', 2**nPossible)
#        locTargetOp = StaticGate( EmbeddedGate(ssLocQ, ['Q%d'%i for i in loc_target_inds],
#                                    targetOp, basisLocQ) )
#        localErr = LindbladParameterizedGate(locTargetOp, locTargetOp,
#                         ham_basis=errbasis, nonham_basis=errbasis, cptp=True,
#                         nonham_diagonal_only=True, truncate=True, mxBasis=basisLocQ)
#        fullLocalErr = EmbeddedGate(ssAllQ, ['Q%d'%i for i in possible_err_qubit_inds],
#                                   localErr, basisAllQ)
#    else:
#        raise ValueError("Invalid Mode: %s" % mode)
#        
#    #Now add errors on "non-local" i.e. spectator gates
#    if spectatorMaxWeight == 0:
#        pass
#    #STILL in progress -- maybe just non-embedding case, since if we embed we'll
#    # need to compose (in general)
        

        
def build_nqn_composed_gate(targetOp, target_qubit_inds, qubitGraph, weight_maxhops_tuples,
                            idle_noise=False, loc_noise_type="onebig",
                            apply_idle_noise_to="all", sparse=False, sim_type="matrix",
                            parameterization="H+S", verbosity=0):
    """ 
    Create an n-qubit gate that is a composition of:
    
    `targetOp(target_qubits) -> idle_noise(all_qubits) -> loc_noise(local_qubits)`

    where `idle_noise` is given by the `idle_noise` argument and `loc_noise` is
    given by the rest of the arguments.  `loc_noise` can be implemented either
    by a single (n-qubit) embedded Lindblad gate with all relevant error
    generators, or as a composition of embedded single-error-term Lindblad gates
    (see param `loc_noise_type`).

    The local noise consists terms up to a maximum weight acting on the qubits
    given reachable by a given maximum number of hops (along the neareset-
    neighbor edges of `qubitGraph`) from the target qubits.


    Parameters
    ----------
    targetOp : numpy array
        The target operation as a dense matrix, specifying the action on just
        the target qubits.

    target_qubit_inds : list
        The indices of the target qubits.
        
    qubitGraph : QubitGraph
        A graph giving the geometry (nearest-neighbor relations) of the qubits.

    weight_maxhops_tuples : iterable
        A list of `(weight,maxhops)` 2-tuples specifying which error weights
        should be included and what region of the graph (as a `maxhops` from
        the set of target qubits) should have errors of the given weight applied
        to it.

    idle_noise : Gate or boolean
        Either given as an existing gate (on all qubits) or a boolean indicating
        whether a composition of weight-1 noise terms (separately on all the qubits),
        is created.  If `apply_idle_noise_to == "nonlocal"` then `idle_noise` is *only*
        applied to the non-local qubits and `idle_noise` must be a ComposedGate or
        ComposedMap with `nQubits` terms so that individual terms for each qubit can
        be extracted as needed.


    loc_noise_type : {"onebig","manylittle"}
        Whether the `loc_noise` portion of the constructed gate should be a
        a single Lindblad gate containing all the allowed error terms (onebig)
        or the composition of many Lindblad gates each containing just a single
        error term (manylittle).  The resulting gate performs the same action
        regardless of the value set here; this just affects how the gate is
        structured internally.

    apply_idle_noise_to : {"all","nonlocal"}
        Whether the `idle_noise` argument represents a Gate to be applied to 
        *all* the qubits or whether it is a composed Gate with `nqubits` terms
        that can be selectively applied only to the non-local qubits. (i.e.
        those that are more than max-hops away from the target qubits).

    sparse : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        gate are represented as sparse or dense matrices.  (This is determied by
        whether they are constructed using sparse basis matrices.)

    sim_type : {"matrix","map","termorder:<N>"}
        The type of forward simulation (probability computation) being used by 
        the gate set this gate is destined for.  This affects what type of 
        gate objects (e.g. `ComposedGate` vs `ComposedGateMap`) are created.
    
    parameterization : {"H+S", "H+S terms", "H+S clifford terms"}
        The type of parameterizaton for the constructed gate.

    verbosity : int, optional
        An integer >= 0 dictating how must output to send to stdout.

    Returns
    -------
    Gate
    """
    if sim_type == "matrix": 
        Composed = _objs.ComposedGate
        Embedded = _objs.EmbeddedGate
    else:
        Composed = _objs.ComposedGateMap
        Embedded = _objs.EmbeddedGateMap
    Static = _get_Static_factory(sim_type, parameterization)
    Lindblad = _get_Lindblad_factory(sim_type, parameterization)

    ##OLD
    #if sparse:
    #    Lindblad = _objs.LindbladParameterizedGateMap
    #    Composed = _objs.ComposedGateMap
    #    Embedded = _objs.EmbeddedGateMap
    #    Static = _objs.StaticGate # TODO: create StaticGateMap
    #else:
    #    Lindblad = _objs.LindbladParameterizedGate
    #    Composed = _objs.ComposedGate
    #    Embedded = _objs.EmbeddedGate
    #    Static = _objs.StaticGate
    
    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("*** Creating composed gate ***")
    
    #Factor1: target operation
    printer.log("Creating %d-qubit target op factor on qubits %s" %
                (len(target_qubit_inds),str(target_qubit_inds)),2)
    ssAllQ = [tuple(['Q%d'%i for i in range(qubitGraph.nqubits)])]
    #basisAllQ = _Basis('pp', 2**qubitGraph.nqubits, sparse=sparse)
    basisAllQ_dim = _Dim(2**qubitGraph.nqubits)
    fullTargetOp = Embedded(ssAllQ, ['Q%d'%i for i in target_qubit_inds],
                            Static(targetOp,"pp"), basisAllQ_dim) 

    #Factor2: idle_noise operation
    printer.log("Creating idle error factor",2)
    if apply_idle_noise_to == "all":
        if isinstance(idle_noise, _objs.Gate):
            printer.log("Using supplied full idle gate",3)
            fullIdleErr = idle_noise
        elif idle_noise == True:
            #build composition of 1Q idle ops
            printer.log("Constructing independend weight-1 idle gate",3)
            # Id_1Q = _sps.identity(4**1,'d','csr') if sparse else  _np.identity(4**1,'d')
            Id_1Q = _np.identity(4**1,'d') #always dense for now...
            fullIdleErr = Composed(
                [ Embedded(ssAllQ, ('Q%d'%i,), Lindblad(Id_1Q.copy()),basisAllQ_dim)
                  for i in range(qubitGraph.nqubits)] )
        elif idle_noise == False:
            printer.log("No idle factor",3)
            fullIdleErr = None
        else:
            raise ValueError("Invalid `idle_noise` argument")
            
    elif apply_idle_noise_to == "nonlocal":
        pass #TODO: only apply (1Q) idle noise to qubits that don't have 1Q local noise.
        assert(False)
    
    else:
        raise ValueError('Invalid `apply_idle_noise_to` argument: %s' % apply_idle_noise_to)

        
    #Factor3: local_noise operation
    printer.log("Creating local-noise error factor (%s)" % loc_noise_type,2)
    if loc_noise_type == "onebig": 
        # make a single embedded Lindblad-gate containing all specified error terms
        loc_noise_errinds = [] # list of basis indices for all local-error terms
        all_possible_err_qubit_inds = _np.array( qubitGraph.radius(
            target_qubit_inds, max([hops for _,hops in weight_maxhops_tuples]) ), _np.int64) # node labels are ints
        nLocal = len(all_possible_err_qubit_inds)
        basisEl_Id = basisProductMatrix(_np.zeros(nPossible,_np.int64),sparse) #identity basis el
        
        for wt, maxHops in weight_maxhops_tuples:
            possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops),_np.int64) # node labels are ints
            nPossible = len(possible_err_qubit_inds)
            possible_to_local = [ all_possible_err_qubit_inds.index(
                possible_err_qubit_inds[i]) for i in range(nPossible)]
            printer.log("Weight %d, max-hops %d: %d possible qubits of %d local" %
                        (wt,maxHops,nPossible,nLocal),3)
            
            for err_qubit_inds in _itertools.combinations(list(range(nPossible)), wt):
                # err_qubit_inds are in range [0,nPossible-1] qubit indices
                #Future: check that err_qubit_inds marks qubits that are connected
                err_qubit_local_inds = possible_to_local[err_qubit_inds]
                                                        
                for err_basis_inds in _iter_basis_inds(wt):  
                    error = _np.zeros(nLocal,_np.int64)
                    error[ err_qubit_local_inds ] = err_basis_inds
                    loc_noise_errinds.append( error )
                    
                printer.log("Error on qubits %s -> error basis now at length %d" %
                            (all_possible_err_qubit_inds[err_qubit_local_inds],1+len(loc_noise_errinds)), 4)
                
        errbasis = [basisEl_Id] + \
                   [ basisProductMatrix(err,sparse) for err in loc_noise_errinds]
        errbasis = _Basis(matrices=errbasis, sparse=sparse) #single element basis (plus identity)
        
        #Construct one embedded Lindblad-gate using all `errbasis` terms
        ssLocQ = [tuple(['Q%d'%i for i in range(nLocal)])]
        basisLocQ = _Basis('pp', 2**nLocal, sparse=sparse)
        locId = _sps.identity(4**nLocal,'d','csr') if sparse else _np.identity(4**nLocal,'d')
        localErr = Lindblad(locId, ham_basis=errbasis,
                            nonham_basis=errbasis, cptp=True,
                            nonham_diagonal_only=True, truncate=True,
                            mxBasis=basisLocQ)
        fullLocalErr = Embedded(ssAllQ, ['Q%d'%i for i in all_possible_err_qubit_inds],
                                localErr, basisAllQ_dim)
        printer.log("Lindblad gate w/dim=%d and %d params (from error basis of len %d) -> embedded to gate w/dim=%d" %
                    (localErr.dim, localErr.num_params(), len(errbasis), fullLocalErr.dim),2)

        
    elif loc_noise_type == "manylittle": 
        # make a composed-gate of embedded single-basis-element Lindblad-gates,
        #  one for each specified error term  
            
        loc_noise_termgates = [] #list of gates to compose
        
        for wt, maxHops in weight_maxhops_tuples:
                
            ## loc_noise_errinds = [] # list of basis indices for all local-error terms 
            possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops),_np.int64) # we know node labels are integers
            nPossible = len(possible_err_qubit_inds) # also == "nLocal" in this case
            basisEl_Id = basisProductMatrix(_np.zeros(wt,_np.int64),sparse) #identity basis el

            wtId = _sps.identity(4**wt,'d','csr') if sparse else _np.identity(4**wt,'d')
            wtBasis = _Basis('pp', 2**wt, sparse=sparse)

            printer.log("Weight %d, max-hops %d: %d possible qubits" % (wt,maxHops,nPossible),3)
            #print("DB: possible qubits = ",possible_err_qubit_inds, " (radius of %d around %s)" % (maxHops,str(target_qubit_inds)))
            
            for err_qubit_local_inds in _itertools.combinations(list(range(nPossible)), wt):
                # err_qubit_inds are in range [0,nPossible-1] qubit indices
                #Future: check that err_qubit_inds marks qubits that are connected

                errbasis = [basisEl_Id]
                for err_basis_inds in _iter_basis_inds(wt):  
                    error = _np.array(err_basis_inds,_np.int64) #length == wt
                    basisEl = basisProductMatrix(error, sparse)
                    errbasis.append(basisEl)

                err_qubit_global_inds = possible_err_qubit_inds[list(err_qubit_local_inds)]
                printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_global_inds,len(errbasis)), 4)
                errbasis = _Basis(matrices=errbasis, sparse=sparse) #single element basis (plus identity)
                termErr = Lindblad(wtId, ham_basis=errbasis,
                                   nonham_basis=errbasis, cptp=True,
                                   nonham_diagonal_only=True, truncate=True,
                                   mxBasis=wtBasis)
        
                fullTermErr = Embedded(ssAllQ, ['Q%d'%i for i in err_qubit_global_inds],
                                       termErr, basisAllQ_dim)
                assert(fullTermErr.num_params() == termErr.num_params())
                printer.log("Lindblad gate w/dim=%d and %d params -> embedded to gate w/dim=%d" %
                            (termErr.dim, termErr.num_params(), fullTermErr.dim))
                
                loc_noise_termgates.append( fullTermErr )
              
        fullLocalErr = Composed(loc_noise_termgates)    
        
    else:
        raise ValueError("Invalid `loc_noise_type` arguemnt: %s" % loc_noise_type)
        
    if fullIdleErr is not None:
        return Composed([fullTargetOp,fullIdleErr,fullLocalErr])
    else:
        return Composed([fullTargetOp,fullLocalErr])



# -----------------------------------------------------------------------------------
#  nqnoise gate sequence construction methods
# -----------------------------------------------------------------------------------

#Note: these methods assume a GateSet with:
# Gx and Gy gates on each qubit that are pi/2 rotations
# a prep labeled "rho0"
# a povm labeled "Mdefault" - so effects labeled "Mdefault_N" for N=0->2^nQubits-1


def _onqubit(s,iQubit):
    """ Takes `s`, a tuple of gate *names* and creates a GateString
        where those names act on the `iQubit`-th qubit """
    return _objs.GateString( [_Lbl(nm,iQubit) for nm in s])

def find_amped_polys_for_syntheticidle(qubit_filter, idleStr, gateset, singleQfiducials=None,
                                       prepLbl=None, effectLbls=None, initJ=None, initJrank=None,
                                       wrtParams=None, algorithm="greedy", require_all_amped=True,
                                       verbosity=0):
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

    idleStr : GateString
        The gate string specifying the idle operation to consider.  This may
        just be a single idle gate, or it could be multiple non-idle gates
        which together act as an idle.

    gateset : GateSet
        The gate set used to compute the polynomial expressions of probabilities
        to first-order.  Thus, this gate set should always have (simulation)
        type "termorder:1".

    singleQfiducials : list, optional
        A list of gate-name tuples (e.g. `('Gx',)`) which specify a set of single-
        qubit fiducials to use when trying to amplify gate parameters.  Note that
        no qubit "state-space" label is required here (i.e. *not* `(('Gx',1),)`);
        the tuples just contain single-qubit gate *names*.  If None, then
        `[(), ('Gx',), ('Gy',)]` is used by default.

    prepLbl : Label, optional
        The state preparation label to use.  If None, then the first (and
        usually the only) state prep label of `gateset` is used, so it's
        usually fine to leave this as None.

    effectLbls : list, optional
        The list of POVM effect labels to use, as a list of `Label` objects.
        These are *compiled* POVM effect labels, so something like "Mdefault_0",
        and if None the default is all the effect labels of the first POVM of
        `gateset`, which is usually what you want.

    initJ : numpy.ndarray, optional
        An initial Jacobian giving the derivatives of some other polynomials
        with respect to the same `wrtParams` that this function is called with.
        This acts as a starting point, and essentially informs the fiducial-pair
        selection algorithm that some parameters (or linear combos of them) are
        *already* amplified (e.g. by some other germ that's already been
        selected) and for which fiducial pairs are not needed.

    initJrank : int, optional
        The rank of `initJ`.  The function could compute this from `initJ`
        but in practice one usually has the rank of `initJ` lying around and
        so this saves a call to `np.linalg.matrix_rank`.

    wrtParams : slice, optional
        The parameters to consider for amplification.  (This function seeks
        fiducial pairs that amplify these parameters.)  If None, then pairs
        which amplify all of `gateset`'s parameters are searched for.
    
    algorithm : {"greedy","sequential"}
        Which algorithm is used internally to find fiducial pairs.  "greedy" 
        will give smaller sets of fiducial pairs (better) but takes longer.
        Usually it's worth the wait and you should use the default ("greedy").

    require_all_amped : bool, optional
        If True and AssertionError is raised when fewer than all of the
        requested parameters (in `wrtParams`) are amplifed by the final set of
        fiducial pairs.
                       
    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.


    Returns
    -------
    J : numpy.ndarray
        The final jacobian with rows equal to the number of chosen amplified
        polynomials (note there is one row per fiducial pair *including* the
        outcome - so there will be two different rows for two different
        outcomes) and one column for each parameter specified by `wrtParams`.

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
    # When the gateset only gives probability polynomials to first order in
    # the error rates this gives the L-dependent and hence amplified part
    # of the polynomial expression for the probability of F1*(germ^L)*F2.
    # This derivative of an amplified polynomial, taken with respect to
    # all the parameters we care about (i.e. wrtParams) would ideally be
    # kept as a polynomial and the "rank" of J would be the number of
    # linearly independent polynomials within the rows of J (each poly
    # would be a vector in the space of polynomials).  We currently take
    # a cheap/HACK way out and evaluate the derivative-polynomial at a
    # random dummy value which should yield linearly dependent vectors
    # in R^n whenever the polynomials are linearly indepdendent - then
    # we can use the usual scipy/numpy routines for computing a matrix
    # rank, etc.

    # Assert that gateset uses termorder:1, as doing L1-L0 to extract the "amplified" part
    # relies on only expanding to *first* order.
    assert(gateset._sim_type == "termorder" and gateset._sim_args[0] == '1'), \
        '`gateset` must use "termorder:1" simulation type!'
    
    printer = _VerbosityPrinter.build_printer(verbosity)
    
    if prepLbl is None:
        prepLbl = list(gateset.preps.keys())[0]
    if effectLbls is None:
        povmLbl = list(gateset.povms.keys())[0]
        effectLbls = [ _Lbl("%s_%s" % (povmLbl,l)) for l in gateset.povms[povmLbl] ]
    if singleQfiducials is None:
        # TODO: assert gate set has Gx and Gy gates?
        singleQfiducials = [(), ('Gx',), ('Gy',)] # ('Gx','Gx')
        
    #dummy = 0.05*_np.ones(gateset.num_params(),'d') # for evaluating derivs...
    #dummy = 0.05*_np.arange(1,gateset.num_params()+1) # for evaluating derivs...
    #dummy = 0.05*_np.random.random(gateset.num_params())
    dummy = 5.0*_np.random.random(gateset.num_params()) + 0.5*_np.ones(gateset.num_params(),'d')
     # expect terms to be either coeff*x or coeff*x^2 - (b/c of latter case don't eval at zero)
    
    #amped_polys = []
    selected_gatename_fidpair_lists = []
    if wrtParams is None: wrtParams = slice(0,gateset.num_params())
    Np = _slct.length(wrtParams) 
    if initJ is None:
        J = _np.empty( (0,Np), 'complex'); Jrank = 0
    else:
        J = initJ; Jrank = initJrank

    if algorithm == "greedy":
        Jrows = _np.empty( (len(effectLbls),Np), 'complex')

    #Outer iteration
    while Jrank < Np:

        if algorithm == "sequential":
            printer.log("Sequential find_amped_polys_for_syntheticidle started. Target rank=%d" % Np)
            
        elif algorithm == "greedy":
            maxRankInc = 0
            bestJrows = None
            printer.log("Greedy find_amped_polys_for_syntheticidle started. Target rank=%d" % Np)
            
        else: raise ValueError("Invalid `algorithm` argument: %s" % algorithm)
        
        
        # loop over all possible (remaining) fiducial pairs
        nQubits = len(qubit_filter); nIters = len(singleQfiducials)**nQubits
        with printer.progress_logging(2):
            for itr,prep in enumerate(_itertools.product(*([singleQfiducials]*nQubits) )):
                printer.show_progress(itr, nIters, prefix='--- Finding amped-polys for idle: ')
                prepFid = _objs.GateString(())
                for i,el in enumerate(prep):
                    prepFid = prepFid + _onqubit(el,qubit_filter[i])
                    
                for meas in _itertools.product(*([singleQfiducials]*nQubits) ):
                    measFid = _objs.GateString(())
                    for i,el in enumerate(meas):
                        measFid = measFid + _onqubit(el,qubit_filter[i])
    
                    gatename_fidpair_list = [ (prep[i],meas[i]) for i in range(nQubits) ]
                    if gatename_fidpair_list in selected_gatename_fidpair_lists:
                        continue # we've already chosen this pair in a previous iteration
            
                    gstr_L0 = prepFid + measFid            # should be a GateString
                    gstr_L1 = prepFid + idleStr + measFid  # should be a GateString
                    ps=gateset._calc().prs_as_polys(prepLbl, effectLbls, gstr_L1 )
                    qs=gateset._calc().prs_as_polys(prepLbl, effectLbls, gstr_L0 )

                    if algorithm == "sequential":
                        added = False
                        for elbl,p,q in zip(effectLbls,ps,qs):
                            amped = p + -1*q # the amplified poly
                            Jrow = _np.array([[ amped.deriv(iParam).evaluate(dummy) for iParam in _slct.as_array(wrtParams)]])
                            if _np.linalg.norm(Jrow) < 1e-8: continue  # row of zeros can fool matrix_rank
                            
                            Jtest = _np.concatenate((J,Jrow),axis=0) 
                            testRank = _np.linalg.matrix_rank(Jtest, tol=RANK_TOL)
                            if testRank > Jrank:
                                printer.log("fidpair: %s,%s (%s) increases rank => %d" %
                                            (str(prep),str(meas),str(elbl),testRank), 4)
                                J = Jtest
                                Jrank = testRank
                                if not added:
                                    selected_gatename_fidpair_lists.append( gatename_fidpair_list )
                                    added = True # only add fidpair once per elabel loop!
                                if Jrank == Np: break # this is the largest rank J can take!

                    elif algorithm == "greedy":
                        #test adding all effect labels - get the overall increase in rank due to this fidpair
                        for k,(elbl,p,q) in enumerate(zip(effectLbls,ps,qs)):
                            amped = p + -1*q # the amplified poly
                            Jrows[k,:] = _np.array([[ amped.deriv(iParam).evaluate(dummy) for iParam in _slct.as_array(wrtParams)]])
                        Jtest = _np.concatenate((J,Jrows),axis=0) 
                        testRank = _np.linalg.matrix_rank(Jtest, tol=RANK_TOL)
                        rankInc = testRank - Jrank
                        if rankInc > maxRankInc:
                            maxRankInc = rankInc
                            bestJrows = Jrows.copy()
                            bestFidpair = gatename_fidpair_list
                            if testRank == Np: break # this is the largest rank we can get!
                        
        if algorithm == "greedy":
            if require_all_amped:
                assert(maxRankInc > 0), "No fiducial pair increased the Jacobian rank!"
            Jrank += maxRankInc
            J = _np.concatenate((J,bestJrows),axis=0)
            selected_gatename_fidpair_lists.append( bestFidpair )
            printer.log("%d fidpairs => rank %d (Np=%d)" %
                        (len(selected_gatename_fidpair_lists), Jrank, Np))
            
    
    #DEBUG
    #print("DB: J = ")
    #_gt.print_mx(J)
    #print("DB: svals of J for synthetic idle: ", _np.linalg.svd(J, compute_uv=False))
    
    return J, Jrank, selected_gatename_fidpair_lists


def test_amped_polys_for_syntheticidle(fidpairs, idleStr, gateset,  prepLbl=None, effectLbls=None,
                                       wrtParams=None, verbosity=0):
    """
    Compute the number of gate set parameters amplified by a given (synthetic)
    idle sequence.

    Parameters
    ----------
    fidpairs : list
        A list of `(prep,meas)` 2-tuples, where `prep` and `meas` are 
        :class:`GateString` objects, specifying the fiducial pairs to test.

    idleStr : GateString
        The gate string specifying the idle operation to consider.  This may
        just be a single idle gate, or it could be multiple non-idle gates
        which together act as an idle.

    gateset : GateSet
        The gate set used to compute the polynomial expressions of probabilities
        to first-order.  Thus, this gate set should always have (simulation)
        type "termorder:1".

    prepLbl : Label, optional
        The state preparation label to use.  If None, then the first (and
        usually the only) state prep label of `gateset` is used, so it's
        usually fine to leave this as None.

    effectLbls : list, optional
        The list of POVM effect labels to use, as a list of `Label` objects.
        These are *compiled* POVM effect labels, so something like "Mdefault_0",
        and if None the default is all the effect labels of the first POVM of
        `gateset`, which is usually what you want.

    wrtParams : slice, optional
        The parameters to consider for amplification.  If None, then pairs
        which amplify all of `gateset`'s parameters are searched for.

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.

    Returns
    -------
    nAmplified : int
        The number of parameters amplified.

    nTotal : int
        The total number of parameters considered for amplification.
    """
    #Assert that gateset uses termorder:1, as doing L1-L0 to extract the "amplified" part
    # relies on only expanding to *first* order.
    assert(gateset._sim_type == "termorder" and gateset._sim_args[0] == '1'), \
        '`gateset` must use "termorder:1" simulation type!'

    printer = _VerbosityPrinter.build_printer(verbosity)
    
    if prepLbl is None:
        prepLbl = list(gateset.preps.keys())[0]
    if effectLbls is None:
        povmLbl = list(gateset.povms.keys())[0]
        effectLbls = [ _Lbl("%s_%s" % (povmLbl,l)) for l in gateset.povms[povmLbl] ]
        
    dummy = 5.0*_np.random.random(gateset.num_params()) + 0.5*_np.ones(gateset.num_params(),'d')
    
    if wrtParams is None: wrtParams = slice(0,gateset.num_params())
    Np = _slct.length(wrtParams) 
    nEffectLbls = len(effectLbls)
    nRows = len(fidpairs) * nEffectLbls # number of jacobian rows
    J = _np.empty( (nRows,Np), 'complex'); Jrank = 0
    
    for i,(prepFid, measFid) in enumerate(fidpairs):
        gstr_L0 = prepFid + measFid            # should be a GateString
        gstr_L1 = prepFid + idleStr + measFid  # should be a GateString
        ps=gateset._calc().prs_as_polys(prepLbl, effectLbls, gstr_L1 )
        qs=gateset._calc().prs_as_polys(prepLbl, effectLbls, gstr_L0 )

        for k,(elbl,p,q) in enumerate(zip(effectLbls,ps,qs)):
            amped = p + -1*q # the amplified poly
            Jrow = _np.array([[ amped.deriv(iParam).evaluate(dummy) for iParam in _slct.as_array(wrtParams)]])
            J[i*nEffectLbls+k,:] = Jrow

    rank = _np.linalg.matrix_rank(J, tol=RANK_TOL)
    #print("Rank = %d, num params = %d" % (rank, Np))
    return rank, Np

    
def find_amped_polys_for_clifford_syntheticidle(qubit_filter, core_filter, trueIdlePairs, idleStr, maxWt,
                                                gateset, singleQfiducials=None,
                                                prepLbl=None, effectLbls=None, initJ=None, initJrank=None,
                                                wrtParams=None, verbosity=0):
    """
    Similar to :function:`find_amped_polys_for_syntheticidle` but
    specialized to "qubit cloud" processing case used in higher-level 
    functions and assumes that `idleStr` is composed of Clifford gates only
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
    :function:`find_amped_polys_for_syntheticidle`.
    

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
        gates within `idleStr` ideally act upon.  This is often a proper subset
        of `qubit_filter` since errors are allowed on qubits which neighbor
        the core qubits in addition to the core qubits themselves.

    trueIdlePairs : dict
        A dictionary whose keys are integer max-weight values and whose values
        are lists of fiducial pairs, each in "gatename-fidpair-list" format,
        whcih give the fiducial pairs needed to amplify all the parameters of
        a non-synthetic idle gate on max-weight qubits.

    idleStr : GateString
        The gate string specifying the idle operation to consider.  This may
        just be a single idle gate, or it could be multiple non-idle gates
        which together act as an idle.

    maxWt : int
        The maximum weight such that the pairs given by `trueIdlePairs[maxWt]`
        will amplify all the possible errors on `idleStr`.  This must account
        for the fact that the nontrivial comprising `idleStr` may increase the
        weight of errors.  For instance if `idleStr` contains CNOT gates 
        on qubits 0 and 1 (the "core") and the noise model allows insertion of
        up to weight-2 errors at any location, then a single weight-2 error
        (recall termorder:1 means there can be only 1 error per circuit) on
        qubits 1 and 2 followed by a CNOT on 0 and 1 could yield an weight-3
        error on qubits 0,1, and 2.

    gateset : GateSet
        The gate set used to compute the polynomial expressions of probabilities
        to first-order.  Thus, this gate set should always have (simulation)
        type "termorder:1".

    singleQfiducials : list, optional
        A list of gate-name tuples (e.g. `('Gx',)`) which specify a set of single-
        qubit fiducials to use when trying to amplify gate parameters.  Note that
        no qubit "state-space" label is required here (i.e. *not* `(('Gx',1),)`);
        the tuples just contain single-qubit gate *names*.  If None, then
        `[(), ('Gx',), ('Gy',)]` is used by default.

    prepLbl : Label, optional
        The state preparation label to use.  If None, then the first (and
        usually the only) state prep label of `gateset` is used, so it's
        usually fine to leave this as None.

    effectLbls : list, optional
        The list of POVM effect labels to use, as a list of `Label` objects.
        These are *compiled* POVM effect labels, so something like "Mdefault_0",
        and if None the default is all the effect labels of the first POVM of
        `gateset`, which is usually what you want.

    initJ : numpy.ndarray, optional
        An initial Jacobian giving the derivatives of some other polynomials
        with respect to the same `wrtParams` that this function is called with.
        This acts as a starting point, and essentially informs the fiducial-pair
        selection algorithm that some parameters (or linear combos of them) are
        *already* amplified (e.g. by some other germ that's already been
        selected) and for which fiducial pairs are not needed.

    initJrank : int, optional
        The rank of `initJ`.  The function could compute this from `initJ`
        but in practice one usually has the rank of `initJ` lying around and
        so this saves a call to `np.linalg.matrix_rank`.

    wrtParams : slice, optional
        The parameters to consider for amplification.  (This function seeks
        fiducial pairs that amplify these parameters.)  If None, then pairs
        which amplify all of `gateset`'s parameters are searched for.

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.


    Returns
    -------
    J : numpy.ndarray
        The final jacobian with rows equal to the number of chosen amplified
        polynomials (note there is one row per fiducial pair *including* the
        outcome - so there will be two different rows for two different
        outcomes) and one column for each parameter specified by `wrtParams`.

    Jrank : int
        The rank of the jacobian `J`, equal to the number of amplified 
        parameters (at most the number requested).

    fidpair_lists : list
        The selected fiducial pairs, each in "gatename-fidpair-list" format.
        See :function:`find_amped_polys_for_syntheticidle` for details.
    """

    #Assert that gateset uses termorder:1, as doing L1-L0 to extract the "amplified" part
    # relies on only expanding to *first* order.
    assert(gateset._sim_type == "termorder" and gateset._sim_args[0] == '1'), \
        '`gateset` must use "termorder:1" simulation type!'

    printer = _VerbosityPrinter.build_printer(verbosity)
    
    if prepLbl is None:
        prepLbl = list(gateset.preps.keys())[0]
    if effectLbls is None:
        povmLbl = list(gateset.povms.keys())[0]
        effectLbls = [ _Lbl("%s_%s" % (povmLbl,l)) for l in gateset.povms[povmLbl] ]
    if singleQfiducials is None:
        # TODO: assert gate set has Gx and Gy gates?
        singleQfiducials = [(), ('Gx',), ('Gy',)] # ('Gx','Gx')
        
    #dummy = 0.05*_np.ones(gateset.num_params(),'d') # for evaluating derivs...
    #dummy = 0.05*_np.arange(1,gateset.num_params()+1) # for evaluating derivs...
    #dummy = 0.05*_np.random.random(gateset.num_params())
    dummy = 5.0*_np.random.random(gateset.num_params()) + 0.5*_np.ones(gateset.num_params(),'d')
     # expect terms to be either coeff*x or coeff*x^2 - (b/c of latter case don't eval at zero)
    
    #amped_polys = []
    selected_gatename_fidpair_lists = []
    if wrtParams is None: wrtParams = slice(0,gateset.num_params())
    Np = _slct.length(wrtParams) 
    if initJ is None:
        J = _np.empty( (0,Np), 'complex'); Jrank = 0
    else:
        J = initJ; Jrank = initJrank
    
    # We presume that we know the fiducial pairs
    #  needed to amplify all "true-idle" errors *of the same
    #  type that are on this synthetic idle* (i.e. H+S
    #  or full LND) up to some weight.  If we also assume
    #  the core-action is Clifford (i.e. maps Paulis->Paulis)
    #  then these same fiducial pairs that find the amplifiable
    #  params of a true idle with up to weight-maxWt terms will
    #  also find all the  amplifiable parameters of the synthetic
    #  idle, with the caveat that the maxWt must account for the
    #  weight-increasing potential of the non-trivial Clifford
    #  action.

    nQubits = len(qubit_filter)
    nCore = len(core_filter)

    #Tile idle_fidpairs for maxWt onto nQubits
    # (similar to tile_idle_fidpairs(...) but don't need to convert to gatestrings?)
    tmpl = get_kcoverage_template(nQubits, maxWt)
    idle_gatename_fidpair_lists = trueIdlePairs[maxWt]
    #print("IDLE GFP LISTS = ",idle_gatename_fidpair_lists)

    gatename_fidpair_lists = []
    for gatename_fidpair_list in idle_gatename_fidpair_lists:
        # replace 0..(k-1) in each template string with the corresponding
        # gatename_fidpair (acts on the single qubit identified by the
        # its index within the template string), then convert to a GateString/Circuit
        gfp = []
        for tmpl_row in tmpl:
            #mod_tmpl_row = tmpl_row[:]
            #for ql in core_filter: mod_tmpl_row[qubit_filter.index(ql)] = 0 # zero out to remove duplicates on non-core
            instance_row = [ gatename_fidpair_list[i] for i in tmpl_row ]
            #DEBUG for ql in core_filter: instance_row[qubit_filter.index(ql)] = ((),()) # zero out to remove duplicates on non-core

            gfp.append( tuple(instance_row) )
            
        gatename_fidpair_lists.extend(gfp)
          # tuple so it can be hashed in remove_duplicates
    _lt.remove_duplicates_in_place( gatename_fidpair_lists )
    ##print("GFP LISTS (nQ=%d) = " % nQubits,gatename_fidpair_lists)
    #printer.log("Testing %d fidpairs for %d-wt idle -> %d after %dQ tiling -> %d w/free %d core (vs %d)"
    #            % (len(idle_gatename_fidpair_lists), maxWt, len(gatename_fidpair_lists),
    #               nQubits, len(gatename_fidpair_lists)*(3**(2*nCore)), nCore, 3**(2*nQubits)))
    #print("DB: over %d qubits -> template w/%d els" % (nQubits, len(tmpl)))
    printer.log("Testing %d fidpairs for %d-wt idle -> %d after %dQ tiling"
                % (len(idle_gatename_fidpair_lists), maxWt, len(gatename_fidpair_lists),nQubits))


    for gfp_list in gatename_fidpair_lists:
        # # replace 0..(k-1) in each template string with the corresponding
        # # gatename_fidpair (acts on the single qubit identified by the
        # # its index within the template string), then convert to a GateString/Circuit
        # tmpl_instance = [ [gatename_fidpair_list[i] for i in tmpl_row]  for tmpl_row in tmpl ]
        # for gfp_list in tmpl_instance: # gatestring-fiducialpair list: one (gn-prepstr,gn-measstr) per qubit

        prep = tuple( (gfp_list[i][0] for i in range(nQubits)) ) # just the prep-part (OLD prep_noncore)
        meas = tuple( (gfp_list[i][1] for i in range(nQubits)) ) # just the meas-part (OLD meas_noncore)

        #OLD: back when we tried iterating over *all* core fiducial pairs
        # (now we think/know this is unnecessary - the "true idle" fidpairs suffice)
        #for prep_core in _itertools.product(*([singleQfiducials]*nCore) ):
        #
        #    #construct prep, a gatename-string, from prep_noncore and prep_core
        #    prep = list(prep_noncore)
        #    for i,core_ql in enumerate(core_filter):
        #        prep[ qubit_filter.index(core_ql) ] = prep_core[i]
        #    prep = tuple(prep)

        prepFid = _objs.GateString(())
        for i,el in enumerate(prep):
            prepFid = prepFid + _onqubit(el,qubit_filter[i])

        #OLD: back when we tried iterating over *all* core fiducial pairs
        # (now we think/know this is unnecessary - the "true idle" fidpairs suffice)
        #    for meas_core in [0]: # DEBUG _itertools.product(*([singleQfiducials]*nCore) ):
        #
        #        #construct meas, a gatename-string, from meas_noncore and meas_core
        #        meas = list(meas_noncore)
        #        #for i,core_ql in enumerate(core_filter):
        #        #    meas[ qubit_filter.index(core_ql) ] = meas_core[i]
        #        meas = tuple(meas)
                
        measFid = _objs.GateString(())
        for i,el in enumerate(meas):
            measFid = measFid + _onqubit(el,qubit_filter[i])

        #print("PREPMEAS = ",prepFid,measFid)
        
        gstr_L0 = prepFid + measFid            # should be a GateString
        gstr_L1 = prepFid + idleStr + measFid  # should be a GateString
        ps=gateset._calc().prs_as_polys(prepLbl, effectLbls, gstr_L1 )
        qs=gateset._calc().prs_as_polys(prepLbl, effectLbls, gstr_L0 )
        added = False
        for elbl,p,q in zip(effectLbls,ps,qs):
            amped = p + -1*q # the amplified poly
            Jrow = _np.array([[ amped.deriv(iParam).evaluate(dummy) for iParam in _slct.as_array(wrtParams)]])
            if _np.linalg.norm(Jrow) < 1e-8: continue  # row of zeros can fool matrix_rank
            
            Jtest = _np.concatenate((J,Jrow),axis=0) 
            testRank = _np.linalg.matrix_rank(Jtest, tol=RANK_TOL)
            #print("find_amped_polys_for_syntheticidle: ",prep,meas,elbl," => rank ",testRank, " (Np=",Np,")")
            if testRank > Jrank:
                J = Jtest
                Jrank = testRank
                if not added:
                    gatename_fidpair_list = [ (prep[i],meas[i]) for i in range(nQubits) ]
                    selected_gatename_fidpair_lists.append( gatename_fidpair_list )
                    added = True # only add fidpair once per elabel loop!
                if Jrank == Np: break # this is the largest rank J can take!

    #DEBUG
    #print("DB: J = (wrt = ",wrtParams,")")
    #_mt.print_mx(J,width=4,prec=1)
    #print("DB: svals of J for synthetic idle: ", _np.linalg.svd(J, compute_uv=False))
    
    return J, Jrank, selected_gatename_fidpair_lists


def get_fidpairs_needed_to_access_amped_polys(qubit_filter, core_filter, germPowerStr, amped_polyJ,
                                              idle_gatename_fidpair_lists, gateset,
                                              singleQfiducials=None, prepLbl=None, effectLbls=None,
                                              wrtParams=None, verbosity=0):
    """
    Computes the fiducial pairs needed to amplify the known-amplifiable
    polynomials corresponding to fiducialpair+germ probabilities.

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
        `germPowerStr` ideally act upon.  This is often a proper subset
        of `qubit_filter` since errors are allowed on qubits which neighbor
        the core qubits in addition to the core qubits themselves.

    germPowerStr : GateString
        The (non-synthetic-idle) germ power string under consideration.

    amped_polyJ : numpy.ndarray
        A jacobian matrix whose rowspace gives the space of amplifiable 
        parameters.  The shape of this matrix is `(Namplified, Np)`, where
        `Namplified` is the number of independent amplified parameters and 
        `Np` is the total number of parameters under consideration (the 
        length of `wrtParams`).  This function seeks to find fiducial pairs
        which amplify this same space of parameters.

    idle_gatename_fidpair_lists : list
        A list of the fiducial pairs which amplify the entire space given
        by `amped_polyJ` for the germ when it is repeated enough to be a
        synthetic idle.  The strategy for finding fiducial pairs in the
        present case it to just monkey with the *core-qubit* parts of the
        *measurement* idle fiducials (non-core qubits are ideally the idle,
        and one can either modify the prep or the measure to "catch" what 
        the non-idle `germPowerStr` does to the amplified portion of the 
        state space).

    gateset : GateSet
        The gate set used to compute the polynomial expressions of probabilities
        to first-order.  Thus, this gate set should always have (simulation)
        type "termorder:1".

    singleQfiducials : list, optional
        A list of gate-name tuples (e.g. `('Gx',)`) which specify a set of single-
        qubit fiducials to use when trying to amplify gate parameters.  Note that
        no qubit "state-space" label is required here (i.e. *not* `(('Gx',1),)`);
        the tuples just contain single-qubit gate *names*.  If None, then
        `[(), ('Gx',), ('Gy',)]` is used by default.

    prepLbl : Label, optional
        The state preparation label to use.  If None, then the first (and
        usually the only) state prep label of `gateset` is used, so it's
        usually fine to leave this as None.

    effectLbls : list, optional
        The list of POVM effect labels to use, as a list of `Label` objects.
        These are *compiled* POVM effect labels, so something like "Mdefault_0",
        and if None the default is all the effect labels of the first POVM of
        `gateset`, which is usually what you want.

    wrtParams : slice, optional
        The parameters being considered for amplification.  (This should be
        the same as that used to produce `idle_gatename_fidpair_lists`).

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.

    Returns
    -------
    fidpair_lists : list
        The selected fiducial pairs, each in "gatename-fidpair-list" format.
        See :function:`find_amped_polys_for_syntheticidle` for details.
    """
    printer = _VerbosityPrinter.build_printer(verbosity)
    
    if prepLbl is None:
        prepLbl = list(gateset.preps.keys())[0]
    if effectLbls is None:
        povmLbl = list(gateset.povms.keys())[0]
        effectLbls = list(gateset.povms[povmLbl].keys())
    if singleQfiducials is None:
        # TODO: assert gate set has Gx and Gy gates?
        singleQfiducials = [(), ('Gx',), ('Gy',)] # ('Gx','Gx')
        
    #dummy = 0.05*_np.ones(gateset.num_params(),'d') # for evaluating derivs...
    #dummy = 0.05*_np.arange(1,gateset.num_params()+1) # for evaluating derivs...
    dummy = 5.0*_np.random.random(gateset.num_params()) + 0.5*_np.ones(gateset.num_params(),'d')
     # expect terms to be either coeff*x or coeff*x^2 - (b/c of latter case don't eval at zero)
    
    #OLD: selected_fidpairs = []
    gatename_fidpair_lists = []
    if wrtParams is None: wrtParams = slice(0,gateset.num_params())
    Np = _slct.length(wrtParams) 
    Namped = amped_polyJ.shape[0]; assert(amped_polyJ.shape[1] == Np)
    J = _np.empty( (0,Namped), 'complex'); Jrank = 0
    
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
                   len(idle_gatename_fidpair_lists)*(3**(nCore)), nCore,
                   3**(2*nQubits)))

    already_tried = set()
    cores = [None] + list(_itertools.product(*([singleQfiducials]*nCore) ))
      # try *no* core insertion at first - leave as idle - before going through them...

    for prep_core in cores: # weird loop order b/c we don't expect to need this one
        if prep_core is not None: # I don't think this *should* happen
            _warnings.warn(("Idle's prep fiducials only amplify %d of %d"
                            " directions!  Falling back to vary prep on core")
                           % (Jrank,Namped))
            
        for gfp_list in idle_gatename_fidpair_lists:
            print("GFP list = ",gfp_list)
            prep_noncore = tuple( (gfp_list[i][0] for i in range(nQubits)) ) # just the prep-part
            meas_noncore = tuple( (gfp_list[i][1] for i in range(nQubits)) ) # just the meas-part
    
            if prep_core is None:
                prep = prep_noncore # special case where we try to leave it unchanged.
            else:
                # construct prep, a gatename-string, from prep_noncore and prep_core
                prep = list(prep_noncore)
                for i,core_ql in enumerate(core_filter):
                    prep[ qubit_filter.index(core_ql) ] = prep_core[i]
                prep = tuple(prep)
    
            prepFid = _objs.GateString(())
            for i,el in enumerate(prep):
                prepFid = prepFid + _onqubit(el,qubit_filter[i])
                
            #for meas in _itertools.product(*([singleQfiducials]*nQubits) ):
            #for meas_core in _itertools.product(*([singleQfiducials]*nCore) ):
            for meas_core in cores:

                if meas_core is None:
                    meas = meas_noncore
                else:
                    #construct meas, a gatename-string, from meas_noncore and meas_core
                    meas = list(meas_noncore)
                    for i,core_ql in enumerate(core_filter):
                        meas[ qubit_filter.index(core_ql) ] = meas_core[i]
                    meas = tuple(meas)
                
                measFid = _objs.GateString(())
                for i,el in enumerate(meas):
                    measFid = measFid + _onqubit(el,qubit_filter[i])
                #print("CONSIDER: ",prep,"-",meas)
            
                gstr = prepFid + germPowerStr + measFid  # should be a GateString
                if gstr in already_tried: continue
                else: already_tried.add(gstr)
                
                ps=gateset._calc().prs_as_polys(prepLbl, effectLbls, gstr)
                #OLD: Jtest = J
                added = False
                for elbl,p in zip(effectLbls,ps):
                    #print(" POLY = ",p)
                    #For each fiducial pair (included pre/effect), determine how the
                    # (polynomial) probability relates to the *amplified* directions 
                    # (also polynomials - now encoded by a "Jac" row/vec)
                    prow = _np.array([ p.deriv(iParam).evaluate(dummy) for iParam in _slct.as_array(wrtParams)]) # complex
                    Jrow = _np.array([[ _np.vdot(prow,amped_row) for amped_row in amped_polyJ]]) # complex
                    if _np.linalg.norm(Jrow) < 1e-8: continue  # row of zeros can fool matrix_rank
                    
                    Jtest = _np.concatenate((J,Jrow),axis=0)  
                    testRank = _np.linalg.matrix_rank(Jtest, tol=RANK_TOL)
                    if testRank > Jrank:
                        #print("ACCESS")
                        #print("ACCESS: ",prep,meas,testRank, _np.linalg.svd(Jtest, compute_uv=False))
                        J = Jtest
                        Jrank = testRank
                        if not added:
                            gatename_fidpair_lists.append([ (prep[i],meas[i]) for i in range(nQubits) ])
                            added = True
                        #OLD selected_fidpairs.append( (prepFid, measFid) )
                        if Jrank == Namped: # then we've selected enough pairs to access all of the amplified directions
                            return gatename_fidpair_lists # (i.e. the rows of `amped_polyJ`)
                        
     
    #DEBUG
    #print("DEBUG: J = ")
    #_mt.print_mx(J)
    #print("SVals = ",_np.linalg.svd(J, compute_uv=False))
    #print("Nullspace = ")
    #_gt.print_mx(pygsti.tools.nullspace(J))
    
    raise ValueError(("Could not find sufficient fiducial pairs to access "
                      "all the amplified directions - only %d of %d were accessible")
                     % (Jrank,Namped))
    #_warnings.warn(("Could not find sufficient fiducial pairs to access "
    #                  "all the amplified directions - only %d of %d were accessible")
    #                 % (Jrank,Namped))
    #return gatename_fidpair_lists # (i.e. the rows of `amped_polyJ`)
    
    
def tile_idle_fidpairs(nQubits, idle_gatename_fidpair_lists, maxIdleWeight):
    """
    "Tile" a set of fiducial pairs sufficient for amplifying all the true-idle
    errors on `maxIdleWeight` qubits (so with weight up to `maxIdleWeight` 
    onto `nQubits` qubits.

    This function essentaily converts fiducial pairs that amplify all 
    up-to-weight-k errors on k qubits to fiducial pairs that amplify all
    up-to-weight-k errors on `nQubits` qubits (where `k = maxIdleWeight`).

    Parameters
    ----------
    nQubits : int
        The number of final qubits.

    idle_gatename_fidpair_lists : list
        A list of the fiducial pairs which amplify the errors on 
        `maxIdleWeight` qubits (so with weight up to `maxIdleWeight`).
        Each element of this list is a fiducial pair in 
        "gatename-fidpair-list" format.  These are the fiducial pairs
        to "tile".

    maxIdleWeight : int
        The number of qubits and maximum amplified error weight for
        the fiducial pairs given by `idle_gatename_fidpair_lists`.
    
    Returns
    -------
    fidpairs : list
        A list of `(prep,meas)` 2-tuples, where `prep` and `meas` are
        :class:`GateString` objects, giving the tiled fiducial pairs.
    """

    # "Tile w/overlap" the fidpairs for a k-qubit subset (where k == maxIdleWeight)
    
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
    
    tmpl = get_kcoverage_template(nQubits, maxIdleWeight)
    final_fidpairs = []
    
    for gatename_fidpair_list in idle_gatename_fidpair_lists:
        # replace 0..(k-1) in each template string with the corresponding
        # gatename_fidpair (acts on the single qubit identified by the
        # its index within the template string), then convert to a GateString/Circuit
        tmpl_instance = [ [gatename_fidpair_list[i] for i in tmpl_row]  for tmpl_row in tmpl ]
        for tmpl_instance_row in tmpl_instance: 
            # tmpl_instance_row row is nQubits long; elements give the 
            # gate *names* to perform on that qubit.
            prep_gates = []
            meas_gates = []
            for iQubit,gatename_fidpair in enumerate(tmpl_instance_row):
                prep_gatenames, meas_gatenames = gatename_fidpair
                prep_gates.extend( [_Lbl(gatename,iQubit) for gatename in prep_gatenames ]) 
                meas_gates.extend( [_Lbl(gatename,iQubit) for gatename in meas_gatenames ]) 
            final_fidpairs.append( (_objs.GateString(prep_gates),
                                    _objs.GateString(meas_gates)) )
            
    _lt.remove_duplicates_in_place(final_fidpairs)    
    return final_fidpairs
    
def tile_cloud_fidpairs(template_gatename_fidpair_lists, template_germPower, L, template_germ, clouds):
    """
    Take a "cloud template", giving the fiducial pairs for a germ power acting
    on qubits labeled 0 to `cloudsize-1`, and map those fiducial pairs into
    fiducial pairs for all the qubits by placing in parallel the pairs for
    as many non-overlapping clouds as possible.  This function performs a
    function analogous to :function:`tile_idle_fidpairs` except here we tile
    fiducial pairs for non-idle operations.

    Parameters
    ----------
    template_gatename_fidpair_lists : list 
        A list of the fiducial pairs for the given template - that is, the
        pairs with which amplify all the desired errors for `template_germPower`
        (acting on qubits labeled by the integers 0 to the cloud size minus one).

    template_germPower : GateString
        The germ power string under consideration.  This gives the action on
        the "core" qubits of the clouds, and is needed to construct the 
        final fiducial + germPower + fiducial sequences returned by this 
        function.

    L : int
        The maximum length used to construct template_germPower.  This is only 
        needed to tag elements of the returned `sequences` list.

    template_germ : GateString
        The germ string under consideration.  This is only needed to tag
        elements of the returned `sequences` list and place elements in 
        the returned `germs` list.

    clouds : list
        A list of `(cloud_dict, template_to_cloud_map)` tuples specifying the
        set of equivalent clouds corresponding to the template.

    Returns
    -------
    sequences : list
        A list of (GateString, L, germ, "XX","XX") tuples specifying the final
        "tiled" fiducial pairs sandwiching `germPowerStr` for as many clouds in
        parallel as possible.  Actual qubit labels (not the always-integer
        labels used in templates) are used in these strings. The "XX" elements
        are placeholders for future use. There are no duplicates in this list.

    germs : list
        A list of GateString objects giving all the germs (with appropriate
        qubit labels).
    """    
    unused_clouds = list(clouds)
    sequences = []
    germs = []
    
    while(len(unused_clouds) > 0):
        
        #figure out what clouds can be processed in parallel
        first_unused = unused_clouds[0] # a cloud_dict, template_to_cloud_map tuple
        parallel_clouds = [first_unused]
        parallel_qubits = set(first_unused[0]['qubits']) # qubits used by parallel_clouds
        del unused_clouds[0]
        
        to_delete = []
        for i,cloud in enumerate(unused_clouds):
            if len(parallel_qubits.intersection(cloud[0]['qubits'])) == 0:
                parallel_qubits.update(cloud[0]['qubits'])
                parallel_clouds.append(cloud)
                to_delete.append(i)
        for i in reversed(to_delete):
            del unused_clouds[i]
            
        #Create gate sequence "info-tuples" by processing in parallel the
        # list of parallel_clouds
        
        for template_gatename_fidpair_list in template_gatename_fidpair_lists:
            prepStr = []
            measStr = []
            germStr = []
            germPowerStr = []
            for cloud in parallel_clouds:
                cloud_dict, template_to_cloud_map = cloud
                cloud_to_template_map = { c:t for t,c in template_to_cloud_map.items() }
                
                germ = template_germ.map_state_space_labels(template_to_cloud_map)
                germPower = template_germPower.map_state_space_labels(template_to_cloud_map)

                for cloud_ql in cloud_dict['qubits']:
                    prep,meas = template_gatename_fidpair_list[cloud_to_template_map[cloud_ql]] # gate-name lists
                    prepStr.extend( [_Lbl(name,cloud_ql) for name in prep] )
                    measStr.extend( [_Lbl(name,cloud_ql) for name in meas] )

                germStr.extend( list(germ) )
                germPowerStr.extend( list(germPower) )
                #TODO: use parallel gate labels when possible here?
                
            germs.append( _objs.GateString(germStr) )
            sequences.append( (_objs.GateString(prepStr + germPowerStr + measStr), L, germs[-1], "XX", "XX") )
              # gatestring, L, germ, prepFidIndex, measFidIndex??
        
    # return a list of gate strings (duplicates removed)
    return _lt.remove_duplicates(sequences), _lt.remove_duplicates(germs)        
  
    
def reps_for_synthetic_idle(gateset, germStr, nqubits, core_qubits):
    """
    Return the number of times `germStr` must be repeated to form a synthetic
    idle gate.

    Parameters
    ----------
    gateset : GateSet
        A gate set containing matrix representations of all the gates
        in `germStr`.

    germStr : GateString
        The germ gate string to repeat.

    nqubits : int
        The total number of qubits that `gateset` acts on.  This
        is used primarily for sanity checks.

    core_qubits : list
        A list of the qubit labels upon which `germStr` ideally acts 
        nontrivially.  This could be inferred from `germStr` but serves
        as a sanity check and more concrete specification of what
        state space the gate action takes place within.

    Returns
    -------
    int
    """
    # First, get a dense representation of germStr on core_qubits
    # Note: only works with one level of embedding...
    def extract_gate(g):
        """ Get the gate action as a dense gate on core_qubits """
        if isinstance(g, _objs.EmbeddedGateMap):
            assert(len(g.stateSpaceLabels.labels) == 1) # 1 tensor product block
            assert(len(g.stateSpaceLabels.labels[0]) == nqubits) # expected qubit count
            qubit_labels = g.stateSpaceLabels.labels[0]
            
            # for now - assume we know the form of qubit_labels
            assert(list(qubit_labels) == [('Q%d'%i) for i in range(nqubits)] or 
                   list(qubit_labels) == [ i for i in range(nqubits)]) 
            new_qubit_labels = []
            for core_ql in core_qubits:
                if core_ql in qubit_labels: new_qubit_labels.append(core_ql) # same convention!
                elif ("Q%d" % core_ql) in qubit_labels: new_qubit_labels.append("Q%d" % core_ql) # HACK!
            ssl = _StateSpaceLabels(new_qubit_labels)
            #DEBUG print("qubit_labels = ", qubit_labels, " new_qubit_labels = ",new_qubit_labels, "targets = ",g.targetLabels)
            assert(all([(tgt in new_qubit_labels) for tgt in g.targetLabels])) # all target qubits should be kept!
            if len(new_qubit_labels) == len(g.targetLabels): 
                # embedded gate acts on entire core-qubit space:
                return g.embedded_gate
            else:                
                return _objs.EmbeddedGate(ssl, g.targetLabels, g.embedded_gate)
        
        elif isinstance(g, _objs.ComposedGateMap):
            return _objs.ComposedGate([extract_gate(f) for f in g.factorgates])
        else:
            raise ValueError("Cannot extract core contrib from %s" % str(type(g)))
        
    core_dim = 4**len(core_qubits)
    product = _np.identity(core_dim, 'd')
    core_gates = {}
    for gl in germStr:
        if gl not in core_gates:
            core_gates[gl] = extract_gate(gateset.gates[gl])
        product = _np.dot(core_gates[gl], product)
    
    # Then just do matrix products until we hit the identity (or a large order)
    reps = 1; target = _np.identity(core_dim,'d')
    repeated = product
    while(_np.linalg.norm(repeated - target) > 1e-6 and reps < 20): # HARDCODED MAX_REPS
        repeated = _np.dot(repeated, product); reps += 1
        
    return reps


def get_candidates_for_core(gateset, core_qubits, candidate_counts, seedStart):
    """
    Returns a list of candidate germs which act on a given set of "core" qubits.

    This function figures out what gates within `gateset` are available to act
    (only) on `core_qubits` and then randomly selects a set of them based on 
    `candidate_counts`.  In each candidate germ, at least one gate will act 
    on *all* of the core qubits (if the core is 2 qubits then this function
    won't return a germ consisting of just 1-qubit gates).

    This list serves as the inital candidate list when a new cloud template is
    created within create_nqubit_sequences.

    Parameters
    ----------
    gateset : GateSet
        The gate set specifying the gates allowed to be in the germs.

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

    seedStart : int 
        A *initial* random number generator seed value to use.  Incrementally
        greater seeds are used for the different keys of `candidate_counts`.

    Returns
    -------
    list : candidate_germs
        A list of GateString objects.
    """
    # or should this be ...for_cloudbank - so then we can check that gates for all "equivalent" clouds exist?

    # collect gates that only act on core_qubits.
    gatelabel_list = []; full_core_list = []
    for gl in gateset.gates.keys():
        if gl.sslbls is None: continue # gates that act on everything (usually just the identity Gi gate)
        if set(gl.sslbls).issubset(core_qubits):
            gatelabel_list.append(gl)
        if set(gl.sslbls) == set(core_qubits):
            full_core_list.append(gl)
        
    # form all low-length strings out of these gates.
    candidate_germs = []
    for i,(germLength, count) in enumerate(candidate_counts.items()):
        if count == "all upto":
            candidate_germs.extend( _gsc.list_all_gatestrings_without_powers_and_cycles(
                    gatelabel_list, maxLength=germLength) )
        else:
            candidate_germs.extend( _gsc.list_random_gatestrings_onelen(
                    gatelabel_list, germLength, count, seed=seedStart+i))

    #filter: make sure there's at least one gate in each germ that acts on the *entire* core
    candidate_germs = [ g for g in candidate_germs if any([(gl in g) for gl in full_core_list]) ] # filter?
    
    return candidate_germs


def create_nqubit_sequences(nQubits, maxLengths, geometry, cnot_edges, maxIdleWeight=1, maxhops=0,
                            extraWeight1Hops=0, extraGateWeight=0, sparse=False, verbosity=0,
                            cache=None, idleOnly=False, algorithm="greedy"):
    """ 
    Generate a list of sequences sufficient for amplifying all of the errors
    given by a geometrically local noise model defined by a graph and various
    maximum weights and maximum hoppings.  This noise model corresponds to the
    :class:`GateSet` returned by :function:`build_nqnoise_gateset` when it is
    given similar arguments.

    Parameters
    ----------
    nQubits : int
        The number of qubits

    maxLengths : list
        A list of integers specifying the different maximum lengths for germ
        powers.  Typically these values start a 1 and increase by powers of
        2, e.g. `[1,2,4,8,16]`.
    
    geometry : {"line","ring","grid","torus"} or QubitGraph
        The type of connectivity among the qubits, specifying a
        graph used to define neighbor relationships.  Alternatively,
        a :class:`QubitGraph` object with 0-`nQubits-1` node labels
        may be passed directly.

    cnot_edges : list, optional
        A list of 2-tuples of (control,target) qubit indices for each
        CNOT gate to be included in the returned GateSet.  If None, then
        the (directed) edges of the `geometry` graph are used.

    maxIdleWeight : int, optional
        The maximum-weight for errors on the global idle gate.

    maxhops : int
        The locality constraint: for a gate, errors (of weight up to the
        maximum weight for the gate) are allowed to occur on the gate's
        target qubits and those reachable by hopping at most `maxhops` times
        from a target qubit along nearest-neighbor links (defined by the 
        `geometry`).  
    
    extraWeight1Hops : int, optional
        Additional hops (adds to `maxhops`) for weight-1 errors.  A value > 0
        can be useful for allowing just weight-1 errors (of which there are 
        relatively few) to be dispersed farther from a gate's target qubits.
        For example, a crosstalk-detecting model might use this.
    
    extraGateWeight : int, optional
        Addtional weight, beyond the number of target qubits (taken as a "base
        weight" - i.e. weight 2 for a 2Q gate), allowed for gate errors.  If
        this equals 1, for instance, then 1-qubit gates can have up to weight-2
        errors and 2-qubit gates can have up to weight-3 errors.

    sparse : bool, optional
        Whether the embedded Lindblad-parameterized gates within the constructed
        `nQubits`-qubit gates are sparse or not.  (This is determied by whether
        they are constructed using sparse basis matrices.)  When sparse, these
        Lindblad gates take up less memory, but their action is slightly slower.
        Usually it's fine to leave this as the default (False), except when
        considering particularly high-weight terms (b/c then the Lindblad gates
        are higher dimensional and sparsity has a significant impact).

    verbosity : int, optional
        The level of detail printed to stdout.  0 means silent.

    cache : dict, optional
        A cache dictionary which holds template information so that repeated 
        calls to `create_nqubit_sequences` can draw on the same pool of
        templates.

    idleOnly : bool, optional
        If True, only sequences for the idle germ are returned.  This is useful
        for idle tomography in particular.

    algorithm : {"greedy","sequential"}
        The algorithm is used internall by 
        :function:`find_amped_polys_for_syntheticidle`.  You should leave this 
        as the default unless you know what you're doing.

    Returns
    -------
    sequences : list
        A list of (GateString, L, germ, "XX","XX") tuples specifying the 
        final sequences categorized by max-length (L) and germ.

    germs : list
        A list of GateString objects specifying all the germs found in
        `sequences`.
    """

    #The algorithm here takes the following basic structure:
    # - compute idle fiducial pairs with a max-weight appropriate for
    #   the true idle gate.
    # - Add the idle germ + fiducial pairs, which amplify all the "idle
    #   parameters" (the parameters of the Gi gate)
    # - precompute other idle fiducial pairs needed for 1 & 2Q synthetic
    #   idles (with maxWeight = gate-error-weight + spreading potential)
    # - To amplify the remaining parameters iterat through the "clouds"
    #   constructed by build_nqnoise_gateset (these essentially give
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
    
    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("Creating full gateset")

    if isinstance(geometry, _objs.QubitGraph):
        qubitGraph = geometry
    else:
        qubitGraph = _objs.QubitGraph.common_graph(nQubits, geometry, directed=False)
        printer.log("Created qubit graph:\n"+str(qubitGraph))

    gateset, clouds = build_nqnoise_gateset(
        nQubits, qubitGraph, cnot_edges, maxIdleWeight, maxhops,
        extraWeight1Hops, extraGateWeight, sparse, verbosity=printer-5,
        sim_type="termorder:1", parameterization="H+S terms", return_clouds=True)
    #print("DB: GATES = ",gateset.gates.keys())
    #print("DB: CLOUDS = ",clouds)

    # clouds is a list of (core_qubits,cloud_qubits) tuples, giving the
    # different "supports" of performing the various gates in the gateset
    # whose parameters we want to amplify.  The 'core' of a cloud is the
    # set of qubits that have a non-trivial ideal action applied to them.
    # The 'qubits' of a cloud are all the qubits that have any action -
    # ideal or error - except that which is the same as the Gi gate.
    
    ideal_gateset = build_nqnoise_gateset(nQubits, qubitGraph, cnot_edges, 0, 0,
                                          0, 0, False, verbosity=printer-5,
                                          sim_type="map", parameterization="H+S") # for testing for synthetic idles
    
    Np = gateset.num_params()
    idleGateStr = _objs.GateString(("Gi",))
    singleQfiducials = [(), ('Gx',), ('Gy',)] # , ('Gx','Gx')
    prepLbl = _Lbl("rho0")
    effectLbls = [ _Lbl("Mdefault_%s" % l) for l in gateset.povms['Mdefault'].keys()]

    # create a gateset with maxIdleWeight qubits that includes all
    # the errors of the actual n-qubit gateset...
    #Note: geometry doens't matter here, since we just look at the idle gate (so just use 'line'; no CNOTs)
    printer.log("Creating \"idle error\" gateset on %d qubits" % maxIdleWeight)
    idle_gateset = build_nqnoise_gateset(maxIdleWeight, 'line', [], maxIdleWeight, maxhops,
                                         extraWeight1Hops, extraGateWeight, sparse, verbosity=printer-5,
                                         sim_type="termorder:1", parameterization="H+S terms")
    idle_params = idle_gateset.gates['Gi'].gpindices # these are the params we want to amplify at first...

    if maxIdleWeight in cache['Idle gatename fidpair lists']:
        printer.log("Getting cached sequences needed for max-weight=%d errors on the idle gate" % maxIdleWeight)
        idle_maxwt_gatename_fidpair_lists = cache['Idle gatename fidpair lists'][maxIdleWeight]
    else:
        #First get "idle germ" sequences since the idle is special
        printer.log("Getting sequences needed for max-weight=%d errors on the idle gate" % maxIdleWeight)
        ampedJ, ampedJ_rank, idle_maxwt_gatename_fidpair_lists = \
            find_amped_polys_for_syntheticidle(list(range(maxIdleWeight)),
                                               idleGateStr, idle_gateset, singleQfiducials,
                                               prepLbl, None, wrtParams=idle_params, 
                                               algorithm=algorithm, verbosity=printer-1)
        #ampedJ, ampedJ_rank, idle_maxwt_gatename_fidpair_lists = None,0,[] # DEBUG GRAPH ISO
        cache['Idle gatename fidpair lists'][maxIdleWeight] = idle_maxwt_gatename_fidpair_lists


    #Since this is the idle, these maxIdleWeight-qubit fidpairs can be "tiled"
    # to the n-qubits
    printer.log("%d \"idle template pairs\".  Tiling these to all %d qubits" % 
                (len(idle_maxwt_gatename_fidpair_lists), nQubits),2)
    idle_fidpairs = tile_idle_fidpairs(nQubits, idle_maxwt_gatename_fidpair_lists, maxIdleWeight)
    printer.log("%d idle pairs found" % len(idle_fidpairs),2)
    
                
    # Create idle sequences by sandwiching Gi^L between all idle fiducial pairs
    sequences = []
    selected_germs = [ idleGateStr ]
    for L in maxLengths:
        for fidpair in idle_fidpairs:
            prepFid, measFid = fidpair
            sequences.append( (prepFid + idleGateStr*L + measFid, L, idleGateStr, "XX", "XX") )
              # gatestring, L, germ, prepFidIndex, measFidIndex??
    printer.log("%d idle sequences (for all max-lengths: %s)" % (len(sequences), str(maxLengths)))
    
    if idleOnly: #Exit now when we just wanted idle-tomography sequences
        return sequences, selected_germs 
    
    #Compute "true-idle" fidpairs for checking synthetic idle errors for 1 & 2Q gates (HARDCODED OK?)
    # NOTE: this works when ideal gates are cliffords and Gi has same type of errors as gates...
    weights = set([ len(gl.sslbls) for gl in gateset.gates if (gl.sslbls is not None)])
    for gateWt in sorted(list(weights)):
        maxSyntheticIdleWt = (gateWt + extraGateWeight) + (gateWt-1) # gate-error-wt + spreading potential
        maxSyntheticIdleWt = min(maxSyntheticIdleWt, nQubits)

        if maxSyntheticIdleWt not in cache['Idle gatename fidpair lists']:
            printer.log("Getting sequences needed for max-weight=%d errors" % maxSyntheticIdleWt)
            printer.log(" on the idle gate (for %d-Q synthetic idles)" % gateWt)
            sidle_gateset = build_nqnoise_gateset(maxSyntheticIdleWt, 'line', [], maxIdleWeight, maxhops,
                                                  extraWeight1Hops, extraGateWeight, sparse, verbosity=printer-5,
                                                  sim_type="termorder:1", parameterization="H+S terms")
            idle_params = sidle_gateset.gates['Gi'].gpindices # these are the params we want to amplify...

            _, _, idle_gatename_fidpair_lists = find_amped_polys_for_syntheticidle(
                list(range(maxSyntheticIdleWt)), idleGateStr, sidle_gateset,
                singleQfiducials, prepLbl, None, wrtParams=idle_params,
                algorithm=algorithm, verbosity=printer-1) 
            #idle_gatename_fidpair_lists = [] # DEBUG GRAPH ISO
            cache['Idle gatename fidpair lists'][maxSyntheticIdleWt] = idle_gatename_fidpair_lists        
        
    #Look for and add additional germs to amplify the *rest* of the gateset's parameters
    Gi_nparams = gateset.gates['Gi'].num_params()
    SPAM_nparams = sum([obj.num_params() for obj in _itertools.chain(gateset.preps.values(), gateset.povms.values())])
    Np_to_amplify = gateset.num_params() - Gi_nparams - SPAM_nparams
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
    for icloud,(core_qubits,cloud_qubits) in enumerate(clouds):
        cloud_dict = {'core': core_qubits, 'qubits': cloud_qubits} # just for clarity, label the pieces

        # Collect "pure gate" params of gates that *exactly* on (just and only) the core_qubits;
        # these are the parameters we want this cloud to amplify.  If all the gates which act on
        # the core act on the entire core (when there are no gates that only act on only a part
        # of the core), then these params will be the *only* ones the choosen germs will amplify.
        # But, if there are partial-core gates, the germs might amplify some of their parameters
        # (e.g. Gx:0 params might get amplified when processing a cloud whose core is [0,1]).
        # This is fine, but we don't demand that such params be amplified, since they *must* be
        # amplified for another cloud with core exaclty equal to the gate's target qubits (e.g. [0])
        wrtParams = set()
        Gi_params = set(_slct.as_array(gateset.gates['Gi'].gpindices))
        pure_gate_labels = []
        for gl in gateset.gates.keys():
            if gl.sslbls is None: continue # gates that act on everything (usually just the identity Gi gate)
            if set(gl.sslbls) == set(core_qubits):
                pure_gate_labels.append(gl)
                wrtParams.update( _slct.as_array(gateset.gates[gl].gpindices) )
        pure_gate_params = wrtParams - Gi_params # (Gi params don't count)
        wrtParams = _slct.list_to_slice( sorted(list(pure_gate_params)), array_ok=True )
        Ngp = _slct.length(wrtParams) # number of "pure gate" params that we want to amplify
                
        J = _np.empty( (0,Ngp), 'complex'); Jrank = 0
        
        printer.log("Cloud %d of %d: qubits = %s, core = %s, nparams = %d" % 
                    (icloud+1,len(clouds),str(cloud_qubits), str(core_qubits), Ngp),2)

        # cache struture:
        #  'Idle gatename fidpair lists' - dict w/keys = ints == max-idle-weights
        #      - values = gatename-fidpair lists (on max-idle-weight qubits)
        #  'Cloud templates' - dict w/ complex cloud-class-identifying keys (tuples)
        #      - values = list of "cloud templates": (gatelabels, qubit_graph, germ_dict) tuples, where
        #        gatelabels is a list/set of the gate labels for this cloud template
        #        qubit_graph is a graph giving the connectivity of the cloud template's qubits
        #        germ_dict is a dict w/keys = germs
        #           - values = (germ_order, access_cache) tuples for each germ, where
        #              germ_order is an integer
        #              access_cache is a dict w/keys = "effective germ reps" = actual_reps % germ_order
        #                 - values = gatename-fidpair lists (on cloud qubits)

        def get_cloud_key(cloud, maxhops, extraWeight1Hops, extraGateWeight):
            """ Get the cache key we use for a cloud """
            return (len(cloud['qubits']), len(cloud['core']), maxhops, extraWeight1Hops, extraGateWeight)

        def map_cloud_template(cloud, gatelabels, graph, template):
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

            #Make sure each has the same number of gate labels
            if len(template_glabels) != len(gatelabels):
                return None

            # Try to match core qubit labels (via gatelabels & graph)
            for possible_perm in _itertools.permutations( cloud['core'] ):
                # possible_perm is a permutation of cloud's core labels, e.g. ('Q1','Q0','Q2')
                # such that the ordering gives the mapping from template index/labels 0 to nCore-1
                possible_template_to_cloud_map = { i:ql for i,ql in enumerate(possible_perm) }

                gr = core_graph.copy()
                for template_edge in template_core_graph.edges():
                    edge = (possible_template_to_cloud_map[template_edge[0]],
                            possible_template_to_cloud_map[template_edge[1]])
                    if gr.has_edge(edge): # works w/directed & undirected graphs
                        gr.remove_edge(edge[0],edge[1])
                    else:
                        break # missing edge -> possible_perm no good
                else: # no missing templage edges!
                    if len(gr.edges()) == 0: # and all edges were present - a match so far!

                        #Now test gate labels
                        for template_gl in template_glabels:
                            gl = template_gl.map_state_space_labels(possible_template_to_cloud_map)
                            if gl not in gatelabels:
                                break
                        else:
                            #All gatelabels match (gatelabels can't have extra b/c we know length are the same)
                            core_map = possible_template_to_cloud_map

                            # Try to match non-core qubit labels (via graph)
                            non_core_qubits = [ ql for ql in cloud['qubits'] if (ql not in cloud['core']) ]
                            for possible_perm in _itertools.permutations( non_core_qubits ):
                                # possible_perm is a permutation of cloud's non-core labels, e.g. ('Q4','Q3')
                                # such that the ordering gives the mapping from template index/labels nCore to nQubits-1
                                possible_template_to_cloud_map = core_map.copy()
                                possible_template_to_cloud_map.update({ i:ql for i,ql in enumerate(possible_perm,start=nCore) })
                                 # now possible_template_to_cloud_map maps *all* of the qubits
                
                                gr =cloud_graph.copy()
                                for template_edge in template_cloud_graph.edges():
                                    edge = (possible_template_to_cloud_map[template_edge[0]],
                                            possible_template_to_cloud_map[template_edge[1]])
                                    if gr.has_edge(edge): # works w/directed & undirected graphs
                                        gr.remove_edge(edge[0],edge[1])
                                    else:
                                        break # missing edge -> possible_perm no good
                                else: # no missing templage edges!
                                    if len(gr.edges()) == 0: # and all edges were present - a match!!!
                                        return possible_template_to_cloud_map
                    
            return None

        def create_cloud_template(cloud, pure_gate_labels, graph):
            """ Creates a new cloud template, currently a (template_glabels, template_graph, germ_dict) tuple """
            nQubits = len(cloud['qubits'])
            cloud_to_template_map = { ql:i for i,ql in enumerate(cloud['core']) } # core qubits always first in template
            cloud_to_template_map.update({ql:i for i,ql in
                                          enumerate(filter(lambda x: x not in cloud['core'],
                                                           cloud['qubits']),start=len(cloud['core'])) }) # then non-core
            template_glabels = [ gl.map_state_space_labels(cloud_to_template_map)
                                 for gl in pure_gate_labels ]
            template_edges = []
            cloud_graph = graph.subgraph(cloud['qubits'])
            for edge in cloud_graph.edges():
                template_edges.append( (cloud_to_template_map[edge[0]],
                                        cloud_to_template_map[edge[1]]) )
            
            template_graph = _objs.QubitGraph( list(range(nQubits)),
                                               initial_edges=template_edges,
                                               directed=graph.directed)
            cloud_template = (template_glabels, template_graph, {})
            template_to_cloud_map = { t:c for c,t in cloud_to_template_map.items() }
            return cloud_template, template_to_cloud_map
            
        
        cloud_class_key = get_cloud_key(cloud_dict, maxhops, extraWeight1Hops, extraGateWeight)
        cloud_class_templates = cache['Cloud templates'][cloud_class_key]
        for cloud_template in cloud_class_templates:
            template_to_cloud_map = map_cloud_template(cloud_dict, pure_gate_labels, qubitGraph, cloud_template)
            if template_to_cloud_map is not None: # a cloud template is found!
                template_glabels, template_graph, _ = cloud_template
                printer.log("Found cached template for this cloud: %d qubits, gates: %s, map: %s" %
                            (len(cloud_qubits), template_glabels, template_to_cloud_map),2)
                break
        else:
            cloud_template, template_to_cloud_map = create_cloud_template(cloud_dict, pure_gate_labels, qubitGraph)
            cloud_class_templates.append(cloud_template)
            printer.log("Created a new template for this cloud: %d qubits, gates: %s, map: %s" %
                        (len(cloud_qubits), cloud_template[0], template_to_cloud_map),2)

        #File this cloud under the found/created "cloud template", as these identify classes of
        # "equivalent" clouds that can be tiled together below
        if id(cloud_template) not in cloudbanks:
            printer.log("Created a new cloudbank (%d) for this cloud" % id(cloud_template),2)
            cloudbanks[id(cloud_template)] = {'template': cloud_template,
                                              'clouds': [] } # a list of (cloud_dict, template->cloud map) tuples
        else:
            printer.log("Adding this cloud to existing cloudbank (%d)" % id(cloud_template),2)
        cloudbanks[id(cloud_template)]['clouds'].append( (cloud_dict,template_to_cloud_map) )

        # *** For the rest of this loop over clouds, we just make sure the identified 
        #     template supports everything we need (it has germs, and fidpairs for all needed L values)
        
        cloud_to_template_map = { c:t for t,c in template_to_cloud_map.items() }
        germ_dict = cloud_template[2] # see above structure
        if len(germ_dict) > 0: # germ_dict should always be non-None
            allLsExist = all([ all([
                    ((_gsc.repeat_count_with_max_length(germ,L) % germ_order) in access_cache)
                    for L in maxLengths ])
                for germ,(germ_order,access_cache) in germ_dict.items() ])
        else: allLsExist = False
        
        if len(germ_dict) == 0 or allLsExist == False:

            if len(germ_dict) == 0:  # we need to do the germ selection using a set of candidate germs
                candidate_counts = {4: 'all upto', 5: 10, 6: 10 } # should be an arg? HARDCODED!
                candidate_germs = get_candidates_for_core(gateset, core_qubits, candidate_counts, seedStart=1234)
                  # candidate_germs should only use gates with support on *core* qubits?
                germ_type = "Candidate"
            else:
                # allLsExist == False, but we have the germs already (since cloud_template is not None),
                # and maybe some L-value support
                #TODO: use qubit_map to translate germ_dict keys to candidate germs
                candidate_germs = [ germ.map_state_space_labels(template_to_cloud_map)
                                    for germ in germ_dict ] # just iterate over the known-good germs
                germ_type = "Pre-computed"
                

            consecutive_unhelpful_germs = 0
            for candidate_germ in candidate_germs:
                template_germ = candidate_germ.map_state_space_labels(cloud_to_template_map)

                #Check if we need any new L-value support for this germ
                if template_germ in germ_dict:
                    germ_order, access_cache = germ_dict[template_germ]
                    if all([ ((_gsc.repeat_count_with_max_length(template_germ,L) % germ_order)
                              in access_cache) for L in maxLengths ]):
                        continue # move on to the next germ
                
                #Let's see if we want to add this germ
                sireps = reps_for_synthetic_idle(ideal_gateset, candidate_germ, nQubits, core_qubits)
                syntheticIdle = candidate_germ * sireps
                maxWt = min( (len(core_qubits) + extraGateWeight) + (len(core_qubits) - 1),
                             len(cloud_qubits) ) # gate-error-wt + spreading potential
                printer.log("%s germ: %s (synthetic idle %s)" %
                            (germ_type,str(candidate_germ),str(syntheticIdle)),3)


                old_Jrank = Jrank
                printer.log("Finding amped-polys for clifford synIdle w/max-weight = %d" % maxWt,3)
                J, Jrank, sidle_gatename_fidpair_lists = find_amped_polys_for_clifford_syntheticidle(
                    cloud_qubits, core_qubits, cache['Idle gatename fidpair lists'], syntheticIdle, maxWt, gateset,
                    singleQfiducials, prepLbl, effectLbls, J, Jrank, wrtParams, printer-2)
                #J, Jrank, sidle_gatename_fidpair_lists = None, 0, None # DEBUG GRAPH ISO

                #J, Jrank, sidle_gatename_fidpair_lists = find_amped_polys_for_syntheticidle(
                #    cloud_qubits, syntheticIdle, gateset, singleQfiducials, prepLbl, effectLbls, J, Jrank, wrtParams)
                
                nNewAmpedDirs = Jrank - old_Jrank  #OLD: not nec. equal to this: len(sidle_gatename_fidpair_lists)
                if nNewAmpedDirs > 0: # then there are some "directions" that this germ amplifies that previous ones didn't...
                    printer.log("Germ amplifies %d additional parameters (so %d of %d amplified for this base cloud)" %
                                (nNewAmpedDirs, Jrank, Ngp), 3) # assume each cloud amplifies an independent set of params

                    if template_germ not in germ_dict:
                        germ_dict[template_germ] = (sireps, {}) # germ_order, access_cache
                    access_fidpairs_cache = germ_dict[template_germ][1] # see above structure
                    access_fidpairs_cache[0] = sidle_gatename_fidpair_lists # idle: effective_reps == 0
                    
                    amped_polyJ = J[-nNewAmpedDirs:, :] # just the rows of the Jacobian corresponding to
                                                        # the directions we want the current germ to amplify
                    #print("DB: amped_polyJ = ",amped_polyJ)
                    #print("DB: amped_polyJ svals = ",_np.linalg.svd(amped_polyJ, compute_uv=False))
                    
                    #Figure out which fiducial pairs access the amplified directions at each value of L
                    for L in maxLengths:
                        reps = _gsc.repeat_count_with_max_length(candidate_germ,L)
                        if reps == 0: continue # don't process when we don't use the germ at all...
                        effective_reps = reps % sireps
                        germPower = candidate_germ * effective_reps # germ^effective_reps  

                        if effective_reps not in access_fidpairs_cache:
                            printer.log("Finding the fiducial pairs needed to amplify %s^%d (L=%d, effreps=%d)" %
                                       (str(candidate_germ),reps,L,effective_reps),4)
                            gatename_fidpair_lists = get_fidpairs_needed_to_access_amped_polys(
                                cloud_qubits, core_qubits, germPower, amped_polyJ, sidle_gatename_fidpair_lists,
                                gateset, singleQfiducials, prepLbl, effectLbls, wrtParams, printer-3)
                            #gatename_fidpair_lists = None # DEBUG GRAPH ISO
                            printer.log("Found %d fiducial pairs" % len(gatename_fidpair_lists),4)

                            #Convert cloud -> template gatename fidpair lists
                            template_gatename_fidpair_lists = []
                            for gatename_fidpair_list in gatename_fidpair_lists:
                                template_gatename_fidpair_lists.append( [
                                    gatename_fidpair_list[cloud_qubits.index(template_to_cloud_map[tl])]
                                               for tl in range(len(cloud_qubits)) ] ) # tl ~= "Q0" is *label* of a template qubit
                            #E.G if template qubit labels are [0,1,2] , cloud_qubits = [Q3,Q4,Q2] and map is 0->Q4, 1->Q2, 2->Q3
                            # then we need to know what *index* Q4,Q2,Q3 are with the template, i.e the index of
                            # template_to_cloud[0], template_to_cloud[1], ... in cloud_qubits
                            
                            access_fidpairs_cache[effective_reps] = gatename_fidpair_lists
                        else:
                            printer.log("Already found fiducial pairs needed to amplify %s^%d (L=%d, effreps=%d)" %
                                       (str(candidate_germ),reps,L,effective_reps),4)                    
                    
                    if Jrank == Np: # really this will never happen b/c we'll never amplify SPAM and gauge directions...
                        break       # instead exit after we haven't seen a germ that amplifies anything new in a while  
                    consecutive_unhelpful_germs = 0
                else:
                    consecutive_unhelpful_germs += 1
                    printer.log(("No additional amplified params: %d consecutive unhelpful germs."
                                 % consecutive_unhelpful_germs), 3)
                    if consecutive_unhelpful_germs == 5: # ??
                        break # next cloudbank
        else:
            printer.log("Fiducials for all L-values are cached!", 3)


    for icb,cloudbank in enumerate(cloudbanks.values()):
        template_glabels, template_graph, germ_dict = cloudbank['template']

        printer.log("Tiling cloudbank %d of %d: %d clouds, template labels = %s, qubits = %s" %
                    (icb+1,len(cloudbanks),len(cloudbank['clouds']),
                     str(template_glabels),str(template_graph.nqubits)),2)

        # At this point, we have a cloud template w/germ_dict that
        #  supports all the L-values we need.  Now tile to this
        #  cloudbank.
        for template_germ,(germ_order,access_cache) in germ_dict.items():
            
            printer.log("Tiling for template germ = %s" % str(template_germ), 3)
            add_germs = True
            for L in maxLengths:
                reps = _gsc.repeat_count_with_max_length(template_germ,L)
                if reps == 0: continue # don't process when we don't use the germ at all...
                effective_reps = reps % germ_order
                template_gatename_fidpair_lists = access_cache[effective_reps]

                template_germPower = template_germ * reps # germ^reps
                addl_seqs, addl_germs = tile_cloud_fidpairs(template_gatename_fidpair_lists,
                                                            template_germPower, L, template_germ,
                                                            cloudbank['clouds'])
                
                sequences.extend(addl_seqs)
                if add_germs: # addl_germs is independent of L - so just add once
                    selected_germs.extend(addl_germs)
                    add_germs = False
                                
                printer.log("After tiling L=%d to cloudbank, have %d sequences, %d germs" %
                            (L, len(sequences),len(selected_germs)),4)
                
                
    printer.log("Done: %d sequences, %d germs" % (len(sequences),len(selected_germs)))
    return sequences, selected_germs


def _get_kcoverage_template_k2(n):
    """ Special case where k == 2 -> use hypercube construction """
    # k = 2 implies binary strings of 0's and 1's
    def bitstr(nQubits, bit):
        """ Returns a length-nQubits list of the values of the bit-th bit in the integers 0->nQubits"""
        return [ ((i>>bit) & 1) for i in range(nQubits)]
    def invert(bstr):
        return [ (0 if x else 1) for x in bstr ]
    
    half = [ bitstr(n,k) for k in range(int(_np.ceil(_np.math.log(n,2)))) ]
    other_half = [ invert(bstr) for bstr in half ]
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
    0 to `n-1`.

    This notion has application to idle-gate fiducial pair tiling, when we have
    found a set of fiducial pairs for `k` qubits and want to find a set of
    sequences on `n > k` qubits such that any subset of `k` qubits experiences
    the entire set of (`k`-qubit) fiducial pairs.  Simply take the k-coverage
    template and replace the letters (0 to `k-1`) with the per-qubit 1Q pieces
    of each k-qubit fiducial pair.
    
    Parameters
    ----------
    n, k : int
        The sequence (word) length and letter count as described above.

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
    cols = [ list() for i in range(k) ]
    for row in _itertools.permutations(range(k),k):
        for i in range(k): 
            cols[i].append(row[i])
    nRows = len(cols[0])
    if verbosity > 0: print("get_template(n=%d,k=%d):" % (n,k))

    # Now add cols k to n-1:
    for a in range(k,n): # a is index of column we're adding
        if verbosity > 1: print(" - Adding column %d: currently %d rows" % (a,nRows))
        
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
        
        col_a = [None]*nRows # the new column - start with None sentinels in all current rows

        # added heuristic step for increased efficiency:
        # preference each open element of the a-th column by taking the
        # "majority vote" among what the existing column values "want"
        # the a-th column to be.
        pref_a = []
        for m in range(nRows):
            votes = _collections.defaultdict(lambda: 0)
            for existing_cols in _itertools.combinations(range(a),k-1):
                vals = set(range(k)) #the values the k-1 existing + a-th columns need to take
                vals = vals - set([cols[i][m] for i in existing_cols])
                if len(vals) > 1: continue # if our chosen existing cols don't
                  # even cover all but one val then don't cast a vote
                assert(len(vals) == 1)
                val = vals.pop() # pops the *only* element
                votes[val] += 1

            majority = None; majority_cnt = 0
            for ky,val in votes.items():
                if val > majority_cnt:
                     majority,majority_cnt = ky,val
            pref_a.append( majority )

        
        for existing_cols in _itertools.combinations(range(a-1,-1,-1),k-1): # reverse-range(a) == heuristic
            if verbosity > 2: print("  - check perms are present for cols %s" % str(existing_cols + (a,)) )
            
            #make sure cols existing_cols + [a] take on all the needed permutations
            # Since existing_cols already takes on all permuations minus the last
            # value (which is determined as it's the only one missing from the k-1
            # existing cols) - we just need to *complete* each existing row and possibly
            # duplicate + add rows to ensure all completions exist. 
            for desired_row in _itertools.permutations(range(k),k):
                                
                matching_rows = [] # rows that match desired_row on existing_cols
                open_rows = [] # rows with a-th column open (unassigned)
                
                for m in range(nRows):
                    if all([ cols[existing_cols[i]][m] == desired_row[i] for i in range(k-1)]):
                        # m-th row matches desired_row on existing_cols
                        matching_rows.append(m)
                    if col_a[m] is None:
                        open_rows.append(m)
                
                if verbosity > 3: print("   - perm %s: %d rows, %d match perm, %d open"
                                % (str(desired_row), nRows, len(matching_rows), len(open_rows)) )
                v = {'value': desired_row[k-1], 'alternate_rows': matching_rows}
                placed = False

                #Best: find a row that already has the value we're looking for (set via previous iteration)
                for m in matching_rows:
                    if col_a[m] and col_a[m]['value'] == desired_row[k-1]: # a perfect match! - no need to take an open slot
                        updated_alts = [ i for i in col_a[m]['alternate_rows'] if i in matching_rows ]
                        if verbosity > 3: print("    -> existing row (index %d) perfectly matches!" % m)
                        col_a[m]['alternate_rows'] = updated_alts; placed=True; break
                if placed: continue

                #Better: find an open row that prefers the value we want to place in it
                for m in matching_rows:
                    if col_a[m] is None and pref_a[m] == desired_row[k-1]: # slot is open & prefers the value we want to place in it - take it!
                        if verbosity > 3: print("    -> open preffered row (index %d) matches!" % m)
                        col_a[m] = v; placed=True; break
                if placed: continue

                #Good: find any open row (FUTURE: maybe try to shift for preference first?)
                for m in matching_rows:
                    if col_a[m] is None: # slot is open - take it!
                        if verbosity > 3: print("    -> open row (index %d) matches!" % m)
                        col_a[m] = v; placed=True; break
                if placed: continue
                        
                # no open slots
                # option1: (if there are any open rows)
                #  Look to swap an existing value in a matching row 
                #   to an open row allowing us to complete the matching
                #   row using the current desired_row.
                open_rows = set(open_rows) # b/c use intersection below
                shift_soln_found = False
                if len(open_rows) > 0:
                    for m in matching_rows:
                        # can assume col_a[m] is *not* None given above logic
                        ist = open_rows.intersection(col_a[m]['alternate_rows'])
                        if len(ist) > 0:
                            m2 = ist.pop() # just get the first element
                            # move value in row m to m2, then put v into the now-open m-th row
                            col_a[m2] = col_a[m]
                            col_a[m] = v
                            if verbosity > 3: print("    -> row %d >> row %d, and row %d matches!" % (m,m2,m))
                            shift_soln_found = True
                            break
                            
                if shift_soln_found == False:
                    # no shifting can be performed to place v into an open row,
                    # so we just create a new row equal to desired_row on existing_cols.
                    # How do we choose the non-(existing & last) colums? For now, just
                    # replicate the first element of matching_rows:
                    if verbosity > 3: print("    -> creating NEW row.")
                    for i in range(a):
                        cols[i].append( cols[i][matching_rows[0]] )
                    col_a.append(v)
                    nRows += 1

        #Check for any remaining open rows that we never needed to use.
        # (the a-th column can then be anything we want, so as heuristic
        #  choose a least-common value in the row already)
        for m in range(nRows):
            if col_a[m] is None:
                cnts = { v:0 for v in range(k) } #count of each possible value
                for i in range(a): cnts[ cols[i][m] ] += 1
                val = 0; mincnt = cnts[ 0 ]
                for v,cnt in cnts.items(): # get value with minimal count
                    if cnt < mincnt:
                        val = v; mincnt = cnt
                col_a[m] = {'value': val, 'alternate_rows': "N/A" }
                        
        # a-th column is complete; "cement" it by replacing 
        # value/alternative_rows dicts with just the values
        col_a = [ d['value'] for d in col_a ]
        cols.append(col_a)    
                
    #convert cols to "strings" (rows)
    assert(len(cols) == n)
    rows = []
    for i in range(len(cols[0])):
        rows.append( [ cols[j][i] for j in range(n)] )
        
    if verbosity > 0: print(" Done: %d rows total" % len(rows))
    return rows
        

def check_kcoverage_template(rows, n, k, verbosity=0):
    """ 
    Verify that `rows` satisfies the `k`-coverage conditions for length-`n`
    sequences.  Raises an AssertionError if the check fails.

    Parameters
    ----------
    rows : list
        A list of k-coverage words.  The same as whas is returned by
        :function:`get_kcoverage_template`.

    n, k : int
        The k-coverate word length and letter count.

    verbosity : int, optional
        Amount of detail to print to stdout.
    """
    if verbosity > 0: print("check_template(n=%d,k=%d)" % (n,k))
    
    #for each set of k qubits (of the total n qubits)
    for cols_to_check in _itertools.combinations(range(n),k):
        if verbosity > 1: print(" - checking cols %s" % str(cols_to_check))
        for perm in _itertools.permutations(range(k),k):
            for m,row in enumerate(rows):
                if all([ row[i] == perm[i] for i in range(k) ]):
                    if verbosity > 2: print("  - perm %s: found at row %d" % (str(perm),m))
                    break
            else:
                assert(False), \
                    "Permutation %s on qubits (cols) %s is not present!" % (str(perm),str(cols_to_check))
    if verbosity > 0: print(" check succeeded!")                                                             
    
