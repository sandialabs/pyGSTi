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

from .. import objects as _objs
from ..tools import basistools as _bt
from ..tools import gatetools as _gt
from ..objects.labeldicts import StateSpaceLabels as _StateSpaceLabels

from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from ..baseobjs.basisconstructors import sqrt2, id2x2, sigmax, sigmay, sigmaz
from ..baseobjs import Basis as _Basis
from ..baseobjs import Label as _Lbl

from .gatesetconstruction import basis_build_vector as _basis_build_vector
    

def iter_basis_inds(weight):
    basisIndList = [ [1,2,3] ]*weight #assume pauli 1Q basis, and only iterate over non-identity els
    for basisInds in _itertools.product(*basisIndList):
        yield basisInds

def basisProductMatrix(sigmaInds, sparse):
    sigmaVec = (id2x2/sqrt2, sigmax/sqrt2, sigmay/sqrt2, sigmaz/sqrt2)
    M = _np.identity(1,'complex')
    for i in sigmaInds:
        M = _np.kron(M,sigmaVec[i])
    return _sps.csr_matrix(M) if sparse else M

def nparams_nqnoise_gateset(nQubits, geometry="line", maxIdleWeight=1, maxhops=0,
                            extraWeight1Hops=0, extraGateWeight=0, requireConnected=False,
                            independent1Qgates=True, ZZonly=False, verbosity=0):
    # noise can be either a seed or a random array that is long enough to use

    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("Computing parameters for a %d-qubit %s gateset" % (nQubits,geometry))

    qubitGraph = _objs.QubitGraph.common_graph(nQubits, geometry)
    #printer.log("Created qubit graph:\n"+str(qubitGraph))

    def idle_count_nparams(maxWeight):
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
        ret = 0
        #Note: no contrib from idle noise (already parameterized)
        for wt, maxHops in weight_maxhops_tuples:
            possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops),'i')
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



def build_nqnoise_gateset(nQubits, geometry="line", maxIdleWeight=1, maxhops=0,
                          extraWeight1Hops=0, extraGateWeight=0, sparse=False,
                          gateNoise=None, prepNoise=None, povmNoise=None,
                          sim_type="matrix", parameterization="H+S", verbosity=0): #, debug=False):
    """ TODO: docstring - entire module! """
    assert(sim_type in ("matrix","map") or sim_type.startswith("termorder"))
    assert(parameterization in ("H+S","H+S terms","H+S clifford terms"))
    from pygsti.construction import std1Q_XY # the base gate set for 1Q gates
    from pygsti.construction import std2Q_XYICNOT # the base gate set for 2Q (CNOT) gate
    
    # noise can be either a seed or a random array that is long enough to use

    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("Creating a %d-qubit %s gateset" % (nQubits,geometry))

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

    qubitGraph = _objs.QubitGraph.common_graph(nQubits, geometry)
    printer.log("Created qubit graph:\n"+str(qubitGraph))

    printer.log("Creating Idle:")
    gs.gates[_Lbl('Gi')] = build_nqn_global_idle(qubitGraph, maxIdleWeight, sparse,
                                        sim_type, parameterization, printer-1)

    #1Q gates: X(pi/2) & Y(pi/2) on each qubit
    Gx = std1Q_XY.gs_target.gates['Gx']
    Gy = std1Q_XY.gs_target.gates['Gy'] 
    weight_maxhops_tuples_1Q = [(1,maxhops+extraWeight1Hops)] + \
                               [ (1+x,maxhops) for x in range(1,extraGateWeight+1) ]
    for i in range(nQubits):
        printer.log("Creating 1Q X(pi/2) gate on qubit %d!!" % i)
        gs.gates[_Lbl("Gx",i)] = build_nqn_composed_gate(
            Gx, (i,), qubitGraph, weight_maxhops_tuples_1Q,
            idle_noise=gs.gates['Gi'], loc_noise_type="manylittle",
            sparse=sparse, sim_type=sim_type, parameterization=parameterization,
            verbosity=printer-1)

        printer.log("Creating 1Q Y(pi/2) gate on qubit %d!!" % i)
        gs.gates[_Lbl("Gy",i)] = build_nqn_composed_gate(
            Gy, (i,), qubitGraph, weight_maxhops_tuples_1Q,
            idle_noise=gs.gates['Gi'], loc_noise_type="manylittle",
            sparse=sparse, sim_type=sim_type, parameterization=parameterization,
            verbosity=printer-1)
        
    #2Q gates: CNOT gates along each graph edge
    Gcnot = std2Q_XYICNOT.gs_target.gates['Gcnot']
    weight_maxhops_tuples_2Q = [(1,maxhops+extraWeight1Hops),(2,maxhops)] + \
                               [ (2+x,maxhops) for x in range(1,extraGateWeight+1) ]
    for i,j in qubitGraph.edges(): #note: all edges have i<j so "control" of CNOT is always lower index (arbitrary)
        printer.log("Creating CNOT gate between qubits %d and %d!!" % (i,j))
        gs.gates[_Lbl("Gcnot",(i,j))] = build_nqn_composed_gate(
            Gcnot, (i,j), qubitGraph, weight_maxhops_tuples_2Q,
            idle_noise=gs.gates['Gi'], loc_noise_type="manylittle",
            sparse=sparse, sim_type=sim_type, parameterization=parameterization,
            verbosity=printer-1)


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



    ##OLD - before we had Lindblad SPAMVecs
    #prepFactors = [ pygsti.obj.TPParameterizedSPAMVec(pygsti.construction.basis_build_vector("0", basis1Q))
    #                for i in range(nQubits)]
    #if prepNoise is not None:
    #    if isinstance(prepNoise,tuple): # use as (seed, strength)
    #        seed,strength = prepNoise
    #        rndm = _np.random.RandomState(seed)
    #        depolAmts = _np.abs(rndm.random_sample(nQubits)*strength)
    #    else:
    #        depolAmts = prepNoise[0:nQubits]
    #    for amt,vec in zip(depolAmts,prepFactors): vec.depolarize(amt) 
    #gs.preps['rho0'] = pygsti.obj.TensorProdSPAMVec('prep',prepFactors)
    #
    #factorPOVMs = []
    #for i in range(nQubits):
    #    effects = [ (l,pygsti.construction.basis_build_vector(l, basis1Q)) for l in ["0","1"] ]
    #    factorPOVMs.append( pygsti.obj.TPPOVM(effects) )
    #if povmNoise is not None:
    #    if isinstance(povmNoise,tuple): # use as (seed, strength)
    #        seed,strength = povmNoise
    #        rndm = _np.random.RandomState(seed)
    #        depolAmts = _np.abs(rndm.random_sample(nQubits)*strength)
    #    else:
    #        depolAmts = povmNoise[0:nQubits]
    #    for amt,povm in zip(depolAmts,factorPOVMs): povm.depolarize(amt) 
    #gs.povms['Mdefault'] = pygsti.obj.TensorProdPOVM( factorPOVMs )
        
    printer.log("DONE! - returning GateSet with dim=%d and gates=%s" % (gs.dim, list(gs.gates.keys())))
    return gs
    

def get_Lindblad_factory(sim_type, parameterization):
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
    def f(gateMatrix, unitaryPostfactor=None,
          ham_basis="pp", nonham_basis="pp", cptp=True,
          nonham_diagonal_only=False, truncate=True, mxBasis="pp"):
        return cls.from_gate_matrix(gateMatrix, unitaryPostfactor,
                                    ham_basis, nonham_basis, cptp,
                                    nonham_diagonal_only, truncate,
                                    mxBasis, evotype)
    return f
                                    

def get_Static_factory(sim_type, parameterization):
    if parameterization == "H+S":
        if sim_type == "matrix":
            return lambda g,b: _objs.StaticGate(g)
        elif sim_type == "map":
            return lambda g,b: _objs.StaticGate(g) # TODO: create StaticGateMap

    elif parameterization in ("H+S terms","H+S clifford terms"):
        assert(sim_type.startswith("termorder"))
        evotype = "svterm" if parameterization == "H+S terms" else "cterm"
        
        def f(gateMatrix, mxBasis="pp"):
            return _objs.LindbladParameterizedGateMap.from_gate_matrix(
                None, gateMatrix, None, None, mxBasis=mxBasis, evotype=evotype)
                # a LindbladParameterizedGate with None as ham_basis and nonham_basis => no parameters
              
        return f
    raise ValueError("Cannot create Static gate factory for ",sim_type, parameterization)


def build_nqn_global_idle(qubitGraph, maxWeight, sparse=False, sim_type="matrix", parameterization="H+S", verbosity=0):
    assert(maxWeight <= 2), "Only `maxWeight` equal to 0, 1, or 2 is supported"

    if sim_type == "matrix": 
        Composed = _objs.ComposedGate
        Embedded = _objs.EmbeddedGate
    else:
        Composed = _objs.ComposedGateMap
        Embedded = _objs.EmbeddedGateMap
    Lindblad = get_Lindblad_factory(sim_type, parameterization)

    #OLD
    #if sparse:
    #    Lindblad = _objs.LindbladParameterizedGateMap
    #    Composed = _objs.ComposedGateMap
    #    Embedded = _objs.EmbeddedGateMap
    #else:
    #    Lindblad = _objs.LindbladParameterizedGate
    #    Composed = _objs.ComposedGate
    #    Embedded = _objs.EmbeddedGate
    
    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("*** Creating global idle ***")
    
    termgates = [] # gates to compose
    ssAllQ = [tuple(['Q%d'%i for i in range(qubitGraph.nqubits)])]
    basisAllQ = _Basis('pp', 2**qubitGraph.nqubits, sparse=sparse)
    
    nQubits = qubitGraph.nqubits
    possible_err_qubit_inds = _np.arange(nQubits)
    nPossible = nQubits  
    for wt in range(1,maxWeight+1):
        printer.log("Weight %d: %d possible qubits" % (wt,nPossible),2)
        basisEl_Id = basisProductMatrix(_np.zeros(wt,'i'),sparse)
        wtId = _sps.identity(4**wt,'d','csr') if sparse else  _np.identity(4**wt,'d')
        wtBasis = _Basis('pp', 2**wt, sparse=sparse)
        
        for err_qubit_inds in _itertools.combinations(possible_err_qubit_inds, wt):
            if len(err_qubit_inds) == 2 and not qubitGraph.is_directly_connected(err_qubit_inds[0],err_qubit_inds[1]):
                continue # TO UPDATE - check whether all wt indices are a connected subgraph

            errbasis = [basisEl_Id]
            for err_basis_inds in iter_basis_inds(wt):        
                error = _np.array(err_basis_inds,'i') #length == wt
                basisEl = basisProductMatrix(error,sparse)
                errbasis.append(basisEl)

            printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_inds,len(errbasis)), 3)
            errbasis = _Basis(matrices=errbasis, sparse=sparse) #single element basis (plus identity)
            termErr = Lindblad(wtId, ham_basis=errbasis, nonham_basis=errbasis, cptp=True,
                               nonham_diagonal_only=True, truncate=True, mxBasis=wtBasis)
        
            err_qubit_global_inds = err_qubit_inds
            fullTermErr = Embedded(ssAllQ, [('Q%d'%i) for i in err_qubit_global_inds],
                                   termErr, basisAllQ.dim)
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
#    possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops),'i')
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
    Final gate is a composition of: 
    targetOp(target qubits) -> idle_noise(all_qubits) -> loc_noise(local_qubits)
    
    where `idle_noise` is given by the `idle_noise` parameter and loc_noise is given
    by the other params.  loc_noise can be implemented either by 
    a single embedded LindbladParameterizedGate with all relevant error generators,
    or as a composition of embedded-single-error-term gates (see param `loc_noise_type`)
    
    Parameters
    ----------
    
    idle_noise : Gate or boolean
        either given as an existing gate (on all qubits) or a boolean indicating
        whether a composition of weight-1 noise terms (separately on all the qubits),
        is created.  If `apply_idle_noise_to == "nonlocal"` then `idle_noise` is *only*
        applied to the non-local qubits and `idle_noise` must be a ComposedGate or
        ComposedMap with nQubits terms so that individual terms for each qubit can
        be extracted as needed.

    TODO   
    """
    if sim_type == "matrix": 
        Composed = _objs.ComposedGate
        Embedded = _objs.EmbeddedGate
    else:
        Composed = _objs.ComposedGateMap
        Embedded = _objs.EmbeddedGateMap
    Static = get_Static_factory(sim_type, parameterization)
    Lindblad = get_Lindblad_factory(sim_type, parameterization)

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
    basisAllQ = _Basis('pp', 2**qubitGraph.nqubits, sparse=sparse)
    fullTargetOp = Embedded(ssAllQ, ['Q%d'%i for i in target_qubit_inds],
                            Static(targetOp,"pp"), basisAllQ.dim) 

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
                [ Embedded(ssAllQ, ('Q%d'%i,), Lindblad(Id_1Q.copy()),basisAllQ.dim)
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
            target_qubit_inds, max([hops for _,hops in weight_maxhops_tuples]) ), 'i') # node labels are ints
        nLocal = len(all_possible_err_qubit_inds)
        basisEl_Id = basisProductMatrix(_np.zeros(nPossible,'i'),sparse) #identity basis el
        
        for wt, maxHops in weight_maxhops_tuples:
            possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops),'i') # node labels are ints
            nPossible = len(possible_err_qubit_inds)
            possible_to_local = [ all_possible_err_qubit_inds.index(
                possible_err_qubit_inds[i]) for i in range(nPossible)]
            printer.log("Weight %d, max-hops %d: %d possible qubits of %d local" %
                        (wt,maxHops,nPossible,nLocal),3)
            
            for err_qubit_inds in _itertools.combinations(list(range(nPossible)), wt):
                # err_qubit_inds are in range [0,nPossible-1] qubit indices
                #Future: check that err_qubit_inds marks qubits that are connected
                err_qubit_local_inds = possible_to_local[err_qubit_inds]
                                                        
                for err_basis_inds in iter_basis_inds(wt):  
                    error = _np.zeros(nLocal,'i')
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
                                localErr, basisAllQ.dim)
        printer.log("Lindblad gate w/dim=%d and %d params (from error basis of len %d) -> embedded to gate w/dim=%d" %
                    (localErr.dim, localErr.num_params(), len(errbasis), fullLocalErr.dim),2)

        
    elif loc_noise_type == "manylittle": 
        # make a composed-gate of embedded single-basis-element Lindblad-gates,
        #  one for each specified error term  
            
        loc_noise_termgates = [] #list of gates to compose
        
        for wt, maxHops in weight_maxhops_tuples:
                
            ## loc_noise_errinds = [] # list of basis indices for all local-error terms 
            possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops),'i') # we know node labels are integers
            nPossible = len(possible_err_qubit_inds) # also == "nLocal" in this case
            basisEl_Id = basisProductMatrix(_np.zeros(wt,'i'),sparse) #identity basis el

            wtId = _sps.identity(4**wt,'d','csr') if sparse else _np.identity(4**wt,'d')
            wtBasis = _Basis('pp', 2**wt, sparse=sparse)

            printer.log("Weight %d, max-hops %d: %d possible qubits" % (wt,maxHops,nPossible),3)
            
            for err_qubit_local_inds in _itertools.combinations(list(range(nPossible)), wt):
                # err_qubit_inds are in range [0,nPossible-1] qubit indices
                #Future: check that err_qubit_inds marks qubits that are connected

                errbasis = [basisEl_Id]
                for err_basis_inds in iter_basis_inds(wt):  
                    error = _np.array(err_basis_inds,'i') #length == wt
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
                                       termErr, basisAllQ.dim)
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

