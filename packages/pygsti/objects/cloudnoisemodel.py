""" Defines the CloudNoiseModel class and supporting functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import copy as _copy
import itertools as _itertools
import collections as _collections
import scipy.sparse as _sps
import warnings as _warnings

from . import model as _mdl
from . import operation as _op
from . import spamvec as _sv
from . import povm as _povm
from . import qubitgraph as _qgraph
from ..tools import optools as _gt

from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from ..baseobjs import Basis as _Basis
from ..baseobjs import Dim as _Dim
from ..baseobjs import Label as _Lbl

from ..baseobjs.basisconstructors import sqrt2, id2x2, sigmax, sigmay, sigmaz

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


class CloudNoiseModel(_mdl.ImplicitOpModel):
    """ 
    A noisy n-qubit model using a low-weight and geometrically local
    error model with a common "global idle" operation.  
    """

    def __init__(self, nQubits, gatedict, availability=None,
                 qubit_labels=None, geometry="line",
                 maxIdleWeight=1, maxSpamWeight=1, maxhops=0,
                 extraWeight1Hops=0, extraGateWeight=0, sparse=False,
                 sim_type="auto", parameterization="H+S",
                 spamtype="lindblad", addIdleNoiseToAllGates=True,
                 errcomp_type="gates", verbosity=0):
        """ 
        TODO: update docstring (see LocalNoiseModel)
        Create a noisy n-qubit model using a low-weight and geometrically local
        error model with a common "global idle" operation.  
    
        This type of model is generally useful for performing GST on a multi-
        qubit model, whereas classes like :class:`LocalNoiseModel`
        are more useful for creating static (non-parameterized) models.
    
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
            CNOT gate to be included in the returned Model.  If None, then
            the (directed) edges of the `geometry` graph are used.
    
        maxIdleWeight : int, optional
            The maximum-weight for errors on the global idle gate.
    
        maxSpamWeight : int, optional
            The maximum-weight for SPAM errors when `spamtype == "linblad"`.
    
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
            `maxSpamWeight` are included.  Tensor-product operations require that
            the state prep and POVM effects have a tensor-product structure; the
            "tensorproduct" mode exists for historical reasons and is *deprecated*
            in favor of `"lindblad"`; use it only if you know what you're doing.
    
        addIdleNoiseToAllGates: bool, optional
            Whether the global idle should be added as a factor following the 
            ideal action of each of the non-idle gates.
        
        verbosity : int, optional
            An integer >= 0 dictating how must output to send to stdout.
        """
        if qubit_labels is None:
            qubit_labels = tuple(range(nQubits))
        if availability is None:
            availability = {}
        
        #Set members
        self.nQubits = nQubits
        self.gatedict = _collections.OrderedDict(
            [(gn,_np.array(gate)) for gn,gate in gatedict.items()]) # only hold numpy arrays (so copying is clean)
        self.availability = availability
        self.qubit_labels = qubit_labels
        self.geometry = geometry
        self.maxIdleWeight = maxIdleWeight
        self.maxSpamWeight = maxSpamWeight
        self.maxhops = maxhops
        self.extraWeight1Hops = extraWeight1Hops
        self.extraGateWeight = extraGateWeight
        self.sparse = sparse
        self.parameterization = parameterization
        self.spamtype = spamtype
        self.addIdleNoiseToAllGates = addIdleNoiseToAllGates
        self.errcomp_type = errcomp_type

        #Process "auto" sim_type
        _,evotype = _gt.split_lindblad_paramtype(parameterization)
        assert(evotype in ("densitymx","svterm","cterm")), "State-vector evolution types not allowed."
        if sim_type == "auto":
            if evotype in ("svterm", "cterm"): sim_type = "termorder:1"
            else: sim_type = "map" if nQubits > 2 else "matrix"
        assert(sim_type in ("matrix","map") or sim_type.startswith("termorder"))

        lizardArgs = {'add_idle_noise': addIdleNoiseToAllGates , 'errcomp_type': errcomp_type, 'sparse_expm': sparse }
        super(CloudNoiseModel,self).__init__(self.qubit_labels, "pp", {}, CloudNoiseLayerLizard,
                                             lizardArgs, sim_type=sim_type, evotype=evotype)

        printer = _VerbosityPrinter.build_printer(verbosity)
        geometry_name = "custom" if isinstance(geometry, _qgraph.QubitGraph) else geometry
        printer.log("Creating a %d-qubit local-noise %s model" % (nQubits,geometry_name))
        
        #Full preps & povms -- maybe another option
        ##Create initial model with std prep & POVM
        #eLbls = []; eExprs = []
        #formatStr = '0' + str(nQubits) + 'b'
        #for i in range(2**nQubits):
        #    eLbls.append( format(i,formatStr))
        #    eExprs.append( str(i) )    
        #Qlbls = tuple( ['Q%d' % i for i in range(nQubits)] )
        #mdl = pygsti.construction.build_explicit_model(
        #    [Qlbls], [], [], 
        #    effectLabels=eLbls, effectExpressions=eExprs)
    
        if isinstance(geometry, _qgraph.QubitGraph):
            qubitGraph = geometry
        else:
            qubitGraph = _qgraph.QubitGraph.common_graph(nQubits, geometry, directed=False,
                                                         qubit_labels=qubit_labels)
            printer.log("Created qubit graph:\n"+str(qubitGraph))
    
        if maxIdleWeight > 0:
            printer.log("Creating Idle:")
            self.operation_blks[_Lbl('globalIdle')] = _build_nqn_global_noise(
                qubitGraph, maxIdleWeight, sparse,
                sim_type, parameterization, errcomp_type, printer-1)
        else:
            self.addIdleNoiseToAllGates = False # there is no idle noise to add!
            #self.operation_blks[_Lbl('globalIdle')] = _build_nqn_global_noise(
            
        # a dictionary of "cloud" objects
        # keys = (target_qubit_indices, cloud_qubit_indices) tuples
        # values = list of gate-labels giving the gates associated with that cloud (necessary?)
        self.clouds = _collections.OrderedDict()

        #Get gates availability
        primitive_ops = []
        oneQ_gates_and_avail = _collections.OrderedDict()
        twoQ_gates_and_avail = _collections.OrderedDict()
        for gateName, gate in self.gatedict.items(): # gate is a numpy array
            gate_nQubits = int(round(_np.log2(gate.shape[0])/2))
            if gate_nQubits not in (1,2):
                raise ValueError("Only 1- and 2-qubit gates are supported.  %s acts on %d qubits!"
                                 % (str(gateName), gate_nQubits))

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
            self.availability[gateName] = tuple(availList)

            if gate_nQubits == 1:
                oneQ_gates_and_avail[gateName] = (gate,availList)
            elif gate_nQubits == 2:
                twoQ_gates_and_avail[gateName] = (gate,availList)
            
        #1Q gates: e.g. X(pi/2) & Y(pi/2) on each qubit
        weight_maxhops_tuples_1Q = [(1,maxhops+extraWeight1Hops)] + \
                                   [ (1+x,maxhops) for x in range(1,extraGateWeight+1) ]
        cloud_maxhops = max( [mx for wt,mx in weight_maxhops_tuples_1Q] ) # max of max-hops
    
        ssAllQ = [tuple(qubit_labels)] # also node-names of qubitGraph ?
        #basisAllQ = _Basis('pp', 2**qubitGraph.nqubits, sparse=sparse)
        basisAllQ_dim = _Dim(2**qubitGraph.nqubits)

        EmbeddedDenseOp = _op.EmbeddedDenseOp if sim_type == "matrix" else _op.EmbeddedOp
        StaticDenseOp = _get_Static_factory(sim_type, parameterization) # always a *gate*

        for gn, (gate,availList) in oneQ_gates_and_avail.items():
            for (i,) in availList: # so 'i' is target qubit label
                #Target operations
                printer.log("Creating 1Q %s gate on qubit %s!!" % (gn,str(i)))
                self.operation_blks[_Lbl(gn,i)] = EmbeddedDenseOp(
                    ssAllQ, [i], StaticDenseOp(gate,"pp"), basisAllQ_dim)
                primitive_ops.append(_Lbl(gn,i))
                        
                self.operation_blks[_Lbl('CloudNoise_'+gn,i)] = _build_nqn_cloud_noise(
                    (i,), qubitGraph, weight_maxhops_tuples_1Q,
                    errcomp_type=errcomp_type, sparse=sparse, sim_type=sim_type,
                    parameterization=parameterization, verbosity=printer-1)
        
                cloud_inds = tuple(qubitGraph.radius((i,), cloud_maxhops))
                cloud_key = ( (i,), tuple(sorted(cloud_inds)) ) # (sets are unhashable)
                if cloud_key not in self.clouds: self.clouds[cloud_key] = []
                self.clouds[cloud_key].append( _Lbl(gn,i) )
            
        #2Q gates: e.g. CNOT gates along each graph edge
        weight_maxhops_tuples_2Q = [(1,maxhops+extraWeight1Hops),(2,maxhops)] + \
                                   [ (2+x,maxhops) for x in range(1,extraGateWeight+1) ]
        cloud_maxhops = max( [mx for wt,mx in weight_maxhops_tuples_2Q] ) # max of max-hops
        for gn, (gate,availList) in twoQ_gates_and_avail.items():
            for (i,j) in availList: # so 'i' and 'j' are target qubit labels
                printer.log("Creating %s gate between qubits %s and %s!!" % (gn,str(i),str(j)))
                self.operation_blks[_Lbl(gn,(i,j))] = EmbeddedDenseOp(
                    ssAllQ, [i,j], StaticDenseOp(gate,"pp"), basisAllQ_dim)
                self.operation_blks[_Lbl('CloudNoise_'+gn,(i,j))] = _build_nqn_cloud_noise(
                    (i,j), qubitGraph, weight_maxhops_tuples_2Q,
                    errcomp_type=errcomp_type, sparse=sparse, sim_type=sim_type,
                    parameterization=parameterization, verbosity=printer-1)
                primitive_ops.append(_Lbl(gn,(i,j)))
                
                cloud_inds = tuple(qubitGraph.radius((i,j), cloud_maxhops))
                cloud_key = (tuple(sorted([i,j])), tuple(sorted(cloud_inds)))
                if cloud_key not in self.clouds: self.clouds[cloud_key] = []
                self.clouds[cloud_key].append( _Lbl(gn,(i,j)) )
    
        #SPAM
        if spamtype == "static" or maxSpamWeight == 0:
            if maxSpamWeight > 0:
                _warnings.warn(("`spamtype == 'static'` ignores the supplied "
                                "`maxSpamWeight=%d > 0`") % maxSpamWeight )
            self.prep_blks[_Lbl('rho0')] = _sv.ComputationalSPAMVec([0]*nQubits,evotype)
            self.povm_blks[_Lbl('Mdefault')] = _povm.ComputationalBasisPOVM(nQubits,evotype)
            
        elif spamtype == "tensorproduct": 
    
            _warnings.warn("`spamtype == 'tensorproduct'` is deprecated!")
            basis1Q = _Basis("pp",2)
            prep_factors = []; povm_factors = []
    
            v0 = _basis_build_vector("0", basis1Q)
            v1 = _basis_build_vector("1", basis1Q)
    
            # Historical use of TP for non-term-based cases?
            #  - seems we could remove this. FUTURE REMOVE?
            povmtyp = rtyp = "TP" if parameterization in \
                             ("CPTP","H+S","S","H+S+A","S+A","H+D+A","D+A","D") \
                             else parameterization
            
            for i in range(nQubits):
                prep_factors.append(
                    _sv.convert(_sv.StaticSPAMVec(v0), rtyp, basis1Q) )
                povm_factors.append(
                    _povm.convert(_povm.UnconstrainedPOVM( ([
                        ('0',_sv.StaticSPAMVec(v0)),
                        ('1',_sv.StaticSPAMVec(v1))]) ), povmtyp, basis1Q) )
    
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
        
            self.prep_blks[_Lbl('rho0')] = _sv.TensorProdSPAMVec('prep', prep_factors)
            self.povm_blks[_Lbl('Mdefault')] = _povm.TensorProdPOVM(povm_factors)
    
        elif spamtype == "lindblad":
    
            prepPure = _sv.ComputationalSPAMVec([0]*nQubits,evotype)
            prepNoiseMap = _build_nqn_global_noise(qubitGraph, maxSpamWeight, sparse, sim_type, parameterization, errcomp_type, printer-1)
            self.prep_blks[_Lbl('rho0')] = _sv.LindbladSPAMVec(prepPure, prepNoiseMap, "prep")
    
            povmNoiseMap = _build_nqn_global_noise(qubitGraph, maxSpamWeight, sparse, sim_type, parameterization, errcomp_type, printer-1)
            self.povm_blks[_Lbl('Mdefault')] = _povm.LindbladPOVM(povmNoiseMap, None, "pp")
    
        else:
            raise ValueError("Invalid `spamtype` argument: %s" % spamtype)
        
        self.set_primitive_op_labels(primitive_ops)
        self.set_primitive_prep_labels(tuple(self.prep_blks.keys()))
        self.set_primitive_povm_labels(tuple(self.povm_blks.keys()))
        #(no instruments)
        
        printer.log("DONE! - created Model with dim=%d and op-blks=%s" % (self.dim, list(self.operation_blks.keys())))    

    def get_clouds(self):
        return self.clouds


def _get_Lindblad_factory(sim_type, parameterization, errcomp_type):
    """ Returns a function that creates a Lindblad-type gate appropriate
        given the simulation type and parameterization """
    _,evotype = _gt.split_lindblad_paramtype(parameterization)
    if errcomp_type == "gates":
        if evotype ==  "densitymx":
            cls = _op.LindbladDenseOp if sim_type == "matrix" \
                  else _op.LindbladOp
        elif evotype in ("svterm","cterm"):
            assert(sim_type.startswith("termorder"))        
            cls = _op.LindbladOp
        else:
            raise ValueError("Cannot create Lindblad gate factory for ",sim_type, parameterization)

        #Just call cls.from_operation_matrix with appropriate evotype
        def _f(opMatrix, #unitaryPostfactor=None,
               proj_basis="pp", mxBasis="pp", relative=False):
            unitaryPostfactor=None #we never use this in gate construction
            p = parameterization
            if relative:
                if parameterization == "CPTP": p = "GLND"
                elif "S" in parameterization: p = parameterization.replace("S","s")
                elif "D" in parameterization: p = parameterization.replace("D","d")
            return cls.from_operation_obj(opMatrix, p, unitaryPostfactor,
                                     proj_basis, mxBasis, truncate=True)
        return _f

    elif errcomp_type == "errorgens":
        def _f(errorGen,
               proj_basis="pp", mxBasis="pp", relative=False):
            p = parameterization
            if relative:
                if parameterization == "CPTP": p = "GLND"
                elif "S" in parameterization: p = parameterization.replace("S","s")
                elif "D" in parameterization: p = parameterization.replace("D","d")
            _,evotype,nonham_mode,param_mode = _op.LindbladOp.decomp_paramtype(p)
            return _op.LindbladErrorgen.from_error_generator(errorGen, proj_basis, proj_basis, 
                                                               param_mode, nonham_mode, mxBasis,
                                                               truncate=True, evotype=evotype)
        return _f

    else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)

                                    

def _get_Static_factory(sim_type, parameterization):
    """ Returns a function that creates a static-type gate appropriate 
        given the simulation and parameterization """
    _,evotype = _gt.split_lindblad_paramtype(parameterization)
    if evotype == "densitymx":
        if sim_type == "matrix":
            return lambda g,b: _op.StaticDenseOp(g)
        elif sim_type == "map":
            return lambda g,b: _op.StaticDenseOp(g) # TODO: create StaticGateMap?

    elif evotype in ("svterm", "cterm"):
        assert(sim_type.startswith("termorder"))
        def _f(opMatrix, mxBasis="pp"):
            return _op.LindbladOp.from_operation_matrix(
                None, opMatrix, None, None, mxBasis=mxBasis, evotype=evotype)
                # a LindbladDenseOp with None as ham_basis and nonham_basis => no parameters
              
        return _f
    raise ValueError("Cannot create Static gate factory for ",sim_type, parameterization)


def _build_nqn_global_noise(qubitGraph, maxWeight, sparse=False, sim_type="matrix",
                           parameterization="H+S", errcomp_type="gates", verbosity=0):
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
        the model this gate is destined for.  This affects what type of 
        gate objects (e.g. `ComposedDenseOp` vs `ComposedOp`) are created.
    
    parameterization : str
        The type of parameterizaton for the constructed gate. E.g. "H+S",
        "H+S terms", "H+S clifford terms", "CPTP", etc.

    errcomp_type : {"onebig","manylittle"} TODO docstring update these to "gates" and "errorgens"
        Whether the `loc_noise` portion of the constructed gate should be a
        a single Lindblad gate containing all the allowed error terms (onebig)
        or the composition of many Lindblad gates each containing just a single
        error term (manylittle).  The resulting gate performs the same action
        regardless of the value set here; this just affects how the gate is
        structured internally.

    verbosity : int, optional
        An integer >= 0 dictating how must output to send to stdout.

    Returns
    -------
    LinearOperator
    """
    assert(maxWeight <= 2), "Only `maxWeight` equal to 0, 1, or 2 is supported"

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
    Lindblad = _get_Lindblad_factory(sim_type, parameterization, errcomp_type)
      #constructs a gate or errorgen based on value of errcomp_type
    
    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("*** Creating global idle ***")
    
    termops = [] # gates or error generators to compose
    ssAllQ = [tuple(['Q%d'%i for i in range(qubitGraph.nqubits)])]
    #basisAllQ = _Basis('pp', 2**qubitGraph.nqubits, sparse=sparse) # TODO: remove - all we need is its 'dim' below
    basisAllQ_dim = _Dim(2**qubitGraph.nqubits)
    
    nQubits = qubitGraph.nqubits
    possible_err_qubit_inds = _np.arange(nQubits)
    nPossible = nQubits  
    for wt in range(1,maxWeight+1):
        printer.log("Weight %d: %d possible qubits" % (wt,nPossible),2)
        basisEl_Id = basisProductMatrix(_np.zeros(wt,_np.int64),sparse)
        if errcomp_type == "gates":
            wtNoErr = _sps.identity(4**wt,'d','csr') if sparse else  _np.identity(4**wt,'d')
        elif errcomp_type == "errorgens":
            wtNoErr = _sps.csr_matrix((4**wt,4**wt)) if sparse else  _np.zeros((4**wt,4**wt),'d')
        else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
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
            termErr = Lindblad(wtNoErr, proj_basis=errbasis, mxBasis=wtBasis)
        
            err_qubit_global_inds = err_qubit_inds
            fullTermErr = Embedded(ssAllQ, [('Q%d'%i) for i in err_qubit_global_inds],
                                   termErr, basisAllQ_dim)
            assert(fullTermErr.num_params() == termErr.num_params())
            printer.log("Lindblad gate w/dim=%d and %d params -> embedded to gate w/dim=%d" %
                        (termErr.dim, termErr.num_params(), fullTermErr.dim))
                    
            termops.append( fullTermErr )
                
    if errcomp_type == "gates":
        return Composed(termops)
    elif errcomp_type == "errorgens":
        errgen = Composed(termops)
        LindbladOp = _op.LindbladDenseOp if sim_type == "matrix" \
            else _op.LindbladOp
        return LindbladOp(None, errgen, sparse)
    else: assert(False)


def _build_nqn_cloud_noise(target_qubit_inds, qubitGraph, weight_maxhops_tuples,
                           errcomp_type="onebig", sparse=False, sim_type="matrix",
                           parameterization="H+S", verbosity=0):
    """ 
    Create an n-qubit gate that is a composition of:
    
    `targetOp(target_qubits) -> idle_noise(all_qubits) -> loc_noise(local_qubits)`

    where `idle_noise` is given by the `idle_noise` argument and `loc_noise` is
    given by the rest of the arguments.  `loc_noise` can be implemented either
    by a single (n-qubit) embedded Lindblad gate with all relevant error
    generators, or as a composition of embedded single-error-term Lindblad gates
    (see param `errcomp_type`).

    The local noise consists terms up to a maximum weight acting on the qubits
    given reachable by a given maximum number of hops (along the neareset-
    neighbor edges of `qubitGraph`) from the target qubits.


    Parameters
    ----------
    targetOp : numpy array TODO: docstring (remove this)
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

    errcomp_type : {"onebig","manylittle"}
        Whether the `loc_noise` portion of the constructed gate should be a
        a single Lindblad gate containing all the allowed error terms (onebig)
        or the composition of many Lindblad gates each containing just a single
        error term (manylittle).  The resulting gate performs the same action
        regardless of the value set here; this just affects how the gate is
        structured internally.

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
    StaticDenseOp = _get_Static_factory(sim_type, parameterization) # always a *gate*
    Lindblad = _get_Lindblad_factory(sim_type, parameterization, errcomp_type)
      #constructs a gate or errorgen based on value of errcomp_type
    
    printer = _VerbosityPrinter.build_printer(verbosity)
    printer.log("Creating local-noise error factor (%s)" % errcomp_type)

    # make a composed-gate of embedded single-basis-element Lindblad-gates or -errorgens,
    #  one for each specified error term  
        
    loc_noise_termops = [] #list of gates to compose
    ssAllQ = [tuple(['Q%d'%i for i in range(qubitGraph.nqubits)])]
    #basisAllQ = _Basis('pp', 2**qubitGraph.nqubits, sparse=sparse) # TODO: remove - all we need is its 'dim' below
    basisAllQ_dim = _Dim(2**qubitGraph.nqubits)
    
    for wt, maxHops in weight_maxhops_tuples:
            
        ## loc_noise_errinds = [] # list of basis indices for all local-error terms 
        possible_err_qubit_inds = _np.array(qubitGraph.radius(target_qubit_inds, maxHops),_np.int64) # we know node labels are integers
        nPossible = len(possible_err_qubit_inds) # also == "nLocal" in this case
        basisEl_Id = basisProductMatrix(_np.zeros(wt,_np.int64),sparse) #identity basis el

        if errcomp_type == "gates":
            wtNoErr = _sps.identity(4**wt,'d','csr') if sparse else  _np.identity(4**wt,'d')
        elif errcomp_type == "errorgens":
            wtNoErr = _sps.csr_matrix((4**wt,4**wt)) if sparse else  _np.zeros((4**wt,4**wt),'d')
        else: raise ValueError("Invalid `errcomp_type`: %s" % errcomp_type)
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
            termErr = Lindblad(wtNoErr, proj_basis=errbasis, mxBasis=wtBasis, relative=True)
    
            fullTermErr = Embedded(ssAllQ, ['Q%d'%i for i in err_qubit_global_inds],
                                   termErr, basisAllQ_dim)
            assert(fullTermErr.num_params() == termErr.num_params())
            printer.log("Lindblad gate w/dim=%d and %d params -> embedded to gate w/dim=%d" %
                        (termErr.dim, termErr.num_params(), fullTermErr.dim))
            
            loc_noise_termops.append( fullTermErr )
          
    fullCloudErr = Composed(loc_noise_termops)
    return fullCloudErr


class CloudNoiseLayerLizard(_mdl.ImplicitLayerLizard):
    def get_prep(self,layerlbl):
        return self.prep_blks[layerlbl] # prep_blks are full prep ops
    def get_effect(self,layerlbl):
        return self.effect_blks[layerlbl] # effect_blks are full effect ops
    def get_operation(self,layerlbl):
        dense = bool(self.model._sim_type == "matrix") # whether dense matrix gates should be created
        add_idle_noise = self.model._lizardArgs['add_idle_noise']
        errcomp_type = self.model._lizardArgs['errcomp_type']
        sparse_expm = self.model._lizardArgs['sparse_expm']
        
        Composed = _op.ComposedDenseOp if dense else _op.ComposedOp
        Lindblad = _op.LindbladDenseOp if dense else _op.LindbladOp
        Sum = _op.ComposedErrorgen
        #print("DB: CloudNoiseLayerLizard building gate %s for %s w/comp-type %s" %
        #      (('matrix' if dense else 'map'), str(oplabel), self.errcomp_type) )

        #TODO REMOVE
        #if self.model.auto_idle_gatename is not None:
        #    component_labels = []
        #    for l in layerlabel.components:
        #        if l.name == self.model.auto_idle_gatename \
        #                and l not in existing_ops:
        #            continue #skip perfect idle placeholders
        #        components_labels.append(l)
        #else:
        
        components = layerlbl.components
        if len(components) == 0 or layerlbl == 'Gi': # idle!
            return self.op_blks['globalIdle']

        #Compose target operation from layer's component labels, which correspond
        # to the perfect (embedded) target ops in op_blks
        if len(components) > 1:
            targetOp = Composed([self.op_blks[l] for l in components], dim=self.model.dim,
                                evotype=self.model._evotype)
        else: targetOp = self.op_blks[components[0]]
        ops_to_compose = [targetOp]

        if errcomp_type == "gates":
            if add_idle_noise: ops_to_compose.append( self.op_blks['globalIdle'] )
            if len(components) > 1:
                localErr = Composed([self.op_blks[_Lbl('CloudNoise_'+l.name,l.sslbls)] for l in components],
                                    dim=self.model.dim,  evotype=self.model._evotype)
            else:
                l = components[0]
                localErr = self.op_blks[_Lbl('CloudNoise_'+l.name,l.sslbls)]

            ops_to_compose.append(localErr)

        elif errcomp_type == "errorgens":
            #We compose the target operations to create a
            # final target op, and compose this with a *singe* Lindblad gate which has as
            # its error generator the composition (sum) of all the factors' error gens.
            errorGens = [ self.op_blks['globalIdle'].errorgen ] if add_idle_noise else []
            errorGens.extend( [self.op_blks[_Lbl('CloudNoise_'+l.name,l.sslbls)]
                               for l in components] )
            if len(errorGens) > 1:
                error = Lindblad(None, Sum(errorGens, dim=self.model.dim,
                                           evotype=self.model._evotype),
                                 sparse_expm=sparse_expm)
            else:
                error = Lindblad(None, errorGens[0], sparse_expm=sparse_expm)
            ops_to_compose.append(error)
        else:
            raise ValueError("Invalid errcomp_type in CloudNoiseLayerLizard: %s" % errcomp_type)

        ret = Composed(ops_to_compose, dim=self.model.dim,
                           evotype=self.model._evotype)
        self.model._init_virtual_obj(ret) # so ret's gpindices get set
        return ret
