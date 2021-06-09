import collections as _collections
import itertools as _itertools
import numpy as _np
import scipy as _scipy
import scipy.sparse as _sps

import pygsti
from pygsti.modelpacks.legacy import std1Q_XY
from pygsti.modelpacks.legacy import std2Q_XYICNOT
import pygsti.objects as _objs

class QubitGraph(object):
    """ Graph data structure """
    def __init__(self, nQubits=0, geometry="line"):
        self._graph = _collections.defaultdict(set)
        self.nQubits = nQubits
        if nQubits == 0: 
            return
        elif nQubits == 1: 
            self._graph[0] = set() # no neighbors
            return
        else: #at least 2 qubits
            if geometry in ("line","ring"):
                for i in range(nQubits-1):
                    self.add(i,i+1)
                if nQubits > 2 and geometry == "ring":
                    self.add(nQubits-1,0)
            elif geometry in ("grid","torus"):
                s = int(round(_np.sqrt(nQubits)))
                assert(nQubits >= 4 and s*s == nQubits), \
                    "`nQubits` must be a perfect square >= 4"
                #row links
                for irow in range(s):
                    for icol in range(s):
                        if icol+1 < s:
                            self.add(irow*s+icol, irow*s+icol+1) #link right
                        elif geometry == "torus" and s > 2:
                            self.add(irow*s+icol, irow*s+0)
                            
                        if irow+1 < s:
                            self.add(irow*s+icol, (irow+1)*s+icol) #link down
                        elif geometry == "torus" and s > 2:
                            self.add(irow*s+icol, 0+icol)
            else:
                raise ValueError("Invalid `geometry`: %s" % geometry)
                
    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to graph """
        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """
        self._graph[node1].add(node2)
        self._graph[node2].add(node1)
        
    def edges(self):
        ret = set()
        for node,neighbors in self._graph.items():
            for neighbor in neighbors:
                if node < neighbor: # all edge tuples have lower index first
                    ret.add( (node,neighbor) )
                else:
                    ret.add( (neighbor,node) )
        return sorted(list(ret))
    
    def radius(self, base_indices, max_hops):
        """ 
        Returns a (sorted) array of indices that can be reached
        from traversing at most `max_hops` edges starting
        from a vertex in base_indices
        """
        ret = set()
        assert(max_hops >= 0)
        
        def traverse(start, hops_left):
            ret.add(start)
            if hops_left <= 0: return
            for i in self._graph[start]:
                traverse(i,hops_left-1)
                
        for node in base_indices:
            traverse(node,max_hops)
        return _np.array(sorted(list(ret)),'i')

    def connected_combos(self, possible_indices, size):
        count = 0
        for selected_inds in _itertools.combinations(possible_indices, size):
            if self.are_connected(selected_inds): count += 1
        return count

#     def remove(self, node):
#         """ Remove all references to node """
#         for n, cxns in self._graph.iteritems():
#             try:
#                 cxns.remove(node)
#             except KeyError:
#                 pass
#         try:
#             del self._graph[node]
#         except KeyError:
#             pass

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 """
        return node1 in self._graph and node2 in self._graph[node1]

    def are_connected(self, indices):
        """
        Are all the nodes in `indices` connected to at least
        one other node in `indices`?
        """
        if len(indices) < 2: return True # 0 or 1 indices are "connected"        

        for node in indices: #check
            if node not in self._graph: return False

        glob = set()
        def add_to_glob(node):
            glob.add(node)
            for neighbor in self._graph[node].intersection(indices):
                if neighbor not in glob:
                    add_to_glob(neighbor)
        
        add_to_glob(indices[0])
        return bool(glob == set(indices))

#     def find_path(self, node1, node2, path=[]):
#         """ Find any path between node1 and node2 (may not be shortest) """
#         path = path + [node1]
#         if node1 == node2:
#             return path
#         if node1 not in self._graph:
#             return None
#         for node in self._graph[node1]:
#             if node not in path:
#                 new_path = self.find_path(node, node2, path)
#                 if new_path:
#                     return new_path
#         return None

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
    
    
## Pauli basis matrices                                                                                                                                            
sqrt2 = _np.sqrt(2)
id2x2 = _np.array([[1,0],[0,1]])
sigmax = _np.array([[0,1],[1,0]])
sigmay = _np.array([[0,-1.0j],[1.0j,0]])
sigmaz = _np.array([[1,0],[0,-1]])

sigmaVec = (id2x2/sqrt2, sigmax/sqrt2, sigmay/sqrt2, sigmaz/sqrt2)


def iter_basis_inds(weight):
    basisIndList = [ [1,2,3] ]*weight #assume pauli 1Q basis, and only iterate over non-identity els
    for basisInds in _itertools.product(*basisIndList):
        yield basisInds

def basisProductMatrix(sigmaInds, sparse):
    M = _np.identity(1,'complex')
    for i in sigmaInds:
        M = _np.kron(M,sigmaVec[i])
    return _sps.csr_matrix(M) if sparse else M

def nparams_nqubit_gateset(nQubits, geometry="line", maxIdleWeight=1, maxhops=0,
                           extraWeight1Hops=0, extraGateWeight=0, requireConnected=False,
                           independent1Qgates=True, ZZonly=False, verbosity=0):
    # noise can be either a seed or a random array that is long enough to use

    printer = pygsti.obj.VerbosityPrinter.create_printer(verbosity)
    printer.log("Computing parameters for a %d-qubit %s model" % (nQubits,geometry))

    qubitGraph = QubitGraph(nQubits, geometry)
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

    def op_count_nparams(target_qubit_inds,weight_maxhops_tuples,debug=False):
        ret = 0
        #Note: no contrib from idle noise (already parameterized)
        for wt, maxHops in weight_maxhops_tuples:
            possible_err_qubit_inds = qubitGraph.radius(target_qubit_inds, maxHops)
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
    nParams['Gi'] = idle_count_nparams(maxIdleWeight)
     
    #1Q gates: X(pi/2) & Y(pi/2) on each qubit
    weight_maxhops_tuples_1Q = [(1,maxhops+extraWeight1Hops)] + \
                               [ (1+x,maxhops) for x in range(1,extraGateWeight+1) ]

    if independent1Qgates:
        for i in range(nQubits):
            printer.log("Creating 1Q X(pi/2) and Y(pi/2) gates on qubit %d!!" % i)
            nParams["Gx%d"%i] = op_count_nparams((i,), weight_maxhops_tuples_1Q)
            nParams["Gy%d"%i] = op_count_nparams((i,), weight_maxhops_tuples_1Q)
    else:
        printer.log("Creating common 1Q X(pi/2) and Y(pi/2) gates")
        rep = int(nQubits / 2)
        nParams["Gxrep"] = op_count_nparams((rep,), weight_maxhops_tuples_1Q)
        nParams["Gyrep"] = op_count_nparams((rep,), weight_maxhops_tuples_1Q)

    #2Q gates: CNOT gates along each graph edge
    weight_maxhops_tuples_2Q = [(1,maxhops+extraWeight1Hops),(2,maxhops)] + \
                               [ (2+x,maxhops) for x in range(1,extraGateWeight+1) ]
    for i,j in qubitGraph.edges(): #note: all edges have i<j so "control" of CNOT is always lower index (arbitrary)
        printer.log("Creating CNOT gate between qubits %d and %d!!" % (i,j))
        nParams["Gc%dt%d"% (i,j)] = op_count_nparams((i,j), weight_maxhops_tuples_2Q)

    #SPAM
    nPOVM_1Q = 4 # params for a single 1Q POVM
    nParams['rho0'] = 3*nQubits # 3 b/c each component is TP
    nParams['Mdefault'] = nPOVM_1Q * nQubits # nQubits 1Q-POVMs

    return nParams, sum(nParams.values())



def create_nqubit_gateset(nQubits, geometry="line", maxIdleWeight=1, maxhops=0,
                          extraWeight1Hops=0, extraGateWeight=0, sparse=False,
                          gateNoise=None, prepNoise=None, povmNoise=None, verbosity=0):
    # noise can be either a seed or a random array that is long enough to use

    printer = pygsti.obj.VerbosityPrinter.create_printer(verbosity)
    printer.log("Creating a %d-qubit %s model" % (nQubits,geometry))

    mdl = pygsti.obj.ExplicitOpModel() # no preps/POVMs
    # TODO: sparse prep & effect vecs... acton(...) analogue?

    #Full preps & povms -- maybe another option
    ##Create initial model with std prep & POVM
    #eLbls = []; eExprs = []
    #formatStr = '0' + str(nQubits) + 'b'
    #for i in range(2**nQubits):
    #    eLbls.append( format(i,formatStr))
    #    eExprs.append( str(i) )    
    #Qlbls = tuple( ['Q%d' % i for i in range(nQubits)] )
    #mdl = pygsti.construction.create_explicit_model(
    #    [2**nQubits], [Qlbls], [], [], 
    #    effect_labels=eLbls, effect_expressions=eExprs)
    printer.log("Created initial model")

    qubitGraph = QubitGraph(nQubits, geometry)
    printer.log("Created qubit graph:\n"+str(qubitGraph))

    printer.log("Creating Idle:")
    mdl.operations['Gi'] = create_global_idle(qubitGraph, maxIdleWeight, sparse, printer-1)
     
    #1Q gates: X(pi/2) & Y(pi/2) on each qubit
    Gx = std1Q_XY.target_model().operations['Gx']
    Gy = std1Q_XY.target_model().operations['Gy'] 
    weight_maxhops_tuples_1Q = [(1,maxhops+extraWeight1Hops)] + \
                               [ (1+x,maxhops) for x in range(1,extraGateWeight+1) ]
    for i in range(nQubits):
        printer.log("Creating 1Q X(pi/2) gate on qubit %d!!" % i)
        mdl.operations["Gx%d"%i] = create_composed_gate(
            Gx, (i,), qubitGraph, weight_maxhops_tuples_1Q,
            idle_noise=mdl.operations['Gi'], loc_noise_type="manylittle",
            sparse=sparse, verbosity=printer-1)

        printer.log("Creating 1Q Y(pi/2) gate on qubit %d!!" % i)
        mdl.operations["Gy%d"%i] = create_composed_gate(
            Gy, (i,), qubitGraph, weight_maxhops_tuples_1Q,
            idle_noise=mdl.operations['Gi'], loc_noise_type="manylittle",
            sparse=sparse, verbosity=printer-1)
        
    #2Q gates: CNOT gates along each graph edge
    Gcnot = std2Q_XYICNOT.target_model().operations['Gcnot']
    weight_maxhops_tuples_2Q = [(1,maxhops+extraWeight1Hops),(2,maxhops)] + \
                               [ (2+x,maxhops) for x in range(1,extraGateWeight+1) ]
    for i,j in qubitGraph.edges(): #note: all edges have i<j so "control" of CNOT is always lower index (arbitrary)
        printer.log("Creating CNOT gate between qubits %d and %d!!" % (i,j))
        mdl.operations["Gc%dt%d"% (i,j)] = create_composed_gate(
            Gcnot, (i,j), qubitGraph, weight_maxhops_tuples_2Q,
            idle_noise=mdl.operations['Gi'], loc_noise_type="manylittle",
            sparse=sparse, verbosity=printer-1)


    #Insert noise on gates
    vecNoSpam = mdl.to_vector()
    assert( _np.linalg.norm(vecNoSpam)/len(vecNoSpam) < 1e-6 )
    if gateNoise is not None:
        if isinstance(gateNoise,tuple): # use as (seed, strength)
            seed,strength = gateNoise
            rndm = _np.random.RandomState(seed)
            vecNoSpam += _np.abs(rndm.random_sample(len(vecNoSpam))*strength) #abs b/c some params need to be positive
        else: #use as a vector
            vecNoSpam += gateNoise[0:len(vecNoSpam)]
        mdl.from_vector(vecNoSpam)

        
    #SPAM
    basis1Q = pygsti.obj.Basis("pp",2)
    prepFactors = [ pygsti.obj.TPSPAMVec(pygsti.construction._basis_create_spam_vector("0", basis1Q))
                    for i in range(nQubits)]
    if prepNoise is not None:
        if isinstance(prepNoise,tuple): # use as (seed, strength)
            seed,strength = prepNoise
            rndm = _np.random.RandomState(seed)
            depolAmts = _np.abs(rndm.random_sample(nQubits)*strength)
        else:
            depolAmts = prepNoise[0:nQubits]
        for amt,vec in zip(depolAmts,prepFactors): vec.depolarize(amt) 
    mdl.preps['rho0'] = pygsti.obj.TensorProdSPAMVec('prep',prepFactors)
    
    factorPOVMs = []
    for i in range(nQubits):
        effects = [ (l,pygsti.construction._basis_create_spam_vector(l, basis1Q)) for l in ["0","1"] ]
        factorPOVMs.append( pygsti.obj.TPPOVM(effects) )
    if povmNoise is not None:
        if isinstance(povmNoise,tuple): # use as (seed, strength)
            seed,strength = povmNoise
            rndm = _np.random.RandomState(seed)
            depolAmts = _np.abs(rndm.random_sample(nQubits)*strength)
        else:
            depolAmts = povmNoise[0:nQubits]
        for amt,povm in zip(depolAmts,factorPOVMs): povm.depolarize(amt) 
    mdl.povms['Mdefault'] = pygsti.obj.TensorProdPOVM( factorPOVMs )
        
    printer.log("DONE! - returning Model with dim=%d and gates=%s" % (mdl.dim, list(mdl.operations.keys())))
    return mdl
    


def create_global_idle(qubitGraph, maxWeight, sparse=False, verbosity=0):
    assert(maxWeight <= 2), "Only `maxWeight` equal to 0, 1, or 2 is supported"

    if sparse:
        Lindblad = _objs.LindbladOp
        Composed = _objs.ComposedOp
        Embedded = _objs.EmbeddedOp
    else:
        Lindblad = _objs.LindbladDenseOp
        Composed = _objs.ComposedDenseOp
        Embedded = _objs.EmbeddedDenseOp
    
    printer = pygsti.obj.VerbosityPrinter.create_printer(verbosity)
    printer.log("*** Creating global idle ***")
    
    termgates = [] # gates to compose
    ssAllQ = [tuple(['Q%d'%i for i in range(qubitGraph.nQubits)])]
    basisAllQ = pygsti.objects.Basis('pp', 2**qubitGraph.nQubits, sparse=sparse)
    
    nQubits = qubitGraph.nQubits
    possible_err_qubit_inds = _np.arange(nQubits)
    nPossible = nQubits  
    for wt in range(1,maxWeight+1):
        printer.log("Weight %d: %d possible qubits" % (wt,nPossible),2)
        basisEl_Id = basisProductMatrix(_np.zeros(wt,'i'),sparse)
        wtId = _sps.identity(4**wt,'d','csr') if sparse else  _np.identity(4**wt,'d')
        wtBasis = pygsti.objects.Basis('pp', 2**wt, sparse=sparse)
        
        for err_qubit_inds in _itertools.combinations(possible_err_qubit_inds, wt):
            if len(err_qubit_inds) == 2 and not qubitGraph.is_connected(err_qubit_inds[0],err_qubit_inds[1]):
                continue # TO UPDATE - check whether all wt indices are a connected subgraph

            errbasis = [basisEl_Id]
            for err_basis_inds in iter_basis_inds(wt):        
                error = _np.array(err_basis_inds,'i') #length == wt
                basisEl = basisProductMatrix(error,sparse)
                errbasis.append(basisEl)

            printer.log("Error on qubits %s -> error basis of length %d" % (err_qubit_inds,len(errbasis)), 3)
            errbasis = pygsti.obj.Basis(matrices=errbasis, sparse=sparse) #single element basis (plus identity)
            termErr = Lindblad(wtId, ham_basis=errbasis, nonham_basis=errbasis, cptp=True,
                               nonham_diagonal_only=True, truncate=True, mx_basis=wtBasis)
        
            err_qubit_global_inds = err_qubit_inds
            fullTermErr = Embedded(ssAllQ, [('Q%d'%i) for i in err_qubit_global_inds],
                                   termErr, basisAllQ.dim)
            assert(fullTermErr.num_params() == termErr.num_params())
            printer.log("Lindblad gate w/dim=%d and %d params -> embedded to gate w/dim=%d" %
                        (termErr.dim, termErr.num_params(), fullTermErr.dim))
                    
            termgates.append( fullTermErr )
                
    return Composed(termgates)         
    
    

#def create_noncomposed_gate(target_op, target_qubit_inds, qubitGraph, max_weight, maxHops,
#                            spectatorMaxWeight=1, mode="embed"):
#
#    assert(spectatorMaxWeight <= 1) #only 0 and 1 are currently supported
#    
#    errinds = [] # list of basis indices for all error terms
#    possible_err_qubit_inds = qubitGraph.radius(target_qubit_inds, maxHops)
#    nPossible = len(possible_err_qubit_inds)
#    for wt in range(max_weight+1):
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
#    ssAllQ = ['Q%d'%i for i in range(qubitGraph.nQubits)]
#    basisAllQ = pygsti.objects.Basis('pp', 2**qubitGraph.nQubits)
#    
#    if mode == "no-embedding":     
#        fullTargetOp = EmbeddedDenseOp(ssAllQ, ['Q%d'%i for i in target_qubit_inds],
#                                    target_op, basisAllQ) 
#        fullTargetOp = StaticArbitraryOp( fullTargetOp ) #Make static
#        fullLocalErr = LindbladDenseOp(fullTargetOp, fullTargetOp,
#                         ham_basis=errbasis, nonham_basis=errbasis, cptp=True,
#                         nonham_diagonal_only=True, truncate=True, mx_basis=basisAllQ)
#          # gate on full qubit space that accounts for error on the "local qubits", that is,
#          # those local to the qubits being operated on
#    elif mode == "embed":
#        possible_list = list(possible_err_qubit_inds)
#        loc_target_inds = [possible_list.index(i) for i in target_qubit_inds]
#        
#        ssLocQ = ['Q%d'%i for i in range(nPossible)]
#        basisLocQ = pygsti.objects.Basis('pp', 2**nPossible)
#        locTargetOp = StaticArbitraryOp( EmbeddedDenseOp(ssLocQ, ['Q%d'%i for i in loc_target_inds],
#                                    target_op, basisLocQ) )
#        localErr = LindbladDenseOp(locTargetOp, locTargetOp,
#                         ham_basis=errbasis, nonham_basis=errbasis, cptp=True,
#                         nonham_diagonal_only=True, truncate=True, mx_basis=basisLocQ)
#        fullLocalErr = EmbeddedDenseOp(ssAllQ, ['Q%d'%i for i in possible_err_qubit_inds],
#                                   localErr, basisAllQ)
#    else:
#        raise ValueError("Invalid Mode: %s" % mode)
#        
#    #Now add errors on "non-local" i.e. spectator gates
#    if spectatorMaxWeight == 0:
#        pass
#    #STILL in progress -- maybe just non-embedding case, since if we embed we'll
#    # need to compose (in general)
        

        
def create_composed_gate(targetOp, target_qubit_inds, qubitGraph, weight_maxhops_tuples,
                         idle_noise=False, loc_noise_type="onebig",
                         apply_idle_noise_to="all", sparse=False, verbosity=0):
    """ 
    Final gate is a composition of: 
    targetOp(target qubits) -> idle_noise(all_qubits) -> loc_noise(local_qubits)
    
    where `idle_noise` is given by the `idle_noise` parameter and loc_noise is given
    by the other params.  loc_noise can be implemented either by 
    a single embedded LindbladDenseOp with all relevant error generators,
    or as a composition of embedded-single-error-term gates (see param `loc_noise_type`)
    
    Parameters
    ----------
    
    idle_noise : LinearOperator or boolean
        either given as an existing gate (on all qubits) or a boolean indicating
        whether a composition of weight-1 noise terms (separately on all the qubits),
        is created.  If `apply_idle_noise_to == "nonlocal"` then `idle_noise` is *only*
        applied to the non-local qubits and `idle_noise` must be a ComposedDenseOp or
        ComposedMap with nQubits terms so that individual terms for each qubit can
        be extracted as needed.

    TODO   
    """
    if sparse:
        Lindblad = _objs.LindbladOp
        Composed = _objs.ComposedOp
        Embedded = _objs.EmbeddedOp
        Static = _objs.StaticDenseOp # TODO: create StaticGateMap
    else:
        Lindblad = _objs.LindbladDenseOp
        Composed = _objs.ComposedDenseOp
        Embedded = _objs.EmbeddedDenseOp
        Static = _objs.StaticDenseOp
    
    printer = pygsti.obj.VerbosityPrinter.create_printer(verbosity)
    printer.log("*** Creating composed gate ***")
    
    #Factor1: target operation
    printer.log("Creating %d-qubit target op factor on qubits %s" %
                (len(target_qubit_inds),str(target_qubit_inds)),2)
    ssAllQ = [tuple(['Q%d'%i for i in range(qubitGraph.nQubits)])]
    basisAllQ = pygsti.objects.Basis('pp', 2**qubitGraph.nQubits, sparse=sparse)
    fullTargetOp = Embedded(ssAllQ, ['Q%d'%i for i in target_qubit_inds],
                            Static(targetOp), basisAllQ.dim) 

    #Factor2: idle_noise operation
    printer.log("Creating idle error factor",2)
    if apply_idle_noise_to == "all":
        if isinstance(idle_noise, pygsti.obj.LinearOperator):
            printer.log("Using supplied full idle gate",3)
            fullIdleErr = idle_noise
        elif idle_noise == True:
            #build composition of 1Q idle ops
            printer.log("Constructing independend weight-1 idle gate",3)
            # Id_1Q = _sps.identity(4**1,'d','csr') if sparse else  _np.identity(4**1,'d')
            Id_1Q = _np.identity(4**1,'d') #always dense for now...
            fullIdleErr = Composed(
                [ Embedded(ssAllQ, ('Q%d'%i,), Lindblad(Id_1Q.copy()),basisAllQ.dim)
                  for i in range(qubitGraph.nQubits)] )
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
        all_possible_err_qubit_inds = qubitGraph.radius(
            target_qubit_inds, max([hops for _,hops in weight_maxhops_tuples]) )
        nLocal = len(all_possible_err_qubit_inds)
        basisEl_Id = basisProductMatrix(_np.zeros(nPossible,'i'),sparse) #identity basis el
        
        for wt, maxHops in weight_maxhops_tuples:
            possible_err_qubit_inds = qubitGraph.radius(target_qubit_inds, maxHops)
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
        errbasis = pygsti.obj.Basis(matrices=errbasis, sparse=sparse) #single element basis (plus identity)
        
        #Construct one embedded Lindblad-gate using all `errbasis` terms
        ssLocQ = [tuple(['Q%d'%i for i in range(nLocal)])]
        basisLocQ = pygsti.objects.Basis('pp', 2**nLocal, sparse=sparse)
        locId = _sps.identity(4**nLocal,'d','csr') if sparse else _np.identity(4**nLocal,'d')
        localErr = Lindblad(locId, ham_basis=errbasis,
                            nonham_basis=errbasis, cptp=True,
                            nonham_diagonal_only=True, truncate=True,
                            mx_basis=basisLocQ)
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
            possible_err_qubit_inds = qubitGraph.radius(target_qubit_inds, maxHops)
            nPossible = len(possible_err_qubit_inds) # also == "nLocal" in this case
            basisEl_Id = basisProductMatrix(_np.zeros(wt,'i'),sparse) #identity basis el

            wtId = _sps.identity(4**wt,'d','csr') if sparse else _np.identity(4**wt,'d')
            wtBasis = pygsti.objects.Basis('pp', 2**wt, sparse=sparse)

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
                errbasis = pygsti.obj.Basis(matrices=errbasis, sparse=sparse) #single element basis (plus identity)
                termErr = Lindblad(wtId, ham_basis=errbasis,
                                   nonham_basis=errbasis, cptp=True,
                                   nonham_diagonal_only=True, truncate=True,
                                   mx_basis=wtBasis)
        
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

