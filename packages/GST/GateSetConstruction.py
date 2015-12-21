"""Functions for the construction of new gate sets."""
import numpy as _np
import itertools as _itertools

import Gate as _Gate
import gateset as _gateset
import BasisTools as _BT

import scipy.linalg as _spl


#############################################
# Build gates based on "standard" gate names
############################################

def buildVector(stateSpaceDims, stateSpaceLabels, vecExpr, basis="gm"):
    """
    Build a rho or E vector from an expression.
    
    Parameters
    ----------
    stateSpaceDims : list of ints
        Dimenstions specifying the structure of the density-matrix space.  Elements
        correspond to block dimensions of an allowed density matrix in the standard
        basis, and the density-matrix space is the direct sum of linear spaces of 
        dimension block-dimension^2.

    stateSpaceLabels : a list of tuples
        Each tuple corresponds to a block of a density matrix in the standard basis
        (and therefore a component of the direct-sum density matrix space). Elements
        of a tuple are user-defined labels beginning with "L" (single level) or "Q"
        (two-level; qubit) which interpret the d-dimensional state space corresponding
        to a d x d block as a tensor product between qubit and single level systems.

    vecExpr : string
        the expression which determines which vector to build.  Currenlty, only
        integers are allowed, which specify a the vector for the pure state of
        that index.  For example, "1" means return vectorize(``|1><1|``).  The index
        labels the absolute index of the state within the entire state space, and
        is independent of the direct-sum decomposition of density matrix space.

    basis : {'gm','pp','std'}, optional
        the basis of the returned vector. 
        'std' == Standard (matrix units)
        'gm' == Gell-Mann 
        'pp' == Pauli-product

    Returns
    -------
    numpy array
        The vector specified by vecExpr in the desired basis.
    """
    dmDim, gateDim, blockDims = _BT._processBlockDims(stateSpaceDims)
    vecInReducedStdBasis = _np.zeros( (gateDim,1), 'd' ) # assume index given as vecExpr refers to a 
                                                         #Hilbert-space state index, so "reduced-std" basis
    
    #So far just allow integer rhoExpressions that give the index of state (within the state space) that we prep/measure
    try:
        index = int(vecExpr)
    except:
        raise ValueError("Expression must be the index of a state (as a string)")
    
    start = 0; vecIndex = 0
    for blockDim in blockDims:
        for i in range(start,start+blockDim):
            for j in range(start,start+blockDim):
                if (i,j) == (index,index):
                    vecInReducedStdBasis[ vecIndex, 0 ] = 1.0  #set diagonal element of density matrix
                    break
                vecIndex += 1
        start += blockDim
    
    if basis == "std": return vecInReducedStdBasis
    elif basis == "gm": return _BT.basisChg_StdToGellMann(vecInReducedStdBasis, stateSpaceDims)
    elif basis == "pp": return _BT.basisChg_StdToPauliProd(vecInReducedStdBasis, stateSpaceDims)
    else: raise ValueError("Invalid basis argument: %s" % basis)

def buildIdentityVector(stateSpaceDims, basis="gm"):
    """
    Build a the identity vector for a given space and basis.
    
    Parameters
    ----------
    stateSpaceDims : list of ints
        Dimenstions specifying the structure of the density-matrix space.  Elements
        correspond to block dimensions of an allowed density matrix in the standard
        basis, and the density-matrix space is the direct sum of linear spaces of 
        dimension block-dimension^2.

    basis : {'gm','pp','std'}, optional
        the basis of the returned vector. 
        'std' == Standard (matrix units)
        'gm' == Gell-Mann 
        'pp' == Pauli-product

    Returns
    -------
    numpy array
        The identity vector in the desired basis.
    """
    dmDim, gateDim, blockDims = _BT._processBlockDims(stateSpaceDims)
    vecInReducedStdBasis = _np.zeros( (gateDim,1), 'd' ) # assume index given as vecExpr refers to a Hilbert-space state index, so "reduced-std" basis

    #set all diagonal elements of density matrix to 1.0 (end result = identity density mx)
    start = 0; vecIndex = 0
    for blockDim in blockDims:
        for i in range(start,start+blockDim):
            for j in range(start,start+blockDim):
                if i == j: vecInReducedStdBasis[ vecIndex, 0 ] = 1.0  #set diagonal element of density matrix
                vecIndex += 1
        start += blockDim
    
    if basis == "std": return vecInReducedStdBasis
    elif basis == "gm": return _BT.basisChg_StdToGellMann(vecInReducedStdBasis, stateSpaceDims)
    elif basis == "pp": return _BT.basisChg_StdToPauliProd(vecInReducedStdBasis, stateSpaceDims)
    else: raise ValueError("Invalid basis argument: %s" % basis)
    


def _oldBuildGate(stateSpaceDims, stateSpaceLabels, gateExpr, basis="gm"):
#coherentStateSpaceBlockDims
    """
    Build a gate matrix from an expression

    Parameters
    ----------
    stateSpaceDims : a list of integers specifying the dimension of each block of a block-diagonal the density matrix
    stateSpaceLabels : a list of tuples, each one corresponding to a block of the density matrix.  Elements of the tuple are
                     user-defined labels beginning with "L" (single level) or "Q" (two-level; qubit) which interpret the
                     states within the block as a tensor product structure between the labelled constituent systems.

    gateExpr : string containing an expression for the gate to build

    basis : string
      - "std" = gate matrix operates on density mx expressed as sum of matrix units
      - "gm"  = gate matrix operates on dentity mx expressed as sum of normalized Gell-Mann matrices
      - "pp"  = gate matrix operates on density mx expresses as sum of tensor-prod of pauli matrices
    """
    # gateExpr can contain single qubit ops: X(theta) ,Y(theta) ,Z(theta)
    #                      two qubit ops: CNOT
    #                      clevel qubit ops: Leak
    #                      two clevel opts: Flip
    #  each of which is given additional parameters specifying which indices it acts upon

    
    #Gate matrix will be in matrix unit basis, which we order by vectorizing
    # (by concatenating rows) each block of coherent states in the order given.
    dmDim, gateDim, blockDims = _BT._processBlockDims(stateSpaceDims)
    fullOpDim = dmDim**2

    #Store each tensor product blocks start index (within the density matrix), which tensor product block
    #  each label is in, and check to make sure dimensions match stateSpaceDims
    tensorBlkIndices = {}; startIndex = []; M = 0
    assert( len(stateSpaceDims) == len(stateSpaceLabels) )
    for k, blockDim in enumerate(stateSpaceDims):
        startIndex.append(M); M += blockDim        

        #Make sure tensor-product interpretation agrees with given dimension            
        tensorBlkDim = 1 #dimension of this coherent block of the *density matrix*
        for s in stateSpaceLabels[k]:
            tensorBlkIndices[s] = k
            if s.startswith('Q'): tensorBlkDim *= 2   
            elif s.startswith('L'): tensorBlkDim *= 1
            else: raise ValueError("Invalid state space specifier: %s" % s)
        if tensorBlkDim != blockDim:
            raise ValueError("State labels %s for tensor product block %d have dimension %d != given dimension %d" \
                                 % stateSpaceLabels[k], k, tensorBlkDim, blockDim)
            

    #print "DB: dim = ",dim, " dmDim = ",dmDim
    gateInStdBasis = _np.identity( fullOpDim, 'complex' )  
      # in full basis of matrix units, which we later reduce to the
      # that basis of matrix units corresponding to the allowed non-zero
      #  elements of the density matrix.

    exprTerms = gateExpr.split(':')
    for exprTerm in exprTerms:

        gateTermInStdBasis = _np.identity( fullOpDim, 'complex' )
        l = exprTerm.index('('); r = exprTerm.index(')')
        gateName = exprTerm[0:l]
        argsStr = exprTerm[l+1:r]
        args = argsStr.split(',')

        if gateName == "I":
            pass

        elif gateName in ('X','Y','Z'): #single-qubit gate names
            assert(len(args) == 2) # theta, qubit-index
            theta = eval( args[0], {"__builtins__":None}, {'pi': _np.pi})
            label = args[1].strip(); assert(label.startswith('Q'))

            if gateName == 'X': ex = -1j * theta*_BT.sigmax/2
            elif gateName == 'Y': ex = -1j * theta*_BT.sigmay/2
            elif gateName == 'Z': ex = -1j * theta*_BT.sigmaz/2
            Ugate = _spl.expm(ex) # 2x2 unitary matrix operating on single qubit in [0,1] basis

            iTensorProdBlk = tensorBlkIndices[label] # index of tensor product block (of state space) this bit label is part of
            cohBlk = stateSpaceLabels[iTensorProdBlk]
            basisInds = []
            for l in cohBlk:
                if l.startswith('L'): basisInds.append([0])
                elif l.startswith('Q'): basisInds.append([0,1])
                else: raise ValueError("Invalid state space label: %s" % l)

            tensorBlkBasis = list(_itertools.product(*basisInds))
            K = cohBlk.index(label)
            N = len(tensorBlkBasis)
            UcohBlk = _np.identity( N, 'complex' ) # unitary matrix operating on relevant tensor product block part of state
            for i,b1 in enumerate(tensorBlkBasis):
                for j,b2 in enumerate(tensorBlkBasis):
                    if (b1[:K]+b1[K+1:]) == (b2[:K]+b2[K+1:]):   #if all part of tensor prod match except for qubit we're operating on
                        UcohBlk[i,j] = Ugate[ b1[K], b2[K] ] # then fill in element

            UcohBlkc = UcohBlk.conjugate()
            gateBlk = _np.kron(UcohBlk,UcohBlkc) # N^2 x N^2 mx operating on vectorized tensor product block of densty matrix

            #Map gateBlk's basis into final gate basis
            mapBlk = []
            s = startIndex[iTensorProdBlk] #within state space (i.e. row or col of density matrix)
            cohBlkSize = UcohBlk.shape[0]
            for i in range(cohBlkSize):
                for j in range(cohBlkSize):
                    vec_ij_index = (s+i)*dmDim + (s+j) #vectorize by concatenating rows
                    mapBlk.append( vec_ij_index ) #build list of vector indices of each element of gateBlk mx
            for i,fi in enumerate(mapBlk):
                for j,fj in enumerate(mapBlk):
                    gateTermInStdBasis[fi,fj] = gateBlk[i,j]


        elif gateName in ('CX','CY','CZ'): #two-qubit gate names
            assert(len(args) == 3) # theta, qubit-label1, qubit-label2
            theta = eval( args[0], {"__builtins__":None}, {'pi': _np.pi})
            label1 = args[1]; assert(label1.startswith('Q'))
            label2 = args[2]; assert(label2.startswith('Q'))

            if gateName == 'CX': ex = -1j * theta*_BT.sigmax/2
            elif gateName == 'CY': ex = -1j * theta*_BT.sigmay/2
            elif gateName == 'CZ': ex = -1j * theta*_BT.sigmaz/2
            Utarget = _spl.expm(ex) # 2x2 unitary matrix operating on target qubit
            Ugate = _np.identity(4, 'complex'); Ugate[2:,2:] = Utarget #4x4 unitary matrix operating on isolated two-qubit space

            iTensorProdBlk = tensorBlkIndices[label1] # index of tensor product block (of state space) this bit label is part of
            assert( iTensorProdBlk == tensorBlkIndices[label2] ) #labels must be members of the same tensor product block
            cohBlk = stateSpaceLabels[iTensorProdBlk]
            basisInds = []
            for l in cohBlk:
                if l.startswith('L'): basisInds.append([0])
                elif l.startswith('Q'): basisInds.append([0,1])
                else: raise ValueError("Invalid state space label: %s" % l)

            tensorBlkBasis = list(_itertools.product(*basisInds))
            K1 = cohBlk.index(label1)
            K2 = cohBlk.index(label2)
            N = len(tensorBlkBasis)
            UcohBlk = _np.identity( N, 'complex' ) # unitary matrix operating on relevant tensor product block part of state
            for i,b1 in enumerate(tensorBlkBasis):
                for j,b2 in enumerate(tensorBlkBasis):
                    b1p = list(b1); del b1p[max(K1,K2)]; del b1p[min(K1,K2)] # b1' -- remove basis indices for tensor
                    b2p = list(b2); del b2p[max(K1,K2)]; del b2p[min(K1,K2)] # b2'      product parts we operate on
                    if b1p == b2p:   #if all parts of tensor product match except for qubits we're operating on
                        UcohBlk[i,j] = Ugate[ 2*b1[K1]+b1[K2], 2*b2[K1]+b2[K2] ] # then fill in element

            #print "UcohBlk = \n",UcohBlk

            UcohBlkc = UcohBlk.conjugate()
            gateBlk = _np.kron(UcohBlk,UcohBlkc) # N^2 x N^2 mx operating on vectorized tensor product block of densty matrix

            #Map gateBlk's basis into final gate basis
            mapBlk = []
            s = startIndex[iTensorProdBlk] #within state space (i.e. row or col of density matrix)
            cohBlkSize = UcohBlk.shape[0]
            for i in range(cohBlkSize):
                for j in range(cohBlkSize):
                    vec_ij_index = (s+i)*dmDim + (s+j) #vectorize by concatenating rows
                    mapBlk.append( vec_ij_index ) #build list of vector indices of each element of gateBlk mx
            for i,fi in enumerate(mapBlk):
                for j,fj in enumerate(mapBlk):
                    gateTermInStdBasis[fi,fj] = gateBlk[i,j]


        elif gateName == "LX":  #TODO - better way to describe leakage?
            assert(len(args) == 3) # theta, dmIndex1, dmIndex2 - X rotation between any two density matrix basis states
            theta = eval( args[0], {"__builtins__":None}, {'pi': _np.pi})
            i1 = int(args[1])
            i2 = int(args[2])
            ex = -1j * theta*_BT.sigmax/2
            Ugate = _spl.expm(ex) # 2x2 unitary matrix operating on the i1-th and i2-th states of the state space basis
            Utot = _np.identity(dmDim, 'complex')
            Utot[ i1,i1 ] = Ugate[0,0]
            Utot[ i1,i2 ] = Ugate[0,1]
            Utot[ i2,i1 ] = Ugate[1,0]
            Utot[ i2,i2 ] = Ugate[1,1]

            Utotc = Utot.conjugate()
            gateBlk = _np.kron(Utot,Utotc) # N^2 x N^2 mx operating on vectorized tensor product block of densty matrix

            #Map gateBlk's basis (vectorized 2x2) into final gate basis
            mapBlk = [] #note: "start index" is effectively zero since we're mapping all the blocs
            for i in range(dmDim):
                for j in range(dmDim):
                    vec_ij_index = (i)*dmDim + (j) #vectorize by concatenating rows
                    mapBlk.append( vec_ij_index ) #build list of vector indices of each element of gateBlk mx
            for i,fi in enumerate(mapBlk):
                for j,fj in enumerate(mapBlk):
                    gateTermInStdBasis[fi,fj] = gateBlk[i,j]

            #sq = startIndex[qbIndex]; sc = startIndex[clIndex]  #sq,sc are density matrix start indices
            #vsq = dmiToVi[ (sq,sq) ]; vsc = dmiToVi[ (sc,sc) ]  # vector indices of (sq,sq) and (sc,sc) density matrix elements
            #vsq1 = dmiToVi[ (sq,sq+1) ]; vsq2 = dmiToVi[ (sq+1,sq) ]  # vector indices of qubit coherences
            #
            ## action = swap (sq,sq) and (sc,sc) elements of a d.mx. and destroy coherences within qubit
            #gateTermInStdBasis[vsq,vsc] = gateTermInStdBasis[vsc,vsq] = 1.0 
            #gateTermInStdBasis[vsq,vsq] = gateTermInStdBasis[vsc,vsc] = 0.0
            #gateTermInStdBasis[vsq1,vsq1] = gateTermInStdBasis[vsq2,vsq2] = 0.0


#        elif gateName == "Flip":
#            assert(len(args) == 2) # clevel-index0, clevel-index1
#            indx0 = int(args[0])
#            indx1 = int(args[1])
#            assert(indx0 != indx1)
#            assert(bitLabels[indx0] == 'L' and bitLabels[indx1] == 'L')
#
#            s0 = startIndex[indx0]; s1 = startIndex[indx1] #density matrix indices
#            vs0 = dmiToVi[ (s0,s0) ]; vs1 = dmiToVi[ (s1,s1) ]  # vector indices of (s0,s0) and (s1,s1) density matrix elements
#
#            # action = swap (s0,s0) and (s1,s1) elements of a d.mx.
#            gateTermInStdBasis[vs0,vs1] = gateTermInStdBasis[vs1,vs0] = 1.0 
#            gateTermInStdBasis[vs0,vs0] = gateTermInStdBasis[vs1,vs1] = 0.0

        else: raise ValueError("Invalid gate name: %s" % gateName)
        
        gateInStdBasis = _np.dot(gateInStdBasis, gateTermInStdBasis)

    #Pare down gateInStdBasis to only include those matrix unit basis elements that are allowed to be nonzero
    gateInReducedStdBasis = _BT.contractToStdDirectSumMx(gateInStdBasis, stateSpaceDims)

    #Change from std (mx unit) basis to another if requested
    if basis == "std":
        return _Gate.FullyParameterizedGate(gateInReducedStdBasis)
    elif basis == "gm":
        return _Gate.FullyParameterizedGate( _BT.basisChg_StdToGellMann(gateInReducedStdBasis, stateSpaceDims) )
    elif basis == "pp":
        return _Gate.FullyParameterizedGate( _BT.basisChg_StdToPauliProd(gateInReducedStdBasis, stateSpaceDims) )
    else: 
        raise ValueError("Invalid 'basis' parameter: %s (must by 'std', 'gm', or 'pp')" % basis)





def buildGate(stateSpaceDims, stateSpaceLabels, gateExpr, basis="gm", parameterization="full", unitaryEmbedding=False):
#coherentStateSpaceBlockDims
    """
    Build a Gate object from an expression.

    Parameters
    ----------
    stateSpaceDims : list of ints
        Dimenstions specifying the structure of the density-matrix space.  Elements
        correspond to block dimensions of an allowed density matrix in the standard
        basis, and the density-matrix space is the direct sum of linear spaces of 
        dimension block-dimension^2.

    stateSpaceLabels : a list of tuples
        Each tuple corresponds to a block of a density matrix in the standard basis
        (and therefore a component of the direct-sum density matrix space). Elements
        of a tuple are user-defined labels beginning with "L" (single level) or "Q"
        (two-level; qubit) which interpret the d-dimensional state space corresponding
        to a d x d block as a tensor product between qubit and single level systems.
        
    gateExpr : string 
        expression for the gate to build.  String is first split into parts delimited
        by the colon (:) character, which are composed together to create the final gate.
        Each part takes on of the allowed forms:

        - I(ssl_0, ...) = identity operation on one or more state space labels (ssl_i)
        - X(theta, ssl) = x-rotation by theta radians of qubit labeled by ssl
        - Y(theta, ssl) = y-rotation by theta radians of qubit labeled by ssl
        - Z(theta, ssl) = z-rotation by theta radians of qubit labeled by ssl
        - CX(theta, ssl0, ssl1) = controlled x-rotation by theta radians.
               Acts on qubit labeled by ssl1 with ssl0 being the control.
        - CY(theta, ssl0, ssl1) = controlled y-rotation by theta radians.
               Acts on qubit labeled by ssl1 with ssl0 being the control.
        - CZ(theta, ssl0, ssl1) = controlled z-rotation by theta radians.
               Acts on qubit labeled by ssl1 with ssl0 being the control.
        - LX(theta, i0, i1) = leakage between states i0 and i1.
               Implemented as an x-rotation between states with integer indices
               i0 and i1 followed by complete decoherence between the states.

    basis : {'gm','pp','std'}, optional
        the basis of the returned gate. 

        - "std" = gate matrix operates on density mx expressed as sum of matrix units
        - "gm"  = gate matrix operates on dentity mx expressed as sum of normalized Gell-Mann matrices
        - "pp"  = gate matrix operates on density mx expresses as sum of tensor-product of Pauli matrices

    parameterization : {"full","linear"}, optional
        How to parameterize the resulting gate.

        - "full" = return a FullyParameterizedGate.
        - "linear" = if possible, return a LinearlyParameterizedGate that parameterizes
          ,          only the pieces explicitly present in gateExpr.

    unitaryEmbedding : bool, optional
        An interal switch determining how the gate is constructed.  Should have no
        bearing on the output except in determining how to parameterized a non-
        FullyParameterizedGate.  It's best to leave this to False unless you
        really know what you're doing.

    Returns
    -------
    Gate
        A gate object representing the gate given by gateExpr in the desired basis.
    """
    # gateExpr can contain single qubit ops: X(theta) ,Y(theta) ,Z(theta)
    #                      two qubit ops: CNOT
    #                      clevel qubit ops: Leak
    #                      two clevel opts: Flip
    #  each of which is given additional parameters specifying which indices it acts upon
    
    dmDim, gateDim, blockDims = _BT._processBlockDims(stateSpaceDims)
    fullOpDim = dmDim**2

    #Store each tensor product blocks start index (within the density matrix), which tensor product block
    #  each label is in, and check to make sure dimensions match stateSpaceDims
    tensorBlkIndices = {}; startIndex = []; M = 0
    assert( len(blockDims) == len(stateSpaceLabels) )
    for k, blockDim in enumerate(blockDims):
        startIndex.append(M); M += blockDim        

        #Make sure tensor-product interpretation agrees with given dimension            
        tensorBlkDim = 1 #dimension of this coherent block of the *density matrix*
        for s in stateSpaceLabels[k]:
            tensorBlkIndices[s] = k
            if s.startswith('Q'): tensorBlkDim *= 2   
            elif s.startswith('L'): tensorBlkDim *= 1
            else: raise ValueError("Invalid state space specifier: %s" % s)
        if tensorBlkDim != blockDim:
            raise ValueError("State labels %s for tensor product block %d have dimension %d != given dimension %d" \
                                 % stateSpaceLabels[k], k, tensorBlkDim, blockDim)
            

    # ----------------------------------------------------------------------------------------------------------------------------------------
    # -- Helper Functions --------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------

    def equalsExcept(list1, list2, exemptIndices):
        for i,(l1,l2) in enumerate(zip(list1,list2)):
            if i in exemptIndices: continue
            if l1 != l2: return False
        return True

    def embedGateUnitary(Ugate, labels):  # Ugate should be in std basis (really no other basis it could be since gm and pp are only for acting on dm space)
        iTensorProdBlks = [ tensorBlkIndices[label] for label in labels ] # index of tensor product block (of state space) a bit label is part of
        if len(set(iTensorProdBlks)) > 1:
            raise ValueError("All qubit labels of a multi-qubit gate must correspond to the same tensor-product-block of the state space")

        iTensorProdBlk = iTensorProdBlks[0] #because they're all the same (tested above)
        tensorProdBlkLabels = stateSpaceLabels[iTensorProdBlk]
        basisInds = [] # list of *state* indices of each component of the tensor product block
        for l in tensorProdBlkLabels:
            if l.startswith('L'): basisInds.append([0])
            elif l.startswith('Q'): basisInds.append([0,1])
            else: raise ValueError("Invalid state space label: %s" % l)

        tensorBlkBasis = list(_itertools.product(*basisInds)) #state-space basis (remember tensor-prod-blocks are in state space)
        N = len(tensorBlkBasis) #size of state space (not density matrix space, which is N**2)

        labelIndices = [ tensorProdBlkLabels.index(label) for label in labels ]
        labelMultipliers = []; stateSpaceDim = 1
        for l in reversed(labels):
            labelMultipliers.append(stateSpaceDim)
            if l.startswith('L'): stateSpaceDim *= 1 #Warning? - having a gate operate on an L label doesn't really do anything...
            elif l.startswith('Q'): stateSpaceDim *= 2
        labelMultipliers.reverse() #reverse back to labels order (labels was reversed in loop above)
        labelMultipliers = _np.array(labelMultipliers,'i') #so we can use _np.dot below
        assert(stateSpaceDim == Ugate.shape[0] == Ugate.shape[1])

        # Unitary op approach: build unitary acting on state space than use kron => map acting on vec(density matrix) space
        UcohBlk = _np.identity( N, 'complex' ) # unitary matrix operating on relevant tensor product block part of state
        for i,b1 in enumerate(tensorBlkBasis):
            for j,b2 in enumerate(tensorBlkBasis):
                if equalsExcept(b1,b2,labelIndices): #if all parts of tensor prod match except for qubit(s) we're operating on
                    gate_b1 = _np.array([ b1[K] for K in labelIndices ],'i') #basis indices for just the qubits we're operating on 
                    gate_b2 = _np.array([ b2[K] for K in labelIndices ],'i') # - i.e. those corresponding to the given Ugate
                    gate_i = _np.dot(labelMultipliers, gate_b1)
                    gate_j = _np.dot(labelMultipliers, gate_b2)
                    UcohBlk[i,j] = Ugate[ gate_i, gate_j ] # fill in element
                    #FUTURE: could keep track of what Ugate <-> UcohBlk elements for parameterization here

        UcohBlkc = UcohBlk.conjugate()
        gateBlk = _np.kron(UcohBlk,UcohBlkc) # N^2 x N^2 mx operating on vectorized tensor product block of densty matrix

        #print "DEBUG: Ugate = \n", Ugate
        #print "DEBUG: UcohBlk = \n", UcohBlk

        #Map gateBlk's basis into final gate basis (shift basis indices due to the composition of different direct-sum
        # blocks along diagonal of final gate mx)
        offset = sum( [ blockDims[i]**2 for i in range(0,iTensorProdBlk) ] ) #number of basis elements preceding our block's elements
        finalGateInStdBasis = _np.identity( gateDim, 'complex' )             # operates on entire state space (direct sum of tensor prod. blocks)
        finalGateInStdBasis[offset:offset+N**2,offset:offset+N**2] = gateBlk # gateBlk gets offset along diagonal by the numer of preceding basis elements

        if parameterization != "full":
            raise ValueError("Unitary embedding is only implemented for parmeterization='full'")

        if basis == "std":
            return _Gate.FullyParameterizedGate(finalGateInStdBasis)
        elif basis == "gm":
            return _Gate.FullyParameterizedGate( _BT.basisChg_StdToGellMann(finalGateInStdBasis, blockDims) )
        elif basis == "pp":
            return _Gate.FullyParameterizedGate( _BT.basisChg_StdToPauliProd(finalGateInStdBasis, blockDims) )
        else: 
            raise ValueError("Invalid 'basis' parameter: %s (must by 'std', 'gm', or 'pp')" % basis)


    def embedGate(gatemx, labels, indicesToParameterize="all"):
        #print "DEBUG: embedGate gatemx = \n", gatemx
        iTensorProdBlks = [ tensorBlkIndices[label] for label in labels ] # index of tensor product block (of state space) a bit label is part of
        if len(set(iTensorProdBlks)) > 1:
            raise ValueError("All qubit labels of a multi-qubit gate must correspond to the same tensor-product-block of the state space")

        iTensorProdBlk = iTensorProdBlks[0] #because they're all the same (tested above)
        tensorProdBlkLabels = stateSpaceLabels[iTensorProdBlk]
        basisInds = [] # list of possible *density-matrix-space* indices of each component of the tensor product block
        for l in tensorProdBlkLabels:
            if l.startswith('L'): basisInds.append([0]) # I
            elif l.startswith('Q'): basisInds.append([0,1,2,3])  # I, X, Y, Z
            else: raise ValueError("Invalid state space label: %s" % l)

        tensorBlkEls = list(_itertools.product(*basisInds)) #dm-space basis
        lookup_blkElIndex = { tuple(b):i for i,b in enumerate(tensorBlkEls) } # index within vec(tensor prod blk) of each basis el
        N = len(tensorBlkEls) #size of density matrix space
        assert( N == blockDims[iTensorProdBlk]**2 )

        # Gate matrix approach: insert elements of gatemx into map acting on vec(density matrix) space
        gateBlk = _np.identity( N, 'd' ) # matrix operating on vec(tensor product block), (tensor prod blk is a part of the total density mx)
          #Note: because we're in the Pauil-product basis this is a *real* matrix (and gatemx should have only real elements and be in the pp basis)
        
        # Separate the components of the tensor product that are not operated on, i.e. that our final map just acts as identity w.r.t.
        basisInds_noop = basisInds[:]
        labelIndices = [ tensorProdBlkLabels.index(label) for label in labels ]
        for labelIndex in sorted(labelIndices,reverse=True):
            del basisInds_noop[labelIndex]
        tensorBlkEls_noop = list(_itertools.product(*basisInds_noop)) #dm-space basis for noop-indices only
        parameterToBaseIndicesMap = {}

        def decomp_gate_index(indx): 
            """ Decompose index of a Pauli-product matrix into indices of each Pauli in the product """
            ret = []; divisor = 1; divisors = []
            #print "Decomp %d" % indx,
            for l in labels: 
                divisors.append(divisor)
                if l.startswith('Q'): divisor *= 4
                elif l.startswith('L'): divisor *= 1
            for d in reversed(divisors):
                ret.append( indx // d )
                indx = indx % d
            #print " => %s (div = %s)" % (str(ret), str(divisors))
            return ret

        def merge_gate_and_noop_bases(gate_b, noop_b): 
            """ 
            Merge the Pauli basis indices for the "gate"-parts of the total basis
            contained in gate_b (i.e. of the components of the tensor product space
            that are operated on) and the "noop"-parts contained in noop_b.  Thus,
            len(gate_b) + len(noop_b) == len(basisInds), and this function merges
            together basis indices for the operated-on and not-operated-on tensor
            product components.  
            Note: return value always have length == len(basisInds) == number of components
            """
            ret = list(noop_b[:])    #start with noop part...
            for li,b_el in sorted( zip(labelIndices,gate_b), key=lambda x: x[0]):
                ret.insert(li, b_el) #... and insert gate parts at proper points
            return ret

        
        for gate_i in range(gatemx.shape[0]):     # rows ~ "output" of the gate map
            for gate_j in range(gatemx.shape[1]): # cols ~ "input"  of the gate map
                if indicesToParameterize == "all":
                    iParam = gate_i*gatemx.shape[1] + gate_j #index of (i,j) gate parameter in 1D array of parameters (flatten gatemx)
                    parameterToBaseIndicesMap[ iParam ] = []
                elif (gate_i,gate_j) in indicesToParameterize:
                    iParam = indicesToParameterize.index( (gate_i,gate_j) )
                    parameterToBaseIndicesMap[ iParam ] = []
                else: 
                    iParam = None #so we don't parameterize below

                gate_b1 = decomp_gate_index(gate_i) # gate_b? are lists of dm basis indices, one index per
                gate_b2 = decomp_gate_index(gate_j) #  tensor product component that the gate operates on (2 components for a 2-qubit gate)

                for i,b_noop in enumerate(tensorBlkEls_noop): #loop over all state configurations we don't operate on - so really a loop over diagonal dm elements
                    b_out = merge_gate_and_noop_bases(gate_b1, b_noop)  # using same b_noop for in and out says we're acting
                    b_in  = merge_gate_and_noop_bases(gate_b2, b_noop)  #  as the identity on the no-op state space
                    out_vec_index = lookup_blkElIndex[ tuple(b_out) ] # index of output dm basis el within vec(tensor block basis)
                    in_vec_index  = lookup_blkElIndex[ tuple(b_in) ]  # index of input dm basis el within vec(tensor block basis)

                    gateBlk[ out_vec_index, in_vec_index ] = gatemx[ gate_i, gate_j ]
                    if iParam is not None: 
                        # keep track of what gateBlk <-> gatemx elements for parameterization
                        parameterToBaseIndicesMap[ iParam ].append( (out_vec_index, in_vec_index) )


        #Map gateBlk's basis into final gate basis (shift basis indices due to the composition of different direct-sum
        # blocks along diagonal of final gate mx)
        offset = sum( [ blockDims[i]**2 for i in range(0,iTensorProdBlk) ] ) #number of basis elements preceding our block's elements
        finalGate = _np.identity( gateDim, 'd' )              # operates on entire state space (direct sum of tensor prod. blocks)
        finalGate[offset:offset+N,offset:offset+N] = gateBlk  # gateBlk gets offset along diagonal by the number of preceding basis elements
        # Note: final is a *real* matrix whose basis is the pauli-product basis in the iTensorProdBlk-th block, concatenated with
        #   bases for the other blocks - say the "std" basis (which one does't matter since the identity is the same for std, gm, and pp)

        #print "DEBUG: embedGate gateBlk = \n", gateBlk
        full_finalToPP = _np.identity( gateDim, 'complex' )
        full_ppToFinal = _np.identity( gateDim, 'complex' )

        if basis == "std":
            ppToStd = _BT.GetPauliProdToStdBasisTransformMatrix(blockDims[iTensorProdBlk]); stdToPP = _np.linalg.inv(ppToStd)
            full_ppToFinal[offset:offset+N,offset:offset+N] = ppToStd
            full_finalToPP[offset:offset+N,offset:offset+N] = stdToPP            
            realMx = False

        elif basis == "gm":
            ppToStd = _BT.GetPauliProdToStdBasisTransformMatrix(blockDims[iTensorProdBlk]); stdToPP = _np.linalg.inv(ppToStd)
            gmToStd = _BT.GetGellMannToStdBasisTransformMatrix(blockDims[iTensorProdBlk]); stdToGM = _np.linalg.inv(gmToStd)
            full_ppToFinal[offset:offset+N,offset:offset+N] = _np.dot( stdToGM, ppToStd )
            full_finalToPP[offset:offset+N,offset:offset+N] = _np.dot( stdToPP, gmToStd )
            realMx = True

        elif basis == "pp":
            # Note: finalGate may have some non-power-of-2 dimensional blocks that can't be in a "pp" basis,
            #  but don't check for this here as they'll be caught by any gate term that operates on that block
            realMx = True

        else: raise ValueError("Invalid 'basis' parameter: %s (must by 'std', 'gm', or 'pp')" % basis)

        if parameterization == "full" or gatemx.shape == gateBlk.shape:
            finalGateInFinalBasis = _np.dot(full_ppToFinal, _np.dot( finalGate, full_finalToPP))
            return _Gate.FullyParameterizedGate( _np.real(finalGateInFinalBasis) if realMx else finalGateInFinalBasis )
        elif parameterization == "linear":
            #OLD (INCORRECT) -- but could give this as paramArray if gave zeros as base matrix instead of finalGate
            # paramArray = gatemx.flatten() if indicesToParameterize == "all" else _np.array([gatemx[t] for t in indicesToParameterize])

            #Set all params to *zero* since base matrix contains all initial elements -- parameters just give deviation
            paramArray = _np.zeros( gatemx.size if indicesToParameterize == "all" else len(indicesToParameterize), 'd' )

            return _Gate.LinearlyParameterizedGate( finalGate, paramArray, parameterToBaseIndicesMap,
                                                    full_ppToFinal, full_finalToPP, realMx )
        else:
            raise ValueError("Invalid 'parameterization' parameter: %s (must by 'full' or 'linear')" % parameterization)


    # ----------------------------------------------------------------------------------------------------------------------------------------
    # -- End Helper Functions ----------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------------------------


    #print "DB: dim = ",dim, " dmDim = ",dmDim
    gateInFinalBasis = None #what will become the final gate matrix

    exprTerms = gateExpr.split(':')
    for exprTerm in exprTerms:

        l = exprTerm.index('('); r = exprTerm.rindex(')')
        gateName = exprTerm[0:l]
        argsStr = exprTerm[l+1:r]
        args = argsStr.split(',')

        if gateName == "I":
            labels = args # qubit labels (TODO: what about 'L' labels? -- not sure if they work with this...)
            stateSpaceDim = 1
            for l in labels:
                if s.startswith('Q'): stateSpaceDim *= 2   
                elif s.startswith('L'): stateSpaceDim *= 1
                else: raise ValueError("Invalid state space label: %s" % l)

            if unitaryEmbedding:
                Ugate = _np.identity(stateSpaceDim, 'complex') #complex because in std state space basis
                gateTermInFinalBasis = embedGateUnitary(Ugate, tuple(labels)) #Ugate assumed to be in std basis (really the only option)
            else:
                pp_gateMx = _np.identity(stateSpaceDim**2, 'd') # *real* 4x4 mx in Pauli-product basis -- still just the identity!
                gateTermInFinalBasis = embedGate(pp_gateMx, tuple(labels)) # pp_gateMx assumed to be in the Pauli-product basis

        elif gateName == "D":  #like 'I', but only parameterize the diagonal elements - so can be a depolarization-type map
            labels = args # qubit labels (TODO: what about 'L' labels? -- not sure if they work with this...)
            stateSpaceDim = 1
            for l in labels:
                if s.startswith('Q'): stateSpaceDim *= 2   
                elif s.startswith('L'): stateSpaceDim *= 1
                else: raise ValueError("Invalid state space label: %s" % l)

            if unitaryEmbedding or parameterization != "linear":
                raise ValueError("'D' gate only makes sense to use when unitaryEmbedding is False and parameterization == 'linear'")
            
            indicesToParameterize = [ (i,i) for i in range(1,stateSpaceDim**2) ] #parameterize only the diagonals els after the first
            pp_gateMx = _np.identity(stateSpaceDim**2, 'd') # *real* 4x4 mx in Pauli-product basis -- still just the identity!
            gateTermInFinalBasis = embedGate(pp_gateMx, tuple(labels), indicesToParameterize) # pp_gateMx assumed to be in the Pauli-product basis


        elif gateName in ('X','Y','Z'): #single-qubit gate names
            assert(len(args) == 2) # theta, qubit-index
            theta = eval( args[0], {"__builtins__":None}, {'pi': _np.pi})
            label = args[1].strip(); assert(label.startswith('Q'))

            if gateName == 'X': ex = -1j * theta*_BT.sigmax/2
            elif gateName == 'Y': ex = -1j * theta*_BT.sigmay/2
            elif gateName == 'Z': ex = -1j * theta*_BT.sigmaz/2
            
            Ugate = _spl.expm(ex) # 2x2 unitary matrix operating on single qubit in [0,1] basis
            if unitaryEmbedding:
                gateTermInFinalBasis = embedGateUnitary(Ugate, (label,)) #Ugate assumed to be in std basis (really the only option)
            else:
                Ugatec = Ugate.conjugate()
                gateMx = _np.kron(Ugate,Ugatec) # complex 4x4 mx operating on vectorized 1Q densty matrix in std basis
                pp_gateMx = _BT.basisChg_StdToPauliProd(gateMx) # *real* 4x4 mx in Pauli-product basis -- better for parameterization
                gateTermInFinalBasis = embedGate(pp_gateMx, (label,)) # pp_gateMx assumed to be in the Pauli-product basis

        elif gateName == 'N': #more general single-qubit gate
            assert(len(args) == 5) # theta, sigmaX-coeff, sigmaY-coeff, sigmaZ-coeff, qubit-index
            theta = eval( args[0], {"__builtins__":None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            sxCoeff = eval( args[1], {"__builtins__":None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            syCoeff = eval( args[2], {"__builtins__":None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            szCoeff = eval( args[3], {"__builtins__":None}, {'pi': _np.pi, 'sqrt': _np.sqrt})
            label = args[4].strip(); assert(label.startswith('Q'))

            ex = -1j * theta * ( sxCoeff * _BT.sigmax/2. + syCoeff * _BT.sigmay/2. + szCoeff * _BT.sigmaz/2.)
            Ugate = _spl.expm(ex) # 2x2 unitary matrix operating on single qubit in [0,1] basis
            if unitaryEmbedding:
                gateTermInFinalBasis = embedGateUnitary(Ugate, (label,)) #Ugate assumed to be in std basis (really the only option)
            else:
                Ugatec = Ugate.conjugate()
                gateMx = _np.kron(Ugate,Ugatec) # complex 4x4 mx operating on vectorized 1Q densty matrix in std basis
                pp_gateMx = _BT.basisChg_StdToPauliProd(gateMx) # *real* 4x4 mx in Pauli-product basis -- better for parameterization
                gateTermInFinalBasis = embedGate(pp_gateMx, (label,)) # pp_gateMx assumed to be in the Pauli-product basis
            
        elif gateName in ('CX','CY','CZ'): #two-qubit gate names
            assert(len(args) == 3) # theta, qubit-label1, qubit-label2
            theta = eval( args[0], {"__builtins__":None}, {'pi': _np.pi})
            label1 = args[1].strip(); assert(label1.startswith('Q'))
            label2 = args[2].strip(); assert(label2.startswith('Q'))

            if gateName == 'CX': ex = -1j * theta*_BT.sigmax/2.
            elif gateName == 'CY': ex = -1j * theta*_BT.sigmay/2.
            elif gateName == 'CZ': ex = -1j * theta*_BT.sigmaz/2.
            Utarget = _spl.expm(ex) # 2x2 unitary matrix operating on target qubit
            Ugate = _np.identity(4, 'complex'); Ugate[2:,2:] = Utarget #4x4 unitary matrix operating on isolated two-qubit space

            if unitaryEmbedding:
                gateTermInFinalBasis = embedGateUnitary(Ugate, (label1,label2)) #Ugate assumed to be in std basis (really the only option)
            else:
                Ugatec = Ugate.conjugate()
                gateMx = _np.kron(Ugate,Ugatec) # complex 16x16 mx operating on vectorized 2Q densty matrix in std basis
                pp_gateMx = _BT.basisChg_StdToPauliProd(gateMx) # *real* 16x16 mx in Pauli-product basis -- better for parameterization
                gateTermInFinalBasis = embedGate(pp_gateMx, (label1,label2)) # pp_gateMx assumed to be in the Pauli-product basis

        elif gateName == "LX":  #TODO - better way to describe leakage?
            assert(len(args) == 3) # theta, dmIndex1, dmIndex2 - X rotation between any two density matrix basis states
            theta = eval( args[0], {"__builtins__":None}, {'pi': _np.pi})
            i1 = int(args[1])  # row/column index of a single *state* within the density matrix
            i2 = int(args[2])  # row/column index of a single *state* within the density matrix
            ex = -1j * theta*_BT.sigmax/2
            Ugate = _spl.expm(ex) # 2x2 unitary matrix operating on the i1-th and i2-th states of the state space basis

            Utot = _np.identity(dmDim, 'complex')
            Utot[ i1,i1 ] = Ugate[0,0]
            Utot[ i1,i2 ] = Ugate[0,1]
            Utot[ i2,i1 ] = Ugate[1,0]
            Utot[ i2,i2 ] = Ugate[1,1]
            Utotc = Utot.conjugate()
            gateTermInStdBasis = _np.kron(Utot,Utotc) # dmDim^2 x dmDim^2 mx operating on vectorized total densty matrix
            gateTermInReducedStdBasis = _BT.contractToStdDirectSumMx(gateTermInStdBasis, blockDims)

            if basis == "std":
                gateTermInFinalBasis = _Gate.FullyParameterizedGate(gateTermInReducedStdBasis)
            elif basis == "gm":
                gateTermInFinalBasis = _Gate.FullyParameterizedGate( _BT.basisChg_StdToGellMann(gateTermInReducedStdBasis, blockDims) )
            elif basis == "pp":
                gateTermInFinalBasis = _Gate.FullyParameterizedGate( _BT.basisChg_StdToPauliProd(gateTermInReducedStdBasis, blockDims) )
            else: 
                raise ValueError("Invalid 'basis' parameter: %s (must by 'std', 'gm', or 'pp')" % basis)

        else: raise ValueError("Invalid gate name: %s" % gateName)

        if gateInFinalBasis is None:
            gateInFinalBasis = gateTermInFinalBasis
        else:
            gateInFinalBasis = _Gate.compose( gateInFinalBasis, gateTermInFinalBasis )
            
    return gateInFinalBasis # a Gate object





def buildGateset(stateSpaceDims, stateSpaceLabels, gateLabelList, gateExpressionList, 
                 rhoExpressions, EExpressions, spamLabelDict, basis="gm", parameterization="full"):
    """
    Build a new GateSet given lists of gate labels and expressions.

    Parameters
    ----------
    stateSpaceDims : list of ints
        Dimenstions specifying the structure of the density-matrix space.  Elements
        correspond to block dimensions of an allowed density matrix in the standard
        basis, and the density-matrix space is the direct sum of linear spaces of 
        dimension block-dimension^2.

    stateSpaceLabels : a list of tuples
        Each tuple corresponds to a block of a density matrix in the standard basis
        (and therefore a component of the direct-sum density matrix space). Elements
        of a tuple are user-defined labels beginning with "L" (single level) or "Q"
        (two-level; qubit) which interpret the d-dimensional state space corresponding
        to a d x d block as a tensor product between qubit and single level systems.

    gateLabelList : list of strings
        A list of labels for each created gate in the final gateset.  To conform with
        text file parsing conventions these names should begin with a capital G and
        can be followed by any number of lowercase characters, numbers, or the
        underscore character.
        
    gateExpressionList : list of strings
        A list of gate expressions, each corresponding to a gate label in gateLabelList,
        which determine what operation each gate performs (see documentation for buildGate).

    rhoExpressions : list of strings
        A list of vector expressions for each state preparation vector (see documentation
        for buildVector).

    EExpressions : list of strings
        A list of vector expressions for each POVM effect vector (see documentation for
        buildVector).

    spamLabelDict : dict
        A dictionary mapping spam labels to (rhoIndex,EIndex) 2-tuples associating a 
        particular state preparation and effect vector with a label.  rhoIndex and EIndex
        must be within the bounds of rhoExpressions and EExpressions respectively except
        for two special cases:

        1.  EIndex can be set to -1 to mean an effect vector that is the identity - (other effect vectors)
        2.  EIndex and rhoIndex can both be set to -1 to mean a spam label that generates
            probabilities that are 1.0 - (sum of probabilities from all other spam labels).

    basis : {'gm','pp','std'}, optional
        the basis of the matrices in the returned GateSet
        - "std" = gate matrix operates on density mx expressed as sum of matrix units
        - "gm"  = gate matrix operates on dentity mx expressed as sum of normalized Gell-Mann matrices
        - "pp"  = gate matrix operates on density mx expresses as sum of tensor-product of Pauli matrices

    parameterization : {"full","linear"}, optional
        How to parameterize the gates of the resulting GateSet (see documentation for buildGate).

    Returns
    -------
    GateSet
        The created gate set.
    """
    ret = _gateset.GateSet()

    for i,rhoExpr in enumerate(rhoExpressions):
        ret.set_rhoVec( buildVector(stateSpaceDims, stateSpaceLabels, rhoExpr, basis), i)
    for i,EExpr in enumerate(EExpressions):
        ret.set_EVec( buildVector(stateSpaceDims, stateSpaceLabels, EExpr, basis), i)

    ret.set_identityVec( buildIdentityVector(stateSpaceDims, basis) )

    for label,val in spamLabelDict.iteritems():
        ret.add_SPAM_label(val[0],val[1],label)

    for (gateLabel,gateExpr) in zip(gateLabelList, gateExpressionList):
        ret.set_gate(gateLabel, buildGate(stateSpaceDims, stateSpaceLabels, gateExpr, basis, parameterization))

    return ret


def singleQubitGate(hx, hy, hz, noise=0):
    """
    Construct the single-qubit gate matrix.

    Build the gate matrix given by exponentiating -i * (hx*X + hy*Y + hz*Z),
    where X, Y, and Z are the sigma matrices.  Thus, hx, hy, and hz 
    correspond to rotation angles divided by 2.  Additionally, a uniform
    depolarization noise can be applied to the gate.

    Parameters
    ----------
    hx : float
        Coefficient of sigma-X matrix in exponent.

    hy : float
        Coefficient of sigma-Y matrix in exponent.

    hz : float
        Coefficient of sigma-Z matrix in exponent.

    noise: float, optional
        The amount of uniform depolarizing noise.

    Returns
    -------
    numpy array
        4x4 gate matrix which operates on a 1-qubit
        density matrix expressed as a vector in the 
        Pauli basis ( {I,X,Y,Z}/sqrt(2) ).
    """
    ex = -1j * (hx*_BT.sigmax + hy*_BT.sigmay + hz*_BT.sigmaz)
    D = _np.diag( [1]+[1-noise]*(4-1) )
    return _np.dot(D, _BT.stateUnitaryToPauliDensityMxOp( _spl.expm(ex) ))

def twoQubitGate(ix=0, iy=0, iz=0, xi=0, xx=0, xy=0, xz=0, yi=0, yx=0, yy=0, yz=0, zi=0, zx=0, zy=0, zz=0):
    """
    Construct the single-qubit gate matrix.

    Build the gate matrix given by exponentiating -i * (xx*XX + xy*XY + ...)
    where terms in the exponent are tensor products of two Pauli matrices.

    Parameters
    ----------
    ix : float, optional
        Coefficient of IX matrix in exponent.

    iy : float, optional
        Coefficient of IY matrix in exponent.

    iz : float, optional
        Coefficient of IZ matrix in exponent.

    xi : float, optional
        Coefficient of XI matrix in exponent.

    xx : float, optional
        Coefficient of XX matrix in exponent.

    xy : float, optional
        Coefficient of XY matrix in exponent.

    xz : float, optional
        Coefficient of XZ matrix in exponent.

    yi : float, optional
        Coefficient of YI matrix in exponent.

    yx : float, optional
        Coefficient of YX matrix in exponent.

    yy : float, optional
        Coefficient of YY matrix in exponent.

    yz : float, optional
        Coefficient of YZ matrix in exponent.

    zi : float, optional
        Coefficient of ZI matrix in exponent.

    zx : float, optional
        Coefficient of ZX matrix in exponent.

    zy : float, optional
        Coefficient of ZY matrix in exponent.

    zz : float, optional
        Coefficient of ZZ matrix in exponent.

    Returns
    -------
    numpy array
        16x16 gate matrix which operates on a 2-qubit
        density matrix expressed as a vector in the 
        Pauli-Product basis.
    """
    ex = _np.zeros( (4,4), 'complex' )
    ex += ix * _BT.sigmaix
    ex += iy * _BT.sigmaiy
    ex += iz * _BT.sigmaiz
    ex += xi * _BT.sigmaxi
    ex += xx * _BT.sigmaxx
    ex += xy * _BT.sigmaxy
    ex += xz * _BT.sigmaxz
    ex += yi * _BT.sigmayi
    ex += yx * _BT.sigmayx
    ex += yy * _BT.sigmayy
    ex += yz * _BT.sigmayz
    ex += zi * _BT.sigmazi
    ex += zx * _BT.sigmazx
    ex += zy * _BT.sigmazy
    ex += zz * _BT.sigmazz
    return _BT.stateUnitaryToPauliDensityMxOp_2Q( _spl.expm(-1j * ex) )  #TODO: fix noise op to depolarizing




###SCRATCH
# Old from embedGate:
#        for gate_i in range(gatemx.shape[0]):     # rows ~ "output" of the gate map
#            for gate_j in range(gatemx.shape[1]): # cols ~ "input"  of the gate map
#                gate_b1_ket, gate_b2_bra = decomp_gate_index(gate_i) # gate_b* are lists of state indices, one index per
#                gate_b2_ket, gate_b2_bra = decomp_gate_index(gate_j) #  tensor product component that the gate operates on (2 components for a 2-qubit gate)
#
#                for i,b_noop in enumerate(tensorBlkBasis_noop): #loop over all state configurations we don't operate on - so really a loop over diagonal dm elements        
#                    
#                    out_ket = insert_gate_basis(gate_b1_ket, b_noop)  # using same b_noop for ket & bra says we're acting
#                    out_bra = insert_gate_basis(gate_b1_bra, b_noop)  #  as the identity on the no-op state space
#                    out_blkBasis_i = lookup_blkBasisIndex[ tuple(out_ket) ] # row index of ket within tensor block basis (state space basis)
#                    out_blkBasis_j = lookup_blkBasisIndex[ tuple(out_bra) ] # col index of ket within tensor block basis (state space basis)
#                    out_vec_index = N*out_blkBasis_i + out_blkBasis_j  # index of (row,col) tensor block element (vectorized row,col)
#
#                    in_ket = insert_gate_basis(gate_b2_ket, b_noop)  # using same b_noop for ket & bra says we're acting
#                    in_bra = insert_gate_basis(gate_b2_bra, b_noop)  #  as the identity on the no-op state space
#                    in_blkBasis_i = lookup_blkBasisIndex[ tuple(in_ket) ] # row index of ket within tensor block basis (state space basis)
#                    in_blkBasis_j = lookup_blkBasisIndex[ tuple(in_bra) ] # col index of ket within tensor block basis (state space basis)
#                    in_vec_index = N*in_blkBasis_i + in_blkBasis_j  # index of (row,col) tensor block element (vectorized row,col)
#
#                    gateBlk[ out_vec_index, in_vec_index ] = gatemx[ gate_i, gate_j ]
