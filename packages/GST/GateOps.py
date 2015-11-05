""" Utility functions operating on gate matrices """
import numpy as _np
import scipy.linalg as _spl
import scipy.optimize as _spo
import BasisTools as _BT
import JamiolkowskiOps as _JOps
import Optimize as _OPT
from MatrixOps import trace as _trace
from GateSetConstruction import singleQubitGate as _singleQubitGate

def _hack_sqrtm(A):
    return _spl.sqrtm(A) #Travis found this scipy function 
                         # to be incorrect in certain cases (we need a workaround)
    #return _np.array( (_np.matrix(A))**0.5 ) #gives error about int power

def Fidelity(A, B):
    """
    Returns the quantum state fidelity between density
      matrices A and B given by :

      F = Tr( sqrt{ sqrt(A) * B * sqrt(A) } )^2

    To compute process fidelity, pass this function the
    Choi matrices of the two processes, or just call
    the ProcessFidelity function with the gate matrices.

    Parameters
    ----------
    A : numpy array
        First density matrix.

    B : numpy array
        Second density matrix.

    Returns
    -------
    float
        The resulting fidelity.
    """
    evals,U = _np.linalg.eig(A)
    if len([ev for ev in evals if abs(ev) > 1e-8]) == 1: #special case when A is rank 1
        ivec = _np.argmax(evals)
        vec  = U[:,ivec:(ivec+1)]
        F = _np.dot( _np.conjugate(_np.transpose(vec)), _np.dot(B, vec) ).real # vec^T * B * vec
        return F

    evals,U = _np.linalg.eig(B)
    if len([ev for ev in evals if abs(ev) > 1e-8]) == 1: #special case when B is rank 1 (recally fidelity is sym in args)
        ivec = _np.argmax(evals)
        vec  = U[:,ivec:(ivec+1)]
        F = _np.dot( _np.conjugate(_np.transpose(vec)), _np.dot(A, vec) ).real # vec^T * A * vec
        return F

    sqrtA = _hack_sqrtm(A) #_spl.sqrtm(A)
    assert( _np.linalg.norm( _np.dot(sqrtA,sqrtA) - A ) < 1e-8 ) #test the scipy sqrtm function
    F = (_trace( _hack_sqrtm( _np.dot( sqrtA, _np.dot(B, sqrtA) ) ) ).real)**2 # Tr( sqrt{ sqrt(A) * B * sqrt(A) } )^2
    return F


def DiamondNorm(A, B, mxBasis='gm', dimOrStateSpaceDims=None):
    """
    Returns the approximate diamond norm describing the difference between gate
    matrices A and B given by :

      D = ||A - B ||_diamond = sup_rho || AxI(rho) - BxI(rho) ||_1 
      
    Parameters
    ----------
    A, B : numpy array
        The *gate* matrices to use when computing the diamond norm.

    mxBasis : {"std","gm","pp"}, optional
        the basis of the gate matrices A and B : standard (matrix units),
        Gell-Mann, or Pauli-product, respectively.

    dimOrStateSpaceDims : int or list of ints, optional
        Structure of the density-matrix space, which further specifies the basis
        of gateMx (see BasisTools).

    Returns
    -------
    float
       Diamond norm
    """

    #currently cvxpy is only needed for this function, so don't import until here
    import cvxpy as _cvxpy 

    # This SDP implementation is a modified version of Kevin's code

    #Compute the diamond norm

    #Uses the primal SDP from arXiv:1207.5726v2, Sec 3.2
    
    #Maximize 1/2 ( < J(phi), X > + < J(phi).dag, X.dag > )
    #Subject to  [[ I otimes rho0, X],
    #            [X.dag, I otimes rho1]] >> 0
    #              rho0, rho1 are density matrices
    #              X is linear operator
    
    #Jamiolkowski representation of the process
    #  J(phi) = sum_ij Phi(Eij) otimes Eij
    
    #< A, B > = Tr(A.dag B)
    
    def vec(matrix_in):
        # Stack the columns of a matrix to return a vector
        return _np.transpose(matrix_in).flatten()
    
    def unvec(vector_in):
        # Slice a vector into columns of a matrix
        d = int(_np.sqrt(vector_in.size))
        return _np.transpose(vector_in.reshape( (d,d) ))


    dim = A.shape[0]
    smallDim = int(_np.sqrt(dim))
    assert(dim == A.shape[1] == B.shape[0] == B.shape[1])

    #Code below assumes *un-normalized* Jamiol-isomorphism, so multiply by density mx dimension
    JAstd = smallDim * _JOps.opWithJamiolkowskiIsomorphism(A, mxBasis, "std", dimOrStateSpaceDims)
    JBstd = smallDim * _JOps.opWithJamiolkowskiIsomorphism(B, mxBasis, "std", dimOrStateSpaceDims)
    
    #CHECK: Kevin's jamiolowski, which implements the un-normalized isomorphism:
    #  smallDim * _JOps.opWithJamiolkowskiIsomorphism(M, "std", "std")
    #def kevins_jamiolkowski(process, representation = 'superoperator'):
    #    # Return the Choi-Jamiolkowski representation of a quantum process
    #    # Add methods as necessary to accept different representations
    #    process = _np.array(process)
    #    if representation == 'superoperator':
    #        # Superoperator is the linear operator acting on vec(rho)
    #        dimension = int(_np.sqrt(process.shape[0]))
    #        print "dim = ",dimension
    #        jamiolkowski_matrix = _np.zeros([dimension**2, dimension**2], dtype='complex')
    #        for i in range(dimension**2):
    #            Ei_vec= _np.zeros(dimension**2)
    #            Ei_vec[i] = 1
    #            output = unvec(_np.dot(process,Ei_vec))
    #            tmp = _np.kron(output, unvec(Ei_vec))
    #            print "E%d = \n" % i,unvec(Ei_vec)
    #            #print "contrib =",_np.kron(output, unvec(Ei_vec))
    #            jamiolkowski_matrix += tmp
    #        return jamiolkowski_matrix
    #JAstd_kev = jamiolkowski(A)
    #JBstd_kev = jamiolkowski(B)
    #print "diff A = ",_np.linalg.norm(JAstd_kev/2.0-JAstd)
    #print "diff B = ",_np.linalg.norm(JBstd_kev/2.0-JBstd)

    #Kevin's function: def diamond_norm( jamiolkowski_matrix ):
    jamiolkowski_matrix = JBstd-JAstd

    # Here we define a bunch of auxiliary matrices because CVXPY doesn't use complex numbers

    K = jamiolkowski_matrix.real # J.real
    L = jamiolkowski_matrix.imag # J.imag

    Y = _cvxpy.Variable(dim, dim) # X.real
    Z = _cvxpy.Variable(dim, dim) # X.imag
    
    sig0 = _cvxpy.Variable(smallDim,smallDim) # rho0.real
    sig1 = _cvxpy.Variable(smallDim,smallDim) # rho1.real
    tau0 = _cvxpy.Variable(smallDim,smallDim) # rho1.imag
    tau1 = _cvxpy.Variable(smallDim,smallDim) # rho1.imag

    ident = _np.identity(smallDim, 'd')

    objective = _cvxpy.Maximize( _cvxpy.trace( K.T * Y + L.T * Z) )
    constraints = [ _cvxpy.bmat( [
                        [ _cvxpy.kron(ident, sig0), Y, -_cvxpy.kron(ident, tau0), -Z],
                        [ Y.T, _cvxpy.kron(ident, sig1), Z.T, -_cvxpy.kron(ident, tau1)],
                        [ _cvxpy.kron(ident, tau0), Z, _cvxpy.kron(ident, sig0), Y],
                        [ -Z.T, _cvxpy.kron(ident, tau1), Y.T, _cvxpy.kron(ident, sig1)]] ) >> 0,
                    _cvxpy.bmat( [[sig0, -tau0],
                           [tau0,  sig0]] ) >> 0,
                    _cvxpy.bmat( [[sig1, -tau1],
                           [tau1,  sig1]] ) >> 0,
                    sig0 == sig0.T,
                    sig1 == sig1.T,
                    tau0 == -tau0.T,
                    tau1 == -tau1.T,
                    _cvxpy.trace(sig0) == 1.,
                    _cvxpy.trace(sig1) == 1. ]

    prob = _cvxpy.Problem(objective, constraints)
#    try:
    prob.solve(solver="CVXOPT")
#        prob.solve(solver="ECOS")
#       prob.solve(solver="SCS")#This always fails
#    except:
#        return -1
    return prob.value

 
#Kenny's Monte-Carlo version   
# had additional params: nSamples=1000, seed=None
#
#    nSamples : int, optional
#        Number of Monte Carlo samples to use for diamond norm approximation.
#
#    seed : int, optional
#        If not None, a seed value for the random number generator used
#        to produce Monte Carlo samples.
#
#
#    gate_dim = A.shape[0]
#    assert(gate_dim == A.shape[1] == B.shape[0] == B.shape[1])
#
#    if dimOrStateSpaceDims is not None:
#        raise ValueError("Diamond Norm not fully implemented when dimOrStateSpaceDims is not None")
#        #Need to think more about how to deal with dimOrStateSpaceDims...
#    state_dim = _np.sqrt(gate_dim) # dimension of state space -- more complicated with dimOrStateSpaceDims?
#
#    #first, get gate matrices into std basis
#    if mxBasis == "std":
#        AInStdBasis, BInStdBasis = A, B
#    elif mxBasis == "gm" or mxBasis == "pauli":
#        AInStdBasis = _BT.basisChg_GellMannToStd(A, dimOrStateSpaceDims)
#        BInStdBasis = _BT.basisChg_GellMannToStd(B, dimOrStateSpaceDims)
#    elif mxBasis == "pp":
#        AInStdBasis = _BT.basisChg_PauliProdToStd(A, dimOrStateSpaceDims)
#        BInStdBasis = _BT.basisChg_PauliProdToStd(B, dimOrStateSpaceDims)
#    else: raise ValueError("Invalid mxBasis: %s" % mxBasis)
#
#    if seed is not None:
#        _np.random.seed(seed)
#
#    AtensorId = _np.kron(AInStdBasis,_np.identity(gate_dim))
#    BtensorId = _np.kron(BInStdBasis,_np.identity(gate_dim))
#    gateDiffTensorId = BtensorId - AtensorId # in std basis by construction
#
#    def random2StateDensityMx():
#        x = _np.random.randn(state_dim*2,1) + 1j * _np.random.randn(state_dim*2,1)
#        x = x / _np.linalg.norm(x,'fro') # normalize state
#        x = _np.dot(x,_np.conj(x).T) # state => density matrix via dm = psi x psi_dag
#        x = _np.reshape(x.T,[(state_dim*2)**2,1]) #density matrix in std basis
#        return x
#
#    #Note:  We are using definition that ||X||_1 = Tr(sqrt(X^dagger X)), not 0.5 * Tr(sqrt(X^dagger X)) 
#    def oneShotOneQubitDiamondNorm(gateDiffTensorId):
#        randStateDM = random2StateDensityMx() # in std basis
#        outDM = _np.dot(gateDiffTensorId,randStateDM) # in std basis
#        outDM = _np.reshape(outDM, [2*state_dim, 2*state_dim] )
#            # Omit transposition here to save time, as that does not affect eigenvalues
#        return _np.sum(_np.abs(_np.linalg.eigvalsh(outDM)))
#
#    oneShotVals = [ oneShotOneQubitDiamondNorm(gateDiffTensorId) for i in xrange(nSamples) ]
#    return max(oneShotVals)



#Scratch -- to remove later
  ##extract unitary from targetGate (assumes it corresponds to a unitary)
  #target_JMx = _JOps.opWithJamiolkowskiIsomorphism( targetGate )
  #target_Jevals, target_Jevecs = _np.linalg.eig(target_JMx)
  #max_target_Jevec = target_Jevecs[:,_np.argmax(target_Jevals)]  # |max_Jevec> = (U x I)|bell>

    #TODO -- need to extract ex matrix from target_U = _spl.expm(ex) and decompose ex in terms of pauli?
    # |bell> = [1,0,0,1]/sqrt(2) (1-qubit) -- not sure how to generalize yet
    #target_U = _np.sqrt(2) * _np.array( [ [max_target_Jevec[0], max_target_Jevec[1]], [max_target_Jevec[2],max_target_Jevec[3]] ] )

  #U = _spl.expm( 1j * (solnVec[0]*sigmax + solnVec[1]*sigmay + solnVec[2]*sigmaz) )
  #vU = _np.dot( _np.kron(U,I), bell ) # "Choi vector" corresponding to unitary U
  #JU = _np.kron( vU, _np.transpose(_np.conjugate(vU))) # Choi matrix corresponding to U

def getClosestUnitaryGateMx(gateMx):
  """
  Get the closest gate matrix (by maximizing Fidelity)
    to gateMx that describes a unitary quantum gate.

  Parameters
  ----------
  gateMx : numpy array
      The gate matrix to act on.

  Returns
  -------
  numpy array
      The resulting closest unitary gate matrix.
  """
  gate_JMx = _JOps.opWithJamiolkowskiIsomorphism( gateMx, choiMxBasis="std" )
  d = _np.sqrt(gateMx.shape[0])
  I = _np.identity(d)

  def getU_1Q(basisVec):  # 1 qubit version
      return _spl.expm( 1j * (basisVec[0]*_BT.sigmax + basisVec[1]*_BT.sigmay + basisVec[2]*_BT.sigmaz) )
  def getGateMx_1Q(basisVec):  # 1 qubit version
      return _singleQubitGate(basisVec[0], basisVec[1], basisVec[2])


  if gateMx.shape[0] == 4:
    #bell = _np.transpose(_np.array( [[1,0,0,1]] )) / _np.sqrt(2)
    initialBasisVec = [ 0, 0, 0]  #start with I until we figure out how to extract target unitary 
    #getU = getU_1Q
    getGateMx = getGateMx_1Q
  #Note: seems like for 2 qubits bell = [1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ]/sqrt(4) (4 zeros between 1's since state dimension is 4 ( == sqrt(gate dimension))
  else:
    raise ValueError("Can't get closest unitary for > 1 qubits yet -- need to generalize.")  

  def objectiveFunc(basisVec):
    gateMx = getGateMx(basisVec)
    JU = _JOps.opWithJamiolkowskiIsomorphism( gateMx, choiMxBasis="std" )
    # OLD: but computes JU in Pauli basis (I think) -> wrong matrix to fidelity check with gate_JMx
    #U = getU(basisVec)
    #vU = _np.dot( _np.kron(U,I), bell ) # "Choi vector" corresponding to unitary U
    #JU = _np.kron( vU, _np.transpose(_np.conjugate(vU))) # Choi matrix corresponding to U
    return -Fidelity(gate_JMx, JU)
  
  printObjFunc = _OPT.createObjFuncPrinter(objectiveFunc)
  solution = _spo.minimize(objectiveFunc, initialBasisVec,  options={'maxiter': 10000},
                           method='Nelder-Mead',callback=None, tol=1e-8) # if verbosity > 2 else None
  gateMx = getGateMx(solution.x)

  #print "DEBUG: Best Fidelity = ",-solution.fun
  #print "DEBUG: Using vector = ", solution.x
  #print "DEBUG: Gate Mx = \n", gateMx
  #print "DEBUG: Chi Mx = \n", _JOps.opWithJamiolkowskiIsomorphism( gateMx)
  #return -solution.fun, gateMx
  return gateMx
  

def getFidelityUpperBound(gateMx):
    """
    Get an upper bound on the fidelity of the given
      gate matrix with any unitary gate matrix.

    The closeness of the result to one tells
     how "unitary" the action of gateMx is.

    Parameters
    ----------
    gateMx : numpy array
        The gate matrix to act on.

    Returns
    -------
    float
        The resulting upper bound on fidelity(gateMx, anyUnitaryGateMx)
    """
    choi = _JOps.opWithJamiolkowskiIsomorphism(gateMx, choiMxBasis="std")
    choi_evals,choi_evecs = _np.linalg.eig(choi)
    maxF_direct = max([_np.sqrt(max(ev.real,0.0)) for ev in choi_evals]) ** 2

    iMax = _np.argmax([ev.real for ev in choi_evals]) #index of maximum eigenval
    closestVec = choi_evecs[:, iMax:(iMax+1)]

    ##print "DEBUG: closest evec = ", closestUnitaryVec
    #new_evals = _np.zeros( len(closestUnitaryVec) ); new_evals[iClosestU] = 1.0
    #closestUnitaryJmx = _np.dot(choi_evecs, _np.dot( _np.diag(new_evals), _np.linalg.inv(choi_evecs) ) ) #gives same result
    closestJmx = _np.kron( closestVec, _np.transpose(_np.conjugate(closestVec))) # closest rank-1 Jmx
    closestJmx /= _trace(closestJmx)  #normalize so trace of Jmx == 1.0


    maxF = Fidelity(choi, closestJmx)
    
    if not _np.isnan(maxF):

        #Uncomment for debugging
        #if abs(maxF - maxF_direct) >= 1e-6:
        #    print "DEBUG: gateMx:\n",gateMx
        #    print "DEBUG: choiMx:\n",choi
        #    print "DEBUG choi_evals = ",choi_evals, " iMax = ",iMax
        #    #print "DEBUG: J = \n", closestUnitaryJmx
        #    print "DEBUG: eigvals(J) = ", _np.linalg.eigvals(closestJmx)
        #    print "DEBUG: trace(J) = ", _trace(closestJmx)
        #    print "DEBUG: maxF = %f,  maxF_direct = %f" % (maxF, maxF_direct)
        #    raise ValueError("ERROR: maxF - maxF_direct = %f" % (maxF -maxF_direct))
        assert(abs(maxF - maxF_direct) < 1e-6)
    else:
        maxF = maxF_direct # case when maxF is nan, due to scipy sqrtm function being buggy - just use direct F

    closestGateMx = _JOps.opWithInvJamiolkowskiIsomorphism( closestJmx, choiMxBasis="std" )
    return maxF, closestGateMx

    #closestU_evals, closestU_evecs = _np.linalg.eig(closestUnitaryGateMx)
    #print "DEBUG: U = \n", closestUnitaryGateMx
    #print "DEBUG: closest U evals = ",closestU_evals
    #print "DEBUG:  evecs = \n",closestU_evecs


#decompose gate matrix into axis of rotation, etc
def decomposeGateMatrix(gateMx):
    """
    Compute how the action of a gate matrix can be
    is decomposed into fixed points, axes of rotation, 
    angles of rotation, and decays.  Also determines
    whether a gate appears to be valid and/or unitary.

    Parameters
    ----------
    gateMx : numpy array
        The gate matrix to act on.

    Returns
    -------
    dict
       A dictionary describing the decomposed action. Keys are:

         'isValid' : bool 
             whether decomposition succeeded
         'isUnitary' : bool
             whether gateMx describes unitary action
         'fixed point' : numpy array
             the fixed point of the action
         'axis of rotation' : numpy array or nan
             the axis of rotation
         'decay of diagonal rotation terms' : float
             decay of diagonal terms
         'rotating axis 1' : numpy array or nan
             1st axis orthogonal to axis of rotation
         'rotating axis 2' : numpy array or nan
             2nd axis orthogonal to axis of rotation
         'decay of off diagonal rotation terms' : float
             decay of off-diagonal terms
         'pi rotations' : float
             angle of rotation in units of pi radians
    """

    gate_evals,gate_evecs = _np.linalg.eig(gateMx)
    fp_eigenvec = None
    aor_eval = None; aor_eigenvec = None
    ra_eval  = None; ra1_eigenvec = None; ra2_eigenvec = None

    TOL = 1e-4 #1e-7

    unit_eval_indices = [ i for (i,ev) in enumerate(gate_evals) if abs(ev - 1.0) < TOL ]
    #unit_eval_indices = [ i for (i,ev) in enumerate(gate_evals) if ev > (1.0-TOL) ]

    conjpair_eval_indices = [ ]
    for (i,ev) in enumerate(gate_evals):
        if i in unit_eval_indices: continue #don't include the unit eigenvalues in the conjugate pair count
        if any( [ (i in conjpair) for conjpair in conjpair_eval_indices] ): continue #don't include existing conjugate pairs
        for (j,ev2) in enumerate(gate_evals[i+1:]):
            if abs(ev - _np.conjugate(ev2)) < TOL:
                conjpair_eval_indices.append( (i,j+(i+1)) )
                break #don't pair i-th eigenvalue with any other (pairs should be disjoint)

    real_eval_indices = []     #indices of real eigenvalues that are not units or a part of any conjugate pair
    complex_eval_indices = []  #indices of complex eigenvalues that are not units or a part of any conjugate pair
    for (i,ev) in enumerate(gate_evals):
        if i in unit_eval_indices: continue #don't include the unit eigenvalues
        if any( [ (i in conjpair) for conjpair in conjpair_eval_indices] ): continue #don't include the conjugate pairs
        if abs(ev.imag) < TOL: real_eval_indices.append(i)
        else: complex_eval_indices.append(i)

    #if len(real_eval_indices + unit_eval_indices) > 0:
    #    max_real_eval = max([ gate_evals[i] for i in real_eval_indices + unit_eval_indices])
    #    min_real_eval = min([ gate_evals[i] for i in real_eval_indices + unit_eval_indices])
    #else: 
    #    max_real_eval = _np.nan
    #    min_real_eval = _np.nan
    #
    #fixed_points = [ gate_evecs[:,i] for i in unit_eval_indices ]
    #real_eval_axes = [ gate_evecs[:,i] for i in real_eval_indices ]
    #conjpair_eval_axes = [ (gate_evecs[:,i],gate_evecs[:,j]) for (i,j) in conjpair_eval_indices ]
    #
    #ret = { }

    nQubits = _np.log2(gateMx.shape[0]) / 2
    if nQubits == 1:
        #print "DEBUG: 1 qubit decomp --------------------------"
        #print "   --> evals = ", gate_evals
        #print "   --> unit eval indices = ", unit_eval_indices
        #print "   --> conj eval indices = ", conjpair_eval_indices
        #print "   --> unpaired real eval indices = ", real_eval_indices

        #Special case: if have two conjugate pairs, check if one (or both) are real
        #  and break the one with the largest (real) value into two unpaired real evals.
        if len(conjpair_eval_indices) == 2:
            iToBreak = None
            if abs(_np.imag( gate_evals[ conjpair_eval_indices[0][0] ] )) < TOL and \
               abs(_np.imag( gate_evals[ conjpair_eval_indices[1][0] ] )) < TOL:
                iToBreak = _np.argmax( [_np.real(conjpair_eval_indices[0][0]), _np.real(conjpair_eval_indices[1][0])] )
            elif abs(_np.imag( gate_evals[ conjpair_eval_indices[0][0] ] )) < TOL: iToBreak = 0
            elif abs(_np.imag( gate_evals[ conjpair_eval_indices[0][0] ] )) < TOL: iToBreak = 1

            if iToBreak is not None:
                real_eval_indices.append( conjpair_eval_indices[iToBreak][0])
                real_eval_indices.append( conjpair_eval_indices[iToBreak][1])
                del conjpair_eval_indices[iToBreak]


        #Find eigenvector corresponding to fixed point (or closest we can get).   This
        # should be a unit eigenvalue with identity eigenvector.
        if len(unit_eval_indices) > 0:
            #Find linear least squares solution within possibly degenerate unit-eigenvalue eigenspace
            # of eigenvector closest to identity density mx (the desired fixed point), then orthogonalize
            # the remaining eigenvectors w.r.t this one.
            A = _np.take(gate_evecs, unit_eval_indices, axis=1)
            b = _np.array( [[1],[0],[0],[0]], 'd') #identity density mx
            x = _np.dot( _np.linalg.inv( _np.dot(A.T,A) ), _np.dot(A.T, b))
            fixedPtVec = _np.dot(A,x); fixedPtVec / _np.linalg.norm(fixedPtVec)
            fixedPtVec = fixedPtVec[:,0]
            
            iLargestContrib = _np.argmax(_np.abs(x)) #index of gate eigenvector which contributed the most
            for ii,i in enumerate(unit_eval_indices):
                if ii == iLargestContrib:
                    gate_evecs[:,i] = fixedPtVec
                    iFixedPt = i
                else:
                    gate_evecs[:,i] = gate_evecs[:,i] - _np.vdot(fixedPtVec,gate_evecs[:,i])*fixedPtVec
                    for jj,j in enumerate(unit_eval_indices[:ii]):
                        if jj == iLargestContrib: continue
                        gate_evecs[:,i] = gate_evecs[:,i] - _np.vdot(gate_evecs[:,j],gate_evecs[:,i])*gate_evecs[:,j]
                    gate_evecs[:,i] /= _np.linalg.norm(gate_evecs[:,i])

        elif len(real_eval_indices) > 0:
            # just take eigenvector corresponding to the largest real eigenvalue?
            #iFixedPt = real_eval_indices[ _np.argmax( [ gate_evals[i] for i in real_eval_indices ] ) ]

            # ...OR take eigenvector corresponding to a real unpaired eigenvalue closest to identity:
            idmx = _np.array( [[1],[0],[0],[0]], 'd') #identity density mx
            iFixedPt = real_eval_indices[ _np.argmin( [ _np.linalg.norm(gate_evecs[i]-idmx) for i in real_eval_indices ] ) ]
            
        else:
            #No unit or real eigenvalues => two complex conjugate pairs or unpaired complex evals --> bail out
            return { 'isValid': False, 'isUnitary': False, 'msg': "All evals are complex." }


        #Find eigenvector corresponding to axis of rotation: find the *largest* unpaired real/unit eval
        indsToConsider = (unit_eval_indices + real_eval_indices)[:]
        del indsToConsider[ indsToConsider.index(iFixedPt) ] #don't consider fixed pt evec
        
        if len(indsToConsider) > 0:
            iRotAxis = indsToConsider[ _np.argmax( [ gate_evals[i] for i in indsToConsider ] ) ]
        else:
            #No unit or real eigenvalues => an unpaired complex eval --> bail out
            return { 'isValid': False, 'isUnitary': False, 'msg': "Unpaired complex eval." }
            
        #There are only 2 eigenvalues left -- hopefully a conjugate pair giving rotation
        inds = list(range(4))
        del inds[ inds.index( iFixedPt ) ]
        del inds[ inds.index( iRotAxis ) ]
        if abs( gate_evals[inds[0]] - _np.conjugate(gate_evals[inds[1]])) < TOL:
            iConjPair1,iConjPair2 = inds
        else:
            return { 'isValid': False, 'isUnitary': False, 'msg': "No conjugate pair for rotn." }

        return { 'isValid': True,
                 'isUnitary': bool(len(unit_eval_indices) >= 2),
                 'fixed point': gate_evecs[:,iFixedPt],
                 'axis of rotation': gate_evecs[:,iRotAxis],
                 'rotating axis 1': gate_evecs[:,iConjPair1],
                 'rotating axis 2': gate_evecs[:,iConjPair2],
                 'decay of diagonal rotation terms': 1.0 - abs(gate_evals[iRotAxis]),
                 'decay of off diagonal rotation terms': 1.0 - abs(gate_evals[iConjPair1]),
                 'pi rotations': _np.angle(gate_evals[iConjPair1])/_np.pi,
                 'msg': "Success" }        
                
    else:
        return { 'isValid': False,
                 'isUnitary': False,
                 'msg': "Unsupported number of qubits: %d" % nQubits }
                         
