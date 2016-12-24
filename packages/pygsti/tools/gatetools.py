from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Utility functions operating on gate matrices """

import numpy as _np
import scipy.linalg as _spl
import warnings as _warnings

from ..tools import jamiolkowski as _jam
from ..tools import matrixtools as _mt
from ..tools import basistools as _bt

def _hack_sqrtm(A):
    return _spl.sqrtm(A) #Travis found this scipy function
                         # to be incorrect in certain cases (we need a workaround)
    #return _np.array( (_np.matrix(A))**0.5 ) #gives error about int power

def fidelity(A, B):
    """
    Returns the quantum state fidelity between density
      matrices A and B given by :

      F = Tr( sqrt{ sqrt(A) * B * sqrt(A) } )^2

    To compute process fidelity, pass this function the
    Choi matrices of the two processes, or just call
    the process_fidelity function with the gate matrices.

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
    F = (_mt.trace( _hack_sqrtm( _np.dot( sqrtA, _np.dot(B, sqrtA) ) ) ).real)**2 # Tr( sqrt{ sqrt(A) * B * sqrt(A) } )^2
    return F

def frobeniusdist(A, B):
    """
    Returns the frobenius distance between gate
      or density matrices A and B given by :

      sqrt( sum( (A_ij-B_ij)^2 ) )

    Parameters
    ----------
    A : numpy array
        First matrix.

    B : numpy array
        Second matrix.

    Returns
    -------
    float
        The resulting frobenius distance.
    """
    return _mt.frobeniusnorm(A-B)


def frobeniusdist2(A, B):
    """
    Returns the square of the frobenius distance between gate
      or density matrices A and B given by :

      sum( (A_ij-B_ij)^2 )

    Parameters
    ----------
    A : numpy array
        First matrix.

    B : numpy array
        Second matrix.

    Returns
    -------
    float
        The resulting frobenius distance.
    """
    return _mt.frobeniusnorm2(A-B)


def tracenorm(A):
    """
    Compute the trace norm of matrix A given by:

      Tr( sqrt{ A^2 } )

    Parameters
    ----------
    A : numpy array
        The matrix to compute the trace norm of.
    """
    evals = _np.linalg.eigvals( A )
    return sum( [abs(ev) for ev in evals] )


def tracedist(A, B):
    """
    Compute the trace distance between matrices A and B,
    given by:

      D = 0.5 * Tr( sqrt{ (A-B)^2 } )

    Parameters
    ----------
    A, B : numpy array
        The matrices to compute the distance between.
    """
    return 0.5 * tracenorm(A-B)



def diamonddist(A, B, mxBasis='gm', dimOrStateSpaceDims=None):
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

    #def vec(matrix_in):
    #    # Stack the columns of a matrix to return a vector
    #    return _np.transpose(matrix_in).flatten()
    #
    #def unvec(vector_in):
    #    # Slice a vector into columns of a matrix
    #    d = int(_np.sqrt(vector_in.size))
    #    return _np.transpose(vector_in.reshape( (d,d) ))


    dim = A.shape[0]
    smallDim = int(_np.sqrt(dim))
    assert(dim == A.shape[1] == B.shape[0] == B.shape[1])

    #Code below assumes *un-normalized* Jamiol-isomorphism, so multiply by density mx dimension
    JAstd = smallDim * _jam.jamiolkowski_iso(A, mxBasis, "std", dimOrStateSpaceDims)
    JBstd = smallDim * _jam.jamiolkowski_iso(B, mxBasis, "std", dimOrStateSpaceDims)

    #CHECK: Kevin's jamiolowski, which implements the un-normalized isomorphism:
    #  smallDim * _jam.jamiolkowski_iso(M, "std", "std")
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

    #Kevin's function: def diamondnorm( jamiolkowski_matrix ):
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
    try:
        prob.solve(solver="CVXOPT")
#       prob.solve(solver="ECOS")
#       prob.solve(solver="SCS")#This always fails
    except:
        _warnings.warn("CVXOPT failed - diamonddist returning -2!")
        return -2
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
#        AInStdBasis = _bt.gm_to_std(A, dimOrStateSpaceDims)
#        BInStdBasis = _bt.gm_to_std(B, dimOrStateSpaceDims)
#    elif mxBasis == "pp":
#        AInStdBasis = _bt.pp_to_std(A, dimOrStateSpaceDims)
#        BInStdBasis = _bt.pp_to_std(B, dimOrStateSpaceDims)
#    else: raise ValueError("Invalid mxBasis: %s" % mxBasis)
#
#    rndm = _np.random.RandomState(seed)
#
#    AtensorId = _np.kron(AInStdBasis,_np.identity(gate_dim))
#    BtensorId = _np.kron(BInStdBasis,_np.identity(gate_dim))
#    gateDiffTensorId = BtensorId - AtensorId # in std basis by construction
#
#    def random_two_state_density_mx():
#        x = rndm.randn(state_dim*2,1) + 1j * rndm.randn(state_dim*2,1)
#        x = x / _np.linalg.norm(x,'fro') # normalize state
#        x = _np.dot(x,_np.conj(x).T) # state => density matrix via dm = psi x psi_dag
#        x = _np.reshape(x.T,[(state_dim*2)**2,1]) #density matrix in std basis
#        return x
#
#    #Note:  We are using definition that ||X||_1 = Tr(sqrt(X^dagger X)), not 0.5 * Tr(sqrt(X^dagger X))
#    def one_shot_one_qubit_diamond_norm(gateDiffTensorId):
#        randStateDM = random_two_state_density_mx() # in std basis
#        outDM = _np.dot(gateDiffTensorId,randStateDM) # in std basis
#        outDM = _np.reshape(outDM, [2*state_dim, 2*state_dim] )
#            # Omit transposition here to save time, as that does not affect eigenvalues
#        return _np.sum(_np.abs(_np.linalg.eigvalsh(outDM)))
#
#    oneShotVals = [ one_shot_one_qubit_diamond_norm(gateDiffTensorId) for i in xrange(nSamples) ]
#    return max(oneShotVals)



#Scratch -- to remove later
  ##extract unitary from targetGate (assumes it corresponds to a unitary)
  #target_JMx = _jam.jamiolkowski_iso( targetGate )
  #target_Jevals, target_Jevecs = _np.linalg.eig(target_JMx)
  #max_target_Jevec = target_Jevecs[:,_np.argmax(target_Jevals)]  # |max_Jevec> = (U x I)|bell>

    #TODO -- need to extract ex matrix from target_U = _spl.expm(ex) and decompose ex in terms of pauli?
    # |bell> = [1,0,0,1]/sqrt(2) (1-qubit) -- not sure how to generalize yet
    #target_U = _np.sqrt(2) * _np.array( [ [max_target_Jevec[0], max_target_Jevec[1]], [max_target_Jevec[2],max_target_Jevec[3]] ] )

  #U = _spl.expm( 1j * (solnVec[0]*sigmax + solnVec[1]*sigmay + solnVec[2]*sigmaz) )
  #vU = _np.dot( _np.kron(U,I), bell ) # "Choi vector" corresponding to unitary U
  #JU = _np.kron( vU, _np.transpose(_np.conjugate(vU))) # Choi matrix corresponding to U


def jtracedist(A, B, mxBasis="gm"): #Jamiolkowski trace distance:  Tr(|J(A)-J(B)|)
    """
    Compute the Jamiolkowski trace distance between gate matrices A and B,
    given by:

      D = 0.5 * Tr( sqrt{ (J(A)-J(B))^2 } )

    where J(.) is the Jamiolkowski isomorphism map that maps a gate matrix
    to it's corresponding Choi Matrix.

    Parameters
    ----------
    A, B : numpy array
        The matrices to compute the distance between.

    mxBasis : {"std","gm","pp"}, optional
        the basis of the gate matrices A and B : standard (matrix units),
        Gell-Mann, or Pauli-product, respectively.
    """
    JA = _jam.jamiolkowski_iso(A, gateMxBasis=mxBasis)
    JB = _jam.jamiolkowski_iso(B, gateMxBasis=mxBasis)
    return tracedist(JA,JB)


def process_fidelity(A, B, mxBasis="gm"):
    """
    Returns the process fidelity between gate
      matrices A and B given by :

      F = Tr( sqrt{ sqrt(J(A)) * J(B) * sqrt(J(A)) } )^2

    where J(.) is the Jamiolkowski isomorphism map that maps a gate matrix
    to it's corresponding Choi Matrix.

    Parameters
    ----------
    A, B : numpy array
        The matrices to compute the fidelity between.

    mxBasis : {"std","gm","pp"}, optional
        the basis of the gate matrices A and B : standard (matrix units),
        Gell-Mann, or Pauli-product, respectively.
    """
    JA = _jam.jamiolkowski_iso(A, gateMxBasis=mxBasis)
    JB = _jam.jamiolkowski_iso(B, gateMxBasis=mxBasis)
    return fidelity(JA,JB)


def fidelity_upper_bound(gateMx):
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
    choi = _jam.jamiolkowski_iso(gateMx, choiMxBasis="std")
    choi_evals,choi_evecs = _np.linalg.eig(choi)
    maxF_direct = max([_np.sqrt(max(ev.real,0.0)) for ev in choi_evals]) ** 2

    iMax = _np.argmax([ev.real for ev in choi_evals]) #index of maximum eigenval
    closestVec = choi_evecs[:, iMax:(iMax+1)]

    ##print "DEBUG: closest evec = ", closestUnitaryVec
    #new_evals = _np.zeros( len(closestUnitaryVec) ); new_evals[iClosestU] = 1.0
    #closestUnitaryJmx = _np.dot(choi_evecs, _np.dot( _np.diag(new_evals), _np.linalg.inv(choi_evecs) ) ) #gives same result
    closestJmx = _np.kron( closestVec, _np.transpose(_np.conjugate(closestVec))) # closest rank-1 Jmx
    closestJmx /= _mt.trace(closestJmx)  #normalize so trace of Jmx == 1.0


    maxF = fidelity(choi, closestJmx)

    if not _np.isnan(maxF):

        #Uncomment for debugging
        #if abs(maxF - maxF_direct) >= 1e-6:
        #    print "DEBUG: gateMx:\n",gateMx
        #    print "DEBUG: choiMx:\n",choi
        #    print "DEBUG choi_evals = ",choi_evals, " iMax = ",iMax
        #    #print "DEBUG: J = \n", closestUnitaryJmx
        #    print "DEBUG: eigvals(J) = ", _np.linalg.eigvals(closestJmx)
        #    print "DEBUG: trace(J) = ", _mt.trace(closestJmx)
        #    print "DEBUG: maxF = %f,  maxF_direct = %f" % (maxF, maxF_direct)
        #    raise ValueError("ERROR: maxF - maxF_direct = %f" % (maxF -maxF_direct))
        assert(abs(maxF - maxF_direct) < 1e-6)
    else:
        maxF = maxF_direct # case when maxF is nan, due to scipy sqrtm function being buggy - just use direct F

    closestGateMx = _jam.jamiolkowski_iso_inv( closestJmx, choiMxBasis="std" )
    return maxF, closestGateMx

    #closestU_evals, closestU_evecs = _np.linalg.eig(closestUnitaryGateMx)
    #print "DEBUG: U = \n", closestUnitaryGateMx
    #print "DEBUG: closest U evals = ",closestU_evals
    #print "DEBUG:  evecs = \n",closestU_evecs


#decompose gate matrix into axis of rotation, etc
def decompose_gate_matrix(gateMx):
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

    gate_evals,gate_evecs = _np.linalg.eig(_np.asarray(gateMx))
    # fp_eigenvec = None
    # aor_eval = None; aor_eigenvec = None
    # ra_eval  = None; ra1_eigenvec = None; ra2_eigenvec = None

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
            elif abs(_np.imag( gate_evals[ conjpair_eval_indices[1][0] ] )) < TOL: iToBreak = 1

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
            x = _np.dot( _np.linalg.pinv( _np.dot(A.T,A) ), _np.dot(A.T, b))
            fixedPtVec = _np.dot(A,x); #fixedPtVec / _np.linalg.norm(fixedPtVec)
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


def unitary_to_process_mx(U):
    """
    Compute the super-operator which acts on (row)-vectorized
    density matrices from a unitary operator (matrix) U which
    acts on state vectors.  This super-operator is given by
    the tensor product of U and conjugate(U), i.e. kron(U,U.conj).

    Parameters
    ----------
    U : numpy array
        The unitary matrix which acts on state vectors.

    Returns
    -------
    numpy array
       The super-operator process matrix.
    """
    # U -> kron(U,Uc) since U rho U_dag -> kron(U,Uc)
    #  since AXB --row-vectorize--> kron(A,B.T)*vec(X)
    return _np.kron(U,_np.conjugate(U))



def error_generator(gate, target_gate, typ="logG-logT"): #or HERE
    """
    Construct the error generator from a gate and its target.

    Computes the value of the error generator given by
    errgen = log( inv(target_gate) * gate ), so that
    gate = target_gate * exp(errgen).

    Parameters
    ----------
    gate : ndarray
      The gate matrix

    target_gate : ndarray
      The target gate matrix

    typ : {"logG-logT", "logTiG"}
      The type of error generator to compute.  Allowed values are:
      
      - "logG-logT" : errgen = log(gate) - log(target_gate)
      - "logTiG" : errgen = log( dot(inv(target_gate), gate) )

    Returns
    -------
    errgen : ndarray
      The error generator.
    """
    if typ == "logG-logT":
        return  _np.real_if_close(_spl.logm(gate) - _spl.logm(target_gate),
                                  tol=10000) # in machine epsilons
    elif typ == "logTiG":
        target_gate_inv = _spl.inv(target_gate)
        return _np.real_if_close(_spl.logm(_np.dot(target_gate_inv,gate)),
                                      tol=10000) # in machine epsilons
    else:
        raise ValueError("Invalid error-generator type: %s" % typ)


def gate_from_error_generator(error_gen, target_gate, typ="logG-logT"):
    """
    Construct a gate from an error generator and a target gate.

    Inverts the computation fone in :func:`error_generator` and
    returns the value of the gate given by
    gate = target_gate * exp(error_gen).

    Parameters
    ----------
    error_gen : ndarray
      The error generator matrix

    target_gate : ndarray
      The target gate matrix

    typ : {"logG-logT", "logTiG"}
      The type of error generator to compute.  Allowed values are:
      
      - "logG-logT" : errgen = log(gate) - log(target_gate)
      - "logTiG" : errgen = log( dot(inv(target_gate), gate) )


    Returns
    -------
    ndarray
      The gate matrix.
    """
    if typ == "logG-logT":
        return _spl.expm(error_gen + _spl.logm(target_gate))
    elif typ == "logTiG":
        return _np.dot(target_gate, _spl.expm(error_gen))
    else:
        raise ValueError("Invalid error-generator type: %s" % typ)


def pauliprod_error_generators(dim, projection_type):
    """
    Compute the projections of a gate error generator onto generators
    for a standard set of errors which correspond to pauli-product elements.

    Parameters
    ----------
    dim : int
      The dimension of the error generators to be returned.  This is also the
      associated gate dimension, and must be a perfect square, as `sqrt(dim)`
      is the dimension of density matrices. For a single qubit, dim == 4.
      
    projection_type : {"hamiltonian", "stochastic"}
      The type of error generators to construct.  If "hamiltonian", then the
      Hamiltonian generators which take a density matrix rho -> -i*[ H, rho ]
      for Pauli-product matrix H.  If "stochastic", then the Stochastic error
      generators which take rho -> P*rho*P for Pauli-product matrix P.


    Returns
    -------
    generators : numpy.ndarray
      An array of shape (#Pauli-prods,dim,dim).  `generators[i]` is the
      generator corresponding to the ith Pauli-product matrix in the 
      *std* (matrix unit) basis.  (Note that  since #Pauli-prods == dim,
      the size is also (dim,dim,dim) ).
    """
    d2 = dim
    d = int(_np.sqrt(d2))

    #Get a list of the d2 Pauli-product matrices
    # (in the standard basis)
    ppMxs = _bt.pp_matrices(d)

    assert(len(ppMxs) == d2)
    assert(_np.isclose(d*d,d2)) #d2 must be a perfect square

    lindbladMxs = _np.empty( (len(ppMxs),d2,d2), 'complex' )
    for i,ppMx in enumerate(ppMxs):
        if projection_type == "hamiltonian":
            lindbladMxs[i] = _bt.hamiltonian_to_lindbladian(ppMx) # in std basis
        elif projection_type == "stochastic":
            lindbladMxs[i] = _bt.stochastic_lindbladian(ppMx) # in std basis
        else:
            raise ValueError("Invalid projection_type argument: %s"
                             % projection_type)
        norm = _np.linalg.norm(lindbladMxs[i].flat)
        if not _np.isclose(norm,0):
            lindbladMxs[i] /= norm #normalize projector
            assert(_np.isclose(_np.linalg.norm(lindbladMxs[i].flat),1.0))

    return lindbladMxs


def pauliprod_errgen_projections(errgen, projection_type,
                                 mxBasis="gm", return_generators=False):
    """
    Compute the projections of a gate error generator onto generators
    for a standard set of errors which correspond to pauli-product elements.

    Parameters
    ----------
    errgen: : ndarray
      The error generator matrix to project.
      
    projection_type : {"hamiltonian", "stochastic"}
      The type of error generators to project the gate error generator onto.
      If "hamiltonian", then use the Hamiltonian generators which take a density
      matrix rho -> -i*[ H, rho ] for Pauli-product matrix H.  If "stochastic",
      then use the Stochastic error generators which take rho -> P*rho*P for
      Pauli-product matrix P (recall P is self adjoint).

    mxBasis : {'std', 'gm','pp'}, optional
      Which basis `errgen` is represented in.  Allowed
      options are Matrix-unit (std), Gell-Mann (gm) and
      Pauli-product (pp).

    return_generators : bool, optional
      If True, return the error generators projected against along with the
      projection values themseves.

    Returns
    -------
    projections : numpy.ndarray
      An array of length equal to the dimension of the gate, which 
      is of course the size of the appropriate Pauli-product basis.

    generators : numpy.ndarray
      Only returned when `return_generators == True`.  An array of shape
      (gate_dim,gate_dim,gate_dim) such that  `generators[i]` is the
      generator corresponding to the ith Pauli-product matrix.  Note 
      that these matricies are in the *std* (matrix unit) basis.
    """

    if mxBasis == "pp":   errgen_std = _bt.pp_to_std(errgen)
    elif mxBasis == "gm": errgen_std = _bt.gm_to_std(errgen)
    elif mxBasis == "std": errgen_std = errgen
    else: raise ValueError("Invalid basis specifier: %s" % mxBasis)

    d2 = errgen.shape[0]
    d = int(_np.sqrt(d2))
    nQubits = _np.log2(d)

    #Get a list of the d2 generators (in corresspondence with the
    #  Pauli-product matrices given by _bt.pp_matrices(d) ).
    lindbladMxs = pauliprod_error_generators(d2, projection_type) # in std basis

    assert(len(lindbladMxs) == d2)
    assert(_np.isclose(d*d,d2)) #d2 must be a perfect square

    projections = _np.empty( len(lindbladMxs), 'd' )
    for i,lindbladMx in enumerate(lindbladMxs):
        proj = _np.real_if_close(_np.vdot( errgen_std.flatten(), lindbladMx.flatten() ))
        assert(_np.isreal(proj))
        projections[i] = proj

    if return_generators:
        return projections, lindbladMxs
    else:
        return projections


def lindblad_error_generators(dmbasis, normalize):
    """
    Compute the superoperator-generators corresponding to Lindblad terms.

    This routine computes the Hamiltonian and Non-Hamiltonian ("other") 
    superoperator generators which correspond to the terms of the Lindblad
    expression:
    
    L(rho) = sum_i( h_i [B_i,rho] ) + 
             sum_ij( o_ij * (B_i rho B_j^dag -
                             0.5( rho B_j^dag B_i + B_j^dag B_i rho) ) )

    where {B_i} is a basis for Hilbert Schmidt (density matrix) space with the
    identity element removed so that each B_i is traceless.  If we write L(rho)
    in terms of superoperators H_i and O_ij,
    
    L(rho) = sum_i( h_i H_i(rho) ) + sum_ij( o_ij O_ij(rho) )

    then this function computes the matrices for H_i and O_ij using the given
    density matrix basis.  Thus, if `dmbasis` is expressed in the standard
    basis (as it should be), the returned matrices are also in this basis.

    If these elements are used as projectors it may be usedful to normalize 
    them (by setting `normalize=True`).  Note, however, that these projectors
    are not all orthogonal - in particular the O_ij's are not orthogonal to 
    one another.

    Parameters
    ----------
    dmbasis : list
        A list of basis matrices {B_i} *including* the identity as the first
        element, making this argument easily obtained by call to
        :func:`pp_matrices` or a similar function.  The matrices are expected
        to be in the standard basis, and should be traceless except for the
        identity.

    normalize : bool
        Whether or not generators should be normalized so that 
        numpy.linalg.norm(generator.flat) == 1.0  Note that the generators 
        will still, in general, be non-orthogonal.


    Returns
    -------
    ham_generators : numpy.ndarray
        An array of shape (d-1,d,d), where d is the size of the basis,
        i.e. d == len(dmbasis).  `ham_generators[i]` gives the matrix for H_i.

    other_generators : numpy.ndarray
        An array of shape (d-1,d-1,d,d), where d is the size of the basis.
        `other_generators[i,j]` gives the matrix for O_ij.
    """
    mxs = dmbasis # list of basis matrices (assumed to be in std basis)
    d2 = len(mxs)
    d = int(_np.sqrt(d2))
    assert(_np.isclose(d*d,d2)), "basis size  must be a perfect square"
    assert(_np.isclose(_np.linalg.norm(mxs[0]-_np.identity(d)/_np.sqrt(d)),0)),\
        "The first matrix in 'dmbasis' must be the identity"

    hamLindbladTerms = _np.empty( (d2-1,d2,d2), 'complex' )
    for i,B in enumerate(mxs[1:]): #don't include identity
        hamLindbladTerms[i] = _bt.hamiltonian_to_lindbladian(B) # in std basis
        if normalize:
            norm = _np.linalg.norm(hamLindbladTerms[i].flat)
            if not _np.isclose(norm,0):
                hamLindbladTerms[i] /= norm #normalize projector
                assert(_np.isclose(_np.linalg.norm(hamLindbladTerms[i].flat),1.0))


    otherLindbladTerms = _np.empty( (d2-1,d2-1,d2,d2), 'complex' )
    for i,Lm in enumerate(mxs[1:]): #don't include identity
        for j,Ln in enumerate(mxs[1:]): #don't include identity
            #print("DEBUG NONHAM LIND (%d,%d)" % (i,j)) #DEBUG!!!
            otherLindbladTerms[i,j] = _bt.nonham_lindbladian(Lm,Ln)
            if normalize:
                norm = _np.linalg.norm(otherLindbladTerms[i,j].flat)
                if not _np.isclose(norm,0):
                    otherLindbladTerms[i,j] /= norm #normalize projector
                    assert(_np.isclose(_np.linalg.norm(
                                otherLindbladTerms[i,j].flat),1.0))
            #I don't think this is true in general, but appears to be true for "pp" basis (why?)
            #if j < i: # check that other[i,j] == other[j,i].C, i.e. other is Hermitian
            #    assert(_np.isclose(_np.linalg.norm(
            #                otherLindbladTerms[i,j]-
            #                otherLindbladTerms[j,i].conjugate()),0))


    #Check for orthogonality - otherLindblad terms are *not* orthogonal!
    #N = otherLindbladTerms.shape[0]
    #for i in range(N):
    #    for j in range(N):
    #        v1 = otherLindbladTerms[i,j].flatten()
    #        for k in range(N):
    #            for l in range(N):
    #                if k == i and l == j: continue
    #                v2 = otherLindbladTerms[k,l].flatten()
    #                if not _np.isclose(0, _np.vdot(v1,v2)):
    #                    print("%d,%d <-> %d,%d dot = %g [%g]" % (i,j,k,l,_np.vdot(v1,v2),_np.dot(v1,v2)))
    #                    #print("v1 = ",v1)
    #                    #print("v2 = ",v2)
    #                #    assert(False)
    #                #assert(_np.isclose(0, _np.vdot(v1,v2)))

    #Check hamiltonian error gens are orthogonal to others
    #N = otherLindbladTerms.shape[0]
    #for i,hlt in enumerate(hamLindbladTerms):
    #    v1 = hlt.flatten()
    #    for j in range(N):
    #        for k in range(N):
    #            v2 = otherLindbladTerms[j,k].flatten()
    #            assert(_np.isclose(0, _np.vdot(v1,v2)))                

    return hamLindbladTerms, otherLindbladTerms


def pauliprod_lindblad_errgen_projections(errgen, mxBasis="gm",
                                          normalize=True, 
                                          return_generators=False):
    """
    Compute the projections of a gate error generator onto generators
    for the Lindblad-term errors when expressed in the pauli-product basis.

    Parameters
    ----------
    errgen: : ndarray
      The error generator matrix to project.
      
    mxBasis : {'std', 'gm','pp'}, optional
      Which basis `errgen` is represented in.  Allowed
      options are Matrix-unit (std), Gell-Mann (gm) and
      Pauli-product (pp).

    normalize : bool, optional
        Whether or not the generators being projected onto are normalized, so
        that numpy.linalg.norm(generator.flat) == 1.0.  Note that the generators
        will still, in general, be non-orthogonal.

    return_generators : bool, optional
      If True, return the error generators projected against along with the
      projection values themseves.

    Returns
    -------
    ham_projections : numpy.ndarray
      An array of length d-1, where d is the dimension of the gate,
      giving the projections onto the Hamiltonian-type Lindblad terms.

    other_projections : numpy.ndarray
      An array of shape (d-1,d-1), where d is the dimension of the gate,
      giving the projections onto the "other"-type Lindblad terms.

    ham_generators : numpy.ndarray
      The Hamiltonian-type Lindblad term generators, as would be returned
      from `lindblad_error_generators(pp_matrices(sqrt(d)), normalize)`.
      Shape is (d-1,d,d), and `ham_generators[i]` is in the standard basis.

    other_generators : numpy.ndarray
      The "other"-type Lindblad term generators, as would be returned
      from `lindblad_error_generators(pp_matrices(sqrt(d)), normalize)`.
      Shape is (d-1,d-1,d,d), and `other_generators[i]` is in the std basis.
    """

    if mxBasis == "pp":   errgen_std = _bt.pp_to_std(errgen)
    elif mxBasis == "gm": errgen_std = _bt.gm_to_std(errgen)
    elif mxBasis == "std": errgen_std = errgen
    else: raise ValueError("Invalid basis specifier: %s" % mxBasis)

    d2 = errgen.shape[0]
    d = int(_np.sqrt(d2))
    nQubits = _np.log2(d)

    #Get a list of the d2 generators (in corresspondence with the
    #  Pauli-product matrices given by _bt.pp_matrices(d) ).
    hamGens,otherGens = lindblad_error_generators(_bt.pp_matrices(d), 
                                                  normalize) # in std basis

    assert(hamGens.shape == (d2-1,d2,d2))
    assert(otherGens.shape == (d2-1,d2-1,d2,d2))
    assert(_np.isclose(d*d,d2)) #d2 must be a perfect square

    #OLD: just lump hamiltonian generators in with the others, even though they
    #   should be orthogonal to all the non-hamiltonian error generators.
    #hamProjs = _np.empty( hamGens.shape[0], 'd' )
    #for i,genMx in enumerate(hamGens):
    #    hamProjs[i] = _np.real_if_close(_np.vdot( errgen_std.flatten(), genMx.flatten() ))
    #    assert(_np.isreal(hamProjs[i]))

    #OLD: doesn't work because otherGens are *not* orthogonal (but they are linearly independent)
    #otherProjs = _np.empty( (otherGens.shape[0],otherGens.shape[1]), 'complex' )
    #for i in range(otherGens.shape[0]):
    #    for j in range(otherGens.shape[1]):
    #        otherProjs[i,j] = _np.vdot( errgen_std.flatten(),
    #                                    otherGens[i,j].flatten())
    #        if i==j: 
    #            otherProjs[i,j] = _np.real_if_close(otherProjs[i,j])
    #            assert(_np.isreal(otherProjs[i,j]))
    #        if i > j:
    #            #print("DEBUG conjugates (%d,%d):" % (i,j)) #DEBUG!!!
    #            #print("other[%d,%d] = \n" % (i,j), otherGens[i,j])
    #            #print("other[%d,%d] = \n" % (j,i), otherGens[j,i])
    #            assert(_np.isclose(otherProjs[i,j], otherProjs[j,i].conjugate())) 
    #             #at least for "pp" basis, otherProjs is hermetian when computed this way...

    #Perform linear least squares solve to find "projections" onto each otherGens element - defined so that
    #  sum_i projection_i * otherGen_i = (errgen_std-ham_errgen) as well as possible.
   
    #ham_error_gen = _np.einsum('i,ijk', hamProjs, hamGens)
    #other_errgen = errgen_std - ham_error_gen #what's left once hamiltonian errors are projected out
    
    #Do linear least squares soln to expressing errgen_std as a linear combo
    # of the lindblad generators
    M = _np.concatenate( (hamGens.reshape((d2-1,d2**2)).T, otherGens.reshape(((d2-1)**2,d2**2)).T), axis=1)
    assert(_np.linalg.matrix_rank(M,1e-7) == M.shape[1]) #check err gens are linearly independent
    Mdag = M.T.conjugate()
    projs = _np.dot(_np.linalg.inv(_np.dot(Mdag,M)),
                         _np.dot(Mdag,errgen_std.flatten()))
    hamProjs = projs[0:(d2-1)]
    otherProjs = projs[(d2-1):]

    hamProjs.shape = (hamGens.shape[0],)
    otherProjs.shape = (otherGens.shape[0],otherGens.shape[1])

    if return_generators:
        return hamProjs, otherProjs, hamGens, otherGens
    else:
        return hamProjs, otherProjs
