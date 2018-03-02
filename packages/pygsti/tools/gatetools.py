""" Utility functions operating on gate matrices """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import scipy.linalg as _spl
import scipy.sparse as _sps
import scipy.sparse.linalg as _spsl
import warnings as _warnings

from . import jamiolkowski as _jam
from . import matrixtools as _mt
from . import lindbladtools as _lt
from . import compattools as _compat
from . import basistools as _bt
from ..baseobjs import Basis as _Basis
from ..baseobjs.basis import basis_matrices as _basis_matrices

def _mut(i,j,N):
    mx = _np.zeros( (N,N), 'd'); mx[i,j] = 1.0
    return mx

def _hack_sqrtm(A):
    sqrt,_ = _spl.sqrtm(A, disp=False) #Travis found this scipy function
                                       # to be incorrect in certain cases (we need a workaround)
    if _np.any(_np.isnan(sqrt)):  #this is sometimes a good fallback when sqrtm doesn't work.
        ev,U = _np.linalg.eig(A)
        sqrt = _np.dot( U, _np.dot( _np.diag(_np.sqrt(ev)), _np.linalg.inv(U)))
        
    return sqrt

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

def residuals(A, B):
    """
    Calculate residuals between the elements of two matrices

    Parameters
    ----------
    A : numpy array
        First matrix.

    B : numpy array
        Second matrix.

    Returns
    -------
    np.array
        residuals
    """
    return (A-B).flatten()


def tracenorm(A):
    """
    Compute the trace norm of matrix A given by:

      Tr( sqrt{ A^dagger * A } )

    Parameters
    ----------
    A : numpy array
        The matrix to compute the trace norm of.
    """
    if _np.linalg.norm(A - _np.conjugate(A.T)) < 1e-8:
        #Hermitian, so just sum eigenvalue magnitudes
        return _np.sum( _np.abs( _np.linalg.eigvals( A ) ) )
    else:
        #Sum of singular values (positive by construction)
        return _np.sum( _np.linalg.svd(A, compute_uv=False) )

def tracedist(A, B):
    """
    Compute the trace distance between matrices A and B,
    given by:

      D = 0.5 * Tr( sqrt{ (A-B)^dagger * (A-B) } )

    Parameters
    ----------
    A, B : numpy array
        The matrices to compute the distance between.
    """
    return 0.5 * tracenorm(A-B)



def diamonddist(A, B, mxBasis='gm', return_x=False):
    """
    Returns the approximate diamond norm describing the difference between gate
    matrices A and B given by :

      D = ||A - B ||_diamond = sup_rho || AxI(rho) - BxI(rho) ||_1

    Parameters
    ----------
    A, B : numpy array
        The *gate* matrices to use when computing the diamond norm.

    mxBasis : Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    return_x : bool, optional
        Whether to return a numpy array encoding the state (rho) at 
        which the maximal trace distance occurs.

    Returns
    -------
    dm : float
       Diamond norm
    W : numpy array
       Only returned if `return_x = True`.  Encodes the state rho, such that
       `dm = trace( |(J(A)-J(B)).T * W| )`.
    """
    mxBasis = _bt.build_basis_for_matrix(A, mxBasis)

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
    JAstd = smallDim * _jam.fast_jamiolkowski_iso_std(A, mxBasis)
    JBstd = smallDim * _jam.fast_jamiolkowski_iso_std(B, mxBasis)

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
        return (-2, _np.zeros((dim,dim))) if return_x else -2

    if return_x:
        X = Y.value + 1j*Z.value #encodes state at which maximum trace-distance occurs
        return prob.value, X
    else:
        return prob.value

def jtracedist(A, B, mxBasis=None): #Jamiolkowski trace distance:  Tr(|J(A)-J(B)|)
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

    mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).
    """
    if mxBasis is None:
        mxBasis = _Basis('gm', int(round(_np.sqrt(A.shape[0]))))
    JA = _jam.fast_jamiolkowski_iso_std(A, mxBasis)
    JB = _jam.fast_jamiolkowski_iso_std(B, mxBasis)
    return tracedist(JA,JB)


def process_fidelity(A, B, mxBasis=None):
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

    mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).
    """
    if A[0,0] == 1.0 and B[0,0] == 1.0: #then assume TP-like gates & use simpler formula
        TrLambda = _np.trace( _np.dot(A, _np.linalg.inv(B)) )
        d2 = A.shape[0]
        return TrLambda / d2
    
    if mxBasis is None:
        mxBasis = _Basis('gm', int(round(_np.sqrt(A.shape[0]))))
    JA = _jam.jamiolkowski_iso(A, mxBasis)
    JB = _jam.jamiolkowski_iso(B, mxBasis)
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


def get_povm_map(gateset, povmlbl):
    """
    Constructs a gate-like quantity for the POVM within `gateset`.

    This is done by embedding the `k`-outcome classical output space of the POVM
    in the Hilbert-Schmidt space of `k` by `k` density matrices by placing the 
    classical probability distribution along the diagonal of the density matrix.
    Currently, this is only implemented for the case when `k` equals `d`, the 
    dimension of the POVM's Hilbert space.

    Parameters
    ----------
    gateset : GateSet
        The gateset supplying the POVM effect vectors and the basis those
        vectors are in.

    povmlbl : str
        The POVM label

    Returns
    -------
    numpy.ndarray
        The matrix of the "POVM map" in the `gateset.basis` basis.
    """
    povmVectors = list(gateset.povms[povmlbl].values())
    d = int(round(_np.sqrt(gateset.dim))) # density matrix is dxd
    nV = len(povmVectors)
    assert(d**2 == gateset.dim), "GateSet dimension (%d) is not a perfect square!" % gateset.dim
    #assert( nV**2 == d ), "Can only compute POVM metrics when num of effects == H space dimension"
    #   I don't think above assert is needed - should work in general (Robin?)
    povm_mx = _np.concatenate( povmVectors, axis=1 ).T # "povm map" ( B(H) -> S_k )
    
    Sk_embedding_in_std = _np.zeros( (d**2, nV) )
    for i in range(nV):
        Sk_embedding_in_std[:,i] = _mut(i,i,d).flatten()

    std_to_basis = _bt.transform_matrix("std", gateset.basis, d)
    assert(std_to_basis.shape == (d**2,d**2))

    return _np.dot(std_to_basis, _np.dot(Sk_embedding_in_std, povm_mx))


def povm_fidelity(gateset, targetGateset, povmlbl):
    """
    Computes the process (entanglement) fidelity between POVM maps.

    Parameters
    ----------
    gateset, targetGateset : GateSet
        Gate sets containing the two POVMs to compare.

    povmlbl : str
        The POVM label

    Returns
    -------
    float
    """
    povm_mx = get_povm_map(gateset, povmlbl)
    target_povm_mx = get_povm_map(targetGateset, povmlbl)
    return process_fidelity(povm_mx, target_povm_mx, targetGateset.basis)


def povm_jtracedist(gateset, targetGateset, povmlbl):
    """
    Computes the Jamiolkowski trace distance between POVM maps using :func:`jtracedist`.

    Parameters
    ----------
    gateset, targetGateset : GateSet
        Gate sets containing the two POVMs to compare.

    povmlbl : str
        The POVM label

    Returns
    -------
    float
    """
    povm_mx = get_povm_map(gateset, povmlbl)
    target_povm_mx = get_povm_map(targetGateset, povmlbl)
    return jtracedist(povm_mx, target_povm_mx, targetGateset.basis)


def povm_diamonddist(gateset, targetGateset, povmlbl):
    """
    Computes the diamond distance between POVM maps using :func:`diamonddist`.

    Parameters
    ----------
    gateset, targetGateset : GateSet
        Gate sets containing the two POVMs to compare.

    povmlbl : str
        The POVM label

    Returns
    -------
    float
    """
    povm_mx = get_povm_map(gateset, povmlbl)
    target_povm_mx = get_povm_map(targetGateset, povmlbl)
    return diamonddist(povm_mx, target_povm_mx, targetGateset.basis)


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


def process_mx_to_unitary(superop):
    """
    Compute the unitary corresponding to the (unitary-action!)
    super-operator `superop` which acts on (row)-vectorized
    density matrices.  The super-operator must be of the form
    `kron(U,U.conj)` or an error will be thrown.

    Parameters
    ----------
    superop : numpy array
        The superoperator matrix which acts on vectorized 
        density matrices (in the 'std' matrix-unit basis).

    Returns
    -------
    numpy array
       The unitary matrix which acts on state vectors.
    """
    d2 = superop.shape[0]; d = int(round(_np.sqrt(d2)))
    U = _np.empty( (d,d), 'complex')
    
    for i in range(d):
        densitymx_i = _np.zeros( (d,d), 'd' ); densitymx_i[i,i] = 1.0 # |i><i|
        UiiU = _np.dot(superop, densitymx_i.flat).reshape((d,d)) # U|i><i|U^dag
        
        if i > 0:
            j=0
            densitymx_ij = _np.zeros( (d,d), 'd' ); densitymx_ij[i,j] = 1.0 # |i><i|
            UijU = _np.dot(superop, densitymx_ij.flat).reshape((d,d)) # U|i><j|U^dag
            Uj = U[:,j]
            Ui = _np.dot(UijU, Uj)
        else:
            ##method1: use random state projection
            #rand_state = _np.random.rand(d)
            #projected_rand_state = _np.dot(UiiU, rand_state)
            #assert(_np.linalg.norm(projected_rand_state) > 1e-8)
            #projected_rand_state /= _np.linalg.norm(projected_rand_state)
            #Ui = projected_rand_state

            #method2: get eigenvector corresponding to largest eigenvalue (more robust)
            evals,evecs = _np.linalg.eig(UiiU)
            imaxeval = _np.argmax(_np.abs(evals))
            #TODO: assert other eigenvalues are much smaller?
            Ui = evecs[:,imaxeval]
            Ui /= _np.linalg.norm(Ui)
        U[:,i] = Ui
        
    return U


def error_generator(gate, target_gate, mxBasis, typ="logG-logT"):
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

    mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    typ : {"logG-logT", "logTiG", "logGTi"}
      The type of error generator to compute.  Allowed values are:
      
      - "logG-logT" : errgen = log(gate) - log(target_gate)
      - "logTiG" : errgen = log( dot(inv(target_gate), gate) )
      - "logGTi" : errgen = log( dot(gate,inv(target_gate)) )

    Returns
    -------
    errgen : ndarray
      The error generator.
    """
    TOL = 1e-8
    
    if typ == "logG-logT":
        try:
            logT = _mt.unitary_superoperator_matrix_log(target_gate, mxBasis)
        except AssertionError: #if not unitary, fall back to just taking the real log
            logT = _mt.real_matrix_log(target_gate, "raise", TOL) #make a fuss if this can't be done
        logG = _mt.approximate_matrix_log(gate, logT)

        # Both logG and logT *should* be real, so we just take the difference.
        if _np.linalg.norm(_np.imag(logG)) < TOL and \
           _np.linalg.norm(_np.imag(logT)) < TOL:
            return _np.real(logG - logT)

        #Otherwise, there could be branch cut issues or worse, so just
        # raise an error for now (maybe return a dummy if needed elsewhere?)
        raise ValueError("Could not construct a real logarithms for the" +
                         "'logG-logT' generator.  Perhaps you should use " +
                         "the 'logTiG' or 'logGTi' generator instead?")

    elif typ == "logTiG":
        target_gate_inv = _spl.inv(target_gate)
        try:
            errgen = _mt.near_identity_matrix_log(_np.dot(target_gate_inv,gate), TOL)
        except AssertionError: #not near the identity, fall back to the real log
            _warnings.warn(("Near-identity matrix log failed; falling back "
                            "to approximate log for logTiG error generator"))
            errgen = _mt.real_matrix_log(_np.dot(target_gate_inv,gate), "warn", TOL)
            
        if _np.linalg.norm(errgen.imag) > TOL:
            _warnings.warn("Falling back to approximate log for logTiG error generator")
            errgen = _mt.approximate_matrix_log(_np.dot(target_gate_inv,gate),
                                                _np.zeros(gate.shape,'d'), TOL=TOL)

    elif typ == "logGTi":
        target_gate_inv = _spl.inv(target_gate)
        try:
            errgen = _mt.near_identity_matrix_log(_np.dot(gate,target_gate_inv), TOL)
        except AssertionError: #not near the identity, fall back to the real log
            _warnings.warn(("Near-identity matrix log failed; falling back "
                            "to approximate log for logGTi error generator"))
            errgen = _mt.real_matrix_log(_np.dot(gate,target_gate_inv), "warn", TOL)
            
        if _np.linalg.norm(errgen.imag) > TOL:
            _warnings.warn("Falling back to approximate log for logGTi error generator")
            errgen = _mt.approximate_matrix_log(_np.dot(gate,target_gate_inv),
                                                _np.zeros(gate.shape,'d'), TOL=TOL)

    else:
        raise ValueError("Invalid error-generator type: %s" % typ)

    if _np.linalg.norm(_np.imag(errgen)) > TOL:
        raise ValueError("Could not construct a real generator!")
        #maybe this is actually ok, but a complex error generator will
        # need to be plotted differently, etc -- TODO
    return _np.real(errgen)



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
    elif typ == "logGTi":
        return _np.dot(_spl.expm(error_gen), target_gate)
    else:
        raise ValueError("Invalid error-generator type: %s" % typ)

def std_scale_factor(dim, projection_type):
    """
    Returns the multiplicative scaling that should be applied to the output of
    :func"`std_error_generators`, before using them as projectors, in order to
    compute the "standard" reported projection onto that type of error (i.e.
    the coefficient of the standard generator terms built un-normalized-Paulis).

    Parameters
    ----------
    dim : int
      The dimension of the error generators; also the  associated gate
      dimension.  This must be a perfect square, as `sqrt(dim)`
      is the dimension of density matrices. For a single qubit, dim == 4.
      
    projection_type : {"hamiltonian", "stochastic", "affine"}
      The type/class of error generators to get the scaling for.

    Returns
    -------
    float
    """
    d2 = dim
    d = int(_np.sqrt(d2))

    if projection_type == "hamiltonian":
        scaleFctr = 1.0 / ( d*_np.sqrt(2) )
        # so projection is coefficient of Hamiltonian term (w/un-normalized Paulis)
    elif projection_type == "stochastic":
        scaleFctr = 1.0 / d
        # so projection is coefficient of P*rho*P stochastic term in generator (w/un-normalized Paulis)
    elif projection_type == "affine":
        scaleFctr = 1.0 # so projection is coefficient of P affine term in generator (w/un-normalized Paulis)
    else:
        raise ValueError("Invalid projection_type argument: %s"
                         % projection_type)
    return scaleFctr


def std_error_generators(dim, projection_type, projection_basis):
    """
    Compute the gate error generators for a standard set of errors which
    correspond to "Hamiltonian"- or "Stochastic"-type errors in terms of the
    elements of the specified basis.

    Parameters
    ----------
    dim : int
      The dimension of the error generators to be returned.  This is also the
      associated gate dimension, and must be a perfect square, as `sqrt(dim)`
      is the dimension of density matrices. For a single qubit, dim == 4.
      
    projection_type : {"hamiltonian", "stochastic", "affine"}
      The type of error generators to construct.  If "hamiltonian", then the
      Hamiltonian generators which take a density matrix rho -> -i*[ H, rho ]
      for Pauli-product matrix H.  If "stochastic", then the Stochastic error
      generators which take rho -> P*rho*P for Pauli-product matrix P.  If
      "affine", then the affine generators which take rho -> P.

    projection_basis : {'std', 'gm', 'pp', 'qt'}
      Which basis is used to construct the error generators.  Allowed
      values are Matrix-unit (std), Gell-Mann (gm),
      Pauli-product (pp) and Qutrit (qt).

    Returns
    -------
    generators : numpy.ndarray
      An array of shape (#basis-elements,dim,dim).  `generators[i]` is the
      generator corresponding to the ith basis matrix in the 
      *std* (matrix unit) basis.  (Note that in most cases #basis-elements
      == dim, so the size of `generators` is (dim,dim,dim) ).  Each 
      generator is normalized so that as a vector it has unit Frobenius norm.
    """
    d2 = dim
    d = int(_np.sqrt(d2))

    #Get a list of the basis matrices
    mxs = _basis_matrices(projection_basis, d)

    assert(len(mxs) == d2)
    assert(_np.isclose(d*d,d2)) #d2 must be a perfect square

    lindbladMxs = _np.empty( (len(mxs),d2,d2), 'complex' )
    for i,basisMx in enumerate(mxs):
        if projection_type == "hamiltonian":
            lindbladMxs[i] = _lt.hamiltonian_to_lindbladian(basisMx) # in std basis
        elif projection_type == "stochastic":
            lindbladMxs[i] = _lt.stochastic_lindbladian(basisMx) # in std basis
        elif projection_type == "affine":
            lindbladMxs[i] = _lt.affine_lindbladian(basisMx) #in std basis
        else:
            raise ValueError("Invalid projection_type argument: %s"
                             % projection_type)
        norm = _np.linalg.norm(lindbladMxs[i].flat)
        if not _np.isclose(norm,0):
            lindbladMxs[i] /= norm #normalize projector
            assert(_np.isclose(_np.linalg.norm(lindbladMxs[i].flat),1.0))

    return lindbladMxs


def std_errgen_projections(errgen, projection_type, projection_basis,
                           mxBasis="gm", return_generators=False,
                           return_scale_fctr=False):
    """
    Compute the projections of a gate error generator onto generators
    for a standard set of errors constructed from the elements of a 
    specified basis.

    Parameters
    ----------
    errgen: : ndarray
      The error generator matrix to project.
      
    projection_type : {"hamiltonian", "stochastic", "affine"}
      The type of error generators to project the gate error generator onto.
      If "hamiltonian", then use the Hamiltonian generators which take a density
      matrix rho -> -i*[ H, rho ] for Pauli-product matrix H.  If "stochastic",
      then use the Stochastic error generators which take rho -> P*rho*P for
      Pauli-product matrix P (recall P is self adjoint).  If "affine", then
      use the affine error generators which take rho -> P (superop is |P>><<1|).

    projection_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    return_generators : bool, optional
      If True, return the error generators projected against along with the
      projection values themseves.

    return_scale_fctr : bool, optional
      If True, also return the scaling factor that was used to multply the
      projections onto *normalized* error generators to get the returned
      values.

    Returns
    -------
    projections : numpy.ndarray
      An array of length equal to the number of elements in the
      basis used to construct the projectors.  Typically this is
      is also the dimension of the gate (e.g. 4 for a single qubit).

    generators : numpy.ndarray
      Only returned when `return_generators == True`.  An array of shape
      (#basis-els,gate_dim,gate_dim) such that  `generators[i]` is the
      generator corresponding to the i-th basis element.  Note 
      that these matricies are in the *std* (matrix unit) basis.

    scale : float
      Only returned when `return_scale_fctr == True`.  A mulitplicative
      scaling constant that *has already been applied* to `projections`.
    """

    errgen_std = _bt.change_basis(errgen, mxBasis, "std")
    d2 = errgen.shape[0]
    d = int(_np.sqrt(d2))
    # nQubits = _np.log2(d)

    #Get a list of the d2 generators (in corresspondence with the
    #  Pauli-product matrices given by _basis.pp_matrices(d) ).
    lindbladMxs = std_error_generators(d2, projection_type, projection_basis) # in std basis

    assert(len(lindbladMxs) == d2)
    assert(_np.isclose(d*d,d2)) #d2 must be a perfect square

    projections = _np.empty( len(lindbladMxs), 'd' )
    for i,lindbladMx in enumerate(lindbladMxs):
        proj = _np.real_if_close(_np.vdot( errgen_std.flatten(), lindbladMx.flatten() ), tol=1000)

        #DEBUG - for checking why perfect gates gave weird projections --> log ambiguity
        #print("DB: rawproj(%d) = " % i,proj)
        #errgen_pp = errgen.copy()#_bt.change_basis(errgen_std,"std","pp")
        #lindbladMx_pp = _bt.change_basis(lindbladMx,"std","pp")
        #if proj > 1.0:
        #    for k in range(errgen_std.shape[0]):
        #        for j in range(errgen_std.shape[1]):
        #            if abs(errgen_pp[k,j].conjugate() * lindbladMx_pp[k,j]) > 1e-2:
        #                print(" [%d,%d]: + " % (k,j),errgen_pp[k,j].conjugate(),"*",lindbladMx_pp[k,j],"=",(errgen_pp[k,j].conjugate() * lindbladMx_pp[i,j]))
        
        #assert(_np.isreal(proj)), "non-real projection: %s" % str(proj) #just a warning now
        if not _np.isreal(proj): 
            _warnings.warn("Taking abs() of non-real projection: %s" % str(proj))
            proj = abs(proj)
        projections[i] = proj

    scaleFctr = std_scale_factor(d2, projection_type)
    projections *= scaleFctr
    lindbladMxs /= scaleFctr # so projections * generators give original

    ret = [projections]
    if return_generators: ret.append(lindbladMxs)
    if return_scale_fctr: ret.append(scaleFctr)
    return ret[0] if len(ret)==1 else tuple(ret)


def _assert_shape(ar, shape, sparse=False):
    """ Asserts ar.shape == shape ; works with sparse matrices too """
    if not sparse or len(shape) == 2:
        assert(ar.shape == shape), \
            "Shape mismatch: %s != %s!" % (str(ar.shape),str(shape))
    else:
        if len(shape) == 3: #first "dim" is a list
            assert(len(ar) == shape[0]), \
                "Leading dim mismatch: %d != %d!" % (len(ar),shape[0])
            assert(shape[0] == 0 or ar[0].shape == (shape[1],shape[2])), \
                "Shape mismatch: %s != %s!" % (str(ar[0].shape),str(shape[1:]))
        elif len(shape) == 4: #first 2 dims are lists
            assert(len(ar) == shape[0]), \
                "Leading dim mismatch: %d != %d!" % (len(ar),shape[0])
            assert(shape[0] == 0 or len(ar[0]) == shape[1]), \
                "Second dim mismatch: %d != %d!" % (len(ar[0]),shape[1])
            assert(shape[0] == 0 or shape[1] == 0 or ar[0][0].shape == (shape[2],shape[3])), \
                "Shape mismatch: %s != %s!" % (str(ar[0][0].shape),str(shape[2:]))
        else:
            raise NotImplementedError("Number of dimensions must be <= 4!")


def lindblad_error_generators(dmbasis_ham, dmbasis_other, normalize,
                              other_diagonal_only=False):
    """
    Compute the superoperator-generators corresponding to Lindblad terms.

    This routine computes the Hamiltonian and Non-Hamiltonian ("other") 
    superoperator generators which correspond to the terms of the Lindblad
    expression:
    
    L(rho) = sum_i( h_i [A_i,rho] ) + 
             sum_ij( o_ij * (B_i rho B_j^dag -
                             0.5( rho B_j^dag B_i + B_j^dag B_i rho) ) )

    where {A_i} and {B_i} are bases (possibly the same) for Hilbert Schmidt
    (density matrix) space with the identity element removed so that each
    A_i and B_i are traceless.  If we write L(rho) in terms of superoperators
    H_i and O_ij,
    
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
    dmbasis_ham : list
        A list of basis matrices {B_i} *including* the identity as the first
        element, for the returned Hamiltonian-type error generators.  This
        argument is easily obtained by call to  :func:`pp_matrices` or a
        similar function.  The matrices are expected to be in the standard
        basis, and should be traceless except for the identity.  Matrices
        should be NumPy arrays or SciPy CSR sparse matrices.

    dmbasis_other : list
        A list of basis matrices {B_i} *including* the identity as the first
        element, for the returned Stochastic-type error generators.  This
        argument is easily obtained by call to  :func:`pp_matrices` or a
        similar function.  The matrices are expected to be in the standard
        basis, and should be traceless except for the identity.  Matrices
        should be NumPy arrays or SciPy CSR sparse matrices.

    normalize : bool
        Whether or not generators should be normalized so that 
        numpy.linalg.norm(generator.flat) == 1.0  Note that the generators 
        will still, in general, be non-orthogonal.

    other_diagonal_only : bool, optional
        If True, only the "diagonal" Stochastic error generators are
        returned; that is, the generators corresponding to the `i==j`
        terms in the Lindblad expression.

    Returns
    -------
    ham_generators : numpy.ndarray or list of SciPy CSR matrices
        If dense matrices where given, an array of shape (d-1,d,d), where d is
        the size of the basis, i.e. d == len(dmbasis).  `ham_generators[i]`
        gives the matrix for H_i.  If sparse matrices were given, a list
        of shape (d,d) CSR matrices.

    other_generators : numpy.ndarray or list of lists of SciPy CSR matrices
        If dense matrices where given, An array of shape (d-1,d-1,d,d), where d
        is the size of the basis. `other_generators[i,j]` gives the matrix for
        O_ij.  If sparse matrices were given, a list of lists of shape (d,d)
        CSR matrices.
    """
    if dmbasis_ham is not None:
        ham_mxs = dmbasis_ham # list of basis matrices (assumed to be in std basis)
        ham_nMxs = len(ham_mxs) # usually == d2, but not necessary (e.g. w/maxWeight)
    else:
        ham_nMxs = 0

    if dmbasis_other is not None:
        other_mxs = dmbasis_other # list of basis matrices (assumed to be in std basis)
        other_nMxs = len(other_mxs) # usually == d2, but not necessary (e.g. w/maxWeight)
    else:
        other_nMxs = 0

    if ham_nMxs > 0:
        d = ham_mxs[0].shape[0]
        sparse = _sps.issparse(ham_mxs[0])
    elif other_nMxs > 0:
        d = other_mxs[0].shape[0]
        sparse = _sps.issparse(other_mxs[0])
    else: 
        d = 0 #will end up returning no generators
        sparse = False
    d2 = d**2
    normfn = _spsl.norm if sparse else _np.linalg.norm
    identityfn = (lambda d: _sps.identity(d,'d','csr')) if sparse else _np.identity

    if ham_nMxs > 0 and other_nMxs > 0:
        assert(other_mxs[0].shape[0] == ham_mxs[0].shape[0]), \
            "Bases must have the same dimension!"

    if ham_nMxs > 0:
        assert(_np.isclose(normfn(ham_mxs[0]-identityfn(d)/_np.sqrt(d)),0)),\
            "The first matrix in 'dmbasis_ham' must be the identity"

        hamLindbladTerms = [ None ] * (ham_nMxs-1) if sparse else \
                           _np.empty( (ham_nMxs-1,d2,d2), 'complex' )

        for i,B in enumerate(ham_mxs[1:]): #don't include identity
            hamLindbladTerms[i] = _lt.hamiltonian_to_lindbladian(B,sparse) # in std basis
            if normalize:
                norm = normfn(hamLindbladTerms[i]) #same as norm(term.flat)
                if not _np.isclose(norm,0):
                    hamLindbladTerms[i] /= norm #normalize projector
                    assert(_np.isclose(normfn(hamLindbladTerms[i]),1.0))
    else:
        hamLindbladTerms = None

    if other_nMxs > 0:
        assert(_np.isclose(normfn(other_mxs[0]-identityfn(d)/_np.sqrt(d)),0)),\
            "The first matrix in 'dmbasis_other' must be the identity"

        if other_diagonal_only:
            otherLindbladTerms = [ None ] * (other_nMxs-1) if sparse else \
                                 _np.empty( (other_nMxs-1,d2,d2), 'complex' )
            for i,Lm in enumerate(other_mxs[1:]): #don't include identity
                otherLindbladTerms[i] = _lt.nonham_lindbladian(Lm,Lm,sparse)
                if normalize:
                    norm = normfn(otherLindbladTerms[i]) #same as norm(term.flat)
                    if not _np.isclose(norm,0):
                        otherLindbladTerms[i] /= norm #normalize projector
                        assert(_np.isclose(normfn(otherLindbladTerms[i]),1.0))
        
        else:
            otherLindbladTerms = \
                [ [ None ] * (other_nMxs-1) for i in range(other_nMxs-1)] if sparse else \
                _np.empty( (other_nMxs-1,other_nMxs-1,d2,d2), 'complex' )

            for i,Lm in enumerate(other_mxs[1:]): #don't include identity
                for j,Ln in enumerate(other_mxs[1:]): #don't include identity
                    #print("DEBUG NONHAM LIND (%d,%d)" % (i,j)) #DEBUG!!!
                    otherLindbladTerms[i][j] = _lt.nonham_lindbladian(Lm,Ln,sparse)
                    if normalize:
                        norm = normfn(otherLindbladTerms[i][j]) #same as norm(term.flat)
                        if not _np.isclose(norm,0):
                            otherLindbladTerms[i][j] /= norm #normalize projector
                            assert(_np.isclose(normfn(otherLindbladTerms[i][j]),1.0))
                    #I don't think this is true in general, but appears to be true for "pp" basis (why?)
                    #if j < i: # check that other[i,j] == other[j,i].C, i.e. other is Hermitian
                    #    assert(_np.isclose(_np.linalg.norm(
                    #                otherLindbladTerms[i][j]-
                    #                otherLindbladTerms[j][i].conjugate()),0))
    else:
        otherLindbladTerms = None



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


def lindblad_errgen_projections(errgen, ham_basis,
                                other_basis, mxBasis="gm",
                                normalize=True, return_generators=False,
                                other_diagonal_only=False, sparse=False):
    """
    Compute the projections of a gate error generator onto generators
    for the Lindblad-term errors when expressed in the given 
    "projection basis".

    Parameters
    ----------
    errgen: : ndarray
      The error generator matrix to project.

    ham_basis: {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
        The basis is used to construct the Stochastic-type lindblad error
        Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt), list of numpy arrays, or a custom basis object.

    other_basis : {'std', 'gm', 'pp', 'qt'}, list of matrices, or Basis object
        The basis is used to construct the Stochastic-type lindblad error
        Allowed values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt), list of numpy arrays, or a custom basis object.
      
    mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source basis. Allowed values are Matrix-unit (std), 
        Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    normalize : bool, optional
      Whether or not the generators being projected onto are normalized, so
      that numpy.linalg.norm(generator.flat) == 1.0.  Note that the generators
      will still, in general, be non-orthogonal.

    return_generators : bool, optional
      If True, return the error generators projected against along with the
      projection values themseves.

    other_diagonal_only : bool, optional
      If True, then only projections onto the "diagonal" terms in the
      Lindblad expresssion are returned.

    sparse : bool, optional
      Whether to create sparse or dense basis matrices when strings
      are given as `ham_basis` and `other_basis`

    Returns
    -------
    ham_projections : numpy.ndarray
      An array of length d-1, where d is the dimension of the gate,
      giving the projections onto the Hamiltonian-type Lindblad terms.

    other_projections : numpy.ndarray
      An array of shape (d-1,d-1), where d is the dimension of the gate,
      giving the projections onto the Stochastic-type Lindblad terms.

    ham_generators : numpy.ndarray
      The Hamiltonian-type Lindblad term generators, as would be returned
      from `lindblad_error_generators(pp_matrices(sqrt(d)), normalize)`.
      Shape is (d-1,d,d), and `ham_generators[i]` is in the standard basis.

    other_generators : numpy.ndarray
      The Stochastic-type Lindblad term generators, as would be returned
      from `lindblad_error_generators(pp_matrices(sqrt(d)), normalize)`.
      Shape is (d-1,d-1,d,d), and `other_generators[i]` is in the std basis.
    """
    errgen_std = _bt.change_basis(errgen, mxBasis, "std")
    if _sps.issparse(errgen_std):
        errgen_std_flat = errgen_std.tolil().reshape(
            (errgen_std.shape[0]*errgen_std.shape[1],1) ).tocsr() # b/c lil's are only type that can reshape...
    else:
        errgen_std_flat = errgen_std.flatten()
    errgen_std = None #ununsed below, and sparse reshape doesn't copy, so mark as None
    

    d2 = errgen.shape[0]
    d = int(_np.sqrt(d2))
    #nQubits = _np.log2(d)
    
    #Get a list of the generators in corresspondence with the
    #  specified basis elements.
    if isinstance(ham_basis, _Basis):
        hamBasisMxs = ham_basis.get_composite_matrices()
    elif _compat.isstr(ham_basis):
        hamBasisMxs = _basis_matrices(ham_basis, d, sparse=sparse)
    else: 
        hamBasisMxs = ham_basis
        
    if isinstance(other_basis, _Basis):
        otherBasisMxs = other_basis.get_composite_matrices()
    elif _compat.isstr(other_basis):
        otherBasisMxs = _basis_matrices(other_basis, d, sparse=sparse)
    else: 
        otherBasisMxs = other_basis
        
    hamGens, otherGens = lindblad_error_generators(
        hamBasisMxs,otherBasisMxs,normalize,other_diagonal_only) # in std basis

    if hamBasisMxs is not None:
        bsH = len(hamBasisMxs) #basis size (not necessarily d2)
    else: bsH = 0

    if otherBasisMxs is not None:
        bsO = len(otherBasisMxs) #basis size (not necessarily d2)
    else: bsO = 0

    if bsH > 0: sparse = _sps.issparse(hamBasisMxs[0])
    elif bsO > 0: sparse = _sps.issparse(otherBasisMxs[0])
    else: sparse = False # default?
    
    assert(_np.isclose(d*d,d2)) #d2 must be a perfect square
    if bsH > 0:
        _assert_shape(hamGens, (bsH-1,d2,d2), sparse)
    if bsO > 0:
        if other_diagonal_only:
            _assert_shape(otherGens, (bsO-1,d2,d2), sparse)
        else:
            _assert_shape(otherGens, (bsO-1,bsO-1,d2,d2), sparse)

    #Perform linear least squares solve to find "projections" onto each otherGens element - defined so that
    #  sum_i projection_i * otherGen_i = (errgen_std-ham_errgen) as well as possible.
   
    #ham_error_gen = _np.einsum('i,ijk', hamProjs, hamGens)
    #other_errgen = errgen_std - ham_error_gen #what's left once hamiltonian errors are projected out
    
    #Do linear least squares soln to expressing errgen_std as a linear combo
    # of the lindblad generators
    if bsH > 0:
        if not sparse:
            H = hamGens.reshape((bsH-1,d2**2)).T #ham generators == columns
            Hdag = H.T.conjugate()

            #Do linear least squares: this is what takes the bulk of the time
            hamProjs   = _np.linalg.solve(_np.dot(Hdag,H), _np.dot(Hdag,errgen_std_flat))
            hamProjs.shape = (hamGens.shape[0],)
        else:
            rows = [ hamGen.tolil().reshape((1,d2**2)) for hamGen in hamGens ]
            H = _sps.vstack(rows, 'csr').transpose()
            Hdag = H.copy().transpose().conjugate()

            #Do linear least squares: this is what takes the bulk of the time
            if _mt.safenorm(errgen_std_flat) < 1e-8: #protect against singular RHS 
                hamProjs = _np.zeros(bsH-1, 'd')
            else:
                hamProjs   = _spsl.spsolve(Hdag.dot(H), Hdag.dot(errgen_std_flat) )
                if _sps.issparse(hamProjs): hamProjs = hamProjs.toarray().flatten()
            hamProjs.shape = (bsH-1,)
    else:
        hamProjs = None

    if bsO > 0:
        if not sparse:
            if other_diagonal_only:
                O = otherGens.reshape((bsO-1,d2**2)).T # other generators == columns
            else:
                O = otherGens.reshape(((bsO-1)**2,d2**2)).T # other generators == columns
            Odag = O.T.conjugate()
    
            #Do linear least squares: this is what takes the bulk of the time
            otherProjs = _np.linalg.solve(_np.dot(Odag,O), _np.dot(Odag,errgen_std_flat))
    
            if other_diagonal_only:
                otherProjs.shape = (otherGens.shape[0],)
            else:
                otherProjs.shape = (otherGens.shape[0],otherGens.shape[1])

        else:
            if other_diagonal_only:
                rows = [oGen.tolil().reshape((1,d2**2)) for oGen in otherGens]
                O = _sps.vstack(rows, 'csr').transpose() # other generators == columns
            else:
                rows = [oGen.tolil().reshape((1,d2**2)) for oGenRow in otherGens for oGen in oGenRow]
                O = _sps.vstack(rows, 'csr').transpose() # other generators == columns
            Odag = O.copy().transpose().conjugate() #TODO: maybe conjugate copies data?

            #Do linear least squares: this is what takes the bulk of the time
            if _mt.safenorm(errgen_std_flat) < 1e-8: #protect against singular RHS 
                otherProjs = _np.zeros(bsO-1, 'd') if other_diagonal_only else \
                             _np.zeros((bsO-1,bsO-1), 'd')
            else:
                otherProjs = _spsl.spsolve(Odag.dot(O), Odag.dot(errgen_std_flat))
                if _sps.issparse(otherProjs): otherProjs = otherProjs.toarray().flatten()
    
            if other_diagonal_only:
                otherProjs.shape = (bsO-1,)
            else:
                otherProjs.shape = (bsO-1,bsO-1)
    else:
        otherProjs = None


    #check err gens are linearly independent -- but can take a very long time, so comment out!
    #assert(_np.linalg.matrix_rank(H,1e-7) == H.shape[1])
    #assert(_np.linalg.matrix_rank(O,1e-7) == O.shape[1])
    #if False: # further check against older (slower) version
    #    M = _np.concatenate( (hamGens.reshape((bs-1,d2**2)).T, otherGens.reshape(((bs-1)**2,d2**2)).T), axis=1)
    #    assert(_np.linalg.matrix_rank(M,1e-7) == M.shape[1]) #check err gens are linearly independent
    #    Mdag = M.T.conjugate()
    #    print("DB D: %.1f" % (time.time()-t)); t = time.time()
    #    projs = _np.linalg.solve(_np.dot(Mdag,M), _np.dot(Mdag,errgen_std_flat))        
    #    hamProjs_chk = projs[0:(bs-1)]
    #    otherProjs_chk = projs[(bs-1):]
    #    assert(_np.linalg.norm(hamProjs-hamProjs_chk) < 1e-6)
    #    assert(_np.linalg.norm(otherProjs-otherProjs_chk) < 1e-6)
    
    if return_generators:
        return hamProjs, otherProjs, hamGens, otherGens
    else:
        return hamProjs, otherProjs

#TODO: replace two_qubit_gate, one_qubit_gate, unitary_to_pauligate_* with
# calls to this one and unitary_to_processmx
def rotation_gate_mx(r, mxBasis="gm"):
    """
    Construct a rotation gate matrix.

    Build the gate matrix corresponding to the unitary
    `exp(-i * (r[0]/2*PP[0]*sqrt(d) + r[1]/2*PP[1]*sqrt(d) + ...) )`
    where `PP' is the array of Pauli-product matrices 
    obtained via `pp_matrices(d)`, where `d = sqrt(len(r)+1)`.
    The division by 2 is for convention, and the sqrt(d) is to
    essentially un-normalise the matrices returned by `pp_matrices`
    to they are equal to products of the *standard* Pauli matrices.

    Parameters
    ----------
    r : tuple
        A tuple of coeffiecients, one per non-identity
        Pauli-product basis element

    mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).
.
    Returns
    -------
    numpy array
        a d^2 x d^2 gate matrix in the specified basis.
    """
    d = int(round(_np.sqrt(len(r)+1)))
    assert(d**2 == len(r)+1), "Invalid number of rotation angles"

    #get Pauli-product matrices (in std basis)
    pp = _basis_matrices('pp', d)
    assert(len(r) == len(pp[1:]))

    #build unitary (in std basis)
    ex = _np.zeros( (d,d), 'complex' )
    for rot, pp_mx in zip(r, pp[1:]):
        ex += rot/2.0 * pp_mx * _np.sqrt(d)
    U = _spl.expm(-1j * ex)
    stdGate = unitary_to_process_mx(U)

    ret = _bt.change_basis(stdGate, 'std', mxBasis, None)

    return ret



def project_gateset(gateset, targetGateset,
                    projectiontypes=('H','S','H+S','LND'),
                    genType="logG-logT"):
    """
    Construct one or more new gatesets by projecting the error generator of
    `gateset` onto some sub-space then reconstructing.

    Parameters
    ----------
    gateset : GateSet
        The gate set whose error generator should be projected.

    targetGateset : GateSet
        The set of target (ideal) gates.

    projectiontypes : tuple of {'H','S','H+S','LND','LNDCP'}
        Which projections to use.  The length of this tuple gives the
        number of `GateSet` objects returned.  Allowed values are:

        - 'H' = Hamiltonian errors
        - 'S' = Stochastic Pauli-channel errors
        - 'H+S' = both of the above error types
        - 'LND' = errgen projected to a normal (CPTP) Lindbladian
        - 'LNDF' = errgen projected to an unrestricted (full) Lindbladian

    genType : {"logG-logT", "logTiG"}
      The type of error generator to compute.  Allowed values are:
      
      - "logG-logT" : errgen = log(gate) - log(target_gate)
      - "logTiG" : errgen = log( dot(inv(target_gate), gate) )

    Returns
    -------
    projected_gatesets : list of GateSets
       Elements are projected versions of `gateset` corresponding to
       the elements of `projectiontypes`.

    Nps : list of parameter counts
       Integer parameter counts for each gate set in `projected_gatesets`.
       Useful for computing the expected log-likelihood or chi2.
    """
    
    gateLabels = list(gateset.gates.keys())  # gate labels
    basis = gateset.basis
    
    if basis.name != targetGateset.basis.name:
        raise ValueError("Basis mismatch between gateset (%s) and target (%s)!"\
                         % (gateset.basis.name, targetGateset.basis.name))
    
    # Note: set to "full" parameterization so we can set the gates below
    #  regardless of what parameterization the original gateset had.
    gsDict = {}; NpDict = {}
    for p in projectiontypes:
        gsDict[p] = gateset.copy()
        gsDict[p].set_all_parameterizations("full")
        NpDict[p] = 0

        
    errgens = [ error_generator(gateset.gates[gl],
                                targetGateset.gates[gl],
                                targetGateset.basis, genType)
                for gl in gateLabels ]

    for gl,errgen in zip(gateLabels,errgens):
        if ('H' in projectiontypes) or ('H+S' in projectiontypes):
            hamProj, hamGens = std_errgen_projections(
                errgen, "hamiltonian", basis.name, basis, True)
            ham_error_gen = _np.einsum('i,ijk', hamProj, hamGens)
            ham_error_gen = _bt.change_basis(ham_error_gen,"std",basis)
            
        if ('S' in projectiontypes) or ('H+S' in projectiontypes):
            stoProj, stoGens = std_errgen_projections(
                errgen, "stochastic", basis.name, basis, True)
            sto_error_gen = _np.einsum('i,ijk', stoProj, stoGens)
            sto_error_gen = _bt.change_basis(sto_error_gen,"std",basis)
            
        if ('LND' in projectiontypes) or ('LNDF' in projectiontypes):
            HProj, OProj, HGens, OGens = \
                lindblad_errgen_projections(
                    errgen, basis.name, basis.name, basis, normalize=False,
                    return_generators=True)
            #Note: return values *can* be None if an empty/None basis is given
            lnd_error_gen = _np.einsum('i,ijk', HProj, HGens) + \
                            _np.einsum('ij,ijkl', OProj, OGens)
            lnd_error_gen = _bt.change_basis(lnd_error_gen,"std",basis)

        targetGate = targetGateset.gates[gl]
            
        if 'H' in projectiontypes:
            gsDict['H'].gates[gl] = gate_from_error_generator(
                ham_error_gen, targetGate, genType)
            NpDict['H'] += len(hamProj)
            
        if 'S' in projectiontypes:
            gsDict['S'].gates[gl]  = gate_from_error_generator(
                sto_error_gen, targetGate, genType)
            NpDict['S'] += len(stoProj)

        if 'H+S' in projectiontypes:
            gsDict['H+S'].gates[gl] = gate_from_error_generator(
                ham_error_gen+sto_error_gen, targetGate, genType)
            NpDict['H+S'] += len(hamProj) + len(stoProj)

        if 'LNDF' in projectiontypes:
            gsDict['LNDF'].gates[gl] = gate_from_error_generator(
                lnd_error_gen, targetGate, genType)
            NpDict['LNDF'] += HProj.size + OProj.size

        if 'LND' in projectiontypes:
            evals,U = _np.linalg.eig(OProj)
            pos_evals = evals.clip(0,1e100) #clip negative eigenvalues to 0
            OProj_cp = _np.dot(U,_np.dot(_np.diag(pos_evals),_np.linalg.inv(U)))
              #OProj_cp is now a pos-def matrix
            lnd_error_gen_cp = _np.einsum('i,ijk', HProj, HGens) + \
                               _np.einsum('ij,ijkl', OProj_cp, OGens)
            lnd_error_gen_cp = _bt.change_basis(lnd_error_gen_cp,"std",basis)

            gsDict['LND'].gates[gl] = gate_from_error_generator(
                lnd_error_gen_cp, targetGate, genType)
            NpDict['LND'] += HProj.size + OProj.size

        #Removed attempt to contract H+S to CPTP by removing positive stochastic projections,
        # but this doesn't always return the gate to being CPTP (maybe b/c of normalization)...
        #sto_error_gen_cp = _np.einsum('i,ijk', stoProj.clip(None,0), stoGens)
        #  # (only negative stochastic projections OK)
        #sto_error_gen_cp = _tools.std_to_pp(sto_error_gen_cp)
        #gsHSCP.gates[gl] = _tools.gate_from_error_generator(
        #    ham_error_gen, targetGate, genType) #+sto_error_gen_cp


    #DEBUG!!!
    #print("DEBUG: BEST sum neg evals = ",_tools.sum_of_negative_choi_evals(gateset))
    #print("DEBUG: LNDCP sum neg evals = ",_tools.sum_of_negative_choi_evals(gsDict['LND']))
    
    #Check for CPTP where expected
    #assert(_tools.sum_of_negative_choi_evals(gsHSCP) < 1e-6)
    #assert(_tools.sum_of_negative_choi_evals(gsDict['LND']) < 1e-6)

    #Collect and return requrested results:
    ret_gs = [ gsDict[p] for p in projectiontypes ]
    ret_Nps = [ NpDict[p] for p in projectiontypes ]
    return ret_gs, ret_Nps

def project_to_target_eigenspace(gateset, targetGateset, EPS=1e-6):
    """
    Project each gate of `gateset` onto the eigenspace of the corresponding
    gate within `targetGateset`.  Return the resulting `GateSet`.

    Parameters
    ----------
    gateset, targetGateset : GateSet
        The gate set being projected and the gate set specifying the "target"
        eigen-spaces, respectively.

    EPS : float, optional
        Small magnitude specifying how much to "nudge" the target gates
        before eigen-decomposing them, so that their spectra will have the
        same conjugacy structure as the gates of `gateset`.

    Returns
    -------
    GateSet
    """
    ret = targetGateset.copy()
    ret.set_all_parameterizations("full") # so we can freely assign gates new values
    
    for gl,gate in gateset.gates.items():
        tgt_gate = targetGateset.gates[gl].copy()
        tgt_gate = (1.0-EPS)*tgt_gate + EPS*gate # breaks tgt_gate's degeneracies w/same structure as gate
        evals_gate = _np.linalg.eigvals(gate)
        evals_tgt, Utgt = _np.linalg.eig(tgt_gate)
        _, pairs = _mt.minweight_match(evals_tgt, evals_gate, return_pairs=True)
        
        evals = evals_gate.copy()
        for i,j in pairs: #replace target evals w/matching eval of gate
            evals[i] = evals_gate[j]

        Utgt_inv = _np.linalg.inv(Utgt)
        epgate = _np.dot(Utgt, _np.dot(_np.diag(evals), Utgt_inv))
        epgate = _np.real_if_close(epgate, tol=1000)
        if _np.linalg.norm(_np.imag(epgate)) > 1e-7:
            _warnings.warn(("Target-eigenspace-projected gate has an imaginary"
                            " component.  This usually isn't desired and"
                            " indicates a failure to match eigenvalues."))
        ret.gates[gl] = epgate

    return ret
    


#TODO: remove later? deprecate?
def unitary_to_pauligate(U):
    """
    Get the linear operator on (vectorized) density
    matrices corresponding to a n-qubit unitary
    operator on states.

    Parameters
    ----------
    U : numpy array
        A dxd array giving the action of the unitary
        on a state in the sigma-z basis.
        where d = 2 ** n-qubits

    Returns
    -------
    numpy array
        The operator on density matrices that have been
        vectorized as d**2 vectors in the Pauli basis.
    """
    assert U.shape[0] == U.shape[1], '"Unitary" matrix is not square'
    return _bt.change_basis(unitary_to_process_mx(U), 'std', 'pp')
