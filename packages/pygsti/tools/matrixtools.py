""" Matrix related utility functions """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy as _np
import scipy.linalg as _spl
import scipy.optimize as _spo
import warnings as _warnings
import itertools as _itertools

from .basistools import change_basis

def array_eq(a, b, tol=1e-8):
    """Test whether arrays `a` and `b` are equal, i.e. if `norm(a-b) < tol` """
    print(_np.linalg.norm(a-b))
    return _np.linalg.norm(a-b) < tol

def trace(M): #memory leak in numpy causes repeated trace calls to eat up all memory --TODO: Cython this
    """
    The trace of a matrix, sum_i M[i,i].

    A memory leak in some version of numpy can cause repeated calls to numpy's
    trace function to eat up all available system memory, and this function
    does not have this problem.

    Parameters
    ----------
    M : numpy array
        the matrix (any object that can be double-indexed)

    Returns
    -------
    element type of M
        The trace of M.
    """
    return sum([ M[i,i] for i in range(M.shape[0]) ])
#    with warnings.catch_warnings():
#        warnings.filterwarnings('error')
#        try:
#            ret =
#        except Warning:
#            print "BAD trace from:\n"
#            for i in range(M.shape[0]):
#                print M[i,i]
#            raise ValueError("STOP")
#    return ret

def is_hermitian(mx, TOL=1e-9):
    """
    Test whether mx is a hermitian matrix.

    Parameters
    ----------
    mx : numpy array
        Matrix to test.

    TOL : float, optional
        Tolerance on absolute magitude of elements.

    Returns
    -------
    bool
        True if mx is hermitian, otherwise False.
    """
    (m,n) = mx.shape
    for i in range(m):
        if abs(mx[i,i].imag) > TOL: return False
        for j in range(i+1,n):
            if abs(mx[i,j] - mx[j,i].conjugate()) > TOL: return False
    return True

def is_pos_def(mx, TOL=1e-9):
    """
    Test whether mx is a positive-definite matrix.

    Parameters
    ----------
    mx : numpy array
        Matrix to test.

    TOL : float, optional
        Tolerance on absolute magitude of elements.

    Returns
    -------
    bool
        True if mx is positive-semidefinite, otherwise False.
    """
    evals = _np.linalg.eigvals( mx )
    return all( [ev > -TOL for ev in evals] )

def is_valid_density_mx(mx, TOL=1e-9):
    """
    Test whether mx is a valid density matrix (hermitian,
      positive-definite, and unit trace).

    Parameters
    ----------
    mx : numpy array
        Matrix to test.

    TOL : float, optional
        Tolerance on absolute magitude of elements.

    Returns
    -------
    bool
        True if mx is a valid density matrix, otherwise False.
    """
    return is_hermitian(mx,TOL) and is_pos_def(mx,TOL) and abs(trace(mx)-1.0) < TOL


def frobeniusnorm(ar):
    """
    Compute the frobenius norm of an array (or matrix),

       sqrt( sum( each_element_of_a^2 ) )

    Parameters
    ----------
    ar : numpy array
        What to compute the frobenius norm of.  Note that ar can be any shape
        or number of dimenions.

    Returns
    -------
    float or complex
        depending on the element type of ar.
    """
    return _np.sqrt(_np.sum(ar**2))

def frobeniusnorm2(ar):
    """
    Compute the squared frobenius norm of an array (or matrix),

       sum( each_element_of_a^2 ) )

    Parameters
    ----------
    ar : numpy array
        What to compute the squared frobenius norm of.  Note that ar can be any
        shape or number of dimenions.

    Returns
    -------
    float or complex
        depending on the element type of ar.
    """
    return _np.sum(ar**2)

def nullspace(m, tol=1e-7):
    """
    Compute the nullspace of a matrix.

    Parameters
    ----------
    m : numpy array
       An matrix of shape (M,N) whose nullspace to compute.

    tol : float (optional)
       Nullspace tolerance, used when comparing singular values with zero.

    Returns
    -------
    An matrix of shape (M,K) whose columns contain nullspace basis vectors.
    """
    _,s,vh = _np.linalg.svd(m)
    rank = (s > tol).sum()
    return vh[rank:].T.copy()


def nullspace_qr(m, tol=1e-7):
    """
    Compute the nullspace of a matrix using the QR decomposition.

    The QR decomposition is faster but less accurate than the SVD
    used by :func:`nullspace`.

    Parameters
    ----------
    m : numpy array
       An matrix of shape (M,N) whose nullspace to compute.

    tol : float (optional)
       Nullspace tolerance, used when comparing diagonal values of R with zero.

    Returns
    -------
    An matrix of shape (M,K) whose columns contain nullspace basis vectors.
    """
    #if M,N = m.shape, and q,r,p = _spl.qr(...)
    # q.shape == (N,N), r.shape = (N,M), p.shape = (M,)
    q,r,_ = _spl.qr(m.T, mode='full', pivoting=True)
    rank = (_np.abs(_np.diagonal(r)) > tol).sum()
    
    #DEBUG: requires q,r,p = _sql.qr(...) above
    #assert( _np.linalg.norm(_np.dot(q,r) - m.T[:,p]) < 1e-8) #check QR decomp
    #print("Rank QR = ",rank)
    #print('\n'.join(map(str,_np.abs(_np.diagonal(r)))))
    #print("Ret = ", q[:,rank:].shape, " Q = ",q.shape, " R = ",r.shape)
    
    return q[:,rank:]

def matrix_sign(M):
    """ The "sign" matrix of `M` """
    #Notes: sign(M) defined s.t. eigvecs of sign(M) are evecs of M
    # and evals of sign(M) are +/-1 or 0 based on sign of eigenvalues of M
    
    #Using the extremely numerically stable (but expensive) Schur method
    # see http://www.maths.manchester.ac.uk/~higham/fm/OT104HighamChapter5.pdf
    N = M.shape[0];  assert(M.shape == (N,N)), "M must be square!"
    T,Z = _spl.schur(M,'complex') # M = Z T Z^H where Z is unitary and T is upper-triangular
    U = _np.zeros(T.shape,'complex') # will be sign(T), which is easy to compute
      # (U is also upper triangular), and then sign(M) = Z U Z^H

    # diagonals are easy
    U[ _np.diag_indices_from(U) ] = _np.sign(_np.diagonal(T))

    #Off diagonals: use U^2 = I or TU = UT
    # Note: Tij = Uij = 0 when i > j and i==j easy so just consider i<j case
    # 0 = sum_k Uik Ukj =  (i!=j b/c off-diag) 
    # FUTURE: speed this up by using np.dot instead of sums below
    for j in range(1,N):
        for i in range(j-1,-1,-1):
            S = U[i,i]+U[j,j]
            if _np.isclose(S,0): # then use TU = UT
                if _np.isclose(T[i,i]-T[j,j],0): # then just set to zero
                    U[i,j] = 0.0 # TODO: check correctness of this case
                else:
                    U[i,j] = T[i,j]*(U[i,i]-U[j,j])/(T[i,i]-T[j,j]) + \
                             sum([U[i,k]*T[k,j]-T[i,k]*U[k,j] for k in range(i+1,j)]) \
                             / (T[i,i]-T[j,j])
            else: # use U^2 = I
                U[i,j] = - sum([U[i,k]*U[k,j] for k in range(i+1,j)]) / S
    return _np.dot(Z, _np.dot(U, _np.conjugate(Z.T)))

    #OLD quick & dirty - not always stable
    #U,_,Vt = _np.linalg.svd(M)
    #return _np.dot(U,Vt)

def print_mx(mx, width=9, prec=4):
    """
    Print matrix in pretty format.

    Will print real or complex matrices with a desired precision and
    "cell" width.

    Parameters
    ----------
    mx : numpy array
        the matrix (2-D array) to print.

    width : int, opitonal
        the width (in characters) of each printed element

    prec : int optional
        the precision (in characters) of each printed element

    """
    print(mx_to_string(mx, width, prec))

def mx_to_string(m, width=9, prec=4):
    """
    Generate a "pretty-format" string for a matrix.

    Will generate strings for real or complex matrices with a desired
    precision and "cell" width.

    Parameters
    ----------
    mx : numpy array
        the matrix (2-D array) to convert.

    width : int, opitonal
        the width (in characters) of each converted element

    prec : int optional
        the precision (in characters) of each converted element

    Returns
    -------
    string
        matrix m as a pretty formated string.
    """
    s = ""; tol = 10**(-prec)
    if _np.max(abs(_np.imag(m))) > tol:
        return mx_to_string_complex(m, width, width, prec)

    if len(m.shape) == 1: m = m[None,:] # so it works w/vectors too
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if abs(m[i,j]) < tol: s += '{0: {w}.0f}'.format(0,w=width)
            else: s += '{0: {w}.{p}f}'.format(m[i,j].real,w=width,p=prec)
        s += "\n"
    return s

def mx_to_string_complex(m, real_width=9, im_width=9, prec=4):
    """
    Generate a "pretty-format" string for a complex-valued matrix.

    Parameters
    ----------
    mx : numpy array
        the matrix (2-D array) to convert.

    real_width : int, opitonal
        the width (in characters) of the real part of each element.

    im_width : int, opitonal
        the width (in characters) of the imaginary part of each element.

    prec : int optional
        the precision (in characters) of each element's real and imaginary parts.

    Returns
    -------
    string
        matrix m as a pretty formated string.
    """
    if len(m.shape) == 1: m = m[None,:] # so it works w/vectors too
    s = ""; tol = 10**(-prec)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if abs(m[i,j].real)<tol: s += "{0: {w}.0f}".format(0,w=real_width)
            else: s += "{0: {w}.{p}f}".format(m[i,j].real,w=real_width,p=prec)
            if abs(m[i,j].imag)<tol: s += "{0: >+{w}.0f}j".format(0,w=im_width)
            else: s += "{0: >+{w}.{p}f}j".format(m[i,j].imag,w=im_width,p=prec)
        s += "\n"
    return s


def unitary_superoperator_matrix_log(M, mxBasis):
    """ 
    Construct the logarithm of superoperator matrix `M` 
    that acts as a unitary on density-matrix space,
    (`M: rho -> U rho Udagger`) so that log(M) can be
    written as the action by Hamiltonian `H`:
    `log(M): rho -> -i[H,rho]`.
    

    Parameters
    ----------
    M : numpy array
        The superoperator matrix whose logarithm is taken

    mxBasis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    Returns
    -------
    numpy array
        A matrix `logM`, of the same shape as `M`, such that `M = exp(logM)`
        and `logM` can be written as the action `rho -> -i[H,rho]`.
    """
    from . import lindbladtools as _lt # (would create circular imports if at top)
    from . import gatetools as _gt # (would create circular imports if at top)

    M_std = change_basis(M, mxBasis, "std")
    evals = _np.linalg.eigvals(M_std)
    assert( _np.allclose(_np.abs(evals), 1.0) ) #simple but technically incomplete check for a unitary superop
                                              # (e.g. could be anti-unitary: diag(1, -1, -1, -1))
    U = _gt.process_mx_to_unitary(M_std)
    H = _spl.logm(U)/-1j # U = exp(-iH)
    logM_std = _lt.hamiltonian_to_lindbladian(H) # rho --> -i[H, rho]
    logM = change_basis(logM_std, "std", mxBasis)
    assert(_np.linalg.norm(_spl.expm(logM) - M) < 1e-8) #expensive b/c of expm - could comment for performance
    return logM
    


def near_identity_matrix_log(M, TOL=1e-8):
    """ 
    Construct the logarithm of superoperator matrix `M` that is
    near the identity.  If `M` is real, the resulting logarithm will be real.

    Parameters
    ----------
    M : numpy array
        The superoperator matrix whose logarithm is taken

    TOL : float, optional
        The tolerance used when testing for zero imaginary parts.

    Returns
    -------
    numpy array
        An matrix `logM`, of the same shape as `M`, such that `M = exp(logM)`
        and `logM` is real when `M` is real.
    """    
    # A near-identity matrix should have a unique logarithm, and it should be
    # real if the original matrix is real
    M_is_real = bool(_np.linalg.norm(M.imag) < TOL)
    logM = _spl.logm(M)
    if M_is_real:
        assert(_np.linalg.norm(logM.imag) < TOL), \
            "near_identity_matrix_log has failed to construct a real logarithm!\n" \
            + "This is probably because M is not near the identity.\n" \
            + "Its eigenvalues are: " + str(_np.linalg.eigvals(M))
        logM = logM.real
    return logM

def approximate_matrix_log(M, target_logM, targetWeight=10.0, TOL=1e-6):
    """ 
    Construct an approximate logarithm of superoperator matrix `M` that is
    real and near the `target_logM`.  The equation `M = exp( logM )` is
    allowed to become inexact in order to make `logM` close to 
    `target_logM`.  In particular, the objective function that is
    minimized is (where `||` indicates the 2-norm):

    `|exp(logM) - M|_1 + targetWeight * ||logM - target_logM||^2`

    Parameters
    ----------
    M : numpy array
        The superoperator matrix whose logarithm is taken

    target_logM : numpy array
        The target logarithm

    targetWeight : float
        A weighting factor used to blance the exactness-of-log term
        with the closeness-to-target term in the optimized objective
        function.  This value multiplies the latter term.

    TOL : float, optional
        Optimzer tolerance.

    Returns
    -------
    logM : numpy array
        An matrix of the same shape as `M`.
    """    

    assert(_np.linalg.norm(M.imag) < 1e-8), "Argument `M` must be a *real* matrix!"
    mx_shape = M.shape
    
    def _objective(flat_logM):
        logM = flat_logM.reshape(mx_shape)
        testM = _spl.expm(logM)
        ret=  targetWeight*_np.linalg.norm(logM-target_logM)**2 + \
                _np.linalg.norm(testM.flatten() - M.flatten(), 1)
        #print("DEBUG: ",ret)
        return ret
    
        #Alt objective1: puts L1 on target term
        #return _np.linalg.norm(testM-M)**2 + targetWeight*_np.linalg.norm(
        #                      logM.flatten() - target_logM.flatten(), 1)
        
        #Alt objective2: all L2 terms (ridge regression)
        #return targetWeight*_np.linalg.norm(logM-target_logM)**2 + \
        #        _np.linalg.norm(testM - M)**2

    #from .. import optimize as _opt
    #print_obj_func = _opt.create_obj_func_printer(_objective) #only ever prints to stdout!                    
    print_obj_func = None

    logM = _np.real( real_matrix_log(M, actionIfImaginary="ignore") ) #just drop any imaginary part
    initial_flat_logM = logM.flatten() # + 0.1*target_logM.flatten()
      # Note: adding some of target_logM doesn't seem to help; and hurts in easy cases

    if _objective(initial_flat_logM) > 1e-16: #otherwise initial logM is fine!
        
        #print("Initial objective fn val = ",_objective(initial_flat_logM))
        #print("Initial inexactness = ",_np.linalg.norm(_spl.expm(logM)-M),
        #      _np.linalg.norm(_spl.expm(logM).flatten()-M.flatten(), 1),
        #      _np.linalg.norm(logM-target_logM)**2)
    
        solution = _spo.minimize(_objective, initial_flat_logM,  options={'maxiter': 1000},
                                           method='L-BFGS-B',callback=print_obj_func, tol=TOL)
        logM = solution.x.reshape(mx_shape)
        #print("Final objective fn val = ",_objective(solution.x))
        #print("Final inexactness = ",_np.linalg.norm(_spl.expm(logM)-M),
        #      _np.linalg.norm(_spl.expm(logM).flatten()-M.flatten(), 1),
        #      _np.linalg.norm(logM-target_logM)**2)

    return logM
            


def real_matrix_log(M, actionIfImaginary="raise", TOL=1e-8):
    """ 
    Construct a *real* logarithm of real matrix `M`.

    This is possible when negative eigenvalues of `M` come in pairs, so
    that they can be viewed as complex conjugate pairs.

    Parameters
    ----------
    M : numpy array
        The matrix to take the logarithm of

    actionIfImaginary : {"raise","warn","ignore"}, optional
        What action should be taken if a real-valued logarithm cannot be found.
        "raise" raises a ValueError, "warn" issues a warning, and "ignore"
        ignores the condition and simply returns the complex-valued result.

    TOL : float, optional
        An internal tolerance used when testing for equivalence and zero
        imaginary parts (real-ness).

    Returns
    -------
    logM : numpy array
        An matrix `logM`, of the same shape as `M`, such that `M = exp(logM)`
    """
    assert( _np.linalg.norm(_np.imag(M)) < TOL ), "real_matrix_log must be passed a *real* matrix!"
    evals,U = _np.linalg.eig(M)
    U = U.astype("complex")

    used_indices = set()
    neg_real_pairs_real_evecs = []
    neg_real_pairs_conj_evecs = []
    unpaired_indices = []
    for i,ev in enumerate(evals):
        if i in used_indices: continue
        used_indices.add(i)
        if abs(_np.imag(ev)) < TOL and _np.real(ev) < 0:
            evec1 = U[:,i]
            if _np.linalg.norm(_np.imag(evec1)) < TOL:
                # evec1 is real, so look for ev2 corresponding to another real evec
                for j,ev2 in enumerate(evals[i+1:],start=i+1):
                    if abs(ev-ev2) < TOL and _np.linalg.norm(_np.imag(U[:,j])) < TOL:
                        used_indices.add(j)
                        neg_real_pairs_real_evecs.append( (i,j) ); break
                else: unpaired_indices.append(i)
            else:
                # evec1 is complex, so look for ev2 corresponding to the conjugate of evec1
                evec1C = evec1.conjugate()
                for j,ev2 in enumerate(evals[i+1:],start=i+1):
                    if abs(ev-ev2) < TOL and _np.linalg.norm(evec1C - U[:,j]) < TOL:
                        used_indices.add(j)
                        neg_real_pairs_conj_evecs.append( (i,j) ); break
                else: unpaired_indices.append(i)

    log_evals = _np.log(evals.astype("complex"))
      # astype guards against case all evals are real but some are negative

    #DEBUG
    #print("DB: evals = ",evals)
    #print("DB: log_evals:",log_evals)
    #for i,ev in enumerate(log_evals):
    #    print(i,": ",ev, ",".join([str(j) for j in range(U.shape[0]) if abs(U[j,i]) > 0.05]))
    #print("DB: neg_real_pairs_real_evecs = ",neg_real_pairs_real_evecs)
    #print("DB: neg_real_pairs_conj_evecs = ",neg_real_pairs_conj_evecs)
    #print("DB: evec[5] = ",mx_to_string(U[:,5]))
    #print("DB: evec[6] = ",mx_to_string(U[:,6]))
    
    for (i,j) in neg_real_pairs_real_evecs: #need to adjust evecs as well
        log_evals[i] = _np.log(-evals[i]) + 1j*_np.pi
        log_evals[j] = log_evals[i].conjugate()
        U[:,i] = (U[:,i] + 1j*U[:,j])/_np.sqrt(2)
        U[:,j] = U[:,i].conjugate()

    for (i,j) in neg_real_pairs_conj_evecs: # evecs already conjugates of each other
        log_evals[i] = _np.log(-evals[i].real) + 1j*_np.pi
        log_evals[j] = log_evals[i].conjugate()
        #Note: if *don't* conjugate j-th, then this picks *consistent* branch cut (what scipy would do), which
        # results, in general, in a complex logarithm BUT one which seems more intuitive (?) - at least permits
        # expected angle extraction, etc.
        
    logM =  _np.dot( U, _np.dot(_np.diag(log_evals), _np.linalg.inv(U) ))

    #if there are unpaired negative real eigenvalues, the logarithm might be imaginary
    mayBeImaginary = bool(len(unpaired_indices) > 0)
    imMag = _np.linalg.norm(_np.imag(logM))

    if mayBeImaginary and imMag > TOL:
        if actionIfImaginary == "raise":
            raise ValueError("Cannot construct a real log: unpaired negative" +
                             " real eigenvalues: %s" % [evals[i] for i in unpaired_indices] +
                             "\nDEBUG M = \n%s" % M + "\nDEBUG evals = %s" % evals)
        elif actionIfImaginary == "warn":
            _warnings.warn("Cannot construct a real log: unpaired negative" +
                         " real eigenvalues: %s" % [evals[i] for i in unpaired_indices])
        elif actionIfImaginary == "ignore":
            pass
        else:
            assert(False), "Invalid 'actionIfImaginary' argument: %s" % actionIfImaginary
    else:
        assert( imMag <= TOL ), "real_matrix_log failed to construct a real logarithm!"
        logM = _np.real(logM)
        
    return logM



def minweight_match(a, b, metricfn=None, return_pairs=True,
                    pass_indices_to_metricfn=False):
    """
    Matches the elements of two vectors, `a` and `b` by minimizing the
    weight between them, defined as the sum of `metricfn(x,y)` over
    all `(x,y)` pairs (`x` in `a` and `y` in `b`).

    Parameters
    ----------
    a, b : list or numpy.ndarray
        1D arrays to match elements between.

    metricfn : function, optional
        A function of two float parameters, `x` and `y`,which defines the cost
        associated with matching `x` with `y`.  If None, `abs(x-y)` is used.

    return_pairs : bool, optional
        If True, the matching is also returned.

    pass_indices_to_metricfn : bool, optional
        If True, the metric function is passed two *indices* into the `a` and
        `b` arrays, respectively, instead of the values.

    Returns
    -------
    weight_array : numpy.ndarray
        The array of weights corresponding to the min-weight matching. The sum
        of this array's elements is the minimized total weight.

    pairs : list
        Only returned when `return_pairs == True`, a list of 2-tuple pairs of
        indices `(ix,iy)` giving the indices into `a` and `b` respectively of 
        each matched pair.
    """
    assert(len(a) == len(b))
    if metricfn is None:
        metricfn = lambda x,y: abs(x-y)
        
    D = len(a)
    weightMx = _np.empty((D,D),'d')
    
    if pass_indices_to_metricfn:
        for i,x in enumerate(a):
            weightMx[i,:] = [metricfn(i,j) for j,y in enumerate(b)]
    else:
        for i,x in enumerate(a):
            weightMx[i,:] = [metricfn(x,y) for j,y in enumerate(b)]
            
    a_inds, b_inds = _spo.linear_sum_assignment(weightMx)
    assert(_np.allclose(a_inds, range(D))), "linear_sum_assignment returned unexpected row indices!"

    matched_pairs = list(zip(a_inds,b_inds))
    min_weights = weightMx[a_inds, b_inds]

    if return_pairs:
        return min_weights, matched_pairs
    else:
        return min_weights

                
def _fas(a, inds, rhs, add=False):
    """ 
    Fancy Assignment, equivalent to `a[*inds] = rhs` but with
    the elements of inds (allowed to be integers, slices, or 
    integer arrays) always specifing a generalize-slice along
    the given dimension.  This avoids some weird numpy indexing
    rules that make using square brackets a pain.
    """
    inds = tuple([slice(None) if (i is None) else i for i in inds])
    
    #Mixes of ints and tuples are fine, and a single
    # index-list index is fine too.  The case we need to
    # deal with is indexing a multi-dimensional array with
    # one or more index-lists
    if all([isinstance(i,(int,slice)) for i in inds]) or len(inds) == 1:
        if add:
            a[inds] += rhs #all integers or slices behave nicely
        else:
            a[inds] = rhs #all integers or slices behave nicely
    else:
        #convert each dimension's index to a list, take a product of
        # these lists, and flatten the right hand side to get the
        # proper assignment:
        b = []
        for ii,i in enumerate(inds):
            if isinstance(i,int): b.append([i])
            elif isinstance(i,slice):
                b.append( list(range(*i.indices(a.shape[ii]))) )
            else:
                b.append( list(i) )

        indx_tups = list(_itertools.product(*b))
        if len(indx_tups) > 0: # b/c a[()] just returns the entire array!
            inds = tuple(zip(*indx_tups)) # un-zips to one list per dim
            if add:
                a[inds] += rhs.flatten()
            else:
                a[inds] = rhs.flatten()
                
    return a

def _findx_shape(a, inds):
    """ Returns the shape of a fancy-indexed array (`a[*inds].shape`) """
    shape = []
    for ii,N in enumerate(a.shape):
        indx = inds[ii] if ii<len(inds) else None
        if indx is None: shape.append(N)
        elif isinstance(indx,slice):
            shape.append( len(range(*indx.indices(N))) )
        else: #assume indx is an index list or array
            shape.append( len(indx) )
    return shape


def _findx(a, inds, always_copy=False):
    """ 
    Fancy Indexing, equivalent to `a[*inds].copy()` but with
    the elements of inds (allowed to be integers, slices, or 
    integer arrays) always specifing a generalize-slice along
    the given dimension.  This avoids some weird numpy indexing
    rules that make using square brackets a pain.
    """
    inds = tuple([slice(None) if (i is None) else i for i in inds])
    
    #Mixes of ints and tuples are fine, and a single
    # index-list index is fine too.  The case we need to
    # deal with is indexing a multi-dimensional array with
    # one or more index-lists
    if all([isinstance(i,(int,slice)) for i in inds]) or len(inds) == 1:
        return a[inds].copy() if always_copy else a[inds] #all integers or slices behave nicely
    
    else:        
        #Need to copy to a new array
        b = []; squeeze = []
        for ii,i in enumerate(inds):
            if isinstance(i,int):
                b.append([i]); squeeze.append(ii) #squeeze ii-th dimension at end
            elif isinstance(i,slice):
                b.append( list(range(*i.indices(a.shape[ii]))) )
            else:
                b.append( list(i) )

        a_inds_shape = [len(x) for x in b]
        indx_tups = list(_itertools.product(*b))
        if len(indx_tups) > 0: # b/c a[()] just returns the entire array!
            inds = tuple(zip(*indx_tups)) # un-zips to one list per dim
            a_inds = a[inds].copy() # a 1D array of flattened "fancy" a[inds]
            a_inds.shape = a_inds_shape #reshape
        else:
            a_inds = _np.zeros( a_inds_shape, a.dtype ) #has zero elements
            asssert(a_inds.size == 0)
        return a_inds
