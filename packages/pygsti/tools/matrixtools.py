from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Matrix related utility functions """

import numpy as _np
import warnings as _warnings


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


def real_matrix_log(M, actionIfImaginary="raise",TOL=1e-8):
    """ 
    Construct a *real* logarithm of matrix `M`.

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
    numpy array
        An matrix `logM`, of the same shape as `M`, such that `M = exp(logM)`
        and `logM` is real (if possible).
    """
    assert(_np.linalg.norm(_np.imag(M)) < TOL) #M should be real to begin with
    evals,U = _np.linalg.eig(M)

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

    log_evals = _np.log(evals)
    
    for (i,j) in neg_real_pairs_real_evecs: #need to adjust evecs as well
        log_evals[i] = _np.log(evals[i])
        log_evals[j] = log_evals[i].conjugate()
        U[:,i] = (U[:,i] + 1j*U[:,j])/_np.sqrt(2)
        U[:,j] = U[:,i].conjugate()

    for (i,j) in neg_real_pairs_conj_evecs: # evecs already conjugates of each other
        log_evals[i] = _np.log(evals[i])
        log_evals[j] = log_evals[i].conjugate()

    logM =  _np.dot( U, _np.dot(_np.diag(log_evals), _np.linalg.inv(U) ))
    imMag = _np.linalg.norm(_np.imag(logM))
    #print("DB: log_evals = ",log_evals)
    #print("DEBUG: |imag(logM)| = ",imMag)

    #if there are unpaired negative real eigenvalues, the logarithm might be imaginary
    mayBeImaginary = bool(len(unpaired_indices) > 0)
    
    if mayBeImaginary and imMag > TOL:
        if actionIfImaginary == "raise":
            raise ValueError("Cannot construct a real log: unpaired negative" +
                         " real eigenvalues: %s" % [evals[i] for i in unpaired_indices])
        elif actionIfImaginary == "warn":
            _warnings.warn("Cannot construct a real log: unpaired negative" +
                         " real eigenvalues: %s" % [evals[i] for i in unpaired_indices])
        elif actionIfImaginary == "ignore":
            pass
        else:
            assert(False), "Invalid 'actionIfImaginary' argument: %s" % actionIfImaginary
    else: # logM should be real!
        assert( imMag < TOL ), "real_matrix_log failed to construct a real logarithm (when it should have!)"
        logM = _np.real(logM)
        
    return logM
