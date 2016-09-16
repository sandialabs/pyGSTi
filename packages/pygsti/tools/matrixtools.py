from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Matrix related utility functions """

import numpy as _np
#import warnings


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
