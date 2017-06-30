from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************


"""
Functions for creating and converting between matrix bases.

There are three different bases that GST can use and convert between:
  - The Standard ("std") basis:
     State space is the tensor product of [0,1] for each qubit, e.g. for two qubits: ``[00,01,10,11] = [ |0>|0>, |0>|1>, ... ]``
     the gate space is thus the tensor product of two qubit spaces, so identical in form to state space
     for twice qubits, but interpret as ket/bra states.  E.g. for a *one* qubit gate, std basis is: = ``[ |0><0|, |0><1|, ... ]``

  - The Pauli-product ("pp") basis:
     Not used for state space - just for gates.  Basis consists of tensor products of the 4 pauli matrices (normalized by sqrt(2)).
     Examples:

     - 1-qubit gate basis is [ I, X, Y, Z ]  (in std basis, each is a pauli mx / sqrt(2))
     - 2-qubit gate basis is [ IxI, IxX, IxY, IxZ, XxI, ... ] (16 of them. In std basis, each is the tensor product of two pauli/sqrt(2) mxs)

  - The Gell-Mann ("gm") basis:
     Not used for state space - just for gates.  Basis consists of the Gell-Mann matrices of the given dimension (useful for dimensions that are not a power of 2)
     Examples:

     - 1-qubit gate basis is [ I, X, Y, Z ]  (in std basis, each is a pauli mx / sqrt(2)) -- SAME as Pauli-product!
     - 2-qubit gate basis is the 16 Gell-Mann matrices of dimension 4. In std basis, each is as given by Wikipedia page up to normalization.

Notes:
  - The elements of each basis are normalized so that Tr(Bi Bj) = delta_ij
  - since density matrices are Hermitian and all Gell-Mann and Pauli-product matrices are Hermitian too,
    gate parameterization by Gell-Mann or Pauli-product matrices have *real* coefficients, whereas
    in the standard basis gate matrices can have complex elements but these elements are additionally
    constrained.  This makes gate matrix parameterization and optimization much more convenient
    in the "gm" or "pp" bases.
"""
import functools    as _functools
import itertools    as _itertools
import numbers      as _numbers
import collections  as _collections
import numpy        as _np
import scipy.linalg as _spl

from . import matrixtools as _mt

from ..objects.basis import *
from ..objects.dim   import Dim

from .basisconstructors import *
from .basisconstructors import _mut

from .gatetools import unitary_to_process_mx, rotation_gate_mx

def basis_longname(basis, dimOrBlockDims=None):
    """
    Get the "long name" for a particular basis,
    which is typically used in reports, etc.

    Parameters
    ----------
    basis : {'std', 'gm', 'pp', 'qt'}
        Which basis the gateset is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp), and Qutrit (qt).

    dimOrBlockDims : int or list, optional
        Dimension of basis matrices, to aid in creating a
        long-name.  For example, a basis of the 2-dimensional
        Gell-Mann matrices is the same as the Pauli matrices,
        and thus the long name is just "Pauli" in this case.
        If a list of integers, then gives the dimensions of
        the terms in a direct-sum decomposition of the density
        matrix space acted on by the basis.
    Returns
    -------
    string
    """
    if basis in ['gm', 'pp'] and dimOrBlockDims in (2,[2],(2,)): 
        return "Pauli"
    else:
        if dimOrBlockDims is None:
            if basis == 'qt':
                dimOrBlockDims = 3
            else:
                dimOrBlockDims = 2
        basis = Basis(basis, dimOrBlockDims)
        return basis.longname

def basis_element_labels(basis, dimOrBlockDims, maxWeight=None):
    """
    Returns a list of short labels corresponding to to the
    elements of the described basis.  These labels are
    typically used to label the rows/columns of a box-plot
    of a matrix in the basis.

    Parameters
    ----------
    basis : {'std', 'gm', 'pp', 'qt'}
        Which basis the gateset is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp) and Qutrit (qt).  If the basis is
        not known, then an empty list is returned.

    dimOrBlockDims : int or list
        Dimension of basis matrices.  If a list of integers,
        then gives the dimensions of the terms in a
        direct-sum decomposition of the density
        matrix space acted on by the basis.

    maxWeight : int, optional
        Restrict the elements returned to those having weight <= `maxWeight`.
        (see :func:`basis_matrices`)


    Returns
    -------
    list of strings
        A list of length dim, whose elements label the basis
        elements.
    """

    if dimOrBlockDims == 1: #Special case of single element basis, in which
        return [ "" ]       # case we return a single label.

    #Note: the loops constructing the labels in this function
    # must be in-sync with those for constructing the matrices
    # in std_matrices, gm_matrices, and pp_matrices.
    _, _, blockDims = Dim(dimOrBlockDims)

    lblList = []; start = 0
    if basis == "std":
        for blockDim in blockDims:
            for i in range(start,start+blockDim):
                for j in range(start,start+blockDim):
                    lblList.append( "(%d,%d)" % (i,j) )
            start += blockDim

    elif basis == "gm":
        if dimOrBlockDims == 2: #Special case of Pauli's
            lblList = ["I","X","Y","Z"]

        else:
            for i,blockDim in enumerate(blockDims):
                d = blockDim

                #labels for gm_matrices of dim "blockDim":
                lblList.append("I^{(%d)}" % i) #identity on i-th block

                #X-like matrices, containing 1's on two off-diagonal elements (k,j) & (j,k)
                lblList.extend( [ "X^{(%d)}_{%d,%d}" % (i,k,j)
                                  for k in range(d) for j in range(k+1,d) ] )

                #Y-like matrices, containing -1j & 1j on two off-diagonal elements (k,j) & (j,k)
                lblList.extend( [ "Y^{(%d)}_{%d,%d}" % (i,k,j)
                                  for k in range(d) for j in range(k+1,d) ] )

                #Z-like matrices, diagonal mxs with 1's on diagonal until (k,k) element == 1-d,
                # then diagonal elements beyond (k,k) are zero.  This matrix is then scaled
                # by sqrt( 2.0 / (d*(d-1)) ) to ensure proper normalization.
                lblList.extend( [ "Z^{(%d)}_{%d}" % (i,k) for k in range(1,d) ] )


    elif basis == "pp":
        if dimOrBlockDims == 2: #Special case of Pauli's
            lblList = ["I","X","Y","Z"]

        else:
            #Some extra checking, since list-of-dims not supported for pp matrices yet.
            def is_integer(x):
                return bool( abs(x - round(x)) < 1e-6 )
            if not isinstance(dimOrBlockDims, _numbers.Integral):
                if (isinstance(dimOrBlockDims, _collections.Container)
                        and len(dimOrBlockDims) == 1):
                    dimOrBlockDims = dimOrBlockDims[0]
                else:
                    raise ValueError("Dimension for Pauli tensor product matrices must be an *integer* power of 2")
            nQubits = _np.log2(dimOrBlockDims)
            if not is_integer(nQubits):
                raise ValueError("Dimension for Pauli tensor product matrices must be an integer *power of 2*")
            nQubits = int(round(nQubits))

            basisLblList = [ ['I','X','Y','Z'] ]*nQubits
            for sigmaLbls in _itertools.product(*basisLblList):
                if maxWeight is not None:
                    if sigmaLbls.count('I') < nQubits-maxWeight: continue
                lblList.append( ''.join(sigmaLbls) )

    elif basis == "qt":
        assert(dimOrBlockDims == 3)
        lblList = ['II', 'X+Y', 'X-Y', 'YZ', 'IX', 'IY', 'IZ', 'XY', 'XZ']

    else:
        lblList = [] #Unknown basis

    return lblList

expand_from_std_direct_sum_mx = _functools.partial(resize_mx, resize='expand', startBasis='std', endBasis='std')
contract_to_std_direct_sum_mx = _functools.partial(resize_mx, resize='contract', startBasis='std', endBasis='std')

#TODO: maybe make these more general than for 1 or 2 qubits??
#############################################################################

def state_to_pauli_density_vec(state_vec):
    """
    Convert a single qubit state vector into a density matrix.

    Parameters
    ----------
    state_vec : list or tuple
       State vector in the sigma-z basis, len(state_vec) == 2

    Returns
    -------
    numpy array
        The 2x2 density matrix of the pure state given by state_vec, given
        as a 4x1 column vector in the Pauli basis.
    """
    assert( len(state_vec) == 2 )
    st_vec = _np.array( [ [state_vec[0]], [state_vec[1]] ] )
    dm_mx = _np.kron( _np.conjugate(_np.transpose(st_vec)), st_vec ) #density matrix in sigmaz basis
    return stdmx_to_ppvec(dm_mx)

def vec_to_stdmx(v, basis, keep_complex=False):
    """
    Convert a vector in any basis to
     a matrix in the standard basis.

    Parameters
    ----------
    v : numpy array
        The vector length 4 or 16 respectively.

    Returns
    -------
    numpy array
        The matrix, 2x2 or 4x4 depending on nqubits 
    """
    dim   = int(_np.sqrt( len(v) )) # len(v) = dim^2, where dim is matrix dimension of Pauli-prod mxs
    basis = Basis(basis, dim)
    return basis.vec_to_stdmx(v, keep_complex)

gmvec_to_stdmx  = _functools.partial(vec_to_stdmx, basis='gm')
ppvec_to_stdmx  = _functools.partial(vec_to_stdmx, basis='pp')
qtvec_to_stdmx  = _functools.partial(vec_to_stdmx, basis='qt')
stdvec_to_stdmx = _functools.partial(vec_to_stdmx, basis='std')

def stdmx_to_vec(m, basis):
    """
    Convert a matrix in the standard basis to
     a vector in the Pauli basis.

    Parameters
    ----------
    m : numpy array
        The matrix, shape 2x2 (1Q) or 4x4 (2Q)

    Returns
    -------
    numpy array
        The vector, length 4 or 16 respectively.
    """

    assert(len(m.shape) == 2 and m.shape[0] == m.shape[1])
    dim = m.shape[0]
    basis = Basis(basis, dim)
    return basis.stdmx_to_vec(m)

stdmx_to_ppvec = _functools.partial(stdmx_to_vec, basis='pp')
stdmx_to_gmvec = _functools.partial(stdmx_to_vec, basis='gm')
stdmx_to_stdvec = _functools.partial(stdmx_to_vec, basis='std')

