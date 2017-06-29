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

from .basisconstructors import *
from .basis import *
from .dim   import Dim

## Pauli basis matrices
sqrt2 = _np.sqrt(2)
id2x2 = _np.array([[1,0],[0,1]])
sigmax = _np.array([[0,1],[1,0]])
sigmay = _np.array([[0,-1.0j],[1.0j,0]])
sigmaz = _np.array([[1,0],[0,-1]])

sigmaii = _np.kron(id2x2,id2x2)
sigmaix = _np.kron(id2x2,sigmax)
sigmaiy = _np.kron(id2x2,sigmay)
sigmaiz = _np.kron(id2x2,sigmaz)
sigmaxi = _np.kron(sigmax,id2x2)
sigmaxx = _np.kron(sigmax,sigmax)
sigmaxy = _np.kron(sigmax,sigmay)
sigmaxz = _np.kron(sigmax,sigmaz)
sigmayi = _np.kron(sigmay,id2x2)
sigmayx = _np.kron(sigmay,sigmax)
sigmayy = _np.kron(sigmay,sigmay)
sigmayz = _np.kron(sigmay,sigmaz)
sigmazi = _np.kron(sigmaz,id2x2)
sigmazx = _np.kron(sigmaz,sigmax)
sigmazy = _np.kron(sigmaz,sigmay)
sigmazz = _np.kron(sigmaz,sigmaz)

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


expand_from_std_direct_sum_mx = _functools.partial(expand_from_direct_sum_mx, basis='std')
contract_to_std_direct_sum_mx = _functools.partial(contract_to_direct_sum_mx, basis='std')

def basis_matrices(basis, dimOrBlockDims, maxWeight=None):
    """
    Get the elements of the specifed basis-type which
    spans the density-matrix space given by dimOrBlockDims.

    Parameters
    ----------
    basis : {'std', 'gm', 'pp', 'qt'}
        The basis type.  Allowed values are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp), and Qutrit (qt).

    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    maxWeight : int, optional
      Restrict the elements returned to those having weight <= `maxWeight`. An
      element's "weight" is defined as the number of non-identity single-qubit
      factors of which it is comprised.  As this notion of factors is only 
      meaningful to the the Pauli-product basis, A non-None `maxWeight` can
      only be used when `basis == "pp"`.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (dmDim, dmDim),
        where dmDim is the matrix-dimension of the overall
        "embedding" density matrix (the sum of dimOrBlockDims)
        and N is the dimension of the density-matrix space,
        equal to sum( block_dim_i^2 ).
    """
    if maxWeight is None:
        return Basis(basis, dimOrBlockDims)
    else:
        return Basis(basis, dimOrBlockDims, maxWeight=maxWeight)
    

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


def unitary_to_pauligate_1q(U):
    """
    Get the linear operator on (vectorized) density
    matrices corresponding to a 1-qubit unitary
    operator on states.

    Parameters
    ----------
    U : numpy array
        A 2x2 array giving the action of the unitary
        on a state in the sigma-z basis.

    Returns
    -------
    numpy array
        The operator on density matrices that have been
        vectorized as length-4 vectors in the Pauli basis.
        Array has shape == (4,4).
    """
    assert( U.shape == (2,2) )
    op_mx = _np.empty( (4,4) ) #, 'complex' )
    Udag = _np.conjugate(_np.transpose(U))

    sigmaVec = pp_matrices(2)

    for i in (0,1,2,3):
        for j in (0,1,2,3):
            op_mx[i,j] = _np.real(_mt.trace(_np.dot(sigmaVec[i],_np.dot(U,_np.dot(sigmaVec[j],Udag)))))
        # in clearer notation: op_mx[i,j] = _mt.trace( sigma[i] * U * sigma[j] * Udag )
    return op_mx

# single qubit density matrix in 2-qubit pauli basis (16x16 matrix)
# U must be a 4x4 matrix
def unitary_to_pauligate_2q(U):
    """
    Get the linear operator on (vectorized) density
    matrices corresponding to a 2-qubit unitary
    operator on states.

    Parameters
    ----------
    U : numpy array
        A 4x4 array giving the action of the unitary
        on a state in the sigma-z basis.

    Returns
    -------
    numpy array
        The operator on density matrices that have been
        vectorized as length-16 vectors in the Pauli-product basis.
        Array has shape == (16,16).
    """

    assert( U.shape == (4,4) )
    op_mx = _np.empty( (16,16), 'd') #, 'complex' )
    Udag = _np.conjugate(_np.transpose(U))

    sigmaVec_2Q = pp_matrices(4)

    for i in range(16):
        for j in range(16):
            op_mx[i,j] = _np.real(_mt.trace(_np.dot(sigmaVec_2Q[i],_np.dot(U,_np.dot(sigmaVec_2Q[j],Udag)))))
        # in clearer notation: op_mx[i,j] = trace( sigma[i] * U * sigma[j] * Udag )
    return op_mx


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
    mxs = basis_matrices(basis, dim)

    ret = _np.zeros( (dim,dim), 'complex' )
    for i, mx in enumerate(mxs):
        if keep_complex:
            ret += v[i]*mx
        else:
            ret += float(v[i])*mx
    return ret
gmvec_to_stdmx = _functools.partial(vec_to_stdmx, basis='gm')
ppvec_to_stdmx = _functools.partial(vec_to_stdmx, basis='pp')
qtvec_to_stdmx = _functools.partial(vec_to_stdmx, basis='qt')
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
    mxs = basis_matrices(basis, dim)
    v = _np.empty((dim**2,1))
    for i, mx in enumerate(mxs):
        if mxs.real:
            v[i,0] = _np.real(_mt.trace(_np.dot(mx,m)))
        else:
            v[i,0] = _mt.trace(_np.dot(mx,m))
    return v

stdmx_to_ppvec = _functools.partial(stdmx_to_vec, basis='pp')
stdmx_to_gmvec = _functools.partial(stdmx_to_vec, basis='gm')
stdmx_to_stdvec = _functools.partial(stdmx_to_vec, basis='std')

def single_qubit_gate(hx, hy, hz, noise=0):
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
    ex = -1j * (hx*sigmax + hy*sigmay + hz*sigmaz)
    D = _np.diag( [1]+[1-noise]*(4-1) )
    return _np.dot(D, unitary_to_pauligate_1q( _spl.expm(ex) ))


def two_qubit_gate(ix=0, iy=0, iz=0, xi=0, xx=0, xy=0, xz=0, yi=0, yx=0, yy=0, yz=0, zi=0, zx=0, zy=0, zz=0, ii=0):
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

    ii : float, optional
        Coefficient of II matrix in exponent.

    Returns
    -------
    numpy array
        16x16 gate matrix which operates on a 2-qubit
        density matrix expressed as a vector in the
        Pauli-Product basis.
    """
    ex = ii * _np.identity(4, 'complex' )
    ex += ix * sigmaix
    ex += iy * sigmaiy
    ex += iz * sigmaiz
    ex += xi * sigmaxi
    ex += xx * sigmaxx
    ex += xy * sigmaxy
    ex += xz * sigmaxz
    ex += yi * sigmayi
    ex += yx * sigmayx
    ex += yy * sigmayy
    ex += yz * sigmayz
    ex += zi * sigmazi
    ex += zx * sigmazx
    ex += zy * sigmazy
    ex += zz * sigmazz
    return unitary_to_pauligate_2q( _spl.expm(-1j * ex) )
      #TODO: fix noise op to depolarizing
