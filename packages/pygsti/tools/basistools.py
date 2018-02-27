""" Utility functions for working with Basis objects """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from functools    import partial
from itertools    import product

import numbers as _numbers
import collections as _collections

import numpy as _np

from ..baseobjs import Dim
from ..baseobjs import Basis

## Import base-object routines, which can act as "tools" too
## (note these are *not* imported by baseobjs.__init__.py)
from ..baseobjs.basisconstructors import *
from ..baseobjs.basis import basis_matrices, basis_longname, basis_element_labels

def is_sparse_basis(nameOrBasis):
    if isinstance(nameOrBasis, Basis):
        return nameOrBasis.sparse
    else: #assume everything else is not sparse
          # (could test for a sparse matrix list in the FUTURE)
        return False

def change_basis(mx, from_basis, to_basis, dimOrBlockDims=None, resize=None):
    """
    Convert a gate matrix from one basis of a density matrix space
    to another.

    Parameters
    ----------
    mx : numpy array
        The gate matrix (a 2D square array) in the `from_basis` basis.

    from_basis, to_basis: {'std', 'gm', 'pp', 'qt'} or Basis object
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mx operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mx.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the `to_basis` basis.
        Array size is the same as `mx`.
    """
    if len(mx.shape) not in (1, 2):
        raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")

    if dimOrBlockDims is None:
        dimOrBlockDims = int(round(_np.sqrt(mx.shape[0])))
        #assert( dimOrBlockDims**2 == mx.shape[0] )

    #if either basis is sparse, attempt to construct sparse bases
    try_for_sparse = is_sparse_basis(from_basis) or is_sparse_basis(to_basis)
        
    dim        = Dim(dimOrBlockDims)
    from_basis = Basis(from_basis, dim, sparse=try_for_sparse)
    to_basis   = Basis(to_basis, dim, sparse=try_for_sparse)
    #TODO: check for 'unknown' basis here and display meaningful warning - otherwise just get 0-dimensional basis...
        
    if from_basis.dim.gateDim != to_basis.dim.gateDim:
        raise ValueError('Automatic basis expanding/contracting is disabled: use flexible_change_basis')

    if from_basis.name == to_basis.name:
        return mx.copy()

    toMx   = from_basis.transform_matrix(to_basis)
    fromMx = to_basis.transform_matrix(from_basis)
    
    isMx = len(mx.shape) == 2 and mx.shape[0] == mx.shape[1]
    if isMx:
        # want ret = toMx.dot( _np.dot(mx, fromMx)) but need to deal
        # with some/all args being sparse:
        ret = _mt.safedot(toMx, _mt.safedot(mx, fromMx))
    else: # isVec
        ret = _mt.safedot(toMx, mx)

    if not to_basis.real:
        return ret

    if _mt.safenorm(ret,'imag') > 1e-8:
        raise ValueError("Array has non-zero imaginary part (%g) after basis change (%s to %s)!\n%s" %
                         (_mt.safenorm(ret,'imag'), from_basis, to_basis, ret))
    return _mt.safereal(ret)

def transform_matrix(from_basis, to_basis, dimOrBlockDims=None, sparse=False):
    '''
    Compute the transformation matrix between two bases

    Parameters
    ----------
    from_basis : Basis or str
        Basis being converted from

    to_basis : Basis or str
        Basis being converted to

    dimOrBlockDims : int or list of ints
        if strings provided as bases, the dimension of basis to use.

    sparse : bool, optional
        Whether to construct a sparse or dense transform matrix
        when this isn't specified already by `from_basis` or
        `to_basis` (e.g. when these are both strings).

    Returns
    -------
    Basis
        the composite basis created
    '''
    if dimOrBlockDims is None:
        assert isinstance(from_basis, Basis)
    else:
        from_basis = Basis(from_basis, dimOrBlockDims, sparse=sparse)
    return from_basis.transform_matrix(to_basis)


def build_basis_pair(mx, from_basis, to_basis):
    """
    Construct a pair of `Basis` objects with types `from_basis` and `to_basis`,
    and dimension appropriate for transforming `mx` (if they're not already
    given by `from_basis` or `to_basis` being a `Basis` rather than a `str`).

    Parameters
    ----------
    mx : numpy.ndarray
        A matrix, assumed to be square and have a dimension that is a perfect
        square.

    from_basis, to_basis: {'std', 'gm', 'pp', 'qt'} or Basis object
        The two bases (named as they are because usually they're the 
        source and destination basis for a basis change).  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt) (or a custom basis object).  If a custom basis 
        object is provided, it's dimension should be equal to
        `sqrt(mx.shape[0]) == sqrt(mx.shape[1])`.

    Returns
    -------
    from_basis, to_basis : Basis
    """
    if isinstance(from_basis, Basis) and not isinstance(to_basis, Basis):
        to_basis = from_basis.equivalent(to_basis)
    elif isinstance(to_basis, Basis) and not isinstance(from_basis, Basis):
        from_basis = to_basis.equivalent(from_basis)
    else: # neither or both from_basis & to_basis are Basis objects - so
          # no need to worry about setting sparse flag in construction
        dimOrBlockDims = int(round(_np.sqrt(mx.shape[0])))
        to_basis = Basis(to_basis, dimOrBlockDims)
        from_basis = Basis(from_basis, dimOrBlockDims)
    return from_basis, to_basis


def build_basis_for_matrix(mx, basis):
    """
    Construct a Basis object with type given by `basis` and dimension (if it's
    not given by `basis`) approprate for transforming `mx`, that is, equal to
    `sqrt(mx.shape[0])`.
    
    Parameters
    ----------
    mx : numpy.ndarray
        A matrix, assumed to be square and have a dimension that is a perfect
        square.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        A basis name or `Basis` object.  Allowed values are Matrix-unit (std),
        Gell-Mann (gm), Pauli-product (pp), and Qutrit (qt) (or a custom basis
        object).  If a custom basis object is provided, it's dimension must
        equal `sqrt(mx.shape[0])`, as this will be checked.

    Returns
    -------
    Basis
    """
    dimOrBlockDims = int(round(_np.sqrt(mx.shape[0])))
    return Basis(basis, dimOrBlockDims)



def resize_std_mx(mx, resize, stdBasis1, stdBasis2):
    """
    Change the basis of `mx`, which is assumed to be in the 'std'-type basis
    given by `stdBasis1`, to a potentially larger or smaller 'std'-type basis
    given by `stdBasis2`.

    This is possible when the two 'std'-type bases have the same "embedding
    dimension", equal to the sum of their block dimensions.  If, for example,
    `stdBasis1` has block dimensions (kite structure) of (4,2,1) then `mx`,
    expressed as a sum of `4^2 + 2^2 + 1^2 = 21` basis elements, can be 
    "embedded" within a larger 'std' basis having a single block with 
    dimension 7 (`7^2 = 49` elements).

    When `stdBasis2` is smaller than `stdBasis1` the reverse happens and `mx`
    is irreversibly truncated, or "contracted" to a basis having a particular
    kite structure.

    Parameters
    ----------
    mx : numpy array
        A square matrix in the `stdBasis1` basis.

    resize : {'expand','contract'}
        Whether `mx` can be expanded or contracted.
    
    stdBasis1 : Basis
        The 'std'-type basis that `mx` is currently in.

    stdBasis2 : Basis
        The 'std'-type basis that `mx` should be converted to.

    Returns
    -------
    numpy.ndarray
    """
    assert stdBasis1.dim.embedDim == stdBasis2.dim.embedDim
    if stdBasis1.dim.gateDim == stdBasis2.dim.gateDim:
        return mx
    #print('{}ing {} to {}'.format(resize, stdBasis1, stdBasis2))
    #print('Dims: ({} to {})'.format(stdBasis1.dim, stdBasis2.dim))
    if resize == 'expand':
        assert stdBasis1.dim.gateDim < stdBasis2.dim.gateDim
        right = _np.dot(mx, stdBasis1.get_expand_mx())
        mid   = _np.dot(stdBasis1.get_contract_mx(), right)
    elif resize == 'contract':
        assert stdBasis1.dim.gateDim > stdBasis2.dim.gateDim
        right = _np.dot(mx, stdBasis2.get_contract_mx())
        mid = _np.dot(stdBasis2.get_expand_mx(), right)
    return mid

def flexible_change_basis(mx, startBasis, endBasis):
    """ 
    Change `mx` from `startBasis` to `endBasis` allowing embedding expansion
    and contraction if needed (see :func:`resize_std_mx` for more details).

    Parameters
    ----------
    mx : numpy array
        The gate matrix (a 2D square array) in the `startBasis` basis.

    startBasis, endBasis : Basis
        The source and destination bases, respectively.

    Returns
    -------
    numpy.ndarray
    """
    if startBasis.dim.gateDim == endBasis.dim.gateDim:
        return change_basis(mx, startBasis, endBasis)
    if startBasis.dim.gateDim < endBasis.dim.gateDim:
        resize = 'expand'
    else:
        resize = 'contract'
    stdBasis1 = startBasis.std_equivalent()
    stdBasis2 = endBasis.std_equivalent()
    #start = change_basis(mx, startBasis, stdBasis1)
    mid   = resize_std_mx(mx, resize, stdBasis1, stdBasis2)
    end   = change_basis(mid, stdBasis2, endBasis)
    return end

def resize_mx(mx, dimOrBlockDims=None, resize=None):
    """ 
    Wrapper for :func:`resize_std_mx` that first constructs two 'std'-type bases
    using `dimOrBlockDims` and `sum(dimOrBlockDims)`.  The matrix `mx` is converted
    from the former to the latter when `resize == "expand"`, and from the latter to
    the former when `resize == "contract"`.

    Parameters
    ----------
    mx: numpy array
        Matrix of size N x N, where N is the dimension
        of the density matrix space, i.e. sum( dimOrBlockDims_i^2 )

    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    resize : {'expand','contract'}
        Whether `mx` should be expanded or contracted.

    Returns
    -------
    numpy.ndarray
    """
    #FUTURE: add a sparse flag?
    if dimOrBlockDims is None:
        return mx
    if resize == 'expand':
        a = Basis('std', dimOrBlockDims)
        b = Basis('std', sum(dimOrBlockDims))
    else:
        a = Basis('std', sum(dimOrBlockDims))
        b = Basis('std', dimOrBlockDims)
    return resize_std_mx(mx, resize, a, b)


def state_to_stdmx(state_vec):
    """
    Convert a state vector into a density matrix.

    Parameters
    ----------
    state_vec : list or tuple
       State vector in the standard (sigma-z) basis.

    Returns
    -------
    numpy.ndarray
        A density matrix of shape (d,d), corresponding to the pure state
        given by the length-`d` array, `state_vec`.
    """
    st_vec = state_vec.view(); st_vec.shape = (len(st_vec),1) #column vector
    dm_mx = _np.kron( _np.conjugate(_np.transpose(st_vec)), st_vec ) 
    return dm_mx #density matrix in standard (sigma-z) basis


def state_to_pauli_density_vec(state_vec):
    """
    Convert a single qubit state vector into a Liouville vector
    in the Pauli basis.

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
    return stdmx_to_ppvec(state_to_stdmx(state_vec))


def vec_to_stdmx(v, basis, keep_complex=False):
    """
    Convert a vector in this basis to
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
    ret = _np.zeros( (dim, dim), 'complex' )
    for i, mx in enumerate(basis._matrices):
        if keep_complex:
            ret += v[i]*mx
        else:
            ret += float(v[i])*mx
    return ret

gmvec_to_stdmx  = partial(vec_to_stdmx, basis='gm')
ppvec_to_stdmx  = partial(vec_to_stdmx, basis='pp')
qtvec_to_stdmx  = partial(vec_to_stdmx, basis='qt')
stdvec_to_stdmx = partial(vec_to_stdmx, basis='std')

from . import matrixtools as _mt

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
    v = _np.empty((dim**2,1))
    for i, mx in enumerate(basis._matrices):
        if basis.real:
            v[i,0] = _np.real(_mt.trace(_np.dot(mx,m)))
        else:
            v[i,0] = _mt.trace(_np.dot(mx,m))
    return v

stdmx_to_ppvec  = partial(stdmx_to_vec, basis='pp')
stdmx_to_gmvec  = partial(stdmx_to_vec, basis='gm')
stdmx_to_stdvec = partial(stdmx_to_vec, basis='std')


