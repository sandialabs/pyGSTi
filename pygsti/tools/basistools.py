"""
Utility functions for working with Basis objects
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from functools import partial, lru_cache

import numpy as _np

from pygsti.baseobjs.basisconstructors import _basis_constructor_dict
# from ..baseobjs.basis import Basis, BuiltinBasis, DirectSumBasis
from pygsti.baseobjs import basis as _basis

@lru_cache(maxsize=1)
def basis_matrices(name_or_basis, dim, sparse=False):
    """
    Get the elements of the specifed basis-type which spans the density-matrix space given by `dim`.

    Parameters
    ----------
    name_or_basis : {'std', 'gm', 'pp', 'qt'} or Basis
        The basis type.  Allowed values are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp), and Qutrit (qt).  If a Basis object, then the basis
        matrices are contained therein, and its dimension is checked to match
        `dim`.

    dim : int
        The dimension of the density-matrix space.

    sparse : bool, optional
        Whether any built matrices should be SciPy CSR sparse matrices
        or dense numpy arrays (the default).

    Returns
    -------
    list
        A list of N numpy arrays each of shape (dmDim, dmDim),
        where dmDim is the matrix-dimension of the overall
        "embedding" density matrix (the sum of dim_or_block_dims)
        and N is the dimension of the density-matrix space,
        equal to sum( block_dim_i^2 ).
    """
    return _basis.Basis.cast(name_or_basis, dim, sparse).elements


def basis_longname(basis):
    """
    Get the "long name" for a particular basis, which is typically used in reports, etc.

    Parameters
    ----------
    basis : Basis or str
        The basis or standard-basis-name.

    Returns
    -------
    string
    """
    if isinstance(basis, _basis.Basis):
        return basis.longname
    return _basis_constructor_dict[basis].longname


def basis_element_labels(basis, dim):
    """
    Get a list of short labels corresponding to to the elements of the described basis.

    These labels are typically used to label the rows/columns of a box-plot
    of a matrix in the basis.

    Parameters
    ----------
    basis : {'std', 'gm', 'pp', 'qt'}
        Which basis the model is represented in.  Allowed
        options are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp) and Qutrit (qt).  If the basis is
        not known, then an empty list is returned.

    dim : int or list
        Dimension of basis matrices.  If a list of integers,
        then gives the dimensions of the terms in a
        direct-sum decomposition of the density
        matrix space acted on by the basis.

    Returns
    -------
    list of strings
        A list of length dim, whose elements label the basis
        elements.
    """
    return _basis.Basis.cast(basis, dim).labels


def is_sparse_basis(name_or_basis):
    """
    Whether a basis contains sparse matrices.

    Parameters
    ----------
    name_or_basis : Basis or str
        The basis or standard-basis-name.

    Returns
    -------
    bool
    """
    if isinstance(name_or_basis, _basis.Basis):
        return name_or_basis.sparse
    else:  # assume everything else is not sparse
        # (could test for a sparse matrix list in the FUTURE)
        return False


def change_basis(mx, from_basis, to_basis, expect_real=True):
    """
    Convert a operation matrix from one basis of a density matrix space to another.

    Parameters
    ----------
    mx : numpy array
        The operation matrix (a 2D square array or 1D vector) in the `from_basis` basis.

    from_basis: {'std', 'gm', 'pp', 'qt'} or Basis object
        The source basis.  Allowed values are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp), and Qutrit (qt) (or a custom basis object).

    to_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The destination basis.  Allowed values are Matrix-unit (std), Gell-Mann
        (gm), Pauli-product (pp), and Qutrit (qt) (or a custom basis object).
    
    expect_real : bool, optional (default True)
        Optional flag specifying whether it is expected that the returned
        array in the new basis is real valued. Default is True.

    Returns
    -------
    numpy array
        The given operation matrix converted to the `to_basis` basis.
        Array size is the same as `mx`.
    """
    return bulk_change_basis([mx], from_basis, to_basis, expect_real)[0]

def bulk_change_basis(mxs, from_basis, to_basis, expect_real=True):
    """
    Convert a list of operation matrices from one basis of a density matrix space to another.

    Parameters
    ----------
    mxs : list of numpy arrays
        List of operation matrices (a 2D square array or 1D vector) in the `from_basis` basis.

    from_basis: {'std', 'gm', 'pp', 'qt'} or Basis object
        The source basis.  Allowed values are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp), and Qutrit (qt) (or a custom basis object).

    to_basis : {'std', 'gm', 'pp', 'qt'} or Basis object
        The destination basis.  Allowed values are Matrix-unit (std), Gell-Mann
        (gm), Pauli-product (pp), and Qutrit (qt) (or a custom basis object).
    
    expect_real : bool, optional (default True)
        Optional flag specifying whether it is expected that the returned
        array in the new basis is real valued. Default is True.

    Returns
    -------
    numpy array
        The given operation matrix converted to the `to_basis` basis.
        Array size is the same as `mx`.
    """
    for mx in mxs:
        if len(mx.shape) not in (1, 2):
            raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")

    #Build Basis objects from to_basis and from_basis as needed.
    from_is_basis = isinstance(from_basis, _basis.Basis)
    to_is_basis = isinstance(to_basis, _basis.Basis)
    dims = [mx.shape[0] for mx in mxs]
    assert all([dim == dims[0] for dim in dims]), 'All arrays must have the same shape'
    dim = dims[0]

    if not from_is_basis and not to_is_basis:
        #Case1: no Basis objects, so just construct builtin bases based on `mx` dim
        if from_basis == to_basis: return mx.copy()  # (shortcut)
        from_basis = _basis.BuiltinBasis(from_basis, dim, sparse=False)
        to_basis = _basis.BuiltinBasis(to_basis, dim, sparse=False)

    elif from_is_basis and to_is_basis:
        #Case2: both Basis objects.  Just make sure they agree :)
        assert(from_basis.dim == to_basis.dim == dim), \
            "Dimension mismatch: %d,%d,%d" % (from_basis.dim, to_basis.dim, dim)
    else:
        # If one is just a string, then use the .create_equivalent of the
        # other basis, since there can be desired structure (in the
        # other basis) that we want to preserve and which would be
        # lost if we just created a new BuiltinBasis with the correct
        # overall dimension.
        if from_is_basis:
            assert(from_basis.dim == dim), "src-basis dimension mismatch: %d != %d" % (from_basis.dim, dim)
            #to_basis = from_basis.create_equivalent(to_basis)
            # ^Don't to this b/c we take strings to always mean *simple* bases, not "equivalent" ones
            to_basis = _basis.BuiltinBasis(to_basis, dim, sparse=from_basis.sparse)
        else:
            assert(to_basis.dim == dim), "dest-basis dimension mismatch: %d != %d" % (to_basis.dim, dim)
            #from_basis = to_basis.create_equivalent(from_basis)
            from_basis = _basis.BuiltinBasis(from_basis, dim, sparse=to_basis.sparse)

    #TODO: check for 'unknown' basis here and display meaningful warning - otherwise just get 0-dimensional basis...

    if from_basis.dim != to_basis.dim:
        raise ValueError('Automatic basis expanding/contracting is disabled: use flexible_change_basis')

    if from_basis == to_basis:
        return [mx.copy() for mx in mxs]

    toMx = from_basis.create_transform_matrix(to_basis)
    fromMx = to_basis.create_transform_matrix(from_basis)

    isMx = len(mxs[0].shape) == 2 and mxs[0].shape[0] == mxs[0].shape[1]
    
    ret_mxs = []
    for mx in mxs:
        if isMx:
            # want ret = toMx.dot( _np.dot(mx, fromMx)) but need to deal
            # with some/all args being sparse:
            ret = toMx @ (mx @ fromMx)
        else:  # isVec
            ret = toMx @ mx

        if not to_basis.real:
            ret_mxs.append(ret)

        elif expect_real and _mt.safe_norm(ret, 'imag') > 1e-8:
            raise ValueError("Array has non-zero imaginary part (%g) after basis change (%s to %s)!\n%s" %
                            (_mt.safe_norm(ret, 'imag'), from_basis, to_basis, ret))
        else:
            ret_mxs.append(ret.real)

    return ret_mxs

def create_basis_pair(mx, from_basis, to_basis):
    """
    Constructs bases from transforming `mx` between two basis names.

    Construct a pair of `Basis` objects with types `from_basis` and `to_basis`,
    and dimension appropriate for transforming `mx` (if they're not already
    given by `from_basis` or `to_basis` being a `Basis` rather than a `str`).

    Parameters
    ----------
    mx : numpy.ndarray
        A matrix, assumed to be square and have a dimension that is a perfect
        square.

    from_basis: {'std', 'gm', 'pp', 'qt'} or Basis object
        The source basis (named because it's usually the source basis for a
        basis change).  Allowed values are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp), and Qutrit (qt) (or a custom basis object).  If a
        custom basis object is provided, it's dimension should be equal to
        `sqrt(mx.shape[0]) == sqrt(mx.shape[1])`.

    to_basis: {'std', 'gm', 'pp', 'qt'} or Basis object
        The destination basis (named because it's usually the destination basis
        for a basis change).  Allowed values are Matrix-unit (std), Gell-Mann
        (gm), Pauli-product (pp), and Qutrit (qt) (or a custom basis object).
        If a custom basis object is provided, it's dimension should be equal to
        `sqrt(mx.shape[0]) == sqrt(mx.shape[1])`.

    Returns
    -------
    from_basis, to_basis : Basis
    """
    dim = mx.shape[0]
    a = isinstance(from_basis, _basis.Basis)
    b = isinstance(to_basis, _basis.Basis)
    if a and b:
        pass  # no Basis creation needed
    elif a and not b:  # only from_basis is a Basis
        to_basis = from_basis.create_equivalent(to_basis)
    elif b and not a:  # only to_basis is a Basis
        from_basis = to_basis.create_equivalent(from_basis)
    else:  # neither ar Basis objects (assume they're strings)
        to_basis = _basis.BuiltinBasis(to_basis, dim)
        from_basis = _basis.BuiltinBasis(from_basis, dim)
    assert(from_basis.dim == to_basis.dim == dim), "Dimension mismatch!"
    return from_basis, to_basis


def create_basis_for_matrix(mx, basis):
    """
    Construct a Basis object with type given by `basis` and dimension approprate for transforming `mx`.

    Dimension is taken from `mx` (if it's not given by `basis`) that is `sqrt(mx.shape[0])`.

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
    dim = mx.shape[0]
    if isinstance(basis, _basis.Basis):
        assert(basis.dim == dim), "Supplied Basis has wrong dimension!"
        return basis
    else:  # assume basis is a string name of a builtin basis
        return _basis.BuiltinBasis(basis, dim)


def resize_std_mx(mx, resize, std_basis_1, std_basis_2):
    """
    Change the basis of `mx` to a potentially larger or smaller 'std'-type basis given by `std_basis_2`.

    (`mx` is assumed to be in the 'std'-type basis given by `std_basis_1`.)

    This is possible when the two 'std'-type bases have the same "embedding
    dimension", equal to the sum of their block dimensions.  If, for example,
    `std_basis_1` has block dimensions (kite structure) of (4,2,1) then `mx`,
    expressed as a sum of `4^2 + 2^2 + 1^2 = 21` basis elements, can be
    "embedded" within a larger 'std' basis having a single block with
    dimension 7 (`7^2 = 49` elements).

    When `std_basis_2` is smaller than `std_basis_1` the reverse happens and `mx`
    is irreversibly truncated, or "contracted" to a basis having a particular
    kite structure.

    Parameters
    ----------
    mx : numpy array
        A square matrix in the `std_basis_1` basis.

    resize : {'expand','contract'}
        Whether `mx` can be expanded or contracted.

    std_basis_1 : Basis
        The 'std'-type basis that `mx` is currently in.

    std_basis_2 : Basis
        The 'std'-type basis that `mx` should be converted to.

    Returns
    -------
    numpy.ndarray
    """
    assert(std_basis_1.elsize == std_basis_2.elsize), '"embedded" space dimensions differ!'
    if std_basis_1.dim == std_basis_2.dim:
        return change_basis(mx, std_basis_1, std_basis_2)  # don't just 'return mx' here
        # - need to change bases if bases are different (e.g. if one is a Tensorprod of std components)

    #print('{}ing {} to {}'.format(resize, std_basis_1, std_basis_2))
    #print('Dims: ({} to {})'.format(std_basis_1.dim, std_basis_2.dim))
    #Below: use 'exp' in comments for 'expanded dimension'
    if resize == 'expand':
        assert std_basis_1.dim < std_basis_2.dim
        right = _np.dot(mx, std_basis_1.from_elementstd_transform_matrix)  # (exp,dim) (dim,dim) (dim,exp) => exp,exp
        mid = _np.dot(std_basis_1.to_elementstd_transform_matrix, right)  # want Ai st.   Ai * A = I(dim)
    elif resize == 'contract':
        assert std_basis_1.dim > std_basis_2.dim
        right = _np.dot(mx, std_basis_2.to_elementstd_transform_matrix)  # (dim,dim) (dim,exp) => dim,exp
        mid = _np.dot(std_basis_2.from_elementstd_transform_matrix, right)  # (dim, exp) (exp, dim) => expdim, exp
    return mid


def flexible_change_basis(mx, start_basis, end_basis):
    """
    Change `mx` from `start_basis` to `end_basis` allowing embedding expansion and contraction if needed.

    (see :func:`resize_std_mx` for more details).

    Parameters
    ----------
    mx : numpy array
        The operation matrix (a 2D square array) in the `start_basis` basis.

    start_basis : Basis
        The source basis.

    end_basis : Basis
        The destination basis.

    Returns
    -------
    numpy.ndarray
    """
    if start_basis.dim == end_basis.dim:  # normal case
        return change_basis(mx, start_basis, end_basis)
    if start_basis.dim < end_basis.dim:
        resize = 'expand'
    else:
        resize = 'contract'
    stdBasis1 = start_basis.create_equivalent('std')
    stdBasis2 = end_basis.create_equivalent('std')
    #start = change_basis(mx, start_basis, stdBasis1)
    mid = resize_std_mx(mx, resize, stdBasis1, stdBasis2)
    end = change_basis(mid, stdBasis2, end_basis)
    return end


def resize_mx(mx, dim_or_block_dims=None, resize=None):
    """
    Wrapper for :func:`resize_std_mx`, that manipulates `mx` to be in another basis.

    This function first constructs two 'std'-type bases using
    `dim_or_block_dims` and `sum(dim_or_block_dims)`.  The matrix `mx` is
    converted from the former to the latter when `resize == "expand"`, and from
    the latter to the former when `resize == "contract"`.

    Parameters
    ----------
    mx : numpy array
        Matrix of size N x N, where N is the dimension
        of the density matrix space, i.e. sum( dimOrBlockDims_i^2 )

    dim_or_block_dims : int or list of ints
        Structure of the density-matrix space.  Gives the *matrix*
        dimensions of each block.

    resize : {'expand','contract'}
        Whether `mx` should be expanded or contracted.

    Returns
    -------
    numpy.ndarray
    """
    #FUTURE: add a sparse flag?
    if dim_or_block_dims is None:
        return mx
    blkBasis = _basis.DirectSumBasis([_basis.BuiltinBasis('std', d**2) for d in dim_or_block_dims])
    simpleBasis = _basis.BuiltinBasis('std', sum(dim_or_block_dims)**2)

    if resize == 'expand':
        a = blkBasis
        b = simpleBasis
    else:
        a = simpleBasis
        b = blkBasis
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
    st_vec = state_vec.view(); st_vec.shape = (len(st_vec), 1)  # column vector
    dm_mx = _np.kron(_np.conjugate(_np.transpose(st_vec)), st_vec)
    return dm_mx  # density matrix in standard (sigma-z) basis


def state_to_pauli_density_vec(state_vec):
    """
    Convert a single qubit state vector into a Liouville vector in the Pauli basis.

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
    assert(len(state_vec) == 2)
    return stdmx_to_ppvec(state_to_stdmx(state_vec))


def vec_to_stdmx(v, basis, keep_complex=False):
    """
    Convert a vector in this basis to a matrix in the standard basis.

    Parameters
    ----------
    v : numpy array
        The vector length 4 or 16 respectively.

    basis : {'std', 'gm', 'pp', 'qt'} or Basis
        The basis type.  Allowed values are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp), and Qutrit (qt).  If a Basis object, then the basis
        matrices are contained therein, and its dimension is checked to match `v`.

    keep_complex : bool, optional
        If True, leave the final (output) array elements as complex numbers when
        `v` is complex.  Usually, the final elements are real (even though `v` is
        complex) and so when `keep_complex=False` the elements are forced to be real
        and the returned array is float (not complex) valued.

    Returns
    -------
    numpy array
        The matrix, 2x2 or 4x4 depending on nqubits
    """
    if not isinstance(basis, _basis.Basis):
        basis = _basis.BuiltinBasis(basis, len(v))
    v = v.ravel()
    ret = _np.zeros(basis.elshape, 'complex')
    if v.ndim > 1:
        assert v.size == v.shape[0]
        v = v.ravel()
    for i, mx in enumerate(basis.elements):
        if keep_complex:
            ret += v[i] * mx
        else:
            ret += float(v[i]) * mx
    return ret


gmvec_to_stdmx = partial(vec_to_stdmx, basis='gm')
ppvec_to_stdmx = partial(vec_to_stdmx, basis='pp')
qtvec_to_stdmx = partial(vec_to_stdmx, basis='qt')
stdvec_to_stdmx = partial(vec_to_stdmx, basis='std')

from . import matrixtools as _mt


def stdmx_to_vec(m, basis):
    """
    Convert a matrix in the standard basis to a vector in the Pauli basis.

    Parameters
    ----------
    m : numpy array
        The matrix, shape 2x2 (1Q) or 4x4 (2Q)

    basis : {'std', 'gm', 'pp', 'qt'} or Basis
        The basis type.  Allowed values are Matrix-unit (std), Gell-Mann (gm),
        Pauli-product (pp), and Qutrit (qt).  If a Basis object, then the basis
        matrices are contained therein, and its dimension is checked to match `m`.

    Returns
    -------
    numpy array
        The vector, length 4 or 16 respectively.
    """

    assert(len(m.shape) == 2 and m.shape[0] == m.shape[1])
    basis = _basis.Basis.cast(basis, m.shape[0]**2)
    v = _np.empty((basis.size, 1))
    for i, mx in enumerate(basis.elements):
        if basis.real:
            v[i, 0] = _np.real(_np.vdot(mx, m))
        else:
            v[i, 0] = _np.real_if_close(_np.vdot(mx, m))
    return v


stdmx_to_ppvec = partial(stdmx_to_vec, basis='pp')
stdmx_to_gmvec = partial(stdmx_to_vec, basis='gm')
stdmx_to_stdvec = partial(stdmx_to_vec, basis='std')
