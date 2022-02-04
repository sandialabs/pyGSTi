"""
Functions for creating the standard sets of matrices in the standard, Pauli, Gell-Mann, and qutrit bases
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
import itertools as _itertools
import numbers as _numbers
import numpy as _np
import scipy.sparse as _sps
from functools import partial as _partial

## Pauli basis matrices
sqrt2 = _np.sqrt(2)
id2x2 = _np.array([[1, 0], [0, 1]])
sigmax = _np.array([[0, 1], [1, 0]])
sigmay = _np.array([[0, -1.0j], [1.0j, 0]])
sigmaz = _np.array([[1, 0], [0, -1]])


##Matrix unit basis
def mut(i, j, n):
    """
    A matrix unit.

    Parameters
    ----------
    i : int
        Row of the single "1" in the matrix unit.

    j : int
        Column of the single "1" in the matrix unit.

    n : int
        Dimension of matrix

    Returns
    -------
    numpy.ndarray
        A `(n,n)`-shaped array that is all zeros except a single "1"
        in the `i`,`j` element.
    """
    mx = _np.zeros((n, n), 'd'); mx[i, j] = 1.0
    return mx


MX_UNIT_VEC = (mut(0, 0, 2), mut(0, 1, 2), mut(1, 0, 2), mut(1, 1, 2))
MX_UNIT_VEC_2Q = (mut(0, 0, 4), mut(0, 1, 4), mut(0, 2, 4), mut(0, 3, 4),
                  mut(1, 0, 4), mut(1, 1, 4), mut(1, 2, 4), mut(1, 3, 4),
                  mut(2, 0, 4), mut(2, 1, 4), mut(2, 2, 4), mut(2, 3, 4),
                  mut(3, 0, 4), mut(3, 1, 4), mut(3, 2, 4), mut(3, 3, 4))

MAX_BASIS_MATRIX_DIM = 2**6


def _check_dim(dim):
    global MAX_BASIS_MATRIX_DIM
    if not isinstance(dim, _numbers.Integral):
        dim = max(dim)  # assume dim is a list/tuple of dims & just consider max
    if dim > MAX_BASIS_MATRIX_DIM:
        raise ValueError(("You have requested to build a basis with %d x %d matrices."
                          " This is pretty big and so we're throwing this error because"
                          " there's a good chance you didn't mean to to this.  If you "
                          " really want to, increase `pygsti.tools.basisconstructors.MAX_BASIS_MATRIX_DIM`"
                          " (currently == %d) to something greater than %d and rerun this.")
                         % (dim, dim, MAX_BASIS_MATRIX_DIM, dim))


class MatrixBasisConstructor(object):
    """
    A factory class for constructing builtin basis types whose elements are matrices.

    Parameters
    ----------
    longname : str
        The long name for the builtin basis.

    matrixgen_fn : function
        A function that generates the matrix elements for this
        basis given the matrix dimension (i.e. the number of rows or
        columns in the matrices to produce).

    labelgen_fn : function
        A function that generates the element labels for this
        basis given the matrix dimension (i.e. the number of rows or
        columns in the matrices to produce).

    real : bool
        Whether vectors expressed in this basis are required to have
        real components.
    """

    def __init__(self, longname, matrixgen_fn, labelgen_fn, real):
        """
        Create a new MatrixBasisConstructor:

        Parameters
        ----------
        longname : str
            The long name for the builtin basis.

        matrixgen_fn : function
            A function that generates the matrix elements for this
            basis given the matrix dimension (i.e. the number of rows or
            columns in the matrices to produce).

        labelgen_fn : function
            A function that generates the element labels for this
            basis given the matrix dimension (i.e. the number of rows or
            columns in the matrices to produce).

        real : bool
            Whether vectors expressed in this basis are required to have
            real components.
        """
        self.matrixgen_fn = matrixgen_fn
        self.labelgen_fn = labelgen_fn
        self.longname = longname
        self.real = real

    def matrix_dim(self, dim):
        """
        Helper function that converts a *vector-space* dimension `dim` to matrix-dimension by taking a sqrt.

        Parameters
        ----------
        dim : int
            Dimension

        Returns
        -------
        int
        """
        d = int(round(_np.sqrt(dim)))
        assert(d**2 == dim), "Matrix bases can only have dimension = perfect square (not %d)!" % dim
        return d

    def labeler(self, dim, sparse):
        """
        Get the labels of a basis to be constructed.

        Parameters
        ----------
        dim : int
            The *vector-space* dimension of the basis.

        sparse : bool
            Whether the basis is sparse or not.

        Returns
        -------
        list of labels (strs)
        """
        return self.labelgen_fn(self.matrix_dim(dim))

    def constructor(self, dim, sparse):
        """
        Get the elements of a basis to be constructed.

        Parameters
        ----------
        dim : int
            The *vector-space* dimension of the basis.

        sparse : bool
            Whether the basis is sparse or not.

        Returns
        -------
        list of basis elements
        """
        els = self.matrixgen_fn(self.matrix_dim(dim))
        if sparse: els = [_sps.csr_matrix(el) for el in els]
        return els

    def sizes(self, dim, sparse):
        """
        Get some relevant sizes/dimensions for constructing a basis.

        This function is needed for constructing Basis objects
        because these objects want to know the size & dimension of
        a basis without having to construct the (potentially
        large) set of elements.

        Parameters
        ----------
        dim : int
            The *vector-space* dimension of the basis.
            e.g. 4 for a basis of 2x2 matrices and 2 for
            a basis of length=2 vectors.

        sparse : bool
            Whether the basis is sparse or not.

        Returns
        -------
        nElements : int
            The number of elements in the basis.
        dim : int
            The vector-space dimension of the basis.
        elshape : tuple
            The shape of the elements that might be
            constructed (if `constructor` was called).
        """
        nElements = dim  # the number of matrices in the basis
        basisDim = dim  # the dimension of the vector space this basis is for
        # (== size for a full basis, > size for a partial basis)
        d = self.matrix_dim(dim); elshape = (d, d)
        return nElements, basisDim, elshape


class DiagonalMatrixBasisConstructor(MatrixBasisConstructor):
    """
    A factory class for constructing builtin basis types whose elements are diagonal matrices.

    The size of these bases is equal to their matrix dimension (so dim == matrix_dim, similar to
    a VectorBasisConstructor, but element are diagonal matrices rather than vectors)
    """

    def constructor(self, dim, sparse):
        """
        Get the elements of a basis to be constructed.

        Parameters
        ----------
        dim : int
            The *vector-space* dimension of the basis.

        sparse : bool
            Whether the basis is sparse or not.

        Returns
        -------
        list of basis elements
        """
        dtype = 'd' if self.real else 'complex'
        d = self.matrix_dim(dim)
        vectorgen_fn = self.matrixgen_fn  # matrixgen really just construct vectors
        els = [_np.array(_np.diag(v), dtype) for v in vectorgen_fn(d)]
        if sparse: els = [_sps.csr_matrix(el) for el in els]
        return els

    def sizes(self, dim, sparse):
        """
        Get some relevant sizes/dimensions for constructing a basis.

        This function is needed for constructing Basis objects
        because these objects want to know the size & dimension of
        a basis without having to construct the (potentially
        large) set of elements.

        Parameters
        ----------
        dim : int
            The *vector-space* dimension of the basis.
            e.g. 4 for a basis of 2x2 matrices and 2 for
            a basis of length=2 vectors.

        sparse : bool
            Whether the basis is sparse or not.

        Returns
        -------
        nElements : int
            The number of elements in the basis.
        dim : int
            The vector-space dimension of the basis.
        elshape : tuple
            The shape of the elements that might be
            constructed (if `constructor` was called).
        """
        d = self.matrix_dim(dim); elshape = (d, d)
        nElements = d  # the number of matrices in the basis
        basisDim = dim  # the dimension of the vector space this basis
        return nElements, basisDim, elshape


class SingleElementMatrixBasisConstructor(MatrixBasisConstructor):
    """
    A constructor for a basis containing just a single element (e.g. the identity).
    """

    def sizes(self, dim, sparse):
        """
        See docstring for :class:`MatrixBasisConstructor`

        Parameters
        ----------
        dim : int
            The *vector-space* dimension of the basis.
            e.g. 4 for a basis of 2x2 matrices and 2 for
            a basis of length=2 vectors.

        sparse : bool
            Whether the basis is sparse or not.

        Returns
        -------
        nElements : int
            The number of elements in the basis.
        dim : int
            The vector-space dimension of the basis.
        elshape : tuple
            The shape of the elements that might be
            constructed (if `constructor` was called).
        """
        nElements = 1   # the number of matrices in the basis
        basisDim = dim  # the dimension of the vector space this basis is for
        # (== size for a full basis, > size for a partial basis)
        d = self.matrix_dim(dim); elshape = (d, d)
        return nElements, basisDim, elshape


class VectorBasisConstructor(object):
    """
    A factory class for constructing builtin basis types whose elements are vectors.

    Parameters
    ----------
    longname : str
        The long name for the builtin basis.

    vectorgen_fn : function
        A function that generates the vector elements for this
        basis given the vector dimension.

    labelgen_fn : function
        A function that generates the element labels for this
        basis given the vector dimension.

    real : bool
        Whether vectors expressed in this basis are required to have
        real components.
    """

    def __init__(self, longname, vectorgen_fn, labelgen_fn, real):
        """
        Create a new VectorBasisConstructor:

        Parameters
        ----------
        longname : str
            The long name for the builtin basis.

        vectorgen_fn : function
            A function that generates the vector elements for this
            basis given the vector dimension.

        labelgen_fn : function
            A function that generates the element labels for this
            basis given the vector dimension.

        real : bool
            Whether vectors expressed in this basis are required to have
            real components.
        """
        self.vectorgen_fn = vectorgen_fn
        self.labelgen_fn = labelgen_fn
        self.longname = longname
        self.real = real

    def labeler(self, dim, sparse):
        """
        Get the labels of a basis to be constructed.

        Parameters
        ----------
        dim : int
            The *vector-space* dimension of the basis.

        sparse : bool
            Whether the basis is sparse or not.

        Returns
        -------
        list of labels (strs)
        """
        return self.labelgen_fn(dim)

    def constructor(self, dim, sparse):
        """
        Get the elements of a basis to be constructed.

        Parameters
        ----------
        dim : int
            The *vector-space* dimension of the basis.

        sparse : bool
            Whether the basis is sparse or not.

        Returns
        -------
        list of basis elements
        """
        els = self.vectorgen_fn(dim)
        assert(not sparse), "Sparse vector bases not supported (yet)"
        return els

    def sizes(self, dim, sparse):
        """
        Get some relevant sizes/dimensions for constructing a basis.

        This function is needed for constructing Basis objects
        because these objects want to know the size & dimension of
        a basis without having to construct the (potentially
        large) set of elements.

        Parameters
        ----------
        dim : int
            The *vector-space* dimension of the basis.
            e.g. 4 for a basis of 2x2 matrices and 2 for
            a basis of length=2 vectors.

        sparse : bool
            Whether the basis is sparse or not.

        Returns
        -------
        nElements : int
            The number of elements in the basis.
        dim : int
            The vector-space dimension of the basis.
        elshape : tuple
            The shape of the elements that might be
            constructed (if `constructor` was called).
        """
        nElements = dim  # the number of matrices in the basis
        basisDim = dim  # the dimension of the vector space this basis
        elshape = (dim,)  # the shape of the (vector) elements
        return nElements, basisDim, elshape


def std_matrices(matrix_dim):
    """
    Get the elements of the matrix unit, or "standard", basis of matrix-dimension `matrix_dim`.
    The matrices are ordered so that the row index changes the fastest.

    Constructs the standard basis spanning the density-matrix space given by
    `matrix_dim` x `matrix_dim` matrices.

    The returned matrices are orthonormal basis under
    the trace inner product, i.e. Tr( dot(Mi,Mj) ) == delta_ij.

    Parameters
    ----------
    matrix_dim : int
        matrix dimension of the density-matrix space, e.g. 2
        for a single qubit in a 2x2 density matrix basis.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (matrix_dim, matrix_dim).

    Notes
    -----
    Each element is a matrix containing
    a single "1" entry amidst a background of zeros.
    """
    _check_dim(matrix_dim)
    basisDim = matrix_dim ** 2

    mxList = []
    for i in range(matrix_dim):
        for j in range(matrix_dim):
            mxList.append(mut(i, j, matrix_dim))
    assert len(mxList) == basisDim
    return mxList


def std_labels(matrix_dim):
    """
    Return the standard-matrix-basis labels based on a matrix dimension.

    Parameters
    ----------
    matrix_dim : int
        The matrix dimension of the basis to generate labels for (the
        number of rows or columns in a matrix).

    Returns
    -------
    list of strs
    """
    if matrix_dim == 0: return []
    if matrix_dim == 1: return ['']  # special case - use empty label instead of "I"
    return ["(%d,%d)" % (i, j) for i in range(matrix_dim) for j in range(matrix_dim)]


def col_matrices(matrix_dim):
    """
    Get the elements of the matrix unit, or "column-stacked", basis of matrix-dimension `matrix_dim`.
    The matrices are ordered so that the column index changes the fastest.

    Constructs the standard basis spanning the density-matrix space given by
    `matrix_dim` x `matrix_dim` matrices.

    The returned matrices are orthonormal basis under
    the trace inner product, i.e. Tr( dot(Mi,Mj) ) == delta_ij.

    Parameters
    ----------
    matrix_dim : int
        matrix dimension of the density-matrix space, e.g. 2
        for a single qubit in a 2x2 density matrix basis.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (matrix_dim, matrix_dim).

    Notes
    -----
    Each element is a matrix containing
    a single "1" entry amidst a background of zeros.
    """
    _check_dim(matrix_dim)
    basisDim = matrix_dim ** 2

    mxList = []
    for row_index in range(matrix_dim):
        for col_index in range(matrix_dim):
            mxList.append(mut(col_index, row_index, matrix_dim))
    assert len(mxList) == basisDim
    return mxList


def col_labels(matrix_dim):
    """
    Return the column-stacked-matrix-basis labels based on a matrix dimension.

    Parameters
    ----------
    matrix_dim : int
        The matrix dimension of the basis to generate labels for (the
        number of rows or columns in a matrix).

    Returns
    -------
    list of strs
    """
    if matrix_dim == 0: return []
    if matrix_dim == 1: return ['']  # special case - use empty label instead of "I"
    return ["(%d,%d)" % (j, i) for i in range(matrix_dim) for j in range(matrix_dim)]


def _get_gell_mann_non_identity_diag_mxs(dimension):
    d = dimension
    listOfMxs = []
    if d > 2:
        dm1_listOfMxs = _get_gell_mann_non_identity_diag_mxs(d - 1)
        for dm1_mx in dm1_listOfMxs:
            mx = _np.zeros((d, d), 'complex')
            mx[0:d - 1, 0:d - 1] = dm1_mx
            listOfMxs.append(mx)
    if d > 1:
        mx = _np.identity(d, 'complex')
        mx[d - 1, d - 1] = 1 - d
        mx *= _np.sqrt(2.0 / (d * (d - 1)))
        listOfMxs.append(mx)

    return listOfMxs


def gm_matrices_unnormalized(matrix_dim):
    """
    Get the elements of the generalized Gell-Mann basis spanning the density-matrix space given by matrix_dim.

    The returned matrices are given in the standard basis of the
    "embedding" density matrix space, that is, the space which
    embeds the block-diagonal matrix structure stipulated in
    dim. These matrices form an orthogonal but not
    orthonormal basis under the trace inner product.

    Parameters
    ----------
    matrix_dim : int
        Dimension of the density-matrix space.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (matrix_dim, matrix_dim),
        where matrix_dim is the matrix-dimension of the overall
        "embedding" density matrix (the sum of matrix_dim)
        and N is the dimension of the density-matrix space,
        equal to sum( block_dim_i^2 ).
    """
    _check_dim(matrix_dim)
    if matrix_dim == 0: return []
    if isinstance(matrix_dim, _numbers.Integral):
        d = matrix_dim
        #Identity Mx
        listOfMxs = [_np.identity(d, 'complex')]

        #Non-diagonal matrices -- only take those whose non-zero elements are not "frozen" in cssb case
        for k in range(d):
            for j in range(k + 1, d):
                mx = _np.zeros((d, d), 'complex')
                mx[k, j] = mx[j, k] = 1.0
                listOfMxs.append(mx)

        for k in range(d):
            for j in range(k + 1, d):
                mx = _np.zeros((d, d), 'complex')
                mx[k, j] = -1.0j; mx[j, k] = 1.0j
                listOfMxs.append(mx)

        #Non-Id Diagonal matrices
        listOfMxs.extend(_get_gell_mann_non_identity_diag_mxs(d))

        assert(len(listOfMxs) == d**2)
        return listOfMxs
    else:
        raise ValueError("Invalid matrix_dim = %s" % str(matrix_dim))


def gm_matrices(matrix_dim):
    """
    Get the normalized elements of the generalized Gell-Mann basis with matrix dimension `matrix_dim`.

    That is, construct the basis spanning the density-matrix space given by `matrix_dim`.

    The returned matrices are given in the standard basis of the
    "embedding" density matrix space, that is, the space which
    embeds the block-diagonal matrix structure stipulated in
    matrix_dim. These matrices form an orthonormal basis
    under the trace inner product, i.e. Tr( dot(Mi,Mj) ) == delta_ij.

    Parameters
    ----------
    matrix_dim : int
        Dimension of the density-matrix space.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (matrix_dim, matrix_dim),
        where matrix_dim is the matrix-dimension of the overall
        "embedding" density matrix (the sum of matrix_dim)
        and N is the dimension of the density-matrix space,
        equal to sum( block_dim_i^2 ).
    """
    mxs = [mx.copy() for mx in gm_matrices_unnormalized(matrix_dim)]
    for mx in mxs:
        mx.flags.writeable = True  # Safe because of above copy
    mxs[0] *= 1 / _np.sqrt(mxs[0].shape[0])  # identity mx
    for mx in mxs[1:]:
        mx *= 1 / sqrt2
    return mxs


def gm_labels(matrix_dim):
    """
    Gell-Mann basis labels.

    Parameters
    ----------
    matrix_dim : int
        The labels (names) of the Gell-Mann basis elements
        with a given matrix dimension.

    Returns
    -------
    list
    """
    if matrix_dim == 0: return []
    if matrix_dim == 1: return ['']  # special case - use empty label instead of "I"
    if matrix_dim == 2:  # Special case of Pauli's
        return ["I", "X", "Y", "Z"]

    d = matrix_dim
    lblList = []

    #labels for gm_matrices of dim "blockDim":
    lblList.append("I")  # identity on i-th block

    #X-like matrices, containing 1's on two off-diagonal elements (k,j) & (j,k)
    lblList.extend(["X_{%d,%d}" % (k, j)
                    for k in range(d) for j in range(k + 1, d)])

    #Y-like matrices, containing -1j & 1j on two off-diagonal elements (k,j) & (j,k)
    lblList.extend(["Y_{%d,%d}" % (k, j)
                    for k in range(d) for j in range(k + 1, d)])

    #Z-like matrices, diagonal mxs with 1's on diagonal until (k,k) element == 1-d,
    # then diagonal elements beyond (k,k) are zero.  This matrix is then scaled
    # by sqrt( 2.0 / (d*(d-1)) ) to ensure proper normalization.
    lblList.extend(["Z_{%d}" % (k) for k in range(1, d)])
    return lblList


def qsim_matrices(matrix_dim):
    """
    Get the elements of the QuantumSim basis with matrix dimension `matrix_dim`.

    These matrices span the space of matrix_dim x matrix_dim density matrices
    (matrix-dimension matrix_dim, space dimension matrix_dim^2).

    The returned matrices are given in the QuantumSim representation of the
    density matrix space, and are thus kronecker products of
    the standard representation of the QuantumSim matrices:
        '0' == [[1, 0],[0,0]]
        'X' == [[0, 1],[1,0]]
        'Y' == [[0,-1.j],[1.j,0]]
        '1' == [[0, 0],[0,1]]
    The normalization is such that the resulting basis is orthonormal
    under the trace inner product:
        Tr( dot(Mi,Mj) ) == delta_ij.
    In the returned list, the right-most factor of the kronecker product
    varies the fastest, so, for example, when matrix_dim == 4 the
    returned list is:
        [ 00,0X,0Y,01,X0,XX,XY,X1,Y0,Y0,YX,YY,Y1,10,1X,1Y,11 ].

    Parameters
    ----------
    matrix_dim : int
        Matrix-dimension of the density-matrix space.  Must be
        a power of 2.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (matrix_dim, matrix_dim), where N == matrix_dim^2,
        the dimension of the density-matrix space.

    Notes
    -----
    Matrices are ordered with first qubit being most significant,
    e.g., for 2 qubits: 00, 0X, 0Y, 01, X0, XX, XY, X1, Y0, ... 11
    """
    sig0q = _np.array([[1., 0], [0, 0]], dtype='complex')
    sigXq = _np.array([[0, 1], [1, 0]], dtype='complex')
    sigYq = _np.array([[0, -1], [1, 0]], dtype='complex') * 1.j
    sig1q = _np.array([[0, 0], [0, 1]], dtype='complex')

    _check_dim(matrix_dim)
    sigmaVec = (sig0q, sigXq / _np.sqrt(2.), sigYq / _np.sqrt(2.), sig1q)
    if matrix_dim == 0: return []

    def _is_integer(x):
        return bool(abs(x - round(x)) < 1e-6)

    nQubits = _np.log2(matrix_dim)
    if not _is_integer(nQubits):
        raise ValueError(
            "Dimension for QuantumSim tensor product matrices must be an integer *power of 2* (not %d)" % matrix_dim)
    nQubits = int(round(nQubits))

    if nQubits == 0:  # special case: return single 1x1 identity mx
        return [_np.identity(1, 'complex')]

    matrices = []
    basisIndList = [[0, 1, 2, 3]] * nQubits
    for sigmaInds in _itertools.product(*basisIndList):

        M = _np.identity(1, 'complex')
        for i in sigmaInds:
            M = _np.kron(M, sigmaVec[i])
        matrices.append(M)

    return matrices


def qsim_labels(matrix_dim):
    """
    QSim basis labels.

    Parameters
    ----------
    matrix_dim : int
        The matrix dimension to get labels for.

    Returns
    -------
    list
    """
    def _is_integer(x):
        return bool(abs(x - round(x)) < 1e-6)
    if matrix_dim == 0: return []
    if matrix_dim == 1: return ['']  # special case - use empty label instead of "I"

    nQubits = _np.log2(matrix_dim)
    if not _is_integer(nQubits):
        raise ValueError("Dimension for QuantumSim tensor product matrices must be an integer *power of 2*")
    nQubits = int(round(nQubits))

    lblList = []
    basisLblList = [['0', 'X', 'Y', '1']] * nQubits
    for sigmaLbls in _itertools.product(*basisLblList):
        lblList.append(''.join(sigmaLbls))
    return lblList


def pp_matrices(matrix_dim, max_weight=None, normalize=True):
    """
    Get the elements of the Pauil-product basis with matrix dimension `matrix_dim`.

    These matrices span the space of matrix_dim x matrix_dim density matrices
    (matrix-dimension matrix_dim, space dimension matrix_dim^2).

    The returned matrices are given in the standard basis of the
    density matrix space, and are thus kronecker products of
    the standard representation of the Pauli matrices, (i.e. where
    sigma_y == [[ 0, -i ], [i, 0]] ) normalized (when `normalize=True`
    so that the resulting basis is orthonormal under the trace inner
    product, i.e. Tr( dot(Mi,Mj) ) == delta_ij.  In the returned list,
    the right-most factor of the kronecker product varies the
    fastest, so, for example, when matrix_dim == 4 the returned list
    is [ II,IX,IY,IZ,XI,XX,XY,XY,YI,YX,YY,YZ,ZI,ZX,ZY,ZZ ].

    Parameters
    ----------
    matrix_dim : int
        Matrix-dimension of the density-matrix space.  Must be
        a power of 2.

    max_weight : int, optional
        Restrict the elements returned to those having weight <= `max_weight`. An
        element's "weight" is defined as the number of non-identity single-qubit
        factors of which it is comprised.  For example, if `matrix_dim == 4` and
        `max_weight == 1` then the returned list is [II, IX, IY, IZ, XI, YI, ZI].

    normalize : bool, optional
        Whether the Pauli matrices are normalized (see above) or not.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (matrix_dim, matrix_dim), where N == matrix_dim^2,
        the dimension of the density-matrix space. (Exception: when max_weight
        is not None, the returned list may have fewer than N elements.)

    Notes
    -----
    Matrices are ordered with first qubit being most significant,
    e.g., for 2 qubits: II, IX, IY, IZ, XI, XX, XY, XZ, YI, ... ZZ
    """
    _check_dim(matrix_dim)
    sigmaVec = (id2x2, sigmax, sigmay, sigmaz)
    if normalize:
        sigmaVec = tuple((s / sqrt2 for s in sigmaVec))

    if matrix_dim == 0: return []

    def _is_integer(x):
        return bool(abs(x - round(x)) < 1e-6)

    nQubits = _np.log2(matrix_dim)
    if not _is_integer(nQubits):
        raise ValueError(
            "Dimension for Pauli tensor product matrices must be an integer *power of 2* (not %d)" % matrix_dim)
    nQubits = int(round(nQubits))

    if nQubits == 0:  # special case: return single 1x1 identity mx
        return [_np.identity(1, 'complex')]

    matrices = []
    basisIndList = [[0, 1, 2, 3]] * nQubits
    for sigmaInds in _itertools.product(*basisIndList):
        if max_weight is not None:
            if sigmaInds.count(0) < nQubits - max_weight: continue

        M = _np.identity(1, 'complex')
        for i in sigmaInds:
            M = _np.kron(M, sigmaVec[i])
        matrices.append(M)

    return matrices


PP_matrices = _partial(pp_matrices, normalize=False)


def pp_labels(matrix_dim):
    """
    Pauli-product basis labels.

    Parameters
    ----------
    matrix_dim : int
        The matrix dimension to get labels for.

    Returns
    -------
    list
    """
    def _is_integer(x):
        return bool(abs(x - round(x)) < 1e-6)
    if matrix_dim == 0: return []
    if matrix_dim == 1: return ['']  # special case - use empty label instead of "I"

    nQubits = _np.log2(matrix_dim)
    if not _is_integer(nQubits):
        raise ValueError("Dimension for Pauli tensor product matrices must be an integer *power of 2*")
    nQubits = int(round(nQubits))

    lblList = []
    basisLblList = [['I', 'X', 'Y', 'Z']] * nQubits
    for sigmaLbls in _itertools.product(*basisLblList):
        lblList.append(''.join(sigmaLbls))
    return lblList


def qt_matrices(matrix_dim, selected_pp_indices=(0, 5, 10, 11, 1, 2, 3, 6, 7)):
    """
    Get the elements of a special basis spanning the density-matrix space of a qutrit.

    The returned matrices are given in the standard basis of the
    density matrix space. These matrices form an orthonormal basis
    under the trace inner product, i.e. Tr( dot(Mi,Mj) ) == delta_ij.

    Parameters
    ----------
    matrix_dim : int
        Matrix-dimension of the density-matrix space.  Must equal 3
        (present just to maintain consistency which other routines)

    selected_pp_indices : tuple, optional
        The indices of the 2-qubit Pauli-product matrices that should be projected
        onto the qutrit space to arrive at a qutrit basis.  Don't alter this value
        unless you know what you're doing.

    Returns
    -------
    list
        A list of 9 numpy arrays each of shape (3, 3).
    """
    if matrix_dim == 1:  # special case of just identity mx
        return [_np.identity(1, 'd')]

    assert(matrix_dim == 3)
    A = _np.array([[1, 0, 0, 0],
                   [0, 1. / _np.sqrt(2), 1. / _np.sqrt(2), 0],
                   [0, 0, 0, 1]], 'd')  # projector onto symmetric space

    def _to_qutrit_space(input_matrix):
        return _np.dot(A, _np.dot(input_matrix, A.transpose()))

    qt_mxs = []
    pp_mxs = pp_matrices(4)
    #selected_pp_indices = [0,5,10,11,1,2,3,6,7] #which pp mxs to project
    # labels = ['II', 'XX', 'YY', 'YZ', 'IX', 'IY', 'IZ', 'XY', 'XZ']
    qt_mxs = [_to_qutrit_space(pp_mxs[i]) for i in selected_pp_indices]

    # Normalize so Tr(BiBj) = delta_ij (done by hand, since only 3x3 mxs)
    qt_mxs[0] *= 1 / _np.sqrt(0.75)

    #TAKE 2 (more symmetric = better?)
    q1 = qt_mxs[1] - qt_mxs[0] * _np.sqrt(0.75) / 3
    q2 = qt_mxs[2] - qt_mxs[0] * _np.sqrt(0.75) / 3
    qt_mxs[1] = (q1 + q2) / _np.sqrt(2. / 3.)
    qt_mxs[2] = (q1 - q2) / _np.sqrt(2)

    #TAKE 1 (XX-II and YY-XX-II terms... not symmetric):
    #qt_mxs[1] = (qt_mxs[1] - qt_mxs[0]*_np.sqrt(0.75)/3) / _np.sqrt(2.0/3.0)
    #qt_mxs[2] = (qt_mxs[2] - qt_mxs[0]*_np.sqrt(0.75)/3 + qt_mxs[1]*_np.sqrt(2.0/3.0)/2) / _np.sqrt(0.5)

    for i in range(3, 9): qt_mxs[i] *= 1 / _np.sqrt(0.5)

    return qt_mxs


def qt_labels(matrix_dim):
    """
    The qutrit-basis labels based on a matrix dimension.

    Parameters
    ----------
    matrix_dim : int
        The matrix dimension of the basis to generate labels for (the
        number of rows or columns in a matrix).

    Returns
    -------
    list of strs
    """
    if matrix_dim == 0: return []
    if matrix_dim == 1: return ['']  # special case
    assert(matrix_dim == 3), "Qutrit basis must have matrix_dim == 3!"
    return ['II', 'X+Y', 'X-Y', 'YZ', 'IX', 'IY', 'IZ', 'XY', 'XZ']


def identity_matrices(matrix_dim):
    """
    Matrices for the "identity" basis of matrix dimension `matrix_dim`.

    The "identity basis" consists of only the identity matrix, so
    this function returns a list of a single `matrix_dim` x `matrix_dim`
    identity matrix.

    Parameters
    ----------
    matrix_dim : int
        The matrix dimension.

    Returns
    -------
    list
    """
    if matrix_dim == 0: return []
    assert(isinstance(matrix_dim, _numbers.Integral))
    d = matrix_dim
    return [_np.identity(d, 'complex')]


def identity_labels(dim):
    """
    The identity-basis labels based on a matrix dimension.

    Parameters
    ----------
    dim : int
        The matrix dimension.
    """
    return ['I']


def cl_vectors(dim):
    """
    Get the elements (vectors) of the classical basis with dimension `dim`

    That is, the `dim` standard unit vectors of length `dim`.

    Parameters
    ----------
    dim : int
        dimension of the vector space.

    Returns
    -------
    list
        A list of `dim` numpy arrays each of shape (dim,).
    """
    vecList = []
    for i in range(dim):
        v = _np.zeros(dim, 'd'); v[i] = 1.0
        vecList.append(v)
    return vecList


def cl_labels(dim):
    """
    Return the classical-basis labels based on a vector dimension.

    Parameters
    ----------
    dim : int
        The dimension of the basis to generate labels for (e.g.
        2 for a single classical bit).

    Returns
    -------
    list of strs
    """
    if dim == 0: return []
    if dim == 1: return ['']  # special case - use empty label instead of "0"
    return ["%d" % i for i in range(dim)]


def clgm_vectors(dim):
    """
    Get the elements (vectors) of the classical Gell-Mann basis with dimension `dim`

    The elements of this basis are the *diagonals* of the un-normalized Gell-Mann
    basis (of the matching dimension) elements with non-zero diagonal.

    Parameters
    ----------
    dim : int
        dimension of the vector space.

    Returns
    -------
    list
        A list of `dim` numpy arrays each of shape (dim,).
    """
    if dim == 0: return []
    vecList = [_np.ones(dim, 'd')]
    for diag_mx in _get_gell_mann_non_identity_diag_mxs(dim):
        vecList.append(_np.diag(diag_mx).copy())
    return vecList


def clgm_labels(dim):
    """
    Return the classical Gell-Mann basis labels based on a vector dimension.

    Parameters
    ----------
    dim : int
        The dimension of the basis to generate labels for (e.g.
        2 for a single classical bit).

    Returns
    -------
    list of strs
    """
    if dim == 0: return []
    if dim == 1: return ['']  # special case - use empty label instead of "I"
    return ["I"] + ["Z_{%d}" % i for i in range(1, dim)]


def clpp_vectors(dim):
    """
    Get the elements (vectors) of the classical Pauli-product basis with dimension `dim`

    The elements of this basis are the *diagonals* of the un-normalized Pauli-product
    basis (of the matching dimension) elements with non-zero diagonal (those using I and Z).

    Parameters
    ----------
    dim : int
        dimension of the vector space.

    Returns
    -------
    list
        A list of `dim` numpy arrays each of shape (dim,).
    """
    if dim == 0: return []
    sigmaVec = (_np.ones(2, 'd'), _np.array([1, -1], 'd'))

    def _is_integer(x):
        return bool(abs(x - round(x)) < 1e-6)

    nBits = _np.log2(dim)
    if not _is_integer(nBits):
        raise ValueError(
            "Dimension for classical Pauli basis must be an integer *power of 2* (not %d)" % dim)
    nBits = int(round(nBits))

    if nBits == 0: return [_np.ones(1, 'd')]

    vecList = [_np.ones(dim, 'd')]
    basisIndList = [[0, 1]] * nBits
    for sigmaInds in _itertools.product(*basisIndList):
        V = _np.ones(1, 'd')
        for i in sigmaInds:
            V = _np.kron(V, sigmaVec[i])
        vecList.append(V)

    return vecList


def clpp_labels(dim):
    """
    Return the classical Pauli-product basis labels based on a vector dimension.

    Parameters
    ----------
    dim : int
        The dimension of the basis to generate labels for (e.g.
        2 for a single classical bit).

    Returns
    -------
    list of strs
    """
    def _is_integer(x):
        return bool(abs(x - round(x)) < 1e-6)

    if dim == 0: return []
    if dim == 1: return ['']  # special case - use empty label instead of "I"

    nBits = _np.log2(dim)
    if not _is_integer(nBits):
        raise ValueError("Dimension for classical Pauli basis must be an integer *power of 2*")
    nBits = int(round(nBits))

    lblList = []
    basisLblList = [['I', 'Z']] * nBits
    for sigmaLbls in _itertools.product(*basisLblList):
        lblList.append(''.join(sigmaLbls))
    return lblList


def sv_vectors(dim):
    """
    Get the elements (vectors) of the complex state-vector basis with dimension `dim`.

    That is, the `dim` standard complex unit vectors of length `dim`.

    Parameters
    ----------
    dim : int
        dimension of the vector space.

    Returns
    -------
    list
        A list of `dim` numpy arrays each of shape (dim,).
    """
    vecList = []
    for i in range(dim):
        v = _np.zeros(dim, complex); v[i] = 1.0
        vecList.append(v)
    return vecList


def sv_labels(dim):
    """
    Return the state-vector-basis labels based on a vector dimension.

    Parameters
    ----------
    dim : int
        The dimension of the basis to generate labels for (e.g.
        2 for a single qubit represented as a state vector).

    Returns
    -------
    list of strs
    """
    if dim == 0: return []
    if dim == 1: return ['']  # special case - use empty label instead of "0"
    return ["|%d>" % i for i in range(dim)]


def unknown_els(dim):
    """
    The elements of the "unknown" basis.  Just returns an empty list.

    Parameters
    ----------
    dim : int
        The dimension of the basis (doesn't really matter).

    Returns
    -------
    list
    """
    assert(dim == 0), "Unknown basis must have dimension 0!"
    return []


def unknown_labels(dim):
    """
    The labels for the "unknown" basis.  Just returns an empty list.

    Parameters
    ----------
    dim : int
        Dimension

    Returns
    -------
    list
    """
    return []


_basis_constructor_dict = dict()  # global dict holding all builtin basis constructors (used by Basis objects)
_basis_constructor_dict['std'] = MatrixBasisConstructor('Matrix-unit basis', std_matrices, std_labels, False)
_basis_constructor_dict['col'] = MatrixBasisConstructor('Column-stacked matrix-unit basis', col_matrices,
                                                        col_labels, False)
_basis_constructor_dict['gm_unnormalized'] = MatrixBasisConstructor(
    'Unnormalized Gell-Mann basis', gm_matrices_unnormalized, gm_labels, True)
_basis_constructor_dict['gm'] = MatrixBasisConstructor('Gell-Mann basis', gm_matrices, gm_labels, True)
_basis_constructor_dict['pp'] = MatrixBasisConstructor('Normalized Pauli-Product basis', pp_matrices, pp_labels, True)
_basis_constructor_dict['PP'] = MatrixBasisConstructor('Pauli-Product basis', PP_matrices, pp_labels, True)
_basis_constructor_dict['qsim'] = MatrixBasisConstructor('QuantumSim basis', qsim_matrices, qsim_labels, True)
_basis_constructor_dict['qt'] = MatrixBasisConstructor('Qutrit basis', qt_matrices, qt_labels, True)
_basis_constructor_dict['id'] = SingleElementMatrixBasisConstructor('Identity-only subbasis', identity_matrices,
                                                                    identity_labels, True)
_basis_constructor_dict['clmx'] = DiagonalMatrixBasisConstructor('Diagonal Matrix-unit basis', cl_vectors,
                                                                 cl_labels, True)
_basis_constructor_dict['clgmmx'] = DiagonalMatrixBasisConstructor('Diagonal Gell-Mann basis', clgm_vectors,
                                                                   clgm_labels, True)
_basis_constructor_dict['clppmx'] = DiagonalMatrixBasisConstructor('Diagonal Pauli-Product basis', clpp_vectors,
                                                                   clpp_labels, True)
_basis_constructor_dict['cl'] = VectorBasisConstructor('Classical basis', cl_vectors, cl_labels, True)
_basis_constructor_dict['clgm'] = VectorBasisConstructor('Classical Gell-Mann basis', clgm_vectors, clgm_labels, True)
_basis_constructor_dict['clpp'] = VectorBasisConstructor('Classical Pauli-Product basis', clpp_vectors,
                                                         clpp_labels, True)
_basis_constructor_dict['sv'] = VectorBasisConstructor('State-vector basis', sv_vectors, sv_labels, False)
_basis_constructor_dict['unknown'] = VectorBasisConstructor('Unknown (0-dim) basis', unknown_els, unknown_labels, False)
