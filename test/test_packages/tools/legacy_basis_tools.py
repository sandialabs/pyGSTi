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
import itertools as _itertools
import numbers as _numbers
import collections as _collections

import numpy as _np
import scipy.linalg as _spl

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

##Matrix unit basis
def _mut(i,j,N):
    mx = _np.zeros( (N,N), 'd'); mx[i,j] = 1.0
    return mx
mxUnitVec = ( _mut(0,0,2), _mut(0,1,2), _mut(1,0,2), _mut(1,1,2) )
mxUnitVec_2Q = ( _mut(0,0,4), _mut(0,1,4), _mut(0,2,4), _mut(0,3,4),
                 _mut(1,0,4), _mut(1,1,4), _mut(1,2,4), _mut(1,3,4),
                 _mut(2,0,4), _mut(2,1,4), _mut(2,2,4), _mut(2,3,4),
                 _mut(3,0,4), _mut(3,1,4), _mut(3,2,4), _mut(3,3,4)  )


def _processBlockDims(dimOrBlockDims):
    """
    Performs basic processing on the dimensions
      of a direct-sum space.

    Parameters
    ----------
    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.
        A list of integers designates the space is
          the direct sum of spaces with the square of the given
          matrix-block dimensions.  Matrices in this space are
          represented in the standard basis by a block-diagonal
          matrix with blocks of the given dimensions.
        A single integer is equivalent to a list with a single
          element, and so designates the space of matrices with
          the given dimension, and thus a space of the dimension^2.

    Returns
    -------
    dmDim : int
        The (matrix) dimension of the overall density matrix
        within which the block-diagonal density matrix described by
        dimOrBlockDims is embedded, equal to the sum of the
        individual block dimensions. (The overall density matrix
        is a dmDim x dmDim matrix, and is contained in a space
        of dimension dmDim**2).
    gateDim : int
        The (matrix) dimension of the "gate-space" corresponding
        to the density matrix space, equal to the dimension
        of the density matrix space, sum( ith-block_dimension^2 ).
        Gate matrices are thus gateDim x gateDim dimensions.
    blockDims : list of ints
        Dimensions of the individual matrix-blocks.  The direct sum
        of the matrix spaces (of dim matrix-block-dim^2) forms the
        density matrix space.  Equals:
        [ dimOrBlockDims ] : if dimOrBlockDims is a single int
          dimOrBlockDims   : otherwise
    """
    # treat as state space dimensions
    if isinstance(dimOrBlockDims, _collections.Container):
        # *full* density matrix is dmDim x dmDim
        dmDim = sum([blockDim for blockDim in dimOrBlockDims])

        # gate matrices will be vecDim x vecDim
        gateDim = sum([blockDim**2 for blockDim in dimOrBlockDims])

        blockDims = dimOrBlockDims
    elif isinstance(dimOrBlockDims, _numbers.Integral):
        dmDim = dimOrBlockDims
        gateDim = dimOrBlockDims**2
        blockDims = [dimOrBlockDims]
    else:
        raise ValueError("Invalid dimOrBlockDims = %s" % str(dimOrBlockDims))

    return dmDim, gateDim, blockDims

def std_matrices(dimOrBlockDims):
    """
    Get the elements of the matrix unit, or "standard", basis
    spanning the density-matrix space given by dimOrBlockDims.

    The returned matrices are given in the standard basis of the
    "embedding" density matrix space, that is, the space which
    embeds the block-diagonal matrix structure stipulated in
    dimOrBlockDims. These matrices form an orthonormal basis under
    the trace inner product, i.e. Tr( dot(Mi,Mj) ) == delta_ij.

    Parameters
    ----------
    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (dmDim, dmDim),
        where dmDim is the matrix-dimension of the overall
        "embedding" density matrix (the sum of dimOrBlockDims)
        and N is the dimension of the density-matrix space,
        equal to sum( block_dim_i^2 ).

    Notes
    -----
    Each element is a matrix containing
    a single "1" entry amidst a background of zeros, and there
    are never "1"s in positions outside the block-diagonal structure.
    """
    dmDim, gateDim, blockDims = _processBlockDims(dimOrBlockDims)

    mxList = []; start = 0
    for blockDim in blockDims:
        for i in range(start,start+blockDim):
            for j in range(start,start+blockDim):
                mxList.append( _mut( i, j, dmDim ) )
        start += blockDim

    assert(len(mxList) == gateDim and start == dmDim)
    return mxList

def _GetGellMannNonIdentityDiagMxs(dimension):
    d = dimension
    listOfMxs = []
    if d > 2:
        dm1_listOfMxs = _GetGellMannNonIdentityDiagMxs(d-1)
        for dm1_mx in dm1_listOfMxs:
            mx = _np.zeros( (d,d), 'complex' )
            mx[0:d-1,0:d-1] = dm1_mx
            listOfMxs.append(mx)
    if d > 1:
        mx = _np.identity( d, 'complex' )
        mx[d-1,d-1] = 1-d
        mx *= _np.sqrt( 2.0 / (d*(d-1)) )
        listOfMxs.append(mx)

    return listOfMxs

def gm_matrices_unnormalized(dimOrBlockDims):
    """
    Get the elements of the generalized Gell-Mann
    basis spanning the density-matrix space given by dimOrBlockDims.

    The returned matrices are given in the standard basis of the
    "embedding" density matrix space, that is, the space which
    embeds the block-diagonal matrix structure stipulated in
    dimOrBlockDims. These matrices form an orthogonal but not
    orthonormal basis under the trace inner product.

    Parameters
    ----------
    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (dmDim, dmDim),
        where dmDim is the matrix-dimension of the overall
        "embedding" density matrix (the sum of dimOrBlockDims)
        and N is the dimension of the density-matrix space,
        equal to sum( block_dim_i^2 ).
    """
    if isinstance(dimOrBlockDims, _numbers.Integral):
        d = dimOrBlockDims

        #Identity Mx
        listOfMxs = [ _np.identity(d, 'complex') ]

        #Non-diagonal matrices -- only take those whose non-zero elements are not "frozen" in cssb case
        for k in range(d):
            for j in range(k+1,d):
                mx = _np.zeros( (d,d), 'complex' )
                mx[k,j] = mx[j,k] = 1.0
                listOfMxs.append( mx )

        for k in range(d):
            for j in range(k+1,d):
                mx = _np.zeros( (d,d), 'complex' )
                mx[k,j] = -1.0j; mx[j,k] = 1.0j
                listOfMxs.append( mx )

        #Non-Id Diagonal matrices
        listOfMxs.extend( _GetGellMannNonIdentityDiagMxs(d) )

        assert(len(listOfMxs) == d**2)
        return listOfMxs

    elif isinstance(dimOrBlockDims, _collections.Container):
        dmDim, gateDim, blockDims = _processBlockDims(dimOrBlockDims)

        listOfMxs = []; start = 0
        for blockDim in blockDims:
            for blockMx in gm_matrices_unnormalized(blockDim):
                mx = _np.zeros( (dmDim, dmDim), 'complex' )
                mx[start:start+blockDim, start:start+blockDim] = blockMx
                listOfMxs.append( mx )
            start += blockDim
        assert(len(listOfMxs) == gateDim)
        return listOfMxs

    else:
        raise ValueError("Invalid dimOrBlockDims = %s" % str(dimOrBlockDims))


def gm_matrices(dimOrBlockDims):
    """
    Get the normalized elements of the generalized Gell-Mann
    basis spanning the density-matrix space given by dimOrBlockDims.

    The returned matrices are given in the standard basis of the
    "embedding" density matrix space, that is, the space which
    embeds the block-diagonal matrix structure stipulated in
    dimOrBlockDims. These matrices form an orthonormal basis
    under the trace inner product, i.e. Tr( dot(Mi,Mj) ) == delta_ij.

    Parameters
    ----------
    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    Returns
    -------
    list
        A list of N numpy arrays each of shape (dmDim, dmDim),
        where dmDim is the matrix-dimension of the overall
        "embedding" density matrix (the sum of dimOrBlockDims)
        and N is the dimension of the density-matrix space,
        equal to sum( block_dim_i^2 ).
    """
    mxs = gm_matrices_unnormalized(dimOrBlockDims)
    mxs[0] *= 1/_np.sqrt( mxs[0].shape[0] ) #identity mx
    for mx in mxs[1:]:
        mx *= 1/sqrt2
    return mxs

def gm_to_std_transform_matrix(dimOrBlockDims):
    """
    Construct the matrix which transforms a gate matrix in
    the Gell-Mann basis for a density matrix space to the
    Standard basis (for the same space).

    Parameters
    ----------
    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    Returns
    -------
    numpy array
        An array of shape (N,N), where N is the dimension
        of the density matrix space, i.e. sum( dimOrBlockDims_i^2 ).

    Notes
    -----
        The returned matrix is block diagonal with one block
        per term in the direct sum of the the density matrix space.
        Each block is the transformation matrix for the corresponding
        part of density matrix space, consisting of flattened Gell-Mann
        basis matrices along it's columns.
    """
    #vectorize Gell Mann mxs and place appropriate elements into columns of a matrix
    _, gateDim, blockDims = _processBlockDims(dimOrBlockDims)
    gmToStd = _np.zeros( (gateDim,gateDim), 'complex' )

    #Since a multi-block basis is just the direct sum of the individual block bases,
    # transform mx is just the transfrom matrices of the individual blocks along the
    # diagonal of the total basis transform matrix

    start = 0
    for blockDim in blockDims:
        mxs = gm_matrices(blockDim)
        assert( len(mxs) == blockDim**2 )

        for j,mx in enumerate(mxs):
            gmToStd[start:start+blockDim**2,start+j] = mx.flatten()

        start += blockDim**2

    assert(start == gateDim)
    return gmToStd

def std_to_gm(mxInStdBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Standard basis of a
    density matrix space to the Gell-Mann basis (of the same space).

    Parameters
    ----------
    mxInStdBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInStdBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInStdBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Gell-Mann basis.
        Array size is the same as mxInStdBasis.
    """
    if dimOrBlockDims is None:
        dimOrBlockDims = int(round(_np.sqrt(mxInStdBasis.shape[0])))
        assert( dimOrBlockDims**2 == mxInStdBasis.shape[0] )

    gmToStd = gm_to_std_transform_matrix(dimOrBlockDims)
    stdToGM = _np.linalg.inv(gmToStd)

    if len(mxInStdBasis.shape) == 2 and mxInStdBasis.shape[0] == mxInStdBasis.shape[1]:
        gm = _np.dot( stdToGM, _np.dot( mxInStdBasis, gmToStd ) )
        if _np.linalg.norm(_np.imag(gm)) > 1e-8:
            raise ValueError("Gell-Mann matrix has non-zero imaginary part (%g)!" %
                             _np.linalg.norm(_np.imag(gm)))
            #For debug, comment out exception above and uncomment this:
            #print "Warning: Gell-Mann matrix has non-zero imaginary part (%g)!" % \
            #    _np.linalg.norm(_np.imag(gm))
            #return gm
        return _np.real(gm)

    elif len(mxInStdBasis.shape) == 1 or \
         (len(mxInStdBasis.shape) == 2 and mxInStdBasis.shape[1] == 1): # (really vecInStdBasis)
        gm = _np.dot( stdToGM, mxInStdBasis )
        if _np.linalg.norm(_np.imag(gm)) > 1e-8:
            raise ValueError("Gell-Mann vector has non-zero imaginary part (%g)!" %
                             _np.linalg.norm(_np.imag(gm)))
            #For debug, comment out exception above and uncomment this:
            #print "Warning: Gell-Mann vector has non-zero imaginary part (%g)!" % \
            #                 _np.linalg.norm(_np.imag(gm))
            #return gm
        return _np.real(gm)

    else: raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")


def gm_to_std(mxInGellMannBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Gell-Mann basis of a
    density matrix space to the Standard basis (of the same space).

    Parameters
    ----------
    mxInGellMannBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInGellMannBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInGellMannBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Standard basis.
        Array size is the same as mxInGellMannBasis.
    """

    if dimOrBlockDims is None:
        dimOrBlockDims = int(round(_np.sqrt(mxInGellMannBasis.shape[0])))
        assert( dimOrBlockDims**2 == mxInGellMannBasis.shape[0] )

    gmToStd = gm_to_std_transform_matrix(dimOrBlockDims)
    stdToGM = _np.linalg.inv(gmToStd)

    if len(mxInGellMannBasis.shape) == 2 and mxInGellMannBasis.shape[0] == mxInGellMannBasis.shape[1]:
        return _np.dot( gmToStd, _np.dot( mxInGellMannBasis, stdToGM ) )

    elif len(mxInGellMannBasis.shape) == 1 or \
         (len(mxInGellMannBasis.shape) == 2 and mxInGellMannBasis.shape[1] == 1): # (really vecInStdBasis)
        return _np.dot( gmToStd, mxInGellMannBasis )

    else: raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")


def pp_matrices(dim, maxWeight=None):
    """
    Get the elements of the Pauil-product basis
    spanning the space of dim x dim density matrices
    (matrix-dimension dim, space dimension dim^2).

    The returned matrices are given in the standard basis of the
    density matrix space, and are thus kronecker products of
    the standard representation of the Pauli matrices, (i.e. where
    sigma_y == [[ 0, -i ], [i, 0]] ) normalized so that the
    resulting basis is orthonormal under the trace inner product,
    i.e. Tr( dot(Mi,Mj) ) == delta_ij.  In the returned list,
    the right-most factor of the kronecker product varies the
    fastsest, so, for example, when dim == 4 the returned list
    is [ II,IX,IY,IZ,XI,XX,XY,XY,YI,YX,YY,YZ,ZI,ZX,ZY,ZZ ].

    Parameters
    ----------
    dim : int
        Matrix-dimension of the density-matrix space.  Must be
        a power of 2.

    maxWeight : int, optional
        Restrict the elements returned to those having weight <= `maxWeight`. An
        element's "weight" is defined as the number of non-identity single-qubit
        factors of which it is comprised.  For example, if `dim == 4` and 
        `maxWeight == 1` then the returned list is [II, IX, IY, IZ, XI, YI, ZI].


    Returns
    -------
    list
        A list of N numpy arrays each of shape (dim, dim), where N == dim^2,
        the dimension of the density-matrix space. (Exception: when maxWeight
        is not None, the returned list may have fewer than N elements.)

    Notes
    -----
    Matrices are ordered with first qubit being most significant,
    e.g., for 2 qubits: II, IX, IY, IZ, XI, XX, XY, XZ, YI, ... ZZ
    """

    sigmaVec = (id2x2/sqrt2, sigmax/sqrt2, sigmay/sqrt2, sigmaz/sqrt2)

    def is_integer(x):
        return bool( abs(x - round(x)) < 1e-6 )

    if not isinstance(dim, _numbers.Integral):
        if isinstance(dim, _collections.Container) and len(dim) == 1:
            dim = dim[0]
        else:
            raise ValueError("Dimension for Pauli tensor product matrices must be an *integer* power of 2")

    nQubits = _np.log2(dim)
    if not is_integer(nQubits):
        raise ValueError("Dimension for Pauli tensor product matrices must be an integer *power of 2*")

    if nQubits == 0: #special case: return single 1x1 identity mx
        return [ _np.identity(1,'complex') ]

    matrices = []
    nQubits = int(round(nQubits))
    basisIndList = [ [0,1,2,3] ]*nQubits
    for sigmaInds in _itertools.product(*basisIndList):
        if maxWeight is not None:
            if sigmaInds.count(0) < nQubits-maxWeight: continue
            
        M = _np.identity(1,'complex')
        for i in sigmaInds:
            M = _np.kron(M,sigmaVec[i])
        matrices.append(M)

    return matrices


def pp_to_std_transform_matrix(dimOrBlockDims):
    """
    Construct the matrix which transforms a gate matrix in
    the Pauil-product basis for a density matrix space to the
    Standard basis (for the same space).

    Parameters
    ----------
    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    Returns
    -------
    numpy array
        An array of shape (N,N), where N is the dimension
        of the density matrix space, i.e. sum( dimOrBlockDims_i^2 ).

    Notes
    -----
        The returned matrix is block diagonal with one block
        per term in the direct sum of the the density matrix space.
        Each block is the transformation matrix for the corresponding
        part of density matrix space, consisting of flattened Pauli-product
        basis matrices along it's columns.
    """

    #vectorize tensor products of Pauli mxs and place them as columns into a matrix
    _, gateDim, blockDims = _processBlockDims(dimOrBlockDims)
    ppToStd = _np.zeros( (gateDim,gateDim), 'complex' )

    #Since a multi-block basis is just the direct sum of the individual block bases,
    # transform mx is just the transfrom matrices of the individual blocks along the
    # diagonal of the total basis transform matrix

    start = 0
    for blockDim in blockDims:
        mxs = pp_matrices(blockDim)
        assert( len(mxs) == blockDim**2 )

        for j,mx in enumerate(mxs):
            ppToStd[start:start+blockDim**2,start+j] = mx.flatten()

        start += blockDim**2

    assert(start == gateDim)
    return ppToStd


def std_to_pp(mxInStdBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Standard basis of a
    density matrix space to the Pauil-product basis (of the same space).

    Parameters
    ----------
    mxInStdBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInStdBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInStdBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Pauil-product basis.
        Array size is the same as mxInStdBasis.
    """

    if dimOrBlockDims is None:
        dimOrBlockDims = int(round(_np.sqrt(mxInStdBasis.shape[0])))
        assert( dimOrBlockDims**2 == mxInStdBasis.shape[0] )

    ppToStd = pp_to_std_transform_matrix(dimOrBlockDims)
    stdToPP = _np.linalg.inv(ppToStd)

    if len(mxInStdBasis.shape) == 2 and mxInStdBasis.shape[0] == mxInStdBasis.shape[1]:
        pp = _np.dot( stdToPP, _np.dot( mxInStdBasis, ppToStd ) )
        if _np.linalg.norm(_np.imag(pp)) > 1e-8:
            raise ValueError("Pauil-product matrix has non-zero imaginary part (%g)!" %
                             _np.linalg.norm(_np.imag(pp)))
            #For debug, comment out exception above and uncomment this:
            #print "Warning: Pauli-product matrix has non-zero imaginary part (%g)!" % \
            #    _np.linalg.norm(_np.imag(pp))
            #return pp
        return _np.real(pp)

    elif len(mxInStdBasis.shape) == 1 or \
         (len(mxInStdBasis.shape) == 2 and mxInStdBasis.shape[1] == 1): # (really vecInStdBasis)
        pp = _np.dot( stdToPP, mxInStdBasis )
        if _np.linalg.norm(_np.imag(pp)) > 1e-8:
            raise ValueError("Pauil-product vector has non-zero imaginary part (%g)!" %
                             _np.linalg.norm(_np.imag(pp)))
            #For debug, comment out exception above and uncomment this:
            #print "Warning: Pauli-product vector has non-zero imaginary part (%g)!" % \
            #    _np.linalg.norm(_np.imag(pp))
            #return pp
        return _np.real(pp)


    else: raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")


def pp_to_std(mxInPauliProdBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Pauli-product basis of a
    density matrix space to the Standard basis (of the same space).

    Parameters
    ----------
    mxInPauliProdBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInPauliProdBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInPauliProdBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Standard basis.
        Array size is the same as mxInPauliProdBasis.
    """

    if dimOrBlockDims is None:
        dimOrBlockDims = int(round(_np.sqrt(mxInPauliProdBasis.shape[0])))
        assert( dimOrBlockDims**2 == mxInPauliProdBasis.shape[0] )

    ppToStd = pp_to_std_transform_matrix(dimOrBlockDims)
    stdToPP = _np.linalg.inv(ppToStd)

    if len(mxInPauliProdBasis.shape) == 2 and mxInPauliProdBasis.shape[0] == mxInPauliProdBasis.shape[1]:
        return _np.dot( ppToStd, _np.dot( mxInPauliProdBasis, stdToPP ) )

    elif len(mxInPauliProdBasis.shape) == 1 or \
         (len(mxInPauliProdBasis.shape) == 2 and mxInPauliProdBasis.shape[1] == 1): # (really vecInPauilProdBasis)
        return _np.dot( ppToStd, mxInPauliProdBasis )

    else: raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")


def qt_matrices(dim, selected_pp_indices=[0,5,10,11,1,2,3,6,7]):
    """
    Get the elements of a special basis spanning the density-matrix space of
    a qutrit.

    The returned matrices are given in the standard basis of the
    density matrix space. These matrices form an orthonormal basis
    under the trace inner product, i.e. Tr( dot(Mi,Mj) ) == delta_ij.

    Parameters
    ----------
    dim : int
        Matrix-dimension of the density-matrix space.  Must equal 3
        (present just to maintain consistency which other routines)

    Returns
    -------
    list
        A list of 9 numpy arrays each of shape (3, 3).
    """
    assert(dim == 3)
    A = _np.array( [[1,0,0,0],
                   [0,1./_np.sqrt(2),1./_np.sqrt(2),0],
                   [0,0,0,1]], 'd') #projector onto symmetric space
    
    def toQutritSpace(inputMat):
        return _np.dot(A,_np.dot(inputMat,A.transpose()))

    qt_mxs = []
    pp_mxs = pp_matrices(4)
    #selected_pp_indices = [0,5,10,11,1,2,3,6,7] #which pp mxs to project
    # labels = ['II', 'XX', 'YY', 'YZ', 'IX', 'IY', 'IZ', 'XY', 'XZ']
    qt_mxs = [toQutritSpace(pp_mxs[i]) for i in selected_pp_indices]

    # Normalize so Tr(BiBj) = delta_ij (done by hand, since only 3x3 mxs)
    qt_mxs[0] *= 1/_np.sqrt(0.75)
    
    #TAKE 2 (more symmetric = better?)
    q1 = qt_mxs[1] - qt_mxs[0]*_np.sqrt(0.75)/3
    q2 = qt_mxs[2] - qt_mxs[0]*_np.sqrt(0.75)/3
    qt_mxs[1] = (q1 + q2)/_np.sqrt(2./3.)
    qt_mxs[2] = (q1 - q2)/_np.sqrt(2)

    #TAKE 1 (XX-II and YY-XX-II terms... not symmetric):
    #qt_mxs[1] = (qt_mxs[1] - qt_mxs[0]*_np.sqrt(0.75)/3) / _np.sqrt(2.0/3.0)
    #qt_mxs[2] = (qt_mxs[2] - qt_mxs[0]*_np.sqrt(0.75)/3 + qt_mxs[1]*_np.sqrt(2.0/3.0)/2) / _np.sqrt(0.5)

    for i in range(3,9): qt_mxs[i] *= 1/ _np.sqrt(0.5)
    
    return qt_mxs


def qt_to_std_transform_matrix(dimOrBlockDims):
    """
    Construct the matrix which transforms a gate matrix in
    the Qutrit basis for a density matrix space to the
    Standard basis (for the same space).

    Parameters
    ----------
    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    Returns
    -------
    numpy array
        An array of shape (N,N), where N is the dimension
        of the density matrix space, i.e. sum( dimOrBlockDims_i^2 ).

    Notes
    -----
        The returned matrix is block diagonal with one block
        per term in the direct sum of the the density matrix space.
        Each block is the transformation matrix for the corresponding
        part of density matrix space, consisting of flattened Qutrit
        basis matrices along it's columns.
    """
    #vectorize Gell Mann mxs and place appropriate elements into columns of a matrix
    _, gateDim, blockDims = _processBlockDims(dimOrBlockDims)
    gmToStd = _np.zeros( (gateDim,gateDim), 'complex' )

    #Since a multi-block basis is just the direct sum of the individual block bases,
    # transform mx is just the transfrom matrices of the individual blocks along the
    # diagonal of the total basis transform matrix

    start = 0
    for blockDim in blockDims:
        mxs = qt_matrices(blockDim)
        assert( len(mxs) == blockDim**2 )

        for j,mx in enumerate(mxs):
            gmToStd[start:start+blockDim**2,start+j] = mx.flatten()

        start += blockDim**2

    assert(start == gateDim)
    return gmToStd

def std_to_qt(mxInStdBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Standard basis of a
    density matrix space to the Qutrit basis (of the same space).

    Parameters
    ----------
    mxInStdBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInStdBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInStdBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Qutrit basis.
        Array size is the same as mxInStdBasis.
    """
    if dimOrBlockDims is None:
        dimOrBlockDims = int(round(_np.sqrt(mxInStdBasis.shape[0])))
        assert( dimOrBlockDims**2 == mxInStdBasis.shape[0] )

    qtToStd = qt_to_std_transform_matrix(dimOrBlockDims)
    stdToQT = _np.linalg.inv(qtToStd)

    if len(mxInStdBasis.shape) == 2 and mxInStdBasis.shape[0] == mxInStdBasis.shape[1]:
        qt = _np.dot( stdToQT, _np.dot( mxInStdBasis, qtToStd ) )
        if _np.linalg.norm(_np.imag(qt)) > 1e-8:
            raise ValueError("Qutrit matrix has non-zero imaginary part (%g)!" %
                             _np.linalg.norm(_np.imag(qt)))
        return _np.real(qt)

    elif len(mxInStdBasis.shape) == 1 or \
         (len(mxInStdBasis.shape) == 2 and mxInStdBasis.shape[1] == 1): # (really vecInStdBasis)
        qt = _np.dot( stdToQT, mxInStdBasis )
        if _np.linalg.norm(_np.imag(qt)) > 1e-8:
            raise ValueError("Qutrit vector has non-zero imaginary part (%g)!" %
                             _np.linalg.norm(_np.imag(qt)))
        return _np.real(qt)

    else: raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")


def qt_to_std(mxInQutritBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Qutrit basis of a
    density matrix space to the Standard basis (of the same space).

    Parameters
    ----------
    mxInQutritBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInQutritBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInQutritBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Standard basis.
        Array size is the same as mxInQutritBasis.
    """

    if dimOrBlockDims is None:
        dimOrBlockDims = int(round(_np.sqrt(mxInQutritBasis.shape[0])))
        assert( dimOrBlockDims**2 == mxInQutritBasis.shape[0] )

    qtToStd = qt_to_std_transform_matrix(dimOrBlockDims)
    stdToQT = _np.linalg.inv(qtToStd)

    if len(mxInQutritBasis.shape) == 2 and mxInQutritBasis.shape[0] == mxInQutritBasis.shape[1]:
        return _np.dot( qtToStd, _np.dot( mxInQutritBasis, stdToQT ) )

    elif len(mxInQutritBasis.shape) == 1 or \
         (len(mxInQutritBasis.shape) == 2 and mxInQutritBasis.shape[1] == 1):
        return _np.dot( qtToStd, mxInQutritBasis )

    else: raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")


#Other permutations for conversions

def gm_to_pp(mxInGellMannBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Gell-Mann basis of a
    density matrix space to the Pauil-product basis (of the same space).

    Parameters
    ----------
    mxInGellMannBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInGellMannBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInGellMannBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Pauli-product basis.
        Array size is the same as mxInGellMannBasis.
    """
    return std_to_pp(gm_to_std(mxInGellMannBasis, dimOrBlockDims), dimOrBlockDims)


def gm_to_qt(mxInGellMannBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Gell-Mann basis of a
    density matrix space to the Qutrit basis (of the same space).

    Parameters
    ----------
    mxInGellMannBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInGellMannBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInGellMannBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Qutrit basis.
        Array size is the same as mxInGellMannBasis.
    """
    return std_to_qt(gm_to_std(mxInGellMannBasis, dimOrBlockDims), dimOrBlockDims)


def pp_to_gm(mxInPauliProdBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Pauli-product basis of a
    density matrix space to the Gell-Mann basis (of the same space).

    Parameters
    ----------
    mxInPauliProdBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInPauliProdBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInPauliProdBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Gell-Mann basis.
        Array size is the same as mxInPauliProdBasis.
    """
    return std_to_gm(pp_to_std(mxInPauliProdBasis, dimOrBlockDims), dimOrBlockDims)


def pp_to_qt(mxInPauliProdBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Pauli-product basis of a
    density matrix space to the Qutrit basis (of the same space).

    Parameters
    ----------
    mxInPauliProdBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInPauliProdBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInPauliProdBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Qutrit basis.
        Array size is the same as mxInPauliProdBasis.
    """
    return std_to_qt(pp_to_std(mxInPauliProdBasis, dimOrBlockDims), dimOrBlockDims)


def qt_to_gm(mxInQutritBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Qutrit basis of a
    density matrix space to the Gell-Mann basis (of the same space).

    Parameters
    ----------
    mxInQutritBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInQutritBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInQutritBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Gell-Mann basis.
        Array size is the same as mxInQutritBasis.
    """
    return std_to_gm(qt_to_std(mxInQutritBasis, dimOrBlockDims), dimOrBlockDims)


def qt_to_pp(mxInQutritBasis, dimOrBlockDims=None):
    """
    Convert a gate matrix in the Qutrit basis of a
    density matrix space to the Pauil-product basis (of the same space).

    Parameters
    ----------
    mxInQutritBasis : numpy array
        The gate matrix, (a 2D square array)

    dimOrBlockDims : int or list of ints, optional
        Structure of the density-matrix space. If None, then assume
        mxInQutritBasis operates on a single-block density matrix space,
        i.e. on K x K density matrices with K == sqrt( mxInQutritBasis.shape[0] ).

    Returns
    -------
    numpy array
        The given gate matrix converted to the Pauli-product basis.
        Array size is the same as mxInQutritBasis.
    """
    return std_to_pp(qt_to_std(mxInQutritBasis, dimOrBlockDims), dimOrBlockDims)


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
    if maxWeight is not None:
        assert(basis == "pp"),'Only the "pp" basis supports a non-None maxWeight'
        
    if basis == "std": return std_matrices(dimOrBlockDims)
    if basis == "gm":  return gm_matrices(dimOrBlockDims)
    if basis == "pp":  return pp_matrices(dimOrBlockDims,maxWeight)
    if basis == "qt":  return qt_matrices(dimOrBlockDims)
    raise ValueError("Invalid 'basis' argument: %s" % basis)
    

def basis_transform_matrix(from_basis, to_basis, dimOrBlockDims):
    """
    Get the matrix which transforms (coverts) from one density-matrix-space
    basis to another.

    Parameters
    ----------
    from_basis, to_basis: {'std', 'gm', 'pp', 'qt'}
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt).

    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    Returns
    -------
    numpy array
        The conversion matrix.
    """
    _, gateDim, _ = _processBlockDims(dimOrBlockDims)
    I = _np.identity(gateDim,'d')
    
    if from_basis == "std":
        if to_basis == "std":  m = I
        elif to_basis == "gm": m = _np.linalg.inv(gm_to_std_transform_matrix(dimOrBlockDims))
        elif to_basis == "pp": m = _np.linalg.inv(pp_to_std_transform_matrix(dimOrBlockDims))
        elif to_basis == "qt": m = _np.linalg.inv(qt_to_std_transform_matrix(dimOrBlockDims))
        else: raise ValueError("Invalid 'to_basis': %s" % to_basis)
        
    elif from_basis == "gm":
        if to_basis == "std":
            m = gm_to_std_transform_matrix(dimOrBlockDims)
        elif to_basis == "gm": m = I
        elif to_basis == "pp":
            m = _np.dot(_np.linalg.inv(pp_to_std_transform_matrix(dimOrBlockDims)),
                        gm_to_std_transform_matrix(dimOrBlockDims))
        elif to_basis == "qt":
            m = _np.dot(_np.linalg.inv(qt_to_std_transform_matrix(dimOrBlockDims)),
                        gm_to_std_transform_matrix(dimOrBlockDims))                                                
        else: raise ValueError("Invalid 'to_basis': %s" % to_basis)

    elif from_basis == "pp":
        if to_basis == "std":
            m = pp_to_std_transform_matrix(dimOrBlockDims)
        elif to_basis == "gm":
            m = _np.dot(_np.linalg.inv(gm_to_std_transform_matrix(dimOrBlockDims)),
                        pp_to_std_transform_matrix(dimOrBlockDims))
        elif to_basis == "pp": m = I
        elif to_basis == "qt":
            m = _np.dot(_np.linalg.inv(qt_to_std_transform_matrix(dimOrBlockDims)),
                        pp_to_std_transform_matrix(dimOrBlockDims))
        else: raise ValueError("Invalid 'to_basis': %s" % to_basis)

    elif from_basis == "qt":
        if to_basis == "std":
            m = qt_to_std_transform_matrix(dimOrBlockDims)
        elif to_basis == "gm":
            m = _np.dot(_np.linalg.inv(gm_to_std_transform_matrix(dimOrBlockDims)),
                        qt_to_std_transform_matrix(dimOrBlockDims))
        elif to_basis == "pp":
            m = _np.dot(_np.linalg.inv(pp_to_std_transform_matrix(dimOrBlockDims)),
                        qt_to_std_transform_matrix(dimOrBlockDims))
        elif to_basis == "qt": m = I
        else: raise ValueError("Invalid 'to_basis': %s" % to_basis)

    else: raise ValueError("Invalid 'from_basis': %s" % from_basis)
    return m


def change_basis(mx, from_basis, to_basis, dimOrBlockDims=None):
    """
    Convert a gate matrix from one basis of a density matrix space
    to another.

    Parameters
    ----------
    mx : numpy array
        The gate matrix (a 2D square array) in the `from_basis` basis.

    from_basis, to_basis: {'std', 'gm', 'pp', 'qt'}
        The source and destination basis, respectively.  Allowed
        values are Matrix-unit (std), Gell-Mann (gm), Pauli-product (pp),
        and Qutrit (qt).

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
    if from_basis == "std":
        if to_basis == "std": return mx.copy()
        elif to_basis == "gm": fn = std_to_gm
        elif to_basis == "pp": fn = std_to_pp
        elif to_basis == "qt": fn = std_to_qt
        else: raise ValueError("Invalid 'to_basis': %s" % to_basis)
        
    elif from_basis == "gm":
        if to_basis == "std":  fn = gm_to_std
        elif to_basis == "gm": return mx.copy()
        elif to_basis == "pp": fn = gm_to_pp
        elif to_basis == "qt": fn = gm_to_qt
        else: raise ValueError("Invalid 'to_basis': %s" % to_basis)

    elif from_basis == "pp":
        if to_basis == "std":  fn = pp_to_std
        elif to_basis == "gm": fn = pp_to_gm
        elif to_basis == "pp": return mx.copy()
        elif to_basis == "qt": fn = pp_to_qt
        else: raise ValueError("Invalid 'to_basis': %s" % to_basis)

    elif from_basis == "qt":
        if to_basis == "std":  fn = qt_to_std
        elif to_basis == "gm": fn = qt_to_gm
        elif to_basis == "pp": fn = qt_to_pp
        elif to_basis == "qt": return mx.copy()
        else: raise ValueError("Invalid 'to_basis': %s" % to_basis)

    else: raise ValueError("Invalid 'from_basis': %s" % from_basis)
    return fn(mx, dimOrBlockDims)
