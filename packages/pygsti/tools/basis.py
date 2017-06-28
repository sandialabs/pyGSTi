from collections  import OrderedDict
from functools    import wraps

import numbers as _numbers
import collections as _collections

from numpy.linalg import inv as _inv
import numpy as _np

from pprint import pprint

from .memoize       import memoize
from .parameterized import parameterized

from .dim import Dim

class Basis(object):
    Constructors = dict()

    def __init__(self, name, f, dim, longname=None, real=True):
        self.dim = Dim(dim)

        self.matrixGroups = []
        for blockDim in self.dim.blockDims:
            self.matrixGroups.append(f(blockDim))
        self.matrices  = f(dim)
        self.name      = name
        self.real      = real

        if longname is None:
            self.longname = self.name
        else:
            self.longname = longname

        self.labels = ['M{}{}'.format(self.name, i) for i in range(len(self.matrices))]

    def __str__(self):
        return '{} Basis : {}'.format(self.longname, ', '.join(self.labels))

    def __getitem__(self, index):
        return self.matrices[index]

    def __setitem__(self, index, value):
        self.matrices[index] = value

    def __len__(self):
        return len(self.matrices)

    def __eq__(self, other):
        if isinstance(other, Basis):
            return _np.array_equal(self.matrices, other.matrices)
        else:
            return _np.array_equal(self.matrices, other)

    def __hash__(self):
        return hash((self.name, self.dim))

    @memoize
    def is_normalized(self):
        for mx in self.matrices:
            t = _np.trace(_np.dot(mx, mx))
            t = _np.real(t)
            if t != 0:
                return False
        return True

    @memoize
    def get_composite_matrices(self):
        '''
        Build the large composite matrices of a composite basis
        ie for std basis with dim [2, 1], build
        [[1 0 0]  [[0 1 0]  [[0 0 0]  [[0 0 0]  [[0 0 0]
         [0 0 0]   [0 0 0]   [1 0 0]   [0 1 0]   [0 0 0]
         [0 0 0]], [0 0 0]], [0 0 0]], [0 0 0]], [0 0 1]]
        For a non composite basis, this just returns the basis matrices
        '''
        compMxs = []
        start = 0
        length = sum(mxs[0].shape[0] for mxs in self.matrixGroups)
        for mxs in self.matrixGroups:
            d = mxs[0].shape[0]
            for mx in mxs:
                compMx = _np.zeros((length, length), 'complex' )
                compMx[start:start+d,start:start+d] = mx
                compMxs.append(compMx)
            start += d
        assert(start == length)
        return compMxs

    @memoize
    def get_expand_mx(self):
        x = sum(len(mxs) for mxs in self.matrixGroups)
        y = sum(mxs[0].shape[0] for mxs in self.matrixGroups) ** 2
        expandMx = _np.zeros((x, y), 'complex')
        start = 0
        for i, compMx in enumerate(self.get_composite_matrices()):
            assert(len(compMx.flatten()) == y), '{} != {}'.format(len(compMx.flatten()), y)
            expandMx[i,0:y] = compMx.flatten()
        return expandMx

    @memoize
    def get_contract_mx(self):
        return self.get_expand_mx().T

    @memoize
    def get_to_std(self):
        toStd = _np.zeros((self.dim.gateDim, self.dim.gateDim), 'complex' )
        #Since a multi-block basis is just the direct sum of the individual block bases,
        # transform mx is just the transfrom matrices of the individual blocks along the
        # diagonal of the total basis transform matrix

        start = 0
        for mxs in self.matrixGroups:
            l = len(mxs)
            for j, mx in enumerate(mxs):
                toStd[start:start+l,start+j] = mx.flatten()
            start += l 
        assert(start == self.dim.gateDim)
        return toStd

    @memoize
    def get_from_std(self):
        return _inv(self.get_to_std())

    @staticmethod
    def create(basisname, dim, *args, **kwargs):
        if basisname in Basis.Constructors:
            return Basis.Constructors[basisname](dim, *args, **kwargs)
        raise NotImplementedError('No instructions to create basis: {} {}'.format(basisname, dim))

def build_basis(basis, dimOrBlockDims):
    if isinstance(basis, Basis):
        return basis
    else:
        return Basis.create(basis, dimOrBlockDims)

def basis_transform_matrix(from_basis, to_basis, dimOrBlockDims):
    from_basis = build_basis(from_basis, dimOrBlockDims)
    to_basis   = build_basis(to_basis, dimOrBlockDims)

    return _np.dot(to_basis.get_from_std(), from_basis.get_to_std())

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
    if from_basis == to_basis:
        return mx.copy()

    if dimOrBlockDims is None:
        dimOrBlockDims = int(round(_np.sqrt(mx.shape[0])))
        assert( dimOrBlockDims**2 == mx.shape[0] )

    dim = Dim(dimOrBlockDims)

    from_basis = build_basis(from_basis, dim)
    to_basis   = build_basis(to_basis,   dim)
    if len(mx.shape) not in [1, 2]:
        raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")
    toMx   = basis_transform_matrix(from_basis, to_basis, dim)
    fromMx = basis_transform_matrix(to_basis, from_basis, dim)
    if len(mx.shape) == 2 and mx.shape[0] == mx.shape[1]:
        ret = _np.dot(toMx, _np.dot(mx, fromMx))
    else:
        ret = _np.dot(toMx, mx)
    if not to_basis.real:
        return ret
    if _np.linalg.norm(_np.imag(ret)) > 1e-8:
        raise ValueError("Array has non-zero imaginary part (%g) after basis change (%s to %s)!\n%s" %
                         (_np.linalg.norm(_np.imag(ret)), from_basis, to_basis, ret))
    return _np.real(ret)

@parameterized
def basis_constructor(f, name, longname, real=True):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return Basis(name, f, *args, longname=longname, real=real)
    Basis.Constructors[name] = wrapper
    return wrapper
