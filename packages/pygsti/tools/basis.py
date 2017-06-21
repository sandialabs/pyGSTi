from collections  import OrderedDict
from functools    import wraps

import numbers as _numbers
import collections as _collections

from numpy.linalg import inv as _inv
import numpy as _np

from .memoize       import memoize
from .parameterized import parameterized

class Basis(object):
    Constructors = dict()

    def __init__(self, name, matrices_creator, dim, longname=None, real=True):
        _, gateDim, blockDims = process_block_dims(dim)

        self.matrixGroups = []
        for blockDim in blockDims:
            self.matrixGroups.append(matrices_creator(blockDim))
        self.matrices = [mx for mxlist in self.matrixGroups for mx in mxlist]
        '''
        print('blockdims:')
        print(blockDims)
        print('blockdims squared:')
        print([bd ** 2 for bd in blockDims])
        '''
        
        #assert len(matrices) > 0, 'Need at least one matrix in basis'

        self.name = name
        self.real = real
        self.dim  = dim
        self.gateDim = gateDim
        if longname is None:
            self.longname = self.name
        else:
            self.longname = longname

        self._mxDict = OrderedDict()
        for i, mx in enumerate(self.matrices):
            if isinstance(mx, tuple):
                label, mx = mx
            else:
                label = 'M{}'.format(i)
            self._mxDict[label] = mx
        self.matrices = list(self._mxDict.values())
        self.labels = list(self._mxDict.keys())

    def __str__(self):
        return '{} Basis (dim {}) : {}'.format(self.longname, self.dim, ', '.join(self.labels))

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
        return hash((self.name, self.shape))

    #@memoize
    def is_normalized(self):
        for mx in self.matrices:
            t = _np.trace(_np.dot(mx, mx))
            t = _np.real(t)
            if t != 0:
                return False
        return True

    #@memoize
    def get_to_std(self):
        toStd = _np.zeros( (self.gateDim, self.gateDim), 'complex' )

        #Since a multi-block basis is just the direct sum of the individual block bases,
        # transform mx is just the transfrom matrices of the individual blocks along the
        # diagonal of the total basis transform matrix

        start = 0
        for mxs in self.matrixGroups:

            l = len(mxs)
            for j, mx in enumerate(mxs):
                toStd[start:start+l,start+j] = mx.flatten()
            start += l 

        assert(start == self.gateDim)
        return toStd

    #@memoize
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

def process_block_dims(dimOrBlockDims):
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
    if isinstance(dimOrBlockDims, str):
        raise TypeError("Invalid dimOrBlockDims = %s" % str(dimOrBlockDims))
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
        raise TypeError("Invalid dimOrBlockDims = %s" % str(dimOrBlockDims))

    return dmDim, gateDim, blockDims

def get_conversion_mx(from_basis, to_basis, dimOrBlockDims):
    from_basis = build_basis(from_basis, dimOrBlockDims)
    to_basis   = build_basis(to_basis, dimOrBlockDims)

    return _np.dot(to_basis.get_from_std(), from_basis.get_to_std())

def change_basis(mx, from_basis, to_basis, dimOrBlockDims):
    from_basis = build_basis(from_basis, dimOrBlockDims)
    to_basis   = build_basis(to_basis, dimOrBlockDims)

    if len(mx.shape) == 2 and mx.shape[0] == mx.shape[1]:
        ret = _np.dot(to_basis.get_from_std(), _np.dot(mx, from_basis.get_to_std()))
    elif len(mx.shape) == 1 or \
        (len(mx.shape) == 2 and mx.shape[1] == 1):
        ret = _np.dot(to_basis.get_from_std(), _np.dot(from_basis.get_to_std(), mx))
    else:
        raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")
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
        return Basis(name, f, dim=args[0], longname=longname, real=real)
    Basis.Constructors[name] = wrapper
    return wrapper
