from collections  import OrderedDict, namedtuple
from functools    import wraps, partial

import numbers as _numbers
import collections as _collections

from numpy.linalg import inv as _inv
import numpy as _np

from pprint import pprint

import math

from ..tools.memoize       import memoize
from ..tools.parameterized import parameterized
from ..tools import matrixtools as _mt

from .dim import Dim

DefaultBasisInfo = namedtuple('DefaultBasisInfo', ['constructor', 'longname', 'real'])

class Basis(object):
    DefaultInfo = dict()
    CustomCount  = 0      # The number of custom bases

    def __init__(self, name=None, dim=None, matrices=None, longname=None, real=None, **kwargs):
        self.name, self.matrixGroups = _build_matrix_groups(name, dim, matrices, **kwargs)
        # Shorthand for retrieving a default value from the Basis.DefaultInfo dict
        def get_info(attr, default):
            try:
                return getattr(Basis.DefaultInfo[self.name], attr)
            except KeyError:
                return default

        if real is None:
            real     = get_info('real', default=True)
        if longname is None:
            longname = get_info('longname', default=self.name)

        blockDims = [int(math.sqrt(len(group))) for group in self.matrixGroups]

        # Equivalent to matrices for non-composite bases
        self.matrices     = self.get_composite_matrices() 
        self.dim          = Dim(blockDims)
        self.name         = name
        self.longname     = longname
        self.real         = real

        self.labels = ['M{}{}'.format(self.name, i) for i in range(len(self.matrices))]

    def __str__(self):
        return '{} Basis : {}'.format(self.longname, ', '.join(self.labels))

    def __repr__(self):
        return str(self)

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

    def stdmx_to_vec(self, m):
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
        v = _np.empty((dim**2,1))
        for i, mx in enumerate(self.matrices):
            if self.real:
                v[i,0] = _np.real(_mt.trace(_np.dot(mx,m)))
            else:
                v[i,0] = _mt.trace(_np.dot(mx,m))
        return v

    def vec_to_stdmx(self, v, keep_complex=False):
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
        ret = _np.zeros( (dim, dim), 'complex' )
        for i, mx in enumerate(self.matrices):
            if keep_complex:
                ret += v[i]*mx
            else:
                ret += float(v[i])*mx
        return ret

    def transform_matrix(self, to_basis, dimOrBlockDims):
        to_basis = Basis(to_basis, dimOrBlockDims)
        return _np.dot(to_basis.get_from_std(), self.get_to_std())

    def change_basis(self, mx, to_basis, dimOrBlockDims=None):
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
        if dimOrBlockDims is None:
            dimOrBlockDims = int(round(_np.sqrt(mx.shape[0])))
            assert( dimOrBlockDims**2 == mx.shape[0] )
        dim = Dim(dimOrBlockDims)
        to_basis   = Basis(to_basis,   dim)
        if self.name == to_basis.name:
            return mx.copy()

        if len(mx.shape) not in [1, 2]:
            raise ValueError("Invalid dimension of object - must be 1 or 2, i.e. a vector or matrix")
        toMx   = self.transform_matrix(to_basis, dim)
        fromMx = to_basis.transform_matrix(self, dim)
        if len(mx.shape) == 2 and mx.shape[0] == mx.shape[1]:
            ret = _np.dot(toMx, _np.dot(mx, fromMx))
        else:
            ret = _np.dot(toMx, mx)
        if not to_basis.real:
            return ret
        if _np.linalg.norm(_np.imag(ret)) > 1e-8:
            raise ValueError("Array has non-zero imaginary part (%g) after basis change (%s to %s)!\n%s" %
                             (_np.linalg.norm(_np.imag(ret)), self, to_basis, ret))
        return _np.real(ret)

    @memoize
    def is_normalized(self):
        for mx in self.matrices:
            t = _np.trace(_np.dot(mx, mx))
            t = _np.real(t)
            if t != 0:
                return False
        return True

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
        print(self)
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

def _build_composite_basis(bases):
    '''
    Build a composite basis from a list of tuples or Basis objects 
      (or a list of mixed tuples and Basis objects)
    '''
    assert len(bases) > 0, 'Need at least one basis-dim pair to compose'
    bases = [Basis(*item) for item in bases]

    matrixGroups  = [basis.matrices for basis in bases]
    name          = ','.join(basis.name for basis in bases)
    longname      = ','.join(basis.longname for basis in bases)
    real          = all(basis.real for basis in bases)

    composite = Basis(matrices=matrixGroups, name=name, longname=longname, real=real)
    return composite

def basis_transform_matrix(from_basis, to_basis, dimOrBlockDims):
    from_basis = Basis(from_basis, dimOrBlockDims)
    return from_basis.transform_matrix(to_basis, dimOrBlockDims)

def change_basis(mx, from_basis, to_basis, dimOrBlockDims=None):
    if dimOrBlockDims is None:
        dimOrBlockDims = int(round(_np.sqrt(mx.shape[0])))
        assert( dimOrBlockDims**2 == mx.shape[0] )
    from_basis = Basis(from_basis, dimOrBlockDims)
    return from_basis.change_basis(mx, to_basis, dimOrBlockDims)

# Allow flexible basis building without cluttering the basis __init__ method with instance checking
def _build_matrix_groups(name=None, dim=None, matrices=None, **kwargs):
    if isinstance(name, Basis):
        matrixGroups = name.matrixGroups
        name         = name.name
    elif isinstance(name, list):
        basis = _build_composite_basis(name)
        matrixGroups = basis.matrixGroups
        name         = basis.name
    else:
        if matrices is None: # built by name and dim, ie Basis('pp', 4)
            assert name is not None, \
                    'If matrices is none, name must be supplied to Basis.__init__'
            matrices = _build_default_matrix_groups(name, dim, **kwargs)

        if len(matrices) > 0:
            first = matrices[0]
            if isinstance(first, tuple) or \
                    isinstance(first, Basis):
                matrixGroups = build_composite_basis(matrices) # really list of Bases or basis tuples
            elif not isinstance(first, list):                  # If not nested lists (really just 'matrices')
                matrixGroups = [matrices]                      # Then nest
            else:
                matrixGroups = matrices                        # Given as nested lists
            if name is None:
                name = 'CustomBasis_{}'.format(Basis.CustomCount)
                CustomCount += 1
        else:
            matrixGroups = []
    return name, matrixGroups


def _build_default_matrix_groups(name, dim, **kwargs):
    if name == 'unknown':
        return []
    if name not in Basis.DefaultInfo:
        raise NotImplementedError('No instructions to create supposed \'default\' basis:  {} of dim {}'.format(
            name, dim))
    f = Basis.DefaultInfo[name].constructor
    matrixGroups = []
    dim = Dim(dim)
    for blockDim in dim.blockDims:
        group = f(blockDim, **kwargs)
        if isinstance(group, Basis):
            matrixGroups.append(group.matrices)
        else:
            matrixGroups.append(f(blockDim, **kwargs))
    return matrixGroups

@parameterized # this "decorator" takes additional arguments (other than just f)
def basis_constructor(f, name, longname, real=True):
    # Really this decorator only saves f to a dictionary for constructing default bases
    #    => No wrapper is created:
    Basis.DefaultInfo[name] = DefaultBasisInfo(f, longname, real)
    return f

def basis_matrices(name, dimOrBlockDims, maxWeight=None):
    '''
    Get the elements of the specifed basis-type which
    spans the density-matrix space given by dimOrBlockDims.

    Parameters
    ----------
    name : {'std', 'gm', 'pp', 'qt'}
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
    '''
    if name not in Basis.DefaultInfo:
        raise NotImplementedError('No instructions to create supposed \'default\' basis:  {} of dim {}'.format(
            name, dimOrBlockDims))
    f = Basis.DefaultInfo[name].constructor

    if maxWeight is None:
        return f(dimOrBlockDims)
    else:
        return f(dimOrBlockDims, maxWeight=maxWeight)

def resize_mx(mx, dimOrBlockDims, resize=None, startBasis='std', endBasis='std'):
    '''
    Convert a gate matrix in an arbitrary basis of either a "direct-sum" or the embedding
    space to a matrix in the same basis in the opposite space.

    Parameters
    ----------
    mx: numpy array
        Matrix of size N x N, where N is the dimension
        of the density matrix space, i.e. sum( dimOrBlockDims_i^2 )

    dimOrBlockDims : int or list of ints
        Structure of the density-matrix space.

    Returns
    -------
    numpy array
        A M x M matrix, where M is the dimension of the
        embedding density matrix space, i.e.
        sum( dimOrBlockDims_i )^2
    '''
    if dimOrBlockDims is None:
        return mx
    elif isinstance(dimOrBlockDims, _numbers.Integral):
        assert(mx.shape == (dimOrBlockDims, dimOrBlockDims) )
        return mx
    else:
        assert resize is not None and resize in ['expand', 'contract'], 'Incorrect resize argument: {}'.format(resize)
        start = change_basis(mx, startBasis, 'std', dimOrBlockDims)

        stdBasis = Basis('std', dimOrBlockDims)
        if resize == 'expand':
            mid = _np.dot(stdBasis.get_contract_mx(), _np.dot(start, stdBasis.get_expand_mx()))
        elif resize == 'contract':
            mid = _np.dot(stdBasis.get_expand_mx(), _np.dot(start, stdBasis.get_contract_mx()))
        # No else case needed, see assert
        return change_basis(mid, 'std', endBasis, dimOrBlockDims)

expand_from_direct_sum_mx = partial(resize_mx, resize='expand')
contract_to_direct_sum_mx = partial(resize_mx, resize='contract')
