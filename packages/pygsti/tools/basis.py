from collections  import OrderedDict
from functools    import wraps

from numpy.linalg import inv as _inv
import numpy as _np

from .memoize       import memoize
from .parameterized import parameterized

class Basis(object):
    Constructors = dict()

    def __init__(self, name, matrices, longname=None):
        assert len(matrices) > 0, 'Need at least one matrix in basis'

        self.shape = matrices[0].shape # wont change ( I think? )
        self.name = name
        if longname is None:
            self.longname = self.name
        else:
            self.longname = longname

        self._mxDict = OrderedDict()
        for i, mx in enumerate(matrices):
            if isinstance(mx, tuple):
                label, mx = mx
            else:
                label = 'M{}'.format(i)
            self._mxDict[label] = mx
        self.matrices = list(self._mxDict.values())
        self.labels = list(self._mxDict.keys())

    def __str__(self):
        return '{} Basis : {}'.format(self.longname, ', '.join(self.labels()))

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
    def get_to_std(self):
        return _np.column_stack([mx.flatten() for mx in self.matrices])

    @memoize
    def get_from_std(self):
        return _inv(self.get_to_std())

    @staticmethod
    def create(basisname, dim):
        if basisname in Basis.Constructors:
            return Basis.Constructors[basisname](dim)
        raise NotImplementedError('No instructions to create basis: {} {}'.format(basisname, dim))

#@memoize
def get_conversion_mx(from_basis, to_basis):
    return _np.dot(to_basis.get_from_std(), from_basis.get_to_std())

def build_basis(basis, dimOrBlockDims=None):
    if isinstance(basis, Basis):
        return basis
    else:
        return Basis.create(basis, dimOrBlockDims)

def change_basis(mx, from_basis, to_basis, dimOrBlockDims):
    if isinstance(dimOrBlockDims, list):
        dimOrBlockDims = tuple(dimOrBlockDims)
    from_basis = build_basis(from_basis, dimOrBlockDims)
    to_basis   = build_basis(to_basis, dimOrBlockDims)
    return _np.dot(get_conversion_mx(from_basis, to_basis), mx)

@parameterized
def basis_constructor(f, name):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return Basis(name, f(*args, **kwargs))
    Basis.Constructors[name] = wrapper
    return wrapper
