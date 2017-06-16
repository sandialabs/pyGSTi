from collections  import OrderedDict
from functools    import wraps

from numpy.linalg import inv as _inv
import numpy as _np

from .memoize       import memoize
from .parameterized import parameterized

class Basis(object):
    BasisDict = dict()
    Constructors = dict()

    def __init__(self, name, matrices, longname=None):
        assert len(matrices) > 0, 'Need at least one matrix in basis'

        self.shape = matrices[0].shape # wont change ( I think? )
        '''
        if (name, self.shape) in Basis.BasisDict:
            self = Basis.get(name, shape=self.shape)
            return
        '''

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

        Basis.add(self)

    def __str__(self):
        return '{} Basis : {}'.format(self.longname, ', '.join(self.labels()))

    def __getitem__(self, index):
        return self.matrices[index]

    def __len__(self):
        return len(self.matrices)

    def __eq__(self, other):
        if isinstance(other, Basis):
            return self.matrices == other.matrices
        else:
            return self.matrices == other

    def __hash__(self):
        return hash((self.name, self.shape))

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
        return _inv(self._get_to_std())

    @staticmethod
    def add(basis):
        Basis.BasisDict[(basis.name, basis.shape)] = basis

    @staticmethod
    def get(basisname, shape=None):
        if shape is None:
            shape = (2, 2)
        if (basisname, shape) not in Basis.BasisDict:
            BasisDict[(basisname, shape)] = Basis.create(basisname, shape)
        return Basis.BasisDict[(basisname, shape)]

    @staticmethod
    def create(basisname, shape):
        raise NotImplementedError('Basis cannot create bases yet')

@memoize
def get_conversion_mx(from_basis, to_basis):
    return _np.dot(from_basis.get_to_std(), to_basis.get_to_std())

def build_basis(basis, shape=None):
    if isinstance(basis, Basis):
        return basis
    else:
        return Basis.get(basis, shape)

def change_basis(mx, from_basis, to_basis):
    from_basis = build_basis(from_basis, mx.shape)
    to_basis   = build_basis(to_basis, mx.shape)
    return _np.dot(get_conversion_mx(from_basis, to_basis), mx)

@parameterized
def basis_constructor(f, name):
    Basis.Constructors[name] = f
    @wraps(f)
    def wrapper(*args, **kwargs):
        return Basis(name, f(*args, **kwargs))
    return wrapper
