from collections  import OrderedDict
from numpy.linalg import inv as _inv
import numpy as _np

from memoize import memoize

class Basis(object):
    BasisDict = dict()

    def __init__(self, name, matrices, longname=None):
        self.name = name
        if longname is None:
            self.longname = self.name
        else:
            self.longname = longname

        self._mxDict = OrderedDict()
        assert len(matrices) > 0, 'Need at least one matrix in basis'
        for i, mx in enumerate(matrices):
            if isinstance(mx, tuple):
                label, mx = mx
            else:
                label = 'M{}'.format(i)
            self._mxDict[label] = mx
        Basis.add(self)

    def __hash__(self):
        return hash((self.name, self.shape()))

    def __str__(self):
        return '{} Basis : {}'.format(self.longname, ', '.join(self.labels()))

    @memoize
    def is_normalized(self):
        for mx in self.matrices():
            t = _np.trace(_np.dot(mx, mx))
            t = _np.real(t)
            if t != 0:
                return False
        return True

    @memoize
    def shape(self):
        return self.matrices()[0].shape

    @memoize
    def matrices(self):
        return list(self._mxDict.values())

    @memoize
    def labels(self):
        return list(self._mxDict.keys())

    @memoize
    def get_to_std(self):
        return _np.column_stack([mx.flatten() for mx in self.matrices()])

    @memoize
    def get_from_std(self):
        return _inv(self._get_to_std())

    @staticmethod
    def add(basis):
        Basis.BasisDict[(basis.name, basis.shape())] = basis

    @staticmethod
    def get(basisname, shape=None):
        if shape is None:
            shape = (2, 2)
        return Basis.BasisDict[(basisname, shape)]

@memoize
def get_conversion_mx(from_basis, to_basis):
    return _np.dot(from_basis.get_to_std(), to_basis.get_to_std())

def change_basis(mx, from_basis, to_basis):
    return _np.dot(get_conversion_mx(from_basis, to_basis), mx)

def build_basis(basis, shape=None):
    if isinstance(basis, Basis):
        return basis
    else:
        return Basis.get(basis, shape)
