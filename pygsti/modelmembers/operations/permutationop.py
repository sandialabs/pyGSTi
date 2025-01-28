from pygsti.modelmembers.operations import DenseOperator
from pygsti.baseobjs.basisconstructors import pp_labels
import numpy as _np

class PermutationOperator(DenseOperator):

    def __init__(self, perm: _np.ndarray):
        dim = perm.size
        mx = _np.eye(dim)
        mx = mx[perm,:]
        super().__init__(mx, 'pp', 'densitymx')
        self._perm = perm

    @property
    def num_params(self):
        return 0
    
    def to_vector(self):
        return _np.array([])

    def from_vector(self, v, close=False, dirty_value=True):
        if v.size > 0:
            raise ValueError()
        return
    
    def transform(self, S):
        raise NotImplementedError("PermutationOperator cannot be transformed!")
    
    def inverse_operator(self):
        iperm = self._perm.copy()
        iperm[iperm] = _np.arange(self.dim)
        return PermutationOperator(iperm)

    @staticmethod
    def pp_braiding_operators(subsystem_perm):
        subsystem_perm = _np.atleast_1d(subsystem_perm).copy()
        n_qubits = subsystem_perm.size
        labels = _np.array(pp_labels(2**n_qubits))
        braid_labels = _np.array([''.join([ell[i] for i in subsystem_perm]) for ell in labels])
        braid_perm = []
        for bl in braid_labels:
            loc = _np.where(labels == bl)[0].item()
            braid_perm.append(loc)
        braid_perm = _np.array(braid_perm)
        pop = PermutationOperator(braid_perm)
        ipop = pop.inverse_operator()
        return pop, ipop
