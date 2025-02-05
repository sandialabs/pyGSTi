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
        iperm = PermutationOperator.inv_perm(self._perm)
        return PermutationOperator(iperm)
    
    @staticmethod
    def inv_perm(perm):
        iperm = perm.copy()
        iperm[iperm] = _np.arange(iperm.size)
        return iperm
    
    @staticmethod
    def perm_from_mx(mx):
        perm = _np.array([_np.where(row == 1)[0][0] for row in mx])
        return perm
    
    ## We need to implement this in order to deserialize.
    @classmethod
    def _from_memoized_dict(cls, mm_dict, serial_memo):
        mx = cls._decodemx(mm_dict['dense_matrix'])
        mx = mx.squeeze()
        # state_space = _statespace.StateSpace.from_nice_serialization(mm_dict['state_space'])
        # basis = _Basis.from_nice_serialization(mm_dict['basis']) if (mm_dict['basis'] is not None) else None
        # return cls(m, basis, mm_dict['evotype'], state_space)
        perm = PermutationOperator.perm_from_mx(mx)
        return PermutationOperator(perm)

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
