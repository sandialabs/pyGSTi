

import numpy as _np
from .linearop import LinearOperator as _LinearOperator
from ...evotypes import Evotype as _Evotype
from ...models import statespace as _statespace


class CliffordOp(_LinearOperator):
    """
    A Clifford operation, represented via a symplectic matrix.

    Parameters
    ----------
    unitary : numpy.ndarray
        The unitary action of the clifford operation.

    symplecticrep : tuple, optional
        A (symplectic matrix, phase vector) 2-tuple specifying the pre-
        computed symplectic representation of `unitary`.  If None, then
        this representation is computed automatically from `unitary`.

    basis : Basis or {'pp','gm','std'}, optional
        The basis used to construct the Hilbert-Schmidt space representation
        of this state as a super-operator.

    evotype : Evotype or str
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, unitary, symplecticrep=None, basis='pp', evotype='default', state_space=None):
        self.unitary = unitary
        assert(self.unitary is not None), "Must supply `unitary` argument!"
        U = self.unitary.to_dense() if isinstance(self.unitary, _LinearOperator) else self.unitary

        state_space = _statespace.default_space_for_udim(U.shape[0]) if (state_space is None) \
            else _statespace.StateSpace.cast(state_space)

        evotype = _Evotype.cast(evotype)
        rep = evotype.create_clifford_rep(U, symplecticrep, basis, state_space)
        _LinearOperator.__init__(self, rep, evotype)

    #NOTE: if this operation had parameters, we'd need to clear inv_smatrix & inv_svector
    # whenever the smatrix or svector changed, respectively (probably in from_vector?)

    def __str__(self):
        """ Return string representation """
        return str(self._rep)
