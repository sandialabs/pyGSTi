import numpy as _np
from .state import State as _State
from .densestate import DenseState as _DenseState
from ...evotypes import Evotype as _Evotype
from ...models import statespace as _statespace


class StaticState(_DenseState):
    """
    A SPAM vector that is completely fixed, or "static" (i.e. that posesses no parameters).

    Parameters
    ----------
    vec : array_like or SPAMVec
        a 1D numpy array representing the SPAM operation.  The
        shape of this array sets the dimension of the SPAM op.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, vec, evotype="default", state_space=None):
        vec = _State._to_vector(vec)

        state_space = _statespace.default_space_for_dim(vec.shape[0]) if (state_space is None) \
            else _statespace.StateSpace.cast(state_space)

        evotype = _Evotype.cast(evotype)
        rep = evotype.create_dense_state_rep(vec, state_space)
        _DenseState.__init__(self, rep, evotype, rep.base)

    def _base_1d_has_changed(self):
        self._rep.base_has_changed()
