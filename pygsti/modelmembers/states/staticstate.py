import numpy as _np
from .state import State as _State
from .densestate import DenseState as _DenseState
from ...evotypes import Evotype as _Evotype


class StaticState(_DenseState):
    """
    A SPAM vector that is completely fixed, or "static" (i.e. that posesses no parameters).

    Parameters
    ----------
    vec : array_like or SPAMVec
        a 1D numpy array representing the SPAM operation.  The
        shape of this array sets the dimension of the SPAM op.

    evotype : {"densitymx", "statevec"}
        the evolution type being used.
    """

    def __init__(self, vec, evotype="auto"):
        vec = _State._to_vector(vec)

        #REMOVE
        #if evotype == "auto":
        #    evotype = "statevec" if _np.iscomplexobj(vec) else "densitymx"
        #elif evotype == "statevec":
        #    vec = _np.asarray(vec, complex)  # ensure all statevec vecs are complex (densitymx could be either?)
        #assert(evotype in ("statevec", "densitymx")), \
        #    "Invalid evolution type '%s' for %s" % (evotype, self.__class__.__name__)

        evotype = _Evotype.cast(evotype)
        rep = evotype.create_dense_state_rep(vec)
        _DenseState.__init__(self, rep, evotype, rep.base)

    def _base_1d_has_changed(self):
        self._rep.base_has_changed()
