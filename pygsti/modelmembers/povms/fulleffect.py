
from .conjugatedeffect import ConjugatedStatePOVMEffect as _ConjugatedStatePOVMEffect
from ..states.fullstate import FullState as _FullState


class FullPOVMEffect(_ConjugatedStatePOVMEffect):
    """
    A "fully parameterized" effect vector where each element is an independent parameter.

    Parameters
    ----------
    vec : array_like or POVMEffect
        a 1D numpy array representing the SPAM operation.  The
        shape of this array sets the dimension of the SPAM op.

    evotype : {"densitymx", "statevec"}
        the evolution type being used.
    """

    def __init__(self, vec, evotype="auto"):
        _ConjugatedStatePOVMEffect.__init__(self, _FullState(vec, evotype))

    def set_dense(self, vec):
        """
        Set the dense-vector value of this SPAM vector.

        Attempts to modify this SPAM vector's parameters so that the raw
        SPAM vector becomes `vec`.  Will raise ValueError if this operation
        is not possible.

        Parameters
        ----------
        vec : array_like or SPAMVec
            A numpy array representing a SPAM vector, or a SPAMVec object.

        Returns
        -------
        None
        """
        self.state.set_dense(vec)
        self.dirty = True
