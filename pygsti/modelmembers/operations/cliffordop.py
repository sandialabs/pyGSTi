

import numpy as _np
from .linearop import LinearOperator as _LinearOperator
from ...evotypes import Evotype as _Evotype


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

    evotype : {"stabilizer"}
        The evolution type.
    """

    def __init__(self, unitary, symplecticrep=None, evotype='default'):
        """
        Creates a new CliffordOp from a unitary operation.

        Note: while the clifford operation is held internally in a symplectic
        representation, it is also be stored as a unitary (so the `unitary`
        argument is required) for keeping track of global phases when updating
        stabilizer frames.

        If a non-Clifford unitary is specified, then a ValueError is raised.

        Parameters
        ----------
        unitary : numpy.ndarray
            The unitary action of the clifford operation.

        symplecticrep : tuple, optional
            A (symplectic matrix, phase vector) 2-tuple specifying the pre-
            computed symplectic representation of `unitary`.  If None, then
            this representation is computed automatically from `unitary`.

        evotype : Evotype or str
            The evolution type.  The special value `"default"` is equivalent
            to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
        """
        #self.superop = superop
        self.unitary = unitary
        assert(self.unitary is not None), "Must supply `unitary` argument!"
        U = self.unitary.to_dense() if isinstance(self.unitary, _LinearOperator) else self.unitary

        evotype = _Evotype.cast(evotype)
        rep = evotype.create_clifford_rep(U, symplecticrep)
        _LinearOperator.__init__(self, rep, evotype)

    #NOTE: if this operation had parameters, we'd need to clear inv_smatrix & inv_svector
    # whenever the smatrix or svector changed, respectively (probably in from_vector?)

    def __str__(self):
        """ Return string representation """
        return str(self._rep)
