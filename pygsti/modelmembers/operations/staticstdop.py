"""
The StaticStandardOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from ...evotypes import Evotype as _Evotype
from ...tools import internalgates as _itgs
from ...models import statespace as _statespace
from .linearop import LinearOperator as _LinearOperator


class StaticStandardOp(_LinearOperator):
    """
    An operation that is completely fixed, or "static" (i.e. that posesses no parameters)
    that can be constructed from "standard" gate names (as defined in pygsti.tools.internalgates).

    Parameters
    ----------
    name : str
        Standard gate name

    basis : Basis or {'pp','gm','std'}, optional
        The basis used to construct the Hilbert-Schmidt space representation
        of this state as a super-operator.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """
    def __init__(self, name, basis='pp', evotype="default", state_space=None):
        self.name = name

        #Create default state space if needed
        std_unitaries = _itgs.standard_gatename_unitaries()
        if name not in std_unitaries:
            raise ValueError("'%s' does not name a standard operation" % self.name)
        state_space = _statespace.default_space_for_udim(std_unitaries[name].shape[0]) if (state_space is None) \
            else _statespace.StateSpace.cast(state_space)

        evotype = _Evotype.cast(evotype)
        rep = evotype.create_standard_rep(name, basis, state_space)
        _LinearOperator.__init__(self, rep, evotype)

#TODO REMOVE
#    # TODO: This should not be necessary to define, but is here as a temporary measure
#    # This will likely be removed as "dense" is reworked in the evotype refactor
#    @property
#    def base(self):
#        """
#        The underlying dense process matrix.
#        """
#        if self._evotype in ['statevec', 'densitymx', 'svterm', 'cterm']:
#            return self._rep.base
#        else:
#            raise NotImplementedError('No base available for evotype "%s"' % self._evotype)

    def to_dense(self, on_space='minimal'):
        """
        Return the dense array used to represent this operation within its evolution type.

        Note: for efficiency, this doesn't copy the underlying data, so
        the caller should copy this data before modifying it.

        Parameters
        ----------
        on_space : {'minimal', 'Hilbert', 'HilbertSchmidt'}
            The space that the returned dense operation acts upon.  For unitary matrices and bra/ket vectors,
            use `'Hilbert'`.  For superoperator matrices and super-bra/super-ket vectors use `'HilbertSchmidt'`.
            `'minimal'` means that `'Hilbert'` is used if possible given this operator's evolution type, and
            otherwise `'HilbertSchmidt'` is used.

        Returns
        -------
        numpy.ndarray
        """
        return self._rep.to_dense(on_space)  # standard rep needs to implement this

    def __str__(self):
        s = "%s with name %s and evotype %s\n" % (self.__class__.__name__, self.name, self._evotype)
        #TODO: move this to __str__ methods of reps??
        #if self._evotype in ['statevec', 'densitymx', 'svterm', 'cterm']:
        #    s += _mt.mx_to_string(self.base, width=4, prec=2)
        #elif self._evotype == 'chp':
        #    s += 'CHP operations: ' + ','.join(self._rep.chp_ops) + '\n'
        return s
