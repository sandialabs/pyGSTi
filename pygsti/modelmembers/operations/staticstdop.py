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
from ...tools import internalgates as _igts
from .linearop import LinearOperator as _LinearOperator


class StaticStandardOp(_LinearOperator):
    """
    An operation that is completely fixed, or "static" (i.e. that posesses no parameters)
    that can be constructed from "standard" gate names (as defined in pygsti.tools.internalgates).

    Parameters
    ----------
    name : str
        Standard gate name

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """
    def __init__(self, name, evotype="default"):
        self.name = name
        evotype = _Evotype.cast(evotype)
        rep = evotype.create_standard_rep(name)
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

    def __str__(self):
        s = "%s with name %s and evotype %s\n" % (self.__class__.__name__, self.name, self._evotype)
        #TODO: move this to __str__ methods of reps??
        #if self._evotype in ['statevec', 'densitymx', 'svterm', 'cterm']:
        #    s += _mt.mx_to_string(self.base, width=4, prec=2)
        #elif self._evotype == 'chp':
        #    s += 'CHP operations: ' + ','.join(self._rep.chp_ops) + '\n'
        return s
