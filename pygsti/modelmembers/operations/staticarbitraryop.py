"""
The StaticArbitraryOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from .denseop import DenseOperator as _DenseOperator


class StaticArbitraryOp(_DenseOperator):
    """
    An operation matrix that is completely fixed, or "static" (i.e. that posesses no parameters).

    Parameters
    ----------
    m : array_like or LinearOperator
        a square 2D array-like or LinearOperator object representing the operation action.
        The shape of m sets the dimension of the operation.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    state_space : StateSpace, optional
        The state space for this operation.  If `None` a default state space
        with the appropriate number of qubits is used.
    """

    def __init__(self, m, evotype="default", state_space=None):
        _DenseOperator.__init__(self, m, evotype, state_space)
        #(default DenseOperator/LinearOperator methods implement an object with no parameters)
