"""
The DepolarizeOp class and supporting functionality.
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
from .stochasticop import StochasticNoiseOp as _StochasticNoiseOp
from ...objects.basis import Basis as _Basis
from ...models import statespace as _statespace


class DepolarizeOp(_StochasticNoiseOp):
    """
    A depolarizing channel.

    Parameters
    ----------
    state_space : StateSpace, optional
        The state space for this operation.

    basis : Basis or {'pp','gm','qt'}, optional
        The basis to use, defining the "principle axes"
        along which there is stochastic noise.  While strictly unnecessary
        since all complete bases yield the same operator, this affects the
        underlying :class:`StochasticNoiseOp` and so is given as an option
        to the user.

    evotype : Evotype or str, optional
        The evolution type.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    initial_rate : float, optional
        the initial error rate.

    seed_or_state : float or RandomState, optional
            Random seed for RandomState (or directly provided RandomState)
            for sampling stochastic superoperators with the 'chp' evotype.
    """
    def __init__(self, state_space, basis="pp", evotype="default", initial_rate=0, seed_or_state=None):
        #TODO - need to fix CHP basis dimension issue (dim ~= statevec but acts as density mx)
        #if evotype == 'chp':
        #    assert (basis == 'pp'), "Only Pauli basis is allowed for 'chp' evotype"
        #    # For chp (and statevec, etc), want full superoperator basis
        #    basis = _Basis.cast(basis, 2**dim, sparse=False)
        #else:
        state_space = _statespace.StateSpace.cast(state_space)
        self.basis = _Basis.cast(basis, state_space.dim, sparse=False)

        num_rates = self.basis.size - 1
        initial_sto_rates = [initial_rate / num_rates] * num_rates
        _StochasticNoiseOp.__init__(self, state_space, self.basis, evotype, initial_sto_rates, seed_or_state)

        # For DepolarizeOp, set params to only first element
        self.params = _np.array([self.params[0]])
        self._paramlbls = _np.array(["common stochastic error rate for depolarization"], dtype=object)

    def _rates_to_params(self, rates):
        """Note: requires rates to all be the same"""
        assert(all([rates[0] == r for r in rates[1:]]))
        return _np.array([_np.sqrt(rates[0])], 'd')

    def _params_to_rates(self, params):
        return _np.array([params[0]**2] * (self.basis.size - 1), 'd')

    def _get_rate_poly_dicts(self):
        """ Return a list of dicts, one per rate, expressing the
            rate as a polynomial of the local parameters (tuple
            keys of dicts <=> poly terms, e.g. (1,1) <=> x1^2) """
        return [{(0, 0): 1.0} for i in range(self.basis.size - 1)]  # rates are all just param0 squared

    @property
    def total_term_magnitude_deriv(self):
        """
        The derivative of the sum of *all* this operator's terms.

        Computes the derivative of the total (sum) of the magnitudes of all this
        operator's terms with respect to the operators (local) parameters.

        Returns
        -------
        numpy array
            An array of length self.num_params
        """
        # abs(rate) = rate = param_0**2, and there are basis.size-1 of them,
        # so d( sum(abs(rates)) )/dparam_0 = 2*(basis.size-1)*param_0
        return 2 * (self.basis.size - 1) * self.to_vector()

    def copy(self, parent=None, memo=None):
        """
        Copy this object.

        Parameters
        ----------
        parent : Model, optional
            The parent model to set for the copy.

        Returns
        -------
        DepolarizeOp
            A copy of this object.
        """
        if memo is not None and id(self) in memo: return memo[id(self)]
        copyOfMe = DepolarizeOp(self.state_space, self.basis, self._evotype, self._params_to_rates(self.to_vector())[0])
        return self._copy_gpindices(copyOfMe, parent, memo)

    def __str__(self):
        s = "Depolarize noise operation map with dim = %d, num params = %d\n" % \
            (self.dim, self.num_params)
        s += 'Strength: %s\n' % (self.params**2 * (self.basis.size - 1))
        return s
