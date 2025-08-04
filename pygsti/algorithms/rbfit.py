"""
Functions for analyzing RB data
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from scipy.optimize import curve_fit as _curve_fit

from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.tools import rbtools as _rbt


def std_least_squares_fit(lengths, asps, n, seed=None, asymptote=None, ftype='full', rtype='EI'):
    """
    Implements a "standard" least-squares fit of RB data.

    Fits the average success probabilities to the exponential decay A + Bp^m, using least-squares fitting.

    Parameters
    ----------
    lengths : list
        The RB lengths to fit to (the 'm' values in A + Bp^m).

    asps : list
        The average survival probabilities to fit (the observed P_m values to fit
        to P_m = A + Bp^m).

    n : int
        The number of qubits the data was generated from.

    seed : list, optional
        Seeds for the fit of B and p (A, if a variable, is seeded to the asymptote defined by `asympote`).

    asymptote : float, optional
        If not None, the A value for the fitting to A + Bp^m with A fixed. Defaults to 1/2^n.
        Note that this value is used even when fitting A; in that case B and p are estimated
        with A fixed to this value, and then this A and the estimated B and p are seed for the
        full fit.

    ftype : {'full','FA','full+FA'}, optional
        The fit type to implement. 'full' corresponds to fitting all of A, B and p. 'FA' corresponds
        to fixing 'A' to the value specified by `asymptote`. 'full+FA' returns the results of both
        fits.

    rtype : {'EI','AGI'}, optional
        The RB error rate rescaling convention. 'EI' results in RB error rates that are associated
        with the entanglement infidelity, which is the error probability with stochastic errors (and
        is equal to the diamond distance). 'AGI' results in RB error rates that are associated with
        average gate infidelity.

    Returns
    -------
    Dict or Dicts
        If `ftype` = 'full' or `ftype`  = 'FA' then a dict containing the results of the relevant fit.
        If `ftype` = 'full+FA' then two dicts are returned. The first dict corresponds to the full fit
        and the second to the fixed-asymptote fit.
    """
    if asymptote is not None: A = asymptote
    else: A = 1 / 2**n
    # First perform a fit with a fixed asymptotic value
    FAF_results = custom_least_squares_fit(lengths, asps, n, a=A, seed=seed, rtype=rtype)
    if ftype == 'FA':
        return FAF_results

    if not all([x in FAF_results['estimates'] for x in ('a', 'b', 'p')]):
        raise ValueError(("Initial fixed-asymptotic RB fit failed and is needed to seed requested %s fit type."
                          " Please check that the RB data is valid.") % ftype)

    # Full fit is seeded by the fixed asymptote fit.
    seed_full = [FAF_results['estimates']['a'], FAF_results['estimates']['b'], FAF_results['estimates']['p']]
    FF_results = custom_least_squares_fit(lengths, asps, n, seed=seed_full, rtype=rtype)

    # Returns the requested fit type.
    if ftype == 'full': return FF_results
    elif ftype == 'full+FA': return FF_results, FAF_results
    else: raise ValueError("The `ftype` value is invalid!")


def custom_least_squares_fit(lengths, asps, n, a=None, b=None, seed=None, rtype='EI'):
    """
    Fits RB average success probabilities to the exponential decay a + Bp^m using least-squares fitting.

    Parameters
    ----------
    lengths : list
        The RB lengths to fit to (the 'm' values in a + Bp^m).

    asps : list
        The average survival probabilities to fit (the observed P_m values to fit
        to P_m = a + Bp^m).

    n : int
        The number of qubits the data was generated from.

    a : float, optional
        If not None, a value to fix a to.

    b : float, optional
        If not None, a value to fix b to.

    seed : list, optional
        Seeds for variables in the fit, in the order [a,b,p] (with a and/or b dropped if it is set
        to a fixed value).

    rtype : {'EI','AGI'}, optional
        The RB error rate rescaling convention. 'EI' results in RB error rates that are associated
        with the entanglement infidelity, which is the error probability with stochastic errors (and
        is equal to the diamond distance). 'AGI' results in RB error rates that are associated with
        average gate infidelity.

    Returns
    -------
    Dict
        The fit results. If item with the key 'success' is False, the fit has failed.
    """
    seed_dict = {}
    variable = {}
    variable['a'] = True
    variable['b'] = True
    variable['p'] = True
    lengths = _np.array(lengths, _np.int64)
    asps = _np.array(asps, 'd')

    # The fit to do if a fixed value for a is given
    if a is not None:

        variable['a'] = False

        if b is not None:

            variable['b'] = False

            def curve_to_fit(m, p):
                return a + b * p**m

            if seed is None:
                seed = 0.9
                seed_dict['a'] = None
                seed_dict['b'] = None
                seed_dict['p'] = seed

            try:
                fitout, junk = _curve_fit(curve_to_fit, lengths, asps, p0=seed, bounds=([0.], [1.]))
                p = fitout
                success = True
            except:
                success = False

        else:

            def curve_to_fit(m, b, p):
                return a + b * p**m

            if seed is None:
                seed = [1. - a, 0.9]
                seed_dict['a'] = None
                seed_dict['b'] = 1. - a
                seed_dict['p'] = 0.9
            try:
                fitout, junk = _curve_fit(curve_to_fit, lengths, asps, p0=seed, bounds=([-_np.inf, 0.], [+_np.inf, 1.]))
                b = fitout[0]
                p = fitout[1]
                success = True
            except:
                success = False

    # The fit to do if a fixed value for a is not given
    else:

        if b is not None:

            variable['b'] = False

            def curve_to_fit(m, a, p):
                return a + b * p**m

            if seed is None:
                seed = [1 / 2**n, 0.9]
                seed_dict['a'] = 1 / 2**n
                seed_dict['b'] = None
                seed_dict['p'] = 0.9

            try:
                fitout, junk = _curve_fit(curve_to_fit, lengths, asps, p0=seed, bounds=([0., 0.], [1., 1.]))
                a = fitout[0]
                p = fitout[1]
                success = True
            except:
                success = False

        else:

            def curve_to_fit(m, a, b, p):
                return a + b * p**m

            if seed is None:
                seed = [1 / 2**n, 1 - 1 / 2**n, 0.9]
                seed_dict['a'] = 1 / 2**n
                seed_dict['b'] = 1 - 1 / 2**n
                seed_dict['p'] = 0.9

            try:
                fitout, junk = _curve_fit(curve_to_fit, lengths, asps, p0=seed,
                                          bounds=([0., -_np.inf, 0.], [1., +_np.inf, 1.]))
                a = fitout[0]
                b = fitout[1]
                p = fitout[2]
                success = True
            except:
                success = False

    estimates = {}
    if success:
        estimates['a'] = a
        estimates['b'] = b
        estimates['p'] = p
        estimates['r'] = _rbt.p_to_r(p, 2**n, rtype)

    results = {}
    results['estimates'] = estimates
    results['variable'] = variable
    results['seed'] = seed_dict
    # Todo : fix this.
    results['success'] = success

    return results


class FitResults(_NicelySerializable):
    """
    An object to contain the results from fitting RB data.

    Currently just a container for the results, and does not include any methods.

    Parameters
    ----------
    fittype : str
        A string to identity the type of fit.

    seed : list
        The seed used in the fitting.

    rtype : {'IE','AGI'}
        The type of RB error rate that the 'r' in these fit results corresponds to.

    success : bool
        Whether the fit was successful.

    estimates : dict
        A dictionary containing the estimates of all parameters

    variable : dict
        A dictionary that specifies which of the parameters in "estimates" where variables
        to estimate (set to True for estimated parameters, False for fixed constants). This
        is useful when fitting to A + B*p^m and fixing one or more of these parameters: because
        then the "estimates" dict can still be queried for all three parameters.

    stds : dict, optional
        Estimated standard deviations for the parameters.

    bootstraps : dict, optional
        Bootstrapped values for the estimated parameters, from which the standard deviations
        were calculated.

    bootstraps_failrate : float, optional
        The proporition of the estimates of the parameters from bootstrapped dataset failed.
    """

    def __init__(self, fittype, seed, rtype, success, estimates, variable, stds=None,
                 bootstraps=None, bootstraps_failrate=None):
        """
        Initialize a FitResults object.

        Parameters
        ----------
        fittype : str
            A string to identity the type of fit.

        seed : list
            The seed used in the fitting.

        rtype : {'IE','AGI'}
            The type of RB error rate that the 'r' in these fit results corresponds to.

        success : bool
            Whether the fit was successful.

        estimates : dict
            A dictionary containing the estimates of all parameters

        variable : dict
            A dictionary that specifies which of the parameters in "estimates" where variables
            to estimate (set to True for estimated parameters, False for fixed constants). This
            is useful when fitting to A + B*p^m and fixing one or more of these parameters: because
            then the "estimates" dict can still be queried for all three parameters.

        stds : dict, optional
            Estimated standard deviations for the parameters.

        bootstraps : dict, optional
            Bootstrapped values for the estimated parameters, from which the standard deviations
            were calculated.

        bootstraps_failrate : float, optional
            The proporition of the estimates of the parameters from bootstrapped dataset failed.
        """
        super().__init__()
        self.fittype = fittype
        self.seed = seed
        self.rtype = rtype
        self.success = success

        self.estimates = estimates
        self.variable = variable
        self.stds = stds

        self.bootstraps = bootstraps
        self.bootstraps_failrate = bootstraps_failrate

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'fit_type': self.fittype,
                      'seed': self.seed,
                      'r_type': self.rtype,
                      'success': self.success,
                      'estimates': self.estimates,
                      'variable': self.variable,
                      'stds': self.stds,
                      'bootstraps': self.bootstraps,
                      'bootstraps_failrate': self.bootstraps_failrate
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        return cls(state['fit_type'], state['seed'], state['r_type'], state['success'],
                   state['estimates'], state['variable'], state['stds'],
                   state['bootstraps'], state['bootstraps_failrate'])
