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


# Obsolute function to be deleted.
# def std_practice_analysis(RBSdataset, seed=[0.8, 0.95], bootstrap_samples=200, asymptote='std', rtype='EI',
#                           datatype='auto'):
#     """
#     Implements a "standard practice" analysis of RB data. Fits the average success probabilities to the exponential
#     decay A + Bp^m, using least-squares fitting, with (1) A fixed (as standard, to 1/2^n where n is the number of
#     qubits the data is for), and (2) A, B and p all allowed to varying. Confidence intervals are also estimated using
#     a standard non-parameteric boostrap.

#     Parameters
#     ----------
#     RBSdataset : RBSummaryDataset
#         An RBSUmmaryDataset containing the data to analyze

#     seed : list, optional
#         Seeds for the fit of B and p (A is seeded to the asymptote defined by `asympote`).

#     bootstrap_samples : int, optional
#         The number of samples in the bootstrap.

#     asymptote : str or float, optional
#         The A value for the fitting to A + Bp^m with A fixed. If a string must be 'std', in
#         in which case A is fixed to 1/2^n.

#     rtype : {'EI','AGI'}, optional
#         The RB error rate rescaling convention. 'EI' results in RB error rates that are associated
#         with the entanglement infidelity, which is the error probability with stochastic errors (and
#         is equal to the diamond distance). 'AGI' results in RB error rates that are associated with
#         average gate infidelity.

#     Returns
#     -------
#     RBResults
#         An object encapsulating the RB results (and data).

#     """
#     assert(datatype == 'raw' or datatype == 'adjusted' or datatype == 'auto'), "Unknown data type!"

#     if datatype == 'auto':
#         if RBSdataset.datatype == 'hamming_distance_counts':
#             datatype = 'adjusted'
#         else:
#             datatype = 'raw'

#     lengths = RBSdataset.lengths
#     n = RBSdataset.num_qubits

#     if isinstance(asymptote, str):
#         assert(asymptote == 'std'), "If `asympotote` is a string it must be 'std'!"
#         if datatype == 'raw':
#             asymptote = 1 / 2**n
#         elif datatype == 'adjusted':
#             asymptote = 1 / 4**n

#     if datatype == 'adjusted':
#         ASPs = RBSdataset.adjusted_ASPs
#     if datatype == 'raw':
#         ASPs = RBSdataset.ASPs

#     FF_results, FAF_results = std_least_squares_fit(lengths, ASPs, n, seed=seed, asymptote=asymptote,
#                                                              ftype='full+FA', rtype=rtype)

#     parameters = ['A', 'B', 'p', 'r']
#     bootstraps_FF = {}
#     bootstraps_FAF = {}

#     if bootstrap_samples > 0:

#         bootstraps_FF = {p: [] for p in parameters}
#         bootstraps_FAF = {p: [] for p in parameters}
#         failcount_FF = 0
#         failcount_FAF = 0

#         # Add bootstrapped data, if neccessary.
#         RBSdataset.add_bootstrapped_datasets(samples=bootstrap_samples)

#         for i in range(bootstrap_samples):

#             if datatype == 'adjusted':
#                 BS_ASPs = RBSdataset.bootstraps[i].adjusted_ASPs
#             if datatype == 'raw':
#                 BS_ASPs = RBSdataset.bootstraps[i].ASPs

#             BS_FF_results, BS_FAF_results = std_least_squares_fit(lengths, BS_ASPs, n, seed=seed,
#                                                                            asymptote=asymptote, ftype='full+FA',
#                                                                            rtype=rtype)

#             if BS_FF_results['success']:
#                 for p in parameters:
#                     bootstraps_FF[p].append(BS_FF_results['estimates'][p])
#             else:
#                 failcount_FF += 1
#             if BS_FAF_results['success']:
#                 for p in parameters:
#                     bootstraps_FAF[p].append(BS_FAF_results['estimates'][p])
#             else:
#                 failcount_FAF += 1

#         failrate_FF = failcount_FF / bootstrap_samples
#         failrate_FAF = failcount_FAF / bootstrap_samples

#         std_FF = {p: _np.std(_np.array(bootstraps_FF[p])) for p in parameters}
#         std_FAF = {p: _np.std(_np.array(bootstraps_FAF[p])) for p in parameters}

#     else:
#         bootstraps_FF = None
#         std_FF = None
#         failrate_FF = None
#         bootstraps_FAF = None
#         std_FAF = None
#         failrate_FAF = None

#     fits = {}
#     fits['full'] = FitResults('LS', FF_results['seed'], rtype, FF_results['success'], FF_results['estimates'],
#                               FF_results['variable'], stds=std_FF, bootstraps=bootstraps_FF,
#                               bootstraps_failrate=failrate_FF)

#     fits['A-fixed'] = FitResults('LS', FAF_results['seed'], rtype, FAF_results['success'],
#                                  FAF_results['estimates'], FAF_results['variable'], stds=std_FAF,
#                                  bootstraps=bootstraps_FAF, bootstraps_failrate=failrate_FAF)

#     results = SimpleRBResults(RBSdataset, rtype, fits)

#     return results


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

# Obsolute RB results class
# class SimpleRBResults(object):
#     """
#     An object to contain the results of an RB analysis.

#     """

#     def __init__(self, data, rtype, fits):
#         """
#         Initialize an RBResults object.

#         Parameters
#         ----------
#         data : RBSummaryDataset
#             The RB summary data that the analysis was performed for.

#         rtype : {'IE','AGI'}
#             The type of RB error rate, corresponding to different dimension-dependent
#             re-scalings of (1-p), where p is the RB decay constant in A + B*p^m.

#         fits : dict
#             A dictionary containing FitResults objects, obtained from one or more
#             fits of the data (e.g., a fit with all A, B and p as free parameters and
#             a fit with A fixed to 1/2^n).
#         """
#         self.data = data
#         self.rtype = rtype
#         self.fits = fits

#     def plot(self, fitkey=None, decay=True, success_probabilities=True, size=(8, 5), ylim=None, xlim=None,
#              legend=True, title=None, figpath=None):
#         """
#         Plots RB data and, optionally, a fitted exponential decay.

#         Parameters
#         ----------
#         fitkey : dict key, optional
#             The key of the self.fits dictionary to plot the fit for. If None, will
#             look for a 'full' key (the key for a full fit to A + Bp^m if the standard
#             analysis functions are used) and plot this if possible. It otherwise checks
#             that there is only one key in the dict and defaults to this. If there are
#             multiple keys and none of them are 'full', `fitkey` must be specified when
#             `decay` is True.

#         decay : bool, optional
#             Whether to plot a fit, or just the data.

#         success_probabilities : bool, optional
#             Whether to plot the success probabilities distribution, as a violin plot. (as well
#             as the *average* success probabilities at each length).

#         size : tuple, optional
#             The figure size

#         ylim, xlim : tuple, optional
#             The x and y limits for the figure.

#         legend : bool, optional
#             Whether to show a legend.

#         title : str, optional
#             A title to put on the figure.

#         figpath : str, optional
#             If specified, the figure is saved with this filename.
#         """

#         # Future : change to a plotly plot.
#         try: import matplotlib.pyplot as _plt
#         except ImportError: raise ValueError("This function requires you to install matplotlib!")

#         if decay and fitkey is None:
#             allfitkeys = list(self.fits.keys())
#             if 'full' in allfitkeys: fitkey = 'full'
#             else:
#                 assert(len(allfitkeys) == 1), \
#                     "There are multiple fits and none have the key 'full'. Please specify the fit to plot!"
#                 fitkey = allfitkeys[0]

#         _plt.figure(figsize=size)
#         _plt.plot(self.data.lengths, self.data.ASPs, 'o', label='Average success probabilities')

#         if decay:
#             lengths = _np.linspace(0, max(self.data.lengths), 200)
#             A = self.fits[fitkey].estimates['A']
#             B = self.fits[fitkey].estimates['B']
#             p = self.fits[fitkey].estimates['p']
#             _plt.plot(lengths, A + B * p**lengths,
#                       label='Fit, r = {:.2} +/- {:.1}'.format(self.fits[fitkey].estimates['r'],
#                                                               self.fits[fitkey].stds['r']))

#         if success_probabilities:
#             _plt.violinplot(list(self.data.success_probabilities), self.data.lengths, points=10, widths=1.,
#                             showmeans=False, showextrema=False, showmedians=False)  # , label='Success probabilities')

#         if title is not None: _plt.title(title)
#         _plt.ylabel("Success probability")
#         _plt.xlabel("RB sequence length $(m)$")
#         _plt.ylim(ylim)
#         _plt.xlim(xlim)

#         if legend: _plt.legend()

#         if figpath is not None: _plt.savefig(figpath, dpi=1000)
#         else: _plt.show()

#         return
