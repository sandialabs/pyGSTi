"""Functions related to computation of the log-likelihood."""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import scipy.stats as _stats
import warnings as _warnings
import itertools as _itertools
import time as _time
import sys as _sys
from collections import OrderedDict as _OrderedDict
from . import basistools as _bt
from . import listtools as _lt
from . import jamiolkowski as _jam
from . import mpitools as _mpit
from . import slicetools as _slct
from ..objects import objectivefns as _objfns
from ..objects.smartcache import smart_cached

TOL = 1e-20

# The log(Likelihood) within the standard (non-Poisson) picture is:
#
# L = prod_{i,sl} p_{i,sl}^N_{i,sl}
#
# Where i indexes the operation sequence, and sl indexes the spam label.  N[i] is the total counts
#  for the i-th circuit, and so sum_{sl} N_{i,sl} == N[i]. We can take the log:
#
# log L = sum_{i,sl} N_{i,sl} log(p_{i,sl})
#
#   after patching (linear extrapolation below min_p and ignore f == 0 terms ( 0*log(0) == 0 ) ):
#
# logl = sum_{i,sl} N_{i,sl} log(p_{i,sl})                                                        if p_{i,sl} >= min_p and N_{i,sl} > 0                         # noqa
#                   N_{i,sl} log(min_p)     + S * (p_{i,sl} - min_p) + S2 * (p_{i,sl} - min_p)**2 if p_{i,sl} < p_min and N_{i,sl} > 0                          # noqa
#                   0                                                                             if N_{i,sl} == 0                                              # noqa
#
# dlogL = sum_{i,sl} N_{i,sl} / p_{i,sl} * dp                    if p_{i,sl} >= min_p and N_{i,sl} > 0                                                          # noqa
#                    (S + 2*S2*(p_{i,sl} - min_p)) * dp          if p_{i,sl} < p_min and N_{i,sl} > 0                                                           # noqa
#                    0                                           if N_{i,sl} == 0                                                                               # noqa
#
# hlogL = sum_{i,sl} -N_{i,sl} / p_{i,sl}**2 * dp1 * dp2 +  N_{i,sl} / p_{i,sl} *hp        if p_{i,sl} >= min_p and N_{i,sl} > 0                                # noqa
#                    2*S2* dp1 * dp2 + (S + 2*S2*(p_{i,sl} - min_p)) * hp                  if p_{i,sl} < p_min and N_{i,sl} > 0                                 # noqa
#                    0                                                                     if N_{i,sl} == 0                                                     # noqa
#
#  where S = N_{i,sl} / min_p is the slope of the line tangent to logl at min_p
#    and S2 = 0.5*( -N_{i,sl} / min_p**2 ) is 1/2 the 2nd derivative of the logl term at min_p
#   and hlogL == d/d1 ( d/d2 ( logl ) )  -- i.e. dp2 is the *first* derivative performed...

#Note: Poisson picture entered use when we allowed an EVec which was 1-{other EVecs} -- a
# (0,-1) spam index -- instead of assuming all probabilities of a given gat string summed
# to one -- a (-1,-1) spam index.  The poisson picture gives a correct log-likelihood
# description when the probabilities (for a given operation sequence) may not sum to one, by
# interpreting them each as rates.  In the standard picture, large circuit probabilities
# are not penalized (each standard logL term increases monotonically with each probability,
# and the reason this is ok when the probabilities sum to one is that for a probabilility
# that gets close to 1, there's another that is close to zero, and logL is very negative
# near zero.

# The log(Likelihood) within the Poisson picture is:
#
# L = prod_{i,sl} lambda_{i,sl}^N_{i,sl} e^{-lambda_{i,sl}} / N_{i,sl}!
#
# Where lamba_{i,sl} := p_{i,sl}*N[i] is a rate, i indexes the operation sequence,
#  and sl indexes the spam label.  N[i] is the total counts for the i-th circuit, and
#  so sum_{sl} N_{i,sl} == N[i]. We can ignore the p-independent N_j! and take the log:
#
# log L = sum_{i,sl} N_{i,sl} log(N[i]*p_{i,sl}) - N[i]*p_{i,sl}
#       = sum_{i,sl} N_{i,sl} log(p_{i,sl}) - N[i]*p_{i,sl}   (where we ignore the p-independent log(N[i]) terms)
#
#   after patching (linear extrapolation below min_p and "softening" f == 0 terms w/cubic below radius "a"):
#
# logl = sum_{i,sl} N_{i,sl} log(p_{i,sl}) - N[i]*p_{i,sl}                                                        if p_{i,sl} >= min_p and N_{i,sl} > 0         # noqa
#                   N_{i,sl} log(min_p)    - N[i]*min_p    + S * (p_{i,sl} - min_p) + S2 * (p_{i,sl} - min_p)**2  if p_{i,sl} < p_min and N_{i,sl} > 0          # noqa
#                   0                      - N[i]*p_{i,sl}                                                        if N_{i,sl} == 0 and p_{i,sl} >= a            # noqa
#                   0                      - N[i]*( -(1/(3a**2))p_{i,sl}**3 + p_{i,sl}**2/a + (1/3)*a )           if N_{i,sl} == 0 and p_{i,sl} < a             # noqa
#                   - N[i]*Y(1-sum(p_omitted)) added to "first" N_{i,sl} > 0 entry for omitted probabilities, where
#                                               Y(p) = p if p >= a else ( -(1/(3a**2))p**3 + p**2/a + (1/3)*a )
#
# dlogL = sum_{i,sl} [ N_{i,sl} / p_{i,sl} - N[i] ] * dp                   if p_{i,sl} >= min_p and N_{i,sl} > 0                                                # noqa
#                    (S + 2*S2*(p_{i,sl} - min_p)) * dp                    if p_{i,sl} < p_min and N_{i,sl} > 0                                                 # noqa
#                    -N[i] * dp                                            if N_{i,sl} == 0 and p_{i,sl} >= a                                                   # noqa
#                    -N[i] * ( (-1/a**2)p_{i,sl}**2 + 2*p_{i,sl}/a ) * dp  if N_{i,sl} == 0 and p_{i,sl} < a
#                    +N[i]*sum(dY/dp_omitted * dp_omitted) added to "first" N_{i,sl} > 0 entry for omitted probabilities
#
# hlogL = sum_{i,sl} -N_{i,sl} / p_{i,sl}**2 * dp1 * dp2 + [ N_{i,sl} / p_{i,sl} - N[i] ]*hp      if p_{i,sl} >= min_p and N_{i,sl} > 0                         # noqa
#                    2*S2* dp1 * dp2 + (S + 2*S2*(p_{i,sl} - min_p)) * hp                         if p_{i,sl} < p_min and N_{i,sl} > 0                          # noqa
#                    -N[i] * hp                                                                   if N_{i,sl} == 0 and p_{i,sl} >= a                            # noqa
#                    -N[i]*( (-2/a**2)p_{i,sl} + 2/a ) * dp1 * dp2                                                                                              # noqa
#                        - N[i]*( (-1/a**2)p_{i,sl}**2 + 2*p_{i,sl}/a ) * hp                      if N_{i,sl} == 0 and p_{i,sl} < a                             # noqa
#                    +N[i]*sum(d2Y/dp_omitted2 * dp_omitted1 * dp_omitted2 +
#                              dY/dp_omitted * hp_omitted)                                 added to "first" N_{i,sl} > 0 entry for omitted probabilities        # noqa
#
#  where S = N_{i,sl} / min_p - N[i] is the slope of the line tangent to logl at min_p
#    and S2 = 0.5*( -N_{i,sl} / min_p**2 ) is 1/2 the 2nd derivative of the logl term at min_p so
#    logL_term = logL_term(min_p) + S * (p-min_p) + S2 * (p-min_p)**2
#   and hlogL == d/d1 ( d/d2 ( logl ) )  -- i.e. dp2 is the *first* derivative performed...
#
# For cubic interpolation, use function F(p) (derived by Robin: match value, 1st-deriv, 2nd-deriv at p == r, and require
# min at p == 0):
#  Given a radius r << 1 (but r>0):
#   F(p) = piecewise{ if( p>r ) then p; else -(1/3)*p^3/r^2 + p^2/r + (1/3)*r }
#  OLD: quadratic that doesn't match 2nd-deriv:
#   F(p) = piecewise{ if( p>r ) then p; else (r-p)^2/(2*r) + p }


def logl(model, dataset, circuit_list=None,
         minProbClip=1e-6, probClipInterval=(-1e6, 1e6), radius=1e-4,
         poissonPicture=True, opLabelAliases=None, wildcard=None,
         cache=None, comm=None, memLimit=None):
    """
    The log-likelihood function.

    Parameters
    ----------
    model : Model
        Model of parameterized gates

    dataset : DataSet
        Probability data

    circuit_list : list of (tuples or Circuits), optional
        Each element specifies a operation sequence to include in the log-likelihood
        sum.  Default value of None implies all the operation sequences in dataset
        should be used.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poissonPicture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logl value.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    wildcard : WildcardBudget
        A wildcard budget to apply to this log-likelihood computation.
        This increases the returned log-likelihood value by adjusting
        (by a maximal amount measured in TVD, given by the budget) the
        probabilities produced by `model` to optimially match the data
        (within the bugetary constraints) evaluating the log-likelihood.

    cache : ComputationCache, optional
        A cache object used to hold results for the same `model` and `dataset` and `circuit_list`.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    Returns
    -------
    float
        The log likelihood
    """
    v = logl_per_circuit(model, dataset, circuit_list,
                         minProbClip, probClipInterval, radius,
                         poissonPicture, opLabelAliases, wildcard,
                         cache, comm, memLimit)
    return _np.sum(v)  # sum over *all* dimensions


def logl_per_circuit(model, dataset, circuit_list=None,
                     minProbClip=1e-6, probClipInterval=(-1e6, 1e6), radius=1e-4,
                     poissonPicture=True, opLabelAliases=None, wildcard=None,
                     cache=None, comm=None, memLimit=None):
    """
    Computes the per-circuit log-likelihood contribution for a set of circuits.

    Parameters
    ----------
    This function takes the same arguments as :func:`logl` except it
    doesn't perform the final sum over circuits.

    Returns
    -------
    numpy.ndarray
        Array of length either `len(circuit_list)` or `len(dataset.keys())`.
        Values are the log-likelihood contributions of the corresponding gate
        string aggregated over outcomes.
    """
    obj_max = _objfns.objfn(_objfns.MaxLogLFunction, model, dataset, circuit_list, cache=cache,
                            opLabelAliases=opLabelAliases, poisson_picture=poissonPicture)
    obj_cls = _objfns.DeltaLogLFunctionPoissonPic if poissonPicture else _objfns.DeltaLogLFunction
    obj = _objfns.objfn(obj_cls, model, dataset, circuit_list,
                        {'minProbClip': minProbClip,
                         'probClipInterval': probClipInterval,
                         'radius': radius}, None,
                        opLabelAliases, cache, comm, memLimit)

    if wildcard:
        assert(poissonPicture), "Wildcard budgets can only be used with `poissonPicture=True`"
        obj = _objfns.LogLWildcardFunction(obj, model.to_vector(), wildcard)

    return obj_max.percircuit_fn() - obj.percircuit_fn()


def logl_jacobian(model, dataset, circuit_list=None,
                  minProbClip=1e-6, probClipInterval=(-1e6, 1e6), radius=1e-4,
                  poissonPicture=True, opLabelAliases=None, cache=None,
                  comm=None, memLimit=None, verbosity=0):
    """
    The jacobian of the log-likelihood function.

    Parameters
    ----------
    model : Model
        Model of parameterized gates (including SPAM)

    dataset : DataSet
        Probability data

    circuit_list : list of (tuples or Circuits), optional
        Each element specifies a operation sequence to include in the log-likelihood
        sum.  Default value of None implies all the operation sequences in dataset
        should be used.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poissonPicture : boolean, optional
        Whether the Poisson-picutre log-likelihood should be differentiated.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    cache : ComputationCache, optional
        A cache object used to hold results for the same `model` and `dataset` and `circuit_list`.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    verbosity : int, optional
        How much detail to print to stdout.

    Returns
    -------
    numpy array
      array of shape (M,), where M is the length of the vectorized model.
    """
    obj_cls = _objfns.DeltaLogLFunctionPoissonPic if poissonPicture else _objfns.DeltaLogLFunction
    obj = _objfns.objfn(obj_cls, model, dataset, circuit_list,
                        {'minProbClip': minProbClip,
                         'probClipInterval': probClipInterval,
                         'radius': radius}, None,
                        opLabelAliases, cache, comm, memLimit)
    return -obj.jfn()  # negative b/c objective is deltaLogL = max_logl - logL


def logl_hessian(model, dataset, circuit_list=None,
                 minProbClip=1e-6, probClipInterval=(-1e6, 1e6), radius=1e-4,
                 poissonPicture=True, opLabelAliases=None, cache=None,
                 comm=None, memLimit=None, verbosity=0):
    """
    The hessian of the log-likelihood function.

    Parameters
    ----------
    model : Model
        Model of parameterized gates (including SPAM)

    dataset : DataSet
        Probability data

    circuit_list : list of (tuples or Circuits), optional
        Each element specifies a operation sequence to include in the log-likelihood
        sum.  Default value of None implies all the operation sequences in dataset
        should be used.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by
        models during MLEGST's search for an optimal model (if not None).
        if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poissonPicture : boolean, optional
        Whether the Poisson-picutre log-likelihood should be differentiated.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    cache : ComputationCache, optional
        A cache object used to hold results for the same `model` and `dataset` and `circuit_list`.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    verbosity : int, optional
        How much detail to print to stdout.

    Returns
    -------
    numpy array
      array of shape (M,M), where M is the length of the vectorized model.
    """
    obj_cls = _objfns.DeltaLogLFunctionPoissonPic if poissonPicture else _objfns.DeltaLogLFunction
    obj = _objfns.objfn(obj_cls, model, dataset, circuit_list,
                        {'minProbClip': minProbClip,
                         'probClipInterval': probClipInterval,
                         'radius': radius}, None,
                        opLabelAliases, cache, comm, memLimit)
    return -obj.hfn()  # negative b/c objective is deltaLogL = max_logl - logL


def logl_approximate_hessian(model, dataset, circuit_list=None,
                             minProbClip=1e-6, probClipInterval=(-1e6, 1e6), radius=1e-4,
                             poissonPicture=True, opLabelAliases=None, cache=None,
                             comm=None, memLimit=None, verbosity=0):
    """
    An approximate Hessian of the log-likelihood function.

    An approximation to the true Hessian is computed using just the Jacobian
    (and *not* the Hessian) of the probabilities w.r.t. the model
    parameters.  Let `J = d(probs)/d(params)` and denote the Hessian of the
    log-likelihood w.r.t. the probabilities as `d2(logl)/dprobs2` (a *diagonal*
    matrix indexed by the term, i.e. probability, of the log-likelihood). Then
    this function computes:

    `H = J * d2(logl)/dprobs2 * J.T`

    Which simply neglects the `d2(probs)/d(params)2` terms of the true Hessian.
    Since this curvature is expected to be small at the MLE point, this
    approximation can be useful for computing approximate error bars.

    Parameters
    ----------
    model : Model
        Model of parameterized gates (including SPAM)

    dataset : DataSet
        Probability data

    circuit_list : list of (tuples or Circuits), optional
        Each element specifies a operation sequence to include in the log-likelihood
        sum.  Default value of None implies all the operation sequences in dataset
        should be used.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poissonPicture : boolean, optional
        Whether the Poisson-picutre log-likelihood should be differentiated.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    cache : ComputationCache, optional
        A cache object used to hold results for the same `model` and `dataset` and `circuit_list`.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    verbosity : int, optional
        How much detail to print to stdout.

    Returns
    -------
    numpy array
      array of shape (M,M), where M is the length of the vectorized model.
    """
    obj_cls = _objfns.DeltaLogLFunctionPoissonPic if poissonPicture else _objfns.DeltaLogLFunction
    obj = _objfns.objfn(obj_cls, model, dataset, circuit_list,
                        {'minProbClip': minProbClip,
                         'probClipInterval': probClipInterval,
                         'radius': radius}, None,
                        opLabelAliases, cache, comm, memLimit)
    return -obj.approx_hfn()  # negative b/c objective is deltaLogL = max_logl - logL


def logl_max(model, dataset, circuit_list=None, poissonPicture=True,
             opLabelAliases=None, cache=None):
    """
    The maximum log-likelihood possible for a DataSet.  That is, the
    log-likelihood obtained by a maximal model that can fit perfectly
    the probability of each operation sequence.

    Parameters
    ----------
    model : Model
        the model, used only for operation sequence compilation

    dataset : DataSet
        the data set to use.

    circuit_list : list of (tuples or Circuits), optional
        Each element specifies a operation sequence to include in the max-log-likelihood
        sum.  Default value of None implies all the operation sequences in dataset should
        be used.

    poissonPicture : boolean, optional
        Whether the Poisson-picture maximum log-likelihood should be returned.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    cache : ComputationCache, optional
        A cache object used to hold results for the same `model` and `dataset` and `circuit_list`.

    Returns
    -------
    float
    """
    obj_max = _objfns.objfn(_objfns.MaxLogLFunction, model, dataset, circuit_list, cache=cache,
                            opLabelAliases=opLabelAliases, poisson_picture=poissonPicture)
    return obj_max.fn()


def logl_max_per_circuit(model, dataset, circuit_list=None,
                         poissonPicture=True, opLabelAliases=None, cache=None):
    """
    The vector of maximum log-likelihood contributions for each circuit,
    aggregated over outcomes.

    Parameters
    ----------
    This function takes the same arguments as :func:`logl_max` except it
    doesn't perform the final sum over operation sequences and SPAM labels.

    Returns
    -------
    numpy.ndarray
        Array of length either `len(circuit_list)` or `len(dataset.keys())`.
        Values are the maximum log-likelihood contributions of the corresponding
        operation sequence aggregated over outcomes.
    """
    obj_max = _objfns.objfn(_objfns.MaxLogLFunction, model, dataset, circuit_list, cache=cache,
                            opLabelAliases=opLabelAliases, poisson_picture=poissonPicture)
    return obj_max.percircuit_fn()


def two_delta_logl_nsigma(model, dataset, circuit_list=None,
                          minProbClip=1e-6, probClipInterval=(-1e6, 1e6), radius=1e-4,
                          poissonPicture=True, opLabelAliases=None,
                          dof_calc_method='nongauge', wildcard=None):
    """See docstring for :function:`pygsti.tools.two_delta_logl` """
    assert(dof_calc_method is not None)
    return two_delta_logl(model, dataset, circuit_list,
                          minProbClip, probClipInterval, radius,
                          poissonPicture, opLabelAliases,
                          None, None, dof_calc_method, wildcard)[1]


def two_delta_logl(model, dataset, circuit_list=None,
                   minProbClip=1e-6, probClipInterval=(-1e6, 1e6), radius=1e-4,
                   poissonPicture=True, opLabelAliases=None,
                   dof_calc_method=None, wildcard=None,
                   cache=None, comm=None):
    """
    Twice the difference between the maximum and actual log-likelihood,
    optionally along with Nsigma (# std deviations from mean) and p-value
    relative to expected chi^2 distribution (when `dof_calc_method` is
    not None).

    This function's arguments are supersets of :function:`logl`, and
    :function:`logl_max`. This is a convenience function, equivalent to
    `2*(logl_max(...) - logl(...))`, whose value is what is often called
    the *log-likelihood-ratio* between the "maximal model" (that which trivially
    fits the data exactly) and the model given by `model`.

    Parameters
    ----------
    model : Model
        Model of parameterized gates

    dataset : DataSet
        Probability data

    circuit_list : list of (tuples or Circuits), optional
        Each element specifies a operation sequence to include in the log-likelihood
        sum.  Default value of None implies all the operation sequences in dataset
        should be used.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poissonPicture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the computed log-likelihood values.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    dof_calc_method : {None, "all", "nongauge"}
        How `model`'s number of degrees of freedom (parameters) are obtained
        when computing the number of standard deviations and p-value relative to
        a chi2_k distribution, where `k` is additional degrees of freedom
        possessed by the maximal model. If None, then `Nsigma` and `pvalue` are
        not returned (see below).

    wildcard : WildcardBudget
        A wildcard budget to apply to this log-likelihood computation.
        This increases the returned log-likelihood value by adjusting
        (by a maximal amount measured in TVD, given by the budget) the
        probabilities produced by `model` to optimially match the data
        (within the bugetary constraints) evaluating the log-likelihood.

    cache : ComputationCache, optional
        A cache object used to hold results for the same `model` and `dataset` and `circuit_list`.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    twoDeltaLogL : float
        2*(loglikelihood(maximal_model,data) - loglikelihood(model,data))

    Nsigma, pvalue : float
        Only returned when `dof_calc_method` is not None.
    """
    obj_cls = _objfns.DeltaLogLFunctionPoissonPic if poissonPicture else _objfns.DeltaLogLFunction
    obj = _objfns.objfn(obj_cls, model, dataset, circuit_list,
                        {'minProbClip': minProbClip,
                         'probClipInterval': probClipInterval,
                         'radius': radius}, None,
                        opLabelAliases, cache, comm)

    if wildcard:
        assert(poissonPicture), "Wildcard budgets can only be used with `poissonPicture=True`"
        obj = _objfns.LogLWildcardFunction(obj, model.to_vector(), wildcard)

    twoDeltaLogL = 2 * obj.fn()

    if dof_calc_method is None:
        return twoDeltaLogL
    elif dof_calc_method == "nongauge":
        if hasattr(model, 'num_nongauge_params'):
            mdl_dof = model.num_nongauge_params()
        else:
            mdl_dof = model.num_params()
    elif dof_calc_method == "all":
        mdl_dof = model.num_params()
    else: raise ValueError("Invalid `dof_calc_method` arg: %s" % dof_calc_method)

    if circuit_list is not None:
        ds_strs = _lt.apply_aliases_to_circuit_list(circuit_list, opLabelAliases)
    else: ds_strs = None

    Ns = dataset.get_degrees_of_freedom(ds_strs)
    k = max(Ns - mdl_dof, 1)
    if Ns <= mdl_dof: _warnings.warn("Max-model params (%d) <= model params (%d)!  Using k == 1." % (Ns, mdl_dof))

    Nsigma = (twoDeltaLogL - k) / _np.sqrt(2 * k)
    pvalue = 1.0 - _stats.chi2.cdf(twoDeltaLogL, k)
    return twoDeltaLogL, Nsigma, pvalue


def two_delta_logl_per_circuit(model, dataset, circuit_list=None,
                               minProbClip=1e-6, probClipInterval=(-1e6, 1e6), radius=1e-4,
                               poissonPicture=True, opLabelAliases=None,
                               dof_calc_method=None, wildcard=None,
                               cache=None, comm=None):
    """
    The vector of twice the difference between the maximum and actual
    log-likelihood for each operation sequence, aggregated over outcomes.

    Optionally (when `dof_calc_method` is not None) returns parallel vectors
    containing the Nsigma (# std deviations from mean) and the p-value relative
    to expected chi^2 distribution for each sequence.

    Parameters
    ----------
    This function takes the same arguments as :func:`two_delta_logl` except it
    doesn't perform the final sum over operation sequences and SPAM labels.

    Returns
    -------
    twoDeltaLogL_terms : numpy.ndarray
    Nsigma, pvalue : numpy.ndarray
        Only returned when `dof_calc_method` is not None.
    """
    obj_cls = _objfns.DeltaLogLFunctionPoissonPic if poissonPicture else _objfns.DeltaLogLFunction
    obj = _objfns.objfn(obj_cls, model, dataset, circuit_list,
                        {'minProbClip': minProbClip,
                         'probClipInterval': probClipInterval,
                         'radius': radius}, None,
                        opLabelAliases, cache, comm)

    if wildcard:
        assert(poissonPicture), "Wildcard budgets can only be used with `poissonPicture=True`"
        obj = _objfns.LogLWildcardFunction(obj, model.to_vector(), wildcard)

    twoDeltaLogL_percircuit = 2 * obj.percircuit_fn()

    if dof_calc_method is None: return twoDeltaLogL_percircuit
    elif dof_calc_method == "all": mdl_dof = model.num_params()
    elif dof_calc_method == "nongauge": mdl_dof = model.num_nongauge_params()
    else: raise ValueError("Invalid `dof_calc_method` arg: %s" % dof_calc_method)

    if circuit_list is not None:
        ds_strs = _lt.apply_aliases_to_circuit_list(circuit_list, opLabelAliases)
    else: ds_strs = None

    Ns = dataset.get_degrees_of_freedom(ds_strs)
    k = max(Ns - mdl_dof, 1)
    # HACK - just take a single average #dof per circuit to use as chi_k distribution!
    k = int(_np.ceil(k / (1.0 * len(circuit_list))))

    Nsigma = (twoDeltaLogL_percircuit - k) / _np.sqrt(2 * k)
    pvalue = _np.array([1.0 - _stats.chi2.cdf(x, k) for x in twoDeltaLogL_percircuit], 'd')
    return twoDeltaLogL_percircuit, Nsigma, pvalue


#UNUSED (REMOVE?)
def forbidden_prob(model, dataset):
    """
    Compute the sum of the out-of-range probabilities
    generated by model, using only those operation sequences
    contained in dataset.  Non-zero value indicates
    that model is not in XP for the supplied dataset.

    Parameters
    ----------
    model : Model
        model to generate probabilities.

    dataset : DataSet
        data set to obtain operation sequences.  Dataset counts are
        used to check for zero or all counts being under a
        single spam label, in which case out-of-bounds probabilities
        are ignored because they contribute zero to the logl sum.

    Returns
    -------
    float
        sum of the out-of-range probabilities.
    """
    forbidden_prob = 0

    for mdl, dsRow in dataset.items():
        probs = model.probs(mdl)
        for (spamLabel, p) in probs.items():
            if p < TOL:
                if round(dsRow[spamLabel]) == 0: continue  # contributes zero to the sum
                else: forbidden_prob += abs(TOL - p) + TOL
            elif p > 1 - TOL:
                if round(dsRow[spamLabel]) == dsRow.total: continue  # contributes zero to the sum
                else: forbidden_prob += abs(p - (1 - TOL)) + TOL

    return forbidden_prob


#UNUSED (REMOVE?)
def prep_penalty(rhoVec, basis):
    """
    Penalty assigned to a state preparation (rho) vector rhoVec.  State
      preparation density matrices must be positive semidefinite
      and trace == 1.  A positive return value indicates an
      these criteria are not met and the rho-vector is invalid.

    Parameters
    ----------
    rhoVec : numpy array
        rho vector array of shape (N,1) for some N.

    basis : {"std", "gm", "pp", "qt"}
        The abbreviation for the basis used to interpret rhoVec
        ("gm" = Gell-Mann, "pp" = Pauli-product, "std" = matrix unit,
         "qt" = qutrit, or standard).

    Returns
    -------
    float
    """
    # rhoVec must be positive semidefinite and trace = 1
    rhoMx = _bt.vec_to_stdmx(_np.asarray(rhoVec), basis)
    evals = _np.linalg.eigvals(rhoMx)  # could use eigvalsh, but wary of this since eigh can be wrong...
    sumOfNeg = sum([-ev.real for ev in evals if ev.real < 0])
    tracePenalty = abs(rhoVec[0, 0] - (1.0 / _np.sqrt(rhoMx.shape[0])))
    # 0th el is coeff of I(dxd)/sqrt(d) which has trace sqrt(d)
    #print "Sum of neg = ",sumOfNeg  #DEBUG
    #print "Trace Penalty = ",tracePenalty  #DEBUG
    return sumOfNeg + tracePenalty


#UNUSED (REMOVE?)
def effect_penalty(EVec, basis):
    """
    Penalty assigned to a POVM effect vector EVec. Effects
      must have eigenvalues between 0 and 1.  A positive return
      value indicates this criterion is not met and the E-vector
      is invalid.

    Parameters
    ----------
    EVec : numpy array
         effect vector array of shape (N,1) for some N.

    basis : {"std", "gm", "pp", "qt"}
        The abbreviation for the basis used to interpret EVec
        ("gm" = Gell-Mann, "pp" = Pauli-product, "std" = matrix unit,
         "qt" = qutrit, or standard).

    Returns
    -------
    float
    """
    # EVec must have eigenvalues between 0 and 1
    EMx = _bt.vec_to_stdmx(_np.asarray(EVec), basis)
    evals = _np.linalg.eigvals(EMx)  # could use eigvalsh, but wary of this since eigh can be wrong...
    sumOfPen = 0
    for ev in evals:
        if ev.real < 0: sumOfPen += -ev.real
        if ev.real > 1: sumOfPen += ev.real - 1.0
    return sumOfPen


#UNUSED (REMOVE?)
def cptp_penalty(model, include_spam_penalty=True):
    """
    The sum of all negative Choi matrix eigenvalues, and
      if include_spam_penalty is True, the rho-vector and
      E-vector penalties of model.  A non-zero value
      indicates that the model is not CPTP.

    Parameters
    ----------
    model : Model
        the model to compute CPTP penalty for.

    include_spam_penalty : bool, optional
        if True, also test model for invalid SPAM
        operation(s) and return sum of CPTP penalty
        with rhoVecPenlaty(...) and effect_penalty(...)
        for each rho and E vector.

    Returns
    -------
    float
        CPTP penalty (possibly with added spam penalty).
    """
    ret = _jam.sum_of_negative_choi_evals(model)
    if include_spam_penalty:
        b = model.basis
        ret += sum([prep_penalty(r, b) for r in model.preps.values()])
        ret += sum([effect_penalty(e, b) for povm in model.povms.values()
                    for e in povm.values()])
    return ret


#TODO: incorporate this into objective functions, or utilize an ObjectiveFunction
# to implement this.
def two_delta_loglfn(N, p, f, minProbClip=1e-6, poissonPicture=True):
    """
    Term of the 2*[log(L)-upper-bound - log(L)] sum corresponding
     to a single operation sequence and spam label.

    Parameters
    ----------
    N : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    minProbClip : float, optional
        Minimum probability clip point to avoid evaluating
        log(number <= zero)

    poissonPicture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logl value.

    Returns
    -------
    float or numpy array
    """
    #TODO: change this function to handle nan's in the inputs without warnings, since
    # fiducial pair reduction may pass inputs with nan's legitimately and the desired
    # behavior is to just let the nan's pass through to nan's in the output.
    cp = _np.clip(p, minProbClip, 1e10)  # effectively no upper bound

    nan_indices = _np.isnan(f)  # get indices of invalid entries
    if not _np.isscalar(f): f[nan_indices] = 0.0
    #set nan's to zero to avoid RuntimeWarnings (invalid value)

    zf = _np.where(f < 1e-10, 0.0, f)  # set zero-freqs to zero
    nzf = _np.where(f < 1e-10, 1.0, f)  # set zero-freqs to one -- together
    # w/above line makes 0 * log(0) == 0
    if not _np.isscalar(f):
        zf[nan_indices] = _np.nan  # set nan indices back to nan
        nzf[nan_indices] = _np.nan  # set nan indices back to nan

    if poissonPicture:
        return 2 * (N * zf * _np.log(nzf / cp) - N * (f - cp))
    else:
        return 2 * N * zf * _np.log(nzf / cp)


#UNUSED (REMOVE?)
def _patched_logl_fn(N, p, min_p):
    """ N * log(p) with min-prob-clip patching """
    if N == 0: return 0.0
    S = N / min_p               # slope term that is derivative of logl at min_p
    S2 = -0.5 * N / (min_p**2)  # 2nd derivative of logl term at min_p
    pos_p = max(min_p, p)
    v = N * _np.log(pos_p)
    if p < min_p:
        v += S * (p - min_p) + S2 * (p - min_p)**2  # quadratic extrapolation of logl at min_p for p < min_p
    return v


##############################################################################################
#   FUNCTIONS FOR HESSIAN ANALYSIS (which take derivatives of the log(likelihood) function)  #
##############################################################################################


#def dlogl_analytic(model, dataset):
#    nP = model.num_params()
#    result = _np.zeros([1,nP])
#    dPmx = dpr_plus(model, [circuit for circuit in dataset])
#
#    for (k,d) in enumerate(dataset.values()):
#        p = model.PrPlus(d.circuit)
#        if _np.fabs(p) < TOL and round(d.nPlus) == 0: continue
#        if _np.fabs(p - 1) < TOL and round(d.nMinus) == 0: continue
#
#        for i in range(nP):
#            #pre = ((1-p)*d.nPlus - p*d.nMinus) / (p*(1-p))
#            #print "%d: Pre(%s) = " % (i,d.circuit), pre, "  (p = %g, np = %g)" % (p, d.nPlus)
#            result[0,i] += ((1-p)*d.nPlus - p*d.nMinus) / (p*(1-p)) * dPmx[i,k]
#
#    return result
#
#
#def dlogl_finite_diff(model, dataset):
#    return numerical_deriv(logl, model, dataset, 1)
#
#def logl_hessian_finite_diff(model, dataset):
#    return numerical_deriv(dlogl_finite_diff, model, dataset, model.num_params())
#
#def logl_hessian_at_ml(model, circuits, nSamples):
#    return nSamples * logl_hessian_at_ML_per_sample(model, circuits)
#
#def logl_hessian_at_ML_per_sample(model, circuits):
#    nP = model.num_params()
#    result = _np.zeros([nP,nP])
#
#    dPmx = dpr_plus(model, circuits)
#
#    for (k,s) in enumerate(circuits):
#        p = model.PrPlus(s)
#        if _np.fabs(p) < TOL: continue
#        if _np.fabs(p - 1) < TOL: continue
#        for i in range(nP):
#            for j in range(nP):
#                result[i,j] += -1.0/(p*(1-p)) * dPmx[i,k] * dPmx[j,k]
#
#    return result
#
#
#
#def dpr_plus(model, circuits):
#    DELTA = 1e-7
#    nP = model.num_params()
#    nCircuits = len(circuits)
#    result = _np.zeros([nP,nCircuits])
#
#    for (j,s) in enumerate(circuits):
#        fMid = model.PrPlus(s)
#
#        for i in range(nP):
#            mdl = model.copy()
#            mdl.add_to_param(i,DELTA)
#            fRight = mdl.PrPlus(s)
#            mdl.add_to_param(i,-2*DELTA)
#            fLeft = mdl.PrPlus(s)
#
#            if fRight is None and fLeft is None:
#                raise ValueError("Cannot take derivative - both sides are out of bounds!")
#            if fRight is None:
#                dP = (fMid - fLeft) / DELTA
#            elif fLeft is None:
#                dP = (fRight - fMid) / DELTA
#            else:
#                dP = (fRight - fLeft) / (2*DELTA)
#
#            result[i,j] = dP
#
#    return result
#
#
#def numerical_deriv(fnToDifferentiate, model, dataset, resultLen):
#    DELTA = 1e-6
#    nP = model.num_params()
#    result = _np.zeros([resultLen,nP])
#
#    fMid = fnToDifferentiate(model, dataset)
#    if fMid is None: return None
#
#    for i in range(nP):
#        mdl = model.copy()
#        mdl.add_to_param(i,DELTA)
#        fRight = fnToDifferentiate(mdl, dataset)
#
#        mdl = model.copy()
#        mdl.add_to_param(i,-DELTA)
#        fLeft = fnToDifferentiate(mdl, dataset)
#
#        #print "DEBUG: %d: l,m,r = " % i,(fLeft,fMid,fRight)
#        if fRight is None and fLeft is None:
#            raise ValueError("numerical_deriv cannot take derivative - both sides are out of bounds!")
#
#        if fRight is None:
#            df = (fMid - fLeft) / DELTA
#        elif fLeft is None:
#            df = (fRight - fMid) / DELTA
#        else:
#            df = (fRight - fLeft) / (2*DELTA)
#
#        #print "DEBUG: df(%d) = " % i,df
#        result[:,i] = _np.transpose(df)
#
#    return result
