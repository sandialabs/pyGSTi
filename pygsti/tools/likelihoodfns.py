"""
Functions related to computation of the log-likelihood.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings

import numpy as _np
import scipy.stats as _stats

from pygsti.tools import basistools as _bt
from pygsti.tools import jamiolkowski as _jam
from pygsti.tools import listtools as _lt

#from ..baseobjs.smartcache import smart_cached

TOL = 1e-20

# The log(Likelihood) within the standard (non-Poisson) picture is:
#
# L = prod_{i,sl} p_{i,sl}^N_{i,sl}
#
# Where i indexes the circuit, and sl indexes the spam label.  N[i] is the total counts
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
# description when the probabilities (for a given circuit) may not sum to one, by
# interpreting them each as rates.  In the standard picture, large circuit probabilities
# are not penalized (each standard logL term increases monotonically with each probability,
# and the reason this is ok when the probabilities sum to one is that for a probabilility
# that gets close to 1, there's another that is close to zero, and logL is very negative
# near zero.

# The log(Likelihood) within the Poisson picture is:
#
# L = prod_{i,sl} lambda_{i,sl}^N_{i,sl} e^{-lambda_{i,sl}} / N_{i,sl}!
#
# Where lamba_{i,sl} := p_{i,sl}*N[i] is a rate, i indexes the circuit,
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
#   F(p) = piecewise{ if( p>r ) then p; else -(1/3)*p^3/r^2 + p^2/r + (1/3)*r }  y = p/r
#     Note p < r portion:  F(p) = p/3 * (-p^2/r^2 + 3*p/r + r)  -- zero at p=0, p at p=r
#                     1st deriv = -p^2/r^2 + 2*p/r = -p/r (p/r - 2)  -- matches slope=1 at p=r
#                     2nd deriv = -2*p/r^2 + 2/r = -2/r * (p/r - 1) -- curvature=0 at p=r
#  OLD: quadratic that doesn't match 2nd-deriv:
#   F(p) = piecewise{ if( p>r ) then p; else (r-p)^2/(2*r) + p }


def logl(model, dataset, circuits=None,
         min_prob_clip=1e-6, prob_clip_interval=(-1e6, 1e6), radius=1e-4,
         poisson_picture=True, op_label_aliases=None, wildcard=None,
         mdc_store=None, comm=None, mem_limit=None):
    """
    The log-likelihood function.

    Parameters
    ----------
    model : Model
        Model of parameterized gates

    dataset : DataSet
        Probability data

    circuits : list of (tuples or Circuits), optional
        Each element specifies a circuit to include in the log-likelihood
        sum.  Default value of None implies all the circuits in dataset
        should be used.

    min_prob_clip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    prob_clip_interval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poisson_picture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logl value.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    wildcard : WildcardBudget
        A wildcard budget to apply to this log-likelihood computation.
        This increases the returned log-likelihood value by adjusting
        (by a maximal amount measured in TVD, given by the budget) the
        probabilities produced by `model` to optimially match the data
        (within the bugetary constraints) evaluating the log-likelihood.

    mdc_store : ModelDatasetCircuitsStore, optional
        An object that bundles cached quantities along with a given model, dataset, and circuit
        list.  If given, `model` and `dataset` and `circuits` should be set to None.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    Returns
    -------
    float
        The log likelihood
    """
    v = logl_per_circuit(model, dataset, circuits,
                         min_prob_clip, prob_clip_interval, radius,
                         poisson_picture, op_label_aliases, wildcard,
                         mdc_store, comm, mem_limit)
    return _np.sum(v)  # sum over *all* dimensions


def logl_per_circuit(model, dataset, circuits=None,
                     min_prob_clip=1e-6, prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                     poisson_picture=True, op_label_aliases=None, wildcard=None,
                     mdc_store=None, comm=None, mem_limit=None):
    """
    Computes the per-circuit log-likelihood contribution for a set of circuits.

    Parameters
    ----------
    model : Model
        Model of parameterized gates

    dataset : DataSet
        Probability data

    circuits : list of (tuples or Circuits), optional
        Each element specifies a circuit to include in the log-likelihood
        sum.  Default value of None implies all the circuits in dataset
        should be used.

    min_prob_clip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    prob_clip_interval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poisson_picture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logl value.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    wildcard : WildcardBudget
        A wildcard budget to apply to this log-likelihood computation.
        This increases the returned log-likelihood value by adjusting
        (by a maximal amount measured in TVD, given by the budget) the
        probabilities produced by `model` to optimially match the data
        (within the bugetary constraints) evaluating the log-likelihood.

    mdc_store : ModelDatasetCircuitsStore, optional
        An object that bundles cached quantities along with a given model, dataset, and circuit
        list.  If given, `model` and `dataset` and `circuits` should be set to None.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    Returns
    -------
    numpy.ndarray
        Array of length either `len(circuits)` or `len(dataset.keys())`.
        Values are the log-likelihood contributions of the corresponding
        circuit aggregated over outcomes.
    """
    from ..objectivefns import objectivefns as _objfns
    regularization = {'min_prob_clip': min_prob_clip, 'radius': radius} if poisson_picture \
        else {'min_prob_clip': min_prob_clip}  # non-poisson-pic logl has no radius
    obj_max = _objfns._objfn(_objfns.MaxLogLFunction, model, dataset, circuits,
                             mdc_store=mdc_store, method_names=('percircuit',),
                             op_label_aliases=op_label_aliases, poisson_picture=poisson_picture)
    obj_cls = _objfns.PoissonPicDeltaLogLFunction if poisson_picture else _objfns.DeltaLogLFunction
    obj = _objfns._objfn(obj_cls, model, dataset, circuits,
                         regularization, {'prob_clip_interval': prob_clip_interval},
                         op_label_aliases, comm, mem_limit, ('percircuit',), (), mdc_store)

    if wildcard:
        assert(poisson_picture), "Wildcard budgets can only be used with `poisson_picture=True`"
        obj.terms()  # objfn used within wildcard objective fn must be pre-evaluated
        obj = _objfns.LogLWildcardFunction(obj, model.to_vector(), wildcard)

    local = obj_max.percircuit() - obj.percircuit()
    return obj.layout.allgather_local_array('c', local)


def logl_jacobian(model, dataset, circuits=None,
                  min_prob_clip=1e-6, prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                  poisson_picture=True, op_label_aliases=None, mdc_store=None,
                  comm=None, mem_limit=None, verbosity=0):
    """
    The jacobian of the log-likelihood function.

    Parameters
    ----------
    model : Model
        Model of parameterized gates (including SPAM)

    dataset : DataSet
        Probability data

    circuits : list of (tuples or Circuits), optional
        Each element specifies a circuit to include in the log-likelihood
        sum.  Default value of None implies all the circuits in dataset
        should be used.

    min_prob_clip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    prob_clip_interval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poisson_picture : boolean, optional
        Whether the Poisson-picutre log-likelihood should be differentiated.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    mdc_store : ModelDatasetCircuitsStore, optional
        An object that bundles cached quantities along with a given model, dataset, and circuit
        list.  If given, `model` and `dataset` and `circuits` should be set to None.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    verbosity : int, optional
        How much detail to print to stdout.

    Returns
    -------
    numpy array
        array of shape (M,), where M is the length of the vectorized model.
    """
    from ..objectivefns import objectivefns as _objfns
    regularization = {'min_prob_clip': min_prob_clip, 'radius': radius} if poisson_picture \
        else {'min_prob_clip': min_prob_clip}  # non-poisson-pic logl has no radius
    obj_cls = _objfns.PoissonPicDeltaLogLFunction if poisson_picture else _objfns.DeltaLogLFunction
    obj = _objfns._objfn(obj_cls, model, dataset, circuits,
                         regularization, {'prob_clip_interval': prob_clip_interval},
                         op_label_aliases, comm, mem_limit, ('jacobian',), (), mdc_store, verbosity)
    local = -obj.jacobian()  # negative b/c objective is deltaLogL = max_logl - logL
    return obj.layout.allgather_local_array('ep', local)


def logl_hessian(model, dataset, circuits=None,
                 min_prob_clip=1e-6, prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                 poisson_picture=True, op_label_aliases=None, mdc_store=None,
                 comm=None, mem_limit=None, verbosity=0):
    """
    The hessian of the log-likelihood function.

    Parameters
    ----------
    model : Model
        Model of parameterized gates (including SPAM)

    dataset : DataSet
        Probability data

    circuits : list of (tuples or Circuits), optional
        Each element specifies a circuit to include in the log-likelihood
        sum.  Default value of None implies all the circuits in dataset
        should be used.

    min_prob_clip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    prob_clip_interval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by
        models during MLEGST's search for an optimal model (if not None).
        if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poisson_picture : boolean, optional
        Whether the Poisson-picutre log-likelihood should be differentiated.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    mdc_store : ModelDatasetCircuitsStore, optional
        An object that bundles cached quantities along with a given model, dataset, and circuit
        list.  If given, `model` and `dataset` and `circuits` should be set to None.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    verbosity : int, optional
        How much detail to print to stdout.

    Returns
    -------
    numpy array or None
        On the root processor, the Hessian matrix of shape (nModelParams, nModelParams),
        where nModelParams = `model.num_params`.  `None` on non-root processors.
    """
    from ..objectivefns import objectivefns as _objfns
    regularization = {'min_prob_clip': min_prob_clip, 'radius': radius} if poisson_picture \
        else {'min_prob_clip': min_prob_clip}  # non-poisson-pic logl has no radius
    obj_cls = _objfns.PoissonPicDeltaLogLFunction if poisson_picture else _objfns.DeltaLogLFunction
    obj = _objfns._objfn(obj_cls, model, dataset, circuits,
                         regularization, {'prob_clip_interval': prob_clip_interval},
                         op_label_aliases, comm, mem_limit, ('hessian',), (), mdc_store, verbosity)
    hessian = obj.hessian()  # Note: hessian is only assembled on root processor
    return -hessian if (comm is None or comm.rank == 0) else None
    # negative b/c objective is deltaLogL = max_logl - logL


def logl_approximate_hessian(model, dataset, circuits=None,
                             min_prob_clip=1e-6, prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                             poisson_picture=True, op_label_aliases=None, mdc_store=None,
                             comm=None, mem_limit=None, verbosity=0):
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

    circuits : list of (tuples or Circuits), optional
        Each element specifies a circuit to include in the log-likelihood
        sum.  Default value of None implies all the circuits in dataset
        should be used.

    min_prob_clip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    prob_clip_interval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poisson_picture : boolean, optional
        Whether the Poisson-picutre log-likelihood should be differentiated.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    mdc_store : ModelDatasetCircuitsStore, optional
        An object that bundles cached quantities along with a given model, dataset, and circuit
        list.  If given, `model` and `dataset` and `circuits` should be set to None.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    mem_limit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    verbosity : int, optional
        How much detail to print to stdout.

    Returns
    -------
    numpy array or None
        On the root processor, the approximate Hessian matrix of shape (nModelParams, nModelParams),
        where nModelParams = `model.num_params`.  `None` on non-root processors.
    """
    from ..objectivefns import objectivefns as _objfns
    obj_cls = _objfns.PoissonPicDeltaLogLFunction if poisson_picture else _objfns.DeltaLogLFunction
    obj = _objfns._objfn(obj_cls, model, dataset, circuits,
                         {'min_prob_clip': min_prob_clip,
                          'radius': radius},
                         {'prob_clip_interval': prob_clip_interval},
                         op_label_aliases, comm, mem_limit, ('approximate_hessian',), (), mdc_store, verbosity)
    hessian = obj.approximate_hessian()  # Note: hessian is only assembled on root processor
    return -hessian if (comm is None or comm.rank == 0) else None
    # negative b/c objective is deltaLogL = max_logl - logL


def logl_max(model, dataset, circuits=None, poisson_picture=True,
             op_label_aliases=None, mdc_store=None):
    """
    The maximum log-likelihood possible for a DataSet.

    That is, the log-likelihood obtained by a maximal model that can
    fit perfectly the probability of each circuit.

    Parameters
    ----------
    model : Model
        the model, used only for circuit compilation

    dataset : DataSet
        the data set to use.

    circuits : list of (tuples or Circuits), optional
        Each element specifies a circuit to include in the max-log-likelihood
        sum.  Default value of None implies all the circuits in dataset should
        be used.

    poisson_picture : boolean, optional
        Whether the Poisson-picture maximum log-likelihood should be returned.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    mdc_store : ModelDatasetCircuitsStore, optional
        An object that bundles cached quantities along with a given model, dataset, and circuit
        list.  If given, `model` and `dataset` and `circuits` should be set to None.

    Returns
    -------
    float
    """
    from ..objectivefns import objectivefns as _objfns
    obj_max = _objfns._objfn(_objfns.MaxLogLFunction, model, dataset, circuits, mdc_store=mdc_store,
                             op_label_aliases=op_label_aliases, poisson_picture=poisson_picture, method_names=('fn',))
    return obj_max.fn()  # gathers internally


def logl_max_per_circuit(model, dataset, circuits=None,
                         poisson_picture=True, op_label_aliases=None, mdc_store=None):
    """
    The vector of maximum log-likelihood contributions for each circuit, aggregated over outcomes.

    Parameters
    ----------
    model : Model
        the model, used only for circuit compilation

    dataset : DataSet
        the data set to use.

    circuits : list of (tuples or Circuits), optional
        Each element specifies a circuit to include in the max-log-likelihood
        sum.  Default value of None implies all the circuits in dataset should
        be used.

    poisson_picture : boolean, optional
        Whether the Poisson-picture maximum log-likelihood should be returned.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    mdc_store : ModelDatasetCircuitsStore, optional
        An object that bundles cached quantities along with a given model, dataset, and circuit
        list.  If given, `model` and `dataset` and `circuits` should be set to None.

    Returns
    -------
    numpy.ndarray
        Array of length either `len(circuits)` or `len(dataset.keys())`.
        Values are the maximum log-likelihood contributions of the corresponding
        circuit aggregated over outcomes.
    """
    from ..objectivefns import objectivefns as _objfns
    obj_max = _objfns._objfn(_objfns.MaxLogLFunction, model, dataset, circuits, mdc_store=mdc_store,
                             op_label_aliases=op_label_aliases, poisson_picture=poisson_picture,
                             method_names=('percircuit',))
    local = obj_max.percircuit()
    return obj_max.layout.allgather_local_array('c', local)


def two_delta_logl_nsigma(model, dataset, circuits=None,
                          min_prob_clip=1e-6, prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                          poisson_picture=True, op_label_aliases=None,
                          dof_calc_method='modeltest', wildcard=None):
    """
    See docstring for :func:`pygsti.tools.two_delta_logl`

    Parameters
    ----------
    model : Model
        Model of parameterized gates

    dataset : DataSet
        Probability data

    circuits : list of (tuples or Circuits), optional
        Each element specifies a circuit to include in the log-likelihood
        sum.  Default value of None implies all the circuits in dataset
        should be used.

    min_prob_clip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    prob_clip_interval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poisson_picture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logl value.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    dof_calc_method : {"all", "modeltest"}
        How `model`'s number of degrees of freedom (parameters) are obtained
        when computing the number of standard deviations and p-value relative to
        a chi2_k distribution, where `k` is additional degrees of freedom
        possessed by the maximal model.  `"all"` uses `model.num_params` whereas
        `"modeltest"` uses `model.num_modeltest_params` (the number of non-gauge
        parameters by default).

    wildcard : WildcardBudget
        A wildcard budget to apply to this log-likelihood computation.
        This increases the returned log-likelihood value by adjusting
        (by a maximal amount measured in TVD, given by the budget) the
        probabilities produced by `model` to optimially match the data
        (within the bugetary constraints) evaluating the log-likelihood.

    Returns
    -------
    float
    """
    assert(dof_calc_method is not None)
    return two_delta_logl(model, dataset, circuits,
                          min_prob_clip, prob_clip_interval, radius,
                          poisson_picture, op_label_aliases,
                          dof_calc_method, wildcard, None, None)[1]


def two_delta_logl(model, dataset, circuits=None,
                   min_prob_clip=1e-6, prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                   poisson_picture=True, op_label_aliases=None,
                   dof_calc_method=None, wildcard=None,
                   mdc_store=None, comm=None):
    """
    Twice the difference between the maximum and actual log-likelihood.

    Optionally also can return the Nsigma (# std deviations from mean) and
    p-value relative to expected chi^2 distribution (when `dof_calc_method`
    is not None).

    This function's arguments are supersets of :func:`logl`, and
    :func:`logl_max`. This is a convenience function, equivalent to
    `2*(logl_max(...) - logl(...))`, whose value is what is often called
    the *log-likelihood-ratio* between the "maximal model" (that which trivially
    fits the data exactly) and the model given by `model`.

    Parameters
    ----------
    model : Model
        Model of parameterized gates

    dataset : DataSet
        Probability data

    circuits : list of (tuples or Circuits), optional
        Each element specifies a circuit to include in the log-likelihood
        sum.  Default value of None implies all the circuits in dataset
        should be used.

    min_prob_clip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    prob_clip_interval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poisson_picture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the computed log-likelihood values.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    dof_calc_method : {None, "all", "modeltest"}
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

    mdc_store : ModelDatasetCircuitsStore, optional
        An object that bundles cached quantities along with a given model, dataset, and circuit
        list.  If given, `model` and `dataset` and `circuits` should be set to None.

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
    from ..objectivefns import objectivefns as _objfns
    obj_cls = _objfns.PoissonPicDeltaLogLFunction if poisson_picture else _objfns.DeltaLogLFunction
    obj = _objfns._objfn(obj_cls, model, dataset, circuits,
                         {'min_prob_clip': min_prob_clip,
                          'radius': radius},
                         {'prob_clip_interval': prob_clip_interval},
                         op_label_aliases, comm, None, ('terms',), (), mdc_store)

    if wildcard:
        assert(poisson_picture), "Wildcard budgets can only be used with `poisson_picture=True`"
        obj.terms()  # objfn used within wildcard objective fn must be pre-evaluated
        obj = _objfns.LogLWildcardFunction(obj, model.to_vector(), wildcard)

    two_delta_logl = 2 * obj.fn()  # gathers internally

    if dof_calc_method is None:
        return two_delta_logl
    elif dof_calc_method == "modeltest":
        mdl_dof = model.num_modeltest_params
    elif dof_calc_method == "all":
        mdl_dof = model.num_params
    else: raise ValueError("Invalid `dof_calc_method` arg: %s" % dof_calc_method)

    if circuits is not None:
        ds_strs = _lt.apply_aliases_to_circuits(circuits, op_label_aliases)
    else: ds_strs = None

    ds_dof = dataset.degrees_of_freedom(ds_strs)
    k = max(ds_dof - mdl_dof, 1)
    if ds_dof <= mdl_dof:
        _warnings.warn("Max-model params (%d) <= model params (%d)!  Using k == 1." % (ds_dof, mdl_dof))

    nsigma = (two_delta_logl - k) / _np.sqrt(2 * k)
    pvalue = 1.0 - _stats.chi2.cdf(two_delta_logl, k)
    return two_delta_logl, nsigma, pvalue


def two_delta_logl_per_circuit(model, dataset, circuits=None,
                               min_prob_clip=1e-6, prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                               poisson_picture=True, op_label_aliases=None,
                               dof_calc_method=None, wildcard=None,
                               mdc_store=None, comm=None):
    """
    Twice the per-circuit difference between the maximum and actual log-likelihood.

    Contributions are aggregated over each circuit's outcomes, but no further.

    Optionally (when `dof_calc_method` is not None) returns parallel vectors
    containing the Nsigma (# std deviations from mean) and the p-value relative
    to expected chi^2 distribution for each sequence.

    Parameters
    ----------
    model : Model
        Model of parameterized gates

    dataset : DataSet
        Probability data

    circuits : list of (tuples or Circuits), optional
        Each element specifies a circuit to include in the log-likelihood
        sum.  Default value of None implies all the circuits in dataset
        should be used.

    min_prob_clip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    prob_clip_interval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLEGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poisson_picture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logl value.

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are tuples
        corresponding to what that operation label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = ('Gx','Gx','Gx')

    dof_calc_method : {"all", "modeltest"}
        How `model`'s number of degrees of freedom (parameters) are obtained
        when computing the number of standard deviations and p-value relative to
        a chi2_k distribution, where `k` is additional degrees of freedom
        possessed by the maximal model.

    wildcard : WildcardBudget
        A wildcard budget to apply to this log-likelihood computation.
        This increases the returned log-likelihood value by adjusting
        (by a maximal amount measured in TVD, given by the budget) the
        probabilities produced by `model` to optimially match the data
        (within the bugetary constraints) evaluating the log-likelihood.

    mdc_store : ModelDatasetCircuitsStore, optional
        An object that bundles cached quantities along with a given model, dataset, and circuit
        list.  If given, `model` and `dataset` and `circuits` should be set to None.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    twoDeltaLogL_terms : numpy.ndarray
    
    Nsigma, pvalue : numpy.ndarray
        Only returned when `dof_calc_method` is not None.
    """
    from ..objectivefns import objectivefns as _objfns
    obj_cls = _objfns.PoissonPicDeltaLogLFunction if poisson_picture else _objfns.DeltaLogLFunction
    obj = _objfns._objfn(obj_cls, model, dataset, circuits,
                         {'min_prob_clip': min_prob_clip,
                          'radius': radius}, {'prob_clip_interval': prob_clip_interval},
                         op_label_aliases, comm, None, ('percircuit',), (), mdc_store)

    if wildcard:
        assert(poisson_picture), "Wildcard budgets can only be used with `poisson_picture=True`"
        obj.percircuit()  # objfn used within wildcard objective fn must be pre-evaluated
        obj = _objfns.LogLWildcardFunction(obj, model.to_vector(), wildcard)

    two_dlogl_percircuit = 2 * obj.layout.allgather_local_array('c', obj.percircuit())

    if dof_calc_method is None: return two_dlogl_percircuit
    elif dof_calc_method == "all": mdl_dof = model.num_params
    elif dof_calc_method == "modeltest": mdl_dof = model.num_modeltest_params
    else: raise ValueError("Invalid `dof_calc_method` arg: %s" % dof_calc_method)

    if circuits is not None:
        ds_strs = _lt.apply_aliases_to_circuits(circuits, op_label_aliases)
    else: ds_strs = None

    ds_dof = dataset.degrees_of_freedom(ds_strs)
    k = max(ds_dof - mdl_dof, 1)
    # HACK - just take a single average #dof per circuit to use as chi_k distribution!
    k = int(_np.ceil(k / (1.0 * len(circuits))))

    nsigma = (two_dlogl_percircuit - k) / _np.sqrt(2 * k)
    pvalue = _np.array([1.0 - _stats.chi2.cdf(x, k) for x in two_dlogl_percircuit], 'd')
    return two_dlogl_percircuit, nsigma, pvalue


def two_delta_logl_term(n, p, f, min_prob_clip=1e-6, poisson_picture=True):
    """
    Term of the 2*[log(L)-upper-bound - log(L)] sum corresponding to a single circuit and spam label.

    Parameters
    ----------
    n : float or numpy array
        Number of samples.

    p : float or numpy array
        Probability of 1st outcome (typically computed).

    f : float or numpy array
        Frequency of 1st outcome (typically observed).

    min_prob_clip : float, optional
        Minimum probability clip point to avoid evaluating
        log(number <= zero)

    poisson_picture : boolean, optional
        Whether the log-likelihood-in-the-Poisson-picture terms should be included
        in the returned logl value.

    Returns
    -------
    float or numpy array
    """
    from ..objectivefns import objectivefns as _objfns
    #Allow this function to pass NaNs through silently, since
    # fiducial pair reduction may pass inputs with nan's legitimately and the desired
    # behavior is to just let the nan's pass through to nan's in the output.

    nan_indices = _np.isnan(f)  # get indices of invalid entries
    if not _np.isscalar(f):
        f = f.copy(); p = p.copy(); n = n.copy()
        f[nan_indices] = p[nan_indices] = n[nan_indices] = 0.0  # so computation runs fine

    if poisson_picture:
        rawfn = _objfns.RawPoissonPicDeltaLogLFunction({'min_prob_clip': min_prob_clip})
    else:
        rawfn = _objfns.RawDeltaLogLFunction({'min_prob_clip': min_prob_clip})

    ret = 2 * rawfn.terms(p, n * f, n, f)
    if not _np.isscalar(f):
        ret[nan_indices] = _np.nan
    return ret
