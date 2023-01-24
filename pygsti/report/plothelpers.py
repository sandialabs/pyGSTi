"""
Helper Functions for generating plots
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings

import numpy as _np

from pygsti import tools as _tools
from pygsti.objectivefns import objectivefns as _objfns
from pygsti.circuits.circuitlist import CircuitList as _CircuitList
from pygsti.baseobjs.smartcache import smart_cached


def small_eigenvalue_err_rate(sigma, direct_gst_models):
    """
    Compute per-gate error rate.

    The per-gate error rate, extrapolated from the smallest eigvalue
    of the Direct GST estimate of the given circuit sigma.

    Parameters
    ----------
    sigma : Circuit or tuple of operation labels
        The gate sequence that is used to estimate the error rate

    direct_gst_models : dictionary of Models
        A dictionary with keys = circuits and
        values = Models.

    Returns
    -------
    float
        the approximate per-gate error rate.
    """
    if sigma is None: return _np.nan  # in plot processing, "None" circuits = no plot output = nan values
    mdl_direct = direct_gst_models[sigma]
    minEigval = min(abs(_np.linalg.eigvals(mdl_direct.operations["GsigmaLbl"])))
    # (approximate) per-gate error rate; max averts divide by zero error
    return 1.0 - minEigval**(1.0 / max(len(sigma), 1))


def _eformat(f, prec):
    """
    Formatting routine for writing compact representations of
    numbers in plot boxes
    """
    if _np.isnan(f): return ""  # show NAN as blanks
    if prec == 'compact' or prec == 'compacthp':
        if f < 0:
            ef = _eformat(-f, prec)
            return "-" + ef if (ef != "0") else "0"

        if prec == 'compacthp':
            if f <= 0.5e-9:  # can't fit in 3 digits; 1e-9 = "1m9" is the smallest 3-digit (not counting minus signs)
                return "0"
            if f < 0.005:  # then need scientific notation since 3-digit float would be 0.00...
                s = "%.0e" % f
                try:
                    mantissa, exp = s.split('e')
                    exp = int(exp); assert(exp < 0)
                    if exp < -9: return "0"  # should have been caugth above, but just in case
                    return "%sm%d" % (mantissa, -exp)
                except:
                    return "?"
            if f < 1:
                z = "%.2f" % f  # print first two decimal places
                if z.startswith("0."): return z[1:]  # fails for '1.00'; then thunk down to next f<10 case
            if f < 10:
                return "%.1f" % f  # print whole number and tenths

        if f < 100:
            return "%.0f" % f  # print nearest whole number if only 1 or 2 digits

        #if f >= 100, minimal scientific notation, such as "4e7", not "4e+07"
        s = "%.0e" % f
        try:
            mantissa, exp = s.split('e')
            exp = int(exp)
            if exp >= 100: return "B"  # if number is too big to print
            if exp >= 10: return "*%d" % exp
            return "%se%d" % (mantissa, exp)
        except:
            return str(s)[0:3]

    elif type(prec) == int:
        if prec >= 0:
            return "%.*f" % (prec, f)
        else:
            return "%.*g" % (-prec, f)
    else:
        return "%g" % f  # fallback to general format


def _num_non_nan(array):
    ixs = _np.where(_np.isnan(_np.array(array).flatten()) == False)[0]  # noqa: E712
    return int(len(ixs))


def _all_same(items):
    return all(x == items[0] for x in items)


def _compute_num_boxes_dof(sub_mxs, sum_up, element_dof):
    """
    A helper function to compute the number of boxes, and corresponding
    number of degrees of freedom, for the GST chi2/logl boxplots.

    """
    if sum_up:
        s = _np.shape(sub_mxs)
        # Reshape the sub_mxs into a "flattened" form (as opposed to a
        # two-dimensional one)
        reshape_subMxs = _np.array(_np.reshape(sub_mxs, (s[0] * s[1], s[2], s[3])))

        #Get all the boxes where the entries are not all NaN
        non_all_NaN = reshape_subMxs[_np.where(_np.array([_np.isnan(k).all() for k in reshape_subMxs]) == False)]  # noqa: E712,E501
        s = _np.shape(non_all_NaN)
        dof_each_box = [_num_non_nan(k) * element_dof for k in non_all_NaN]

        # Don't assert this anymore -- just use average below
        if not _all_same(dof_each_box):
            _warnings.warn('Number of degrees of freedom different for different boxes!')

        # The number of boxes is equal to the number of rows in non_all_NaN
        n_boxes = s[0]

        if n_boxes > 0:
            # Each box is a chi2_(sum) random variable
            dof_per_box = _np.average(dof_each_box)
        else:
            dof_per_box = None  # unknown, since there are no boxes
    else:
        # Each box is a chi2_m random variable currently dictated by the number of
        # dataset degrees of freedom.
        dof_per_box = element_dof

        # Gets all the non-NaN boxes, flattens the resulting
        # array, and does the sum.
        n_boxes = _np.sum(~_np.isnan(sub_mxs).flatten())

    return n_boxes, dof_per_box


#@smart_cached
def _compute_sub_mxs(gss, model, sub_mx_creation_fn, dataset=None, sub_mx_creation_fn_extra_arg=None):
    subMxs = [[sub_mx_creation_fn(gss.plaquette(x, y, True), x, y, sub_mx_creation_fn_extra_arg)
               for x in gss.used_xs] for y in gss.used_ys]
    #Note: subMxs[y-index][x-index] is proper usage
    return subMxs
    
#define a modified version that is meant for working with CircuitList objects of lists of them.
#@smart_cached
def _compute_sub_mxs_circuit_list(circuit_lists, model, sub_mx_creation_fn, dataset=None, sub_mx_creation_fn_extra_arg=None):
    subMxs = [sub_mx_creation_fn(circuit_list, sub_mx_creation_fn_extra_arg) for circuit_list in circuit_lists]

    return subMxs

@smart_cached
def dscompare_llr_matrices(gsplaq, dscomparator):
    """
    Computes matrix of 2*log-likelihood-ratios comparing the data of `dscomparator`.

    Parameters
    ----------
    gsplaq : CircuitPlaquette
        Obtained via :method:`CircuitStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which circuits
        they correspond to.

    dscomparator : DataComparator
        The object specifying the data to be compared.

    Returns
    -------
    numpy array of shape ( len(effect_strs), len(prep_strs) )
        log-likelihood-ratio values corresponding to the circuits
        where a base circuit is sandwiched between the each prep-fiducial and
        effect-fiducial pair.
    """
    ret = _np.nan * _np.ones((gsplaq.num_rows, gsplaq.num_cols), 'd')
    for i, j, opstr in gsplaq:
        ret[i, j] = dscomparator.llrs[opstr]
    return ret


@smart_cached
def genericdict_matrices(gsplaq, genericdict):
    ret = _np.nan * _np.ones((gsplaq.num_rows, gsplaq.num_cols), 'd')
    for i, j, opstr in gsplaq:
        ret[i, j] = genericdict[opstr]
    return ret


@smart_cached
def drift_neglog10pvalue_matrices(gsplaq, drifttuple):
    """
    Computes matrix of -log10(pvalues) for testing the stable-circuit ("no drift") null hypothesis in each circuit.

    This uses the "max power in spectra" test.

    Parameters
    ----------
    gsplaq : CircuitPlaquette
        Obtained via :method:`CircuitStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which circuits
        they correspond to.

    drifttuple : 2-tuple
        The first element of the tuple is a StabilityAnalyzer. The second element is a
        tuple that specifies the hypothesis test(s) from which to extract the p-values.
        This can be None, and then the default is used.

    Returns
    -------
    numpy array of shape ( len(effect_strs), len(prep_strs) )
        -log10(pvalues) for testing the "no drift" null hypothesis, using the "max power in
        spectra" test, on the relevant sequences. This circuits correspond to the
        circuits where a base circuit is sandwiched between the each prep-fiducial
        and effect-fiducial pair.
    """
    ret = _np.nan * _np.ones((gsplaq.num_rows, gsplaq.num_cols), 'd')
    stabilityanalyzer = drifttuple[0]
    dictlabel = drifttuple[1]
    assert(dictlabel == ('circuit',)), "Currently can only create these matrices for this single type of test!"
    for i, j, opstr in gsplaq:
        try:
            pval = stabilityanalyzer.maximum_power_pvalue(dictlabel={'circuit': opstr}, cutoff=1e-16)
            ret[i, j] = -1 * _np.log10(pval)
        except:
            pass
    return ret


@smart_cached
def drift_maxtvd_matrices(gsplaq, drifttuple):
    """
    Computes matrix of max-tvds for quantifying the size of any detected drift.

    Parameters
    ----------
    gsplaq : CircuitPlaquette
        Obtained via :method:`CircuitStructure.get_plaquette`, this object
        specifies which matrix indices should be computed and which circuits
        they correspond to.

    drifttuple : 2-tuple
        The first element of the tuple is a StabilityAnalyzer. The second element is a
        tuple that specifies the estimatorkey, and the third element is an estimator
        name, that specifies the estimates to use (both can be None, and then the
        default is used).

    Returns
    -------
    numpy array of shape ( len(effect_strs), len(prep_strs) )
        The max tvd for quantifying deviations from the data mean. This
        circuits correspond to the circuits where a base circuit
        is sandwiched between the each prep-fiducial and effect-fiducial pair.
    """
    ret = _np.nan * _np.ones((gsplaq.num_rows, gsplaq.num_cols), 'd')
    stabilityanalyzer = drifttuple[0]
    estimatekey = drifttuple[1]
    estimator = drifttuple[2]
    for i, j, opstr in gsplaq:
        try:
            ret[i, j] = stabilityanalyzer.maximum_tvd_bound(opstr, dskey=None,
                                                            estimatekey=estimatekey, estimator=estimator)
        except:
            pass
    return ret


def rated_n_sigma(dataset, model, circuits, objfn_builder, np=None, wildcard=None, return_all=False, comm=None):
    """
    Computes the number of standard deviations of model violation between `model` and `data`.

    Function compares the data in `dataset` with the `model` model at the "points" (circuits)
    specified by `circuits`.

    Parameters
    ----------
    dataset : DataSet
        The data set.

    model : Model
        The model (model).

    circuits : CircuitList or list of Circuits
        The circuits to use when computing the model violation.  A
        :class:`CircuitList` object may be given to include additional information
        (e.g. aliases) along with the list of circuits.

    objfn_builder : ObjectiveFunctionBuilder
        Builds the objective function to be used to compute the model violation.

    np : int, optional
        The number of free parameters in the model.  If None, then
        `model.num_nongauge_params` is used.

    wildcard : WildcardBudget
        A wildcard budget to apply to the objective function (`objective`),
        which increases the goodness of fit by adjusting (by an amount measured
        in TVD) the probabilities produced by `model` before comparing with
        the frequencies in `dataset`.  Currently, this functionality is only
        supported for `objective == "logl"`.

    return_all : bool, optional
        Returns additional information such as the raw and expected model
        violation (see below).

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    Nsig : float
        The number of sigma of model violaition
    rating : int
        A 1-5 rating (e.g. "number of stars") used to indicate the rough
        abililty of the model to fit the data (better fit = higher rating).
    modelViolation : float
        The raw value of the objective function.  Only returned when
        `return_all==True`.
    expectedViolation : float
        The expected value of the objective function.  Only returned when
        `return_all==True`.
    Ns, np : int
        The number of dataset and model parameters, respectively. Only
        returned when `return_all==True`.
    """
    if isinstance(objfn_builder, str):
        objfn_builder = _objfns.ObjectiveFunctionBuilder.create_from(objfn_builder)

    objfn = objfn_builder.build(model, dataset, circuits, {'comm': comm})
    if wildcard:
        objfn.terms()  # objfn used within wildcard objective fn must be pre-evaluated
        objfn = _objfns.LogLWildcardFunction(objfn, model.to_vector(), wildcard)
    fitqty = objfn.chi2k_distributed_qty(objfn.fn())

    aliases = circuits.op_label_aliases if isinstance(circuits, _CircuitList) else None
    ds_gstrs = _tools.apply_aliases_to_circuits(circuits, aliases)
    np = model.num_modeltest_params

    Ns = dataset.degrees_of_freedom(ds_gstrs)  # number of independent parameters in dataset
    k = max(Ns - np, 1)  # expected chi^2 or 2*(logL_ub-logl) mean
    Nsig = (fitqty - k) / _np.sqrt(2 * k)
    if Ns <= np: _warnings.warn("Max-model params (%d) <= model params (%d)!  Using k == 1." % (Ns, np))
    #pv = 1.0 - _stats.chi2.cdf(chi2,k) # reject GST model if p-value < threshold (~0.05?)

    if Nsig <= 2: rating = 5
    elif Nsig <= 20: rating = 4
    elif Nsig <= 100: rating = 3
    elif Nsig <= 500: rating = 2
    else: rating = 1

    if return_all:
        return Nsig, rating, fitqty, k, Ns, np
    else:
        return Nsig, rating
