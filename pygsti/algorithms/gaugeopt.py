"""
GST gauge optimization algorithms
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import time as _time
import warnings as _warnings

import numpy as _np

from pygsti import baseobjs as _baseobjs
from pygsti import optimize as _opt
from pygsti import tools as _tools
from pygsti.tools import mpitools as _mpit
from pygsti.tools import slicetools as _slct
from pygsti.models.gaugegroup import TrivialGaugeGroupElement as _TrivialGaugeGroupElement


def gaugeopt_to_target(model, target_model, item_weights=None,
                       cptp_penalty_factor=0, spam_penalty_factor=0,
                       gates_metric="frobenius", spam_metric="frobenius",
                       gauge_group=None, method='auto', maxiter=100000,
                       maxfev=None, tol=1e-8, oob_check_interval=0,
                       convert_model_to=None, return_all=False, comm=None,
                       verbosity=0, check_jac=False, n_leak=0):
    """
    Optimize the gauge degrees of freedom of a model to that of a target.

    Parameters
    ----------
    model : Model
        The model to gauge-optimize

    target_model : Model
        The model to optimize to.  The metric used for comparing models
        is given by `gates_metric` and `spam_metric`.

    item_weights : dict, optional
        Dictionary of weighting factors for gates and spam operators.  Keys can
        be gate, state preparation, or POVM effect, as well as the special values
        "spam" or "gates" which apply the given weighting to *all* spam operators
        or gates respectively.  Values are floating point numbers.  Values given
        for specific gates or spam operators take precedence over "gates" and
        "spam" values.  The precise use of these weights depends on the model
        metric(s) being used.

    cptp_penalty_factor : float, optional
        If greater than zero, the objective function also contains CPTP penalty
        terms which penalize non-CPTP-ness of the gates being optimized.  This factor
        multiplies these CPTP penalty terms.

    spam_penalty_factor : float, optional
        If greater than zero, the objective function also contains SPAM penalty
        terms which penalize non-positive-ness of the state preps being optimized.  This
        factor multiplies these SPAM penalty terms.

    gates_metric : {"frobenius", "fidelity", "tracedist"}, optional
        The metric used to compare gates within models. "frobenius" computes
        the normalized sqrt(sum-of-squared-differences), with weights
        multiplying the squared differences (see :func:`Model.frobeniusdist`).
        "fidelity" and "tracedist" sum the individual infidelities or trace
        distances of each gate, weighted by the weights.

    spam_metric : {"frobenius", "fidelity", "tracedist"}, optional
        The metric used to compare spam vectors within models. "frobenius"
        computes the normalized sqrt(sum-of-squared-differences), with weights
        multiplying the squared differences (see :func:`Model.frobeniusdist`).
        "fidelity" and "tracedist" sum the individual infidelities or trace
        distances of each "SPAM gate", weighted by the weights.

    gauge_group : GaugeGroup, optional
        The gauge group which defines which gauge trasformations are optimized
        over.  If None, then the `model`'s default gauge group is used.

    method : string, optional
        The method used to optimize the objective function.  Can be any method
        known by scipy.optimize.minimize such as 'BFGS', 'Nelder-Mead', 'CG', 'L-BFGS-B',
        or additionally:

        - 'auto' -- 'ls' when allowed, otherwise 'L-BFGS-B'
        - 'ls' -- custom least-squares optimizer.
        - 'custom' -- custom CG that often works better than 'CG'
        - 'supersimplex' -- repeated application of 'Nelder-Mead' to converge it
        - 'basinhopping' -- scipy.optimize.basinhopping using L-BFGS-B as a local optimizer
        - 'swarm' -- particle swarm global optimization algorithm
        - 'evolve' -- evolutionary global optimization algorithm using DEAP
        - 'brute' -- Experimental: scipy.optimize.brute using 4 points along each dimensions

    maxiter : int, optional
        Maximum number of iterations for the gauge optimization.

    maxfev : int, optional
        Maximum number of function evaluations for the gauge optimization.
        Defaults to maxiter.

    tol : float, optional
        The tolerance for the gauge optimization.

    oob_check_interval : int, optional
        If greater than zero, gauge transformations are allowed to fail (by raising
        any exception) to indicate an out-of-bounds condition that the gauge optimizer
        will avoid.  If zero, then any gauge-transform failures just terminate the
        optimization.

    convert_model_to : str, dict, list, optional
        For use when `model` is an `ExplicitOpModel`.  When not `None`, calls
        `model.convert_members_inplace(convert_model_to, set_default_gauge_group=False)` if
        `convert_model_to` is a string, `model.convert_members_inplace(**convert_model_to)` if it
        is a dict, and repeated calls to either of the above instances when `convert_model_to`
        is a list or tuple  prior to performing the gauge optimization.  This allows the gauge
        optimization to be performed using a differently constrained model.

    return_all : bool, optional
        When True, return best "goodness" value and gauge matrix in addition to the
        gauge optimized model.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    verbosity : int, optional
        How much detail to send to stdout.

    check_jac : bool
        When True, check least squares analytic jacobian against finite differences

    Returns
    -------
    model : if return_all == False
    
    (goodnessMin, gaugeMx, model) : if return_all == True
        Where goodnessMin is the minimum value of the goodness function (the best 'goodness') 
        found, gaugeMx is the gauge matrix used to transform the model, and model is the 
        final gauge-transformed model.
    """
    
    if item_weights is None: item_weights = {}

    ls_mode_allowed = bool(target_model is not None
                           and gates_metric.startswith("frobenius")
                           and spam_metric.startswith("frobenius"))
    #and model.dim < 64: # least squares optimization seems uneffective if more than 3 qubits
    #  -- observed by Lucas - should try to debug why 3 qubits seemed to cause trouble...

    if method == "ls" and not ls_mode_allowed:
        raise ValueError("Least-squares method is not allowed! Target"
                         " model must be non-None and frobenius metrics"
                         " must be used.")
    if method == "auto":
        method = 'ls' if ls_mode_allowed else 'L-BFGS-B'

    if convert_model_to is not None:
        conversion_args = convert_model_to if isinstance(convert_model_to, (list, tuple)) else (convert_model_to,)
        model = model.copy()  # don't alter the original model's parameterization (this would be unexpected)
        for args in conversion_args:
            if isinstance(args, str):
                model.convert_members_inplace(args, set_default_gauge_group=True)
            elif isinstance(args, dict):
                model.convert_members_inplace(**args)
            else:
                raise ValueError("Invalid `convert_model_to` arguments: %s" % str(args))

    objective_fn, jacobian_fn = _create_objective_fn(
        model, target_model, item_weights,
        cptp_penalty_factor, spam_penalty_factor,
        gates_metric, spam_metric, method, comm, check_jac, n_leak)

    result = gaugeopt_custom(model, objective_fn, gauge_group, method,
                             maxiter, maxfev, tol, oob_check_interval,
                             return_all, jacobian_fn, comm, verbosity)

    #If we've gauge optimized to a target model, declare that the
    # resulting model is now in the same basis as the target.
    if target_model is not None:
        newModel = result[-1] if return_all else result
        newModel.basis = target_model.basis.copy()

    return result


def gaugeopt_custom(model, objective_fn, gauge_group=None,
                    method='L-BFGS-B', maxiter=100000, maxfev=None, tol=1e-8,
                    oob_check_interval=0, return_all=False, jacobian_fn=None,
                    comm=None, verbosity=0):
    """
    Optimize the gauge of a model using a custom objective function.

    Parameters
    ----------
    model : Model
        The model to gauge-optimize

    objective_fn : function
        The function to be minimized.  The function must take a single `Model`
        argument and return a float.

    gauge_group : GaugeGroup, optional
        The gauge group which defines which gauge trasformations are optimized
        over.  If None, then the `model`'s default gauge group is used.

    method : string, optional
        The method used to optimize the objective function.  Can be any method
        known by scipy.optimize.minimize such as 'BFGS', 'Nelder-Mead', 'CG', 'L-BFGS-B',
        or additionally:

        - 'custom' -- custom CG that often works better than 'CG'
        - 'supersimplex' -- repeated application of 'Nelder-Mead' to converge it
        - 'basinhopping' -- scipy.optimize.basinhopping using L-BFGS-B as a local optimizer
        - 'swarm' -- particle swarm global optimization algorithm
        - 'evolve' -- evolutionary global optimization algorithm using DEAP
        - 'brute' -- Experimental: scipy.optimize.brute using 4 points along each dimensions

    maxiter : int, optional
        Maximum number of iterations for the gauge optimization.

    maxfev : int, optional
        Maximum number of function evaluations for the gauge optimization.
        Defaults to maxiter.

    tol : float, optional
        The tolerance for the gauge optimization.

    oob_check_interval : int, optional
        If greater than zero, gauge transformations are allowed to fail (by raising
        any exception) to indicate an out-of-bounds condition that the gauge optimizer
        will avoid.  If zero, then any gauge-transform failures just terminate the
        optimization.

    return_all : bool, optional
        When True, return best "goodness" value and gauge matrix in addition to the
        gauge optimized model.

    jacobian_fn : function, optional
        The jacobian of `objective_fn`.  The function must take three parameters,
        1) the un-transformed `Model`, 2) the transformed `Model`, and 3) the
        `GaugeGroupElement` representing the transformation that brings the first
        argument into the second.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    model                            
        if return_all == False
    
    (goodnessMin, gaugeMx, model)    
        if return_all == True
        where goodnessMin is the minimum value of the goodness function (the best 'goodness')
        found, gaugeMx is the gauge matrix used to transform the model, and model is the
        final gauge-transformed model.
    """

    printer = _baseobjs.VerbosityPrinter.create_printer(verbosity, comm)
    tStart = _time.time()

    if comm is not None:
        mdl_cmp = comm.bcast(model if (comm.Get_rank() == 0) else None, root=0)
        try:
            if model.frobeniusdist(mdl_cmp) > 1e-6:
                raise ValueError("MPI ERROR in gaugeopt: *different* models"
                                 " given to different processors!")  # pragma: no cover
        except NotImplementedError: pass  # OK if some gates (maps) don't implement this

    if gauge_group is None:
        gauge_group = model.default_gauge_group
        if gauge_group is None:
            #don't do any gauge optimization (assume trivial gauge group)
            _warnings.warn("No gauge group specified, so no gauge optimization performed.")
        if gauge_group is None or gauge_group.num_params == 0 or \
           model.num_params == 0:
            if return_all:
                trivialEl = _TrivialGaugeGroupElement(model.dim)
                return None, trivialEl, model.copy()
            else:
                return model.copy()

    x0 = gauge_group.initial_params  # gauge group picks a good initial el
    gaugeGroupEl = gauge_group.compute_element(x0)  # re-used element for evals

    def _call_objective_fn(gauge_group_el_vec, oob_check=False):
        # Note: oob_check can be True if oob_check_interval>=1 is given to the custom_leastsq below
        gaugeGroupEl.from_vector(gauge_group_el_vec)
        return objective_fn(gaugeGroupEl, oob_check)

    if jacobian_fn:
        def _call_jacobian_fn(gauge_group_el_vec):
            gaugeGroupEl.from_vector(gauge_group_el_vec)
            return jacobian_fn(gaugeGroupEl)
    else:
        _call_jacobian_fn = None

    printer.log("--- Gauge Optimization (%s method, %s) ---" % (method, str(type(gauge_group))), 2)
    if method == 'ls':
        assert(_call_jacobian_fn is not None), "Cannot use 'ls' method unless jacobian is available"
        ralloc = _baseobjs.ResourceAllocation(comm)  # FUTURE: plumb up a resource alloc object?
        test_f = _call_objective_fn(x0)
        solnX, converged, msg, _, _, _, _, _ = _opt.custom_leastsq(
            _call_objective_fn, _call_jacobian_fn, x0, f_norm2_tol=tol,
            jac_norm_tol=tol, rel_ftol=tol, rel_xtol=tol,
            max_iter=maxiter, resource_alloc=ralloc,
            arrays_interface=_opt.UndistributedArraysInterface(len(test_f), len(x0)),
            oob_check_interval=oob_check_interval,
            verbosity=printer.verbosity - 2)
        printer.log("Least squares message = %s" % msg, 2)
        assert(converged)
        solnF = _call_objective_fn(solnX) if return_all else None

    else:
        if comm is not None and comm.Get_rank() == 0:
            _warnings.warn("MPI comm was given for gauge optimization but can"
                           " only be used with the least-squares method.")

        bToStdout = (printer.verbosity >= 2 and printer.filename is None)
        if bToStdout and (comm is None or comm.Get_rank() == 0):
            print_obj_func = _opt.create_objfn_printer(_call_objective_fn)  # only ever prints to stdout!
            # print_obj_func(x0) #print initial point (can be a large vector though)
        else: print_obj_func = None

        minSol = _opt.minimize(_call_objective_fn, x0,
                               method=method, maxiter=maxiter, maxfev=maxfev,
                               tol=tol, jac=_call_jacobian_fn,
                               callback=print_obj_func)
        solnX = minSol.x
        solnF = minSol.fun

    gaugeGroupEl.from_vector(solnX)
    newModel = model.copy()
    newModel.transform_inplace(gaugeGroupEl)

    printer.log("Gauge optimization completed in %gs." % (_time.time() - tStart))

    if return_all:
        return solnF, gaugeGroupEl, newModel
    else:
        return newModel


def _create_objective_fn(model, target_model, item_weights=None,
                         cptp_penalty_factor=0, spam_penalty_factor=0,
                         gates_metric="frobenius", spam_metric="frobenius",
                         method=None, comm=None, check_jac=False, n_leak=0):
    """
    Creates the objective function and jacobian (if available)
    for gaugeopt_to_target
    """
    if item_weights is None: item_weights = {}
    opWeight = item_weights.get('gates', 1.0)
    spamWeight = item_weights.get('spam', 1.0)
    mxBasis = model.basis

    #Use the target model's basis if model's is unknown
    # (as it can often be if it's just come from an logl opt,
    #  since from_vector will clear any basis info)
    if mxBasis.name == "unknown" and target_model is not None:
        mxBasis = target_model.basis

    def _transform_with_oob_check(mdl, gauge_group_el, oob_check):
        """ Helper function that sometimes checks if mdl.transform_inplace(gauge_group_el) fails. """
        mdl = mdl.copy()
        if oob_check:
            try: mdl.transform_inplace(gauge_group_el)
            except Exception as e:
                raise ValueError("Out of bounds: %s" % str(e))  # signals OOB condition
        else:
            mdl.transform_inplace(gauge_group_el)
        return mdl

    if method == "ls":
        # least-squares case where objective function returns an array of
        # the before-they're-squared difference terms and there's an analytic jacobian

        assert(gates_metric.startswith("frobenius") and spam_metric.startswith("frobenius")), \
            "Only 'frobenius' and 'frobeniustt' metrics can be used when `method='ls'`!"
        assert(gates_metric == spam_metric)
        frobenius_transform_target = bool(gates_metric == 'frobeniustt')  # tt = "transform target"

        if frobenius_transform_target:
            full_target_model = target_model.copy()
            full_target_model.convert_members_inplace("full")  # so we can gauge-transform the target model.
        else:
            full_target_model = None  # in case it get's referenced by mistake

        def _objective_fn(gauge_group_el, oob_check):

            if frobenius_transform_target:
                transformed = _transform_with_oob_check(full_target_model, gauge_group_el.inverse(), oob_check)
                other = model
            else:
                transformed = _transform_with_oob_check(model, gauge_group_el, oob_check)
                other = target_model

            residuals, _ = transformed.residuals(other, None, item_weights)

            # We still the non-target model to be transformed and checked for these penalties
            if cptp_penalty_factor > 0 or spam_penalty_factor > 0:
                if frobenius_transform_target:
                    transformed = _transform_with_oob_check(model, gauge_group_el, oob_check)

                if cptp_penalty_factor > 0:
                    transformed.basis = mxBasis
                    cpPenaltyVec = _cptp_penalty(transformed, cptp_penalty_factor, transformed.basis)
                else: cpPenaltyVec = []  # so concatenate ignores

                if spam_penalty_factor > 0:
                    transformed.basis = mxBasis
                    spamPenaltyVec = _spam_penalty(transformed, spam_penalty_factor, transformed.basis)
                else: spamPenaltyVec = []  # so concatenate ignores

                return _np.concatenate((residuals, cpPenaltyVec, spamPenaltyVec))
            else:
                return residuals

        def _jacobian_fn(gauge_group_el):

            #Penalty terms below always act on the transformed non-target model.
            original_gauge_group_el = gauge_group_el

            if frobenius_transform_target:
                gauge_group_el = gauge_group_el.inverse()
                mdl_pre = full_target_model.copy()
                mdl_post = mdl_pre.copy()
            else:
                mdl_pre = model.copy()
                mdl_post = mdl_pre.copy()
            mdl_post.transform_inplace(gauge_group_el)

            # Indices: Jacobian output matrix has shape (L, N)
            start = 0
            d = mdl_pre.dim
            N = gauge_group_el.num_params
            L = mdl_pre.num_elements

            #Compute "extra" (i.e. beyond the model-element) rows of jacobian
            if cptp_penalty_factor != 0: L += _cptp_penalty_size(mdl_pre)
            if spam_penalty_factor != 0: L += _spam_penalty_size(mdl_pre)

            #Set basis for pentaly term calculation
            if cptp_penalty_factor != 0 or spam_penalty_factor != 0:
                mdl_pre.basis = mxBasis
                mdl_post.basis = mxBasis

            jacMx = _np.zeros((L, N))

            #Overview of terms:
            # objective: op_term = (S_inv * gate * S - target_op)
            # jac:       d(op_term) = (d (S_inv) * gate * S + S_inv * gate * dS )
            #            d(op_term) = (-(S_inv * dS * S_inv) * gate * S + S_inv * gate * dS )

            # objective: rho_term = (S_inv * rho - target_rho)
            # jac:       d(rho_term) = d (S_inv) * rho
            #            d(rho_term) = -(S_inv * dS * S_inv) * rho

            # objective: ET_term = (E.T * S - target_E.T)
            # jac:       d(ET_term) = E.T * dS

            #Overview of terms when frobenius_transform_target == True).  Note that the objective
            #expressions are identical to the above except for an additional overall minus sign and S <=> S_inv.

            # objective: op_term = (gate - S * target_op * S_inv)
            # jac:       d(op_term) = -(dS * target_op * S_inv + S * target_op * -(S_inv * dS * S_inv) )
            #            d(op_term) = (-dS * target_op * S_inv + S * target_op * (S_inv * dS * S_inv) )

            # objective: rho_term = (rho - S * target_rho)
            # jac:       d(rho_term) = - dS * target_rho

            # objective: ET_term = (E.T - target_E.T * S_inv)
            # jac:       d(ET_term) = - target_E.T * -(S_inv * dS * S_inv)
            #            d(ET_term) = target_E.T * (S_inv * dS * S_inv)

            #Distribute computation across processors
            allDerivColSlice = slice(0, N)
            derivSlices, myDerivColSlice, derivOwners, mySubComm = \
                _mpit.distribute_slice(allDerivColSlice, comm)
            if mySubComm is not None:
                _warnings.warn("Note: more CPUs(%d)" % comm.Get_size()
                               + " than gauge-opt derivative columns(%d)!" % N)  # pragma: no cover

            n = _slct.length(myDerivColSlice)
            wrtIndices = _slct.indices(myDerivColSlice) if (n < N) else None
            my_jacMx = jacMx[:, myDerivColSlice]  # just the columns I'm responsible for

            # S, and S_inv are shape (d,d)
            #S       = gauge_group_el.transform_matrix
            S_inv = gauge_group_el.transform_matrix_inverse
            dS = gauge_group_el.deriv_wrt_params(wrtIndices)  # shape (d*d),n
            dS.shape = (d, d, n)  # call it (d1,d2,n)
            dS = _np.rollaxis(dS, 2)  # shape (n, d1, d2)
            assert(dS.shape == (n, d, d))

            # --- NOTE: ordering here, with running `start` index MUST
            #           correspond to those in Model.residuals, which in turn
            #           must correspond to those in ForwardSimulator.residuals - which
            #           currently orders as: gates, simplified_ops, preps, effects.

            # -- LinearOperator terms
            # -------------------------
            for lbl, G in mdl_pre.operations.items():
                # d(op_term) = S_inv * (-dS * S_inv * G * S + G * dS) = S_inv * (-dS * G' + G * dS)
                #   Note: (S_inv * G * S) is G' (transformed G)
                wt = item_weights.get(lbl, opWeight)
                left = -1 * _np.dot(dS, mdl_post.operations[lbl].to_dense(on_space='minimal'))  # shape (n,d1,d2)
                right = _np.swapaxes(_np.dot(G.to_dense(on_space='minimal'), dS), 0, 1)  # shape (d1,n,d2) -> (n,d1,d2)
                result = _np.swapaxes(_np.dot(S_inv, left + right), 1, 2)  # shape (d1, d2, n)
                result = result.reshape((d**2, n))  # must copy b/c non-contiguous
                my_jacMx[start:start + d**2] = wt * result
                start += d**2

            # -- Instrument terms
            # -------------------------
            for ilbl, Inst in mdl_pre.instruments.items():
                wt = item_weights.get(ilbl, opWeight)
                for lbl, G in Inst.items():
                    # same calculation as for operation terms
                    left = -1 * _np.dot(dS, mdl_post.instruments[ilbl][lbl].to_dense(on_space='minimal'))  # (n,d1,d2)
                    right = _np.swapaxes(_np.dot(G.to_dense(on_space='minimal'), dS), 0, 1)  # (d1,n,d2) -> (n,d1,d2)
                    result = _np.swapaxes(_np.dot(S_inv, left + right), 1, 2)  # shape (d1, d2, n)
                    result = result.reshape((d**2, n))  # must copy b/c non-contiguous
                    my_jacMx[start:start + d**2] = wt * result
                    start += d**2

            # -- prep terms
            # -------------------------
            for lbl, rho in mdl_post.preps.items():
                # d(rho_term) = -(S_inv * dS * S_inv) * rho
                #   Note: (S_inv * rho) is transformed rho
                wt = item_weights.get(lbl, spamWeight)
                Sinv_dS = _np.dot(S_inv, dS)  # shape (d1,n,d2)
                result = -1 * _np.dot(Sinv_dS, rho.to_dense(on_space='minimal'))  # shape (d,n)
                my_jacMx[start:start + d] = wt * result
                start += d

            # -- effect terms
            # -------------------------
            for povmlbl, povm in mdl_pre.povms.items():
                for lbl, E in povm.items():
                    # d(ET_term) = E.T * dS
                    wt = item_weights.get(povmlbl + "_" + lbl, spamWeight)
                    result = _np.dot(E.to_dense(on_space='minimal')[None, :], dS).T  # shape (1,n,d2).T => (d2,n,1)
                    my_jacMx[start:start + d] = wt * result.squeeze(2)  # (d2,n)
                    start += d

            # -- penalty terms  -- Note: still use original gauge transform applied to `model`
            # -------------------------
            if cptp_penalty_factor > 0 or spam_penalty_factor > 0:
                if frobenius_transform_target:  # reset back to non-target-tranform "mode"
                    gauge_group_el = original_gauge_group_el
                    mdl_pre = model.copy()
                    mdl_post = mdl_pre.copy()
                    mdl_post.transform_inplace(gauge_group_el)

                if cptp_penalty_factor > 0:
                    start += _cptp_penalty_jac_fill(my_jacMx[start:], mdl_pre, mdl_post,
                                                    gauge_group_el, cptp_penalty_factor,
                                                    mdl_pre.basis, wrtIndices)

                if spam_penalty_factor > 0:
                    start += _spam_penalty_jac_fill(my_jacMx[start:], mdl_pre, mdl_post,
                                                    gauge_group_el, spam_penalty_factor,
                                                    mdl_pre.basis, wrtIndices)

            #At this point, each proc has filled the portions (columns) of jacMx that
            # it's responsible for, and so now we gather them together.
            _mpit.gather_slices(derivSlices, derivOwners, jacMx, [], 1, comm)
            #Note jacMx is completely filled (on all procs)

            if check_jac and (comm is None or comm.Get_rank() == 0):
                def _mock_objective_fn(v):
                    return _objective_fn(gauge_group_el, False)
                vec = gauge_group_el.to_vector()
                _opt.check_jac(_mock_objective_fn, vec, jacMx, tol=1e-5, eps=1e-9, err_type='abs',
                               verbosity=1)

            return jacMx

    else:
        # non-least-squares case where objective function returns a single float
        # and (currently) there's no analytic jacobian
        dim = int(_np.sqrt(mxBasis.dim))
        if n_leak > 0:
            B = _tools.leading_dxd_submatrix_basis_vectors(dim - n_leak, dim, mxBasis)
            """
            ^ Need to do something else. 
            
              In a leakage-friendly basis the operation matrices can be partitioned as 
                  [comp     , semileak1]
                  [semileak2, fulleak  ].
              Instead of projecting only on to comp, we want to keep everything
              except fullleak. ... But I don't think we can implement that just by
              pre-multiplying by a projector.
            
              I think this distinction might not be significant under certain idealized
              assumptions, but small deviations from those conditions (which are certain
              to happen in practice) might make it matter.
            """
            P = B @ B.T.conj()
            if _np.linalg.norm(P.imag) > 1e-12:
                raise ValueError()
            else:
                P = P.real
            transform_mx_arg = (P, None)
            # ^ The semantics of this tuple are defined by the frobeniusdist function
            #   in the ExplicitOpModelCalc class. There are intended semantics for 
            #   the second element of the tuple, but those aren't implemented yet so
            #   for now I'm setting the second entry to None. -- Riley
        else:
            transform_mx_arg = (_np.eye(mxBasis.dim), None)

        assert gates_metric != "frobeniustt"
        assert spam_metric  != "frobeniustt"
        assert spam_metric == gates_metric
        metric = spam_metric
        # assert spam_metric == gates_metric
        # ^ Erik and Corey said these are rarely used. I've removed support for
        # them in this codepath (non-LS optimizer) in order to make it easier to
        # read my updated code for leakage-aware metrics. It wouldn't be hard to
        # add support back, but I just want to keep things simple. -- Riley

        def _objective_fn(gauge_group_el, oob_check):
            mdl = _transform_with_oob_check(model, gauge_group_el, oob_check)
            ret = 0

            if cptp_penalty_factor > 0:
                mdl.basis = mxBasis  # set basis for jamiolkowski iso
                cpPenaltyVec = _cptp_penalty(mdl, cptp_penalty_factor, mdl.basis)
                ret += _np.sum(cpPenaltyVec)

            if spam_penalty_factor > 0:
                mdl.basis = mxBasis
                spamPenaltyVec = _spam_penalty(mdl, spam_penalty_factor, mdl.basis)
                ret += _np.sum(spamPenaltyVec)

            assert target_model is not None
            """
            Leakage-aware metric supported, per implementation in mdl.frobeniusdist.
            Refer to how mdl.frobeniusdist handles the case when transform_mx_arg
            is a tuple in order to understand how the leakage-aware metric is defined.
            
                Idea: raise an error if mxBasis isn't leakage-friendly. 
                    PROBLEM: if we're deep in the code then we end up needing to be
                            working in a basis that has the identity matrix as its
                            first element. The leakage-friendly basis doesn't even
                            have the identity matrix as an element, let alone the first element
            
                TODO: dig into function calls below and see where we can
                    access the mxBasis object. (I'm sure we can.)
                    Looks like it isn't accessible within ExplicitOpModelCalc objects.
                    It's most definitely available in ExplicitOpModel.       
            """
            if "frobenius" in metric:
                val = mdl.frobeniusdist(target_model, transform_mx_arg, item_weights)
                if "squared" in metric:
                    val = val ** 2
                ret += val
                return ret

            elif metric == "fidelity":
                # Leakage-aware metric supported for gates, using leaky_entanglement_fidelity.
                for opLbl in mdl.operations:
                    wt = item_weights.get(opLbl, opWeight)
                    top = target_model.operations[opLbl].to_dense()
                    mop = mdl.operations[opLbl].to_dense()
                    ret += wt * (1.0 - _tools.leaky_entanglement_fidelity(top, mop, mxBasis, n_leak))**2
                # Leakage-aware metrics NOT available for SPAM
                for preplabel, m_prep in mdl.preps.items():
                    wt = item_weights.get(preplabel, spamWeight)
                    rhoMx1 = _tools.vec_to_stdmx(m_prep.to_dense(), mxBasis)
                    t_prep = target_model.preps[preplabel]
                    rhoMx2 = _tools.vec_to_stdmx(t_prep.to_dense(), mxBasis)
                    ret += wt * (1.0 - _tools.fidelity(rhoMx1, rhoMx2))**2
                for povmlabel in mdl.povms.keys():
                    wt = item_weights.get(povmlabel, spamWeight)
                    fidelity = _tools.povm_fidelity(mdl, target_model, povmlabel)
                    ret += wt * (1.0 - fidelity)**2
                return ret

            elif metric == "tracedist":
                # Leakage-aware metric supported, using leaky_jtracedist.
                for opLbl in mdl.operations:
                    wt = item_weights.get(opLbl, opWeight)
                    top = target_model.operations[opLbl].to_dense()
                    mop = mdl.operations[opLbl].to_dense()
                    ret += wt * _tools.leaky_jtracedist(top, mop, mxBasis, n_leak)
                # Leakage-aware metrics NOT available for SPAM
                for preplabel, m_prep in mdl.preps.items():
                    wt = item_weights.get(preplabel, spamWeight)
                    rhoMx1 = _tools.vec_to_stdmx(m_prep.to_dense(), mxBasis)
                    t_prep = target_model.preps[preplabel]
                    rhoMx2 = _tools.vec_to_stdmx(t_prep.to_dense(), mxBasis)
                    ret += wt * _tools.tracedist(rhoMx1, rhoMx2)
                for povmlabel in mdl.povms.keys():
                    wt = item_weights.get(povmlabel, spamWeight)
                    ret += wt * _tools.povm_jtracedist(mdl, target_model, povmlabel)
                return ret

            else:
                raise ValueError("Invalid metric: %s" % metric)
            # end _objective_fn
        
        _jacobian_fn = None

    return _objective_fn, _jacobian_fn


def _cptp_penalty_size(mdl):
    """
    Helper function - *same* as that in core.py.
    """
    from pygsti.objectivefns.objectivefns import _cptp_penalty_size as _core_cptp_penalty_size
    return _core_cptp_penalty_size(mdl)


def _spam_penalty_size(mdl):
    """
    Helper function - *same* as that in core.py.
    """
    from pygsti.objectivefns.objectivefns import _spam_penalty_size as _core_spam_penalty_size
    return _core_spam_penalty_size(mdl)


def _cptp_penalty(mdl, prefactor, op_basis):
    """
    Helper function - CPTP penalty: (sum of tracenorms of gates),
    which in least squares optimization means returning an array
    of the sqrt(tracenorm) of each gate.  This function is the
    *same* as that in core.py.

    Returns
    -------
    numpy array
        a (real) 1D array of length len(mdl.operations).
    """
    from pygsti.objectivefns.objectivefns import _cptp_penalty as _core_cptp_penalty
    return _core_cptp_penalty(mdl, prefactor, op_basis)


def _spam_penalty(mdl, prefactor, op_basis):
    """
    Helper function - CPTP penalty: (sum of tracenorms of gates),
    which in least squares optimization means returning an array
    of the sqrt(tracenorm) of each gate.  This function is the
    *same* as that in core.py.

    Returns
    -------
    numpy array
        a (real) 1D array of length _spam_penalty_size(mdl)
    """
    from pygsti.objectivefns.objectivefns import _spam_penalty as _core_spam_penalty
    return _core_spam_penalty(mdl, prefactor, op_basis)


def _cptp_penalty_jac_fill(cp_penalty_vec_grad_to_fill, mdl_pre, mdl_post,
                           gauge_group_el, prefactor, op_basis, wrt_filter):
    """
    Helper function - jacobian of CPTP penalty (sum of tracenorms of gates)
    Returns a (real) array of shape (len(mdl.operations), gauge_group_el.num_params).
    """

    # d( sqrt(|chi|_Tr) ) = (0.5 / sqrt(|chi|_Tr)) * d( |chi|_Tr )
    # but here, unlike in core.py, chi = J(S_inv * G * S) == J(G')
    # and we're differentiating wrt the parameters of S, the
    # gauge_group_el.

    # S, and S_inv are shape (d,d)
    d, N = mdl_pre.dim, gauge_group_el.num_params
    n = N if (wrt_filter is None) else len(wrt_filter)
    #S       = gauge_group_el.transform_matrix
    S_inv = gauge_group_el.transform_matrix_inverse
    dS = gauge_group_el.deriv_wrt_params(wrt_filter)  # shape (d*d),n
    dS.shape = (d, d, n)  # call it (d1,d2,n)
    dS = _np.rollaxis(dS, 2)  # shape (n, d1, d2)

    for i, (gl, gate) in enumerate(mdl_post.operations.items()):
        pre_op = mdl_pre.operations[gl]

        #get sgn(chi-matrix) == d(|chi|_Tr)/dchi in std basis
        # so sgnchi == d(|chi_std|_Tr)/dchi_std
        chi = _tools.fast_jamiolkowski_iso_std(gate, op_basis)
        assert(_np.linalg.norm(chi - chi.T.conjugate()) < 1e-4), \
            "chi should be Hermitian!"

        sgnchi = _tools.matrix_sign(chi)
        assert(_np.linalg.norm(sgnchi - sgnchi.T.conjugate()) < 1e-4), \
            "sgnchi should be Hermitian!"

        # Let M be the "shuffle" operation performed by fast_jamiolkowski_iso_std
        # which maps a gate onto the choi-jamiolkowsky "basis" (i.e. performs that C-J
        # transform).  This shuffle op commutes with the derivative, so that
        # dchi_std/dp := d(M(G'))/dp = M(d(S_inv*G*S)/dp) = M( d(S_inv)*G*S + S_inv*G*dS )
        #              = M( (-S_inv*dS*S_inv)*G*S + S_inv*G*dS ) = M( S_inv*(-dS*S_inv*G*S) + G*dS )
        left = -1 * _np.dot(dS, gate)  # shape (n,d1,d2)
        right = _np.swapaxes(_np.dot(pre_op, dS), 0, 1)  # shape (d1, n, d2) -> (n,d1,d2)
        result = _np.swapaxes(_np.dot(S_inv, left + right), 0, 1)  # shape (n, d1, d2)

        dchi_std = _np.empty((n, d, d), 'complex')
        for p in range(n):  # p indexes param
            dchi_std[p] = _tools.fast_jamiolkowski_iso_std(result[p], op_basis)  # "M(results)"
            assert(_np.linalg.norm(dchi_std[p] - dchi_std[p].T.conjugate()) < 1e-8)  # check hermitian

        dchi_std = _np.conjugate(dchi_std)  # so element-wise multiply
        # of complex number (einsum below) results in separately adding
        # Re and Im parts (also see NOTE in spam_penalty_jac_fill below)

        #contract to get (note contract along both mx indices b/c treat like a
        # mx basis): d(|chi_std|_Tr)/dp = d(|chi_std|_Tr)/dchi_std * dchi_std/dp
        #v =  _np.einsum("ij,aij->a",sgnchi,dchi_std)
        v = _np.tensordot(sgnchi, dchi_std, ((0, 1), (1, 2)))
        v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(chi)))  # add 0.5/|chi|_Tr factor
        assert(_np.linalg.norm(v.imag) < 1e-4)
        cp_penalty_vec_grad_to_fill[i, :] = v.real
        chi = sgnchi = dchi_std = v = None  # free mem

    return len(mdl_pre.operations)  # the number of leading-dim indicies we filled in


def _spam_penalty_jac_fill(spam_penalty_vec_grad_to_fill, mdl_pre, mdl_post,
                           gauge_group_el, prefactor, op_basis, wrt_filter):
    """
    Helper function - jacobian of CPTP penalty (sum of tracenorms of gates)
    Returns a (real) array of shape (_spam_penalty_size(mdl), gauge_group_el.num_params).
    """
    BMxs = op_basis.elements  # shape [mdl.dim, dmDim, dmDim]
    ddenMxdV = dEMxdV = BMxs.conjugate()  # b/c denMx = sum( spamvec[i] * Bmx[i] ) and "V" == spamvec
    #NOTE: conjugate() above is because ddenMxdV and dEMxdV will get *elementwise*
    # multiplied (einsum below) by another complex matrix (sgndm or sgnE) and summed
    # in order to gather the different components of the total derivative of the trace-norm
    # wrt some spam-vector change dV.  If left un-conjugated, we'd get A*B + A.C*B.C (just
    # taking the (i,j) and (j,i) elements of the sum, say) which would give us
    # 2*Re(A*B) = A.r*B.r - B.i*A.i when we *want* (b/c Re and Im parts are thought of as
    # separate, independent degrees of freedom) A.r*B.r + A.i*B.i = 2*Re(A*B.C) -- so
    # we need to conjugate the "B" matrix, which is ddenMxdV or dEMxdV below.

    assert(ddenMxdV.size > 0), "Could not obtain basis matrices from " \
        + "'%s' basis for spam pentalty factor!" % op_basis.name

    # S, and S_inv are shape (d,d)
    d, N = mdl_pre.dim, gauge_group_el.num_params
    n = N if (wrt_filter is None) else len(wrt_filter)
    S_inv = gauge_group_el.transform_matrix_inverse
    dS = gauge_group_el.deriv_wrt_params(wrt_filter)  # shape (d*d),n
    dS.shape = (d, d, n)  # call it (d1,d2,n)
    dS = _np.rollaxis(dS, 2)  # shape (n, d1, d2)

    # d( sqrt(|denMx|_Tr) ) = (0.5 / sqrt(|denMx|_Tr)) * d( |denMx|_Tr )
    # but here, unlike in core.py, denMx = StdMx(S_inv * rho) = StdMx(rho')
    # and we're differentiating wrt the parameters of S, the
    # gauge_group_el.

    for i, (lbl, prepvec) in enumerate(mdl_post.preps.items()):

        #get sgn(denMx) == d(|denMx|_Tr)/d(denMx) in std basis
        # dmDim = denMx.shape[0]
        denMx = _tools.vec_to_stdmx(prepvec.to_dense(on_space='minimal')[:, None], op_basis)
        assert(_np.linalg.norm(denMx - denMx.T.conjugate()) < 1e-4), \
            "denMx should be Hermitian!"

        sgndm = _tools.matrix_sign(denMx)
        if _np.linalg.norm(sgndm - sgndm.T.conjugate()) >= 1e-4:
            _warnings.warn("Matrix sign mapped Hermitian->Non-hermitian; correcting...")  # pragma: no cover
            sgndm = (sgndm + sgndm.T.conjugate()) / 2.0                                    # pragma: no cover
        assert(_np.linalg.norm(sgndm - sgndm.T.conjugate()) < 1e-4), \
            "sgndm should be Hermitian!"

        # get d(prepvec')/dp = d(S_inv * prepvec)/dp in op_basis [shape == (n,dim)]
        #                    = (-S_inv*dS*S_inv) * prepvec = -S_inv*dS * prepvec'
        Sinv_dS = _np.dot(S_inv, dS)  # shape (d1,n,d2)
        dVdp = -1 * _np.dot(Sinv_dS, prepvec.to_dense(on_space='minimal')[:, None]).squeeze(2)  # shape (d,n,1) => (d,n)
        assert(dVdp.shape == (d, n))

        # denMx = sum( spamvec[i] * Bmx[i] )

        #contract to get (note contract along both mx indices b/c treat like a mx basis):
        # d(|denMx|_Tr)/dp = d(|denMx|_Tr)/d(denMx) * d(denMx)/d(spamvec) * d(spamvec)/dp
        # [dmDim,dmDim] * [mdl.dim, dmDim,dmDim] * [mdl.dim, n]
        #v =  _np.einsum("ij,aij,ab->b",sgndm,ddenMxdV,dVdp)
        v = _np.tensordot(_np.tensordot(sgndm, ddenMxdV, ((0, 1), (1, 2))), dVdp, (0, 0))
        v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(denMx)))  # add 0.5/|denMx|_Tr factor
        assert(_np.linalg.norm(v.imag) < 1e-4)
        spam_penalty_vec_grad_to_fill[i, :] = v.real
        denMx = sgndm = dVdp = v = None  # free mem

    # d( sqrt(|EMx|_Tr) ) = (0.5 / sqrt(|EMx|_Tr)) * d( |EMx|_Tr )
    # but here, unlike in core.py, EMx = StdMx(S.T * E) = StdMx(E')
    # and we're differentiating wrt the parameters of S, the
    # gauge_group_el.

    i = len(mdl_post.preps)
    for povmlbl, povm in mdl_post.povms.items():
        for lbl, effectvec in povm.items():

            #get sgn(EMx) == d(|EMx|_Tr)/d(EMx) in std basis
            EMx = _tools.vec_to_stdmx(effectvec.to_dense(on_space='minimal')[:, None], op_basis)
            # dmDim = EMx.shape[0]
            assert(_np.linalg.norm(EMx - EMx.T.conjugate()) < 1e-4), \
                "denMx should be Hermitian!"

            sgnE = _tools.matrix_sign(EMx)
            if(_np.linalg.norm(sgnE - sgnE.T.conjugate()) >= 1e-4):
                _warnings.warn("Matrix sign mapped Hermitian->Non-hermitian; correcting...")  # pragma: no cover
                sgnE = (sgnE + sgnE.T.conjugate()) / 2.0                                       # pragma: no cover
            assert(_np.linalg.norm(sgnE - sgnE.T.conjugate()) < 1e-4), \
                "sgnE should be Hermitian!"

            # get d(effectvec')/dp = [d(effectvec.T * S)/dp].T in op_basis [shape == (n,dim)]
            #                      = [effectvec.T * dS].T
            #  OR = dS.T * effectvec
            pre_effectvec = mdl_pre.povms[povmlbl][lbl].to_dense(on_space='minimal')[:, None]
            dVdp = _np.dot(pre_effectvec.T, dS).squeeze(0).T
            # shape = (1,d) * (n, d1,d2) = (1,n,d2) => (n,d2) => (d2,n)
            assert(dVdp.shape == (d, n))

            # EMx = sum( spamvec[i] * Bmx[i] )

            #contract to get (note contract along both mx indices b/c treat like a mx basis):
            # d(|EMx|_Tr)/dp = d(|EMx|_Tr)/d(EMx) * d(EMx)/d(spamvec) * d(spamvec)/dp
            # [dmDim,dmDim] * [mdl.dim, dmDim,dmDim] * [mdl.dim, n]
            #v =  _np.einsum("ij,aij,ab->b",sgnE,dEMxdV,dVdp)
            v = _np.tensordot(_np.tensordot(sgnE, dEMxdV, ((0, 1), (1, 2))), dVdp, (0, 0))
            v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(EMx)))  # add 0.5/|EMx|_Tr factor
            assert(_np.linalg.norm(v.imag) < 1e-4)
            spam_penalty_vec_grad_to_fill[i, :] = v.real

            denMx = sgndm = dVdp = v = None  # free mem
            i += 1

    #return the number of leading-dim indicies we filled in
    return len(mdl_post.preps) + sum([len(povm) for povm in mdl_post.povms.values()])
