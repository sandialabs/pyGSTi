"""
GST gauge optimization algorithms
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
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
from pygsti.models import (
    ExplicitOpModel as _ExplicitOpModel
)
from pygsti.models.model import OpModel as _OpModel
from pygsti.models.gaugegroup import (
    TrivialGaugeGroupElement as _TrivialGaugeGroupElement,
    GaugeGroupElement as _GaugeGroupElement
)

from typing import Callable, Union, Optional, Any


class GaugeoptToTargetArgs:
    """
    This class is basically a namespace. It was introduced to strip out tons of complexity
    in gaugeopt_to_target(...) without breaking old code that might call gaugeopt_to_target.
    """

    old_trailing_positional_args = (
        'item_weights', 'cptp_penalty_factor', 'spam_penalty_factor',
        'gates_metric', 'spam_metric', 'gauge_group', 'method',
        'maxiter', 'maxfev', 'tol', 'oob_check_interval', 
        'convert_model_to', 'return_all', 'comm', 'verbosity',
        'check_jac'
    )

    @staticmethod
    def parsed_model(model: Union[_OpModel, _ExplicitOpModel], convert_model_to: Optional[Any]) -> _OpModel:
        if convert_model_to is None:
            return model
        
        if not isinstance(model, _ExplicitOpModel):
            raise ValueError('Gauge optimization only supports model conversion for ExplicitOpModels.')
        conversion_args = convert_model_to if isinstance(convert_model_to, (list, tuple)) else (convert_model_to,)
        model_out = model.copy()  # don't alter the original model's parameterization (this would be unexpected)
        for args in conversion_args:
            if isinstance(args, str):
                model_out.convert_members_inplace(args, set_default_gauge_group=True)
            elif isinstance(args, dict):
                model_out.convert_members_inplace(**args)
            else:
                raise ValueError("Invalid `convert_model_to` arguments: %s" % str(args))
        return model_out

    @staticmethod
    def parsed_method(target_model: Optional[_OpModel], method: str, gates_metric: str, spam_metric: str, n_leak: int) -> str:
        ls_mode_allowed = bool(target_model is not None
                            and gates_metric.startswith("frobenius")
                            and spam_metric.startswith("frobenius")
                            and n_leak == 0)
        # and model.dim < 64: # least squares optimization seems uneffective if more than 3 qubits
        #   -- observed by Lucas - should try to debug why 3 qubits seemed to cause trouble...
        if method == "ls" and not ls_mode_allowed:
            raise ValueError("Least-squares method is not allowed! Target"
                            " model must be non-None and frobenius metrics"
                            " must be used.")
        elif method == "auto":
            return 'ls' if ls_mode_allowed else 'L-BFGS-B'
        else:
            return method

    # The static dicts of default values are substituted in gaugeopt_to_target's kwargs.
    # This is a safe thing to do because no invocation of "gaugeopt_to_target" within pyGSTi
    # used positional arguments past target_model.

    create_objective_passthrough_kwargs : dict[str,Any] = dict(
        item_weights=None,     # gets cast to a dict.
        cptp_penalty_factor=0,
        spam_penalty_factor=0,
        check_jac=False,
    )
    """
    item_weights : dict
        Dictionary of weighting factors for gates and spam operators.  Keys can
        be gate, state preparation, or POVM effect, as well as the special values
        "spam" or "gates" which apply the given weighting to *all* spam operators
        or gates respectively.  Values are floating point numbers.  Values given
        for specific gates or spam operators take precedence over "gates" and
        "spam" values.  The precise use of these weights depends on the model
        metric(s) being used.

    cptp_penalty_factor : float
        If greater than zero, the objective function also contains CPTP penalty
        terms which penalize non-CPTP-ness of the gates being optimized.  This factor
        multiplies these CPTP penalty terms.

    spam_penalty_factor : float
        If greater than zero, the objective function also contains SPAM penalty
        terms which penalize non-positive-ness of the state preps being optimized.  This
        factor multiplies these SPAM penalty terms.

    check_jac : bool
        When True, check least squares analytic jacobian against finite differences.
    """
     
    gaugeopt_custom_passthrough_kwargs : dict[str, Any] = dict(
        maxiter=100000,
        maxfev=None,
        tol=1e-8,
        oob_check_interval=0,
        verbosity=0,
    )
    """
    maxiter : int
        Maximum number of iterations for the gauge optimization.

    maxfev : int
        Maximum number of function evaluations for the gauge optimization.
        Defaults to maxiter.

    tol : float
        The tolerance for the gauge optimization.

    oob_check_interval : int
        If greater than zero, gauge transformations are allowed to fail (by raising
        any exception) to indicate an out-of-bounds condition that the gauge optimizer
        will avoid.  If zero, then any gauge-transform failures just terminate the
        optimization.
    """

    other_kwargs : dict[str, Any] = dict(
        gates_metric="frobenius",    # defines ls_mode_allowed, then passed to _create_objective_fn,
        spam_metric="frobenius",     # defines ls_mode_allowed, then passed to _create_objective_fn,
        gauge_group=None,            # used iff convert_model_to is not None, then passed to _create_objective_fn and gaugeopt_custom
        method='auto',               # validated, then passed to _create_objective_fn and gaugeopt_custom
        return_all=False,            # used in the function body (only for branching after passing to gaugeopt_custom)
        convert_model_to=None,       # passthrough to parsed_model.
        comm=None,                   # passthrough to _create_objective_fn and gaugeopt_custom
        n_leak=0                     # passthrough to parsed_objective and _create_objective_fn
    )
    """
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

    n_leak : int
       Used in leakage modeling. If positive, this specifies how modify defintiions
       of gate and SPAM metrics to reflect the fact that target gates do not have
       well-defined actions outside the computational subspace.
    """

    @staticmethod
    def create_full_kwargs(args: tuple[Any,...], kwargs: dict[str, Any]):
        full_kwargs =      GaugeoptToTargetArgs.create_objective_passthrough_kwargs.copy()
        full_kwargs.update(GaugeoptToTargetArgs.gaugeopt_custom_passthrough_kwargs)
        full_kwargs.update(GaugeoptToTargetArgs.other_kwargs)
        full_kwargs.update(kwargs)

        if extra_posargs := len(args) > 0:
            msg = \
            f"""
            Recieved {extra_posargs} positional arguments past `model` and `target_model`.
            These should be passed as appropriate keyword arguments instead. This version
            of pyGSTi will infer intended keyword arguments based on the legacy argument 
            positions. Future versions of pyGSTi will raise an error.
            """
            _warnings.warn(msg)
            for k,v in zip(GaugeoptToTargetArgs.old_trailing_positional_args, args):
                full_kwargs[k] = v

        if full_kwargs['item_weights'] is None:
            full_kwargs['item_weights'] = dict()
                
        return full_kwargs


def gaugeopt_to_target(model, target_model, *args, **kwargs):
    """
    Legacy function to optimize the gauge degrees of freedom of a model to that of a target.

    Use of more than two positional arguments is deprecated; keyword arguments should be used
    instead. See the GaugeoptToTargetArgs class for available keyword arguments and their
    default values.

    Returns
    -------
    model : if kwargs.get('return_all', False) == False
    
    (goodnessMin, gaugeMx, model) : if kwargs.get('return_all', False) == True
    
        Where goodnessMin is the minimum value of the goodness function (the best 'goodness') 
        found, gaugeMx is the gauge matrix used to transform the model, and model is the 
        final gauge-transformed model.
    """

    full_kwargs = GaugeoptToTargetArgs.create_full_kwargs(args, kwargs)
    """
    This function handles a strange situation where `target_model` can be None.

    In this case, the objective function will only depend on `cptp_penality_factor`
    and `spam_penalty_factor`; it's forbidden to use method == 'ls'; and the
    returned OpModel might not have its basis set.
    """

    # arg parsing: validating `method`
    gates_metric = full_kwargs['gates_metric']
    spam_metric  = full_kwargs['spam_metric']
    n_leak       = full_kwargs['n_leak']
    method       = full_kwargs['method']
    method = GaugeoptToTargetArgs.parsed_method(
        target_model, method, gates_metric, spam_metric, n_leak
    )

    # arg parsing: (possibly) converting `model`
    convert_model_to = full_kwargs['convert_model_to']
    model  = GaugeoptToTargetArgs.parsed_model(model, convert_model_to)

    # actual work: constructing objective_fn and jacobian_fn
    item_weights = full_kwargs['item_weights']
    cptp_penalty = full_kwargs['cptp_penalty_factor']
    spam_penalty = full_kwargs['spam_penalty_factor']
    comm         = full_kwargs['comm']
    check_jac    = full_kwargs['check_jac']
    objective_fn, jacobian_fn = _create_objective_fn(
        model, target_model, item_weights, cptp_penalty, spam_penalty,
        gates_metric, spam_metric, method, comm, check_jac, n_leak
    )

    # actual work: calling the (wrapper of the wrapper of the ...) optimizer
    gauge_group = full_kwargs['gauge_group']
    maxiter     = full_kwargs['maxiter']
    maxfev      = full_kwargs['maxfev']
    tol         = full_kwargs['tol']
    oob_check   = full_kwargs['oob_check_interval']
    return_all  = full_kwargs['return_all']
    verbosity   = full_kwargs['verbosity']
    result = gaugeopt_custom(
        model, objective_fn, gauge_group, method, maxiter, maxfev,
        tol, oob_check, return_all, jacobian_fn, comm, verbosity
    )

    # If we've gauge optimized to a target model, declare that the
    # resulting model is now in the same basis as the target.
    if target_model is not None:
        newModel = result[-1] if return_all else result
        newModel.basis = target_model.basis.copy()

    return result


GGElObjective = Callable[[_GaugeGroupElement, bool], Union[float, _np.ndarray]]

GGElJacobian  = Union[None, Callable[[_GaugeGroupElement], _np.ndarray]]


def gaugeopt_custom(model, objective_fn: GGElObjective, gauge_group=None,
                    method='L-BFGS-B', maxiter=100000, maxfev=None, tol=1e-8,
                    oob_check_interval=0, return_all=False, jacobian_fn: Optional[GGElJacobian]=None,
                    comm=None, verbosity=0):
    """
    Optimize the gauge of a model using a custom objective function.

    Parameters
    ----------
    model : Model
        The model to gauge-optimize

    objective_fn : function
        The function to be minimized. The function must take a GaugeGroupElement
        and a bool. If method == 'ls' then objective_fn must return an ndarray; if
        method != 'ls' then objective_fn must return a float.

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

    #replace model with a new copy of itself so as to not propagate the conversion back to the 
    #instance of the model object we are gauge optimizing.
    model = model.copy()

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
        # Note: oob_check can be True if oob_check_interval>=1 is given to the simplish_leastsq below
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
        solnX, converged, msg, _, _, _, _ = _opt.simplish_leastsq(
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


def _transform_with_oob_check(mdl, gauge_group_el, oob_check):
    """ Helper function that sometimes checks if mdl.transform_inplace(gauge_group_el) fails. """
    mdl = mdl.copy()
    if oob_check:
        try:
            mdl.transform_inplace(gauge_group_el)
        except Exception as e:
            raise ValueError("Out of bounds: %s" % str(e))  # signals OOB condition
    else:
        mdl.transform_inplace(gauge_group_el)
    return mdl


def _gate_fidelity_targets(model, target_model):
    from pygsti.report.reportables import eigenvalue_entanglement_infidelity
    gate_fidelity_targets : dict[ _baseobjs.Label, tuple[_np.ndarray, Union[float, _np.floating]] ] = dict()
    for lbl in target_model.operations:
        G_target = target_model.operations[lbl].to_dense()
        G_curest = model.operations[lbl].to_dense()
        t = 1 - eigenvalue_entanglement_infidelity(G_curest, G_target, model.basis)
        t = _np.clip(t, a_min=0.0, a_max=1.0)
        gate_fidelity_targets[lbl] = (G_target, t)
    return gate_fidelity_targets


def _prep_fidelity_targets(model, target_model):
    prep_fidelity_targets : dict[ _baseobjs.Label, tuple[_np.ndarray, Union[float, _np.floating]] ] = dict()
    for preplbl in target_model.preps:
        rho_target = _tools.vec_to_stdmx( target_model.preps[preplbl].to_dense(), model.basis )
        rho_curest = _tools.vec_to_stdmx(        model.preps[preplbl].to_dense(), model.basis )
        t = _tools.eigenvalue_fidelity(rho_curest, rho_target)
        t = _np.clip(t, a_min=0.0, a_max=1.0)
        prep_fidelity_targets[preplbl] = (rho_target, t)
    return prep_fidelity_targets


def _povm_effect_fidelity_targets(model, target_model):
    povm_fidelity_targets : dict[ _baseobjs.Label, tuple[dict, dict] ] = dict()
    for povmlbl in target_model.povms:
        M_target  = target_model.povms[povmlbl]
        M_curest  =        model.povms[povmlbl]
        Es_target = {elbl : _tools.vec_to_stdmx(e_target.to_dense(), model.basis) for (elbl, e_target) in M_target.items() }
        Es_curest = {elbl : _tools.vec_to_stdmx(e_target.to_dense(), model.basis) for (elbl, e_target) in M_curest.items() }
        ts = {elbl : _tools.eigenvalue_fidelity(Es_target[elbl], Es_curest[elbl])   for elbl in M_target }
        ts = {elbl : _np.clip(t, a_min=0.0, a_max=1.0) for (elbl, t) in ts.items()}
        povm_fidelity_targets[povmlbl] = (Es_target, ts)
    return povm_fidelity_targets


def _legacy_create_scalar_objective(model, target_model,
        item_weights: dict[str,float], cptp_penalty_factor: float, spam_penalty_factor: float,
        gates_metric: str, spam_metric: str, n_leak: int
    ) -> tuple[GGElObjective, GGElJacobian]:

    opWeight = item_weights.get('gates', 1.0)
    spamWeight = item_weights.get('spam', 1.0)
    mxBasis = model.basis

    #Use the target model's basis if model's is unknown
    # (as it can often be if it's just come from an logl opt,
    #  since from_vector will clear any basis info)
    if mxBasis.name == "unknown" and target_model is not None:
        mxBasis = target_model.basis

    assert gates_metric != "frobeniustt"
    assert spam_metric  != "frobeniustt"
    # ^ PR #410 removed support for Frobenius transform-target metrics in this codepath.

    dim = int(_np.sqrt(mxBasis.dim))
    I = _tools.IdentityOperator()
    if n_leak > 0:
        B = _tools.leading_dxd_submatrix_basis_vectors(dim - n_leak, dim, mxBasis)
        P = B @ B.T.conj()
        if _np.linalg.norm(P.imag) > 1e-12:
            msg  = f"Attempting to run leakage-aware gauge optimization with basis {mxBasis}\n"
            msg +=  "is resulting an orthogonal projector onto the computational subspace that\n"
            msg +=  "is not real-valued. Try again with a different basis, like 'l2p1' or 'gm'."
            raise ValueError(msg)
        else:
            P = P.real
    else:
        P = I
    transform_mx_arg = (P, I)
    # ^ The semantics of this tuple are defined by the frobeniusdist function
    #   in the ExplicitOpModelCalc class.

    gate_fidelity_targets : dict[ _baseobjs.Label, tuple[_np.ndarray, Union[float, _np.floating]] ] = dict()
    if gates_metric == 'fidelity':
        gate_fidelity_targets.update(_gate_fidelity_targets(model, target_model))

    prep_fidelity_targets : dict[ _baseobjs.Label, tuple[_np.ndarray, Union[float, _np.floating]] ] = dict()
    povm_fidelity_targets : dict[ _baseobjs.Label, tuple[dict, dict] ] = dict()
    if spam_metric == 'fidelity':
        prep_fidelity_targets.update(_prep_fidelity_targets(model, target_model))
        povm_fidelity_targets.update(_povm_effect_fidelity_targets(model, target_model))

    def _objective_fn(gauge_group_el: _GaugeGroupElement, oob_check: bool) -> float:
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

        if target_model is None:
            return ret
        
        if "frobenius" in gates_metric:
            if spam_metric == gates_metric:
                val = mdl.frobeniusdist(target_model, transform_mx_arg, item_weights)
            else:
                wts = item_weights.copy()
                wts['spam'] = 0.0
                for k in wts:
                    if k in mdl.preps or k in mdl.povms:
                        wts[k] = 0.0
                val = mdl.frobeniusdist(target_model, transform_mx_arg, wts, n_leak)
            if "squared" in gates_metric:
                val = val ** 2
            ret += val

        elif gates_metric == "fidelity":
            # Leakage-aware metrics NOT available
            for opLbl in mdl.operations:
                wt = item_weights.get(opLbl, opWeight)
                mop = mdl.operations[opLbl].to_dense()
                top, t = gate_fidelity_targets[opLbl]
                v = _tools.entanglement_fidelity(mop, top, mxBasis)
                z = _np.abs(t - v)
                ret += wt * z

        elif gates_metric == "tracedist":
            # If n_leak==0, then subspace_jtracedist is just jtracedist.
            for opLbl in mdl.operations:
                wt = item_weights.get(opLbl, opWeight)
                top = target_model.operations[opLbl].to_dense()
                mop = mdl.operations[opLbl].to_dense()
                ret += wt * _tools.subspace_jtracedist(top, mop, mxBasis, n_leak)

        else:
            raise ValueError("Invalid gates_metric: %s" % gates_metric)

        if "frobenius" in spam_metric and gates_metric == spam_metric:
            # We already handled SPAM error in this case. Just return.
            return ret

        if "frobenius" in spam_metric:
            # SPAM and gates can have different choices for squared vs non-squared.
            wts = item_weights.copy(); wts['gates'] = 0.0
            for k in wts:
                if k in mdl.operations or k in mdl.instruments:
                    wts[k] = 0.0
            val = mdl.frobeniusdist(target_model, transform_mx_arg, wts)
            if "squared" in spam_metric:
                val = val ** 2
            ret += val 

        elif spam_metric == "fidelity":
            # Leakage-aware metrics NOT available
            val_prep = 0.0
            for preplbl in target_model.preps:
                wt_prep = item_weights.get(preplbl, spamWeight)
                rho_curest = _tools.vec_to_stdmx(model.preps[preplbl].to_dense(), mxBasis)
                rho_target, t = prep_fidelity_targets[preplbl]
                v = _tools.fidelity(rho_curest, rho_target)
                val_prep += wt_prep * abs(t - v)
            val_povm = 0.0
            for povmlbl in target_model.povms:
                wt_povm  = item_weights.get(povmlbl, spamWeight)
                M_target = target_model.povms[povmlbl]
                M_curest =        model.povms[povmlbl]
                Es_target, ts = povm_fidelity_targets[povmlbl]
                for elbl in M_target:
                    t_e = ts[elbl]
                    E   = _tools.vec_to_stdmx(M_curest[elbl].to_dense(), mxBasis)
                    v_e = _tools.fidelity(E, Es_target[elbl])
                    val_povm += wt_povm * abs(t_e - v_e)
            ret += (val_prep + val_povm)

        elif spam_metric == "tracedist":
            # Leakage-aware metrics NOT available.
            for preplabel, m_prep in mdl.preps.items():
                wt = item_weights.get(preplabel, spamWeight)
                rhoMx1 = _tools.vec_to_stdmx(m_prep.to_dense(), mxBasis)
                t_prep = target_model.preps[preplabel]
                rhoMx2 = _tools.vec_to_stdmx(t_prep.to_dense(), mxBasis)
                ret += wt * _tools.tracedist(rhoMx1, rhoMx2)

            for povmlabel in mdl.povms.keys():
                wt = item_weights.get(povmlabel, spamWeight)
                ret += wt * _tools.povm_jtracedist(mdl, target_model, povmlabel)
        else:
            raise ValueError("Invalid spam_metric: %s" % spam_metric)

        return ret

    return _objective_fn, None


def _legacy_create_least_squares_objective(model, target_model,
        item_weights: dict[str,float], cptp_penalty_factor: float, spam_penalty_factor: float,
        gates_metric: str, spam_metric: str, comm: Optional[Any], check_jac: bool
    ) -> tuple[GGElObjective, GGElJacobian]:

    opWeight = item_weights.get('gates', 1.0)
    spamWeight = item_weights.get('spam', 1.0)
    mxBasis = model.basis

    #Use the target model's basis if model's is unknown
    # (as it can often be if it's just come from an logl opt,
    #  since from_vector will clear any basis info)
    if mxBasis.name == "unknown" and target_model is not None:
        mxBasis = target_model.basis

    assert(gates_metric.startswith("frobenius") and spam_metric.startswith("frobenius")), \
        "Only 'frobenius' and 'frobeniustt' metrics can be used when `method='ls'`!"
    assert(gates_metric == spam_metric)
    frobenius_transform_target = bool(gates_metric == 'frobeniustt')  # tt = "transform target"

    if frobenius_transform_target:
        full_target_model = target_model.copy()
        full_target_model.convert_members_inplace("full")  # so we can gauge-transform the target model.
    else:
        full_target_model = None  # in case it get's referenced by mistake

    def _objective_fn(gauge_group_el: _GaugeGroupElement, oob_check: bool) -> _np.ndarray:

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

    def _jacobian_fn(gauge_group_el: _GaugeGroupElement) -> _np.ndarray:

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
            left = -1 * _np.dot(dS, mdl_post.operations[lbl].to_dense('minimal'))  # shape (n,d1,d2)
            right = _np.swapaxes(_np.dot(G.to_dense('minimal'), dS), 0, 1)  # shape (d1,n,d2) -> (n,d1,d2)
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
                left = -1 * _np.dot(dS, mdl_post.instruments[ilbl][lbl].to_dense('minimal'))  # (n,d1,d2)
                right = _np.swapaxes(_np.dot(G.to_dense('minimal'), dS), 0, 1)  # (d1,n,d2) -> (n,d1,d2)
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
            result = -1 * _np.dot(Sinv_dS, rho.to_dense('minimal'))  # shape (d,n)
            my_jacMx[start:start + d] = wt * result
            start += d

        # -- effect terms
        # -------------------------
        for povmlbl, povm in mdl_pre.povms.items():
            for lbl, E in povm.items():
                # d(ET_term) = E.T * dS
                wt = item_weights.get(povmlbl + "_" + lbl, spamWeight)
                result = _np.dot(E.to_dense('minimal')[None, :], dS).T  # shape (1,n,d2).T => (d2,n,1)
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

    return _objective_fn, _jacobian_fn


def _create_objective_fn(model, target_model, item_weights: Optional[dict[str,float]]=None,
                         cptp_penalty_factor: float=0.0, spam_penalty_factor: float=0.0,
                         gates_metric="frobenius", spam_metric="frobenius",
                         method=None, comm=None, check_jac=False, n_leak: int=0) -> tuple[GGElObjective, GGElJacobian]:
    if item_weights is None:
        item_weights = dict()
    if method == 'ls':
        return _legacy_create_least_squares_objective(
            model, target_model, item_weights, cptp_penalty_factor, spam_penalty_factor, gates_metric,
            spam_metric, comm, check_jac
        )
    else:
        return _legacy_create_scalar_objective(
            model, target_model, item_weights, cptp_penalty_factor, spam_penalty_factor, gates_metric,
            spam_metric, n_leak
        )


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
        denMx = _tools.vec_to_stdmx(prepvec.to_dense('minimal')[:, None], op_basis)
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
        dVdp = -1 * _np.dot(Sinv_dS, prepvec.to_dense('minimal')[:, None]).squeeze(2)  # shape (d,n,1) => (d,n)
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
            EMx = _tools.vec_to_stdmx(effectvec.to_dense('minimal')[:, None], op_basis)
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
            pre_effectvec = mdl_pre.povms[povmlbl][lbl].to_dense('minimal')[:, None]
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
