"""
Wildcard budget fitting routines
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import pickle as _pickle

import numpy as _np

from pygsti.objectivefns.wildcardbudget import update_circuit_probs as _update_circuit_probs
from pygsti.optimize.optimize import minimize as _minimize

"""Developer notes

Removed functions
-----------------

    This file used to have three algorithms for optimizing wildcard budgets that relied on
    CVXOPT's nonlinear optimization interface. In June 2024 we investigated whether these
    algorithms could be re-implemented to rely only on CVXPY's modeling capabilities. We
    came to the conclusion that while that may have been possible, it would have involved
    an inordinate amount of work, and that for the sake of maintainability it was better to
    remove these CVXOPT-based algorithms from pyGSTi altogether.

    Here's a hash for one of the last commits on pyGSTi's develop branch that had these
    algorithms: 723cd24aec3b90d28b0fcd9b31145b920c256acf.

    See https://github.com/sandialabs/pyGSTi/pull/444 for more information.

"""


def optimize_wildcard_budget_neldermead(budget, L1weights, wildcard_objfn, two_dlogl_threshold,
                                        redbox_threshold, printer, smart_init=True, max_outer_iters=10,
                                        initial_eta=10.0):
    """
    Uses repeated Nelder-Mead to optimize the wildcard budget.
    Includes both aggregate and per-circuit constraints.
    """
    objfn = wildcard_objfn.logl_objfn
    layout = objfn.layout

    num_circuits = len(layout.circuits)
    dlogl_percircuit = objfn.percircuit()
    assert(len(dlogl_percircuit) == num_circuits)

    def L1term(wv): return _np.sum(_np.abs(wv) * L1weights)

    def _wildcard_fit_criteria(wv):
        dlogl_elements = wildcard_objfn.terms(wv)
        for i in range(num_circuits):
            dlogl_percircuit[i] = _np.sum(dlogl_elements[layout.indices_for_index(i)], axis=0)

        two_dlogl_percircuit = 2 * dlogl_percircuit
        two_dlogl = sum(two_dlogl_percircuit)
        two_dlogl = layout.allsum_local_quantity('c', two_dlogl)

        percircuit_penalty = sum(_np.clip(two_dlogl_percircuit - redbox_threshold, 0, None))
        percircuit_penalty = layout.allsum_local_quantity('c', percircuit_penalty)

        return max(0, two_dlogl - two_dlogl_threshold) + percircuit_penalty

    num_iters = 0
    wvec_init = budget.to_vector()

    # Optional: set initial wildcard budget by pushing on each Wvec component individually
    if smart_init:
        MULT = 2                                                                                                    # noqa
        probe = wvec_init.copy()
        for i in range(len(wvec_init)):
            #print("-------- Index ----------", i)
            wv = wvec_init.copy()
            #See how big Wv[i] needs to get before penalty stops decreasing
            last_penalty = 1e100; fit_penalty = 0.9e100
            delta = 1e-6
            while fit_penalty < last_penalty:
                wv[i] = delta
                last_penalty = fit_penalty
                fit_penalty = _wildcard_fit_criteria(wv)
                #print("  delta=%g  => penalty = %g" % (delta, penalty))
                delta *= MULT
            probe[i] = delta / MULT**2
            #print(" ==> Probe[%d] = %g" % (i, probe[i]))

        probe /= len(wvec_init)  # heuristic: set as new init point
        budget.from_vector(probe)
        wvec_init = budget.to_vector()

    printer.log("Beginning Nelder-Mead wildcard budget optimization.")
    #printer.log("Initial budget (smart_init=%s) = %s" % (str(smart_init), str(budget)))

    # Find a value of eta that is small enough that the "first terms" are 0.
    eta = initial_eta  # some default starting value - this *shouldn't* really matter
    while num_iters < max_outer_iters:
        printer.log("  Iter %d: trying eta = %g" % (num_iters, eta))

        def _wildcard_objective(wv):
            return _wildcard_fit_criteria(wv) + eta * L1term(wv)

        if printer.verbosity > 1:
            printer.log(("NOTE: optimizing wildcard budget with verbose progress messages"
                         " - this *increases* the runtime significantly."), 2)

            def callbackf(wv):
                a, b = _wildcard_fit_criteria(wv), eta * L1term(wv)
                printer.log('wildcard: misfit + L1_reg = %.3g + %.3g = %.3g Wvec=%s' %
                            (a, b, a + b, str(wv)), 2)
        else:
            callbackf = None

        #DEBUG: If you need to debug a wildcard budget, uncommend the function above and try this:
        # import bpdb; bpdb.set_trace()
        # wv_test = _np.array([5e-1, 5e-1, 5e-1, 5e-1, 0.2])  # trial budget
        # _wildcard_fit_criteria_debug(wv_test)  # try this
        # callbackf(_np.array([5e-1, 5e-1, 5e-1, 5e-1, 0.2]))  # or this

        #OLD: scipy optimize - proved unreliable
        #soln = _spo.minimize(_wildcard_objective, wvec_init,
        #                     method='Nelder-Mead', callback=callbackf, tol=1e-6)
        #if not soln.success:
        #    _warnings.warn("Nelder-Mead optimization failed to converge!")
        soln = _minimize(_wildcard_objective, wvec_init, 'supersimplex',
                         callback=callbackf, maxiter=10, tol=1e-2, abs_outer_tol=1e-4,
                         min_inner_maxiter=1000, max_inner_maxiter=1000, inner_tol=1e-6,
                         verbosity=printer)
        wvec = soln.x
        fit_penalty = _wildcard_fit_criteria(wvec)
        #printer.log("  Firstterms value = %g" % firstTerms)
        meets_conditions = bool(fit_penalty < 1e-4)  # some zero-tolerance here
        if meets_conditions:  # try larger eta
            break
        else:  # nonzero objective => take Wvec as new starting point; try smaller eta
            wvec_init = wvec
            eta /= 10

        printer.log("  Trying eta = %g" % eta)
        num_iters += 1

    budget.from_vector(wvec)
    printer.log("Optimal wildcard vector = " + str(wvec))
    return


def optimize_wildcard_budget_percircuit_only_cvxpy(budget, L1weights, objfn, redbox_threshold, printer):
    """Uses CVXPY to optimize the wildcard budget.  Includes only per-circuit constraints."""
    # Try using cvxpy to solve the problem with only per-circuit constraints
    # convex program to solve:
    # Minimize |wv|_1 (perhaps weighted) subject to the constraint:
    #  dot(percircuit_budget_deriv, wv) >= critical_percircuit_budgets
    import cvxpy as _cvxpy
    wv = budget.to_vector().copy()
    var_wv = _cvxpy.Variable(wv.shape, value=wv.copy())
    critical_percircuit_budgets = _get_critical_circuit_budgets(objfn, redbox_threshold)
    percircuit_budget_deriv = budget.precompute_for_same_circuits(objfn.global_circuits)

    constraints = [percircuit_budget_deriv @ var_wv >= critical_percircuit_budgets,
                   var_wv >= 0]
    obj = _cvxpy.Minimize(L1weights @ _cvxpy.abs(var_wv))
    # obj = _cvxpy.Minimize(_cvxpy.norm(var_wv,1))  # for special equal-weight 1-norm case
    problem = _cvxpy.Problem(obj, constraints)
    problem.solve(verbose=True)  # solver="ECOS")

    # assuming there is a step 2, walk probabilities to wv found by cvxpy to continue with more stages
    #wv_dest = var_wv.value
    #print("CVXPY solution gives wv = ", wv_dest, " advancing probs to this point...")
    #probs = wildcard_probs_propagation(budget, wv, wv_dest, objfn, layout, num_steps=10)

    printer.log("CVXPY solution (using only per-circuit constraints) gives wv = " + str(var_wv.value))
    budget.from_vector(var_wv.value)
    return


def _get_critical_circuit_budgets(objfn, redbox_threshold):
    # get set of "critical" wildcard budgets per circuit:
    # Note: this gathers budgets for all (global) circuits at end, so returned
    # `critical_percircuit_budgets` is for objfn.global_circuits

    layout = objfn.layout
    num_circuits = len(layout.circuits)  # *local* circuits
    critical_percircuit_budgets = layout.allocate_local_array('c', 'd', zero_out=True)
    raw_objfn = objfn.raw_objfn

    for i in range(num_circuits):
        p = objfn.probs[layout.indices_for_index(i)]
        f = objfn.freqs[layout.indices_for_index(i)]
        N = objfn.total_counts[layout.indices_for_index(i)]
        n = objfn.counts[layout.indices_for_index(i)]

        #This could be done more intelligently in future:
        # to hit budget, need deltaLogL = redbox_threshold
        # and decrease deltaLogL in steps: move prob from smallest_chi => largest_chi
        # - get list of "chi points" (distinct values of chi)
        # - for largest chi point, get max amount of probability to move
        # - for smallest, do the same
        # - move the smaller amt of probability
        # - check if delta logl is below threshold - if so backtrack and search for optimal movement
        #   if not, then continue

        def two_delta_logl(circuit_budget):
            q = _update_circuit_probs(p, f, circuit_budget)
            dlogl_per_outcome = raw_objfn.terms(q, n, N, f)  # N * f * _np.log(f / q)
            return 2 * float(_np.sum(dlogl_per_outcome))  # for this circuit

        TOL = 1e-6
        lbound = 0.0
        ubound = 1.0
        while ubound - lbound > TOL:
            mid = (ubound + lbound) / 2
            mid_val = two_delta_logl(mid)
            if mid_val < redbox_threshold:  # fits well, can decrease budget
                ubound = mid
            else:  # fits poorly (red box!), must increase budget
                lbound = mid
        percircuit_budget = (ubound + lbound) / 2
        critical_percircuit_budgets[i] = percircuit_budget

    global_critical_percircuit_budgets = layout.allgather_local_array('c', critical_percircuit_budgets)
    layout.free_local_array(critical_percircuit_budgets)
    return global_critical_percircuit_budgets


def _agg_dlogl(current_probs, objfn, two_dlogl_threshold):
    #Note: current_probs is a *local* quantity
    p, f, n, N = current_probs, objfn.freqs, objfn.counts, objfn.total_counts
    dlogl_elements = objfn.raw_objfn.terms(p, n, N, f)  # N * f * _np.log(f / p)
    global_dlogl_sum = objfn.layout.allsum_local_quantity('c', float(_np.sum(dlogl_elements)))
    return 2 * global_dlogl_sum - two_dlogl_threshold


def _agg_dlogl_deriv(current_probs, objfn, percircuit_budget_deriv, probs_deriv_wrt_percircuit_budget):
    #Note: current_probs and percircuit_budget_deriv are *local* quantities
    #dlogl_delements = objfn.raw_objfn.dterms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
    p, f, n, N = current_probs, objfn.freqs, objfn.counts, objfn.total_counts
    dlogl_delements = objfn.raw_objfn.dterms(p, n, N, f)  # -N*f/p
    #chi_elements = -dlogl_delements / N  # f/p = -dlogl_delements / N
    layout = objfn.layout
    num_circuits = len(layout.circuits)

    # derivative of firstterms wrt per-circuit wilcard budgets - namely if that budget goes up how to most efficiently
    # reduce firstterms in doing so, this computes how the per-circuit budget should be allocated to probabilities
    # (i.e. how probs should be updated) to achieve this decrease in firstterms
    agg_dlogl_deriv_wrt_percircuit_budgets = _np.zeros(num_circuits, 'd')
    for i in range(num_circuits):
        elInds = layout.indices_for_index(i)

        #OLD
        #chis = chi_elements[elInds]  # ~ f/p
        #Nloc = N[elInds]
        #agg_dlogl_deriv_wrt_percircuit_budgets[i] = -2 * Nloc[0] * (_np.max(chis) - _np.min(chis))

        dlogl_dp = dlogl_delements[elInds]
        dp_dW = probs_deriv_wrt_percircuit_budget[elInds]
        agg_dlogl_deriv_wrt_percircuit_budgets[i] = 2 * _np.sum(dlogl_dp * dp_dW)

        #agg_dlogl_deriv_wrt_percircuit_budgets[i] = -2 * Nloc[0] * (_softmax(chis) - _softmin(chis)) # SOFT MAX/MIN

        #wts = _np.abs(dlogl_helements[layout.indices_for_index(i)])
        #maxes = _np.array(_np.abs(chis - _np.max(chis)) < 1.e-4, dtype=int)
        #mins = _np.array(_np.abs(chis - _np.min(chis)) < 1.e-4, dtype=int)
        #agg_dlogl_deriv_wrt_percircuit_budgets[i] = -_np.sum(chis * ((mins * wts) / sum(mins * wts) \
        #    - (maxes * wts) / sum(maxes * wts)))
    assert(_np.all(agg_dlogl_deriv_wrt_percircuit_budgets <= 1e-6)), \
        "Derivative of aggregate LLR wrt any circuit budget should be negative"
    local_deriv = _np.dot(agg_dlogl_deriv_wrt_percircuit_budgets, percircuit_budget_deriv)
    return objfn.layout.allsum_local_quantity('c', local_deriv, use_shared_mem=False)


def _agg_dlogl_hessian(current_probs, objfn, percircuit_budget_deriv, probs_deriv_wrt_percircuit_budget):
    #dlogl_delements = objfn.raw_objfn.dterms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
    #dlogl_helements = objfn.raw_objfn.hterms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
    p, f, n, N = current_probs, objfn.freqs, objfn.counts, objfn.total_counts
    #dlogl_delements = objfn.raw_objfn.dterms(p, n, N, f)  # -N*f/p  < 0
    dlogl_helements = objfn.raw_objfn.hterms(p, n, N, f)  # N*f/p**2 > 0
    #chi_elements = -dlogl_delements / N  # f / p
    #dchi_elements = dlogl_helements / N  # f / p**2
    layout = objfn.layout
    num_circuits = len(layout.circuits)

    # derivative of firstterms wrt per-circuit wilcard budgets - namely if that budget goes up how to most efficiently
    # reduce firstterms. In doing so, this computes how the per-circuit budget should be allocated to probabilities
    # (i.e. how probs should be updated) to achieve this decrease in firstterms
    #TOL = 1e-6
    agg_dlogl_hessian_wrt_percircuit_budgets = _np.zeros(num_circuits)
    for i in range(num_circuits):
        elInds = layout.indices_for_index(i)

        # agg_dlogl(p(W))
        # d(agg_dlogl)/dW = dagg_dlogl(p(W)) * dp_dW  (directional derivative of agg_dlogl)
        # d2(agg_dlogl)/dW = dp_dW * hagg_dlogl(p(W)) * dp_dW   ("directional" Hessian of agg_dlogl)
        hlogl_dp = dlogl_helements[elInds]
        dp_dW = probs_deriv_wrt_percircuit_budget[elInds]

        old_err = _np.seterr(over='ignore')
        agg_dlogl_hessian_wrt_percircuit_budgets[i] = 2 * _np.sum(hlogl_dp * dp_dW**2)  # check for overflow
        _np.seterr(**old_err)

        if not _np.isfinite(agg_dlogl_hessian_wrt_percircuit_budgets[i]):  # deal with potential overflow
            agg_dlogl_hessian_wrt_percircuit_budgets[i] = 1e100  # something huge

        #TODO: see if there's anything useful here, and then REMOVE
        #NOTE - starting to think about alternate objectives with softened "Hessian jump" at dlogl == 0 point.
        # when two outcomes and very close to all f/p == 1: f1/p1 = f1/(f1-eps) ~= 1 + eps/f1   ,   f2/p2 = f2/(f2 + eps) ~= 1 - eps/f2  # noqa
        # then hessian is f1/p1^2 + f2/p2^2 ~= 1/p1 + eps/(f1p1) + 1/p2 + eps/(f2p2) = 1/(f1-eps) + eps/(f1*(f1-eps)) ... ~= 1/f1 + 1/f2  # noqa

        # at all chi=f/p == 1 (where dlogl = 0), hessian is sum( (f/p) * 1/p * f/f_sum ) = sum( f/p ) = N_outcomes
        # if added -Noutcomes to hessian, then get:
        #  -Noutcomes*wc_budget + C1  addition to derivative
        #  -0.5*Noutcomes*wc_budget^2 + C1*wc_budget + C2   addition to objective
        # #maxes = _np.exp(chis) / _np.sum(_np.exp(chis))  # SOFT MAX
        # #mins = _np.exp(-chis) / _np.sum(_np.exp(-chis))  # SOFT MIN
        # one_over_dchi = one_over_dchi_elements[layout.indices_for_index(i)]  # ~ p**2/f
        # agg_dlogl_hessian_wrt_percircuit_budgets[i] = 2 * Nloc[0] * (1 / _np.sum(one_over_dchi * maxes) \
        #   + 1 / _np.sum(one_over_dchi * mins))

        #wts = 1.0 / _np.abs(dlogl_helements[layout.indices_for_index(i)])
        #hterms = dlogl_helements[layout.indices_for_index(i)]  # ~ -f/p**2
        #maxes = _np.array(_np.abs(chis - _np.max(chis)) < 1.e-4, dtype=int)
        #mins = _np.array(_np.abs(chis - _np.min(chis)) < 1.e-4, dtype=int)
        ##Deriv of -N*f/p * (N*f/p**2) /
        #agg_dlogl_hessian_wrt_percircuit_budgets[i] = _np.sum(hterms * ((mins * wts) / sum(mins * wts)
        #  - (maxes * wts) / sum(maxes * wts)))

    assert(_np.all(agg_dlogl_hessian_wrt_percircuit_budgets >= 0)), \
        "Hessian of aggregate LLR wrt any circuit budget should be positive"
    local_H = _np.dot(percircuit_budget_deriv.T,
                      _np.dot(_np.diag(agg_dlogl_hessian_wrt_percircuit_budgets),
                              percircuit_budget_deriv))   # (nW, nC)(nC)(nC, nW)
    #local_evals = _np.linalg.eigvals(local_H)
    #assert(_np.all(local_evals >= -1e-8))
    return objfn.layout.allsum_local_quantity('c', local_H, use_shared_mem=False)


def _get_percircuit_budget_deriv(budget, layout):
    """ Returns local_percircuit_budget_deriv, global_percircuit_budget_deriv """
    percircuit_budget_deriv = budget.precompute_for_same_circuits(layout.circuits)  # for *local* circuits

    #Note: maybe we could do this gather in 1 call (?), but we play it safe and do it col-by-col
    global_percircuit_budget_deriv_cols = []
    for i in range(percircuit_budget_deriv.shape[1]):
        global_percircuit_budget_deriv_cols.append(
            layout.allgather_local_array('c', percircuit_budget_deriv[:, i]))
    return percircuit_budget_deriv, _np.column_stack(global_percircuit_budget_deriv_cols)


def optimize_wildcard_bisect_alpha(budget, objfn, two_dlogl_threshold, redbox_threshold, printer,
                                   guess=0.1, tol=1e-3):
    printer.log("Beginning wildcard budget optimization using alpha bisection method.")

    layout = objfn.layout
    critical_percircuit_budgets = _get_critical_circuit_budgets(objfn, redbox_threshold)  # for *global* circuits
    percircuit_budget_deriv, global_percircuit_budget_deriv = _get_percircuit_budget_deriv(budget, layout)
    if _np.linalg.norm(percircuit_budget_deriv) < 1e-10:
        raise ValueError("Wildcard scaling does not affect feasibility (deriv is zero)!")

    initial_probs = objfn.probs.copy()
    current_probs = initial_probs.copy()
    probs_freqs_precomp = budget.precompute_for_same_probs_freqs(initial_probs, objfn.freqs, layout)

    def is_feasible(x):
        budget.from_vector(x)
        budget.update_probs(initial_probs, current_probs, objfn.freqs, objfn.layout, percircuit_budget_deriv,
                            probs_freqs_precomp)
        f0 = _np.array([_agg_dlogl(current_probs, objfn, two_dlogl_threshold)])
        fi = critical_percircuit_budgets - _np.dot(global_percircuit_budget_deriv, x)
        return _np.all(_np.concatenate((f0, fi)) <= 0)  # All constraints must be negative to be feasible

    left = None
    right = None

    while left is None or right is None:
        printer.log(f'Searching for interval [{left}, {right}] with guess {guess}', 2)
        # Test for feasibility
        if is_feasible(_np.array([guess], 'd')):
            printer.log('Guess value is feasible, ', 2)
            left = guess
            guess = left / 2
        else:
            printer.log('Guess value is infeasible, ', 2)
            right = guess
            guess = 2 * right
            if guess > 1e10:
                raise ValueError("Feasible wildcard scaling cannot be found!")
    printer.log('Interval found!', 2)

    # We now have an interval containing the crossover point
    # Perform bisection
    while abs(left - right) > tol:
        printer.log(f'Performing bisection on interval [{left}, {right}]', 2)
        test = left - (left - right) / 2.0

        if is_feasible(_np.array([test], 'd')):
            # Feasible, so shift left down
            printer.log('Test value is feasible, ', 2)
            left = test
        else:
            printer.log('Test value is infeasible, ', 2)
            right = test

    printer.log('Interval within tolerance!', 2)

    budget.from_vector(_np.array([left], 'd'))  # set budget to the feasible one
    printer.log(f'Optimized value of alpha = {left}')
    return


def optimize_wildcard_budget_barrier(budget, L1weights, objfn, two_dlogl_threshold,
                                     redbox_threshold, printer, tol=1e-7, max_iters=50, num_steps=3,
                                     save_debugplot_data=False):
    """
    Uses a barrier method (for convex optimization) to optimize the wildcard budget.
    Includes both aggregate and per-circuit constraints.
    """
    #BARRIER method:
    # Solve:            min c^T * x
    # Subject to:       F(x) <= 0
    # by actually solving (via Newton):
    #  min t * c^T * x + phi(x)
    # where phi(x) = -log(-F(x))
    # for increasing values of t until 1/t <= epsilon (precision tolerance)
    printer.log("Beginning wildcard budget optimization using a barrier method.")
    layout = objfn.layout
    critical_percircuit_budgets = _get_critical_circuit_budgets(objfn, redbox_threshold)  # for *global* circuits
    percircuit_budget_deriv, global_percircuit_budget_deriv = _get_percircuit_budget_deriv(budget, layout)

    x0 = budget.to_vector()
    initial_probs = objfn.probs.copy()
    current_probs = initial_probs.copy()
    probs_freqs_precomp = budget.precompute_for_same_probs_freqs(initial_probs, objfn.freqs, layout)

    # f0 = 2DLogL - threshold <= 0
    # fi = critical_budget_i - circuit_budget_i <= 0
    #    = critical_percircuit_budgets - dot(percircuit_budget_deriv, x) <= 0
    # fj = -x_j <= 0

    def penalty_vec(x):
        budget.from_vector(x)
        budget.update_probs(initial_probs, current_probs, objfn.freqs, layout, percircuit_budget_deriv,
                            probs_freqs_precomp)
        f0 = _np.array([_agg_dlogl(current_probs, objfn, two_dlogl_threshold)])
        fi = critical_percircuit_budgets - _np.dot(global_percircuit_budget_deriv, x)
        return _np.concatenate((f0, fi))

    def barrierF(x, compute_deriv=True):
        assert(min(x) >= 0)  # don't allow negative wildcard vector components

        budget.from_vector(_np.array(x))
        p_deriv = budget.update_probs(initial_probs, current_probs, objfn.freqs, layout,
                                      percircuit_budget_deriv, probs_freqs_precomp, return_deriv=True)
        f0 = _np.array([_agg_dlogl(current_probs, objfn, two_dlogl_threshold)])
        fi = critical_percircuit_budgets - _np.dot(global_percircuit_budget_deriv, x)
        f = _np.concatenate((f0, fi, -x))  # adds -x for x >= 0 constraint
        val = -_np.sum(_np.log(-f))
        if not compute_deriv: return val

        Df0 = _agg_dlogl_deriv(current_probs, objfn, percircuit_budget_deriv, p_deriv)
        deriv = -1 / f0 * Df0 - _np.dot(1 / fi, percircuit_budget_deriv) - 1 / x

        Hf0 = _agg_dlogl_hessian(current_probs, objfn, percircuit_budget_deriv, p_deriv)
        hess = 1 / f0**2 * Df0[:, None] * Df0[None, :] - 1 / f0 * Hf0 \
            + _np.einsum('i,ij,ik->jk', 1 / fi**2, global_percircuit_budget_deriv, global_percircuit_budget_deriv) \
            + _np.diag(1 / x**2)
        # sum_i 1 / fi[i]**2 * percircuit_budget_deriv[i,:,None] * percircuit_budget_deriv[i,None,:]
        # (i,) (i,j) (i,k)
        return val, deriv, hess

    #Find a valid initial point
    initial_penalty_vec = penalty_vec(x0)
    num_constraints = len(initial_penalty_vec) + len(x0)  # 2nd term b/c x >= 0 constraints
    if _np.all(initial_penalty_vec <= 0):
        printer.log("Initial (feasible) point: " + str(x0))
    else:
        if _np.linalg.norm(x0) < 1e-5: x0[:] = 1e-5  # just so we don't start at all zeros
        i = 0
        while i < 100:
            if _np.all(penalty_vec(x0) <= 0): break
            x0 *= 2.0; i += 1
        else:
            raise ValueError("Could not find feasible starting point!")
        printer.log("Found initial feasible point: " + str(x0))
    x = x0.copy()  # set initial point

    log10_end = int(_np.ceil(_np.log10(2 * num_constraints / tol)))  # 2 factor just for good measure
    log10_begin = log10_end - (num_steps - 1)
    t_values = _np.logspace(log10_begin, log10_end, num_steps)
    #t_values = [1e5, 1e6, 1.e7, 1e8, 1e9]

    SMALL_values = [0] * len(t_values)
    #SMALL_values = [1 / (10 * t) for t in t_values]

    if save_debugplot_data:
        with open("debug/num_stages", 'wb') as pipe:
            _pickle.dump(len(t_values), pipe)

    for iStage, (t, SMALL) in enumerate(zip(t_values, SMALL_values)):  # while 1/t > epsilon:

        printer.log("*** Beginning stage %d with t=%g, SMALL=%g ***" % (iStage, t, SMALL))
        SMALL2 = SMALL**2
        bFn = barrierF

        #  min t * c^T * x + phi(x)
        # where phi(x) = -log(-F(x))
        # - try: c^T * x = sum(c_i * x_i) => sum( sqrt((c_i * x_i)^2 + SMALL^2) )
        #        deriv =>  0.5/sqrt(...)* 2*c_i^2*x_i
        #        hess => sum( -1.0/(...)^(3/2) * (c_i^2*x_i)^2 + 1.0/sqrt(...)*c_i^2 )
        c = L1weights

        def NewtonObjective(x):
            barrier_val = bFn(x, compute_deriv=False)
            #return t_value * _np.dot(c.T, x) + barrier_val
            return float(t * _np.sum(_np.sqrt((c * x)**2 + SMALL2)) + barrier_val)

        def NewtonObjective_derivs(x):
            barrier, Dbarrier, Hbarrier = bFn(x)
            #obj = t * _np.dot(c.T, x) + barrier
            #Dobj = t * c.T + Dbarrier
            #Hobj = Hbarrier
            if SMALL2 == 0.0:  # then obj = |c * x|, Dobj = c, Hobj = 0
                obj = t * sum(_np.abs(c * x)) + barrier
                Dobj = t * c + Dbarrier
                Hobj = Hbarrier
            else:
                sqrtVec = _np.sqrt((c * x)**2 + SMALL2)
                obj = t * _np.sum(sqrtVec) + barrier
                Dobj = t * (c**2 * x / sqrtVec) + Dbarrier
                Hobj = t * _np.diag(-1.0 / (sqrtVec**3) * (c**2 * x)**2 + c**2 / sqrtVec) + Hbarrier
            return obj, Dobj, Hobj

        x, debug_x_list = NewtonSolve(x, NewtonObjective, NewtonObjective_derivs, tol, max_iters, printer - 1)
        #x, debug_x_list = NewtonSolve(x, NewtonObjective, None, tol, max_iters, printer - 1)  # use finite-diff derivs

        if save_debugplot_data:
            with open("debug/xlist_stage%d" % iStage, 'wb') as pipe:
                _pickle.dump(debug_x_list, pipe)

            VIEW = 0.00002
            with open("debug/x_stage%d" % iStage, 'wb') as pipe:
                _pickle.dump(x, pipe)
            for ii in range(len(x)):
                xcopy = x.copy(); pairs = []
                for xx in _np.linspace(max(x[ii] - VIEW, 0), x[ii] + VIEW, 100):
                    xcopy[ii] = xx
                    pairs.append((xx, NewtonObjective(xcopy)))
                with open("debug/pairs_stage%d_axis%d" % (iStage, ii), 'wb') as pipe:
                    _pickle.dump(pairs, pipe)

            if len(x0) >= 2:  # Contour plots works only when there are at least 2 coordinates
                w0_list = [xx[0] for xx in debug_x_list]
                w1_list = [xx[1] for xx in debug_x_list]
                w0 = _np.linspace(min(w0_list) * 0.9, max(w0_list) * 1.1, 50)
                w1 = _np.linspace(min(w1_list) * 0.9, max(w1_list) * 1.1, 50)

                with open("debug/contour_w0_stage%d" % iStage, 'wb') as pipe:
                    _pickle.dump(w0, pipe)
                with open("debug/contour_w1_stage%d" % iStage, 'wb') as pipe:
                    _pickle.dump(w1, pipe)

                zvals = _np.zeros((len(w1), len(w0)), 'd')
                for jj, ww1 in enumerate(w1):
                    for kk, ww0 in enumerate(w0):
                        xvec = x.copy(); xvec[0] = ww0; xvec[1] = ww1
                        zvals[jj, kk] = NewtonObjective(xvec)
                with open("debug/contour_vals_stage%d" % iStage, 'wb') as pipe:
                    _pickle.dump(zvals, pipe)

    budget.from_vector(x)
    printer.log("Optimal wildcard vector = " + str(x))
    return


def NewtonSolve(initial_x, fn, fn_with_derivs=None, dx_tol=1e-6, max_iters=20, printer=None, lmbda=0.0):
    # lmbda crudely interpolates between Newton (0.0) and gradient (1.0) descent

    x_list = [initial_x.copy()]
    x = initial_x.copy()
    I = _np.identity(len(x), 'd')
    test_obj = None

    i = 0
    while i < max_iters:
        if fn_with_derivs:
            obj, Dobj, Hobj = fn_with_derivs(x)
            #DEBUG - check against finite diff
            #obj_chk = fn(x)
            #Dobj_chk, Hobj_chk = _compute_fd(x, fn)
            #print("Chks = ",_np.linalg.norm(obj - obj_chk),
            #      _np.linalg.norm(Dobj - Dobj_chk) / _np.linalg.norm(Dobj),
            #      _np.linalg.norm(Hobj - Hobj_chk) / _np.linalg.norm(Hobj))
        else:
            obj = fn(x)
            Dobj, Hobj = _compute_fd(x, fn)
        Hobj += Hobj.T
        Hobj /= 2
        evalsH = _np.linalg.eigvalsh(Hobj)
        assert(min(evalsH) >= 0 or abs(min(evalsH) / max(evalsH)) < 1e-8)
        # Note: OK if evalsH has small negative elements, where "small" is relative to positive elements

        norm_Dobj = _np.linalg.norm(Dobj)
        #dx = - _np.dot(_np.linalg.inv(H), Df.T)
        Hrank = _np.linalg.matrix_rank(Hobj)
        if Hrank < Hobj.shape[0]:
            if printer: printer.log("Rank defficient Hessian (%d < %d) - using gradient step" % (Hrank, Hobj.shape[0]))
            dx = - Dobj / _np.linalg.norm(Dobj)
        else:
            dx = - _np.dot((1 - lmbda) * _np.linalg.inv(Hobj) + lmbda * I, Dobj)

        #if debug and i == 0:
        #    print(" initial newton iter: f=%g, |Df|=%g, |Hf|=%g" % (obj, norm_Dobj, _np.linalg.norm(Hobj)))
        #    print(" dx = ",dx)
        if test_obj is not None:
            assert(_np.isclose(obj, test_obj))  # Sanity check

        #downhill_direction = - Dobj / _np.linalg.norm(Dobj)
        #dx_before_backtrack = dx.copy()

        #print("DB: last obj = ",obj)
        orig_err = _np.geterr()
        _np.seterr(divide='ignore', invalid='ignore')
        while(_np.linalg.norm(dx) >= dx_tol):
            test_x = _np.clip(x + dx, 0, None)

            test_obj = fn(test_x)
            #print("DB: test obj = ",test_obj, " (dx = ",_np.linalg.norm(dx),")")
            if test_obj < obj: break
            else:
                dx *= 0.1  # backtrack
                #if debug: print("Backtrack |dx| = ",_np.linalg.norm(dx))
        else:
            # if debug: print("Can't step in Newton direction and reduce objective - trying gradient descent")
            #
            # dx = - Dobj.T / _np.linalg.norm(Dobj)
            # while(_np.linalg.norm(dx) >= dx_tol):
            #     test_x = _np.clip(x + dx,0,None)
            #     test_obj = fn(test_x)
            #     #print("TEST: ",list(test_x),test_obj,obj,test_obj[0,0] < obj[0,0],dx)
            #     if test_obj < obj: break
            #     else: dx *= 0.5
            # else:
            #     if debug: print("Can't step in gradient direction and reduce objective - converged at f=%g" % obj)
            #     break

            _np.seterr(**orig_err)
            if printer: printer.log("Can't step in Newton direction and reduce objective - converged at f=%g" % obj)
            break

        _np.seterr(**orig_err)
        norm_x = _np.linalg.norm(x)
        norm_dx = _np.linalg.norm(dx)
        if printer:
            printer.log(" newton iter %d: f=%g, |x|=%g |Df|=%g, |dx|=%g |Hf|=%g" %
                        (i, obj, norm_x, norm_Dobj, norm_dx, _np.linalg.norm(Hobj)))
            #print("   downhill = ", list(downhill_direction.flat))
            #print("   dx_before_backtrack = ", list(dx_before_backtrack.flat))
            #print("   dx = ", list(dx.flat))
            #print("   new_x = ", list((x + dx).flat))
            #print("   H evals = ", evalsH)
            #print("   H eigenvecs = \n", eigvecsH)
            #print("   H = \n", Hobj)
        x += dx
        x = _np.clip(x, 0, None)
        x_list.append(x.copy())
        i += 1
        if norm_dx < dx_tol: break  # norm_Dobj < 1e-4 or
    if i == max_iters and printer:
        printer.log("WARNING: max iterations exceeded!!!")
    return x, x_list


def _compute_fd(x, fn, compute_hessian=True, eps=1e-7):
    x_len = len(x)
    grad = _np.zeros(x_len, 'd')
    f0 = fn(x)
    for k in range(x_len):
        x_eps = x.copy(); x_eps[k] += eps
        f_eps = fn(x_eps)
        grad[k] = (f_eps - f0) / eps
    if compute_hessian is False: return grad

    eps = 1e-5
    hess = _np.zeros((x_len, x_len), 'd')
    for k in range(x_len):
        x_eps_k = x.copy(); x_eps_k[k] += eps
        f_eps_k = fn(x_eps_k)
        for l in range(x_len):
            x_eps_l = x.copy(); x_eps_l[l] += eps
            f_eps_l = fn(x_eps_l)
            x_eps_kl = x_eps_k.copy(); x_eps_kl[l] += eps
            f_eps_kl = fn(x_eps_kl)
            hess[k, l] = (f_eps_kl - f_eps_k - f_eps_l + f0) / eps**2
    return grad, hess
