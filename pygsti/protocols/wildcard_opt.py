

import numpy as _np
import pickle as _pickle
from ..objects.wildcardbudget import update_circuit_probs as _update_circuit_probs


def optimize_wildcard_budget_percircuit_only_cvxpy(budget, L1weights, objfn, layout, redbox_threshold):
    # Try using cvxpy to solve the problem with only per-circuit constraints
    # convex program to solve:
    # Minimize |wv|_1 (perhaps weighted) subject to the constraint:
    #  dot(percircuit_budget_deriv, wv) >= critical_percircuit_budgets
    import cvxpy as _cvxpy
    wv = budget.to_vector().copy()
    var_wv = _cvxpy.Variable(wv.shape, value=wv.copy())
    critical_percircuit_budgets = _get_critical_circuit_budgets(objfn, layout, redbox_threshold)
    percircuit_budget_deriv = budget.precompute_for_same_circuits(layout.circuits)
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

    print("CVXPY solution (using only per-circuit constraints) gives wv = ", var_wv.value)
    budget.from_vector(var_wv.value)
    return


def _get_critical_circuit_budgets(objfn, layout, redbox_threshold):
    # get set of "critical" wildcard budgets per circuit:
    num_circuits = len(layout.circuits)
    critical_percircuit_budgets = _np.zeros(num_circuits, 'd')
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
            dlogl_per_outcome = N * f * _np.log(f / q)
            return 2 * float(_np.sum(dlogl_per_outcome))  # for this circuit

        TOL = 1e-6
        lbound = 0.0
        ubound = 1.0
        while ubound-lbound > TOL:
            mid = (ubound + lbound) / 2
            mid_val = two_delta_logl(mid)
            if mid_val < redbox_threshold:  # fits well, can decrease budget
                ubound = mid
            else:  # fits poorly (red box!), must increase budget
                lbound = mid
        percircuit_budget = (ubound + lbound) / 2
        
        #percircuit_budget = 0; step = 1e-5
        #while True:
        #    #dlogl_per_outcome = objfn.raw_objfn.terms(p, n, N, f)
        #    dlogl_per_outcome = N * f * _np.log(f / p)
        #    dlogl = float(_np.sum(dlogl_per_outcome))  # for this circuit
        #    if 2 * dlogl <= redbox_threshold: break
        #
        #    chis = objfn.raw_objfn.dterms(p, n, N, f)
        #    maxes = _np.array(_np.abs(chis - _np.max(chis)) < 1.e-8, dtype=int)
        #    mins = _np.array(_np.abs(chis - _np.min(chis)) < 1.e-8, dtype=int)
        #    add_to = step * mins / sum(mins)
        #    take_from = step * maxes / sum(maxes)
        #    p += add_to - take_from
        #    percircuit_budget += step

        critical_percircuit_budgets[i] = percircuit_budget
    return critical_percircuit_budgets


# Aggregate 2-delta-logl criteria (for cvxopt call below, as we want this function to be <= 0)
#  - for each circuit, we have the sum of -2Nf*logl(p) + const. terms
#  - the derivatives taken below are complicated because they're derivatives with respect to
#     the circuit's *wildcard budget*, which is effectively w.r.t `p` except all the p's must
#     sum to 1.  We compute these derivatives as follows:
#
#    - 1st deriv: the first derivative of each term is -Nf/p and N is common to all the terms of
#      a single circuit so this is dictated by chi = f/p >= 0.  All these terms are positive (the
#      deriv is negative), and we want to move probability from the terms with smallest chi to
#      largest chi.  Note here that positive `p` means *more* wildcard budget and so the largest-chi
#      terms have their p_i increase (dp_i = dp) whereas the smallest-chi terms have p_i decrease
#      (dp_i = -dp).  When multiple terms have the same chi then we split the total dp
#      (delta-probability) according to 1 / 2nd-deriv = p**2/Nf.  This is so that if
#      chi1 = f1/p1 = chi2 = f2/p2 and we want the chi's to remain equal after
#      p1 -> p1 + lambda1*dp, p2 -> p2 + lambda2*dp then we get:
#      (p1 + lambda1*dp) / f1 = 1/chi1 + lambda1/f1 * dp = 1/chi2 + lambda2/f2 * dp, so
#      lambda1/f1 = lambda2/f2 => lambda1/lambda2 = f1/f2.  Since lambda1 + lambda2 = 1,
#      we get lambda1 (1 + f2/f1) = 1 => lambda1 = f1 / (f1 + f2)
#      In general, lambda_i = f_i / sum_fs_with_max_chi.
#      Note: f1/p1 = f2/p2 => f1/f2 = p1/p2 so lambda_i also could be = p_i / sum_ps_with_max_chi
#      We could also derive by wanting the derivs wrt chi be equal:
#       d(chi1)/dp = d(chi2)/dp => -f1/p1**2 * lambda_1 = -f2/p2**2 * lambda_2
#       => lambda1/lambda2 = p1/p2 as before (recall dp1 = lambda1 * dp)
#      Note that this also means the lambdas could be weighted by the full 2nd deriv: Nf/p**2
#      ** IN SUMMARY, the total derivative is:
#           -2N * (sum_max_chi(f_i/p_i * lambda_i) - sum_min_chi(f_i/p_i * lambda_i))
#           = -2N * (max_chi - min_chi)
#
#    - 2nd deriv: same as above, but now different lambda_i matter:
#         = 2N * (sum_max_chi(f_i/p_i**2 * lambda_i**2) - sum_min_chi(f_i/p_i**2 * lambda_i**2))
#         (where we take the lambda_i as given by the frequencies, so they aren't diff'd)
#      If we took lambda_i = p_i / sum_of_ps then we'd get:
#      d/dp (f_i/p_i * lambda_i) = -f_i/p_i**2 * lambda_i**2 + f_i/p_i * dlambda_i/dp
#                                = -f_i/p_i**2 * lambda_i**2 (see below)
#      Note dlambda_i/dp = lambda_i / sum_of_ps - p_i / (sum_ps)**2 * sum(lambda_i) = 0
#      So we get the same result.


def _agg_dlogl(current_probs, objfn, two_dlogl_threshold):
    p, f, N = current_probs, objfn.freqs, objfn.total_counts
    dlogl_elements = N * f * _np.log(f / p)
    #dlogl_elements = objfn.raw_objfn.terms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
    return 2 * float(_np.sum(dlogl_elements)) - two_dlogl_threshold  # ~ -Nf*log(p)


def _agg_dlogl_deriv(current_probs, objfn, layout, percircuit_budget_deriv):
    #dlogl_delements = objfn.raw_objfn.dterms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
    p, f, N = current_probs, objfn.freqs, objfn.total_counts
    #dlogl_delements = -N*f/p
    chi_elements = f / p  # -dlogl_delements / N  # i.e. f/p
    num_circuits = len(layout.circuits)

    # derivative of firstterms wrt per-circuit wilcard budgets - namely if that budget goes up how to most efficiently
    # reduce firstterms in doing so, this computes how the per-circuit budget should be allocated to probabilities
    # (i.e. how probs should be updated) to achieve this decrease in firstterms
    agg_dlogl_deriv_wrt_percircuit_budgets = _np.zeros(num_circuits, 'd')
    for i in range(num_circuits):
        chis = chi_elements[layout.indices_for_index(i)]  # ~ f/p
        Nloc = N[layout.indices_for_index(i)]
        agg_dlogl_deriv_wrt_percircuit_budgets[i] = -2 * Nloc[0] * (_np.max(chis) - _np.min(chis))
        #agg_dlogl_deriv_wrt_percircuit_budgets[i] = -2 * Nloc[0] * (_softmax(chis) - _softmin(chis)) # SOFT MAX/MIN

        #wts = _np.abs(dlogl_helements[layout.indices_for_index(i)])
        #maxes = _np.array(_np.abs(chis - _np.max(chis)) < 1.e-4, dtype=int)
        #mins = _np.array(_np.abs(chis - _np.min(chis)) < 1.e-4, dtype=int)
        #agg_dlogl_deriv_wrt_percircuit_budgets[i] = -_np.sum(chis * ((mins * wts) / sum(mins * wts) \
        #    - (maxes * wts) / sum(maxes * wts)))
    assert(_np.all(agg_dlogl_deriv_wrt_percircuit_budgets <= 0)), \
        "Derivative of aggregate LLR wrt any circuit budget should be negative"
    return _np.dot(agg_dlogl_deriv_wrt_percircuit_budgets, percircuit_budget_deriv)


def _agg_dlogl_hessian(current_probs, objfn, layout, percircuit_budget_deriv):
    #dlogl_delements = objfn.raw_objfn.dterms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
    #dlogl_helements = objfn.raw_objfn.hterms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
    p, f, N = current_probs, objfn.freqs, objfn.total_counts
    #dlogl_delements = -N*f/p  < 0
    #dlogl_helements = N*f/p**2 > 0
    chi_elements = f / p  # -dlogl_delements / N  # i.e. f/p
    #one_over_dchi_elements = p**2 / f  # N / dlogl_helements # i.e. p**2/f
    dchi_elements = f / p**2  # N / dlogl_helements # i.e. p**2/f
    num_circuits = len(layout.circuits)

    # derivative of firstterms wrt per-circuit wilcard budgets - namely if that budget goes up how to most efficiently
    # reduce firstterms. In doing so, this computes how the per-circuit budget should be allocated to probabilities
    # (i.e. how probs should be updated) to achieve this decrease in firstterms
    TOL = 1e-6
    agg_dlogl_hessian_wrt_percircuit_budgets = _np.zeros(num_circuits)
    for i in range(num_circuits):
        chis = chi_elements[layout.indices_for_index(i)]  # ~ f/p
        Nloc = N[layout.indices_for_index(i)]
        max_chi = _np.max(chis)
        min_chi = _np.min(chis)
        if (max_chi - min_chi) < TOL:  # Special case when all f==p - nothing more to change
            agg_dlogl_hessian_wrt_percircuit_budgets[i] = 0
            continue

        max_mask = _np.abs(chis - max_chi) < TOL
        min_mask = _np.abs(chis - min_chi) < TOL
        # maxes = _np.array(max_mask, dtype=int)
        # mins = _np.array(min_mask, dtype=int)

        freqs = f[layout.indices_for_index(i)]
        lambdas_max = freqs[max_mask] / sum(freqs[max_mask])
        lambdas_min = freqs[min_mask] / sum(freqs[min_mask])

        dchi = dchi_elements[layout.indices_for_index(i)]  # ~ f/p**2
        agg_dlogl_hessian_wrt_percircuit_budgets[i] = \
            2 * Nloc[0] * (sum(dchi[max_mask] * lambdas_max**2)
                           + sum(dchi[min_mask] * lambdas_min**2))

        #HERE - starting to think about alternate objectives with softened "Hessian jump" at dlogl == 0 point.
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
    H = _np.dot(percircuit_budget_deriv.T,
                _np.dot(_np.diag(agg_dlogl_hessian_wrt_percircuit_budgets),
                        percircuit_budget_deriv))   # (nW, nC)(nC)(nC, nW)
    evals = _np.linalg.eigvals(H)
    assert(_np.all(evals >= -1e-8))
    return H

def _proxy_agg_dlogl(x, tvds, fn0s, layout, percircuit_budget_deriv, two_dlogl_threshold):
    percircuit_budgets = _np.dot(percircuit_budget_deriv, x)
    num_circuits = len(layout.circuits)
    a = 4; b = 2  # fit params: must be same in all proxy fns

    f = 0
    for i in range(num_circuits):
        fn0 = fn0s[i]; tvd = tvds[i]; x = percircuit_budgets[i]
        f += (fn0 / _np.exp(a)) * _np.exp(a - b * (x / tvd)**2 - _np.sqrt(2 * b) * (x / tvd))
    return f - two_dlogl_threshold


def _proxy_agg_dlogl_deriv(x, tvds, fn0s, layout, percircuit_budget_deriv):
    num_circuits = len(layout.circuits)
    percircuit_budgets = _np.dot(percircuit_budget_deriv, x)
    a = 4; b = 2  # fit params: must be same in all proxy fns

    agg_dlogl_deriv_wrt_percircuit_budgets = _np.zeros(num_circuits, 'd')
    for i in range(num_circuits):
        fn0 = fn0s[i]; tvd = tvds[i]; x = percircuit_budgets[i]
        agg_dlogl_deriv_wrt_percircuit_budgets[i] = \
            (fn0 / _np.exp(a)) * _np.exp(a - b * (x / tvd)**2
                                         - _np.sqrt(2 * b) * (x / tvd)) * (-2 * b * x / tvd**2
                                                                           - _np.sqrt(2 * b) / tvd)
    assert(_np.all(agg_dlogl_deriv_wrt_percircuit_budgets <= 0)), \
        "Derivative of aggregate LLR wrt any circuit budget should be negative"
    return _np.dot(agg_dlogl_deriv_wrt_percircuit_budgets, percircuit_budget_deriv)


def _proxy_agg_dlogl_hessian(x, tvds, fn0s, layout, percircuit_budget_deriv):
    num_circuits = len(layout.circuits)
    percircuit_budgets = _np.dot(percircuit_budget_deriv, x)
    a = 4; b = 2  # fit params: must be same in all proxy fns

    agg_dlogl_hessian_wrt_percircuit_budgets = _np.zeros(num_circuits)
    for i in range(num_circuits):
        fn0 = fn0s[i]; tvd = tvds[i]; x = percircuit_budgets[i]
        agg_dlogl_hessian_wrt_percircuit_budgets[i] = \
            (fn0 / _np.exp(a)) * _np.exp(a - b * (x / tvd)**2 - _np.sqrt(2 * b) * (x / tvd)) * (
                (-2 * b * x / tvd**2 - _np.sqrt(2 * b) / tvd)**2 - 2 * b / tvd**2)
    assert(_np.all(agg_dlogl_hessian_wrt_percircuit_budgets >= -1e-8)), \
        "Hessian of aggregate LLR wrt any circuit budget should be positive"
    H = _np.dot(percircuit_budget_deriv.T,
                _np.dot(_np.diag(agg_dlogl_hessian_wrt_percircuit_budgets),
                        percircuit_budget_deriv))   # (nW, nC)(nC)(nC, nW)
    evals = _np.linalg.eigvals(H)
    assert(_np.all(evals >= -1e-8))
    return H


def optimize_wildcard_budget_cvxopt(budget, L1weights, objfn, layout, two_dlogl_threshold, redbox_threshold):
    #Use cvxopt
    import cvxopt as _cvxopt
    # Minimize f_0(wv) = |wv|_1 (perhaps weighted) subject to the constraints:
    #  dot(percircuit_budget_deriv, wv) >= critical_percircuit_budgets
    #  2 * aggregate_dlogl <= two_dlogl_threshold  => f_1(wv) = 2 * aggregate_dlogl(wv) - threshold <= 0

    wv = budget.to_vector().copy()
    n = len(wv)
    x0 = wv.reshape((n, 1))  # TODO - better guess?

    initial_probs = objfn.probs.copy()
    current_probs = initial_probs.copy()
    percircuit_budget_deriv = budget.precompute_for_same_circuits(layout.circuits)
    critical_percircuit_budgets = _get_critical_circuit_budgets(objfn, layout, redbox_threshold)
    critical_percircuit_budgets.shape = (len(critical_percircuit_budgets), 1)

    _cvxopt.solvers.options['abstol'] = 1e-5
    _cvxopt.solvers.options['reltol'] = 1e-5
    _cvxopt.solvers.options['maxiters'] = 50

    def F(x=None, z=None, debug=True):
        if z is None and x is None:
            # (m, x0) where m is number of nonlinear constraints and x0 is in domain of f
            return (1, _cvxopt.matrix(x0))

        if min(x) < 0.0:
            return None  # don't allow negative wildcard vector components

        budget.from_vector(_np.array(x))
        budget.update_probs(initial_probs, current_probs, objfn.freqs, layout, percircuit_budget_deriv)

        #Evaluate F(x) => return (f, Df)
        f = _cvxopt.matrix(_np.array([_agg_dlogl(current_probs, objfn, two_dlogl_threshold)]).reshape((1, 1)))  # shape (m,1)
        Df = _cvxopt.matrix(_np.empty((1, n), 'd'))  # shape (m, n)
        Df[0, :] = _agg_dlogl_deriv(current_probs, objfn, layout, percircuit_budget_deriv)
        #print("DB: rank Df=", _np.linalg.matrix_rank(Df))  # REMOVE

        if z is None:
            #print("DB wvec = ", ",".join(["%.3g" % vv for vv in x]), "=> %g" % f[0], ["%g" % vv for vv in Df])
            #if debug: check_fd(x, False)
            return f, Df

        # additionally, compute H = z_0 * Hessian(f_0)(wv)
        H = _cvxopt.matrix(z[0] * _agg_dlogl_hessian(current_probs, objfn, layout, percircuit_budget_deriv))
        #DEBUG REMOVE
        #print("rank Hf=", _np.linalg.matrix_rank(H), " z[0]=",z[0])
        #print(H)
        #print(_np.linalg.eigvals(H))
        evals = _np.linalg.eigvals(H)
        assert(_np.all(evals >= -1e-8))
        #print("DB wvec = ", ",".join(["%.3g" % vv for vv in x]), "=> f=%g" % f[0])
        #print("  Df = ",["%g" % vv for vv in Df])
        #print("  evals(H)= ", ["%g" % vv for vv in evals], " z=",z[0])
        #if debug: check_fd(x, True)
        return f, Df, H

    #check_fd([0.0001] * n, True)

    #CVXOPT
    print("Beginning cvxopt.cpl solve...")
    c = _cvxopt.matrix(L1weights.reshape((n, 1)))
    G = -_cvxopt.matrix(_np.concatenate((percircuit_budget_deriv, _np.identity(n, 'd')), axis=0))
    h = -_cvxopt.matrix(_np.concatenate((critical_percircuit_budgets, _np.zeros((n, 1), 'd')), axis=0))
    #result = _cvxopt.solvers.cpl(c, F)  # kktsolver='ldl2'
    result = _cvxopt.solvers.cpl(c, F, G, h)  # kktsolver='ldl2'

    #This didn't seem to help much:
    #print("Attempting restart...")
    #x0[:,0] = list(result['x'])
    #result = _cvxopt.solvers.cpl(c, F) # kktsolver='ldl2'

    print("CVXOPT result = ", result)
    print("x = ", list(result['x']))
    print("y = ", list(result['y']))
    print("znl = ", list(result['znl']))
    print("snl = ", list(result['snl']))
    budget.from_vector(result['x'])
    return


def optimize_wildcard_budget_cvxopt_SMALL(budget, L1weights, objfn, layout, two_dlogl_threshold, redbox_threshold,
                                          SMALL=1e-6):
    #Use cvxopt
    import cvxopt as _cvxopt
    # Minimize f_0(wv) = |wv|_1 (perhaps weighted) subject to the constraints:
    #  dot(percircuit_budget_deriv, wv) >= critical_percircuit_budgets
    #  2 * aggregate_dlogl <= two_dlogl_threshold  => f_1(wv) = 2 * aggregate_dlogl(wv) - threshold <= 0

    wv = budget.to_vector().copy()
    n = len(wv)
    x0 = wv.reshape((n, 1))
    c = L1weights.reshape((n, 1))
    SMALL2 = SMALL**2

    initial_probs = objfn.probs.copy()
    current_probs = initial_probs.copy()
    percircuit_budget_deriv = budget.precompute_for_same_circuits(layout.circuits)
    critical_percircuit_budgets = _get_critical_circuit_budgets(objfn, layout, redbox_threshold)
    critical_percircuit_budgets.shape = (len(critical_percircuit_budgets), 1)
    assert(_np.all(critical_percircuit_budgets >= 0))
    assert(_np.all(percircuit_budget_deriv >= 0))

    _cvxopt.solvers.options['abstol'] = 1e-5
    _cvxopt.solvers.options['reltol'] = 1e-5
    _cvxopt.solvers.options['maxiters'] = 50

    def F(x=None, z=None):
        if z is None and x is None:
            # (m, x0) where m is number of nonlinear constraints and x0 is in domain of f
            return (1, _cvxopt.matrix(x0))

        if min(x) < 0.0:
            return None  # don't allow negative wildcard vector components

        budget.from_vector(x)
        budget.update_probs(initial_probs, current_probs, objfn.freqs, layout, percircuit_budget_deriv)

        #Evaluate F(x) => return (f, Df)
        sqrtVec = _np.sqrt((c * x)**2 + SMALL2)
        f = _cvxopt.matrix(_np.array([float(_np.sum(sqrtVec)),
                                      _agg_dlogl(current_probs, objfn,
                                                 two_dlogl_threshold)]).reshape((2,1)))  # shape (m+1,1)

        L1term_grad = c if SMALL2 == 0.0 else c**2 * x / sqrtVec
        Df = _cvxopt.matrix(_np.empty((2, n), 'd'))  # shape (m+1, n)
        Df[0, :] = L1term_grad[:, 0]
        Df[1, :] = _agg_dlogl_deriv(current_probs, objfn, layout, percircuit_budget_deriv)
        #print("rank Df=", _np.linalg.matrix_rank(Df))
        if z is None:
            return f, Df

        # additionally, compute H = z_0 * Hessian(f_0)(wv) + z_1 * Hessian(f_1)(wv)
        L1_term_hess = _np.zeros((n, n), 'd') if SMALL2 == 0.0 else \
            _np.diag(-1.0 / (sqrtVec**3) * (c**2 * x)**2 + c**2 / sqrtVec)
        Hf = _cvxopt.matrix(z[0] * L1_term_hess + z[1] * _agg_dlogl_hessian(current_probs, objfn,
                                                                            layout, percircuit_budget_deriv))
        #print("rank Hf=", _np.linalg.matrix_rank(Hf), " z[1]=",z[1])
        return f, Df, Hf

    #CVXOPT
    print("Beginning cvxopt.cp solve...")
    #print("Rank G = ",_np.linalg.matrix_rank(percircuit_budget_deriv))
    #result = _cvxopt.solvers.cp(F)
    # Condition is Gx <= h => -Gx >= -h
    G = -_cvxopt.matrix(_np.concatenate((percircuit_budget_deriv, _np.identity(n, 'd')), axis=0))
    h = -_cvxopt.matrix(_np.concatenate((critical_percircuit_budgets, _np.zeros((n, 1), 'd')), axis=0))
    result = _cvxopt.solvers.cp(F, G, h)
                                

    #This didn't seem to help much:
    #print("Attempting restart...")
    #x0[:,0] = list(result['x'])
    #result = _cvxopt.solvers.cpl(c, F) # kktsolver='ldl2'

    print("CVXOPT result = ", result)
    print("x = ", list(result['x']))
    print("y = ", list(result['y']))
    print("znl = ", list(result['znl']))
    print("snl = ", list(result['snl']))
    budget.from_vector(result['x'])
    return  


def optimize_wildcard_budget_barrier(budget, L1weights, objfn, layout, two_dlogl_threshold,
                                     redbox_threshold, tol=1e-7, max_iters=50, save_debugplot_data=False):
    #BARRIER method:
    # Solve:            min c^T * x
    # Subject to:       F(x) <= 0
    # by actually solving (via Newton):
    #  min t * c^T * x + phi(x)
    # where phi(x) = -log(-F(x))
    # for increasing values of t until 1/t <= epsilon (precision tolerance)
    print("Beginning custom barrier method solve...")
    num_circuits = len(layout.circuits)
    percircuit_budget_deriv = budget.precompute_for_same_circuits(layout.circuits)
    critical_percircuit_budgets = _get_critical_circuit_budgets(objfn, layout, redbox_threshold)

    x0 = budget.to_vector()
    initial_probs = objfn.probs.copy()
    current_probs = initial_probs.copy()
    n = len(x0)

    # f0 = 2DLogL - threshold <= 0
    # fi = critical_budget_i - circuit_budget_i <= 0
    #    = critical_percircuit_budgets - dot(percircuit_budget_deriv, x) <= 0
    # fj = -x_j <= 0

    def penalty_vec(x):
        budget.from_vector(x)
        budget.update_probs(initial_probs, current_probs, objfn.freqs, layout, percircuit_budget_deriv)
        f0 = _np.array([_agg_dlogl(current_probs, objfn, two_dlogl_threshold)])
        fi = critical_percircuit_budgets - _np.dot(percircuit_budget_deriv, x)
        return _np.concatenate((f0, fi))

    def barrierF(x, compute_deriv=True):
        assert(min(x) >= 0)  # don't allow negative wildcard vector components

        budget.from_vector(_np.array(x))
        budget.update_probs(initial_probs, current_probs, objfn.freqs, layout, percircuit_budget_deriv)
        f0 = _np.array([_agg_dlogl(current_probs, objfn, two_dlogl_threshold)])
        fi = critical_percircuit_budgets - _np.dot(percircuit_budget_deriv, x)
        f = _np.concatenate((f0, fi, -x))  # adds -x for x >= 0 constraint
        val = -_np.sum(_np.log(-f))
        if not compute_deriv: return val

        Df0 = _agg_dlogl_deriv(current_probs, objfn, layout, percircuit_budget_deriv)
        deriv = -1 / f0 * Df0 - _np.dot(1 / fi, percircuit_budget_deriv) - 1 / x

        Hf0 = _agg_dlogl_hessian(current_probs, objfn, layout, percircuit_budget_deriv)
        hess =  1 / f0**2 * Df0[:, None] * Df0[None, :] - 1 / f0 * Hf0 \
            + _np.einsum('i,ij,ik->jk', 1 / fi**2, percircuit_budget_deriv, percircuit_budget_deriv) \
            + _np.diag(1 / x**2)
        # sum_i 1 / fi[i]**2 * percircuit_budget_deriv[i,:,None] * percircuit_budget_deriv[i,None,:]
        # (i,) (i,j) (i,k)
        return val, deriv, hess

    #Find a valid initial point
    initial_penalty_vec = penalty_vec(x0)
    num_constraints = len(initial_penalty_vec) + len(x0)  # 2nd term b/c x >= 0 constraints
    if _np.all(initial_penalty_vec <= 0):
        print("Initial (feasible) point: ", x0)
    else:
        if _np.linalg.norm(x0) < 1e-5: x0[:] = 1e-5  # just so we don't start at all zeros
        i = 0
        while i < 100:
            if _np.all(penalty_vec(x0) <= 0): break
            x0 *= 2.0; i += 1
        else:
            raise ValueError("Could not find feasible starting point!")
        print("Found initial feasible point: ", x0)
    x = x0.copy()  # set initial point

    nSteps = 3
    log10_end = int(_np.ceil(_np.log10(2 * num_constraints / tol)))  # 2 factor just for good measure
    log10_begin = log10_end - (nSteps-1)
    t_values = _np.logspace(log10_begin, log10_end, nSteps)
    #t_values = [1e5, 1e6, 1.e7, 1e8, 1e9]
    
    SMALL_values = [0]*len(t_values)
    #SMALL_values = [1 / (10 * t) for t in t_values]

    if save_debugplot_data:
        with open("debug/num_stages", 'wb') as pipe:
            _pickle.dump(len(t_values), pipe)

    for iStage, (t, SMALL) in enumerate(zip(t_values, SMALL_values)):  # while 1/t > epsilon:

        print("*** Beginning stage %d with t=%g, SMALL=%g ***" % (iStage, t, SMALL))
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

        #import scipy.optimize
        #def barrier_obj(x):
        #    x = _np.clip(x, 1e-10, None)
        #    return t * _np.dot(c.T, x) - _np.log(-barrierF(x, False))
        #result = scipy.optimize.minimize(barrier_obj, x, method="CG")
        #x = _np.clip(result.x, 0, None)

        x, debug_x_list = NewtonSolve(x, NewtonObjective, NewtonObjective_derivs, tol, max_iters, debug=True)
        #x, debug_x_list = NewtonSolve(x, NewtonObjective, None, tol, max_iters, debug=True)  # use finite-diff derivs

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
                w0 = _np.linspace(min(w0_list)*0.9, max(w0_list)*1.1, 50)
                w1 = _np.linspace(min(w1_list)*0.9, max(w1_list)*1.1, 50)
    
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

    print("Finished! Final x = ", x)
    budget.from_vector(x)
    return


def NewtonSolve(initial_x, fn, fn_with_derivs=None, dx_tol=1e-6, max_iters=20, lmbda=0.0, debug=False):
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

        evalsH, eigvecsH = _np.linalg.eig(Hobj)
        assert(_np.all(evalsH >= -1e-8))
        #print(" evalsH = ",evalsH)

        norm_Dobj = _np.linalg.norm(Dobj)
        #dx = - _np.dot(_np.linalg.inv(H), Df.T)
        Hrank = _np.linalg.matrix_rank(Hobj)
        if Hrank < Hobj.shape[0]:
            if debug: print("Rank defficient Hessian (%d < %d) - using gradient step" % (Hrank, Hobj.shape[0]))
            dx = - Dobj / _np.linalg.norm(Dobj)
        else:
            dx = - _np.dot((1 - lmbda) * _np.linalg.inv(Hobj) + lmbda * I, Dobj)

        #if debug and i == 0:
        #    print(" initial newton iter: f=%g, |Df|=%g, |Hf|=%g" % (obj, norm_Dobj, _np.linalg.norm(Hobj)))
        #    print(" dx = ",dx)
        if test_obj is not None:
            assert(_np.isclose(obj, test_obj))  # Sanity check

        downhill_direction = - Dobj / _np.linalg.norm(Dobj)
        dx_before_backtrack = dx.copy()

        #print("DB: last obj = ",obj)
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

            if debug: print("Can't step in Newton direction and reduce objective - converged at f=%g" % obj)
            break

        norm_dx = _np.linalg.norm(dx)
        if debug:
            print(" newton iter %d: f=%g, |Df|=%g, |dx|=%g |Hf|=%g" %
                  (i, obj, norm_Dobj, norm_dx, _np.linalg.norm(Hobj)))
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
        if norm_Dobj < 1e-4 or norm_dx < dx_tol: break
    if i == max_iters:
        print("WARNING: max iterations exceeded!!!")
    return x, x_list


def optimize_wildcard_budget_cvxopt_smoothed(budget, L1weights, objfn, layout, two_dlogl_threshold, redbox_threshold):
    import cvxopt as _cvxopt
    wv = budget.to_vector().copy()
    n = len(wv)
    x0 = wv.reshape((n, 1))  # TODO - better guess?

    #initial_probs = objfn.probs.copy()
    #current_probs = initial_probs.copy()
    percircuit_budget_deriv = budget.precompute_for_same_circuits(layout.circuits)
    critical_percircuit_budgets = _get_critical_circuit_budgets(objfn, layout, redbox_threshold)
    critical_percircuit_budgets.shape = (len(critical_percircuit_budgets), 1)
    num_circuits = len(layout.circuits)

    _cvxopt.solvers.options['abstol'] = 1e-5
    _cvxopt.solvers.options['reltol'] = 1e-5
    _cvxopt.solvers.options['maxiters'] = 50

    #Prepare for proxy_barrierF evaluations
    tvds = _np.zeros(num_circuits, 'd')
    fn0s = _np.zeros(num_circuits, 'd')
    for i in range(num_circuits):
        p = objfn.probs[layout.indices_for_index(i)]
        f = objfn.freqs[layout.indices_for_index(i)]
        N = objfn.total_counts[layout.indices_for_index(i)]
        dlogl_elements = N * f * _np.log(f / p)
        fn0s[i] = 2 * _np.sum(dlogl_elements)
        tvds[i] = 0.5 * _np.sum(_np.abs(p - f))
        #return tvds, fn0s

    def F(x=None, z=None, debug=True):
        if z is None and x is None:
            # (m, x0) where m is number of nonlinear constraints and x0 is in domain of f
            return (1, _cvxopt.matrix(x0))

        if min(x) < 0.0:
            return None  # don't allow negative wildcard vector components

        #budget.from_vector(_np.array(x))
        #budget.update_probs(initial_probs, current_probs, objfn.freqs, layout, percircuit_budget_deriv)

        #Evaluate F(x) => return (f, Df)
        f = _cvxopt.matrix(_np.array([_proxy_agg_dlogl(x, tvds, fn0s, layout, percircuit_budget_deriv,
                                                       two_dlogl_threshold)]).reshape((1, 1)))  # shape (m,1)
        Df = _cvxopt.matrix(_np.empty((1, n), 'd'))  # shape (m, n)
        Df[0, :] = _proxy_agg_dlogl_deriv(x, tvds, fn0s, layout, percircuit_budget_deriv)

        if z is None:
            return f, Df

        # additionally, compute H = z_0 * Hessian(f_0)(wv)
        H = _cvxopt.matrix(z[0] * _proxy_agg_dlogl_hessian(x, tvds, fn0s, layout, percircuit_budget_deriv))
        evals = _np.linalg.eigvals(H)
        assert(_np.all(evals >= -1e-8))
        return f, Df, H

    print("Beginning cvxopt.cpl solve with smoothed (proxy) fn...")
    c = _cvxopt.matrix(L1weights.reshape((n, 1)))
    G = -_cvxopt.matrix(_np.concatenate((percircuit_budget_deriv, _np.identity(n, 'd')), axis=0))
    h = -_cvxopt.matrix(_np.concatenate((critical_percircuit_budgets, _np.zeros((n, 1), 'd')), axis=0))
    result = _cvxopt.solvers.cpl(c, F, G, h)  # kktsolver='ldl2'

    print("CVXOPT result = ", result)
    print("x = ", list(result['x']))
    print("y = ", list(result['y']))
    print("znl = ", list(result['znl']))
    print("snl = ", list(result['snl']))
    budget.from_vector(result['x'])
    return


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


#DEBUG: check with finite diff derivatives:
#def _check_fd(wv_base, chk_hessian=False):
#    wv_base = _np.array(wv_base, 'd')  # [0.0001]*3
#    wv_len = len(wv_base)
#    grad = _np.zeros(wv_len, 'd')
#    f0, grad_chk = F(wv_base, debug=False)
#    eps = 1e-7
#    for k in range(len(wv_base)):
#        wv_eps = wv_base.copy(); wv_eps[k] += eps
#        f_eps, _ = F(wv_eps, debug=False)
#        grad[k] = (f_eps[0] - f0[0]) / eps
#    rel_diff_norm = _np.linalg.norm(grad - grad_chk) / _np.linalg.norm(grad)
#    #print("GRAD CHECK:")
#    #print(grad)
#    #print(grad_chk)
#    #print("  diff = ",grad - grad_chk, " rel_diff_norm=", rel_diff_norm)
#    print("GRAD CHK ", rel_diff_norm)
#    assert(rel_diff_norm < 1e-3)
#    if chk_hessian is False: return
#
#    hess = _np.zeros((wv_len, wv_len), 'd')
#    f0, _, H_chk = F(wv_base, [1.0], debug=False)
#    eps = 1e-7
#    for k in range(wv_len):
#        wv_eps_k = wv_base.copy(); wv_eps_k[k] += eps
#        f_eps_k, _ = F(wv_eps_k, debug=False)
#        for l in range(wv_len):
#            wv_eps_l = wv_base.copy(); wv_eps_l[l] += eps
#            f_eps_l, _ = F(wv_eps_l, debug=False)
#            wv_eps_kl = wv_eps_k.copy(); wv_eps_kl[l] += eps
#            f_eps_kl, _ = F(wv_eps_kl, debug=False)
#            hess[k, l] = (f_eps_kl[0] - f_eps_k[0] - f_eps_l[0] + f0[0]) / eps**2
#    rel_diff_norm = _np.linalg.norm(hess - H_chk) / _np.linalg.norm(hess)
#    #print("HESSIAN CHECK:")
#    #print(hess)
#    #print(H_chk)
#    #print("  diff = ",hess - H_chk, " rel_diff_norm=", rel_diff_norm)
#    print("HESS CHK ", rel_diff_norm)
#    #assert(rel_diff_norm < 5e-2)


#UNUSED?
#def _wildcard_objective_firstterms(current_probs):
#    dlogl_elements = objfn.raw_objfn.terms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
#    for i in range(num_circuits):
#        dlogl_percircuit[i] = _np.sum(dlogl_elements[layout.indices_for_index(i)], axis=0)
#
#    two_dlogl_percircuit = 2 * dlogl_percircuit
#    two_dlogl = sum(two_dlogl_percircuit)
#    return max(0, two_dlogl - two_dlogl_threshold) \
#        + sum(_np.clip(two_dlogl_percircuit - redbox_threshold, 0, None))
#
#def _advance_probs(layout, current_probs, dlogl_percircuit, dlogl_delements, delta_percircuit_budgets):
#    num_circuits = len(layout.circuits)
#    delta_probs = _np.zeros(len(current_probs), 'd')
#    for i in range(num_circuits):
#        #if 2 * dlogl_percircuit[i] <= redbox_threshold and global_criteria_met: continue
#
#        step = delta_percircuit_budgets[i]
#        #p = current_probs[layout.indices_for_index(i)]
#        chis = dlogl_delements[layout.indices_for_index(i)]
#        maxes = _np.array(_np.abs(chis - _np.max(chis)) < 1.e-4, dtype=int)
#        mins = _np.array(_np.abs(chis - _np.min(chis)) < 1.e-4, dtype=int)
#        add_to = step * mins / sum(mins)
#        take_from = step * maxes / sum(maxes)
#        delta_probs[layout.indices_for_index(i)] = add_to - take_from
#    return delta_probs
#
#
#def wildcard_probs_propagation(budget, initial_wv, final_wv, objfn, layout, num_steps=10):
#    #Begin with a zero budget
#    current_probs = objfn.probs.copy()
#
#    percircuit_budget_deriv = budget.precompute_for_same_circuits(layout.circuits)
#    dlogl_percircuit = objfn.percircuit()
#
#    num_circuits = len(layout.circuits)
#    assert(len(dlogl_percircuit) == num_circuits)
#
#    delta_wv = (final_wv - initial_wv) / num_steps
#    wv = initial_wv.copy()
#    for i in range(nSteps):
#        wv += delta_wv
#        dlogl_elements = objfn.raw_objfn.terms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
#        for i in range(num_circuits):
#            dlogl_percircuit[i] = _np.sum(dlogl_elements[layout.indices_for_index(i)], axis=0)
#        dlogl_delements = objfn.raw_objfn.dterms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
#
#        two_dlogl = sum(2 * dlogl_percircuit)
#        perbox_residual = sum(_np.clip(2 * dlogl_percircuit - redbox_threshold, 0, None))
#        print("Advance: global=", two_dlogl - two_dlogl_threshold, " percircuit=", perbox_residual)
#        print("  wv=", wv)
#
#        delta_percircuit_budgets = _np.dot(percircuit_budget_deriv, delta_wv)
#        delta_probs = _advance_probs(layout, current_probs, dlogl_percircuit, dlogl_delements, delta_percircuit_budgets)  # updates current_probs
#        print("|delta probs| = ", _np.linalg.norm(delta_probs))
#        current_probs += delta_probs
#    return currrent_probs
#def wildcard_opt_by_propagation()  #TODO
#    # Time-evolution approach:  Walk downhill in steps until constraints ("firstterms") are satisfied
#    #wv = budget.to_vector().copy()
#
#    def _criteria_deriv(current_probs, dlogl_percircuit, dlogl_delements, mode, global_criteria_met):
#        # derivative of firstterms wrt per-circuit wilcard budgets - namely if that budget goes up how to most efficiently reduce firstterms
#        # in doing so, this computes how the per-circuit budget should be allocated to probabilities (i.e. how probs should be updated) to achieve this decrease in firstterms
#        ret = _np.zeros(num_circuits)
#        max_delta = _np.zeros(num_circuits)  # maximum amount of change in per-circuit budget before hitting a discontinuity in 2nd deriv
#        for i in range(num_circuits):
#            if mode == "percircuit" and 2 * dlogl_percircuit[i] <= redbox_threshold:
#                continue  # don't include this circuit's contribution
#            elif mode == "aggregate":  # all circuits contribute
#                prefactor = 1.0
#            else:  # mode == "both"
#                prefactor = 2.0  # contributes twice: once for per-circuit and once for aggregate
#                if 2 * dlogl_percircuit[i] <= redbox_threshold:
#                    if global_criteria_met: continue  # no contribution at all_circuits_needing_data
#                    else: prefactor = 1.0
#
#            chis = dlogl_delements[layout.indices_for_index(i)]  # ~ f/p  (deriv of f*log(p))
#            highest_chi, lowest_chi = _np.max(chis), _np.min(chis)
#            bmaxes = _np.array(_np.abs(chis - highest_chi) < 1.e-4, dtype=bool)
#            bmins = _np.array(_np.abs(chis - lowest_chi) < 1.e-4, dtype=bool)
#            maxes = _np.array(_np.abs(chis - _np.max(chis)) < 1.e-4, dtype=int)
#            mins = _np.array(_np.abs(chis - _np.min(chis)) < 1.e-4, dtype=int)
#
#            next_chis = chis.copy(); next_chis[bmaxes] = 1.0; next_chis[bmins] = 1.0
#            #p = current_probs[layout.indices_for_index(i)]
#            f = objfn.freqs[layout.indices_for_index(i)]
#            next_highest_chi = _np.max(next_chis)  # 2nd highest chi value (may be duplicated)
#            next_lowest_chi = _np.min(next_chis)  # 2nd lowest chi value (may be duplicated)
#
#            # 1/chi = p/f, (1/chi'-1/chi) = dp/f => dp = f(chi - chi')/(chi chi')
#            delta_p = _np.zeros(chis.shape, 'd')
#            delta_p[bmaxes] = f[bmaxes] * (1. / chis[bmaxes] - 1 / next_highest_chi)
#            delta_p[bmins] = f[bmins] * (1. / chis[bmins] - 1 / next_lowest_chi)
#            max_delta[i] = _np.max(_np.abs(delta_p))
#
#            ret[i] = prefactor * _np.sum(chis * (mins / sum(mins) - maxes / sum(maxes)))
#        return ret, max_delta
#
#
#    for mode in (): #("both",): #("percircuit", "aggregate"):  # choose how many and which criteria to enforce on each pass.
#        print("Stage w/mode = ",mode)
#        step = 0.01
#        itr = 0
#        L1grad = L1weights
#        imax = None
#        last_objfn_value = None; last_probs = None  # DEBUG
#        last_dlogl_percircuit = last_dlogl_elements = None # DEBUG
#        while True:
#
#            #Compute current log-likelihood values and derivates wrt probabilities
#            dlogl_elements = objfn.raw_objfn.terms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
#            for i in range(num_circuits):
#                dlogl_percircuit[i] = _np.sum(dlogl_elements[layout.indices_for_index(i)], axis=0)
#            dlogl_delements = objfn.raw_objfn.dterms(current_probs, objfn.counts, objfn.total_counts, objfn.freqs)
#            two_dlogl_percircuit = 2 * dlogl_percircuit
#            two_dlogl = sum(two_dlogl_percircuit)
#            global_criteria_met = two_dlogl < two_dlogl_threshold
#
#            # check aggregate and per-circuit criteria - exit if met
#            if mode == "aggregate":
#                objfn_value = max(two_dlogl - two_dlogl_threshold, 0)
#            elif mode == "percircuit":
#                perbox_residual = sum(_np.clip(two_dlogl_percircuit - redbox_threshold, 0, None))
#                objfn_value = perbox_residual
#            elif mode == "both":
#                objfn_value = max(two_dlogl - two_dlogl_threshold, 0) + sum(_np.clip(two_dlogl_percircuit - redbox_threshold, 0, None))
#
#            print("Iter ", itr, ": mode=", mode, " objfn=", objfn_value, " moved in", imax)
#            print("  wv=", wv); itr += 1
#            if objfn_value < 1e-10: # if global_criteria_met and perbox_residual < 1e-10:
#                break  # DONE!
#            if last_objfn_value is not None and last_objfn_value < objfn_value:
#                iproblem = _np.argmax(dlogl_percircuit - last_dlogl_percircuit)
#                print("Circuit  ",iproblem," dlogl=", last_dlogl_percircuit[iproblem], " => ", dlogl_percircuit[iproblem])
#                print("  probs: ",last_probs[layout.indices_for_index(iproblem)], " => ", current_probs[layout.indices_for_index(iproblem)])
#                print("  freqs: ",objfn.freqs[layout.indices_for_index(iproblem)])
#                import bpdb; bpdb.set_trace()
#                assert(False), "Objective function should be monotonic!!!"
#            last_objfn_value = objfn_value
#            last_probs = current_probs.copy()
#            last_dlogl_percircuit = dlogl_percircuit.copy()
#            last_dlogl_elements = dlogl_elements.copy()
#
#            #import bpdb; bpdb.set_trace()
#            criteria_deriv_wrt_percircuit_budgets, maximum_percircuit_budget_delta = \
#                _criteria_deriv(current_probs, dlogl_percircuit, dlogl_delements, mode, global_criteria_met)
#            wv_grad = _np.dot(criteria_deriv_wrt_percircuit_budgets, percircuit_budget_deriv) #+ L1grad
#            grad_norm = _np.linalg.norm(wv_grad)
#            assert(grad_norm > 1e-6), \
#                "Gradient norm == 0! - cannot reduce constraint residuals with more wildcard!"
#
#            imax = _np.argmax(_np.abs(wv_grad / L1grad)); sgn = _np.sign(wv_grad[imax])
#            wv_grad[:] = 0; wv_grad[imax] = sgn
#            downhill_direction = (-wv_grad / _np.linalg.norm(wv_grad))
#
#            #Constant step:
#            #step = 1e-5
#            # Variable step: expected reduction = df/dw * dw, so set |dw| = 0.01 * current_f / |df/dw|
#            #step = (0.01 * objfn_value / grad_norm)
#
#            #Step based on next discontinuity ("breakpoint")
#            # require _np.dot(percircuit_budget_deriv, step * downhill_direction) < maximum_percircuit_budget_delta
#            step = _np.min(maximum_percircuit_budget_delta / _np.dot(percircuit_budget_deriv, downhill_direction))
#            assert(step > 0)
#            step = min(step, 1e-5)  # don't allow too large of a step...
#
#            delta_wv = downhill_direction * step
#            wv += delta_wv
#
#            delta_percircuit_budgets = _np.dot(percircuit_budget_deriv, delta_wv)
#            #assert(_np.all(delta_percircuit_budgets >= 0))
#            if not _np.all(delta_percircuit_budgets >= 0):
#                import bpdb; bpdb.set_trace()
#                pass
#
#            delta_probs = _advance_probs(layout, current_probs, dlogl_percircuit, dlogl_delements, delta_percircuit_budgets)  #, global_criteria_met)  # updates current_probs
#            print("|delta probs| = ", _np.linalg.norm(delta_probs))
#            current_probs += delta_probs
#
#    #assert(False), "STOP"
#    wv_new = wv
#    print("NEW TEST - final wildcard is ", wv_new)
#
#This didn't work well:
##Experiment with "soft" min and max functions to see if that fixes cvxopt getting stuck
## so far, this hasn't helped.
#
#def _softmax(ar):
#    return _np.log(_np.sum([_np.exp(x) for x in ar]))
#
#def _softmin(ar):
#    return -_np.log(_np.sum([_np.exp(-x) for x in ar]))
