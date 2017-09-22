from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" GST gauge optimization algorithms """

import numpy as _np
import warnings as _warnings

from .. import objects as _objs
from .. import tools as _tools
from .. import optimize as _opt
from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct


from ..objects import Basis


def gaugeopt_to_target(gateset, targetGateset, itemWeights=None,
                       cptp_penalty_factor=0, spam_penalty_factor=0,
                       gatesMetric="frobenius", spamMetric="frobenius",
                       gauge_group=None, method='auto', maxiter=100000,
                       maxfev=None, tol=1e-8, returnAll=False, comm=None,
                       verbosity=0, checkJac=False):
    """
    Optimize the gauge degrees of freedom of a gateset to that of a target.

    Parameters
    ----------
    gateset : GateSet
        The gateset to gauge-optimize

    targetGateset : GateSet
        The gateset to optimize to.  The metric used for comparing gatesets
        is given by `gatesMetric` and `spamMetric`.

    itemWeights : dict, optional
        Dictionary of weighting factors for gates and spam operators.  Keys can
        be gate, state preparation, or POVM effect, as well as the special values
        "spam" or "gates" which apply the given weighting to *all* spam operators
        or gates respectively.  Values are floating point numbers.  Values given 
        for specific gates or spam operators take precedence over "gates" and
        "spam" values.  The precise use of these weights depends on the gateset
        metric(s) being used.
       
    cptp_penalty_factor : float, optional
        If greater than zero, the objective function also contains CPTP penalty
        terms which penalize non-CPTP-ness of the gates being optimized.  This factor
        multiplies these CPTP penalty terms.

    spam_penalty_factor : float, optional
        If greater than zero, the objective function also contains SPAM penalty
        terms which penalize non-positive-ness of the state preps being optimized.  This
        factor multiplies these SPAM penalty terms.
        
    gatesMetric : {"frobenius", "fidelity", "tracedist"}, optional
        The metric used to compare gates within gate sets. "frobenius" computes 
        the normalized sqrt(sum-of-squared-differences), with weights
        multiplying the squared differences (see :func:`GateSet.frobeniusdist`).
        "fidelity" and "tracedist" sum the individual infidelities or trace
        distances of each gate, weighted by the weights.

    spamMetric : {"frobenius", "fidelity", "tracedist"}, optional
        The metric used to compare spam vectors within gate sets. "frobenius"
        computes the normalized sqrt(sum-of-squared-differences), with weights
        multiplying the squared differences (see :func:`GateSet.frobeniusdist`).
        "fidelity" and "tracedist" sum the individual infidelities or trace
        distances of each "SPAM gate", weighted by the weights.

    gauge_group : GaugeGroup, optional
        The gauge group which defines which gauge trasformations are optimized
        over.  If None, then the `gateset`'s default gauge group is used.

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

    returnAll : bool, optional
        When True, return best "goodness" value and gauge matrix in addition to the
        gauge optimized gateset.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    verbosity : int, optional
        How much detail to send to stdout.

    checkJac : bool
        When True, check least squares analytic jacobian against finite differences

    Returns
    -------
    gateset                            if returnAll == False

    (goodnessMin, gaugeMx, gateset)    if returnAll == True

      where goodnessMin is the minimum value of the goodness function (the best 'goodness')
      found, gaugeMx is the gauge matrix used to transform the gateset, and gateset is the
      final gauge-transformed gateset.
    """
    if itemWeights is None: itemWeights = {}

    ls_mode_allowed = bool(targetGateset is not None and
                           gatesMetric == "frobenius" and
                           spamMetric  == "frobenius")
        #and gateset.dim < 64: # least squares optimization seems uneffective if more than 3 qubits
        #  -- observed by Lucas - should try to debug why 3 qubits seemed to cause trouble...
    
    if method == "ls" and not ls_mode_allowed:
       raise ValueError("Least-squares method is not allowed! Target" +
                        " gateset must be non-None and frobenius metrics" +
                        " must be used.")
    if method == "auto":
        method = 'ls' if ls_mode_allowed else 'L-BFGS-B'
        
    objective_fn, jacobian_fn = _create_objective_fn(
        gateset, targetGateset, itemWeights,
        cptp_penalty_factor, spam_penalty_factor,
        gatesMetric, spamMetric, method, comm, checkJac)

    result = gaugeopt_custom(gateset, objective_fn, gauge_group, method,
                             maxiter, maxfev, tol, returnAll, jacobian_fn,
                             comm, verbosity)

    #If we've gauge optimized to a target gate set, declare that the
    # resulting gate set is now in the same basis as the target.
    if targetGateset is not None:
        newGateset = result[-1] if returnAll else result
        newGateset.basis = targetGateset.basis.copy()
    
    return result


def gaugeopt_custom(gateset, objective_fn, gauge_group=None,
                    method='L-BFGS-B', maxiter=100000, maxfev=None, tol=1e-8,
                    returnAll=False, jacobian_fn=None, comm=None, verbosity=0):
    """
    Optimize the gauge of a gateset using a custom objective function.

    Parameters
    ----------
    gateset : GateSet
        The gateset to gauge-optimize

    objective_fn : function
        The function to be minimized.  The function must take a single `GateSet`
        argument and return a float.

    gauge_group : GaugeGroup, optional
        The gauge group which defines which gauge trasformations are optimized
        over.  If None, then the `gateset`'s default gauge group is used.

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

    returnAll : bool, optional
        When True, return best "goodness" value and gauge matrix in addition to the
        gauge optimized gateset.

    jacobian_fn : function, optional
        The jacobian of `objective_fn`.  The function must take three parameters,
        1) the un-transformed `GateSet`, 2) the transformed `GateSet`, and 3) the
        `GaugeGroup.element` representing the transformation that brings the first
        argument into the second.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    gateset                            if returnAll == False

    (goodnessMin, gaugeMx, gateset)    if returnAll == True

      where goodnessMin is the minimum value of the goodness function (the best 'goodness')
      found, gaugeMx is the gauge matrix used to transform the gateset, and gateset is the
      final gauge-transformed gateset.
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    if comm is not None:
        gs_cmp = comm.bcast(gateset if (comm.Get_rank() == 0) else None, root=0)
        if gateset.frobeniusdist(gs_cmp) > 1e-6:
            raise ValueError("MPI ERROR in gaugeopt: *different* gatesets" +
                             " given to different processors!")

    if gauge_group is None:
        gauge_group = gateset.default_gauge_group
        if gauge_group is None:
            #don't do any gauge optimization (assume trivial gauge group)
            _warnings.warn("No gauge group specified, so no gauge optimization performed.")
            if returnAll:
                return None, None, gateset.copy() 
            else: 
                return gateset.copy()

    x0 = gauge_group.get_initial_params() #gauge group picks a good initial el
    gaugeGroupEl = gauge_group.get_element(x0) #re-used element for evals

    def call_objective_fn(gaugeGroupElVec):
        gaugeGroupEl.from_vector(gaugeGroupElVec)
        gs = gateset.copy()
        gs.transform(gaugeGroupEl)
        return objective_fn(gs)

    if jacobian_fn:
        def call_jacobian_fn(gaugeGroupElVec):
            gaugeGroupEl.from_vector(gaugeGroupElVec)
            gs = gateset.copy()
            gs.transform(gaugeGroupEl)
            return jacobian_fn(gateset, gs, gaugeGroupEl)
    else:
        call_jacobian_fn = None

    
    if method == 'ls':
        #minSol  = _opt.least_squares(call_objective_fn, x0, #jac=call_jacobian_fn,
        #                            max_nfev=maxfev, ftol=tol)
        #solnX = minSol.x
        assert(call_jacobian_fn is not None), "Cannot use 'ls' method unless jacobian is available"
        solnX,converged,msg = _opt.custom_leastsq(
            call_objective_fn, call_jacobian_fn, x0, f_norm2_tol=tol,
            jac_norm_tol=tol, rel_ftol=tol, rel_xtol=tol,
            max_iter=maxiter, comm=comm,
            verbosity=printer.verbosity-2)
        printer.log("Least squares message = %s" % msg,2)
        assert(converged)
        solnF = call_objective_fn(solnX) if returnAll else None

    else:
        if comm is not None and comm.Get_rank() == 0:
            _warnings.warn("MPI comm was given for gauge optimization but can" +
                           " only be used with the least-squares method.")

        bToStdout = (printer.verbosity > 2 and printer.filename is None)
        if bToStdout and (comm is None or comm.Get_rank() == 0): 
            print_obj_func = _opt.create_obj_func_printer(call_objective_fn) #only ever prints to stdout!
            # print_obj_func(x0) #print initial point (can be a large vector though)
        else: print_obj_func = None
            
        minSol = _opt.minimize(call_objective_fn, x0,
                               method=method, maxiter=maxiter, maxfev=maxfev,
                               tol=tol, jac=call_jacobian_fn,                               
                               callback = print_obj_func)
        solnX = minSol.x
        solnF = minSol.fun

    gaugeGroupEl.from_vector(solnX)
    newGateset = gateset.copy()
    newGateset.transform(gaugeGroupEl)

    if returnAll:
        return solnF, gaugeMat, newGateset
    else:  
        return newGateset


def _create_objective_fn(gateset, targetGateset, itemWeights=None,
                         cptp_penalty_factor=0, spam_penalty_factor=0,
                         gatesMetric="frobenius", spamMetric="frobenius",
                         method="auto", comm=None, checkJac=False):
    """ 
    Creates the objective function and jacobian (if available)
    for gaugeopt_to_target 
    """
    if itemWeights is None: itemWeights = {}
    gateWeight = itemWeights.get('gates',1.0)
    spamWeight = itemWeights.get('spam',1.0)
    mxBasis = gateset.basis

    #Use the target gateset's basis if gateset's is unknown
    # (as it can often be if it's just come from an logl opt,
    #  since from_vector will clear any basis info)
    if mxBasis.name == "unknown" and targetGateset is not None: 
        mxBasis = targetGateset.basis

    if  method == "ls":
        # least-squares case where objective function returns an array of
        # the before-they're-squared difference terms and there's an analytic jacobian

        def objective_fn(gs):
            residuals, nsummands = gs.residuals(targetGateset, None, gateWeight, spamWeight, itemWeights)

            if cptp_penalty_factor > 0:
                gs.basis = mxBasis
                cpPenaltyVec = _cptp_penalty(gs,cptp_penalty_factor,gs.basis)
            else: cpPenaltyVec = [] # so concatenate ignores

            if spam_penalty_factor > 0:
                gs.basis = mxBasis
                spamPenaltyVec = _spam_penalty(gs,spam_penalty_factor,gs.basis)
            else: spamPenaltyVec = [] # so concatenate ignores

            return _np.concatenate( (residuals, cpPenaltyVec, spamPenaltyVec) )

        
        def jacobian_fn(gs_pre, gs_post, gaugeGroupEl):
            
            # Indices: Jacobian output matrix has shape (L, N)
            start = 0
            d = gs_pre.dim
            N = gaugeGroupEl.num_params()
            L = gs_pre.num_elements(include_povm_identity=True)

            #Compute "extra" (i.e. beyond the gateset-element) rows of jacobian
            if cptp_penalty_factor != 0: L += len(gs_pre.gates)
            if spam_penalty_factor != 0: L += len(gs_pre.preps)

            #Set basis for pentaly term calculation
            if cptp_penalty_factor != 0 or spam_penalty_factor != 0:
                gs_pre.basis = mxBasis
                gs_post.basis = mxBasis

            jacMx = _np.zeros((L, N))
    
            #Overview of terms:
            # objective: gate_term = (S_inv * gate * S - target_gate)
            # jac:       d(gate_term) = (d (S_inv) * gate * S + S_inv * gate * dS )
            #            d(gate_term) = (-(S_inv * dS * S_inv) * gate * S + S_inv * gate * dS )
            
            # objective: rho_term = (S_inv * rho - target_rho)
            # jac:       d(rho_term) = d (S_inv) * rho 
            #            d(rho_term) = -(S_inv * dS * S_inv) * rho
    
            # objective: ET_term = (E.T * S - target_E.T)
            # jac:       d(ET_term) = E.T * dS


            #Distribute computation across processors
            allDerivColSlice = slice(0,N)
            derivSlices, myDerivColSlice, derivOwners, mySubComm = \
                    _mpit.distribute_slice(allDerivColSlice, comm)
            if mySubComm is not None:
                _warnings.warn("Note: more CPUs(%d)" % comm.Get_size()
                       +" than gauge-opt derivative columns(%d)!" % N)

            n = _slct.length(myDerivColSlice)
            wrtIndices = _slct.indices(myDerivColSlice) if (n < N) else None
            my_jacMx = jacMx[:,myDerivColSlice] #just the columns I'm responsible for

            # S, and S_inv are shape (d,d)
            S       = gaugeGroupEl.get_transform_matrix()
            S_inv   = gaugeGroupEl.get_transform_matrix_inverse()
            dS      = gaugeGroupEl.deriv_wrt_params(wrtIndices) # shape (d*d),n
            dS.shape = (d, d, n) # call it (d1,d2,n)
            dS  = _np.rollaxis(dS, 2) # shape (n, d1, d2)
            assert(dS.shape == (n,d,d))

            # -- Gate terms
            # -------------------------
            for lbl, G in gs_pre.gates.items():
                # d(gate_term) = S_inv * (-dS * S_inv * G * S + G * dS) = S_inv * (-dS * G' + G * dS)
                #   Note: (S_inv * G * S) is G' (transformed G)
                wt   = itemWeights.get(lbl, gateWeight)
                left = -1 * _np.dot(dS, gs_post.gates[lbl]) # shape (n,d1,d2)
                right = _np.swapaxes(_np.dot(G, dS), 0,1) # shape (d1, n, d2) -> (n,d1,d2)
                result = _np.swapaxes(_np.dot(S_inv, left + right), 1,2) # shape (d1, d2, n)
                result = result.reshape( (d**2, n) ) #must copy b/c non-contiguous
                my_jacMx[start:start+d**2] = wt * result
                start += d**2

            # -- prep terms
            # -------------------------                
            for lbl, rho in gs_post.preps.items():
                # d(rho_term) = -(S_inv * dS * S_inv) * rho
                #   Note: (S_inv * rho) is transformed rho
                wt   = itemWeights.get(lbl, spamWeight)
                Sinv_dS  = _np.dot(S_inv, dS) # shape (d1,n,d2)
                result = -1 * _np.dot(Sinv_dS, rho).squeeze(2) # shape (d,n,1) => (d,n)
                my_jacMx[start:start+d] = wt * result
                start += d


            # -- effect terms
            # -------------------------
            for lbl, E in gs_pre.effects.items():
                # d(ET_term) = E.T * dS
                wt   = itemWeights.get(lbl, spamWeight)
                result = _np.dot(E.T, dS).T  # shape (1,n,d2).T => (d2,n,1)
                my_jacMx[start:start+d] = wt * result.squeeze(2) # (d2,n)
                start += d
                
            if gs_pre._povm_identity is not None:
                # same as effects
                wt   = itemWeights.get(gs_pre._identitylabel, spamWeight)
                result = _np.dot(gs_pre._povm_identity.T, dS).T
                my_jacMx[start:start+d] = wt * result.squeeze(2) # (d2,n)
                start += d

            # -- penalty terms
            # -------------------------
            if cptp_penalty_factor > 0:
                start += _cptp_penalty_jac_fill(my_jacMx[start:], gs_pre, gs_post,
                                                gaugeGroupEl, cptp_penalty_factor,
                                                gs_pre.basis, wrtIndices)

            if spam_penalty_factor > 0:
                start += _spam_penalty_jac_fill(my_jacMx[start:], gs_pre, gs_post,
                                                gaugeGroupEl, spam_penalty_factor,
                                                gs_pre.basis, wrtIndices)
                

            #At this point, each proc has filled the portions (columns) of jacMx that
            # it's responsible for, and so now we gather them together.
            _mpit.gather_slices(derivSlices, derivOwners, jacMx, 1, comm)
            #Note jacMx is completely filled (on all procs)

            if checkJac and (comm is None or comm.Get_rank() == 0):
                def mock_objective_fn(v):
                    gaugeGroupEl.from_vector(v)
                    gs = gs_pre.copy()
                    gs.transform(gaugeGroupEl)
                    return objective_fn(gs)

                vec = gaugeGroupEl.to_vector()
                _opt.check_jac(mock_objective_fn, vec, jacMx, tol=1e-5, eps=1e-7, errType='abs')
                
            return jacMx

    else:
        # non-least-squares case where objective function returns a single float
        # and (currently) there's no analytic jacobian
        
        def objective_fn(gs):
            ret = 0
    
            if cptp_penalty_factor > 0:
                gs.basis = mxBasis #set basis for jamiolkowski iso
                cpPenaltyVec = _cptp_penalty(gs,cptp_penalty_factor,gs.basis)
                ret += _np.sum(cpPenaltyVec)
    
            if spam_penalty_factor > 0:
                gs.basis = mxBasis
                spamPenaltyVec = _spam_penalty(gs,spam_penalty_factor,gs.basis)
                ret += _np.sum(spamPenaltyVec)
    
            if targetGateset is not None:
                if gatesMetric == "frobenius":
                    if spamMetric == "frobenius":
                        ret += gs.frobeniusdist(targetGateset, None, gateWeight,
                                                spamWeight, itemWeights)
                    else:
                        ret += gs.frobeniusdist(targetGateset,None,gateWeight,0,itemWeights)
        
                elif gatesMetric == "fidelity":
                    for gateLbl in gs.gates:
                        wt = itemWeights.get(gateLbl, gateWeight)
                        ret += wt * (1.0 - _tools.process_fidelity(
                                targetGateset.gates[gateLbl], gs.gates[gateLbl]))**2
        
                elif gatesMetric == "tracedist":
                        for gateLbl in gs.gates:
                            wt = itemWeights.get(gateLbl, gateWeight)
                            ret += gateWeight * _tools.jtracedist(
                                targetGateset.gates[gateLbl], gs.gates[gateLbl])
        
                else: raise ValueError("Invalid gatesMetric: %s" % gatesMetric)
        
                if spamMetric == "frobenius":
                    pass #added in special case above to match normalization in frobeniusdist
                    #ret += gs.frobeniusdist(targetGateset,None,0,spamWeight,itemWeights)
        
                elif spamMetric == "fidelity":
                    for spamlabel in gs.get_spam_labels():
                        wt = itemWeights.get(spamlabel, spamWeight)
                        ret += wt * (1.0 - _tools.process_fidelity(
                                targetGateset.get_spamgate(spamlabel),
                                gs.get_spamgate(spamlabel)))**2
        
                elif spamMetric == "tracedist":
                    for spamlabel in gs.get_spam_labels():
                        wt = itemWeights.get(spamlabel, spamWeight)
                        ret += wt * _tools.jtracedist(
                            targetGateset.get_spamgate(spamlabel),
                            gs.get_spamgate(spamlabel))
        
                else: raise ValueError("Invalid spamMetric: %s" % spamMetric)
    
            return ret

        jacobian_fn = None
        
    return objective_fn, jacobian_fn


def _cptp_penalty(gs,prefactor,gateBasis):
    """
    Helper function - CPTP penalty: (sum of tracenorms of gates),
    which in least squares optimization means returning an array
    of the sqrt(tracenorm) of each gate.  This function is the
    *same* as that in core.py.

    Returns
    -------
    numpy array
        a (real) 1D array of length len(gs.gates).
    """
    from ..algorithms.core import _cptp_penalty as _core_cptp_penalty
    return _core_cptp_penalty(gs, prefactor, gateBasis)


def _spam_penalty(gs,prefactor,gateBasis):
    """
    Helper function - CPTP penalty: (sum of tracenorms of gates),
    which in least squares optimization means returning an array
    of the sqrt(tracenorm) of each gate.  This function is the
    *same* as that in core.py.

    Returns
    -------
    numpy array
        a (real) 1D array of length len(gs.gates).
    """
    from ..algorithms.core import _spam_penalty as _core_spam_penalty
    return _core_spam_penalty(gs, prefactor, gateBasis)


def _cptp_penalty_jac_fill(cpPenaltyVecGradToFill, gs_pre, gs_post,
                           gaugeGroupEl, prefactor, gateBasis, wrtFilter):
    """
    Helper function - jacobian of CPTP penalty (sum of tracenorms of gates)
    Returns a (real) array of shape (len(gs.gates), gaugeGroupEl.num_params()).
    """
    
    # d( sqrt(|chi|_Tr) ) = (0.5 / sqrt(|chi|_Tr)) * d( |chi|_Tr )
    # but here, unlike in core.py, chi = J(S_inv * G * S) == J(G')
    # and we're differentiating wrt the parameters of S, the
    # gaugeGroupEl.

    # S, and S_inv are shape (d,d)
    d,N     = gs_pre.dim, gaugeGroupEl.num_params()
    n       = N if (wrtFilter is None) else len(wrtFilter)
    S       = gaugeGroupEl.get_transform_matrix()
    S_inv   = gaugeGroupEl.get_transform_matrix_inverse()
    dS      = gaugeGroupEl.deriv_wrt_params(wrtFilter) # shape (d*d),n
    dS.shape = (d, d, n) # call it (d1,d2,n)
    dS  = _np.rollaxis(dS, 2) # shape (n, d1, d2)

    for i,(gl,gate) in enumerate(gs_post.iter_gates()):
        pre_gate = gs_pre.gates[gl]

        #get sgn(chi-matrix) == d(|chi|_Tr)/dchi in std basis
        # so sgnchi == d(|chi_std|_Tr)/dchi_std
        chi = _tools.fast_jamiolkowski_iso_std(gate, gateBasis)
        assert(_np.linalg.norm(chi - chi.T.conjugate()) < 1e-4), \
            "chi should be Hermitian!"

        U,s,Vt = _np.linalg.svd(chi)
        sgnvals = [ sv/abs(sv) if (abs(sv) > 1e-7) else 0.0 for sv in s]
        sgnchi = _np.dot(U,_np.dot(_np.diag(sgnvals),Vt))
        assert(_np.linalg.norm(sgnchi - sgnchi.T.conjugate()) < 1e-4), \
            "sngchi should be Hermitian!"
        
        # Let M be the "shuffle" operation performed by fast_jamiolkowski_iso_std
        # which maps a gate onto the choi-jamiolkowsky "basis" (i.e. performs that C-J
        # transform).  This shuffle op commutes with the derivative, so that
        # dchi_std/dp := d(M(G'))/dp = M(d(S_inv*G*S)/dp) = M( d(S_inv)*G*S + S_inv*G*dS )
        #              = M( (-S_inv*dS*S_inv)*G*S + S_inv*G*dS ) = M( S_inv*(-dS*S_inv*G*S) + G*dS )
        left = -1 * _np.dot(dS, gate) # shape (n,d1,d2)
        right = _np.swapaxes(_np.dot(pre_gate, dS), 0,1) # shape (d1, n, d2) -> (n,d1,d2)
        result = _np.swapaxes(_np.dot(S_inv, left + right), 0,1) # shape (n, d1, d2)

        dchi_std = _np.empty((n,d,d), 'complex')
        for p in range(n): #p indexes param
            dchi_std[p] = _tools.fast_jamiolkowski_iso_std(result[p], gateBasis) # "M(results)"
            assert(_np.linalg.norm(dchi_std[p] - dchi_std[p].T.conjugate()) < 1e-8) #check hermitian

        dchi_std = _np.conjugate(dchi_std) # I'm not sure
          # why this is needed: I think b/c els of sgnchi and MdGdp_std[p]
          # are hermitian and we want to think of the real and im parts of
          # (i,j) and (j,i) els as the two "variables" we differentiate wrt.
          # TODO: work this out on paper.

        #contract to get (note contract along both mx indices b/c treat like a
        # mx basis): d(|chi_std|_Tr)/dp = d(|chi_std|_Tr)/dchi_std * dchi_std/dp
        v =  _np.einsum("ij,aij->a",sgnchi,dchi_std)
        v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(chi))) #add 0.5/|chi|_Tr factor
        assert(_np.linalg.norm(v.imag) < 1e-4)
        cpPenaltyVecGradToFill[i,:] = v.real
        chi = sgnchi = dchi_std = v = None #free mem
        
    return len(gs_pre.gates) #the number of leading-dim indicies we filled in


def _spam_penalty_jac_fill(spamPenaltyVecGradToFill, gs_pre, gs_post,
                           gaugeGroupEl, prefactor, gateBasis, wrtFilter):
    """
    Helper function - jacobian of CPTP penalty (sum of tracenorms of gates)
    Returns a (real) array of shape (len(gs.preps), gaugeGroupEl.num_params()).
    """
    BMxs = gateBasis.get_composite_matrices() #shape [gs.dim, dmDim, dmDim]
    ddenMxdV = BMxs # b/c denMx = sum( spamvec[i] * Bmx[i] ) and "V" == spamvec
    assert(ddenMxdV.size > 0), "Could not obtain basis matrices from " \
        + "'%s' basis for spam pentalty factor!" % gateBasis.name

    # S, and S_inv are shape (d,d)
    d,N     = gs_pre.dim, gaugeGroupEl.num_params()
    n       = N if (wrtFilter is None) else len(wrtFilter)
    S_inv   = gaugeGroupEl.get_transform_matrix_inverse()
    dS      = gaugeGroupEl.deriv_wrt_params(wrtFilter) # shape (d*d),n
    dS.shape = (d, d, n) # call it (d1,d2,n)
    dS  = _np.rollaxis(dS, 2) # shape (n, d1, d2)

    # d( sqrt(|denMx|_Tr) ) = (0.5 / sqrt(|denMx|_Tr)) * d( |denMx|_Tr )
    # but here, unlike in core.py, denMx = StdMx(S_inv * rho) = StdMx(rho')
    # and we're differentiating wrt the parameters of S, the
    # gaugeGroupEl.
    
    for i,(lbl,prepvec) in enumerate(gs_post.iter_preps()):

        #get sgn(denMx) == d(|denMx|_Tr)/d(denMx) in std basis
        denMx = _tools.vec_to_stdmx(prepvec, gateBasis)
        dmDim = denMx.shape[0]
        assert(_np.linalg.norm(denMx - denMx.T.conjugate()) < 1e-4), \
            "denMx should be Hermitian!"
        
        U,s,Vt = _np.linalg.svd(denMx)
        sgnvals = [ sv/abs(sv) if (abs(sv) > 1e-7) else 0.0 for sv in s]
        sgndm = _np.dot(U,_np.dot(_np.diag(sgnvals),Vt))
        assert(_np.linalg.norm(sgndm - sgndm.T.conjugate()) < 1e-4), \
            "sngdm should be Hermitian!"

        # get d(prepvec')/dp = d(S_inv * prepvec)/dp in gateBasis [shape == (n,dim)]
        #                    = (-S_inv*dS*S_inv) * prepvec = -S_inv*dS * prepvec'
        Sinv_dS  = _np.dot(S_inv, dS) # shape (d1,n,d2)
        dVdp = -1 * _np.dot(Sinv_dS, prepvec).squeeze(2) # shape (d,n,1) => (d,n)
        assert(dVdp.shape == (d,n))

        # denMx = sum( spamvec[i] * Bmx[i] )

        #contract to get (note contract along both mx indices b/c treat like a mx basis):
        # d(|denMx|_Tr)/dp = d(|denMx|_Tr)/d(denMx) * d(denMx)/d(spamvec) * d(spamvec)/dp
        # [dmDim,dmDim] * [gs.dim, dmDim,dmDim] * [gs.dim, n]
        v =  _np.einsum("ij,aij,ab->b",sgndm,ddenMxdV,dVdp)
        v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(denMx))) #add 0.5/|denMx|_Tr factor
        assert(_np.linalg.norm(v.imag) < 1e-4)
        spamPenaltyVecGradToFill[i,:] = v.real
        denMx = sgndm = dVdp = v = None #free mem

    return len(gs_pre.preps) #the number of leading-dim indicies we filled in
