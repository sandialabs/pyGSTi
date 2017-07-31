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

from ..objects import Basis

def create_objective_fn(gateset, targetGateset, itemWeights=None,
                       CPpenalty=0, TPpenalty=0, validSpamPenalty=0,
                       gatesMetric="frobenius", spamMetric="frobenius"):
    if itemWeights is None: itemWeights = {}
    gateWeight = itemWeights.get('gates',1.0)
    spamWeight = itemWeights.get('spam',1.0)
    mxBasis = gateset.basis.name
    basisDim = gateset.basis.dim.blockDims

    #Use the target gateset's basis if gateset's is unknown
    # (as it can often be if it's just come from an logl opt,
    #  since from_vector will clear any basis info)
    if mxBasis == "unknown" and targetGateset is not None: 
        mxBasis = targetGateset.basis.name
        basisDim = targetGateset.basis.dim.blockDims
    def objective_fn(gs):
        ret = 0

        if CPpenalty != 0:
            gs.basis = Basis(mxBasis,basisDim) #set basis for jamiolkowski iso
            s = _tools.sum_of_negative_choi_evals(gs, weights=None)
              # we desire *uniform* weights (don't specify itemWeights here)
            ret += CPpenalty * s

        if TPpenalty != 0:
            rhoVecFirstEl = 1.0 / (gs.dim**0.25)  # note: sqrt(gateDim) gives linear dim of density mx
            for gate in gs.gates.values():
                ret += TPpenalty * abs(1.0-gate[0,0])
                for k in range(1,gate.shape[1]):
                    ret += TPpenalty * abs(gate[0,k])
            for rhoVec in gs.preps.values():
                ret += TPpenalty * abs(rhoVecFirstEl - rhoVec[0])

        if validSpamPenalty != 0:
            sp =  sum( [ _tools.prep_penalty(rhoVec,mxBasis) for rhoVec in gs.preps.values() ] )
            sp += sum( [ _tools.effect_penalty(EVec,mxBasis) for EVec   in gs.effects.values() ] )
            ret += validSpamPenalty*sp

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
    return objective_fn

def gaugeopt_to_target(gateset, targetGateset, itemWeights=None,
                       CPpenalty=0, TPpenalty=0, validSpamPenalty=0,
                       gatesMetric="frobenius", spamMetric="frobenius",
                       gauge_group=None, method='L-BFGS-B', maxiter=100000,
                       maxfev=None, tol=1e-8, returnAll=False, verbosity=0,
                       checkJac=False):
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
       
    CPpenalty : float, optional
        A prefactor for a L1-like CP pentalty term that prefers gatesets 
        which are completely positive.  The penalty term equals the sum
        of the magnitudes of the negative eigenvalues of each gate's 
        Choi matrix.

    TPpenalty : float, optional
        A prefactor for an L1-like TP pentalty term which penalizes
        gates when their first row is *not* equal to [1, 0, ... 0],
        and penalizes state preparation vectors when their first element
        does not equal sqrt( 1/sqrt(dim) ).

    validSpamPenalty : float, optional
        A prefactor for a term in the objective function which penalizes
        state preparation vectors which don't correspond to *positive*
        (density) matrices and penalizes effects which have eigenvalues
        outside of the [0,1] range.
        
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
    
    if CPpenalty         == 0 and \
        TPpenalty        == 0 and \
        validSpamPenalty == 0 and \
        targetGateset is not None and \
        gatesMetric == "frobenius" and \
        spamMetric  == "frobenius":# and \
        #gateset.dim < 64:

        gateWeight = itemWeights.get('gates',1.0)
        spamWeight = itemWeights.get('spam',1.0)
        def objective_fn(gs):
            residuals, nsummands = gs.residuals(targetGateset, None, gateWeight, spamWeight, itemWeights)
            return residuals

        algorithm = 'ls'
    else:
        objective_fn = create_objective_fn(gateset, targetGateset,
                itemWeights, 
                CPpenalty, TPpenalty, 
                validSpamPenalty, gatesMetric, 
                spamMetric)
        algorithm = 'min'

    result = gaugeopt_custom(gateset, objective_fn, gauge_group, method,
            maxiter, maxfev, tol, returnAll, verbosity, algorithm=algorithm, 
            itemWeights=itemWeights, checkJac=checkJac)

    #If we've gauge optimized to a target gate set, declare that the
    # resulting gate set is now in the same basis as the target.
    if targetGateset is not None:
        newGateset = result[-1] if returnAll else result
        newGateset.basis = Basis(targetGateset.basis.name,
                             targetGateset.basis.dim.blockDims)
        #DEBUG
        #print('gauge opt result distance to target')
        #print(newGateset.frobeniusdist(targetGateset))
    
    return result

def calculate_ls_jacobian(gaugeGroupEl, gateset, call_objective_fn, itemWeights, check=False):
    gateWeight = itemWeights.get('gates',1.0)
    spamWeight = itemWeights.get('spam',1.0)
    def jacobian(vec):
        '''
        The derivative of the objective function with respect the input parameter vector, v
        This is UNSQUARED, because the residuals are unsquared.
        '''
        # equivalent to a 'from numpy import dot', but only for the scope of this function
        dot = _np.dot

        gaugeGroupEl.from_vector(vec)
        gs = gateset.copy()
        gs.transform(gaugeGroupEl)

        gates   = list(gs.gates.values())
        preps   = list(gs.preps.values())
        effects = list(gs.effects.values())
        assert len(gates) > 0

        # Indices: Jacobian output matrix has shape (L, N)
        start = 0
        d = gates[0].shape[0]
        N = vec.shape[0]
        L = (d**2 * len(gates)   + 
             d    * len(preps)   + 
             d    * len(effects) +
             d    * (0 if gs._povm_identity is None else 1))
        jacMx = _np.zeros((L, N))

        # S, and S_inv are (dxd)
        S       = gaugeGroupEl.get_transform_matrix()
        S_inv   = gaugeGroupEl.get_transform_matrix_inverse()
        # dS is ((d*d)xlen(v))
        dS      = gaugeGroupEl.gate.deriv_wrt_params()
        # dS is (dxdxlen(v))
        dS.shape = (d, d, N)
        rolled  = _np.rollaxis(dS, 2)
        rolled2 = _np.rollaxis(dS, 2, 1)
        for Glabel, G in gs.gates.items():
            '''
            Jac_gate = S_inv @ [dS @ S_inv @ Gk @ S - Gk @ dS]
                                ^^^^^^^^^^^^^^^^^^^   ^^^^^^^
                                left                  right

            => (d**2, N) mx
            '''
            wt   = itemWeights.get(Glabel, gateWeight)

            right = dot(G, rolled)
            right = _np.rollaxis(right, 2, 1)

            left = dot(S_inv, dot(G, S))
            left = dot(rolled, left)
            left = _np.rollaxis(left, 0, 3)

            result = left - right
            result = _np.rollaxis(result, 2) # Roll the same way as dS
            result = dot(S_inv, result)
            result = _np.rollaxis(result, 2, 1) # Don't mix up d sides
            result = -1 * result.reshape((d**2, N))
            assert result.shape == (d ** 2, N)
            jacMx[start:start+d**2] = wt * result
            start += d**2
        for Plabel, P in gs.preps.items():
            '''
            Jac_prep = - S_inv @ dS @ S_inv @ P
                         ^^^^^^^^^^   ^^^^^^^^^       
                         left         right
            => (d, N) mx
            '''
            wt   = itemWeights.get(Plabel, spamWeight)

            left   = dot(S_inv, rolled)
            right  = dot(S_inv, P)
            result = dot(left, right)
            result *= -1
            result.shape = (d, N)
            assert result.shape == (d, N)
            jacMx[start:start+d] = wt * result
            start += d
        for Elabel, E in gs.effects.items():
            '''
            Jac_effect = Ek.T @ dS
            => (d, N) mx
            '''
            wt   = itemWeights.get(Elabel, spamWeight)

            result = dot(E.T, rolled).T
            # (1,d, N) to (d, N)
            result.shape = (d, N)
            assert result.shape == (d, N)
            jacMx[start:start+d] = wt * result
            start += d
        if gs._povm_identity is not None:
            '''
            Jac_effect = Ek.T @ dS
            => (d, N) mx
            '''
            wt   = itemWeights.get(gs._identitylabel, spamWeight)

            result = dot(gs._povm_identity.T, rolled).T
            result.shape = (d, N)
            # (1,d, N) to (d, N)
            assert result.shape == (d, N)
            jacMx[start:start+d] = wt * result
            start += d
        if check:
            alt_jac = _opt.optimize._fwd_diff_jacobian(call_objective_fn, vec, 1e-2)
            if not _tools.array_eq(jacMx, alt_jac, tol=1e-3):
                # Testing and comparison with the plotting code below has show that usually this difference is very small, even when a finer color scale is used
                print('Warning: finite differences jacobian differs from analytic jacobian')
                '''
                import matplotlib.pyplot as plt
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

                ax1.matshow(jacMx, vmin=-1, vmax=1)
                ax1.set_title('analytic')
                ax2.matshow(alt_jac, vmin=-1, vmax=1)
                ax2.set_title('ffd')
                im = ax3.matshow(alt_jac - jacMx, vmin=-1, vmax=1)
                ax3.set_title('combined')

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(im, cax=cbar_ax)
                plt.show()
                '''
        return jacMx
    return jacobian

def gaugeopt_custom(gateset, objective_fn, gauge_group=None,
                    method='L-BFGS-B', maxiter=100000, maxfev=None, tol=1e-8,
                    returnAll=False, verbosity=0, algorithm='min', itemWeights=None,
                    checkJac=False):
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

    verbosity : int, optional
        How much detail to send to stdout.

    algorithm : string, either 'ls' or 'min', optional
        The algorithm to use when performing gauge optimization

    itemWeights : dict, optional
        Dictionary of weighting factors for gates and spam operators.  Keys can
        be gate, state preparation, or POVM effect, as well as the special values
        "spam" or "gates" which apply the given weighting to *all* spam operators
        or gates respectively.  Values are floating point numbers.  Values given 
        for specific gates or spam operators take precedence over "gates" and
        "spam" values.  The precise use of these weights depends on the gateset
        metric(s) being used.

    checkJac : bool
        Check the Jacobian against finite differences

    Returns
    -------
    gateset                            if returnAll == False

    (goodnessMin, gaugeMx, gateset)    if returnAll == True

      where goodnessMin is the minimum value of the goodness function (the best 'goodness')
      found, gaugeMx is the gauge matrix used to transform the gateset, and gateset is the
      final gauge-transformed gateset.
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    if itemWeights is None:
        itemWeights = dict()

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
        transform = gaugeGroupEl.get_transform_matrix()
        gs.transform(gaugeGroupEl)
        return objective_fn(gs)

    bToStdout = (printer.verbosity > 2 and printer.filename is None)
    print_obj_func = _opt.create_obj_func_printer(call_objective_fn) #only ever prints to stdout!
    if bToStdout: 
        print_obj_func(x0) #print initial point


    if algorithm == 'min':
        minSol = _opt.minimize(call_objective_fn, x0,
                              method=method, maxiter=maxiter, maxfev=maxfev, tol=tol,
                              callback = print_obj_func if bToStdout else None)
    elif algorithm == 'ls':
        jacobian = calculate_ls_jacobian(gaugeGroupEl, gateset, call_objective_fn, itemWeights, checkJac)
        minSol  = _opt.least_squares(call_objective_fn, x0, jac=jacobian,
                                    max_nfev=maxfev, ftol=tol)
    else:
        raise ValueError('Unknown algorithm inside of gauge opt: {}'.format(algorithm))

    gaugeGroupEl.from_vector(minSol.x)
    newGateset = gateset.copy()
    newGateset.transform(gaugeGroupEl)

    if returnAll:
        return minSol.fun, gaugeMat, newGateset
    else:  
        return newGateset


# OLD ############################################################################################


#Note: this code overlaps do_mlgst a lot -- consolidate in FUTURE?
def optimize_gauge(gateset, toGetTo, maxiter=100000, maxfev=None, tol=1e-8,
                   method='L-BFGS-B', targetGateset=None, targetFactor=0.0001,
                   constrainToTP=False, constrainToCP=False, constrainToValidSpam=False,
                   returnAll=False, gateWeight=1.0, spamWeight=1.0, itemWeights=None,
                   targetGatesMetric="frobenius", targetSpamMetric="frobenius",
                   verbosity=0):
    """
    Optimize the gauge of a GateSet using some 'goodness' function.

    Note: this function is DEPRECATED, and should be replaced, usually,
    by a call to :func:`gaugeopt_to_target` or :func:`gaugeopt_custom`.

    Parameters
    ----------
    gateset : GateSet
        The gateset to gauge-optimize

    toGetTo : string
        Specifies which goodness function is used.  Allowed values are:

        - 'target' -- minimize the frobenius norm of the difference between
          gateset and targetGateset, which must be specified.
        - 'CPTP'   -- minimize the non-CPTP-ness of the gateset.
        - 'TP'     -- minimize the non-TP-ness of the gateset.
        - 'TP and target' -- minimize the non-TP-ness of the gateset and
          the frobenius norm distance to targetGateset using targetFactor
          to multiply this distance.
        - 'CPTP and target' -- minimize the non-CPTP-ness of the gateset and
          the frobenius norm distance to targetGateset using targetFactor to
          multiply this distance.
        - 'Completely Depolarized' -- minimize the frobenius norm distance
          between gateset and the completely-depolarized gateset.

    maxiter : int, optional
        Maximum number of iterations for the gauge optimization.

    maxfev : int, optional
        Maximum number of function evaluations for the gauge optimization.
        Defaults to maxiter.

    tol : float, optional
        The tolerance for the gauge optimization.

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

    targetGateset : GateSet, optional
        The target gateset used by the 'target', 'TP and target' and 'CPTP and target'
        values of the toGetTo parameter (above).

    targetFactor : float, optional
        A weighting factor that multiplies by the frobenius norm difference between gateset
        and targetGateset with toGetTo is either "TP and target" or "CPTP and target".

    constrainToTP : bool, optional
       When toGetTo == 'target', whether gauge optimization is constrained to TP gatesets.

    constrainToCP : bool, optional
       When toGetTo == 'target', whether gauge optimization is constrained to CP gatesets.

    constrainToValidSpam : bool, optional
       When toGetTo == 'target', whether gauge optimization is constrained to gatesets with
       valid state preparation and measurements.

    returnAll : bool, optional
       When True, return best "goodness" value and gauge matrix in addition to the
       gauge optimized gateset.

    gateWeight : float, optional
       Weighting factor that multiplies each single-gate norm before summing it
       into the total frobenius norm between two gatesets.

    spamWeight : float, optional
       Weighting factor that multiplies the norms of between surface-preparation
       and measuement vectors (or gates, depending on the metric used) before
       summing them into the total norm between two gatesets.

    itemWeights : dict, optional
       Dictionary of weighting factors for individual gates and spam operators.
       Keys can be gate, state preparation, POVM effect, or spam labels.  Values
       are floating point numbers.  By default, weights are set by gateWeight
       and spamWeight.  All values *present* in itemWeights override gateWeight
       and spamWeight.

    targetGatesMetric : string, optional
       When toGetTo == "target", this specifies the metric used to evaluate what
       "close to the target" means for the gate matrices.  Allowed values are
       "frobenius", "fidelity", and "tracedist". Contributions for the individual
       gates are summed, and in the case of frobenius, ultimately normalized by
       the number of elements.

    targetSpamMetric : string, optional
       When toGetTo == "target", this specifies the metric used to evaluate what
       "close to the target" means for the spam vectors.  Allowed values are
       "frobenius", "fidelity", and "tracedist". Contributions for the individual
       vectors are summed, and in the case of frobenius, ultimately normalized
       by the number of elements.

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

    _warnings.warn("The function 'optimize_gauge' is deprecated," +
                   " and may be removed in future versions of pyGSTi")

    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    #OLD
    #    - 'best fidelity' -- minimize the sum of gate fidelities between
    #      gateset and targetGateset, which must be specified.
    #    - 'best trace distance' -- minimize the sum of trace distances between
    #      gateset and targetGateset, which must be specified.

    if maxfev is None: maxfev = maxiter
    if itemWeights is None: itemWeights = {}

    gateDim = gateset.get_dimension()
    firstRowForTP = _np.zeros(gateDim); firstRowForTP[0] = 1.0
    mxBasis = gateset.basis.name
    basisDim = gateset.basis.dim.blockDims

    if toGetTo == "target":
        if targetGateset is None: raise ValueError("Must specify a targetGateset != None")

        CLIFF = 1e10
        cpPenalty = CLIFF if constrainToCP else 0
        spamPenalty = CLIFF if constrainToValidSpam else 0
        assert(method != "custom") #do not allow use of custom method yet (since it required a different obj func)

        #printer.log('', 2)
        printer.log(("--- Gauge Optimization to a target (%s) ---" % method), 1)

        def objective_func(vectorM):
            if constrainToTP: vectorM = _np.concatenate( (firstRowForTP,vectorM) )
            matM = vectorM.reshape( (gateDim,gateDim) )
            ggEl = _objs.FullGaugeGroup.element(matM)
            gs = gateset.copy(); gs.transform(ggEl)

            if cpPenalty != 0:
                gs.basis = Basis(mxBasis,basisDim) #set basis for jamiolkowski iso
                s = _tools.sum_of_negative_choi_evals(gs)
                if s > 1e-8: return cpPenalty #*(1.0+s) #1e-8 should match TOL in contract to CP routines

            if spamPenalty != 0:
                sp =  sum( [ _tools.prep_penalty(rhoVec,mxBasis) for rhoVec in list(gs.preps.values()) ] )
                sp += sum( [ _tools.effect_penalty(EVec,mxBasis) for EVec   in list(gs.effects.values()) ] )
                if sp > 1e-8: return spamPenalty #*(1.0+sp) #1e-8 should match TOL in contract to CP routines


            #Special case of full frobenius norm
                # TODO: remove?  but note this is different from the separate cases summed b/c of normalization
            if targetGatesMetric == "frobenius" and targetSpamMetric == "frobenius":
                return gs.frobeniusdist(targetGateset, None, gateWeight,
                                        spamWeight, itemWeights)

            diff = 0
            if targetGatesMetric == "frobenius":
                diff += gs.frobeniusdist(targetGateset, None, gateWeight,
                                         0.0, itemWeights)

            elif targetGatesMetric == "fidelity":
                for gateLbl in gs.gates:
                    wt = itemWeights.get(gateLbl, gateWeight)
                    diff += wt * (1.0 - _tools.process_fidelity(
                            targetGateset.gates[gateLbl], gs.gates[gateLbl]))**2

            elif targetGatesMetric == "tracedist":
                for gateLbl in gs.gates:
                    wt = itemWeights.get(gateLbl, gateWeight)
                    diff += gateWeight * _tools.jtracedist(
                        targetGateset.gates[gateLbl], gs.gates[gateLbl])
            else: raise ValueError("Invalid targetGatesMetric: %s" % targetGatesMetric)

            if targetSpamMetric == "frobenius":
                diff += gs.frobeniusdist(targetGateset, None, 0.0,
                                         spamWeight, itemWeights)

            elif targetSpamMetric == "fidelity":
                for spamlabel in gs.get_spam_labels():
                    wt = itemWeights.get(spamlabel, spamWeight)
                    diff += wt * (1.0 - _tools.process_fidelity(
                            targetGateset.get_spamgate(spamlabel),
                            gs.get_spamgate(spamlabel)))**2

            elif targetSpamMetric == "tracedist":
                for spamlabel in gs.get_spam_labels():
                    wt = itemWeights.get(spamlabel, spamWeight)
                    diff += wt * _tools.jtracedist(
                        targetGateset.get_spamgate(spamlabel),
                        gs.get_spamgate(spamlabel))

            else: raise ValueError("Invalid targetSpamMetric: %s" % targetGatesMetric)

            return diff


    elif toGetTo == "CPTP":

        if constrainToTP: #assume gateset is already in TP so no TP optimization needed
            tpGateset = gateset
            tpGaugeMx = _np.identity( gateDim, 'd' )
        else:
            _, tpGaugeMx, tpGateset = optimize_gauge(
                gateset,"TP", maxiter, maxfev, tol,
                'L-BFGS-B', targetGateset, targetFactor,
                constrainToTP, constrainToCP, constrainToValidSpam, True,
                gateWeight, spamWeight, itemWeights, printer-1)

        #printer.log('', 2)
        printer.log(("--- Gauge Optimization to CPTP w/valid SPAM (%s) ---" % method), 1)
        constrainToTP = True #always constrain next optimization to TP

        #DEBUG
        #import pickle
        #bestGaugeMx = pickle.load(open("bestGaugeMx.debug"))

        def objective_func(vectorM):
            #matM = vectorM.reshape( (gateDim,gateDim) )
            #gs = gateset.copy(); gs.transform(matM)

            vectorM = _np.concatenate( (firstRowForTP,vectorM) )
            matM = vectorM.reshape( (gateDim,gateDim) )
            ggEl = _objs.TPGaugeGroup.element(matM)
            gs = tpGateset.copy(); gs.transform(ggEl)

            gs.basis = Basis(mxBasis,basisDim) #set basis for jamiolkowski iso
            cpPenalties = _tools.sums_of_negative_choi_evals(gs)
            #numNonCP = sum([ 1 if p > 1e-4 else 0 for p in cpPenalties ])
            #cpPenalty = sum( [ 10**i*cp for (i,cp) in enumerate(cpPenalties)] ) + 100*numNonCP #DEBUG
            #print "DB: diff from best = ", frobeniusnorm(bestGaugeMx - matM) #DEBUG
            cpPenalty = sum( cpPenalties )

            spamPenalty =  sum( [ _tools.prep_penalty(rhoVec,mxBasis) for rhoVec in list(gs.preps.values()) ] )
            spamPenalty += sum( [ _tools.effect_penalty(EVec,mxBasis) for EVec   in list(gs.effects.values()) ] )

            #OLD
            #tpPenalty = 0
            #for gate in gs.gates.values():
            #  tpPenalty += (1.0-gate[0,0])**2
            #  for k in range(1,gate.shape[1]):
            #    tpPenalty += gate[0,k]**2
            #return cpPenalty + spamPenalty + tpPenalty

            penalty = cpPenalty + spamPenalty
            if penalty > 1e-100: return _np.log10(penalty)
            else: return -100


    elif toGetTo == "TP":
        #printer.log('', 2)
        printer.log(("--- Gauge Optimization to TP (%s) ---" % method), 1)
        if constrainToTP: raise ValueError("Cannot gauge optimize to TP and constrain to TP")
        rhoVecFirstEl = 1.0 / gateDim**0.25  # note: sqrt(gateDim) gives linear dim of density mx

        def objective_func(vectorM):
            matM = vectorM.reshape( (gateDim,gateDim) )
            ggEl = _objs.FullGaugeGroup.element(matM)
            gs = gateset.copy(); gs.transform(ggEl)

            tpPenalty = 0
            for gate in list(gs.gates.values()):
                tpPenalty += (1.0-gate[0,0])**2
                for k in range(1,gate.shape[1]):
                    tpPenalty += gate[0,k]**2

            for rhoVec in list(gs.preps.values()):
                tpPenalty += (rhoVecFirstEl - rhoVec[0])**2

            return tpPenalty

    elif toGetTo == "TP and target":
        #printer.log('', 2)
        printer.log(("--- Gauge Optimization to TP and target (%s) ---" % method), 1)
        if targetGateset is None: raise ValueError("Must specify a targetGateset != None")
        if constrainToTP: raise ValueError("Cannot gauge optimize to TP and constrain to TP")
        rhoVecFirstEl = 1.0 / gateDim**0.25  # note: sqrt(gateDim) gives linear dim of density mx

        def objective_func(vectorM):
            matM = vectorM.reshape( (gateDim,gateDim) )
            ggEl = _objs.FullGaugeGroup.element(matM)
            gs = gateset.copy(); gs.transform(ggEl)

            tpPenalty = 0
            for gate in list(gs.gates.values()):
                tpPenalty += (1.0-gate[0,0])**2
                for k in range(1,gate.shape[1]):
                    tpPenalty += gate[0,k]**2

            for rhoVec in list(gs.preps.values()):
                tpPenalty += (rhoVecFirstEl - rhoVec[0])**2

            return tpPenalty + gs.frobeniusdist(targetGateset, None,
                                                gateWeight, spamWeight,
                                                itemWeights) * targetFactor


    elif toGetTo == "CPTP and target":
        if targetGateset is None: raise ValueError("Must specify a targetGateset != None")

        if constrainToTP: #assume gateset is already in TP so no TP optimization needed
            tpGateset = gateset
            tpGaugeMx = _np.identity( gateDim, 'd' )
        else:
            _, tpGaugeMx, tpGateset = optimize_gauge(gateset, "TP and target", maxiter, maxfev, tol,
                                                     'L-BFGS-B', targetGateset, targetFactor,
                                                     constrainToTP, constrainToCP, constrainToValidSpam, True,
                                                     gateWeight, spamWeight, itemWeights, printer-1)

        #printer.log('', 2)
        printer.log(("--- Gauge Optimization to CPTP and target w/valid SPAM (%s) ---" % method), 1)
        constrainToTP = True # always constrain next optimization to TP

        def objective_func(vectorM):
            vectorM = _np.concatenate( (firstRowForTP,vectorM) ) #constraining to TP
            matM = vectorM.reshape( (gateDim,gateDim) )
            ggEl = _objs.TPGaugeGroup.element(matM)
            gs = tpGateset.copy(); gs.transform(ggEl)

            gs.basis = Basis(mxBasis,basisDim) #set basis for jamiolkowski iso
            cpPenalties = _tools.sums_of_negative_choi_evals(gs)
            cpPenalty = sum( cpPenalties )

            spamPenalty =  sum( [ _tools.prep_penalty(rhoVec,mxBasis) for rhoVec in list(gs.preps.values()) ] )
            spamPenalty += sum( [ _tools.effect_penalty(EVec,mxBasis) for EVec   in list(gs.effects.values()) ] )

            targetPenalty = targetFactor * gs.frobeniusdist(
                targetGateset, None, gateWeight, spamWeight, itemWeights)

            penalty = cpPenalty + spamPenalty + targetPenalty
            if penalty > 1e-100: return _np.log10(penalty)
            else: return -100


    elif toGetTo == "Completely Depolarized":
        #printer.log('', 2)
        printer.log(("--- Gauge Optimization to Completely Depolarized w/valid SPAM (%s) ---" % method), 1)
        complDepolGate = _np.zeros( (gateDim,gateDim) )
        complDepolGate[0,0] = 1.0

        def objective_func(vectorM):
            if constrainToTP: vectorM = _np.concatenate( (firstRowForTP,vectorM) )
            matM = vectorM.reshape( (gateDim,gateDim) )
            ggEl = _objs.FullGaugeGroup.element(matM)

            gs = gateset.copy(); gs.transform(ggEl); d=0
            for gateLabel in gs.gates:
                d += _tools.frobeniusdist(gs.gates[gateLabel],complDepolGate)
            spamPenalty  = sum( [ _tools.prep_penalty(rhoVec,mxBasis) for rhoVec in list(gs.preps.values()) ] )
            spamPenalty += sum( [ _tools.effect_penalty(EVec,mxBasis) for EVec   in list(gs.effects.values()) ] )
            return d + spamPenalty

    else: raise ValueError("Invalid toGetTo passed to optimize_gauge: %s" % toGetTo)

    #Run Minimization Algorithm
    startM = _np.identity(gateDim)  #take identity as initial gauge matrix

    x0 = startM.flatten() if not constrainToTP else startM[1:,:].flatten()
    bToStdout = (printer.verbosity > 2 and printer.filename is None)
    print_obj_func = _opt.create_obj_func_printer(objective_func) #only ever prints to stdout!
    minSol = _opt.minimize(objective_func, x0,
                          method=method, maxiter=maxiter, maxfev=maxfev, tol=tol,
                          stopval= -20 if toGetTo == "CPTP" else None,
                          callback = print_obj_func if bToStdout else None) #stopval=1e-7 -- (before I added log10)

    if constrainToTP:
        v = _np.concatenate( (firstRowForTP,minSol.x) )
        gaugeMat = v.reshape( (gateDim,gateDim) )
    else:
        gaugeMat = minSol.x.reshape( (gateDim,gateDim) )

    if toGetTo in ("CPTP","CPTP and target"):
        gaugeMat = _np.dot(tpGaugeMx, gaugeMat) #include initial TP gauge tranform

    ggEl = _objs.FullGaugeGroup.element(gaugeMat)
    newGateset = gateset.copy()
    newGateset.transform(ggEl)
    #newGateset.log("Optimize Gauge", { 'method': method, 'tol': tol, 'toGetTo': toGetTo } )

    #If we've optimized to a target, set the basis of the new gatset
    if toGetTo in ("target", "TP and target", "CPTP and target"):
        newGateset.basis = Basis(targetGateset.basis.name,
                             targetGateset.basis.dim.blockDims)

    if toGetTo == "target":
        printer.log(('The resulting Frobenius-norm distance is: %g' % minSol.fun), 2)
        for gateLabel in newGateset.gates:
            printer.log("  frobenius norm diff of %s = %g" % (gateLabel,
              _tools.frobeniusdist(newGateset.gates[gateLabel],targetGateset.gates[gateLabel])), 2)
        for (rhoLbl,rhoV) in newGateset.preps.items():
            printer.log("  frobenius norm diff of %s = %g" % \
              (rhoLbl, _tools.frobeniusdist(rhoV,targetGateset.preps[rhoLbl])), 2)
        for (ELbl,Evec) in newGateset.effects.items():
            printer.log("  frobenius norm diff of %s = %g" % \
                (ELbl, _tools.frobeniusdist(Evec,targetGateset.effects[ELbl])), 2)
    else:
        printer.log('The resulting %s penalty is: %g' % (toGetTo, minSol.fun), 2)

    printer.log('The gauge matrix found (B^-1) is:\n' + str(gaugeMat) + '\n', 3)
    printer.log( 'The gauge-corrected gates are:\n' + str(newGateset), 3)
    printer.log('',2) #extra newline if we print messages at verbosity >= 2

    if returnAll:
        return minSol.fun, gaugeMat, newGateset
    else:
        return newGateset


#def optimize_unitary_gauge(gateset,dataset,verbosity,**kwargs):
#    """ Experimental -- works only for single qubit case:
#        Try to find a unitary that maximizes the norm of the logl gradient """
#
#    tol = kwargs.get('tol',1e-8)
#    method = kwargs.get('method','BFGS')
#    gateDim = gateset.get_dimension() # The dimension of the space: TODO = cleaner way
#    if verbosity > 1: print "\n--- Unitary Gauge Optimization ---"
#
#    def objective_func(v):
#      matM = single_qubit_gate( v[0], v[1], v[2] )
#      gs = gateset.copy()
#      gs.transform(matM)
#      ret = -_np.linalg.norm(_tools.dlogL(gs, dataset, **kwargs))
#      print "DEBUG: ",ret
#      return ret
#
#    startV = _np.array( [0,0,0] )
#    print_obj_func = _opt.create_obj_func_printer(objective_func)
#    minSol = _opt.minimize(objective_func,startV,method=method, tol=tol,
#                          callback = print_obj_func if verbosity > 2 else None)
#
#    gaugeMat = single_qubit_gate( minSol.x[0], minSol.x[1], minSol.x[2] )
#    newGateset = gateset.copy()
#    newGateset.transform(gaugeMat)
#    newGateset.log("Optimize Unitary Gauge To Max dLogL", { 'method': method, 'tol': tol } )
#
#    if verbosity > 1:
#        print 'The resulting norm(dLog) is: %g' % -minSol.fun
#        #print 'The gauge matrix found (B^-1) is:\n' + str(gaugeMat) + '\n'
#        #print 'The gauge-corrected gates are:\n' + str(newGateset)
#
#    return -minSol.fun, gaugeMat, newGateset

