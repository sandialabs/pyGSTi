from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Classes for constructing confidence regions """

import numpy       as _np
import scipy.stats as _stats
import warnings    as _warnings
import itertools   as _itertools
from .. import optimize as _opt

from .gateset import P_RANK_TOL
from .verbosityprinter import VerbosityPrinter

# NON-MARKOVIAN ERROR BARS
#Connection with Robin's notes:
#
# Robins notes: pg 21 : want to set radius delta'(alpha,r2)
#   via: lambda(G) = lambda(G_mle) + delta'
#
# Connecting with pyGSTi Hessian (H) calculations:
#   lambda(G) = 2(maxLogL - logL(G)) ~ chi2_k   (as defined in notes)
#   lambda(G_mle) = 2(maxLogL - logL(G_mle))
#
#  expand logL around max:
#    logL(G_mle + dx) ~= logL(G_mle) - 1/2 dx*H*dx (no first order term)
#
#  Thus, delta'
#    delta' = lambda(G) - lambda(G_mle) = -2(log(G)-log(G_mle))
#           = dx*H*dx ==> delta' is just like C1 or Ck scaling factors
#                         used for computing normal confidence regions
#    (recall delta' is computed as the alpha-th quantile of a
#     non-central chi^2_{K',r2} where K' = #of gateset params and
#     r2 = lambda(G_mle) - (K-K'), where K = #max-model params (~#gatestrings)
#     is the difference between the expected (mean) lambda (=K-K') and what
#     we actually observe (=lambda(G_mle)).
#
#TODO: add "type" argument to ConfidenceRegion constructor?
#   ==> "std" or "non-markovian"

class ConfidenceRegion(object):
    """
    Encapsulates a hessian-based confidence region in gate-set space.

    A ConfidenceRegion computes and stores the quadratic form for an approximate
    confidence region based on a confidence level and a hessian, typically of either
    loglikelihood function or its proxy, the chi2 function.
    """

    def __init__(self, gateset, hessian, confidenceLevel,
                 hessianProjection="std", tol=1e-6, maxiter=10000,
                 nonMarkRadiusSq=0, linresponse_mlgst_params=None):
        """
        Initializes a new ConfidenceRegion.

        Parameters
        ----------
        gateset : GateSet
            the gate set point estimate that maximizes the logl or minimizes
            the chi2, and marks the point in gateset-space where the Hessian
            has been evaluated.

        hessian : numpy array
            A nParams x nParams Hessian matrix, where nParams is the number
            of dimensions of gateset space, i.e. gateset.num_params().  This 
            can and must be None when `hessianProjection` equals
            `linear response` (see below).

        confidenceLevel : float
            The confidence level as a percentage, i.e. between 0 and 100.

        hessianProjection : string, optional
            Specifies how (and whether) to project the given hessian matrix
            onto a non-gauge space.  Allowed values are:

            - 'std' -- standard projection onto the space perpendicular to the
              gauge space.
            - 'none' -- no projection is performed.  Useful if the supplied
              hessian has already been projected.
            - 'optimal gate CIs' -- a lengthier projection process in which a
              numerical optimization is performed to find the non-gauge space
              which minimizes the (average) size of the confidence intervals
              corresponding to gate (as opposed to SPAM vector) parameters.
            - 'intrinsic error' -- compute separately the intrinsic error
              in the gate and spam GateSet parameters and set weighting metric
              based on their ratio.
            - 'linear response' -- obtain elements of the Hessian via the
              linear response of a "forcing term".  This requres a likelihood
              optimization for *every* computed error bar, but avoids pre-
              computation of the entire Hessian matrix, which can be 
              prohibitively costly on large parameter spaces.

        tol : float, optional
            Tolerance for optimal Hessian projection.  Only used when
            hessianProjection == 'optimal gate CIs'

        maxiter : int, optional
            Maximum iterations for optimal Hessian projection.  Only used when
            hessianProjection == 'optimal gate CIs'

        nonMarkRadiusSq : float, optional
            When non-zero, "a non-Markovian error region" is constructed using
            this value as the squared "non-markovian radius". This specifies the
            portion of 2*(max-log-likelihood - gateset-log-likelihood) that we
            attribute to non-Markovian errors (typically the previous
            difference minus it's expected value, the difference in number of
            parameters between the maximal and gateset models).  If set to
            zero (the default), a standard and thereby statistically rigorous
            conficence region is created.  Non-zero values should only be
            supplied if you really know what you're doing.

        linresponse_mlgst_params : dict, optional
            Only used when `hessianProjection == 'linear response'`, this 
            dictionary gives the arguments passed to the :func:`do_mlgst`
            calls used to compute linear-response error bars.
        """

        assert(not(hessian is None and linresponse_mlgst_params is None)), \
            "Must supply either a Hessian matrix or MLGST parameters"

        proj_non_gauge = gateset.get_nongauge_projector()
        self.nNonGaugeParams = _np.linalg.matrix_rank(proj_non_gauge, P_RANK_TOL)
        self.nGaugeParams = gateset.num_params() - self.nNonGaugeParams

        #Project Hessian onto non-gauge space
        if hessian is not None:
            if hessianProjection == 'none':
                projected_hessian = hessian
            elif hessianProjection == 'std':
                projected_hessian = _np.dot(proj_non_gauge, _np.dot(hessian, proj_non_gauge))
            elif hessianProjection == 'optimal gate CIs':
                projected_hessian = _optProjectionForGateCIs(gateset, hessian, self.nNonGaugeParams,
                                                             self.nGaugeParams, confidenceLevel,
                                                             "L-BFGS-B", maxiter, maxiter,
                                                             tol, verbosity=3) #verbosity for DEBUG
            elif hessianProjection == 'intrinsic error':
                projected_hessian = _optProjectionFromSplit(gateset, hessian, confidenceLevel,
                                                            verbosity=3) #verbosity for DEBUG
            elif hessianProjection == 'linear response':
                raise ValueError("'hessian' must be None when using the " +
                                 "'linear response' hessian projection type")
            else:
                raise ValueError("Invalid value of hessianProjection argument: %s" % hessianProjection)
        else:
            assert(hessianProjection == 'linear response')
            projected_hessian = None

        #Scale projected Hessian for desired confidence level => quadratic form for confidence region
        # assume hessian gives Fisher info, so asymptotically normal => confidence interval = +/- seScaleFctr * 1/sqrt(hessian)
        # where seScaleFctr gives the scaling factor for a normal distribution, i.e. integrating the
        # std normal distribution between -seScaleFctr and seScaleFctr == confidenceLevel/100 (as a percentage)
        assert(confidenceLevel > 0.0 and confidenceLevel < 100.0)
        if confidenceLevel < 1.0:
            _warnings.warn("You've specified a %f%% confidence interval, " % confidenceLevel \
                               + "which is usually small.  Be sure to specify this" \
                               + "number as a percentage in (0,100) and not a fraction in (0,1)." )

        # Get constants C such that xT*Hessian*x = C gives contour for the desired confidence region.
        #  C1 == Single DOF case: constant for a single-DOF likelihood, (or a profile likelihood in our case)
        #  Ck == Total DOF case: constant for a region of the likelihood as a function of *all non-gauge* gateset parameters
        if nonMarkRadiusSq == 0.0: #use == to test for *exact* zero floating pt value as herald
            C1 = _stats.chi2.ppf(confidenceLevel/100.0, 1)
            Ck = _stats.chi2.ppf(confidenceLevel/100.0, self.nNonGaugeParams)

              # Alt. method to get C1: square the result of a single gaussian (normal distribution)
              #Note: scipy's ppf gives inverse of cdf, so want to know where cdf == the leftover probability on left side
            seScaleFctr = -_stats.norm.ppf((1.0 - confidenceLevel/100.0)/2.0) #std error scaling factor for desired confidence region
            assert(_np.isclose(C1, seScaleFctr**2))

            # save quadratic form Q s.t. xT*Q*x = 1 gives confidence region using C1, i.e. a
            #  region appropriate for generating 1-D confidence intervals.
            if projected_hessian is not None:
                self.regionQuadcForm = projected_hessian / C1
            else:
                self.regionQuadcForm = None

            self.intervalScaling = _np.sqrt( Ck / C1 ) # multiplicative scaling required to convert intervals
                                                   # to those obtained using a full (using Ck) confidence region.
            self.stdIntervalScaling = 1.0 # multiplicative scaling required to convert intervals
                                          # to *standard* (e.g. not non-Mark.) intervals.
            self.stdRegionScaling = self.intervalScaling # multiplicative scaling required to convert intervals
                                                  # to those obtained using a full *standard* confidence region.

        else:
            C1 = _stats.ncx2.ppf(confidenceLevel/100.0, 1, nonMarkRadiusSq)
            Ck = _stats.ncx2.ppf(confidenceLevel/100.0, self.nNonGaugeParams, nonMarkRadiusSq)

            # save quadratic form Q s.t. xT*Q*x = 1 gives confidence region using C1, i.e. a
            #  region appropriate for generating 1-D confidence intervals.
            if projected_hessian is not None:
                self.regionQuadcForm = projected_hessian / C1
                self.regionQuadcForm *=  _np.sqrt(self.nNonGaugeParams) #make a *worst case* non-mark. region...
            else:
                self.regionQuadcForm = None

            self.intervalScaling = _np.sqrt( Ck / C1 ) # multiplicative scaling required to convert intervals
                                                   # to those obtained using a full (using Ck) confidence region.

            stdC1 = _stats.chi2.ppf(confidenceLevel/100.0, 1)
            stdCk = _stats.chi2.ppf(confidenceLevel/100.0, self.nNonGaugeParams)
            self.stdIntervalScaling = _np.sqrt( stdC1 / C1 ) # see above description
            self.stdRegionScaling = _np.sqrt( stdCk / C1 ) # see above description

            _warnings.warn("Non-Markovian error bars are experimental and" +
                           " cannot be interpreted as standard error bars." +
                           " Proceed with caution!")

        #print "DEBUG: C1 =",C1," Ck =",Ck," scaling =",self.intervalScaling

        if self.regionQuadcForm is not None:
            #Invert *non-gauge-part* of quadratic by eigen-decomposing ->
            #   inverting the non-gauge eigenvalues -> re-constructing via eigenvectors.
            # (Note that Hessian & quadc form mxs are symmetric so eigenvalues == singular values)
            evals,U = _np.linalg.eigh(self.regionQuadcForm)  # regionQuadcForm = U * diag(evals) * U.dag (b/c U.dag == inv(U) )
            Udag = _np.conjugate(_np.transpose(U))
    
              #invert only the non-gauge eigenvalues (those with ordering index >= nGaugeParams)
            orderInds = [ el[0] for el in sorted( enumerate(evals), key=lambda x: abs(x[1])) ] # ordering index of each eigenvalue
            invEvals = _np.zeros( evals.shape, evals.dtype )
            for i in orderInds[self.nGaugeParams:]:
                invEvals[i] = 1.0/evals[i]
            #print "nGaugeParams = ",self.nGaugeParams; print invEvals #DEBUG
    
              #re-construct "inverted" quadratic form
            invDiagQuadcForm = _np.diag( invEvals )
            self.invRegionQuadcForm = _np.dot(U, _np.dot(invDiagQuadcForm, Udag))
            self.U, self.Udag = U, Udag
        else:
            self.invRegionQuadcForm = None
            self.U = self.Udag = None

        #Store params and offsets of gateset for future use
        self.gateset_offsets = gateset.get_vector_offsets()

        #Store list of profile-likelihood confidence intervals
        #  which == sqrt(diagonal els) of invRegionQuadcForm
        if self.invRegionQuadcForm is not None:
            self.profLCI = [ _np.sqrt(abs(self.invRegionQuadcForm[k,k])) for k in range(len(evals))]
            self.profLCI = _np.array( self.profLCI, 'd' )
        else:
            self.profLCI = None

        self.gateset = gateset
        self.level = confidenceLevel #a percentage, i.e. btwn 0 and 100

        self.mlgst_params = linresponse_mlgst_params
        self._C1 = C1 #save for linear response scaling
        self.mlgst_evaltree_cache = {} #for _do_mlgst_base speedup

        #DEBUG
        #print "DEBUG HESSIAN:"
        #print "shape = ",hessian.shape
        #print "nGaugeParams = ",self.nGaugeParams
        #print "nNonGaugeParams = ",self.nNonGaugeParams
        ##print "Eval,InvEval = ",zip(evals,invEvals)
        #N = self.regionQuadcForm.shape[0]
        ##diagEls = [ self.regionQuadcForm[k,k] for k in range(N) ]
        #invdiagEls = sorted([ abs(self.invRegionQuadcForm[k,k]) for k in range(N) ])
        #evals = sorted( _np.abs(_np.linalg.eigvals( self.regionQuadcForm )) )
        #invEvals = sorted( _np.abs(_np.linalg.eigvals( self.invRegionQuadcForm )) )
        #print "index  sorted(abs(inv-diag))     sorted(eigenval)    sorted(inv-eigenval):"
        #for i,(invDiag,eigenval,invev) in enumerate(zip(invdiagEls,evals,invEvals)):
        #    print "%d %15g %15g %15g" % (i,invDiag,eigenval,invev)
        #
        #import sys
        #sys.exit()

    def has_hessian(self):
        """
        Returns whether or not the full Hessian has already been computed.

        When True, computation of confidence regions and intervals is
        fast, since the difficult work of computing the inverse Hessian is
        already done.  When False, a slower method must be used to estimate
        the necessary portion of the Hessian.  The result of this function
        is often used to decide whether or not to proceed with an error-bar
        computation.

        Returns
        -------
        bool
        """
        return bool(self.invRegionQuadcForm is not None)


    def get_gateset(self):
        """
        Retrieve the associated gate set.

        Returns
        -------
        GateSet
            the gate set marking the center location of this confidence region.
        """
        return self.gateset


    def get_profile_likelihood_confidence_intervals(self, label=None):
        """
        Retrieve the profile-likelihood confidence intervals for a specified
        gate set object (or all such intervals).

        Parameters
        ----------
        label : string, optional
            If not None, can be either a gate or SPAM vector label
            to specify the confidence intervals corresponding to gate, rhoVec,
            or EVec parameters respectively.  If None, then intervals corresponding
            to all of the gateset's parameters are returned.

        Returns
        -------
        numpy array
            One-dimensional array of (positive) interval half-widths which specify
            a symmetric confidence interval.
        """
        if label is None:
            return self.profLCI
        else:
            start,end = self.gateset_offsets[label]
            return self.profLCI[start:end]

    def get_gate_fn_confidence_interval(self, fnOfGate, gateLabel, eps=1e-7,
                                        returnFnVal=False, verbosity=0):
        """
        Compute the confidence interval for a function of a single gate.

        Parameters
        ----------
        fnOfGate : function
            A function which takes as its only argument a gate matrix.  The
            returned confidence interval is based on linearizing this function
            and propagating the gateset-space confidence region.

        gateLabel : string
            The label specifying which gate to use in evaluations of fnOfGate.

        eps : float, optional
            Step size used when taking finite-difference derivatives of fnOfGate.

        returnFnVal : bool, optional
            If True, return the value of fnOfGate along with it's confidence
            region half-widths.

        verbosity : int, optional
            Specifies level of detail in standard output.

        Returns
        -------
        df : float or numpy array
            Half-widths of confidence intervals for each of the elements
            in the float or array returned by fnOfGate.  Thus, shape of
            df matches that returned by fnOfGate.

        f0 : float or numpy array
            Only returned when returnFnVal == True. Value of fnOfGate
            at the gate specified by gateLabel.
        """

        nParams = self.gateset.num_params()

        gateObj = self.gateset.gates[gateLabel].copy() # copy because we add eps to this gate
        gateMx = _np.asarray(gateObj).copy()
        gpo = self.gateset_offsets[gateLabel][0] #starting "gate parameter offset"

        f0 = fnOfGate(gateMx) #function value at "base point" gateMx
        nGateParams = gateObj.num_params()
        gateVec0 = gateObj.to_vector()

        #Get finite difference derivative gradF that is shape (nParams, <shape of f0>)
        if type(f0) == float:
            gradF = _np.zeros( nParams, 'd' ) #assume all functions are real-valued for now...
        else:
            gradSize = (nParams,) + tuple(f0.shape)
            gradF = _np.zeros( gradSize, f0.dtype ) #assume all functions are real-valued for now...

        for i in range(nGateParams):
            gateVec = gateVec0.copy(); gateVec[i] += eps; gateObj.from_vector(gateVec)
            gradF[gpo + i] = ( fnOfGate( gateObj ) - f0 ) / eps

        return self._compute_df_from_gradF(gradF, f0, returnFnVal, verbosity)


    def get_prep_fn_confidence_interval(self, fnOfPrep, prepLabel, eps=1e-7,
                                        returnFnVal=False, verbosity=0):
        """
        Compute the confidence interval for a function of a single state prep.

        Parameters
        ----------
        fnOfPrep : function
            A function which takes as its only argument a prepration vector.  The
            returned confidence interval is based on linearizing this function
            and propagating the gateset-space confidence region.

        prepLabel : string
            The label specifying which preparation to use in evaluations of fnOfPrep.

        eps : float, optional
            Step size used when taking finite-difference derivatives of fnOfPrep.

        returnFnVal : bool, optional
            If True, return the value of fnOfPrep along with it's confidence
            region half-widths.

        verbosity : int, optional
            Specifies level of detail in standard output.

        Returns
        -------
        df : float or numpy array
            Half-widths of confidence intervals for each of the elements
            in the float or array returned by fnOfPrep.  Thus, shape of
            df matches that returned by fnOfPrep.

        f0 : float or numpy array
            Only returned when returnFnVal == True. Value of fnOfPrep
            at the state preparation specified by prepLabel.
        """

        nParams = self.gateset.num_params()

        prepObj = self.gateset.preps[prepLabel].copy() # copy because we add eps to this gate
        spamVec = _np.asarray(prepObj).copy()
        gpo = self.gateset_offsets[prepLabel][0] #starting "gateset parameter offset"

        f0 = fnOfPrep(spamVec) #function value at "base point"
        nPrepParams = prepObj.num_params()
        prepVec0 = prepObj.to_vector()

        #Get finite difference derivative gradF that is shape (nParams, <shape of f0>)
        if type(f0) == float:
            gradF = _np.zeros( nParams, 'd' ) #assume all functions are real-valued for now...
        else:
            gradSize = (nParams,) + tuple(f0.shape)
            gradF = _np.zeros( gradSize, f0.dtype ) #assume all functions are real-valued for now...

        for i in range(nPrepParams):
            prepVec = prepVec0.copy(); prepVec[i] += eps; prepObj.from_vector(prepVec)
            gradF[gpo + i] = ( fnOfPrep( prepObj ) - f0 ) / eps

        return self._compute_df_from_gradF(gradF, f0, returnFnVal, verbosity)


    def get_effect_fn_confidence_interval(self, fnOfEffect, effectLabel, eps=1e-7,
                                        returnFnVal=False, verbosity=0):
        """
        Compute the confidence interval for a function of a single POVM effect.

        Parameters
        ----------
        fnOfEffect : function
            A function which takes as its only argument a POVM vector.  The
            returned confidence interval is based on linearizing this function
            and propagating the gateset-space confidence region.

        effectLabel : string
            The label specifying which POVM to use in evaluations of fnOfEffect.

        eps : float, optional
            Step size used when taking finite-difference derivatives of fnOfEffect.

        returnFnVal : bool, optional
            If True, return the value of fnOfEffect along with it's confidence
            region half-widths.

        verbosity : int, optional
            Specifies level of detail in standard output.

        Returns
        -------
        df : float or numpy array
            Half-widths of confidence intervals for each of the elements
            in the float or array returned by fnOfEffect.  Thus, shape of
            df matches that returned by fnOfEffect.

        f0 : float or numpy array
            Only returned when returnFnVal == True. Value of fnOfEffect
            at the POVM effect specified by effectLabel.
        """

        nParams = self.gateset.num_params()

        effectObj = self.gateset.effects[effectLabel].copy() # copy because we add eps to this gate
        spamVec = _np.asarray(effectObj).copy()
        f0 = fnOfEffect(spamVec) #function value at "base point"

        if effectLabel not in self.gateset_offsets: #e.g. "remainder" is not...
            #Assume this effect label has not official "parameters" and just
            # return 0 as the confidence interval.
            return (0.0,f0) if returnFnVal else 0.0

        gpo = self.gateset_offsets[effectLabel][0] #starting "gateset parameter offset"
        nEffectParams = effectObj.num_params()
        effectVec0 = effectObj.to_vector()

        #Get finite difference derivative gradF that is shape (nParams, <shape of f0>)
        if type(f0) == float:
            gradF = _np.zeros( nParams, 'd' ) #assume all functions are real-valued for now...
        else:
            gradSize = (nParams,) + tuple(f0.shape)
            gradF = _np.zeros( gradSize, f0.dtype ) #assume all functions are real-valued for now...

        for i in range(nEffectParams):
            effectVec = effectVec0.copy(); effectVec[i] += eps; effectObj.from_vector(effectVec)
            gradF[gpo + i] = ( fnOfEffect( effectObj ) - f0 ) / eps

        return self._compute_df_from_gradF(gradF, f0, returnFnVal, verbosity)


    def get_gateset_fn_confidence_interval(self, fnOfGateset, eps=1e-7, returnFnVal=False, verbosity=0):
        """
        Compute the confidence interval for a function of a GateSet.

        Parameters
        ----------
        fnOfGateset : function
            A function which takes as its only argument a GateSet object.  The
            returned confidence interval is based on linearizing this function
            and propagating the gateset-space confidence region.

        eps : float, optional
            Step size used when taking finite-difference derivatives of fnOfGateset.

        returnFnVal : bool, optional
            If True, return the value of fnOfGateset along with it's confidence
            region half-widths.

        verbosity : int, optional
            Specifies level of detail in standard output.

        Returns
        -------
        df : float or numpy array
            Half-widths of confidence intervals for each of the elements
            in the float or array returned by fnOfGateset.  Thus, shape of
            df matches that returned by fnOfGateset.

        f0 : float or numpy array
            Only returned when returnFnVal == True. Value of fnOfGateset
            at the gate specified by gateLabel.
        """

        nParams = self.gateset.num_params()

        gateset = self.gateset.copy() # copy because we add eps to this gateset

        f0 = fnOfGateset(gateset) #function value at "base point" gateMx
        gatesetVec0 = gateset.to_vector()
        assert(len(gatesetVec0) == nParams)

        #Get finite difference derivative gradF that is shape (nParams, <shape of f0>)
        if type(f0) == float:
            gradF = _np.zeros( nParams, 'd' ) #assume all functions are real-valued for now...
        else:
            gradSize = (nParams,) + tuple(f0.shape)
            gradF = _np.zeros( gradSize, f0.dtype ) #assume all functions are real-valued for now...

        for i in range(nParams):
            gatesetVec = gatesetVec0.copy(); gatesetVec[i] += eps
            gateset.from_vector(gatesetVec)
            gradF[i] = ( fnOfGateset(gateset) - f0 ) / eps

        return self._compute_df_from_gradF(gradF, f0, returnFnVal, verbosity)


    def get_spam_fn_confidence_interval(self, fnOfSpamVecs, eps=1e-7, returnFnVal=False, verbosity=0):
        """
        Compute the confidence interval for a function of spam vectors.

        Parameters
        ----------
        fnOfSpamVecs : function
            A function which takes two arguments, rhoVecs and EVecs, each of which
            is a list of column vectors.  Note that the EVecs list will include
            *all* the effect vectors, including the a "compliment" vector.  The
            returned confidence interval is based on linearizing this function
            and propagating the gateset-space confidence region.

        eps : float, optional
            Step size used when taking finite-difference derivatives of fnOfSpamVecs.

        returnFnVal : bool, optional
            If True, return the value of fnOfSpamVecs along with it's confidence
            region half-widths.

        verbosity : int, optional
            Specifies level of detail in standard output.

        Returns
        -------
        df : float or numpy array
            Half-widths of confidence intervals for each of the elements
            in the float or array returned by fnOfSpamVecs.  Thus, shape of
            df matches that returned by fnOfSpamVecs.

        f0 : float or numpy array
            Only returned when returnFnVal == True. Value of fnOfSpamVecs.
        """
        nParams = self.gateset.num_params()

        f0 = fnOfSpamVecs(self.gateset.get_preps(), self.gateset.get_effects())
          #Note: .get_Evecs() can be different from .EVecs b/c the former includes compliment EVec

        #Get finite difference derivative gradF that is shape (nParams, <shape of f0>)
        if type(f0) == float:
            gradF = _np.zeros( nParams, 'd' ) #assume all functions are real-valued for now...
        else:
            gradSize = (nParams,) + tuple(f0.shape)
            gradF = _np.zeros( gradSize, f0.dtype ) #assume all functions are real-valued for now...


        gsEps = self.gateset.copy()

        #loop just over parameterized objects - don't use get_preps() here...
        for prepLabel,rhoVec in self.gateset.preps.items():
            nRhoParams = rhoVec.num_params()
            off = self.gateset_offsets[prepLabel][0]
            vec = rhoVec.to_vector()
            for i in range(nRhoParams):
                vecEps = vec.copy(); vecEps[i] += eps
                gsEps.preps[prepLabel].from_vector(vecEps) #update gsEps parameters
                gradF[off + i] = ( fnOfSpamVecs( gsEps.get_preps(),
                                   gsEps.get_effects() ) - f0 ) / eps
            gsEps.preps[prepLabel] = rhoVec.copy()  #reset gsEps (copy() just to be safe)

        #loop just over parameterized objects - don't use get_effects() here...
        for ELabel,EVec in self.gateset.effects.items():
            nEParams = EVec.num_params()
            off = self.gateset_offsets[ELabel][0]
            vec = EVec.to_vector()
            for i in range(nEParams):
                vecEps = vec.copy(); vecEps[i] += eps
                gsEps.effects[ELabel].from_vector(vecEps) #update gsEps parameters
                gradF[off + i] = ( fnOfSpamVecs( gsEps.get_preps(),
                                   gsEps.get_effects() ) - f0 ) / eps
            gsEps.effects[ELabel] = EVec.copy()  #reset gsEps (copy() just to be safe)

        return self._compute_df_from_gradF(gradF, f0, returnFnVal, verbosity)


    def _compute_df_from_gradF(self, gradF, f0, returnFnVal, verbosity):
        if self.regionQuadcForm is None:
            df = self._compute_df_from_gradF_linresponse(
                gradF, f0, verbosity)
        else:
            df = self._compute_df_from_gradF_hessian(
                gradF, f0, verbosity)
        return (df, f0) if returnFnVal else df


    def _compute_df_from_gradF_linresponse(self, gradF, f0, verbosity):
        from .. import algorithms as _alg
        assert(self.mlgst_params is not None)

        if hasattr(f0,'dtype') and f0.dtype == _np.dtype("complex"):
            raise NotImplementedError("Can't handle complex-valued functions yet")

        #massage gradF, which has shape (nParams,) + f0.shape
        # to that expected by _do_mlgst_base, which is
        # (flat_f0_size, nParams)
        if len(gradF.shape) == 1:
            gradF.shape = (1,gradF.shape[0])
        else:
            flatDim = _np.prod(f0.shape)
            gradF.shape = (gradF.shape[0], flatDim)
            gradF = _np.transpose(gradF) #now shape == (flatDim, nParams)
        assert(len(gradF.shape) == 2)

        mlgst_args = self.mlgst_params.copy()
        mlgst_args['startGateset'] = self.gateset
        mlgst_args['forcefn_grad'] = gradF
        mlgst_args['shiftFctr'] = 100.0
        mlgst_args['evaltree_cache'] = self.mlgst_evaltree_cache
        maxLogL, bestGS = _alg.core._do_mlgst_base(**mlgst_args)
        bestGS = _alg.gaugeopt_to_target(bestGS, self.gateset) #maybe more params here?
        norms = _np.array([_np.dot(gradF[i],gradF[i]) for i in range(gradF.shape[0])])
        delta2 = _np.abs(_np.dot(gradF, bestGS.to_vector() - self.gateset.to_vector()) \
            * _np.where(norms > 1e-10, 1.0/norms, 0.0))
        delta2 *= self._C1 #scaling appropriate for confidence level
        delta = _np.sqrt(delta2) # error^2 -> error

        if hasattr(f0,'shape'):
            delta.shape = f0.shape #reshape to un-flattened
        else:
            assert(type(f0) == float)
            delta = float(delta)

        return delta
        


    def _compute_df_from_gradF_hessian(self, gradF, f0, verbosity):
        """
        Internal function which computes error bars given an function value
        and gradient (using linear approx. to function)
        """

        #Compute df = sqrt( gradFu.dag * 1/D * gradFu )
        #  where regionQuadcForm = U * D * U.dag and gradFu = U.dag * gradF
        #  so df = sqrt( gradF.dag * U * 1/D * U.dag * gradF )
        #        = sqrt( gradF.dag * invRegionQuadcForm * gradF )

        printer = VerbosityPrinter.build_printer(verbosity)

        printer.log("gradF = %s" % gradF)

        if type(f0) == float:
            gradFdag = _np.conjugate(_np.transpose(gradF))

            #DEBUG
            #arg = _np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, gradF))
            #print "HERE: taking sqrt(abs(%s))" % arg

            df = _np.sqrt( abs(_np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, gradF))) )
        else:
            fDims = len(f0.shape)
            gradF = _np.rollaxis(gradF, 0, 1+fDims) # roll parameter axis to be the last index, preceded by f-shape
            df = _np.empty( f0.shape, f0.dtype)

            if f0.dtype == _np.dtype("complex"): #real and imaginary parts separately
                if fDims == 0: #same as float case above
                    gradFdag = _np.transpose(gradF)
                    df = _np.sqrt( abs( _np.dot(gradFdag.real, _np.dot(self.invRegionQuadcForm, gradF.real))) ) \
                        + 1j * _np.sqrt( abs( _np.dot(gradFdag.imag, _np.dot(self.invRegionQuadcForm, gradF.imag))) )
                elif fDims == 1:
                    for i in range(f0.shape[0]):
                        gradFdag = _np.transpose(gradF[i])
                        df[i] = _np.sqrt( abs( _np.dot(gradFdag.real, _np.dot(self.invRegionQuadcForm, gradF[i].real))) ) \
                            + 1j * _np.sqrt( abs( _np.dot(gradFdag.imag, _np.dot(self.invRegionQuadcForm, gradF[i].imag))) )
                elif fDims == 2:
                    for i in range(f0.shape[0]):
                        for j in range(f0.shape[1]):
                            gradFdag = _np.transpose(gradF[i,j])
                            df[i,j] = _np.sqrt( abs( _np.dot(gradFdag.real, _np.dot(self.invRegionQuadcForm, gradF[i,j].real))) ) \
                                + 1j * _np.sqrt( abs( _np.dot(gradFdag.imag, _np.dot(self.invRegionQuadcForm, gradF[i,j].imag))) )
                else:
                    raise ValueError("Unsupported number of dimensions returned by fnOfGate or fnOfGateset: %d" % fDims)

            else: #assume real -- so really don't need conjugate calls below
                if fDims == 0: #same as float case above
                    gradFdag = _np.conjugate(_np.transpose(gradF))

                    #DEBUG
                    #arg = _np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, gradF))
                    #print "HERE2: taking sqrt(abs(%s))" % arg

                    df = _np.sqrt( abs( _np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, gradF))) )
                elif fDims == 1:
                    for i in range(f0.shape[0]):
                        gradFdag = _np.conjugate(_np.transpose(gradF[i]))
                        df[i] = _np.sqrt( abs(_np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, gradF[i]))) )
                elif fDims == 2:
                    for i in range(f0.shape[0]):
                        for j in range(f0.shape[1]):
                            gradFdag = _np.conjugate(_np.transpose(gradF[i,j]))
                            df[i,j] = _np.sqrt( abs(_np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, gradF[i,j]))) )
                else:
                    raise ValueError("Unsupported number of dimensions returned by fnOfGate or fnOfGateset: %d" % fDims)

        printer.log("df = %s" % df)

        return df

    def __getstate__(self):
        #Return the state (for pickling) -- *don't* pickle any Comm objects
        to_pickle = self.__dict__.copy()
        if self.mlgst_params and self.mlgst_params.has_key("comm"):
            del self.mlgst_params['comm'] # one *cannot* pickle Comm objects
        return to_pickle



def _optProjectionForGateCIs(gateset, base_hessian, nNonGaugeParams, nGaugeParams,
                             level, method = "L-BFGS-B", maxiter = 10000,
                             maxfev = 10000, tol = 1e-6, verbosity = 0):
    printer = VerbosityPrinter.build_printer(verbosity)

    printer.log('', 3)
    printer.log("--- Hessian Projector Optimization for gate CIs (%s) ---" % method, 2, indentOffset=-1)

    def objective_func(vectorM):
        matM = vectorM.reshape( (nNonGaugeParams,nGaugeParams) )
        proj_extra = gateset.get_nongauge_projector(nonGaugeMixMx=matM)
        projected_hessian_ex = _np.dot(proj_extra, _np.dot(base_hessian, proj_extra))

        ci = ConfidenceRegion(gateset, projected_hessian_ex, level, hessianProjection="none")
        gateCIs = _np.concatenate( [ ci.get_profile_likelihood_confidence_intervals(gl).flatten()
                                     for gl in gateset.gates] )
        return _np.sqrt( _np.sum(gateCIs**2) )

    #Run Minimization Algorithm
    startM = _np.zeros( (nNonGaugeParams,nGaugeParams), 'd')
    x0 = startM.flatten()
    print_obj_func = _opt.create_obj_func_printer(objective_func)
    minSol = _opt.minimize(objective_func, x0,
                                    method=method, maxiter=maxiter,
                                    maxfev=maxfev, tol=tol,
                                    callback = print_obj_func if verbosity > 2 else None)

    mixMx = minSol.x.reshape( (nNonGaugeParams,nGaugeParams) )
    proj_extra = gateset.get_nongauge_projector(nonGaugeMixMx=mixMx)
    projected_hessian_ex = _np.dot(proj_extra, _np.dot(base_hessian, proj_extra))

    printer.log('The resulting min sqrt(sum(gateCIs**2)): %g' % minSol.fun, 2)

    return projected_hessian_ex


def _optProjectionFromSplit(gateset, base_hessian, level, verbosity = 0):
    printer = VerbosityPrinter.build_printer(verbosity)

    printer.log('', 3)
    printer.log("--- Hessian Projector Optimization from separate SPAM and Gate weighting ---", 2, indentOffset=-1)


    #get gate-intrinsic-error
    proj = gateset.get_nongauge_projector(itemWeights={'gates':1.0,'spam': 0.0})
    projected_hessian = _np.dot(proj, _np.dot(base_hessian, proj))
    ci = ConfidenceRegion(gateset, projected_hessian, level, hessianProjection="none")
    gateCIs = _np.concatenate( [ ci.get_profile_likelihood_confidence_intervals(gl).flatten()
                                     for gl in gateset.gates] )
    gate_intrinsic_err = _np.sqrt( _np.sum(gateCIs**2) )

    #get spam-intrinsic-error
    proj = gateset.get_nongauge_projector(itemWeights={'gates':0.0,'spam': 1.0})
    projected_hessian = _np.dot(proj, _np.dot(base_hessian, proj))
    ci = ConfidenceRegion(gateset, projected_hessian, level, hessianProjection="none")
    spamCIs = _np.concatenate( [ ci.get_profile_likelihood_confidence_intervals(gl).flatten()
                                     for sl in _itertools.chain(iter(gateset.preps),
                                                                iter(gateset.effects))] )
    spam_intrinsic_err = _np.sqrt( _np.sum(spamCIs**2) )

    ratio = gate_intrinsic_err / spam_intrinsic_err
    proj = gateset.get_nongauge_projector(itemWeights={'gates':1.0,'spam': ratio})
    projected_hessian = _np.dot(proj, _np.dot(base_hessian, proj))

    if printer.verbosity >= 2:
        #Create ci here just to extract #'s for print stmts
        ci = ConfidenceRegion(gateset, projected_hessian, level, hessianProjection="none")
        gateCIs = _np.concatenate( [ ci.get_profile_likelihood_confidence_intervals(gl).flatten()
                                         for gl in gateset.gates] )
        spamCIs = _np.concatenate( [ ci.get_profile_likelihood_confidence_intervals(gl).flatten()
                                         for sl in _itertools.chain(iter(gateset.preps),
                                                                    iter(gateset.effects))] )
        gate_err = _np.sqrt( _np.sum(gateCIs**2) )
        spam_err = _np.sqrt( _np.sum(spamCIs**2) )
        printer.log('Resulting intrinsic errors: %g (gates), %g (spam)' %
                    (gate_intrinsic_err, spam_intrinsic_err), 2)
        printer.log('Resulting sqrt(sum(gateCIs**2)): %g' % gate_err, 2)
        printer.log('Resulting sqrt(sum(spamCIs**2)): %g' % spam_err, 2)

    return projected_hessian
