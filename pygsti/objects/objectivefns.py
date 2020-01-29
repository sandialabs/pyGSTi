""" Defines objective-function objects """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import time as _time
import numpy as _np

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .. import optimize as _opt
from ..tools import listtools as _lt


class ObjectiveFunction(object):
    pass


#NOTE on chi^2 expressions:
#in general case:   chi^2 = sum (p_i-f_i)^2/p_i  (for i summed over outcomes)
#in 2-outcome case: chi^2 = (p+ - f+)^2/p+ + (p- - f-)^2/p-
#                         = (p - f)^2/p + (1-p - (1-f))^2/(1-p)
#                         = (p - f)^2 * (1/p + 1/(1-p))
#                         = (p - f)^2 * ( ((1-p) + p)/(p*(1-p)) )
#                         = 1/(p*(1-p)) * (p - f)^2

class Chi2Function(ObjectiveFunction):

    def __init__(self, mdl, evTree, lookup, circuitsToUse, opLabelAliases, regularizeFactor, cptp_penalty_factor,
                 spam_penalty_factor, cntVecMx, N, minProbClipForWeighting, probClipInterval, wrtBlkSize,
                 gthrMem, check=False, check_jacobian=False, comm=None, profiler=None, verbosity=0):

        from ..tools import slicetools as _slct

        self.mdl = mdl
        self.evTree = evTree
        self.lookup = lookup
        self.circuitsToUse = circuitsToUse
        self.comm = comm
        self.profiler = profiler
        self.check = check
        self.check_jacobian = check_jacobian

        KM = evTree.num_final_elements()  # shorthand for combined spam+circuit dimension
        vec_gs_len = mdl.num_params()
        self.printer = _VerbosityPrinter.build_printer(verbosity, comm)
        self.opBasis = mdl.basis

        #Compute "extra" (i.e. beyond the (circuit,spamlabel)) rows of jacobian
        self.ex = 0
        if regularizeFactor != 0:
            self.ex = vec_gs_len
        else:
            if cptp_penalty_factor != 0: self.ex += _cptp_penalty_size(mdl)
            if spam_penalty_factor != 0: self.ex += _spam_penalty_size(mdl)

        self.KM = KM
        self.vec_gs_len = vec_gs_len
        self.regularizeFactor = regularizeFactor
        self.cptp_penalty_factor = cptp_penalty_factor
        self.spam_penalty_factor = spam_penalty_factor
        self.minProbClipForWeighting = minProbClipForWeighting
        self.probClipInterval = probClipInterval
        self.wrtBlkSize = wrtBlkSize
        self.gthrMem = gthrMem

        #  Allocate peristent memory
        #  (must be AFTER possible operation sequence permutation by
        #   tree and initialization of dsCircuitsToUse)
        self.probs = _np.empty(KM, 'd')
        self.jac = _np.empty((KM + self.ex, vec_gs_len), 'd')

        #Detect omitted frequences (assumed to be 0) so we can compute chi2 correctly
        self.firsts = []; self.indicesOfCircuitsWithOmittedData = []
        for i, c in enumerate(circuitsToUse):
            lklen = _slct.length(lookup[i])
            if 0 < lklen < mdl.get_num_outcomes(c):
                self.firsts.append(_slct.as_array(lookup[i])[0])
                self.indicesOfCircuitsWithOmittedData.append(i)
        if len(self.firsts) > 0:
            self.firsts = _np.array(self.firsts, 'i')
            self.indicesOfCircuitsWithOmittedData = _np.array(self.indicesOfCircuitsWithOmittedData, 'i')
            self.dprobs_omitted_rowsum = _np.empty((len(self.firsts), vec_gs_len), 'd')
            self.printer.log("SPARSE DATA: %d of %d rows have sparse data" % (len(self.firsts), len(circuitsToUse)))
        else:
            self.firsts = None  # no omitted probs

        self.cntVecMx = cntVecMx
        self.N = N
        self.f = cntVecMx / N
        self.maxCircuitLength = max([len(x) for x in circuitsToUse])

        if self.printer.verbosity < 4:  # Fast versions of functions
            if regularizeFactor == 0 and cptp_penalty_factor == 0 and spam_penalty_factor == 0 \
               and mdl.get_simtype() != "termgap":
                # Fast un-regularized version
                self.fn = self.simple_chi2
                self.jfn = self.simple_jac

            elif regularizeFactor != 0:
                # Fast regularized version
                assert(cptp_penalty_factor == 0), "Cannot have regularizeFactor and cptp_penalty_factor != 0"
                assert(spam_penalty_factor == 0), "Cannot have regularizeFactor and spam_penalty_factor != 0"
                self.fn = self.regularized_chi2
                self.jfn = self.regularized_jac

            elif mdl.get_simtype() == "termgap":
                assert(cptp_penalty_factor == 0), "Cannot have termgap_pentalty_factor and cptp_penalty_factor != 0"
                assert(spam_penalty_factor == 0), "Cannot have termgap_pentalty_factor and spam_penalty_factor != 0"
                self.fn = self.termgap_chi2
                self.jfn = self.simple_jac

            else:  # cptp_pentalty_factor != 0 and/or spam_pentalty_factor != 0
                assert(regularizeFactor == 0), "Cannot have regularizeFactor and other penalty factors > 0"
                self.fn = self.penalized_chi2
                self.jfn = self.penalized_jac

        else:  # Verbose (DEBUG) version of objective_func
            if mdl.get_simtype() == "termgap":
                raise NotImplementedError("Still need to add termgap support to verbose chi2!")
            self.fn = self.verbose_chi2
            self.jfn = self.verbose_jac

    def get_weights(self, p):
        cp = _np.clip(p, self.minProbClipForWeighting, 1 - self.minProbClipForWeighting)
        return _np.sqrt(self.N / cp)  # nSpamLabels x nCircuits array (K x M)

    def get_dweights(self, p, wts):  # derivative of weights w.r.t. p
        cp = _np.clip(p, self.minProbClipForWeighting, 1 - self.minProbClipForWeighting)
        dw = -0.5 * wts / cp   # nSpamLabels x nCircuits array (K x M)
        dw[_np.logical_or(p < self.minProbClipForWeighting, p > (1 - self.minProbClipForWeighting))] = 0.0
        return dw

    def update_v_for_omitted_probs(self, v, probs):
        # if i-th circuit has omitted probs, have sqrt( N*(p_i-f_i)^2/p_i + sum_k(N*p_k) )
        # so we need to take sqrt( v_i^2 + N*sum_k(p_k) )
        omitted_probs = 1.0 - _np.array([_np.sum(probs[self.lookup[i]])
                                         for i in self.indicesOfCircuitsWithOmittedData])
        clipped_oprobs = _np.clip(omitted_probs, self.minProbClipForWeighting, 1 - self.minProbClipForWeighting)
        v[self.firsts] = _np.sqrt(v[self.firsts]**2 + self.N[self.firsts] * omitted_probs**2 / clipped_oprobs)

    def update_dprobs_for_omitted_probs(self, dprobs, probs, weights, dprobs_omitted_rowsum):
        # with omitted terms, new_obj = sqrt( obj^2 + corr ) where corr = N*omitted_p^2/clipped_omitted_p
        # so then d(new_obj) = 1/(2*new_obj) *( 2*obj*dobj + dcorr )*domitted_p where dcorr = N when not clipped
        #    and 2*N*omitted_p/clip_bound * domitted_p when clipped
        v = (probs - self.f) * weights
        omitted_probs = 1.0 - _np.array([_np.sum(probs[self.lookup[i]])
                                         for i in self.indicesOfCircuitsWithOmittedData])
        clipped_oprobs = _np.clip(omitted_probs, self.minProbClipForWeighting, 1 - self.minProbClipForWeighting)
        dprobs_factor_omitted = _np.where(omitted_probs == clipped_oprobs, self.N[self.firsts],
                                          2 * self.N[self.firsts] * omitted_probs / clipped_oprobs)
        fullv = _np.sqrt(v[self.firsts]**2 + self.N[self.firsts] * omitted_probs**2 / clipped_oprobs)
        # avoid NaNs when both fullv and v[firsts] are zero - result should be *zero* in this case
        fullv[v[self.firsts] == 0.0] = 1.0
        dprobs[self.firsts, :] = (0.5 / fullv[:, None]) * (
            2 * v[self.firsts, None] * dprobs[self.firsts, :]
            - dprobs_factor_omitted[:, None] * dprobs_omitted_rowsum)

    #Objective Function

    def simple_chi2(self, vectorGS):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval, self.check, self.comm)
        v = (self.probs - self.f) * self.get_weights(self.probs)  # dims K x M (K = nSpamLabels, M = nCircuits)

        if self.firsts is not None:
            self.update_v_for_omitted_probs(v, self.probs)

        self.profiler.add_time("do_mc2gst: OBJECTIVE", tm)
        assert(v.shape == (self.KM,))  # reshape ensuring no copy is needed
        return v

    def termgap_chi2(self, vectorGS, oob_check=False):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval, self.check, self.comm)

        if oob_check:
            if not self.mdl.bulk_probs_paths_are_sufficient(self.evTree,
                                                            self.probs,
                                                            self.comm,
                                                            memLimit=None,
                                                            verbosity=1):
                raise ValueError("Out of bounds!")  # signals LM optimizer

        v = (self.probs - self.f) * self.get_weights(self.probs)  # dims K x M (K = nSpamLabels, M = nCircuits)

        if self.firsts is not None:
            self.update_v_for_omitted_probs(v, self.probs)

        self.profiler.add_time("do_mc2gst: OBJECTIVE", tm)
        assert(v.shape == (self.KM,))  # reshape ensuring no copy is needed
        return v

    def regularized_chi2(self, vectorGS):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval, self.check, self.comm)
        weights = self.get_weights(self.probs)
        v = (self.probs - self.f) * weights  # dim KM (K = nSpamLabels, M = nCircuits)

        if self.firsts is not None:
            self.update_v_for_omitted_probs(v, self.probs)

        gsVecNorm = self.regularizeFactor * _np.array([max(0, absx - 1.0) for absx in map(abs, vectorGS)], 'd')
        self.profiler.add_time("do_mc2gst: OBJECTIVE", tm)
        return _np.concatenate((v.reshape([self.KM]), gsVecNorm))

    def penalized_chi2(self, vectorGS):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval, self.check, self.comm)
        weights = self.get_weights(self.probs)
        v = (self.probs - self.f) * weights  # dims K x M (K = nSpamLabels, M = nCircuits)

        if self.firsts is not None:
            self.update_v_for_omitted_probs(v, self.probs)

        if self.cptp_penalty_factor > 0:
            cpPenaltyVec = _cptp_penalty(self.mdl, self.cptp_penalty_factor, self.opBasis)
        else: cpPenaltyVec = []  # so concatenate ignores

        if self.spam_penalty_factor > 0:
            spamPenaltyVec = _spam_penalty(self.mdl, self.spam_penalty_factor, self.opBasis)
        else: spamPenaltyVec = []  # so concatenate ignores

        self.profiler.add_time("do_mc2gst: OBJECTIVE", tm)
        return _np.concatenate((v, cpPenaltyVec, spamPenaltyVec))

    def verbose_chi2(self, vectorGS):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval, self.check, self.comm)
        weights = self.get_weights(self.probs)

        v = (self.probs - self.f) * weights
        if self.firsts is not None:
            self.update_v_for_omitted_probs(v, self.probs)

        chisq = _np.sum(v * v)

        nClipped = len((_np.logical_or(self.probs < self.minProbClipForWeighting,
                                       self.probs > (1 - self.minProbClipForWeighting))).nonzero()[0])
        self.printer.log("MC2-OBJ: chi2=%g\n" % chisq
                         + "         p in (%g,%g)\n" % (_np.min(self.probs), _np.max(self.probs))
                         + "         weights in (%g,%g)\n" % (_np.min(weights), _np.max(weights))
                         + "         mdl in (%g,%g)\n" % (_np.min(vectorGS), _np.max(vectorGS))
                         + "         maxLen = %d, nClipped=%d" % (self.maxCircuitLength, nClipped), 4)

        assert((self.cptp_penalty_factor == 0 and self.spam_penalty_factor == 0) or self.regularizeFactor == 0), \
            "Cannot have regularizeFactor and other penalty factors != 0"
        if self.regularizeFactor != 0:
            gsVecNorm = self.regularizeFactor * _np.array([max(0, absx - 1.0) for absx in map(abs, vectorGS)], 'd')
            self.profiler.add_time("do_mc2gst: OBJECTIVE", tm)
            return _np.concatenate((v, gsVecNorm))
        elif self.cptp_penalty_factor != 0 or self.spam_penalty_factor != 0:
            if self.cptp_penalty_factor != 0:
                cpPenaltyVec = _cptp_penalty(self.mdl, self.cptp_penalty_factor, self.opBasis)
            else: cpPenaltyVec = []

            if self.spam_penalty_factor != 0:
                spamPenaltyVec = _spam_penalty(self.mdl, self.spam_penalty_factor, self.opBasis)
            else: spamPenaltyVec = []
            self.profiler.add_time("do_mc2gst: OBJECTIVE", tm)
            return _np.concatenate((v, cpPenaltyVec, spamPenaltyVec))
        else:
            self.profiler.add_time("do_mc2gst: OBJECTIVE", tm)
            assert(v.shape == (self.KM,))
            return v

    # Jacobian function
    def simple_jac(self, vectorGS):
        tm = _time.time()
        dprobs = self.jac.view()  # avoid mem copying: use jac mem for dprobs
        dprobs.shape = (self.KM, self.vec_gs_len)
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_dprobs(dprobs, self.evTree,
                                  prMxToFill=self.probs, clipTo=self.probClipInterval,
                                  check=self.check, comm=self.comm, wrtBlockSize=self.wrtBlkSize,
                                  profiler=self.profiler, gatherMemLimit=self.gthrMem)

        #DEBUG TODO REMOVE - test dprobs to make sure they look right.
        #EPS = 1e-7
        #db_probs = _np.empty(self.probs.shape, 'd')
        #db_probs2 = _np.empty(self.probs.shape, 'd')
        #db_dprobs = _np.empty(dprobs.shape, 'd')
        #self.mdl.bulk_fill_probs(db_probs, self.evTree, self.probClipInterval, self.check, self.comm)
        #for i in range(self.vec_gs_len):
        #    vectorGS_eps = vectorGS.copy()
        #    vectorGS_eps[i] += EPS
        #    self.mdl.from_vector(vectorGS_eps)
        #    self.mdl.bulk_fill_probs(db_probs2, self.evTree, self.probClipInterval, self.check, self.comm)
        #    db_dprobs[:,i] = (db_probs2 - db_probs) / EPS
        #if _np.linalg.norm(dprobs - db_dprobs)/dprobs.size > 1e-6:
        #    #assert(False), "STOP: %g" % (_np.linalg.norm(dprobs - db_dprobs)/db_dprobs.size)
        #    print("DB: dprobs per el mismatch = ",_np.linalg.norm(dprobs - db_dprobs)/db_dprobs.size)
        #self.mdl.from_vector(vectorGS)
        #dprobs[:,:] = db_dprobs[:,:]

        if self.firsts is not None:
            for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.lookup[i], :], axis=0)

        weights = self.get_weights(self.probs)
        dprobs *= (weights + (self.probs - self.f) * self.get_dweights(self.probs, weights))[:, None]
        # (KM,N) * (KM,1)   (N = dim of vectorized model)
        # this multiply also computes jac, which is just dprobs
        # with a different shape (jac.shape == [KM,vec_gs_len])

        if self.firsts is not None:
            self.update_dprobs_for_omitted_probs(dprobs, self.probs, weights, self.dprobs_omitted_rowsum)

        if self.check_jacobian: _opt.check_jac(lambda v: self.simple_chi2(
            v), vectorGS, self.jac, tol=1e-3, eps=1e-6, errType='abs')  # TO FIX

        # dpr has shape == (nCircuits, nDerivCols), weights has shape == (nCircuits,)
        # return shape == (nCircuits, nDerivCols) where ret[i,j] = dP[i,j]*(weights+dweights*(p-f))[i]
        self.profiler.add_time("do_mc2gst: JACOBIAN", tm)
        return self.jac

    def regularized_jac(self, vectorGS):
        tm = _time.time()
        dprobs = self.jac[0:self.KM, :]  # avoid mem copying: use jac mem for dprobs
        dprobs.shape = (self.KM, self.vec_gs_len)
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_dprobs(dprobs, self.evTree,
                                  prMxToFill=self.probs, clipTo=self.probClipInterval,
                                  check=self.check, comm=self.comm, wrtBlockSize=self.wrtBlkSize,
                                  profiler=self.profiler, gatherMemLimit=self.gthrMem)
        if self.firsts is not None:
            for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.lookup[i], :], axis=0)
        weights = self.get_weights(self.probs)
        dprobs *= (weights + (self.probs - self.f) * self.get_dweights(self.probs, weights))[:, None]
        # (KM,N) * (KM,1)   (N = dim of vectorized model)
        # Note: this also computes jac[0:KM,:]

        if self.firsts is not None:
            self.update_dprobs_for_omitted_probs(dprobs, self.probs, weights, self.dprobs_omitted_rowsum)

        gsVecGrad = _np.diag([(self.regularizeFactor * _np.sign(x) if abs(x) > 1.0 else 0.0)
                              for x in vectorGS])  # (N,N)
        self.jac[self.KM:, :] = gsVecGrad  # jac.shape == (KM+N,N)

        if self.check_jacobian: _opt.check_jac(lambda v: self.regularized_chi2(
            v), vectorGS, self.jac, tol=1e-3, eps=1e-6, errType='abs')

        # dpr has shape == (nCircuits, nDerivCols), gsVecGrad has shape == (nDerivCols, nDerivCols)
        # return shape == (nCircuits+nDerivCols, nDerivCols)
        self.profiler.add_time("do_mc2gst: JACOBIAN", tm)
        return self.jac

    def penalized_jac(self, vectorGS):  # Fast cptp-penalty version
        tm = _time.time()
        dprobs = self.jac[0:self.KM, :]  # avoid mem copying: use jac mem for dprobs
        dprobs.shape = (self.KM, self.vec_gs_len)
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_dprobs(dprobs, self.evTree,
                                  prMxToFill=self.probs, clipTo=self.probClipInterval,
                                  check=self.check, comm=self.comm, wrtBlockSize=self.wrtBlkSize,
                                  profiler=self.profiler, gatherMemLimit=self.gthrMem)
        if self.firsts is not None:
            for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.lookup[i], :], axis=0)
        weights = self.get_weights(self.probs)
        dprobs *= (weights + (self.probs - self.f) * self.get_dweights(self.probs, weights))[:, None]
        # (KM,N) * (KM,1)   (N = dim of vectorized model)
        # Note: this also computes jac[0:KM,:]

        if self.firsts is not None:
            self.update_dprobs_for_omitted_probs(dprobs, self.probs, weights, self.dprobs_omitted_rowsum)

        off = 0
        if self.cptp_penalty_factor > 0:
            off += _cptp_penalty_jac_fill(
                self.jac[self.KM + off:, :], self.mdl, self.cptp_penalty_factor, self.opBasis)
        if self.spam_penalty_factor > 0:
            off += _spam_penalty_jac_fill(
                self.jac[self.KM + off:, :], self.mdl, self.spam_penalty_factor, self.opBasis)

        if self.check_jacobian: _opt.check_jac(lambda v: self.penalized_chi2(
            v), vectorGS, self.jac, tol=1e-3, eps=1e-6, errType='abs')
        self.profiler.add_time("do_mc2gst: JACOBIAN", tm)
        return self.jac

    def verbose_jac(self, vectorGS):
        tm = _time.time()
        dprobs = self.jac[0:self.KM, :]  # avoid mem copying: use jac mem for dprobs
        dprobs.shape = (self.KM, self.vec_gs_len)
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_dprobs(dprobs, self.evTree,
                                  prMxToFill=self.probs, clipTo=self.probClipInterval,
                                  check=self.check, comm=self.comm, wrtBlockSize=self.wrtBlkSize,
                                  profiler=self.profiler, gatherMemLimit=self.gthrMem)
        if self.firsts is not None:
            for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.lookup[i], :], axis=0)

        weights = self.get_weights(self.probs)

        #Attempt to control leastsq by zeroing clipped weights -- this doesn't seem to help (nor should it)
        #weights[ _np.logical_or(pr < minProbClipForWeighting, pr > (1-minProbClipForWeighting)) ] = 0.0

        dPr_prefactor = (weights + (self.probs - self.f) * self.get_dweights(self.probs, weights))  # (KM)
        dprobs *= dPr_prefactor[:, None]  # (KM,N) * (KM,1) = (KM,N)  (N = dim of vectorized model)

        if self.firsts is not None:
            self.update_dprobs_for_omitted_probs(dprobs, self.probs, weights, self.dprobs_omitted_rowsum)

        if self.regularizeFactor != 0:
            gsVecGrad = _np.diag([(self.regularizeFactor * _np.sign(x) if abs(x) > 1.0 else 0.0) for x in vectorGS])
            self.jac[self.KM:, :] = gsVecGrad  # jac.shape == (KM+N,N)

        else:
            off = 0
            if self.cptp_penalty_factor != 0:
                off += _cptp_penalty_jac_fill(self.jac[self.KM + off:, :], self.mdl, self.cptp_penalty_factor,
                                              self.opBasis)

                if self.spam_penalty_factor != 0:
                    off += _spam_penalty_jac_fill(self.jac[self.KM + off:, :], self.mdl, self.spam_penalty_factor,
                                                  self.opBasis)

        # Zero-out insignificant entries in jacobian -- seemed to help some, but leaving this out,
        # thinking less complicated == better
        #absJac = _np.abs(jac);  maxabs = _np.max(absJac)
        #jac[ absJac/maxabs < 5e-8 ] = 0.0

        #Rescale jacobian so it's not too large -- an attempt to fix wild leastsq behavior but didn't help
        #if maxabs > 1e7:
        #  print "Rescaling jacobian to 1e7 maxabs"
        #  jac = (jac / maxabs) * 1e7

        #U,s,V = _np.linalg.svd(jac)
        #print "DEBUG: s-vals of jac %s = " % (str(jac.shape)), s

        nClipped = len((_np.logical_or(self.probs < self.minProbClipForWeighting,
                                       self.probs > (1 - self.minProbClipForWeighting))).nonzero()[0])
        self.printer.log("MC2-JAC: jac in (%g,%g)\n" % (_np.min(self.jac), _np.max(self.jac))
                         + "         pr in (%g,%g)\n" % (_np.min(self.probs), _np.max(self.probs))
                         + "         dpr in (%g,%g)\n" % (_np.min(dprobs), _np.max(dprobs))
                         + "         prefactor in (%g,%g)\n" % (_np.min(dPr_prefactor), _np.max(dPr_prefactor))
                         + "         mdl in (%g,%g)\n" % (_np.min(vectorGS), _np.max(vectorGS))
                         + "         maxLen = %d, nClipped = %d" % (self.maxCircuitLength, nClipped), 4)

        if self.check_jacobian:
            errSum, errs, fd_jac = _opt.check_jac(lambda v: self.verbose_chi2(
                v), vectorGS, self.jac, tol=1e-3, eps=1e-6, errType='abs')
            self.printer.log("Jacobian has error %g and %d of %d indices with error > tol" %
                             (errSum, len(errs), self.jac.shape[0] * self.jac.shape[1]), 4)
            if len(errs) > 0:
                i, j = errs[0][0:2]; maxabs = _np.max(_np.abs(self.jac))
                self.printer.log(" ==> Worst index = %d,%d. p=%g,  Analytic jac = %g, Fwd Diff = %g" %
                                 (i, j, self.probs[i], self.jac[i, j], fd_jac[i, j]), 4)
                self.printer.log(" ==> max err = ", errs[0][2], 4)
                self.printer.log(" ==> max err/max = ", max([x[2] / maxabs for x in errs]), 4)

        self.profiler.add_time("do_mc2gst: JACOBIAN", tm)
        return self.jac


class FreqWeightedChi2Function(Chi2Function):

    def __init__(self, mdl, evTree, lookup, circuitsToUse, opLabelAliases, regularizeFactor, cptp_penalty_factor,
                 spam_penalty_factor, cntVecMx, N, fweights, minProbClipForWeighting, probClipInterval, wrtBlkSize,
                 gthrMem, check=False, check_jacobian=False, comm=None, profiler=None, verbosity=0):

        Chi2Function.__init__(self, mdl, evTree, lookup, circuitsToUse, opLabelAliases, regularizeFactor,
                              cptp_penalty_factor, spam_penalty_factor, cntVecMx, N, minProbClipForWeighting,
                              probClipInterval, wrtBlkSize, gthrMem, check, check_jacobian, comm, profiler, verbosity=0)
        self.fweights = fweights
        self.z = _np.zeros(self.KM, 'd')

    def _get_weights(self, p):
        return self.fweights

    def _get_dweights(self, p, wts):
        return self.z


class TimeDependentChi2Function(ObjectiveFunction):

    #This objective function can handle time-dependent circuits - that is, circuitsToUse are treated as
    # potentially time-dependent and mdl as well.  For now, we don't allow any regularization or penalization
    # in this case.
    def __init__(self, mdl, evTree, lookup, circuitsToUse, opLabelAliases, regularizeFactor, cptp_penalty_factor,
                 spam_penalty_factor, dataset, dsCircuitsToUse, minProbClipForWeighting, probClipInterval, wrtBlkSize,
                 gthrMem, check=False, check_jacobian=False, comm=None, profiler=None, verbosity=0):

        assert(regularizeFactor == 0 and cptp_penalty_factor == 0 and spam_penalty_factor == 0), \
            "Cannot apply regularization or penalization in time-dependent chi2 case (yet)"

        from ..tools import slicetools as _slct

        self.mdl = mdl
        self.evTree = evTree
        self.lookup = lookup
        self.dataset = dataset
        self.dsCircuitsToUse = dsCircuitsToUse
        self.circuitsToUse = circuitsToUse
        self.num_total_outcomes = [mdl.get_num_outcomes(c) for c in circuitsToUse]  # for sparse data detection
        self.comm = comm
        self.profiler = profiler
        self.check = check
        self.check_jacobian = check_jacobian

        KM = evTree.num_final_elements()  # shorthand for combined spam+circuit dimension
        vec_gs_len = mdl.num_params()
        self.printer = _VerbosityPrinter.build_printer(verbosity, comm)
        self.opBasis = mdl.basis

        #Compute "extra" (i.e. beyond the (circuit,spamlabel)) rows of jacobian
        self.ex = 0
        self.KM = KM
        self.vec_gs_len = vec_gs_len
        #self.regularizeFactor = regularizeFactor
        #self.cptp_penalty_factor = cptp_penalty_factor
        #self.spam_penalty_factor = spam_penalty_factor
        self.minProbClipForWeighting = minProbClipForWeighting
        self.probClipInterval = probClipInterval
        self.wrtBlkSize = wrtBlkSize
        self.gthrMem = gthrMem

        #  Allocate peristent memory
        #  (must be AFTER possible operation sequence permutation by
        #   tree and initialization of dsCircuitsToUse)
        self.v = _np.empty(KM, 'd')
        self.jac = _np.empty((KM + self.ex, vec_gs_len), 'd')

        #REMOVE: these are time dependent now...
        #self.cntVecMx = cntVecMx
        #self.N = N
        #self.f = cntVecMx / N
        self.maxCircuitLength = max([len(x) for x in circuitsToUse])

        # Fast un-regularized version
        self.fn = self.simple_chi2
        self.jfn = self.simple_jac

    def get_weights(self, p):
        cp = _np.clip(p, self.minProbClipForWeighting, 1 - self.minProbClipForWeighting)
        return _np.sqrt(self.N / cp)  # nSpamLabels x nCircuits array (K x M)

    def get_dweights(self, p, wts):  # derivative of weights w.r.t. p
        cp = _np.clip(p, self.minProbClipForWeighting, 1 - self.minProbClipForWeighting)
        dw = -0.5 * wts / cp   # nSpamLabels x nCircuits array (K x M)
        dw[_np.logical_or(p < self.minProbClipForWeighting, p > (1 - self.minProbClipForWeighting))] = 0.0
        return dw

    #Objective Function
    def simple_chi2(self, vectorGS):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        fsim = self.mdl._fwdsim()
        v = self.v
        fsim.bulk_fill_timedep_chi2(v, self.evTree, self.dsCircuitsToUse, self.num_total_outcomes,
                                    self.dataset, self.minProbClipForWeighting, self.probClipInterval, self.comm)
        #self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval, self.check, self.comm)
        #v = (self.probs - self.f) * self.get_weights(self.probs)  # dims K x M (K = nSpamLabels, M = nCircuits)
        self.profiler.add_time("do_mc2gst: OBJECTIVE", tm)
        assert(v.shape == (self.KM,))  # reshape ensuring no copy is needed
        return v.copy()  # copy() needed for FD deriv, and we don't need to be stingy w/memory at objective fn level

    # Jacobian function
    def simple_jac(self, vectorGS):
        tm = _time.time()
        dprobs = self.jac.view()  # avoid mem copying: use jac mem for dprobs
        dprobs.shape = (self.KM, self.vec_gs_len)
        self.mdl.from_vector(vectorGS)
        #self.mdl.bulk_fill_dprobs(dprobs, self.evTree,
        #                          prMxToFill=self.probs, clipTo=self.probClipInterval,
        #                          check=self.check, comm=self.comm, wrtBlockSize=self.wrtBlkSize,
        #                          profiler=self.profiler, gatherMemLimit=self.gthrMem)
        #weights = self.get_weights(self.probs)
        #dprobs *= (weights + (self.probs - self.f) * self.get_dweights(self.probs, weights))[:, None]
        fsim = self.mdl._fwdsim()
        fsim.bulk_fill_timedep_dchi2(dprobs, self.evTree, self.dsCircuitsToUse, self.num_total_outcomes,
                                     self.dataset, self.minProbClipForWeighting, self.probClipInterval, None,
                                     self.comm, wrtBlockSize=self.wrtBlkSize, profiler=self.profiler,
                                     gatherMemLimit=self.gthrMem)
        # (KM,N) * (KM,1)   (N = dim of vectorized model)
        # this multiply also computes jac, which is just dprobs
        # with a different shape (jac.shape == [KM,vec_gs_len])

        if self.check_jacobian: _opt.check_jac(lambda v: self.simple_chi2(
            v), vectorGS, self.jac, tol=1e-3, eps=1e-6, errType='abs')  # TO FIX

        # dpr has shape == (nCircuits, nDerivCols), weights has shape == (nCircuits,)
        # return shape == (nCircuits, nDerivCols) where ret[i,j] = dP[i,j]*(weights+dweights*(p-f))[i]
        self.profiler.add_time("do_mc2gst: JACOBIAN", tm)
        return self.jac


# The log(Likelihood) within the Poisson picture is:                                                                                                    # noqa
#                                                                                                                                                       # noqa
# L = prod_{i,sl} lambda_{i,sl}^N_{i,sl} e^{-lambda_{i,sl}} / N_{i,sl}!                                                                                 # noqa
#                                                                                                                                                       # noqa
# Where lamba_{i,sl} := p_{i,sl}*N[i] is a rate, i indexes the operation sequence,                                                                      # noqa
#  and sl indexes the spam label.  N[i] is the total counts for the i-th circuit, and                                                                   # noqa
#  so sum_{sl} N_{i,sl} == N[i]. We can ignore the p-independent N_j! and take the log:                                                                 # noqa
#                                                                                                                                                       # noqa
# log L = sum_{i,sl} N_{i,sl} log(N[i]*p_{i,sl}) - N[i]*p_{i,sl}                                                                                        # noqa
#       = sum_{i,sl} N_{i,sl} log(p_{i,sl}) - N[i]*p_{i,sl}   (where we ignore the p-independent log(N[i]) terms)                                       # noqa
#                                                                                                                                                       # noqa
# The objective function computes the negative log(Likelihood) as a vector of leastsq                                                                   # noqa
#  terms, where each term == sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} )                                                                        # noqa
#                                                                                                                                                       # noqa
# See LikelihoodFunctions.py for details on patching                                                                                                    # noqa

# The log(Likelihood) within the standard picture is:
#
# L = prod_{i,sl} p_{i,sl}^N_{i,sl}
#
# Where i indexes the operation sequence, and sl indexes the spam label.
#  N[i] is the total counts for the i-th circuit, and
#  so sum_{sl} N_{i,sl} == N[i]. We take the log:
#
# log L = sum_{i,sl} N_{i,sl} log(p_{i,sl})
#
# The objective function computes the negative log(Likelihood) as a vector of leastsq
#  terms, where each term == sqrt( N_{i,sl} * -log(p_{i,sl}) )
#
# See LikelihoodFunction.py for details on patching
class LogLFunction(ObjectiveFunction):

    @classmethod
    def simple_init(cls, model, dataset, circuit_list=None,
                    minProbClip=1e-6, probClipInterval=(-1e6, 1e6), radius=1e-4,
                    poissonPicture=True, check=False, opLabelAliases=None,
                    evaltree_cache=None, comm=None, wildcard=None):
        """
        Create a log-likelihood objective function using a simpler set of arguments.
        """

        if circuit_list is None:
            circuit_list = list(dataset.keys())

        if evaltree_cache and 'evTree' in evaltree_cache:
            evalTree = evaltree_cache['evTree']
            lookup = evaltree_cache['lookup']
            outcomes_lookup = evaltree_cache['outcomes_lookup']
            #tree_circuit_list = evalTree.generate_circuit_list()
            # Note: this is != circuit_list, as the tree hold *simplified* circuits
        else:
            #OLD: evalTree,lookup,outcomes_lookup = smart(model.bulk_evaltree,circuit_list, dataset=dataset)
            evalTree, _, _, lookup, outcomes_lookup = model.bulk_evaltree_from_resources(
                circuit_list, comm, dataset=dataset)

            #Fill cache dict if one was given
            if evaltree_cache is not None:
                evaltree_cache['evTree'] = evalTree
                evaltree_cache['lookup'] = lookup
                evaltree_cache['outcomes_lookup'] = outcomes_lookup

        nEls = evalTree.num_final_elements()
        if evaltree_cache and 'cntVecMx' in evaltree_cache:
            countVecMx = evaltree_cache['cntVecMx']
            totalCntVec = evaltree_cache['totalCntVec']
        else:
            ds_circuit_list = _lt.apply_aliases_to_circuit_list(circuit_list, opLabelAliases)

            countVecMx = _np.empty(nEls, 'd')
            totalCntVec = _np.empty(nEls, 'd')
            for (i, opStr) in enumerate(ds_circuit_list):
                cnts = dataset[opStr].counts
                totalCntVec[lookup[i]] = sum(cnts.values())  # dataset[opStr].total
                countVecMx[lookup[i]] = [cnts.get(x, 0) for x in outcomes_lookup[i]]

            #could add to cache, but we don't have option of circuitWeights
            # here yet, so let's be conservative and not do this:
            #if evaltree_cache is not None:
            #    evaltree_cache['cntVecMx'] = countVecMx
            #    evaltree_cache['totalCntVec'] = totalCntVec

        return cls(model, evalTree, lookup, circuit_list, opLabelAliases, cptp_penalty_factor=0,
                   spam_penalty_factor=0, cntVecMx=countVecMx, totalCntVec=totalCntVec, minProbClip=minProbClip,
                   radius=radius, probClipInterval=probClipInterval, wrtBlkSize=None, gthrMem=None,
                   forcefn_grad=None, poissonPicture=poissonPicture, shiftFctr=100, check=False,
                   comm=comm, profiler=None, verbosity=0)

    def __init__(self, mdl, evTree, lookup, circuitsToUse, opLabelAliases, cptp_penalty_factor,
                 spam_penalty_factor, cntVecMx, totalCntVec, minProbClip,
                 radius, probClipInterval, wrtBlkSize, gthrMem, forcefn_grad, poissonPicture,
                 shiftFctr=100, check=False, comm=None, profiler=None, verbosity=0):
        from .. import tools as _tools

        self.mdl = mdl
        self.evTree = evTree
        self.lookup = lookup
        self.circuitsToUse = circuitsToUse
        self.comm = comm
        self.profiler = profiler
        self.check = check

        self.KM = evTree.num_final_elements()  # shorthand for combined spam+circuit dimension
        self.vec_gs_len = mdl.num_params()
        self.wrtBlkSize = wrtBlkSize
        self.gthrMem = gthrMem

        self.printer = _VerbosityPrinter.build_printer(verbosity, comm)
        self.opBasis = mdl.basis
        self.cptp_penalty_factor = cptp_penalty_factor
        self.spam_penalty_factor = spam_penalty_factor

        #Compute "extra" (i.e. beyond the (circuit,spamlable)) rows of jacobian
        self.ex = 0
        if cptp_penalty_factor != 0: self.ex += _cptp_penalty_size(mdl)
        if spam_penalty_factor != 0: self.ex += _spam_penalty_size(mdl)
        if forcefn_grad is not None: self.ex += forcefn_grad.shape[0]

        #Allocate peristent memory
        self.probs = _np.empty(self.KM, 'd')
        self.jac = _np.empty((self.KM + self.ex, self.vec_gs_len), 'd')

        #Detect omitted frequences (assumed to be 0) so we can compute liklihood correctly
        self.firsts = []; self.indicesOfCircuitsWithOmittedData = []
        for i, c in enumerate(circuitsToUse):
            lklen = _tools.slicetools.length(lookup[i])
            if 0 < lklen < mdl.get_num_outcomes(c):
                self.firsts.append(_tools.slicetools.as_array(lookup[i])[0])
                self.indicesOfCircuitsWithOmittedData.append(i)
        if len(self.firsts) > 0:
            self.firsts = _np.array(self.firsts, 'i')
            self.indicesOfCircuitsWithOmittedData = _np.array(self.indicesOfCircuitsWithOmittedData, 'i')
            self.dprobs_omitted_rowsum = _np.empty((len(self.firsts), self.vec_gs_len), 'd')
        else:
            self.firsts = None

        self.minusCntVecMx = -1.0 * cntVecMx
        self.totalCntVec = totalCntVec

        self.freqs = cntVecMx / totalCntVec
        # set zero freqs to 1.0 so np.log doesn't complain
        self.freqs_nozeros = _np.where(cntVecMx == 0, 1.0, self.freqs)

        if poissonPicture:
            self.freqTerm = cntVecMx * (_np.log(self.freqs_nozeros) - 1.0)
        else:
            self.freqTerm = cntVecMx * _np.log(self.freqs_nozeros)
            #DB_freqTerm = cntVecMx * (_np.log(freqs_nozeros) - 1.0)
            #DB_freqTerm[cntVecMx == 0] = 0.0
        # set 0 * log(0) terms explicitly to zero since numpy doesn't know this limiting behavior
        #freqTerm[cntVecMx == 0] = 0.0

        #CHECK OBJECTIVE FN
        #max_logL_terms = _tools.logl_max_terms(mdl, dataset, dsCircuitsToUse,
        #                                             poissonPicture, opLabelAliases, evaltree_cache)
        #print("DIFF1 = ",abs(_np.sum(max_logL_terms) - _np.sum(freqTerm)))

        self.min_p = minProbClip
        self.a = radius  # parameterizes "roundness" of f == 0 terms
        self.probClipInterval = probClipInterval
        self.forcefn_grad = forcefn_grad

        if forcefn_grad is not None:
            ffg_norm = _np.linalg.norm(forcefn_grad)
            start_norm = _np.linalg.norm(mdl.to_vector())
            self.forceShift = ffg_norm * (ffg_norm + start_norm) * shiftFctr
            #used to keep forceShift - _np.dot(forcefn_grad,vectorGS) positive
            # Note -- not analytic, just a heuristic!
            self.forceOffset = self.KM
            if cptp_penalty_factor != 0: self.forceOffset += _cptp_penalty_size(mdl)
            if spam_penalty_factor != 0: self.forceOffset += _spam_penalty_size(mdl)
            #index to jacobian row of first forcing term

        if poissonPicture:
            if mdl.get_simtype() == "termgap":
                assert(cptp_penalty_factor == 0), "Cannot have cptp_penalty_factor != 0 when using the termgap simtype"
                assert(spam_penalty_factor == 0), "Cannot have spam_penalty_factor != 0 when using the termgap simtype"
                assert(self.forcefn_grad is None), "Cannot use force functions when using the termgap simtype"
                self.fn = self.termgap_poisson_picture_logl
                self.jfn = self.poisson_picture_jacobian  # same jacobian as normal case
                self.v_from_probs_fn = None
            else:
                self.fn = self.poisson_picture_logl
                self.jfn = self.poisson_picture_jacobian
                self.v_from_probs_fn = self._poisson_picture_v_from_probs
        else:
            self.fn = None
            self.jfn = None

            raise NotImplementedError(("Non-poisson-picture optimization must be done with something other than a "
                                       "least-squares optimizer and isn't implemented yet."))

    def poisson_picture_logl(self, vectorGS):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval,
                                 self.check, self.comm)
        return self._poisson_picture_v_from_probs(tm)

    def _poisson_picture_v_from_probs(self, tm_start):
        pos_probs = _np.where(self.probs < self.min_p, self.min_p, self.probs)
        S = self.minusCntVecMx / self.min_p + self.totalCntVec
        S2 = -0.5 * self.minusCntVecMx / (self.min_p**2)
        v = self.freqTerm + self.minusCntVecMx * _np.log(pos_probs) + self.totalCntVec * \
            pos_probs  # dims K x M (K = nSpamLabels, M = nCircuits)

        #TODO REMOVE - pseudocode used for testing/debugging
        #nExpectedOutcomes = 2
        #for i in range(ng): # len(circuitsToUse)
        #    ps = pos_probs[lookup[i]]
        #    if len(ps) < nExpectedOutcomes:
        #        #omitted_prob = max(1.0-sum(ps),0) # if existing probs add to >1 just forget correction
        #        #iFirst = lookup[i].start #assumes lookup holds slices
        #        #v[iFirst] += totalCntVec[iFirst] * omitted_prob #accounts for omitted terms (sparse data)
        #        for j in range(lookup[i].start,lookup[i].stop):
        #            v[j] += totalCntVec[j]*(1.0/len(ps) - pos_probs[j])

        # omit = 1-p1-p2  => 1/2-p1 + 1/2-p2

        # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
        v = _np.maximum(v, 0)
        # quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where(self.probs < self.min_p, v + S * (self.probs - self.min_p) + S2 * (self.probs - self.min_p)**2, v)
        v = _np.where(self.minusCntVecMx == 0,
                      self.totalCntVec * _np.where(self.probs >= self.a,
                                                   self.probs,
                                                   (-1.0 / (3 * self.a**2)) * self.probs**3 + self.probs**2 / self.a
                                                   + self.a / 3.0),
                      v)
        # special handling for f == 0 terms
        # using quadratic rounding of function with minimum: max(0,(a-p)^2)/(2a) + p

        if self.firsts is not None:
            omitted_probs = 1.0 - _np.array([_np.sum(pos_probs[self.lookup[i]])
                                             for i in self.indicesOfCircuitsWithOmittedData])
            v[self.firsts] += self.totalCntVec[self.firsts] * \
                _np.where(omitted_probs >= self.a, omitted_probs,
                          (-1.0 / (3 * self.a**2)) * omitted_probs**3 + omitted_probs**2 / self.a + self.a / 3.0)

        #CHECK OBJECTIVE FN
        #logL_terms = _tools.logl_terms(mdl, dataset, circuitsToUse,
        #                                     min_p, probClipInterval, a, poissonPicture, False,
        #                                     opLabelAliases, evaltree_cache) # v = maxL - L so L + v - maxL should be 0
        #print("DIFF2 = ",_np.sum(logL_terms), _np.sum(v), _np.sum(freqTerm), abs(_np.sum(logL_terms)
        #      + _np.sum(v)-_np.sum(freqTerm)))

        v = _np.sqrt(v)
        v.shape = [self.KM]  # reshape ensuring no copy is needed
        if self.cptp_penalty_factor != 0:
            cpPenaltyVec = _cptp_penalty(self.mdl, self.cptp_penalty_factor, self.opBasis)
        else: cpPenaltyVec = []

        if self.spam_penalty_factor != 0:
            spamPenaltyVec = _spam_penalty(self.mdl, self.spam_penalty_factor, self.opBasis)
        else: spamPenaltyVec = []

        v = _np.concatenate((v, cpPenaltyVec, spamPenaltyVec))

        if self.forcefn_grad is not None:
            forceVec = self.forceShift - _np.dot(self.forcefn_grad, self.mdl.to_vector())
            assert(_np.all(forceVec >= 0)), "Inadequate forcing shift!"
            v = _np.concatenate((v, _np.sqrt(forceVec)))

        # TODO: handle dummy profiler generation in simple_init??
        if self.profiler: self.profiler.add_time("do_mlgst: OBJECTIVE", tm_start)
        return v  # Note: no test for whether probs is in [0,1] so no guarantee that
        #      sqrt is well defined unless probClipInterval is set within [0,1].

    #  derivative of  sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} ) terms:
    #   == 0.5 / sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} ) * ( -N_{i,sl} / p_{i,sl} + N[i] ) * dp
    #  with ommitted correction: sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} + N[i] * Y(1-other_ps)) terms (Y is a fn of other ps == omitted_probs)  # noqa
    #   == 0.5 / sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} + N[i]*(1-other_ps) ) * ( -N_{i,sl} / p_{i,sl} + N[i] ) * dp_{i,sl} +                   # noqa
    #      0.5 / sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} + N[i]*(1-other_ps) ) * ( N[i]*dY/dp_j(1-other_ps) ) * -dp_j (for p_j in other_ps)      # noqa

    #  if p <  p_min then term == sqrt( N_{i,sl} * -log(p_min) + N[i] * p_min + S*(p-p_min) )
    #   and deriv == 0.5 / sqrt(...) * S * dp

    def poisson_picture_jacobian(self, vectorGS):
        tm = _time.time()
        dprobs = self.jac[0:self.KM, :]  # avoid mem copying: use jac mem for dprobs
        dprobs.shape = (self.KM, self.vec_gs_len)
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_dprobs(dprobs, self.evTree,
                                  prMxToFill=self.probs, clipTo=self.probClipInterval,
                                  check=self.check, comm=self.comm, wrtBlockSize=self.wrtBlkSize,
                                  profiler=self.profiler, gatherMemLimit=self.gthrMem)

        pos_probs = _np.where(self.probs < self.min_p, self.min_p, self.probs)
        S = self.minusCntVecMx / self.min_p + self.totalCntVec
        S2 = -0.5 * self.minusCntVecMx / (self.min_p**2)
        v = self.freqTerm + self.minusCntVecMx * _np.log(pos_probs) + self.totalCntVec * \
            pos_probs  # dims K x M (K = nSpamLabels, M = nCircuits)

        # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
        v = _np.maximum(v, 0)
        # quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where(self.probs < self.min_p, v + S * (self.probs - self.min_p) + S2 * (self.probs - self.min_p)**2, v)
        v = _np.where(self.minusCntVecMx == 0,
                      self.totalCntVec * _np.where(self.probs >= self.a,
                                                   self.probs,
                                                   (-1.0 / (3 * self.a**2)) * self.probs**3 + self.probs**2 / self.a
                                                   + self.a / 3.0),
                      v)

        if self.firsts is not None:
            omitted_probs = 1.0 - _np.array([_np.sum(pos_probs[self.lookup[i]])
                                             for i in self.indicesOfCircuitsWithOmittedData])
            v[self.firsts] += self.totalCntVec[self.firsts] * \
                _np.where(omitted_probs >= self.a, omitted_probs,
                          (-1.0 / (3 * self.a**2)) * omitted_probs**3 + omitted_probs**2 / self.a + self.a / 3.0)

        v = _np.sqrt(v)
        # derivative diverges as v->0, but v always >= 0 so clip v to a small positive value to avoid divide by zero
        # below
        v = _np.maximum(v, 1e-100)
        dprobs_factor_pos = (0.5 / v) * (self.minusCntVecMx / pos_probs + self.totalCntVec)
        dprobs_factor_neg = (0.5 / v) * (S + 2 * S2 * (self.probs - self.min_p))
        dprobs_factor_zerofreq = (0.5 / v) * self.totalCntVec * _np.where(self.probs >= self.a,
                                                                          1.0, (-1.0 / self.a**2) * self.probs**2
                                                                          + 2 * self.probs / self.a)
        dprobs_factor = _np.where(self.probs < self.min_p, dprobs_factor_neg, dprobs_factor_pos)
        dprobs_factor = _np.where(self.minusCntVecMx == 0, dprobs_factor_zerofreq, dprobs_factor)

        if self.firsts is not None:
            dprobs_factor_omitted = (-0.5 / v[self.firsts]) * self.totalCntVec[self.firsts] \
                * _np.where(omitted_probs >= self.a,
                            1.0, (-1.0 / self.a**2) * omitted_probs**2 + 2 * omitted_probs / self.a)

            for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.lookup[i], :], axis=0)

        dprobs *= dprobs_factor[:, None]  # (KM,N) * (KM,1)   (N = dim of vectorized model)
        #Note: this also sets jac[0:KM,:]

        # need to multipy dprobs_factor_omitted[i] * dprobs[k] for k in lookup[i] and
        # add to dprobs[firsts[i]] for i in indicesOfCircuitsWithOmittedData
        if self.firsts is not None:
            dprobs[self.firsts, :] += dprobs_factor_omitted[:, None] * self.dprobs_omitted_rowsum
            # nCircuitsWithOmittedData x N

        off = 0
        if self.cptp_penalty_factor != 0:
            off += _cptp_penalty_jac_fill(self.jac[self.KM + off:, :], self.mdl, self.cptp_penalty_factor,
                                          self.opBasis)
        if self.spam_penalty_factor != 0:
            off += _spam_penalty_jac_fill(self.jac[self.KM + off:, :], self.mdl, self.spam_penalty_factor,
                                          self.opBasis)

        if self.forcefn_grad is not None:
            self.jac[self.forceOffset:, :] = -self.forcefn_grad

        if self.check: _opt.check_jac(lambda v: self.poisson_picture_logl(v), vectorGS, self.jac,
                                      tol=1e-3, eps=1e-6, errType='abs')
        self.profiler.add_time("do_mlgst: JACOBIAN", tm)
        return self.jac

    def _termgap_v2_from_probs(self, probs, S, S2):
        pos_probs = _np.where(probs < self.min_p, self.min_p, probs)
        v = self.freqTerm + self.minusCntVecMx * _np.log(pos_probs) + self.totalCntVec * \
            pos_probs  # dims K x M (K = nSpamLabels, M = nCircuits)
        v = _np.maximum(v, 0)

        # quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where(probs < self.min_p, v + S * (probs - self.min_p) + S2 * (probs - self.min_p)**2, v)
        v = _np.where(self.minusCntVecMx == 0,
                      self.totalCntVec * _np.where(probs >= self.a,
                                                   probs,
                                                   (-1.0 / (3 * self.a**2)) * probs**3 + probs**2 / self.a
                                                   + self.a / 3.0),
                      v)
        # special handling for f == 0 terms
        # using quadratic rounding of function with minimum: max(0,(a-p)^2)/(2a) + p

        if self.firsts is not None:
            omitted_probs = 1.0 - _np.array([_np.sum(pos_probs[self.lookup[i]])
                                             for i in self.indicesOfCircuitsWithOmittedData])
            v[self.firsts] += self.totalCntVec[self.firsts] * \
                _np.where(omitted_probs >= self.a, omitted_probs,
                          (-1.0 / (3 * self.a**2)) * omitted_probs**3 + omitted_probs**2 / self.a + self.a / 3.0)

        v.shape = [self.KM]  # reshape ensuring no copy is needed
        return v

    def termgap_poisson_picture_logl(self, vectorGS, oob_check=False):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval,
                                 self.check, self.comm)

        if oob_check:
            if not self.mdl.bulk_probs_paths_are_sufficient(self.evTree,
                                                            self.probs,
                                                            self.comm,
                                                            memLimit=None,
                                                            verbosity=1):
                raise ValueError("Out of bounds!")  # signals LM optimizer

        S = self.minusCntVecMx / self.min_p + self.totalCntVec
        S2 = -0.5 * self.minusCntVecMx / (self.min_p**2)
        v2 = self._termgap_v2_from_probs(self.probs, S, S2)
        v = _np.sqrt(v2)

        v.shape = [self.KM]  # reshape ensuring no copy is needed
        if self.cptp_penalty_factor != 0:
            cpPenaltyVec = _cptp_penalty(self.mdl, self.cptp_penalty_factor, self.opBasis)
        else: cpPenaltyVec = []

        if self.spam_penalty_factor != 0:
            spamPenaltyVec = _spam_penalty(self.mdl, self.spam_penalty_factor, self.opBasis)
        else: spamPenaltyVec = []

        v = _np.concatenate((v, cpPenaltyVec, spamPenaltyVec))

        if self.forcefn_grad is not None:
            forceVec = self.forceShift - _np.dot(self.forcefn_grad, vectorGS)
            assert(_np.all(forceVec >= 0)), "Inadequate forcing shift!"
            v = _np.concatenate((v, _np.sqrt(forceVec)))

        self.profiler.add_time("do_mlgst: OBJECTIVE", tm)
        return v  # Note: no test for whether probs is in [0,1] so no guarantee that


class TimeDependentLogLFunction(ObjectiveFunction):
    def __init__(self, mdl, evTree, lookup, circuitsToUse, opLabelAliases, cptp_penalty_factor,
                 spam_penalty_factor, dsCircuitsToUse, dataset, minProbClip, radius, probClipInterval, wrtBlkSize,
                 gthrMem, forcefn_grad, poissonPicture, shiftFctr=100,
                 check=False, comm=None, profiler=None, verbosity=0):
        from .. import tools as _tools
        assert(cptp_penalty_factor == 0 and spam_penalty_factor == 0), \
            "Cannot apply CPTP or SPAM penalization in time-dependent logl case (yet)"
        assert(forcefn_grad is None), "forcing functions not supported with time-dependent logl function yet"

        self.mdl = mdl
        self.evTree = evTree
        self.lookup = lookup
        self.circuitsToUse = circuitsToUse
        self.num_total_outcomes = [mdl.get_num_outcomes(c) for c in circuitsToUse]  # for sparse data detection
        self.comm = comm
        self.profiler = profiler
        self.check = check

        self.KM = evTree.num_final_elements()  # shorthand for combined spam+circuit dimension
        self.vec_gs_len = mdl.num_params()
        self.wrtBlkSize = wrtBlkSize
        self.gthrMem = gthrMem

        self.printer = _VerbosityPrinter.build_printer(verbosity, comm)
        self.opBasis = mdl.basis
        #self.cptp_penalty_factor = cptp_penalty_factor
        #self.spam_penalty_factor = spam_penalty_factor

        #Compute "extra" (i.e. beyond the (circuit,spamlable)) rows of jacobian
        self.ex = 0

        #Allocate peristent memory
        self.v = _np.empty(self.KM, 'd')
        self.jac = _np.empty((self.KM + self.ex, self.vec_gs_len), 'd')

        self.dataset = dataset
        self.dsCircuitsToUse = dsCircuitsToUse

        self.min_p = minProbClip
        self.a = radius  # parameterizes "roundness" of f == 0 terms
        self.probClipInterval = probClipInterval

        if poissonPicture:
            self.fn = self.poisson_picture_logl
            self.jfn = self.poisson_picture_jacobian
        else:
            self.fn = None
            self.jfn = None

            raise NotImplementedError(("Non-poisson-picture optimization must be done with something other than a "
                                       "least-squares optimizer and isn't implemented yet."))

    def poisson_picture_logl(self, vectorGS):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        fsim = self.mdl._fwdsim()
        v = self.v
        fsim.bulk_fill_timedep_loglpp(v, self.evTree, self.dsCircuitsToUse, self.num_total_outcomes,
                                      self.dataset, self.min_p, self.a, self.probClipInterval, self.comm)
        v = _np.sqrt(v)
        v.shape = [self.KM]  # reshape ensuring no copy is needed

        self.profiler.add_time("do_mlgst: OBJECTIVE", tm)
        return v  # Note: no test for whether probs is in [0,1] so no guarantee that
        #      sqrt is well defined unless probClipInterval is set within [0,1].

    #  derivative of  sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} ) terms:
    #   == 0.5 / sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} ) * ( -N_{i,sl} / p_{i,sl} + N[i] ) * dp
    #  with ommitted correction: sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} + N[i] * Y(1-other_ps)) terms (Y is a fn of other ps == omitted_probs)  # noqa
    #   == 0.5 / sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} + N[i]*(1-other_ps) ) * ( -N_{i,sl} / p_{i,sl} + N[i] ) * dp_{i,sl} +                   # noqa
    #      0.5 / sqrt( N_{i,sl} * -log(p_{i,sl}) + N[i] * p_{i,sl} + N[i]*(1-other_ps) ) * ( N[i]*dY/dp_j(1-other_ps) ) * -dp_j (for p_j in other_ps)      # noqa

    #  if p <  p_min then term == sqrt( N_{i,sl} * -log(p_min) + N[i] * p_min + S*(p-p_min) )
    #   and deriv == 0.5 / sqrt(...) * S * dp
    def poisson_picture_jacobian(self, vectorGS):
        tm = _time.time()
        dlogl = self.jac[0:self.KM, :]  # avoid mem copying: use jac mem for dlogl
        dlogl.shape = (self.KM, self.vec_gs_len)
        self.mdl.from_vector(vectorGS)

        fsim = self.mdl._fwdsim()
        fsim.bulk_fill_timedep_dloglpp(dlogl, self.evTree, self.dsCircuitsToUse, self.num_total_outcomes,
                                       self.dataset, self.min_p, self.a, self.probClipInterval, self.v,
                                       self.comm, wrtBlockSize=self.wrtBlkSize, profiler=self.profiler,
                                       gatherMemLimit=self.gthrMem)

        # want deriv( sqrt(logl) ) = 0.5/sqrt(logl) * deriv(logl)
        v = _np.sqrt(self.v)
        # derivative diverges as v->0, but v always >= 0 so clip v to a small positive value to avoid divide by zero
        # below
        v = _np.maximum(v, 1e-100)
        dlogl_factor = (0.5 / v)
        dlogl *= dlogl_factor[:, None]  # (KM,N) * (KM,1)   (N = dim of vectorized model)

        if self.check: _opt.check_jac(lambda v: self.poisson_picture_logl(v), vectorGS, self.jac,
                                      tol=1e-3, eps=1e-6, errType='abs')
        self.profiler.add_time("do_mlgst: JACOBIAN", tm)
        return self.jac


def _cptp_penalty_size(mdl):
    return len(mdl.operations)


def _spam_penalty_size(mdl):
    return len(mdl.preps) + sum([len(povm) for povm in mdl.povms.values()])


def _cptp_penalty(mdl, prefactor, opBasis):
    """
    Helper function - CPTP penalty: (sum of tracenorms of gates),
    which in least squares optimization means returning an array
    of the sqrt(tracenorm) of each gate.

    Returns
    -------
    numpy array
        a (real) 1D array of length len(mdl.operations).
    """
    from .. import tools as _tools
    return prefactor * _np.sqrt(_np.array([_tools.tracenorm(
        _tools.fast_jamiolkowski_iso_std(gate, opBasis)
    ) for gate in mdl.operations.values()], 'd'))


def _spam_penalty(mdl, prefactor, opBasis):
    """
    Helper function - CPTP penalty: (sum of tracenorms of gates),
    which in least squares optimization means returning an array
    of the sqrt(tracenorm) of each gate.

    Returns
    -------
    numpy array
        a (real) 1D array of length len(mdl.operations).
    """
    from .. import tools as _tools
    return prefactor * (_np.sqrt(
        _np.array([
            _tools.tracenorm(
                _tools.vec_to_stdmx(prepvec.todense(), opBasis)
            ) for prepvec in mdl.preps.values()
        ] + [
            _tools.tracenorm(
                _tools.vec_to_stdmx(mdl.povms[plbl][elbl].todense(), opBasis)
            ) for plbl in mdl.povms for elbl in mdl.povms[plbl]], 'd')
    ))


def _cptp_penalty_jac_fill(cpPenaltyVecGradToFill, mdl, prefactor, opBasis):
    """
    Helper function - jacobian of CPTP penalty (sum of tracenorms of gates)
    Returns a (real) array of shape (len(mdl.operations), nParams).
    """
    from .. import tools as _tools

    # d( sqrt(|chi|_Tr) ) = (0.5 / sqrt(|chi|_Tr)) * d( |chi|_Tr )
    for i, gate in enumerate(mdl.operations.values()):
        nP = gate.num_params()

        #get sgn(chi-matrix) == d(|chi|_Tr)/dchi in std basis
        # so sgnchi == d(|chi_std|_Tr)/dchi_std
        chi = _tools.fast_jamiolkowski_iso_std(gate, opBasis)
        assert(_np.linalg.norm(chi - chi.T.conjugate()) < 1e-4), \
            "chi should be Hermitian!"

        # Alt#1 way to compute sgnchi (evals) - works equally well to svd below
        #evals,U = _np.linalg.eig(chi)
        #sgnevals = [ ev/abs(ev) if (abs(ev) > 1e-7) else 0.0 for ev in evals]
        #sgnchi = _np.dot(U,_np.dot(_np.diag(sgnevals),_np.linalg.inv(U)))

        # Alt#2 way to compute sgnchi (sqrtm) - DOESN'T work well; sgnchi NOT very hermitian!
        #sgnchi = _np.dot(chi, _np.linalg.inv(
        #        _spl.sqrtm(_np.matrix(_np.dot(chi.T.conjugate(),chi)))))

        sgnchi = _tools.matrix_sign(chi)
        assert(_np.linalg.norm(sgnchi - sgnchi.T.conjugate()) < 1e-4), \
            "sgnchi should be Hermitian!"

        # get d(gate)/dp in opBasis [shape == (nP,dim,dim)]
        #OLD: dGdp = mdl.dproduct((gl,)) but wasteful
        dGdp = gate.deriv_wrt_params()  # shape (dim**2, nP)
        dGdp = _np.swapaxes(dGdp, 0, 1)  # shape (nP, dim**2, )
        dGdp.shape = (nP, mdl.dim, mdl.dim)

        # Let M be the "shuffle" operation performed by fast_jamiolkowski_iso_std
        # which maps a gate onto the choi-jamiolkowsky "basis" (i.e. performs that C-J
        # transform).  This shuffle op commutes with the derivative, so that
        # dchi_std/dp := d(M(G))/dp = M(dG/dp), which we call "MdGdp_std" (the choi
        # mapping of dGdp in the std basis)
        MdGdp_std = _np.empty((nP, mdl.dim, mdl.dim), 'complex')
        for p in range(nP):  # p indexes param
            MdGdp_std[p] = _tools.fast_jamiolkowski_iso_std(dGdp[p], opBasis)  # now "M(dGdp_std)"
            assert(_np.linalg.norm(MdGdp_std[p] - MdGdp_std[p].T.conjugate()) < 1e-8)  # check hermitian

        MdGdp_std = _np.conjugate(MdGdp_std)  # so element-wise multiply
        # of complex number (einsum below) results in separately adding
        # Re and Im parts (also see NOTE in spam_penalty_jac_fill below)

        #contract to get (note contract along both mx indices b/c treat like a
        # mx basis): d(|chi_std|_Tr)/dp = d(|chi_std|_Tr)/dchi_std * dchi_std/dp
        #v =  _np.einsum("ij,aij->a",sgnchi,MdGdp_std)
        v = _np.tensordot(sgnchi, MdGdp_std, ((0, 1), (1, 2)))
        v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(chi)))  # add 0.5/|chi|_Tr factor
        assert(_np.linalg.norm(v.imag) < 1e-4)
        cpPenaltyVecGradToFill[i, :] = 0.0
        cpPenaltyVecGradToFill[i, gate.gpindices] = v.real  # indexing w/array OR
        #slice works as expected in this case
        chi = sgnchi = dGdp = MdGdp_std = v = None  # free mem

    return len(mdl.operations)  # the number of leading-dim indicies we filled in


def _spam_penalty_jac_fill(spamPenaltyVecGradToFill, mdl, prefactor, opBasis):
    """
    Helper function - jacobian of CPTP penalty (sum of tracenorms of gates)
    Returns a (real) array of shape ( _spam_penalty_size(mdl), nParams).
    """
    from .. import tools as _tools
    BMxs = opBasis.elements  # shape [mdl.dim, dmDim, dmDim]
    ddenMxdV = dEMxdV = BMxs.conjugate()  # b/c denMx = sum( spamvec[i] * Bmx[i] ) and "V" == spamvec
    #NOTE: conjugate() above is because ddenMxdV and dEMxdV will get *elementwise*
    # multiplied (einsum below) by another complex matrix (sgndm or sgnE) and summed
    # in order to gather the different components of the total derivative of the trace-norm
    # wrt some spam-vector change dV.  If left un-conjugated, we'd get A*B + A.C*B.C (just
    # taking the (i,j) and (j,i) elements of the sum, say) which would give us
    # 2*Re(A*B) = A.r*B.r - B.i*A.i when we *want* (b/c Re and Im parts are thought of as
    # separate, independent degrees of freedom) A.r*B.r + A.i*B.i = 2*Re(A*B.C) -- so
    # we need to conjugate the "B" matrix, which is ddenMxdV or dEMxdV below.

    # d( sqrt(|denMx|_Tr) ) = (0.5 / sqrt(|denMx|_Tr)) * d( |denMx|_Tr )
    for i, prepvec in enumerate(mdl.preps.values()):
        nP = prepvec.num_params()

        #get sgn(denMx) == d(|denMx|_Tr)/d(denMx) in std basis
        # dmDim = denMx.shape[0]
        denMx = _tools.vec_to_stdmx(prepvec, opBasis)
        assert(_np.linalg.norm(denMx - denMx.T.conjugate()) < 1e-4), \
            "denMx should be Hermitian!"

        sgndm = _tools.matrix_sign(denMx)
        assert(_np.linalg.norm(sgndm - sgndm.T.conjugate()) < 1e-4), \
            "sgndm should be Hermitian!"

        # get d(prepvec)/dp in opBasis [shape == (nP,dim)]
        dVdp = prepvec.deriv_wrt_params()  # shape (dim, nP)
        assert(dVdp.shape == (mdl.dim, nP))

        # denMx = sum( spamvec[i] * Bmx[i] )

        #contract to get (note contrnact along both mx indices b/c treat like a mx basis):
        # d(|denMx|_Tr)/dp = d(|denMx|_Tr)/d(denMx) * d(denMx)/d(spamvec) * d(spamvec)/dp
        # [dmDim,dmDim] * [mdl.dim, dmDim,dmDim] * [mdl.dim, nP]
        #v =  _np.einsum("ij,aij,ab->b",sgndm,ddenMxdV,dVdp)
        v = _np.tensordot(_np.tensordot(sgndm, ddenMxdV, ((0, 1), (1, 2))), dVdp, (0, 0))
        v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(denMx)))  # add 0.5/|denMx|_Tr factor
        assert(_np.linalg.norm(v.imag) < 1e-4)
        spamPenaltyVecGradToFill[i, :] = 0.0
        spamPenaltyVecGradToFill[i, prepvec.gpindices] = v.real  # slice or array index works!
        denMx = sgndm = dVdp = v = None  # free mem

    #Compute derivatives for effect terms
    i = len(mdl.preps)
    for povmlbl, povm in mdl.povms.items():
        #Simplify effects of povm so we can take their derivatives
        # directly wrt parent Model parameters
        for _, effectvec in povm.simplify_effects(povmlbl).items():
            nP = effectvec.num_params()

            #get sgn(EMx) == d(|EMx|_Tr)/d(EMx) in std basis
            EMx = _tools.vec_to_stdmx(effectvec, opBasis)
            # dmDim = EMx.shape[0]
            assert(_np.linalg.norm(EMx - EMx.T.conjugate()) < 1e-4), \
                "EMx should be Hermitian!"

            sgnE = _tools.matrix_sign(EMx)
            assert(_np.linalg.norm(sgnE - sgnE.T.conjugate()) < 1e-4), \
                "sgnE should be Hermitian!"

            # get d(prepvec)/dp in opBasis [shape == (nP,dim)]
            dVdp = effectvec.deriv_wrt_params()  # shape (dim, nP)
            assert(dVdp.shape == (mdl.dim, nP))

            # EMx = sum( spamvec[i] * Bmx[i] )

            #contract to get (note contract along both mx indices b/c treat like a mx basis):
            # d(|EMx|_Tr)/dp = d(|EMx|_Tr)/d(EMx) * d(EMx)/d(spamvec) * d(spamvec)/dp
            # [dmDim,dmDim] * [mdl.dim, dmDim,dmDim] * [mdl.dim, nP]
            #v =  _np.einsum("ij,aij,ab->b",sgnE,dEMxdV,dVdp)
            v = _np.tensordot(_np.tensordot(sgnE, dEMxdV, ((0, 1), (1, 2))), dVdp, (0, 0))
            v *= prefactor * (0.5 / _np.sqrt(_tools.tracenorm(EMx)))  # add 0.5/|EMx|_Tr factor
            assert(_np.linalg.norm(v.imag) < 1e-4)

            spamPenaltyVecGradToFill[i, :] = 0.0
            spamPenaltyVecGradToFill[i, effectvec.gpindices] = v.real
            i += 1

            sgnE = dVdp = v = None  # free mem

    #return the number of leading-dim indicies we filled in
    return len(mdl.preps) + sum([len(povm) for povm in mdl.povms.values()])


class LogLWildcardFunction(ObjectiveFunction):

    def __init__(self, logl_objective_fn, base_pt, wildcard):
        from .. import tools as _tools

        self.logl_objfn = logl_objective_fn
        self.basept = base_pt
        self.wildcard_budget = wildcard
        self.wildcard_budget_precomp = wildcard.get_precomp_for_circuits(self.logl_objfn.circuitsToUse)

        self.fn = self.logl_wildcard
        self.jfn = None  # no jacobian yet

        #calling fn(...) initializes the members of self.logl_objfn
        self.probs = self.logl_objfn.probs.copy()

    def logl_wildcard(self, Wvec):
        tm = _time.time()
        self.wildcard_budget.from_vector(Wvec)
        self.wildcard_budget.update_probs(self.probs,
                                          self.logl_objfn.probs,
                                          self.logl_objfn.freqs,
                                          self.logl_objfn.circuitsToUse,
                                          self.logl_objfn.lookup,
                                          self.wildcard_budget_precomp)

        return self.logl_objfn._poisson_picture_v_from_probs(tm)
