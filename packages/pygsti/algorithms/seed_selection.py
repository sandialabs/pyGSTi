""" Core GST algorithms """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import numpy          as _np
import scipy.optimize as _spo
import scipy.stats    as _stats
import warnings       as _warnings
import time           as _time
import random         as _random
import math           as _math

from .. import optimize     as _opt
from .. import tools        as _tools
from .. import objects      as _objs
from .. import baseobjs     as _baseobjs
from .. import construction as _pc
from ..baseobjs import DummyProfiler as _DummyProfiler
_dummy_profiler = _DummyProfiler()

from .core import *

def do_seed_selection_iterative_mlgst(dataset, startGateset, gateStringSetsToUseInEstimation,
                       maxiter=100000, maxfev=None, tol=1e-6,
                       cptp_penalty_factor=0, spam_penalty_factor=0,
                       minProbClip=1e-4, probClipInterval=(-1e6,1e6), radius=1e-4,
                       poissonPicture=True,returnMaxLogL=False,returnAll=False,
                       gateStringSetLabels=None, useFreqWeightedChiSq=False,
                       verbosity=0, check=False, gatestringWeightsDict=None,
                       gateLabelAliases=None, memLimit=None,
                       profiler=None, comm=None, distributeMethod = "deriv",
                       alwaysPerformMLE=False, evaltree_cache=None, nSeeds=10, seeds=None):
    """
    Performs Iterative Maximum Likelihood Estimation Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to generate MLGST gate estimates

    startGateset : GateSet
        The GateSet used as a starting point for the least-squares
        optimization.

    gateStringSetsToUseInEstimation : list of lists of (tuples or GateStrings)
        The i-th element is a list of the gate strings to be used in the i-th iteration
        of MLGST.  Each element of these lists is a gate string, specifed as
        either a GateString object or as a tuple of gate labels (but all must be specified
        using the same type).
        e.g. [ [ (), ('Gx',) ], [ (), ('Gx',), ('Gy',) ], [ (), ('Gx',), ('Gy',), ('Gx','Gy') ]  ]

    maxiter : int, optional
        Maximum number of iterations for the logL optimization.

    maxfev : int, optional
        Maximum number of function evaluations for the logL optimization.

    tol : float, optional
        The tolerance for the logL optimization.  If a dict, allowed keys are
        `'relx'`, `'relf'`, `'f'`, and `'jac'`.  If a float, then
        `{'relx': 1e-8, 'relf': tol, 'f': 1.0, 'jac': tol }` is used.

    cptp_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains CPTP penalty
        terms which penalize non-CPTP-ness of the gates being optimized.  This factor
        multiplies these CPTP penalty terms.

    spam_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains SPAM penalty
        terms which penalize non-positive-ness of the state preps being optimized.  This
        factor multiplies these SPAM penalty terms.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by gatesets during MLGST's
        search for an optimal gateset (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poissonPicture : boolean, optional
        Whether the Poisson-picture log-likelihood should be used.

    returnAll : bool, optional
        If True return a list of gatesets (and maxLogLs if returnMaxLogL == True),
        one per iteration, instead of the results from just the final iteration.

    gateStringSetLabels : list of strings, optional
        An identification label for each of the gate string sets (used for displaying
        progress).  Must be the same length as gateStringSetsToUseInEstimation.

    useFreqWeightedChiSq : bool, optional
        If True, chi-square objective function uses the approximate chi^2 weighting:  N/(f*(1-f))
        where f is the frequency obtained from the dataset, instead of the true chi^2: N/(p*(1-p))
        where p is a predicted probability.  Defaults to False, and only should use
        True for backward compatibility.

    returnAll : boolean, optional
        If True return a list of gatesets
                        gateStringSetLabels=None,

    verbosity : int, optional
        How much detail to send to stdout.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    gatestringWeightsDict : dictionary, optional
        A dictionary with keys == gate strings and values == multiplicative scaling
        factor for the corresponding gate string. The default is no weight scaling at all.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    profiler : Profiler, optional
         A profiler object used for to track timing and memory usage.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    distributeMethod : {"gatestrings", "deriv"}
        How to distribute calculation amongst processors (only has effect
        when comm is not None).  "gatestrings" will divide the list of
        gatestrings; "deriv" will divide the columns of the jacobian matrix.

    alwaysPerformMLE : bool, optional
        When True, perform a maximum-likelihood estimate after *every* iteration,
        not just the final one.  When False, chi2 minimization is used for all
        except the final iteration (for improved numerical stability).

    evaltree_cache : dict, optional
        An empty dictionary which gets filled with the *final* computed EvalTree
        (and supporting info) used in this computation.

    nseeds : int, optional
        The number of perturb seeds used at each iteration

    Returns
    -------
    gateset               if returnAll == False and returnMaxLogL == False
    gatesets              if returnAll == True  and returnMaxLogL == False
    (maxLogL, gateset)    if returnAll == False and returnMaxLogL == True
    (maxLogL, gatesets)   if returnAll == True  and returnMaxLogL == True
        where maxLogL is the maximum log-likelihood, and gateset is the GateSet containing
        the final estimated gates.  In cases when returnAll == True, maxLogLs and gatesets
        are lists whose i-th elements are the maxLogL and gateset corresponding to the results
        of the i-th iteration.
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if profiler is None: profiler = _dummy_profiler

    #convert lists of GateStrings to lists of raw tuples since that's all we'll need
    if len(gateStringSetsToUseInEstimation ) > 0 and \
       len(gateStringSetsToUseInEstimation[0]) > 0 and \
       isinstance(gateStringSetsToUseInEstimation[0][0],_objs.GateString):
        gateStringLists = [ [gstr.tup for gstr in gsList] for gsList in gateStringSetsToUseInEstimation ]
    else:
        gateStringLists = gateStringSetsToUseInEstimation

    #Run extended MLGST iteratively on given sets of estimatable strings
    mleGatesets = [ ]; maxLogLs = [ ] #for returnAll == True case
    mleGateset = startGateset.copy(); nIters = len(gateStringLists)
    tStart = _time.time()
    tRef = tStart

    if seeds is None:
        seeds = [(mleGateset, 0)]
        for i in range(nSeeds):
            seed = mleGateset.copy()
            depol_amount = 1 / 10 ** i
            seeds.append((seed.depolarize(gate_noise=_random.uniform(0, depol_amount)), 0))
    elif isinstance(seeds, list):
        seeds = [(k, 0) for k in seeds]

    with printer.progress_logging(1):
        for (i,stringsToEstimate) in enumerate(gateStringLists):
            for j, (mleGateset, previousScore) in enumerate(seeds):
                extraMessages = [("(%s) " % gateStringSetLabels[i])] if gateStringSetLabels else []
                printer.show_progress(i, nIters, verboseMessages=extraMessages,
                                      prefix="--- Iterative MLGST:",
                                      suffix=" %d gate strings ---" % len(stringsToEstimate))

                if stringsToEstimate is None or len(stringsToEstimate) == 0: continue

                if gatestringWeightsDict is not None:
                    gatestringWeights = _np.ones( len(stringsToEstimate), 'd')
                    for gatestr, weight in gatestringWeightsDict.items():
                        if gatestr in stringsToEstimate:
                            gatestringWeights[ stringsToEstimate.index(gatestr) ] = weight
                else: gatestringWeights = None

                mleGateset.basis = startGateset.basis
                  #set basis in case of CPTP constraints

                evt_cache = {} # get the eval tree that's created so we can reuse it
                _, mleGateset = do_mc2gst(dataset, mleGateset, stringsToEstimate,
                                          maxiter, maxfev, tol, cptp_penalty_factor,
                                          spam_penalty_factor, minProbClip, probClipInterval,
                                          useFreqWeightedChiSq, 0,printer-1, check,
                                          check, gatestringWeights, gateLabelAliases,
                                          memLimit, comm, distributeMethod, profiler, evt_cache)

                if alwaysPerformMLE:
                    _, mleGateset = do_mlgst(dataset, mleGateset, stringsToEstimate,
                                             maxiter, maxfev, tol,
                                             cptp_penalty_factor, spam_penalty_factor,
                                             minProbClip, probClipInterval, radius,
                                             poissonPicture, printer-1, check, gatestringWeights,
                                             gateLabelAliases, memLimit, comm, distributeMethod, profiler, evt_cache)


                tNxt = _time.time();
                profiler.add_time('do_iterative_mlgst: iter %d chi2-opt'%(i+1),tRef)
                tRef2=tNxt

                logL_ub = _tools.logl_max(mleGateset, dataset, stringsToEstimate, poissonPicture, check, gateLabelAliases)
                maxLogL = _tools.logl(mleGateset, dataset, stringsToEstimate, minProbClip, probClipInterval,
                                      radius, poissonPicture, check, gateLabelAliases, evt_cache, comm)  #get maxLogL from chi2 estimate
                two_d_logl = 2*(logL_ub - maxLogL)

                printer.log("2*Delta(log(L)) = %g" % (two_d_logl),2)

                seeds[j] = (mleGateset, two_d_logl)

                tNxt = _time.time();
                profiler.add_time('do_iterative_mlgst: iter %d logl-comp' % (i+1),tRef2)
                printer.log("Iteration %d took %.1fs" % (i+1,tNxt-tRef),2)
                printer.log('',2) #extra newline
                tRef=tNxt

                if i == len(gateStringLists)-1 and not alwaysPerformMLE: #on the last iteration, do ML
                    # pick the seed which worked best
                    mleGateset, _ = min(seeds, key=lambda t : t[1])
                    printer.log("Switching to ML objective (last iteration)",2)

                    mleGateset.basis = startGateset.basis

                    maxLogL_p, mleGateset_p = do_mlgst(
                        dataset, mleGateset, stringsToEstimate, maxiter, maxfev, tol,
                        cptp_penalty_factor, spam_penalty_factor, minProbClip, probClipInterval, radius,
                        poissonPicture, printer-1, check, gatestringWeights, gateLabelAliases,
                        memLimit, comm, distributeMethod, profiler, evt_cache)

                    printer.log("2*Delta(log(L)) = %g" % (2*(logL_ub - maxLogL_p)),2)

                    if maxLogL_p > maxLogL: #if do_mlgst improved the maximum log-likelihood
                        maxLogL = maxLogL_p
                        mleGateset = mleGateset_p
                    else:
                        printer.warning("MLGST failed to improve logl: retaining chi2-objective estimate")
                    bestSeed, seed_logl = min(seeds, key=lambda t : t[1])
                    if seed_logl > maxLogL: #if do_mlgst improved the maximum log-likelihood
                        maxLogL = seed_logl
                        mleGateset = bestSeed
                        printer.log('Improved final iteration estimate by selecting a previous gateset')

                    tNxt = _time.time();
                    profiler.add_time('do_iterative_mlgst: iter %d logl-opt' % (i+1),tRef)
                    printer.log("Final MLGST took %.1fs" % (tNxt-tRef),2)
                    printer.log('',2) #extra newline
                    tRef=tNxt

                    if evaltree_cache is not None:
                        evaltree_cache.update(evt_cache) # final evaltree cache

                if returnAll and j == 0:
                    mleGatesets.append(mleGateset)
                    maxLogLs.append(maxLogL)

    printer.log('Iterative MLGST Total Time: %.1fs' % (_time.time()-tStart))
    profiler.add_time('do_iterative_mlgst: total time', tStart)

    if returnMaxLogL:
        return (maxLogL, mleGatesets) if returnAll else (maxLogL, mleGateset)
    else:
        return mleGatesets if returnAll else mleGateset
