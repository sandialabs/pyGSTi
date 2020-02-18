""" Defines the TermForwardSimulator calculator class"""
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
import time as _time
import itertools as _itertools
import functools as _functools
import operator as _operator

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools import listtools as _lt
from ..tools.matrixtools import _fas
from .profiler import DummyProfiler as _DummyProfiler
from .label import Label as _Label
from .termevaltree import TermEvalTree as _TermEvalTree
from .termevaltree import TermPathSet as _TermPathSet
from .termevaltree import UnsplitTreeTermPathSet as _UnsplitTreeTermPathSet
from .termevaltree import SplitTreeTermPathSet as _SplitTreeTermPathSet
from .forwardsim import ForwardSimulator
from .polynomial import Polynomial as _Polynomial
from . import replib

# For debug: sometimes helpful as it prints (python-only) tracebacks from segfaults
#import faulthandler
#faulthandler.enable()

from .opcalc import compact_deriv as _compact_deriv, safe_bulk_eval_compact_polys as _safe_bulk_eval_compact_polys

_dummy_profiler = _DummyProfiler()


class TermForwardSimulator(ForwardSimulator):
    """
    Encapsulates a calculation tool used by model objects that evaluates
    probabilities to some order in a small (error) parameter using Gates
    that can be expanded into terms of different orders and PureStateSPAMVecs.
    """

    def __init__(self, dim, simplified_op_server, paramvec,  # below here are simtype-specific args
                 mode, max_order, desired_perr=None, allowed_perr=None,
                 min_term_mag=None, max_paths_per_outcome=1000, perr_heuristic="none",
                 max_term_stages=5, path_fraction_threshold=0.9, oob_check_interval=10, cache=None):
        """
        Construct a new TermForwardSimulator object.
        TODO: fix this docstring (and maybe other fwdsim __init__ functions?

        Parameters
        ----------
        dim : int
            The gate-dimension.  All operation matrices should be dim x dim, and all
            SPAM vectors should be dim x 1.

        gates, preps, effects : OrderedDict
            Ordered dictionaries of LinearOperator, SPAMVec, and SPAMVec objects,
            respectively.  Must be *ordered* dictionaries to specify a
            well-defined column ordering when taking derivatives.

        paramvec : ndarray
            The parameter vector of the Model.

        autogator : AutoGator
            An auto-gator object that may be used to construct virtual gates
            for use in computations.

        max_order : int
            The maximum order of error-rate terms to include in probability
            computations.

        pathmag_gap : float
            TODO: docstring

        max_paths_per_outcome : int, optional
            The maximum number of paths that are allowed to be traversed when
            computing any single circuit outcome probability.

        gap_inflation_factor : float, optional
            A multiplicative factor typically > 1 that multiplies `pathmag_gap`
            to creat a new "inflated gap".  If an achieved sum-of-path-magnitudes
            is more than this inflated-gap below the maximum for a circuit,
            computation of the circuit outcome's probability will result in an
            error being raised.

        cache : dict, optional
            A dictionary of pre-computed compact polynomial objects.  Keys are
            `(max_order, rholabel, elabel, circuit)` tuples, where
            `max_order` is an integer, `rholabel` and `elabel` are
            :class:`Label` objects, and `circuit` is a :class:`Circuit`.
            Computed values are added to any dictionary that is supplied, so
            supplying an empty dictionary and using this calculator will cause
            the dictionary to be filled with values.
        """
        # self.unitary_evolution = False # Unused - idea was to have this flag
        #    allow unitary-evolution calcs to be term-based, which essentially
        #    eliminates the "pRight" portion of all the propagation calcs, and
        #    would require pLeft*pRight => |pLeft|^2
        assert(mode in ("taylor-order", "pruned", "direct")), "Invalid term-fwdsim mode: %s" % mode
        assert(perr_heuristic in ("none", "scaled", "meanscaled")), "Invalid perr_heuristic: %s" % perr_heuristic
        self.mode = mode
        self.max_order = max_order
        self.cache = cache

        # only used in "pruned" mode:
        # used when generating a list of paths - try to get gaps to be this (*no* heuristic)
        self.desired_pathmagnitude_gap = desired_perr
        self.allowed_perr = allowed_perr  # used to abort optimizations when errors in probs are too high
        self.perr_heuristic = perr_heuristic  # method used to compute expected errors in probs (often heuristic)
        self.min_term_mag = min_term_mag  # minimum abs(term coeff) to consider
        self.max_paths_per_outcome = max_paths_per_outcome

        self.poly_vindices_per_int = _Polynomial.get_vindices_per_int(len(paramvec))
        super(TermForwardSimulator, self).__init__(
            dim, simplified_op_server, paramvec)

        if self.evotype not in ("svterm", "cterm"):
            raise ValueError(("Evolution type %s is incompatbile with "
                              "term-based calculations" % self.evotype))

        #DEBUG - for profiling cython routines TODO REMOVE (& references)
        #print("DEBUG: termfwdsim: ",self.max_order, self.pathmagnitude_gap, self.min_term_mag)
        self.times_debug = {'tstartup': 0.0, 'total': 0.0,
                            't1': 0.0, 't2': 0.0, 't3': 0.0, 't4': 0.0,
                            'n1': 0, 'n2': 0, 'n3': 0, 'n4': 0}

        # not used except by _do_term_runopt in core.py -- maybe these should move to advancedoptions?
        self.max_term_stages = max_term_stages if mode == "pruned" else 1
        self.path_fraction_threshold = path_fraction_threshold if mode == "pruned" else 0.0
        self.oob_check_interval = oob_check_interval if mode == "pruned" else 0

    def copy(self):
        """ Return a shallow copy of this MatrixForwardSimulator """
        return TermForwardSimulator(self.dim, self.sos, self.paramvec,
                                    self.max_order, self.cache)

    def _rhoE_from_spamTuple(self, spamTuple):
        assert(len(spamTuple) == 2)
        if isinstance(spamTuple[0], _Label):
            rholabel, elabel = spamTuple
            rho = self.sos.get_prep(rholabel)
            E = self.sos.get_effect(elabel)
        else:
            # a "custom" spamLabel consisting of a pair of SPAMVec (or array)
            #  objects: (prepVec, effectVec)
            rho, E = spamTuple
        return rho, E

    def _rhoEs_from_labels(self, rholabel, elabels):
        """ Returns SPAMVec *objects*, so must call .todense() later """
        rho = self.sos.get_prep(rholabel)
        Es = [self.sos.get_effect(elabel) for elabel in elabels]
        #No support for "custom" spamlabel stuff here
        return rho, Es

    def propagate_state(self, rho, factors, adjoint=False):
        # TODO UPDATE
        """
        State propagation by MapOperator objects which have 'acton'
        methods.  This function could easily be overridden to
        perform some more sophisticated state propagation
        (i.e. Monte Carlo) in the future.

        Parameters
        ----------
        rho : SPAMVec
           The spam vector representing the initial state.

        circuit : Circuit or tuple
           A tuple of labels specifying the gate sequence to apply.

        Returns
        -------
        SPAMVec
        """
        if adjoint:
            for f in factors:
                rho = f.adjoint_acton(rho)  # LEXICOGRAPHICAL VS MATRIX ORDER
        else:
            for f in factors:
                rho = f.acton(rho)  # LEXICOGRAPHICAL VS MATRIX ORDER
        return rho

    def prs_directly(self, evalTree, comm=None, memLimit=None, resetWts=True, repcache=None):

        prs = _np.empty(evalTree.num_final_elements(), 'd')
        #print("Computing prs directly for %d circuits" % len(circuit_list))
        if repcache is None: repcache = {}  # new repcache...
        k = 0   # *linear* evaluation order so we know final indices are just running
        for i in evalTree.get_evaluation_order():
            circuit = evalTree[i]
            #print("Computing prs directly: circuit %d of %d" % (i,len(circuit_list)))
            assert(self.evotype == "svterm")  # for now, just do SV case
            fastmode = False  # start with slow mode
            wtTol = 0.1
            rholabel = circuit[0]
            opStr = circuit[1:]
            elabels = evalTree.simplified_circuit_elabels[i]
            prs[k:k + len(elabels)] = replib.SV_prs_directly(self, rholabel, elabels, opStr,
                                                             repcache, comm, memLimit, fastmode, wtTol, resetWts,
                                                             self.times_debug)
            k += len(elabels)
        #print("PRS = ",prs)
        return prs

    def dprs_directly(self, evalTree, wrtSlice, comm=None, memLimit=None, resetWts=True, repcache=None):
        #Finite difference derivatives (SLOW!)

        if wrtSlice is None:
            wrt_indices = list(range(self.Np))
        elif isinstance(wrtSlice, slice):
            wrt_indices = _slct.indices(wrtSlice)
        else:
            wrt_indices = wrtSlice

        eps = 1e-6  # HARDCODED
        probs = self.prs_directly(evalTree, comm, memLimit, resetWts, repcache)
        dprobs = _np.empty((evalTree.num_final_elements(), len(wrt_indices)), 'd')
        orig_vec = self.to_vector().copy()
        iParamToFinal = {i: ii for ii, i in enumerate(wrt_indices)}
        for i in range(self.Np):
            #print("direct dprobs cache %d of %d" % (i,self.Np))
            if i in iParamToFinal:  # LATER: add MPI support?
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.from_vector(vec, close=True)
                dprobs[:, iFinal] = (self.prs_directly(evalTree,
                                                       comm=None,
                                                       memLimit=None,
                                                       resetWts=False,
                                                       repcache=repcache) - probs) / eps
        self.from_vector(orig_vec, close=True)
        return dprobs

    #TODO REMOVE - UNNEEDED NOW - except maybe for helping w/docstrings
    # def OLD_prs_as_pruned_polyreps(self,
    #                            rholabel,
    #                            elabels,
    #                            circuit,
    #                            repcache,
    #                            opcache,
    #                            circuitsetup_cache,
    #                            comm=None,
    #                            memLimit=None,
    #                            pathmagnitude_gap=0.0,
    #                            min_term_mag=0.01,
    #                            max_paths=500,
    #                            current_threshold=None,
    #                            compute_polyreps=True):
    #     """
    #     Computes polynomial-representations of the probabilities for multiple
    #     spam-tuples of `circuit`, sharing the same state preparation (so with
    #     different POVM effects).  Employs a truncated or pruned path-integral
    #     approach, as opposed to just including everything up to some Taylor
    #     order as in :method:`prs_as_polys`.
    #
    #     Parameters
    #     ----------
    #     rho_label : Label
    #         The state preparation label.
    #
    #     elabels : list
    #         A list of :class:`Label` objects giving the *simplified* effect labels.
    #
    #     circuit : Circuit or tuple
    #         A tuple-like object of *simplified* gates (e.g. may include
    #         instrument elements like 'Imyinst_0')
    #
    #     repcache, opcache : dict, optional
    #         Dictionaries used to cache operator representations and
    #         operators themselves (respectively) to speed up future calls
    #         to this function that would use the same set of operations.
    #
    #     comm : mpi4py.MPI.Comm, optional
    #         When not None, an MPI communicator for distributing the computation
    #         across multiple processors.
    #
    #     memLimit : int, optional
    #         A memory limit in bytes to impose on the computation.
    #
    #     pathmagnitude_gap : float, optional
    #         The amount less than the perfect sum-of-path-magnitudes that
    #         is desired.  This sets the target sum-of-path-magnitudes for each
    #         circuit -- the threshold that determines how many paths are added.
    #
    #     min_term_mag : float, optional
    #         A technical parameter to the path pruning algorithm; this value
    #         sets a threshold for how small a term magnitude (one factor in
    #         a path magnitude) must be before it is removed from consideration
    #         entirely (to limit the number of even *potential* paths).  Terms
    #         with a magnitude lower than this values are neglected.
    #
    #     current_threshold : float, optional
    #         A more sophisticated aspect of the term-based calculation is that
    #         path polynomials should not be re-computed when we've already
    #         computed them up to a more stringent threshold than we currently
    #         need them.  This can happen, for instance, if in iteration 5 we
    #         compute all paths with magnitudes < 0.1 and now, in iteration 6,
    #         we need all paths w/mags < 0.08.  Since we've already computed more
    #         paths than what we need previously, we shouldn't recompute them now.
    #         This argument tells this function that, before any paths are computed,
    #         if it is determined that the threshold is less than this value, the
    #         function should exit immediately and return an empty list of
    #         polynomial reps.
    #
    #     Returns
    #     -------
    #     polyreps : list
    #         A list of PolynomialRep objects.
    #     npaths : int
    #         The number of paths computed.
    #     threshold : float
    #         The path-magnitude threshold used.
    #     target_sopm : float
    #         The desired sum-of-path-magnitudes.  This is `pathmagnitude_gap`
    #         less than the perfect "all-paths" sum.
    #     achieved_sopm : float
    #         The achieved sum-of-path-magnitudes.  Ideally this would equal
    #         `target_sopm`.
    #     """
    #     #Cache hold *compact* polys now: see prs_as_compact_polys
    #     #cache_keys = [(self.max_order, rholabel, elabel, circuit) for elabel in tuple(elabels)]
    #     #if self.cache is not None and all([(ck in self.cache) for ck in cache_keys]):
    #     #    return [ self.cache[ck] for ck in cache_keys ]
    #
    #     fastmode = True
    #     if repcache is None: repcache = {}
    #     if current_threshold is None: current_threshold = -1.0  # use negatives to signify "None" in C
    #     circuitsetup_cache = {}
    #
    #     if self.evotype == "svterm":
    #         poly_reps, npaths, threshold, target_sopm, achieved_sopm = \
    #             replib.SV_prs_as_pruned_polys(self, rholabel, elabels, circuit, repcache, opcache, circuitsetup_cache,
    #                                           comm, memLimit, fastmode, pathmagnitude_gap, min_term_mag, max_paths,
    #                                           current_threshold, compute_polyreps)
    #         # sopm = "sum of path magnitudes"
    #     else:  # "cterm" (stabilizer-based term evolution)
    #         poly_reps, npaths, threshold, target_sopm, achieved_sopm = \
    #             replib.SB_prs_as_pruned_polys(self, rholabel, elabels, circuit, repcache, opcache, comm, memLimit,
    #                                           fastmode, pathmagnitude_gap, min_term_mag, max_paths,
    #                                           current_threshold, compute_polyreps)
    #
    #     if len(poly_reps) == 0:  # HACK - length=0 => there's a cache hit, which we signify by None here
    #         prps = None
    #     else:
    #         prps = poly_reps
    #
    #     return prps, npaths, threshold, target_sopm, achieved_sopm

    def prs_as_pruned_polyreps(self,
                               threshold,
                               rholabel,
                               elabels,
                               circuit,
                               repcache,
                               opcache,
                               circuitsetup_cache,
                               comm=None,
                               memLimit=None,
                               mode="normal"):
        """
        TODO: docstring - see OLD version
        """
        #Cache hold *compact* polys now: see prs_as_compact_polys
        #cache_keys = [(self.max_order, rholabel, elabel, circuit) for elabel in tuple(elabels)]
        #if self.cache is not None and all([(ck in self.cache) for ck in cache_keys]):
        #    return [ self.cache[ck] for ck in cache_keys ]

        if mode == "normal":
            fastmode = 1
        elif mode == "achieved-sopm":
            fastmode = 2
        else:
            raise ValueError("Invalid mode argument: %s" % mode)

        if repcache is None: repcache = {}
        circuitsetup_cache = {}

        if self.evotype == "svterm":
            poly_reps = replib.SV_compute_pruned_path_polys_given_threshold(
                threshold, self, rholabel, elabels, circuit, repcache,
                opcache, circuitsetup_cache, comm, memLimit, fastmode)
            # sopm = "sum of path magnitudes"
        else:  # "cterm" (stabilizer-based term evolution)
            raise NotImplementedError("Just need to mimic SV version")

        if len(poly_reps) == 0:  # HACK - length=0 => there's a cache hit, which we signify by None here
            prps = None
        else:
            prps = poly_reps

        return prps

    def compute_pruned_pathmag_threshold(self,
                                         rholabel,
                                         elabels,
                                         circuit,
                                         repcache,
                                         opcache,
                                         circuitsetup_cache,
                                         comm=None,
                                         memLimit=None,
                                         threshold_guess=None):
        """
        TODO: docstring - see OLD version
        """
        #Cache hold *compact* polys now: see prs_as_compact_polys
        #cache_keys = [(self.max_order, rholabel, elabel, circuit) for elabel in tuple(elabels)]
        #if self.cache is not None and all([(ck in self.cache) for ck in cache_keys]):
        #    return [ self.cache[ck] for ck in cache_keys ]

        if repcache is None: repcache = {}
        if threshold_guess is None: threshold_guess = -1.0  # use negatives to signify "None" in C
        circuitsetup_cache = {}

        if self.evotype == "svterm":
            npaths, threshold, target_sopm, achieved_sopm = \
                replib.SV_find_best_pathmagnitude_threshold(
                    self, rholabel, elabels, circuit, repcache, opcache, circuitsetup_cache,
                    comm, memLimit, self.desired_pathmagnitude_gap, self.min_term_mag,
                    self.max_paths_per_outcome, threshold_guess
                )
            # sopm = "sum of path magnitudes"
        else:  # "cterm" (stabilizer-based term evolution)
            raise NotImplementedError("Just need to mimic SV version")

        return npaths, threshold, target_sopm, achieved_sopm

    def circuit_achieved_and_max_sopm(self, rholabel, elabels, circuit, repcache,
                                      opcache, threshold):
        if self.evotype == "svterm":
            return replib.SV_circuit_achieved_and_max_sopm(
                self, rholabel, elabels, circuit, repcache, opcache, threshold, self.min_term_mag)
        else:
            raise NotImplementedError("TODO mimic SV case")

    # LATER? , resetWts=True, repcache=None):
    def prs_as_polys(self, rholabel, elabels, circuit, comm=None, memLimit=None):
        """
        Computes polynomials of the probabilities for multiple spam-tuples of
        `circuit`, sharing the same state preparation (so with different
        POVM effects).

        Parameters
        ----------
        rho_label : Label
            The state preparation label.

        elabels : list
            A list of :class:`Label` objects giving the *simplified* effect labels.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        memLimit : int, optional
            A memory limit in bytes to impose on the computation.

        Returns
        -------
        list
            A list of Polynomial objects.
        """
        #Cache hold *compact* polys now: see prs_as_compact_polys
        #cache_keys = [(self.max_order, rholabel, elabel, circuit) for elabel in tuple(elabels)]
        #if self.cache is not None and all([(ck in self.cache) for ck in cache_keys]):
        #    return [ self.cache[ck] for ck in cache_keys ]

        fastmode = True
        if self.evotype == "svterm":
            poly_reps = replib.SV_prs_as_polys(self, rholabel, elabels, circuit, comm, memLimit, fastmode)
        else:  # "cterm" (stabilizer-based term evolution)
            poly_reps = replib.SB_prs_as_polys(self, rholabel, elabels, circuit, comm, memLimit, fastmode)
        prps = [_Polynomial.fromrep(rep) for rep in poly_reps]

        #Cache hold *compact* polys now: see prs_as_compact_polys
        #if self.cache is not None:
        #    for ck,poly in zip(cache_keys,prps):
        #        self.cache[ck] = poly
        return prps

    def pr_as_poly(self, spamTuple, circuit, comm=None, memLimit=None):
        """
        Compute probability of a single "outcome" (spam-tuple) for a single
        operation sequence.

        Parameters
        ----------
        spamTuple : (rho_label, simplified_effect_label)
            Specifies the prep and POVM effect used to compute the probability.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        memLimit : int, optional
            A memory limit in bytes to impose on the computation.

        Returns
        -------
        Polynomial
        """
        return self.prs_as_polys(spamTuple[0], [spamTuple[1]], circuit,
                                 comm, memLimit)[0]

    def prs_as_compact_polys(self, rholabel, elabels, circuit, comm=None, memLimit=None):
        """
        Computes compact-form polynomials of the probabilities for multiple
        spam-tuples of `circuit`, sharing the same state preparation (so
        with different POVM effects).

        Parameters
        ----------
        rholabel : Label
            The state preparation label.

        elabels : list
            A list of :class:`Label` objects giving the *simplified* effect labels.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        memLimit : int, optional
            A memory limit in bytes to impose on the computation.

        Returns
        -------
        list
            A list of Polynomial objects.
        """
        cache_keys = [(self.max_order, rholabel, elabel, circuit) for elabel in tuple(elabels)]
        if self.cache is not None and all([(ck in self.cache) for ck in cache_keys]):
            return [self.cache[ck] for ck in cache_keys]

        raw_prps = self.prs_as_polys(rholabel, elabels, circuit, comm, memLimit)
        prps = [poly.compact(complex_coeff_tape=True) for poly in raw_prps]
        # create compact polys w/*complex* coeffs always since we're likely
        # going to concatenate a bunch of them.

        if self.cache is not None:
            for ck, poly in zip(cache_keys, prps):
                self.cache[ck] = poly
        return prps

    def prs(self, rholabel, elabels, circuit, clipTo, bUseScaling=False, time=None):
        """
        Compute probabilities of a multiple "outcomes" (spam-tuples) for a single
        operation sequence.  The spam tuples may only vary in their effect-label (their
        prep labels must be the same)

        Parameters
        ----------
        rholabel : Label
            The state preparation label.

        elabels : list
            A list of :class:`Label` objects giving the *simplified* effect labels.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        clipTo : 2-tuple
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

        bUseScaling : bool, optional
          Unused.  Present to match function signature of other calculators.

        time : float, optional
          The *start* time at which `circuit` is evaluated.

        Returns
        -------
        numpy.ndarray
            An array of floating-point probabilities, corresponding to
            the elements of `elabels`.
        """
        assert(time is None), "TermForwardSimulator currently doesn't support time-dependent circuits"
        cpolys = self.prs_as_compact_polys(rholabel, elabels, circuit)
        vals = [_safe_bulk_eval_compact_polys(cpoly[0], cpoly[1], self.paramvec, (1,))[0]
                for cpoly in cpolys]
        ps = _np.array([_np.real_if_close(val) for val in vals])
        if clipTo is not None: ps = _np.clip(ps, clipTo[0], clipTo[1])
        return ps

    def dpr(self, spamTuple, circuit, returnPr, clipTo):
        """
        Compute the derivative of a probability generated by a operation sequence and
        spam tuple as a 1 x M numpy array, where M is the number of model
        parameters.

        Parameters
        ----------
        spamTuple : (rho_label, simplified_effect_label)
            Specifies the prep and POVM effect used to compute the probability.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        returnPr : bool
          when set to True, additionally return the probability itself.

        clipTo : 2-tuple
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

        Returns
        -------
        derivative : numpy array
            a 1 x M numpy array of derivatives of the probability w.r.t.
            each model parameter (M is the length of the vectorized model).

        probability : float
            only returned if returnPr == True.
        """
        dp = _np.empty((1, self.Np), 'd')

        poly = self.pr_as_poly(spamTuple, circuit, comm=None, memLimit=None)
        for i in range(self.Np):
            dpoly_di = poly.deriv(i)
            dp[0, i] = dpoly_di.evaluate(self.paramvec)

        if returnPr:
            p = poly.evaluate(self.paramvec)
            if clipTo is not None: p = _np.clip(p, clipTo[0], clipTo[1])
            return dp, p
        else: return dp

    def hpr(self, spamTuple, circuit, returnPr, returnDeriv, clipTo):
        """
        Compute the Hessian of a probability generated by a operation sequence and
        spam tuple as a 1 x M x M array, where M is the number of model
        parameters.

        Parameters
        ----------
        spamTuple : (rho_label, simplified_effect_label)
            Specifies the prep and POVM effect used to compute the probability.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        returnPr : bool
          when set to True, additionally return the probability itself.

        returnDeriv : bool
          when set to True, additionally return the derivative of the
          probability.

        clipTo : 2-tuple
          (min,max) to clip returned probability to if not None.
          Only relevant when prMxToFill is not None.

        Returns
        -------
        hessian : numpy array
            a 1 x M x M array, where M is the number of model parameters.
            hessian[0,j,k] is the derivative of the probability w.r.t. the
            k-th then the j-th model parameter.

        derivative : numpy array
            only returned if returnDeriv == True. A 1 x M numpy array of
            derivatives of the probability w.r.t. each model parameter.

        probability : float
            only returned if returnPr == True.
        """
        hp = _np.empty((1, self.Np, self.Np), 'd')
        if returnDeriv:
            dp = _np.empty((1, self.Np), 'd')

        poly = self.pr_as_poly(spamTuple, circuit, comm=None, memLimit=None)
        for j in range(self.Np):
            dpoly_dj = poly.deriv(j)
            if returnDeriv:
                dp[0, j] = dpoly_dj.evaluate(self.paramvec)

            for i in range(self.Np):
                dpoly_didj = dpoly_dj.deriv(i)
                hp[0, i, j] = dpoly_didj.evaluate(self.paramvec)

        if returnPr:
            p = poly.evaluate(self.paramvec)
            if clipTo is not None: p = _np.clip(p, clipTo[0], clipTo[1])

            if returnDeriv: return hp, dp, p
            else: return hp, p
        else:
            if returnDeriv: return hp, dp
            else: return hp

    def default_distribute_method(self):
        """
        Return the preferred MPI distribution mode for this calculator.
        """
        return "circuits"

    def construct_evaltree(self, simplified_circuits, numSubtreeComms):
        """
        Constructs an EvalTree object appropriate for this calculator.

        Parameters
        ----------
        simplified_circuits : list
            A list of Circuits or tuples of operation labels which specify
            the operation sequences to create an evaluation tree out of
            (most likely because you want to computed their probabilites).
            These are a "simplified" circuits in that they should only contain
            "deterministic" elements (no POVM or Instrument labels).

        TODO: docstring opcache

        numSubtreeComms : int
            The number of processor groups that will be assigned to
            subtrees of the created tree.  This aids in the tree construction
            by giving the tree information it needs to distribute itself
            among the available processors.

        Returns
        -------
        TermEvalTree
        """
        evTree = _TermEvalTree()
        evTree.initialize(simplified_circuits, numSubtreeComms)
        return evTree

    def estimate_mem_usage(self, subcalls, cache_size, num_subtrees,
                           num_subtree_proc_groups, num_param1_groups,
                           num_param2_groups, num_final_strs):
        """
        Estimate the memory required by a given set of subcalls to computation functions.

        Parameters
        ----------
        subcalls : list of strs
            A list of the names of the subcalls to estimate memory usage for.

        cache_size : int
            The size of the evaluation tree that will be passed to the
            functions named by `subcalls`.

        num_subtrees : int
            The number of subtrees to split the full evaluation tree into.

        num_subtree_proc_groups : int
            The number of processor groups used to (in parallel) iterate through
            the subtrees.  It can often be useful to have fewer processor groups
            then subtrees (even == 1) in order to perform the parallelization
            over the parameter groups.

        num_param1_groups : int
            The number of groups to divide the first-derivative parameters into.
            Computation will be automatically parallelized over these groups.

        num_param2_groups : int
            The number of groups to divide the second-derivative parameters into.
            Computation will be automatically parallelized over these groups.

        num_final_strs : int
            The number of final strings (may be less than or greater than
            `cacheSize`) the tree will hold.

        Returns
        -------
        int
            The memory estimate in bytes.
        """
        np1, np2 = num_param1_groups, num_param2_groups
        FLOATSIZE = 8  # in bytes: TODO: a better way

        wrtLen1 = (self.Np + np1 - 1) // np1  # ceiling(num_params / np1)
        wrtLen2 = (self.Np + np2 - 1) // np2  # ceiling(num_params / np2)

        mem = 0
        for fnName in subcalls:
            if fnName == "bulk_fill_probs":
                mem += num_final_strs  # pr cache final (* #elabels!)

            elif fnName == "bulk_fill_dprobs":
                mem += num_final_strs * wrtLen1  # dpr cache final (* #elabels!)

            elif fnName == "bulk_fill_hprobs":
                mem += num_final_strs * wrtLen1 * wrtLen2  # hpr cache final (* #elabels!)

            else:
                raise ValueError("Unknown subcall name: %s" % fnName)

        return mem * FLOATSIZE

    def _fill_probs_block(self, mxToFill, dest_indices, evalTree, comm=None, memLimit=None):
        nEls = evalTree.num_final_elements()
        if self.mode == "direct":
            probs = self.prs_directly(evalTree, comm, memLimit)  # could make into a fill_routine?
        else:  # "pruned" or "taylor order"
            polys = evalTree.merged_compact_polys
            probs = _safe_bulk_eval_compact_polys(
                polys[0], polys[1], self.paramvec, (nEls,))  # shape (nElements,) -- could make this a *fill*
        _fas(mxToFill, [dest_indices], probs)

    def _fill_dprobs_block(self, mxToFill, dest_indices, dest_param_indices, evalTree, param_slice, comm=None,
                           memLimit=None):
        if param_slice is None: param_slice = slice(0, self.Np)
        if dest_param_indices is None: dest_param_indices = slice(0, _slct.length(param_slice))

        if self.mode == "direct":
            dprobs = self.dprs_directly(evalTree, param_slice, comm, memLimit)
        else:  # "pruned" or "taylor order"
            # evaluate derivative of polys
            nEls = evalTree.num_final_elements()
            polys = evalTree.merged_compact_polys
            wrtInds = _np.ascontiguousarray(_slct.indices(param_slice), _np.int64)  # for Cython arg mapping
            dpolys = _compact_deriv(polys[0], polys[1], wrtInds)
            dprobs = _safe_bulk_eval_compact_polys(dpolys[0], dpolys[1], self.paramvec, (nEls, len(wrtInds)))
        _fas(mxToFill, [dest_indices, dest_param_indices], dprobs)

    def _fill_hprobs_block(self, mxToFill, dest_indices, dest_param_indices1,
                           dest_param_indices2, evalTree, param_slice1, param_slice2,
                           comm=None, memLimit=None):
        if param_slice1 is None or param_slice1.start is None: param_slice1 = slice(0, self.Np)
        if param_slice2 is None or param_slice2.start is None: param_slice2 = slice(0, self.Np)
        if dest_param_indices1 is None: dest_param_indices1 = slice(0, _slct.length(param_slice1))
        if dest_param_indices2 is None: dest_param_indices2 = slice(0, _slct.length(param_slice2))

        if self.mode == "direct":
            raise NotImplementedError("hprobs does not support direct path-integral evaluation yet")
            # hprobs = self.hprs_directly(evalTree, ...)
        else:  # "pruned" or "taylor order"
            # evaluate derivative of polys
            nEls = evalTree.num_final_elements()
            polys = evalTree.merged_compact_polys
            wrtInds1 = _np.ascontiguousarray(_slct.indices(param_slice1), _np.int64)
            wrtInds2 = _np.ascontiguousarray(_slct.indices(param_slice2), _np.int64)
            dpolys = _compact_deriv(polys[0], polys[1], wrtInds1)
            hpolys = _compact_deriv(dpolys[0], dpolys[1], wrtInds2)
            hprobs = _safe_bulk_eval_compact_polys(
                hpolys[0], hpolys[1], self.paramvec, (nEls, len(wrtInds1), len(wrtInds2)))
        _fas(mxToFill, [dest_indices, dest_param_indices1, dest_param_indices2], hprobs)

    def bulk_test_if_paths_are_sufficient(self, evalTree, probs, comm, memLimit, printer):
        """TODO: docstring
           returns nFailures, failed_circuits """
        if self.mode != "pruned":
            return True  # no "failures" for non-pruned-path mode

        # # done in bulk_get_achieved_and_max_sopm
        # replib.SV_refresh_magnitudes_in_repcache(evalTree.highmag_termrep_cache, self.to_vector())
        achieved_sopm, max_sopm = self.bulk_get_achieved_and_max_sopm(evalTree, comm, memLimit)
        # a strict bound on the error in each outcome probability, but often pessimistic
        gaps = max_sopm - achieved_sopm
        assert(_np.all(gaps >= 0))

        if self.perr_heuristic == "none":
            nFailures = _np.count_nonzero(gaps > self.allowed_perr)
            if nFailures > 0:
                printer.log("Paths are insufficient! (%d failures using strict error bound of %g)"
                            % (nFailures, self.allowed_perr))
                return False
        elif self.perr_heuristic == "scaled":
            scale = probs / achieved_sopm
            nFailures = _np.count_nonzero(gaps * scale > self.allowed_perr)
            if nFailures > 0:
                printer.log("Paths are insufficient! (%d failures using %s heuristic with error bound of %g)"
                            % (nFailures, self.perr_heuristic, self.allowed_perr))
                return False
        elif self.perr_heuristic == "meanscaled":
            scale = probs / achieved_sopm
            bFailed = _np.mean(gaps * scale) > self.allowed_perr
            if bFailed:
                printer.log("Paths are insufficient! (Using %s heuristic with error bound of %g)"
                            % (self.perr_heuristic, self.allowed_perr))
                return False
        else:
            raise ValueError("Unknown probability-error heuristic name: %s" % self.perr_heuristic)

        return True

    def bulk_get_achieved_and_max_sopm(self, evalTree, comm=None, memLimit=None):
        """TODO: docstring """

        assert(self.mode == "pruned")
        max_sopm = _np.empty(evalTree.num_final_elements(), 'd')
        achieved_sopm = _np.empty(evalTree.num_final_elements(), 'd')

        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(evalTree)

            replib.SV_refresh_magnitudes_in_repcache(evalSubTree.pathset.highmag_termrep_cache, self.to_vector())
            maxx, achieved = evalSubTree.get_achieved_and_max_sopm(self)

            _fas(max_sopm, [felInds], maxx)
            _fas(achieved_sopm, [felInds], achieved)

        #collect/gather results
        subtreeElementIndices = [t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             max_sopm, [], 0, comm)
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             achieved_sopm, [], 0, comm)

        return max_sopm, achieved_sopm

    def bulk_get_sopm_gaps(self, evalTree, comm=None, memLimit=None):
        """TODO: docstring  """
        achieved_sopm, max_sopm = self.bulk_get_achieved_and_max_sopm(evalTree, comm, memLimit)
        gaps = max_sopm - achieved_sopm
        # Gaps can be slightly negative b/c of SMALL magnitude given to acutually-0-weight paths.
        assert(_np.all(gaps >= -1e-6))
        gaps = _np.clip(gaps, 0, None)

        return gaps

    def bulk_get_sopm_gaps_jacobian(self, evalTree, comm=None, memLimit=None):
        """TODO: docstring """

        assert(self.mode == "pruned")
        termgap_penalty_jac = _np.empty((evalTree.num_final_elements(), self.Np), 'd')
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(evalTree)

            replib.SV_refresh_magnitudes_in_repcache(evalSubTree.pathset.highmag_termrep_cache, self.to_vector())
            #gaps = evalSubTree.get_sopm_gaps_using_current_paths(self)
            gap_jacs = evalSubTree.get_sopm_gaps_jacobian(self)
            # # set deriv to zero where gap would be clipped to 0
            #gap_jacs[ _np.where(gaps < self.pathmagnitude_gap) ] = 0.0
            _fas(termgap_penalty_jac, [felInds], gap_jacs)

        #collect/gather results
        subtreeElementIndices = [t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             termgap_penalty_jac, [], 0, comm)

        return termgap_penalty_jac

    # should assert(nFailures == 0) at end - this is to prep="lock in" probs & they should be good
    def find_minimal_paths_set(self, evalTree, comm=None, memLimit=None, exit_after_this_many_failures=0):
        """
        TODO: docstring
        """
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)
        local_subtree_pathsets = []  # call this list of TermPathSets for each subtree a "pathset" too

        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            if self.mode == "pruned":
                subPathSet = evalSubTree.find_minimal_paths_set(self, mySubComm, memLimit,
                                                                exit_after_this_many_failures)
            else:
                subPathSet = _UnsplitTreeTermPathSet(evalSubTree, None, None, None, 0, 0, 0)
            local_subtree_pathsets.append(subPathSet)

        return _SplitTreeTermPathSet(evalTree, local_subtree_pathsets, comm)

    # should assert(nFailures == 0) at end - this is to prep="lock in" probs & they should be good
    def select_paths_set(self, pathSet, comm=None, memLimit=None):
        """
        TODO: docstring
        """
        evalTree = pathSet.tree
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        for iSubTree, subtree_pathset in zip(mySubTreeIndices, pathSet.local_subtree_pathsets):
            evalSubTree = subtrees[iSubTree]

            if self.mode == "pruned":
                evalSubTree.select_paths_set(self, subtree_pathset, mySubComm, memLimit)
                #This computes (&caches) polys for this path set as well
            else:
                evalSubTree.cache_p_polys(self, mySubComm)

    def get_current_pathset(self, evalTree, comm):
        """ TODO: docstring """
        if self.mode == "pruned":
            subtrees = evalTree.get_sub_trees()
            mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)
            local_subtree_pathsets = [subtrees[iSubTree].get_paths_set() for iSubTree in mySubTreeIndices]
            return _SplitTreeTermPathSet(evalTree, local_subtree_pathsets, comm)
        else:
            return None

    # should assert(nFailures == 0) at end - this is to prep="lock in" probs & they should be good
    def bulk_prep_probs(self, evalTree, comm=None, memLimit=None):
        """
        Performs initial computation, such as computing probability polynomials,
        needed for bulk_fill_probs and related calls.  This is usually coupled with
        the creation of an evaluation tree, but is separated from it because this
        "preparation" may use `comm` to distribute a computationally intensive task.

        Parameters
        ----------
        evalTree : EvalTree
            The evaluation tree used to define a list of circuits and hold (cache)
            any computed quantities.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed over
           subtrees of `evalTree` (if it is split).

        memLimit : TODO: docstring
        """
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        nTotFailed = 0  # the number of failures to create an accurate-enough polynomial for a given circuit probability
        #all_failed_circuits = []
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]

            if self.mode == "pruned":
                #nFailed = evalSubTree.cache_p_pruned_polys(self, mySubComm, memLimit, self.pathmagnitude_gap,
                #                                           self.min_term_mag, self.max_paths_per_outcome)
                pathset = evalSubTree.find_minimal_paths_set(
                    self, mySubComm, memLimit, exit_after_this_many_failures=0)  # pruning_thresholds_and_highmag_terms
                # this sets these as internal cached qtys
                evalSubTree.select_paths_set(self, pathset, mySubComm, memLimit)
            else:
                evalSubTree.cache_p_polys(self, mySubComm)
                pathset = _TermPathSet(evalSubTree, 0, 0, 0)

            nTotFailed += pathset.num_failures

        nTotFailed = _mpit.sum_across_procs(nTotFailed, comm)
        #assert(nTotFailed == 0), "bulk_prep_probs could not compute polys that met the pathmagnitude gap constraints!"
        if nTotFailed > 0:
            _warnings.warn(("Unable to find a path set that achieves the desired "
                            "pathmagnitude gap (%d circuits failed)") % nTotFailed)

    def bulk_fill_probs(self, mxToFill, evalTree, clipTo=None, check=False,
                        comm=None):
        """
        Compute the outcome probabilities for an entire tree of operation sequences.

        This routine fills a 1D array, `mxToFill` with the probabilities
        corresponding to the *simplified* operation sequences found in an evaluation
        tree, `evalTree`.  An initial list of (general) :class:`Circuit`
        objects is *simplified* into a lists of gate-only sequences along with
        a mapping of final elements (i.e. probabilities) to gate-only sequence
        and prep/effect pairs.  The evaluation tree organizes how to efficiently
        compute the gate-only sequences.  This routine fills in `mxToFill`, which
        must have length equal to the number of final elements (this can be
        obtained by `evalTree.num_final_elements()`.  To interpret which elements
        correspond to which strings and outcomes, you'll need the mappings
        generated when the original list of `Circuits` was simplified.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated 1D numpy array of length equal to the
          total number of computed elements (i.e. evalTree.num_final_elements())

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
           strings to compute the bulk operation on.

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed over
           subtrees of evalTree (if it is split).


        Returns
        -------
        None
        """

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]

            felInds = evalSubTree.final_element_indices(evalTree)
            self._fill_probs_block(mxToFill, felInds, evalSubTree, mySubComm, memLimit=None)

        #collect/gather results
        subtreeElementIndices = [t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mxToFill, [], 0, comm)
        #note: pass mxToFill, dim=(KS,), so gather mxToFill[felInds] (axis=0)

        if clipTo is not None:
            _np.clip(mxToFill, clipTo[0], clipTo[1], out=mxToFill)  # in-place clip

#Will this work?? TODO
#        if check:
#            self._check(evalTree, spam_label_rows, mxToFill, clipTo=clipTo)

    def bulk_fill_dprobs(self, mxToFill, evalTree,
                         prMxToFill=None, clipTo=None, check=False,
                         comm=None, wrtFilter=None, wrtBlockSize=None,
                         profiler=None, gatherMemLimit=None):
        """
        Compute the outcome probability-derivatives for an entire tree of gate
        strings.

        Similar to `bulk_fill_probs(...)`, but fills a 2D array with
        probability-derivatives for each "final element" of `evalTree`.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated ExM numpy array where E is the total number of
          computed elements (i.e. evalTree.num_final_elements()) and M is the
          number of model parameters.

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
           strings to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with probabilities, just like in bulk_fill_probs(...).

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtFilter : list of ints, optional
          If not None, a list of integers specifying which parameters
          to include in the derivative dimension. This argument is used
          internally for distributing calculations across multiple
          processors and to control memory usage.  Cannot be specified
          in conjuction with wrtBlockSize.

        wrtBlockSize : int or float, optional
          The maximum number of derivative columns to compute *products*
          for simultaneously.  None means compute all requested columns
          at once.  The  minimum of wrtBlockSize and the size that makes
          maximal use of available processors is used as the final block size.
          This argument must be None if wrtFilter is not None.  Set this to
          non-None to reduce amount of intermediate memory required.

        profiler : Profiler, optional
          A profiler object used for to track timing and memory usage.

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.

        Returns
        -------
        None
        """

        #print("DB: bulk_fill_dprobs called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        tStart = _time.time()
        if profiler is None: profiler = _dummy_profiler

        if wrtFilter is not None:
            assert(wrtBlockSize is None)  # Cannot specify both wrtFilter and wrtBlockSize
            wrtSlice = _slct.list_to_slice(wrtFilter)  # for now, require the filter specify a slice
        else:
            wrtSlice = None

        profiler.mem_check("bulk_fill_dprobs: begin (expect ~ %.2fGB)"
                           % (mxToFill.nbytes / (1024.0**3)))

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(evalTree)
            #nEls = evalSubTree.num_final_elements()

            if prMxToFill is not None:
                self._fill_probs_block(prMxToFill, felInds, evalSubTree, mySubComm, memLimit=None)

            #Set wrtBlockSize to use available processors if it isn't specified
            blkSize = self._setParamBlockSize(wrtFilter, wrtBlockSize, mySubComm)

            if blkSize is None:
                self._fill_dprobs_block(mxToFill, felInds, None, evalSubTree, wrtSlice, mySubComm, memLimit=None)
                profiler.mem_check("bulk_fill_dprobs: post fill")

            else:  # Divide columns into blocks of at most blkSize
                assert(wrtFilter is None)  # cannot specify both wrtFilter and blkSize
                nBlks = int(_np.ceil(self.Np / blkSize))
                # num blocks required to achieve desired average size == blkSize
                blocks = _mpit.slice_up_range(self.Np, nBlks)

                #distribute derivative computation across blocks
                myBlkIndices, blkOwners, blkComm = \
                    _mpit.distribute_indices(list(range(nBlks)), mySubComm)
                if blkComm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                                   + " than derivative columns(%d)!" % self.Np
                                   + " [blkSize = %.1f, nBlks=%d]" % (blkSize, nBlks))  # pragma: no cover

                for iBlk in myBlkIndices:
                    paramSlice = blocks[iBlk]  # specifies which deriv cols calc_and_fill computes
                    self._fill_dprobs_block(mxToFill, felInds, paramSlice, evalSubTree, paramSlice,
                                            blkComm, memLimit=None)
                    profiler.mem_check("bulk_fill_dprobs: post fill blk")

                #gather results
                tm = _time.time()
                _mpit.gather_slices(blocks, blkOwners, mxToFill, [felInds],
                                    1, mySubComm, gatherMemLimit)
                #note: gathering axis 1 of mxToFill[:,fslc], dim=(ks,M)
                profiler.add_time("MPI IPC", tm)
                profiler.mem_check("bulk_fill_dprobs: post gather blocks")

        #collect/gather results
        tm = _time.time()
        subtreeElementIndices = [t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mxToFill, [], 0, comm, gatherMemLimit)
        #note: pass mxToFill, dim=(KS,M), so gather mxToFill[felInds] (axis=0)

        if prMxToFill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 prMxToFill, [], 0, comm)
            #note: pass prMxToFill, dim=(KS,), so gather prMxToFill[felInds] (axis=0)

        profiler.add_time("MPI IPC", tm)
        profiler.mem_check("bulk_fill_dprobs: post gather subtrees")

        if clipTo is not None and prMxToFill is not None:
            _np.clip(prMxToFill, clipTo[0], clipTo[1], out=prMxToFill)  # in-place clip

        #TODO: will this work?
        #if check:
        #    self._check(evalTree, spam_label_rows, prMxToFill, mxToFill,
        #                clipTo=clipTo)
        profiler.add_time("bulk_fill_dprobs: total", tStart)
        profiler.add_count("bulk_fill_dprobs count")
        profiler.mem_check("bulk_fill_dprobs: end")
        #print("DB: time debug after bulk_fill_dprobs: ", self.times_debug)
        #self.times_debug = { 'tstartup': 0.0, 'total': 0.0,
        #                     't1': 0.0, 't2': 0.0, 't3': 0.0, 't4': 0.0,
        #                     'n1': 0, 'n2': 0, 'n3': 0, 'n4': 0 }

    def bulk_fill_hprobs(self, mxToFill, evalTree,
                         prMxToFill=None, deriv1MxToFill=None, deriv2MxToFill=None,
                         clipTo=None, check=False, comm=None, wrtFilter1=None, wrtFilter2=None,
                         wrtBlockSize1=None, wrtBlockSize2=None, gatherMemLimit=None):
        """
        Compute the outcome probability-Hessians for an entire tree of gate
        strings.

        Similar to `bulk_fill_probs(...)`, but fills a 3D array with
        probability-Hessians for each "final element" of `evalTree`.

        Parameters
        ----------
        mxToFill : numpy ndarray
          an already-allocated ExMxM numpy array where E is the total number of
          computed elements (i.e. evalTree.num_final_elements()) and M1 & M2 are
          the number of selected gate-set parameters (by wrtFilter1 and wrtFilter2).

        evalTree : EvalTree
           given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
           strings to compute the bulk operation on.

        prMxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with probabilities, just like in bulk_fill_probs(...).

        derivMxToFill1, derivMxToFill2 : numpy array, optional
          when not None, an already-allocated ExM numpy array that is filled
          with probability derivatives, similar to bulk_fill_dprobs(...), but
          where M is the number of model parameters selected for the 1st and 2nd
          differentiation, respectively (i.e. by wrtFilter1 and wrtFilter2).

        clipTo : 2-tuple, optional
           (min,max) to clip return value if not None.

        check : boolean, optional
          If True, perform extra checks within code to verify correctness,
          generating warnings when checks fail.  Used for testing, and runs
          much slower when True.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is first performed over
           subtrees of evalTree (if it is split), and then over blocks (subsets)
           of the parameters being differentiated with respect to (see
           wrtBlockSize).

        wrtFilter1, wrtFilter2 : list of ints, optional
          If not None, a list of integers specifying which model parameters
          to differentiate with respect to in the first (row) and second (col)
          derivative operations, respectively.

        wrtBlockSize2, wrtBlockSize2 : int or float, optional
          The maximum number of 1st (row) and 2nd (col) derivatives to compute
          *products* for simultaneously.  None means compute all requested
          rows or columns at once.  The  minimum of wrtBlockSize and the size
          that makes maximal use of available processors is used as the final
          block size.  These arguments must be None if the corresponding
          wrtFilter is not None.  Set this to non-None to reduce amount of
          intermediate memory required.

        profiler : Profiler, optional
          A profiler object used for to track timing and memory usage.

        gatherMemLimit : int, optional
          A memory limit in bytes to impose upon the "gather" operations
          performed as a part of MPI processor syncronization.

        Returns
        -------
        None
        """

        if wrtFilter1 is not None:
            assert(wrtBlockSize1 is None and wrtBlockSize2 is None)  # Cannot specify both wrtFilter and wrtBlockSize
            wrtSlice1 = _slct.list_to_slice(wrtFilter1)  # for now, require the filter specify a slice
        else:
            wrtSlice1 = None

        if wrtFilter2 is not None:
            assert(wrtBlockSize1 is None and wrtBlockSize2 is None)  # Cannot specify both wrtFilter and wrtBlockSize
            wrtSlice2 = _slct.list_to_slice(wrtFilter2)  # for now, require the filter specify a slice
        else:
            wrtSlice2 = None

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        if self.mode == "direct":
            raise NotImplementedError("hprobs does not support direct path-integral evaluation")

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(evalTree)

            if prMxToFill is not None:
                self._fill_probs_block(prMxToFill, felInds, evalSubTree, mySubComm, memLimit=None)

            #Set wrtBlockSize to use available processors if it isn't specified
            blkSize1 = self._setParamBlockSize(wrtFilter1, wrtBlockSize1, mySubComm)
            blkSize2 = self._setParamBlockSize(wrtFilter2, wrtBlockSize2, mySubComm)

            if blkSize1 is None and blkSize2 is None:

                if deriv1MxToFill is not None:
                    self._fill_dprobs_block(deriv1MxToFill, felInds, None, evalSubTree, wrtSlice1,
                                            mySubComm, memLimit=None)
                if deriv2MxToFill is not None:
                    if deriv1MxToFill is not None and wrtSlice1 == wrtSlice2:
                        deriv2MxToFill[felInds, :] = deriv1MxToFill[felInds, :]
                    else:
                        self._fill_dprobs_block(deriv2MxToFill, felInds, None, evalSubTree, wrtSlice2,
                                                mySubComm, memLimit=None)
                self.fill_hprobs_block(mxToFill, felInds, None, None, evalTree,
                                       wrtSlice1, wrtSlice2, mySubComm, memLimit=None)

            else:  # Divide columns into blocks of at most blkSize
                assert(wrtFilter1 is None and wrtFilter2 is None)  # cannot specify both wrtFilter and blkSize
                nBlks1 = int(_np.ceil(self.Np / blkSize1))
                nBlks2 = int(_np.ceil(self.Np / blkSize2))
                # num blocks required to achieve desired average size == blkSize1 or blkSize2
                blocks1 = _mpit.slice_up_range(self.Np, nBlks1)
                blocks2 = _mpit.slice_up_range(self.Np, nBlks2)

                #distribute derivative computation across blocks
                myBlk1Indices, blk1Owners, blk1Comm = \
                    _mpit.distribute_indices(list(range(nBlks1)), mySubComm)

                myBlk2Indices, blk2Owners, blk2Comm = \
                    _mpit.distribute_indices(list(range(nBlks2)), blk1Comm)

                if blk2Comm is not None:
                    _warnings.warn("Note: more CPUs(%d)" % mySubComm.Get_size()
                                   + " than hessian elements(%d)!" % (self.Np**2)
                                   + " [blkSize = {%.1f,%.1f}, nBlks={%d,%d}]" % (blkSize1, blkSize2, nBlks1, nBlks2))  # pragma: no cover # noqa

                #in this case, where we've just divided the entire range(self.Np) into blocks, the two deriv mxs
                # will always be the same whenever they're desired (they'll both cover the entire range of params)
                derivMxToFill = deriv1MxToFill if (deriv1MxToFill is not None) else deriv2MxToFill  # first non-None

                for iBlk1 in myBlk1Indices:
                    paramSlice1 = blocks1[iBlk1]
                    if derivMxToFill is not None:
                        self._fill_dprobs_block(derivMxToFill, felInds, paramSlice1, evalSubTree, paramSlice1,
                                                blk1Comm, memLimit=None)

                    for iBlk2 in myBlk2Indices:
                        paramSlice2 = blocks2[iBlk2]
                        self.fill_hprobs_block(mxToFill, felInds, paramSlice1, paramSlice2, evalTree,
                                               paramSlice1, paramSlice2, blk2Comm, memLimit=None)

                    #gather column results: gather axis 2 of mxToFill[felInds,blocks1[iBlk1]], dim=(ks,blk1,M)
                    _mpit.gather_slices(blocks2, blk2Owners, mxToFill, [felInds, blocks1[iBlk1]],
                                        2, blk1Comm, gatherMemLimit)

                #gather row results; gather axis 1 of mxToFill[felInds], dim=(ks,M,M)
                _mpit.gather_slices(blocks1, blk1Owners, mxToFill, [felInds],
                                    1, mySubComm, gatherMemLimit)
                if deriv1MxToFill is not None:
                    _mpit.gather_slices(blocks1, blk1Owners, deriv1MxToFill, [felInds],
                                        1, mySubComm, gatherMemLimit)
                if deriv2MxToFill is not None:
                    _mpit.gather_slices(blocks2, blk2Owners, deriv2MxToFill, [felInds],
                                        1, blk1Comm, gatherMemLimit)
                    #Note: deriv2MxToFill gets computed on every inner loop completion
                    # (to save mem) but isn't gathered until now (but using blk1Comm).
                    # (just as prMxToFill is computed fully on each inner loop *iteration*!)

        #collect/gather results
        subtreeElementIndices = [t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mxToFill, [], 0, comm, gatherMemLimit)
        if deriv1MxToFill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 deriv1MxToFill, [], 0, comm, gatherMemLimit)
        if deriv2MxToFill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 deriv2MxToFill, [], 0, comm, gatherMemLimit)
        if prMxToFill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 prMxToFill, [], 0, comm)

        if clipTo is not None and prMxToFill is not None:
            _np.clip(prMxToFill, clipTo[0], clipTo[1], out=prMxToFill)  # in-place clip

        #TODO: check if this works
        #if check:
        #    self._check(evalTree, spam_label_rows,
        #                prMxToFill, deriv1MxToFill, mxToFill, clipTo)
