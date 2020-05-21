"""
Defines the TermForwardSimulator calculator class
"""
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
    A forward-simulation calculator that uses term-path-integration to compute probabilities.

    Parameters
    ----------
    dim : int
        The model-dimension.  All operations act on a `dim`-dimensional Hilbert-Schmidt space.

    layer_op_server : LayerLizard
        An object that can be queried for circuit-layer operations.

    paramvec : numpy.ndarray
        The current parameter vector of the Model.

    mode : {"taylor-order", "pruned", "direct"}
        Overall method used to compute (approximate) circuit probabilities.
        The `"taylor-order"` mode includes all taylor-expansion terms up to a
        fixed and pre-defined order, fixing a single "path set" at the outset.
        The `"pruned"` mode selects a path set based on a heuristic (sometimes a
        true upper bound) calculation of the error in the approximate probabilities.
        This method effectively "prunes" the paths to meet a fixed desired accuracy.
        The `"direct"` method is still under development.  Its intention is to perform
        path integrals directly without the use of polynomial computation and caching.
        Initial testing showed the direct method to be much slower for common QCVV tasks,
        making it less urgent to complete.

    max_order : int
        The maximum order of error-rate terms to include in probability
        computations.  When polynomials are computed, the maximum Taylor
        order to compute polynomials to.

    desired_perr : float, optional
        The desired maximum-error when computing probabilities..
        Path sets are selected (heuristically) to target this error, within the
        bounds set by `max_order`, etc.

    allowed_perr : float, optional
        The allowed maximum-error when computing probabilities.
        When rigorous bounds cannot guarantee that probabilities are correct to
        within this error, additional paths are added to the path set.

    min_term_mag : float, optional
        Terms with magnitudes less than this value will be ignored, i.e. not
        considered candidates for inclusion in paths.  If this number is too
        low, the number of possible paths to consder may be very large, impacting
        performance.  If too high, then not enough paths will be considered to
        achieve an accurate result.  By default this value is set automatically
        based on the desired error and `max_paths_per_outcome`.  Only adjust this
        if you know what you're doing.

    max_paths_per_outcome : int, optional
        The maximum number of paths that can be used (summed) to compute a
        single outcome probability.

    perr_heuristic : {"none", "scaled", "meanscaled"}
        Which heuristic (if any) to use when deciding whether a given path set is
        sufficient given the allowed error (`allowed_perr`).
        - `"none"`:  This is the strictest setting, and is absence of any heuristic.
        if the path-magnitude gap (the maximum - achieved sum-of-path-magnitudes,
        a rigorous upper bound on the approximation error for a circuit
        outcome probability) is greater than `allowed_perr` for any circuit, the
        path set is deemed insufficient.
        - `"scaled"`: a path set is deemed insufficient when, for any circuit, the
        path-magnitude gap multiplied by a scaling factor is greater than `allowed_perr`.
        This scaling factor is equal to the computed probability divided by the
        achieved sum-of-path-magnitudes and is always less than 1.  This scaling
        is essentially the ratio of the sum of the path amplitudes without and with
        an absolute value, and tries to quantify and offset the degree of pessimism
        in the computed path-magnitude gap.
        - `"meanscaled"` : a path set is deemed insufficient when, the *mean* of all
        the scaled (as above) path-magnitude gaps is greater than `allowed_perr`.  The
        mean here is thus over the circuit outcomes.  This heuristic is even more
        permissive of potentially bad path sets than `"scaled"`, as it allows badly
        approximated circuits to be offset by well approximated ones.

    max_term_stages : int, optional
        The maximum number of "stage", i.e. re-computations of a path set, are
        allowed before giving up.

    path_fraction_threshold : float, optional
        When greater than this fraction of the total available paths (set by
        other constraints) are considered, no further re-compuation of the
        path set will occur, as it is expected to give little improvement.

    oob_check_interval : int, optional
        The optimizer will check whether the computed probabilities have sufficiently
        small error every `oob_check_interval` (outer) optimizer iteration.

    cache : dict, optional
        A dictionary of pre-computed compact polynomial objects.  Keys are
        `(max_order, rholabel, elabel, circuit)` tuples, where
        `max_order` is an integer, `rholabel` and `elabel` are
        :class:`Label` objects, and `circuit` is a :class:`Circuit`.
        Computed values are added to any dictionary that is supplied, so
        supplying an empty dictionary and using this calculator will cause
        the dictionary to be filled with values.
    """

    def __init__(self, dim, layer_op_server, paramvec,  # below here are simtype-specific args
                 mode, max_order, desired_perr=None, allowed_perr=None,
                 min_term_mag=None, max_paths_per_outcome=1000, perr_heuristic="none",
                 max_term_stages=5, path_fraction_threshold=0.9, oob_check_interval=10, cache=None):
        """
        Construct a new TermForwardSimulator object.

        Parameters
        ----------
        dim : int
            The model-dimension.  All operations act on a `dim`-dimensional Hilbert-Schmidt space.

        layer_op_server : LayerLizard
            An object that can be queried for circuit-layer operations.

        paramvec : numpy.ndarray
            The current parameter vector of the Model.

        mode : {"taylor-order", "pruned", "direct"}
            Overall method used to compute (approximate) circuit probabilities.
            The `"taylor-order"` mode includes all taylor-expansion terms up to a
            fixed and pre-defined order, fixing a single "path set" at the outset.
            The `"pruned"` mode selects a path set based on a heuristic (sometimes a
            true upper bound) calculation of the error in the approximate probabilities.
            This method effectively "prunes" the paths to meet a fixed desired accuracy.
            The `"direct"` method is still under development.  Its intention is to perform
            path integrals directly without the use of polynomial computation and caching.
            Initial testing showed the direct method to be much slower for common QCVV tasks,
            making it less urgent to complete.

        max_order : int
            The maximum order of error-rate terms to include in probability
            computations.  When polynomials are computed, the maximum Taylor
            order to compute polynomials to.

        desired_perr : float, optional
            The desired maximum-error when computing probabilities..
            Path sets are selected (heuristically) to target this error, within the
            bounds set by `max_order`, etc.

        allowed_perr : float, optional
            The allowed maximum-error when computing probabilities.
            When rigorous bounds cannot guarantee that probabilities are correct to
            within this error, additional paths are added to the path set.

        min_term_mag : float, optional
            Terms with magnitudes less than this value will be ignored, i.e. not
            considered candidates for inclusion in paths.  If this number is too
            low, the number of possible paths to consder may be very large, impacting
            performance.  If too high, then not enough paths will be considered to
            achieve an accurate result.  By default this value is set automatically
            based on the desired error and `max_paths_per_outcome`.  Only adjust this
            if you know what you're doing.

        max_paths_per_outcome : int, optional
            The maximum number of paths that can be used (summed) to compute a
            single outcome probability.

        perr_heuristic : {"none", "scaled", "meanscaled"}
            Which heuristic (if any) to use when deciding whether a given path set is
            sufficient given the allowed error (`allowed_perr`).
            - `"none"`:  This is the strictest setting, and is absence of any heuristic.
            if the path-magnitude gap (the maximum - achieved sum-of-path-magnitudes,
            a rigorous upper bound on the approximation error for a circuit
            outcome probability) is greater than `allowed_perr` for any circuit, the
            path set is deemed insufficient.
            - `"scaled"`: a path set is deemed insufficient when, for any circuit, the
            path-magnitude gap multiplied by a scaling factor is greater than `allowed_perr`.
            This scaling factor is equal to the computed probability divided by the
            achieved sum-of-path-magnitudes and is always less than 1.  This scaling
            is essentially the ratio of the sum of the path amplitudes without and with
            an absolute value, and tries to quantify and offset the degree of pessimism
            in the computed path-magnitude gap.
            - `"meanscaled"` : a path set is deemed insufficient when, the *mean* of all
            the scaled (as above) path-magnitude gaps is greater than `allowed_perr`.  The
            mean here is thus over the circuit outcomes.  This heuristic is even more
            permissive of potentially bad path sets than `"scaled"`, as it allows badly
            approximated circuits to be offset by well approximated ones.

        max_term_stages : int, optional
            The maximum number of "stage", i.e. re-computations of a path set, are
            allowed before giving up.

        path_fraction_threshold : float, optional
            When greater than this fraction of the total available paths (set by
            other constraints) are considered, no further re-compuation of the
            path set will occur, as it is expected to give little improvement.

        oob_check_interval : int, optional
            The optimizer will check whether the computed probabilities have sufficiently
            small error every `oob_check_interval` (outer) optimizer iteration.

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
            dim, layer_op_server, paramvec)

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
        """
        Return a shallow copy of this TermForwardSimulator.

        Returns
        -------
        TermForwardSimulator
        """
        return TermForwardSimulator(self.dim, self.sos, self.paramvec,
                                    self.max_order, self.cache)

    def _rho_e_from_spam_tuple(self, spam_tuple):
        assert(len(spam_tuple) == 2)
        if isinstance(spam_tuple[0], _Label):
            rholabel, elabel = spam_tuple
            rho = self.sos.get_prep(rholabel)
            E = self.sos.get_effect(elabel)
        else:
            # a "custom" spamLabel consisting of a pair of SPAMVec (or array)
            #  objects: (prepVec, effectVec)
            rho, E = spam_tuple
        return rho, E

    def _rho_es_from_labels(self, rholabel, elabels):
        """ Returns SPAMVec *objects*, so must call .todense() later """
        rho = self.sos.get_prep(rholabel)
        Es = [self.sos.get_effect(elabel) for elabel in elabels]
        #No support for "custom" spamlabel stuff here
        return rho, Es

    def propagate_state(self, rho, factors, adjoint=False):
        """
        State propagation by MapOperator objects which have 'acton' methods.

        This function could easily be overridden to
        perform some more sophisticated state propagation
        (i.e. Monte Carlo) in the future.

        Parameters
        ----------
        rho : SPAMVec
            The spam vector representing the initial state.

        factors : list or tuple
            A list or tuple of operator objects possessing `acton` methods.

        adjoint : bool, optional
            Whether to use `adjoint_acton` instead of `acton` to propagate `rho`.

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

    def prs_directly(self, eval_tree, comm=None, mem_limit=None, reset_wts=True, repcache=None):
        """
        Compute probabilities of `eval_tree`'s circuits using "direct" mode.

        Parameters
        ----------
        eval_tree : TermEvalTree
            The evaluation tree.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of eval_tree (if it is split).

        mem_limit : int, optional
            A rough memory limit in bytes.

        reset_wts : bool, optional
            Whether term magnitudes should be updated based on current term coefficients
            (which are based on the current point in model-parameter space) or not.

        repcache : dict, optional
            A cache of term representations for increased performance.
        """
        prs = _np.empty(eval_tree.num_final_elements(), 'd')
        #print("Computing prs directly for %d circuits" % len(circuit_list))
        if repcache is None: repcache = {}  # new repcache...
        k = 0   # *linear* evaluation order so we know final indices are just running
        for i in eval_tree.get_evaluation_order():
            circuit = eval_tree[i]
            #print("Computing prs directly: circuit %d of %d" % (i,len(circuit_list)))
            assert(self.evotype == "svterm")  # for now, just do SV case
            fastmode = False  # start with slow mode
            wtTol = 0.1
            rholabel = circuit[0]
            opStr = circuit[1:]
            elabels = eval_tree.simplified_circuit_elabels[i]
            prs[k:k + len(elabels)] = replib.SV_prs_directly(self, rholabel, elabels, opStr,
                                                             repcache, comm, mem_limit, fastmode, wtTol, reset_wts,
                                                             self.times_debug)
            k += len(elabels)
        #print("PRS = ",prs)
        return prs

    def dprs_directly(self, eval_tree, wrt_slice, comm=None, mem_limit=None, reset_wts=True, repcache=None):
        """
        Compute probability derivatives of `eval_tree`'s circuits using "direct" mode.

        Parameters
        ----------
        eval_tree : TermEvalTree
            The evaluation tree.

        wrt_slice : slice
            A slice specifying which model parameters to differentiate with respect to.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of eval_tree (if it is split).

        mem_limit : int, optional
            A rough memory limit in bytes.

        reset_wts : bool, optional
            Whether term magnitudes should be updated based on current term coefficients
            (which are based on the current point in model-parameter space) or not.

        repcache : dict, optional
            A cache of term representations for increased performance.
        """
        #Note: Finite difference derivatives are SLOW!
        if wrt_slice is None:
            wrt_indices = list(range(self.Np))
        elif isinstance(wrt_slice, slice):
            wrt_indices = _slct.indices(wrt_slice)
        else:
            wrt_indices = wrt_slice

        eps = 1e-6  # HARDCODED
        probs = self.prs_directly(eval_tree, comm, mem_limit, reset_wts, repcache)
        dprobs = _np.empty((eval_tree.num_final_elements(), len(wrt_indices)), 'd')
        orig_vec = self.to_vector().copy()
        iParamToFinal = {i: ii for ii, i in enumerate(wrt_indices)}
        for i in range(self.Np):
            #print("direct dprobs cache %d of %d" % (i,self.Np))
            if i in iParamToFinal:  # LATER: add MPI support?
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.from_vector(vec, close=True)
                dprobs[:, iFinal] = (self.prs_directly(eval_tree,
                                                       comm=None,
                                                       mem_limit=None,
                                                       reset_wts=False,
                                                       repcache=repcache) - probs) / eps
        self.from_vector(orig_vec, close=True)
        return dprobs

    def prs_as_pruned_polyreps(self,
                               threshold,
                               rholabel,
                               elabels,
                               circuit,
                               repcache,
                               opcache,
                               circuitsetup_cache,
                               comm=None,
                               mem_limit=None,
                               mode="normal"):
        """
        Computes polynomial-representations of circuit-outcome probabilities.

        In particular, the circuit-outcomes under consideration share the same state
        preparation and differ only in their POVM effects.  Employs a truncated or pruned
        path-integral approach, as opposed to just including everything up to some Taylor
        order as in :method:`prs_as_polys`.

        Parameters
        ----------
        threshold : float
            The path-magnitude threshold.  Only include (sum) paths whose magnitudes are
            greater than or equal to this threshold.

        rholabel : Label
            The state preparation label.

        elabels : list
            A list of :class:`Label` objects giving the *simplified* effect labels.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        repcache : dict, optional
            Dictionaries used to cache operator representations  to speed up future
            calls to this function that would use the same set of operations.

        opcache : dict, optional
            Dictionary used to cache operators themselves to speed up future calls
            to this function that would use the same set of operations.

        circuitsetup_cache : dict
            A cache holding specialized elements that store and eliminate
            the need to recompute per-circuit information.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        mem_limit : int, optional
            A memory limit in bytes to impose on the computation.

        mode : {"normal", "achieved_sopm"}
            Controls whether polynomials are actually computed (`"normal"`) or whether only the
            achieved sum-of-path-magnitudes is computed (`"achieved_sopm"`).  The latter mode is
            useful when a `threshold` is being tested but not committed to, as computing only the
            achieved sum-of-path-magnitudes is significantly faster.

        Returns
        -------
        list
           A list of :class:`PolyRep` objects.  These polynomial represetations are essentially
           bare-bones polynomials stored efficiently for performance.  (To get a full
           :class:`Polynomial` object, use :classmethod:`Polynomial.fromrep`.)
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
                opcache, circuitsetup_cache, comm, mem_limit, fastmode)
            # sopm = "sum of path magnitudes"
        else:  # "cterm" (stabilizer-based term evolution)
            raise NotImplementedError("Just need to mimic SV version")

        #TODO REMOVE this case -- we don't check for cache hits anymore; I think we can just set prps = poly_reps here
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
                                         mem_limit=None,
                                         threshold_guess=None):
        """
        Finds a good path-magnitude threshold for `circuit` at the current parameter-space point.

        Parameters
        ----------
        rholabel : Label
            The state preparation label.

        elabels : list
            A list of :class:`Label` objects giving the *simplified* effect labels.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        repcache : dict, optional
            Dictionaries used to cache operator representations  to speed up future
            calls to this function that would use the same set of operations.

        opcache : dict, optional
            Dictionary used to cache operators themselves to speed up future calls
            to this function that would use the same set of operations.

        circuitsetup_cache : dict
            A cache holding specialized elements that store and eliminate
            the need to recompute per-circuit information.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        mem_limit : int, optional
            A memory limit in bytes to impose on the computation.

        threshold_guess : float, optional
            A guess estimate of a good path-magnitude threshold.

        Returns
        -------
        npaths : int
            The number of paths found.  (total over all circuit outcomes)

        threshold : float
            The final path-magnitude threshold used.

        target_sopm : float
            The target (desired) sum-of-path-magnitudes. (summed over all circuit outcomes)

        achieved_sopm : float
            The achieved sum-of-path-magnitudes. (summed over all circuit outcomes)
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
                    comm, mem_limit, self.desired_pathmagnitude_gap, self.min_term_mag,
                    self.max_paths_per_outcome, threshold_guess
                )
            # sopm = "sum of path magnitudes"
        else:  # "cterm" (stabilizer-based term evolution)
            raise NotImplementedError("Just need to mimic SV version")

        return npaths, threshold, target_sopm, achieved_sopm

    def circuit_achieved_and_max_sopm(self, rholabel, elabels, circuit, repcache,
                                      opcache, threshold):
        """
        Computes the achieved and maximum sum-of-path-magnitudes for `circuit`.

        Parameters
        ----------
        rholabel : Label
            The state preparation label.

        elabels : list
            A list of :class:`Label` objects giving the *simplified* effect labels.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates.

        repcache : dict, optional
            Dictionaries used to cache operator representations  to speed up future
            calls to this function that would use the same set of operations.

        opcache : dict, optional
            Dictionary used to cache operators themselves to speed up future calls
            to this function that would use the same set of operations.

        threshold : float
            path-magnitude threshold.  Only sum path magnitudes above or equal to this threshold.

        Returns
        -------
        achieved_sopm : float
            The achieved sum-of-path-magnitudes. (summed over all circuit outcomes)

        max_sopm : float
            The maximum possible sum-of-path-magnitudes. (summed over all circuit outcomes)
        """
        if self.evotype == "svterm":
            return replib.SV_circuit_achieved_and_max_sopm(
                self, rholabel, elabels, circuit, repcache, opcache, threshold, self.min_term_mag)
        else:
            raise NotImplementedError("TODO mimic SV case")

    # LATER? , reset_wts=True, repcache=None):
    def prs_as_polys(self, rholabel, elabels, circuit, comm=None, mem_limit=None):
        """
        Computes polynomial-representations of circuit-outcome probabilities.

        In particular, the circuit-outcomes under consideration share the same state
        preparation and differ only in their POVM effects.

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

        mem_limit : int, optional
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
            poly_reps = replib.SV_prs_as_polys(self, rholabel, elabels, circuit, comm, mem_limit, fastmode)
        else:  # "cterm" (stabilizer-based term evolution)
            poly_reps = replib.SB_prs_as_polys(self, rholabel, elabels, circuit, comm, mem_limit, fastmode)
        prps = [_Polynomial.fromrep(rep) for rep in poly_reps]

        #Cache hold *compact* polys now: see prs_as_compact_polys
        #if self.cache is not None:
        #    for ck,poly in zip(cache_keys,prps):
        #        self.cache[ck] = poly
        return prps

    def pr_as_poly(self, spam_tuple, circuit, comm=None, mem_limit=None):
        """
        Compute probability of a single "outcome" (spam-tuple) for a single circuit.

        Parameters
        ----------
        spam_tuple : (rho_label, simplified_effect_label)
            Specifies the prep and POVM effect used to compute the probability.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        mem_limit : int, optional
            A memory limit in bytes to impose on the computation.

        Returns
        -------
        Polynomial
        """
        return self.prs_as_polys(spam_tuple[0], [spam_tuple[1]], circuit,
                                 comm, mem_limit)[0]

    def prs_as_compact_polys(self, rholabel, elabels, circuit, comm=None, mem_limit=None):
        """
        Compute compact-form polynomials of the outcome probabilities for `circuit`.

        Note that these outcomes are defined as having the same state preparation
        and different POVM effects.

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

        mem_limit : int, optional
            A memory limit in bytes to impose on the computation.

        Returns
        -------
        list
            A list of Polynomial objects.
        """
        cache_keys = [(self.max_order, rholabel, elabel, circuit) for elabel in tuple(elabels)]
        if self.cache is not None and all([(ck in self.cache) for ck in cache_keys]):
            return [self.cache[ck] for ck in cache_keys]

        raw_prps = self.prs_as_polys(rholabel, elabels, circuit, comm, mem_limit)
        prps = [poly.compact(complex_coeff_tape=True) for poly in raw_prps]
        # create compact polys w/*complex* coeffs always since we're likely
        # going to concatenate a bunch of them.

        if self.cache is not None:
            for ck, poly in zip(cache_keys, prps):
                self.cache[ck] = poly
        return prps

    def prs(self, rholabel, elabels, circuit, clip_to, use_scaling=False, time=None):
        """
        Compute the outcome probabilities for `circuit`.

        Note that these outcomes are defined as having the same state preparation
        and different POVM effects.

        Parameters
        ----------
        rholabel : Label
            The state preparation label.

        elabels : list
            A list of :class:`Label` objects giving the *simplified* effect labels.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        clip_to : 2-tuple
            (min,max) to clip returned probability to if not None.
            Only relevant when pr_mx_to_fill is not None.

        use_scaling : bool, optional
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
        if clip_to is not None: ps = _np.clip(ps, clip_to[0], clip_to[1])
        return ps

    def dpr(self, spam_tuple, circuit, return_pr, clip_to):
        """
        Compute the outcome probability derivatives for `circuit`.

        Note that these outcomes are defined as having the same state preparation
        and different POVM effects.  The derivatives are computed as a 1 x M numpy
        array, where M is the number of model parameters.

        Parameters
        ----------
        spam_tuple : (rho_label, simplified_effect_label)
            Specifies the prep and POVM effect used to compute the probability.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        return_pr : bool
            when set to True, additionally return the probability itself.

        clip_to : 2-tuple
            (min,max) to clip returned probability to if not None.
            Only relevant when pr_mx_to_fill is not None.

        Returns
        -------
        derivative : numpy array
            a 1 x M numpy array of derivatives of the probability w.r.t.
            each model parameter (M is the length of the vectorized model).

        probability : float
            only returned if return_pr == True.
        """
        dp = _np.empty((1, self.Np), 'd')

        poly = self.pr_as_poly(spam_tuple, circuit, comm=None, mem_limit=None)
        for i in range(self.Np):
            dpoly_di = poly.deriv(i)
            dp[0, i] = dpoly_di.evaluate(self.paramvec)

        if return_pr:
            p = poly.evaluate(self.paramvec)
            if clip_to is not None: p = _np.clip(p, clip_to[0], clip_to[1])
            return dp, p
        else: return dp

    def hpr(self, spam_tuple, circuit, return_pr, return_deriv, clip_to):
        """
        Compute the outcome probability second derivatives for `circuit`.

        Note that these outcomes are defined as having the same state preparation
        and different POVM effects.  The derivatives are computed as a 1 x M x M array,
        where M is the number of model parameters.

        Parameters
        ----------
        spam_tuple : (rho_label, simplified_effect_label)
            Specifies the prep and POVM effect used to compute the probability.

        circuit : Circuit or tuple
            A tuple-like object of *simplified* gates (e.g. may include
            instrument elements like 'Imyinst_0')

        return_pr : bool
            when set to True, additionally return the probability itself.

        return_deriv : bool
            when set to True, additionally return the derivative of the
            probability.

        clip_to : 2-tuple
            (min,max) to clip returned probability to if not None.
            Only relevant when pr_mx_to_fill is not None.

        Returns
        -------
        hessian : numpy array
            a 1 x M x M array, where M is the number of model parameters.
            hessian[0,j,k] is the derivative of the probability w.r.t. the
            k-th then the j-th model parameter.

        derivative : numpy array
            only returned if return_deriv == True. A 1 x M numpy array of
            derivatives of the probability w.r.t. each model parameter.

        probability : float
            only returned if return_pr == True.
        """
        hp = _np.empty((1, self.Np, self.Np), 'd')
        if return_deriv:
            dp = _np.empty((1, self.Np), 'd')

        poly = self.pr_as_poly(spam_tuple, circuit, comm=None, mem_limit=None)
        for j in range(self.Np):
            dpoly_dj = poly.deriv(j)
            if return_deriv:
                dp[0, j] = dpoly_dj.evaluate(self.paramvec)

            for i in range(self.Np):
                dpoly_didj = dpoly_dj.deriv(i)
                hp[0, i, j] = dpoly_didj.evaluate(self.paramvec)

        if return_pr:
            p = poly.evaluate(self.paramvec)
            if clip_to is not None: p = _np.clip(p, clip_to[0], clip_to[1])

            if return_deriv: return hp, dp, p
            else: return hp, p
        else:
            if return_deriv: return hp, dp
            else: return hp

    def default_distribute_method(self):
        """
        Return the preferred MPI distribution mode for this calculator.

        Returns
        -------
        str
        """
        return "circuits"

    def construct_evaltree(self, simplified_circuits, num_subtree_comms):
        """
        Constructs an EvalTree object appropriate for this calculator.

        Parameters
        ----------
        simplified_circuits : list
            A list of Circuits or tuples of operation labels which specify
            the circuits to create an evaluation tree out of
            (most likely because you want to computed their probabilites).
            These are a "simplified" circuits in that they should only contain
            "deterministic" elements (no POVM or Instrument labels).

        num_subtree_comms : int
            The number of processor groups that will be assigned to
            subtrees of the created tree.  This aids in the tree construction
            by giving the tree information it needs to distribute itself
            among the available processors.

        Returns
        -------
        TermEvalTree
        """
        evTree = _TermEvalTree()
        evTree.initialize(simplified_circuits, num_subtree_comms)
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
            `cache_size`) the tree will hold.

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

    def _fill_probs_block(self, mx_to_fill, dest_indices, eval_tree, comm=None, mem_limit=None):
        nEls = eval_tree.num_final_elements()
        if self.mode == "direct":
            probs = self.prs_directly(eval_tree, comm, mem_limit)  # could make into a fill_routine?
        else:  # "pruned" or "taylor order"
            polys = eval_tree.merged_compact_polys
            probs = _safe_bulk_eval_compact_polys(
                polys[0], polys[1], self.paramvec, (nEls,))  # shape (nElements,) -- could make this a *fill*
        _fas(mx_to_fill, [dest_indices], probs)

    def _fill_dprobs_block(self, mx_to_fill, dest_indices, dest_param_indices, eval_tree, param_slice, comm=None,
                           mem_limit=None):
        if param_slice is None: param_slice = slice(0, self.Np)
        if dest_param_indices is None: dest_param_indices = slice(0, _slct.length(param_slice))

        if self.mode == "direct":
            dprobs = self.dprs_directly(eval_tree, param_slice, comm, mem_limit)
        else:  # "pruned" or "taylor order"
            # evaluate derivative of polys
            nEls = eval_tree.num_final_elements()
            polys = eval_tree.merged_compact_polys
            wrtInds = _np.ascontiguousarray(_slct.indices(param_slice), _np.int64)  # for Cython arg mapping
            dpolys = _compact_deriv(polys[0], polys[1], wrtInds)
            dprobs = _safe_bulk_eval_compact_polys(dpolys[0], dpolys[1], self.paramvec, (nEls, len(wrtInds)))
        _fas(mx_to_fill, [dest_indices, dest_param_indices], dprobs)

    def _fill_hprobs_block(self, mx_to_fill, dest_indices, dest_param_indices1,
                           dest_param_indices2, eval_tree, param_slice1, param_slice2,
                           comm=None, mem_limit=None):
        if param_slice1 is None or param_slice1.start is None: param_slice1 = slice(0, self.Np)
        if param_slice2 is None or param_slice2.start is None: param_slice2 = slice(0, self.Np)
        if dest_param_indices1 is None: dest_param_indices1 = slice(0, _slct.length(param_slice1))
        if dest_param_indices2 is None: dest_param_indices2 = slice(0, _slct.length(param_slice2))

        if self.mode == "direct":
            raise NotImplementedError("hprobs does not support direct path-integral evaluation yet")
            # hprobs = self.hprs_directly(eval_tree, ...)
        else:  # "pruned" or "taylor order"
            # evaluate derivative of polys
            nEls = eval_tree.num_final_elements()
            polys = eval_tree.merged_compact_polys
            wrtInds1 = _np.ascontiguousarray(_slct.indices(param_slice1), _np.int64)
            wrtInds2 = _np.ascontiguousarray(_slct.indices(param_slice2), _np.int64)
            dpolys = _compact_deriv(polys[0], polys[1], wrtInds1)
            hpolys = _compact_deriv(dpolys[0], dpolys[1], wrtInds2)
            hprobs = _safe_bulk_eval_compact_polys(
                hpolys[0], hpolys[1], self.paramvec, (nEls, len(wrtInds1), len(wrtInds2)))
        _fas(mx_to_fill, [dest_indices, dest_param_indices1, dest_param_indices2], hprobs)

    def bulk_test_if_paths_are_sufficient(self, eval_tree, probs, comm, mem_limit, printer):
        """
        Determine whether `eval_tree`'s current path set (perhaps heuristically) achieves the desired accuracy.

        The current path set is determined by the current (per-circuti) path-magnitude thresholds
        (stored in the evaluation tree) and the current parameter-space point (also reflected in
        the terms cached in the evaluation tree).

        Parameters
        ----------
        eval_tree : TermEvalTree
            The evaluation tree.

        probs : numpy.ndarray
            The element array of (approximate) circuit outcome probabilities.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        mem_limit : int, optional
            A memory limit in bytes to impose on the computation.

        printer : VerbosityPrinter
            An printer object for displaying messages.

        Returns
        -------
        bool
        """
        if self.mode != "pruned":
            return True  # no "failures" for non-pruned-path mode

        # # done in bulk_get_achieved_and_max_sopm
        # replib.SV_refresh_magnitudes_in_repcache(eval_tree.highmag_termrep_cache, self.to_vector())
        achieved_sopm, max_sopm = self.bulk_get_achieved_and_max_sopm(eval_tree, comm, mem_limit)
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

    def bulk_get_achieved_and_max_sopm(self, eval_tree, comm=None, mem_limit=None):
        """
        Compute element arrays of achieved and maximum-possible sum-of-path-magnitudes.

        These values are computed for the current set of paths contained in `eval_tree`.

        Parameters
        ----------
        eval_tree : TermEvalTree
            The evaluation tree.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        mem_limit : int, optional
            A memory limit in bytes to impose on the computation.

        Returns
        -------
        max_sopm : numpy.ndarray
            An array containing the per-circuit-outcome maximum sum-of-path-magnitudes.

        achieved_sopm : numpy.ndarray
            An array containing the per-circuit-outcome achieved sum-of-path-magnitudes.
        """

        assert(self.mode == "pruned")
        max_sopm = _np.empty(eval_tree.num_final_elements(), 'd')
        achieved_sopm = _np.empty(eval_tree.num_final_elements(), 'd')

        subtrees = eval_tree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = eval_tree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(eval_tree)

            replib.SV_refresh_magnitudes_in_repcache(evalSubTree.pathset.highmag_termrep_cache, self.to_vector())
            maxx, achieved = evalSubTree.get_achieved_and_max_sopm(self)

            _fas(max_sopm, [felInds], maxx)
            _fas(achieved_sopm, [felInds], achieved)

        #collect/gather results
        subtreeElementIndices = [t.final_element_indices(eval_tree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             max_sopm, [], 0, comm)
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             achieved_sopm, [], 0, comm)

        return max_sopm, achieved_sopm

    def bulk_get_sopm_gaps(self, eval_tree, comm=None, mem_limit=None):
        """
        Compute an element array sum-of-path-magnitude gaps (the difference between maximum and achieved).

        These values are computed for the current set of paths contained in `eval_tree`.

        Parameters
        ----------
        eval_tree : TermEvalTree
            The evaluation tree.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        mem_limit : int, optional
            A memory limit in bytes to impose on the computation.

        Returns
        -------
        numpy.ndarray
            An array containing the per-circuit-outcome sum-of-path-magnitude gaps.
        """
        achieved_sopm, max_sopm = self.bulk_get_achieved_and_max_sopm(eval_tree, comm, mem_limit)
        gaps = max_sopm - achieved_sopm
        # Gaps can be slightly negative b/c of SMALL magnitude given to acutually-0-weight paths.
        assert(_np.all(gaps >= -1e-6))
        gaps = _np.clip(gaps, 0, None)

        return gaps

    def bulk_get_sopm_gaps_jacobian(self, eval_tree, comm=None, mem_limit=None):
        """
        Compute the jacobian of the the output of :method:`bulk_get_sopm_gaps`.

        Parameters
        ----------
        eval_tree : TermEvalTree
            The evaluation tree.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        mem_limit : int, optional
            A memory limit in bytes to impose on the computation.

        Returns
        -------
        numpy.ndarray
            An number-of-elements by number-of-model-parameters array containing the jacobian
            of the sum-of-path-magnitude gaps.
        """
        assert(self.mode == "pruned")
        termgap_penalty_jac = _np.empty((eval_tree.num_final_elements(), self.Np), 'd')
        subtrees = eval_tree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = eval_tree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(eval_tree)

            replib.SV_refresh_magnitudes_in_repcache(evalSubTree.pathset.highmag_termrep_cache, self.to_vector())
            #gaps = evalSubTree.get_sopm_gaps_using_current_paths(self)
            gap_jacs = evalSubTree.get_sopm_gaps_jacobian(self)
            # # set deriv to zero where gap would be clipped to 0
            #gap_jacs[ _np.where(gaps < self.pathmagnitude_gap) ] = 0.0
            _fas(termgap_penalty_jac, [felInds], gap_jacs)

        #collect/gather results
        subtreeElementIndices = [t.final_element_indices(eval_tree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             termgap_penalty_jac, [], 0, comm)

        return termgap_penalty_jac

    # should assert(nFailures == 0) at end - this is to prep="lock in" probs & they should be good
    def find_minimal_paths_set(self, eval_tree, comm=None, mem_limit=None, exit_after_this_many_failures=0):
        """
        Find a good, i.e. minimial, path set for the current model-parameter space point.

        Parameters
        ----------
        eval_tree : TermEvalTree
            The evaluation tree.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        mem_limit : int, optional
            A memory limit in bytes to impose on the computation.

        exit_after_this_many_failures : int, optional
           If > 0, give up after this many circuits fail to meet the desired accuracy criteria.
           This short-circuits doomed attempts to find a good path set so they don't take too long.

        Returns
        -------
        SplitTreeTermPathSet
        """
        subtrees = eval_tree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = eval_tree.distribute(comm)
        local_subtree_pathsets = []  # call this list of TermPathSets for each subtree a "pathset" too

        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            if self.mode == "pruned":
                subPathSet = evalSubTree.find_minimal_paths_set(self, mySubComm, mem_limit,
                                                                exit_after_this_many_failures)
            else:
                subPathSet = _UnsplitTreeTermPathSet(evalSubTree, None, None, None, 0, 0, 0)
            local_subtree_pathsets.append(subPathSet)

        return _SplitTreeTermPathSet(eval_tree, local_subtree_pathsets, comm)

    # should assert(nFailures == 0) at end - this is to prep="lock in" probs & they should be good
    def select_paths_set(self, path_set, comm=None, mem_limit=None):
        """
        Selects (makes "current") a path set *and* computes polynomials the new set.

        Parameters
        ----------
        path_set : PathSet
            The path set to select.

        comm : mpy4py.MPI.Comm
            An MPI communicator for dividing the compuational task.

        mem_limit : int
            Rough memory limit (per processor) in bytes.

        Returns
        -------
        None
        """
        evalTree = path_set.tree
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        for iSubTree, subtree_pathset in zip(mySubTreeIndices, path_set.local_subtree_pathsets):
            evalSubTree = subtrees[iSubTree]

            if self.mode == "pruned":
                evalSubTree.select_paths_set(self, subtree_pathset, mySubComm, mem_limit)
                #This computes (&caches) polys for this path set as well
            else:
                evalSubTree.cache_p_polys(self, mySubComm)

    def get_current_pathset(self, eval_tree, comm):
        """
        Returns the current path set (held in eval_tree).

        Note that this works even when `eval_tree` is split,
        and always returns a :class:`SplitTreeTermPathSet`.

        Parameters
        ----------
        eval_tree : TermEvalTree
            The evaluation tree.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        Returns
        -------
        SplitTreeTermPathSet or None
        """
        if self.mode == "pruned":
            subtrees = eval_tree.get_sub_trees()
            mySubTreeIndices, subTreeOwners, mySubComm = eval_tree.distribute(comm)
            local_subtree_pathsets = [subtrees[iSubTree].get_paths_set() for iSubTree in mySubTreeIndices]
            return _SplitTreeTermPathSet(eval_tree, local_subtree_pathsets, comm)
        else:
            return None

    # should assert(nFailures == 0) at end - this is to prep="lock in" probs & they should be good
    def bulk_prep_probs(self, eval_tree, comm=None, mem_limit=None):
        """
        Performs preparatory work for computing circuit outcome probabilities.

        Finds a good path set that meets (if possible) the accuracy requirements
        and computes needed polynomials.

        Parameters
        ----------
        eval_tree : EvalTree
            The evaluation tree used to define a list of circuits and hold (cache)
            any computed quantities.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of `eval_tree` (if it is split).

        mem_limit : int
            Rough memory limit (per processor) in bytes.

        Returns
        -------
        None
        """
        subtrees = eval_tree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = eval_tree.distribute(comm)

        #eval on each local subtree
        nTotFailed = 0  # the number of failures to create an accurate-enough polynomial for a given circuit probability
        #all_failed_circuits = []
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]

            if self.mode == "pruned":
                #nFailed = evalSubTree.cache_p_pruned_polys(self, mySubComm, mem_limit, self.pathmagnitude_gap,
                #                                           self.min_term_mag, self.max_paths_per_outcome)
                pathset = evalSubTree.find_minimal_paths_set(
                    self, mySubComm, mem_limit, exit_after_this_many_failures=0)  # pruning_thresholds_and_highmag_terms
                # this sets these as internal cached qtys
                evalSubTree.select_paths_set(self, pathset, mySubComm, mem_limit)
            else:
                evalSubTree.cache_p_polys(self, mySubComm)
                pathset = _TermPathSet(evalSubTree, 0, 0, 0)

            nTotFailed += pathset.num_failures

        nTotFailed = _mpit.sum_across_procs(nTotFailed, comm)
        #assert(nTotFailed == 0), "bulk_prep_probs could not compute polys that met the pathmagnitude gap constraints!"
        if nTotFailed > 0:
            _warnings.warn(("Unable to find a path set that achieves the desired "
                            "pathmagnitude gap (%d circuits failed)") % nTotFailed)

    def bulk_fill_probs(self, mx_to_fill, eval_tree, clip_to=None, check=False,
                        comm=None):
        """
        Compute the outcome probabilities for an entire tree of circuits.

        This routine fills a 1D array, `mx_to_fill` with the probabilities
        corresponding to the *simplified* circuits found in an evaluation
        tree, `eval_tree`.  An initial list of (general) :class:`Circuit`
        objects is *simplified* into a lists of gate-only sequences along with
        a mapping of final elements (i.e. probabilities) to gate-only sequence
        and prep/effect pairs.  The evaluation tree organizes how to efficiently
        compute the gate-only sequences.  This routine fills in `mx_to_fill`, which
        must have length equal to the number of final elements (this can be
        obtained by `eval_tree.num_final_elements()`.  To interpret which elements
        correspond to which strings and outcomes, you'll need the mappings
        generated when the original list of `Circuits` was simplified.

        Parameters
        ----------
        mx_to_fill : numpy ndarray
            an already-allocated 1D numpy array of length equal to the
            total number of computed elements (i.e. eval_tree.num_final_elements())

        eval_tree : EvalTree
            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
            strings to compute the bulk operation on.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        check : boolean, optional
            If True, perform extra checks within code to verify correctness,
            generating warnings when checks fail.  Used for testing, and runs
            much slower when True.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of eval_tree (if it is split).

        Returns
        -------
        None
        """

        #get distribution across subtrees (groups if needed)
        subtrees = eval_tree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = eval_tree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]

            felInds = evalSubTree.final_element_indices(eval_tree)
            self._fill_probs_block(mx_to_fill, felInds, evalSubTree, mySubComm, mem_limit=None)

        #collect/gather results
        subtreeElementIndices = [t.final_element_indices(eval_tree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mx_to_fill, [], 0, comm)
        #note: pass mx_to_fill, dim=(KS,), so gather mx_to_fill[felInds] (axis=0)

        if clip_to is not None:
            _np.clip(mx_to_fill, clip_to[0], clip_to[1], out=mx_to_fill)  # in-place clip

#Will this work?? TODO
#        if check:
#            self._check(eval_tree, spam_label_rows, mx_to_fill, clip_to=clip_to)

    def bulk_fill_dprobs(self, mx_to_fill, eval_tree,
                         pr_mx_to_fill=None, clip_to=None, check=False,
                         comm=None, wrt_filter=None, wrt_block_size=None,
                         profiler=None, gather_mem_limit=None):
        """
        Compute the outcome probability-derivatives for an entire tree of circuits.

        Similar to `bulk_fill_probs(...)`, but fills a 2D array with
        probability-derivatives for each "final element" of `eval_tree`.

        Parameters
        ----------
        mx_to_fill : numpy ndarray
            an already-allocated ExM numpy array where E is the total number of
            computed elements (i.e. eval_tree.num_final_elements()) and M is the
            number of model parameters.

        eval_tree : EvalTree
            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
            strings to compute the bulk operation on.

        pr_mx_to_fill : numpy array, optional
            when not None, an already-allocated length-E numpy array that is filled
            with probabilities, just like in bulk_fill_probs(...).

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        check : boolean, optional
            If True, perform extra checks within code to verify correctness,
            generating warnings when checks fail.  Used for testing, and runs
            much slower when True.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is first performed over
            subtrees of eval_tree (if it is split), and then over blocks (subsets)
            of the parameters being differentiated with respect to (see
            wrt_block_size).

        wrt_filter : list of ints, optional
            If not None, a list of integers specifying which parameters
            to include in the derivative dimension. This argument is used
            internally for distributing calculations across multiple
            processors and to control memory usage.  Cannot be specified
            in conjuction with wrt_block_size.

        wrt_block_size : int or float, optional
            The maximum number of derivative columns to compute *products*
            for simultaneously.  None means compute all requested columns
            at once.  The  minimum of wrt_block_size and the size that makes
            maximal use of available processors is used as the final block size.
            This argument must be None if wrt_filter is not None.  Set this to
            non-None to reduce amount of intermediate memory required.

        profiler : Profiler, optional
            A profiler object used for to track timing and memory usage.

        gather_mem_limit : int, optional
            A memory limit in bytes to impose upon the "gather" operations
            performed as a part of MPI processor syncronization.

        Returns
        -------
        None
        """

        #print("DB: bulk_fill_dprobs called!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        tStart = _time.time()
        if profiler is None: profiler = _dummy_profiler

        if wrt_filter is not None:
            assert(wrt_block_size is None)  # Cannot specify both wrt_filter and wrt_block_size
            wrtSlice = _slct.list_to_slice(wrt_filter)  # for now, require the filter specify a slice
        else:
            wrtSlice = None

        profiler.mem_check("bulk_fill_dprobs: begin (expect ~ %.2fGB)"
                           % (mx_to_fill.nbytes / (1024.0**3)))

        #get distribution across subtrees (groups if needed)
        subtrees = eval_tree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = eval_tree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(eval_tree)
            #nEls = evalSubTree.num_final_elements()

            if pr_mx_to_fill is not None:
                self._fill_probs_block(pr_mx_to_fill, felInds, evalSubTree, mySubComm, mem_limit=None)

            #Set wrt_block_size to use available processors if it isn't specified
            blkSize = self._set_param_block_size(wrt_filter, wrt_block_size, mySubComm)

            if blkSize is None:
                self._fill_dprobs_block(mx_to_fill, felInds, None, evalSubTree, wrtSlice, mySubComm, mem_limit=None)
                profiler.mem_check("bulk_fill_dprobs: post fill")

            else:  # Divide columns into blocks of at most blkSize
                assert(wrt_filter is None)  # cannot specify both wrt_filter and blkSize
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
                    self._fill_dprobs_block(mx_to_fill, felInds, paramSlice, evalSubTree, paramSlice,
                                            blkComm, mem_limit=None)
                    profiler.mem_check("bulk_fill_dprobs: post fill blk")

                #gather results
                tm = _time.time()
                _mpit.gather_slices(blocks, blkOwners, mx_to_fill, [felInds],
                                    1, mySubComm, gather_mem_limit)
                #note: gathering axis 1 of mx_to_fill[:,fslc], dim=(ks,M)
                profiler.add_time("MPI IPC", tm)
                profiler.mem_check("bulk_fill_dprobs: post gather blocks")

        #collect/gather results
        tm = _time.time()
        subtreeElementIndices = [t.final_element_indices(eval_tree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mx_to_fill, [], 0, comm, gather_mem_limit)
        #note: pass mx_to_fill, dim=(KS,M), so gather mx_to_fill[felInds] (axis=0)

        if pr_mx_to_fill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 pr_mx_to_fill, [], 0, comm)
            #note: pass pr_mx_to_fill, dim=(KS,), so gather pr_mx_to_fill[felInds] (axis=0)

        profiler.add_time("MPI IPC", tm)
        profiler.mem_check("bulk_fill_dprobs: post gather subtrees")

        if clip_to is not None and pr_mx_to_fill is not None:
            _np.clip(pr_mx_to_fill, clip_to[0], clip_to[1], out=pr_mx_to_fill)  # in-place clip

        #TODO: will this work?
        #if check:
        #    self._check(eval_tree, spam_label_rows, pr_mx_to_fill, mx_to_fill,
        #                clip_to=clip_to)
        profiler.add_time("bulk_fill_dprobs: total", tStart)
        profiler.add_count("bulk_fill_dprobs count")
        profiler.mem_check("bulk_fill_dprobs: end")
        #print("DB: time debug after bulk_fill_dprobs: ", self.times_debug)
        #self.times_debug = { 'tstartup': 0.0, 'total': 0.0,
        #                     't1': 0.0, 't2': 0.0, 't3': 0.0, 't4': 0.0,
        #                     'n1': 0, 'n2': 0, 'n3': 0, 'n4': 0 }

    def bulk_fill_hprobs(self, mx_to_fill, eval_tree,
                         pr_mx_to_fill=None, deriv1_mx_to_fill=None, deriv2_mx_to_fill=None,
                         clip_to=None, check=False, comm=None, wrt_filter1=None, wrt_filter2=None,
                         wrt_block_size1=None, wrt_block_size2=None, gather_mem_limit=None):
        """
        Compute the outcome probability-Hessians for an entire tree of circuits.

        Similar to `bulk_fill_probs(...)`, but fills a 3D array with
        probability-Hessians for each "final element" of `eval_tree`.

        Parameters
        ----------
        mx_to_fill : numpy ndarray
            an already-allocated ExMxM numpy array where E is the total number of
            computed elements (i.e. eval_tree.num_final_elements()) and M1 & M2 are
            the number of selected gate-set parameters (by wrt_filter1 and wrt_filter2).

        eval_tree : EvalTree
            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
            strings to compute the bulk operation on.

        pr_mx_to_fill : numpy array, optional
            when not None, an already-allocated length-E numpy array that is filled
            with probabilities, just like in bulk_fill_probs(...).

        deriv1_mx_to_fill : numpy array, optional
            when not None, an already-allocated ExM numpy array that is filled
            with probability derivatives, similar to bulk_fill_dprobs(...), but
            where M is the number of model parameters selected for the 1st
            differentiation (i.e. by wrt_filter1).

        deriv2_mx_to_fill : numpy array, optional
            when not None, an already-allocated ExM numpy array that is filled
            with probability derivatives, similar to bulk_fill_dprobs(...), but
            where M is the number of model parameters selected for the 2nd
            differentiation (i.e. by wrt_filter2).

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        check : boolean, optional
            If True, perform extra checks within code to verify correctness,
            generating warnings when checks fail.  Used for testing, and runs
            much slower when True.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is first performed over
            subtrees of eval_tree (if it is split), and then over blocks (subsets)
            of the parameters being differentiated with respect to (see
            wrt_block_size).

        wrt_filter1 : list of ints, optional
            If not None, a list of integers specifying which model parameters
            to differentiate with respect to in the first (row) derivative operations.

        wrt_filter2 : list of ints, optional
            If not None, a list of integers specifying which model parameters
            to differentiate with respect to in the second (col) derivative operations.

        wrt_block_size1: int or float, optional
            The maximum number of 1st (row) derivatives to compute
            *products* for simultaneously.  None means compute all requested
            rows or columns at once.  The minimum of wrt_block_size and the size
            that makes maximal use of available processors is used as the final
            block size.  This argument must be None if the corresponding
            wrt_filter is not None.  Set this to non-None to reduce amount of
            intermediate memory required.

        wrt_block_size2 : int or float, optional
            The maximum number of 2nd (col) derivatives to compute
            *products* for simultaneously.  None means compute all requested
            rows or columns at once.  The minimum of wrt_block_size and the size
            that makes maximal use of available processors is used as the final
            block size.  This argument must be None if the corresponding
            wrt_filter is not None.  Set this to non-None to reduce amount of
            intermediate memory required.

        gather_mem_limit : int, optional
            A memory limit in bytes to impose upon the "gather" operations
            performed as a part of MPI processor syncronization.

        Returns
        -------
        None
        """

        if wrt_filter1 is not None:
            assert(wrt_block_size1 is None and wrt_block_size2 is None), \
                "Cannot specify both wrt_filter and wrt_block_size"
            wrtSlice1 = _slct.list_to_slice(wrt_filter1)  # for now, require the filter specify a slice
        else:
            wrtSlice1 = None

        if wrt_filter2 is not None:
            assert(wrt_block_size1 is None and wrt_block_size2 is None), \
                "Cannot specify both wrt_filter and wrt_block_size"
            wrtSlice2 = _slct.list_to_slice(wrt_filter2)  # for now, require the filter specify a slice
        else:
            wrtSlice2 = None

        #get distribution across subtrees (groups if needed)
        subtrees = eval_tree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = eval_tree.distribute(comm)

        if self.mode == "direct":
            raise NotImplementedError("hprobs does not support direct path-integral evaluation")

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(eval_tree)

            if pr_mx_to_fill is not None:
                self._fill_probs_block(pr_mx_to_fill, felInds, evalSubTree, mySubComm, mem_limit=None)

            #Set wrt_block_size to use available processors if it isn't specified
            blkSize1 = self._set_param_block_size(wrt_filter1, wrt_block_size1, mySubComm)
            blkSize2 = self._set_param_block_size(wrt_filter2, wrt_block_size2, mySubComm)

            if blkSize1 is None and blkSize2 is None:

                if deriv1_mx_to_fill is not None:
                    self._fill_dprobs_block(deriv1_mx_to_fill, felInds, None, evalSubTree, wrtSlice1,
                                            mySubComm, mem_limit=None)
                if deriv2_mx_to_fill is not None:
                    if deriv1_mx_to_fill is not None and wrtSlice1 == wrtSlice2:
                        deriv2_mx_to_fill[felInds, :] = deriv1_mx_to_fill[felInds, :]
                    else:
                        self._fill_dprobs_block(deriv2_mx_to_fill, felInds, None, evalSubTree, wrtSlice2,
                                                mySubComm, mem_limit=None)
                self.fill_hprobs_block(mx_to_fill, felInds, None, None, eval_tree,
                                       wrtSlice1, wrtSlice2, mySubComm, mem_limit=None)

            else:  # Divide columns into blocks of at most blkSize
                assert(wrt_filter1 is None and wrt_filter2 is None)  # cannot specify both wrt_filter and blkSize
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
                derivMxToFill = deriv1_mx_to_fill if (deriv1_mx_to_fill is not None) else deriv2_mx_to_fill  # non-None

                for iBlk1 in myBlk1Indices:
                    paramSlice1 = blocks1[iBlk1]
                    if derivMxToFill is not None:
                        self._fill_dprobs_block(derivMxToFill, felInds, paramSlice1, evalSubTree, paramSlice1,
                                                blk1Comm, mem_limit=None)

                    for iBlk2 in myBlk2Indices:
                        paramSlice2 = blocks2[iBlk2]
                        self.fill_hprobs_block(mx_to_fill, felInds, paramSlice1, paramSlice2, eval_tree,
                                               paramSlice1, paramSlice2, blk2Comm, mem_limit=None)

                    #gather column results: gather axis 2 of mx_to_fill[felInds,blocks1[iBlk1]], dim=(ks,blk1,M)
                    _mpit.gather_slices(blocks2, blk2Owners, mx_to_fill, [felInds, blocks1[iBlk1]],
                                        2, blk1Comm, gather_mem_limit)

                #gather row results; gather axis 1 of mx_to_fill[felInds], dim=(ks,M,M)
                _mpit.gather_slices(blocks1, blk1Owners, mx_to_fill, [felInds],
                                    1, mySubComm, gather_mem_limit)
                if deriv1_mx_to_fill is not None:
                    _mpit.gather_slices(blocks1, blk1Owners, deriv1_mx_to_fill, [felInds],
                                        1, mySubComm, gather_mem_limit)
                if deriv2_mx_to_fill is not None:
                    _mpit.gather_slices(blocks2, blk2Owners, deriv2_mx_to_fill, [felInds],
                                        1, blk1Comm, gather_mem_limit)
                    #Note: deriv2_mx_to_fill gets computed on every inner loop completion
                    # (to save mem) but isn't gathered until now (but using blk1Comm).
                    # (just as pr_mx_to_fill is computed fully on each inner loop *iteration*!)

        #collect/gather results
        subtreeElementIndices = [t.final_element_indices(eval_tree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mx_to_fill, [], 0, comm, gather_mem_limit)
        if deriv1_mx_to_fill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 deriv1_mx_to_fill, [], 0, comm, gather_mem_limit)
        if deriv2_mx_to_fill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 deriv2_mx_to_fill, [], 0, comm, gather_mem_limit)
        if pr_mx_to_fill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 pr_mx_to_fill, [], 0, comm)

        if clip_to is not None and pr_mx_to_fill is not None:
            _np.clip(pr_mx_to_fill, clip_to[0], clip_to[1], out=pr_mx_to_fill)  # in-place clip

        #TODO: check if this works
        #if check:
        #    self._check(eval_tree, spam_label_rows,
        #                pr_mx_to_fill, deriv1_mx_to_fill, mx_to_fill, clip_to)
