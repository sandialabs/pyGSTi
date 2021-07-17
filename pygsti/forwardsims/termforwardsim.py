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

import importlib as _importlib
import warnings as _warnings

import numpy as _np

from pygsti.baseobjs.opcalc import compact_deriv as _compact_deriv, \
    bulk_eval_compact_polynomials as _bulk_eval_compact_polynomials, \
    bulk_eval_compact_polynomials_derivs as _bulk_eval_compact_polynomials_derivs
from pygsti.forwardsims.distforwardsim import DistributableForwardSimulator as _DistributableForwardSimulator
from pygsti.layouts.termlayout import TermCOPALayout as _TermCOPALayout
from pygsti.baseobjs.polynomial import Polynomial as _Polynomial
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.tools import mpitools as _mpit
from pygsti.tools import slicetools as _slct
from pygsti.tools.matrixtools import _fas


# from . import replib
# For debug: sometimes helpful as it prints (python-only) tracebacks from segfaults
# import faulthandler
# faulthandler.enable()


# MEM from .profiler import Profiler


class TermForwardSimulator(_DistributableForwardSimulator):
    """
    A forward-simulation calculator that uses term-path-integration to compute probabilities.

    Parameters
    ----------
    model : Model, optional
        The parent model of this simulator.  It's fine if this is `None` at first,
        but it will need to be set (by assigning `self.model` before using this simulator.

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

    num_atoms : int, optional
        The number of atoms (sub-tables) to use when creating the layout (i.e. when calling
        :method:`create_layout`).  This determines how many units the element (circuit outcome
        probability) dimension is divided into, and doesn't have to correclate with the number of
        processors.  When multiple processors are used, if `num_atoms` is less than the number of
        processors then `num_atoms` should divide the number of processors evenly, so that
        `num_atoms // num_procs` groups of processors can be used to divide the computation
        over parameter dimensions.

    processor_grid : tuple optional
        Specifies how the total number of processors should be divided into a number of
        atom-processors, 1st-parameter-deriv-processors, and 2nd-parameter-deriv-processors.
        Each level of specification is optional, so this can be a 1-, 2-, or 3- tuple of
        integers (or None).  Multiplying the elements of `processor_grid` together should give
        at most the total number of processors.

    param_blk_sizes : tuple, optional
        The parameter block sizes along the first or first & second parameter dimensions - so
        this can be a 0-, 1- or 2-tuple of integers or `None` values.  A block size of `None`
        means that there should be no division into blocks, and that each block processor
        computes all of its parameter indices at once.
    """

    @classmethod
    def _array_types_for_method(cls, method_name):
        # no caches used, so fill methods don't add additional arrays
        return super()._array_types_for_method(method_name)

    def __init__(self, model=None,  # below here are simtype-specific args
                 mode="pruned", max_order=3, desired_perr=0.01, allowed_perr=0.1,
                 min_term_mag=None, max_paths_per_outcome=1000, perr_heuristic="none",
                 max_term_stages=5, path_fraction_threshold=0.9, oob_check_interval=10, cache=None,
                 num_atoms=None, processor_grid=None, param_blk_sizes=None):
        # self.unitary_evolution = False # Unused - idea was to have this flag
        #    allow unitary-evolution calcs to be term-based, which essentially
        #    eliminates the "pRight" portion of all the propagation calcs, and
        #    would require pLeft*pRight => |pLeft|^2
        super().__init__(model, num_atoms, processor_grid, param_blk_sizes)

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
        self.min_term_mag = min_term_mag if (min_term_mag is not None) \
            else desired_perr / (10 * max_paths_per_outcome)   # minimum abs(term coeff) to consider
        self.max_paths_per_outcome = max_paths_per_outcome

        #DEBUG - for profiling cython routines TODO REMOVE (& references)
        #print("DEBUG: termfwdsim: ",self.max_order, self.pathmagnitude_gap, self.min_term_mag)
        self.times_debug = {'tstartup': 0.0, 'total': 0.0,
                            't1': 0.0, 't2': 0.0, 't3': 0.0, 't4': 0.0,
                            'n1': 0, 'n2': 0, 'n3': 0, 'n4': 0}

        # not used except by _do_term_runopt in core.py -- maybe these should move to advancedoptions?
        self.max_term_stages = max_term_stages if mode == "pruned" else 1
        self.path_fraction_threshold = path_fraction_threshold if mode == "pruned" else 0.0
        self.oob_check_interval = oob_check_interval if mode == "pruned" else 0

    def _set_evotype(self, evotype):
        """ Called when the evotype being used (defined by the parent model) changes.
            `evotype` will be `None` when the current model is None"""
        if evotype is not None:
            try:
                self.calclib = _importlib.import_module("pygsti.forwardsims.termforwardsim_calc_" + evotype.name)
            except ImportError:
                self.calclib = _importlib.import_module("pygsti.forwardsims.termforwardsim_calc_generic")
        else:
            self.calclib = None

    def __getstate__(self):
        state = super(TermForwardSimulator, self).__getstate__()
        if 'calclib' in state: del state['calclib']
        #Note: I don't think we need to implement __setstate__ since the model also needs to be reset,
        # and this is done by the parent model which will cause _set_evotype to be called.
        return state

    #OLD - now we have a _set_evotype method.
    #@_ForwardSimulator.model.setter
    #def model(self, val):
    #    _ForwardSimulator.model.fset(self, val)  # set the base class property (self.model)
    #
    #    #Do some additional initialization
    #    if self.model.evotype not in ("svterm", "cterm"):
    #        #raise ValueError(f"Evolution type {self.model.evotype} is incompatible with term-based calculations")
    #        _warnings.warn("Evolution type %s is incompatible with term-based calculations" % self.model.evotype)

    def copy(self):
        """
        Return a shallow copy of this TermForwardSimulator.

        Returns
        -------
        TermForwardSimulator
        """
        return TermForwardSimulator(self.model, self.mode, self.max_order, self.desired_pathmagnitude_gap,
                                    self.allowed_perr, self.min_term_mag, self.max_paths_per_outcome,
                                    self.perr_heuristic, self. max_term_stages, self.path_fraction_threshold,
                                    self.oob_check_interval, self.cache)

    def create_layout(self, circuits, dataset=None, resource_alloc=None, array_types=('E',),
                      derivative_dimension=None, verbosity=0):
        """
        Constructs an circuit-outcome-probability-array (COPA) layout for a list of circuits.

        Parameters
        ----------
        circuits : list
            The circuits whose outcome probabilities should be included in the layout.

        dataset : DataSet
            The source of data counts that will be compared to the circuit outcome
            probabilities.  The computed outcome probabilities are limited to those
            with counts present in `dataset`.

        resource_alloc : ResourceAllocation
            A available resources and allocation information.  These factors influence how
            the layout (evaluation strategy) is constructed.

        array_types : tuple, optional
            A tuple of string-valued array types.  See :method:`ForwardSimulator.create_layout`.

        derivative_dimension : int, optional
            Optionally, the parameter-space dimension used when taking first
            and second derivatives with respect to the cirucit outcome probabilities.  This must be
            non-None when `array_types` contains `'ep'` or `'epp'` types.

        verbosity : int or VerbosityPrinter
            Determines how much output to send to stdout.  0 means no output, higher
            integers mean more output.

        Returns
        -------
        TermCOPALayout
        """
        #Since there's never any "cache" associated with Term-layouts, there's no way to reduce the
        # memory consumption by using more atoms - every processor still needs to hold the entire
        # output array (until we get gather=False mode) - so for now, just create a layout with
        # numAtoms == numProcs.
        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        mem_limit = resource_alloc.mem_limit  # *per-processor* memory limit

        #MEM debug_prof = Profiler(comm)
        #MEM debug_prof.print_memory("CreateLayout1", True)

        printer = _VerbosityPrinter.create_printer(verbosity, resource_alloc)
        nprocs = resource_alloc.comm_size
        num_params = derivative_dimension if (derivative_dimension is not None) else self.model.num_params
        polynomial_vindices_per_int = _Polynomial._vindices_per_int(num_params)
        C = 1.0 / (1024.0**3)

        if mem_limit is not None:
            if mem_limit <= 0:
                raise MemoryError("Attempted layout creation w/memory limit = %g <= 0!" % mem_limit)
            printer.log("Layout creation w/mem limit = %.2fGB" % (mem_limit * C))

        natoms, na, npp, param_dimensions, param_blk_sizes = self._compute_processor_distribution(
            array_types, nprocs, num_params, len(circuits), default_natoms=nprocs)

        printer.log("TermLayout: %d processors divided into %s (= %d) grid along circuit and parameter directions." %
                    (nprocs, ' x '.join(map(str, (na,) + npp)), _np.product((na,) + npp)))
        printer.log("   %d atoms, parameter block size limits %s" % (natoms, str(param_blk_sizes)))
        assert(_np.product((na,) + npp) <= nprocs), "Processor grid size exceeds available processors!"

        layout = _TermCOPALayout(circuits, self.model, dataset, natoms, na, npp, param_dimensions,
                                 param_blk_sizes, resource_alloc, printer)
        #MEM debug_prof.print_memory("CreateLayout2 - nAtoms = %d" % len(layout.atoms), True)
        self._prepare_layout(layout, polynomial_vindices_per_int)
        #MEM debug_prof.print_memory("CreateLayout3 - nEls = %d, nprocs=%d" % (layout.num_elements, nprocs), True)

        return layout

    def _bulk_fill_probs_atom(self, array_to_fill, layout_atom, resource_alloc):

        if not resource_alloc.is_host_leader:
            # (same as "if resource_alloc.host_comm is not None and resource_alloc.host_comm.rank != 0")
            # we cannot further utilize multiplie processors when computing a single block.  The required
            # ending condition is that array_to_fill on each processor has been filled.  But if memory
            # is being shared and resource_alloc contains multiple processors on a single host, we only
            # want *one* (the rank=0) processor to perform the computation, since array_to_fill will be
            # shared memory that we don't want to have muliple procs using simultaneously to compute the
            # same thing.  Thus, we just do nothing on all of the non-root host_comm processors.
            # We could also print a warning (?), or we could carefully guard any shared mem updates
            # using "if resource_alloc.is_host_leader" conditions (if we could use  multiple procs elsewhere).
            return

        nEls = layout_atom.num_elements
        if self.mode == "direct":
            probs = self._prs_directly(layout_atom, resource_alloc)  # could make into a fill_routine? HERE
        else:  # "pruned" or "taylor order"
            polys = layout_atom.merged_compact_polys
            probs = _bulk_eval_compact_polynomials(
                polys[0], polys[1], self.model.to_vector(), (nEls,))  # shape (nElements,) -- could make this a *fill*
        _fas(array_to_fill, [slice(0, array_to_fill.shape[0])], probs)

    def _bulk_fill_dprobs_atom(self, array_to_fill, dest_param_slice, layout_atom, param_slice, resource_alloc):
        if not resource_alloc.is_host_leader:
            return  # see above

        if param_slice is None: param_slice = slice(0, self.model.num_params)
        if dest_param_slice is None: dest_param_slice = slice(0, _slct.length(param_slice))

        if self.mode == "direct":
            dprobs = self._dprs_directly(layout_atom, param_slice, resource_alloc)
        else:  # "pruned" or "taylor order"
            # evaluate derivative of polys
            nEls = layout_atom.num_elements
            polys = layout_atom.merged_compact_polys
            wrtInds = _np.ascontiguousarray(_slct.indices(param_slice), _np.int64)  # for Cython arg mapping
            #OLD dpolys = _compact_deriv(polys[0], polys[1], wrtInds)
            #OLD dprobs = _bulk_eval_compact_polynomials(dpolys[0], dpolys[1],
            #OLD                                         self.model.to_vector(), (nEls, len(wrtInds)))
            dprobs = _bulk_eval_compact_polynomials_derivs(polys[0], polys[1], wrtInds, self.model.to_vector(),
                                                           (nEls, len(wrtInds)))
            #assert(_np.allclose(dprobs, dprobs_chk))

        _fas(array_to_fill, [slice(0, array_to_fill.shape[0]), dest_param_slice], dprobs)

    def _bulk_fill_hprobs_atom(self, array_to_fill, dest_param_slice1, dest_param_slice2, layout_atom,
                               param_slice1, param_slice2, resource_alloc):
        if not resource_alloc.is_host_leader:
            return  # see above

        if param_slice1 is None or param_slice1.start is None: param_slice1 = slice(0, self.model.num_params)
        if param_slice2 is None or param_slice2.start is None: param_slice2 = slice(0, self.model.num_params)
        if dest_param_slice1 is None: dest_param_slice1 = slice(0, _slct.length(param_slice1))
        if dest_param_slice2 is None: dest_param_slice2 = slice(0, _slct.length(param_slice2))

        if self.mode == "direct":
            raise NotImplementedError("hprobs does not support direct path-integral evaluation yet")
            # hprobs = self.hprs_directly(eval_tree, ...)
        else:  # "pruned" or "taylor order"
            # evaluate derivative of polys
            nEls = layout_atom.num_elements
            polys = layout_atom.merged_compact_polys
            wrtInds1 = _np.ascontiguousarray(_slct.indices(param_slice1), _np.int64)
            wrtInds2 = _np.ascontiguousarray(_slct.indices(param_slice2), _np.int64)
            dpolys = _compact_deriv(polys[0], polys[1], wrtInds1)
            hpolys = _compact_deriv(dpolys[0], dpolys[1], wrtInds2)
            hprobs = _bulk_eval_compact_polynomials(
                hpolys[0], hpolys[1], self.model.to_vector(), (nEls, len(wrtInds1), len(wrtInds2)))
        _fas(array_to_fill, [slice(0, array_to_fill.shape[0]), dest_param_slice1, dest_param_slice2], hprobs)

    #DIRECT FNS - keep these around, but they need to be updated (as do routines in fastreplib.pyx)
    #def _prs_directly(self, layout_atom, resource_alloc): #comm=None, mem_limit=None, reset_wts=True, repcache=None):
    #    """
    #    Compute probabilities of `layout`'s circuits using "direct" mode.
    #
    #    Parameters
    #    ----------
    #    layout : CircuitOutcomeProbabilityArrayLayout
    #        The layout.
    #
    #    comm : mpi4py.MPI.Comm, optional
    #        When not None, an MPI communicator for distributing the computation
    #        across multiple processors.  Distribution is performed over
    #        subtrees of eval_tree (if it is split).
    #
    #    mem_limit : int, optional
    #        A rough memory limit in bytes.
    #
    #    reset_wts : bool, optional
    #        Whether term magnitudes should be updated based on current term coefficients
    #        (which are based on the current point in model-parameter space) or not.
    #
    #    repcache : dict, optional
    #        A cache of term representations for increased performance.
    #    """
    #    prs = _np.empty(layout_atom.num_elements, 'd')
    #    #print("Computing prs directly for %d circuits" % len(circuit_list))
    #    if repcache is None: repcache = {}  # new repcache...
    #    k = 0   # *linear* evaluation order so we know final indices are just running
    #    for i in eval_tree.evaluation_order():
    #        circuit = eval_tree[i]
    #        #print("Computing prs directly: circuit %d of %d" % (i,len(circuit_list)))
    #        assert(self.evotype == "svterm")  # for now, just do SV case
    #        fastmode = False  # start with slow mode
    #        wtTol = 0.1
    #        rholabel = circuit[0]
    #        opStr = circuit[1:]
    #        elabels = eval_tree.simplified_circuit_elabels[i]
    #        prs[k:k + len(elabels)] = replib.SV_prs_directly(self, rholabel, elabels, opStr,
    #                                                         repcache, comm, mem_limit, fastmode, wtTol, reset_wts,
    #                                                         self.times_debug)
    #        k += len(elabels)
    #    #print("PRS = ",prs)
    #    return prs
    #
    #def _dprs_directly(self, eval_tree, wrt_slice, comm=None, mem_limit=None, reset_wts=True, repcache=None):
    #    """
    #    Compute probability derivatives of `eval_tree`'s circuits using "direct" mode.
    #
    #    Parameters
    #    ----------
    #    eval_tree : TermEvalTree
    #        The evaluation tree.
    #
    #    wrt_slice : slice
    #        A slice specifying which model parameters to differentiate with respect to.
    #
    #    comm : mpi4py.MPI.Comm, optional
    #        When not None, an MPI communicator for distributing the computation
    #        across multiple processors.  Distribution is performed over
    #        subtrees of eval_tree (if it is split).
    #
    #    mem_limit : int, optional
    #        A rough memory limit in bytes.
    #
    #    reset_wts : bool, optional
    #        Whether term magnitudes should be updated based on current term coefficients
    #        (which are based on the current point in model-parameter space) or not.
    #
    #    repcache : dict, optional
    #        A cache of term representations for increased performance.
    #    """
    #    #Note: Finite difference derivatives are SLOW!
    #    if wrt_slice is None:
    #        wrt_indices = list(range(self.Np))
    #    elif isinstance(wrt_slice, slice):
    #        wrt_indices = _slct.indices(wrt_slice)
    #    else:
    #        wrt_indices = wrt_slice
    #
    #    eps = 1e-6  # HARDCODED
    #    probs = self._prs_directly(eval_tree, comm, mem_limit, reset_wts, repcache)
    #    dprobs = _np.empty((eval_tree.num_final_elements(), len(wrt_indices)), 'd')
    #    orig_vec = self.to_vector().copy()
    #    iParamToFinal = {i: ii for ii, i in enumerate(wrt_indices)}
    #    for i in range(self.Np):
    #        #print("direct dprobs cache %d of %d" % (i,self.Np))
    #        if i in iParamToFinal:  # LATER: add MPI support?
    #            iFinal = iParamToFinal[i]
    #            vec = orig_vec.copy(); vec[i] += eps
    #            self.from_vector(vec, close=True)
    #            dprobs[:, iFinal] = (self._prs_directly(eval_tree,
    #                                                   comm=None,
    #                                                   mem_limit=None,
    #                                                   reset_wts=False,
    #                                                   repcache=repcache) - probs) / eps
    #    self.from_vector(orig_vec, close=True)
    #    return dprobs

    ## ----- Find a "minimal" path set (i.e. find thresholds for each circuit -----
    def _compute_pruned_pathmag_threshold(self, rholabel, elabels, circuit, polynomial_vindices_per_int,
                                          repcache, circuitsetup_cache,
                                          resource_alloc, threshold_guess=None):
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

        polynomial_vindices_per_int : int
            The number of variable indices that can fit into a single platform-width integer
            (can be computed from number of model params, but passed in for performance).

        repcache : dict, optional
            Dictionaries used to cache operator representations  to speed up future
            calls to this function that would use the same set of operations.

        circuitsetup_cache : dict
            A cache holding specialized elements that store and eliminate
            the need to recompute per-circuit information.

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

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
        #Cache hold *compact* polys now: see _prs_as_compact_polynomials
        #cache_keys = [(self.max_order, rholabel, elabel, circuit) for elabel in tuple(elabels)]
        #if self.cache is not None and all([(ck in self.cache) for ck in cache_keys]):
        #    return [ self.cache[ck] for ck in cache_keys ]

        if threshold_guess is None: threshold_guess = -1.0  # use negatives to signify "None" in C
        circuitsetup_cache = {}  # DEBUG REMOVE?
        #repcache = {}  # DEBUG REMOVE

        npaths, threshold, target_sopm, achieved_sopm = \
            self.calclib.find_best_pathmagnitude_threshold(
                self, rholabel, elabels, circuit, polynomial_vindices_per_int, repcache, circuitsetup_cache,
                resource_alloc.comm, resource_alloc.mem_limit, self.desired_pathmagnitude_gap,
                self.min_term_mag, self.max_paths_per_outcome, threshold_guess
            )
        # sopm = "sum of path magnitudes"

        return npaths, threshold, target_sopm, achieved_sopm

    def _find_minimal_paths_set_atom(self, layout_atom, resource_alloc, exit_after_this_many_failures=0):
        """
        Find the minimal (smallest) path set that achieves the desired accuracy conditions.

        Parameters
        ----------
        layout_atom : _TermCOPALayoutAtom
            The probability array layout containing the circuits to find a path-set for.

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        exit_after_this_many_failures : int, optional
           If > 0, give up after this many circuits fail to meet the desired accuracy criteria.
           This short-circuits doomed attempts to find a good path set so they don't take too long.

        Returns
        -------
        TermPathSetAtom
        """
        tot_npaths = 0
        tot_target_sopm = 0
        tot_achieved_sopm = 0

        #We're only testing how many failures there are, don't update the "locked in" persistent
        # set of paths given by layout_atom.percircuit_p_polys and layout_atom.pathset.highmag_termrep_cache
        # - just use temporary caches:
        repcache = {}
        circuitsetup_cache = {}

        thresholds = {}
        num_failed = 0  # number of circuits which fail to achieve the target sopm
        failed_circuits = []
        polynomial_vindices_per_int = _Polynomial._vindices_per_int(self.model.num_params)

        for sep_povm_circuit in layout_atom.expanded_circuits:
            rholabel = sep_povm_circuit.circuit_without_povm[0]
            opstr = sep_povm_circuit.circuit_without_povm[1:]
            elabels = sep_povm_circuit.full_effect_labels

            npaths, threshold, target_sopm, achieved_sopm = \
                self._compute_pruned_pathmag_threshold(rholabel, elabels, opstr, polynomial_vindices_per_int,
                                                       repcache, circuitsetup_cache,
                                                       resource_alloc, None)  # add guess?
            thresholds[sep_povm_circuit] = threshold

            if achieved_sopm < target_sopm:
                num_failed += 1
                failed_circuits.append(sep_povm_circuit)  # (circuit,npaths, threshold, target_sopm, achieved_sopm))
                if exit_after_this_many_failures > 0 and num_failed == exit_after_this_many_failures:
                    return _AtomicTermPathSet(None, None, None, 0, 0, num_failed)

            tot_npaths += npaths
            tot_target_sopm += target_sopm
            tot_achieved_sopm += achieved_sopm

        #if comm is None or comm.Get_rank() == 0:
        comm = resource_alloc.comm
        rank = resource_alloc.comm_rank
        nC = len(layout_atom.expanded_circuits)
        max_npaths = self.max_paths_per_outcome * layout_atom.num_elements

        if rank == 0:
            rankStr = "Rank%d: " % rank if comm is not None else ""
            print(("%sPruned path-integral: kept %d paths (%.1f%%) w/magnitude %.4g "
                   "(target=%.4g, #circuits=%d, #failed=%d)") %
                  (rankStr, tot_npaths, 100 * tot_npaths / max_npaths, tot_achieved_sopm, tot_target_sopm,
                   nC, num_failed))
            print("%s  (avg per circuit paths=%d, magnitude=%.4g, target=%.4g)" %
                  (rankStr, tot_npaths // nC, tot_achieved_sopm / nC, tot_target_sopm / nC))

        return _AtomicTermPathSet(thresholds, repcache, circuitsetup_cache, tot_npaths, max_npaths, num_failed)

    # should assert(nFailures == 0) at end - this is to prep="lock in" probs & they should be good
    def find_minimal_paths_set(self, layout, exit_after_this_many_failures=0):
        """
        Find a good, i.e. minimal, path set for the current model-parameter space point.

        Parameters
        ----------
        layout : TermCOPALayout
            The layout specifiying the quantities (circuit outcome probabilities) to be
            computed, and related information.

        exit_after_this_many_failures : int, optional
           If > 0, give up after this many circuits fail to meet the desired accuracy criteria.
           This short-circuits doomed attempts to find a good path set so they don't take too long.

        Returns
        -------
        TermPathSet
        """
        atom_resource_alloc = layout.resource_alloc('atom-processing')
        local_atom_pathsets = []
        for layout_atom in layout.atoms:
            if self.mode == "pruned":
                pathset = self._find_minimal_paths_set_atom(layout_atom, atom_resource_alloc,
                                                            exit_after_this_many_failures)
            else:
                pathset = _AtomicTermPathSet(None, None, None, 0, 0, 0)
            local_atom_pathsets.append(pathset)

        return TermPathSet(local_atom_pathsets, layout.resource_alloc().comm)

    ## ----- Get maximum possible sum-of-path-magnitudes and that which was actually achieved -----
    def _circuit_achieved_and_max_sopm(self, rholabel, elabels, circuit, repcache, threshold):
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

        threshold : float
            path-magnitude threshold.  Only sum path magnitudes above or equal to this threshold.

        Returns
        -------
        achieved_sopm : float
            The achieved sum-of-path-magnitudes. (summed over all circuit outcomes)

        max_sopm : float
            The maximum possible sum-of-path-magnitudes. (summed over all circuit outcomes)
        """
        return self.calclib.circuit_achieved_and_max_sopm(
            self, rholabel, elabels, circuit, repcache, threshold, self.min_term_mag)

    def _achieved_and_max_sopm_atom(self, layout_atom):
        """
        Compute the achieved and maximum possible sum-of-path-magnitudes for a single layout atom.

        This gives a sense of how accurately the current path set is able
        to compute probabilities.

        Parameters
        ----------
        layout_atom : _TermCOPALayoutAtom
            The probability array layout specifying the circuits and outcomes.

        Returns
        -------
        numpy.ndarray
        """
        achieved_sopm = []
        max_sopm = []

        for sep_povm_circuit in layout_atom.expanded_circuits:
            # must have selected a set of paths for this to be populated!
            current_threshold, _ = layout_atom.percircuit_p_polys[sep_povm_circuit]

            rholabel = sep_povm_circuit.circuit_without_povm[0]
            opstr = sep_povm_circuit.circuit_without_povm[1:]
            elabels = sep_povm_circuit.full_effect_labels

            achieved, maxx = self._circuit_achieved_and_max_sopm(rholabel, elabels, opstr,
                                                                 layout_atom.pathset.highmag_termrep_cache,
                                                                 current_threshold)
            achieved_sopm.extend(list(achieved))
            max_sopm.extend(list(maxx))

        assert(len(achieved_sopm) == len(max_sopm) == layout_atom.num_elements)
        return _np.array(achieved_sopm, 'd'), _np.array(max_sopm, 'd')

    def _bulk_fill_achieved_and_max_sopm(self, achieved_sopm, max_sopm, layout):
        """
        Compute element arrays of achieved and maximum-possible sum-of-path-magnitudes.

        These values are computed for the current set of paths contained in `eval_tree`.

        Parameters
        ----------
        achieved_sopm : numpy.ndarray
            An array containing the per-circuit-outcome achieved sum-of-path-magnitudes.

        max_sopm : numpy.ndarray
            An array containing the per-circuit-outcome maximum sum-of-path-magnitudes.

        layout : TermCOPALayout
            The layout specifiying the quantities (circuit outcome probabilities) to be
            computed, and related information.

        Returns
        -------
        None
        """
        assert(self.mode == "pruned")
        #atom_resource_alloc = layout.resource_alloc('atom-processing')
        #MEM debug_prof = Profiler(resource_alloc.comm)

        for layout_atom in layout.atoms:
            # compute SOPM for layout_atom
            elInds = layout_atom.element_slice
            # MEM debug_prof.print_memory("_bulk_achieved_and_max_sop1", True)
            self.calclib.refresh_magnitudes_in_repcache(layout_atom.pathset.highmag_termrep_cache,
                                                        self.model.to_vector())
            # MEM debug_prof.print_memory("_bulk_achieved_and_max_sop2", True)
            achieved, maxx = self._achieved_and_max_sopm_atom(layout_atom)
            # MEM debug_prof.print_memory("_bulk_achieved_and_max_sop3", True)
            _fas(max_sopm, [elInds], maxx)
            _fas(achieved_sopm, [elInds], achieved)

    ## ----- A couple more bulk_* convenience functions that wrap _bulk_achieved_and_max_sopm -----
    def bulk_test_if_paths_are_sufficient(self, layout, probs, verbosity=0):
        """
        Determine whether `layout`'s current path set (perhaps heuristically) achieves the desired accuracy.

        The current path set is determined by the current (per-circuti) path-magnitude thresholds
        (stored in the evaluation tree) and the current parameter-space point (also reflected in
        the terms cached in the evaluation tree).

        Parameters
        ----------
        layout : TermCOPALayout
            The layout specifiying the quantities (circuit outcome probabilities) to be
            computed, and related information.

        probs : numpy.ndarray
            The element array of (approximate) circuit outcome probabilities.  This is
            needed because some heuristics take into account an probability's value when
            computing an acceptable path-magnitude gap.

        verbosity : int or VerbosityPrinter, optional
            An integer verbosity level or printer object for displaying messages.

        Returns
        -------
        bool
        """
        if self.mode != "pruned":
            return True  # no "failures" for non-pruned-path mode

        printer = _VerbosityPrinter.create_printer(verbosity, layout.resource_alloc())
        resource_alloc = layout.resource_alloc()
        atom_resource_alloc = layout.resource_alloc('atom-processing')

        # # done in _bulk_achieved_and_max_sopm:
        # self.calclib.refresh_magnitudes_in_repcache(eval_tree.highmag_termrep_cache, self.to_vector())
        max_sopm = layout.allocate_local_array('e', 'd')
        achieved_sopm = layout.allocate_local_array('e', 'd')
        self._bulk_fill_achieved_and_max_sopm(achieved_sopm, max_sopm, layout)  # fills *local* quantities

        # a strict bound on the error in each outcome probability, but often pessimistic
        gaps = max_sopm - achieved_sopm
        #assert(_np.all(gaps >= 0))  # gaps is *local* so we don't want some procs to raise errors when others don't

        ret = True
        if self.perr_heuristic == "none":
            nFailures = _np.count_nonzero(gaps > self.allowed_perr)
            nFailures = resource_alloc.allreduce_sum_simple(nFailures, unit_ralloc=atom_resource_alloc)
            if nFailures > 0:
                printer.log("Paths are insufficient! (%d failures using strict error bound of %g)"
                            % (nFailures, self.allowed_perr))
                ret = False
        elif self.perr_heuristic == "scaled":
            scale = probs / achieved_sopm
            nFailures = _np.count_nonzero(gaps * scale > self.allowed_perr)
            nFailures = resource_alloc.allreduce_sum_simple(nFailures, unit_ralloc=atom_resource_alloc)
            if nFailures > 0:
                printer.log("Paths are insufficient! (%d failures using %s heuristic with error bound of %g)"
                            % (nFailures, self.perr_heuristic, self.allowed_perr))
                ret = False
        elif self.perr_heuristic == "meanscaled":
            scale = probs / achieved_sopm
            bFailed = 1.0 if (_np.mean(gaps * scale) > self.allowed_perr) else 0.0
            bFailed = resource_alloc.allreduce_sum_simple(bFailed, unit_ralloc=atom_resource_alloc)
            if bFailed > 0.0:
                printer.log("Paths are insufficient! (Using %s heuristic with error bound of %g)"
                            % (self.perr_heuristic, self.allowed_perr))
                ret = False
        else:
            raise ValueError("Unknown probability-error heuristic name: %s" % self.perr_heuristic)

        resource_alloc.host_comm_barrier()  # make sure host is finished before freeing
        layout.free_local_array(max_sopm)
        layout.free_local_array(achieved_sopm)
        return ret

    def bulk_sopm_gaps(self, layout):
        """
        Compute an element array sum-of-path-magnitude gaps (the difference between maximum and achieved).

        These values are computed for the current set of paths contained in `eval_tree`.

        Parameters
        ----------
        layout : TermCOPALayout
            The layout specifiying the quantities (circuit outcome probabilities) to be
            computed, and related information.

        Returns
        -------
        numpy.ndarray
            An array containing the per-circuit-outcome sum-of-path-magnitude gaps.
        """
        max_sopm = layout.allocate_local_array('e', 'd')
        achieved_sopm = layout.allocate_local_array('e', 'd')
        self._bulk_fill_achieved_and_max_sopm(achieved_sopm, max_sopm, layout)  # fills *local* quantities

        global_max_sopm = layout.allgather_local_array('e', max_sopm)  # return_shared=True?
        global_achieved_sopm = layout.allgather_local_array('e', achieved_sopm)  # return_shared=True?
        layout.free_local_array(max_sopm)
        layout.free_local_array(achieved_sopm)

        gaps = global_max_sopm - global_achieved_sopm
        # Gaps can be slightly negative b/c of SMALL magnitude given to acutually-0-weight paths.
        assert(_np.all(gaps >= -1e-6))
        return _np.clip(gaps, 0, None)

    ## ----- Jacobian of gaps (don't need jacobian of achieved and max SOPM separately) -----
    def _achieved_and_max_sopm_jacobian_atom(self, layout_atom):
        """
        Compute the jacobian of the achieved and maximum possible sum-of-path-magnitudes for a single layout atom.

        Parameters
        ----------
        layout_atom : _TermCOPALayoutAtom
            The probability array layout specifying the circuits and outcomes.

        Returns
        -------
        achieved_sopm_jacobian: numpy.ndarray
            The jacobian of the achieved sum-of-path-magnitudes.

        max_sopm_jacobian: numpy.ndarray
            The jacobian of the maximum possible sum-of-path-magnitudes.
        """
        paramvec = self.model.to_vector()
        Np = len(paramvec)

        nEls = layout_atom.num_elements
        polys = layout_atom.merged_achievedsopm_compact_polys
        #OLD dpolys = _compact_deriv(polys[0], polys[1], _np.arange(Np))
        #OLD d_achieved_mags = _bulk_eval_compact_polynomials_complex(
        #OLD    dpolys[0], dpolys[1], _np.abs(paramvec), (nEls, Np))
        d_achieved_mags = _bulk_eval_compact_polynomials_derivs(polys[0], polys[1], _np.arange(Np),
                                                                _np.abs(paramvec), (nEls, Np))
        #assert(_np.allclose(d_achieved_mags, d_achieved_mags_chk))

        assert(_np.linalg.norm(_np.imag(d_achieved_mags)) < 1e-8)
        d_achieved_mags = d_achieved_mags.real
        d_achieved_mags[:, (paramvec < 0)] *= -1

        d_max_sopms = _np.empty((nEls, Np), 'd')
        k = 0  # current element position for loop below

        for sep_povm_circuit in layout_atom.expanded_circuits:
            rholabel = sep_povm_circuit.circuit_without_povm[0]
            opstr = sep_povm_circuit.circuit_without_povm[1:]
            elabels = sep_povm_circuit.full_effect_labels

            #Get MAX-SOPM for circuit outcomes and thereby the SOPM gap (via MAX - achieved)
            # Here we take d(MAX) (above merged_achievedsopm_compact_polys give d(achieved)).  Since each
            # MAX-SOPM value is a product of max term magnitudes, to get deriv we use the chain rule:
            partial_ops = [self.model.circuit_layer_operator(rholabel, 'prep')]
            for glbl in opstr:
                partial_ops.append(self.model.circuit_layer_operator(glbl, 'op'))
            Eops = [self.model.circuit_layer_operator(elbl, 'povm') for elbl in elabels]
            partial_op_maxmag_values = [op.total_term_magnitude() for op in partial_ops]
            Eop_maxmag_values = [Eop.total_term_magnitude() for Eop in Eops]
            maxmag_partial_product = _np.product(partial_op_maxmag_values)
            maxmag_products = [maxmag_partial_product * Eop_val for Eop_val in Eop_maxmag_values]

            deriv = _np.zeros((len(elabels), Np), 'd')
            for i in range(len(partial_ops)):  # replace i-th element of product with deriv
                dop_local = partial_ops[i].total_term_magnitude_deriv()
                dop_global = _np.zeros(Np, 'd')
                dop_global[partial_ops[i].gpindices] = dop_local
                dop_global /= partial_op_maxmag_values[i]

                for j in range(len(elabels)):
                    deriv[j, :] += dop_global * maxmag_products[j]

            for j in range(len(elabels)):  # replace final element with appropriate derivative
                dop_local = Eops[j].total_term_magnitude_deriv()
                dop_global = _np.zeros(Np, 'd')
                dop_global[Eops[j].gpindices] = dop_local
                dop_global /= Eop_maxmag_values[j]
                deriv[j, :] += dop_global * maxmag_products[j]

            d_max_sopms[k:k + len(elabels), :] = deriv
            k += len(elabels)

        return d_achieved_mags, d_max_sopms

    def _sopm_gaps_jacobian_atom(self, layout_atom):
        """
        Compute the jacobian of the (maximum-possible - achieved) sum-of-path-magnitudes for a single layout atom.

        Parameters
        ----------
        layout_atom : _TermCOPALayoutAtom
            The probability array layout.

        Returns
        -------
        numpy.ndarray
            The jacobian of the sum-of-path-magnitudes gap.
        """
        d_achieved_mags, d_max_sopms = self._achieved_and_max_sopm_jacobian(layout_atom)
        dgaps = d_max_sopms - d_achieved_mags
        return dgaps

    def bulk_sopm_gaps_jacobian(self, layout):
        """
        Compute the jacobian of the the output of :method:`bulk_sopm_gaps`.

        Parameters
        ----------
        layout : TermCOPALayout
            The layout specifiying the quantities (circuit outcome probabilities) to be
            computed, and related information.

        resource_alloc : ResourceAllocation, optional
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        Returns
        -------
        numpy.ndarray
            An number-of-elements by number-of-model-parameters array containing the jacobian
            of the sum-of-path-magnitude gaps.
        """
        assert(self.mode == "pruned")
        termgap_penalty_jac = layout.allocate_local_array('ep', 'd')
        # OLD: _np.empty((layout.num_elements, self.model.num_params), 'd')

        #TODO: need to update this function for distributed mem if it's ever used (currently it's not)
        # -> _sopm_gaps_jacobian_atom(layout_atom) needs to compute derivs wrt just the *local* params (previously all)
        # -> this routine used to gather the jac, but it shouldn't need to do this in a dist. mem framework -- make sure
        #    upstream callers know what to expect, and if we return an allocated local array they'll need to free it (we
        #    should probably just *fill* an already allocated jac array)
        raise NotImplementedError("Need to update this function to use distributed memory!")

        for layout_atom in layout.atoms:
            elInds = layout_atom.element_slice
            self.calclib.refresh_magnitudes_in_repcache(layout_atom.pathset.highmag_termrep_cache,
                                                        self.model.to_vector())
            gap_jacs = self._sopm_gaps_jacobian_atom(layout_atom)
            #gap_jacs[ _np.where(gaps < self.pathmagnitude_gap) ] = 0.0  # set deriv to zero where gap was clipped to 0
            _fas(termgap_penalty_jac, [elInds], gap_jacs)

        #REMOVE #collect/gather results
        #all_atom_element_slices = [atom.element_slice for atom in layout.atoms]
        #_mpit.gather_slices(all_atom_element_slices, atomOwners, termgap_penalty_jac, [], 0, resource_alloc.comm)

        return termgap_penalty_jac

    ## ----- Select, or "lock in" a path set, which includes preparing to compute approx. probs using these paths -----
    def _prs_as_pruned_polynomial_reps(self,
                                       threshold,
                                       rholabel,
                                       elabels,
                                       circuit,
                                       polynomial_vindices_per_int,
                                       repcache,
                                       circuitsetup_cache,
                                       resource_alloc,
                                       mode="normal"):
        """
        Computes polynomial-representations of circuit-outcome probabilities.

        In particular, the circuit-outcomes under consideration share the same state
        preparation and differ only in their POVM effects.  Employs a truncated or pruned
        path-integral approach, as opposed to just including everything up to some Taylor
        order as in :method:`_prs_as_polynomials`.

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

        polynomial_vindices_per_int : int
            The number of variable indices that can fit into a single platform-width integer
            (can be computed from number of model params, but passed in for performance).

        repcache : dict, optional
            Dictionaries used to cache operator representations  to speed up future
            calls to this function that would use the same set of operations.

        circuitsetup_cache : dict
            A cache holding specialized elements that store and eliminate
            the need to recompute per-circuit information.

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        mode : {"normal", "achieved_sopm"}
            Controls whether polynomials are actually computed (`"normal"`) or whether only the
            achieved sum-of-path-magnitudes is computed (`"achieved_sopm"`).  The latter mode is
            useful when a `threshold` is being tested but not committed to, as computing only the
            achieved sum-of-path-magnitudes is significantly faster.

        Returns
        -------
        list
           A list of :class:`PolynomialRep` objects.  These polynomial represetations are essentially
           bare-bones polynomials stored efficiently for performance.  (To get a full
           :class:`Polynomial` object, use :classmethod:`Polynomial.from_rep`.)
        """
        #Cache hold *compact* polys now: see _prs_as_compact_polynomials
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

        poly_reps = self.calclib.compute_pruned_path_polynomials_given_threshold(
            threshold, self, rholabel, elabels, circuit, polynomial_vindices_per_int, repcache,
            circuitsetup_cache, resource_alloc.comm, resource_alloc.mem_limit, fastmode)
        # sopm = "sum of path magnitudes"

        #TODO REMOVE this case -- we don't check for cache hits anymore; I think we can just set prps = poly_reps here
        if len(poly_reps) == 0:  # HACK - length=0 => there's a cache hit, which we signify by None here
            prps = None
        else:
            prps = poly_reps

        return prps

    def _select_paths_set_atom(self, layout_atom, pathset, resource_alloc):
        """
        Selects (makes "current") a path set *and* computes polynomials the new set for a single layout atom.

        This routine updates the information held in `layout_atom`. After this call,
        `layout_atom.pathset == pathset`.

        Parameters
        ----------
        layout_atom : _TermCOPALayoutAtom
            The probability array layout whose path-set is being set.

        pathset : PathSet
            The path set to select.

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        Returns
        -------
        None
        """

        # **This may be out of date**
        # We're finding and "locking in" a set of paths to use in subsequent evaluations.  This
        # means we're going to re-compute the high-magnitude terms for each operation (in
        # self.pathset.highmag_termrep_cache) and re-compute the thresholds (in self.percircuit_p_polys)
        # for each circuit (using the computed high-magnitude terms).  This all occurs for
        # the particular current value of the parameter vector (found via calc.to_vector());
        # these values determine what is a "high-magnitude" term and the path magnitudes that are
        # summed to get the overall sum-of-path-magnitudes for a given circuit outcome.

        layout_atom.pathset = pathset
        layout_atom.percircuit_p_polys = {}
        repcache = layout_atom.pathset.highmag_termrep_cache
        circuitsetup_cache = layout_atom.pathset.circuitsetup_cache
        thresholds = layout_atom.pathset.thresholds
        polynomial_vindices_per_int = _Polynomial._vindices_per_int(self.model.num_params)

        all_compact_polys = []  # holds one compact polynomial per final *element*

        for sep_povm_circuit in layout_atom.expanded_circuits:
            #print("Computing pruned-path polynomial for ", sep_povm_circuit)
            rholabel = sep_povm_circuit.circuit_without_povm[0]
            opstr = sep_povm_circuit.circuit_without_povm[1:]
            elabels = sep_povm_circuit.full_effect_labels
            threshold = thresholds[sep_povm_circuit]

            raw_polyreps = self._prs_as_pruned_polynomial_reps(
                threshold, rholabel, elabels, opstr, polynomial_vindices_per_int,
                repcache, circuitsetup_cache, resource_alloc)

            compact_polys = [polyrep.compact_complex() for polyrep in raw_polyreps]
            layout_atom.percircuit_p_polys[sep_povm_circuit] = (threshold, compact_polys)
            all_compact_polys.extend(compact_polys)  # ok b/c *linear* evaluation order

        tapes = all_compact_polys  # each "compact polynomials" is a (vtape, ctape) 2-tupe
        vtape = _np.concatenate([t[0] for t in tapes])  # concat all the vtapes
        ctape = _np.concatenate([t[1] for t in tapes])  # concat all teh ctapes
        layout_atom.merged_compact_polys = (vtape, ctape)  # Note: ctape should always be complex here
        return

    def _prs_as_polynomials(self, rholabel, elabels, circuit, polynomial_vindices_per_int,
                            resource_alloc, fastmode=True):
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

        polynomial_vindices_per_int : int
            The number of variable indices that can fit into a single platform-width integer
            (can be computed from number of model params, but passed in for performance).

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        fastmode : bool, optional
            Whether to use a faster and slightly more memory-hungry implementation for
            computing the polynomial terms.  (Usually best to leave this as `True`).

        Returns
        -------
        list
            A list of Polynomial objects.
        """
        poly_reps = self.calclib.prs_as_polynomials(self, rholabel, elabels, circuit, polynomial_vindices_per_int,
                                                    resource_alloc.comm, resource_alloc.mem_limit, fastmode)
        return [_Polynomial.from_rep(rep) for rep in poly_reps]

    def _prs_as_compact_polynomials(self, rholabel, elabels, circuit, polynomial_vindices_per_int, resource_alloc):
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

        polynomial_vindices_per_int : int
            The number of variable indices that can fit into a single platform-width integer
            (can be computed from number of model params, but passed in for performance).

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        Returns
        -------
        list
            A list of Polynomial objects.
        """
        cache_keys = [(self.max_order, rholabel, elabel, circuit) for elabel in tuple(elabels)]
        if self.cache is not None and all([(ck in self.cache) for ck in cache_keys]):
            return [self.cache[ck] for ck in cache_keys]

        raw_prps = self._prs_as_polynomials(rholabel, elabels, circuit, polynomial_vindices_per_int, resource_alloc)
        prps = [poly.compact(complex_coeff_tape=True) for poly in raw_prps]
        # create compact polys w/*complex* coeffs always since we're likely
        # going to concatenate a bunch of them.

        if self.cache is not None:
            for ck, poly in zip(cache_keys, prps):
                self.cache[ck] = poly
        return prps

    def _cache_p_polynomials(self, layout_atom, resource_alloc, polynomial_vindices_per_int):
        """
        Compute and cache the compact-form polynomials that evaluate the probabilities of a single layout atom.

        These polynomials corresponding to all this tree's operation sequences sandwiched
        between each state preparation and effect.  The result is cached to speed
        up subsequent calls.

        Parameters
        ----------
        layout_atom : _TermCOPALayoutAtom
            The probability array layout containing the circuits to compute polynomials for.

        resource_alloc : ResourceAllocation
            Available resources for this computation. Includes the number of processors
            (MPI comm) and memory limit.

        polynomial_vindices_per_int : int
            The number of variable indices that can fit into a single platform-width integer
            (can be computed from number of model params, but passed in for performance).

        Returns
        -------
        None
        """
        layout_atom.pathset = None  # used for "pruned" mode, but not here
        layout_atom.percircuit_p_polys = None  # used for "pruned" mode, but not here

        all_compact_polys = []  # holds one compact polynomial per final *element*
        for sep_povm_circuit in layout_atom.expanded_circuits:
            rholabel = sep_povm_circuit.circuit_without_povm[0]
            opstr = sep_povm_circuit.circuit_without_povm[1:]
            elabels = sep_povm_circuit.full_effect_labels
            compact_polys = self._prs_as_compact_polynomials(rholabel, elabels, opstr, polynomial_vindices_per_int,
                                                             resource_alloc)
            all_compact_polys.extend(compact_polys)  # ok b/c *linear* evaluation order

        tapes = all_compact_polys  # each "compact polynomials" is a (vtape, ctape) 2-tupe
        vtape = _np.concatenate([t[0] for t in tapes])  # concat all the vtapes
        ctape = _np.concatenate([t[1] for t in tapes])  # concat all teh ctapes
        layout_atom.merged_compact_polys = (vtape, ctape)  # Note: ctape should always be complex here

    # should assert(nFailures == 0) at end - this is to prep="lock in" probs & they should be good
    def select_paths_set(self, layout, path_set):
        """
        Selects (makes "current") a path set *and* computes polynomials the new set.

        Parameters
        ----------
        layout : TermCOPALayout
            The layout whose path-set should be set.

        path_set : PathSet
            The path set to select.

        Returns
        -------
        None
        """
        atom_resource_alloc = layout.resource_alloc('atom-processing')
        local_path_sets = list(path_set.local_atom_pathsets) if self.mode == "pruned" else None
        polynomial_vindices_per_int = _Polynomial._vindices_per_int(self.model.num_params)

        for layout_atom in layout.atoms:
            if self.mode == "pruned":
                sub_pathset = local_path_sets.pop(0)  # take the next path set (assumes same order of calling atoms!)
                self._select_paths_set_atom(layout_atom, sub_pathset, atom_resource_alloc)
                #This computes (&caches) polys for this path set as well
            else:
                self._cache_p_polynomials(layout_atom, atom_resource_alloc, polynomial_vindices_per_int)

    ## ----- Other functions -----
    def _prepare_layout(self, layout, polynomial_vindices_per_int):
        """
        Performs preparatory work for computing circuit outcome probabilities.

        Finds a good path set that meets (if possible) the accuracy requirements
        and computes needed polynomials.

        Parameters
        ----------
        layout : TermCOPALayout
            The layout to prepare.

        polynomial_vindices_per_int : int
            The number of variable indices that can fit into a single platform-width integer
            (can be computed from number of model params, but passed in for performance).

        Returns
        -------
        None
        """
        # MEM debug_prof = Profiler(resource_alloc.comm)
        # MEM if resource_alloc.comm is None or resource_alloc.comm.rank == 0:
        # MEM     print("Finding & selecting path set using %d atoms" % len(layout.atoms))

        atom_resource_alloc = layout.resource_alloc('atom-processing')
        local_nFailures = 0

        for layout_atom in layout.atoms:
            if self.mode == "pruned":
                # MEM debug_prof.print_memory("find_and_select_pathset1 - nEls = %d, nExpanded=%d, rank=%d" %
                # MEM                         (layout_atom.num_elements, len(layout_atom.expanded_circuits),
                # MEM                         resource_alloc.comm.rank), True)
                pathset = self._find_minimal_paths_set_atom(layout_atom, atom_resource_alloc,
                                                            exit_after_this_many_failures=0)
                # MEM debug_prof.print_memory("find_and_select_pathset2", True)
                # MEM DEBUG import sys; sys.exit(0)
                self._select_paths_set_atom(layout_atom, pathset, atom_resource_alloc)  # "locks in" path set
                local_nFailures += pathset.num_failures
            else:
                self._cache_p_polynomials(layout_atom, atom_resource_alloc, polynomial_vindices_per_int)

        # Get the number of failures to create an accurate-enough polynomial (# of circuit probabilities, i.e. elements)
        if atom_resource_alloc.comm is not None and atom_resource_alloc.comm.rank > 0:
            local_nFailures = 0  # don't count atom-processor non-root failure counts in sum (these are duplicates)
        nTotFailed = _mpit.sum_across_procs(local_nFailures, layout.resource_alloc().comm)
        if nTotFailed > 0:
            _warnings.warn(("Unable to find a path set that achieves the desired "
                            "pathmagnitude gap (%d circuits failed)") % nTotFailed)


class _TermPathSetBase(object):
    """
    A set of error-term paths.

    Each such path is comprised of a single "term" (usually a Taylor term of an
    error-generator expansion) for each gate operation or circuit layer (more
    generally, each factor within the product that evaluates to the probability).

    A set of paths is specified by giving a path-magnitude threshold for each
    circuit in a COPA layout.  All paths with magnitude less than this threshold
    are a part of the path set.  The term magnitudes that determine a path magnitude
    are held in Term objects resulting from a Model at a particular parameter-space
    point.  Representations of these term objects (actually just the "high-magnitude" ones,
    as determined by a different, term-magnitude, threshold) are also held
    within the path set.

    Parameters
    ----------
    npaths : int
        The number of total paths.

    maxpaths : int
        The maximum-allowed-paths limit that was in place when this
        path set was created.

    nfailed : int
        The number of circuits that failed to meet the desired accuracy
        (path-magnitude gap) requirements.
    """

    def __init__(self, npaths, maxpaths, nfailed):
        self.npaths = npaths
        self.max_allowed_paths = maxpaths
        self.num_failures = nfailed  # number of failed *circuits* (not outcomes)

    @property
    def allowed_path_fraction(self):
        """
        The fraction of maximal allowed paths that are in this path set.

        Returns
        -------
        float
        """
        return self.npaths / self.max_allowed_paths


class _AtomicTermPathSet(_TermPathSetBase):
    """
    A path set, as specified for each atom of a :class:`TermCOPALayout`.

    Parameters
    ----------
    thresholds : dict
        A dictionary whose keys are circuits and values are path-magnitude thresholds.
        These thresholds store what

    highmag_termrep_cache : dict
        A dictionary whose keys are gate or circuit-layer labels and whose values are
        internally-used "rep-cache" elements that each hold a list of the term representations
        for that gate having a "high" magnitude (magnitude above some threshold).  This
        cache is an essential link between the path-magnitude thresholds in `thresholds` and
        the actual set of paths that are evaluated by processing `layout_atom` (e.g. updating
        this cache by re-computing term magnitudes at a new parameter-space point will also
        update the set of paths that are evaluated given the *same* set of `thresholds`).

    circuitsetup_cache : dict
        A dictionary that caches per-circuit setup information and can be used to
        speed up multiple calls which use the same circuits.

    npaths : int
        The number of total paths.

    maxpaths : int
        The maximum-allowed-paths limit that was in place when this
        path set was created.

    nfailed : int
        The number of circuits that failed to meet the desired accuracy
        (path-magnitude gap) requirements.
    """
    def __init__(self, thresholds, highmag_termrep_cache,
                 circuitsetup_cache, npaths, maxpaths, nfailed):
        super().__init__(npaths, maxpaths, nfailed)
        self.thresholds = thresholds
        self.highmag_termrep_cache = highmag_termrep_cache
        self.circuitsetup_cache = circuitsetup_cache


class TermPathSet(_TermPathSetBase):
    """
    A path set for a split :class:`TermEvalTree`.

    Parameters
    ----------
    local_atom_pathsets : list
        A list of path sets for each of the *local* layout atom (i.e. the
        atoms assigned to the current processor).

    comm : mpi4py.MPI.Comm
        When not None, an MPI communicator for distributing the computation
        across multiple processors.
    """
    def __init__(self, local_atom_pathsets, comm):

        #Get local-atom totals
        nTotPaths = sum([sps.npaths for sps in local_atom_pathsets])
        nTotFailed = sum([sps.num_failures for sps in local_atom_pathsets])
        nAllowed = sum([sps.max_allowed_paths for sps in local_atom_pathsets])

        #Get global totals
        nTotFailed = _mpit.sum_across_procs(nTotFailed, comm)
        nTotPaths = _mpit.sum_across_procs(nTotPaths, comm)
        nAllowed = _mpit.sum_across_procs(nAllowed, comm)

        super().__init__(nTotPaths, nAllowed, nTotFailed)
        self.local_atom_pathsets = local_atom_pathsets
