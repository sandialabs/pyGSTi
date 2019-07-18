""" Defines the MapForwardSimulator calculator class"""
from __future__ import division, print_function, absolute_import, unicode_literals
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

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from ..tools import compattools as _compat
from ..tools.matrixtools import _fas
from ..tools import symplectic as _symp
from ..baseobjs import DummyProfiler as _DummyProfiler
from ..baseobjs import Label as _Label
from .mapevaltree import MapEvalTree as _MapEvalTree
from .forwardsim import ForwardSimulator
from . import replib


_dummy_profiler = _DummyProfiler()

# FUTURE: use enum (make sure it's supported in Python2.7?)
SUPEROP = 0
UNITARY = 1
CLIFFORD = 2


class MapForwardSimulator(ForwardSimulator):
    """
    Encapsulates a calculation tool used by model objects to perform product
    and derivatives-of-product calculations.

    This is contained in a class separate from Model to allow for additional
    model classes (e.g. ones which use entirely different -- non-gate-local
    -- parameterizations of operation matrices and SPAM vectors) access to these
    fundamental operations.
    """

    def __init__(self, dim, simplified_op_server, paramvec, max_cache_size=None):
        """
        Construct a new MapForwardSimulator object.

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
        """
        self.max_cache_size = max_cache_size
        super(MapForwardSimulator, self).__init__(
            dim, simplified_op_server, paramvec)
        if self.evotype not in ("statevec", "densitymx", "stabilizer"):
            raise ValueError(("Evolution type %s is incompatbile with "
                              "map-based calculations" % self.evotype))

    def copy(self):
        """ Return a shallow copy of this MatrixForwardSimulator """
        return MapForwardSimulator(self.dim, self.sos, self.paramvec)

    def _rhoEs_from_labels(self, rholabel, elabels):
        """ Returns SPAMVec *objects*, so must call .todense() later """
        rho = self.sos.get_prep(rholabel)
        Es = [self.sos.get_effect(elabel) for elabel in elabels]
        #No support for "custom" spamlabel stuff here
        return rho, Es

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
        if time is None:  # time-independent state propagation
            rhorep = self.sos.get_prep(rholabel).torep('prep')
            ereps = [self.sos.get_effect(elabel).torep('effect') for elabel in elabels]
            rhorep = replib.propagate_staterep(rhorep, [self.sos.get_operation(gl).torep() for gl in circuit])
            ps = _np.array([erep.probability(rhorep) for erep in ereps], 'd')
            #outcome probabilities
        else:
            t = time
            op = self.sos.get_prep(rholabel); op.set_time(t); t += rholabel.time
            state = op.torep('prep')
            for gl in circuit:
                op = self.sos.get_operation(gl); op.set_time(t); t += gl.time  # time in labels == duration
                state = op.torep().acton(state)
            ps = []
            for elabel in elabels:
                op = self.sos.get_effect(elabel); op.set_time(t)  # don't advance time (all effects occur at same time)
                ps.append(op.torep('effect').probability(state))
            ps = _np.array(ps, 'd')

        if _np.any(_np.isnan(ps)):
            if len(circuit) < 10:
                strToPrint = str(circuit)
            else:
                strToPrint = str(circuit[0:10]) + " ... (len %d)" % len(circuit)
            _warnings.warn("pr(%s) == nan" % strToPrint)

        if clipTo is not None:
            return _np.clip(ps, clipTo[0], clipTo[1])
        else: return ps

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

        #Finite difference derivative
        eps = 1e-7  # hardcoded?
        p = self.prs(spamTuple[0], [spamTuple[1]], circuit, clipTo)[0]
        dp = _np.empty((1, self.Np), 'd')

        orig_vec = self.to_vector().copy()
        for i in range(self.Np):
            vec = orig_vec.copy(); vec[i] += eps
            self.from_vector(vec)
            dp[0, i] = (self.prs(spamTuple[0], [spamTuple[1]], circuit, clipTo) - p)[0] / eps
        self.from_vector(orig_vec)

        if returnPr:
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

        #Finite difference hessian
        eps = 1e-4  # hardcoded?
        if returnPr:
            dp, p = self.dpr(spamTuple, circuit, returnPr, clipTo)
        else:
            dp = self.dpr(spamTuple, circuit, returnPr, clipTo)
        hp = _np.empty((1, self.Np, self.Np), 'd')

        orig_vec = self.to_vector().copy()
        for i in range(self.Np):
            vec = orig_vec.copy(); vec[i] += eps
            self.from_vector(vec)
            hp[0, i, :] = (self.dpr(spamTuple, circuit, False, clipTo) - dp) / eps
        self.from_vector(orig_vec)

        if returnPr and clipTo is not None:
            p = _np.clip(p, clipTo[0], clipTo[1])

        if returnDeriv:
            if returnPr: return hp, dp, p
            else: return hp, dp
        else:
            if returnPr: return hp, p
            else: return hp

    def _compute_pr_cache(self, rholabel, elabels, evalTree, comm, scratch=None):
        return replib.DM_compute_pr_cache(self, rholabel, elabels, evalTree, comm)

    def _compute_dpr_cache(self, rholabel, elabels, evalTree, wrtSlice, comm, scratch=None):
        return replib.DM_compute_dpr_cache(self, rholabel, elabels, evalTree, wrtSlice, comm, scratch)

    def _compute_hpr_cache(self, rholabel, elabels, evalTree, wrtSlice1, wrtSlice2, comm):
        #Compute finite difference hessians, one parameter at a time.

        param_indices1 = range(self.Np) if (wrtSlice1 is None) else _slct.indices(wrtSlice1)
        param_indices2 = range(self.Np) if (wrtSlice2 is None) else _slct.indices(wrtSlice2)
        nDerivCols1 = len(param_indices1)  # *all*, not just locally computed ones
        nDerivCols2 = len(param_indices2)  # *all*, not just locally computed ones

        dim = self.dim
        cacheSize = evalTree.cache_size()
        hpr_cache = _np.zeros((len(evalTree), len(elabels), nDerivCols1, nDerivCols2), 'd')

        # Allocate scratch space for compute_dpr_cache
        if self.evotype in ("statevec", "densitymx"):
            typ = complex if self.evotype == "statevec" else 'd'
            dpr_scratch = _np.zeros((cacheSize, dim), typ)
        else:  # evotype == "stabilizer" case
            dpr_scratch = [None] * cacheSize  # so we can store (s,p) tuples in cache

        eps = 1e-4  # hardcoded?
        dpCache = self._compute_dpr_cache(rholabel, elabels, evalTree, wrtSlice2, comm,
                                          dpr_scratch).copy()
        #need copy here b/c scratch space is used by sub-calls to
        # _compute_dpr_cache below in finite difference computation.

        all_slices, my_slice, owners, subComm = \
            _mpit.distribute_slice(slice(0, len(param_indices1)), comm)

        my_param_indices = param_indices1[my_slice]
        st = my_slice.start  # beginning of where my_param_indices results
        # get placed into dpr_cache

        #Get a map from global parameter indices to the desired
        # final index within dpr_cache
        iParamToFinal = {i: st + ii for ii, i in enumerate(my_param_indices)}

        orig_vec = self.to_vector().copy()
        for i in range(self.Np):
            if i in iParamToFinal:
                iFinal = iParamToFinal[i]
                vec = orig_vec.copy(); vec[i] += eps
                self.from_vector(vec)
                hpr_cache[:, :, iFinal, :] = (self._compute_dpr_cache(
                    rholabel, elabels, evalTree, wrtSlice2, subComm, dpr_scratch) - dpCache) / eps
        self.from_vector(orig_vec)

        #Now each processor has filled the relavant parts of dpr_cache,
        # so gather together:
        _mpit.gather_slices(all_slices, owners, hpr_cache, [], axes=2, comm=comm)

        return hpr_cache

    def default_distribute_method(self):
        """
        Return the preferred MPI distribution mode for this calculator.
        """
        return "circuits"

    def estimate_cache_size(self, nCircuits):
        """
        Return an estimate of the ideal/desired cache size given a number of
        operation sequences.

        Returns
        -------
        int
        """
        return int(0.7 * nCircuits)

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

        numSubtreeComms : int
            The number of processor groups that will be assigned to
            subtrees of the created tree.  This aids in the tree construction
            by giving the tree information it needs to distribute itself
            among the available processors.

        Returns
        -------
        MapEvalTree
        """
        evTree = _MapEvalTree()
        evTree.initialize(simplified_circuits, numSubtreeComms, self.max_cache_size)
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

        dim = self.dim
        nspam = int(round(_np.sqrt(self.dim)))  # an estimate - could compute?
        wrtLen1 = (self.Np + np1 - 1) // np1  # ceiling(num_params / np1)
        wrtLen2 = (self.Np + np2 - 1) // np2  # ceiling(num_params / np2)

        mem = 0
        for fnName in subcalls:
            if fnName == "bulk_fill_probs":
                mem += cache_size * dim  # pr cache intermediate
                mem += num_final_strs  # pr cache final (* #elabels!)

            elif fnName == "bulk_fill_dprobs":
                mem += cache_size * dim  # dpr cache scratch
                mem += cache_size * dim  # pr cache intermediate
                mem += num_final_strs * wrtLen1  # dpr cache final (* #elabels!)

            elif fnName == "bulk_fill_hprobs":
                mem += cache_size * dim  # dpr cache intermediate (scratch)
                mem += cache_size * wrtLen2 * dim * 2  # dpr cache (x2)
                mem += num_final_strs * wrtLen1 * wrtLen2  # hpr cache final (* #elabels!)

            elif fnName == "bulk_hprobs_by_block":
                #Note: includes "results" memory since this is allocated within
                # the generator and yielded, *not* allocated by the user.
                mem += 2 * cache_size * nspam * wrtLen1 * wrtLen2  # hprobs & dprobs12 results
                mem += cache_size * nspam * (wrtLen1 + wrtLen2)  # dprobs1 & dprobs2
                #TODO: check this -- maybe more mem we're forgetting

            else:
                raise ValueError("Unknown subcall name: %s" % fnName)

        return mem * FLOATSIZE

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

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """
                #Fill cache info
                prCache = self._compute_pr_cache(rholabel, elabels, evalSubTree, mySubComm)

                #use cached data to final values
                ps = evalSubTree.final_view(prCache, axis=0)  # ( nCircuits, len(elabels))
                for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                    _fas(mxToFill, [fInds], ps[gInds, i], add=sumInto)

            self._fill_result_tuple_collectrho((mxToFill,), evalSubTree,
                                               slice(None), slice(None), calc_and_fill)

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

            #Free memory from previous subtree iteration before computing caches
            paramSlice = slice(None)
            fillComm = mySubComm  # comm used by calc_and_fill

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """
                tm = _time.time()

                if prMxToFill is not None:
                    prCache = self._compute_pr_cache(rholabel, elabels, evalSubTree, fillComm)
                    ps = evalSubTree.final_view(prCache, axis=0)  # ( nCircuits, len(elabels))
                    for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                        _fas(prMxToFill, [fInds], ps[gInds, i], add=sumInto)

                #Fill cache info
                dprCache = self._compute_dpr_cache(rholabel, elabels, evalSubTree, paramSlice, fillComm)
                dps = evalSubTree.final_view(dprCache, axis=0)  # ( nCircuits, len(elabels), nDerivCols)
                for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                    _fas(mxToFill, [fInds, pslc1], dps[gInds, i], add=sumInto)
                profiler.add_time("bulk_fill_dprobs: calc_and_fill", tm)

            #Set wrtBlockSize to use available processors if it isn't specified
            if wrtFilter is None:
                blkSize = wrtBlockSize  # could be None
                if (mySubComm is not None) and (mySubComm.Get_size() > 1):
                    comm_blkSize = self.Np / mySubComm.Get_size()
                    blkSize = comm_blkSize if (blkSize is None) \
                        else min(comm_blkSize, blkSize)  # override with smaller comm_blkSize
            else:
                blkSize = None  # wrtFilter dictates block

            if blkSize is None:
                #Fill derivative cache info
                paramSlice = wrtSlice  # specifies which deriv cols calc_and_fill computes

                #Compute all requested derivative columns at once
                self._fill_result_tuple_collectrho((prMxToFill, mxToFill), evalSubTree,
                                                   slice(None), slice(None), calc_and_fill)
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
                fillComm = blkComm  # comm used by calc_and_fill

                for iBlk in myBlkIndices:
                    paramSlice = blocks[iBlk]  # specifies which deriv cols calc_and_fill computes
                    self._fill_result_tuple_collectrho(
                        (mxToFill,), evalSubTree,
                        blocks[iBlk], slice(None), calc_and_fill)
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

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(evalTree)
            fillComm = mySubComm

            #Free memory from previous subtree iteration before computing caches
            paramSlice1 = slice(None)
            paramSlice2 = slice(None)

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """

                if prMxToFill is not None:
                    prCache = self._compute_pr_cache(rholabel, elabels, evalSubTree, fillComm)
                    ps = evalSubTree.final_view(prCache, axis=0)  # ( nCircuits, len(elabels))
                    for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                        _fas(prMxToFill, [fInds], ps[gInds, i], add=sumInto)

                if deriv1MxToFill is not None:
                    dprCache = self._compute_dpr_cache(rholabel, elabels, evalSubTree, paramSlice1, fillComm)
                    dps1 = evalSubTree.final_view(dprCache, axis=0)  # ( nCircuits, len(elabels), nDerivCols)
                    for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                        _fas(deriv1MxToFill, [fInds, pslc1], dps1[gInds, i], add=sumInto)

                if deriv2MxToFill is not None:
                    if deriv1MxToFill is not None and paramSlice1 == paramSlice2:
                        dps2 = dps1
                    else:
                        dprCache = self._compute_dpr_cache(rholabel, elabels, evalSubTree, paramSlice2, fillComm)
                        dps2 = evalSubTree.final_view(dprCache, axis=0)  # ( nCircuits, len(elabels), nDerivCols)
                    for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                        _fas(deriv2MxToFill, [fInds, pslc2], dps2[gInds, i], add=sumInto)

                #Fill cache info
                hprCache = self._compute_hpr_cache(rholabel, elabels, evalSubTree, paramSlice1, paramSlice2, fillComm)
                hps = evalSubTree.final_view(hprCache, axis=0)  # ( nCircuits, len(elabels), nDerivCols1, nDerivCols2)

                for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                    _fas(mxToFill, [fInds, pslc1, pslc2], hps[gInds, i], add=sumInto)

            #Set wrtBlockSize to use available processors if it isn't specified
            if wrtFilter1 is None and wrtFilter2 is None:
                blkSize1 = wrtBlockSize1  # could be None
                blkSize2 = wrtBlockSize2  # could be None
                if (mySubComm is not None) and (mySubComm.Get_size() > 1):
                    comm_blkSize = self.Np / mySubComm.Get_size()
                    blkSize1 = comm_blkSize if (blkSize1 is None) \
                        else min(comm_blkSize, blkSize1)  # override with smaller comm_blkSize
                    blkSize2 = comm_blkSize if (blkSize2 is None) \
                        else min(comm_blkSize, blkSize2)  # override with smaller comm_blkSize
            else:
                blkSize1 = blkSize2 = None  # wrtFilter1 & wrtFilter2 dictates block

            if blkSize1 is None and blkSize2 is None:
                #Fill hessian cache info
                paramSlice1 = wrtSlice1  # specifies which deriv cols calc_and_fill computes
                paramSlice2 = wrtSlice2  # specifies which deriv cols calc_and_fill computes

                #Compute all requested derivative columns at once
                self._fill_result_tuple_collectrho(
                    (prMxToFill, deriv1MxToFill, deriv2MxToFill, mxToFill),
                    evalSubTree, slice(None), slice(None), calc_and_fill)

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
                fillComm = blk2Comm  # comm used by calc_and_fill

                for iBlk1 in myBlk1Indices:
                    paramSlice1 = blocks1[iBlk1]

                    for iBlk2 in myBlk2Indices:
                        paramSlice2 = blocks2[iBlk2]
                        self._fill_result_tuple_collectrho
                        ((prMxToFill, deriv1MxToFill, deriv2MxToFill, mxToFill),
                         evalSubTree, blocks1[iBlk1], blocks2[iBlk2], calc_and_fill)

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

    def bulk_hprobs_by_block(self, evalTree, wrtSlicesList,
                             bReturnDProbs12=False, comm=None):
        """
        Constructs a generator that computes the 2nd derivatives of the
        probabilities generated by a each gate sequence given by evalTree
        column-by-column.

        This routine can be useful when memory constraints make constructing
        the entire Hessian at once impractical, and one is able to compute
        reduce results from a single column of the Hessian at a time.  For
        example, the Hessian of a function of many gate sequence probabilities
        can often be computed column-by-column from the using the columns of
        the operation sequences.


        Parameters
        ----------
        spam_label_rows : dictionary
            a dictionary with keys == spam labels and values which
            are integer row indices into mxToFill, specifying the
            correspondence between rows of mxToFill and spam labels.

        evalTree : EvalTree
            given by a prior call to bulk_evaltree.  Specifies the operation sequences
            to compute the bulk operation on.  This tree *cannot* be split.

        wrtSlicesList : list
            A list of `(rowSlice,colSlice)` 2-tuples, each of which specify
            a "block" of the Hessian to compute.  Iterating over the output
            of this function iterates over these computed blocks, in the order
            given by `wrtSlicesList`.  `rowSlice` and `colSlice` must by Python
            `slice` objects.

        bReturnDProbs12 : boolean, optional
           If true, the generator computes a 2-tuple: (hessian_col, d12_col),
           where d12_col is a column of the matrix d12 defined by:
           d12[iSpamLabel,iOpStr,p1,p2] = dP/d(p1)*dP/d(p2) where P is is
           the probability generated by the sequence and spam label indexed
           by iOpStr and iSpamLabel.  d12 has the same dimensions as the
           Hessian, and turns out to be useful when computing the Hessian
           of functions of the probabilities.

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed as in
           bulk_product, bulk_dproduct, and bulk_hproduct.


        Returns
        -------
        block_generator
          A generator which, when iterated, yields the 3-tuple
          `(rowSlice, colSlice, hprobs)` or `(rowSlice, colSlice, dprobs12)`
          (the latter if `bReturnDProbs12 == True`).  `rowSlice` and `colSlice`
          are slices directly from `wrtSlicesList`. `hprobs` and `dprobs12` are
          arrays of shape K x S x B x B', where:

          - K is the length of spam_label_rows,
          - S is the number of operation sequences (i.e. evalTree.num_final_strings()),
          - B is the number of parameter rows (the length of rowSlice)
          - B' is the number of parameter columns (the length of colSlice)

          If `mx`, `dp1`, and `dp2` are the outputs of :func:`bulk_fill_hprobs`
          (i.e. args `mxToFill`, `deriv1MxToFill`, and `deriv1MxToFill`), then:

          - `hprobs == mx[:,:,rowSlice,colSlice]`
          - `dprobs12 == dp1[:,:,rowSlice,None] * dp2[:,:,None,colSlice]`
        """
        assert(not evalTree.is_split()), "`evalTree` cannot be split"
        nElements = evalTree.num_final_elements()

        #NOTE: don't distribute wrtSlicesList across comm procs,
        # as we assume the user has already done any such distribution
        # and has given each processor a list appropriate for it.
        # Use comm only for speeding up the calcs of the given
        # wrtSlicesList

        for wrtSlice1, wrtSlice2 in wrtSlicesList:

            if bReturnDProbs12:
                dprobs1 = _np.zeros((nElements, _slct.length(wrtSlice1)), 'd')
                dprobs2 = _np.zeros((nElements, _slct.length(wrtSlice2)), 'd')
            else:
                dprobs1 = dprobs2 = None
            hprobs = _np.zeros((nElements, _slct.length(wrtSlice1),
                                _slct.length(wrtSlice2)), 'd')

            self.bulk_fill_hprobs(hprobs, evalTree, None, dprobs1, dprobs2, comm=comm,
                                  wrtFilter1=_slct.indices(wrtSlice1),
                                  wrtFilter2=_slct.indices(wrtSlice2), gatherMemLimit=None)

            if bReturnDProbs12:
                dprobs12 = dprobs1[:, :, None] * dprobs2[:, None, :]  # (KM,N,1) * (KM,1,N') = (KM,N,N')
                yield wrtSlice1, wrtSlice2, hprobs, dprobs12
            else:
                yield wrtSlice1, wrtSlice2, hprobs

    # --------------------------------------------------- TIMEDEP FUNCTIONS -----------------------------------------

    def bulk_fill_timedep_chi2(self, mxToFill, evalTree, dsCircuitsToUse, num_total_outcomes, dataset,
                               minProbClipForWeighting, probClipInterval, comm=None):
        """
        Compute the chi2 contributions for an entire tree of circuits, computing
        and then summing together the contributions for each time the circuit is
        run, as given by the timestamps in `dataset`.

        Parameters
        ----------
        mxToFill : numpy ndarray
            an already-allocated 1D numpy array of length equal to the
            total number of computed elements (i.e. evalTree.num_final_elements())

        evalTree : EvalTree
            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
            strings to compute the bulk operation on.

        dsCircuitsToUse : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `evalTree` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(dsCircuitsToUse)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the chi2 contributions.

        minProbClipForWeighting : float, optional
            Sets the minimum and maximum probability p allowed in the chi^2
            weights: N/(p*(1-p)) by clipping probability p values to lie within
            the interval [ minProbClipForWeighting, 1-minProbClipForWeighting ].

        probClipInterval : 2-tuple or None, optional
           (min,max) values used to clip the predicted probabilities to.
           If None, no clipping is performed.

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
            dataset_rows = [dataset[dsCircuitsToUse[i]] for i in _slct.indices(evalSubTree.final_slice(evalTree))]
            num_outcomes = [num_total_outcomes[i] for i in _slct.indices(evalSubTree.final_slice(evalTree))]

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """
                #Fill cache info
                chi2Cache = replib.DM_compute_TDchi2_cache(self, rholabel, elabels, num_outcomes,
                                                           evalSubTree, dataset_rows, minProbClipForWeighting,
                                                           probClipInterval, mySubComm)

                #use cached data to final values
                ps = evalSubTree.final_view(chi2Cache, axis=0)  # ( nCircuits, len(elabels))
                for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                    _fas(mxToFill, [fInds], ps[gInds, i], add=sumInto)

            self._fill_result_tuple_collectrho((mxToFill,), evalSubTree,
                                               slice(None), slice(None), calc_and_fill)

        #collect/gather results
        subtreeElementIndices = [t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mxToFill, [], 0, comm)
        #note: pass mxToFill, dim=(KS,), so gather mxToFill[felInds] (axis=0)

    def bulk_fill_timedep_dchi2(self, mxToFill, evalTree, dsCircuitsToUse, num_total_outcomes, dataset,
                                minProbClipForWeighting, probClipInterval, chi2MxToFill=None,
                                comm=None, wrtFilter=None, wrtBlockSize=None,
                                profiler=None, gatherMemLimit=None):
        """
        Similar to :method:`bulk_fill_timedep_chi2` but compute the *jacobian*
        of the summed chi2 contributions for each circuit with respect to the
        model's parameters.

        Parameters
        ----------
        mxToFill : numpy ndarray
            an already-allocated ExM numpy array where E is the total number of
            computed elements (i.e. evalTree.num_final_elements()) and M is the
            number of model parameters.

        evalTree : EvalTree
            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
            strings to compute the bulk operation on.

        dsCircuitsToUse : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `evalTree` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(dsCircuitsToUse)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the chi2 contributions.

        minProbClipForWeighting : float, optional
            Sets the minimum and maximum probability p allowed in the chi^2
            weights: N/(p*(1-p)) by clipping probability p values to lie within
            the interval [ minProbClipForWeighting, 1-minProbClipForWeighting ].

        probClipInterval : 2-tuple or None, optional
           (min,max) values used to clip the predicted probabilities to.
           If None, no clipping is performed.

        chi2MxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with the per-circuit chi2 contributions, just like in
          bulk_fill_timedep_chi2(...).

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed over
           subtrees of evalTree (if it is split).

        Returns
        -------
        None
        """
        def dchi2(rholabel, elabels, num_tot_outcomes, evalSubTree, dataset_rows, wrtSlice, fillComm):
            return replib.DM_compute_TDdchi2_cache(self, rholabel, elabels, num_tot_outcomes, evalSubTree, dataset_rows,
                                                   minProbClipForWeighting, probClipInterval, wrtSlice, fillComm)

        def chi2(rholabel, elabels, num_tot_outcomes, evalSubTree, dataset_rows, fillComm):
            return replib.DM_compute_TDchi2_cache(self, rholabel, elabels, num_tot_outcomes, evalSubTree, dataset_rows,
                                                  minProbClipForWeighting, probClipInterval, fillComm)

        return self.bulk_fill_timedep_deriv(evalTree, dataset, dsCircuitsToUse, num_total_outcomes,
                                            mxToFill, dchi2, chi2MxToFill, chi2,
                                            comm, wrtFilter, wrtBlockSize, profiler, gatherMemLimit)

    def bulk_fill_timedep_loglpp(self, mxToFill, evalTree, dsCircuitsToUse, num_total_outcomes, dataset,
                                 minProbClip, radius, probClipInterval, comm=None):
        """
        Compute the log-likelihood contributions (within the "poisson picture")
        for an entire tree of circuits, computing and then summing together
        the contributions for each time the circuit is run, as given by the
        timestamps in `dataset`.

        Parameters
        ----------
        mxToFill : numpy ndarray
            an already-allocated 1D numpy array of length equal to the
            total number of computed elements (i.e. evalTree.num_final_elements())

        evalTree : EvalTree
            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
            strings to compute the bulk operation on.

        dsCircuitsToUse : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `evalTree` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(dsCircuitsToUse)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the logl contributions.

        minProbClip : float, optional
            The minimum probability treated normally in the evaluation of the
            log-likelihood.  A penalty function replaces the true log-likelihood
            for probabilities that lie below this threshold so that the
            log-likelihood never becomes undefined (which improves optimizer
            performance).

        radius : float, optional
            Specifies the severity of rounding used to "patch" the
            zero-frequency terms of the log-likelihood.

        probClipInterval : 2-tuple or None, optional
           (min,max) values used to clip the predicted probabilities to.
           If None, no clipping is performed.

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
            dataset_rows = [dataset[dsCircuitsToUse[i]] for i in _slct.indices(evalSubTree.final_slice(evalTree))]
            num_outcomes = [num_total_outcomes[i] for i in _slct.indices(evalSubTree.final_slice(evalTree))]

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """
                #Fill cache info
                loglCache = replib.DM_compute_TDloglpp_cache(self, rholabel, elabels, num_outcomes,
                                                             evalSubTree, dataset_rows, minProbClip,
                                                             radius, probClipInterval, mySubComm)

                #use cached data to final values
                logl = evalSubTree.final_view(loglCache, axis=0)  # ( nCircuits, len(elabels))
                for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                    _fas(mxToFill, [fInds], logl[gInds, i], add=sumInto)

            self._fill_result_tuple_collectrho((mxToFill,), evalSubTree,
                                               slice(None), slice(None), calc_and_fill)

        #collect/gather results
        subtreeElementIndices = [t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             mxToFill, [], 0, comm)
        #note: pass mxToFill, dim=(KS,), so gather mxToFill[felInds] (axis=0)

    def bulk_fill_timedep_dloglpp(self, mxToFill, evalTree, dsCircuitsToUse, num_total_outcomes, dataset,
                                  minProbClip, radius, probClipInterval, loglMxToFill=None,
                                  comm=None, wrtFilter=None, wrtBlockSize=None,
                                  profiler=None, gatherMemLimit=None):
        """
        Similar to :method:`bulk_fill_timedep_loglpp` but compute the *jacobian*
        of the summed logl (in posison picture) contributions for each circuit
        with respect to the model's parameters.

        Parameters
        ----------
        mxToFill : numpy ndarray
            an already-allocated ExM numpy array where E is the total number of
            computed elements (i.e. evalTree.num_final_elements()) and M is the
            number of model parameters.

        evalTree : EvalTree
            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
            strings to compute the bulk operation on.

        dsCircuitsToUse : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `evalTree` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(dsCircuitsToUse)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        dataset : DataSet
            the data set used to compute the logl contributions.

        minProbClipForWeighting : float, optional
            Sets the minimum and maximum probability p allowed in the chi^2
            weights: N/(p*(1-p)) by clipping probability p values to lie within
            the interval [ minProbClipForWeighting, 1-minProbClipForWeighting ].

        probClipInterval : 2-tuple or None, optional
           (min,max) values used to clip the predicted probabilities to.
           If None, no clipping is performed.

        loglMxToFill : numpy array, optional
          when not None, an already-allocated length-E numpy array that is filled
          with the per-circuit logl contributions, just like in
          bulk_fill_timedep_loglpp(...).

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed over
           subtrees of evalTree (if it is split).

        Returns
        -------
        None
        """
        def dloglpp(rholabel, elabels, num_tot_outcomes, evalSubTree, dataset_rows, wrtSlice, fillComm):
            return replib.DM_compute_TDdloglpp_cache(self, rholabel, elabels, num_tot_outcomes, evalSubTree,
                                                     dataset_rows, minProbClip, radius, probClipInterval,
                                                     wrtSlice, fillComm)

        def loglpp(rholabel, elabels, num_tot_outcomes, evalSubTree, dataset_rows, fillComm):
            return replib.DM_compute_TDloglpp_cache(self, rholabel, elabels, num_tot_outcomes, evalSubTree,
                                                    dataset_rows, minProbClip, radius, probClipInterval, fillComm)

        return self.bulk_fill_timedep_deriv(evalTree, dataset, dsCircuitsToUse, num_total_outcomes,
                                            mxToFill, dloglpp, loglMxToFill, loglpp,
                                            comm, wrtFilter, wrtBlockSize, profiler, gatherMemLimit)

    #A generic function - move to base class?
    def bulk_fill_timedep_deriv(self, evalTree, dataset, dsCircuitsToUse, num_total_outcomes,
                                derivMxToFill, deriv_fn, mxToFill=None, fn=None,
                                comm=None, wrtFilter=None, wrtBlockSize=None,
                                profiler=None, gatherMemLimit=None):
        """
        A generic method providing the scaffolding used when computing (filling)
        the derivative of a time-dependent quantity.  In particular, it
        distributes the computation among the subtrees of `evalTree` and
        relies on the caller to supply "compute_cache" and "compute_dcache"
        functions which just need to compute the quantitiy being filled and
        its derivative given a sub-tree and a parameter-slice.

        Parameters
        ----------
        evalTree : EvalTree
            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
            strings to compute the bulk operation on.

        dataset : DataSet
            the data set passed on to the computation functions.

        dsCircuitsToUse : list of Circuits
            the circuits to use as they should be queried from `dataset` (see
            below).  This is typically the same list of circuits used to
            construct `evalTree` potentially with some aliases applied.

        num_total_outcomes : list or array
            a list of the total number of *possible* outcomes for each circuit
            (so `len(num_total_outcomes) == len(dsCircuitsToUse)`).  This is
            needed for handling sparse data, where `dataset` may not contain
            counts for all the possible outcomes of each circuit.

        derivMxToFill : numpy ndarray
            an already-allocated ExM numpy array where E is the total number of
            computed elements (i.e. evalTree.num_final_elements()) and M is the
            number of model parameters.

        deriv_fn : function
            A function with the signature:
            `deriv_fn(rholabel, elabels, num_outcomes, evalSubTree,
                      dataset_rows, paramSlice, fillComm)` which computes the
            derivative of a quantity to be stored in `derivMxToFill`.  This
            jacobian is computed for all the circuits in `evalSubTree` with respect
            to the slice of model parameters given by `paramSlice`.  This function
            must return an array of shape (E',M') where E' is the total number of
            computed elements (i.e. evalSubTree.num_final_elements()) and M' is
            the number of model parameters in paramSlice.

        mxToFill : numpy array, optional
            when not None, an already-allocated length-E numpy array that is filled
            with the per-circuit contributions computed using `fn` below.

        fn : function
            A function with the signature:
            `fn(rholabel, elabels, num_outcomes, evalSubTree, dataset_rows,
                fillComm)` which computes the quantity to store in `mxToFill`
            (usually the quantity `deriv_fn` gives the derivative of).  This
            quantity is computed for all the circuits in `evalSubTree`, and `fn`
            must return a 1D array of length E' where E' is the total number of
            computed elements (i.e. evalSubTree.num_final_elements()).

        comm : mpi4py.MPI.Comm, optional
           When not None, an MPI communicator for distributing the computation
           across multiple processors.  Distribution is performed over
           subtrees of evalTree (if it is split).

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

        #tStart = _time.time()
        if profiler is None: profiler = _dummy_profiler

        if wrtFilter is not None:
            assert(wrtBlockSize is None)  # Cannot specify both wrtFilter and wrtBlockSize
            wrtSlice = _slct.list_to_slice(wrtFilter)  # for now, require the filter specify a slice
        else:
            wrtSlice = None

        #profiler.mem_check("bulk_fill_timedep_dchi2: begin")

        #get distribution across subtrees (groups if needed)
        subtrees = evalTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = evalTree.distribute(comm)

        #eval on each local subtree
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            felInds = evalSubTree.final_element_indices(evalTree)
            dataset_rows = [dataset[dsCircuitsToUse[i]] for i in _slct.indices(evalSubTree.final_slice(evalTree))]
            num_outcomes = [num_total_outcomes[i] for i in _slct.indices(evalSubTree.final_slice(evalTree))]

            #Free memory from previous subtree iteration before computing caches
            paramSlice = slice(None)
            fillComm = mySubComm  # comm used by calc_and_fill

            def calc_and_fill(rholabel, elabels, fIndsList, gIndsList, pslc1, pslc2, sumInto):
                """ Compute and fill result quantities for given arguments """
                #tm = _time.time()

                if mxToFill is not None:
                    cache = fn(rholabel, elabels, num_outcomes, evalSubTree, dataset_rows, fillComm)
                    cache = evalSubTree.final_view(cache, axis=0)  # ( nCircuits, len(elabels))
                    for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                        _fas(mxToFill, [fInds], cache[gInds, i], add=sumInto)

                #Fill cache info
                dcache = deriv_fn(rholabel, elabels, num_outcomes, evalSubTree,
                                  dataset_rows, paramSlice, fillComm)
                dcache = evalSubTree.final_view(dcache, axis=0)  # ( nCircuits, len(elabels), nDerivCols)
                for i, (fInds, gInds) in enumerate(zip(fIndsList, gIndsList)):
                    _fas(derivMxToFill, [fInds, pslc1], dcache[gInds, i], add=sumInto)
                #profiler.add_time("bulk_fill_timedep_dchi2: calc_and_fill", tm)

            #Set wrtBlockSize to use available processors if it isn't specified
            if wrtFilter is None:
                blkSize = wrtBlockSize  # could be None
                if (mySubComm is not None) and (mySubComm.Get_size() > 1):
                    comm_blkSize = self.Np / mySubComm.Get_size()
                    blkSize = comm_blkSize if (blkSize is None) \
                        else min(comm_blkSize, blkSize)  # override with smaller comm_blkSize
            else:
                blkSize = None  # wrtFilter dictates block

            if blkSize is None:
                #Fill derivative cache info
                paramSlice = wrtSlice  # specifies which deriv cols calc_and_fill computes

                #Compute all requested derivative columns at once
                self._fill_result_tuple_collectrho((mxToFill, derivMxToFill), evalSubTree,
                                                   slice(None), slice(None), calc_and_fill)
                #profiler.mem_check("bulk_fill_dprobs: post fill")

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
                fillComm = blkComm  # comm used by calc_and_fill

                for iBlk in myBlkIndices:
                    paramSlice = blocks[iBlk]  # specifies which deriv cols calc_and_fill computes
                    self._fill_result_tuple_collectrho(
                        (derivMxToFill,), evalSubTree,
                        blocks[iBlk], slice(None), calc_and_fill)
                    #profiler.mem_check("bulk_fill_dprobs: post fill blk")

                #gather results
                tm = _time.time()
                _mpit.gather_slices(blocks, blkOwners, derivMxToFill, [felInds],
                                    1, mySubComm, gatherMemLimit)
                #note: gathering axis 1 of derivMxToFill[:,fslc], dim=(ks,M)
                profiler.add_time("MPI IPC", tm)
                #profiler.mem_check("bulk_fill_dprobs: post gather blocks")

        #collect/gather results
        tm = _time.time()
        subtreeElementIndices = [t.final_element_indices(evalTree) for t in subtrees]
        _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                             derivMxToFill, [], 0, comm, gatherMemLimit)
        #note: pass derivMxToFill, dim=(KS,M), so gather derivMxToFill[felInds] (axis=0)

        if mxToFill is not None:
            _mpit.gather_indices(subtreeElementIndices, subTreeOwners,
                                 mxToFill, [], 0, comm)
            #note: pass mxToFill, dim=(KS,), so gather mxToFill[felInds] (axis=0)

        profiler.add_time("MPI IPC", tm)
        #profiler.mem_check("bulk_fill_timedep_dchi2: post gather subtrees")
        #
        #profiler.add_time("bulk_fill_timedep_dchi2: total", tStart)
        #profiler.add_count("bulk_fill_timedep_dchi2 count")
        #profiler.mem_check("bulk_fill_timedep_dchi2: end")
