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
import itertools as _itertools
import sys as _sys

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .. import optimize as _opt
from .. import tools as _tools
from ..tools import listtools as _lt
from ..tools import slicetools as _slct
from ..tools import mpitools as _mpit
from . import profiler as _profiler
from .circuitstructure import LsGermsStructure as _LsGermsStructure

_dummy_profiler = _profiler.DummyProfiler()
CHECK = False
CHECK_JACOBIAN = False
FLOATSIZE = 8  # TODO - get bytes-in-float a better way!


def objfn(objfn_cls, model, dataset, circuits=None,
          regularization=None, penalties=None, opLabelAliases=None,
          cache=None, comm=None, memLimit=None, **addl_args):
    """ TODO: docstring """

    if circuits is None:
        circuits = list(dataset.keys())

    if opLabelAliases:
        circuits = BulkCircuitList(circuits, opLabelAliases)

    resource_alloc = ResourceAllocation(comm, memLimit)
    ofn = objfn_cls(model, dataset, circuits, resource_alloc,
                    regularization, penalties, cache, verbosity=0, *addl_args)
    return ofn


class ComputationCache(object):
    def __init__(self,
                 evTree=None, lookup=None, outcomes_lookup=None, wrtBlkSize=None, wrtBlkSize2=None,
                 cntVecMx=None, totalCntVec=None):
        self.evTree = evTree
        self.lookup = lookup
        self.outcomes_lookup = outcomes_lookup
        self.wrtBlkSize = wrtBlkSize
        self.wrtBlkSize2 = wrtBlkSize2
        self.cntVecMx = cntVecMx
        self.totalCntVec = totalCntVec

    def has_evaltree(self):
        return (self.evTree is not None)

    def add_evaltree(self, model, dataset=None, circuitsToUse=None, resource_alloc=None, subcalls=(), verbosity=0):
        """TODO: docstring """
        comm = resource_alloc.comm if resource_alloc else None
        mlim = resource_alloc.memLimit if resource_alloc else None
        distributeMethod = resource_alloc.distributeMethod
        self.evTree, self.wrtBlkSize, self.wrtBlkSize2, self.lookup, self.outcomes_lookup = \
            model.bulk_evaltree_from_resources(circuitsToUse, comm, mlim, distributeMethod,
                                               subcalls, dataset, verbosity)

    def has_count_vectors(self):
        return (self.cntVecMx is not None) and (self.totalCntVec is not None)

    def add_count_vectors(self, dataset, circuitsToUse, dsCircuitsToUse, circuitWeights=None):
        assert(self.has_evaltree()), "Must `add_evaltree` before calling `add_count_vectors`!"
        KM = self.evTree.num_final_elements()
        cntVecMx = _np.empty(KM, 'd')
        N = _np.empty(KM, 'd')

        for (i, opStr) in enumerate(dsCircuitsToUse):
            cnts = dataset[opStr].counts
            N[self.lookup[i]] = sum(cnts.values())  # dataset[opStr].total
            cntVecMx[self.lookup[i]] = [cnts.get(x, 0) for x in self.outcomes_lookup[i]]

        if circuitWeights is not None:
            for i in range(len(circuitsToUse)):
                cntVecMx[self.lookup[i]] *= circuitWeights[i]  # dim KM (K = nSpamLabels, M = nCircuits )
                N[self.lookup[i]] *= circuitWeights[i]  # multiply N's by weights

        self.cntVecMx = cntVecMx
        self.totalCntVec = N


class BulkCircuitList(list):

    #TODO REMOVE
    #@classmethod
    #def from_circuit_list(cls, circuit_list, model, dataset=None, resource_alloc=None,
    #                      subcalls=(), verbosity=0): # cache=None
    #    if circuit_list is None and dataset is not None:
    #        circuit_list = list(dataset.keys())
    #    comm = resource_alloc.comm if resource_alloc else None
    #    mlim = resource_alloc.memLimit
    #    distributeMethod = resource_alloc.distributeMethod
    #    evalTree, wrtBlkSize, _, lookup, outcomes_lookup = model.bulk_evaltree_from_resources(
    #        circuit_list, comm, mlim, distributeMethod, subcalls, dataset, verbosity)
    #
    #    # Note: evalTree.generate_circuit_list() != circuit_list, as the tree holds *simplified* circuits
    #    return cls(circuit_list, evalTree, lookup, outcomes_lookup)

    def __init__(self, circuit_list_or_structure, opLabelAliases=None, circuitWeights=None, name=None):

        #validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
        if isinstance(circuit_list_or_structure, (list, tuple)):
            self.circuitsToUse = circuit_list_or_structure
            self.circuitStructure = _LsGermsStructure([], [], [], [], None)  # create a dummy circuit structure
            self.circuitStructure.add_unindexed(circuit_list_or_structure)   # which => "no circuit structure"
        else:
            self.circuitStructure = circuit_list_or_structure
            self.circuitsToUse = self.circuitStructure.allstrs

        self.opLabelAliases = opLabelAliases
        self.circuitWeights = circuitWeights
        self.name = name  # an optional name for this circuit list
        self[:] = self.circuitsToUse  # maybe get rid of self.circuitsToUse in the future...

    #def __len__(self):
    #    return len(self.circuitsToUse)


class ResourceAllocation(object):
    @classmethod
    def build_resource_allocation(cls, arg):
        if arg is None:
            return cls()
        elif isinstance(arg, ResourceAllocation):
            return arg
        else:  # assume argument is a dict of args
            return cls(arg.get('comm', None), arg.get('memLimit', None),
                       arg.get('profiler', None), arg.get('distributeMethod', 'default'))

    def __init__(self, comm=None, memLimit=None, profiler=None, distributeMethod="default"):
        self.comm = comm
        self.memLimit = memLimit
        if profiler is not None:
            self.profiler = profiler
        else:
            self.profiler = _dummy_profiler
        self.distributeMethod = distributeMethod

    def copy(self):
        return ResourceAllocation(self.comm, self.memLimit, self.profiler, self.distributeMethod)


class ObjectiveFunctionBuilder(object):
    @classmethod
    def simple(cls, objective='logl', freqWeightedChi2=False):
        if objective == "chi2":
            if freqWeightedChi2:
                builder = FreqWeightedChi2Function.builder(
                    name='fqchi2',
                    regularization={'minProbClipForWeighting': 1e-4,
                                    'probClipInterval': (-1e6, 1e6),
                                    'radius': 1e-4})
            else:
                builder = Chi2Function.builder(
                    name='chi2',
                    regularization={'minProbClipForWeighting': 1e-4,
                                    'probClipInterval': (-1e6, 1e6)})

        elif objective == "logl":
            builder = DeltaLogLFunctionPoissonPic.builder(
                name='logl',
                regularization={'minProbClip': 1e-4,
                                'probClipInterval': (-1e6, 1e6),
                                'radius': 1e-4},
                penalties={'cptp_penalty_factor': 0,
                           'spam_penalty_factor': 0})
        else:
            raise ValueError("Invalid objective: %s" % objective)
        assert(isinstance(builder, cls)), "This function should always return an ObjectiveFunctionBuilder!"
        return builder

    def __init__(self, cls_to_build, name=None, desc=None, regularization=None, penalties=None, **kwargs):
        self.name = name if (name is not None) else cls_to_build.__name__
        self.description = desc if (desc is not None) else "objfn"  # "Sum of Chi^2"  OR "2*Delta(log(L))"
        self.cls_to_build = cls_to_build
        self.regularization = regularization
        self.penalties = penalties
        self.additional_args = kwargs

    def build(self, mdl, dataset, circuit_list, resource_alloc=None, cache=None, verbosity=0):
        return self.cls_to_build(mdl=mdl, dataset=dataset, circuit_list=circuit_list,
                                 resource_alloc=resource_alloc, cache=cache, verbosity=verbosity,
                                 regularization=self.regularization, penalties=self.penalties,
                                 name=self.name, **self.additional_args)


class ObjectiveFunction(object):
    @classmethod
    def builder(cls, name=None, desc=None, regularization=None, penalties=None, **kwargs):
        return ObjectiveFunctionBuilder(cls, name, desc, regularization, penalties, **kwargs)

    def __init__(self, mdl, dataset, circuit_list, regularization=None, penalties=None,
                 cache=None, resource_alloc=None, name=None, description=None, verbosity=0):
        """
        TODO: docstring - note: 'cache' is for repeated calls with same mdl, circuit_list,
        and dataset (but different derived objective fn class).  Note: circuit_list can be
        either a normal list of Circuits or a BulkCircuitList object (or None)
        """
        resource_alloc = ResourceAllocation.build_resource_allocation(resource_alloc)
        self.comm = resource_alloc.comm
        self.profiler = resource_alloc.profiler
        self.memLimit = resource_alloc.memLimit
        self.gthrMem = None  # set below

        self.printer = _VerbosityPrinter.build_printer(verbosity, self.comm)
        self.name = name if (name is not None) else self.__class__.__name__
        self.description = description if (description is not None) else "objfn"
        self.mdl = mdl
        self.vec_gs_len = mdl.num_params()
        self.opBasis = mdl.basis
        self.dataset = dataset

        circuit_list = circuit_list if (circuit_list is not None) else list(dataset.keys())
        bulk_circuit_list = circuit_list if isinstance(circuit_list, BulkCircuitList) else BulkCircuitList(circuit_list)
        self.circuitsToUse = bulk_circuit_list[:]
        self.circuitWeights = bulk_circuit_list.circuitWeights
        self.dsCircuitsToUse = _tools.apply_aliases_to_circuit_list(self.circuitsToUse,
                                                                    bulk_circuit_list.opLabelAliases)

        persistentMem = self.get_persistent_memory_estimate()
        subcalls = self.get_evaltree_subcalls()

        if self.memLimit:
            if self.memLimit < persistentMem:
                in_GB = 1.0 / 1024.0**3
                raise MemoryError("Memory limit ({} GB) is < memory required to hold final results "
                                  "({} GB)".format(self.memLimit * in_GB, persistentMem * in_GB))

            curMem = _profiler._get_max_mem_usage(self.comm)  # is this what we want??
            self.gthrMem = int(0.1 * (self.memLimit - persistentMem))
            evt_mlim = self.memLimit - persistentMem - self.gthrMem - curMem
            self.printer.log("Memory limit = %.2fGB" % (self.memLimit * in_GB))
            self.printer.log("Cur, Persist, Gather = %.2f, %.2f, %.2f GB" %
                             (curMem * in_GB, persistentMem * in_GB, self.gthrMem * in_GB))
            assert(evt_mlim > 0), 'Not enough memory, exiting..'
        else:
            evt_mlim = None

        self.cache = cache if (cache is not None) else ComputationCache()
        if not self.cache.has_evaltree():
            evt_resource_alloc = resource_alloc.copy(); evt_resource_alloc.memLimit = evt_mlim
            self.cache.add_evaltree(self.mdl, self.dataset, self.circuitsToUse, evt_resource_alloc,
                                    subcalls, self.printer - 1)

        self.evTree = self.cache.evTree
        self.lookup = self.cache.lookup
        self.outcomes_lookup = self.cache.outcomes_lookup
        self.wrtBlkSize = self.cache.wrtBlkSize
        self.wrtBlkSize2 = self.cache.wrtBlkSize2

        self.time_dependent = False
        self.check = CHECK
        self.check_jacobian = CHECK_JACOBIAN

        self.KM = self.evTree.num_final_elements()  # shorthand for combined spam+circuit dimension
        self.firsts = None  # no omitted probs by default

        if penalties is None: penalties = {}
        self.ex = self.set_penalties(**penalties)  # "extra" (i.e. beyond the (circuit,spamlabel)) rows of jacobian

        if regularization is None: regularization = {}
        self.set_regularization(**regularization)

        self.ls_fn = None
        self.ls_jfn = None
        self.term_fn = self._default_term_fn
        self.percircuit_fn = self._default_percircuit_fn
        self.fn = self._default_fn
        self.jfn = self._default_jfn  # jacobian of .fn (the entire, summed-up objective fn)
        self.hfn = None  # no default hessian
        self.approx_hfn = self._default_approx_hfn

    def _default_evalpt(self):
        """The default point to evaluate functions at """
        return self.mdl.to_vector()  # just use current model's point

    def _default_term_fn(self, pv=None):
        assert(self.ls_fn is not None), "Default per-term function requires leastsq objective to be implemented!"
        if pv is None: pv = self._default_evalpt()
        v = self.ls_fn(pv)  # least-squares objective fn: v is a vector s.t. obj_fn = ||v||^2 (L2 norm)
        obj_per_term = v**2  # per term or "element" (one per circuit outcome)
        return obj_per_term

    def _default_percircuit_fn(self, pv=None):
        """
        TODO: docstring - pv = "parameter vector"
        """
        assert(self.term_fn is not None), "Default objective function requires per-term objective to be implemented!"
        if pv is None: pv = self._default_evalpt()
        obj_per_term = self.term_fn(pv)

        #Aggregate over outcomes:
        # obj_per_el[iElement] contains contributions per element - now aggregate over outcomes
        # percircuit[iCircuit] will contain contributions for each original circuit (aggregated over outcomes)
        nCircuits = len(self.circuitsToUse)
        percircuit = _np.empty(nCircuits, 'd')
        for i in range(nCircuits):
            percircuit[i] = _np.sum(obj_per_term[self.lookup[i]], axis=0)
        return percircuit

    def _default_fn(self, pv=None):
        return _np.sum(self.percircuit_fn(pv))  # just sum contributions of all circuits => total obj_fn

    def _default_jfn(self, pv=None):
        assert(self.ls_fn is not None and self.ls_jfn is not None), \
            "Default jacobian function requires leastsq objective & jacobian to be implemented!"
        if pv is None: pv = self._default_evalpt()
        v = self.ls_fn(pv)  # least-squares objective fn: v is a vector s.t. obj_fn = ||v||^2 (L2 norm)
        J = self.ls_jfn(pv)  # jacobian of dim N x M where N = len(v) and M = len(pv)
        assert(J.shape == (len(v), len(pv))), "ls_jfn returned a Jacobian with the wrong shape!"

        #Objective is sum(v**2), so Jacobian = 2*sum(v * dv/dp)
        jacobian = 2.0 * _np.sum(v[:, None] * J, axis=0)  # (N,1) * (N,M) => (N,M) =sum=> (M,)
        return jacobian

    def _default_approx_hfn(self, pv=None):
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval, self.check, self.comm)  # fill .probs
        d2v_dp2 = self._d2v_dp2()  # 2nd derivative of objective fn elements wrt probs
        assert(self.jac is not None)  # this function also assumes we have allocated self.jac
        self.mld.bulk_fill_dprobs(self.jac, self.evTree, None, self.probClipInterval,
                                  self.check, self.comm)  # use cache?
        JT = self.jac[:, None, :]  # (KM,1,N) - will "dot" using 2nd parameter col
        J = self.jac[:, :, None]  # (KM,N,1) - will "dot" using 1st parameter col
        approx_hessian = d2v_dp2[:, None, None] * JT * J
        return approx_hessian

    def _d2v_dp2(self):
        """
        Derivative of the objective function elements (one per probability)
        wrt the probabilities (*not* the parameters)
        """
        raise NotImplementedError(("Derived classes should implement this to make use "
                                   "of the default approximate-hessian implementation!"))

    def set_penalties(self):
        return 0  # no additional penalty terms

    def set_regularization(self):
        pass  # no regularization parameters

    def get_chi2k_distributed_qty(self, objective_function_value):
        return objective_function_value  # default is to assume the value *is* chi2_k distributed

    def get_persistent_memory_estimate(self, num_elements=None):
        #  Estimate & check persistent memory (from allocs within objective function)
        if num_elements is None:
            nout = int(round(_np.sqrt(self.mdl.dim)))  # estimate of avg number of outcomes per string
            nc = len(self.circuitsToUse)
            ne = nc * nout  # estimate of the number of elements (e.g. probabilities, # LS terms, etc) to compute
        else:
            ne = num_elements
        np = self.mdl.num_params()

        # "persistent" memory is that used to store the final results.
        obj_fn_mem = FLOATSIZE * ne
        jac_mem = FLOATSIZE * ne * np
        hess_mem = FLOATSIZE * ne * np**2
        persistentMem = 4 * obj_fn_mem + jac_mem  # 4 different objective-function sized arrays, 1 jacobian array?
        if any([nm == "bulk_fill_hprobs" for nm in self.get_evaltree_subcalls()]):
            persistentMem += hess_mem  # we need room for the hessian too!
        # TODO: what about "bulk_hprobs_by_block"?

        return persistentMem

    def get_evaltree_subcalls(self):
        return ["bulk_fill_probs", "bulk_fill_dprobs"]

    def get_num_data_params(self):
        return self.dataset.get_degrees_of_freedom(
            self.dsCircuitsToUse, aggregate_times=not self.time_dependent)

    def precompute_omitted_freqs(self):
        #Detect omitted frequences (assumed to be 0) so we can compute objective fn correctly
        self.firsts = []; self.indicesOfCircuitsWithOmittedData = []
        for i, c in enumerate(self.circuitsToUse):
            lklen = _slct.length(self.lookup[i])
            if 0 < lklen < self.mdl.get_num_outcomes(c):
                self.firsts.append(_slct.as_array(self.lookup[i])[0])
                self.indicesOfCircuitsWithOmittedData.append(i)
        if len(self.firsts) > 0:
            self.firsts = _np.array(self.firsts, 'i')
            self.indicesOfCircuitsWithOmittedData = _np.array(self.indicesOfCircuitsWithOmittedData, 'i')
            self.dprobs_omitted_rowsum = _np.empty((len(self.firsts), self.vec_gs_len), 'd')
            self.printer.log("SPARSE DATA: %d of %d rows have sparse data" %
                             (len(self.firsts), len(self.circuitsToUse)))
        else:
            self.firsts = None  # no omitted probs

    def compute_count_vectors(self):
        if not self.cache.has_count_vectors():
            self.cache.add_count_vectors(self.dataset, self.circuitsToUse, self.dsCircuitsToUse, self.circuitWeights)
        return self.cache.cntVecMx, self.cache.totalCntVec

    def _construct_hessian(self, cntVecMx_all, totalCntVec_all, probClipInterval):
        """
        Framework for constructing a hessian matrix row by row using a derived
        class's `_hessian_from_hprobs` method.  This function expects that this
        objective function has been setup for hessian computation, and it's evaltree
        may be split in order to facilitate this.
        """
        #Note - we could in the future use comm to distribute over
        # subtrees here.  We currently don't because we parallelize
        # over columns and it seems that in almost all cases of
        # interest there will be more hessian columns than processors,
        # so adding the additional ability to parallelize over
        # subtrees would just add unnecessary complication.

        #get distribution across subtrees (groups if needed)
        subtrees = self.evTree.get_sub_trees()
        mySubTreeIndices, subTreeOwners, mySubComm = self.evTree.distribute(self.comm)

        nP = self.mdl.num_params()
        blkSize1, blkSize2 = self.wrtBlkSize, self.wrtBlkSize2
        rowParts = int(round(nP / blkSize1)) if (blkSize1 is not None) else 1
        colParts = int(round(nP / blkSize2)) if (blkSize2 is not None) else 1

        #  Allocate memory (alloc max required & take views)
        max_nEls = max([subtrees[i].num_final_elements() for i in mySubTreeIndices])
        probs_mem = _np.empty(max_nEls, 'd')

        #  Allocate persistent memory
        final_hessian = _np.zeros((nP, nP), 'd')

        tStart = _time.time()

        #Loop over subtrees
        for iSubTree in mySubTreeIndices:
            evalSubTree = subtrees[iSubTree]
            sub_nEls = evalSubTree.num_final_elements()

            if evalSubTree.myFinalElsToParentFinalElsMap is not None:
                #Then `evalSubTree` is a nontrivial sub-tree and its .spamtuple_indices
                # will index the *parent's* final index array space, which we
                # usually want but NOT here, where we fill arrays just big
                # enough for each subtree separately - so re-init spamtuple_indices
                evalSubTree = evalSubTree.copy()
                evalSubTree.recompute_spamtuple_indices(bLocal=True)

            # Create views into pre-allocated memory
            probs = probs_mem[0:sub_nEls]

            # Take portions of count arrays for this subtree
            cntVecMx = cntVecMx_all[evalSubTree.final_element_indices(self.evTree)]
            totalCntVec = totalCntVec_all[evalSubTree.final_element_indices(self.evTree)]
            assert(len(cntVecMx) == len(probs))

            #compute pos_probs separately
            self.mdl.bulk_fill_probs(probs, evalSubTree,
                                     clipTo=probClipInterval, check=self.check,
                                     comm=mySubComm)

            nCols = self.mdl.num_params()
            blocks1 = _mpit.slice_up_range(nCols, rowParts)
            blocks2 = _mpit.slice_up_range(nCols, colParts)
            sliceTupList_all = list(_itertools.product(blocks1, blocks2))
            #cull out lower triangle blocks, which have no overlap with
            # the upper triangle of the hessian
            sliceTupList = [(slc1, slc2) for slc1, slc2 in sliceTupList_all
                            if slc1.start <= slc2.stop]

            loc_iBlks, blkOwners, blkComm = \
                _mpit.distribute_indices(list(range(len(sliceTupList))), mySubComm)
            mySliceTupList = [sliceTupList[i] for i in loc_iBlks]

            subtree_hessian = _np.zeros((nP, nP), 'd')

            k, kmax = 0, len(mySliceTupList)
            for (slice1, slice2, hprobs, dprobs12) in self.mdl.bulk_hprobs_by_block(
                    evalSubTree, mySliceTupList, True, blkComm):
                rank = self.comm.Get_rank() if (self.comm is not None) else 0

                if self.printer.verbosity > 3 or (self.printer.verbosity == 3 and rank == 0):
                    iSub = mySubTreeIndices.index(iSubTree)
                    print("rank %d: %gs: block %d/%d, sub-tree %d/%d, sub-tree-len = %d"
                          % (rank, _time.time() - tStart, k, kmax, iSub,
                             len(mySubTreeIndices), len(evalSubTree)))
                    _sys.stdout.flush(); k += 1

                subtree_hessian[slice1, slice2] = \
                    self._hessian_from_hprobs(hprobs, dprobs12, cntVecMx,
                                              totalCntVec, probs)
                #NOTE: _hessian_from_hprobs MAY modify hprobs and dprobs12

            #Gather columns from different procs and add to running final hessian
            #_mpit.gather_slices_by_owner(slicesIOwn, subtree_hessian,[], (0,1), mySubComm)
            _mpit.gather_slices(sliceTupList, blkOwners, subtree_hessian, [], (0, 1), mySubComm)
            final_hessian += subtree_hessian

        #gather (add together) final_hessians from different processors
        if self.comm is not None and len(set(subTreeOwners.values())) > 1:
            if self.comm.Get_rank() not in subTreeOwners.values():
                # this proc is not the "owner" of its subtrees and should not send a contribution to the sum
                final_hessian[:, :] = 0.0  # zero out hessian so it won't contribute
            final_hessian = self.comm.allreduce(final_hessian)

        #copy upper triangle to lower triangle (we only compute upper)
        for i in range(final_hessian.shape[0]):
            for j in range(i + 1, final_hessian.shape[1]):
                final_hessian[j, i] = final_hessian[i, j]

        return final_hessian  # (N,N)

    def _hessian_from_hprobs(self, hprobs, dprobs12, cntVecMx, totalCntVec, probs):
        raise NotImplementedError("Derived classes should implement this!")


#NOTE on chi^2 expressions:
#in general case:   chi^2 = sum (p_i-f_i)^2/p_i  (for i summed over outcomes)
#in 2-outcome case: chi^2 = (p+ - f+)^2/p+ + (p- - f-)^2/p-
#                         = (p - f)^2/p + (1-p - (1-f))^2/(1-p)
#                         = (p - f)^2 * (1/p + 1/(1-p))
#                         = (p - f)^2 * ( ((1-p) + p)/(p*(1-p)) )
#                         = 1/(p*(1-p)) * (p - f)^2

class Chi2Function(ObjectiveFunction):

    def __init__(self, mdl, dataset, circuit_list, regularization=None, penalties=None,
                 cache=None, resource_alloc=None, name=None, verbosity=0, enable_hessian=False):

        self.enable_hessian = enable_hessian
        super().__init__(mdl, dataset, circuit_list, regularization, penalties, cache, resource_alloc, name, verbosity)

        #  Allocate peristent memory
        #  (must be AFTER possible operation sequence permutation by
        #   tree and initialization of dsCircuitsToUse)
        self.probs = _np.empty(self.KM, 'd')
        self.jac = _np.empty((self.KM + self.ex, self.vec_gs_len), 'd')
        self.hprobs = _np.empty((self.KM, self.vec_gs_len, self.vec_gs_len), 'd') if self.enable_hessian else None
        self.cntVecMx, self.N = self.compute_count_vectors()
        self.f = self.cntVecMx / self.N
        self.maxCircuitLength = max([len(x) for x in self.circuitsToUse])
        self.precompute_omitted_freqs()  # sets self.firsts

        if self.printer.verbosity < 4:  # Fast versions of functions
            if self.regularizeFactor == 0 and self.cptp_penalty_factor == 0 and self.spam_penalty_factor == 0 \
               and mdl.get_simtype() != "termgap":
                # Fast un-regularized version
                self.ls_fn = self.simple_chi2
                self.ls_jfn = self.simple_jac

            elif self.regularizeFactor != 0:
                # Fast regularized version
                assert(self.cptp_penalty_factor == 0), "Cannot have regularizeFactor and cptp_penalty_factor != 0"
                assert(self.spam_penalty_factor == 0), "Cannot have regularizeFactor and spam_penalty_factor != 0"
                self.ls_fn = self.regularized_chi2
                self.ls_jfn = self.regularized_jac

            elif mdl.get_simtype() == "termgap":
                assert(self.cptp_penalty_factor == 0), "Cannot have termgap_pentalty_factor and cptp_penalty_factor!=0"
                assert(self.spam_penalty_factor == 0), "Cannot have termgap_pentalty_factor and spam_penalty_factor!=0"
                self.ls_fn = self.termgap_chi2
                self.ls_jfn = self.simple_jac

            else:  # cptp_pentalty_factor != 0 and/or spam_pentalty_factor != 0
                assert(self.regularizeFactor == 0), "Cannot have regularizeFactor and other penalty factors > 0"
                self.ls_fn = self.penalized_chi2
                self.ls_jfn = self.penalized_jac

        else:  # Verbose (DEBUG) version of objective_func
            if mdl.get_simtype() == "termgap":
                raise NotImplementedError("Still need to add termgap support to verbose chi2!")
            self.ls_fn = self.verbose_chi2
            self.ls_jfn = self.verbose_jac

        self.jfn = self._jacobian_fn
        self.hfn = self._hessian_fn
        self.approx_hfn = self._approx_hessian_fn

    def get_evaltree_subcalls(self):
        calls = ["bulk_fill_probs", "bulk_fill_dprobs"]
        if self.enable_hessian: calls.append("bulk_fill_hprobs")
        return calls

    def set_penalties(self, regularizeFactor=0, cptp_penalty_factor=0, spam_penalty_factor=0):
        self.regularizeFactor = regularizeFactor
        self.cptp_penalty_factor = cptp_penalty_factor
        self.spam_penalty_factor = spam_penalty_factor

        # Compute "extra" (i.e. beyond the (circuit,spamlabel)) rows of jacobian
        ex = 0
        if regularizeFactor != 0:
            ex = self.vec_gs_len
        else:
            if cptp_penalty_factor != 0: ex += _cptp_penalty_size(self.mdl)
            if spam_penalty_factor != 0: ex += _spam_penalty_size(self.mdl)
        return ex

    def set_regularization(self, minProbClipForWeighting=1e-4, probClipInterval=(-10000, 10000)):
        self.minProbClipForWeighting = minProbClipForWeighting
        self.probClipInterval = probClipInterval

    def get_weights(self, p):
        cp = _np.clip(p, self.minProbClipForWeighting, None)
        return _np.sqrt(self.N / cp)  # nSpamLabels x nCircuits array (K x M)

    def get_dweights(self, p, wts):  # derivative of weights w.r.t. p
        cp = _np.clip(p, self.minProbClipForWeighting, None)
        dw = -0.5 * wts / cp   # nSpamLabels x nCircuits array (K x M)
        dw[_np.logical_or(p < self.minProbClipForWeighting, p > (1 - self.minProbClipForWeighting))] = 0.0
        return dw

    def zero_freq_chi2(self, N, probs):
        clipped_probs = _np.clip(probs, self.minProbClipForWeighting, None)
        return N * probs**2 / clipped_probs

    def zero_freq_dchi2(self, N, probs):
        clipped_probs = _np.clip(probs, self.minProbClipForWeighting, None)
        return _np.where(probs == clipped_probs, N, 2 * N * probs / clipped_probs)

    def update_v_for_omitted_probs(self, v, probs):
        # if i-th circuit has omitted probs, have sqrt( N*(p_i-f_i)^2/p_i + sum_k(N*p_k) )
        # so we need to take sqrt( v_i^2 + N*sum_k(p_k) )
        omitted_probs = 1.0 - _np.array([_np.sum(probs[self.lookup[i]])
                                         for i in self.indicesOfCircuitsWithOmittedData])
        v[self.firsts] = _np.sqrt(v[self.firsts]**2 + self.zero_freq_chi2(self.N[self.firsts], omitted_probs))

    def update_dprobs_for_omitted_probs(self, dprobs, probs, weights, dprobs_omitted_rowsum):
        # with omitted terms, new_obj = sqrt( obj^2 + corr ) where corr = N*omitted_p^2/clipped_omitted_p
        # so then d(new_obj) = 1/(2*new_obj) *( 2*obj*dobj + dcorr )*domitted_p where dcorr = N when not clipped
        #    and 2*N*omitted_p/clip_bound * domitted_p when clipped
        v = (probs - self.f) * weights
        omitted_probs = 1.0 - _np.array([_np.sum(probs[self.lookup[i]])
                                         for i in self.indicesOfCircuitsWithOmittedData])
        dprobs_factor_omitted = self.zero_freq_dchi2(self.N[self.firsts], omitted_probs)
        fullv = _np.sqrt(v[self.firsts]**2 + self.zero_freq_chi2(self.N[self.firsts], omitted_probs))

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

    # Scratch work:
    # chi^2 = sum_i N_i*(p_i-f_i)^2 / p_i  (i over circuits & spam labels)                                                                                      # noqa
    # d(chi^2)/dx = sum_i N_i * [ 2(p_i-f_i)*dp_i/dx / p_i - (p_i-f_i)^2 / p_i^2 * dp_i/dx ]                                                                    # noqa
    #             = sum_i N_i * (p_i-f_i) / p_i * [2 - (p_i-f_i)/p_i   ] * dp_i/dx                                                                              # noqa
    #             = sum_i N_i * t_i * [2 - t_i ] * dp_i/dx     where t_i = (p_i-f_i) / p_i                                                                      # noqa
    # d2(chi^2)/dydx = sum_i N_i * [ dt_i/dy * [2 - t_i ] * dp_i/dx - t_i * dt_i/dy * dp_i/dx + t_i * [2 - t_i] * d2p_i/dydx ]                                  # noqa
    #                          where dt_i/dy = [ 1/p_i - (p_i-f_i) / p_i^2 ] * dp_i/dy                                                                          # noqa
    def _jacobian_fn(self, pv=None):
        dprobs = self.jac
        self.mdl.bulk_fill_dprobs(dprobs, self.evTree,
                                  self.probs, self.probClipInterval, self.check, self.comm)  # use cache?

        if self.firsts is not None:
            for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.lookup[i], :], axis=0)

        cprobs = _np.clip(self.probs, self.minProbClipForWeighting, None)

        #account for omitted probs (sparse data)
        if self.firsts is not None:
            omitted_probs = 1.0 - _np.array([_np.sum(self.probs[self.lookup[i]])
                                             for i in self.indicesOfCircuitsWithOmittedData])
            clipped_oprobs = _np.clip(omitted_probs, self.minProbClipForWeighting, None)

        t = ((self.probs - self.f) / cprobs)[:, None]  # (iElement, 0) = (KM,1)
        dchi2 = self.N[:, None] * t * (2 - t) * dprobs  # (KM,1) * (KM,1) * (KM,N)  (K=#spam, M=#strings, N=#vec_gs)

        #account for omitted probs
        if self.firsts is not None:
            t_firsts = (omitted_probs / clipped_oprobs)[:, None]
            dchi2[self.firsts, :] -= self.N[self.firsts, None] * t_firsts * (2 - t_firsts) * self.dprobs_omitted_rowsum

        dchi2 = _np.sum(dchi2, axis=0)  # sum over operation sequences and spam labels => (N)
        return dchi2

    def _hessian_fn(self, pv=None):
        if self.firsts is not None:
            raise NotImplementedError("Chi2 hessian not implemented for sparse data (yet)")

        dprobs = self.jac
        self.mdl.bulk_fill_hprobs(self.hprobs, self.evTree, self.probs, dprobs,
                                  self.probClipInterval, self.check, self.comm)  # use cache?

        cprobs = _np.clip(self.probs, self.minProbClipForWeighting, None)
        dprobs_p = dprobs[:, None, :]  # (KM,1,N)
        t = ((self.probs - self.f) / cprobs)[:, None, None]  # (iElement, 0,0) = (KM,1,1)
        dt = ((1.0 / cprobs - (self.probs - self.f) / cprobs**2)[:, None]
              * dprobs)[:, :, None]  # (KM,1) * (KM,N) = (KM,N) => (KM,N,1)

        d2chi2 = self.N[:, None, None] * (dt * (2 - t) * dprobs_p - t * dt * dprobs_p + t * (2 - t) * self.hprobs)
        return d2chi2

    def _approx_hessian_fn(self, pv=None):
        #Almost the same as function above but drops hprobs term
        if self.firsts is not None:
            raise NotImplementedError("Chi2 hessian not implemented for sparse data (yet)")

        dprobs = self.jac
        self.mld.bulk_fill_dprobs(dprobs, self.evTree, self.probs, self.probClipInterval,
                                  self.check, self.comm)  # use cache?

        cprobs = _np.clip(self.probs, self.minProbClipForWeighting, None)
        dprobs_p = dprobs[:, None, :]  # (KM,1,N)
        t = ((self.probs - self.f) / cprobs)[:, None, None]  # (iElement, 0,0) = (KM,1,1)
        dt = ((1.0 / cprobs - (self.probs - self.f) / cprobs**2)[:, None]
              * dprobs)[:, :, None]  # (KM,1) * (KM,N) = (KM,N) => (KM,N,1)

        d2chi2 = self.N[:, None, None] * (dt * (2 - t) * dprobs_p - t * dt * dprobs_p)  # collapse this?
        return d2chi2

    def _d2v_dp2(self):
        # v = N * (p-f)**2 / p  => dv/dp = 2N * (p-f)/p - N * (p-f)**2 / p**2 = 2N * t - N * t**2
        # => d2v/dp2 = 2N*dt - 2N*t*dt = 2N(1-t)*dt
        # Note: in above _approx_hessian_fn we have (note that the `dt` there = JT*dt using the notation here):
        # N * JT*dt * (2 - t) * J - N * t * JT*dt * J  = -2 * N * dt * t * J + 2 * N * dt * J = JT * 2N(1-t)*dt * J, so
        # this is consistent with generic approx hessian formula H = JT * d2v/dp2 * J.
        cprobs = _np.clip(self.probs, self.minProbClipForWeighting, None)
        iclip = (self.probs == self.minProbClipForWeighting)
        t = ((self.probs - self.f) / cprobs)  # should think of as (p-f)/p
        dtdp = (1.0 - (self.probs - self.f) / cprobs) / cprobs  # 1/p - (p-f)/p**2 => 1/cp - (p-f)/cp**2
        d2v_dp2 = 2 * self.N * (1 - t) * dtdp
        d2v_dp2[iclip] = 2 * self.N[iclip] / self.minProbClipForWeighting
        # with cp constant v = N*(p-f)**2/cp => dv/dp = 2N*(p-f)/cp => d2v/dp2 = 2N/cp
        return d2v_dp2


class FreqWeightedChi2Function(Chi2Function):

    def __init__(self, mdl, dataset, circuit_list, regularization=None, penalties=None,
                 cache=None,  resource_alloc=None, name=None, verbosity=0):

        super().__init__(mdl, dataset, circuit_list, regularization, penalties, cache, resource_alloc, name, verbosity)
        self.fweights = _np.sqrt(self.N / _np.clip(self.f, 1e-7, None))
        self.z = _np.zeros(self.KM, 'd')

        #OLD, more complex and seemingly less accurate way to perform weighting:
        #self.fweights = _np.empty(self.KM, 'd')
        #for (i, opStr) in enumerate(self.dsCircuitsToUse):
        #    wts = []
        #    for x in self.outcomes_lookup[i]:
        #        Nx = dataset[opStr].total
        #        f1 = dataset[opStr].fraction(x); f2 = (f1 + 1) / (Nx + 2)
        #        wts.append(_np.sqrt(Nx / (f2 * (1 - f2))))
        #    self.fweights[self.lookup[i]] = wts

    def set_regularization(self, minProbClipForWeighting=1e-4, probClipInterval=(-10000, 10000), radius=1e-4):
        super().set_regularization(minProbClipForWeighting, probClipInterval)
        self.a = radius

    def get_weights(self, p):
        return self.fweights

    def get_dweights(self, p, wts):
        return self.z

    def zero_freq_chi2(self, N, probs):
        a = self.a
        return N * _np.where(probs >= a, probs,
                             (-1.0 / (3 * a**2)) * probs**3 + probs**2 / a + a / 3.0)

    def zero_freq_dchi2(self, N, probs):
        a = self.a
        return N * _np.where(probs >= a, 1.0, (-1.0 / a**2) * probs**2 + 2 * probs / a)


class TimeDependentChi2Function(ObjectiveFunction):

    #This objective function can handle time-dependent circuits - that is, circuitsToUse are treated as
    # potentially time-dependent and mdl as well.  For now, we don't allow any regularization or penalization
    # in this case.
    def __init__(self, mdl, dataset, circuit_list, regularization=None, penalties=None,
                 cache=None, resource_alloc=None, name=None, verbosity=0):

        super().__init__(mdl, dataset, circuit_list, regularization, penalties, cache, resource_alloc, name, verbosity)
        self.time_dependent = True

        #  Allocate peristent memory
        #  (must be AFTER possible operation sequence permutation by
        #   tree and initialization of dsCircuitsToUse)
        self.v = _np.empty(self.KM, 'd')
        self.jac = _np.empty((self.KM + self.ex, self.vec_gs_len), 'd')
        self.maxCircuitLength = max([len(x) for x in self.circuitsToUse])
        self.num_total_outcomes = [mdl.get_num_outcomes(c) for c in self.circuitsToUse]  # for sparse data detection

        # Fast un-regularized version
        self.ls_fn = self.simple_chi2
        self.ls_jfn = self.simple_jac

    def set_penalties(self, regularizeFactor=0, cptp_penalty_factor=0, spam_penalty_factor=0):
        assert(regularizeFactor == 0 and cptp_penalty_factor == 0 and spam_penalty_factor == 0), \
            "Cannot apply regularization or penalization in time-dependent chi2 case (yet)"

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


class ChiAlphaFunction(ObjectiveFunction):

    def __init__(self, mdl, dataset, circuit_list, regularization=None, penalties=None,
                 cache=None, resource_alloc=None, name=None, verbosity=0, alpha=1):
        """ TODO: docstring - note: if radius is None, then the "relaxed" zero-f-term mode is used
            whereas a radius > 0 implies that the "harsh" zero-f-term mode is used.
        """
        super().__init__(mdl, dataset, circuit_list, regularization, penalties, cache, resource_alloc, name, verbosity)
        self.alpha = alpha

        #  Allocate peristent memory
        #  (must be AFTER possible operation sequence permutation by
        #   tree and initialization of dsCircuitsToUse)
        self.probs = _np.empty(self.KM, 'd')
        self.jac = _np.empty((self.KM + self.ex, self.vec_gs_len), 'd')
        self.precompute_omitted_freqs()  # sets self.firsts
        self.cntVecMx, self.totalCntVec = self.compute_count_vectors()
        self.freqs = self.cntVecMx / self.totalCntVec

        # set zero freqs to 1.0 so we don't get divide-by-zero errors
        self.freqs_nozeros = _np.where(self.cntVecMx == 0, 1.0, self.freqs)
        self.fmin = max(1e-7, _np.min(self.freqs_nozeros))  # lowest non-zero frequency
        # (can only be as low as 1e-7 b/c freqs can be arbitarily small in no-sample-error case)

        self.maxCircuitLength = max([len(x) for x in self.circuitsToUse])

        if self.mdl.get_simtype() != "termgap":
            # Fast un-regularized version
            self.ls_fn = self.simple_chi_alpha
            self.ls_jfn = self.simple_jac

        else:
            raise NotImplementedError("Still need to add termgap support to chi-alpha!")
            self.ls_fn = self.termgap_chi_alpha
            self.ls_jfn = self.simple_jac

    def set_regularization(self, pfratio_stitchpt=0.01, pfratio_derivpt=0.01,
                           probClipInterval=(-10000, 10000), radius=None):
        self.x0 = pfratio_stitchpt
        self.x1 = pfratio_derivpt
        self.probClipInterval = probClipInterval

        if radius is None:
            #Infer the curvature of the regularized zero-f-term functions from
            # the largest curvature we use at the stitch-points of nonzero-f terms.
            self.a = None
            self.zero_freq_chialpha = self._zero_freq_chialpha_relaxed
            self.zero_freq_dchialpha = self._zero_freq_dchialpha_relaxed
        else:
            #Use radius to specify the curvature/"roundness" of f == 0 terms,
            # though this uses a more aggressive p^3 function to penalize negative probs.
            self.a = radius
            self.zero_freq_chialpha = self._zero_freq_chialpha_harsh
            self.zero_freq_dchialpha = self._zero_freq_dchialpha_harsh

    def _zero_freq_chialpha_harsh(self, N, probs):
        a = self.a
        return N * _np.where(probs >= a, probs,
                             (-1.0 / (3 * a**2)) * probs**3 + probs**2 / a + a / 3.0)

    def _zero_freq_dchialpha_harsh(self, N, probs):
        a = self.a
        return N * _np.where(probs >= a, 1.0, (-1.0 / a**2) * probs**2 + 2 * probs / a)

    def _zero_freq_chialpha_relaxed(self, N, probs):
        C0 = (0.5 / self.fmin) * (1. + self.alpha) / (self.x1**(2 + self.alpha))
        p0 = 1.0 / C0
        return N * _np.where(probs > p0, probs, C0 * probs**2)

    def _zero_freq_dchialpha_relaxed(self, N, probs):
        C0 = (0.5 / self.fmin) * (1. + self.alpha) / (self.x1**(2 + self.alpha))
        p0 = 1.0 / C0
        return N * _np.where(probs > p0, 1.0, 2 * C0 * probs)

    def _chialpha_from_probs(self, tm, extra=False, debug=False):
        x0 = self.x0
        x1 = self.x1
        x = self.probs / self.freqs_nozeros
        xt = x.copy()
        itaylor = x < x0  # indices where we patch objective function with taylor series
        xt[itaylor] = x0  # so we evaluate function at x0 (first taylor term) at itaylor indices
        v = self.cntVecMx * (xt + 1.0 / (self.alpha * xt**self.alpha) - (1.0 + 1.0 / self.alpha))

        S = 1. - 1. / (x1**(1 + self.alpha))
        S2 = 0.5 * (1. + self.alpha) / x1**(2 + self.alpha)
        v = _np.where(itaylor, v + S * self.cntVecMx * (x - x0) + S2 * self.cntVecMx * (x - x0)**2, v)
        v = _np.where(self.cntVecMx == 0, self.zero_freq_chialpha(self.totalCntVec, self.probs), v)

        #DEBUG TODO REMOVE
        if debug and (self.comm is None or self.comm.Get_rank() == 0):
            print("ALPHA OBJECTIVE: ", S, S2)
            print(" KM=",len(x), " nTaylored=",_np.count_nonzero(itaylor), " nZero=",_np.count_nonzero(self.cntVecMx==0))
            print(" xrange = ",_np.min(x),_np.max(x))
            print(" vrange = ",_np.min(v),_np.max(v))
            print(" |v|^2 = ",_np.sum(v))
            print(" |v(normal)|^2 = ",_np.sum(v[x >= x0]))
            print(" |v(taylor)|^2 = ",_np.sum(v[x < x0]))
            imax = _np.argmax(v)
            print(" MAX: v=",v[imax]," x=",x[imax]," p=",self.probs[imax]," f=",self.freqs[imax])

        if self.firsts is not None:
            #self.update_v_for_omitted_probs(v, self.probs)
            omitted_probs = 1.0 - _np.array([_np.sum(self.probs[self.lookup[i]])
                                             for i in self.indicesOfCircuitsWithOmittedData])
            #omitted_probs = _np.maximum(omitted_probs, 0.0)  # don't let other probs adding to > 1 reduce penalty
            v[self.firsts] += self.zero_freq_chialpha(self.totalCntVec[self.firsts], omitted_probs)

            #DEBUG TODO REMOVE
            if debug and (self.comm is None or self.comm.Get_rank() == 0):
                print(" vrange2 = ",_np.min(v),_np.max(v))
                print(" omitted_probs range = ", _np.min(omitted_probs), _np.max(omitted_probs))
                #print(" nSparse = ",len(self.firsts), " nOmitted >radius=", _np.count_nonzero(omitted_probs >= self.a),
                #      " <0=", _np.count_nonzero(omitted_probs < 0))
                p0 = 1.0 / (0.5 * (1. + self.alpha) / (self.x1**(2 + self.alpha) * self.fmin))
                print(" nSparse = ",len(self.firsts), " nOmitted >p0=", _np.count_nonzero(omitted_probs >= p0),
                      " <0=", _np.count_nonzero(omitted_probs < 0))
                print(" |v(post-sparse)|^2 = ",_np.sum(v))
        else:
            omitted_probs = None  # b/c we might return this

        v = _np.sqrt(v)
        self.profiler.add_time("chi-alpha: OBJECTIVE", tm)
        assert(v.shape == (self.KM,))  # reshape ensuring no copy is needed

        if extra:
            return v, (x, itaylor, S, S2, omitted_probs)
        else:
            return v

    #Objective Function
    def simple_chi_alpha(self, vectorGS):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval, self.check, self.comm)
        return self._chialpha_from_probs(tm, debug=True)

    #def termgap_chi_alpha(self, vectorGS, oob_check=False):
    #    TODO: this need to be updated - this is just the chi2 version:
    #    tm = _time.time()
    #    self.mdl.from_vector(vectorGS)
    #    self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval, self.check, self.comm)
    #
    #    if oob_check:
    #        if not self.mdl.bulk_probs_paths_are_sufficient(self.evTree,
    #                                                        self.probs,
    #                                                        self.comm,
    #                                                        memLimit=None,
    #                                                        verbosity=1):
    #            raise ValueError("Out of bounds!")  # signals LM optimizer
    #
    #    v = (self.probs - self.f) * self.get_weights(self.probs)  # dims K x M (K = nSpamLabels, M = nCircuits)
    #
    #    if self.firsts is not None:
    #        self.update_v_for_omitted_probs(v, self.probs)
    #
    #    self.profiler.add_time("do_mc2gst: OBJECTIVE", tm)
    #    assert(v.shape == (self.KM,))  # reshape ensuring no copy is needed
    #    return v

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

        if self.firsts is not None:
            for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.lookup[i], :], axis=0)

        #objective is sqrt(Nf * (x + 1/(alpha*x^alpha) + C))
        # deriv is 0.5/objective * Nf * (1 - 1/x^(1+alpha)) * dx/dprobs , where dx/dprobs = 1/f * dp/dprobs
        # so deriv = 0.5/objective * N * (1 - 1/x^(1+alpha)) * dp/dprobs
        v, (x, itaylor, S, S2, omitted_probs) = self._chialpha_from_probs(tm, extra=True)

        # derivative diverges as v->0, but v always >= 0 so clip v to a small positive value to
        # avoid divide by zero below
        v = _np.maximum(v, 1e-100)

        # compute jacobian = 0.5/v * N * (1 - 1/x^(1+alpha)) * dp/dprobs
        x0 = self.x0
        dprobs_factor = (0.5 / v) * self.totalCntVec * (1 - 1. / x**(1. + self.alpha))
        dprobs_factor_taylor = (0.5 / v) * self.totalCntVec * (S + 2 * S2 * (x - x0))
        dprobs_factor_zerofreq = (0.5 / v) * self.zero_freq_dchialpha(self.totalCntVec, self.probs)
        dprobs_factor[itaylor] = dprobs_factor_taylor[itaylor]
        dprobs_factor = _np.where(self.cntVecMx == 0, dprobs_factor_zerofreq, dprobs_factor)

        if self.firsts is not None:
            dprobs_factor_omitted = (-0.5 / v[self.firsts]) * self.zero_freq_dchialpha(
                self.totalCntVec[self.firsts], omitted_probs)

            for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.lookup[i], :], axis=0)

        dprobs *= dprobs_factor[:, None]  # (KM,N) * (KM,1)   (N = dim of vectorized model)

        # need to multipy dprobs_factor_omitted[i] * dprobs[k] for k in lookup[i] and
        # add to dprobs[firsts[i]] for i in indicesOfCircuitsWithOmittedData
        if self.firsts is not None:
            dprobs[self.firsts, :] += dprobs_factor_omitted[:, None] * self.dprobs_omitted_rowsum
            # nCircuitsWithOmittedData x N

        if self.check_jacobian: _opt.check_jac(lambda v: self.simple_chi_alpha(
            v), vectorGS, self.jac, tol=1e-3, eps=1e-6, errType='abs')  # TO FIX

        self.profiler.add_time("chi-alpha: JACOBIAN", tm)
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
class DeltaLogLFunctionPoissonPic(ObjectiveFunction):

    def __init__(self, mdl, dataset, circuit_list, regularization=None, penalties=None,
                 cache=None, resource_alloc=None, name=None, verbosity=0, enable_hessian=False):

        self.enable_hessian = enable_hessian
        super().__init__(mdl, dataset, circuit_list, regularization, penalties, cache, resource_alloc, name, verbosity)

        #Allocate peristent memory
        self.probs = _np.empty(self.KM, 'd')
        self.jac = _np.empty((self.KM + self.ex, self.vec_gs_len), 'd')

        self.precompute_omitted_freqs()  # sets self.firsts
        cntVecMx, self.totalCntVec = self.compute_count_vectors()
        self.minusCntVecMx = -1.0 * cntVecMx
        self.freqs = cntVecMx / self.totalCntVec
        # set zero freqs to 1.0 so np.log doesn't complain
        self.freqs_nozeros = _np.where(cntVecMx == 0, 1.0, self.freqs)
        self.freqTerm = cntVecMx * (_np.log(self.freqs_nozeros) - 1.0)
        self.fmin = max(1e-7, _np.min(self.freqs_nozeros))  # lowest non-zero frequency
        # (can only be as low as 1e-7 b/c freqs can be arbitarily small in no-sample-error case)

        if mdl.get_simtype() == "termgap":
            assert(self.regtype == "minp"), "termgap simtype is only implemented for 'minp' reg-type thus far."
            assert(self.cptp_penalty_factor == 0), "Cannot have cptp_penalty_factor != 0 when using the termgap simtype"
            assert(self.spam_penalty_factor == 0), "Cannot have spam_penalty_factor != 0 when using the termgap simtype"
            assert(self.forcefn_grad is None), "Cannot use force functions when using the termgap simtype"
            self.ls_fn = self.termgap_poisson_picture_logl
            self.ls_jfn = self.poisson_picture_jacobian  # same jacobian as normal case
        else:
            self.ls_fn = self.poisson_picture_logl
            self.ls_jfn = self.poisson_picture_jacobian
        self.hfn = self.poisson_picture_hessian

    def get_evaltree_subcalls(self):
        calls = ["bulk_fill_probs", "bulk_fill_dprobs"]
        if self.enable_hessian: calls.append('bulk_hprobs_by_block')
        return calls

    def set_penalties(self, cptp_penalty_factor=0, spam_penalty_factor=0, forcefn_grad=None, shiftFctr=100):
        self.cptp_penalty_factor = cptp_penalty_factor
        self.spam_penalty_factor = spam_penalty_factor
        self.forcefn_grad = forcefn_grad

        #Compute "extra" (i.e. beyond the (circuit,spamlable)) rows of jacobian
        ex = 0
        if cptp_penalty_factor != 0: ex += _cptp_penalty_size(self.mdl)
        if spam_penalty_factor != 0: ex += _spam_penalty_size(self.mdl)

        if forcefn_grad is not None:
            ex += forcefn_grad.shape[0]

            ffg_norm = _np.linalg.norm(forcefn_grad)
            start_norm = _np.linalg.norm(self.mdl.to_vector())
            self.forceShift = ffg_norm * (ffg_norm + start_norm) * shiftFctr
            #used to keep forceShift - _np.dot(forcefn_grad,vectorGS) positive
            # Note -- not analytic, just a heuristic!
            self.forceOffset = self.KM
            if cptp_penalty_factor != 0: self.forceOffset += _cptp_penalty_size(self.mdl)
            if spam_penalty_factor != 0: self.forceOffset += _spam_penalty_size(self.mdl)
            #index to jacobian row of first forcing term

        return ex

    def set_regularization(self, minProbClip=1e-4, pfratio_stitchpt=None, pfratio_derivpt=None,
                           probClipInterval=(-10000, 10000), radius=1e-4):
        if minProbClip is not None:
            assert(pfratio_stitchpt is None and pfratio_derivpt is None), \
                "Cannot specify pfratio and minProbClip arguments as non-None!"
            self.min_p = minProbClip
            self.regtype = "minp"
        else:
            assert(minProbClip is None), "Cannot specify pfratio and minProbClip arguments as non-None!"
            self.x0 = pfratio_stitchpt
            self.x1 = pfratio_derivpt
            self.regtype = "pfratio"
        self.probClipInterval = probClipInterval

        if radius is None:
            #Infer the curvature of the regularized zero-f-term functions from
            # the largest curvature we use at the stitch-points of nonzero-f terms.
            assert(self.regtype == 'pfratio'), "Must specify `radius` when %s regularization type" % self.regtype
            self.a = None
            self.zero_freq_poisson_logl = self._zero_freq_poisson_logl_relaxed
            self.zero_freq_poisson_dlogl = self._zero_freq_poisson_dlogl_relaxed
        else:
            #Use radius to specify the curvature/"roundness" of f == 0 terms,
            # though this uses a more aggressive p^3 function to penalize negative probs.
            self.a = radius
            self.zero_freq_poisson_logl = self._zero_freq_poisson_logl_harsh
            self.zero_freq_poisson_dlogl = self._zero_freq_poisson_dlogl_harsh

    def get_chi2k_distributed_qty(self, objective_function_value):
        return 2 * objective_function_value  # 2 * deltaLogL is what is chi2_k distributed

    def poisson_picture_logl(self, vectorGS):
        tm = _time.time()
        self.mdl.from_vector(vectorGS)
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval,
                                 self.check, self.comm)
        return self._v_from_probs(tm, debug=True)

    def _zero_freq_poisson_logl_harsh(self, N, probs):
        a = self.a
        return N * _np.where(probs >= a, probs,
                             (-1.0 / (3 * a**2)) * probs**3 + probs**2 / a + a / 3.0)

    def _zero_freq_poisson_dlogl_harsh(self, N, probs):
        a = self.a
        return N * _np.where(probs >= a, 1.0, (-1.0 / a**2) * probs**2 + 2 * probs / a)

    def _zero_freq_poisson_logl_relaxed(self, N, probs):
        C0 = (0.5 / self.fmin) * 1.0 / (self.x1**2)
        p0 = 1.0 / C0
        return N * _np.where(probs > p0, probs, C0 * probs**2)

    def _zero_freq_poisson_dlogl_relaxed(self, N, probs):
        C0 = (0.5 / self.fmin) * 1.0 / (self.x1**2)
        p0 = 1.0 / C0
        return N * _np.where(probs > p0, 1.0, 2 * C0 * probs)

    def _v_from_probs(self, tm_start, extra=False, debug=False):

        if self.regtype == 'pfratio':
            x0 = self.x0
            x1 = self.x1
            x = self.probs / self.freqs_nozeros  # objective is -Nf*(log(x) + 1 - x)

            #DEBUG TODO REMOVE
            #if self.comm.Get_rank() == 0 and debug:
            #    print(">>>> DEBUG ----------------------------------")
            #    print("x range = ",_np.min(x), _np.max(x))
            #    print("p range = ",_np.min(self.probs), _np.max(self.probs))
            #    #print("f range = ",_np.min(self.freqs), _np.max(self.freqs))
            #    #print("fnz range = ",_np.min(self.freqs_nozeros), _np.max(self.freqs_nozeros))
            #    #print("TVD = ", _np.sum(_np.abs(self.probs - self.freqs)))
            #    print(" KM=",len(x), " nTaylored=",_np.count_nonzero(x < x0), " nZero=",_np.count_nonzero(self.minusCntVecMx==0))
            #    #for i,el in enumerate(x):
            #    #    if el < 0.1 or el > 10.0:
            #    #        print("-> x=%g  p=%g  f=%g  fnz=%g" % (el, self.probs[i], self.freqs[i], self.freqs_nozeros[i]))
            #    print("<<<<< DEBUG ----------------------------------")
            pos_x = _np.where(x < x0, x0, x)
            S = self.minusCntVecMx * (1 / x1 - 1)  # deriv wrt x at x == x0 (=min_p)
            S2 = -0.5 * self.minusCntVecMx / (x1**2)  # 0.5 * 2nd deriv at x0

            #pos_x = _np.where(x > 1 / x0, 1 / x0, pos_x)
            #T = self.minusCntVecMx * (x0 - 1)  # deriv wrt x at x == 1/x0
            #T2 = -0.5 * self.minusCntVecMx / (1 / x0**2)  # 0.5 * 2nd deriv at 1/x0

            v = self.minusCntVecMx * (1.0 - pos_x + _np.log(pos_x))
            #Note: order of +/- terms above is important to avoid roundoff errors when x is near 1.0
            # (see patching line below).  For example, using log(x) + 1 - x causes significant loss
            # of precision because log(x) is tiny and so is |1-x| but log(x) + 1 == 1.0.

            # omit = 1-p1-p2  => 1/2-p1 + 1/2-p2

            # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
            v = _np.maximum(v, 0)
            # quadratic extrapolation of logl at x0 for probabilities/frequencies < x0
            v = _np.where(x < x0, v + S * (x - x0) + S2 * (x - x0)**2, v)
            #v = _np.where(x > 1 / x0, v + T * (x - x0) + T2 * (x - x0)**2, v)

        elif self.regtype == 'minp':
            pos_probs = _np.where(self.probs < self.min_p, self.min_p, self.probs)
            S = self.minusCntVecMx / self.min_p + self.totalCntVec
            S2 = -0.5 * self.minusCntVecMx / (self.min_p**2)
            v = self.freqTerm + self.minusCntVecMx * _np.log(pos_probs) + self.totalCntVec * \
                pos_probs  # dims K x M (K = nSpamLabels, M = nCircuits)

            # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
            v = _np.maximum(v, 0)
            # quadratic extrapolation of logl at min_p for probabilities < min_p
            v = _np.where(self.probs < self.min_p,
                          v + S * (self.probs - self.min_p) + S2 * (self.probs - self.min_p)**2, v)
        else:
            raise ValueError("Invalid regularization type: %s" % self.regtype)

        v = _np.where(self.minusCntVecMx == 0, self.zero_freq_poisson_logl(self.totalCntVec, self.probs), v)
        # special handling for f == 0 terms
        # using cubit rounding of function that smooths N*p for p>0:
        #  has minimum at p=0; matches value, 1st, & 2nd derivs at p=a.

        #DEBUG TODO REMOVE
        #if debug and (self.comm is None or self.comm.Get_rank() == 0):
        #    print("LOGL OBJECTIVE: ")
        #    #print(" KM=",len(x), " nTaylored=",_np.count_nonzero(x < x0), " nZero=",_np.count_nonzero(self.minusCntVecMx==0))
        #    print(" KM=",len(self.probs), " nTaylored=",_np.count_nonzero(self.probs < self.min_p), " nZero=",_np.count_nonzero(self.minusCntVecMx==0))
        #    #print(" xrange = ",_np.min(x),_np.max(x))
        #    print(" prange = ",_np.min(self.probs),_np.max(self.probs))
        #    print(" vrange = ",_np.min(v),_np.max(v))
        #    print(" |v|^2 = ",_np.sum(v))
        #    #print(" |v(normal)|^2 = ",_np.sum(v[x >= x0]))
        #    #print(" |v(taylor)|^2 = ",_np.sum(v[x < x0]))
        #    print(" |v(normal)|^2 = ",_np.sum(v[self.probs >= self.min_p]))
        #    print(" |v(taylor)|^2 = ",_np.sum(v[self.probs < self.min_p]))
        #    imax = _np.argmax(v)
        #    print(" MAX: v=",v[imax]," p=",self.probs[imax]," f=",self.freqs[imax]) # " x=",x[imax]," pos_x=",pos_x[imax],

        if self.firsts is not None:
            omitted_probs = 1.0 - _np.array([_np.sum(pos_x[self.lookup[i]] * self.freqs_nozeros[self.lookup[i]])
                                             for i in self.indicesOfCircuitsWithOmittedData])
            v[self.firsts] += self.zero_freq_poisson_logl(self.totalCntVec[self.firsts], omitted_probs)

            #DEBUG TODO REMOVE
            #if debug and (self.comm is None or self.comm.Get_rank() == 0):
            #    print(" vrange2 = ",_np.min(v),_np.max(v))
            #    print(" omitted_probs range = ", _np.min(omitted_probs), _np.max(omitted_probs))
            #    p0 = 1.0 / ((0.5 / self.fmin) * 1.0 / self.x1**2)
            #    print(" nSparse = ",len(self.firsts), " nOmitted >p0=", _np.count_nonzero(omitted_probs >= p0),
            #          " <0=", _np.count_nonzero(omitted_probs < 0))
            #    print(" |v(post-sparse)|^2 = ",_np.sum(v))
        else:
            omitted_probs = None  # b/c we might return this

        #CHECK OBJECTIVE FN
        #logL_terms = _tools.logl_terms(mdl, dataset, circuitsToUse,
        #                                     min_p, probClipInterval, a, poissonPicture, False,
        #                                     opLabelAliases, evaltree_cache) # v = maxL - L so L + v - maxL should be 0
        #print("DIFF2 = ",_np.sum(logL_terms), _np.sum(v), _np.sum(freqTerm), abs(_np.sum(logL_terms)
        #      + _np.sum(v)-_np.sum(freqTerm)))

        v.shape = [self.KM]  # reshape ensuring no copy is needed
        v = _np.sqrt(v)
        if self.regtype == "pfratio":
            # post-sqrt(v) 1st order taylor patch for x near 1.0 - maybe unnecessary
            v = _np.where(_np.abs(x - 1) < 1e-6,
                          _np.sqrt(-self.minusCntVecMx) * _np.abs(x - 1) / _np.sqrt(2), v)

            if extra:  # then used for jacobian where penalty terms are added later, so return now
                return v, (x, pos_x, S, S2, omitted_probs)
        elif self.regtype == 'minp':
            if extra:  # then used for jacobian where penalty terms are added later, so return now
                return v, (pos_probs, S, S2, omitted_probs)

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

    #TODO REMOVE
    #def OLD_poisson_picture_v_from_probs(self, tm_start):
    #    pos_probs = _np.where(self.probs < self.min_p, self.min_p, self.probs)
    #    S = self.minusCntVecMx / self.min_p + self.totalCntVec
    #    S2 = -0.5 * self.minusCntVecMx / (self.min_p**2)
    #    v = self.freqTerm + self.minusCntVecMx * _np.log(pos_probs) + self.totalCntVec * \
    #        pos_probs  # dims K x M (K = nSpamLabels, M = nCircuits)
    #
    #    #TODO REMOVE - pseudocode used for testing/debugging
    #    #nExpectedOutcomes = 2
    #    #for i in range(ng): # len(circuitsToUse)
    #    #    ps = pos_probs[lookup[i]]
    #    #    if len(ps) < nExpectedOutcomes:
    #    #        #omitted_prob = max(1.0-sum(ps),0) # if existing probs add to >1 just forget correction
    #    #        #iFirst = lookup[i].start #assumes lookup holds slices
    #    #        #v[iFirst] += totalCntVec[iFirst] * omitted_prob #accounts for omitted terms (sparse data)
    #    #        for j in range(lookup[i].start,lookup[i].stop):
    #    #            v[j] += totalCntVec[j]*(1.0/len(ps) - pos_probs[j])
    #
    #    # omit = 1-p1-p2  => 1/2-p1 + 1/2-p2
    #
    #    # remove small negative elements due to roundoff error (above expression *cannot* really be negative)
    #    v = _np.maximum(v, 0)
    #    # quadratic extrapolation of logl at min_p for probabilities < min_p
    #    v = _np.where(self.probs < self.min_p, v + S * (self.probs - self.min_p) + S2 * (self.probs - self.min_p)**2, v)
    #    v = _np.where(self.minusCntVecMx == 0,
    #                  self.totalCntVec * _np.where(self.probs >= self.a,
    #                                               self.probs,
    #                                               (-1.0 / (3 * self.a**2)) * self.probs**3 + self.probs**2 / self.a
    #                                               + self.a / 3.0),
    #                  v)
    #    # special handling for f == 0 terms
    #    # using cubit rounding of function that smooths N*p for p>0:
    #    #  has minimum at p=0; matches value, 1st, & 2nd derivs at p=a.
    #
    #    if self.firsts is not None:
    #        omitted_probs = 1.0 - _np.array([_np.sum(pos_probs[self.lookup[i]])
    #                                         for i in self.indicesOfCircuitsWithOmittedData])
    #        v[self.firsts] += self.totalCntVec[self.firsts] * \
    #            _np.where(omitted_probs >= self.a, omitted_probs,
    #                      (-1.0 / (3 * self.a**2)) * omitted_probs**3 + omitted_probs**2 / self.a + self.a / 3.0)
    #
    #    #CHECK OBJECTIVE FN
    #    #logL_terms = _tools.logl_terms(mdl, dataset, circuitsToUse,
    #    #                                     min_p, probClipInterval, a, poissonPicture, False,
    #    #                                     opLabelAliases, evaltree_cache) # v = maxL - L so L + v - maxL should be 0
    #    #print("DIFF2 = ",_np.sum(logL_terms), _np.sum(v), _np.sum(freqTerm), abs(_np.sum(logL_terms)
    #    #      + _np.sum(v)-_np.sum(freqTerm)))
    #
    #    v = _np.sqrt(v)
    #    v.shape = [self.KM]  # reshape ensuring no copy is needed
    #    if self.cptp_penalty_factor != 0:
    #        cpPenaltyVec = _cptp_penalty(self.mdl, self.cptp_penalty_factor, self.opBasis)
    #    else: cpPenaltyVec = []
    #
    #    if self.spam_penalty_factor != 0:
    #        spamPenaltyVec = _spam_penalty(self.mdl, self.spam_penalty_factor, self.opBasis)
    #    else: spamPenaltyVec = []
    #
    #    v = _np.concatenate((v, cpPenaltyVec, spamPenaltyVec))
    #
    #    if self.forcefn_grad is not None:
    #        forceVec = self.forceShift - _np.dot(self.forcefn_grad, self.mdl.to_vector())
    #        assert(_np.all(forceVec >= 0)), "Inadequate forcing shift!"
    #        v = _np.concatenate((v, _np.sqrt(forceVec)))
    #
    #    # TODO: handle dummy profiler generation in simple_init??
    #    if self.profiler: self.profiler.add_time("do_mlgst: OBJECTIVE", tm_start)
    #    return v  # Note: no test for whether probs is in [0,1] so no guarantee that
    #    #      sqrt is well defined unless probClipInterval is set within [0,1].

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

        if self.regtype == 'pfratio':
            v, (x, pos_x, S, S2, omitted_probs) = self._v_from_probs(tm, extra=True)
            x0 = self.x0
        elif self.regtype == 'minp':
            v, (pos_probs, S, S2, omitted_probs) = self._v_from_probs(tm, extra=True)

        # derivative should not actually diverge as v->0, but clip v (always >= 0) to a small positive
        # value to avoid divide by zero errors below.
        v = _np.maximum(v, 1e-100)

        # compute jacobian
        if self.regtype == 'pfratio':
            dprobs_factor_pos = (0.5 / v) * (self.totalCntVec * (-1 / pos_x + 1))
            dprobs_factor_neg = (0.5 / v) * (S + 2 * S2 * (x - x0)) / self.freqs_nozeros
            #dprobs_factor_neg2 = (0.5 / v) * (T + 2 * T2 * (x - x0)) / self.freqs_nozeros
            dprobs_factor = _np.where(x < x0, dprobs_factor_neg, dprobs_factor_pos)
            #dprobs_factor = _np.where(x > 1 / x0, dprobs_factor_neg2, dprobs_facto

        elif self.regtype == 'minp':
            dprobs_factor_pos = (0.5 / v) * (self.minusCntVecMx / pos_probs + self.totalCntVec)
            dprobs_factor_neg = (0.5 / v) * (S + 2 * S2 * (self.probs - self.min_p))
            dprobs_factor = _np.where(self.probs < self.min_p, dprobs_factor_neg, dprobs_factor_pos)

        dprobs_factor_zerofreq = (0.5 / v) * self.zero_freq_poisson_dlogl(self.totalCntVec, self.probs)
        dprobs_factor = _np.where(self.minusCntVecMx == 0, dprobs_factor_zerofreq, dprobs_factor)

        if self.firsts is not None:
            dprobs_factor_omitted = (-0.5 / v[self.firsts]) * self.zero_freq_poisson_dlogl(
                self.totalCntVec[self.firsts], omitted_probs)

            for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
                self.dprobs_omitted_rowsum[ii, :] = _np.sum(dprobs[self.lookup[i], :], axis=0)

        dprobs *= dprobs_factor[:, None]  # (KM,N) * (KM,1)   (N = dim of vectorized model)
        #Note: this also sets jac[0:KM,:]

        #if (self.comm is None or self.comm.Get_rank() == 0):
        #    print("LOGL JACOBIAN: ")
        #    print(" vrange = ",_np.min(v**2),_np.max(v**2))  # v**2 to match OBJECTIVE print stmts
        #    print(" |v|^2 = ",_np.sum(v**2))
        #    print(" |jac|^2 = ",_np.linalg.norm(dprobs))
        #    print(" |dprobs_factor_pos| = ",_np.linalg.norm(dprobs_factor_pos[_np.logical_and(x >= x0, self.minusCntVecMx != 0)]))
        #    print(" |dprobs_factor_neg| = ",_np.linalg.norm(dprobs_factor_neg[_np.logical_and(x < x0, self.minusCntVecMx != 0)]))
        #    print(" |dprobs_factor_zero| = ",_np.linalg.norm(dprobs_factor_zerofreq[self.minusCntVecMx == 0]))
        #    chk = dprobs_factor_pos.copy()
        #    chk[_np.logical_or(x < x0, self.minusCntVecMx == 0)] = 0
        #    imax = _np.argmax(chk)
        #    print(" MAX: chk=",chk[imax]," v=",v[imax]," x=",x[imax]," pos_x=",pos_x[imax]," p=",self.probs[imax]," f=",self.freqs[imax])

        # need to multipy dprobs_factor_omitted[i] * dprobs[k] for k in lookup[i] and
        # add to dprobs[firsts[i]] for i in indicesOfCircuitsWithOmittedData
        if self.firsts is not None:
            dprobs[self.firsts, :] += dprobs_factor_omitted[:, None] * self.dprobs_omitted_rowsum
            # nCircuitsWithOmittedData x N
            if (self.comm is None or self.comm.Get_rank() == 0):
                print(" |dprobs_omitted_rowsum| = ",_np.linalg.norm(self.dprobs_omitted_rowsum))
                print(" |dprobs_factor_omitted| = ",_np.linalg.norm(dprobs_factor_omitted))
                print(" |jac(post-sparse)| = ",_np.linalg.norm(dprobs))

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
        assert(self.regtype == 'minp'), "Only regtype='minp' is supported for termgap calcs so far!"
        pos_probs = _np.where(probs < self.min_p, self.min_p, probs)
        v = self.freqTerm + self.minusCntVecMx * _np.log(pos_probs) + self.totalCntVec * \
            pos_probs  # dims K x M (K = nSpamLabels, M = nCircuits)
        v = _np.maximum(v, 0)

        # quadratic extrapolation of logl at min_p for probabilities < min_p
        v = _np.where(probs < self.min_p, v + S * (probs - self.min_p) + S2 * (probs - self.min_p)**2, v)
        v = _np.where(self.minusCntVecMx == 0, self.zero_freq_poisson_logl(self.totalCntVec, probs), v)
        # special handling for f == 0 terms
        # using cubit rounding of function that smooths N*p for p>0:
        #  has minimum at p=0; matches value, 1st, & 2nd derivs at p=a.

        if self.firsts is not None:
            omitted_probs = 1.0 - _np.array([_np.sum(pos_probs[self.lookup[i]])
                                             for i in self.indicesOfCircuitsWithOmittedData])
            v[self.firsts] += self.zero_freq_poisson_logl(self.totalCntVec[self.firsts], omitted_probs)

        v.shape = [self.KM]  # reshape ensuring no copy is needed
        return v

    def termgap_poisson_picture_logl(self, vectorGS, oob_check=False):
        assert(self.regtype == 'minp'), "Only regtype='minp' is supported for termgap calcs so far!"
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

    def poisson_picture_hessian(self, pvec=None):
        assert(self.regtype == 'minp'), "Only regtype='minp' is supported for hessian calcs so far!"
        if pvec is None: pvec = self.mdl.to_vector()
        return self._construct_hessian(-self.minusCntVecMx, self.totalCntVec, self.probClipInterval)

    def _hessian_from_hprobs(self, hprobs, dprobs12, cntVecMx, totalCntVec, probs):
        """ Factored-out computation of hessian from raw components """
        # Notation:  (K=#spam, M=#strings, N=#wrtParams1, N'=#wrtParams2 )
        a = self.a
        min_p = self.min_p
        totCnts = totalCntVec  # shorthand
        pos_probs = _np.where(probs < self.min_p, self.min_p, probs)
        S = -cntVecMx / min_p + totCnts  # slope term that is derivative of logl at min_p
        S2 = 0.5 * cntVecMx / (min_p**2)          # 2nd derivative of logl term at min_p

        #Allocate these above?  Need to know block sizes of dprobs12 & hprobs...
        if self.firsts is not None:
            dprobs12_omitted_rowsum = _np.empty((len(self.firsts),) + dprobs12.shape[1:], 'd')
            hprobs_omitted_rowsum = _np.empty((len(self.firsts),) + hprobs.shape[1:], 'd')

        # # (K,M,1,1) * (K,M,N,N')
        # hprobs_pos  = (-cntVecMx / pos_probs**2)[:,:,None,None] * dprobs12
        # # (K,M,1,1) * (K,M,N,N')
        # hprobs_pos += (cntVecMx / pos_probs - totalCntVec[None,:])[:,:,None,None] * hprobs
        # # (K,M,1,1) * (K,M,N,N')
        # hprobs_neg  = (2*S2)[:,:,None,None] * dprobs12 + (S + 2*S2*(probs - min_p))[:,:,None,None] * hprobs
        # hprobs_zerofreq = _np.where( (probs >= a)[:,:,None,None],
        #                             -totalCntVec[None,:,None,None] * hprobs,
        #                             (-totalCntVec[None,:] * ( (-2.0/a**2)*probs + 2.0/a))[:,:,None,None] \
        #                              * dprobs12
        #                             - (totalCntVec[None,:] * ((-1.0/a**2)*probs**2 + 2*probs/a))[:,:,None,None] \
        #                              * hprobs )
        # hessian = _np.where( (probs < min_p)[:,:,None,None], hprobs_neg, hprobs_pos)
        # hessian = _np.where( (cntVecMx == 0)[:,:,None,None], hprobs_zerofreq, hessian) # (K,M,N,N')

        omitted_probs = 1.0 - _np.array([_np.sum(pos_probs[self.lookup[i]])
                                         for i in self.indicesOfCircuitsWithOmittedData])
        for ii, i in enumerate(self.indicesOfCircuitsWithOmittedData):
            dprobs12_omitted_rowsum[ii, :, :] = _np.sum(dprobs12[self.lookup[i], :, :], axis=0)
            hprobs_omitted_rowsum[ii, :, :] = _np.sum(hprobs[self.lookup[i], :, :], axis=0)

        #Accomplish the same thing as the above commented-out lines,
        # but with more memory effiency:
        dprobs12_coeffs = \
            _np.where(probs < min_p, 2 * S2, cntVecMx / pos_probs**2)
        zfc = _np.where(probs >= a, 0.0, totCnts * ((-2.0 / a**2) * probs + 2.0 / a))
        dprobs12_coeffs = _np.where(cntVecMx == 0, zfc, dprobs12_coeffs)

        hprobs_coeffs = \
            _np.where(probs < min_p, S + 2 * S2 * (probs - min_p),
                      -cntVecMx / pos_probs + totCnts)
        zfc = _np.where(probs >= a, totCnts,
                        totCnts * ((-1.0 / a**2) * probs**2 + 2 * probs / a))
        hprobs_coeffs = _np.where(cntVecMx == 0, zfc, hprobs_coeffs)

        if self.firsts is not None:
            dprobs12_omitted_coeffs = totCnts[self.firsts] * _np.where(
                omitted_probs >= a, 0.0, (2.0 / a**2) * omitted_probs - 2.0 / a)
            hprobs_omitted_coeffs = totCnts[self.firsts] * _np.where(
                omitted_probs >= a, 1.0,
                (1.0 / a**2) * omitted_probs**2 - 2 * omitted_probs / a)

        # hessian = hprobs_coeffs * hprobs + dprobs12_coeff * dprobs12
        #  but re-using dprobs12 and hprobs memory (which is overwritten!)
        hprobs *= hprobs_coeffs[:, None, None]
        dprobs12 *= dprobs12_coeffs[:, None, None]
        if self.firsts is not None:
            hprobs[self.firsts, :, :] += hprobs_omitted_coeffs[:, None, None] * hprobs_omitted_rowsum
            dprobs12[self.firsts, :, :] += dprobs12_omitted_coeffs[:, None, None] * dprobs12_omitted_rowsum
        hessian = dprobs12; hessian += hprobs

        # hessian[iSpamLabel,iCircuit,iModelParam1,iModelParams2] contains all
        #  d2(logl)/d(modelParam1)d(modelParam2) contributions
        return _np.sum(hessian, axis=0)
        # sum over spam label and operation sequence dimensions (operation sequences in evalSubTree)
        # adds current subtree contribution for (N,N')-sized block of Hessian

    def _d2v_dp2(self):
        assert(self.regtype == "minp")
        pos_probs = _np.where(self.probs < self.min_p, self.min_p, self.probs)
        S2 = -0.5 * self.minusCntVecMx / (self.min_p**2)   # 2nd derivative of logl term at min_p
        d2v_dp2 = _np.where(self.probs < self.min_p,
                            2 * S2,
                            -self.minusCntVecMx / pos_probs**2)
        zfc = _np.where(self.probs >= self.a, 0.0,
                        self.totalCntVec * ((-2.0 / self.a**2) * self.probs + 2.0 / self.a))
        d2v_dp2 = _np.where(self.minusCntVecMx == 0, zfc, d2v_dp2)
        return d2v_dp2  # a 1D array of the diagonal of d2(logl)/dprobs2; shape = (nEls,)


class DeltaLogLFunction(ObjectiveFunction):
    """ TODO: standard logl function, which will *not* implement self.ls_fn, as it doesn't have always >=0 terms"""
    def __init__(self, mdl, dataset, circuit_list, regularization=None, penalties=None,
                 cache=None, resource_alloc=None, name=None, verbosity=0, enable_hessian=False):
        self.enable_hessian = enable_hessian
        super().__init__(mdl, dataset, circuit_list, regularization, penalties, cache, resource_alloc, name, verbosity)

        #Allocate peristent memory
        self.probs = _np.empty(self.KM, 'd')
        self.jac = _np.empty((self.KM + self.ex, self.vec_gs_len), 'd')

        cntVecMx, self.totalCntVec = self.compute_count_vectors()
        self.minusCntVecMx = -1.0 * cntVecMx
        self.freqs = cntVecMx / self.totalCntVec
        # set zero freqs to 1.0 so np.log doesn't complain
        self.freqs_nozeros = _np.where(cntVecMx == 0, 1.0, self.freqs)
        self.freqTerm = cntVecMx * _np.log(self.freqs_nozeros)

        if mdl.get_simtype() == "termgap":
            raise NotImplementedError("Term-calculations are not implemented for non-poisson-picture logl function.")
        else:
            self.ls_fn = None
            self.ls_jfn = None
            self.fn = self._logl_fn
            self.jfn = self._logl_jac
            self.hfn = self._logl_hessian

    def get_evaltree_subcalls(self):
        calls = ["bulk_fill_probs", "bulk_fill_dprobs"]
        if self.enable_hessian: calls.append('bulk_hprobs_by_block')
        return calls

    def set_regularization(self, minProbClip=1e-4, pfratio_stitchpt=None, pfratio_derivpt=None,
                           probClipInterval=(-10000, 10000)):
        if minProbClip is not None:
            assert(pfratio_stitchpt is None and pfratio_derivpt is None), \
                "Cannot specify pfratio and minProbClip arguments as non-None!"
            self.min_p = minProbClip
            self.regtype = "minp"
        else:
            assert(minProbClip is None), "Cannot specify pfratio and minProbClip arguments as non-None!"
            self.x0 = pfratio_stitchpt
            self.x1 = pfratio_derivpt
            self.regtype = "pfratio"
        self.probClipInterval = probClipInterval

    def get_chi2k_distributed_qty(self, objective_function_value):
        return 2 * objective_function_value  # 2 * deltaLogL is what is chi2_k distributed

    def _terms_from_probs(self, extra=False):

        if self.regtype == 'pfratio':
            x0 = self.x0
            x1 = self.x1
            x = self.probs / self.freqs_nozeros  # objective is -Nf*log(x)

            pos_x = _np.where(x < x0, x0, x)
            S = self.minusCntVecMx * (1 / x1)  # deriv wrt x at x == x0 (=min_p)
            S2 = -0.5 * self.minusCntVecMx / (x1**2)  # 0.5 * 2nd deriv at x0

            v = self.minusCntVecMx * _np.log(pos_x)
            v = _np.where(x < x0, v + S * (x - x0) + S2 * (x - x0)**2, v)

        elif self.regtype == 'minp':
            pos_probs = _np.where(self.probs < self.min_p, self.min_p, self.probs)
            S = self.minusCntVecMx / self.min_p
            S2 = -0.5 * self.minusCntVecMx / (self.min_p**2)
            v = self.freqTerm + self.minusCntVecMx * _np.log(pos_probs)  # dims K x M (K = nSpamLabels, M = nCircuits)
            v = _np.where(self.probs < self.min_p,
                          v + S * (self.probs - self.min_p) + S2 * (self.probs - self.min_p)**2, v)
        else:
            raise ValueError("Invalid regularization type: %s" % self.regtype)

        v = _np.where(self.minusCntVecMx == 0, 0.0, v)

        #Note: no penalty for omitted probabilities (objective fn == 0 whenever counts == 0)

        v.shape = [self.KM]  # reshape ensuring no copy is needed
        if self.regtype == "pfratio":
            if extra:  # then used for jacobian where penalty terms are added later, so return now
                return v, (x, pos_x, S, S2)
        elif self.regtype == 'minp':
            if extra:  # then used for jacobian where penalty terms are added later, so return now
                return v, (pos_probs, S, S2)
        else:
            return v

    def _logl_terms(self, pvec):
        tm = _time.time()
        self.mdl.from_vector(pvec)
        self.mdl.bulk_fill_probs(self.probs, self.evTree, self.probClipInterval,
                                 self.check, self.comm)
        terms = self._terms_from_probs(tm)
        self.profiler.add_time(self.__class__.__name__ + " OBJECTIVE", tm)
        return terms

    def _logl_fn(self, pvec=None):
        if pvec is None: pvec = self.mdl.to_vector()
        return _np.sum(self._logl_terms(pvec))

    def _logl_jac_terms(self, pvec):
        tm = _time.time()
        dprobs = self.jac[0:self.KM, :]  # avoid mem copying: use jac mem for dprobs
        dprobs.shape = (self.KM, self.vec_gs_len)
        self.mdl.from_vector(pvec)
        self.mdl.bulk_fill_dprobs(dprobs, self.evTree,
                                  prMxToFill=self.probs, clipTo=self.probClipInterval,
                                  check=self.check, comm=self.comm, wrtBlockSize=self.wrtBlkSize,
                                  profiler=self.profiler, gatherMemLimit=self.gthrMem)

        if self.regtype == 'pfratio':
            terms, (x, pos_x, S, S2) = self._terms_from_probs(extra=True)
            x0 = self.x0
        elif self.regtype == 'minp':
            terms, (pos_probs, S, S2) = self._terms_from_probs(extra=True)

        # compute jacobian
        if self.regtype == 'pfratio':
            dprobs_factor_pos = (self.totalCntVec * (-1 / pos_x))  # objective = -Nf*log(x) => -Nf/x * (1/f) = -N/x
            dprobs_factor_neg = (S + 2 * S2 * (x - x0)) / self.freqs_nozeros
            dprobs_factor = _np.where(x < x0, dprobs_factor_neg, dprobs_factor_pos)

        elif self.regtype == 'minp':
            dprobs_factor_pos = self.minusCntVecMx / pos_probs  # objective = Nf*(log(f) - log(p))
            dprobs_factor_neg = S + 2 * S2 * (self.probs - self.min_p)
            dprobs_factor = _np.where(self.probs < self.min_p, dprobs_factor_neg, dprobs_factor_pos)

        dprobs_factor = _np.where(self.minusCntVecMx == 0, 0.0, dprobs_factor)
        dprobs *= dprobs_factor[:, None]  # (KM,N) * (KM,1)   (N = dim of vectorized model)
        #Note: this also sets jac[0:KM,:]

        self.profiler.add_time("logl JACOBIAN", tm)
        return self.jac

    def _logl_jac(self, pvec=None):
        if pvec is None: pvec = self.mdl.to_vector()
        return _np.sum(self._logl_jac_terms(pvec), axis=0)

    def _logl_hessian(self, pvec):
        assert(self.regtype == 'minp'), "Only regtype='minp' is supported for hessian calcs so far!"
        if pvec is None: pvec = self.mdl.to_vector()
        return self._construct_hessian(-self.minusCntVecMx, self.totalCntVec, self.probClipInterval)

    def _hessian_from_hprobs(self, hprobs, dprobs12, cntVecMx, totalCntVec, probs):
        """ Factored-out computation of hessian from raw components """
        # Notation:  (K=#spam, M=#strings, N=#wrtParams1, N'=#wrtParams2 )
        min_p = self.min_p
        pos_probs = _np.where(probs < self.min_p, self.min_p, probs)
        S = -cntVecMx / min_p  # slope term that is derivative of logl at min_p
        S2 = 0.5 * cntVecMx / (min_p**2)  # 2nd derivative of logl term at min_p

        # # (K,M,1,1) * (K,M,N,N')
        # hprobs_pos  = (-cntVecMx / pos_probs**2)[:,:,None,None] * dprobs12
        # # (K,M,1,1) * (K,M,N,N')
        # hprobs_pos += (cntVecMx / pos_probs)[:,:,None,None] * hprobs
        # # (K,M,1,1) * (K,M,N,N')
        # hprobs_neg  = (2*S2)[:,:,None,None] * dprobs12 + (S + 2*S2*(probs - min_p))[:,:,None,None] * hprobs
        # hessian = _np.where( (probs < min_p)[:,:,None,None], hprobs_neg, hprobs_pos)
        # # (K,M,N,N')
        # hessian = _np.where( (cntVecMx == 0)[:,:,None,None], 0.0, hessian)

        #Accomplish the same thing as the above commented-out lines,
        # but with more memory effiency:
        dprobs12_coeffs = \
            _np.where(probs < min_p, 2 * S2, cntVecMx / pos_probs**2)
        dprobs12_coeffs = _np.where(cntVecMx == 0, 0.0, dprobs12_coeffs)

        hprobs_coeffs = \
            _np.where(probs < min_p, S + 2 * S2 * (probs - min_p),
                      -cntVecMx / pos_probs)
        hprobs_coeffs = _np.where(cntVecMx == 0, 0.0, hprobs_coeffs)

        # hessian = hprobs_coeffs * hprobs + dprobs12_coeff * dprobs12
        #  but re-using dprobs12 and hprobs memory (which is overwritten!)
        hprobs *= hprobs_coeffs[:, None, None]
        dprobs12 *= dprobs12_coeffs[:, None, None]
        hessian = dprobs12; hessian += hprobs
        #Note: no need to correct for omitted probs (zero contribution)

        return _np.sum(hessian, axis=0)
        # sum over spam label and operation sequence dimensions (operation sequences in evalSubTree)
        # adds current subtree contribution for (N,N')-sized block of Hessian

    def _d2v_dp2(self):
        assert(self.regtype == "minp")
        pos_probs = _np.where(self.probs < self.min_p, self.min_p, self.probs)
        S2 = -0.5 * self.minusCntVecMx / (self.min_p**2)   # 2nd derivative of logl term at min_p
        d2v_dp2 = _np.where(self.probs < self.min_p,
                            2 * S2,
                            -self.minusCntVecMx / pos_probs**2)
        d2v_dp2 = _np.where(self.minusCntVecMx == 0, 0.0, d2v_dp2)
        return d2v_dp2  # a 1D array of the diagonal of d2(logl)/dprobs2; shape = (nEls,)


class MaxLogLFunction(ObjectiveFunction):
    def __init__(self, mdl, dataset, circuit_list, regularization=None, penalties=None,
                 cache=None, resource_alloc=None, name=None, verbosity=0, poisson_picture=True):
        super().__init__(mdl, dataset, circuit_list, regularization, penalties, cache, resource_alloc, name, verbosity)

        #Allocate peristent memory
        self.precompute_omitted_freqs()  # sets self.firsts
        self.cntVecMx, self.totalCntVec = self.compute_count_vectors()
        self.freqs = self.cntVecMx / self.totalCntVec
        # set zero freqs to 1.0 so np.log doesn't complain
        self.freqs_nozeros = _np.where(self.cntVecMx == 0, 1.0, self.freqs)
        self.poisson_picture = poisson_picture

        if poisson_picture:
            self.terms = self.cntVecMx * (_np.log(self.freqs_nozeros) - 1.0)
        else:
            self.terms = self.cntVecMx * _np.log(self.freqs_nozeros)
        self.terms[self.cntVecMx == 0] = 0.0
        # set 0 * log(0) terms explicitly to zero since numpy doesn't know this limiting behavior

        self.term_fn = self.max_logl_terms
        self.jfn = None
        self.hfn = None

    def max_logl_terms(self, pvec=None):
        return self.terms


class TimeDependentLogLFunctionPoissonPic(ObjectiveFunction):

    def __init__(self, mdl, dataset, circuit_list, regularization=None, penalties=None,
                 cache=None, resource_alloc=None, name=None, verbosity=0):

        super().__init__(mdl, dataset, circuit_list, regularization, penalties, cache, resource_alloc, name, verbosity)
        self.time_dependent = True

        #Allocate peristent memory
        self.v = _np.empty(self.KM, 'd')
        self.jac = _np.empty((self.KM + self.ex, self.vec_gs_len), 'd')

        self.num_total_outcomes = [mdl.get_num_outcomes(c) for c in self.circuitsToUse]  # for sparse data detection

        self.ls_fn = self.poisson_picture_logl
        self.ls_jfn = self.poisson_picture_jacobian

    def set_penalties(self, cptp_penalty_factor=0, spam_penalty_factor=0, forcefn_grad=None, shiftFctr=100):
        assert(cptp_penalty_factor == 0 and spam_penalty_factor == 0), \
            "Cannot apply CPTP or SPAM penalization in time-dependent logl case (yet)"
        assert(forcefn_grad is None), "forcing functions not supported with time-dependent logl function yet"

    def set_regularization(self, minProbClip=1e-4, probClipInterval=(-10000, 10000), radius=1e-4):
        self.min_p = minProbClip
        self.a = radius  # parameterizes "roundness" of f == 0 terms
        self.probClipInterval = probClipInterval

    def get_chi2k_distributed_qty(self, objective_function_value):
        return 2 * objective_function_value  # 2 * deltaLogL is what is chi2_k distributed

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

        self.ls_fn = self.logl_wildcard
        self.ls_jfn = None  # no jacobian yet

        #calling fn(...) initializes the members of self.logl_objfn
        self.probs = self.logl_objfn.probs.copy()

    def _default_evalpt(self):
        """The default point to evaluate functions at """
        return self.wildcard_budget.to_vector()

    def logl_wildcard(self, Wvec):
        tm = _time.time()
        self.wildcard_budget.from_vector(Wvec)
        self.wildcard_budget.update_probs(self.probs,
                                          self.logl_objfn.probs,
                                          self.logl_objfn.freqs,
                                          self.logl_objfn.circuitsToUse,
                                          self.logl_objfn.lookup,
                                          self.wildcard_budget_precomp)

        return self.logl_objfn._v_from_probs(tm)
