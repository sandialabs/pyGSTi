"""
Defines the OplessModel class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import collections as _collections

from .model import Model as _Model
from .labeldicts import OutcomeLabelDict as _OutcomeLabelDict
from .circuit import Circuit as _Circuit
from .polynomial import Polynomial as _Polynomial
from .resourceallocation import ResourceAllocation as _ResourceAllocation
from .successfailfwdsim import SuccessFailForwardSimulator as _SuccessFailForwardSimulator
from ..tools import slicetools as _slct

from .opcalc import compact_deriv as _compact_deriv, float_product as prod, \
    bulk_eval_compact_polynomials as _bulk_eval_compact_polynomials

#REMOVE
#class OplessModelTree(_EvalTree):
#    """
#    An evaluation tree for an :class:`OplessModel`.
#
#    Parameters
#    ----------
#    circuit_list : list
#        A list of the circuits to compute values for.
#
#    lookup : collections.OrderedDict
#        A dictionary whose keys are integer indices into `circuit_list` and
#        whose values are slices and/or integer-arrays into the space/axis of
#        final elements returned by the 'bulk fill' routines.
#
#    outcome_lookup : collections.OrderedDict
#        A dictionary whose keys are integer indices into `circuit_list` and
#        whose values are lists of outcome labels (an outcome label is a tuple
#        of POVM-effect and/or instrument-element labels).
#
#    cache : dict, optional
#        A dictionary for holding cached intermediate results.
#    """
#    def __init__(self, circuit_list, lookup, outcome_lookup, cache=None):
#        _EvalTree.__init__(self, circuit_list)
#        self.element_indices = lookup
#        self.outcomes = outcome_lookup
#        self.num_final_strs = len(circuit_list)  # circuits
#        max_el_index = -1
#        for elIndices in lookup.values():
#            max_i = elIndices.stop - 1 if isinstance(elIndices, slice) else max(elIndices)
#            max_el_index = max(max_el_index, max_i)
#        self.num_final_els = max_el_index + 1
#        self.cache = cache


class OplessModel(_Model):
    """
    A model that does *not* have independent component operations.

    :class:`OplessModel`-derived classes often implement coarser models that
    predict the success or outcome probability of a circuit based on simple
    properties of the circuit and not detailed gate-level modeling.

    Parameters
    ----------
    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.
    """

    def __init__(self, state_space_labels):
        """
        Creates a new Model.  Rarely used except from derived classes
        `__init__` functions.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels or list or tuple
            The decomposition (with labels) of (pure) state-space this model
            acts upon.  Regardless of whether the model contains operators or
            superoperators, this argument describes the Hilbert space dimension
            and imposed structure.  If a list or tuple is given, it must be
            of a from that can be passed to `StateSpaceLabels.__init__`.
        """
        _Model.__init__(self, state_space_labels)

        #Setting things the rest of pyGSTi expects but probably shouldn't...
        self.basis = None

    @property
    def dim(self):
        return 0

    def circuit_outcomes(self, circuit):  # needed for sparse data detection
        """
        Get all the possible outcome labels produced by simulating this circuit.

        Parameters
        ----------
        circuit : Circuit
            Circuit to get outcomes of.

        Returns
        -------
        tuple
        """
        raise NotImplementedError("Derived classes should implement this!")

    def probabilities(self, circuit, outcomes=None, time=None):
        """
        Construct a dictionary containing the outcome probabilities of `circuit`.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        outcomes : list or tuple
            A sequence of outcomes, which can themselves be either tuples
            (to include intermediate measurements) or simple strings, e.g. `'010'`.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        Returns
        -------
        probs : OutcomeLabelDict
            A dictionary with keys equal to outcome labels and
            values equal to probabilities.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def bulk_probabilities(self, circuits, clip_to=None, comm=None, mem_limit=None, smartc=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : (list of Circuits) or CircuitOutcomeProbabilityArrayLayout
            When a list, each element specifies a circuit to compute outcome probabilities for.
            A :class:`CircuitOutcomeProbabilityArrayLayout` specifies the circuits along with
            an internal memory layout that reduces the time required by this function and can
            restrict the computed probabilities to those corresponding to only certain outcomes.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of evalTree (if it is split).

        mem_limit : int, optional
            A rough memory limit in bytes which is used to determine processor
            allocation.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(outcome, p)` tuples, where `outcome` is a tuple of labels
            and `p` is the corresponding probability.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def __str__(self):
        raise NotImplementedError("Derived classes should implement OplessModel.__str__ !!")

    #TODO REMOVE
    #def dprobs(self, circuit, return_pr=False, clip_to=None):
    #    """
    #    Construct a dictionary of outcome-probability derivatives for `circuit`.
    #
    #    Parameters
    #    ----------
    #    circuit : Circuit or tuple of operation labels
    #        The sequence of operation labels specifying the circuit.
    #
    #    return_pr : bool, optional
    #        when set to True, additionally return the probabilities.
    #
    #    clip_to : 2-tuple, optional
    #        (min,max) to clip returned probability to if not None.
    #        Only relevant when return_pr == True.
    #
    #    Returns
    #    -------
    #    dprobs : dictionary
    #        A dictionary of outcome-probability derivatives, or `(derivative, probability)`
    #        tuples if `return_pr=True`.
    #    """
    #    eps = 1e-7
    #    orig_pvec = self.to_vector()
    #    Np = self.num_params()
    #    probs0 = self.probabilities(circuit, clip_to, None)
    #
    #    deriv = {k: _np.empty(Np, 'd') for k in probs0.keys()}
    #    for i in range(Np):
    #        p_plus_dp = orig_pvec.copy()
    #        p_plus_dp[i] += eps
    #        self.from_vector(p_plus_dp)
    #        probs1 = self.probabilities(circuit, clip_to, None)
    #        for k, p0 in probs0.items():
    #            deriv[k][i] = (probs1[k] - p0) / eps
    #    self.from_vector(orig_pvec)
    #
    #    if return_pr:
    #        return {k: (p0, deriv[k]) for k in probs0.keys()}
    #    else:
    #        return deriv

#    def bulk_evaltree_from_resources(self, circuit_list, comm=None, mem_limit=None,
#                                     distribute_method="default", subcalls=[],
#                                     dataset=None, verbosity=0):
#        """
#        Create an evaluation tree based on available memory and CPUs.
#
#        This tree can be used by other Bulk_* functions, and is it's own
#        function so that for many calls to Bulk_* made with the same
#        circuit_list, only a single call to bulk_evaltree is needed.
#
#        Parameters
#        ----------
#        circuit_list : list of (tuples or Circuits)
#            Each element specifies a circuit to include in the evaluation tree.
#
#        comm : mpi4py.MPI.Comm
#            When not None, an MPI communicator for distributing computations
#            across multiple processors.
#
#        mem_limit : int, optional
#            A rough memory limit in bytes which is used to determine subtree
#            number and size.
#
#        distribute_method : {"circuits", "deriv"}
#            How to distribute calculation amongst processors (only has effect
#            when comm is not None).  "circuits" will divide the list of
#            circuits and thereby result in more subtrees; "deriv" will divide
#            the columns of any jacobian matrices, thereby resulting in fewer
#            (larger) subtrees.
#
#        subcalls : list, optional
#            A list of the names of the Model functions that will be called
#            using the returned evaluation tree, which are necessary for
#            estimating memory usage (for comparison to mem_limit).  If
#            mem_limit is None, then there's no need to specify `subcalls`.
#
#        dataset : DataSet, optional
#            If not None, restrict what is computed to only those
#            probabilities corresponding to non-zero counts (observed
#            outcomes) in this data set.
#
#        verbosity : int, optional
#            How much detail to send to stdout.
#
#        Returns
#        -------
#        evt : EvalTree
#            The evaluation tree object, split as necesary.
#
#        paramBlockSize1 : int or None
#            The maximum size of 1st-deriv-dimension parameter blocks
#            (i.e. the maximum number of parameters to compute at once
#             in calls to dprobs, etc., usually specified as wrt_block_size
#             or wrt_block_size1).
#
#        paramBlockSize2 : int or None
#            The maximum size of 2nd-deriv-dimension parameter blocks
#            (i.e. the maximum number of parameters to compute at once
#             in calls to hprobs, etc., usually specified as wrt_block_size2).
#
#        elIndices : collections.OrderedDict
#            A dictionary whose keys are integer indices into `circuit_list` and
#            whose values are slices and/or integer-arrays into the space/axis of
#            final elements returned by the 'bulk fill' routines.  Thus, to get the
#            final elements corresponding to `circuits[i]`, use
#            `filledArray[ elIndices[i] ]`.
#
#        outcomes : collections.OrderedDict
#            A dictionary whose keys are integer indices into `circuit_list` and
#            whose values are lists of outcome labels (an outcome label is a tuple
#            of POVM-effect and/or instrument-element labels).  Thus, to obtain
#            what outcomes the i-th circuit's final elements
#            (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.
#        """
#        #TODO: choose these based on resources, and enable split trees
#        minSubtrees = 0
#        numSubtreeComms = 1
#        maxTreeSize = None
#        evTree = self.bulk_evaltree(circuit_list, minSubtrees, maxTreeSize,
#                                    numSubtreeComms, dataset, verbosity)
#        return evTree, 0, 0, evTree.element_indices, evTree.outcomes
#
#    def bulk_evaltree(self, circuit_list, min_subtrees=None, max_tree_size=None,
#                      num_subtree_comms=1, dataset=None, verbosity=0):
#        """
#        Create an evaluation tree for all the circuits in `circuit_list`.
#
#        This tree can be used by other Bulk_* functions, and is it's own
#        function so that for many calls to Bulk_* made with the same
#        circuit_list, only a single call to bulk_evaltree is needed.
#
#        Parameters
#        ----------
#        circuit_list : list of (tuples or Circuits)
#            Each element specifies a circuit to include in the evaluation tree.
#
#        min_subtrees : int , optional
#            The minimum number of subtrees the resulting EvalTree must have.
#
#        max_tree_size : int , optional
#            The maximum size allowed for the single un-split tree or any of
#            its subtrees.
#
#        num_subtree_comms : int, optional
#            The number of processor groups (communicators)
#            to divide the subtrees of the EvalTree among
#            when calling its `distribute` method.
#
#        dataset : DataSet, optional
#            If not None, restrict what is computed to only those
#            probabilities corresponding to non-zero counts (observed
#            outcomes) in this data set.
#
#        verbosity : int, optional
#            How much detail to send to stdout.
#
#        Returns
#        -------
#        evt : EvalTree
#            An evaluation tree object.
#
#        elIndices : collections.OrderedDict
#            A dictionary whose keys are integer indices into `circuit_list` and
#            whose values are slices and/or integer-arrays into the space/axis of
#            final elements returned by the 'bulk fill' routines.  Thus, to get the
#            final elements corresponding to `circuits[i]`, use
#            `filledArray[ elIndices[i] ]`.
#
#        outcomes : collections.OrderedDict
#            A dictionary whose keys are integer indices into `circuit_list` and
#            whose values are lists of outcome labels (an outcome label is a tuple
#            of POVM-effect and/or instrument-element labels).  Thus, to obtain
#            what outcomes the i-th circuit's final elements
#            (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.
#        """
#        raise NotImplementedError("Derived classes should implement this!")


#    def bulk_dprobs(self, circuit_list, return_pr=False, clip_to=None,
#                    check=False, comm=None, wrt_block_size=None, dataset=None):
#        """
#        Construct a dictionary containing the probability-derivatives for an entire list of circuits.
#
#        Parameters
#        ----------
#        circuit_list : list of (tuples or Circuits)
#            Each element specifies a circuit to compute quantities for.
#
#        return_pr : bool, optional
#            when set to True, additionally return the probabilities.
#
#        clip_to : 2-tuple, optional
#            (min,max) to clip returned probability to if not None.
#            Only relevant when return_pr == True.
#
#        check : boolean, optional
#            If True, perform extra checks within code to verify correctness,
#            generating warnings when checks fail.  Used for testing, and runs
#            much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not None, an MPI communicator for distributing the computation
#            across multiple processors.  Distribution is first performed over
#            subtrees of evalTree (if it is split), and then over blocks (subsets)
#            of the parameters being differentiated with respect to (see
#            wrt_block_size).
#
#        wrt_block_size : int or float, optional
#            The maximum average number of derivative columns to compute *products*
#            for simultaneously.  None means compute all columns at once.
#            The minimum of wrt_block_size and the size that makes maximal
#            use of available processors is used as the final block size. Use
#            this argument to reduce amount of intermediate memory required.
#
#        dataset : DataSet, optional
#            If not None, restrict what is computed to only those
#            probabilities corresponding to non-zero counts (observed
#            outcomes) in this data set.
#
#        Returns
#        -------
#        dprobs : dictionary
#            A dictionary such that `probs[opstr]` is an ordered dictionary of
#            `(outcome, dp, p)` tuples, where `outcome` is a tuple of labels,
#            `p` is the corresponding probability, and `dp` is an array containing
#            the derivative of `p` with respect to each parameter.  If `return_pr`
#            if False, then `p` is not included in the tuples (so they're just
#            `(outcome, dp)`).>
#        """
#        memLimit = None
#        evalTree, _, _, elIndices, outcomes = self.bulk_evaltree_from_resources(circuit_list, comm, memLimit,
#                                                                                "default", [], dataset)
#        nElements = evalTree.num_final_elements()
#        nDerivCols = self.num_params()
#
#        vdp = _np.empty((nElements, nDerivCols), 'd')
#        vp = _np.empty(nElements, 'd') if return_pr else None
#
#        self.bulk_fill_dprobs(vdp, evalTree,
#                              vp, clip_to, check, comm,
#                              None, wrt_block_size)
#
#        ret = _collections.OrderedDict()
#        for i, opstr in enumerate(evalTree):
#            elInds = _slct.indices(elIndices[i]) \
#                if isinstance(elIndices[i], slice) else elIndices[i]
#            if return_pr:
#                ret[opstr] = _OutcomeLabelDict(
#                    [(outLbl, (vdp[ei], vp[ei])) for ei, outLbl in zip(elInds, outcomes[i])])
#            else:
#                ret[opstr] = _OutcomeLabelDict(
#                    [(outLbl, vdp[ei]) for ei, outLbl in zip(elInds, outcomes[i])])
#        return ret


class SuccessFailModel(OplessModel):
    """
    An op-less model that always outputs 2 (success & failure) probabilities for each circuit.

    Parameters
    ----------
    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    use_cache : bool, optional
        Whether a cache should be used to increase performance.
    """
    def __init__(self, state_space_labels, use_cache=False):
        OplessModel.__init__(self, state_space_labels)
        self.use_cache = use_cache
        self.sim = _SuccessFailForwardSimulator(self)

    def _post_copy(self, copy_into):
        """
        Called after all other copying is done, to perform "linking" between
        the new model (`copy_into`) and its members.
        """
        copy_into.sim.model = copy_into  # set copy's `.model` link

    def circuit_outcomes(self, circuit):  # needed for sparse data detection
        """
        Get all the possible outcome labels produced by simulating this circuit.

        Parameters
        ----------
        circuit : Circuit
            Circuit to get outcomes of.

        Returns
        -------
        tuple
        """
        return (('success',), ('fail',))

    def _success_prob(self, circuit, cache):
        raise NotImplementedError("Derived classes should implement this!")

    def _success_dprob(self, circuit, param_slice, cache):
        """ Derived classes can override this.  Default implemntation is to use finite difference. """
        eps = 1e-7
        orig_pvec = self.to_vector()
        wrtIndices = _slct.indices(param_slice) if (param_slice is not None) else list(range(self.num_params()))
        sp0 = self._success_prob(circuit, cache)

        deriv = _np.empty(len(wrtIndices), 'd')
        for i in wrtIndices:
            p_plus_dp = orig_pvec.copy()
            p_plus_dp[i] += eps
            self.from_vector(p_plus_dp)
            sp1 = self._success_prob(circuit, cache)
            deriv[i] = (sp1 - sp0) / eps
        self.from_vector(orig_pvec)
        return deriv

    def probabilities(self, circuit, outcomes=None, time=None):
        """
        Construct a dictionary containing the outcome probabilities of `circuit`.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        outcomes : list or tuple
            A sequence of outcomes, which can themselves be either tuples
            (to include intermediate measurements) or simple strings, e.g. `'010'`.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        Returns
        -------
        probs : OutcomeLabelDict
            A dictionary with keys equal to outcome labels and
            values equal to probabilities.
        """
        return self._sim.probs(circuit, outcomes, time)

    def bulk_probabilities(self, circuits, clip_to=None, comm=None, mem_limit=None, smartc=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : (list of Circuits) or CircuitOutcomeProbabilityArrayLayout
            When a list, each element specifies a circuit to compute outcome probabilities for.
            A :class:`CircuitOutcomeProbabilityArrayLayout` specifies the circuits along with
            an internal memory layout that reduces the time required by this function and can
            restrict the computed probabilities to those corresponding to only certain outcomes.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of evalTree (if it is split).

        mem_limit : int, optional
            A rough memory limit in bytes which is used to determine processor
            allocation.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(outcome, p)` tuples, where `outcome` is a tuple of labels
            and `p` is the corresponding probability.
        """
        resource_alloc = _ResourceAllocation(comm, mem_limit)
        return self.sim.bulk_probs(circuits, clip_to, resource_alloc, smartc)

#    def polynomial_probabilities(self, circuit):
#        """
#        Construct a dictionary of the outcome probabilities of `circuit` as *polynomials*.
#
#        Parameters
#        ----------
#        circuit : Circuit or tuple of operation labels
#            The sequence of operation labels specifying the circuit.
#
#        Returns
#        -------
#        probs : dictionary
#            A dictionary containing probabilities as polynomials.
#        """
#        sp = self._success_prob_polynomial(circuit)
#        return _OutcomeLabelDict([('success', sp), ('fail', _Polynomial({(): 1.0}) - sp)])


#REMOVE
#    def dprobs(self, circuit, return_pr=False, clip_to=None, cache=None):
#        """
#        Construct a dictionary of outcome-probability derivatives for `circuit`.
#
#        Parameters
#        ----------
#        circuit : Circuit or tuple of operation labels
#            The sequence of operation labels specifying the circuit.
#
#        return_pr : bool, optional
#            when set to True, additionally return the probabilities.
#
#        clip_to : 2-tuple, optional
#            (min,max) to clip returned probability to if not None.
#            Only relevant when return_pr == True.
#
#        cache : dict, optional
#            A cache for increasing performance.
#
#        Returns
#        -------
#        dprobs : dictionary
#            A dictionary of outcome-probability derivatives, or `(derivative, probability)`
#            tuples if `return_pr=True`.
#        """
#        try:
#            dsp = self._success_dprob(circuit, cache)
#        except NotImplementedError:
#            return OplessModel.dprobs(self, circuit, return_pr, clip_to)
#
#        if return_pr:
#            sp = self._success_prob(circuit, cache)
#            if clip_to is not None: sp = _np.clip(sp, clip_to[0], clip_to[1])
#            return {('success',): (sp, dsp), ('fail',): (1 - sp, -dsp)}
#        else:
#            return {('success',): dsp, ('fail',): -dsp}


#    def simplify_circuits(self, circuits, dataset=None):
#        """
#        Simplifies a list of :class:`Circuit`s.
#
#        Parameters
#        ----------
#        circuits : list of Circuits
#            The list to simplify.
#
#        dataset : DataSet, optional
#            If not None, restrict what is simplified to only those
#            probabilities corresponding to non-zero counts (observed
#            outcomes) in this data set.
#
#        Returns
#        -------
#        raw_elabels_dict : collections.OrderedDict
#            A dictionary whose keys are simplified circuits (containing just
#            "simplified" gates, i.e. not instruments) that include preparation
#            labels but no measurement (POVM). Values are lists of simplified
#            effect labels, each label corresponds to a single "final element" of
#            the computation, e.g. a probability.  The ordering is important - and
#            is why this needs to be an ordered dictionary - when the lists of tuples
#            are concatenated (by key) the resulting tuple orderings corresponds to
#            the final-element axis of an output array that is being filled (computed).
#        elIndices : collections.OrderedDict
#            A dictionary whose keys are integer indices into `circuits` and
#            whose values are slices and/or integer-arrays into the space/axis of
#            final elements.  Thus, to get the final elements corresponding to
#            `circuits[i]`, use `filledArray[ elIndices[i] ]`.
#        outcomes : collections.OrderedDict
#            A dictionary whose keys are integer indices into `circuits` and
#            whose values are lists of outcome labels (an outcome label is a tuple
#            of POVM-effect and/or instrument-element labels).  Thus, to obtain
#            what outcomes the i-th circuit's final elements
#            (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.
#        nTotElements : int
#            The total number of "final elements" - this is how big of an array
#            is need to hold all of the probabilities `circuits` generates.
#        """
#        rawdict = None  # TODO - is this needed?
#        lookup = {i: slice(2 * i, 2 * i + 2, 1) for i in range(len(circuits))}
#        outcome_lookup = {i: (('success',), ('fail',)) for i in range(len(circuits))}
#
#        return rawdict, lookup, outcome_lookup, 2 * len(circuits)
#
#    def bulk_evaltree(self, circuit_list, min_subtrees=None, max_tree_size=None,
#                      num_subtree_comms=1, dataset=None, verbosity=0):
#        """
#        Create an evaluation tree for all the circuits in `circuit_list`.
#
#        This tree can be used by other Bulk_* functions, and is it's own
#        function so that for many calls to Bulk_* made with the same
#        circuit_list, only a single call to bulk_evaltree is needed.
#
#        Parameters
#        ----------
#        circuit_list : list of (tuples or Circuits)
#            Each element specifies a circuit to include in the evaluation tree.
#
#        min_subtrees : int , optional
#            The minimum number of subtrees the resulting EvalTree must have.
#
#        max_tree_size : int , optional
#            The maximum size allowed for the single un-split tree or any of
#            its subtrees.
#
#        num_subtree_comms : int, optional
#            The number of processor groups (communicators)
#            to divide the subtrees of the EvalTree among
#            when calling its `distribute` method.
#
#        dataset : DataSet, optional
#            If not None, restrict what is computed to only those
#            probabilities corresponding to non-zero counts (observed
#            outcomes) in this data set.
#
#        verbosity : int, optional
#            How much detail to send to stdout.
#
#        Returns
#        -------
#        evt : EvalTree
#            An evaluation tree object.
#
#        elIndices : collections.OrderedDict
#            A dictionary whose keys are integer indices into `circuit_list` and
#            whose values are slices and/or integer-arrays into the space/axis of
#            final elements returned by the 'bulk fill' routines.  Thus, to get the
#            final elements corresponding to `circuits[i]`, use
#            `filledArray[ elIndices[i] ]`.
#
#        outcomes : collections.OrderedDict
#            A dictionary whose keys are integer indices into `circuit_list` and
#            whose values are lists of outcome labels (an outcome label is a tuple
#            of POVM-effect and/or instrument-element labels).  Thus, to obtain
#            what outcomes the i-th circuit's final elements
#            (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.
#        """
#        lookup = {i: slice(2 * i, 2 * i + 2, 1) for i in range(len(circuit_list))}
#        outcome_lookup = {i: (('success',), ('fail',)) for i in range(len(circuit_list))}
#
#        if self.use_cache == "poly":
#            #Do precomputation here
#            polys = []
#            for i, circuit in enumerate(circuit_list):
#                print("Generating probs for circuit %d of %d" % (i + 1, len(circuit_list)))
#                probs = self.polynomial_probs(circuit)
#                polys.append(probs['success'])
#                polys.append(probs['fail'])
#            compact_polys = compact_polynomial_list(polys)
#            cache = compact_polys
#        elif self.use_cache is True:
#            cache = [self._circuit_cache(circuit) for circuit in circuit_list]
#        else:
#            cache = None
#
#        return OplessModelTree(circuit_list, lookup, outcome_lookup, cache)


class ErrorRatesModel(SuccessFailModel):

    """
    A success-fail model based on per-gate error rates.

    Parameters
    ----------
    error_rates : dict
        A dictionary with "gates" and "readout" keys, each of which corresponds to a
        dictionary of error-rates for gates or SPAM elements, respectively.

    n_qubits : int
        The number of qubits in the model.

    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    alias_dict : dict, optional
        An alias dictionary mapping the gate labels in circuits to the
        keys of a (nested) `error_rates` dictionary.  This allows, for instance,
        many gates' error rates to be set by the same model parameter.

    idlename : str, optional
        The gate name to be used for the 1-qubit idle gate (this should be
        set in `error_rates` to add idle errors.
    """
    def __init__(self, error_rates, n_qubits, state_space_labels=None, alias_dict={}, idlename='Gi'):
        if state_space_labels is None:
            state_space_labels = ['Q%d' % i for i in range(n_qubits)]
        else:
            assert(len(state_space_labels) == n_qubits)

        SuccessFailModel.__init__(self, state_space_labels, use_cache=True)

        gate_error_rate_keys = (list(error_rates['gates'].keys()))
        readout_error_rate_keys = (list(error_rates['readout'].keys()))

        # if gate_error_rate_keys[0] in state_space_labels:
        #     self._gateind = True
        # else:
        #     self._gateind = False

        self._idlename = idlename
        self._alias_dict = alias_dict.copy()
        self._gate_error_rate_indices = {k: i for i, k in enumerate(gate_error_rate_keys)}
        self._readout_error_rate_indices = {k: i + len(gate_error_rate_keys)
                                            for i, k in enumerate(readout_error_rate_keys)}
        self._paramvec = _np.concatenate(
            (_np.array([_np.sqrt(error_rates['gates'][k]) for k in gate_error_rate_keys], 'd'),
             _np.array([_np.sqrt(error_rates['readout'][k]) for k in readout_error_rate_keys], 'd'))
        )

    def __str__(self):
        s = "Error Rates model with error rates: \n" + \
            "\n".join(["%s = %g" % (k, self._paramvec[i]**2) for k, i in self._gate_error_rate_indices.items()]) + \
            "\n" + \
            "\n".join(["%s = %g" % (k, self._paramvec[i]**2) for k, i in self._readout_error_rate_indices.items()])
        return s

    def to_dict(self):
        """
        Convert this model to a dictionary (for debugging or easy printing).
        """
        error_rate_dict = {'gates': {}, 'readout': {}}
        error_rate_dict['gates'] = {k: self._paramvec[i]**2 for k, i in self._gate_error_rate_indices.items()}
        error_rate_dict['readout'] = {k: self._paramvec[i]**2 for k, i in self._readout_error_rate_indices.items()}
        asdict = {'error_rates': error_rate_dict, 'alias_dict': self._alias_dict.copy()}
        return asdict

    def _circuit_cache(self, circuit):
        if not isinstance(circuit, _Circuit):
            circuit = _Circuit.from_tuple(circuit)

        depth = circuit.depth
        width = circuit.width
        g_inds = self._gate_error_rate_indices
        r_inds = self._readout_error_rate_indices

        # if self._gateind:
        #     inds_to_mult_by_layer = []
        #     for i in range(depth):

        #         layer = circuit.get_layer(i)
        #         inds_to_mult = []
        #         usedQs = []

        #         for gate in layer:
        #             if len(gate.qubits) > 1:
        #                 usedQs += list(gate.qubits)
        #                 inds_to_mult.append(g_inds[frozenset(gate.qubits)])

        #         for q in circuit.line_labels:
        #             if q not in usedQs:
        #                 inds_to_mult.append(g_inds[q])

        #         inds_to_mult_by_layer.append(_np.array(inds_to_mult, int))

        # else:
        layers_with_idles = [circuit.layer_with_idles(i, idle_gate_name=self._idlename) for i in range(depth)]
        inds_to_mult_by_layer = [_np.array([g_inds[self._alias_dict.get(str(gate), str(gate))] for gate in layer], int)
                                 for layer in layers_with_idles]

        # Bit-flip readout error as a pre-measurement depolarizing channel.
        inds_to_mult = [r_inds[q] for q in circuit.line_labels]
        inds_to_mult_by_layer.append(_np.array(inds_to_mult, int))

        # The scaling constant such that lambda = 1 - alpha * epsilon where lambda is the diagonal of a depolarizing
        # channel with entanglement infidelity of epsilon.
        alpha = 4**width / (4**width - 1)

        return (width, depth, alpha, 1 / 2**width, inds_to_mult_by_layer)


class TwirledLayersModel(ErrorRatesModel):
    """
    A model where twirled-layer error rates are computed and multiplied together to compute success probabilities.

    In this model, the success probability of a circuit is the product of
    `1.0 - alpha * pfail` terms, one per layer of the circuit (including idles).
    The `pfail` of a circuit layer is given as `1 - prod(1 - error_rate_i)`, where
    `i` ranges over the gates in the layer.  `alpha` is the constant `4^w / (4^w - 1)`
    where `w` is the circuit width.

    Parameters
    ----------
    error_rates : dict
        A dictionary with "gates" and "readout" keys, each of which corresponds to a
        dictionary of error-rates for gates or SPAM elements, respectively.

    n_qubits : int
        The number of qubits in the model.

    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    alias_dict : dict, optional
        An alias dictionary mapping the gate labels in circuits to the
        keys of a (nested) `error_rates` dictionary.  This allows, for instance,
        many gates' error rates to be set by the same model parameter.

    idlename : str, optional
        The gate name to be used for the 1-qubit idle gate (this should be
        set in `error_rates` to add idle errors.
    """
    def __init__(self, error_rates, n_qubits, state_space_labels=None, alias_dict={}, idlename='Gi'):
        ErrorRatesModel.__init__(self, error_rates, n_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idlename=idlename)

    def _success_prob(self, circuit, cache):
        pvec = self._paramvec**2
        if cache is None:
            cache = self._circuit_cache(circuit)

        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = cache
        # The success probability for all the operations (the entanglment fidelity for the gates)
        sp = 1.0 - pvec

        # The depolarizing constant for the full sequence of twirled layers.
        lambda_all_layers = 1.0
        for inds_to_mult in inds_to_mult_by_layer[:-1]:
            lambda_all_layers *= 1 - alpha * (1 - prod(sp[inds_to_mult]))
        # lambda_all_layers = prod([(1 - alpha * (1 - prod(sp[inds_to_mult])))
        #                           for inds_to_mult in inds_to_mult_by_layer[:-1]])

        # The readout success probability.
        successprob_readout = prod(sp[inds_to_mult_by_layer[-1]])
        # THe success probability of the circuit.
        successprob_circuit = lambda_all_layers * (successprob_readout - one_over_2_width) + one_over_2_width

        return successprob_circuit

    def _success_dprob(self, circuit, param_slice, cache):
        assert(param_slice is None or _slct.length(param_slice) == len(self._paramvec)), \
            "No support for derivatives with respect to a subset of model parameters yet!"
        pvec = self._paramvec**2
        dpvec_dparams = 2 * self._paramvec

        if cache is None:
            cache = self._circuit_cache(circuit)

        # p = product_layers(1 - alpha * (1 - prod_[inds4layer](1 - param))) * \
        #     (prod_[inds4LASTlayer](1 - param) - 1 / 2**width)
        # Note: indices cannot be repeated in a layer, i.e. either a given index appears one or zero times in inds4layer

        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = cache
        sp = 1.0 - pvec
        deriv = _np.zeros(len(pvec), 'd')

        nLayers = len(inds_to_mult_by_layer)
        lambda_per_layer = _np.empty(nLayers, 'd')
        for i, inds_to_mult in enumerate(inds_to_mult_by_layer[:-1]):
            lambda_per_layer[i] = 1 - alpha * (1 - prod(sp[inds_to_mult]))

        successprob_readout = prod(sp[inds_to_mult_by_layer[-1]])
        lambda_per_layer[nLayers - 1] = successprob_readout - one_over_2_width
        lambda_all_layers = prod(lambda_per_layer)  # includes readout factor as last layer

        #All layers except last
        for i, inds_to_mult in enumerate(inds_to_mult_by_layer[:-1]):
            lambda_all_but_current_layer = lambda_all_layers / lambda_per_layer[i]
            # for each such ind, when we take deriv wrt this index, we need to differentiate this layer, etc.
            for ind in inds_to_mult:
                deriv[ind] += lambda_all_but_current_layer * alpha * \
                    (prod(sp[inds_to_mult]) / sp[ind]) * -1.0  # what if sp[ind] == 0?

        #Last layer
        lambda_all_but_current_layer = lambda_all_layers / lambda_per_layer[-1]
        for ind in inds_to_mult_by_layer[-1]:
            deriv[ind] += lambda_all_but_current_layer * (successprob_readout / sp[ind]) * -1.0  # what if sp[ind] == 0?

        return deriv * dpvec_dparams


class TwirledGatesModel(ErrorRatesModel):
    """
    A model where twirled-gate error rates are computed and multiplied together to compute success probabilities.

    In this model, the success probability of a circuit is the product of
    `1.0 - alpha * pfail` terms, one per gate of the circuit (including idles).
    The `pfail` of a gate is given as `1 - error_rate`, and `alpha` is the constant
    `4^w / (4^w - 1)` where `w` is the circuit width.

    Parameters
    ----------
    error_rates : dict
        A dictionary with "gates" and "readout" keys, each of which corresponds to a
        dictionary of error-rates for gates or SPAM elements, respectively.

    n_qubits : int
        The number of qubits in the model.

    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    alias_dict : dict, optional
        An alias dictionary mapping the gate labels in circuits to the
        keys of a (nested) `error_rates` dictionary.  This allows, for instance,
        many gates' error rates to be set by the same model parameter.

    idlename : str, optional
        The gate name to be used for the 1-qubit idle gate (this should be
        set in `error_rates` to add idle errors.
    """
    def __init__(self, error_rates, n_qubits, state_space_labels=None, alias_dict={}, idlename='Gi'):
        """
        todo
        """
        ErrorRatesModel.__init__(self, error_rates, n_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idlename=idlename)

    def _circuit_cache(self, circuit):
        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = super()._circuit_cache(circuit)
        all_inds_to_mult = _np.concatenate(inds_to_mult_by_layer[:-1])
        readout_inds_to_mult = inds_to_mult_by_layer[-1]
        all_inds_to_mult_cnt = _np.zeros(self.num_params(), int)
        for i in all_inds_to_mult:
            all_inds_to_mult_cnt[i] += 1
        return width, depth, alpha, one_over_2_width, all_inds_to_mult, readout_inds_to_mult, all_inds_to_mult_cnt

    def _success_prob(self, circuit, cache):
        """
        todo
        """
        pvec = self._paramvec**2
        if cache is None:
            cache = self._circuit_cache(circuit)

        width, depth, alpha, one_over_2_width, all_inds_to_mult, readout_inds_to_mult, all_inds_to_mult_cnt = cache
        # The success probability for all the operations (the entanglment fidelity for the gates)
        sp = 1.0 - pvec

        # The 'lambda' for all gates (+ readout, which isn't used).
        lambda_ops = 1.0 - alpha * pvec

        # The depolarizing constant for the full sequence of twirled gates.
        lambda_all_layers = prod(lambda_ops[all_inds_to_mult])
        # The readout success probability.
        successprob_readout = prod(sp[readout_inds_to_mult])
        # THe success probability of the circuit.
        successprob_circuit = lambda_all_layers * (successprob_readout - one_over_2_width) + one_over_2_width

        return successprob_circuit

    def _success_dprob(self, circuit, param_slice, cache):
        """
        todo
        """
        assert(param_slice is None or _slct.length(param_slice) == len(self._paramvec)), \
            "No support for derivatives with respect to a subset of model parameters yet!"
        pvec = self._paramvec**2
        dpvec_dparams = 2 * self._paramvec

        if cache is None:
            cache = self._circuit_cache(circuit)

        width, depth, alpha, one_over_2_width, all_inds_to_mult, readout_inds_to_mult, all_inds_to_mult_cnt = cache
        sp = 1.0 - pvec
        lambda_ops = 1.0 - alpha * pvec
        deriv = _np.zeros(len(pvec), 'd')

        # The depolarizing constant for the full sequence of twirled gates.
        lambda_all_layers = prod(lambda_ops[all_inds_to_mult])
        for i, n in enumerate(all_inds_to_mult_cnt):
            deriv[i] = n * lambda_all_layers / lambda_ops[i] * -alpha  # -alpha = d(lambda_ops/dparam)

        # The readout success probability.
        readout_deriv = _np.zeros(len(pvec), 'd')
        successprob_readout = prod(sp[readout_inds_to_mult])
        for ind in readout_inds_to_mult:
            readout_deriv[ind] = (successprob_readout / sp[ind]) * -1.0  # what if sp[ind] == 0?

        # The success probability of the circuit.
        #successprob_circuit = lambda_all_layers * (successprob_readout - one_over_2_width) + one_over_2_width

        # product rule
        return (deriv * (successprob_readout - one_over_2_width) + lambda_all_layers * readout_deriv) * dpvec_dparams


class AnyErrorCausesFailureModel(ErrorRatesModel):
    """
    A model where any gate failure causes a circuit failure.

    Specifically, the success probability of a circuit is give by
    `1 - prod(1 - error_rate_i)` where `i` ranges over all the gates in the circuit.
    That is, a circuit success probability is just the product of all its gate
    success probabilities. In this pessimistic model, any gate failure causes the
    circuit to fail.

    Parameters
    ----------
    error_rates : dict
        A dictionary with "gates" and "readout" keys, each of which corresponds to a
        dictionary of error-rates for gates or SPAM elements, respectively.

    n_qubits : int
        The number of qubits in the model.

    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    alias_dict : dict, optional
        An alias dictionary mapping the gate labels in circuits to the
        keys of a (nested) `error_rates` dictionary.  This allows, for instance,
        many gates' error rates to be set by the same model parameter.

    idlename : str, optional
        The gate name to be used for the 1-qubit idle gate (this should be
        set in `error_rates` to add idle errors.
    """
    def __init__(self, error_rates, n_qubits, state_space_labels=None, alias_dict={}, idlename='Gi'):
        ErrorRatesModel.__init__(self, error_rates, n_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idlename=idlename)

    def _circuit_cache(self, circuit):
        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = super()._circuit_cache(circuit)
        all_inds_to_mult = _np.concatenate(inds_to_mult_by_layer)
        all_inds_to_mult_cnt = _np.zeros(self.num_params(), int)
        for i in all_inds_to_mult:
            all_inds_to_mult_cnt[i] += 1
        return all_inds_to_mult, all_inds_to_mult_cnt

    def _success_prob(self, circuit, cache):
        pvec = self._paramvec**2

        if cache is None:
            cache = self._circuit_cache(circuit)

        all_inds_to_mult, all_inds_to_mult_cnt = cache
        # The success probability for all the operations (the entanglment fidelity for the gates)
        sp = 1.0 - pvec

        # The probability that every operation succeeds.
        successprob_circuit = prod(sp[all_inds_to_mult])

        return successprob_circuit

    def _success_dprob(self, circuit, param_slice, cache):
        assert(param_slice is None or _slct.length(param_slice) == len(self._paramvec)), \
            "No support for derivatives with respect to a subset of model parameters yet!"
        pvec = self._paramvec**2
        dpvec_dparams = 2 * self._paramvec

        if cache is None:
            cache = self._circuit_cache(circuit)

        all_inds_to_mult, all_inds_to_mult_cnt = cache
        sp = 1.0 - pvec
        successprob_circuit = prod(sp[all_inds_to_mult])
        deriv = _np.zeros(len(pvec), 'd')
        for i, n in enumerate(all_inds_to_mult_cnt):
            deriv[i] = n * successprob_circuit / sp[i] * -1.0

        return deriv * dpvec_dparams


class AnyErrorCausesRandomOutputModel(ErrorRatesModel):
    """
    A model where any gate error causes a random circuit output.

    Specifically, the success probability of a circuit is give by
    `all_ok + (1 - all_ok) * 1 / 2^circuit_width` where `all_ok` is the
    probability that all the gates succeed:
    `all_ok = 1 - prod(1 - error_rate_i)` with `i` ranging over all the
    gates in the circuit.  In this model, any gate failure causes the
    circuit to produce a random output.

    Parameters
    ----------
    error_rates : dict
        A dictionary with "gates" and "readout" keys, each of which corresponds to a
        dictionary of error-rates for gates or SPAM elements, respectively.

    n_qubits : int
        The number of qubits in the model.

    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    alias_dict : dict, optional
        An alias dictionary mapping the gate labels in circuits to the
        keys of a (nested) `error_rates` dictionary.  This allows, for instance,
        many gates' error rates to be set by the same model parameter.

    idlename : str, optional
        The gate name to be used for the 1-qubit idle gate (this should be
        set in `error_rates` to add idle errors.
    """
    def __init__(self, error_rates, n_qubits, state_space_labels=None, alias_dict={}, idlename='Gi'):
        ErrorRatesModel.__init__(self, error_rates, n_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idlename=idlename)

    def _circuit_cache(self, circuit):
        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = super()._circuit_cache(circuit)
        all_inds_to_mult = _np.concatenate(inds_to_mult_by_layer)
        all_inds_to_mult_cnt = _np.zeros(self.num_params(), int)
        for i in all_inds_to_mult:
            all_inds_to_mult_cnt[i] += 1
        return one_over_2_width, all_inds_to_mult, all_inds_to_mult_cnt

    def _success_prob(self, circuit, cache):
        pvec = self._paramvec**2
        if cache is None:
            cache = self._circuit_cache(circuit)

        one_over_2_width, all_inds_to_mult, all_inds_to_mult_cnt = cache
        # The success probability for all the operations (the entanglment fidelity for the gates)
        sp = 1.0 - pvec

        # The probability that every operation succeeds.
        successprob_all_ops = prod(sp[all_inds_to_mult])
        # The circuit succeeds if all ops succeed, and has a random outcome otherwise.
        successprob_circuit = successprob_all_ops + (1 - successprob_all_ops) * one_over_2_width

        return successprob_circuit

    def _success_dprob(self, circuit, param_slice, cache):
        """
        todo
        """
        assert(param_slice is None or _slct.length(param_slice) == len(self._paramvec)), \
            "No support for derivatives with respect to a subset of model parameters yet!"
        pvec = self._paramvec**2
        dpvec_dparams = 2 * self._paramvec

        if cache is None:
            cache = self._circuit_cache(circuit)

        one_over_2_width, all_inds_to_mult, all_inds_to_mult_cnt = cache
        sp = 1.0 - pvec

        successprob_all_ops = prod(sp[all_inds_to_mult])
        deriv = _np.zeros(len(pvec), 'd')
        for i, n in enumerate(all_inds_to_mult_cnt):
            deriv[i] = n * successprob_all_ops / sp[i] * -1.0

        # The circuit succeeds if all ops succeed, and has a random outcome otherwise.
        # successprob_circuit = successprob_all_ops + (1 - successprob_all_ops) / 2**width
        # = const + (1-1/2**width)*successprobs_all_ops
        deriv *= (1.0 - one_over_2_width)
        return deriv * dpvec_dparams

    # def ORIGINAL_success_prob(self, circuit, cache):
    #     """
    #     todo
    #     """
    #     if not isinstance(circuit, _Circuit):
    #         circuit = _Circuit.from_tuple(circuit)

    #     depth = circuit.depth
    #     width = circuit.width()
    #     pvec = self._paramvec
    #     g_inds = self._gate_error_rate_indices
    #     r_inds = self._readout_error_rate_indices

    #     if self.model_type in ('FE', 'FiE+U'):

    #         two_q_gates = []
    #         for i in range(depth):
    #             layer = circuit.get_layer(i)
    #             two_q_gates += [q.qubits for q in layer if len(q.qubits) > 1]

    #         sp = 1
    #         oneqs = {q: depth for q in circuit.line_labels}

    #         for qs in two_q_gates:
    #             sp = sp * (1 - pvec[g_inds[frozenset(qs)]])
    #             oneqs[qs[0]] += -1
    #             oneqs[qs[1]] += -1

    #         sp = sp * _np.prod([(1 - pvec[g_inds[q]])**oneqs[q]
    #                             * (1 - pvec[r_inds[q]]) for q in circuit.line_labels])

    #         if self.model_type == 'FiE+U':
    #             sp = sp + (1 - sp) * (1 / 2**width)

    #         return sp

    #     if self.model_type == 'GlobalDep':

    #         p = 1
    #         for i in range(depth):

    #             layer = circuit.get_layer(i)
    #             sp_layer = 1
    #             usedQs = []

    #             for gate in layer:
    #                 if len(gate.qubits) > 1:
    #                     usedQs += list(gate.qubits)
    #                     sp_layer = sp_layer * (1 - pvec[g_inds[frozenset(gate.qubits)]])

    #             for q in circuit.line_labels:
    #                 if q not in usedQs:
    #                     sp_layer = sp_layer * (1 - pvec[g_inds[q]])

    #             p_layer = 1 - 4**width * (1 - sp_layer) / (4**width - 1)
    #             p = p * p_layer

    #         # Bit-flip readout error as a pre-measurement depolarizing channel.
    #         sp_layer = _np.prod([(1 - 3 * pvec[r_inds[q]] / 2) for q in circuit.line_labels])
    #         p_layer = 1 - 4**width * (1 - sp_layer) / (4**width - 1)
    #         p = p * p_layer
    #         sp = p + (1 - p) * (1 / 2**width)

    #         return sp
