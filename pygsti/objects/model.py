"""
Defines the Model class and supporting functionality.
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
import scipy as _scipy
import itertools as _itertools
import collections as _collections
import warnings as _warnings
import time as _time
import uuid as _uuid
import bisect as _bisect
import copy as _copy

from ..tools import matrixtools as _mt
from ..tools import optools as _gt
from ..tools import slicetools as _slct
from ..tools import likelihoodfns as _lf
from ..tools import jamiolkowski as _jt
from ..tools import basistools as _bt
from ..tools import listtools as _lt
from ..tools import symplectic as _symp

from . import modelmember as _gm
from . import circuit as _cir
from . import operation as _op
from . import spamvec as _sv
from . import povm as _povm
from . import instrument as _instrument
from . import labeldicts as _ld
from . import gaugegroup as _gg
from . import forwardsim as _fwdsim
from . import matrixforwardsim as _matrixfwdsim
from . import mapforwardsim as _mapfwdsim
from . import termforwardsim as _termfwdsim
from . import explicitcalc as _explicitcalc

from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from .label import Label as _Label
from .bulkcircuitlist import BulkCircuitList as _BulkCircuitList
from .layerrules import LayerRules as _LayerRules


class Model(object):
    """
    A predictive model for a Quantum Information Processor (QIP).

    The main function of a `Model` object is to compute the outcome
    probabilities of :class:`Circuit` objects based on the action of the
    model's ideal operations plus (potentially) noise which makes the
    outcome probabilities deviate from the perfect ones.
v
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
        if isinstance(state_space_labels, _ld.StateSpaceLabels):
            self._state_space_labels = state_space_labels
        else:
            self._state_space_labels = _ld.StateSpaceLabels(state_space_labels)

        self._hyperparams = {}
        self._paramvec = _np.zeros(0, 'd')
        self._paramlbls = None  # a placeholder for FUTURE functionality
        self.uuid = _uuid.uuid4()  # a Model's uuid is like a persistent id(), useful for hashing

    @property
    def state_space_labels(self):
        """
        State space labels

        Returns
        -------
        StateSpaceLabels
        """
        return self._state_space_labels

    @property
    def hyperparams(self):
        """
        Dictionary of hyperparameters associated with this model

        Returns
        -------
        dict
        """
        return self._hyperparams  # Note: no need to set this param - just set/update values

    def num_params(self):
        """
        The number of free parameters when vectorizing this model.

        Returns
        -------
        int
            the number of model parameters.
        """
        return len(self._paramvec)

    def to_vector(self):
        """
        Returns the model vectorized according to the optional parameters.

        Returns
        -------
        numpy array
            The vectorized model parameters.
        """
        return self._paramvec

    def from_vector(self, v, reset_basis=False):
        """
        Sets this Model's operations based on parameter values `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A vector of parameters, with length equal to `self.num_params()`.

        reset_basis : bool, optional
            UNUSED

        Returns
        -------
        None
        """
        assert(len(v) == self.num_params())
        self._paramvec = v.copy()

    def probs(self, circuit, clip_to=None):
        """
        Construct a dictionary containing the outcome probabilities of `circuit`.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        clip_to : 2-tuple, optional
            (min,max) to clip probabilities to if not None.

        Returns
        -------
        probs : dictionary
            A dictionary such that
            probs[SL] = pr(SL,circuit,clip_to)
            for each spam label (string) SL.
        """
        raise NotImplementedError("Derived classes should implement this!")

#REMOVE
#    def dprobs(self, circuit, return_pr=False, clip_to=None):
#        """
#        Construct a dictionary containing the outcome probability derivatives of `circuit`.
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
#        Returns
#        -------
#        dprobs : dictionary
#            A dictionary such that
#            dprobs[SL] = dpr(SL,circuit,gates,G0,SPAM,SP0,return_pr,clip_to)
#            for each spam label (string) SL.
#        """
#        #Finite difference default?
#        raise NotImplementedError("Derived classes should implement this!")
#
#    def hprobs(self, circuit, return_pr=False, return_deriv=False, clip_to=None):
#        """
#        Construct a dictionary containing the outcome probability 2nd derivatives of `circuit`.
#
#        Parameters
#        ----------
#        circuit : Circuit or tuple of operation labels
#            The sequence of operation labels specifying the circuit.
#
#        return_pr : bool, optional
#            when set to True, additionally return the probabilities.
#
#        return_deriv : bool, optional
#            when set to True, additionally return the derivatives of the
#            probabilities.
#
#        clip_to : 2-tuple, optional
#            (min,max) to clip returned probability to if not None.
#            Only relevant when return_pr == True.
#
#        Returns
#        -------
#        hprobs : dictionary
#            A dictionary such that
#            hprobs[SL] = hpr(SL,circuit,gates,G0,SPAM,SP0,return_pr,return_deriv,clip_to)
#            for each spam label (string) SL.
#        """
#        raise NotImplementedError("Derived classes should implement this!")
#
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
#        raise NotImplementedError("Derived classes should implement this!")
#        #return circuit_list # MORE?
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
#        #return circuit_list # MORE?
#
#    #def uses_evaltrees(self):
#    #    """
#    #    Whether or not this model uses evaluation trees to compute many
#    #    (bulk) probabilities and their derivatives.
#    #
#    #    Returns
#    #    -------
#    #    bool
#    #    """
#    #    return False

    def bulk_probs(self, circuit_list, clip_to=None, check=False,
                   comm=None, mem_limit=None, dataset=None, smartc=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuit_list : list of (tuples or Circuits)
            Each element specifies a circuit to compute quantities for.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        check : boolean, optional
            If True, perform extra checks within code to verify correctness,
            generating warnings when checks fail.  Used for testing, and runs
            much slower when True.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of evalTree (if it is split).

        mem_limit : int, optional
            A rough memory limit in bytes which is used to determine processor
            allocation.

        dataset : DataSet, optional
            If not None, restrict what is computed to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.

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

#REMOVE
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
#            `(outcome, dp)`).
#        """
#        raise NotImplementedError("Derived classes should implement this!")
#
#    def bulk_hprobs(self, circuit_list, return_pr=False, return_deriv=False,
#                    clip_to=None, check=False, comm=None,
#                    wrt_block_size1=None, wrt_block_size2=None, dataset=None):
#        """
#        Construct a dictionary containing the probability-Hessians for an entire list of circuits.
#
#        Parameters
#        ----------
#        circuit_list : list of (tuples or Circuits)
#            Each element specifies a circuit to compute quantities for.
#
#        return_pr : bool, optional
#            when set to True, additionally return the probabilities.
#
#        return_deriv : bool, optional
#            when set to True, additionally return the probability derivatives.
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
#            across multiple processors.
#
#        wrt_block_size1 : int or float, optional
#            The maximum number of 1st (row) derivatives to compute
#            *products* for simultaneously.  None means compute all requested
#            rows or columns at once.  Set this to non-None to reduce amount of
#            intermediate memory required.
#
#        wrt_block_size2 : int or float, optional
#            The maximum number of 2nd (col) derivatives to compute
#            *products* for simultaneously.  None means compute all requested
#            rows or columns at once.  Set this to non-None to reduce amount of
#            intermediate memory required.
#
#        dataset : DataSet, optional
#            If not None, restrict what is computed to only those
#            probabilities corresponding to non-zero counts (observed
#            outcomes) in this data set.
#
#        Returns
#        -------
#        hprobs : dictionary
#            A dictionary such that `probs[opstr]` is an ordered dictionary of
#            `(outcome, hp, dp, p)` tuples, where `outcome` is a tuple of labels,
#            `p` is the corresponding probability, `dp` is a 1D array containing
#            the derivative of `p` with respect to each parameter, and `hp` is a
#            2D array containing the Hessian of `p` with respect to each parameter.
#            If `return_pr` if False, then `p` is not included in the tuples.
#            If `return_deriv` if False, then `dp` is not included in the tuples.
#        """
#        raise NotImplementedError("Derived classes should implement this!")
#
#    def bulk_fill_probs(self, mx_to_fill, eval_tree, clip_to=None, check=False, comm=None):
#        """
#        Compute the outcome probabilities for an entire tree of circuits.
#
#        This routine fills a 1D array, `mx_to_fill` with the probabilities
#        corresponding to the *simplified* circuits found in an evaluation
#        tree, `eval_tree`.  An initial list of (general) :class:`Circuit`
#        objects is *simplified* into a lists of gate-only sequences along with
#        a mapping of final elements (i.e. probabilities) to gate-only sequence
#        and prep/effect pairs.  The evaluation tree organizes how to efficiently
#        compute the gate-only sequences.  This routine fills in `mx_to_fill`, which
#        must have length equal to the number of final elements (this can be
#        obtained by `eval_tree.num_final_elements()`.  To interpret which elements
#        correspond to which strings and outcomes, you'll need the mappings
#        generated when the original list of `Circuits` was simplified.
#
#        Parameters
#        ----------
#        mx_to_fill : numpy ndarray
#            an already-allocated 1D numpy array of length equal to the
#            total number of computed elements (i.e. eval_tree.num_final_elements())
#
#        eval_tree : EvalTree
#            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
#            strings to compute the bulk operation on.
#
#        clip_to : 2-tuple, optional
#            (min,max) to clip return value if not None.
#
#        check : boolean, optional
#            If True, perform extra checks within code to verify correctness,
#            generating warnings when checks fail.  Used for testing, and runs
#            much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not None, an MPI communicator for distributing the computation
#            across multiple processors.  Distribution is performed over
#            subtrees of eval_tree (if it is split).
#
#        Returns
#        -------
#        None
#        """
#        raise NotImplementedError("Derived classes should implement this!")
#
#    def bulk_fill_dprobs(self, mx_to_fill, eval_tree, pr_mx_to_fill=None, clip_to=None,
#                         check=False, comm=None, wrt_block_size=None,
#                         profiler=None, gather_mem_limit=None):
#        """
#        Compute the outcome probability-derivatives for an entire tree of circuits.
#
#        Similar to `bulk_fill_probs(...)`, but fills a 2D array with
#        probability-derivatives for each "final element" of `eval_tree`.
#
#        Parameters
#        ----------
#        mx_to_fill : numpy ndarray
#            an already-allocated ExM numpy array where E is the total number of
#            computed elements (i.e. eval_tree.num_final_elements()) and M is the
#            number of model parameters.
#
#        eval_tree : EvalTree
#            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
#            strings to compute the bulk operation on.
#
#        pr_mx_to_fill : numpy array, optional
#            when not None, an already-allocated length-E numpy array that is filled
#            with probabilities, just like in bulk_fill_probs(...).
#
#        clip_to : 2-tuple, optional
#            (min,max) to clip return value if not None.
#
#        check : boolean, optional
#            If True, perform extra checks within code to verify correctness,
#            generating warnings when checks fail.  Used for testing, and runs
#            much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not None, an MPI communicator for distributing the computation
#            across multiple processors.  Distribution is first performed over
#            subtrees of eval_tree (if it is split), and then over blocks (subsets)
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
#        profiler : Profiler, optional
#            A profiler object used for to track timing and memory usage.
#
#        gather_mem_limit : int, optional
#            A memory limit in bytes to impose upon the "gather" operations
#            performed as a part of MPI processor syncronization.
#
#        Returns
#        -------
#        None
#        """
#        raise NotImplementedError("Derived classes should implement this!")
#
#    def bulk_fill_hprobs(self, mx_to_fill, eval_tree=None,
#                         pr_mx_to_fill=None, deriv_mx_to_fill=None,
#                         clip_to=None, check=False, comm=None,
#                         wrt_block_size1=None, wrt_block_size2=None,
#                         gather_mem_limit=None):
#
#        """
#        Compute the outcome probability-Hessians for an entire tree of circuits.
#
#        Similar to `bulk_fill_probs(...)`, but fills a 3D array with
#        probability-Hessians for each "final element" of `eval_tree`.
#
#        Parameters
#        ----------
#        mx_to_fill : numpy ndarray
#            an already-allocated ExMxM numpy array where E is the total number of
#            computed elements (i.e. eval_tree.num_final_elements()) and M1 & M2 are
#            the number of selected gate-set parameters (by wrt_filter1 and wrt_filter2).
#
#        eval_tree : EvalTree
#            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
#            strings to compute the bulk operation on.
#
#        pr_mx_to_fill : numpy array, optional
#            when not None, an already-allocated length-E numpy array that is filled
#            with probabilities, just like in bulk_fill_probs(...).
#
#        deriv_mx_to_fill : numpy array, optional
#            when not None, an already-allocated ExM numpy array that is filled
#            with probability derivatives.
#
#        clip_to : 2-tuple, optional
#            (min,max) to clip return value if not None.
#
#        check : boolean, optional
#            If True, perform extra checks within code to verify correctness,
#            generating warnings when checks fail.  Used for testing, and runs
#            much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not None, an MPI communicator for distributing the computation
#            across multiple processors.  Distribution is first performed over
#            subtrees of eval_tree (if it is split), and then over blocks (subsets)
#            of the parameters being differentiated with respect to (see
#            wrt_block_size).
#
#        wrt_block_size1 : int or float, optional
#            The maximum number of 1st (row) derivatives to compute
#            *products* for simultaneously.  None means compute all requested
#            rows or columns at once.  Set this to non-None to reduce amount of
#            intermediate memory required.
#
#        wrt_block_size2 : int or float, optional
#            The maximum number of 2nd (col) derivatives to compute
#            *products* for simultaneously.  None means compute all requested
#            rows or columns at once.  Set this to non-None to reduce amount of
#            intermediate memory required.
#
#        gather_mem_limit : int, optional
#            A memory limit in bytes to impose upon the "gather" operations
#            performed as a part of MPI processor syncronization.
#
#        Returns
#        -------
#        None
#        """
#        raise NotImplementedError("Derived classes should implement this!")

    def _init_copy(self, copy_into):
        """
        Copies any "tricky" member of this model into `copy_into`, before
        deep copying everything else within a .copy() operation.
        """
        copy_into.uuid = _uuid.uuid4()  # new uuid for a copy (don't duplicate!)

    def copy(self):
        """
        Copy this model.

        Returns
        -------
        Model
            a (deep) copy of this model.
        """
        #Avoid having to reconstruct everything via __init__;
        # essentially deepcopy this object, but give the
        # class opportunity to initialize tricky members instead
        # of letting deepcopy do it.
        newModel = type(self).__new__(self.__class__)  # empty object

        #first call _init_copy to initialize any tricky members
        # (like those that contain references to self or other members)
        self._init_copy(newModel)

        for attr, val in self.__dict__.items():
            if not hasattr(newModel, attr):
                assert(attr != "uuid"), "Should not be copying UUID!"
                setattr(newModel, attr, _copy.deepcopy(val))

        return newModel

    def __str__(self):
        pass

    def __hash__(self):
        if self.uuid is not None:
            return hash(self.uuid)
        else:
            raise TypeError('Use digest hash')

    def circuit_outcomes(self, circuit):
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
        return ()  # default = no outcomes

    def compute_num_outcomes(self, circuit):
        """
        The number of outcomes of `circuit`, given by it's existing or implied POVM label.

        Parameters
        ----------
        circuit : Circuit
            The circuit to simplify

        Returns
        -------
        int
        """
        return len(self.circuit_outcomes(circuit))

    def complete_circuit(self, circuit):
        """
        Adds any implied preparation or measurement layers to `circuit`

        Parameters
        ----------
        circuit : Circuit
            Circuit to act on.

        Returns
        -------
        Circuit
            Possibly the same object as `circuit`, if no additions are needed.
        """
        return circuit


class OpModel(Model):
    """
    A Model that contains operators (i.e. "members"), having a container structure.

    These operators are independently (sort of) parameterized and can be thought
    to have dense representations (even if they're not actually stored that way).
    This gives rise to the model having `basis` and `evotype` members.

    Secondly, attached to an `OpModel` is the idea of "circuit simplification"
    whereby the operators (preps, operations, povms, instruments) within
    a circuit get simplified to things corresponding to a single outcome
    probability, i.e. pseudo-circuits containing just preps, operations,
    and POMV effects.

    Thirdly, an `OpModel` is assumed to use a *layer-by-layer* evolution, and,
    because of circuit simplification process, the calculaton of circuit
    outcome probabilities has been pushed to a :class:`ForwardSimulator`
    object which just deals with the forward simulation of simplified circuits.
    Furthermore, instead of relying on a static set of operations a forward
    simulator queries a :class:`LayerLizard` for layer operations, making it
    possible to build up layer operations in an on-demand fashion from pieces
    within the model.

    Parameters
    ----------
    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    basis : Basis
        The basis used for the state space by dense operator representations.

    evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
        The evolution type of this model, describing how states are
        represented, allowing compatibility checks with (super)operator
        objects.

    simplifier_helper : SimplifierHelper
        Provides a minimal interface for compiling circuits for forward
        simulation.

    sim_type : {"auto", "matrix", "map", "termorder:X"}
        The type of forward simulator this model should use.  `"auto"`
        tries to determine the best type automatically.
    """

    #Whether to perform extra parameter-vector integrity checks
    _pcheck = False

    def __init__(self, state_space_labels, basis, evotype, layer_rules, simulator="auto"):
        """
        Creates a new OpModel.  Rarely used except from derived classes `__init__` functions.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels or list or tuple
            The decomposition (with labels) of (pure) state-space this model
            acts upon.  Regardless of whether the model contains operators or
            superoperators, this argument describes the Hilbert space dimension
            and imposed structure.  If a list or tuple is given, it must be
            of a from that can be passed to `StateSpaceLabels.__init__`.

        basis : Basis
            The basis used for the state space by dense operator representations.

        evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
            The evolution type of this model, describing how states are
            represented, allowing compatibility checks with (super)operator
            objects.

        layer_rules : LayerRules
            The "layer rules" used for constructing operators for circuit
            layers.  This functionality is essential to using this model to
            simulate ciruits, and is typically supplied by derived classes.

        simulator : ForwardSimulator or {"auto", "matrix", "map"}
            The forward simulator this model should use.  `"auto"`
            tries to determine and instantiate the best type automatically.
        """
        self._evotype = evotype
        self._set_state_space(state_space_labels, basis)
        #sets self._state_space_labels, self._basis, self._dim

        if simulator == "auto":
            d = self._dim if (self._dim is not None) else 0
            simulator = "matrix" if d <= 16 else "map"
        if simulator == "matrix":
            self._sim = _matrixfwdsim.MatrixForwardSimulator(self)
        elif simulator == "map":
            self._sim = _mapfwdsim.MapForwardSimulator(max_cache_size=0)  # default is to *not* use a cache
        else:
            assert(isinstance(simulator, _fwdsim.ForwardSimulator)), "`simulator` argument must be a ForwardSimulator!"
            self._sim = simulator

        self._layer_rules = layer_rules if (layer_rules is not None) else _LayerRules()
        self._layerop_cache = {}  # for all (any type) of non-primitive layer operation
        self._need_to_rebuild = True  # whether we call _rebuild_paramvec() in to_vector() or num_params()
        self.dirty = False  # indicates when objects and _paramvec may be out of sync

        super(OpModel, self).__init__(self.state_space_labels)

    ##########################################
    ## Get/Set methods
    ##########################################

    @property
    def sim(self):
        """ Forward simulator for this model """
        self._clean_paramvec()  # clear opcache and rebuild paramvec when needed
        return self._sim

    #@property
    #def simtype(self):
    #    """
    #    Forward simulation type
    #
    #    Returns
    #    -------
    #    str
    #    """
    #    return type(self._sim)

    @property
    def evotype(self):
        """
        Evolution type

        Returns
        -------
        str
        """
        return self._evotype

    @property
    def basis(self):
        """
        The basis used to represent dense (super)operators of this model

        Returns
        -------
        Basis
        """
        return self._basis

    @basis.setter
    def basis(self, basis):
        """
        The basis used to represent dense (super)operators of this model
        """
        if isinstance(basis, _Basis):
            assert(basis.dim == self.state_space_labels.dim), \
                "Cannot set basis w/dim=%d when sslbls dim=%d!" % (basis.dim, self.state_space_labels.dim)
            self._basis = basis
        else:  # create a basis with the proper structure & dimension
            self._basis = _Basis.cast(basis, self.state_space_labels)

    #TODO REMOVE
    #def reset_basis(self):
    #    """
    #    "Forgets" the current basis, so that
    #    self.basis becomes a dummy Basis w/name "unknown".
    #    """
    #    self._basis = _BuiltinBasis('unknown', 0)

    def _set_state_space(self, lbls, basis="pp"):
        """
        Sets labels for the components of the Hilbert space upon which the gates of this Model act.

        Parameters
        ----------
        lbls : list or tuple or StateSpaceLabels object
            A list of state-space labels (can be strings or integers), e.g.
            `['Q0','Q1']` or a :class:`StateSpaceLabels` object.

        basis : Basis or str
            A :class:`Basis` object or a basis name (like `"pp"`), specifying
            the basis used to interpret the operators in this Model.  If a
            `Basis` object, then its dimensions must match those of `lbls`.

        Returns
        -------
        None
        """
        if isinstance(lbls, _ld.StateSpaceLabels):
            self._state_space_labels = lbls
        else:
            self._state_space_labels = _ld.StateSpaceLabels(lbls, evotype=self._evotype)
        self.basis = basis  # invokes basis setter to set self._basis

        #Operator dimension of this Model
        self._dim = self.state_space_labels.dim
        #e.g. 4 for 1Q (densitymx) or 2 for 1Q (statevec)

    @property
    def dim(self):
        """
        The dimension of the model.

        This equals d when the gate (or, more generally, circuit-layer) matrices
        would have shape d x d and spam vectors would have shape d x 1 (if they
        were computed).

        Returns
        -------
        int
            model dimension
        """
        return self._dim

    #TODO REMOVE - use dim property
    def get_dimension(self):
        """
        Get the dimension of the model.

        This equals d when the gate matrices have shape d x d and spam vectors
        have shape d x 1.  Equivalent to model.dim.

        Returns
        -------
        int
            model dimension
        """
        return self._dim

    ####################################################
    ## Parameter vector maintenance
    ####################################################

    def num_params(self):
        """
        The number of free parameters when vectorizing this model.

        Returns
        -------
        int
            the number of model parameters.
        """
        self._clean_paramvec()
        return len(self._paramvec)

    def _iter_parameterized_objs(self):
        raise NotImplementedError("Derived Model classes should implement _iter_parameterized_objs")
        #return # default is to have no parameterized objects

    def _check_paramvec(self, debug=False):
        if debug: print("---- Model._check_paramvec ----")

        TOL = 1e-8
        for lbl, obj in self._iter_parameterized_objs():
            if debug: print(lbl, ":", obj.num_params(), obj.gpindices)
            w = obj.to_vector()
            msg = "None" if (obj.parent is None) else id(obj.parent)
            assert(obj.parent is self), "%s's parent is not set correctly (%s)!" % (lbl, msg)
            if obj.gpindices is not None and len(w) > 0:
                if _np.linalg.norm(self._paramvec[obj.gpindices] - w) > TOL:
                    if debug: print(lbl, ".to_vector() = ", w, " but Model's paramvec = ",
                                    self._paramvec[obj.gpindices])
                    raise ValueError("%s is out of sync with paramvec!!!" % lbl)
            if not self.dirty and obj.dirty:
                raise ValueError("%s is dirty but Model.dirty=False!!" % lbl)

    def _clean_paramvec(self):
        """ Updates _paramvec corresponding to any "dirty" elements, which may
            have been modified without out knowing, leaving _paramvec out of
            sync with the element's internal data.  It *may* be necessary
            to resolve conflicts where multiple dirty elements want different
            values for a single parameter.  This method is used as a safety net
            that tries to insure _paramvec & Model elements are consistent
            before their use."""

        #print("Cleaning Paramvec (dirty=%s, rebuild=%s)" % (self.dirty, self._need_to_rebuild))
        #import inspect, pprint
        #pprint.pprint([(x.filename,x.lineno,x.function) for x in inspect.stack()[0:7]])

        if self._need_to_rebuild:
            self._rebuild_paramvec()
            self._need_to_rebuild = False
            self._reinit_layerop_cache()  # changes to parameter vector structure invalidate cached ops

        if self.dirty:  # if any member object is dirty (ModelMember.dirty setter should set this value)
            TOL = 1e-8

            #Note: lbl args used *just* for potential debugging - could strip out once
            # we're confident this code always works.
            def clean_single_obj(obj, lbl):  # sync an object's to_vector result w/_paramvec
                if obj.dirty:
                    w = obj.to_vector()
                    chk_norm = _np.linalg.norm(self._paramvec[obj.gpindices] - w)
                    #print(lbl, " is dirty! vec = ", w, "  chk_norm = ",chk_norm)
                    if (not _np.isfinite(chk_norm)) or chk_norm > TOL:
                        self._paramvec[obj.gpindices] = w
                    obj.dirty = False

            def clean_obj(obj, lbl):  # recursive so works with objects that have sub-members
                for i, subm in enumerate(obj.submembers()):
                    clean_obj(subm, _Label(lbl.name + ":%d" % i, lbl.sslbls))
                clean_single_obj(obj, lbl)

            def reset_dirty(obj):  # recursive so works with objects that have sub-members
                for i, subm in enumerate(obj.submembers()): reset_dirty(subm)
                obj.dirty = False

            for lbl, obj in self._iter_parameterized_objs():
                clean_obj(obj, lbl)

            #re-update everything to ensure consistency ~ self.from_vector(self._paramvec)
            #print("DEBUG: non-trivially CLEANED paramvec due to dirty elements")
            for _, obj in self._iter_parameterized_objs():
                obj.from_vector(self._paramvec[obj.gpindices], nodirty=True)
                reset_dirty(obj)  # like "obj.dirty = False" but recursive
                #object is known to be consistent with _paramvec

            self.dirty = False
            self._reinit_layerop_cache()  # changes to parameter vector structure invalidate cached ops

        if OpModel._pcheck: self._check_paramvec()

    def _mark_for_rebuild(self, modified_obj=None):
        #re-initialze any members that also depend on the updated parameters
        self._need_to_rebuild = True
        for _, o in self._iter_parameterized_objs():
            if o._obj_refcount(modified_obj) > 0:
                o.clear_gpindices()  # ~ o.gpindices = None but works w/submembers
                # (so params for this obj will be rebuilt)
        self.dirty = True
        #since it's likely we'll set at least one of our object's .dirty flags
        # to True (and said object may have parent=None and so won't
        # auto-propagate up to set this model's dirty flag (self.dirty)

    def _print_gpindices(self):
        print("PRINTING MODEL GPINDICES!!!")
        for lbl, obj in self._iter_parameterized_objs():
            print("LABEL ", lbl)
            obj._print_gpindices()

    def _rebuild_paramvec(self):
        """ Resizes self._paramvec and updates gpindices & parent members as needed,
            and will initialize new elements of _paramvec, but does NOT change
            existing elements of _paramvec (use _update_paramvec for this)"""
        v = self._paramvec; Np = len(self._paramvec)  # NOT self.num_params() since the latter calls us!
        off = 0; shift = 0
        #print("DEBUG: rebuilding...")

        #Step 1: remove any unused indices from paramvec and shift accordingly
        used_gpindices = set()
        for _, obj in self._iter_parameterized_objs():
            if obj.gpindices is not None:
                assert(obj.parent is self), "Member's parent is not set correctly (%s)!" % str(obj.parent)
                used_gpindices.update(obj.gpindices_as_array())
            else:
                assert(obj.parent is self or obj.parent is None)
                #Note: ok for objects to have parent == None if their gpindices is also None

        indices_to_remove = sorted(set(range(Np)) - used_gpindices)

        if len(indices_to_remove) > 0:
            #print("DEBUG: Removing %d params:"  % len(indices_to_remove), indices_to_remove)
            v = _np.delete(v, indices_to_remove)
            def get_shift(j): return _bisect.bisect_left(indices_to_remove, j)
            memo = set()  # keep track of which object's gpindices have been set
            for _, obj in self._iter_parameterized_objs():
                if obj.gpindices is not None:
                    if id(obj) in memo: continue  # already processed
                    if isinstance(obj.gpindices, slice):
                        new_inds = _slct.shift(obj.gpindices,
                                               -get_shift(obj.gpindices.start))
                    else:
                        new_inds = []
                        for i in obj.gpindices:
                            new_inds.append(i - get_shift(i))
                        new_inds = _np.array(new_inds, _np.int64)
                    obj.set_gpindices(new_inds, self, memo)

        # Step 2: add parameters that don't exist yet
        #  Note that iteration order (that of _iter_parameterized_objs) determines
        #  parameter index ordering, so "normally" an object that occurs before
        #  another in the iteration order will have gpindices which are lower - and
        #  when new indices are allocated we try to maintain this normal order by
        #  inserting them at an appropriate place in the parameter vector.
        #  off : holds the current point where new params should be inserted
        #  shift : holds the amount existing parameters that are > offset (not in `memo`) should be shifted
        # Note: Adding more explicit "> offset" logic may obviate the need for the memo arg?
        memo = set()  # keep track of which object's gpindices have been set
        for lbl, obj in self._iter_parameterized_objs():

            if shift > 0 and obj.gpindices is not None:
                if isinstance(obj.gpindices, slice):
                    obj.set_gpindices(_slct.shift(obj.gpindices, shift), self, memo)
                else:
                    obj.set_gpindices(obj.gpindices + shift, self, memo)  # works for integer arrays

            if obj.gpindices is None or obj.parent is not self:
                #Assume all parameters of obj are new independent parameters
                num_new_params = obj.allocate_gpindices(off, self, memo)
                objvec = obj.to_vector()  # may include more than "new" indices
                if num_new_params > 0:
                    new_local_inds = _gm._decompose_gpindices(obj.gpindices, slice(off, off + num_new_params))
                    assert(len(objvec[new_local_inds]) == num_new_params)
                    v = _np.insert(v, off, objvec[new_local_inds])
                # print("objvec len = ",len(objvec), "num_new_params=",num_new_params,
                #       " gpinds=",obj.gpindices) #," loc=",new_local_inds)

                #obj.set_gpindices( slice(off, off+obj.num_params()), self )
                #shift += obj.num_params()
                #off += obj.num_params()

                shift += num_new_params
                off += num_new_params
                #print("DEBUG: %s: alloc'd & inserted %d new params.  indices = " \
                #      % (str(lbl),obj.num_params()), obj.gpindices, " off=",off)
            else:
                inds = obj.gpindices_as_array()
                M = max(inds) if len(inds) > 0 else -1; L = len(v)
                #print("DEBUG: %s: existing indices = " % (str(lbl)), obj.gpindices, " M=",M," L=",L)
                if M >= L:
                    #Some indices specified by obj are absent, and must be created.
                    w = obj.to_vector()
                    v = _np.concatenate((v, _np.empty(M + 1 - L, 'd')), axis=0)  # [v.resize(M+1) doesn't work]
                    shift += M + 1 - L
                    for ii, i in enumerate(inds):
                        if i >= L: v[i] = w[ii]
                    #print("DEBUG:    --> added %d new params" % (M+1-L))
                if M >= 0:  # M == -1 signifies this object has no parameters, so we'll just leave `off` alone
                    off = M + 1

        self._paramvec = v
        #print("DEBUG: Done rebuild: %d params" % len(v))

    def _init_virtual_obj(self, obj):
        """
        Initializes a "virtual object" - an object (e.g. LinearOperator) that *could* be a
        member of the Model but won't be, as it's just built for temporary
        use (e.g. the parallel action of several "base" gates).  As such
        we need to fully initialize its parent and gpindices members so it
        knows it belongs to this Model BUT it's not allowed to add any new
        parameters (they'd just be temporary).  It's also assumed that virtual
        objects don't need to be to/from-vectored as there are already enough
        real (non-virtual) gates/spamvecs/etc. to accomplish this.
        """
        if obj.gpindices is not None:
            assert(obj.parent is self), "Virtual obj has incorrect parent already set!"
            return  # if parent is already set we assume obj has already been init

        #Assume all parameters of obj are new independent parameters
        num_new_params = obj.allocate_gpindices(self.num_params(), self)
        assert(num_new_params == 0), "Virtual object is requesting %d new params!" % num_new_params

    def _obj_refcount(self, obj):
        """ Number of references to `obj` contained within this Model """
        cnt = 0
        for _, o in self._iter_parameterized_objs():
            cnt += o._obj_refcount(obj)
        return cnt

    def to_vector(self):
        """
        Returns the model vectorized according to the optional parameters.

        Returns
        -------
        numpy array
            The vectorized model parameters.
        """
        self._clean_paramvec()  # will rebuild if needed
        return self._paramvec

    def from_vector(self, v):
        """
        Sets this Model's operations based on parameter values `v`.

        The inverse of to_vector.

        Parameters
        ----------
        v : numpy.ndarray
            A vector of parameters, with length equal to `self.num_params()`.

        Returns
        -------
        None
        """
        assert(len(v) == self.num_params())

        self._paramvec = v.copy()
        for _, obj in self._iter_parameterized_objs():
            obj.from_vector(v[obj.gpindices])
            obj.dirty = False  # object is known to be consistent with _paramvec

        #if reset_basis:
        #    self.reset_basis()
            # assume the vector we're loading isn't producing gates & vectors in
            # a known basis.
        if OpModel._pcheck: self._check_paramvec()

    ######################################
    ## Compilation
    ######################################

    #REMOVE
    #def _layer_lizard(self):
    #    """ Return a layer lizard for this model """
    #    raise NotImplementedError("Derived Model classes should implement this!")
    #def simplify_circuits(self, circuits, dataset=None):
    #    """
    #    Simplifies a list of :class:`Circuit`s.
    #
    #    Circuits must be "simplified" before probabilities can be computed for
    #    them. Each string corresponds to some number of "outcomes", indexed by an
    #    "outcome label" that is a tuple of POVM-effect or instrument-element
    #    labels like "0".  Compiling creates maps between circuits and their
    #    outcomes and the structures used in probability computation (see return
    #    values below).
    #
    #    Parameters
    #    ----------
    #    circuits : list of Circuits
    #        The list to simplify.
    #
    #    dataset : DataSet, optional
    #        If not None, restrict what is simplified to only those
    #        probabilities corresponding to non-zero counts (observed
    #        outcomes) in this data set.
    #
    #    Returns
    #    -------
    #    raw_elabels_dict : collections.OrderedDict
    #        A dictionary whose keys are simplified circuits (containing just
    #        "simplified" gates, i.e. not instruments) that include preparation
    #        labels but no measurement (POVM). Values are lists of simplified
    #        effect labels, each label corresponds to a single "final element" of
    #        the computation, e.g. a probability.  The ordering is important - and
    #        is why this needs to be an ordered dictionary - when the lists of tuples
    #        are concatenated (by key) the resulting tuple orderings corresponds to
    #        the final-element axis of an output array that is being filled (computed).
    #    elIndices : collections.OrderedDict
    #        A dictionary whose keys are integer indices into `circuits` and
    #        whose values are slices and/or integer-arrays into the space/axis of
    #        final elements.  Thus, to get the final elements corresponding to
    #        `circuits[i]`, use `filledArray[ elIndices[i] ]`.
    #    outcomes : collections.OrderedDict
    #        A dictionary whose keys are integer indices into `circuits` and
    #        whose values are lists of outcome labels (an outcome label is a tuple
    #        of POVM-effect and/or instrument-element labels).  Thus, to obtain
    #        what outcomes the i-th circuit's final elements
    #        (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.
    #    nTotElements : int
    #        The total number of "final elements" - this is how big of an array
    #        is need to hold all of the probabilities `circuits` generates.
    #    """
    #    # model.simplify -> odict[raw_gstr] = spamTuples, elementIndices, nElements
    #    # dataset.simplify -> outcomeLabels[i] = list_of_ds_outcomes, elementIndices, nElements
    #    # simplify all gsplaq strs -> elementIndices[(i,j)],
    #
    #    aliases = circuits.op_label_aliases if isinstance(circuits, _BulkCircuitList) else None
    #    circuits = list(map(_cir.Circuit.cast, circuits))  # cast to Circuits
    #    ds_circuits = _lt.apply_aliases_to_circuits(circuits, aliases)
    #
    #    #Indexed by raw circuit
    #    raw_elabels_dict = _collections.OrderedDict()  # final
    #    raw_opOutcomes_dict = _collections.OrderedDict()
    #    raw_offsets = _collections.OrderedDict()
    #
    #    #Indexed by parent index (an integer)
    #    elIndicesByParent = _collections.OrderedDict()  # final
    #    outcomesByParent = _collections.OrderedDict()  # final
    #    elIndsToOutcomesByParent = _collections.OrderedDict()
    #
    #    def resolve_elabels(circuit, ds_circuit):
    #        """ Determines simplified effect labels that correspond to circuit
    #            and strips any spam-related pieces off """
    #        prep_lbl, circuit, povm_lbl = \
    #            self._split_circuit(circuit)
    #        if prep_lbl is None or povm_lbl is None:
    #            assert(prep_lbl is None and povm_lbl is None)
    #            elabels = [None]  # put a single "dummy" elabel placeholder
    #            # so that there's a single "element" for each simplified string,
    #            # which means that the usual "lookup" or "elIndices" will map
    #            # original circuit-list indices to simplified-string, i.e.,
    #            # eval_tree index, which is useful when computing products
    #            # (often the case when a Model has no preps or povms,
    #            #  e.g. in germ selection)
    #        else:
    #            if dataset is not None:
    #                #Then we don't need to consider *all* possible spam tuples -
    #                # just the ones that are observed, i.e. that correspond to
    #                # a final element in the "full" (tuple) outcome labels that
    #                # were observed.
    #                observed_povm_outcomes = sorted(set(
    #                    [full_out_tup[-1] for full_out_tup in dataset[ds_circuit].outcomes]))
    #                elabels = [povm_lbl + "_" + elbl
    #                           for elbl in observed_povm_outcomes]
    #                # elbl = oout[-1] -- the last element corresponds
    #                # to the POVM (earlier ones = instruments)
    #            else:
    #                if isinstance(povm_lbl, _Label):  # support for POVMs being labels, e.g. for marginalized POVMs
    #                    elabels = [_Label(povm_lbl.name + "_" + elbl, povm_lbl.sslbls)
    #                               for elbl in self._shlp.effect_labels_for_povm(povm_lbl)]
    #                else:
    #                    elabels = [povm_lbl + "_" + elbl
    #                               for elbl in self._shlp.effect_labels_for_povm(povm_lbl)]
    #
    #        #Include prep-label as part of circuit
    #        if prep_lbl is not None:
    #            circuit = circuit.copy(editable=True)
    #            circuit.insert_layer(prep_lbl, 0)
    #            circuit.done_editing()
    #
    #        return circuit, elabels
    #
    #    def process(s, elabels, observed_outcomes, el_inds_to_outcomes,
    #                op_outcomes=(), start=0):
    #        """
    #        Implements recursive processing of a string. Separately
    #        implements two different behaviors:
    #          "add" : add entries to raw_spamTuples_dict and raw_opOutcomes_dict
    #          "index" : adds entries to elIndicesByParent and outcomesByParent
    #                    assuming that raw_spamTuples_dict and raw_opOutcomes_dict
    #                    are already build (and won't be modified anymore).
    #        """
    #        sub = s if start == 0 else s[start:]
    #        for i, op_label in enumerate(sub, start=start):
    #
    #            # OLD: now allow "gate-level" labels which can contain
    #            # multiple (parallel) instrument labels
    #            #if op_label in self.instruments:
    #            #    #we've found an instrument - recurse!
    #            #    for inst_el_lbl in self.instruments[op_label]:
    #            #        simplified_el_lbl = op_label + "_" + inst_el_lbl
    #            #        process(s[0:i] + _cir.Circuit((simplified_el_lbl,)) + s[i+1:],
    #            #                spamtuples, el_inds_to_outcomes, op_outcomes + (inst_el_lbl,), i+1)
    #            #    break
    #
    #            if any([self._shlp.is_instrument_lbl(sub_gl) for sub_gl in op_label.components]):
    #                # we've found an instrument - recurse!
    #                sublabel_tups_to_iter = []  # one per label component (may be only 1)
    #                for sub_gl in op_label.components:
    #                    if self._shlp.is_instrument_lbl(sub_gl):
    #                        sublabel_tups_to_iter.append(
    #                            [(sub_gl, inst_el_lbl)
    #                             for inst_el_lbl in self._shlp.member_labels_for_instrument(sub_gl)])
    #                    else:
    #                        sublabel_tups_to_iter.append([(sub_gl, None)])  # just a single element
    #
    #                for sublabel_tups in _itertools.product(*sublabel_tups_to_iter):
    #                    sublabels = []  # the sub-labels of the overall operation label to add
    #                    outcomes = []  # the outcome tuple associated with this overall label
    #                    for sub_gl, inst_el_lbl in sublabel_tups:
    #                        if inst_el_lbl is not None:
    #                            sublabels.append(_Label(sub_gl.name + "_" + inst_el_lbl, sub_gl.sslbls))
    #                            outcomes.append(inst_el_lbl)
    #                        else:
    #                            sublabels.append(sub_gl)
    #
    #                    simplified_el_lbl = _Label(sublabels)
    #                    simplified_el_outcomes = tuple(outcomes)
    #                    process(s[0:i] + _cir.Circuit((simplified_el_lbl,)) + s[i + 1:],
    #                            elabels, observed_outcomes, el_inds_to_outcomes,
    #                            op_outcomes + simplified_el_outcomes, i + 1)
    #                break
    #
    #        else:  # no instruments -- add "raw" circuit s
    #            if s in raw_elabels_dict:
    #                assert(op_outcomes == raw_opOutcomes_dict[s])  # DEBUG
    #                #if action == "add":
    #                od = raw_elabels_dict[s]  # ordered dict
    #                for elabel in elabels:
    #                    outcome_tup = op_outcomes + (_gt.effect_label_to_outcome(elabel),)
    #                    if (observed_outcomes is not None) and \
    #                       (outcome_tup not in observed_outcomes): continue
    #                    # don't add elabels we don't observe
    #
    #                    elabel_indx = od.get(elabel, None)
    #                    if elabel is None:
    #                        # although we've seen this raw string, we haven't
    #                        # seen elabel yet - add it at end
    #                        elabel_indx = len(od)
    #                        od[elabel] = elabel_indx
    #
    #                    #Link the current iParent to this index (even if it was already going to be computed)
    #                    el_inds_to_outcomes[(s, elabel_indx)] = outcome_tup
    #            else:
    #                # Note: store elements of raw_elabels_dict as dicts for
    #                # now, for faster lookup during "index" mode
    #                outcome_tuples = [op_outcomes + (_gt.effect_label_to_outcome(x),) for x in elabels]
    #
    #                if observed_outcomes is not None:
    #                    # only add els of `elabels` corresponding to observed data (w/indexes starting at 0)
    #                    elabel_dict = _collections.OrderedDict(); ist = 0
    #                    for elabel, outcome_tup in zip(elabels, outcome_tuples):
    #                        if outcome_tup in observed_outcomes:
    #                            elabel_dict[elabel] = ist
    #                            el_inds_to_outcomes[(s, ist)] = outcome_tup
    #                            ist += 1
    #                else:
    #                    # add all els of `elabels` (w/indexes starting at 0)
    #                    elabel_dict = _collections.OrderedDict([
    #                        (elabel, i) for i, elabel in enumerate(elabels)])
    #
    #                    for ist, out_tup in enumerate(outcome_tuples):  # ist = spamtuple index
    #                        # element index is given by (parent_circuit, spamtuple_index) tuple
    #                        el_inds_to_outcomes[(s, ist)] = out_tup
    #                        # Note: works even if `i` already exists - doesn't reorder keys then
    #
    #                raw_elabels_dict[s] = elabel_dict
    #                raw_opOutcomes_dict[s] = op_outcomes  # DEBUG
    #
    #    #Begin actual processing work:
    #
    #    # Step1: recursively populate raw_elabels_dict,
    #    #        raw_opOutcomes_dict, and elIndsToOutcomesByParent
    #    resolved_circuits = [resolve_elabels(c, dsc) for c, dsc in zip(circuits, ds_circuits)]
    #    for iParent, ((opstr, elabels), orig_circuit, orig_dscircuit) in enumerate(zip(resolved_circuits,
    #                                                                                   circuits, ds_circuits)):
    #        elIndsToOutcomesByParent[iParent] = _collections.OrderedDict()
    #        oouts = None if (dataset is None) else set(dataset[orig_dscircuit].outcomes)
    #        process(opstr, elabels, oouts, elIndsToOutcomesByParent[iParent])
    #
    #    # Step2: fill raw_offsets dictionary
    #    off = 0
    #    for raw_str, elabels in raw_elabels_dict.items():
    #        raw_offsets[raw_str] = off; off += len(elabels)
    #    nTotElements = off
    #
    #    # Step3: split elIndsToOutcomesByParent into
    #    #        elIndicesByParent and outcomesByParent
    #    for iParent, elIndsToOutcomes in elIndsToOutcomesByParent.items():
    #        elIndicesByParent[iParent] = []
    #        outcomesByParent[iParent] = []
    #        for (raw_str, rel_elabel_indx), outcomes in elIndsToOutcomes.items():
    #            elIndicesByParent[iParent].append(raw_offsets[raw_str] + rel_elabel_indx)
    #            outcomesByParent[iParent].append(outcomes)
    #        elIndicesByParent[iParent] = _slct.list_to_slice(elIndicesByParent[iParent], array_ok=True)
    #
    #    #Step3b: convert elements of raw_elabels_dict from OrderedDicts
    #    # to lists now that we don't need to use them for lookups anymore.
    #    for s in list(raw_elabels_dict.keys()):
    #        raw_elabels_dict[s] = list(raw_elabels_dict[s].keys())
    #
    #    #Step4: change lists/slices -> index arrays for user convenience
    #    elIndicesByParent = _collections.OrderedDict(
    #        [(k, (v if isinstance(v, slice) else _np.array(v, _np.int64)))
    #         for k, v in elIndicesByParent.items()])
    #
    #    ##DEBUG: SANITY CHECK
    #    #if len(circuits) > 1:
    #    #    for k,opstr in enumerate(circuits):
    #    #        _,outcomes_k = self.simplify_circuit(opstr)
    #    #        nIndices = _slct.length(elIndicesByParent[k]) if isinstance(elIndicesByParent[k], slice) \
    #    #                      else len(elIndicesByParent[k])
    #    #        assert(len(outcomes_k) == nIndices)
    #    #        assert(outcomes_k == outcomesByParent[k])
    #
    #    #print("Model.simplify debug:")
    #    #print("input = ",'\n'.join(["%d: %s" % (i,repr(c)) for i,c in enumerate(circuits)]))
    #    #print("raw_dict = ", raw_spamTuples_dict)
    #    #print("elIndices = ", elIndicesByParent)
    #    #print("outcomes = ", outcomesByParent)
    #    #print("total els = ",nTotElements)
    #
    #    return (raw_elabels_dict, elIndicesByParent,
    #            outcomesByParent, nTotElements)
    #
    #def simplify_circuit(self, circuit, dataset=None):
    #    """
    #    Simplifies a single :class:`Circuit`.
    #
    #    Parameters
    #    ----------
    #    circuit : Circuit
    #        The circuit to simplify
    #
    #    dataset : DataSet, optional
    #        If not None, restrict what is simplified to only those
    #        probabilities corresponding to non-zero counts (observed
    #        outcomes) in this data set.
    #
    #    Returns
    #    -------
    #    raw_elabels_dict : collections.OrderedDict
    #        A dictionary whose keys are simplified circuits (containing just
    #        "simplified" gates, i.e. not instruments) that include preparation
    #        labels but no measurement (POVM). Values are lists of simplified
    #        effect labels, each label corresponds to a single "final element" of
    #        the computation, e.g. a probability.  The ordering is important - and
    #        is why this needs to be an ordered dictionary - when the lists of tuples
    #        are concatenated (by key) the resulting tuple orderings corresponds to
    #        the final-element axis of an output array that is being filled (computed).
    #    outcomes : list
    #        A list of outcome labels (an outcome label is a tuple
    #        of POVM-effect and/or instrument-element labels), corresponding to
    #        the final elements.
    #    """
    #    raw_dict, _, outcomes, nEls = self.simplify_circuits([circuit], dataset)
    #    assert(len(outcomes[0]) == nEls)
    #    return raw_dict, outcomes[0]

    def circuit_outcomes(self, circuit):
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
        outcomes = circuit.expand_instruments_and_separate_povm(self)  # dict w/keys=sep-povm-circuits, vals=outcomes
        return tuple(_itertools.chain(*outcomes.values()))  # concatenate outputs from all sep-povm-circuits

    def split_circuit(self, circuit, erroron=('prep', 'povm')):
        """
        Splits a circuit into prep_layer + op_layers + povm_layer components.

        If `circuit` does not contain a prep label or a
        povm label a default label is returned if one exists.

        Parameters
        ----------
        circuit : Circuit
            A circuit, possibly beginning with a state preparation
            label and ending with a povm label.

        erroron : tuple of {'prep','povm'}
            A ValueError is raised if a preparation or povm label cannot be
            resolved when 'prep' or 'povm' is included in 'erroron'.  Otherwise
            `None` is returned in place of unresolvable labels.  An exception
            is when this model has no preps or povms, in which case `None`
            is always returned and errors are never raised, since in this
            case one usually doesn't expect to use the Model to compute
            probabilities (e.g. in germ selection).
    
        Returns
        -------
        prepLabel : str or None
        opsOnlyString : Circuit
        povmLabel : str or None
        """
        if len(circuit) > 0 and self._is_primitive_prep_layer_lbl(circuit[0]):
            prep_lbl = circuit[0]
            circuit = circuit[1:]
        elif self._default_primitive_prep_layer_lbl() is not None:
            prep_lbl = self._default_primitive_prep_layer_lbl()
        else:
            if 'prep' in erroron and self._has_primitive_preps():
                raise ValueError("Cannot resolve state prep in %s" % circuit)
            else: prep_lbl = None

        if len(circuit) > 0 and self._is_primitive_povm_layer_lbl(circuit[-1]):
            povm_lbl = circuit[-1]
            circuit = circuit[:-1]
        elif self._default_primitive_povm_layer_lbl(circuit.line_labels) is not None:
            povm_lbl = self._default_primitive_povm_layer_lbl(circuit.line_labels)
        else:
            if 'povm' in erroron and self._has_primitive_povms():
                raise ValueError("Cannot resolve POVM in %s" % str(circuit))
            else: povm_lbl = None

        return prep_lbl, circuit, povm_lbl

    def complete_circuit(self, circuit):
        """
        Adds any implied preparation or measurement layers to `circuit`

        Converts `circuit` into a "complete circuit", where the first (0-th)
        layer is a state preparation and the final layer is a measurement (POVM) layer.

        Parameters
        ----------
        circuit : Circuit
            Circuit to act on.

        Returns
        -------
        Circuit
            Possibly the same object as `circuit`, if no additions are needed.
        """
        prep_lbl_to_prepend = None
        povm_lbl_to_append = None

        if len(circuit) == 0 or not self._is_primitive_prep_layer_lbl(circuit[0]):
            prep_lbl_to_prepend = self._default_primitive_prep_layer_label()
            if prep_lbl_to_prepend is None:
                raise ValueError(f"Missing state prep in {circuit.str} and there's no default!")
        if len(circuit) == 0 or not self._is_primitive_povm_layer_lbl(circuit[-1]):
            povm_lbl_to_append = self._default_primitive_povm_layer_label()
            if povm_lbl_to_append is None:
                raise ValueError(f"Missing POVM in {circuit.str} and there's no default!")

        if prep_lbl_to_prepend or povm_lbl_to_append:
            circuit = circuit.copy(editable=True)
            if prep_lbl_to_prepend: circuit.insert_layer(prep_lbl_to_prepend, 0)
            if povm_lbl_to_append: circuit.insert_layer(povm_lbl_to_append, len(circuit))
            circuit.done_editing()

        return circuit

    # ---- Operation container interface ----
    # These functions allow oracle access to whether a label of a given type
    # "exists" (or can be created by) this model.

    # Support notion of "primitive" *layer* operations, which are
    # stored somewhere in this model (and so don't need to be cached)
    # and represent the fundamental building blocks of other layer operations.
    # "Primitive" layers are used to <TODO>

    # These properties should return the keys of an OrdereDict,
    #  to be used as an ordered set.
    @property
    def _primitive_prep_labels(self):
        raise NotImplementedError("Derived classes must implement this!")

    @property
    def _primitive_povm_labels(self):
        raise NotImplementedError("Derived classes must implement this!")

    @property
    def _primitive_op_labels(self):
        raise NotImplementedError("Derived classes must implement this!")

    @property
    def _primitive_instruments_labels(self):
        raise NotImplementedError("Derived classes must implement this!")

    # These are the public properties that return tuples
    @property
    def primitive_prep_labels(self):
        return tuple(self._primitive_prep_labels)

    @property
    def primitive_povm_labels(self):
        return tuple(self._primitive_povm_labels)

    @property
    def primitive_op_labels(self):
        return tuple(self._primitive_op_labels)

    @property
    def primitive_instruments_labels(self):
        return tuple(self._primitive_instrument_labels)

    def _is_primitive_prep_layer_lbl(self, lbl):
        """
        Whether `lbl` is a valid state prep label (returns boolean)

        Parameters
        ----------
        lbl : Label
            The label to test.

        Returns
        -------
        bool
        """
        return lbl in self._primitive_prep_labels

    def _is_primitive_povm_layer_lbl(self, lbl):
        """
        Whether `lbl` is a valid POVM label (returns boolean)

        Parameters
        ----------
        lbl : Label
            The label to test.

        Returns
        -------
        bool
        """
        return lbl in self._primitive_povm_labels

    def _is_primitive_op_layer_lbl(self, lbl):
        """
        Whether `lbl` is a valid operation label (returns boolean)

        Parameters
        ----------
        lbl : Label
            The label to test.

        Returns
        -------
        bool
        """
        return lbl in self._primitive_op_labels

    def _is_primitive_instrument_layer_lbl(self, lbl):
        """
        Whether `lbl` is a valid instrument label (returns boolean)

        Parameters
        ----------
        lbl : Label
            The label to test.

        Returns
        -------
        bool
        """
        return lbl in self._primitive_instrument_labels

    def _default_primitive_prep_layer_lbl(self):
        """
        Gets the default state prep label.

        This is often used when a circuit is specified without a preparation layer.
        Returns `None` if there is no default and one *must* be specified.

        Returns
        -------
        Label or None
        """
        if len(self._primitive_prep_labels) == 1:
            return next(iter(self._primitive_prep_labels))
        else:
            return None

    def _default_primitive_povm_layer_lbl(self, sslbls):
        """
        Gets the default POVM label.

        This is often used when a circuit  is specified without an ending POVM layer.
        Returns `None` if there is no default and one *must* be specified.

        Parameters
        ----------
        sslbls : tuple or None
            The state space labels being measured, and for which a default POVM is desired.

        Returns
        -------
        Label or None
        """
        if len(self._primitive_povm_labels) == 1:
            return next(iter(self._primitive_povm_labels))
        else:
            return None

    def _has_primitive_preps(self):
        """
        Whether this model contains any state preparations.

        Returns
        -------
        bool
        """
        return len(self._primitive_prep_labels) > 0

    def _has_primitive_povms(self):
        """
        Whether this model contains any POVMs (measurements).

        Returns
        -------
        bool
        """
        return len(self._primitive_povm_labels) > 0

    def _effect_labels_for_povm(self, povm_lbl):
        """
        Gets the effect labels corresponding to the possible outcomes of POVM label `povm_lbl`.

        Parameters
        ----------
        povm_lbl : Label
            POVM label.

        Returns
        -------
        list
            A list of strings which label the POVM outcomes.
        """
        raise NotImplementedError("Derived classes must implement this!")

    def _member_labels_for_instrument(self, inst_lbl):
        """
        Get the member labels corresponding to the possible outcomes of the instrument labeled by `inst_lbl`.

        Parameters
        ----------
        inst_lbl : Label
            Instrument label.

        Returns
        -------
        list
            A list of strings which label the instrument members.
        """
        raise NotImplementedError("Derived classes must implement this!")

    # END operation container interface functions

    def circuit_layer_operator(self, layerlbl, typ="auto"):
        """
        Construct or retrieve the operation associated with a circuit layer.

        Parameters
        ----------
        layerlbl : Label
            The circuit-layer label to construct an operation for.

        typ : {'op','prep','povm','auto'}
            The type of layer `layerlbl` refers to: `'prep'` is for state
            preparation (only at the beginning of a circuit), `'povm'` is for
            a measurement: a POVM or effect label (only at the end of a circuit),
            and `'op'` is for all other "middle" circuit layers.

        Returns
        -------
        LinearOperator or SPAMVec
        """
        fns = {'op': self._layer_rules.operation_layer_operator,
               'prep': self._layer_rules.prep_layer_operator,
               'povm': self._layer_rules.povm_layer_operator}
        if typ == 'auto':
            for fn in fns:
                try:
                    return fn(self, layerlbl, self._layerop_cache)
                except KeyError: pass  # Indicates failure to create op: try next type
            raise ValueError(f"Cannot create operator for non-primitive circuit layer: {layerlbl}")
        else:
            return fns[typ](self, layerlbl, self._layerop_cache)

    def _reinit_layerop_cache(self):
        """ Called when parameter vector structure changes and self._layerop_cache should be cleared & re-initialized """
        self._layerop_cache.clear()

    def probs(self, circuit, clip_to=None, time=None):
        """
        Construct a dictionary containing the outcome probabilities of `circuit`.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        clip_to : 2-tuple, optional
            (min,max) to clip probabilities to if not None.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        Returns
        -------
        probs : dictionary
            A dictionary such that
            probs[SL] = pr(SL,circuit,clip_to)
            for each spam label (string) SL.
        """
        return self._fwdsim().probs(self.simplify_circuit(circuit), clip_to, time)

#    def dprobs(self, circuit, return_pr=False, clip_to=None):
#        """
#        Construct a dictionary containing the outcome probability derivatives of `circuit`.
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
#        Returns
#        -------
#        dprobs : dictionary
#            A dictionary such that
#            dprobs[SL] = dpr(SL,circuit,gates,G0,SPAM,SP0,return_pr,clip_to)
#            for each spam label (string) SL.
#        """
#        return self._fwdsim().dprobs(self.simplify_circuit(circuit),
#                                     return_pr, clip_to)
#
#    def hprobs(self, circuit, return_pr=False, return_deriv=False, clip_to=None):
#        """
#        Construct a dictionary containing the outcome probability 2nd derivatives of `circuit`.
#
#        Parameters
#        ----------
#        circuit : Circuit or tuple of operation labels
#            The sequence of operation labels specifying the circuit.
#
#        return_pr : bool, optional
#            when set to True, additionally return the probabilities.
#
#        return_deriv : bool, optional
#            when set to True, additionally return the derivatives of the
#            probabilities.
#
#        clip_to : 2-tuple, optional
#            (min,max) to clip returned probability to if not None.
#            Only relevant when return_pr == True.
#
#        Returns
#        -------
#        hprobs : dictionary
#            A dictionary such that
#            hprobs[SL] = hpr(SL,circuit,gates,G0,SPAM,SP0,return_pr,return_deriv,clip_to)
#            for each spam label (string) SL.
#        """
#        return self._fwdsim().hprobs(self.simplify_circuit(circuit),
#                                     return_pr, return_deriv, clip_to)
#
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
#
#        # Let np = # param groups, so 1 <= np <= num_params, size of each param group = num_params/np
#        # Let ng = # circuit groups == # subtrees, so 1 <= ng <= max_split_num; size of each group = size of
#        #          corresponding subtree
#        # With nprocs processors, split into Ng comms of ~nprocs/Ng procs each.  These comms are each assigned some
#        #  number of circuit groups, where their ~nprocs/Ng processors are used to partition the np param
#        #  groups. Note that 1 <= Ng <= min(ng,nprocs).
#        # Notes:
#        #  - making np or ng > nprocs can be useful for saving memory.  Raising np saves *Jacobian* and *Hessian*
#        #     function memory without evaltree overhead, and I think will typically be preferred over raising
#        #     ng which will also save Product function memory but will incur evaltree overhead.
#        #  - any given CPU will be running a *single* (ng-index,np-index) pair at any given time, and so many
#        #     memory estimates only depend on ng and np, not on Ng.  (The exception is when a routine *gathers*
#        #     the end results from a divided computation.)
#        #  - "circuits" distribute_method: never distribute num_params (np == 1, Ng == nprocs always).
#        #     Choose ng such that ng >= nprocs, mem_estimate(ng,np=1) < mem_limit, and ng % nprocs == 0 (ng % Ng == 0).
#        #  - "deriv" distribute_method: if possible, set ng=1, nprocs <= np <= num_params, Ng = 1 (np % nprocs == 0?)
#        #     If memory constraints don't allow this, set np = num_params, Ng ~= nprocs/num_params (but Ng >= 1),
#        #     and ng set by mem_estimate and ng % Ng == 0 (so comms are kept busy)
#        #
#        # find ng, np, Ng such that:
#        # - mem_estimate(ng,np,Ng) < mem_limit
#        # - full cpu usage:
#        #       - np*ng >= nprocs (all procs used)
#        #       - ng % Ng == 0 (all subtree comms kept busy)
#        #     -nice, but not essential:
#        #       - num_params % np == 0 (each param group has same size)
#        #       - np % (nprocs/Ng) == 0 would be nice (all procs have same num of param groups to process)
#
#        printer = _VerbosityPrinter.create_printer(verbosity, comm)
#
#        nprocs = 1 if comm is None else comm.Get_size()
#        num_params = self.num_params()
#        evt_cache = {}  # cache of eval trees based on # min subtrees, to avoid re-computation
#        C = 1.0 / (1024.0**3)
#        calc = self._fwdsim()
#
#        bNp2Matters = ("bulk_fill_hprobs" in subcalls) or ("bulk_hprobs_by_block" in subcalls)
#
#        if mem_limit is not None:
#            if mem_limit <= 0:
#                raise MemoryError("Attempted evaltree generation "
#                                  "w/memlimit = %g <= 0!" % mem_limit)
#            printer.log("Evaltree generation (%s) w/mem limit = %.2fGB"
#                        % (distribute_method, mem_limit * C))
#
#        def mem_estimate(n_groups, np1, np2, n_comms, fast_cache_size=False, verb=0, cache_size=None):
#            """ Returns a memory estimate based on arguments """
#            tm = _time.time()
#
#            nFinalStrs = int(round(len(circuit_list) / n_groups))  # may not need to be an int...
#
#            if cache_size is None:
#                #Get cache size
#                if not fast_cache_size:
#                    #Slower (but more accurate way)
#                    if n_groups not in evt_cache:
#                        evt_cache[n_groups] = self.bulk_evaltree(
#                            circuit_list, min_subtrees=n_groups, num_subtree_comms=n_comms,
#                            dataset=dataset, verbosity=printer)
#                        # FUTURE: make a _bulk_evaltree_presimplified version that takes simplified
#                        # circuits as input so don't have to re-simplify every time we hit this line.
#                    cache_size = max([s.cache_size() for s in evt_cache[n_groups][0].sub_trees()])
#                    nFinalStrs = max([s.num_final_circuits() for s in evt_cache[n_groups][0].sub_trees()])
#                else:
#                    #heuristic (but fast)
#                    cache_size = calc._estimate_cache_size(nFinalStrs)
#
#            mem = calc.estimate_memory_usage(subcalls, cache_size, n_groups, n_comms, np1, np2, nFinalStrs)
#
#            if verb == 1:
#                if (not fast_cache_size):
#                    fast_estimate = calc.estimate_memory_usage(
#                        subcalls, cache_size, n_groups, n_comms, np1, np2, nFinalStrs)
#                    fc_est_str = " (%.2fGB fc)" % (fast_estimate * C)
#                else: fc_est_str = ""
#
#                printer.log(" mem(%d subtrees, %d,%d param-grps, %d proc-grps)"
#                            % (n_groups, np1, np2, n_comms) + " in %.0fs = %.2fGB%s"
#                            % (_time.time() - tm, mem * C, fc_est_str))
#            elif verb == 2:
#                wrtLen1 = (num_params + np1 - 1) // np1  # ceiling(num_params / np1)
#                wrtLen2 = (num_params + np2 - 1) // np2  # ceiling(num_params / np2)
#                nSubtreesPerProc = (n_groups + n_comms - 1) // n_comms  # ceiling(n_groups / n_comms)
#                printer.log(" Memory estimate = %.2fGB" % (mem * C)
#                            + " (cache=%d, wrtLen1=%d, wrtLen2=%d, subsPerProc=%d)."
#                            % (cache_size, wrtLen1, wrtLen2, nSubtreesPerProc))
#                #printer.log("  subcalls = %s" % str(subcalls))
#                #printer.log("  cache_size = %d" % cache_size)
#                #printer.log("  wrtLen = %d" % wrtLen)
#                #printer.log("  nSubtreesPerProc = %d" % nSubtreesPerProc)
#
#            return mem
#
#        if distribute_method == "default":
#            distribute_method = calc.default_distribute_method()
#
#        if distribute_method == "circuits":
#            Nstrs = len(circuit_list)
#            np1 = 1; np2 = 1; Ng = min(nprocs, Nstrs)
#            ng = Ng
#            if mem_limit is not None:
#                #Increase ng in amounts of Ng (so ng % Ng == 0).  Start
#                # with fast cache_size computation then switch to slow
#                while mem_estimate(ng, np1, np2, Ng, False) > mem_limit:
#                    ng += Ng
#                    if ng >= Nstrs:
#                        # even "maximal" splitting (num trees == num strings)
#                        # won't help - see if we can squeeze the this maximally-split tree
#                        # to have zero cachesize
#                        if Nstrs not in evt_cache:
#                            mem_estimate(Nstrs, np1, np2, Ng, verb=1)
#                        if hasattr(evt_cache[Nstrs], "squeeze") and \
#                           mem_estimate(Nstrs, np1, np2, Ng, cache_size=0) <= mem_limit:
#                            evt_cache[Nstrs].squeeze(0)  # To get here, need to use higher-dim models
#                        else:
#                            raise MemoryError("Cannot split or squeeze tree to achieve memory limit")
#
#                estimate = mem_estimate(ng, np1, np2, Ng, verb=1)
#                while estimate > mem_limit:
#                    ng += Ng; _next = mem_estimate(ng, np1, np2, Ng, verb=1)
#                    if(_next >= estimate): raise MemoryError("Not enough memory: splitting unproductive")
#                    estimate = _next
#
#                    #Note: could do these while loops smarter, e.g. binary search-like?
#                    #  or assume mem_estimate scales linearly in ng? E.g:
#                    #     if mem_limit < estimate:
#                    #         reductionFactor = float(estimate) / float(mem_limit)
#                    #         maxTreeSize = int(nstrs / reductionFactor)
#            else:
#                mem_estimate(ng, np1, np2, Ng)  # to compute & cache final EvalTree
#
#        elif distribute_method == "deriv":
#
#            def set_ng(desired_ng):
#                """ Set Ng, the number of subTree processor groups, such
#                    that Ng divides nprocs evenly or vice versa. """
#                if desired_ng >= nprocs:
#                    return nprocs * int(_np.ceil(1. * desired_ng / nprocs))
#                else:
#                    fctrs = sorted(_mt.prime_factors(nprocs)); i = 1
#                    if int(_np.ceil(desired_ng)) in fctrs:
#                        return int(_np.ceil(desired_ng))  # we got lucky
#                    while _np.product(fctrs[0:i]) < desired_ng: i += 1
#                    return _np.product(fctrs[0:i])
#
#            ng = Ng = 1
#            if bNp2Matters:
#                if nprocs > num_params**2:
#                    np1 = np2 = max(num_params, 1)
#                    ng = Ng = set_ng(nprocs / max(num_params**2, 1))  # Note floating-point division
#                elif nprocs > num_params:
#                    np1 = max(num_params, 1)
#                    np2 = int(_np.ceil(nprocs / max(num_params, 1)))
#                else:
#                    np1 = nprocs; np2 = 1
#            else:
#                np2 = 1
#                if nprocs > num_params:
#                    np1 = max(num_params, 1)
#                    ng = Ng = set_ng(nprocs / max(num_params, 1))
#                else:
#                    np1 = nprocs
#
#            if mem_limit is not None:
#
#                ok = False
#                if (not ok) and np1 < num_params:
#                    #First try to decrease mem consumption by increasing np1
#                    mem_estimate(ng, np1, np2, Ng, verb=1)  # initial estimate (to screen)
#                    for n in range(np1, num_params + 1, nprocs):
#                        if mem_estimate(ng, n, np2, Ng) < mem_limit:
#                            np1 = n; ok = True; break
#                    else: np1 = num_params
#
#                if (not ok) and bNp2Matters and np2 < num_params:
#                    #Next try to decrease mem consumption by increasing np2
#                    for n in range(np2, num_params + 1):
#                        if mem_estimate(ng, np1, n, Ng) < mem_limit:
#                            np2 = n; ok = True; break
#                    else: np2 = num_params
#
#                if not ok:
#                    #Finally, increase ng in amounts of Ng (so ng % Ng == 0).  Start
#                    # with fast cache_size computation then switch to slow
#                    while mem_estimate(ng, np1, np2, Ng, True) > mem_limit: ng += Ng
#                    estimate = mem_estimate(ng, np1, np2, Ng, verb=1)
#                    while estimate > mem_limit:
#                        ng += Ng; _next = mem_estimate(ng, np1, np2, Ng, verb=1)
#                        if _next >= estimate:
#                            raise MemoryError("Not enough memory: splitting unproductive")
#                        estimate = _next
#            else:
#                mem_estimate(ng, np1, np2, Ng)  # to compute & cache final EvalTree
#
#        elif distribute_method == "balanced":
#            # try to minimize "unbalanced" procs
#            #np = gcf(num_params, nprocs)
#            #ng = Ng = max(nprocs / np, 1)
#            #if mem_limit is not None:
#            #    while mem_estimate(ng,np1,np2,Ng) > mem_limit: ng += Ng #so ng % Ng == 0
#            raise NotImplementedError("balanced distribution still todo")
#
#        # Retrieve final EvalTree (already computed from estimates above)
#        assert (ng in evt_cache), "Tree Caching Error"
#        evt, lookup, outcome_lookup = evt_cache[ng]
#        evt.distribution['numSubtreeComms'] = Ng
#
#        paramBlkSize1 = num_params / np1
#        paramBlkSize2 = num_params / np2  # the *average* param block size
#        # (in general *not* an integer), which ensures that the intended # of
#        # param blocks is communicatd to gsCalc.py routines (taking ceiling or
#        # floor can lead to inefficient MPI distribution)
#
#        printer.log("Created evaluation tree with %d subtrees.  " % ng
#                    + "Will divide %d procs into %d (subtree-processing)" % (nprocs, Ng))
#        if bNp2Matters:
#            printer.log(" groups of ~%d procs each, to distribute over " % (nprocs / Ng)
#                        + "(%d,%d) params (taken as %d,%d param groups of ~%d,%d params)."
#                        % (num_params, num_params, np1, np2, paramBlkSize1, paramBlkSize2))
#        else:
#            printer.log(" groups of ~%d procs each, to distribute over " % (nprocs / Ng)
#                        + "%d params (taken as %d param groups of ~%d params)."
#                        % (num_params, np1, paramBlkSize1))
#
#        if mem_limit is not None:
#            mem_estimate(ng, np1, np2, Ng, False, verb=2)  # print mem estimate details
#
#        if (comm is None or comm.Get_rank() == 0) and evt.is_split():
#            if printer.verbosity >= 2: evt._print_analysis()
#
#        if np1 == 1:  # (paramBlkSize == num_params)
#            paramBlkSize1 = None  # == all parameters, and may speed logic in dprobs, etc.
#        else:
#            if comm is not None:
#                blkSizeTest = comm.bcast(paramBlkSize1, root=0)
#                assert(abs(blkSizeTest - paramBlkSize1) < 1e-3)
#                #all procs should have *same* paramBlkSize1
#
#        if np2 == 1:  # (paramBlkSize == num_params)
#            paramBlkSize2 = None  # == all parameters, and may speed logic in hprobs, etc.
#        else:
#            if comm is not None:
#                blkSizeTest = comm.bcast(paramBlkSize2, root=0)
#                assert(abs(blkSizeTest - paramBlkSize2) < 1e-3)
#                #all procs should have *same* paramBlkSize2
#
#        #Prepare any computationally intensive preparation
#        calc.bulk_prep_probs(evt, comm, mem_limit)
#
#        return evt, paramBlkSize1, paramBlkSize2, lookup, outcome_lookup
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
#        elIndices : collections.OrderedDict
#            A dictionary whose keys are integer indices into `circuit_list` and
#            whose values are slices and/or integer-arrays into the space/axis of
#            final elements returned by the 'bulk fill' routines.  Thus, to get the
#            final elements corresponding to `circuits[i]`, use
#            `filledArray[ elIndices[i] ]`.
#        outcomes : collections.OrderedDict
#            A dictionary whose keys are integer indices into `circuit_list` and
#            whose values are lists of outcome labels (an outcome label is a tuple
#            of POVM-effect and/or instrument-element labels).  Thus, to obtain
#            what outcomes the i-th circuit's final elements
#            (`filledArray[ elIndices[i] ]`)  correspond to, use `outcomes[i]`.
#        """
#        tm = _time.time()
#        printer = _VerbosityPrinter.create_printer(verbosity)
#
#        simplified_circuits, elIndices, outcomes, nEls = \
#            self.simplify_circuits(circuit_list, dataset)
#
#        evalTree = self._fwdsim().construct_evaltree(simplified_circuits, num_subtree_comms)
#
#        printer.log("bulk_evaltree: created initial tree (%d strs) in %.0fs" %
#                    (len(circuit_list), _time.time() - tm)); tm = _time.time()
#
#        if max_tree_size is not None:
#            elIndices = evalTree.split(elIndices, max_tree_size, None, printer - 1)  # won't split if unnecessary
#
#        if min_subtrees is not None:
#            if not evalTree.is_split() or len(evalTree.sub_trees()) < min_subtrees:
#                evalTree.original_index_lookup = None  # reset this so we can re-split TODO: cleaner
#                elIndices = evalTree.split(elIndices, None, min_subtrees, printer - 1)
#                if max_tree_size is not None and \
#                        any([len(sub) > max_tree_size for sub in evalTree.sub_trees()]):
#                    _warnings.warn("Could not create a tree with min_subtrees=%d" % min_subtrees
#                                   + " and max_tree_size=%d" % max_tree_size)
#                    evalTree.original_index_lookup = None  # reset this so we can re-split TODO: cleaner
#                    elIndices = evalTree.split(elIndices, max_tree_size, None)  # fall back to split for max size
#
#        if max_tree_size is not None or min_subtrees is not None:
#            printer.log("bulk_evaltree: split tree (%d subtrees) in %.0fs"
#                        % (len(evalTree.sub_trees()), _time.time() - tm))
#
#        assert(evalTree.num_final_elements() == nEls)
#        return evalTree, elIndices, outcomes
#
#    def bulk_prep_probs(self, eval_tree, comm=None, mem_limit=None):
#        """
#        Performs initial computation needed for bulk_fill_probs and related calls.
#
#        For example, as computing probability polynomials. This is usually coupled with
#        the creation of an evaluation tree, but is separated from it because this
#        "preparation" may use `comm` to distribute a computationally intensive task.
#
#        Parameters
#        ----------
#        eval_tree : EvalTree
#            The evaluation tree used to define a list of circuits and hold (cache)
#            any computed quantities.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not None, an MPI communicator for distributing the computation
#            across multiple processors.  Distribution is performed over
#            subtrees of `eval_tree` (if it is split).
#
#        mem_limit : int, optional
#            A rough memory limit in bytes which is used to determine resource
#            allocation.
#
#        Returns
#        -------
#        None
#        """
#        return self._fwdsim().bulk_prep_probs(eval_tree, comm, mem_limit)
#
#    def bulk_probs_paths_are_sufficient(self, eval_tree, probs, comm=None, mem_limit=None, verbosity=0):
#        """
#        Only applicable for models with a term-based (path-integral) forward simulator.
#
#        Returns a boolean indicating whether the currently selected paths are able to
#        predict the outcome probabilities of the circuits in `eval_tree` accurately enough,
#        as defined by the simulation-type arguments such as `allowed_perr`.
#
#        Parameters
#        ----------
#        eval_tree : EvalTree
#            The evaluation tree used to define a list of circuits and hold (cache)
#            any computed quantities.
#
#        probs : ndarray, optional
#            A list of the pre-computed probabilities for the circuits in `eval_tree`,
#            as these are needed by some heuristics used to predict errors in the
#            probabilities.  If None, then the probabilities are computed internally.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not None, an MPI communicator for distributing the computation
#            across multiple processors.  Distribution is performed over
#            subtrees of `eval_tree` (if it is split).
#
#        mem_limit : int, optional
#            A rough memory limit in bytes which is used to determine resource
#            allocation.
#
#        verbosity : int, optional
#            Level of information to print to stdout.  0 means none, higher values
#            mean more information.
#
#        Returns
#        -------
#        bool
#        """
#        #print("BULK PROBS NUM TERMGAP FAILURES") #TODO REMOVE
#        fwdsim = self._fwdsim()
#        assert(isinstance(fwdsim, _termfwdsim.TermForwardSimulator)), \
#            "bulk_probs_num_term_failures(...) can only be called on models with a term-based forward simulator!"
#        printer = _VerbosityPrinter.create_printer(verbosity, comm)
#        if probs is None:
#            probs = _np.empty(eval_tree.num_final_elements(), 'd')
#            self.bulk_fill_probs(probs, eval_tree, clip_to=None, comm=comm)
#        return fwdsim.bulk_test_if_paths_are_sufficient(eval_tree, probs, comm, mem_limit, printer)

    def bulk_probs(self, circuit_list, clip_to=None, check=False,
                   comm=None, mem_limit=None, dataset=None, smartc=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuit_list : list of (tuples or Circuits)
            Each element specifies a circuit to compute quantities for.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        check : boolean, optional
            If True, perform extra checks within code to verify correctness,
            generating warnings when checks fail.  Used for testing, and runs
            much slower when True.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of evalTree (if it is split).

        mem_limit : int, optional
            A rough memory limit in bytes which is used to determine processor
            allocation.

        dataset : DataSet, optional
            If not None, restrict what is computed to only those
            probabilities corresponding to non-zero counts (observed
            outcomes) in this data set.

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
        circuit_list = [opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr)
                        for opstr in circuit_list]  # cast to Circuits
        evalTree, _, _, elIndices, outcomes = self.bulk_evaltree_from_resources(
            circuit_list, comm, mem_limit, subcalls=['bulk_fill_probs'],
            dataset=dataset, verbosity=0)  # FUTURE (maybe make verbosity into an arg?)

        return self._fwdsim().bulk_probs(circuit_list, evalTree, elIndices,
                                         outcomes, clip_to, check, comm, smartc)

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
#            `(outcome, dp)`).
#        """
#        circuit_list = [opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr)
#                        for opstr in circuit_list]  # cast to Circuits
#        evalTree, elIndices, outcomes = self.bulk_evaltree(circuit_list, dataset=dataset)
#        return self._fwdsim().bulk_dprobs(circuit_list, evalTree, elIndices,
#                                          outcomes, return_pr, clip_to,
#                                          check, comm, None, wrt_block_size)
#
#    def bulk_hprobs(self, circuit_list, return_pr=False, return_deriv=False,
#                    clip_to=None, check=False, comm=None,
#                    wrt_block_size1=None, wrt_block_size2=None, dataset=None):
#        """
#        Construct a dictionary containing the probability-Hessians for an entire list of circuits.
#
#        Parameters
#        ----------
#        circuit_list : list of (tuples or Circuits)
#            Each element specifies a circuit to compute quantities for.
#
#        return_pr : bool, optional
#            when set to True, additionally return the probabilities.
#
#        return_deriv : bool, optional
#            when set to True, additionally return the probability derivatives.
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
#            across multiple processors.
#
#        wrt_block_size1 : int or float, optional
#            The maximum number of 1st (row) derivatives to compute
#            *products* for simultaneously.  None means compute all requested
#            rows or columns at once.  Set this to non-None to reduce amount of
#            intermediate memory required.
#
#        wrt_block_size2 : int or float, optional
#            The maximum number of 2nd (col) derivatives to compute
#            *products* for simultaneously.  None means compute all requested
#            rows or columns at once.  Set this to non-None to reduce amount of
#            intermediate memory required.
#
#        dataset : DataSet, optional
#            If not None, restrict what is computed to only those
#            probabilities corresponding to non-zero counts (observed
#            outcomes) in this data set.
#
#        Returns
#        -------
#        hprobs : dictionary
#            A dictionary such that `probs[opstr]` is an ordered dictionary of
#            `(outcome, hp, dp, p)` tuples, where `outcome` is a tuple of labels,
#            `p` is the corresponding probability, `dp` is a 1D array containing
#            the derivative of `p` with respect to each parameter, and `hp` is a
#            2D array containing the Hessian of `p` with respect to each parameter.
#            If `return_pr` if False, then `p` is not included in the tuples.
#            If `return_deriv` if False, then `dp` is not included in the tuples.
#        """
#        circuit_list = [opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr)
#                        for opstr in circuit_list]  # cast to Circuits
#        evalTree, elIndices, outcomes = self.bulk_evaltree(circuit_list, dataset=dataset)
#        return self._fwdsim().bulk_hprobs(circuit_list, evalTree, elIndices,
#                                          outcomes, return_pr, return_deriv,
#                                          clip_to, check, comm, None, None,
#                                          wrt_block_size1, wrt_block_size2)
#
#    def bulk_fill_probs(self, mx_to_fill, eval_tree, clip_to=None, check=False, comm=None):
#        """
#        Compute the outcome probabilities for an entire tree of circuits.
#
#        This routine fills a 1D array, `mx_to_fill` with the probabilities
#        corresponding to the *simplified* circuits found in an evaluation
#        tree, `eval_tree`.  An initial list of (general) :class:`Circuit`
#        objects is *simplified* into a lists of gate-only sequences along with
#        a mapping of final elements (i.e. probabilities) to gate-only sequence
#        and prep/effect pairs.  The evaluation tree organizes how to efficiently
#        compute the gate-only sequences.  This routine fills in `mx_to_fill`, which
#        must have length equal to the number of final elements (this can be
#        obtained by `eval_tree.num_final_elements()`.  To interpret which elements
#        correspond to which strings and outcomes, you'll need the mappings
#        generated when the original list of `Circuits` was simplified.
#
#        Parameters
#        ----------
#        mx_to_fill : numpy ndarray
#            an already-allocated 1D numpy array of length equal to the
#            total number of computed elements (i.e. eval_tree.num_final_elements())
#
#        eval_tree : EvalTree
#            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
#            strings to compute the bulk operation on.
#
#        clip_to : 2-tuple, optional
#            (min,max) to clip return value if not None.
#
#        check : boolean, optional
#            If True, perform extra checks within code to verify correctness,
#            generating warnings when checks fail.  Used for testing, and runs
#            much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not None, an MPI communicator for distributing the computation
#            across multiple processors.  Distribution is performed over
#            subtrees of eval_tree (if it is split).
#
#        Returns
#        -------
#        None
#        """
#        return self._fwdsim().bulk_fill_probs(mx_to_fill,
#                                              eval_tree, clip_to, check, comm)
#
#    def bulk_fill_dprobs(self, mx_to_fill, eval_tree, pr_mx_to_fill=None, clip_to=None,
#                         check=False, comm=None, wrt_block_size=None,
#                         profiler=None, gather_mem_limit=None):
#        """
#        Compute the outcome probability-derivatives for an entire tree of circuits.
#
#        Similar to `bulk_fill_probs(...)`, but fills a 2D array with
#        probability-derivatives for each "final element" of `eval_tree`.
#
#        Parameters
#        ----------
#        mx_to_fill : numpy ndarray
#            an already-allocated ExM numpy array where E is the total number of
#            computed elements (i.e. eval_tree.num_final_elements()) and M is the
#            number of model parameters.
#
#        eval_tree : EvalTree
#            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
#            strings to compute the bulk operation on.
#
#        pr_mx_to_fill : numpy array, optional
#            when not None, an already-allocated length-E numpy array that is filled
#            with probabilities, just like in bulk_fill_probs(...).
#
#        clip_to : 2-tuple, optional
#            (min,max) to clip return value if not None.
#
#        check : boolean, optional
#            If True, perform extra checks within code to verify correctness,
#            generating warnings when checks fail.  Used for testing, and runs
#            much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not None, an MPI communicator for distributing the computation
#            across multiple processors.  Distribution is first performed over
#            subtrees of eval_tree (if it is split), and then over blocks (subsets)
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
#        profiler : Profiler, optional
#            A profiler object used for to track timing and memory usage.
#
#        gather_mem_limit : int, optional
#            A memory limit in bytes to impose upon the "gather" operations
#            performed as a part of MPI processor syncronization.
#
#        Returns
#        -------
#        None
#        """
#        return self._fwdsim().bulk_fill_dprobs(mx_to_fill,
#                                               eval_tree, pr_mx_to_fill, clip_to,
#                                               check, comm, None, wrt_block_size,
#                                               profiler, gather_mem_limit)
#
#    def bulk_fill_hprobs(self, mx_to_fill, eval_tree=None,
#                         pr_mx_to_fill=None, deriv_mx_to_fill=None,
#                         clip_to=None, check=False, comm=None,
#                         wrt_block_size1=None, wrt_block_size2=None,
#                         gather_mem_limit=None):
#        """
#        Compute the outcome probability-Hessians for an entire tree of circuits.
#
#        Similar to `bulk_fill_probs(...)`, but fills a 3D array with
#        probability-Hessians for each "final element" of `eval_tree`.
#
#        Parameters
#        ----------
#        mx_to_fill : numpy ndarray
#            an already-allocated ExMxM numpy array where E is the total number of
#            computed elements (i.e. eval_tree.num_final_elements()) and M1 & M2 are
#            the number of selected gate-set parameters (by wrt_filter1 and wrt_filter2).
#
#        eval_tree : EvalTree
#            given by a prior call to bulk_evaltree.  Specifies the *simplified* gate
#            strings to compute the bulk operation on.
#
#        pr_mx_to_fill : numpy array, optional
#            when not None, an already-allocated length-E numpy array that is filled
#            with probabilities, just like in bulk_fill_probs(...).
#
#        deriv_mx_to_fill : numpy array, optional
#            when not None, an already-allocated ExM numpy array that is filled
#            with probability derivatives.
#
#        clip_to : 2-tuple, optional
#            (min,max) to clip return value if not None.
#
#        check : boolean, optional
#            If True, perform extra checks within code to verify correctness,
#            generating warnings when checks fail.  Used for testing, and runs
#            much slower when True.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not None, an MPI communicator for distributing the computation
#            across multiple processors.  Distribution is first performed over
#            subtrees of eval_tree (if it is split), and then over blocks (subsets)
#            of the parameters being differentiated with respect to (see
#            wrt_block_size).
#
#        wrt_block_size1 : int or float, optional
#            The maximum number of 1st (row) derivatives to compute
#            *products* for simultaneously.  None means compute all requested
#            rows or columns at once.  Set this to non-None to reduce amount of
#            intermediate memory required.
#
#        wrt_block_size2 : int or float, optional
#            The maximum number of 2nd (col) derivatives to compute
#            *products* for simultaneously.  None means compute all requested
#            rows or columns at once.  Set this to non-None to reduce amount of
#            intermediate memory required.
#
#        gather_mem_limit : int, optional
#            A memory limit in bytes to impose upon the "gather" operations
#            performed as a part of MPI processor syncronization.
#
#        Returns
#        -------
#        None
#        """
#        return self._fwdsim().bulk_fill_hprobs(mx_to_fill,
#                                               eval_tree, pr_mx_to_fill, deriv_mx_to_fill, None,
#                                               clip_to, check, comm, None, None,
#                                               wrt_block_size1, wrt_block_size2, gather_mem_limit)
#
#    def bulk_hprobs_by_block(self, eval_tree, wrt_slices_list,
#                             return_dprobs_12=False, comm=None):
#        """
#        An iterator that computes 2nd derivatives of the `eval_tree`'s circuit probabilities column-by-column.
#
#        This routine can be useful when memory constraints make constructing
#        the entire Hessian at once impractical, and one is able to compute
#        reduce results from a single column of the Hessian at a time.  For
#        example, the Hessian of a function of many gate sequence probabilities
#        can often be computed column-by-column from the using the columns of
#        the circuits.
#
#        Parameters
#        ----------
#        eval_tree : EvalTree
#            given by a prior call to bulk_evaltree.  Specifies the circuits
#            to compute the bulk operation on.  This tree *cannot* be split.
#
#        wrt_slices_list : list
#            A list of `(rowSlice,colSlice)` 2-tuples, each of which specify
#            a "block" of the Hessian to compute.  Iterating over the output
#            of this function iterates over these computed blocks, in the order
#            given by `wrt_slices_list`.  `rowSlice` and `colSlice` must by Python
#            `slice` objects.
#
#        return_dprobs_12 : boolean, optional
#            If true, the generator computes a 2-tuple: (hessian_col, d12_col),
#            where d12_col is a column of the matrix d12 defined by:
#            d12[iSpamLabel,iOpStr,p1,p2] = dP/d(p1)*dP/d(p2) where P is is
#            the probability generated by the sequence and spam label indexed
#            by iOpStr and iSpamLabel.  d12 has the same dimensions as the
#            Hessian, and turns out to be useful when computing the Hessian
#            of functions of the probabilities.
#
#        comm : mpi4py.MPI.Comm, optional
#            When not None, an MPI communicator for distributing the computation
#            across multiple processors.  Distribution is performed as in
#            bulk_product, bulk_dproduct, and bulk_hproduct.
#
#        Returns
#        -------
#        block_generator
#            A generator which, when iterated, yields the 3-tuple
#            `(rowSlice, colSlice, hprobs)` or `(rowSlice, colSlice, dprobs12)`
#            (the latter if `return_dprobs_12 == True`).  `rowSlice` and `colSlice`
#            are slices directly from `wrt_slices_list`. `hprobs` and `dprobs12` are
#            arrays of shape K x S x B x B', where:
#
#            - K is the length of spam_label_rows,
#            - S is the number of circuits (i.e. eval_tree.num_final_circuits()),
#            - B is the number of parameter rows (the length of rowSlice)
#            - B' is the number of parameter columns (the length of colSlice)
#
#            If `mx` and `dp` the outputs of :func:`bulk_fill_hprobs`
#            (i.e. args `mx_to_fill` and `deriv_mx_to_fill`), then:
#
#            - `hprobs == mx[:,:,rowSlice,colSlice]`
#            - `dprobs12 == dp[:,:,rowSlice,None] * dp[:,:,None,colSlice]`
#        """
#        return self._fwdsim().bulk_hprobs_by_block(
#            eval_tree, wrt_slices_list,
#            return_dprobs_12, comm)

    def _init_copy(self, copy_into):
        """
        Copies any "tricky" member of this model into `copy_into`, before
        deep copying everything else within a .copy() operation.
        """
        self._clean_paramvec()  # make sure _paramvec is valid before copying (necessary?)
        copy_into._shlp = None  # must be set by a derived-class _init_copy() method
        copy_into._need_to_rebuild = True  # copy will have all gpindices = None, etc.
        copy_into._opcache = {}  # don't copy opcache
        super(OpModel, self)._init_copy(copy_into)

    def copy(self):
        """
        Copy this model.

        Returns
        -------
        Model
            a (deep) copy of this model.
        """
        self._clean_paramvec()  # ensure _paramvec is rebuilt if needed
        if OpModel._pcheck: self._check_paramvec()
        ret = Model.copy(self)
        if OpModel._pcheck: ret._check_paramvec()
        return ret
