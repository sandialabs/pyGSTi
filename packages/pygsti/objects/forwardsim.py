""" Defines the ForwardSimulator calculator class"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import numpy.linalg as _nla
import collections as _collections
import itertools as _itertools

from ..tools import slicetools as _slct
from ..tools import basistools as _bt
from ..tools import matrixtools as _mt
from .profiler import DummyProfiler as _DummyProfiler
from . import spamvec as _sv
from . import operation as _op
from . import labeldicts as _ld

_dummy_profiler = _DummyProfiler()


class ForwardSimulator(object):
    """
    Encapsulates a calculation tool used by model objects to perform product
    and derivatives-of-product calculations.

    This is contained in a class separate from Model to allow for additional
    model classes (e.g. ones which use entirely different -- non-gate-local
    -- parameterizations of operation matrices and SPAM vectors) access to these
    fundamental operations.
    """

    def __init__(self, dim, simplified_op_server, paramvec):
        """
        Construct a new ForwardSimulator object.

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
        self.dim = dim
        self.sos = simplified_op_server

        #Conversion of labels -> integers for speed & C-compatibility
        #self.operation_lookup = { lbl:i for i,lbl in enumerate(gates.keys()) }
        #self.prep_lookup = { lbl:i for i,lbl in enumerate(preps.keys()) }
        #self.effect_lookup = { lbl:i for i,lbl in enumerate(effects.keys()) }
        #
        #self.operationreps = { i:self.operations[lbl].torep() for lbl,i in self.operation_lookup.items() }
        #self.prepreps = { lbl:p.torep('prep') for lbl,p in preps.items() }
        #self.effectreps = { lbl:e.torep('effect') for lbl,e in effects.items() }

        self.paramvec = paramvec
        self.Np = len(paramvec)
        self.evotype = simplified_op_server.get_evotype()

    def to_vector(self):
        """
        Returns the elements of the parent Model vectorized as a 1D array.
        Used for computing finite-difference derivatives.

        Returns
        -------
        numpy array
            The vectorized model parameters.
        """
        return self.paramvec

    def from_vector(self, v, close=False, nodirty=False):
        """
        The inverse of to_vector.  Initializes the Model-like members of this
        calculator based on `v`. Used for computing finite-difference derivatives.
        """
        #Note: this *will* initialize the parent Model's objects too,
        # since only references to preps, effects, and gates are held
        # by the calculator class.  ORDER is important, as elements of
        # POVMs and Instruments rely on a fixed from_vector ordering
        # of their simplified effects/gates.
        self.paramvec = v.copy()  # now self.paramvec is *not* the same as the Model's paramvec
        self.sos.from_vector(v, close, nodirty)  # so don't always want ", nodirty=True)" - we
        # need to set dirty flags so *parent* will re-init it's paramvec...

        #Re-init reps for computation
        #self.operationreps = { i:self.operations[lbl].torep() for lbl,i in self.operation_lookup.items() }
        #self.operationreps = { lbl:g.torep() for lbl,g in gates.items() }
        #self.prepreps = { lbl:p.torep('prep') for lbl,p in preps.items() }
        #self.effectreps = { lbl:e.torep('effect') for lbl,e in effects.items() }

    def propagate(self, state, simplified_circuit, time=None):
        raise NotImplementedError()  # TODO - create an interface for running circuits

    def probs(self, simplified_circuit, clipTo=None, time=None):
        """
        Construct a dictionary containing the probabilities of every spam label
        given a operation sequence.

        Parameters
        ----------
        simplified_circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the operation sequence.
            This is a "simplified" circuit in that it should not contain any
            POVM or Instrument labels (but can have effect or Instrument-member
            labels).

        clipTo : 2-tuple, optional
           (min,max) to clip probabilities to if not None.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        Returns
        -------
        probs : dictionary
            A dictionary such that
            probs[SL] = pr(SL,circuit,clipTo)
            for each spam label (string) SL.
        """
        probs = _ld.OutcomeLabelDict()
        raw_dict, outcomeLbls = simplified_circuit
        iOut = 0  # outcome index

        for raw_circuit, elabels in raw_dict.items():
            # evaluate spamTuples w/same rholabel together
            for pval in self.prs(raw_circuit[0], elabels, raw_circuit[1:], clipTo, False, time):
                probs[outcomeLbls[iOut]] = pval; iOut += 1
        return probs

    def dprobs(self, simplified_circuit, returnPr=False, clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given operation sequence.

        Parameters
        ----------
        simplified_circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the operation sequence.
            This is a "simplified" circuit in that it should not contain any
            POVM or Instrument labels (but can have effect or Instrument-member
            labels).

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that
            dprobs[SL] = dpr(SL,circuit,gates,G0,SPAM,SP0,returnPr,clipTo)
            for each spam label (string) SL.
        """
        dprobs = {}
        raw_dict, outcomeLbls = simplified_circuit
        iOut = 0  # outcome index
        for raw_circuit, elabels in raw_dict.items():
            for elabel in elabels:
                dprobs[outcomeLbls[iOut]] = self.dpr(
                    (raw_circuit[0], elabel), raw_circuit[1:], returnPr, clipTo)
                iOut += 1
        return dprobs

    def hprobs(self, simplified_circuit, returnPr=False, returnDeriv=False, clipTo=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given operation sequence.

        Parameters
        ----------
        simplified_circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the operation sequence.
            This is a "simplified" circuit in that it should not contain any
            POVM or Instrument labels (but can have effect or Instrument-member
            labels).

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        returnDeriv : bool, optional
          when set to True, additionally return the derivatives of the
          probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

        Returns
        -------
        hprobs : dictionary
            A dictionary such that
            hprobs[SL] = hpr(SL,circuit,gates,G0,SPAM,SP0,returnPr,returnDeriv,clipTo)
            for each spam label (string) SL.
        """
        hprobs = {}
        raw_dict, outcomeLbls = simplified_circuit
        iOut = 0  # outcome index
        for raw_circuit, elabels in raw_dict.items():
            for elabel in elabels:
                hprobs[outcomeLbls[iOut]] = self.hpr(
                    (raw_circuit[0], elabel), raw_circuit[1:], returnPr, returnDeriv, clipTo)
                iOut += 1
        return hprobs

    def bulk_probs(self, circuits, evalTree, elIndices, outcomes,
                   clipTo=None, check=False, comm=None, smartc=None):
        """
        Construct a dictionary containing the probabilities
        for an entire list of operation sequences.

        Parameters
        ----------
        circuits : list of Circuits
            The list of (non-simplified) original operation sequences.

        evalTree : EvalTree
            An evalution tree corresponding to `circuits`.

        elIndices : dict
            A dictionary of indices for each original operation sequence.

        outcomes : dict
            A dictionary of outcome labels (string or tuple) for each original
            operation sequence.

        clipTo : 2-tuple, optional
            (min,max) to clip return value if not None.

        check : boolean, optional
            If True, perform extra checks within code to verify correctness,
            generating warnings when checks fail.  Used for testing, and runs
            much slower when True.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.


        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            outcome probabilities whose keys are (tuples of) outcome labels.
        """
        vp = _np.empty(evalTree.num_final_elements(), 'd')
        if smartc:
            smartc.cached_compute(self.bulk_fill_probs, vp, evalTree,
                                  clipTo, check, comm, _filledarrays=(0,))
        else:
            self.bulk_fill_probs(vp, evalTree, clipTo, check, comm)

        ret = _collections.OrderedDict()
        for i, opstr in enumerate(circuits):
            elInds = _slct.indices(elIndices[i]) \
                if isinstance(elIndices[i], slice) else elIndices[i]
            ret[opstr] = _ld.OutcomeLabelDict(
                [(outLbl, vp[ei]) for ei, outLbl in zip(elInds, outcomes[i])])
        return ret

    def bulk_dprobs(self, circuits, evalTree, elIndices, outcomes,
                    returnPr=False, clipTo=None,
                    check=False, comm=None,
                    wrtFilter=None, wrtBlockSize=None):
        """
        Construct a dictionary containing the probability-derivatives
        for an entire list of operation sequences.

        Parameters
        ----------
        circuits : list of Circuits
            The list of (non-simplified) original operation sequences.

        evalTree : EvalTree
            An evalution tree corresponding to `circuits`.

        elIndices : dict
            A dictionary of indices for each original operation sequence.

        outcomes : dict
            A dictionary of outcome labels (string or tuple) for each original
            operation sequence.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

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
          The maximum average number of derivative columns to compute *products*
          for simultaneously.  None means compute all requested columns
          at once.  The  minimum of wrtBlockSize and the size that makes
          maximal use of available processors is used as the final block size.
          This argument must be None if wrtFilter is not None.  Set this to
          non-None to reduce amount of intermediate memory required.


        Returns
        -------
        dprobs : dictionary
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(dp, p)` tuples whose keys are (tuples of) outcome labels,
            where `p` is the corresponding probability, and `dp` is an array
            containing the derivative of `p` with respect to each parameter.
            If `returnPr` is False, then `p` is not included in the tuples
            (so values are just `dp`).
        """
        nElements = evalTree.num_final_elements()
        nDerivCols = self.Np

        vdp = _np.empty((nElements, nDerivCols), 'd')
        vp = _np.empty(nElements, 'd') if returnPr else None

        self.bulk_fill_dprobs(vdp, evalTree,
                              vp, clipTo, check, comm,
                              wrtFilter, wrtBlockSize)

        ret = _collections.OrderedDict()
        for i, opstr in enumerate(circuits):
            elInds = _slct.indices(elIndices[i]) \
                if isinstance(elIndices[i], slice) else elIndices[i]
            if returnPr:
                ret[opstr] = _ld.OutcomeLabelDict(
                    [(outLbl, (vdp[ei], vp[ei])) for ei, outLbl in zip(elInds, outcomes[i])])
            else:
                ret[opstr] = _ld.OutcomeLabelDict(
                    [(outLbl, vdp[ei]) for ei, outLbl in zip(elInds, outcomes[i])])
        return ret

    def bulk_hprobs(self, circuits, evalTree, elIndices, outcomes,
                    returnPr=False, returnDeriv=False, clipTo=None,
                    check=False, comm=None,
                    wrtFilter1=None, wrtFilter2=None,
                    wrtBlockSize1=None, wrtBlockSize2=None):
        """
        Construct a dictionary containing the probability-Hessians
        for an entire list of operation sequences.

        Parameters
        ----------
        circuits : list of Circuits
            The list of (non-simplified) original operation sequences.

        evalTree : EvalTree
            An evalution tree corresponding to `circuits`.

        elIndices : dict
            A dictionary of indices for each original operation sequence.

        outcomes : dict
            A dictionary of outcome labels (string or tuple) for each original
            operation sequence.

        returnPr : bool, optional
          when set to True, additionally return the probabilities.

        returnDeriv : bool, optional
          when set to True, additionally return the probability derivatives.

        clipTo : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when returnPr == True.

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


        Returns
        -------
        hprobs : dictionary
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(hp, dp, p)` tuples whose keys are (tuples of) outcome labels,
            where `p` is the corresponding probability, `dp` is an array
            containing the derivative of `p` with respect to each parameter,
            and `hp` is a 2D array containing the Hessian of `p` with respect
            to each parameter.  If `returnPr` is False, then `p` is not
            included in the tuples.  If `returnDeriv` if False, then `dp` is
            not included in the tuples (if both are false then values are
            just `hp`, and not a tuple).
        """
        nElements = evalTree.num_final_elements()
        nDerivCols1 = self.Np if (wrtFilter1 is None) \
            else len(wrtFilter1)
        nDerivCols2 = self.Np if (wrtFilter2 is None) \
            else len(wrtFilter2)

        vhp = _np.empty((nElements, nDerivCols1, nDerivCols2), 'd')
        vdp1 = _np.empty((nElements, self.Np), 'd') \
            if returnDeriv else None
        vdp2 = vdp1.copy() if (returnDeriv and wrtFilter1 != wrtFilter2) else None
        vp = _np.empty(nElements, 'd') if returnPr else None

        self.bulk_fill_hprobs(vhp, evalTree,
                              vp, vdp1, vdp2, clipTo, check, comm,
                              wrtFilter1, wrtFilter1, wrtBlockSize1, wrtBlockSize2)

        ret = _collections.OrderedDict()
        for i, opstr in enumerate(circuits):
            elInds = _slct.indices(elIndices[i]) \
                if isinstance(elIndices[i], slice) else elIndices[i]
            outcomeQtys = _ld.OutcomeLabelDict()
            for ei, outLbl in zip(elInds, outcomes[i]):
                if returnDeriv:
                    if vdp2 is None:
                        if returnPr: t = (vhp[ei], vdp1[ei], vp[ei])
                        else: t = (vhp[ei], vdp1[ei])
                    else:
                        if returnPr: t = (vhp[ei], vdp1[ei], vdp2[ei], vp[ei])
                        else: t = (vhp[ei], vdp1[ei], vdp2[ei])
                else:
                    if returnPr: t = (vhp[ei], vp[ei])
                    else: t = vhp[ei]
                outcomeQtys[outLbl] = t
            ret[opstr] = outcomeQtys

        return ret

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
        EvalTree
        """
        raise NotImplementedError("construct_evaltree(...) is not implemented!")

    def _setParamBlockSize(self, wrtFilter, wrtBlockSize, comm):
        if wrtFilter is None:
            blkSize = wrtBlockSize  # could be None
            if (comm is not None) and (comm.Get_size() > 1):
                comm_blkSize = self.Np / comm.Get_size()
                blkSize = comm_blkSize if (blkSize is None) \
                    else min(comm_blkSize, blkSize)  # override with smaller comm_blkSize
        else:
            blkSize = None  # wrtFilter dictates block
        return blkSize

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
        pass  # default is to have no pre-computed quantities (but not an error to call this fn)

    def bulk_fill_probs(self, mxToFill, evalTree,
                        clipTo=None, check=False, comm=None):
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
        raise NotImplementedError("bulk_fill_probs(...) is not implemented!")

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
        raise NotImplementedError("bulk_fill_dprobs(...) is not implemented!")

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
        raise NotImplementedError("bulk_fill_hprobs(...) is not implemented!")

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
        raise NotImplementedError("bulk_hprobs_by_block(...) is not implemented!")
