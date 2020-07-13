"""
Defines the DistributableCOPALayout class.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections

from ..tools import mpitools as _mpit
from ..tools import slicetools as _slct
from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .copalayout import CircuitOutcomeProbabilityArrayLayout as _CircuitOutcomeProbabilityArrayLayout

import numpy as _np
import warnings as _warnings
#import time as _time #DEBUG TIMERS


class _DistributableAtom(object):
    """
    Behaves as a sub-layout for general purposes...
    needs .element_slice to indicate "final"/"global" indices.
    needs .wrt_block_size[1,2] to indicate how to distribute derivative calculations in arrays
      with derivative dimensions.
    needs __len__ and .iter_circuits like a COPA layout (so functions as a sub-layout)
    """

    def __init__(self, element_slice, num_elements=None):
        self.element_slice = element_slice
        self.num_elements = _slct.length(element_slice) if (num_elements is None) else num_elements


class DistributableCOPALayout(_CircuitOutcomeProbabilityArrayLayout):

    def __init__(self, circuits, unique_circuits, to_unique, elindex_outcome_tuples, unique_complete_circuits,
                 atoms, additional_dimensions=()):
        """
        num_strategy_subcomms : int, optional
            The number of processor groups (communicators) to divide the "atomic" portions
            of this strategy (a circuit probability array layout) among when calling `distribute`.
            By default, the communicator is not divided.  This default behavior is fine for cases
            when derivatives are being taken, as multiple processors are used to process differentiations
            with respect to different variables.  If no derivaties are needed, however, this should be
            set to (at least) the number of processors.
        """
        super().__init__(circuits, unique_circuits, to_unique, elindex_outcome_tuples, unique_complete_circuits,
                         additional_dimensions)

        self.atoms = atoms
        self.num_atom_processing_subcomms = 1
        self.additional_dimension_blk_sizes = (None,) * len(self._additional_dimensions)
        self.gather_mem_limit = None

        # ********* Only non-None for sub-strategies ******************

        ## The mapping between this tree's final operation sequence indices and its parent's
        #self.myFinalToParentFinalMap = None
        #
        ## The mapping between this tree's final element indices and its parent's
        #self.myFinalElsToParentFinalElsMap = None
        #
        ## The parent's index of each of this tree's *final* indices
        #self.parentIndexMap = None
        #
        ## list of the operation labels
        #self.opLabels = []
        #
        ## A dictionary whose keys are the "original" (as-given to initialize)
        ## indices and whose values are the new "permuted" indices.  So if you
        ## want to know where in a tree the ith-element of circuit_list (as
        ## passed to initialize(...) is, it's at index original_index_lookup[i]
        #self.original_index_lookup = None

    def allocate_local_array(self, array_type, zero_out=False, dtype='d'):
        raise NotImplementedError(("This function should only allocate the space needed "
                                   "for this processor when fwdsim's gather=False (?)"))

    def local_memory_estimate(self, nprocs, array_type, dtype='d'):
        """
        Per-processor memory required to allocate a local array (an estimate in bytes).
        """
        #bytes_per_element = _np.dtype(dtype).itemsize
        raise NotImplementedError()

    def set_distribution_params(self, num_atom_processing_subcomms, additional_dimension_blk_sizes,
                                gather_mem_limit):
        self.num_atom_processing_subcomms = num_atom_processing_subcomms
        self.additional_dimension_blk_sizes = additional_dimension_blk_sizes
        self.gather_mem_limit = gather_mem_limit

    #def is_split(self):
    #    """
    #    Whether strategy contains multiple atomic parts (sub-strategies).
    #
    #    Returns
    #    -------
    #    bool
    #    """
    #    return len(self.atoms) > 0

    def distribute(self, comm, verbosity=0):
        """
        Distributes this strategy's atomic parts across multiple processors.


        TODO: update this docstring text (it's outdated):
        This function checks how many processors are present in
        `comm` and divides this tree's subtrees into groups according to
        the number of subtree comms provided as an argument to
        `initialize`.  Note that this does *not* always divide the subtrees
        among the processors as much as possible, as this is not always what is
        desired (computing several subtrees serially using multiple
        processors simultaneously can be faster, due to better balancing, than
        computing all the subtrees simultaneously using smaller processor groups
        for each).

        For example, if the number of subtree comms given to
        `initialize` == 1, then all the subtrees are assigned to the one and
        only processor group containing all the processors of `comm`.  If the
        number of subtree comms == 2 and there are 4 subtrees and 8 processors,
        then `comm` is divided into 2 groups of 4 processors each, and two
        subtrees are assigned to each processor group.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            When not None, an MPI communicator for distributing subtrees
            across processor groups

        verbosity : int, optional
            How much detail to send to stdout.

        Returns
        -------
        myAtomIndices : list
            A list of integer indices specifying which atoms this
            processor is responsible for.
        atomOwners : dict
            A dictionary whose keys are integer atom indices and
            whose values are processor ranks, which indicates which
            processor is responsible for communicating the final
            results of each atom.
        mySubComm : mpi4py.MPI.Comm or None
            The communicator for the processor group that is responsible
            for computing the same `myAtomIndices` list.  This
            communicator is used for further processor division (e.g.
            for parallelization across derivative columns).
        """
        # split tree into local atoms, each which contains one/group of
        # processors (group can then parallelize derivative calcs over
        # model parameters)

        #rank = 0 if (comm is None) else comm.Get_rank()
        nprocs = 1 if (comm is None) else comm.Get_size()
        nAtomComms = self.num_atom_processing_subcomms
        nAtoms = len(self.atoms)
        assert(nAtomComms <= nAtoms), "Cannot request more sub-comms ({nAtomComms}) than there are atoms ({nAtoms})!"

        assert(nAtomComms <= nprocs), "Not enough processors (%d) to make nAtomComms=%d" % (nprocs, nAtomComms)
        mySubCommIndices, subCommOwners, mySubComm = \
            _mpit.distribute_indices(list(range(nAtomComms)), comm)
        assert(len(mySubCommIndices) == 1), "Each rank should be assigned to exactly 1 subComm group"
        mySubCommIndex = mySubCommIndices[0]

        myAtomIndices, atomOwners = _mpit.distribute_indices_base(
            list(range(nAtoms)), nAtomComms, mySubCommIndex)

        # atomOwners contains index of owner subComm, but we really want
        #  the owning processor, i.e. the owner of the subComm
        atomOwners = {iAtom: subCommOwners[atomOwners[iAtom]]
                      for iAtom in atomOwners}

        printer = _VerbosityPrinter.create_printer(verbosity, comm)
        printer.log("*** Distributing %d atoms into %d sub-comms (%s processors) ***" %
                    (nAtoms, nAtomComms, nprocs))

        return myAtomIndices, atomOwners, mySubComm

    def distribution_info(self, nprocs):
        """
        Generates information about how this layout is distributed across multiple processors.

        This is useful when comparing and selecting a layout, as this information
        can be used to compute the amount of required memory *per processor*.

        Parameters
        ----------
        nprocs : int
            The number of processors.

        Returns
        -------
        dict
        """
        # layout is already split into atoms.  We split these atoms
        # into `self.num_atom_processing_subcomms` groups, each of which
        # has roughly nprocs / self.num_atom_processing_subcomms processors.
        # Parallelization is then performed over the additional dimensions
        # (usually model parameters).
        info = {}
        subcomm_ranks = _collections.defaultdict(list)

        nAtomComms = self.num_atom_processing_subcomms
        nAtoms = len(self.atoms)
        assert(nAtomComms <= nAtoms), "Cannot request more sub-comms ({nAtomComms}) than there are atoms ({nAtoms})!"

        assert(nAtomComms <= nprocs), "Not enough processors (%d) to make nAtomComms=%d" % (nprocs, nAtomComms)
        for rank in range(nprocs):
            mySubCommIndices, _ = \
                _mpit.distribute_indices_base(list(range(nAtomComms)), nprocs, rank)
            assert(len(mySubCommIndices) == 1), "Each rank should be assigned to exactly 1 subComm group"
            mySubCommIndex = mySubCommIndices[0]
            subcomm_ranks[mySubCommIndex].append(rank)

            myAtomIndices, _ = _mpit.distribute_indices_base(
                list(range(nAtoms)), nAtomComms, mySubCommIndex)

            info[rank] = {'atom_indices': myAtomIndices, 'subcomm_index': mySubCommIndex}

        #Set the subcomm size (# of processors) that each rank is a part of.
        for rank in range(nprocs):
            info[rank]['subcomm_size'] = len(subcomm_ranks[info[rank]['subcomm_index']])

        return info
