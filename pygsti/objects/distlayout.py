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
from ..tools import sharedmemtools as _smt
from .verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from .copalayout import CircuitOutcomeProbabilityArrayLayout as _CircuitOutcomeProbabilityArrayLayout
from .resourceallocation import ResourceAllocation as _ResourceAllocation

import numpy as _np
import warnings as _warnings
#import time as _time #DEBUG TIMERS


def _assert_sequential(lst):
    last = lst[0]
    for x in lst[1:]:
        assert(last + 1 == x)
        last = x


class _DistributableAtom(object):
    """
    Behaves as a sub-layout for general purposes...
    needs .element_slice to indicate "final"/"global" indices.
    needs .wrt_block_size[1,2] to indicate how to distribute derivative calculations in arrays
      with derivative dimensions.
    needs __len__ and .iter_unique_circuits like a COPA layout (so functions as a sub-layout)
    """

    def __init__(self, element_slice, num_elements=None):
        self.element_slice = element_slice
        self.num_elements = _slct.length(element_slice) if (num_elements is None) else num_elements

    @property
    def cache_size(self):
        return 0


class DistributableCOPALayout(_CircuitOutcomeProbabilityArrayLayout):

    def __init__(self, circuits, unique_circuits, to_unique, unique_complete_circuits,
                 create_atom_fn, create_atom_args, num_atom_processors,
                 num_param_dimension_processors=(), param_dimensions=(), param_dimension_blk_sizes=(),
                 resource_alloc=None, verbosity=0):
        """
        TODO: docstring
        num_strategy_subcomms : int, optional
            The number of processor groups (communicators) to divide the "atomic" portions
            of this strategy (a circuit probability array layout) among when calling `distribute`.
            By default, the communicator is not divided.  This default behavior is fine for cases
            when derivatives are being taken, as multiple processors are used to process differentiations
            with respect to different variables.  If no derivaties are needed, however, this should be
            set to (at least) the number of processors.
        """
        #self.atoms = atoms
        #self.num_atom_processing_subcomms = 1
        #self.param_dimension_blk_sizes = (None,) * len(self._param_dimensions)
        #self.gather_mem_limit = None
        from mpi4py import MPI

        comm = resource_alloc.comm if (resource_alloc is not None) else None
        printer = _VerbosityPrinter.create_printer(verbosity, comm)

        rank = 0 if (comm is None) else comm.Get_rank()
        nprocs = 1 if (comm is None) else comm.Get_size()
        nAtomComms = num_atom_processors
        nAtoms = len(create_atom_args)
        printer.log("*** Distributing %d atoms to %d atom-processing groups (%s cores) ***" %
                    (nAtoms, nAtomComms, nprocs))

        assert(nAtomComms <= nAtoms), ("Cannot request more atom-processors (%d) than there are atoms (%d)!"
                                       % (nAtomComms, nAtoms))
        assert(nAtomComms <= nprocs), "Not enough processors (%d) to make nAtomComms=%d" % (nprocs, nAtomComms)

        if (resource_alloc is not None) and (resource_alloc.host_comm is not None):
            nHosts = len(resource_alloc.interhost_ranks)
            host_nprocs = resource_alloc.host_comm.Get_size()
            host_index = resource_alloc.host_index
            host_comm = resource_alloc.host_comm
            printer.log("*** Using shared memory to communicate between the %d cores on each of %d hosts ***" %
                        (host_nprocs, nHosts))
        else:
            # treat all procs as separate hosts
            nHosts = nprocs
            host_nprocs = 1
            host_index = rank
            host_comm = None

        # Allocate nAtomComms "atom processing comms" to the nHosts.

        # each proc should have a host_index (~hostname), and a proc gets
        myHostsAtomCommIndices, hostAtomCommOwners, peer_hosts = _mpit.distribute_indices_base(
            list(range(nAtomComms)), nHosts, host_index)

        if nHosts <= nAtomComms:
            # then each atom processing comm occupies a single host, and likewise a host
            # may be assigned multiple atom-processing comms.

            # Assign blocks of host procs to each atom comm index
            myAtomCommIndices, atomCommOwners, atom_processing_subcomm = \
                _mpit.distribute_indices(myHostsAtomCommIndices, host_comm)

            assert(len(myAtomCommIndices) == 1), ("There needs to be at least one processor per atom-processing comm."
                                                  " Try using fewer atom-processors!")
            myAtomCommIndex = myAtomCommIndices[0]
            printer.log("    More atom-processors than hosts: each host gets ~%d atom-processors"
                        % len(myHostsAtomCommIndices))

        else:  # nAtomComms < nHosts
            assert(len(myHostsAtomCommIndices) == 1), "Each host should be assigned to exactly 1 atomComm"
            myAtomCommIndex = myHostsAtomCommIndices[0]
            atom_processing_subcomm = comm.Split(color=myAtomCommIndex, key=rank) \
                if (comm is not None) else None
            printer.log("    More hosts than atom-processors: each host helps 1 atom-processor")

        #Mark whether this processor is a part of the last/final atom processor group, as only
        # this group will add "extra" penalty term elements to their objective function arrays.
        self.part_of_final_atom_processor = bool(myAtomCommIndex == nAtomComms - 1)

        #Get atom indices for this processor (the ones for this proc's atomComm)
        myAtomIndices, atomOwners, _ = _mpit.distribute_indices_base(
            list(range(nAtoms)), nAtomComms, myAtomCommIndex)
        _assert_sequential(myAtomIndices)

        #Create atoms needed by this proc (this must be done now so we know how many elements they compute)
        atoms_dict = {iAtom: create_atom_fn(create_atom_args[iAtom]) for iAtom in myAtomIndices}

        #Communicate atom sizes to other procs on same host so we can
        # compute the total number of elements on this host and our offset into it.
        atom_sizes = {i: atom.num_elements for i, atom in atoms_dict.items()}
        if comm is not None:
            all_atom_sizes = comm.allgather(atom_sizes if (atom_processing_subcomm is None
                                                           or atom_processing_subcomm.rank == 0) else {})
            atom_sizes = {}
            for sizes in all_atom_sizes: atom_sizes.update(sizes)

        # Get global element indices & so some setup for global elindex_outcome_tuples (TODO - do we want this?)
        offset = 0
        #atom_element_slices = {}
        global_elindex_outcome_tuples = _collections.OrderedDict([
            (unique_i, list()) for unique_i in range(len(unique_circuits))])
        local_elindex_outcome_tuples = _collections.OrderedDict()

        ##Create a subgroup that contains only the rank-0 processors of all the atom processing comms.
        #if atom_processing_subcomm is not None:
        #    sub_ranks = _np.array(comm.allgather(atom_processing_subcomm.rank))
        #    root_ranks = _np.where(sub_ranks == 0)[0]
        #    atom_owners_subcomm = comm.Create_group(comm.Incl(root_ranks))
        #else:
        #    atom_owners_subcomm = None

        start = stop = None
        my_unique_is = []
        my_unique_is_set = set()
        for i in range(nAtoms):
            to_send = 0  # default = contribute nothing to MPI.SUM below

            if i in atoms_dict:
                #print("DB (%d): updating elindex_outcome_tuples w/Atom %d:\n%s"
                #      % (rank, i, "\n".join(["%d: %s" % (indx, str(tups))
                #                             for indx, tups in atoms_dict[i].elindex_outcome_tuples.items()])))
                if start is None:
                    start = stop = offset
                assert(stop == offset)  # This should be checked by _assert_sequential(myAtomIndices) above
                stop += atom_sizes[i]
                if atom_processing_subcomm is None or atom_processing_subcomm.rank == 0:
                    to_send = rank  # this rank claims to be the owner of this atom

                for unique_i, eolist in atoms_dict[i].elindex_outcome_tuples.items():
                    if len(eolist) == 0: continue
                    assert(unique_i not in my_unique_is_set), "%d is already in my_unique_is" % unique_i
                    local_elindex_outcome_tuples[len(my_unique_is_set)] = [((offset - start) + elindex, outcome)
                                                                           for (elindex, outcome) in eolist]
                    my_unique_is.append(unique_i)
                    my_unique_is_set.add(unique_i)

            if comm is not None:
                owner_rank = comm.allreduce(to_send, op=MPI.SUM)
                atom_elindex_outcome_tuples = comm.bcast(atoms_dict[i].elindex_outcome_tuples
                                                         if (rank == owner_rank) else None, root=owner_rank)
            else:
                atom_elindex_outcome_tuples = atoms_dict[i].elindex_outcome_tuples
            for unique_i, eolist in atom_elindex_outcome_tuples.items():
                global_elindex_outcome_tuples[unique_i].extend([(offset + elindex, outcome)
                                                                for (elindex, outcome) in eolist])

            #atom_element_slices[i] = slice(offset, offset + atom_sizes[i])
            offset += atom_sizes[i]
        self.global_num_elements = offset
        self.global_element_slice = slice(start, stop)

        # Get size of allocated memory and indices into it (in element direction)
        # concatenate the slices of all the elements computed by this host (should be contiguous)
        myHostsAtomIndices = []
        for iAtomComm in myHostsAtomCommIndices:
            atomIndices, _, _ = _mpit.distribute_indices_base(
                list(range(nAtoms)), nAtomComms, iAtomComm)
            myHostsAtomIndices.extend(atomIndices)
        _assert_sequential(myHostsAtomIndices)  # so we know indices of these atoms are contiguous

        offset = 0; start = stop = None
        for i in myHostsAtomIndices:
            if i in atoms_dict:
                if start is None:
                    start = stop = offset
                assert(stop == offset)
                stop += atom_sizes[i]
                atoms_dict[i].element_slice = slice(offset - start, stop)  # atom's slice to index into *local* array
            offset += atom_sizes[i]
        self.host_num_elements = offset
        self.host_element_slice = slice(start, stop)

        # Similar for param distribution --------------------------------------------
        if len(param_dimensions) > 0:
            num_params = param_dimensions[0]
            num_param_processors = num_param_dimension_processors[0]

            if nHosts <= nAtomComms:  # then distribute params among atom_processing_subcomm's procs.
                # Assign blocks of procs to each param comm index
                myParamCommIndices, paramCommOwners, param_processing_subcomm = \
                    _mpit.distribute_indices(list(range(num_param_processors)), atom_processing_subcomm)

                assert(len(myParamCommIndices) == 1), ("There needs to be at least one processor per param-processing"
                                                       " comm. Try using fewer param-processors!")
                myParamCommIndex = myParamCommIndices[0]
                nHostsWithinAtomComm = 1  # needed for param2 test below
                printer.log(("    Atom-processors already occupy a single node, dividing atom-processor into %d"
                             " param-processors.") % num_param_processors)
                # atom_subcomm_owner_rank = paramCommOwners[myParamCommIndex]

            else:  # then distribute params among hosts, then among host_comm's procs - this is similar
                # to the world-comm distribution into atom-processing comms above (FUTURE: consolidate code?)
                sub_host_index = peer_hosts.index(host_index)
                nHostsWithinAtomComm = len(peer_hosts)
                myHostsParamCommIndices, hostParamCommOwners, peer_sub_hosts = _mpit.distribute_indices_base(
                    list(range(num_param_processors)), nHostsWithinAtomComm, sub_host_index)
                # ihost = hostParamCommOwners[myParamCommIndex] (within the nHostsWithinAtom)

                if nHostsWithinAtomComm <= num_param_processors:
                    # then each host gets a *diferent* set of (1 or more) param-comm indices and we can:
                    # Assign blocks of host procs to each param comm index
                    myParamCommIndices, paramCommOwners, param_processing_subcomm = \
                        _mpit.distribute_indices(myHostsParamCommIndices, host_comm)
                    # host_comm rank = paramCommOwners[myParamCommIndex]

                    assert(len(myParamCommIndices) == 1), ("There needs to be at least one processor per param-"
                                                           "processing comm. Try using fewer param-processors!")
                    myParamCommIndex = myParamCommIndices[0]
                    printer.log("   More param-processors than hosts-per-atom-proc: each host gets ~%d param-processors"
                                % len(myHostsParamCommIndices))

                else:  # num_param_processors < nHostsWithinAtomComm
                    # Then each param-processor will span an integer number (maybe 1) of hosts.
                    assert(len(myHostsParamCommIndices) == 1), "Each host should be assigned to exactly 1 paramComm"
                    myParamCommIndex = myHostsParamCommIndices[0]
                    param_processing_subcomm = atom_processing_subcomm.Split(color=myParamCommIndex, key=rank) \
                        if (atom_processing_subcomm is not None) else None
                    printer.log("    More host-per-atom-proc than param-processors: each host helps 1 param-processor")

            printer.log("*** Divided %d-host atom-processor (~%d procs) into %d param-processing groups ***" %
                        (nHostsWithinAtomComm,
                         atom_processing_subcomm.size if (atom_processing_subcomm is not None) else 1,
                         num_param_processors))

            #Get param indices for this processor (the ones for this proc's param_processing_subcomm)
            myParamIndices, paramOwners, _ = _mpit.distribute_indices_base(
                list(range(num_params)), num_param_processors, myParamCommIndex)
            _assert_sequential(myParamIndices)

            owned_paramCommIndex = myParamCommIndex if (param_processing_subcomm is None
                                                        or param_processing_subcomm.rank == 0) else -1
            owned_paramCommIndex_by_atomproc_rank = atom_processing_subcomm.allgather(owned_paramCommIndex) \
                if (atom_processing_subcomm is not None) else [myParamCommIndex]
            self.param_slices = _mpit.slice_up_slice(slice(0, num_params),
                                                     num_param_processors)  # matches param comm indices
            self.param_slice_owners = {ipc: atomproc_rank for atomproc_rank, ipc
                                       in enumerate(owned_paramCommIndex_by_atomproc_rank) if ipc >= 0}
            self.my_owned_paramproc_index = owned_paramCommIndex
            # Note: if muliple procs within atomproc com have the same myParamCommIndex (possible when
            #  param_processing_subcomm.size > 1) then the "owner" of a param slice is the
            #  param_processing_subcomm.rank == 0 processor.

            interatom_param_subcomm = comm.Split(color=myParamCommIndex, key=rank) if (comm is not None) else None

            self.global_num_params = num_params
            self.global_param_slice = _slct.list_to_slice(myParamIndices)

            ##Create a subgroup that contains only the rank-0 processors of all the param processing comms.
            #if param_processing_subcomm is not None:
            #    sub_ranks = _np.array(atom_processing_subcomm.allgather(param_processing_subcomm.rank))
            #    root_ranks = _np.where(sub_ranks == 0)[0]
            #    param_owners_subcomm = atom_processing_subcomm.Create_group(
            #        atom_processing_subcomm.Incl(root_ranks))
            #else:
            #    param_owners_subcomm = None

            # Get total parameter range on this host
            # (concat param slices of all the param-processing comms on this host)
            if nHosts <= nAtomComms:  # then each host contains one or more *complete* atoms & therefore *all* params
                self.host_num_params = self.global_num_params
                self.host_param_slice = self.global_param_slice
            else:
                _assert_sequential(myHostsParamCommIndices)
                offset = 0; host_param_slice = None
                for iParamComm in myHostsParamCommIndices:
                    pinds, _, _ = _mpit.distribute_indices_base(
                        list(range(num_params)), num_param_processors, iParamComm)
                    if iParamComm == myParamCommIndex:
                        host_param_slice = slice(offset, offset + len(pinds))
                    offset += len(pinds)
                self.host_num_params = offset
                self.host_param_slice = host_param_slice

            # split up myParamIndices into interatom_param_subcomm.size (~nAtomComms * param_processing_subcomm.size)
            #  groups for processing quantities that don't have any 'element' dimension, where we want all procs working
            #  on the parameter dimension.  We call this the "param_fine" subcomm.
            if interatom_param_subcomm is not None and interatom_param_subcomm.size > 1:
                myParamFineIndices, _, _ = _mpit.distribute_indices_base(
                    list(range(len(myParamIndices))), interatom_param_subcomm.size, interatom_param_subcomm.rank)
                _assert_sequential(myParamFineIndices)
                self.fine_param_subslice = _slct.list_to_slice(myParamFineIndices)
            else:
                self.fine_param_subslice = slice(0, len(myParamIndices))

            #self.host_param_fine_slice = _slct.slice_of_slice(self.fine_param_subslice, self.host_param_slice)
            self.global_param_fine_slice = _slct.slice_of_slice(self.fine_param_subslice, self.global_param_slice)
            param_fine_subcomm = comm.Split(color=self.global_param_fine_slice.start, key=rank) \
                if (comm is not None) else None

            if host_comm is not None:
                fine_slices_on_host = host_comm.allgather(self.global_param_fine_slice)
                offset = 0
                for iFine, slc in enumerate(fine_slices_on_host):
                    n = _slct.length(slc)
                    if iFine == host_comm.rank:
                        self.host_param_fine_slice = slice(offset, offset + n)
                    offset += n
                self.host_num_params_fine = offset
            else:
                self.host_num_params_fine = _slct.length(self.global_param_fine_slice)
                self.host_param_fine_slice = slice(0, self.host_num_params_fine)

            # When param_fine_subcomm.size > 1 this means we have multiple processors computing the same "fine" param
            # slice.  This can be tricky when needing to perform point-to-point IPC of fine-param data, and so we
            # consider the subset of param_fine_subcomm.rank == 0 processors as the "owners" of their fine-param slices,
            # and create the below lists and lookup dict so that it's easy to find the owner of a particular index.
            # At the end of point-to-point communication, the results can be broadcast using param_fine_subcomm to give
            # results to the rank > 0 processors.
            gps = self.global_param_fine_slice if (param_fine_subcomm is None or param_fine_subcomm.rank == 0) else None
            hps = self.host_param_fine_slice if (param_fine_subcomm is None or param_fine_subcomm.rank == 0) else None
            if host_comm is not None:
                owned_pslices_on_host = tuple(host_comm.allgather((gps, hps)))
                slices = resource_alloc.interhost_comm.allgather(owned_pslices_on_host)
                owners = resource_alloc.interhost_comm.allgather(resource_alloc.host_ranks)
                self.param_fine_slices_by_host = tuple([tuple(zip(owners_for_host, slcs_for_host))
                                                        for owners_for_host, slcs_for_host in zip(owners, slices)])
            else:
                gpss = comm.allgather((gps, hps))  # each host is a single proc, to this host the single slice per host
                self.param_fine_slices_by_host = tuple([(rank_and_gps,) for rank_and_gps in enumerate(gpss)])

            self.param_fine_slices_by_rank = tuple(comm.allgather(gps))

            self.owner_host_and_rank_of_global_fine_param_index = {}
            for host_index, ranks_and_pslices in enumerate(self.param_fine_slices_by_host):
                for gbl_rank, (gpslice, _) in ranks_and_pslices:
                    if gpslice is None: continue   # indicates a non-owner proc (param_fine_subcomm.rank > 0)
                    for p in range(gpslice.start, gpslice.stop):
                        self.owner_host_and_rank_of_global_fine_param_index[p] = (host_index, gbl_rank)

            # Similar for param2 distribution --------------------------------------------
            if len(param_dimensions) > 1:
                num_params2 = param_dimensions[1]
                num_param2_processors = num_param_dimension_processors[1]

                #divide param_processing_subcomm into as many chunks as we have procs (there's
                # no specified size like num_atom_processors or num_param_processors.

                if nHostsWithinAtomComm <= num_param_processors:
                    # then param_processing_subcomm is entirely on a host, so we can just divide it up:
                    myParam2CommIndices, param2CommOwners, param2_processing_subcomm = \
                        _mpit.distribute_indices(list(range(num_param2_processors)), param_processing_subcomm)

                    assert(len(myParam2CommIndices) == 1), ("There needs to be at least one processor per param2"
                                                            "-processing comm. Try using fewer param-processors!")
                    myParam2CommIndex = myParam2CommIndices[0]
                    nHostsWithinParamComm = 1  # needed ?
                    printer.log(("    Param-processors already occupy a single node, dividing param-processor into %d"
                                 " param2-processors.") % num_param2_processors)

                else:  # then distribute param2-processors among hosts, then among host_comm's procs - this is similar
                    # to the world-comm distribution into atom-processing comms above (FUTURE: consolidate code?)
                    # (param_processing_subcomm contains one or more hosts, so we need to ensure
                    #  that a param2-processor doesn't straddle a host boundary.)

                    sub_sub_host_index = peer_sub_hosts.index(sub_host_index)
                    nHostsWithinParamComm = len(peer_sub_hosts)
                    myHostsParam2CommIndices, hostParam2CommOwners, _ = _mpit.distribute_indices_base(
                        list(range(num_param2_processors)), nHostsWithinParamComm, sub_sub_host_index)

                    if nHostsWithinParamComm <= num_param2_processors:
                        # then each host gets a *diferent* set of (1 or more) param2-comm indices and we can:
                        # Assign blocks of host procs to each param comm index
                        myParam2CommIndices, param2CommOwners, param2_processing_subcomm = \
                            _mpit.distribute_indices(myHostsParam2CommIndices, host_comm)

                        assert(len(myParam2CommIndices) == 1), ("There needs to be at least one processor per param2-"
                                                                "processing comm. Try using fewer param2-processors!")
                        myParam2CommIndex = myParam2CommIndices[0]
                        printer.log(("    More param2-processors than hosts-per-param1-proc: each host gets ~%d"
                                     " param2-processors") % len(myHostsParam2CommIndices))

                    else:  # num_param2_processors < nHostsWithinParamComm
                        # Then each param2-processor will span an integer number (maybe 1) of hosts,
                        assert(len(myHostsParam2CommIndices) == 1), \
                            "Each host should be assigned to exactly 1 param2Comm"
                        myParam2CommIndex = myHostsParam2CommIndices[0]
                        param2_processing_subcomm = param_processing_subcomm.Split(color=myParam2CommIndex, key=rank) \
                            if (param_processing_subcomm is not None) else None
                        printer.log(("    More host-per-param-proc than param2-processors: "
                                     "each host helps 1 param2-processor"))

                printer.log("*** Divided %d-host param-processor (~%d procs) into %d param2-processing groups ***" %
                            (nHostsWithinParamComm,
                             param2_processing_subcomm.size if (param2_processing_subcomm is not None) else 1,
                             num_param2_processors))

                ##Create a subgroup that contains only the rank-0 processors of all the param2 processing comms.
                #if param2_processing_subcomm is not None:
                #    sub_ranks = _np.array(param_processing_subcomm.allgather(param2_processing_subcomm.rank))
                #    root_ranks = _np.where(sub_ranks == 0)[0]
                #    param2_owners_subcomm = param_processing_subcomm.Create_group(
                #        param_processing_subcomm.Incl(root_ranks))
                #else:
                #    param2_owners_subcomm = None

                #Get param2 indices for this processor (the ones for this proc's param2_processing_subcomm)
                myParam2Indices, param2Owners, _ = _mpit.distribute_indices_base(
                    list(range(num_params2)), num_param2_processors, myParam2CommIndex)
                _assert_sequential(myParam2Indices)

                interatom_param2_subcomm = interatom_param_subcomm.Split(
                    color=myParam2CommIndex, key=rank) if (interatom_param_subcomm is not None) else None

                # Now myParam2Indices and param2_processing_subcomm have been computed, and we can set slices:
                self.global_num_params2 = num_params2
                self.global_param2_slice = _slct.list_to_slice(myParam2Indices)

                # Get total parameter range on this host
                # (concat param slices of all the param-processing comms on this host)
                if nHostsWithinAtomComm <= num_param_processors:
                    # each host contains one or more *complete* param-comms & therefore *all* param2 range
                    self.host_num_params2 = self.global_num_params2
                    self.host_param2_slice = self.global_param2_slice
                else:
                    _assert_sequential(myHostsParam2CommIndices)
                    offset = 0; host_param2_slice = None
                    for iParam2Comm in myHostsParam2CommIndices:
                        pinds2, _, _ = _mpit.distribute_indices_base(
                            list(range(num_params2)), num_param2_processors, iParam2Comm)
                        if iParam2Comm == myParam2CommIndex:
                            host_param2_slice = slice(offset, offset + len(pinds2))
                        offset += len(pinds2)
                    self.host_num_params2 = offset
                    self.host_param2_slice = host_param2_slice

            else:
                self.global_num_params2 = self.global_param2_slice = None
                self.host_num_params2 = self.host_param2_slice = None
                param2_processing_subcomm = None
                interatom_param2_subcomm = None

        else:
            self.global_num_params = self.global_param_slice = None
            self.global_num_params2 = self.global_param2_slice = None
            self.host_num_params = self.host_param_slice = None
            self.host_num_params2 = self.host_param2_slice = None
            self.host_num_params_fine = None
            self.fine_param_subslice = self.host_param_fine_slice = self.global_param_fine_slice = None
            self.param_fine_slices_by_rank = self.param_fine_slices_by_host = None
            self.owner_host_and_rank_of_global_fine_param_index = None
            param_processing_subcomm = None
            param2_processing_subcomm = None
            interatom_param_subcomm = None
            interatom_param2_subcomm = None
            param_fine_subcomm = None

        # save subcomms as sub-resource-allocations
        resource_alloc.layout_allocs['atom-processing'] = _ResourceAllocation(
            atom_processing_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)
        resource_alloc.layout_allocs['param-processing'] = _ResourceAllocation(
            param_processing_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)
        resource_alloc.layout_allocs['param2-processing'] = _ResourceAllocation(
            param2_processing_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)
        resource_alloc.layout_allocs['param-interatom'] = _ResourceAllocation(
            interatom_param_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)
        resource_alloc.layout_allocs['param2-interatom'] = _ResourceAllocation(
            interatom_param2_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)
        resource_alloc.layout_allocs['param-fine'] = _ResourceAllocation(
            param_fine_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)

        if resource_alloc.host_comm is not None:  # signals that we want to use shared intra-host memory
            resource_alloc.layout_allocs['atom-processing'].build_hostcomms()
            resource_alloc.layout_allocs['param-processing'].build_hostcomms()
            resource_alloc.layout_allocs['param2-processing'].build_hostcomms()
            resource_alloc.layout_allocs['param-interatom'].build_hostcomms()
            resource_alloc.layout_allocs['param2-interatom'].build_hostcomms()
            resource_alloc.layout_allocs['param-fine'].build_hostcomms()

        self.atoms = [atoms_dict[i] for i in myAtomIndices]
        self.param_dimension_blk_sizes = param_dimension_blk_sizes

        self.global_layout = _CircuitOutcomeProbabilityArrayLayout(circuits, unique_circuits, to_unique,
                                                                   global_elindex_outcome_tuples,
                                                                   unique_complete_circuits,
                                                                   param_dimensions)
        #Select the local portions of the global arrays to create *this* layout.
        local_unique_complete_circuits = []
        local_unique_circuits = []
        local_circuits = []
        local_to_unique = {}
        circuits_dict = {i: c for i, c in enumerate(circuits)}  # for fast lookup
        rev_unique = _collections.defaultdict(list)
        for orig_i, unique_i in to_unique.items():
            rev_unique[unique_i].append(orig_i)

        #TODO: Could make this a bit faster using dict lookups?
        for i in my_unique_is:
            c = unique_circuits[i]
            cc = unique_complete_circuits[i]
            local_unique_index = len(local_unique_circuits)  # an index into local_unique_circuits
            local_unique_complete_circuits.append(cc)
            local_unique_circuits.append(c)
            start = len(local_circuits)
            local_to_unique.update({start + k: local_unique_index for k in range(len(rev_unique[i]))})
            local_circuits.extend([circuits_dict[orig_i] for orig_i in rev_unique[i]])

        super().__init__(local_circuits, local_unique_circuits, local_to_unique, local_elindex_outcome_tuples,
                         local_unique_complete_circuits, param_dimensions)

        #DEBUG LAYOUT PRINTING
        #def cnt_str(cnt):
        #    if cnt is None: return "%4s" % '-'
        #    return "%4d" % cnt
        #def slc_str(slc):
        #    if slc is None: return "%14s" % '--'
        #    return "%3d->%3d (%3d)" % (slc.start, slc.stop, slc.stop - slc.start) \
        #        if isinstance(slc, slice) else "%14s" % str(slc)
        #shm = bool(resource_alloc.host_comm is not None)  # shared mem?
        #if rank == 0:
        #    print("%11s %-14s %-14s %-14s   %-14s %-4s %-14s %-4s %-14s %-4s" % (
        #        '#', 'g-elements', 'g-params', 'g-param2s',
        #        'h-elements','tot', 'h-params','tot', 'h-params2','tot'),
        #          flush=True)
        #resource_alloc.comm.barrier()
        #for r in range(resource_alloc.comm.size):
        #    if r == rank:
        #        my_desc = ("%3d (%2d.%2d)" % (rank, resource_alloc.host_index, resource_alloc.host_comm.rank)) \
        #                  if shm else ("%11d" % rank)
        #        print(my_desc, slc_str(self.global_element_slice), slc_str(self.global_param_slice),
        #              slc_str(self.global_param2_slice), ' ',
        #              slc_str(self.host_element_slice), cnt_str(self.host_num_elements),
        #              slc_str(self.host_param_slice), cnt_str(self.host_num_params),
        #              slc_str(self.host_param2_slice), cnt_str(self.host_num_params2),  flush=True)
        #    resource_alloc.comm.barrier()
        #
        #if rank == 0:
        #    print("%11s %-14s %-14s %-4s" % ('#', 'g-pfine', 'h-pfine', 'tot'), flush=True)
        #resource_alloc.comm.barrier()
        #for r in range(resource_alloc.comm.size):
        #    if r == rank:
        #        my_desc = ("%3d (%2d.%2d)" % (rank, resource_alloc.host_index, resource_alloc.host_comm.rank)) \
        #                  if shm else ("%11d" % rank)
        #        print(my_desc, slc_str(self.global_param_fine_slice), slc_str(self.host_param_fine_slice),
        #              cnt_str(self.host_num_params_fine), flush=True)
        #    resource_alloc.comm.barrier()

    @property
    def max_atom_elements(self):
        if len(self.atoms) == 0: return 0
        return max([atom.num_elements for atom in self.atoms])

    @property
    def max_atom_cachesize(self):
        if len(self.atoms) == 0: return 0
        return max([atom.cache_size for atom in self.atoms])

    def allocate_local_array(self, array_type, dtype, resource_alloc=None, zero_out=False, track_memory=False,
                             extra_elements=0):
        """
        Allocate an array that is distributed according to this layout.

        array_type : {'e', 'ep', 'epp', 'hessian', 'jtj', 'jtf'}
            The type of array being gathered.  TODO: docstring - more description

        TODO: docstring - returns the *local* memory and shared mem handle
        """

        if array_type in ('e', 'ep', 'ep2', 'epp'):
            array_shape = (self.host_num_elements + extra_elements,) if self.part_of_final_atom_processor \
                else (self.host_num_elements,)
            if array_type in ('ep', 'epp'): array_shape += (self.host_num_params,)
            if array_type in ('ep2', 'epp'): array_shape += (self.host_num_params2,)
            allocating_ralloc = resource_alloc  # share mem between these processors
        elif array_type == 'p':
            array_shape = (self.host_num_params,)
            allocating_ralloc = resource_alloc  # share mem between these processors
        #elif array_type == 'atom-hessian':
        #    array_shape = (self.host_num_params, self.host_num_params2)
        #    allocating_ralloc = resource_alloc.layout_allocs['atom-processing']  # don't share mem btwn atoms,
        #    # as each atom will have procs with the same (param1, param2) index block but we want separate mem
        elif array_type == 'jtj':
            array_shape = (self.host_num_params_fine, self.global_num_params)
            allocating_ralloc = resource_alloc  # .layout_allocs['param-interatom']
        elif array_type == 'jtf':  # or array_type == 'pfine':
            array_shape = (self.host_num_params_fine,)
            allocating_ralloc = resource_alloc  # .layout_allocs['param-interatom']
        else:
            raise ValueError("Invalid array_type: %s" % str(array_type))

        host_array, host_array_shm = _smt.create_shared_ndarray(allocating_ralloc, array_shape, dtype,
                                                                zero_out, track_memory)

        if array_type in ('e', 'ep', 'ep2', 'epp'):
            elslice = slice(self.host_element_slice.start, self.host_element_slice.stop + extra_elements) \
                if self.part_of_final_atom_processor else self.host_element_slice
            tuple_of_slices = (elslice,)
            if array_type in ('ep', 'epp'): tuple_of_slices += (self.host_param_slice,)
            if array_type in ('ep2', 'epp'): tuple_of_slices += (self.host_param2_slice,)
        elif array_type == 'p':
            tuple_of_slices = (self.host_param_slice,)
        #elif array_type == 'atom-hessian':
        #    tuple_of_slices = (self.host_param_slice, self.host_param2_slice)
        elif array_type == 'jtj':
            tuple_of_slices = (self.host_param_fine_slice, slice(0, self.global_num_params))
        elif array_type == 'jtf':  # or 'x'
            tuple_of_slices = (self.host_param_fine_slice,)

        local_array = host_array[tuple_of_slices]
        local_array = local_array.view(_smt.LocalNumpyArray)
        local_array.host_array = host_array
        local_array.slices_into_host_array = tuple_of_slices,
        local_array.shared_memory_handle = host_array_shm

        return local_array, host_array_shm

    def gather_local_array(self, array_type, array_portion, resource_alloc=None, extra_elements=0):
        """
        Gathers an array onto the root processor.

        Gathers the portions of an array that was distributed using this
        layout (i.e. according to the host_element_slice, etc. slices in
        this layout).  Arrays can be 1, 2, or 3-dimensional.  The dimensions
        are understood to be along the "element", "parameter", and
        "2nd parameter" directions in that order.

        Parameters
        ----------
        array_type : {'e', 'ep', 'epp', 'hessian', 'jtj', 'jtf'}
            The type of array being gathered.  TODO: docstring - more description

        array_portion : numpy.ndarray
            The portion of the final array that is local to the calling
            processor.  This should be a shared memory array when a
            `resource_alloc` with shared memory enabled was used to construct
            this layout.

        resource_alloc : ResourceAllocation, optional
            The resource allocation object that was used to construt this
            layout, specifying the number and organization of processors
            to distribute arrays among.

        Returns
        -------
        numpy.ndarray or None
            The full (global) output array on the root (rank=0) processor.
            `None` on all other processors.
        """
        if resource_alloc is None or resource_alloc.comm is None:
            return array_portion

        # Gather the "extra_elements" when they are present,
        # by enlarging the element slice of all the procs in the final atom processor.
        global_num_els = self.global_num_elements + extra_elements
        if extra_elements > 0 and self.part_of_final_atom_processor:
            global_el_slice = slice(self.global_element_slice.start, self.global_element_slice.stop + extra_elements)
        else:
            global_el_slice = self.global_element_slice

        # Set two resource allocs based on the array_type:
        # gather_ralloc.comm groups the processors that we gather data over.
        # unit_ralloc.comm groups all procs that compute the *same* unit being gathered (e.g. the
        #  same (atom, param_slice) tuple, so that only the rank=0 procs of this comm need to
        #  participate in the gathering (others would be redundant and set memory multiple times)
        if array_type == 'e':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['atom-processing']
            global_shape = (global_num_els,)
            slice_of_global = global_el_slice
        elif array_type == 'ep':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['param-processing']
            global_shape = (global_num_els, self.global_num_params)
            slice_of_global = (global_el_slice, self.global_param_slice)
        elif array_type == 'ep2':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['param2-processing']  # this may not be right...
            global_shape = (global_num_els, self.global_num_params2)
            slice_of_global = (global_el_slice, self.global_param2_slice)
        elif array_type == 'epp':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['param2-processing']
            global_shape = (global_num_els, self.global_num_params, self.global_num_params2)
            slice_of_global = (global_el_slice, self.global_param_slice, self.global_param2_slice)
        elif array_type == 'p':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['param-processing']
            global_shape = (self.global_num_params,)
            slice_of_global = (self.global_param_slice,)
        elif array_type == 'jtj':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['param-fine']
            global_shape = (self.global_num_params, self.global_num_params)
            slice_of_global = (self.global_param_fine_slice, slice(0, self.global_num_params))
        elif array_type == 'jtf':  # or 'x'
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['param-fine']
            global_shape = (self.global_num_params,)
            slice_of_global = self.global_param_fine_slice
        else:
            raise ValueError("Invalid array type: %s" % str(array_type))

        gather_comm = gather_ralloc.interhost_comm if (gather_ralloc.host_comm is not None) else gather_ralloc.comm
        global_array, global_array_shm = _smt.create_shared_ndarray(
            resource_alloc, global_shape, 'd') if gather_comm.rank == 0 else (None, None)

        gather_ralloc.gather(global_array, array_portion, slice_of_global, unit_ralloc)

        ret = global_array.copy() if resource_alloc.comm.rank == 0 else None  # so no longer shared mem (needed?)
        resource_alloc.comm.barrier()  # make sure global_array is copied before we free it
        if gather_comm.rank == 0:
            _smt.cleanup_shared_ndarray(global_array_shm)
        return ret

    def fill_jtf(self, j, f, jtf, resource_alloc):
        """TODO: docstring  - assumes j, f are local arrays, allocated using 'ep' and 'e' types, respectively.
        Returns an array allocated using the 'jtf' type.
        """
        param_ralloc = resource_alloc.layout_allocs['param-processing']  # this group acts on (element, param) blocks
        interatom_ralloc = resource_alloc.layout_allocs['param-interatom']  # procs with same param slice & diff atoms

        if interatom_ralloc.comm is None:  # only 1 atom, so no need to sum below
            jtf[:] = _np.dot(j.T[self.fine_param_subslice, :], f)
            return

        local_jtf = _np.dot(j.T, f)  # need to sum this value across all atoms

        # assume jtf is created from allocate_local_array('jtf', 'd', resource_alloc)
        scratch, scratch_shm = _smt.create_shared_ndarray(
            interatom_ralloc, (_slct.length(self.host_param_slice),), 'd')
        interatom_ralloc.comm.barrier()  # wait for scratch to be ready
        interatom_ralloc.allreduce_sum(scratch, local_jtf, unit_ralloc=param_ralloc)
        jtf[:] = scratch[self.fine_param_subslice]  # takes sub-portion to move to "fine" parameter distribution
        interatom_ralloc.comm.barrier()  # don't free scratch too early
        _smt.cleanup_shared_ndarray(scratch_shm)

        #if param_comm.host_comm is not None and param_comm.host_comm.rank != 0:
        #    return None  # this processor doesn't need to do any more - root host proc will fill returned shared mem

    # jtj, jtj_shm = self.allocate_local_array('jtj', 'd', resource_alloc, zero_out=False)
    def fill_jtj(self, j, jtj, resource_alloc):
        """TODO: docstring  - assumes j is a local array, allocated using 'ep' and 'e' types, respectively.
        Returns an array allocated using the 'jtj' type.
        """
        jT = j.T
        param_ralloc = resource_alloc.layout_allocs['param-processing']  # this group acts on (element, param) blocks
        atom_ralloc = resource_alloc.layout_allocs['atom-processing']  # this group acts on (element,) blocks
        interatom_ralloc = resource_alloc.layout_allocs['param-interatom']  # procs with same param slice & diff atoms
        atom_jtj = _np.empty((_slct.length(self.host_param_slice), self.global_num_params), 'd')  # for my atomproc

        for i, param_slice in enumerate(self.param_slices):
            if i == self.my_owned_paramproc_index:
                assert(param_slice == self.global_param_slice)
                if atom_ralloc.comm is not None:
                    assert(self.param_slice_owners[i] == atom_ralloc.comm.rank)
                    atom_ralloc.comm.bcast(j, root=atom_ralloc.comm.rank)
                    #Note: we only really need to broadcast this to other param_ralloc.comm.rank == 0
                    # procs as these are the only atom_jtj's that contribute in the allreduce_sum below.
                else:
                    assert(self.param_slice_owners[i] == 0)
                atom_jtj[:, param_slice] = _np.dot(jT, j)
            else:
                other_j = atom_ralloc.comm.bcast(None, root=self.param_slice_owners[i])
                atom_jtj[:, param_slice] = _np.dot(jT, other_j)

        #Now need to sum atom_jtj over atoms to get jtj:
        # assume jtj is created from allocate_local_array('jtj', 'd', resource_alloc)
        if interatom_ralloc.comm is None:  # only 1 atom - nothing to sum!
            #Note: in this case, we could have just done jT = j.T[self.fine_param_subslice, :] above...
            jtj[:, :] = atom_jtj[self.fine_param_subslice, :]  # takes sub-portion to move to "fine" param distribution
            return

        # Note: could allocate scratch in advance?
        scratch, scratch_shm = _smt.create_shared_ndarray(
            interatom_ralloc, (_slct.length(self.host_param_slice), self.global_num_params), 'd')
        interatom_ralloc.comm.barrier()  # wait for scratch to be ready
        interatom_ralloc.allreduce_sum(scratch, atom_jtj, unit_ralloc=param_ralloc)
        jtj[:, :] = scratch[self.fine_param_subslice, :]  # takes sub-portion to move to "fine" parameter distribution
        interatom_ralloc.comm.barrier()  # don't free scratch too early

        _smt.cleanup_shared_ndarray(scratch_shm)

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
        # Parallelization is then performed over the parameter (additional) dimensions
        info = {}
        subcomm_ranks = _collections.defaultdict(list)

        nAtomComms = self.num_atom_processing_subcomms
        nAtoms = len(self.atoms)
        assert(nAtomComms <= nAtoms), "Cannot request more sub-comms ({nAtomComms}) than there are atoms ({nAtoms})!"

        assert(nAtomComms <= nprocs), "Not enough processors (%d) to make nAtomComms=%d" % (nprocs, nAtomComms)
        for rank in range(nprocs):
            mySubCommIndices, _, _ = \
                _mpit.distribute_indices_base(list(range(nAtomComms)), nprocs, rank)
            assert(len(mySubCommIndices) == 1), "Each rank should be assigned to exactly 1 subComm group"
            mySubCommIndex = mySubCommIndices[0]
            subcomm_ranks[mySubCommIndex].append(rank)

            myAtomIndices, _, _ = _mpit.distribute_indices_base(
                list(range(nAtoms)), nAtomComms, mySubCommIndex)

            info[rank] = {'atom_indices': myAtomIndices, 'subcomm_index': mySubCommIndex}

        #Set the subcomm size (# of processors) that each rank is a part of.
        for rank in range(nprocs):
            info[rank]['subcomm_size'] = len(subcomm_ranks[info[rank]['subcomm_index']])

        return info
