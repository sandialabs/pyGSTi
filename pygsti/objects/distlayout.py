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
            host_comm = comm

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
            (orig_i, list()) for orig_i in range(len(unique_circuits))])
        local_elindex_outcome_tuples = _collections.OrderedDict()

        ##Create a subgroup that contains only the rank-0 processors of all the atom processing comms.
        #if atom_processing_subcomm is not None:
        #    sub_ranks = _np.array(comm.allgather(atom_processing_subcomm.rank))
        #    root_ranks = _np.where(sub_ranks == 0)[0]
        #    atom_owners_subcomm = comm.Create_group(comm.Incl(root_ranks))
        #else:
        #    atom_owners_subcomm = None

        start = stop = None
        my_unique_is = set()
        for i in range(nAtoms):
            to_send = 0  # default = contribute nothing to MPI.SUM below

            if i in atoms_dict:
                #print("DB (%d): updating elindex_outcome_tuples w/Atom %d:\n%s" 
                #      % (rank, i, "\n".join(["%d: %s" % (indx, str(tups))
                #                             for indx, tups in atoms_dict[i].elindex_outcome_tuples.items()])))
                if start == None: 
                    start = stop = offset
                assert(stop == offset)  # This should be checked by _assert_sequential(myAtomIndices) above
                stop += atom_sizes[i]
                if atom_processing_subcomm is None or atom_processing_subcomm.rank == 0:
                    to_send = rank  # this rank claims to be the owner of this atom

                for unique_i, eolist in atoms_dict[i].elindex_outcome_tuples.items():
                    assert(unique_i not in my_unique_is)
                    if len(eolist) == 0: continue
                    local_elindex_outcome_tuples[len(my_unique_is)] = [((offset - start) + elindex, outcome)
                                                                       for (elindex, outcome) in eolist]
                    my_unique_is.add(unique_i)

                    
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

        #REMOVE
        #if rank == 0:
        #    print("DB (root): final elindex_outcome_tuples :\n%s" 
        #          % ("\n".join(["%d: %s" % (indx, str(tups))
        #                        for indx, tups in elindex_outcome_tuples.items()])))
        
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
        #REMOVE print("Atom sizes = ",atom_sizes, comm.rank)
        #REMOVE print("Element slice = ",self.host_element_slice, self.host_num_elements, host_index, comm.rank)
        
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
                    printer.log("    More param-processors than hosts-per-atom-proc: each host gets ~%d param-processors"
                        % len(myHostsParamCommIndices))

                else:  # num_param_processors < nHostsWithinAtomComm
                    # Then each param-processor will span an integer number (maybe 1) of hosts.
                    assert(len(myHostsParamCommIndices) == 1), "Each host should be assigned to exactly 1 paramComm"
                    myParamCommIndex = myHostsParamCommIndices[0]
                    param_processing_subcomm = atom_processing_subcomm.Split(color=myParamCommIndex, key=rank) \
                                                    if (atom_processing_comm is not None) else None
                    printer.log("    More host-per-atom-proc than param-processors: each host helps 1 param-processor")

            printer.log("*** Divided %d-host atom-processor (~%d procs) into %d param-processing groups ***" %
                        (nHostsWithinAtomComm, 
                         atom_processing_subcomm.size if (atom_processing_subcomm is not None) else 1,
                         num_param_processors))

            #Get param indices for this processor (the ones for this proc's param_processing_subcomm)
            myParamIndices, paramOwners, _ = _mpit.distribute_indices_base(
                list(range(num_params)), num_param_processors, myParamCommIndex)
            _assert_sequential(myParamIndices)

            paramCommIndex_by_atomproc_rank = atom_processing_subcomm.allgather(myParamCommIndex)
            self.param_slices = _mpit.slice_up_slice(slice(0,num_params), num_param_processors)  # matches param comm indices
            self.param_slice_owners = {ipc: atomproc_rank for atomproc_rank, ipc
                                       in enumerate(paramCommIndex_by_atomproc_rank)}
            self.my_paramproc_index = myParamCommIndex

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
                self.host_param_slice= host_param_slice

            # split up myParamIndices into interatom_param_subcomm.size (~nAtomComms * param_processing_subcomm.size)
            #  groups for processing quantities that don't have any 'element' dimension, where we want all procs working
            #  on the parameter dimension.  We call this the "param_fine" subcomm.
            if interatom_param_subcomm is not None and interatom_param_subcomm.size > 1:
                myParamFineIndices, _, _ = _mpit.distribute_indices_base(
                    list(range(myParamIndices)), interatom_param_subcomm.size, interatom_param_subcomm.rank)
                _assert_sequential(myParamFineIndices)
                self.fine_param_subslice = _slct.list_to_slice(myParamFineIndices)
            else:
                self.fine_param_subslice = slice(0, len(myParamIndices))
            self.host_param_fine_slice = _slct.slice_of_slice(self.fine_param_subslice, self.host_param_slice)
            self.global_param_fine_slice = _slct.slice_of_slice(self.fine_param_subslice, self.global_param_slice)
            param_fine_subcomm = comm.Split(color=self.global_param_fine_slice.start, key=rank) \
                                 if (comm is not None) else None

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

                    #REMOVE I think this is an unnecessary special case:
                    # if num_params2 >= param_processing_subcomm.size:
                    #     # then each processor will get *different* param2 indices and there
                    #     # can be no host-boundary crossing (2+ procs w/the same param2 indices
                    #     # that lie on different hosts), and we can do exactly what we did above:
                    #     myParam2Indices, param2Owners, param2_processing_subcomm = \
                    #         _mpit.distribute_indices(list(range(num_params2)), param_processing_subcomm)
                    # else:

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
                        assert(len(myHostsParam2CommIndices) == 1), "Each host should be assigned to exactly 1 param2Comm"
                        myParam2CommIndex = myHostsParam2CommIndices[0]
                        param2_processing_subcomm = param_processing_comm.Split(color=myParam2CommIndex, key=rank) \
                                                    if (param_processing_comm is not None) else None
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
                    self.host_param2_slice= host_param_slice

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
            self.fine_param_subslice = self.host_param_fine_slice = None
            param_processing_subcomm = None
            param2_processing_subcomm = None
            interatom_param_subcomm = None
            interatom_param2_subcomm = None
            param_fine_subcomm = None


        #    #REMOVE
        #    #def _get_param_indices(paramCommIndex):
        #    #    if num_params > num_param_processors or len(additional_dimensions) == 1:
        #    #        #Just divide num_params range into num_param_processors parts
        #    #        paramIndices, hostParamCommOwners, _ = _mpit.distribute_indices_base(
        #    #            list(range(num_params)), num_param_processors, paramCommIndex)
        #    #        param2Indices = list(range(num_params2)) if len(additional_dimensions) > 1 else []
        #    #    else:
        #    #        reduction = (num_param_processors - 1) // num_params + 1
        #    #        paramIndices, hostParamCommOwners, _ = _mpit.distribute_indices_base(
        #    #            list(range(num_params)), num_param_processors // reduction,
        #    #            paramCommIndex // reduction)
        #    #        param2Indices, hostParam2CommOwners, _ = _mpit.distribute_indices_base(
        #    #            list(range(num_params2)), reduction, paramCommIndex % reduction)
        #    #    return paramIndices, param2Indices
        #    #myParamIndices, myParam2Indices = _get_param_indices(myParamCommIndex)
        #
        #    #if len(additional_dimensions) > 1:
        #    #    self.global_num_params2 = num_params2
        #    #    self.global_param2_slice = _slct.list_to_slice(myParam2Indices)
        #    #else:
        #    #    self.global_num_params2 = None
        #    #    self.global_param2_slice = None
        #                            
        #    #if nHosts <= nAtomComms:  # then each host contains one or more *complete* atoms & therefor *all* params
        #    #    self.host_num_params = self.global_num_params
        #    #    self.host_param_slice = self.global_param_slice
        #    #else:
        #    #    # we need to concat param slices of all the param-processing comms on this host
        #    #    _assert_sequential(myHostsParamCommIndices)
        #    #    offset = offset2 = 0
        #    #    host_param_slice = host_param2_slice = None
        #    #    for iParamComm in myHostsParamCommIndices:
        #    #        pinds, _, _ = _mpit.distribute_indices_base(
        #    #            list(range(num_params)), num_param_processors, iParamComm)
        #    #        pinds2 = 
        #    #        #pinds, pinds2 = _get_param_indices(iParamComm)
        #    #
        #    #        if iParamComm == myParamCommIndex:
        #    #            host_param_slice = slice(offset, offset + len(pinds))
        #    #            host_param2_slice = slice(offset2, offset2 + len(pinds2))
        #    #
        #    #        offset += len(pinds)
        #    #        offset2 += len(pinds2)
        #    #    self.host_num_params = offset
        #    #    self.host_num_params2 = offset2
        #    #    self.host_param_slice= host_param_slice
        #    #    self.host_param2_slice = host_param2_slice
        #    #
        #    ## 2nd param direction is always confined to a single host
        #    #self.host_num_params2 = self.global_num_params2
        #    #self.host_param2_slice = self.global_param2_slice
        #else:  # no parameter dimensions
        #    self.host_num_params = None
        #    self.host_num_params2 = None
        #    self.host_param_slice= None
        #    self.host_param2_slice = None
        #    self.global_num_params = None
        #    self.global_num_params2 = None
        #    self.global_param_slice= None
        #    self.global_param2_slice = None
        #    param_processing_subcomm = None

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

        #Each procesor is a part of a "gather group" - either the full world
        # comm (when memory isn't shared) or the interhost_comm (when memory
        # is shared).  These comms are held in the resource_alloc object, but
        # within the layout we store the corresponding indices
        resource_alloc.gather_comm = resource_alloc.interhost_comm if (resource_alloc.host_comm is not None) else comm
        resource_alloc.element_gather_comm

        if resource_alloc.gather_comm is not None:
            self.gather_element_slices = resource_alloc.gather_comm.gather(self.global_element_slice, root=0)
            self.gather_param_slices = resource_alloc.gather_comm.gather(self.global_param_slice, root=0) \
                                       if (self.global_param_slice is not None) else None                                
            self.gather_param2_slices = resource_alloc.gather_comm.gather(self.global_param2_slice, root=0) \
                                        if (self.global_param2_slice is not None) else None
        else:
            self.gather_element_slices = self.gather_param_slices = self.gather_param2_slices = None

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
        for i, (c, cc) in enumerate(zip(unique_circuits, unique_complete_circuits)):
            if i in my_unique_is:
                local_unique_index = len(local_unique_circuits)  # an index into local_unique_circuits
                local_unique_complete_circuits.append(cc)
                local_unique_circuits.append(c)
                start = len(local_circuits)
                local_to_unique.update({start + k: local_unique_index for k in range(len(rev_unique[i]))})
                local_circuits.extend([circuits_dict[orig_i] for orig_i in rev_unique[i]])

        super().__init__(local_circuits, local_unique_circuits, local_to_unique, local_elindex_outcome_tuples,
                         local_unique_complete_circuits, param_dimensions)

        #DEBUG LAYOUT PRINTING
        #def slc_str(slc):
        #    return "%5d -> %5d (%4d)" % (slc.start, slc.stop, slc.stop - slc.start) \
        #        if isinstance(slc, slice) else str(slc)
        #if resource_alloc.host_comm is None:  # gather_comm == comm
        #    if rank == 0:
        #        print("DEBUG - layout root proc will gather:")
        #        pslcs = self.gather_param_slices if (self.gather_param_slices is not None) else \
        #                ['--'] * len(self.gather_element_slices)
        #        pslcs2 = self.gather_param2_slices if (self.gather_param2_slices is not None) else \
        #                ['--'] * len(self.gather_element_slices)
        #        for kk, (eslc, pslc, pslc2) in enumerate(zip(self.gather_element_slices, pslcs, pslcs2)):
        #            print("Host (rank) %d: " % kk, slc_str(eslc), slc_str(pslc), slc_str(pslc2))
        #else:  # gather_comm = interhost_comm
        #    #REMOVE print("DB: ",rank,resource_alloc.interhost_comm.size, resource_alloc.interhost_comm.rank, resource_alloc.host_index)
        #    for intrahost_rank in range(resource_alloc.host_comm.size):
        #        if resource_alloc.host_comm.rank == intrahost_rank and resource_alloc.interhost_comm.rank == 0:
        #            assert(resource_alloc.host_index == 0)  # all rank-0s of interhost_comms are on host 0
        #            print("DEBUG - layout intra-host rank %d (on host 0) will gather:" % intrahost_rank)
        #            pslcs = self.gather_param_slices if (self.gather_param_slices is not None) else \
        #                    ['--'] * len(self.gather_element_slices)
        #            pslcs2 = self.gather_param2_slices if (self.gather_param2_slices is not None) else \
        #                    ['--'] * len(self.gather_element_slices)
        #            for kk, (eslc, pslc, pslc2) in enumerate(zip(self.gather_element_slices, pslcs, pslcs2)):
        #                print("Host %d: " % kk, slc_str(eslc), slc_str(pslc), slc_str(pslc2))
        #        comm.barrier()

        #print("DEBUG EL (%d): host=%s of %d, global=%s of %d" % (rank, str(self.host_element_slice), self.host_num_elements,
        #                                                      str(self.global_element_slice), self.global_num_elements))
        #
        #if self.host_num_params is not None:
        #    print("DEBUG P1 (%d): host=%s of %d, global=%s of %d" % (rank, str(self.host_param_slice), self.host_num_params,
        #                                                             str(self.global_param_slice), self.global_num_params))
        

            
        #OLD
        ## No shared memory - just divide all procs into nAtomComm groups
        #    # (Note: this case also must work when comm = None)
        #    mySubCommIndices, subCommOwners, mySubCommOrRalloc = \
        #        _mpit.distribute_indices(list(range(nAtomComms)), resource_alloc)
        #    assert(len(mySubCommIndices) == 1), "Each rank should be assigned to exactly 1 subComm group"
        #    mySubCommIndex = mySubCommIndices[0]
        #
        #    myAtomIndices, atomOwners = _mpit.distribute_indices_base(
        #        list(range(nAtoms)), nAtomComms, mySubCommIndex)
        #
        #    # atomOwners contains index of owner subComm, but we really want
        #    #  the owning processor, i.e. the owner of the subComm
        #    atomOwners = {iAtom: subCommOwners[atomOwners[iAtom]]
        #                  for iAtom in atomOwners}
    

    #UNUSED - REMOVE?
    #def local_memory_estimate(self, nprocs, array_type, dtype='d'):
    #    """
    #    Per-processor memory required to allocate a local array (an estimate in bytes).
    #    """
    #    #bytes_per_element = _np.dtype(dtype).itemsize
    #    raise NotImplementedError()

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

        if array_type in ('e', 'ep', 'epp'):
            array_shape = (self.host_num_elements + extra_elements,) if self.part_of_final_atom_processor \
                          else (self.host_num_elements,)
            if 'p' in array_type: array_shape += (self.host_num_params,)
            if 'pp' in array_type: array_shape += (self.host_num_params2,)
            allocating_ralloc = resource_alloc  # share mem between these processors
        elif array_type == 'hessian':
            array_shape = (self.host_num_params, self.host_num_params2)
            allocating_ralloc = resource_alloc.layout_allocs['atom-processing']  # don't share mem btwn atoms,
            # as each atom will have procs with the same (param1, param2) index block but we want separate mem
        elif array_type == 'jtj':
            array_shape = (_slct.length(self.fine_param_subslice), self.global_num_params)
            allocating_ralloc = resource_alloc  #.layout_allocs['param-interatom']
        elif array_type == 'jtf':
            array_shape = (_slct.length(self.fine_param_subslice),)
            allocating_ralloc = resource_alloc  #.layout_allocs['param-interatom']
        else:
            raise ValueError("Invalid array_type: %s" % str(array_type))

        host_array, host_array_shm = _smt.create_shared_ndarray(resource_alloc, array_shape, dtype,
                                                                zero_out, track_memory)

        if array_type in ('e', 'ep', 'epp'):
            elslice = slice(self.host_element_slice.start, self.host_element_slice.stop + extra_elements) \
                      if self.part_of_final_atom_processor else self.host_element_slice
            tuple_of_slices = (elslice,)
            if 'p' in array_type: tuple_of_slices += (self.host_param_slice,)
            if 'pp' in array_type: tuple_of_slices += (self.host_param2_slice,)
        elif array_type == 'hessian':
            tuple_of_slices = (self.host_param_slice, self.host_param2_slice)
        elif array_type == 'jtj':
            # REMOVE tuple_of_slices = (self.host_param_slice, slice(0, self.global_num_params))
            tuple_of_slices = (self.host_param_fine_slice, slice(0, self.global_num_params))
        elif array_type == 'jtf':
            # REMOVE tuple_of_slices = (self.host_param_slice,)
            tuple_of_slices = (self.host_param_fine_slice,)

        local_array = host_array[tuple_of_slices]
        assert(local_array.data.contiguous) # make sure contiguous for buffer= to work below?
        local_array = _smt.LocalNumpyArray(local_array.shape, buffer=local_array.dtype,
                                           host_array=host_array,
                                           slices_into_host_array=tuple_of_slices,
                                           shared_memory_handle=host_array_shm)

        return local_array, host_array_shm

    def gather_local_array(self, array_type, array_portion, resource_alloc=None):
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
        #TODO - need to gather the "extra_elements" when they are present,
        # by enlarging the element slice of all the procs in the final atom processor.
        if resource_alloc is None or resource_alloc.gather_comm is None:
            return array_portion

        # Set two resource allocs based on the array_type:
        # gather_ralloc.comm groups the processors that we gather data over.
        # unit_ralloc.comm groups all procs that compute the *same* unit being gathered (e.g. the 
        #  same (atom, param_slice) tuple, so that only the rank=0 procs of this comm need to
        #  participate in the gathering (others would be redundant and set memory multiple times)
        if array_type == 'e':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['atom-processing']
            global_shape = (self.global_num_elements,)
            slice_of_global = self.global_element_slice
        elif array_type == 'ep':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['param-processing']
            global_shape = (self.global_num_elements, self.global_num_params)
            slice_of_global = (self.global_element_slice, self.global_param_slice)
        elif array_type == 'epp':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['param2-processing']
            global_shape = (self.global_num_elements, self.global_num_params, self.global_num_params2)
            slice_of_global = (self.global_element_slice, self.global_param_slice, self.global_param2_slice)
        elif array_type == 'jtj':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['param-fine']
            global_shape = (self.global_num_params, self.global_num_params)
            slice_of_global = (self.fine_param_subslice, slice(0, self.global_num_params))
        elif array_type == 'jtf':
            gather_ralloc = resource_alloc
            unit_ralloc = resource_alloc.layout_allocs['param-fine']
            global_shape = (self.global_num_params,)
            slice_of_global = self.fine_param_subslice
        else:
            raise ValueError("Invalid array type: %s" % str(array_type))

        gather_comm = gather_alloc.interhost_comm if (gather_alloc.host_comm is not None) else gather_alloc.comm
        global_array, global_array_shm = _smt.create_shared_ndarray(
            resource_alloc, global_shape, 'd', track_memory=False) if gather_comm.rank == 0 else (None, None)

        gather_alloc.gather(global_array, array_portion, slice_of_global, unit_ralloc)

        ret = global_array.copy() if resource_alloc.comm.rank == 0 else None # so no longer shared mem (needed?)
        resource_alloc.comm.barrier()  # make sure global_array is copied before we free it
        if gather_comm.rank == 0:
            _smt.cleanup_shared_ndarray(global_array_shm)
        return ret

    # jtf, jtf_shm = self.allocate_local_array('jtf', 'd', resource_alloc, zero_out=False)
    def compute_jtf(self, j, f, jtf, resource_alloc):
        """TODO: docstring  - assumes j, f are local arrays, allocated using 'ep' and 'e' types, respectively.
        Returns an array allocated using the 'jtf' type.
        """
        param_ralloc = resource_alloc.layout_allocs['param-processing']  # this group acts on (element, param) blocks
        # local_jtf = _np.dot(j.T, f)  # need to sum this value across all atoms
        # local_jtf = local_jtf[self.fine_param_subslice]  # take sub-portion to move to "fine" parameter distribution
        local_jtf = _np.dot(j.T[self.fine_param_subslice, :], f)  #equivalent to two commented lines above

        # assume jtf is created from allocate_local_array('jtf', 'd', resource_alloc)
        resource_alloc.layout_allocs['param-fine'].allreduce_sum(jtf, local_jtf, unit_ralloc=param_ralloc)
        
        #if param_comm.host_comm is not None and param_comm.host_comm.rank != 0: 
        #    return None  # this processor doesn't need to do any more - root host proc will fill returned shared mem

    # jtj, jtj_shm = self.allocate_local_array('jtj', 'd', resource_alloc, zero_out=False)
    def compute_jtj(self, j, jtj, resource_alloc):
        """TODO: docstring  - assumes j is a local array, allocated using 'ep' and 'e' types, respectively. 
        Returns an array allocated using the 'jtj' type.
        """
        jT = j.T[self.fine_param_subslice, :] # takes sub-portion to move to "fine" parameter distribution
        param_ralloc = resource_alloc.layout_allocs['param-processing']  # this group acts on (element, param) blocks
        atom_ralloc = resource_alloc.layout_allocs['atom-processing']  # this group acts on (element,) blocks
        atom_jtj = _np.empty((_slct.length(self.host_param_fine_slice), self.global_num_params), 'd')  # for my atomproc
        # REMOVE  atom_jtj, atom_jtj_shm = self.allocate_local_array('jtj', 'd', atom_alloc, zero_out=False)
        for i, param_slice in enumerate(self.param_slices):
            if i == self.my_paramproc_index:
                assert(param_slice == self.global_param_slice)
                assert(self.param_slice_owners[i] == atom_ralloc.comm.rank)
                atom_ralloc.comm.bcast(j, root=atom_ralloc.comm.rank)
                atom_jtj[:, param_slice] = _np.dot(jT, j)
            else:
                other_j = atom_ralloc.comm.bcast(None, root=self.param_slice_owners[i])
                atom_jtj[:, param_slice] = _np.dot(jT, other_j)

        #Now need to sum atom_jtj over atoms to get jtj:
        # assume jtj is created from allocate_local_array('jtj', 'd', resource_alloc)
        resource_alloc.layout_allocs['param-fine'].allreduce_sum(jtj, atom_jtj, unit_ralloc=param_ralloc)
        
                

    #UNUSED / OLD - REMOVE
    #def set_distribution_params(self, num_atom_processing_subcomms, additional_dimension_blk_sizes,
    #                            gather_mem_limit):
    #    self.num_atom_processing_subcomms = num_atom_processing_subcomms
    #    self.additional_dimension_blk_sizes = additional_dimension_blk_sizes
    #    self.gather_mem_limit = gather_mem_limit

    #REMOVE
    #def is_split(self):
    #    """
    #    Whether strategy contains multiple atomic parts (sub-strategies).
    #
    #    Returns
    #    -------
    #    bool
    #    """
    #    return len(self.atoms) > 0

    #REMOVE
    #def distribute(self, comm, verbosity=0):
    #    """
    #    Distributes this strategy's atomic parts across multiple processors.
    #
    #
    #    TODO: update this docstring text (it's outdated):
    #    This function checks how many processors are present in
    #    `comm` and divides this tree's subtrees into groups according to
    #    the number of subtree comms provided as an argument to
    #    `initialize`.  Note that this does *not* always divide the subtrees
    #    among the processors as much as possible, as this is not always what is
    #    desired (computing several subtrees serially using multiple
    #    processors simultaneously can be faster, due to better balancing, than
    #    computing all the subtrees simultaneously using smaller processor groups
    #    for each).
    #
    #    For example, if the number of subtree comms given to
    #    `initialize` == 1, then all the subtrees are assigned to the one and
    #    only processor group containing all the processors of `comm`.  If the
    #    number of subtree comms == 2 and there are 4 subtrees and 8 processors,
    #    then `comm` is divided into 2 groups of 4 processors each, and two
    #    subtrees are assigned to each processor group.
    #
    #    Parameters
    #    ----------
    #    comm : mpi4py.MPI.Comm or ResourceAllocation
    #        When not None, an MPI communicator for distributing subtrees
    #        across processor groups.  Providing a :class:`ResourceAllocation`
    #        causes a :class:`ResourceAllocation` to be returned (when not None)
    #        as `mySubComm` (see below), allowing additional accounting so that
    #        shared memory can be utilized between processors on the same host.
    #
    #    verbosity : int, optional
    #        How much detail to send to stdout.
    #
    #    Returns
    #    -------
    #    myAtomIndices : list
    #        A list of integer indices specifying which atoms this
    #        processor is responsible for.
    #    atomOwners : dict
    #        A dictionary whose keys are integer atom indices and
    #        whose values are processor ranks, which indicates which
    #        processor is responsible for communicating the final
    #        results of each atom.
    #    mySubComm : mpi4py.MPI.Comm or ResourceAllocation or None
    #        The communicator for the processor group that is responsible
    #        for computing the same `myAtomIndices` list.  This
    #        communicator is used for further processor division (e.g.
    #        for parallelization across derivative columns).
    #    """
    #    # split tree into local atoms, each which contains one/group of
    #    # processors (group can then parallelize derivative calcs over
    #    # model parameters)
    #
    #    comm_or_ralloc = comm
    #    if isinstance(comm, _ResourceAllocation):
    #        comm = comm_or_ralloc.comm
    #
    #    #rank = 0 if (comm is None) else comm.Get_rank()
    #    nprocs = 1 if (comm is None) else comm.Get_size()
    #    nAtomComms = self.num_atom_processing_subcomms
    #    nAtoms = len(self.atoms)
    #    assert(nAtomComms <= nAtoms), "Cannot request more sub-comms ({nAtomComms}) than there are atoms ({nAtoms})!"
    #
    #    assert(nAtomComms <= nprocs), "Not enough processors (%d) to make nAtomComms=%d" % (nprocs, nAtomComms)
    #    mySubCommIndices, subCommOwners, mySubCommOrRalloc = \
    #        _mpit.distribute_indices(list(range(nAtomComms)), comm_or_ralloc)
    #    assert(len(mySubCommIndices) == 1), "Each rank should be assigned to exactly 1 subComm group"
    #    mySubCommIndex = mySubCommIndices[0]
    #
    #    myAtomIndices, atomOwners = _mpit.distribute_indices_base(
    #        list(range(nAtoms)), nAtomComms, mySubCommIndex)
    #
    #    # atomOwners contains index of owner subComm, but we really want
    #    #  the owning processor, i.e. the owner of the subComm
    #    atomOwners = {iAtom: subCommOwners[atomOwners[iAtom]]
    #                  for iAtom in atomOwners}
    #
    #    printer = _VerbosityPrinter.create_printer(verbosity, comm)
    #    printer.log("*** Distributing %d atoms into %d sub-comms (%s processors) ***" %
    #                (nAtoms, nAtomComms, nprocs))
    #
    #    #HERE - set these as the indices per *host* comm when there is shared memory
    #    # these are indices of into the global array dimensions.
    #    # there are also indices of each processor into the shared "host comm" space - 
    #    # these are atom.element_slice indices.
    #    # then the "shared mem leader" will need to communicate these indices to other nodes?
    #    # self.my_element_indices 
    #    # self.my_param_indices
    #    # self.my_param_indices2
    #
    #    return myAtomIndices, atomOwners, mySubCommOrRalloc

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
