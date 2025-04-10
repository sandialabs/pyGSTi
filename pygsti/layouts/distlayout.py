"""
Defines the DistributableCOPALayout class.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections

import numpy as _np

from pygsti.layouts.copalayout import CircuitOutcomeProbabilityArrayLayout as _CircuitOutcomeProbabilityArrayLayout
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.tools import mpitools as _mpit
from pygsti.tools import sharedmemtools as _smt
from pygsti.tools import slicetools as _slct


#import time as _time #DEBUG TIMERS


def _assert_sequential(lst):
    if len(lst) > 0:
        last = lst[0]
        for x in lst[1:]:
            assert(last + 1 == x)
            last = x


class _DistributableAtom(object):
    """
    A component of a :class:`DistributableCOPALayout` corresponding to a segment of the element dimension.

    When a distributed layout divides up the work of computing circuit outcome probabilities to
    multiple processors, it divides the element-dimension of this computation (as opposed to the
    0, 1, or 2 parameter-dimensions) into "atoms", which this class encapsulates.  The purpose of
    this class is to house all of the information necessary for computing a slice (the atom's
    `self.element_slice`) of the entire range of elements. In many ways, an atom behaves as a sub-layout.

    Paramters
    ---------
    element_slice : slice
        The "global" indices into the parent layout's element array.

    num_elements : int
        The number of global indices.  If `None`, then the length of `element_slice` is
        computed internally.
    """

    def __init__(self, element_slice, num_elements=None):
        self.element_slice = element_slice
        self.num_elements = _slct.length(element_slice) if (num_elements is None) else num_elements

    def _update_indices(self, old_unique_indices_in_order):
        """
        Updates any internal indices held as a result of the unique-circuit indices of the layout changing.

        This function is called during layout construction to alert the atom that the layout
        being created will only hold a subset of the `unique_complete_circuits` provided to
        to the atom's `__init__` method.  Thus, if the atom keeps indices to unique circuits
        within the layout, it should update these indices accordingly.

        Parameters
        ----------
        old_unique_is_by_new_unique_is : list
            The indices within the `unique_complete_circuits` given to `__init__` that index the
            unique circuits of the created layout - thus, these  that will become (in order) all of
            the unique circuits of the created layout.

        Returns
        -------
        None
        """
        pass  # nothing to be done.

    @property
    def cache_size(self):
        return 0

    def as_layout(self, resource_alloc):
        """
        Convert this atom into a fully-fledged layout.

        This allows the same computation methods that operate on layouts
        to be used on atoms, so that an even more minimal forward-simulator
        implemenation is needed.  This is only needed when a forward simulator
        is used that doesn't implement the `_bulk_fill_*probs_atom` functions
        (e.g. a plain `DistributableForwardSimulator`).

        Parameters
        ----------
        resource_alloc : ResourceAllocation
            The resource allocation object that the created layout should "own".  Atoms
            don't own resource allocation objects like layouts do, so this is needed to
            build an atom into a layout object.

        Returns
        -------
        CircuitOutcomeProbabilityArrayLayout
        """
        raise NotImplementedError("This probably should be implemented, but isn't yet: TODO!")


class DistributableCOPALayout(_CircuitOutcomeProbabilityArrayLayout):
    """
    A circuit-outcome-probability-array (COPA) layout that is distributed among many processors.

    This layout divides the work of computing arrays with one dimension corresponding to the
    layout's "elements" (circuit outcomes) and 0, 1, or 2 parameter dimensions corresponding
    to first or second derivatives of a by-element quantity with respect to a model's parameters.

    The size of element dimension is given by the number of unique circuits and the outcomes
    retained for each circuit.  Computation along the element dimension is broken into "atoms",
    which hold a slice that indexes the element dimension along with the necessary information
    (used by a forward simulator) to compute those elements.  This often includes the circuits
    and outcomes an atom's elements correspond to, and perhaps precomputed structures for speeding
    up the circuit computation.  An atom-creating function is used to initialize a
    :class:`DistributableCOPALayout`.

    Technical note: the atoms themselves determine which outcomes for each circuit are included in
    the layout, so the layout doesn't know how many elements it contains until the atoms are created.
    This makes for an awkward `_update_indices` callback that adjusts an atom's indices based on the
    selected circuits of the (local) layout, since this selection can only be performed after the
    atoms are created.

    The size of the parameter dimensions is given directly via the `param_dimensions` argument. These
    dimensions are divided into "blocks" (slices of the entire dimension) but there is no analogous
    atom-like object for the blocks, as there isn't any need to hold meta-data specific to a block.
    The size of the parameter-blocks is essentially constant along each parameter dimension, and specified
    by the `param_dimension_blk_sizes` argument.

    Along each of the (possible) array dimensions, we also assign a number of atom (for the element
    dimension) or block (for the parameter dimensions) "processors".  These are *not* physical CPUs
    but are logical objects act by processing atoms or blocks, respectively.  A single atom processor
    is assigned one or *more* atoms to process, and similarly with block processors.

    The total number of physical processors, N, is arranged in a grid so that:

    N = num_atom_processors x num_param1_processors x num_param2_processors

    This may restricts the allowed values of N is the number of atom/block processors is
    fixed or constrained.  The reason there are 2 levels of "breaking up" the computation
    are so that intermediate memory may be controlled.  If we merged the notion of atoms
    and atom-processors, for instance, so that each atom processor always had exactly 1
    atom to process, then the only way to divide up a compuation would be to use more
    processors.  Since computations can involve intermediate memory usage that far exceeds
    the memory required to hold the results, it is useful to be able to break up a computation
    into chunks even when there is, e.g., just a single processor.  Separating atom/blocks from
    atom-processors and param-block-processors allow us to divide a computation into chunks
    that use manageable amounts of intermediate memory regardless of the number of processors
    available.  When intermediate memory is not a concern, then there is no reason to assign
    more than one atom/block to it's corresponding processor type.

    When creating a :class:`DistributableCOPALayout` the caller can separately specify the
    number of atoms (length of `create_atom_args`) or the size of parameter blocks and the
    number of atom-processors or the number of param-block-processors.

    Furthermore, a :class:`ResourceAllocation` object can be given that specifies a
    shared-memory structure to the physical processors, where the total number of cores
    is divided into node-groups that are able to share memory.  The total number of
    cores is divided like this:
    
    - first, we divide the cores into atom-processing groups, i.e. "atom-processors".
      An atom-processor is most accurately seen as a comm (group of processors).  If
      shared memory is being used, either the entire atom-processor must be contained
      within a single node OR the atom-processor must contain an integer number of
      nodes (it *cannot* contain a mixed fractional number of nodes, e.g. 1+1/2).
    - each atom processor is divided into param1-processors, which process sections
      arrays within that atom processor's element slice and within the param1-processors
      parameter slice.  Similarly, each param1-processor cannot contain a mixed
      fraction number of nodes - it must be a fraction < 1 or an integer number of nodes.
    - each param1-processor is divided into param2-processors, with exactly the same
      rules as for the param1-processors.

    These nested MPI communicators neatly divide up the entries of arrays that have
    shape (nElements, nParams1, nParams2) or arrays with fewer dimensions, in which
    case processors that would have computed different entries of a missing dimension
    just duplicate the computation of array entries in the existing dimensions.

    Arrays will also be used that do not have a leading `nElements` dimension
    (e.g. when element-contributions have been summed over), with shapes involving
    just the parameter dimensions.  For these arrays, we also construct a "fine"
    processor grouping where all the cores are divided among the (first) parameter
    dimension.  The array types `"jtf"` and `"jtj"` are distributed according to
    this "fine" grouping.

    Parameters
    ----------
    circuits : list of Circuits
        The circuits whose outcome probabilities are to be computed.  This list may
        contain duplicates.

    unique_circuits : list of Circuits
        The same as `circuits`, except duplicates are removed.  Often this value is obtained
        by a derived class calling the class method :meth:`_compute_unique_circuits`.

    to_unique : dict
        A mapping that translates an index into `circuits` to one into `unique_circuits`.
        Keys are the integers 0 to `len(circuits)` and values are indices into `unique_circuits`.

    unique_complete_circuits : list, optional
        A list, parallel to `unique_circuits`, that contains the "complete" version of these
        circuits.  This information is currently unused, and is included for potential future
        expansion and flexibility.

    create_atom_fn: function
        A function that creates an atom when given one of the elements of `create_atom_args`.

    create_atom_args : list
        A list of tuples such that each element is a tuple of arguments for `create_atom_fn`.
        The length of this list specifies the number of atoms, and the caller must provide
        the same list on all processors.  When the layout is created, `create_atom_fn` will
        be used to create some subset of the atoms on each processor.

    num_atom_processors : int
        The number of "atom processors".  An atom processor is not a physical processor, but
        a group of physical processors that is assigned one or more of the atoms (see above).

    num_param_dimension_processors : tuple, optional
        A 1- or 2-tuple of integers specifying how many parameter-block processors (again,
        not physical processors, but groups of processors that are assigned to parameter
        blocks) are used when dividing the physical processors into a grid.  The first and
        second elements correspond to counts for the first and second parameter dimensions,
        respecively.

    param_dimensions : tuple, optional
        The full (global) number of parameters along each parameter dimension.  Can be an
        empty, 1-, or 2-tuple of integers which dictates how many parameter dimensions this
        layout supports.

    param_dimension_blk_sizes : tuple, optional
        The parameter block sizes along each present parameter dimension, so this should
        be the same shape as `param_dimensions`.  A block size of `None` means that there
        should be no division into blocks, and that each block processor computes all of
        its parameter indices at once.

    resource_alloc : ResourceAllocation, optional
        The resources available for computing circuit outcome probabilities.

    verbosity : int or VerbosityPrinter
        Determines how much output to send to stdout.  0 means no output, higher
        integers mean more output.
    """

    def __init__(self, circuits, unique_circuits, to_unique, unique_complete_circuits,
                 create_atom_fn, create_atom_args, num_atom_processors,
                 num_param_dimension_processors=(), param_dimensions=(), param_dimension_blk_sizes=(),
                 resource_alloc=None, verbosity=0):
        resource_alloc = _ResourceAllocation.cast(resource_alloc)
        printer = _VerbosityPrinter.create_printer(verbosity, resource_alloc)
        comm = resource_alloc.comm
        if comm is not None:
            from mpi4py import MPI

        rank = resource_alloc.comm_rank
        nprocs = resource_alloc.comm_size
        nAtomComms = num_atom_processors
        nAtoms = len(create_atom_args)
        printer.log("*** Distributing %d atoms to %d atom-processing groups (%s cores) ***" %
                    (nAtoms, nAtomComms, nprocs))

        assert(nAtomComms <= nAtoms), ("Cannot request more atom-processors (%d) than there are atoms (%d)!"
                                       % (nAtomComms, nAtoms))
        assert(nAtomComms <= nprocs), "Not enough processors (%d) to make nAtomComms=%d" % (nprocs, nAtomComms)

        if resource_alloc.host_comm is not None:
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

        #Create this resource alloc now, as logic below needs to know its host structure
        atom_processing_ralloc = _ResourceAllocation(
            atom_processing_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)
        if resource_alloc.host_comm is not None:  # signals that we want to use shared intra-host memory
            atom_processing_ralloc.build_hostcomms()

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

        # Get global element indices & so some setup for global elindex_outcome_tuples
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
                atoms_dict[i].element_slice = slice(offset - start, stop - start)
                # .element_slice is atom's slice to index into it's *local* array
            offset += atom_sizes[i]
        self.host_num_elements = offset  # total number of elements on this host (useful?)
        self.host_element_slice = slice(start, stop)

        #FUTURE: if we wanted to hold an *index* instead of a host_slice:
        # each atom comm on the same host will hold a unique index giving it's ordering
        # on it's host, indexing its shared_mem_array (~it's slice into a big shared array)
        #self.host_shared_array_index = myHostsAtomCommIndices.index(myAtomCommIndex)
        #self.host_shared_array_size = stop - start

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
            hostindex_by_atomproc_rank = atom_processing_subcomm.allgather(atom_processing_ralloc.host_index) \
                if (atom_processing_subcomm is not None) else [resource_alloc.host_index]

            self.param_slices = _mpit.slice_up_slice(slice(0, num_params),
                                                     num_param_processors)  # matches param comm indices
            self.max_param_slice_length = max([_slct.length(s) for s in self.param_slices])  # useful for buffer sizing
            self.param_slice_owners = {ipc: atomproc_rank for atomproc_rank, ipc
                                       in enumerate(owned_paramCommIndex_by_atomproc_rank) if ipc >= 0}
            self.param_slice_owner_hostindices = {owned_paramCommIndex_by_atomproc_rank[atomproc_rank]: hi
                                                  for atomproc_rank, hi in enumerate(hostindex_by_atomproc_rank)
                                                  if owned_paramCommIndex_by_atomproc_rank[atomproc_rank] >= 0}
            self.my_owned_paramproc_index = owned_paramCommIndex
            # Note: if muliple procs within atomproc com have the same myParamCommIndex (possible when
            #  param_processing_subcomm.size > 1) then the "owner" of a param slice is the
            #  param_processing_subcomm.rank == 0 processor.

            interatom_param_subcomm = comm.Split(color=myParamCommIndex, key=rank) if (comm is not None) else None

            self.global_num_params = num_params
            self.global_param_slice = _slct.list_to_slice(myParamIndices)
            self.num_params = _slct.length(self.global_param_slice)

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

            #Each processor needs to know the host_param_slice of the "owner" of each param slice
            # (i.e. parameter comm/processor) for owners thater are other processors on the same host.
            if atom_processing_ralloc.host_comm is not None:
                host_param_slices_by_intrahost_rank = atom_processing_ralloc.host_comm.allgather(self.host_param_slice)
                owned_paramCommIndex_by_intrahost_rank = \
                    atom_processing_ralloc.host_comm.allgather(owned_paramCommIndex)
                self.my_hosts_param_slices = {ipc: hpc for ipc, hpc in zip(owned_paramCommIndex_by_intrahost_rank,
                                                                           host_param_slices_by_intrahost_rank)
                                              if ipc >= 0}
            else:
                self.my_hosts_param_slices = {owned_paramCommIndex: self.host_param_slice}

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
                if comm is not None:
                    gpss = comm.allgather((gps, hps))  # each host is a single proc; gpss = the single slice per host
                else:
                    gpss = [(gps, hps)]
                self.param_fine_slices_by_host = tuple([(rank_and_gps,) for rank_and_gps in enumerate(gpss)])

            self.param_fine_slices_by_rank = tuple(comm.allgather(self.global_param_fine_slice)) \
                if (comm is not None) else [self.global_param_fine_slice]

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
                self.num_params2 = _slct.length(self.global_param2_slice)

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
                self.num_params2 = None
                param2_processing_subcomm = None
                interatom_param2_subcomm = None

        else:
            self.global_num_params = self.global_param_slice = None
            self.global_num_params2 = self.global_param2_slice = None
            self.host_num_params = self.host_param_slice = None
            self.host_num_params2 = self.host_param2_slice = None
            self.host_num_params_fine = None
            self.num_params = self.num_params2 = None
            self.fine_param_subslice = self.host_param_fine_slice = self.global_param_fine_slice = None
            self.param_fine_slices_by_rank = self.param_fine_slices_by_host = None
            self.owner_host_and_rank_of_global_fine_param_index = None
            param_processing_subcomm = None
            param2_processing_subcomm = None
            interatom_param_subcomm = None
            interatom_param2_subcomm = None
            param_fine_subcomm = None

        # save sub-resource-allocations
        self._sub_resource_allocs = {}  # dict of sub-resource-allocations for use with this layout
        self._sub_resource_allocs['atom-processing'] = atom_processing_ralloc  # created above b/c needed earlier
        self._sub_resource_allocs['param-processing'] = _ResourceAllocation(
            param_processing_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)
        self._sub_resource_allocs['param2-processing'] = _ResourceAllocation(
            param2_processing_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)
        self._sub_resource_allocs['param-interatom'] = _ResourceAllocation(
            interatom_param_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)
        self._sub_resource_allocs['param2-interatom'] = _ResourceAllocation(
            interatom_param2_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)
        self._sub_resource_allocs['param-fine'] = _ResourceAllocation(
            param_fine_subcomm, resource_alloc.mem_limit, resource_alloc.profiler,
            resource_alloc.distribute_method, resource_alloc.allocated_memory)

        if resource_alloc.host_comm is not None:  # signals that we want to use shared intra-host memory
            #self._sub_resource_allocs['atom-processing'].build_hostcomms()  # done above
            self._sub_resource_allocs['param-processing'].build_hostcomms()
            self._sub_resource_allocs['param2-processing'].build_hostcomms()
            self._sub_resource_allocs['param-interatom'].build_hostcomms()
            self._sub_resource_allocs['param2-interatom'].build_hostcomms()
            self._sub_resource_allocs['param-fine'].build_hostcomms()

        self.atoms = [atoms_dict[i] for i in myAtomIndices]
        self.param_dimension_blk_sizes = param_dimension_blk_sizes

        self._global_layout = _CircuitOutcomeProbabilityArrayLayout(circuits, unique_circuits, to_unique,
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

        #Communicate (local) number of circuits to other procs on same host so we can
        # compute the total number of circuits on this host and our offset into it.
        circuit_counts = {myAtomCommIndex: len(local_circuits)}  # keys = atom-processor (atomComm) index
        if comm is not None:
            all_circuit_counts = comm.allgather(circuit_counts if (atom_processing_subcomm is None
                                                                   or atom_processing_subcomm.rank == 0) else {})
            circuit_counts = {}  # replace local qty above with dict containg *all* atom comm indices
            for counts in all_circuit_counts: circuit_counts.update(counts)

        offset = 0
        for iAtomComm in myHostsAtomCommIndices:
            if iAtomComm == myAtomCommIndex:
                self.host_circuit_slice = slice(offset, offset + circuit_counts[iAtomComm])
            offset += circuit_counts[iAtomComm]
        self.host_num_circuits = offset

        # Give atoms a chance to update any indices they have to the *old* unique indices
        # or "original" (full circuit list) indices, as this layout may hold a subset or
        # a different circuit ordering from the "global" layout.
        for atom in self.atoms:
            atom._update_indices(my_unique_is)

        #Store the global-circuit-index of each of this processor's circuits (local_circuits)
        # Note: unlike other quantities (elements, params, etc.), a proc's local circuits are not guaranteed to be
        #  contiguous portions of the global circuit list, so we must use an index array rather than a slice:
        self.global_circuit_indices = _np.concatenate([rev_unique[unique_i] for unique_i in my_unique_is])
        self.global_num_circuits = len(circuits)
        assert(len(self.global_circuit_indices) == _slct.length(self.host_circuit_slice) == len(local_circuits))

        super().__init__(local_circuits, local_unique_circuits, local_to_unique, local_elindex_outcome_tuples,
                         local_unique_complete_circuits, param_dimensions, resource_alloc)


    @property
    def max_atom_elements(self):
        """ The most elements owned by a single atom. """
        if len(self.atoms) == 0: return 0
        return max([atom.num_elements for atom in self.atoms])

    @property
    def max_atom_cachesize(self):
        """ The largest cache size among all this layout's atoms """
        if len(self.atoms) == 0: return 0
        return max([atom.cache_size for atom in self.atoms])

    @property
    def global_layout(self):
        """ The global layout that this layout is or is a part of.  Cannot be comm-dependent. """
        return self._global_layout

    def resource_alloc(self, sub_alloc_name=None, empty_if_missing=True):
        """
        Retrieves the resource-allocation objectfor this layout.

        Sub-resource-allocations can also be obtained by passing a non-None
        `sub_alloc_name`.

        Parameters
        ----------
        sub_alloc_name : str
            The name to retrieve

        empty_if_missing : bool
            When `True`, an empty resource allocation object is returned when
            `sub_alloc_name` doesn't exist for this layout.  Otherwise a
            `KeyError` is raised when this occurs.

        Returns
        -------
        ResourceAllocation
        """
        if sub_alloc_name is None:
            return self._resource_alloc
        if empty_if_missing and sub_alloc_name not in self._sub_resource_allocs:
            if self._resource_alloc:
                return _ResourceAllocation(None, self._resource_alloc.mem_limit,
                                           self._resource_alloc.profiler, self._resource_alloc.distribute_method)
            else:
                return _ResourceAllocation(None)
        return self._sub_resource_allocs[sub_alloc_name]

    def allocate_local_array(self, array_type, dtype, zero_out=False, memory_tracker=None, extra_elements=0):
        """
        Allocate an array that is distributed according to this layout.

        Creates an array for holding elements and/or derivatives with respect
        to model parameters, possibly distributed among multiple processors
        as dictated by this layout.

        Parameters
        ----------
        array_type : ("e", "ep", "ep2", "epp", "p", "jtj", "jtf", "c", "cp", "cp2", "cpp")
            The type of array to allocate, often corresponding to the array shape.  Let
            `nE` be the layout's number of elements, `nP1` and `nP2` be the number of
            parameters we differentiate with respect to (for first and second derivatives),
            and `nC` be the number of circuits.  Then the array types designate the
            following array shapes:
            - `"e"`: (nE,)
            - `"ep"`: (nE, nP1)
            - `"ep2"`: (nE, nP2)
            - `"epp"`: (nE, nP1, nP2)
            - `"p"`: (nP1,)
            - `"jtj"`: (nP1, nP2)
            - `"jtf"`: (nP1,)
            - `"c"`: (nC,)
            - `"cp"`: (nC, nP1)
            - `"cp2"`: (nC, nP2)
            - `"cpp"`: (nC, nP1, nP2)
            Note that, even though the `"p"` and `"jtf"` types are the same shape
            they are used for different purposes and are distributed differently
            when there are multiple processors.  The `"p"` type is for use with
            other element-dimentions-containing arrays, whereas the `"jtf"` type
            assumes that the element dimension has already been summed over.

        dtype : numpy.dtype
            The NumPy data type for the array.

        zero_out : bool, optional
            Whether the array should be zeroed out initially.

        memory_tracker : ResourceAllocation, optional
            If not None, the amount of memory being allocated is added, using
            :meth:`add_tracked_memory` to this resource allocation object.

        extra_elements : int, optional
            The number of additional "extra" elements to append to the element
            dimension, beyond those called for by this layout.  Such additional
            elements are used to store penalty terms that are treated by the
            objective function just like usual outcome-probability-type terms.

        Returns
        -------
        LocalNumpyArray
            An array that looks and acts just like a normal NumPy array, but potentially
            with internal handles to utilize shared memory.
        """
        resource_alloc = self._resource_alloc
        if array_type in ('e', 'ep', 'ep2', 'epp'):
            my_slices = (slice(self.host_element_slice.start, self.host_element_slice.stop + extra_elements),) \
                if self.part_of_final_atom_processor else (self.host_element_slice,)
            array_shape = (self.host_num_elements + extra_elements,) if self.part_of_final_atom_processor \
                else (self.host_num_elements,)
            if array_type in ('ep', 'epp'):
                my_slices += (self.host_param_slice,)
                array_shape += (self.host_num_params,)
            if array_type in ('ep2', 'epp'):
                my_slices += (self.host_param2_slice,)
                array_shape += (self.host_num_params2,)
            allocating_ralloc = resource_alloc  # share mem between these processors
        elif array_type == 'p':
            my_slices = (self.host_param_slice,)
            array_shape = (self.host_num_params,)
            allocating_ralloc = resource_alloc  # share mem between these processors
        #elif array_type == 'atom-hessian':
        #    array_shape = (self.host_num_params, self.host_num_params2)
        #    allocating_ralloc = self.resource_alloc('atom-processing')  # don't share mem btwn atoms,
        #    # as each atom will have procs with the same (param1, param2) index block but we want separate mem
        elif array_type == 'jtj':
            my_slices = (self.host_param_fine_slice, slice(0, self.global_num_params))
            array_shape = (self.host_num_params_fine, self.global_num_params)
            allocating_ralloc = resource_alloc  # self.resource_alloc('param-interatom')
        elif array_type == 'jtf':  # or array_type == 'pfine':
            my_slices = (self.host_param_fine_slice,)
            array_shape = (self.host_num_params_fine,)
            allocating_ralloc = resource_alloc  # self.resource_alloc('param-interatom')
        elif array_type in ('c', 'cp', 'cp2', 'cpp'):
            my_slices = (slice(self.host_circuit_slice.start, self.host_circuit_slice.stop + extra_elements),) \
                if self.part_of_final_atom_processor else (self.host_circuit_slice,)
            array_shape = (self.host_num_circuits + extra_elements,) if self.part_of_final_atom_processor \
                else (self.host_num_circuits,)
            if array_type in ('cp', 'cpp'):
                my_slices += (self.host_param_slice,)
                array_shape += (self.host_num_params,)
            if array_type in ('cp2', 'cpp'):
                my_slices += (self.host_param2_slice,)
                array_shape += (self.host_num_params2,)
            allocating_ralloc = resource_alloc  # share mem between these processors
        else:
            raise ValueError("Invalid array_type: %s" % str(array_type))

        #Previously: create a single array - this is fine, but suffers from slow write speeds when many
        # procs need to write to the memory, even when different regions are written to
        #host_array, host_array_shm = _smt.create_shared_ndarray(allocating_ralloc, array_shape, dtype,
        #                                                        zero_out, memory_tracker)

        #Instead, allocate separate shared arrays for each segment, so they can be written
        # to independently:
        host_array = {}; host_array_shm = {}
        all_slice_tups = [my_slices] if allocating_ralloc.comm is None \
            else allocating_ralloc.comm.allgather(my_slices)
        for slices in all_slice_tups:
            hashed_slices = tuple([_slct.slice_hash(s) for s in slices])
            array_shape = [_slct.length(s) for s in slices]
            if hashed_slices in host_array: continue  # a slice we've already created (some procs target *same* slice)
            host_array[hashed_slices], host_array_shm[hashed_slices] = _smt.create_shared_ndarray(
                allocating_ralloc, array_shape, dtype, zero_out, memory_tracker)

        # (OLD single shared array construction REMOVED)

        my_hashed_slices = tuple([_slct.slice_hash(s) for s in my_slices])
        local_array = host_array[my_hashed_slices]
        local_array = local_array.view(_smt.LocalNumpyArray)
        local_array.host_array = host_array
        local_array.slices_into_host_array = my_slices
        local_array.shared_memory_handle = host_array_shm

        return local_array

    def free_local_array(self, local_array):
        """
        Frees an array allocated by :meth:`allocate_local_array`.

        This method should always be paired with a call to
        :meth:`allocate_local_array`, since the allocated array
        may utilize shared memory, which must be explicitly de-allocated.

        Parameters
        ----------
        local_array : numpy.ndarray or LocalNumpyArray
            The array to free, as returned from `allocate_local_array`.

        Returns
        -------
        None
        """
        if local_array is not None and hasattr(local_array, 'shared_memory_handle'):
            for shm_handle in local_array.shared_memory_handle.values():
                _smt.cleanup_shared_ndarray(shm_handle)

    def gather_local_array_base(self, array_type, array_portion, extra_elements=0, all_gather=False,
                                return_shared=False):
        """
        Gathers an array onto the root processor or all the processors.

        Gathers the portions of an array that was distributed using this
        layout (i.e. according to the host_element_slice, etc. slices in
        this layout).  This could be an array allocated by :meth:`allocate_local_array`
        but need not be, as this routine does not require that `array_portion` be
        shared.  Arrays can be 1, 2, or 3-dimensional.  The dimensions
        are understood to be along the "element", "parameter", and
        "2nd parameter" directions in that order.

        Parameters
        ----------
        array_type : ("e", "ep", "ep2", "epp", "p", "jtj", "jtf", "c", "cp", "cp2", "cpp")
            The type of array to allocate, often corresponding to the array shape.  See
            :meth:`allocate_local_array` for a more detailed description.

        array_portion : numpy.ndarray
            The portion of the final array that is local to the calling
            processor.  This should be a shared memory array when a
            `resource_alloc` with shared memory enabled was used to construct
            this layout.

        extra_elements : int, optional
            The number of additional "extra" elements to append to the element
            dimension, beyond those called for by this layout.  Should match
            usage in :meth:`allocate_local_array`.

        all_gather : bool, optional
            Whether the result should be returned on all the processors (when `all_gather=True`)
            or just the rank-0 processor (when `all_gather=False`).

        return_shared : bool, optional
            Whether the returned array is allowed to be a shared-memory array, which results
            in a small performance gain because the array used internally to gather the results
            can be returned directly. When `True` a shared memory handle is also returned, and
            the caller assumes responsibilty for freeing the memory via
            :func:`pygsti.tools.sharedmemtools.cleanup_shared_ndarray`.

        Returns
        -------
        gathered_array : numpy.ndarray or None
            The full (global) output array on the root (rank=0) processor and
            `None` on all other processors, unless `all_gather == True`, in which
            case the array is returned on all the processors.
        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            Returned only when `return_shared == True`.  The shared memory handle
            associated with `gathered_array`, which is needed to free the memory.
        """
        resource_alloc = self._resource_alloc
        if resource_alloc.comm is None:
            return array_portion

        # Gather the "extra_elements" when they are present,
        # by enlarging the element slice of all the procs in the final atom processor.
        global_num_els = self.global_num_elements + extra_elements
        global_num_circuits = self.global_num_circuits + extra_elements
        if extra_elements > 0 and self.part_of_final_atom_processor:
            global_el_slice = slice(self.global_element_slice.start, self.global_element_slice.stop + extra_elements)
            global_circuit_indices = _np.concatenate([self.global_circuit_indices,
                                                      _np.arange(self.global_num_circuits,
                                                                 self.global_num_circuits + extra_elements)])
            #NOTE: I'm not sure that the concatenation above is correct - we've never tested "lsvec_mode='percircuit'"
            # with penalty terms (extra_elements > 0)
        else:
            global_el_slice = self.global_element_slice
            global_circuit_indices = self.global_circuit_indices

        # Set two resource allocs based on the array_type:
        # gather_ralloc.comm groups the processors that we gather data over.
        # unit_ralloc.comm groups all procs that compute the *same* unit being gathered (e.g. the
        #  same (atom, param_slice) tuple, so that only the rank=0 procs of this comm need to
        #  participate in the gathering (others would be redundant and set memory multiple times)
        if array_type == 'e':
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('atom-processing')
            global_shape = (global_num_els,)
            slice_of_global = global_el_slice
        elif array_type == 'ep':
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('param-processing')
            global_shape = (global_num_els, self.global_num_params)
            slice_of_global = (global_el_slice, self.global_param_slice)
        elif array_type == 'ep2':
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('param2-processing')  # this may not be right...
            global_shape = (global_num_els, self.global_num_params2)
            slice_of_global = (global_el_slice, self.global_param2_slice)
        elif array_type == 'epp':
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('param2-processing')
            global_shape = (global_num_els, self.global_num_params, self.global_num_params2)
            slice_of_global = (global_el_slice, self.global_param_slice, self.global_param2_slice)
        elif array_type == 'p':
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('param-processing')
            global_shape = (self.global_num_params,)
            slice_of_global = (self.global_param_slice,)
        elif array_type == 'jtj':
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('param-fine')
            global_shape = (self.global_num_params, self.global_num_params)
            slice_of_global = (self.global_param_fine_slice, slice(0, self.global_num_params))
        elif array_type == 'jtf':  # or 'x'
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('param-fine')
            global_shape = (self.global_num_params,)
            slice_of_global = self.global_param_fine_slice
        elif array_type == 'c':
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('atom-processing')
            global_shape = (global_num_circuits,)
            slice_of_global = global_circuit_indices  # NOTE: this is not a slice!! (but works as an index, so ok)
        elif array_type == 'cp':
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('param-processing')
            global_shape = (global_num_circuits, self.global_num_params)
            slice_of_global = (global_circuit_indices, self.global_param_slice)
        elif array_type == 'cp2':
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('param2-processing')  # this may not be right...
            global_shape = (global_num_circuits, self.global_num_params2)
            slice_of_global = (global_circuit_indices, self.global_param2_slice)
        elif array_type == 'cpp':
            gather_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('param2-processing')
            global_shape = (global_num_circuits, self.global_num_params, self.global_num_params2)
            slice_of_global = (global_circuit_indices, self.global_param_slice, self.global_param2_slice)
        else:
            raise ValueError("Invalid array type: %s" % str(array_type))

        gather_comm = gather_ralloc.interhost_comm if (gather_ralloc.host_comm is not None) else gather_ralloc.comm
        global_array, global_array_shm = _smt.create_shared_ndarray(
            resource_alloc, global_shape, 'd') if gather_comm.rank == 0 or all_gather else (None, None)

        gather_ralloc.gather_base(global_array, array_portion, slice_of_global, unit_ralloc, all_gather)

        if return_shared:
            resource_alloc.comm.barrier()  # needed for sync (above gather_base call doesn't always gather)
            return global_array, global_array_shm

        ret = global_array.copy() if (resource_alloc.comm.rank == 0 or all_gather) else None  # so no longer shared mem
        resource_alloc.comm.barrier()  # make sure global_array is copied before we free it
        _smt.cleanup_shared_ndarray(global_array_shm)  # ok if _shm is None
        return ret

    def allsum_local_quantity(self, typ, value, use_shared_mem="auto"):
        """
        Gathers an array onto all the processors.

        Gathers the portions of an array that was distributed using this
        layout (i.e. according to the host_element_slice, etc. slices in
        this layout).  This could be an array allocated by :meth:`allocate_local_array`
        but need not be, as this routine does not require that `array_portion` be
        shared.  Arrays can be 1, 2, or 3-dimensional.  The dimensions
        are understood to be along the "element", "parameter", and
        "2nd parameter" directions in that order.

        Parameters
        ----------
        array_portion : numpy.ndarray
            The portion of the final array that is local to the calling
            processor.  This could be a shared memory array, but just needs
            to be of the correct size.

        extra_elements : int, optional
            The number of additional "extra" elements to append to the element
            dimension, beyond those called for by this layout.  Should match
            usage in :meth:`allocate_local_array`.

        return_shared : bool, optional
            If `True` then, when shared memory is being used, the shared array used
            to accumulate the gathered results is returned directly along with its
            shared-memory handle (`None` if shared memory isn't used).  This results
            in a small performance gain.

        Returns
        -------
        result : numpy.ndarray or None
            The full (global) output array.

        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            Returned only when `return_shared == True`.  The shared memory handle
            associated with `result`, which is needed to free the memory.
        """
        resource_alloc = self._resource_alloc
        if typ in ('c', 'e'):  # value depends only on the "circuits" or "elements" of this layout
            sum_ralloc = resource_alloc
            unit_ralloc = self.resource_alloc('atom-processing')
        else:
            raise ValueError("Invalid `typ` argument: %s" % str(typ))

        if use_shared_mem == "auto":
            use_shared_mem = hasattr(value, 'shape')  # test if this is a numpy array
        else:
            assert(not use_shared_mem or hasattr(value, 'shape')), "Only arrays can be summed using shared mem!"

        if use_shared_mem and sum_ralloc.comm is not None:  # create a temporary shared array to sum into
            result, result_shm = _smt.create_shared_ndarray(sum_ralloc, value.shape, 'd')
            sum_ralloc.comm.barrier()  # wait for result to be ready
            sum_ralloc.allreduce_sum(result, value, unit_ralloc=unit_ralloc)
            result = result.copy()  # so it isn't shared memory anymore
            sum_ralloc.comm.barrier()  # don't free result too early
            _smt.cleanup_shared_ndarray(result_shm)
            return result
        else:
            return sum_ralloc.allreduce_sum_simple(value, unit_ralloc=unit_ralloc)

    def fill_jtf(self, j, f, jtf):
        """
        Calculate the matrix-vector product `j.T @ f`.

        Here `j` is often a jacobian matrix, and `f` a vector of objective function term
        values.  `j` and `f` must be local arrays, created with :meth:`allocate_local_array`.
        This function performs any necessary MPI/shared-memory communication when the
        arrays are distributed over multiple processors.

        Parameters
        ----------
        j : LocalNumpyArray
            A local 2D array (matrix) allocated using `allocate_local_array` with the `"ep"`
            (jacobian) type.

        f : LocalNumpyArray
            A local array allocated using `allocate_local_array` with the `"e"` (element array)
            type.

        jtf : LocalNumpyArray
            The result.  This must be a pre-allocated local array of type `"jtf"`.

        Returns
        -------
        None
        """
        param_ralloc = self.resource_alloc('param-processing')  # acts on (element, param) blocks
        interatom_ralloc = self.resource_alloc('param-interatom')  # procs w/same param slice & diff atoms

        if interatom_ralloc.comm is None:  # only 1 atom, so no need to sum below
            jtf[:] = _np.dot(j.T[self.fine_param_subslice, :], f)
            return

        local_jtf = _np.dot(j.T, f)  # need to sum this value across all atoms

        # assume jtf is created from allocate_local_array('jtf', 'd')
        scratch, scratch_shm = _smt.create_shared_ndarray(
            interatom_ralloc, (_slct.length(self.host_param_slice),), 'd')
        interatom_ralloc.comm.barrier()  # wait for scratch to be ready
        interatom_ralloc.allreduce_sum(scratch, local_jtf, unit_ralloc=param_ralloc)
        jtf[:] = scratch[self.fine_param_subslice]  # takes sub-portion to move to "fine" parameter distribution
        interatom_ralloc.comm.barrier()  # don't free scratch too early
        _smt.cleanup_shared_ndarray(scratch_shm)

        #if param_comm.host_comm is not None and param_comm.host_comm.rank != 0:
        #    return None  # this processor doesn't need to do any more - root host proc will fill returned shared mem

    def _allocate_jtj_shared_mem_buf(self):
        """
        Used internally by the DistributedQuantityCalc class.
        """
        interatom_ralloc = self.resource_alloc('param-interatom')  # procs w/same param slice & diff atoms
        buf, buf_shm = _smt.create_shared_ndarray(
            interatom_ralloc, (_slct.length(self.host_param_slice), self.global_num_params), 'd')
        if interatom_ralloc.comm is not None:
            interatom_ralloc.comm.barrier()  # wait for scratch to be ready
        return buf, buf_shm

    def fill_jtj(self, j, jtj, shared_mem_buf=None):
        """
        Calculate the matrix-matrix product `j.T @ j`.

        Here `j` is often a jacobian matrix, so the result is an approximate Hessian.
        This function performs any necessary MPI/shared-memory communication when the
        arrays are distributed over multiple processors.

        Parameters
        ----------
        j : LocalNumpyArray
            A local 2D array (matrix) allocated using `allocate_local_array` with the `"ep"`
            (jacobian) type.

        jtj : LocalNumpyArray
            The result.  This must be a pre-allocated local array of type `"jtj"`.

        Returns
        -------
        None
        """
        param_ralloc = self.resource_alloc('param-processing')  # acts on (element, param) blocks
        atom_ralloc = self.resource_alloc('atom-processing')  # acts on (element,) blocks
        interatom_ralloc = self.resource_alloc('param-interatom')  # procs w/same param slice & diff atoms
        atom_jtj = _np.empty((_slct.length(self.host_param_slice), self.global_num_params), 'd')  # for my atomproc
        buf = _np.empty((self.max_param_slice_length, j.shape[0]), 'd')

        if atom_ralloc.host_comm is not None:
            jT = j.copy().T  # copy so jT is *not* shared mem, which speeds up dot() calls
            # Note: also doing j_i.copy() in the dot call below could speed the dot up even more, but empirically only
            #  slightly, and so it's not worth doing.
            for i, param_slice in enumerate(self.param_slices):  # for each parameter slice <=> param "processor"
                owning_host_index = self.param_slice_owner_hostindices[i]
                if atom_ralloc.host_index == owning_host_index:
                    # then my host contains the i-th parameter slice (= row block of jT)
                    j_i = j if i == self.my_owned_paramproc_index \
                        else j.host_array[_slct.slice_hash(self.host_element_slice),
                                          _slct.slice_hash(self.my_hosts_param_slices[i])]
                    if atom_ralloc.interhost_comm.size > 1:
                        ncols = _slct.length(param_slice)
                        buf[0:ncols, :] = j_i.T  # broadcast *transpose* so buf slice is contiguous
                        atom_ralloc.interhost_comm.Bcast(buf[0:ncols, :], root=owning_host_index)
                    atom_jtj[:, param_slice] = jT @ j_i  # _np.dot(jT, j_i)
                else:
                    ncols = _slct.length(param_slice)
                    atom_ralloc.interhost_comm.Bcast(buf[0:ncols, :], root=owning_host_index)
                    j_i = buf[0:ncols, :].T
                    atom_jtj[:, param_slice] = jT @ j_i  # _np.dot(jT, j_i)
        else:
            jT = j.T
            for i, param_slice in enumerate(self.param_slices):  # for each parameter slice <=> param "processor"
                ncols = _slct.length(param_slice)
                if i == self.my_owned_paramproc_index:
                    assert(param_slice == self.global_param_slice)
                    if atom_ralloc.comm is not None:
                        assert(self.param_slice_owners[i] == atom_ralloc.comm.rank)
                        buf[0:ncols, :] = jT[:, :]  # broadcast *transpose* so buf slice is contiguous
                        atom_ralloc.comm.Bcast(buf[0:ncols, :], root=atom_ralloc.comm.rank)
                        #Note: we only really need to broadcast this to other param_ralloc.comm.rank == 0
                        # procs as these are the only atom_jtj's that contribute in the allreduce_sum below.
                    else:
                        assert(self.param_slice_owners[i] == 0)

                    atom_jtj[:, param_slice] = jT @ j  # _np.dot(jT, j)
                else:
                    atom_ralloc.comm.Bcast(buf[0:ncols, :], root=self.param_slice_owners[i])
                    other_j = buf[0:ncols, :].T
                    atom_jtj[:, param_slice] = jT @ other_j  # _np.dot(jT, other_j)

        #Now need to sum atom_jtj over atoms to get jtj:
        # assume jtj is created from allocate_local_array('jtj', 'd')
        if interatom_ralloc.comm is None or interatom_ralloc.comm.size == 1:  # only 1 atom - nothing to sum!
            #Note: in this case, we could have just done jT = j.T[self.fine_param_subslice, :] above...
            jtj[:, :] = atom_jtj[self.fine_param_subslice, :]  # takes sub-portion to move to "fine" param distribution
            return

        scratch, scratch_shm = self.allocate_jtj_shared_mem_buf() if shared_mem_buf is None else shared_mem_buf
        interatom_ralloc.allreduce_sum(scratch, atom_jtj, unit_ralloc=param_ralloc)
        jtj[:, :] = scratch[self.fine_param_subslice, :]  # takes sub-portion to move to "fine" parameter distribution
        if shared_mem_buf is None:
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
