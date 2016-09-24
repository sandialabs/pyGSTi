from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for working with MPI processor distributions"""

import time as _time
import numpy as _np

#TIMER FNS (TODO: move to own module within tools?)
def add_time(timer_dict, timerName, val):
    """ TODO: docstring """
    if timer_dict is not None:
        if timerName in timer_dict:
            timer_dict[timerName] += val
        else:
            timer_dict[timerName] = val


def distribute_indices(indices, comm, allow_split_comm=True):
    """ TODO: docstring """
    if comm is None:
        nprocs, rank = 1, 0
    else:
        nprocs = comm.Get_size()
        rank = comm.Get_rank()

    loc_indices, owners = distribute_indices_base(indices, nprocs, rank,
                                                  allow_split_comm)

    #Split comm into sub-comms when there are more procs than
    # indices, resulting in all procs getting only a 
    # single index and multiple procs getting the *same*
    # (single) index.
    if nprocs > len(indices) and (comm is not None) and allow_split_comm:
        loc_comm = comm.Split(color=loc_indices[0], key=rank)  
    else: 
        loc_comm = None

    return loc_indices, owners, loc_comm


def distribute_indices_base(indices, nprocs, rank, allow_split_comm=True):
    """ TODO: docstring """
    nIndices = len(indices)
    assert(nIndices > 0) #need a special case when == 0?

    if nprocs >= nIndices:
        if allow_split_comm:
            nloc_std =  nprocs // nIndices
            extra = nprocs - nloc_std*nIndices # extra procs
            if rank < extra*(nloc_std+1):
                loc_indices = [ indices[rank // (nloc_std+1)] ]
            else:
                loc_indices = [ indices[
                        extra + (rank-extra*(nloc_std+1)) // nloc_std ] ]
    
            # owners dict gives rank of first (chief) processor for each index
            # (the "owner" of a given index is responsible for communicating
            #  results for that index to the other processors)
            owners = { indices[i]: i*(nloc_std+1) for i in range(extra) }
            owners.update( { indices[i]: extra*(nloc_std+1) + (i-extra)*nloc_std
                             for i in range(extra, nIndices) } )
        else:
            #Not allowed to assign multiple procs the same local index
            # (presumably b/c there is no way to sub-divide the work 
            #  performed for a single index among multiple procs)
            if rank < nIndices:
                loc_indices = [ indices[rank] ]
            else:
                loc_indices = [ ] #extra procs do nothing
            owners = { indices[i]: i for i in range(nIndices) }
            
    else:
        nloc_std =  nIndices // nprocs
        extra = nIndices - nloc_std*nprocs # extra indices
          # so assign (nloc_std+1) indices to first extra procs
        if rank < extra:
            nloc = nloc_std+1
            nstart = rank * (nloc_std+1)
            loc_indices = [ indices[rank // (nloc_std+1)] ]
        else:
            nloc = nloc_std
            nstart = extra * (nloc_std+1) + (rank-extra)*nloc_std
        loc_indices = [ indices[i] for i in range(nstart,nstart+nloc)]

        owners = { } #which rank "owns" each index
        for r in range(extra):
            nstart = r * (nloc_std+1)
            for i in range(nstart,nstart+(nloc_std+1)):
                owners[indices[i]] = r
        for r in range(extra,nprocs):
            nstart = extra * (nloc_std+1) + (r-extra)*nloc_std
            for i in range(nstart,nstart+nloc_std):
                owners[indices[i]] = r

    return loc_indices, owners



def gather_subtree_results(evt, spam_label_rows,
                            gIndex_owners, my_gIndices,
                            result_tup, my_results, comm, timer_dict=None):
    """ TODO: docstring """
    from mpi4py import MPI #not at top so can import pygsti on cluster login nodes

    #Doesn't need to be a member function: TODO - move to
    # an MPI helper class?
    S = evt.num_final_strings() # number of strings
    result_range = list(range(len(result_tup)))

    assert(result_tup[-1].shape[1] == S) #when this isn't true, (e.g. flat==True
    #  # for bulk_dproduct), we need to copy blocks instead of single indices
    #  # in the myFinalToParentFinalMap line below...

    #NEW WAY (TESTING HERE)

    #Perform broadcasts for each spam label in definite order on all procs
    #  (just iterating over spam_label_rows.items() does *not* ensure this)
    spamLabels = sorted(list(spam_label_rows.keys()))
    myRank = comm.Get_rank() if (comm is not None) else 0
    for spamLabel in spamLabels:
        rowIndex = spam_label_rows[spamLabel]
    
        for r in result_range:
            if result_tup[r] is None:
                continue #skip None result_tup elements (will be None on all procs)

            #Figure out which sub-results this processor "owns" - i.e. is
            # responsible for communicating to the other procs.
            li_gi_tups = [] # local-subtree-index, global-subtree-index tuples
            for i,subtree in enumerate(evt.get_sub_trees()):
                if gIndex_owners[i] == myRank:
                    li_gi_tups.append( (my_gIndices.index(i), i) )

            #Compute the maximum number of communication rounds needed
            # (i.e. the max. # of subtrees owned by any single proc)
            if comm is None: 
                nRounds = len(li_gi_tups)
            else:
                tm = _time.time()
                nRounds = comm.allreduce(len(li_gi_tups),MPI.MAX)
                add_time(timer_dict, "MPI IPC2 allreduce", _time.time()-tm)

            #Start sending data in rounds
            for rnd in range(nRounds):
                if rnd < len(li_gi_tups):
                    li,gi = li_gi_tups[rnd]
                    sub_result = my_results[li][spamLabel][r]
                    sz = sub_result.size
                else:
                    gi, sub_result = -1, _np.empty(0, 'd')

                if comm is None:
                    gInds = [gi]; shapes=[sub_result.shape];
                    sizes = [sub_result.size]; displacements=[0]
                    scratch = sub_result.flat
                else:
                    tm = _time.time()
                    gInds = comm.allgather(gi)  #gather global subtree indices into an array
                    shapes = comm.allgather(sub_result.shape)  #gather sizes (integers) into an array
                    sizes = [_np.product(shape) for shape in shapes]
                    displacements = [sum(sizes[:i]) for i in range(len(sizes))] #calc displacements
                    scratch = _np.empty(sum(sizes), 'd')
                    add_time(timer_dict, "MPI IPC2 allgather", _time.time()-tm)

                    tm = _time.time()
                    comm.Allgatherv([sub_result.flatten(), sub_result.size, MPI.F_DOUBLE],
                                    [scratch, sizes, displacements, MPI.F_DOUBLE])
                    add_time(timer_dict, "MPI IPC2 Allgatherv", _time.time()-tm)
                
                #Place each of the communicated segments into place
                tm = _time.time()
                for gi,disp,sz,shape in zip(gInds,displacements,sizes,shapes):
                    if sz == 0: continue
                    sub_result = scratch[disp:disp+sz].reshape(shape)
                    if evt.is_split():
                        subtree = evt.get_sub_trees()[gi]
                        result_tup[r][rowIndex][ subtree.myFinalToParentFinalMap ] = sub_result
                    else: #subtree is actually the entire tree (evt), so just "copy" all
                        result_tup[r][rowIndex] = sub_result
                add_time(timer_dict, "MPI IPC2 copy", _time.time()-tm)
                
    return


    #ORIGINAL
    for i,subtree in enumerate(evt.get_sub_trees()):
        li = my_gIndices.index(i) if (i in my_gIndices) else None

        #Perform broadcasts for each spam label in definite order on all procs
        #  (just iterating over spam_label_rows.items() does *not* ensure this)
        spamLabels = sorted(list(spam_label_rows.keys()))
        for spamLabel in spamLabels:
            rowIndex = spam_label_rows[spamLabel]
            for r in result_range:
                if result_tup[r] is None:
                    continue #skip None result_tup elements

                sub_result = my_results[li][spamLabel][r] \
                    if (li is not None) else None

                if comm is None: #No comm; rank 0 owns everything
                    assert(gIndex_owners[i] == 0)
                else:
                    tm = _time.time()
                    sub_result_shape = sub_result.shape if (sub_result is not None) else None
                    sub_result_shape = comm.bcast(sub_result_shape, root=gIndex_owners[i])
                    add_time(timer_dict, "MPI IPC2 bcast1", _time.time()-tm)
                    tm = _time.time()
                    if sub_result is None: sub_result = _np.empty(sub_result_shape,'d') #can assume type == 'd' so far
                    comm.Bcast(sub_result, root=gIndex_owners[i]) # broadcast w/out pickling (fast)
                    #OLD: sub_result = comm.bcast(sub_result, root=gIndex_owners[i]) #broadcasts *with* pickling (slow)
                    add_time(timer_dict, "MPI IPC2 bcast2", _time.time()-tm)
                    add_time(timer_dict, "MPI IPC2 bcast size", sub_result.size)

                tm = _time.time()
                if evt.is_split():
                    result_tup[r][rowIndex][ subtree.myFinalToParentFinalMap ] = sub_result
                else: #subtree is actually the entire tree (evt), so just "copy" all
                    result_tup[r][rowIndex] = sub_result
                add_time(timer_dict, "MPI IPC2 copy", _time.time()-tm)


def gather_blk_results(nBlocks, blk_owners, my_blkIndices,
                        my_blk_results, comm):
    """ TODO: docstring """

    if comm is None:
        assert(all([x == 0 for x in list(blk_owners.values())]))
        assert(nBlocks == len(my_blk_results))
        all_blk_results = [ my_blk_results ]
        all_blk_indices = [ my_blkIndices ]
    else:
        # gather a list of each processor's list of computed "blocks"
        all_blk_results = comm.allgather(my_blk_results)
        all_blk_indices = comm.allgather(my_blkIndices)

    # edit this list to put blocks in the correct order
    blk_list = []
    for iBlk in range(nBlocks):
        owner_rank = blk_owners[iBlk]
        loc_iBlk = all_blk_indices[owner_rank].index(iBlk)
        blk_list.append( all_blk_results[owner_rank][loc_iBlk] )

    return blk_list

def distribute_for_dot(contracted_dim, comm):
    loc_indices,_,_ = distribute_indices(
        list(range(contracted_dim)), comm, False)

    #Make sure local columns are contiguous
    start,stop = loc_indices[0], loc_indices[-1]+1
    assert(loc_indices == list(range(start,stop)))
    return slice(start, stop) # local column range as a slice

def mpidot(a,b,loc_slice,comm):
    """ TODO: docstring """
    from mpi4py import MPI #not at top so can import pygsti on cluster login nodes
    if comm is None or comm.Get_size() == 1:
        assert(loc_slice == slice(0,b.shape[1]))
        return _np.dot(a,b)

    loc_dot = _np.dot(a[:,loc_slice],b[loc_slice,:])
    result = _np.empty( loc_dot.shape, loc_dot.dtype )
    comm.Allreduce(loc_dot, result, op=MPI.SUM)

    #DEBUG: assert(_np.linalg.norm( _np.dot(a,b) - result ) < 1e-6)
    return result

    #myNCols = loc_col_slice.stop - loc_col_slice.start
    ## Gather pieces of coulomb tensor together
    #nCols = comm.allgather(myNCols)  #gather column counts into an array
    #displacements = _np.concatenate(([0],_np.cumsum(sizes))) #calc displacements
    #
    #result = np.empty(displacements[-1], a.dtype)
    #comm.Allgatherv([CTelsLoc, size, MPI.F_DOUBLE_COMPLEX], \
    #                [CTels, (sizes,displacements[:-1]), MPI.F_DOUBLE_COMPLEX])

    
