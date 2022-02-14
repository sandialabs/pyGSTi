"""
A custom MPI-enabled linear solver.
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

from pygsti.optimize.arraysinterface import UndistributedArraysInterface as _UndistributedArraysInterface
from pygsti.tools import sharedmemtools as _smt
from pygsti.tools import slicetools as _slct

try:
    from ..tools import fastcalc as _fastcalc
except ImportError:
    _fastcalc = None


def custom_solve(a, b, x, ari, resource_alloc, proc_threshold=100):
    """
    Simple parallel Gaussian Elimination with pivoting.

    This function was built to provide a parallel alternative to
    `scipy.linalg.solve`, and can achieve faster runtimes compared
    with the serial SciPy routine when the number of available processors
    and problem size are large enough.

    When the number of processors is greater than `proc_threshold` (below
    this number the routine just calls `scipy.linalg.solve` on the root
    processor) the method works as follows:

    - each processor "owns" some subset of the rows of `a` and `b`.
    - iteratively (over pivot columns), the best pivot row is found, and this row is used to
      eliminate all other elements in the current pivot column.  This procedure operations on
      the joined matrix `a|b`, and when it completes the matrix `a` is in reduced row echelon
      form (RREF).
    - back substitution (trivial because `a` is in *reduced* REF) is performed to find
      the solution `x` such that `a @ x = b`.

    Parameters
    ----------
    a : LocalNumpyArray
        A 2D array with the `'jtj'` distribution, holding the rows of the `a` matrix belonging
        to the current processor.  (This belonging is dictated by the "fine" distribution in
        a distributed layout.)

    b : LocalNumpyArray
        A 1D array with the `'jtf'` distribution, holding the rows of the `b` vector belonging
        to the current processor.

    x : LocalNumpyArray
        A 1D array with the `'jtf'` distribution, holding the rows of the `x` vector belonging
        to the current processor.  This vector is filled by this function.

    ari : ArraysInterface
        An object that provides an interface for creating and manipulating data arrays.

    resource_alloc : ResourceAllocation
        Gives the resources (e.g., processors and memory) available for use.

    proc_threshold : int, optional
        Below this number of processors this routine will simply gather `a` and `b` to a single
        (the rank 0) processor, call SciPy's serial linear solver, `scipy.linalg.solve`, and
        scatter the results back onto all the processors.

    Returns
    -------
    None
    """

    #DEBUG
    #for i in range(a.shape[1]):
    #    print(i, " = ", _np.linalg.norm(a[:,i]))
    #assert(False), "STOP"

    pivot_row_indices = []
    #potential_pivot_indices = list(range(a.shape[0]))  # *local* row indices of rows not already chosen as pivot rows
    potential_pivot_mask = _np.ones(a.shape[0], dtype=bool)  # *local* row indices of rows not already chosen pivot rows
    all_row_indices = _np.arange(a.shape[0])
    my_row_slice = ari.jtf_param_slice()

    comm = resource_alloc.comm
    host_comm = resource_alloc.host_comm
    ok_buf = _np.empty(1, _np.int64)

    if comm is None or isinstance(ari, _UndistributedArraysInterface):
        x[:] = _scipy.linalg.solve(a, b, assume_a='pos')
        return

    #Just gather everything to one processor and compute there:
    if comm.size < proc_threshold and a.shape[1] < 10000:
        # We're not exactly sure where scipy is better, but until we speed up / change gaussian-elim
        # alg the scipy alg is much faster for small numbers of procs and so should be used unless
        # A is too large to be gathered to the root proc.
        global_a, a_shm = ari.gather_jtj(a, return_shared=True)
        global_b, b_shm = ari.gather_jtf(b, return_shared=True)
        #global_a = ari.gather_jtj(a)
        #global_b = ari.gather_jtf(b)
        if comm.rank == 0:
            try:
                global_x = _scipy.linalg.solve(global_a, global_b, assume_a='pos')
                ok_buf[0] = 1  # ok
            except _scipy.linalg.LinAlgError as e:
                ok_buf[0] = 0  # failure!
                err = e
        else:
            global_x = None
            err = _scipy.linalg.LinAlgError("Linear solver fail on root proc!")  # just in case...

        comm.Bcast(ok_buf, root=0)
        if ok_buf[0] == 0:
            _smt.cleanup_shared_ndarray(a_shm)
            _smt.cleanup_shared_ndarray(b_shm)
            raise err  # all procs must raise in sync

        ari.scatter_x(global_x, x)
        _smt.cleanup_shared_ndarray(a_shm)
        _smt.cleanup_shared_ndarray(b_shm)
        return

    if host_comm is not None:
        shared_floats, shared_floats_shm = _smt.create_shared_ndarray(
            resource_alloc, (host_comm.size,), 'd')
        shared_ints, shared_ints_shm = _smt.create_shared_ndarray(
            resource_alloc, (max(host_comm.size, 3),), _np.int64)
        shared_rowb, shared_rowb_shm = _smt.create_shared_ndarray(
            resource_alloc, (a.shape[1] + 1,), 'd')

        # Scratch buffers
        host_index_buf = _np.empty((resource_alloc.interhost_comm.size, 2), _np.int64) \
            if resource_alloc.interhost_comm.rank == 0 else None
        host_val_buf = _np.empty((resource_alloc.interhost_comm.size, 1), 'd') \
            if resource_alloc.interhost_comm.rank == 0 else None
    else:
        shared_floats = shared_ints = shared_rowb = None

        host_index_buf = _np.empty((comm.size, 1), _np.int64) if comm.rank == 0 else None
        host_val_buf = _np.empty((comm.size, 1), 'd') if comm.rank == 0 else None

    # Idea: bring a|b into RREF then back-substitute to get x.

    # for each column, find the best "pivot" row to use to eliminate other rows.
    # (note: column pivoting is not used
    a_orig = a.copy()  # So we can restore original values of a and b
    b_orig = b.copy()  # (they're updated as we go)

    #Scratch space
    local_pivot_rowb = _np.empty(a.shape[1] + 1, 'd')
    smbuf1 = _np.empty(1, 'd')
    smbuf2 = _np.empty(2, _np.int64)
    smbuf3 = _np.empty(3, _np.int64)

    for icol in range(a.shape[1]):

        # Step 1: find the index of the row that is the best pivot.
        # each proc looks for its best pivot (Note: it should not consider rows already pivoted on)
        potential_pivot_indices = all_row_indices[potential_pivot_mask]
        ibest_global, ibest_local, h, k = _find_pivot(a, b, icol, potential_pivot_indices, my_row_slice,
                                                      shared_floats, shared_ints, resource_alloc, comm, host_comm,
                                                      smbuf1, smbuf2, smbuf3, host_index_buf, host_val_buf)

        # Step 2: proc that owns best row (holds that row and is root of param-fine comm) broadcasts it
        pivot_row, pivot_b = _broadcast_pivot_row(a, b, ibest_local, h, k, shared_rowb, local_pivot_rowb,
                                                  potential_pivot_mask, resource_alloc, comm, host_comm)

        if abs(pivot_row[icol]) < 1e-6:
            # There's no non-zero element in this column to use as a pivot - the column is all zeros.
            # By convention, we just set the corresponding x-value to zero (below) and don't need to do Step 3.
            # NOTE: it's possible that a previous pivot row could have a non-zero element in the icol-th column,
            #  and we could still get here (because we don't consider previously chosen rows as pivot-row candidtate).
            #  But this is ok, since we set the corresponding x-values to 0 so the end result is effectively in RREF.
            pivot_row_indices.append(-1)
            continue

        pivot_row_indices.append(ibest_global)

        # Step 3: all procs update their rows based on the pivot row (including `b`)
        #  - need to update non-pivot rows to eliminate iCol-th entry: row -= alpha * pivot_row
        #    where alpha = row[iCol] / pivot_row[iCol]
        # (Note: don't do this when there isn't a nonzero pivot)
        ipivot_local = ibest_global - my_row_slice.start  # *local* row index of pivot row (ok if negative)
        _update_rows(a, b, icol, ipivot_local, pivot_row, pivot_b)

    # Back substitution:
    # We've accumulated a list of (global) row indices of the rows containing a nonzero
    # element in a given column and zeros in prior columns.
    pivot_row_indices = _np.array(pivot_row_indices)
    _back_substitution(a, b, x, pivot_row_indices, my_row_slice, ari, resource_alloc, host_comm)

    a[:, :] = a_orig  # restore original values of a and b
    b[:] = b_orig    # so they're the same as when we were called.
    # Note: maybe we could just use array copies in the algorithm, but we may need to use the
    # real a and b because they can be shared mem (check?)

    if host_comm is not None:
        _smt.cleanup_shared_ndarray(shared_floats_shm)
        _smt.cleanup_shared_ndarray(shared_ints_shm)
        _smt.cleanup_shared_ndarray(shared_rowb_shm)
    return


def _find_pivot(a, b, icol, potential_pivot_inds, my_row_slice, shared_floats, shared_ints,
                resource_alloc, comm, host_comm, buf1, buf2, buf3, best_host_indices, best_host_vals):

    if len(potential_pivot_inds) > 0:
        best_abs_local_potential_pivot, ibest_local = _restricted_abs_argmax(a[:, icol], potential_pivot_inds)
        #abs_local_potential_pivots = _np.abs(a[potential_pivot_inds, icol])
        #ibest_local_pivot = _np.argmax(abs_local_potential_pivots)  # an index into abs_local_potential_pivots
        #ibest_local = potential_pivot_inds[ibest_local_pivot]  # a *local* row index
        #best_abs_local_potential_pivot = abs_local_potential_pivots[ibest_local_pivot]
    else:
        ibest_local = 0  # these don't matter since we should never be selected as the winner
        best_abs_local_potential_pivot = -1  # dummy -1 value (so it won't be chosen as the max)

    ibest_local_as_global = ibest_local + my_row_slice.start  # a *global* row index (but a local "best")

    if host_comm is not None:  # Shared memory case:

        # procs send best element and row# to root
        shared_floats[host_comm.rank] = best_abs_local_potential_pivot
        shared_ints[host_comm.rank] = ibest_local_as_global
        host_comm.barrier()
        if host_comm.rank == 0:
            k = _np.argmax(shared_floats[0: host_comm.size])  # winning rank within host_comm
            ibest_host = shared_ints[k]

            if resource_alloc.interhost_comm.size > 1:
                #best_host_vals = resource_alloc.interhost_comm.gather(shared_floats[k], root=0)
                #best_host_indices = resource_alloc.interhost_comm.gather((ibest_host, k), root=0)

                buf1[0] = shared_floats[k]
                resource_alloc.interhost_comm.Gather(buf1, best_host_vals, root=0)

                buf2[0] = ibest_host; buf2[1] = k
                resource_alloc.interhost_comm.Gather(buf2, best_host_indices, root=0)
            else:
                best_host_vals[0, :] = [shared_floats[k]]  # best *host* values
                best_host_indices[0, :] = (ibest_host, k)  # and indices

            if comm.rank == 0:
                h = _np.argmax(best_host_vals)  # winning host index
                ibest_global, k = best_host_indices[h]  # chosen (global) pivot row index; updates k = winner on *h*

                assert(resource_alloc.interhost_comm.rank == 0)  # this should be the root proc
                if resource_alloc.interhost_comm.size > 1:
                    buf3[:] = (ibest_global, h, k)
                    resource_alloc.interhost_comm.Bcast(buf3, root=0)
            else:
                resource_alloc.interhost_comm.Bcast(buf3, root=0)
                ibest_global, h, k = buf3[:]

            shared_ints[0] = ibest_global
            shared_ints[1] = h
            shared_ints[2] = k
            host_comm.barrier()
        else:
            host_comm.barrier()
            ibest_global = shared_ints[0]
            h = shared_ints[1]
            k = shared_ints[2]

    else:  # Simpler, no shared memory case:

        # procs send best element and row# to root
        #best_local_vals = comm.gather(best_abs_local_potential_pivot, root=0)
        #best_local_gindices = comm.gather(ibest_local_as_global, root=0)  # but *global* indices
        buf1[0] = best_abs_local_potential_pivot
        best_local_vals = best_host_vals  # each proc is a "host"
        comm.Gather(buf1, best_local_vals, root=0)

        buf1[0] = ibest_local_as_global
        best_local_gindices = best_host_indices  # each proc is a "host"
        comm.Gather(buf1, best_local_gindices, root=0)

        # root proc determines best global pivot and broadcasts row# to others (& it's recorded for later)
        if comm.rank == 0:
            k = _np.argmax(best_local_vals)  # winning & by fiat "owning" rank within comm
            ibest_global = best_local_gindices[k]  # chosen (global) pivot row index
            buf2[0] = ibest_global; buf2[1] = k
            comm.Bcast(buf2, root=0)
        else:
            comm.Bcast(buf2, root=0)
            ibest_global, k = buf2[:]
        h = None

    return ibest_global, ibest_local, h, k


def _broadcast_pivot_row(a, b, ibest_local, h, k, shared_rowb, local_pivot_rowb,
                         potential_pivot_mask, resource_alloc, comm, host_comm):
    if host_comm is not None:
        if host_comm.rank == k:  # the k-th processor on each host communicate the pivot row
            # (one of these "k-th" processors, namely the one on host `h`, holds the pivot row)
            if resource_alloc.interhost_comm.rank == h:
                local_pivot_rowb[0:a.shape[1]] = a[ibest_local, :]
                local_pivot_rowb[a.shape[1]] = b[ibest_local]
                potential_pivot_mask[ibest_local] = False

            if resource_alloc.interhost_comm.size > 1:
                resource_alloc.interhost_comm.Bcast(local_pivot_rowb, root=h)  # pivot row -> sh'd mem
            shared_rowb[:] = local_pivot_rowb
        host_comm.barrier()  # wait (on each host) until shared_row is filled
        pivot_rowb = shared_rowb
    else:
        if comm.rank == k:
            local_pivot_rowb[0:a.shape[1]] = a[ibest_local, :]
            local_pivot_rowb[a.shape[1]] = b[ibest_local]
            potential_pivot_mask[ibest_local] = False
        comm.Bcast(local_pivot_rowb, root=k)
        pivot_rowb = local_pivot_rowb

    pivot_row, pivot_b = pivot_rowb[0:a.shape[1]], pivot_rowb[a.shape[1]]
    return pivot_row, pivot_b


if _fastcalc is None:
    def _update_rows(a, b, icol, ipivot_local, pivot_row, pivot_b):
        for i in range(a.shape[0]):
            if i == ipivot_local: continue  # don't update the pivot row!
            row = a[i, :]
            alpha = row[icol] / pivot_row[icol]
            row[:] -= alpha * pivot_row
            b[i] = b[i] - alpha * pivot_b
            #assert(abs(row[icol]) < 1e-6), " Pivot did not eliminate row %d: %g" % (i, row[icol])
            row[icol] = 0.0  # this sometimes isn't exactly true because of finite precision error,
            # but we know it must be exactly 0

    def _restricted_abs_argmax(ar, restrict_to):
        i = _np.argmax(_np.abs(ar[restrict_to]))
        indx = restrict_to[i]
        return abs(ar[indx]), indx
else:
    _update_rows = _fastcalc.faster_update_rows
    _restricted_abs_argmax = _fastcalc.restricted_abs_argmax


def _back_substitution(a, b, x, pivot_row_indices, my_row_slice, ari, resource_alloc, host_comm):
    ##x[n-1] = b[pivot_row_indices[n-1]] / a[pivot_row[n-1], n-1]

    # x_indices_host = XXX  # x values to send to other procs -- TODO: slice of SHARED host array
    # x_values_host
    # x_indices = _np.empty(_slct.length(my_row_slice), _np.int64)
    # x_valuess = _np.empty(_slct.length(my_row_slice), 'd')
    xval_buf = _np.empty(1, 'd')  # for MPI Send/Recv to work

    #pivot_row_col_dict = {row: col for col, row in enuerate(pivot_row_indices)}
    #for ii, i in enumerate(range(my_row_slice.start, my_row_slice.stop)):
    #    j = pivot_row_col_dict[i]
    #    xval = b[ii] / a[ii, j]
    #
    #    if my_row_slice.start <= j < my_row_slice.stop:
    #        x[j - my_row_slice.start] = xval
    #
    #    #x_indices[ii] = j
    #    #x_values[ii] = xval

    # now need to send the x-values we computed locally to the appropriate processor
    # i.e. *we* need to recive the x-values for x-indices == my_row_slice.
    # Algorithm: all procs loop through *global* list of indices by destination processor.
    #    If this proc *is* the destination processor, then (if it isn't also the source)
    #    it needs to receive from the source processor.  If this proc is the source, it
    #    needs to send its value to the destination processor.
    #  If shared memory is used, first do this within the host, then do again for
    #    only inter-host transfers.
    comm = resource_alloc.comm
    my_host_index = resource_alloc.host_index if (host_comm is not None) else 0
    my_rank = comm.rank
    param_fine_slices_by_host, owner_host_and_rank_of_global_fine_param_index = ari.param_fine_info()
    for col_host_index, ranks_and_pslices in enumerate(param_fine_slices_by_host):
        for col_rank, (gpslice, hpslice) in ranks_and_pslices:
            if gpslice is None: continue

            for p in range(gpslice.start, gpslice.stop):
                irow = pivot_row_indices[p]  # index of the row whose data computes x[p] (p = global param index)

                if irow == -1:  # signals a non-zero pivot could not be found => x-value = 0
                    if my_rank == col_rank:  # (my_host_index == col_host_index is implied)
                        assert(my_row_slice.start <= p < my_row_slice.stop)
                        x[p - my_row_slice.start] = 0
                    continue

                row_host_index, row_rank = \
                    owner_host_and_rank_of_global_fine_param_index[irow]

                if my_rank == row_rank:  # then I'm the source (my_host_index == row_host_index is implied)
                    #Compute the x-value we need, since I own it
                    assert(my_row_slice.start <= irow < my_row_slice.stop)
                    local_irow = irow - my_row_slice.start
                    if abs(a[local_irow, p]) >= 1e-6:
                        xval = b[local_irow] / a[local_irow, p]
                    else:
                        if abs(b[local_irow]) < 1e-6:
                            xval = 0  # convention - just zero-out x-values corresponding to 0 * x = 0
                        else:
                            assert(False), "Singular matrix => %g * x = %g!" % (a[local_irow, p], b[local_irow])

                    #Send it to the destination using the fastest way possible
                    if my_host_index == col_host_index:  # then destination is on same host (yay!)
                        if my_rank == col_rank:  # then we own everything - no need to transfer
                            assert(my_row_slice.start <= p < my_row_slice.stop)
                            x[p - my_row_slice.start] = xval
                        else:
                            # use shared mem to place x directly into destination
                            host_x = x.host_array  # the larger shared array that x is a portion of
                            plocal = p - gpslice.start  # local to the destination proc
                            host_x[_slct.slice_hash(hpslice), ][plocal] = xval
                            # note: index to host_x is always a tuple of hashed slices (even when there's just one)

                    else:  # destination is on different host - need to use MPI
                        xval_buf[0] = xval
                        comm.Send(xval_buf, dest=col_rank, tag=1234)

                elif my_rank == col_rank:  # (my_host_index == col_host_index is implied)
                    assert(my_row_slice.start <= p < my_row_slice.stop)
                    if my_host_index != row_host_index:  # otherwise src did it for us (shared mem)
                        comm.Recv(xval_buf, source=row_rank, tag=1234)
                        x[p - my_row_slice.start] = xval_buf[0]


#NOTE: this implementation is partly done, and was stopped after realizing
# that the given reference had a row/col typo and really A should be split into
# *rows* not columns, and this makes the algorithm not so useful for us.  Kept
# around for MPI usage reference.
def _tallskinny_custom_solve(a, b, resource_alloc):
    """
    Note
    ----
    Based on "Parallel QR algorithm for data-driven decompositions" by Sayadi et al.
    (Center for Turbulence Research 335 Proceedings of the Summer Program 2014)
    """
    from mpy4py import MPI
    assert(a.shape[0] >= a.shape[1]), "This routine assumes tall-skinny matrices!"
    # Note: the assertion above is needed because we assume below that the R matrices
    # returned from scipy.qr have shape (N,N) where the input is shape (M,N), which is
    # only true when M >= N (see scipy docstring for 'economic' mode).

    #Perform parallel QR decomposition of A (`a`)
    comm = resource_alloc.comm
    rank = comm.rank
    nColsPerProc = int(_np.ceil(a.shape[1] / comm.size))
    if rank < nColsPerProc * comm.size:
        loc_col_slice = slice(rank * nColsPerProc, (rank + 1) * nColsPerProc)
    else:
        loc_col_slice = None  # don't use this processor - we need a uniform distribution of rows

    # Step 1: perform local QR decomp on this processor's columns of A
    if loc_col_slice is not None:
        Ai = a[:, loc_col_slice]
        Q1i, Ri = _scipy.linalg.qr(Ai, mode='economic', check_finite=True)
        Ri = _np.ascontiguousarray(Ri)
        assert(Ri.shape == (nColsPerProc, nColsPerProc))  # follows from M >= N assertion above
    else:
        Q1i = Ri = _np.empty((0, 0), 'd')

    # Step 2: gather all Ri matrices onto root proc (or host),
    # perform a local QR decomp there, and scatter resulting Q2i matrices.
    sizes = comm.gather(Ri.size, root=0)
    if comm.rank == 0:
        displacements = _np.concatenate(([0], _np.cumsum(sizes)))
        Rprime = _np.empty(displacements[-1], 'd')
        comm.Gatherv(Ri, [Rprime, sizes, displacements[0:-1], MPI.DOUBLE], root=0)
        Rprime.shape = (displacements[-1] // nColsPerProc, nColsPerProc)
        Q2, R = _scipy.linalg.qr(Rprime, mode='economic', check_finite=True)
        Q2 = _np.ascontiguousarray(Q2)
        Q2_sizes = _np.array([(s // nColsPerProc) * Q2.shape[1] for s in sizes], 'd')
        Q2_displacements = _np.concatenate(([0], _np.cumsum(Q2_sizes)))
        Q2i = _np.empty((Ri.shape[0], Q2.shape[1]), 'd')
        comm.Scatterv([Q2, Q2_sizes, Q2_displacements[0:-1], MPI.DOUBLE], Q2i, root=0)
    else:
        comm.Gatherv(Ri, [None, None, None, MPI.DOUBLE], root=0)
        Q2i = _np.empty((Ri.shape[0], Ri.shape[0]), 'd')  # assume Q2.shape[1] == Ri.shape[0]
        comm.Scatterv([None, None, None, MPI.DOUBLE], Q2i, root=0)

    # Step 3:  all processors performs a simple dot product to get
    # the pieces of the final Q matrix
    Qi = _np.dot(Q1i, Q2i)

    # we could gather Qi => Q, but just need b' = Q.T * b, so compute:
    bprime_i = _np.dot(Qi.T, b)

    # Step 4: gather b'_i => b' on root proc (or host) then solve Rx = b'
    # (on root) via back-substitution.
    if comm.rank == 0:
        #nActiveProcs = a.shape[1] / nColsPerProc
        sizes = Qi.shape[1]
    #TODO: finish with something like:
    #comm.Gatherv(bprime_i, [bprime, sizes, displacements[0:-1], MPI.DOUBLE], root=0)
    bprime = comm.gather(bprime_i, root=0)
    bprime
    #if comm.rank == 0:
    #    x = back_substitute(R, bprime)
    #x = comm.bcast(x, root=0)
    #return x
