"""
Implements the DistributedQuantityCalc object and supporting functionality.
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
from ..tools import sharedmemtools as _smt


class UndistributedQuantityCalc(object):
    """
    A helper class for the custom LM method that encapsulates all the calculations involving
    potentially distributed-memory arrays.

    This class implements the case when the arrays are not distributed.
    """

    def __init__(self, num_global_elements, num_global_params):
        self.num_global_elements = num_global_elements
        self.num_global_params = num_global_params

    def allocate_jtf(self):
        return _np.empty(self.num_global_params, 'd')

    def allocate_jtj(self):
        return _np.empty((self.num_global_params, self.num_global_params), 'd')

    #def allocate_x_for_jac(self):
    #    return _np.empty(self.num_global_params, 'd')

    def allocate_jac(self):
        return _np.empty((self.num_global_elements, self.num_global_params), 'd')

    def deallocate_jtf(self, jtf):
        pass

    def deallocate_jtj(self, jtj):
        pass

    #def deallocate_x_for_jac(self, x_for_jac):
    #    pass

    def deallocate_jac(self, jac):
        pass

    #def convert_x_for_jac(self, x):
    #    return x, None

    #def global_x_size(self):
    #    return self.num_global_params

    def global_num_elements(self):
        return self.num_global_elements

    def jac_param_slice(self, only_if_leader=False):
        return slice(0, self.num_global_params)

    def jtf_param_slice(self):
        return slice(0, self.num_global_params)

    def param_fine_info(self):
        all_params = slice(0, self.num_global_params)
        ranks_and_pslices_for_host0 = [(0, (all_params, all_params))]
        param_fine_slices_by_host = [ranks_and_pslices_for_host0]
        owner_host_and_rank_of_global_fine_param_index = {i: (0, 0) for i in range(self.num_global_params)}
        return param_fine_slices_by_host, \
            owner_host_and_rank_of_global_fine_param_index

    def allgather_x(self, x, global_x):
        global_x[:] = x

    def allscatter_x(self, global_x, x):
        x[:] = global_x

    def allgather_f(self, f, global_f):
        global_f[:] = f

    def gather_jtj(self, jtj):
        return jtj  # gathers just onto the root proc

    def scatter_jtj(self, global_jtj, jtj):
        jtj[:, :] = global_jtj

    def global_svd_dot(self, jac_v, minus_jtf):
        return _np.dot(jac_v.T, minus_jtf)

    def fill_dx_svd(self, jac_v, global_vec, dx):
        dx[:] = _np.dot(jac_v, global_vec)

    def dot_x(self, x1, x2):
        return _np.dot(x1, x2)

    def norm2_x(self, x):
        return _np.dot(x, x)

    def infnorm_x(self, x):
        return _np.linalg.norm(x, ord=_np.inf)  # (max(sum(abs(x), axis=1))) = max(abs(x))

    def max_x(self, x):
        return _np.max(x)

    def norm2_f(self, f):
        return _np.dot(f, f)

    def norm2_jac(self, j):
        return _np.linalg.norm(j)

    def fill_jtf(self, j, f, jtf):
        jtf[:] = _np.dot(j.T, f)

    def fill_jtj(self, j, jtj):
        jtj[:, :] = _np.dot(j.T, j)

    def jtj_diag_indices(self, jtj):
        return _np.diag_indices_from(jtj)

    #def fill_x_for_jac(self, x, x_for_jac):
    #    x_for_jac[:] = x


class DistributedQuantityCalc(object):
    """
    A helper class for the custom LM method that encapsulates all the calculations involving
    potentially distributed-memory arrays.

    This class implements the case when the arrays are distributed according to a distributed layout.
    """

    def __init__(self, dist_layout, resource_alloc, extra_elements=0):
        from ..objects.distlayout import DistributableCOPALayout as _DL
        assert(isinstance(dist_layout, _DL))
        self.layout = dist_layout
        self.resource_alloc = resource_alloc
        self.extra_elements = extra_elements

    def allocate_jtf(self):
        local_array, redundant_shm = self.layout.allocate_local_array('jtf', 'd', self.resource_alloc,
                                                                      track_memory=False,
                                                                      extra_elements=self.extra_elements)
        return local_array

    def allocate_jtj(self):
        local_array, redundant_shm = self.layout.allocate_local_array('jtj', 'd', self.resource_alloc,
                                                                      track_memory=False,
                                                                      extra_elements=self.extra_elements)
        return local_array

    #def allocate_x_for_jac(self):
    #    local_array, redundant_shm = self.layout.allocate_local_array('p', 'd', self.resource_alloc,
    #                                                                  track_memory=False, extra_elements=0)
    #    return local_array

    def allocate_jac(self):
        local_array, redundant_shm = self.layout.allocate_local_array('ep', 'd', self.resource_alloc,
                                                                      track_memory=False,
                                                                      extra_elements=self.extra_elements)
        return local_array

    def deallocate_jtf(self, jtf):
        _smt.cleanup_shared_ndarray(jtf.shared_memory_handle)  # cleaup shared memory, if it was used

    def deallocate_jtj(self, jtj):
        _smt.cleanup_shared_ndarray(jtj.shared_memory_handle)  # cleaup shared memory, if it was used

    #def deallocate_x_for_jac(self, x_for_jac):
    #    _smt.cleanup_shared_ndarray(x_for_jac.shared_memory_handle)  # cleaup shared memory, if it was used

    def deallocate_jac(self, jac):
        _smt.cleanup_shared_ndarray(jac.shared_memory_handle)  # cleaup shared memory, if it was used

    #def fill_x_for_jac(self, x, x_for_jac):
    #    # need to gather fine-param-slices => param_slice
    #    interatom_param_ralloc = self.resource_alloc.layout_allocs['param-interatom']
    #    fine_param_ralloc = self.resource_alloc.layout_allocs['param-fine']
    #    interatom_param_ralloc.gather(x_for_jac, x, self.layout.fine_param_subslice,
    #                                  unit_ralloc=fine_param_ralloc)

    #def global_x_size(self):
    #    return self.layout.global_num_params

    def global_num_elements(self):
        return self.layout.global_num_elements + self.extra_elements

    def jac_param_slice(self, only_if_leader=False):
        if only_if_leader and not self.resource_alloc.layout_allocs['param-processing'].is_host_leader:
            return slice(0, 0)  # not the leader of the group of procs computing this same jac portion
        return self.layout.global_param_slice

    def jtf_param_slice(self):
        return self.layout.global_param_fine_slice

    def param_fine_info(self):
        return self.layout.param_fine_slices_by_host, \
            self.layout.owner_host_and_rank_of_global_fine_param_index

    def allgather_x(self, x, global_x):
        #TODO: do this more efficiently in future:
        global_x_on_root = self.layout.gather_local_array('jtf', x, self.resource_alloc)
        if self.resource_alloc.comm is not None:
            global_x[:] = self.resource_alloc.comm.bcast(
                global_x_on_root if self.resource_alloc.comm.rank == 0 else None, root=0)
        else:
            global_x[:] = global_x_on_root

    def allscatter_x(self, global_x, x):
        x[:] = global_x[self.layout.global_param_fine_slice]

    def allgather_f(self, f, global_f):
        #TODO: do this more efficiently in future:
        global_f_on_root = self.layout.gather_local_array('e', f, self.resource_alloc,
                                                          extra_elements=self.extra_elements)
        if self.resource_alloc.comm is not None:
            global_f[:] = self.resource_alloc.comm.bcast(
                global_f_on_root if self.resource_alloc.comm.rank == 0 else None, root=0)
        else:
            global_f[:] = global_f_on_root

    def gather_jtj(self, jtj):
        # gathers just onto the root proc
        return self.layout.gather_local_array('jtj', jtj, self.resource_alloc)

    def scatter_jtj(self, global_jtj, jtj):
        # Don't bother trying to be fancy with shared mem here - we need to send the
        # entire global_jtj from the (single) root proc anyway.
        comm = self.resource_alloc.comm
        if comm is not None:
            jtj[:, :] = comm.scatter([global_jtj[pslc, :] for pslc in self.layout.param_fine_slices_by_rank]
                                     if comm.rank == 0 else None, root=0)
        else:
            jtj[:, :] = global_jtj

    def global_svd_dot(self, jac_v, minus_jtf):
        # Assumes jac_v is 'jtj' type and minus_jtf is 'jtf' type.
        # Returns a *global* parameter array that is dot(jac_v.T, minus_jtf)
        local_dot = _np.dot(jac_v.T, minus_jtf)  # (nP, nP_fine) * (nP_fine) = (nP,)

        #Note: Could make this more efficient by being given a shared array like this as the destination
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (jac_v.shape[1],), 'd')
        self.resource_alloc.allreduce_sum(result, local_dot,
                                          unit_ralloc=self.resource_alloc.layout_allocs['param-fine'])
        ret = result.copy()
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def fill_dx_svd(self, jac_v, global_vec, dx):
        #  Assumes dx is of type 'jtf' (only locally holds fine param slice)
        #  Assumes jac_v is of type 'jtj' (locally hosts fine param slice rows)
        #  Assumes global_vec is a global parameter vector
        #  fills dx = dot(jac, global_vec
        dx[:] = _np.dot(jac_v, global_vec)  # everything is local in this case!

    def dot_x(self, x1, x2):
        # assumes x's are in "fine" mode
        local_dot = _np.dot(x1, x2)
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_sum(result, local_dot,
                                          unit_ralloc=self.resource_alloc.layout_allocs['param-fine'])
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def norm2_x(self, x):
        return self.dot_x(x, x)

    def infnorm_x(self, x):  # (max(sum(abs(x), axis=1))) = max(abs(x))
        # assumes x's are in "fine" mode
        local_infnorm = _np.linalg.norm(x, ord=_np.inf)
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_max(result, local_infnorm,
                                          unit_ralloc=self.resource_alloc.layout_allocs['param-fine'])
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def max_x(self, x):
        # assumes x's are in "fine" mode
        local_max = _np.max(x)
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_max(result, local_max,
                                          unit_ralloc=self.resource_alloc.layout_allocs['param-fine'])
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def norm2_f(self, f):
        local_dot = _np.dot(f, f)
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_sum(result, local_dot,
                                          unit_ralloc=self.resource_alloc.layout_allocs['atom-processing'])
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def norm2_jac(self, j):
        local_norm2 = _np.linalg.norm(j)**2
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_sum(result, local_norm2,
                                          unit_ralloc=self.resource_alloc.layout_allocs['param-processing'])
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def norm2_jtj(self, jtj):
        local_norm2 = _np.linalg.norm(jtj)**2
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_sum(result, local_norm2,
                                          unit_ralloc=self.resource_alloc.layout_allocs['param-fine'])
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def fill_jtf(self, j, f, jtf):
        self.layout.fill_jtf(j, f, jtf, self.resource_alloc)

    def fill_jtj(self, j, jtj):
        self.layout.fill_jtj(j, jtj, self.resource_alloc)

    def jtj_diag_indices(self, jtj):
        global_param_indices = self.layout.global_param_fine_slice
        row_indices = _np.arange(jtj.shape[0])  # row dimension is always smaller
        col_indices = _np.arange(global_param_indices.start, global_param_indices.stop)
        assert(len(row_indices) == len(col_indices))  # checks that global_param_indices is good
        return row_indices, col_indices  # ~ _np.diag_indices_from(jtj)
