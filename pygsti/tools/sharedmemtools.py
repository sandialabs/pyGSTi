"""
Utility functions for working with shared memory
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import os as _os

import numpy as _np

if _os.environ.get('PYGSTI_USE_SHARED_MEMORY', '1') in ('1', 'True', 'true'):
    try:
        #Enables the use of shared memory in Python 3.8+
        from multiprocessing import shared_memory as _shared_memory
        from multiprocessing import resource_tracker as _resource_tracker
    except ImportError:
        _shared_memory = None
        _resource_tracker = None
else:
    _shared_memory = None
    _resource_tracker = None


class LocalNumpyArray(_np.ndarray):
    """
    Numpy array with metadata for referencing how this "local" array is part
    of a larger shared memory array.
    """
    def __new__(subtype, shape=None, dtype=float, buffer=None, offset=0,
                strides=None, order=None, host_array=None, slices_into_host_array=None,
                shared_memory_handle=None):
        obj = super(LocalNumpyArray, subtype).__new__(subtype, shape, dtype,
                                                      buffer, offset, strides,
                                                      order)
        obj.host_array = host_array
        obj.slices_into_host_array = slices_into_host_array
        obj.shared_memory_handle = shared_memory_handle
        return obj

    def __pygsti_reduce__(self):
        reconstruct_fn, reconstruct_args, state = super(LocalNumpyArray, self).__reduce__()

        # Serialized dtypes as strings since the np.dtype object cannot be serialized
        updated_state = []; dtype_arg_indices = []
        for i, v in enumerate(state):
            if isinstance(v, _np.dtype):
                updated_state.append(str(v))
                dtype_arg_indices.append(i)
            else:
                updated_state.append(v)

        #Note: maybe host/shared info should all be set to 'None' since we shouldn't really be serializing these?
        state_as_dict = {'ndarray_state': updated_state,
                         'dtype_arg_indices': dtype_arg_indices,
                         'host_array': self.host_array,
                         'slices_into_host_array': self.slices_into_host_array,
                         'shared_memory_handle': self.shared_memory_handle}
        return (reconstruct_fn, reconstruct_args, state_as_dict)

    def __pygsti_setstate__(self, state_dict):
        dtype_arg_indices = state_dict.get('dtype_arg_indices', [])
        updated_state = [(_np.dtype(v) if (i in dtype_arg_indices) else v)
                         for i, v in enumerate(state_dict.get('ndarray_state', []))]
        self.__setstate__(tuple(updated_state))
        self.host_array = state_dict.get("host_array", None)
        self.slices_into_host_array = state_dict.get("slices_into_host_array", None)
        self.shared_memory_handle = state_dict.get("shared_memory_handle", None)

    def __array_finalize__(self, obj):
        if obj is None: return
        self.host_array = getattr(obj, 'host_array', None)
        self.slices_into_host_array = getattr(obj, 'slices_into_host_array', None)
        self.shared_memory_handle = getattr(obj, 'shared_memory_handle', None)


def shared_mem_is_enabled():
    """
    Whether shared memory functionality is available (Python 3.8+)

    Returns
    -------
    bool
    """
    return bool(_shared_memory is not None)


def create_shared_ndarray(resource_alloc, shape, dtype, zero_out=False, memory_tracker=None):
    """
    Creates a `numpy.ndarray` that is potentially shared between processors.

    A shared memory array is created when `resource_alloc.host_comm` is
    not `None`, in which case it indicates which processors belong to the
    same host and have access to the same shared memory.

    Parameters
    ----------
    resource_alloc : ResourceAllocation
        The resource allocation object containing information about whether
        or not to create shared memory arrays and how to do so (see above).

    shape : tuple
        The shape of the returned array

    dtype : numpy.dtype
        The numpy data type of the returned array.

    zero_out : bool, optional
        Whether to initialize the array to all zeros.  When `True`,
        this function behaves as `numpy.zeros`; when `False` as `numpy.empty`.

    memory_tracker : ResourceAllocation, optional
        If not none, callc `memory_tracker.add_tracked_memory` to track the
        size of the allocated array.

    Returns
    -------
    ar : numpy.ndarray
        The potentially shared-memory array.

    shm : multiprocessing.shared_memory.SharedMemory
        A shared memory object needed to cleanup the shared memory.  If
        a normal array is created, this is `None`.  Provide this to
        :function:`cleanup_shared_ndarray` to ensure `ar` is deallocated properly.
    """
    hostcomm = resource_alloc.host_comm if shared_mem_is_enabled() else None
    nelements = _np.prod(shape)
    if hostcomm is None or nelements == 0:  # Note: shared memory must be for size > 0
        # every processor allocates its own memory
        if memory_tracker is not None: memory_tracker.add_tracked_memory(nelements)
        ar = _np.zeros(shape, dtype) if zero_out else _np.empty(shape, dtype)
        shm = None
    else:
        if memory_tracker: memory_tracker.add_tracked_memory(nelements // hostcomm.size)
        if hostcomm.rank == 0:
            #SharedMemory expects the size to be a python integer of the same type used as
            #system default.
            shm = _shared_memory.SharedMemory(create=True, size= int(nelements * _np.dtype(dtype).itemsize))
            assert(shm.size >= nelements * _np.dtype(dtype).itemsize)  # Note: not always == (minimum shm.size?)
            hostcomm.bcast(shm.name, root=0)
        else:
            shm_name = hostcomm.bcast(None, root=0)
            shm = _shared_memory.SharedMemory(name=shm_name)
            assert(shm.size >= nelements * _np.dtype(dtype).itemsize)
        hostcomm.barrier()  # needed to protect against root proc processing & freeing mem
        # before non-root procs finish .SharedMemory call above.
        ar = _np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        if zero_out: ar.fill(0)
    return ar, shm


def cleanup_shared_ndarray(shm):
    """
    De-allocates a (potentially) shared numpy array, created by :function:`create_shared_ndarray`.

    Parameters
    ----------
    shm : multiprocessing.shared_memory.SharedMemory or None
        The shared memory object to deallocate.  If None, no deallocation is
        is performed.

    Returns
    -------
    None
    """
    if shm is not None:
        #Close and unlink *if* it's still alive, otherwise unregister it so resourcetracker doesn't complain
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            _resource_tracker.unregister('/' + shm.name, 'shared_memory')
