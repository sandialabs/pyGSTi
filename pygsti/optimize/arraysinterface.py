"""
Implements the ArraysInterface object and supporting functionality.
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
import scipy.linalg as _la
from pygsti.tools import sharedmemtools as _smt


class ArraysInterface(object):
    """
    An interface between pyGSTi's optimization methods and data storage arrays.

    This class provides an abstract interface to algorithms (particularly the Levenberg-Marquardt
    nonlinear least-squares algorithm) for creating an manipulating potentially distributed data
    arrays with types such as "jtj" (Jacobian^T * Jacobian), "jtf" (Jacobian^T * objectivefn_vector),
    and "x" (model parameter vector).  The class encapsulates all the operations on these arrays so
    that the algorithm doesn't need to worry about how the arrays are actually stored in memory,
    e.g. whether shared memory is used or not.
    """
    pass  # just a base class - maybe make an abc abtract class in FUTURE?


class UndistributedArraysInterface(ArraysInterface):
    """
    An arrays interface for the case when the arrays are not actually distributed.

    Parameters
    ----------
    num_global_elements : int
        The total number of objective function "elements", i.e. the size of the
        objective function array `f`.

    num_global_params : int
        The total number of (model) parameters, i.e. the size of the `x` array.
    """

    def __init__(self, num_global_elements, num_global_params):
        self.num_global_elements = num_global_elements
        self.num_global_params = num_global_params

    def allocate_jtf(self):
        """
        Allocate an array for holding a `'jtf'`-type value.

        Returns
        -------
        numpy.ndarray or LocalNumpyArray
        """
        return _np.empty(self.num_global_params, 'd')

    def allocate_jtj(self):
        """
        Allocate an array for holding an approximated Hessian (type `'jtj'`).

        Returns
        -------
        numpy.ndarray or LocalNumpyArray
        """
        return _np.empty((self.num_global_params, self.num_global_params), 'd')

    def allocate_jac(self):
        """
        Allocate an array for holding a Jacobian matrix (type `'ep'`).

        Returns
        -------
        numpy.ndarray or LocalNumpyArray
        """
        return _np.empty((self.num_global_elements, self.num_global_params), 'd')

    def deallocate_jtf(self, jtf):
        """
        Free an array for holding an objective function value (type `'jtf'`).

        Returns
        -------
        None
        """
        pass

    def deallocate_jtj(self, jtj):
        """
        Free an array for holding an approximated Hessian (type `'jtj'`).

        Returns
        -------
        None
        """
        pass

    def deallocate_jac(self, jac):
        """
        Free an array for holding a Jacobian matrix (type `'ep'`).

        Returns
        -------
        None
        """
        pass

    def global_num_elements(self):
        """
        The total number of objective function "elements".

        This is the size/length of the objective function `f` vector.

        Returns
        -------
        int
        """
        return self.num_global_elements

    def jac_param_slice(self, only_if_leader=False):
        """
        The slice into a Jacobian's columns that belong to this processor.

        Parameters
        ----------
        only_if_leader : bool, optional
            If `True`, the current processor's parameter slice is ony returned if
            the processor is the "leader" (i.e. the first) of the processors that
            calculate the same parameter slice.  All non-leader processors return
            the zero-slice `slice(0,0)`.

        Returns
        -------
        slice
        """
        return slice(0, self.num_global_params)

    def jtf_param_slice(self):
        """
        The slice into a `'jtf'` vector giving the rows of owned by this processor.

        Returns
        -------
        slice
        """
        return slice(0, self.num_global_params)

    def param_fine_info(self):
        """
        Returns information regarding how model parameters are distributed among hosts and processors.

        This information relates to the "fine" distribution used in distributed layouts,
        and is needed by some algorithms which utilize shared-memory communication between
        processors on the same host.

        Returns
        -------
        param_fine_slices_by_host : list
            A list with one entry per host.  Each entry is itself a list of
            `(rank, (global_param_slice, host_param_slice))` elements where `rank` is the top-level
            overall rank of a processor, `global_param_slice` is the parameter slice that processor owns
            and `host_param_slice` is the same slice relative to the parameters owned by the host.

        owner_host_and_rank_of_global_fine_param_index : dict
            A mapping between parameter indices (keys) and the owning processor rank and host index.
            Values are `(host_index, processor_rank)` tuples.
        """
        all_params = slice(0, self.num_global_params)
        ranks_and_pslices_for_host0 = [(0, (all_params, all_params))]
        param_fine_slices_by_host = [ranks_and_pslices_for_host0]
        owner_host_and_rank_of_global_fine_param_index = {i: (0, 0) for i in range(self.num_global_params)}
        return param_fine_slices_by_host, \
            owner_host_and_rank_of_global_fine_param_index

    def allgather_x(self, x, global_x):
        """
        Gather a parameter (`x`) vector onto all the processors.

        Parameters
        ----------
        x : numpy.array or LocalNumpyArray
            The input vector.

        global_x : numpy.array or LocalNumpyArray
            The output (gathered) vector.

        Returns
        -------
        None
        """
        global_x[:] = x

    def allscatter_x(self, global_x, x):
        """
        Pare down an already-scattered global parameter (`x`) vector to be just a local `x` vector.

        Parameters
        ----------
        global_x : numpy.array or LocalNumpyArray
            The input vector.  This global vector is already present on all the processors,
            so there's no need to do any MPI communication.

        x : numpy.array or LocalNumpyArray
            The output vector, typically a slice of `global_x`.

        Returns
        -------
        None
        """
        x[:] = global_x

    def scatter_x(self, global_x, x):
        """
        Scatter a global parameter (`x`) vector onto all the processors.

        Parameters
        ----------
        global_x : numpy.array or LocalNumpyArray
            The input vector.

        x : numpy.array or LocalNumpyArray
            The output (scattered) vector.

        Returns
        -------
        None
        """
        x[:] = global_x

    def allgather_f(self, f, global_f):
        """
        Gather an objective funtion (`f`) vector onto all the processors.

        Parameters
        ----------
        f : numpy.array or LocalNumpyArray
            The input vector.

        global_f : numpy.array or LocalNumpyArray
            The output (gathered) vector.

        Returns
        -------
        None
        """
        global_f[:] = f

    def gather_jtj(self, jtj, return_shared=False):
        """
        Gather a Hessian (`jtj`) matrix onto the root processor.

        Parameters
        ----------
        jtj : numpy.array or LocalNumpyArray
            The (local) input matrix to gather.

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
            `None` on all other processors.
        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            Returned only when `return_shared == True`.  The shared memory handle
            associated with `gathered_array`, which is needed to free the memory.
        """
        return (jtj, None) if return_shared else jtj  # gathers just onto the root proc

    def scatter_jtj(self, global_jtj, jtj):
        """
        Scatter a Hessian (`jtj`) matrix onto all the processors.

        Parameters
        ----------
        global_jtj : numpy.ndarray
            The global Hessian matrix to scatter.

        jtj : numpy.ndarray or LocalNumpyArray
            The local destination array.

        Returns
        -------
        None
        """
        jtj[:, :] = global_jtj

    def gather_jtf(self, jtf, return_shared=False):
        """
        Gather a `jtf` vector onto the root processor.

        Parameters
        ----------
        jtf : numpy.array or LocalNumpyArray
            The local input vector to gather.

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
            `None` on all other processors.
        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            Returned only when `return_shared == True`.  The shared memory handle
            associated with `gathered_array`, which is needed to free the memory.
        """
        return (jtf, None) if return_shared else jtf

    def scatter_jtf(self, global_jtf, jtf):
        """
        Scatter a `jtf` vector onto all the processors.

        Parameters
        ----------
        global_jtf : numpy.ndarray
            The global vector to scatter.

        jtf : numpy.ndarray or LocalNumpyArray
            The local destination array.

        Returns
        -------
        None
        """
        jtf[:] = global_jtf

    def global_svd_dot(self, jac_v, minus_jtf):
        """
        Gathers the dot product between a `jtj`-type matrix and a `jtf`-type vector into a global result array.

        This is typically used within SVD-defined basis calculations, where `jac_v` is the "V"
        matrix of the SVD of a jacobian, and `minus_jtf` is the negative dot product between the Jacobian
        matrix and objective function vector.

        Parameters
        ----------
        jac_v : numpy.ndarray or LocalNumpyArray
            An array of `jtj`-type.

        minus_jtf : numpy.ndarray or LocalNumpyArray
            An array of `jtf`-type.

        Returns
        -------
        numpy.ndarray
            The global (gathered) parameter vector `dot(jac_v.T, minus_jtf)`.
        """
        return _np.dot(jac_v.T, minus_jtf)

    def fill_dx_svd(self, jac_v, global_vec, dx):
        """
        Computes the dot product of a `jtj`-type array with a global parameter array.

        The result (`dx`) is a `jtf`-type array.  This is typically used for
        computing the x-update vector in the LM method when using a SVD-defined basis.

        Parameters
        ----------
        jac_v : numpy.ndarray or LocalNumpyArray
            An array of `jtj`-type.

        global_vec : numpy.ndarray
            A global parameter vector.

        dx : numpy.ndarray or LocalNumpyArray
            An array of `jtf`-type.  Filled with `dot(jac_v, global_vec)`
            values.

        Returns
        -------
        None
        """
        dx[:] = _np.dot(jac_v, global_vec)

    def dot_x(self, x1, x2):
        """
        Take the dot product of two `x`-type vectors.

        Parameters
        ----------
        x1, x2 : numpy.ndarray or LocalNumpyArray
            The vectors to operate on.

        Returns
        -------
        float
        """
        return _np.dot(x1, x2)

    def norm2_x(self, x):
        """
        Compute the Frobenius norm squared of an `x`-type vector.

        Parameters
        ----------
        x : numpy.ndarray or LocalNumpyArray
            The vector to operate on.

        Returns
        -------
        float
        """
        return _np.dot(x, x)

    def infnorm_x(self, x):
        """
        Compute the infinity-norm of an `x`-type vector.

        Parameters
        ----------
        x : numpy.ndarray or LocalNumpyArray
            The vector to operate on.

        Returns
        -------
        float
        """
        return _np.linalg.norm(x, ord=_np.inf)  # (max(sum(abs(x), axis=1))) = max(abs(x))

    def max_x(self, x):
        """
        Compute the maximum of an `x`-type vector.

        Parameters
        ----------
        x : numpy.ndarray or LocalNumpyArray
            The vector to operate on.

        Returns
        -------
        float
        """
        return _np.max(x)

    def norm2_f(self, f):
        """
        Compute the Frobenius norm squared of an `f`-type vector.

        Parameters
        ----------
        f : numpy.ndarray or LocalNumpyArray
            The vector to operate on.

        Returns
        -------
        float
        """
        return _np.dot(f, f)

    def norm2_jtj(self, jtj):
        """
        Compute the Frobenius norm squared of an `jtj`-type matrix.

        Parameters
        ----------
        jtj : numpy.ndarray or LocalNumpyArray
            The array to operate on.

        Returns
        -------
        float
        """
        return _np.linalg.norm(jtj)**2

    def norm2_jac(self, j):
        """
        Compute the Frobenius norm squared of an Jacobian matrix (`ep`-type).

        Parameters
        ----------
        j : numpy.ndarray or LocalNumpyArray
            The Jacobian to operate on.

        Returns
        -------
        float
        """
        return _np.linalg.norm(j)

    def fill_jtf(self, j, f, jtf):
        """
        Compute dot(Jacobian.T, f) in supplied memory.

        Parameters
        ----------
        j : numpy.ndarray or LocalNumpyArray
            Jacobian matrix (type `ep`).

        f : numpy.ndarray or LocalNumpyArray
            Objective function vector (type `e`).

        jtf : numpy.ndarray or LocalNumpyArray
            Output array, type `jtf`.  Filled with `dot(j.T, f)` values.

        Returns
        -------
        None
        """
        jtf[:] = _np.dot(j.T, f)

    def fill_jtj(self, j, jtj, shared_mem_buf=None):
        """
        Compute dot(Jacobian.T, Jacobian) in supplied memory.

        Parameters
        ----------
        j : numpy.ndarray or LocalNumpyArray
            Jacobian matrix (type `ep`).

        jtf : numpy.ndarray or LocalNumpyArray
            Output array, type `jtj`.  Filled with `dot(j.T, j)` values.

        shared_mem_buf : tuple or None
            Scratch space of shared memory used to speed up repeated calls to `fill_jtj`.
            If not none, the value returned from :meth:`allocate_jtj_shared_mem_buf`.

        Returns
        -------
        None
        """
        jtj[:, :] = _np.dot(j.T, j)

    def allocate_jtj_shared_mem_buf(self):
        """
        Allocate scratch space to be used for repeated calls to :meth:`fill_jtj`.

        Returns
        -------
        scratch : numpy.ndarray or None
            The scratch array.
        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            The shared memory handle associated with `scratch`, which is needed to
            free the memory.
        """
        return None, None

    def deallocate_jtj_shared_mem_buf(self, jtj_buf):
        """
        Frees the scratch memory allocated by :meth:`allocate_jtj_shared_mem_buf`.

        Parameters
        ----------
        jtj_buf : tuple or None
            The value returned from :meth:`allocate_jtj_shared_mem_buf`
        """
        pass

    def jtj_diag_indices(self, jtj):
        """
        The indices into a `jtj`-type array that correspond to diagonal elements of the global matrix.

        If `jtj` were a global quantity, then this would just be `numpy.diag_indices_from(jtj)`,
        however, it may be more complicated in actuality when different processors hold different
        sections of the global matrix.

        Parameters
        ----------
        jtj : numpy.ndarray or None
            The `jtj`-type array to get the indices with respect to.

        Returns
        -------
        tuple
            A tuple of 1D arrays that can be used to index the elements of `jtj` that
            correspond to diagonal elements of the global jtj matrix.
        """
        return _np.diag_indices_from(jtj)


class ImplicitArraysInterface(ArraysInterface):

    def __init__(self, num_global_elements, num_global_params): 
        self.num_global_elements = num_global_elements 
        self.num_global_params = num_global_params 
        pass
    
    def allocate_jac(self):
        return _np.empty((self.num_global_elements, self.num_global_params), 'd')

    def allocate_jtf(self):
        return _np.empty(self.num_global_params, 'd') 
    
    def jtf_param_slice(self):
        return slice(0, self.num_global_params)
    
    def jac_param_slice(self, only_if_leader=False):
        return slice(0, self.num_global_params)

    def allgather_x(self, x, global_x):
        global_x[:] = x
        return
    
    def allgather_f(self, f, global_f):
        global_f[:] = f
        return

    def allscatter_x(self, global_x, x):
        x[:] = global_x
        return
    
    def dot_x(self, x1, x2):
        return x1 @ x2
    
    def norm2_x(self, x):
        return _la.norm(x)**2

    def infnorm_x(self, x):
        return _la.norm(x, ord=_np.inf)
    
    def max_x(self, x):
        return _np.max(x)
    
    def norm2_f(self, f):
        return _la.norm(f)**2
    
    def fill_jtf(self, j, f, jtf):
        jtf[:] = j.T @ f
        return

    def global_num_elements(self):
        return self.num_global_elements

    """
    Functions where you need to think a bit about how they 
    should be implemented when Jacobians are implict.
    """

    def allocate_jtj(self):
        return _np.empty((self.num_global_params, self.num_global_params), 'd') 
    
    def norm2_jac(self, j):
        return _la.norm(j)**2

    def fill_jtj(self, j, jtj, shared_mem_buff=None):
        assert shared_mem_buff is None or shared_mem_buff == (None, None)
        jtj[:] = j.T @ j
        return

    def jtj_diag_indices(self, jtj):
        # The question of how to implement this
        # is less significant than the question of how
        # to handle people's attempts at indexing into jtj
        # when it's only defined implicitly.
        return _np.diag_indices_from(jtj)

    """No-ops"""

    def deallocate_jac(self, jac): pass

    def deallocate_jtf(self, jtf): pass

    def deallocate_jtj(self, jtj): pass

    def allocate_jtj_shared_mem_buf(self): return None, None
    
    def deallocate_jtj_shared_mem_buf(self, jtj_buff): pass

    """
    Omitted function definitions:
        param_fine_info
            Only called in customsolve.py::_back_substitution.
        gather_jtj
            Only called in customsolve.py::custom_solve.
        gather_jtf
            Only called in customsolve.py::custom_solve.     
        scatter_x
            Only called in customsolve.py::custom_solve.  
        scatter_jtf
            Not called anywhere!
        scatter_jtj
            Called in customlm.py::custom_leastsq, in the codepath
            that uses non-identity damping in the singular value basis.
        global_svd_dot
            Called in customlm.py::custom_leastsq, in the codepath
            that uses non-identity damping in the singular value basis.
        fill_dx_svd
            Called in customlm.py::custom_leastsq, in the codepath
            that uses non-identity damping in the singular value basis.
        norm2_jtj
            Not called anywhere!
    """


class DistributedArraysInterface(ArraysInterface):
    """
    An arrays interface where the arrays are distributed according to a distributed layout.

    Parameters
    ----------
    dist_layout : DistributableCOPALayout
        The layout giving the distribution of the arrays.

    extra_elements : int, optional
        The number of additional objective function "elements" beyond those
        specified by `dist_layout`.  These are often used for penalty terms.
    """

    def __init__(self, dist_layout, lsvec_mode, extra_elements=0):
        from ..layouts.distlayout import DistributableCOPALayout as _DL
        assert(isinstance(dist_layout, _DL))
        self.layout = dist_layout
        self.resource_alloc = self.layout.resource_alloc()
        self.extra_elements = extra_elements
        self.lsvec_mode = lsvec_mode  # e.g. 'normal' or 'circuits'

    def allocate_jtf(self):
        """
        Allocate an array for holding a `'jtf'`-type value.

        Returns
        -------
        numpy.ndarray or LocalNumpyArray
        """
        return self.layout.allocate_local_array('jtf', 'd', extra_elements=self.extra_elements)

    def allocate_jtj(self):
        """
        Allocate an array for holding an approximated Hessian (type `'jtj'`).

        Returns
        -------
        numpy.ndarray or LocalNumpyArray
        """
        return self.layout.allocate_local_array('jtj', 'd', extra_elements=self.extra_elements)

    def allocate_jac(self):
        """
        Allocate an array for holding a Jacobian matrix (type `'ep'`).

        Note: this function is only called when the Jacobian needs to be
        approximated with finite differences.

        Returns
        -------
        numpy.ndarray or LocalNumpyArray
        """
        if self.lsvec_mode == 'normal':
            return self.layout.allocate_local_array('ep', 'd', extra_elements=self.extra_elements)
        elif self.lsvec_mode == 'percircuit':
            return self.layout.allocate_local_array('cp', 'd', extra_elements=self.extra_elements)
        else:
            raise ValueError("Invlid lsvec_mode: %s" % str(self.lsvec_mode))

    def deallocate_jtf(self, jtf):
        """
        Free an array for holding an objective function value (type `'jtf'`).

        Returns
        -------
        None
        """
        self.layout.free_local_array(jtf)  # cleaup shared memory, if it was used

    def deallocate_jtj(self, jtj):
        """
        Free an array for holding an approximated Hessian (type `'jtj'`).

        Returns
        -------
        None
        """
        self.layout.free_local_array(jtj)  # cleaup shared memory, if it was used

    def deallocate_jac(self, jac):
        """
        Free an array for holding a Jacobian matrix (type `'ep'`).

        Returns
        -------
        None
        """
        self.layout.free_local_array(jac)  # cleaup shared memory, if it was used

    def global_num_elements(self):
        """
        The total number of objective function "elements".

        This is the size/length of the objective function `f` vector.

        Returns
        -------
        int
        """
        if self.lsvec_mode == "normal":
            return self.layout.global_num_elements + self.extra_elements
        elif self.lsvec_mode == "percircuit":
            return self.layout.global_num_circuits + self.extra_elements
        else:
            raise ValueError("Invalid lsvec_mode: %s" % str(self.lsvec_mode))

    def jac_param_slice(self, only_if_leader=False):
        """
        The slice into a Jacobian's columns that belong to this processor.

        Parameters
        ----------
        only_if_leader : bool, optional
            If `True`, the current processor's parameter slice is ony returned if
            the processor is the "leader" (i.e. the first) of the processors that
            calculate the same parameter slice.  All non-leader processors return
            the zero-slice `slice(0,0)`.

        Returns
        -------
        slice
        """
        if only_if_leader and not self.layout.resource_alloc('param-processing').is_host_leader:
            return slice(0, 0)  # not the leader of the group of procs computing this same jac portion
        return self.layout.global_param_slice

    def jtf_param_slice(self):
        """
        The slice into a `'jtf'` vector giving the rows of owned by this processor.

        Returns
        -------
        slice
        """
        return self.layout.global_param_fine_slice

    def param_fine_info(self):
        """
        Returns information regarding how model parameters are distributed among hosts and processors.

        This information relates to the "fine" distribution used in distributed layouts,
        and is needed by some algorithms which utilize shared-memory communication between
        processors on the same host.

        Returns
        -------
        param_fine_slices_by_host : list
            A list with one entry per host.  Each entry is itself a list of
            `(rank, (global_param_slice, host_param_slice))` elements where `rank` is the top-level
            overall rank of a processor, `global_param_slice` is the parameter slice that processor owns
            and `host_param_slice` is the same slice relative to the parameters owned by the host.

        owner_host_and_rank_of_global_fine_param_index : dict
            A mapping between parameter indices (keys) and the owning processor rank and host index.
            Values are `(host_index, processor_rank)` tuples.
        """
        return self.layout.param_fine_slices_by_host, \
            self.layout.owner_host_and_rank_of_global_fine_param_index

    def allgather_x(self, x, global_x):
        """
        Gather a parameter (`x`) vector onto all the processors.

        Parameters
        ----------
        x : numpy.array or LocalNumpyArray
            The input vector.

        global_x : numpy.array or LocalNumpyArray
            The output (gathered) vector.

        Returns
        -------
        None
        """
        #TODO: do this more efficiently in future:
        global_x_on_root = self.layout.gather_local_array('jtf', x)
        if self.resource_alloc.comm is not None:
            global_x[:] = self.resource_alloc.comm.bcast(
                global_x_on_root if self.resource_alloc.comm.rank == 0 else None, root=0)
        else:
            global_x[:] = global_x_on_root

    def allscatter_x(self, global_x, x):
        """
        Pare down an already-scattered global parameter (`x`) vector to be just a local `x` vector.

        Parameters
        ----------
        global_x : numpy.array or LocalNumpyArray
            The input vector.  This global vector is already present on all the processors,
            so there's no need to do any MPI communication.

        x : numpy.array or LocalNumpyArray
            The output vector, typically a slice of `global_x`.

        Returns
        -------
        None
        """
        x[:] = global_x[self.layout.global_param_fine_slice]

    def scatter_x(self, global_x, x):
        """
        Scatter a global parameter (`x`) vector onto all the processors.

        Parameters
        ----------
        global_x : numpy.array or LocalNumpyArray
            The input vector.

        x : numpy.array or LocalNumpyArray
            The output (scattered) vector.

        Returns
        -------
        None
        """
        self.scatter_jtf(global_x, x)

    def allgather_f(self, f, global_f):
        """
        Gather an objective funtion (`f`) vector onto all the processors.

        Parameters
        ----------
        f : numpy.array or LocalNumpyArray
            The input vector.

        global_f : numpy.array or LocalNumpyArray
            The output (gathered) vector.

        Returns
        -------
        None
        """
        #TODO: do this more efficiently in future:
        artype = 'c' if self.lsvec_mode == 'percircuit' else 'e'
        global_f_on_root = self.layout.gather_local_array(artype, f, extra_elements=self.extra_elements)
        if self.resource_alloc.comm is not None:
            global_f[:] = self.resource_alloc.comm.bcast(
                global_f_on_root if self.resource_alloc.comm.rank == 0 else None, root=0)
        else:
            global_f[:] = global_f_on_root

    def gather_jtj(self, jtj, return_shared=False):
        """
        Gather a Hessian (`jtj`) matrix onto the root processor.

        Parameters
        ----------
        jtj : numpy.array or LocalNumpyArray
            The (local) input matrix to gather.

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
            `None` on all other processors.
        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            Returned only when `return_shared == True`.  The shared memory handle
            associated with `gathered_array`, which is needed to free the memory.
        """
        # gathers just onto the root proc
        return self.layout.gather_local_array('jtj', jtj, return_shared=return_shared)

    def scatter_jtj(self, global_jtj, jtj):
        """
        Scatter a Hessian (`jtj`) matrix onto all the processors.

        Parameters
        ----------
        global_jtj : numpy.ndarray
            The global Hessian matrix to scatter.

        jtj : numpy.ndarray or LocalNumpyArray
            The local destination array.

        Returns
        -------
        None
        """
        # Don't bother trying to be fancy with shared mem here - we need to send the
        # entire global_jtj from the (single) root proc anyway.
        comm = self.resource_alloc.comm
        if comm is not None:
            jtj[:, :] = comm.scatter([global_jtj[pslc, :] for pslc in self.layout.param_fine_slices_by_rank]
                                     if comm.rank == 0 else None, root=0)
        else:
            jtj[:, :] = global_jtj

    def gather_jtf(self, jtf, return_shared=False):
        """
        Gather a `jtf` vector onto the root processor.

        Parameters
        ----------
        jtf : numpy.array or LocalNumpyArray
            The local input vector to gather.

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
            `None` on all other processors.
        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            Returned only when `return_shared == True`.  The shared memory handle
            associated with `gathered_array`, which is needed to free the memory.
        """
        # gathers just onto the root proc
        return self.layout.gather_local_array('jtf', jtf, return_shared=return_shared)

    def scatter_jtf(self, global_jtf, jtf):
        """
        Scatter a `jtf` vector onto all the processors.

        Parameters
        ----------
        global_jtf : numpy.ndarray
            The global vector to scatter.

        jtf : numpy.ndarray or LocalNumpyArray
            The local destination array.

        Returns
        -------
        None
        """
        # Don't bother trying to be fancy with shared mem here - we need to send the
        # entire global_jtj from the (single) root proc anyway.
        comm = self.resource_alloc.comm
        if comm is not None:
            to_scatter = [global_jtf[pslc] for pslc in self.layout.param_fine_slices_by_rank] \
                if (comm.rank == 0) else None
            jtf[:] = comm.scatter(to_scatter, root=0)
        else:
            jtf[:] = global_jtf

    def global_svd_dot(self, jac_v, minus_jtf):
        """
        Gathers the dot product between a `jtj`-type matrix and a `jtf`-type vector into a global result array.

        This is typically used within SVD-defined basis calculations, where `jac_v` is the "V"
        matrix of the SVD of a jacobian, and `minus_jtf` is the negative dot product between the Jacobian
        matrix and objective function vector.

        Parameters
        ----------
        jac_v : numpy.ndarray or LocalNumpyArray
            An array of `jtj`-type.

        minus_jtf : numpy.ndarray or LocalNumpyArray
            An array of `jtf`-type.

        Returns
        -------
        numpy.ndarray
            The global (gathered) parameter vector `dot(jac_v.T, minus_jtf)`.
        """
        # Assumes jac_v is 'jtj' type and minus_jtf is 'jtf' type.
        # Returns a *global* parameter array that is dot(jac_v.T, minus_jtf)
        local_dot = _np.dot(jac_v.T, minus_jtf)  # (nP, nP_fine) * (nP_fine) = (nP,)

        #Note: Could make this more efficient by being given a shared array like this as the destination
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (jac_v.shape[1],), 'd')
        self.resource_alloc.allreduce_sum(result, local_dot,
                                          unit_ralloc=self.layout.resource_alloc('param-fine'))
        ret = result.copy()
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def fill_dx_svd(self, jac_v, global_vec, dx):
        """
        Computes the dot product of a `jtj`-type array with a global parameter array.

        The result (`dx`) is a `jtf`-type array.  This is typically used for
        computing the x-update vector in the LM method when using a SVD-defined basis.

        Parameters
        ----------
        jac_v : numpy.ndarray or LocalNumpyArray
            An array of `jtj`-type.

        global_vec : numpy.ndarray
            A global parameter vector.

        dx : numpy.ndarray or LocalNumpyArray
            An array of `jtf`-type.  Filled with `dot(jac_v, global_vec)`
            values.

        Returns
        -------
        None
        """
        #  Assumes dx is of type 'jtf' (only locally holds fine param slice)
        #  Assumes jac_v is of type 'jtj' (locally hosts fine param slice rows)
        #  Assumes global_vec is a global parameter vector
        #  fills dx = dot(jac, global_vec
        dx[:] = _np.dot(jac_v, global_vec)  # everything is local in this case!

    def dot_x(self, x1, x2):
        """
        Take the dot product of two `x`-type vectors.

        Parameters
        ----------
        x1, x2 : numpy.ndarray or LocalNumpyArray
            The vectors to operate on.

        Returns
        -------
        float
        """
        # assumes x's are in "fine" mode
        local_dot = _np.array(_np.dot(x1, x2))
        local_dot.shape = (1,)  # for compatibility with allreduce_sum
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_sum(result, local_dot,
                                          unit_ralloc=self.layout.resource_alloc('param-fine'))
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def norm2_x(self, x):
        """
        Compute the Frobenius norm squared of an `x`-type vector.

        Parameters
        ----------
        x : numpy.ndarray or LocalNumpyArray
            The vector to operate on.

        Returns
        -------
        float
        """
        return self.dot_x(x, x)

    def infnorm_x(self, x):  # (max(sum(abs(x), axis=1))) = max(abs(x))
        """
        Compute the infinity-norm of an `x`-type vector.

        Parameters
        ----------
        x : numpy.ndarray or LocalNumpyArray
            The vector to operate on.

        Returns
        -------
        float
        """
        # assumes x's are in "fine" mode
        local_infnorm = _np.array(_np.linalg.norm(x, ord=_np.inf))
        local_infnorm.shape = (1,)  # for compatibility with allreduce_sum
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_max(result, local_infnorm,
                                          unit_ralloc=self.layout.resource_alloc('param-fine'))
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def min_x(self, x):
        """
        Compute the minimum of an `x`-type vector.

        Parameters
        ----------
        x : numpy.ndarray or LocalNumpyArray
            The vector to operate on.

        Returns
        -------
        float
        """
        # assumes x's are in "fine" mode
        local_min = _np.array(_np.min(x))
        local_min.shape = (1,)  # for compatibility with allreduce_sum
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_min(result, local_min,
                                          unit_ralloc=self.layout.resource_alloc('param-fine'))
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def max_x(self, x):
        """
        Compute the maximum of an `x`-type vector.

        Parameters
        ----------
        x : numpy.ndarray or LocalNumpyArray
            The vector to operate on.

        Returns
        -------
        float
        """
        # assumes x's are in "fine" mode
        local_max = _np.array(_np.max(x))
        local_max.shape = (1,)  # for compatibility with allreduce_sum
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_max(result, local_max,
                                          unit_ralloc=self.layout.resource_alloc('param-fine'))
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def norm2_f(self, f):
        """
        Compute the Frobenius norm squared of an `f`-type vector.

        Parameters
        ----------
        f : numpy.ndarray or LocalNumpyArray
            The vector to operate on.

        Returns
        -------
        float
        """
        local_dot = _np.array(_np.dot(f, f))
        local_dot.shape = (1,)  # for compatibility with allreduce_sum
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_sum(result, local_dot,
                                          unit_ralloc=self.layout.resource_alloc('atom-processing'))
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def norm2_jac(self, j):
        """
        Compute the Frobenius norm squared of an Jacobian matrix (`ep`-type).

        Parameters
        ----------
        j : numpy.ndarray or LocalNumpyArray
            The Jacobian to operate on.

        Returns
        -------
        float
        """
        local_norm2 = _np.array(_np.linalg.norm(j)**2)
        local_norm2.shape = (1,)  # for compatibility with allreduce_sum
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_sum(result, local_norm2,
                                          unit_ralloc=self.layout.resource_alloc('param-processing'))
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def norm2_jtj(self, jtj):
        """
        Compute the Frobenius norm squared of an `jtj`-type matrix.

        Parameters
        ----------
        jtj : numpy.ndarray or LocalNumpyArray
            The array to operate on.

        Returns
        -------
        float
        """
        local_norm2 = _np.array(_np.linalg.norm(jtj)**2)
        local_norm2.shape = (1,)  # for compatibility with allreduce_sum
        result, result_shm = _smt.create_shared_ndarray(self.resource_alloc, (1,), 'd')
        self.resource_alloc.allreduce_sum(result, local_norm2,
                                          unit_ralloc=self.layout.resource_alloc('param-fine'))
        ret = result[0]  # "copies" the single returned element
        self.resource_alloc.host_comm_barrier()  # make sure we don't cleanup too quickly
        _smt.cleanup_shared_ndarray(result_shm)
        return ret

    def fill_jtf(self, j, f, jtf):
        """
        Compute dot(Jacobian.T, f) in supplied memory.

        Parameters
        ----------
        j : numpy.ndarray or LocalNumpyArray
            Jacobian matrix (type `ep`).

        f : numpy.ndarray or LocalNumpyArray
            Objective function vector (type `e`).

        jtf : numpy.ndarray or LocalNumpyArray
            Output array, type `jtf`.  Filled with `dot(j.T, f)` values.

        Returns
        -------
        None
        """
        self.layout.fill_jtf(j, f, jtf)

    def fill_jtj(self, j, jtj, shared_mem_buf=None):
        """
        Compute dot(Jacobian.T, Jacobian) in supplied memory.

        Parameters
        ----------
        j : numpy.ndarray or LocalNumpyArray
            Jacobian matrix (type `ep`).

        jtf : numpy.ndarray or LocalNumpyArray
            Output array, type `jtj`.  Filled with `dot(j.T, j)` values.

        shared_mem_buf : tuple or None
            Scratch space of shared memory used to speed up repeated calls to `fill_jtj`.
            If not none, the value returned from :meth:`allocate_jtj_shared_mem_buf`.

        Returns
        -------
        None
        """
        self.layout.fill_jtj(j, jtj, shared_mem_buf)

    def allocate_jtj_shared_mem_buf(self):
        """
        Allocate scratch space to be used for repeated calls to :meth:`fill_jtj`.

        Returns
        -------
        scratch : numpy.ndarray or None
            The scratch array.
        shared_memory_handle : multiprocessing.shared_memory.SharedMemory or None
            The shared memory handle associated with `scratch`, which is needed to
            free the memory.
        """
        return self.layout._allocate_jtj_shared_mem_buf()

    def deallocate_jtj_shared_mem_buf(self, jtj_buf):
        """
        Frees the scratch memory allocated by :meth:`allocate_jtj_shared_mem_buf`.

        Parameters
        ----------
        jtj_buf : tuple or None
            The value returned from :meth:`allocate_jtj_shared_mem_buf`
        """
        buf, buf_shm = jtj_buf
        _smt.cleanup_shared_ndarray(buf_shm)

    def jtj_diag_indices(self, jtj):
        """
        The indices into a `jtj`-type array that correspond to diagonal elements of the global matrix.

        If `jtj` were a global quantity, then this would just be `numpy.diag_indices_from(jtj)`,
        however, it may be more complicated in actuality when different processors hold different
        sections of the global matrix.

        Parameters
        ----------
        jtj : numpy.ndarray or None
            The `jtj`-type array to get the indices with respect to.

        Returns
        -------
        tuple
            A tuple of 1D arrays that can be used to index the elements of `jtj` that
            correspond to diagonal elements of the global jtj matrix.
        """
        global_param_indices = self.layout.global_param_fine_slice
        row_indices = _np.arange(jtj.shape[0])  # row dimension is always smaller
        col_indices = _np.arange(global_param_indices.start, global_param_indices.stop)
        assert(len(row_indices) == len(col_indices))  # checks that global_param_indices is good
        return row_indices, col_indices  # ~ _np.diag_indices_from(jtj)
