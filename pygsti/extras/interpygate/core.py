"""
Defines interpolated gate and factory classes
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
import pickle as _pickle
from scipy.interpolate import LinearNDInterpolator as _linND
from scipy.linalg import logm as _logm, expm as _expm
import itertools as _itertools
import copy as _copy
import pathlib as _pathlib

from ...tools.basistools import change_basis as _change_basis
from ...tools import optools as _ot
from ...objects.operation import DenseOperator as _DenseOperator
from ...objects.opfactory import OpFactory as _OpFactory

#TODO REMOVE / INCORPORATE
# import dill
#try:
#    from mpi4py import MPI
#    _comm = MPI.COMM_WORLD
#    _rank = _comm.Get_rank()
#    _size = _comm.Get_size()
#except ImportError:
#    _comm = None

__version__ = '0.9.0'

#TODO: replace with VerbosityPrinter usage
def _print(l):
    if _rank == 0:
        print(l)

#TODO move elsewhere?
def _split(n, a, cast_to_array=True):
    k, m = divmod(len(a), n)
    lst = list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    return _np.array(lst) if cast_to_array else lst

def _flatten(x):
    try:
        return [b for a in x for b in a]
    except TypeError:
        return None


class PhysicalProcess(object):

    def __init__(self, num_params, process_shape, aux_shape=None, num_params_evaluated_as_group=0,
                 can_compute_errorgens_directly=False):
        self.num_params = num_params
        self.process_shape = process_shape
        self.aux_shape = aux_shape  # None means no aux data
        self.can_compute_errorgens_directly = can_compute_errorgens_directly
        self.num_params_evaluated_as_group = num_params_evaluated_as_group

    def create_process_matrix(self, v, time=None, comm=None):
        raise NotImplementedError("Derived classes must implement create_process_matrix!")

    #def create_target_matrix(self, v, time=None, comm=None):
    #    raise NotImplementedError("Derived classes must implement create_target_matrix!")
    #
    #def create_inverse_target_matrix(self, v, time=None, comm=None):
    #    return _np.linalg.inv(self.create_target_matrix(v, time, comm))

    def create_errorgen_matrix(self, v, time=None, comm=None):
        raise NotImplementedError("Derived classes must implement create_errorgen_matrix!")

    def create_process_matrices(self, v, times=None, comm=None):
        return _np.stack([self.create_process_matrix(v, t, comm) for t in times], axis=0)

    #def create_target_matrices(self, v, times=None, comm=None):
    #    return _np.stack([self.create_target_matrix(v, t, comm) for t in times], axis=0)
    #
    #def create_inverse_target_matrices(self, v, times=None, comm=None):
    #    return _np.stack([self.create_inverse_target_matrix(v, t, comm) for t in times], axis=0)

    def create_errorgen_matrices(self, v, times=None, comm=None):
        return _np.stack([self.create_errorgen_matrix(v, t, comm) for t in times], axis=0)

    def create_aux_data(self, v, times, comm=None):
        raise NotImplementedError("Derived classes must implement create_aux_data!")

    def create_aux_datum(self, v, comm=None):
        raise NotImplementedError("Derived classes must implement create_aux_datum!")


class OpPhysicalProcess(PhysicalProcess):

    def __init__(self, op, is_errorgen=False):
        self.op = op
        super().__init__(op.num_params, has_aux_data=False, can_compute_errorgens_directly=is_errorgen)

    def create_process_matrix(self, v, time=None, comm=None):
        assert(not self.can_compute_errorgens_directly), "Cannot get process matrix of an *error generator* process!"
        self.op.from_vector(v)
        if time is not None: self.op.set_time(time)
        return self.op.to_dense()

    def create_errorgen_matrix(self, v, time=None, comm=None):
        assert(self.can_compute_errorgens_directly), "Cannot get error generator of an normal physical process!"
        self.op.from_vector(v)
        if time is not None: self.op.set_time(time)
        return self.op.to_dense()


class InterpolatedOpFactory(_OpFactory):

    @classmethod
    def create_by_interpolating_physical_process(cls, target_factory, physical_process, argument_ranges,
                                                 parameter_ranges, argument_indices=None, comm=None,
                                                 mpi_workers_per_process=1, verbosity=0):
        nargs = len(argument_ranges)
        if argument_indices is None:
            argument_indices = _np.arange(nargs, dtype=int)
        else:
            argument_indices = _np.array(argument_indices, dtype=int)
        param_indices = _np.array(sorted(set(range(physical_process.num_params)) - set(argument_indices)), dtype=int)

        ngroups = physical_process.num_params_evaluated_as_group
        process_shape = physical_process.process_shape
        if physical_process.can_compute_errorgens_directly:
            if ngroups > 0:
                def fn(v, grouped_v, comm):
                    return physical_process.create_errorgen_matrices(v, grouped_v, comm=comm)
            else:
                def fn(v, comm):
                    return physical_process.create_errorgen_matrix(v, comm=comm)
        else:
            if ngroups > 0:
                def fn(v, grouped_v, comm):
                    grouped_dims = tuple(map(len, grouped_v))
                    ret = _np.empty(grouped_dims + process_shape, 'd')
                    process_mxs = physical_process.create_process_matrices(v, grouped_v, comm=comm)
                    assert(process_mxs.shape == ret.shape)

                    for index_tup, gv in zip(_itertools.product(*[range(d) for d in grouped_dims]),
                                             _itertools.product(*grouped_v)):
                        fullv = _np.concatenate((v, gv))
                        args = fullv[argument_indices]
                        params = fullv[param_indices]

                        target_op = target_factory.create_op(args, sslbls=None)
                        target_op.from_vector(params[0:target_op.num_params])
                        target_mx = target_op.to_dense()
                        
                        ret[index_tup] = _ot.error_generator(process_mxs[index_tup], target_mx, "pp", "logGTi-quick")
                    return ret
            else:
                def fn(v, comm):
                    args = v[argument_indices]
                    params = v[param_indices]
                    target_op = target_factory.create_op(args, sslbls=None)
                    target_op.from_vector(params[0:target_op.num_params])
                    target_mx = target_op.to_dense()
                    process_mx = physical_process.create_process_matrix(v, comm=comm)
                    return _ot.error_generator(process_mx, target_mx, "pp", "logGTi-quick")

        ranges = [None]*(len(argument_ranges) + len(parameter_ranges))
        for i, arg_range in zip(argument_indices, argument_ranges): ranges[i] = arg_range
        for i, param_range in zip(param_indices, parameter_ranges): ranges[i] = param_range
            
        base_interp_builder = InterpolatedQuantityFactory(fn, process_shape, ranges, None, ngroups)
        base_interpolator = base_interp_builder.build(comm, mpi_workers_per_process, verbosity)

        if physical_process.aux_shape is not None:

            aux_shape = physical_process.aux_shape
            if ngroups > 0:
                def aux_fn(v, grouped_v, comm):
                    return physical_process.create_aux_data(v, grouped_v, comm=comm)
            else:
                def aux_fn(v, comm):
                    return physical_process.create_aux_datum(v, comm=comm)

            aux_interp_builder = InterpolatedQuantityFactory(aux_fn, aux_shape, ranges, None, ngroups)
            aux_interpolator = aux_interp_builder.build(comm, mpi_workers_per_process, verbosity)
        else:
            aux_interpolator = None

        return cls(target_factory, argument_indices, base_interpolator, aux_interpolator)

    def __init__(self, target_factory, argument_indices, base_interpolator, aux_interpolator=None):
        # NOTE: factory_argument_indices refer to the *interpolated* parameters, i.e. those of the interpolators.
        self.target_factory = target_factory
        self._argument_indices = argument_indices
        self.base_interpolator = base_interpolator
        self.aux_interpolator = aux_interpolator

        dim = self.base_interpolator.qty_shape[0]
        assert(self.base_interpolator.qty_shape == (dim, dim)), \
            "Base interpolator must interpolate a square matrix value!"
        assert(target_factory.dim == dim), "Target factory dim must match interpolated matrix dim!"

        num_interp_params = self.base_interpolator.num_params
        self.num_factory_args = len(self._argument_indices)
        self._parameterized_indices = _np.array(sorted(set(range(num_interp_params)) - set(self._argument_indices)))

        initial_point = []
        for i in self._parameterized_indices:
            min_val, max_val = self.base_interpolator.parameter_ranges[i]
            initial_point.append((min_val + max_val) / 2)
        self._paramvec = _np.array(initial_point, 'd')
        
        super().__init__(dim, evotype="densitymx")
        self.from_vector(self._paramvec)  # initialize object

    def create_object(self, args=None, sslbls=None):
        target_op = self.target_factory.create_op(args, sslbls=None)  # sets vector of target_op
        assert(len(args) == self.num_factory_args), \
            "Wrong number of factory args! (Expected %d and got %d)" % (self.num_factory_args, len(args))

        return InterpolatedDenseOp(target_op, self.base_interpolator, self.aux_interpolator, self.to_vector(),
                                   _np.array(args), self._argument_indices)

    #def write(self, dirname):
    #    dirname = _pathlib.Path(dirname)
    #    with open(str(dirname / "targetop.pkl"), 'wb') as f:
    #        _pickle.dump(self.target_op, f)
    #    _np.save(dirname / "paramvec.np", self._paramvec_with_time)
    #    self.base_interpolator.write(dirname / "base.interp")
    #    if self.aux_interpolator is not None:
    #        self.aux_interptolator.write(dirname / "aux.interp")

    @property
    def num_params(self):
        return len(self._paramvec)

    def to_vector(self):
        return self._paramvec

    def from_vector(self, v, close=False, dirty_value=True):
        self._paramvec[:] = v
        self.target_factory.from_vector(v[0:self.target_factory.num_params])
        self.dirty = dirty_value

    ##----------------------------------------------------------------------------


class InterpolatedDenseOp(_DenseOperator):
    
    #@classmethod
    #def from_dir(cls, dirname):
    #    dirname = _pathlib.Path(dirname)
    #    with open(str(dirname / "targetop.pkl"), 'rb') as f:
    #        target_op = _pickle.load(f)
    #    pt = _np.load(dirname / "paramvec.np")
    #    base_interp = InterpolatedQuantity.from_file(dirname / "base.interp")
    #    aux_interp = InterpolatedQuantity.from_file(dirname / "aux.interp") \
    #        if (dirname / "aux.interp").exists() else None
    #
    #    if base_interp.times is not None:
    #        tm = pt[-1]
    #        pt = pt[0:-1]
    #    else:
    #        tm = None
    #
    #    return cls(target_op, base_interp, aux_interp, pt, tm)

    @classmethod
    def create_by_interpolating_physical_process(cls, target_op, physical_process, parameter_ranges=None,
                                                 parameter_points=None, comm=None,
                                                 mpi_workers_per_process=1, verbosity=0):
        # object_to_interpolate is a PhysicalProcess (or a LinearOperator with adapter?)
        # XXX- anything with from_vector and to_dense methods
        # or a create_process_matrix(v, time=None) method.
        # if times is not None, then this operator's set_time functions nontrivially and object_to_interpolate must be a
        # PhysicalProcess that implements the create_process_matrices(v, times) method

        ngroups = physical_process.num_params_evaluated_as_group
        process_shape = physical_process.process_shape
        if physical_process.can_compute_errorgens_directly:
            if ngroups > 0:
                def fn(v, grouped_v, comm):
                    return physical_process.create_errorgen_matrices(v, grouped_v, comm=comm)
            else:
                def fn(v, comm):
                    return physical_process.create_errorgen_matrix(v, comm=comm)
        else:
            if ngroups > 0:
                def fn(v, grouped_v, comm):
                    grouped_dims = tuple(map(len, grouped_v))
                    ret = _np.empty(grouped_dims + process_shape, 'd')
                    process_mxs = physical_process.create_process_matrices(v, grouped_v, comm=comm)
                    assert(process_mxs.shape == ret.shape)
                    
                    for index_tup, gv in zip(_itertools.product(*[range(d) for d in grouped_dims]),
                                             _itertools.product(*grouped_v)):
                        params = _np.concatenate((v, gv))
                        target_op.from_vector(params[0:target_op.num_params])
                        target_mx = target_op.to_dense()
                        ret[index_tup] = _ot.error_generator(process_mxs[index_tup], target_mx, "pp", "logGTi-quick")
                    return ret
            else:
                def fn(v, comm):
                    target_op.from_vector(v[0:target_op.num_params])
                    target_mx = target_op.to_dense()
                    process_mx = physical_process.create_process_matrix(v, comm=comm)
                    return _ot.error_generator(process_mx, target_mx, "pp", "logGTi-quick")

        base_interp_builder = InterpolatedQuantityFactory(fn, process_shape, parameter_ranges, parameter_points,
                                                          ngroups)
        base_interpolator = base_interp_builder.build(comm, mpi_workers_per_process, verbosity)

        if physical_process.aux_shape is not None:

            aux_shape = physical_process.aux_shape
            if ngroups > 0:
                def aux_fn(v, grouped_v, comm):
                    return physical_process.create_aux_data(v, grouped_v, comm=comm)
            else:
                def aux_fn(v, comm):
                    return physical_process.create_aux_datum(v, comm=comm)

            aux_interp_builder = InterpolatedQuantityFactory(aux_fn, aux_shape, parameter_ranges, parameter_points,
                                                             ngroups)
            aux_interpolator = aux_interp_builder.build(comm, mpi_workers_per_process, verbosity)
        else:
            aux_interpolator = None

        return cls(target_op, base_interpolator, aux_interpolator)

    def __init__(self, target_op, base_interpolator, aux_interpolator=None, initial_point=None,
                 frozen_parameter_values=None, frozen_parameter_indices=None):
        # NOTE: frozen_parameter_indices refer to the *interpolated* parameters, i.e. those of the interpolators.
        self.target_op = target_op
        self.base_interpolator = base_interpolator
        self.aux_interpolator = aux_interpolator

        num_interp_params = self.base_interpolator.num_params
        self._frozen_indices = _np.array(frozen_parameter_indices) \
            if (frozen_parameter_indices is not None) else _np.empty(0, int)
        self._frozen_values = _np.array(frozen_parameter_values) \
            if (frozen_parameter_values is not None) else _np.empty(0, 'd')
        self._parameterized_indices = _np.array(sorted(set(range(num_interp_params)) - set(self._frozen_indices)))
        self.aux_data = None

        dim = self.base_interpolator.qty_shape[0]
        assert(self.base_interpolator.qty_shape == (dim, dim)), \
            "Base interpolator must interpolate a square matrix value!"
        assert(target_op.dim == dim), "Target operation dim must match interpolated matrix dim!"
        nfrozen = len(self._frozen_indices)

        if initial_point is None:
            initial_point = []
            for i in self._parameterized_indices:
                min_val, max_val = self.base_interpolator.parameter_ranges[i]
                initial_point.append((min_val + max_val) / 2)
        else:
            assert(len(initial_point) == len(self._parameterized_indices)), \
                "`initial_point` has the wrong length! (expected %d, got %d)" % (
                    len(self._parameterized_indices), len(initial_point))
        self._paramvec = _np.array(initial_point, 'd')

        super().__init__(_np.identity(dim, 'd'), evotype="densitymx")

        # initialize object
        self.from_vector(self._paramvec)

    #def write(self, dirname):
    #    dirname = _pathlib.Path(dirname)
    #    with open(str(dirname / "targetop.pkl"), 'wb') as f:
    #        _pickle.dump(self.target_op, f)
    #    _np.save(dirname / "paramvec.np", self._paramvec_with_time)
    #    self.base_interpolator.write(dirname / "base.interp")
    #    if self.aux_interpolator is not None:
    #        self.aux_interptolator.write(dirname / "aux.interp")

    @property
    def num_params(self):
        return len(self._paramvec)

    def to_vector(self):
        return self._paramvec

    def from_vector(self, v, close=False, dirty_value=True):
        self._paramvec[:] = v
        self.target_op.from_vector(v[0:self.target_op.num_params])
        fullv = _np.empty(self.base_interpolator.num_params, 'd')
        fullv[self._parameterized_indices] = self._paramvec
        fullv[self._frozen_indices] = self._frozen_values
        errorgen = self.base_interpolator(fullv)
        self.base[:, :] = _ot.operation_from_error_generator(errorgen, self.target_op.to_dense(), 'logGTi')

        if self.aux_interpolator is not None:
            self.aux_data = self.aux_interpolator(fullv)
        self.dirty = dirty_value

    def transform_inplace(self, S):
        # Update self with inverse(S) * self * S (used in gauge optimization)
        raise NotImplementedError("Cannot be transformed!")


class InterpolatedQuantityFactory(object):

    def __init__(self, fn_to_interpolate, qty_shape=(), parameter_ranges=None, parameter_points=None,
                 num_params_to_evaluate_as_group=0):
        """
        Creates an InterpolatedQuantityFactory object.

        These objects are used to create :class:`InterpolatedQuantity` objects, which hold interpolated
        quantities, using multiple processors.

        Parameters
        ----------
        fn_to_interpolate : function
            The function to interpolate, which usually takes considerable resources to evaluate.  If
            `num_params_to_evaluate_as_group == 0` then the expected function definition is:
            `def fn_to_interpolate(point, comm)`.  The `point` argument is an array that specifies values
            of all the parameters, and `comm` is an MPI communicator.  If `num_params_to_evaluate_as_group > 0`
            then the function's  definition must be `def fn_to_interpolate(point, grouped_axial_pts, comm)`.
            The `point` argument then omits values for the final `num_params_to_evaluate_as_group` parameters,
            which are instead specified by arrays of values within the list `grouped_axial_pts`.

        qty_shape : tuple
            The shape of the quantity that is being interpolated.  This is the shape of the array returned
            by `fn_to_interpolate` if `num_params_to_evaluate_as_group == 0`.  In general, the shape of
            the array returned by `fn_to_interpolate` is `qty_shape` *preceded* by the number of values in
            each of the `num_params_to_evaluate_as_group` groups.  An empty tuple means a floating point value.

        parameter_ranges : list, optional
            A list of elements that each specify the values a parameter ranges over.  If the elements are
            tuples, they should be of the form `(min_value, max_value, num_points)` to specify a set of
            equally spaced `num_points` points.  If the elements are `numpy.ndarray` objects, then they
            specify the values directly, e.g. `array([0, 0.1, 0.4, 1.0, 5.0])`.  If `parameter_ranges`
            is specified, `parameter_points` must be left as `None`.
            
        parameter_points : list or numpy.ndarray, optional
            A list or array of parameter-space points, which can be used instead of `parameter_ranges`
            to specify a non-rectangular grid of points.  Each element is an array of real values specifying
            a single point in parameter space (the length of each element must be the same, and sets the 
            number of parameters).  If `parameter_points` is used, then `num_params_to_evaluate_as_group`
            must be 0.

        num_params_to_evaluate_as_group : int, optional
            The number of parameter ranges, counted back from the last one, that should be passed to
            `fn_to_interpolate` as an entire range of values, i.e. via the `grouped_axial_pts` argument.
        """
        self.fn_to_interpolate = fn_to_interpolate
        assert(bool(parameter_ranges is not None) ^ bool(parameter_points is not None)), \
               "Exactly one of `parameter_ranges` or `parameter_points` must be specified!"
        self._parameter_ranges = parameter_ranges
        self._parameter_points = _np.array(parameter_points) if (parameter_points is not None) \
            else None  # ensures all points have same length
        self._num_params_to_evaluate_as_group = num_params_to_evaluate_as_group
        self.data = None
        self.points = None
        self.qty_shape = qty_shape

    def compute_data(self, comm=None, mpi_workers_per_process=1, verbose=False):

        # Define the MPI parameters
        if comm is not None:
            comm.Set_name('comm_world')
            rank = comm.Get_rank()
            size = comm.Get_size()
            mpi_workers_per_process = min(size, mpi_workers_per_process)
        else:
            rank = 0
            size = 1

        # Create communicators for each chunk
        color = rank // mpi_workers_per_process
        root_ranks = [r for r in range(size) if r % mpi_workers_per_process == 0]
        num_mpi_groups = len(root_ranks)
        if comm is not None:
            groupcomm = comm.Split(color, rank)
            groupcomm.Set_name('comm_group_%d' % color)
            #grouprank = groupcomm.Get_rank()
            #groupsize = groupcomm.Get_size()
            rootcomm = comm.Create_group(comm.group.Incl(root_ranks))
            if rank in root_ranks:
                rootcomm.Set_name('comm_root')
        else:
            groupcomm = None

        # build the interpolation grid
        if self._parameter_ranges is not None:
            assert(self._parameter_points is None)
            ngroups = self._num_params_to_evaluate_as_group
            iFirstGroup = len(self._parameter_ranges) - ngroups
            axial_points = []
            for rng in self._parameter_ranges:
                if isinstance(rng, tuple):
                    assert(len(rng) == 3), "Tuple range specifiers must have (min, max, npoints) form!"
                    axial_points.append(_np.linspace(*rng))
                else:
                    assert(isinstance(rng, _np.ndarray)), "Parameter ranges must be specified by tuples or arrays!"
                    axial_points.append(rng)

            points_to_distribute = _np.array(list(_itertools.product(*axial_points[0:iFirstGroup])))
            grouped_axial_pts = axial_points[iFirstGroup:]
            all_points = _np.array(list(_itertools.product(*axial_points)))
        else:
            assert(self._parameter_points is not None and self._num_params_to_evaluate_as_group == 0)
            points_to_distribute = self._parameter_points
            grouped_axial_pts = []
            all_points = points_to_distribute

        expected_fn_output_shape = tuple(map(len, grouped_axial_pts)) + self.qty_shape

        # scatter across mpi workers
        if rank in root_ranks:
            my_points = _split(num_mpi_groups, points_to_distribute)
            if comm is not None:
                my_points = rootcomm.scatter(my_points, root=0)
        else:
            my_points = []

        if comm is not None:
            my_points = groupcomm.bcast(my_points, root=0)
        else:
            my_points = my_points[0]

        if (rank in root_ranks) and (comm is not None):
            print("Group %d processing %d points on %d processors." % (color, len(my_points), mpi_workers_per_process))

        # compute the process matrices at each data point
        flat_data = _np.empty(len(my_points) * int(_np.product(expected_fn_output_shape)), dtype='d')
        data = flat_data.view(); data.shape = (len(my_points),) + expected_fn_output_shape
        for ind, point in enumerate(my_points):
            if verbose: print("Evaluating index %d , data = %s" % (ind, str(point)))
            data[ind] = self.fn_to_interpolate(point, grouped_axial_pts, comm=groupcomm) if grouped_axial_pts \
                else self.fn_to_interpolate(point, comm=groupcomm)

        # Gather data from groups
        if rank in root_ranks:
            if comm is not None:
                sizes = rootcomm.gather(flat_data.size, root=0)
                recvbuf = (_np.empty(sum(sizes), flat_data.dtype), sizes) if (rootcomm.Get_rank() == 0) else None
                rootcomm.Gatherv(sendbuf=flat_data, recvbuf=recvbuf, root=0)
                if rootcomm.Get_rank() == 0:
                    assert(rank == 0), "The rank=0 root-comm processor should also be rank=0 globally"
                    flat_data = recvbuf[0]
        else:
            flat_data = None

        #if rank == 0 and grouped_by_time:  # convert time dimension => points
        #    gathered_data = _np.swapaxes(gathered_data, 0, 1)  # Make time index come first
        #    gathered_data = _flatten(gathered_data)  # Flatten (iTime, iPoint, ...) => (iPoint, ...)
        #    points_to_distribute = _np.array(list(_itertools.product(self.times, *axial_points)))  # Add time to points_to_distribute

        if comm is not None:
            flat_data = comm.bcast(flat_data, root=0)  # NEEDED?

        self.points = all_points
        self.data = flat_data.view()
        self.data.shape = (len(all_points),) + self.qty_shape  # indices are (iPoint, <data_indices>)

    def build(self, comm=None, mpi_workers_per_process=1, verbose=False):

        if comm is not None:
            size = comm.Get_size()
            rank = comm.Get_rank()
        else:
            size = 1
            rank = 0

        if self.data is None or self.points is None:
            self.compute_data(comm, mpi_workers_per_process, verbose)

        self.interpolator = _np.empty(self.qty_shape, dtype=object)
        all_index_tuples = _split(size, list(_itertools.product(*[range(d) for d in self.qty_shape])),
                                  cast_to_array=False)
        my_index_tuples = all_index_tuples[rank]
        my_interpolators = _np.empty(len(my_index_tuples), dtype=object)

        # Build the interpolators
        for int_ind, index_tuple in enumerate(my_index_tuples):
            values = [data_at_point[index_tuple] for data_at_point in self.data]
            my_interpolators[int_ind] = _linND(self.points, values, rescale=True)

        if comm is not None:
            all_interpolators = comm.gather(my_interpolators, root=0)
        else:
            all_interpolators = [my_interpolators]

        if rank == 0:
            all_interpolators = _flatten(all_interpolators)
            interpolators = _np.empty(self.qty_shape, dtype='object')
            for interp, index_tuple in zip(all_interpolators, _flatten(all_index_tuples)):
                interpolators[index_tuple] = interp
            if comm is not None:
                comm.bcast(interpolators, root=0)
        else:
            interpolators = comm.bcast(None, root=0)

        if self._parameter_ranges is not None:
            parameter_range_bounds = [(rng[0], rng[1]) if isinstance(rng, tuple) else (min(rng), max(rng))
                                      for rng in self._parameter_ranges]
        else:
            parameter_range_bounds = [(min(self._parameter_points[:, i]), max(self._parameter_points[:, i]))
                                      for i in range(self._parameter_points.shape[1])]

        return InterpolatedQuantity(interpolators, parameter_range_bounds)


class InterpolatedQuantity(object):

    @classmethod
    def from_file(cls, filename):
        raise NotImplementedError()

    def __init__(self, interpolators, parameter_ranges, verbose=False):
        self.interpolators = interpolators
        self.parameter_ranges = tuple(parameter_ranges)

    @property
    def qty_shape(self):
        return self.interpolators.shape

    @property
    def num_params(self):
        return len(self.parameter_ranges)

    def __call__(self, v):
        assert(len(v) == self.num_params)
        if not all([(a <= b <= c) for b, (a, c) in zip(v, self.parameter_ranges)]):
            raise ValueError("Parameter out of range.")

        value = _np.zeros(self.qty_shape, dtype='d')
        for i, interpolator in enumerate(self.interpolators.flat):
            value.flat[i] = interpolator(*v)
        return value

    def write(self, filename):
        raise NotImplementedError()


#OLD REMOVE
#_comm = None
#
## This is modeled after the (now depricated?) pygsti.obj.DenseOperator class
#class OLDPhysicalProcess(object):
#    """
#    Returns a new PhysicalProcess object that can pre-compute itself over a range of parameters
#    and then use interpolation to reduce forward simulation overhead.
#
#    Parameters
#    ----------
#
#
#    Returns
#    -------
#    PhysicalProcess
#       The gate object. 
#    """
#
#    def __init__(self, fname=None, basis=None, comm=_comm, mpi_workers_per_process=1, verbose=False, continuous_gates=False):
#        """
#        Create a new class instance and initialize variables.
#        """
#        if fname is None:
#            self.__version__ = __version__
#            self.base = _np.empty(0)
#            self.data = None
#            self.name = "UNNAMED"
#            self.target = None
#            self.parameter_range = None
#            self.grouped_by_time = False
#            self.interpolation_order = None
#            self.interpolated = False
#            self._process_function = lambda v: None
#            self.verbose = verbose
#            self.kwargs = {}
#            self.continuous_gates = continuous_gates
#
#            self.has_metadata = False
#            self.has_auxdata = False
#
#            if basis is None:
#                raise ValueError("'basis' keyword arguement must be specified.")
#            else:
#                self.basis = basis
#
#            # class variables beginning with _x_ are not saved to disk
#            # Define the MPI parameters
#
#            self._x_comm = comm
#            self._x_comm.Set_name('comm_world')
#            self._x_rank = comm.Get_rank()
#            self._x_size = comm.Get_size()
#            self._x_mpi_workers_per_process = min(self._x_size, mpi_workers_per_process)
#
#            # Create communicators for each chunk
#            self._x_mpi_enabled = False
#            self._x_color = self._x_rank // mpi_workers_per_process
#            self._x_roots = [x for x in range(self._x_size) if x % mpi_workers_per_process == 0]
#            self._x_n_mpi_groups = len(self._x_roots)
#            self._x_groupcomm = self._x_comm.Split(self._x_color, self._x_rank)
#            self._x_groupcomm.Set_name(f'comm_group_{self._x_color}')
#            self._x_grouprank = self._x_groupcomm.Get_rank()
#            self._x_groupsize = self._x_groupcomm.Get_size()
#            self._x_rootcomm = self._x_comm.Create_group(comm.group.Incl(self._x_roots))
#            if self._x_rank in self._x_roots:
#                self._x_rootcomm.Set_name('comm_root')
#        else:
#            self.load(fname)
#
#    def num_params(self):
#        # ORIGINAL
#        # TODO: This should be populated automatically when the process_function is defined
#        return self.n_params
#
#    def to_vector(self):
#        # ORIGINAL
#        return _np.array(self.v)
#
#    def from_vector(self, v, basis=None, allow_physical=False, return_generator=False):
#        self.v = v
#
#        if not self.interpolated:
#            if allow_physical:
#                print("Computing physical process directly without using interpolation.")
#                self.base = self.from_vector_physical(v)
#                return self.base
#            else:
#                raise NotImplementedError("No interpolator is available and physical processes are disallowed.")
#
#        if len(self.parameter_range) == len(v) - 1:
#            self.parameter_range = [[min(self.times), max(self.times)]] + self.parameter_range
#
#        if not all([(a >= b) and (a <= c) for a, (b, c) in zip(v, self.parameter_range)]):
#            raise ValueError("Parameter out of range.")
#
#        if self.interpolated:
#            error_generator = _np.zeros([self.dimension, self.dimension], dtype='float')
#            for indi in range(self.dimension):
#                for indj in range(self.dimension):
#                    error_generator[indi, indj] = _np.float(_np.real(self.interpolator[indi, indj](*v)))
#
#        self.base = self._gate_from_error_generator(error_generator, v=v)
#
#        if return_generator:
#            return_object = error_generator
#        else:
#            return_object = self.base
#
#        # You can request a a different basis
#        if basis is not None:
#            if basis != self.basis:
#                return_object = _change_basis(return_object, self.basis, basis)
#
#        return return_object
#
#    def transform(self, S):
#        # ORIGINAL
#        # Update self with inverse(S) * self * S (used in gauge optimization)
#        raise NotImplementedError(f"{self.name} cannot be transformed!")
#
#    def __call__(self, v, basis=None):
#        return self.from_vector(v, basis=basis)
#
#    def from_vector_physical(self, v):
#        """
#        Empty Docstring
#        :rtype: numpy.ndarray
#        """
#        # Don't edit this
#        # It is set up like this (rather than being defined directly) so that the parameter vector can be saved
#        self.v = v
#        if self.verbose:
#            print(f"from_vector_physical using {self.kwargs['comm'].Get_name()}")
#        return self._process_function(v, **self.kwargs)
#
#    def set_name(self, name):
#        # Give the gate a name
#        self.name = name
#        return self
#
#    def set_target(self, target):
#        # Define the target gate. This is used to build the error generators. 
#        if self.continuous_gates:
#            self.target = target
#            self.inv_target = lambda v: _np.linalg.inv(target(v))
#
#        else:
#            self.target = lambda v: _np.array(target)
#            self.inv_target = lambda v: _np.linalg.inv(target)
#
#        return self
#
#    def set_parameter_range(self, parameter_range):
#        if self.interpolation_order is not None:
#            assert len(self.interpolation_order) == len(
#                parameter_range), "'interpolation_order' and 'parameter_range' must be the same length"
#        self.parameter_range = parameter_range
#
#        return self
#
#    def set_interpolation_order(self, interpolation_order):
#        # Gx.set_interpolation_order([5,6,3,3,3,3])
#
#        if self.parameter_range is not None:
#            assert len(interpolation_order) == len(
#                self.parameter_range), "'interpolation_order' and 'parameter_range' must be the same length"
#        self.interpolation_order = interpolation_order
#
#        return self
#
#    def set_metadata(self, metadata):
#        self.has_metadata = True
#        self.metadata = metadata
#
#    def get_auxdata(self, v):
#        if not self.aux_interpolated:
#            raise NotImplementedError("No auxiliary data interpolator is available.")
#
#        if len(self.parameter_range) == len(v) - 1:
#            self.parameter_range = [[min(self.times), max(self.times)]] + self.parameter_range
#
#        if not all([(a >= b) and (a <= c) for a, (b, c) in zip(v, self.parameter_range)]):
#            raise ValueError("Parameter out of range.")
#
#        if self.aux_interpolated:
#            aux_data = _np.zeros([self.auxdim], dtype='float')
#            for indi in range(self.auxdim):
#                aux_data[indi] = _np.float(self.aux_interpolator[indi](*v))
#
#        return aux_data
#
#    def set_process_function(self, fn, mpi_enabled=False, has_auxdata=False):
#        self._process_function = fn
#        self.has_auxdata = has_auxdata
#        if mpi_enabled:
#            # If the function is MPI enabled, it should gather its results to the communicator's 0 node
#            # when it's finished.
#            self.kwargs['comm'] = self._x_groupcomm
#            self._x_mpi_enabled = True
#
#    def set_times(self, times):
#        """ Tell the interpolator that it is easy to extract the process matrix for multiple time points if
#        all other parameters are the same.
#
#        If this is set, the process function MUST have two modes of operation:
#            process_fn([t,p1,p2,...]) for t a single time
#            returning a single process matrix
#        and
#            process_fn([p1,p2,...], times=times) for times a list of times
#            returning a list of process matrices, one for each time
#
#        The time parameter is then EXCLUDED from the parameter ranges, and it is set here explicitly.
#        """
#        self.grouped_by_time = True
#        self.times = times
#        self.kwargs['times'] = times
#
#    def _error_generator_from_gate(self, gate, v=None):
#        # g = exp(e).t
#        # e = log(g t^-1)
#        if self.verbose:
#            print(gate)
#        if isinstance(gate[0][1], complex):
#            generator = _np.array(_logm(_np.dot(gate, self.inv_target(v))), dtype=complex)
#        else:
#            generator = _np.array(_np.real(_logm(_np.dot(gate, self.inv_target(v)))), dtype=float)
#        return generator
#
#    def _gate_from_error_generator(self, error_generator, v=None):
#        # g = exp(e).t
#        # e = log(g t^-1)
#        dtype = type(error_generator[0,1])
#        self.error_generator = error_generator
#        return _np.array(_np.dot(_expm(error_generator), self.target(v)), dtype=dtype)
#
#    def interpolate(self):
#        # TODO: Interpolate the AUX function as well ** HAVEN'T STARTED THIS HERE
#        # TODO: Decide if the aux function should be returned along side the process matrices?
#        #       This is easier, so I'll probably stick to it unless people complain.
#
#        # build the interpolation grid
#        axial_points = _np.array(
#            [_np.linspace(a, b, c) for (a, b), c in zip(self.parameter_range, self.interpolation_order)])
#        all_points = _np.array(list(_itertools.product(*axial_points)))
#
#        # scatter across mpi workers
#        if self._x_rank in self._x_roots:
#            my_points = _split(self._x_n_mpi_groups, all_points)
#            my_points = self._x_rootcomm.scatter(my_points, root=0)
#        else:
#            my_points = []
#        my_points = self._x_groupcomm.bcast(my_points, root=0)
#        if self._x_rank in self._x_roots:
#            print(f"Group {self._x_color} processing {len(my_points)} points on " +
#                  f"{self._x_mpi_workers_per_process} processors.")
#            
#        # compute the process matrices at each data point
#        data = _np.empty(len(my_points), dtype=_np.ndarray)
#        if self.has_auxdata:
#            auxdata = _np.empty(len(my_points), dtype=_np.ndarray)
#
#        # print(_rank, len(data))
#        for ind, point in enumerate(my_points):
#            if self.verbose:
#                print("Evaluating index {ind}, data = {point}")
#            if self.grouped_by_time:
#                if self.has_auxdata:
#                    data_by_time, auxdata_by_time = self.from_vector_physical(point)
#                else:
#                    data_by_time = self.from_vector_physical(point)
#
#                # "data_by_time or []" is because "data_by_time" might be None and you can't iterate over None
#                try:
#                    generators_by_times = [self._error_generator_from_gate(gate, v=point) for gate in data_by_time or []]
#                except:
#                    generators_by_times = []
#                    
#                data[ind] = generators_by_times
#                if self.has_auxdata:
#                    auxdata[ind] = auxdata_by_time
#            else:
#                if self.verbose:
#                    print('Computing error generator')
#                if self.has_auxdata:
#                    data[ind], auxdata[ind] = self._error_generator_from_gate(self.from_vector_physical(point), v=point)
#                else:
#                    data[ind] = self._error_generator_from_gate(self.from_vector_physical(point), v=point)
#
#                if self.verbose:
#                    print('Computed error generator')
#
#        # Gather data from groups
#        if self._x_rank in self._x_roots:
#            gathered_data = _np.array(_flatten(self._x_rootcomm.gather(data, root=0)))
#            if self.has_auxdata:
#                gathered_auxdata = _np.array(_flatten(self._x_rootcomm.gather(auxdata, root=0)))
#
#            if self.grouped_by_time:
#                self.dimension = data[0][0].shape[0]
#                if self.has_auxdata:
#                    print(f"Auxdata shape: {auxdata.shape}, {auxdata}")
#                    self.auxdim = len(auxdata[0])
#
#            else:
#                self.dimension = data[0].shape[0]
#                if self.has_auxdata:
#                    self.auxdim = len(auxdata[0])
#        else:
#            gathered_data = None
#            gathered_auxdata = None
#            self.dimension = None
#            self.auxdim = None
#
#        # construct the interpolators
#        if self._x_rank == 0:
#            if self.grouped_by_time:
#                gathered_data = _np.transpose(gathered_data, [1, 0, 2, 3])  # Make time index come first
#                gathered_data = _flatten(gathered_data)  # Flatten along time axis
#                all_points = _np.array(list(_itertools.product(self.times, *axial_points)))  # Add time to all_points
#
#                if self.has_auxdata:
#                    gathered_auxdata = _np.transpose(gathered_auxdata, [2,0,1])
#                    gathered_auxdata = _flatten(gathered_auxdata)
#                    for x,y in zip(gathered_auxdata, all_points):
#                        print(x, y)
#
#            self.interpolator = _np.empty([self.dimension, self.dimension], dtype=object)                            
#            self.data = gathered_data
#            if self.has_auxdata:
#                self.aux_interpolator = _np.empty(self.auxdim, dtype=object)
#                self.auxdata = gathered_auxdata
#            self.points = all_points
#
#        all_points = self._x_comm.bcast(all_points, root=0)
#        gathered_data = self._x_comm.bcast(gathered_data, root=0)
#        self.dimension = self._x_comm.bcast(self.dimension, root=0)
#
#        if self.has_auxdata:
#            gathered_auxdata = self._x_comm.bcast(gathered_auxdata, root=0)
#            self.auxdim = self._x_comm.bcast(self.auxdim, root=0)
#
#
#        all_pairs = _split(self._x_size,
#                              [[indi, indj] for indi in range(self.dimension) for indj in range(self.dimension)])
#        my_pairs = all_pairs[self._x_rank]
#        my_interpolators = _np.empty(len(my_pairs), dtype='object')
#
#        if self.has_auxdata:
#            all_aux_inds = _split(self._x_size, [indx for indx in range(self.auxdim)])
#            my_aux_inds = all_aux_inds[self._x_rank]
#            my_aux_interpolators = _np.empty(len(my_aux_inds), dtype='object')
#
#        # Build the interpolators
#        for int_ind, (indi, indj) in enumerate(my_pairs):
#                values = [datum[indi, indj] for datum in gathered_data]
#                my_interpolators[int_ind] = _linND(all_points, values, rescale=True)
#        
#        if self.has_auxdata:
#            for int_ind, aux_ind in enumerate(my_aux_inds):
#                aux_values = [datum[aux_ind] for datum in gathered_auxdata]
#                my_aux_interpolators[int_ind] = _linND(all_points, aux_values, rescale=True)
#
#        all_interpolators = self._x_comm.gather(my_interpolators, root=0)
#        if self._x_rank == 0:
#            all_interpolators = _flatten(all_interpolators)
#            self.interpolator = _np.empty([self.dimension, self.dimension], dtype='object')
#            for interp, (indi, indj) in zip(all_interpolators, _flatten(all_pairs)):
#                self.interpolator[indi, indj] = interp
#        
#        if self.has_auxdata:
#            all_aux_interpolators = self._x_comm.gather(my_aux_interpolators, root=0)
#            print(all_aux_inds)
#            for x in all_aux_interpolators:
#                print(x)
#            if self._x_rank == 0:
#                all_aux_interpolators = _flatten(all_aux_interpolators)
#                self.aux_interpolator = _np.empty(self.auxdim, dtype='object')
#                for interp, (ind) in zip(all_aux_interpolators, _flatten(all_aux_inds)):
#                    self.aux_interpolator[ind] = interp
#            self.aux_interpolated = True
#
#        self.interpolated = True
#
#    def save(self, fname):
#        """Save the data for this object to a file"""
#        if self._x_rank == 0:
#            # save_keys = ['interpolator', 'interpolation_order',
#            #              'parameter_range', 'times',
#            #              'target', 'inv_target']
#            # save_dict = {}
#            # for key in save_keys:
#            #     save_dict[key] = _copy.copy(self.__dict__[key])
#
#
#            # # make a copy of the object dictionary and delete the MPI components
#            safe_dict = self.__dict__.copy()
#            del_keys = ['_process_function', '_aux_function', 'kwargs']
#            for key in safe_dict.keys():
#                if '_x_' in key:
#                    del_keys += [key]
#            for key in del_keys:
#                try:
#                    del safe_dict[key]
#                except KeyError:
#                    pass
#
#            # dump to a file
#            with open(fname, 'wb') as f:
#                dill.dump(safe_dict, f)
#
#    def load(self, fname):
#        """Read the data in from a file"""
#        # self.__dict__.clear()
#        with open(fname, 'rb') as f:
#            loaded_dict = dill.load(f)
#            if "__version__" not in loaded_dict.keys():
#                loaded_dict["__version__"] = "0.1.0"
#            self.__dict__.update(loaded_dict)
#            
#            # If the version is earlier than 0.8.0, then fix the target function
#            print(f"Loading {fname} made with version {self.__version__}")
#            
#            if tuple(map(int, self.__version__.split('.'))) < (0, 8, 0):  # compare w/ version "0.8.0"
#                if not callable(self.target):
#                    target = self.target.copy()
#                    self.target = lambda v: target
#                    self.inv_target = lambda v: _np.linalg.inv(target)
#            
#            test_params = [_np.mean(x) for x in self.parameter_range]    


#OLD REMOVE
#if __name__ == '__main__':
#
#    # from interface import *
#    from . import do_process_tomography, vec, unvec
#    from ...tools.basistools import change_basis
#
#    class ProcessFunction(object):
#        def __init__(self):
#            import numpy as _np
#            import scipy as _sp
#
#            self.Hx = _np.array([[0, 0, 0, 0],
#                                [0, 0, 0, 0],
#                                [0, 0, 0, -1],
#                                [0, 0, 1, 0]], dtype='float')
#            self.Hy = _np.array([[0, 0, 0, 0],
#                                [0, 0, 0, 1],
#                                [0, 0, 0, 0],
#                                [0, -1, 0, 0]], dtype='float')
#            self.Hz = _np.array([[0, 0, 0, 0],
#                                [0, 0, -1, 0],
#                                [0, 1, 0, 0],
#                                [0, 0, 0, 0]], dtype='float')
#
#            self.dephasing_generator = _np.diag([0, -1, -1, 0])
#            self.decoherence_generator = _np.diag([0, -1, -1, -1])
#
#        def advance(self, state, v = None, times = None):
#            state = _np.array(state, dtype='complex')
#            if times is None:
#                t, omega, phase, detuning, dephasing, decoherence = v
#                times = [t]
#            else:
#                omega, phase, detuning, dephasing, decoherence = v
#
#            H = (omega * _np.cos(phase) * self.Hx + omega * _np.sin(phase) * self.Hy + detuning * self.Hz)
#            L = dephasing * self.dephasing_generator + decoherence * self.decoherence_generator
#
#            processes = [_change_basis(_expm((H + L) * t),
#                                                              'pp', 'col') for t in times]
#            states = [unvec(_np.dot(process, vec(_np.outer(state, state.conj())))) for process in processes]
#
#            return states
#
#        def __call__(self, v, times=None, comm=_comm):
#            print(f'Calling process tomography as {comm.Get_rank()} of {comm.Get_size()} on {comm.Get_name()}.')
#            processes = do_process_tomography(self.advance, opt_args={'v':v, 'times':times},
#                                              n_qubits = 1, basis='pp', time_dependent=True, comm=comm)
#
#            return processes
#
#    gy = PhysicalProcess(mpi_workers_per_process=1, basis='pp')
#    gy.set_process_function(ProcessFunction(), mpi_enabled=True)
#    target = _change_basis(_np.array([[1,0,0,0],
#                                     [0,1,0,0],
#                                     [0,0,0,-1],
#                                     [0,0,1,0]], dtype='complex'), 'pp', 'pp')
#    gy.set_target(target)
#
#    # # # Evaluate one time point per run
#    # # gy.set_parameter_range([[_np.pi/2-.5, _np.pi/2+.5],[0.9,1.1],[-.1,.1],[-.1,.1],[0,0.1],[0,0.1]])
#    # # gy.set_interpolation_order([3,3,3,3,3,3])
#    #
#    # Evaluate many time points per run
#    # Notice that the number of parameters listed below is 1 fewer than in the previous example
#    gy.set_parameter_range([[0.9, 1.1], [-.1, .1], [-.1, .1], [0, 0.1], [0, 0.1]])
#    gy.set_interpolation_order([3,3,3,3,3])
#    gy.set_times(_np.linspace(_np.pi / 2, _np.pi / 2 + .5, 10))
#
#    gy.interpolate()
#    gy.save('intertest.dat')
#
#    # #
#    # # def diamond_norm(a, b):
#    # #     A = pygsti.tools.jamiolkowski_iso(a)
#    # #     B = pygsti.tools.jamiolkowski_iso(b)
#    # #     return pygsti.tools.diamonddist(a, b)
#    # #
#    # #
#
#    if _rank == 0:
#        from matplotlib import pyplot as plt
#        from process_tomography import change_basis
#        cb = lambda x: _change_basis(x, 'qsim', 'pp')
#        gx = PhysicalProcess('intertest.dat')
#        print(_np.round_(gx.from_vector([_np.pi / 2 + .3, 1, 0, 0, 0, 0]),3))
#        # print(_np.round_(gx.from_vector_physical([_np.pi / 2 + .3, 1, 0, 0, 0, 0]),3))
#        nom = _np.array([_np.pi / 2, 1, 0, 0, 0, 0])
#        labels = ["Timing Error", "Amplitude Error", "Phase Error", "Frequency Error", "Additional Dephasing", "Additional Decoherence"]
#        for ind in range(len(nom)):
#            tnom = nom.copy()
#            tnom[ind] += .1
#            ergen = _np.round_(cb(gx._error_generator_from_gate(gx(tnom))), 3) / .1
#            plt.matshow(ergen.real)
#            plt.title(labels[ind])
#            print(ind)
#            print(ergen)
#            plt.savefig(f'./figures/param_{ind}.pdf')
