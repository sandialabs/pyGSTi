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
from ...objects.verbosityprinter import VerbosityPrinter as _VerbosityPrinter


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


class _PhysicalBase(object):
    def __init__(self, num_params, item_shape, aux_shape=None, num_params_evaluated_as_group=0):
        self.num_params = num_params
        self.item_shape = item_shape
        self.aux_shape = aux_shape  # None means no aux data
        self.num_params_evaluated_as_group = num_params_evaluated_as_group

    def create_aux_info(self, v, comm=None):
        raise NotImplementedError("Derived classes must implement `create_aux_info`!")

    def create_aux_infos(self, v, grouped_v, comm=None):
        raise NotImplementedError("Derived class must implement `create_aux_infos`!")


class PhysicalProcess(_PhysicalBase):

    def __init__(self, num_params, process_shape, aux_shape=None, num_params_evaluated_as_group=0):
        super().__init__(num_params, process_shape, aux_shape, num_params_evaluated_as_group)

    def create_process_matrix(self, v, comm=None):
        raise NotImplementedError("Derived classes must implement create_process_matrix!")

    def create_process_matrices(self, v, grouped_v, comm=None):
        raise NotImplementedError("Derived class must implement `create_process_matrices`!")


class PhysicalErrorGenerator(_PhysicalBase):

    def __init__(self, num_params, errorgen_shape, aux_shape=None, num_params_evaluated_as_group=0):
        super().__init__(num_params, errorgen_shape, aux_shape, num_params_evaluated_as_group)

    def create_errorgen_matrix(self, v, comm=None):
        raise NotImplementedError("Derived classes must implement create_errorgen_matrix!")

    def create_errorgen_matrices(self, v, grouped_v, comm=None):
        raise NotImplementedError("Derived class must implement `create_errorgen_matrices`!")


class OpPhysicalProcess(PhysicalProcess):

    def __init__(self, op):
        self.op = op
        super().__init__(op.num_params, (op.dim, op.dim), None, 0)

    def create_process_matrix(self, v, comm=None):
        self.op.from_vector(v)
        return self.op.to_dense()


class InterpolatedOpFactory(_OpFactory):

    @classmethod
    def create_by_interpolating_physical_process(cls, target_factory, physical_process, argument_ranges,
                                                 parameter_ranges, argument_indices=None, comm=None,
                                                 mpi_workers_per_process=1, interpolator_and_args=None, verbosity=0):
        #printer = _VerbosityPrinter.create_printer(verbosity)
        nargs = len(argument_ranges)
        if argument_indices is None:
            argument_indices = _np.arange(nargs, dtype=int)
        else:
            argument_indices = _np.array(argument_indices, dtype=int)
        param_indices = _np.array(sorted(set(range(physical_process.num_params)) - set(argument_indices)), dtype=int)

        ngroups = physical_process.num_params_evaluated_as_group
        process_shape = physical_process.item_shape
        if isinstance(physical_process, PhysicalErrorGenerator):
            if ngroups > 0:
                def fn(v, grouped_v, comm):
                    return physical_process.create_errorgen_matrices(v, grouped_v, comm=comm)
            else:
                def fn(v, comm):
                    return physical_process.create_errorgen_matrix(v, comm=comm)
        else:
            if ngroups > 0:
                def fn(v, grouped_v, comm):
                    process_mxs = physical_process.create_process_matrices(v, grouped_v, comm=comm)
                    if comm is not None and comm.Get_rank() != 0:
                        return None  # a "slave" processor that doesn't need to report a value (process_mxs can be None)

                    grouped_dims = tuple(map(len, grouped_v))
                    ret = _np.empty(grouped_dims + process_shape, 'd')
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
                    process_mx = physical_process.create_process_matrix(v, comm=comm)
                    if comm is not None and comm.Get_rank() != 0:
                        return None  # a "slave" processor that doesn't need to report a value (process_mx can be None)

                    args = v[argument_indices]
                    params = v[param_indices]
                    target_op = target_factory.create_op(args, sslbls=None)
                    target_op.from_vector(params[0:target_op.num_params])
                    target_mx = target_op.to_dense()
                    return _ot.error_generator(process_mx, target_mx, "pp", "logGTi-quick")

        ranges = [None] * (len(argument_ranges) + len(parameter_ranges))
        for i, arg_range in zip(argument_indices, argument_ranges): ranges[i] = arg_range
        for i, param_range in zip(param_indices, parameter_ranges): ranges[i] = param_range

        base_interp_builder = InterpolatedQuantityFactory(fn, process_shape, ranges, None, ngroups,
                                                          interpolator_and_args)
        base_interpolator = base_interp_builder.build(comm, mpi_workers_per_process, verbosity)

        if physical_process.aux_shape is not None:

            aux_shape = physical_process.aux_shape
            if ngroups > 0:
                def aux_fn(v, grouped_v, comm):
                    return physical_process.create_aux_infos(v, grouped_v, comm=comm)
            else:
                def aux_fn(v, comm):
                    return physical_process.create_aux_info(v, comm=comm)

            aux_interp_builder = InterpolatedQuantityFactory(aux_fn, aux_shape, ranges, None, ngroups,
                                                             interpolator_and_args)
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
                                                 mpi_workers_per_process=1, interpolator_and_args=None, verbosity=0):
        # object_to_interpolate is a PhysicalProcess (or a LinearOperator with adapter?)
        # XXX- anything with from_vector and to_dense methods
        # or a create_process_matrix(v, time=None) method.
        # if times is not None, then this operator's set_time functions nontrivially and object_to_interpolate must be a
        # PhysicalProcess that implements the create_process_matrices(v, times) method

        #printer = _VerbosityPrinter.create_printer(verbosity)
        ngroups = physical_process.num_params_evaluated_as_group
        process_shape = physical_process.item_shape
        if isinstance(physical_process, PhysicalErrorGenerator):
            if ngroups > 0:
                def fn(v, grouped_v, comm):
                    return physical_process.create_errorgen_matrices(v, grouped_v, comm=comm)
            else:
                def fn(v, comm):
                    return physical_process.create_errorgen_matrix(v, comm=comm)
        else:
            if ngroups > 0:
                def fn(v, grouped_v, comm):
                    process_mxs = physical_process.create_process_matrices(v, grouped_v, comm=comm)
                    if comm is not None and comm.Get_rank() != 0:
                        return None  # a "slave" processor that doesn't need to report a value (process_mxs can be None)

                    grouped_dims = tuple(map(len, grouped_v))
                    ret = _np.empty(grouped_dims + process_shape, 'd')
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
                    process_mx = physical_process.create_process_matrix(v, comm=comm)
                    if comm is not None and comm.Get_rank() != 0:
                        return None  # a "slave" processor that doesn't need to report a value (process_mx can be None)

                    target_op.from_vector(v[0:target_op.num_params])
                    target_mx = target_op.to_dense()
                    return _ot.error_generator(process_mx, target_mx, "pp", "logGTi-quick")

        base_interp_builder = InterpolatedQuantityFactory(fn, process_shape, parameter_ranges, parameter_points,
                                                          ngroups, interpolator_and_args)
        base_interpolator = base_interp_builder.build(comm, mpi_workers_per_process, verbosity)

        if physical_process.aux_shape is not None:

            aux_shape = physical_process.aux_shape
            if ngroups > 0:
                def aux_fn(v, grouped_v, comm):
                    return physical_process.create_aux_infos(v, grouped_v, comm=comm)
            else:
                def aux_fn(v, comm):
                    return physical_process.create_aux_info(v, comm=comm)

            aux_interp_builder = InterpolatedQuantityFactory(aux_fn, aux_shape, parameter_ranges, parameter_points,
                                                             ngroups, interpolator_and_args)
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
        self.aux_info = None

        dim = self.base_interpolator.qty_shape[0]
        assert(self.base_interpolator.qty_shape == (dim, dim)), \
            "Base interpolator must interpolate a square matrix value!"
        assert(target_op.dim == dim), "Target operation dim must match interpolated matrix dim!"

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
            self.aux_info = self.aux_interpolator(fullv)
        self.dirty = dirty_value

    def transform_inplace(self, S):
        # Update self with inverse(S) * self * S (used in gauge optimization)
        raise NotImplementedError("Cannot be transformed!")


class InterpolatedQuantityFactory(object):

    def __init__(self, fn_to_interpolate, qty_shape=(), parameter_ranges=None, parameter_points=None,
                 num_params_to_evaluate_as_group=0, interpolator_and_args=None):
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

        interpolator_and_args : tuple, optional
            Optionally a 2-tuple of an interpolation class and argument dictionary.  If None, the
            default of `(scipy.interpolate.LinearNDInterpolator, {'rescale': True})` is used.
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
        self.interpolator_and_args = interpolator_and_args

    def compute_data(self, comm=None, mpi_workers_per_process=1, verbosity=0):

        printer = _VerbosityPrinter.create_printer(verbosity, comm)

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
            iFirstGrouped = len(self._parameter_ranges) - ngroups
            axial_points = []
            for rng in self._parameter_ranges:
                if isinstance(rng, tuple):
                    assert(len(rng) == 3), "Tuple range specifiers must have (min, max, npoints) form!"
                    axial_points.append(_np.linspace(*rng))
                else:
                    assert(isinstance(rng, _np.ndarray)), "Parameter ranges must be specified by tuples or arrays!"
                    axial_points.append(rng)

            points_to_distribute = _np.array(list(_itertools.product(*axial_points[0:iFirstGrouped])))
            grouped_axial_pts = axial_points[iFirstGrouped:]
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

        if rank in root_ranks:
            #Only root ranks store data (fn_to_interpolate only needs to return results on root proc)
            flat_data = _np.empty(len(my_points) * int(_np.product(expected_fn_output_shape)), dtype='d')
            data = flat_data.view(); data.shape = (len(my_points),) + expected_fn_output_shape
            if (comm is not None):
                printer.log("Group %d processing %d points on %d processors." % (color, len(my_points),
                                                                                 mpi_workers_per_process))
        else:
            flat_data = data = None  # to keep us from accidentally misusing these below

        # compute the process matrices at each data point
        for ind, point in enumerate(my_points):
            printer.log("Evaluating index %d , data = %s" % (ind, str(point)))
            val = self.fn_to_interpolate(point, grouped_axial_pts, comm=groupcomm) if grouped_axial_pts \
                else self.fn_to_interpolate(point, comm=groupcomm)
            if rank in root_ranks:  # only the root proc of each groupcomm needs to produce a result
                data[ind] = val     # (other procs can just return None, so val = None)

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

        if comm is not None:
            flat_data = comm.bcast(flat_data, root=0)
            # Needed because otherwise only some procs contain data and *all* procs will be building
            # interpolators from this data in build(...) below.

        self.points = all_points
        self.data = flat_data.view()
        self.data.shape = (len(all_points),) + self.qty_shape  # indices are (iPoint, <data_indices>)

    def build(self, comm=None, mpi_workers_per_process=1, verbosity=0):

        printer = _VerbosityPrinter.create_printer(verbosity, comm)
        if comm is not None:
            size = comm.Get_size()
            rank = comm.Get_rank()
        else:
            size = 1
            rank = 0

        if self.data is None or self.points is None:
            self.compute_data(comm, mpi_workers_per_process, printer)

        self.interpolator = _np.empty(self.qty_shape, dtype=object)
        all_index_tuples = _split(size, list(_itertools.product(*[range(d) for d in self.qty_shape])),
                                  cast_to_array=False)
        my_index_tuples = all_index_tuples[rank]
        my_interpolators = _np.empty(len(my_index_tuples), dtype=object)

        if self.interpolator_and_args is None:
            interp_cls, interp_kwargs = (_linND, {'rescale': True})
        else:
            interp_cls, interp_kwargs = self.interpolator_and_args

        # Build the interpolators
        for int_ind, index_tuple in enumerate(my_index_tuples):
            values = [data_at_point[index_tuple] for data_at_point in self.data]
            my_interpolators[int_ind] = interp_cls(self.points, values, **interp_kwargs)

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

    def __init__(self, interpolators, parameter_ranges):
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
