import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['GOTO_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'

import numpy as _np

from scipy.interpolate import LinearNDInterpolator as _linND
from scipy.linalg import logm as _logm, expm as _expm

from ...tools.basistools import change_basis

import dill
import itertools as _itertools
import copy

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
except ImportError:
    _comm = None

__version__ = '0.9.0'

def _print(l):
    if _rank == 0:
        print(l)

def _version_less_than(v1_string, v2_string):
    """check if v1 is less than to v2"""
    v1 = [int(x) for x in v1_string.split('.')]
    v2 = [int(x) for x in v2_string.split('.')]

    for pair in zip(v1,v2):
        if pair[0] > pair[1]:
            return False
        elif pair[0] < pair[1]:
            return True
    # If they are equal, return False
    return False


# This is modeled after the (now depricated?) pygsti.obj.DenseOperator class
class PhysicalProcess(object):
    """
    Returns a new PhysicalProcess object that can pre-compute itself over a range of parameters
    and then use interpolation to reduce forward simulation overhead.

    Parameters
    ----------


    Returns
    -------
    PhysicalProcess
       The gate object. 
    """

    def __init__(self, fname=None, basis=None, comm=_comm, mpi_workers_per_process=1, verbose=False, continuous_gates=False):
        """
        Create a new class instance and initialize variables.
        """
        if fname is None:
            self.__version__ = __version__
            self.base = _np.empty(0)
            self.data = None
            self.name = "UNNAMED"
            self.target = None
            self.parameter_range = None
            self.grouped_by_time = False
            self.interpolation_order = None
            self.interpolated = False
            self._process_function = lambda v: None
            self.verbose = verbose
            self.kwargs = {}
            self.continuous_gates = continuous_gates

            self.has_metadata = False
            self.has_auxdata = False

            if basis is None:
                raise ValueError("'basis' keyword arguement must be specified.")
            else:
                self.basis = basis

            # class variables beginning with _x_ are not saved to disk
            # Define the MPI parameters

            self._x_comm = comm
            self._x_comm.Set_name('comm_world')
            self._x_rank = comm.Get_rank()
            self._x_size = comm.Get_size()
            self._x_mpi_workers_per_process = min(self._x_size, mpi_workers_per_process)

            # Create communicators for each chunk
            self._x_mpi_enabled = False
            self._x_color = self._x_rank // mpi_workers_per_process
            self._x_roots = [x for x in range(self._x_size) if x % mpi_workers_per_process == 0]
            self._x_n_mpi_groups = len(self._x_roots)
            self._x_groupcomm = self._x_comm.Split(self._x_color, self._x_rank)
            self._x_groupcomm.Set_name(f'comm_group_{self._x_color}')
            self._x_grouprank = self._x_groupcomm.Get_rank()
            self._x_groupsize = self._x_groupcomm.Get_size()
            self._x_rootcomm = self._x_comm.Create_group(comm.group.Incl(self._x_roots))
            if self._x_rank in self._x_roots:
                self._x_rootcomm.Set_name('comm_root')
        else:
            self.load(fname)

    def num_params(self):
        # ORIGINAL
        # TODO: This should be populated automatically when the process_function is defined
        return self.n_params

    def to_vector(self):
        # ORIGINAL
        return _np.array(self.v)

    def from_vector(self, v, basis=None, allow_physical=False, return_generator=False):
        self.v = v

        if not self.interpolated:
            if allow_physical:
                print("Computing physical process directly without using interpolation.")
                self.base = self.from_vector_physical(v)
                return self.base
            else:
                raise NotImplementedError("No interpolator is available and physical processes are disallowed.")

        if len(self.parameter_range) == len(v) - 1:
            self.parameter_range = [[min(self.times), max(self.times)]] + self.parameter_range

        if not all([(a >= b) and (a <= c) for a, (b, c) in zip(v, self.parameter_range)]):
            raise ValueError("Parameter out of range.")

        if self.interpolated:
            error_generator = _np.zeros([self.dimension, self.dimension], dtype='float')
            for indi in range(self.dimension):
                for indj in range(self.dimension):
                    error_generator[indi, indj] = _np.float(_np.real(self.interpolator[indi, indj](*v)))

        self.base = self._gate_from_error_generator(error_generator, v=v)

        if return_generator:
            return_object = error_generator
        else:
            return_object = self.base

        # You can request a a different basis
        if basis is not None:
            if basis != self.basis:
                return_object = change_basis(return_object, self.basis, basis)

        return return_object

    def transform(self, S):
        # ORIGINAL
        # Update self with inverse(S) * self * S (used in gauge optimization)
        raise NotImplementedError(f"{self.name} cannot be transformed!")

    def __call__(self, v, basis=None):
        return self.from_vector(v, basis=basis)

    def from_vector_physical(self, v):
        """
        Empty Docstring
        :rtype: numpy.ndarray
        """
        # Don't edit this
        # It is set up like this (rather than being defined directly) so that the parameter vector can be saved
        self.v = v
        if self.verbose:
            print(f"from_vector_physical using {self.kwargs['comm'].Get_name()}")
        return self._process_function(v, **self.kwargs)

    def set_name(self, name):
        # Give the gate a name
        self.name = name
        return self

    def set_target(self, target):
        # Define the target gate. This is used to build the error generators. 
        if self.continuous_gates:
            self.target = target
            self.inv_target = lambda v: _np.linalg.inv(target(v))

        else:
            self.target = lambda v: _np.array(target)
            self.inv_target = lambda v: _np.linalg.inv(target)
        
        return self

    def set_parameter_range(self, parameter_range):
        if self.interpolation_order is not None:
            assert len(self.interpolation_order) == len(
                parameter_range), "'interpolation_order' and 'parameter_range' must be the same length"
        self.parameter_range = parameter_range

        return self

    def set_interpolation_order(self, interpolation_order):
        # Gx.set_interpolation_order([5,6,3,3,3,3])

        if self.parameter_range is not None:
            assert len(interpolation_order) == len(
                self.parameter_range), "'interpolation_order' and 'parameter_range' must be the same length"
        self.interpolation_order = interpolation_order

        return self

    def set_metadata(self, metadata):
        self.has_metadata = True
        self.metadata = metadata

    def get_auxdata(self, v):
        if not self.aux_interpolated:
            raise NotImplementedError("No auxiliary data interpolator is available.")

        if len(self.parameter_range) == len(v) - 1:
            self.parameter_range = [[min(self.times), max(self.times)]] + self.parameter_range

        if not all([(a >= b) and (a <= c) for a, (b, c) in zip(v, self.parameter_range)]):
            raise ValueError("Parameter out of range.")

        if self.aux_interpolated:
            aux_data = _np.zeros([self.auxdim], dtype='float')
            for indi in range(self.auxdim):
                aux_data[indi] = _np.float(self.aux_interpolator[indi](*v))

        return aux_data

    def set_process_function(self, fn, mpi_enabled=False, has_auxdata=False):
        self._process_function = fn
        self.has_auxdata = has_auxdata
        if mpi_enabled:
            # If the function is MPI enabled, it should gather its results to the communicator's 0 node
            # when it's finished.
            self.kwargs['comm'] = self._x_groupcomm
            self._x_mpi_enabled = True

    def set_times(self, times):
        """ Tell the interpolator that it is easy to extract the process matrix for multiple time points if
        all other parameters are the same.

        If this is set, the process function MUST have two modes of operation:
            process_fn([t,p1,p2,...]) for t a single time
            returning a single process matrix
        and
            process_fn([p1,p2,...], times=times) for times a list of times
            returning a list of process matrices, one for each time

        The time parameter is then EXCLUDED from the parameter ranges, and it is set here explicitly.
        """
        self.grouped_by_time = True
        self.times = times
        self.kwargs['times'] = times

    def _error_generator_from_gate(self, gate, v=None):
        # g = exp(e).t
        # e = log(g t^-1)
        if self.verbose:
            print(gate)
        if isinstance(gate[0][1], complex):
            generator = _np.array(_logm(_np.dot(gate, self.inv_target(v))), dtype=complex)
        else:
            generator = _np.array(_np.real(_logm(_np.dot(gate, self.inv_target(v)))), dtype=float)
        return generator

    def _gate_from_error_generator(self, error_generator, v=None):
        # g = exp(e).t
        # e = log(g t^-1)
        dtype = type(error_generator[0,1])
        self.error_generator = error_generator
        return _np.array(_np.dot(_expm(error_generator), self.target(v)), dtype=dtype)

    def _split(self, n, a):
        k, m = divmod(len(a), n)
        return _np.array(list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))

    def _flatten(self, x):
        try:
            return [b for a in x for b in a]
        except TypeError:
            return None

    def interpolate(self):
        # TODO: Interpolate the AUX function as well ** HAVEN'T STARTED THIS HERE
        # TODO: Decide if the aux function should be returned along side the process matrices?
        #       This is easier, so I'll probably stick to it unless people complain.

        # build the interpolation grid
        axial_points = _np.array(
            [_np.linspace(a, b, c) for (a, b), c in zip(self.parameter_range, self.interpolation_order)])
        all_points = _np.array(list(_itertools.product(*axial_points)))

        # scatter across mpi workers
        if self._x_rank in self._x_roots:
            my_points = self._split(self._x_n_mpi_groups, all_points)
            my_points = self._x_rootcomm.scatter(my_points, root=0)
        else:
            my_points = []
        my_points = self._x_groupcomm.bcast(my_points, root=0)
        if self._x_rank in self._x_roots:
            print(f"Group {self._x_color} processing {len(my_points)} points on " +
                  f"{self._x_mpi_workers_per_process} processors.")
            
        # compute the process matrices at each data point
        data = _np.empty(len(my_points), dtype=_np.ndarray)
        if self.has_auxdata:
            auxdata = _np.empty(len(my_points), dtype=_np.ndarray)

        # print(_rank, len(data))
        for ind, point in enumerate(my_points):
            if self.verbose:
                print("Evaluating index {ind}, data = {point}")
            if self.grouped_by_time:
                if self.has_auxdata:
                    data_by_time, auxdata_by_time = self.from_vector_physical(point)
                else:
                    data_by_time = self.from_vector_physical(point)
                # generators_by_times = [self._error_generator_from_gate(gate, v=point) for gate in data_by_time or []]                
                generators_by_times = [self._error_generator_from_gate(gate, v=point) for gate in data_by_time]
                data[ind] = generators_by_times
                if self.has_auxdata:
                    auxdata[ind] = auxdata_by_time
            else:
                if self.verbose:
                    print('Computing error generator')
                if self.has_auxdata:
                    data[ind], auxdata[ind] = self._error_generator_from_gate(self.from_vector_physical(point), v=point)
                else:
                    data[ind] = self._error_generator_from_gate(self.from_vector_physical(point), v=point)

                if self.verbose:
                    print('Computed error generator')

        # Gather data from groups
        if self._x_rank in self._x_roots:
            gathered_data = _np.array(self._flatten(self._x_rootcomm.gather(data, root=0)))
            if self.has_auxdata:
                gathered_auxdata = _np.array(self._flatten(self._x_rootcomm.gather(auxdata, root=0)))

            if self.grouped_by_time:
                self.dimension = data[0][0].shape[0]
                if self.has_auxdata:
                    print(f"Auxdata shape: {auxdata.shape}, {auxdata}")
                    self.auxdim = len(auxdata[0])

            else:
                self.dimension = data[0].shape[0]
                if self.has_auxdata:
                    self.auxdim = len(auxdata[0])
        else:
            gathered_data = None
            gathered_auxdata = None
            self.dimension = None
            self.auxdim = None

        # construct the interpolators
        if self._x_rank == 0:
            if self.grouped_by_time:
                gathered_data = _np.transpose(gathered_data, [1, 0, 2, 3])  # Make time index come first
                gathered_data = self._flatten(gathered_data)  # Flatten along time axis
                all_points = _np.array(list(_itertools.product(self.times, *axial_points)))  # Add time to all_points

                if self.has_auxdata:
                    gathered_auxdata = _np.transpose(gathered_auxdata, [2,0,1])
                    gathered_auxdata = self._flatten(gathered_auxdata)
                    for x,y in zip(gathered_auxdata, all_points):
                        print(x, y)

            self.interpolator = _np.empty([self.dimension, self.dimension], dtype=object)                            
            self.data = gathered_data
            if self.has_auxdata:
                self.aux_interpolator = _np.empty(self.auxdim, dtype=object)
                self.auxdata = gathered_auxdata
            self.points = all_points

        all_points = self._x_comm.bcast(all_points, root=0)
        gathered_data = self._x_comm.bcast(gathered_data, root=0)
        self.dimension = self._x_comm.bcast(self.dimension, root=0)

        if self.has_auxdata:
            gathered_auxdata = self._x_comm.bcast(gathered_auxdata, root=0)
            self.auxdim = self._x_comm.bcast(self.auxdim, root=0)


        all_pairs = self._split(self._x_size,
                              [[indi, indj] for indi in range(self.dimension) for indj in range(self.dimension)])
        my_pairs = all_pairs[self._x_rank]
        my_interpolators = _np.empty(len(my_pairs), dtype='object')

        if self.has_auxdata:
            all_aux_inds = self._split(self._x_size, [indx for indx in range(self.auxdim)])
            my_aux_inds = all_aux_inds[self._x_rank]
            my_aux_interpolators = _np.empty(len(my_aux_inds), dtype='object')

        # Build the interpolators
        for int_ind, (indi, indj) in enumerate(my_pairs):
                values = [datum[indi, indj] for datum in gathered_data]
                my_interpolators[int_ind] = _linND(all_points, values, rescale=True)
        
        if self.has_auxdata:
            for int_ind, aux_ind in enumerate(my_aux_inds):
                aux_values = [datum[aux_ind] for datum in gathered_auxdata]
                my_aux_interpolators[int_ind] = _linND(all_points, aux_values, rescale=True)

        all_interpolators = self._x_comm.gather(my_interpolators, root=0)
        if self._x_rank == 0:
            all_interpolators = self._flatten(all_interpolators)
            self.interpolator = _np.empty([self.dimension, self.dimension], dtype='object')
            for interp, (indi, indj) in zip(all_interpolators, self._flatten(all_pairs)):
                self.interpolator[indi, indj] = interp
        
        if self.has_auxdata:
            all_aux_interpolators = self._x_comm.gather(my_aux_interpolators, root=0)
            print(all_aux_inds)
            for x in all_aux_interpolators:
                print(x)
            if self._x_rank == 0:
                all_aux_interpolators = self._flatten(all_aux_interpolators)
                self.aux_interpolator = _np.empty(self.auxdim, dtype='object')
                for interp, (ind) in zip(all_aux_interpolators, self._flatten(all_aux_inds)):
                    self.aux_interpolator[ind] = interp
            self.aux_interpolated = True

        self.interpolated = True

    def save(self, fname):
        """Save the data for this object to a file"""
        if self._x_rank == 0:
            # save_keys = ['interpolator', 'interpolation_order',
            #              'parameter_range', 'times',
            #              'target', 'inv_target']
            # save_dict = {}
            # for key in save_keys:
            #     save_dict[key] = copy.copy(self.__dict__[key])


            # # make a copy of the object dictionary and delete the MPI components
            safe_dict = self.__dict__.copy()
            del_keys = ['_process_function', '_aux_function', 'kwargs']
            for key in safe_dict.keys():
                if '_x_' in key:
                    del_keys += [key]
            for key in del_keys:
                try:
                    del safe_dict[key]
                except KeyError:
                    pass

            # dump to a file
            with open(fname, 'wb') as f:
                dill.dump(safe_dict, f)

    def load(self, fname):
        """Read the data in from a file"""
        # self.__dict__.clear()
        with open(fname, 'rb') as f:
            loaded_dict = dill.load(f)
            if "__version__" not in loaded_dict.keys():
                loaded_dict["__version__"] = "0.1.0"
            self.__dict__.update(loaded_dict)
            
            # If the version is earlier than 0.8.0, then fix the target function
            print(f"Loading {fname} made with version {self.__version__}")
            
            if _version_less_than(self.__version__, "0.8.0"):
                if not callable(self.target):
                    target = self.target.copy()
                    self.target = lambda v: target
                    self.inv_target = lambda v: _np.linalg.inv(target)
            
            test_params = [_np.mean(x) for x in self.parameter_range]    



if __name__ == '__main__':

    # from interface import *
    from process_tomography import do_process_tomography, vec, unvec, change_basis

    class ProcessFunction(object):
        def __init__(self):
            import numpy as _np
            import scipy as _sp

            self.Hx = _np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, -1],
                                [0, 0, 1, 0]], dtype='float')
            self.Hy = _np.array([[0, 0, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 0, 0],
                                [0, -1, 0, 0]], dtype='float')
            self.Hz = _np.array([[0, 0, 0, 0],
                                [0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0]], dtype='float')

            self.dephasing_generator = _np.diag([0, -1, -1, 0])
            self.decoherence_generator = _np.diag([0, -1, -1, -1])

        def advance(self, state, v = None, times = None):
            state = _np.array(state, dtype='complex')
            if times is None:
                t, omega, phase, detuning, dephasing, decoherence = v
                times = [t]
            else:
                omega, phase, detuning, dephasing, decoherence = v

            H = (omega * _np.cos(phase) * self.Hx + omega * _np.sin(phase) * self.Hy + detuning * self.Hz)
            L = dephasing * self.dephasing_generator + decoherence * self.decoherence_generator

            processes = [change_basis(_expm((H + L) * t),
                                                              'pp', 'col') for t in times]
            states = [unvec(_np.dot(process, vec(_np.outer(state, state.conj())))) for process in processes]

            return states

        def __call__(self, v, times=None, comm=_comm):
            print(f'Calling process tomography as {comm.Get_rank()} of {comm.Get_size()} on {comm.Get_name()}.')
            processes = do_process_tomography(self.advance, opt_args={'v':v, 'times':times},
                                              n_qubits = 1, time_dependent=True, comm=comm)

            return processes

    gy = PhysicalProcess(mpi_workers_per_process=1)
    gy.set_process_function(ProcessFunction(), mpi_enabled=True)
    target = change_basis(_np.array([[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,0,-1],
                                     [0,0,1,0]], dtype='complex'), 'pp', 'qsim')
    gy.set_target(target)

    # # # Evaluate one time point per run
    # # gy.set_parameter_range([[_np.pi/2-.5, _np.pi/2+.5],[0.9,1.1],[-.1,.1],[-.1,.1],[0,0.1],[0,0.1]])
    # # gy.set_interpolation_order([3,3,3,3,3,3])
    #
    # Evaluate many time points per run
    # Notice that the number of parameters listed below is 1 fewer than in the previous example
    gy.set_parameter_range([[0.9, 1.1], [-.1, .1], [-.1, .1], [0, 0.1], [0, 0.1]])
    gy.set_interpolation_order([3,3,3,3,3])
    gy.set_times(_np.linspace(_np.pi / 2, _np.pi / 2 + .5, 10))

    gy.interpolate()
    gy.save('intertest.dat')

    # #
    # # def diamond_norm(a, b):
    # #     A = pygsti.tools.jamiolkowski_iso(a)
    # #     B = pygsti.tools.jamiolkowski_iso(b)
    # #     return pygsti.tools.diamonddist(a, b)
    # #
    # #

    if _rank == 0:
        from matplotlib import pyplot as plt
        from process_tomography import change_basis
        cb = lambda x: change_basis(x, 'qsim', 'pp')
        gx = PhysicalProcess('intertest.dat')
        print(_np.round_(gx.from_vector([_np.pi / 2 + .3, 1, 0, 0, 0, 0]),3))
        # print(_np.round_(gx.from_vector_physical([_np.pi / 2 + .3, 1, 0, 0, 0, 0]),3))
        nom = _np.array([_np.pi / 2, 1, 0, 0, 0, 0])
        labels = ["Timing Error", "Amplitude Error", "Phase Error", "Frequency Error", "Additional Dephasing", "Additional Decoherence"]
        for ind in range(len(nom)):
            tnom = nom.copy()
            tnom[ind] += .1
            ergen = _np.round_(cb(gx._error_generator_from_gate(gx(tnom))), 3) / .1
            plt.matshow(ergen.real)
            plt.title(labels[ind])
            print(ind)
            print(ergen)
            plt.savefig(f'./figures/param_{ind}.pdf')
