from ..testutils import BaseTestCase, compare_files, temp_files, regenerate_references
import unittest
import numpy as np
import pickle
import time
import warnings

import pygsti
from pygsti.extras import interpygate as interp
from pygsti.tools import change_basis
from pygsti.extras.interpygate.process_tomography import run_process_tomography, vec, unvec

from scipy.linalg import logm as _logm, expm as _expm
import numpy as _np
import scipy as _sp

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
except ImportError:
    _comm = None
    _rank = 0
    _size = 1

mpi_workers_per_process = 1


class ExampleProcess(interp.PhysicalProcess):
    def __init__(self):
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
        num_params = 6  # omega (0), phase (1), detuning (2), dephasing (3), decoherence (4), time (5)
        process_shape = (4, 4)
        super().__init__(num_params, process_shape,
                         aux_shape=(),  # a single float
                         num_params_evaluated_as_group=0)

    def advance(self, state, v, t):
        state = _np.array(state, dtype='complex')
        omega, phase, detuning, dephasing, decoherence = v

        H = (omega * _np.cos(phase) * self.Hx + omega * _np.sin(phase) * self.Hy + detuning * self.Hz)
        L = dephasing * self.dephasing_generator + decoherence * self.decoherence_generator

        process = change_basis(_expm((H + L) * t), 'pp', 'col')
        state = unvec(_np.dot(process, vec(_np.outer(state, state.conj()))))
        return state

    def create_process_matrix(self, v, comm=None):

        t = v[5]
        vv = v[:5]

        def state_to_process_mxs(state):
            return self.advance(state, vv, t)
        #print(f'Calling process tomography as {comm.Get_rank()} of {comm.Get_size()} on {comm.Get_name()}.')
        process = run_process_tomography(state_to_process_mxs,
                                         n_qubits=1, basis='pp', time_dependent=False, comm=comm, verbose=False)
        return _np.array(process) if (process is not None) else None  # must return an *array* of appropriate shape

    def create_aux_info(self, v, comm=None):
        omega, phase, detuning, dephasing, decoherence, t = v
        return t * omega


class ExampleProcess_timedep(interp.PhysicalProcess):
    def __init__(self):
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
        num_params = 6  # omega (0), phase (1), detuning (2), dephasing (3), decoherence (4), time (5)
        process_shape = (4, 4)
        super().__init__(num_params, process_shape,
                         aux_shape=(),  # a single float
                         num_params_evaluated_as_group=1)  # time values can be evaluated all at once

    def advance(self, state, v, times):
        state = _np.array(state, dtype='complex')
        omega, phase, detuning, dephasing, decoherence = v

        H = (omega * _np.cos(phase) * self.Hx + omega * _np.sin(phase) * self.Hy + detuning * self.Hz)
        L = dephasing * self.dephasing_generator + decoherence * self.decoherence_generator

        processes = [change_basis(_expm((H + L) * t), 'pp', 'col') for t in times]
        states = [unvec(_np.dot(process, vec(_np.outer(state, state.conj())))) for process in processes]

        return states

    def create_process_matrices(self, v, grouped_v, comm=None):
        times = grouped_v[0]
        def state_to_process_mxs(state):
            return self.advance(state, v, times)
        #print(f'Calling process tomography as {comm.Get_rank()} of {comm.Get_size()} on {comm.Get_name()}.')
        #print(f'DEBUG {comm.Get_rank()}: ', times, v)
        processes = run_process_tomography(state_to_process_mxs,
                                           n_qubits=1, basis='pp', time_dependent=True, comm=comm, verbose=False)
        return _np.array(processes) if (processes is not None) else None  # must return an *array* of appropriate shape

    def create_aux_infos(self, v, grouped_v, comm=None):
        omega, phase, detuning, dephasing, decoherence = v
        times = grouped_v[0]
        return _np.array([t * omega for t in times], 'd')


class InterpygateTestCase(BaseTestCase):

    def test_timedep_op(self):
        example_process = ExampleProcess_timedep()
        target_mxs = example_process.create_process_matrices(_np.array([1.0, 0.0, 0.0, 0.0, 0.0]), [[_np.pi / 2]], comm=_comm)
        if _comm is None or _comm.rank == 0:
            target_mx = target_mxs[0]
            target_op = pygsti.obj.StaticDenseOp(target_mx)
            print(target_op)
            if _comm is not None: _comm.bcast(target_op, root=0)
        else:
            target_op = _comm.bcast(None, root=0)

        param_ranges = ([(0.9, 1.1, 2),  # omega
                         (-.1, 0, 2),   # phase
                         (-.2, -.1, 2),   # detuning
                         (0, 0.1, 2),    # dephasing
                         (0.1, 0.2, 2),    # decoherence
                         _np.linspace(_np.pi / 2, _np.pi / 2 + .5, 10)  # time
                        ])
        interp_op = interp.InterpolatedDenseOp.create_by_interpolating_physical_process(
            target_op, example_process, param_ranges, comm=_comm,
            mpi_workers_per_process=mpi_workers_per_process)

        self.assertEqual(interp_op.num_params, 6)
        interp_op.from_vector([1.1, -0.01, -0.11, 0.055, 0.155, 1.59])
        self.assertArraysAlmostEqual(_np.array([1.1, -0.01, -0.11, 0.055, 0.155, 1.59]), interp_op.to_vector())

        expected = _np.array([[ 1.00000000e+00, -5.14632352e-17,  1.58551100e-17, -8.59219991e-18],
                              [ 6.10412172e-19,  7.07796561e-01,  6.01596594e-02, -9.41693123e-02],
                              [-1.58441909e-18, -7.76254825e-02, -1.56153689e-01, -7.30978833e-01],
                              [ 7.43467815e-18, -7.91773419e-02,  7.32730647e-01, -1.10922086e-01]])
        #print(interp_op.to_dense())
        self.assertArraysAlmostEqual(expected, interp_op.to_dense())

    def test_timedep_factory(self):
        class TargetOpFactory(pygsti.obj.OpFactory):
            def __init__(self):
                self.process = ExampleProcess_timedep()
                pygsti.obj.OpFactory.__init__(self, dim=4, evotype="densitymx")

            def create_object(self, args=None, sslbls=None):
                assert(sslbls is None)  # don't worry about sslbls for now -- these are for factories that can create gates placed at arbitrary circuit locations
                assert(len(args) == 2)  # t (time), omega
                t, omega = args
                mx = self.process.create_process_matrices(_np.array([omega, 0.0, 0.0, 0.0, 0.0]), [[t]], comm=None)[0]
                return pygsti.obj.StaticDenseOp(mx)

        arg_ranges = [_np.linspace(_np.pi / 2, _np.pi / 2 + .5, 10),  # time
                      (0.9, 1.1, 2)  # omega
                      ]

        param_ranges = [(-.1, .1, 2),  # phase
                        (-.1, .1, 2),  # detuning
                        (0, 0.1, 2),   # dephasing
                        (0, 0.1, 2)    # decoherence
                        ]
        arg_indices = [5, 0]  # indices for time and omega within ExampleProcess's parameters (see ExampleProcess.__init__)

        example_process = ExampleProcess_timedep()
        opfactory = interp.InterpolatedOpFactory.create_by_interpolating_physical_process(
            TargetOpFactory(), example_process, arg_ranges, param_ranges, arg_indices, comm=_comm,
            mpi_workers_per_process=mpi_workers_per_process)

        self.assertEqual(opfactory.num_params, 4)
        v = _np.array([0.01, 0.01, 0.055, 0.055])
        opfactory.from_vector(v)
        self.assertArraysAlmostEqual(v, opfactory.to_vector())
        op = opfactory.create_op((1.59, 1.1))
        self.assertArraysAlmostEqual(v, op.to_vector())
        #print(op.to_dense())
        #print(op.aux_info)
        expected = _np.array([[ 1.00000000e+00, -3.67730279e-17, -4.28676069e-17,  6.20501240e-18],
                              [ 5.44719386e-19,  8.41211070e-01,  5.73783231e-03,  1.81516805e-02],
                              [-1.70671506e-17,  1.36836825e-02, -1.75019744e-01, -8.64632441e-01],
                              [ 7.80124509e-17, -7.41873094e-04,  8.65665135e-01, -1.31573407e-01]])
        self.assertArraysAlmostEqual(expected, op.to_dense())
        self.assertAlmostEqual(op.aux_info, 1.749)

    def test_timeindep_op(self):
        example_process = ExampleProcess()
        target_mx = example_process.create_process_matrix(_np.array([1.0, 0.0, 0.0, 0.0, 0.0, _np.pi / 2]), comm=_comm)
        if _comm is None or _comm.rank == 0:
            target_op = pygsti.obj.StaticDenseOp(target_mx)
            print(target_op)
            if _comm: _comm.bcast(target_op, root=0)
        else:
            target_op = _comm.bcast(None, root=0)

        param_ranges = ([(0.9, 1.1, 2),  # omega
                         (-.1, 0, 2),   # phase
                         (-.2, -.1, 2),   # detuning
                         (0, 0.1, 2),    # dephasing
                         (0.1, 0.2, 2),    # decoherence
                         _np.linspace(_np.pi / 2, _np.pi / 2 + .5, 10)  # time
                         ])
        interp_op = interp.InterpolatedDenseOp.create_by_interpolating_physical_process(
            target_op, example_process, param_ranges, comm=_comm,
            mpi_workers_per_process=mpi_workers_per_process)

        self.assertEqual(interp_op.num_params, 6)
        interp_op.from_vector([1.1, -0.01, -0.11, 0.055, 0.155, 1.59])
        self.assertArraysAlmostEqual(_np.array([1.1, -0.01, -0.11, 0.055, 0.155, 1.59]), interp_op.to_vector())

        expected = _np.array([[ 1.00000000e+00, -5.14632352e-17,  1.58551100e-17, -8.59219991e-18],
                              [ 6.10412172e-19,  7.07796561e-01,  6.01596594e-02, -9.41693123e-02],
                              [-1.58441909e-18, -7.76254825e-02, -1.56153689e-01, -7.30978833e-01],
                              [ 7.43467815e-18, -7.91773419e-02,  7.32730647e-01, -1.10922086e-01]])
        #print(interp_op.to_dense())
        self.assertArraysAlmostEqual(expected, interp_op.to_dense())

    def test_timeindep_factory(self):
        class TargetOpFactory(pygsti.obj.OpFactory):
            def __init__(self):
                self.process = ExampleProcess()
                pygsti.obj.OpFactory.__init__(self, dim=4, evotype="densitymx")

            def create_object(self, args=None, sslbls=None):
                assert(sslbls is None)  # don't worry about sslbls for now -- these are for factories that can create gates placed at arbitrary circuit locations
                assert(len(args) == 2)  # t (time), omega
                t, omega = args
                mx = self.process.create_process_matrix(_np.array([omega, 0.0, 0.0, 0.0, 0.0, t]), comm=None)
                return pygsti.obj.StaticDenseOp(mx)

        arg_ranges = [_np.linspace(_np.pi / 2, _np.pi / 2 + .5, 10),  # time
                      (0.9, 1.1, 2)  # omega
                      ]

        param_ranges = [(-.1, .1, 2),  # phase
                        (-.1, .1, 2),  # detuning
                        (0, 0.1, 2),   # dephasing
                        (0, 0.1, 2)    # decoherence
                        ]
        arg_indices = [5, 0]  # indices for time and omega within ExampleProcess's parameters (see ExampleProcess.__init__)

        example_process = ExampleProcess()
        opfactory = interp.InterpolatedOpFactory.create_by_interpolating_physical_process(
            TargetOpFactory(), example_process, arg_ranges, param_ranges, arg_indices, comm=_comm,
            mpi_workers_per_process=mpi_workers_per_process)

        self.assertEqual(opfactory.num_params, 4)
        v = _np.array([0.01, 0.01, 0.055, 0.055])
        opfactory.from_vector(v)
        self.assertArraysAlmostEqual(v, opfactory.to_vector())
        op = opfactory.create_op((1.59, 1.1))
        self.assertArraysAlmostEqual(v, op.to_vector())
        #print(op.to_dense())
        #print(op.aux_info)
        expected = _np.array([[ 1.00000000e+00, -3.67730279e-17, -4.28676069e-17,  6.20501240e-18],
                              [ 5.44719386e-19,  8.41211070e-01,  5.73783231e-03,  1.81516805e-02],
                              [-1.70671506e-17,  1.36836825e-02, -1.75019744e-01, -8.64632441e-01],
                              [ 7.80124509e-17, -7.41873094e-04,  8.65665135e-01, -1.31573407e-01]])
        self.assertArraysAlmostEqual(expected, op.to_dense())
        self.assertAlmostEqual(op.aux_info, 1.749)

    def test_process_tomography(self):
        """ Demonstrate the process tomography function with (potentially) time-dependent outputs. """
        sigI = _np.array([[1, 0], [0, 1]], dtype='complex')
        sigX = _np.array([[0, 1], [1, 0]], dtype='complex')
        sigY = _np.array([[0, -1.j], [1.j, 0]], dtype='complex')
        sigZ = _np.array([[1, 0], [0, -1]], dtype='complex')
        theta = .32723
        u = _np.cos(theta) * sigI + 1.j * _np.sin(theta) * sigX
        v = _np.sin(theta) * sigI - 1.j * _np.cos(theta) * sigX

        U = _np.kron(u, v)
        test_process = _np.kron(U.conj().T, U)

        def single_time_test_function(pure_state, test_process=test_process):
            rho = vec(_np.outer(pure_state, pure_state.conj()))
            return unvec(_np.dot(test_process, rho))

        def multi_time_test_function(pure_state, test_process=test_process):
            rho = vec(_np.outer(pure_state, pure_state.conj()))
            return [unvec(_np.dot(test_process, rho)), unvec(_np.dot(_np.linalg.matrix_power(test_process, 2), rho))]

        process_matrix = run_process_tomography(single_time_test_function, n_qubits=2, verbose=False)
        if _rank == 0:
            test_process_pp = change_basis(test_process, 'col', 'pp')
            print("\nSingle-time test result should be True:")
            print(_np.isclose(process_matrix, test_process_pp).all())

        process_matrices = run_process_tomography(multi_time_test_function, n_qubits=2, verbose=False, time_dependent=True)
        if _rank == 0:
            test_process = change_basis(test_process, 'col', 'pp')
            print("\nMulti-time test result should be [True, False]:")
            print([_np.isclose(x, test_process).all() for x in process_matrices])
