# This file is designed to be run via: mpiexec -np 4 python -W ignore run_me_with_mpiexec.py
# This does not use nosetests because I want to set verbosity differently based on rank (quiet if not rank 0)
# By wrapping asserts in comm.rank == 0, only rank 0 should fail (should help with output)
# Can run with different number of procs, but 4 is minimum to test all modes (pure MPI, pure shared mem, and mixed)

import os

import numpy as np
from mpi4py import MPI

import pygsti
from pygsti.modelpacks import smq1Q_XY as smq1Q_XY
from pygsti.modelpacks import smq1Q_XYI as std
from pygsti.optimize import arraysinterface as _ari
from pygsti.optimize import simplerlm as lm

wcomm = MPI.COMM_WORLD


class _Protocol4OnlyRestart:
    def __reduce_ex__(self, protocol):
        if protocol != 4:
            raise ValueError('This restart state requires pickle protocol 4')
        return (_Protocol4OnlyRestart, ())


class ParallelTest(object):
    # No setup here, must be defined in the derived classes

    def run_optimizer_error_payloads(self):
        from pygsti.optimize import OptimizationProblem, OptimizerOutcome, OptimizationStatus
        from pygsti.optimize.collective import solve_on_root, CollectiveEvaluationError
        from pygsti.optimize.domain import EvaluationDomain, EvaluationDomainError, guard_problem
        comm = self.ralloc.comm
        for domain_metadata in (True, False):
            def value(x):
                if not domain_metadata and comm.rank == 1 and x[0] == 1.:
                    raise EvaluationDomainError(x, generation=lambda: None)
                return float(x @ x)
            problem = OptimizationProblem([0.], value)
            if domain_metadata:
                generation = (lambda: None) if comm.rank == 1 else 0
                problem = guard_problem(problem, EvaluationDomain(
                    lambda x: not (comm.rank == 1 and x[0] == 1.), generation=generation))
            def solve(proxy):
                try:
                    proxy.value([1.])
                except CollectiveEvaluationError as error:
                    assert 'Unserializable evaluation error metadata' in str(error)
                else:
                    raise AssertionError('Unpicklable domain error was not rejected')
                assert proxy.value([0.]) == 0.
                return OptimizerOutcome([0.], OptimizationStatus.CONVERGED)
            outcome = solve_on_root(solve, problem, comm)
            assert outcome.status == OptimizationStatus.CONVERGED
            np.testing.assert_array_equal(outcome.x, [0.])

        previous_protocol = MPI.pickle.PROTOCOL
        try:
            MPI.pickle.PROTOCOL = 4
            problem = OptimizationProblem([0.], lambda x: float(x @ x))
            def solve(proxy):
                return OptimizerOutcome([0.], OptimizationStatus.CONVERGED,
                                        restart_state=_Protocol4OnlyRestart())
            outcome = solve_on_root(solve, problem, comm)
            assert isinstance(outcome.restart_state, _Protocol4OnlyRestart)
        finally:
            MPI.pickle.PROTOCOL = previous_protocol

    def test_simulate_data(self):
        comm = self.ralloc.comm

        exp_design = std.create_gst_experiment_design(4)
        mdl_datagen = std.target_model().depolarize(op_noise=0.1, spam_noise=0.01)

        ds_serial = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234, comm=None)
        ds_parallel = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234, comm=comm)

        if comm is None or comm.rank == 0:
            assert (set(ds_serial.keys()) == set(ds_parallel.keys()))
            for key in ds_serial.keys():
                assert (ds_serial[key].to_dict() == ds_parallel[key].to_dict())

    def test_MPI_probs(self):
        comm = self.ralloc.comm

        #Create some model
        mdl = std.target_model()
        mdl.kick(0.1,seed=1234)
    
        #Get some operation sequences
        maxLengths = [1, 2, 4]
        circuits = pygsti.circuits.create_lsgst_circuits(
            list(std.target_model().operations.keys()), std.prep_fiducials(),
            std.meas_fiducials(), std.germs(), maxLengths)
    
        #Check all-spam-label bulk probabilities
        def compare_prob_dicts(a,b,indices=None):
            for opstr in circuits:
                for outcome in a[opstr].keys():
                    if indices is None:
                        assert (np.linalg.norm(a[opstr][outcome] - b[opstr][outcome]) < 1e-6)
                    else:
                        for i in indices:
                            assert (np.linalg.norm(a[opstr][outcome][i] - b[opstr][outcome][i]) < 1e-6)
    
        # non-split tree => automatically adjusts wrt_block_size to accomodate
        #                    the number of processors
        serial = mdl.sim.bulk_probs(circuits, clip_to=(-1e6,1e6))
        parallel = mdl.sim.bulk_probs(circuits, clip_to=(-1e6,1e6), resource_alloc=self.ralloc)
        if comm is None or comm.rank == 0:  # output is only given on root proc
            compare_prob_dicts(serial, parallel)
    
        serial = mdl.sim.bulk_dprobs(circuits)
        parallel = mdl.sim.bulk_dprobs(circuits, resource_alloc=self.ralloc)
        if comm is None or comm.rank == 0:  # output is only given on root proc
            compare_prob_dicts(serial, parallel)
    
        serial = mdl.sim.bulk_hprobs(circuits)
        parallel = mdl.sim.bulk_hprobs(circuits, resource_alloc=self.ralloc)
        if comm is None or comm.rank == 0:  # output is only given on root proc
            compare_prob_dicts(serial, parallel, (0, 1, 2))

    def test_MPI_products(self):
        comm = self.ralloc.comm
        
        #Create some model
        mdl = std.target_model()

        mdl.kick(0.1, seed=1234)

        #Get some operation sequences
        maxLengths = [1,2,4,8]
        gstrs = pygsti.circuits.create_lsgst_circuits(
            std.target_model(), std.fiducials(), std.fiducials(), std.germs(), maxLengths)

        #Check bulk products

        #bulk_product - no parallelization unless layout is split
        serial = mdl.sim.bulk_product(gstrs, scale=False)
        parallel = mdl.sim.bulk_product(gstrs, scale=False, resource_alloc=self.ralloc)
        assert(np.linalg.norm(serial-parallel) < 1e-6)

        serial_scl, sscale = mdl.sim.bulk_product(gstrs, scale=True)
        parallel, pscale = mdl.sim.bulk_product(gstrs, scale=True, resource_alloc=self.ralloc)
        assert(np.linalg.norm(serial_scl*sscale[:,None,None] -
                              parallel*pscale[:,None,None]) < 1e-6)

        #bulk_dproduct - no split tree => parallel by col
        serial = mdl.sim.bulk_dproduct(gstrs, scale=False)
        parallel = mdl.sim.bulk_dproduct(gstrs, scale=False, resource_alloc=self.ralloc)
        assert(np.linalg.norm(serial-parallel) < 1e-6)

        serial_scl, sscale = mdl.sim.bulk_dproduct(gstrs, scale=True)
        parallel, pscale = mdl.sim.bulk_dproduct(gstrs, scale=True, resource_alloc=self.ralloc)
        assert(np.linalg.norm(serial_scl*sscale[:,None,None,None] -
                              parallel*pscale[:,None,None,None]) < 1e-6)
    
    def test_objfn_generator(self):
        params = [
            ("map", "logl", 1), ("map", "logl", 4),
            ("map", "chi2", 1), ("map", "chi2", 4),
            ("matrix", "logl", 1), ("matrix", "logl", 4),
            ("matrix", "chi2", 1), ("matrix", "chi2", 4)
        ]
        for sim, objfn, natoms in params:
            yield self.run_objfn_values, sim, objfn, natoms

    def run_objfn_values(self, sim, objfn, natoms):
        comm = self.ralloc.comm

        mdl = std.target_model()
        exp_design = std.create_gst_experiment_design(1)
        mdl_datagen = mdl.depolarize(op_noise=0.01, spam_noise=0.01)
        ds = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234, comm=comm)
    
        builder = pygsti.objectivefns.ObjectiveFunctionBuilder.create_from(objfn)
        builder.additional_args['array_types'] = ('EP', 'EPP')  # HACK - todo this better
    
        if sim == 'map':
            mdl.sim = pygsti.forwardsims.MapForwardSimulator(num_atoms=natoms)
        elif sim == 'matrix':
            mdl.sim = pygsti.forwardsims.MatrixForwardSimulator(num_atoms=natoms)
        else:
            raise RuntimeError("Improper sim type passed by test_objfn_generator")

        circuits = exp_design.all_circuits_needing_data[0:10]
        objfn_parallel = builder.build(mdl, ds, circuits, self.ralloc, verbosity=0)
        objfn_serial = builder.build(mdl, ds, circuits, None, verbosity=0)
    
        #LSVEC TEST
        v_ref = objfn_serial.lsvec()
        v = objfn_parallel.lsvec()
        globalv = objfn_parallel.layout.gather_local_array('e', v)
    
        if comm is None or comm.rank == 0:
            finalv = np.empty_like(globalv); off = 0
            for c in circuits:
                indices, outcomes = objfn_parallel.layout.global_layout.indices_and_outcomes(c)
                assert(outcomes == (('0',), ('1',)))  # I think this should always be the ordering (?)
                finalv[off:off + len(outcomes)] = globalv[indices]
                off += len(outcomes)
    
            finalv_ref = np.empty_like(v_ref); off = 0
            for c in circuits:
                indices, outcomes = objfn_serial.layout.indices_and_outcomes(c)
                assert(outcomes == (('0',), ('1',)))  # I think this should always be the ordering (?)
                finalv_ref[off:off + len(outcomes)] = v_ref[indices]
                off += len(outcomes)
    
            assert np.allclose(finalv, finalv_ref)
    
        #TODO: DLSVEC?
    
        #HESSIAN TEST
        hessian_ref = objfn_serial.hessian()
        hessian = objfn_parallel.hessian()  # already a global qty, just on root proc
        bhessian_ref = objfn_serial.hessian_brute()
        bhessian = objfn_parallel.hessian_brute()
    
        if comm is None or comm.rank == 0:
            assert np.allclose(bhessian_ref, hessian_ref)
            assert np.allclose(hessian, hessian_ref)
            assert np.allclose(bhessian, bhessian_ref)

    def test_fills_generator(self):
        sims = ["map", "matrix"]
        # XYI model with maxL = 1 has 92 circuits and 60 parameters
        layout_params = [(1, None), (4, None),
                         (1, 15), (4, 15)
                     ]
        for s in sims:
            for natoms, nparams in layout_params:
                yield self.run_fills, s, natoms, nparams

    def run_fills(self, sim, natoms, nparams):
        comm = self.ralloc.comm

        #Create some model
        mdl = std.target_model()
        mdl.kick(0.1,seed=1234)
    
        #Get some operation sequences
        maxLengths = [1]
        circuits = pygsti.circuits.create_lsgst_circuits(
            list(std.target_model().operations.keys()), std.prep_fiducials(),
            std.meas_fiducials(), std.germs(), maxLengths)
        nP = mdl.num_params

        if sim == 'map':
            mdl.sim = pygsti.forwardsims.MapForwardSimulator(num_atoms=natoms, param_blk_sizes=(nparams, nparams))
        elif sim == 'matrix':
            mdl.sim = pygsti.forwardsims.MatrixForwardSimulator(num_atoms=natoms, param_blk_sizes=(nparams, nparams))
        else:
            raise RuntimeError("Improper sim type passed by test_fills_generator")

        serial_layout = mdl.sim.create_layout(circuits, array_types=('E','EP','EPP'), derivative_dimensions=(nP,))

        nE = serial_layout.num_elements
        nC = len(circuits)

        vp_serial  = np.empty(  nE, 'd')
        vdp_serial = np.empty( (nE,nP), 'd')
        vhp_serial = np.empty( (nE,nP,nP), 'd')
    
        mdl.sim.bulk_fill_probs(vp_serial, serial_layout)
        mdl.sim.bulk_fill_dprobs(vdp_serial, serial_layout)
        mdl.sim.bulk_fill_hprobs(vhp_serial, serial_layout)

        #Note: when there are multiple atoms, the serial_layout returned above may not preserve
        # the original circuit ordering, i.e., serial_layout.circuits != circuits.  The global
        # layout does have this property, and so we use it to avoid having to lookup which circuit
        # is which index in serial_layout.  No gathering is needed, since there only 1 processor.
        global_serial_layout = serial_layout.global_layout
    
        #Use a parallel layout to compute the same probabilities & their derivatives
        local_layout = mdl.sim.create_layout(circuits, array_types=('E','EP','EPP'), derivative_dimensions=(nP,),
                                             resource_alloc=self.ralloc)
    
        vp_local = local_layout.allocate_local_array('e', 'd')
        vdp_local = local_layout.allocate_local_array('ep', 'd')
        vhp_local = local_layout.allocate_local_array('epp', 'd')
    
        mdl.sim.bulk_fill_probs(vp_local, local_layout)
        mdl.sim.bulk_fill_dprobs(vdp_local, local_layout)
        mdl.sim.bulk_fill_hprobs(vhp_local, local_layout)
    
    
        #Gather data to produce arrays for the "global" layout (global_parallel_layout should be the same on all procs)
        # but only on proc 0
        global_parallel_layout = local_layout.global_layout
        vp_global_parallel = local_layout.gather_local_array('e', vp_local)
        vdp_global_parallel = local_layout.gather_local_array('ep', vdp_local)
        vhp_global_parallel = local_layout.gather_local_array('epp', vhp_local)
    
        #Free the local arrays when we're done with them (they could be shared mem)
        local_layout.free_local_array(vp_local)
        local_layout.free_local_array(vdp_local)
        local_layout.free_local_array(vhp_local)
    
        #Compare the two, but note that different layouts may order the elements differently,
        # so we can't just compare the arrays directly - we have to use the layout to map
        # circuit index -> element indices:
        if comm is None or comm.rank == 0:
            for i,opstr in enumerate(circuits):
                assert(np.linalg.norm(vp_global_parallel[ global_parallel_layout.indices_for_index(i) ] -
                                      vp_serial[ global_serial_layout.indices_for_index(i) ]) < 1e-6)
                assert(np.linalg.norm(vdp_global_parallel[ global_parallel_layout.indices_for_index(i) ] -
                                      vdp_serial[ global_serial_layout.indices_for_index(i) ]) < 1e-6)
                assert(np.linalg.norm(vhp_global_parallel[ global_parallel_layout.indices_for_index(i) ] -
                                      vhp_serial[ global_serial_layout.indices_for_index(i) ]) < 1e-6)

    def test_MPI_printer(self):
        comm = self.ralloc.comm

        #Test output of each rank to separate file:
        pygsti.baseobjs.VerbosityPrinter._comm_path = "./"
        pygsti.baseobjs.VerbosityPrinter._comm_file_name = "mpi_test_output"
        printer = pygsti.baseobjs.VerbosityPrinter(verbosity=2, comm=comm)
        printer.log("HELLO!")

    def _create_dist_layout(self, natoms):
        mdl = smq1Q_XY.target_model('CPTPLND')
        mdl.sim = pygsti.forwardsims.MatrixForwardSimulator(num_atoms=natoms)
        circuits = smq1Q_XY.create_gst_experiment_design(1).all_circuits_needing_data
        layout = mdl.sim.create_layout(
            circuits, array_types=('e', 'ep', 'jtj', 'jtf'),
            derivative_dimensions=(mdl.num_params,), resource_alloc=self.ralloc,
        )
        return layout

    def run_degenerate_simplish_leastsq(self, natoms, x_limits_clip=False):
        layout = self._create_dist_layout(natoms)
        nE = layout.global_num_elements
        nP = layout.global_num_params  # 24 (6 per rank in fine param distribution on 4 ranks)

        x0 = np.zeros(nP, 'd')
        x0[7] = 0.5
        x_limits = None
        if x_limits_clip:
            x_limits = np.column_stack((-10.0 * np.ones(nP, 'd'), 10.0 * np.ones(nP, 'd')))
            x_limits[0, 1] = 0.5 * lm.JAC_SHIFT

        def full_obj(x, oob_check=False):
            r = np.zeros(nE, 'd')
            r[0] = x[0] ** 2 - 0.04                   # rank 0: exact zero col at x0[0]=0 -> target 0.2
            r[1] = (x[7] - 0.5) ** 2 - 0.09           # rank 1: exact zero col at x0[7]=0.5 -> target 0.8
            r[2] = 1e-12 * x[14] + x[14] ** 2 - 0.01  # rank 2: norm 1e-12 <= ZERO_COLUMN_RTOL * global max -> target 0.1
            for k, idx in enumerate(range(18, 24)):   # rank 3: all 6 cols have norm 1.0 (no zero cols on rank 3)
                r[3 + k] = x[idx] - 1.0
            return r

        def full_jac(x):
            J = np.zeros((nE, nP), 'd')
            J[0, 0] = 2.0 * x[0]
            J[1, 7] = 2.0 * (x[7] - 0.5)
            J[2, 14] = 1e-12 + 2.0 * x[14]
            for k, idx in enumerate(range(18, 24)):
                J[3 + k, idx] = 1.0
            return J

        ser_jac_points = []

        def ser_jac(x):
            ser_jac_points.append(x.copy())
            return full_jac(x)

        ari_ser = _ari.UndistributedArraysInterface(nE, nP)
        xf_ser, conv_ser, _, *_ = lm.simplish_leastsq(
            full_obj, ser_jac, x0.copy(), f_norm2_tol=1e-10, max_iter=100, arrays_interface=ari_ser,
            x_limits=x_limits, linesearch={'mode': 'none'},
        )
        assert conv_ser

        ari = _ari.DistributedArraysInterface(layout, 'normal', 0)
        f_buf = layout.allocate_local_array('e', 'd')
        jac_buf = ari.allocate_jac()
        jac_points = []

        def dist_obj(x, oob_check=False):
            if layout.resource_alloc('atom-processing').is_host_leader:
                f_buf[:] = full_obj(x)[layout.global_element_slice]
            layout.resource_alloc('atom-processing').host_comm_barrier()
            return f_buf

        def dist_jac(x):
            jac_points.append(x.copy())
            if layout.resource_alloc('param-processing').is_host_leader:
                jac_buf[:, :] = full_jac(x)[layout.global_element_slice, layout.global_param_slice]
            layout.resource_alloc('atom-processing').host_comm_barrier()
            return jac_buf

        xf_mpi, conv_mpi, _, *_ = lm.simplish_leastsq(
            dist_obj, dist_jac, x0.copy(), f_norm2_tol=1e-10, max_iter=100, resource_alloc=self.ralloc,
            arrays_interface=ari, x_limits=x_limits, linesearch={'mode': 'none'},
        )
        wcomm.barrier()
        layout.free_local_array(f_buf)
        ari.deallocate_jac(jac_buf)

        assert conv_mpi
        np.testing.assert_array_equal(jac_points[0], ser_jac_points[0])
        np.testing.assert_array_equal(jac_points[1], ser_jac_points[1])
        expected_shift_0 = 0.5 * lm.JAC_SHIFT if x_limits_clip else lm.JAC_SHIFT
        assert jac_points[1][0] == expected_shift_0
        assert jac_points[1][7] == 0.5 + lm.JAC_SHIFT
        assert jac_points[1][14] == lm.JAC_SHIFT
        assert np.all(jac_points[1][18:24] == 0.0)
        np.testing.assert_allclose(xf_mpi, xf_ser, atol=1e-12, rtol=1e-12)
        if not x_limits_clip:
            np.testing.assert_allclose(xf_mpi[0], 0.2, atol=1e-3)
            np.testing.assert_allclose(xf_mpi[7], 0.8, atol=1e-3)
            np.testing.assert_allclose(xf_mpi[14], 0.1, atol=1e-3)
            np.testing.assert_allclose(xf_mpi[18:24], 1.0, atol=1e-6)
        untouched = [i for i in range(18) if i not in (0, 7, 14)]
        assert np.all(xf_mpi[untouched] == 0.0)

    def run_nondegenerate_simplish_leastsq(self, natoms):
        layout = self._create_dist_layout(natoms)
        nE = layout.global_num_elements
        nP = layout.global_num_params

        x0 = np.zeros(nP, 'd')
        x0[0] = 0.5

        def full_obj(x, oob_check=False):
            r = np.zeros(nE, 'd')
            r[0] = x[0] ** 2 - 0.04
            r[1] = x[1] - 1.0
            for j in range(2, nP):
                r[j] = x[j]
            return r

        def full_jac(x):
            J = np.zeros((nE, nP), 'd')
            J[0, 0] = 2.0 * x[0]
            for j in range(1, nP):
                J[j, j] = 1.0
            return J

        ari = _ari.DistributedArraysInterface(layout, 'normal', 0)
        f_buf = layout.allocate_local_array('e', 'd')
        jac_buf = ari.allocate_jac()
        jac_points = []

        def dist_obj(x, oob_check=False):
            if layout.resource_alloc('atom-processing').is_host_leader:
                f_buf[:] = full_obj(x)[layout.global_element_slice]
            layout.resource_alloc('atom-processing').host_comm_barrier()
            return f_buf

        def dist_jac(x):
            jac_points.append(x.copy())
            if layout.resource_alloc('param-processing').is_host_leader:
                jac_buf[:, :] = full_jac(x)[layout.global_element_slice, layout.global_param_slice]
            layout.resource_alloc('atom-processing').host_comm_barrier()
            return jac_buf

        xf_mpi, conv_mpi, _, *_ = lm.simplish_leastsq(
            dist_obj, dist_jac, x0.copy(), max_iter=100, resource_alloc=self.ralloc,
            arrays_interface=ari, linesearch={'mode': 'none'},
        )
        wcomm.barrier()
        layout.free_local_array(f_buf)
        ari.deallocate_jac(jac_buf)

        assert conv_mpi
        np.testing.assert_allclose(xf_mpi[0], 0.2, atol=1e-3)
        np.testing.assert_allclose(xf_mpi[1], 1.0, atol=1e-6)
        assert all(np.all(p[2:] == 0.0) for p in jac_points)

    @classmethod
    def _cptplnd_serial_reference(cls):
        if not hasattr(cls, '_cached_cptplnd_ref'):
            truth = smq1Q_XY.target_model().depolarize(op_noise=0.01, spam_noise=0.01).rotate((0.01, 0, 0))
            edesign = smq1Q_XY.create_gst_experiment_design(1)
            circuits = edesign.all_circuits_needing_data
            ds = pygsti.data.simulate_data(truth, circuits, 1000, seed=2026, comm=wcomm)
            builder = pygsti.objectivefns.ObjectiveFunctionBuilder.create_from('chi2')
            builder.additional_args['array_types'] = ('E', 'EP')
            printer = pygsti.baseobjs.VerbosityPrinter(0, warnings=False)
            profiler = pygsti.baseobjs.Profiler(wcomm, False)

            mdl_ser1 = smq1Q_XY.target_model('CPTPLND')
            mdl_ser1.sim = pygsti.forwardsims.MatrixForwardSimulator(num_atoms=1)
            obj_ser1 = builder.build(mdl_ser1, ds, circuits, None, verbosity=0)
            res_ser1 = lm.SimplerLMOptimizer(maxiter=1).run(obj_ser1, profiler, printer)
            f_ser1 = float(res_ser1.f)
            vec_ser1 = mdl_ser1.to_vector().copy()
            del res_ser1, obj_ser1

            mdl_ser = smq1Q_XY.target_model('CPTPLND')
            mdl_ser.sim = pygsti.forwardsims.MatrixForwardSimulator(num_atoms=1)
            obj_ser = builder.build(mdl_ser, ds, circuits, None, verbosity=0)
            init_f = float(np.dot(obj_ser.lsvec(), obj_ser.lsvec()))
            opt_ser = lm.SimplerLMOptimizer(maxiter=100)
            res_ser = opt_ser.run(obj_ser, profiler, printer)
            assert res_ser.f < 0.1 * init_f
            cls._cached_cptplnd_ref = (
                circuits, ds, builder, init_f, f_ser1, vec_ser1,
                float(res_ser.f), mdl_ser.sim.bulk_probs(circuits),
            )
            del res_ser, obj_ser
        return cls._cached_cptplnd_ref

    def run_degenerate_cptplnd_fit(self, natoms):
        circuits, ds, builder, init_f, f_ser1, vec_ser1, f_ser, probs_ser = self._cptplnd_serial_reference()
        printer = pygsti.baseobjs.VerbosityPrinter(0, warnings=False)
        profiler = pygsti.baseobjs.Profiler(wcomm, False)

        mdl_mpi1 = smq1Q_XY.target_model('CPTPLND')
        mdl_mpi1.sim = pygsti.forwardsims.MatrixForwardSimulator(num_atoms=natoms)
        obj_mpi1 = builder.build(mdl_mpi1, ds, circuits, self.ralloc, verbosity=0)
        res_mpi1 = lm.SimplerLMOptimizer(maxiter=1).run(obj_mpi1, profiler, printer)
        assert abs(res_mpi1.f - f_ser1) < 1e-6, f"natoms={natoms}: {res_mpi1.f} vs {f_ser1}"
        np.testing.assert_allclose(mdl_mpi1.to_vector(), vec_ser1, atol=1e-12)
        wcomm.barrier()
        del res_mpi1, obj_mpi1

        mdl_mpi = smq1Q_XY.target_model('CPTPLND')
        mdl_mpi.sim = pygsti.forwardsims.MatrixForwardSimulator(num_atoms=natoms)
        obj_mpi = builder.build(mdl_mpi, ds, circuits, self.ralloc, verbosity=0)
        opt_mpi = lm.SimplerLMOptimizer(maxiter=100)
        res_mpi = opt_mpi.run(obj_mpi, profiler, printer)

        assert res_mpi.f < 0.1 * init_f
        assert abs(res_mpi.f - f_ser) < 1e-4, f"natoms={natoms}: {res_mpi.f} vs {f_ser}"
        v_mpi = mdl_mpi.to_vector()
        v_root = wcomm.bcast(v_mpi if wcomm.rank == 0 else None, root=0)
        np.testing.assert_allclose(v_mpi, v_root, atol=1e-14)
        probs_mpi = mdl_mpi.sim.bulk_probs(circuits)
        for c in circuits:
            for outcome, p_ser in probs_ser[c].items():
                assert abs(probs_mpi[c][outcome] - p_ser) < 1e-6
        wcomm.barrier()
        del res_mpi, obj_mpi
    #
    #
    #    #Note: no need to test "wrtFilter" business - that was removed
    #
    #
    ## There are other tests within testmpiMain.py that don't need much/any alteration and would be
    ## good to transfer to the new MPI unit tests, but:
    ## - test_MPI_mlgst_forcefn(comm) -- this is never used anymore, you can ignore this test
    ## - test_MPI_derivcols(comm) -- this is essentially tested by varying the layout types


class PureMPIParallel_Test(ParallelTest):
    @classmethod
    def setup_class(cls):
        # Turn off all shared memory usage
        os.environ['PYGSTI_USE_SHARED_MEMORY'] = "0"
        cls.ralloc = pygsti.baseobjs.ResourceAllocation(wcomm)

#class OnePerHostShmemParallel_Test(ParallelTest):
#    @classmethod
#    def setup_class(cls):
#        # Use 1 host per shared memory group (i.e. no shared mem communication)
#        os.environ['PYGSTI_MAX_HOST_PROCS'] = "1"
#        cls.ralloc = pygsti.baseobjs.ResourceAllocation(wcomm)
#
#class TwoPerHostShmemParallel_Test(ParallelTest):
#    @classmethod
#    def setup_class(cls):
#        # Use 2 hosts per shared memory group (i.e. mixed MPI + shared mem if more than 2 procs)
#        os.environ['PYGSTI_MAX_HOST_PROCS'] = "2"
#        cls.ralloc = pygsti.baseobjs.ResourceAllocation(wcomm)
#
#class AllShmemParallel_Test(ParallelTest):
#    @classmethod
#    def setup_class(cls):
#        # Set as many procs per host as possible to use shared memory
#        os.environ['PYGSTI_MAX_HOST_PROCS'] = str(wcomm.size)
#        cls.ralloc = pygsti.baseobjs.ResourceAllocation(wcomm)


if __name__ == '__main__':
    tester = PureMPIParallel_Test()
    tester.setup_class()
    tester.ralloc = pygsti.baseobjs.ResourceAllocation(wcomm)
    tester.run_optimizer_error_payloads()
    tester.run_objfn_values('matrix','logl',4)
    tester.run_fills('map', 1, None)
    tester.run_fills('map', 4, None)
    tester.run_fills('matrix', 4, 15)

    for build_hostcomms, max_host_procs in ((False, None), (True, str(wcomm.size)), (True, "2")):
        if max_host_procs is None:
            os.environ.pop('PYGSTI_MAX_HOST_PROCS', None)
        else:
            os.environ['PYGSTI_MAX_HOST_PROCS'] = max_host_procs
        tester.ralloc = pygsti.baseobjs.ResourceAllocation(wcomm)
        if build_hostcomms:
            tester.ralloc.build_hostcomms()
        for natoms in (1, 2, 4):
            tester.run_degenerate_simplish_leastsq(natoms)
            tester.run_degenerate_cptplnd_fit(natoms)
        tester.run_degenerate_simplish_leastsq(2, x_limits_clip=True)
        tester.run_nondegenerate_simplish_leastsq(2)
