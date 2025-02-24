
# This file is designed to be run via: mpiexec -np 4 python -W ignore run_me_with_mpiexec.py
# This does not use nosetests because I want to set verbosity differently based on rank (quiet if not rank 0)
# By wrapping asserts in comm.rank == 0, only rank 0 should fail (should help with output)
# Can run with different number of procs, but 4 is minimum to test all modes (pure MPI, pure shared mem, and mixed)

import os

import numpy as np
from mpi4py import MPI

import pygsti
from pygsti.modelpacks import smq1Q_XYI as std

pygsti.optimize.customsolve.CUSTOM_SOLVE_THRESHOLD = 10
wcomm = MPI.COMM_WORLD
print(f'Running with CUSTOM_SOLVE_THRESHOLD = {pygsti.optimize.customsolve.CUSTOM_SOLVE_THRESHOLD}')


class ParallelTest(object):
    # No setup here, must be defined in the derived classes

    def test_simulate_data(self):
        comm = self.ralloc.comm

        exp_design = std.get_gst_experiment_design(4)
        mdl_datagen = std.target_model().depolarize(op_noise=0.1, spam_noise=0.01)

        ds_serial = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234, comm=None)
        ds_parallel = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234, comm=comm)

        if comm is None or comm.rank == 0:
            assert (set(ds_serial.keys()) == set(ds_parallel.keys()))
            for key in ds_serial.keys():
                assert (ds_serial[key].to_dict() == ds_parallel[key].to_dict())

    def test_gst(self):
        comm = self.ralloc.comm

        exp_design = std.get_gst_experiment_design(4)
        mdl_datagen = std.target_model().depolarize(op_noise=0.1, spam_noise=0.01)
        ds = pygsti.data.simulate_data(mdl_datagen, exp_design, 1000, seed=1234, comm=comm)
        data = pygsti.protocols.ProtocolData(exp_design, ds)

        initial_model = std.target_model("full TP")
        proto = pygsti.protocols.GateSetTomography(initial_model, verbosity=1,
                                                   optimizer={'maxiter': 100, 'serial_solve_proc_threshold': 100})

        results_serial = proto.run(data, comm=None)            
        results_parallel = proto.run(data, comm=comm)
        
        # compare resulting models
        if comm is None or comm.rank == 0:
            best_params_serial = results_serial.estimates["GateSetTomography"].models['stdgaugeopt'].to_vector()
            best_params_parallel = results_parallel.estimates["GateSetTomography"].models['stdgaugeopt'].to_vector()

            assert np.allclose(best_params_serial, best_params_parallel)

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
        exp_design = std.get_gst_experiment_design(1)
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
    tester.run_objfn_values('matrix','logl',4)
    tester.run_fills('map', 1, None)
    tester.run_fills('map', 4, None)
    tester.run_fills('matrix', 4, 15)
