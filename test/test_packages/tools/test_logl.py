import os
import psutil
import pytest
import sys

import pygsti
from pygsti.modelpacks import smq1Q_XY
import pygsti.protocols as proto
from ..testutils import BaseTestCase, compare_files


class LogLTestCase(BaseTestCase):
    def test_memory(self):

        model = smq1Q_XY.target_model()
        model = model.depolarize(spam_noise = .01, op_noise = .001)
        model = model.rotate(max_rotate=.005, seed=1234)

        prep_fiducials = smq1Q_XY.prep_fiducials()
        meas_fiducials = smq1Q_XY.meas_fiducials()
        germs = smq1Q_XY.germs()
        op_labels = list(model.operations.keys()) # also == std.gates
        maxLengthList = [1]
        #circuits for XY model.
        gss = pygsti.circuits.make_lsgst_structs(op_labels, prep_fiducials[0:4], 
                                                          meas_fiducials[0:3], smq1Q_XY.germs(), maxLengthList)

        edesign =  proto.CircuitListsDesign([pygsti.circuits.CircuitList(circuit_struct) for circuit_struct in gss])

        ds = pygsti.data.simulate_data(model, edesign.all_circuits_needing_data, 1000, seed = 1234)

        def musage(prefix):
            p = psutil.Process(os.getpid())
            print(prefix, p.memory_info()[0])
        current_mem = pygsti.baseobjs.profiler._get_mem_usage

        musage("Pt1")
        
        with self.assertRaises(MemoryError):
            pygsti.logl_hessian(model, ds,
                                prob_clip_interval=(-1e6,1e6),
                                poisson_picture=True, mem_limit=0) # No memory for you

        musage("Pt2")
        L = pygsti.logl_hessian(model, ds, prob_clip_interval=(-1e6, 1e6),
                                poisson_picture=True, mem_limit=None, verbosity=10) # Reference: no mem limit
        musage("Pt3")
        L1 = pygsti.logl_hessian(model, ds, prob_clip_interval=(-1e6, 1e6),
                                 poisson_picture=True, mem_limit=1024.0**3, verbosity=10) # Limit memory (1GB)
        musage("Pt4")
        #L2 = pygsti.logl_hessian(model, ds,prob_clip_interval=(-1e6,1e6),
        #                         poisson_picture=True, mem_limit=current_mem()+1000000, verbosity=10) # Limit memory a bit more
        #musage("Pt5")
        #L3 = pygsti.logl_hessian(model, ds, prob_clip_interval=(-1e6,1e6),
        #                         poisson_picture=True, mem_limit=current_mem()+300000, verbosity=10) # Very low memory (splits tree)

        #We've currently disabled memory errors - re-enable this after we get memory checks working again.
        #with self.assertRaises(MemoryError):
        pygsti.logl_hessian(model, ds,
                            prob_clip_interval=(-1e6,1e6),
                            poisson_picture=True, mem_limit=1024.0**3)


        #print("****DEBUG LOGL HESSIAN L****")
        #print("shape = ",L.shape)
        #to_check = L
        #for i in range(L.shape[0]):
        #    for j in range(L.shape[1]):
        #        diff = abs(L3[i,j]-L[i,j])
        #        if diff > 1e-6:
        #            print("[%d,%d] diff = %g - %g = %g" % (i,j,L3[i,j],L[i,j],L3[i,j]-L[i,j]))
        self.assertArraysAlmostEqual(L, L1)
        #self.assertArraysAlmostEqual(L, L2, places=6) # roundoff?)
        #self.assertArraysAlmostEqual(L, L3, places=6) # roundoff?

    def test_hessian_mpi(self):
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            current_mem = pygsti.baseobjs.profiler._get_mem_usage
            ds   = pygsti.data.DataSet(file_to_load_from=compare_files + "/analysis.dataset")
            model = pygsti.io.load_model(compare_files + "/analysis.model")
            L = pygsti.logl_hessian(model, ds,
                                    prob_clip_interval=(-1e6,1e6), mem_limit=500*1024**2+current_mem(),
                                    poisson_picture=True, comm=comm)

            print(L)
        except (ImportError, RuntimeError):
            self.skipTest('Skipping because failed to import MPI')
