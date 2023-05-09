import os
import psutil

import pygsti
from ..testutils import BaseTestCase, compare_files


class LogLTestCase(BaseTestCase):
    def test_memory(self):

        def musage(prefix):
            p = psutil.Process(os.getpid())
            print(prefix, p.memory_info()[0])
        current_mem = pygsti.baseobjs.profiler._get_mem_usage

        musage("Initial")
        ds = pygsti.data.DataSet(file_to_load_from=compare_files + "/analysis.dataset")
        model = pygsti.io.load_model(compare_files + "/analysis.model")
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
        except ImportError:
            self.skipTest('Skipping because failed to import MPI')
