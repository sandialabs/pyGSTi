from ..testutils import BaseTestCase, compare_files, temp_files
from pygsti.modelpacks.legacy import std1Q_XYI as std
import pygsti


class LogLTestCase(BaseTestCase):
    def test_memory(self):
        ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/analysis.dataset")
        model = pygsti.io.load_model(compare_files + "/analysis.model")

        with self.assertRaises(MemoryError):
            pygsti.logl_hessian(model, ds,
                                prob_clip_interval=(-1e6,1e6),
                                poisson_picture=True, check=False, mem_limit=0) # No memory for you

        L = pygsti.logl_hessian(model, ds, prob_clip_interval=(-1e6,1e6),
                                poisson_picture=True, check=False, mem_limit=None, verbosity=10) # Reference: no mem limit
        L1 = pygsti.logl_hessian(model, ds, prob_clip_interval=(-1e6,1e6),
                                 poisson_picture=True, check=False, mem_limit=370000000, verbosity=10) # Limit memory a bit
        L2 = pygsti.logl_hessian(model, ds,prob_clip_interval=(-1e6,1e6),
                                 poisson_picture=True, check=False, mem_limit=1000000, verbosity=10) # Limit memory a bit more
        L3 = pygsti.logl_hessian(model, ds, prob_clip_interval=(-1e6,1e6),
                                 poisson_picture=True, check=False, mem_limit=300000, verbosity=10) # Very low memory (splits tree)

        with self.assertRaises(MemoryError):
            pygsti.logl_hessian(model, ds,
                                prob_clip_interval=(-1e6,1e6),
                                poisson_picture=True, check=False, mem_limit=70000) # Splitting unproductive


        #print("****DEBUG LOGL HESSIAN L****")
        #print("shape = ",L.shape)
        #to_check = L
        #for i in range(L.shape[0]):
        #    for j in range(L.shape[1]):
        #        diff = abs(L3[i,j]-L[i,j])
        #        if diff > 1e-6:
        #            print("[%d,%d] diff = %g - %g = %g" % (i,j,L3[i,j],L[i,j],L3[i,j]-L[i,j]))
        self.assertArraysAlmostEqual(L, L1)
        self.assertArraysAlmostEqual(L, L2)
        self.assertArraysAlmostEqual(L, L3, places=6) # roundoff?

    def test_hessian_mpi(self):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        ds   = pygsti.objects.DataSet(file_to_load_from=compare_files + "/analysis.dataset")
        model = pygsti.io.load_model(compare_files + "/analysis.model")
        L = pygsti.logl_hessian(model, ds,
                                prob_clip_interval=(-1e6,1e6), mem_limit=25000000,
                                poisson_picture=True, check=False, comm=comm)

        print(L)
