from ..testutils import BaseTestCase, compare_files, temp_files
from pygsti.construction import std1Q_XYI as std
import pygsti


class LogLTestCase(BaseTestCase):

    def test_logl_fn(self):
        ds          = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset%s" % self.versionsuffix)
        gatestrings = pygsti.construction.gatestring_list( [ ('Gx',), ('Gy',), ('Gx','Gx') ] )
        #OLD spam_labels = std.gs_target.get_spam_labels()
        #OLD pygsti.create_count_vec_dict( spam_labels, ds, gatestrings )
        gateset = pygsti.io.load_gateset(compare_files + "/analysis.gateset")

        L1 = pygsti.logl(gateset, ds, gatestrings,
                         probClipInterval=(-1e6,1e6),
                         poissonPicture=True, check=False)
        L2 = pygsti.logl(gateset, ds, gatestrings,
                         probClipInterval=(-1e6,1e6),
                         poissonPicture=False, check=False) #Non-poisson-picture

        dL1 = pygsti.logl_jacobian(gateset, ds, gatestrings,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=True, check=False)
        dL2 = pygsti.logl_jacobian(gateset, ds, gatestrings,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=False, check=False)
        dL2b = pygsti.logl_jacobian(gateset, ds, None,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=False, check=False) #test None as gs list


        hL1 = pygsti.logl_hessian(gateset, ds, gatestrings,
                                  probClipInterval=(-1e6,1e6), radius=1e-4,
                                  poissonPicture=True, check=False)

        hL2 = pygsti.logl_hessian(gateset, ds, gatestrings,
                                  probClipInterval=(-1e6,1e6), radius=1e-4,
                                  poissonPicture=False, check=False)
        hL2b = pygsti.logl_hessian(gateset, ds, None,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=False, check=False) #test None as gs list


        maxL1 = pygsti.logl_max(gateset, ds, gatestrings, poissonPicture=True, check=True)
        maxL2 = pygsti.logl_max(gateset, ds, gatestrings, poissonPicture=False, check=True)

        pygsti.cptp_penalty(gateset, include_spam_penalty=True)
        twoDelta1 = pygsti.two_delta_loglfn(N=100, p=0.5, f=0.6, minProbClip=1e-6, poissonPicture=True)
        twoDelta2 = pygsti.two_delta_loglfn(N=100, p=0.5, f=0.6, minProbClip=1e-6, poissonPicture=False)

    def test_no_gatestrings(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset%s" % self.versionsuffix)
        gateset = std.gs_target #could use pygsti.io.load_gateset(compare_files + "/analysis.gateset"), but then change hardcoded #'s
        L1 = pygsti.logl(gateset, ds,
                         probClipInterval=(-1e6,1e6),
                         poissonPicture=True, check=False)
        self.assertAlmostEqual(L1,-21393568.52986, 2)
        #self.assertAlmostEqual(L1,-21579292.1837, 2) #OLD2
        #self.assertAlmostEqual(L1, -4531934.43735, 2) #OLD
        
        L2 = pygsti.logl_max(gateset, ds)
        
        self.assertAlmostEqual(L2, -14028782.1039, 2)
        #self.assertAlmostEqual(L2, -13402461.9294, 2) #OLD2
        #self.assertAlmostEqual(L2, -1329179.7675, 5) #OLD

    def test_memory(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset%s" % self.versionsuffix)
        gateset = pygsti.io.load_gateset(compare_files + "/analysis.gateset")

        with self.assertRaises(MemoryError):
            pygsti.logl_hessian(gateset, ds,
                                probClipInterval=(-1e6,1e6),
                                poissonPicture=True, check=False, memLimit=0) # No memory for you

        L = pygsti.logl_hessian(gateset, ds, probClipInterval=(-1e6,1e6),
                                poissonPicture=True, check=False, memLimit=None, verbosity=10) # Reference: no mem limit
        L1 = pygsti.logl_hessian(gateset, ds, probClipInterval=(-1e6,1e6),
                                 poissonPicture=True, check=False, memLimit=370000000, verbosity=10) # Limit memory a bit
        L2 = pygsti.logl_hessian(gateset, ds,probClipInterval=(-1e6,1e6),
                                 poissonPicture=True, check=False, memLimit=1000000, verbosity=10) # Limit memory a bit more
        L3 = pygsti.logl_hessian(gateset, ds, probClipInterval=(-1e6,1e6),
                                 poissonPicture=True, check=False, memLimit=300000, verbosity=10) # Very low memory (splits tree)
        
        with self.assertRaises(MemoryError):
            pygsti.logl_hessian(gateset, ds,
                                probClipInterval=(-1e6,1e6),
                                poissonPicture=True, check=False, memLimit=70000) # Splitting unproductive


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

    def test_forbidden_probablity(self):
        ds   = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset%s" % self.versionsuffix)
        prob = pygsti.forbidden_prob(std.gs_target, ds)
        self.assertAlmostEqual(prob, 1.276825378318927e-13)

    def test_hessian_mpi(self):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        ds   = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset%s" % self.versionsuffix)
        gateset = pygsti.io.load_gateset(compare_files + "/analysis.gateset")
        L = pygsti.logl_hessian(gateset, ds,
                                probClipInterval=(-1e6,1e6), memLimit=25000000,
                                poissonPicture=True, check=False, comm=comm)

        print(L)
