from ..testutils import BaseTestCase, compare_files, temp_files
from pygsti.construction import std1Q_XYI as std
import pygsti


class LogLTestCase(BaseTestCase):

    def test_logl_fn(self):
        ds          = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset")
        gatestrings = pygsti.construction.gatestring_list( [ ('Gx',), ('Gy',), ('Gx','Gx') ] )
        spam_labels = std.gs_target.get_spam_labels()
        pygsti.create_count_vec_dict( spam_labels, ds, gatestrings )

        L1 = pygsti.logl(std.gs_target, ds, gatestrings,
                         probClipInterval=(-1e6,1e6), countVecMx=None,
                         poissonPicture=True, check=False)
        L2 = pygsti.logl(std.gs_target, ds, gatestrings,
                         probClipInterval=(-1e6,1e6), countVecMx=None,
                         poissonPicture=False, check=False) #Non-poisson-picture

        dL1 = pygsti.logl_jacobian(std.gs_target, ds, gatestrings,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=True, check=False)
        dL2 = pygsti.logl_jacobian(std.gs_target, ds, gatestrings,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=False, check=False)
        dL2b = pygsti.logl_jacobian(std.gs_target, ds, None,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=False, check=False) #test None as gs list


        hL1 = pygsti.logl_hessian(std.gs_target, ds, gatestrings,
                                  probClipInterval=(-1e6,1e6), radius=1e-4,
                                  poissonPicture=True, check=False)

        hL2 = pygsti.logl_hessian(std.gs_target, ds, gatestrings,
                                  probClipInterval=(-1e6,1e6), radius=1e-4,
                                  poissonPicture=False, check=False)
        hL2b = pygsti.logl_hessian(std.gs_target, ds, None,
                                   probClipInterval=(-1e6,1e6), radius=1e-4,
                                   poissonPicture=False, check=False) #test None as gs list


        maxL1 = pygsti.logl_max(ds, gatestrings, poissonPicture=True, check=True)
        maxL2 = pygsti.logl_max(ds, gatestrings, poissonPicture=False, check=True)

        pygsti.cptp_penalty(std.gs_target, include_spam_penalty=True)
        twoDelta1 = pygsti.two_delta_loglfn(N=100, p=0.5, f=0.6, minProbClip=1e-6, poissonPicture=True)
        twoDelta2 = pygsti.two_delta_loglfn(N=100, p=0.5, f=0.6, minProbClip=1e-6, poissonPicture=False)

    def test_no_gatestrings(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset")
        L1 = pygsti.logl(std.gs_target, ds,
                         probClipInterval=(-1e6,1e6), countVecMx=None,
                         poissonPicture=True, check=False)
        self.assertAlmostEqual(L1, -4531934.43735, 2)
        L2 = pygsti.logl_max(ds)
        self.assertAlmostEqual(L2, -1329179.7675, 5)

    def test_memory(self):
        ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset")
        with self.assertRaises(MemoryError):
            pygsti.logl_hessian(std.gs_target, ds,
                                probClipInterval=(-1e6,1e6), countVecMx=None,
                                poissonPicture=True, check=False, memLimit=0) # No memory for you

        L = pygsti.logl_hessian(std.gs_target, ds,
                            probClipInterval=(-1e6,1e6), countVecMx=None,
                            poissonPicture=True, check=False, memLimit=370000000) # Limit memory a bit
        pygsti.logl_hessian(std.gs_target, ds,
                            probClipInterval=(-1e6,1e6), countVecMx=None,
                            poissonPicture=True, check=False, memLimit=25000000) # Limit memory a bit more
        with self.assertRaises(MemoryError):
            pygsti.logl_hessian(std.gs_target, ds,
                                probClipInterval=(-1e6,1e6), countVecMx=None,
                                poissonPicture=True, check=False, memLimit=30000) # Until another error is thrown

    def test_forbidden_probablity(self):
        ds   = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset")
        prob = pygsti.forbidden_prob(std.gs_target, ds)
        self.assertAlmostEqual(prob, 1.276825378318927e-13)

    def test_hessian_mpi(self):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        ds   = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/analysis.dataset")
        L = pygsti.logl_hessian(std.gs_target, ds,
                                probClipInterval=(-1e6,1e6), countVecMx=None, memLimit=25000000,
                                poissonPicture=True, check=False, comm=comm)

        print(L)
