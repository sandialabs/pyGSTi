from ..util import BaseCase
from . import fixtures as pkg

from pygsti.tools import likelihoodfns as lfn
from pygsti.objects.dataset import DataSet
from pygsti.modelpacks.legacy import std1Q_XYI as std
from pygsti import construction


class LikelihoodFunctionsBase(BaseCase):
    def setUp(self):
        self.ds = pkg.dataset.copy()
        self.model = pkg.mdl_lsgst_go.copy()


class LikelihoodFunctionsTester(LikelihoodFunctionsBase):
    def test_forbidden_probablity(self):
        prob = lfn.forbidden_prob(std.target_model(), self.ds)
        self.assertAlmostEqual(prob, 1.276825378318927e-13)


class LogLTester(LikelihoodFunctionsBase):
    def setUp(self):
        super(LogLTester, self).setUp()
        self.circuits = construction.circuit_list([('Gx',), ('Gy',), ('Gx', 'Gx')])

    def test_logl(self):
        L1 = lfn.logl(self.model, self.ds, self.circuits,
                      probClipInterval=(-1e6, 1e6),
                      poissonPicture=True, check=False)
        # Non-poisson-picture
        L2 = lfn.logl(self.model, self.ds, self.circuits,
                      probClipInterval=(-1e6, 1e6),
                      poissonPicture=False, check=False)
        # TODO assert correctness

    def test_logl_jacobian(self):
        dL1 = lfn.logl_jacobian(self.model, self.ds, self.circuits,
                                probClipInterval=(-1e6, 1e6), radius=1e-4,
                                poissonPicture=True, check=False)
        dL2 = lfn.logl_jacobian(self.model, self.ds, self.circuits,
                                probClipInterval=(-1e6, 1e6), radius=1e-4,
                                poissonPicture=False, check=False)
        # test None as mdl list
        dL2b = lfn.logl_jacobian(self.model, self.ds, None,
                                 probClipInterval=(-1e6, 1e6), radius=1e-4,
                                 poissonPicture=False, check=False)
        # TODO assert correctness

    def test_logl_hessian(self):
        # TODO optimize
        hL1 = lfn.logl_hessian(self.model, self.ds, self.circuits,
                               probClipInterval=(-1e6, 1e6), radius=1e-4,
                               poissonPicture=True, check=False)

        hL2 = lfn.logl_hessian(self.model, self.ds, self.circuits,
                               probClipInterval=(-1e6, 1e6), radius=1e-4,
                               poissonPicture=False, check=False)
        # test None as mdl list
        hL2b = lfn.logl_hessian(self.model, self.ds, None,
                                probClipInterval=(-1e6, 1e6), radius=1e-4,
                                poissonPicture=False, check=False)
        # TODO assert correctness

    def test_logl_max(self):
        maxL1 = lfn.logl_max(self.model, self.ds, self.circuits, poissonPicture=True, check=True)
        maxL2 = lfn.logl_max(self.model, self.ds, self.circuits, poissonPicture=False, check=True)
        # TODO assert correctness

    def test_cptp_penalty(self):
        lfn.cptp_penalty(self.model, include_spam_penalty=True)
        # TODO assert correctness

    def test_two_delta_logl(self):
        twoDelta1 = lfn.two_delta_loglfn(N=100, p=0.5, f=0.6, minProbClip=1e-6, poissonPicture=True)
        twoDelta2 = lfn.two_delta_loglfn(N=100, p=0.5, f=0.6, minProbClip=1e-6, poissonPicture=False)
        # TODO assert correctness

    def test_no_gatestrings(self):
        # TODO what edge case does this cover?
        model = std.target_model()
        L1 = lfn.logl(model, self.ds,
                      probClipInterval=(-1e6, 1e6),
                      poissonPicture=True, check=False)
        self.assertAlmostEqual(L1, -21393568.52986, 2)

        L2 = lfn.logl_max(model, self.ds)
        self.assertAlmostEqual(L2, -14028782.1039, 2)
