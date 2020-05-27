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
        self.circuits = construction.to_circuits([('Gx',), ('Gy',), ('Gx', 'Gx')])

    def test_logl(self):
        L1 = lfn.logl(self.model, self.ds, self.circuits,
                      prob_clip_interval=(-1e6, 1e6),
                      poisson_picture=True)
        # Non-poisson-picture
        L2 = lfn.logl(self.model, self.ds, self.circuits,
                      prob_clip_interval=(-1e6, 1e6),
                      poisson_picture=False)
        # TODO assert correctness

    def test_logl_jacobian(self):
        dL1 = lfn.logl_jacobian(self.model, self.ds, self.circuits,
                                prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                                poisson_picture=True)
        dL2 = lfn.logl_jacobian(self.model, self.ds, self.circuits,
                                prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                                poisson_picture=False)
        # test None as mdl list
        dL2b = lfn.logl_jacobian(self.model, self.ds, None,
                                 prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                                 poisson_picture=False)
        # TODO assert correctness

    def test_logl_hessian(self):
        # TODO optimize
        hL1 = lfn.logl_hessian(self.model, self.ds, self.circuits,
                               prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                               poisson_picture=True)

        hL2 = lfn.logl_hessian(self.model, self.ds, self.circuits,
                               prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                               poisson_picture=False)
        # test None as mdl list
        hL2b = lfn.logl_hessian(self.model, self.ds, None,
                                prob_clip_interval=(-1e6, 1e6), radius=1e-4,
                                poisson_picture=False)
        # TODO assert correctness

    def test_logl_max(self):
        maxL1 = lfn.logl_max(self.model, self.ds, self.circuits, poisson_picture=True)
        maxL2 = lfn.logl_max(self.model, self.ds, self.circuits, poisson_picture=False)
        # TODO assert correctness

    def test_cptp_penalty(self):
        lfn.cptp_penalty(self.model, include_spam_penalty=True)
        # TODO assert correctness

    def test_two_delta_logl(self):
        twoDelta1 = lfn.two_delta_logl_term(n=100, p=0.5, f=0.6, min_prob_clip=1e-6, poisson_picture=True)
        twoDelta2 = lfn.two_delta_logl_term(n=100, p=0.5, f=0.6, min_prob_clip=1e-6, poisson_picture=False)
        # TODO assert correctness

    def test_no_gatestrings(self):
        # TODO what edge case does this cover?
        model = std.target_model()
        L1 = lfn.logl(model, self.ds,
                      prob_clip_interval=(-1e6, 1e6),
                      poisson_picture=True)
        self.assertAlmostEqual(L1, -21393568.52986, 2)

        L2 = lfn.logl_max(model, self.ds)
        self.assertAlmostEqual(L2, -14028782.1039, 2)
