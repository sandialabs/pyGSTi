# XXX rewrite and optimize
import pygsti.algorithms as alg
import pygsti.algorithms.gaugeopt as go
from pygsti.models.gaugegroup import TPGaugeGroup
from . import fixtures
from ..util import BaseCase


class GaugeOptMethodBase(object):
    def setUp(self):
        super(GaugeOptMethodBase, self).setUp()
        self.options = dict(
            verbosity=10,
            check_jac=True
        )

    def test_gaugeopt(self):
        go_result = go.gaugeopt_to_target(self.model, self.target, **self.options)
        # TODO assert correctness

    # def test_gaugeopt_no_target(self):
    #     go_result = go.gaugeopt_to_target(self.model, None, **self.options)
    #     # TODO assert correctness


class GaugeOptMetricMethods(GaugeOptMethodBase):
    def test_gaugeopt_gates_metrics(self):
        go_result = go.gaugeopt_to_target(
            self.model, self.target, gates_metric='fidelity', **self.options
        )
        # TODO assert correctness
        go_result = go.gaugeopt_to_target(
            self.model, self.target, gates_metric='tracedist', **self.options
        )
        # TODO assert correctness

    def test_gaugeopt_spam_metrics(self):
        go_result = go.gaugeopt_to_target(
            self.model, self.target, spam_metric='fidelity', **self.options
        )
        # TODO assert correctness
        go_result = go.gaugeopt_to_target(
            self.model, self.target, spam_metric='tracedist', **self.options
        )
        # TODO assert correctness

    def test_gaugeopt_raises_on_invalid_metrics(self):
        with self.assertRaises(ValueError):
            go.gaugeopt_to_target(
                self.model, self.target, spam_metric='foobar', **self.options
            )
        with self.assertRaises(ValueError):
            go.gaugeopt_to_target(
                self.model, self.target, gates_metric='foobar', **self.options
            )


class GaugeOptInstanceBase(object):
    def setUp(self):
        super(GaugeOptInstanceBase, self).setUp()
        self.target = fixtures.model
        self.model = self._model.copy()


class GaugeOptWithGaugeGroupInstance(GaugeOptInstanceBase):
    @classmethod
    def setUpClass(cls):
        super(GaugeOptWithGaugeGroupInstance, cls).setUpClass()
        cls.gauge_group = TPGaugeGroup(fixtures.mdl_lgst.state_space)

    def test_gaugeopt_with_gauge_group(self):
        go_result = go.gaugeopt_to_target(
            self.model, self.target,
            gauge_group=self.gauge_group,
            **self.options
        )
        # TODO assert correctness


class GaugeOptNoTargetInstance(GaugeOptInstanceBase):
    def setUp(self):
        super(GaugeOptNoTargetInstance, self).setUp()
        self.target = None


class LGSTGaugeOptInstance(GaugeOptWithGaugeGroupInstance):
    @classmethod
    def setUpClass(cls):
        super(LGSTGaugeOptInstance, cls).setUpClass()
        # cls._model = alg.run_lgst(
        #     fixtures.ds, fixtures.fiducials, fixtures.fiducials, fixtures.model,
        #     svd_truncate_to=4, verbosity=0
        # )

        # TODO construct directly
        mdl_lgst_target = go.gaugeopt_to_target(fixtures.mdl_lgst, fixtures.model, check_jac=True)
        cls._model = mdl_lgst_target


# class LGSTGaugeOptTester(GaugeOptMethodBase, LGSTGaugeOptInstance, BaseCase):
#     pass


class LGSTGaugeOptAutoMethodTester(GaugeOptMetricMethods, LGSTGaugeOptInstance, BaseCase):
    def setUp(self):
        super(LGSTGaugeOptAutoMethodTester, self).setUp()
        self.options.update(
            method='auto'
        )

    def test_gaugeopt_return_all(self):
        # XXX does this need to be tested independently of everything else? EGN: probably not - better pattern for this?
        soln, trivialEl, mdl = go.gaugeopt_to_target(self.model, self.target, return_all=True, **self.options)
        # TODO assert correctness


class LGSTGaugeOptBFGSMethodTester(GaugeOptMethodBase, LGSTGaugeOptInstance, BaseCase):
    def setUp(self):
        super(LGSTGaugeOptBFGSMethodTester, self).setUp()
        self.options.update(
            method='BFGS'
        )


class LGSTGaugeOptLSMethodTester(GaugeOptMethodBase, LGSTGaugeOptInstance, BaseCase):
    def setUp(self):
        super(LGSTGaugeOptLSMethodTester, self).setUp()
        self.options.update(
            method='ls'
        )

    def test_ls_gaugeopt_raises_on_bad_metrics(self):
        with self.assertRaises(ValueError):
            go.gaugeopt_to_target(
                self.model, self.target, spam_metric='tracedist', **self.options
            )
        with self.assertRaises(ValueError):
            go.gaugeopt_to_target(
                self.model, self.target, spam_metric='fidelity', **self.options
            )
        with self.assertRaises(ValueError):
            go.gaugeopt_to_target(
                self.model, self.target, gates_metric='tracedist', **self.options
            )
        with self.assertRaises(ValueError):
            go.gaugeopt_to_target(
                self.model, self.target, gates_metric='fidelity', **self.options
            )

    # def test_gaugeopt_no_target(self):
    #     with self.assertRaises(ValueError):
    #         go.gaugeopt_to_target(self.model, None, **self.options)


class CPTPGaugeOptTester(GaugeOptMethodBase, GaugeOptWithGaugeGroupInstance, BaseCase):
    @classmethod
    def setUpClass(cls):
        super(CPTPGaugeOptTester, cls).setUpClass()
        # TODO construct directly
        mdl_lgst_target = go.gaugeopt_to_target(fixtures.mdl_lgst, fixtures.model, check_jac=True)
        mdl_clgst_cptp = alg.contract(mdl_lgst_target, "CPTP", verbosity=10, tol=10.0)
        cls._model = mdl_clgst_cptp


class CPTPGaugeOptCPTPPenaltyTester(CPTPGaugeOptTester):
    def setUp(self):
        super(CPTPGaugeOptCPTPPenaltyTester, self).setUp()
        self.options.update(
            cptp_penalty_factor=1.0
        )


class CPTPGaugeOptSPAMPenaltyTester(CPTPGaugeOptTester):
    def setUp(self):
        super(CPTPGaugeOptSPAMPenaltyTester, self).setUp()
        self.options.update(
            spam_penalty_factor=1.0
        )


class CPTPGaugeOptAllPenaltyTester(CPTPGaugeOptCPTPPenaltyTester, CPTPGaugeOptSPAMPenaltyTester):
    pass


class LGSTGaugeOptPenaltyBase(GaugeOptMethodBase, LGSTGaugeOptInstance):
    def test_gaugeopt_no_target(self):
        go_result = go.gaugeopt_to_target(self.model, None, **self.options)
        # TODO assert correctness

    def test_gaugeopt_with_none_gauge_group(self):
        go_result = go.gaugeopt_to_target(
            self.model, self.target,
            gauge_group=None,
            **self.options
        )


class LGSTGaugeOptCPTPPenaltyTester(LGSTGaugeOptPenaltyBase, BaseCase):
    def setUp(self):
        super(LGSTGaugeOptCPTPPenaltyTester, self).setUp()
        self.options.update(
            cptp_penalty_factor=1.0
        )


class LGSTGaugeOptSPAMPenaltyTester(LGSTGaugeOptPenaltyBase, BaseCase):
    def setUp(self):
        super(LGSTGaugeOptSPAMPenaltyTester, self).setUp()
        self.options.update(
            spam_penalty_factor=1.0
        )


class LGSTGaugeOptAllPenaltyTester(LGSTGaugeOptCPTPPenaltyTester, LGSTGaugeOptSPAMPenaltyTester):
    pass
