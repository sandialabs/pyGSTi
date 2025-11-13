
from pygsti.models.gaugegroup import TPGaugeGroup, TPGaugeGroupElement, \
      UnitaryGaugeGroup, UnitaryGaugeGroupElement,\
      FullGaugeGroup, FullGaugeGroupElement
from ..util import BaseCase

import time
from collections import OrderedDict
import numpy as np
from pygsti.modelpacks import smq1Q_XYI
import scipy.linalg as la
import pygsti.tools.optools as pgo
import pygsti.baseobjs.label as pgl
import pygsti.baseobjs.basisconstructors as pgbc
import pygsti.baseobjs as pgb
import pygsti.algorithms.gaugeopt as gop
from pygsti.models import ExplicitOpModel
from pygsti.modelmembers.operations import FullArbitraryOp
import pytest



def gate_metrics_dict(model, target):
    metrics = {'infids': OrderedDict(), 'frodists': OrderedDict(), 'tracedists': OrderedDict()}
    for lbl in model.operations.keys():
        model_gate = model[lbl]
        target_gate = target[lbl]
        metrics['infids'][lbl] = pgo.entanglement_infidelity(model_gate, target_gate, model.basis)
        metrics['frodists'][lbl] = pgo.frobeniusdist(model_gate, target_gate)
        metrics['tracedists'][lbl] = pgo.tracedist(model_gate, target_gate)
    return metrics


def check_gate_metrics_are_nontrivial(metrics, tol):
    """
    "metrics" is a dict of the kind produced by gate_metrics_dict(model, target).

    This function checks if all values in that dict are strictly greater than tol,
    with the exception of those corresponding to the idle gate.

    This makes sure that gauge optimization for "model" matching "target" is a
    nontrivial problem.
    """
    idle_label = pgl.Label(tuple())
    for lbl, val in metrics['infids'].items():
        if lbl == idle_label:
            continue
        assert val > tol, f"{val} is at most {tol}, failure for gate {lbl} w.r.t. infidelity."
    for lbl, val in metrics['frodists'].items():
        if lbl == idle_label:
            continue
        assert val > tol, f"{val} is at most {tol}, failure for gate {lbl} w.r.t. frobenius distance."
    for lbl, val in metrics['tracedists'].items():
        if lbl == idle_label:
            continue
        assert val > tol, f"{val} is at most {tol}, failure for gate {lbl} w.r.r. trace distance."
    return


def check_gate_metrics_near_zero(metrics, tol):
    """
    "metrics" is a dict of the kind produced by gate_metrics_dict(model, target).
    
    This function should be called to check if gauge optimization for "model"
    matching "target" succeeded.
    """
    for lbl, val in metrics['infids'].items():
        assert val <= tol, f"{val} exceeds {tol}, failure for gate {lbl} w.r.t infidelity."
    for lbl, val in metrics['frodists'].items():
        assert val <= tol, f"{val} exceeds {tol}, failure for gate {lbl} w.r.t. frobenius distance."
    for lbl, val in metrics['tracedists'].items():
        assert val <= tol, f"{val} exceeds {tol}, failure for gate {lbl} w.r.t. trace distance."
    return


class FindPerfectGauge_sm1QXYI:

    @property
    def default_gauge_group(self):
        """
        Instance of an object that subclasses GaugeGroup.
        The state space for this gauge group is set based
        on that of "self.target".
        """
        return self._default_gauge_group
    
    @property 
    def gauge_grp_el_class(self):
        """
        Handle for some subclass of GaugeGroupElement.
        Must be compatible with self.default_gauge_group.
        """
        return self._gauge_grp_el_class
    
    @property
    def gauge_space_name(self) -> str:
        """
        Only used for printing purposes in "self._main_tester".
        """
        return self._gauge_space_name

    def setUp(self):
        # The end of this function raises an error! The function is here just to be
        # a template for usable implementations.
        np.random.seed(0)
        target = smq1Q_XYI.target_model()
        self.target = target.depolarize(op_noise=0.01, spam_noise=0.001)
        # ^ That's a noisy data-generating model. We'll gauge-optimize to this noisy
        #   data-generating model rather than to an ideal model.
        self.model = None
        # ^ We'll initialize that later on as something that's gauge-equivalent to target.
        self._default_gauge_group = None
        self._gauge_grp_el_class = None
        self._gauge_space_name = ''
        self.N_REPS = 1
        raise NotImplementedError()
    
    def _gauge_transform_model(self, seed):
        self.model = self.target.copy()
        self.model.default_gauge_group = self.default_gauge_group
        np.random.seed(seed)
        self.U = la.expm(np.random.randn()/2 * -1j * (pgbc.sigmax + pgbc.sigmaz)/np.sqrt(2))
        self.gauge_grp_el = self.gauge_grp_el_class(pgo.unitary_to_pauligate(self.U))
        self.model.transform_inplace(self.gauge_grp_el)
        self.metrics_before = gate_metrics_dict(self.model, self.target)
        check_gate_metrics_are_nontrivial(self.metrics_before, tol=1e-2)
        return

    def _main_tester(self, method, seed, test_tol, alg_tol, gop_objective):
        assert gop_objective in {'frobenius', 'frobenius_squared', 'tracedist', 'fidelity'}
        self._gauge_transform_model(seed)
        tic = time.time()
        newmodel = gop.gaugeopt_to_target(self.model, self.target, method=method, tol=alg_tol, spam_metric=gop_objective, gates_metric=gop_objective)
        toc = time.time()
        dt = toc - tic
        metrics_after = gate_metrics_dict(newmodel, self.target)
        check_gate_metrics_near_zero(metrics_after, test_tol)
        return dt
    
    def test_lbfgs_frodist(self):
        self.setUp()
        times = []
        for seed in range(self.N_REPS):
            dt = self._main_tester('L-BFGS-B', seed, test_tol=1e-6, alg_tol=1e-8, gop_objective='frobenius')
            times.append(dt)
        print(f'GaugeOpt over {self.gauge_space_name} w.r.t. Frobenius dist, using L-BFGS: {times}.')
        return
    
    def test_lbfgs_tracedist(self):
        self.setUp()
        times = []
        for seed in range(self.N_REPS):
            dt = self._main_tester('L-BFGS-B', seed, test_tol=1e-6, alg_tol=1e-8, gop_objective='tracedist')
            times.append(dt)
        print(f'GaugeOpt over {self.gauge_space_name} w.r.t. trace dist, using L-BFGS: {times}.')
        return

    def test_ls(self):
        self.setUp()
        times = []
        for seed in range(self.N_REPS):
            dt = self._main_tester('ls', seed, test_tol=1e-6, alg_tol=1e-15, gop_objective='frobenius')
            times.append(dt)
        print(f'GaugeOpt over {self.gauge_space_name} w.r.t. Frobenius dist, using LS    : {times}.')
        return


class FindPerfectGauge_TPGroup_Tester(BaseCase, FindPerfectGauge_sm1QXYI):
    """
    This class' test for minimizing trace distance with L-BFGS takes excessively long (about 5 seconds).
    """

    def setUp(self):
        np.random.seed(0)
        target = smq1Q_XYI.target_model()
        self.target = target.depolarize(op_noise=0.01, spam_noise=0.001)
        self.model = None
        self._default_gauge_group = TPGaugeGroup(self.target.state_space, self.target.basis)
        self._gauge_grp_el_class = TPGaugeGroupElement
        self._gauge_space_name = 'TPGaugeGroup'
        self.N_REPS = 1
        return


class FindPerfectGauge_UnitaryGroup_Tester(BaseCase, FindPerfectGauge_sm1QXYI):

    def setUp(self):
        np.random.seed(0)
        target = smq1Q_XYI.target_model()
        self.target = target.depolarize(op_noise=0.01, spam_noise=0.001)
        self.model = None
        self._default_gauge_group = UnitaryGaugeGroup(self.target.state_space, self.target.basis)
        self._gauge_grp_el_class = UnitaryGaugeGroupElement
        self._gauge_space_name = 'UnitaryGaugeGroup'
        self.N_REPS = 3
        # ^ This is fast, so we can afford more tests.
        return


class FindPerfectGauge_FullGroup_Tester(BaseCase, FindPerfectGauge_sm1QXYI):
    """
    This class' test for minimizing trace distance with L-BFGS takes excessively long (about 5 seconds).
    """

    def setUp(self):
        np.random.seed(0)
        target = smq1Q_XYI.target_model()
        self.target = target.depolarize(op_noise=0.01, spam_noise=0.001)
        self.model = None
        self._default_gauge_group = FullGaugeGroup(self.target.state_space, self.target.basis)
        self._gauge_grp_el_class = FullGaugeGroupElement
        self._gauge_space_name = 'FullGaugeGroup'
        self.N_REPS = 1
        return


class FindPerfectGauge_Instruments_Tester(BaseCase):

    def setUp(self, seed: int = 0):
        from pygsti.baseobjs import QubitSpace
        from pygsti.modelmembers.povms import TPPOVM
        from pygsti.modelmembers.instruments import Instrument
        from pygsti.tools.basistools import stdmx_to_ppvec
        from pygsti.tools.optools import unitary_to_superop

        ss = QubitSpace(1)
        target = ExplicitOpModel(ss, basis='pp')
        E0 = np.array([[1,0],[0,0]], dtype=np.cdouble)
        target.preps['rho0'] = stdmx_to_ppvec(E0)
        E1 = np.eye(2) - E0
        target.povms['Mdefault'] = TPPOVM({'0': stdmx_to_ppvec(E0), '1': stdmx_to_ppvec(E1)}, state_space=ss, evotype='default')

        rng = np.random.default_rng(seed)
        U = la.qr(rng.standard_normal((2,2)) + 1j*rng.standard_normal((2,2)))[0]
        G = unitary_to_superop(U, 'pp')
        target.instruments['I_trivial'] = Instrument({'p0': 0.25*G, 'p1': 0.75*G})
        target.default_gauge_group = UnitaryGaugeGroup(target.state_space, target.basis)

        self.target = target
        self.model : ExplicitOpModel = target.copy() # type: ignore
        tweak = 1.0 + 0.5j
        tweak /= abs(tweak)
        ggel = UnitaryGaugeGroupElement(unitary_to_superop(np.array([[1,0], [0, tweak]])))
        self.model.transform_inplace(ggel)

        self.dist_initial = self.model.frobeniusdist(self.target)
        self.assertGreaterEqual(self.dist_initial, 1e-2)
        return

    def _main_tester(self, method, seed, test_tol, alg_tol, gop_objective):
        assert gop_objective in {'frobenius', 'frobenius_squared', 'tracedist', 'fidelity'}
        self.setUp(seed)
        tic = time.time()
        item_weights = {'spam': 1.0, 'gates': 1.0} if gop_objective == 'tracedist' else None
        newmodel : ExplicitOpModel = gop.gaugeopt_to_target(
            self.model, self.target, item_weights=item_weights, method=method,
            tol=alg_tol, spam_metric=gop_objective, gates_metric=gop_objective
        ) # type: ignore
        toc = time.time()
        dt = toc - tic
        dist_final = newmodel.frobeniusdist(self.target)
        self.assertLessEqual(dist_final, test_tol)
        return dt
    
    def test_lbfgs_frodist(self):
        times = []
        for seed in range(3):
            dt = self._main_tester('L-BFGS-B', seed, test_tol=1e-6, alg_tol=1e-8, gop_objective='frobenius')
            times.append(dt)
        print(f'L-BFGS GaugeOpt over UnitaryGaugeGroup w.r.t. Frobenius dist: {times}.')
        return
    
    def test_lbfgs_tracedist(self):
        times = []
        for seed in [1]:
            dt = self._main_tester('L-BFGS-B', seed, test_tol=1e-6, alg_tol=1e-8, gop_objective='tracedist')
            times.append(dt)
        print(f'L-BFGS GaugeOpt over UnitaryGaugeGroup w.r.t. trace dist: {times}.')
        return
    
    @pytest.mark.skip('See https://github.com/sandialabs/pyGSTi/pull/672#issuecomment-3428804478.')
    def test_lbfgs_fidelity(self):
        times = []
        for seed in range(3):
            dt = self._main_tester('L-BFGS-B', seed, test_tol=1e-6, alg_tol=1e-8, gop_objective='fidelity')
            times.append(dt)
        print(f'L-BFGS GaugeOpt over UnitaryGaugeGroup w.r.t. fidelity : {times}.')
        return

    def test_ls(self):
        times = []
        for seed in range(3):
            dt = self._main_tester('ls', seed, test_tol=1e-6, alg_tol=1e-15, gop_objective='frobenius')
            times.append(dt)
        print(f'LS GaugeOpt over UnitaryGaugeGroup : {times}.')
        return


class FindPerfectGauge_DirectSumGaugeGroupTester(BaseCase):

    
    def _prep(self, seed):
        from pygsti.tools.leakage import leaky_qubit_model_from_pspec
        tm2 = smq1Q_XYI.target_model()
        self.target = leaky_qubit_model_from_pspec(tm2.create_processor_spec())
        self.model = self.target.copy()
        np.random.seed(seed)
        U2x2 = la.expm(np.random.randn()/2 * -1j * (pgbc.sigmax + pgbc.sigmaz)/np.sqrt(2))
        self.U = la.block_diag(U2x2, np.array([[1j]]))
        U_superop = FullArbitraryOp(pgo.unitary_to_superop(self.U, 'l2p1'), 'l2p1')
        self.gauge_grp_el = FullGaugeGroupElement(U_superop)
        self.model.transform_inplace(self.gauge_grp_el)
        self.metrics_before = gate_metrics_dict(self.model, self.target)
        check_gate_metrics_are_nontrivial(self.metrics_before, tol=1e-2)
        return
    
    def test_lbfgs_frodist(self):
        self.setUp()
        times = []
        for seed in [1]:
            self._prep(seed)
            tic = time.time()
            newmodel = gop.gaugeopt_to_target(self.model, self.target, method='L-BFGS-B', tol=1e-14, spam_metric='frobenius squared', gates_metric='frobenius squared')
            toc = time.time()
            dt = toc - tic
            metrics_after = gate_metrics_dict(newmodel, self.target)
            check_gate_metrics_near_zero(metrics_after, tol=1e-6)
            times.append(dt)
        return

