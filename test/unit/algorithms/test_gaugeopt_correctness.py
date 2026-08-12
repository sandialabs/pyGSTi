
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
import pygsti.algorithms.gaugeopt as gop
from pygsti.models import ExplicitOpModel
from pygsti.modelmembers.operations import FullArbitraryOp
import pytest



def gate_metrics_dict(model, target):
    metrics = {'infids': OrderedDict(), 'frodists': OrderedDict(), 'tracedists': OrderedDict()}
    for lbl in model.operations.keys():
        model_gate = model.operations[lbl].to_dense('minimal')
        target_gate = target.operations[lbl].to_dense('minimal')
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
        assert val > tol, f"{val} is <= {tol}, failure for gate {lbl} w.r.t. infidelity."
    for lbl, val in metrics['frodists'].items():
        if lbl == idle_label:
            continue
        assert val > tol, f"{val} is <= {tol}, failure for gate {lbl} w.r.t. frobenius distance."
    for lbl, val in metrics['tracedists'].items():
        if lbl == idle_label:
            continue
        assert val > tol, f"{val} is <= {tol}, failure for gate {lbl} w.r.r. trace distance."
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
        strength = (np.random.rand() + 1)/100
        self.U = la.expm(strength * -1j * (pgbc.sigmax + pgbc.sigmaz)/np.sqrt(2))
        self.gauge_grp_el = self.gauge_grp_el_class(pgo.unitary_to_pauligate(self.U))
        self.model.transform_inplace(self.gauge_grp_el)
        self.metrics_before = gate_metrics_dict(self.model, self.target)
        check_gate_metrics_are_nontrivial(self.metrics_before, tol=1e-4)
        return

    def _main_tester(self, method, seed, test_tol, alg_tol, gop_objective):
        assert gop_objective in {'frobenius', 'frobenius_squared', 'tracedist', 'fidelity'}
        self._gauge_transform_model(seed)
        tic = time.time()
        verbosity = 0
        newmodel = gop.gaugeopt_to_target(self.model, self.target, method=method, tol=alg_tol, spam_metric=gop_objective, gates_metric=gop_objective, verbosity=verbosity)
        toc = time.time()
        dt = toc - tic
        metrics_after = gate_metrics_dict(newmodel, self.target)
        check_gate_metrics_near_zero(metrics_after, test_tol)
        return dt
    
    def test_lbfgs_frodist(self):
        self.setUp()
        times = []
        for seed in range(self.N_REPS):
            dt = self._main_tester('L-BFGS-B', seed, test_tol=1e-5, alg_tol=1e-7, gop_objective='frobenius')
            times.append(dt)
        print(f'GaugeOpt over {self.gauge_space_name} w.r.t. Frobenius dist, using L-BFGS: {times}.')
        return
    
    def test_lbfgs_tracedist(self):
        self.setUp()
        times = []
        for seed in range(self.N_REPS):
            dt = self._main_tester('L-BFGS-B', seed, test_tol=1e-5, alg_tol=1e-7, gop_objective='tracedist')
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

    
    def _prep(self, seed, use_u1gaugegroup: bool):
        from pygsti.baseobjs.basis import Basis
        from pygsti.leakage import leaky_qubit_model_from_pspec
        from pygsti.leakage.gaugeopt import _direct_sum_unitary_group
        tm2 = smq1Q_XYI.target_model()
        self.target = leaky_qubit_model_from_pspec(tm2.create_processor_spec())

        bases  = [Basis.cast('pp', 4), Basis.cast('pp', 1)]
        tflags = [False, not use_u1gaugegroup]
        ggrp = _direct_sum_unitary_group(bases, self.target.basis, triviality_flags=tflags)
        self.target.default_gauge_group = ggrp

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
            for tf in [False, True]:
                self._prep(seed, tf)
                tic = time.time()
                newmodel = gop.gaugeopt_to_target(self.model, self.target, method='L-BFGS-B', tol=1e-14, spam_metric='frobenius squared', gates_metric='frobenius squared')
                toc = time.time()
                dt = toc - tic
                metrics_after = gate_metrics_dict(newmodel, self.target)
                check_gate_metrics_near_zero(metrics_after, tol=1e-6)
                times.append(dt)
        return



class LeakageDirectSumGroupTester(BaseCase):
    """
    Structural tests for pygsti.leakage.gaugeopt._leakage_direct_sum_group, which
    infers a U(k) (+) U(m) direct-sum gauge group from a leakage basis.
    """

    def test_leaky_qubit_matches_old_hardcode(self):
        # For l2p1 (k=2, m=1) the group must match the previously hardcoded
        # U(2) (+) U(1) construction: a unitary factor and a trivial factor.
        from pygsti.baseobjs.basis import Basis
        from pygsti.models.gaugegroup import TrivialGaugeGroup
        from pygsti.leakage.gaugeopt import _leakage_direct_sum_group
        gg = _leakage_direct_sum_group(Basis.cast('l2p1', 9))
        self.assertEqual(len(gg.subgroups), 2)
        self.assertIsInstance(gg.subgroups[0], UnitaryGaugeGroup)
        self.assertIsInstance(gg.subgroups[1], TrivialGaugeGroup)

    def test_qutrit_computational_subspace(self):
        # k=3, m=1: a ququart with a qutrit computational subspace.
        from pygsti.baseobjs.basis import Basis
        from pygsti.models.gaugegroup import TrivialGaugeGroup
        from pygsti.leakage.core import augment_for_leakage_modeling
        from pygsti.leakage.gaugeopt import _leakage_direct_sum_group
        basis = augment_for_leakage_modeling(Basis.cast('gm', 16), np.diag([1., 1., 1., 0.]))
        gg = _leakage_direct_sum_group(basis)
        self.assertIsInstance(gg.subgroups[0], UnitaryGaugeGroup)
        self.assertEqual(gg.subgroups[0].state_space.udim, 3)
        self.assertIsInstance(gg.subgroups[1], TrivialGaugeGroup)

    def test_two_leakage_levels(self):
        # k=2, m=2: both factors are nontrivial unitary groups.
        from pygsti.baseobjs.basis import Basis
        from pygsti.leakage.core import augment_for_leakage_modeling
        from pygsti.leakage.gaugeopt import _leakage_direct_sum_group
        basis = augment_for_leakage_modeling(Basis.cast('gm', 16), np.diag([1., 1., 0., 0.]))
        gg = _leakage_direct_sum_group(basis)
        self.assertIsInstance(gg.subgroups[0], UnitaryGaugeGroup)
        self.assertIsInstance(gg.subgroups[1], UnitaryGaugeGroup)
        self.assertEqual(gg.subgroups[0].state_space.udim, 2)
        self.assertEqual(gg.subgroups[1].state_space.udim, 2)

    def test_non_leakage_basis_rejected(self):
        from pygsti.baseobjs.basis import Basis
        from pygsti.leakage.gaugeopt import _leakage_direct_sum_group
        with self.assertRaises(ValueError):
            _leakage_direct_sum_group(Basis.cast('gm', 9))

    def test_interleaved_computational_levels_accepted(self):
        # pp (x) l2p1 has computational levels {0,1,3,4} interleaved with leakage
        # levels {2,5}. The group carries a level_partition so its block-diagonal
        # unitaries are permuted onto those levels: U(4) on comp (+) U(2) on leak.
        from pygsti.baseobjs.basis import Basis, TensorProdBasis
        from pygsti.leakage.gaugeopt import _leakage_direct_sum_group
        basis = TensorProdBasis((Basis.cast('pp', 4), Basis.cast('l2p1', 9)))
        gg = _leakage_direct_sum_group(basis)
        self.assertEqual(len(gg.subgroups), 2)
        self.assertIsInstance(gg.subgroups[0], UnitaryGaugeGroup)
        self.assertIsInstance(gg.subgroups[1], UnitaryGaugeGroup)
        self.assertEqual(gg.subgroups[0].state_space.udim, 4)
        self.assertEqual(gg.subgroups[1].state_space.udim, 2)
        self.assertEqual(gg.level_partition, ((0, 1, 3, 4), (2, 5)))
        # The resulting unitary must not couple computational and leakage levels.
        el = gg.compute_element(np.random.RandomState(0).randn(gg.num_params))
        comp, leak = [0, 1, 3, 4], [2, 5]
        self.assertLess(np.abs(el._u[np.ix_(comp, leak)]).max(), 1e-12)
        self.assertLess(np.abs(el._u[np.ix_(leak, comp)]).max(), 1e-12)

    def test_non_coordinate_subspace_rejected(self):
        # A computational effect that is a projector onto a non-coordinate subspace
        # (not diagonal in the standard basis) cannot be handled by a level
        # permutation, so the construction must refuse.
        from pygsti.baseobjs.basis import Basis
        from pygsti.leakage.core import augment_for_leakage_modeling
        from pygsti.leakage.gaugeopt import _leakage_direct_sum_group
        # Rank-1 projector onto (|0>+|1>)/sqrt(2): Hermitian, but not diagonal.
        v = np.array([1., 1., 0.]) / np.sqrt(2)
        E = np.outer(v, v)
        basis = augment_for_leakage_modeling(Basis.cast('gm', 9), E)
        with self.assertRaises(NotImplementedError):
            _leakage_direct_sum_group(basis)


class LagoifiedGopparamsDictsTester(BaseCase):
    """
    Regression test for pygsti.leakage.gaugeopt.lagoified_gopparams_dicts.

    It used to build its unitary gauge group from `tm.basis.state_space`, which raises
    AttributeError whenever `tm.basis` is a composite basis (e.g. TensorProdBasis) --
    the common case for any multi-subsystem model, leakage or not. It should instead use
    `tm.state_space`, which every Model has regardless of its basis's type.
    """

    def test_tensor_prod_basis_target_model_does_not_crash(self):
        from pygsti.baseobjs import ExplicitStateSpace
        from pygsti.baseobjs.basis import Basis, TensorProdBasis
        from pygsti.leakage.gaugeopt import lagoified_gopparams_dicts
        from pygsti.models.gaugegroup import DirectSumUnitaryGroup

        # A qubit tensored with a leaky qutrit: `basis` is a TensorProdBasis, which (unlike
        # BuiltinBasis) has no `state_space` attribute of its own.
        basis = TensorProdBasis((Basis.cast('pp', 4), Basis.cast('l2p1', 9)))
        self.assertFalse(hasattr(basis, 'state_space'))
        state_space = ExplicitStateSpace(['Q0', 'Q1'], [2, 3])
        target_model = ExplicitOpModel(state_space, basis)

        gopparams_dicts = lagoified_gopparams_dicts([{'target_model': target_model}])

        self.assertEqual(len(gopparams_dicts), 2)
        self.assertIsInstance(gopparams_dicts[0]['gauge_group'], UnitaryGaugeGroup)
        self.assertIsInstance(gopparams_dicts[1]['gauge_group'], DirectSumUnitaryGroup)


class FindPerfectGauge_DirectSumGaugeGroup4LevelTester(BaseCase):
    """
    Gauge-recovery test for the generalized direct-sum gauge group on a 4-level
    system: a 2-dimensional computational subspace plus a 2-dimensional leakage
    subspace (k=2, m=2), which the old hardcoded U(2) (+) U(1) group could not
    express.
    """

    def _prep(self, seed):
        from pygsti.baseobjs import ExplicitStateSpace
        from pygsti.baseobjs.basis import Basis
        from pygsti.leakage.core import augment_for_leakage_modeling
        from pygsti.leakage.gaugeopt import _leakage_direct_sum_group
        from pygsti.modelmembers.povms import UnconstrainedPOVM
        from pygsti.modelmembers.states import FullState
        from pygsti.tools.basistools import stdmx_to_vec

        basis = augment_for_leakage_modeling(Basis.cast('gm', 16), np.diag([1., 1., 0., 0.]))
        ss = ExplicitStateSpace(['Q0'], [4])
        target = ExplicitOpModel(ss, basis)

        rho0 = np.zeros((4, 4)); rho0[0, 0] = 1.0
        E0 = np.diag([1., 0., 0., 0.]).astype(complex)
        E1 = np.eye(4) - E0
        # The basis is Hermitian, so these superkets are real up to rounding; take the
        # real part explicitly to avoid ComplexWarnings from FullState's real storage.
        target.preps['rho0'] = FullState(np.real(stdmx_to_vec(rho0, basis)))
        target.povms['Mdefault'] = UnconstrainedPOVM(
            [('0', np.real(stdmx_to_vec(E0, basis))), ('1', np.real(stdmx_to_vec(E1, basis)))],
            evotype='default')

        u_x = la.expm(-0.25j * np.pi * pgbc.sigmax)
        u_y = la.expm(-0.25j * np.pi * pgbc.sigmay)
        for name, u2 in [('Gxpi2', u_x), ('Gypi2', u_y)]:
            u4 = la.block_diag(u2, np.eye(2))
            target.operations[pgl.Label((name, 'Q0'))] = np.real(pgo.unitary_to_superop(u4, basis))

        target.default_gauge_group = _leakage_direct_sum_group(basis)
        self.target = target
        self.model = target.copy()

        np.random.seed(seed)
        def _rand_u2():
            h = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
            return la.expm(-0.5j * (h + h.T.conj()))
        U = la.block_diag(_rand_u2(), _rand_u2())
        U_superop = FullArbitraryOp(np.real(pgo.unitary_to_superop(U, basis)), basis)
        self.gauge_grp_el = FullGaugeGroupElement(U_superop)
        self.model.transform_inplace(self.gauge_grp_el)

        self.metrics_before = gate_metrics_dict(self.model, self.target)
        check_gate_metrics_are_nontrivial(self.metrics_before, tol=1e-2)
        return

    def test_lbfgs_frodist(self):
        for seed in [1]:
            self._prep(seed)
            newmodel = gop.gaugeopt_to_target(
                self.model, self.target, method='L-BFGS-B', tol=1e-14,
                spam_metric='frobenius squared', gates_metric='frobenius squared')
            metrics_after = gate_metrics_dict(newmodel, self.target)
            check_gate_metrics_near_zero(metrics_after, tol=1e-6)
        return


class FindPerfectGauge_DirectSumGaugeGroupInterleavedTester(BaseCase):
    """
    Gauge-recovery test for the generalized direct-sum gauge group on a leakage basis
    with *interleaved* computational and leakage levels: pp (x) l2p1, a 6-level system
    whose computational subspace is levels {0,1,3,4} and leakage subspace is {2,5}
    (k=4, m=2). The gauge group carries a level_partition so its U(4) (+) U(2)
    block-diagonal unitaries are permuted onto the correct levels.
    """

    def _embed(self, blocks, level_partition, udim):
        # Scatter a block-diagonal unitary onto the (possibly interleaved) levels named
        # by level_partition -- the same permutation the gauge group applies internally.
        grouped_order = [lvl for block in level_partition for lvl in block]
        u_grouped = la.block_diag(*blocks)
        perm = np.zeros((udim, udim))
        for grouped_idx, level in enumerate(grouped_order):
            perm[level, grouped_idx] = 1.0
        return perm @ u_grouped @ perm.T

    def _prep(self, seed):
        from pygsti.baseobjs import ExplicitStateSpace
        from pygsti.baseobjs.basis import Basis, TensorProdBasis
        from pygsti.leakage.gaugeopt import _leakage_direct_sum_group
        from pygsti.modelmembers.povms import UnconstrainedPOVM
        from pygsti.modelmembers.states import FullState
        from pygsti.tools.basistools import stdmx_to_vec

        basis = TensorProdBasis((Basis.cast('pp', 4), Basis.cast('l2p1', 9)))
        gg = _leakage_direct_sum_group(basis)
        partition = gg.level_partition  # ((0,1,3,4), (2,5))
        udim = 6

        ss = ExplicitStateSpace(['Q0'], [6])
        target = ExplicitOpModel(ss, basis)

        rho0 = np.zeros((6, 6)); rho0[0, 0] = 1.0
        E0 = np.diag([1., 0., 0., 0., 0., 0.]).astype(complex)
        E1 = np.eye(6) - E0
        # The basis is Hermitian, so these superkets are real up to rounding; take the
        # real part explicitly to avoid ComplexWarnings from FullState's real storage.
        target.preps['rho0'] = FullState(np.real(stdmx_to_vec(rho0, basis)))
        target.povms['Mdefault'] = UnconstrainedPOVM(
            [('0', np.real(stdmx_to_vec(E0, basis))), ('1', np.real(stdmx_to_vec(E1, basis)))],
            evotype='default')

        u_x = la.expm(-0.25j * np.pi * pgbc.sigmax)
        u_y = la.expm(-0.25j * np.pi * pgbc.sigmay)
        for name, u2 in [('Gxpi2', u_x), ('Gypi2', u_y)]:
            # gate acts as u2 on two of the four computational levels, identity elsewhere
            u_comp = la.block_diag(u2, np.eye(2))
            u6 = self._embed([u_comp, np.eye(2)], partition, udim)
            target.operations[pgl.Label((name, 'Q0'))] = np.real(pgo.unitary_to_superop(u6, basis))

        target.default_gauge_group = gg
        self.target = target
        self.model = target.copy()

        np.random.seed(seed)
        def _rand_u(n):
            h = np.random.randn(n, n) + 1j * np.random.randn(n, n)
            return la.expm(-0.5j * (h + h.T.conj()))
        U = self._embed([_rand_u(4), _rand_u(2)], partition, udim)
        U_superop = FullArbitraryOp(np.real(pgo.unitary_to_superop(U, basis)), basis)
        self.gauge_grp_el = FullGaugeGroupElement(U_superop)
        self.model.transform_inplace(self.gauge_grp_el)

        self.metrics_before = gate_metrics_dict(self.model, self.target)
        check_gate_metrics_are_nontrivial(self.metrics_before, tol=1e-2)
        return

    def test_lbfgs_frodist(self):
        for seed in [1]:
            self._prep(seed)
            newmodel = gop.gaugeopt_to_target(
                self.model, self.target, method='L-BFGS-B', tol=1e-14,
                spam_metric='frobenius squared', gates_metric='frobenius squared')
            metrics_after = gate_metrics_dict(newmodel, self.target)
            check_gate_metrics_near_zero(metrics_after, tol=1e-6)
        return
