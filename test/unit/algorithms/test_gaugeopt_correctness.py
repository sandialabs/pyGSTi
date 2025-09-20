
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



def gate_metrics_dict(model, target):
    metrics = {'infids': OrderedDict(), 'frodists': OrderedDict(), 'tracedists': OrderedDict()}
    for lbl in model.operations.keys():
        model_gate = model[lbl]
        target_gate = target[lbl]
        metrics['infids'][lbl] = pgo.entanglement_infidelity(model_gate, target_gate)
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



class sm1QXYI_FindPerfectGauge:

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
        np.random.seed(0)
        target = smq1Q_XYI.target_model()
        self.target = target.depolarize(op_noise=0.01, spam_noise=0.001)
        self.model = None
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

    pass


"""
This class' test for minimizing trace distance with L-BFGS takes excessively long (about 5 seconds).
"""
class FindPerfectGauge_TPGroup_Tester(BaseCase, sm1QXYI_FindPerfectGauge):

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
    
    def gauge_transform_model(self, seed):
        sm1QXYI_FindPerfectGauge.gauge_transform_model(self, seed)
        # ^ That sets self.model
        return


class FindPerfectGauge_UnitaryGroup_Tester(BaseCase, sm1QXYI_FindPerfectGauge):

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
    
    def gauge_transform_model(self, seed):
        sm1QXYI_FindPerfectGauge.gauge_transform_model(self, seed)
        # ^ That sets self.model
        return


"""
This class' test for minimizing trace distance with L-BFGS takes excessively long (about 5 seconds).
"""
class FindPerfectGauge_FullGroup_Tester(BaseCase, sm1QXYI_FindPerfectGauge):

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

    def gauge_transform_model(self, seed):
        sm1QXYI_FindPerfectGauge.gauge_transform_model(self, seed)
        # ^ That sets self.model
        return
