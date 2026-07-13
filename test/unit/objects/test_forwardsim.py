from unittest import mock, TestCase

import numpy as np
import pytest
import scipy.linalg as la

import pygsti.models as models
from pygsti.forwardsims import ForwardSimulator, \
    MapForwardSimulator, SimpleMapForwardSimulator, \
    MatrixForwardSimulator,  SimpleMatrixForwardSimulator, \
    TorchForwardSimulator
from pygsti.forwardsims import torchfwdsim
from pygsti.models import ExplicitOpModel
from pygsti.circuits import Circuit, create_lsgst_circuit_lists
from pygsti.baseobjs import Label as L
from ..util import BaseCase

from pygsti.data import simulate_data
from pygsti.modelpacks import smq1Q_XYI, smq2Q_XYICNOT
from pygsti.modelmembers.operations import EmbeddedOp, FullTPOp
from pygsti.modelmembers.povms import ComputationalBasisPOVM
from pygsti.modelmembers.instruments import Instrument, TPInstrument
from pygsti.protocols import gst
from pygsti.protocols.protocol import ProtocolData
from pygsti.tools import two_delta_logl


class AbstractForwardSimTester(BaseCase):
    # XXX is it really neccessary to test an abstract base class?
    def setUp(self):
        mock_model = mock.MagicMock()
        mock_model.evotype.return_value = "densitymx"
        mock_model.circuit_outcomes.return_value = ('NA',)
        mock_model.num_params = 0
        self.fwdsim = ForwardSimulator(mock_model)
        self.circuit = Circuit("GxGx")

    def test_create_layout(self):
        self.fwdsim.create_layout([self.circuit])

    def test_bulk_fill_probs(self):
        layout = self.fwdsim.create_layout([self.circuit])
        with self.assertRaises(NotImplementedError):
            self.fwdsim.bulk_fill_probs(np.zeros(1), layout)

    def test_bulk_fill_dprobs(self):
        layout = self.fwdsim.create_layout([self.circuit])
        with self.assertRaises(NotImplementedError):
            self.fwdsim.bulk_fill_dprobs(np.zeros((1,0)), layout)

    def test_bulk_fill_hprobs(self):
        layout = self.fwdsim.create_layout([self.circuit])
        with self.assertRaises(NotImplementedError):
            self.fwdsim.bulk_fill_hprobs(np.zeros((1,0,0)), layout)


class ForwardSimBase(object):
    @classmethod
    def setUpClass(cls):
        ExplicitOpModel._strict = False
        cls.model = models.create_explicit_model_from_expressions(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"]
        )
        cls.model_matrix = cls.model.copy()
        cls.model_matrix.sim = 'matrix'

    def setUp(self):
        self.fwdsim = self.model.sim
        self.fwdsim_matrix = self.model_matrix.sim
        self.layout = self.fwdsim.create_layout([('Gx',), ('Gx', 'Gx')], array_types=('e', 'ep', 'epp'))
        self.nP = self.model.num_params
        self.nEls = self.layout.num_elements

    def test_bulk_fill_probs(self):
        pmx = np.empty(self.nEls, 'd')
        print(self.fwdsim.model._opcaches)
        self.fwdsim.bulk_fill_probs(pmx, self.layout)
        # TODO assert correctness

    def test_bulk_fill_dprobs(self):
        dmx = np.empty((self.nEls, self.nP), 'd')
        pmx = np.empty(self.nEls, 'd')
        self.fwdsim.bulk_fill_dprobs(dmx, self.layout, pr_array_to_fill=pmx)
        # TODO assert correctness

    def test_bulk_fill_dprobs_with_block_size(self):
        dmx = np.empty((self.nEls, self.nP), 'd')
        self.fwdsim.bulk_fill_dprobs(dmx, self.layout)
        # TODO assert correctness

    def test_bulk_fill_hprobs(self):
        hmx = np.zeros((self.nEls, self.nP, self.nP), 'd')
        dmx = np.zeros((self.nEls, self.nP), 'd')
        pmx = np.zeros(self.nEls, 'd')
        self.fwdsim.bulk_fill_hprobs(hmx, self.layout,
                                     pr_array_to_fill=pmx, deriv1_array_to_fill=dmx, deriv2_array_to_fill=dmx)
        # TODO assert correctness

        hmx = np.zeros((self.nEls, self.nP, self.nP), 'd')
        dmx1 = np.zeros((self.nEls, self.nP), 'd')
        dmx2 = np.zeros((self.nEls, self.nP), 'd')
        pmx = np.zeros(self.nEls, 'd')
        self.fwdsim.bulk_fill_hprobs(hmx, self.layout,
                                     pr_array_to_fill=pmx, deriv1_array_to_fill=dmx1, deriv2_array_to_fill=dmx2)
        # TODO assert correctness

    def test_iter_hprobs_by_rectangle(self):
        # TODO optimize
        mx = np.zeros((self.nEls, self.nP, self.nP), 'd')
        dmx1 = np.zeros((self.nEls, self.nP), 'd')
        dmx2 = np.zeros((self.nEls, self.nP), 'd')
        pmx = np.zeros(self.nEls, 'd')
        self.fwdsim.bulk_fill_hprobs(mx, self.layout, pr_array_to_fill=pmx,
                                     deriv1_array_to_fill=dmx1, deriv2_array_to_fill=dmx2)
        # TODO assert correctness


class MatrixForwardSimTester(ForwardSimBase, BaseCase):
    def test_doperation(self):
        dg = self.fwdsim_matrix._doperation(L('Gx'), flat=False)
        dgflat = self.fwdsim_matrix._doperation(L('Gx'), flat=True)
        # TODO assert correctness

    def test_hoperation(self):
        hg = self.fwdsim_matrix._hoperation(L('Gx'), flat=False)
        hgflat = self.fwdsim_matrix._hoperation(L('Gx'), flat=True)
        # TODO assert correctness


class CPTPMatrixForwardSimTester(MatrixForwardSimTester):
    @classmethod
    def setUpClass(cls):
        super(CPTPMatrixForwardSimTester, cls).setUpClass()
        cls.model = cls.model_matrix.copy()
        cls.model.set_all_parameterizations("CPTPLND")  # so gates have nonzero hessians


class MapForwardSimTester(ForwardSimBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        super(MapForwardSimTester, cls).setUpClass()
        cls.model = cls.model.copy()
        cls.model.sim = MapForwardSimulator()


class ZeroParamModelTester(BaseCase):
    def test_dprobs_hprobs_empty_for_zero_param_model(self):
        # static model has no free params -> dprobs/hprobs return empty arrays
        # quietly instead of an opaque IndexError (#604)
        model = smq1Q_XYI.target_model('static')
        self.assertEqual(model.num_params, 0)
        circuit = Circuit([L('Gxpi2', 0)], line_labels=(0,))
        for sim in (MapForwardSimulator(), MatrixForwardSimulator()):
            model.sim = sim
            with self.assertNoWarns():
                dprobs = model.sim.bulk_dprobs([circuit])
                hprobs = model.sim.bulk_hprobs([circuit])
            for dprob in next(iter(dprobs.values())).values():
                self.assertEqual(dprob.shape, (0,))
            for hprob in next(iter(hprobs.values())).values():
                self.assertEqual(hprob.shape, (0, 0))


class BaseProtocolData:

    @classmethod
    def setUpClass(cls):
        cls.gst_design = smq1Q_XYI.create_gst_experiment_design(max_max_length=16)
        cls.mdl_target = smq1Q_XYI.target_model()
        cls.mdl_datagen = cls.mdl_target.depolarize(op_noise=0.05, spam_noise=0.025)

        ds = simulate_data(cls.mdl_datagen, cls.gst_design.all_circuits_needing_data, 20000, sample_error='none')
        cls.gst_data = ProtocolData(cls.gst_design, ds)


def dict_to_arrays(od, ravel=True):
    keys, vals = zip(*od.items())
    keys = np.array([k for k in keys], dtype=object)
    vals = np.array(vals)
    if ravel:
        return keys.ravel(), vals.ravel()
    else:
        return keys, vals


def _sorted_outcome_vals(od):
    """Stack an outcome-label dict's values in sorted-key order.

    Different forward simulators can build a circuit's per-outcome dict by iterating
    instrument members and POVM effects in different nesting orders (e.g. when a circuit
    contains instrument layers), so raw dict-iteration order isn't guaranteed to line up
    across simulators even though every (key, value) pair is individually correct. Sorting
    by the outcome-label tuple gives a canonical order all simulators can be compared against.
    """
    return np.array([od[k] for k in sorted(od.keys())])


class ForwardSimTestHelper:
    """
    Compute outcome probabilities and Jacobians by parsing all circuits in a block.
    It would be (probably?) be much more expensive to iterate through the circuits
    and compute probabilities and Jacobians one at a time.
    """

    def __init__(self, model, sims, circuits):
        assert isinstance(sims, list)
        assert len(sims) > 1, "At least two ForwardSimulators must be provided."
        assert isinstance(sims[0], ForwardSimulator)
        self.sims = sims
        self.base_model = model
        self.models = [model.copy() for _ in range(len(self.sims))]
        for i,m in enumerate(self.models):
            m.sim = sims[i]
        assert len(circuits) > 0
        assert isinstance(circuits[0], Circuit)

        self.circuits = circuits
        self.circuit_strs = np.array([str(c) for c in self.circuits])
        self.outcome_probs = None
        self.outcome_probs_jacs = None
        return
    
    def compute_outcome_probs(self):
        agg_probs = []
        for m in self.models:
            out = m.sim.bulk_probs(self.circuits)
            # ^ a dict from circuits to outcome probability dicts
            circs, probs = dict_to_arrays(out)
            curr_cstrs = np.array([str(c) for c in circs])
            assert np.all(self.circuit_strs  == curr_cstrs), "Circuits outcome probabilities were returned in a different order than given."
            temp2 = [_sorted_outcome_vals(p) for p in probs]
            probs = np.concatenate(temp2)
            agg_probs.append(probs)
        probs = np.vstack(agg_probs)
        self.outcome_probs = probs
        return
    
    def compute_outcome_probs_jac(self):
        agg_jacs = []
        for m in self.models:
            out = m.sim.bulk_dprobs(self.circuits)
            circs, jac_dicts = dict_to_arrays(out)
            curr_cstrs = np.array([str(c) for c in circs])
            assert np.all(self.circuit_strs  == curr_cstrs),  "Circuit outcome probability Jacobians were returned in a different order than given."
            temp2 = [_sorted_outcome_vals(jd) for jd in jac_dicts]
            jac = np.concatenate(temp2, axis=0)
            agg_jacs.append(jac)
        self.outcome_probs_jacs = np.stack(agg_jacs)
        return
    
    def probs_colinearities(self):
        if self.outcome_probs is None:
            self.compute_outcome_probs()
        _, _, vt = la.svd(self.outcome_probs, full_matrices=False)
        v = vt[0,:]
        row_norms = la.norm(self.outcome_probs, axis=1)
        scaled_outcome_probs = self.outcome_probs / row_norms[:, np.newaxis]
        colinearities = scaled_outcome_probs @ v
        # ^ That has a sign ambiguity that we need to resolve.
        num_neg = np.count_nonzero(colinearities < 0)
        # ^ we expect that to be zero or == colinearities.size
        if num_neg > colinearities.size/2:
            colinearities *= -1
        return colinearities
    
    def jac_colinearities(self):
        if self.outcome_probs_jacs is None:
            self.compute_outcome_probs_jac()
        assert self.outcome_probs_jacs is not None  # make linter happy
        alljacs = np.stack([J.ravel() for J in self.outcome_probs_jacs])
        # ^ Each row of alljacs is a vectorized Jacobian of circuit outcome probabilities.
        #   The length of a given row is len(self.circuits) * (number of model parameters).
        _, _, vt = la.svd(alljacs, full_matrices=False)
        v = vt[0,:]
        row_norms = la.norm(alljacs, axis=1)
        scaled_alljacs = alljacs / row_norms[:, np.newaxis]
        colinearities = scaled_alljacs @ v
        # ^ That has a sign ambiguity that we need to resolve.
        num_neg = np.count_nonzero(colinearities < 0)
        # ^ we expect that to be zero or == colinearities.size
        if num_neg > colinearities.size/2:
            colinearities *= -1
        return colinearities

    

class ForwardSimConsistencyTester(TestCase):
    """Base class for Testers that check TorchForwardSimulator agrees with the classic forward
    simulators (SimpleMap, SimpleMatrix, Map, Matrix) on circuit outcome probabilities and their
    Jacobians, for some noisy model + circuit list.

    Subclasses override `build_helper` to supply the model + circuits under test, and may
    override PROBS_TOL / JACS_TOL. The base implementation (exercised directly by this class, not
    just inherited) checks smq1Q_XYI's ideal target model under depolarizing noise.
    """

    PROBS_TOL = 1e-14
    JACS_TOL = 1e-10
    STASHED_DTYPE = torchfwdsim.DEFAULT_REAL_TYPE

    @staticmethod
    def standard_sims():
        """The 4 always-available forward simulators, plus TorchForwardSimulator when installed.
        Constructing TorchForwardSimulator reads the torchfwdsim.DEFAULT_REAL_TYPE module global,
        so we overwrite it with float64 first; tearDown restores it."""
        sims = [
            SimpleMapForwardSimulator(),
            SimpleMatrixForwardSimulator(),
            MapForwardSimulator(),
            MatrixForwardSimulator()
        ]
        if TorchForwardSimulator.ENABLED:
            import torch
            # TorchForwardSimulator.__init__ captures DEFAULT_REAL_TYPE at construction time, so we
            # must overwrite it with float64 *before* constructing the sim; tearDown restores it.
            torchfwdsim.DEFAULT_REAL_TYPE = torch.float64
            sims.append(TorchForwardSimulator())
        return sims

    @staticmethod
    def standard_lsgst_circuits(model, max_lengths=(4,)):
        prep_fiducials = smq1Q_XYI.prep_fiducials()
        meas_fiducials = smq1Q_XYI.meas_fiducials()
        germs = smq1Q_XYI.germs()
        return create_lsgst_circuit_lists(model, prep_fiducials, meas_fiducials, germs, list(max_lengths))[0]

    def build_helper(self):
        """Build the ForwardSimTestHelper this Tester checks. Override in subclasses to swap in a
        different model and/or circuit list; the default exercises smq1Q_XYI's target model."""
        model_ideal = smq1Q_XYI.target_model()
        if TorchForwardSimulator.ENABLED:
            # TorchForwardSimulator can only work with TP modelmembers.
            model_ideal.convert_members_inplace(to_type='full TP')
        model_noisy = model_ideal.depolarize(op_noise=0.05, spam_noise=0.025)
        circuits = self.standard_lsgst_circuits(model_noisy)
        return ForwardSimTestHelper(model_noisy, self.standard_sims(), circuits)

    def tearDown(self) -> None:
        torchfwdsim.DEFAULT_REAL_TYPE = self.STASHED_DTYPE
        return super().tearDown()

    def test_consistent_probs(self):
        fsth = self.build_helper()
        pcl = fsth.probs_colinearities()
        if np.any(pcl < 1 - self.PROBS_TOL):
            locs = np.where(pcl < 1 - self.PROBS_TOL)[0]
            msg = f"""
            We've compared outcome probabilities produced by each forward simulator to a reference
            value obtained with consideration to all forward simulators at once. At least one of
            the forward simulators returned a vector that points in a meaingfully different direction
            than the reference value. Specifically, we required a colinearity of at least {1 - self.PROBS_TOL},
            but ...
            """
            for idx in locs:
                msg += f"""
                The colinearity of the probabilities from {fsth.sims[idx]} and the reference was {pcl[idx]}.
                """
            self.assertTrue(False, msg)

    def test_consistent_jacs(self):
        fsth = self.build_helper()
        jcl = fsth.jac_colinearities()
        if np.any(jcl < 1 - self.JACS_TOL):
            locs = np.where(jcl < 1 - self.JACS_TOL)[0]
            msg = f"""
            We've compared the Jacobians of circuit outcome probabilities produced by each forward simulator
            to a reference value obtained with consideration to all forward simulators at once. At least one
            of the forward simulators returned a Jacobian that was meaningfully different than the reference,
            as measured by colinearity in the sense of the trace inner product. Specifically, we required a
            colinearity of at least {1 - self.JACS_TOL}, but ...
            """
            for idx in locs:
                msg += f"""
                The colinearity of the Jacobian from {fsth.sims[idx]} and the reference was {jcl[idx]}.
                """
            self.assertTrue(False, msg)


class LindbladForwardSimConsistencyTester(ForwardSimConsistencyTester):
    """End-to-end check that TorchForwardSimulator runs on a full CPTPLND (Lindblad) model and agrees
    with the other forward simulators (GitHub issue 607).  Every model member is a Lindblad-composed
    object: the prep is a ComposedState, the POVM a ComposedPOVM, and each gate a
    ComposedOp([static ideal, ExpErrorgenOp]) -- exercising the whole Torchable op + SPAM chain."""

    PROBS_TOL = 1e-12
    JACS_TOL = 1e-9

    def build_helper(self):
        # Lindblad SPAM *and* operations: ComposedState prep, ComposedPOVM, ComposedOp gates.
        model = smq1Q_XYI.target_model()
        model.set_all_parameterizations('CPTPLND')
        # perturb off the ideal so the error generators are nonzero
        rng = np.random.default_rng(7)
        model.from_vector(model.to_vector() + 0.03 * rng.standard_normal(model.num_params))
        circuits = self.standard_lsgst_circuits(model)
        return ForwardSimTestHelper(model, self.standard_sims(), circuits)


class GLNDULindbladForwardSimConsistencyTester(ForwardSimConsistencyTester):
    """Like LindbladForwardSimConsistencyTester, but with the GLNDU (unconstrained-'other'-block)
    Lindblad parameterization, to exercise _EEGVectorElements.torch_stateless_data (GitHub issue 607)."""

    PROBS_TOL = 1e-12
    JACS_TOL = 1e-9

    def build_helper(self):
        model = smq1Q_XYI.target_model()
        model.set_all_parameterizations('GLNDU')
        rng = np.random.default_rng(70)
        model.from_vector(model.to_vector() + 0.03 * rng.standard_normal(model.num_params))
        circuits = self.standard_lsgst_circuits(model)
        return ForwardSimTestHelper(model, self.standard_sims(), circuits)


class EmbeddedOpForwardSimConsistencyTester(ForwardSimConsistencyTester):
    """End-to-end check that TorchForwardSimulator agrees with the other forward simulators on a
    2-qubit model where two of the single-qubit gates are EmbeddedOp-wrapped, noisy FullTPOps
    (GitHub issue 607)."""

    PROBS_TOL = 1e-12
    JACS_TOL = 1e-9

    def build_helper(self):
        model = smq2Q_XYICNOT.target_model('full TP')
        ss = model.state_space
        rng = np.random.default_rng(21)

        def noisy_1q_tp(scale=0.05):
            mx = np.eye(4)
            mx[1:, :] += scale * rng.standard_normal((3, 4))
            return FullTPOp(mx, basis='pp')

        model.operations[L('Gxpi2', 0)] = EmbeddedOp(ss, [0], noisy_1q_tp())
        model.operations[L('Gypi2', 1)] = EmbeddedOp(ss, [1], noisy_1q_tp())

        circuits = [
            Circuit([L('Gxpi2', 0)], line_labels=(0, 1)),
            Circuit([L('Gypi2', 1)], line_labels=(0, 1)),
            Circuit([L('Gxpi2', 0), L('Gypi2', 1)], line_labels=(0, 1)),
            Circuit([L('Gxpi2', 0), L('Gcnot', (0, 1)), L('Gypi2', 1)], line_labels=(0, 1)),
            Circuit([L('Gypi2', 1), L('Gxpi2', 0), L('Gxpi2', 0)], line_labels=(0, 1)),
        ]
        return ForwardSimTestHelper(model, self.standard_sims(), circuits)


class ComputationalPOVMForwardSimConsistencyTester(ForwardSimConsistencyTester):
    """End-to-end check that TorchForwardSimulator agrees with the other forward simulators when
    the model's POVM is a 0-parameter ComputationalBasisPOVM, exercising the 0-param code path in
    StatelessModelCircuitStore.get_free_params (GitHub issue 607)."""

    PROBS_TOL = 1e-12
    JACS_TOL = 1e-9

    def build_helper(self):
        model = smq1Q_XYI.target_model('full TP')
        model = model.depolarize(op_noise=0.05, spam_noise=0.025)
        model.povms['Mdefault'] = ComputationalBasisPOVM(1, model.evotype)
        circuits = self.standard_lsgst_circuits(model)
        return ForwardSimTestHelper(model, self.standard_sims(), circuits)


def _iz_instrument_model(instrument_cls, seed):
    """A noisy 'full TP' model with a weak-Z-measurement instrument 'Iz' installed, whose two
    members are built from the ideal POVM's effect projectors (pattern from test_instrument.py)."""
    model = smq1Q_XYI.target_model('full TP')
    model = model.depolarize(op_noise=0.05, spam_noise=0.025)
    E = model.povms['Mdefault']['0'].to_dense().ravel()
    Erem = model.povms['Mdefault']['1'].to_dense().ravel()
    model.instruments[L('Iz', 0)] = instrument_cls({'plus': np.outer(E, E), 'minus': np.outer(Erem, Erem)})
    rng = np.random.default_rng(seed)
    model.from_vector(model.to_vector() + 0.01 * rng.standard_normal(model.num_params))
    return model


def _iz_instrument_circuits():
    """Circuits mixing ordinary gates and the 'Iz' instrument, including one with two Iz layers
    (so the layout expands to the *product* of both layers' member outcomes)."""
    return [
        Circuit([L('Gxpi2', 0), L('Iz', 0)], line_labels=(0,)),
        Circuit([L('Iz', 0), L('Gxpi2', 0), L('Iz', 0)], line_labels=(0,)),
        Circuit([L('Gypi2', 0), L('Iz', 0), L('Gxpi2', 0)], line_labels=(0,)),
    ]


class InstrumentForwardSimConsistencyTester(ForwardSimConsistencyTester):
    """End-to-end check that TorchForwardSimulator agrees with the other forward simulators on
    circuits containing a noisy Instrument layer, including a circuit with two Iz layers
    (GitHub issue 607)."""

    PROBS_TOL = 1e-12
    JACS_TOL = 1e-9

    def build_helper(self):
        model = _iz_instrument_model(Instrument, seed=51)
        return ForwardSimTestHelper(model, self.standard_sims(), _iz_instrument_circuits())


class TPInstrumentForwardSimConsistencyTester(ForwardSimConsistencyTester):
    """Like InstrumentForwardSimConsistencyTester, but the 'Iz' instrument is a TPInstrument, so
    its members are TPInstrumentOp objects (GitHub issue 607)."""

    PROBS_TOL = 1e-12
    JACS_TOL = 1e-9

    def build_helper(self):
        model = _iz_instrument_model(TPInstrument, seed=52)
        return ForwardSimTestHelper(model, self.standard_sims(), _iz_instrument_circuits())


class CPTRInstrumentForwardSimConsistencyTester(ForwardSimConsistencyTester):
    """End-to-end check of the full RootConjOperator / ComposedPOVMEffect chain: an Instrument
    built via Instrument.from_effects on a CPTPLND-parameterized model (GitHub issue 607). The
    reference simulators fall back to finite-differencing for RootConjOperator's derivative
    (it has no analytic deriv_wrt_params), so this Tester uses a looser Jacobian tolerance than
    the other forward-sim consistency Testers."""

    PROBS_TOL = 1e-12
    JACS_TOL = 1e-6

    def build_helper(self):
        model = smq1Q_XYI.target_model()
        model.set_all_parameterizations('CPTPLND')
        rng = np.random.default_rng(44)
        model.from_vector(model.to_vector() + 0.02 * rng.standard_normal(model.num_params))

        # Effects spectrally interior to (0, 1) and non-degenerate, so RootConjOperator's
        # eigh-backward Jacobian is well-conditioned and no NumericalDomainWarning fires.
        E0 = np.diag([0.7, 0.3]).astype(complex)
        E1 = np.diag([0.3, 0.7]).astype(complex)
        instr = Instrument.from_effects({'p0': E0, 'p1': E1}, model.basis)
        model.instruments[L('Iz', 0)] = instr
        iz = model.instruments[L('Iz', 0)]
        rng2 = np.random.default_rng(45)
        iz.from_vector(iz.to_vector() + 0.01 * rng2.standard_normal(iz.num_params))

        return ForwardSimTestHelper(model, self.standard_sims(), _iz_instrument_circuits())


@pytest.mark.skipif(not TorchForwardSimulator.ENABLED, reason="PyTorch is not installed.")
def test_torch_instrument_expansion_plumbing():
    """Directly exercise StatelessModelCircuitStore's instrument-expansion bookkeeping: its
    torch_bases dict must expose each instrument member under its expanded op label (e.g.
    Iz_plus, Iz_minus), and the resulting circuit outcome probabilities must match
    model.sim.bulk_probs (GitHub issue 607)."""
    import torch
    from pygsti.forwardsims.torchfwdsim import StatelessModelCircuitStore

    stashed_dtype = torchfwdsim.DEFAULT_REAL_TYPE
    torchfwdsim.DEFAULT_REAL_TYPE = torch.float64
    try:
        model = _iz_instrument_model(Instrument, seed=53)
        circuits = _iz_instrument_circuits()
        model.sim = TorchForwardSimulator()
        layout = model.sim.create_layout(circuits)
        smcs = StatelessModelCircuitStore(model, layout, torch.float64, 'cpu')
        free_params = smcs.get_free_params(model, torch.float64, 'cpu')
        torch_bases = smcs.get_torch_bases(free_params)

        assert L('Iz_plus', 0) in torch_bases
        assert L('Iz_minus', 0) in torch_bases

        probs = smcs.circuit_probs_from_torch_bases(torch_bases).detach().numpy()
        ref = model.sim.bulk_probs(circuits)
        ref_probs = np.concatenate([np.array(list(p.values())) for p in ref.values()])
        assert np.allclose(probs, ref_probs, atol=1e-9)
    finally:
        torchfwdsim.DEFAULT_REAL_TYPE = stashed_dtype


class ForwardSimIntegrationTester(BaseProtocolData):

    def _run(self, obj : ForwardSimulator.Castable):
        self.setUpClass()
        proto = gst.GateSetTomography(smq1Q_XYI.target_model("full TP"), 'stdgaugeopt', name="testGST")
        results = proto.run(self.gst_data, simulator=obj)
        mdl_result = results.estimates["testGST"].models['stdgaugeopt']
        twoDLogL = two_delta_logl(mdl_result, self.gst_data.dataset)
        assert twoDLogL <= 0.05  # should be near 0 for perfect data
        pass

    # shared memory forward simulators
    def test_simple_matrix_fwdsim(self):
        self._run(SimpleMatrixForwardSimulator)

    def test_simple_map_fwdsim(self):
        self._run(SimpleMapForwardSimulator)

    @pytest.mark.skipif(not TorchForwardSimulator.ENABLED, reason="PyTorch is not installed.")
    def test_torch_fwdsim(self):
        self._run(TorchForwardSimulator)

    # distributed-memory forward simulators
    def test_map_fwdsim(self):
        self._run(MapForwardSimulator)

    def test_matrix_fwdsim(self):
        self._run(MatrixForwardSimulator)

