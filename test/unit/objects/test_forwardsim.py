from unittest import mock, TestCase

import numpy as np
import pytest
import scipy.linalg as la

import pygsti.models as models
from pygsti.forwardsims import ForwardSimulator, \
    MapForwardSimulator, SimpleMapForwardSimulator, \
    MatrixForwardSimulator,  SimpleMatrixForwardSimulator, \
    TorchForwardSimulator
from pygsti.models import ExplicitOpModel
from pygsti.circuits import Circuit, create_lsgst_circuit_lists
from pygsti.baseobjs import Label as L
from ..util import BaseCase

from pygsti.data import simulate_data
from pygsti.modelpacks import smq1Q_XYI
from pygsti.protocols import gst
from pygsti.protocols.protocol import ProtocolData
from pygsti.tools import two_delta_logl


def Ls(*args):
    """ Convert args to a tuple to Labels """
    return tuple([L(x) for x in args])


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
            temp1 = [dict_to_arrays(p) for p in probs]
            temp2 = [t[1].ravel() for t in temp1]
            probs = np.stack(temp2).ravel()
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
            temp1 = [dict_to_arrays(jd) for jd in jac_dicts]
            temp2 = [t[1] for t in temp1]
            jac = np.stack(temp2)
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

    PROBS_TOL = 1e-14
    JACS_TOL = 1e-10

    def setUp(self):
        self.model_ideal = smq1Q_XYI.target_model()
        if TorchForwardSimulator.ENABLED:
            # TorchFowardSimulator can only work with TP modelmembers.
            self.model_ideal.convert_members_inplace(to_type='full TP')
        
        self.model_noisy = self.model_ideal.depolarize(op_noise=0.05, spam_noise=0.025)
        prep_fiducials = smq1Q_XYI.prep_fiducials()
        meas_fiducials = smq1Q_XYI.meas_fiducials()
        germs = smq1Q_XYI.germs()
        max_lengths = [4]
        circuits = create_lsgst_circuit_lists(
            self.model_noisy, prep_fiducials, meas_fiducials, germs, max_lengths
        )[0]
        sims = [
            SimpleMapForwardSimulator(),
            SimpleMatrixForwardSimulator(),
            MapForwardSimulator(),
            MatrixForwardSimulator()
        ]
        if TorchForwardSimulator.ENABLED:
            sims.append(TorchForwardSimulator())
        fsth = ForwardSimTestHelper(self.model_noisy, sims, circuits)
        return fsth
        
    def test_consistent_probs(self):
        fsth = self.setUp()
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
                temp = f"""
                The colinearity of the probabilities from {fsth.sims[idx]} and the reference was {pcl[idx]}.
                """
                msg += temp
            msg += '\n'
            self.assertTrue(False, msg)
        return
    
    def test_consistent_jacs(self):
        fsth = self.setUp()
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
                temp = f"""
                The colinearity of the Jacobian from {fsth.sims[idx]} and the reference was {jcl[idx]}.
                """
                msg += temp
            msg += '\n'
            self.assertTrue(False, msg)
        return


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

