# XXX rewrite or remove

from unittest import mock

import numpy as np

import pygsti.models as models
from pygsti.forwardsims.forwardsim import ForwardSimulator
from pygsti.forwardsims.mapforwardsim import MapForwardSimulator
from pygsti.models import ExplicitOpModel
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label as L
from ..util import BaseCase


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

    def setUp(self):
        self.fwdsim = self.model.sim
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
        dg = self.fwdsim._doperation(L('Gx'), flat=False)
        dgflat = self.fwdsim._doperation(L('Gx'), flat=True)
        # TODO assert correctness

    def test_hoperation(self):
        hg = self.fwdsim._hoperation(L('Gx'), flat=False)
        hgflat = self.fwdsim._hoperation(L('Gx'), flat=True)
        # TODO assert correctness


class CPTPMatrixForwardSimTester(MatrixForwardSimTester):
    @classmethod
    def setUpClass(cls):
        super(CPTPMatrixForwardSimTester, cls).setUpClass()
        cls.model = cls.model.copy()
        cls.model.set_all_parameterizations("CPTPLND")  # so gates have nonzero hessians


class MapForwardSimTester(ForwardSimBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        super(MapForwardSimTester, cls).setUpClass()
        cls.model = cls.model.copy()
        cls.model.sim = MapForwardSimulator()
