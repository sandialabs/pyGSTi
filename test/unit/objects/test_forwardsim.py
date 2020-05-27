import numpy as np
from unittest import mock

from ..util import BaseCase

import pygsti.construction as pc
from pygsti.objects import ExplicitOpModel, Label as L
from pygsti.objects.forwardsim import ForwardSimulator


def Ls(*args):
    """ Convert args to a tuple to Labels """
    return tuple([L(x) for x in args])


class AbstractForwardSimTester(BaseCase):
    # XXX is it really neccessary to test an abstract base class?
    def setUp(self):
        mock_SOS = mock.MagicMock()
        mock_SOS.evotype.return_value = "densitymx"  # SOS = Simplified Op Server
        self.fwdsim = ForwardSimulator(4, mock_SOS, np.zeros(16, 'd'))

    def test_construct_evaltree(self):
        with self.assertRaises(NotImplementedError):
            self.fwdsim.construct_evaltree(None, None)

    def test_bulk_fill_probs(self):
        with self.assertRaises(NotImplementedError):
            self.fwdsim.bulk_fill_probs(None, None)

    def test_bulk_fill_dprobs(self):
        with self.assertRaises(NotImplementedError):
            self.fwdsim.bulk_fill_dprobs(None, None)

    def test_bulk_fill_hprobs(self):
        with self.assertRaises(NotImplementedError):
            self.fwdsim.bulk_fill_hprobs(None, None)

    def test_bulk_hprobs_by_block(self):
        with self.assertRaises(NotImplementedError):
            self.fwdsim.bulk_fill_hprobs(None, None)


class ForwardSimBase(object):
    @classmethod
    def setUpClass(cls):
        # XXX can this be constructed directly instead of taking it from a model instance?  EGN: yet, but maybe painful - see model's ._fwdsim()
        ExplicitOpModel._strict = False
        cls.model = pc.create_explicit_model(
            [('Q0',)], ['Gi', 'Gx', 'Gy'],
            ["I(Q0)", "X(pi/8,Q0)", "Y(pi/8,Q0)"]
        )

    def setUp(self):
        self.fwdsim = self.model._fwdsim()
        self.evt, self.lookup, _ = self.model.bulk_evaltree([('Gx',), ('Gx', 'Gx')])
        self.nP = self.model.num_params()
        self.nEls = self.evt.num_final_elements()

    def test_bulk_fill_probs(self):
        pmx = np.empty(self.nEls, 'd')
        self.fwdsim.bulk_fill_probs(pmx, self.evt)
        # TODO assert correctness

    def test_bulk_fill_dprobs(self):
        dmx = np.empty((self.nEls, 3), 'd')
        pmx = np.empty(self.nEls, 'd')
        self.fwdsim.bulk_fill_dprobs(dmx, self.evt, pr_mx_to_fill=pmx, wrt_filter=[0, 1, 2])
        # TODO assert correctness

    def test_bulk_fill_dprobs_with_block_size(self):
        dmx = np.empty((self.nEls, self.nP), 'd')
        self.fwdsim.bulk_fill_dprobs(dmx, self.evt, wrt_block_size=2)
        # TODO assert correctness

    def test_bulk_fill_hprobs(self):
        hmx = np.zeros((self.nEls, 3, 3), 'd')
        dmx = np.zeros((self.nEls, 3), 'd')
        pmx = np.zeros(self.nEls, 'd')
        self.fwdsim.bulk_fill_hprobs(hmx, self.evt,
                                     pr_mx_to_fill=pmx, deriv1_mx_to_fill=dmx, deriv2_mx_to_fill=dmx,
                                     wrt_filter1=[0, 1, 2], wrt_filter2=[0, 1, 2])  # same slice on each deriv
        # TODO assert correctness

        hmx = np.zeros((self.nEls, 3, 2), 'd')
        dmx1 = np.zeros((self.nEls, 3), 'd')
        dmx2 = np.zeros((self.nEls, 2), 'd')
        pmx = np.zeros(self.nEls, 'd')
        self.fwdsim.bulk_fill_hprobs(hmx, self.evt,
                                     pr_mx_to_fill=pmx, deriv1_mx_to_fill=dmx1, deriv2_mx_to_fill=dmx2,
                                     wrt_filter1=[0, 1, 2], wrt_filter2=[2, 3])  # different slices on 1st vs. 2nd deriv
        # TODO assert correctness

    def test_bulk_hprobs_by_block(self):
        # TODO optimize
        mx = np.zeros((self.nEls, self.nP, self.nP), 'd')
        dmx1 = np.zeros((self.nEls, self.nP), 'd')
        dmx2 = np.zeros((self.nEls, self.nP), 'd')
        pmx = np.zeros(self.nEls, 'd')
        self.fwdsim.bulk_fill_hprobs(mx, self.evt, clip_to=(-1, 1),
                                     pr_mx_to_fill=pmx, deriv1_mx_to_fill=dmx1, deriv2_mx_to_fill=dmx2,
                                     wrt_block_size1=2, wrt_block_size2=3)  # use block sizes
        # TODO assert correctness

    def test_prs(self):
        self.fwdsim._prs(L('rho0'), [L('Mdefault_0')], Ls('Gx', 'Gx'), clip_to=(-1, 1))
        self.fwdsim._prs(L('rho0'), [L('Mdefault_0')], Ls('Gx', 'Gx'), clip_to=(-1, 1), use_scaling=True)
        # TODO assert correctness

    def test_estimate_cache_size(self):
        self.fwdsim._estimate_cache_size(100)
        # TODO assert correctness

    def test_estimate_mem_usage(self):
        est = self.fwdsim.estimate_memory_usage(
            ["bulk_fill_probs", "bulk_fill_dprobs", "bulk_fill_hprobs"],
            cache_size=100, num_subtrees=2, num_subtree_proc_groups=1,
            num_param1_groups=1, num_param2_groups=1, num_final_strs=100
        )
        # TODO assert correctness

    def test_estimate_mem_usage_raises_on_bad_subcall_key(self):
        with self.assertRaises(ValueError):
            self.fwdsim.estimate_memory_usage(["foobar"], 1, 1, 1, 1, 1, 1)


class MatrixForwardSimTester(ForwardSimBase, BaseCase):
    def test_doperation(self):
        dg = self.fwdsim._doperation(L('Gx'), flat=False)
        dgflat = self.fwdsim._doperation(L('Gx'), flat=True)
        # TODO assert correctness

    def test_hproduct(self):
        self.fwdsim.hproduct(Ls('Gx', 'Gx'), flat=True, wrt_filter1=[0, 1], wrt_filter2=[1, 2, 3])
        # TODO assert correctness

    def test_hoperation(self):
        hg = self.fwdsim._hoperation(L('Gx'), flat=False)
        hgflat = self.fwdsim._hoperation(L('Gx'), flat=True)
        # TODO assert correctness

    def test_hpr(self):
        self.fwdsim._hpr(Ls('rho0', 'Mdefault_0'), Ls('Gx', 'Gx'), False, False, clip_to=(-1, 1))
        # TODO assert correctness

    def test_rhoE_from_spamTuple(self):
        # XXX does this need to be tested?  EGN: No
        custom_spamTuple = (np.zeros((4, 1), 'd'), np.zeros((4, 1), 'd'))
        self.fwdsim._rho_e_from_spam_tuple(custom_spamTuple)
        # TODO assert correctness


class CPTPMatrixForwardSimTester(MatrixForwardSimTester):
    @classmethod
    def setUpClass(cls):
        super(CPTPMatrixForwardSimTester, cls).setUpClass()
        cls.model = cls.model.copy()
        cls.model.set_all_parameterizations("CPTP")  # so gates have nonzero hessians


class MapForwardSimTester(ForwardSimBase, BaseCase):
    @classmethod
    def setUpClass(cls):
        super(MapForwardSimTester, cls).setUpClass()
        cls.model = cls.model.copy()
        cls.model.set_simtype('map')
