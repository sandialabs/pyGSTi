import unittest
from ..testutils import BaseTestCase

import pygsti
from pygsti.modelpacks import smq1Q_XYI as std
from pygsti.evotypes import qibo as evo_qibo  # don't clobber qibo!


class TestQiboGSTCase(BaseTestCase):
    def setUp(self):
        evo_qibo.densitymx_mode = True
        evo_qibo.minimal_space = 'HilbertSchmidt'  # maybe this should be set automatically?

    def _rungst_comparison(self, ptype):
        mdl_densitymx = std.target_model(ptype, evotype='densitymx', simulator='map')
        mdl_qibo = std.target_model(ptype, evotype='qibo', simulator='map')

        edesign = std.create_gst_experiment_design(1)
        mdl_datagen = std.target_model().depolarize(op_noise=0.05, spam_noise=0.02)
        ds = pygsti.data.simulate_data(mdl_datagen, edesign, 1000, seed=1234)
        data = pygsti.protocols.ProtocolData(edesign, ds)

        proto = pygsti.protocols.GST(mdl_densitymx, gaugeopt_suite=None, optimizer={'maxiter': 100}, verbosity=3)
        results_densitymx = proto.run(data)

        proto = pygsti.protocols.GST(mdl_qibo, gaugeopt_suite=None, optimizer={'maxiter': 3}, verbosity=3)
        results_qibo = proto.run(data)  # profiling this shows that all time is bound up in qibo object construction overhead

        #TODO: verify that results are the approximately the same

    @unittest.skip("Qibo GST is currently too slow to test")
    def test_qibo_gst_fullCPTP(self):
        return self._rungst_comparison('full CPTP')

    @unittest.skip("Qibo GST is currently too slow to test")
    def test_qibo_gst_1plusCPTPLND(self):
        return self._rungst_comparison('1+(CPTPLND)')
