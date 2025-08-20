

from pygsti.modelpacks import smq1Q_XY
from pygsti.tools.leakage import leaky_qubit_model_from_pspec, construct_leakage_report
from pygsti.data import simulate_data
from pygsti.protocols import StandardGST, ProtocolData
import unittest


class TestLeakageGSTPipeline(unittest.TestCase):
    """
    This is a system-integration smoke test for our leakage modeling abilities.
    """

    def test_smoke(self):
        # This is adapted from the Leakage-automagic ipython notebook.
        mp = smq1Q_XY
        ed = mp.create_gst_experiment_design(max_max_length=32)
        tm3 = leaky_qubit_model_from_pspec(mp.processor_spec(), mx_basis='l2p1')
        ds = simulate_data(tm3, ed.all_circuits_needing_data, num_samples=1000, seed=1997)
        gst = StandardGST( modes=('CPTPLND',), target_model=tm3, verbosity=2)
        pd = ProtocolData(ed, ds)
        res = gst.run(pd)
        _, updated_res = construct_leakage_report(res, title='easy leakage analysis!')
        est = updated_res.estimates['CPTPLND']
        assert 'LAGO' in est.models
        # ^ That's the leakage-aware version of 'stdgaugeopt'
        return
