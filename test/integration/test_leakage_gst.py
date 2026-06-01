

from tempfile import TemporaryDirectory
from pygsti.modelpacks import smq1Q_XYI
from pygsti.tools.leakage import leaky_qubit_model_from_pspec, construct_leakage_report
from pygsti.data import simulate_data
from pygsti.protocols import StandardGST, ProtocolData
import unittest
import numpy as np
import scipy.linalg as la


def with_leaky_gate(m, gate_label, strength):
    rng = np.random.default_rng(0)
    v = np.concatenate([[0.0], rng.standard_normal(size=(2,))])
    v /= la.norm(v)
    H = v.reshape((-1, 1)) @ v.reshape((1, -1))
    H *= strength
    U = la.expm(1j*H)
    m_copy = m.copy()
    G_ideal = m_copy.operations[gate_label]
    from pygsti.modelmembers.operations import ComposedOp, StaticUnitaryOp
    m_copy.operations[gate_label] = ComposedOp([G_ideal, StaticUnitaryOp(U, basis=m.basis)])
    return m_copy, v


class TestLeakageGSTPipeline(unittest.TestCase):
    """
    This is a simple system-integration test for our leakage modeling abilities;
    it takes just under 1 minute to run on an Apple M2 Max machine.
    """

    def test_pipeline_1Q_XYI(self):
        # This is adapted from the Leakage-automagic ipython notebook.
        mp = smq1Q_XYI
        ed = mp.create_gst_experiment_design(max_max_length=4)
        # ^ The max length is small so we don't have to wait as long for the GST fit.
        #
        #       GST at L=2 has 168 data params and 161 effective model params;
        #       this is barely outside the overparameterized regime. GST at L=4
        #       (which we use here) has 285 data params and 160 model params.
        #
        tm3 = leaky_qubit_model_from_pspec(mp.processor_spec())
        # ^ Target model.
        dgm3, _ = with_leaky_gate(tm3, ('Gxpi2', 0), strength=0.125)
        # ^ Data generating model. 
        num_samples = 100_000
        # ^ The number of samples is large to compensate for short circuit length.
        from pygsti.objectivefns import objectivefns
        OLD_MIN_PROB_CLIP = objectivefns.DEFAULT_MIN_PROB_CLIP  # restore before exit
        OLD_RADIUS        = objectivefns.DEFAULT_RADIUS         # restore before exit
        objectivefns.DEFAULT_MIN_PROB_CLIP = objectivefns.DEFAULT_RADIUS = 1e-12
        # ^ The lines above change numerical thresholding rules in objective evaluation
        #   to be appropriate when the number of shots/circuit is extremely large.
        ds = simulate_data(dgm3, ed.all_circuits_needing_data, num_samples=num_samples, seed=1997)
        gst = StandardGST(
            modes=('CPTPLND',), target_model=tm3, verbosity=4,
            badfit_options={'actions': ['wildcard1d'], 'threshold': 0.0}
        )
        pd = ProtocolData(ed, ds)
        res = gst.run(pd)
        kwargs_stdreport = {
            'advanced_options' : {
                'show_unmodeled_error' : True,
                'skip_sections'        : ['colorbox', 'input', 'meta', 'help']
            },
        }
        report, updated_res = construct_leakage_report(res, title='easy leakage analysis!', kwargs_stdreport=kwargs_stdreport)
        # ^ updated_res contains the leakage-aware gauge optimization model.
        with TemporaryDirectory() as dirname:
            report.write_html(dirname)
            # ^ That update makes sure the report generates without error.
        est = updated_res.estimates['CPTPLND']

        """
        Original results are shown below. We don't rely on the exact numbers here. What matters is
        qualitative aspects of how Gxpi2 and Gypi2 deviate from their respective targets. Since our
        data generating model only applied leakage to Gxpi2, a "good" result reports much more error
        in Gxpi2 than Gypi2. (It's not clear to me why stdgaugeopt lacks wildcard error.)

        Leakage-aware guage optimization.

            | Gate    | ent. infidelity | 1/2 trace dist | 1/2 diamond dist | Max TOP  | Unmodeled error |
            |---------|-----------------|----------------|------------------|----------|-----------------|
            | []      | 0.000009        | 0.002639   	 | 0.003724	        | 0.000014 | 0.00007         |
            | Gxpi2:0 | 0.000965        | 0.030946       | 0.040751	        | 0.001629 | 0.000764        |
            | Gypi2:0 | 0.000375        | 0.019367       | 0.025308         | 0.000531 | 0.000475        |

        
        Standard gauge optimization

            | Gate    | ent. infidelity | 1/2 trace dist | 1/2 diamond dist | Max TOP  |
            |---------|-----------------|----------------|------------------|----------|
            | []      | 0.000016        | 0.003644       | 0.005146         | 0.000026 |
            | Gxpi2:0 | 0.000620        | 0.024758       | 0.032616         | 0.001000 |
            | Gypi2:0 | 0.000611        | 0.024714       | 0.033323         | 0.001001 |

        We'll run tests with subspace entanglement infidelity.

            * For LAGO, infidelity of Gxpi2 is 2.57x larger than that of Gypi2;
              we'll test for a 2x difference.
            
            * For standard gauge optimization, Gxpi2 and Gypi2 have almost identical infidelities;
              we'll test for a factor 1.1x there.
        """
        from pygsti.tools.leakage import subspace_entanglement_fidelity as fidelity
        
        mdls = {lbl: est.models[lbl] for lbl in {'target', 'LAGO', 'stdgaugeopt'}}
        assert mdls['target'].basis.name == mdls['stdgaugeopt'].basis.name == mdls['LAGO'].basis.name == 'l2p1'

        gates = dict()
        for lbl, mdl in mdls.items():
            gates[lbl] = {g: mdl.operations[(f'G{g}pi2',0)].to_dense() for g in ['x', 'y']}

        infids = dict()
        for lbl in ['LAGO', 'stdgaugeopt']:
            infids[lbl] = {g: 1 - fidelity(gates[lbl][g], gates['target'][g], 'l2p1', n_leak=1) for g in ['x', 'y'] } 

        self.assertGreater( infids['LAGO']['x'],        2.0 * infids['LAGO']['y']        )
        self.assertLess(    infids['stdgaugeopt']['x'], 1.1 * infids['stdgaugeopt']['y'] )

        objectivefns.DEFAULT_MIN_PROB_CLIP = OLD_MIN_PROB_CLIP
        objectivefns.DEFAULT_RADIUS = OLD_RADIUS
        return


if __name__ == '__main__':
    TestLeakageGSTPipeline().test_pipeline_1Q_XYI()
    print()
