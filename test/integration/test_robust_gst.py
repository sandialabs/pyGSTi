import pygsti
from pygsti.modelpacks import smq1Q_XYI, GSTModelPack
from pygsti.protocols import GateSetTomography, GateSetTomographyDesign, ProtocolData
from pygsti.report import construct_standard_report
from pygsti.models import ExplicitOpModel
from pygsti.data.dataset import DataSet
from pygsti.circuits.circuit import Circuit
import numpy as np
import pandas as pd
import unittest
import tempfile


class TestRobustGSTPipeline(unittest.TestCase):

    @staticmethod
    def corrupt_dataset(ds: DataSet, prop_corrupt: float, rng=0) -> tuple[DataSet, list[Circuit]]:
        assert 0 <= prop_corrupt and prop_corrupt <= 1
        dsc = ds.copy_nonstatic()
        rng = np.random.default_rng(rng)
        num_circs = len(dsc)
        selected = rng.choice(np.arange(num_circs), size=int(num_circs*prop_corrupt), replace=False)
        circuits = list(dsc.keys())
        selected = [circuits[i] for i in selected]
        for c in selected:
            num_shots = dsc[c].total
            old_row   = dsc[c].to_dict()
            distn = rng.random(len(old_row))
            distn /= np.sum(distn)
            new_row = {k: num_shots * distn[i] for i,k in enumerate(old_row.keys())}
            dsc[c] = new_row
        dsc.comment  = 'corrupt'
        return dsc, selected

    @staticmethod
    def gst_runner(ds: DataSet, edesign: GateSetTomographyDesign, target_model: ExplicitOpModel, final_objective: str, verbosity: int, mode: str):
        target_model = target_model.copy()
        target_model.convert_members_inplace(mode)
        target_model.default_gauge_group = 'unitary'
        pdata   = ProtocolData(edesign, ds)
        proto   = GateSetTomography(target_model, 'stdgaugeopt', objfn_builders={'objective': final_objective}, name=mode, verbosity=verbosity)
        results = proto.run(pdata, disable_checkpointing=True)
        return results

    def test_pipeline_1Q_XYI(self, generate_report=True):
        """
        Variable naming conventions:
            * Suffix _ori refers to "original" dataset. Only has Markovian error (depolarization).
            * Suffix _cor refers to "corrupted" dataset. A fraction of the circuit outcomes have
              been replaced with out a sample from the uniform distribution.
        """
        mp : GSTModelPack  = smq1Q_XYI  # type: ignore

        target       = mp.target_model()
        circuitlists = mp.create_gst_circuitlists(16)
        edesign      = GateSetTomographyDesign(target.create_processor_spec(), circuitlists, nested=True)
        depol_model  = target.depolarize(op_noise=0.01)
        ds_ori       = pygsti.data.simulate_data(depol_model, circuitlists[-1], num_samples=10_000, seed=0)
        fit_mode     = 'full TP'

        print('##### fits using original data #######')
        results_ori = TestRobustGSTPipeline.gst_runner(ds_ori, edesign, target, 'logl', verbosity=0, mode=fit_mode)
        results_ori.rename_estimate(fit_mode, 'fit-original-logl')
        temp = TestRobustGSTPipeline.gst_runner(ds_ori, edesign, target, 'tvd', verbosity=0, mode=fit_mode)
        temp = temp.estimates.pop(fit_mode)
        temp.parent = None
        results_ori.add_estimate(temp, 'fit-original-tvd')

        print('##### fits using corrupted data #######')
        ds_cor, _   = TestRobustGSTPipeline.corrupt_dataset(ds_ori, prop_corrupt=0.025)
        results_cor = TestRobustGSTPipeline.gst_runner(ds_cor, edesign, target, 'logl', verbosity=0, mode=fit_mode)
        results_cor.rename_estimate(fit_mode, 'fit-corrupted-logl')
        temp = TestRobustGSTPipeline.gst_runner(ds_cor, edesign, target, 'tvd', verbosity=0, mode=fit_mode)
        temp = temp.estimates.pop(fit_mode)
        temp.parent = None
        results_cor.add_estimate(temp, 'fit-corrupted-tvd')

        ##### add test models (fitted w.r.t. corrupted data) to results_ori ######
        mdl = results_cor.estimates['fit-corrupted-logl'].models['stdgaugeopt']
        results_ori.add_model_test(target, mdl, 'fit-corrupted-logl')
        mdl = results_cor.estimates['fit-corrupted-tvd'].models['stdgaugeopt']
        results_ori.add_model_test(target, mdl, 'fit-corrupted-tvd')

        ##### add test models (fitted w.r.t. original data) to results_cor ######
        mdl = results_ori.estimates['fit-original-logl'].models['stdgaugeopt']
        results_cor.add_model_test(target, mdl, 'fit-original-logl')
        mdl = results_ori.estimates['fit-original-tvd'].models['stdgaugeopt']
        results_cor.add_model_test(target, mdl, 'fit-original-tvd')

        def log_likelihood_summaries():
            f_cor = results_cor.estimates['fit-corrupted-logl'].final_objective_fn()
            f_ori = results_ori.estimates['fit-original-logl'].final_objective_fn()

            vals_cor = [(en, f_cor.fn_from_model(e.models['final iteration estimate'])) for en,e in results_cor.estimates.items()  ]
            vals_ori = [(en, f_ori.fn_from_model(e.models['final iteration estimate'])) for en,e in results_cor.estimates.items()  ]

            dfo = pd.DataFrame([
                    [ v  for n,v in vals_cor   if 'original' in n],
                    [ v  for n,v in vals_ori   if 'original' in n]
                ],
                columns=[n for n,_ in vals_cor if 'original' in n], index=['corrupted','original']
            )
            dfo.index.name = 'test dataset'
            dfo = dfo.rename(columns={'fit-original-logl': '-logl', 'fit-original-tvd': 'tvd'})
            
            dfc = pd.DataFrame([
                    [ v  for n,v in vals_cor   if 'corrupted' in n],
                    [ v  for n,v in vals_ori   if 'corrupted' in n]
                ],
                columns=[n for n,_ in vals_cor if 'corrupted' in n], index=['corrupted','original']
            )
            dfc.index.name = 'test dataset'
            dfc = dfc.rename(columns={'fit-corrupted-logl': '-logl', 'fit-corrupted-tvd': 'tvd'})
            return dfo, dfc

        lls_train_ori, lls_train_cor = log_likelihood_summaries()

        test_ratio = lls_train_cor['-logl']['original'] / lls_train_cor['tvd']['original']
        self.assertGreaterEqual(test_ratio, 10)
        # ^ We observe (-logl)/(tvd) == 36.07064855751099; test for >= 10.

        val  = lls_train_ori['tvd']['original'] - lls_train_ori['-logl']['original']
        val /= lls_train_ori['tvd']['original']
        # ^ We observe val == 0.028266922986669028; we test for  0 <= val <= 0.05.
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 0.05)

        if generate_report:
            with tempfile.TemporaryDirectory() as tmpdirname:
                report = construct_standard_report(
                    {'eval-original'  : results_ori, 'eval-corrupted' : results_cor},
                    advanced_options={'skip_sections':
                        ('colorbox', 'input', 'meta', 'help', 'variantraw', 'varianterrorgen')
                    },
                    title="Total variation distance (TVD) GST", verbosity=0
                )
                report.write_html(tmpdirname, auto_open=False, verbosity=0)
                # We only check that we can write the report without error.
        return


if __name__ == '__main__':
    TestRobustGSTPipeline().test_pipeline_1Q_XYI()
    print()
