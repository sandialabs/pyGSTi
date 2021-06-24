import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import pygsti

import numpy as np
from scipy import polyfit

from ..testutils import compare_files, regenerate_references
from .basecase import AlgorithmsBase

class TestCoreMethods(AlgorithmsBase):
    def test_LGST(self):

        ds = self.ds

        print("GG0 = ",self.model.default_gauge_group)
        mdl_lgst = pygsti.run_lgst(ds, self.fiducials, self.fiducials, self.model, svd_truncate_to=4, verbosity=0)
        mdl_lgst_verb = self.runSilent(pygsti.run_lgst, ds, self.fiducials, self.fiducials, self.model, svd_truncate_to=4, verbosity=10)
        self.assertAlmostEqual(mdl_lgst.frobeniusdist(mdl_lgst_verb),0)

        print("GG = ",mdl_lgst.default_gauge_group)
        mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst, self.model, {'spam':1.0, 'gates': 1.0}, check_jac=True)
        mdl_clgst = pygsti.contract(mdl_lgst_go, "CPTP")

        # RUN BELOW LINES TO SEED SAVED GATESET FILES
        if regenerate_references():
            pygsti.io.write_model(mdl_lgst, compare_files + "/lgst.model", "Saved LGST Model before gauge optimization")
            pygsti.io.write_model(mdl_lgst_go, compare_files + "/lgst_go.model", "Saved LGST Model after gauge optimization")
            pygsti.io.write_model(mdl_clgst, compare_files + "/clgst.model", "Saved LGST Model after G.O. and CPTP contraction")

        mdl_lgst_compare = pygsti.io.load_model(compare_files + "/lgst.model")
        mdl_lgst_go_compare = pygsti.io.load_model(compare_files + "/lgst_go.model")
        mdl_clgst_compare = pygsti.io.load_model(compare_files + "/clgst.model")

        self.assertAlmostEqual( mdl_lgst.frobeniusdist(mdl_lgst_compare), 0, places=5)
        self.assertAlmostEqual( mdl_lgst_go.frobeniusdist(mdl_lgst_go_compare), 0, places=5)
        self.assertAlmostEqual( mdl_clgst.frobeniusdist(mdl_clgst_compare), 0, places=5)

    def test_LGST_no_sample_error(self):
        #change rep-count type so dataset can hold fractional counts for sampleError = 'none'
        oldType = pygsti.datasets.dataset.Repcount_type
        pygsti.datasets.dataset.Repcount_type = np.float64
        ds = pygsti.construction.simulate_data(self.datagen_gateset, self.lgstStrings,
                                               num_samples=10000, sample_error='none')
        pygsti.datasets.dataset.Repcount_type = oldType

        mdl_lgst = pygsti.run_lgst(ds, self.fiducials, self.fiducials, self.model, svd_truncate_to=4, verbosity=0)
        print("DATAGEN:")
        print(self.datagen_gateset)
        print("\nLGST RAW:")
        print(mdl_lgst)
        mdl_lgst = pygsti.gaugeopt_to_target(mdl_lgst, self.datagen_gateset, {'spam':1.0, 'gates': 1.0}, check_jac=False)
        print("\nAfter gauge opt:")
        print(mdl_lgst)
        print(mdl_lgst.strdiff(self.datagen_gateset))
        self.assertAlmostEqual( mdl_lgst.frobeniusdist(self.datagen_gateset), 0, places=4)

    def test_LGST_1overSqrtN_dependence(self):
        my_datagen_gateset = self.model.depolarize(op_noise=0.05, spam_noise=0)
        # !!don't depolarize spam or 1/sqrt(N) dependence saturates!!

        nSamplesList = np.array([ 16, 128, 1024, 8192 ])
        diffs = []
        for nSamples in nSamplesList:
            ds = pygsti.construction.simulate_data(my_datagen_gateset, self.lgstStrings, nSamples,
                                                   sample_error='binomial', seed=100)
            mdl_lgst = pygsti.run_lgst(ds, self.fiducials, self.fiducials, self.model, svd_truncate_to=4, verbosity=0)
            mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst, my_datagen_gateset, {'spam':1.0, 'gate': 1.0}, check_jac=True)
            diffs.append( my_datagen_gateset.frobeniusdist(mdl_lgst_go) )

        diffs = np.array(diffs, 'd')
        a, b = polyfit(np.log10(nSamplesList), np.log10(diffs), deg=1)
        #print "\n",nSamplesList; print diffs; print a #DEBUG
        self.assertLess( a+0.5, 0.05 )

    def test_miscellaneous(self):
        self.runSilent(self.model.print_info) #just make sure it works

if __name__ == "__main__":
    unittest.main(verbosity=2)
