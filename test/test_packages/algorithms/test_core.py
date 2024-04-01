import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import pygsti

import numpy as np

from .basecase import AlgorithmsBase

class TestCoreMethods(AlgorithmsBase):
    def test_LGST(self):

        ds = self.ds

        mdl_lgst = pygsti.run_lgst(ds, self.fiducials, self.fiducials, self.model, svd_truncate_to=4, verbosity=0)
        mdl_lgst_verb = self.runSilent(pygsti.run_lgst, ds, self.fiducials, self.fiducials, self.model, svd_truncate_to=4, verbosity=10)
        self.assertAlmostEqual(mdl_lgst.frobeniusdist(mdl_lgst_verb),0)

        mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst, self.model, {'spam':1.0, 'gates': 1.0}, check_jac=True)
        mdl_clgst = pygsti.contract(mdl_lgst_go, "CPTP")


    def test_LGST_no_sample_error(self):
        #change rep-count type so dataset can hold fractional counts for sampleError = 'none'
        oldType = pygsti.data.dataset.Repcount_type
        pygsti.data.dataset.Repcount_type = np.float64
        ds = pygsti.data.simulate_data(self.datagen_gateset, self.lgstStrings,
                                               num_samples=10000, sample_error='none')
        pygsti.data.dataset.Repcount_type = oldType

        mdl_lgst = pygsti.run_lgst(ds, self.fiducials, self.fiducials, self.model, svd_truncate_to=4, verbosity=0)
        mdl_lgst = pygsti.gaugeopt_to_target(mdl_lgst, self.datagen_gateset, {'spam':1.0, 'gates': 1.0}, check_jac=False)
        self.assertAlmostEqual( mdl_lgst.frobeniusdist(self.datagen_gateset), 0, places=4)

    def test_LGST_1overSqrtN_dependence(self):
        my_datagen_gateset = self.model.depolarize(op_noise=0.05, spam_noise=0)
        # !!don't depolarize spam or 1/sqrt(N) dependence saturates!!
        nSamplesList = np.array([ 16, 128, 1024, 8192 ])
        diffs = []
        for nSamples in nSamplesList:
            ds = pygsti.data.simulate_data(my_datagen_gateset, self.lgstStrings, nSamples,
                                                   sample_error='binomial', seed=100)
            mdl_lgst = pygsti.run_lgst(ds, self.fiducials, self.fiducials, self.model, svd_truncate_to=4, verbosity=0)
            mdl_lgst_go = pygsti.gaugeopt_to_target(mdl_lgst, my_datagen_gateset, {'spam':1.0, 'gate': 1.0}, check_jac=True)
            diffs.append( my_datagen_gateset.frobeniusdist(mdl_lgst_go) )

        diffs = np.array(diffs, 'd')
        p = np.polyfit(np.log10(nSamplesList), np.log10(diffs), deg=1)
        a = p[0]
        b = p[1]
        #print "\n",nSamplesList; print diffs; print a #DEBUG
        self.assertLess( a+0.5, 0.05 )

    def test_miscellaneous(self):
        self.runSilent(self.model.print_info) #just make sure it works

if __name__ == "__main__":
    unittest.main(verbosity=2)
