from ..testutils import BaseTestCase, compare_files, temp_files
import unittest
import numpy as np

import pygsti
#from pygsti.extras import rb
from pygsti.objects import Label

class RBTestCase(BaseTestCase):
    @unittest.skip("Need to update RB unit tests since major code update")
    def test_rb_io_results_and_analysis(self):

        # Just checks that we can succesfully import the standard data type.
        data = rb.io.import_rb_summary_data([compare_files + '/rb_io_test.txt',])
        # This is a basic test that the imported dataset makes sense : we can
        # successfully run the analysis on it.
        out = rb.analysis.std_practice_analysis(data,bootstrap_samples=100)
        # Checks plotting works. This requires matplotlib, so should do a try/except

        from pygsti.report import workspace
        w = workspace.Workspace()
        #w.init_notebook_mode(connected=False)
        plt = w.RandomizedBenchmarkingPlot(out)

        # TravisCI doesn't install matplotlib
        #plt.saveas(temp_files + "/rbdecay_plot.pdf")
        #out.plot() # matplotlib version (keep around for now)
        return


    @unittest.skip("Need to update RB unit tests since major code update")
    def test_rb_simulate(self):
        n = 3
        glist = ['Gxpi','Gypi','Gzpi','Gh','Gp','Gcphase'] # 'Gi',
        availability = {'Gcphase':[(0,1),(1,2)]}
        pspec = pygsti.obj.ProcessorSpec(n,glist,availability=availability,verbosity=0)

        errormodel = rb.simulate.create_iid_pauli_error_model(pspec, oneQgate_errorrate=0.01, twoQgate_errorrate=0.05,
                                                              idle_errorrate=0.005, measurement_errorrate=0.05,
                                                              ptype='uniform')
        errormodel = rb.simulate.create_iid_pauli_error_model(pspec, oneQgate_errorrate=0.001, twoQgate_errorrate=0.01,
                                                              idle_errorrate=0.005, measurement_errorrate=0.05,
                                                              ptype='X')

        out = rb.simulate.rb_with_pauli_errors(pspec,errormodel,[0,2,4],2,3,filename=temp_files + '/simtest_CRB.txt',rbtype='CRB',
                                        returndata=True, verbosity=0)

        errormodel = rb.simulate.create_locally_gate_independent_pauli_error_model(pspec, {0: 0.0, 1: 0.01, 2: 0.02},
                                                                                   {0: 0.0, 1: 0.1, 2: 0.01},ptype='uniform')

        out = rb.simulate.rb_with_pauli_errors(pspec,errormodel,[0,10,20],2,2,filename=temp_files + '/simtest_DRB.txt',rbtype='DRB',
                                        returndata=True, verbosity=0)

    @unittest.skip("Need to update RB unit tests since major code update")
    def test_clifford_compilations(self):

        # Tests the Clifford compilations hard-coded into the various std models. Perhaps this can be
        # automated to run over all the std models that contain a Clifford compilation?

        from pygsti.modelpacks.legacy import std1Q_Cliffords
        target_model = std1Q_Cliffords.target_model()
        clifford_group = rb.group.construct_1q_clifford_group()

        from pygsti.modelpacks.legacy import std1Q_XY
        target_model = std1Q_XY.target_model()
        clifford_compilation = std1Q_XY.clifford_compilation
        compiled_cliffords = pygsti.construction.create_explicit_alias_model(target_model,clifford_compilation)

        for key in list(compiled_cliffords.operations.keys()):
            self.assertLess(np.sum(abs(compiled_cliffords.operations[key]-clifford_group.matrix(key))), 10**(-10))

        from pygsti.modelpacks.legacy import std1Q_XYI
        target_model = std1Q_XYI.target_model()
        clifford_compilation = std1Q_XYI.clifford_compilation
        compiled_cliffords = pygsti.construction.create_explicit_alias_model(target_model,clifford_compilation)

        for key in list(compiled_cliffords.operations.keys()):
            self.assertLess(np.sum(abs(compiled_cliffords.operations[key]-clifford_group.matrix(key))), 10**(-10))

        # Future : add the remaining Clifford compilations here.



if __name__ == '__main__':
    unittest.main(verbosity=2)
