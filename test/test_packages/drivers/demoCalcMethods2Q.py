import os
import unittest

import numpy as np

import pygsti
import pygsti.construction as pc
from pygsti.io import json
from pygsti.modelpacks.legacy import std1Q_XY
from pygsti.modelpacks.legacy import std2Q_XYCNOT as std
from pygsti.objects import Label as L
from ..testutils import BaseTestCase, compare_files


class CalcMethods2QTestCase(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        """
        Handle all once-per-class (slow) computation and loading,
         to avoid calling it for each test (like setUp).  Store
         results in class variable for use within setUp.
        """
        super(CalcMethods2QTestCase, cls).setUpClass()

        #Change to test_packages directory (since setUp hasn't been called yet...)
        origDir = os.getcwd()
        os.chdir(os.path.abspath(os.path.dirname(__file__)))
        os.chdir('..') # The test_packages directory

        #Note: std is a 2Q model
        cls.maxLengths = [1]
        #cls.germs = std.germs_lite
        cls.germs = pygsti.construction.to_circuits([(gl,) for gl in std.target_model().operations])
        cls.mdl_datagen = std.target_model().depolarize(op_noise=0.1, spam_noise=0.001)
        cls.listOfExperiments = pygsti.construction.create_lsgst_circuits(
            std.target_model(), std.prepStrs, std.effectStrs, cls.germs, cls.maxLengths)

        #RUN BELOW FOR DATAGEN (UNCOMMENT to regenerate)
        #ds = pygsti.construction.simulate_data(cls.mdl_datagen, cls.listOfExperiments,
        #                                            n_samples=1000, sample_error="multinomial", seed=1234)
        #ds.save(compare_files + "/calcMethods2Q.dataset")

        cls.ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/calcMethods2Q.dataset")
        cls.advOpts = {'tolerance': 1e-2 }

        #Reduced model GST dataset
        cls.nQubits=2
        cls.mdl_redmod_datagen = pc.build_nqnoise_model(cls.nQubits, geometry="line", max_idle_weight=1, maxhops=1,
                                                    extra_weight_1_hops=0, extra_gate_weight=1, sparse=False, sim_type="matrix", verbosity=1,
                                                    gateNoise=(1234,0.01), prepNoise=(456,0.01), povmNoise=(789,0.01))

        #Create a reduced set of fiducials and germs
        op_labels = list(cls.mdl_redmod_datagen.operations.keys())
        fids1Q = std1Q_XY.fiducials[0:2] # for speed
        cls.redmod_fiducials = []
        for i in range(cls.nQubits):
            cls.redmod_fiducials.extend( pygsti.construction.manipulate_circuits(
                fids1Q, [ ( (L('Gx'),) , (L('Gx',i),) ), ( (L('Gy'),) , (L('Gy',i),) ) ]) )
        #print(redmod_fiducials, "Fiducials")

        cls.redmod_germs = pygsti.construction.to_circuits([(gl,) for gl in op_labels])
        cls.redmod_maxLs = [1]
        expList = pygsti.construction.create_lsgst_circuits(
            cls.mdl_redmod_datagen, cls.redmod_fiducials, cls.redmod_fiducials,
            cls.redmod_germs, cls.redmod_maxLs)

        #RUN BELOW FOR DATAGEN (UNCOMMENT to regenerate)
        #redmod_ds = pygsti.construction.simulate_data(cls.mdl_redmod_datagen, expList, 1000, "round", seed=1234)
        #redmod_ds.save(compare_files + "/calcMethods2Q_redmod.dataset")

        cls.redmod_ds = pygsti.objects.DataSet(file_to_load_from=compare_files + "/calcMethods2Q_redmod.dataset")

        #print(len(expList)," reduced model sequences")

        #Random starting points - little kick so we don't get hung up at start
        np.random.seed(1234)
        cls.rand_start18 = np.random.random(18)*1e-6
        cls.rand_start206 = np.random.random(206)*1e-6
        cls.rand_start228 = np.random.random(228)*1e-6

        os.chdir(origDir) # return to original directory


    ## GST using "full" (non-embedded/composed) gates
    # All of these calcs use dense matrices; While sparse operation matrices (as Maps) could be used,
    # they'd need to enter as a sparse basis to a LindbladDenseOp (maybe add this later?)

    def test_stdgst_matrix(self):
        # Using matrix-based calculations
        target_model = std.target_model().copy()
        target_model.set_all_parameterizations("CPTP")
        target_model.set_simtype('matrix') # the default for 1Q, so we could remove this line
        results = pygsti.run_long_sequence_gst(self.ds, target_model, std.prepStrs, std.effectStrs,
                                               self.germs, self.maxLengths, advanced_options=self.advOpts,
                                               verbosity=4)
        #RUN BELOW LINES TO SAVE GATESET (UNCOMMENT to regenerate)
        #pygsti.io.write_model(results.estimates['default'].models['go0'],
        #                        compare_files + "/test2Qcalc_std_exact.model","Saved Standard-Calc 2Q test model")


        #Note: expected nSigma of 143 is so high b/c we use very high tol of 1e-2 => result isn't very good
        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 143, delta=2.0)
        mdl_compare = pygsti.io.load_model(compare_files + "/test2Qcalc_std_exact.model")
        self.assertAlmostEqual( results.estimates['default'].models['go0'].frobeniusdist(mdl_compare), 0, places=3)


    def test_stdgst_map(self):
        # Using map-based calculation
        target_model = std.target_model().copy()
        target_model.set_all_parameterizations("CPTP")
        target_model.set_simtype('map')
        results = pygsti.run_long_sequence_gst(self.ds, target_model, std.prepStrs, std.effectStrs,
                                               self.germs, self.maxLengths, advanced_options=self.advOpts,
                                               verbosity=4)

        #Note: expected nSigma of 143 is so high b/c we use very high tol of 1e-2 => result isn't very good
        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 143, delta=2.0)
        mdl_compare = pygsti.io.load_model(compare_files + "/test2Qcalc_std_exact.model")
        self.assertAlmostEqual( results.estimates['default'].models['go0'].frobeniusdist(mdl_compare), 0, places=3)


    def test_stdgst_terms(self):
        # Using term-based (path integral) calculation
        # This performs a map-based unitary evolution along each path.
        target_model = std.target_model().copy()
        target_model.set_all_parameterizations("H+S terms")
        target_model.set_simtype('termorder:1') # this is the default set by set_all_parameterizations above
        results = pygsti.run_long_sequence_gst(self.ds, target_model, std.prepStrs, std.effectStrs,
                                               self.germs, self.maxLengths, verbosity=4)

        #RUN BELOW LINES TO SAVE GATESET (UNCOMMENT to regenerate)
        #pygsti.io.json.dump(results.estimates['default'].models['go0'],
        #                    open(compare_files + "/test2Qcalc_std_terms.model",'w'))

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 5, delta=1.0)
        mdl_compare = pygsti.io.json.load(open(compare_files + "/test2Qcalc_std_terms.model"))
        self.assertAlmostEqual( np.linalg.norm(results.estimates['default'].models['go0'].to_vector()
                                               - mdl_compare.to_vector()), 0, places=3)


    # ## GST using "reduced" models
    # Reduced, meaning that we use composed and embedded gates to form a more complex error model with
    # shared parameters and qubit connectivity graphs.  Calculations *can* use dense matrices and matrix calcs,
    # but usually will use sparse mxs and map-based calcs.

    def test_reducedmod_matrix(self):
        # Using dense matrices and matrix-based calcs
        target_model = pc.build_nqnoise_model(self.nQubits, geometry="line", max_idle_weight=1, maxhops=1,
                                             extra_weight_1_hops=0, extra_gate_weight=1, sparse=False,
                                             sim_type="matrix", verbosity=1)
        target_model.from_vector(self.rand_start206)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3})

        #RUN BELOW LINES TO SAVE GATESET (UNCOMMENT to regenerate)
        #pygsti.io.json.dump(results.estimates['default'].models['go0'],
        #                    open(compare_files + "/test2Qcalc_redmod_exact.model",'w'))

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 1.0, delta=1.0)
        mdl_compare = pygsti.io.json.load(open(compare_files + "/test2Qcalc_redmod_exact.model"))
        self.assertAlmostEqual( results.estimates['default'].models['go0'].frobeniusdist(mdl_compare), 0, places=3)


    def test_reducedmod_map1(self):
        # Using dense embedded matrices and map-based calcs (maybe not really necessary to include?)
        target_model = pc.build_nqnoise_model(self.nQubits, geometry="line", max_idle_weight=1, maxhops=1,
                                             extra_weight_1_hops=0, extra_gate_weight=1, sparse=False,
                                             sim_type="map", verbosity=1)
        target_model.from_vector(self.rand_start206)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3})

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 1.0, delta=1.0)
        mdl_compare = pygsti.io.json.load(open(compare_files + "/test2Qcalc_redmod_exact.model"))
        self.assertAlmostEqual( results.estimates['default'].models['go0'].frobeniusdist(mdl_compare), 0, places=1)
          #Note: models aren't necessarily exactly equal given gauge freedoms that we don't know
          # how to optimizize over exactly - so this is a very loose test...



    def test_reducedmod_map2(self):
        # Using sparse embedded matrices and map-based calcs
        target_model = pc.build_nqnoise_model(self.nQubits, geometry="line", max_idle_weight=1, maxhops=1,
                                             extra_weight_1_hops=0, extra_gate_weight=1, sparse=True,
                                             sim_type="map", verbosity=1)
        target_model.from_vector(self.rand_start206)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3})

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 1.0, delta=1.0)
        mdl_compare = pygsti.io.json.load(open(compare_files + "/test2Qcalc_redmod_exact.model"))
        self.assertAlmostEqual( np.linalg.norm(results.estimates['default'].models['go0'].to_vector()
                                               - mdl_compare.to_vector()), 0, places=1)
          #Note: models aren't necessarily exactly equal given gauge freedoms that we don't know
          # how to optimizize over exactly - so this is a very loose test...



    def test_reducedmod_svterm(self):
        # Using term-based calcs using map-based state-vector propagation
        target_model = pc.build_nqnoise_model(self.nQubits, geometry="line", max_idle_weight=1, maxhops=1,
                                      extra_weight_1_hops=0, extra_gate_weight=1, sparse=False, verbosity=1,
                                      sim_type="termorder:1", parameterization="H+S terms")
        target_model.from_vector(self.rand_start228)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3})

        #RUN BELOW LINES TO SAVE GATESET (UNCOMMENT to regenerate)
        #pygsti.io.json.dump(results.estimates['default'].models['go0'],
        #                    open(compare_files + "/test2Qcalc_redmod_terms.model",'w'))

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 3.0, delta=1.0)
        mdl_compare = pygsti.io.json.load(open(compare_files + "/test2Qcalc_redmod_terms.model"))
        self.assertAlmostEqual( np.linalg.norm(results.estimates['default'].models['go0'].to_vector()
                                               - mdl_compare.to_vector()), 0, places=3)


    def test_reducedmod_cterm(self):
        # Using term-based calcs using map-based stabilizer-state propagation
        target_model = pc.build_nqnoise_model(self.nQubits, geometry="line", max_idle_weight=1, maxhops=1,
                                             extra_weight_1_hops=0, extra_gate_weight=1, sparse=False, verbosity=1,
                                             sim_type="termorder:1", parameterization="H+S clifford terms")
        target_model.from_vector(self.rand_start228)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3})

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 3.0, delta=1.0)
        mdl_compare = pygsti.io.json.load(open(compare_files + "/test2Qcalc_redmod_terms.model"))
        self.assertAlmostEqual( np.linalg.norm(results.estimates['default'].models['go0'].to_vector()
                                               - mdl_compare.to_vector()), 0, places=3)

    def test_circuitsim_stabilizer_2Qcheck(self):
        #Test 2Q circuits
        #from pygsti.modelpacks.legacy import std2Q_XYICNOT as stdChk
        from pygsti.modelpacks.legacy import std2Q_XYICPHASE as stdChk

        maxLengths = [1,2,4]
        listOfExperiments = pygsti.construction.create_lsgst_circuits(
            stdChk.target_model(), stdChk.prepStrs, stdChk.effectStrs, stdChk.germs, maxLengths)
        #listOfExperiments = pygsti.construction.to_circuits([ ('Gcnot','Gxi') ])
        #listOfExperiments = pygsti.construction.to_circuits([ ('Gxi','Gcphase','Gxi','Gix') ])

        mdl_normal = stdChk.target_model().copy()
        mdl_clifford = stdChk.target_model().copy()
        #print(mdl_clifford['Gcnot'])
        self.assertTrue(stdChk.target_model()._evotype == "densitymx")
        mdl_clifford.set_all_parameterizations('static unitary') # reduces dim...
        self.assertTrue(mdl_clifford._evotype == "statevec")
        mdl_clifford.set_all_parameterizations('clifford')
        self.assertTrue(mdl_clifford._evotype == "stabilizer")

        for opstr in listOfExperiments:
            #print(str(opstr))
            p_normal = mdl_normal.probabilities(opstr)
            p_clifford = mdl_clifford.probabilities(opstr)
            #p_clifford = bprobs[opstr]
            for outcm in p_normal.keys():
                if abs(p_normal[outcm]-p_clifford[outcm]) > 1e-8:
                    print(str(opstr)," ERR: \n",p_normal,"\n",p_clifford);
                    self.assertTrue(False)
        print("Done checking %d sequences!" % len(listOfExperiments))



if __name__ == "__main__":
    unittest.main(verbosity=2)
