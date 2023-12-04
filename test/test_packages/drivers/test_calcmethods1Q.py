
#quiet down matplotlib!
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import numpy as np
import scipy.linalg as spl
import pygsti
import pygsti.models.modelconstruction as mc
from pygsti.processors.processorspec import QubitProcessorSpec as _ProcessorSpec
from pygsti.modelpacks import smq1Q_XY as std
from pygsti.baseobjs import Label as L
from pygsti.circuits import Circuit
from pygsti.serialization import json
from pygsti.models import ExplicitOpModel, CloudNoiseModel

import os

from ..testutils import BaseTestCase, compare_files, regenerate_references

#Mimics a function that used to be in pyGSTi, replaced with create_cloudnoise_model_from_hops_and_weights
def build_XYCNOT_cloudnoise_model(nQubits, geometry="line", cnot_edges=None,
                                  maxIdleWeight=1, maxSpamWeight=1, maxhops=0,
                                  extraWeight1Hops=0, extraGateWeight=0,
                                  roughNoise=None, simulator="matrix", parameterization="H+S",
                                  spamtype="lindblad", addIdleNoiseToAllGates=True,
                                  errcomp_type="gates", evotype="default", return_clouds=False, verbosity=0):

    availability = {}; nonstd_gate_unitaries = {}
    if cnot_edges is not None: availability['Gcnot'] = cnot_edges
    pspec = _ProcessorSpec(nQubits, ['Gidle', 'Gxpi2','Gypi2','Gcnot'], nonstd_gate_unitaries, availability, geometry)
    assert(spamtype == "lindblad")  # unused and should remove this arg, but should always be "lindblad"
    mdl = mc.create_cloud_crosstalk_model_from_hops_and_weights(
        pspec, None,
        maxIdleWeight, maxSpamWeight, maxhops,
        extraWeight1Hops, extraGateWeight,
        simulator, evotype, parameterization, parameterization,
        "add_global" if addIdleNoiseToAllGates else "none",
        errcomp_type, True, True, True, 'pp', verbosity)

    if return_clouds:
        #FUTURE - just return cloud *keys*? (operation label values are never used
        # downstream, but may still be useful for debugging, so keep for now)
        return mdl, mdl.clouds
    else:
        return mdl


class CalcMethods1QTestCase(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        """
        Handle all once-per-class (slow) computation and loading,
         to avoid calling it for each test (like setUp).  Store
         results in class variable for use within setUp.
        """
        super(CalcMethods1QTestCase, cls).setUpClass()

        #Change to test_packages directory (since setUp hasn't been called yet...)
        origDir = os.getcwd()
        os.chdir(os.path.abspath(os.path.dirname(__file__)))
        os.chdir('..') # The test_packages directory

        #Standard GST dataset
        cls.maxLengths = [1,2,4]
        cls.mdl_datagen = std.target_model().depolarize(op_noise=0.03, spam_noise=0.001)
        cls.listOfExperiments = pygsti.circuits.create_lsgst_circuits(
            std.target_model(), std.prep_fiducials(), std.meas_fiducials(), std.germs(), cls.maxLengths)

        #RUN BELOW FOR DATAGEN (SAVE)
        if regenerate_references():
            ds = pygsti.data.simulate_data(cls.mdl_datagen, cls.listOfExperiments,
                                                   num_samples=1000, sample_error="multinomial", seed=1234)
            ds.save(compare_files + "/calcMethods1Q.dataset")

        #DEBUG TEST- was to make sure data files have same info -- seemed ultimately unnecessary
        #ds_swp = pygsti.objects.DataSet(file_to_load_from=compare_files + "/calcMethods1Q.datasetv3") # run in Python3
        #pygsti.io.write_dataset(temp_files + "/dataset.3to2.txt", ds_swp) # run in Python3
        #ds_swp = pygsti.io.read_dataset(temp_files + "/dataset.3to2.txt") # run in Python2
        #ds_swp.save(compare_files + "/calcMethods1Q.dataset") # run in Python2
        #assert(False),"STOP"

        cls.ds = pygsti.data.DataSet(file_to_load_from=compare_files + "/calcMethods1Q.dataset")

        #Reduced model GST dataset
        cls.nQubits=1 # can't just change this now - see op_labels below
        cls.mdl_redmod_datagen = build_XYCNOT_cloudnoise_model(cls.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                               extraWeight1Hops=0, extraGateWeight=1,
                                                               simulator="matrix", verbosity=1, roughNoise=(1234,0.01))

        #Create a reduced set of fiducials and germs
        op_labels = [ L('Gxpi2',0), L('Gypi2',0) ] # 1Q gate labels
        fids1Q = std.prep_fiducials()[1:2] # for speed, just take 1 non-empty fiducial
        cls.redmod_fiducials = [ Circuit([], line_labels=(0,)) ]  # special case for empty fiducial (need to change line label)
        for i in range(cls.nQubits):
            cls.redmod_fiducials.extend( pygsti.circuits.manipulate_circuits(
                fids1Q, [ ( (L('Gxpi2', 0),) , (L('Gxpi2',i),) ), ( (L('Gypi2',0),) , (L('Gypi2',i),) ) ]) )
        #print(redmod_fiducials, "Fiducials")

        cls.redmod_germs = pygsti.circuits.to_circuits([(gl,) for gl in op_labels])
        cls.redmod_maxLs = [1]
        expList = pygsti.circuits.create_lsgst_circuits(
            op_labels, cls.redmod_fiducials, cls.redmod_fiducials,
            cls.redmod_germs, cls.redmod_maxLs)

        #RUN BELOW FOR DATAGEN (SAVE)
        if regenerate_references():
            redmod_ds = pygsti.data.simulate_data(cls.mdl_redmod_datagen, expList, 1000, "round", seed=1234)
            redmod_ds.save(compare_files + "/calcMethods1Q_redmod.dataset")

        cls.redmod_ds = pygsti.data.DataSet(file_to_load_from=compare_files + "/calcMethods1Q_redmod.dataset")

        #print(len(expList)," reduced model sequences")

        #Random starting points - little kick so we don't get hung up at start
        np.random.seed(1234)
        cls.rand_start18 = np.random.random(18)*1e-6
        cls.rand_start25 = np.random.random(30)*1e-6 # TODO: rename?
        cls.rand_start36 = np.random.random(30)*1e-6 # TODO: rename?

        #Circuit Simulation circuits
        cls.csim_nQubits=3
        cls.circuit1 = pygsti.circuits.Circuit((('Gxpi2',0), ('Gypi2',0)))
          # now Circuit adds qubit labels... pygsti.circuits.Circuit(layer_labels=('Gx','Gy'), num_lines=1) # 1-qubit circuit
        cls.circuit3 = pygsti.circuits.Circuit(layer_labels=[('Gxpi', 0), ('Gypi', 1), ('Gcnot', 1, 2)], num_lines=3) # 3-qubit circuit

        os.chdir(origDir) # return to original directory


    def assert_outcomes(self, probs, expected):
        for k,v in probs.items():
            self.assertAlmostEqual(v, expected[k])

    ## GST using "full" (non-embedded/composed) gates
    # All of these calcs use dense matrices; While sparse operation matrices (as Maps) could be used,
    # they'd need to enter as a sparse basis to a LindbladDenseOp (maybe add this later?)

    def test_stdgst_matrix(self):
        # Using matrix-based calculations
        target_model = std.target_model('CPTPLND')
        target_model.sim = 'matrix' # the default for 1Q, so we could remove this line
        results = pygsti.run_long_sequence_gst(self.ds, target_model, std.prep_fiducials(), std.meas_fiducials(),
                                               std.germs(), self.maxLengths, verbosity=4, disable_checkpointing=True)

        #CHECK that copy gives identical models - this is checked by other
        # unit tests but here we're using a true "GST model" - so do it again:
        print("CHECK COPY")
        mdl = results.estimates[results.name].models['go0']
        mdl_copy = mdl.copy()
        print(mdl.strdiff(mdl_copy))
        self.assertAlmostEqual( mdl.frobeniusdist(mdl_copy), 0, places=2)

        #RUN BELOW LINES TO SAVE GATESET (SAVE)
        if regenerate_references():
            results.estimates[results.name].models['go0'].write(compare_files + "/test1Qcalc_std_exact.json")

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 1.0, delta=2.0)
        mdl_compare = ExplicitOpModel.read(compare_files + "/test1Qcalc_std_exact.json")

        #gauge opt before compare
        gsEstimate = results.estimates[results.name].models['go0'].copy()
        gsEstimate.set_all_parameterizations("full")
        gsEstimate = pygsti.algorithms.gaugeopt_to_target(gsEstimate, mdl_compare)
        print(gsEstimate.strdiff(mdl_compare))
        self.assertAlmostEqual( gsEstimate.frobeniusdist(mdl_compare), 0, places=1)


    def test_stdgst_map(self):
        # Using map-based calculation
        target_model = std.target_model('CPTPLND')
        target_model.sim = 'map'
        results = pygsti.run_long_sequence_gst(self.ds, target_model, std.prep_fiducials(), std.meas_fiducials(),
                                               std.germs(), self.maxLengths, verbosity=4,
                                               disable_checkpointing=True)

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 1.0, delta=2.0)
        mdl_compare = ExplicitOpModel.read(compare_files + "/test1Qcalc_std_exact.json")

        gsEstimate = results.estimates[results.name].models['go0'].copy()
        gsEstimate.set_all_parameterizations("full")
        gsEstimate = pygsti.algorithms.gaugeopt_to_target(gsEstimate, mdl_compare)
        self.assertAlmostEqual( gsEstimate.frobeniusdist(mdl_compare), 0, places=0)
         # with low tolerance (1e-6), "map" tends to go for more iterations than "matrix",
         # resulting in a model that isn't exactly the same as the "matrix" one


    def test_stdgst_terms(self):
        # Using term-based (path integral) calculation
        # This performs a map-based unitary evolution along each path.
        target_model = std.target_model("static unitary", evotype='statevec')
        target_model.set_all_parameterizations("H+S")
        target_model.sim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1)
        target_model._print_gpindices()
        target_model.from_vector(1e-10 * np.ones(target_model.num_params))  # to seed term calc (starting with perfect zeros causes trouble)
        results = pygsti.run_long_sequence_gst(self.ds, target_model, std.prep_fiducials(), std.meas_fiducials(),
                                               std.germs(lite=False), self.maxLengths, verbosity=4,
                                               disable_checkpointing=True)

        #RUN BELOW LINES TO SAVE GATESET (SAVE)
        if regenerate_references():
            results.estimates[results.name].models['go0'].write(compare_files + "/test1Qcalc_std_terms.json")

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 1, delta=2.2)
        mdl_compare = ExplicitOpModel.read(compare_files + "/test1Qcalc_std_terms.json")

        # can't easily gauge opt b/c term-based models can't be converted to "full"
        #mdl_compare.set_all_parameterizations("full")
        #
        #gsEstimate = results.estimates[results.name].models['go0'].copy()
        #gsEstimate.set_all_parameterizations("full")
        #gsEstimate = pygsti.algorithms.gaugeopt_to_target(gsEstimate, mdl_compare)
        #self.assertAlmostEqual( gsEstimate.frobeniusdist(mdl_compare), 0, places=0)

        #A direct vector comparison works if python (&numpy?) versions are identical, but
        # gauge freedoms make this incorrectly fail in other cases - so just check sigmas
        print("VEC DIFF = ",(results.estimates[results.name].models['go0'].to_vector()
                                               - mdl_compare.to_vector()))
        self.assertAlmostEqual( np.linalg.norm(results.estimates[results.name].models['go0'].to_vector()
                                               - mdl_compare.to_vector()), 0, places=1)
        # Note: used to be places=3 above when comparing with cython-built files, but to match cython with
        # non-cython builds we loosen to places=1

    def test_stdgst_prunedpath(self):
        # Using term-based (path integral) calculation with path pruning
        # This performs a map-based unitary evolution along each path.
        target_model = std.target_model("static unitary", evotype='statevec')
        target_model.set_all_parameterizations("H+S")
        target_model.sim = pygsti.forwardsims.TermForwardSimulator(mode='pruned', max_order=3, desired_perr=0.01,
                                                               allowed_perr=0.1, max_paths_per_outcome=1000,
                                                               perr_heuristic='meanscaled', max_term_stages=5)
        target_model.from_vector(1e-10 * np.ones(target_model.num_params))  # to seed term calc (starting with perfect zeros causes trouble)
        results = pygsti.run_long_sequence_gst(self.ds, target_model, std.prep_fiducials(), std.meas_fiducials(),
                                               std.germs(lite=False), self.maxLengths, verbosity=3,
                                               disable_checkpointing=True)

        #RUN BELOW LINES TO SAVE GATESET (SAVE)
        if regenerate_references():
            results.estimates[results.name].models['go0'].write(compare_files + "/test1Qcalc_std_prunedpath.json")

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 1, delta=2.2)
        #mdl_compare = pygsti.io.json.load(open(compare_files + "/test1Qcalc_std_prunedpath.model"))

        # Note: can't easily gauge opt b/c term-based models can't be converted to "full"

        #A direct vector comparison works if python (&numpy?) versions are identical, but
        # gauge freedoms make this incorrectly fail in other cases - so just check sigmas
        #print("VEC DIFF = ",(results.estimates[results.name].models['go0'].to_vector()
        #                                       - mdl_compare.to_vector()))
        #self.assertAlmostEqual( np.linalg.norm(results.estimates[results.name].models['go0'].to_vector()
        #                                       - mdl_compare.to_vector()), 0, places=3)


    # ## GST using "reduced" models
    # Reduced, meaning that we use composed and embedded gates to form a more complex error model with
    # shared parameters and qubit connectivity graphs.  Calculations *can* use dense matrices and matrix calcs,
    # but usually will use sparse mxs and map-based calcs.

    def test_reducedmod_matrix(self):
        # Using dense matrices and matrix-based calcs
        target_model = build_XYCNOT_cloudnoise_model(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                     extraWeight1Hops=0, extraGateWeight=1,
                                                     simulator="matrix", verbosity=1)
        print("Num params = ",target_model.num_params)
        target_model.from_vector(self.rand_start25)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3},
                                               disable_checkpointing=True)

        #RUN BELOW LINES TO SAVE GATESET (SAVE)
        if regenerate_references():
            results.estimates[results.name].models['go0'].write(compare_files + "/test1Qcalc_redmod_exact.json")

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 0.0, delta=1.0)
        #mdl_compare = pygsti.io.json.load( open(compare_files + "/test1Qcalc_redmod_exact.json"))
        #self.assertAlmostEqual( results.estimates[results.name].models['go0'].frobeniusdist(mdl_compare), 0, places=3)
        #NO frobeniusdist for implicit models (yet)

    def test_reducedmod_map1(self):
        # Using dense embedded matrices and map-based calcs (maybe not really necessary to include?)
        target_model = build_XYCNOT_cloudnoise_model(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                     extraWeight1Hops=0, extraGateWeight=1,
                                                     simulator="map", errcomp_type='gates', verbosity=1)
        print("Num params = ",target_model.num_params)
        target_model.from_vector(self.rand_start25)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3},
                                               disable_checkpointing=True)

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 0.0, delta=1.0)
        #mdl_compare = pygsti.io.json.load( open(compare_files + "/test1Qcalc_redmod_exact.json"))
        #self.assertAlmostEqual( results.estimates[results.name].models['go0'].frobeniusdist(mdl_compare), 0, places=1)
        #NO frobeniusdist for implicit models (yet)
          #Note: models aren't necessarily exactly equal given gauge freedoms that we don't know
          # how to optimizize over exactly - so this is a very loose test...


    def test_reducedmod_map1_errorgens(self):
        # Using dense embedded matrices and map-based calcs (same as above)
        # but w/*errcomp_type=errogens* Model (maybe not really necessary to include?)
        target_model = build_XYCNOT_cloudnoise_model(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                     extraWeight1Hops=0, extraGateWeight=1,
                                                     simulator="map", errcomp_type='errorgens', verbosity=1)
        print("Num params = ",target_model.num_params)
        target_model.from_vector(self.rand_start25)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3},
                                               disable_checkpointing=True)

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 0.0, delta=1.0)
        #Note: we don't compare errorgens models to a reference model yet...

    def test_reducedmod_map2(self):
        # Using sparse embedded matrices and map-based calcs
        target_model = build_XYCNOT_cloudnoise_model(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                     extraWeight1Hops=0, extraGateWeight=1,
                                                     simulator="map", errcomp_type='gates', verbosity=1)
        print("Num params = ",target_model.num_params)
        target_model.from_vector(self.rand_start25)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3},
                                               disable_checkpointing=True)

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 0.0, delta=1.0)
        mdl_compare = CloudNoiseModel.read(compare_files + "/test1Qcalc_redmod_exact.json")
        self.assertAlmostEqual( np.linalg.norm(results.estimates[results.name].models['go0'].to_vector()
                                               - mdl_compare.to_vector()), 0, places=1)
          #Note: models aren't necessarily exactly equal given gauge freedoms that we don't know
          # how to optimizize over exactly - so this is a very loose test...


    def test_reducedmod_map2_errorgens(self):
        # Using sparse embedded matrices and map-based calcs (same as above)
        # but w/*errcomp_type=errogens* Model (maybe not really necessary to include?)
        target_model = build_XYCNOT_cloudnoise_model(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                     extraWeight1Hops=0, extraGateWeight=1,
                                                     simulator="map", errcomp_type='errorgens', verbosity=1)
        print("Num params = ",target_model.num_params)
        target_model.from_vector(self.rand_start25)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3},
                                               disable_checkpointing=True)

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 0.0, delta=1.0)
        #Note: we don't compare errorgens models to a reference model yet...


    def test_reducedmod_svterm(self):
        # Using term-based calcs using map-based state-vector propagation
        termsim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1)
        target_model = build_XYCNOT_cloudnoise_model(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                     extraWeight1Hops=0, extraGateWeight=1, evotype="statevec",
                                                     verbosity=1, simulator=termsim, parameterization="H+S",
                                                     errcomp_type='gates')
        print("Num params = ",target_model.num_params)
        target_model.from_vector(self.rand_start36)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3}, 
                                               disable_checkpointing=True)

        #RUN BELOW LINES TO SAVE GATESET (SAVE)
        if regenerate_references():
            results.estimates[results.name].models['go0'].write(compare_files + "/test1Qcalc_redmod_terms.json")

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 0.0, delta=1.0)
        mdl_compare = CloudNoiseModel.read(compare_files + "/test1Qcalc_redmod_terms.json")
        self.assertAlmostEqual( np.linalg.norm(results.estimates[results.name].models['go0'].to_vector()
                                               - mdl_compare.to_vector()), 0, places=3)

    def test_reducedmod_svterm_errorgens(self):
        # Using term-based calcs using map-based state-vector propagation (same as above)
        # but w/errcomp_type=errogens Model
        termsim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1)
        target_model = build_XYCNOT_cloudnoise_model(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                     extraWeight1Hops=0, extraGateWeight=1,
                                                     verbosity=1, evotype="statevec",
                                                     simulator=termsim, parameterization="H+S", errcomp_type='errorgens')
        print("Num params = ",target_model.num_params)
        target_model.from_vector(self.rand_start36)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3},
                                               disable_checkpointing=True)

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 0.0, delta=1.0)
        #Note: we don't compare errorgens models to a reference model yet...

    def test_reducedmod_prunedpath_svterm_errorgens(self):
        termsim = pygsti.forwardsims.TermForwardSimulator(mode='pruned')
        target_model = build_XYCNOT_cloudnoise_model(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                     extraWeight1Hops=0, extraGateWeight=1,
                                                    verbosity=1, evotype="statevec",
                                                     simulator=termsim, parameterization="H+S", errcomp_type='errorgens')

        #separately call set_simtype to set other params
        target_model.sim = pygsti.forwardsims.TermForwardSimulator(mode='pruned', max_order=3, desired_perr=0.01,
                                                               allowed_perr=0.05, max_paths_per_outcome=1000,
                                                               perr_heuristic='none', max_term_stages=5)

        print("Num params = ",target_model.num_params)
        target_model.from_vector(self.rand_start36)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3},
                                               disable_checkpointing=True)

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 0.0, delta=1.0)
        #Note: we don't compare errorgens models to a reference model yet...



    def test_reducedmod_cterm(self):
        # Using term-based calcs using map-based stabilizer-state propagation
        termsim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1)
        target_model = build_XYCNOT_cloudnoise_model(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                     extraWeight1Hops=0, extraGateWeight=1,
                                                     verbosity=1, evotype="stabilizer",
                                                     simulator=termsim, parameterization="H+S", errcomp_type='gates')
        print("Num params = ",target_model.num_params)
        target_model.from_vector(self.rand_start36)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3},
                                               disable_checkpointing=True)

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 0.0, delta=1.0)
        mdl_compare = CloudNoiseModel.read(compare_files + "/test1Qcalc_redmod_terms.json")
        self.assertAlmostEqual( np.linalg.norm(results.estimates[results.name].models['go0'].to_vector()
                                               - mdl_compare.to_vector()), 0, places=1)  #TODO: why this isn't more similar to svterm case??

    def test_reducedmod_cterm_errorgens(self):
        # Using term-based calcs using map-based stabilizer-state propagation (same as above)
        # but w/errcomp_type=errogens Model
        termsim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1)
        target_model = build_XYCNOT_cloudnoise_model(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                     extraWeight1Hops=0, extraGateWeight=1,
                                                     verbosity=1, evotype="stabilizer",
                                                     simulator=termsim, parameterization="H+S", errcomp_type='errorgens')
        print("Num params = ",target_model.num_params)
        target_model.from_vector(self.rand_start36)
        results = pygsti.run_long_sequence_gst(self.redmod_ds, target_model, self.redmod_fiducials,
                                               self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                               verbosity=4, advanced_options={'tolerance': 1e-3},
                                               disable_checkpointing=True)

        print("MISFIT nSigma = ",results.estimates[results.name].misfit_sigma())
        self.assertAlmostEqual( results.estimates[results.name].misfit_sigma(), 0.0, delta=1.0)
        #Note: we don't compare errorgens models to a reference model yet...


    # ### Circuit Simulation

    def test_circuitsim_densitymx(self):
        # Density-matrix simulation (of superoperator gates)
        # These are the typical type of simulations used within GST.
        # The probability calculations can be done in a matrix- or map-based way.

        #Using simple "std" models (which are all density-matrix/superop type)
        mdl = std.target_model()
        probs1 = mdl.probabilities(self.circuit1)
        #self.circuit1.simulate(mdl) # calls probs - same as above line
        print(probs1)

        gs2 = std.target_model()
        gs2.sim = "map"
        probs1 = gs2.probabilities(self.circuit1)
        #self.circuit1.simulate(gs2) # calls probs - same as above line
        print(probs1)
        self.assert_outcomes(probs1, {('0',): 0.5,  ('1',): 0.5} )

        #Using n-qubit models
        pspec = pygsti.processors.QubitProcessorSpec(self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], geometry='line')
        mdl = pygsti.models.modelconstruction.create_crosstalk_free_model(
            pspec, simulator="matrix", ensure_composed_gates=False)
        probs1 = mdl.probabilities(self.circuit3)

        mdl = pygsti.models.modelconstruction.create_crosstalk_free_model(
            pspec, simulator="map", ensure_composed_gates=False)
        probs2 = mdl.probabilities(self.circuit3)

        expected = { ('000',): 0.0,
                     ('001',): 0.0,
                     ('010',): 0.0,
                     ('011',): 0.0,
                     ('100',): 0.0,
                     ('101',): 0.0,
                     ('110',): 0.0,
                     ('111',): 1.0 }
        print(probs1)
        print(probs2)
        self.assert_outcomes(probs1, expected)
        self.assert_outcomes(probs2, expected)

    def test_circuitsim_statevec(self):
        # State-vector simulation (of unitary gates)
        # This can be done with matrix- or map-based calculations.

        #Unitary model in pygsti (from scratch, since "std" modules don't include them)
        sigmax = np.array([[0,1],[1,0]])
        sigmay = np.array([[0,-1.0j],[1.0j,0]])
        sigmaz = np.array([[1,0],[0,-1]])

        def Uop(exp):
            return np.array(spl.expm(-1j * exp/2),complex) # 2x2 unitary matrix operating on single qubit in [0,1] basis

        #Create a model with unitary gates and state vectors (instead of the usual superoperators and density mxs)
        simple_matrix_sim = pygsti.forwardsims.SimpleMatrixForwardSimulator()
        mdl = pygsti.models.ExplicitOpModel([0], evotype='statevec', simulator=simple_matrix_sim)
        mdl.operations[('Gi',0)] = pygsti.modelmembers.operations.StaticUnitaryOp(np.identity(2, 'complex'), evotype='statevec')
        mdl.operations[('Gxpi2',0)] = pygsti.modelmembers.operations.StaticUnitaryOp(Uop(np.pi / 2 * sigmax), evotype='statevec')
        mdl.operations[('Gypi2',0)] = pygsti.modelmembers.operations.StaticUnitaryOp(Uop(np.pi / 2 * sigmay), evotype='statevec')
        mdl.preps['rho0'] = pygsti.modelmembers.states.StaticPureState([1, 0], evotype='statevec')
        mdl.povms['Mdefault'] = pygsti.modelmembers.povms.UnconstrainedPOVM(
            {'0': pygsti.modelmembers.povms.StaticPOVMPureEffect([1, 0], evotype='statevec'),
             '1': pygsti.modelmembers.povms.StaticPOVMPureEffect([0, 1], evotype='statevec')})

        probs1 = mdl.probabilities(self.circuit1)
        #self.circuit1.simulate(mdl) # calls probs - same as above line
        print(probs1)
        self.assert_outcomes(probs1, {('0',): 0.5,  ('1',): 0.5} )

        gs2 = mdl.copy()
        gs2.sim = pygsti.forwardsims.SimpleMapForwardSimulator()
        gs2.probabilities(self.circuit1)
        #self.circuit1.simulate(gs2) # calls probs - same as above line

        #Using n-qubit models
        pspec = pygsti.processors.QubitProcessorSpec(self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], geometry='line')
        mdl = pygsti.models.modelconstruction.create_crosstalk_free_model(
            pspec, evotype="statevec",
            simulator=pygsti.forwardsims.SimpleMatrixForwardSimulator(), ensure_composed_gates=False)
        probs1 = mdl.probabilities(self.circuit3)
        probs2 = mdl.probabilities(self.circuit3)

        expected = { ('000',): 0.0,
                     ('001',): 0.0,
                     ('010',): 0.0,
                     ('011',): 0.0,
                     ('100',): 0.0,
                     ('101',): 0.0,
                     ('110',): 0.0,
                     ('111',): 1.0 }
        print(probs1)
        print(probs2)
        self.assert_outcomes(probs1, expected)
        self.assert_outcomes(probs2, expected)

    def test_circuitsim_svterm(self):
        # ### Density-matrix simulation (of superoperator gates) using map/matrix-based terms calcs
        # In this mode, "term calcs" use many state-vector propagation paths to simulate density
        # matrix propagation up to some desired order (in the assumed-to-be-small error rates).
        #mdl = std.target_model("static unitary", evotype='statevec')
        #mdl.set_all_parameterizations("H+S")
        #mdl.sim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1) # 1st-order in error rates
        #
        #probs1 = mdl.probabilities(self.circuit1)
        ##self.circuit1.simulate(mdl) # calls probs - same as above line
        #
        #print(probs1)
        #self.assert_outcomes(probs1, {('0',): 0.5,  ('1',): 0.5} )

        #Using n-qubit models ("H+S terms" parameterization constructs embedded/composed gates containing LindbladTermGates, etc.)
        termsim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1) # 1st-order in error rates
        pspec = pygsti.processors.QubitProcessorSpec(self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], geometry='line')
        mdl = pygsti.models.modelconstruction.create_crosstalk_free_model(
            pspec, simulator=termsim, ideal_gate_type="H+S", ideal_spam_type="H+S",
            evotype='statevec', ensure_composed_gates=False)

        probs1 = mdl.probabilities(self.circuit3)
        probs2 = self.circuit3.simulate(mdl) # calls probs - same as above line
        print(probs1)
        print(probs2)
        self.assert_outcomes(probs1, { ('000',): 0.0,
                                       ('001',): 0.0,
                                       ('010',): 0.0,
                                       ('011',): 0.0,
                                       ('100',): 0.0,
                                       ('101',): 0.0,
                                       ('110',): 0.0,
                                       ('111',): 1.0 } )
        self.assert_outcomes(probs2, { ('111',): 1.0 } ) # only returns nonzero outcomes by default



    def test_circuitsim_stabilizer(self):
        # Stabilizer-state simulation (of Clifford gates) using map-based calc
        c0 = pygsti.circuits.Circuit(layer_labels=(), num_lines=1) # 1-qubit circuit
        c1 = pygsti.circuits.Circuit(layer_labels=(('Gx', 0),), num_lines=1)
        c2 = pygsti.circuits.Circuit(layer_labels=(('Gx', 0), ('Gx', 0)), num_lines=1)
        c3 = pygsti.circuits.Circuit(layer_labels=(('Gx', 0), ('Gx', 0), ('Gx', 0), ('Gx', 0)), num_lines=1)

        cliffordsim = pygsti.forwardsims.SimpleMapForwardSimulator()
        pspec = pygsti.processors.QubitProcessorSpec(1, ['Gi','Gx','Gy'])
        mdl = pygsti.models.modelconstruction.create_crosstalk_free_model(
            pspec, ensure_composed_gates=False, simulator=cliffordsim, evotype='stabilizer')

        probs0 = mdl.probabilities(c0)
        probs1 = mdl.probabilities(c1)
        probs2 = mdl.probabilities(c2)
        probs3 = mdl.probabilities(c3)

        self.assert_outcomes(probs0, {('0',): 1.0,  ('1',): 0.0} )
        self.assert_outcomes(probs1, {('0',): 0.5,  ('1',): 0.5} )
        self.assert_outcomes(probs2, {('0',): 0.0,  ('1',): 1.0} )
        self.assert_outcomes(probs3, {('0',): 1.0,  ('1',): 0.0} )


    def test_circuitsim_stabilizer_1Qcheck(self):
        from pygsti.modelpacks import smq1Q_XYI as stdChk

        maxLengths = [1,2,4]
        listOfExperiments = pygsti.circuits.create_lsgst_circuits(
            stdChk.target_model(), stdChk.prep_fiducials(), stdChk.meas_fiducials(),
            stdChk.germs(), maxLengths)
        #listOfExperiments = pygsti.circuits.to_circuits([ ('Gcnot','Gxi') ])
        #listOfExperiments = pygsti.circuits.to_circuits([ ('Gxi','Gcphase','Gxi','Gix') ])

        mdl_normal = stdChk.target_model()
        self.assertTrue(mdl_normal._evotype == "densitymx")

        mdl_clifford = stdChk.target_model("static clifford", evotype='stabilizer')
        mdl_clifford.sim = "map"  # only map sim works for stabilizer evotype
        #print(mdl_clifford['Gcnot'])
        #mdl_clifford.set_all_parameterizations('static unitary') # reduces dim...
        #self.assertTrue(mdl_clifford._evotype == "statevec")
        #mdl_clifford.set_all_parameterizations('clifford')
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

    def test_circuitsim_cterm(self):
        # Density-matrix simulation (of superoperator gates) using stabilizer-based term calcs
        c0 = pygsti.circuits.Circuit(layer_labels=(), num_lines=1) # 1-qubit circuit
        c1 = pygsti.circuits.Circuit(layer_labels=(('Gx', 0),), num_lines=1)
        c2 = pygsti.circuits.Circuit(layer_labels=(('Gx', 0), ('Gx', 0)), num_lines=1)
        c3 = pygsti.circuits.Circuit(layer_labels=(('Gx', 0), ('Gx', 0), ('Gx', 0), ('Gx', 0)), num_lines=1)

        termsim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1) # 1st-order in error rates
        pspec = pygsti.processors.QubitProcessorSpec(1, ['Gi','Gx','Gy'])
        mdl = pygsti.models.modelconstruction.create_crosstalk_free_model(
            pspec, simulator=termsim, ideal_gate_type="H+S", ideal_spam_type="H+S",
            ensure_composed_gates=False, evotype="stabilizer")

        probs0 = mdl.probabilities(c0)
        probs1 = mdl.probabilities(c1)
        probs2 = mdl.probabilities(c2)
        probs3 = mdl.probabilities(c3)

        self.assert_outcomes(probs0, {('0',): 1.0,  ('1',): 0.0} )
        self.assert_outcomes(probs1, {('0',): 0.5,  ('1',): 0.5} )
        self.assert_outcomes(probs2, {('0',): 0.0,  ('1',): 1.0} )
        self.assert_outcomes(probs3, {('0',): 1.0,  ('1',): 0.0} )


if __name__ == "__main__":
    unittest.main(verbosity=2)
