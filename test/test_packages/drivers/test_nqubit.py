import unittest
import pygsti
import pytest
import numpy as np
from pygsti.modelpacks.legacy import std1Q_XY
from pygsti.baseobjs import Label as L
from pygsti.circuits import Circuit
from pygsti.tools.exceptions import pyGSTiDeprecationWarning
import warnings

from ..testutils import BaseTestCase, compare_files, regenerate_references, build_XYCNOT_cloudnoise_model
from pygsti.models import modelconstruction
from pygsti.circuits import cloudcircuitconstruction
from pygsti.circuits.circuitstructure import PlaquetteGridCircuitStructure


class NQubitTestCase(BaseTestCase):

    def setUp(self):
        super(NQubitTestCase, self).setUp()

    def test_construction(self):
        mdl_test = build_XYCNOT_cloudnoise_model(
            nQubits=1, geometry="line", maxIdleWeight=1, maxhops=0, verbosity=0)
        mdl_test = build_XYCNOT_cloudnoise_model(
            nQubits=2, geometry="line", maxIdleWeight=1, maxhops=0, verbosity=0)
        mdl_test = build_XYCNOT_cloudnoise_model(
            nQubits=3, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1, simulator="map", verbosity=0)
        #                                    roughNoise=(1234,0.1))

        #print("Constructed model with %d gates, dim=%d, and n_params=%d.  Norm(paramvec) = %g" %
        #      (len(mdl_test.operations),mdl_test.dim,mdl_test.num_params, np.linalg.norm(mdl_test.to_vector()) ))

    def test_sequential_sequenceselection(self):

        nQubits = 2
        maxLengths = [1,2]
        cnot_edges = [(i,i+1) for i in range(nQubits-1)] #only single direction

        mdl_datagen = build_XYCNOT_cloudnoise_model(nQubits, "line", cnot_edges, maxIdleWeight=2, maxhops=1,
                                                    extraWeight1Hops=0, extraGateWeight=0,
                                                    verbosity=0,
                                                    simulator="matrix", parameterization="H+S",
                                                    roughNoise=(1234,0.01))

        cache = {}
        with pytest.warns(pyGSTiDeprecationWarning):
            # _create_xycnot_cloudnoise_circuits is deprecated, but the replacement
            # function, create_cloudnoise_circuits, is not a 1-to-1 match. Just suppressing
            # this warning for now. Will need to go back and add a test for create_cloudnoise_circuits
            # in the future.
            gss = cloudcircuitconstruction._create_xycnot_cloudnoise_circuits(
                nQubits, maxLengths, 'line', cnot_edges, max_idle_weight=2, maxhops=1,
                extra_weight_1_hops=0, extra_gate_weight=0, verbosity=0, cache=cache, algorithm="sequential")
        expList = list(gss) #[ tup[0] for tup in expList_tups]
        
        #RUN to SAVE list & dataset
        if regenerate_references():
            gss.write(compare_files + "/nqubit_2Q_seqs.json")
            ds = pygsti.data.simulate_data(mdl_datagen, expList, 10000, "multinomial", seed=1234)
            pygsti.io.write_dataset(compare_files + "/nqubit_2Q.dataset", ds)

        compare_gss = PlaquetteGridCircuitStructure.read(compare_files + "/nqubit_2Q_seqs.json")

        self.assertEqual(set(gss), set(compare_gss))

    def test_greedy_sequenceselection(self):
        nQubits = 1
        maxLengths = [1,2]
        cnot_edges = []

        cache = {}
        with pytest.warns(pyGSTiDeprecationWarning):
            # _create_xycnot_cloudnoise_circuits is deprecated, but the replacement
            # function, create_cloudnoise_circuits, is not a 1-to-1 match. Just suppressing
            # this warning for now. Will need to go back and add a test for create_cloudnoise_circuits
            # in the future.
            gss = cloudcircuitconstruction._create_xycnot_cloudnoise_circuits(
                nQubits, maxLengths, 'line', cnot_edges, max_idle_weight=1, maxhops=0,
                extra_weight_1_hops=0, extra_gate_weight=0, verbosity=4, cache=cache, algorithm="greedy")
        #expList = gss.allstrs #[ tup[0] for tup in expList_tups]

        #RUN to SAVE list
        if regenerate_references():
            gss.write(compare_files + "/nqubit_1Q_seqs.json")

        compare_gss = PlaquetteGridCircuitStructure.read(compare_files + "/nqubit_1Q_seqs.json")

        self.assertEqual(set(gss), set(compare_gss))

    def test_2Q(self):

        gss = PlaquetteGridCircuitStructure.read(compare_files + "/nqubit_2Q_seqs.json")
        expList = list(gss)

        ds = pygsti.io.read_dataset(compare_files + "/nqubit_2Q.dataset")

        nQubits = 2
        maxLengths = [1] #,2]
        cnot_edges = [(i,i+1) for i in range(nQubits-1)] #only single direction

        #OLD
        #lsgstLists = []; lst = []
        #for L in maxLengths:
        #    for tup in expList_tups:
        #        if tup[1] == L: lst.append( tup[0] )
        #    lsgstLists.append(lst[:]) # append *running* list
        lsgstLists = gss.truncate(xs_to_keep=maxLengths) # can just use gss as input to pygsti.run_long_sequence_gst_base

        mdl_to_optimize = build_XYCNOT_cloudnoise_model(nQubits, "line", cnot_edges, maxIdleWeight=2, maxhops=1,
                                                        extraWeight1Hops=0, extraGateWeight=1, verbosity=1,
                                                        simulator="matrix", parameterization="H+S")
                                                        #switching the to matrix forward simulator made the tests run way faster it seems.
        results = pygsti.run_long_sequence_gst_base(ds, mdl_to_optimize,
                                                    lsgstLists, gauge_opt_params=False,
                                                    advanced_options={'tolerance': 1e-1, 'max_iterations': 5}, verbosity=0,
                                                    disable_checkpointing= True,
                                                    gauge_opt_suite_name='none') #probably don't care about convergence for same reason we
                                                    #don't for the 3Q case?

    def test_2Q_terms(self):

        gss = PlaquetteGridCircuitStructure.read(compare_files + "/nqubit_2Q_seqs.json")
        expList = list(gss)

        ds = pygsti.io.read_dataset(compare_files + "/nqubit_2Q.dataset")

        nQubits = 2
        maxLengths = [1,2]
        cnot_edges = [(i,i+1) for i in range(nQubits-1)] #only single direction

        #OLD
        #lsgstLists = []; lst = []
        #for L in maxLengths:
        #    for tup in expList_tups:
        #        if tup[1] == L: lst.append( tup[0] )
        #    lsgstLists.append(lst[:]) # append *running* list
        lsgstLists = gss # can just use gss as input to pygsti.run_long_sequence_gst_base

        termsim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1)
        mdl_to_optimize = build_XYCNOT_cloudnoise_model(nQubits, "line", cnot_edges, maxIdleWeight=2, maxhops=1,
                                                        extraWeight1Hops=0, extraGateWeight=1, verbosity=1,
                                                        simulator=termsim, parameterization="H+S", evotype='statevec')

        mdl_to_optimize.sim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1)

        results = pygsti.run_long_sequence_gst_base(ds, mdl_to_optimize,
                                                    lsgstLists, gauge_opt_params=False,
                                                    advanced_options={'tolerance': 1e-3}, verbosity=0,
                                                    disable_checkpointing= True,
                                                    gauge_opt_suite_name='none')

    def test_3Q(self):

        nQubits = 3
        target_model = build_XYCNOT_cloudnoise_model(
            nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1, simulator="map",verbosity=1)
        #print("nElements test = ",target_model.num_elements)
        #print("nParams test = ",target_model.num_params)
        #print("nNonGaugeParams test = ",target_model.num_nongauge_params)

        mdl_datagen = build_XYCNOT_cloudnoise_model(
            nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1, verbosity=0, roughNoise=(1234,0.1), simulator="matrix")

        mdl_test = mdl_datagen

        op_labels = target_model.primitive_op_labels
        line_labels = tuple(range(nQubits))
        fids1Q = std1Q_XY.fiducials
        fiducials = []
        for i in range(nQubits):
            fiducials.extend( pygsti.circuits.manipulate_circuits(
                fids1Q, [ ( (L('Gx'),) , (L('Gx',i),) ), ( (L('Gy'),) , (L('Gy',i),) ) ], line_labels=line_labels) )
        prep_fiducials = meas_fiducials = fiducials
        #TODO: add fiducials for 2Q pairs (edges on graph)

        germs = pygsti.circuits.to_circuits([(gl,) for gl in op_labels], line_labels=line_labels)
        maxLs = [1]
        expList = pygsti.circuits.create_lsgst_circuits(mdl_datagen, prep_fiducials, meas_fiducials, germs, maxLs)
        self.assertTrue( Circuit((),line_labels) in expList)

        ds = pygsti.data.simulate_data(mdl_datagen, expList, 1000, "multinomial", seed=1234)

        logL = pygsti.tools.logl(mdl_datagen, ds, expList)
        max_logL = pygsti.tools.logl_max(mdl_datagen, ds, expList)
        twoDeltaLogL = 2*(max_logL-logL)
        chi2 = pygsti.tools.chi2(mdl_datagen, ds, expList)

        dof = ds.degrees_of_freedom()
        nParams = mdl_datagen.num_params
        #print("EXIT"); exit()
        return

        results = pygsti.run_long_sequence_gst(ds, target_model, prep_fiducials, meas_fiducials, germs, maxLs, verbosity=5,
                                               advanced_options={'max_iterations': 2}) #keep this short; don't care if it doesn't converge.



    def test_SPAM(self):
        nQubits = 3
        factorPOVMs = []
        basis1Q = pygsti.baseobjs.Basis.cast("pp", 4)
        basisNQ = pygsti.baseobjs.Basis.cast("pp", 4 ** nQubits)
        for i in range(nQubits):
            effects = [(l, modelconstruction.create_spam_vector(l, "Q0", basis1Q)) for l in ["0", "1"]]
            factorPOVMs.append(pygsti.modelmembers.povms.TPPOVM(effects, evotype='default'))
        povm = pygsti.modelmembers.povms.TensorProductPOVM(factorPOVMs)

        v = povm.to_vector()
        v += np.random.random( len(v) )
        povm.from_vector(v)

        mdl = pygsti.models.ExplicitOpModel(['Q0', 'Q1', 'Q2'])
        prepFactors = [pygsti.modelmembers.states.TPState(modelconstruction.create_spam_vector("0", "Q0", basis1Q))
                       for i in range(nQubits)]
        mdl.preps['rho0'] = pygsti.modelmembers.states.TensorProductState(prepFactors, mdl.state_space)
        # OR one big prep: mdl.preps['rho0'] = modelconstruction.create_spam_vector("0", basisNQ)

        print("Before adding to model:")
        print(" povm.gpindices = ",povm.gpindices, "parent is None?", bool(povm.parent is None))
        for i,fpovm in enumerate(povm.factorPOVMs):
            print(" factorPOVM%d.gpindices = " % i, fpovm.gpindices, "parent is None?", bool(fpovm.parent is None))
        for lbl,effect in povm.simplify_effects().items():
            print(" compiled[%s].gpindices = " % lbl, effect.gpindices, "parent is None?", bool(effect.parent is None))

        mdl.povms['Mtest'] = povm

        print("\nAfter adding to model:")
        print(" povm.gpindices = ",povm.gpindices, "parent is None?", bool(povm.parent is None))
        for i,fpovm in enumerate(povm.factorPOVMs):
            print(" factorPOVM%d.gpindices = " % i, fpovm.gpindices, "parent is None?", bool(fpovm.parent is None))
        for lbl,effect in povm.simplify_effects("Mtest").items():
            print(" compiled[%s].gpindices = " % lbl, effect.gpindices, "parent is None?", bool(effect.parent is None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
