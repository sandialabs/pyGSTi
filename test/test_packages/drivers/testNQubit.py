import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import pygsti
import numpy as np
from pygsti.modelpacks.legacy import std1Q_XY
from pygsti.baseobjs import Label as L
from pygsti.circuits import Circuit
import pygsti.models.modelconstruction as mc
from pygsti.processors.processorspec import QubitProcessorSpec as _ProcessorSpec
import warnings

from ..testutils import BaseTestCase, compare_files, regenerate_references
from pygsti.models import modelconstruction
from pygsti.circuits import cloudcircuitconstruction


#from .nqubitconstruction import *

#Mimics a function that used to be in pyGSTi, replaced with create_cloudnoise_model_from_hops_and_weights
def build_XYCNOT_cloudnoise_model(nQubits, geometry="line", cnot_edges=None,
                                  maxIdleWeight=1, maxSpamWeight=1, maxhops=0,
                                  extraWeight1Hops=0, extraGateWeight=0,
                                  roughNoise=None, simulator="matrix", parameterization="H+S",
                                  spamtype="lindblad", addIdleNoiseToAllGates=True,
                                  errcomp_type="gates", evotype="default", return_clouds=False, verbosity=0):

    availability = {}; nonstd_gate_unitaries = {}
    if cnot_edges is not None: availability['Gcnot'] = cnot_edges
    pspec = _ProcessorSpec(nQubits, ['Gidle', 'Gx','Gy','Gcnot'], nonstd_gate_unitaries, availability, geometry)
    assert(spamtype == "lindblad")  # unused and should remove this arg, but should always be "lindblad"
    mdl = mc.create_cloud_crosstalk_model_from_hops_and_weights(
        pspec, None,
        maxIdleWeight, maxSpamWeight, maxhops,
        extraWeight1Hops, extraGateWeight,
        simulator, evotype, parameterization, parameterization,
        "add_global" if addIdleNoiseToAllGates else "none",
        errcomp_type, True, True, True, verbosity)

    if return_clouds:
        #FUTURE - just return cloud *keys*? (operation label values are never used
        # downstream, but may still be useful for debugging, so keep for now)
        return mdl, mdl.clouds
    else:
        return mdl


class NQubitTestCase(BaseTestCase):

    def setUp(self):
        super(NQubitTestCase, self).setUp()

    def test_construction(self):
        print("TEST1")
        mdl_test = build_XYCNOT_cloudnoise_model(
            nQubits=1, geometry="line", maxIdleWeight=1, maxhops=0, verbosity=10)
        print("TEST2")
        mdl_test = build_XYCNOT_cloudnoise_model(
            nQubits=2, geometry="line", maxIdleWeight=1, maxhops=0, verbosity=10)
        print("TEST3")
        mdl_test = build_XYCNOT_cloudnoise_model(
            nQubits=3, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1, simulator="map", verbosity=10)
        #                                    roughNoise=(1234,0.1))

        #print("Constructed model with %d gates, dim=%d, and n_params=%d.  Norm(paramvec) = %g" %
        #      (len(mdl_test.operations),mdl_test.dim,mdl_test.num_params, np.linalg.norm(mdl_test.to_vector()) ))

    def test_sequential_sequenceselection(self):

        #only test when reps are fast (b/c otherwise this test is slow!)
        #try: from pygsti.objects.replib import fastreplib
        #except ImportError:
        #    warnings.warn("Skipping test_sequential_sequenceselection b/c no fastreps!")
        #    return

        nQubits = 2
        maxLengths = [1,2]
        cnot_edges = [(i,i+1) for i in range(nQubits-1)] #only single direction

        mdl_datagen = build_XYCNOT_cloudnoise_model(nQubits, "line", cnot_edges, maxIdleWeight=2, maxhops=1,
                                                    extraWeight1Hops=0, extraGateWeight=0,
                                                    verbosity=1,
                                                    simulator="map", parameterization="H+S",
                                                    roughNoise=(1234,0.01))

        cache = {}
        gss = cloudcircuitconstruction._create_xycnot_cloudnoise_circuits(
            nQubits, maxLengths, 'line', cnot_edges, max_idle_weight=2, maxhops=1,
            extra_weight_1_hops=0, extra_gate_weight=0, verbosity=4, cache=cache, algorithm="sequential")
        expList = list(gss) #[ tup[0] for tup in expList_tups]

        #RUN to SAVE list & dataset
        if regenerate_references():
            pygsti.serialization.json.dump(gss, open(compare_files + "/nqubit_2Q_seqs.json", 'w'))
            ds = pygsti.data.simulate_data(mdl_datagen, expList, 1000, "multinomial", seed=1234)
            pygsti.serialization.json.dump(ds, open(compare_files + "/nqubit_2Q_dataset.json", 'w'))

        compare_gss = pygsti.serialization.json.load(open(compare_files + "/nqubit_2Q_seqs.json"))

        self.assertEqual(set(gss), set(compare_gss))

    def test_greedy_sequenceselection(self):
        nQubits = 1
        maxLengths = [1,2]
        cnot_edges = []

        mdl_datagen = build_XYCNOT_cloudnoise_model(nQubits, "line", cnot_edges, maxIdleWeight=1, maxhops=0,
                                                    extraWeight1Hops=0, extraGateWeight=0,
                                                    verbosity=1, simulator="map", parameterization="H+S",
                                                    roughNoise=(1234,0.01))

        cache = {}
        gss = cloudcircuitconstruction._create_xycnot_cloudnoise_circuits(
            nQubits, maxLengths, 'line', cnot_edges, max_idle_weight=1, maxhops=0,
            extra_weight_1_hops=0, extra_gate_weight=0, verbosity=4, cache=cache, algorithm="greedy")
        #expList = gss.allstrs #[ tup[0] for tup in expList_tups]

        #RUN to SAVE list
        if regenerate_references():
            pygsti.serialization.json.dump(gss, open(compare_files + "/nqubit_1Q_seqs.json", 'w'))

        compare_gss = pygsti.serialization.json.load(open(compare_files + "/nqubit_1Q_seqs.json"))

        #expList_tups_mod = [tuple( etup[0:3] + ('XX','XX')) for etup in expList_tups ]
        #for etup in expList_tups:
        #    etup_mod = tuple( etup[0:3] + ('XX','XX'))
        #    if etup_mod not in compare_tups:
        #        print("Not found: ", etup)
        #
        #    #if (etup[0] != ctup[0]) or (etup[1] != ctup[1]) or (etup[2] != ctup[2]):
        #    #    print("Mismatch:",(etup[0] != ctup[0]), (etup[1] != ctup[1]), (etup[2] != ctup[2]))
        #    #    print(etup); print(ctup)
        #    #    print(tuple(etup[0]))
        #    #    print(tuple(ctup[0]))

        self.assertEqual(set(gss), set(compare_gss))

    def test_2Q(self):

        #only test when reps are fast (b/c otherwise this test is slow!)
        #try: from pygsti.objects.replib import fastreplib
        #except ImportError:
        #    warnings.warn("Skipping test_2Q b/c no fastreps!")
        #    return

        gss = pygsti.serialization.json.load(open(compare_files + "/nqubit_2Q_seqs.json"))
        expList = list(gss)

        ds = pygsti.serialization.json.load(open(compare_files + "/nqubit_2Q_dataset.json"))
        print(len(expList)," sequences")

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
                                                        simulator="map", parameterization="H+S")
        results = pygsti.run_long_sequence_gst_base(ds, mdl_to_optimize,
                                                    lsgstLists, gauge_opt_params=False,
                                                    advanced_options={'tolerance': 1e-1}, verbosity=4)

    def test_2Q_terms(self):

        #only test when reps are fast (b/c otherwise this test is slow!)
        #try: from pygsti.objects.replib import fastreplib
        #except ImportError:
        #    warnings.warn("Skipping test_2Q_terms b/c no fastreps!")
        #    return

        gss = pygsti.serialization.json.load(open(compare_files + "/nqubit_2Q_seqs.json"))
        expList = list(gss)

        ds = pygsti.serialization.json.load(open(compare_files + "/nqubit_2Q_dataset.json"))
        print(len(expList)," sequences")

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

        #RUN to create cache (SAVE)
        if regenerate_references():
            calc_cache = {}
            mdl_to_optimize.sim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1, cache=calc_cache)
            mdl_to_optimize.sim.bulk_probs(gss) #lsgstLists[-1]
            pygsti.serialization.json.dump(calc_cache, open(compare_files + '/nqubit_2Qterms.cache', 'w'))

        #Just load precomputed cache (we test run_long_sequence_gst_base here, not cache computation)
        calc_cache = pygsti.serialization.json.load(open(compare_files + '/nqubit_2Qterms.cache'))
        mdl_to_optimize.sim = pygsti.forwardsims.TermForwardSimulator(mode='taylor-order', max_order=1, cache=calc_cache)

        results = pygsti.run_long_sequence_gst_base(ds, mdl_to_optimize,
                                                    lsgstLists, gauge_opt_params=False,
                                                    advanced_options={'tolerance': 1e-3}, verbosity=4)


    def test_3Q(self):

        ##only test when reps are fast (b/c otherwise this test is slow!)
        #try: from pygsti.objects.replib import fastreplib
        #except ImportError:
        #    warnings.warn("Skipping test_3Q b/c no fastreps!")
        #    return

        nQubits = 3
        print("Constructing Target LinearOperator Set")
        target_model = build_XYCNOT_cloudnoise_model(
            nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1, simulator="map",verbosity=1)
        #print("nElements test = ",target_model.num_elements)
        #print("nParams test = ",target_model.num_params)
        #print("nNonGaugeParams test = ",target_model.num_nongauge_params)

        print("Constructing Datagen LinearOperator Set")
        mdl_datagen = build_XYCNOT_cloudnoise_model(
            nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1, verbosity=1, roughNoise=(1234,0.1), simulator="map")

        mdl_test = mdl_datagen
        print("Constructed model with %d op-blks, dim=%d, and nParams=%d.  Norm(paramvec) = %g" %
              (len(mdl_test.operation_blks),mdl_test.dim,mdl_test.num_params, np.linalg.norm(mdl_test.to_vector()) ))

        op_labels = target_model.primitive_op_labels
        line_labels = tuple(range(nQubits))
        fids1Q = std1Q_XY.fiducials
        fiducials = []
        for i in range(nQubits):
            fiducials.extend( pygsti.circuits.manipulate_circuits(
                fids1Q, [ ( (L('Gx'),) , (L('Gx',i),) ), ( (L('Gy'),) , (L('Gy',i),) ) ], line_labels=line_labels) )
        print(len(fiducials), "Fiducials")
        prep_fiducials = meas_fiducials = fiducials
        #TODO: add fiducials for 2Q pairs (edges on graph)

        germs = pygsti.circuits.to_circuits([(gl,) for gl in op_labels], line_labels=line_labels)
        maxLs = [1]
        expList = pygsti.circuits.create_lsgst_circuits(mdl_datagen, prep_fiducials, meas_fiducials, germs, maxLs)
        self.assertTrue( Circuit((),line_labels) in expList)

        ds = pygsti.data.simulate_data(mdl_datagen, expList, 1000, "multinomial", seed=1234)
        print("Created Dataset with %d strings" % len(ds))

        logL = pygsti.tools.logl(mdl_datagen, ds, expList)
        max_logL = pygsti.tools.logl_max(mdl_datagen, ds, expList)
        twoDeltaLogL = 2*(max_logL-logL)
        chi2 = pygsti.tools.chi2(mdl_datagen, ds, expList)

        dof = ds.degrees_of_freedom()
        nParams = mdl_datagen.num_params
        print("Datagen 2DeltaLogL = 2(%g-%g) = %g" % (logL,max_logL,twoDeltaLogL))
        print("Datagen chi2 = ",chi2)
        print("Datagen expected DOF = ",dof)
        print("nParams = ",nParams)
        print("Expected 2DeltaLogL or chi2 ~= %g-%g =%g" % (dof,nParams,dof-nParams))
        #print("EXIT"); exit()
        return

        results = pygsti.run_long_sequence_gst(ds, target_model, prep_fiducials, meas_fiducials, germs, maxLs, verbosity=5,
                                               advanced_options={'max_iterations': 2}) #keep this short; don't care if it doesn't converge.
        print("DONE!")



    def test_SPAM(self):
        nQubits = 3
        factorPOVMs = []
        basis1Q = pygsti.baseobjs.Basis.cast("pp", 4)
        basisNQ = pygsti.baseobjs.Basis.cast("pp", 4 ** nQubits)
        for i in range(nQubits):
            effects = [(l, modelconstruction.create_spam_vector(l, "Q0", basis1Q)) for l in ["0", "1"]]
            factorPOVMs.append(pygsti.modelmembers.povms.TPPOVM(effects, evotype='default'))
        povm = pygsti.modelmembers.povms.TensorProductPOVM(factorPOVMs)
        print(list(povm.keys()))
        print("params = ",povm.num_params,"dim = ",povm.state_space.dim)
        print(povm)

        v = povm.to_vector()
        v += np.random.random( len(v) )
        povm.from_vector(v)
        print("Post adding noise:"); print(povm)

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
