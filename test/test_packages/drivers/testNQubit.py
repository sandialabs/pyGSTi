import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import pygsti
import numpy as np
from pygsti.modelpacks.legacy import std1Q_XY
from pygsti.modelpacks.legacy import std2Q_XYICNOT
from pygsti.objects import Label as L
from pygsti.objects import Circuit
import pygsti.construction as pc
import sys
import warnings

from ..testutils import BaseTestCase, compare_files, temp_files, regenerate_references
from pygsti.construction import modelconstruction, nqnoiseconstruction

#from .nqubitconstruction import *

#Mimics a function that used to be in pyGSTi, replaced with create_cloudnoise_model_from_hops_and_weights
def build_XYCNOT_cloudnoise_model(nQubits, geometry="line", cnot_edges=None,
                                      maxIdleWeight=1, maxSpamWeight=1, maxhops=0,
                                      extraWeight1Hops=0, extraGateWeight=0,
                                      sparse_lindblad_basis=False, sparse_lindblad_reps=False,
                                      roughNoise=None, simulator="matrix", parameterization="H+S",
                                      spamtype="lindblad", addIdleNoiseToAllGates=True,
                                      errcomp_type="gates", return_clouds=False, verbosity=0):

    availability = {}; nonstd_gate_unitaries = {}
    if cnot_edges is not None: availability['Gcnot'] = cnot_edges
    return pc.create_cloudnoise_model_from_hops_and_weights(
        nQubits, ['Gx','Gy','Gcnot'], nonstd_gate_unitaries, None, availability,
        None, geometry, maxIdleWeight, maxSpamWeight, maxhops,
        extraWeight1Hops, extraGateWeight,
        roughNoise, sparse_lindblad_basis, sparse_lindblad_reps,
        simulator, parameterization,
        spamtype, addIdleNoiseToAllGates,
        errcomp_type, True, return_clouds, verbosity)


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
            extraWeight1Hops=0, extraGateWeight=1, sparse_lindblad_basis=True,
            sparse_lindblad_reps=True, simulator="map", verbosity=10)
        #                                    roughNoise=(1234,0.1))

        #print("Constructed model with %d gates, dim=%d, and n_params=%d.  Norm(paramvec) = %g" %
        #      (len(mdl_test.operations),mdl_test.dim,mdl_test.num_params(), np.linalg.norm(mdl_test.to_vector()) ))

    def test_sequential_sequenceselection(self):

        #only test when reps are fast (b/c otherwise this test is slow!)
        try: from pygsti.objects.replib import fastreplib
        except ImportError:
            warnings.warn("Skipping test_sequential_sequenceselection b/c no fastreps!")
            return

        nQubits = 2
        maxLengths = [1,2]
        cnot_edges = [(i,i+1) for i in range(nQubits-1)] #only single direction

        mdl_datagen = build_XYCNOT_cloudnoise_model(nQubits, "line", cnot_edges, maxIdleWeight=2, maxhops=1,
                                                    extraWeight1Hops=0, extraGateWeight=0,
                                                    sparse_lindblad_basis=True, sparse_lindblad_reps=True, verbosity=1,
                                                    simulator="map", parameterization="H+S",
                                                    roughNoise=(1234,0.01))

        cache = {}
        gss = nqnoiseconstruction._create_xycnot_cloudnoise_circuits(
            nQubits, maxLengths, 'line', cnot_edges, max_idle_weight=2, maxhops=1,
            extra_weight_1_hops=0, extra_gate_weight=0, verbosity=4, cache=cache, algorithm="sequential")
        expList = list(gss) #[ tup[0] for tup in expList_tups]

        #RUN to SAVE list & dataset
        if regenerate_references():
            pygsti.io.json.dump(gss, open(compare_files + "/nqubit_2Q_seqs.json",'w'))
            ds = pygsti.construction.simulate_data(mdl_datagen, expList, 1000, "multinomial", seed=1234)
            pygsti.io.json.dump(ds,open(compare_files + "/nqubit_2Q_dataset.json",'w'))

        compare_gss = pygsti.io.json.load(open(compare_files + "/nqubit_2Q_seqs.json"))
        self.assertEqual(set(gss), set(compare_gss))



    def test_greedy_sequenceselection(self):
        nQubits = 1
        maxLengths = [1,2]
        cnot_edges = []

        mdl_datagen = build_XYCNOT_cloudnoise_model(nQubits, "line", cnot_edges, maxIdleWeight=1, maxhops=0,
                                                    extraWeight1Hops=0, extraGateWeight=0,
                                                    sparse_lindblad_basis=True, sparse_lindblad_reps=True,
                                                    verbosity=1, simulator="map", parameterization="H+S",
                                                    roughNoise=(1234,0.01))

        cache = {}
        gss = nqnoiseconstruction._create_xycnot_cloudnoise_circuits(
            nQubits, maxLengths, 'line', cnot_edges, max_idle_weight=1, maxhops=0,
            extra_weight_1_hops=0, extra_gate_weight=0, verbosity=4, cache=cache, algorithm="greedy")
        #expList = gss.allstrs #[ tup[0] for tup in expList_tups]

        #RUN to SAVE list
        if regenerate_references():
            pygsti.io.json.dump(gss, open(compare_files + "/nqubit_1Q_seqs.json",'w'))

        compare_gss = pygsti.io.json.load(open(compare_files + "/nqubit_1Q_seqs.json"))

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
        try: from pygsti.objects.replib import fastreplib
        except ImportError:
            warnings.warn("Skipping test_2Q b/c no fastreps!")
            return

        gss = pygsti.io.json.load(open(compare_files + "/nqubit_2Q_seqs.json"))
        expList = list(gss)

        ds = pygsti.io.json.load(open(compare_files + "/nqubit_2Q_dataset.json"))
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
        lsgstLists = gss.truncate(maxLengths) # can just use gss as input to pygsti.run_long_sequence_gst_base

        mdl_to_optimize = build_XYCNOT_cloudnoise_model(nQubits, "line", cnot_edges, maxIdleWeight=2, maxhops=1,
                                                         extraWeight1Hops=0, extraGateWeight=1, verbosity=1,
                                                         simulator="map", parameterization="H+S",
                                                        sparse_lindblad_basis=True, sparse_lindblad_reps=True)
        results = pygsti.run_long_sequence_gst_base(ds, mdl_to_optimize,
                                                   lsgstLists, gauge_opt_params=False,
                                                   advanced_options={'tolerance': 1e-1}, verbosity=4)

    def test_2Q_terms(self):

        #only test when reps are fast (b/c otherwise this test is slow!)
        try: from pygsti.objects.replib import fastreplib
        except ImportError:
            warnings.warn("Skipping test_2Q_terms b/c no fastreps!")
            return

        gss = pygsti.io.json.load(open(compare_files + "/nqubit_2Q_seqs.json"))
        expList = list(gss)

        ds = pygsti.io.json.load(open(compare_files + "/nqubit_2Q_dataset.json"))
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

        termsim = pygsti.objects.TermForwardSimulator(mode='taylor-order', max_order=1)
        mdl_to_optimize = build_XYCNOT_cloudnoise_model(nQubits, "line", cnot_edges, maxIdleWeight=2, maxhops=1,
                                                        extraWeight1Hops=0, extraGateWeight=1, verbosity=1,
                                                        simulator=termsim, parameterization="H+S terms",
                                                        sparse_lindblad_basis=False, sparse_lindblad_reps=False)

        #RUN to create cache (SAVE)
        if regenerate_references():
            calc_cache = {}
            mdl_to_optimize.sim = pygsti.objects.TermForwardSimulator(mode='taylor-order', max_order=1, cache=calc_cache)
            mdl_to_optimize.sim.bulk_probs(gss) #lsgstLists[-1]
            pygsti.io.json.dump(calc_cache, open(compare_files + '/nqubit_2Qterms.cache','w'))

        #Just load precomputed cache (we test run_long_sequence_gst_base here, not cache computation)
        calc_cache = pygsti.io.json.load(open(compare_files + '/nqubit_2Qterms.cache'))
        mdl_to_optimize.sim = pygsti.objects.TermForwardSimulator(mode='taylor-order', max_order=1, cache=calc_cache)

        results = pygsti.run_long_sequence_gst_base(ds, mdl_to_optimize,
                                                    lsgstLists, gauge_opt_params=False,
                                                    advanced_options={'tolerance': 1e-3}, verbosity=4)


    def test_3Q(self):

        #only test when reps are fast (b/c otherwise this test is slow!)
        try: from pygsti.objects.replib import fastreplib
        except ImportError:
            warnings.warn("Skipping test_3Q b/c no fastreps!")
            return

        nQubits = 3
        print("Constructing Target LinearOperator Set")
        target_model = build_XYCNOT_cloudnoise_model(
            nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1,
            sparse_lindblad_basis=True, sparse_lindblad_reps=True,
            simulator="map",verbosity=1)
        #print("nElements test = ",target_model.num_elements())
        #print("nParams test = ",target_model.num_params())
        #print("nNonGaugeParams test = ",target_model.num_nongauge_params())

        print("Constructing Datagen LinearOperator Set")
        mdl_datagen = build_XYCNOT_cloudnoise_model(
            nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1,
            sparse_lindblad_basis=True, sparse_lindblad_reps=True,
            verbosity=1, roughNoise=(1234,0.1), simulator="map")

        mdl_test = mdl_datagen
        print("Constructed model with %d op-blks, dim=%d, and nParams=%d.  Norm(paramvec) = %g" %
              (len(mdl_test.operation_blks),mdl_test.dim,mdl_test.num_params(), np.linalg.norm(mdl_test.to_vector()) ))

        op_labels = target_model.primitive_op_labels
        line_labels = tuple(range(nQubits))
        fids1Q = std1Q_XY.fiducials
        fiducials = []
        for i in range(nQubits):
            fiducials.extend( pygsti.construction.manipulate_circuits(
                fids1Q, [ ( (L('Gx'),) , (L('Gx',i),) ), ( (L('Gy'),) , (L('Gy',i),) ) ], line_labels=line_labels) )
        print(len(fiducials), "Fiducials")
        prep_fiducials = meas_fiducials = fiducials
        #TODO: add fiducials for 2Q pairs (edges on graph)

        germs = pygsti.construction.to_circuits([ (gl,) for gl in op_labels ], line_labels=line_labels)
        maxLs = [1]
        expList = pygsti.construction.create_lsgst_circuits(mdl_datagen, prep_fiducials, meas_fiducials, germs, maxLs)
        self.assertTrue( Circuit((),line_labels) in expList)

        ds = pygsti.construction.simulate_data(mdl_datagen, expList, 1000, "multinomial", seed=1234)
        print("Created Dataset with %d strings" % len(ds))

        logL = pygsti.tools.logl(mdl_datagen, ds, expList)
        max_logL = pygsti.tools.logl_max(mdl_datagen, ds, expList)
        twoDeltaLogL = 2*(max_logL-logL)
        chi2 = pygsti.tools.chi2(mdl_datagen, ds, expList)

        dof = ds.degrees_of_freedom()
        nParams = mdl_datagen.num_params()
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
        basis1Q = pygsti.obj.Basis.cast("pp",4)
        basisNQ = pygsti.obj.Basis.cast("pp",4**nQubits)
        for i in range(nQubits):
            effects = [ (l,modelconstruction._basis_create_spam_vector(l, basis1Q)) for l in ["0","1"] ]
            factorPOVMs.append( pygsti.obj.TPPOVM(effects) )
        povm = pygsti.obj.TensorProdPOVM( factorPOVMs )
        print(list(povm.keys()))
        print("params = ",povm.num_params(),"dim = ",povm.dim)
        print(povm)

        v = povm.to_vector()
        v += np.random.random( len(v) )
        povm.from_vector(v)
        print("Post adding noise:"); print(povm)

        mdl = pygsti.obj.ExplicitOpModel(['Q0','Q1','Q2'])
        prepFactors = [ pygsti.obj.TPSPAMVec(modelconstruction._basis_create_spam_vector("0", basis1Q))
                        for i in range(nQubits)]
        mdl.preps['rho0'] = pygsti.obj.TensorProdSPAMVec('prep',prepFactors)
        # OR one big prep: mdl.preps['rho0'] = modelconstruction._basis_create_spam_vector("0", basisNQ)

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
