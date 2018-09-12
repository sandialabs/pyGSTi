import unittest
import pygsti
import numpy as np
from pygsti.construction import std1Q_XY
from pygsti.construction import std2Q_XYICNOT
from pygsti.objects import Label as L
import pygsti.construction as pc
import sys, os

from ..testutils import BaseTestCase, compare_files, temp_files

#from .nqubitconstruction import *


class NQubitTestCase(BaseTestCase):

    def setUp(self):
        super(NQubitTestCase, self).setUp()

    def test_construction(self):
        print("TEST1")
        gs_test = pygsti.construction.build_nqnoise_gateset(
            nQubits=1, geometry="line", maxIdleWeight=1, maxhops=0, verbosity=10)
        print("TEST2")
        gs_test = pygsti.construction.build_nqnoise_gateset(
            nQubits=2, geometry="line", maxIdleWeight=1, maxhops=0, verbosity=10)
        print("TEST3")
        gs_test = pygsti.construction.build_nqnoise_gateset(
            nQubits=3, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1, sparse=True, sim_type="map", verbosity=10)
        #                                    gateNoise=(1234,0.1), prepNoise=(456,0.01), povmNoise=(789,0.01))
        
        #print("Constructed gateset with %d gates, dim=%d, and nParams=%d.  Norm(paramvec) = %g" %
        #      (len(gs_test.gates),gs_test.dim,gs_test.num_params(), np.linalg.norm(gs_test.to_vector()) ))

    def test_sequential_sequenceselection(self):
        nQubits = 2
        maxLengths = [1,2]
        cnot_edges = [(i,i+1) for i in range(nQubits-1)] #only single direction

        gs_datagen = pc.build_nqnoise_gateset(nQubits, "line", cnot_edges, maxIdleWeight=2, maxhops=1,
                                      extraWeight1Hops=0, extraGateWeight=0, sparse=True, verbosity=1,
                                      sim_type="map", parameterization="H+S",
                                      gateNoise=(1234,0.01), prepNoise=(456,0.01), povmNoise=(789,0.01))

        cache = {}
        expList_tups, germs = pygsti.construction.create_nqubit_sequences(
            nQubits, maxLengths, 'line', cnot_edges, maxIdleWeight=2, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=0, verbosity=4, cache=cache, algorithm="sequential")
        expList = [ tup[0] for tup in expList_tups]

        #RUN to save list & dataset
        #pygsti.io.json.dump(expList_tups, open(compare_files + "/nqubit_2Q_seqs.json",'w'))
        #ds = pygsti.construction.generate_fake_data(gs_datagen, expList, 1000, "multinomial", seed=1234)
        #pygsti.io.json.dump(ds,open(compare_files + "/nqubit_2Q_dataset.json",'w'))

        compare_tups = pygsti.io.json.load(open(compare_files + "/nqubit_2Q_seqs.json"))
        self.assertEqual(set(expList_tups), set(compare_tups))



    def test_greedy_sequenceselection(self):
        nQubits = 1
        maxLengths = [1,2]
        cnot_edges = []

        gs_datagen = pc.build_nqnoise_gateset(nQubits, "line", cnot_edges, maxIdleWeight=1, maxhops=0,
                                      extraWeight1Hops=0, extraGateWeight=0, sparse=True, verbosity=1,
                                      sim_type="map", parameterization="H+S",
                                      gateNoise=(1234,0.01), prepNoise=(456,0.01), povmNoise=(789,0.01))

        cache = {}
        expList_tups, germs = pygsti.construction.create_nqubit_sequences(
            nQubits, maxLengths, 'line', cnot_edges, maxIdleWeight=1, maxhops=0,
            extraWeight1Hops=0, extraGateWeight=0, verbosity=4, cache=cache, algorithm="greedy")
        #expList = [ tup[0] for tup in expList_tups]

        #RUN to save list
        #pygsti.io.json.dump(expList_tups, open(compare_files + "/nqubit_1Q_seqs.json",'w'))

        compare_tups = pygsti.io.json.load(open(compare_files + "/nqubit_1Q_seqs.json"))
        self.assertEqual(set(expList_tups), set(compare_tups))

        
    def test_2Q(self):

        expList_tups = pygsti.io.json.load(open(compare_files + "/nqubit_2Q_seqs.json"))
        expList = [ tup[0] for tup in expList_tups]

        ds = pygsti.io.json.load(open(compare_files + "/nqubit_2Q_dataset.json"))
        print(len(expList)," sequences")   

        nQubits = 2
        maxLengths = [1,2]
        cnot_edges = [(i,i+1) for i in range(nQubits-1)] #only single direction

        lsgstLists = []; lst = []
        for L in maxLengths:
            for tup in expList_tups:
                if tup[1] == L: lst.append( tup[0] )
            lsgstLists.append(lst[:]) # append *running* list
            
        gs_to_optimize = pc.build_nqnoise_gateset(nQubits, "line", cnot_edges, maxIdleWeight=2, maxhops=1,
                                                  extraWeight1Hops=0, extraGateWeight=1, verbosity=1,
                                                  sim_type="map", parameterization="H+S", sparse=True)

        results = pygsti.do_long_sequence_gst_base(ds, gs_to_optimize,
                                                   lsgstLists, gaugeOptParams=False,
                                                   advancedOptions={'tolerance': 1e-2}, verbosity=4)

    def test_2Q_terms(self):

        expList_tups = pygsti.io.json.load(open(compare_files + "/nqubit_2Q_seqs.json"))
        expList = [ tup[0] for tup in expList_tups]
        
        ds = pygsti.io.json.load(open(compare_files + "/nqubit_2Q_dataset.json"))
        print(len(expList)," sequences")   

        nQubits = 2
        maxLengths = [1,2]
        cnot_edges = [(i,i+1) for i in range(nQubits-1)] #only single direction

        lsgstLists = []; lst = []
        for L in maxLengths:
            for tup in expList_tups:
                if tup[1] == L: lst.append( tup[0] )
            lsgstLists.append(lst[:]) # append *running* list

        gs_to_optimize = pc.build_nqnoise_gateset(nQubits, "line", cnot_edges, maxIdleWeight=2, maxhops=1,
                                                  extraWeight1Hops=0, extraGateWeight=1, verbosity=1,
                                                  sim_type="termorder:1", parameterization="H+S terms", sparse=False)

        #RUN to create cache
        #calc_cache = {}
        #gs_to_optimize.set_simtype("termorder:1",calc_cache)
        #gs_to_optimize.bulk_probs(lsgstLists[-1])
        #pygsti.io.json.dump(calc_cache, open(compare_files + '/nqubit_2Qterms.cache','w'))

        #Just load precomputed cache (we test do_long_sequence_gst_base here, not cache computation)
        calc_cache = pygsti.io.json.load(open(compare_files + '/nqubit_2Qterms.cache'))
        gs_to_optimize.set_simtype("termorder:1",calc_cache)

        results = pygsti.do_long_sequence_gst_base(ds, gs_to_optimize,
                                                   lsgstLists, gaugeOptParams=False,
                                                   advancedOptions={'tolerance': 1e-3}, verbosity=4)
        
        
    def test_3Q(self):

        nQubits = 3
        print("Constructing Target Gate Set")
        gs_target = pygsti.construction.build_nqnoise_gateset(
            nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1, sparse=True, sim_type="map",verbosity=1)
        #print("nElements test = ",gs_target.num_elements())
        #print("nParams test = ",gs_target.num_params())
        #print("nNonGaugeParams test = ",gs_target.num_nongauge_params())
        
        print("Constructing Datagen Gate Set")
        gs_datagen = pygsti.construction.build_nqnoise_gateset(
            nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1, sparse=True, verbosity=1,
            gateNoise=(1234,0.1), prepNoise=(456,0.01), povmNoise=(789,0.01), sim_type="map")
        
        gs_test = gs_datagen
        print("Constructed gateset with %d gates, dim=%d, and nParams=%d.  Norm(paramvec) = %g" %
              (len(gs_test.gates),gs_test.dim,gs_test.num_params(), np.linalg.norm(gs_test.to_vector()) ))
        
        gateLabels = list(gs_target.gates.keys())
        fids1Q = std1Q_XY.fiducials
        fiducials = []
        for i in range(nQubits):
            fiducials.extend( pygsti.construction.manipulate_gatestring_list(
                fids1Q, [ ( (L('Gx'),) , (L('Gx',i),) ), ( (L('Gy'),) , (L('Gy',i),) ) ]) )
        print(len(fiducials), "Fiducials")
        prep_fiducials = meas_fiducials = fiducials
        #TODO: add fiducials for 2Q pairs (edges on graph)
        
        germs = pygsti.construction.gatestring_list([ (gl,) for gl in gateLabels ]) 
        maxLs = [1]
        expList = pygsti.construction.make_lsgst_experiment_list(gs_datagen, prep_fiducials, meas_fiducials, germs, maxLs)
        self.assertTrue(() in expList)
        
        ds = pygsti.construction.generate_fake_data(gs_datagen, expList, 1000, "multinomial", seed=1234)
        print("Created Dataset with %d strings" % len(ds))
            
        logL = pygsti.tools.logl(gs_datagen, ds, expList)
        max_logL = pygsti.tools.logl_max(gs_datagen, ds, expList)
        twoDeltaLogL = 2*(max_logL-logL)
        chi2 = pygsti.tools.chi2(gs_datagen, ds, expList)
        
        dof = ds.get_degrees_of_freedom()
        nParams = gs_datagen.num_params()
        print("Datagen 2DeltaLogL = 2(%g-%g) = %g" % (logL,max_logL,twoDeltaLogL))
        print("Datagen chi2 = ",chi2)
        print("Datagen expected DOF = ",dof)
        print("nParams = ",nParams)
        print("Expected 2DeltaLogL or chi2 ~= %g-%g =%g" % (dof,nParams,dof-nParams))
        #print("EXIT"); exit()
        return
        
        results = pygsti.do_long_sequence_gst(ds, gs_target, prep_fiducials, meas_fiducials, germs, maxLs, verbosity=5,
                                              advancedOptions={'maxIterations': 2}) #keep this short; don't care if it doesn't converge.
        print("DONE!")



    def test_SPAM(self):
        nQubits = 3
        factorPOVMs = []
        basis1Q = pygsti.obj.Basis("pp",2)
        basisNQ = pygsti.obj.Basis("pp",2**nQubits)
        for i in range(nQubits):
            effects = [ (l,pygsti.construction.basis_build_vector(l, basis1Q)) for l in ["0","1"] ]
            factorPOVMs.append( pygsti.obj.TPPOVM(effects) )
        povm = pygsti.obj.TensorProdPOVM( factorPOVMs )
        print(list(povm.keys()))
        print("params = ",povm.num_params(),"dim = ",povm.dim)
        print(povm)
    
        v = povm.to_vector()
        v += np.random.random( len(v) )
        povm.from_vector(v)
        print("Post adding noise:"); print(povm)
    
        gs = pygsti.obj.GateSet()
        prepFactors = [ pygsti.obj.TPParameterizedSPAMVec(pygsti.construction.basis_build_vector("0", basis1Q))
                        for i in range(nQubits)]
        gs.preps['rho0'] = pygsti.obj.TensorProdSPAMVec('prep',prepFactors)
        # OR one big prep: gs.preps['rho0'] = pygsti.construction.basis_build_vector("0", basisNQ)
        
        print("Before adding to gateset:")
        print(" povm.gpindices = ",povm.gpindices, "parent is None?", bool(povm.parent is None))
        for i,fpovm in enumerate(povm.factorPOVMs):
            print(" factorPOVM%d.gpindices = " % i, fpovm.gpindices, "parent is None?", bool(fpovm.parent is None))
        for lbl,effect in povm.compile_effects().items():
            print(" compiled[%s].gpindices = " % lbl, effect.gpindices, "parent is None?", bool(effect.parent is None))
        
        gs.povms['Mtest'] = povm
        
        print("\nAfter adding to gateset:")
        print(" povm.gpindices = ",povm.gpindices, "parent is None?", bool(povm.parent is None))
        for i,fpovm in enumerate(povm.factorPOVMs):
            print(" factorPOVM%d.gpindices = " % i, fpovm.gpindices, "parent is None?", bool(fpovm.parent is None))
        for lbl,effect in povm.compile_effects("Mtest").items():
            print(" compiled[%s].gpindices = " % lbl, effect.gpindices, "parent is None?", bool(effect.parent is None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
