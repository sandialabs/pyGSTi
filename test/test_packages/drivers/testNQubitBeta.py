import unittest
import pygsti
import numpy as np
from pygsti.construction import std1Q_XY
from pygsti.construction import std2Q_XYICNOT
from pygsti.objects.gatemapcalc import GateMapCalc
import sys, os

from ..testutils import BaseTestCase, compare_files, temp_files

from .nqubitconstruction import *


class NQubitTestCase(BaseTestCase):

    def setUp(self):
        super(NQubitTestCase, self).setUp()

    def test_construction(self):
        print("TEST1")
        gs_test = create_nqubit_gateset(nQubits=1, geometry="line", maxIdleWeight=1, maxhops=0, verbosity=10)
        print("TEST2")
        gs_test = create_nqubit_gateset(nQubits=2, geometry="line", maxIdleWeight=1, maxhops=0, verbosity=10)
        print("TEST3")
        gs_test = create_nqubit_gateset(nQubits=3, geometry="line", maxIdleWeight=1, maxhops=1,
                                        extraWeight1Hops=0, extraGateWeight=1, sparse=True, verbosity=10)
        #                                    gateNoise=(1234,0.1), prepNoise=(456,0.01), povmNoise=(789,0.01))
        
        #print("Constructed gateset with %d gates, dim=%d, and nParams=%d.  Norm(paramvec) = %g" %
        #      (len(gs_test.gates),gs_test.dim,gs_test.num_params(), np.linalg.norm(gs_test.to_vector()) ))

        
    def test_3Q(self):

        nQubits = 3
        print("Constructing Target Gate Set")
        gs_target = create_nqubit_gateset(nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                          extraWeight1Hops=0, extraGateWeight=1, sparse=True, verbosity=1)
        gs_target._calcClass = GateMapCalc
        #print("nElements test = ",gs_target.num_elements())
        #print("nParams test = ",gs_target.num_params())
        #print("nNonGaugeParams test = ",gs_target.num_nongauge_params())
        
        print("Constructing Datagen Gate Set")
        gs_datagen = create_nqubit_gateset(nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                           extraWeight1Hops=0, extraGateWeight=1, sparse=True, verbosity=1,
                                           gateNoise=(1234,0.1), prepNoise=(456,0.01), povmNoise=(789,0.01))
        gs_datagen._calcClass = GateMapCalc
        
        gs_test = gs_datagen
        print("Constructed gateset with %d gates, dim=%d, and nParams=%d.  Norm(paramvec) = %g" %
              (len(gs_test.gates),gs_test.dim,gs_test.num_params(), np.linalg.norm(gs_test.to_vector()) ))
        
        gateLabels = list(gs_target.gates.keys())
        fids1Q = std1Q_XY.fiducials
        fiducials = []
        for i in range(nQubits):
            fiducials.extend( pygsti.construction.manipulate_gatestring_list(
                fids1Q, [ ( ('Gx',) , ('Gx%d'%i,) ), ( ('Gy',) , ('Gy%d'%i,) ) ]) )
        print(len(fiducials), "Fiducials")
        prep_fiducials = meas_fiducials = fiducials
        #TODO: add fiducials for 2Q pairs (edges on graph)
        
        germs = pygsti.construction.gatestring_list([ (gl,) for gl in gateLabels ]) 
        maxLs = [1]
        expList = pygsti.construction.make_lsgst_experiment_list(gs_datagen, prep_fiducials, meas_fiducials, germs, maxLs)
        
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
