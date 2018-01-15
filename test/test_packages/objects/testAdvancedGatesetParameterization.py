from __future__ import print_function

import unittest
import pygsti
import numpy as np

from pygsti.construction import std1Q_XYI
from pygsti.objects import LindbladParameterizedGate, ComposedGate, EmbeddedGate, StaticGate

from ..testutils import BaseTestCase, compare_files, temp_files

class AdvancedParameterizationTestCase(BaseTestCase):

    def setUp(self):
        super(AdvancedParameterizationTestCase, self).setUp()
        #Nothing yet...

    def test_composed_embedded_param(self):
        #Test out above cell
        gs1Q = std1Q_XYI.gs_target.copy()
        
        print("START")
        
        nQubits = 3 # say
        Id_1Q = np.identity(4**1,'d')
        idleErr0 = LindbladParameterizedGate(Id_1Q) # 1-qubit error generator
        idleErr1 = LindbladParameterizedGate(Id_1Q) # allow different "idle" 
        idleErr2 = LindbladParameterizedGate(Id_1Q) # 1Q errors on each qubit
        # so far no gpindices have been set...
        
        ss3Q = [('Q0','Q1','Q2')] #3Q state space
        basis3Q = pygsti.objects.Basis('pp', 2**nQubits) #3Q basis
        Giii = ComposedGate([ EmbeddedGate(ss3Q, ('Q0',), idleErr0, basis3Q),
                              EmbeddedGate(ss3Q, ('Q1',), idleErr1, basis3Q),
                              EmbeddedGate(ss3Q, ('Q2',), idleErr2, basis3Q)
                            ])
        
        targetGx = StaticGate(gs1Q.gates['Gx'])
        Gxii_xErr = LindbladParameterizedGate(Id_1Q) 
        Gxii_xGate = ComposedGate( [targetGx, idleErr0, Gxii_xErr])
        Gxii = ComposedGate([ EmbeddedGate(ss3Q, ('Q0',), Gxii_xGate, basis3Q),
                              EmbeddedGate(ss3Q, ('Q1',), idleErr1, basis3Q),
                              EmbeddedGate(ss3Q, ('Q2',), idleErr2, basis3Q)
                            ])
        
        def printInfo():
            def pp(x): return id(x) if (x is not None) else x #print's parent nicely
            print("INDEX INFO")
            print("idleErr0 (%d):" % id(idleErr0), idleErr0.gpindices, pp(idleErr0.parent), idleErr0.num_params())
            print("idleErr1 (%d):" % id(idleErr1), idleErr1.gpindices, pp(idleErr1.parent), idleErr1.num_params())
            print("idleErr2 (%d):" % id(idleErr2), idleErr2.gpindices, pp(idleErr2.parent), idleErr2.num_params())
            print("Gxii_xErr (%d):" % id(Gxii_xErr), Gxii_xErr.gpindices, pp(Gxii_xErr.parent), Gxii_xErr.num_params())
            print("Gxii_xGate (%d):" % id(Gxii_xGate), Gxii_xGate.gpindices, pp(Gxii_xGate.parent), Gxii_xGate.num_params())
            print("Giii (%d):" % id(Giii), Giii.gpindices, pp(Giii.parent), Giii.num_params())
            print("Gxii (%d):" % id(Gxii), Gxii.gpindices, pp(Gxii.parent), Gxii.num_params())
            print()
        
        # rubber meets road: how to assign gpindices??
        # need gateset.from_vector() to work, and also to_vector(), but maybe less important
        print("PREGAME")
        printInfo()
        print("BEGIN")
        gs_constructed = pygsti.obj.GateSet()
        print("Gateset id = ",id(gs_constructed))
        print("INSERT1: Giii indices = ", Giii.gpindices, " parent = ", Giii.parent)
        gs_constructed.gates['Giii'] = Giii # will set gpindices of Giii, which will set those of 
         # of it's contained EmbeddedGates which will set gpindices of idleErr0, idleErr1, & idleErr2
         # (because they're None to begin with)
        print("POST")
        Giii = gs_constructed.gates['Giii'] #so printInfo works
        printInfo()
        print("INSERT2: Gxii indices = ", Gxii.gpindices, " parent = ", Gxii.parent)
        gs_constructed.gates['Gxii'] = Gxii # similar, but will only set indices of Gxii_xGate (gpindices
         # of idleErr1 and idleErr2 are already set), which will only set indices of Gxii_xErr
         # (since idleErr0.gpindices is already set )
        Giii = gs_constructed.gates['Giii'] #so printInfo works
        Gxii = gs_constructed.gates['Gxii'] #so printInfo works
        printInfo()
        
        print("TOTAL Params = ", gs_constructed.num_params())
        
        v = gs_constructed.to_vector()
        print("len(v) =",len(v))
        gs_constructed2 = gs_constructed.copy()
        print("Copy's total params = ", gs_constructed2.num_params())
        gs_constructed2.from_vector(v)
        print("Diff = %g (should be 0)" % gs_constructed.frobeniusdist(gs_constructed2))
        self.assertAlmostEqual(gs_constructed.frobeniusdist(gs_constructed2),0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
