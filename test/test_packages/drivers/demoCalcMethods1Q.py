from __future__ import print_function
import unittest
import numpy as np
import scipy.linalg as spl
import pygsti
import pygsti.construction as pc
from pygsti.construction import std1Q_XYI as std
from pygsti.construction import std1Q_XY
from pygsti.objects import Label as L

import sys, os


from ..testutils import BaseTestCase, compare_files, temp_files

class CalcMethods1QTestCase(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        """ 
        Handle all once-per-class (slow) computation and loading,
         to avoid calling it for each test (like setUp).  Store
         results in class variable for use within setUp.
        """
        super(CalcMethods1QTestCase, cls).setUpClass()

        #Standard GST dataset
        cls.maxLengths = [1,2,4]
        cls.gs_datagen = std.gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
        cls.listOfExperiments = pygsti.construction.make_lsgst_experiment_list(
            std.gs_target, std.prepStrs, std.effectStrs, std.germs, cls.maxLengths)
        cls.ds = pygsti.construction.generate_fake_data(cls.gs_datagen, cls.listOfExperiments,
                                                         nSamples=1000, sampleError="multinomial", seed=1234)

        #Reduced model GST dataset
        cls.nQubits=1
        cls.gs_redmod_datagen = pc.build_nqnoise_gateset(cls.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                                    extraWeight1Hops=0, extraGateWeight=1, sparse=False, sim_type="matrix", verbosity=1,
                                                    gateNoise=(1234,0.01), prepNoise=(456,0.01), povmNoise=(789,0.01))

        #Create a reduced set of fiducials and germs
        gateLabels = list(cls.gs_redmod_datagen.gates.keys())
        fids1Q = std1Q_XY.fiducials[0:2] # for speed
        cls.redmod_fiducials = []
        for i in range(cls.nQubits):
            cls.redmod_fiducials.extend( pygsti.construction.manipulate_gatestring_list(
                fids1Q, [ ( (L('Gx'),) , (L('Gx',i),) ), ( (L('Gy'),) , (L('Gy',i),) ) ]) )
        #print(redmod_fiducials, "Fiducials")     
        
        cls.redmod_germs = pygsti.construction.gatestring_list([ (gl,) for gl in gateLabels ])
        cls.redmod_maxLs = [1]
        expList = pygsti.construction.make_lsgst_experiment_list(
            cls.gs_redmod_datagen, cls.redmod_fiducials, cls.redmod_fiducials,
            cls.redmod_germs, cls.redmod_maxLs)
        cls.redmod_ds = pygsti.construction.generate_fake_data(cls.gs_redmod_datagen, expList, 1000, "round", seed=1234)
        #print(len(expList)," reduced model sequences")

        #Random starting points - little kick so we don't get hung up at start
        np.random.seed(1234)
        cls.rand_start18 = np.random.random(18)*1e-6
        cls.rand_start25 = np.random.random(25)*1e-6
        cls.rand_start36 = np.random.random(36)*1e-6

        #Circuit Simulation circuits
        cls.csim_nQubits=3
        cls.circuit1 = pygsti.obj.Circuit(gatestring=('Gx','Gy'), num_lines=1) # 1-qubit circuit
        cls.circuit3 = pygsti.obj.Circuit(gatestring=[ ('Gxpi',0), ('Gypi',1),('Gcnot',1,2)], num_lines=3) # 3-qubit circuit
        
    ## GST using "full" (non-embedded/composed) gates
    # All of these calcs use dense matrices; While sparse gate matrices (as Maps) could be used,
    # they'd need to enter as a sparse basis to a LindbladParameterizedGate (maybe add this later?)
    
    def test_stdgst_matrix(self):
        # Using matrix-based calculations
        gs_target = std.gs_target.copy()
        gs_target.set_all_parameterizations("CPTP")
        gs_target.set_simtype('matrix') # the default for 1Q, so we could remove this line
        results = pygsti.do_long_sequence_gst(self.ds, gs_target, std.prepStrs, std.effectStrs,
                                              std.germs, self.maxLengths, verbosity=4)


    def test_stdgst_map(self):
        # Using map-based calculation
        gs_target = std.gs_target.copy()
        gs_target.set_all_parameterizations("CPTP")
        gs_target.set_simtype('map')
        results = pygsti.do_long_sequence_gst(self.ds, gs_target, std.prepStrs, std.effectStrs,
                                              std.germs, self.maxLengths, verbosity=4)

    def test_stdgst_terms(self):
        # Using term-based (path integral) calculation
        # This performs a map-based unitary evolution along each path. 
        gs_target = std.gs_target.copy()
        gs_target.set_all_parameterizations("H+S terms")
        gs_target.set_simtype('termorder:1') # this is the default set by set_all_parameterizations above
        results = pygsti.do_long_sequence_gst(self.ds, gs_target, std.prepStrs, std.effectStrs,
                                              std.germs, self.maxLengths, verbosity=1)    


    # ## GST using "reduced" models
    # Reduced, meaning that we use composed and embedded gates to form a more complex error model with
    # shared parameters and qubit connectivity graphs.  Calculations *can* use dense matrices and matrix calcs,
    # but usually will use sparse mxs and map-based calcs.

    def test_reducedmod_matrix(self):
        # Using dense matrices and matrix-based calcs
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=False,
                                             sim_type="matrix", verbosity=1)
        gs_target.from_vector(self.rand_start25)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})

    def test_reducedmod_map1(self):
        # Using dense embedded matrices and map-based calcs (maybe not really necessary to include?)
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=False,
                                             sim_type="map", verbosity=1)
        gs_target.from_vector(self.rand_start25)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})


    def test_reducedmod_map2(self):
        # Using sparse embedded matrices and map-based calcs
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=True,
                                             sim_type="map", verbosity=1)
        gs_target.from_vector(self.rand_start25)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})

    def test_reducedmod_svterm(self):
        # Using term-based calcs using map-based state-vector propagation
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                      extraWeight1Hops=0, extraGateWeight=1, sparse=False, verbosity=1,
                                      sim_type="termorder:1", parameterization="H+S terms")
        gs_target.from_vector(self.rand_start36)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})

    def test_reducedmod_cterm(self):
        # Using term-based calcs using map-based stabilizer-state propagation
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=False, verbosity=1,
                                             sim_type="termorder:1", parameterization="H+S clifford terms")
        gs_target.from_vector(self.rand_start36)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})


    # ### Circuit Simulation

    def test_circuitsim_densitymx(self):
        # Density-matrix simulation (of superoperator gates)
        # These are the typical type of simulations used within GST.
        # The probability calculations can be done in a matrix- or map-based way.

        #Using simple "std" gatesets (which are all density-matrix/superop type)
        gs = std.gs_target.copy()
        gs.probs(self.circuit1)
        self.circuit1.simulate(gs) # calls probs - same as above line

        gs2 = std.gs_target.copy()
        gs2.set_simtype("map")
        gs2.probs(self.circuit1)
        self.circuit1.simulate(gs2) # calls probs - same as above line

        #Using n-qubit gatesets
        gs = pygsti.construction.build_nqubit_standard_gateset(
            self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], sim_type="matrix")
        gs.probs(self.circuit3)

        gs = pygsti.construction.build_nqubit_standard_gateset(
            self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], sim_type="map")
        gs.probs(self.circuit3)

    def test_circuitsim_statevec(self):
        # State-vector simulation (of unitary gates)
        # This can be done with matrix- or map-based calculations.

        #Unitary gateset in pygsti (from scratch, since "std" modules don't include them)
        sigmax = np.array([[0,1],[1,0]])
        sigmay = np.array([[0,-1.0j],[1.0j,0]])
        sigmaz = np.array([[1,0],[0,-1]])

        def Ugate(exp):
            return np.array(spl.expm(-1j * exp/2),complex) # 2x2 unitary matrix operating on single qubit in [0,1] basis

        #Create a gateset with unitary gates and state vectors (instead of the usual superoperators and density mxs)
        gs = pygsti.obj.GateSet(sim_type="matrix")
        gs.gates['Gi'] = pygsti.obj.StaticGate( np.identity(2,'complex') )
        gs.gates['Gx'] = pygsti.obj.StaticGate(Ugate(np.pi/2 * sigmax))
        gs.gates['Gy'] = pygsti.obj.StaticGate(Ugate(np.pi/2 * sigmay))
        gs.preps['rho0'] = pygsti.obj.StaticSPAMVec( [1,0], 'statevec')
        gs.povms['Mdefault'] = pygsti.obj.UnconstrainedPOVM(
            {'0': pygsti.obj.StaticSPAMVec( [1,0], 'statevec'),
             '1': pygsti.obj.StaticSPAMVec( [0,1], 'statevec')})
             
        gs.probs(self.circuit1)
        self.circuit1.simulate(gs) # calls probs - same as above line

        gs2 = gs.copy()
        gs2.set_simtype("map")
        gs2.probs(self.circuit1)
        self.circuit1.simulate(gs2) # calls probs - same as above line

        #Using n-qubit gatesets
        gs = pygsti.construction.build_nqubit_standard_gateset(
            self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], evotype="statevec", sim_type="matrix")
        gs.probs(self.circuit3)
        gs = pygsti.construction.build_nqubit_standard_gateset(
            self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'],  evotype="statevec", sim_type="map")
        gs.probs(self.circuit3)


    def test_circuitsim_svterm(self):
        # ### Density-matrix simulation (of superoperator gates) using map/matrix-based terms calcs
        # In this mode, "term calcs" use many state-vector propagation paths to simulate density
        # matrix propagation up to some desired order (in the assumed-to-be-small error rates).
        gs = std.gs_target.copy()
        gs.set_simtype('termorder:1') # 1st-order in error rates
        gs.set_all_parameterizations("H+S terms")

        gs.probs(self.circuit1)
        self.circuit1.simulate(gs) # calls probs - same as above line

        #Using n-qubit gatesets ("H+S terms" parameterization constructs embedded/composed gates containing LindbladTermGates, etc.)
        gs = pygsti.construction.build_nqubit_standard_gateset(
            self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], sim_type="termorder:1", parameterization="H+S terms")
        gs.probs(self.circuit3)
        self.circuit3.simulate(gs) # calls probs - same as above line


    def test_circuitsim_stabilizer(self):
        # Stabilizer-state simulation (of Clifford gates) using map-based calc
        c0 = pygsti.obj.Circuit(gatestring=(), num_lines=1) # 1-qubit circuit
        c1 = pygsti.obj.Circuit(gatestring=(('Gx',0),), num_lines=1) 
        c2 = pygsti.obj.Circuit(gatestring=(('Gx',0),('Gx',0)), num_lines=1)
        c3 = pygsti.obj.Circuit(gatestring=(('Gx',0),('Gx',0),('Gx',0),('Gx',0)), num_lines=1)

        gs = pygsti.construction.build_nqubit_standard_gateset(
            1, ['Gi','Gx','Gy'], parameterization="clifford")

        print(gs.probs(c0))
        print(gs.probs(c1))
        print(gs.probs(c2))
        print(gs.probs(c3))

    def test_circuitsim_stabilizer_2Qcheck(self):
        #Test 2Q circuits
        #from pygsti.construction import std1Q_XYI as std
        #from pygsti.construction import std2Q_XYICNOT as std
        from pygsti.construction import std2Q_XYICPHASE as stdChk

        maxLengths = [1,2,4]
        listOfExperiments = pygsti.construction.make_lsgst_experiment_list(
            stdChk.gs_target, stdChk.prepStrs, stdChk.effectStrs, stdChk.germs, maxLengths)
        #listOfExperiments = pygsti.construction.gatestring_list([ ('Gcnot','Gxi') ])
        #listOfExperiments = pygsti.construction.gatestring_list([ ('Gxi','Gcphase','Gxi','Gix') ])

        gs_normal = stdChk.gs_target.copy()
        gs_clifford = stdChk.gs_target.copy()
        #print(gs_clifford['Gcnot'])
        self.assertTrue(stdChk.gs_target._evotype == "densitymx")
        gs_clifford.set_all_parameterizations('static unitary') # reduces dim...
        self.assertTrue(gs_clifford._evotype == "statevec")
        gs_clifford.set_all_parameterizations('clifford')
        self.assertTrue(gs_clifford._evotype == "stabilizer")

        for gstr in listOfExperiments:
            #print(str(gstr))
            p_normal = gs_normal.probs(gstr)
            p_clifford = gs_clifford.probs(gstr)
            #p_clifford = bprobs[gstr]
            for outcm in p_normal.keys():
                if abs(p_normal[outcm]-p_clifford[outcm]) > 1e-8:
                    print(str(gstr)," ERR: \n",p_normal,"\n",p_clifford);
                    self.assertTrue(False)
        print("Done checking %d sequences!" % len(listOfExperiments))

    def test_circuitsim_cterm(self):
        # Density-matrix simulation (of superoperator gates) using stabilizer-based term calcs
        c0 = pygsti.obj.Circuit(gatestring=(), num_lines=1) # 1-qubit circuit
        c1 = pygsti.obj.Circuit(gatestring=(('Gx',0),), num_lines=1) 
        c2 = pygsti.obj.Circuit(gatestring=(('Gx',0),('Gx',0)), num_lines=1)
        c3 = pygsti.obj.Circuit(gatestring=(('Gx',0),('Gx',0),('Gx',0),('Gx',0)), num_lines=1)
        
        gs = pygsti.construction.build_nqubit_standard_gateset(
            1, ['Gi','Gx','Gy'], sim_type="termorder:1", parameterization="H+S clifford terms")

        print(gs.probs(c0))
        print(gs.probs(c1))
        print(gs.probs(c2))
        print(gs.probs(c3))



if __name__ == "__main__":
    unittest.main(verbosity=2)
