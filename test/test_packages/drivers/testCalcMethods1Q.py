from __future__ import print_function
import unittest
import numpy as np
import scipy.linalg as spl
import pygsti
import pygsti.construction as pc
from pygsti.construction import std1Q_XYI as std
from pygsti.construction import std1Q_XY
from pygsti.objects import Label as L
from pygsti.io import json

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

        #Change to test_packages directory (since setUp hasn't been called yet...)
        origDir = os.getcwd()
        os.chdir(os.path.abspath(os.path.dirname(__file__)))
        os.chdir('..') # The test_packages directory

        #Standard GST dataset
        cls.maxLengths = [1,2,4]
        cls.gs_datagen = std.gs_target.depolarize(gate_noise=0.1, spam_noise=0.001)
        cls.listOfExperiments = pygsti.construction.make_lsgst_experiment_list(
            std.gs_target, std.prepStrs, std.effectStrs, std.germs, cls.maxLengths)

        #RUN BELOW FOR DATAGEN (UNCOMMENT to regenerate)
        #ds = pygsti.construction.generate_fake_data(cls.gs_datagen, cls.listOfExperiments,
        #                                                 nSamples=1000, sampleError="multinomial", seed=1234)
        #ds.save(compare_files + "/calcMethods1Q.dataset%s" % cls.versionsuffix)

        cls.ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/calcMethods1Q.dataset%s" % cls.versionsuffix)

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

        #RUN BELOW FOR DATAGEN (UNCOMMENT to regenerate)
        #redmod_ds = pygsti.construction.generate_fake_data(cls.gs_redmod_datagen, expList, 1000, "round", seed=1234)
        #redmod_ds.save(compare_files + "/calcMethods1Q_redmod.dataset%s" % cls.versionsuffix)

        cls.redmod_ds = pygsti.objects.DataSet(fileToLoadFrom=compare_files + "/calcMethods1Q_redmod.dataset%s" % cls.versionsuffix)
        
        #print(len(expList)," reduced model sequences")

        #Random starting points - little kick so we don't get hung up at start
        np.random.seed(1234)
        cls.rand_start18 = np.random.random(18)*1e-6
        cls.rand_start25 = np.random.random(25)*1e-6
        cls.rand_start36 = np.random.random(36)*1e-6

        #Circuit Simulation circuits
        cls.csim_nQubits=3
        cls.circuit1 = pygsti.obj.GateString(('Gx','Gy'))
          # now Circuit adds qubit labels... pygsti.obj.Circuit(gatestring=('Gx','Gy'), num_lines=1) # 1-qubit circuit
        cls.circuit3 = pygsti.obj.Circuit(gatestring=[ ('Gxpi',0), ('Gypi',1),('Gcnot',1,2)], num_lines=3) # 3-qubit circuit

        os.chdir(origDir) # return to original directory


    def assert_outcomes(self, probs, expected):
        for k,v in probs.items():
            self.assertAlmostEqual(v, expected[k])
        
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
        #RUN BELOW LINES TO SAVE GATESET (UNCOMMENT to regenerate)
        #pygsti.io.json.dump(results.estimates['default'].gatesets['go0'],
        #                    open(compare_files + "/test1Qcalc_std_exact.gateset",'w'))

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 3.0, delta=2.0)
        gs_compare = pygsti.io.json.load(open(compare_files + "/test1Qcalc_std_exact.gateset"))

        #gauge opt before compare
        gsEstimate = results.estimates['default'].gatesets['go0'].copy()
        gsEstimate.set_all_parameterizations("full")
        gsEstimate = pygsti.algorithms.gaugeopt_to_target(gsEstimate, gs_compare)
        print(gsEstimate.strdiff(gs_compare))
        self.assertAlmostEqual( gsEstimate.frobeniusdist(gs_compare), 0, places=3)


    def test_stdgst_map(self):
        # Using map-based calculation
        gs_target = std.gs_target.copy()
        gs_target.set_all_parameterizations("CPTP")
        gs_target.set_simtype('map')
        results = pygsti.do_long_sequence_gst(self.ds, gs_target, std.prepStrs, std.effectStrs,
                                              std.germs, self.maxLengths, verbosity=4)

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 3.0, delta=2.0)
        gs_compare = pygsti.io.json.load(open(compare_files + "/test1Qcalc_std_exact.gateset"))

        gsEstimate = results.estimates['default'].gatesets['go0'].copy()
        gsEstimate.set_all_parameterizations("full")
        gsEstimate = pygsti.algorithms.gaugeopt_to_target(gsEstimate, gs_compare)
        self.assertAlmostEqual( gsEstimate.frobeniusdist(gs_compare), 0, places=0)
         # with low tolerance (1e-6), "map" tends to go for more iterations than "matrix",
         # resulting in a gateset that isn't exactly the same as the "matrix" one


    def test_stdgst_terms(self):
        # Using term-based (path integral) calculation
        # This performs a map-based unitary evolution along each path. 
        gs_target = std.gs_target.copy()
        gs_target.set_all_parameterizations("H+S terms")
        gs_target.set_simtype('termorder:1') # this is the default set by set_all_parameterizations above
        results = pygsti.do_long_sequence_gst(self.ds, gs_target, std.prepStrs, std.effectStrs,
                                              std.germs, self.maxLengths, verbosity=1)

        #RUN BELOW LINES TO SAVE GATESET (UNCOMMENT to regenerate)
        #pygsti.io.json.dump(results.estimates['default'].gatesets['go0'],
        #                    open(compare_files + "/test1Qcalc_std_terms.gateset",'w'))

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 5, delta=1.0)
        gs_compare = pygsti.io.json.load(open(compare_files + "/test1Qcalc_std_terms.gateset"))

        # can't easily gauge opt b/c term-based gatesets can't be converted to "full"
        #gs_compare.set_all_parameterizations("full")
        #
        #gsEstimate = results.estimates['default'].gatesets['go0'].copy()
        #gsEstimate.set_all_parameterizations("full")
        #gsEstimate = pygsti.algorithms.gaugeopt_to_target(gsEstimate, gs_compare)
        #self.assertAlmostEqual( gsEstimate.frobeniusdist(gs_compare), 0, places=0)

        #A direct vector comparison works if python (&numpy?) versions are identical, but
        # gauge freedoms make this incorrectly fail in other cases - so just check sigmas
        print("VEC DIFF = ",(results.estimates['default'].gatesets['go0'].to_vector()
                                               - gs_compare.to_vector()))
        self.assertAlmostEqual( np.linalg.norm(results.estimates['default'].gatesets['go0'].to_vector()
                                               - gs_compare.to_vector()), 0, places=3)



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

        #RUN BELOW LINES TO SAVE GATESET (UNCOMMENT to regenerate)
        #pygsti.io.json.dump(results.estimates['default'].gatesets['go0'],
        #                    open(compare_files + "/test1Qcalc_redmod_exact.gateset",'w'))

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 1.0, delta=1.0)
        gs_compare = pygsti.io.json.load( open(compare_files + "/test1Qcalc_redmod_exact.gateset"))
        self.assertAlmostEqual( results.estimates['default'].gatesets['go0'].frobeniusdist(gs_compare), 0, places=3)


    def test_reducedmod_map1(self):
        # Using dense embedded matrices and map-based calcs (maybe not really necessary to include?)
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=False,
                                             sim_type="map", verbosity=1)
        gs_target.from_vector(self.rand_start25)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 1.0, delta=1.0)
        gs_compare = pygsti.io.json.load( open(compare_files + "/test1Qcalc_redmod_exact.gateset"))
        self.assertAlmostEqual( results.estimates['default'].gatesets['go0'].frobeniusdist(gs_compare), 0, places=1)
          #Note: gatesets aren't necessarily exactly equal given gauge freedoms that we don't know
          # how to optimizize over exactly - so this is a very loose test...


    def test_reducedmod_map2(self):
        # Using sparse embedded matrices and map-based calcs
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=True,
                                             sim_type="map", verbosity=1)
        gs_target.from_vector(self.rand_start25)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 1.0, delta=1.0)
        gs_compare = pygsti.io.json.load( open(compare_files + "/test1Qcalc_redmod_exact.gateset"))
        self.assertAlmostEqual( np.linalg.norm(results.estimates['default'].gatesets['go0'].to_vector()
                                               - gs_compare.to_vector()), 0, places=1)
          #Note: gatesets aren't necessarily exactly equal given gauge freedoms that we don't know
          # how to optimizize over exactly - so this is a very loose test...



    def test_reducedmod_svterm(self):
        # Using term-based calcs using map-based state-vector propagation
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                      extraWeight1Hops=0, extraGateWeight=1, sparse=False, verbosity=1,
                                      sim_type="termorder:1", parameterization="H+S terms")
        gs_target.from_vector(self.rand_start36)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})

        #RUN BELOW LINES TO SAVE GATESET (UNCOMMENT to regenerate)
        #pygsti.io.json.dump(results.estimates['default'].gatesets['go0'],
        #                    open(compare_files + "/test1Qcalc_redmod_terms.gateset",'w'))

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 0.0, delta=1.0)
        gs_compare = pygsti.io.json.load( open(compare_files + "/test1Qcalc_redmod_terms.gateset"))
        self.assertAlmostEqual( np.linalg.norm(results.estimates['default'].gatesets['go0'].to_vector()
                                               - gs_compare.to_vector()), 0, places=3)


    def test_reducedmod_cterm(self):
        # Using term-based calcs using map-based stabilizer-state propagation
        gs_target = pc.build_nqnoise_gateset(self.nQubits, geometry="line", maxIdleWeight=1, maxhops=1,
                                             extraWeight1Hops=0, extraGateWeight=1, sparse=False, verbosity=1,
                                             sim_type="termorder:1", parameterization="H+S clifford terms")
        gs_target.from_vector(self.rand_start36)
        results = pygsti.do_long_sequence_gst(self.redmod_ds, gs_target, self.redmod_fiducials,
                                              self.redmod_fiducials, self.redmod_germs, self.redmod_maxLs,
                                              verbosity=4, advancedOptions={'tolerance': 1e-3})

        print("MISFIT nSigma = ",results.estimates['default'].misfit_sigma())
        self.assertAlmostEqual( results.estimates['default'].misfit_sigma(), 0.0, delta=1.0)
        gs_compare = pygsti.io.json.load( open(compare_files + "/test1Qcalc_redmod_terms.gateset"))
        self.assertAlmostEqual( np.linalg.norm(results.estimates['default'].gatesets['go0'].to_vector()
                                               - gs_compare.to_vector()), 0, places=3)



    # ### Circuit Simulation

    def test_circuitsim_densitymx(self):
        # Density-matrix simulation (of superoperator gates)
        # These are the typical type of simulations used within GST.
        # The probability calculations can be done in a matrix- or map-based way.

        #Using simple "std" gatesets (which are all density-matrix/superop type)
        gs = std.gs_target.copy()
        probs1 = gs.probs(self.circuit1)
        #self.circuit1.simulate(gs) # calls probs - same as above line
        print(probs1)

        gs2 = std.gs_target.copy()
        gs2.set_simtype("map")
        probs1 = gs2.probs(self.circuit1)
        #self.circuit1.simulate(gs2) # calls probs - same as above line
        print(probs1)
        self.assert_outcomes(probs1, {('0',): 0.5,  ('1',): 0.5} )

        #Using n-qubit gatesets
        gs = pygsti.construction.build_nqubit_standard_gateset(
            self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], sim_type="matrix")
        probs1 = gs.probs(self.circuit3)

        gs = pygsti.construction.build_nqubit_standard_gateset(
            self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], sim_type="map")
        probs2 = gs.probs(self.circuit3)

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
             
        probs1 = gs.probs(self.circuit1)
        #self.circuit1.simulate(gs) # calls probs - same as above line
        print(probs1)
        self.assert_outcomes(probs1, {('0',): 0.5,  ('1',): 0.5} )

        gs2 = gs.copy()
        gs2.set_simtype("map")
        gs2.probs(self.circuit1)
        #self.circuit1.simulate(gs2) # calls probs - same as above line

        #Using n-qubit gatesets
        gs = pygsti.construction.build_nqubit_standard_gateset(
            self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], evotype="statevec", sim_type="matrix")
        probs1 = gs.probs(self.circuit3)
        gs = pygsti.construction.build_nqubit_standard_gateset(
            self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'],  evotype="statevec", sim_type="map")
        probs2 = gs.probs(self.circuit3)

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
        gs = std.gs_target.copy()
        gs.set_simtype('termorder:1') # 1st-order in error rates
        gs.set_all_parameterizations("H+S terms")

        probs1 = gs.probs(self.circuit1)
        #self.circuit1.simulate(gs) # calls probs - same as above line

        print(probs1)
        self.assert_outcomes(probs1, {('0',): 0.5,  ('1',): 0.5} )

        #Using n-qubit gatesets ("H+S terms" parameterization constructs embedded/composed gates containing LindbladTermGates, etc.)
        gs = pygsti.construction.build_nqubit_standard_gateset(
            self.csim_nQubits, ['Gi','Gxpi','Gypi','Gcnot'], sim_type="termorder:1", parameterization="H+S terms")
        probs1 = gs.probs(self.circuit3)
        probs2 = self.circuit3.simulate(gs) # calls probs - same as above line
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
        c0 = pygsti.obj.Circuit(gatestring=(), num_lines=1) # 1-qubit circuit
        c1 = pygsti.obj.Circuit(gatestring=(('Gx',0),), num_lines=1) 
        c2 = pygsti.obj.Circuit(gatestring=(('Gx',0),('Gx',0)), num_lines=1)
        c3 = pygsti.obj.Circuit(gatestring=(('Gx',0),('Gx',0),('Gx',0),('Gx',0)), num_lines=1)

        gs = pygsti.construction.build_nqubit_standard_gateset(
            1, ['Gi','Gx','Gy'], parameterization="clifford")

        probs0 = gs.probs(c0)
        probs1 = gs.probs(c1)
        probs2 = gs.probs(c2)
        probs3 = gs.probs(c3)

        self.assert_outcomes(probs0, {('0',): 1.0,  ('1',): 0.0} )
        self.assert_outcomes(probs1, {('0',): 0.5,  ('1',): 0.5} )
        self.assert_outcomes(probs2, {('0',): 0.0,  ('1',): 1.0} )
        self.assert_outcomes(probs3, {('0',): 1.0,  ('1',): 0.0} )


    def test_circuitsim_stabilizer_1Qcheck(self): 
        from pygsti.construction import std1Q_XYI as stdChk

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

        probs0 = gs.probs(c0)
        probs1 = gs.probs(c1)
        probs2 = gs.probs(c2)
        probs3 = gs.probs(c3)

        self.assert_outcomes(probs0, {('0',): 1.0,  ('1',): 0.0} )
        self.assert_outcomes(probs1, {('0',): 0.5,  ('1',): 0.5} )
        self.assert_outcomes(probs2, {('0',): 0.0,  ('1',): 1.0} )
        self.assert_outcomes(probs3, {('0',): 1.0,  ('1',): 0.0} )


if __name__ == "__main__":
    unittest.main(verbosity=2)
