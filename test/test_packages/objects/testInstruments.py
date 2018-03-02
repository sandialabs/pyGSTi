import unittest
import pygsti
import numpy as np
import warnings
import pickle
import os

from pygsti.construction import std1Q_XYI as std

from ..testutils import BaseTestCase, compare_files, temp_files

# This class is for unifying some gatesets that get used in this file and in testGateSets2.py
class InstrumentTestCase(BaseTestCase):

    def setUp(self):
        #Add an instrument to the standard target gate set
        self.gs_target = std.gs_target.copy()
        E = self.gs_target.povms['Mdefault']['0']
        Erem = self.gs_target.povms['Mdefault']['1']
        Gmz_plus = np.dot(E,E.T)
        Gmz_minus = np.dot(Erem,Erem.T)
        self.gs_target.instruments['Iz'] = pygsti.obj.Instrument({'plus': Gmz_plus, 'minus': Gmz_minus})
        self.povm_ident = self.gs_target.povms['Mdefault']['0'] + self.gs_target.povms['Mdefault']['1']

        self.gs_target_wTP = self.gs_target.copy()
        self.gs_target_wTP.instruments['IzTP'] = pygsti.obj.TPInstrument({'plus': Gmz_plus, 'minus': Gmz_minus})

        super(InstrumentTestCase, self).setUp()

    def testFailures(self):
        with self.assertRaises(ValueError):
            pygsti.objects.instrument.convert( self.gs_target.instruments["Iz"], "foobar", self.gs_target.basis)

        #Constructor
        with self.assertRaises(AssertionError):
            pygsti.obj.Instrument(["Non-none-matrices"], ["Non-none-items"]) #can't both be non-None
        with self.assertRaises(ValueError):
            pygsti.obj.Instrument("foobar") #gate_matrices must be a list or dict

        #TP Constructor
        with self.assertRaises(AssertionError):
            pygsti.obj.TPInstrument(["Non-none-matrices"], ["Non-none-items"]) #can't both be non-None
        with self.assertRaises(ValueError):
            pygsti.obj.TPInstrument("foobar") #gate_matrices must be a list or dict

        tpi = self.gs_target_wTP.instruments["IzTP"]
        with self.assertRaises(ValueError):
            tpi['plus'] = None #can't set value of a TP Instrument element

            
    def testFutureFunctionality(self):
        #Test instrument construction with elements whose gpindices are already initialized.
        # Since this isn't allowed currently (a future functionality), we need to do some hacking
        E = self.gs_target.povms['Mdefault']['0']
        InstEl = pygsti.obj.FullyParameterizedGate( np.dot(E,E.T) )
        InstEl2 = InstEl.copy()
        nParams = InstEl.num_params() # should be 16
        
        I = pygsti.obj.Instrument({})
        InstEl.set_gpindices(slice(0,16), I)
        InstEl2.set_gpindices(slice(8,24), I) # some overlap - to test _build_paramvec

        # TESTING ONLY - so we can add items!!!
        I._readonly = False 
        I['A'] = InstEl
        I['B'] = InstEl2
        I._readonly = True

        I._paramvec = I._build_paramvec()
          # this whole test was to exercise this function's ability to
          # form a parameter vector with weird overlapping gpindices.
        self.assertEqual( len(I._paramvec) , 24 )


    def testInstrumentMethods(self):
        
        v = self.gs_target_wTP.to_vector()
        
        gs = self.gs_target_wTP.copy()
        gs.from_vector(v)
        gs.basis = self.gs_target_wTP.basis.copy()
        
        self.assertAlmostEqual(gs.frobeniusdist(self.gs_target_wTP),0.0)

        for lbl in ('Iz','IzTP'):
            deriv = gs.instruments[lbl]['plus'].deriv_wrt_params()
            try:
                self.assertEqual(deriv.shape[1], gs.instruments[lbl]['plus'].num_params())
            except NotImplementedError:
                pass # TPInstrumentGate doesn't implement num_params (yet?)
            deriv = gs.instruments[lbl]['plus'].deriv_wrt_params([0])
            self.assertEqual(deriv.shape[1], 1)

            #DON'T do this -- we *could* alter an Instrument's constituents, but it doesn't update
            # the instrument's underlying paramvec (yet)
            #try:
            #    gs.instruments[lbl]['plus'].set_value(np.identity(4,'d'))
            #except ValueError:
            #    pass # OK: not allowed for TPInstrumentGate objects

            nEls =  gs.instruments[lbl].num_elements()
            igate = gs.instruments[lbl]['plus'].copy()
            print("igate = ",str(igate))
            str_inst = str(gs.instruments[lbl])
            
            inst_copy = gs.instruments[lbl].copy()
            T = pygsti.objects.FullGaugeGroupElement(
                np.array( [ [1,0,0,0],
                            [0,1,0,0],
                            [0,0,0,1],
                            [0,0,1,0] ], 'd') )
            inst_copy.transform(T)

            v = gs.to_vector()
            gates = gs.instruments[lbl].compile_gates(prefix="ABC")
            for igate in gates.values():
                igate.from_vector(v[igate.gpindices]) # gpindices should be setup relative to GateSet's param vec

        gs.depolarize(0.01)
        gs.rotate((0,0,0.01))
        gs.rotate(max_rotate=0.01, seed=1234)

    def testChangeDimension(self):
        gs = self.gs_target.copy()
        new_gs = gs.increase_dimension(6)
        new_gs = gs.decrease_dimension(3)

        #TP
        gs = self.gs_target.copy()
        gs.set_all_parameterizations("TP")
        new_gs = gs.increase_dimension(6)
        new_gs = gs.decrease_dimension(3)

        
    def testIntermediateMeas(self):
        # Mess with the target gateset to add some error to the povm and instrument
        self.assertEqual(self.gs_target.num_params(),92) # 4*3 + 16*5 = 92
        gs = self.gs_target.depolarize(gate_noise=0.01, spam_noise=0.01)
        gs2 = self.gs_target.depolarize(max_gate_noise=0.01, max_spam_noise=0.01, seed=1234) #another way to depolarize
        gs.povms['Mdefault'].depolarize(0.01)

        # Introducing a rotation error to the measurement
        Uerr = pygsti.rotation_gate_mx([0,0.02,0]) # input angles are halved by the method
        E = np.dot(gs.povms['Mdefault']['0'].T,Uerr).T # effect is stored as column vector
        Erem = self.povm_ident - E
        gs.povms['Mdefault'] = pygsti.obj.UnconstrainedPOVM({'0': E, '1': Erem})
        
        # Now add the post-measurement gates from the vector E0 and remainder = id-E0
        Gmz_plus = np.dot(E,E.T) #since E0 is stored internally as column spamvec
        Gmz_minus = np.dot(Erem,Erem.T)
        gs.instruments['Iz'] = pygsti.obj.Instrument({'plus': Gmz_plus, 'minus': Gmz_minus})
        self.assertEqual(gs.num_params(),92) # 4*3 + 16*5 = 92
        #print(gs)

        germs = std.germs
        fiducials = std.fiducials
        max_lengths = [1] #,2,4,8]
        glbls = list(gs.gates.keys()) + list(gs.instruments.keys())
        lsgst_list = pygsti.construction.make_lsgst_experiment_list(
            glbls,fiducials,fiducials,germs,max_lengths)
        lsgst_list2 = pygsti.construction.make_lsgst_experiment_list(
            gs,fiducials,fiducials,germs,max_lengths) #use gs as source
        self.assertEqual(lsgst_list, lsgst_list2)

        
        
        gs_datagen = gs
        ds = pygsti.construction.generate_fake_data(gs,lsgst_list,1000,'none') #'multinomial')
        pygsti.io.write_dataset(temp_files + "/intermediate_meas_dataset.txt",ds)
        ds2 = pygsti.io.load_dataset(temp_files + "/intermediate_meas_dataset.txt")
        for gstr,dsRow in ds.items():
            for lbl,cnt in dsRow.counts.items():
                self.assertAlmostEqual(cnt, ds2[gstr].counts[lbl],places=2)
        #print(ds)
        
        #LGST
        gs_lgst = pygsti.do_lgst(ds, fiducials,fiducials, self.gs_target) #, guessGatesetForGauge=gs_datagen)
        self.assertTrue("Iz" in gs_lgst.instruments)
        gs_opt = pygsti.gaugeopt_to_target(gs_lgst,gs_datagen) #, method="BFGS")
        print(gs_datagen.strdiff(gs_opt))
        print("Frobdiff = ",gs_datagen.frobeniusdist( gs_lgst))
        print("Frobdiff after GOpt = ",gs_datagen.frobeniusdist(gs_opt))
        self.assertAlmostEqual(gs_datagen.frobeniusdist(gs_opt), 0.0, places=4)
        #print(gs_lgst)
        #print(gs_datagen)
        
    
        #LSGST
        results = pygsti.do_long_sequence_gst(ds,self.gs_target,fiducials,fiducials,germs,max_lengths)
        #print(results.estimates['default'].gatesets['go0'])
        gs_est = results.estimates['default'].gatesets['go0']
        gs_est_opt = pygsti.gaugeopt_to_target(gs_est,gs_datagen)
        print("Frobdiff = ", gs_datagen.frobeniusdist(gs_est))
        print("Frobdiff after GOpt = ", gs_datagen.frobeniusdist(gs_est_opt))
        self.assertAlmostEqual(gs_datagen.frobeniusdist(gs_est_opt), 0.0, places=4)
        
        #LGST w/TP gates
        gs_targetTP = self.gs_target.copy()
        gs_targetTP.set_all_parameterizations("TP")
        self.assertEqual(gs_targetTP.num_params(),71) # 3 + 4*2 + 12*5 = 71
        #print(gs_targetTP)
        resultsTP = pygsti.do_long_sequence_gst(ds,gs_targetTP,fiducials,fiducials,germs,max_lengths)
        gs_est = resultsTP.estimates['default'].gatesets['go0']
        gs_est_opt = pygsti.gaugeopt_to_target(gs_est,gs_datagen)
        print("TP Frobdiff = ", gs_datagen.frobeniusdist(gs_est))
        print("TP Frobdiff after GOpt = ", gs_datagen.frobeniusdist(gs_est_opt))
        self.assertAlmostEqual(gs_datagen.frobeniusdist(gs_est_opt), 0.0, places=4)
        
    def testBasicGatesetOps(self):
        # This test was made from a debug script used to get the code working
        gateset = pygsti.construction.build_gateset(
            [2], [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"])
        #    prepLabels=["rho0"], prepExpressions=["0"],
        #    effectLabels=["0","1"], effectExpressions=["0","complement"])
        
        v0 = pygsti.construction.basis_build_vector("0", pygsti.obj.Basis("pp",2))
        v1 = pygsti.construction.basis_build_vector("1", pygsti.obj.Basis("pp",2))
        P0 = np.dot(v0,v0.T)
        P1 = np.dot(v1,v1.T)
        #print("v0 = ",v0)
        #print("P0 = ",P0)
        #print("P0+P1 = ",P0+P1)
        
        gateset.instruments["Itest"] = pygsti.obj.Instrument( [('0',P0),('1',P1)] )
        
        for param in ("full","TP","CPTP"):
            print(param)
            gateset.set_all_parameterizations(param)
            for lbl,obj in gateset.preps.items():
                print(lbl,':',obj.gpindices, pygsti.tools.length(obj.gpindices))
            for lbl,obj in gateset.povms.items():
                print(lbl,':',obj.gpindices, pygsti.tools.length(obj.gpindices))
                for sublbl,subobj in obj.items():
                    print("  > ",sublbl,':',subobj.gpindices, pygsti.tools.length(subobj.gpindices))
            for lbl,obj in gateset.gates.items():
                print(lbl,':',obj.gpindices, pygsti.tools.length(obj.gpindices))
            for lbl,obj in gateset.instruments.items():
                print(lbl,':',obj.gpindices, pygsti.tools.length(obj.gpindices))
                for sublbl,subobj in obj.items():
                    print("  > ",sublbl,':',subobj.gpindices, pygsti.tools.length(subobj.gpindices))
        
        
            print("NPARAMS = ",gateset.num_params())
            print("")
        
        
        print("PICKLING")
        
        x = gateset.preps #.copy(None)
        p = pickle.dumps(x) #gateset.preps)
        print("loading")
        preps = pickle.loads(p)
        self.assertEqual(list(preps.keys()),list(gateset.preps.keys()))
        
        #p = pickle.dumps(gateset.effects)
        #effects = pickle.loads(p)
        #assert(list(effects.keys()) == list(gateset.effects.keys()))
        
        p = pickle.dumps(gateset.gates)
        gates = pickle.loads(p)
        self.assertEqual(list(gates.keys()),list(gateset.gates.keys()))
        
        p = pickle.dumps(gateset)
        g = pickle.loads(p)
        self.assertAlmostEqual(gateset.frobeniusdist(g), 0.0)
        
        
        print("GateSet IO")
        pygsti.io.write_gateset(gateset, temp_files + "/testGateset.txt")
        gateset2 = pygsti.io.load_gateset(temp_files + "/testGateset.txt")
        self.assertAlmostEqual(gateset.frobeniusdist(gateset2),0.0)
        
        print("Multiplication")
        
        gatestring1 = ('Gx','Gy')
        gatestring2 = ('Gx','Gy','Gy')
        
        p1 = np.dot( gateset.gates['Gy'], gateset.gates['Gx'] )
        p2 = gateset.product(gatestring1, bScale=False)
        p3,scale = gateset.product(gatestring1, bScale=True)
        
        print(p1)
        print(p2)
        print(p3*scale)
        self.assertAlmostEqual(np.linalg.norm(p1-scale*p3),0.0)
        
        dp = gateset.dproduct(gatestring1)
        dp_flat = gateset.dproduct(gatestring1,flat=True)
        
        evt, lookup, outcome_lookup = gateset.bulk_evaltree( [gatestring1,gatestring2] )
        
        p1 = np.dot( gateset.gates['Gy'], gateset.gates['Gx'] )
        p2 = np.dot( gateset.gates['Gy'], np.dot( gateset.gates['Gy'], gateset.gates['Gx'] ))
        
        bulk_prods = gateset.bulk_product(evt)
        bulk_prods_scaled, scaleVals = gateset.bulk_product(evt, bScale=True)
        bulk_prods2 = scaleVals[:,None,None] * bulk_prods_scaled
        self.assertArraysAlmostEqual(bulk_prods[0],p1)
        self.assertArraysAlmostEqual(bulk_prods[1],p2)
        self.assertArraysAlmostEqual(bulk_prods2[0],p1)
        self.assertArraysAlmostEqual(bulk_prods2[1],p2)
        
        print("Probabilities")
        gatestring1 = ('Gx','Gy') #,'Itest')
        gatestring2 = ('Gx','Gy','Gy')
        
        evt, lookup, outcome_lookup = gateset.bulk_evaltree( [gatestring1,gatestring2] )
        
        p1 = np.dot( np.transpose(gateset.povms['Mdefault']['0']),
                     np.dot( gateset.gates['Gy'],
                             np.dot(gateset.gates['Gx'],
                                    gateset.preps['rho0'])))
        probs = gateset.probs(gatestring1)
        print(probs)
        p20,p21 = probs[('0',)],probs[('1',)]
        
        #probs = gateset.probs(gatestring1, bUseScaling=True)
        #print(probs)
        #p30,p31 = probs['0'],probs['1']
        
        self.assertArraysAlmostEqual(p1,p20)
        #assertArraysAlmostEqual(p1,p30)
        #assertArraysAlmostEqual(p21,p31)
        
        bulk_probs = gateset.bulk_probs([gatestring1,gatestring2],check=True)
        
        evt_split = evt.copy()
        new_lookup = evt_split.split(lookup, numSubTrees=2)
        print("SPLIT TREE: new elIndices = ",new_lookup)
        probs_to_fill = np.empty( evt_split.num_final_elements(), 'd')
        gateset.bulk_fill_probs(probs_to_fill,evt_split,check=True)
        
        dProbs = gateset.dprobs(gatestring1)
        bulk_dProbs = gateset.bulk_dprobs([gatestring1,gatestring2], returnPr=False, check=True)
        
        hProbs = gateset.hprobs(gatestring1)
        bulk_hProbs = gateset.bulk_hprobs([gatestring1,gatestring2], returnPr=False, check=True)
        
        
        print("DONE")

    def testAdvancedGateStrs(self):
        #specify prep and/or povm labels in gate string:
        gs_normal = pygsti.obj.GateString( ('Gx',) )
        gs_wprep = pygsti.obj.GateString( ('rho0','Gx') )
        gs_wpovm = pygsti.obj.GateString( ('Gx','Mdefault') )
        gs_wboth = pygsti.obj.GateString( ('rho0','Gx','Mdefault') )

        #Now compute probabilities for these:
        gateset = self.gs_target.copy()
        probs_normal = gateset.probs(gs_normal)
        probs_wprep = gateset.probs(gs_wprep)
        probs_wpovm = gateset.probs(gs_wpovm)
        probs_wboth = gateset.probs(gs_wboth)

        print(probs_normal)
        print(probs_wprep)
        print(probs_wpovm)
        print(probs_wboth)

        self.assertEqual( probs_normal, probs_wprep )
        self.assertEqual( probs_normal, probs_wpovm )
        self.assertEqual( probs_normal, probs_wboth )

        #now try bulk op
        bulk_probs = gateset.bulk_probs([gs_normal, gs_wprep, gs_wpovm, gs_wboth],check=True)

    def testWriteAndLoad(self):
        gs = self.gs_target.copy()

        s = str(gs) #stringify with instruments

        for param in ('full','TP','CPTP','static'):
            print("Param: ",param)
            gs.set_all_parameterizations(param)
            filename = temp_files + "/gateset_with_instruments_%s.txt" % param
            pygsti.io.write_gateset(gs, filename)
            gs2 = pygsti.io.read_gateset(filename)
            self.assertAlmostEqual( gs.frobeniusdist(gs2), 0.0 )
            for lbl in gs.gates:
                self.assertEqual( type(gs.gates[lbl]), type(gs2.gates[lbl]))
            for lbl in gs.preps:
                self.assertEqual( type(gs.preps[lbl]), type(gs2.preps[lbl]))
            for lbl in gs.povms:
                self.assertEqual( type(gs.povms[lbl]), type(gs2.povms[lbl]))
            for lbl in gs.instruments:
                self.assertEqual( type(gs.instruments[lbl]), type(gs2.instruments[lbl]))
