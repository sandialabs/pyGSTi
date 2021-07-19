import pickle

import numpy as np

import pygsti
from pygsti.models import modelconstruction
from pygsti.modelpacks.legacy import std1Q_XYI as std
from ..testutils import BaseTestCase, temp_files


# This class is for unifying some models that get used in this file and in testGateSets2.py
class InstrumentTestCase(BaseTestCase):

    def setUp(self):
        #Add an instrument to the standard target model
        self.target_model = std.target_model()
        E = self.target_model.povms['Mdefault']['0']
        Erem = self.target_model.povms['Mdefault']['1']
        Gmz_plus = np.dot(E,E.T)
        Gmz_minus = np.dot(Erem,Erem.T)
        self.target_model.instruments['Iz'] = pygsti.modelmembers.instruments.Instrument({'plus': Gmz_plus, 'minus': Gmz_minus})
        self.povm_ident = self.target_model.povms['Mdefault']['0'] + self.target_model.povms['Mdefault']['1']

        self.mdl_target_wTP = self.target_model.copy()
        self.mdl_target_wTP.instruments['IzTP'] = pygsti.modelmembers.instruments.TPInstrument({'plus': Gmz_plus, 'minus': Gmz_minus})

        super(InstrumentTestCase, self).setUp()

    def testFutureFunctionality(self):
        #Test instrument construction with elements whose gpindices are already initialized.
        # Since this isn't allowed currently (a future functionality), we need to do some hacking
        E = self.target_model.povms['Mdefault']['0']
        InstEl = pygsti.modelmembers.operations.FullArbitraryOp(np.dot(E, E.T))
        InstEl2 = InstEl.copy()
        nParams = InstEl.num_params # should be 16

        I = pygsti.modelmembers.instruments.Instrument({},
                                                       evotype='default',
                                                       state_space=pygsti.baseobjs.statespace.default_space_for_udim(2))
        InstEl.set_gpindices(slice(0,16), I)
        InstEl2.set_gpindices(slice(8,24), I) # some overlap - to test _build_paramvec

        # TESTING ONLY - so we can add items!!!
        I._readonly = False
        I['A'] = InstEl
        I['B'] = InstEl2
        I._readonly = True

        I._paramvec, I._paramlbls = I._build_paramvec()
          # this whole test was to exercise this function's ability to
          # form a parameter vector with weird overlapping gpindices.
        self.assertEqual( len(I._paramvec) , 24 )

    def testInstrumentMethods(self):

        v = self.mdl_target_wTP.to_vector()

        mdl = self.mdl_target_wTP.copy()
        mdl.from_vector(v)
        mdl.basis = self.mdl_target_wTP.basis.copy()

        self.assertAlmostEqual(mdl.frobeniusdist(self.mdl_target_wTP),0.0)

        for lbl in ('Iz','IzTP'):
            v = mdl.to_vector()
            gates = mdl.instruments[lbl].simplify_operations(prefix="ABC")
            for igate in gates.values():
                igate.from_vector(v[igate.gpindices]) # gpindices should be setup relative to Model's param vec

        mdl.depolarize(0.01)
        mdl.rotate((0,0,0.01))
        mdl.rotate(max_rotate=0.01, seed=1234)

    def testChangeDimension(self):
        larger_ss = pygsti.baseobjs.ExplicitStateSpace([('L%d' % i,) for i in range(6)])
        smaller_ss = pygsti.baseobjs.ExplicitStateSpace([('L%d' % i,) for i in range(3)])

        mdl = self.target_model.copy()
        new_gs = mdl.increase_dimension(larger_ss)
        new_gs = mdl._decrease_dimension(smaller_ss)

        #TP
        mdl = self.target_model.copy()
        mdl.set_all_parameterizations("TP")
        new_gs = mdl.increase_dimension(larger_ss)
        new_gs = mdl._decrease_dimension(smaller_ss)


    def testIntermediateMeas(self):
        # Mess with the target model to add some error to the povm and instrument
        self.assertEqual(self.target_model.num_params,92) # 4*3 + 16*5 = 92
        mdl = self.target_model.depolarize(op_noise=0.01, spam_noise=0.01)
        gs2 = self.target_model.depolarize(max_op_noise=0.01, max_spam_noise=0.01, seed=1234) #another way to depolarize
        mdl.povms['Mdefault'].depolarize(0.01)

        # Introducing a rotation error to the measurement
        Uerr = pygsti.rotation_gate_mx([0, 0.02, 0]) # input angles are halved by the method
        E = np.dot(mdl.povms['Mdefault']['0'].T,Uerr).T # effect is stored as column vector
        Erem = self.povm_ident - E
        mdl.povms['Mdefault'] = pygsti.modelmembers.povms.UnconstrainedPOVM({'0': E, '1': Erem}, evotype='default')

        # Now add the post-measurement gates from the vector E0 and remainder = id-E0
        Gmz_plus = np.dot(E,E.T) #since E0 is stored internally as column spamvec
        Gmz_minus = np.dot(Erem,Erem.T)
        mdl.instruments['Iz'] = pygsti.modelmembers.instruments.Instrument({'plus': Gmz_plus, 'minus': Gmz_minus})
        self.assertEqual(mdl.num_params,92) # 4*3 + 16*5 = 92
        #print(mdl)

        germs = std.germs
        fiducials = std.fiducials
        max_lengths = [1] #,2,4,8]
        glbls = list(mdl.operations.keys()) + list(mdl.instruments.keys())
        lsgst_struct = pygsti.circuits.create_lsgst_circuits(
            glbls,fiducials,fiducials,germs,max_lengths)
        lsgst_struct2 = pygsti.circuits.create_lsgst_circuits(
            mdl,fiducials,fiducials,germs,max_lengths) #use mdl as source
        self.assertEqual(list(lsgst_struct), list(lsgst_struct2))



        mdl_datagen = mdl
        ds = pygsti.data.simulate_data(mdl, lsgst_struct, 1000, 'none') #'multinomial')
        pygsti.io.write_dataset(temp_files + "/intermediate_meas_dataset.txt", ds)
        ds2 = pygsti.io.load_dataset(temp_files + "/intermediate_meas_dataset.txt")
        for opstr,dsRow in ds.items():
            for lbl,cnt in dsRow.counts.items():
                self.assertAlmostEqual(cnt, ds2[opstr].counts[lbl],places=2)
        #print(ds)

        #LGST
        mdl_lgst = pygsti.run_lgst(ds, fiducials, fiducials, self.target_model) #, guessModelForGauge=mdl_datagen)
        self.assertTrue("Iz" in mdl_lgst.instruments)
        mdl_opt = pygsti.gaugeopt_to_target(mdl_lgst, mdl_datagen) #, method="BFGS")
        print(mdl_datagen.strdiff(mdl_opt))
        print("Frobdiff = ",mdl_datagen.frobeniusdist( mdl_lgst))
        print("Frobdiff after GOpt = ",mdl_datagen.frobeniusdist(mdl_opt))
        self.assertAlmostEqual(mdl_datagen.frobeniusdist(mdl_opt), 0.0, places=4)
        #print(mdl_lgst)
        #print(mdl_datagen)

        #DEBUG compiling w/dataset
        #dbList = pygsti.circuits.create_lsgst_circuits(self.target_model,fiducials,fiducials,germs,max_lengths)
        ##self.target_model.simplify_circuits(dbList, ds)
        #self.target_model.simplify_circuits([ pygsti.circuits.Circuit(None,stringrep="Iz") ], ds )
        #assert(False),"STOP"

        #LSGST
        results = pygsti.run_long_sequence_gst(ds, self.target_model, fiducials, fiducials, germs, max_lengths)
        #print(results.estimates[results.name].models['go0'])
        mdl_est = results.estimates[results.name].models['go0']
        mdl_est_opt = pygsti.gaugeopt_to_target(mdl_est, mdl_datagen)
        print("Frobdiff = ", mdl_datagen.frobeniusdist(mdl_est))
        print("Frobdiff after GOpt = ", mdl_datagen.frobeniusdist(mdl_est_opt))
        self.assertAlmostEqual(mdl_datagen.frobeniusdist(mdl_est_opt), 0.0, places=4)

        #LGST w/TP gates
        mdl_targetTP = self.target_model.copy()
        mdl_targetTP.set_all_parameterizations("TP")
        self.assertEqual(mdl_targetTP.num_params,71) # 3 + 4*2 + 12*5 = 71
        #print(mdl_targetTP)
        resultsTP = pygsti.run_long_sequence_gst(ds, mdl_targetTP, fiducials, fiducials, germs, max_lengths)
        mdl_est = resultsTP.estimates[resultsTP.name].models['go0']
        mdl_est_opt = pygsti.gaugeopt_to_target(mdl_est, mdl_datagen)
        print("TP Frobdiff = ", mdl_datagen.frobeniusdist(mdl_est))
        print("TP Frobdiff after GOpt = ", mdl_datagen.frobeniusdist(mdl_est_opt))
        self.assertAlmostEqual(mdl_datagen.frobeniusdist(mdl_est_opt), 0.0, places=4)

    def testBasicGatesetOps(self):
        # This test was made from a debug script used to get the code working
        model = pygsti.models.modelconstruction.create_explicit_model_from_expressions(
            [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"])
        #    prep_labels=["rho0"], prep_expressions=["0"],
        #    effect_labels=["0","1"], effect_expressions=["0","complement"])

        v0 = modelconstruction._basis_create_spam_vector("0", pygsti.baseobjs.Basis.cast("pp", 4))
        v1 = modelconstruction._basis_create_spam_vector("1", pygsti.baseobjs.Basis.cast("pp", 4))
        P0 = np.dot(v0,v0.T)
        P1 = np.dot(v1,v1.T)
        print("v0 = ",v0)
        print("P0 = ",P0)
        print("P1 = ",P0)
        #print("P0+P1 = ",P0+P1)

        model.instruments["Itest"] = pygsti.modelmembers.instruments.Instrument([('0', P0), ('1', P1)])

        for param in ("full","TP","CPTP"):
            print(param)
            model.set_all_parameterizations(param)
            model.to_vector() # builds & cleans paramvec for tests below
            for lbl,obj in model.preps.items():
                print(lbl,':', obj.gpindices, pygsti.tools.length(obj.gpindices))
            for lbl,obj in model.povms.items():
                print(lbl,':', obj.gpindices, pygsti.tools.length(obj.gpindices))
                for sublbl,subobj in obj.items():
                    print("  > ", sublbl,':', subobj.gpindices, pygsti.tools.length(subobj.gpindices))
            for lbl,obj in model.operations.items():
                print(lbl,':', obj.gpindices, pygsti.tools.length(obj.gpindices))
            for lbl,obj in model.instruments.items():
                print(lbl,':', obj.gpindices, pygsti.tools.length(obj.gpindices))
                for sublbl,subobj in obj.items():
                    print("  > ", sublbl,':', subobj.gpindices, pygsti.tools.length(subobj.gpindices))


            print("NPARAMS = ",model.num_params)
            print("")


        print("PICKLING")

        x = model.preps #.copy(None)
        p = pickle.dumps(x) #model.preps)
        print("loading")
        preps = pickle.loads(p)
        self.assertEqual(list(preps.keys()),list(model.preps.keys()))

        #p = pickle.dumps(model.effects)
        #effects = pickle.loads(p)
        #assert(list(effects.keys()) == list(model.effects.keys()))

        p = pickle.dumps(model.operations)
        gates = pickle.loads(p)
        self.assertEqual(list(gates.keys()),list(model.operations.keys()))

        p = pickle.dumps(model)
        g = pickle.loads(p)
        self.assertAlmostEqual(model.frobeniusdist(g), 0.0)


        print("Model IO")
        pygsti.io.write_model(model, temp_files + "/testGateset.txt")
        model2 = pygsti.io.load_model(temp_files + "/testGateset.txt")
        self.assertAlmostEqual(model.frobeniusdist(model2),0.0)
        print("Multiplication")

        gatestring1 = ('Gx','Gy')
        gatestring2 = ('Gx','Gy','Gy')

        p1 = np.dot( model.operations['Gy'].to_dense(), model.operations['Gx'].to_dense())
        p2 = model.sim.product(gatestring1, scale=False)
        p3,scale = model.sim.product(gatestring1, scale=True)

        print(p1)
        print(p2)
        print(p3*scale)
        self.assertAlmostEqual(np.linalg.norm(p1-scale*p3),0.0)

        dp = model.sim.dproduct(gatestring1)
        dp_flat = model.sim.dproduct(gatestring1,flat=True)

        layout = model.sim.create_layout( [gatestring1,gatestring2] )

        p1 = np.dot( model.operations['Gy'].to_dense(), model.operations['Gx'].to_dense() )
        p2 = np.dot( model.operations['Gy'].to_dense(), np.dot( model.operations['Gy'].to_dense(), model.operations['Gx'].to_dense() ))

        bulk_prods = model.sim.bulk_product([gatestring1,gatestring2])
        bulk_prods_scaled, scaleVals = model.sim.bulk_product([gatestring1,gatestring2], scale=True)
        bulk_prods2 = scaleVals[:,None,None] * bulk_prods_scaled
        self.assertArraysAlmostEqual(bulk_prods[0],p1)
        self.assertArraysAlmostEqual(bulk_prods[1],p2)
        self.assertArraysAlmostEqual(bulk_prods2[0],p1)
        self.assertArraysAlmostEqual(bulk_prods2[1],p2)

        print("Probabilities")
        gatestring1 = ('Gx','Gy') #,'Itest')
        gatestring2 = ('Gx','Gy','Gy')

        layout = model.sim.create_layout( [gatestring1,gatestring2] )

        p1 = np.dot( np.transpose(model.povms['Mdefault']['0'].to_dense()),
                     np.dot( model.operations['Gy'].to_dense(),
                             np.dot(model.operations['Gx'].to_dense(),
                                    model.preps['rho0'].to_dense())))
        probs = model.probabilities(gatestring1)
        print(probs)
        p20,p21 = probs[('0',)],probs[('1',)]

        #probs = model.probabilities(gatestring1, use_scaling=True)
        #print(probs)
        #p30,p31 = probs['0'],probs['1']

        self.assertArraysAlmostEqual(p1,p20)
        #assertArraysAlmostEqual(p1,p30)
        #assertArraysAlmostEqual(p21,p31)

        bulk_probs = model.sim.bulk_probs([gatestring1,gatestring2])

        #Need to add way to force split a layout to check this:
        #evt_split = evt.copy()
        #new_lookup = evt_split.split(lookup, num_sub_trees=2)
        #print("SPLIT TREE: new el_indices = ",new_lookup)
        #probs_to_fill = np.empty( evt_split.num_final_elements(), 'd')
        #model.bulk_fill_probs(probs_to_fill,evt_split,check=True)

        #dProbs = model.sim.dprobs(gatestring1)  #Removed this functionality (unused)
        bulk_dProbs = model.sim.bulk_dprobs([gatestring1,gatestring2])

        #hProbs = model.sim.hprobs(gatestring1)  #Removed this functionality (unused)
        bulk_hProbs = model.sim.bulk_hprobs([gatestring1,gatestring2])

        print("DONE")

    def testAdvancedGateStrs(self):
        #specify prep and/or povm labels in operation sequence:
        circuit_normal = pygsti.circuits.Circuit(('Gx',))
        circuit_wprep = pygsti.circuits.Circuit(('rho0', 'Gx'))
        circuit_wpovm = pygsti.circuits.Circuit(('Gx', 'Mdefault'))
        circuit_wboth = pygsti.circuits.Circuit(('rho0', 'Gx', 'Mdefault'))

        #Now compute probabilities for these:
        model = self.target_model.copy()
        probs_normal = model.probabilities(circuit_normal)
        probs_wprep = model.probabilities(circuit_wprep)
        probs_wpovm = model.probabilities(circuit_wpovm)
        probs_wboth = model.probabilities(circuit_wboth)

        print(probs_normal)
        print(probs_wprep)
        print(probs_wpovm)
        print(probs_wboth)

        self.assertEqual( probs_normal, probs_wprep )
        self.assertEqual( probs_normal, probs_wpovm )
        self.assertEqual( probs_normal, probs_wboth )

        #now try bulk op
        bulk_probs = model.sim.bulk_probs([circuit_normal, circuit_wprep, circuit_wpovm, circuit_wboth])

    def testWriteAndLoad(self):
        mdl = self.target_model.copy()

        s = str(mdl) #stringify with instruments

        for param in ('full','TP','static'):  # skip 'CPTP' b/c cannot serialize that to text anymore
            print("Param: ",param)
            mdl.set_all_parameterizations(param)
            filename = temp_files + "/gateset_with_instruments_%s.txt" % param
            pygsti.io.write_model(mdl, filename)
            gs2 = pygsti.io.parse_model(filename)

            self.assertAlmostEqual( mdl.frobeniusdist(gs2), 0.0 )
            for lbl in mdl.operations:
                self.assertEqual( type(mdl.operations[lbl]), type(gs2.operations[lbl]))
            for lbl in mdl.preps:
                self.assertEqual( type(mdl.preps[lbl]), type(gs2.preps[lbl]))
            for lbl in mdl.povms:
                self.assertEqual( type(mdl.povms[lbl]), type(gs2.povms[lbl]))
            for lbl in mdl.instruments:
                self.assertEqual( type(mdl.instruments[lbl]), type(gs2.instruments[lbl]))
