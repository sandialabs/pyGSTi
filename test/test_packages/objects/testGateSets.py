import unittest
import itertools
import collections
import pygsti
import numpy as np
import warnings
import pickle
import os

from ..testutils import BaseTestCase, compare_files, temp_files
#from pygsti.objects.gatemapcalc import GateMapCalc

#Note: calcs expect tuples (or GateStrings) of *Labels*
from pygsti.objects import Label as L

from pygsti.construction import std1Q_XYI

def Ls(*args):
    """ Convert args to a tuple to Labels """
    return tuple([L(x) for x in args])

FD_JAC_PLACES = 5 # loose checking when computing finite difference derivatives (currently in map calcs)
FD_HESS_PLACES = 1 # looser checking when computing finite difference hessians (currently in map calcs)

# This class is for unifying some gatesets that get used in this file and in testGateSets2.py
class GateSetTestCase(BaseTestCase):

    def setUp(self):
        super(GateSetTestCase, self).setUp()

        #OK for these tests, since we test user interface?
        #Set GateSet objects to "strict" mode for testing
        pygsti.objects.GateSet._strict = False

        self.gateset = pygsti.construction.build_gateset(
            [2], [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"])
        
        self.tp_gateset = pygsti.construction.build_gateset(
            [2], [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
            parameterization="TP")

        self.static_gateset = pygsti.construction.build_gateset(
            [2], [('Q0',)],['Gi','Gx','Gy'],
            [ "I(Q0)","X(pi/8,Q0)", "Y(pi/8,Q0)"],
            parameterization="static")

        self.mgateset = self.gateset.copy()
        #self.mgateset._calcClass = GateMapCalc
        self.mgateset.set_simtype('map')


class TestGateSetMethods(GateSetTestCase):

    def test_creation(self):
        self.assertIsInstance(self.gateset, pygsti.objects.GateSet)

    def test_pickling(self):
        p = pickle.dumps(self.gateset.preps)
        preps = pickle.loads(p)
        self.assertEqual(list(preps.keys()), list(self.gateset.preps.keys()))

        p = pickle.dumps(self.gateset.povms)
        povms = pickle.loads(p)
        self.assertEqual(list(povms.keys()), list(self.gateset.povms.keys()))

        p = pickle.dumps(self.gateset.gates)
        gates = pickle.loads(p)
        self.assertEqual(list(gates.keys()), list(self.gateset.gates.keys()))

        p = pickle.dumps(self.gateset)
        g = pickle.loads(p)
        self.assertAlmostEqual(self.gateset.frobeniusdist(g), 0.0)

    def test_counting(self):

        self.assertEqual( len(self.gateset.preps), 1)
        self.assertEqual( len(self.gateset.povms['Mdefault']), 2)
        
        for default_param in ("full","TP","static"):
            print("Case: default_param = ",default_param)
            nGates = 3 if default_param in ("full","TP") else 0
            nSPVecs = 1 if default_param in ("full","TP") else 0
            if default_param == "full": nEVecs = 2
            elif default_param == "TP": nEVecs = 1 #complement doesn't add params
            else: nEVecs = 0
            nParamsPerGate = 16 if default_param == "full" else 12
            nParamsPerSP = 4 if default_param == "full" else 3
            nParams =  nGates * nParamsPerGate + nSPVecs * nParamsPerSP + nEVecs * 4
            self.gateset.set_all_parameterizations(default_param)

            for lbl,obj in self.gateset.preps.items():
                print(lbl,':',obj.gpindices)
            for lbl,obj in self.gateset.povms.items():
                print(lbl,':',obj.gpindices)
            for lbl,obj in self.gateset.gates.items():
                print(lbl,':',obj.gpindices)
            print("NPARAMS = ",self.gateset.num_params())

            self.assertEqual(self.gateset.num_params(), nParams)

        self.assertEqual(list(self.gateset.preps.keys()), ["rho0"])
        self.assertEqual(list(self.gateset.povms.keys()), ["Mdefault"])

    def test_getset_full(self):
        self.getset_helper(self.gateset)

    def test_getset_tp(self):
        self.getset_helper(self.tp_gateset)

    def test_getset_static(self):
        self.getset_helper(self.static_gateset)

    def getset_helper(self, gs):

        #Test default prep/effects
        default_prep = gs.prep
        self.assertArraysAlmostEqual(default_prep,gs.preps["rho0"])
        default_povm = gs.effects
        assert(set(default_povm.keys()) == set(['0','1']))
        
        v = np.array( [[1.0/np.sqrt(2)],[0],[0],[1.0/np.sqrt(2)]], 'd')

        #gs['identity'] = v
        #w = gs['identity']
        #self.assertArraysAlmostEqual(w,v)

        gs['rho1'] = v
        w = gs['rho1']
        self.assertArraysAlmostEqual(w,v)

        # Can't just assign to POVM...
        #gs['Mdefault']['2'] = v
        #w = gs['Mdefault']['2']
        #self.assertArraysAlmostEqual(w,v)

        Gi_matrix = np.identity(4, 'd')
        self.assertTrue( isinstance(gs['Gi'], pygsti.objects.Gate) )

        Gi_test_matrix = np.random.random( (4,4) )
        Gi_test_matrix[0,:] = [1,0,0,0] # so TP mode works
        Gi_test = pygsti.objects.FullyParameterizedGate( Gi_test_matrix  )
        print("POINT 1")
        try:
            gs["Gi"] = Gi_test_matrix #set gate matrix
        except ValueError:
            pass # can't always set via matrix - e.g. doesn't work for *static* case
        
        print("POINT 2")
        gs["Gi"] = Gi_test #set gate object
        self.assertArraysAlmostEqual( gs['Gi'], Gi_test_matrix )

        print("DEL")
        del gs.preps['rho1']

        with self.assertRaises(KeyError):
            gs.preps['foobar'] = [1.0/np.sqrt(2),0,0,0] #bad key prefix

        with self.assertRaises(KeyError):
            print("COPYING")
            gs2 = gs.copy()
            error = gs2.povms['foobar']

        Iz = pygsti.obj.Instrument( [('0', np.random.random( (4,4) ))] )
        gs["Iz"] = Iz #set an instrument
        Iz2 = gs["Iz"] #get an instrument

        deriv = gs.deriv_wrt_params()



    def test_copy(self):
        cp = self.gateset.copy()
        self.assertAlmostEqual( self.gateset.frobeniusdist(cp), 0 )
        self.assertAlmostEqual( self.gateset.jtracedist(cp), 0 )
        self.assertAlmostEqual( self.gateset.diamonddist(cp), 0 )


    def test_vectorize(self):
        cp = self.gateset.copy()
        v = cp.to_vector()
        cp.from_vector(v)
        self.assertAlmostEqual( self.gateset.frobeniusdist(cp), 0 )


    def test_transform(self):
        T = np.array([[ 0.36862036,  0.49241519,  0.35903944,  0.90069522],
                      [ 0.12347698,  0.45060548,  0.61671491,  0.64854769],
                      [ 0.4038386 ,  0.89518315,  0.20206879,  0.6484708 ],
                      [ 0.44878029,  0.42095514,  0.27645424,  0.41766033]]) #some random array
        Tinv = np.linalg.inv(T)
        elT = pygsti.objects.FullGaugeGroupElement(T)
        cp = self.gateset.copy()
        cp.set_all_parameterizations('full') # so POVM can be transformed...
        cp.transform(elT)

        self.assertAlmostEqual( self.gateset.frobeniusdist(cp, T, normalize=False), 0 ) #test out normalize=False
        self.assertAlmostEqual( self.gateset.jtracedist(cp, T), 0 )
        self.assertAlmostEqual( self.gateset.diamonddist(cp, T), 0 )

        for gateLabel in cp.gates:
            self.assertArraysAlmostEqual(cp[gateLabel], np.dot(Tinv, np.dot(self.gateset[gateLabel], T)))
        for prepLabel in cp.preps:
            self.assertArraysAlmostEqual(cp[prepLabel], np.dot(Tinv, self.gateset[prepLabel]))
        for povmLabel in cp.povms:
            for effectLabel,eVec in cp.povms[povmLabel].items():
                self.assertArraysAlmostEqual(eVec,  np.dot(np.transpose(T), self.gateset.povms[povmLabel][effectLabel]))


    def test_simple_multiplicationA(self):
        gatestring = ('Gx','Gy')
        p1 = np.dot( self.gateset['Gy'], self.gateset['Gx'] )
        p2 = self.gateset.product(gatestring, bScale=False)
        p3,scale = self.gateset.product(gatestring, bScale=True)
        self.assertArraysAlmostEqual(p1,p2)
        self.assertArraysAlmostEqual(p1,scale*p3)

        #Artificially reset the "smallness" threshold for scaling to be
        # sure to engate the scaling machinery
        PORIG = pygsti.objects.gatematrixcalc.PSMALL; pygsti.objects.gatematrixcalc.PSMALL = 10
        p4,scale = self.gateset.product(gatestring, bScale=True)
        pygsti.objects.gatematrixcalc.PSMALL = PORIG
        self.assertArraysAlmostEqual(p1,scale*p4)

        dp = self.gateset.dproduct(gatestring)
        dp_flat = self.gateset.dproduct(gatestring,flat=True)


    def test_simple_multiplicationB(self):
        gatestring = ('Gx','Gy','Gy')
        p1 = np.dot( self.gateset['Gy'], np.dot( self.gateset['Gy'], self.gateset['Gx'] ))
        p2 = self.gateset.product(gatestring, bScale=False)
        p3,scale = self.gateset.product(gatestring, bScale=True)
        self.assertArraysAlmostEqual(p1,p2)
        self.assertArraysAlmostEqual(p1,scale*p3)

        #Artificially reset the "smallness" threshold for scaling to be
        # sure to engate the scaling machinery
        PORIG = pygsti.objects.gatematrixcalc.PSMALL; pygsti.objects.gatematrixcalc.PSMALL = 10
        p4,scale = self.gateset.product(gatestring, bScale=True)
        pygsti.objects.gatematrixcalc.PSMALL = PORIG
        self.assertArraysAlmostEqual(p1,scale*p4)


    def test_bulk_multiplication(self):
        gatestring1 = ('Gx','Gy')
        gatestring2 = ('Gx','Gy','Gy')
        evt,lookup,outcome_lookup = self.gateset.bulk_evaltree( [gatestring1,gatestring2] )

        p1 = np.dot( self.gateset['Gy'], self.gateset['Gx'] )
        p2 = np.dot( self.gateset['Gy'], np.dot( self.gateset['Gy'], self.gateset['Gx'] ))

        bulk_prods = self.gateset.bulk_product(evt)
        bulk_prods_scaled, scaleVals = self.gateset.bulk_product(evt, bScale=True)
        bulk_prods2 = scaleVals[:,None,None] * bulk_prods_scaled
        self.assertArraysAlmostEqual(bulk_prods[ 0 ],p1)
        self.assertArraysAlmostEqual(bulk_prods[ 1 ],p2)
        self.assertArraysAlmostEqual(bulk_prods2[ 0 ],p1)
        self.assertArraysAlmostEqual(bulk_prods2[ 1 ],p2)

        #Artificially reset the "smallness" threshold for scaling to be
        # sure to engate the scaling machinery
        PORIG = pygsti.objects.gatematrixcalc.PSMALL; pygsti.objects.gatematrixcalc.PSMALL = 10
        bulk_prods_scaled, scaleVals3 = self.gateset.bulk_product(evt, bScale=True)
        bulk_prods3 = scaleVals3[:,None,None] * bulk_prods_scaled
        pygsti.objects.gatematrixcalc.PSMALL = PORIG
        self.assertArraysAlmostEqual(bulk_prods3[0],p1)
        self.assertArraysAlmostEqual(bulk_prods3[1],p2)


        #tag on a few extra EvalTree tests
        debug_stuff = evt.get_analysis_plot_infos()


    def test_simple_probabilityA(self):
        gatestring = ('Gx','Gy')
        p0a = np.dot( np.transpose(self.gateset.povms['Mdefault']['0']),
                     np.dot( self.gateset['Gy'],
                             np.dot(self.gateset['Gx'],
                                    self.gateset.preps['rho0'])))

        probs = self.gateset.probs(gatestring)
        p0b,p1b = probs[('0',)], probs[('1',)]
        self.assertArraysAlmostEqual(p0a,p0b)
        self.assertArraysAlmostEqual(1.0-p0a,p1b)
        
        dprobs = self.gateset.dprobs(gatestring)
        dprobs2 = self.gateset.dprobs(gatestring,returnPr=True)
        self.assertArraysAlmostEqual(dprobs[('0',)],dprobs2[('0',)][0])
        self.assertArraysAlmostEqual(dprobs[('1',)],dprobs2[('1',)][0])

        #Compare with map-based computation
        mprobs = self.mgateset.probs(gatestring)
        mp0b,mp1b = mprobs[('0',)], mprobs[('1',)]
        self.assertArraysAlmostEqual(p0b,mp0b)
        self.assertArraysAlmostEqual(p1b,mp1b)
        
        mdprobs = self.mgateset.dprobs(gatestring)
        mdprobs2 = self.mgateset.dprobs(gatestring,returnPr=True)
        self.assertArraysAlmostEqual(dprobs[('0',)],mdprobs[('0',)])
        self.assertArraysAlmostEqual(dprobs[('1',)],mdprobs[('1',)])
        self.assertArraysAlmostEqual(dprobs[('0',)],mdprobs2[('0',)][0])
        self.assertArraysAlmostEqual(dprobs[('1',)],mdprobs2[('1',)][0])


    def test_simple_probabilityB(self):
        gatestring = ('Gx','Gy','Gy')
        p1 = np.dot( np.transpose(self.gateset.povms['Mdefault']['0']),
                     np.dot( self.gateset['Gy'],
                             np.dot( self.gateset['Gy'],
                                     np.dot(self.gateset['Gx'],
                                            self.gateset.preps['rho0']))))
        p2 = self.gateset.probs(gatestring)[('0',)]
        self.assertSingleElemArrayAlmostEqual(p1, p2)
        
        gateset_with_nan = self.gateset.copy()
        gateset_with_nan['rho0'][:] = np.nan
        self.assertWarns(gateset_with_nan.probs,gatestring)
        self.assertWarns(gateset_with_nan.probs,gatestring*5) # long gatestring: warning uses elipsis

        mgateset_with_nan = self.mgateset.copy()
        mgateset_with_nan['rho0'][:] = np.nan
        self.assertWarns(mgateset_with_nan.probs,gatestring)
        self.assertWarns(mgateset_with_nan.probs,gatestring*5) # long gatestring: warning uses elipsis

        
    def test_bulk_probabilities(self):
        gatestring1 = pygsti.obj.GateString(('Gx','Gy'))
        gatestring2 = pygsti.obj.GateString(('Gx','Gy','Gy'))
        evt,lookup,outcome_lookup = self.gateset.bulk_evaltree( [gatestring1,gatestring2] )
        mevt,mlookup,moutcome_lookup = self.mgateset.bulk_evaltree( [gatestring1,gatestring2] )

        p1 = np.dot( np.transpose(self.gateset.povms['Mdefault']['0']),
                     np.dot( self.gateset['Gy'],
                             np.dot(self.gateset['Gx'],
                                    self.gateset.preps['rho0'])))

        p2 = np.dot( np.transpose(self.gateset.povms['Mdefault']['0']),
                     np.dot( self.gateset['Gy'],
                             np.dot( self.gateset['Gy'],
                                     np.dot(self.gateset['Gx'],
                                            self.gateset.preps['rho0']))))

        #bulk_pr removed
        ##check == true could raise a warning if a mismatch is detected
        #bulk_pr = self.assertNoWarnings(self.gateset.bulk_pr,'0',evt,check=True)
        #bulk_pr_m = self.assertNoWarnings(self.gateset.bulk_pr,'1',evt,check=True)
        #mbulk_pr = self.assertNoWarnings(self.mgateset.bulk_pr,'0',mevt,check=True)
        #mbulk_pr_m = self.assertNoWarnings(self.mgateset.bulk_pr,'1',mevt,check=True)
        #self.assertSingleElemArrayAlmostEqual(p1, bulk_pr[0])
        #self.assertSingleElemArrayAlmostEqual(p2, bulk_pr[1])
        #self.assertSingleElemArrayAlmostEqual(1.0 - p1, bulk_pr_m[0])
        #self.assertSingleElemArrayAlmostEqual(1.0 - p2, bulk_pr_m[1])
        #self.assertSingleElemArrayAlmostEqual(p1, mbulk_pr[0])
        #self.assertSingleElemArrayAlmostEqual(p2, mbulk_pr[1])
        #self.assertSingleElemArrayAlmostEqual(1.0 - p1, mbulk_pr_m[0])
        #self.assertSingleElemArrayAlmostEqual(1.0 - p2, mbulk_pr_m[1])

        #non-bulk probabilities (again?)
        probs1 = self.gateset.probs(gatestring1)
        probs2 = self.gateset.probs(gatestring2)
        mprobs1 = self.mgateset.probs(gatestring1)
        mprobs2 = self.mgateset.probs(gatestring2)
        self.assertSingleElemArrayAlmostEqual(p1, probs1[('0',)])
        self.assertSingleElemArrayAlmostEqual(p2, probs2[('0',)])
        self.assertSingleElemArrayAlmostEqual(1.0 - p1, probs1[('1',)])
        self.assertSingleElemArrayAlmostEqual(1.0 - p2, probs2[('1',)])
        self.assertSingleElemArrayAlmostEqual(p1, mprobs1[('0',)])
        self.assertSingleElemArrayAlmostEqual(p2, mprobs2[('0',)])
        self.assertSingleElemArrayAlmostEqual(1.0 - p1, mprobs1[('1',)])
        self.assertSingleElemArrayAlmostEqual(1.0 - p2, mprobs2[('1',)])

        #bulk_probs
        bulk_probs = self.assertNoWarnings(self.gateset.bulk_probs,[gatestring1,gatestring2],check=True)
        mbulk_probs = self.assertNoWarnings(self.mgateset.bulk_probs,[gatestring1,gatestring2],check=True)
        self.assertSingleElemArrayAlmostEqual(p1, bulk_probs[gatestring1][('0',)])
        self.assertSingleElemArrayAlmostEqual(p2, bulk_probs[gatestring2][('0',)])
        self.assertSingleElemArrayAlmostEqual(1.0 - p1, bulk_probs[gatestring1][('1',)])
        self.assertSingleElemArrayAlmostEqual(1.0 - p2, bulk_probs[gatestring2][('1',)])
        self.assertSingleElemArrayAlmostEqual(p1, mbulk_probs[gatestring1][('0',)])
        self.assertSingleElemArrayAlmostEqual(p2, mbulk_probs[gatestring2][('0',)])
        self.assertSingleElemArrayAlmostEqual(1.0 - p1, mbulk_probs[gatestring1][('1',)])
        self.assertSingleElemArrayAlmostEqual(1.0 - p2, mbulk_probs[gatestring2][('1',)])


        def elIndx(iGateStr, outcome):
            inds = pygsti.tools.indices(lookup[iGateStr]) if isinstance(lookup[iGateStr],slice) \
                   else lookup[iGateStr] #an index array
            return inds[ outcome_lookup[iGateStr].index( outcome ) ]
        def melIndx(iGateStr, outcome):
            inds = pygsti.tools.indices(mlookup[iGateStr]) if isinstance(mlookup[iGateStr],slice) \
                   else mlookup[iGateStr] #an index array
            return inds[ moutcome_lookup[iGateStr].index( outcome ) ]
        
        nElements = evt.num_final_elements()
        probs_to_fill = np.empty( nElements, 'd')
        mprobs_to_fill = np.empty( nElements, 'd')
        self.assertNoWarnings(self.gateset.bulk_fill_probs, probs_to_fill, evt, check=True)
        self.assertNoWarnings(self.mgateset.bulk_fill_probs, mprobs_to_fill, mevt, check=True)
        self.assertSingleElemArrayAlmostEqual(p1, probs_to_fill[ elIndx(0, ('0',)) ])
        self.assertSingleElemArrayAlmostEqual(p2, probs_to_fill[ elIndx(1, ('0',))])
        self.assertSingleElemArrayAlmostEqual(1-p1, probs_to_fill[ elIndx(0, ('1',))])
        self.assertSingleElemArrayAlmostEqual(1-p2, probs_to_fill[ elIndx(1, ('1',))])
        self.assertSingleElemArrayAlmostEqual(p1, mprobs_to_fill[ melIndx(0, ('0',))])
        self.assertSingleElemArrayAlmostEqual(p2, mprobs_to_fill[ melIndx(1, ('0',))])
        self.assertSingleElemArrayAlmostEqual(1-p1, mprobs_to_fill[ melIndx(0, ('1',))])
        self.assertSingleElemArrayAlmostEqual(1-p2, mprobs_to_fill[ melIndx(1, ('1',))])

        #test with split eval tree
        evt_split = evt.copy(); lookup_splt = evt_split.split(lookup,numSubTrees=2)
        mevt_split = mevt.copy(); mlookup_splt = mevt_split.split(mlookup,numSubTrees=2)
        probs_to_fill_splt = np.empty( nElements, 'd')
        mprobs_to_fill_splt = np.empty( nElements, 'd')

        bulk_probs_splt = self.assertNoWarnings(self.gateset.bulk_fill_probs,
                                                probs_to_fill_splt, evt_split, check=True)
        mbulk_probs_splt = self.assertNoWarnings(self.mgateset.bulk_fill_probs,
                                                 mprobs_to_fill_splt, mevt_split, check=True)

        evt_split.print_analysis()
        mevt_split.print_analysis()

        #Note: Outcome labels stay in same order across tree splits (i.e.
        #   evalTree.split() doesn't need to update outcome_lookup)
        for i,gstr in enumerate([gatestring1,gatestring2]): #original gate strings
            self.assertArraysAlmostEqual(probs_to_fill[ lookup[i] ],
                                         probs_to_fill_splt[ lookup_splt[i] ])
            self.assertArraysAlmostEqual(mprobs_to_fill[ mlookup[i] ],
                                         mprobs_to_fill_splt[ mlookup_splt[i] ])

            #Also check map vs matrix fills:
            assert(outcome_lookup[i] == moutcome_lookup[i]) # should stay in same ordering... I think
            self.assertArraysAlmostEqual(probs_to_fill[ lookup[i] ],
                                         mprobs_to_fill[ mlookup[i] ], places=FD_JAC_PLACES)

        prods = self.gateset.bulk_product(evt) #TODO: test output?


    def test_derivatives(self):
        gatestring0 = pygsti.obj.GateString(('Gi',)) #,'Gx'
        gatestring1 = pygsti.obj.GateString(('Gx','Gy'))
        gatestring2 = pygsti.obj.GateString(('Gx','Gy','Gy'))

        gatestringList = [gatestring0,gatestring1,gatestring2]
        evt,lookup,outcome_lookup = self.gateset.bulk_evaltree( gatestringList )
        mevt,mlookup,moutcome_lookup = self.mgateset.bulk_evaltree( gatestringList )
        
        dP0 = self.gateset.dprobs(gatestring0)[('0',)]
        dP1 = self.gateset.dprobs(gatestring1)[('0',)]
        dP2 = self.gateset.dprobs(gatestring2)[('0',)]
        dP0m = self.gateset.dprobs(gatestring0)[('1',)]
        dP1m = self.gateset.dprobs(gatestring1)[('1',)]
        dP2m = self.gateset.dprobs(gatestring2)[('1',)]

        #Removed bulk_dpr
        #bulk_dP = self.gateset.bulk_dpr('0', evt, returnPr=False, check=True)
        #bulk_dP_m = self.gateset.bulk_dpr('1', evt, returnPr=False, check=True)
        #bulk_dP_chk, bulk_P = self.gateset.bulk_dpr('0', evt, returnPr=True, check=False)
        #bulk_dP_m_chk, bulk_Pm = self.gateset.bulk_dpr('1', evt, returnPr=True, check=False)
        #
        #mbulk_dP = self.mgateset.bulk_dpr('0', mevt, returnPr=False, check=True)
        #mbulk_dP_m = self.mgateset.bulk_dpr('1', mevt, returnPr=False, check=True)
        #mbulk_dP_chk, mbulk_P = self.mgateset.bulk_dpr('0', mevt, returnPr=True, check=False)
        #mbulk_dP_m_chk, mbulk_Pm = self.mgateset.bulk_dpr('1', mevt, returnPr=True, check=False)
        #
        #self.assertArraysAlmostEqual(bulk_dP,bulk_dP_chk)
        #self.assertArraysAlmostEqual(bulk_dP[0,:],dP0)
        #self.assertArraysAlmostEqual(bulk_dP[1,:],dP1)
        #self.assertArraysAlmostEqual(bulk_dP[2,:],dP2)
        #self.assertArraysAlmostEqual(bulk_dP_m,bulk_dP_m_chk) 
        #self.assertArraysAlmostEqual(bulk_dP_m[0,:],dP0m) 
        #self.assertArraysAlmostEqual(bulk_dP_m[1,:],dP1m)
        #self.assertArraysAlmostEqual(bulk_dP_m[2,:],dP2m)
        #
        #self.assertArraysAlmostEqual(mbulk_dP,mbulk_dP_chk, places=FD_JAC_PLACES) #relax tolerance for 
        #self.assertArraysAlmostEqual(mbulk_dP[0,:],dP0, places=FD_JAC_PLACES)     # finite diff derivs...
        #self.assertArraysAlmostEqual(mbulk_dP[1,:],dP1, places=FD_JAC_PLACES)
        #self.assertArraysAlmostEqual(mbulk_dP[2,:],dP2, places=FD_JAC_PLACES)
        #self.assertArraysAlmostEqual(mbulk_dP_m,mbulk_dP_m_chk, places=FD_JAC_PLACES)
        #self.assertArraysAlmostEqual(mbulk_dP_m[0,:],dP0m, places=FD_JAC_PLACES)
        #self.assertArraysAlmostEqual(mbulk_dP_m[1,:],dP1m, places=FD_JAC_PLACES)
        #self.assertArraysAlmostEqual(mbulk_dP_m[2,:],dP2m, places=FD_JAC_PLACES)


        dProbs0 = self.gateset.dprobs(gatestring0)
        dProbs1 = self.gateset.dprobs(gatestring1)
        dProbs2 = self.gateset.dprobs(gatestring2)

        mdProbs0 = self.mgateset.dprobs(gatestring0)
        mdProbs1 = self.mgateset.dprobs(gatestring1)
        mdProbs2 = self.mgateset.dprobs(gatestring2)

        dProbs0b = self.gateset.dprobs(gatestring0, returnPr=True)
        mdProbs0b = self.mgateset.dprobs(gatestring0, returnPr=True)


        self.assertArraysAlmostEqual(dProbs0[('0',)], dP0)
        self.assertArraysAlmostEqual(dProbs1[('0',)], dP1)
        self.assertArraysAlmostEqual(dProbs2[('0',)], dP2)
        self.assertArraysAlmostEqual(mdProbs0[('0',)], dP0, places=FD_JAC_PLACES)
        self.assertArraysAlmostEqual(mdProbs1[('0',)], dP1, places=FD_JAC_PLACES)
        self.assertArraysAlmostEqual(mdProbs2[('0',)], dP2, places=FD_JAC_PLACES)


        bulk_dProbs = self.assertNoWarnings(self.gateset.bulk_dprobs,
                                            gatestringList, returnPr=False, check=True)
        bulk_dProbs_chk = self.assertNoWarnings(self.gateset.bulk_dprobs,
                                                gatestringList, returnPr=True, check=True)
        mbulk_dProbs = self.assertNoWarnings(self.mgateset.bulk_dprobs,
                                             gatestringList, returnPr=False, check=True)
        mbulk_dProbs_chk = self.assertNoWarnings(self.mgateset.bulk_dprobs,
                                                 gatestringList, returnPr=True, check=True)

        for gstr in gatestringList:
            for outLbl in bulk_dProbs[gstr]:
                self.assertArraysAlmostEqual(bulk_dProbs[gstr][outLbl],
                                             bulk_dProbs_chk[gstr][outLbl][0]) #[0] b/c _chk also contains probs
                self.assertArraysAlmostEqual(mbulk_dProbs[gstr][outLbl],
                                             mbulk_dProbs_chk[gstr][outLbl][0]) #[0] b/c _chk also contains probs
                self.assertArraysAlmostEqual(bulk_dProbs[gstr][outLbl],
                                             mbulk_dProbs[gstr][outLbl], places=FD_JAC_PLACES) # map vs. matrix

                
        self.assertArraysAlmostEqual(bulk_dProbs[gatestring0][('0',)],dP0)
        self.assertArraysAlmostEqual(bulk_dProbs[gatestring1][('0',)],dP1)
        self.assertArraysAlmostEqual(bulk_dProbs[gatestring2][('0',)],dP2)

        self.assertArraysAlmostEqual(mbulk_dProbs[gatestring0][('0',)],mdProbs0[('0',)])
        self.assertArraysAlmostEqual(mbulk_dProbs[gatestring1][('0',)],mdProbs1[('0',)])
        self.assertArraysAlmostEqual(mbulk_dProbs[gatestring2][('0',)],mdProbs2[('0',)])



        
        nElements = evt.num_final_elements(); nParams = self.gateset.num_params()
        probs_to_fill = np.empty( nElements, 'd')
        dprobs_to_fill = np.empty( (nElements,nParams), 'd')
        dprobs_to_fillB = np.empty( (nElements,nParams), 'd')
        mprobs_to_fill = np.empty( nElements, 'd')
        mdprobs_to_fill = np.empty( (nElements,nParams), 'd')
        mdprobs_to_fillB = np.empty( (nElements,nParams), 'd')
        spam_label_rows = { '0': 0, '1': 1 }
        self.assertNoWarnings(self.gateset.bulk_fill_dprobs, dprobs_to_fill, evt,
                              prMxToFill=probs_to_fill,check=True)
        self.assertNoWarnings(self.mgateset.bulk_fill_dprobs, mdprobs_to_fill, mevt,
                              prMxToFill=mprobs_to_fill,check=True)

        def elIndx(iGateStr, outcome):
            inds = pygsti.tools.indices(lookup[iGateStr]) if isinstance(lookup[iGateStr],slice) \
                   else lookup[iGateStr] #an index array
            return inds[ outcome_lookup[iGateStr].index( outcome ) ]
        def melIndx(iGateStr, outcome):
            inds = pygsti.tools.indices(mlookup[iGateStr]) if isinstance(mlookup[iGateStr],slice) \
                   else mlookup[iGateStr] #an index array
            return inds[ moutcome_lookup[iGateStr].index( outcome ) ]

        self.assertArraysAlmostEqual(dprobs_to_fill[elIndx(0,('0',)),:],dP0)
        self.assertArraysAlmostEqual(dprobs_to_fill[elIndx(1,('0',)),:],dP1)
        self.assertArraysAlmostEqual(dprobs_to_fill[elIndx(2,('0',)),:],dP2)
        self.assertArraysAlmostEqual(mdprobs_to_fill[melIndx(0,('0',)),:],dP0, places=FD_JAC_PLACES)
        self.assertArraysAlmostEqual(mdprobs_to_fill[melIndx(1,('0',)),:],dP1, places=FD_JAC_PLACES)
        self.assertArraysAlmostEqual(mdprobs_to_fill[melIndx(2,('0',)),:],dP2, places=FD_JAC_PLACES)


        #without probs
        self.assertNoWarnings(self.gateset.bulk_fill_dprobs, dprobs_to_fillB, evt, check=True)
        self.assertNoWarnings(self.mgateset.bulk_fill_dprobs, mdprobs_to_fillB, mevt, check=True)
        self.assertArraysAlmostEqual(dprobs_to_fill,dprobs_to_fillB)
        self.assertArraysAlmostEqual(mdprobs_to_fill,mdprobs_to_fillB, places=FD_JAC_PLACES)


        #Artificially reset the "smallness" threshold for scaling
        # to be sure to engate the scaling machinery
        PORIG = pygsti.objects.gatematrixcalc.PSMALL; pygsti.objects.gatematrixcalc.PSMALL = 10
        DORIG = pygsti.objects.gatematrixcalc.DSMALL; pygsti.objects.gatematrixcalc.DSMALL = 10
        self.gateset.bulk_fill_dprobs(dprobs_to_fillB, evt, check=True)
        self.assertArraysAlmostEqual(dprobs_to_fill,dprobs_to_fillB)
        pygsti.objects.gatematrixcalc.PSMALL = PORIG
        self.gateset.bulk_fill_dprobs(dprobs_to_fillB, evt, check=True)
        self.assertArraysAlmostEqual(dprobs_to_fill,dprobs_to_fillB)
        pygsti.objects.gatematrixcalc.DSMALL = DORIG


        #test with split eval tree
        evt_split = evt.copy(); lookup_splt = evt_split.split(lookup,numSubTrees=2)
        mevt_split = mevt.copy(); mlookup_splt = mevt_split.split(mlookup,numSubTrees=2)
        dprobs_to_fill_splt = np.empty( (nElements,nParams), 'd')
        mdprobs_to_fill_splt = np.empty( (nElements,nParams), 'd')
        self.assertNoWarnings(self.gateset.bulk_fill_dprobs, dprobs_to_fill_splt, evt_split,
                              prMxToFill=None,check=True)
        self.assertNoWarnings(self.mgateset.bulk_fill_dprobs, mdprobs_to_fill_splt, mevt_split,
                              prMxToFill=None,check=True)

        
        #Note: Outcome labels stay in same order across tree splits (i.e.
        #   evalTree.split() doesn't need to update outcome_lookup)
        for i,gstr in enumerate(gatestringList): #original gate strings
            print("Gatestring %d: comparing " % i,lookup[i]," to ", lookup_splt[i])
            self.assertArraysAlmostEqual(dprobs_to_fill[ lookup[i] ],
                                         dprobs_to_fill_splt[ lookup_splt[i] ])
            print("Gatestring %d: comparing " % i,mlookup[i]," to ", mlookup_splt[i])
            self.assertArraysAlmostEqual(mdprobs_to_fill[ mlookup[i] ],
                                         mdprobs_to_fill_splt[ mlookup_splt[i] ])

            #Also check map vs matrix fills:
            assert(outcome_lookup[i] == moutcome_lookup[i]) # should stay in same ordering... I think
            self.assertArraysAlmostEqual(dprobs_to_fill[ lookup[i] ],
                                         mdprobs_to_fill[ mlookup[i] ], places=FD_JAC_PLACES)


        dProds = self.gateset.bulk_dproduct(evt) #TODO: test output?
        with self.assertRaises(NotImplementedError):
            self.mgateset.bulk_dproduct(mevt) # map-based computation doesn't compute "products"



    def test_hessians(self):
        gatestring0 = pygsti.obj.GateString(('Gi','Gx'))
        gatestring1 = pygsti.obj.GateString(('Gx','Gy'))
        gatestring2 = pygsti.obj.GateString(('Gx','Gy','Gy'))

        gatestringList = pygsti.construction.gatestring_list([gatestring0,gatestring1,gatestring2])
        evt,lookup,outcome_lookup = self.gateset.bulk_evaltree( [gatestring0,gatestring1,gatestring2] )
        mevt,mlookup,moutcome_lookup = self.mgateset.bulk_evaltree( [gatestring0,gatestring1,gatestring2] )

        hP0 = self.gateset.hprobs(gatestring0)[('0',)]
        hP1 = self.gateset.hprobs(gatestring1)[('0',)]
        hP2 = self.gateset.hprobs(gatestring2)[('0',)]
        hP0m = self.gateset.hprobs(gatestring0)[('1',)]
        hP1m = self.gateset.hprobs(gatestring1)[('1',)]
        hP2m = self.gateset.hprobs(gatestring2)[('1',)]

        hP0b,P0 = self.gateset.hprobs(gatestring0, returnPr=True)[('0',)]
        hP0b,dP0 = self.gateset.hprobs(gatestring0, returnDeriv=True)[('0',)]
        hP0mb,P0m = self.gateset.hprobs(gatestring0, returnPr=True)[('1',)]
        hP0mb,dP0m = self.gateset.hprobs(gatestring0, returnDeriv=True)[('1',)]

        #Removed bulk_hpr
        #bulk_hP = self.gateset.bulk_hpr('0', evt, returnPr=False, returnDeriv=False, check=True)
        #bulk_hP_m = self.gateset.bulk_hpr('1', evt, returnPr=False, returnDeriv=False, check=True)
        #bulk_hP_chk, bulk_dP, bulk_P = self.gateset.bulk_hpr('0', evt, returnPr=True, returnDeriv=True, check=False)
        #bulk_hP_m_chk, bulk_dP_m, bulk_P_m = self.gateset.bulk_hpr('1', evt, returnPr=True, returnDeriv=True, check=False)
        #
        #mbulk_hP = self.mgateset.bulk_hpr('0', mevt, returnPr=False, returnDeriv=False, check=True)
        #mbulk_hP_m = self.mgateset.bulk_hpr('1', mevt, returnPr=False, returnDeriv=False, check=True)
        #mbulk_hP_chk, mbulk_dP, mbulk_P = self.mgateset.bulk_hpr('0', mevt, returnPr=True, returnDeriv=True, check=False)
        #mbulk_hP_m_chk, mbulk_dP_m, mbulk_P_m = self.mgateset.bulk_hpr('1', mevt, returnPr=True, returnDeriv=True, check=False)
        #
        #self.assertArraysAlmostEqual(bulk_hP,bulk_hP_chk)
        #self.assertArraysAlmostEqual(bulk_hP[0,:,:],hP0)
        #self.assertArraysAlmostEqual(bulk_hP[1,:,:],hP1)
        #self.assertArraysAlmostEqual(bulk_hP[2,:,:],hP2)
        #self.assertArraysAlmostEqual(bulk_hP_m,bulk_hP_m_chk)
        #self.assertArraysAlmostEqual(bulk_hP_m[0,:,:],hP0m)
        #self.assertArraysAlmostEqual(bulk_hP_m[1,:,:],hP1m)
        #self.assertArraysAlmostEqual(bulk_hP_m[2,:,:],hP2m)
        #
        #self.assertArraysAlmostEqual(mbulk_hP,mbulk_hP_chk, places=FD_HESS_PLACES)
        #print("DB: hP0 = ",hP0)
        #print("DB: mhP0 = ",mbulk_hP[0,:,:])
        #self.assertArraysAlmostEqual(mbulk_hP[0,:,:],hP0, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mbulk_hP[1,:,:],hP1, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mbulk_hP[2,:,:],hP2, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mbulk_hP_m,mbulk_hP_m_chk, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mbulk_hP_m[0,:,:],hP0m, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mbulk_hP_m[1,:,:],hP1m, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mbulk_hP_m[2,:,:],hP2m, places=FD_HESS_PLACES)
        

        hProbs0 = self.gateset.hprobs(gatestring0)
        hProbs1 = self.gateset.hprobs(gatestring1)
        hProbs2 = self.gateset.hprobs(gatestring2)
        mhProbs0 = self.mgateset.hprobs(gatestring0)
        mhProbs1 = self.mgateset.hprobs(gatestring1)
        mhProbs2 = self.mgateset.hprobs(gatestring2)

        self.assertArraysAlmostEqual(hProbs0[('0',)], hP0)
        self.assertArraysAlmostEqual(hProbs1[('0',)], hP1)
        self.assertArraysAlmostEqual(hProbs2[('0',)], hP2)
        self.assertArraysAlmostEqual(mhProbs0[('0',)], hP0, places=FD_HESS_PLACES)
        self.assertArraysAlmostEqual(mhProbs1[('0',)], hP1, places=FD_HESS_PLACES)
        self.assertArraysAlmostEqual(mhProbs2[('0',)], hP2, places=FD_HESS_PLACES)


        bulk_hProbs = self.assertNoWarnings(self.gateset.bulk_hprobs,
                                            gatestringList, returnPr=False, check=True)
        bulk_hProbs_chk = self.assertNoWarnings(self.gateset.bulk_hprobs,
                                                gatestringList, returnPr=True, check=True)
        mbulk_hProbs = self.assertNoWarnings(self.mgateset.bulk_hprobs,
                                            gatestringList, returnPr=False, check=True)
        mbulk_hProbs_chk = self.assertNoWarnings(self.mgateset.bulk_hprobs,
                                                gatestringList, returnPr=True, check=True)

        for gstr in gatestringList:
            for outLbl in bulk_hProbs[gstr]:
                self.assertArraysAlmostEqual(bulk_hProbs[gstr][outLbl],
                                             bulk_hProbs_chk[gstr][outLbl][0]) #[0] b/c _chk also contains probs
                self.assertArraysAlmostEqual(mbulk_hProbs[gstr][outLbl],
                                             mbulk_hProbs_chk[gstr][outLbl][0]) #[0] b/c _chk also contains probs
                self.assertArraysAlmostEqual(bulk_hProbs[gstr][outLbl],
                                             mbulk_hProbs[gstr][outLbl], places=FD_HESS_PLACES) # map vs. matrix

        self.assertArraysAlmostEqual(bulk_hProbs[gatestring0][('0',)],hP0)
        self.assertArraysAlmostEqual(bulk_hProbs[gatestring1][('0',)],hP1)
        self.assertArraysAlmostEqual(bulk_hProbs[gatestring2][('0',)],hP2)

        self.assertArraysAlmostEqual(mbulk_hProbs[gatestring0][('0',)],mhProbs0[('0',)], places=FD_HESS_PLACES)
        self.assertArraysAlmostEqual(mbulk_hProbs[gatestring1][('0',)],mhProbs1[('0',)], places=FD_HESS_PLACES)
        self.assertArraysAlmostEqual(mbulk_hProbs[gatestring2][('0',)],mhProbs2[('0',)], places=FD_HESS_PLACES)

        #Vary keyword args
        hProbs0b = self.gateset.hprobs(gatestring0,returnPr=True)
        hProbs0c = self.gateset.hprobs(gatestring0,returnDeriv=True)
        hProbs0d = self.gateset.hprobs(gatestring0,returnDeriv=True,returnPr=True)
        bulk_hProbs_B = self.gateset.bulk_hprobs(gatestringList, returnPr=True, returnDeriv=True)
        bulk_hProbs_C = self.gateset.bulk_hprobs(gatestringList, returnDeriv=True)

        mhProbs0b = self.mgateset.hprobs(gatestring0,returnPr=True)
        mhProbs0c = self.mgateset.hprobs(gatestring0,returnDeriv=True)
        mhProbs0d = self.mgateset.hprobs(gatestring0,returnDeriv=True,returnPr=True)
        mbulk_hProbs_B = self.mgateset.bulk_hprobs(gatestringList, returnPr=True, returnDeriv=True)
        mbulk_hProbs_C = self.mgateset.bulk_hprobs(gatestringList, returnDeriv=True)

        
        nElements = evt.num_final_elements(); nParams = self.gateset.num_params()
        probs_to_fill = np.empty( nElements, 'd')
        probs_to_fillB = np.empty( nElements, 'd')
        dprobs_to_fill = np.empty( (nElements,nParams), 'd')
        dprobs_to_fillB = np.empty( (nElements,nParams), 'd')
        hprobs_to_fill = np.empty( (nElements,nParams,nParams), 'd')
        hprobs_to_fillB = np.empty( (nElements,nParams,nParams), 'd')
        mprobs_to_fill = np.empty( nElements, 'd')
        mprobs_to_fillB = np.empty( nElements, 'd')
        mdprobs_to_fill = np.empty( (nElements,nParams), 'd')
        mdprobs_to_fillB = np.empty( (nElements,nParams), 'd')
        mhprobs_to_fill = np.empty( (nElements,nParams,nParams), 'd')
        mhprobs_to_fillB = np.empty( (nElements,nParams,nParams), 'd')
        spam_label_rows = { '0': 0, '1': 1 }
        self.assertNoWarnings(self.gateset.bulk_fill_hprobs, hprobs_to_fill, evt,
                              prMxToFill=probs_to_fill, derivMxToFill=dprobs_to_fill, check=True)
        self.assertNoWarnings(self.mgateset.bulk_fill_hprobs, mhprobs_to_fill, mevt,
                              prMxToFill=mprobs_to_fill, derivMxToFill=mdprobs_to_fill, check=True)

        def elIndx(iGateStr, outcome):
            inds = pygsti.tools.indices(lookup[iGateStr]) if isinstance(lookup[iGateStr],slice) \
                   else lookup[iGateStr] #an index array
            return inds[ outcome_lookup[iGateStr].index( outcome ) ]
        def melIndx(iGateStr, outcome):
            inds = pygsti.tools.indices(mlookup[iGateStr]) if isinstance(mlookup[iGateStr],slice) \
                   else mlookup[iGateStr] #an index array
            return inds[ moutcome_lookup[iGateStr].index( outcome ) ]
                
        self.assertArraysAlmostEqual(hprobs_to_fill[elIndx(0,('0',)),:,:],hP0)
        self.assertArraysAlmostEqual(hprobs_to_fill[elIndx(1,('0',)),:,:],hP1)
        self.assertArraysAlmostEqual(hprobs_to_fill[elIndx(2,('0',)),:,:],hP2)
        self.assertArraysAlmostEqual(mhprobs_to_fill[melIndx(0,('0',)),:,:],hP0, places=FD_HESS_PLACES)
        self.assertArraysAlmostEqual(mhprobs_to_fill[melIndx(1,('0',)),:,:],hP1, places=FD_HESS_PLACES)
        self.assertArraysAlmostEqual(mhprobs_to_fill[melIndx(2,('0',)),:,:],hP2, places=FD_HESS_PLACES)

        #without derivative
        self.assertNoWarnings(self.gateset.bulk_fill_hprobs, hprobs_to_fillB, evt,
                              prMxToFill=probs_to_fillB, check=True)
        self.assertNoWarnings(self.mgateset.bulk_fill_hprobs, mhprobs_to_fillB, mevt,
                              prMxToFill=mprobs_to_fillB, check=True)

        self.assertArraysAlmostEqual(hprobs_to_fill,hprobs_to_fillB)
        self.assertArraysAlmostEqual(probs_to_fill,probs_to_fillB)
        self.assertArraysAlmostEqual(mhprobs_to_fill,mhprobs_to_fillB, places=FD_HESS_PLACES)
        self.assertArraysAlmostEqual(mprobs_to_fill,mprobs_to_fillB, places=FD_HESS_PLACES)


        #without probs
        self.assertNoWarnings(self.gateset.bulk_fill_hprobs, hprobs_to_fillB, evt,
                              derivMxToFill=dprobs_to_fillB, check=True)
        self.assertNoWarnings(self.mgateset.bulk_fill_hprobs, mhprobs_to_fillB, mevt,
                              derivMxToFill=mdprobs_to_fillB, check=True)
                
        self.assertArraysAlmostEqual(hprobs_to_fill,hprobs_to_fillB)
        self.assertArraysAlmostEqual(dprobs_to_fill,dprobs_to_fillB)
        self.assertArraysAlmostEqual(mhprobs_to_fill,mhprobs_to_fillB, places=FD_HESS_PLACES)
        self.assertArraysAlmostEqual(mdprobs_to_fill,mdprobs_to_fillB, places=FD_HESS_PLACES)

        #without either
        self.assertNoWarnings(self.gateset.bulk_fill_hprobs, hprobs_to_fillB, evt, check=True)
        self.assertNoWarnings(self.mgateset.bulk_fill_hprobs, mhprobs_to_fillB, mevt, check=True)
        self.assertArraysAlmostEqual(hprobs_to_fill,hprobs_to_fillB)
        self.assertArraysAlmostEqual(mhprobs_to_fill,mhprobs_to_fillB, places=FD_HESS_PLACES)


        #Artificially reset the "smallness" threshold for scaling
        # to be sure to engate the scaling machinery
        PORIG = pygsti.objects.gatematrixcalc.PSMALL; pygsti.objects.gatematrixcalc.PSMALL = 10
        DORIG = pygsti.objects.gatematrixcalc.DSMALL; pygsti.objects.gatematrixcalc.DSMALL = 10
        HORIG = pygsti.objects.gatematrixcalc.HSMALL; pygsti.objects.gatematrixcalc.HSMALL = 10
        self.gateset.bulk_fill_hprobs(hprobs_to_fillB, evt, check=True)
        self.assertArraysAlmostEqual(hprobs_to_fill,hprobs_to_fillB)
        pygsti.objects.gatematrixcalc.PSMALL = PORIG
        self.gateset.bulk_fill_hprobs(hprobs_to_fillB, evt, check=True)
        self.assertArraysAlmostEqual(hprobs_to_fill,hprobs_to_fillB)
        pygsti.objects.gatematrixcalc.DSMALL = DORIG
        pygsti.objects.gatematrixcalc.HSMALL = HORIG


        #test with split eval tree
        evt_split = evt.copy(); lookup_splt = evt_split.split(lookup,maxSubTreeSize=4)
        mevt_split = mevt.copy(); mlookup_splt = mevt_split.split(mlookup,numSubTrees=2)
        hprobs_to_fill_splt = np.empty( (nElements,nParams,nParams), 'd')
        mhprobs_to_fill_splt = np.empty( (nElements,nParams,nParams), 'd')
        self.assertNoWarnings(self.gateset.bulk_fill_hprobs, hprobs_to_fill_splt, evt_split, check=True)
        self.assertNoWarnings(self.mgateset.bulk_fill_hprobs, mhprobs_to_fill_splt, mevt_split, check=True)

        #Note: Outcome labels stay in same order across tree splits (i.e.
        #   evalTree.split() doesn't need to update outcome_lookup)
        for i,gstr in enumerate(gatestringList): #original gate strings
            self.assertArraysAlmostEqual(hprobs_to_fill[ lookup[i] ],
                                         hprobs_to_fill_splt[ lookup_splt[i] ])
            self.assertArraysAlmostEqual(mhprobs_to_fill[ mlookup[i] ],
                                         mhprobs_to_fill_splt[ mlookup_splt[i] ], places=FD_HESS_PLACES)

            #Also check map vs matrix fills:
            assert(outcome_lookup[i] == moutcome_lookup[i]) # should stay in same ordering... I think
            self.assertArraysAlmostEqual(hprobs_to_fill[ lookup[i] ],
                                         mhprobs_to_fill[ mlookup[i] ], places=FD_HESS_PLACES)

        
        #products
        N = self.gateset.get_dimension()**2 #number of elements in a gate matrix

        hProds = self.gateset.bulk_hproduct(evt)
        hProdsB,scales = self.gateset.bulk_hproduct(evt, bScale=True)
        
        self.assertArraysAlmostEqual(hProds, scales[:,None,None,None,None]*hProdsB)

        hProdsFlat = self.gateset.bulk_hproduct(evt, flat=True, bScale=False)
        hProdsFlatB,S1 = self.gateset.bulk_hproduct(evt, flat=True, bScale=True)

        self.assertArraysAlmostEqual(hProdsFlat, np.repeat(S1,N)[:,None,None]*hProdsFlatB)

        hProdsC, dProdsC, prodsC = self.gateset.bulk_hproduct(evt, bReturnDProdsAndProds=True, bScale=False)
        hProdsD, dProdsD, prodsD, S2 = self.gateset.bulk_hproduct(evt, bReturnDProdsAndProds=True, bScale=True)

        self.assertArraysAlmostEqual(hProds, hProdsC)
        self.assertArraysAlmostEqual(hProds, S2[:,None,None,None,None]*hProdsD)
        self.assertArraysAlmostEqual(dProdsC, S2[:,None,None,None]*dProdsD)
        self.assertArraysAlmostEqual(prodsC, S2[:,None,None]*prodsD)

        hProdsF, dProdsF, prodsF    = self.gateset.bulk_hproduct(evt, bReturnDProdsAndProds=True, flat=True, bScale=False)
        hProdsF2, dProdsF2, prodsF2, S3 = self.gateset.bulk_hproduct(evt, bReturnDProdsAndProds=True, flat=True, bScale=True)
        
        self.assertArraysAlmostEqual(hProdsFlat, hProdsF)
        self.assertArraysAlmostEqual(hProdsFlat, np.repeat(S3,N)[:,None,None]*hProdsF2)
        self.assertArraysAlmostEqual(dProdsF, np.repeat(S3,N)[:,None]*dProdsF2)
        self.assertArraysAlmostEqual(prodsF, S3[:,None,None]*prodsF2)

        
        nP = self.gateset.num_params()

        hcols = []
        d12cols = []
        slicesList = [ (slice(0,nP),slice(i,i+1)) for i in range(nP) ]
        for s1,s2, hprobs_col, dprobs12_col in self.gateset.bulk_hprobs_by_block(
            evt, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate( hcols, axis=2 )  #axes = (spam+gatestring, derivParam1, derivParam2)
        all_d12cols = np.concatenate( d12cols, axis=2 )
        dprobs12 = dprobs_to_fill[:,:,None] * dprobs_to_fill[:,None,:]

        #NOTE: Currently bulk_hprobs_by_block isn't implemented in map calculator - but it could
        # (and probably should) be later on, at which point the commented code here and
        # below would test it.
        
        #mhcols = []
        #md12cols = []
        #mslicesList = [ (slice(0,nP),slice(i,i+1)) for i in range(nP) ]
        #for s1,s2, hprobs_col, dprobs12_col in self.mgateset.bulk_hprobs_by_block(
        #    mevt, mslicesList, True):
        #    mhcols.append(hprobs_col)
        #    md12cols.append(dprobs12_col)
        #mall_hcols = np.concatenate( mhcols, axis=2 )  #axes = (spam+gatestring, derivParam1, derivParam2)
        #mall_d12cols = np.concatenate( md12cols, axis=2 )
        #mdprobs12 = mdprobs_to_fill[:,:,None] * mdprobs_to_fill[:,None,:]

        self.assertArraysAlmostEqual(all_hcols,hprobs_to_fill)
        self.assertArraysAlmostEqual(all_d12cols,dprobs12)
        #self.assertArraysAlmostEqual(mall_hcols,mhprobs_to_fill, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,mdprobs12, places=FD_HESS_PLACES)
        #
        #self.assertArraysAlmostEqual(mall_hcols,all_hcols, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,all_d12cols, places=FD_HESS_PLACES)

        
        hcols = []
        d12cols = []
        slicesList = [ (slice(0,nP),slice(i,i+1)) for i in range(1,10) ]
        for s1,s2, hprobs_col, dprobs12_col in self.gateset.bulk_hprobs_by_block(
            evt, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate( hcols, axis=2 )  #axes = (spam+gatestring, derivParam1, derivParam2)
        all_d12cols = np.concatenate( d12cols, axis=2 )

        #mhcols = []
        #md12cols = []
        #mslicesList = [ (slice(0,nP),slice(i,i+1)) for i in range(1,10) ]
        #for s1,s2, hprobs_col, dprobs12_col in self.mgateset.bulk_hprobs_by_block(
        #    spam_label_rows, mevt, mslicesList, True):
        #    mhcols.append(hprobs_col)
        #    md12cols.append(dprobs12_col)
        #mall_hcols = np.concatenate( mhcols, axis=2 )  #axes = (spam+gatestring, derivParam1, derivParam2)
        #mall_d12cols = np.concatenate( md12cols, axis=2 )

        self.assertArraysAlmostEqual(all_hcols,hprobs_to_fill[:,:,1:10])
        self.assertArraysAlmostEqual(all_d12cols,dprobs12[:,:,1:10])
        #self.assertArraysAlmostEqual(mall_hcols,mhprobs_to_fill[:,:,1:10], places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,mdprobs12[:,:,1:10], places=FD_HESS_PLACES)
        #
        #self.assertArraysAlmostEqual(mall_hcols,all_hcols, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,all_d12cols, places=FD_HESS_PLACES)

        
        hcols = []
        d12cols = []
        slicesList = [ (slice(2,12),slice(i,i+1)) for i in range(1,10) ]
        for s1,s2, hprobs_col, dprobs12_col in self.gateset.bulk_hprobs_by_block(
            evt, slicesList, True):
            hcols.append(hprobs_col)
            d12cols.append(dprobs12_col)
        all_hcols = np.concatenate( hcols, axis=2 )  #axes = (spam+gatestring, derivParam1, derivParam2)
        all_d12cols = np.concatenate( d12cols, axis=2 )

        #mhcols = []
        #md12cols = []
        #mslicesList = [ (slice(2,12),slice(i,i+1)) for i in range(1,10) ]
        #for s1,s2, hprobs_col, dprobs12_col in self.mgateset.bulk_hprobs_by_block(
        #    mevt, mslicesList, True):
        #    mhcols.append(hprobs_col)
        #    md12cols.append(dprobs12_col)
        #mall_hcols = np.concatenate( mhcols, axis=2 )  #axes = (spam+gatestring, derivParam1, derivParam2)
        #mall_d12cols = np.concatenate( md12cols, axis=2 )

        self.assertArraysAlmostEqual(all_hcols,hprobs_to_fill[:,2:12,1:10])
        self.assertArraysAlmostEqual(all_d12cols,dprobs12[:,2:12,1:10])
        #self.assertArraysAlmostEqual(mall_hcols,mhprobs_to_fill[:,2:12,1:10], places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,mdprobs12[:,2:12,1:10], places=FD_HESS_PLACES)
        #
        #self.assertArraysAlmostEqual(mall_hcols,all_hcols, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mall_d12cols,all_d12cols, places=FD_HESS_PLACES)


        hprobs_by_block = np.zeros(hprobs_to_fill.shape,'d')
        dprobs12_by_block = np.zeros(dprobs12.shape,'d')
        #mhprobs_by_block = np.zeros(mhprobs_to_fill.shape,'d')
        #mdprobs12_by_block = np.zeros(mdprobs12.shape,'d')
        blocks1 = pygsti.tools.mpitools.slice_up_range(nP, 3)
        blocks2 = pygsti.tools.mpitools.slice_up_range(nP, 5)
        slicesList = list(itertools.product(blocks1,blocks2))
        for s1,s2, hprobs_blk, dprobs12_blk in self.gateset.bulk_hprobs_by_block(
            evt, slicesList, True):
            hprobs_by_block[:,s1,s2] = hprobs_blk
            dprobs12_by_block[:,s1,s2] = dprobs12_blk

        #again, but no dprobs12
        hprobs_by_block2 = np.zeros(hprobs_to_fill.shape,'d')
        for s1,s2, hprobs_blk in self.gateset.bulk_hprobs_by_block(
                evt, slicesList, False):
            hprobs_by_block2[:,s1,s2] = hprobs_blk

        #for s1,s2, hprobs_blk, dprobs12_blk in self.mgateset.bulk_hprobs_by_block(
        #    mevt, slicesList, True):
        #    mhprobs_by_block[:,s1,s2] = hprobs_blk
        #    mdprobs12_by_block[:,s1,s2] = dprobs12_blk

        self.assertArraysAlmostEqual(hprobs_by_block,hprobs_to_fill)
        self.assertArraysAlmostEqual(hprobs_by_block2,hprobs_to_fill)
        self.assertArraysAlmostEqual(dprobs12_by_block,dprobs12)
        #self.assertArraysAlmostEqual(mhprobs_by_block,hprobs_to_fill, places=FD_HESS_PLACES)
        #self.assertArraysAlmostEqual(mdprobs12_by_block,dprobs12, places=FD_HESS_PLACES)


        #print("****DEBUG HESSIAN BY COL****")
        #print("shape = ",all_hcols.shape)
        #to_check = hprobs_to_fill[:,2:12,1:10]
        #for si in range(all_hcols.shape[0]):
        #    for stri in range(all_hcols.shape[1]):
        #        diff = np.linalg.norm(all_hcols[si,stri]-to_check[si,stri])
        #        print("[%d,%d] diff = %g" % (si,stri,diff))
        #        if diff > 1e-6:
        #            for i in range(all_hcols.shape[2]):
        #                for j in range(all_hcols.shape[3]):
        #                    x = all_hcols[si,stri,i,j]
        #                    y = to_check[si,stri,i,j]
        #                    if abs(x-y) > 1e-6:
        #                        print("  el(%d,%d):  %g - %g = %g" % (i,j,x,y,x-y))


    def test_tree_construction(self):
        gatestrings = pygsti.construction.gatestring_list(
            [('Gx',),
             ('Gy',),
             ('Gx','Gy'),
             ('Gy','Gy'),
             ('Gy','Gx'),
             ('Gx','Gx','Gx'),
             ('Gx','Gy','Gx'),
             ('Gx','Gy','Gy'),
             ('Gy','Gy','Gy'),
             ('Gy','Gx','Gx') ] )
        evt,lookup,outcome_lookup = self.gateset.bulk_evaltree( gatestrings, maxTreeSize=4 )
        mevt,mlookup,moutcome_lookup = self.mgateset.bulk_evaltree( gatestrings, maxTreeSize=4 )

        evt,lookup,outcome_lookup = self.gateset.bulk_evaltree( gatestrings, minSubtrees=2, maxTreeSize=4 )
        self.assertWarns(self.gateset.bulk_evaltree, gatestrings, minSubtrees=3, maxTreeSize=8 )
           #balanced to trigger 2 re-splits! (Warning: could not create a tree ...)
           
        mevt,mlookup,moutcome_lookup = self.mgateset.bulk_evaltree( gatestrings, minSubtrees=2, maxTreeSize=4 )

        ##Make a few-param gateset to better test mem limits
        gs_few = self.gateset.copy()
        gs_few.set_all_parameterizations("static")
        gs_few.preps['rho0'] = self.gateset.preps['rho0'].copy()
        self.assertEqual(gs_few.num_params(),4)

        #gs_big = pygsti.construction.build_gateset(
        #    [8], [('Q0','Q3','Q2')],['Gi'], [ "I(Q0)"])
        #gs_big._calcClass = GateMapCalc

        class FakeComm(object):
            def __init__(self,size): self.size = size
            def Get_rank(self): return 0
            def Get_size(self): return self.size
            def bcast(self,obj, root=0): return obj
            
        for nprocs in (1,4,10,40,100):
            fake_comm = FakeComm(nprocs)
            for distributeMethod in ('deriv','gatestrings'):
                for memLimit in (-100, 1024, 10*1024, 100*1024, 1024**2, 10*1024**2):
                    print("Nprocs = %d, method = %s, memLim = %g" % (nprocs, distributeMethod, memLimit))
                    try:
                        evt,_,_,lookup,outcome_lookup = self.gateset.bulk_evaltree_from_resources(
                            gatestrings, memLimit=memLimit, distributeMethod=distributeMethod,
                            subcalls=['bulk_fill_hprobs'], comm=fake_comm)
                        evt,_,_,lookup,outcome_lookup = self.mgateset.bulk_evaltree_from_resources(
                            gatestrings, memLimit=memLimit, distributeMethod=distributeMethod,
                            subcalls=['bulk_fill_hprobs'], comm=fake_comm)
                        evt,_,_,lookup,outcome_lookup = gs_few.bulk_evaltree_from_resources(
                            gatestrings, memLimit=memLimit, distributeMethod=distributeMethod,
                            subcalls=['bulk_fill_hprobs'], comm=fake_comm)
                        evt,_,_,lookup,outcome_lookup = gs_few.bulk_evaltree_from_resources(
                            gatestrings, memLimit=memLimit, distributeMethod=distributeMethod,
                            subcalls=['bulk_fill_dprobs'], comm=fake_comm) #where bNp2Matters == False
                                                
                    except MemoryError:
                        pass #OK - when memlimit is too small and splitting is unproductive

        #balanced not implemented
        with self.assertRaises(NotImplementedError):
            evt,_,_,lookup,outcome_lookup = self.gateset.bulk_evaltree_from_resources(
                gatestrings, memLimit=memLimit, distributeMethod="balanced", subcalls=['bulk_fill_hprobs'])
                
                

    def test_tree_splitting(self):
        gatestrings = [('Gx',),
                       ('Gy',),
                       ('Gx','Gy'),
                       ('Gy','Gy'),
                       ('Gy','Gx'),
                       ('Gx','Gx','Gx'),
                       ('Gx','Gy','Gx'),
                       ('Gx','Gy','Gy'),
                       ('Gy','Gy','Gy'),
                       ('Gy','Gx','Gx') ]
        evtA,lookupA,outcome_lookupA = self.gateset.bulk_evaltree( gatestrings )

        evtB,lookupB,outcome_lookupB = self.gateset.bulk_evaltree( gatestrings )
        lookupB = evtB.split(lookupB, maxSubTreeSize=4)

        evtC,lookupC,outcome_lookupC = self.gateset.bulk_evaltree( gatestrings )
        lookupC = evtC.split(lookupC, numSubTrees=3)

        with self.assertRaises(ValueError):
            evtBad,lkup,_ = self.gateset.bulk_evaltree( gatestrings )
            evtBad.split(lkup, numSubTrees=3, maxSubTreeSize=4) #can't specify both

        self.assertFalse(evtA.is_split())
        self.assertTrue(evtB.is_split())
        self.assertTrue(evtC.is_split())
        self.assertEqual(len(evtA.get_sub_trees()), 1)
        self.assertEqual(len(evtB.get_sub_trees()), 5) #empirically
        self.assertEqual(len(evtC.get_sub_trees()), 3)
        self.assertLessEqual(max([len(subTree)
                             for subTree in evtB.get_sub_trees()]), 4)

        #print "Lenghts = ",len(evtA.get_sub_trees()),len(evtB.get_sub_trees()),len(evtC.get_sub_trees())
        #print "SubTree sizes = ",[len(subTree) for subTree in evtC.get_sub_trees()]

        bulk_probsA = np.empty( evtA.num_final_elements(), 'd')
        bulk_probsB = np.empty( evtB.num_final_elements(), 'd')
        bulk_probsC = np.empty( evtC.num_final_elements(), 'd')
        self.gateset.bulk_fill_probs(bulk_probsA, evtA)
        self.gateset.bulk_fill_probs(bulk_probsB, evtB)
        self.gateset.bulk_fill_probs(bulk_probsC, evtC)

        for i,gstr in enumerate(gatestrings):
            self.assertArraysAlmostEqual(bulk_probsA[ lookupA[i] ],
                                         bulk_probsB[ lookupB[i] ])
            self.assertArraysAlmostEqual(bulk_probsA[ lookupA[i] ],
                                         bulk_probsC[ lookupC[i] ])


    def test_failures(self):

        with self.assertRaises(KeyError):
            self.gateset['Non-existent-key']

        with self.assertRaises(KeyError):
            self.gateset['Non-existent-key'] = np.zeros((4,4),'d') #can't set things not in the gateset

        #with self.assertRaises(ValueError):
        #    self.gateset['Gx'] = np.zeros((4,4),'d') #can't set matrices

        #with self.assertRaises(ValueError):
        #    self.gateset.update( {'Gx': np.zeros((4,4),'d') } )

        #with self.assertRaises(ValueError):
        #    self.gateset.update( Gx=np.zeros((4,4),'d') )

        #with self.assertRaises(TypeError):
        #    self.gateset.update( 1, 2 ) #too many positional arguments...

        #with self.assertRaises(ValueError):
        #    self.gateset.setdefault('Gx',np.zeros((4,4),'d'))

        with self.assertRaises(ValueError):
            self.gateset['Gbad'] = pygsti.obj.FullyParameterizedGate(np.zeros((5,5),'d')) #wrong gate dimension

        gs_multispam = self.gateset.copy()
        gs_multispam.preps['rho1'] = gs_multispam.preps['rho0'].copy()
        gs_multispam.povms['M2'] = gs_multispam.povms['Mdefault'].copy()
        with self.assertRaises(ValueError):
            gs_multispam.prep #can only use this property when there's a *single* prep
        with self.assertRaises(ValueError):
            gs_multispam.effects #can only use this property when there's a *single* POVM
        with self.assertRaises(ValueError):
            prep,gates,povm = gs_multispam.split_gatestring( pygsti.obj.GateString(('Gx','Mdefault')) )
        with self.assertRaises(ValueError):
            prep,gates,povm = gs_multispam.split_gatestring( pygsti.obj.GateString(('rho0','Gx')) )

        gs = self.gateset.copy()
        gs._paramvec[:] = 0.0 #mess with paramvec to get error below
        with self.assertRaises(ValueError):
            gs._check_paramvec(debug=True) # param vec is now out of sync!


    def test_iteration(self):
        #Iterate over all gates and SPAM matrices
        #for mx in self.gateset.iterall():
        pass

    def test_deprecated_functions(self):
        name = self.gateset.get_basis_name()
        dim  = self.gateset.get_basis_dimension()
        self.gateset.set_basis(name, dim)

        with self.assertRaises(AssertionError):
            self.gateset.get_prep_labels()
        with self.assertRaises(AssertionError):
            self.gateset.get_effect_labels()
        with self.assertRaises(AssertionError):
            self.gateset.get_preps()
        with self.assertRaises(AssertionError):
            self.gateset.get_effects()
        with self.assertRaises(AssertionError):
            self.gateset.num_preps()
        with self.assertRaises(AssertionError):
            self.gateset.num_effects()
        with self.assertRaises(AssertionError):
            self.gateset.get_reverse_spam_defs()
        with self.assertRaises(AssertionError):
            self.gateset.get_spam_labels()
        with self.assertRaises(AssertionError):
            self.gateset.get_spamgate(None)
        with self.assertRaises(AssertionError):
            self.gateset.iter_gates()
        with self.assertRaises(AssertionError):
            self.gateset.iter_preps()
        with self.assertRaises(AssertionError):
            self.gateset.iter_effects()


        #simulate copying an old gateset
        old_gs = self.gateset.copy()
        del old_gs.__dict__['_calcClass']
        del old_gs.__dict__['basis']
        old_gs._basisNameAndDim = ('pp',2)
        copy_of_old = old_gs.copy()

    def test_load_old_gateset(self):
        vs = "v2" if self.versionsuffix == "" else "v3"
        pygsti.obj.results.enable_old_python_results_unpickling()
        with open(compare_files + "/pygsti0.9.3.gateset.pkl.%s" % vs,'rb') as f:
            gs = pickle.load(f)
        pygsti.obj.results.disable_old_python_results_unpickling()
        with open(temp_files + "/repickle_old_gateset.pkl.%s" % vs,'wb') as f:
            pickle.dump(gs, f)

        #also test automatic setting of _calcClass
        gs = self.gateset.copy()
        del gs._calcClass
        c = gs._calc() #automatically sets _calcClass
        self.assertTrue(hasattr(gs,'_calcClass'))


    def test_base_gatecalc(self):
        rawCalc = pygsti.objects.gatecalc.GateCalc(4, {pygsti.obj.Label('Gx'): pygsti.obj.FullyParameterizedGate(np.identity(4,'d')) },
                                                   {},{}, np.zeros(16,'d'), None)

        #Lots of things that derived classes implement
        #with self.assertRaises(NotImplementedError):
        #    rawCalc._buildup_dPG() # b/c gates are not GateMatrix-derived (they're strings in fact!)
        with self.assertRaises(NotImplementedError):
            rawCalc.product(('Gx',))
        with self.assertRaises(NotImplementedError):
            rawCalc.dproduct(('Gx',))
        with self.assertRaises(NotImplementedError):
            rawCalc.hproduct(('Gx',))
        with self.assertRaises(NotImplementedError):
            rawCalc.construct_evaltree()
        with self.assertRaises(NotImplementedError):
            rawCalc.bulk_product(None)
        with self.assertRaises(NotImplementedError):
            rawCalc.bulk_dproduct(None)
        with self.assertRaises(NotImplementedError):
            rawCalc.bulk_hproduct(None)
        with self.assertRaises(NotImplementedError):
            rawCalc.bulk_fill_probs(None,None)
        with self.assertRaises(NotImplementedError):
            rawCalc.bulk_fill_dprobs(None,None)
        with self.assertRaises(NotImplementedError):
            rawCalc.bulk_fill_hprobs(None,None)
        with self.assertRaises(NotImplementedError):
            rawCalc.bulk_hprobs_by_block(None,None)

    def test_base_gatematrixcalc(self):
        rawCalc = self.gateset._calc()

        #Make call variants that aren't called by GateSet routines
        dg = rawCalc.dgate(L('Gx'), flat=False)
        dgflat = rawCalc.dgate(L('Gx'), flat=True)

        rawCalc.hproduct(Ls('Gx','Gx'), flat=True, wrtFilter1=[0,1], wrtFilter2=[1,2,3])
        #rawCalc.pr( Ls('rho0','Mdefault_0'), Ls('Gx','Gx'), clipTo=(-1,1))
        #rawCalc.pr( Ls('rho0','Mdefault_0'), Ls('Gx','Gx'), clipTo=(-1,1), bUseScaling=True)
        rawCalc.prs( L('rho0'), [L('Mdefault_0')], Ls('Gx','Gx'), clipTo=(-1,1))
        rawCalc.prs( L('rho0'), [L('Mdefault_0')], Ls('Gx','Gx'), clipTo=(-1,1), bUseScaling=True)

        custom_spamTuple = ( np.zeros((4,1),'d'), np.zeros((4,1),'d') )
        rawCalc._rhoE_from_spamTuple(custom_spamTuple)

        evt,lookup,outcome_lookup = self.gateset.bulk_evaltree( [('Gx',), ('Gx','Gx')] )
        nEls = evt.num_final_elements()

        mx = np.zeros((nEls,3,3),'d')
        dmx = np.zeros((nEls,3),'d')
        pmx = np.zeros(nEls,'d')
        rawCalc.bulk_fill_hprobs(mx, evt, 
                                 prMxToFill=pmx, deriv1MxToFill=dmx, deriv2MxToFill=dmx,
                                 wrtFilter1=[0,1,2], wrtFilter2=[0,1,2]) #same slice on each deriv
        
        mx = np.zeros((nEls,3,2),'d')
        dmx1 = np.zeros((nEls,3),'d')
        dmx2 = np.zeros((nEls,2),'d')
        pmx = np.zeros(nEls,'d')
        rawCalc.bulk_fill_hprobs(mx, evt, 
                                 prMxToFill=pmx, deriv1MxToFill=dmx1, deriv2MxToFill=dmx2,
                                 wrtFilter1=[0,1,2], wrtFilter2=[2,3]) #different slices on 1st vs. 2nd deriv
                                 

        with self.assertRaises(ValueError):
            rawCalc.estimate_mem_usage(["foobar"], 1,1,1,1,1,1)

        cptpGateset = self.gateset.copy()
        cptpGateset.set_all_parameterizations("CPTP") # so gates have nonzero hessians
        cptpCalc = cptpGateset._calc()

        hg = cptpCalc.hgate(L('Gx'), flat=False)
        hgflat = cptpCalc.hgate(L('Gx'), flat=True)

        cptpCalc.hpr( Ls('rho0','Mdefault_0'), Ls('Gx','Gx'), False,False, clipTo=(-1,1))
        


    def test_base_gatemapcalc(self):
        rawCalc = self.mgateset._calc()
        
        #Make call variants that aren't called by GateSet routines
        #rawCalc.pr( Ls('rho0','Mdefault_0'), Ls('Gx','Gx'), clipTo=(-1,1))
        #rawCalc.pr( Ls('rho0','Mdefault_0'), Ls('Gx','Gx'), clipTo=(-1,1), bUseScaling=True)
        rawCalc.prs( L('rho0'),[L('Mdefault_0')], Ls('Gx','Gx'), clipTo=(-1,1))
        rawCalc.prs( L('rho0'),[L('Mdefault_0')], Ls('Gx','Gx'), clipTo=(-1,1), bUseScaling=True)
        rawCalc.hpr( Ls('rho0','Mdefault_0'), Ls('Gx','Gx'), False,False, clipTo=(-1,1))
        rawCalc.hpr( Ls('rho0','Mdefault_0'), Ls('Gx','Gx'), True,True, clipTo=(-1,1))

        #Custom spamtuples aren't supported anymore
        #custom_spamTuple = ( np.nan*np.ones((4,1),'d'), np.zeros((4,1),'d') )
        #rawCalc.pr( custom_spamTuple, ('Gx','Gx'), clipTo=(-1,1), bUseScaling=True)

        rawCalc.estimate_cache_size(100)
        with self.assertRaises(ValueError):
            rawCalc.estimate_mem_usage(["foobar"], 1,1,1,1,1,1)

        evt,lookup,outcome_lookup = self.mgateset.bulk_evaltree( [('Gx',), ('Gx','Gx')] )
        nEls = evt.num_final_elements()
        nP = self.mgateset.num_params()

        mx = np.zeros((nEls,3),'d')
        pmx = np.zeros(nEls,'d')
        rawCalc.bulk_fill_dprobs(mx, evt, 
                                 prMxToFill=pmx, clipTo=(-1,1),wrtFilter=[0,1,2])

        mx = np.zeros((nEls,nP),'d')
        rawCalc.bulk_fill_dprobs(mx, evt, 
                                 prMxToFill=pmx, clipTo=(-1,1), wrtBlockSize=2)


        mx = np.zeros((nEls,3,3),'d')
        dmx = np.zeros((nEls,3),'d')
        pmx = np.zeros(nEls,'d')
        rawCalc.bulk_fill_hprobs(mx, evt, clipTo=(-1,1),
                                 prMxToFill=pmx, deriv1MxToFill=dmx, deriv2MxToFill=dmx,
                                 wrtFilter1=[0,1,2], wrtFilter2=[0,1,2]) #same slice on each deriv

        mx = np.zeros((nEls,3,2),'d')
        dmx1 = np.zeros((nEls,3),'d')
        dmx2 = np.zeros((nEls,2),'d')
        pmx = np.zeros(nEls,'d')
        rawCalc.bulk_fill_hprobs(mx, evt, clipTo=(-1,1),
                                 prMxToFill=pmx, deriv1MxToFill=dmx1, deriv2MxToFill=dmx2,
                                 wrtFilter1=[0,1,2], wrtFilter2=[2,3]) #different slices on 1st vs. 2nd deriv

        mx = np.zeros((nEls,nP,nP),'d')
        dmx1 = np.zeros((nEls,nP),'d')
        dmx2 = np.zeros((nEls,nP),'d')
        pmx = np.zeros(nEls,'d')
        rawCalc.bulk_fill_hprobs(mx, evt, clipTo=(-1,1),
                                 prMxToFill=pmx, deriv1MxToFill=dmx1, deriv2MxToFill=dmx2,
                                 wrtBlockSize1=2, wrtBlockSize2=3) #use block sizes

        
    def test_base_gatesetmember(self):
        #Test some parts of GateSetMember that aren't tested elsewhere
        raw_member = pygsti.objects.gatesetmember.GateSetMember(dim=4, evotype="densitymx")
        with self.assertRaises(ValueError):
            raw_member.gpindices = slice(0,3) # read-only!
        with self.assertRaises(ValueError):
            raw_member.parent = None # read-only!

        #Test _compose_gpindices
        parent_gpindices = slice(10,20)
        child_gpindices = slice(2,4)
        x = pygsti.objects.gatesetmember._compose_gpindices(
            parent_gpindices, child_gpindices)
        self.assertEqual(x, slice(12,14))
            
        parent_gpindices = slice(10,20)
        child_gpindices = np.array([0,2,4],'i')
        x = pygsti.objects.gatesetmember._compose_gpindices(
            parent_gpindices, child_gpindices)
        self.assertEqual(list(x), list(np.array([10,12,14],'i'))) # lists so assertEqual works

        parent_gpindices = np.array([2,4,6,8,10],'i')
        child_gpindices = np.array([0,2,4],'i')
        x = pygsti.objects.gatesetmember._compose_gpindices(
            parent_gpindices, child_gpindices)
        self.assertEqual(list(x), list(np.array([2,6,10],'i')))

        #Test _decompose_gpindices
        parent_gpindices = slice(10,20)
        sibling_gpindices = slice(12,14)
        x = pygsti.objects.gatesetmember._decompose_gpindices(
            parent_gpindices, sibling_gpindices)
        self.assertEqual(x, slice(2,4))
            
        parent_gpindices = slice(10,20)
        sibling_gpindices = np.array([10,12,14],'i')
        x = pygsti.objects.gatesetmember._decompose_gpindices(
            parent_gpindices, sibling_gpindices)
        self.assertEqual(list(x), list(np.array([0,2,4],'i')))

        parent_gpindices = np.array([2,4,6,8,10],'i')
        sibling_gpindices = np.array([2,6,10],'i')
        x = pygsti.objects.gatesetmember._decompose_gpindices(
            parent_gpindices, sibling_gpindices)
        self.assertEqual(list(x), list(np.array([0,2,4],'i')))

    def test_gpindices(self):
        #Test instrument construction with elements whose gpindices are already initialized.
        # Since this isn't allowed currently (a future functionality), we need to do some hacking
        gs = self.gateset.copy()
        gs.gates['Gnew1'] = pygsti.obj.FullyParameterizedGate( np.identity(4,'d') )
        del gs.gates['Gnew1']

        v = gs.to_vector()
        Np = gs.num_params()
        gate_with_gpindices = pygsti.obj.FullyParameterizedGate( np.identity(4,'d') )
        gate_with_gpindices[0,:] = v[0:4]
        gate_with_gpindices.set_gpindices(np.concatenate( (np.arange(0,4), np.arange(Np,Np+12)) ), gs) #manually set gpindices
        gs.gates['Gnew2'] = gate_with_gpindices
        gs.gates['Gnew3'] = pygsti.obj.FullyParameterizedGate( np.identity(4,'d') )
        del gs.gates['Gnew3'] #this causes update of Gnew2 indices
        del gs.gates['Gnew2']

    def test_ondemand_probabilities(self):
        #First create a "sparse" dataset
        # # Columns = 0 count, 1 count 
        dataset_txt = \
"""# Test Sparse format data set
{}    0:0  1:100
Gx    0:10 1:90 2:0
GxGy  0:40 1:60
Gx^4  0:100
"""
        with open(temp_files + "/SparseDataset.txt",'w') as f:
            f.write(dataset_txt)

        ds = pygsti.io.load_dataset(temp_files + "/SparseDataset.txt")
        self.assertEqual(ds.get_outcome_labels(), [('0',), ('1',), ('2',)])
        self.assertEqual(ds[()].outcomes, [('1',)]) # only nonzero count is 1-count
        self.assertEqual(ds[()]['2'], 0) # but we can query '2' since it's a valid outcome label

        gstrs = list(ds.keys())
        raw_dict, elIndices, outcome_lookup, ntotal = std1Q_XYI.gs_target.compile_gatestrings(gstrs, ds)

        print("Raw gs -> spamtuple dict:\n","\n".join(["%s: %s" % (str(k),str(v)) for k,v in raw_dict.items()]))
        print("\nElement indices lookup (orig gstr index -> element indices):\n",elIndices)
        print("\nOutcome lookup (orig gstr index -> list of outcome for each element):\n",outcome_lookup)
        print("\ntotal elements = ", ntotal)

        self.assertEqual(raw_dict[()], [(L('rho0'), L('Mdefault_1'))])
        self.assertEqual(raw_dict[('Gx',)], [(L('rho0'), L('Mdefault_0')),(L('rho0'), L('Mdefault_1'))])
        self.assertEqual(raw_dict[('Gx','Gy')], [(L('rho0'), L('Mdefault_0')),(L('rho0'), L('Mdefault_1'))])
        self.assertEqual(raw_dict[('Gx',)*4], [(L('rho0'), L('Mdefault_0'))])

        self.assertEqual(elIndices, collections.OrderedDict(
            [(0, slice(0, 1, None)), (1, slice(1, 3, None)), (2, slice(3, 5, None)), (3, slice(5, 6, None))]) )

        self.assertEqual(outcome_lookup, collections.OrderedDict(
            [(0, [('1',)]), (1, [('0',), ('1',)]), (2, [('0',), ('1',)]), (3, [('0',)])]) )

        self.assertEqual(ntotal, 6)


        #A sparse dataset loading test using the more common format:
        dataset_txt2 = \
"""## Columns = 0 count, 1 count
{} 0 100
Gx 10 90
GxGy 40 60
Gx^4 100 0
"""

        with open(temp_files + "/SparseDataset2.txt",'w') as f:
            f.write(dataset_txt2)

        ds = pygsti.io.load_dataset(temp_files + "/SparseDataset2.txt")
        self.assertEqual(ds.get_outcome_labels(), [('0',), ('1',)])
        self.assertEqual(ds[()].outcomes, [('1',)]) # only nonzero count is 1-count
        with self.assertRaises(KeyError):
            ds[()]['2'] # *can't* query '2' b/c it's NOT a valid outcome label here

        

            

if __name__ == "__main__":
    unittest.main(verbosity=2)
