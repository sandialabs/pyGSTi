import unittest
import pygsti
import numpy as np
import warnings
import pickle
import os

import pygsti.construction as pc
from pygsti.io import jsoncodec
from pygsti.objects import Label as L

from ..testutils import BaseTestCase, compare_files, temp_files

# This class is for unifying some gatesets that get used in this file and in testGateSets2.py
class LabelTestCase(BaseTestCase):

    def setUp(self):
        super(LabelTestCase, self).setUp()

    def testLabels(self):
        labels = []
        labels.append( pygsti.obj.Label('Gx',0) ) # a LabelTup
        labels.append( pygsti.obj.Label('Gx',(0,1)) ) # a LabelTup
        labels.append( pygsti.obj.Label( ('Gx',0,1) ) ) # a LabelTup
        labels.append( pygsti.obj.Label('Gx') ) # a LabelStr
        labels.append( pygsti.obj.Label('Gx',None) ) #still a LabelStr
        labels.append( pygsti.obj.Label( [('Gx',0),('Gy',0)] ) ) # a LabelTupTup of LabelTup objs
        labels.append( pygsti.obj.Label( (('Gx',None),('Gy',None)) ) ) # a LabelTupTup of LabelStr objs
        labels.append( pygsti.obj.Label( [('Gx',0)] )  ) # just a LabelTup b/c only one component
        labels.append( pygsti.obj.Label( [L('Gx'),L('Gy')] )  ) # a LabelTupTup of LabelStrs
        labels.append( pygsti.obj.Label(L('Gx')) ) # Init from another label
        
        for l in labels:
            native = l.tonative()
            print(l, " (", type(l), "): native =",native)
            if isinstance(l, pygsti.baseobjs.label.LabelTupTup):
                print("  comps: ", ", ".join(["%s (%s)" % (c,str(type(c))) for c in l.components]))
                
            from_native = pygsti.obj.Label(native)
            self.assertEqual(from_native,l)
                
            s = pickle.dumps(l)
            l2 = pickle.loads(s)
            self.assertEqual(type(l),type(l2))
            
            j = jsoncodec.encode_obj(l,False)
            #print("Json: ",j)
            l3 = jsoncodec.decode_obj(j,False)
            #print("Unjsoned ", l3, " a ",type(l3))
            self.assertEqual(type(l),type(l3))
        
    def test_loadsave(self):
        #test saving and loading "parallel" gate labels
        gslist = pygsti.construction.gatestring_list( [('Gx','Gy'), (('Gx',0),('Gy',1)), ((('Gx',0),('Gy',1)),('Gcnot',0,1)) ])

        pygsti.io.write_gatestring_list(temp_files + "/test_gslist.txt", gslist)
        gslist2 = pygsti.io.load_gatestring_list(temp_files + "/test_gslist.txt")
        self.assertEqual(gslist,gslist2)

    def test_circuit_init(self):
        #Check that parallel gate labels get converted to circuits properly
        gstr = pygsti.obj.GateString( ((('Gx',0),('Gy',1)),('Gcnot',0,1)) )
        c = pygsti.obj.Circuit(gatestring=gstr, num_lines=2)
        print(c)
        self.assertEqual(c.line_items, [[L(('Gx',0)), L(('Gcnot',0,1))], [ L(('Gy',1)), L(('Gcnot',0,1))]])


    def test_autogator(self):
        #Test this here b/c auto-gators are associated with parallel gate labels
        gs = pc.build_nqnoise_gateset(2, "line", [(0,1)], maxIdleWeight=2, maxhops=1,
                                      extraWeight1Hops=0, extraGateWeight=1, verbosity=1,
                                      sim_type="map", parameterization="H+S", sparse=True)
        
        # gs[('Gx',0)].factorgates  # Composed([fullTargetOp,fullIdleErr,fullLocalErr])
        self.assertEqual( set(gs.gates.keys()), set([L('Gi'), L('Gx',0), L('Gy',0), L('Gx',1), L('Gy',1), L('Gcnot',(0,1))]))

        #But we can *compute* with gatestrings containing parallel labels...
        parallelLbl = L( [('Gx',0),('Gy',1)] )

        with self.assertRaises(KeyError):
            gs.gates[parallelLbl]
        
        gstr = pygsti.obj.GateString( (parallelLbl,) )
        probs = gs.probs(gstr)
        print(probs)

        expected = { ('00',): 0.25, ('01',): 0.25, ('10',): 0.25, ('11',): 0.25 }
        for k,v in probs.items():
            self.assertAlmostEqual(v, expected[k])
        
