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

class LabelTestCase(BaseTestCase):

    def setUp(self):
        super(LabelTestCase, self).setUp()

    def testLabels(self):
        labels = []
        labels.append( L('Gx',0) ) # a LabelTup
        labels.append( L('Gx',(0,1)) ) # a LabelTup
        labels.append( L( ('Gx',0,1) ) ) # a LabelTup
        labels.append( L('Gx') ) # a LabelStr
        labels.append( L('Gx',None) ) #still a LabelStr
        labels.append( L( [('Gx',0),('Gy',0)] ) ) # a LabelTupTup of LabelTup objs
        labels.append( L( (('Gx',None),('Gy',None)) ) ) # a LabelTupTup of LabelStr objs
        labels.append( L( [('Gx',0)] )  ) # just a LabelTup b/c only one component
        labels.append( L( [L('Gx'),L('Gy')] )  ) # a LabelTupTup of LabelStrs
        labels.append( L(L('Gx')) ) # Init from another label
        
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
        #test saving and loading "parallel" operation labels
        gslist = pygsti.construction.circuit_list( [('Gx','Gy'), (('Gx',0),('Gy',1)), ((('Gx',0),('Gy',1)),('Gcnot',0,1)) ])

        pygsti.io.write_circuit_list(temp_files + "/test_gslist.txt", gslist)
        gslist2 = pygsti.io.load_circuit_list(temp_files + "/test_gslist.txt")
        self.assertEqual(gslist,gslist2)

    def test_circuit_init(self):
        #Check that parallel operation labels get converted to circuits properly
        opstr = pygsti.obj.Circuit( ((('Gx',0),('Gy',1)),('Gcnot',0,1)) )
        c = pygsti.obj.Circuit(layer_labels=opstr, num_lines=2)
        print(c._labels)
        self.assertEqual(c._labels, ( L( (('Gx',0),('Gy',1)) ), L('Gcnot',(0,1)) ))


    def test_layerlizzard(self):
        #Test this here b/c auto-gators are associated with parallel operation labels
        availability = {'Gcnot': [(0,1)]}
        mdl = pc.build_cloudnoise_model_from_hops_and_weights(
            2, ['Gx','Gy','Gcnot'], {}, None, availability,
            None, "line", maxIdleWeight=1, maxhops=1,
            extraWeight1Hops=0, extraGateWeight=1, sparse=True,
            sim_type="map", parameterization="H+S")
        
        # mdl[('Gx',0)].factorops  # Composed([fullTargetOp,fullIdleErr,fullLocalErr])
        self.assertEqual( set(mdl.get_primitive_op_labels()), set([L('Gx',0), L('Gy',0), L('Gx',1), L('Gy',1), L('Gcnot',(0,1))]))

        #But we can *compute* with circuits containing parallel labels...
        parallelLbl = L( [('Gx',0),('Gy',1)] )

        with self.assertRaises(KeyError):
            mdl.operation_blks[parallelLbl]
        
        opstr = pygsti.obj.Circuit( (parallelLbl,) )
        probs = mdl.probs(opstr)
        print(probs)

        expected = { ('00',): 0.25, ('01',): 0.25, ('10',): 0.25, ('11',): 0.25 }
        for k,v in probs.items():
            self.assertAlmostEqual(v, expected[k])

    def test_labels_with_time_and_arguments(self):

        #Label with time and args
        l = L('Gx',(0,1),time=1.2, args=('1.4','1.7'))
        self.assertEqual(l.time, 1.2)
        self.assertEqual(l.args,('1.4','1.7'))
        self.assertEqual(tuple(l), ('Gx', 4, '1.4', '1.7', 0, 1) )

        l2 = L(('Gx',';1.4',';1.7',0,1,'!1.25'))
        self.assertEqual(tuple(l2), ('Gx', 4, '1.4', '1.7', 0, 1) )

        l3 = L(('Gx',';','1.4',';','1.7',0,1,'!',1.3))
        self.assertEqual(tuple(l3), ('Gx', 4, '1.4', '1.7', 0, 1) )

        self.assertTrue(l == l2 == l3)

        #Time without args
        l = L('Gx',(0,1),time=1.2)
        self.assertEqual(l.time, 1.2)
        self.assertEqual(l.args,())
        self.assertEqual(tuple(l), ('Gx', 0, 1) )

        #Args without time
        l = L('Gx',(0,1),args=('1.4',))
        self.assertEqual(l.time, 0)
        self.assertEqual(l.args,('1.4',))
        self.assertEqual(tuple(l), ('Gx', 3, '1.4', 0, 1) )

    def test_label_time_is_not_hashed(self):
        #Ensure that time is not considered in the equality (or hashing) of labels - it's a
        # tag-along "comment" that does not change the real value of a Label.
        l1 = L('Gx',time=1.2)
        l2 = L('Gx')
        self.assertEqual(l1,l2)
        self.assertTrue(l1.time != l2.time)

        l1 = L('Gx',(0,),time=1.2)
        l2 = L('Gx',(0,))
        self.assertEqual(l1,l2)
        self.assertTrue(l1.time != l2.time)

    def test_only_nonzero_time_is_printed(self):
        l = L('GrotX',(0,1),args=('1.4',))
        self.assertEqual(str(l), "GrotX;1.4:0:1")  # make sure we don't print time when it's not given (i.e. zero)
        self.assertEqual(l.time, 0.0) # BUT l.time is still 0, not None
        l = L('GrotX',(0,1),args=('1.4',),time=0.2)
        self.assertEqual(str(l), "GrotX;1.4:0:1!0.2")  # make sure we do print time when it's nonzero
        self.assertEqual(l.time, 0.2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
