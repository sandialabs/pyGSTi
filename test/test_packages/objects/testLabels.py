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


# This class is for unifying some models that get used in this file and in testGateSets2.py
class LabelTestCase(BaseTestCase):
    def test_layerlizzard(self):
        #Test this here b/c auto-gators are associated with parallel operation labels
        availability = {'Gcnot': [(0,1)]}
        mdl = pc.build_standard_cloudnoise_model(2, ['Gx','Gy','Gcnot'], {}, availability,
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
