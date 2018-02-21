from pygsti.construction import std1Q_XYI as std
import pygsti
import pygsti.algorithms.directx as directx
import unittest

import numpy as np
import sys, os

from ..testutils import BaseTestCase, temp_files, compare_files

class DirectXTestCase(BaseTestCase):

    def setUp(self):
        super(DirectXTestCase, self).setUp()
        self.tgt = std.gs_target
        #OLDself.specs = pygsti.construction.build_spam_specs(None, std.prepStrs, std.effectStrs)
        self.prepStrs = std.prepStrs
        self.effectStrs = std.effectStrs
        self.strs = pygsti.construction.gatestring_list(
            [(),  #always need empty string
             ('Gx',), ('Gy',), ('Gi',), #need these for includeTargetGates=True
             ('Gx','Gx'), ('Gx','Gy','Gx')]) #additional

        expstrs = pygsti.construction.create_gatestring_list("f0+base+f1",order=['f0','f1','base'],
                                                            f0=std.prepStrs, f1=std.effectStrs,
                                                            base=self.strs)
#        expstrs.extend( pygsti.construction.create_gatestring_list("f0+base+f1",order=['f0','f1','base'],
#                                                            f0=std.prepStrs, f1=std.effectStrs,
#                                                            base=self.strs)

#        expstrs = [ pygsti.objects.GateString( () ) ] + expstrs #add empty string, which is always needed
        
        gs_datagen = self.tgt.depolarize(gate_noise=0.05, spam_noise=0.1)
        self.ds = pygsti.construction.generate_fake_data(gs_datagen, expstrs, 1000, "multinomial", seed=1234)

    def test_direct_core(self):
        gs = directx.gateset_with_lgst_gatestring_estimates(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt, includeTargetGates=True,
            gateStringLabels=None, svdTruncateTo=4, verbosity=10)

        gs = directx.gateset_with_lgst_gatestring_estimates(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt, includeTargetGates=False,
            gateStringLabels=['G0','G1','G2','G3','G4','G5'], svdTruncateTo=4, verbosity=10)
        self.assertEqual(set(gs.gates.keys()), set(['G0','G1','G2','G3','G4','G5']))

        aliases = {'Gy2': ('Gy',)}
        gs = directx.gateset_with_lgst_gatestring_estimates(
            [pygsti.obj.GateString(('Gy2',))], self.ds, self.prepStrs, self.effectStrs, self.tgt, includeTargetGates=True,
            gateStringLabels=None, svdTruncateTo=4, verbosity=10, gateLabelAliases=aliases)



    def test_direct_lgst(self):
        gslist = directx.direct_lgst_gatesets(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            gateLabelAliases=None, svdTruncateTo=4, verbosity=10)

    def test_direct_mc2gst(self):
        gslist = directx.direct_mc2gst_gatesets(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            gateLabelAliases=None, minProbClipForWeighting=1e-4,
            probClipInterval=(-1e6,1e6), svdTruncateTo=4, verbosity=10)

    def test_direct_mlgst(self):
        gslist = directx.direct_mlgst_gatesets(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            gateLabelAliases=None, minProbClip=1e-6,
            probClipInterval=(-1e6,1e6), svdTruncateTo=4, verbosity=10)
        
    def test_focused_mc2gst(self):
        gslist = directx.focused_mc2gst_gatesets(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            gateLabelAliases=None, minProbClipForWeighting=1e-4,
            probClipInterval=(-1e6,1e6), verbosity=10)

if __name__ == '__main__':
    unittest.main(verbosity=2)
