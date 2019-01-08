from pygsti.construction import std1Q_XYI as std
import pygsti
import pygsti.algorithms.directx as directx
import unittest

import numpy as np
import sys, os

from ..testutils import BaseTestCase, temp_files, compare_files
from pygsti.objects import Label as L

class DirectXTestCase(BaseTestCase):

    def setUp(self):
        super(DirectXTestCase, self).setUp()
        self.tgt = std.target_model()
        #OLDself.specs = pygsti.construction.build_spam_specs(None, std.prepStrs, std.effectStrs)
        self.prepStrs = std.prepStrs
        self.effectStrs = std.effectStrs
        self.strs = pygsti.construction.circuit_list(
            [(),  #always need empty string
             ('Gx',), ('Gy',), ('Gi',), #need these for includeTargetOps=True
             ('Gx','Gx'), ('Gx','Gy','Gx')]) #additional

        expstrs = pygsti.construction.create_circuit_list("f0+base+f1",order=['f0','f1','base'],
                                                            f0=std.prepStrs, f1=std.effectStrs,
                                                            base=self.strs)
#        expstrs.extend( pygsti.construction.create_circuit_list("f0+base+f1",order=['f0','f1','base'],
#                                                            f0=std.prepStrs, f1=std.effectStrs,
#                                                            base=self.strs)

#        expstrs = [ pygsti.objects.Circuit( () ) ] + expstrs #add empty string, which is always needed
        
        mdl_datagen = self.tgt.depolarize(op_noise=0.05, spam_noise=0.1)
        self.ds = pygsti.construction.generate_fake_data(mdl_datagen, expstrs, 1000, "multinomial", seed=1234)

    def test_direct_core(self):
        mdl = directx.model_with_lgst_circuit_estimates(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt, includeTargetOps=True,
            circuitLabels=None, svdTruncateTo=4, verbosity=10)

        mdl = directx.model_with_lgst_circuit_estimates(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt, includeTargetOps=False,
            circuitLabels=[L('G0'),L('G1'),L('G2'),L('G3'),L('G4'),L('G5')], svdTruncateTo=4, verbosity=10)
        self.assertEqual(set(mdl.operations.keys()), set([L('G0'),L('G1'),L('G2'),L('G3'),L('G4'),L('G5')]))

        aliases = {'Gy2': ('Gy',)}
        mdl = directx.model_with_lgst_circuit_estimates(
            [pygsti.obj.Circuit(('Gy2',))], self.ds, self.prepStrs, self.effectStrs, self.tgt, includeTargetOps=True,
            circuitLabels=None, svdTruncateTo=4, verbosity=10, opLabelAliases=aliases)



    def test_direct_lgst(self):
        gslist = directx.direct_lgst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            opLabelAliases=None, svdTruncateTo=4, verbosity=10)

    def test_direct_mc2gst(self):
        gslist = directx.direct_mc2gst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            opLabelAliases=None, minProbClipForWeighting=1e-4,
            probClipInterval=(-1e6,1e6), svdTruncateTo=4, verbosity=10)

    def test_direct_mlgst(self):
        gslist = directx.direct_mlgst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            opLabelAliases=None, minProbClip=1e-6,
            probClipInterval=(-1e6,1e6), svdTruncateTo=4, verbosity=10)
        
    def test_focused_mc2gst(self):
        gslist = directx.focused_mc2gst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            opLabelAliases=None, minProbClipForWeighting=1e-4,
            probClipInterval=(-1e6,1e6), verbosity=10)

if __name__ == '__main__':
    unittest.main(verbosity=2)
