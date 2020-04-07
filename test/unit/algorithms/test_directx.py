from ..util import BaseCase
from . import fixtures

import pygsti.construction as pc
from pygsti.objects import Label as L
from pygsti.algorithms import directx

_SEED = 1234

# TODO optimize!
class DirectXTester(BaseCase):
    @classmethod
    def setUpClass(cls):
        super(DirectXTester, cls).setUpClass()
        cls._tgt = fixtures.model.copy()
        cls.prepStrs = fixtures.fiducials
        cls.effectStrs = fixtures.fiducials
        cls.strs = pc.circuit_list([
            (),  # always need empty string
            ('Gx',), ('Gy',), ('Gi',),  # need these for includeTargetOps=True
            ('Gx', 'Gx'), ('Gx', 'Gy', 'Gx')  # additional
        ])
        expstrs = pc.create_circuit_list(
            "f0+base+f1", order=['f0', 'f1', 'base'], f0=fixtures.fiducials,
            f1=fixtures.fiducials, base=cls.strs
        )
        cls._ds = pc.generate_fake_data(fixtures.datagen_gateset.copy(), expstrs, 1000, 'multinomial', seed=_SEED)

    def setUp(self):
        self.tgt = self._tgt.copy()
        self.ds = self._ds.copy()

    def test_model_with_lgst_circuit_estimates(self):
        model = directx.model_with_lgst_circuit_estimates(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            svdTruncateTo=4, verbosity=10
        )
        # TODO assert correctness

        model = directx.model_with_lgst_circuit_estimates(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            includeTargetOps=False, svdTruncateTo=4, verbosity=10
        )
        # TODO assert correctness

        circuit_labels = [L('G0'), L('G1'), L('G2'), L('G3'), L('G4'), L('G5')]
        # circuit_labels = [L('G0'), L('G1'), L('G2'), L('G3')]
        model = directx.model_with_lgst_circuit_estimates(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            circuitLabels=circuit_labels,
            includeTargetOps=False, svdTruncateTo=4, verbosity=10
        )
        self.assertEqual(
            set(model.operations.keys()),
            set(circuit_labels)
        )

    def test_direct_lgst_models(self):
        gslist = directx.direct_lgst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            opLabelAliases=None, svdTruncateTo=4, verbosity=10)
        # TODO assert correctness

    def test_direct_mc2gst_models(self):
        gslist = directx.direct_mc2gst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            opLabelAliases=None, minProbClipForWeighting=1e-4,
            probClipInterval=(-1e6, 1e6), svdTruncateTo=4, verbosity=10)
        # TODO assert correctness

    def test_direct_mlgst_models(self):
        gslist = directx.direct_mlgst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            opLabelAliases=None, minProbClip=1e-6, probClipInterval=(-1e6, 1e6),
            svdTruncateTo=4, verbosity=10)
        # TODO assert correctness

    def test_focused_mc2gst_models(self):
        gslist = directx.focused_mc2gst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            opLabelAliases=None, minProbClipForWeighting=1e-4,
            probClipInterval=(-1e6, 1e6), verbosity=10)
        # TODO assert correctness
