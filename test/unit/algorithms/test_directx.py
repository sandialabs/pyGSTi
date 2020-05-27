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
        cls.strs = pc.to_circuits([
            (),  # always need empty string
            ('Gx',), ('Gy',), ('Gi',),  # need these for include_target_ops=True
            ('Gx', 'Gx'), ('Gx', 'Gy', 'Gx')  # additional
        ])
        expstrs = pc.create_circuits(
            "f0+base+f1", order=['f0', 'f1', 'base'], f0=fixtures.fiducials,
            f1=fixtures.fiducials, base=cls.strs
        )
        cls._ds = pc.simulate_data(fixtures.datagen_gateset.copy(), expstrs, 1000, 'multinomial', seed=_SEED)

    def setUp(self):
        self.tgt = self._tgt.copy()
        self.ds = self._ds.copy()

    def test_model_with_lgst_circuit_estimates(self):
        model = directx.model_with_lgst_circuit_estimates(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            svd_truncate_to=4, verbosity=10
        )
        # TODO assert correctness

        model = directx.model_with_lgst_circuit_estimates(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            include_target_ops=False, svd_truncate_to=4, verbosity=10
        )
        # TODO assert correctness

        circuit_labels = [L('G0'), L('G1'), L('G2'), L('G3'), L('G4'), L('G5')]
        # circuit_labels = [L('G0'), L('G1'), L('G2'), L('G3')]
        model = directx.model_with_lgst_circuit_estimates(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            circuit_labels=circuit_labels,
            include_target_ops=False, svd_truncate_to=4, verbosity=10
        )
        self.assertEqual(
            set(model.operations.keys()),
            set(circuit_labels)
        )

    def test_direct_lgst_models(self):
        gslist = directx.direct_lgst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            op_label_aliases=None, svd_truncate_to=4, verbosity=10)
        # TODO assert correctness

    def test_direct_mc2gst_models(self):
        gslist = directx.direct_mc2gst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            op_label_aliases=None, min_prob_clip_for_weighting=1e-4,
            prob_clip_interval=(-1e6, 1e6), svd_truncate_to=4, verbosity=10)
        # TODO assert correctness

    def test_direct_mlgst_models(self):
        gslist = directx.direct_mlgst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            op_label_aliases=None, min_prob_clip=1e-6, prob_clip_interval=(-1e6, 1e6),
            svd_truncate_to=4, verbosity=10)
        # TODO assert correctness

    def test_focused_mc2gst_models(self):
        gslist = directx.focused_mc2gst_models(
            self.strs, self.ds, self.prepStrs, self.effectStrs, self.tgt,
            op_label_aliases=None, min_prob_clip_for_weighting=1e-4,
            prob_clip_interval=(-1e6, 1e6), verbosity=10)
        # TODO assert correctness
