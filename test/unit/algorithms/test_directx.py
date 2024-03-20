import pygsti.circuits as pc
import pygsti.data as pdata
from pygsti.algorithms import directx
from pygsti.baseobjs import Label as L
from pygsti.circuits import Circuit
from . import fixtures
from ..util import BaseCase

_SEED = 1234

# TODO optimize!
class DirectXTester(BaseCase):
    @classmethod
    def setUpClass(cls):
        super(DirectXTester, cls).setUpClass()
        cls._tgt = fixtures.model.copy()
        cls.prepStrs = fixtures.prep_fids
        cls.effectStrs = fixtures.meas_fids
        cls.strs = [Circuit([], line_labels=(0,)),
                    Circuit([L('Gxpi2',0)], line_labels=(0,)),
                    Circuit([L('Gypi2',0)], line_labels=(0,)),
                    Circuit([L('Gxpi2',0), L('Gxpi2',0)], line_labels=(0,)),
                    Circuit([L('Gxpi2',0), L('Gypi2',0), L('Gxpi2',0)], line_labels=(0,))
                    ]
                    
        expstrs = pc.create_circuits(
            "f0+base+f1", order=['f0', 'f1', 'base'], f0=cls.prepStrs,
            f1=cls.effectStrs, base=cls.strs
        )
        cls._ds = pdata.simulate_data(fixtures.datagen_gateset.copy(), expstrs, 1000, 'multinomial', seed=_SEED)

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

        circuit_labels = [L('G0'), L('G1'), L('G2'), L('G3'), L('G4')]
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
