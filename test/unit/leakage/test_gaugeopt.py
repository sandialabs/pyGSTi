import numpy as np

from pygsti.baseobjs.basis import Basis, BuiltinBasis
from pygsti.leakage.gaugeopt import _direct_sum_unitary_group
from pygsti.models.gaugegroup import DirectSumUnitaryGroup, U1Group, TrivialGaugeGroup
from ..util import BaseCase


class DirectSumUnitaryGroupBuilderTester(BaseCase):
    """
    Tests for the private _direct_sum_unitary_group helper.

    The standard leaky-qubit call uses subspace_bases = [pp(4), pp(1)] where
    pp(4) has dim=4 and pp(1) has dim=1.  By default the dim-1 subspace is
    marked trivial (TrivialGaugeGroup); passing triviality_flags=[False, False]
    forces it to produce a U1Group instead.
    """

    def setUp(self):
        # pp(4): computational subspace (1-qubit, 4-dim superop)
        # pp(1): leakage level (1-dim Hilbert space, 1-dim superop)
        self.subspace_bases = [Basis.cast('pp', 4), Basis.cast('pp', 1)]
        self.full_basis      = BuiltinBasis('l2p1', 9)

    def test_default_flags_uses_trivial_for_dim1(self):
        # triviality_flags defaults to [dim==1 for each basis]:
        # pp(4): dim=4 → False → UnitaryGaugeGroup
        # pp(1): dim=1 → True  → TrivialGaugeGroup (0 params)
        gg = _direct_sum_unitary_group(self.subspace_bases, self.full_basis)
        self.assertIsInstance(gg, DirectSumUnitaryGroup)

    def test_explicit_false_flag_on_dim1_uses_u1group(self):
        gg_default = _direct_sum_unitary_group(self.subspace_bases, self.full_basis)
        gg_u1      = _direct_sum_unitary_group(
            self.subspace_bases, self.full_basis, triviality_flags=[False, False]
        )
        self.assertIsInstance(gg_u1, DirectSumUnitaryGroup)
        # U1Group contributes 1 param; TrivialGaugeGroup contributes 0.
        self.assertEqual(gg_u1.num_params, gg_default.num_params + 1)

    def test_explicit_true_flag_matches_default(self):
        # Explicitly passing the same flags as the default should give same num_params.
        gg_default  = _direct_sum_unitary_group(self.subspace_bases, self.full_basis)
        gg_explicit = _direct_sum_unitary_group(
            self.subspace_bases, self.full_basis, triviality_flags=[False, True]
        )
        self.assertEqual(gg_explicit.num_params, gg_default.num_params)
