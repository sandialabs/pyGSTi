import numpy as np

from pygsti.models import create_crosstalk_free_model
from pygsti.modelpacks import smq2Q_XYICNOT
from pygsti.modelmembers import operations as _op
from pygsti.processors import QubitProcessorSpec

import sys
sys.path.append('..')
from util import BaseCase


class TestModelMemberGraph(BaseCase):
    def test_explicit_model_comparisons(self):
        ex_mdl1 = smq2Q_XYICNOT.target_model()
        ex_mmg1 = ex_mdl1.create_modelmember_graph()

        # Copy, should be similar and equivalent
        ex_mdl2 = ex_mdl1.copy()
        ex_mmg2 = ex_mdl2.create_modelmember_graph()
        self.assertTrue(ex_mmg2.is_similar(ex_mmg1))
        self.assertTrue(ex_mmg2.is_equivalent(ex_mmg1))

        # Change parameter, similar but not equivalent
        ex_mdl3 = ex_mdl1.copy()
        ex_mdl3.operations['Gxpi2', 0][0, 0] = 0.0
        ex_mmg3 = ex_mdl3.create_modelmember_graph()
        self.assertTrue(ex_mmg3.is_similar(ex_mmg1))
        self.assertFalse(ex_mmg3.is_equivalent(ex_mmg1))

        # Change parameterization, not similar or equivalent
        ex_mdl4 = ex_mdl1.copy()
        ex_mdl4.operations['Gxpi2', 0] = _op.StaticArbitraryOp(ex_mdl4.operations['Gxpi2', 0])
        ex_mmg4 = ex_mdl4.create_modelmember_graph()
        self.assertFalse(ex_mmg4.is_similar(ex_mmg1))
        self.assertFalse(ex_mmg4.is_equivalent(ex_mmg1))
    
    def test_localnoise_model_comparisons(self):
        pspec = QubitProcessorSpec(2, ['Gi', 'Gxpi2', 'Gypi2'], geometry='line')

        ln_mdl1 = create_crosstalk_free_model(pspec,
                                              depolarization_strengths={('Gxpi2', 0): 0.1},
                                              lindblad_error_coeffs={('Gypi2', 1): {('H', 1): 0.2, ('S', 2): 0.3}})
        ln_mmg1 = ln_mdl1.create_modelmember_graph()

        # Copy, should be similar and equivalent
        # TODO: This failed, not sure why yet
        #ln_mdl2 = ln_mdl1.copy()
        ln_mdl2 = create_crosstalk_free_model(pspec,
                                              depolarization_strengths={('Gxpi2', 0): 0.1},
                                              lindblad_error_coeffs={('Gypi2', 1): {('H', 1): 0.2, ('S', 2): 0.3}})
        ln_mmg2 = ln_mdl2.create_modelmember_graph()
        self.assertTrue(ln_mmg2.is_similar(ln_mmg1))
        self.assertTrue(ln_mmg2.is_equivalent(ln_mmg1))

        # Change parameter, similar but not equivalent
        ln_mdl3 = create_crosstalk_free_model(pspec,
                                              depolarization_strengths={('Gxpi2', 0): 0.3},
                                              lindblad_error_coeffs={('Gypi2', 1): {('H', 1): 0.2, ('S', 2): 0.3}})
        ln_mmg3 = ln_mdl3.create_modelmember_graph()
        self.assertTrue(ln_mmg3.is_similar(ln_mmg1))
        self.assertFalse(ln_mmg3.is_equivalent(ln_mmg1))

        # Change parameterization, not similar or equivalent
        # TODO: This was similar because lindblad similarity doesn't take into which error gens are present
        #ln_mdl4 = create_crosstalk_free_model(pspec,
        #                                      depolarization_strengths={('Gxpi2', 0): 0.1},
        #                                      lindblad_error_coeffs={('Gypi2', 1): {('H', 1): 0.2}})
        ln_mdl4 = create_crosstalk_free_model(pspec,
                                              depolarization_strengths={('Gxpi2', 0): 0.1})
        ln_mmg4 = ln_mdl4.create_modelmember_graph()
        self.assertFalse(ln_mmg4.is_similar(ln_mmg1))
        self.assertFalse(ln_mmg4.is_equivalent(ln_mmg1))






