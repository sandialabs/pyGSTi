from pygsti.processors import QubitProcessorSpec
from pygsti.models import create_crosstalk_free_model
from pygsti.circuits import Circuit
from pygsti.modelmembers.operations.opfactory import ComposedOpFactory
from pygsti.modelmembers.operations.depolarizeop import DepolarizeOp

from ..util import BaseCase


class ModelNoiseTester(BaseCase):
    def test_linblad_agrees_with_depol(self):
        pspec = QubitProcessorSpec(1, ["Gi"], geometry="line")

        mdl1 = create_crosstalk_free_model(
            pspec,
            depolarization_parameterization="lindblad",
            depolarization_strengths={'Gi': 0.02}
        )

        mdl2 = create_crosstalk_free_model(
            pspec,
            depolarization_parameterization="depolarize",
            depolarization_strengths={'Gi': 0.02}
        )

        c = Circuit("Gi:0@(0)")
        p1 = mdl1.probabilities(c)
        p2 = mdl2.probabilities(c)
        self.assertAlmostEqual(p1['0'], p2['0'], places=3)
        self.assertAlmostEqual(p1['1'], p2['1'], places=3)
