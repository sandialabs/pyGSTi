"""The pure-Python ``stabilizer_slow`` evotype, the fallback when the extensions aren't built."""
from pygsti.circuits import Circuit
from pygsti.models import create_crosstalk_free_model
from pygsti.processors import QubitProcessorSpec
from ..util import BaseCase


class StabilizerSlowSimulationTester(BaseCase):
    def test_bell_circuit_probabilities(self):
        # Building the model makes computational effect reps; simulating it copies state reps
        pspec = QubitProcessorSpec(2, ['Gh', 'Gcnot'], geometry='line')
        model = create_crosstalk_free_model(pspec, evotype='stabilizer_slow', simulator='map')
        probs = model.probabilities(Circuit([('Gh', 0), ('Gcnot', 0, 1)], line_labels=(0, 1)))
        self.assertAlmostEqual(probs['00'], 0.5)
        self.assertAlmostEqual(probs['11'], 0.5)
        self.assertAlmostEqual(probs['01'] + probs['10'], 0.0)
