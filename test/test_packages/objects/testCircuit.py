import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import pygsti

from ..testutils import BaseTestCase

class CircuitTestCase(BaseTestCase):
    def test_simulate_circuitlabels(self):
        from pygsti.modelpacks.legacy import std1Q_XYI

        pygsti.obj.Circuit.default_expand_subcircuits = False # so mult/exponentiation => CircuitLabels

        try:
            Gi = pygsti.obj.Circuit(None,stringrep='Gi',editable=True)
            Gy = pygsti.obj.Circuit(None,stringrep='Gy',editable=True)
            c2 = Gy*2
            #print(c2.tup)
            c3 = Gi + c2
            c2.done_editing()
            c3.done_editing()

            Gi.done_editing()
            Gy.done_editing()

            tgt = std1Q_XYI.target_model()
            tgt.sim = pygsti.objects.MapForwardSimulator()  # or use *simple* matrix fwdsim
            # but usual matrix fwdsim takes a long time because it builds a tree.
            for N,zeroProb in zip((1,2,10,100,10000),(0.5, 0, 0, 1, 1)):
                p1 = tgt.probabilities(('Gi',) + ('Gy',)*N)
                p2 = tgt.probabilities( Gi + Gy*N )
                self.assertAlmostEqual(p1['0'], zeroProb)
                self.assertAlmostEqual(p2['0'], zeroProb)
        finally:
            pygsti.obj.Circuit.default_expand_subcircuits = True

    def test_replace_with_idling_line(self):
        c = pygsti.obj.Circuit( [('Gcnot',0,1)], editable=True)
        c.replace_with_idling_line_inplace(0)
        self.assertEqual(c.layertup, ((),))

if __name__ == "__main__":
    unittest.main(verbosity=2)
