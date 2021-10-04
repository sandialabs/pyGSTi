from ..util import BaseCase

import pygsti
from pygsti.protocols import vb as _vb
from pygsti.processors import CliffordCompilationRules as CCR
from pygsti.processors import QubitProcessorSpec as QPS

class TestPeriodicMirrorCircuitsDesign(BaseCase):

    def test_design_construction(self):

        n = 4
        qs = ['Q'+str(i) for i in range(n)]
        ring = [('Q'+str(i),'Q'+str(i+1)) for i in range(n-1)]

        gateset1 = ['Gcphase'] + ['Gc'+str(i) for i in range(24)]
        pspec1 = QPS(n, gateset1, availability={'Gcphase':ring}, qubit_labels=qs)
        tmodel1 = pygsti.models.create_crosstalk_free_model(pspec1)

        depths = [0, 2, 8]
        q_set = ('Q0', 'Q1', 'Q2')

        clifford_compilations = {'absolute': CCR.create_standard(pspec1, 'absolute', ('paulis', '1Qcliffords'), verbosity=0)}

        design1 = _vb.PeriodicMirrorCircuitDesign (pspec1, depths, 3, qubit_labels=q_set,
                                        clifford_compilations=clifford_compilations, sampler='edgegrab', samplerargs=(0.25,))

        [[self.assertAlmostEqual(c.simulate(tmodel1)[bs],1.) for c, bs in zip(cl, bsl)] for cl, bsl in zip(design1.circuit_lists, design1.idealout_lists)]

