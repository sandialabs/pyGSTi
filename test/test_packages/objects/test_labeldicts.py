import pygsti
import numpy as np

import pygsti
from pygsti.modelpacks.legacy import std1Q_XYI
from ..testutils import BaseTestCase


class LabelDictTestCase(BaseTestCase):

    def setUp(self):
        super(LabelDictTestCase, self).setUp()

    def test_classical_statespacelabels(self):

        mdl1Q = std1Q_XYI.target_model()
        mdl = pygsti.models.ExplicitOpModel(['Q0', 'C0'])  # one qubit, one classical bit
        self.assertEqual(mdl.dim, 8)
        self.assertEqual(mdl.state_space.dim, 8)
        self.assertEqual(mdl.state_space.tensor_product_blocks_labels, (('Q0', 'C0'),) )
        self.assertEqual(mdl.state_space.tensor_product_blocks_dimensions, ((4,2),))

        mdl.preps['rho_c0'] = np.kron([[1.0],[0.0]], mdl1Q.preps['rho0'])
        mdl.preps['rho_c1'] = np.kron([[0.0],[1.0]], mdl1Q.preps['rho0'])
        mdl.preps['rho_mix'] = np.kron([[0.5],[0.5]], mdl1Q.preps['rho0'])

        EffectVecs = [ (elbl +'c0', np.kron([[1.0],[0.0]], evec)) for elbl,evec in mdl1Q.povms['Mdefault'].items() ] \
                   + [ (elbl +'c1', np.kron([[0.0],[1.0]], evec)) for elbl,evec in mdl1Q.povms['Mdefault'].items() ]
        mdl.povms['Mboth'] = pygsti.modelmembers.povms.UnconstrainedPOVM(EffectVecs, evotype='default', state_space=mdl.state_space)
        EffectVecs = [ (elbl, np.kron([[1.0],[1.0]], evec)) for elbl,evec in mdl1Q.povms['Mdefault'].items() ]
        mdl.povms['Mqubit'] = pygsti.modelmembers.povms.UnconstrainedPOVM(EffectVecs, evotype='default', state_space=mdl.state_space)

        G = np.kron([[1.0,0],[0,1.0]], mdl1Q.operations['Gx'])
        mdl.operations['Gx'] = G

        G = np.kron([[1.0,0],[0,1.0]], mdl1Q.operations['Gy'])
        mdl.operations['Gy'] = G

        G = np.kron([[0,1.0],[1.0,0]], mdl1Q.operations['Gi'])
        mdl.operations['Gcflip'] = G

        G = np.kron([[0.8,0.2],[0.2,0.8]], mdl1Q.operations['Gi'])
        mdl.operations['Gcmix'] = G

        p = mdl.probabilities( ('rho_c1','Gx','Gy','Gcmix','Gx','Mboth') )
        self.assertDictsAlmostEqual(p, pygsti.baseobjs.OutcomeLabelDict([(('0c0',), 0.0), (('1c0',), 0.2), (('0c1',), 0.0), (('1c1',), 0.8)]))
        p = mdl.probabilities( ('rho_c1','Gx','Gy','Gcmix','Gx','Mqubit') )
        self.assertDictsAlmostEqual(p, pygsti.baseobjs.OutcomeLabelDict([(('0',), 0.0), (('1',), 1.0)]))

        p = mdl.probabilities( ('rho_c1','Gx','Gy','Gcmix','Mboth') )
        self.assertDictsAlmostEqual(p, pygsti.baseobjs.OutcomeLabelDict([(('0c0',), 0.1), (('1c0',), 0.1), (('0c1',), 0.4), (('1c1',), 0.4)]))
        p = mdl.probabilities( ('rho_c1','Gx','Gy','Gcmix','Mqubit') )
        self.assertDictsAlmostEqual(p, pygsti.baseobjs.OutcomeLabelDict([(('0',), 0.5), (('1',), 0.5)]))

        p = mdl.probabilities( ('rho_mix', 'Mboth') )
        self.assertDictsAlmostEqual(p, pygsti.baseobjs.OutcomeLabelDict([(('0c0',), 0.5), (('1c0',), 0.0), (('0c1',), 0.5), (('1c1',), 0.0)]))
