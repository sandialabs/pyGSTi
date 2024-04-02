
import unittest
import numpy as np
from packaging import version
    
from pygsti.processors import QubitProcessorSpec
from pygsti.models import create_crosstalk_free_model
from pygsti.circuits import Circuit
from pygsti.modelpacks import smq2Q_XYI as std
from pygsti.modelpacks import smq1Q_XYI as std1Q

from pygsti.evotypes.densitymx_slow.opreps import OpRepIdentityPlusErrorgen
from pygsti.evotypes.densitymx.opreps import OpRepDenseSuperop
from ..util import BaseCase

#also catch the attribute error here
try:
    np.int = int  # because old versions of qibo use deprecated (and now removed)
    np.float = float  # types within numpy.  So this is a HACK to get around this.
    np.complex = complex
    import qibo as _qibo
    if version.parse(_qibo.__version__) != version.parse("0.1.7"):
        _qibo = None  # version too low - doesn't contain all the builtin gates, e.g. qibo.gates.S
except (ImportError, AttributeError):
    _qibo = None

#Deprecated numpy calls are currently breaking the qibo import
#so add in a catch for this exception and skip this test if that happens.
try:
    from pygsti.evotypes import qibo as evo_qibo  # don't clobber qibo!
except AttributeError:
    evo_qibo = None



class QiboEvotypeTester(BaseCase):

    def setUp(self):
        self.pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcnot'], geometry='line')
        self.test_circuit = Circuit("Gxpi2:0^2", line_labels=(0, 1))
        # Circuit("Gxpi2:0Gypi2:1Gcnot:0:1", line_labels=(0,1))

        self.mdl_densitymx = create_crosstalk_free_model(
            self.pspec, evotype='densitymx', simulator='map',
            depolarization_strengths={('Gxpi2',0): 0.075, ('Gypi2',0): 0.075})
        self.probs_densitymx = self.mdl_densitymx.probabilities(self.test_circuit)

    def check_probs(self, probs1, probs2, delta=1e-6):
        for k, v in probs2.items():
            self.assertAlmostEqual(probs1[k], v, delta=delta)

    @unittest.skipIf(_qibo is None, "qibo package not installed so cannot test")
    def test_qibo_circuitsim_statevec(self):
        evo_qibo.densitymx_mode = False
        evo_qibo.nshots = 1000
        mdl_qibo = create_crosstalk_free_model(self.pspec, evotype='qibo', simulator='map',
                                               depolarization_strengths={('Gxpi2',0): 0.075, ('Gypi2',0): 0.075})
        probs = mdl_qibo.probabilities(self.test_circuit)
        self.check_probs(probs, self.probs_densitymx, delta=0.04)  # loose check for 1000 shots

    @unittest.skipIf(_qibo is None, "qibo package not installed so cannot test")
    def test_qibo_circuitsim_densitymx(self):
        evo_qibo.densitymx_mode = True
        mdl_qibo = create_crosstalk_free_model(self.pspec, evotype='qibo', simulator='map',
                                               depolarization_strengths={('Gxpi2',0): 0.075, ('Gypi2',0): 0.075})
        probs = mdl_qibo.probabilities(self.test_circuit)
        self.check_probs(probs, self.probs_densitymx, delta=1e-6)  # tight check (should be ~exact)

    #Note: for FUTURE work - this doesn't work for map fwdsim like the densitymx version below
    # because the qibo effect reps (needed for explicit models) only work for densitymx mode.  These
    # 'matrix' simulator runs but really shouldn't (I think it uses the qibo std-basis matrices?) and
    # gets bogus results, and we should probably at least make sure this errors appropriately.
    #def test_qibo_stdmodel_statevec(self):
    #    pass

    @unittest.skipIf(_qibo is None, "qibo package not installed so cannot test")
    def test_qibo_stdmodel_densitymx(self):
        evo_qibo.densitymx_mode = True
        mdl_std_qibo = std.target_model('static unitary', evotype='qibo', simulator='map')
        probs = mdl_std_qibo.probabilities(self.test_circuit)
        self.assertAlmostEqual(probs['00'], 0.0)
        self.assertAlmostEqual(probs['01'], 0.0)
        self.assertAlmostEqual(probs['10'], 1.0)
        self.assertAlmostEqual(probs['11'], 0.0)

    @unittest.skipIf(_qibo is None, "qibo package not installed so cannot test")
    def test_FullCPTP_parameterization(self):  # maybe move or split this test elsewhere too?
        evo_qibo.densitymx_mode = True
        evo_qibo.minimal_space = 'HilbertSchmidt'  # maybe this should be set automatically?

        # 'full CPTP' or test new '1+(CPTPLND)'
        mdl_densitymx_slow = std1Q.target_model('full CPTP', evotype='densitymx_slow', simulator='map')
        mdl_densitymx = std1Q.target_model('full CPTP', evotype='densitymx', simulator='map')
        mdl_qibo = std1Q.target_model('full CPTP', evotype='qibo', simulator='map')

        c = Circuit("Gxpi2:0", line_labels=(0,))
        probs1 = mdl_densitymx_slow.probabilities(c)
        probs2 = mdl_densitymx.probabilities(c)
        probs3 = mdl_qibo.probabilities(c)
        self.assertAlmostEqual(probs1['0'], 0.5)
        self.assertAlmostEqual(probs1['1'], 0.5)
        self.check_probs(probs1, probs2, delta=1e-6)
        self.check_probs(probs1, probs3, delta=1e-6)

    @unittest.skipIf(_qibo is None, "qibo package not installed so cannot test")
    def test_1plusCPTPLND_parameterization(self):  # maybe move or split this test elsewhere too?
        evo_qibo.densitymx_mode = True
        evo_qibo.minimal_space = 'HilbertSchmidt'  # maybe this should be set automatically?

        mdl_densitymx_slow = std1Q.target_model('1+(CPTPLND)', evotype='densitymx_slow', simulator='map')
        mdl_densitymx = std1Q.target_model('1+(CPTPLND)', evotype='densitymx', simulator='map')
        mdl_qibo = std1Q.target_model('1+(CPTPLND)', evotype='qibo', simulator='map')

        self.assertTrue(isinstance(mdl_densitymx_slow.operations['Gxpi2', 0]._rep.factor_reps[1],
                                   OpRepIdentityPlusErrorgen))
        self.assertTrue(isinstance(mdl_densitymx.operations['Gxpi2', 0]._rep.factor_reps[1],
                                   OpRepDenseSuperop))
        # Note: we haven't mirrored OpRepIdentityPlusErrorgen in densitymx evotype

        c = Circuit("Gxpi2:0", line_labels=(0,))
        probs1 = mdl_densitymx_slow.probabilities(c)
        probs2 = mdl_densitymx.probabilities(c)
        probs3 = mdl_qibo.probabilities(c)
        self.assertAlmostEqual(probs1['0'], 0.5)
        self.assertAlmostEqual(probs1['1'], 0.5)
        self.check_probs(probs1, probs2, delta=1e-6)
        self.check_probs(probs1, probs3, delta=1e-6)
