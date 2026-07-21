import unittest

import numpy as np
import scipy

from pygsti.processors import QubitProcessorSpec
from pygsti.models import modelconstruction as mc
from pygsti.circuits import Circuit
from ..util import BaseCase, with_temp_path


def save_and_load(obj, pth):
    obj.write(pth + ".json")
    return QubitProcessorSpec.read(pth + '.json')


class ProcessorSpecTester(BaseCase):
    @unittest.skip("REMOVEME")
    def test_construct_with_nonstd_gate_unitary_factory(self):
        nQubits = 2

        def fn(args):
            if args is None: args = (0,)
            a, = args
            sigmaZ = np.array([[1, 0], [0, -1]], 'd')
            return scipy.linalg.expm(1j * float(a) * sigmaZ)

        ps = QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot', 'Ga'), nonstd_gate_unitaries={'Ga': fn})
        mdl = mc.create_crosstalk_free_model(ps)

        c = Circuit("Gx:1Ga;0.3:1Gx:1@(0,1)")
        p = mdl.probabilities(c)

        self.assertAlmostEqual(p['00'], 0.08733219254516078)
        self.assertAlmostEqual(p['01'], 0.9126678074548386)

        c2 = Circuit("Gx:1Ga;0.78539816:1Gx:1@(0,1)")  # a clifford: 0.78539816 = pi/4
        p2 = mdl.probabilities(c2)
        self.assertAlmostEqual(p2['00'], 0.5)
        self.assertAlmostEqual(p2['01'], 0.5)

    @with_temp_path
    def test_instrument_with_sslbls_serialization(self, pth):
        # Regression test: an instrument whose name carries explicit state-space labels
        # (as opposed to a bare name like 'Iz' that implicitly acts on all qudits) serializes
        # its Label to a JSON list. On load, QubitProcessorSpec._from_nice_serialization must
        # turn each entry of `instrument_names` back into a hashable object; if it doesn't, the
        # unhashable list survives into `self.instrument_names` and any lookup against
        # `self.nonstd_instruments` (e.g. via `instrument_specifier`, as called from
        # `_create_explicit_model`) raises `TypeError: unhashable type: 'list'`.
        iname = Label('Iz', (0,))
        pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2'], geometry='line',
                                   instrument_names=(iname,),
                                   nonstd_instruments={iname: 'Iz'})

        loaded_pspec = save_and_load(pspec, pth)

        # Compound names come back as plain tuples (hash-equal to the original Labels).
        self.assertEqual(loaded_pspec.instrument_names, (('Iz', 0),))
        for loaded_iname in loaded_pspec.instrument_names:
            self.assertEqual(loaded_pspec.instrument_specifier(loaded_iname), 'Iz')

        # Exercise the reported call chain (_create_explicit_model) on a single-qubit spec,
        # where the instrument's sslbls cover the full state space.  On the two-qubit spec
        # above this raises NotImplementedError even without serialization, because
        # instruments cannot be embedded onto a subset of the qudits yet.
        pspec_1q = QubitProcessorSpec(1, ['Gxpi2', 'Gypi2'],
                                      instrument_names=(iname,),
                                      nonstd_instruments={iname: 'Iz'})
        loaded_pspec_1q = save_and_load(pspec_1q, pth)

        from pygsti.models.modelconstruction import _create_explicit_model
        mdl = _create_explicit_model(loaded_pspec_1q, None, evotype='default', simulator='auto',
                                     ideal_gate_type='static', ideal_prep_type='auto', ideal_povm_type='auto',
                                     embed_gates=False, basis='pp')
        self.assertEqual(list(mdl.instruments.keys()), [iname])

    @with_temp_path
    def test_instrument_with_custom_spec_serialization(self, pth):
        # Regression test: `nonstd_instruments` keys used to be flattened with a lossy colon-join,
        # which exploded plain-string names character-by-character ('Iparity' -> 'I:p:a:r:i:t:y'),
        # so a custom instrument spec could never be found by `instrument_specifier` after a
        # round trip through JSON.
        spec = {'plus': [('00', '00'), ('11', '11')],
                'minus': [('10', '10'), ('01', '01')]}
        pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2'], geometry='line',
                                   instrument_names=('Iparity',),
                                   nonstd_instruments={'Iparity': spec})

        loaded_pspec = save_and_load(pspec, pth)

        self.assertEqual(loaded_pspec.instrument_names, ('Iparity',))
        self.assertEqual(loaded_pspec.instrument_specifier('Iparity'), spec)

        from pygsti.models.modelconstruction import create_explicit_model
        mdl = create_explicit_model(loaded_pspec)
        self.assertEqual(list(mdl.instruments['Iparity'].keys()), ['plus', 'minus'])

    @with_temp_path
    def test_qudit_instrument_serialization(self, pth):
        # Regression test: QuditProcessorSpec._from_nice_serialization applied tuple() to every
        # loaded instrument name, exploding plain-string names into character tuples
        # ('Iz' -> ('I', 'z')).
        iname = Label('Iz', ('Q0',))
        pspec = QuditProcessorSpec(('Q0', 'Q1'), (2, 2), ['Gxpi2', 'Gypi2'], geometry='line',
                                   instrument_names=('Iz', iname),
                                   nonstd_instruments={iname: 'Iz'})

        loaded_pspec = save_and_load(pspec, pth)

        self.assertEqual(loaded_pspec.instrument_names, ('Iz', ('Iz', 'Q0')))
        self.assertEqual(loaded_pspec.instrument_specifier('Iz'), 'Iz')
        self.assertEqual(loaded_pspec.instrument_specifier(('Iz', 'Q0')), 'Iz')

    @with_temp_path
    def test_legacy_instrument_serialization_format(self, pth):
        # Files written before the `nonstd_instruments` format change stored a dict whose keys
        # were flattened with ':'.join(map(str, key)).  Loading must repair those keys by
        # matching them against `instrument_names`, falling back to a split on ':' for keys it
        # cannot match.
        import json

        iname = Label('Iz', (0,))
        parity_spec = {'plus': [('00', '00'), ('11', '11')],
                       'minus': [('10', '10'), ('01', '01')]}
        pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2'], geometry='line',
                                   instrument_names=('Iparity', iname),
                                   nonstd_instruments={'Iparity': parity_spec, iname: 'Iz'})

        pspec.write(pth + '.json')
        with open(pth + '.json') as f:
            state = json.load(f)
        state['nonstd_instruments'] = {':'.join(map(str, k)): v for k, v in state['nonstd_instruments']}
        state['nonstd_instruments']['Ighost:2'] = 'Iz'  # matches no instrument name
        with open(pth + '.json', 'w') as f:
            json.dump(state, f)

        loaded_pspec = QubitProcessorSpec.read(pth + '.json')

        self.assertEqual(loaded_pspec.instrument_names, ('Iparity', ('Iz', 0)))
        self.assertEqual(loaded_pspec.instrument_specifier('Iparity'), parity_spec)
        self.assertEqual(loaded_pspec.instrument_specifier(('Iz', 0)), 'Iz')
        # Unmatched legacy keys keep the old best-effort split-on-colon reconstruction.
        self.assertEqual(loaded_pspec.nonstd_instruments[('Ighost', '2')], 'Iz')

    @with_temp_path
    def test_with_spam(self, pth):
        pspec_defaults = QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line')

        pspec_names = QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line',
                                         prep_names=("rho1", "rho_1100"), povm_names=("Mz",))

        prep_vec = np.zeros(2**4, complex)
        prep_vec[4] = 1.0
        EA = np.zeros(2**4, complex)
        EA[14] = 1.0
        EB = np.zeros(2**4, complex)
        EB[15] = 1.0

        pspec_vecs = QubitProcessorSpec(4, ['Gxpi2', 'Gypi2'], geometry='line',
                                        prep_names=("rhoA", "rhoC"), povm_names=("Ma", "Mc"),
                                        nonstd_preps={'rhoA': "rho0", 'rhoC': prep_vec},
                                        nonstd_povms={'Ma': {'0': "0000", '1': EA},
                                                      'Mc': {'OutA': "0000", 'OutB': [EA, EB]}})

        pspec_defaults = save_and_load(pspec_defaults, pth)
        pspec_names = save_and_load(pspec_names, pth)
        pspec_vecs = save_and_load(pspec_vecs, pth)
