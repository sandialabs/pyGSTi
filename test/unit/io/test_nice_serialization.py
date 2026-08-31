import unittest
from ..util import BaseCase, with_temp_path
import numpy as np

import pygsti
import pygsti.io as io
from pygsti.modelpacks import smq1Q_XYI

from pygsti.processors import QubitProcessorSpec
from pygsti.models import create_crosstalk_free_model, create_cloud_crosstalk_model
from pygsti.baseobjs.label import Label
from pygsti.modelmembers.operations import StaticArbitraryOp


class NiceSerializationTester(BaseCase):

    def helper_serialize(self, obj, temp_pth):
        s = obj.dumps()
        obj2 = obj.__class__.loads(s)

        obj.write(temp_pth + ".json")
        obj_from_file = obj.__class__.read(temp_pth + ".json")
        self.assertTrue(isinstance(obj_from_file, type(obj)))

        return obj2, obj_from_file

    def setUp(self):
        self.gst_design = smq1Q_XYI.create_gst_experiment_design(4, qubit_labels=[0])

        nQubits = 2
        self.pspec_2Q = QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
                                           qubit_labels=['qb{}'.format(i) for i in range(nQubits)])

    @with_temp_path
    def test_processor_spec(self, pth):
        pspec = pygsti.processors.QubitProcessorSpec(4, ('Gxpi2', 'Gypi2', 'Gcnot'), geometry='line')
        self.helper_serialize(pspec, pth)

    @with_temp_path
    def test_explicit_model(self, pth):
        mdl = smq1Q_XYI.target_model()
        mdl2, mdl_file = self.helper_serialize(mdl, pth)
        self.assertTrue(mdl.frobeniusdist(mdl2) < 1e-6)
        self.assertTrue(mdl.is_similar(mdl2))
        self.assertTrue(mdl.is_equivalent(mdl2))
        self.assertTrue(mdl.is_similar(mdl_file))

    @with_temp_path
    def test_circuit_list(self, pth):
        circuit_plaq = self.gst_design.circuit_lists[0]
        self.helper_serialize(circuit_plaq, pth)

    @with_temp_path
    def test_localnoise_model(self, pth):
        mdl_local = create_crosstalk_free_model(self.pspec_2Q,
                                                ideal_gate_type='H+S', ideal_spam_type='tensor product H+S',
                                                independent_gates=False,
                                                ensure_composed_gates=False)
        mdl_local2, mdl_from_file = self.helper_serialize(mdl_local, pth)
        self.assertTrue(mdl_local.is_similar(mdl_local2))
        self.assertTrue(mdl_local.is_equivalent(mdl_local2))
        self.assertTrue(mdl_local.is_similar(mdl_from_file))
        self.assertTrue(mdl_local.is_equivalent(mdl_from_file))

        # Member lookup and prefix validation on both deserialized models
        for m in (mdl_local2, mdl_from_file):
            self.assertIn('rho0', m.prep_blks['layers'])
            self.assertIn('Mdefault', m.povm_blks['layers'])
            self.assertIn('Gx', m.operation_blks['gates'])
            self.assertIn(('Gx', 'qb0'), m.operation_blks['layers'])

            self.assertIs(m['rho0'], m.prep_blks['layers']['rho0'])
            self.assertIs(m['Mdefault'], m.povm_blks['layers']['Mdefault'])
            self.assertIs(m['Gx'], m.operation_blks['gates']['Gx'])
            self.assertIs(m[('Gx', 'qb0')], m.operation_blks['layers'][('Gx', 'qb0')])
            self.assertIs(m['Gx:qb0'], m.operation_blks['layers'][('Gx', 'qb0')])
            self.assertIs(m[Label(('Gx', 'qb0'))], m.operation_blks['layers'][('Gx', 'qb0')])

            # Reconstructed dictionaries enforce member prefix policies
            op = m['Gx'].copy()
            prep = m['rho0'].copy()
            povm = m['Mdefault'].copy()

            with self.assertRaises(KeyError):
                m.prep_blks['layers']['Gx'] = prep
            with self.assertRaises(KeyError):
                m.povm_blks['layers']['rho0'] = povm
            with self.assertRaises(KeyError):
                m.operation_blks['gates']['rho0'] = op
            with self.assertRaises(KeyError):
                m.operation_blks['layers']['Mdefault'] = op
            with self.assertRaises(KeyError):
                m.instrument_blks['layers']['Gx'] = op
            with self.assertRaises(KeyError):
                m.factories['gates']['rho0'] = op
            with self.assertRaises(KeyError):
                m.factories['layers']['rho0'] = op

            # Valid insertions succeed
            m.prep_blks['layers']['rho1'] = prep
            m.povm_blks['layers']['M1'] = povm
            m.operation_blks['gates']['Gz'] = op
            m.operation_blks['gates']['{custom_gate}'] = op
            self.assertIs(m['rho1'], m.prep_blks['layers']['rho1'])
            self.assertIs(m['Gz'], m.operation_blks['gates']['Gz'])
            self.assertIs(m['{custom_gate}'], m.operation_blks['gates']['{custom_gate}'])

    @with_temp_path
    def test_localnoise_model_with_global_idle(self, pth):
        nQubits = 2
        noisy_idle = StaticArbitraryOp(
            np.array([[1, 0, 0, 0],
                      [0, 0.9, 0, 0],
                      [0, 0, 0.9, 0],
                      [0, 0, 0, 0.9]], 'd')
        )
        qubit_labels = ['qb{}'.format(i) for i in range(nQubits)]
        pspec = QubitProcessorSpec(
            nQubits, ('Gx', 'Gy', 'Gcnot', 'Gidle'), geometry="line",
            availability={'Gidle': [('qb0',), ('qb1',)]},
            qubit_labels=qubit_labels
        )

        mdl_idle = create_crosstalk_free_model(
            pspec, {'Gidle': noisy_idle}, ideal_gate_type='static',
            independent_gates=False, ensure_composed_gates=False,
            implicit_idle_mode='add_global'
        )
        self.assertIn('{auto_global_idle}', mdl_idle.operation_blks['layers'])

        mdl_idle2, mdl_idle_file = self.helper_serialize(mdl_idle, pth)
        self.assertTrue(mdl_idle.is_similar(mdl_idle2))
        self.assertTrue(mdl_idle.is_equivalent(mdl_idle2))
        self.assertTrue(mdl_idle.is_similar(mdl_idle_file))
        self.assertTrue(mdl_idle.is_equivalent(mdl_idle_file))

        for m in (mdl_idle2, mdl_idle_file):
            # Reserved brace label was preserved through item-based deserialization
            self.assertIn('{auto_global_idle}', m.operation_blks['layers'])
            self.assertIs(m['{auto_global_idle}'], m.operation_blks['layers']['{auto_global_idle}'])
            self.assertIs(m[Label('{auto_global_idle}')],
                          m.operation_blks['layers']['{auto_global_idle}'])

            # Multi-prefix policy survives deserialization with reserved brace labels
            op = m.operation_blks['layers']['{auto_global_idle}'].copy()
            with self.assertRaises(KeyError):
                m.operation_blks['layers']['rho0'] = op
            with self.assertRaises(KeyError):
                m.operation_blks['layers']['M0'] = op

            m.operation_blks['layers']['{custom_idle}'] = op
            self.assertIs(m['{custom_idle}'], m.operation_blks['layers']['{custom_idle}'])

    @with_temp_path
    def test_cloudnoise_model(self, pth):
        mdl_cloud = create_cloud_crosstalk_model(self.pspec_2Q, depolarization_strengths={'Gx': 0.05},
                                                 stochastic_error_probs={'Gy': (0.01, 0.02, 0.03)},
                                                 lindblad_error_coeffs={'Gcnot': {('H','ZZ'): 0.07, ('S','XX'): 0.10}},
                                                 independent_gates=False, independent_spam=True, verbosity=2)
        mdl_cloud2, mdl_cloud_file = self.helper_serialize(mdl_cloud, pth)
        self.assertTrue(mdl_cloud.is_similar(mdl_cloud2))
        self.assertTrue(mdl_cloud.is_equivalent(mdl_cloud2))


