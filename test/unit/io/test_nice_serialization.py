import unittest
from ..util import BaseCase, with_temp_path
import numpy as np

import pygsti
import pygsti.io as io
from pygsti.modelpacks import smq1Q_XYI

from pygsti.processors import QubitProcessorSpec
from pygsti.models import create_crosstalk_free_model, create_cloud_crosstalk_model


class NiceSerializationTester(BaseCase):

    def helper_serialize(self, obj, temp_pth):
        s = obj.dumps()
        obj2 = obj.__class__.loads(s)

        obj.write(temp_pth + ".json")
        obj_from_file = obj.__class__.read(temp_pth + ".json")
        self.assertTrue(isinstance(obj_from_file, type(obj)))

        return obj2

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
        mdl2 = self.helper_serialize(mdl, pth)
        self.assertTrue(mdl.frobeniusdist(mdl2) < 1e-6)
        self.assertTrue(mdl.is_similar(mdl2))
        self.assertTrue(mdl.is_equivalent(mdl2))

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
        mdl_local2 = self.helper_serialize(mdl_local, pth)
        self.assertTrue(mdl_local.is_similar(mdl_local2))
        self.assertTrue(mdl_local.is_equivalent(mdl_local2))
        #TODO: assert correctness

    @with_temp_path
    def test_cloudnoise_model(self, pth):
        mdl_cloud = create_cloud_crosstalk_model(self.pspec_2Q, depolarization_strengths={'Gx': 0.05},
                                                 stochastic_error_probs={'Gy': (0.01, 0.02, 0.03)},
                                                 lindblad_error_coeffs={'Gcnot': {('H','ZZ'): 0.07, ('S','XX'): 0.10}},
                                                 independent_gates=False, independent_spam=True, verbosity=2)
        mdl_cloud2 = self.helper_serialize(mdl_cloud, pth)
        self.assertTrue(mdl_cloud.is_similar(mdl_cloud2))
        self.assertTrue(mdl_cloud.is_equivalent(mdl_cloud2))


class ModelEquivalenceTester(BaseCase):

    def setUp(self):
        nQubits = 2
        self.pspec_2Q = QubitProcessorSpec(nQubits, ('Gx', 'Gy', 'Gcnot'), geometry="line",
                                           qubit_labels=['qb{}'.format(i) for i in range(nQubits)])

    def check_model(self, mdl):
        mcopy = mdl.copy()
        self.assertFalse(mdl is mcopy)
        self.assertTrue(mcopy.is_similar(mdl))
        self.assertTrue(mcopy.is_equivalent(mdl))

        if mdl.num_params > 0:
            r = np.random.random(mdl.num_params)
            if np.linalg.norm(r) < 1e6:  # just in case we randomly get all zeros!
                r = 0.1 * np.ones(mcopy.num_params)
            v_prime = mdl.to_vector() + r
            mcopy.from_vector(v_prime)
            self.assertTrue(mcopy.is_similar(mdl))
            self.assertFalse(mcopy.is_equivalent(mdl))

    def test_explicit_model_equal(self):
        mdl_explicit = smq1Q_XYI.target_model()
        self.check_model(mdl_explicit)

    def test_local_model_equal(self):
        mdl_local = create_crosstalk_free_model(self.pspec_2Q,
                                                ideal_gate_type='H+S', ideal_spam_type='tensor product H+S',
                                                independent_gates=False,
                                                ensure_composed_gates=False)
        self.check_model(mdl_local)

        mdl_local = create_crosstalk_free_model(self.pspec_2Q,
                                                ideal_gate_type='H+S', ideal_spam_type='computational',
                                                independent_gates=True,
                                                ensure_composed_gates=True)
        self.check_model(mdl_local)

    def test_cloud_model_equal(self):
        mdl_cloud = create_cloud_crosstalk_model(self.pspec_2Q, depolarization_strengths={'Gx': 0.05},
                                                 stochastic_error_probs={'Gy': (0.01, 0.02, 0.03)},
                                                 lindblad_error_coeffs={'Gcnot': {('H','ZZ'): 0.07, ('S','XX'): 0.10}},
                                                 independent_gates=False, independent_spam=True, verbosity=2)
        self.check_model(mdl_cloud)

        mdl_cloud = create_cloud_crosstalk_model(self.pspec_2Q, depolarization_strengths={'Gx': 0.05},
                                                 stochastic_error_probs={'Gy': (0.01, 0.02, 0.03)},
                                                 lindblad_error_coeffs={'Gcnot': {('H','ZZ'): 0.07, ('S','XX'): 0.10}},
                                                 independent_gates=True, independent_spam=True, verbosity=2)
        self.check_model(mdl_cloud)


