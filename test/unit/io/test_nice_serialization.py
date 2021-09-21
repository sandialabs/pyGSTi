import unittest
from ..util import BaseCase, with_temp_path

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
        self.gst_design = smq1Q_XYI.get_gst_experiment_design(4, qubit_labels=[0])

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
        #TODO: assert correctness

    @with_temp_path
    def test_cloudnoise_model(self, pth):
        mdl_cloud = create_cloud_crosstalk_model(self.pspec_2Q, depolarization_strengths={'Gx': 0.05}, 
                                                 stochastic_error_probs={'Gy': (0.01, 0.02, 0.03)},
                                                 lindblad_error_coeffs={'Gcnot': {('H','ZZ'): 0.07, ('S','XX'): 0.10}},
                                                 independent_gates=False, independent_spam=True, verbosity=2)
        mdl_cloud2 = self.helper_serialize(mdl_cloud, pth)
        #mdl_cloud._print_gpindices()
        #mdl_cloud2._print_gpindices()

        #TODO: assert correctness
