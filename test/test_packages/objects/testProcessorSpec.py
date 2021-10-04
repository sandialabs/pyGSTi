from pygsti.processors import QubitProcessorSpec

from ..testutils import BaseTestCase


class ProcessorSpecCase(BaseTestCase):

    def test_processorspec(self):

        # Tests init a pspec using standard gatenames, and all standards.
        n = 3
        gate_names = ['Gh','Gp','Gxpi','Gypi','Gzpi','Gpdag','Gcphase', 'Gi']
        ps = QubitProcessorSpec(n,gate_names=gate_names, geometry='line')

        # Tests init a pspec containing 1 qubit (as special case which could break)
        n = 1
        gate_names = ['Gh','Gp','Gxpi','Gypi','Gzpi','Gpdag','Gcphase', 'Gi']
        ps = QubitProcessorSpec(n,gate_names=gate_names)  # no geometry needed for 1-qubit specs

