import unittest
import pygsti
from pygsti.objects import ProcessorSpec
import numpy as np

from ..testutils import BaseTestCase, compare_files, temp_files

class ProcessorSpecCase(BaseTestCase):

    def test_processorspec(self):
        
        # Tests init a pspec using standard gatenames, and all standards.
        n = 3
        gate_names = ['Gh','Gp','Gxpi','Gypi','Gzpi','Gpdag','Gcphase'] # 'Gi',
        ps = ProcessorSpec(n,gate_names=gate_names, construct_models=('target','clifford'))
    
        # Tests init a pspec containing 1 qubit (as special case which could break)
        n = 1
        gate_names = ['Gh','Gp','Gxpi','Gypi','Gzpi','Gpdag','Gcphase'] # 'Gi',
        ps = ProcessorSpec(n,gate_names=gate_names, construct_models=('target','clifford'))
        
        # Note: More complex pspec objects are created for testing clifford compilers.
