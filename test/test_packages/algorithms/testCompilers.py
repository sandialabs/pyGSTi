import unittest
import pygsti
from pygsti.objects import Circuit
from pygsti.objects import GateString
from pygsti.baseobjs import Label
from pygsti.objects import ProcessorSpec
from pygsti.tools import symplectic
from pygsti.algorithms import compilers
import numpy as np

from ..testutils import BaseTestCase, compare_files, temp_files
from ..algorithms.basecase import AlgorithmsBase

class TestCompilers(AlgorithmsBase):

    def test_compilers(self):
        
        n = 10
        # Pick a random Clifford to compile
        s, p = symplectic.random_clifford(n)
        # Directly test the core algorithm
        c = compilers.compile_symplectic_using_GGE_core(s)
        sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c)
        self.assertArraysEqual(s,sout)
        # Test accessing all the allowed algorithms, without a pspec or a subsetQs
        c = compilers.compile_symplectic(s, iterations=3, algorithms=['BGGE','ROGGE'])
    
        # Tests init a pspec with limited availability, and user-specified labels.
        n = 5
        qubit_labels = ['Q'+str(i) for i in range(n)]
        availability = {'Gcnot':[('Q'+str(i),'Q'+str(i+1)) for i in range(0,n-1)]}
        gate_names = ['Gi','Gh','Gp','Gxpi','Gpdag','Gcnot']
        pspec = ProcessorSpec(n,gate_names=gate_names,availability=availability,qubit_labels=qubit_labels)
        s, p = symplectic.random_clifford(n)
        # Test accessing all the allowed algorithms, with a pspec but no subsetQs
        c = compilers.compile_symplectic(s, pspec=pspec,iterations=3, algorithms=['BGGE','ROGGE'])
    
        # Test accessing all the allowed algorithms, with a pspec and a subsetQs
        n = 2
        s, p = symplectic.random_clifford(n)
        c = compilers.compile_symplectic(s, pspec=pspec,subsetQs=['Q2','Q3'],iterations=2, algorithms=['BGGE','ROGGE','iAGvGE'])
        sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
        self.assertArraysEqual(s,sout)
    
        # Test the main function that we'll access -- compile_clifford 
        n = 5
        s, p = symplectic.random_clifford(n)
        c = compilers.compile_clifford(s, p, pspec=pspec,subsetQs=None,iterations=2, algorithm='ROGGE',
                                 prefixpaulis=True, paulirandomize=True)
        sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
        self.assertArraysEqual(s,sout)
        c = compilers.compile_clifford(s, p, pspec=None, subsetQs=None,iterations=2, algorithm='ROGGE')
        sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
        self.assertArraysEqual(s,sout)
    
        n = 2
        s, p = symplectic.random_clifford(n)
        c = compilers.compile_clifford(s, p, pspec=pspec,subsetQs=['Q2','Q3'],iterations=2, algorithm='ROGGE',
                                 prefixpaulis=True, paulirandomize=True)
        sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
        self.assertArraysEqual(s,sout)
        c = compilers.compile_clifford(s, p, pspec=pspec,subsetQs=['Q2','Q3'],iterations=2, algorithm='BGGE',
                                 prefixpaulis=True, paulirandomize=True)
        sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
        self.assertArraysEqual(s,sout)
        c = compilers.compile_clifford(s, p, pspec=pspec,subsetQs=['Q2','Q3'],iterations=2, algorithm='iAGvGE',
                                 prefixpaulis=True, paulirandomize=False)
        sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
        self.assertArraysEqual(s,sout)
        
        # Check it works for the 1-qubit case.
        n = 1
        # Pick a random Clifford to compile
        s, p = symplectic.random_clifford(1)
        c = compilers.compile_clifford(s, p)
        sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c) 
        c = compilers.compile_clifford(s, p, pspec=pspec,subsetQs=['Q3'],iterations=2, algorithm='ROGGE',
                                 prefixpaulis=False, paulirandomize=True)
        sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
        self.assertArraysEqual(s,sout)
        
        # Tests all CNOT compiler algorithms
        n = 8
        qubit_labels = ['Q'+str(i) for i in range(n)] 
        availability = {'Gcnot':[('Q'+str(i),'Q'+str(i+1)) for i in range(0,n-1)]+[('Q0','Q2'),]}
        gate_names = ['Gi','Gh','Gp','Gxpi','Gpdag','Gcnot']
        pspec8 = ProcessorSpec(n,gate_names=gate_names,availability=availability,qubit_labels=qubit_labels)
        n = 6
        qubit_labels = ['Q'+str(i) for i in range(n)] 
        availability = {'Gcphase':[('Q'+str(i),'Q'+str(i+1)) for i in range(0,n-1)]+[('Q'+str(n-1),'Q'+str(0))]}
        gate_names = ['Gi','Gh','Gxpi2','Gp','Gcphase']
        pspec6 = ProcessorSpec(n,gate_names=gate_names,availability=availability,qubit_labels=qubit_labels)
    
        nsubset = 6
        gatestring = []
        for i in range(100):
            a = np.random.randint(nsubset)
            b = np.random.randint(nsubset)
            if a != b:
                gatestring.append(Label('CNOT',('Q'+str(a),'Q'+str(b))))
    
                subsetQs = ['Q'+str(i) for i in range(nsubset)] 
        circuit = Circuit(gatestring=gatestring, line_labels = subsetQs)
        s, p  = pygsti.tools.symplectic.symplectic_rep_of_clifford_circuit(circuit)
    
        aargs= {}
        aargs['COCAGE'] = []
        aargs['COiCAGE'] = []
        aargs['OCAGE'] = [['Q1', 'Q0','Q2', 'Q5', 'Q3', 'Q4'],]
        # This ordering must be a 'contraction' of the graph, with the remaining graph always connected.
        aargs['OiCAGE'] = [['Q0', 'Q1', 'Q2', 'Q5', 'Q3', 'Q4'],]
        aargs['ROCAGE'] = []
        for algorithm in ['COiCAGE','OiCAGE','COCAGE','ROCAGE']:
            c = compilers.compile_cnot_circuit(s, pspec6, algorithm=algorithm, subsetQs=None, aargs=aargs[algorithm])    
            c = compilers.compile_cnot_circuit(s, pspec8, algorithm=algorithm, subsetQs=subsetQs, aargs=aargs[algorithm])
            
        # Tests stabilizer state and measurement functions.
        
        # Tests the stabilizer compilers for n = 1
        n = 1
        pspec1 = ProcessorSpec(nQubits=n,gate_names=['Gi','Gcnot','Gh','Gp','Gxpi','Gypi','Gzpi'])
        s, p  = symplectic.random_clifford(n)
        c = compilers.compile_stabilizer_state(s,p,pspec1,algorithm='COCAGE',paulirandomize=False)
        c = compilers.compile_stabilizer_measurement(s,p,pspec1,algorithm='ROCAGE',paulirandomize=True)
        c = compilers.compile_stabilizer_measurement(s,p,pspec6,subsetQs=['Q3',], algorithm='COiCAGE',paulirandomize=False)
        
        def check_out_symplectic(c, pspec, s, p, n):
            s0, p0 = symplectic.prep_stabilizer_state(n)
            sc, pc = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
            scout, pcout = symplectic.apply_clifford_to_stabilizer_state(sc,pc,s0,p0)
            stargetout, ptargetout = symplectic.apply_clifford_to_stabilizer_state(s,p,s0,p0)
            for i in range(n):
                mtout = symplectic.pauli_z_measurement(stargetout,ptargetout,i)
                mcout = symplectic.pauli_z_measurement(scout,pcout,i)
                self.assertArraysAlmostEqual(mtout[0],mcout[0])
            
        n = 6
        s, p  = symplectic.random_clifford(n)
        c = compilers.compile_stabilizer_state(s,p,pspec6,algorithm='ROCAGE',paulirandomize=False)
        check_out_symplectic(c, pspec6, s, p, n)
    
        s, p  = symplectic.random_clifford(n)
        c = compilers.compile_stabilizer_state(s,p,pspec6,algorithm='COiCAGE',paulirandomize=True)
        check_out_symplectic(c, pspec6, s, p, n)
    
        s, p  = symplectic.random_clifford(3)
        c = compilers.compile_stabilizer_measurement(s,p,pspec6,subsetQs=['Q3','Q4','Q5'], algorithm='COiCAGE',
                                                     paulirandomize=False)
        sc, pc = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec6)
        # The state the c should map to |0,0,0,....>.
        sstate, pstate = symplectic.prep_stabilizer_state(3)
        sstate, pstate =symplectic.apply_clifford_to_stabilizer_state(s,p,sstate,pstate)
        sout, pout = symplectic.apply_clifford_to_stabilizer_state(sc,pc,sstate,pstate)
        for i in range(3):
            mtout = symplectic.pauli_z_measurement(sout,pout,i)
            self.assertArraysAlmostEqual(mtout[1],0.)
    
        s, p  = symplectic.random_clifford(n)
        c1 = compilers.compile_stabilizer_state(s,p,pspec6,algorithm='COiCAGE',paulirandomize=False)
        c2 = compilers.compile_stabilizer_measurement(s,p,pspec6,algorithm='COiCAGE',paulirandomize=True)
        c2.prefix_circuit(c1)
        zerosstate_s, zerosstate_p = symplectic.prep_stabilizer_state(n)
        sc, pc = symplectic.symplectic_rep_of_clifford_circuit(c2,pspec=pspec6)
        scout, pcout = symplectic.apply_clifford_to_stabilizer_state(sc,pc,zerosstate_s, zerosstate_p)
        for i in range(n):
            mtout = symplectic.pauli_z_measurement(scout,pcout,i)
            self.assertArraysAlmostEqual(mtout[1],0.)

