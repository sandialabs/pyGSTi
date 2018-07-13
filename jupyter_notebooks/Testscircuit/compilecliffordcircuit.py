import pygsti
from pygsti.objects import Circuit
from pygsti.objects import GateString
from pygsti.baseobjs import Label
from pygsti.objects import ProcessorSpec
from pygsti.tools import symplectic
from pygsti.algorithms import compilecliffordcircuit as ccc
import numpy as np

def test_compilecliffordcircuit():
    
    n = 10
    # Pick a random Clifford to compile
    s, p = symplectic.random_clifford(n)
    # Directly test the core algorithm
    c = ccc.compile_symplectic_with_basic_global_gaussian_elimination(s)
    sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c)
    assert(np.array_equal(s,sout))
    # Test accessing all the allowed algorithms, without a pspec or a subsetQs
    c = ccc.compile_symplectic(s, iterations=3, algorithms=['BGGE','ROGGE'])

    # Tests init a pspec with limited availability, and user-specified labels.
    n = 5
    qubit_labels = ['Q'+str(i) for i in range(n)]
    availability = {'Gcnot':[('Q'+str(i),'Q'+str(i+1)) for i in range(0,n-1)]}
    gate_names = ['Gi','Gh','Gp','Gxpi','Gpdag','Gcnot']
    pspec = ProcessorSpec(n,gate_names=gate_names,availability=availability,qubit_labels=qubit_labels)
    s, p = symplectic.random_clifford(n)
    # Test accessing all the allowed algorithms, with a pspec but no subsetQs
    c = ccc.compile_symplectic(s, pspec=pspec,iterations=3, algorithms=['BGGE','ROGGE'])

    # Test accessing all the allowed algorithms, with a pspec and a subsetQs
    n = 2
    s, p = symplectic.random_clifford(n)
    c = ccc.compile_symplectic(s, pspec=pspec,subsetQs=['Q2','Q3'],iterations=20, algorithms=['BGGE','ROGGE'])
    sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
    assert(np.array_equal(s,sout))

    # Test the main function that we'll access -- compile_clifford 
    n = 5
    s, p = symplectic.random_clifford(n)
    c = ccc.compile_clifford(s, p, pspec=pspec,subsetQs=None,iterations=2, algorithm='ROGGE',
                             prefixpaulis=True, paulirandomize=True)
    sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
    assert(np.array_equal(s,sout))
    c = ccc.compile_clifford(s, p, pspec=None, subsetQs=None,iterations=2, algorithm='ROGGE')
    sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
    assert(np.array_equal(s,sout))

    n = 2
    s, p = symplectic.random_clifford(n)
    c = ccc.compile_clifford(s, p, pspec=pspec,subsetQs=['Q2','Q3'],iterations=2, algorithm='ROGGE',
                             prefixpaulis=False, paulirandomize=True)
    sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
    assert(np.array_equal(s,sout))
    
    # Check it works for the 1-qubit case.
    n = 1
    # Pick a random Clifford to compile
    s, p = symplectic.random_clifford(1)
    c = ccc.compile_clifford(s, p)
    sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c) 
    c = ccc.compile_clifford(s, p, pspec=pspec,subsetQs=['Q3'],iterations=2, algorithm='ROGGE',
                             prefixpaulis=False, paulirandomize=True)
    sout, pout = symplectic.symplectic_rep_of_clifford_circuit(c,pspec=pspec)
    assert(np.array_equal(s,sout))
    assert(np.array_equal(s,sout))