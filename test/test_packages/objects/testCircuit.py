import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import unittest
import pygsti
from pygsti.objects import Circuit
from pygsti.baseobjs import Label
from pygsti.objects import ProcessorSpec
import numpy as np

from ..testutils import BaseTestCase, compare_files, temp_files

class CircuitTestCase(BaseTestCase):

    def test_circuit(self):
    
        # Test initializing a circuit from an empty circuit.
        c = pygsti.obj.Circuit(num_lines=5)
        self.assertEqual(c.depth(), 0)
        self.assertEqual(c.size(), 0)
        self.assertEqual(c.number_of_lines(), 5)
        self.assertEqual(c.line_labels, tuple(range(5)))
    
        c = pygsti.obj.Circuit(layer_labels=[],num_lines=5)
        self.assertEqual(c.depth(), 0)
        self.assertEqual(c.size(), 0)
        self.assertEqual(c.number_of_lines(), 5)
        self.assertEqual(c.line_labels, tuple(range(5)))
    
        # Test initializing a circuit from a non-empty circuit that is a list
        # containing Label objects. Also test that it can have non-integer line_labels
        # and a different identity identifier.
        circuit=[Label('Gi','Q0'),Label('Gp','Q8')]
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1','Q8','Q12'])
        # Not parallelized by default, so will be depth 2.
        self.assertEqual(c.depth(), 2)
        self.assertEqual(c.size(), 2)
        self.assertEqual(c.number_of_lines(), 4)
        self.assertEqual(c.line_labels, ('Q0','Q1','Q8','Q12'))
    
        # Do again with parallelization
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1','Q8']) #,parallelize=True)
        c = c.parallelize()
        self.assertEqual(c.depth(), 1)
        self.assertEqual(c.size(), 2)
    
        # Now repeat the read-in with no parallelize, but a list of lists of oplabels
        circuit=[[Label('Gi','Q0'),Label('Gp','Q8')],[Label('Gh','Q1'),Label('Gp','Q12')]]
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1','Q8','Q12'])
        self.assertLess(0, c.depth())
    
        # Check we can read-in a circuit that has no qubit labels: enforces them to be on
        # all of the lines.
        circuit = Circuit( None, stringrep="Gx^2GyGxGi" )
        c = pygsti.obj.Circuit(layer_labels=circuit,num_lines=1)
        self.assertEqual(c.depth(), 5)
    
        # Check that we can create a circuit from a string and that we end up with
        # the correctly structured circuit.
        circuit = Circuit( None, stringrep="[Gx:Q0Gy:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1" )
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1'],editable=True)
        c2 = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1'],editable=False)
        self.assertEqual(c.tup,c2.tup)
        self.assertEqual(c.depth(), 5)
        self.assertEqual(c.size(), 8)
    
        # Check we can init from another circuit
        cnew = pygsti.obj.Circuit(c)
        self.assertEqual(cnew, c)
                
        # Test copy() and clear()
        ccopy = c.copy(editable=True)
        ccopy.clear()
        self.assertEqual(ccopy.size(), 0)
        self.assertEqual(c.size(), 8)
    
        # Test inserting a gate when the relevant qubits aren't 
        # idling at that layer
        c.insert_labels_into_layers([Label('Gx','Q0')],2)
        self.assertEqual(c.size(), 9)
        self.assertEqual(c.depth(), 6)
        self.assertEqual(c[2,'Q0'], Label('Gx','Q0'))
    
        # Test inserting a gate when the relevant qubits are 
        # idling at that layer -- depth shouldn't increase
        c[2,'Q1'] = Label('Gx','Q1')
        self.assertEqual(c.size(), 10)
        self.assertEqual(c.depth(), 6)
        self.assertEqual(c[2,'Q1'], Label('Gx','Q1'))
    
        # Test layer insertion
        layer = [Label('Gx','Q1'),]
        c.insert_layer(layer ,1)
        self.assertEqual(c.size(), 11)
        self.assertEqual(c.depth(), 7)
        self.assertEqual(c[1], Label('Gx','Q1'))
        c.insert_layer([] ,1)
        self.assertTrue(len(c[1,('Q0','Q1')].components) == 0) # c.lines_are_idle_at_layer(['Q0','Q1'],1)
        self.assertTrue(len(c[1,'Q0'].components) == 0) #c.lines_are_idle_at_layer(['Q0'],2))
        self.assertFalse(len(c[2,'Q1'].components) == 0) #c.lines_are_idle_at_layer(['Q1'],2))
        self.assertFalse(c.is_line_idling('Q1')) #c.is_idling_qubit('Q1'))
        c.append_idling_lines(['Q3'])
        self.assertFalse(c.is_line_idling('Q0'))
        self.assertFalse(c.is_line_idling('Q1'))
        self.assertTrue(c.is_line_idling('Q3'))
    
        # Test replacing a layer with a layer.
        c = pygsti.obj.Circuit(layer_labels=circuit, line_labels=['Q0','Q1'],editable=True)
        newlayer = [Label('Gx','Q0')]
        c[1] = newlayer #c.replace_layer_with_layer(newlayer,1)
        self.assertEqual(c.depth(), 5)
    
        # Test replacing a layer with a circuit
        c.replace_layer_with_circuit(c.copy(),1)
        self.assertEqual(c.depth(), 2*5 - 1)
    
        # Test layer deletion
        ccopy = c.copy()
        ccopy.insert_layer(layer ,1)
        ccopy.delete_layers([1])
        self.assertEqual(c, ccopy)
    
        # Test inserting a circuit when they are over the same labels.
        circuit = Circuit( None, stringrep="[Gx:Q0Gy:Q1][Gy:Q0Gx:Q1]Gx:Q0Giz:Q1" )
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1'],editable=True)
        ccopy = c.copy()
        c.insert_circuit(ccopy,2)
        self.assertTrue(Label('Gx','Q0') in c[2].components) #c.get_layer(2)
    
        # Test insert a circuit that is over *more* qubits but which has the additional
        # lines idling.
        c1 = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1'],editable=True)
        c2 = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1','Q2','Q3'])
        c1.insert_circuit(c2,0)
        self.assertEqual(c1.line_labels, ('Q0','Q1'))
        self.assertEqual(c1.number_of_lines(), 2)
    
        # Test inserting a circuit that is on *less* qubits.
        c1 = pygsti.obj.Circuit(layer_labels=circuit, line_labels=['Q0','Q1'],editable=True)
        c2 = pygsti.obj.Circuit(layer_labels=[Label('Gx','Q0')], line_labels=['Q0',])
        c1.insert_circuit(c2,1)
        self.assertEqual(c1.line_labels, ('Q0','Q1'))
        self.assertEqual(c1.number_of_lines(), 2)
    
        # Test appending and prefixing a circuit
        c1 = pygsti.obj.Circuit(layer_labels=circuit, line_labels=['Q0','Q1'],editable=True)
        c2 = pygsti.obj.Circuit(layer_labels=[Label('Gx','Q0')], line_labels=['Q0',],editable=True)
        c1.append_circuit(c2)
        c1.prefix_circuit(c2)
        
        # Test tensoring circuits of same length
        gatestring1 = Circuit( None, stringrep="[Gx:Q0Gy:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1" )
        gatestring2 = Circuit( None, stringrep="[Gx:Q2Gy:Q3]^2[Gy:Q2Gx:Q3]Gi:Q2Gi:Q3" )
        c1 = pygsti.obj.Circuit(layer_labels=gatestring1,line_labels=['Q0','Q1'], editable=True)
        c2 = pygsti.obj.Circuit(layer_labels=gatestring2,line_labels=['Q2','Q3'])
        c1.tensor_circuit(c2)
        self.assertEqual(c1.depth(), max(c1.depth(),c2.depth()))
        self.assertEqual(c1[:,'Q2'], c2[:,'Q2'])
    
        # Test tensoring circuits where the inserted circuit is shorter
        gatestring1 = Circuit( None, stringrep="[Gx:Q0Gy:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1Gy:Q0" )
        gatestring2 = Circuit( None, stringrep="[Gx:Q2Gy:Q3]^2[Gy:Q2Gx:Q3]Gi:Q2Gi:Q3" )
        c1 = pygsti.obj.Circuit(layer_labels=gatestring1,line_labels=['Q0','Q1'],editable=True)
        c2 = pygsti.obj.Circuit(layer_labels=gatestring2,line_labels=['Q2','Q3'])
        c1.tensor_circuit(c2,line_order=['Q1','Q3','Q0','Q2'])
        self.assertEqual(c1.depth(), max(c1.depth(),c2.depth()))
    
        # Test tensoring circuits where the inserted circuit is longer
        gatestring1 = Circuit( None, stringrep="[Gx:Q0Gy:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1" )
        gatestring2 = Circuit( None, stringrep="[Gx:Q2Gy:Q3]^2[Gy:Q2Gx:Q3]Gi:Q2Gi:Q3Gy:Q2" )
        c1 = pygsti.obj.Circuit(layer_labels=gatestring1,line_labels=['Q0','Q1'],editable=True)
        c2 = pygsti.obj.Circuit(layer_labels=gatestring2,line_labels=['Q2','Q3'])
        c1.tensor_circuit(c2)
        self.assertEqual(c1.depth(), max(c1.depth(),c2.depth()))
    
        # Test changing a gate name
        circuit = Circuit( None, stringrep="[Gx:Q0Gy:Q1][Gy:Q0Gx:Q1]Gx:Q0Gi:Q1")
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1'],editable=True)
        c.replace_gatename_inplace('Gx','Gz')
        circuit = Circuit( None, stringrep="[Gz:Q0Gy:Q1][Gy:Q0Gz:Q1]Gz:Q0Gi:Q1" )
        c2 = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1'])
        self.assertEqual(c, c2)
    
        # Change gate library using an ordinary dict with every gate as a key. (we test
        # changing gate library using a CompilationLibrary elsewhere in the tests).
        comp = {}
        comp[Label('Gz','Q0')] = pygsti.obj.Circuit(layer_labels=[Label('Gx','Q0')],line_labels=['Q0'])
        comp[Label('Gy','Q0')] = pygsti.obj.Circuit(layer_labels=[Label('Gx','Q0')],line_labels=['Q0'])
        comp[Label('Gz','Q1')] = pygsti.obj.Circuit(layer_labels=[Label('Gx','Q1')],line_labels=['Q1'])
        comp[Label('Gy','Q1')] = pygsti.obj.Circuit(layer_labels=[Label('Gx','Q1')],line_labels=['Q1'])
        comp[Label('Gi','Q1')] = pygsti.obj.Circuit(layer_labels=[Label('Gi','Q1')],line_labels=['Q1'])
        c.change_gate_library(comp)
        self.assertTrue(Label('Gx','Q0') in c[0].components)
    
        # Change gate library using a dict with some gates missing
        comp = {}
        comp[Label('Gz','Q0')] = pygsti.obj.Circuit(layer_labels=[Label('Gx','Q0')],line_labels=['Q0'])
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1'],editable=True)
        c.change_gate_library(comp,allow_unchanged_gates=True)
        self.assertTrue(Label('Gx','Q0') in c[0].components) # c.get_layer(0)
        self.assertTrue(Label('Gy','Q1') in c[0].components)
    
        # Test we can change the labels of the lines.
        c.map_state_space_labels_inplace({'Q0':0,'Q1':1})
        self.assertEqual(c.line_labels, (0,1))
        self.assertEqual(c[0,0].qubits[0], 0)
    
        # Check we can re-order wires
        c.reorder_lines([1,0])
        self.assertEqual(c.line_labels, (1,0))
        # Can't use .get_line as that takes the line label as i_nput.
        #N/A: self.assertEqual(c.line_items[0][0].qubits[0], 1)
    
        # Test deleting and inserting idling wires.
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1'],editable=True)
        c.append_idling_lines(['Q2'])
        self.assertEqual(c.line_labels, ('Q0','Q1','Q2'))
        self.assertEqual(c.number_of_lines(), 3)
        c.delete_idling_lines()
        self.assertEqual(c.line_labels, ('Q0','Q1'))
        self.assertEqual(c.number_of_lines(), 2)
    
        # Test circuit reverse.
        op1 = c[0,'Q0']
        c.reverse()
        op2 = c[-1,'Q0']
        self.assertEqual(op1, op2)
    
        # Test 2-qubit and multi-qubit gate count
        self.assertEqual(c.twoQgate_count(), 0)
        circuit = Circuit( None, stringrep="[Gcnot:Q0:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1" )
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1'])
        self.assertEqual(c.twoQgate_count(), 2)
        self.assertEqual(c.multiQgate_count(), 2)
        circuit = Circuit( None, stringrep="[Gccnot:Q0:Q1:Q2]^2[Gccnot:Q0:Q1]Gi:Q0Gi:Q1" )
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1','Q2'])
        self.assertEqual(c.twoQgate_count(), 1)
        self.assertEqual(c.multiQgate_count(), 3)
    
        # Test the error-probability prediction method
        circuit = Circuit( None, stringrep="[Gx:Q0][Gi:Q0Gi:Q1]")
        c = pygsti.obj.Circuit(layer_labels=circuit,line_labels=['Q0','Q1'])
        infidelity_dict = {}
        infidelity_dict[Label('Gi','Q0')] = 0.7
        infidelity_dict[Label('Gi','Q1')] = 0.9
        infidelity_dict[Label('Gx','Q0')] = 0.8
        infidelity_dict[Label('Gx','Q2')] = 0.9
        #FIX epsilon = c.predicted_error_probability(infidelity_dict)
        #FIX self.assertLess(abs(epsilon - (1 - (1-0.7)*(1-0.8)*(1-0.9)**2)), 10**-10)
    
        # Check we can succesfully create a circuit string.
        s = c.__str__()
    
        # Check we can write to a Qcircuit file.
        #FIX c.write_Qcircuit_tex(temp_files + '/test_qcircuit.tex')
    
        # Test depth compression both with and without 1-qubit gate compression
        ls = [Label('H',1),Label('P',1),Label('P',1),Label(()),Label('CNOT',(2,3))]
        ls += [Label('HP',1),Label('PH',1),Label('CNOT',(1,2))]
        ls += [Label(()),Label(()),Label('CNOT',(1,2))]
        circuit = Circuit(ls)
        c = pygsti.obj.Circuit(layer_labels=circuit, num_lines=4, editable=True)
        c.compress_depth(verbosity=0)
        self.assertEqual(c.depth(), 7)
        # Get a dictionary that relates H, P gates etc.
        oneQrelations = pygsti.symplectic.oneQclifford_symplectic_group_relations()
        c.compress_depth(oneQgate_relations = oneQrelations)
        self.assertEqual(c.depth(), 3)

        #NEEDED? (has been removed)
        # Test the is_valid_circuit checker.
        #c.is_valid_circuit()
        #with self.assertRaises(AssertionError):
        #    c.line_items[0][2] = Label('CNOT',(2,3))
        #    c.is_valid_circuit()
        #    fail = False
        
        # Check that convert_to_quil runs, doesn't check the output makes sense.
        circuit = [Label(('Gi','Q1')),Label(('Gxpi','Q1')),Label('Gcnot',('Q1','Q2'))]
        c = Circuit(layer_labels=circuit,line_labels=['Q1','Q2'])
        #FIX s = c.convert_to_quil()
    
        # Check done_editing makes the circuit static.
        c.done_editing()
        with self.assertRaises(AssertionError):
            c.clear()
        
        # Create a pspec, to test the circuit simulator.
        n = 4
        qubit_labels = ['Q'+str(i) for i in range(n)]
        availability = {'Gcnot':[('Q'+str(i),'Q'+str(i+1)) for i in range(0,n-1)]}
        gate_names = ['Gh','Gp','Gxpi','Gpdag','Gcnot'] # 'Gi',
        ps = ProcessorSpec(n,gate_names=gate_names,qubit_labels=qubit_labels)
    
        # Tests the circuit simulator
        c = Circuit(layer_labels=[Label('Gh','Q0'),Label('Gcnot',('Q0','Q1'))],line_labels=['Q0','Q1'])
        out = c.simulate(ps.models['target'])
        self.assertLess(abs(out['00'] - 0.5), 10**-10)
        self.assertLess(abs(out['11'] - 0.5), 10**-10)

    def test_create_circuitlabel(self):
        # test automatic creation of "power" subcircuits when needed
        Gi = pygsti.obj.Circuit(None, stringrep='Gi(Gx)^256000', editable=True, expand_subcircuits=False)
        self.assertTrue(isinstance(Gi.tup[1], pygsti.baseobjs.CircuitLabel))

        cl = Gi.tup[1]
        self.assertEqual(str(cl), "Gx^256000")
        self.assertEqual(cl.components, ('Gx',) )
        self.assertEqual(cl.reps, 256000)
        self.assertEqual(Gi.tup, ('Gi', pygsti.baseobjs.CircuitLabel(name='', tupOfLayers=('Gx',),
                                                                     stateSpaceLabels=None, reps=256000)))

    def test_expand_and_factorize_circuitlabel(self):
        c = pygsti.obj.Circuit(None, stringrep='Gi(Gx:1)^2',num_lines=3, editable=True, expand_subcircuits=False)
        c[1,0] = "Gx"
        self.assertEqual(c, ('Gi', (pygsti.baseobjs.CircuitLabel('',[('Gx',1)],(1,),2), ('Gx',0))) )

        c.expand_subcircuits()
        self.assertEqual(c, ('Gi', (('Gx',0),('Gx',1)), ('Gx',1)) )

        c2 = pygsti.obj.Circuit(None, stringrep='GiGxGxGxGxGy', editable=True)
        self.assertEqual(c2, ('Gi','Gx','Gx','Gx','Gx','Gy'))
        
        c2.factorize_repetitions()
        self.assertEqual(c2, ('Gi',pygsti.baseobjs.CircuitLabel('',['Gx'],None,4),'Gy') )

    def test_circuitlabel_inclusion(self):
        c = pygsti.obj.Circuit(None,stringrep="GxGx(GyGiGi)^2", expand_subcircuits=False)
        self.assertTrue('Gi' in c)
        self.assertEqual(['Gi' in layer for layer in c], [False, False, True])

        c = pygsti.obj.Circuit(None,stringrep="Gx:0[Gx:0(Gy:1GiGi)^2]", num_lines=2, expand_subcircuits=False)
        self.assertTrue('Gi' in c)
        self.assertEqual(['Gi' in layer for layer in c], [False, True])

    def test_circuit_str_is_updated(self):
        #Test that .str is updated
        c = pygsti.obj.Circuit(None, stringrep="GxGx(GyGiGi)^2", editable=True)
        self.assertTrue(c._str is None)
        self.assertEqual(c.str, "GxGxGyGiGiGyGiGi")

        c.delete_layers(slice(1,4))
        self.assertTrue(c._str is None)
        self.assertEqual(c.str, "GxGiGyGiGi")
        c.done_editing()
        self.assertEqual(c.str, "GxGiGyGiGi")

        c = pygsti.obj.Circuit('Gi')
        c = c.copy(editable=True)
        self.assertEqual(c.str, "Gi")
        c.replace_gatename_inplace('Gi','Gx')
        self.assertTrue(c._str is None)
        self.assertEqual(c.str, "Gx")

    def test_simulate_circuitlabels(self):
        from pygsti.construction import std1Q_XYI

        pygsti.obj.Circuit.default_expand_subcircuits = False # so mult/exponentiation => CircuitLabels

        Gi = pygsti.obj.Circuit(None,stringrep='Gi',editable=True)
        Gy = pygsti.obj.Circuit(None,stringrep='Gy',editable=True)
        c2 = Gy*2
        #print(c2.tup)
        c3 = Gi + c2
        c2.done_editing()
        c3.done_editing()

        Gi.done_editing()
        Gy.done_editing()

        tgt = std1Q_XYI.target_model()
        for N,zeroProb in zip((1,2,10,100,10000),(0.5, 0, 0, 1, 1)):
            p1 = tgt.probs(('Gi',) + ('Gy',)*N)
            p2 = tgt.probs( Gi + Gy*N )
            self.assertAlmostEqual(p1['0'], zeroProb)
            self.assertAlmostEqual(p2['0'], zeroProb)

        pygsti.obj.Circuit.default_expand_subcircuits = True

    def test_circuit_exponentiation(self):
        pygsti.obj.Circuit.default_expand_subcircuits = False
        Gi = pygsti.obj.Circuit("Gi")
        Gy = pygsti.obj.Circuit("Gy")

        c = Gi + Gy**2048
        self.assertEqual(c[1].tonative(), ('', None, 2048, 'Gy'))  # more a label test? - but tests circuit exponentiation
        pygsti.obj.Circuit.default_expand_subcircuits = True

    def test_circuit_as_label(self):
        #test Circuit -> CircuitLabel conversion w/exponentiation
        c1 = pygsti.obj.Circuit(None, stringrep='Gi[][]',num_lines=4,editable=True)
        c2 = pygsti.obj.Circuit(None, stringrep='[Gx:0Gx:1][Gy:1]',num_lines=2)

        #Insert the 2Q circuit c2 into the 4Q circuit c as an exponentiated block (so c2 is exponentiated as well)
        c = c1.copy()
        c[1, 0:2] = c2.as_label(nreps=2)

        self.assertEqual(c, ('Gi', ('', (0, 1), 2, (('Gx', 0), ('Gx', 1)), ('Gy', 1)), ()))
        self.assertEqual(c.num_layers(), 3)
        self.assertEqual(c.depth(), 6)
        # Qubit 0 ---|Gi|-||([Gx:0Gx:1]Gy:1)^2||-| |---
        # Qubit 1 ---|Gi|-||([Gx:0Gx:1]Gy:1)^2||-| |---
        # Qubit 2 ---|Gi|-|                    |-| |---
        # Qubit 3 ---|Gi|-|                    |-| |---

        c = c1.copy()
        c[1, 0:2] = c2  # special behavior: c2 is converted to a label to cram it into a single layer
        self.assertEqual(c, ('Gi', ('', (0, 1), 1, (('Gx', 0), ('Gx', 1)), ('Gy', 1)), ()))
        self.assertEqual(c.num_layers(), 3)
        self.assertEqual(c.depth(), 4)
        
        # Qubit 0 ---|Gi|-||([Gx:0Gx:1]Gy:1)||-| |---
        # Qubit 1 ---|Gi|-||([Gx:0Gx:1]Gy:1)||-| |---
        # Qubit 2 ---|Gi|-|                  |-| |---
        # Qubit 3 ---|Gi|-|                  |-| |---

        c = c1.copy()
        c[(1,2), 0:2] = c2 # writes into described block
        self.assertEqual(c, ('Gi', (('Gx', 0), ('Gx', 1)), ('Gy',1)))
        self.assertEqual(c.num_layers(), 3)
        self.assertEqual(c.depth(), 3)
        # Qubit 0 ---|Gi|-|Gx|-|  |---
        # Qubit 1 ---|Gi|-|Gx|-|Gy|---
        # Qubit 2 ---|Gi|-|  |-|  |---
        # Qubit 3 ---|Gi|-|  |-|  |---
        
        c = c1.copy()
        c[(1,2), 0:2] = c2.as_label().components  # same as above, but more roundabout
        self.assertEqual(c, ('Gi', (('Gx', 0), ('Gx', 1)), ('Gy',1)))
        self.assertEqual(c.num_layers(), 3)
        self.assertEqual(c.depth(), 3)

    def test_empty_tuple_makes_idle_layer(self):
        c = pygsti.obj.Circuit( ['Gi', pygsti.obj.Label(())] )
        self.assertEqual(len(c), 2)

    def test_replace_with_idling_line(self):
        c = pygsti.obj.Circuit( [('Gcnot',0,1)], editable=True)
        c.replace_with_idling_line(0)
        self.assertEqual(c, ((),))

if __name__ == "__main__":
    unittest.main(verbosity=2)
