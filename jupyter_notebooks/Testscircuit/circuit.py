import pygsti
from pygsti.objects import Circuit
from pygsti.objects import GateString
from pygsti.baseobjs import Label
from pygsti.objects import ProcessorSpec
import numpy as _np

def test_circuit():

    # Test initializing a circuit from an empty gatestring.
    c = pygsti.obj.Circuit(num_lines=5)
    assert(c.depth()==0)
    assert(c.size()==0)
    assert(c.number_of_lines == 5)
    assert(c.line_labels == list(range(5)))

    c = pygsti.obj.Circuit(gatestring=[],num_lines=5)
    assert(c.depth()==0)
    assert(c.size()==0)
    assert(c.number_of_lines == 5)
    assert(c.line_labels == list(range(5)))

    # Test initializing a circuit from a non-empty gatestring that is a list
    # containing Label objects. Also test that it can have non-integer line_labels
    # and a different identity identifier.
    gatestring=[Label('Gi','Q0'),Label('Gp','Q8')]
    c = pygsti.obj.Circuit(gatestring=gatestring,line_labels=['Q0','Q1','Q8','Q12'],identity='idle')
    # Not parallelized by default, so will be depth 2.
    assert(c.depth()==2)
    assert(c.size()==2)
    assert(c.number_of_lines == 4)
    assert(c.line_labels == ['Q0','Q1','Q8','Q12'])

    # Do again with parallelization
    c = pygsti.obj.Circuit(gatestring=gatestring,line_labels=['Q0','Q1','Q8'],parallelize=True)
    assert(c.depth()==1)
    assert(c.size()==2)

    # Now repeat the read-in with no parallelize, but a list of lists of gatelabels
    gatestring=[[Label('Gi','Q0'),Label('Gp','Q8')],[Label('Gh','Q1'),Label('Gp','Q12')]]
    c = pygsti.obj.Circuit(gatestring=gatestring,line_labels=['Q0','Q1','Q8','Q12'],identity='id')
    assert(c.depth())

    # Check we can read-in a gatestring that has no qubit labels: enforces them to be on
    # all of the lines.
    gatestring = GateString( None, "Gx^2GyGxGi" )
    c = pygsti.obj.Circuit(gatestring=gatestring,num_lines=1)
    assert(c.depth()==5)

    # Check that we can create a gatestring from a string and that we end up with
    # the correctly structured circuit.
    gatestring = GateString( None, "[Gx:Q0Gy:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1" )
    c = pygsti.obj.Circuit(gatestring=gatestring,parallelize=False,line_labels=['Q0','Q1'])
    assert(c.depth()==5)
    assert(c.size()==8)

    # Check we can init from a line_items list of lists
    cnew = pygsti.obj.Circuit(c.line_items)
    assert(cnew == c)

    # Check can't give line_items and gatestring.
    fail = True
    try:
        c = pygsti.obj.Circuit(line_items=line_items,gatestring=gatestring)
        fail = False
    except:
        pass
    assert(fail)

    # Test copy() and clear()
    ccopy = c.copy()
    ccopy.clear()
    assert(ccopy.size() == 0)
    assert(c.size() == 8)

    # Test inserting a gate when the relevant qubits aren't 
    # idling at that layer
    c.insert_gate(Label('Gx','Q0'),2)
    assert(c.size() == 9)
    assert(c.depth() == 6)
    assert(c.get_line('Q0')[2] == Label('Gx','Q0'))

    # Test inserting a gate when the relevant qubits are 
    # idling at that layer -- depth shouldn't increase
    c.insert_gate(Label('Gx','Q1'),2)
    assert(c.size() == 10)
    assert(c.depth() == 6)
    assert(c.get_line('Q1')[2] == Label('Gx','Q1'))

    # Test layer insertion
    layer = [Label('Gx','Q1'),]
    c.insert_layer(layer ,1)
    assert(c.size() == 11)
    assert(c.depth() == 7)
    assert(c.get_layer(1) == [Label('Gx','Q1'),])
    c.insert_layer([] ,1)
    assert(c.lines_are_idle_at_layer(['Q0','Q1'],1))
    assert(c.lines_are_idle_at_layer(['Q0'],2))
    assert(not c.lines_are_idle_at_layer(['Q1'],2))
    assert(not c.is_idling_qubit('Q1'))
    c.insert_idling_wires(['Q0','Q1','Q3'])
    assert(not c.is_idling_qubit('Q0'))
    assert(not c.is_idling_qubit('Q1'))
    assert(c.is_idling_qubit('Q3'))

    # Test replacing a layer with a layer.
    c = pygsti.obj.Circuit(gatestring=gatestring, line_labels=['Q0','Q1'], identity='id')
    newlayer = [Label('Gx','Q0')]
    c.replace_layer_with_layer(newlayer,1)
    assert(c.depth() == 5)

    # Test replacing a layer with a circuit
    c.replace_layer_with_circuit(c.copy(),1)
    assert(c.depth() == 2*5 - 1)

    # Test layer deletion
    ccopy = c.copy()
    ccopy.insert_layer(layer ,1)
    ccopy.delete_layer(1)
    assert(c == ccopy)

    # Test inserting a circuit when they are over the same labels.
    gatestring = GateString( None, "[Gx:Q0Gy:Q1][Gy:Q0Gx:Q1]Gx:Q0Giz:Q1" )
    c = pygsti.obj.Circuit(gatestring=gatestring,parallelize=False,line_labels=['Q0','Q1'])
    ccopy = c.copy()
    c.insert_circuit(ccopy,2)
    assert(Label('Gx','Q0') in c.get_layer(2))

    # Test insert a circuit that is over *more* qubits but which has the additional
    # lines idling.
    c1 = pygsti.obj.Circuit(gatestring=gatestring,line_labels=['Q0','Q1'])
    c2 = pygsti.obj.Circuit(gatestring=gatestring,line_labels=['Q0','Q1','Q2','Q3'])
    c1.insert_circuit(c2,0)
    assert(c1.line_labels == ['Q0','Q1'])
    assert(c1.number_of_lines == 2)

    # Test inserting a circuit that is on *less* qubits.
    c1 = pygsti.obj.Circuit(gatestring=gatestring, line_labels=['Q0','Q1'], identity='id')
    c2 = pygsti.obj.Circuit(gatestring=[Label('Gx','Q0')], line_labels=['Q0',])
    c1.insert_circuit(c2,1)
    assert(c1.line_labels == ['Q0','Q1'])
    assert(c1.number_of_lines == 2)

    c1 = pygsti.obj.Circuit(gatestring=gatestring, line_labels=['Q0','Q1'], identity='id')
    c2 = pygsti.obj.Circuit(gatestring=[Label('Gx','Q0')], line_labels=['Q0',])
    c1.append_circuit(c2)
    c1.prefix_circuit(c2)

    # Test changing a gate name
    gatestring = GateString( None, "[Gx:Q0Gy:Q1][Gy:Q0Gx:Q1]Gx:Q0Gi:Q1")
    c = pygsti.obj.Circuit(gatestring=gatestring,line_labels=['Q0','Q1'],identity='Gi')
    c.replace_gatename('Gx','Gz')
    gatestring = GateString( None, "[Gz:Q0Gy:Q1][Gy:Q0Gz:Q1]Gz:Q0Gi:Q1" )
    c2 = pygsti.obj.Circuit(gatestring=gatestring,parallelize=False,line_labels=['Q0','Q1'],identity='Gi')
    assert(c == c2)

    # Change gate library using an ordinary dict with every gate as a key. (we test
    # changing gate library using a CompilationLibrary elsewhere in the tests).
    comp = {}
    comp[Label('Gz','Q0')] = pygsti.obj.Circuit(gatestring=[Label('Gx','Q0')],line_labels=['Q0'])
    comp[Label('Gy','Q0')] = pygsti.obj.Circuit(gatestring=[Label('Gx','Q0')],line_labels=['Q0'])
    comp[Label('Gz','Q1')] = pygsti.obj.Circuit(gatestring=[Label('Gx','Q1')],line_labels=['Q1'])
    comp[Label('Gy','Q1')] = pygsti.obj.Circuit(gatestring=[Label('Gx','Q1')],line_labels=['Q1'])
    c.change_gate_library(comp)
    assert(Label('Gx','Q0') in c.get_layer(0))

    # Change gate library using a dict with some gates missing
    comp = {}
    comp[Label('Gz','Q0')] = pygsti.obj.Circuit(gatestring=[Label('Gx','Q0')],line_labels=['Q0'])
    c = pygsti.obj.Circuit(gatestring=gatestring,line_labels=['Q0','Q1'],identity='Gi')
    c.change_gate_library(comp,allow_unchanged_gates=True)
    assert(Label('Gx','Q0') in c.get_layer(0))
    assert(Label('Gy','Q1') in c.get_layer(0))

    # Test we can change the labels of the lines.
    c.map_state_space_labels({'Q0':0,'Q1':1})
    assert(c.line_labels == [0,1])
    assert(c.get_line(0)[0].qubits[0] == 0)

    # Check we can re-order wires
    c.reorder_wires([1,0])
    assert(c.line_labels == [1,0])
    # Can't use .get_line as that takes the line label as i_nput.
    assert(c.line_items[0][0].qubits[0] == 1)

    # Test deleting and inserting idling wires.
    c = pygsti.obj.Circuit(gatestring=gatestring,line_labels=['Q0','Q1'],identity='Gi')
    c.insert_idling_wires(['Q0','Q1','Q2'])
    assert(c.line_labels == ['Q0','Q1','Q2'])
    assert(c.number_of_lines == 3)
    c.delete_idling_wires()
    assert(c.line_labels == ['Q0','Q1'])
    assert(c.number_of_lines == 2)

    # Test circuit reverse.
    gate1 = c.get_line('Q0')[0]
    c.reverse()
    gate2 = c.get_line('Q0')[-1]
    assert(gate1 == gate2)

    # Test 2-qubit and multi-qubit gate count
    assert(c.twoqubit_gatecount() == 0)
    gatestring = GateString( None, "[Gcnot:Q0:Q1]^2[Gy:Q0Gx:Q1]Gi:Q0Gi:Q1" )
    c = pygsti.obj.Circuit(gatestring=gatestring,parallelize=False,line_labels=['Q0','Q1'])
    assert(c.twoqubit_gatecount() == 2)
    assert(c.multiqubit_gatecount() == 2)
    gatestring = GateString( None, "[Gccnot:Q0:Q1:Q2]^2[Gccnot:Q0:Q1]Gi:Q0Gi:Q1" )
    c = pygsti.obj.Circuit(gatestring=gatestring,parallelize=False,line_labels=['Q0','Q1','Q2'])
    assert(c.twoqubit_gatecount() == 1)
    assert(c.multiqubit_gatecount() == 3)

    # Test the error-probability prediction method
    gatestring = GateString( None, "[Gx:Q0][Gi:Q0Gi:Q1]")
    c = pygsti.obj.Circuit(gatestring=gatestring,parallelize=False,line_labels=['Q0','Q1'],identity='Gi')
    infidelity_dict = {}
    infidelity_dict[Label('Gi','Q0')] = 0.7
    infidelity_dict[Label('Gi','Q1')] = 0.9
    infidelity_dict[Label('Gx','Q0')] = 0.8
    infidelity_dict[Label('Gx','Q2')] = 0.9
    epsilon = c.predicted_error_probability(infidelity_dict)
    assert(abs(epsilon - (1 - (1-0.7)*(1-0.8)*(1-0.9)**2)) < 10**-10)

    # Check we can succesfully create a circuit string.
    s = c.__str__()

    # Check we can write to a Qcircuit file.
    c.write_Qcircuit_tex('test_qcircuit.tex')

    # Test depth compression both with and without 1-qubit gate compression
    ls = [Label('H',1),Label('P',1),Label('P',1),Label('I',1),Label('CNOT',(2,3))]
    ls += [Label('HP',1),Label('PH',1),Label('CNOT',(1,2))]
    ls += [Label('I',1),Label('I',2),Label('CNOT',(1,2))]
    gatestring = GateString(ls)
    c = pygsti.obj.Circuit(gatestring=gatestring, num_lines=4)
    c.compress_depth(verbosity=0)
    assert(c.depth() == 7)
    # Gate a dictionary that relates H, P gates etc.
    oneQrelations = pygsti.symplectic.oneQclifford_symplectic_group_relations()
    c.compress_depth(oneQgate_relations = oneQrelations)
    assert(c.depth() == 3)
    
    # Test the is_valid_circuit checker.
    c.is_valid_circuit()
    fail = True
    try:
        c.line_items[0][2] = Label('CNOT',(2,3))
        c.is_valid_circuit()
        fail = False
    except:
        pass
    assert(fail)
    
    # Check that convert_to_quil runs, doesn't check the output makes sense.
    gatestring = [Label(('Gi','Q1')),Label(('Gxpi','Q1')),Label('Gcnot',('Q1','Q2'))]
    c = Circuit(gatestring=gatestring,line_labels=['Q1','Q2'],identity='Gi')
    s = c.convert_to_quil()

    # Check done_editing makes the circuit static.
    c.done_editing()
    fail = True
    try:
        c.clear()
        fail = False
    except:
        pass
    assert(fail)
    
    # Create a pspec, to test the circuit simulator.
    n = 4
    qubit_labels = ['Q'+str(i) for i in range(n)]
    availability = {'Gcnot':[('Q'+str(i),'Q'+str(i+1)) for i in range(0,n-1)]}
    gate_names = ['Gi','Gh','Gp','Gxpi','Gpdag','Gcnot']
    ps = ProcessorSpec(n,gate_names=gate_names,qubit_labels=qubit_labels)

    # Tests the circuit simulator
    c = Circuit(gatestring=[Label('Gh','Q0'),Label('Gcnot',('Q0','Q1'))],line_labels=['Q0','Q1'],identity='Gi')
    out = c.simulate(ps.models['target'])
    assert(abs(out['00'] - 0.5) < 10**-10)
    assert(abs(out['11'] - 0.5) < 10**-10)