import numpy as np
import pygsti
from pygsti.extras import rb
from pygsti.baseobjs import Label

def test_sample():
    
    # -- pspecs to use in all the tests. They cover a variety of possibilities -- #
    
    n_1 = 4
    glist = ['Gi','Gxpi2','Gypi2','Gcnot']
    pspec_1 = pygsti.obj.ProcessorSpec(n_1,glist,verbosity=0,qubit_labels=['Q0','Q1','Q2','Q3'])

    n_2 = 3
    glist = ['Gi','Gxpi','Gypi','Gzpi','Gh','Gp','Gcphase']
    availability = {'Gcphase':[(0,1),(1,2)]}
    pspec_2 = pygsti.obj.ProcessorSpec(n_2,glist,availability=availability,verbosity=0)
    
    # Tests Clifford RB samplers
    lengths = [0,2,5]
    circuits_per_length = 2
    subsetQs = ['Q1','Q2','Q3']
    out = rb.sample.clifford_rb_experiment(pspec_1, lengths, circuits_per_length, subsetQs=subsetQs, randomizeout=False, 
                                       citerations=2, compilerargs=[], descriptor='A Clifford RB experiment', verbosity=0)
    for key in list(out['idealout'].keys()):
        assert(out['idealout'][key] == (0,0,0))

    assert(len(out['circuits']) == circuits_per_length * len(lengths))

    out = rb.sample.clifford_rb_experiment(pspec_2, lengths, circuits_per_length, subsetQs=None, randomizeout=True, 
                                       citerations=1, compilerargs=[], descriptor='A Clifford RB experiment',verbosity=0)
    
    # --- Tests of the circuit layer samplers --- #

    # Tests for the sampling by pairs function
    layer = rb.sample.circuit_layer_by_pairing_qubits(pspec_1, twoQprob=0.0, oneQgatenames='all', twoQgatenames='all', 
                                      gatesetname = 'clifford')
    assert(len(layer) == n_1)
    layer = rb.sample.circuit_layer_by_pairing_qubits(pspec_1, twoQprob=1.0, oneQgatenames='all', twoQgatenames='all', 
                                      gatesetname = 'clifford')
    assert(len(layer) == n_1//2)
    layer = rb.sample.circuit_layer_by_pairing_qubits(pspec_1, twoQprob=0.0, oneQgatenames=['Gx',], twoQgatenames='all', 
                                      gatesetname = 'target')
    assert(layer[0].name == 'Gx')

    layer = rb.sample.circuit_layer_by_Qelimination(pspec_2, twoQprob=0.0, oneQgates='all', twoQgates='all',
                                                    gatesetname='clifford')
    assert(len(layer) == n_2)
    layer = rb.sample.circuit_layer_by_Qelimination(pspec_2, twoQprob=1.0, oneQgates='all', twoQgates='all',
                                                    gatesetname='clifford')
    assert(len(layer) == (n_2 % 2) + n_2//2)
    layer = rb.sample.circuit_layer_by_pairing_qubits(pspec_1, twoQprob=0.0, oneQgatenames=['Gxpi'], twoQgatenames='all', 
                                      gatesetname = 'target')
    assert(layer[0].name == 'Gxpi')

    # Tests for the sampling by Co2QGs function
    C01 = Label('Gcnot',('Q0','Q1'))
    C23 = Label('Gcnot',('Q2','Q3'))
    Co2QGs = [[],[C01,C23]]
    layer = rb.sample.circuit_layer_by_Co2QGs(pspec_1, None, Co2QGs, Co2QGsprob='uniform', twoQprob=1.0, 
                                               oneQgatenames='all', gatesetname='clifford')
    assert(len(layer) == n_1 or len(layer) == n_1//2)
    layer = rb.sample.circuit_layer_by_Co2QGs(pspec_1, None, Co2QGs, Co2QGsprob=[0.,1.], twoQprob=1.0, 
                                               oneQgatenames='all', gatesetname='clifford')
    assert(len(layer) == n_1//2)
    layer = rb.sample.circuit_layer_by_Co2QGs(pspec_1, None, Co2QGs, Co2QGsprob=[1.,0.], twoQprob=1.0, 
                                               oneQgatenames=['Gx',], gatesetname='clifford')
    assert(len(layer) == n_1)
    assert(layer[0].name == 'Gx')
    
    Co2QGs = [[],[C23,]]
    layer = rb.sample.circuit_layer_by_Co2QGs(pspec_1, ['Q2','Q3'], Co2QGs, Co2QGsprob=[0.25,0.75], twoQprob=0.5, 
                                               oneQgatenames='all', gatesetname='clifford')
    Co2QGs = [[C01,]]
    layer = rb.sample.circuit_layer_by_Co2QGs(pspec_1, None, Co2QGs, Co2QGsprob=[1.], twoQprob=1.0, 
                                               oneQgatenames='all', gatesetname='clifford')
    assert(layer[0].name == 'Gcnot')
    assert(len(layer) == 3)
    
    # Tests the nested Co2QGs option.
    Co2QGs = [[],[[C01,C23],[C01,]]]
    layer = rb.sample.circuit_layer_by_Co2QGs(pspec_1, None, Co2QGs, Co2QGsprob='uniform', twoQprob=1.0, 
                                               oneQgatenames='all', gatesetname='clifford')
    # Tests for the sampling a layer of 1Q gates.
    layer = rb.sample.circuit_layer_of_oneQgates(pspec_1, oneQgatenames='all', pdist='uniform',
                                                gatesetname='clifford')
    assert(len(layer) == n_1)
    layer = rb.sample.circuit_layer_of_oneQgates(pspec_1, subsetQs=['Q1','Q2'], oneQgatenames=['Gx','Gy'], pdist=[1.,0.],
                                                gatesetname='clifford')
    assert(len(layer) == 2)
    assert(layer[0].name == 'Gx')
    layer = rb.sample.circuit_layer_of_oneQgates(pspec_1, subsetQs=['Q2'],oneQgatenames=['Gx'], pdist=[3.,],
                                                gatesetname='clifford')
    assert(layer[0] == Label('Gx','Q2'))
    assert(len(layer) == 1)
    layer = rb.sample.circuit_layer_of_oneQgates(pspec_1, oneQgatenames=['Gx'], pdist='uniform',
                                                gatesetname='clifford')
    
    # Tests of the random_circuit sampler that is a wrap-around for the circuit-layer samplers
    
    C01 = Label('Gcnot',('Q0','Q1'))
    C23 = Label('Gcnot',('Q2','Q3'))
    Co2QGs = [[],[[C01,C23],[C01,]]]
    circuit = rb.sample.random_circuit(pspec_1, length=100, sampler='Qelimination')
    assert(circuit.depth() == 100)
    circuit = rb.sample.random_circuit(pspec_2, length=100, sampler='Qelimination', samplerargs=[0.1,], addlocal = True)
    assert(circuit.depth() == 201)
    assert(len(circuit.get_layer(0)) <= n_2)
    circuit = rb.sample.random_circuit(pspec_1, length=100, sampler='pairingQs')
    circuit = rb.sample.random_circuit(pspec_1, length=10, sampler='pairingQs', samplerargs=[0.1,['Gx',]])

    circuit = rb.sample.random_circuit(pspec_1, length=100, sampler='Co2QGs', samplerargs=[Co2QGs])
    circuit = rb.sample.random_circuit(pspec_1, length=100, sampler='Co2QGs', samplerargs=[Co2QGs,[0.1,0.2],0.1], 
                                addlocal = True, lsargs=[['Gx',]])
    assert(circuit.depth() == 201)
    circuit = rb.sample.random_circuit(pspec_1, length=5, sampler='local')
    assert(circuit.depth() == 5)
    circuit = rb.sample.random_circuit(pspec_1, length=5, sampler='local',samplerargs=[['Gx']])
    assert(circuit.line_items[0][0].name == 'Gx')
    
    lengths = [0,2,5]
    circuits_per_length = 2
    # Test DRB experiment with all defaults.
    exp = rb.sample.direct_rb_experiment(pspec_2, lengths, circuits_per_length, verbosity=0)
    
    exp = rb.sample.direct_rb_experiment(pspec_2, lengths, circuits_per_length, subsetQs=[0,1], sampler='pairingQs',
                                        cliffordtwirl=False, conditionaltwirl=False, citerations=2, partitioned=True,
                                        verbosity=0)
    
    exp = rb.sample.direct_rb_experiment(pspec_2, lengths, circuits_per_length, subsetQs=[0,1], sampler='Co2QGs',
                                         samplerargs = [[[],[Label('Gcphase',(0,1)),]],[0.,1.]],
                                        cliffordtwirl=False, conditionaltwirl=False, citerations=2, partitioned=True,
                                        verbosity=0)
    
    exp = rb.sample.direct_rb_experiment(pspec_2, lengths, circuits_per_length, subsetQs=[0,1], sampler='local',
                                        cliffordtwirl=False, conditionaltwirl=False, citerations=2, partitioned=True,
                                        verbosity=0)
    
    # Tests of MRB : gateset must have self-inverses in the gate-set.
    n_1 = 4
    glist = ['Gi','Gxpi2','Gxmpi2','Gypi2','Gympi2','Gcnot']
    pspec_inv = pygsti.obj.ProcessorSpec(n_1, glist, verbosity=0, qubit_labels=['Q0','Q1','Q2','Q3'])
    lengths = [0,4,8]
    circuits_per_length = 10
    exp = rb.sample.mirror_rb_experiment(pspec_inv, lengths, circuits_per_length, subsetQs=['Q1','Q2','Q3'],
                                         sampler='Qelimination', samplerargs=[], localclifford=True, 
                                         paulirandomize=True)
    
    exp = rb.sample.mirror_rb_experiment(pspec_inv, lengths, circuits_per_length, subsetQs=['Q1','Q2','Q3'],
                                         sampler='Qelimination', samplerargs=[], localclifford=True, 
                                         paulirandomize=False)
    
    exp = rb.sample.mirror_rb_experiment(pspec_inv, lengths, circuits_per_length, subsetQs=['Q1','Q2','Q3'],
                                         sampler='Qelimination', samplerargs=[], localclifford=False, 
                                         paulirandomize=False)
 
    exp = rb.sample.mirror_rb_experiment(pspec_inv, lengths, circuits_per_length, subsetQs=['Q1','Q2','Q3'],
                                         sampler='Qelimination', samplerargs=[], localclifford=False, 
                                         paulirandomize=True)    