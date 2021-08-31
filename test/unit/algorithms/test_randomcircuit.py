import numpy as np

from pygsti.baseobjs import Label as L
from pygsti.processors import QubitProcessorSpec as QPS
from ..util import BaseCase

from pygsti.algorithms import randomcircuit as _rc

class RandomCircuitTest(BaseCase):

    def test_random_circuit(self):

        n_qubits = 4
        qubit_labels = ['Q0','Q1','Q2','Q3'] 
        gate_names = ['Gxpi2', 'Gxmpi2', 'Gypi2', 'Gympi2', 'Gcphase'] 
        availability = {'Gcphase':[('Q0','Q1'), ('Q1','Q2'), ('Q2','Q3'), ('Q3','Q0')]}
        pspec = QPS(n_qubits, gate_names, availability=availability, qubit_labels=qubit_labels)
        depth = 10
        circuit = _rc.create_random_circuit(pspec, depth, sampler='Qelimination', samplerargs=[0.5])
        circuit = _rc.create_random_circuit(pspec, depth, sampler='edgegrab', samplerargs=[0.25])


        C2QGs1 = [] #  A list containing no 2-qubit gates is an acceptable set of compatible 2-qubit gates.
        C2QGs2 = [L('Gcphase',('Q0','Q1')),]
        C2QGs3 = [L('Gcphase',('Q1','Q2')),] 
        C2QGs4 = [L('Gcphase',('Q2','Q3')),] 
        C2QGs5 = [L('Gcphase',('Q3','Q0')),] 
        C2QGs6 = [L('Gcphase',('Q0','Q1')), L('Gcphase',('Q2','Q3')),] 
        C2QGs7 = [L('Gcphase',('Q1','Q2')), L('Gcphase',('Q3','Q0')),]

        co2Qgates = [C2QGs1, C2QGs2, C2QGs3, C2QGs4, C2QGs5, C2QGs6, C2QGs7]
        co2Qgatesprob = [0.5, 0.125, 0.125, 0.125, 0.125, 0, 0]
        twoQprob = 1 

        samplerargs = [co2Qgates, co2Qgatesprob, twoQprob]
        circuit = _rc.create_random_circuit(pspec, depth, sampler='co2Qgates', samplerargs=samplerargs)

        co2Qgates = [C2QGs1,[C2QGs2,C2QGs3,C2QGs4, C2QGs5]]
        co2Qgatesprob = [0.5,0.5] 
        twoQprob = 1 
        samplerargs = [co2Qgates,] 
        circuit = _rc.create_random_circuit(pspec, depth, sampler='co2Qgates', samplerargs=samplerargs)


class LayerSamplerTester(BaseCase):

    def test_edgrab(self):

        n = 4
        qs = ['Q'+str(i) for i in range(n)]
        ring = [('Q'+str(i),'Q'+str(i+1)) for i in range(n-1)]

        gateset1 =  ['Gcphase'] + ['Gc'+str(i) for i in range(24)]
        gateset2 =  ['Gcphase'] + ['Gxpi2', 'Gzr']
        gateset3 =  ['Gczr'] + ['Gxpi2', 'Gzr']

        pspec1 = QPS(n, gateset1, availability={'Gcphase':ring}, qubit_labels=qs)
        pspec2 = QPS(n, gateset2, availability={'Gcphase':ring}, qubit_labels=qs)
        pspec3 = QPS(n, gateset3, availability={'Gczr':ring}, qubit_labels=qs)

        q_set = ('Q0', 'Q1', 'Q2')

        l1 = _rc.sample_circuit_layer_by_edgegrab(pspec1, qubit_labels=q_set, two_q_gate_density=0.25, one_q_gate_names=None, 
                        gate_args_lists={'Gczr':[('-0.1',),('+0.1',)]})

        l2 = _rc.sample_circuit_layer_by_edgegrab(pspec2, qubit_labels=q_set,  two_q_gate_density=0.25, one_q_gate_names=[], 
         gate_args_lists=None)

        l3 = _rc.sample_circuit_layer_by_edgegrab(pspec3, qubit_labels=q_set,  two_q_gate_density=0.25, one_q_gate_names=[], 
                gate_args_lists=None)

        l4 = _rc.sample_circuit_layer_by_edgegrab(pspec3, qubit_labels=q_set,  two_q_gate_density=0.25, one_q_gate_names=['Gxpi2',], 
                gate_args_lists={'Gczr':[('-0.1',),('+0.1',)]})
