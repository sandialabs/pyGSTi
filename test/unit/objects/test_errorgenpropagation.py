from ..util import BaseCase
from pygsti.algorithms.randomcircuit import create_random_circuit
from pygsti.errorgenpropagation.errorpropagator import ErrorGeneratorPropagator
from pygsti.processors import QubitProcessorSpec
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.baseobjs import Label, BuiltinBasis, QubitSpace, CompleteElementaryErrorgenBasis
from pygsti.baseobjs.errorgenlabel import GlobalElementaryErrorgenLabel, LocalElementaryErrorgenLabel
from pygsti.tools import errgenproptools as _eprop
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.tools.matrixtools import print_mx
from itertools import product
import numpy as np
import stim


class ErrorgenPropTester(BaseCase):

    def setUp(self):
        num_qubits = 4
        gate_names = ['Gcphase', 'Gxpi2', 'Gypi2']
        availability = {'Gcphase':[(0,1), (1,2), (2,3), (3,0)]}
        pspec = QubitProcessorSpec(num_qubits, gate_names, availability=availability)
        self.target_model = create_crosstalk_free_model(processor_spec = pspec)
        self.circuit = create_random_circuit(pspec, 4, sampler='edgegrab', samplerargs=[0.4,], rand_state=12345)

        typ = 'H'
        max_stochastic = {'S': .0005, 'H': 0, 'H+S': .0001}
        max_hamiltonian = {'S': 0, 'H': .00005, 'H+S': .0001}
        max_strengths = {1: {'S': max_stochastic[typ], 'H': max_hamiltonian[typ]},
                        2: {'S': 3*max_stochastic[typ], 'H': 3*max_hamiltonian[typ]}
                        }
        error_rates_dict = sample_error_rates_dict(pspec, max_strengths, seed=12345)
        self.error_model = create_crosstalk_free_model(pspec, lindblad_error_coeffs=error_rates_dict)

    def test_exact_propagation_probabilities(self):
        #This should simultaneously confirm that the propagation code runs
        #and also that it is giving the correct values by directly comparing
        #to the probabilities from direct forward simulation.
        error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        probabilities_exact_propagation = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit)
        probabilities_forward_simulation = probabilities_fwdsim(self.error_model, self.circuit)

        self.assertTrue(np.linalg.norm(probabilities_exact_propagation - probabilities_forward_simulation, ord=1) < 1e-10)

    def test_approx_propagation_probabilities_BCH(self):
        error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        probabilities_BCH_order_1 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=1)
        probabilities_BCH_order_2 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=2)
        probabilities_BCH_order_3 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=3)
        probabilities_BCH_order_4 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=4)
        probabilities_BCH_order_5 = probabilities_errorgen_prop(error_propagator, self.target_model, self.circuit, use_bch=True, bch_order=5)
        probabilities_forward_simulation = probabilities_fwdsim(self.error_model, self.circuit)

        #use a much looser constraint on the agreement between the BCH results and forward simulation. Mostly testing to catch things exploding.
        TVD_order_1 = np.linalg.norm(probabilities_BCH_order_1 - probabilities_forward_simulation, ord=1)
        TVD_order_2 = np.linalg.norm(probabilities_BCH_order_2 - probabilities_forward_simulation, ord=1)
        TVD_order_3 = np.linalg.norm(probabilities_BCH_order_3 - probabilities_forward_simulation, ord=1)
        TVD_order_4 = np.linalg.norm(probabilities_BCH_order_4 - probabilities_forward_simulation, ord=1)
        TVD_order_5 = np.linalg.norm(probabilities_BCH_order_5 - probabilities_forward_simulation, ord=1)
        
        #loose bound is just to make sure nothing exploded.
        self.assertTrue(TVD_order_1 < 1e-2)
        self.assertTrue(TVD_order_2 < 1e-2)
        self.assertTrue(TVD_order_3 < 1e-2)
        self.assertTrue(TVD_order_4 < 1e-2)
        self.assertTrue(TVD_order_5 < 1e-2)

        #also assert that the TVDs get smaller in general as you go up in order.
        self.assertTrue((TVD_order_1>TVD_order_2) and (TVD_order_2>TVD_order_3) and (TVD_order_3>TVD_order_4) and (TVD_order_4>TVD_order_5))
        
    def test_eoc_error_channel(self):
        error_propagator = ErrorGeneratorPropagator(self.error_model.copy())
        eoc_error_channel = error_propagator.eoc_error_channel(self.circuit)

        #manually compute end-of-circuit error generator
        ideal_channel = self.target_model.sim.product(self.circuit)
        noisy_channel_exact = self.error_model.sim.product(self.circuit)
        eoc_error_channel_exact = noisy_channel_exact@ideal_channel.conj().T  

        assert np.linalg.norm(eoc_error_channel - eoc_error_channel_exact) < 1e-10
        

class LocalStimErrorgenLabelTester(BaseCase):
    def setUp(self):
        self.local_eel = LocalElementaryErrorgenLabel('C', ['XX', 'YY'])
        self.global_eel = GlobalElementaryErrorgenLabel('C', ['XX', 'YY'], (0,1))
        self.sslbls = [0,1]
        self.tableau = stim.PauliString('XI').to_tableau()

    def test_cast(self):
        correct_lse = _LSE('C', [stim.PauliString('XX'), stim.PauliString('YY')])

        self.assertEqual(correct_lse, _LSE.cast(self.local_eel))
        self.assertEqual(correct_lse, _LSE.cast(self.global_eel, self.sslbs))

    def test_to_local_global_eel(self):
        lse = _LSE('C', [stim.PauliString('XX'), stim.PauliString('YY')])

        self.assertEqual(lse.to_local_eel(), self.local_eel)
        self.assertEqual(lse.to_global_eel(), self.global_eel)
    
    def test_propagate_error_gen_tableau(self):
        lse = _LSE('C', [stim.PauliString('XX'), stim.PauliString('YY')])
        propagated_lse = lse.propagate_error_gen_tableau(self.tableau, 1)
        self.assertEqual(propagated_lse, (_LSE('C', [stim.PauliString('XX'), stim.PauliString('YY')]), -1))
        
        lse = _LSE('S', [stim.PauliString('ZI')])
        propagated_lse = lse.propagate_error_gen_tableau(self.tableau, 1)
        self.assertEqual(propagated_lse, (_LSE('S', [stim.PauliString('ZI')]), 1))

#Helper Functions:
def probabilities_errorgen_prop(error_propagator, target_model, circuit, use_bch=False, bch_order=1, truncation_threshold=1e-14):
    #get the eoc error channel, and the process matrix for the ideal circuit:
    if use_bch:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True, use_bch=use_bch,
                                                        bch_kwargs={'bch_order':bch_order,
                                                                    'truncation_threshold':truncation_threshold})
    else:
        eoc_channel = error_propagator.eoc_error_channel(circuit, include_spam=True)
    ideal_channel = target_model.sim.product(circuit)
    #also get the ideal state prep and povm:
    ideal_prep = target_model.circuit_layer_operator(Label('rho0'), typ='prep').copy()
    ideal_meas = target_model.circuit_layer_operator(Label('Mdefault'), typ='povm').copy()
    #calculate the probabilities.
    prob_vec = np.zeros(len(ideal_meas))
    for i, effect in enumerate(ideal_meas.values()):
        dense_effect = effect.to_dense().copy()
        dense_prep = ideal_prep.to_dense().copy()
        prob_vec[i] = np.linalg.multi_dot([dense_effect.reshape((1,len(dense_effect))), eoc_channel, ideal_channel, dense_prep.reshape((len(dense_prep),1))])
    return prob_vec

def probabilities_fwdsim(noise_model, circuit):
    prob_dict = noise_model.sim.probs(circuit)
    prob_vec = np.fromiter(prob_dict.values(), dtype=np.double)
    return prob_vec

def sample_error_rates_dict(pspec, strengths, seed=None):
    """
    For example:
        strengths = {1: {'S':0.001, 'H':0.01}, 
                    2: {'S':0.01,'H':0.1}}

    The 'S' and 'H' entries in the strengths dictionary give 
    the maximum possible contribution to the infidelity from a given gate.
    """
    qubits = pspec.qubit_labels
    errors_rates_dict = {}
    for gate, availability in pspec.availability.items():
        n = pspec.gate_num_qubits(gate)
        if availability == 'all-edges':
            assert(n == 1), "Currently require all 2-qubit gates have a specified availability!"
            qubits_for_gate = qubits
        else:
            qubits_for_gate = availability  
        for qs in qubits_for_gate:
            label = Label(gate, qs)
            # First, check if there's a strength specified for this specific gate.
            max_stength = strengths.get(label, None) # to get highly biased errors can set generic error rates to be low, then set it to be high for one or two particular gates.
            # Next, check if there's a strength specified for all gates with this name
            if max_stength is None:
                max_stength = strengths.get(gate, None)
            # Finally, get error rate for all gates on this number of qubits.
            if max_stength is None:
                max_stength = strengths[n]
            # Sample error rates.
            errors_rates_dict[label] = sample_error_rates(max_stength, n, seed)
    return errors_rates_dict

def sample_error_rates(strengths, n, seed = None):
    '''
    Samples an error rates dictionary for dependent gates.
    '''
    error_rates_dict = {}
    
    #create a basis to get the basis element labels.
    basis = BuiltinBasis('pp', 4**n)
    
    #set the rng
    rng = np.random.default_rng(seed)
    
    # Sample stochastic error rates. First we sample the overall stochastic error rate.
    # Then we sample (and normalize) the individual stochastic error rates
    stochastic_strength = strengths['S'] * rng.random()
    s_error_rates = rng.random(4 ** n - 1)
    s_error_rates = s_error_rates / np.sum(s_error_rates) * stochastic_strength

    hamiltonian_strength = strengths['H'] * rng.random()
    h_error_rates = rng.random(4 ** n - 1)
    h_error_rates = h_error_rates * np.sqrt(hamiltonian_strength) / np.sqrt(np.sum(h_error_rates**2))

    error_rates_dict.update({('S', basis.labels[i + 1]): s_error_rates[i] for i in range(4 ** n - 1)})
    error_rates_dict.update({('H', basis.labels[i + 1]): h_error_rates[i] for i in range(4 ** n - 1)})

    return error_rates_dict

def comm_list_to_matrix(comm_list, errorgen_matrix_dict, num_qubits):
    #if the list is empty return all zeros
    #initialize empty array for accumulation.
    mat = np.zeros((4**num_qubits, 4**num_qubits), dtype=np.complex128)
    if not comm_list:
        return mat
    
    #infer the correct label type.
    if errorgen_matrix_dict:
        first_label = next(iter(errorgen_matrix_dict))
        if isinstance(first_label, LocalElementaryErrorgenLabel):
            label_type = 'local'
        elif isinstance(first_label, GlobalElementaryErrorgenLabel):
            label_type = 'global'
        else:
            msg = f'Label type {type(first_label)} is not supported as a key for errorgen_matrix_dict.'\
                  + 'Please use either LocalElementaryErrorgenLabel or GlobalElementaryErrorgenLabel.'
            raise ValueError()
    else:
        raise ValueError('Non-empty commutatory result list, but the dictionary is empty. Cannot convert.')
        
    #loop through comm_list and accumulate the weighted error generators prescribed.
    if label_type == 'local':
        for comm_tup in comm_list:
            mat +=  comm_tup[1]*errorgen_matrix_dict[comm_tup[0].to_local_eel()]
    else:
        for comm_tup in comm_list:
            mat +=  comm_tup[1]*errorgen_matrix_dict[comm_tup[0].to_global_eel()]
            
    return mat

def error_generator_commutator_numerical(errorgen_1, errorgen_2, errorgen_matrix_dict):
    return errorgen_matrix_dict[errorgen_1]@errorgen_matrix_dict[errorgen_2] - errorgen_matrix_dict[errorgen_2]@errorgen_matrix_dict[errorgen_1]



