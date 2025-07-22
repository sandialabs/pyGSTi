import numpy as np
import scipy.linalg as la

from pygsti.baseobjs import qubitgraph as _qgraph
from pygsti.baseobjs import QubitSpace
from pygsti.models import modelconstruction as pgmc
from pygsti.processors import QubitProcessorSpec
from pygsti.modelmembers.states import ComposedState, ComputationalBasisState
from pygsti.modelmembers.povms import ComposedPOVM
from pygsti.modelmembers.operations import ComposedOp, LindbladErrorgen, ExpErrorgenOp
# from pygsti.tools.lindbladtools import random_error_generator_rates
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.circuits import Circuit


from pygsti.baseobjs import qubitgraph as _qgraph
from pygsti.processors import QubitProcessorSpec

# Nick's additions to make it work.
from pygsti.algorithms import BuiltinBasis
from pygsti.tools import unitary_to_superop

from pygsti.baseobjs import Label
from pygsti.modelmembers import operations as op
from pygsti.baseobjs import UnitaryGateFunction

from pygsti.forwardsims.matrixforwardsim import LCSEvalTreeMatrixForwardSimulator


def assert_probability_densities_are_equal(op_dict: dict, exp_dict: dict, cir: Circuit):

    for key, val in op_dict.items():
        assert key in exp_dict
        assert np.allclose(exp_dict[key], val), f"Circuit {cir}, Outcome {key}, Expected: {exp_dict[key]}, Got: {val}"

#region Model Construction
def construct_arbitrary_single_qubit_unitary(alpha, beta, gamma, delta):

    first_term = np.exp(alpha*1j)
    left_mat = np.array([np.exp(-1j*beta/2), 0, 0, np.exp(1j*beta/2)]).reshape(2,2)
    rotate_mat = np.array([np.cos(gamma/2), -np.sin(gamma/2), np.sin(gamma/2), np.cos(gamma/2)]).reshape(2,2)
    right_mat = np.array([np.exp(-1j*delta/2), 0, 0, np.exp(1j*delta/2)]).reshape(2,2)

    my_matrix = first_term * (left_mat @ (rotate_mat @ right_mat))

    assert np.allclose(np.conjugate( my_matrix.T) @ my_matrix, np.eye(2))

    return my_matrix


class MyContinuouslyParameterizedGateFunction(UnitaryGateFunction):
    shape = (2, 2)
    def __call__(self, alpha, beta, gamma, delta):                                                                                                                                                                                                                                                                                                                                                                                                                   
        return construct_arbitrary_single_qubit_unitary(alpha, beta, gamma, delta)


class ArbParameterizedOpFactory(op.OpFactory):
    def __init__(self, state_space, location:int):
        op.OpFactory.__init__(self, state_space=state_space, evotype="densitymx")
        self.my_state_space = state_space
        self.interesting_qubit = location

    def create_object(self, args=None, sslbls=None):
        # Note: don't worry about sslbls (unused) -- this argument allow factories to create different operations on different target qubits
        assert(len(args) == 4)
        alpha, beta, gamma, delta = args
        
        unitary = construct_arbitrary_single_qubit_unitary(alpha, beta, gamma, delta)
        
        superop = unitary_to_superop(unitary)
        return op.EmbeddedOp(state_space = self.my_state_space, target_labels=self.interesting_qubit, operation_to_embed=op.StaticArbitraryOp(superop))

def make_spam(num_qubits):
    state_space = QubitSpace(num_qubits)
    max_weights = {'H':1, 'S':1, 'C':1, 'A':1}
    egbn_H_only = CompleteElementaryErrorgenBasis(BuiltinBasis("PP", 4), state_space, ('H',), max_weights)

    rho_errgen_rates = {ell: 0.0 for ell in egbn_H_only.labels}
    rho_lindblad = LindbladErrorgen.from_elementary_errorgens(rho_errgen_rates, parameterization='H', state_space=state_space, evotype='densitymx')
    rho_errorgen = ExpErrorgenOp(rho_lindblad)
    rho_ideal    = ComputationalBasisState([0]*num_qubits)
    rho          = ComposedState(rho_ideal, rho_errorgen)

    M_errgen_rates = {ell: 0.0 for ell in egbn_H_only.labels}
    M_lindblad = LindbladErrorgen.from_elementary_errorgens(M_errgen_rates, parameterization='H', state_space=state_space, evotype='densitymx')
    M_errorgen = ExpErrorgenOp(M_lindblad)
    M = ComposedPOVM(M_errorgen)

    return rho, M


def make_target_model(num_qubits, independent_gates: bool = True, arbitrary_unit: bool = False):
    ps_geometry = _qgraph.QubitGraph.common_graph(
        num_qubits, geometry='line',
        directed=True, all_directions=True,
        qubit_labels=tuple(range(num_qubits))
    )
    u_ecr = 1/np.sqrt(2)*np.array([[0,0,1,1j],[0,0,1j,1],[1,-1j,0,0],[-1j,1,0,0]])
    gatenames = ['Gxpi2', 'Gypi2', 'Gzpi2', 'Gi', 'Gii',  'Gecr']
    ps = QubitProcessorSpec(
        num_qubits=num_qubits,
        gate_names=gatenames,
        nonstd_gate_unitaries={'Gecr': u_ecr, 'Gii': np.eye(4)},
        geometry=ps_geometry
    )
    gateerrs = dict()
    basis = BuiltinBasis("PP", QubitSpace(1))
    egb1 = CompleteElementaryErrorgenBasis(basis, QubitSpace(1), ('H','S')) # XXXX From Riley's code, default_label_type='local')
    for gn in gatenames[:-1]:
        gateerrs[gn] = {ell: 0 for ell in egb1.labels}
    egb2 = CompleteElementaryErrorgenBasis(basis, QubitSpace(2), ('H','S')) # XXXX From Riley's code, default_label_type='local')
    gateerrs['Gecr'] = {ell: 0 for ell in egb2.labels}
    gateerrs['Gii'] = gateerrs['Gecr']

    tmn = pgmc.create_crosstalk_free_model(ps, lindblad_error_coeffs=gateerrs, independent_gates=independent_gates) #,
                                        #    ideal_spam_type="CPTPLND")

    rho, M = make_spam(num_qubits)
    tmn.prep_blks['layers']['rho0'] = rho
    tmn.povm_blks['layers']['Mdefault'] = M
    # tmn._mark_for_rebuild()
    tmn._rebuild_paramvec()
    
    # tmn._layer_rules.implicit_idle_mode = "pad_1Q"

    if arbitrary_unit:
        for i in range(num_qubits):
            Ga_factory = ArbParameterizedOpFactory(state_space=QubitSpace(num_qubits), location=(i,))
            tmn.factories["layers"][("Gcustom", i)] = Ga_factory # add in the factory for every qubit.
    
    return tmn

def build_models_for_testing(num_qubits):

    tgt_model = make_target_model(num_qubits)

    # target_model.sim.calclib = pygsti.forwardsims.mapforwardsim_calc_generic
    tgt_model.sim = LCSEvalTreeMatrixForwardSimulator()

    tgt_model2 = make_target_model(num_qubits)

    return tgt_model, tgt_model2


#endregion Model Construction

#region Building Random Circuits
def build_circuit(num_qubits: int, depth_L: int, allowed_gates: set[str]):
    my_circuit = []
    for lnum in range(depth_L):
        layer = []
        for qnum in range(num_qubits):
            gate = str(np.random.choice(allowed_gates))
            layer.append((gate, qnum))
        my_circuit.append(layer)
    return Circuit(my_circuit)

def build_circuit_with_arbitrarily_random_single_qubit_gates(num_qubits: int, depth_L: int):

    my_circuit = []
    gate_name = "Gcustom"

    full_args = np.random.random((depth_L, num_qubits, 4)) * 4 * np.pi # Need to be in [0, 2 \pi] for the half angles.

    for lnum in range(depth_L):
        layer = []
        for qnum in range(num_qubits):
            gate = Label(gate_name, qnum, args=(full_args[lnum, qnum]))
            layer.append(gate)
        my_circuit.append(layer)
    return Circuit(my_circuit, num_lines=num_qubits)

def build_circuit_with_multiple_qubit_gates_with_designated_lanes(num_qubits: int, depth_L: int, lane_end_points: list[int], gates_to_qubits_used: dict[str, int]):

    assert lane_end_points[-1] <= num_qubits # if < then we have a lane from there to num_qubits.
    assert lane_end_points[0] > 0
    assert np.all(np.diff(lane_end_points) > 0) # then it is sorted in increasing order.

    if lane_end_points[-1] < num_qubits:
        lane_end_points.append(num_qubits)

    my_circuit = []
    n_qs_to_gates_avail = {}
    for key, val in gates_to_qubits_used.items():
        if val in n_qs_to_gates_avail:
            n_qs_to_gates_avail[val].append(key)
        else:
            n_qs_to_gates_avail[val] = [key]

    for lnum in range(depth_L):
        layer = []
        start_point = 0

        for lane_ep in lane_end_points:
            num_used: int = 0
            while num_used < (lane_ep - start_point):
                navail = (lane_ep - start_point) - num_used
                nchosen = 0
                if navail >= max(n_qs_to_gates_avail):
                    # we can use any gate
                    nchosen = np.random.randint(1, max(n_qs_to_gates_avail) + 1)
                else:
                    # we need to first choose how many to use.
                    nchosen = np.random.randint(1, navail + 1)
                gate = str(np.random.choice(n_qs_to_gates_avail[nchosen]))
                tmp = list(np.random.permutation(nchosen) + num_used + start_point) # Increase to offset.
                perm_of_qubits_used = [int(tmp[ind]) for ind in range(len(tmp))]
                if gate == "Gcustom":
                    layer.append(Label(gate, *perm_of_qubits_used, args=(np.random.random(4)*4*np.pi)))
                else:
                    layer.append((gate, *perm_of_qubits_used))
                num_used += nchosen

            if num_used > (lane_ep - start_point) + 1:
                print(num_used, f"lane ({start_point}, {lane_ep})")
                raise AssertionError("lane barrier is broken")
            
            start_point = lane_ep
        my_circuit.append(layer)
    return Circuit(my_circuit, num_lines=num_qubits)


def build_circuit_with_multiple_qubit_gates(num_qubits: int, depth_L: int, gates_to_qubits_used: dict[str, int], starting_qubit: int=0):

    my_circuit = []
    n_qs_to_gates_avail = {}
    for key, val in gates_to_qubits_used.items():
        if val in n_qs_to_gates_avail:
            n_qs_to_gates_avail[val].append(key)
        else:
            n_qs_to_gates_avail[val] = [key]

    for lnum in range(depth_L):
        layer = []
        num_used: int = 0
        while num_used < num_qubits:
            navail = num_qubits - num_used
            nchosen = 0
            if navail >= max(n_qs_to_gates_avail):
                # we can use any gate
                nchosen = np.random.randint(1, max(n_qs_to_gates_avail) + 1)
            else:
                # we need to first choose how many to use.
                nchosen = np.random.randint(1, navail + 1)
            gate = str(np.random.choice(n_qs_to_gates_avail[nchosen]))
            tmp = list(np.random.permutation(nchosen) + num_used) # Increase to offset.
            perm_of_qubits_used = [int(tmp[ind]) for ind in range(len(tmp))]
            if gate == "Gcustom":
                layer.append(Label(gate, *perm_of_qubits_used, args=(np.random.random(4)*4*np.pi)))
            else:
                layer.append((gate, *perm_of_qubits_used))
            num_used += nchosen

        my_circuit.append(layer)
    return Circuit(my_circuit, num_lines=num_qubits)

#endregion Building Random Circuits


#region Consistency of Probability
def test_tensor_product_single_unitaries_yield_right_results():

    num_qubits = 4

    under_test, expected_model = build_models_for_testing(num_qubits)

    circuitNone = Circuit([], num_lines=num_qubits)
    circuitX = Circuit([("Gxpi2", i) for i in range(num_qubits)], num_lines=num_qubits)
    circuitY = Circuit([("Gypi2", i) for i in range(num_qubits)], num_lines=num_qubits)
    circuitZ = Circuit([("Gzpi2", i) for i in range(num_qubits)], num_lines=num_qubits)
    circuitIdle = Circuit([("Gi", i) for i in range(num_qubits)], num_lines=num_qubits)

    for cir in [circuitNone, circuitX, circuitY, circuitZ, circuitIdle]:
        probs = under_test.probabilities(cir)
        exp = expected_model.probabilities(cir)

        assert_probability_densities_are_equal(probs, exp, cir)

def test_tensor_product_single_unitaries_random_collection_of_xyz():

    for qb in range(2, 6):

        under_test, expected_model = build_models_for_testing(qb)
        allowed_gates = ['Gxpi2', 'Gypi2', "Gzpi2", 'Gi']

        circuit100 = build_circuit(qb, 100, allowed_gates=allowed_gates)

        probs = under_test.probabilities(circuit100)
        exp = expected_model.probabilities(circuit100)

        assert_probability_densities_are_equal(probs, exp, circuit100)

def test_tensor_product_arbitrarily_random_rotations():

    for qb in range(2, 6):

        under_test, expected_model = build_models_for_testing(qb)

        circuit = build_circuit_with_arbitrarily_random_single_qubit_gates(qb, 10)

        probs = under_test.probabilities(circuit)
        exp = expected_model.probabilities(circuit)

        assert_probability_densities_are_equal(probs, exp, circuit)

def test_tensor_product_two_qubit_gates():

    num_qubits = 4

    under_test, expected_model = build_models_for_testing(num_qubits)


    circuitECR01 = Circuit([[("Gecr", 0,1), ("Gi", 2), ("Gzpi2", 3)]])    
    circuitECR10 = Circuit([[("Gecr", 1,0), ("Gi", 2), ("Gzpi2", 3)]])

    for cir in [circuitECR01, circuitECR10]:
        probs = under_test.probabilities(cir)
        exp = expected_model.probabilities(cir)

        assert_probability_densities_are_equal(probs, exp, cir)

def test_tensor_product_gates_with_implicit_idles():

    num_qubits = 5

    under_test, expected_model = build_models_for_testing(num_qubits)
    
    gatenames = ["Gxpi2", "Gypi2", "Gzpi2", "Gi"]
    for gate in gatenames:
        for i in range(num_qubits):
            cir = Circuit([[(gate, i)]], num_lines=num_qubits)

            probs = under_test.probabilities(cir)
            exp = expected_model.probabilities(cir)
            assert_probability_densities_are_equal(probs, exp, cir)

    # Now for the two qubit gates. Gecr and GCNOT

    # gatenames = ["Gecr", "Gcnot"]
    gatenames = ["Gecr"]
    for gate in gatenames:
        for i in range(num_qubits - 1):
            cir = Circuit([[(gate, i, i+1)]], num_lines=num_qubits)

            probs = under_test.probabilities(cir)
            exp = expected_model.probabilities(cir)
            assert_probability_densities_are_equal(probs, exp, cir)

            # Order swapped.
            cir = Circuit([[(gate, i+1, i)]], num_lines=num_qubits)

            probs = under_test.probabilities(cir)
            exp = expected_model.probabilities(cir)
            assert_probability_densities_are_equal(probs, exp, cir)

def test_tensor_product_multi_qubit_gates_arbitrarily_random_rotations():


    gates_to_used_qubits = {'Gxpi2': 1, 'Gypi2': 1, 'Gzpi2': 1, 'Gi': 1, 'Gcustom': 1, 'Gswap': 2, 'Gcnot': 2, 'Gecr': 2}
    for qb in range(3, 6):

        under_test, expected_model = build_models_for_testing(qb)

        circuit = build_circuit_with_multiple_qubit_gates(qb, 100, gates_to_qubits_used=gates_to_used_qubits)

        print(circuit)

        probs = under_test.probabilities(circuit)
        exp = expected_model.probabilities(circuit)

        assert_probability_densities_are_equal(probs, exp, circuit)

def test_tensor_product_multi_qubit_gates_with_structured_lanes():

    gates_to_used_qubits = {'Gxpi2': 1, 'Gypi2': 1, 'Gzpi2': 1, 'Gi': 1, 'Gswap': 2, 'Gcnot': 2, 'Gecr': 2}
    for qb in range(5, 6):

        lanes = [1, 2, 4]

        under_test, expected_model = build_models_for_testing(qb)

        circuit = build_circuit_with_multiple_qubit_gates_with_designated_lanes(qb, 100, lanes, gates_to_qubits_used=gates_to_used_qubits)

        probs = under_test.probabilities(circuit)
        exp = expected_model.probabilities(circuit)

        assert_probability_densities_are_equal(probs, exp, circuit)
#endregion Probabilities Consistency tests


#region D Probabilities Consistency Tests

def test_tensor_product_two_qubit_gates_dprobs():

    num_qubits = 4

    under_test, expected_model = build_models_for_testing(num_qubits)


    circuitECR01 = Circuit([[("Gecr", 0,1), ("Gi", 2), ("Gzpi2", 3)]])    
    circuitECR10 = Circuit([[("Gecr", 1,0), ("Gi", 2), ("Gzpi2", 3)]])

    for cir in [circuitECR01, circuitECR10]:
        probs = under_test.sim.dprobs(cir)
        exp = expected_model.sim.dprobs(cir)

        assert_probability_densities_are_equal(probs, exp, cir)


def test_tensor_product_single_unitaries_yield_right_results_dprobs():

    import importlib as _importlib
    from pygsti.forwardsims import MapForwardSimulator
    num_qubits = 2

    under_test, expected_model = build_models_for_testing(num_qubits)

    # circuitNone = Circuit([], num_lines=num_qubits)
    circuitX = Circuit([("Gxpi2", i) for i in range(num_qubits)], num_lines=num_qubits)
    circuitY = Circuit([("Gypi2", i) for i in range(num_qubits)], num_lines=num_qubits)
    circuitZ = Circuit([("Gzpi2", i) for i in range(num_qubits)], num_lines=num_qubits)
    circuitIdle = Circuit([("Gi", i) for i in range(num_qubits)], num_lines=num_qubits)

    circuits = [circuitX]
    for cir in circuits:
        under_test.sim = MapForwardSimulator()
        probs = under_test.sim.dprobs(cir)
        expected_model.sim.calclib = _importlib.import_module("pygsti.forwardsims.mapforwardsim_calc_generic")

        exp = expected_model.sim.dprobs(cir)

        assert_probability_densities_are_equal(probs, exp, cir)

test_tensor_product_single_unitaries_yield_right_results_dprobs()