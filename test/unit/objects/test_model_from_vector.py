from pygsti.baseobjs import qubitgraph as _qgraph
from pygsti.baseobjs import QubitSpace
from pygsti.models import modelconstruction as pgmc
from pygsti.processors import QubitProcessorSpec
from pygsti.models import LocalNoiseModel, Model
from pygsti.modelmembers.states import ComposedState, ComputationalBasisState
from pygsti.modelmembers.povms import ComposedPOVM
from pygsti.modelmembers.operations import ComposedOp, LindbladErrorgen, ExpErrorgenOp
# from pygsti.tools.lindbladtools import random_error_generator_rates
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.data import simulate_data
from pygsti.circuits import CircuitList
from pygsti.circuits import Circuit

from pygsti.baseobjs import qubitgraph as _qgraph
from pygsti.protocols import ProtocolData, GateSetTomography, CircuitListsDesign
from pygsti.processors import QubitProcessorSpec


from pygsti.algorithms import BuiltinBasis

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


def make_target_model(num_qubits, independent_gates: bool = True):
    ps_geometry = _qgraph.QubitGraph.common_graph(
        num_qubits, geometry='line',
        directed=True, all_directions=True,
        qubit_labels=tuple(range(num_qubits))
    )
    u_ecr = 1/np.sqrt(2)*np.array([[0,0,1,1j],[0,0,1j,1],[1,-1j,0,0],[-1j,1,0,0]])
    gatenames = ['Gxpi2', 'Gypi2', 'Gi', 'Gii',  'Gecr']
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

    tmn = pgmc.create_crosstalk_free_model(ps, lindblad_error_coeffs=gateerrs, independent_gates=independent_gates)

    rho, M = make_spam(num_qubits)
    tmn.prep_blks['layers']['rho0'] = rho
    tmn.povm_blks['layers']['Mdefault'] = M
    tmn._rebuild_paramvec()
    
    # tmn._layer_rules.implicit_idle_mode = "pad_1Q"

    return tmn


def test_model_from_vector_passable_within_dprobs():

    num_qubits = 4

    model = make_target_model(num_qubits)

    vec = model.to_vector()

    circuit = Circuit([("Gxpi2", 0)], num_lines=num_qubits)

    dprobs = model.sim.bulk_dprobs([circuit])

