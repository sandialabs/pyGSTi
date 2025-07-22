from pygsti.baseobjs import qubitgraph as _qgraph
from pygsti.baseobjs import QubitSpace
from pygsti.models import modelconstruction as pgmc
from pygsti.processors import QubitProcessorSpec
from pygsti.modelmembers.states import ComposedState, ComputationalBasisState
from pygsti.modelmembers.povms import ComposedPOVM
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp
# from pygsti.tools.lindbladtools import random_error_generator_rates
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.circuits import Circuit

from pygsti.baseobjs import qubitgraph as _qgraph
from pygsti.tools import slicetools as _slct
from pygsti.processors import QubitProcessorSpec

from pygsti.modelmembers.operations import ComposedOp, EmbeddedOp

import numpy as np
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
    # gatenames = ['Gxpi2', 'Gypi2', 'Gi', 'Gii',  'Gecr']
    gatenames = ["Gxpi2", "Gi", "Gecr"]
    ps = QubitProcessorSpec(
        num_qubits=num_qubits,
        gate_names=gatenames,
        nonstd_gate_unitaries={'Gecr': u_ecr}, # 'Gii': np.eye(4)},
        geometry=ps_geometry
    )
    gateerrs = dict()
    basis = BuiltinBasis("PP", QubitSpace(1))
    egb1 = CompleteElementaryErrorgenBasis(basis, QubitSpace(1), ('H','S')) # XXXX From Riley's code, default_label_type='local')
    for gn in gatenames[:-1]:
        gateerrs[gn] = {ell: 0 for ell in egb1.labels}
    egb2 = CompleteElementaryErrorgenBasis(basis, QubitSpace(2), ('H','S')) # XXXX From Riley's code, default_label_type='local')
    gateerrs['Gecr'] = {ell: 0 for ell in egb2.labels}
    # gateerrs['Gii'] = gateerrs['Gecr']

    tmn = pgmc.create_crosstalk_free_model(ps, lindblad_error_coeffs=gateerrs, independent_gates=independent_gates)


    
    # tmn._layer_rules.implicit_idle_mode = "pad_1Q"

    return tmn


def test_add_state_prep_after_creation_of_implicit_noisy_model():

    num_qubits = 2

    model = make_target_model(num_qubits)

    paramlbls = model._paramlbls.copy()

    rho, M = make_spam(num_qubits)
    model.prep_blks['layers']['rho0'] = rho
    # model.povm_blks['layers']['Mdefault'] = M
    model._rebuild_paramvec()

    out_labels = model._paramlbls.copy()

    inds = _slct.indices_as_array(rho._gpindices)

    avail_inds = np.arange(model.num_params)

    cross_check = np.where(avail_inds[:, None] != inds[None, :], 1, 0)

    totals = np.sum(cross_check, axis=1)
    assert len(totals) == model.num_params

    used_inds = np.where(totals == len(inds))

    assert all(paramlbls == out_labels[used_inds])


def test_add_povm_after_creation_of_implicit_noisy_model():

    num_qubits = 2

    model = make_target_model(num_qubits)

    paramlbls = model._paramlbls.copy()

    rho, M = make_spam(num_qubits)
    # model.prep_blks['layers']['rho0'] = rho
    model.povm_blks['layers']['Mdefault'] = M
    model._rebuild_paramvec()

    out_labels = model._paramlbls.copy()

    inds = _slct.indices_as_array(M._gpindices)

    avail_inds = np.arange(model.num_params)

    cross_check = np.where(avail_inds[:, None] != inds[None, :], 1, 0)

    totals = np.sum(cross_check, axis=1)
    assert len(totals) == model.num_params

    used_inds = np.where(totals == len(inds))

    assert all(paramlbls == out_labels[used_inds])


def test_add_gate_operation_after_creation_of_implicit_noisy_model():

    from pygsti.modelmembers.operations import create_from_unitary_mx

    dummy_unitary = np.eye(2)

    num_qubits = 2

    model = make_target_model(num_qubits)
    paramlbls = model._paramlbls.copy()

    gate_name = "Gypi2"

    basis = BuiltinBasis("PP", QubitSpace(1))
    egb1 = CompleteElementaryErrorgenBasis(basis, QubitSpace(1), ('H','S')) # XXXX From Riley's code, default_label_type='local')
    
    new_gate_errgen_rates = {ell: 0.0 for ell in egb1.labels}

    my_lindbladian = LindbladErrorgen.from_elementary_errorgens(new_gate_errgen_rates, parameterization='auto',
                                                                state_space=QubitSpace(1),
                                                                evotype='densitymx')
    my_new_op = ExpErrorgenOp(my_lindbladian)

    standard_op = create_from_unitary_mx(dummy_unitary, "static standard", stdname=gate_name)

    comp = ComposedOp([standard_op, my_new_op])
    comp1 = comp.copy()

    model.operation_blks["gates"][(gate_name, 0)] = comp
    model.operation_blks["gates"][(gate_name, 1)] = comp1

    # Add the embedded op in a layer as well.

    embedded = EmbeddedOp(QubitSpace(num_qubits), [0], comp)
    embedded1 = EmbeddedOp(QubitSpace(num_qubits), [1], comp1)

    model.operation_blks["layers"][(gate_name, 0)] = embedded
    model.operation_blks["layers"][(gate_name, 1)] = embedded1

    model._rebuild_paramvec()

    out_labels = model._paramlbls.copy()

    excluded_inds = np.array(list(_slct.to_array(comp.gpindices)) + list(_slct.to_array(comp1.gpindices)))
    
    avail_inds = np.arange(model.num_params)

    cross_check = np.where(avail_inds[:, None] != excluded_inds[None, :], 1, 0)

    totals = np.sum(cross_check, axis=1)
    assert len(totals) == model.num_params

    assert not np.allclose(comp.gpindices_as_array(), comp1.gpindices_as_array())
    used_inds = np.where(totals == len(excluded_inds))

    assert np.all(paramlbls == out_labels[used_inds])

test_add_gate_operation_after_creation_of_implicit_noisy_model()