
from typing import Literal
import numpy as np
from pygsti.baseobjs import qubitgraph as _qgraph
from pygsti.baseobjs import QubitSpace
from pygsti.models import modelconstruction as pgmc
from pygsti.processors import QubitProcessorSpec
from pygsti.modelmembers.states import ComposedState, ComputationalBasisState
from pygsti.modelmembers.povms import ComposedPOVM
from pygsti.modelmembers.operations import LindbladErrorgen, ExpErrorgenOp
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.tools import slicetools as _slct
from pygsti.modelmembers.operations import ComposedOp, EmbeddedOp
from pygsti.algorithms import BuiltinBasis
from pygsti.modelmembers.operations import create_from_unitary_mx

from ..util import BaseCase

#region Create Model


def make_spam(num_qubits):
    state_space = QubitSpace(num_qubits)
    max_weights = {'H': 1, 'S': 1, 'C': 1, 'A': 1}
    egbn_hamiltonian_only = CompleteElementaryErrorgenBasis(BuiltinBasis("PP", 4),
                                                            state_space,
                                                            ('H', ),
                                                            max_weights)

    rho_errgen_rates = {ell: 0.0 for ell in egbn_hamiltonian_only.labels}
    rho_lindblad = LindbladErrorgen.from_elementary_errorgens(rho_errgen_rates,
                                                              parameterization='H',
                                                              state_space=state_space,
                                                              evotype='densitymx')
    rho_errorgen = ExpErrorgenOp(rho_lindblad)
    rho_ideal = ComputationalBasisState([0] * num_qubits)
    rho = ComposedState(rho_ideal, rho_errorgen)

    povm_errgen_rates = {ell: 0.0 for ell in egbn_hamiltonian_only.labels}
    povm_linblad = LindbladErrorgen.from_elementary_errorgens(povm_errgen_rates,
                                                              parameterization='H',
                                                              state_space=state_space,
                                                              evotype='densitymx')

    measure = ComposedPOVM(ExpErrorgenOp(povm_linblad))

    return rho, measure


def make_target_model(num_qubits, independent_gates: bool = True):
    ps_geometry = _qgraph.QubitGraph.common_graph(
        num_qubits, geometry='line',
        directed=True, all_directions=True,
        qubit_labels=tuple(range(num_qubits))
    )
    u_ecr = 1 / np.sqrt(2) * np.array([[0, 0, 1, 1j],
                                       [0, 0, 1j, 1],
                                       [1, -1j, 0, 0],
                                       [-1j, 1, 0, 0]])

    gatenames = ["Gxpi2", "Gi", "Gecr"]
    ps = QubitProcessorSpec(
        num_qubits=num_qubits,
        gate_names=gatenames,
        nonstd_gate_unitaries={'Gecr': u_ecr},
        geometry=ps_geometry
    )
    gateerrs = {}
    basis = BuiltinBasis("PP", QubitSpace(1))
    egb1 = CompleteElementaryErrorgenBasis(basis, QubitSpace(1), ('H', 'S'))
    for gn in gatenames[:-1]:
        gateerrs[gn] = {ell: 0 for ell in egb1.labels}
    egb2 = CompleteElementaryErrorgenBasis(basis, QubitSpace(2), ('H', 'S'))
    gateerrs['Gecr'] = {ell: 0 for ell in egb2.labels}

    tmn = pgmc.create_crosstalk_free_model(ps, lindblad_error_coeffs=gateerrs, independent_gates=independent_gates)

    return tmn
#endregion Create Model


class TestRebuildParamVec(BaseCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.num_qubits = 2

    def setup_model_for_test(self, where: Literal["op", "sp", "povm"]):

        model = make_target_model(self.num_qubits)

        paramlbls = model._paramlbls.copy()
        obj_of_interest = None
        if where in ["povm", "sp"]:
            rho, povm = make_spam(self.num_qubits)
            if where == "sp":
                model.prep_blks['layers']['rho0'] = rho
                obj_of_interest = rho
            else:
                model.povm_blks['layers']['Mdefault'] = povm
                obj_of_interest = povm

        elif where == "op":
            gate_name = "Gypi2"
            egb1 = CompleteElementaryErrorgenBasis(BuiltinBasis("PP", QubitSpace(1)),
                                                   QubitSpace(1), ('H', 'S'))

            new_gate_errgen_rates = {ell: 0.0 for ell in egb1.labels}
            my_lindbladian = LindbladErrorgen.from_elementary_errorgens(new_gate_errgen_rates,
                                                                        parameterization='auto',
                                                                        state_space=QubitSpace(1),
                                                                        evotype='densitymx')

            comp = ComposedOp([create_from_unitary_mx(np.eye(2), "static standard",
                                                      stdname=gate_name),
                               ExpErrorgenOp(my_lindbladian)])
            comp1 = comp.copy()

            model.operation_blks["gates"][(gate_name, 0)] = comp
            model.operation_blks["gates"][(gate_name, 1)] = comp1

            # Add the embedded op in a layer as well.

            embedded = EmbeddedOp(QubitSpace(self.num_qubits), [0], comp)
            embedded1 = EmbeddedOp(QubitSpace(self.num_qubits), [1], comp1)
            model.operation_blks["layers"][(gate_name, 0)] = embedded
            model.operation_blks["layers"][(gate_name, 1)] = embedded1

            obj_of_interest = (comp, comp1)
        else:
            raise ValueError("Unexpected value for the type of new layer to add to the model.")

        model._rebuild_paramvec()

        return paramlbls, model, model._paramlbls.copy(), obj_of_interest

    def test_add_state_prep_after_creation_of_implicit_noisy_model(self):

        original_lbls, model, modified_lbls, rho = self.setup_model_for_test("sp")

        inds = _slct.indices_as_array(rho._gpindices)
        avail_inds = np.arange(model.num_params)
        cross_check = np.where(avail_inds[:, None] != inds[None, :], 1, 0)

        totals = np.sum(cross_check, axis=1)
        self.assertEqual(len(totals), model.num_params)

        used_inds = np.where(totals == len(inds))
        self.assertArraysEqual(original_lbls, modified_lbls[used_inds])

    def test_add_povm_after_creation_of_implicit_noisy_model(self):

        original_lbls, model, modified_lbls, povm = self.setup_model_for_test("povm")

        inds = _slct.indices_as_array(povm._gpindices)
        avail_inds = np.arange(model.num_params)
        cross_check = np.where(avail_inds[:, None] != inds[None, :], 1, 0)

        totals = np.sum(cross_check, axis=1)
        self.assertEqual(len(totals), model.num_params)

        used_inds = np.where(totals == len(inds))
        self.assertArraysEqual(original_lbls, modified_lbls[used_inds])

    def test_add_gate_operation_after_creation_of_implicit_noisy_model(self):

        original_lbls, model, modified_lbls, (comp, comp1) = self.setup_model_for_test("op")

        excluded_inds = np.array(list(_slct.to_array(comp.gpindices)) + list(_slct.to_array(comp1.gpindices)))
        avail_inds = np.arange(model.num_params)
        cross_check = np.where(avail_inds[:, None] != excluded_inds[None, :], 1, 0)
        totals = np.sum(cross_check, axis=1)
        self.assertEqual(len(totals), model.num_params)

        norm = np.linalg.norm(comp.gpindices_as_array() - comp1.gpindices_as_array(), 2)
        tol = 2**-53  # episilon machine for double precision.
        msg = "Composed Op copy is treated as the same object as the original in creation of gpindices."
        self.assertGreater(norm, tol, msg)

        used_inds = np.where(totals == len(excluded_inds))
        self.assertArraysEqual(original_lbls, modified_lbls[used_inds])
