import itertools

import numpy as np
import scipy.linalg as spl
import pytest

from ..util import BaseCase

from pygsti.baseobjs import Label, BuiltinBasis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as LEEL
from pygsti.circuits import Circuit
from pygsti.processors import QubitProcessorSpec
from pygsti.models.modelconstruction import create_crosstalk_free_model
from pygsti.tools import basistools as _bt, lindbladtools as _lt
from pygsti.errorgenpropagation import mcmgadget as mg

PAULIS_2Q = [''.join(p) for p in itertools.product('IXYZ', repeat=2)][1:]
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
PAULI_MXS = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}


def _pauli_mx(s):
    mx = np.array([[1.0 + 0j]])
    for ch in s:
        mx = np.kron(mx, PAULI_MXS[ch])
    return mx


class MCMGadgetCircuitTester(BaseCase):

    def test_expand_single_mcm(self):
        c = Circuit([Label('Gxpi2', 0), Label('Iz', 0), Label('Gcphase', (0, 1))], line_labels=(0, 1))
        ce, infos = mg.expand_mcm_circuit(c)
        self.assertEqual(ce.line_labels, (0, 1, 2))
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].layer_index, 1)
        self.assertEqual(infos[0].data_qubit, 0)
        self.assertEqual(infos[0].virtual_qubit, 2)
        self.assertEqual(ce[1], Label('Gcnot', (0, 2)))
        self.assertEqual(ce[0], c[0])
        self.assertEqual(ce[2], c[2])
        self.assertEqual(mg.num_mcms(c), 1)

    def test_expand_ordering_and_multiqubit_labels(self):
        c = Circuit([Label([Label('Iz', 1), Label('Gxpi2', 0)]), Label('Iz', (0, 1)), Label('Iz', 0)], line_labels=(0, 1))
        ce, infos = mg.expand_mcm_circuit(c)
        self.assertEqual(mg.num_mcms(c), 4)
        self.assertEqual([info.data_qubit for info in infos], [1, 0, 1, 0])
        self.assertEqual([info.virtual_qubit for info in infos], [2, 3, 4, 5])
        self.assertEqual([info.layer_index for info in infos], [0, 1, 1, 2])
        self.assertEqual(ce.line_labels, (0, 1, 2, 3, 4, 5))
        # first layer keeps the parallel gate and replaces the MCM by a CNOT onto virtual qubit 2
        self.assertEqual(set(ce[0].components), {Label('Gcnot', (1, 2)), Label('Gxpi2', 0)})
        self.assertEqual(set(ce[1].components), {Label('Gcnot', (0, 3)), Label('Gcnot', (1, 4))})
        self.assertEqual(ce[2], Label('Gcnot', (0, 5)))

    def test_expand_string_labels_and_no_mcm(self):
        c = Circuit([Label('Gxpi2', 'Q0'), Label('Iz', 'Q1')], line_labels=('Q0', 'Q1'))
        ce, infos = mg.expand_mcm_circuit(c)
        self.assertEqual(ce.line_labels, ('Q0', 'Q1', 'v0'))
        self.assertEqual(infos[0].virtual_qubit, 'v0')
        c2 = Circuit([Label('Gxpi2', 0)], line_labels=(0, 1))
        ce2, infos2 = mg.expand_mcm_circuit(c2)
        self.assertEqual(ce2, c2)
        self.assertEqual(infos2, [])
        self.assertEqual(mg.virtual_qubit_labels(('v0', 'a'), 2), ['v1', 'v2'])

    def test_joint_outcome_to_bitstring(self):
        self.assertEqual(mg.joint_outcome_to_bitstring(('0', '1', '01')), '0101')
        self.assertEqual(mg.joint_outcome_to_bitstring(('01', '10')), '1001')
        self.assertEqual(mg.joint_outcome_to_bitstring(('11',)), '11')
        self.assertEqual(mg.joint_outcome_to_bitstring('10'), '10')
        with self.assertRaises(ValueError):
            mg.joint_outcome_to_bitstring(('p0', '01'))
        d = mg.joint_outcome_probabilities_to_bitstring_dict({('0', '01'): 0.25, ('1', '01'): 0.75})
        self.assertEqual(d, {'010': 0.25, '011': 0.75})


class MCMGadgetDenseTester(BaseCase):

    def test_elementary_errorgen_convention(self):
        # unnormalized-Pauli convention, matching LindbladErrorgen / errgenproptools
        for typ, bels, mxs in [('H', ('ZY',), [_pauli_mx('ZY')]), ('S', ('IX',), [_pauli_mx('IX')]),
                               ('C', ('XI', 'ZY'), [_pauli_mx('XI'), _pauli_mx('ZY')]),
                               ('A', ('XX', 'YX'), [_pauli_mx('XX'), _pauli_mx('YX')])]:
            expected = _bt.change_basis(_lt.create_elementary_errorgen(typ, *mxs), 'std', 'pp')
            self.assertArraysAlmostEqual(mg.elementary_errorgen_matrix(typ, bels), expected)

    def test_ideal_instrument(self):
        members = mg.mcm_gadget_instrument_members({})
        P0 = np.array([[.5, 0, 0, .5], [0, 0, 0, 0], [0, 0, 0, 0], [.5, 0, 0, .5]])
        P1 = np.array([[.5, 0, 0, -.5], [0, 0, 0, 0], [0, 0, 0, 0], [-.5, 0, 0, .5]])
        self.assertArraysAlmostEqual(members['0'], P0)
        self.assertArraysAlmostEqual(members['1'], P1)

    def test_readout_error_instrument(self):
        s = 0.05
        members = mg.mcm_gadget_instrument_members({('S', 'IX'): s})
        # trace preservation of the instrument as a whole
        self.assertArraysAlmostEqual((members['0'] + members['1'])[0, :], np.array([1, 0, 0, 0]))
        # X on the virtual qubit flips the readout with probability (1 - e^{-2s})/2 but leaves the state alone
        p_flip = (1 - np.exp(-2 * s)) / 2
        rho0 = np.array([1, 0, 0, 1]) / np.sqrt(2)  # |0><0|
        out0 = members['0'] @ rho0
        out1 = members['1'] @ rho0
        self.assertAlmostEqual(np.sqrt(2) * out0[0], 1 - p_flip)
        self.assertAlmostEqual(np.sqrt(2) * out1[0], p_flip)
        # post-measurement state is still |0><0| in both branches
        self.assertArraysAlmostEqual(out0 / (np.sqrt(2) * out0[0]), rho0)
        self.assertArraysAlmostEqual(out1 / (np.sqrt(2) * out1[0]), rho0)
        # first-order members agree with exact ones to O(s^2)
        fo = mg.mcm_gadget_instrument_members({('S', 'IX'): s}, first_order=True)
        self.assertLess(np.abs(fo['0'] - members['0']).max(), 2 * s**2)
        self.assertGreater(np.abs(fo['0'] - members['0']).max(), 0.1 * s**2)

    def test_crunch_map_ranks_and_gauge(self):
        all_keys = mg.all_gadget_errorgen_keys(('H', 'S', 'C', 'A'))
        self.assertEqual(len(all_keys), 240)
        F = mg.mcm_crunch_matrix(all_keys)
        self.assertEqual(F.shape, (32, 240))
        self.assertEqual(np.linalg.matrix_rank(F, tol=1e-9), 28)  # TP instrument deviations: 2*16 - 4
        h_keys = [('H', p) for p in PAULIS_2Q]
        s_keys = [('S', p) for p in PAULIS_2Q]
        self.assertEqual(np.linalg.matrix_rank(mg.mcm_crunch_matrix(h_keys), tol=1e-9), 10)
        self.assertEqual(np.linalg.matrix_rank(mg.mcm_crunch_matrix(s_keys), tol=1e-9), 3)
        self.assertEqual(mg.mcm_gauge_directions(h_keys + s_keys).shape, (30, 17))
        self.assertEqual(mg.mcm_identifiable_directions(h_keys + s_keys).shape, (30, 13))
        # Hamiltonian MCM gauge = span{ZI, IZ, ZZ, XX+YY, XY-YX}
        FH = mg.mcm_crunch_matrix(h_keys)
        for vec in [{'ZI': 1}, {'IZ': 1}, {'ZZ': 1}, {'XX': 1, 'YY': 1}, {'XY': 1, 'YX': -1}]:
            v = np.array([vec.get(p, 0.0) for p in PAULIS_2Q])
            self.assertLess(np.linalg.norm(FH @ v), 1e-10)
        # dephasing-type stochastic errors are invisible
        FS = mg.mcm_crunch_matrix(s_keys)
        for p in ('ZI', 'IZ', 'ZZ'):
            self.assertLess(np.linalg.norm(FS[:, PAULIS_2Q.index(p)]), 1e-10)
        # S_IX and S_IY (and S_ZX, S_ZY) generate the same deviation (pure readout error)
        for p in ('IY', 'ZX', 'ZY'):
            self.assertArraysAlmostEqual(FS[:, PAULIS_2Q.index(p)], FS[:, PAULIS_2Q.index('IX')])

    def test_fomgi_representatives_and_memberships(self):
        reps = mg.fomgi_representatives('all')
        self.assertEqual(len(reps), 28)
        F_rep = mg.mcm_crunch_matrix(list(reps.values()))
        self.assertEqual(np.linalg.matrix_rank(F_rep, tol=1e-9), 28)
        hs = mg.fomgi_representatives()
        self.assertEqual(len(hs), 13)
        self.assertEqual(set(mg.FOMGI_SECTORS[n] for n in hs), {'H', 'S'})
        # memberships from Tables 2, 4, 6 of arXiv:2602.03938
        q = mg.fomgi_quantities({('S', 'IX'): 1.0, ('S', 'IY'): 2.0, ('S', 'ZX'): 3.0, ('S', 'ZY'): 4.0})
        self.assertAlmostEqual(q['s_read'], 10.0)
        self.assertTrue(all(abs(v) < 1e-9 for k, v in q.items() if k != 's_read'))
        q = mg.fomgi_quantities({('S', 'XI'): 1.0, ('S', 'YI'): 1.0, ('S', 'XZ'): 1.0, ('S', 'YZ'): 1.0})
        self.assertAlmostEqual(q['s_prep'], 4.0)
        q = mg.fomgi_quantities({('S', 'XX'): 1.0, ('S', 'XY'): 1.0, ('S', 'YX'): 1.0, ('S', 'YY'): 1.0})
        self.assertAlmostEqual(q['s_meas'], 4.0)
        q = mg.fomgi_quantities({('H', 'XX'): 0.01, ('H', 'YY'): 0.004, ('H', 'YX'): 0.002, ('H', 'XY'): 0.003})
        self.assertAlmostEqual(q['r_meas_x'], 0.006)   # h_xx - h_yy
        self.assertAlmostEqual(q['r_meas_y'], 0.005)   # h_yx + h_xy
        q = mg.fomgi_quantities({('H', 'ZY'): 0.01, ('H', 'IY'): 0.02, ('H', 'ZX'): 0.03, ('H', 'IX'): 0.04,
                                 ('H', 'ZI'): 0.5, ('H', 'IZ'): 0.5, ('H', 'ZZ'): 0.5})  # last three are gauge
        self.assertAlmostEqual(q['w0'], 0.01)
        self.assertAlmostEqual(q['w1'], 0.02)
        self.assertAlmostEqual(q['w2'], 0.03)
        self.assertAlmostEqual(q['w3'], 0.04)
        self.assertTrue(all(abs(v) < 1e-9 for k, v in q.items() if k not in ('w0', 'w1', 'w2', 'w3')))
        # C/A generators trigger the full 28-quantity decomposition
        q = mg.fomgi_quantities({('A', 'XX', 'YX'): 0.01, ('S', 'XX'): 0.02})
        self.assertEqual(len(q), 28)
        self.assertAlmostEqual(q['a_meas'], 0.01)
        self.assertAlmostEqual(q['s_meas'], 0.02)

    def test_fomgi_ansatz_helpers(self):
        ans = mg.fomgi_ansatz()
        self.assertEqual(list(ans.keys()), [('S', 'XX'), ('S', 'XI'), ('S', 'IX'), ('H', 'XX'), ('H', 'YX'), ('H', 'XI'),
                                            ('H', 'XZ'), ('H', 'YI'), ('H', 'YZ'), ('H', 'ZY'), ('H', 'IY'), ('H', 'ZX'),
                                            ('H', 'IX')])
        self.assertTrue(all(v == 0.0 for v in ans.values()))
        ans = mg.fomgi_ansatz(rates={'s_read': 0.01, 'w0': -0.02})
        self.assertEqual(ans[('S', 'IX')], 0.01)
        self.assertEqual(ans[('H', 'ZY')], -0.02)
        with self.assertRaises(ValueError):
            mg.fomgi_ansatz(rates={'a_meas': 0.1})  # not an H/S quantity
        names = mg.fomgi_names_for_ansatz([('H', 'ZY'), ('S', 'IX'), ('S', 'IY'), LEEL('H', ('XX',))])
        self.assertEqual(names[('H', 'ZY')], 'w0')
        self.assertEqual(names[('S', 'IX')], 's_read')
        self.assertEqual(names[LEEL('H', ('XX',))], 'r_meas_x')
        self.assertNotIn(('S', 'IY'), names)
        # decomposition matrix reproduces fomgi_quantities and is exactly the identity on the representatives
        names, T = mg.fomgi_decomposition_matrix(list(ans.keys()))
        self.assertArraysAlmostEqual(T, np.eye(13))


class MCMGadgetModelTester(BaseCase):

    def setUp(self):
        self.pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2', 'Gcphase'], geometry='line')

    def _model(self, rates_by_qubit, product_qubits=None):
        mdl = create_crosstalk_free_model(self.pspec, simulator='matrix')
        for q, rates in rates_by_qubit.items():
            mg.insert_mcm_gadget_instrument(mdl, q, rates)
        if product_qubits is not None:
            mg.insert_mcm_gadget_instrument(mdl, product_qubits, rates_by_qubit)
        return mdl

    def test_inserted_instrument_probabilities(self):
        s_read, w = 0.04, 0.03
        # prepare |1> on qubit 0 and measure it: a pure readout error flips the outcome with prob. (1-e^{-2s})/2
        mdl = self._model({0: {('S', 'IX'): s_read}})
        probs = mdl.probabilities(Circuit([Label('Gxpi2', 0), Label('Gxpi2', 0), Label('Iz', 0)], line_labels=(0, 1)))
        p_flip = (1 - np.exp(-2 * s_read)) / 2
        self.assertAlmostEqual(probs[('1', '10')] + probs[('1', '11')], 1 - p_flip, places=10)
        self.assertAlmostEqual(probs[('0', '10')] + probs[('0', '11')], p_flip, places=10)
        # (unitary weakness H_ZY adds an O(w^2) readout error on top, cf. Fig. 4b of arXiv:2602.03938)
        mdl = self._model({0: {('S', 'IX'): s_read, ('H', 'ZY'): w}})
        probs = mdl.probabilities(Circuit([Label('Gxpi2', 0), Label('Gxpi2', 0), Label('Iz', 0)], line_labels=(0, 1)))
        extra = (probs[('0', '10')] + probs[('0', '11')]) - p_flip
        self.assertGreater(extra, 0.5 * w**2)
        self.assertLess(extra, 1.5 * w**2)
        # compare against the dense gadget members directly for a |+> input followed by a second Y/2 gate
        members = mg.mcm_gadget_instrument_members({('S', 'IX'): s_read, ('H', 'ZY'): w})
        circ = Circuit([Label('Gypi2', 0), Label('Iz', 0), Label('Gypi2', 0)], line_labels=(0, 1))
        probs = mdl.probabilities(circ)
        from pygsti.tools import optools as _ot
        from pygsti.tools.internalgates import standard_gatename_unitaries
        Gy = _ot.unitary_to_superop(standard_gatename_unitaries()['Gypi2'], 'pp')
        rho0 = np.array([1, 0, 0, 1]) / np.sqrt(2)
        E0 = np.array([1, 0, 0, 1]) / np.sqrt(2)
        E1 = np.array([1, 0, 0, -1]) / np.sqrt(2)
        for c in ('0', '1'):
            out = Gy @ members[c] @ Gy @ rho0
            self.assertAlmostEqual(probs[(c, '00')], E0 @ out, places=10)
            self.assertAlmostEqual(probs[(c, '10')], E1 @ out, places=10)

    def test_product_instrument_matches_parallel_single_mcms(self):
        rates = {0: {('S', 'IX'): 0.02, ('H', 'ZY'): 0.01}, 1: {('S', 'XI'): 0.03, ('H', 'IX'): 0.02}}
        mdl = self._model(rates, product_qubits=(0, 1))
        pre = [Label('Gxpi2', 0), Label('Gypi2', 1), Label('Gcphase', (0, 1))]
        post = [Label('Gypi2', 0)]
        p_prod = mdl.probabilities(Circuit(pre + [Label('Iz', (0, 1))] + post, line_labels=(0, 1)))
        p_par = mdl.probabilities(Circuit(pre + [Label([Label('Iz', 0), Label('Iz', 1)])] + post, line_labels=(0, 1)))
        self.assertEqual(len(p_prod), 16)
        for (c01, final), p in p_prod.items():
            self.assertAlmostEqual(p, p_par[(c01[0], c01[1], final)], places=12)
