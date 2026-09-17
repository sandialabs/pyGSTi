import warnings
import numpy as np
import numpy.testing as npt
import scipy.linalg as la
import pytest

import pygsti
import pygsti.models.explicitmodel as mdl
from pygsti.baseobjs import ExplicitStateSpace
from pygsti.models.modelconstruction import create_explicit_model_from_expressions, create_operation
from pygsti.models.explicitmodel import transform_composed_model
from pygsti.models.gaugegroup import UnitaryGaugeGroupElement
from pygsti.modelmembers.instruments import Instrument, TPInstrument
from pygsti.modelmembers.operations import ComposedOp, EmbeddedOp, StaticArbitraryOp
from pygsti.modelpacks.legacy import std1Q_XYI as std
import pygsti.modelpacks.smq1Q_XYI as smq1Q_XYI
from pygsti.tools.optools import unitary_to_pauligate
from ..util import BaseCase


class ExplicitOpModelStrictAccessTester(BaseCase):
    def setUp(self):
        self.model = std.target_model().randomize_with_unitary(0.001, seed=1234)
        # Enable strict mode on this model instance only, so the test doesn't
        # mutate shared ExplicitOpModel class state (which would race with /
        # leak into other tests under parallel execution).
        self.model._strict = True

    def test_strict_access(self):
        #test strict mode, which forbids all these accesses
        with self.assertRaises(KeyError):
            self.model['identity'] = [1, 0, 0, 0]
        with self.assertRaises(KeyError):
            self.model['Gx'] = np.identity(4, 'd')
        with self.assertRaises(KeyError):
            self.model['E0'] = [1, 0, 0, 0]
        with self.assertRaises(KeyError):
            self.model['rho0'] = [1, 0, 0, 0]

        with self.assertRaises(KeyError):
            self.model['identity']
        with self.assertRaises(KeyError):
            self.model['Gx']
        with self.assertRaises(KeyError):
            self.model['E0']
        with self.assertRaises(KeyError):
            self.model['rho0']


class ExplicitOpModelToolTester(BaseCase):
    def setUp(self):
        mdl.ExplicitOpModel._strict = False
        # XXX can these be constructed directly?  EGN: yes, some model-construction tests should do it.
        self.model = create_explicit_model_from_expressions([('Q0',)], ['Gi', 'Gx', 'Gy'],
                                                            ["I(Q0)", "X(pi/2,Q0)", "Y(pi/2,Q0)"])

        self.gateset_2q = create_explicit_model_from_expressions(
            [('Q0', 'Q1')], ['Gix', 'Giy', 'Gxi', 'Gyi', 'Gcnot'],
            ["I(Q0):X(pi/2,Q1)", "I(Q0):Y(pi/2,Q1)", "X(pi/2,Q0):I(Q1)", "Y(pi/2,Q0):I(Q1)", "CX(pi,Q0,Q1)"])

    def test_randomize_with_unitary(self):
        gateset_randu = self.model.randomize_with_unitary(0.01)
        gateset_randu = self.model.randomize_with_unitary(0.01, seed=1234)
        # TODO assert correctness

    def test_rotate_1q(self):
        sslbls = ExplicitStateSpace("Q0")
        rotXPi = create_operation("X(pi,Q0)", sslbls, "pp")
        rotXPiOv2 = create_operation("X(pi/2,Q0)", sslbls, "pp")
        rotYPiOv2 = create_operation("Y(pi/2,Q0)", sslbls, "pp")
        gateset_rot = self.model.rotate((np.pi / 2, 0, 0))  # rotate all gates by pi/2 about X axis
        self.assertArraysAlmostEqual(gateset_rot['Gi'].to_dense(), rotXPiOv2.to_dense())
        self.assertArraysAlmostEqual(gateset_rot['Gx'].to_dense(), rotXPi.to_dense())
        self.assertArraysAlmostEqual(gateset_rot['Gx'].to_dense(), np.dot(rotXPiOv2.to_dense(), rotXPiOv2.to_dense()))
        self.assertArraysAlmostEqual(gateset_rot['Gy'].to_dense(), np.dot(rotXPiOv2.to_dense(), rotYPiOv2.to_dense()))

    def test_rotate_2q(self):
        gateset_2q_rot = self.gateset_2q.rotate(rotate=list(np.zeros(15, 'd')))
        gateset_2q_rot_same = self.gateset_2q.rotate(rotate=(0.01,) * 15)
        gateset_2q_randu = self.gateset_2q.randomize_with_unitary(0.01)
        gateset_2q_randu = self.gateset_2q.randomize_with_unitary(0.01, seed=1234)
        # TODO assert correctness

    def test_depolarize(self):
        Gi_dep = np.array([[1, 0, 0, 0],
                           [0, 0.9, 0, 0],
                           [0, 0, 0.9, 0],
                           [0, 0, 0, 0.9]], 'd')
        Gx_dep = np.array([[1, 0, 0, 0],
                           [0, 0.9, 0, 0],
                           [0, 0, 0, -0.9],
                           [0, 0, 0.9, 0]], 'd')
        Gy_dep = np.array([[1, 0, 0, 0],
                           [0, 0, 0, 0.9],
                           [0, 0, 0.9, 0],
                           [0, -0.9, 0, 0]], 'd')
        gateset_dep = self.model.depolarize(op_noise=0.1)
        self.assertArraysAlmostEqual(gateset_dep['Gi'].to_dense(), Gi_dep)
        self.assertArraysAlmostEqual(gateset_dep['Gx'].to_dense(), Gx_dep)
        self.assertArraysAlmostEqual(gateset_dep['Gy'].to_dense(), Gy_dep)

    def test_depolarize_with_spam_noise(self):
        gateset_spam = self.model.depolarize(spam_noise=0.1)
        self.assertAlmostEqual(float(np.dot(self.model['Mdefault']['0'].to_dense().T, self.model['rho0'].to_dense())), 1.0)
        # Since np.ndarray doesn't implement __round__... (assertAlmostEqual() doesn't work)
        # Compare the single element dot product result to 0.095 instead (coverting the array's contents ([[ 0.095 ]]) to a **python** float (0.095))
        # print("DEBUG gateset_spam = ")
        # print(gateset_spam['Mdefault']['0'].T)
        # print(gateset_spam['rho0'].T)
        # print(gateset_spam)
        # print(gateset_spam['Mdefault']['0'].T)
        # print(gateset_spam['rho0'].T)
        # not 0.905 b/c effecs aren't depolarized now
        self.assertAlmostEqual(np.dot(gateset_spam['Mdefault']['0'].to_dense().T, gateset_spam['rho0'].to_dense()).reshape(-1,)[0], 0.95)
        self.assertArraysAlmostEqual(gateset_spam['rho0'].to_dense(), 1 / np.sqrt(2) * np.array([1, 0, 0, 0.9]))
        #self.assertArraysAlmostEqual(gateset_spam['Mdefault']['0'], 1/np.sqrt(2)*np.array([1,0,0,0.9]).reshape(-1,1) ) #not depolarized now
        print(gateset_spam['Mdefault']['0'].to_dense())
        self.assertArraysAlmostEqual(gateset_spam['Mdefault']['0'].to_dense(), 1 / np.sqrt(2) * np.array([1, 0, 0, 1]))  # not depolarized now

    def test_random_rotate_1q(self):
        gateset_rand_rot = self.model.rotate(max_rotate=0.2)
        gateset_rand_rot = self.model.rotate(max_rotate=0.2, seed=1234)

    def test_random_rotate_2q(self):
        gateset_2q_rand_rot = self.gateset_2q.rotate(max_rotate=0.2)
        gateset_2q_rand_rot = self.gateset_2q.rotate(max_rotate=0.2, seed=1234)
        # TODO assert correctness

    def test_random_depolarize(self):
        gateset_rand_dep = self.model.depolarize(max_op_noise=0.1)
        gateset_rand_dep = self.model.depolarize(max_op_noise=0.1, seed=1234)
        # TODO assert correctness

    def test_random_depolarize_with_spam_noise(self):
        gateset_rand_spam = self.model.depolarize(max_spam_noise=0.1)
        gateset_rand_spam = self.model.depolarize(max_spam_noise=0.1, seed=1234)
        # TODO assert correctness

    def test_rotate_raises_on_bad_arg_spec(self):
        with self.assertRaises(ValueError):
            self.model.rotate(rotate=(0.2,) * 3, max_rotate=0.2)  # can't specify both
        with self.assertRaises(ValueError):
            self.model.rotate()  # must specify rotate or max_rotate
        with self.assertRaises(ValueError):
            self.gateset_2q.rotate(rotate=(0.2,) * 15, max_rotate=0.2)  # can't specify both
        with self.assertRaises(ValueError):
            self.gateset_2q.rotate()  # must specify rotate or max_rotate

    def test_rotate_raises_on_bad_dim(self):
        with self.assertRaises(AssertionError):
            self.model.rotate((1, 2, 3, 4))  # tuple must be length 3
        with self.assertRaises(AssertionError):
            self.gateset_2q.rotate(rotate=(0, 0, 0))  # wrong dimension model
        with self.assertRaises(AssertionError):
            self.gateset_2q.rotate((1, 2, 3, 4))  # tuple must be length 15
        with self.assertRaises(AssertionError):
            self.model.rotate(rotate=np.zeros(15, 'd'))  # wrong dimension model

    def test_rotate_raises_on_bad_type(self):
        with self.assertRaises(AssertionError):
            self.model.rotate("a string!")  # must be a 3-tuple
        with self.assertRaises(AssertionError):
            self.gateset_2q.rotate("a string!")  # must be a 15-tuple

    def test_depolarize_raises_on_bad_arg_spec(self):
        with self.assertRaises(ValueError):
            self.model.depolarize(op_noise=0.1, max_op_noise=0.1, spam_noise=0)  # can't specify both
        with self.assertRaises(ValueError):
            self.model.depolarize(spam_noise=0.1, max_spam_noise=0.1)  # can't specify both


def _make_cptplnd_model_with_instrument():
    """Build a 1Q CPTPLND model (has ComposedState/ComposedPOVM) with one Instrument."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        base = smq1Q_XYI.target_model('CPTPLND')
    # Two instrument members with distinct dense matrices (not particularly meaningful,
    # but they just need to be different from each other and from the gates).
    G0 = np.diag([0.5,  0.25, 0.0,  0.5])
    G1 = np.diag([0.5, -0.25, 0.0,  0.5])
    base.instruments['Iz'] = Instrument({'0': G0, '1': G1})
    return base


def _make_cptplnd_model_with_from_effects_instrument():
    """Like `_make_cptplnd_model_with_instrument`, but the instrument uses the
    effect-then-gate parameterization, whose members all share a single
    ComposedPOVM error map."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        base = smq1Q_XYI.target_model('CPTPLND')
    E0 = np.array([1., 0., 0.,  1.]) / np.sqrt(2)   # |0><0| as a pp superket
    E1 = np.array([1., 0., 0., -1.]) / np.sqrt(2)   # |1><1|
    base.instruments['Iz'] = Instrument.from_effects({'p0': E0, 'p1': E1}, base.basis)
    base.to_vector()  # force the parameter vector (and gpindices) to be built
    return base


def _povm_errormap_of(instrument, key):
    """The ComposedPOVM error map behind `instrument[key]`'s measurement effect, for
    an instrument member wrapped by `transform_composed_model` (i.e. one extra
    ComposedOp([U, member, invU]) layer) or for a bare member."""
    member = instrument[key]
    if len(member.factorops) == 3:
        member = member.factorops[1]  # unwrap the ComposedOp([U, member, invU])
    root_conj = member.factorops[0]                      # RootConjOperator
    composed_effect = root_conj.submembers()[0]          # ComposedPOVMEffect
    return composed_effect.submembers()[0]               # the shared error map


def _unitary_ggel(theta: float) -> UnitaryGaugeGroupElement:
    U = la.expm(theta * 1j * np.array([[1, -1],[-1, 1]]))
    return UnitaryGaugeGroupElement(unitary_to_pauligate(U))


class Test_TransformComposedModelInstrument(BaseCase):
    """Tests for transform_composed_model focusing on Instrument support."""

    def setUp(self):
        self.mdl = _make_cptplnd_model_with_instrument()
        self.ggel = _unitary_ggel(0.3)
        self.orig_inst = {
            ek: self.mdl.instruments['Iz'][ek].to_dense().copy()
            for ek in self.mdl.instruments['Iz'].keys()
        }

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    def test_instrument_keys_preserved(self):
        result = transform_composed_model(self.mdl, self.ggel)
        self.assertEqual(
            list(result.instruments['Iz'].keys()),
            list(self.mdl.instruments['Iz'].keys()),
        )

    def test_instrument_members_are_composed_ops(self):
        result = transform_composed_model(self.mdl, self.ggel)
        for ek in result.instruments['Iz'].keys():
            self.assertIsInstance(result.instruments['Iz'][ek], ComposedOp)

    def test_instrument_readonly_restored(self):
        # transform_composed_model temporarily clears _readonly and must restore it.
        result = transform_composed_model(self.mdl, self.ggel)
        self.assertTrue(result.instruments['Iz']._readonly)

    # ------------------------------------------------------------------
    # Correctness: ComposedOp([U, op, invU]) evaluates as invU @ op @ U
    # ------------------------------------------------------------------

    def test_instrument_member_matrices_correct(self):
        result = transform_composed_model(self.mdl, self.ggel)
        U_mx   = self.ggel.transform_matrix
        invU   = self.ggel.transform_matrix_inverse
        for ek, orig_op in self.orig_inst.items():
            expected     = invU @ orig_op @ U_mx
            result_dense = result.instruments['Iz'][ek].to_dense()
            npt.assert_allclose(result_dense, expected, atol=1e-12,
                                err_msg=f'instrument member {ek!r}')

    def test_identity_transform_leaves_instruments_unchanged(self):
        ggel_id = UnitaryGaugeGroupElement(np.eye(4))
        result = transform_composed_model(self.mdl, ggel_id)
        for ek, orig_op in self.orig_inst.items():
            npt.assert_allclose(result.instruments['Iz'][ek].to_dense(), orig_op,
                                atol=1e-12, err_msg=f'instrument member {ek!r}')

    # ------------------------------------------------------------------
    # Non-mutation: the original model must be untouched.
    # ------------------------------------------------------------------

    def test_original_model_unchanged(self):
        transform_composed_model(self.mdl, self.ggel)
        for ek, orig_op in self.orig_inst.items():
            npt.assert_allclose(self.mdl.instruments['Iz'][ek].to_dense(), orig_op,
                                atol=1e-12, err_msg=f'instrument member {ek!r}')

    # ------------------------------------------------------------------
    # Multiple instruments: all must be transformed.
    # ------------------------------------------------------------------

    def test_multiple_instruments_all_transformed(self):
        G2 = np.eye(4) * 0.3
        G3 = np.eye(4) * 0.7
        self.mdl.instruments['Iw'] = Instrument({'a': G2, 'b': G3})
        orig_iw = {
            ek: self.mdl.instruments['Iw'][ek].to_dense().copy()
            for ek in self.mdl.instruments['Iw'].keys()
        }
        result = transform_composed_model(self.mdl, self.ggel)
        U_mx = self.ggel.transform_matrix
        invU = self.ggel.transform_matrix_inverse
        for ek, orig_op in orig_iw.items():
            expected = invU @ orig_op @ U_mx
            npt.assert_allclose(result.instruments['Iw'][ek].to_dense(), expected,
                                atol=1e-12, err_msg=f'second instrument member {ek!r}')

    # ------------------------------------------------------------------
    # TPInstrument: uses transform_inplace rather than ComposedOp wrapping,
    # because TPInstrument's constrained parameterization (TPInstrumentOp
    # members with _construct_matrix) is incompatible with member replacement.
    # ------------------------------------------------------------------

    def test_tp_instrument_member_matrices_correct(self):
        G0 = np.diag([0.5, 0.0, 0.0,  0.5])
        G1 = np.diag([0.5, 0.0, 0.0, -0.5])
        self.mdl.instruments['Itp'] = TPInstrument({'0': G0, '1': G1})
        orig = {ek: self.mdl.instruments['Itp'][ek].to_dense().copy()
                for ek in self.mdl.instruments['Itp'].keys()}
        result = transform_composed_model(self.mdl, self.ggel)
        invU = self.ggel.transform_matrix_inverse
        U_mx = self.ggel.transform_matrix
        for ek, orig_op in orig.items():
            expected = invU @ orig_op @ U_mx
            npt.assert_allclose(result.instruments['Itp'][ek].to_dense(), expected,
                                atol=1e-12, err_msg=f'TPInstrument member {ek!r}')


class AutoEmbedTester(BaseCase):
    """
    Tests for `ExplicitOpModel._embed_operation`, i.e. the auto-embedding that
    `OrderedMemberDict.__setitem__` performs when a key carries state-space labels.
    """

    CNOT_01 = unitary_to_pauligate(np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 0, 1],
                                             [0, 0, 1, 0]], dtype=complex))
    CNOT_10 = unitary_to_pauligate(np.array([[1, 0, 0, 0],
                                             [0, 0, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 1, 0, 0]], dtype=complex))

    def setUp(self):
        pspec = pygsti.processors.QubitProcessorSpec(2, ('Gxpi2', 'Gypi2', 'Gcnot'),
                                                     availability={'Gcnot': 'all-permutations'})
        self.model = pygsti.models.modelconstruction.create_explicit_model(
            pspec, ideal_gate_type='static', ideal_spam_type='static')

    def test_subsystem_operation_is_embedded(self):
        # A one-qubit operation assigned under a two-qubit model's ('Gy', 1) key gets embedded
        # onto qubit 1, leaving qubit 0 alone.
        one_qubit_y = unitary_to_pauligate(la.expm(-1j * (np.pi / 4) * np.array([[0, -1j], [1j, 0]])))
        # Note the operation has to be handed over as a one-qubit ModelMember: a bare numpy array is
        # cast onto the *parent's* state space by OrderedMemberDict.cast_to_model_member.
        self.model.operations[('Gtest', 1)] = StaticArbitraryOp(
            one_qubit_y, 'pp', 'default', ExplicitStateSpace([1], [2]))
        stored = self.model.operations[('Gtest', 1)]
        self.assertIsInstance(stored, EmbeddedOp)
        npt.assert_allclose(stored.to_dense('HilbertSchmidt'),
                            np.kron(np.eye(4), one_qubit_y), atol=1e-12)

    def test_full_state_space_operation_is_stored_verbatim(self):
        # An operation that already spans the model's full state space is taken to be *already*
        # embedded, whatever the key's state-space labels say.  This is what lets an
        # ExplicitOpModel round-trip through copy() and serialization: both re-insert every
        # (already-embedded) operation under its original, possibly permuted, key.  Re-embedding
        # here would apply the permutation a second time on each round trip.
        self.model.operations[('Gcnot', 1, 0)] = self.CNOT_01
        npt.assert_allclose(self.model.operations[('Gcnot', 1, 0)].to_dense('HilbertSchmidt'),
                            self.CNOT_01, atol=1e-12)

    def test_construction_and_round_trips_agree(self):
        # The corollary that actually matters to users: the value `create_explicit_model` puts
        # under a permuted key is the correctly permuted gate, and it stays that way.
        npt.assert_allclose(self.model.operations[('Gcnot', 0, 1)].to_dense('HilbertSchmidt'),
                            self.CNOT_01, atol=1e-12)
        npt.assert_allclose(self.model.operations[('Gcnot', 1, 0)].to_dense('HilbertSchmidt'),
                            self.CNOT_10, atol=1e-12)
        for round_tripped in (self.model.copy(),
                              pygsti.models.ExplicitOpModel.loads(self.model.dumps())):
            npt.assert_allclose(round_tripped.operations[('Gcnot', 1, 0)].to_dense('HilbertSchmidt'),
                                self.CNOT_10, atol=1e-12)


class Test_TransformComposedModelPreservesParameterization(BaseCase):
    """`transform_composed_model` must be parameterization-preserving: the returned
    model has to have the *same* free parameters as its input, and must not share
    any parameterized object with it."""

    def setUp(self):
        self.mdl = _make_cptplnd_model_with_from_effects_instrument()
        self.inst_keys = list(self.mdl.instruments['Iz'].keys())

    def test_from_effects_instrument_shares_one_errormap(self):
        # Baseline for the tests below: the members really do share one error map.
        inst = self.mdl.instruments['Iz']
        self.assertIs(_povm_errormap_of(inst, self.inst_keys[0]),
                      _povm_errormap_of(inst, self.inst_keys[1]))

    def test_identity_transform_preserves_num_params(self):
        result = transform_composed_model(self.mdl, UnitaryGaugeGroupElement(np.eye(4)))
        self.assertEqual(result.num_params, self.mdl.num_params)

    def test_nontrivial_transform_preserves_num_params(self):
        result = transform_composed_model(self.mdl, _unitary_ggel(0.3))
        self.assertEqual(result.num_params, self.mdl.num_params)

    def test_transformed_instrument_still_shares_one_errormap(self):
        # The shared error map is what ties the effects together, and therefore what
        # enforces the instrument's trace preservation.  Duplicating it per member
        # would let the effects drift apart under re-optimization.
        inst = transform_composed_model(self.mdl, _unitary_ggel(0.3)).instruments['Iz']
        self.assertIs(_povm_errormap_of(inst, self.inst_keys[0]),
                      _povm_errormap_of(inst, self.inst_keys[1]))

    def test_transformed_instrument_stays_tp_under_reoptimization(self):
        # Perturb the transformed model's parameters; the instrument's members must
        # still sum to a trace-preserving channel.
        result = transform_composed_model(self.mdl, _unitary_ggel(0.3))
        v = result.to_vector()
        result.from_vector(v + 0.02 * np.random.default_rng(0).standard_normal(len(v)))
        total = sum(result.instruments['Iz'][ek].to_dense() for ek in self.inst_keys)
        trace_functional = np.array([np.sqrt(2), 0., 0., 0.])  # <<I| in the pp basis
        npt.assert_allclose(trace_functional @ total, trace_functional, atol=1e-9)

    def test_result_does_not_alias_input_model(self):
        # A rebuilt member holds a reference to the member it wraps, so the rebuild
        # must consume the *copy's* members: otherwise from_vector on the result
        # silently mutates the input model.
        result = transform_composed_model(self.mdl, _unitary_ggel(0.3))
        before = {k: np.array(self.mdl.operations[k].to_dense()) for k in self.mdl.operations}
        before.update({('rho', k): np.array(self.mdl.preps[k].to_dense()) for k in self.mdl.preps})
        before.update({('Iz', ek): np.array(self.mdl.instruments['Iz'][ek].to_dense())
                       for ek in self.inst_keys})
        v = result.to_vector()
        result.from_vector(v + 0.05 * np.random.default_rng(1).standard_normal(len(v)))
        for k in self.mdl.operations:
            npt.assert_allclose(self.mdl.operations[k].to_dense(), before[k], atol=1e-12,
                                err_msg=f'operation {k!r} was mutated via the transformed model')
        for k in self.mdl.preps:
            npt.assert_allclose(self.mdl.preps[k].to_dense(), before[('rho', k)], atol=1e-12,
                                err_msg=f'prep {k!r} was mutated via the transformed model')
        for ek in self.inst_keys:
            npt.assert_allclose(self.mdl.instruments['Iz'][ek].to_dense(), before[('Iz', ek)],
                                atol=1e-12,
                                err_msg=f'instrument member {ek!r} was mutated via the transformed model')
