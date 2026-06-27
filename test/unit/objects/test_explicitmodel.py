import warnings
import numpy as np
import numpy.testing as npt
import scipy.linalg as la
import pytest

import pygsti.models.explicitmodel as mdl
from pygsti.baseobjs import ExplicitStateSpace
from pygsti.models.modelconstruction import create_explicit_model_from_expressions, create_operation
from pygsti.models.explicitmodel import transform_composed_model
from pygsti.models.gaugegroup import UnitaryGaugeGroupElement
from pygsti.modelmembers.instruments import Instrument, TPInstrument
from pygsti.modelmembers.operations import ComposedOp
from pygsti.modelpacks.legacy import std1Q_XYI as std
import pygsti.modelpacks.smq1Q_XYI as smq1Q_XYI
from pygsti.tools.optools import unitary_to_pauligate
from ..util import BaseCase


class ExplicitOpModelStrictAccessTester(BaseCase):
    def setUp(self):
        mdl.ExplicitOpModel._strict = True
        self.model = std.target_model().randomize_with_unitary(0.001, seed=1234)

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
