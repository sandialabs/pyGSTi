import numpy as np
import pickle

from ..util import BaseCase, unittest
from .. import _SKIP_CVXPY

from pygsti.objects import FullGaugeGroupElement, UnitaryGaugeGroupElement, ExplicitOpModel, Basis, FullSPAMVec
from pygsti.baseobjs import basisconstructors as bc
import pygsti.construction as pc
import pygsti.objects.operation as op


def OperationUtilTester(BaseCase):
    def test_convert_to_matrix_raises_on_bad_dim(self):
        with self.assertRaises(ValueError):
            op.DenseOperator.convert_to_matrix(np.zeros((2, 2, 2), 'd'))

    def test_convert_to_matrix_raises_on_bad_shape(self):
        with self.assertRaises(ValueError):
            op.DenseOperator.convert_to_matrix(np.zeros((2, 4), 'd'))

    def test_convert_to_matrix_raises_on_bad_input(self):
        bad_mxs = ['akdjsfaksdf',
                   [[], [1, 2]],
                   [[[]], [[1, 2]]]]
        for bad_mx in bad_mxs:
            with self.assertRaises(ValueError):
                op.DenseOperator.convert_to_matrix(bad_mx)


class OpBase:
    def setUp(self):
        ExplicitOpModel._strict = False
        self.gate = self.build_gate()

    def test_num_params(self):
        self.assertEqual(self.gate.num_params(), self.n_params)

    def test_copy(self):
        gate_copy = self.gate.copy()
        self.assertArraysAlmostEqual(gate_copy, self.gate)
        self.assertEqual(type(gate_copy), type(self.gate))

    def test_get_dimension(self):
        self.assertEqual(self.gate.get_dimension(), 4)

    def test_vector_conversion(self):
        v = self.gate.to_vector()
        self.gate.from_vector(v)
        # TODO assert correctness

    def test_deriv_wrt_params(self):
        deriv = self.gate.deriv_wrt_params()
        self.assertEqual(deriv.shape, (self.gate.dim**2, self.n_params))
        # TODO assert correctness

    def test_hessian_wrt_params(self):
        hessian = self.gate.hessian_wrt_params()
        hessian = self.gate.hessian_wrt_params([1,2],None)
        hessian = self.gate.hessian_wrt_params(None,[1,2])
        hessian = self.gate.hessian_wrt_params([1,2],[1,2])
        # TODO assert correctness

    def test_has_nonzero_hessian(self):
        self.assertFalse(self.gate.has_nonzero_hessian())

    def test_torep(self):
        state = np.zeros((4, 1), 'd')
        state[0] = state[3] = 1.0
        self.gate.torep().acton(FullSPAMVec(state).torep("prep"))
        # TODO assert correctness

    def test_frobeniusdist(self):
        self.assertAlmostEqual(self.gate.frobeniusdist(self.gate), 0.0)
        self.assertAlmostEqual(self.gate.frobeniusdist2(self.gate), 0.0)
        # TODO test non-trivial case

    def test_jtracedist(self):
        self.assertAlmostEqual(self.gate.jtracedist(self.gate), 0.0)

    @unittest.skipIf(_SKIP_CVXPY, "skipping cvxpy tests")
    def test_diamonddist(self):
        self.assertAlmostEqual(self.gate.diamonddist(self.gate), 0.0)

    def test_to_string(self):
        gate_as_str = str(self.gate)
        # TODO assert correctness

    def test_pickle(self):
        pklstr = pickle.dumps(self.gate)
        gate_pickle = pickle.loads(pklstr)
        self.assertArraysAlmostEqual(gate_pickle, self.gate)
        self.assertEqual(type(gate_pickle), type(self.gate))


class LinearOpTester(OpBase):
    n_params = 0

    @staticmethod
    def build_gate():
        return op.LinearOperator(4, 'densitymx')

    def test_raise_on_invalid_method(self):
        T = FullGaugeGroupElement(np.array([[0, 1], [1, 0]], 'd'))
        with self.assertRaises(NotImplementedError):
            self.gate.transform(T)
        with self.assertRaises(NotImplementedError):
            self.gate.depolarize(0.05)
        with self.assertRaises(NotImplementedError):
            self.gate.rotate((0.01, 0, 0), 'gm')
        with self.assertRaises(NotImplementedError):
            self.gate.frobeniusdist2(self.gate)
        with self.assertRaises(NotImplementedError):
            self.gate.frobeniusdist(self.gate)
        with self.assertRaises(NotImplementedError):
            self.gate.jtracedist(self.gate)
        with self.assertRaises(NotImplementedError):
            self.gate.diamonddist(self.gate)


class DenseOpBase(OpBase):
    def setUp(self):
        ExplicitOpModel._strict = False
        self.gate = self.build_gate()

    def test_set_value_raises_on_bad_size(self):
        with self.assertRaises((ValueError, AssertionError)):
            self.gate.set_value(np.zeros((1, 1), 'd'))  # bad size

    def test_arithmetic(self):
        result = self.gate + self.gate
        self.assertEqual(type(result), np.ndarray)
        result = self.gate + (-self.gate)
        self.assertEqual(type(result), np.ndarray)
        result = self.gate - self.gate
        self.assertEqual(type(result), np.ndarray)
        result = self.gate - abs(self.gate)
        self.assertEqual(type(result), np.ndarray)
        result = 2 * self.gate
        self.assertEqual(type(result), np.ndarray)
        result = self.gate * 2
        self.assertEqual(type(result), np.ndarray)
        result = 2 / self.gate
        self.assertEqual(type(result), np.ndarray)
        result = self.gate / 2
        self.assertEqual(type(result), np.ndarray)
        result = self.gate // 2
        self.assertEqual(type(result), np.ndarray)
        result = self.gate**2
        self.assertEqual(type(result), np.ndarray)
        result = self.gate.transpose()
        self.assertEqual(type(result), np.ndarray)

        M = np.identity(4, 'd')

        result = self.gate + M
        self.assertEqual(type(result), np.ndarray)
        result = self.gate - M
        self.assertEqual(type(result), np.ndarray)
        result = M + self.gate
        self.assertEqual(type(result), np.ndarray)
        result = M - self.gate
        self.assertEqual(type(result), np.ndarray)


class MutableGateBase(DenseOpBase):
    def test_set_value(self):
        M = np.asarray(self.gate)  # gate as a matrix
        self.gate.set_value(M)
        # TODO assert correctness

    def test_transform(self):
        gate_copy = self.gate.copy()
        T = FullGaugeGroupElement(np.identity(4, 'd'))
        gate_copy.transform(T)
        self.assertArraysAlmostEqual(gate_copy, self.gate)
        # TODO test a non-trivial case

    def test_element_accessors(self):
        e1 = self.gate[1, 1]
        e2 = self.gate[1][1]
        self.assertAlmostEqual(e1, e2)

        s1 = self.gate[1, :]
        s2 = self.gate[1]
        s3 = self.gate[1][:]
        a1 = self.gate[:]
        self.assertArraysAlmostEqual(s1, s2)
        self.assertArraysAlmostEqual(s1, s3)

        s4 = self.gate[2:4, 1]

        self.gate[1, 1] = e1
        self.gate[1, :] = s1
        self.gate[1] = s1
        self.gate[2:4, 1] = s4

        result = len(self.gate)
        # TODO assert correctness

    def test_depolarize(self):
        dp = self.gate.depolarize(0.05)
        dp = self.gate.depolarize([0.05, 0.10, 0.15])
        # TODO assert correctness

    def test_rotate(self):
        self.gate.rotate([0.01, 0.02, 0.03], 'gm')
        # TODO assert correctness

    def test_compose(self):
        cgate = self.gate.compose(self.gate)
        # TODO assert correctness


class ImmutableGateBase(DenseOpBase):
    def test_raises_on_set_value(self):
        M = np.asarray(self.gate)  # gate as a matrix
        with self.assertRaises(ValueError):
            self.gate.set_value(M)

    def test_raises_on_transform(self):
        T = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises((ValueError, NotImplementedError)):
            self.gate.transform(T)

    def test_element_accessors(self):
        e1 = self.gate[1, 1]
        e2 = self.gate[1][1]
        self.assertAlmostEqual(e1, e2)

        s1 = self.gate[1, :]
        s2 = self.gate[1]
        s3 = self.gate[1][:]
        a1 = self.gate[:]
        self.assertArraysAlmostEqual(s1, s2)
        self.assertArraysAlmostEqual(s1, s3)

        s4 = self.gate[2:4, 1]

        result = len(self.gate)
        # TODO assert correctness


class DenseOpTester(ImmutableGateBase, BaseCase):
    n_params = 0

    @staticmethod
    def build_gate():
        return op.DenseOperator(np.zeros((4, 4)), 'densitymx')


class FullOpTester(MutableGateBase, BaseCase):
    n_params = 16

    @staticmethod
    def build_gate():
        return pc.build_operation([(4,)], [('Q0',)], "X(pi/8,Q0)", "gm", parameterization="full")

    def test_composition(self):
        gate_linear = LinearlyParamOpTester.build_gate()
        gate_tp = TPOpTester.build_gate()
        gate_static = StaticOpTester.build_gate()

        c = op.compose(self.gate, self.gate, "gm", "full")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, self.gate))
        self.assertEqual(type(c), op.FullDenseOp)

        c = op.compose(self.gate, gate_tp, "gm")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, gate_tp))
        self.assertEqual(type(c), op.FullDenseOp)

        c = op.compose(self.gate, gate_static, "gm")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, gate_static))
        self.assertEqual(type(c), op.FullDenseOp)

        c = op.compose(self.gate, gate_linear, "gm")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, gate_linear))
        self.assertEqual(type(c), op.FullDenseOp)

    def test_raises_on_unallowed_conversion(self):
        with self.assertRaises(ValueError):
            op.convert(self.gate, "linear", "gm")  # unallowed
        with self.assertRaises(ValueError):
            op.convert(self.gate, "foobar", "gm")

    def test_raise_on_bad_type_conversion(self):
        with self.assertRaises(TypeError):
            int(self.gate)
        with self.assertRaises(TypeError):
            int(self.gate)
        with self.assertRaises(TypeError):
            float(self.gate)
        with self.assertRaises(TypeError):
            complex(self.gate)

    def test_build_from_scratch(self):
        # TODO what is actually being tested here?
        gate_full_B = op.FullDenseOp([[1, 0], [0, 1]])

        numParams = gate_full_B.num_params()
        v = gate_full_B.to_vector()
        gate_full_B.from_vector(v)
        deriv = gate_full_B.deriv_wrt_params()
        # TODO assert correctness


class LinearlyParamOpTester(MutableGateBase, BaseCase):
    n_params = 16

    @staticmethod
    def build_gate():
        # 'I' was 'D', 'full' was 'linear'
        return pc.build_operation([(4,)], [('Q0',)], "I(Q0)", "gm", parameterization="full")

    def test_constructor_raises_on_real_param_constraint_violation(self):
        baseMx = np.zeros((2, 2))
        parameterToBaseIndicesMap = {0: [(0, 0)], 1: [(1, 1)]}  # parameterize only the diag els
        with self.assertRaises(AssertionError):
            op.LinearlyParamDenseOp(baseMx, np.array([1.0 + 1j, 1.0]),
                                    parameterToBaseIndicesMap, real=True)  # must be real

    def test_composition(self):
        gate_full = FullOpTester.build_gate()

        c = op.compose(self.gate, gate_full, "gm")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, gate_full))
        self.assertEqual(type(c), op.FullDenseOp)

        #c = op.compose(self.gate, gate_tp, "gm")
        #self.assertArraysAlmostEqual(c, np.dot(self.gate,gate_tp) )
        #self.assertEqual(type(c), op.TPDenseOp)

        #c = op.compose(self.gate, gate_static, "gm")
        #self.assertArraysAlmostEqual(c, np.dot(self.gate,gate_static) )
        #self.assertEqual(type(c), op.LinearlyParamDenseOp)

        #c = op.compose(self.gate, self.gate, "gm")
        #self.assertArraysAlmostEqual(c, np.dot(self.gate,self.gate) )
        #self.assertEqual(type(c), op.LinearlyParamDenseOp)

    def test_build_from_scratch(self):
        # TODO what is actually being tested here?
        baseMx = np.zeros((2, 2))
        paramArray = np.array([1.0, 1.0])
        parameterToBaseIndicesMap = {0: [(0, 0)], 1: [(1, 1)]}  # parameterize only the diagonal els
        gate_linear_B = op.LinearlyParamDenseOp(baseMx, paramArray, parameterToBaseIndicesMap, real=True)
        with self.assertRaises(AssertionError):
            op.LinearlyParamDenseOp(baseMx, np.array([1.0 + 1j, 1.0]),
                                    parameterToBaseIndicesMap, real=True)  # must be real

        numParams = gate_linear_B.num_params()
        v = gate_linear_B.to_vector()
        gate_linear_B.from_vector(v)
        deriv = gate_linear_B.deriv_wrt_params()
        # TODO assert correctness


class TPOpTester(MutableGateBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        return pc.build_operation([(4,)], [('Q0',)], "Y(pi/4,Q0)", "gm", parameterization="TP")

    def test_composition(self):
        gate_full = FullOpTester.build_gate()
        gate_static = StaticOpTester.build_gate()

        c = op.compose(self.gate, gate_full, "gm")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, gate_full))
        self.assertEqual(type(c), op.FullDenseOp)

        c = op.compose(self.gate, self.gate, "gm")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, self.gate))
        self.assertEqual(type(c), op.TPDenseOp)

        c = op.compose(self.gate, gate_static, "gm")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, gate_static))
        self.assertEqual(type(c), op.TPDenseOp)

        #c = op.compose(self.gate, gate_linear, "gm")
        #self.assertArraysAlmostEqual(c, np.dot(self.gate,gate_linear) )
        #self.assertEqual(type(c), op.TPDenseOp)

    def test_convert(self):
        conv = op.convert(self.gate, "full", "gm")
        conv = op.convert(self.gate, "TP", "gm")
        # TODO assert correctness

    def test_first_row_read_only(self):
        # check that first row is read-only
        e1 = self.gate[0, 0]
        with self.assertRaises(ValueError):
            self.gate[0, 0] = e1
        with self.assertRaises(ValueError):
            self.gate[0][0] = e1
        with self.assertRaises(ValueError):
            self.gate[0, :] = [e1, 0, 0, 0]
        with self.assertRaises(ValueError):
            self.gate[0][:] = [e1, 0, 0, 0]
        with self.assertRaises(ValueError):
            self.gate[0, 1:2] = [0]
        with self.assertRaises(ValueError):
            self.gate[0][1:2] = [0]


class StaticOpTester(ImmutableGateBase, BaseCase):
    n_params = 0

    @staticmethod
    def build_gate():
        return pc.build_operation([(4,)], [('Q0',)], "Z(pi/3,Q0)", "gm", parameterization="static")

    def test_compose(self):
        gate_full = FullOpTester.build_gate()
        gate_tp = TPOpTester.build_gate()

        c = op.compose(self.gate, gate_full, "gm")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, gate_full))
        self.assertEqual(type(c), op.FullDenseOp)

        c = op.compose(self.gate, gate_tp, "gm")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, gate_tp))
        self.assertEqual(type(c), op.TPDenseOp)

        c = op.compose(self.gate, self.gate, "gm")
        self.assertArraysAlmostEqual(c, np.dot(self.gate, self.gate))
        self.assertEqual(type(c), op.StaticDenseOp)

        #c = op.compose(self.gate, gate_linear, "gm")
        #self.assertArraysAlmostEqual(c, np.dot(self.gate,gate_linear) )
        #self.assertEqual(type(c), op.LinearlyParamDenseOp)

    def test_convert(self):
        conv = op.convert(self.gate, "static", "gm")
        # TODO assert correctness


class RealEigenvalueParamDenseOpTester(ImmutableGateBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        return op.EigenvalueParamDenseOp(mx, includeOffDiagsInDegen2Blocks=False, TPconstrainedAndUnital=False)


class ComplexEigenvalueParamDenseOpTester(ImmutableGateBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_gate():
        mx = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, -1, 0]], 'd')
        return op.EigenvalueParamDenseOp(mx, includeOffDiagsInDegen2Blocks=False, TPconstrainedAndUnital=False)


class LindbladOpBase(MutableGateBase):
    def test_has_nonzero_hessian(self):
        self.assertTrue(self.gate.has_nonzero_hessian())

    def test_transform(self):
        gate_copy = self.gate.copy()
        T = UnitaryGaugeGroupElement(np.identity(4, 'd'))
        gate_copy.transform(T)
        self.assertArraysAlmostEqual(gate_copy, self.gate)
        # TODO test a non-trivial case

    def test_element_accessors(self):
        e1 = self.gate[1, 1]
        e2 = self.gate[1][1]
        self.assertAlmostEqual(e1, e2)

        s1 = self.gate[1, :]
        s2 = self.gate[1]
        s3 = self.gate[1][:]
        a1 = self.gate[:]
        self.assertArraysAlmostEqual(s1, s2)
        self.assertArraysAlmostEqual(s1, s3)

        s4 = self.gate[2:4, 1]

        result = len(self.gate)
        # TODO assert correctness


class CPTPLindbladDenseOpTester(LindbladOpBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        return op.LindbladDenseOp.from_operation_matrix(
            mx, unitaryPostfactor=None, ham_basis="pp", nonham_basis="pp", param_mode="cptp", nonham_mode="all",
            truncate=True, mxBasis="pp"
        )


class DiagonalCPTPLindbladDenseOpTester(LindbladOpBase, BaseCase):
    n_params = 6

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        return op.LindbladDenseOp.from_operation_matrix(
            mx, unitaryPostfactor=None, ham_basis="pp", nonham_basis="pp", param_mode="cptp", nonham_mode="diagonal",
            truncate=True, mxBasis="pp"
        )


class UnconstrainedLindbladDenseOpTester(LindbladOpBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        ppBasis = Basis.cast("pp", 4)
        return op.LindbladDenseOp.from_operation_matrix(
            mx, unitaryPostfactor=None, ham_basis=ppBasis, nonham_basis=ppBasis, param_mode="unconstrained",
            nonham_mode="all", truncate=True, mxBasis="pp"
        )


class DiagonalUnconstrainedLindbladDenseOpTester(LindbladOpBase, BaseCase):
    n_params = 6

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        ppMxs = bc.pp_matrices(2)
        return op.LindbladDenseOp.from_operation_matrix(
            mx, unitaryPostfactor=None, ham_basis=ppMxs, nonham_basis=ppMxs, param_mode="unconstrained",
            nonham_mode="diagonal", truncate=True, mxBasis="pp"
        )


class UntruncatedLindbladDenseOpTester(LindbladOpBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        ppBasis = Basis.cast("pp", 4)
        return op.LindbladDenseOp.from_operation_matrix(
            mx, unitaryPostfactor=None, ham_basis=ppBasis, nonham_basis=ppBasis, param_mode="unconstrained",
            nonham_mode="all", truncate=False, mxBasis="pp"
        )


class ComposedDenseOpTester(ImmutableGateBase, BaseCase):
    n_params = 48

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        mx2 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, -1, 0]], 'd')
        gate = op.ComposedDenseOp([
            op.StaticDenseOp(mx),
            op.FullDenseOp(mx),
            op.FullDenseOp(mx2),
            op.StaticDenseOp(mx),
            op.FullDenseOp(mx2)
        ])

        # TODO does this need to be done?
        dummyGS = ExplicitOpModel(['Q0'])
        dummyGS.operations['Gcomp'] = gate  # so to/from vector works
        dummyGS.to_vector()
        return gate


class EmbeddedDenseOpTester(ImmutableGateBase, BaseCase):
    n_params = 16

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        return op.EmbeddedDenseOp([('Q0',)], ['Q0'], op.FullDenseOp(mx))
