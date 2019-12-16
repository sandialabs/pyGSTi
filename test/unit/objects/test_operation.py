import pickle
import numpy as np
import scipy.sparse as sps

from ..util import BaseCase, needs_cvxpy

from pygsti.objects import FullGaugeGroupElement, UnitaryGaugeGroupElement, \
    ExplicitOpModel, Basis, FullSPAMVec, TPInstrument
from pygsti.tools import basisconstructors as bc
import pygsti.construction as pc
import pygsti.objects.operation as op


class OpBase(object):
    def setUp(self):
        ExplicitOpModel._strict = False
        self.gate = self.build_gate()

    def test_num_params(self):
        self.assertEqual(self.gate.num_params(), self.n_params)

    def test_copy(self):
        gate_copy = self.gate.copy()
        self.assertArraysEqual(gate_copy, self.gate)
        self.assertEqual(type(gate_copy), type(self.gate))

    def test_get_dimension(self):
        self.assertEqual(self.gate.get_dimension(), 4)

    def test_vector_conversion(self):
        v = self.gate.to_vector()
        self.gate.from_vector(v)
        # TODO assert correctness

    def test_has_nonzero_hessian(self):
        self.assertFalse(self.gate.has_nonzero_hessian())

    def test_torep(self):
        state = np.zeros((4, 1), 'd')
        state[0] = state[3] = 1.0
        self.gate._rep.acton(FullSPAMVec(state, typ="prep")._rep)
        # TODO assert correctness

    def test_to_string(self):
        gate_as_str = str(self.gate)
        # TODO assert correctness

    def test_pickle(self):
        pklstr = pickle.dumps(self.gate)
        gate_pickle = pickle.loads(pklstr)
        self.assertArraysEqual(gate_pickle, self.gate)
        self.assertEqual(type(gate_pickle), type(self.gate))

    def test_tosparse(self):
        sparseMx = self.gate.tosparse()
        # TODO assert correctness


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

    def test_frobeniusdist(self):
        self.assertAlmostEqual(self.gate.frobeniusdist(self.gate), 0.0)
        self.assertAlmostEqual(self.gate.frobeniusdist2(self.gate), 0.0)
        # TODO test non-trivial case

    def test_jtracedist(self):
        self.assertAlmostEqual(self.gate.jtracedist(self.gate), 0.0)

    @needs_cvxpy
    def test_diamonddist(self):
        self.assertAlmostEqual(self.gate.diamonddist(self.gate), 0.0)

    def test_deriv_wrt_params(self):
        deriv = self.gate.deriv_wrt_params()
        self.assertEqual(deriv.shape, (self.gate.dim**2, self.n_params))
        # TODO assert correctness

    def test_hessian_wrt_params(self):
        hessian = self.gate.hessian_wrt_params()
        hessian = self.gate.hessian_wrt_params([1, 2], None)
        hessian = self.gate.hessian_wrt_params(None, [1, 2])
        hessian = self.gate.hessian_wrt_params([1, 2], [1, 2])
        # TODO assert correctness


class MutableDenseOpBase(DenseOpBase):
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


class ImmutableDenseOpBase(DenseOpBase):
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


class DenseOpTester(ImmutableDenseOpBase, BaseCase):
    n_params = 0

    @staticmethod
    def build_gate():
        return op.DenseOperator(np.zeros((4, 4)), 'densitymx')

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


class FullOpTester(MutableDenseOpBase, BaseCase):
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


class LinearlyParamOpTester(MutableDenseOpBase, BaseCase):
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


class TPOpTester(MutableDenseOpBase, BaseCase):
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


class StaticOpTester(ImmutableDenseOpBase, BaseCase):
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


class EigenvalueParamDenseOpBase(ImmutableDenseOpBase):
    pass


class RealEigenvalueParamDenseOpTester(EigenvalueParamDenseOpBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        return op.EigenvalueParamDenseOp(
            mx, includeOffDiagsInDegen2Blocks=False,
            TPconstrainedAndUnital=False
        )

    def test_include_off_diags_in_degen_2_blocks(self):
        mx = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, -1]], 'complex')
        # 2 degenerate real pairs of evecs => should add off-diag els
        g2 = op.EigenvalueParamDenseOp(
            mx, includeOffDiagsInDegen2Blocks=True, TPconstrainedAndUnital=False
        )
        self.assertEqual(
            g2.params,
            [[(1.0, (0, 0))], [(1.0, (1, 1))],
             [(1.0, (0, 1))], [(1.0, (1, 0))],  # off diags blk 1
             [(1.0, (2, 2))], [(1.0, (3, 3))],
             [(1.0, (2, 3))], [(1.0, (3, 2))]]  # off diags blk 2
        )


class ComplexEigenvalueParamDenseOpTester(EigenvalueParamDenseOpBase, BaseCase):
    n_params = 4

    @staticmethod
    def build_gate():
        mx = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, -1, 0]], 'd')
        return op.EigenvalueParamDenseOp(
            mx, includeOffDiagsInDegen2Blocks=False,
            TPconstrainedAndUnital=False
        )

    def test_include_off_diags_in_degen_2_blocks(self):
        mx = np.array([[1, -0.1, 0, 0],
                       [0.1, 1, 0, 0],
                       [0, 0, 1 + 1, -0.1],
                       [0, 0, 0.1, 1 + 1]], 'complex')
        # complex pairs of evecs => make sure combined parameters work
        g3 = op.EigenvalueParamDenseOp(
            mx, includeOffDiagsInDegen2Blocks=True, TPconstrainedAndUnital=False
        )
        self.assertEqual(
            g3.params,
            [[(1.0, (0, 0)), (1.0, (1, 1))],  # single param that is Re part of 0,0 and 1,1 els
             [(1j, (0, 0)), (-1j, (1, 1))],   # Im part of 0,0 and 1,1 els
             [(1.0, (2, 2)), (1.0, (3, 3))],  # Re part of 2,2 and 3,3 els
             [(1j, (2, 2)), (-1j, (3, 3))]]   # Im part of 2,2 and 3,3 els
        )

        # TODO I don't understand what edge case is being covered here
        mx = np.array([[1, -0.1, 0, 0],
                       [0.1, 1, 0, 0],
                       [0, 0, 1, -0.1],
                       [0, 0, 0.1, 1]], 'complex')
        # 2 degenerate complex pairs of evecs => should add off-diag els
        g4 = op.EigenvalueParamDenseOp(
            mx, includeOffDiagsInDegen2Blocks=True, TPconstrainedAndUnital=False
        )
        self.assertArraysAlmostEqual(g4.evals, [1. + 0.1j, 1. + 0.1j, 1. - 0.1j, 1. - 0.1j])  # Note: evals are sorted!
        self.assertEqual(
            g4.params,
            [[(1.0, (0, 0)), (1.0, (2, 2))],  # single param that is Re part of 0,0 and 2,2 els (conj eval pair, since sorted)
             [(1j, (0, 0)), (-1j, (2, 2))],   # Im part of 0,0 and 2,2 els
             [(1.0, (1, 1)), (1.0, (3, 3))],  # Re part of 1,1 and 3,3 els
             [(1j, (1, 1)), (-1j, (3, 3))],   # Im part of 1,1 and 3,3 els
             [(1.0, (0, 1)), (1.0, (2, 3))],  # Re part of 0,1 and 2,3 els (upper triangle)
             [(1j, (0, 1)), (-1j, (2, 3))],   # Im part of 0,1 and 2,3 els (upper triangle); (0,1) and (2,3) must be conjugates
             [(1.0, (1, 0)), (1.0, (3, 2))],  # Re part of 1,0 and 3,2 els (lower triangle)
             [(1j, (1, 0)), (-1j, (3, 2))]]   # Im part of 1,0 and 3,2 els (lower triangle); (1,0) and (3,2) must be conjugates
        )


class LindbladOpBase(object):
    def test_has_nonzero_hessian(self):
        self.assertTrue(self.gate.has_nonzero_hessian())


class LindbladDenseOpBase(LindbladOpBase, MutableDenseOpBase):
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

    def test_convert(self):
        g = op.convert(self.gate, "CPTP", Basis.cast("pp", 4))
        # TODO assert correctness


class LindbladSparseOpBase(LindbladOpBase, OpBase):
    def assertArraysEqual(self, a, b):
        # Sparse LindbladOp does not support equality natively, so compare errorgen matrices
        self.assertEqual((a.errorgen.tosparse() != b.errorgen.tosparse()).nnz, 0)


class CPTPLindbladDenseOpTester(LindbladDenseOpBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        return op.LindbladDenseOp.from_operation_matrix(
            mx, unitaryPostfactor=None, ham_basis="pp",
            nonham_basis="pp", param_mode="cptp", nonham_mode="all",
            truncate=True, mxBasis="pp"
        )


class DiagonalCPTPLindbladDenseOpTester(LindbladDenseOpBase, BaseCase):
    n_params = 6

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        return op.LindbladDenseOp.from_operation_matrix(
            mx, unitaryPostfactor=None, ham_basis="pp",
            nonham_basis="pp", param_mode="cptp",
            nonham_mode="diagonal", truncate=True, mxBasis="pp"
        )


class CPTPLindbladSparseOpTester(LindbladSparseOpBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        densemx = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, -1, 0]], 'd')
        sparsemx = sps.csr_matrix(densemx, dtype='d')
        return op.LindbladOp.from_operation_matrix(
            sparsemx, unitaryPostfactor=None, ham_basis="pp",
            nonham_basis="pp", param_mode="cptp", nonham_mode="all",
            truncate=True, mxBasis="pp"
        )


class PostFactorCPTPLindbladSparseOpTester(LindbladSparseOpBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        densemx = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, -1, 0]], 'd')
        sparsemx = sps.csr_matrix(densemx, dtype='d')
        return op.LindbladOp.from_operation_matrix(
            None, unitaryPostfactor=sparsemx, ham_basis="pp",
            nonham_basis="pp", param_mode="cptp", nonham_mode="all",
            truncate=True, mxBasis="pp"
        )


class UnconstrainedLindbladDenseOpTester(LindbladDenseOpBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        ppBasis = Basis.cast("pp", 4)
        return op.LindbladDenseOp.from_operation_matrix(
            mx, unitaryPostfactor=None, ham_basis=ppBasis,
            nonham_basis=ppBasis, param_mode="unconstrained",
            nonham_mode="all", truncate=True, mxBasis="pp"
        )


class DiagonalUnconstrainedLindbladDenseOpTester(LindbladDenseOpBase, BaseCase):
    n_params = 6

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        ppMxs = bc.pp_matrices(2)
        return op.LindbladDenseOp.from_operation_matrix(
            mx, unitaryPostfactor=None, ham_basis=ppMxs,
            nonham_basis=ppMxs, param_mode="unconstrained",
            nonham_mode="diagonal", truncate=True, mxBasis="pp"
        )


class UntruncatedLindbladDenseOpTester(LindbladDenseOpBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        ppBasis = Basis.cast("pp", 4)
        return op.LindbladDenseOp.from_operation_matrix(
            mx, unitaryPostfactor=None, ham_basis=ppBasis,
            nonham_basis=ppBasis, param_mode="unconstrained",
            nonham_mode="all", truncate=False, mxBasis="pp"
        )


class ComposedDenseOpTester(ImmutableDenseOpBase, BaseCase):
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


class EmbeddedDenseOpTester(ImmutableDenseOpBase, BaseCase):
    n_params = 16

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        return op.EmbeddedDenseOp([('Q0',)], ['Q0'], op.FullDenseOp(mx))

    def test_constructor_raises_on_bad_state_space_label(self):
        mx = np.identity(4, 'd')
        with self.assertRaises(ValueError):
            op.EmbeddedOp([('L0', 'foobar')], ['Q0'], op.FullDenseOp(mx))

    def test_constructor_raises_on_state_space_label_mismatch(self):
        mx = np.identity(4, 'd')
        with self.assertRaises(ValueError):
            op.EmbeddedOp([('Q0',), ('Q1',)], ['Q0', 'Q1'], op.FullDenseOp(mx))


class TPInstrumentOpTester(ImmutableDenseOpBase, BaseCase):
    n_params = 28

    @staticmethod
    def build_gate():
        # XXX can this be constructed directly?  EGN: what do you mean?
        Gmz_plus = np.array([[0.5, 0, 0, 0.5],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0.5, 0, 0, 0.5]])
        Gmz_minus = np.array([[0.5, 0, 0, -0.5],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [-0.5, 0, 0, 0.5]])
        inst = TPInstrument({'plus': Gmz_plus, 'minus': Gmz_minus})
        return inst['plus']

    def test_vector_conversion(self):
        with self.assertRaises(ValueError):
            self.gate.to_vector()

    def test_deriv_wrt_params(self):
        super(TPInstrumentOpTester, self).test_deriv_wrt_params()

        # XXX does this check anything meaningful?  EGN: yes, this checks that when I give deriv_wrt_params a length-1 list it's return value has the right shape.
        deriv = self.gate.deriv_wrt_params([0])
        self.assertEqual(deriv.shape[1], 1)


class StochasticNoiseOpTester(BaseCase):
    def test_instance(self):
        sop = op.StochasticNoiseOp(4)

        sop.from_vector(np.array([0.1, 0.0, 0.0]))
        self.assertArraysAlmostEqual(sop.to_vector(), np.array([0.1, 0., 0.]))

        expected_mx = np.identity(4); expected_mx[2, 2] = expected_mx[3, 3] = 0.98  # = 2*(0.1^2)
        self.assertArraysAlmostEqual(sop.todense(), expected_mx)

        rho = pc.build_vector([4], ['Q0'], "0", 'pp')
        self.assertAlmostEqual(float(np.dot(rho.T, np.dot(sop.todense(), rho))),
                               0.99)  # b/c X dephasing w/rate is 0.1^2 = 0.01


class DepolarizeOpTester(BaseCase):
    def test_depol_noise_op(self):
        dop = op.DepolarizeOp(4)

        dop.from_vector(np.array([0.1]))
        self.assertArraysAlmostEqual(dop.to_vector(), np.array([0.1]))

        expected_mx = np.identity(4); expected_mx[1, 1] = expected_mx[2, 2] = expected_mx[3, 3] = 0.96  # = 4*(0.1^2)
        self.assertArraysAlmostEqual(dop.todense(), expected_mx)

        rho = pc.build_vector([4], ['Q0'], "0", 'pp')
        # b/c both X and Y dephasing rates => 0.01 reduction
        self.assertAlmostEqual(float(np.dot(rho.T, np.dot(dop.todense(), rho))), 0.98)
