import pickle

import sys
import numpy as np
import scipy.sparse as sps
import pygsti.modelmembers.operations as op
import pygsti.tools.internalgates as itgs
import pygsti.tools.lindbladtools as lt
import pygsti.tools.basistools as bt
import pygsti.tools.optools as gt
from pygsti.models.modelconstruction import create_spam_vector, create_operation
from pygsti.evotypes import Evotype
from pygsti.modelmembers.instruments import TPInstrument
from pygsti.modelmembers.states import FullState
from pygsti.models import ExplicitOpModel
from pygsti.baseobjs import statespace, basisconstructors as bc
from pygsti.models.gaugegroup import FullGaugeGroupElement, UnitaryGaugeGroupElement
from pygsti.baseobjs import Basis
from ..util import BaseCase, needs_cvxpy

SKIP_DIAMONDIST_ON_WIN = True


class OpBase:
    def setUp(self):
        ExplicitOpModel._strict = False
        self.gate = self.build_gate()

    def test_num_params(self):
        self.assertEqual(self.gate.num_params, self.n_params)

    def test_copy(self):
        gate_copy = self.gate.copy()
        self.assertArraysEqual(gate_copy.to_dense(), self.gate.to_dense())
        self.assertEqual(type(gate_copy), type(self.gate))

    def test_get_dimension(self):
        self.assertEqual(self.gate.dim, 4)

    def test_vector_conversion(self):
        v = self.gate.to_vector()
        self.gate.from_vector(v)
        # TODO assert correctness

    def test_has_nonzero_hessian(self):
        self.assertFalse(self.gate.has_nonzero_hessian())

    def test_torep(self):
        state = np.zeros((4, 1), 'd')
        state[0] = state[3] = 1.0
        self.gate._rep.acton(FullState(state)._rep)
        # TODO assert correctness

    def test_to_string(self):
        gate_as_str = str(self.gate)
        # TODO assert correctness

    def test_pickle(self):
        pklstr = pickle.dumps(self.gate)
        gate_pickle = pickle.loads(pklstr)
        self.assertEqual(type(gate_pickle), type(self.gate))
        self.assertArraysEqual(gate_pickle.to_dense(), self.gate.to_dense())

    def test_tosparse(self):
        sparseMx = self.gate.to_sparse()
        # TODO assert correctness

    def test_frobeniusdist(self):
        self.assertAlmostEqual(self.gate.frobeniusdist(self.gate), 0.0)
        self.assertAlmostEqual(self.gate.frobeniusdist_squared(self.gate), 0.0)
        # TODO test non-trivial case

    def test_jtracedist(self):
        self.assertAlmostEqual(self.gate.jtracedist(self.gate), 0.0)

    @needs_cvxpy
    def test_diamonddist(self):
        if SKIP_DIAMONDIST_ON_WIN and sys.platform.startswith('win'): return
        self.assertAlmostEqual(self.gate.diamonddist(self.gate), 0.0)

    def test_deriv_wrt_params(self):
        deriv = self.gate.deriv_wrt_params()
        self.assertEqual(deriv.shape, (self.gate.dim**2, self.n_params))
        # TODO assert correctness

    def test_hessian_wrt_params(self):
        try:
            hessian = self.gate.hessian_wrt_params()
            hessian = self.gate.hessian_wrt_params([1, 2], None)
            hessian = self.gate.hessian_wrt_params(None, [1, 2])
            hessian = self.gate.hessian_wrt_params([1, 2], [1, 2])
            # TODO assert correctness
        except NotImplementedError:
            pass  # ok if some classes don't implement this


class LinearOpTester(BaseCase):
    n_params = 0

    @staticmethod
    def build_gate():
        dim = 4
        evotype = Evotype.cast('default')
        state_space = statespace.default_space_for_dim(dim)
        # rep = evotype.create_dense_superop_rep(np.identity(dim, 'd'), state_space)
        #   ^ Original, failing line. My fix below.
        rep = evotype.create_dense_superop_rep(None, np.identity(dim, 'd'), state_space)
        return op.LinearOperator(rep, evotype)

    def setUp(self):
        ExplicitOpModel._strict = False
        self.gate = self.build_gate()

    def test_raise_on_invalid_method(self):
        mat = np.kron(np.array([[0, 1], [1, 0]], 'd'), np.eye(2))
        T = FullGaugeGroupElement(mat)
        with self.assertRaises(NotImplementedError):
            self.gate.transform_inplace(T)
        with self.assertRaises(NotImplementedError):
            self.gate.depolarize(0.05)
        with self.assertRaises(NotImplementedError):
            self.gate.rotate((0.01, 0, 0), 'gm')
        with self.assertRaises(NotImplementedError):
            self.gate.frobeniusdist_squared(self.gate)
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
            self.gate.set_dense(np.zeros((1, 1), 'd'))  # bad size

class MutableDenseOpBase(DenseOpBase):
    def test_set_value(self):
        M = np.asarray(self.gate.to_dense())  # gate as a matrix
        self.gate.set_dense(M)
        # TODO assert correctness

    def test_transform(self):
        gate_copy = self.gate.copy()
        T = FullGaugeGroupElement(np.identity(4, 'd'))
        gate_copy.transform_inplace(T)
        self.assertArraysAlmostEqual(gate_copy.to_dense(), self.gate.to_dense())
        # TODO test a non-trivial case

    def test_depolarize(self):
        dp = self.gate.depolarize(0.05)
        dp = self.gate.depolarize([0.05, 0.10, 0.15])
        # TODO assert correctness

    def test_rotate(self):
        self.gate.rotate([0.01, 0.02, 0.03], 'gm')
        # TODO assert correctness

class ImmutableDenseOpBase(DenseOpBase):
    def test_raises_on_set_value(self):
        M = np.asarray(self.gate.to_dense())  # gate as a matrix
        with self.assertRaises(ValueError):
            self.gate.set_dense(M)

    def test_raises_on_transform(self):
        T = FullGaugeGroupElement(np.identity(4, 'd'))
        with self.assertRaises((ValueError, NotImplementedError)):
            self.gate.transform_inplace(T)

class DenseOpTester(ImmutableDenseOpBase, BaseCase):
    n_params = 0

    @staticmethod
    def build_gate():
        return op.DenseOperator(np.zeros((4, 4)), None, 'default', state_space=None)

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

class StaticStdOpTester(BaseCase):
    def test_statevec(self):
        std_unitaries = itgs.standard_gatename_unitaries()

        for name, U in std_unitaries.items():
            if callable(U): continue  # skip unitary functions (create factories)
            try:
                svop = op.StaticStandardOp(name, 'pp', 'statevec', state_space=None)
            except ModuleNotFoundError:  # if 'statevec' isn't built (no cython)
                svop = op.StaticStandardOp(name, 'pp', 'statevec_slow', state_space=None)
            self.assertArraysAlmostEqual(svop._rep.to_dense('Hilbert'), U)

    def test_densitymx_svterm_cterm(self):
        std_unitaries = itgs.standard_gatename_unitaries()

        for evotype in ['default']:  # 'densitymx', 'svterm', 'cterm'
            for name, U in std_unitaries.items():
                if callable(U): continue  # skip unitary functions (create factories)
                dmop = op.StaticStandardOp(name, 'pp', evotype, state_space=None)
                self.assertArraysAlmostEqual(dmop._rep.to_dense('HilbertSchmidt'), gt.unitary_to_pauligate(U))

    def test_chp(self):
        std_chp_ops = itgs.standard_gatenames_chp_conversions()

        for name, ops in std_chp_ops.items():
            if not name.startswith('G'): continue  # currently the 'h', 'p', 'm' gates aren't "standard" yet because they lack unitaries
            chpop = op.StaticStandardOp(name, 'pp', 'chp', state_space=None)
            op_str = '\n'.join(ops)
            self.assertEqual('\n'.join(chpop._rep._chp_ops()), op_str)
        
    def test_raises_on_bad_values(self):
        with self.assertRaises(ValueError):
            op.StaticStandardOp('BadGate', 'pp', 'statevec')
        with self.assertRaises(ValueError):
            op.StaticStandardOp('BadGate', 'pp', 'densitymx')

        with self.assertRaises(ModuleNotFoundError):
            op.StaticStandardOp('Gi', 'pp', 'not_an_evotype')


class FullOpTester(MutableDenseOpBase, BaseCase):
    n_params = 16

    @staticmethod
    def build_gate():
        return create_operation("X(pi/8,Q0)", [('Q0',)], "gm", parameterization="full")

    def test_convert_to_linear(self):
        converted = op.convert(self.gate, "linear", "gm")
        self.assertArraysAlmostEqual(converted.to_dense(), self.gate.to_dense())

    def test_raises_on_unallowed_conversion(self):
        #with self.assertRaises(ValueError):
        #op.convert(self.gate, "linear", "gm")  # unallowed
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
        gate_full_B = op.FullArbitraryOp(np.identity(4, 'd'))

        numParams = gate_full_B.num_params
        v = gate_full_B.to_vector()
        gate_full_B.from_vector(v)
        deriv = gate_full_B.deriv_wrt_params()
        # TODO assert correctness


class LinearlyParamOpTester(MutableDenseOpBase, BaseCase):
    n_params = 16

    @staticmethod
    def build_gate():
        # 'I' was 'D', 'full' was 'linear'
        return create_operation("I(Q0)", [('Q0',)], "gm", parameterization="full")

    def test_constructor_raises_on_real_param_constraint_violation(self):
        baseMx = np.zeros((2, 2))
        parameterToBaseIndicesMap = {0: [(0, 0)], 1: [(1, 1)]}  # parameterize only the diag els
        with self.assertRaises(AssertionError):
            op.LinearlyParamArbitraryOp(baseMx, np.array([1.0 + 1j, 1.0]), parameterToBaseIndicesMap,
                                        real=True)  # must be real

    def test_build_from_scratch(self):
        # TODO what is actually being tested here?
        baseMx = np.zeros((4, 4))
        paramArray = np.array([1.0, 1.0])
        parameterToBaseIndicesMap = {0: [(0, 0)], 1: [(1, 1)]}  # parameterize only the diagonal els
        gate_linear_B = op.LinearlyParamArbitraryOp(baseMx, paramArray, parameterToBaseIndicesMap, real=True)
        with self.assertRaises(AssertionError):
            op.LinearlyParamArbitraryOp(baseMx, np.array([1.0 + 1j, 1.0]), parameterToBaseIndicesMap,
                                        real=True)  # must be real

        numParams = gate_linear_B.num_params
        v = gate_linear_B.to_vector()
        gate_linear_B.from_vector(v)
        deriv = gate_linear_B.deriv_wrt_params()
        # TODO assert correctness


class TPOpTester(MutableDenseOpBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        return create_operation("Y(pi/4,Q0)", [('Q0',)], "gm", parameterization="full TP")

    def test_convert(self):
        conv = op.convert(self.gate, "full", "gm")
        conv = op.convert(self.gate, "full TP", "gm")
        # TODO assert correctness

class AffineShiftOpTester(DenseOpBase, BaseCase):
    n_params = 3

    @staticmethod
    def build_gate():
        mat = np.array([[1,0,0,0],[.1, 1, 0, 0], [.1, 0, 1, 0], [.1, 0, 0, 1]])
        return op.AffineShiftOp(mat)

    def test_set_dense(self):
        M = np.asarray(self.gate.to_dense())  # gate as a matrix
        self.gate.set_dense(M)

    def test_transform(self):
        gate_copy = self.gate.copy()
        T = FullGaugeGroupElement(np.identity(4, 'd'))
        gate_copy.transform_inplace(T)
        self.assertArraysAlmostEqual(gate_copy.to_dense(), self.gate.to_dense())

class StaticOpTester(ImmutableDenseOpBase, BaseCase):
    n_params = 0

    @staticmethod
    def build_gate():
        return create_operation("Z(pi/3,Q0)", [('Q0',)], "gm", parameterization="static")

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
            mx, include_off_diags_in_degen_blocks=False,
            tp_constrained_and_unital=False
        )

    def test_include_off_diags_in_degen_blocks(self):
        mx = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, -1]], 'complex')
        # 2 degenerate real pairs of evecs => should add off-diag els
        g2 = op.EigenvalueParamDenseOp(
            mx, include_off_diags_in_degen_blocks=True, tp_constrained_and_unital=False
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
            mx, include_off_diags_in_degen_blocks=False,
            tp_constrained_and_unital=False
        )

    def test_include_off_diags_in_degen_blocks(self):
        mx = np.array([[1, -0.1, 0, 0],
                       [0.1, 1, 0, 0],
                       [0, 0, 1 + 1, -0.1],
                       [0, 0, 0.1, 1 + 1]], 'complex')
        # complex pairs of evecs => make sure combined parameters work

        g3 = op.EigenvalueParamDenseOp(
            mx, include_off_diags_in_degen_blocks=True, tp_constrained_and_unital=False
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
            mx, include_off_diags_in_degen_blocks=True, tp_constrained_and_unital=False
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


class LindbladErrorgenTester(BaseCase):

    def test_errgen_construction(self):
        from pygsti.models.gaugegroup import UnitaryGaugeGroupElement

        mx_basis = Basis.cast('pp', 4)
        basis = Basis.cast('pp', 4)
        X = basis['X']
        Y = basis['Y']
        Z = basis['Z']

        # Build known combination to project back to
        errgen = (0.1 * lt.create_elementary_errorgen('H', Z)
                  - 0.01 * lt.create_elementary_errorgen('H', X)
                  + 0.2 * lt.create_elementary_errorgen('S', X)
                  + 0.25 * lt.create_elementary_errorgen('S', Y)
                  + 0.05 * lt.create_elementary_errorgen('C', X, Y)
                  - 0.01 * lt.create_elementary_errorgen('A', X, Y))
        errgen = bt.change_basis(errgen, 'std', mx_basis)

        eg = op.LindbladErrorgen.from_error_generator(                                                                                                              
            errgen, "CPTPLND", 'pp', truncate=False, mx_basis="pp", evotype='default')
        self.assertTrue(np.allclose(eg.to_dense(), errgen))

        errgen_copy = eg.copy()

        T = UnitaryGaugeGroupElement(np.identity(4, 'd'))
        errgen_copy.transform_inplace(T)
        self.assertTrue(np.allclose(errgen_copy.to_dense(), eg.to_dense()))

    def test_errgen_construction_from_op(self):
        densemx = np.array([[0, 0, 0, 0],
                            [0.1, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, -1, 0]], 'd')
        eg = op.LindbladErrorgen.from_error_generator(
            densemx, "CPTPLND", 'pp', truncate=True, mx_basis="pp", evotype='default')
        errgen_copy = eg.copy()
        T = UnitaryGaugeGroupElement(np.identity(4, 'd'))
        errgen_copy.transform_inplace(T)
        self.assertTrue(np.allclose(errgen_copy.to_dense(), eg.to_dense()))


class LindbladErrorgenBase(OpBase):
    def test_has_nonzero_hessian(self):
        self.assertTrue(self.gate.has_nonzero_hessian())

    def test_transform(self):
        errgen_copy = self.gate.copy()
        T = UnitaryGaugeGroupElement(np.identity(4, 'd'))
        errgen_copy.transform_inplace(T)
        self.assertArraysAlmostEqual(errgen_copy.to_dense(), self.gate.to_dense())
        # TODO test a non-trivial case


class CPTPLindbladErrorgenTester(LindbladErrorgenBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        return op.LindbladErrorgen.from_operation_matrix(
            mx, "CPTPLND", "pp", truncate=True, mx_basis="pp", evotype='default'
        )


class DiagonalCPTPLindbladDenseOpTester(LindbladErrorgenBase, BaseCase):
    n_params = 6

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        return op.LindbladErrorgen.from_operation_matrix(
            mx, "H+S", 'pp', truncate=True, mx_basis="pp", evotype='default'
        )


class CPTPLindbladSparseOpTester(LindbladErrorgenBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        densemx = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, -1, 0]], 'd')
        sparsemx = sps.csr_matrix(densemx, dtype='d')
        return op.LindbladErrorgen.from_operation_matrix(
            sparsemx, "CPTPLND", 'pp', truncate=True, mx_basis="pp", evotype='default'
        )


#Maybe make this into another test - there's no more LindbladOp and the
# LindbladErrorgen doesn't include any post-factor
#class PostFactorCPTPLindbladSparseOpTester(LindbladErrorgenBase, BaseCase):
#    n_params = 12
#
#    @staticmethod
#    def build_gate():
#        densemx = np.array([[1, 0, 0, 0],
#                            [0, 1, 0, 0],
#                            [0, 0, 0, 1],
#                            [0, 0, -1, 0]], 'd')
#        sparsemx = sps.csr_matrix(densemx, dtype='d')
#        return op.LindbladErrorgen.from_operation_matrix(
#            None, unitary_postfactor=sparsemx, ham_basis="pp",
#            nonham_basis="pp", param_mode="cptp", nonham_mode="all",
#            truncate=True, mx_basis="pp"
#        )


class UnconstrainedLindbladDenseOpTester(LindbladErrorgenBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        ppBasis = Basis.cast("pp", 4)
        return op.LindbladErrorgen.from_operation_matrix(
            mx, "GLND", ppBasis, truncate=True, mx_basis="pp", evotype='default'
        )


class DiagonalUnconstrainedLindbladDenseOpTester(LindbladErrorgenBase, BaseCase):
    n_params = 6

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        ppMxs = bc.pp_matrices(2)

        return op.LindbladErrorgen.from_operation_matrix(
            mx, "H+s", ppMxs, truncate=True, mx_basis="pp", evotype='default'
        )


class UntruncatedLindbladDenseOpTester(LindbladErrorgenBase, BaseCase):
    n_params = 12

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        ppBasis = Basis.cast("pp", 4)
        return op.LindbladErrorgen.from_operation_matrix(
            mx, "GLND", ppBasis, truncate=False, mx_basis="pp", evotype='default'
        )


class ComposedOpTester(OpBase, BaseCase):
    n_params = 48

    @staticmethod
    def build_gate():
        mx = np.identity(4, 'd')
        mx2 = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, -1, 0]], 'd')
        evotype = 'default'
        state_space = None  # constructs a default based on size of mx
        gate = op.ComposedOp([
            op.StaticArbitraryOp(mx, evotype=evotype, state_space=state_space),
            op.FullArbitraryOp(mx, evotype=evotype, state_space=state_space),
            op.FullArbitraryOp(mx2, evotype=evotype, state_space=state_space),
            op.StaticArbitraryOp(mx, evotype=evotype, state_space=state_space),
            op.FullArbitraryOp(mx2, evotype=evotype, state_space=state_space)
        ])

        # TODO does this need to be done?
        dummyGS = ExplicitOpModel(['Q0'])
        dummyGS.operations['Gcomp'] = gate  # so to/from vector works
        dummyGS.to_vector()
        return gate


class EmbeddedDenseOpTester(OpBase, BaseCase):
    n_params = 16

    @staticmethod
    def build_gate():
        evotype = 'default'
        state_space = statespace.StateSpace.cast([('Q0',)])
        mx = np.identity(state_space.dim, 'd')
        return op.EmbeddedOp(state_space, ['Q0'], op.FullArbitraryOp(mx, evotype=evotype, state_space=None))

    def test_constructor_raises_on_state_space_label_mismatch(self):
        mx = np.identity(4, 'd')
        state_space = statespace.StateSpace.cast([('Q0',), ('Q1',)])
        evotype = 'default'
        with self.assertRaises(AssertionError):
            op.EmbeddedOp(state_space, ['Q0', 'Q1'], op.FullArbitraryOp(mx, evotype=evotype, state_space=None))


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
        evotype = 'default'
        inst = TPInstrument({'plus': op.FullArbitraryOp(Gmz_plus, evotype=evotype), 'minus': op.FullArbitraryOp(
            Gmz_minus, evotype=evotype)})
        return inst['plus']

    def test_vector_conversion(self):
        self.gate.to_vector()  # now to_vector is allowed

    def test_deriv_wrt_params(self):
        super(TPInstrumentOpTester, self).test_deriv_wrt_params()

        # XXX does this check anything meaningful?  EGN: yes, this checks that when I give deriv_wrt_params a length-1 list it's return value has the right shape.
        deriv = self.gate.deriv_wrt_params([0])
        self.assertEqual(deriv.shape[1], 1)


class StochasticNoiseOpTester(BaseCase):
    def test_instance(self):
        state_space = statespace.default_space_for_dim(4)
        sop = op.StochasticNoiseOp(state_space)

        sop.from_vector(np.array([0.1, 0.0, 0.0]))
        self.assertArraysAlmostEqual(sop.to_vector(), np.array([0.1, 0., 0.]))

        expected_mx = np.identity(4); expected_mx[2, 2] = expected_mx[3, 3] = 0.98  # = 2*(0.1^2)
        self.assertArraysAlmostEqual(sop.to_dense(), expected_mx)

        rho = create_spam_vector("0", "Q0", Basis.cast("pp", [4]))
        self.assertAlmostEqual(float(np.dot(rho.T, np.dot(sop.to_dense(), rho))),
                               0.99)  # b/c X dephasing w/rate is 0.1^2 = 0.01


class DepolarizeOpTester(BaseCase):
    def test_depol_noise_op(self):
        state_space = statespace.default_space_for_dim(4)
        dop = op.DepolarizeOp(state_space)

        dop.from_vector(np.array([0.1]))
        self.assertArraysAlmostEqual(dop.to_vector(), np.array([0.1]))

        expected_mx = np.identity(4); expected_mx[1, 1] = expected_mx[2, 2] = expected_mx[3, 3] = 0.96  # = 4*(0.1^2)
        self.assertArraysAlmostEqual(dop.to_dense(), expected_mx)

        rho = create_spam_vector("0", "Q0", Basis.cast("pp", [4]))
        # b/c both X and Y dephasing rates => 0.01 reduction
        self.assertAlmostEqual(float(np.dot(rho.T, np.dot(dop.to_dense(), rho))), 0.98)
