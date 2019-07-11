import numpy as np
import copy

from ..util import BaseCase

from pygsti.objects import reportableqty as rq


class NonHermitianReportableQtyTester(BaseCase):
    def test_hermitian_to_real_raises_on_non_hermitian(self):
        non_herm_mx = np.array([[1, 1j],
                                [0, 1]], 'complex')
        r = rq.ReportableQty(non_herm_mx)
        with self.assertRaises(ValueError):
            r.hermitian_to_real()


class ReportableQtyBase(object):
    def setUp(self):
        self.q = rq.ReportableQty(self.val)

    def test_string_rep(self):
        s = str(self.q)
        s = repr(self.q)
        # TODO assert correctness

    def test_arithmetic(self):
        r = self.q + 1.0
        r = self.q * 2.0
        r = self.q / 2.0
        # TODO assert correctness

    def test_copy(self):
        q_cpy = copy.copy(self.q)
        q_dcpy = copy.deepcopy(self.q)
        # TODO assert correctness

    def test_log(self):
        self.q.log()
        # TODO assert correctness

    def test_scale(self):
        self.q.scale(2.0)  # like *=
        # TODO assert correctness

    def test_real_imag(self):
        self.q.real()
        self.q.imag()
        # TODO assert correctness

    def test_absdiff(self):
        self.q.absdiff(1.0, separate_re_im=True)
        self.q.absdiff(1.0, separate_re_im=False)
        # TODO assert correctness

    def test_infidelity_diff(self):
        self.q.infidelity_diff(1.0)
        # TODO assert correctness

    def test_mod(self):
        self.q.mod(1.0)
        # TODO assert correctness

    def test_has_eb(self):
        self.assertFalse(self.q.has_eb())

    def test_accessors(self):
        self.q.get_value()
        self.q.get_err_bar()
        self.q.get_value_and_err_bar()
        # TODO assert correctness


class ReportableQtyErrorbarBase(ReportableQtyBase):
    def setUp(self):
        self.q = rq.ReportableQty(self.val, self.eb)

    def test_has_eb(self):
        self.assertTrue(self.q.has_eb())


class BasicReportableQtyData(object):
    val = 0.0
    eb = 0.01


class VectorReportableQtyData(object):
    val = np.ones(4, 'd')
    eb = 0.1 * np.ones(4, 'd')


class MatrixReportableQtyData(object):
    val = np.identity(4, 'd')
    eb = 0.1 * np.ones((4, 4), 'd')

    def test_hermitian_to_real(self):
        self.q.hermitian_to_real()
        # TODO assert correctness


# actual test cases are permutations of base case & data
class BasicReportableQtyTester(ReportableQtyBase, BasicReportableQtyData, BaseCase):
    pass


class BasicReportableQtyErrorbarTester(ReportableQtyErrorbarBase, BasicReportableQtyData, BaseCase):
    pass


class VectorReportableQtyTester(ReportableQtyBase, VectorReportableQtyData, BaseCase):
    pass


class VectorReportableQtyErrorbarTester(ReportableQtyErrorbarBase, VectorReportableQtyData, BaseCase):
    pass


class MatrixReportableQtyTester(ReportableQtyBase, MatrixReportableQtyData, BaseCase):
    pass


class MatrixReportableQtyErrorbarTester(ReportableQtyErrorbarBase, MatrixReportableQtyData, BaseCase):
    pass
