import numpy as np

from pygsti.baseobjs.opcalc import compact_deriv
from pygsti.objects import polynomial as poly
from ..util import BaseCase


class CompactPolynomialTester(BaseCase):
    def test_compact_polys(self):
        # TODO break apart
        p = poly.Polynomial({(): 1.0, (1, 2): 2.0, (1, 1, 2): 3.0})
        v, c = p.compact()
        self.assertArraysAlmostEqual(v, np.array([3, 0, 2, 1, 2, 3, 1, 1, 2]))
        self.assertArraysAlmostEqual(c, np.array([1.0, 2.0, 3.0]))
        # 3x1^2 x2 + 2 x1x2 + 3

        q = poly.Polynomial({(): 4.0, (1, 1): 5.0, (2, 2, 3): 6.0})
        v2, c2 = q.compact()
        self.assertArraysAlmostEqual(v2, np.array([3, 0, 2, 1, 1, 3, 2, 2, 3]))
        self.assertArraysAlmostEqual(c2, np.array([4.0, 5.0, 6.0]))
        # 6x2^2 x3 + 5 x1^2 + 4

        v = np.concatenate((v, v2))
        c = np.concatenate((c, c2))
        c = np.ascontiguousarray(c, complex)

        vout, cout = compact_deriv(v, c, np.array([1, 2, 3]))
        compact_polys = poly.bulk_load_compact_polynomials(vout, cout, keep_compact=True)

        def assertCompactPolysEqual(vctups1, vctups2):
            for (v1, c1), (v2, c2) in zip(vctups1, vctups2):
                self.assertArraysAlmostEqual(v1, v2)  # integer arrays
                self.assertArraysAlmostEqual(c1, c2)  # complex arrays

        assertCompactPolysEqual(compact_polys,
                                ((np.array([2, 1, 2, 2, 1, 2]), np.array([2. + 0.j, 6. + 0.j])),
                                 (np.array([2, 1, 1, 2, 1, 1]), np.array([2. + 0.j, 3. + 0.j])),
                                    (np.array([0]), np.array([], dtype=np.complex128)),
                                    (np.array([1, 1, 1]), np.array([10. + 0.j])),
                                    (np.array([1, 2, 2, 3]), np.array([12. + 0.j])),
                                    (np.array([1, 2, 2, 2]), np.array([6. + 0.j]))))

        polys = poly.bulk_load_compact_polynomials(vout, cout)
        self.assertEqual(str(polys[0]), "6.000x1x2 + 2.000x2")
        self.assertEqual(str(polys[1]), "2.000x1 + 3.000x1^2")
        self.assertEqual(str(polys[2]), "0")
        self.assertEqual(str(polys[3]), "10.000x1")
        self.assertEqual(str(polys[4]), "12.000x2x3")
        self.assertEqual(str(polys[5]), "6.000x2^2")

        self.assertEqual(list(vout), [2, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 2, 2, 3, 1, 2, 2, 2])
        self.assertEqual(list(cout), [ 2.+0.j,  6.+0.j,  2.+0.j,  3.+0.j, 10.+0.j, 12.+0.j,  6.+0.j])
