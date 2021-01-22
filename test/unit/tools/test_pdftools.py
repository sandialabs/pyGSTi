from ..util import BaseCase
from pygsti.tools import pdftools


class PDFToolsTester(BaseCase):

    def test_pdf_tools(self):
        p = {'a': 0., 'b': 1.0}
        q = {'a': 0.5, 'b': 0.5}
        self.assertAlmostEqual(pdftools.tvd(p, q), .5)
        self.assertAlmostEqual(pdftools.classical_fidelity(p, q), .5)

        p = {'b': 1.0}
        q = {'a': 0.5, 'b': 0.5}
        self.assertAlmostEqual(pdftools.tvd(p, q), .5)
        self.assertAlmostEqual(pdftools.classical_fidelity(p, q), .5)

        p = {'b': 1.0}
        q = {'a': 1.0}
        self.assertAlmostEqual(pdftools.tvd(p, q), 1.)
        self.assertAlmostEqual(pdftools.classical_fidelity(p, q), 0.)

        p = {'a': 0., 'b': 1.0}
        q = {'a': 1.0, 'b': .0}
        self.assertAlmostEqual(pdftools.tvd(p, q), 1.)
        self.assertAlmostEqual(pdftools.classical_fidelity(p, q), 0.)

        p = {'a': 0., 'b': 1.0}
        q = {'a': 0., 'b': 1.0}
        self.assertAlmostEqual(pdftools.tvd(p, q), 0.)
        self.assertAlmostEqual(pdftools.classical_fidelity(p, q), 1.)
