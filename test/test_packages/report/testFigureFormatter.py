import unittest
import pygsti.report.plotting as pplt
import pygsti.construction    as pc
import pygsti

from .testFormatter      import FormatterBaseTestCase
from pygsti.construction import std1Q_XYI as std
from pygsti.report.table import ReportTable

class FigureFormatterTest(FormatterBaseTestCase):

    def setUp(self):
        super(FigureFormatterTest, self).setUp()

        self.figLatex = '\\begin{tabular}[l]{|c|}\n\hline\n\\vcenteredhbox{\includegraphics[width=100.00in,height=100.00in,keepaspectratio]{temp_test_files/test_figure.pdf}} \\\\ \hline\n\end{tabular}\n'

        stateSpace  = [2] # Hilbert space has dimension 2; density matrix is a 2x2 matrix
        spaceLabels = [('Q0',)] #interpret the 2x2 density matrix as a single qubit named 'Q0'
        gx          = pc.build_gate(stateSpace,spaceLabels,"X(pi/2,Q0)")
        reportFig   = pplt.gate_matrix_boxplot(gx, mxBasis="pp", mxBasisDims=2,
                                               xlabel="testX", ylabel="testY", title="mytitle",
                                               boxLabels=True)
        figInfo     = (reportFig, 'test_figure', 100, 100) # Fig, Name, Size, Size
        headings    = [figInfo]
        formatters  = ['Figure']
        self.table  = ReportTable(headings, formatters)

    def test_figure_formatting(self):
        self.assertEqual(self.table.render('latex', scratchDir='temp_test_files'), self.figLatex)

    def test_unsupplied_scratchdir(self):
        with self.assertRaises(ValueError):
            self.table.render('latex')

if __name__ == '__main__':
    unittest.main(verbosity=2)
