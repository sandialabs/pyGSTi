import unittest
import pygsti

from .testFormatter import FormatterBaseTestCase

class EBFormatterTest(FormatterBaseTestCase):

    def setUp(self):
        super(EBFormatterTest, self).setUp()

        self.ebLatexString1 = '\\begin{tabular}[l]{|c|}\n\\hline\n$ \\begin{array}{c} %s \\\\ \\pm %s \\end{array} $ \\\\ \\hline\n\\end{tabular}\n'
        self.ebLatexString2 = '\\begin{tabular}[l]{|c|}\n\\hline\n%s \\\\ \\hline\n\\end{tabular}\n'

        self.ebPrecise1   = self.ebLatexString1 % (self.arbitraryNum, self.arbitraryNum)
        self.ebPrecise2   = self.ebLatexString2 % self.arbitraryNum
        self.ebImprecise1 = self.ebLatexString1 % (self.roundedNum, self.roundedNum)
        self.ebImprecise2 = self.ebLatexString2 % self.roundedNum

        formatters = ['Normal']
        self.ebtable1 = pygsti.report.table.ReportTable([(self.arbitraryNum, self.arbitraryNum)],
        formatters)
        self.ebtable2 = pygsti.report.table.ReportTable([(self.arbitraryNum, None)],
        formatters)

    def test_EB_formatter(self):
        self.assertEqual(self.ebPrecise1,   self.ebtable1.render('latex', precision=6)['latex'])
        self.assertEqual(self.ebPrecise2,   self.ebtable2.render('latex', precision=6)['latex'])
        self.assertEqual(self.ebImprecise1, self.ebtable1.render('latex', precision=2)['latex'])
        self.assertEqual(self.ebImprecise2, self.ebtable2.render('latex', precision=2)['latex'])

class PiEBFormatterTest(FormatterBaseTestCase):

    def setUp(self):
        super(PiEBFormatterTest, self).setUp()
        # Constant values for checking against
        self.piLatexString1 = '\\begin{tabular}[l]{|c|}\n\\hline\n$ \\begin{array}{c}(%s \\\\ \\pm %s)\\pi \\end{array} $ \\\\ \\hline\n\\end{tabular}\n'
        self.piLatexString2 = '\\begin{tabular}[l]{|c|}\n\\hline\n%s$\\pi$ \\\\ \\hline\n\\end{tabular}\n'

        self.piPrecise1   = self.piLatexString1 % (self.arbitraryNum, self.arbitraryNum)
        self.piPrecise2   = self.piLatexString2 % self.arbitraryNum
        self.piImprecise1 = self.piLatexString1 % (self.roundedNum, self.roundedNum)
        self.piImprecise2 = self.piLatexString2 % self.roundedNum

        formatters  = ['Pi'] # Just 'Pi' should work.... 
        self.piebtable1 = pygsti.report.table.ReportTable([(self.arbitraryNum, self.arbitraryNum)],
                                                            formatters)
        self.piebtable2 = pygsti.report.table.ReportTable([(self.arbitraryNum, None)],
                                                            formatters)

        # Pretend to create a confidence region that wants non markovian error bars
        class MockCRI(object):
            def __init__(self):
                self.nonMarkRadiusSq  = 1
        self.piebtable3 = pygsti.report.table.ReportTable([(self.arbitraryNum, .1)],
            formatters, confidenceRegionInfo=MockCRI())

    def test_PiEB_formatter(self):
        self.assertEqual(self.piPrecise1,   self.piebtable1.render('latex', precision=6)['latex'])
        self.assertEqual(self.piPrecise2,   self.piebtable2.render('latex', precision=6)['latex'])
        self.assertEqual(self.piImprecise1, self.piebtable1.render('latex', precision=2)['latex'])
        self.assertEqual(self.piImprecise2, self.piebtable2.render('latex', precision=2)['latex'])
        print(self.piebtable3.render('html', precision=6)['html'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
