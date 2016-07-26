import unittest
import pygsti

arbitraryNum = 1.819743 # Six digits after decimal
roundedNum   = 1.82     # Two digits after decimal

ebLatexString1 = '\\begin{tabular}[l]{|c|}\n\\hline\n$ \\begin{array}{c} %s \\\\ \\pm %s \\end{array} $ \\\\ \\hline\n\\end{tabular}\n'
ebLatexString2 = '\\begin{tabular}[l]{|c|}\n\\hline\n%s \\\\ \\hline\n\\end{tabular}\n'

ebPrecise1   = ebLatexString1 % (arbitraryNum, arbitraryNum)
ebPrecise2   = ebLatexString2 % arbitraryNum
ebImprecise1 = ebLatexString1 % (roundedNum, roundedNum)
ebImprecise2 = ebLatexString2 % roundedNum

# Constant values for checking against
piLatexString1 = '\\begin{tabular}[l]{|c|}\n\\hline\n$ \\begin{array}{c}(%s \\\\ \\pm %s)\\pi \\end{array} $ \\\\ \\hline\n\\end{tabular}\n'
piLatexString2 = '\\begin{tabular}[l]{|c|}\n\\hline\n%s$\\pi$ \\\\ \\hline\n\\end{tabular}\n'

piPrecise1   = piLatexString1 % (arbitraryNum, arbitraryNum)
piPrecise2   = piLatexString2 % arbitraryNum
piImprecise1 = piLatexString1 % (roundedNum, roundedNum)
piImprecise2 = piLatexString2 % roundedNum

class EBFormatterTest(unittest.TestCase):

    def setUp(self):
        formatters = ['ErrorBars']
        self.table1 = pygsti.report.table.ReportTable([(arbitraryNum, arbitraryNum)],
                                                     formatters)
        self.table2 = pygsti.report.table.ReportTable([(arbitraryNum, None)],
                                                     formatters)

    def test_EB_formatter(self):
        self.assertEqual(ebPrecise1,   self.table1.render('latex', precision=6))
        self.assertEqual(ebPrecise2,   self.table2.render('latex', precision=6))
        self.assertEqual(ebImprecise1, self.table1.render('latex', precision=2))
        self.assertEqual(ebImprecise2, self.table2.render('latex', precision=2))

class PiEBFormatterTest(unittest.TestCase):

    def setUp(self):
        formatters  = ['PiErrorBars']
        self.table1 = pygsti.report.table.ReportTable([(arbitraryNum, arbitraryNum)],
                                                     formatters)
        self.table2 = pygsti.report.table.ReportTable([(arbitraryNum, None)],
                                                     formatters)

    def test_PiEB_formatter(self):
        self.assertEqual(piPrecise1,   self.table1.render('latex', precision=6))
        self.assertEqual(piPrecise2,   self.table2.render('latex', precision=6))
        self.assertEqual(piImprecise1, self.table1.render('latex', precision=2))
        self.assertEqual(piImprecise2, self.table2.render('latex', precision=2))

if __name__ == '__main__':
    unittest.main(verbosity=2)
