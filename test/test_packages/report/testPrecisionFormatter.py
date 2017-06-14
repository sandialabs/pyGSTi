import unittest
import pygsti

# ppt format ommitted because the string it produces isn't necessarily in the correct order
#   -> if latex and html pass, it will likely pass as well
# text formatting has also been ommitted, since providing it with a precision has no effect

from ..testutils import BaseTestCase, compare_files, temp_files
from .testFormatter import FormatterBaseTestCase

class PrecisionTest(FormatterBaseTestCase):

    def setUp(self):
        super(PrecisionTest, self).setUp()

        headings   = [self.arbitraryNum]
        formatters = ['Normal']
        self.table = pygsti.report.table.ReportTable(headings, formatters)

    def test_precision_formatting(self):
        # Precise first
        for fmt in ['html', 'latex']: # text format ommitted - it doesn't care about precision :)
            self.assertEqual(self.precise[fmt], self.table.render(fmt, precision=6, polarprecision=3)[fmt])

        # Imprecise second
        for fmt in ['html', 'latex']:
            self.assertEqual(self.imprecise[fmt], self.table.render(fmt, precision=2, polarprecision=3)[fmt])

if __name__ == '__main__':
    unittest.main(verbosity=2)
