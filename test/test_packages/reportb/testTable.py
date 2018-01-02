import unittest
import pygsti

from pygsti.report.table import ReportTable
from ..testutils         import BaseTestCase, compare_files, temp_files

# Covers some missing tests, but NOT all of report.table.py
class TestTable(BaseTestCase):
    def setUp(self):
        super(TestTable, self).setUp()

        self.headings   = [0.1]
        self.formatters = ['Normal']
        self.customHeadings = {'html'  : 'test',
                               'python': 'test',
                               'latex' : 'test'}

    def custom_headings(self, fmt):
        table = ReportTable(self.headings, self.formatters, self.customHeadings)
        table.render(fmt)

    def custom_headings_no_format(self, fmt):
        table = ReportTable(self.customHeadings, None)
        table.render(fmt)

    def standard_table(self, fmt):
        # Render
        table = ReportTable(self.customHeadings, ['Normal']*4) # Four formats
        table.addrow(['1.0'], ['Normal'])
        table.render(fmt)
        table.finish()

    # From testReport.py
    # Test ReportTable object
    def test_general(self):

        table = ReportTable(['1.0'], ['Normal'])
        table.addrow(['1.0'], ['Normal'])
        table.render('html')

        self.assertTrue(table.has_key('1.0'))

        rowLabels = list(table.keys())
        row1Data  = table[rowLabels[0]]
        colLabels = list(row1Data.keys())

        self.assertTrue(rowLabels, table.row_names)
        self.assertTrue(colLabels, table.col_names)
        self.assertTrue(len(rowLabels), table.num_rows)
        self.assertTrue(len(colLabels), table.num_cols)

        el00 = table[rowLabels[0]][colLabels[0]]
        self.assertTrue( rowLabels[0] in table )
        self.assertTrue( rowLabels[0] in table )

        table_len = len(table)
        self.assertEqual(table_len, table.num_rows)

        table_as_str = str(table)
        row1a = table.row(key=rowLabels[0])
        col1a = table.col(key=colLabels[0])
        row1b = table.row(index=0)
        col1b = table.col(index=0)
        self.assertEqual(row1a,row1b)
        self.assertEqual(col1a,col1b)

        with self.assertRaises(KeyError):
            table['foobar']
        with self.assertRaises(KeyError):
            table.row(key='foobar') #invalid key
        with self.assertRaises(ValueError):
            table.row(index=100000) #out of bounds
        with self.assertRaises(ValueError):
            table.row() #must specify key or index
        with self.assertRaises(ValueError):
            table.row(key='foobar',index=1) #cannot specify key and index
        with self.assertRaises(KeyError):
            table.col(key='foobar') #invalid key
        with self.assertRaises(ValueError):
            table.col(index=100000) #out of bounds
        with self.assertRaises(ValueError):
            table.col() #must specify key or index
        with self.assertRaises(ValueError):
            table.col(key='foobar',index=1) #cannot specify key and index

    # For supported custom headers
    def run_format(self, fmt):
        self.custom_headings(fmt)
        self.custom_headings_no_format(fmt)
        self.standard_table(fmt)

    # For unsupported custom headers
    def run_unsupported_custom_headers(self, fmt):
        with self.assertRaises(ValueError):
            self.custom_headings(fmt)
        self.custom_headings_no_format(fmt)
        self.standard_table(fmt)


    def test_html(self):
        self.run_format('html')

    def test_latex(self):
        self.run_format('latex')

    def test_unknown_format(self):
        with self.assertRaises(NotImplementedError):
            self.run_format('aksdjjfa')





if __name__ == '__main__':
    unittest.main(verbosity=2)
