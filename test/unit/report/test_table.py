from collections import defaultdict

from ..util import BaseCase

from pygsti.report.table import ReportTable


class TableInstanceTester(BaseCase):
    custom_headings = {
        'html': 'test',
        'python': 'test',
        'latex': 'test'
    }

    def setUp(self):
        self.table = ReportTable(self.custom_headings, ['Normal'] * 4)  # Four formats

    def test_element_accessors(self):
        self.table.addrow(['1.0'], ['Normal'])

        self.assertTrue('1.0' in self.table)

        self.assertEqual(len(self.table), self.table.num_rows)

        row_by_key = self.table.row(key=self.table.row_names[0])
        row_by_idx = self.table.row(index=0)
        self.assertEqual(row_by_key, row_by_idx)

        col_by_key = self.table.col(key=self.table.col_names[0])
        col_by_idx = self.table.col(index=0)
        self.assertEqual(col_by_key, col_by_idx)

    def test_to_string(self):
        s = str(self.table)
        # TODO assert correctness

    def test_render_HTML(self):
        self.table.addrow(['1.0'], ['Normal'])
        self.table.addrow(['1.0'], ['Normal'])
        render = self.table.render('html')
        # TODO assert correctness

    def test_render_LaTeX(self):
        self.table.addrow(['1.0'], ['Normal'])
        self.table.addrow(['1.0'], ['Normal'])
        render = self.table.render('latex')
        # TODO assert correctness

    def test_finish(self):
        self.table.addrow(['1.0'], ['Normal'])
        self.table.finish()
        # TODO assert correctness

    def test_render_raises_on_unknown_format(self):
        with self.assertRaises(NotImplementedError):
            self.table.render('foobar')

    def test_raise_on_invalid_accessor(self):
        # XXX are these neccessary?  EGN: maybe not - checks invalid inputs, which maybe shouldn't need testing?
        with self.assertRaises(KeyError):
            self.table['foobar']
        with self.assertRaises(KeyError):
            self.table.row(key='foobar')  # invalid key
        with self.assertRaises(ValueError):
            self.table.row(index=100000)  # out of bounds
        with self.assertRaises(ValueError):
            self.table.row()  # must specify key or index
        with self.assertRaises(ValueError):
            self.table.row(key='foobar', index=1)  # cannot specify key and index
        with self.assertRaises(KeyError):
            self.table.col(key='foobar')  # invalid key
        with self.assertRaises(ValueError):
            self.table.col(index=100000)  # out of bounds
        with self.assertRaises(ValueError):
            self.table.col()  # must specify key or index
        with self.assertRaises(ValueError):
            self.table.col(key='foobar', index=1)  # cannot specify key and index


class CustomHeadingTableTester(TableInstanceTester):
    def setUp(self):
        self.table = ReportTable([0.1], ['Normal'], self.custom_headings)

    def test_labels(self):
        self.table.addrow(['1.0'], ['Normal'])
        self.assertTrue('1.0' in self.table)

        rowLabels = list(self.table.keys())
        self.assertEqual(rowLabels, self.table.row_names)
        self.assertEqual(len(rowLabels), self.table.num_rows)
        self.assertTrue(rowLabels[0] in self.table)

        row1Data = self.table[rowLabels[0]]
        colLabels = list(row1Data.keys())
        self.assertEqual(colLabels, self.table.col_names)
        self.assertEqual(len(colLabels), self.table.num_cols)


class CustomHeadingNoFormatTableTester(TableInstanceTester):
    def setUp(self):
        self.table = ReportTable(self.custom_headings, None)
