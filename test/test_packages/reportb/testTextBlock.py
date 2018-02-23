import unittest
import pygsti

from pygsti.report.textblock import ReportText
from ..testutils         import BaseTestCase, compare_files, temp_files

# Covers some missing tests, but NOT all of report.table.py
class TestTextBlock(BaseTestCase):
    def setUp(self):
        super(TestTextBlock, self).setUp()

    def test_block(self):
        text = ReportText("Hello")
        text.render("html")
        s = str(text)

        with self.assertRaises(ValueError):
            t = ReportText("Hello","foobar")
            t.render("html")




if __name__ == '__main__':
    unittest.main(verbosity=2)
