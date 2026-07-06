import unittest
import tempfile
import os
from io import StringIO
from unittest import mock

from pygsti.baseobjs import verbosityprinter as vbp


def demo(verbosity_or_printer):
    # usage of the show_progress function
    printer = vbp.VerbosityPrinter.create_printer(verbosity_or_printer)
    data = range(10)
    with printer.progress_logging(2):
        for i, item in enumerate(data):
            printer.show_progress(i, len(data)-1,
                                  verbose_messages=['%s gates' % i], prefix='--- GST (', suffix=') ---')


def nested_demo(verbosity_or_printer):
    printer = vbp.VerbosityPrinter.create_printer(verbosity_or_printer)
    printer.warning('Beginning demonstration of the verbosityprinter class. This could go wrong..')
    data = range(10)
    with printer.progress_logging(1):
        for i, item in enumerate(data):
            printer.show_progress(i, len(data)-1,
                                  verbose_messages=['%s circuits' % i], prefix='-- IterativeGST (', suffix=') --')
            if i == 5:
                printer.error('The iterator is five. This caused an error, apparently')
            demo(printer - 1)


class TestVerbosityPrinterDemo(unittest.TestCase):
    def test_nested_demo_stdout(self):
        # We test nested_demo at levels 0 to 4 and ensure output is printed correctly
        # at each level, capturing stdout and stderr.
        outputs = {}
        for v in range(5):
            with mock.patch('sys.stdout', new_callable=StringIO) as mock_stdout, \
                 mock.patch('sys.stderr', new_callable=StringIO) as mock_stderr:
                nested_demo(v)
                outputs[v] = (mock_stdout.getvalue(), mock_stderr.getvalue())

        # Assertions on stdout / stderr matching expected behavior for demo
        # Level 0 (tersest): stderr has error, warning, no stdout
        self.assertIn("ERROR: The iterator is five", outputs[0][1])
        self.assertIn("WARNING: Beginning demonstration", outputs[0][1])
        self.assertEqual(outputs[0][0], "")

        # Level 1 (terse): has stdout (iterative progress bar), stderr has warning and error
        self.assertIn("ERROR: The iterator is five", outputs[1][1])
        self.assertIn("WARNING: Beginning demonstration", outputs[1][1])
        self.assertIn("IterativeGST", outputs[1][0])
        # Level 1 nested has demo(printer - 1) which is verbosity 0. So no nested output.

        # Level 2 (standard): Level 2 nested has demo(1), which prints nothing for nested demo, but prints verbose iterations for outer demo.
        self.assertIn("-- IterativeGST (", outputs[2][0])
        self.assertNotIn("--- GST (", outputs[2][0])

        # Level 3 (verbose): Level 3 nested has demo(2), which prints progress bars for the nested demo
        self.assertIn("--- GST (", outputs[3][0])
        self.assertNotIn("gates", outputs[3][0])

        # Level 4 (most verbose): Level 4 nested has demo(3), which prints verbose iterations for nested demo
        self.assertIn("--- GST (", outputs[4][0])
        self.assertIn("gates", outputs[4][0])

        # Verify that output is not empty for verbosity levels >= 1
        for v in range(1, 5):
            self.assertGreater(len(outputs[v][0]), 0)

    def test_demo_file_output(self):
        # Create deterministic sequential file output tests
        with tempfile.TemporaryDirectory() as tmpdirname:
            # We run 4 independent printer instances writing to output files
            for i in range(4):
                file_path = os.path.join(tmpdirname, f"output{i}.txt")
                printer = vbp.VerbosityPrinter(i, file_path)
                demo(printer)
                
                self.assertTrue(os.path.exists(file_path))
                with open(file_path, "r") as f:
                    content = f.read()
                
                if i < 3:
                    # Verbosity levels 0, 1, and 2 do not output to file (progress bar to file is skipped)
                    self.assertEqual(content, "")
                else:
                    # Verbosity level 3 is greater than the progress logging level,
                    # so it prints verbose iterations instead of a progress bar.
                    self.assertIn("--- GST (", content)
                    self.assertIn("gates", content)  # verbose_messages output at level 3
                    self.assertNotIn("\r", content)  # verbose iterations end in newlines


if __name__ == '__main__':
    unittest.main()
