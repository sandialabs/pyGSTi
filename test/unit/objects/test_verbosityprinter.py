import pickle
from contextlib import contextmanager
from io import StringIO
from unittest import mock

from pygsti.baseobjs import verbosityprinter as vbp
from ..util import BaseCase

warn_message = 'This might go badly'
error_message = 'Something terrible happened'
log_message = 'Data received'


class VerbosityPrinterMethodBase(object):
    def test_log(self):
        with self.redirect_output() as out:
            self.vbp.log(log_message)
            self.assertEqual(out(), self.expected_log)

    def test_warning(self):
        with self.redirect_error() as err:
            self.vbp.warning(warn_message)
            self.assertEqual(err(), self.expected_warn)

    def test_error(self):
        with self.redirect_error() as err:
            self.vbp.error(error_message)
            self.assertEqual(err(), self.expected_error)

    def test_progress_logging(self):
        data = list(range(2))
        expected = ""
        with self.redirect_output() as out:
            with self.vbp.progress_logging():
                for i in data:
                    self.vbp.show_progress(i, len(data))
                    expected += self.expected_progress[i]
                    self.assertEqual(out(), expected)
            # output once more after context is returned
            expected += self.expected_progress[i + 1]
            self.assertEqual(out(), expected)

    def test_to_string(self):
        vbp_str = str(self.vbp)
        self.assertTrue(vbp_str.startswith("Printer Object:"))

    def test_pickle(self):
        s = pickle.dumps(self.vbp)
        vbp_pickled = pickle.loads(s)
        self.assertEqual(vbp_pickled.verbosity, self.vbp.verbosity)
        self.assertEqual(vbp_pickled.filename, self.vbp.filename)
        self.assertIsNone(vbp_pickled._comm)
        
        # Test __getstate__ directly
        state = self.vbp.__getstate__()
        self.assertNotIn('_comm', state)

    def test_recording(self):
        self.assertFalse(self.vbp.is_recording())
        self.vbp.start_recording()
        self.assertTrue(self.vbp.is_recording())
        
        self.vbp.error("recorded error")
        self.vbp.warning("recorded warn")
        self.vbp.log("recorded log", message_level=1)
        
        recorded = self.vbp.stop_recording()
        self.assertFalse(self.vbp.is_recording())
        
        self.assertTrue(any(r[0] == "ERROR" and "recorded error" in r[2] for r in recorded))
        self.assertTrue(any(r[0] == "WARNING" and "recorded warn" in r[2] for r in recorded))
        
        if self.verbosity >= 1:
            self.assertTrue(any(r[0] == "LOG" and "recorded log" in r[2] for r in recorded))
        else:
            self.assertFalse(any(r[0] == "LOG" for r in recorded))


class VerbosityPrinterStreamInstance(object):
    def setUp(self):
        super(VerbosityPrinterStreamInstance, self).setUp()
        self.vbp = vbp.VerbosityPrinter.create_printer(self.verbosity)

    @contextmanager
    def redirect_output(self):
        with mock.patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            yield mock_stdout.getvalue

    @contextmanager
    def redirect_error(self):
        with mock.patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            yield mock_stderr.getvalue


class VerbosityPrinterFileInstance(object):
    def setUp(self):
        super(VerbosityPrinterFileInstance, self).setUp()
        self.redirect_output = self.redirect_file_io
        self.redirect_error = self.redirect_file_io
        with self.redirect_file_io():
            self.vbp = vbp.VerbosityPrinter(self.verbosity, filename='/tmp/test_file.log')

    @contextmanager
    def redirect_file_io(self):
        with mock.patch.object(vbp, 'open', mock.mock_open()) as mock_open:
            sio = StringIO()
            sio.close = mock.MagicMock()
            mock_open.return_value = sio
            yield sio.getvalue


class VerbosityPrinterLevel0Tester(VerbosityPrinterMethodBase, VerbosityPrinterStreamInstance, BaseCase):
    verbosity = 0
    expected_log = ""
    expected_warn = "\nWARNING: {}\n\n".format(warn_message)
    expected_error = "\nERROR: {}\n".format(error_message)
    expected_progress = ["", "", ""]


class VerbosityPrinterLevel1Tester(VerbosityPrinterLevel0Tester):
    verbosity = 1
    expected_log = "{}\n".format(log_message)
    expected_progress = [
        'Progress: [--------------------------------------------------] 0.0% \r',
        'Progress: [#########################-------------------------] 50.0% \r',
        'Progress: [##################################################] 100.0% \n'
    ]


class VerbosityPrinterLevel2Tester(VerbosityPrinterLevel1Tester):
    verbosity = 2
    expected_progress = [
        'Progress: Iter 1 of 2 : \n',
        'Progress: Iter 2 of 2 : \n',
        ''
    ]


class VerbosityPrinterLevel3Tester(VerbosityPrinterLevel2Tester):
    verbosity = 3


class VerbosityPrinterFileLevel0Tester(VerbosityPrinterMethodBase, VerbosityPrinterFileInstance, BaseCase):
    verbosity = 0
    expected_log = ""
    expected_warn = "\nWARNING: {}\n\n".format(warn_message)
    expected_error = "\nERROR: {}\n".format(error_message)
    expected_progress = ["", "", ""]


class VerbosityPrinterFileLevel1Tester(VerbosityPrinterFileLevel0Tester):
    verbosity = 1
    expected_log = "{}\n".format(log_message)


class VerbosityPrinterFileLevel2Tester(VerbosityPrinterFileLevel1Tester):
    verbosity = 2
    expected_progress = [
        'Progress: Iter 1 of 2 : \n',
        'Progress: Iter 2 of 2 : \n',
        ''
    ]


class VerbosityPrinterFileLevel3Tester(VerbosityPrinterFileLevel2Tester):
    verbosity = 3


class VerbosityPrinterFactoryTester(BaseCase):
    def test_create_printer_int(self):
        printer = vbp.VerbosityPrinter.create_printer(3)
        self.assertIsInstance(printer, vbp.VerbosityPrinter)
        self.assertEqual(printer.verbosity, 3)

    def test_create_printer_from_printer(self):
        orig_printer = vbp.VerbosityPrinter(2, filename="some_file.log", warnings=False, split=True)
        new_printer = vbp.VerbosityPrinter.create_printer(orig_printer)
        self.assertIsNot(new_printer, orig_printer)
        self.assertEqual(new_printer.verbosity, 2)
        self.assertEqual(new_printer.filename, "some_file.log")
        self.assertFalse(new_printer.warnings)
        self.assertTrue(new_printer.split)

    def test_clone_and_deep_copy_stacks(self):
        orig_printer = vbp.VerbosityPrinter(1)
        orig_printer._delayQueue.append("queued message")
        orig_printer._progressStack.append(2)
        orig_printer._progressParamsStack.append((1, 2, 3))

        cloned = orig_printer.clone()
        self.assertIsNot(cloned, orig_printer)
        self.assertEqual(cloned.verbosity, orig_printer.verbosity)
        
        # Verify stack independence (deep copied)
        self.assertEqual(cloned._delayQueue, ["queued message"])
        cloned._delayQueue.append("new queued")
        self.assertNotEqual(cloned._delayQueue, orig_printer._delayQueue)

        self.assertEqual(cloned._progressStack, [2])
        cloned._progressStack.append(3)
        self.assertNotEqual(cloned._progressStack, orig_printer._progressStack)

        self.assertEqual(cloned._progressParamsStack, [(1, 2, 3)])
        cloned._progressParamsStack.append((4, 5, 6))
        self.assertNotEqual(cloned._progressParamsStack, orig_printer._progressParamsStack)

    def test_dunder_add_sub(self):
        printer = vbp.VerbosityPrinter(2)
        self.assertEqual(printer.verbosity, 2)
        self.assertEqual(printer.extra_indents, 0)

        # Addition
        more_verbose = printer + 1
        self.assertEqual(more_verbose.verbosity, 3)
        self.assertEqual(more_verbose.extra_indents, -1)
        # Original should not be modified
        self.assertEqual(printer.verbosity, 2)
        self.assertEqual(printer.extra_indents, 0)

        # Subtraction
        less_verbose = printer - 2
        self.assertEqual(less_verbose.verbosity, 0)
        self.assertEqual(less_verbose.extra_indents, 2)
        # Original should not be modified
        self.assertEqual(printer.verbosity, 2)
        self.assertEqual(printer.extra_indents, 0)

    def test_verbosity_env(self):
        printer = vbp.VerbosityPrinter(1)
        self.assertEqual(printer.defaultVerbosity, 1)

        with printer.verbosity_env(3):
            self.assertEqual(printer.defaultVerbosity, 3)

        # Restored
        self.assertEqual(printer.defaultVerbosity, 1)

        # Restored even on exception
        try:
            with printer.verbosity_env(4):
                self.assertEqual(printer.defaultVerbosity, 4)
                raise ValueError("Intentional error")
        except ValueError:
            pass

        self.assertEqual(printer.defaultVerbosity, 1)

    def test_progress_logging_queued_log(self):
        printer = vbp.VerbosityPrinter(1)
        with mock.patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with printer.progress_logging():
                printer.show_progress(0, 2)
                printer.log("queued log message", 1)
                self.assertIn("queued log message", printer._delayQueue[0])
                self.assertNotIn("queued log message", mock_stdout.getvalue())
            
            # After context manager exits, the queue should be flushed
            self.assertIn("queued log message", mock_stdout.getvalue())
            self.assertEqual(len(printer._delayQueue), 0)

    def test_progress_logging_verbose_messages(self):
        printer = vbp.VerbosityPrinter(2)
        with mock.patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            with printer.progress_logging(1):
                printer.show_progress(0, 2, verbose_messages=["verbose 1"])
            
            output = mock_stdout.getvalue()
            self.assertIn("verbose 1", output)

    def test_log_custom_end(self):
        printer = vbp.VerbosityPrinter(1)
        with mock.patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            printer.log("hello", end="---")
            self.assertEqual(mock_stdout.getvalue(), "hello---")

    def test_warning_suppressed(self):
        printer = vbp.VerbosityPrinter(1, warnings=False)
        with mock.patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            printer.warning("suppressed warn")
            self.assertEqual(mock_stderr.getvalue(), "")


class VerbosityPrinterHelperTester(BaseCase):
    def test_num_digits(self):
        self.assertEqual(vbp._num_digits(0), 1)
        self.assertEqual(vbp._num_digits(1), 1)
        self.assertEqual(vbp._num_digits(9), 1)
        self.assertEqual(vbp._num_digits(10), 2)
        self.assertEqual(vbp._num_digits(99), 2)
        self.assertEqual(vbp._num_digits(100), 3)
        self.assertEqual(vbp._num_digits(1000), 4)

    def test_build_progress_bar(self):
        # Full progress bar at 100% (should have end='\n' if iteration == total)
        res_full = vbp._build_progress_bar(10, 10, bar_length=10)
        self.assertTrue(res_full.endswith("\n"))
        self.assertIn("##########", res_full)
        self.assertIn("100.0%", res_full)

        # Partial progress bar (should have end='\r' by default if iteration != total)
        res_half = vbp._build_progress_bar(5, 10, bar_length=10)
        self.assertTrue(res_half.endswith("\r"))
        self.assertIn("#####-----", res_half)
        self.assertIn("50.0%", res_half)

        # Custom empty/fill characters and suffix/prefix
        res_custom = vbp._build_progress_bar(3, 10, bar_length=10, fill_char="*", empty_char=".", prefix="Prep:", suffix="Done")
        self.assertIn("***.......", res_custom)
        self.assertIn("Prep:", res_custom)
        self.assertIn("Done", res_custom)
        self.assertIn("30.0%", res_custom)

    def test_build_verbose_iteration(self):
        res = vbp._build_verbose_iteration(0, 10, prefix="Prep:", suffix="Done", end="\n")
        # 10 is 2 digits, so 0+1 = 1 should be zero-padded to "01"
        self.assertIn("Prep: Iter 01 of 10 Done:", res)
