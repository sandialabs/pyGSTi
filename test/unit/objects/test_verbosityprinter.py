import pickle
from contextlib import contextmanager
from io import StringIO
from unittest import mock

from pygsti.objects import verbosityprinter as vbp
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
        # TODO assert correctness

    def test_pickle(self):
        s = pickle.dumps(self.vbp)
        vbp_pickled = pickle.loads(s)
        # TODO assert correctness


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
    expected_warn = "\nWARNING: {}\n".format(warn_message)
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
    expected_warn = "\nWARNING: {}\n".format(warn_message)
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
