import os
import signal
import threading
import unittest

from ..util import BaseCase
from pygsti.optimize._sigint import install_sigint_handler


class SigintHandlerTester(BaseCase):

    def setUp(self):
        self._orig_handler = signal.getsignal(signal.SIGINT)

    def tearDown(self):
        signal.signal(signal.SIGINT, self._orig_handler)

    @unittest.skipIf('PYGSTI_NO_CUSTOMLM_SIGINT' in os.environ,
                     "PYGSTI_NO_CUSTOMLM_SIGINT is set; handler installation is suppressed")
    def test_installs_default_int_handler_in_main_thread(self):
        """
        Verify that install_sigint_handler() sets SIGINT to signal.default_int_handler.

        The three assertions establish:
          1. SIG_DFL and default_int_handler are distinct objects, so assertions 2 and 3
             are not vacuously consistent with each other.
          2. The precondition holds: after resetting to SIG_DFL we are not already at
             default_int_handler, so a passing assertion 3 is non-trivial.
          3. install_sigint_handler() switched the handler to default_int_handler.

        setUp/tearDown restore the original handler so this test has no side effects on
        the rest of the test suite.
        """
        self.assertIsNot(signal.default_int_handler, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.assertIs(signal.getsignal(signal.SIGINT), signal.SIG_DFL)
        install_sigint_handler()
        self.assertIs(signal.getsignal(signal.SIGINT), signal.default_int_handler)
        return

    def test_no_error_in_non_main_thread(self):
        # Regression test: importing pygsti from a worker thread (pytest-xdist, Dask, MPI)
        # must not raise "signal only works in main thread of the main interpreter".
        errors = []
        def worker():
            try:
                install_sigint_handler()
            except Exception as e:
                errors.append(e)
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        self.assertEqual(errors, [])
