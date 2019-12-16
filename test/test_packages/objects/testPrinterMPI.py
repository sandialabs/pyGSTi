from pygsti.objects.verbosityprinter import *
#unneeded: from mpi4py                          import MPI
from ..testutils import BaseTestCase, compare_files, temp_files

import unittest, os

class mock_comm():
    def __init__(self, rank):
        self.rank = rank

    def Get_rank(self):
        return self.rank

class TestPrinterMPI(BaseTestCase):
    def test_mpi(self):
        comm    = mock_comm(0)
        print(('Running test on process %s' % comm.Get_rank()))
        # Here, processes 1, 2, and 3 will print to their own files. The filename is being overloaded so that the files end up out of the way
        printer = VerbosityPrinter(2, filename=temp_files + '/comm_output_%s.txt' % comm.Get_rank(), comm=comm) # override the default location of comm output
        # Just testing that nothing terrible happens. Most of the printer's features can be tested in testPrinter.py
        printer.log('testing')

        comm = mock_comm(1)
        printer = VerbosityPrinter(2, comm=comm) # override the default location of comm output
        printer.log('testing')


if __name__ == "__main__":
    unittest.main(verbosity=2)
