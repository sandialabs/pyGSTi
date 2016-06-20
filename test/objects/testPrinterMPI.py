from pygsti.objects.verbosityprinter import *
from mpi4py                          import MPI


import unittest, os


class TestPrinterMPI(unittest.TestCase):

    def setUp(self):
        # move working directories
        self.old = os.getcwd()
        os.chdir(os.path.abspath(os.path.dirname(__file__)))

    def tearDown(self):
        os.chdir(self.old)

    def test_mpi(self):
        comm    = MPI.COMM_WORLD
        print('Running test on process %s' % comm.Get_rank())
        # Here, processes 1, 2, and 3 will print to their own files. The filename is being overloaded so that the files end up out of the way
        printer = VerbosityPrinter(2, filename='../temp_test_files/comm_output_%s.txt' % comm.Get_rank(), comm=comm) # override the default location of comm output
        # Just testing that nothing terrible happens. Most of the printer's features can be tested in testPrinter.py
        printer.log('testing')

if __name__ == "__main__":
    unittest.main()
