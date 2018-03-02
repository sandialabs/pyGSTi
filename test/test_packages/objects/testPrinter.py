from pygsti.baseobjs.verbosityprinter import *
from ..testutils import BaseTestCase, compare_files, temp_files
import unittest, sys, os
import pickle

# The path for a temporary file to be generated in
filePath        = temp_files + '/printer_output.txt'
# Some basic messages to make assertions easier
warningMessage  = 'This might go badly'
errorMessage    = 'Something terrible happened'
logMessage      = 'Data received'

def _generate_with(printer):
    data     = list(range(2))
    printer.log(logMessage, 3)
    printer.warning(warningMessage)
    with printer.progress_logging(1):
        for i, item in enumerate(data):
            printer.show_progress(i, len(data), verboseMessages=[('(%s data members remaining)' % (len(data) - (i + 1)))])
            printer.log(logMessage)
            if i == 1:
                printer.error(errorMessage)
            with printer.progress_logging(2):
                for i, item in enumerate(data):
                    printer.show_progress(i, len(data)) #, messageLevel=2)

def _to_temp_file(printer):
    data     = list(range(2))
    _generate_with(printer)

    generated = []
    with open(filePath, 'r') as output:
        for line in output.read().splitlines():
            if line != '':
                generated.append(line)

    os.remove(filePath)
    return generated


class ListStream:
    def __init__(self):
        self.data = []
    def write(self, s):
        self.data.append(s)
    def flush(self):
        pass
    def __enter__(self):
        sys.stderr = self
        sys.stdout = self
        return self
    def __exit__(self, ext_type, exc_value, traceback):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def _to_redirected_stream(printer):
    generated = []
    with ListStream() as output:
        _generate_with(printer)
    for line in ''.join(output.data).splitlines():
        if line != '':
            generated.append(line)
    return generated

# Defines the rules for testing output, leaving the method up to the caller. Not to be called by itself
def _test_output_with(testcase, method, printer):
    # Normal output testing (Warnings, Errors, Outer Iterations, Inner Progress bars)

    normal    = printer
    generated = method(normal)

    testcase.assertEqual(generated[0], 'WARNING: %s' % warningMessage)
    testcase.assertEqual(generated[1], 'Progress: Iter 1 of 2 : ')
    testcase.assertEqual(generated[2], '(1 data members remaining)')
    testcase.assertEqual(generated[3], '%s' % logMessage)

    # Verbose output testing (Warnings, Errors, Verbose Log Messages, Outer Iterations, Innner Iterations)

    verbose   = VerbosityPrinter.build_printer(normal + 0)
    verbose.verbosity += 1  #increase verbosity to 3
    generated = method(verbose)
    
    if printer.filename != None:
        testcase.assertEqual(generated, ['    Data received',
                                         'WARNING: This might go badly',
                                         'Progress: Iter 1 of 2 : ',
                                         '(1 data members remaining)', 
                                         'Data received', 
                                         '  Progress: Iter 1 of 2 : ',
                                         '  Progress: Iter 2 of 2 : ',
                                         '  Progress: Iter 2 of 2 : ', 
                                         '  (0 data members remaining)',
                                         'Data received', 
                                         'ERROR: Something terrible happened', 
                                         '  Progress: Iter 1 of 2 : ', 
                                         '  Progress: Iter 2 of 2 : '])

    else:
        testcase.assertEqual(generated, ['    Data received',
                                         'WARNING: This might go badly', 
                                         'Progress: Iter 1 of 2 : ', 
                                         '(1 data members remaining)',
                                         'Data received', 
                                         '  Progress: Iter 1 of 2 : ', 
                                         '  Progress: Iter 2 of 2 : ', 
                                         '  Progress: Iter 2 of 2 : ',
                                         '  (0 data members remaining)', 
                                         'Data received', 
                                         'ERROR: Something terrible happened', 
                                         '  Progress: Iter 1 of 2 : ',
                                         '  Progress: Iter 2 of 2 : '])

    # Terse output testing (Warnings, Errors, and an unnested ProgressBar)

    terse     = normal - 1
    generated = method(terse)

    if printer.filename != None:
        testcase.assertEqual(generated, ['WARNING: This might go badly',
                                         '  Data received', 
                                         '  Data received', 
                                         'ERROR: Something terrible happened'])

    else:
        testcase.assertEqual(generated, ['WARNING: This might go badly',
                                         '  Progress: [--------------------------------------------------] 0.0% ',
                                         '  Progress: [#########################-------------------------] 50.0% ',
                                         'ERROR: Something terrible happened',
                                         '  Progress: [##################################################] 100.0% ', 
                                         '  INVALID LEVEL:   Data received', 
                                         '  INVALID LEVEL:   Data received'])

    # Tersest output testing (Errors only)

    tersest   = terse - 1
    generated = method(tersest)

    testcase.assertEqual(generated[0], 'WARNING: %s' % warningMessage)
    testcase.assertEqual(generated[1], 'ERROR: %s'   % errorMessage)

class TestVerbosePrinter(BaseTestCase):

    def test_file_output(self):
        _test_output_with(self, _to_temp_file, VerbosityPrinter(2, filename=filePath))

    def test_stream_output(self):
        _test_output_with(self, _to_redirected_stream, VerbosityPrinter.build_printer(2))

    def test_str(self):
        str(VerbosityPrinter.build_printer(2))

    def test_pickleable(self):
        vbp = VerbosityPrinter.build_printer(2)
        s = pickle.dumps(vbp)
        vbp2 = pickle.loads(s)

    def test_log_variants(self):
        vbp = VerbosityPrinter.build_printer(2)
        vbp.log("Hello",end="\n\n")

        
if __name__ == '__main__':
    unittest.main(verbosity=2)


'''

Sample output

['      Data received', 'WARNING: This might go badly', '  Progress: Iter 0 of 1 : ', '  (1 data members remaining)', '  Data received', '    Progress: Iter 0 of 1 : ', '    Progress: Iter 1 of 1 : ', '    Progress: Iter 1 of 1 : ', '    (0 data members remaining)', '  Data received', 'ERROR: Something terrible happened', '    Progress: Iter 0 of 1 : ', '    Progress: Iter 1 of 1 : ']
['WARNING: This might go badly', '  Data received', '  Data received', 'ERROR: Something terrible happened']
.['      Data received', 'WARNING: This might go badly', '  Progress: Iter 0 of 1 : ', '  (1 data members remaining)', '  Data received', '    Progress: Iter 0 of 1 : ', '    Progress: Iter 1 of 1 : ', '    Progress: Iter 1 of 1 : ', '    (0 data members remaining)', '  Data received', 'ERROR: Something terrible happened', '    Progress: Iter 0 of 1 : ', '    Progress: Iter 1 of 1 : ']
['WARNING: This might go badly', '  Progress: [--------------------------------------------------] 0.0% ', '  Progress: [##################################################] 100.0% ', 'ERROR: Something terrible happened', '  INVALID LEVEL:   Data received', '  INVALID LEVEL:   Data received']
.
----------------------------------------------------------------------

'''
