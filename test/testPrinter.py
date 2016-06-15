from pygsti.objects.verbosityprinter import *
from contextlib                      import contextmanager

import unittest
import sys
import os

# The path for a temporary file to be generated in 
filePath        = 'temp_test_files/printer_output.txt'
# Some basic messages to make assertions easier
warningMessage  = 'This might go badly'
errorMessage    = 'Something terrible happened'    
logMessage      = 'Data recieved'

def _generate_with(printer):
    data     = range(2)
    printer.log(logMessage, 3)
    printer.warning(warningMessage)
    for i, item in enumerate(data):
	printer.show_progress(i, len(data)-1, verboseMessages=[('(%s data members remaining)' % (len(data) - (i + 1)))])
	printer.log(logMessage)
	if i == 1:
	    printer.error(errorMessage)
	for i, item in enumerate(data):
	    printer.show_progress(i, len(data)-1, messageLevel=2)
	printer.end_progress()
    printer.end_progress()     

def _to_temp_file(printer):
    data     = range(2)
    
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
        sys.stdout = self
        return self
    def __exit__(self, ext_type, exc_value, traceback):
        sys.stdout = sys.__stdout__  


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
    testcase.assertEqual(generated[1], '  Progress: Iter 0 of 1 : ')
    testcase.assertEqual(generated[2], '  (1 data members remaining)')
    testcase.assertEqual(generated[3], '  %s' % logMessage)

    # Verbose output testing (Warnings, Errors, Verbose Log Messages, Outer Iterations, Innner Iterations)

    verbose   = VerbosityPrinter.build_printer(normal + 1)   
    generated = method(verbose)       

    if printer.filename != None:

	testcase.assertEqual(generated, ['      Data recieved', 'WARNING: This might go badly', '  Progress: Iter 0 of 1 : ', 
					 '  (1 data members remaining)', '  Data recieved', '    Progress: Iter 0 of 1 : ', 
					 '    Progress: Iter 1 of 1 : ', '    Progress: Iter 1 of 1 : ', '    (0 data members remaining)', 
					 '  Data recieved', 'ERROR: Something terrible happened', '    Progress: Iter 0 of 1 : ', '    Progress: Iter 1 of 1 : '])

    else:
        testcase.assertEqual(generated, ['      Data recieved', 'WARNING: This might go badly', '  Progress: Iter 0 of 1 : ', '  (1 data members remaining)', 
                                         '  Data recieved', '    Progress: Iter 0 of 1 : ', '    Progress: Iter 1 of 1 : ', '    Progress: Iter 1 of 1 : ', 
                                         '    (0 data members remaining)', '  Data recieved', 'ERROR: Something terrible happened', '    Progress: Iter 0 of 1 : ', 
                                         '    Progress: Iter 1 of 1 : '])

    # Terse output testing (Warnings, Errors, and an unnested ProgressBar)

    terse     = normal - 1
    generated = method(terse)

    if printer.filename != None:
        testcase.assertEqual(generated, ['WARNING: This might go badly', '  Data recieved', '  Data recieved', 'ERROR: Something terrible happened'])

    else:
        testcase.assertEqual(generated, ['WARNING: This might go badly', '  Progress: [--------------------------------------------------] 0.0% ', 
                                         '  Progress: [##################################################] 100.0% ', 'ERROR: Something terrible happened', 
                                         '  INVALID LEVEL:   Data recieved', '  INVALID LEVEL:   Data recieved'])

    # Tersest output testing (Errors only)

    tersest   = terse - 1
    generated = method(tersest)

    testcase.assertEqual(generated[0], 'ERROR: %s' % errorMessage)

class TestVerbosePrinter(unittest.TestCase):

    def test_file_output(self):
        _test_output_with(self, _to_temp_file, VerbosityPrinter(2, filename=filePath)) 

    def test_stream_output(self): 
        _test_output_with(self, _to_redirected_stream, VerbosityPrinter.build_printer(2))           
        
if __name__ == '__main__':
    unittest.main()
     

'''

Sample output

['      Data recieved', 'WARNING: This might go badly', '  Progress: Iter 0 of 1 : ', '  (1 data members remaining)', '  Data recieved', '    Progress: Iter 0 of 1 : ', '    Progress: Iter 1 of 1 : ', '    Progress: Iter 1 of 1 : ', '    (0 data members remaining)', '  Data recieved', 'ERROR: Something terrible happened', '    Progress: Iter 0 of 1 : ', '    Progress: Iter 1 of 1 : ']
['WARNING: This might go badly', '  Data recieved', '  Data recieved', 'ERROR: Something terrible happened']
.['      Data recieved', 'WARNING: This might go badly', '  Progress: Iter 0 of 1 : ', '  (1 data members remaining)', '  Data recieved', '    Progress: Iter 0 of 1 : ', '    Progress: Iter 1 of 1 : ', '    Progress: Iter 1 of 1 : ', '    (0 data members remaining)', '  Data recieved', 'ERROR: Something terrible happened', '    Progress: Iter 0 of 1 : ', '    Progress: Iter 1 of 1 : ']
['WARNING: This might go badly', '  Progress: [--------------------------------------------------] 0.0% ', '  Progress: [##################################################] 100.0% ', 'ERROR: Something terrible happened', '  INVALID LEVEL:   Data recieved', '  INVALID LEVEL:   Data recieved']
.
----------------------------------------------------------------------

''' 
