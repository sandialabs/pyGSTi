from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
from . import results as _results

def import_rb_summary_data(filenames, countsdata=True, circuitdata=True, finitesampling=True, number_of_qubits=None, 
                           totalcounts=None, verbosity=1):
    """
    #
    filenames can be a single name as a string or a list of names.

    qubit number
    # RB length // Success counts // Number of counts // Circuit depth // Circuit two-qubit gate count
    data
    """

    assert(not(countsdata and totalcounts != None)), "For counts data, the total counts must be included in the data file, so should not be manually input!"

    lengths = []
    if countsdata:
        scounts = []
        tcounts = []
        SPs = None
    else:
        scounts = None
        tcounts = totalcounts
        SPs = []
    if circuitdata:
        cdepths = []
        c2Qgc = []
    else:
        cdepths = None
        c2Qgc = None
 
    if type(filenames) == str:
        filenames = [filenames,]

    for filename in filenames:
        if verbosity > 0:
            print("Importing "+filename+"...",end='')
        try:
            with open(filename,'r') as f:
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')
                    if line[0][0] != '#':
                        if len(line) > 1:
                            lengths.append(int(line[0]))
                            if countsdata: 
                                scounts.append(int(line[1]))
                                tcounts.append(int(line[2]))
                                if circuitdata:
                                    cdepths.append(int(line[3]))
                                    c2Qgc.append(int(line[4]))
                            else:
                                SPs.append(float(line[1]))
                                if circuitdata:
                                    cdepths.append(int(line[2]))
                                    c2Qgc.append(int(line[3]))
                        else:
                            if number_of_qubits is not None:
                                assert(number_of_qubits == int(line[0])), "The file is for data on a different number of qubits to that specified!"
                            else:
                                number_of_qubits = int(line[0])
            if verbosity > 0:
                print("Complete.")
        except:
            raise ValueError("Date import failed! File does not exist or the format is incorrect.")

    assert(number_of_qubits is not None), "The number of qubits was not specified as input to this function *or* found in the data file!"
   
    RBSdataset = _results.RBSummaryDataset(number_of_qubits, lengths, successcounts=scounts, successprobabilities=SPs, totalcounts=tcounts, 
                                            circuitdepths=cdepths, circuit2Qgcounts=c2Qgc, sortedinput=False, finitesampling=finitesampling)

    return RBSdataset

def write_rb_summary_data_to_file(RBSdataset,filename):
    """
    Todo : docstring.
    """

    with open(filename,'w') as f:
        f.write('# Number of qubits\n')
        f.write('{}\n'.format(RBSdataset.number_of_qubits))

        if (RBSdataset.circuitdepths is not None) and (RBSdataset.circuit2Qgcounts is not None):

            if RBSdataset.successcounts is not None:
                f.write('# RB length // Success counts // Total counts // Circuit depth // Circuit two-qubit gate count\n')
                for i in range(len(RBSdataset.lengths)):
                    l = RBSdataset.lengths[i]
                    for j in range(len(RBSdataset.successcounts[i])):
                        f.write('{} {} {} {} {}\n'.format(l,RBSdataset.successcounts[i][j],RBSdataset.totalcounts[i][j],RBSdataset.circuitdepths[i][j],RBSdataset.circuit2Qgcounts[i][j]))
            else:
                f.write('# RB length // Success probability // Circuit depth // Circuit two-qubit gate count\n')
                for i in range(len(RBSdataset.lengths)):
                    l = RBSdataset.lengths[i]
                    for j in range(len(RBSdataset.successprobabilities[i])):
                        f.write('{} {} {} {}\n'.format(l,RBSdataset.successprobabilities[i][j],RBSdataset.circuitdepths[i][j],RBSdataset.circuit2Qgcounts[i][j]))

        else:
            if RBSdataset.successcounts is not None:
                f.write('# RB length // Success counts // Total counts\n')
                for i in range(len(RBSdataset.lengths)):
                    l = RBSdataset.lengths[i]
                    for j in range(len(RBSdataset.successcounts[i])):
                        f.write('{} {} {}\n'.format(l,RBSdataset.successcounts[i][j],RBSdataset.totalcounts[i][j]))

            else:
                f.write('# RB length // Success probability\n')
                for i in range(len(RBSdataset.lengths)):
                    l = RBSdataset.lengths[i]
                    for j in range(len(RBSdataset.successprobabilities[i])):
                        f.write('{} {} {} {}\n'.format(l,RBSdataset.successprobabilities[i][j]))