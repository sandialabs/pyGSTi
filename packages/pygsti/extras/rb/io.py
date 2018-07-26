from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as _np
from . import results as _results

def import_rb_summary_data(filenames, is_counts_data=True, contains_circuit_data=True, 
                           finitesampling=True, number_of_qubits=None, total_counts=None, 
                           verbosity=1):
    """
    Reads in one or more text files of summary RB data into a RBSummaryDataset object. This format 
    is appropriate for using the RB analysis functions. The datafile(s) should have one of the 
    following two formats:

    Format 1 (`is_counts_data` is True):

        # The number of qubits
        The number of qubits (this line is optional if `number_of_qubits` is specified)
        # RB length // Success counts // Total counts // Circuit depth // Circuit two-qubit gate count
        Between 3 and 5 columns of data (the last two columns are expected only if `contains_circuit_data` is True).

    Format 2 (`is_counts_data` is False):
        
        # The number of qubits
        The number of qubits (this line is optional if `number_of_qubits` is specified)
        # RB length // Survival probabilities // Circuit depth // Circuit two-qubit gate count
        Between 2 and 4 columns of data (the last two columns are expected only if `contains_circuit_data` is True).

    Parameters
    ----------
    filenames : str or list.
        The filename, or a list of filenams, where the data is stored. The data from all files is read
        into a *single* dataset, so normally it should all be data for a single RB experiment.

    is_counts_data : bool, optional
        Whether the data to be read contains success counts data (True) or survival probability data (False).

    contains_circuit_data : bool, optional.
        Whether the data counts summary circuit data.

    finitesampling : bool, optional
        Records in the RBSummaryDataset whether the survival probability for each circuit was obtained
        from finite sampling of the outcome probabilities. This is there to, by default, warn the user 
        that any finite sampling cannot be taken into account if the input is not counts data (when
        they run any analysis on the data). But it is useful to be able to set this to False for simulated
        data obtained from perfect outcome sampling.

    number_of_qubits : int, optional.
        The number of qubits the data is for. Must be specified if this isn't in the input file.

    total_counts : int, optional
        If the data is success probability data, the total counts can optional be input here.

    verbosity : int, optional 
        The amount of print-to-screen.

    Returns
    -------
    None
    """
    assert(not(is_counts_data and total_counts != None)), "For counts data, the total counts must be included in the data file, so should not be manually input!"

    lengths = []
    if is_counts_data:
        scounts = []
        tcounts = []
        SPs = None
    else:
        scounts = None
        tcounts = total_counts
        SPs = []
    if contains_circuit_data:
        cdepths = []
        c2Qgc = []
    else:
        cdepths = None
        c2Qgc = None
 
    if type(filenames) == str:
        filenames = [filenames,]

    for filename in filenames:
        if verbosity > 0: print("Importing "+filename+"...",end='')
        try:
            with open(filename,'r') as f:
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')
                    if line[0][0] != '#':
                        if len(line) > 1:
                            lengths.append(int(line[0]))
                            if is_counts_data: 
                                scounts.append(int(line[1]))
                                tcounts.append(int(line[2]))
                                if contains_circuit_data:
                                    cdepths.append(int(line[3]))
                                    c2Qgc.append(int(line[4]))
                            else:
                                SPs.append(float(line[1]))
                                if contains_circuit_data:
                                    cdepths.append(int(line[2]))
                                    c2Qgc.append(int(line[3]))
                        else:
                            if number_of_qubits is not None:
                                assert(number_of_qubits == int(line[0])), "The file is for data on a different number of qubits to that specified!"
                            else:
                                number_of_qubits = int(line[0])
            if verbosity > 0: print("Complete.")
        except:
            raise ValueError("Date import failed! File does not exist or the format is incorrect.")

    assert(number_of_qubits is not None), "The number of qubits was not specified as input to this function *or* found in the data file!"
   
    RBSdataset = _results.RBSummaryDataset(number_of_qubits, lengths, success_counts=scounts, success_probabilities=SPs, total_counts=tcounts, 
                                            circuit_depths=cdepths, circuit_twoQgate_counts=c2Qgc, sortedinput=False, finitesampling=finitesampling)

    return RBSdataset

def write_rb_summary_data_to_file(RBSdataset,filename):
    """
    Writes an RBSSummaryDataset to file, in the format that can be read back in by
    import_rb_summary_data().

    Parameters
    ----------
    RBSdataset : RBSummaryDataset
        The data to write to file.

    filename : str
        The filename where the dataset should be written.

    Returns
    -------
    None
    """
    with open(filename,'w') as f:
        f.write('# Number of qubits\n')
        f.write('{}\n'.format(RBSdataset.number_of_qubits))

        if (RBSdataset.circuit_depths is not None) and (RBSdataset.circuit_twoQgate_counts is not None):

            if RBSdataset.success_counts is not None:
                f.write('# RB length // Success counts // Total counts // Circuit depth // Circuit two-qubit gate count\n')
                for i in range(len(RBSdataset.lengths)):
                    l = RBSdataset.lengths[i]
                    for j in range(len(RBSdataset.success_counts[i])):
                        f.write('{} {} {} {} {}\n'.format(l,RBSdataset.success_counts[i][j],RBSdataset.total_counts[i][j],RBSdataset.circuit_depths[i][j],RBSdataset.circuit_twoQgate_counts[i][j]))
            else:
                f.write('# RB length // Success probability // Circuit depth // Circuit two-qubit gate count\n')
                for i in range(len(RBSdataset.lengths)):
                    l = RBSdataset.lengths[i]
                    for j in range(len(RBSdataset.success_probabilities[i])):
                        f.write('{} {} {} {}\n'.format(l,RBSdataset.success_probabilities[i][j],RBSdataset.circuit_depths[i][j],RBSdataset.circuit_twoQgate_counts[i][j]))

        else:
            if RBSdataset.success_counts is not None:
                f.write('# RB length // Success counts // Total counts\n')
                for i in range(len(RBSdataset.lengths)):
                    l = RBSdataset.lengths[i]
                    for j in range(len(RBSdataset.success_counts[i])):
                        f.write('{} {} {}\n'.format(l,RBSdataset.success_counts[i][j],RBSdataset.total_counts[i][j]))

            else:
                f.write('# RB length // Success probability\n')
                for i in range(len(RBSdataset.lengths)):
                    l = RBSdataset.lengths[i]
                    for j in range(len(RBSdataset.success_probabilities[i])):
                        f.write('{} {} {} {}\n'.format(l,RBSdataset.success_probabilities[i][j]))
    return 