from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import ast as _ast
import warnings as _warnings
import pickle as _pickle
import os as _os

# todo : update
from . import analysis as _results
from . import sample as _sample
from . import dataset as _dataset
from . import rbanalyzer as _rbanalyzer
from ... import io as _io


def import_dataset_and_export_summary_data(filename, outfolder='summarydata', verbosity=1):

    try:
        _os.mkdir(outfolder)
        if verbosity > 0:
            print(" - Created `" + outfolder + "` folder to store the summary data files.")
    except:
        if verbosity > 0:
            print(" - `" + outfolder + "` folder already exists. Will write data into that folder.")

    rbanalyzer = import_data(filename, verbosity=verbosity)
    rbanalyzer.create_summary_data()

    with open(outfolder + '/readme.txt', 'w') as f:
        f.write('# This folder contains RB summary data\n')
        f.write('# The RB specifications (read in as RBSpec objects) are in files RBSpec*.txt where * is an integer\n')
        f.write('# The summary data for the RB specification in RBSpec*.txt is stored in rbsummary_data*-#.txt where # is an integer running from 0 to the number of different qubits sets that simul., independent RB was performed on.\n')

    for i, spec in enumerate(rbanalyzer._specs):
        structure = spec.get_structure()
        write_rb_spec_to_file(spec, outfolder + '/rbspec' + str(i) + '.txt', warning=0)
        for j, qubits in enumerate(structure):
            write_rb_summary_data_to_file(rbanalyzer._summary_data[i][qubits], outfolder + '/rbsummarydata' + str(i) + '-'
                                          + str(j) + '.txt')

    return


def import_data(ds_filename=None, summarydatasets_filenames=None, verbosity=1):
    """
    todo

    """
    if ds_filename is not None:

        if isinstance(ds_filename, str):

            if ds_filename[-4:] == '.txt':

                ds = _io.load_dataset(ds_filename, collisionAction='aggregate', recordZeroCnts=False, verbosity=verbosity)

            elif ds_filename[-4:] == '.pkl':

                if verbosity > 0:
                    print(" - Loading DataSet from pickle file...", end='')
                with open(ds_filename, 'rb') as f:
                    ds = _pickle.load(f)
                if verbosity > 0:
                    print("complete.")

            else:
                raise ValueError("File must end in .pkl or .txt!")

        else:

            ds = ds_filename

        if verbosity > 0:
            print(" - Extracting RB metadata from the imported DataSet...", end='')

        specfiles = []
        speccircuits = {}
        for circ in ds.keys():

            speclist = ds.auxInfo[circ]['spec']
            l = ds.auxInfo[circ]['length']
            target = ds.auxInfo[circ]['target']

            if isinstance(speclist, str):
                speclist = [speclist, ]

            #print(speclist)

            for spec in speclist:
                if spec not in specfiles:
                    specfiles.append(spec)
                    speccircuits[spec] = {}

                if l not in speccircuits[spec].keys():
                    speccircuits[spec][l] = []

                speccircuits[spec][l].append((circ, target))

        if verbosity > 0:
            print("complete.")
            print(" - Reading in the metadata from the extracted filenames...", end='')

        rbspeclist = []
        directory = ds_filename.split('/')
        directory = '/'.join(directory[: -1])

        for specfilename in specfiles:

            rbspec = import_rb_spec(directory + '/' + specfilename)
            rbspec.add_circuits(speccircuits[spec])
            rbspeclist.append(rbspec)

        if verbosity > 0:
            print("complete.")
            print(" - Recording all of the data in an RBAnalyzer...", end='')

        rbanalyzer = _rbanalyzer.RBAnalyzer(rbspeclist, ds=ds, summary_data=None)

        if verbosity > 0:
            print("complete.")

        return rbanalyzer

    elif summarydatasets_filenames is not None:

        rbspeclist = []
        specfiles = list(summarydatasets_filenames.keys())
        summary_data = {}

        for specfilename in specfiles:

            rbspec = import_rb_spec(specfilename)
            rbspeclist.append(rbspec)

        for i, specfilename, rbspec in enumerate(zip(specfiles, rbspeclist)):

            summary_data[i] = {}
            sds_filenames = summarydatasets_filenames[specfilename]

            structure = rbspec.get_structure()
            if len(structure) == 1:
                if isinstance(sds_filenames, str):
                    sds_filenames = [sds_filenames, ]
            assert(len(sds_filenames) == len(structure))

            for sdsfn, qubits in zip(sds_filenames, structure):
                summary_data[i][qubits] = import_rb_summary_data(sdsfn)

            rbanalyzer = _rbanalyzer.RBAnalyzer(rbspeclist, ds=None, summary_data=summary_data)

        else:
            raise ValueError("Either a filename for a DataSet or filenames for a set of RBSpecs "
                             + "and RBSummaryDatasets must be provided!")

        return None


def import_rb_spec(filename):
    """
    todo

    """
    d = {}
    with open(filename) as f:
        for line in f:
            if len(line) > 0 and line[0] != '#':
                line = line.strip('\n')
                line = line.split(' ', 1)
                try:
                    d[line[0]] = _ast.literal_eval(line[1])
                except:
                    d[line[0]] = line[1]

    assert(d.get('type', None) == 'rb'), "This is for importing RB specs!"

    try:
        rbtype = d['rbtype']
    except:
        raise ValueError("Input file does not contain a line specifying the RB type!")
    assert(isinstance(rbtype, str)), "The RB type (specified as rbtype) must be a string!"

    try:
        structure = d['structure']
    except:
        raise ValueError("Input file does not contain a line specifying the structure!")
    assert(isinstance(structure, tuple)), "The structure must be a tuple!"

    try:
        sampler = d['sampler']
    except:
        raise ValueError("Input file does not contain a line specifying the circuit layer sampler!")
    assert(isinstance(sampler, str)), "The sampler name must be a string!"

    samplerargs = d.get('samplerargs', None)
    lengths = d.get('lengths', None)
    numcircuits = d.get('numcircuits', None)
    subtype = d.get('subtype', None)

    if samplerargs is not None:
        assert(isinstance(samplerargs, dict)), "The samplerargs must be a dict!"

    if lengths is not None:
        assert(isinstance(lengths, list) or isinstance(lengths, tuple)), "The lengths must be a list of tuple!"

    if numcircuits is not None:
        assert(isinstance(numcircuits, list) or isinstance(numcircuits, tuple)), "numcircuits must be a list of tuple!"

    spec = _sample.RBSpec(rbtype, structure, sampler, samplerargs, circuits=None, lengths=lengths,
                          numcircuits=numcircuits, subtype=subtype)

    return spec


def write_rb_spec_to_file(rbspec, filename, warning=1):
    """
    todo

    """
    if rbspec._circuits is not None:
        if warning > 0:
            _warnings.warn("The circuits recorded in this RBSpec are not being written to file!")

    with open(filename, 'w') as f:
        f.write('type rb\n')
        f.write('rbtype ' + rbspec._rbtype + '\n')
        f.write('structure ' + str(rbspec._structure) + '\n')
        f.write('sampler ' + rbspec._sampler + '\n')
        f.write('lengths ' + str(rbspec._lengths) + '\n')
        f.write('numcircuits ' + str(rbspec._numcircuits) + '\n')
        f.write('rbsubtype ' + str(rbspec._subtype) + '\n')
        f.write('samperargs ' + str(rbspec._samplerargs) + '\n')

    return


def import_rb_summary_data(filename, numqubits, datatype='auto', verbosity=1):
    """
    todo

    """
    try:
        with open(filename, 'r') as f:
            if verbosity > 0: print("Importing " + filename + "...", end='')
    except:
        raise ValueError("Date import failed! File does not exist or the format is incorrect.")

    aux = []
    descriptor = ''
    # Work out the type of data we're importing
    with open(filename, 'r') as f:
        for line in f:

            if (len(line) == 0 or line[0] != '#'): break

            elif line.startswith("# "):
                descriptor += line[2:]

            elif line.startswith("## "):

                line = line.strip('\n')
                line = line.split(' ')
                del line[0]

                if line[0:2] == ['rblength', 'success_probabilities']:

                    auxind = 2
                    if datatype == 'auto':
                        datatype = 'success_probabilities'
                    else:
                        assert(datatype == 'success_probabilities'), "The data format appears to be " + \
                            "success probabilities!"

                elif line[0:3] == ['rblength', 'success_counts', 'total_counts']:

                    auxind = 3
                    if datatype == 'auto':
                        datatype = 'success_counts'
                    else:
                        assert(datatype == 'success_counts'), "The data format appears to be success counts!"

                elif line[0: numqubits + 2] == ['rblength',] + ['hd{}c'.format(i) for i in range(numqubits + 1)]:

                    auxind = numqubits + 2
                    if datatype == 'auto':
                        datatype = 'hamming_distance_counts'
                    else:
                        assert(datatype == 'hamming_distance_counts'), "The data format appears to be Hamming " + \
                            "distance counts!"

                elif line[0: numqubits + 2] == ['rblength',] + ['hd{}p'.format(i) for i in range(numqubits + 1)]:

                    auxind = numqubits + 2
                    if datatype == 'auto':
                        datatype = 'hamming_distance_probabilities'
                    else:
                        assert(datatype == 'hamming_distance_probabilities'), "The data format appears to be " + \
                            "Hamming distance probabilities!"

                else:
                    raise ValueError("Invalid file format!")

                if len(line) > auxind:
                    assert(line[auxind] == '#')
                    if len(line) > auxind + 1:
                        auxlabels = line[auxind + 1:]

                break

    # Prepare an aux dict to hold any auxillary data
    aux = {key: {} for key in auxlabels}

    # Read in the data, using a different parser depending on the data type.
    if datatype == 'success_counts':

        success_counts = {}
        total_counts = {}
        finitecounts = True
        hamming_distance_counts = None

        with open(filename, 'r') as f:
            for line in f:
                if (len(line) > 0 and line[0] != '#'):

                    line = line.strip('\n')
                    line = line.split(' ')
                    l = int(line[0])

                    if l not in success_counts:
                        success_counts[l] = []
                        total_counts[l] = []
                        for key in auxlabels:
                            aux[key][l] = []

                    success_counts[l].append(float(line[1]))
                    total_counts[l].append(float(line[2]))

                    if len(aux) > 0:
                        assert(line[3] == '#'), "Auxillary data must be divided from the core data!"
                    for i, key in enumerate(auxlabels):
                        aux[key][l].append(float(line[4 + i]))

    elif datatype == 'success_probabilities':

        success_counts = {}
        total_counts = None
        finitecounts = False
        hamming_distance_counts = None

        with open(filename, 'r') as f:
            for line in f:
                if (len(line) > 0 and line[0] != '#'):

                    line = line.strip('\n')
                    line = line.split(' ')
                    l = int(line[0])

                    if l not in success_counts:
                        success_counts[l] = []
                        for key in auxlabels:
                            aux[key][l] = []

                    success_counts[l].append(float(line[1]))

                    if len(aux) > 0:
                        assert(line[2] == '#'), "Auxillary data must be divided from the core data!"
                    for i, key in enumerate(auxlabels):
                        aux[key][l].append(float(line[3 + i]))

    elif datatype == 'hamming_distance_counts' or datatype == 'hamming_distance_probabilities':

        hamming_distance_counts = {}
        success_counts = None
        total_counts = None

        if datatype == 'hamming_distance_counts': finitecounts = True
        if datatype == 'hamming_distance_probabilities': finitecounts = False

        with open(filename, 'r') as f:
            for line in f:
                if (len(line) > 0 and line[0] != '#'):

                    line = line.strip('\n')
                    line = line.split(' ')
                    l = int(line[0])

                    if l not in hamming_distance_counts:
                        hamming_distance_counts[l] = []
                        for key in auxlabels:
                            aux[key][l] = []

                    hamming_distance_counts[l].append([float(line[1 + i]) for i in range(0,numqubits + 1)])

                    if len(aux) > 0:
                        assert(line[numqubits + 2] == '#'), "Auxillary data must be divided from the core data!"
                    for i, key in enumerate(auxlabels):
                        aux[key][l].append(float(line[numqubits + 3 + i]))

    else:
        raise ValueError("The data format couldn't be extracted from the file!")

    rbdataset = _dataset.RBSummaryDataset(numqubits, success_counts=success_counts, total_counts=total_counts,
                                          hamming_distance_counts=hamming_distance_counts, aux=aux,
                                          finitecounts=finitecounts, descriptor=descriptor)

    if verbosity > 0:
        print('complete')

    return rbdataset


def write_rb_summary_data_to_file(ds, filename):
    """
    todo
    
    """
    numqubits = ds.number_of_qubits
    with open(filename, 'w') as f:

        descriptor_string = ds.descriptor.split("\n")

        for s in descriptor_string:
            if len(s) > 0:
                f.write("# " + s + "\n")

        if ds.datatype == 'success_counts':
            if ds.finitecounts:
                topline = '## rblength success_counts total_counts'
            else:
                topline = '## rblength success_probabilities'

        elif ds.datatype == 'hamming_distance_counts':
            if ds.finitecounts:
                topline = '## rblength' + ''.join([' hd{}c'.format(i) for i in range(0, numqubits + 1)])
            else:
                topline = '## rblength' + ''.join([' hd{}p'.format(i) for i in range(0, numqubits + 1)])

        auxlabels = list(ds.aux.keys())
        if len(auxlabels) > 0:
            topline += ' #'
            for key in auxlabels: topline += ' ' + key

        f.write(topline + '\n')

        for l, counts in ds.counts.items():

            for i, c in enumerate(counts):

                if ds.datatype == 'success_counts':
                    if ds.finitecounts:
                        dataline = str(l) + ' ' + str(c) + ' ' + str(ds._total_counts[l][i])
                    else:
                        dataline = str(l) + ' ' + str(c)
                elif ds.datatype == 'hamming_distance_counts':
                    dataline = str(l) + ''.join([' ' + str(c[i]) for i in range(0,numqubits + 1)])

                if len(auxlabels) > 0:
                    dataline += ' #' + ''.join([' ' + str(ds.aux[key][l][i]) for key in auxlabels])

                f.write(dataline + '\n')

    return


# # todo update this.
# def import_rb_summary_data(filenames, numqubits, type='auto', verbosity=1):
#     """
#     todo : redo 
#     Reads in one or more text files of summary RB data into a RBSummaryDataset object. This format
#     is appropriate for using the RB analysis functions. The datafile(s) should have one of the
#     following two formats:

#     Format 1 (`is_counts_data` is True):

#         # The number of qubits
#         The number of qubits (this line is optional if `number_of_qubits` is specified)
#         # RB length // Success counts // Total counts // Circuit depth // Circuit two-qubit gate count
#         Between 3 and 5 columns of data (the last two columns are expected only if `contains_circuit_data` is True).

#     Format 2 (`is_counts_data` is False):

#         # The number of qubits
#         The number of qubits (this line is optional if `number_of_qubits` is specified)
#         # RB length // Survival probabilities // Circuit depth // Circuit two-qubit gate count
#         Between 2 and 4 columns of data (the last two columns are expected only if `contains_circuit_data` is True).

#     Parameters
#     ----------
#     filenames : str or list.
#         The filename, or a list of filenams, where the data is stored. The data from all files is read
#         into a *single* dataset, so normally it should all be data for a single RB experiment.

#     is_counts_data : bool, optional
#         Whether the data to be read contains success counts data (True) or survival probability data (False).

#     contains_circuit_data : bool, optional.
#         Whether the data counts summary circuit data.

#     finitesampling : bool, optional
#         Records in the RBSummaryDataset whether the survival probability for each circuit was obtained
#         from finite sampling of the outcome probabilities. This is there to, by default, warn the user
#         that any finite sampling cannot be taken into account if the input is not counts data (when
#         they run any analysis on the data). But it is useful to be able to set this to False for simulated
#         data obtained from perfect outcome sampling.

#     number_of_qubits : int, optional.
#         The number of qubits the data is for. Must be specified if this isn't in the input file.

#     total_counts : int, optional
#         If the data is success probability data, the total counts can optional be input here.

#     verbosity : int, optional
#         The amount of print-to-screen.

#     Returns
#     -------
#     None
#     """


# # todo : update this.
# def write_rb_summary_data_to_file(RBSdataset, filename):
#     """
#     Writes an RBSSummaryDataset to file, in the format that can be read back in by
#     import_rb_summary_data().

#     Parameters
#     ----------
#     RBSdataset : RBSummaryDataset
#         The data to write to file.

#     filename : str
#         The filename where the dataset should be written.

#     Returns
#     -------
#     None
#     """