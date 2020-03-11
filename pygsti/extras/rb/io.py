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
import json as _json

# todo : update
from . import analysis as _results
from . import sample as _sample
from . import dataset as _dataset
from . import benchmarker as _benchmarker
from ... import io as _io
from ...objects import circuit as _cir
from ...objects import multidataset as _mds


#def load_benchmarking_data(basedir):

def load_benchmarker(directory, load_datasets=True, verbosity=1):
    """

    """
    with open(directory + '/global.txt', 'r') as f:
        globaldict = _json.load(f)

    numpasses = globaldict['numpasses']
    speckeys = globaldict['speckeys']
    success_key = globaldict['success_key']
    success_outcome = globaldict['success_outcome']
    dscomparator = globaldict['dscomparator']

    if load_datasets:
        dskeys = [dskey.name for dskey in _os.scandir(directory + '/datasets') if dskey.is_dir()]
        multidsdict = {dskey: _mds.MultiDataSet()for dskey in dskeys}

        for dskey in dskeys:
            for passnum in range(numpasses):
                dsfn = directory + '/datasets/{}/ds{}.txt'.format(dskey, passnum)
                ds = _io.load_dataset(dsfn, collision_action='keepseparate', record_zero_counts=False,
                                      ignore_zero_count_lines=False, verbosity=verbosity)
                multidsdict[dskey].add_dataset(passnum, ds)
    else:
        multidsdict = None

    specs = {}
    for i, speckey in enumerate(speckeys):
        specs[speckey] = load_benchmarkspec(directory + '/specs/{}.txt'.format(i))

    summary_data = {'global': {}, 'pass': {}, 'aux': {}}
    predictionkeys = [pkey.name for pkey in _os.scandir(directory + '/predictions') if pkey.is_dir()]
    predicted_summary_data = {pkey: {} for pkey in predictionkeys}

    for i, spec in enumerate(specs.values()):

        summary_data['pass'][i] = {}
        summary_data['global'][i] = {}
        summary_data['aux'][i] = {}
        for pkey in predictionkeys:
            predicted_summary_data[pkey][i] = {}

        structure = spec.get_structure()

        for j, qubits in enumerate(structure):

            # Import the summary data for that spec and qubit subset
            with open(directory + '/summarydata/{}-{}.txt'.format(i, j), 'r') as f:
                sd = _json.load(f)
                summary_data['pass'][i][qubits] = {}
                for dtype, data in sd['pass'].items():
                    summary_data['pass'][i][qubits][dtype] = {int(key): value for (key, value) in data.items()}
                summary_data['global'][i][qubits] = {}
                for dtype, data in sd['global'].items():
                    summary_data['global'][i][qubits][dtype] = {int(key): value for (key, value) in data.items()}

            # Import the auxillary data
            with open(directory + '/aux/{}-{}.txt'.format(i, j), 'r') as f:
                aux = _json.load(f)
                summary_data['aux'][i][qubits] = {}
                for dtype, data in aux.items():
                    summary_data['aux'][i][qubits][dtype] = {int(key): value for (key, value) in data.items()}

            # Import the predicted summary data for that spec and qubit subset
            for pkey in predictionkeys:
                with open(directory + '/predictions/{}/summarydata/{}-{}.txt'.format(pkey, i, j), 'r') as f:
                    psd = _json.load(f)
                    predicted_summary_data[pkey][i][qubits] = {}
                    for dtype, data in psd.items():
                        predicted_summary_data[pkey][i][qubits][dtype] = {
                            int(key): value for (key, value) in data.items()}

    benchmarker = _benchmarker.Benchmarker(specs, ds=multidsdict, summary_data=summary_data,
                                           predicted_summary_data=predicted_summary_data,
                                           dstype='dict', success_outcome=success_outcome,
                                           success_key=success_key, dscomparator=dscomparator)

    return benchmarker


def write_benchmarker(benchmarker, outdir, overwrite=False, verbosity=0):

    try:
        _os.makedirs(outdir)
        if verbosity > 0:
            print(" - Created `" + outdir + "` folder to store benchmarker in txt format.")
    except:
        if overwrite:
            if verbosity > 0:
                print(" - `" + outdir + "` folder already exists. Will write data into that folder.")
        else:
            raise ValueError("Directory already exists! Set overwrite to True or change the directory name!")

    globaldict = {}
    globaldict['speckeys'] = benchmarker._speckeys
    globaldict['numpasses'] = benchmarker.numpasses
    globaldict['success_outcome'] = benchmarker.success_outcome
    globaldict['success_key'] = benchmarker.success_key

    if benchmarker.dscomparator is not None:

        globaldict['dscomparator'] = {}
        globaldict['dscomparator']['pVal_pseudothreshold'] = benchmarker.dscomparator.pVal_pseudothreshold
        globaldict['dscomparator']['llr_pseudothreshold'] = benchmarker.dscomparator.llr_pseudothreshold
        globaldict['dscomparator']['pVal_pseudothreshold'] = benchmarker.dscomparator.pVal_pseudothreshold
        globaldict['dscomparator']['jsd_pseudothreshold'] = benchmarker.dscomparator.jsd_pseudothreshold
        globaldict['dscomparator']['aggregate_llr'] = benchmarker.dscomparator.aggregate_llr
        globaldict['dscomparator']['aggregate_llr_threshold'] = benchmarker.dscomparator.aggregate_llr_threshold
        globaldict['dscomparator']['aggregate_nsigma'] = benchmarker.dscomparator.aggregate_nsigma
        globaldict['dscomparator']['aggregate_nsigma_threshold'] = benchmarker.dscomparator.aggregate_nsigma_threshold
        globaldict['dscomparator']['aggregate_pVal'] = benchmarker.dscomparator.aggregate_pVal
        globaldict['dscomparator']['aggregate_pVal_threshold'] = benchmarker.dscomparator.aggregate_pVal_threshold
        globaldict['dscomparator']['inconsistent_datasets_detected'] = \
            benchmarker.dscomparator.inconsistent_datasets_detected
        globaldict['dscomparator']['number_of_significant_sequences'] = int(
            benchmarker.dscomparator.number_of_significant_sequences)
        globaldict['dscomparator']['significance'] = benchmarker.dscomparator.significance

    else:
        globaldict['dscomparator'] = None

    # Write global details to file
    with open(outdir + '/global.txt', 'w') as f:
        _json.dump(globaldict, f, indent=4)

    _os.makedirs(outdir + '/specs')
    _os.makedirs(outdir + '/summarydata')
    _os.makedirs(outdir + '/aux')

    for pkey in benchmarker.predicted_summary_data.keys():
        _os.makedirs(outdir + '/predictions/{}/summarydata'.format(pkey))

    for i, spec in enumerate(benchmarker._specs):
        structure = spec.get_structure()
        write_benchmarkspec(spec, outdir + '/specs/{}.txt'.format(i), warning=0)

        for j, qubits in enumerate(structure):
            summarydict = {'pass': benchmarker.pass_summary_data[i][qubits],
                           'global': benchmarker.global_summary_data[i][qubits]
                           }
            fname = outdir + '/summarydata/' + '{}-{}.txt'.format(i, j)
            with open(fname, 'w') as f:
                _json.dump(summarydict, f, indent=4)

            aux = benchmarker.aux[i][qubits]
            fname = outdir + '/aux/' + '{}-{}.txt'.format(i, j)
            with open(fname, 'w') as f:
                _json.dump(aux, f, indent=4)

            for pkey in benchmarker.predicted_summary_data.keys():
                summarydict = benchmarker.predicted_summary_data[pkey][i][qubits]
                fname = outdir + '/predictions/{}/summarydata/'.format(pkey) + '{}-{}.txt'.format(i, j)
                with open(fname, 'w') as f:
                    _json.dump(summarydict, f, indent=4)

    for dskey in benchmarker.multids.keys():
        fdir = outdir + '/datasets/{}'.format(dskey)
        _os.makedirs(fdir)
        for dsind in benchmarker.multids[dskey].keys():
            fname = fdir + '/ds{}.txt'.format(dsind)
            _io.write_dataset(fname, benchmarker.multids[dskey][dsind], fixed_column_mode=False)


def create_benchmarker(dsfilenames, predictions={}, test_stability=True, auxtypes=[], verbosity=1):
    benchmarker = load_data_into_benchmarker(dsfilenames, verbosity=verbosity)
    if test_stability:
        if verbosity > 0:
            print(" - Running stability analysis...", end='')
        benchmarker.test_pass_stability(formatdata=True, verbosity=0)
        if verbosity > 0:
            print("complete.")

    benchmarker.create_summary_data(predictions=predictions, auxtypes=auxtypes)

    return benchmarker

# Todo : just make this and create_benchmarker a single function? This import has been superceded
# by load_benchmarker


def load_data_into_benchmarker(dsfilenames=None, summarydatasets_filenames=None, summarydatasets_folder=None,
                               predicted_summarydatasets_folders={}, verbosity=1):
    """
    todo

    """
    if len(predicted_summarydatasets_folders) > 0:
        assert(summarydatasets_folder is not None)
        #if len(predicted_summarydatasets_folders) > 1:
        #    raise NotImplementedError("This is not yet supported!")

    if dsfilenames is not None:

        # If it is a filename, then we import the dataset from file.
        if isinstance(dsfilenames, str):
            dsfilenames = [dsfilenames, ]
        elif not isinstance(dsfilenames, list):
            raise ValueError("dsfilenames must be a str or a list of strings!")

        mds = _mds.MultiDataSet()
        for dsfn_ind, dsfn in enumerate(dsfilenames):

            if dsfn[-4:] == '.txt':
                print(dsfn)
                mds.add_dataset(dsfn_ind, _io.load_dataset(dsfn,
                                                           collision_action='keepseparate',
                                                           record_zero_counts=False,
                                                           ignore_zero_count_lines=False,
                                                           verbosity=verbosity))

            elif dsfn[-4:] == '.pkl':

                if verbosity > 0:
                    print(" - Loading DataSet from pickle file...", end='')
                with open(dsfn, 'rb') as f:
                    mds.add_dataset(dsfn_ind, _pickle.load(f))
                if verbosity > 0:
                    print("complete.")

            else:
                raise ValueError("File must end in .pkl or .txt!")

        # # If it isn't a string, we assume that `dsfilenames` is a DataSet.
        # else:

        #     ds = dsfilenames

        if verbosity > 0: print(" - Extracting metadata from the DataSet...", end='')

        # To store the aux information about the RB experiments.
        all_spec_filenames = []
        # circuits_for_specfile = {}
        # outdslist = []

        # We go through the dataset and extract all the necessary auxillary information.
        for circ in mds[mds.keys()[0]].keys():

            # The spec filename or names for this circuits
            specfns_forcirc = mds.auxInfo[circ]['spec']
            # The RB length for this circuit
            # try:
            # l = mds.auxInfo[circ]['depth']
            # except:
            # l = mds.auxInfo[circ]['length']
            # The target bitstring for this circuit.
            # target = mds.auxInfo[circ]['target']

            # This can be a string (a single spec filename) or a list, so make always a list.
            if isinstance(specfns_forcirc, str):
                specfns_forcirc = [specfns_forcirc, ]

            for sfn_forcirc in specfns_forcirc:
                # If this is the first instance of seeing this filename then...
                if sfn_forcirc not in all_spec_filenames:
                    # ... we store it in the list of all spec filenames to import later.
                    all_spec_filenames.append(sfn_forcirc)
                    # And it won't yet be a key in the circuits_for_specfile dict, so we add it.
            #         circuits_for_specfile[sfn_forcirc] = {}

            #     # If we've not yet had this length for that spec filename, we add that as a key.
            #     if l not in circuits_for_specfile[sfn_forcirc].keys():
            #         circuits_for_specfile[sfn_forcirc][l] = []

            #     # We add the circuit and target output to the dict for the corresponding spec files.
            #     circuits_for_specfile[sfn_forcirc][l].append((circ, target))

            # circ_specindices = []
            # for sfn_forcirc in specfns_forcirc:
            #     circ_specindices.append(all_spec_filenames.index(sfn_forcirc))

        if verbosity > 0:
            print("complete.")
            print(" - Reading in the metadata from the extracted filenames...", end='')

        # We put RB specs that we create via file import (and the circuits above) into this dict
        rbspecdict = {}

        # We look for spec files in the same directory as the datafiles, so we find what that is.
        # THIS REQUIRES ALL THE FILES TO BE IN THE SAME DIRECTORY
        directory = dsfilenames[0].split('/')
        directory = '/'.join(directory[: -1])
        if len(directory) > 0:
            directory += '/'

        for specfilename in all_spec_filenames:

            # Import the RB spec file.
            rbspec = load_benchmarkspec(directory + specfilename)
            # Add in the circuits that correspond to each spec, extracted from the dataset.
            # rbspec.add_circuits(circuits_for_specfile[specfilename])
            # Record the spec in a list, to be given to an RBAnalyzer object.
            rbspecdict[specfilename] = rbspec

        if verbosity > 0:
            print("complete.")
            print(" - Recording all of the data in a Benchmarker...", end='')

        # Put everything into an RBAnalyzer object, which is a container for RB data, and return this.
        benchmarker = _benchmarker.Benchmarker(rbspecdict, ds=mds, summary_data=None)

        if verbosity > 0: print("complete.")

        return benchmarker

    elif (summarydatasets_filenames is not None) or (summarydatasets_folder is not None):

        rbspecdict = {}

        # If a dict, its just the keys of the dict that are the rbspec file names.
        if summarydatasets_filenames is not None:

            specfiles = list(summarydatasets_filenames.keys())

        # If a folder, we look for files in that folder with the standard name format.
        elif summarydatasets_folder is not None:
            specfiles = []
            specfilefound = True
            i = 0
            while specfilefound:
                try:
                    filename = summarydatasets_folder + "/spec{}.txt".format(i)
                    with open(filename, 'r') as f:
                        if verbosity > 0:
                            print(filename + " found")
                    specfiles.append(filename)
                    i += 1
                except:
                    specfilefound = False
                    if verbosity > 0:
                        print(filename + " not found so terminating spec file search.")

        for sfn_ind, specfilename in enumerate(specfiles):

            rbspec = load_benchmarkspec(specfilename)
            rbspecdict[sfn_ind] = rbspec

        summary_data = {}
        predicted_summary_data = {pkey: {} for pkey in predicted_summarydatasets_folders.keys()}

        for i, (specfilename, rbspec) in enumerate(zip(specfiles, rbspecdict.values())):

            structure = rbspec.get_structure()
            summary_data[i] = {}
            for pkey in predicted_summarydatasets_folders.keys():
                predicted_summary_data[pkey][i] = {}

            if summarydatasets_filenames is not None:
                sds_filenames = summarydatasets_filenames[specfilename]
            elif summarydatasets_folder is not None:
                sds_filenames = [summarydatasets_folder + '/{}-{}.txt'.format(i, j) for j in range(len(structure))]
                predsds_filenames_dict = {}
                for pkey, pfolder in predicted_summarydatasets_folders.items():
                    predsds_filenames_dict[pkey] = [pfolder + '/{}-{}.txt'.format(i, j) for j in range(len(structure))]

            for sdsfn, qubits in zip(sds_filenames, structure):
                summary_data[i][qubits] = import_rb_summary_data(sdsfn, len(qubits), verbosity=verbosity)

            for pkey, predsds_filenames in predsds_filenames_dict.items():
                for sdsfn, qubits in zip(predsds_filenames, structure):
                    predicted_summary_data[pkey][i][qubits] = import_rb_summary_data(
                        sdsfn, len(qubits), verbosity=verbosity)

        benchmarker = _benchmarker.Benchmarker(rbspecdict, ds=None, summary_data=summary_data,
                                               predicted_summary_data=predicted_summary_data)

        return benchmarker

    else:
        raise ValueError("Either a filename for a DataSet or filenames for a set of RBSpecs "
                         + "and RBSummaryDatasets must be provided!")


def load_benchmarkspec(filename, circuitsfilename=None):
    """
    todo

    """
    #d = {}
    with open(filename) as f:
        d = _json.load(f)
        # for line in f:
        #     if len(line) > 0 and line[0] != '#':
        #         line = line.strip('\n')
        #         line = line.split(' ', 1)
        #         try:
        #             d[line[0]] = _ast.literal_eval(line[1])
        #         except:
        #             d[line[0]] = line[1]

    #assert(d.get('type', None) == 'rb'), "This is for importing RB specs!"

    try:
        rbtype = d['type']
    except:
        raise ValueError("Input file does not contain a line specifying the RB type!")
    assert(isinstance(rbtype, str)), "The RB type (specified as rbtype) must be a string!"

    try:
        structure = d['structure']
    except:
        raise ValueError("Input file does not contain a line specifying the structure!")
    if isinstance(structure, list):
        structure = tuple([tuple(qubits) for qubits in structure])
    assert(isinstance(structure, tuple)), "The structure must be a tuple!"

    try:
        sampler = d['sampler']
    except:
        raise ValueError("Input file does not contain a line specifying the circuit layer sampler!")
    assert(isinstance(sampler, str)), "The sampler name must be a string!"

    samplerargs = d.get('samplerargs', None)
    depths = d.get('depths', None)
    numcircuits = d.get('numcircuits', None)
    subtype = d.get('subtype', None)

    if samplerargs is not None:
        assert(isinstance(samplerargs, dict)), "The samplerargs must be a dict!"

    if depths is not None:
        assert(isinstance(depths, list) or isinstance(depths, tuple)), "The depths must be a list or tuple!"

    if numcircuits is not None:
        assert(isinstance(numcircuits, list) or isinstance(numcircuits, int)), "numcircuits must be an int or list!"

    spec = _sample.BenchmarkSpec(rbtype, structure, sampler, samplerargs, depths=depths,
                                 numcircuits=numcircuits, subtype=subtype)

    return spec


def write_benchmarkspec(spec, filename, circuitsfilename=None, warning=1):
    """
    todo

    """
    if spec.circuits is not None:
        if circuitsfilename is not None:
            circuitlist = [circ for sublist in [spec.circuits[l] for l in spec.depths] for circ in sublist]
            _io.write_circuit_list(circuitsfilename, circuitlist)
        elif warning > 0:
            _warnings.warn("The circuits recorded in this RBSpec are not being written to file!")

    # with open(filename, 'w') as f:
    #     f.write('type rb\n')
    #     f.write('rbtype ' + rbspec._rbtype + '\n')
    #     f.write('structure ' + str(rbspec._structure) + '\n')
    #     f.write('sampler ' + rbspec._sampler + '\n')
    #     f.write('lengths ' + str(rbspec._lengths) + '\n')
    #     f.write('numcircuits ' + str(rbspec._numcircuits) + '\n')
    #     f.write('rbsubtype ' + str(rbspec._rbsubtype) + '\n')
    #     f.write('samplerargs ' + str(rbspec._samplerargs) + '\n')

    specdict = spec.to_dict()
    del specdict['circuits']  # Don't write the circuits to this file.

    with open(filename, 'w') as f:
        _json.dump(specdict, f, indent=4)


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

                elif line[0: numqubits + 2] == ['rblength', ] + ['hd{}c'.format(i) for i in range(numqubits + 1)]:

                    auxind = numqubits + 2
                    if datatype == 'auto':
                        datatype = 'hamming_distance_counts'
                    else:
                        assert(datatype == 'hamming_distance_counts'), "The data format appears to be Hamming " + \
                            "distance counts!"

                elif line[0: numqubits + 2] == ['rblength', ] + ['hd{}p'.format(i) for i in range(numqubits + 1)]:

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
                else:
                    auxlabels = []

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
                        if key != 'target' and key != 'circuit':
                            aux[key][l].append(_ast.literal_eval(line[4 + i]))
                        else:
                            if key == 'target':
                                aux[key][l].append(line[4 + i])
                            if key == 'circuit':
                                aux[key][l].append(_cir.Circuit(line[4 + i]))

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
                        if key != 'target' and key != 'circuit':
                            aux[key][l].append(_ast.literal_eval(line[3 + i]))
                        else:
                            if key == 'target':
                                aux[key][l].append(line[3 + i])
                            if key == 'circuit':
                                aux[key][l].append(_cir.Circuit(line[3 + i]))

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

                    hamming_distance_counts[l].append([float(line[1 + i]) for i in range(0, numqubits + 1)])

                    if len(aux) > 0:
                        assert(line[numqubits + 2] == '#'), "Auxillary data must be divided from the core data!"
                    for i, key in enumerate(auxlabels):
                        if key != 'target' and key != 'circuit':
                            aux[key][l].append(_ast.literal_eval(line[numqubits + 3 + i]))
                        else:
                            if key == 'target':
                                aux[key][l].append(line[numqubits + 3 + i])
                            if key == 'circuit':
                                aux[key][l].append(line[numqubits + 3 + i])
                                #aux[key][l].append(_cir.Circuit(line[numqubits + 3 + i]))
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
                    dataline = str(l) + ''.join([' ' + str(c[i]) for i in range(0, numqubits + 1)])

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
