#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""Core integrated routines for detecting and characterizing crosstalk"""

import numpy as _np
from . import objects as _obj
from ... import objects as _pygobjs
from ... import io as _pygio
import pcalg
from gsq.ci_tests import ci_test_dis
import collections
from sympy import isprime


def tuple_replace_at_index(tup, ix, val):
    return tup[:ix] + (val,) + tup[ix + 1:]


def load_pygsti_dataset(filename):
    """
    Loads a pygsti dataset from file.

    This is a wrapper that just checks the first line, and replaces it with the newer outcome specification
    format if its the old type.
    """
    try:
        # file = open(filename, "r")
        open(filename, "r")
    except IOError:
        print("File not found, or other file IO error.")

    # lines = file.readlines()
    # file.close()

    # if lines[0] == "## Columns = 00 count, 01 count, 10 count, 11 count\n":
    # 	lines[0] = "## Columns = 0:0 count, 0:1 count, 1:0 count, 1:1 count\n"
    # 	file = open(filename, "w")
    # 	file.writelines(lines)
    # 	file.close()

    data = _pygio.load_dataset(filename)

    return data


def flatten(l):
    """
    Flattens an irregualr list.
    From https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    """
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def form_ct_data_matrix(ds, number_of_regions, settings, filter_lengths=[]):
    # This converts a DataSet to an array since the code below uses arrays
    if type(ds) == _pygobjs.dataset.DataSet:

        opstr = ds.keys()[0]
        temp = ds.auxInfo[opstr]['settings']
        num_settings = len(temp)

        settings_shape = _np.shape(settings)
        # Check that settings is a list of length number_of_regions
        assert((len(settings_shape) == 1) and (settings_shape[0] == number_of_regions))
        "settings should be a list of the same length as number_of_regions."

        dscopy = ds.copy_nonstatic()
        # filter out lengths not in filter_lengths
        if len(filter_lengths) > 0:
            for k in dscopy.keys():
                if len(k) not in filter_lengths:
                    dscopy.remove([k])

        dscopy.done_adding_data()

        # num columns = number of settings + number of regions (b/c we assume one outcome per region)
        #num_columns = num_settings + number_of_regions

        num_data = len(dscopy.keys())

        data = []
        collect_settings = {key: [] for key in range(num_settings)}
        for row in range(num_data):
            opstr = dscopy.keys()[row]

            templine_set = [0] * num_settings
            settings_row = dscopy.auxInfo[opstr]['settings']

            for key in settings_row:
                if len(key) == 1:  # single region/qubit gate
                    templine_set[key[0]] = settings_row[key]
                    collect_settings[key[0]].append(settings_row[key])
                else:  # two-region/two-qubit gate
                    print("Two qubit gate, not sure what to do!!")  # TODO
                    return

            outcomes_row = dscopy[opstr]
            for outcome in outcomes_row:
                templine_out = [0] * number_of_regions

                if len(outcome[0]) == 1:
                    # outcomes labeled by bitstrings
                    for r in range(number_of_regions):
                        templine_out[r] = int(outcome[0][0][r])
                    num_rep = int(outcome[2])

                    templine_out.append(templine_set)
                    flattened_line = list(flatten(templine_out))
                else:
                    # outcomes labeled by tuples of bits
                    for r in range(number_of_regions):
                        templine_out[r] = int(outcome[0][r])
                    num_rep = int(outcome[2])

                    templine_out.append(templine_set)
                    flattened_line = list(flatten(templine_out))

                for r in range(num_rep):
                    data.append(flattened_line)

        #num_seqs = [len(set(collect_settings[i])) for i in range(num_settings)]

        data = _np.asarray(data)

    # if the dataset is specified by a string assume its a filename with a saved numpy array
    elif type(ds) == str:
        data = _np.loadtxt(ds)
        data = data.astype(int)

        data_shape = _np.shape(data)
        settings_shape = _np.shape(settings)

        # Check that the input data is a 2D array
        assert(len(data_shape) == 2)
        "Input data format is incorrect!If the input is a numpy array it must be 2-dimensional."

        # Check that settings is a list of length number_of_regions
        assert((len(settings_shape) == 1) and (settings_shape[0] == number_of_regions))
        "settings should be a list of the same length as number_of_regions."

        # The number of columns in the data must be consistent with the number of settings
        assert(data_shape[1] == (sum(settings) + number_of_regions))
        "Mismatch between the number of settings specified for each region and the number of columns in data"

        num_data = data_shape[0]
        #num_columns = data_shape[1]

    # if neither a pygsti data set or string, assume a numpy array was passed in
    else:
        data_shape = _np.shape(ds)
        settings_shape = _np.shape(settings)

        # Check that the input data is a 2D array
        assert(len(data_shape) == 2)
        "Input data format is incorrect!If the input is a numpy array it must be 2-dimensional."

        # Check that settings is a list of length number_of_regions
        assert((len(settings_shape) == 1) and (settings_shape[0] == number_of_regions))
        "settings should be a list of the same length as number_of_regions."

        # The number of columns in the data must be consistent with the number of settings
        assert(data_shape[1] == (sum(settings) + number_of_regions))
        "Mismatch between the number of settings specified for each region and the number of columns in data"

        data = ds

    data_shape = _np.shape(data)

    return data, data_shape


def do_basic_crosstalk_detection(ds, number_of_regions, settings, confidence=0.95, verbosity=1, name=None,
                                 assume_independent_settings=True, filter_lengths=[]):
    """
    Implements crosstalk detection on multiqubit data (fine-grained data with entries for each experiment).

    Parameters
    ----------
    ds : pyGSTi DataSet or numpy array
        The multiqubit data to analyze. If this is a numpy array, it must contain time series data and it must
        be 2-dimensional with each entry being a sequence of settings and measurment outcomes for each qubit region.
        A region is a set of one or more qubits and crosstalk is assessed between regions. The first n entries are
        the outcomes and the following entries are settings.

    number_of_regions: int, number of regions in experiment

    settings: list of length number_of_regions, indicating the number of settings for each qubit region.

    confidence : float, optional

    verbosity : int, optional

    name : str, optional

    filter_lengths : list of lengths. If this is not empty the dataset will be filtered and the analysis will only be
        done on the sequences of lengths specified in this list. This argument is only used if the dataset is passed in
        as a pyGSTi DataSet

    Returns
    -------
    results : CrosstalkResults object
        The results of the crosstalk detection analysis. This contains: output skeleton graph and DAG from
        PC Algorithm indicating regions with detected crosstalk, all of the input information.

    """
    # -------------------------- #
    # Format and check the input #
    # -------------------------- #

    # This converts a DataSet to an array since the code below uses arrays
    # -------------------------- #

    if type(ds) != _pygobjs.dataset.DataSet:

        data_shape = _np.shape(ds)
        settings_shape = _np.shape(settings)

        # Check that the input data is a 2D array
        assert(len(data_shape) == 2), \
            "Input data format is incorrect!If the input is a numpy array it must be 2-dimensional."

        # Check that settings is a list of length number_of_regions
        assert((len(settings_shape) == 1) and (settings_shape[0] == number_of_regions))
        "settings should be a list of the same length as number_of_regions."

        # The number of columns in the data must be consistent with the number of settings
        assert(data_shape[1] == (sum(settings) + number_of_regions))
        "Mismatch between the number of settings specified for each region and the number of columns in data"

        data = ds
        num_data = data_shape[0]
        num_columns = data_shape[1]

    # This converts a DataSet to an array, as the code below uses arrays
    if type(ds) == _pygobjs.dataset.DataSet:

        opstr = ds.keys()[0]
        temp = ds.auxInfo[opstr]['settings']
        num_settings = len(temp)

        settings_shape = _np.shape(settings)
        # Check that settings is a list of length number_of_regions
        assert((len(settings_shape) == 1) and (settings_shape[0] == number_of_regions))
        "settings should be a list of the same length as number_of_regions."

        dscopy = ds.copy_nonstatic()
        # filter out lengths not in filter_lengths
        if len(filter_lengths) > 0:
            for k in dscopy.keys():
                if len(k) not in filter_lengths:
                    dscopy.remove([k])

        dscopy.done_adding_data()

        # num columns = number of settings + number of regions (b/c we assume one outcome per region)
        num_columns = num_settings + number_of_regions

        num_data = len(dscopy.keys())

        data = []
        collect_settings = {key: [] for key in range(num_settings)}
        for row in range(num_data):
            opstr = dscopy.keys()[row]

            templine_set = [0] * num_settings
            settings_row = dscopy.auxInfo[opstr]['settings']

            for key in settings_row:
                if len(key) == 1:  # single region/qubit gate
                    templine_set[key[0]] = settings_row[key]
                    collect_settings[key[0]].append(settings_row[key])
                else:  # two-region/two-qubit gate
                    print("Two qubit gate, not sure what to do!!")  # TODO
                    return
            outcomes_row = dscopy[opstr]

            for outcome in outcomes_row:
                templine_out = [0] * number_of_regions

                if len(outcome[0]) == 1:
                    # outcomes labeled by bitstrings
                    for r in range(number_of_regions):
                        templine_out[r] = int(outcome[0][0][r])

                    num_rep = int(outcome[2])

                    templine_out.append(templine_set)
                    flattened_line = list(flatten(templine_out))
                else:
                    # outcomes labeled by tuples of bits
                    for r in range(number_of_regions):
                        templine_out[r] = int(outcome[0][1][0][r])  # templine_out[r] = int(outcome[0][r])
                        # print(templine_out[r])
                    num_rep = int(outcome[2])

                    templine_out.append(templine_set)
                    flattened_line = list(flatten(templine_out))

                for r in range(num_rep):
                    data.append(flattened_line)

        data = _np.asarray(data)

    # if the dataset is specified by a string assume its a filename with a saved numpy array
    elif type(ds) == str:
        data = _np.loadtxt(ds)
        data = data.astype(int)

        data_shape = _np.shape(data)
        settings_shape = _np.shape(settings)

        # Check that the input data is a 2D array
        assert(len(data_shape) == 2)
        "Input data format is incorrect!If the input is a numpy array it must be 2-dimensional."

        # Check that settings is a list of length number_of_regions
        assert((len(settings_shape) == 1) and (settings_shape[0] == number_of_regions))
        "settings should be a list of the same length as number_of_regions."

        # The number of columns in the data must be consistent with the number of settings
        assert(data_shape[1] == (sum(settings) + number_of_regions))
        "Mismatch between the number of settings specified for each region and the number of columns in data"

        num_data = data_shape[0]
        num_columns = data_shape[1]

    # if neither a pygsti data set or string, assume a numpy array was passed in
    else:
        data_shape = _np.shape(ds)
        settings_shape = _np.shape(settings)

        # Check that the input data is a 2D array
        assert(len(data_shape) == 2)
        "Input data format is incorrect!If the input is a numpy array it must be 2-dimensional."

        # Check that settings is a list of length number_of_regions
        assert((len(settings_shape) == 1) and (settings_shape[0] == number_of_regions))
        "settings should be a list of the same length as number_of_regions."

        # The number of columns in the data must be consistent with the number of settings
        assert(data_shape[1] == (sum(settings) + number_of_regions))
        "Mismatch between the number of settings specified for each region and the number of columns in data"

        data = ds

    data_shape = _np.shape(data)
    num_data = data_shape[0]
    num_columns = data_shape[1]

    # dump the array form of the dataset into a file for diagnostics
    _np.savetxt('dataset_dump.txt', data, fmt='%d')

    # --------------------------------------------------------- #
    # Prepare a results object, and store the input information #
    # --------------------------------------------------------- #

    # Initialize an empty results object.
    results = _obj.CrosstalkResults()

    # Records input information into the results object.
    results.name = name
    results.data = data
    if type(ds) == _pygobjs.dataset.DataSet:
        results.pygsti_ds = dscopy
    results.number_of_regions = number_of_regions
    results.settings = settings
    results.number_of_datapoints = num_data
    results.number_of_columns = num_columns
    results.confidence = confidence

    # ------------------------------------------------- #
    #     Calculate the causal graph skeleton           #
    # ------------------------------------------------- #

    if assume_independent_settings:
        # List edges between settings so that these can be ignored when constructing skeleton
        ignore_edges = []
        for set1 in range(number_of_regions, num_columns):
            for set2 in range(number_of_regions, num_columns):
                if set1 > set2:
                    ignore_edges.append((set1, set2))
    else:
        ignore_edges = []

    print("Calculating causal graph skeleton ...")
    (skel, sep_set) = pcalg.estimate_skeleton(ci_test_dis, data, 1 - confidence, ignore_edges)

    print("Calculating directed causal graph ...")
    g = pcalg.estimate_cpdag(skel_graph=skel, sep_set=sep_set)

    # Store skeleton and graph in results object
    results.skel = skel
    results.sep_set = sep_set
    results.graph = g

    # Calculate the column index for the first setting for each region
    setting_indices = {x: number_of_regions + sum(settings[:x]) for x in range(number_of_regions)}
    results.setting_indices = setting_indices

    node_labels = {}
    cnt = 0
    for col in range(num_columns):
        if col < number_of_regions:
            node_labels[cnt] = r'R$_{%d}$' % col
            cnt += 1
#            node_labels.append("$%d^O$" % col)
        else:
            for region in range(number_of_regions):
                if col in range(setting_indices[region],
                                (setting_indices[(region + 1)] if region < (number_of_regions - 1) else num_columns)):
                    break
            node_labels[cnt] = r'S$_{%d}^{(%d)}$' % (region, (col - setting_indices[region]))
            cnt += 1
            #node_labels.append("%d^S_{%d}$" % (region, (col-setting_indices[region]+1)))

    results.node_labels = node_labels

    # Generate crosstalk detected matrix and assign weight to each edge according to TVD variation in distribution of
    # destination variable when source variable is varied.
    print("Examining edges for crosstalk ...")

    cmatrix = _np.zeros((number_of_regions, number_of_regions))
    edge_weights = _np.zeros(len(g.edges()))
    is_edge_ct = _np.zeros(len(g.edges()))
    edge_tvds = {}
    source_levels_dict = {}
    max_tvds = {}
    median_tvds = {}
    max_tvd_explanations = {}

    def _setting_range(x):
        return range(
            setting_indices[x],
            setting_indices[x + 1] if x < (number_of_regions - 1) else num_columns
        )

    for idx, edge in enumerate(g.edges()):
        source = edge[0]
        dest = edge[1]

        if verbosity > 1:
            print("** Edge: ", edge, " **")

        # For each edge, decide if it represents crosstalk
        #   Crosstalk is:
        #       (1) an edge between outcomes on different regions
        #       (2) an edge between a region's outcome and a setting of another region

        # source and destination are results
        if source < number_of_regions and dest < number_of_regions:
            cmatrix[source, dest] = 1
            is_edge_ct[idx] = 1
            print("Crosstalk detected. Regions " + str(source) + " and " + str(dest))

        # source is a result, destination is a setting
        if source < number_of_regions and dest >= number_of_regions:
            if dest not in range(setting_indices[source],
                                 (setting_indices[(source + 1)] if source < (number_of_regions - 1) else num_columns)):
                # make sure destination is not a setting for that region
                for region in range(number_of_regions):
                    # search among regions to find the one that this destination setting belongs to
                    if dest in range(setting_indices[region],
                                     (setting_indices[(region + 1)] if region < (number_of_regions - 1)
                                      else num_columns)):
                        break
                cmatrix[source, region] = 1
                is_edge_ct[idx] = 1
                print("Crosstalk detected. Regions " + str(source) + " and " + str(region))

        # source is a setting, destination is a result
        if source >= number_of_regions and dest < number_of_regions:
            if source not in range(setting_indices[dest],
                                   (setting_indices[(dest + 1)] if dest < (number_of_regions - 1) else num_columns)):
                # make sure source is not a setting for that region
                for region in range(number_of_regions):
                    # search among regions to find the one that this source setting belongs to
                    if source in range(setting_indices[region],
                                       (setting_indices[(region + 1)] if region < (number_of_regions - 1)
                                        else num_columns)):
                        break
                cmatrix[region, dest] = 1
                is_edge_ct[idx] = 1
                print("Crosstalk detected. Regions " + str(region) + " and " + str(dest))

        # For each edge in causal graph that represents crosstalk, calculate the TVD between distributions of dependent
        # variable when other variable is varied

        if is_edge_ct[idx] == 1:

            # the TVD calculation depends on what kind of crosstalk it is.

            # source and destination are results OR source is a result, destination is a setting
            if (source < number_of_regions and dest < number_of_regions) or \
               (source < number_of_regions and dest >= number_of_regions):
                source_levels, level_cnts = _np.unique(data[:, source], return_counts=True)
                num_levels = len(source_levels)

                if any(level_cnts < 10):
                    print((" ***   Warning: n<10 data points for some levels. "
                           "TVD calculations may have large error bars."))

                tvds = _np.zeros((num_levels, num_levels))
                calculated_tvds = []
                for i in range(num_levels):
                    for j in range(i):

                        marg1 = data[data[:, source] == source_levels[i], dest]
                        marg2 = data[data[:, source] == source_levels[j], dest]
                        n1, n2 = len(marg1), len(marg2)

                        marg1_levels, marg1_level_cnts = _np.unique(marg1, return_counts=True)
                        marg2_levels, marg2_level_cnts = _np.unique(marg2, return_counts=True)

                        #print(marg1_levels, marg1_level_cnts)
                        #print(marg2_levels, marg2_level_cnts)

                        tvd_sum = 0.0
                        for lidx, level in enumerate(marg1_levels):
                            temp = _np.where(marg2_levels == level)
                            if len(temp[0]) == 0:
                                tvd_sum += marg1_level_cnts[lidx] / n1
                            else:
                                tvd_sum += _np.fabs(marg1_level_cnts[lidx] / n1 - marg2_level_cnts[temp[0][0]] / n2)

                        tvds[i, j] = tvds[j, i] = tvd_sum / 2.0
                        calculated_tvds.append(tvds[i, j])

                edge_tvds[idx] = tvds
                source_levels_dict[idx] = source_levels
                max_tvds[idx] = _np.max(calculated_tvds)
                median_tvds[idx] = _np.median(calculated_tvds)

            # source is a setting, destination is a result
            else:
                source_levels, level_cnts = _np.unique(data[:, source], return_counts=True)
                num_levels = len(source_levels)

                if any(level_cnts < 10):
                    print((" ***   Warning: n<10 data points for some levels. "
                           "TVD calculations may have large error bars."))

                tvds = _np.zeros((num_levels, num_levels))
                max_dest_levels = _np.zeros((num_levels, num_levels))
                calculated_tvds = []
                for i in range(num_levels):
                    for j in range(i):

                        marg1 = data[data[:, source] == source_levels[i], ]
                        marg2 = data[data[:, source] == source_levels[j], ]

                        if(settings[dest] > 1):
                            print(('Region {} has more than one setting -- '
                                   'TVD code not implemented yet for this case').format(dest))
                            edge_tvds[idx] = tvds
                            source_levels_dict[idx] = source_levels
                        else:
                            dest_setting = setting_indices[dest]
                            dest_levels_i, dest_level_i_cnts = _np.unique(marg1[:, dest_setting], return_counts=True)
                            dest_levels_j, dest_level_j_cnts = _np.unique(marg2[:, dest_setting], return_counts=True)

                            common_dest_levels = list(set(dest_levels_i).intersection(dest_levels_j))

                            if common_dest_levels == []:
                                # No common settings on the destination regions for this combination of settings
                                # for the source region
                                # No sensible TVD here
                                tvds[i, j] = tvds[j, i] = -1
                            else:
                                max_tvd = 0
                                max_dest_level = 0
                                for dest_level in common_dest_levels:
                                    marg1d = marg1[marg1[:, dest_setting] == dest_level, dest]
                                    marg2d = marg2[marg2[:, dest_setting] == dest_level, dest]

                                    n1, n2 = len(marg1d), len(marg2d)

                                    marg1d_levels, marg1d_level_cnts = _np.unique(marg1d, return_counts=True)
                                    marg2d_levels, marg2d_level_cnts = _np.unique(marg2d, return_counts=True)

                                    #print(marg1_levels, marg1_level_cnts)
                                    #print(marg2_levels, marg2_level_cnts)

                                    tvd_sum = 0.0
                                    for lidx, level in enumerate(marg1d_levels):
                                        temp = _np.where(marg2d_levels == level)
                                        if len(temp[0]) == 0:
                                            tvd_sum += marg1d_level_cnts[lidx] / n1
                                        else:
                                            tvd_sum += _np.fabs(marg1d_level_cnts[lidx] / n1
                                                                - marg2d_level_cnts[temp[0][0]] / n2)

                                    if tvd_sum > max_tvd:
                                        max_tvd = tvd_sum
                                        max_dest_level = dest_level

                                tvds[i, j] = tvds[j, i] = max_tvd / 2.0
                                calculated_tvds.append(tvds[i, j])
                                max_dest_levels[i, j] = max_dest_levels[j, i] = max_dest_level

                edge_tvds[idx] = tvds
                source_levels_dict[idx] = source_levels
                max_tvds[idx] = _np.max(calculated_tvds)
                median_tvds[idx] = _np.median(calculated_tvds)  # take median over the calculated TVDs vector instead of
                # tvds matrix since that might have -1 elements that will skew median

                if max_tvds[idx] > 0:
                    i = _np.floor_divide(_np.argmax(tvds), num_levels)
                    j = _np.mod(_np.argmax(tvds), num_levels)

                    source_setting1 = source_levels[i]
                    source_setting2 = source_levels[j]
                    dest_setting = max_dest_levels[i, j]

                    # The following assumes each region is a single qubit -- need to generalize # TODO
                    source_qubit = source - results.number_of_regions
                    dest_qubit = dest

                    if results.pygsti_ds is None:
                        max_tvd_explanations[idx] = ("Max TVD = {}. Settings on source qubit: {}, {}. Setting on "
                                                     "destination qubit: {}").format(max_tvds[idx], source_setting1,
                                                                                     source_setting2, dest_setting)
                    else:
                        source_setting1_seq = 0
                        for key in results.pygsti_ds.keys():
                            if results.pygsti_ds.auxInfo[key]['settings'][(source_qubit,)] == source_setting1:
                                key_copy = key.copy(editable=True)
                                key_copy.delete_lines([i for i in range(key.number_of_lines()) if i != source_qubit])

                                source_setting1_seq = key_copy
                                break

                        source_setting2_seq = 0
                        for key in results.pygsti_ds.keys():
                            if results.pygsti_ds.auxInfo[key]['settings'][(source_qubit,)] == source_setting2:
                                key_copy = key.copy(editable=True)
                                key_copy.delete_lines([i for i in range(key.number_of_lines()) if i != source_qubit])

                                source_setting2_seq = key_copy
                                break

                        dest_seq = 0
                        for key in results.pygsti_ds.keys():
                            if results.pygsti_ds.auxInfo[key]['settings'][(dest_qubit,)] == dest_setting:
                                key_copy = key.copy(editable=True)
                                key_copy.delete_lines([i for i in range(key.number_of_lines()) if i != dest_qubit])
                                dest_seq = key_copy
                                break

                        for key in results.pygsti_ds.keys():
                            if (results.pygsti_ds.auxInfo[key]['settings'][(source_qubit,)] == source_setting1) and \
                               (results.pygsti_ds.auxInfo[key]['settings'][(dest_qubit,)] == dest_setting):
                                res1 = results.pygsti_ds[key]

                            if (results.pygsti_ds.auxInfo[key]['settings'][(source_qubit,)] == source_setting2) and \
                               (results.pygsti_ds.auxInfo[key]['settings'][(dest_qubit,)] == dest_setting):
                                res2 = results.pygsti_ds[key]

                        max_tvd_explanations[idx] = \
                            ("Max TVD = {}. Settings on source qubit: {}, {}. Setting on destination qubit: {}\n"
                             "    Sequences on source (qubit {})\n {}\n {}\n    Sequence on destination (qubit {})\n"
                             " {}\n"
                             "    Results when source={}, destination={}:\n"
                             " {}\n"
                             "    Results when source={}, destination={}:\n {}\n").format(
                                 max_tvds[idx], source_setting1, source_setting2, dest_setting, source_qubit,
                                 source_setting1_seq, source_setting2_seq, dest_qubit, dest_seq, source_setting1,
                                 dest_setting, res1, source_setting2, dest_setting, res2)
                else:
                    max_tvd_explanations[idx] = "Max TVD = 0. Experiment not rich enough to calculate TVD."

            if any(level_cnts < 10):
                print(" ***   Warning: n<10 data points for some levels. TVD calculations may have large error bars.")

            tvds = _np.zeros((num_levels, num_levels))
            for i in range(num_levels):
                for j in range(i):

                    marg1 = data[data[:, source] == source_levels[i], dest]
                    marg2 = data[data[:, source] == source_levels[j], dest]
                    n1, n2 = len(marg1), len(marg2)

                    tvd_sum = 0.0
                    for lidx, level in enumerate(marg1_levels):
                        temp = _np.where(marg2_levels == level)
                        if len(temp[0]) == 0:
                            tvd_sum += marg1_level_cnts[lidx] / n1
                        else:
                            tvd_sum += _np.fabs(marg1_level_cnts[lidx] / n1 - marg2_level_cnts[temp[0][0]] / n2)

                    tvds[i, j] = tvds[j, i] = tvd_sum / 2.0

    results.cmatrix = cmatrix
    results.is_edge_ct = is_edge_ct
    results.crosstalk_detected = _np.sum(is_edge_ct) > 0
    results.edge_weights = edge_weights
    results.edge_tvds = edge_tvds
    results.max_tvds = max_tvds
    results.median_tvds = median_tvds
    results.max_tvd_explanations = max_tvd_explanations

    return results


"""

def crosstalk_detection_experiment(pspec, lengths, circuits_per_length, circuit_population_sz, idle_prob=0.1,
 structure='1Q', descriptor='A set of crosstalk detections experiments', verbosity=1):

# This is the original experiment design (Sampling from exhaustive experiment). This is deprecated.

    experiment_dict = {}
    experiment_dict['spec'] = {}
    experiment_dict['spec']['lengths'] = lengths
    experiment_dict['spec']['circuits_per_length'] = circuits_per_length
    experiment_dict['spec']['circuit_population_sz'] = circuit_population_sz
    experiment_dict['spec']['idle_prob'] = idle_prob
    experiment_dict['spec']['descriptor'] = descriptor
    experiment_dict['spec']['createdby'] = 'extras.crosstalk.crosstalk_detection_experiment'

    if isinstance(structure,str):
        assert(structure == '1Q'), "The only default `structure` option is the string '1Q'"
        structure = tuple([(q,) for q in pspec.qubit_labels])
        n = pspec.number_of_qubits
    else:
        assert(isinstance(structure,list) or isinstance(structure,tuple)), \
            "If not a string, `structure` must be a list or tuple."
        qubits_used = []
        for subsetQs in structure:
            assert(isinstance(subsetQs, list) or isinstance(subsetQs, tuple)), "SubsetQs must be a list or a tuple!"
            qubits_used = qubits_used + list(subsetQs)
            assert(len(set(qubits_used)) == len(qubits_used)), \
                "The qubits in the tuples/lists of `structure must all be unique!"

        assert(set(qubits_used).issubset(set(pspec.qubit_labels))), \
            "The qubits to benchmark must all be in the ProcessorSpec `pspec`!"
        n = len(qubits_used)

    experiment_dict['spec']['structure'] = structure
    experiment_dict['circuits'] = {}
    experiment_dict['settings'] = {}

    gates_available = list(pspec.models['target'].get_primitive_op_labels())
    gates_by_qubit = [[] for _ in range(0,n)]
    for i in range(0,len(gates_available)):
        for q in range(0,n):
            if gates_available[i].qubits == (q,):
                gates_by_qubit[q].append(gates_available[i])

    for lnum, l in enumerate(lengths):

        # generate menu of circuits for each qubit
        circuit_menu = [[] for _ in range(0,n)]
        for q in range(0,n):
            d = len(gates_by_qubit[q])

            if d**l < circuit_population_sz:
                print(('- Warning: circuit population specified too large for qubit {} -- '
                       'there will be redundant circuits').format(q))

            for rep in range(0,circuit_population_sz):
                singleQcirc = []
                for j in range(0,l):
                    r = _np.random.randint(0,d)
                    singleQcirc.append(gates_by_qubit[q][r])
                circuit_menu[q].append(singleQcirc)

        if verbosity > 0:
            print('- Sampling {} random circuits at length {} ({} of {} lengths)'.format(
                  circuits_per_length,l,lnum+1,len(lengths)))
            print('  - Number of circuits sampled = ',end='')

        # sample from this menu
        for j in range(circuits_per_length):
            circuit = _pygobjs.circuit.Circuit(num_lines=0, editable=True)

            settings = {}
            for q in range(0,n):
                r = _np.random.randint(0,circuit_population_sz)
                settings[(q,)] = lnum*(circuit_population_sz+1) + r + 1

                singleQcircuit = _pygobjs.circuit.Circuit(num_lines=1,line_labels=[q],editable=True)
                for layer in range(0,l):
                    singleQcircuit.insert_layer(circuit_menu[q][r][layer],layer)
                singleQcircuit.done_editing()

                circuit.tensor_circuit(singleQcircuit)

            # add sampled circuit to experiment dictionary, and the corresponding qubit settings
            experiment_dict['settings'][l,j] = settings

#
#   How are settings labeled?
#       For each circuit length, we label the random circuits of that length by consecutive integers, with the all
#       idle being the first number in the range for that circuit length.
#
#       So if circuit_population_sz is 10, and lengths are [10,20,30] then the the labels are:
#
#       0 (idle of length 10),  1, ...., 10  [all length 10 circuits]
#       11 (idle of length 20), 12, ..., 21  [all length 20 circuits]
#       22 (idle of length 30), 23, ..., 32  [all length 30 circuits]
#

            # for each line, replace sequence with an idle independently according to idle_prob
            if idle_prob>0:
                for q in range(0,n):
                    idle = bool(_np.random.binomial(1,idle_prob))
                    if idle:
                        circuit.replace_with_idling_line(q)
                        # Update the setting on that qubit to the idling setting (denoted by the length index)
                        experiment_dict['settings'][l,j][(q,)] = lnum*(circuit_population_sz+1)
                        if verbosity > 0: print('Idled {}'.format(q))

            circuit.done_editing()
            experiment_dict['circuits'][l,j] = circuit

            #print(circuit)

            if verbosity > 0: print(j+1,end=',')
        if verbosity >0: print('')

    return experiment_dict
"""


def crosstalk_detection_experiment2(pspec, lengths, circuits_per_length, circuit_population_sz, multiplier=3,
                                    idle_prob=0.1, structure='1Q',
                                    descriptor='A set of crosstalk detections experiments', verbosity=1):

    experiment_dict = {}
    experiment_dict['spec'] = {}
    experiment_dict['spec']['lengths'] = lengths
    experiment_dict['spec']['circuit_population_sz'] = circuit_population_sz
    experiment_dict['spec']['multiplier'] = multiplier
    experiment_dict['spec']['idle_prob'] = idle_prob
    experiment_dict['spec']['descriptor'] = descriptor
    experiment_dict['spec']['createdby'] = 'extras.crosstalk.crosstalk_detection_experiment2'

    if isinstance(structure, str):
        assert(structure == '1Q'), "The only default `structure` option is the string '1Q'"
        structure = tuple([(q,) for q in pspec.qubit_labels])
        n = pspec.number_of_qubits
    else:
        assert(isinstance(structure, list) or isinstance(structure, tuple)), \
            "If not a string, `structure` must be a list or tuple."
        qubits_used = []
        for subsetQs in structure:
            assert(isinstance(subsetQs, list) or isinstance(subsetQs, tuple)), "SubsetQs must be a list or a tuple!"
            qubits_used = qubits_used + list(subsetQs)
            assert(len(set(qubits_used)) == len(qubits_used)), \
                "The qubits in the tuples/lists of `structure must all be unique!"

        assert(set(qubits_used).issubset(set(pspec.qubit_labels))), \
            "The qubits to benchmark must all be in the ProcessorSpec `pspec`!"
        n = len(qubits_used)

    experiment_dict['spec']['circuits_per_length'] = circuits_per_length * multiplier * n

    experiment_dict['spec']['structure'] = structure
    experiment_dict['circuits'] = {}
    experiment_dict['settings'] = {}

    gates_available = list(pspec.models['target'].get_primitive_op_labels())
    gates_by_qubit = [[] for _ in range(0, n)]
    for i in range(0, len(gates_available)):
        for q in range(0, n):
            if gates_available[i].qubits == (q,):
                gates_by_qubit[q].append(gates_available[i])

    for lnum, l in enumerate(lengths):

        # generate menu of circuits for each qubit
        circuit_menu = [[] for _ in range(0, n)]
        for q in range(0, n):
            d = len(gates_by_qubit[q])

            if d**l < circuit_population_sz:
                print(('- Warning: circuit population specified too large for qubit {}'
                       ' -- there will be redundant circuits').format(q))

            for rep in range(0, circuit_population_sz):
                singleQcirc = []
                for j in range(0, l):
                    r = _np.random.randint(0, d)
                    singleQcirc.append(gates_by_qubit[q][r])
                circuit_menu[q].append(singleQcirc)

        if verbosity > 0:
            print('- Sampling {} random circuits per qubit at length {} ({} of {} lengths)'.format(
                circuits_per_length, l, lnum + 1, len(lengths)))

        # loop over qubits (should generalize this to regions)
        cnt = 0
        for q in range(0, n):
            print('  - Qubit {} = '.format(q))

            # need circuits_per_length number of settings for this qubit
            for j in range(circuits_per_length):
                if verbosity > 0: print('     Circuit {}: ('.format(j), end='')

                # draw a setting for the central qubit (q)
                #qr = _np.random.randint(0,circuit_population_sz)

                # instead of randomly drawing a setting for the central qubit,
                #  iteratively choose each of the circuits in the menu
                qr = j

                # generate "multiplier" number of random circuits on the other qubits with qr setting
                #  on the central qubit
                for m in range(0, multiplier):
                    circuit = _pygobjs.circuit.Circuit(num_lines=0, editable=True)
                    settings = {}

                    for q1 in range(0, n):

                        if q1 == q:
                            # the setting for the central qubit is fixed
                            r = qr
                        else:
                            # draw a setting
                            r = _np.random.randint(0, circuit_population_sz)

                        settings[(q1,)] = lnum * (circuit_population_sz + 1) + r + 1

                        singleQcircuit = _pygobjs.circuit.Circuit(num_lines=1, line_labels=[q1], editable=True)
                        for layer in range(0, l):
                            singleQcircuit.insert_layer(circuit_menu[q1][r][layer], layer)
                        singleQcircuit.done_editing()

                        circuit.tensor_circuit(singleQcircuit)

                    experiment_dict['settings'][l, cnt] = settings

                    # for each line, except the central qubit, replace sequence with an idle
                    #  independently according to idle_prob
                    if idle_prob > 0:
                        for q1 in range(0, n):
                            if q1 != q:
                                idle = bool(_np.random.binomial(1, idle_prob))
                                if idle:
                                    circuit.replace_with_idling_line(q1)
                                    # Update the setting on that qubit to the idling setting
                                    #  (denoted by the length index)
                                    experiment_dict['settings'][l, cnt][(q1,)] = lnum * (circuit_population_sz + 1)
                                    if verbosity > 0: print('(Idled {}) '.format(q1), end='')

                    circuit.done_editing()
                    experiment_dict['circuits'][l, cnt] = circuit
                    cnt += 1

                    if verbosity > 0: print('{}, '.format(m), end='')

                if verbosity > 0: print(')')

#   How are settings labeled?
#       For each circuit length, we label the random circuits of that length by consecutive integers, with the all
#       idle being the first number in the range for that circuit length.
#
#       So if circuit_population_sz is 10, and lengths are [10,20,30] then the the labels are:
#
#       0 (idle of length 10),  1, ...., 10  [all length 10 circuits]
#       11 (idle of length 20), 12, ..., 21  [all length 20 circuits]
#       22 (idle of length 30), 23, ..., 32  [all length 30 circuits]
#
        print('cnt: {}'.format(cnt))

    return experiment_dict


"""
def pairwise_indep_expts(q):
    vals = _np.zeros([q**2,q],dtype='int')
    for i in range(q):
        for a in range(q):
            for b in range(q):
                vals[a*q+b,i] = _np.int(_np.mod(a*i+b,q))

    length = _np.shape(vals)[0]
    return length, vals

def crosstalk_detection_experiment3(pspec, lengths, circuit_population_sz, include_idle=False, structure='1Q',
                         descriptor='A set of crosstalk detection experiments', verbosity=1):

# Pairwise independent experiment design. This is deprecated.

    experiment_dict = {}
    experiment_dict['spec'] = {}
    experiment_dict['spec']['lengths'] = lengths
    experiment_dict['spec']['circuit_population_sz'] = circuit_population_sz
    experiment_dict['spec']['include_idle'] = include_idle
    experiment_dict['spec']['descriptor'] = descriptor
    experiment_dict['spec']['createdby'] = 'extras.crosstalk.crosstalk_detection_experiment3'

    if isinstance(structure,str):
        assert(structure == '1Q'), "The only default `structure` option is the string '1Q'"
        structure = tuple([(q,) for q in pspec.qubit_labels])
        n = pspec.number_of_qubits
    else:
        assert(isinstance(structure,list) or isinstance(structure,tuple)), \
            "If not a string, `structure` must be a list or tuple."
        qubits_used = []
        for subsetQs in structure:
            assert(isinstance(subsetQs, list) or isinstance(subsetQs, tuple)), "SubsetQs must be a list or a tuple!"
            qubits_used = qubits_used + list(subsetQs)
            assert(len(set(qubits_used)) == len(qubits_used)), \
                "The qubits in the tuples/lists of `structure must all be unique!"

        assert(set(qubits_used).issubset(set(pspec.qubit_labels))), \
            "The qubits to benchmark must all be in the ProcessorSpec `pspec`!"
        n = len(qubits_used)

    assert(isprime(circuit_population_sz)), "circuit_population_sz must be prime"

    experiment_dict['spec']['circuits_per_length'] = circuit_population_sz**2

    experiment_dict['spec']['structure'] = structure
    experiment_dict['circuits'] = {}
    experiment_dict['settings'] = {}

    gates_available = list(pspec.models['target'].get_primitive_op_labels())
    gates_by_qubit = [[] for _ in range(0,n)]
    for i in range(0,len(gates_available)):
        for q in range(0,n):
            if gates_available[i].qubits == (q,):
                gates_by_qubit[q].append(gates_available[i])

    for lnum, l in enumerate(lengths):

        # generate menu of circuits for each qubit
        circuit_menu = [[] for _ in range(0,n)]
        for q in range(0,n):
            d = len(gates_by_qubit[q])

            if d**l < circuit_population_sz:
                print(('- Warning: circuit population specified too large for qubit {} -- '
                       'there will be redundant circuits').format(q))

            for rep in range(0,circuit_population_sz):
                singleQcirc = []
                for j in range(0,l):
                    r = _np.random.randint(0,d)
                    singleQcirc.append(gates_by_qubit[q][r])
                circuit_menu[q].append(singleQcirc)

        if verbosity > 0:
            print('- Sampling {} random circuits per qubit at length {} ({} of {} lengths)'.format(
                  circuit_population_sz**2,l,lnum+1,len(lengths)))
            print('( ')

        nexp, expt_idx = pairwise_indep_expts(circuit_population_sz)
        cnt = 0
        for exp in range(0,nexp):
            if verbosity > 0: print('{}, '.format(exp),end='')

            circuit = _pygobjs.circuit.Circuit(num_lines=0, editable=True)
            settings = {}
            for q in range(0,n):
                settings[(q,)] = lnum*(circuit_population_sz+1) + expt_idx[exp,q] + 1

                singleQcircuit = _pygobjs.circuit.Circuit(num_lines=1,line_labels=[q],editable=True)
                for layer in range(0,l):
                    singleQcircuit.insert_layer(circuit_menu[q][expt_idx[exp,q]][layer],layer)

                # if the idle should be included, replace the expt_idx==0 circuit with the idle
                if include_idle:
                    if expt_idx[exp,q]==0:
                        singleQcircuit.replace_with_idling_line(q)

                singleQcircuit.done_editing()

                circuit.tensor_circuit(singleQcircuit)

            experiment_dict['settings'][l,cnt] = settings

#            # for each line, except the central qubit, replace sequence with an idle independently
#            # according to idle_prob
#            if idle_prob>0:
#                for q in range(0,n):
#                    idle = bool(_np.random.binomial(1,idle_prob))
#                    if idle:
#                        circuit.replace_with_idling_line(q)
#                        # Update the setting on that qubit to the idling setting (denoted by the length index)
#                        experiment_dict['settings'][l,cnt][(q,)] = lnum*(circuit_population_sz+1)
#                        if verbosity > 0: print('(Idled {}) '.format(q),end='')

            circuit.done_editing()
            experiment_dict['circuits'][l,cnt] = circuit
            cnt += 1


        if verbosity > 0: print(')')


#   How are settings labeled?
#       For each circuit length, we label the random circuits of that length by consecutive integers, with the all
#       idle being the first number in the range for that circuit length.
#
#       So if circuit_population_sz is 10, and lengths are [10,20,30] then the the labels are:
#
#       0 (idle of length 10),  1, ...., 10  [all length 10 circuits]
#       11 (idle of length 20), 12, ..., 21  [all length 20 circuits]
#       22 (idle of length 30), 23, ..., 32  [all length 30 circuits]
#
        print('cnt: {}'.format(cnt))

    return experiment_dict
"""
