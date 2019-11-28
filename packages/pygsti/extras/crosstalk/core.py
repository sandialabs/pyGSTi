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


def do_basic_crosstalk_detection(ds, number_of_regions, settings, confidence=0.95, verbosity=1, name=None,
                                 assume_independent_settings=True):
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

    Returns
    -------
    results : CrosstalkResults object
        The results of the crosstalk detection analysis. This contains: output skeleton graph and DAG from
        PC Algorithm indicating regions with detected crosstalk, all of the input information.

    """
    # -------------------------- #
    # Format and check the input #
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

        # num columns = number of settings + number of regions (b/c we assume one outcome per region)
        num_columns = num_settings + number_of_regions

        num_data = len(ds.keys())

        data = []
        collect_settings = {key: [] for key in range(num_settings)}
        for row in range(num_data):
            opstr = ds.keys()[row]

            templine_set = [0] * num_settings
            settings_row = ds.auxInfo[opstr]['settings']

            for key in settings_row:
                if len(key) == 1:  # single region/qubit gate
                    templine_set[key[0]] = settings_row[key]
                    collect_settings[key[0]].append(settings_row[key])
                else:  # two-region/two-qubit gate
                    print("Two qubit gate, not sure what to do!!")  # TODO
                    return

            outcomes_row = ds[opstr]
            for outcome in outcomes_row:
                templine_out = [0] * number_of_regions

                for r in range(number_of_regions):
                    templine_out[r] = int(outcome[0][r])
                num_rep = int(outcome[2])

                templine_out.append(templine_set)
                flattened_line = list(flatten(templine_out))

                for r in range(num_rep):
                    data.append(flattened_line)

        data = _np.asarray(data)

    # --------------------------------------------------------- #
    # Prepare a results object, and store the input information #
    # --------------------------------------------------------- #

    # Initialize an empty results object.
    results = _obj.CrosstalkResults()

    # Records input information into the results object.
    results.name = name
    results.data = data
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
        if source < number_of_regions and dest < number_of_regions:
            cmatrix[source, dest] = 1
            is_edge_ct[idx] = 1
            print("Crosstalk detected. Regions " + str(source) + " and " + str(dest))

        if source < number_of_regions and dest >= number_of_regions:
            if dest not in _setting_range(source):
                for region in range(number_of_regions):
                    if dest in _setting_range(region):
                        break
                cmatrix[source, region] = 1
                is_edge_ct[idx] = 1
                print("Crosstalk detected. Regions " + str(source) + " and " + str(region))

        if source >= number_of_regions and dest < number_of_regions:
            if source not in _setting_range(dest):
                for region in range(number_of_regions):
                    if source in _setting_range(region):
                        break
                cmatrix[region, dest] = 1
                is_edge_ct[idx] = 1
                print("Crosstalk detected. Regions " + str(region) + " and " + str(dest))

        # For each edge in causal graph that represents crosstalk, calculate the TVD between distributions of dependent
        # variable when other variable is varied

        if is_edge_ct[idx] == 1:

            source_levels, level_cnts = _np.unique(data[:, source], return_counts=True)
            num_levels = len(source_levels)

            if any(level_cnts < 10):
                print(" ***   Warning: n<10 data points for some levels. TVD calculations may have large error bars.")

            tvds = _np.zeros((num_levels, num_levels))
            for i in range(num_levels):
                for j in range(i):

                    marg1 = data[data[:, source] == source_levels[i], dest]
                    marg2 = data[data[:, source] == source_levels[j], dest]
                    n1, n2 = len(marg1), len(marg2)

                    marg1_levels, marg1_level_cnts = _np.unique(marg1, return_counts=True)
                    marg2_levels, marg2_level_cnts = _np.unique(marg2, return_counts=True)

                    tvd_sum = 0.0
                    for lidx, level in enumerate(marg1_levels):
                        temp = _np.where(marg2_levels == level)
                        if len(temp[0]) == 0:
                            tvd_sum += marg1_level_cnts[lidx] / n1
                        else:
                            tvd_sum += _np.fabs(marg1_level_cnts[lidx] / n1 - marg2_level_cnts[temp[0][0]] / n2)

                    tvds[i, j] = tvds[j, i] = tvd_sum / 2.0

            edge_tvds[idx] = tvds

    results.cmatrix = cmatrix
    results.is_edge_ct = is_edge_ct
    results.crosstalk_detected = _np.sum(is_edge_ct) > 0
    results.edge_weights = edge_weights
    results.edge_tvds = edge_tvds

    return results
