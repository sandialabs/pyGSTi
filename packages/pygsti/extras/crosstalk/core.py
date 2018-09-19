from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Core integrated routines for detecting and characterizing crosstalk"""

import numpy as _np
from . import objects as _obj
import pcalg
from gsq.ci_tests import ci_test_dis

def do_basic_crosstalk_detection(ds, number_of_qubits, settings, confidence=0.95, verbosity=1, name=None):
    """
    Implements crosstalk detection on multiqubit data (fine-grained data with entries for each experiment).
    
    Parameters
    ----------
    ds : pyGSTi DataSet or numpy array
        The multiqubit data to analyze. If this is a DataSet it should contain fine-grained (or time series)
        data (rather than, e.g., a total counts per-outcome per-GateString). If this is a
        numpy array, it must again contain time series data and it may be 2-dimensional with each entry being 
        a sequence of settings and measurment outcomes for each qubit. The first n entries are the outcomes 
        and the following entries are settings.

    number_of_qubits: int, number of qubits in experiment
    
    settings: list of length number_of_qubits, indicating the number of settings for each qubit  
    
    confidence : float, optional
    
    verbosity : int, optional
    
    name : str, optional
    
    Returns
    -------
    results : CrosstalkResults object
        The results of the crosstalk detection analysis. This contains: output skeleton graph PC Algorithm indicating
        qubits with detected crosstalk, all of the input information.

    """    
    # -------------------------- #
    # Format and check the input #
    # -------------------------- #
    
    if True :
    # TODO type(ds) != _objs.dataset.DataSet:
    
        data_shape = _np.shape(ds)
        settings_shape = _np.shape(settings)
    
        # Check that the input data is a 2D array
        assert(len(data_shape) == 2), \
        "Input data format is incorrect!If the input is a numpy array it must be 2-dimensional."
        
        # Check that settings is a list of length number_of_qubits
        assert( (len(settings_shape) == 1) and (settings_shape[0] == number_of_qubits) )
        "settings should be a list of the same length as number_of_qubits."
        
        # The number of columns in the data must be consistent with the number of settings
        assert( data_shape[1] == (sum(settings) + number_of_qubits) )
        "Mismatch between the number of settings specified for each qubit and the number of columns in data"
        
        data = ds
        num_data = data_shape[0]
        num_columns = data_shape[1]
        
    # This converts a DataSet to an array, as the code below uses arrays 
    if False :
    #TODO type(ds) == _objs.dataset.DataSet:
        print("Error: crosstalk analysis currently does not support input specificed as DataSet")
        return
        
    # --------------------------------------------------------- #
    # Prepare a results object, and store the input information #
    # --------------------------------------------------------- #
    
    # Initialize an empty results object.
    results = _obj.CrosstalkResults()
    
    # Records input information into the results object.
    results.name = name
    results.data = data
    results.number_of_qubits = number_of_qubits
    results.settings = settings
    results.number_of_datapoints = num_data
    results.number_of_columns = num_columns
    results.confidence = confidence

    # ------------------------------------------------- #
    #     Calculate the causal graph skeleton           #
    # ------------------------------------------------- #
       
    print("Calculating causal graph skeleton ...")
    (skel,sep_set) = pcalg.estimate_skeleton(ci_test_dis, data, 1-confidence)
    
    print("Calculating directed causal graph ...")
    g = pcalg.estimate_cpdag(skel_graph=skel, sep_set=sep_set)
    # TODO: explicitly exclude edges between settings
    
    # Store skeleton and graph in results object
    results.skel = skel
    results.sep_set = sep_set
    results.graph = g
    
    # Calculate the column index for the first setting for each qubit
    setting_indices = {x: number_of_qubits+sum(settings[:x]) for x in range(number_of_qubits) };
    results.setting_indices = setting_indices

    node_labels = {}
    cnt=0
    for col in range(num_columns) :
        if col < number_of_qubits :
            node_labels[cnt] = r'$%d^O$' % col
            cnt += 1
#            node_labels.append("$%d^O$" % col)
        else :
            for qubit in range(number_of_qubits):
                if col in range(setting_indices[qubit],
                                 (setting_indices[(qubit + 1)] if qubit < (number_of_qubits - 1) else num_columns)):
                    break
            node_labels[cnt] = r'$%d^S_{%d}$' % (qubit, (col-setting_indices[qubit]+1))
            cnt += 1
            #node_labels.append("%d^S_{%d}$" % (qubit, (col-setting_indices[qubit]+1)))

    results.node_labels = node_labels

    # Generate crosstalk detected matrix and assign weight to each edge according to TVD variation in distribution of
    # destination variable when source variable is varied.
    print("Examining edges for crosstalk ...")
    
    cmatrix = _np.zeros((number_of_qubits, number_of_qubits))
    edge_weights = _np.zeros(len(g.edges()))
    is_edge_ct = _np.zeros(len(g.edges()))
    edge_tvds = {}

    for idx, edge in enumerate(g.edges()) :
        source = edge[0]
        dest = edge[1]

        if verbosity>1 :
            print("** Edge: ", edge, " **")

        # For each edge, decide if it represents crosstalk
        #   Crosstalk is:
        #       (1) an edge between outcomes on different qubits
        #       (2) an edge between a qubit's outcome and a setting of another qubit
        if source < number_of_qubits and dest < number_of_qubits :
            cmatrix[source, dest] = 1
            is_edge_ct[idx] = 1
            print("Crosstalk detected. Qubits " + str(source) + " and " + str(dest))
        
        if source < number_of_qubits and dest >= number_of_qubits :
            if dest not in range(setting_indices[source], (setting_indices[(source+1)] if source<(number_of_qubits-1) else num_columns)) :
                for qubit in range(number_of_qubits) :
                    if dest in range(setting_indices[qubit], (setting_indices[(qubit+1)] if qubit<(number_of_qubits-1) else num_columns)) :
                        break    
                cmatrix[source, qubit] = 1
                is_edge_ct[idx] = 1
                print("Crosstalk detected. Qubits " + str(source) + " and " + str(qubit))
                
        if source >= number_of_qubits and dest < number_of_qubits :
            if source not in range(setting_indices[dest], (setting_indices[(dest+1)] if dest<(number_of_qubits-1) else num_columns)) :
                for qubit in range(number_of_qubits) :
                    if source in range(setting_indices[qubit], (setting_indices[(qubit+1)] if qubit<(number_of_qubits-1) else num_columns)) :
                        break
                cmatrix[qubit, dest] = 1
                is_edge_ct[idx] = 1
                print("Crosstalk detected. Qubits " + str(qubit) + " and " + str(dest))

        # For each edge in causal graph that represents crosstalk, calculate the TVD between distributions of dependent
        # variable when other variable is varied

        if is_edge_ct[idx] == 1 :

            source_levels, level_cnts = _np.unique(data[:, source], return_counts=True)
            num_levels = len(source_levels)

            if any(level_cnts<10) :
                print( " ***   Warning: n<10 data points for some levels. TVD calculations may have large error bars.")

            tvds = _np.zeros((num_levels, num_levels))
            for i in range(num_levels) :
                for j in range(i) :

                    marg1 = data[data[:, source]==source_levels[i], dest]
                    marg2 = data[data[:, source] == source_levels[j], dest]
                    n1, n2 = len(marg1), len(marg2)

                    marg1_levels, marg1_level_cnts = _np.unique(marg1, return_counts=True)
                    marg2_levels, marg2_level_cnts = _np.unique(marg2, return_counts=True)

                    tvd_sum = 0.0
                    for lidx, level in enumerate(marg1_levels) :
                        temp = _np.where(marg2_levels==level)
                        if len(temp[0]) == 0 :
                            tvd_sum += marg1_level_cnts[lidx]/n1
                        else :
                            tvd_sum += _np.fabs(marg1_level_cnts[lidx]/n1 - marg2_level_cnts[temp[0][0]]/n2)

                    tvds[i,j] = tvds[j,i] = tvd_sum/2.0

            edge_tvds[idx] = tvds


    results.cmatrix = cmatrix
    results.is_edge_ct = is_edge_ct
    results.crosstalk_detected = _np.sum(is_edge_ct)>0
    results.edge_weights = edge_weights
    results.edge_tvds = edge_tvds

    return results