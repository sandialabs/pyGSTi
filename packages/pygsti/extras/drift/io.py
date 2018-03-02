from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""Functions for Fourier analysis of equally spaced time-series data"""

import numpy as _np
from ... import objects as _objs

def load_bitstring_probabilities(filename, sequences_to_indices=None):
    """
    TODO: docstring
    """
    pdict = {}
    with open(filename,'r') as f:
        for line in f:
            # Skips comment rows
            if not line.startswith("#"):
                row = line.split()
                # Skips rows containing nothing
                if len(row) != 0:
                    gstr = row[0]
                    data = row[1].split(',')
                    pdict[_objs.GateString(None,gstr)] = _np.array([float(p) for p in data])
                    
    if sequences_to_indices is None:
        return pdict
    
    if sequences_to_indices is not None:
        sequences = list(sequences_to_indices.keys())
        parray = _np.zeros((len(sequences),1,2,len(pdict[list(pdict.keys())[0]])),float)
        
        for i in range(0,len(sequences)):
            parray[sequences_to_indices[sequences[i]],0,1,:] = pdict[sequences[i]]
            parray[sequences_to_indices[sequences[i]],0,0,:] = 1 - pdict[sequences[i]]
        return parray
    
def load_bitstring_data(filename, sequences_to_indices=None):
    """
    TODO: docstring
    """
    datadict = {}
    with open(filename,'r') as f:
        for line in f:
            # Skips comment rows
            if not line.startswith("#"):
                row = line.split()
                # Skips rows containing nothing
                if len(row) != 0:
                    gstr = row[0]
                    data = row[1]
                    datadict[_objs.GateString(None,gstr)] = _np.array([float(p) for p in data])
    
    sequences = list(datadict.keys())            
    sequences_to_indices = {}
    for i in range(0,len(sequences)):
        sequences_to_indices[sequences[i]] = i
    
    dataarray = _np.zeros((len(sequences),1,2,len(datadict[list(datadict.keys())[0]])),float)
        
    for i in range(0,len(sequences)):
        dataarray[sequences_to_indices[sequences[i]],0,1,:] = datadict[sequences[i]]
        dataarray[sequences_to_indices[sequences[i]],0,0,:] = 1 - datadict[sequences[i]]
        
    indices_to_sequences = {}
    for i in range(0,len(sequences)):
        indices_to_sequences[i] = sequences[i]
        
    return dataarray, indices_to_sequences