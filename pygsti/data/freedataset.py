"""
Defines the FreeformDataSet class and supporting classes and functions
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import copy as _copy
from collections import OrderedDict as _OrderedDict

from pygsti.circuits import circuit as _cir
from pygsti.tools import NamedDict as _NamedDict


class FreeformDataSet(object):
    """
    An association between Circuits and arbitrary data.

    Parameters
    ----------
    circuits : list of (tuples or Circuits), optional
        Each element is a tuple of operation labels or a Circuit object.  Indices for these strings
        are assumed to ascend from 0.  These indices must correspond to the time series of spam-label
        indices (above).   Only specify this argument OR circuit_indices, not both.

    circuit_indices : ordered dictionary, optional
        An OrderedDict with keys equal to circuits (tuples of operation labels) and values equal to
        integer indices associating a row/element of counts with the circuit.  Only
        specify this argument OR circuits, not both.
    """

    def __init__(self, circuits=None, circuit_indices=None):
        # uuid for efficient hashing (set when done adding data or loading from file)
        self.uuid = None

        # self.cirIndex  :  Ordered dictionary where keys = Circuit objects,
        #   values = slices into oli, time, & rep arrays (static case) or
        #            integer list indices (non-static case)
        if circuit_indices is not None:
            self.cirIndex = _OrderedDict([(opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr), i)
                                          for opstr, i in circuit_indices.items()])
            #convert keys to Circuits if necessary
        else:  # if not static:
            if circuits is not None:
                dictData = [(opstr if isinstance(opstr, _cir.Circuit) else _cir.Circuit(opstr), i)
                            for (i, opstr) in enumerate(circuits)]  # convert to Circuits if necessary
                self.cirIndex = _OrderedDict(dictData)
            else:
                self.cirIndex = _OrderedDict()

        #FUTURE - make this more like DataSet and hold some/all columns more efficiently?
        self._info = {i: {} for i in self.cirIndex.values()}

        # comment
        #self.comment = comment

    def to_dataframe(self, pivot_valuename=None, pivot_value="Value", drop_columns=False):
        """
        Create a Pandas dataframe with the data from this free-form dataset.

        Parameters
        ----------
        pivot_valuename : str, optional
            If not None, the resulting dataframe is pivoted using `pivot_valuename`
            as the column whose values name the pivoted table's column names.
            If None and `pivot_value` is not None,`"ValueName"` is used.

        pivot_value : str, optional
            If not None, the resulting dataframe is pivoted such that values of
            the `pivot_value` column are rearranged into new columns whose names
            are given by the values of the `pivot_valuename` column. If None and
            `pivot_valuename` is not None,`"Value"` is used.

        drop_columns : bool or list, optional
            A list of column names to drop (prior to performing any pivot).  If
            `True` appears in this list or is given directly, then all
            constant-valued columns are dropped as well.  No columns are dropped
            when `drop_columns == False`.

        Returns
        -------
        pandas.DataFrame
        """
        from pygsti.tools.dataframetools import _process_dataframe
        cdict = _NamedDict('Circuit', None)
        for cir, i in self.cirIndex.items():
            cdict[cir.str] = _NamedDict('ValueName', 'category', items=self._info[i])
        df = cdict.to_dataframe()
        return _process_dataframe(df, pivot_valuename, pivot_value, drop_columns)

    def __iter__(self):
        return self.cirIndex.__iter__()  # iterator over circuits

    def __len__(self):
        return len(self.cirIndex)

    def __contains__(self, circuit):
        """
        Test whether data set contains a given circuit.

        Parameters
        ----------
        circuit : tuple or Circuit
            A tuple of operation labels or a Circuit instance
            which specifies the the circuit to check for.

        Returns
        -------
        bool
            whether circuit was found.
        """
        if not isinstance(circuit, _cir.Circuit):
            circuit = _cir.Circuit(circuit)
        return circuit in self.cirIndex

    def __hash__(self):
        if self.uuid is not None:
            return hash(self.uuid)
        else:
            raise TypeError('Use digest hash')

    def __getitem__(self, circuit):
        return self._info[self.cirIndex[circuit]]

    def __setitem__(self, circuit, info_dict):
        #collision action?
        if circuit not in self.cirIndex:
            self.cirIndex[circuit] = len(self.cirIndex)  # add a new circuit index
        self._info[self.cirIndex[circuit]] = info_dict

    def __delitem__(self, circuit):
        if not isinstance(circuit, _cir.Circuit):
            circuit = _cir.Circuit(circuit)
        del self._info[self.cirIndex[circuit]]
        del self.cirIndex[circuit]

    def keys(self):
        """
        Returns the circuits used as keys of this DataSet.

        Returns
        -------
        list
            A list of Circuit objects which index the data
            counts within this data set.
        """
        yield from self.cirIndex.keys()

    def items(self):
        """
        Iterator over `(circuit, info_dict)` pairs.

        Returns
        -------
        Iterator
        """
        return self.cirIndex.keys()

    def values(self):
        """
        Iterator over info-dicts for each circuit.

        Returns
        -------
        Iterator
        """
        return self._info.values

    def copy(self):
        """
        Make a copy of this FreeformDataSet.

        Returns
        -------
        DataSet
        """
        copyOfMe = FreeformDataSet(circuit_indices=self.cirIndex)
        copyOfMe._info = _copy.deepcopy(self._info)
        return copyOfMe
