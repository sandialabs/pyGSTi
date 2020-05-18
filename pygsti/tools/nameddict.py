"""
The NamedDict class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np


class NamedDict(dict):
    """
    A dictionary that also holds category names and types.

    This `dict`-derived class holds a catgory name applicable to
    its keys, and key and value type names indicating the types
    of its keys and values.

    The main purpose of this class is to utilize its :method:`as_dataframe` method.

    Parameters
    ----------
    name : str, optional
        A category name for the keys of this dict.  For example, if the
        dict contained the keys `"dog"` and `"cat"`, this might be `"animals"`.
        This becomse a column header if this dict is converted to a data frame.

    keytype : {"float", "int", "categor", None}, optional
        The key-type, in correspondence with different pandas series types.

    valtype : {"float", "int", "categor", None}, optional
        The value-type, in correspondence with different pandas series types.

    items : list or dict, optional
        Initial items, used in serialization.
    """
    def __init__(self, name=None, keytype=None, valtype=None, items=()):
        super().__init__(items)
        self.name = name
        self.keytype = keytype
        self.valtype = valtype

    def __reduce__(self):
        return (NamedDict, (self.name, self.keytype, self.valtype, list(self.items())), None)

    def as_dataframe(self):
        """
        Render this dict as a pandas data frame.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as _pandas

        columns = {'value': []}
        seriestypes = {'value': "unknown"}
        self._add_to_columns(columns, seriestypes, {})

        columns_as_series = {}
        for colname, lst in columns.items():
            seriestype = seriestypes[colname]
            if seriestype == 'float':
                s = _np.array(lst, dtype='d')
            elif seriestype == 'int':
                s = _np.array(lst, dtype=int)  # or pd.Series w/dtype?
            elif seriestype == 'category':
                s = _pandas.Categorical(lst)
            else:
                s = lst  # will infer an object array?

            columns_as_series[colname] = s

        df = _pandas.DataFrame(columns_as_series)
        return df

    def _add_to_columns(self, columns, seriestypes, row_prefix):
        nm = self.name
        if nm not in columns:
            #add column; assume 'value' is always a column
            columns[nm] = [None] * len(columns['value'])
            seriestypes[nm] = self.keytype
        elif seriestypes[nm] != self.keytype:
            seriestypes[nm] = None  # conflicting types, so set to None

        assert(nm not in row_prefix), \
            ("Column %s is assigned at multiple dict-levels (latter levels will "
             "overwrite the values of earlier levels)! keys-so-far=%s") % (nm, tuple(row_prefix.keys()))

        row = row_prefix.copy()
        for k, v in self.items():
            row[nm] = k
            if isinstance(v, NamedDict):
                v._add_to_columns(columns, seriestypes, row)
            elif hasattr(v, 'as_nameddict'):  # e.g., for other ProtocolResults
                v.as_nameddict()._add_to_columns(columns, seriestypes, row)
            else:
                #Add row
                complete_row = row.copy()
                complete_row['value'] = v

                if seriestypes['value'] == "unknown":
                    seriestypes['value'] = self.valtype
                elif seriestypes['value'] != self.valtype:
                    seriestypes['value'] = None  # conflicting type, so set to None

                for rk, rv in complete_row.items():
                    columns[rk].append(rv)

                absent_colvals = set(columns.keys()) - set(complete_row.keys())
                for rk in absent_colvals:  # Ensure all columns stay the same length
                    columns[rk].append(None)
                    seriestypes[rk] = None  # can't store Nones in special types
