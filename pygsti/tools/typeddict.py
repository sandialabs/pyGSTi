""" The TypedDict class """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np


def _columndict_to_dataframe(columns, seriestypes):
    import pandas as _pandas
    columns_as_series = {}
    for colname, lst in columns.items():
        seriestype = seriestypes[colname]
        if seriestype == 'float':
            s = _np.array(lst, dtype='d')
        elif seriestype == 'int':
            s = _np.array(lst, dtype=int)  # or pd.Series w/dtype?
        elif seriestype == 'category':
            s = _pandas.Categorical(lst)
        elif seriestype == 'object':
            s = _pandas.Series(lst, dtype=object)
        else:
            s = lst  # will infer an object array?

        columns_as_series[colname] = s
    return _pandas.DataFrame(columns_as_series)


class TypedDict(dict):
    def __init__(self, types=None, items=()):
        super().__init__(items)
        self._types = types if (types is not None) else {}

    def __reduce__(self):
        return (TypedDict, (self._types, list(self.items())), None)

    def as_dataframe(self):
        columns = {}; seriestypes = {}
        self._add_to_columns(columns, seriestypes, {})
        return _columndict_to_dataframe(columns, seriestypes)

    def _add_to_columns(self, columns, seriestypes, row_prefix):
        ncols = len(next(iter(columns.values()))) if len(columns) > 0 else 0
        for nm, v in self.items():
            typ = self._types.get(nm, None)
            if nm not in columns:  # then add a column
                columns[nm] = [None] * ncols
                seriestypes[nm] = typ
            elif seriestypes[nm] != typ:
                seriestypes[nm] = None  # conflicting types, so set to None

            assert(nm not in row_prefix), \
                ("Column %s is assigned at multiple dict-levels (latter levels will "
                 "overwrite the values of earlier levels)! keys-so-far=%s") % (nm, tuple(row_prefix.keys()))

        #Add row
        row = row_prefix.copy()
        row.update(self)
        for rk, rv in row.items():
            columns[rk].append(rv)
        absent_colvals = set(columns.keys()) - set(row.keys())
        for rk in absent_colvals:  # Ensure all columns stay the same length
            columns[rk].append(None)
            seriestypes[rk] = None  # can't store Nones in special types
