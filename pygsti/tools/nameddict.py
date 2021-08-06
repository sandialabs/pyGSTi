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

from pygsti.tools import typeddict as _typeddict


class NamedDict(dict):
    """
    A dictionary that also holds category names and types.

    This `dict`-derived class holds a catgory name applicable to
    its keys, and key and value type names indicating the types
    of its keys and values.

    The main purpose of this class is to utilize its :method:`to_dataframe` method.

    Parameters
    ----------
    keyname : str, optional
        A category name for the keys of this dict.  For example, if the
        dict contained the keys `"dog"` and `"cat"`, this might be `"animals"`.
        This becomes a column header if this dict is converted to a data frame.

    keytype : {"float", "int", "category", None}, optional
        The key-type, in correspondence with different pandas series types.

    valname : str, optional
        A category name for the keys of this dict. This becomse a column header
        if this dict is converted to a data frame.

    valtype : {"float", "int", "category", None}, optional
        The value-type, in correspondence with different pandas series types.

    items : list or dict, optional
        Initial items, used in serialization.
    """

    @classmethod
    def create_nested(cls, key_val_type_list, inner):
        """
        Creates a nested NamedDict.

        Parameters
        ----------
        key_val_type_list : list
            A list of (key, value, type) tuples, one per nesting layer.

        inner : various
            The value that will be set to the inner-most nested
            dictionary's value, supplying any additional layers of
            nesting (if `inner` is a `NamedDict`) or the value
            contained in all of the nested layers.
        """
        head = tail = {}; val = None
        for next_key, next_val, next_type in key_val_type_list:
            tail[val] = cls(next_key, next_type); tail = tail[val]
            val = next_val

        tail[val] = inner
        return head[None]

    def __init__(self, keyname=None, keytype=None, valname=None, valtype=None, items=()):
        super().__init__(items)
        self.keyname = keyname
        self.valname = valname
        self.keytype = keytype
        self.valtype = valtype

    def __reduce__(self):
        return (NamedDict, (self.keyname, self.keytype, self.valname, self.valtype, list(self.items())), None)

    def to_dataframe(self):
        """
        Render this dict as a pandas data frame.

        Returns
        -------
        pandas.DataFrame
        """
        columns = {}; seriestypes = {}
        self._add_to_columns(columns, seriestypes, {})
        return _typeddict._columndict_to_dataframe(columns, seriestypes)

    def _add_to_columns(self, columns, seriestypes, row_prefix):
        #Add key column if needed
        nm = self.keyname
        ncols = len(next(iter(columns.values()))) if len(columns) > 0 else 0
        if nm not in columns:  # then add a column
            columns[nm] = [None] * ncols
            seriestypes[nm] = self.keytype
        elif seriestypes[nm] != self.keytype:
            seriestypes[nm] = None  # conflicting types, so set to None

        assert(nm not in row_prefix), \
            ("Column %s is assigned at multiple dict-levels (latter levels will "
             "overwrite the values of earlier levels)! keys-so-far=%s") % (nm, tuple(row_prefix.keys()))

        #Add value column if needed
        valname = self.valname if (self.valname is not None) else 'Value'
        add_value_col = not all([(isinstance(v, (NamedDict, _typeddict.TypedDict)) or hasattr(v, 'to_nameddict'))
                                 for v in self.values()])
        if add_value_col:
            if valname not in columns:  # then add a column
                columns[valname] = [None] * ncols
                seriestypes[valname] = self.valtype if (ncols == 0) else None  # can't store Nones in special types
            elif seriestypes[valname] != self.valtype:
                seriestypes[valname] = None  # conflicting types, so set to None

            assert(valname not in row_prefix), \
                ("Column %s is assigned at multiple dict-levels (latter levels will "
                 "overwrite the values of earlier levels)! keys-so-far=%s") % (valname, tuple(row_prefix.keys()))

        #Add rows
        row = row_prefix.copy()
        for k, v in self.items():
            row[nm] = k
            if isinstance(v, (NamedDict, _typeddict.TypedDict)):
                v._add_to_columns(columns, seriestypes, row)
            elif hasattr(v, 'to_nameddict'):  # e.g., for other ProtocolResults
                v.to_nameddict()._add_to_columns(columns, seriestypes, row)
            else:
                #Add row
                complete_row = row.copy()
                complete_row[valname] = v

                for rk, rv in complete_row.items():
                    columns[rk].append(rv)

                absent_colvals = set(columns.keys()) - set(complete_row.keys())
                for rk in absent_colvals:  # Ensure all columns stay the same length
                    columns[rk].append(None)
                    seriestypes[rk] = None  # can't store Nones in special types
