"""
Routines for converting python objects to python.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
from ..objects.reportableqty import ReportableQty as _ReportableQty

'''
table() and cell() functions are used by table.py in table creation
everything else is used in creating formatters in formatters.py
'''


def table(custom_headings, col_headings_formatted, rows, spec):
    """
    Create a "Python table" - really a pandas DataFrame

    Parameters
    ----------
    custom_headings : None, dict
        optional dictionary of custom table headings

    col_headings_formatted : list
        formatted column headings

    rows : list of lists of cell-strings
        Data in the table, pre-formatted

    spec : dict
        options for the formatter

    Returns
    -------
    dict : contains key 'python', which corresponds to a
        pandas.DataFrame object representing the table
    """
    try:
        import pandas as _pd
    except ImportError:
        raise ValueError(("You must have the optional 'pandas' package "
                          "installed to render tables in the 'python' format"))

    def getval(lbl):
        return lbl.value if isinstance(lbl, _ReportableQty) else lbl

    if custom_headings is not None \
            and "python" in custom_headings:
        colLabels = custom_headings['python']
    else:
        colLabels = [getval(x) for x in col_headings_formatted]
    nCols = len(colLabels)

    if nCols == 0: return {'python': _pd.DataFrame()}

    #Remove duplicate in colLabels (otherwise these cols get merged weirdly below)
    for i in range(len(colLabels)):
        if colLabels[i] in colLabels[0:i]:
            k = 1
            while colLabels[i] + str(k) in colLabels[0:i]: k += 1
            colLabels[i] = colLabels[i] + str(k)

    #Add addition error-bar columns for any columns that have error bar info
    cols_containing_ebs = set()
    for formatted_rowData in rows:
        assert(len(formatted_rowData) == nCols)
        for i, formatted_cellData in enumerate(formatted_rowData):
            if isinstance(formatted_cellData, _ReportableQty) and \
               formatted_cellData.has_errorbar:
                cols_containing_ebs.add(i)

    n = 0  # number of cols inserted
    for iCol in sorted(cols_containing_ebs):
        origLbl = colLabels[iCol + n]
        colLabels.insert(iCol + n + 1, origLbl + " Error Bar")
        n += 1

    rowLabels = []
    rowIndexName = getval(colLabels[0])
    if len(rowIndexName.strip()) == 0:
        rowIndexName = None

    dict_of_columns = _collections.OrderedDict()
    for colLabel in colLabels[1:]:
        dict_of_columns[colLabel] = []

    for formatted_rowData in rows:
        rowLabels.append(getval(formatted_rowData[0])); n = 0

        for i, formatted_cellData in enumerate(formatted_rowData[1:], start=1):
            if i in cols_containing_ebs:
                if isinstance(formatted_cellData, _ReportableQty):
                    val, eb = formatted_cellData.value_and_errorbar
                else:
                    val, eb = formatted_cellData, None
                dict_of_columns[colLabels[i + n]].append(val)
                dict_of_columns[colLabels[i + n + 1]].append(eb)
                n += 1
            else:
                dict_of_columns[colLabels[i + n]].append(getval(formatted_cellData))

    indx = _pd.Index(rowLabels, name=rowIndexName)
    #print("DB PANDAS: headings=",colLabels)  #DEBUG
    #print("col_dict(cnt) = ", [(k,len(v)) for k,v in dict_of_columns.items()]) #DEBUG
    df = _pd.DataFrame(dict_of_columns,
                       columns=dict_of_columns.keys(),
                       index=indx)

    return {'python': df}


def cell(data, label, spec):
    """
    Format the cell of a python table

    Parameters
    ----------
    data : string
        string representation of cell content

    label : string
        optional cell label, used for tooltips

    spec : dict
        options for the formatters

    Returns
    -------
    string
    """
    return data


def list(l, specs):
    """
    Stub for conversion that isn't needed in python case.

    (Convert a python list to python.)

    Parameters
    ----------
    l : list
        list to convert into latex. sub-items pre formatted

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    list
    """
    return l


def vector(v, specs):
    """
    Stub for conversion that isn't needed in python case.

    (Convert a 1D numpy array to python.)

    Parameters
    ----------
    v : numpy array
        1D array to convert.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    numpy array
    """
    return v


def matrix(m, specs):
    """
    Stub for conversion that isn't needed in python case.

    Convert a 2D numpy array to python.

    Parameters
    ----------
    m : numpy array
        2D array to convert.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    numpy array
    """
    return m


def value(el, specs):
    """
    Stub for conversion that isn't needed in python case.

    (this function would be for converting python to python).

    Parameters
    ----------
    el : float or complex
        Value to convert into latex.

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    float or complex
    """
    return el


def escaped(txt, specs):
    """
    Stub for conversion that isn't needed in python case.

    (Escape txt so it is python safe.)

    Parameters
    ----------
    txt : string
        value to escape

    specs : dictionary
        Dictionary of user-specified and default parameters to formatting

    Returns
    -------
    string
    """
    return txt
