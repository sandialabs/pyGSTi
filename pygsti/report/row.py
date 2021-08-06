"""
Defines the Row class
"""

#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.report.reportableqty import ReportableQty as _ReportableQty
from pygsti.report.cell import Cell


class Row(object):
    """
    Representation of a table row

    Parameters
    ----------
    row_data : list
        Raw data for the table

    formatters : list[string], optional
        Formatting options for each cell

    labels : list[string], optional
        Labeling options for each cell

    non_markovian_ebs : bool
        Whether non-Markovian error bars should be used
    """

    def __init__(self, row_data=None, formatters=None, labels=None, non_markovian_ebs=False):
        '''
        Create a row object

        Parameters
        ----------
        row_data : list
            raw data for the table
        formatters : optional, list[string]
            formatting options for each cell
        labels : optional list[string]
            labeling options for each cell
        non_markovian_ebs : bool
            boolean indicating if non markovian error bars should be used
        '''
        if row_data is None:
            row_data = []
        else:
            row_data = [_ReportableQty.from_val(item, non_markovian_ebs) for item in row_data]
        if formatters is None:
            formatters = []
        if labels is None:
            labels = row_data

        lendiff = max(abs(len(formatters) - len(row_data)), 0)
        formatters = list(formatters) + [None] * lendiff

        lendiff = max(abs(len(labels) - len(row_data)), 0)
        labels = list(labels) + [None] * lendiff

        self.cells = [Cell(item, formatter, label)
                      for item, formatter, label in
                      zip(row_data, formatters, labels)]

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        return state_dict

    def __setstate__(self, d):
        self.__dict__.update(d)

    def add(self, data, formatter=None, label=None):
        """
        Adds a cell with the given `data`, `formatter` and `label`

        Parameters
        ----------
        data : ReportableQty
            Cell data to be reported

        formatter : string, optional
            Name of the cell formatter to be used (ie 'Effect')

        label : string, optional
            Label of the cell

        Returns
        -------
        None
        """
        self.cells.append(Cell(data, formatter, label))

    def render(self, fmt, specs):
        """
        Render a row of cells

        Parameters
        ----------
        fmt : string
            Format to be rendered in

        specs : dict
            Options for formatting

        Returns
        -------
        list
        """
        formattedItems = []
        for cell in self.cells:
            formattedItem = cell.render(fmt, specs)
            if isinstance(formattedItem, list):  # formatters can create multiple table cells by returning *lists*
                formattedItems.extend(formattedItem)
            else:
                formattedItems.append(formattedItem)
        return formattedItems
