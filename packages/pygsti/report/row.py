""" Defines the Row class """
from __future__ import division, print_function, absolute_import, unicode_literals

#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from .cell        import Cell
from ..objects.reportableqty import ReportableQty as _ReportableQty

class Row(object):
    '''
    Representation of a table row
    '''
    def __init__(self, rowData=None, formatters=None, labels=None, nonMarkovianEBs=False):
        '''
        Create a row object 

        Parameters
        ----------
        rowData : list
            raw data for the table
        formatters : optional, list[string]
            formatting options for each cell
        labels : optional list[string]
            labeling options for each cell
        nonMarkovianEBs : bool
            boolean indicating if non markovian error bars should be used
        '''
        if rowData is None:
            rowData = []
        else:
            rowData = [_ReportableQty.from_val(item, nonMarkovianEBs) for item in rowData]
        if formatters is None:
            formatters = []
        if labels is None:
            labels = rowData

        lendiff = max(abs(len(formatters) - len(rowData)), 0)
        formatters = list(formatters) + [None] * lendiff
        
        lendiff = max(abs(len(labels) - len(rowData)), 0)
        labels = list(labels) + [None] * lendiff 

        self.cells = [Cell(item, formatter, label) 
                for item, formatter, label in 
                zip(rowData, formatters, labels)]

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        return state_dict 

    def __setstate__(self, d):
        self.__dict__.update(d)

    def add(self, data, formatter=None, label=None):
        """ Adds a cell with the given `data`, `formatter` and `label` """
        self.cells.append(Cell(data, formatter, label))

    def render(self, fmt, specs):
        '''
        Render a row of cells

        Parameters
        ----------
        fmt : string
            format to be rendered in
        specs : dict
            options for formatting
        Returns
        -------
        list
        '''
        formattedItems = []
        for cell in self.cells:
            formattedItem = cell.render(fmt, specs)
            if isinstance(formattedItem, list): #formatters can create multiple table cells by returning *lists* 
                formattedItems.extend(formattedItem)
            else:
                formattedItems.append(formattedItem)
        return formattedItems
