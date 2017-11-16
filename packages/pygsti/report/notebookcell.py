""" Defines the NotebookCell class """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

import json as _json
import os   as _os

class NotebookCell(object):
    '''
    Struct representing either a code or markdown cell
    '''
    def __init__(self, cellType='code', source=None):
        '''
        Build a notebook cell

        Parameters
        ----------
        cellType : str, optional
            tag for the cell: either 'code' or 'markdown'
        source : list(str), optional
            lines of code/markdown in the cell
        '''
        if source is None:
            source = []
        self.cellType = cellType
        self.source   = source

    def to_json_dict(self):
        '''
        Convert this cell to a json representation of a cell,
        using a default template
        '''
        if self.cellType == 'markdown':
            templateFilename = 'MDcell.json'
        elif self.cellType == 'code':
            templateFilename = 'CodeCell.json'
        templateFilename = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                          'templates', templateFilename )
        with open(templateFilename, 'r') as infile:
            cellDict = _json.load(infile)
        cellDict['source'].extend(self.source)
        return cellDict
