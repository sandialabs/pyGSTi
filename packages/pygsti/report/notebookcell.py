""" Defines the NotebookCell class """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import json as _json
import os as _os


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
        self.source = source

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
                                         'templates', templateFilename)
        with open(templateFilename, 'r') as infile:
            cellDict = _json.load(infile)
        cellDict['source'].extend(self.source)
        return cellDict
