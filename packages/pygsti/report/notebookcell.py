import json as _json
import os   as _os

class NotebookCell(object):
    def __init__(self, cellType='code', source=None):
        if source is None:
            source = []
        self.cellType = cellType
        self.source   = source

    def to_json_dict(self):
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
