from .notebookcell import NotebookCell

import os   as _os
import json as _json

class Notebook(object):
    def __init__(self):
        self.cells = []

    def to_json_dict(self, templateFilename='Empty.ipynb'):
        templateFilename = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                          'templates', templateFilename )
        with open(templateFilename, 'r') as infile:
            notebookDict = _json.load(infile)
        notebookDict['cells'].extend([c.to_json_dict() for c in self.cells])
        return notebookDict

    def save_to(self, outputFilename, templateFilename='Empty.ipynb'):
        with open(outputFilename, 'w') as outfile:
            _json.dump(self.to_json_dict(templateFilename), outfile)

    def add(self, cell):
        self.cells.append(cell)
