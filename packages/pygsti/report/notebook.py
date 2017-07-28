from .notebookcell import NotebookCell

import os   as _os
import json as _json

from subprocess import call as _call

class Notebook(object):
    DefaultTemplate = 'Empty.ipynb'

    def __init__(self, cells=None):
        if cells is None:
            cells = []
        self.cells = cells

    def to_json_dict(self, templateFilename=DefaultTemplate):
        templateFilename = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                          'templates', templateFilename )
        with open(templateFilename, 'r') as infile:
            notebookDict = _json.load(infile)
        notebookDict['cells'].extend([c.to_json_dict() for c in self.cells])
        return notebookDict

    def save_to(self, outputFilename, templateFilename=DefaultTemplate):
        jsonDict = self.to_json_dict(templateFilename)
        with open(outputFilename, 'w') as outfile:
            _json.dump(jsonDict, outfile)

    def add(self, cell):
        self.cells.append(cell)

    def add_block(self, block, cellType):
        lines = block.splitlines(True)
        self.cells.append(NotebookCell(cellType, lines))

    def add_file(self, filename, cellType):
        with open(filename, 'r') as infile:
            block = infile.read()
        self.add_block(block, cellType)

    def add_code(self, block):
        self.add_block(block, 'code')

    def add_markdown(self, block):
        self.add_block(block, 'markdown')

    def add_code_file(self, filename):
        self.add_file(filename, 'code')

    def add_markdown_file(self, filename):
        self.add_file(filename, 'markdown')

    def launch_as(self, outputFilename, templateFilename=DefaultTemplate):
        self.save_to(outputFilename, templateFilename)
        _call('jupyter notebook {}'.format(outputFilename), shell=True) 
