from .notebookcell import NotebookCell

import os   as _os
import json as _json

from subprocess import call as _call

class Notebook(object):
    DefaultTemplate = 'Empty.ipynb'

    def __init__(self, cells=None, notebookTextFiles=None):
        if cells is None:
            cells = []
        self.cells = cells
        if notebookTextFiles is not None:
            for filename in notebookTextFiles:
                self.add_notebook_text_file(filename)

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

    def add_notebook_text(self, text):
        for block in text.split('@@'):
            if block == '':
                continue
            if block.startswith('code'):
                block = block.replace('code', '', 1)
                '''
                TODO: Move comments to markdown
                lines = []
                for line in block.splitlines():
                    if '#' in line:
                        i = line.index('#')
                '''
                self.add_code(block)
            elif block.startswith('markdown'):
                block = block.replace('markdown', '', 1)
                self.add_markdown(block)
            else:
                raise ValueError('Invalid notebook text block heading:\n{}'.format(block))

    def add_notebook_text_file(self, filename):
        with open(filename, 'r') as infile:
            self.add_notebook_text(infile.read())

    def add_notebook_text_files(self, filenames):
        for filename in filenames:
            self.add_notebook_text_file(filename)

    def add_notebook_file(filename):
        with open(templateFilename, 'r') as infile:
            notebookDict = _json.load(infile)
        for cell in notebookDict['cells']:
            self.cells.append(NotebookCell(cell['cell_type'], cell['source']))


    def launch_as(self, outputFilename, templateFilename=DefaultTemplate):
        self.save_to(outputFilename, templateFilename)
        _call('jupyter notebook {}'.format(outputFilename), shell=True) 
