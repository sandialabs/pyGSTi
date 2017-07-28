#!/usr/bin/env python3
from notebook import Notebook, NotebookCell

#a = NotebookCell('markdown', ['#test'])
#print(a.to_json_dict())
n = Notebook()
n.add(NotebookCell('markdown', ['#Custom generated notebook']))
n.add(NotebookCell('code', ['print(\'Running custom code\')']))
n.save_to('test.ipynb')
