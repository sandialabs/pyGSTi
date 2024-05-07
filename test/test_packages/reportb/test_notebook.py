import time
import unittest

from pygsti.report import Notebook
from ..testutils import BaseTestCase, temp_files

example_python = """\
import numpy as np
a = np.array([1,2,3],'d')
"""

example_markdown = """\
#Heading1
## Heading2
Some text about something important.
"""

example_multicell_text = """\
@@markdown
# Title
@@code
print('code goes here')
"""

example_multicell_text2 = """\
@@markdown
# Title2
@@code
print('more code goes here')
"""


# Covers some missing tests, but NOT all of report.table.py
class TestNotebook(BaseTestCase):
    def setUp(self):
        super(TestNotebook, self).setUp()

        #Needs to be in setup so temp_files works
        with open(temp_files+"/nb_example.py",'w') as f:
            f.write(example_python)
        with open(temp_files+"/nb_example.md",'w') as f:
            f.write(example_markdown)
        with open(temp_files+"/nb_example.txt",'w') as f:
            f.write(example_multicell_text)
        with open(temp_files+"/nb_example2.txt",'w') as f:
            f.write(example_multicell_text2)


    def test_notebook_construction(self):
        
        #Notebook object
        nb = Notebook()

        nb.add_markdown('# Pygsti report\n(Created on {})'.format(time.strftime("%B %d, %Y")))
        nb.add_code('print("Hello World")')
        nb.add_notebook_text(
            """@@markdown
            ### Sub-Title
            @@code
            print('Hello again!')
            """)

        with self.assertRaises(ValueError):
            nb.add_notebook_text("""@@foobar
            Unknown cell type!
            """)
                    
        nb.add_file(temp_files+"/nb_example.md","markdown")
        nb.add_file(temp_files+"/nb_example.py","code")
        
        nb.add_code_file(temp_files+"/nb_example.py")
        nb.add_markdown_file(temp_files+"/nb_example.md")
        nb.add_notebook_text_file(temp_files+"/nb_example.txt")

        nb.add_notebook_text_files([temp_files+"/nb_example.txt",
                                    temp_files+"/nb_example2.txt"])
        nb.save_to(temp_files+'/TestNotebook1.ipynb')

        nb2 = Notebook(notebook_text_files=
                       [temp_files+"/nb_example.txt",
                        temp_files+"/nb_example2.txt"])
        nb2.add_notebook_files([temp_files+'/TestNotebook1.ipynb'])
        nb.save_to(temp_files+'/TestNotebook2.ipynb')


if __name__ == '__main__':
    unittest.main(verbosity=2)
