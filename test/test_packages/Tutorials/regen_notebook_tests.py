from __future__ import print_function
import os, sys

#Header for output file
header = """from notebookstestcase import NotebooksTestCase

class NotebooksMethods(NotebooksTestCase):
"""

def main(args):
    if len(args) != 2:
        print("Usage: <path_to_notebook_root> <output_name>")
        print(" (e.g. '../../../jupyer_notebooks/Tutorials' 'testTutorials.py')")
        return -1

    #tutorialDir = os.path.join("../../..","jupyter_notebooks","Tutorials")
    notebookDir = args[0]
    outfilename = args[1]
    assert(os.path.isdir(notebookDir)), "%s must be a directory!" % notebookDir
    unit_tests = []
    
    #Loop through all notebook files (recursively), creating a test case for each
    def process_dir(path,level=0):
        for fn in os.listdir(path):
            name, ext = os.path.splitext(fn)
            full_fn = os.path.join(path,fn)
            if os.path.isdir(full_fn):
                process_dir(full_fn,level+1)
            elif ext == '.ipynb' and 'DEBUG' not in name and 'debug' not in name \
                 and 'checkpoint' not in name: # skip notebooks w/'DEBUG' or 'checkpoint' in them
                testname = name.replace(' ','_').replace('-','_') # get rid of spaces & dashes
                unit_tests.append( (level, name,
                                    ('    def test_%s(self):\n'
                                    '        self.runNotebook_jupyter("%s","%s")\n') % (testname, path, fn)))

    process_dir(notebookDir) # fills unit_tests with (level, name, text) tuples
    unit_tests.sort(key=lambda x: (x[0],x[1])) # sort by level, then by name

    outfile_str = header + '\n'.join([x[2] for x in unit_tests])
    with open(outfilename,'w') as f:
        f.write(outfile_str)
        
    print("Wrote %s" % outfilename)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
