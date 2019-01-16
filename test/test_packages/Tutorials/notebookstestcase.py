import unittest
import os, sys
import importlib
import pygsti

#if __name__ == "__main__":
#    tutorialFile = "01 Models.ipynb"
#    tutorialDir = os.path.join("..","jupyter_notebooks","Tutorials")
#    tutorialModuleName = os.path.splitext(tutorialFile)[0]
#    os.chdir(tutorialDir)
#    sys.path.append(".") #tutorialDir)
#    #cwd = os.getcwd()
#    #os.chdir(tutorialDir)
#    tutorial_module = importlib.import_module(tutorialModuleName)
#    #os.chdir(cwd)
#    #sys.path.pop()
#    exit()

class NotebooksTestCase(unittest.TestCase):

    def setUp(self):
        self.old = os.getcwd()
        os.chdir(os.path.abspath(os.path.dirname(__file__)))

        #Set Model objects to non-"strict" mode, as this would be the
        # tutorial environment
        pygsti.objects.ExplicitOpModel._strict = False

    def tearDown(self):
        os.chdir(self.old)

    def runNotebook_as_script(self, tutorialDir, tutorialFile):
        tutorialModuleName = os.path.splitext(tutorialFile)[0]
        cwd = os.getcwd()
        os.chdir(tutorialDir)
        os.system('jupyter nbconvert --to script "%s"' % tutorialFile)
        sys.path.append(".") # not "tutorialDir" b/c of chdir above

        orig_stdout = sys.stdout
        sys.stdout = open(tutorialModuleName + ".out","w")

        try:
            tutorial_module = importlib.import_module(tutorialModuleName)
            sys.stdout.close()
            os.remove(tutorialModuleName + ".py") #only remove if all is OK
            os.remove(tutorialModuleName + ".pyc")

            #do comparison with accepted tutorial output?
            os.remove(tutorialModuleName + ".out") #only remove if all is OK

        finally:
            sys.stdout.close()
            sys.stdout = orig_stdout
            sys.path.pop()
            os.chdir(cwd)

    def runNotebook_jupyter(self, tutorialDir, tutorialFile, inplace=True):

        #check whether an jupyter notebook contains any "error" cells
        def containsErrors(fn):
            with open(fn, 'r') as fn_file:
                for line in fn_file:
                    if '"output_type": "error"' in line:
                        return True
                return False

        
        tutorialName,Ext = os.path.splitext(tutorialFile)

        cwd = os.getcwd()
        os.chdir(tutorialDir)
        if inplace:
            os.system('jupyter nbconvert --to notebook --execute ' +
                      '--ExecutePreprocessor.timeout=3600 --inplace ' +
                      '"%s"' % tutorialFile)
            outName = tutorialFile # executed notebook (side benefit of updating notebook)
            if containsErrors(outName):
                os.chdir(cwd)
                raise ValueError("Error(s) occurred when running tutorial '%s'." % tutorialFile +
                                 "  open notebook to see error details.")
            # (never remove original notebook)
        else:
            os.system('jupyter nbconvert --to notebook --execute "%s"' % tutorialFile)
            outName = tutorialName + ".nbconvert." + Ext # executed notebook
            if containsErrors(outName):
                os.chdir(cwd)
                raise ValueError("Error(s) occurred when running tutorial '%s'." % tutorialFile +
                                 "  Open output notebook '%s' for details." % outName)

            else:
                os.remove(outName)
        os.chdir(cwd)

