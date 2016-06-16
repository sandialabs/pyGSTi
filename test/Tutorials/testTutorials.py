import unittest
import os, sys
import importlib
import pygsti

#if __name__ == "__main__":
#    tutorialFile = "01 GateSets.ipynb"
#    tutorialDir = os.path.join("..","ipython_notebooks","Tutorials")
#    tutorialModuleName = os.path.splitext(tutorialFile)[0]
#    os.chdir(tutorialDir)
#    sys.path.append(".") #tutorialDir)
#    #cwd = os.getcwd()
#    #os.chdir(tutorialDir)
#    tutorial_module = importlib.import_module(tutorialModuleName)
#    #os.chdir(cwd)
#    #sys.path.pop()
#    exit()

class TutorialsTestCase(unittest.TestCase):

    def setUp(self):
        #Set GateSet objects to non-"strict" mode, as this would be the 
        # tutorial environment
        pygsti.objects.GateSet._strict = False

        pass

    def runTutorial_as_script(self, tutorialFile):
        tutorialDir = os.path.join("..","ipython_notebooks","Tutorials")
        tutorialModuleName = os.path.splitext(tutorialFile)[0]
        cwd = os.getcwd()
        os.chdir(tutorialDir)
        os.system('ipython nbconvert --to script "%s"' % tutorialFile)
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

    def runTutorial_ipython(self, tutorialFile, inplace=True):

        #check whether an ipython notebook contains any "error" cells
        def containsErrors(fn):
            for line in open(fn, 'r'):
                if '"output_type": "error"' in line:
                    return True
            return False

        tutorialDir = os.path.join("..","ipython_notebooks","Tutorials")
        tutorialName,Ext = os.path.splitext(tutorialFile)

        cwd = os.getcwd()
        os.chdir(tutorialDir)
        if inplace:
            os.system('ipython nbconvert --to notebook --execute ' +
                      '--ExecutePreprocessor.timeout=3600 --inplace ' +
                      '"%s"' % tutorialFile)
            outName = tutorialFile # executed notebook (side benefit of updating notebook)
            if containsErrors(outName): 
                os.chdir(cwd)
                raise ValueError("Error(s) occurred when running tutorial '%s'." % tutorialFile +
                                 "  open notebook to see error details.")
            # (never remove original notebook)
        else:
            os.system('ipython nbconvert --to notebook --execute "%s"' % tutorialFile)
            outName = tutorialName + ".nbconvert." + Ext # executed notebook
            if containsErrors(outName): 
                os.chdir(cwd)
                raise ValueError("Error(s) occurred when running tutorial '%s'." % tutorialFile +
                                 "  Open output notebook '%s' for details." % outName)

            else:
                os.remove(outName)
        os.chdir(cwd)


class TutorialsMethods(TutorialsTestCase):

    def test_tutorial_00(self):
        self.runTutorial_ipython("00 Quick and easy GST.ipynb")

    def test_tutorial_01(self):
        self.runTutorial_ipython("01 GateSets.ipynb")

    def test_tutorial_02(self):
        self.runTutorial_ipython("02 Gatestring lists.ipynb")
    
    def test_tutorial_03(self):
        self.runTutorial_ipython("03 DataSets.ipynb")
    
    def test_tutorial_04(self):
        self.runTutorial_ipython("04 Algorithms.ipynb")

    def test_tutorial_05(self):
        self.runTutorial_ipython("05 Plotting.ipynb")

    def test_tutorial_06(self):
        self.runTutorial_ipython("06 Advanced Algorithms.ipynb")
    
    def test_tutorial_07(self):
        self.runTutorial_ipython("07 Report Generation.ipynb")
    
    def test_tutorial_08(self):
        self.runTutorial_ipython("08 Fiducial Reduction.ipynb")

    def test_tutorial_09(self):
        self.runTutorial_ipython("09 Bootstrapped Error Bars.ipynb")

    def test_tutorial_10(self):
        self.runTutorial_ipython("10 Fiducial Selection - 1 qubit (X, Y).ipynb")

    def test_tutorial_11(self):
        self.runTutorial_ipython("11 Fiducial Selection - 2 qubits (IX, IY, XI, XY, Entangling).ipynb")

    def test_tutorial_12(self):
        self.runTutorial_ipython("12 Germ Selection.ipynb")

    def test_tutorial_13(self):
        self.runTutorial_ipython("13 GST on 2 qubits.ipynb")

    def test_tutorial_14(self):
        self.runTutorial_ipython("14 GST on 2 qubits - custom 2Q gate.ipynb")


      
if __name__ == "__main__":
    unittest.main(verbosity=2)
