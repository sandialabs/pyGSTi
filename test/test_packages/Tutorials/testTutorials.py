import unittest
import os, sys
import importlib
import pygsti

#if __name__ == "__main__":
#    tutorialFile = "01 GateSets.ipynb"
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

tutorialDir = os.path.join("../../..","jupyter_notebooks","Tutorials")

class TutorialsTestCase(unittest.TestCase):

    def setUp(self):
        self.old = os.getcwd()
        os.chdir(os.path.abspath(os.path.dirname(__file__)))

        #Set GateSet objects to non-"strict" mode, as this would be the
        # tutorial environment
        pygsti.objects.GateSet._strict = False

    def tearDown(self):
        os.chdir(self.old)

    def runTutorial_as_script(self, tutorialFile):
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

    def runTutorial_jupyter(self, tutorialFile, inplace=True):

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


class TutorialsMethods(TutorialsTestCase):

class TutorialsMethods(TutorialsTestCase):

    def test_tutorial_00(self):
        self.runTutorial_jupyter("00 Getting Started.ipynb")

    def test_tutorial_01(self):
        self.runTutorial_jupyter("01 GateSets.ipynb")

    def test_tutorial_02(self):
        self.runTutorial_jupyter("02 Matrix Bases.ipynb")

    def test_tutorial_03(self):
        self.runTutorial_jupyter("03 GateStrings.ipynb")

    def test_tutorial_04(self):
        self.runTutorial_jupyter("04 DataSets.ipynb")

    def test_tutorial_05(self):
        self.runTutorial_jupyter("05 Text file IO.ipynb")

    def test_tutorial_06(self):
        self.runTutorial_jupyter("06 Fiducials, Germs, and Maximum Lengths.ipynb")

    def test_tutorial_07(self):
        self.runTutorial_jupyter("07 Fiducial and Germ Selection.ipynb")

    def test_tutorial_08(self):
        self.runTutorial_jupyter("08 Algorithms low-level.ipynb")

    def test_tutorial_09(self):
        self.runTutorial_jupyter("09 Algorithms high-level.ipynb")

    def test_tutorial_10(self):
        self.runTutorial_jupyter("10 Fiducial Pair Reduction.ipynb")

    def test_tutorial_11(self):
        self.runTutorial_jupyter("11 Results.ipynb")

    def test_tutorial_12(self):
        self.runTutorial_jupyter("12 Report Generation.ipynb")

    def test_tutorial_13(self):
        self.runTutorial_jupyter("13 Workspace Basics.ipynb")

    def test_tutorial_14(self):
        self.runTutorial_jupyter("14 Workspace Switchboards.ipynb")

    def test_tutorial_15(self):
        self.runTutorial_jupyter("15 Randomized Benchmarking.ipynb")

    def test_tutorial_16(self):
        self.runTutorial_jupyter("16 Robust Phase Estimation.ipynb")

    def test_tutorial_17(self):
        self.runTutorial_jupyter("17 Pure Data Analysis.ipynb")

    def test_tutorial_18(self):
        self.runTutorial_jupyter("18 Model Testing.ipynb")

    def test_tutorial_19(self):
        self.runTutorial_jupyter("19 Basic drift characterization.ipynb")

    def test_tutorial_20(self):
        self.runTutorial_jupyter("20 Intermediate Measurements.ipynb")


if __name__ == "__main__":
    unittest.main(verbosity=2)
