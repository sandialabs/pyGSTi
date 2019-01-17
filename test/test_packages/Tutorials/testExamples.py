from notebookstestcase import NotebooksTestCase

class NotebooksMethods(NotebooksTestCase):
    def test_2QGST_CreatingModels(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","2QGST-CreatingModels.ipynb")

    def test_2QGST_RunningIt(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","2QGST-RunningIt.ipynb")

    #Intentional keyboard interrupt
    #def test_2QGST_ErrorBars(self):
    #    self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","2QGST-ErrorBars.ipynb")

    def test_BootstrappedErrorBars(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","BootstrappedErrorBars.ipynb")

    def test_ContextDependence(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","ContextDependence.ipynb")

    def test_GOpt_AddingNewOptimizations(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","GOpt-AddingNewOptimizations.ipynb")

    def test_GOpt_NonIdealTargets(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","GOpt-NonIdealTargets.ipynb")

    def test_Leakage(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","Leakage.ipynb")

    #Intentional keyboard interrupt
    #def test_MPI_GermSelection(self):
    #    self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","MPI-GermSelection.ipynb")

    def test_MPI_RunningGST(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","MPI-RunningGST.ipynb")

    def test_QutritGST(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","QutritGST.ipynb")

    def test_Reports_LGSTonly(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Examples/","Reports-LGSTonly.ipynb")
