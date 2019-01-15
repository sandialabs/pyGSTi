from notebookstestcase import NotebooksTestCase

class NotebooksMethods(NotebooksTestCase):
    def test_01_Essential_Objects(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/","01-Essential-Objects.ipynb")

    def test_02_Applications(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/","02-Applications.ipynb")

    def test_03_Miscellaneous(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/","03-Miscellaneous.ipynb")

    def test_BasicDriftCharacterization(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","BasicDriftCharacterization.ipynb")

    def test_Circuit(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects","Circuit.ipynb")

    def test_CircuitLists(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects","CircuitLists.ipynb")

    def test_CliffordRB(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","CliffordRB.ipynb")

    def test_DataSet(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects","DataSet.ipynb")

    def test_DatasetComparison(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","DatasetComparison.ipynb")

    def test_DirectRB(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","DirectRB.ipynb")

    def test_ExplicitModel(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects","ExplicitModel.ipynb")

    def test_FileIO(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/other","FileIO.ipynb")

    def test_GST_Drivers(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","GST-Drivers.ipynb")

    def test_GST_Overview(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","GST-Overview.ipynb")

    def test_ImplicitModel(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects","ImplicitModel.ipynb")

    def test_Metrics(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting","Metrics.ipynb")

    def test_ModelAnalysisMetrics(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting","ModelAnalysisMetrics.ipynb")

    def test_ModelTesting(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","ModelTesting.ipynb")

    def test_RBAnalysis(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","RBAnalysis.ipynb")

    def test_ReportGeneration(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting","ReportGeneration.ipynb")

    def test_RobustPhaseEstimation(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","RobustPhaseEstimation.ipynb")

    def test_Workspace(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting","Workspace.ipynb")

    def test_WorkspaceExamples(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting","WorkspaceExamples.ipynb")

    def test_CustomOperator(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","CustomOperator.ipynb")

    def test_ForwardSimulationTypes(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms/advanced","ForwardSimulationTypes.ipynb")

    def test_GST_FiducialAndGermSelection(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms/advanced","GST-FiducialAndGermSelection.ipynb")

    def test_GST_FiducialPairReduction(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms/advanced","GST-FiducialPairReduction.ipynb")

    def test_GST_LowLevel(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms/advanced","GST-LowLevel.ipynb")

    def test_GSTCircuitConstruction(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","GSTCircuitConstruction.ipynb")

    def test_GaugeOpt(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms/advanced","GaugeOpt.ipynb")

    def test_Instruments(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","Instruments.ipynb")

    def test_MatrixBases(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","MatrixBases.ipynb")

    def test_MultiDataSet(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","MultiDataSet.ipynb")

    def test_Operators(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","Operators.ipynb")

    def test_ProcessorSpec(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","ProcessorSpec.ipynb")

    def test_Results(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","Results.ipynb")

    def test_StandardModules(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","StandardModules.ipynb")

    def test_StateSpaceLabels(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","StateSpaceLabels.ipynb")

    def test_TimestampedDataSets(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","TimestampedDataSets.ipynb")

    def test_WorkspaceSwitchboards(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting/advanced","WorkspaceSwitchboards.ipynb")
