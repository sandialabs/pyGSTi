from notebookstestcase import NotebooksTestCase

class NotebooksMethods(NotebooksTestCase):
    def test_00_Protocols(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials","00-Protocols.ipynb")

    def test_01_Essential_Objects(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials","01-Essential-Objects.ipynb")

    def test_02_Using_Essential_Objects(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials","02-Using-Essential-Objects.ipynb")

    def test_03_Miscellaneous(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials","03-Miscellaneous.ipynb")

    def test_Circuit(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects","Circuit.ipynb")

    def test_CircuitLists(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects","CircuitLists.ipynb")

    def test_CircuitSimulation(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","CircuitSimulation.ipynb")

    def test_DataSet(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects","DataSet.ipynb")

    def test_DatasetComparison(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","DatasetComparison.ipynb")

    def test_DriftCharacterization(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","DriftCharacterization.ipynb")

    def test_ExplicitModel(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects","ExplicitModel.ipynb")

    def test_FileIO(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/other","FileIO.ipynb")

    def test_GST_Driverfunctions(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","GST-Driverfunctions.ipynb")

    def test_GST_Overview(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","GST-Overview.ipynb")

    def test_GST_Overview_functionbased(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","GST-Overview-functionbased.ipynb")

    def test_GST_Protocols(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","GST-Protocols.ipynb")

    def test_IdleTomography(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","IdleTomography.ipynb")

    def test_ImplicitModel(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects","ImplicitModel.ipynb")

    def test_Metrics(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting","Metrics.ipynb")

    def test_ModelAnalysisMetrics(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting","ModelAnalysisMetrics.ipynb")

    def test_ModelTesting(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","ModelTesting.ipynb")

    def test_ModelTesting_functions(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","ModelTesting-functions.ipynb")

    def test_RB_CliffordRB(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","RB-CliffordRB.ipynb")

    def test_RB_DirectRB(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","RB-DirectRB.ipynb")

    def test_RB_MirrorRB(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","RB-MirrorRB.ipynb")

    def test_RB_MultiRBExperiments(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","RB-MultiRBExperiments.ipynb")

    def test_RB_Overview(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","RB-Overview.ipynb")

    def test_RB_Samplers(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","RB-Samplers.ipynb")

    def test_ReportGeneration(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting","ReportGeneration.ipynb")

    def test_RobustPhaseEstimation(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","RobustPhaseEstimation.ipynb")

    def test_VolumetricBenchmarks(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms","VolumetricBenchmarks.ipynb")

    def test_Workspace(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting","Workspace.ipynb")

    def test_WorkspaceExamples(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting","WorkspaceExamples.ipynb")

    def test_CliffordRB_Simulation_ExplicitModel(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms/advanced","CliffordRB-Simulation-ExplicitModel.ipynb")

    def test_CliffordRB_Simulation_ImplicitModel(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms/advanced","CliffordRB-Simulation-ImplicitModel.ipynb")

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

    def test_ModelPacks(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","ModelPacks.ipynb")

    def test_MultiDataSet(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","MultiDataSet.ipynb")

    def test_OperationFactories(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","OperationFactories.ipynb")

    def test_Operators(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","Operators.ipynb")

    def test_ProcessorSpec(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","ProcessorSpec.ipynb")

    def test_Results(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","Results.ipynb")

    def test_StateSpaceLabels(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","StateSpaceLabels.ipynb")

    def test_Time_dependent_GST(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/algorithms/advanced","Time-dependent-GST.ipynb")

    def test_TimestampedDataSets(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/objects/advanced","TimestampedDataSets.ipynb")

    def test_WorkspaceSwitchboards(self):
        self.runNotebook_jupyter("../../../jupyter_notebooks/Tutorials/reporting/advanced","WorkspaceSwitchboards.ipynb")
