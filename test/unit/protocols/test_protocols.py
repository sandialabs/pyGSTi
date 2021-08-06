import pathlib

import pygsti
from pygsti.modelpacks import smq1Q_XYI as std
from ..util import BaseCase, with_temp_path


class ExperimentDesignTester(BaseCase):

    @classmethod
    def setUpClass(cls):
        cls.gst_design = std.get_gst_experiment_design(4)

    def test_promotion(self):
        circuits = pygsti.circuits.to_circuits(["{}@(0)", "Gxpi2:0", "Gypi2:0"])
        edesign1 = pygsti.protocols.ExperimentDesign(circuits)
        combined_edesign1 = edesign1.promote_to_combined('ed1')
        self.assertTrue(isinstance(combined_edesign1, pygsti.protocols.CombinedExperimentDesign))

        circuits2 = pygsti.circuits.to_circuits(["Gxpi2:0^2", "Gypi2:0^2"])
        edesign2 = pygsti.protocols.CircuitListsDesign([circuits, circuits2])

        combined_edesign2 = edesign2.promote_to_combined('ed2')
        self.assertTrue(isinstance(combined_edesign2, pygsti.protocols.CombinedExperimentDesign))

        combined_edesign3 = combined_edesign2.promote_to_combined('ednested')
        self.assertTrue(isinstance(combined_edesign3, pygsti.protocols.CombinedExperimentDesign))
        self.assertEqual(set(combined_edesign3.keys()), set(['ednested']))

    def test_add_to_combined_design(self):
        circuits1 = pygsti.circuits.to_circuits(["{}@(0)", "Gxpi2:0", "Gypi2:0"])
        circuits2 = pygsti.circuits.to_circuits(["Gxpi2:0^2", "Gypi2:0^2"])

        edesign1 = pygsti.protocols.ExperimentDesign(circuits1)
        edesign2 = pygsti.protocols.ExperimentDesign(circuits2)

        combined_edesign = pygsti.protocols.CombinedExperimentDesign({"one": edesign1, "two": edesign2})

        edesign3 = pygsti.protocols.ExperimentDesign(pygsti.circuits.to_circuits(["Gxpi2:0", "Gypi2:0^2"]))
        combined_edesign['three'] = edesign3

        with self.assertRaises(ValueError):
            edesign4 = pygsti.protocols.ExperimentDesign(pygsti.circuits.to_circuits(["Gypi2:0^4"]))
            combined_edesign['four'] = edesign4
            
    #These might be more "system tests"
    @with_temp_path
    def test_create_edesign_fromdir_single(self, root_path):
        # Simple edesign subdirectory with a single circuit list
        root = pathlib.Path(root_path)
        root.mkdir(exist_ok=True)
        (root / 'edesign').mkdir(exist_ok=True)
        pygsti.io.write_circuit_list(root / 'edesign' / 'circuits.txt', self.gst_design.circuit_lists[0])

        edesign1 = pygsti.io.create_edesign_from_dir(str(root))
        self.assertTrue(isinstance(edesign1, pygsti.protocols.ExperimentDesign))
        self.assertTrue(all([a == b for a,b in zip(edesign1.all_circuits_needing_data, self.gst_design.circuit_lists[0])]))


    @with_temp_path
    def test_create_edesign_fromdir_multi(self, root_path):
        # Simple edesign subdirectory with multiple circuit lists
        root = pathlib.Path(root_path)
        root.mkdir(exist_ok=True)
        (root / 'edesign').mkdir(exist_ok=True)
        pygsti.io.write_circuit_list(root / 'edesign' / 'circuits0.txt', self.gst_design.circuit_lists[0])
        pygsti.io.write_circuit_list(root / 'edesign' / 'circuits1.txt', self.gst_design.circuit_lists[1])

        edesign2 = pygsti.io.create_edesign_from_dir(str(root))
        self.assertTrue(isinstance(edesign2, pygsti.protocols.CircuitListsDesign))
        self.assertTrue(all([a == b for a,b in zip(edesign2.circuit_lists[0], self.gst_design.circuit_lists[0])]))
        self.assertTrue(all([a == b for a,b in zip(edesign2.circuit_lists[1], self.gst_design.circuit_lists[1])]))

    @with_temp_path
    def test_create_edesign_fromdir_subdirs(self, root_path):        
        # directory with several edesign subdirectories
        root = pathlib.Path(root_path)
        root.mkdir(exist_ok=True)
        (root / 'subdir1').mkdir(exist_ok=True)
        (root / 'subdir2').mkdir(exist_ok=True)
        (root / 'subdir1' / 'edesign').mkdir(exist_ok=True)
        (root / 'subdir2' / 'edesign').mkdir(exist_ok=True)
        pygsti.io.write_circuit_list(root / 'subdir1' / 'edesign' / 'circuits0.txt', self.gst_design.circuit_lists[0])
        pygsti.io.write_circuit_list(root / 'subdir2' / 'edesign' / 'circuits1.txt', self.gst_design.circuit_lists[1])

        edesign3 = pygsti.io.create_edesign_from_dir(str(root))
        self.assertTrue(isinstance(edesign3, pygsti.protocols.CombinedExperimentDesign))
        self.assertTrue(all([a == b for a,b in zip(edesign3['subdir1'].all_circuits_needing_data, self.gst_design.circuit_lists[0])]))
        self.assertTrue(all([a == b for a,b in zip(edesign3['subdir2'].all_circuits_needing_data, self.gst_design.circuit_lists[1])]))

