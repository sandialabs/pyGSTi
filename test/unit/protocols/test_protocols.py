import pathlib

import pygsti
from pygsti.modelpacks import smq1Q_XYI as std
from ..util import BaseCase, with_temp_path


class ExperimentDesignTester(BaseCase):

    @classmethod
    def setUpClass(cls):
        cls.gst_design = std.create_gst_experiment_design(4)

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

    def test_map_edesign_sslbls(self):
        edesigns = self._get_tester_edesigns()
        for edesign in edesigns:
            print("Testing edesign of type: ", str(type(edesign)))
            orig_qubits = edesign.qubit_labels
            for c in edesign.all_circuits_needing_data:
                self.assertTrue(set(c.line_labels).issubset(orig_qubits))

            if orig_qubits == (0,):
                mapper = {0: 4}; mapped_qubits = (4,)
            if orig_qubits == (1,):
                mapper = {1: 5}; mapped_qubits = (5,)
            if orig_qubits == (0,1):
                mapper = {0:4, 1: 5}; mapped_qubits = (4,5)
            mapped_edesign = edesign.map_qubit_labels(mapper)
            self.assertEqual(mapped_edesign.qubit_labels, mapped_qubits)
            for c in mapped_edesign.all_circuits_needing_data:
                self.assertTrue(set(c.line_labels).issubset(mapped_qubits))

    @with_temp_path
    def test_serialization(self, root_path):
        edesigns = self._get_tester_edesigns()
        for i, edesign in enumerate(edesigns):
            print("Testing edesign of type: ", str(type(edesign)))
            root = pathlib.Path(root_path) / str(i)
            edesign.write(root)
            loaded_edesign = type(edesign).from_dir(root)
            # TODO: We don't have good edesign equality
            self.assertEqual(set(edesign.all_circuits_needing_data), set(loaded_edesign.all_circuits_needing_data))
            self.assertEqual(edesign.auxfile_types, loaded_edesign.auxfile_types)
            self.assertEqual(edesign._vals.keys(), loaded_edesign._vals.keys())

            if isinstance(edesign, (pygsti.protocols.CombinedExperimentDesign, pygsti.protocols.FreeformDesign)):
                # We also need to test that all_circuits_needing_data is not dumped by default
                self.assertTrue(not (root / 'edesign' / 'all_circuits_needing_data.txt').exists())

                root2 = pathlib.Path(root_path) / f'{i}_2'
                edesign.all_circuits_needing_data = []
                edesign.write(root2)
                loaded_edesign = type(edesign).from_dir(root2)
                # TODO: We don't have good edesign equality
                self.assertEqual(set(edesign.all_circuits_needing_data), set(loaded_edesign.all_circuits_needing_data))
                self.assertEqual(edesign.auxfile_types, loaded_edesign.auxfile_types)
                self.assertEqual(edesign._vals.keys(), loaded_edesign._vals.keys())
                self.assertTrue((root2 / 'edesign' / 'all_circuits_needing_data.txt').exists())
    
    def _get_tester_edesigns(self):
        #Create a bunch of experiment designs:
        from pygsti.protocols import ExperimentDesign, CircuitListsDesign, CombinedExperimentDesign, \
            SimultaneousExperimentDesign, FreeformDesign, StandardGSTDesign, GateSetTomographyDesign, \
            CliffordRBDesign, DirectRBDesign, MirrorRBDesign
        from pygsti.processors import CliffordCompilationRules as CCR

        circuits_on0 = pygsti.circuits.to_circuits(["{}@(0)", "Gxpi2:0", "Gypi2:0"], line_labels=(0,))
        circuits_on0b = pygsti.circuits.to_circuits(["Gxpi2:0^2", "Gypi2:0^2"], line_labels=(0,))
        circuits_on1 = pygsti.circuits.to_circuits(["Gxpi2:1^2", "Gypi2:1^2"], line_labels=(1,))
        circuits_on01 = pygsti.circuits.to_circuits(["Gcnot:0:1", "Gxpi2:0Gypi2:1^2Gcnot:0:1Gxpi:0"],
                                                    line_labels=(0,1))

        #For GST edesigns
        mdl = std.target_model()
        gst_pspec = mdl.create_processor_spec()

        #For RB edesigns
        pspec = pygsti.processors.QubitProcessorSpec(2, ["Gxpi2", "Gypi2","Gxx"],
                                                     geometry='line', qubit_labels=(0,1))
        compilations = {"absolute": CCR.create_standard(pspec, "absolute", ("paulis", "1Qcliffords"), verbosity=0),
                        "paulieq": CCR.create_standard(pspec, "paulieq", ("1Qcliffords", "allcnots"), verbosity=0),
                        }

        pspec1Q = pygsti.processors.QubitProcessorSpec(1, ["Gxpi2", "Gypi2","Gxmpi2", "Gympi2"],
                                                       geometry='line', qubit_labels=(0,))
        compilations1Q = {"absolute": CCR.create_standard(pspec1Q, "absolute", ("paulis", "1Qcliffords"), verbosity=0),
                          "paulieq": CCR.create_standard(pspec1Q, "paulieq", ("1Qcliffords", "allcnots"), verbosity=0),
                          }

        edesigns = []
        edesigns.append(ExperimentDesign(circuits_on0))
        edesigns.append(CircuitListsDesign([circuits_on0, circuits_on0b]))
        edesigns.append(CombinedExperimentDesign({'one': ExperimentDesign(circuits_on0),
                                                  'two': ExperimentDesign(circuits_on1),
                                                  'three': ExperimentDesign(circuits_on01)}, qubit_labels=(0,1)))
        edesigns.append(SimultaneousExperimentDesign([ExperimentDesign(circuits_on0), ExperimentDesign(circuits_on1)]))
        edesigns.append(FreeformDesign(circuits_on01))
        edesigns.append(std.create_gst_experiment_design(2))
        edesigns.append(GateSetTomographyDesign(gst_pspec, [circuits_on0, circuits_on0b]))
        edesigns.append(CliffordRBDesign(pspec, compilations, depths=[0,2,5], circuits_per_depth=4))
        edesigns.append(DirectRBDesign(pspec, compilations, depths=[0,2,5], circuits_per_depth=4))
        edesigns.append(MirrorRBDesign(pspec1Q, depths=[0,2,4], circuits_per_depth=4,
                                       clifford_compilations=compilations1Q))

        return edesigns