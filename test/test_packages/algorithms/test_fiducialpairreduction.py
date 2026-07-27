import pickle
import unittest

import pytest

import pygsti
from pygsti.algorithms import germselection
from pygsti.modelpacks import smq1Q_XYI as std
from pygsti.tools.exceptions import pyGSTiDeprecationWarning
from .algorithmsTestCase import AlgorithmTestCase
from ..testutils import compare_files, regenerate_references

from pygsti.circuits import Circuit
from pygsti.processors import QubitProcessorSpec
from pygsti.models import create_crosstalk_free_model, ExplicitOpModel
from pygsti.algorithms.fiducialpairreduction import _copy_to_static_explicitop_model, _set_up_prep_POVM_tuples      


class FiducialPairReductionTestCase(AlgorithmTestCase):
    def test_memlimit(self):
        with self.assertRaises(MemoryError):
            # A very low memlimit
            pygsti.alg.find_sufficient_fiducial_pairs_per_germ(
                std.target_model(), std.prep_fiducials(), std.meas_fiducials(),
                std.germs(lite=True), n_random=2, verbosity=0,
                mem_limit=100)  # 100 bytes!
    
    #Two out of the three tests that were in the following function were superfluous, and taking
    #n_random out to very large values takes a long time to run, so I don't think it is worth the time
    #from a testing standpoint.

    def test_FPR_test_pairs(self):
        target_model = std.target_model()
        prep_fiducials = std.prep_fiducials()
        meas_fiducials = std.meas_fiducials()
        germs = std.germs(lite = False)

        op_labels = list(target_model.operations.keys())

        nTotal = germselection._remove_spam_vectors(target_model).num_nongauge_params
        self.assertEqual(nTotal, 34)

        fidPairsDict = pygsti.alg.find_sufficient_fiducial_pairs_per_germ(
            target_model, prep_fiducials, meas_fiducials, germs,
            search_mode="random", constrain_to_tp=True,
            n_random=10, seed=1234, verbosity=1,
            mem_limit=int(2*(1024)**3))

        nAmplified = pygsti.alg.test_fiducial_pairs(fidPairsDict, target_model, prep_fiducials,
                                                    meas_fiducials, germs,
                                                    verbosity=3, mem_limit=None,
                                                    test_lengths=(64, 512),
                                                    tol = 0.5)

        print("PFPR: %d AMPLIFIED out of %d total (non-spam non-gauge) params" % (nAmplified, nTotal))
        self.assertEqual(nAmplified, 34)

    def test_copy_to_static_explicitop_model_with_implicit(self):
        """Test _copy_to_static_explicitop_model with an ImplicitOpModel."""
        # Create a simple ImplicitOpModel (LocalNoiseModel is a subclass of ImplicitOpModel)
        pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2'], geometry='line')
        implicit_model = create_crosstalk_free_model(pspec, 
                                                     depolarization_strengths={'Gxpi2': 0.01, 'Gypi2': 0.01})
        
        # Verify it's an ImplicitOpModel
        from pygsti.models import ImplicitOpModel
        self.assertIsInstance(implicit_model, ImplicitOpModel)
        
        # Convert to static explicit model
        static_model = _copy_to_static_explicitop_model(implicit_model)
        
        # Verify the result is an ExplicitOpModel
        from pygsti.models import ExplicitOpModel
        self.assertIsInstance(static_model, ExplicitOpModel)
        
        # Verify it has zero parameters (all static)
        self.assertEqual(static_model.num_params, 0)
        
        # Verify it has the expected operations (gates are labeled with qubit indices)
        operation_names = [op.name if hasattr(op, 'name') else str(op) for op in static_model.operations.keys()]
        self.assertTrue(any('Gxpi2' in name for name in operation_names))
        self.assertTrue(any('Gypi2' in name for name in operation_names))
        
        # Verify it has preps and POVMs
        self.assertTrue(len(static_model.preps) > 0)
        self.assertTrue(len(static_model.povms) > 0)

    def test_copy_to_static_explicitop_model_with_explicit(self):
        """Test _copy_to_static_explicitop_model with an ExplicitOpModel."""
        # Use the standard 1-qubit model (ExplicitOpModel)
        explicit_model = std.target_model()
        
        # Verify it's an ExplicitOpModel
        self.assertIsInstance(explicit_model, ExplicitOpModel)
        
        # Convert to static
        static_model = _copy_to_static_explicitop_model(explicit_model)
        
        # Verify the result is still an ExplicitOpModel
        self.assertIsInstance(static_model, ExplicitOpModel)
        
        # Verify it has zero parameters (all static)
        self.assertEqual(static_model.num_params, 0)

    def test_set_up_prep_POVM_tuples_with_implicit_first(self):
        """Test _set_up_prep_POVM_tuples with ImplicitOpModel and prep_povm_tuples='first'."""
        from pygsti.processors import QubitProcessorSpec
        from pygsti.models import create_crosstalk_free_model
        from pygsti.algorithms.fiducialpairreduction import _set_up_prep_POVM_tuples
        from pygsti.circuits import Circuit
        
        # Create an ImplicitOpModel
        pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2'], geometry='line')
        implicit_model = create_crosstalk_free_model(pspec)
        
        # Test with prep_povm_tuples="first"
        result_tuples, dof = _set_up_prep_POVM_tuples(implicit_model, "first", return_meas_dofs=True)
        
        # Verify we get a list of tuples
        self.assertIsInstance(result_tuples, list)
        self.assertEqual(len(result_tuples), 1)
        
        # Verify each element is a tuple of Circuits
        prep_circuit, povm_circuit = result_tuples[0]
        self.assertIsInstance(prep_circuit, Circuit)
        self.assertIsInstance(povm_circuit, Circuit)
        
        # Verify dof_per_povm is reasonable
        self.assertIsInstance(dof, int)
        self.assertGreater(dof, 0)

    def test_set_up_prep_POVM_tuples_with_implicit_explicit_list(self):
        """Test _set_up_prep_POVM_tuples with ImplicitOpModel and explicit prep_povm_tuples list."""
        from pygsti.processors import QubitProcessorSpec
        from pygsti.models import create_crosstalk_free_model
        from pygsti.algorithms.fiducialpairreduction import _set_up_prep_POVM_tuples
        from pygsti.circuits import Circuit
        
        # Create an ImplicitOpModel
        pspec = QubitProcessorSpec(2, ['Gxpi2', 'Gypi2'], geometry='line')
        implicit_model = create_crosstalk_free_model(pspec)
        
        # Get the actual prep and POVM labels
        prep_label = list(implicit_model.prep_blks['layers'].keys())[0]
        povm_label = list(implicit_model.povm_blks['layers'].keys())[0]
        
        # Test with explicit list
        result_tuples = _set_up_prep_POVM_tuples(implicit_model, 
                                                 [(prep_label, povm_label)], 
                                                 return_meas_dofs=False)
        
        # Verify we get a list with one tuple
        self.assertIsInstance(result_tuples, list)
        self.assertEqual(len(result_tuples), 1)
        
        # Verify each element is a tuple of Circuits
        prep_circuit, povm_circuit = result_tuples[0]
        self.assertIsInstance(prep_circuit, Circuit)
        self.assertIsInstance(povm_circuit, Circuit)

    def test_set_up_prep_POVM_tuples_with_explicit(self):
        """Test _set_up_prep_POVM_tuples with ExplicitOpModel."""
        # Use the standard 1-qubit model (ExplicitOpModel)
        explicit_model = std.target_model()
        
        # Test with prep_povm_tuples="first"
        result_tuples, dof = _set_up_prep_POVM_tuples(explicit_model, "first", return_meas_dofs=True)
        
        # Verify we get a list of tuples
        self.assertIsInstance(result_tuples, list)
        self.assertEqual(len(result_tuples), 1)
        
        # Verify each element is a tuple of Circuits
        prep_circuit, povm_circuit = result_tuples[0]
        self.assertIsInstance(prep_circuit, Circuit)
        self.assertIsInstance(povm_circuit, Circuit)
        
        # Verify dof_per_povm is reasonable
        self.assertIsInstance(dof, int)
        self.assertGreater(dof, 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
