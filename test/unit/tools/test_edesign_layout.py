"""Layout checks for the pygsti.tools.edesign subpackage and its edesigntools shim.

The public names moved out of ``pygsti/tools/edesigntools.py`` into the
``pygsti.tools.edesign`` subpackage.  ``edesigntools`` stays behind as a
compatibility alias, and ``pygsti/tools/__init__.py`` still star-imports from
the alias rather than the subpackage, so that the alias sits on the main import
path and cannot silently rot.  These tests pin that arrangement.
"""
import importlib

import pygsti.tools as tools
import pygsti.tools.edesign as edesign
import pygsti.tools.edesigntools as edesigntools

from ..util import BaseCase


class EdesignLayoutTester(BaseCase):
    def test_public_names_are_the_same_objects_everywhere(self):
        for name in edesign.__all__:
            with self.subTest(name=name):
                obj = getattr(edesign, name)
                self.assertIs(getattr(edesigntools, name), obj)
                self.assertIs(getattr(tools, name), obj)

    def test_shim_and_subpackage_export_the_same_names(self):
        self.assertEqual(sorted(edesigntools.__all__), sorted(edesign.__all__))

    def test_documented_import_spellings_still_work(self):
        # The three spellings used in docs/markdown/.
        from pygsti.tools import edesigntools as edtools
        from pygsti.tools.edesigntools import calculate_fisher_information_matrices_by_L
        import pygsti
        self.assertIs(edtools.calculate_fisher_information_matrix,
                      pygsti.tools.edesigntools.calculate_fisher_information_matrix)
        self.assertIs(calculate_fisher_information_matrices_by_L,
                      edesign.calculate_fisher_information_matrices_by_L)

    def test_private_modules_are_importable_and_disjoint(self):
        # Each implementation module owns its names; nothing was duplicated by the split.
        owners = {'_runtime': ['calculate_edesign_estimated_runtime'],
                  '_fisher': ['calculate_fisher_information_per_circuit',
                              'calculate_fisher_information_matrix',
                              'calculate_fisher_information_matrices_by_L'],
                  '_padding': ['pad_edesign_with_idle_lines']}
        for modname, names in owners.items():
            mod = importlib.import_module('pygsti.tools.edesign.' + modname)
            for name in names:
                with self.subTest(module=modname, name=name):
                    self.assertIs(getattr(mod, name), getattr(edesign, name))

    def test_star_import_namespace_is_curated(self):
        # edesigntools used to leak math.ceil into pygsti.tools via star-import,
        # because it had no __all__.  Adding one removes that accidental name.
        self.assertFalse(hasattr(tools, 'ceil'))
