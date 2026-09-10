"""Layout checks for the pygsti.tools.edesigntools subpackage.

``pygsti/tools/edesigntools.py`` became the ``pygsti/tools/edesigntools/``
package.  Its public names now live in private implementation modules and are
re-exported by the package ``__init__``, which declares ``__all__``.
``pygsti/tools/__init__.py`` still star-imports from ``edesigntools`` exactly as
before, so every pre-existing import spelling keeps working.  These tests pin
that arrangement.
"""
import importlib

import pygsti.tools as tools
import pygsti.tools.edesigntools as edesigntools

from ..util import BaseCase


class EdesignLayoutTester(BaseCase):
    def test_public_names_are_the_same_objects_everywhere(self):
        for name in edesigntools.__all__:
            with self.subTest(name=name):
                self.assertIs(getattr(tools, name), getattr(edesigntools, name))

    def test_documented_import_spellings_still_work(self):
        # The three spellings used in docs/markdown/.
        from pygsti.tools import edesigntools as edtools
        from pygsti.tools.edesigntools import calculate_fisher_information_matrices_by_L
        import pygsti
        self.assertIs(edtools.calculate_fisher_information_matrix,
                      pygsti.tools.edesigntools.calculate_fisher_information_matrix)
        self.assertIs(calculate_fisher_information_matrices_by_L,
                      edesigntools.calculate_fisher_information_matrices_by_L)

    def test_private_modules_are_importable_and_disjoint(self):
        # Each implementation module owns its names; nothing was duplicated by the split.
        owners = {'_runtime': ['calculate_edesign_estimated_runtime'],
                  '_fisher': ['calculate_fisher_information_per_circuit',
                              'calculate_fisher_information_matrix',
                              'calculate_fisher_information_matrices_by_L'],
        }
        for modname, names in owners.items():
            mod = importlib.import_module('pygsti.tools.edesigntools.' + modname)
            for name in names:
                with self.subTest(module=modname, name=name):
                    self.assertIs(getattr(mod, name), getattr(edesigntools, name))

    def test_star_import_namespace_is_curated(self):
        # edesigntools used to leak math.ceil into pygsti.tools via star-import,
        # because it had no __all__.  Adding one removes that accidental name.
        self.assertFalse(hasattr(tools, 'ceil'))
