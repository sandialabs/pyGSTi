import inspect
import sys
import warnings

import pygsti.tools.errorgenpropagation as _eprop
import pygsti.tools.errorgenpropagation.parallelutils as _eprop_par
import pygsti.tools.errgenpolytools as _shim_poly
import pygsti.tools.errgenproptools as _shim_prop
import pygsti.tools.fasterrgencalc as _shim_fast
from pygsti.tools.exceptions import pyGSTiDeprecationWarning

from ...util import BaseCase


class ErrorGenPropagationLayoutTester(BaseCase):

    def test_submodules_and_all_exports(self):
        expected_submodules = [
            'errgentypes', 'fasterrgencalc', 'algebra', 'parallelutils',
            'sensitivities', 'expansions', 'corrections', 'numerical', 'polynomials',
        ]
        # 1. All submodules exist as attributes on the package
        for name in expected_submodules:
            self.assertTrue(hasattr(_eprop, name), f"Missing submodule {name} on errorgenpropagation")

        # 2. Check non-overlapping __all__ sets across submodules
        all_sets = {}
        for name in expected_submodules:
            mod = getattr(_eprop, name)
            if mod is not None and hasattr(mod, '__all__'):
                all_sets[name] = set(mod.__all__)

        for i, name1 in enumerate(all_sets):
            for name2 in list(all_sets)[i + 1:]:
                overlap = all_sets[name1] & all_sets[name2]
                self.assertEqual(overlap, set(), f"Overlap in __all__ between {name1} and {name2}: {overlap}")

        # 3. Union of public exports plus submodule names equals package __all__
        public_union = set(expected_submodules)
        for mod_all in all_sets.values():
            public_union.update(name for name in mod_all if not name.startswith('_'))

        pkg_all = set(_eprop.__all__)
        self.assertEqual(public_union, pkg_all)

    def test_shared_mutable_state_identity(self):
        self.assertIs(
            _eprop._MANUAL_PARALLEL_OVERRIDES,
            _eprop_par._MANUAL_PARALLEL_OVERRIDES,
            "_MANUAL_PARALLEL_OVERRIDES is not identical across package and parallelutils",
        )
        self.assertIs(
            _eprop._ACTIVE_PARALLEL_THRESHOLDS,
            _eprop_par._ACTIVE_PARALLEL_THRESHOLDS,
            "_ACTIVE_PARALLEL_THRESHOLDS is not identical across package and parallelutils",
        )

    def test_acyclic_runtime_import_numerical(self):
        import pygsti.tools.errorgenpropagation.numerical as num_mod

        self.assertNotIn(
            'ErrorGeneratorPropagator',
            num_mod.__dict__,
            "numerical should not import ErrorGeneratorPropagator at module level",
        )
        source = inspect.getsource(num_mod)
        self.assertNotIn(
            'import ErrorGeneratorPropagator',
            source,
            "numerical source should not import ErrorGeneratorPropagator",
        )

    def test_pep562_deprecation_shims_once_per_session(self):
        shims = [
            ('errgenproptools', _shim_prop, 'error_generator_commutator'),
            ('errgenpolytools', _shim_poly, 'magnus_symbolic_polynomial'),
            ('fasterrgencalc', _shim_fast, 'fast_bel_product_str'),
        ]
        for name, shim, attr in shims:
            orig_warned = shim._WARNED
            try:
                shim._WARNED = False
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always', pyGSTiDeprecationWarning)
                    try:
                        _ = getattr(shim, attr)
                    except ImportError:
                        pass
                    first_warns = [item for item in w if issubclass(item.category, pyGSTiDeprecationWarning)]
                    self.assertEqual(len(first_warns), 1, f"{name} did not emit deprecation warning on first access")
                    self.assertTrue(shim._WARNED, f"{name}._WARNED not set to True after first access")

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always', pyGSTiDeprecationWarning)
                    try:
                        _ = getattr(shim, attr)
                    except ImportError:
                        pass
                    second_warns = [item for item in w if issubclass(item.category, pyGSTiDeprecationWarning)]
                    self.assertEqual(len(second_warns), 0, f"{name} emitted unexpected deprecation warning on second access")
            finally:
                shim._WARNED = orig_warned
