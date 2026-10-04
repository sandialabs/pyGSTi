import warnings

from pygsti.tools import legacytools
from pygsti.tools.exceptions import pyGSTiDeprecationWarning
from ..util import BaseCase


class LegacyTestCase(BaseCase):
    def test_deprecation_warning(self):

        @legacytools.deprecate("Replacement function name")
        def oldFn(x):
            return x

        with self.assertWarns(UserWarning):
            oldFn(5)

    def test_default_message_unchanged(self):

        @legacytools.deprecate("newFn")
        def oldFn(x):
            return x

        with self.assertWarns(pyGSTiDeprecationWarning) as cm:
            self.assertEqual(oldFn(5), 5)
        msg = str(cm.warning)
        self.assertIn("The function oldFn is deprecated", msg)
        self.assertIn("Please use newFn instead.", msg)

    def test_preserves_metadata(self):

        @legacytools.deprecate(message="gone")
        def oldFn(x):
            """Original docstring.

            .. deprecated:: 0.11
            """
            return x

        self.assertEqual(oldFn.__name__, "oldFn")
        self.assertIn("Original docstring.", oldFn.__doc__)
        self.assertIn(".. deprecated:: 0.11", oldFn.__doc__)
        self.assertTrue(hasattr(oldFn, "__wrapped__"))

    def test_custom_message_with_name_substitution(self):

        class Foo:
            @legacytools.deprecate(message="{name} is deprecated; see {issue}.")
            def bar(self):
                return 7

        with self.assertWarns(pyGSTiDeprecationWarning) as cm:
            self.assertEqual(Foo().bar(), 7)
        # {name} -> __qualname__; other braces are left untouched
        self.assertEqual(str(cm.warning), "Foo.bar is deprecated; see {issue}.")

    def test_under_property(self):

        class Foo:
            @property
            @legacytools.deprecate(message="{name} is deprecated.")
            def baz(self):
                """Baz docstring."""
                return 3

        self.assertEqual(Foo.baz.__doc__, "Baz docstring.")
        with self.assertWarns(pyGSTiDeprecationWarning) as cm:
            self.assertEqual(Foo().baz, 3)
        self.assertEqual(str(cm.warning), "Foo.baz is deprecated.")

    def test_warning_attributed_to_caller(self):

        @legacytools.deprecate(message="x")
        def oldFn():
            return None

        with self.assertWarns(pyGSTiDeprecationWarning) as cm:
            oldFn()
        self.assertEqual(cm.filename, __file__)

    def test_warn_deprecated_custom_message(self):
        with self.assertWarns(pyGSTiDeprecationWarning) as cm:
            legacytools.warn_deprecated("SomeClass", message="{name} will be removed in 0.12.")
        self.assertEqual(str(cm.warning), "SomeClass will be removed in 0.12.")
        self.assertEqual(cm.filename, __file__)

    def test_warn_deprecated_replacement_ignored_with_message(self):
        with self.assertWarns(pyGSTiDeprecationWarning) as cm:
            legacytools.warn_deprecated("old", replacement="new", message="custom")
        self.assertEqual(str(cm.warning), "custom")

    def test_can_be_silenced(self):

        @legacytools.deprecate(message="x")
        def oldFn():
            return 1

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.simplefilter("ignore", pyGSTiDeprecationWarning)
            self.assertEqual(oldFn(), 1)
