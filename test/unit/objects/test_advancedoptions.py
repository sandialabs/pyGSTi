import importlib.util

from pygsti.baseobjs.advancedoptions import AdvancedOptions, GSTAdvancedOptions
from ..util import BaseCase


class AdvancedOptionsTester(BaseCase):
    def test_reimport_executes_module_body(self):
        # The advancedoptions module is imported once when pygsti is first
        # loaded (during test collection), which is *before* coverage begins
        # tracing.  As a result the module-level statements (class bodies,
        # ``valid_keys`` definitions, ``__init_subclass__``) never register as
        # executed.  Re-importing the module here forces its body to run again
        # while measurement is active, so those lines are attributed to the
        # test suite.  We reload a *copy* under a private name so we do not
        # disturb the canonical ``sys.modules`` entry that the rest of pygsti
        # (and the other tests below) rely on.
        modname = 'pygsti.baseobjs.advancedoptions'
        spec = importlib.util.find_spec(modname)
        assert spec is not None and spec.loader is not None
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)

        # The freshly executed module must define the same public classes with
        # the derived frozenset built by __init_subclass__.
        self.assertEqual(fresh.AdvancedOptions._valid_keys_set, frozenset())
        self.assertEqual(fresh.GSTAdvancedOptions._valid_keys_set,
                         frozenset(fresh.GSTAdvancedOptions.valid_keys))

    def test_init_and_validation(self):
        # Test base class
        self.assertEqual(AdvancedOptions._valid_keys_set, frozenset())

        # Test subclass
        self.assertEqual(GSTAdvancedOptions._valid_keys_set, frozenset(GSTAdvancedOptions.valid_keys))
        self.assertGreater(len(GSTAdvancedOptions._valid_keys_set), 0)

    def test_base_class_rejects_all_keys(self):
        # The base AdvancedOptions has an empty valid_keys, so every key is
        # invalid -- exercises __init__ -> update on the base class directly.
        with self.assertRaises(ValueError):
            AdvancedOptions({'anything': 1})

        empty = AdvancedOptions()  # empty dict is fine
        self.assertEqual(len(empty), 0)

    def test_empty_and_none_construction(self):
        # __init__ with None and with an empty dict both produce empty options.
        self.assertEqual(len(GSTAdvancedOptions()), 0)
        self.assertEqual(len(GSTAdvancedOptions(None)), 0)
        self.assertEqual(len(GSTAdvancedOptions({})), 0)

    def test_valid_key_assignment(self):
        # Test constructor
        opts = GSTAdvancedOptions({'objective': 'logl'})
        self.assertEqual(opts['objective'], 'logl')

        # Test __setitem__
        opts['tolerance'] = 1e-6
        self.assertEqual(opts['tolerance'], 1e-6)

        # Test update
        opts.update({'max_iterations': 100})
        self.assertEqual(opts['max_iterations'], 100)

        # update with an empty dict is a no-op that still passes validation
        opts.update({})
        self.assertEqual(len(opts), 3)

    def test_update_returns_none(self):
        # update() is documented to return None.
        opts = GSTAdvancedOptions()
        self.assertIsNone(opts.update({'objective': 'logl'}))

    def test_invalid_key_assignment(self):
        # Test constructor with one invalid key
        with self.assertRaises(ValueError) as cm:
            GSTAdvancedOptions({'invalid_key': 'foo'})
        self.assertIn("'invalid_key'", str(cm.exception))
        self.assertIn("Valid keys are:", str(cm.exception))

        # Test constructor with multiple invalid keys
        with self.assertRaises(ValueError) as cm:
            GSTAdvancedOptions({'invalid_key': 'foo', 'another_invalid': 'bar'})
        self.assertIn("Valid keys are:", str(cm.exception))

        # Test __setitem__
        opts = GSTAdvancedOptions()
        with self.assertRaises(ValueError) as cm:
            opts['invalid_key'] = 'foo'
        self.assertIn("'invalid_key'", str(cm.exception))

        # Test update
        with self.assertRaises(ValueError) as cm:
            opts.update({'another_invalid': 'bar'})
        self.assertIn("'another_invalid'", str(cm.exception))

    def test_error_message_lists_keys_sorted(self):
        # The error message must list the valid keys in sorted order.
        with self.assertRaises(ValueError) as cm:
            GSTAdvancedOptions({'invalid_key': 'foo'})
        msg = str(cm.exception)
        listed = msg.split("Valid keys are: '", 1)[1].rstrip("'")
        keys = listed.split("', '")
        self.assertEqual(keys, sorted(GSTAdvancedOptions.valid_keys))

    def test_non_string_key_is_rejected(self):
        # A non-string key that is not in valid_keys is coerced to str in the
        # message and still raises.
        with self.assertRaises(ValueError) as cm:
            GSTAdvancedOptions({42: 'foo'})
        self.assertIn("'42'", str(cm.exception))

    def test_custom_advanced_options(self):
        class CustomOptions(AdvancedOptions):
            valid_keys = ('custom_key_1', 'custom_key_2')

        # Check that __init_subclass__ worked
        self.assertEqual(CustomOptions._valid_keys_set, frozenset(['custom_key_1', 'custom_key_2']))

        # Test valid assignment
        opts = CustomOptions({'custom_key_1': 'value1'})
        self.assertEqual(opts['custom_key_1'], 'value1')

        # __setitem__ and update on the custom subclass
        opts['custom_key_2'] = 'value2'
        self.assertEqual(opts['custom_key_2'], 'value2')
        opts.update({'custom_key_1': 'updated'})
        self.assertEqual(opts['custom_key_1'], 'updated')

        # Test invalid assignment
        with self.assertRaises(ValueError):
            CustomOptions({'invalid_key': 'foo'})
        with self.assertRaises(ValueError):
            opts['invalid_key'] = 'foo'
