from __future__ import annotations
import inspect
import types
from pygsti.report.workspace import Workspace
from ..util import BaseCase


class NonExistentType:
    pass


class DummyPlainClass:
    def __init__(self, ws, a, b, c=3):
        """Plain class docstring."""
        self.ws = ws
        self.a = a
        self.b = b
        self.c = c


class DummyVarArgsClass:
    """Var-args class docstring."""
    def __init__(self, ws, fn, *args):
        self.ws = ws
        self.fn = fn
        self.args = args


class DummyAutoDisplayClass:
    """Auto-display class docstring."""
    def __init__(self, ws, a):
        self.ws = ws
        self.a = a
        self.displayed = False

    def display(self):
        self.displayed = True


class DummyAnnotatedClass:
    def __init__(self, ws, a: int, b: NonExistentType, c: str = 'x'):
        """Annotated class docstring."""
        self.ws = ws
        self.a = a
        self.b = b
        self.c = c


class WorkspaceMakeFactoryTester(BaseCase):
    def setUp(self):
        super().setUp()
        self.dummy_ws = types.SimpleNamespace(tag="dummy")

    def test_makefactory_plain(self):
        # Test basic _makefactory with standard positional and default args
        factory = Workspace._makefactory(self.dummy_ws, DummyPlainClass, autodisplay=False)
        
        # Verify function metadata
        self.assertEqual(factory.__name__, '__init__')
        self.assertEqual(factory.__doc__, 'Plain class docstring.')
        
        # Verify call forwarding
        inst = factory(10, 20)
        self.assertIsInstance(inst, DummyPlainClass)
        self.assertIs(inst.ws, self.dummy_ws)
        self.assertEqual(inst.a, 10)
        self.assertEqual(inst.b, 20)
        self.assertEqual(inst.c, 3)

        # Verify call forwarding with overriding defaults
        inst = factory(10, 20, c=30)
        self.assertEqual(inst.c, 30)

        # Verify signature
        sig = inspect.signature(factory)
        # Note: self and ws should be stripped, a, b, and c=3 should remain
        self.assertEqual(list(sig.parameters.keys()), ['a', 'b', 'c'])
        self.assertEqual(sig.parameters['c'].default, 3)

    def test_makefactory_varargs(self):
        # Test basic _makefactory with *args
        factory = Workspace._makefactory(self.dummy_ws, DummyVarArgsClass, autodisplay=False)
        
        inst = factory('my_fn', 1, 2, 3)
        self.assertIsInstance(inst, DummyVarArgsClass)
        self.assertIs(inst.ws, self.dummy_ws)
        self.assertEqual(inst.fn, 'my_fn')
        self.assertEqual(inst.args, (1, 2, 3))

    def test_makefactory_autodisplay(self):
        # Test _makefactory with autodisplay=True
        factory = Workspace._makefactory(self.dummy_ws, DummyAutoDisplayClass, autodisplay=True)
        
        inst = factory(100)
        self.assertIsInstance(inst, DummyAutoDisplayClass)
        self.assertTrue(inst.displayed)
        self.assertEqual(inst.a, 100)


class WorkspaceMakeFactoryAnnotationsTester(BaseCase):
    def setUp(self):
        super().setUp()
        self.dummy_ws = types.SimpleNamespace(tag="dummy")

    def test_makefactory_annotated(self):
        # This will fail on unpatched codebase because of annotations in exec string
        factory = Workspace._makefactory(self.dummy_ws, DummyAnnotatedClass, autodisplay=False)

        # Add a breakpoint here and check print(inspect.getsource(factory)) if the following line
        # is failing due to a syntax error in the call.
        inst = factory(1, 'hello', c='world')
        self.assertIsInstance(inst, DummyAnnotatedClass)
        self.assertIs(inst.ws, self.dummy_ws)
        self.assertEqual(inst.a, 1)
        self.assertEqual(inst.b, 'hello')
        self.assertEqual(inst.c, 'world')

        # Verify that annotations are preserved in the signature of the generated function
        sig = inspect.signature(factory)
        self.assertEqual(sig.parameters['a'].annotation, 'int')
        self.assertEqual(sig.parameters['b'].annotation, 'NonExistentType')
        self.assertEqual(sig.parameters['c'].annotation, 'str')

    def test_full_workspace_construction_smoke(self):
        # This smoke test will fail on unpatched codebase during Workspace.__init__ -> _register_components
        ws = Workspace()
        self.assertIsNotNone(ws)
        # Verify some standard registered factories are accessible
        self.assertTrue(callable(ws.CircuitTable))
        self.assertTrue(callable(ws.SpamTable))
