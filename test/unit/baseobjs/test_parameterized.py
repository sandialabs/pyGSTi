""" Unit tests covering pygsti.baseobjs.parameterized """
from ..util import BaseCase

from pygsti.baseobjs.parameterized import parameterized


class ParameterizedTester(BaseCase):
    def test_decorator_args(self):
        decorator_args = None

        @parameterized
        def decorated(fn, a, b):
            nonlocal decorator_args
            decorator_args = (a, b)

            def inner(*args, **kwargs):
                return fn(*args, **kwargs)

            return inner

        @decorated(1, 2)
        def func(a, b):
            return a + b

        result = func(3, 4)
        self.assertEqual(result, 7, "unexpected side effect in decorated function")
        self.assertEqual(decorator_args, (1, 2), "args not passed to decorator")
