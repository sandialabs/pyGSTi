""" Unit tests covering pygsti.baseobjs.parameterized """
from ..util import BaseCase, mock

from pygsti.baseobjs.parameterized import parameterized


class ParameterizedTester(BaseCase):
    def test_decorator_args(self):
        mock_sink = mock.MagicMock()

        @parameterized
        def decorated(fn, a, b):
            mock_sink(a, b)

            def inner(*args, **kwargs):
                return fn(*args, **kwargs)

            return inner

        @decorated(1, 2)
        def func(a, b):
            return a + b

        result = func(3, 4)
        self.assertEqual(result, 7, "unexpected side effect in decorated function")
        mock_sink.assert_called_once_with(1, 2)
