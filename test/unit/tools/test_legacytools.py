from pygsti.tools import legacytools
from ..util import BaseCase


class LegacyTestCase(BaseCase):
    def test_deprecation_warning(self):

        @legacytools.deprecate("Replacement function name")
        def oldFn(x):
            return x

        with self.assertWarns(UserWarning):
            oldFn(5)
