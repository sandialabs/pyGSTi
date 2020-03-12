from ...util import BaseCase

from pygsti.extras import rpe


class RPEConfigTester(BaseCase):
    def test_bad_rpeconfig(self):
        # Note: this doesn't actually raise an exception, it just prints a warning message to stdout
        # XXX what does that mean???
        with self.assertRaises(ValueError):
            rpe.RPEconfig({'alpha': 0, 'epsilon': 1, 'theta': 2})  # need lots more keys...
