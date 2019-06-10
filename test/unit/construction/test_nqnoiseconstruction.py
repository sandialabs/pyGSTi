from ..util import BaseCase

from pygsti.construction import nqnoiseconstruction as nc


class KCoverageTester(BaseCase):
    def test_kcoverage(self):
        n = 10  # nqubits
        k = 4  # number of "labels" needing distribution
        rows = nc.get_kcoverage_template(n, k, verbosity=2)
        nc.check_kcoverage_template(rows, n, k, verbosity=1)
