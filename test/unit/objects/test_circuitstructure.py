from ..util import BaseCase

from pygsti.construction import std1Q_XYI as std
import pygsti.objects.circuitstructure as cs


class AbstractCircuitStructureTester(BaseCase):
    def setUp(self):
        self.gss = cs.CircuitStructure()

    def test_xvals(self):
        with self.assertRaises(NotImplementedError):
            self.gss.xvals()

    def test_yvals(self):
        with self.assertRaises(NotImplementedError):
            self.gss.yvals()

    def test_minor_xvals(self):
        with self.assertRaises(NotImplementedError):
            self.gss.minor_xvals()

    def test_minor_yvals(self):
        with self.assertRaises(NotImplementedError):
            self.gss.minor_yvals()

    def test_create_plaquette(self):
        with self.assertRaises(NotImplementedError):
            self.gss.create_plaquette(baseStr="")

    def test_get_plaquette(self):
        with self.assertRaises(NotImplementedError):
            self.gss.get_plaquette(x=0, y=0)

    def test_plaquette_rows_cols(self):
        with self.assertRaises(NotImplementedError):
            self.gss.plaquette_rows_cols()


class LSGermsStructureTester(AbstractCircuitStructureTester):
    def setUp(self):
        self.gss = cs.LsGermsStructure([1, 2, 4], std.germs, std.prepStrs, std.effectStrs)

    def test_truncate(self):
        self.gss.truncate([1, 2])
        # TODO assert correctness

    # TODO coverage for all other member units
    # TODO test cases for other members
