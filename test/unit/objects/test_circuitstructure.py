from ..util import BaseCase

from pygsti.modelpacks.legacy import std1Q_XYI as std
import pygsti.objects.circuitstructure as cs


class AbstractCircuitStructureTester(BaseCase):
    # XXX is testing an abstract base class really useful?  EGN: I guess it tests the interface.
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
            self.gss.create_plaquette(base_str="")

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

    def test_xvals(self):
        self.skipTest("TODO")

    def test_yvals(self):
        self.skipTest("TODO")

    def test_minor_xvals(self):
        self.skipTest("TODO")

    def test_minor_yvals(self):
        self.skipTest("TODO")

    def test_create_plaquette(self):
        self.skipTest("TODO")

    def test_get_plaquette(self):
        self.skipTest("TODO")

    def test_plaquette_rows_cols(self):
        self.skipTest("TODO")
