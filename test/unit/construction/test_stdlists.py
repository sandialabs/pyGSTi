from ..util import BaseCase

from pygsti.objects import Circuit, Label, DataSet
from pygsti.modelpacks.legacy import std1Q_XY
from pygsti.construction import stdlists, circuitconstruction as cc


class StdListTester(BaseCase):
    def setUp(self):
        self.opLabels = [Label('Gx'), Label('Gy')]
        self.strs = cc.circuit_list([('Gx',), ('Gy',), ('Gx', 'Gx')])
        self.germs = cc.circuit_list([('Gx', 'Gy'), ('Gy', 'Gy')])
        self.testFidPairs = [(0, 1)]
        self.testFidPairsDict = {(Label('Gx'), Label('Gy')): [(0, 0), (0, 1)], (Label('Gy'), Label('Gy')): [(0, 0)]}
        self.ds = DataSet(outcomeLabels=['0', '1'])  # a dataset that is missing
        self.ds.add_count_dict(('Gx',), {'0': 10, '1': 90})     # almost all our strings...
        self.ds.done_adding_data()

    def test_lsgst_lists_structs(self):
        maxLens = [1, 2]
        lsgstLists = stdlists.make_lsgst_lists(
            std1Q_XY.target_model(), self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers")  # also try a Model as first arg
        lsgstStructs = stdlists.make_lsgst_structs(
            std1Q_XY.target_model(), self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers")  # also try a Model as first arg
        self.assertEqual(set(lsgstLists[-1]), set(lsgstStructs[-1].allstrs))

        lsgstLists2 = stdlists.make_lsgst_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="truncated germ powers")
        lsgstStructs2 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="truncated germ powers")
        self.assertEqual(set(lsgstLists2[-1]), set(lsgstStructs2[-1].allstrs))

        lsgstLists3 = stdlists.make_lsgst_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="length as exponent")
        lsgstStructs3 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="length as exponent")
        self.assertEqual(set(lsgstLists3[-1]), set(lsgstStructs3[-1].allstrs))

        maxLens = [1, 2]
        lsgstLists4 = stdlists.make_lsgst_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers", nest=False)
        lsgstStructs4 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers", nest=False)
        self.assertEqual(set(lsgstLists4[-1]), set(lsgstStructs4[-1].allstrs))

        lsgstLists5 = stdlists.make_lsgst_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=self.testFidPairs,
            truncScheme="whole germ powers")
        lsgstStructs5 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=self.testFidPairs,
            truncScheme="whole germ powers")
        self.assertEqual(set(lsgstLists5[-1]), set(lsgstStructs5[-1].allstrs))

        lsgstLists6 = stdlists.make_lsgst_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=self.testFidPairsDict,
            truncScheme="whole germ powers")
        lsgstStructs6 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=self.testFidPairsDict,
            truncScheme="whole germ powers")
        self.assertEqual(set(lsgstLists6[-1]), set(lsgstStructs6[-1].allstrs))

        lsgstLists7 = stdlists.make_lsgst_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers", keepFraction=0.5, keepSeed=1234)
        lsgstStructs7 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers", keepFraction=0.5, keepSeed=1234)
        self.assertEqual(set(lsgstLists7[-1]), set(lsgstStructs7[-1].allstrs))

        lsgstLists8 = stdlists.make_lsgst_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=self.testFidPairs,
            truncScheme="whole germ powers", keepFraction=0.7, keepSeed=1234)
        lsgstStructs8 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=self.testFidPairs,
            truncScheme="whole germ powers", keepFraction=0.7, keepSeed=1234)
        self.assertEqual(set(lsgstLists8[-1]), set(lsgstStructs8[-1].allstrs))
        # TODO assert correctness

        # empty max-lengths ==> no output
        lsgstStructs9 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, [])
        self.assertEqual(len(lsgstStructs9), 0)

        # checks against datasets
        lgst_strings = cc.list_lgst_circuits(self.strs, self.strs, self.opLabels)
        lsgstStructs10 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, dscheck=self.ds,
            actionIfMissing="drop", verbosity=4)
        self.assertEqual([Circuit(('Gx',))], lsgstStructs10[-1].allstrs)

    def test_lsgst_experiment_list(self):
        maxLens = [1, 2]
        lsgstExpList = stdlists.make_lsgst_experiment_list(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers")
        lsgstExpListb = stdlists.make_lsgst_experiment_list(
            std1Q_XY.target_model(), self.strs, self.strs, self.germs, maxLens, fidPairs=None,
            truncScheme="whole germ powers")  # with Model as first arg
        # TODO assert correctness

    def test_lsgst_lists_structs_raises_on_bad_scheme(self):
        maxLens = [1, 2]
        with self.assertRaises(ValueError):
            stdlists.make_lsgst_lists(
                self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
                truncScheme="foobar")
        with self.assertRaises(ValueError):
            stdlists.make_lsgst_structs(
                self.opLabels, self.strs, self.strs, self.germs, maxLens, fidPairs=None,
                truncScheme="foobar")
        with self.assertRaises(ValueError):
            stdlists.make_lsgst_structs(
                self.opLabels, self.strs, self.strs, self.germs, maxLens, dscheck=self.ds,
                actionIfMissing="foobar")

    def test_lsgst_lists_structs_raises_on_missing_ds_sequence(self):
        with self.assertRaises(ValueError):
            stdlists.make_lsgst_structs(
                self.opLabels, self.strs, self.strs, self.germs, [1, 2], dscheck=self.ds)  # missing sequences

    def test_elgst_lists_structs(self):
        # ELGST
        maxLens = [1, 2]
        elgstLists = stdlists.make_elgst_lists(
            self.opLabels, self.germs, maxLens, truncScheme="whole germ powers")

        maxLens = [1, 2]
        elgstLists2 = stdlists.make_elgst_lists(
            self.opLabels, self.germs, maxLens, truncScheme="whole germ powers",
            nest=False, includeLGST=False)
        elgstLists2b = stdlists.make_elgst_lists(
            std1Q_XY.target_model(), self.germs, maxLens, truncScheme="whole germ powers",
            nest=False, includeLGST=False)  # with a Model as first arg
        # TODO assert correctness

    def test_elgst_experiment_list(self):
        elgstExpLists = stdlists.make_elgst_experiment_list(
            self.opLabels, self.germs, [1, 2], truncScheme="whole germ powers")
        # TODO assert correctness

    def test_elgst_lists_structs_raises_on_bad_scheme(self):
        with self.assertRaises(ValueError):
            stdlists.make_elgst_lists(
                self.opLabels, self.germs, [1, 2], truncScheme="foobar")
