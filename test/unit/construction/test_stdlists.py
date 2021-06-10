import unittest

from pygsti.construction import stdlists, circuitconstruction as cc
from pygsti.modelpacks.legacy import std1Q_XY
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit
from pygsti.datasets import DataSet
from ..util import BaseCase


class StdListTester(BaseCase):
    def setUp(self):
        self.opLabels = [Label('Gx'), Label('Gy')]
        self.strs = cc.to_circuits([('Gx',), ('Gy',), ('Gx', 'Gx')])
        self.germs = cc.to_circuits([('Gx',), ('Gx', 'Gy'), ('Gy', 'Gy')])
        self.testFidPairs = [(0, 1)]
        self.testFidPairsDict = {(Label('Gx'), Label('Gy')): [(0, 0), (0, 1)], (Label('Gy'), Label('Gy')): [(0, 0)]}
        self.ds = DataSet(outcome_labels=['0', '1'])  # a dataset that is missing
        self.ds.add_count_dict(('Gx',), {'0': 10, '1': 90})     # almost all our strings...
        self.ds.done_adding_data()

    def test_lsgst_lists_structs(self):
        maxLens = [1, 2]
        lsgstLists = stdlists.create_lsgst_circuit_lists(
            std1Q_XY.target_model(), self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
            trunc_scheme="whole germ powers")  # also try a Model as first arg
        self.assertEqual(lsgstLists[-1][26]._str, 'GxGx(Gx)^2GxGx')  # ensure that (.)^2 appears in string (*not* expanded)

        lsgstLists2 = stdlists.create_lsgst_circuit_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
            trunc_scheme="truncated germ powers")
        self.assertEqual(set(lsgstLists[-1]), set(lsgstLists2[-1]))

        lsgstLists3 = stdlists.create_lsgst_circuit_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
            trunc_scheme="length as exponent")
        lsgstStructs3 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
            trunc_scheme="length as exponent")
        self.assertEqual(set(lsgstLists3[-1]), set(lsgstStructs3[-1]))

        maxLens = [1, 2]
        lsgstLists4 = stdlists.create_lsgst_circuit_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
            trunc_scheme="whole germ powers", nest=False)
        lsgstStructs4 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
            trunc_scheme="whole germ powers", nest=False)
        self.assertEqual(set(lsgstLists4[-1]), set(lsgstStructs4[-1]))

        lsgstLists5 = stdlists.create_lsgst_circuit_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=self.testFidPairs,
            trunc_scheme="whole germ powers")
        lsgstStructs5 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=self.testFidPairs,
            trunc_scheme="whole germ powers")
        self.assertEqual(set(lsgstLists5[-1]), set(lsgstStructs5[-1]))

        lsgstLists6 = stdlists.create_lsgst_circuit_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=self.testFidPairsDict,
            trunc_scheme="whole germ powers")
        lsgstStructs6 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=self.testFidPairsDict,
            trunc_scheme="whole germ powers")
        self.assertEqual(set(lsgstLists6[-1]), set(lsgstStructs6[-1]))

        lsgstLists7 = stdlists.create_lsgst_circuit_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
            trunc_scheme="whole germ powers", keep_fraction=0.5, keep_seed=1234)
        lsgstStructs7 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
            trunc_scheme="whole germ powers", keep_fraction=0.5, keep_seed=1234)
        self.assertEqual(set(lsgstLists7[-1]), set(lsgstStructs7[-1]))

        lsgstLists8 = stdlists.create_lsgst_circuit_lists(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=self.testFidPairs,
            trunc_scheme="whole germ powers", keep_fraction=0.7, keep_seed=1234)
        lsgstStructs8 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=self.testFidPairs,
            trunc_scheme="whole germ powers", keep_fraction=0.7, keep_seed=1234)
        self.assertEqual(set(lsgstLists8[-1]), set(lsgstStructs8[-1]))

        # empty max-lengths ==> no output
        lsgstStructs9 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, [], include_lgst=False)
        self.assertEqual(len(lsgstStructs9), 0)

        # checks against datasets
        lgst_strings = cc.create_lgst_circuits(self.strs, self.strs, self.opLabels)
        lsgstStructs10 = stdlists.make_lsgst_structs(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, dscheck=self.ds,
            action_if_missing="drop", verbosity=4)
        self.assertEqual([Circuit(('Gx',))], list(lsgstStructs10[-1]))

    def test_lsgst_experiment_list(self):
        maxLens = [1, 2]
        lsgstExpList = stdlists.create_lsgst_circuits(
            self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
            trunc_scheme="whole germ powers")
        lsgstExpListb = stdlists.create_lsgst_circuits(
            std1Q_XY.target_model(), self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
            trunc_scheme="whole germ powers")  # with Model as first arg
        self.assertEqual(set(lsgstExpList), set(lsgstExpListb))

    def test_lsgst_lists_structs_raises_on_bad_scheme(self):
        maxLens = [1, 2]
        with self.assertRaises(ValueError):
            stdlists.create_lsgst_circuit_lists(
                self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
                trunc_scheme="foobar")
        
        # make_lsgst_structs deprecated
        # with self.assertRaises(ValueError):
        #     stdlists.make_lsgst_structs(
        #         self.opLabels, self.strs, self.strs, self.germs, maxLens, fid_pairs=None,
        #         trunc_scheme="foobar")
        # with self.assertRaises(ValueError):
        #     stdlists.make_lsgst_structs(
        #         self.opLabels, self.strs, self.strs, self.germs, maxLens, dscheck=self.ds,
        #         action_if_missing="foobar")

    def test_lsgst_lists_structs_raises_on_missing_ds_sequence(self):
        with self.assertRaises(ValueError):
            stdlists.make_lsgst_structs(
                self.opLabels, self.strs, self.strs, self.germs, [1, 2], dscheck=self.ds)  # missing sequences

    @unittest.skip("Skipping due to deprecation of eLGST")
    def test_elgst_lists_structs(self):
        # ELGST
        maxLens = [1, 2]
        elgstLists = stdlists.create_elgst_lists(
            self.opLabels, self.germs, maxLens, trunc_scheme="whole germ powers")

        maxLens = [1, 2]
        elgstLists2 = stdlists.create_elgst_lists(
            self.opLabels, self.germs, maxLens, trunc_scheme="whole germ powers",
            nest=False, include_lgst=False)
        elgstLists2b = stdlists.create_elgst_lists(
            std1Q_XY.target_model(), self.germs, maxLens, trunc_scheme="whole germ powers",
            nest=False, include_lgst=False)  # with a Model as first arg

    @unittest.skip("Skipping due to deprecation of eLGST")
    def test_elgst_experiment_list(self):
        elgstExpLists = stdlists.create_elgst_experiment_list(
            self.opLabels, self.germs, [1, 2], trunc_scheme="whole germ powers")

    @unittest.skip("Skipping due to deprecation of eLGST")
    def test_elgst_lists_structs_raises_on_bad_scheme(self):
        with self.assertRaises(ValueError):
            stdlists.create_elgst_lists(
                self.opLabels, self.germs, [1, 2], trunc_scheme="foobar")
