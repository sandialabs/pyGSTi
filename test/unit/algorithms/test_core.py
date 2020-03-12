import numpy as np

from ..util import BaseCase
from . import fixtures

import pygsti.construction as pc
from pygsti.objects import Circuit, Label
from pygsti.algorithms import core


class CoreStdData(object):
    def setUp(self):
        super(CoreStdData, self).setUp()
        self.ds = fixtures.ds.copy()
        self.model = fixtures.model.copy()
        self.fiducials = fixtures.fiducials


class CoreFuncTester(CoreStdData, BaseCase):
    def test_gram_rank_and_evals(self):
        rank, evals, target_evals = core.gram_rank_and_evals(self.ds, self.fiducials, self.fiducials, self.model)
        # TODO assert correctness

    def test_gram_rank_and_evals_raises_on_no_target(self):
        # XXX is this neccessary?  EGN: probably not
        with self.assertRaises(ValueError):
            core.gram_rank_and_evals(self.ds, self.fiducials, self.fiducials, None)

    def test_find_closest_unitary_opmx_raises_on_multi_qubit(self):
        with self.assertRaises(ValueError):
            core.find_closest_unitary_opmx(np.identity(16, 'd'))


class CoreLGSTTester(CoreStdData, BaseCase):
    def setUp(self):
        super(CoreLGSTTester, self).setUp()
        self.datagen_gateset = fixtures.datagen_gateset
        self.lgstStrings = fixtures.lgstStrings

    def test_do_lgst(self):
        mdl_lgst = core.do_lgst(
            self.ds, self.fiducials, self.fiducials, self.model,
            svdTruncateTo=4
        )
        # TODO assert correctness

        # XXX is this neccessary? EGN: tests higher verbosity printing.
        mdl_lgst_2 = core.do_lgst(
            self.ds, self.fiducials, self.fiducials, self.model,
            svdTruncateTo=4, verbosity=10
        )
        # TODO assert correctness

        self.assertAlmostEqual(mdl_lgst.frobeniusdist(mdl_lgst_2), 0)

    def test_do_lgst_raises_on_no_target(self):
        # XXX is this neccessary?
        with self.assertRaises(ValueError):
            core.do_lgst(
                self.ds, self.fiducials, self.fiducials, None, svdTruncateTo=4
            )

    def test_do_lgst_raises_on_no_spam_dict(self):
        with self.assertRaises(ValueError):
            core.do_lgst(
                self.ds, self.fiducials, self.fiducials, None,
                opLabels=list(self.model.operations.keys()), svdTruncateTo=4
            )

    def test_do_lgst_raises_on_bad_fiducials(self):
        bad_fids = pc.circuit_list([('Gx',), ('Gx',), ('Gx',), ('Gx',)])
        with self.assertRaises(ValueError):
            core.do_lgst(
                self.ds, bad_fids, bad_fids, self.model, svdTruncateTo=4
            )  # bad fiducials (rank deficient)

    def test_do_lgst_raises_on_incomplete_ab_matrix(self):
        incomplete_strings = self.lgstStrings[5:]  # drop first 5 strings...
        bad_ds = pc.generate_fake_data(
            self.datagen_gateset, incomplete_strings,
            n_samples=10, sample_error='none')
        with self.assertRaises(KeyError):
            core.do_lgst(
                bad_ds, self.fiducials, self.fiducials, self.model,
                svdTruncateTo=4
            )

    def test_do_lgst_raises_on_incomplete_x_matrix(self):
        incomplete_strings = self.lgstStrings[:-5]  # drop last 5 strings...
        bad_ds = pc.generate_fake_data(
            self.datagen_gateset, incomplete_strings,
            n_samples=10, sample_error='none')
        with self.assertRaises(KeyError):
            core.do_lgst(
                bad_ds, self.fiducials, self.fiducials, self.model,
                svdTruncateTo=4
            )


class CoreELGSTTester(CoreStdData, BaseCase):
    def setUp(self):
        super(CoreELGSTTester, self).setUp()
        self.mdl_clgst = fixtures.mdl_clgst.copy()
        self.elgstStrings = fixtures.elgstStrings

    def test_do_exlgst(self):
        err_vec, model = core.do_exlgst(
            self.ds, self.mdl_clgst, self.elgstStrings[0], self.fiducials,
            self.fiducials, self.model, regularizeFactor=1e-3, svdTruncateTo=4
        )
        model._check_paramvec()
        # TODO assert correctness

        # XXX is this neccesary? (verbosity increase)
        err_vec_2, model_2 = core.do_exlgst(
            self.ds, self.mdl_clgst, self.elgstStrings[0], self.fiducials,
            self.fiducials, self.model, regularizeFactor=1e-3, svdTruncateTo=4,
            verbosity=10
        )
        model_2._check_paramvec()
        # TODO assert correctness

        self.assertAlmostEqual(model.frobeniusdist(model_2), 0)

    def test_do_iterative_exlgst(self):
        mdl_exlgst = core.do_iterative_exlgst(
            self.ds, self.mdl_clgst, self.fiducials, self.fiducials,
            self.elgstStrings, targetModel=self.model, svdTruncateTo=4
        )
        # TODO assert correctness

        # XXX this doesn't really look useful...
        mdl_exlgst_2 = core.do_iterative_exlgst(
            self.ds, self.mdl_clgst, self.fiducials, self.fiducials,
            self.elgstStrings, targetModel=self.model, svdTruncateTo=4,
            verbosity=10
        )
        # TODO assert correctness
        self.assertAlmostEqual(mdl_exlgst.frobeniusdist(mdl_exlgst_2), 0)

        # XXX this doesn't look useful either
        all_min_errs, all_gs_exlgst_tups = core.do_iterative_exlgst(
            self.ds, self.mdl_clgst, self.fiducials, self.fiducials,
            [[cir.tup for cir in gsList] for gsList in self.elgstStrings],
            targetModel=self.model, svdTruncateTo=4,
            returnAll=True, returnErrorVec=True
        )
        # TODO assert correctness
        self.assertAlmostEqual(mdl_exlgst.frobeniusdist(all_gs_exlgst_tups[-1]), 0)

    def test_do_iterative_exlgst_with_regularize_factor(self):
        mdl_exlgst = core.do_iterative_exlgst(
            self.ds, self.mdl_clgst, self.fiducials, self.fiducials,
            self.elgstStrings, targetModel=self.model, svdTruncateTo=4,
            regularizeFactor=10
        )
        # TODO assert correctness

    def test_do_iterative_exlgst_check_jacobian(self):
        mdl_exlgst = core.do_iterative_exlgst(
            self.ds, self.mdl_clgst, self.fiducials, self.fiducials,
            self.elgstStrings, targetModel=self.model, svdTruncateTo=4,
            check_jacobian=True
        )
        # TODO assert correctness


class CoreMC2GSTTester(CoreStdData, BaseCase):
    def setUp(self):
        super(CoreMC2GSTTester, self).setUp()
        self.mdl_clgst = fixtures.mdl_clgst.copy()
        self.lsgstStrings = fixtures.lsgstStrings

    def test_do_mc2gst(self):
        mdl_lsgst = core.do_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings[0],
            minProbClipForWeighting=1e-4, probClipInterval=(-1e6, 1e6)
        )
        # TODO assert correctness

    def test_do_mc2gst_regularize_factor(self):
        mdl_lsgst = core.do_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings[0],
            minProbClipForWeighting=1e-4, probClipInterval=(-1e6, 1e6),
            regularizeFactor=1e-3
        )
        # TODO assert correctness

    def test_do_mc2gst_CPTP_penalty_factor(self):
        mdl_lsgst = core.do_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings[0],
            minProbClipForWeighting=1e-4, probClipInterval=(-1e6, 1e6),
            cptp_penalty_factor=1.0
        )
        # TODO assert correctness

    def test_do_mc2gst_SPAM_penalty_factor(self):
        mdl_lsgst = core.do_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings[0],
            minProbClipForWeighting=1e-4, probClipInterval=(-1e6, 1e6),
            spam_penalty_factor=1.0
        )
        # TODO assert correctness

    def test_do_mc2gst_CPTP_SPAM_penalty_factor(self):
        mdl_lsgst = core.do_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings[0],
            minProbClipForWeighting=1e-4, probClipInterval=(-1e6, 1e6),
            cptp_penalty_factor=1.0, spam_penalty_factor=1.0
        )
        # TODO assert correctness

    def test_do_mc2gst_alias_model(self):
        aliased_list = [
            Circuit([
                (x if x != Label("Gx") else Label("GA1")) for x in mdl
            ]) for mdl in self.lsgstStrings[0]
        ]
        aliased_model = self.mdl_clgst.copy()
        aliased_model.operations['GA1'] = self.mdl_clgst.operations['Gx']
        aliased_model.operations.pop('Gx')

        mdl_lsgst = core.do_mc2gst(
            self.ds, aliased_model, aliased_list, minProbClipForWeighting=1e-4,
            probClipInterval=(-1e6, 1e6),
            opLabelAliases={Label('GA1'): Circuit(['Gx'])}
        )
        # TODO assert correctness

    def test_do_iterative_mc2gst(self):
        mdl_lsgst = core.do_iterative_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6, 1e6)
        )
        # TODO assert correctness

        # XXX are these useful? (verbosity test)
        mdl_lsgst_2 = core.do_iterative_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings, verbosity=10,
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6, 1e6)
        )
        # TODO assert correctness
        self.assertAlmostEqual(mdl_lsgst.frobeniusdist(mdl_lsgst_2), 0)

        all_min_errs, all_gs_lsgst_tups = core.do_iterative_mc2gst(
            self.ds, self.mdl_clgst,
            [[mdl.tup for mdl in gsList] for gsList in self.lsgstStrings],
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6, 1e6),
            returnAll=True, returnErrorVec=True
        )
        # TODO assert correctness
        self.assertAlmostEqual(mdl_lsgst.frobeniusdist(all_gs_lsgst_tups[-1]), 0)

    def test_do_iterative_mc2gst_regularize_factor(self):
        mdl_lsgst = core.do_iterative_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6, 1e6),
            regularizeFactor=10
        )
        # TODO assert correctness

    def test_do_iterative_mc2gst_check_jacobian(self):
        mdl_lsgst = core.do_iterative_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6, 1e6),
            check_jacobian=True
        )
        # TODO assert correctness

    def test_do_iterative_mc2gst_use_freq_weighted_chi2(self):
        mdl_lsgst = core.do_iterative_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6, 1e6),
            useFreqWeightedChiSq=True
        )
        # TODO assert correctness

    def test_do_iterative_mc2gst_circuit_set_labels(self):
        mdl_lsgst = core.do_iterative_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6, 1e6),
            circuitSetLabels=["Set1", "Set2", "Set3"]
        )
        # TODO assert correctness

    def test_do_iterative_mc2gst_circuit_weights_dict(self):
        mdl_lsgst = core.do_iterative_mc2gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            minProbClipForWeighting=1e-6, probClipInterval=(-1e6, 1e6),
            circuitWeightsDict={('Gx',): 2.0}
        )
        # TODO assert correctness

    def test_do_mc2gst_raises_on_out_of_memory(self):
        with self.assertRaises(MemoryError):
            core.do_mc2gst(
                self.ds, self.mdl_clgst, self.lsgstStrings[0],
                minProbClipForWeighting=1e-4, probClipInterval=(-1e6, 1e6),
                memLimit=0
            )

    def test_do_mc2gst_raises_on_conflicting_spec(self):
        with self.assertRaises(AssertionError):
            core.do_mc2gst(
                self.ds, self.mdl_clgst, self.lsgstStrings[0],
                minProbClipForWeighting=1e-4, probClipInterval=(-1e6, 1e6),
                regularizeFactor=1e-3, cptp_penalty_factor=1.0
            )


# XXX shouldn't this code be reused?
class CoreMLGSTTester(CoreStdData, BaseCase):
    def setUp(self):
        super(CoreMLGSTTester, self).setUp()
        self.mdl_clgst = fixtures.mdl_clgst.copy()
        self.lsgstStrings = fixtures.lsgstStrings

    def test_do_mlgst(self):
        model = core.do_mlgst(
            self.ds, self.mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
            probClipInterval=(-1e2, 1e2)
        )
        # TODO assert correctness

    def test_do_mlgst_CPTP_penalty_factor(self):
        model = core.do_mlgst(
            self.ds, self.mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
            probClipInterval=(-1e2, 1e2), cptp_penalty_factor=1.0
        )
        # TODO assert correctness

    def test_do_mlgst_SPAM_penalty_factor(self):
        model = core.do_mlgst(
            self.ds, self.mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
            probClipInterval=(-1e2, 1e2), spam_penalty_factor=1.0
        )
        # TODO assert correctness

    def test_do_mlgst_CPTP_SPAM_penalty_factor(self):
        # this test often gives an assetion error "finite Jacobian has
        # inf norm!" on Travis CI Python 3 case. Just ignore for now.
        # FUTURE: see what we can do in custom LM about scaling large
        # jacobians...
        self.skipTest("Ignore for now.")
        model = core.do_mlgst(
            self.ds, self.mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
            probClipInterval=(-1e2, 1e2), cptp_penalty_factor=1.0,
            spam_penalty_factor=1.0
        )
        # TODO assert correctness

    def test_do_mlgst_alias_model(self):
        aliased_list = [
            Circuit([
                (x if x != Label("Gx") else Label("GA1")) for x in mdl
            ]) for mdl in self.lsgstStrings[0]
        ]
        aliased_model = self.mdl_clgst.copy()
        aliased_model.operations['GA1'] = self.mdl_clgst.operations['Gx']
        aliased_model.operations.pop('Gx')

        model = core.do_mlgst(
            self.ds, aliased_model, aliased_list, minProbClip=1e-4,
            probClipInterval=(-1e6, 1e6),
            opLabelAliases={Label('GA1'): Circuit(['Gx'])}
        )
        # TODO assert correctness

    def test_do_iterative_mlgst(self):
        model = core.do_iterative_mlgst(
            self.ds, self.mdl_clgst, self.lsgstStrings, minProbClip=1e-4,
            probClipInterval=(-1e2, 1e2)
        )

    # # XXX This probably shouldn't exist?
    # # From the core.do_iterative_mlgst docstring:
    # #   check : boolean, optional
    # #       If True, perform extra checks within code to verify correctness.  Used
    # #       for testing, and runs much slower when True.
    # def test_do_iterative_mlgst_with_check(self):
    #     model = core.do_iterative_mlgst(
    #         self.ds, self.mdl_clgst, self.lsgstStrings, minProbClip=1e-4,
    #         probClipInterval=(-1e2, 1e2), check=True
    #     )

    def test_do_iterative_mlgst_circuit_set_labels(self):
        model = core.do_iterative_mlgst(
            self.ds, self.mdl_clgst, self.lsgstStrings, minProbClip=1e-4,
            probClipInterval=(-1e2, 1e2), circuitSetLabels=["Set1", "Set2", "Set3"]
        )
        # TODO assert correctness

    def test_do_iterative_mlgst_use_freq_weighted_chi2(self):
        model = core.do_iterative_mlgst(
            self.ds, self.mdl_clgst, self.lsgstStrings, minProbClip=1e-4,
            probClipInterval=(-1e2, 1e2), useFreqWeightedChiSq=True
        )
        # TODO assert correctness

    def test_do_iterative_mlgst_circuit_weights_dict(self):
        model = core.do_iterative_mlgst(
            self.ds, self.mdl_clgst, self.lsgstStrings, minProbClip=1e-4,
            probClipInterval=(-1e2, 1e2), circuitWeightsDict={(Label('Gx'),): 2.0}
        )
        # TODO assert correctness

    def test_do_iterative_mlgst_always_perform_MLE(self):
        model = core.do_iterative_mlgst(
            self.ds, self.mdl_clgst, self.lsgstStrings, minProbClip=1e-4,
            probClipInterval=(-1e2, 1e2), alwaysPerformMLE=True
        )
        # TODO assert correctness

    def test_do_mlgst_raises_on_out_of_memory(self):
        with self.assertRaises(MemoryError):
            core.do_mlgst(
                self.ds, self.mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
                probClipInterval=(-1e2, 1e2), memLimit=0
            )

    # XXX if this function needs explicit coverage, it should be public!
    def test_do_mlgst_base_forcefn_grad(self):
        forcefn_grad = np.ones((1, self.mdl_clgst.num_params()), 'd')
        model = core._do_mlgst_base(
            self.ds, self.mdl_clgst, self.lsgstStrings[0], minProbClip=1e-4,
            probClipInterval=(-1e2, 1e2), forcefn_grad=forcefn_grad
        )
        # TODO assert correctness
