import numpy as np

import pygsti.circuits as pc
import pygsti.data as pdata
from pygsti.algorithms import core
from pygsti.baseobjs import Label
from pygsti.circuits import Circuit, CircuitList
from pygsti.objectivefns import Chi2Function, FreqWeightedChi2Function, \
    PoissonPicDeltaLogLFunction
from . import fixtures
from ..util import BaseCase


class CoreStdData(object):
    def setUp(self):
        super(CoreStdData, self).setUp()
        self.ds = fixtures.ds.copy()
        self.model = fixtures.model.copy()
        self.fiducials = fixtures.fiducials


class CoreFuncTester(CoreStdData, BaseCase):
    def test_gram_rank_and_evals(self):
        rank, evals, target_evals = core.gram_rank_and_eigenvalues(self.ds, self.fiducials, self.fiducials, self.model)
        # TODO assert correctness

    def test_gram_rank_and_evals_raises_on_no_target(self):
        # XXX is this neccessary?  EGN: probably not
        with self.assertRaises(ValueError):
            core.gram_rank_and_eigenvalues(self.ds, self.fiducials, self.fiducials, None)

    def test_find_closest_unitary_opmx_raises_on_multi_qubit(self):
        with self.assertRaises(ValueError):
            core.find_closest_unitary_opmx(np.identity(16, 'd'))


class CoreLGSTTester(CoreStdData, BaseCase):
    def setUp(self):
        super(CoreLGSTTester, self).setUp()
        self.datagen_gateset = fixtures.datagen_gateset
        self.lgstStrings = fixtures.lgstStrings

    def test_do_lgst(self):
        mdl_lgst = core.run_lgst(
            self.ds, self.fiducials, self.fiducials, self.model,
            svd_truncate_to=4
        )
        # TODO assert correctness

        # XXX is this neccessary? EGN: tests higher verbosity printing.
        mdl_lgst_2 = core.run_lgst(
            self.ds, self.fiducials, self.fiducials, self.model,
            svd_truncate_to=4, verbosity=10
        )
        # TODO assert correctness

        self.assertAlmostEqual(mdl_lgst.frobeniusdist(mdl_lgst_2), 0)

    def test_do_lgst_raises_on_no_target(self):
        # XXX is this neccessary?
        with self.assertRaises(ValueError):
            core.run_lgst(
                self.ds, self.fiducials, self.fiducials, None, svd_truncate_to=4
            )

    def test_do_lgst_raises_on_no_spam_dict(self):
        with self.assertRaises(ValueError):
            core.run_lgst(
                self.ds, self.fiducials, self.fiducials, None,
                op_labels=list(self.model.operations.keys()), svd_truncate_to=4
            )

    def test_do_lgst_raises_on_bad_fiducials(self):
        bad_fids = pc.to_circuits([('Gx',), ('Gx',), ('Gx',), ('Gx',)])
        with self.assertRaises(ValueError):
            core.run_lgst(
                self.ds, bad_fids, bad_fids, self.model, svd_truncate_to=4
            )  # bad fiducials (rank deficient)

    def test_do_lgst_raises_on_incomplete_ab_matrix(self):
        incomplete_strings = self.lgstStrings[5:]  # drop first 5 strings...
        bad_ds = pdata.simulate_data(
            self.datagen_gateset, incomplete_strings,
            num_samples=10, sample_error='none')
        with self.assertRaises(KeyError):
            core.run_lgst(
                bad_ds, self.fiducials, self.fiducials, self.model,
                svd_truncate_to=4
            )

    def test_do_lgst_raises_on_incomplete_x_matrix(self):
        incomplete_strings = self.lgstStrings[:-5]  # drop last 5 strings...
        bad_ds = pdata.simulate_data(
            self.datagen_gateset, incomplete_strings,
            num_samples=10, sample_error='none')
        with self.assertRaises(KeyError):
            core.run_lgst(
                bad_ds, self.fiducials, self.fiducials, self.model,
                svd_truncate_to=4
            )


class CoreMC2GSTTester(CoreStdData, BaseCase):
    def setUp(self):
        super(CoreMC2GSTTester, self).setUp()
        self.mdl_clgst = fixtures.mdl_clgst.copy()
        self.lsgstStrings = fixtures.lsgstStrings

    def test_do_mc2gst(self):
        mdl_lsgst = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                            optimizer=None, objective_function_builder="chi2",
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_mc2gst_regularize_factor(self):
        obj_builder = Chi2Function.builder(
            name='chi2',
            description="Sum of chi^2",
            regularization={'min_prob_clip_for_weighting': 1e-4},
            penalties={'regularize_factor': 1e-3}
        )
        mdl_lsgst = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                            {'tol': 1e-5}, obj_builder,
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_mc2gst_CPTP_penalty_factor(self):
        obj_builder = Chi2Function.builder(
            name='chi2',
            description="Sum of chi^2",
            regularization={'min_prob_clip_for_weighting': 1e-4},
            penalties={'cptp_penalty_factor': 1.0}
        )
        mdl_lsgst = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                            {'tol': 1e-5}, obj_builder,
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_mc2gst_SPAM_penalty_factor(self):
        obj_builder = Chi2Function.builder(
            name='chi2',
            description="Sum of chi^2",
            regularization={'min_prob_clip_for_weighting': 1e-4},
            penalties={'spam_penalty_factor': 1.0}
        )
        mdl_lsgst = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                            {'tol': 1e-5}, obj_builder,
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_mc2gst_CPTP_SPAM_penalty_factor(self):
        obj_builder = Chi2Function.builder(
            name='chi2',
            description="Sum of chi^2",
            regularization={'min_prob_clip_for_weighting': 1e-4},
            penalties={'cptp_penalty_factor': 1.0,
                       'spam_penalty_factor': 1.0}
        )
        mdl_lsgst = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                            {'tol': 1e-5}, obj_builder,
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_mc2gst_alias_model(self):
        aliased_list = [
            Circuit([
                (x if x != Label("Gx") else Label("GA1")) for x in mdl
            ]) for mdl in self.lsgstStrings[0]
        ]
        aliases = {Label('GA1'): Circuit(['Gx'])}
        aliased_list = CircuitList(aliased_list, aliases)

        aliased_model = self.mdl_clgst.copy()
        aliased_model.operations['GA1'] = self.mdl_clgst.operations['Gx']
        aliased_model.operations.pop('Gx')

        mdl_lsgst = core.run_gst_fit_simple(self.ds, aliased_model, aliased_list,
                                            {'tol': 1e-5}, "chi2",
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_iterative_mc2gst(self):
        mdl_lsgst = core.run_iterative_gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=[],
            resource_alloc=None
        )
        # TODO assert correctness

    def test_do_iterative_mc2gst_regularize_factor(self):
        obj_builder = Chi2Function.builder(
            name='chi2',
            description="Sum of chi^2",
            regularization={'min_prob_clip_for_weighting': 1e-4},
            penalties={'regularize_factor': 10.0}
        )
        mdl_lsgst = core.run_iterative_gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=[obj_builder],
            final_objfn_builders=[],
            resource_alloc=None
        )
        # TODO assert correctness

    def test_do_iterative_mc2gst_use_freq_weighted_chi2(self):
        obj_builder = FreqWeightedChi2Function.builder(
            name='freq-weighted-chi2',
            description="Sum of chi^2",
            regularization={'min_freq_clip_for_weighting': 1e-4}
        )
        mdl_lsgst = core.run_iterative_gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=[obj_builder],
            final_objfn_builders=[],
            resource_alloc=None
        )
        # TODO assert correctness

    def test_do_iterative_mc2gst_circuit_weights_dict(self):
        def make_weights_array(l, weights_dict):
            return np.array([weights_dict.get(circuit, 1.0) for circuit in l])
        weighted_lists = [CircuitList(lst, circuit_weights=make_weights_array(lst, {('Gx',): 2.0}))
                          for lst in self.lsgstStrings]
        mdl_lsgst = core.run_iterative_gst(
            self.ds, self.mdl_clgst, weighted_lists,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=[],
            resource_alloc=None
        )
        # TODO assert correctness

    def test_do_mc2gst_raises_on_out_of_memory(self):
        with self.assertRaises(MemoryError):
            core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                    {'tol': 1e-5}, 'chi2',
                                    resource_alloc={'mem_limit': 0}
                                    )


# XXX shouldn't this code be reused?
class CoreMLGSTTester(CoreStdData, BaseCase):
    def setUp(self):
        super(CoreMLGSTTester, self).setUp()
        self.mdl_clgst = fixtures.mdl_clgst.copy()
        self.lsgstStrings = fixtures.lsgstStrings

    def test_do_mlgst(self):
        mdl_lsgst = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                            optimizer=None, objective_function_builder="logl",
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_mlgst_CPTP_penalty_factor(self):
        obj_builder = PoissonPicDeltaLogLFunction.builder(
            name='logl',
            description="2*DeltaLogL",
            regularization={'min_prob_clip': 1e-4},
            penalties={'cptp_penalty_factor': 1.0,
                       'prob_clip_interval': (-1e2, 1e2)}
        )
        mdl_lsgst = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                            optimizer={'tol': 1e-5}, objective_function_builder=obj_builder,
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_mlgst_SPAM_penalty_factor(self):
        obj_builder = PoissonPicDeltaLogLFunction.builder(
            name='logl',
            description="2*DeltaLogL",
            regularization={'min_prob_clip': 1e-4},
            penalties={'spam_penalty_factor': 1.0,
                       'prob_clip_interval': (-1e2, 1e2)}
        )
        mdl_lsgst = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                            optimizer={'tol': 1e-5}, objective_function_builder=obj_builder,
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_mlgst_CPTP_SPAM_penalty_factor(self):
        # this test often gives an assetion error "finite Jacobian has
        # inf norm!" on Travis CI Python 3 case. Just ignore for now.
        # FUTURE: see what we can do in custom LM about scaling large
        # jacobians...
        #self.skipTest("Ignore for now.")
        obj_builder = PoissonPicDeltaLogLFunction.builder(
            name='logl',
            description="2*DeltaLogL",
            regularization={'min_prob_clip': 1e-4},
            penalties={'cptp_penalty_factor': 1.0,
                       'spam_penalty_factor': 1.0,
                       'prob_clip_interval': (-1e2, 1e2)}
        )
        mdl_lsgst = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                            optimizer={'tol': 1e-5}, objective_function_builder=obj_builder,
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_mlgst_alias_model(self):
        aliased_list = [
            Circuit([
                (x if x != Label("Gx") else Label("GA1")) for x in mdl
            ]) for mdl in self.lsgstStrings[0]
        ]
        aliases = {Label('GA1'): Circuit(['Gx'])}
        aliased_list = CircuitList(aliased_list, aliases)

        aliased_model = self.mdl_clgst.copy()
        aliased_model.operations['GA1'] = self.mdl_clgst.operations['Gx']
        aliased_model.operations.pop('Gx')

        mdl_lsgst = core.run_gst_fit_simple(self.ds, aliased_model, aliased_list,
                                            {'tol': 1e-5}, "logl",
                                            resource_alloc=None
                                            )
        # TODO assert correctness

    def test_do_iterative_mlgst(self):
        model = core.run_iterative_gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=['logl'],
            resource_alloc=None
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

    def test_do_iterative_mlgst_use_freq_weighted_chi2(self):
        obj_builder = FreqWeightedChi2Function.builder(
            name='freq-weighted-chi2',
            description="Sum of chi^2",
            regularization={'min_freq_clip_for_weighting': 1e-4}
        )
        model = core.run_iterative_gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=[obj_builder],
            final_objfn_builders=['logl'],
            resource_alloc=None
        )
        # TODO assert correctness

    def test_do_iterative_mlgst_circuit_weights_dict(self):
        def make_weights_array(l, weights_dict):
            return np.array([weights_dict.get(circuit, 1.0) for circuit in l])
        weighted_lists = [CircuitList(lst, circuit_weights=make_weights_array(lst, {('Gx',): 2.0}))
                          for lst in self.lsgstStrings]
        model = core.run_iterative_gst(
            self.ds, self.mdl_clgst, weighted_lists,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=['logl'],
            resource_alloc=None
        )
        # TODO assert correctness

    def test_do_iterative_mlgst_always_perform_MLE(self):
        model = core.run_iterative_gst(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=['chi2', 'logl'],
            final_objfn_builders=[],
            resource_alloc=None
        )
        # TODO assert correctness

    def test_do_mlgst_raises_on_out_of_memory(self):
        with self.assertRaises(MemoryError):
            mdl_lsgst = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                                optimizer=None, objective_function_builder="logl",
                                                resource_alloc={'mem_limit': 0}
                                                )

    # XXX if this function needs explicit coverage, it should be public!
    def test_do_mlgst_base_forcefn_grad(self):
        forcefn_grad = np.ones((1, self.mdl_clgst.num_params), 'd')
        obj_builder = PoissonPicDeltaLogLFunction.builder(
            name='logl',
            description="2*DeltaLogL",
            regularization={'min_prob_clip': 1e-4},
            penalties={'forcefn_grad': forcefn_grad,
                       'prob_clip_interval': (-1e2, 1e2)}
        )
        model = core.run_gst_fit_simple(self.ds, self.mdl_clgst, self.lsgstStrings[0],
                                        optimizer=None, objective_function_builder=obj_builder,
                                        resource_alloc=None
                                        )
        # TODO assert correctness
