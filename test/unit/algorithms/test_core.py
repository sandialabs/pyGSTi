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
        self.prep_fids = fixtures.prep_fids
        self.meas_fids = fixtures.meas_fids


class CoreFuncTester(CoreStdData, BaseCase):
    def test_gram_rank_and_evals(self):
        rank, evals, target_evals = core.gram_rank_and_eigenvalues(self.ds, self.prep_fids, self.meas_fids, self.model)
        # TODO assert correctness

    def test_gram_rank_and_evals_raises_on_no_target(self):
        # XXX is this neccessary?  EGN: probably not
        with self.assertRaises(ValueError):
            core.gram_rank_and_eigenvalues(self.ds, self.prep_fids, self.meas_fids, None)

    def test_find_closest_unitary_opmx_raises_on_multi_qubit(self):
        with self.assertRaises(ValueError):
            core.find_closest_unitary_opmx(np.identity(16, 'd'))


class CoreLGSTTester(CoreStdData, BaseCase):
    def setUp(self):
        super(CoreLGSTTester, self).setUp()
        self.datagen_gateset = fixtures.datagen_gateset
        self.lgstStrings = fixtures.lgstStrings

    def test_do_lgst(self):
        print(self.model)
        mdl_lgst = core.run_lgst(
            self.ds, self.prep_fids, self.meas_fids, self.model,
            svd_truncate_to=4
        )
        # TODO assert correctness

        # XXX is this neccessary? EGN: tests higher verbosity printing.
        mdl_lgst_2 = core.run_lgst(
            self.ds, self.prep_fids, self.meas_fids, self.model,
            svd_truncate_to=4, verbosity=10
        )
        # TODO assert correctness

        self.assertAlmostEqual(mdl_lgst.frobeniusdist(mdl_lgst_2), 0)

    def test_do_lgst_raises_on_no_target(self):
        # XXX is this neccessary?
        with self.assertRaises(ValueError):
            core.run_lgst(
                self.ds, self.prep_fids, self.meas_fids, None, svd_truncate_to=4
            )

    def test_do_lgst_raises_on_no_spam_dict(self):
        with self.assertRaises(ValueError):
            core.run_lgst(
                self.ds, self.prep_fids, self.meas_fids, None,
                op_labels=list(self.model.operations.keys()), svd_truncate_to=4
            )

    def test_do_lgst_raises_on_bad_fiducials(self):
        bad_fids = [Circuit([Label('Gxpi2',0)], line_labels=(0,)), Circuit([Label('Gxpi2',0)], line_labels=(0,)), 
                    Circuit([Label('Gxpi2',0)], line_labels=(0,)), Circuit([Label('Gxpi2',0)], line_labels=(0,))]
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
                bad_ds, self.prep_fids, self.meas_fids, self.model,
                svd_truncate_to=4
            )

    def test_do_lgst_raises_on_incomplete_x_matrix(self):
        incomplete_strings = self.lgstStrings[:-5]  # drop last 5 strings...
        bad_ds = pdata.simulate_data(
            self.datagen_gateset, incomplete_strings,
            num_samples=10, sample_error='none')
        with self.assertRaises(KeyError):
            core.run_lgst(
                bad_ds, self.prep_fids, self.meas_fids, self.model,
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
                (x if x != Label(('Gxpi2',0)) else Label("GA1")) for x in mdl
            ], line_labels = (0,)) for mdl in self.lsgstStrings[0]
        ]
        aliases = {Label('GA1'): Circuit([Label('Gxpi2',0)], line_labels= (0,))}
        aliased_list = CircuitList(aliased_list, aliases)

        print(list(aliased_list))

        aliased_model = self.mdl_clgst.copy()
        aliased_model.operations['GA1'] = self.mdl_clgst.operations['Gxpi2',0]
        aliased_model.operations.pop(('Gxpi2',0))

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
        weighted_lists = [CircuitList(lst, circuit_weights=make_weights_array(lst, {('Gxpi2',0): 2.0}))
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
                (x if x != Label('Gxpi2',0) else Label("GA1")) for x in mdl
            ], line_labels=(0,)) for mdl in self.lsgstStrings[0]
        ]
        aliases = {Label('GA1'): Circuit([Label('Gxpi2',0)], line_labels=(0,))}
        aliased_list = CircuitList(aliased_list, aliases)

        aliased_model = self.mdl_clgst.copy()
        aliased_model.operations['GA1'] = self.mdl_clgst.operations['Gxpi2',0]
        aliased_model.operations.pop(('Gxpi2',0))

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

    #Add a test for the new generator method.
    #run_iterative_gst uses this under the hood, so we really only need to separately
    #test starting at a different starting index in the fits.

    def test_iterative_gst_generator_starting_index(self):
        generator = core.iterative_gst_generator(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=['logl'],
            resource_alloc=None, starting_index=0, verbosity=0
        )

        models = []

        for _ in range(len(self.lsgstStrings)):
            models.append(next(generator)[0])

        #now make a new generator starting from a different index
        #with a start model corresponding to the model in models for
        #that index.
        generator1 = core.iterative_gst_generator(
            self.ds, models[0], self.lsgstStrings,
            optimizer={'tol': 1e-5},
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=['logl'],
            resource_alloc=None, starting_index=1, verbosity=0
        )
        models1= []
        for _ in range(1,len(self.lsgstStrings)):
            models1.append(next(generator1)[0])

        #Make sure we get the same result in both cases.
        self.assertArraysAlmostEqual(models[-1].to_vector(), models1[-1].to_vector())
    def test_iterative_gst_generator_optimizers_list(self):
        
        #Test that passing a different optimizer per iteration works as intended
        optimizers = [] 
        tols = [1e1, 1e-8]
        maxiters = [10, 150]

        #First create substantially different optimizers
        for i in range(len(self.lsgstStrings)):
            optimizers.append({'tol': tols[i], 'maxiter':maxiters[i]})

        assert len(self.lsgstStrings) == len(tols), f' If you change {self.lsgstStrings=}, this unit test must be modified to account for it'
        
        generator_optimizers = core.iterative_gst_generator(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer=optimizers,
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=['logl'],
            resource_alloc=None, verbosity=0
        )

        models1 = []
        models0 = []
        #loop over all iterations
        for j in range(0,len(self.lsgstStrings)):

            models0.append(next(generator_optimizers)[0])

            #create a gst generator for the iteration that we are in,
            #to be compared with generator
            generator_step = core.iterative_gst_generator(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer={'tol':tols[j], 'maxiter':maxiters[j]},
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=['logl'],
            resource_alloc=None, verbosity=0,
            starting_index=j
            )
            
            models1.append(next(generator_step)[0])
            
            self.assertArraysAlmostEqual(models0[-1].to_vector(), models1[-1].to_vector())

        # we also test use case of optimzer=[optimizer] being equivalent to optimzer=optimizer
        generator_single_item_optimizers0 = core.iterative_gst_generator(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer=[optimizers[0]],
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=['logl'],
            resource_alloc=None, verbosity=0
        )
        generator_single_item_optimizers1 = core.iterative_gst_generator(
            self.ds, self.mdl_clgst, self.lsgstStrings,
            optimizer=optimizers[0],
            iteration_objfn_builders=['chi2'],
            final_objfn_builders=['logl'],
            resource_alloc=None, verbosity=0
        )

        models0 = []
        models1 = []
        for j in range(0,len(self.lsgstStrings)):

            models0.append(next(generator_single_item_optimizers0)[0])
            models1.append(next(generator_single_item_optimizers1)[0])
            self.assertArraysAlmostEqual(models0[-1].to_vector(), models1[-1].to_vector())
        