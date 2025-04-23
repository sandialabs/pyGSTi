import pygsti
import numpy as np
from typing import Union, List
from pygsti.algorithms import run_gst_fit
from pygsti.drivers.longsequence import _get_optimizer, _get_badfit_options, _update_objfn_builders
from pygsti.objectivefns import ObjectiveFunctionBuilder, ModelDatasetCircuitsStore
import pygsti.objectivefns
from pygsti.optimize import SimplerLMOptimizer


def make_depolarized_dataset(modelpack, depol_level=0.01, max_max_len=128):
    ideal_model = modelpack.target_model()
    prep_fids = modelpack.prep_fiducials()
    meas_fids = modelpack.meas_fiducials()
    germs = modelpack.germs()
    max_lens = [2**p for p in range(1+int(np.log2(max_max_len)))]
    lsgst_circuit_lists = pygsti.circuits.create_lsgst_circuit_lists(ideal_model, prep_fids, meas_fids, germs, max_lens)
    all_circuits = lsgst_circuit_lists[-1]
    shots_per_circuit = 1000
    rng_state = np.random.default_rng(0)
    depol_model = ideal_model.depolarize(op_noise=depol_level)
    ds = pygsti.data.simulate_data(depol_model, all_circuits, shots_per_circuit, rand_state=rng_state)
    return ds, depol_model


def make_tweaked_dataset(modelpack, depol_level=0.01, rand_unitary_scale=0.001, max_max_len=128):
    ideal_model = modelpack.target_model()
    prep_fids = modelpack.prep_fiducials()
    meas_fids = modelpack.meas_fiducials()
    germs = modelpack.germs()
    max_lens = [2**p for p in range(1+int(np.log2(max_max_len)))]
    lsgst_circuit_lists = pygsti.circuits.create_lsgst_circuit_lists(ideal_model, prep_fids, meas_fids, germs, max_lens)
    all_circuits = lsgst_circuit_lists[-1]
    shots_per_circuit = 1000
    depol_model = ideal_model.depolarize(op_noise=depol_level, spam_noise=depol_level/2, seed=1997)
    final_model = depol_model.randomize_with_unitary(scale=rand_unitary_scale, seed=250422)
    rng_state = np.random.default_rng(0)
    ds = pygsti.data.simulate_data(final_model, all_circuits, shots_per_circuit, rand_state=rng_state)
    return ds, final_model



def corrupt_dataset(ds, prop_corrupt, rng=0):
    dsc = ds.copy_nonstatic()
    rng = np.random.default_rng(rng)
    num_circs = len(dsc)
    selected = rng.choice(np.arange(num_circs), size=int(num_circs*prop_corrupt), replace=False)
    circuits = list(dsc.keys())
    selected = [circuits[i] for i in selected]
    for c in selected:
        num_shots = dsc[c].total
        old_row   = dsc[c].to_dict()
        distn = rng.random(len(old_row))
        distn /= np.sum(distn)
        new_row = {k: num_shots * distn[i] for i,k in enumerate(old_row.keys())}
        dsc[c] = new_row
    dsc.comment  = 'corrupt'
    return dsc, selected


def run_gst(ds, fids, germs, target_model, final_objectives: List[Union[str, tuple]], verbosity: int, mode='CPTPLND',
             iteration_objective='chi2'):
    """
    In the context of this notebook, `ds` is produced by either make_depolarized_dataset or corrupt_dataset.
    final_objective can be anything accepted by `ObjectiveFunctionBuilder.create_from`.

    This function wraps up three steps of a GST pipeline.

    1. Construct a StandardGSTDesign based on (target_model, ds, fids, germs).
         * processor_spec is the value returned from target_model.create_processor_spec.
         * max_lens list is all powers of two that are <= the depth of the longest circuit in ds.
         * circuits in the design are filtered to only include circuits that appeared in ds.

    2. Construct a StandardGST protocol object based on (final_objective, mode, verbosity).
         * The gauge optimization suite is 'stdgaugeopt', minus the TPSpam optimization step.
         * objfn_builders, optimizer, and badfit_options are all set so the final 
           iteration's objective function is based on final_objective.

    3. Run GST with checkpointing turned off. 
        We dot NOT save the results to disk! The calling function is responsible for that.
    """
    if isinstance(final_objectives, str):
        final_objectives = [final_objectives]
    assert isinstance(final_objectives, list)

    max_exp = int(np.log2(np.max([len(c) for c in ds.keys()])))
    max_lens = [2**p for p in range(1 + max_exp)]
    prep_fids, meas_fids = fids

    target_model = target_model.copy()
    target_model.default_gauge_group = 'unitary'

    gos = pygsti.protocols.gst.GSTGaugeOptSuite.cast('stdgaugeopt')
    gop_params = gos.to_dictionary(target_model)
    # ^ a dict with one key, 'stdgaugeopt', whose corresponding value is a list of dicts.
    #   The internal dicts will indicate Frobenius-based losses for gates and SPAM,
    #   along with varying weights. Additional elements can be added to any one of these
    #   internal dicts to be passed to gaugeopt_to_target.
    gop_params['stdgaugeopt'] = gop_params['stdgaugeopt'][:-1]
    # ^ drop the 1-dimensional TPSpam gauge optimization step.

    exp_design = pygsti.protocols.StandardGSTDesign(
        target_model.create_processor_spec(),
        prep_fids, meas_fids, germs, max_lens,
        None,           # germ_length_limits
        None, 1, None,  # fidPairs, keepFraction, keepSeed
        True, True,     # include_lgst, nested_circuit_lists
        None,           # string_manipulation_rules
        None,           # op_label_aliases
        ds, 'drop', verbosity=verbosity
    )
    data = pygsti.protocols.ProtocolData(exp_design, ds)

    #
    #   Run long-sequence GST where the final objective is the first entry
    #   in the final_objectives list.
    #
    final_objective = final_objectives[0]
    builders = pygsti.protocols.GSTObjFnBuilders(
        [ObjectiveFunctionBuilder.create_from(iteration_objective)],
        [ObjectiveFunctionBuilder.create_from(final_objective)]
    )
    _update_objfn_builders(builders.iteration_builders, dict())
    optim_iter = SimplerLMOptimizer.cast(
        _get_optimizer(dict(), target_model)
    )
    advanced_options = {
        'extra_lm_opts': {'tol':
            {'relx': 1e-8, 'relf': 1e-6, 'f': -1.0, 'jac': -1, 'maxdx': 1.0},                                       
        }
    }
    _update_objfn_builders(builders.final_builders, advanced_options)
    optim_last = SimplerLMOptimizer.cast(
        _get_optimizer(advanced_options, target_model)
    )
    bfops = _get_badfit_options(advanced_options)
    proto = pygsti.protocols.StandardGST(
        (mode,), gop_params, target_model, None, 
        objfn_builders    = builders,
        optimizer         = optim_iter,
        badfit_options    = bfops,
        verbosity         = verbosity
    )
    modelest_results = proto.run(data, disable_checkpointing=True)
    modelest_results.rename_estimate(mode, str(final_objective))

    # 
    #   Run one GST fit for each entry in final_objectives[1:].
    #   Initialize the fit at the last model that fit with iteration_objective.
    #
    est = modelest_results.estimates[str(final_objective)]
    seed_name = f'iteration {est.num_iterations - 1} estimate'
    seed_model = est.models[seed_name]
    seed_vec = seed_model.to_vector()
    circuits = exp_design.all_circuits_needing_data
    printer = pygsti.VerbosityPrinter.create_printer(verbosity, None)
    import copy
    for final_objective in final_objectives[1:]:
        builder = ObjectiveFunctionBuilder.create_from(final_objective)
        curr_seed_model = copy.deepcopy(seed_model)
        # ^ A copy is needed because this will be used as the foundational of a ModelDatasetCircuitStore,
        #   which in turn will be the foundation for an MDCObjective.
        curr_seed_model.from_vector(seed_vec)
        array_types = optim_last.array_types + \
            builder.compute_array_types(optim_last.called_objective_methods, curr_seed_model.sim)
        mdc_store = ModelDatasetCircuitsStore(curr_seed_model, data.dataset, circuits, None, array_types=array_types)
        printer.log('')
        _, outobjective = run_gst_fit(mdc_store, optim_last, builder, verbosity - 2)
        
        fobjstr = str(final_objective)
        
        curr_est = copy.deepcopy(est)
        curr_est._final_mdc_store = outobjective
        curr_est._final_objfn = outobjective
        curr_est._final_objfn_cache = None
        curr_est._final_objective_fn_cache = None
        modelest_results.add_estimate(curr_est, fobjstr)

        curr_est = modelest_results.estimates[fobjstr]
        curr_est.models['final iteration estimate'] = outobjective.model
        curr_est.models.pop('stdgaugeopt')
        curr_est.add_gaugeoptimized(gop_params['stdgaugeopt'], label='stdgaugeopt')

        pass

    return modelest_results

