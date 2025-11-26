import numpy as np
import pygsti
import pickle
from pygsti.objectivefns import objectivefns as _objfns
import warnings
import time
from pygsti.tools.modelselectiontools import create_red_model as _create_red_model, reduced_model_approx_GST_fast as _reduced_model_approx_GST_fast
from pygsti.tools.modelselectiontools import parallel_GST as _parallel_GST, AMSCheckpoint as _AMSCheckpoint, remove_param, create_approx_logl_fn
from pygsti.tools.modelselectiontools import custom_builder as _custom_builders, create_projector_matrix_from_trace
import os as _os


def do_greedy_from_full_fast(full_model, data, er_thresh=2.0, verbosity=2, maxiter=100, tol=1.0, prob_clip=1e-3, recompute_H_thresh_percentage = .1, disable_checkpoints = False, checkpoint = None, comm = None, mem_limit = None):
    """
    An automated model selection greedy algorithm. Specifically made for FOGI models, but it should be compatible
    with any model that has a linear interposer. It is considered "fast" because on most model fits, GST analysis is
    replaced by approximate MLE explained in step 3 below. The high level picture of the algorithm goes as follows:
    
        1) Do a GST fit on full_model

        2) Consider a list of full_model.num_params reduced models of full_model, each missing a single and unique parameter.

        3) Estimate the MLE of these smaller models through linear inversion of a second order taylor series approximation of the
        likelihood function. This is done in reduced_model_approx_GST() and reduced_model_approx_GST_fast(). If multiple processes
        are available, the list of reduced models to be evaluated gets evenly split amongst them.

        4) Grab the reduced model with the lowest difference in MLE (MLE(full_model) - MLE(red_model)), use this as your full_model
        and go back to step 1.

    Parameters
    ----------
    full_model : Model

        pyGSTi model with a linear interposer to be used as a starting point to AMS. Reduced models
        are created from this model by removing a single parameter at a time.

    data :  ProtocolData
        The input data to be used to fit models 

    er_thresh : float, optional
        The threshold that determines how much likelihood we are willing to lose per 
        parameter removed. In other words, if after removing a parameter, all resulting model fits
        are above the treshold
            L(full_model) - L(red_models) > er_thresh/2
        Then the algorithm rejects all current models and stops. This is called the "evidence ratio"
        within the field of model selection. If not provided, it is set to the Akaike information
        criterion (2).

    verbosity : int, optional
        Amount of detail to print to stdout.
        
    maxiter : int
        The maximum number of (outer) interations for the optimizer.

    tol : float
        Tolerance value used for optimization algorithms. This parameters will index the 'f' tolerance
        (also called f_norm2_tol) in optimization subroutines, described as follows:
            Tolerace for `F^2` where `F = `norm( sum(obj_fn(x)**2) )` is the
            least-squares residual.  If `F**2 < f_norm2_tol`, then mark converged.  

    prob_clip : float
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).
    
    recompute_H_thresh_percentage : float
        TODO,

    graph_checkpoint : TODO this will likely get heavily modified

    Returns
    -------
    trace:
        A list of every level computed within AMS, each level contains a list of characteristics of all
        reduced models considered
    trace[i][j][k]
        i: indexes over levels of AMS, where the last entry is the last level considered, in this case
        containing the smallest models

        j: indexes model characteristics of all reduced models considered in level i

        k: indexes different characteristics for the jth best model at level i
            [param_vec , evidence_ratio, parameter_that_was_removed]

    lowest_vec : numpy array
        vector corresponding to the fitted best reduced model found at its corresponding level
    """
    new_checkpoint = None
    loaded_checkpoint = None
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1
    if not disable_checkpoints:
        new_checkpoint = _AMSCheckpoint(None, data.dataset.to_str(), er_thresh, maxiter, tol, prob_clip, None, None)
    
    recompute_Hessian = False
    graph_levels = []
    full_model.param_interposer.full_span_inv_transform_matrix = full_model.param_interposer.inv_transform_matrix
    full_model.param_interposer.inv_transform_matrix_projector = np.eye(full_model.num_params)
    full_model_fit = None
    builders = _custom_builders(prob_clip)
    deltalogl_fn =  builders.final_builders[0].build(full_model, data.dataset, list(data.dataset.keys()))
    H = None
    x0 = None
    result = None

    if checkpoint is not None:
        #TODO: maybe not have all MPI processes read from memory in the future
        loaded_checkpoint = _AMSCheckpoint.read(checkpoint)
        if loaded_checkpoint.check_valid_checkpoint(full_model, data.dataset.to_str(), er_thresh, maxiter, tol, prob_clip):
            H = loaded_checkpoint.H
            x0 = loaded_checkpoint.x0
            full_model_fit = full_model.copy()
            full_model_fit.sim._processor_grid = (1,1,1)
            full_model_fit.from_vector(x0)
            deltalogl_fn.model = full_model_fit
            original_dlogl = deltalogl_fn.fn()

            if rank == 0:
                print('Warm starting AMS from checkpoint ' + checkpoint)
    
        else:
            if rank == 0:
                raise ValueError('Invalid AMS checkpoint provided. The checkpoint settings are:', f"{loaded_checkpoint.er_thresh=}, {loaded_checkpoint.maxiter=}, {loaded_checkpoint.tol=}, {loaded_checkpoint.prob_clip=}", ' make sure that these match the arguments provided match these, and that the target model and data are correct.')
        if new_checkpoint is not None:
            new_checkpoint.x0 = x0
            new_checkpoint.H = H
            new_checkpoint.full_model = full_model_fit
    
    
    
    if full_model_fit is None:
        if rank == 0: #and verbosity > 0:
            print('starting GST ', size)
        full_model.sim = pygsti.forwardsims.MapForwardSimulator(processor_grid=(1,size))
        
        if rank == 0:
            start = time.time()

        result = _parallel_GST(full_model, data, builders, tol, maxiter, verbosity, comm=comm, mem_limit=mem_limit)

        full_model_fit = result.estimates['GateSetTomography'].models['final iteration estimate']

        x0 = full_model_fit.to_vector()

        full_model_fit.sim._processor_grid = (1,1,1)
        deltalogl_fn.model = full_model_fit.copy()
        original_dlogl = deltalogl_fn.fn()

        if not disable_checkpoints and rank == 0:
            new_checkpoint.full_model = full_model_fit
            new_checkpoint.x0 = x0
            new_checkpoint.save()
            print('Checkpoint saved in', new_checkpoint.path)
    
    if H is None:
        if rank == 0 and verbosity > 0: 
            print("computing Hessian")
        full_model_fit.sim = pygsti.forwardsims.MapForwardSimulator(processor_grid=(1,1,size),param_blk_sizes=(100,100))
        #H = pygsti.tools.logl_hessian(full_model_fit, data.dataset, comm=comm, mem_limit=mem_limit, verbosity = verbosity)
        tmp_ckp=_AMSCheckpoint.read('./ams_checkpoints/HPC_checkpoint11-16.json')
        H = tmp_ckp.H
        x0 = tmp_ckp.x0
        full_model_fit.from_vector(x0)
        if comm is not None:
            H = comm.bcast(H, root = 0)
        if not disable_checkpoints and rank == 0:
            new_checkpoint.H = H
            new_checkpoint.save()
            print('Checkpoint saved in', new_checkpoint.path)
    
    if loaded_checkpoint is not None:
        if loaded_checkpoint.graph_levels is not None:
            if rank == 0 and verbosity > 0:
                print(f'Checkpoint contains {len(checkpoint.graph_levels)} levels')
            graph_levels = loaded_checkpoint.graph_levels
            red_model = loaded_checkpoint.red_model
            reducer = red_model.projector_matrix.T#create_projector_matrix_from_trace(graph_levels).T
            red_row_H = reducer.T @ H
            red_rowandcol_H = red_row_H @ reducer
    
    #if we did not load a checkpoint with levels in it
    if len(graph_levels) == 0:
        red_model = full_model_fit.copy()
        red_row_H = H
        red_rowandcol_H = H
        if result is not None:
            print('difference: ', result.estimates['GateSetTomography'].final_objective_fn().fn() - original_dlogl)
            if result.estimates['GateSetTomography'].final_objective_fn().fn() != original_dlogl:
                print('fns did not match!')
                result.write('obj_fn_test')
                print(f'{original_dlogl=}, ', result.estimates['GateSetTomography'].final_objective_fn().fn())

        print(f'{original_dlogl=}')
        prev_dlogl = original_dlogl
        graph_levels.append([[x0,original_dlogl, 0]])
        exceeded_threshold = False
        
        parent_model_projector = full_model.param_interposer.projector_matrix.copy()

    approx_logl_fn = create_approx_logl_fn(H, x0, original_dlogl)
    while not exceeded_threshold:
        if rank == 0 and verbosity >1:
            print(f'>> Working on level {len(graph_levels)} <<',flush = True)
            start = time.time()
        
        if False: #np.abs(approx_error) > recompute_H_thresh_percentage * 2:#er_thresh:
            recompute_Hessian = True
            break
        else:

            num_total_models = len(graph_levels[-1][0][0])
            bucket_size = num_total_models // size
            chunk_range = range(rank*bucket_size, (rank+1)*bucket_size)
            finalists = []
            lowest_imdl = -1
            lowest_quantity = None
            lowest_vec = None
            for i in chunk_range:

                reduced_model_projector_matrix = np.delete(parent_model_projector, i, axis=1)
            
                vec = _reduced_model_approx_GST_fast(red_rowandcol_H, red_row_H, x0, i)

                deltalogl_fn.model.from_vector(reduced_model_projector_matrix @ vec)

                quantity = (deltalogl_fn.fn()- prev_dlogl)*2
                if  verbosity > 1:
                    if True:#i  % (100/verbosity) == 0:
                        print('Model ', i, ' has ev. ratio of ', quantity, flush=True)
                if lowest_quantity == None or lowest_quantity > quantity:
                    lowest_quantity = quantity
                    lowest_imdl = i
                    lowest_vec = vec
                
            left_over_start_index = size * bucket_size

            if (left_over_start_index + rank) < num_total_models:

                i = left_over_start_index + rank
                reduced_model_projector_matrix = np.delete(parent_model_projector, i, axis=1)     
                vec = _reduced_model_approx_GST_fast(red_rowandcol_H, red_row_H, x0, i)
                deltalogl_fn.model.from_vector(reduced_model_projector_matrix @ vec)

                quantity = (deltalogl_fn.fn() - prev_dlogl )*2
                
                if lowest_quantity == None or lowest_quantity > quantity:
                    lowest_quantity = quantity
                    lowest_imdl = i
                    lowest_vec = vec
            
            best_of_chunk = [lowest_vec, lowest_quantity, lowest_imdl]
            if comm is not None:
                finalists_raw = comm.allgather(best_of_chunk)
            else:
                finalists_raw = [best_of_chunk]
            finalists = []
            for finalist in finalists_raw:
                finalists.append(finalist)
            #Purge out all entries from processes that did not compute anything
            finalists = [finalist for finalist in finalists if finalist[2] != -1]
            
            sorted_finalists = sorted(finalists, key=lambda x: x[1])

            red_model = remove_param(red_model, sorted_finalists[0][2])
            finalist_proj_matrix = np.delete(parent_model_projector, sorted_finalists[0][2], axis=1)
            deltalogl_fn.model.from_vector(finalist_proj_matrix @ sorted_finalists[0][0])
            finalist_real_logl = deltalogl_fn.fn()
            finalist_approx_logl = approx_logl_fn(red_row_H, red_rowandcol_H, sorted_finalists[0][2])
            error = finalist_real_logl - finalist_approx_logl
            if rank == 0:
                print(f'{error=}', flush=True)
        #DEBUG Delete
        if False: #rank == 0:
            print(f'{error=}', 'compared to ', recompute_H_thresh_percentage*er_thresh)

        if  False: #np.abs(error) > recompute_H_thresh_percentage*er_thresh:
            recompute_Hessian = True
            

        if recompute_Hessian:
            if verbosity and rank == 0:
                print(f'Model {sorted_finalists[0][2]} has lowest evidence ratio {sorted_finalists[0][1]:.4f}')
            if verbosity > 0 and rank == 0:
                print("Recomputing Hessian, approximation error is ", error)
            red_model.from_vector(sorted_finalists[0][0])
            result = _parallel_GST(red_model, data, builders, tol, maxiter, verbosity, comm=comm, mem_limit=mem_limit)
        
            red_model_fit = result.estimates['GateSetTomography'].models['final iteration estimate']
            red_model_fit.sim = pygsti.forwardsims.MapForwardSimulator(processor_grid=(1,1,size),param_blk_sizes=(100,100))
            H = pygsti.tools.logl_hessian(red_model_fit, data.dataset, comm=comm, mem_limit=mem_limit, verbosity = verbosity)
            if comm is not None:
                H = comm.bcast(H, root = 0)
            x0 = red_model_fit.to_vector()
            red_model_fit.sim._processor_grid = (1,1,1)
            deltalogl_fn.model = red_model_fit.copy()
            sorted_finalists[0][1] = (deltalogl_fn.fn() - prev_dlogl )*2
            expansion_point_logl = deltalogl_fn.fn()
            parent_model_projector = np.eye(len(x0))
            approx_logl_fn = create_approx_logl_fn(H, x0, expansion_point_logl)
            red_row_H = H
            red_rowandcol_H = H
            if rank == 0 and verbosity > 0:
                print('New exact evidence ratio is ', sorted_finalists[0][1])
            
        if verbosity and rank == 0:
            print(f'Model {sorted_finalists[0][2]} has lowest evidence ratio {sorted_finalists[0][1]:.4f}')
        
        if sorted_finalists[0][1] > er_thresh or len(sorted_finalists[0][0]) == 1:
                if rank == 0 and verbosity > 0:
                    print('All models from this level exceeded evidence ratio threshold, model rejected. Stopping!')
                #print('Final accepted model (parent from previous iteration):')
                #print('\n'.join(graph_levels[-1][0][0].parameter_labels_pretty))
                exceeded_threshold = True
                final_projector_matrix = parent_model_projector

                if recompute_Hessian:
                    final_fit = red_model_fit
                    curr_dlogl = expansion_point_logl

                else:
                    final_fit = _parallel_GST(red_model, data, builders, tol, maxiter, verbosity, comm=comm, mem_limit=mem_limit)
                    final_fit_model = final_fit.estimates['GateSetTomography'].models['final iteration estimate']
                    final_fit_model.sim._processor_grid = (1,1,1)
                    deltalogl_fn.model = final_fit_model.copy()
                    curr_dlogl = deltalogl_fn.fn()
                new_quantity = 2*(curr_dlogl - prev_dlogl)
                total_ev_ratio = 2*(curr_dlogl - original_dlogl)/len(graph_levels)
                
                evratios = [level[0][1] for level in graph_levels[1:]]
                pre_opt = np.average(evratios)
                if rank == 0 and verbosity > 0:
                    print('Evidence ratio wrt first model is ', total_ev_ratio)
                    print('Evidence ratio pre-optimization was ', pre_opt)
                    print('Evidence ratio wrt first model improved by ', pre_opt - total_ev_ratio)
                    print('Removed ', len(graph_levels)-1, ' parameters')
                graph_levels[-1][0] = [final_fit_model.to_vector(), new_quantity, graph_levels[-1][0][2], graph_levels[-1][0][1]]
                if total_ev_ratio > er_thresh:
                     warnings.warn("Final model does not meet er_thresh specified. This is likely a result of approximating logl calculations. Evidence ratio between seed model and model being returned: " + total_ev_ratio)

        else:
            if recompute_Hessian:
                recompute_Hessian = False
            else:
                parent_model_projector = np.delete(parent_model_projector, sorted_finalists[0][2], axis=1)
                red_row_H = np.delete(red_row_H, sorted_finalists[0][2], axis=0)
                red_rowandcol_H = np.delete(np.delete(red_rowandcol_H,sorted_finalists[0][2] , axis=0), sorted_finalists[0][2], axis=1)
            
        if not exceeded_threshold:
            # Save level
            graph_levels.append(sorted_finalists)

        if len(graph_levels) > 1:
            prev_dlogl = prev_dlogl + sorted_finalists[0][1]/2
            
        # Generate next level
        
        if rank == 0 and exceeded_threshold == False and verbosity > 1:
            print("next level")
            end = time.time()
            print('time this level ', end-start)
    if not disable_checkpoints and rank == 0 and new_checkpoint is not None:
        _os.remove(new_checkpoint.path)
    return graph_levels
