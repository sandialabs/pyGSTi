import numpy as np
import pygsti
import pickle
from pygsti.objectivefns import objectivefns as _objfns
import warnings
import time
from pygsti.tools.modelselectiontools import create_red_model, reduced_model_approx_GST_fast as _reduced_model_approx_GST_fast
from pygsti.tools.modelselectiontools import parallel_GST as _parallel_GST, AMSCheckpoint as _AMSCheckpoint, remove_param, remove_params, create_approx_logl_fn
from pygsti.tools.modelselectiontools import custom_builder as _custom_builders
import os as _os


def do_greedy_from_full_fast(initial_model, data, er_thresh=2.0, verbosity=2, maxiter=100, tol=1.0, prob_clip=1e-3, recompute_H_thresh_percentage = .1, disable_checkpoints = False, checkpoint = None, comm = None, mem_limit = None):
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
        A list of every level computed within AMS, each level i contains the reduced model with the best evidence ratio
        with i parameters removed from the full model
    trace[i][j]
        
        i: indexes AMS levels, where each level i contains the best model found after removing i parameters

        j: indexes different characteristics for the jth best model at level i
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
        new_checkpoint = _AMSCheckpoint(data.dataset.to_str(), er_thresh, maxiter, tol, prob_clip, None, None, None)
    
    recompute_Hessian = False
    graph_levels = []
    initial_model.param_interposer.full_span_inv_transform_matrix = initial_model.param_interposer.inv_transform_matrix
    initial_model.param_interposer.inv_transform_matrix_projector = np.eye(initial_model.num_params)
    builders = _custom_builders(prob_clip)
    deltalogl_fn =  builders.final_builders[0].build(initial_model, data.dataset, list(data.dataset.keys()))
    H = None
    expansion_point_x0 = None

    if checkpoint is not None:
        #TODO: maybe not have all MPI processes read from memory in the future
        loaded_checkpoint = _AMSCheckpoint.load_checkpoint(checkpoint)
        if loaded_checkpoint.check_valid_checkpoint( data.dataset.to_str(), er_thresh, maxiter, tol, prob_clip):
            if rank == 0:
                print('Warm starting AMS from checkpoint ' + checkpoint)
            H = loaded_checkpoint.H
            expansion_point_x0 = loaded_checkpoint.x0
            assert expansion_point_x0 is not None
            original_dlogl = loaded_checkpoint.original_dlogl
            graph_levels = loaded_checkpoint.graph_levels
            if not (len(graph_levels) == 0 and expansion_point_x0 is not None):
                if rank == 0:
                    print(f'Checkpoint contains {len(graph_levels)} levels')
                params_to_remove = [level[2] for level in graph_levels[1:]]
                initial_model = remove_params(initial_model, params_to_remove)
                deltalogl_fn.model = initial_model
                assert initial_model.num_params == len(expansion_point_x0)
            deltalogl_fn.model.sim._processor_grid = (1,1,1)
            deltalogl_fn.model.from_vector(expansion_point_x0)
            prev_dlogl = loaded_checkpoint.prev_dlogl

            if not disable_checkpoints:
                new_checkpoint = loaded_checkpoint
    
        else:
            if rank == 0:
                raise ValueError('Invalid AMS checkpoint provided. The checkpoint settings are:', f"{loaded_checkpoint.er_thresh=}, {loaded_checkpoint.maxiter=}, {loaded_checkpoint.tol=}, {loaded_checkpoint.prob_clip=}", ' make sure that these match the arguments provided match these, and that the target model and data are correct.')
        
    if expansion_point_x0 is None:
        if rank == 0: #and verbosity > 0:
            print('starting GST ', size)
        
        if rank == 0:
            start = time.time()
        initial_model.sim = pygsti.forwardsims.MapForwardSimulator(processor_grid=(1,size))
        
        expansion_point_x0 = _parallel_GST(initial_model, data, builders, tol, maxiter, verbosity, comm=comm, mem_limit=mem_limit).estimates['GateSetTomography'].models['final iteration estimate'].to_vector().copy()
        initial_model.sim._processor_grid = (1,1,1)
        deltalogl_fn.model.from_vector(expansion_point_x0)
        if comm is not None:
            comm.free_all_children()
        #Is the model stored in deltalogl_fn the same as initial_model?
        assert np.allclose(initial_model.to_vector(), deltalogl_fn.model.to_vector())
        original_dlogl = deltalogl_fn.fn()
        prev_dlogl = original_dlogl

        if not disable_checkpoints and rank == 0:
            new_checkpoint.x0 = expansion_point_x0
            new_checkpoint.original_dlogl = original_dlogl
            new_checkpoint.save()
            print('Checkpoint saved in', new_checkpoint.path)
    
    if H is None:
        if rank == 0 and verbosity > 0: 
            print("computing Hessian")
        initial_model.sim = pygsti.forwardsims.MapForwardSimulator(processor_grid=(1,1,size),param_blk_sizes=(100,100))
        H = pygsti.tools.logl_hessian(initial_model, data.dataset, comm=comm, mem_limit=mem_limit, verbosity = verbosity)
        if comm is not None:
            comm.free_all_children()
        initial_model.sim._processor_grid = (1,1,1)

        if comm is not None:
            H = comm.bcast(H, root = 0)
        if not disable_checkpoints and rank == 0:
            new_checkpoint.H = H
            new_checkpoint.save()
            print('Checkpoint saved in', new_checkpoint.path)

    
    if len(graph_levels) == 0:
        print(f'{original_dlogl=}')
        graph_levels.append([expansion_point_x0,original_dlogl, 0])
        
    red_row_H = H
    red_rowandcol_H = H
    deltalogl_model_projector = np.eye(initial_model.num_params)
    
    approx_logl_fn = create_approx_logl_fn(H, expansion_point_x0, prev_dlogl)
    exceeded_threshold = False
    while not exceeded_threshold:

        if rank == 0 and verbosity > 0:
            print(f'>> Working on level {len(graph_levels)} <<',flush = True)
            start = time.time()

        num_total_models = len(graph_levels[-1][0])
        bucket_size = num_total_models // size
        chunk_range = range(rank*bucket_size, (rank+1)*bucket_size)
        finalists = []
        lowest_imdl = -1
        lowest_quantity = None
        lowest_vec = None
        for i in chunk_range:
        
            vec = _reduced_model_approx_GST_fast(red_rowandcol_H, red_row_H, expansion_point_x0, i)

            deltalogl_fn.model.from_vector(np.delete(deltalogl_model_projector, i, axis=1) @ vec)

            quantity = (deltalogl_fn.fn()- prev_dlogl)*2
            if  verbosity > 1:
                if i  % (100/verbosity) == 0:
                    print('Model ', i, ' has ev. ratio of ', quantity, flush=True)
            if lowest_quantity == None or lowest_quantity > quantity:
                lowest_quantity = quantity
                lowest_imdl = i
                lowest_vec = vec
            
        left_over_start_index = size * bucket_size

        if (left_over_start_index + rank) < num_total_models:

            i = left_over_start_index + rank 
            vec = _reduced_model_approx_GST_fast(red_rowandcol_H, red_row_H, expansion_point_x0, i)
            deltalogl_fn.model.from_vector(np.delete(deltalogl_model_projector, i, axis=1) @ vec)

            quantity = (deltalogl_fn.fn() - prev_dlogl )*2
            
            if lowest_quantity == None or lowest_quantity > quantity:
                lowest_quantity = quantity
                lowest_imdl = i
                lowest_vec = vec
        
        best_of_chunk = [lowest_vec, lowest_quantity, lowest_imdl]
        if comm is not None:
            finalists_raw = comm.gather(best_of_chunk, root = 0)
        else:
            finalists_raw = [best_of_chunk]
        if rank == 0:
            finalists = []
            for finalist in finalists_raw:
                finalists.append(finalist)
            #Purge out all entries from processes that did not compute anything
            finalists = [finalist for finalist in finalists if finalist[2] != -1]
        
            best_model = sorted(finalists, key=lambda x: x[1])[0]
        else:
            best_model = None
        if comm is not None:
            best_model = comm.bcast(best_model, root = 0)

        if comm is None:
            best_model = best_of_chunk
        deltalogl_fn.model.from_vector(np.delete(deltalogl_model_projector, best_model[2], axis=1) @ best_model[0])
        finalist_real_logl = deltalogl_fn.fn()
        finalist_approx_logl = approx_logl_fn(red_row_H, red_rowandcol_H, best_model[2])
        error = finalist_real_logl - finalist_approx_logl

        if rank == 0:
            print(f'{error=}', flush=True)

        if  np.abs(error) > recompute_H_thresh_percentage*er_thresh:
            recompute_Hessian = True
            
        if verbosity and rank == 0:
                print(f'Model {best_model[2]} has lowest evidence ratio {best_model[1]}')
        if recompute_Hessian:
            
            if verbosity > 0 and rank == 0:
                print("Recomputing Hessian, approximation error is ", error)
            sim = pygsti.forwardsims.MapForwardSimulator(processor_grid=(1,size))
            red_model_fit = _parallel_GST(create_red_model(deltalogl_fn.model, np.delete(deltalogl_model_projector, best_model[2], axis=1), sim=sim, vec=None), data, builders, tol, maxiter, verbosity, comm=comm, mem_limit=mem_limit).estimates['GateSetTomography'].models['final iteration estimate']
            red_model_fit.sim = pygsti.forwardsims.MapForwardSimulator(processor_grid=(1,1,size),param_blk_sizes=(100,100))
            H = pygsti.tools.logl_hessian(red_model_fit, data.dataset, comm=comm, mem_limit=mem_limit, verbosity = verbosity)
            if comm is not None:
                H = comm.bcast(H, root = 0)
                comm.free_all_children()

            expansion_point_x0 = red_model_fit.to_vector().copy()
            red_model_fit.sim._processor_grid = (1,1,1)
            temp_model = deltalogl_fn.model
            deltalogl_fn.model = red_model_fit
            #Free memory of initial model
            initial_model = None
            best_model[1] = (deltalogl_fn.fn() - prev_dlogl )*2
            print(f'{deltalogl_fn.fn()=}, {prev_dlogl=}')
            expansion_point_logl = deltalogl_fn.fn()
            approx_logl_fn = create_approx_logl_fn(H, expansion_point_x0, expansion_point_logl)
            red_row_H = H
            red_rowandcol_H = H
            if rank == 0 and verbosity > 0:
                print('New exact evidence ratio is ', best_model[1])
            

        
        if best_model[1] > er_thresh or len(best_model[0]) == 1:
                if rank == 0 and verbosity > 0:
                    print('All models from this level exceeded evidence ratio threshold, model rejected. Stopping!')
                
                exceeded_threshold = True

                sim = pygsti.forwardsims.MapForwardSimulator(processor_grid=(1,size))
                if recompute_Hessian:
                    deltalogl_fn.model = temp_model
                else:
                    deltalogl_fn.model = create_red_model(deltalogl_fn.model, deltalogl_model_projector,sim=sim)
                deltalogl_fn.model.sim = sim
                final_fit_model = _parallel_GST(deltalogl_fn.model, data, builders, tol, maxiter, verbosity, comm=comm, mem_limit=mem_limit).estimates['GateSetTomography'].models['final iteration estimate']
                final_fit_model.sim._processor_grid = (1,1,1)
                deltalogl_fn.model = final_fit_model
                curr_dlogl = deltalogl_fn.fn()
                new_quantity = 2*(curr_dlogl - prev_dlogl)
                total_ev_ratio = 2*(curr_dlogl - original_dlogl)/len(graph_levels)
                
                evratios = [level[1] for level in graph_levels[1:]]
                pre_opt = np.average(evratios)
                if rank == 0 and verbosity > 0:
                    print('Evidence ratio wrt first model is ', total_ev_ratio)
                    print('Evidence ratio pre-optimization was ', pre_opt)
                    print('Evidence ratio wrt first model improved by ', pre_opt - total_ev_ratio)
                    print('Removed ', len(graph_levels)-1, ' parameters')
                graph_levels[-1] = [final_fit_model.to_vector(), new_quantity, graph_levels[-1][2], graph_levels[-1][1]]
                if total_ev_ratio > er_thresh:
                     warnings.warn("Final model does not meet er_thresh specified. This is likely a result of approximating logl calculations. Evidence ratio between seed model and model being returned: " + total_ev_ratio)

        else:
            prev_dlogl = prev_dlogl + best_model[1]/2
            graph_levels.append(best_model)
            if (not disable_checkpoints and rank == 0 and recompute_Hessian):
                new_checkpoint.graph_levels = graph_levels
                new_checkpoint.prev_dlogl = prev_dlogl
                new_checkpoint.H = H
                new_checkpoint.x0 = expansion_point_x0
                new_checkpoint.save()
            if recompute_Hessian:
                recompute_Hessian = False
                temp_model = None
                deltalogl_model_projector = np.eye(len(expansion_point_x0))
                
            else:
                deltalogl_model_projector = np.delete(deltalogl_model_projector, best_model[2], axis=1)
                red_row_H = np.delete(red_row_H, best_model[2], axis=0)
                red_rowandcol_H = np.delete(np.delete(red_rowandcol_H,best_model[2] , axis=0), best_model[2], axis=1) 

        if rank == 0 and exceeded_threshold == False and verbosity > 1:
            print("next level")
            end = time.time()
            print('time this level ', end-start)
    #if not disable_checkpoints and rank == 0 and new_checkpoint is not None:
    #    _os.remove(new_checkpoint.path)
    return graph_levels



def do_greedy_from_full_exact(initial_model, data, er_thresh=2.0, verbosity=2, maxiter=100, tol=1e-7, prob_clip=1e-3, comm = None, mem_limit = None):
    """
    An automated model selection greedy algorithm. Specifically made for FOGI models, but it should be compatible
    with any model that has a linear interposer. It is considered "exact" because every model considered is found through GST, as opposed to
    some MLE approximations. Because of this, calling this function can yield O(initial_model.num_params^2) GST calculations (expensive!!).
    
    The high level picture of the algorithm goes as follows:
    
        1) Do a GST fit on full_model

        2) Consider a list of full_model.num_params reduced models of full_model, each missing a single and unique parameter.

        3) Find the GST fit of every model in the list constructed in step (2). If multiple processes
        are available, the list of reduced models to be evaluated gets evenly split amongst them.

        4) Grab the reduced model with the lowest difference in MLE (MLE(full_model) - MLE(red_model)), use this as your full_model
        and go back to step 1. If this difference is greater than er_thresh/2, then we stop.

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

    graph_checkpoint : TODO this will likely get heavily modified

    Returns
    -------
    trace:
        A list of every level computed within AMS, each level i contains the reduced model with the best evidence ratio
        with i parameters removed from the full model
    trace[i][j]
        
        i: indexes AMS levels, where each level i contains the best model found after removing i parameters

        j: indexes different characteristics for the jth best model at level i
            [param_vec , evidence_ratio, parameter_that_was_removed]

    lowest_vec : numpy array
        vector corresponding to the fitted best reduced model found at its corresponding level
    """
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1

    graph_levels = []
    initial_model.param_interposer.full_span_inv_transform_matrix = initial_model.param_interposer.inv_transform_matrix
    initial_model.param_interposer.inv_transform_matrix_projector = np.eye(initial_model.num_params)
    builders = _custom_builders(prob_clip)
    deltalogl_fn =  builders.final_builders[0].build(initial_model, data.dataset, list(data.dataset.keys()))
    original_dlogl = None

    if original_dlogl is None:
        if rank == 0: 
            if verbosity > 0:
                print('starting GST ', size)
        
            start = time.time()

            deltalogl_fn.model.from_vector(_parallel_GST(initial_model, data, builders, tol, maxiter, verbosity).estimates['GateSetTomography'].models['final iteration estimate'].to_vector().copy())
        
        if comm is not None:
            deltalogl_fn.model.from_vector(comm.bcast(deltalogl_fn.model.to_vector(), root = 0))
        
        
        if comm is not None:
            comm.free_all_children()
        original_dlogl = deltalogl_fn.fn()
        prev_dlogl = original_dlogl
    
    if len(graph_levels) == 0:
        print(f'{original_dlogl=}')
        graph_levels.append([deltalogl_fn.model.to_vector(),original_dlogl, 0])
        
    reduced_model = deltalogl_fn.model.copy()
    
    exceeded_threshold = False
    while not exceeded_threshold:
        if rank == 0 and verbosity > 0:
            print(f'>> Working on level {len(graph_levels)} <<',flush = True)
            start = time.time()
                        
            print(f'comparing with {prev_dlogl=}')

        num_total_models = len(graph_levels[-1][0])
        bucket_size = num_total_models // size
        chunk_range = range(rank*bucket_size, (rank+1)*bucket_size)
        level_chunk = []
        for i in chunk_range:
            deltalogl_fn.model = _parallel_GST(remove_param(reduced_model, i), data, builders, maxiter=maxiter, tol=tol, verbosity=0).estimates['GateSetTomography'].models['final iteration estimate']

            quantity = (deltalogl_fn.fn()- prev_dlogl)*2
            if  verbosity > 1:
                if i  % 1 == 0:
                    print('Model ', i, ' has ev. ratio of ', quantity, f' with its own logl {deltalogl_fn.fn()=}', flush=True)
            level_chunk.append([deltalogl_fn.model.to_vector(), quantity, i])

            
        left_over_start_index = size * bucket_size

        if (left_over_start_index + rank) < num_total_models:

            deltalogl_fn.model = _parallel_GST(remove_param(reduced_model, i), data, builders, maxiter=maxiter, tol=tol, verbosity=0).estimates['GateSetTomography'].models['final iteration estimate']

            quantity = (deltalogl_fn.fn()- prev_dlogl)*2
            if  verbosity > 1:
                if i  % 1 == 0:
                    print('Model ', i, ' has ev. ratio of ', quantity, f' with its own logl {deltalogl_fn.fn()=}', flush=True)
            
            level_chunk.append([deltalogl_fn.model.to_vector(), quantity, i])
        
        if comm is not None:
            print('waiting', flush=True)
            level_raw = comm.allgather(level_chunk)
            level = [item for sublist in level_raw for item in sublist]
            comm.free_all_children()
        else:
            level = level_chunk

        #Purge out all entries from processes that did not compute anything
        level = [candidate for candidate in level if candidate[2] != -1]
            
        best_model = sorted(level, key=lambda x: x[1])[0]

        reduced_model = remove_param(reduced_model, best_model[2])

        if verbosity and rank == 0:
            print(f'Model {best_model[2]} has lowest evidence ratio {best_model[1]:.4f}')
        
        if best_model[1] > er_thresh or len(best_model[0]) == 1:
                if rank == 0 and verbosity > 0:
                    print('All models from this level exceeded evidence ratio threshold, model rejected. Stopping!')
                exceeded_threshold = True
                if rank == 0 and verbosity > 0:
                    print('Removed ', len(graph_levels)-1, ' parameters')
                final_costs = [candidate[1] for candidate in level]

        else:
            prev_dlogl = prev_dlogl + best_model[1]/2
            if rank == 0:
                print(f'{prev_dlogl=}')
            graph_levels.append(best_model)
                

        if rank == 0 and exceeded_threshold == False and verbosity > 1:
            print("next level")
            end = time.time()
            print('time this level ', end-start)

    return graph_levels, final_costs
