import numpy as np
import pygsti
import pickle
from pygsti.objectivefns import objectivefns as _objfns
import warnings
import time
from pygsti.tools.modelselectiontools import create_red_model as _create_red_model, reduced_model_approx_GST_fast as _reduced_model_approx_GST_fast
from pygsti.tools.modelselectiontools import parallel_GST as _parallel_GST, AMSCheckpoint as _AMSCheckpoint, remove_param, create_approx_logl_fn
import os as _os

def do_greedy_from_full_fast(target_model, data, er_thresh=2.0, verbosity=2, maxiter=100, tol=1.0, prob_clip=1e-3, recompute_H_thresh_percentage = .1, disable_checkpoints = False, checkpoint = None, comm = None, mem_limit = None):
    """
    An automated model selection greedy algorithm. Specifically made for FOGI models, but it should be compatible
    with any model that has a linear interposer. It is considered "fast" because on most model fits, GST analysis is
    replaced by approximate MLE explained in step 3 below. The high level picture of the algorithm goes as follows:
    
        1) Do a GST fit on target_model

        2) Consider a list of target_model.num_params reduced models of target_model, each missing a single and unique parameter.

        3) Estimate the MLE of these smaller models through linear inversion of a second order taylor series approximation of the
        likelihood function. This is done in reduced_model_approx_GST() and reduced_model_approx_GST_fast(). If multiple processes
        are available, the list of reduced models to be evaluated gets evenly split amongst them.

        4) Grab the reduced model with the lowest difference in MLE (MLE(target_model) - MLE(red_model)), use this as your target_model
        and go back to step 1.

    Parameters
    ----------
    target_model : Model

        pyGSTi model with a linear interposer to be used as a starting point to AMS. Reduced models
        are created from this model by removing a single parameter at a time.

    data :  ProtocolData
        The input data to be used to fit models 

    er_thresh : float, optional
        The threshold that determines how much likelihood we are willing to lose per 
        parameter removed. In other words, if after removing a parameter, all resulting model fits
        are above the treshold
            L(target_model) - L(red_models) > er_thresh/2
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
        TODO, parameter not implemented yet

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
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank = 0
        size = 1
    #logl function will not change throughout this algorithm, so we can save a lot of time
    #by setting it up once, and recomputing its value with different inputs throughout AMS
    def create_logl_obj_fn(parent_model, dataset, min_prob_clip = 1e-6, comm = None, mem_limit = None):
        prob_clip_interval=(-1e6, 1e6) 
        radius=1e-4
        poisson_picture = True
        regularization = {'min_prob_clip': min_prob_clip, 'radius': radius} if poisson_picture \
            else {'min_prob_clip': min_prob_clip}
        op_label_aliases = None 
        mdc_store = None
        circuits = None
        obj_cls = _objfns.PoissonPicDeltaLogLFunction if poisson_picture else _objfns.DeltaLogLFunction
        model_copy = parent_model.copy()
        obj = _objfns._objfn(obj_cls, model_copy, dataset, circuits,
                            regularization, {'prob_clip_interval': prob_clip_interval},
                            op_label_aliases, comm, mem_limit, ('percircuit',), (), mdc_store)
        return obj
    
    recompute_Hessian = False
    graph_levels = []
    target_model.param_interposer.full_span_inv_transform_matrix = target_model.param_interposer.inv_transform_matrix
    target_model.param_interposer.inv_transform_matrix_projector = np.eye(target_model.num_params)
    parent_model_projector = target_model.param_interposer.projector_matrix.copy()

    if checkpoint is not None:
        #TODO: maybe not have all MPI processes read from memory in the future
        loaded_checkpoint = _AMSCheckpoint.read(checkpoint)
        if loaded_checkpoint.check_valid_checkpoint(target_model, data.dataset.to_str(), er_thresh, maxiter, tol, prob_clip, recompute_H_thresh_percent=recompute_H_thresh_percentage):
            H = loaded_checkpoint.H
            x0 = loaded_checkpoint.x0
            target_model_fit = target_model.copy()
            target_model_fit.from_vector(x0)
            if rank == 0:
                print('Warm starting AMS from checkpoint ' + checkpoint)
        else:
            if rank == 0:
                raise ValueError('Invalid AMS checkpoint provided. The checkpoint settings are:', f"{loaded_checkpoint.er_thresh=}, {loaded_checkpoint.maxiter=}, {loaded_checkpoint.tol=}, {loaded_checkpoint.prob_clip=}, {loaded_checkpoint.recompute_H_thresh_percent=}", ' make sure that these match the arguments provided match these, and that the target model and data are correct.')
    
    else:
        if rank == 0:
                print('starting GST')
        target_model.sim = pygsti.forwardsims.MapForwardSimulator(param_blk_sizes=(100,100))
        result = _parallel_GST(target_model, data, prob_clip, tol, maxiter, verbosity, comm=comm, mem_limit=mem_limit)
        
        target_model_fit = result.estimates['GateSetTomography'].models['final iteration estimate']
        target_model_fit.sim = pygsti.forwardsims.MapForwardSimulator(param_blk_sizes=(100,100))
        if rank == 0 : 
            print("computing Hessian")
        
        H = pygsti.tools.logl_hessian(target_model_fit, data.dataset, comm=comm, mem_limit=mem_limit, verbosity = verbosity)
        #TODO how to make this work without mpiexec call
        H = comm.bcast(H, root = 0)
        x0 = target_model_fit.to_vector()
        if not disable_checkpoints:
            new_checkpoint = _AMSCheckpoint(target_model, data.dataset.to_str(), er_thresh, maxiter, tol, prob_clip,  recompute_H_thresh_percentage, H, x0)
            new_checkpoint.save()
            print('Checkpoint saved in', new_checkpoint.path)
    red_model = target_model.copy()
    logl_fn = create_logl_obj_fn(target_model_fit, data.dataset)
    original_dlogl = -logl_fn.fn()
    expansion_point_logl = pygsti.tools.logl(target_model_fit, data.dataset, poisson_picture=False)
    approx_logl_fn = create_approx_logl_fn(H, x0, expansion_point_logl)
    prev_logl = original_dlogl
    graph_levels.append([[x0,original_dlogl, 0]])
    exceeded_threshold = False
    red_row_H = H
    red_rowandcol_H = H
    counter = 0
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
                
                #vec = reduced_model_approx_GST(H,reduced_model_projector_matrix.T, x0)
                vec = _reduced_model_approx_GST_fast(red_rowandcol_H, red_row_H, x0, i)

                #vec = _parallel_GST(remove_param(red_model, i), data, prob_clip, tol, maxiter, verbosity=0, comm=None, mem_limit=None).estimates['GateSetTomography'].models['final iteration estimate'].to_vector()
                logl_fn.model.from_vector(reduced_model_projector_matrix @ vec)
                quantity = (prev_logl + logl_fn.fn())*2
                if i  % 50 == 0 and verbosity > 1:
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
                logl_fn.model.from_vector(reduced_model_projector_matrix @ vec)

                quantity = (prev_logl + logl_fn.fn())*2
                
                if lowest_quantity == None or lowest_quantity > quantity:
                    lowest_quantity = quantity
                    lowest_imdl = i
                    lowest_vec = vec
            
            best_of_chunk = [lowest_vec, lowest_quantity, lowest_imdl]

            finalists_raw = comm.allgather(best_of_chunk)
            finalists = []
            for finalist in finalists_raw:
                finalists.append(finalist)
            #Purge out all entries from processes that did not compute anything
            finalists = [finalist for finalist in finalists if finalist[2] != -1]
            
            sorted_finalists = sorted(finalists, key=lambda x: x[1])

            
        red_model = remove_param(red_model, sorted_finalists[0][2])
        red_model.from_vector(sorted_finalists[0][0])
        finalist_real_logl = pygsti.tools.logl(red_model, data.dataset, poisson_picture=False, min_prob_clip=prob_clip)
        finalist_approx_logl = approx_logl_fn(red_row_H, red_rowandcol_H, sorted_finalists[0][2])
        error = finalist_real_logl - finalist_approx_logl
        #DEBUG Delete
        if False: #rank == 0:
            print(f'{error=}', 'compared to ', recompute_H_thresh_percentage*er_thresh)

        if  np.abs(error) > recompute_H_thresh_percentage*er_thresh:
            recompute_Hessian = True
            counter += 1
            

        if recompute_Hessian:
            if verbosity > 0:
                print("Recomputing Hessian, approximation error is ", error)

            result = _parallel_GST(red_model, data, prob_clip, tol, maxiter, verbosity, comm=comm, mem_limit=mem_limit)
        
            red_model_fit = result.estimates['GateSetTomography'].models['final iteration estimate']
            red_model_fit.sim = pygsti.forwardsims.MapForwardSimulator(param_blk_sizes=(100,100))
            H = pygsti.tools.logl_hessian(red_model_fit, data.dataset, comm=comm, mem_limit=mem_limit, verbosity = verbosity)
            H = comm.bcast(H, root = 0)
            x0 = red_model_fit.to_vector()
        
            reduced_model_projector_matrix = np.delete(parent_model_projector, sorted_finalists[0][2], axis=1)     
            logl_fn.model.from_vector(reduced_model_projector_matrix @ x0)
            sorted_finalists[0][1] = (prev_logl + logl_fn.fn())*2
            expansion_point_logl = pygsti.tools.logl(red_model_fit, data.dataset, poisson_picture=False)
            approx_logl_fn = create_approx_logl_fn(H, x0, expansion_point_logl)
            
        if verbosity and rank == 0:
            print(f'Model {sorted_finalists[0][2]} has lowest evidence ratio {sorted_finalists[0][1]:.4f}')
        
        if sorted_finalists[0][1] > er_thresh or len(sorted_finalists[0][0]) == 1:
                if rank == 0 and verbosity > 0:
                    print('All models from this level exceeded evidence ratio threshold, model rejected. Stopping!')
                #print('Final accepted model (parent from previous iteration):')
                #print('\n'.join(graph_levels[-1][0][0].parameter_labels_pretty))
                exceeded_threshold = True
                final_projector_matrix = parent_model_projector
                final_model = _create_red_model(target_model.copy(), final_projector_matrix, graph_levels[-1][0][0])

                final_fit = _parallel_GST(final_model.copy(), data, prob_clip, tol, maxiter, verbosity, comm=comm, mem_limit=mem_limit).estimates['GateSetTomography'].models['final iteration estimate']
    
                logl_fn.model.from_vector(final_projector_matrix @ final_fit.to_vector())
                curr_logl = - logl_fn.fn()
                new_quantity = 2*(prev_logl - curr_logl)
                total_ev_ratio = 2*(original_dlogl - curr_logl)/len(graph_levels)
                
                    
                evratios = [level[0][1] for level in graph_levels[1:]]
                pre_opt = np.average(evratios)
                if rank == 0 and verbosity > 0:
                    print('Evidence ratio wrt first model is ', total_ev_ratio)
                    print('Evidence ratio pre-optimization was ', pre_opt)
                    print('Evidence ratio wrt first model improved by ', pre_opt - total_ev_ratio)
                graph_levels[-1][0] = [final_fit.to_vector(), new_quantity, graph_levels[-1][0][2], graph_levels[-1][0][1]]
                if total_ev_ratio > er_thresh:
                     warnings.warn("Final model does not meet er_thresh specified. This can only happen if there is a bug, or if logl approximations were used" + str(2*(original_dlogl - curr_logl)/len(graph_levels)))

        else:
            parent_model_projector = np.delete(parent_model_projector, sorted_finalists[0][2], axis=1)
            if recompute_Hessian:
                red_row_H = H
                red_rowandcol_H = H
                recompute_Hessian = False
            else:
                red_row_H = np.delete(red_row_H, sorted_finalists[0][2], axis=0)
                red_rowandcol_H = np.delete(np.delete(red_rowandcol_H,sorted_finalists[0][2] , axis=0), sorted_finalists[0][2], axis=1)
            
        if not exceeded_threshold:
            # Save level
            graph_levels.append(sorted_finalists)

        if len(graph_levels) > 1:
            prev_logl = prev_logl - sorted_finalists[0][1]/2
            
        # Generate next level
        
        if rank == 0 and exceeded_threshold == False and verbosity > 1:
            print("next level")
            end = time.time()
            print('time this level ', end-start)
    if not disable_checkpoints and rank == 0 and new_checkpoint is not None:
        _os.remove(new_checkpoint.path)
    return graph_levels
