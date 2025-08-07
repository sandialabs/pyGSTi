
import os

#TODO where should these env variables go?
thread_limit = '1'
os.environ['OPENBLAS_NUM_THREADS'] = thread_limit
os.environ['GOTO_NUM_THREADS'] = thread_limit
os.environ['OMP_NUM_THREADS'] = thread_limit
os.environ['NUMEXPR_NUM_THREADS'] = thread_limit
os.environ['VECLIB_MAXIMUM_THREADS'] = thread_limit
os.environ['MKL_NUM_THREADS'] = thread_limit

import numpy as np
import pygsti
from AMS.utils import get_next_reduced_models #compare_parameters_simple, compare_parameters
from AMS.greedy import do_greedy_from_full_fast
from mpi4py import MPI
import pickle
from pygsti.forwardsims import MatrixForwardSimulator
from pygsti.objectivefns import objectivefns as _objfns
from pygsti.circuits import Circuit
from pygsti.baseobjs import Label
import warnings
from pygsti.processors import QubitProcessorSpec
from pygsti.models.modelconstruction import create_explicit_model
import scipy
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
mem = 7
mem_limit = mem*(2**10)**3

def create_projector_matrix_from_trace(graph_levels, projector_matrix = None):
    """TODO

    Args:
        graph_levels (_type_): _description_
        projector_matrix (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if projector_matrix is None:
        projector_matrix = np.eye(len(graph_levels[0][0][0]))
    for level in graph_levels[1:]:
        param_to_remove = level[0][2]
        projector_matrix = np.delete(projector_matrix, param_to_remove, axis=1)
    return projector_matrix

def parallel_GST(target_model, data, min_prob_clip, tol, maxiter, verbosity):
    """
    Wrapper to run GST with MPI with custom builders where the tolerance, probability clip and max iterations
    are easily accesible. The seed model, "target model", gets reset to its error-less version before doing
    GST. This function is specifically made to be used for FOGI AMS, but should work with non-FOGI models too.

    TODO: This function assumes global access to comm and mem_limit, to be changed in the future.
    
    Parameters
    ----------
    
    target_model : Model
        The model to be fit to the data. All errors are eraed, yielding an ideal version of target_model
        as a GST seed.
        
    data : ProtocolData
        The input data to be used for GST analysis
          
    min_prob_clip : float
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).
        
    tol : float
        Tolerance value used for optimization algorithms. This parameters will index the 'f' tolerance
        (also called f_norm2_tol) in optimization subroutines, described as follows:
            Tolerace for `F^2` where `F = `norm( sum(obj_fn(x)**2) )` is the
            least-squares residual.  If `F**2 < f_norm2_tol`, then mark converged.  
               
    maxiter : int
        The maximum number of (outer) interations for the optimizer.
        
    verbosity : int
        Amount of detail to print to stdout.

    Returns:
        ModelEstimateResults
          The return value of running protocols.GateSetTomography()
    """
     
    chi2_builder = pygsti.objectivefns.Chi2Function.builder(
        'chi2', regularization={'min_prob_clip_for_weighting': min_prob_clip}, penalties={'cptp_penalty_factor': 0.0})
    mle_builder = pygsti.objectivefns.PoissonPicDeltaLogLFunction.builder(
        'logl', regularization={'min_prob_clip': min_prob_clip, 'radius': min_prob_clip})
    iteration_builders = [chi2_builder] 
    final_builders = [mle_builder]
    builders = pygsti.protocols.GSTObjFnBuilders(iteration_builders, final_builders)
    
    target_model.from_vector([0]*target_model.num_params)

    optimizer = pygsti.optimize.customlm.CustomLMOptimizer(maxiter=maxiter, tol={'f':tol})
    protoOpt = pygsti.protocols.GateSetTomography(target_model, verbosity=verbosity, optimizer=optimizer, gaugeopt_suite=None, objfn_builders=builders)

    result = protoOpt.run(data, comm=comm,
                memlimit=mem_limit)
    return result

def do_greedy_from_full_fast(target_model, data, er_thresh=2.0, verbosity=2, maxiter=100, tol=1.0, prob_clip=1e-3, recompute_H_thresh_percentage = .2, graph_checkpoint = None):
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

    def create_red_model(parent_model, projector_matrix, vec):
        """TODO

        Args:
            parent_model (_type_): _description_
            projector_matrix (_type_): _description_
            vec (_type_): _description_

        Returns:
            _type_: _description_
        """
        assert projector_matrix.shape[0] == len(vec)
        assert projector_matrix.shape[1] == parent_model.num_params
        
        red_model = parent_model.copy()
        red_model.param_interposer.num_params = len(vec)
        red_model.param_interposer.transform_matrix = parent_model.param_interposer.full_span_transform_matrix @ projector_matrix
        red_model.param_interposer.projector_matrix = projector_matrix.copy()
        reduced_inv_matrix =  projector_matrix.T @ parent_model.param_interposer.inv_transform_matrix
        red_model.param_interposer.inv_transform_matrix = reduced_inv_matrix
        red_model._paramvec = vec.copy()
        red_model.from_vector(red_model._paramvec)

        red_model._need_to_rebuild = True
        red_model._clean_paramvec()
        assert red_model.num_params < parent_model.num_params
        return red_model
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
    
    def reduced_model_approx_GST_fast(red_rowandcol_H, red_row_H, x0, param_to_remove):
        """
        A way to obtain an approximation of the argmax(Likelihood) of a reduced model, assuming the argmax(Likelihood)
        of the parent model is known. Here is how it works:

        Consider a full model with parameter vector x, and a reduced model of this full model with 
        parameter vector xr. Assuming we know argmax(Likelihood) of the full model (call its position x0)
        , we can approximate MLE(xr). To do this, we will approximate the likelihood function up to second 
        order around its maximum x0. Because we are tasked with finding the point at which it peaks, the 
        constant and first order term are not relevant. This yields

        L_approx(x0) = .5(x - x0).T H (x - x0)

        Where H is the Hessian of L. As mentioned above, we are interested in the MLE of a constrained
        version of x, where some entries are pinned to 0. To apply this constrain, we change the input
        of the function to be E @ xr, where E is an embedder map which embeds xr into a higher dimensional
        vector whose extra entries are set to 0. The resulting vector has the same shape as x.

        L_approx(E @ xr) = .5(E @ xr - x0).T H (E @ xr - x0)

        To find its argmax, we take its derivative with respect to xr. After some algebra this results in

        xr_max = (E.T @ H @ E)^-1 E.T @ H @ x0

        This function computes xr_max and returns it

        Parameters
        ----------

        red_rowandcol_H : numpy array
            This is a reduced version of the Hessian of the likelihood function of the full model.
            It is missing both rows and columns corresponding to the indices that xr is missing with 
            respect to x.
            It is obtained through E.T @ H @ E. This variable is meant to hold the reduced version of the
            Hessian corresponding to the previous model evaluated in AMS. Together with param_to_remove
            it will be further reduced down to compute the approximate argmax of a further reduced model.
            For example, the first time this function is called within AMS, if starting from a full model,
            red_rowandcol_H will be equal to H.

        red_row_H : numpy array
            Same as red_rowandcol_H but for E.T @ H

        x0 : numpy array
            The argmax of the likelihood of a more expressive model from which we are approximating

        param_to_remove : int
            New index to remove for a further reduced model

        Returns:
            The argmax of the approximate likelihood function for a reduced model lacking param_to_remove
            and all other parameters corresponding to missing rows and columns in red_rowandcol_H
        """
        #A faster version of np.linalg.inv(E.T @ H @ E) E.T @ H @ x0
        x0_prime = np.linalg.inv(np.delete(np.delete(red_rowandcol_H, param_to_remove, axis=0), param_to_remove, axis=1)) @ np.delete(red_row_H, param_to_remove, axis=0) @ x0

        return x0_prime
    
    print('doing AMS ', comm.Get_rank())
    recompute_Hessian = False
    graph_levels = []
    target_model.param_interposer.full_span_inv_transform_matrix = target_model.param_interposer.inv_transform_matrix
    target_model.param_interposer.inv_transform_matrix_projector = np.eye(target_model.num_params)
    parent_model_projector = target_model.param_interposer.projector_matrix.copy()
    
    if rank == 0:
            print('starting GST')
    result = parallel_GST(target_model, data, prob_clip, tol, maxiter, verbosity)
    
    target_model_fit = result.estimates['GateSetTomography'].models['final iteration estimate']
    logl_fn = create_logl_obj_fn(target_model_fit, data.dataset)
    original_logl = -logl_fn.fn()
    prev_logl = original_logl

    target_model_fit.sim = pygsti.forwardsims.MapForwardSimulator(param_blk_sizes=(100,100))
    if rank == 0 : 
        print("computing Hessian")
    
    H = pygsti.tools.logl_hessian(target_model_fit, data.dataset, comm=comm, mem_limit=mem*(2**10)**3, verbosity = verbosity)
    H = comm.bcast(H, root = 0)
    
    graph_levels.append([[target_model_fit.to_vector(),original_logl, 0]])
    x0 = target_model_fit.to_vector()
    
    if graph_checkpoint != None:
        
        with open('./pickle_jar/graph_levels.pkl', 'rb') as file:
            graph_levels = pickle.load(file)
        sorted_finalists = graph_levels[-1]
        assert original_logl == graph_levels[0][0][1]
        prev_logl = original_logl
        for level in graph_levels[1:]:
            prev_logl += -level[0][1]/2
        parent_model_projector = create_projector_matrix_from_trace(graph_levels)

    exceeded_threshold = False
    red_row_H = H
    red_rowandcol_H = H
    while not exceeded_threshold:
        if rank == 0:
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
            print('total ', num_total_models)
            if size < num_total_models:
                for i in chunk_range:

                    reduced_model_projector_matrix = np.delete(parent_model_projector, i, axis=1)
                    
                    #vec = reduced_model_approx_GST(H,reduced_model_projector_matrix.T, x0)
                    vec = reduced_model_approx_GST_fast(red_rowandcol_H, red_row_H, x0, i)
                    logl_fn.model.from_vector(reduced_model_projector_matrix @ vec)

                    quantity = (prev_logl + logl_fn.fn())*2
                    if i  % 50 == 0:
                        print('Model ', i, ' has ev. ratio of ', quantity, flush=True)
                    if lowest_quantity == None or lowest_quantity > quantity:
                        lowest_quantity = quantity
                        lowest_imdl = i
                        lowest_vec = vec
                    

                left_over_start_index = size * bucket_size

                if (left_over_start_index + rank) < num_total_models:
                    i = left_over_start_index + rank
                    reduced_model_projector_matrix = np.delete(parent_model_projector, i, axis=1)     
                    vec = reduced_model_approx_GST_fast(red_rowandcol_H, red_row_H, x0, i)
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
            else:
                    bucket_size=0
            
            
            sorted_finalists = sorted(finalists, key=lambda x: x[1])

        if recompute_Hessian:
            if verbosity > 0:
                print("Recomputing Hessian, approximation error is ")
            #parent_model = graph_levels[-1][0][0]
            #GST_fit, _ = reduced_model_GST_non_plel(parent_model, data, 0, maxiter,tol, prob_clip, False, verbosity)
            #H = pygsti.tools.logl_hessian(GST_fit, data.dataset)
            #x0_model = GST_fit
            #x0 = GST_fit.to_vector()
            #reducer = np.eye(GST_fit.num_params)
            recompute_Hessian = False

        if verbosity and rank == 0:
            print(f'Model {sorted_finalists[0][2]} has lowest evidence ratio {sorted_finalists[0][1]:.4f}')
        
            
        if sorted_finalists[0][1] > er_thresh or len(sorted_finalists[0][0]) == 1:
                if rank == 0 and verbosity > 0:
                    print('All models from this level exceeded evidence ratio threshold, model rejected. Stopping!')
                #print('Final accepted model (parent from previous iteration):')
                #print('\n'.join(graph_levels[-1][0][0].parameter_labels_pretty))
                exceeded_threshold = True
                final_projector_matrix = parent_model_projector
                final_model = create_red_model(target_model.copy(), final_projector_matrix, graph_levels[-1][0][0])

                final_fit = parallel_GST(final_model.copy(), data, prob_clip, tol, maxiter, verbosity).estimates['GateSetTomography'].models['final iteration estimate']
    
                logl_fn.model.from_vector(final_projector_matrix @ final_fit.to_vector())
                curr_logl = - logl_fn.fn()
                new_quantity = 2*(prev_logl - curr_logl)
                total_ev_ratio = 2*(original_logl - curr_logl)/len(graph_levels)
                print('Evidence ratio wrt first model is ', total_ev_ratio)
                evratios = [level[0][1] for level in graph_levels[1:]]
                pre_opt = np.average(evratios)
                print('Evidence ratio pre-optimization was ', pre_opt)
                print('Evidence ratio wrt first model improved by ', pre_opt - total_ev_ratio)
                graph_levels[-1][0] = [final_fit.to_vector(), new_quantity, graph_levels[-1][0][2], graph_levels[-1][0][1]]
                if total_ev_ratio > er_thresh:
                     warnings.warn("Final model does not meet er_thresh specified. This can only happen if there is a bug, or if logl approximations were used" + str(2*(original_logl - curr_logl)/len(graph_levels)))

        else:

            parent_model_projector = np.delete(parent_model_projector, sorted_finalists[0][2], axis=1)
            red_row_H = np.delete(red_row_H, sorted_finalists[0][2], axis=0)
            red_rowandcol_H = np.delete(np.delete(red_rowandcol_H,sorted_finalists[0][2] , axis=0), sorted_finalists[0][2], axis=1)
            

        if not exceeded_threshold:
            # Save level
            graph_levels.append(sorted_finalists)
            if False:#rank == 0 and (len(graph_levels) % 20 == 0):
                with open('./pickle_jar/graph_levels.pkl', 'wb') as file:
                    pickle.dump(graph_levels, file)

        if len(graph_levels) > 1:
            prev_logl = prev_logl - graph_levels[-1][0][1]/2
            
        # Generate next level
        
        if rank == 0 and exceeded_threshold == False:
            print("next level")
            end = time.time()
            print('time this level ', end-start)
    return graph_levels

