
import os
import numpy as _np
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
import pygsti
from mpi4py import MPI
import time as _time
import datetime as _datetime
from pygsti.models import Model as _Model

class TrackedComm():
    def __init__(self, comm, parent=None):
        self.children = []
        self._comm = comm
        if parent is not None:
            parent.children.append(self)

    def Split(self, color, key):
        new_comm = self._comm.Split(color,key)
        wrap = TrackedComm(new_comm, parent=self)
        return wrap
    
    def free_all_children(self):
        for child in self.children:
            child.free_all_children()
        if self._comm is not None:
            if self._comm not in (MPI.COMM_WORLD, MPI.COMM_SELF) and self._comm != MPI.COMM_NULL:
                self._comm.Free()
             
    def __getattr__(self, name):

        return getattr(self._comm, name)
            
class AMSGreedyResult(_NicelySerializable):
    """Class for storing and saving AMS result objects comprised of a model graph trace (trace)
    and a list final evidence ratio cost (ev_ratio_cost).

    trace contains a list of lists which contain information of the models considered in the greedy AMS
    algorithm. Each level "i" contains the reduced model with the best evidence ratio at that level
    with i parameters removed from the full model:
    
    trace[i][j]
        
        i: indexes AMS levels, where each level i contains the best model found after removing i parameters

        j: indexes different characteristics for the jth best model at level i
            [param_vec , evidence_ratio, parameter_that_was_removed] 

    ev_ratio_cost contains a list of evidence ratios (floats) for the last models considered, which were all not accepted,
    due to all of them being over the error threshold specified by the user. This information is particularly useful
    to construct a results table with the function ams_results_table().
    """

    def __init__(self, trace, ev_ratio_costs=[], full_model=None):
        super().__init__()
        self.trace = trace
        self.ev_ratio_costs = ev_ratio_costs
        self.full_model = full_model
    
    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        encoded_trace = [[self._encodemx(level[0]), level[1], level[2]]for level in self.trace]
        encoded_model = self.full_model._to_nice_serialization() if self.full_model is not None else None
        state.update({
            'trace' : encoded_trace,
            'ev_ratios' : self.ev_ratio_costs,
            'full_model' : encoded_model
        })
        return state
    
    @classmethod
    def _from_nice_serialization(cls, state):
        decoded_trace = [[cls._decodemx(level[0]), level[1], level[2]]for level in state['trace']]
        model = state['full_model']
        decoded_model = _Model.from_nice_serialization(model) if model is not None else None
        return cls(decoded_trace, ev_ratio_costs=state['ev_ratios'], full_model=decoded_model)
    
    @property
    def model(self):
        if self.full_model is None:
            print('These results do not contain the full model, please set self.full_model to the correct model to obtain a reduced model.')
            return None
            
        else:
            return create_red_model(self.full_model, create_embedder_matrix_from_trace(self.trace), vec=self.trace[-1][0])



class AMSCheckpoint(_NicelySerializable):
    """
    Class for storing checkpointing intermediate progress during
    the running of an automated model selection function in order
    to enable restarting subsequent runs of the protocol from that point.
    """ 

    def __init__(self, datasetstr, er_thresh, maxiter, tol, prob_clip, H, x0, original_dlogl, time=None, path=None, graph_levels=[], prev_dlogl=None):
        super().__init__()
        self.datasetstr = datasetstr
        self.maxiter = maxiter
        self.tol = tol
        self.prob_clip = prob_clip
        self.H = H
        self.x0 = x0
        self.er_thresh = er_thresh
        self.graph_levels = graph_levels
        self.original_dlogl = original_dlogl
        self.prev_dlogl = prev_dlogl

        if time is None:
            self.last_update_time = _time.ctime()
        else:
             self.last_update_time = time

        if path is None:
            self.path = './ams_checkpoints/' + _datetime.datetime.now().strftime('%Y-%m-%d|%H.%M.%S') + '.json'
        if 'ams_checkpoints' in self.path.split('/')[0:2]:
            try:
                os.mkdir('./ams_checkpoints/')
            except FileExistsError:
                 pass
            except Exception as e:
                print(f"Error creating checkpoint folder: {e}")
    
    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        encoded_graph_levels = [[self._encodemx(level[0]), level[1], level[2]]for level in self.graph_levels]
        state.update({
                        'datasetstr' : self.datasetstr,
                        'er_thresh' : self.er_thresh,
                        'maxiter' : self.maxiter,
                        'tol' : self.tol,
                        'prob_clip' : self.prob_clip,                        
                        'H' : self._encodemx(self.H) if self.H is not None else None,
                        'x0' : self._encodemx(self.x0),
                        'time': self.last_update_time,
                        'graph_levels':encoded_graph_levels,
                        'original_dlogl': self.original_dlogl,
                        'prev_dlogl': self.prev_dlogl
                        })
                       
        return state
    @classmethod
    def _from_nice_serialization(cls, state):
        H = cls._decodemx(state['H']) if state['H'] is not None else None
        x0 = cls._decodemx(state['x0'])
        er_thresh = state['er_thresh']
        datasetstr = state['datasetstr']
        maxiter = state['maxiter']
        prob_clip = state['prob_clip']
        time= state['time']
        tol = state['tol']
        graph_levels = state['graph_levels']
        decoded_graph_levels = [[cls._decodemx(level[0]), level[1], level[2]]for level in graph_levels]
        original_dlogl = state['original_dlogl']
        prev_dlogl = state['prev_dlogl']

        return cls( datasetstr, er_thresh, maxiter,tol , prob_clip ,H, x0,original_dlogl, time=time, graph_levels=decoded_graph_levels, prev_dlogl=prev_dlogl)
    def save(self):
        """
        Saves self to memory. If path is not specified, a default one is created.

        Parameters
        ----------
        path : str
            file path to save checkpoint.
        """
        
        self.write(self.path)
        print('Checkpoint saved in ', self.path)
    @classmethod
    def load_checkpoint(cls, path):
        try:
            return cls.read(path)
        except:
            return cls.read('./ams_checkpoints/' + path)

    def checkpoint_settings(self):
        """
        Returns a list of all fast AMS settings, typically used to check if the checkpoint is valid

        Returns:
            list of full_model, datasetstr, er_thresh, maxiter, tol, prob_clip, recompute_H_thresh_percent]
        """
        return [self.full_model, self.datasetstr, self.er_thresh, self.maxiter, self.tol, self.prob_clip, self.recompute_H_thresh_percent]
         
    def check_valid_checkpoint(self, datasetstr, er_thresh, maxiter, tol, prob_clip):
        """Check if self is a checkpoint with the same settings as the ones passed in.

        Parameters
        ----------

        data : ProtocolData
            input data to fit models within AMS

        er_thresh : float
            Evidence ratio used as a stopping point for AMS

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
        recompute_H_thresh_percent : float
            TODO 

        Returns
        -------
        True iff all arguments match the data inside self.
        """


        if  datasetstr == self.datasetstr and er_thresh == self.er_thresh and maxiter == self.maxiter and tol == self.tol and prob_clip == self.prob_clip:
             return True
        else:
             return False

def remove_params(parent_model, params_to_remove):
    for param in params_to_remove:
        parent_model = remove_param(parent_model, param )
    return parent_model

def remove_param(parent_model, param_to_remove, zero = True):
    next_model = parent_model.copy()
    embedder_matrix = _np.delete(parent_model.param_interposer.embedder_matrix, param_to_remove, axis=1)
    next_model.param_interposer.transform_matrix = parent_model.param_interposer.full_span_transform_matrix @ embedder_matrix
    next_model.param_interposer.embedder_matrix = embedder_matrix
    reduced_inv_matrix = _np.delete(parent_model.param_interposer.inv_transform_matrix, param_to_remove, axis=0)
    next_model.param_interposer.inv_transform_matrix = reduced_inv_matrix
    next_model._paramvec = _np.delete(parent_model._paramvec, param_to_remove)
    next_model.from_vector(next_model._paramvec)

    next_model._need_to_rebuild = True
    next_model._clean_paramvec()
    if zero:
        next_model.from_vector(_np.zeros(next_model.num_params))
    assert next_model.num_params < parent_model.num_params
    return next_model

def create_embedder_matrix_from_trace(graph_levels, embedder_matrix = None):
    """
    Given a trace (graph_levels) representing a path through a directed model graph, where traversing to an adjacent node corresponds to
    removing a parameter, create a projector matrix that appropriately embeds a reduced vector up in the space of parent (full) model
    used to seed graph_levels.
    
    Parameters
    ----------
    
    graph_levels : a list of lists of lists 
        Let us look at every index from this object separately, call them i,k such that graph_levels[i][k]

          -  i: AMS works by traversing different "levels" of all possible reduced models. Each level considers 
                a subset of reduced models where they each lack a single parameter from a shared parent model. 
                After every model in a level is evaulated, one is picked to begin the next level. Variable 'i' 
                indexes into a specific level within AMS, with i=0 being the first level which contains only one 
                model, the seed model, and the rest of the levels contain reduced models from the level above.

          -  k: At every level of AMS, only the most important features of the best model is saved. This constitutes:

                [param_vec , evidence_ratio, parameter_that_was_removed]. k indexes within this list.

                param_vec : numpy array
                    the parameter vector of the corresponding reduced model

                evidence_ratio : float
                    2 * (log-L(parent) - log-L(this model))
                
                parameter_that_was_removed : int
                    index of parameter from parent model which is missing in this corresponding reduced model
                

        embedder_matrix (optional, defaults to None): numpy array
            If this function is used to further reduce down a projector matrix, then an initial one must be provided.
            Otherwise, it initializes to the identity.


    Returns:
        embedder_matrix : numpy array
            An identity matrix, whose columns are missing for every parameter that was removed in AMS
    """
    if embedder_matrix is None:
        embedder_matrix = _np.eye(len(graph_levels[0][0]))
    for level in graph_levels[1:]:
        param_to_remove = level[2]
        embedder_matrix = _np.delete(embedder_matrix, param_to_remove, axis=1)
    return embedder_matrix


def custom_builder(min_prob_clip):

    chi2_builder = pygsti.objectivefns.Chi2Function.builder(
        'chi2', regularization={'min_prob_clip_for_weighting': min_prob_clip}, penalties={'cptp_penalty_factor': 0.0})
    mle_builder = pygsti.objectivefns.PoissonPicDeltaLogLFunction.builder(
        'logl', regularization={'min_prob_clip': min_prob_clip, 'radius': min_prob_clip})
    iteration_builders = [chi2_builder] 
    final_builders = [mle_builder]
    builders = pygsti.protocols.GSTObjFnBuilders(iteration_builders, final_builders)
    return builders

def parallel_GST(full_model, data, builders, tol=1e-10, maxiter=300, verbosity=0, comm=None, mem_limit=None):
    """
    Wrapper to run GST with MPI with custom builders where the tolerance, probability clip and max iterations
    are easily accesible. The seed model, "target model", gets reset to its error-less version before doing
    GST. This function is specifically made to be used for FOGI AMS, but should work with non-FOGI models too.
    
    Parameters
    ----------
    
    full_model : Model
        The model to be fit to the data. All errors are eraed, yielding an ideal version of full_model
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
     
    if isinstance(maxiter, list) or isinstance(tol, list):
        if not (isinstance(maxiter, list) and isinstance(tol, list)):
            raise ValueError('Invalid optimizer settings')
        else:
            optimizers = []
            for i in range(len(maxiter)):
                optimizers.append(pygsti.optimize.customlm.CustomLMOptimizer(maxiter=maxiter[i], tol={'f':tol[i], 'relf': tol[i]}))
    else:
        optimizers = [pygsti.optimize.customlm.CustomLMOptimizer(maxiter=maxiter, tol={'f':tol, 'relf': tol})]
    protoOpt = pygsti.protocols.GateSetTomography(full_model, verbosity=verbosity, optimizer=optimizers[0], gaugeopt_suite=None, objfn_builders=builders)

    result = protoOpt.run(data, comm=comm,
                memlimit=mem_limit, optimizers=optimizers)
    return result

def create_red_model(parent_model, embedder_matrix, vec=None, sim=None):
    """TODO

    Args:
        parent_model (_type_): _description_
        embedder_matrix (_type_): _description_
        vec (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    if vec is None:
        vec = _np.zeros(embedder_matrix.shape[1])
    assert embedder_matrix.shape[1] == len(vec)
    red_model = parent_model.copy()
    red_model.param_interposer.num_params = len(vec)
    red_model.param_interposer.transform_matrix = parent_model.param_interposer.full_span_transform_matrix @ parent_model.param_interposer.embedder_matrix @ embedder_matrix
    red_model.param_interposer.embedder_matrix = parent_model.param_interposer.embedder_matrix @ embedder_matrix
    reduced_inv_matrix =  embedder_matrix.T @ parent_model.param_interposer.inv_transform_matrix
    red_model.param_interposer.inv_transform_matrix = reduced_inv_matrix
    red_model._paramvec = vec.copy()
    red_model.from_vector(red_model._paramvec)

    red_model._need_to_rebuild = True
    red_model._clean_paramvec()
    assert red_model.num_params < parent_model.num_params
    if sim is not None:
        red_model.sim = sim
    return red_model

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
        version of x, where some entries are pinned to 0. To apply this constrain, we change the i_nput
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
        #A faster version of _np.linalg.inv(E.T @ H @ E) E.T @ H @ x0
        x0_prime = _np.linalg.inv(_np.delete(_np.delete(red_rowandcol_H, param_to_remove, axis=0), param_to_remove, axis=1)) @ _np.delete(red_row_H, param_to_remove, axis=0) @ x0

        return x0_prime

def compare_parameters_simple(parent_model_vec, red_model_vec, embedder_matrix):
    """TODO: docstring

    Args:
        parent_model_vec (_type_): _description_
        red_model_vec (_type_): _description_
        embedder_matrix (_type_): _description_
    """
    projector = embedder_matrix @ embedder_matrix.T
    assert len(parent_model_vec) == len(projector)
    table_data = [['Full', 'Reduced']]
    j = 0
    for i in range(len(parent_model_vec)):
          
        if projector[i][i] == 0:
            table_data.append([parent_model_vec[i], 'removed'])
        else:
            table_data.append([parent_model_vec[i], red_model_vec[j]])
            j += 1
    assert len(red_model_vec) == j
    for row in table_data:
        print("{: <25} {: <25}".format(*row), '\n')

def ams_results_table(trace, ev_ratio_costs, extra_column=None):
    reducer = create_embedder_matrix_from_trace(trace).T
    projector = reducer.T @ reducer
    #embedded_costs = reducer.T @ ev_ratio_costs
    parent_model_vec = trace[0][0]
    red_model_vec = trace[-1][0]

    assert len(parent_model_vec) == len(projector)
    table_data = [['Full', 'Reduced', 'Cost']]
    j = 0
    for i in range(len(parent_model_vec)):
          
        if projector[i][i] == 0:
            table_data.append([parent_model_vec[i], 'removed', 'N/A'])
        else:
            table_data.append([parent_model_vec[i], red_model_vec[j], ev_ratio_costs[j]])
            j += 1
    assert len(red_model_vec) == j
    for i,row in enumerate(table_data):
        if extra_column is not None:
            print("{: <25} {: <25} {: <25} {: <25}".format(*row, extra_column[i]), '\n')
        else:
            print("{: <25} {: <25} {: <25}".format(*row), '\n')


def create_approx_logl_fn(H, x0, initial_logl):
    constant_term = x0.T @ H @ x0 + initial_logl
    def approx_logl(red_row_H, red_rowandcol_H, param_to_remove):
        red_rowandcol_H = _np.delete(_np.delete(red_rowandcol_H, param_to_remove, axis=0), param_to_remove, axis=1)
        red_row_H = _np.delete(red_row_H, param_to_remove, axis=0)
        return constant_term - x0.T @ red_row_H.T @ _np.linalg.inv(red_rowandcol_H) @ red_row_H @ x0
    
    return approx_logl

