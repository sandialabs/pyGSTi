
import os

#TODO where should these env variables go?
thread_limit = '1'
os.environ['OPENBLAS_NUM_THREADS'] = thread_limit
os.environ['GOTO_NUM_THREADS'] = thread_limit
os.environ['OMP_NUM_THREADS'] = thread_limit
os.environ['NUMEXPR_NUM_THREADS'] = thread_limit
os.environ['VECLIB_MAXIMUM_THREADS'] = thread_limit
os.environ['MKL_NUM_THREADS'] = thread_limit

import numpy as _np
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
import pygsti
import mpi4py as MPI
import time as _time
import datetime as _datetime
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
mem = 7
mem_limit = mem*(2**10)**3

class AMSCheckpoint(_NicelySerializable):
    """
    Class for storing checkpointing intermediate progress during
    the running of an automated model selection function in order
    to enable restarting subsequent runs of the protocol from that point.

    Parameters
    ----------
    TODO
    """ 

    def __init__(self, target_model, data, er_thresh, maxiter, tol, prob_clip, recompute_H_thresh_percent, H, x0, time = None):

        self.target_model = target_model
        self.datastr = data.to_str()
        self.er_thresh = er_thresh
        self.maxiter = maxiter
        self.tol = tol
        self.prob_clip = prob_clip
        self.recompute_H_thresh_percent = recompute_H_thresh_percent
        self.H = H
        self.x0 = x0
        if time is None:
            self.last_update_time = _time.ctime()
        else:
             self.last_update_time = time
    
    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'target_model' : self.target_model._to_nice_serialization(),
                        'datastr' : self.datastr,
                        'er_thresh' : self.er_thresh,
                        'maxiter' : self.maxiter,
                        'tol' : self.tol,
                        'prob_clip' : self.prob_clip,
                        'recompute_H_thresh_percent' : self.recompute_H_thresh_percent,
                        'H' : self._encodemx(self.H),
                        'x0' : self._encodemx(self.x0),
                        'time': self.last_update_time})
                       
        return state
    @classmethod
    def from_nice_serialization(cls, state):
        return cls(state['target_model'], state['datastr'], state['er_thresh'], state['maxiter'], state['tol'], state['prob_clip'], state['recompute_H_thresh_percent'] ,cls._decodemx(state['H']), cls._decodemx(state['x0'], state['time']))
    def save(self, path='None'):
        if path is None:
            path = 'AMSCheckpoint-' + _datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.write(path)
    def check_valid_checkpoint(self, target_model, data, er_thresh, maxiter, tol, prob_clip, recompute_H_thresh_percent):
        """Check if self is a checkpoint with the same settings as the ones passed in.

        Parameters
        ----------
        target_model : Model
            Model used as a seed for AMS

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

        if target_model == self.target_model and data.to_str() == self.datastr and er_thresh == self.er_thresh and maxiter == self.maxiter and tol == self.tol and prob_clip == self.prob_clip and recompute_H_thresh_percent == self.recompute_H_thresh_percent:
             return True
        else:
             return False
    
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

def compare_parameters_simple(parent_model_vec, red_model_vec, projector_matrix):
    """TODO: docstring

    Args:
        parent_model_vec (_type_): _description_
        red_model_vec (_type_): _description_
        projector_matrix (_type_): _description_
    """
    projector = projector_matrix @ projector_matrix.T
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