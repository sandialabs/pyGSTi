"""Core routines for detecting and characterizing drift with time-stamped data"""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

#from . import signal as _sig

import numpy as _np
import warnings as _warnings
import copy as _copy

class ProbabilityTrajectoryModel(object):
    """
    todo
    """
    def __init__(self, outcomes, modeltype, hyperparameters, parameters):
        """
        todo:

        Parameters
        ----------

        modetype : str
            The type of basis function for the model.

        hyperparameters : 

        parameters :
        
        Returns
        -------
        None
        """
        self.outcomes = outcomes
        self.independent_outcomes = outcomes[:-1]
        self.constrained_outcome = outcomes[-1]
        self.numoutcomes = len(outcomes)
        self.set_basisfunctions(modeltype, hyperparameters, parameters)

        return None        
    
    def set_basisfunctions(self, modeltype, hyperparameters, parameters):
        """
        Todo:
        """
        self.modeltype = modeltype

        if modeltype == 'null':
            
            self.fullmodelsize = 1            
            def basisfunction(i, times):
                if i < self.fullmodelsize:
                    return _np.array([1 for t in times]) #_np.cos(freqIns[i]*_np.pi*(t-starttime+0.5)/timedif)
                else:
                    raise ValueError("Invalid basis function index!")
        
        elif modeltype == 'DCT':

            try:
                starttime = hyperparameters['starttime']
                timestep = hyperparameters['timestep']
                numsteps = hyperparameters['numsteps']
                #timedif = endtime - starttime              
            except:
                raise ValueError("The hyperparameters are invalid for the model type! Need the start time and end time to creat the basis functions!")
            
            self.fullmodelsize = _np.inf         
            def basisfunction(i, times):
                #return _np.array([_np.cos(i*_np.pi*(t-starttime+0.5)/timedif) for t in times])
                return _np.array([_np.cos(i*_np.pi*((t-starttime)/timestep+0.5)/numsteps) for t in times])
        else:
            raise ValueError("Invalid model type!")

        self.basisfunction = basisfunction
        # This sets the hyperparameters and parameters so that they agree with the new model.
        self.set_hyperparameters(hyperparameters, parameters)

        return None

    def get_basis_function(self, i, times):
        """
        todo.
        """
        return self.basisfunction(i,times)

    def set_hyperparameters(self, hyperparameters, parameters):
        """
        todo
        """
        basisfunctionInds = list(hyperparameters['basisfunctionInds'])
        basisfunctionInds.sort()
        assert(max(basisfunctionInds) < self.fullmodelsize)
        assert(0. in basisfunctionInds), "The constant function must be one of the basis functions!"
        self.basisfunctionInds = basisfunctionInds
        self.set_parameters(parameters)
        self.modelsize_unconstrained = (self.numoutcomes-1)*len(basisfunctionInds)
        self.modelsize_unconstrained_peroutcome= len(basisfunctionInds)
        return None

    def set_parameters(self, parameters):
        """
        todo
        """
        self.parameters = parameters
        return None 

    def set_parameters_from_list(self, parameterslist):
        """
        todo:
        """
        numbasisfuncs = len(self.basisfunctionInds)
        parameters = {o : parameterslist[oind:(1+oind)*numbasisfuncs] for oind,o in enumerate(self.independent_outcomes)}
        self.parameters = parameters
        return None

    def get_parameters_as_list(self):
        """
        todo:
        """
        parameterslist = []
        for o in self.independent_outcomes:
            parameterslist += self.parameters[o]
        return parameterslist

    def get_parameters(self):
        """
        todo:
        """
        return self.parameters
   
    def get_probabilities(self, times):
        """
        todo
        """ 
        p_for_constrained_outcome = _np.ones(len(times))
        for o in self.independent_outcomes:
            p = _np.sum(_np.array([self.parameters[o][ind]*self.basisfunction(i,times) for ind, i in enumerate(self.basisfunctionInds)]), axis=0) 
            p_for_constrained_outcome = p_for_constrained_outcome - p
            probs = {o : p}    
        probs[self.constrained_outcome] = p_for_constrained_outcome 
        return probs

    def get_independent_probabilities(self, times):
        """
        todo
        """
        return {o : _np.sum(_np.array([self.parameters[o][ind]*self.basisfunction(i,times) for ind, i in enumerate(self.basisfunctionInds)]), axis=0) for o in self.independent_outcomes}

    #def get_probabilities(times):
    #               
    #   return {o : [_np.sum(_np.array([self.parameters[o][0] + self.parameters[o][i]*_np.cos(freqIns[i]*_np.pi*(t-starttime+0.5)/timedif) for i in range(self.dof)])) for t in times]}    

    def copy(self):
        return _copy.deepcopy(self)