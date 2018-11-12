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
        """
        self.outcomes = outcomes
        self.numoutcomes = len(outcomes)
        self.set_basisfunctions(modeltype, hyperparameters, parameters)

        return None        
    
    def set_basisfunctions(self, modeltype, hyperparameters, parameters):

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
                endtime = hyperparameters['endtime']
                timedif = endtime - starttime              
            except:
                raise ValueError("The hyperparameters are invalid for the model type! Need the start time and end time to creat the basis functions!")
            
            self.fullmodelsize = _np.inf         
            def basisfunction(i, times):
                return _np.array([_np.cos(i*_np.pi*(t-starttime+0.5)/timedif) for t in times])

        else:
            raise ValueError("Invalid model type!")

        self.basisfunction = basisfunction
        # This sets the hyperparameters and parameters so that they agree with the new model.
        self.set_hyperparameters(hyperparameters, parameters)

        return None

    def get_basis_function(i, times):
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
   
    def get_probabilities(self, times):
        """
        todo
        """ 
        return {o : _np.sum(_np.array([self.parameters[o][ind]*self.basisfunction(i,times) for ind, i in enumerate(self.basisfunctionInds)]), axis=0) for o in self.outcomes}    

    #def get_probabilities(times):
    #               
    #   return {o : [_np.sum(_np.array([self.parameters[o][0] + self.parameters[o][i]*_np.cos(freqIns[i]*_np.pi*(t-starttime+0.5)/timedif) for i in range(self.dof)])) for t in times]}    

    def copy(self):

        return _copy.deepcopy(self)