"""
Defines the various forward simulators for use with models containing operations with
fluctuating Hamiltonian parameters.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import numpy as _np
import collections as _collections
import itertools as _itertools
from pygsti.modelmembers.operations import LindbladErrorgen as _LindbladErrorgen
from pygsti.forwardsims import WeakForwardSimulator as _WeakForwardsimulator
from pygsti.forwardsims import MapForwardSimulator as _MapForwardSimulator
from pygsti.forwardsims import SimpleMapForwardSimulator as _SimpleMapForwardSimulator
from pygsti.forwardsims import MatrixForwardSimulator as _MatrixForwardSimulator
from pygsti.evotypes import Evotype as _Evotype
from pygsti.extras.lfh.lfherrorgen import LFHLindbladErrorgen as _LFHLindbladErrorgen
import pygsti.tools.slicetools as _slct


from pygsti.forwardsims import ForwardSimulator as _ForwardSimulator
from pygsti.models import ExplicitOpModel as _ExplicitOpModel
from pygsti.modelmembers.operations import ExpErrorgenOp as _ExpErrorgenOp
from pygsti.modelmembers.operations import ComposedOp as _ComposedOp
from pygsti.baseobjs import statespace as _statespace
from pygsti.baseobjs.basis import Basis as _Basis, BuiltinBasis as _BuiltinBasis
from pygsti.baseobjs.errorgenlabel import LocalElementaryErrorgenLabel as _LocalElementaryErrorgenLabel
from pygsti.modelmembers.operations import LindbladParameterization
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as _LindbladCoefficientBlock

from scipy.special import roots_hermite
from math import sqrt, pi

#Next we need to define a new custom weak forward simulator
class LFHWeakForwardSimulator(_ForwardSimulator):
    """
    Weak forward simulator specialized for dealing with low-frequency hamiltonian models.
    """
    
    def __init__(self, shots, model=None, base_seed=None):
        """
        Construct a new WeakForwardSimulator object.

        Parameters
        ----------
        shots: int
            Number of times to run each circuit to obtain an approximate probability
        model : Model
            Optional parent Model to be stored with the Simulator
        """
        self.shots = shots        
        super().__init__(model)

    def bulk_probs(self, circuits, clip_to=None, resource_alloc=None, smartc=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[circuit]` is an ordered dictionary of
            outcome probabilities whose keys are outcome labels.
        """
        
        #We want to loop through each of the circuits in a "rasterization pass" collecting one
        #one shot each. At the start of each loop we want to resample the randomly fluctuating
        #hamiltonian parameters.
        #We should be able to farm out the probability calculation to another forward simulator
        #though.
        probs_for_shot = []
        for i in range(self.shots):
            #Have the model resample the hamiltonian rates:
            self.model.sample_hamiltonian_rates()
            helper_sim = _MapForwardSimulator(model=self.model)
            
            #Now that we've sampled the hamiltonian rates calculate the probabilities for
            #all of the circuits.
            #import pdb
            #pdb.set_trace()
            probs_for_shot.append(helper_sim.bulk_probs(circuits))
        #Now loop through and perform an averaging over the output probabilities.
        #Initialize a dictionary for storing the final results.
        #print(probs_for_shot)
        outcome_labels= probs_for_shot[0][circuits[0]].keys()
        averaged_probs = {ckt:{lbl:0 for lbl in outcome_labels}  for ckt in circuits}    
        
        for prob_dict in probs_for_shot:
            for ckt in circuits:
                for lbl in outcome_labels:
                    averaged_probs[ckt][lbl] += prob_dict[ckt][lbl]/self.shots
                    
        #return the averaged probabilities:
        return averaged_probs

    def bulk_dprobs(self, circuits, resource_alloc=None, smartc=None):
        """
        Construct a dictionary containing the probability derivatives for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that `dprobs[circuit]` is an ordered dictionary of
            derivative arrays (one element per differentiated parameter) whose
            keys are outcome labels
        """
        
        #If _compute_circuit_outcome_probability_derivatives is implemented, use it!
        #resource_alloc = layout.resource_alloc()

        eps = 1e-7  # hardcoded?
#         if param_slice is None:
#             param_slice = slice(0, self.model.num_params)
#         param_indices = _slct.to_array(param_slice)

#         if dest_param_slice is None:
#             dest_param_slice = slice(0, len(param_indices))
#         dest_param_indices = _slct.to_array(dest_param_slice)

#         iParamToFinal = {i: dest_param_indices[ii] for ii, i in enumerate(param_indices)}

        probs = self.bulk_probs(circuits)
        orig_vec = self.model.to_vector().copy()
        
        #pull out the requisite outcome labels:
        outcome_labels= probs[circuits[0]].keys()
        
        #initialize a dprobs array:
        dprobs= {ckt: {lbl: _np.empty(self.model.num_params, dtype= _np.double) for lbl in outcome_labels} for ckt in circuits}
        
        for i in range(self.model.num_params):
            vec = orig_vec.copy()
            vec[i] += eps
            self.model.from_vector(vec, close=True)
            probs2 = self.bulk_probs(circuits)
            
            #need to parse this and construct the corresponding entries of the dprobs dict.
        
            for ckt in circuits:
                for lbl in outcome_labels:
                    dprobs[ckt][lbl][i] = (probs2[ckt][lbl] - probs[ckt][lbl]) / eps
            
        #restore the model to it's original value
        self.model.from_vector(orig_vec, close=True)
        
        return dprobs
    

    #Try out a different "weak" forward simulator that doesn't use sampling to do the integration
#over the gaussian, but rather approximates the expectation values using gauss-hermite quadrature
class LFHIntegratingForwardSimulator(_ForwardSimulator):
    """
    Weak forward simulator specialized for dealing with low-frequency hamiltonian models.
    """
    
    def __init__(self, order, model=None, base_seed=None):
        """
        Construct a new WeakForwardSimulator object.

        Parameters
        ----------
        order: int
            order of the gauss-hermite approximation for the integral.
        model : Model
            Optional parent Model to be stored with the Simulator
        """
        self.order = order
        self.helper_sim = None
        super().__init__(model)
        
    def build_sampling_grid(self):
        #build the grid of sample points and weights
        #for the simulators model.
        
        #Need to identify how many deviation parameters there are.
        num_deviances= 0
        dev_values= []
        mean_values = []
        for op in self.model.operations.values():
            if isinstance(op, _ComposedOp):
                for subop in op.factorops:
                    if isinstance(subop, _ExpErrorgenOp):
                        if isinstance(subop.errorgen, _LFHLindbladErrorgen):
                            dev_values.extend(subop.errorgen.devs)
                            mean_values.extend(subop.errorgen.means)
                            num_deviances += len(subop.errorgen.devs)
        
        #Once we know the number of deviances and their values we can start building 
        #out the grid of sampling points and weights.
        base_one_d_points , base_one_d_weights= roots_hermite(self.order)
        
        #print(base_one_d_points)
        #print(base_one_d_weights) 
        
        #print(mean_values)
        #print(dev_values)
        
        #The weights remain the same, but I need to modify the sampling points
        #Now I need to get updates
        gaussian_one_d_points = [[] for _ in range(len(dev_values))]
        
        for i,(dev, mean) in enumerate(zip(dev_values, mean_values)):
            for point in base_one_d_points:
                gaussian_one_d_points[i].append(mean+sqrt(2)*dev*point)
                
        #print(gaussian_one_d_points[0])
        
        return gaussian_one_d_points, (1/sqrt(pi))*base_one_d_weights
        

    def bulk_probs(self, circuits, clip_to=None, resource_alloc=None, smartc=None, return_layout= False, cached_layout= None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[circuit]` is an ordered dictionary of
            outcome probabilities whose keys are outcome labels.
        """
        
        sample_points_lists , weights = self.build_sampling_grid()
        
        #The grid of points is the cartesian product of the sample point lists.
        
        sample_point_grid = _itertools.product(*sample_points_lists)
        
        #do this for convienience
        weight_grid = _itertools.product(*([weights]*len(sample_points_lists)))
            
        #I need to identify where in the model vector the sampled hamiltonian weights
        #need to go.
        hamiltonian_model_indices = []
        for op in self.model.operations.values():
            if isinstance(op, _ComposedOp):
                for subop in op.factorops:
                    if isinstance(subop, _ExpErrorgenOp):
                        if isinstance(subop.errorgen, _LFHLindbladErrorgen):
                            hamiltonian_model_indices.extend(list(range(op.gpindices.start, op.gpindices.start+ len(subop.errorgen.means)))) 
        
        orig_vec = self.model.to_vector()
        
        if self.helper_sim is None:
            self.add_helper_sim()
        
        #create a circuit layout that we can reuse to speed things up
        #(We'll be using the same circuit list at every evaluation point)
        if cached_layout is None:
            ckt_layout = self.helper_sim.create_layout(circuits)
        else:
            ckt_layout = cached_layout
        
        weighted_probs_for_point = []
        
        for sample_grid_point, weight_grid_point in zip(sample_point_grid, weight_grid):
            vec = orig_vec.copy()
            vec[hamiltonian_model_indices] = _np.array(sample_grid_point)
            
            #despite storing it as a grid, we just need the scalar product of the weights
            weight_value = _np.prod(weight_grid_point)
            
            #set the model to this current vec value
            self.model.from_vector(vec)
            
            #next simulate the model using the helper simulator:
            #We can pass in a COPAlayout for this instead of a list of circuits, which speeds things up.
            probs_for_point = self.helper_sim.bulk_probs(ckt_layout)
            #probs_for_point = helper_sim.bulk_probs(circuits)
            
            #print(probs_for_point)
            
            #Iterate through and add weight terms.
            outcome_labels= probs_for_point[circuits[0]].keys()
            weighted_probs = {ckt:{lbl:0 for lbl in outcome_labels}  for ckt in circuits}    
        
            for ckt in circuits:
                for lbl in outcome_labels:
                    weighted_probs[ckt][lbl] = probs_for_point[ckt][lbl] * weight_value
            
            weighted_probs_for_point.append(weighted_probs)
            
        #reset the model to it's original value
        self.model.from_vector(orig_vec)
        
        #print(len(weighted_probs_for_point))
        
        #Aggregate all of the probability values into a final_result
        averaged_probs = {ckt:{lbl:0 for lbl in outcome_labels}  for ckt in circuits}    
        
        for prob_dict in weighted_probs_for_point:
            for ckt in circuits:
                for lbl in outcome_labels:
                    averaged_probs[ckt][lbl] += prob_dict[ckt][lbl]
                    
        #return the averaged probabilities:
        if return_layout:
            return averaged_probs, ckt_layout
        else:
            return averaged_probs

    def bulk_dprobs(self, circuits, resource_alloc=None, smartc=None):
        """
        Construct a dictionary containing the probability derivatives for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that `dprobs[circuit]` is an ordered dictionary of
            derivative arrays (one element per differentiated parameter) whose
            keys are outcome labels
        """
        
        #If _compute_circuit_outcome_probability_derivatives is implemented, use it!
        #resource_alloc = layout.resource_alloc()

        eps = 1e-7  # hardcoded?
#         if param_slice is None:
#             param_slice = slice(0, self.model.num_params)
#         param_indices = _slct.to_array(param_slice)

#         if dest_param_slice is None:
#             dest_param_slice = slice(0, len(param_indices))
#         dest_param_indices = _slct.to_array(dest_param_slice)

#         iParamToFinal = {i: dest_param_indices[ii] for ii, i in enumerate(param_indices)}

        probs, ckt_layout = self.bulk_probs(circuits, return_layout= True)
        orig_vec = self.model.to_vector().copy()
        
        #pull out the requisite outcome labels:
        outcome_labels= probs[circuits[0]].keys()
        
        #initialize a dprobs array:
        dprobs= {ckt: {lbl: _np.empty(self.model.num_params, dtype= _np.double) for lbl in outcome_labels} for ckt in circuits}
        
        for i in range(self.model.num_params):
            vec = orig_vec.copy()
            vec[i] += eps
            self.model.from_vector(vec, close=True)
            probs2 = self.bulk_probs(circuits, cached_layout= ckt_layout)
            
            #need to parse this and construct the corresponding entries of the dprobs dict.
        
            for ckt in circuits:
                for lbl in outcome_labels:
                    dprobs[ckt][lbl][i] = (probs2[ckt][lbl] - probs[ckt][lbl]) / eps
            
        #restore the model to it's original value
        self.model.from_vector(orig_vec)
        
        return dprobs
    
    def add_helper_sim(self):
        if self.model is not None:
            self.helper_sim = _MatrixForwardSimulator(model=self.model)
    
    def create_layout(self, bulk_circuit_list, dataset=None, resource_alloc=None,
                      array_types=(), verbosity=1):
        
        if self.helper_sim is None:
            self.add_helper_sim()
        
        return self.helper_sim.create_layout(bulk_circuit_list, dataset, resource_alloc, 
                                             array_types, verbosity=verbosity)
    
    #Add a bulk_fill_probs method that does something similar to bulk_probs but returns
    #an array filled according to a layout instead of an outcome dictionary
    def bulk_fill_probs(self, array_to_fill, layout):
        
        sample_points_lists , weights = self.build_sampling_grid()
        
        #The grid of points is the cartesian product of the sample point lists.
        
        sample_point_grid = list(_itertools.product(*sample_points_lists))
        
        #do this for convienience
        weight_grid = list(_itertools.product(*([weights]*len(sample_points_lists))))
        
        #I need to identify where in the model vector the sampled hamiltonian weights
        #need to go.
        hamiltonian_model_indices = []
        for op in self.model.operations.values():
            if isinstance(op, _ComposedOp):
                for subop in op.factorops:
                    if isinstance(subop, _ExpErrorgenOp):
                        if isinstance(subop.errorgen, _LFHLindbladErrorgen):
                            hamiltonian_model_indices.extend(list(range(op.gpindices.start, op.gpindices.start+ len(subop.errorgen.means)))) 
        
        orig_vec = self.model.to_vector()
        
        #If I have a layout then I should have a helper sim by this point
        #if self.helper_sim is None:
        #    self.add_helper_sim()
        
        #create copies of the array being filled
        temp_arrays = [array_to_fill.copy() for _ in sample_point_grid]
        
        for i, (sample_grid_point, weight_grid_point) in enumerate(zip(sample_point_grid, weight_grid)):
            
            vec = orig_vec.copy()
            vec[hamiltonian_model_indices] = _np.array(sample_grid_point)
            
            #despite storing it as a grid, we just need the scalar product of the weights
            weight_value = _np.prod(weight_grid_point)
            
            #set the model to this current vec value
            self.model.from_vector(vec)
            
            #next simulate the model using the helper simulator:
            self.helper_sim.bulk_fill_probs(temp_arrays[i], layout)
            
            #Iterate through and add weight terms. 
            temp_arrays[i] = weight_value*temp_arrays[i] 
            
        #reset the model to it's original value
        self.model.from_vector(orig_vec)
        
        #Aggregate all of the probability values into a final_result
        averaged_array = temp_arrays[0]
        for temp_array in temp_arrays[1:]:
            averaged_array += temp_array
            
        #print('averaged: ', averaged_array)
        
        array_to_fill[:]= averaged_array
        #return averaged_array
        
        
    #Next I need a version of bulk_fill_dprobs:
        
    def bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill=None):

        eps = 1e-7  # hardcoded?

        if pr_array_to_fill is not None:
            self.bulk_fill_probs(pr_array_to_fill, layout)
            probs = pr_array_to_fill.copy()
        
        else:
            probs = layout.allocate_local_array('e', 'd')
            self.bulk_fill_probs(probs, layout)
        
        orig_vec = self.model.to_vector().copy()
        
        for i in range(self.model.num_params):
            probs2 = probs.copy()
            vec = orig_vec.copy()
            vec[i] += eps
            self.model.from_vector(vec, close=True)
            self.bulk_fill_probs(probs2,layout)
            
            #now put this result into the array to be filled.
            array_to_fill[: , i] =(probs2 - probs) / eps
            
        #restore the model to it's original value
        self.model.from_vector(orig_vec)
        
        #print('dprobs: ', array_to_fill)
        #return dprobs
        
class LFHSigmaForwardSimulator(_ForwardSimulator):
    """
    Weak forward simulator specialized for dealing with low-frequency hamiltonian models.
    This version uses sigma point methods (unscented transform) to approximate the requisite
    integrals.
    """
    
    def __init__(self, model=None):
        """
        Construct a new WeakForwardSimulator object.

        Parameters
        ----------
        order: int
            order of the gauss-hermite approximation for the integral.
        model : Model
            Optional parent Model to be stored with the Simulator
        """
        self.helper_sim = None
        super().__init__(model)
        
    def sigma_points(self):
        #build the grid of sample points and weights
        #for the simulators model.
        
        #Need to identify how many deviation parameters there are.
        num_deviances= 0
        dev_values= []
        mean_values = []
        for op in self.model.operations.values():
            if isinstance(op, _ComposedOp):
                for subop in op.factorops:
                    if isinstance(subop, _ExpErrorgenOp):
                        if isinstance(subop.errorgen, _LFHLindbladErrorgen):
                            dev_values.extend(subop.errorgen.devs)
                            mean_values.extend([subop.errorgen.means[i] for i in subop.errorgen.dev_dict.keys()])
                            num_deviances += len(subop.errorgen.devs)

        #Now construct the set of points and weights:
        mean_vec = _np.array(mean_values).reshape((num_deviances,1))
        std_vec = _np.array(dev_values)

        #Currently only have _LFHLindbladErrorgen objects that are configured for
        #diagonal covariances, so we can simplify the sigma point construction logic
        #a bit. Use a heuristic from Julier and Uhlmann.
        #The first sigma point is just the mean.
        #columns of this matrix will become sigma vectors.
        sigma_vec_array = _np.repeat(mean_vec, repeats=2*num_deviances+1, axis=1)

        #calculate a special scaling factor used in the Unscented transform.
        #This scale factor is n + kappa in Julier and Uhlmann, and they claim
        #a value of n+kappa =3 is a good heuristic for gaussian distributions.
        scale_factor = 3
        #columns of offsets correspond to the offset vectors
        offsets = _np.diag(_np.sqrt(scale_factor)*std_vec)
        #Note: the application of these shifts can be done much more efficiently
        #by appropriately using slicing and broadcasting, but this is easy for now.
        shifts = _np.concatenate([_np.zeros_like(mean_vec), offsets, -offsets], axis=1)
        #Add these offsets to columns 1 to L and subtract from
        #columns L+1 to 2L+1
        sigma_vec_array += shifts

        #next we need the weights
        kappa = scale_factor - num_deviances
        weights = _np.array([kappa/scale_factor, 1/(2*scale_factor)])

        return sigma_vec_array, weights

        

    def bulk_probs(self, circuits, clip_to=None, resource_alloc=None, smartc=None, return_layout= False, cached_layout= None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[circuit]` is an ordered dictionary of
            outcome probabilities whose keys are outcome labels.
        """
        
        sigma_points , weights = self.sigma_points()
        
        #I need to identify where in the model vector the sampled hamiltonian weights
        #need to go. I should probably make this something that gets cached, as it usually
        #won't need recomputation.
        hamiltonian_model_indices = []
        for op in self.model.operations.values():
            if isinstance(op, _ComposedOp):
                for subop in op.factorops:
                    if isinstance(subop, _ExpErrorgenOp):
                        if isinstance(subop.errorgen, _LFHLindbladErrorgen):
                            hamiltonian_model_indices.extend([op.gpindices.start+i for i in subop.errorgen.dev_dict.keys()])
                            #hamiltonian_model_indices.extend(list(range(op.gpindices.start, op.gpindices.start+ len(subop.errorgen.means)))) 
        
        orig_vec = self.model.to_vector()
        
        if self.helper_sim is None:
            self.add_helper_sim()
        
        #create a circuit layout that we can reuse to speed things up
        #(We'll be using the same circuit list at every evaluation point)
        if cached_layout is None:
            ckt_layout = self.helper_sim.create_layout(circuits)
        else:
            ckt_layout = cached_layout
        
        weighted_probs_for_point = []
        weight_iter = _itertools.chain([0] ,_itertools.repeat(1, sigma_points.shape[1]-1))
        for i, j in zip(range(sigma_points.shape[1]), weight_iter):
            vec = orig_vec.copy()
            vec[hamiltonian_model_indices] = sigma_points[:,i]
            
            #set the model to this current vec value
            self.model.from_vector(vec)
            
            #next simulate the model using the helper simulator:
            #We can pass in a COPAlayout for this instead of a list of circuits, which speeds things up.
            probs_for_point = self.helper_sim.bulk_probs(ckt_layout)
            #probs_for_point = helper_sim.bulk_probs(circuits)
            
            #print(probs_for_point)
            
            #Iterate through and add weight terms.
            outcome_labels= probs_for_point[circuits[0]].keys()
            weighted_probs = {ckt:{lbl:0 for lbl in outcome_labels}  for ckt in circuits}    
        
            for ckt in circuits:
                for lbl in outcome_labels:
                    weighted_probs[ckt][lbl] = probs_for_point[ckt][lbl] * weights[j]
            
            weighted_probs_for_point.append(weighted_probs)
            
        #reset the model to it's original value
        self.model.from_vector(orig_vec)
        
        #print(len(weighted_probs_for_point))
        
        #Aggregate all of the probability values into a final_result
        averaged_probs = {ckt:{lbl:0 for lbl in outcome_labels}  for ckt in circuits}    
        
        for prob_dict in weighted_probs_for_point:
            for ckt in circuits:
                for lbl in outcome_labels:
                    averaged_probs[ckt][lbl] += prob_dict[ckt][lbl]
                    
        #return the averaged probabilities:
        if return_layout:
            return averaged_probs, ckt_layout
        else:
            return averaged_probs

    def bulk_dprobs(self, circuits, resource_alloc=None, smartc=None):
        """
        Construct a dictionary containing the probability derivatives for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        resource_alloc : ResourceAllocation, optional
            A resource allocation object describing the available resources and a strategy
            for partitioning them.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that `dprobs[circuit]` is an ordered dictionary of
            derivative arrays (one element per differentiated parameter) whose
            keys are outcome labels
        """
        
        #If _compute_circuit_outcome_probability_derivatives is implemented, use it!
        #resource_alloc = layout.resource_alloc()

        eps = 1e-7  # hardcoded?
#         if param_slice is None:
#             param_slice = slice(0, self.model.num_params)
#         param_indices = _slct.to_array(param_slice)

#         if dest_param_slice is None:
#             dest_param_slice = slice(0, len(param_indices))
#         dest_param_indices = _slct.to_array(dest_param_slice)

#         iParamToFinal = {i: dest_param_indices[ii] for ii, i in enumerate(param_indices)}

        probs, ckt_layout = self.bulk_probs(circuits, return_layout= True)
        orig_vec = self.model.to_vector().copy()
        
        #pull out the requisite outcome labels:
        outcome_labels= probs[circuits[0]].keys()
        
        #initialize a dprobs array:
        dprobs= {ckt: {lbl: _np.empty(self.model.num_params, dtype= _np.double) for lbl in outcome_labels} for ckt in circuits}
        
        for i in range(self.model.num_params):
            vec = orig_vec.copy()
            vec[i] += eps
            self.model.from_vector(vec, close=True)
            probs2 = self.bulk_probs(circuits, cached_layout= ckt_layout)
            
            #need to parse this and construct the corresponding entries of the dprobs dict.
        
            for ckt in circuits:
                for lbl in outcome_labels:
                    dprobs[ckt][lbl][i] = (probs2[ckt][lbl] - probs[ckt][lbl]) / eps
            
        #restore the model to it's original value
        self.model.from_vector(orig_vec)
        
        return dprobs
    
    def add_helper_sim(self):
        if self.model is not None:
            self.helper_sim = _MatrixForwardSimulator(model=self.model)
    
    def create_layout(self, bulk_circuit_list, dataset=None, resource_alloc=None,
                      array_types=(), verbosity=1):
        
        if self.helper_sim is None:
            self.add_helper_sim()
        
        return self.helper_sim.create_layout(bulk_circuit_list, dataset, resource_alloc, 
                                             array_types, verbosity=verbosity)
    
    #Add a bulk_fill_probs method that does something similar to bulk_probs but returns
    #an array filled according to a layout instead of an outcome dictionary
    def bulk_fill_probs(self, array_to_fill, layout):
        
        sigma_points , weights = self.sigma_points()
        
        #I need to identify where in the model vector the sampled hamiltonian weights
        #need to go.
        hamiltonian_model_indices = []
        for op in self.model.operations.values():
            if isinstance(op, _ComposedOp):
                for subop in op.factorops:
                    if isinstance(subop, _ExpErrorgenOp):
                        if isinstance(subop.errorgen, _LFHLindbladErrorgen):
                            hamiltonian_model_indices.extend([op.gpindices.start+i for i in subop.errorgen.dev_dict.keys()])
                            #hamiltonian_model_indices.extend(list(range(op.gpindices.start, op.gpindices.start+ len(subop.errorgen.means)))) 
        
        orig_vec = self.model.to_vector()
        
        #If I have a layout then I should have a helper sim by this point
        #if self.helper_sim is None:
        #    self.add_helper_sim()
        
        #create copies of the array being filled
        temp_arrays = [array_to_fill.copy() for _ in range(sigma_points.shape[1])]

        weight_iter = _itertools.chain([0] ,_itertools.repeat(1, sigma_points.shape[1]-1))

        for i, j in zip(range(sigma_points.shape[1]), weight_iter):
            
            vec = orig_vec.copy()
            vec[hamiltonian_model_indices] = sigma_points[:,i]
            
            #set the model to this current vec value
            self.model.from_vector(vec)
            
            #next simulate the model using the helper simulator:
            self.helper_sim.bulk_fill_probs(temp_arrays[i], layout)
            
            #Iterate through and add weight terms. 
            temp_arrays[i] = weights[j]*temp_arrays[i] 
            
        #reset the model to it's original value
        self.model.from_vector(orig_vec)
        
        #Aggregate all of the probability values into a final_result
        averaged_array = temp_arrays[0]
        for temp_array in temp_arrays[1:]:
            averaged_array += temp_array
            
        #print('averaged: ', averaged_array)
        
        array_to_fill[:]= averaged_array
        #return averaged_array
        
    #Next I need a version of bulk_fill_dprobs:
    def bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill=None):

        eps = 1e-7  # hardcoded?

        if pr_array_to_fill is not None:
            self.bulk_fill_probs(pr_array_to_fill, layout)
            probs = pr_array_to_fill.copy()
        
        else:
            probs = layout.allocate_local_array('e', 'd')
            self.bulk_fill_probs(probs, layout)
        
        orig_vec = self.model.to_vector().copy()
        
        for i in range(self.model.num_params):
            probs2 = probs.copy()
            vec = orig_vec.copy()
            vec[i] += eps
            self.model.from_vector(vec, close=True)
            self.bulk_fill_probs(probs2,layout)
            
            #now put this result into the array to be filled.
            array_to_fill[: , i] =(probs2 - probs) / eps
            
        #restore the model to it's original value
        self.model.from_vector(orig_vec)
        
        #print('dprobs: ', array_to_fill)
        #return dprobs

    #add a version of bulk_fill_hprobs    

    def bulk_fill_hprobs(self, array_to_fill, layout,
                    pr_array_to_fill=None, deriv1_array_to_fill=None, 
                    deriv2_array_to_fill=None):
        """
        Compute the outcome probability-Hessians for an entire list of circuits.

        Similar to `bulk_fill_probs(...)`, but fills a 3D array with
        the Hessians for each circuit outcome probability.

        Parameters
        ----------
        array_to_fill : numpy ndarray
            an already-allocated numpy array of shape `(len(layout),M1,M2)` where
            `M1` and `M2` are the number of selected model parameters (by `wrt_filter1`
            and `wrt_filter2`).

        layout : CircuitOutcomeProbabilityArrayLayout
            A layout for `array_to_fill`, describing what circuit outcome each
            element corresponds to.  Usually given by a prior call to :meth:`create_layout`.

        pr_mx_to_fill : numpy array, optional
            when not None, an already-allocated length-`len(layout)` numpy array that is
            filled with probabilities, just as in :meth:`bulk_fill_probs`.

        deriv1_array_to_fill : numpy array, optional
            when not None, an already-allocated numpy array of shape `(len(layout),M1)`
            that is filled with probability derivatives, similar to
            :meth:`bulk_fill_dprobs` (see `array_to_fill` for a definition of `M1`).

        deriv2_array_to_fill : numpy array, optional
            when not None, an already-allocated numpy array of shape `(len(layout),M2)`
            that is filled with probability derivatives, similar to
            :meth:`bulk_fill_dprobs` (see `array_to_fill` for a definition of `M2`).

        Returns
        -------
        None
        """

        if pr_array_to_fill is not None:
            self.bulk_fill_probs(pr_array_to_fill, layout)
        if deriv1_array_to_fill is not None:
            self.bulk_fill_dprobs(deriv1_array_to_fill, layout)
            dprobs = deriv1_array_to_fill.copy()
        if deriv2_array_to_fill is not None:
            deriv2_array_to_fill[:, :] = deriv1_array_to_fill[:, :]

        eps = 1e-4  # hardcoded?
        dprobs = _np.empty((len(layout), self.model.num_params), 'd')
        self.bulk_fill_dprobs(dprobs, layout)

        dprobs2 = _np.empty((len(layout), self.model.num_params), 'd')

        orig_vec = self.model.to_vector().copy()
        for i in range(self.model.num_params):
            vec = orig_vec.copy() 
            vec[i] += eps
            self.model.from_vector(vec, close=True)
            self.bulk_fill_dprobs(dprobs2, layout)
            array_to_fill[:, i, :] = (dprobs2 - dprobs) / eps
        self.model.from_vector(orig_vec, close=True)
