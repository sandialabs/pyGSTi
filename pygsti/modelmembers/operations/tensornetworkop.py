"""
The TensorNetworkOp class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.modelmembers import ModelMember as _ModelMember
from pygsti.modelmembers.povms import POVM as _POVM
from quimb.tensor import Tensor as _Tensor
from quimb.tensor import TensorNetwork as _TensorNetwork
from quimb.tensor import rand_uuid
import numpy as _np
import collections as _collections


#Add a parent class that the rest of the tensor network operations will subclass off of.

#TODO:Sort out state_space and evotype args

class TensorNetworkOp(_ModelMember):

    @classmethod
    def from_tensor_network(cls, tensor_network, state_space=None, evotype= None):
        tensor_list = list(tensor_network.tensors)
        
        tensor_arrays = [tensor.data for tensor in tensor_list]
        tensor_index_tuples = [list(tensor.inds) for tensor in tensor_list]
        site_tags = [list(tensor.tags) for tensor in tensor_list]
        
        return cls(tensor_arrays, tensor_index_tuples, state_space, evotype, site_tags)

    @classmethod
    def from_choi_matrix(cls, mx, bond_dim , basis):
        pass
    
    @classmethod
    def from_process_mx(cls, mx, bond_dim, basis):
        pass

    def __init__(self, tensor_array_list, index_tags, state_space, evotype, site_tags= None):
        """
        tensor_array_list : list of ndarrays
            This is a list of arrays that are used to construct quimb tensor objects
            
        index_tags : list of tuples of strings
            A list of tuples, one for each tensor in tensor_array_list. The elements of each
            tuple correspond to labels for the legs of the tensor 
            
        site_tags : list of strings (optional, default None)
            Strings labeling tensors, primarily used for visualization of tensor networks.
        """
        
        #initialize the list of tensor objects:
        if site_tags is None:
            site_tags= [None for _ in range(len(index_tags))]
        
        self.tensor_list= [_Tensor(array, inds= idx_tag, tags= [site_tag]) for array, idx_tag, site_tag in zip(tensor_array_list, index_tags, site_tags)]
        
        self.tensor_network = _TensorNetwork(self.tensor_list, virtual=True)
           
        #initialize parameter vector
        self.paramvals = _np.concatenate([tensor_array for tensor_array in tensor_array_list], axis=None)
        
        #track the shapes of the tensors
        self.tensor_shapes = [tensor.shape for tensor in tensor_array_list]
        
        #also track the number of elements for conveinence
        self.tensor_element_counts = [_np.prod(shape) for shape in self.tensor_shapes]
           
        super().__init__(state_space, evotype)
           
    def from_vector(self, v, close=False, dirty_value=True):
        #print('RUNNING IN TENSOR NETWORK')
        #print(f'{v=}')
        #start by partitioning the vector v into sections with sizes given by tensor_element_counts
        partitioned_vector = []
        for i, element_count in enumerate(self.tensor_element_counts):
            if i ==0:
                partitioned_vector.append(v[0:element_count])
                total_seen = element_count
            else:
                partitioned_vector.append(v[total_seen: total_seen+element_count])
                total_seen += element_count
            
        #reshape the flattened_array arrays in the multi-dimensional tensors
        reshaped_arrays = [_np.reshape(flattened_array, new_shape) for flattened_array, new_shape in zip(partitioned_vector, self.tensor_shapes)]
        
        #update the quimb tensor reps
        for quimb_tensor, new_data in zip(self.tensor_list, reshaped_arrays):
            quimb_tensor.modify(data= new_data)
        
        #print(f'{self.paramvals=}')
        #print(f'{self.tensor_list[0].data=}')
              
        self.paramvals = v
        
        #print(f'{self.paramvals=}')

        self.dirty = dirty_value
        
    def to_vector(self):
        return self.paramvals
        
    @property
    def num_params(self):
        return len(self.paramvals)

class LPDOTensorOp(TensorNetworkOp):

    @classmethod
    def from_ket_network(cls, ket_network, state_space=None, evotype= None):
        tensor_list = list(ket_network.tensors)
        
        
        tensor_arrays = [tensor.data for tensor in tensor_list]
        tensor_index_tuples = [tuple(tensor.inds) for tensor in tensor_list]
        site_tags = [tuple(tensor.tags) for tensor in tensor_list]
        
        return cls(tensor_arrays, tensor_index_tuples, state_space, evotype, site_tags)

    
    def __init__(self, tensor_array_list, index_tags, state_space, evotype, site_tags= None):
        
        super().__init__(tensor_array_list, index_tags, state_space, evotype, site_tags)
        
        self.bra_tensor_network = self.tensor_network.copy().H
        
        self.bra_tensor_network.reindex_({ind : 'b' + ind[1:] for ind in self.bra_tensor_network.outer_inds() if ind[0]=='k'})
        self.bra_tensor_network.reindex_({I : rand_uuid() for I in self.bra_tensor_network.inner_inds()})
        
        self.LPDO_tensor_network = self.tensor_network & self.bra_tensor_network
        
     
class LPDOTensorState(LPDOTensorOp):
    
    @classmethod
    def from_ket_network(cls, ket_network, state_space=None, evotype= None):
        tensor_list = list(ket_network.tensors)
        
        
        tensor_arrays = [tensor.data for tensor in tensor_list]
        tensor_index_tuples = [tuple(tensor.inds) for tensor in tensor_list]
        site_tags = [tuple(tensor.tags) for tensor in tensor_list]
        
        return cls(tensor_arrays, tensor_index_tuples, state_space, evotype, site_tags)

    def __init__(self, tensor_array_list, index_tags, state_space, evotype, site_tags= None):
        super().__init__(tensor_array_list, index_tags, state_space, evotype, site_tags)
        
        
        
class LPDOTensorEffect(LPDOTensorOp):
    
    @classmethod
    def from_ket_network(cls, ket_network, state_space=None, evotype= None):
        tensor_list = list(ket_network.tensors)
        
        
        tensor_arrays = [tensor.data for tensor in tensor_list]
        tensor_index_tuples = [tuple(tensor.inds) for tensor in tensor_list]
        site_tags = [tuple(tensor.tags) for tensor in tensor_list]
        
        return cls(tensor_arrays, tensor_index_tuples, state_space, evotype, site_tags)
    
    def __init__(self, tensor_array_list, index_tags, state_space, evotype, site_tags= None):
        super().__init__(tensor_array_list, index_tags, state_space, evotype, site_tags)
        
        
class TensorNetworkPOVM(_POVM):
    def __init__(self, state_space=None, evotype=None, rep=None, items=[]):
        super().__init__(state_space, evotype, rep, items)
        self.init_gpindices()
    
    def submembers(self):
        """
        Returns a sequence of any sub-ModelMember objects contained in this one.

        Sub-members are processed by other :class:`ModelMember` methods
        (e.g. `unlink_parent` and `set_gpindices`) as though the parent
        object is *just* a container for these sub-members and has no
        parameters of its own.  Member objects that contain other members
        *and* possess their own independent parameters should implement
        the appropriate `ModelMember` functions (usually just
        `allocate_gpindices`, using the base implementation as a reference).

        Returns
        -------
        list or tuple
        """
        return tuple(self.values())
    
    def simplify_effects(self, prefix=""):
        """
        Creates a dictionary of simplified effect vectors.

        Returns a dictionary of effect POVMEffects that belong to the POVM's parent
        `Model` - that is, whose `gpindices` are set to all or a subset of
        this POVM's gpindices.  Such effect vectors are used internally within
        computations involving the parent `Model`.

        Parameters
        ----------
        prefix : str
            A string, usually identitying this POVM, which may be used
            to prefix the simplified gate keys.

        Returns
        -------
        OrderedDict of POVMEffects
        """
        if prefix: prefix = prefix + "_"
        simplified = _collections.OrderedDict()
        for lbl, effect in self.items():
            simplified[prefix + lbl] = effect

        return simplified
    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this POVM.

        Returns
        -------
        int
           the number of independent parameters.
        """
        return len(self.gpindices_as_array())
    
    def to_vector(self):
        """
        Extract a vector of the underlying gate parameters from this POVM.

        Returns
        -------
        numpy array
            a 1D numpy array with length == num_params().
        """
        v = _np.empty(self.num_params, 'd')
        for (lbl, effect), effect_local_inds in zip(self.items(), self._submember_rpindices):
            #if lbl == self.complement_label: continue
            v[effect_local_inds] = effect.to_vector()
        return v
    
    def from_vector(self, v, close=False, dirty_value=True):
        """
        Initialize this POVM using a vector of its parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of POVM parameters.  Length
            must == num_params().

        close : bool, optional
            Whether `v` is close to this POVM's current
            set of parameters.  Under some circumstances, when this
            is true this call can be completed more quickly.

        dirty_value : bool, optional
            The value to set this object's "dirty flag" to before exiting this
            call.  This is passed as an argument so it can be updated *recursively*.
            Leave this set to `True` unless you know what you're doing.

        Returns
        -------
        None
        """
        for (lbl, effect), effect_local_inds in zip(self.items(), self._submember_rpindices):
            #if lbl == self.complement_label: continue
            effect.from_vector(v[effect_local_inds], close, dirty_value)
        #if self.complement_label:  # re-init Ec
        #    self[self.complement_label]._construct_vector()
    
    

       
        
        
        
        
    
    
    



        
        
        
        
        
        
        
        
        
    
    




