"""
Defines the LFHLindbladErrorgen class, an extension of LindbladErrorgen with
support for fluctuating Hamiltonian parameters.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
'''
import numpy as _np
import collections as _collections
import itertools as _itertools
from pygsti.modelmembers.operations import LindbladErrorgen as _LindbladErrorgen
from pygsti.forwardsims import WeakForwardSimulator as _WeakForwardsimulator
from pygsti.forwardsims import MapForwardSimulator as _MapForwardSimulator
from pygsti.forwardsims import SimpleMapForwardSimulator as _SimpleMapForwardSimulator
from pygsti.forwardsims import MatrixForwardSimulator as _MatrixForwardSimulator
from pygsti.evotypes import Evotype as _Evotype

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


#--------- New LindbladErrorgen ------------#
#Pattern match a bit off of the parameterized lindblad error generator Jordan cooked up
class LFHLindbladErrorgen(_LindbladErrorgen):
    """
    A Lindblad error generator with parameters that are combined
    to get the target error generator based on some function params_to_coeffs of the parameter vector
    params_to_coeffs should return a numpy array
    """
    def coeff_dict_from_vector(self):
        basis = _BuiltinBasis('pp', 4)
        v = self.current_rates
        #print(len(v))
        error_rates_dict = {}
        for i in range(3):
            error_rates_dict[('H',basis.labels[i+1])] = v[i]
        labels = [('S', 'X'), ('A','X','Y'),('A','X','Z'),('C','X','Z'),('S','Y'),('A','Y','Z'),('C','X','Y'),('C','Y','Z'),('S','Z')]
        for i in range(3,12):
            error_rates_dict[(labels[i-3])] = v[i]
        return error_rates_dict
    
    def __init__(self, h_means, otherlindbladparams, h_devs, lindblad_basis='auto', elementary_errorgen_basis='pp',
                 evotype="default", state_space=1, parameterization='CPTPLND', truncate=True, rng= None):
        #Pass in a vector of standard lindblad parameters as well as a vector of standard deviations
        #for each of the hamiltonian parameters
        
        #Store the values of the mean hamiltonian rates.
        self.means= h_means
        self.otherlindbladparams = otherlindbladparams
        
        self.paramvals = _np.array([param for param in self.means] + [param for param in self.otherlindbladparams]) #the parameters
        self.current_rates = self.paramvals.copy()
        
        #let's make the h deviations a dictionary instead, so we can control which of the hamiltonian rates are fluctuating
        #to make the marginalization more efficient (avoiding duplicated calculations when std. devs are 0.
        #We'll make the keys of the dictionary the index in h_means that the deviation corresponds to.

        self.dev_dict = h_devs
        self.devs= _np.fromiter(h_devs.values(), dtype = _np.double)
        
        #set the random number generator used for sampling from a normal distribution.
        if rng is not None:
            if isinstance(rng, int):
                self.rng= _np.random.default_rng(rng)
            else:
                self.rng = rng
        else:
            self.rng= _np.random.default_rng()
        
        #Get the coefficient dictionary for this parameter vector
        self.coefficients = self.coeff_dict_from_vector()
        #super().from_elementary_errorgens(coeff_dict, state_space = 1)
        
        state_space = _statespace.StateSpace.cast(state_space)
        dim = state_space.dim  # Store superop dimension
        basis = _Basis.cast(elementary_errorgen_basis, dim)

        #convert elementary errorgen labels to *local* labels (ok to specify w/global labels)
        identity_label_1Q = 'I'  # maybe we could get this from a 1Q basis somewhere?
        sslbls = state_space.tensor_product_block_labels(0)  # just take first TPB labels as all labels
        elementary_errorgens = _collections.OrderedDict(
            [(_LocalElementaryErrorgenLabel.cast(lbl, sslbls, identity_label_1Q), val)
             for lbl, val in self.coefficients.items()])

        parameterization = LindbladParameterization.minimal_from_elementary_errorgens(elementary_errorgens) \
            if parameterization == "auto" else LindbladParameterization.cast(parameterization)
        
        eegs_by_typ = {
            'ham': {eeglbl: v for eeglbl, v in elementary_errorgens.items() if eeglbl.errorgen_type == 'H'},
            'other_diagonal': {eeglbl: v for eeglbl, v in elementary_errorgens.items() if eeglbl.errorgen_type == 'S'},
            'other': {eeglbl: v for eeglbl, v in elementary_errorgens.items() if eeglbl.errorgen_type != 'H'}
        }

        blocks = []
        for blk_type, blk_param_mode in zip(parameterization.block_types, parameterization.param_modes):
            relevant_eegs = eegs_by_typ[blk_type]  # KeyError => unrecognized block type!
            bels = sorted(set(_itertools.chain(*[lbl.basis_element_labels for lbl in relevant_eegs.keys()])))
            blk = _LindbladCoefficientBlock(blk_type, basis, bels, param_mode=blk_param_mode)
            blk.set_elementary_errorgens(relevant_eegs, truncate=truncate)
            blocks.append(blk)
            #print(blk)
        
        evotype= _Evotype.cast(evotype)
        evotype.prefer_dense_reps = True
        
        super().__init__(blocks, evotype=evotype, state_space=1)
        
    @property
    def num_params(self):
        """
        Get the number of independent parameters which specify this operation.

        Returns
        -------
        int
            the number of independent parameters.
        """
        return len(self.paramvals) + len(self.devs)
        
    def to_vector(self):
        ret_vec= [param for param in self.paramvals] + [dev for dev in self.devs]
        
        return _np.array(ret_vec)
        
    def from_vector(self,v, close=False, dirty_value=True):
        """
        Initialize the operation using a vector of parameters.

        Parameters
        ----------
        v : numpy array
            The 1D vector of operation parameters.  Length
            must == num_params()

        close : bool, optional
            Whether `v` is close to this operation's current
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
        assert(len(v) == self.num_params)
        
        #split off the terms that go into paramvals and devs
        v = _np.array(v)
        new_paramvals= v[:len(self.paramvals)]
        new_otherlindblad_params = v[3:len(self.paramvals)]
        new_devs= v[len(self.paramvals):]
        new_means= v[0:3]
        
        self.paramvals = new_paramvals
        self.means= new_means
        self.devs= new_devs
        self.dev_dict = {key:val for key,val in zip(self.dev_dict.keys(), new_devs)}
        self.otherlindbladparams = new_otherlindblad_params
        
        self.coefficients = self.coeff_dict_from_vector() 
        
        #coefficient blocks and current rates get reset to the new mean values passed in
        #resampling can cause the values of the coefficient blocks and the rates to become
        #different though.
        self.current_rates= self.paramvals.copy()
        off = 0
        u = self.paramvals
        for blk in self.coefficient_blocks:
            blk.from_vector(u[off: off + blk.num_params])
            off += blk.num_params
        self._update_rep()
        self.dirty = dirty_value
        
    #Now the special ingredient we need is functionality for resampling
    #What we want to be able to do is use the current hamiltonian means
    #and std deviations to get a new set of hamiltonian weights.
    
    def sample_hamiltonian_rates(self):#, dirty_value=True):
        
        new_h_rates = [self.rng.normal(loc=mean, scale=self.dev_dict[i]) if i in self.dev_dict else mean 
                       for i, mean in enumerate(self.means)] 
        
        #now we want to update the coefficent blocks and current rates:
        self.current_rates = _np.array(new_h_rates + [other_lindblad for other_lindblad in self.otherlindbladparams])
        off = 0
        u = self.current_rates
        for blk in self.coefficient_blocks:
            blk.from_vector(u[off: off + blk.num_params])
            off += blk.num_params
        self._update_rep()
        #self.dirty = dirty_value

'''