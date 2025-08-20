"""
Defines the LFHExplicitOpModel class, an extension of ExplicitOpModel with
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

import numpy as np
import collections as _collections
import itertools as _itertools
from pygsti.modelmembers.operations import LindbladErrorgen as _LindbladErrorgen
from pygsti.forwardsims import WeakForwardSimulator as _WeakForwardsimulator
from pygsti.forwardsims import MapForwardSimulator as _MapForwardSimulator
from pygsti.forwardsims import SimpleMapForwardSimulator as _SimpleMapForwardSimulator
from pygsti.forwardsims import MatrixForwardSimulator as _MatrixForwardSimulator
from pygsti.evotypes import Evotype as _Evotype
from pygsti.extras.lfh.lfherrorgen import LFHLindbladErrorgen as _LFHLindbladErrorgen

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

#I think the last thing I need is a model which can propagate through the resampling to any
#underlying LFHLindbladErrorgen objects
class LFHExplicitOpModel(_ExplicitOpModel):
    
    #Use the same init as explicit op model:
    def __init__(self, state_space, basis="pp", default_gate_type="full",
                 default_prep_type="auto", default_povm_type="auto",
                 default_instrument_type="auto", prep_prefix="rho", effect_prefix="E",
                 gate_prefix="G", povm_prefix="M", instrument_prefix="I",
                 simulator="auto", evotype="default"):
        
        super().__init__(state_space, basis, default_gate_type,
                 default_prep_type, default_povm_type,
                 default_instrument_type, prep_prefix, effect_prefix,
                 gate_prefix, povm_prefix, instrument_prefix,
                 simulator, evotype)
        
    #Add a method that resamples the hamiltonian rates when requested.
    def sample_hamiltonian_rates(self):
        #loop through the elements of the operations dictionary
        for member in self.operations.values():
            if isinstance(member, _ComposedOp):
                #next check is any of the factor ops are exponentiated error generators
                for factor in member.factorops:
                    if isinstance(factor, _ExpErrorgenOp):
                        #check to see if the error generator is a LFHLindbladErrorgen
                        if isinstance(factor.errorgen, _LFHLindbladErrorgen):
                            #then propagate the resampling through.
                            factor.errorgen.sample_hamiltonian_rates()
                            #update the representation of the exponentiated error generator
                            factor._update_rep()
                            
                #Once I have updated the reps of the factors I need to reinitalize the rep of
                #the composed op.
                #print([op._rep for op in member.factorops])
                member._update_denserep()
                #.reinit_factor_op_reps([op._rep for op in member.factorops])
    
    #need a version of the circuit_layer_operator method which doesn't call clean_paramvec
    #since I think this is what is causing the value of the 

'''