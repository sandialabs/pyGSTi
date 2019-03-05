""" Defines the LayerLizard class and supporting functionality."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************


import numpy as _np
import scipy as _scipy
import itertools as _itertools
import collections as _collections
import warnings as _warnings
import time as _time
import uuid as _uuid
import bisect as _bisect
import copy as _copy

from ..tools import matrixtools as _mt
from ..tools import optools as _gt
from ..tools import slicetools as _slct
from ..tools import likelihoodfns as _lf
from ..tools import jamiolkowski as _jt
from ..tools import compattools as _compat
from ..tools import basistools as _bt
from ..tools import listtools as _lt
from ..tools import symplectic as _symp

from . import modelmember as _gm
from . import circuit as _cir
from . import operation as _op
from . import spamvec as _sv
from . import povm as _povm
from . import instrument as _instrument
from . import labeldicts as _ld
from . import gaugegroup as _gg
from . import matrixforwardsim as _matrixfwdsim
from . import mapforwardsim as _mapfwdsim
from . import termforwardsim as _termfwdsim
from . import explicitcalc as _explicitcalc

from ..baseobjs import VerbosityPrinter as _VerbosityPrinter
from ..baseobjs import Basis as _Basis
from ..baseobjs import Label as _Label
from ..baseobjs.label import CircuitLabel as _CircuitLabel


class LayerLizard(object):
    """ 
    Helper class for interfacing a Model and a forward simulator
    (which just deals with *simplified* operations).  Can be thought
    of as a "server" of simplified operations for a forward simulator
    which pieces together layer operations from components.
    """
    # TODO docstring - add not-implemented members & docstrings?
    
    def __init__(self, model):
        """
        TODO: docstring
        """
        self.model = model

    #Helper functions for derived classes:
    def get_circuitlabel_op(self, circuitlbl, dense):
        """TODO: docstring
           build an op for this circuit label - a composed op (of sub-circuit)
           exponentiated to the power N, where N=#of repetitions
        """
        Composed = _op.ComposedDenseOp if dense else _op.ComposedOp
        if len(circuitlbl.components) != 1: #works for 0 components too
            subCircuitOp = Composed([self.get_operation(l) for l in circuitlbl.components],
                                    dim=self.model.dim, evotype=self.model._evotype)
        else:
            subCircuitOp = self.get_operation(circuitlbl.components[0])
        if circuitlbl.reps != 1:
            #finalOp = Composed([subCircuitOp]*circuitlbl.reps,
            #                   dim=self.model.dim, evotype=self.model._evotype)
            finalOp = _op.ExponentiatedOp(subCircuitOp,circuitlbl.reps, evotype=self.model._evotype)
        else:
            finalOp = subCircuitOp
        return finalOp

    
class ExplicitLayerLizard(LayerLizard):
    """
    This layer lizard (see :class:`LayerLizard`) only serves up layer 
    operations it have been explicitly provided upon initialization.
    """
    def __init__(self,preps,ops,effects,model):
        """
        Creates a new ExplicitLayerLizard.

        Parameters
        ----------
        preps, ops, effects : OrderedMemberDict
            Dictionaries of simplified layer operations available for 
            serving to a forwared simulator.

        model : Model
            The model associated with the simplified operations.
        """
        self.preps, self.ops, self.effects = preps,ops,effects
        super(ExplicitLayerLizard,self).__init__(model)
        
    def get_evotype(self):
        """ 
        Return the evolution type of the operations being served.

        Returns
        -------
        str
        """
        return self.model._evotype

    def get_prep(self,layerlbl):
        """
        Return the (simplified) preparation layer operator given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        return self.preps[layerlbl]
    
    def get_effect(self,layerlbl):
        """
        Return the (simplified) POVM effect layer operator given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        return self.effects[layerlbl]
    
    def get_operation(self,layerlbl):
        """
        Return the (simplified) layer operation given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        if isinstance(layerlbl,_CircuitLabel):
            dense = bool(self.model._sim_type == "matrix") # whether dense matrix gates should be created
            return self.get_circuitlabel_op(layerlbl, dense)
        else:
            return self.ops[layerlbl]

    def from_vector(self, v):
        """
        Re-initialize the simplified operators from model-parameter-vector `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A vector of parameters for `Model` associated with this layer lizard.
        """
        for _,obj in _itertools.chain(self.preps.items(),
                                      self.effects.items(),
                                      self.ops.items()):
            obj.from_vector( v[obj.gpindices] )


class ImplicitLayerLizard(LayerLizard):
    """ 
    This layer lizard (see :class:`LayerLizard`) is used as a base class for
    objects which serve up layer operations for implicit models (and so provide
    logic for how to construct layer operations from model components).
    """
    def __init__(self,preps,ops,effects,model):
        """
        Creates a new ExplicitLayerLizard.

        Parameters
        ----------
        preps, ops, effects : dict
            Dictionaries of :class:`OrderedMemberDict` objects, one per
            "category" of simplified operators.  These are stored and used
            to build layer operations for serving to a forwared simulator.

        model : Model
            The model associated with the simplified operations.
        """
        self.prep_blks, self.op_blks, self.effect_blks = preps,ops,effects
        super(ImplicitLayerLizard,self).__init__(model)
        
    def get_prep(self,layerlbl):
        """
        Return the (simplified) preparation layer operator given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        raise NotImplementedError("ImplicitLayerLizard-derived classes must implement `get_preps`")
    
    def get_effect(self,layerlbl):
        """
        Return the (simplified) POVM effect layer operator given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        raise NotImplementedError("ImplicitLayerLizard-derived classes must implement `get_effect`")
    
    def get_operation(self,layerlbl):
        """
        Return the (simplified) layer operation given by `layerlbl`.

        Returns
        -------
        LinearOperator
        """
        raise NotImplementedError("ImplicitLayerLizard-derived classes must implement `get_operation`")

    def get_evotype(self):
        """ 
        Return the evolution type of the operations being served.

        Returns
        -------
        str
        """
        return self.model._evotype

    def from_vector(self, v):
        """
        Re-initialize the simplified operators from model-parameter-vector `v`.

        Parameters
        ----------
        v : numpy.ndarray
            A vector of parameters for `Model` associated with this layer lizard.
        """
        for _,objdict in _itertools.chain(self.prep_blks.items(),
                                          self.effect_blks.items(),
                                          self.op_blks.items()):
            for _,obj in objdict.items():
                obj.from_vector( v[obj.gpindices] )
