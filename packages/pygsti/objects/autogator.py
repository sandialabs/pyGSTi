""" Defines the AutoGator class and supporting functions."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from . import gatesetmember as _gsm
from . import gate as _gate
from ..baseobjs import label as _label

class AutoGator(_gsm.GateSetChild):
    """
    TODO: docstrings in this module
    Currently this class is essentially a function,
    but we want allow room for future expansion.
    """

    def __init__(self,parent):
        super(AutoGator,self).__init__(parent)
        
    def __call__(self, existing_gates, gatelabel):
        """
        Create a Gate for `gateLabel` using existing gates.

        Parameters
        ----------
        existing_gates : dict
            A dictionary with keys = gate labels and values = Gate 
            objects.

        gateLabel : Label
            The gate label to construct a gate for.

        Returns
        -------
        Gate
        """
        raise NotImplementedError("Derived classes should implement this!")



class SimpleCompositionAutoGator(AutoGator):
    """
    TODO: docstring
    Just composes existing gates together to form 
    """

    def __init__(self, parent):
        super(SimpleCompositionAutoGator,self).__init__(parent)

    def __call__(self, existing_gates, gatelabel):
        """
        Create a Gate for `gateLabel` using existing gates.

        Parameters
        ----------
        existing_gates : dict
            A dictionary with keys = gate labels and values = Gate 
            objects.

        gateLabel : Label
            The gate label to construct a gate for.

        Returns
        -------
        Gate
        """

        dense = bool(self.parent._sim_type == "matrix") # whether dense matrix gates should be created
        #print("DB: SimpleCompositionAutoGator building gate %s for %s" %
        #      (('matrix' if dense else 'map'), str(gatelabel)) )
        if isinstance(gatelabel, _label.LabelTupTup):
            factor_gates = [ existing_gates[l] for l in gatelabel.components ]
            ret = _gate.ComposedGate(factor_gates) if dense else _gate.ComposedGateMap(factor_gates)
            self.parent._init_virtual_obj(ret) # so ret's gpindices get set
            return ret
        else: raise ValueError("Cannot auto-create gate for label %s" % str(gatelabel))
        
    
