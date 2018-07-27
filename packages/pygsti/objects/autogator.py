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
    The base class for "auto-gator" objects.

    An auto-gator is an object that generates "virtual", i.e. temporary,
    gates for a GateCalculator object to facilitate building variants or 
    combinations of gates which aren't permanently stored in the GateSet.
    
    Often, auto-gator objects can be used to generate gates for "parallel
    gate labels" (gate labels corresponding to multiple "elementary" gates
    performed simultaneously) from the elementary gates stored in a gate
    set.

    Auto-gators behave in some ways like a function, in that their main
    method is __call__, which is asked to construct a Gate for a given
    label given a set of existing permanent Gate objects.
    """
    def __init__(self,parent):
        """
        Create a new AutoGator

        Parameters
        ----------
        parent : GateSet
            The parent gate set within which this AutoGator is contained.

        """
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
    An auto-gator that creates virtual gates for parallel
    gate labels by simply composing elementary gates
    """

    def __init__(self, parent):
        """
        Create a new SimpleCompositionAutoGator

        Parameters
        ----------
        parent : GateSet
        """
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
        Composed = _gate.ComposedGate if dense else _gate.ComposedGateMap
        #print("DB: SimpleCompositionAutoGator building gate %s for %s" %
        #      (('matrix' if dense else 'map'), str(gatelabel)) )
        if isinstance(gatelabel, _label.LabelTupTup):
            factor_gates = [ existing_gates[l] for l in gatelabel.components ]
            ret = Composed(factor_gates)
            self.parent._init_virtual_obj(ret) # so ret's gpindices get set
            return ret
        else: raise ValueError("Cannot auto-create gate for label %s" % str(gatelabel))
        
    
class SharedIdleAutoGator(AutoGator):
    """
    An auto-gator that creates virtual gates for parallel
    gate labels by composing the non-identity parts of
    elementary gates and keeping just a single instance of
    the identity-noise component which is assumed to be 
    contained within every gate. 

    This autogator assumes a that the exisitng gates have a
    certain structure - that contained in the GateSet given by
    :function:`build_nqnoise_gateset`.  In particular, it assumes
    each non-idle gate is a ComposedGateMap of the form
    `Composed([fullTargetOp,fullIdleErr,fullLocalErr])`, and that
    parallel gates should be combined by composing the target ops
    and local errors but keeping just a single idle error (which
    should be the same for all the gates).
    """

    def __init__(self, parent):
        """
        Create a new SharedIdleAutoGator

        Parameters
        ----------
        parent : GateSet
        """
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
        Composed = _gate.ComposedGate if dense else _gate.ComposedGateMap
        #print("DB: SharedIdleAutoGator building gate %s for %s" %
        #      (('matrix' if dense else 'map'), str(gatelabel)) )
        if isinstance(gatelabel, _label.LabelTupTup):
            gates = [ existing_gates[l] for l in gatelabel.components ]
            #each gate in gates is Composed([fullTargetOp,fullIdleErr,fullLocalErr])
            # so we compose 1st & 3rd factors of parallel gates and keep just a single 2nd factor...
            
            targetOp = Composed([g.factorgates[0] for g in gates])
            idleErr = gates[0].factorgate[1]
            localErr = Composed([g.factorgates[2] for g in gates])
            
            ret = Composed([targetOp,idleErr,localErr])
            self.parent._init_virtual_obj(ret) # so ret's gpindices get set
            return ret
        else: raise ValueError("Cannot auto-create gate for label %s" % str(gatelabel))
