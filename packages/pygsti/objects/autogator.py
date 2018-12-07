""" Defines the AutoGator class and supporting functions."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from . import modelmember as _mm
from . import gate as _op
from ..baseobjs import label as _label

class AutoGator(_mm.ModelChild):
    """
    The base class for "auto-gator" objects.

    An auto-gator is an object that generates "virtual", i.e. temporary,
    gates for a ForwardSimulator object to facilitate building variants or 
    combinations of gates which aren't permanently stored in the Model.
    
    Often, auto-gator objects can be used to generate gates for "parallel
    operation labels" (operation labels corresponding to multiple "elementary" gates
    performed simultaneously) from the elementary gates stored in a gate
    set.

    Auto-gators behave in some ways like a function, in that their main
    method is __call__, which is asked to construct a LinearOperator for a given
    label given a set of existing permanent LinearOperator objects.
    """
    def __init__(self,parent):
        """
        Create a new AutoGator

        Parameters
        ----------
        parent : Model
            The parent model within which this AutoGator is contained.

        """
        super(AutoGator,self).__init__(parent)

        
    def __call__(self, existing_ops, oplabel):
        """
        Create a LinearOperator for `opLabel` using existing gates.

        Parameters
        ----------
        existing_ops : dict
            A dictionary with keys = operation labels and values = LinearOperator 
            objects.

        opLabel : Label
            The operation label to construct a gate for.

        Returns
        -------
        LinearOperator
        """
        raise NotImplementedError("Derived classes should implement this!")



class SimpleCompositionAutoGator(AutoGator):
    """
    An auto-gator that creates virtual gates for parallel
    operation labels by simply composing elementary gates
    """

    def __init__(self, parent):
        """
        Create a new SimpleCompositionAutoGator

        Parameters
        ----------
        parent : Model
        """
        super(SimpleCompositionAutoGator,self).__init__(parent)

    def __call__(self, existing_ops, oplabel):
        """
        Create a LinearOperator for `opLabel` using existing gates.

        Parameters
        ----------
        existing_ops : dict
            A dictionary with keys = operation labels and values = LinearOperator 
            objects.

        opLabel : Label
            The operation label to construct a gate for.

        Returns
        -------
        LinearOperator
        """
        dense = bool(self.parent._sim_type == "matrix") # whether dense matrix gates should be created
        Composed = _op.ComposedDenseOp if dense else _op.ComposedOp
        #print("DB: SimpleCompositionAutoGator building gate %s for %s" %
        #      (('matrix' if dense else 'map'), str(oplabel)) )
        if isinstance(oplabel, _label.LabelTupTup):
            if self.parent.auto_idle_gatename is not None:
                factor_ops = []
                for l in oplabel.components:
                    if l.name == self.parent.auto_idle_gatename \
                            and l not in existing_ops:
                        continue #skip perfect idle placeholders
                    factor_ops.append(existing_ops[l])
            else:
                factor_ops = [ existing_ops[l] for l in oplabel.components ]
            ret = Composed(factor_ops, dim=self.parent.dim,
                           evotype=self.parent._evotype)
            self.parent._init_virtual_obj(ret) # so ret's gpindices get set
            return ret
        else: raise ValueError("Cannot auto-create gate for label %s" % str(oplabel))
        
    
class SharedIdleAutoGator(AutoGator):
    """
    An auto-gator that creates virtual gates for parallel
    operation labels by composing the non-identity parts of
    elementary gates and keeping just a single instance of
    the identity-noise component which is assumed to be 
    contained within every gate. 

    This autogator assumes a that the exisitng gates have a
    certain structure - that contained in the Model given by
    :function:`build_nqnoise_model`.  In particular, it assumes
    each non-idle gate is a ComposedOp of the form
    `Composed([fullTargetOp,fullIdleErr,fullLocalErr])`, and that
    parallel gates should be combined by composing the target ops
    and local errors but keeping just a single idle error (which
    should be the same for all the gates).
    """

    def __init__(self, parent, errcomp_type):
        """
        Create a new SharedIdleAutoGator

        Parameters
        ----------
        parent : Model
        """
        super(SharedIdleAutoGator,self).__init__(parent)
        self.errcomp_type = errcomp_type

    def __call__(self, existing_ops, oplabel):
        """
        Create a LinearOperator for `opLabel` using existing gates.

        Parameters
        ----------
        existing_ops : dict
            A dictionary with keys = operation labels and values = LinearOperator 
            objects.

        opLabel : Label
            The operation label to construct a gate for.

        Returns
        -------
        LinearOperator
        """
        dense = bool(self.parent._sim_type == "matrix") # whether dense matrix gates should be created
        Composed = _op.ComposedDenseOp if dense else _op.ComposedOp
        Lindblad = _op.LindbladDenseOp if dense else _op.LindbladOp
        Sum = _op.ComposedErrorgen
        #print("DB: SharedIdleAutoGator building gate %s for %s w/comp-type %s" %
        #      (('matrix' if dense else 'map'), str(oplabel), self.errcomp_type) )
        if isinstance(oplabel, _label.LabelTupTup):

            if self.parent.auto_idle_gatename is not None:
                gates = []
                for l in oplabel.components:
                    if l.name == self.parent.auto_idle_gatename \
                            and l not in existing_ops:
                        continue #skip perfect idle placeholders
                    gates.append(existing_ops[l])
            else:
                gates = [ existing_ops[l] for l in oplabel.components ]

            if self.errcomp_type == "gates":
                assert( all([len(g.factorops) == 3 for g in gates]) or
                        all([len(g.factorops) == 2 for g in gates]) )
                #each gate in gates is Composed([fullTargetOp,fullIdleErr,fullLocalErr]) OR
                # just Composed([fullTargetOp,fullLocalErr]).  In the former case, 
                # we compose 1st & 3rd factors of parallel gates and keep just a single 2nd factor.
                # In the latter case, we just compose the 1st and 2nd factors of parallel gates.
                
                if len(gates[0].factorops) == 3:
                    targetOp = Composed([g.factorops[0] for g in gates], dim=self.parent.dim,
                                        evotype=self.parent._evotype)
                    idleErr = gates[0].factorops[1]
                    localErr = Composed([g.factorops[2] for g in gates], dim=self.parent.dim,
                                        evotype=self.parent._evotype)
                    ops_to_compose = [targetOp,idleErr,localErr]
    
                else: # == 2 case
                    targetOp = Composed([g.factorops[0] for g in gates], dim=self.parent.dim,
                                        evotype=self.parent._evotype)
                    localErr = Composed([g.factorops[1] for g in gates], dim=self.parent.dim,
                                        evotype=self.parent._evotype)
                    ops_to_compose = [targetOp,localErr]
    
                #DEBUG could perform a check that gpindices are the same for idle gates
                #import numpy as _np 
                #from ..tools import slicetools as _slct
                #for g in gates:
                #    assert(_np.array_equal(_slct.as_array(g.factorops[1].gpindices),
                #                           _slct.as_array(idleErr.gpindices)))

            elif self.errcomp_type == "errorgens":
                assert( all([len(g.factorops) == 2 for g in gates]) )
                assert( all([(g.factorops[1].unitary_postfactor is None) for g in gates]) )
                #each gate in gates is Composed([fullTargetOp,fullErr]), where fullErr is 
                # a 'exp(Errgen)' Lindblad gate. We compose the target operations to create a
                # final target op, and compose this with a *singe* Lindblad gate which has as
                # its error generator the composition (sum) of all the factors' error gens.
                # (Note: the 'Gi' gate is only a single Lindblad gate, but it shouldn't be 
                #  in the list of factors as it acts on *all* the qubits).
                
                targetOp = Composed([g.factorops[0] for g in gates], dim=self.parent.dim,
                                    evotype=self.parent._evotype)
                errorGens = [g.factorops[1].errorgen for g in gates]
                error = Lindblad(None, Sum(errorGens, dim=self.parent.dim,
                                           evotype=self.parent._evotype),
                                 sparse_expm=not dense)
                ops_to_compose = [targetOp,error]
            else:
                raise ValueError("Invalid errcomp_type in SharedIdleAutoGator: %s" % self.errcomp_type)
            
            ret = Composed(ops_to_compose, dim=self.parent.dim,
                           evotype=self.parent._evotype)
            self.parent._init_virtual_obj(ret) # so ret's gpindices get set
            return ret
        else: raise ValueError("Cannot auto-create gate for label %s" % str(oplabel))
