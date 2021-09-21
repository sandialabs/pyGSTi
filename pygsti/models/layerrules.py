"""
Defines the LayerLizard class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


from pygsti.modelmembers import operations as _op
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable

class LayerRules(_NicelySerializable):
    """
    Rules for composing a layer operation from the elements stored within a model.

    A :class:`LayerRules` object serves as an intermediary between a :class:`ImplicitModel`
    object and a :class:`ForwardSimulator`.  It contains the logic for creating
    layer operations based on the partial/fundamental operation "blocks" stored within the
    model.  Since different models hold different operation blocks, layer rules are usually
    tailored to a specific models.
    """

    def _create_op_for_circuitlabel(self, model, circuitlbl):
        """
        A helper method for derived classes used for processing :class:`CircuitLabel` labels.

        (:class:`CircuitLabel` labels encapsulate sub-circuits repeated some integer number of times).

        This method build an operator for `circuitlbl` by creating a composed-op
        (using :class:`ComposedOp`) of the sub-circuit that is exponentiated (using
        :class:`RepeatedOp`) to the power `circuitlbl.reps`.

        Parameters
        ----------
        circuitlbl : CircuitLabel
            The (sub-circuit)^power to create an operator for.

        Returns
        -------
        LinearOperator
        """
        if len(circuitlbl.components) != 1:  # works for 0 components too
            subCircuitOp = _op.ComposedOp([model.circuit_layer_operator(l, 'op') for l in circuitlbl.components],
                                          evotype=model.evotype, state_space=model.state_space)
        else:
            subCircuitOp = model.circuit_layer_operator(circuitlbl.components[0], 'op')
        if circuitlbl.reps != 1:
            #finalOp = _op.ComposedOp([subCircuitOp]*circuitlbl.reps,
            #                         evotype=model.evotype, state_space=model.state_space)
            finalOp = _op.RepeatedOp(subCircuitOp, circuitlbl.reps, evotype=model.evotype)
        else:
            finalOp = subCircuitOp

        model._init_virtual_obj(finalOp)  # so ret's gpindices get set, essential for being in cache
        return finalOp

    def prep_layer_operator(self, model, layerlbl, cache):
        """
        Create the operator corresponding to `layerlbl`.

        Parameters
        ----------
        layerlbl : Label
            A circuit layer label.

        Returns
        -------
        State
        """
        #raise KeyError(f"Cannot create operator for non-primitive prep layer: {layerlbl}")
        raise KeyError("Cannot create operator for non-primitive prep layer: %s" % str(layerlbl))

    def povm_layer_operator(self, model, layerlbl, cache):
        """
        Create the operator corresponding to `layerlbl`.

        Parameters
        ----------
        layerlbl : Label
            A circuit layer label.

        Returns
        -------
        POVM or POVMEffect
        """
        #raise KeyError(f"Cannot create operator for non-primitive prep layer: {layerlbl}")
        raise KeyError("Cannot create operator for non-primitive prep layer: %s" % str(layerlbl))

    def operation_layer_operator(self, model, layerlbl, cache):
        """
        Create the operator corresponding to `layerlbl`.

        Parameters
        ----------
        layerlbl : Label
            A circuit layer label.

        Returns
        -------
        LinearOperator
        """
        #raise KeyError(f"Cannot create operator for non-primitive layer: {layerlbl}")
        raise KeyError("Cannot create operator for non-primitive layer: %s" % str(layerlbl))
