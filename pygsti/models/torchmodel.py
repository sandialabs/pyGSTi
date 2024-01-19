"""
Defines the TorchOpModel class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from typing import Union

import numpy as np
import scipy.linalg as la
try:
    import torch
    ENABLED = True
except ImportError:
    ENABLED = False

from pygsti.models.explicitmodel import ExplicitOpModel
from pygsti.models.memberdict import OrderedMemberDict as _OrderedMemberDict
from pygsti.models.layerrules import LayerRules as _LayerRules
from pygsti.modelmembers import instruments as _instrument
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.modelmembers import states as _state
from pygsti.modelmembers.operations import opfactory as _opfactory
from pygsti.baseobjs.label import Label as _Label, CircuitLabel as _CircuitLabel


class TorchOpModel(ExplicitOpModel):
    """
    Encapsulates a set of gate, state preparation, and POVM effect operations.

    An ExplictOpModel stores a set of labeled LinearOperator objects and
    provides dictionary-like access to their matrices.  State preparation
    and POVM effect operations are represented as column vectors.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this model.

    basis : {"pp","gm","qt","std","sv"} or Basis, optional
        The basis used for the state space by dense superoperator representations.

    default_param : {"full", "TP", "CPTP", etc.}, optional
        Specifies the default gate and SPAM vector parameterization type.
        Can be any value allowed by :meth:`set_all_parameterizations`,
        which also gives a description of each parameterization type.

    prep_prefix: string, optional
        Key prefixe for state preparations, allowing the model to determing what
        type of object a key corresponds to.

    effect_prefix : string, optional
        Key prefix for POVM effects, allowing the model to determing what
        type of object a key corresponds to.

    gate_prefix : string, optional
        Key prefix for gates, allowing the model to determing what
        type of object a key corresponds to.

    povm_prefix : string, optional
        Key prefix for POVMs, allowing the model to determing what
        type of object a key corresponds to.

    instrument_prefix : string, optional
        Key prefix for instruments, allowing the model to determing what
        type of object a key corresponds to.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The circuit simulator used to compute any
        requested probabilities, e.g. from :meth:`probs` or
        :meth:`bulk_probs`.  The default value of `"auto"` automatically
        selects the simulation type, and is usually what you want. Other
        special allowed values are:

        - "matrix" : op_matrix-op_matrix products are computed and
          cached to get composite gates which can then quickly simulate
          a circuit for any preparation and outcome.  High memory demand;
          best for a small number of (1 or 2) qubits.
        - "map" : op_matrix-state_vector products are repeatedly computed
          to simulate circuits.  Slower for a small number of qubits, but
          faster and more memory efficient for higher numbers of qubits (3+).

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.
    """

    #Whether access to gates & spam vecs via Model indexing is allowed
    _strict = False

    def __init__(self, state_space, basis="pp", default_gate_type="full",
                 default_prep_type="auto", default_povm_type="auto",
                 default_instrument_type="auto", prep_prefix="rho", effect_prefix="E",
                 gate_prefix="G", povm_prefix="M", instrument_prefix="I",
                 simulator="auto", evotype="default"):

        def flagfn(typ): return {'auto_embed': True, 'match_parent_statespace': True,
                                 'match_parent_evotype': True, 'cast_to_type': typ}

        if default_prep_type == "auto":
            default_prep_type = _state.state_type_from_op_type(default_gate_type)
        if default_povm_type == "auto":
            default_povm_type = _povm.povm_type_from_op_type(default_gate_type)
        if default_instrument_type == "auto":
            default_instrument_type = _instrument.instrument_type_from_op_type(default_gate_type)

        self.preps = _OrderedMemberDict(self, default_prep_type, prep_prefix, flagfn("state"))
        self.povms = _OrderedMemberDict(self, default_povm_type, povm_prefix, flagfn("povm"))
        self.operations = _OrderedMemberDict(self, default_gate_type, gate_prefix, flagfn("operation"))
        self.instruments = _OrderedMemberDict(self, default_instrument_type, instrument_prefix, flagfn("instrument"))
        self.factories = _OrderedMemberDict(self, default_gate_type, gate_prefix, flagfn("factory"))
        self.effects_prefix = effect_prefix
        self._default_gauge_group = None

        super(ExplicitOpModel, self).__init__(state_space, basis, evotype, TorchLayerRules(), simulator)
        # ^ call __init__ for our parent class's parent class, not our own parent class.

    def __get_state__(self):
        return self.__dict__.copy()

    def __set_state__(self, state):
        self.__dict__.update(state)
        self._layer_rules = TorchLayerRules()


class TorchLayerRules(_LayerRules):
    """ Directly copy the implementation of ExplicitLayerRules """

    def prep_layer_operator(self, model: TorchOpModel, layerlbl: _Label, caches: dict) -> _state.State:
        return model.preps[layerlbl]

    def povm_layer_operator(self, model: TorchOpModel, layerlbl: _Label, caches: dict) -> Union[_povm.POVM, _povm.POVMEffect]:
        if layerlbl in caches['povm-layers']:
            return caches['povm-layers'][layerlbl]
        # else, don't cache return value - it's not a new operator
        return model.povms[layerlbl]

    def operation_layer_operator(self, model: TorchOpModel, layerlbl: _Label, caches: dict) -> _op.linearop.LinearOperator:
        if layerlbl in caches['op-layers']:
            return caches['op-layers'][layerlbl]
        if isinstance(layerlbl, _CircuitLabel):
            op = self._create_op_for_circuitlabel(model, layerlbl)
            caches['op-layers'][layerlbl] = op
            return op
        elif layerlbl in model.operations:
            return model.operations[layerlbl]
        else:
            return _opfactory.op_from_factories(model.factories, layerlbl)
