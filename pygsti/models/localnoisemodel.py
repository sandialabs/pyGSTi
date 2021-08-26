"""
Defines the LocalNoiseModel class and supporting functions
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import collections as _collections
import itertools as _itertools
import warnings as _warnings

from pygsti.models.implicitmodel import ImplicitOpModel as _ImplicitOpModel, _init_spam_layers
from pygsti.models.layerrules import LayerRules as _LayerRules
from pygsti.models.memberdict import OrderedMemberDict as _OrderedMemberDict
from pygsti.baseobjs import qubitgraph as _qgraph, statespace as _statespace
from pygsti.evotypes import Evotype as _Evotype
from pygsti.forwardsims.forwardsim import ForwardSimulator as _FSim
from pygsti.forwardsims.mapforwardsim import MapForwardSimulator as _MapFSim
from pygsti.forwardsims.matrixforwardsim import MatrixForwardSimulator as _MatrixFSim
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.modelmembers import states as _state
from pygsti.modelmembers.operations import opfactory as _opfactory
from pygsti.baseobjs.basis import BuiltinBasis as _BuiltinBasis
from pygsti.baseobjs.label import Label as _Lbl, CircuitLabel as _CircuitLabel
from pygsti.tools import basistools as _bt
from pygsti.tools import internalgates as _itgs
from pygsti.tools import optools as _ot


class LocalNoiseModel(_ImplicitOpModel):
    """
    A n-qubit implicit model that allows for only local noise.

    This model holds as building blocks individual noisy gates
    which are trivially embedded into circuit layers as requested.

    Parameters
    ----------
    num_qubits : int
        The total number of qubits.

    gatedict : dict
        A dictionary (an `OrderedDict` if you care about insertion order) that
        associates with gate names (e.g. `"Gx"`) :class:`LinearOperator`,
        `numpy.ndarray` objects. When the objects may act on fewer than the total
        number of qubits (determined by their dimension/shape) then they are
        repeatedly embedded into `num_qubits`-qubit gates as specified by `availability`.
        While the keys of this dictionary are usually string-type gate *names*,
        labels that include target qubits, e.g. `("Gx",0)`, may be used to
        override the default behavior of embedding a reference or a copy of
        the gate associated with the same label minus the target qubits
        (e.g. `"Gx"`).  Furthermore, :class:`OpFactory` objects may be used
        in place of `LinearOperator` objects to allow the evaluation of labels
        with arguments.

    prep_layers : None or operator or dict or list
        The state preparateion operations as n-qubit layer operations.  If
        `None`, then no state preparations will be present in the created model.
        If a dict, then the keys are labels and the values are layer operators.
        If a list, then the elements are layer operators and the labels will be
        assigned as "rhoX" where X is an integer starting at 0.  If a single
        layer operation of type :class:`State` is given, then this is used as
        the sole prep and is assigned the label "rho0".

    povm_layers : None or operator or dict or list
        The state preparateion operations as n-qubit layer operations.  If
        `None`, then no POVMS will be present in the created model.  If a dict,
        then the keys are labels and the values are layer operators.  If a list,
        then the elements are layer operators and the labels will be assigned as
        "MX" where X is an integer starting at 0.  If a single layer operation
        of type :class:`POVM` is given, then this is used as the sole POVM and
        is assigned the label "Mdefault".

    availability : dict, optional
        A dictionary whose keys are the same gate names as in
        `gatedict` and whose values are lists of qubit-label-tuples.  Each
        qubit-label-tuple must have length equal to the number of qubits
        the corresponding gate acts upon, and causes that gate to be
        embedded to act on the specified qubits.  For example,
        `{ 'Gx': [(0,),(1,),(2,)], 'Gcnot': [(0,1),(1,2)] }` would cause
        the `1-qubit `'Gx'`-gate to be embedded three times, acting on qubits
        0, 1, and 2, and the 2-qubit `'Gcnot'`-gate to be embedded twice,
        acting on qubits 0 & 1 and 1 & 2.  Instead of a list of tuples,
        values of `availability` may take the special values:

        - `"all-permutations"` and `"all-combinations"` equate to all possible
        permutations and combinations of the appropriate number of qubit labels
        (deterined by the gate's dimension).
        - `"all-edges"` equates to all the vertices, for 1Q gates, and all the
        edges, for 2Q gates of the geometry.
        - `"arbitrary"` or `"*"` means that the corresponding gate can be placed
        on any target qubits via an :class:`EmbeddingOpFactory` (uses less
        memory but slower than `"all-permutations"`.

        If a gate name (a key of `gatedict`) is not present in `availability`,
        the default is `"all-edges"`.

    qubit_labels : tuple, optional
        The circuit-line labels for each of the qubits, which can be integers
        and/or strings.  Must be of length `num_qubits`.  If None, then the
        integers from 0 to `num_qubits-1` are used.

    geometry : {"line","ring","grid","torus"} or QubitGraph
        The type of connectivity among the qubits, specifying a
        graph used to define neighbor relationships.  Alternatively,
        a :class:`QubitGraph` object with node labels equal to
        `qubit_labels` may be passed directly.

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The circuit simulator used to compute any
        requested probabilities, e.g. from :method:`probs` or
        :method:`bulk_probs`.  The default value of `"auto"` automatically
        selects the simulation type, and is usually what you want. Other
        special allowed values are:

        - "matrix" : op_matrix-op_matrix products are computed and
          cached to get composite gates which can then quickly simulate
          a circuit for any preparation and outcome.  High memory demand;
          best for a small number of (1 or 2) qubits.
        - "map" : op_matrix-state_vector products are repeatedly computed
          to simulate circuits.  Slower for a small number of qubits, but
          faster and more memory efficient for higher numbers of qubits (3+).

    on_construction_error : {'raise','warn',ignore'}
        What to do when the conversion from a value in `gatedict` to a
        :class:`LinearOperator` of the type given by `parameterization` fails.
        Usually you'll want to `"raise"` the error.  In some cases,
        for example when converting as many gates as you can into
        `parameterization="clifford"` gates, `"warn"` or even `"ignore"`
        may be useful.

    independent_gates : bool, optional
        Whether gates are allowed independent local noise or not.  If False,
        then all gates with the same name (e.g. "Gx") will have the *same*
        (local) noise (e.g. an overrotation by 1 degree), and the
        `operation_bks['gates']` dictionary contains a single key per gate
        name.  If True, then gates with the same name acting on different
        qubits may have different local noise, and so the
        `operation_bks['gates']` dictionary contains a key for each gate
         available gate placement.

    ensure_composed_gates : bool, optional
        If True then the elements of the `operation_bks['gates']` will always
        be :class:`ComposedOp` objects.  The purpose of this is to
        facilitate modifying the gate operations after the model is created.
        If False, then the appropriately parameterized gate objects (often
        dense gates) are used directly.

    global_idle : LinearOperator, optional
        A global idle operation, which is performed once at the beginning
        of every circuit layer.  If `None`, no such operation is performed.
        If a 1-qubit operator is given and `num_qubits > 1` the global idle
        is the parallel application of this operator on each qubit line.
        Otherwise the given operator must act on all `num_qubits` qubits.
    """

    def __init__(self, processor_spec, gatedict, prep_layers=None, povm_layers=None, evotype="default",
                 simulator="auto", on_construction_error='raise',
                 independent_gates=False, ensure_composed_gates=False, implicit_idle_mode="none"):

        qubit_labels = processor_spec.qubit_labels
        state_space = _statespace.QubitSpace(qubit_labels)

        simulator = _FSim.cast(simulator, state_space.num_qubits)
        prefer_dense_reps = isinstance(simulator, _MatrixFSim)
        evotype = _Evotype.cast(evotype, default_prefer_dense_reps=prefer_dense_reps)

        # Build gate dictionaries. A value of `gatedict` can be an array, a LinearOperator, or an OpFactory.
        # For later processing, we'll create mm_gatedict to contain each item as a ModelMember.  In local noise
        # models, these gates can be parameterized however the user desires - the LocalNoiseModel just embeds these
        # operators appropriately.
        mm_gatedict = _collections.OrderedDict()  # ops as ModelMembers

        for key, gate in gatedict.items():
            if isinstance(gate, (_op.LinearOperator, _opfactory.OpFactory)):
                mm_gatedict[key] = gate
            else:  # presumably a numpy array or something like it.
                mm_gatedict[key] = _op.StaticArbitraryOp(gate, evotype, state_space)  # static gates by default

        self.processor_spec = processor_spec
        idle_names = processor_spec.idle_gate_names
        global_idle_layer_label = processor_spec.global_idle_layer_label

        layer_rules = _SimpleCompLayerRules(global_idle_layer_label, implicit_idle_mode)

        super(LocalNoiseModel, self).__init__(state_space, layer_rules, 'pp',
                                              simulator=simulator, evotype=evotype)

        flags = {'auto_embed': False, 'match_parent_statespace': False,
                 'match_parent_evotype': True, 'cast_to_type': None}
        self.prep_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
        self.povm_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
        self.operation_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
        self.operation_blks['gates'] = _OrderedMemberDict(self, None, None, flags)
        self.instrument_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
        self.factories['gates'] = _OrderedMemberDict(self, None, None, flags)
        self.factories['layers'] = _OrderedMemberDict(self, None, None, flags)

        _init_spam_layers(self, prep_layers, povm_layers)  # SPAM

        for gateName in self.processor_spec.gate_names:
            # process gate names (no sslbls, e.g. "Gx", not "Gx:0") - we'll check for the
            # latter when we process the corresponding gate name's availability
            gate_unitary = self.processor_spec.gate_unitaries[gateName]
            resolved_avail = self.processor_spec.resolved_availability(gateName)

            gate_is_idle = gateName in idle_names
            gate_is_factory = callable(gate_unitary)

            if not independent_gates:  # then get our "template" gate ready
                # for non-independent gates, need to specify gate name alone (no sslbls):
                gate = mm_gatedict.get(gateName, None)
                gate_is_factory = gate_is_factory or isinstance(gate, _opfactory.OpFactory)

                if gate is not None:  # (a gate name may not be in gatedict if it's an identity without any noise)
                    if ensure_composed_gates and not isinstance(gate, _op.ComposedOp) and not gate_is_factory:
                        #Make a single ComposedOp *here*, which is used
                        # in all the embeddings for different target qubits
                        gate = _op.ComposedOp([gate], state_space="auto", evotype="auto")  # to make adding factors easy

                    if gate_is_factory:
                        self.factories['gates'][_Lbl(gateName)] = gate
                    else:
                        self.operation_blks['gates'][_Lbl(gateName)] = gate

                    if gate_is_idle and gate.state_space.num_qubits == 1 and global_idle_layer_label is None:
                        # then attempt to turn this 1Q idle into a global idle (for implied idle layers)
                        global_idle = _op.ComposedOp([_op.EmbeddedOp(state_space, (qlbl,), gate)
                                                      for qlbl in qubit_labels])
                        self.operation_blks['layers'][_Lbl('(auto_global_idle)')] = global_idle
                        global_idle_layer_label = layer_rules.global_idle_layer_label = _Lbl('(auto_global_idle)')
            else:
                gate = None  # this is set to something useful in the "elif independent_gates" block below

            if callable(resolved_avail) or resolved_avail == '*':
                # then `gate` has function-determined or arbitrary availability, and we just need to
                # put it in an EmbeddingOpFactory - no need to copy it or look
                # for overrides in `gatedict` - there's always just *one* instance
                # of an arbitrarily available gate or factory.
                base_gate = mm_gatedict[gateName]

                # Note: can't use automatic-embedding b/c we need to force embedding
                # when just ordering doesn't align (e.g. Gcnot:1:0 on 2-qubits needs to embed)
                allowed_sslbls_fn = resolved_avail if callable(resolved_avail) else None
                gate_nQubits = self.processor_spec.gate_num_qubits(gateName)
                embedded_op = _opfactory.EmbeddingOpFactory(state_space, base_gate,
                                                            num_target_labels=gate_nQubits,
                                                            allowed_sslbls_fn=allowed_sslbls_fn)
                self.factories['layers'][_Lbl(gateName)] = embedded_op

            else:  # resolved_avail is a list/tuple of available sslbls for the current gate/factory
                gates_for_auto_global_idle = _collections.OrderedDict()

                for inds in resolved_avail:
                    if _Lbl(gateName, inds) in mm_gatedict and inds is not None:
                        #Allow elements of `gatedict` that *have* sslbls override the
                        # default copy/reference of the "name-only" gate:
                        base_gate = mm_gatedict[_Lbl(gateName, inds)]
                        gate_is_factory = gate_is_factory or isinstance(base_gate, _opfactory.OpFactory)

                        if gate_is_factory:
                            self.factories['gates'][_Lbl(gateName, inds)] = base_gate
                        else:
                            self.operation_blks['gates'][_Lbl(gateName, inds)] = base_gate

                    elif independent_gates:  # then we need to ~copy `gate` so it has indep params
                        gate = mm_gatedict.get(gateName, None)  # was set to `None` above; reset here
                        gate_is_factory = gate_is_factory or isinstance(gate, _opfactory.OpFactory)

                        if gate is not None:  # (may be False if gate is an identity without any noise)
                            if ensure_composed_gates and not gate_is_factory:
                                #Make a single ComposedOp *here*, for *only this* embedding
                                # Don't copy gate here, as we assume it's ok to be shared when we
                                #  have independent composed gates
                                base_gate = _op.ComposedOp([gate], evotype="auto", state_space="auto")
                            else:  # want independent params but not a composed gate, so .copy()
                                base_gate = gate.copy()  # so independent parameters

                            if gate_is_factory:
                                self.factories['gates'][_Lbl(gateName, inds)] = base_gate
                            else:
                                self.operation_blks['gates'][_Lbl(gateName, inds)] = base_gate

                    else:  # (not independent_gates, so `gate` is set to non-None above)
                        base_gate = gate  # already a Composed operator (for easy addition
                        # of factors) if ensure_composed_gates == True and not gate_is_factory

                    if base_gate is None:
                        continue  # end loop here if base_gate is just a perfect identity that shouldn't be added

                    #At this point, `base_gate` is the operator or factory that we want to embed into inds
                    # into inds (except in the special case inds[0] == '*' where we make an EmbeddingOpFactory)
                    try:
                        if gate_is_factory:
                            if inds is None or inds == tuple(qubit_labels):  # then no need to embed
                                embedded_op = base_gate
                            else:
                                embedded_op = _opfactory.EmbeddedOpFactory(state_space, inds, base_gate)
                            self.factories['layers'][_Lbl(gateName, inds)] = embedded_op
                        else:
                            if inds is None or inds == tuple(qubit_labels):  # then no need to embed
                                embedded_op = base_gate
                            else:
                                embedded_op = _op.EmbeddedOp(state_space, inds, base_gate)
                            self.operation_blks['layers'][_Lbl(gateName, inds)] = embedded_op

                            # If a 1Q idle gate (factories not supported yet) then turn this into a global idle
                            if gate_is_idle and base_gate.state_space.num_qubits == 1 \
                               and global_idle_layer_label is None:
                                gates_for_auto_global_idle[inds] = embedded_op

                    except Exception as e:
                        if on_construction_error == 'warn':
                            _warnings.warn("Failed to embed %s gate. Dropping it." % str(_Lbl(gateName, inds)))
                        if on_construction_error in ('warn', 'ignore'): continue
                        else: raise e

                if len(gates_for_auto_global_idle) > 0:  # then create a global idle based on 1Q idle gates
                    global_idle = _op.ComposedOp(list(gates_for_auto_global_idle.values()))
                    global_idle_layer_label = layer_rules.global_idle_layer_label = _Lbl('(auto_global_idle)')
                    self.operation_blks['layers'][_Lbl('(auto_global_idle)')] = global_idle

    def create_processor_spec(self):
        import copy as _copy
        return _copy.deepcopy(self.processor_spec)


class _SimpleCompLayerRules(_LayerRules):

    def __init__(self, global_idle_layer_label, implicit_idle_mode):
        self.global_idle_layer_label = global_idle_layer_label
        self.implicit_idle_mode = implicit_idle_mode  # how to handle implied idles ("blanks") in circuits
        self._add_global_idle_to_all_layers = False

        if implicit_idle_mode is None or implicit_idle_mode == "none":  # no noise on idles
            pass  # just use defaults above
        elif implicit_idle_mode == "add_global":  # add global idle to all layers
            self._add_global_idle_to_all_layers = True
        else:
            raise ValueError("Invalid `implicit_idle_mode`: '%s'" % str(implicit_idle_mode))

    def prep_layer_operator(self, model, layerlbl, caches):
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
        #No cache for preps
        return model.prep_blks['layers'][layerlbl]  # prep_blks['layer'] are full prep ops

    def povm_layer_operator(self, model, layerlbl, caches):
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
        # caches['povm-layers'] *are* just complete layers
        if layerlbl in caches['povm-layers']: return caches['povm-layers'][layerlbl]
        if layerlbl in model.povm_blks['layers']:
            return model.povm_blks['layers'][layerlbl]
        else:
            # See if this effect label could correspond to a *marginalized* POVM, and
            # if so, create the marginalized POVM and add its effects to model.effect_blks['layers']
            #assert(isinstance(layerlbl, _Lbl))  # Sanity check
            povmName = _ot.effect_label_to_povm(layerlbl)
            if povmName in model.povm_blks['layers']:
                # implicit creation of marginalized POVMs whereby an existing POVM name is used with sslbls that
                # are not present in the stored POVM's label.
                mpovm = _povm.MarginalizedPOVM(model.povm_blks['layers'][povmName],
                                               model.state_space, layerlbl.sslbls)  # cache in FUTURE
                mpovm_lbl = _Lbl(povmName, layerlbl.sslbls)
                caches['povm-layers'].update(mpovm.simplify_effects(mpovm_lbl))
                assert(layerlbl in caches['povm-layers']), "Failed to create marginalized effect!"
                return caches['povm-layers'][layerlbl]
            else:
                #raise KeyError(f"Could not build povm/effect for {layerlbl}!")
                raise KeyError("Could not build povm/effect for %s!" % str(layerlbl))

    def operation_layer_operator(self, model, layerlbl, caches):
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
        if layerlbl in caches['complete-layers']: return caches['complete-layers'][layerlbl]
        components = layerlbl.components
        add_idle = (self.global_idle_layer_label is not None) and self._add_global_idle_to_all_layers

        if isinstance(layerlbl, _CircuitLabel):
            op = self._create_op_for_circuitlabel(model, layerlbl)
            caches['complete-layers'][layerlbl] = op
            return op

        if len(components) == 1 and add_idle is False:
            ret = self._layer_component_operation(model, components[0], caches['op-layers'])
        else:
            gblIdle = [model.operation_blks['layers'][self.global_idle_layer_label]] if add_idle else []

            #Note: OK if len(components) == 0, as it's ok to have a composed gate with 0 factors
            ret = _op.ComposedOp(gblIdle + [self._layer_component_operation(model, l, caches['op-layers'])
                                            for l in components],
                                 evotype=model.evotype, state_space=model.state_space)
            model._init_virtual_obj(ret)  # so ret's gpindices get set

        caches['complete-layers'][layerlbl] = ret  # cache the final label value
        return ret

    def _layer_component_operation(self, model, complbl, cache):
        """
        Retrieves the operation corresponding to one component of a layer operation.

        Parameters
        ----------
        complbl : Label
            A component label of a larger layer label.

        Returns
        -------
        LinearOperator
        """
        if complbl in cache:
            return cache[complbl]

        #Note: currently we don't cache complbl because it's not the final
        # label being created, but we could if it would improve performance.
        if isinstance(complbl, _CircuitLabel):
            ret = self._create_op_for_circuitlabel(model, complbl)
        elif complbl in model.operation_blks['layers']:
            ret = model.operation_blks['layers'][complbl]
        else:
            ret = _opfactory.op_from_factories(model.factories['layers'], complbl)
        return ret
