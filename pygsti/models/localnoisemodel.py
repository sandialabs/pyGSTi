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
import numpy as _np

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
from pygsti.modelmembers.modelmembergraph import ModelMemberGraph as _MMGraph
from pygsti.baseobjs.basis import BuiltinBasis as _BuiltinBasis
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.label import Label as _Lbl, CircuitLabel as _CircuitLabel
from pygsti.circuits.circuitparser import parse_label as _parse_label
from pygsti.tools import basistools as _bt
from pygsti.tools import internalgates as _itgs
from pygsti.tools import optools as _ot
from pygsti.tools import listtools as _lt
from pygsti.processors.processorspec import ProcessorSpec as _ProcessorSpec, QubitProcessorSpec as _QubitProcessorSpec


class LocalNoiseModel(_ImplicitOpModel):
    """
    A n-qudit implicit model that allows for only local noise.

    This model holds as building blocks individual noisy gates
    which are trivially embedded into circuit layers as requested.

    Parameters
    ----------
    processor_spec : ProcessorSpec
        The processor specification to create a model for.  This object specifies the
        gate names and unitaries for the processor, and their availability on the
        processor.

    gatedict : dict
        A dictionary (an `OrderedDict` if you care about insertion order) that
        associates with gate names (e.g. `"Gx"`) :class:`LinearOperator`,
        `numpy.ndarray` objects. When the objects may act on fewer than the total
        number of qudits (determined by their dimension/shape) then they are
        repeatedly embedded into operation on the entire state space as specified
        by their availability within `processor_spec`.  While the keys of this
        dictionary are usually string-type gate *names*, labels that include target
        qudits, e.g. `("Gx",0)`, may be used to override the default behavior of
        embedding a reference or a copy of the gate associated with the same label
        minus the target qudits (e.g. `"Gx"`).  Furthermore, :class:`OpFactory` objects
        may be used in place of `LinearOperator` objects to allow the evaluation of labels
        with arguments.

    prep_layers : None or operator or dict or list
        The state preparateion operations as n-qudit layer operations.  If
        `None`, then no state preparations will be present in the created model.
        If a dict, then the keys are labels and the values are layer operators.
        If a list, then the elements are layer operators and the labels will be
        assigned as "rhoX" where X is an integer starting at 0.  If a single
        layer operation of type :class:`State` is given, then this is used as
        the sole prep and is assigned the label "rho0".

    povm_layers : None or operator or dict or list
        The state preparateion operations as n-qudit layer operations.  If
        `None`, then no POVMS will be present in the created model.  If a dict,
        then the keys are labels and the values are layer operators.  If a list,
        then the elements are layer operators and the labels will be assigned as
        "MX" where X is an integer starting at 0.  If a single layer operation
        of type :class:`POVM` is given, then this is used as the sole POVM and
        is assigned the label "Mdefault".

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

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
        qudits may have different local noise, and so the
        `operation_bks['gates']` dictionary contains a key for each gate
        available gate placement.

    ensure_composed_gates : bool, optional
        If True then the elements of the `operation_bks['gates']` will always
        be :class:`ComposedOp` objects.  The purpose of this is to
        facilitate modifying the gate operations after the model is created.
        If False, then the appropriately parameterized gate objects (often
        dense gates) are used directly.

    implicit_idle_mode : {'none', 'add_global', 'pad_1Q', 'only_global'}
        The way idle operations are added implicitly within the created model. `"none"`
        doesn't add any "extra" idle operations when there is a layer that contains some
        gates but not gates on all the qubits.  `"add_global"` adds the global idle operation,
        i.e., the operation for a global idle layer (zero gates - a completely empty layer),
        to every layer that is simulated, using the global idle as a background idle that always
        occurs regardless of the operation.  `"pad_1Q"` applies the 1-qubit idle gate (if one
        exists) to all idling qubits within a circuit layer. `"only_global"` uses a global idle
        operation, if one exists, to simulate the completely empty layer but nothing else, i.e.,
        this idle operation is *not* added to other layers as in `"add_global"`.
    """

    def __init__(self, processor_spec, gatedict, prep_layers=None, povm_layers=None, evotype="default",
                 simulator="auto", on_construction_error='raise',
                 independent_gates=False, ensure_composed_gates=False, implicit_idle_mode="none"):

        qudit_labels = processor_spec.qudit_labels
        state_space = _statespace.QubitSpace(qudit_labels) if isinstance(processor_spec, _QubitProcessorSpec) \
            else _statespace.QuditSpace(qudit_labels, processor_spec.qudit_udims)

        simulator = _FSim.cast(simulator,
                               state_space.num_qubits if isinstance(state_space, _statespace.QubitSpace) else None)
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
                mm_gatedict[key] = _op.StaticArbitraryOp(gate, basis=None, evotype=evotype,
                                                         state_space=state_space)  # static gates by default

        self.processor_spec = processor_spec
        idle_names = processor_spec.idle_gate_names
        global_idle_layer_label = processor_spec.global_idle_layer_label

        layer_rules = _SimpleCompLayerRules(qudit_labels, implicit_idle_mode, None, global_idle_layer_label)

        super(LocalNoiseModel, self).__init__(state_space, layer_rules, 'pp',
                                              simulator=simulator, evotype=evotype)

        flags = {'auto_embed': False, 'match_parent_statespace': False,
                 'match_parent_evotype': True, 'cast_to_type': None}
        self.prep_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
        self.povm_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
        self.operation_blks['gates'] = _OrderedMemberDict(self, None, None, flags)
        self.operation_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
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
                        # in all the embeddings for different target qudits
                        gate = _op.ComposedOp([gate], state_space="auto", evotype="auto")  # to make adding factors easy

                    if gate_is_factory:
                        self.factories['gates'][_Lbl(gateName)] = gate
                    else:
                        self.operation_blks['gates'][_Lbl(gateName)] = gate

                    if gate_is_idle and gate.state_space.num_qudits == 1 and global_idle_layer_label is None:
                        # then attempt to turn this 1Q idle into a global idle (for implied idle layers)
                        global_idle = _op.ComposedOp([_op.EmbeddedOp(state_space, (qlbl,), gate)
                                                      for qlbl in qudit_labels])
                        self.operation_blks['layers'][_Lbl('{auto_global_idle}')] = global_idle
                        global_idle_layer_label = layer_rules.global_idle_layer_label = _Lbl('{auto_global_idle}')
            else:
                gate = None  # this is set to something useful in the "elif independent_gates" block below

            if callable(resolved_avail) or resolved_avail == '*':
                # then `gate` has function-determined or arbitrary availability, and we just need to
                # put it in an EmbeddingOpFactory - no need to copy it or look
                # for overrides in `gatedict` - there's always just *one* instance
                # of an arbitrarily available gate or factory.
                base_gate = mm_gatedict[gateName]

                # Note: can't use automatic-embedding b/c we need to force embedding
                # when just ordering doesn't align (e.g. Gcnot:1:0 on 2-qudits needs to embed)
                allowed_sslbls_fn = resolved_avail if callable(resolved_avail) else None
                gate_nQudits = self.processor_spec.gate_num_qudits(gateName)
                embedded_op = _opfactory.EmbeddingOpFactory(state_space, base_gate,
                                                            num_target_labels=gate_nQudits,
                                                            allowed_sslbls_fn=allowed_sslbls_fn)
                self.factories['layers'][_Lbl(gateName)] = embedded_op

            else:  # resolved_avail is a list/tuple of available sslbls for the current gate/factory
                singleQ_idle_layer_labels = _collections.OrderedDict()

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
                            if inds is None or inds == tuple(qudit_labels):  # then no need to embed
                                embedded_op = base_gate
                            else:
                                embedded_op = _opfactory.EmbeddedOpFactory(state_space, inds, base_gate)
                            self.factories['layers'][_Lbl(gateName, inds)] = embedded_op
                        else:
                            if inds is None or inds == tuple(qudit_labels):  # then no need to embed
                                embedded_op = base_gate
                            else:
                                embedded_op = _op.EmbeddedOp(state_space, inds, base_gate)
                            self.operation_blks['layers'][_Lbl(gateName, inds)] = embedded_op

                            # If a 1Q idle gate (factories not supported yet) then turn this into a global idle
                            if gate_is_idle and base_gate.state_space.num_qubits == 1 \
                               and global_idle_layer_label is None:
                                singleQ_idle_layer_labels[inds] = _Lbl(gateName, inds)  # allow custom setting of this?

                    except Exception as e:
                        if on_construction_error == 'warn':
                            _warnings.warn("Failed to embed %s gate. Dropping it." % str(_Lbl(gateName, inds)))
                        if on_construction_error in ('warn', 'ignore'): continue
                        else: raise e

                if len(singleQ_idle_layer_labels) > 0:
                    if implicit_idle_mode in ('add_global', 'only_global') and global_idle_layer_label is None:
                        # then create a global idle based on 1Q idle gates
                        global_idle = _op.ComposedOp([self.operation_blks['layers'][lbl]
                                                      for lbl in singleQ_idle_layer_labels.values()])
                        global_idle_layer_label = layer_rules.global_idle_layer_label = _Lbl('{auto_global_idle}')
                        self.operation_blks['layers'][_Lbl('{auto_global_idle}')] = global_idle
                    elif implicit_idle_mode == 'pad_1Q':
                        layer_rules.single_qubit_idle_layer_labels = singleQ_idle_layer_labels

        self._clean_paramvec()

    def create_processor_spec(self):
        import copy as _copy
        return _copy.deepcopy(self.processor_spec)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'processor_spec': self.processor_spec.to_nice_serialization(),
                      })
        mmgraph = self.create_modelmember_graph()
        state['modelmembers'] = mmgraph.create_serialization_dict()
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        state_space = _statespace.StateSpace.from_nice_serialization(state['state_space'])
        #basis = _from_nice_serialization(state['basis'])
        simulator = _FSim.from_nice_serialization(state['simulator'])
        layer_rules = _LayerRules.from_nice_serialization(state['layer_rules'])
        processor_spec = _ProcessorSpec.from_nice_serialization(state['processor_spec'])
        param_labels = state.get('parameter_labels', None)
        param_bounds = state.get('parameter_bounds', None)

        # __init__ does too much, so we need to create an alternate __init__ function here:
        mdl = cls.__new__(cls)
        mdl.processor_spec = processor_spec
        _ImplicitOpModel.__init__(mdl, state_space, layer_rules, 'pp',
                                  simulator=simulator, evotype=state['evotype'])

        modelmembers = _MMGraph.load_modelmembers_from_serialization_dict(state['modelmembers'], mdl)
        flags = {'auto_embed': False, 'match_parent_statespace': False,
                 'match_parent_evotype': True, 'cast_to_type': None}
        mdl.prep_blks['layers'] = _OrderedMemberDict(mdl, None, None, flags, modelmembers.get('prep_blks|layers', []))
        mdl.povm_blks['layers'] = _OrderedMemberDict(mdl, None, None, flags, modelmembers.get('povm_blks|layers', []))
        mdl.operation_blks['gates'] = _OrderedMemberDict(mdl, None, None, flags,
                                                         modelmembers.get('operation_blks|gates', []))
        mdl.operation_blks['layers'] = _OrderedMemberDict(mdl, None, None, flags,
                                                          modelmembers.get('operation_blks|layers', []))
        mdl.instrument_blks['layers'] = _OrderedMemberDict(mdl, None, None, flags,
                                                           modelmembers.get('instrument_blks|layers', []))
        mdl.factories['gates'] = _OrderedMemberDict(mdl, None, None, flags, modelmembers.get('factories|gates', []))
        mdl.factories['layers'] = _OrderedMemberDict(mdl, None, None, flags, modelmembers.get('factories|layers', []))
        mdl._clean_paramvec()

        Np = len(mdl._paramlbls)  # _clean_paramvec sets up ._paramlbls so its length == # of params
        if param_labels and len(param_labels) == Np:
            mdl._paramlbls[:] = [_lt.lists_to_tuples(lbl) for lbl in param_labels]
        if param_bounds is not None:
            param_bounds = cls._decodemx(param_bounds)
            if param_bounds.shape == (Np, 2):
                mdl._param_bounds

        return mdl

    def _op_decomposition(self, op_label):
        """Returns the target and error-generator-containing error map parts of the operation for `op_label` """
        return self.operation_blks['layers'][op_label], self.operation_blks['layers'][op_label]

    def errorgen_coefficients(self, normalized_elem_gens=True):
        """TODO: docstring - returns a nested dict containing all the error generator coefficients for all
           the operations in this model. """
        if not normalized_elem_gens:
            def rescale(coeffs):
                """ HACK: rescales errorgen coefficients for normalized-Pauli-basis elementary error gens
                         to be coefficients for the usual un-normalied-Pauli-basis elementary gens.  This
                         is only needed in the Hamiltonian case, as the non-ham "elementary" gen has a
                         factor of d2 baked into it.
                """
                d2 = _np.sqrt(self.dim); d = _np.sqrt(d2)
                return {lbl: (val / d if lbl.errorgen_type == 'H' else val) for lbl, val in coeffs.items()}

            op_coeffs = {op_label: rescale(self.operation_blks['layers'][op_label].errorgen_coefficients())
                         for op_label in self.operation_blks['layers']}
            op_coeffs.update({prep_label: rescale(self.prep_blks['layers'][prep_label].errorgen_coefficients())
                              for prep_label in self.prep_blks['layers']})
            op_coeffs.update({povm_label: rescale(self.povm_blks['layers'][povm_label].errorgen_coefficients())
                              for povm_label in self.povm_blks['layers']})
        else:
            op_coeffs = {op_label: self.operation_blks['layers'][op_label].errorgen_coefficients()
                         for op_label in self.operation_blks['layers']}
            op_coeffs.update({prep_label: self.prep_blks['layers'][prep_label].errorgen_coefficients()
                              for prep_label in self.prep_blks['layers']})
            op_coeffs.update({povm_label: self.povm_blks['layers'][povm_label].errorgen_coefficients()
                              for povm_label in self.povm_blks['layers']})

        return op_coeffs


class _SimpleCompLayerRules(_LayerRules):

    def __init__(self, qubit_labels, implicit_idle_mode, singleq_idle_layer_labels, global_idle_layer_label):
        super().__init__()
        self.implicit_idle_mode = implicit_idle_mode  # how to handle implied idles ("blanks") in circuits
        self.qubit_labels = qubit_labels
        self._use_global_idle = False
        self._add_global_idle_to_all_layers = False
        self._add_padded_idle = False
        self.use_op_caching = True  # expert functionality - can be turned off if needed

        if implicit_idle_mode not in ('none', 'add_global', 'only_global', 'pad_1Q'):
            raise ValueError("Invalid `implicit_idle_mode`: '%s'" % str(implicit_idle_mode))

        self.single_qubit_idle_layer_labels = singleq_idle_layer_labels
        self.global_idle_layer_label = global_idle_layer_label

    @property
    def single_qubit_idle_layer_labels(self):
        return self._singleq_idle_layer_labels

    @single_qubit_idle_layer_labels.setter
    def single_qubit_idle_layer_labels(self, val):
        self._singleq_idle_layer_labels = val
        if self.implicit_idle_mode == "pad_1Q":
            self._add_padded_idle = bool(self._singleq_idle_layer_labels is not None)

    @property
    def global_idle_layer_label(self):
        return self._global_idle_layer_label

    @global_idle_layer_label.setter
    def global_idle_layer_label(self, val):
        self._global_idle_layer_label = val
        if self.implicit_idle_mode == "add_global":
            self._add_global_idle_to_all_layers = bool(self._global_idle_layer_label is not None)
        elif self.implicit_idle_mode == "only_global":
            self._use_global_idle = bool(self._global_idle_layer_label is not None)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        assert((self.single_qubit_idle_layer_labels is None)
               or all([len(k) == 1 for k in self.single_qubit_idle_layer_labels.keys()])), \
            "All keys of single_qubit_idle_layer_labels should be 1-tuples of a *single* sslbl!"
        state.update({'global_idle_layer_label': (str(self.global_idle_layer_label)
                                                  if (self.global_idle_layer_label is not None) else None),
                      'single_qubit_idle_layer_labels': ({str(sslbls[0]): str(idle_lbl) for sslbls, idle_lbl
                                                          in self.single_qubit_idle_layer_labels.items()}
                                                         if self.single_qubit_idle_layer_labels is not None else None),
                      'implicit_idle_mode': self.implicit_idle_mode,
                      'qubit_labels': list(self.qubit_labels),
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        def _to_int(x):  # (same as in slowcircuitparser.py)
            return int(x) if x.isdigit() else x

        global_idle_label = _parse_label(state['global_idle_layer_label']) \
            if (state['global_idle_layer_label'] is not None) else None

        if state.get('single_qubit_idle_layer_labels', None) is not None:
            singleQ_idle_lbls = {(_to_int(k),): _parse_label(v)
                                 for k, v in state['single_qubit_idle_layer_labels'].items()}
        else:
            singleQ_idle_lbls = None

        qubit_labels = tuple(state['qubit_labels']) if ('qubit_labels' in state) else None
        return cls(qubit_labels, state['implicit_idle_mode'], singleQ_idle_lbls, global_idle_label)

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
                simplified_effects = mpovm.simplify_effects(mpovm_lbl)
                assert(layerlbl in simplified_effects), "Failed to create marginalized effect!"
                if self.use_op_caching:
                    caches['povm-layers'].update(simplified_effects)
                return simplified_effects[layerlbl]
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
        lbl = _Lbl(layerlbl) if isinstance(layerlbl, list) else layerlbl
        if lbl in caches['complete-layers']: return caches['complete-layers'][lbl]
        components = lbl.components
        use_global_idle = self._use_global_idle
        add_global_idle = self._add_global_idle_to_all_layers
        add_padded_idle = self._add_padded_idle
        add_idle = add_global_idle or add_padded_idle

        if isinstance(layerlbl, _CircuitLabel):
            op = self._create_op_for_circuitlabel(model, layerlbl)
            if self.use_op_caching: caches['complete-layers'][layerlbl] = op
            return op

        if len(components) == 1 and add_idle is False:
            ret = self._layer_component_operation(model, components[0], caches['op-layers'])
        elif add_padded_idle:
            component_sslbls = [c.sslbls for c in components]
            if None not in component_sslbls:  # sslbls == None => label covers *all* labels, no padding needed
                present_sslbl_components = set(_itertools.chain(*[sslbls for sslbls in component_sslbls]))
                absent_sslbls = [sslbl for sslbl in self.qubit_labels if (sslbl not in present_sslbl_components)]
                factors = {sslbl: model.operation_blks['layers'][self.single_qubit_idle_layer_labels[(sslbl,)]]
                           for sslbl in absent_sslbls}  # key = *lowest* (and only) sslbl
            else:
                factors = {}

            factors.update({(sorted(l.sslbls)[0] if (l.sslbls is not None) else 0):
                            self._layer_component_operation(model, l, caches['op-layers'])
                            for l in components})  # key = *lowest* sslbl or 0 (special case) if sslbls is None

            #Note: OK if len(components) == 0, as it's ok to have a composed gate with 0 factors
            ret = _op.ComposedOp([factors[k] for k in sorted(factors.keys())],
                                 evotype=model.evotype, state_space=model.state_space)
            model._init_virtual_obj(ret)  # so ret's gpindices get set
        else:  # add_global_idle
            gblIdle = [model.operation_blks['layers'][self.global_idle_layer_label]] \
                if (use_global_idle or add_idle) else []

            #Note: OK if len(components) == 0, as it's ok to have a composed gate with 0 factors
            ret = _op.ComposedOp(gblIdle + [self._layer_component_operation(model, l, caches['op-layers'])
                                            for l in components],
                                 evotype=model.evotype, state_space=model.state_space, allocated_to_parent=model)
            model._init_virtual_obj(ret)  # so ret's gpindices get set - I don't think this is needed...

        if self.use_op_caching:
            caches['complete-layers'][lbl] = ret  # cache the final label value
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
