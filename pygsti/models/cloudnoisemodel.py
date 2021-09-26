"""
Defines the CloudNoiseModel class and supporting functions
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
import scipy.sparse as _sps

from pygsti.baseobjs import statespace as _statespace
from pygsti.models.implicitmodel import ImplicitOpModel as _ImplicitOpModel, _init_spam_layers
from pygsti.models.layerrules import LayerRules as _LayerRules
from pygsti.models.memberdict import OrderedMemberDict as _OrderedMemberDict
from pygsti.evotypes import Evotype as _Evotype
from pygsti.forwardsims.forwardsim import ForwardSimulator as _FSim
from pygsti.forwardsims.mapforwardsim import MapForwardSimulator as _MapFSim
from pygsti.forwardsims.matrixforwardsim import MatrixForwardSimulator as _MatrixFSim
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.modelmembers import states as _state
from pygsti.modelmembers.operations import opfactory as _opfactory
from pygsti.modelmembers.modelmembergraph import ModelMemberGraph as _MMGraph
from pygsti.baseobjs.basis import BuiltinBasis as _BuiltinBasis, ExplicitBasis as _ExplicitBasis
from pygsti.baseobjs.label import Label as _Lbl, CircuitLabel as _CircuitLabel
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.baseobjs.qubitgraph import QubitGraph as _QubitGraph
from pygsti.tools import basistools as _bt
from pygsti.tools import internalgates as _itgs
from pygsti.tools import optools as _ot
from pygsti.baseobjs.basisconstructors import sqrt2, id2x2, sigmax, sigmay, sigmaz
from pygsti.processors.processorspec import ProcessorSpec as _ProcessorSpec


class CloudNoiseModel(_ImplicitOpModel):
    """
    A n-qubit model using a low-weight and geometrically local error model with a common "global idle" operation.

    Parameters
    ----------
    num_qubits : int
        The number of qubits

    gatedict : dict
        A dictionary (an `OrderedDict` if you care about insertion order) that
        associates with string-type gate names (e.g. `"Gx"`) :class:`LinearOperator`,
        `numpy.ndarray`, or :class:`OpFactory` objects. When the objects may act on
        fewer than the total number of qubits (determined by their dimension/shape) then
        they are repeatedly embedded into `num_qubits`-qubit gates as specified by their
        `availability`.  These operations represent the ideal target operations, and
        thus, any `LinearOperator` or `OpFactory` objects must be *static*, i.e., have
        zero parameters.

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

    global_idle_layer : LinearOperator
        A global idle operation which acts on all the qubits and
        is, if `add_idle_noise_to_all_gates=True`, composed with the
        actions of specific gates to form the layer operation of
        any circuit layer.

    prep_layers, povm_layers : None or operator or dict or list, optional
        The SPAM operations as n-qubit layer operations.  If `None`, then
        no preps (or POVMs) are created.  If a dict, then the keys are
        labels and the values are layer operators.  If a list, then the
        elements are layer operators and the labels will be assigned as
        "rhoX" and "MX" where X is an integer starting at 0.  If a single
        layer operation is given, then this is used as the sole prep or
        POVM and is assigned the label "rho0" or "Mdefault" respectively.

    build_cloudnoise_fn : function, optional
        A function which takes a single :class:`Label` as an argument and
        returns the cloud-noise operation for that primitive layer
        operation.  Note that if `errcomp_type="gates"` the returned
        operator should be a superoperator whereas if
        `errcomp_type="errorgens"` then the returned operator should be
        an error generator (not yet exponentiated).

    build_cloudkey_fn : function, optional
        An function which takes a single :class:`Label` as an argument and
        returns a "cloud key" for that primitive layer.  The "cloud" is the
        set of qubits that the error (the operator returned from
        `build_cloudnoise_fn`) touches -- and the "key" returned from this
        function is meant to identify that cloud.  This is used to keep track
        of which primitive layer-labels correspond to the same cloud - e.g.
        the cloud-key for ("Gx",2) and ("Gy",2) might be the same and could
        be processed together when selecing sequences that amplify the parameters
        in the cloud-noise operations for these two labels.  The return value
        should be something hashable with the property that two noise
        which act on the same qubits should have the same cloud key.

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

    evotype : Evotype or str, optional
        The evolution type of this model, describing how states are
        represented.  The special value `"default"` is equivalent
        to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

    errcomp_type : {"gates","errorgens"}
        How errors are composed when creating layer operations in the created
        model.  `"gates"` means that the errors on multiple gates in a single
        layer are composed as separate and subsequent processes.  Specifically,
        the layer operation has the form `Composed(target,idleErr,cloudErr)`
        where `target` is a composition of all the ideal gate operations in the
        layer, `idleErr` is idle error (`.operation_blks['layers']['globalIdle']`),
        and `cloudErr` is the composition (ordered as layer-label) of cloud-
        noise contributions, i.e. a map that acts as the product of exponentiated
        error-generator matrices.  `"errorgens"` means that layer operations
        have the form `Composed(target, error)` where `target` is as above and
        `error` results from composing the idle and cloud-noise error
        *generators*, i.e. a map that acts as the exponentiated sum of error
        generators (ordering is irrelevant in this case).

    add_idle_noise_to_all_gates: bool, optional
        Whether the global idle should be added as a factor following the
        ideal action of each of the non-idle gates.

    verbosity : int, optional
        An integer >= 0 dictating how must output to send to stdout.
    """

    def __init__(self, processor_spec, gatedict,
                 prep_layers=None, povm_layers=None,
                 build_cloudnoise_fn=None, build_cloudkey_fn=None,
                 simulator="map", evotype="default", errcomp_type="gates",
                 implicit_idle_mode="none", verbosity=0):

        qubit_labels = processor_spec.qubit_labels
        state_space = _statespace.QubitSpace(qubit_labels)

        simulator = _FSim.cast(simulator, state_space.num_qubits)
        prefer_dense_reps = isinstance(simulator, _MatrixFSim)
        evotype = _Evotype.cast(evotype, default_prefer_dense_reps=prefer_dense_reps)

        # Build gate dictionaries. A value of `gatedict` can be an array, a LinearOperator, or an OpFactory.
        # For later processing, we'll create mm_gatedict to contain each item as a ModelMember.  For cloud-
        # noise models, these gate operations should be *static* (no parameters) as they represent the target
        # operations and all noise (and parameters) are assumed to enter through the cloudnoise members.
        mm_gatedict = _collections.OrderedDict()  # static *target* ops as ModelMembers
        for key, gate in gatedict.items():
            if isinstance(gate, _op.LinearOperator):
                assert(gate.num_params == 0), "Only *static* ideal operators are allowed in `gatedict`!"
                mm_gatedict[key] = gate
            elif isinstance(gate, _opfactory.OpFactory):
                assert(gate.num_params == 0), "Only *static* ideal factories are allowed in `gatedict`!"
                mm_gatedict[key] = gate
            else:  # presumably a numpy array or something like it:
                mm_gatedict[key] = _op.StaticArbitraryOp(gate, evotype, state_space=None)  # use default state space
            assert(mm_gatedict[key]._evotype == evotype), \
                ("Custom gate object supplied in `gatedict` for key %s has evotype %s (!= expected %s)"
                 % (str(key), str(mm_gatedict[key]._evotype), str(evotype)))

        #Set other members
        self.processor_spec = processor_spec
        self.errcomp_type = errcomp_type

        idle_names = self.processor_spec.idle_gate_names
        global_idle_name = self.processor_spec.global_idle_gate_name

        # Set noisy_global_idle_name == global_idle_name if the global idle gate isn't the perfect identity
        #  and if we're generating cloudnoise members (if we're not then layer rules could encouter a key error
        #  if we let noisy_global_idle_name be non-None).
        global_idle_gate = mm_gatedict.get(global_idle_name, None)
        if (global_idle_gate is not None) and (build_cloudnoise_fn is not None) \
           and not (isinstance(global_idle_gate, _op.ComposedOp) and len(global_idle_gate.factorops) == 0):
            noisy_global_idle_name = global_idle_name
        else:
            noisy_global_idle_name = None
        
        assert(set(idle_names).issubset([global_idle_name])), \
            "Only global idle operations are allowed in a CloudNoiseModel!"

        layer_rules = CloudNoiseLayerRules(errcomp_type, noisy_global_idle_name, implicit_idle_mode)
        super(CloudNoiseModel, self).__init__(state_space, layer_rules, "pp", simulator=simulator, evotype=evotype)

        flags = {'auto_embed': False, 'match_parent_statespace': False,
                 'match_parent_evotype': True, 'cast_to_type': None}
        self.prep_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
        self.povm_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
        self.operation_blks['gates'] = _OrderedMemberDict(self, None, None, flags)
        self.operation_blks['cloudnoise'] = _OrderedMemberDict(self, None, None, flags)
        self.operation_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
        self.instrument_blks['layers'] = _OrderedMemberDict(self, None, None, flags)
        self.factories['gates'] = _OrderedMemberDict(self, None, None, flags)
        self.factories['cloudnoise'] = _OrderedMemberDict(self, None, None, flags)
        self.factories['layers'] = _OrderedMemberDict(self, None, None, flags)

        printer = _VerbosityPrinter.create_printer(verbosity)
        printer.log("Creating a %d-qubit cloud-noise model" % self.processor_spec.num_qubits)

        # a dictionary of "cloud" objects
        # keys = cloud identifiers, e.g. (target_qubit_indices, cloud_qubit_indices) tuples
        # values = list of gate-labels giving the gates (primitive layers?) associated with that cloud (necessary?)
        self._clouds = _collections.OrderedDict()

        for gn in self.processor_spec.gate_names:
            # process gate names (no sslbls, e.g. "Gx", not "Gx:0") - we'll check for the
            # latter when we process the corresponding gate name's availability

            gate_unitary = self.processor_spec.gate_unitaries[gn]
            resolved_avail = self.processor_spec.resolved_availability(gn)
            gate = mm_gatedict.get(gn, None)  # a static op or factory, no need to consider if "independent" (no params)
            gate_is_factory = callable(gate_unitary) or isinstance(gate, _opfactory.OpFactory)
            #gate_is_noiseless_identity = (gate is None) or \
            #    (isinstance(gate, _op.ComposedOp) and len(gate.factorops) == 0)

            if gate is not None:  # (a gate name may not be in gatedict if it's an identity without any noise)
                if gate_is_factory:
                    self.factories['gates'][_Lbl(gn)] = gate
                else:
                    self.operation_blks['gates'][_Lbl(gn)] = gate

            if callable(resolved_avail) or resolved_avail == '*':

                # Target operation
                if gate is not None:
                    allowed_sslbls_fn = resolved_avail if callable(resolved_avail) else None
                    gate_nQubits = self.processor_spec.gate_num_qubits(gn)
                    printer.log("Creating %dQ %s gate on arbitrary qubits!!" % (gate_nQubits, gn))
                    self.factories['layers'][_Lbl(gn)] = _opfactory.EmbeddingOpFactory(
                        state_space, gate, num_target_labels=gate_nQubits, allowed_sslbls_fn=allowed_sslbls_fn)
                    # add any primitive ops for this embedding factory?

                # Cloudnoise operation
                if build_cloudnoise_fn is not None:
                    cloudnoise = build_cloudnoise_fn(_Lbl(gn))
                    if cloudnoise is not None:  # build function can return None to signify no noise
                        assert (isinstance(cloudnoise, _opfactory.EmbeddingOpFactory)), \
                            ("`build_cloudnoise_fn` must return an EmbeddingOpFactory for gate %s"
                             " with arbitrary availability") % gn
                        self.factories['cloudnoise'][_Lbl(gn)] = cloudnoise

            else:  # resolved_avail is a list/tuple of available sslbls for the current gate/factory
                for inds in resolved_avail:  # inds are target qubit labels

                    #Target operation
                    if gate is not None:
                        printer.log("Creating %dQ %s gate on qubits %s!!"
                                    % ((len(qubit_labels) if inds is None else len(inds)), gn, inds))
                        assert(inds is None or _Lbl(gn, inds) not in gatedict), \
                            ("Cloudnoise models do not accept primitive-op labels, e.g. %s, in `gatedict` as this dict "
                             "specfies the ideal target gates. Perhaps make the cloudnoise depend on the target qubits "
                             "of the %s gate?") % (str(_Lbl(gn, inds)), gn)

                        if gate_is_factory:
                            self.factories['layers'][_Lbl(gn, inds)] = gate if (inds is None) else \
                                _opfactory.EmbeddedOpFactory(state_space, inds, gate)
                            # add any primitive ops for this factory?
                        else:
                            self.operation_blks['layers'][_Lbl(gn, inds)] = gate if (inds is None) else \
                                _op.EmbeddedOp(state_space, inds, gate)

                    #Cloudnoise operation
                    if build_cloudnoise_fn is not None:
                        cloudnoise = build_cloudnoise_fn(_Lbl(gn, inds))
                        if cloudnoise is not None:  # build function can return None to signify no noise
                            if isinstance(cloudnoise, _opfactory.OpFactory):
                                self.factories['cloudnoise'][_Lbl(gn, inds)] = cloudnoise
                            else:
                                self.operation_blks['cloudnoise'][_Lbl(gn, inds)] = cloudnoise

                    if build_cloudkey_fn is not None:
                        # TODO: is there any way to get a default "key", e.g. the
                        # qubits touched by the corresponding cloudnoise op?
                        # need a way to identify a clound (e.g. Gx and Gy gates on some qubit will have *same* cloud)
                        cloud_key = build_cloudkey_fn(_Lbl(gn, inds))
                        if cloud_key not in self.clouds: self.clouds[cloud_key] = []
                        self.clouds[cloud_key].append(_Lbl(gn, inds))
                    #keep track of the primitive-layer labels in each cloud,
                    # used to specify which gate parameters should be amplifiable by germs for a given cloud (?)
                    # TODO CHECK THIS

        _init_spam_layers(self, prep_layers, povm_layers)  # SPAM

        printer.log("DONE! - created Model with nqubits=%d and op-blks=" % self.state_space.num_qubits)
        for op_blk_lbl, op_blk in self.operation_blks.items():
            printer.log("  %s: %s" % (op_blk_lbl, ', '.join(map(str, op_blk.keys()))))
        self._clean_paramvec()

    def create_processor_spec(self):
        import copy as _copy
        return _copy.deepcopy(self.processor_spec)

    @property
    def clouds(self):
        """
        Returns the set of cloud-sets used when creating sequences which amplify the parameters of this model.

        Returns
        -------
        dict
        """
        return self._clouds

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'processor_spec': self.processor_spec.to_nice_serialization(),
                      'error_composition_mode': self.errcomp_type,
                      })
        mmgraph = self.create_modelmember_graph()
        state['modelmembers'] = mmgraph.create_serialization_dict()
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        state_space = _statespace.StateSpace.from_nice_serialization(state['state_space'])
        #basis = _nice_serialization(state['basis'])
        modelmembers = _MMGraph.load_modelmembers_from_serialization_dict(state['modelmembers'])
        simulator = _FSim.from_nice_serialization(state['simulator'])
        layer_rules = _LayerRules.from_nice_serialization(state['layer_rules'])
        processor_spec = _ProcessorSpec.from_nice_serialization(state['processor_spec'])

        # __init__ does too much, so we need to create an alternate __init__ function here:
        mdl = cls.__new__(cls)
        mdl.processor_spec = processor_spec
        mdl.errcomp_type = state['error_composition_mode']
        _ImplicitOpModel.__init__(mdl, state_space, layer_rules, 'pp',
                                  simulator=simulator, evotype=state['evotype'])

        flags = {'auto_embed': False, 'match_parent_statespace': False,
                 'match_parent_evotype': True, 'cast_to_type': None}
        mdl.prep_blks['layers'] = _OrderedMemberDict(mdl, None, None, flags, modelmembers.get('prep_blks|layers', []))
        mdl.povm_blks['layers'] = _OrderedMemberDict(mdl, None, None, flags, modelmembers.get('povm_blks|layers', []))
        mdl.operation_blks['gates'] = _OrderedMemberDict(mdl, None, None, flags,
                                                         modelmembers.get('operation_blks|gates', []))
        mdl.operation_blks['cloudnoise'] = _OrderedMemberDict(mdl, None, None, flags,
                                                              modelmembers.get('operation_blks|cloudnoise', []))
        mdl.operation_blks['layers'] = _OrderedMemberDict(mdl, None, None, flags,
                                                          modelmembers.get('operation_blks|layers', []))
        mdl.instrument_blks['layers'] = _OrderedMemberDict(mdl, None, None, flags,
                                                           modelmembers.get('instrument_blks|layers', []))
        mdl.factories['gates'] = _OrderedMemberDict(mdl, None, None, flags, modelmembers.get('factories|gates', []))
        mdl.factories['cloudnoise'] = _OrderedMemberDict(mdl, None, None, flags,
                                                         modelmembers.get('factories|cloudnoise', []))
        mdl.factories['layers'] = _OrderedMemberDict(mdl, None, None, flags, modelmembers.get('factories|layers', []))

        mdl._clouds = _collections.OrderedDict()
        mdl._clean_paramvec()

        return mdl


class CloudNoiseLayerRules(_LayerRules):

    def __init__(self, errcomp_type, implied_global_idle_label, implicit_idle_mode):
        self.errcomp_type = errcomp_type
        self.implied_global_idle_label = implied_global_idle_label
        self.implicit_idle_mode = implicit_idle_mode  # how to handle implied idles ("blanks") in circuits
        self._add_global_idle_to_all_layers = False

        if implicit_idle_mode is None or implicit_idle_mode == "none":  # no noise on idles
            pass  # just use defaults above
        elif implicit_idle_mode == "add_global":  # add global idle to all layers
            self._add_global_idle_to_all_layers = True
        else:
            raise ValueError("Invalid `implicit_idle_mode`: '%s'" % str(implicit_idle_mode))

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'error_composition_mode': self.errcomp_type,
                      'implied_global_idle_label': (str(self.implied_global_idle_label)
                                                    if (self.implied_global_idle_label is not None) else None),
                      'implicit_idle_mode': self.implicit_idle_mode,
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        from pygsti.circuits.circuitparser import parse_label as _parse_label
        gi_label = _parse_label(state['implied_global_idle_label']) \
            if (state['implied_global_idle_label'] is not None) else None
        return cls(state['error_composition_mode'], gi_label, state['implicit_idle_mode'])

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
        #Note: cache uses 'op-layers' for *simple target* layers, not complete ones
        if layerlbl in caches['complete-layers']: return caches['complete-layers'][layerlbl]

        if isinstance(layerlbl, _CircuitLabel):
            op = self._create_op_for_circuitlabel(model, layerlbl)
            caches['complete-layers'][layerlbl] = op
            return op

        Composed = _op.ComposedOp
        ExpErrorgen = _op.ExpErrorgenOp
        Sum = _op.ComposedErrorgen
        add_idle = (self.implied_global_idle_label is not None) and self._add_global_idle_to_all_layers
        #print("DB: CloudNoiseLayerLizard building gate %s for %s w/comp-type %s" %
        #      (('matrix' if dense else 'map'), str(oplabel), self.errcomp_type) )

        components = layerlbl.components
        if (len(components) == 0 and self.implied_global_idle_label is not None) \
           or components == (self.implied_global_idle_label,):
            if self.errcomp_type == "gates":
                return model.operation_blks['cloudnoise'][self.implied_global_idle_label]  # idle!
            elif self.errcomp_type == "errorgens":
                return ExpErrorgen(model.operation_blks['cloudnoise'][self.implied_global_idle_label])
            else:
                raise ValueError("Invalid errcomp_type in CloudNoiseLayerRules: %s" % str(self.errcomp_type))

        #Compose target operation from layer's component labels, which correspond
        # to the perfect (embedded) target ops in op_blks
        if len(components) > 1:
            #Note: _layer_component_targetop can return `None` for a (static) identity op
            to_compose = [self._layer_component_targetop(model, l, caches['op-layers']) for l in components]
            targetOp = Composed([op for op in to_compose if op is not None],
                                evotype=model.evotype, state_space=model.state_space)
        else:
            targetOp = self._layer_component_targetop(model, components[0], caches['op-layers'])
        ops_to_compose = [targetOp] if (targetOp is not None) else []

        if self.errcomp_type == "gates":
            if add_idle: ops_to_compose.append(model.operation_blks['cloudnoise'][self.implied_global_idle_label])
            component_cloudnoise_ops = self._layer_component_cloudnoises(model, components, caches['op-cloudnoise'])
            if len(component_cloudnoise_ops) > 0:
                if len(component_cloudnoise_ops) > 1:
                    localErr = Composed(component_cloudnoise_ops,
                                        evotype=model.evotype, state_space=model.state_space)
                else:
                    localErr = component_cloudnoise_ops[0]
                ops_to_compose.append(localErr)

        elif self.errcomp_type == "errorgens":
            #We compose the target operations to create a
            # final target op, and compose this with a *single* ExpErrorgen operation which has as
            # its error generator the composition (sum) of all the factors' error gens.
            errorGens = [model.operation_blks['cloudnoise'][self.implied_global_idle_label]] if add_idle else []
            errorGens.extend(self._layer_component_cloudnoises(model, components, caches['op-cloudnoise']))
            if len(errorGens) > 0:
                if len(errorGens) > 1:
                    error = ExpErrorgen(Sum(errorGens, state_space=model.state_space, evotype=model.evotype))
                else:
                    error = ExpErrorgen(errorGens[0])
                ops_to_compose.append(error)
        else:
            raise ValueError("Invalid errcomp_type in CloudNoiseLayerRules: %s" % str(self.errcomp_type))

        ret = Composed(ops_to_compose, evotype=model.evotype, state_space=model.state_space)
        model._init_virtual_obj(ret)  # so ret's gpindices get set
        caches['complete-layers'][layerlbl] = ret  # cache the final label value
        return ret

    def _layer_component_targetop(self, model, complbl, cache):
        """
        Retrieves the target- or ideal-operation portion of one component of a layer operation.

        Parameters
        ----------
        complbl : Label
            A component label of a larger layer label.

        Returns
        -------
        LinearOperator
        """
        if complbl in cache:
            return cache[complbl]  # caches['op-layers'] would hold "simplified" instrument members

        if complbl == self.implied_global_idle_label:
            # special case of the implied global idle, which give `None` instead of the
            # identity as its target operation since we don't want to include an unnecesseary idle op.
            return None

        if isinstance(complbl, _CircuitLabel):
            raise NotImplementedError("Cloud noise models cannot simulate circuits with partial-layer subcircuits.")
            # In the FUTURE, could easily implement this for errcomp_type == "gates", but it's unclear what to
            #  do for the "errorgens" case - how do we gate an error generator of an entire (mulit-layer) sub-circuit?
            # Maybe we just need to expand the label and create a composition of those layers?
        elif complbl in model.operation_blks['layers']:
            return model.operation_blks['layers'][complbl]
        else:
            return _opfactory.op_from_factories(model.factories['layers'], complbl)

    def _layer_component_cloudnoises(self, model, complbl_list, cache):
        """
        Retrieves cloud-noise portion of the components of a layer operation.

        Get any present cloudnoise ops from a list of components.  This function processes
        a list rather than an item because it's OK if some components don't have
        corresponding cloudnoise ops - we just leave those off.

        Parameters
        ----------
        complbl_list : list
            A list of circuit-layer component labels.

        Returns
        -------
        list
        """
        ret = []
        for complbl in complbl_list:
            if complbl in cache:
                ret.append(cache[complbl])  # caches['cloudnoise-layers'] would hold "simplified" instrument members
            elif complbl in model.operation_blks['cloudnoise']:
                ret.append(model.operation_blks['cloudnoise'][complbl])
            else:
                try:
                    ret.append(_opfactory.op_from_factories(model.factories['cloudnoise'], complbl))
                except KeyError: pass  # OK if cloudnoise doesn't exist (means no noise)
        return ret
