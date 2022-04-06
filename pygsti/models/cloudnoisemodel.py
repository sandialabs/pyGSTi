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
from pygsti.baseobjs.basis import Basis as _Basis, TensorProdBasis as _TensorProdBasis
from pygsti.baseobjs.label import Label as _Lbl, CircuitLabel as _CircuitLabel
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.baseobjs.qubitgraph import QubitGraph as _QubitGraph
from pygsti.tools import basistools as _bt
from pygsti.tools import internalgates as _itgs
from pygsti.tools import optools as _ot
from pygsti.baseobjs.basisconstructors import sqrt2, id2x2, sigmax, sigmay, sigmaz
from pygsti.processors.processorspec import ProcessorSpec as _ProcessorSpec, QubitProcessorSpec as _QubitProcessorSpec


class CloudNoiseModel(_ImplicitOpModel):
    """
    A n-qudit model using a low-weight and geometrically local error model with a common "global idle" operation.

    Parameters
    ----------
    processor_spec : ProcessorSpec
        The processor specification to create a model for.  This object specifies the
        gate names and unitaries for the processor, and their availability on the
        processor.

    gatedict : dict
        A dictionary (an `OrderedDict` if you care about insertion order) that
        associates with string-type gate names (e.g. `"Gx"`) :class:`LinearOperator`,
        `numpy.ndarray`, or :class:`OpFactory` objects. When the objects may act on
        fewer than the total number of qudits (determined by their dimension/shape) then
        they are repeatedly embedded into operation on the entire state space as specified
        by their availability within `processor_spec`.  These operations represent the ideal
        target operations, and thus, any `LinearOperator` or `OpFactory` objects must be *static*,
        i.e., have zero parameters.

    prep_layers, povm_layers : None or operator or dict or list, optional
        The SPAM operations as n-qudit layer operations.  If `None`, then
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
        set of qudits that the error (the operator returned from
        `build_cloudnoise_fn`) touches -- and the "key" returned from this
        function is meant to identify that cloud.  This is used to keep track
        of which primitive layer-labels correspond to the same cloud - e.g.
        the cloud-key for ("Gx",2) and ("Gy",2) might be the same and could
        be processed together when selecing sequences that amplify the parameters
        in the cloud-noise operations for these two labels.  The return value
        should be something hashable with the property that two noise
        which act on the same qudits should have the same cloud key.

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

    implicit_idle_mode : {'none', 'add_global', 'pad_1Q'}
        The way idle operations are added implicitly within the created model. `"none"`
        doesn't add any "extra" idle operations when there is a layer that contains some
        gates but not gates on all the qubits.  `"add_global"` adds the global idle operation,
        i.e., the operation for a global idle layer (zero gates - a completely empty layer),
        to every layer that is simulated, using the global idle as a background idle that always
        occurs regardless of the operation.  `"pad_1Q"` applies the 1-qubit idle gate (if one
        exists) to all idling qubits within a circuit layer.

    verbosity : int, optional
        An integer >= 0 dictating how must output to send to stdout.
    """

    def __init__(self, processor_spec, gatedict,
                 prep_layers=None, povm_layers=None,
                 build_cloudnoise_fn=None, build_cloudkey_fn=None,
                 simulator="map", evotype="default", errcomp_type="gates",
                 implicit_idle_mode="none", verbosity=0):

        qudit_labels = processor_spec.qudit_labels
        state_space = _statespace.QubitSpace(qudit_labels) if isinstance(processor_spec, _QubitProcessorSpec) \
            else _statespace.QuditSpace(qudit_labels, processor_spec.qudit_udims)

        simulator = _FSim.cast(simulator,
                               state_space.num_qubits if isinstance(state_space, _statespace.QubitSpace) else None)
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
           and (build_cloudnoise_fn(self.processor_spec.global_idle_layer_label) is not None):
            noisy_global_idle_name = global_idle_name
        else:
            noisy_global_idle_name = None

        singleq_idle_layer_labels = {}
        for idle_name in idle_names:
            if self.processor_spec.gate_num_qubits(idle_name) == 1:
                for idlelayer_sslbls in self.processor_spec.resolved_availability(idle_name, 'tuple'):
                    if idlelayer_sslbls is None: continue  # case of 1Q model with "global" idle
                    assert(len(idlelayer_sslbls) == 1)  # should be a 1-qubit gate!
                    if idlelayer_sslbls not in singleq_idle_layer_labels:
                        singleq_idle_layer_labels[idlelayer_sslbls] = _Lbl(idle_name, idlelayer_sslbls)
        #assert(set(idle_names).issubset([global_idle_name])), \
        #    "Only global idle operations are allowed in a CloudNoiseModel!"

        layer_rules = CloudNoiseLayerRules(errcomp_type, qudit_labels, implicit_idle_mode, singleq_idle_layer_labels,
                                           noisy_global_idle_name)
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
        printer.log("Creating a %d-qudit cloud-noise model" % self.processor_spec.num_qudits)

        # a dictionary of "cloud" objects
        # keys = cloud identifiers, e.g. (target_qudit_indices, cloud_qudit_indices) tuples
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
                    gate_nQudits = self.processor_spec.gate_num_qudits(gn)
                    printer.log("Creating %dQ %s gate on arbitrary qudits!!" % (gate_nQudits, gn))
                    self.factories['layers'][_Lbl(gn)] = _opfactory.EmbeddingOpFactory(
                        state_space, gate, num_target_labels=gate_nQudits, allowed_sslbls_fn=allowed_sslbls_fn)
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
                for inds in resolved_avail:  # inds are target qudit labels

                    #Target operation
                    if gate is not None:
                        printer.log("Creating %dQ %s gate on qudits %s!!"
                                    % ((len(qudit_labels) if inds is None else len(inds)), gn, inds))
                        assert(inds is None or _Lbl(gn, inds) not in gatedict), \
                            ("Cloudnoise models do not accept primitive-op labels, e.g. %s, in `gatedict` as this dict "
                             "specfies the ideal target gates. Perhaps make the cloudnoise depend on the target qudits "
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
                        # qudits touched by the corresponding cloudnoise op?
                        # need a way to identify a clound (e.g. Gx and Gy gates on some qudit will have *same* cloud)
                        cloud_key = build_cloudkey_fn(_Lbl(gn, inds))
                        if cloud_key not in self.clouds: self.clouds[cloud_key] = []
                        self.clouds[cloud_key].append(_Lbl(gn, inds))
                    #keep track of the primitive-layer labels in each cloud,
                    # used to specify which gate parameters should be amplifiable by germs for a given cloud (?)
                    # TODO CHECK THIS

        _init_spam_layers(self, prep_layers, povm_layers)  # SPAM

        printer.log("DONE! - created Model with nqudits=%d and op-blks=" % self.state_space.num_qudits)
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

    def _add_reparameterization(self, primitive_op_labels, fogi_dirs, errgenset_space_labels):
        raise NotImplementedError("TODO: need to implement this for implicit model FOGI parameterization to work!")

    def setup_fogi(self, initial_gauge_basis, create_complete_basis_fn=None,
                   op_label_abbrevs=None, reparameterize=False, reduce_to_model_space=True,
                   dependent_fogi_action='drop', include_spam=True):

        from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis as _CompleteElementaryErrorgenBasis
        from pygsti.baseobjs.errorgenbasis import ExplicitElementaryErrorgenBasis as _ExplicitElementaryErrorgenBasis
        from pygsti.baseobjs.errorgenspace import ErrorgenSpace as _ErrorgenSpace
        #import scipy.sparse as _sps

        #from pygsti.tools import basistools as _bt
        from pygsti.tools import fogitools as _fogit
        from pygsti.models.fogistore import FirstOrderGaugeInvariantStore as _FOGIStore

        # ExplicitOpModel-specific - and assumes model's ops have specific structure (see extract_std_target*) !!
        primitive_op_labels = self.primitive_op_labels

        primitive_prep_labels = self.primitive_prep_labels if include_spam else []
        primitive_povm_labels = self.primitive_povm_labels if include_spam else []

        # "initial" gauge space is the space of error generators initially considered as
        # gauge transformations.  It can be reduced by the errors allowed on operations (by
        # their type and support).

        def extract_std_target_mx(op, op_basis):
            # TODO: more general decomposition of op - here it must be Composed(UnitaryOp, ExpErrorGen)
            #       or just ExpErrorGen
            if isinstance(op, _op.EmbeddedOp):
                all_sslbls = op.state_space.sole_tensor_product_block_labels
                op_component_bases = [op_basis.component_bases[all_sslbls.index(lbl)] for lbl in op.target_labels]
                op_basis = _TensorProdBasis(op_component_bases)
                return extract_std_target_mx(op.embedded_op, op_basis)
            elif isinstance(op, _op.ExpErrorgenOp):  # assume just an identity op
                U = _np.identity(op.state_space.dim, 'd')
            elif isinstance(op, _op.ComposedOp):  # ASSUMES first element gives unitary
                op_mx = op.factorops[0].to_dense()  # ASSUMES first factor is ideal gate
                nQubits = int(round(_np.log(op_mx.shape[0]) / _np.log(4))); assert(op_mx.shape[0] == 4**nQubits)
                tensorprod_std_basis = _Basis.cast('std', [(4,) * nQubits])
                U = _bt.change_basis(op_mx, op_basis, tensorprod_std_basis)  # 'std' is incorrect
            elif isinstance(op, _op.StaticStandardOp):
                op_mx = op.to_dense()
                nQubits = int(round(_np.log(op_mx.shape[0]) / _np.log(4))); assert(op_mx.shape[0] == 4**nQubits)
                tensorprod_std_basis = _Basis.cast('std', [(4,) * nQubits])
                U = _bt.change_basis(op_mx, op_basis, tensorprod_std_basis)  # 'std' is incorrect
            else:
                raise ValueError("Could not extract target matrix from %s op!" % str(type(op)))
            return U

        def extract_std_target_vec(v):
            #TODO - make more sophisticated...
            dim = v.state_space.dim
            nQubits = int(round(_np.log(dim) / _np.log(4))); assert(dim == 4**nQubits)
            tensorprod_std_basis = _Basis.cast('std', [(4,) * nQubits])
            v = _bt.change_basis(v.to_dense(), self.basis, tensorprod_std_basis)  # 'std' is incorrect
            return v

        if create_complete_basis_fn is None:
            assert(isinstance(initial_gauge_basis, _CompleteElementaryErrorgenBasis)), \
                ("Must supply a custom `create_complete_basis_fn` if initial gauge basis is not a complete basis!")

            def create_complete_basis_fn(target_sslbls):
                return initial_gauge_basis.create_subbasis(target_sslbls, retain_max_weights=False)

        # get gauge action matrices on the initial space
        gauge_action_matrices = _collections.OrderedDict()
        gauge_action_gauge_spaces = _collections.OrderedDict()
        errorgen_coefficient_labels = _collections.OrderedDict()  # by operation
        for op_label in primitive_op_labels:  # Note: "ga" stands for "gauge action" in variable names below
            #print("DB FOGI: ",op_label)  #REMOVE
            op = self.operation_blks['layers'][op_label]
            U = extract_std_target_mx(op, self.basis)
            all_sslbls = self.state_space.sole_tensor_product_block_labels

            if op_label.sslbls is None:
                target_sslbls = all_sslbls
            elif U.shape[0] == self.state_space.dim and len(op_label.sslbls) < len(all_sslbls):  # don't "trust" sslbls
                target_sslbls = all_sslbls  # e.g., for 2Q explicit models with 2Q gate matched with Gx:0 label
            else:
                target_sslbls = op_label.sslbls

            #cloud_sslbls = all_sslbls  # DEBUG!!! need to get this from cloudnoise elements

            op_gauge_basis = initial_gauge_basis.create_subbasis(target_sslbls)  # gauge space lbls that overlap target
            # Note: can assume gauge action is zero (U acts as identity) on all basis elements not in op_gauge_basis

            initial_row_basis = create_complete_basis_fn(all_sslbls)  #target_sslbls)

            #support_sslbls, gauge_errgen_basis = get_overlapping_labels(gauge_errgen_space_labels, target_sslbls)
            #FOGI DEBUG print("DEBUG -- ", op_label)
            mx, row_basis = _fogit.first_order_gauge_action_matrix(U, target_sslbls, self.state_space,
                                                                   op_gauge_basis, initial_row_basis)
            #print("DB FOGI: action mx: ", mx.shape) #REMOVE
            #FOGI DEBUG print("DEBUG => mx is ", mx.shape)
            # Note: mx is a sparse lil matrix
            # mx cols => op_gauge_basis, mx rows => row_basis, as zero rows have already been removed
            # (DONE: - remove all all-zero rows from mx (and corresponding basis labels) )
            # Note: row_basis is a simple subset of initial_row_basis

            op_with_errorgen = self.operation_blks['cloudnoise'][op_label]
            allowed_rowspace_mx, allowed_row_basis, op_gauge_space = \
                self._format_gauge_action_matrix(mx, op_with_errorgen, reduce_to_model_space, row_basis, op_gauge_basis,
                                                 create_complete_basis_fn)
            #DEBUG
            #print("DB FOGI: action matrix formatting done:")
            #if allowed_rowspace_mx.shape[0] < 10:
            #    print(_np.round(allowed_rowspace_mx.toarray(), 4))
            #else:
            #    print(repr(allowed_rowspace_mx))
            #print(" on ", allowed_row_basis.labels)

            errorgen_coefficient_labels[op_label] = allowed_row_basis.labels
            gauge_action_matrices[op_label] = allowed_rowspace_mx
            gauge_action_gauge_spaces[op_label] = op_gauge_space
            #FOGI DEBUG print("DEBUG => final allowed_rowspace_mx shape =", allowed_rowspace_mx.shape)

        # Similar for SPAM
        for prep_label in primitive_prep_labels:
            prep = self.prep_blks['layers'][prep_label]
            v = extract_std_target_vec(prep)
            target_sslbls = prep_label.sslbls if (prep_label.sslbls is not None and v.shape[0] < self.state_space.dim) \
                else self.state_space.sole_tensor_product_block_labels
            op_gauge_basis = initial_gauge_basis.create_subbasis(target_sslbls)  # gauge space lbls that overlap target
            initial_row_basis = create_complete_basis_fn(target_sslbls)

            mx, row_basis = _fogit.first_order_gauge_action_matrix_for_prep(v, target_sslbls, self.state_space,
                                                                            op_gauge_basis, initial_row_basis)

            allowed_rowspace_mx, allowed_row_basis, op_gauge_space = \
                self._format_gauge_action_matrix(mx, prep, reduce_to_model_space, row_basis, op_gauge_basis,
                                                 create_complete_basis_fn)

            errorgen_coefficient_labels[prep_label] = allowed_row_basis.labels
            gauge_action_matrices[prep_label] = allowed_rowspace_mx
            gauge_action_gauge_spaces[prep_label] = op_gauge_space

        for povm_label in primitive_povm_labels:
            povm = self.povm_blks['layers'][povm_label]
            vecs = [extract_std_target_vec(effect) for effect in povm.values()]
            target_sslbls = povm_label.sslbls if (povm_label.sslbls is not None
                                                  and vecs[0].shape[0] < self.state_space.dim) \
                else self.state_space.sole_tensor_product_block_labels
            op_gauge_basis = initial_gauge_basis.create_subbasis(target_sslbls)  # gauge space lbls that overlap target
            initial_row_basis = create_complete_basis_fn(target_sslbls)

            mx, row_basis = _fogit.first_order_gauge_action_matrix_for_povm(vecs, target_sslbls, self.state_space,
                                                                            op_gauge_basis, initial_row_basis)

            allowed_rowspace_mx, allowed_row_basis, op_gauge_space = \
                self._format_gauge_action_matrix(mx, povm, reduce_to_model_space, row_basis, op_gauge_basis,
                                                 create_complete_basis_fn)

            errorgen_coefficient_labels[povm_label] = allowed_row_basis.labels
            gauge_action_matrices[povm_label] = allowed_rowspace_mx
            gauge_action_gauge_spaces[povm_label] = op_gauge_space

        norm_order = "auto"  # NOTE - should be 1 for normalizing 'S' quantities and 2 for 'H',
        # so 'auto' utilizes intelligence within FOGIStore
        self.fogi_store = _FOGIStore(gauge_action_matrices, gauge_action_gauge_spaces,
                                     errorgen_coefficient_labels,  # gauge_errgen_space_labels,
                                     op_label_abbrevs, reduce_to_model_space, dependent_fogi_action,
                                     norm_order=norm_order)

        if reparameterize:
            self.param_interposer = self._add_reparameterization(
                primitive_op_labels + primitive_prep_labels + primitive_povm_labels,
                self.fogi_store.fogi_directions.toarray(),  # DENSE now (leave sparse in FUTURE?)
                self.fogi_store.errorgen_space_op_elem_labels)


class CloudNoiseLayerRules(_LayerRules):

    def __init__(self, errcomp_type, qubit_labels, implicit_idle_mode, singleq_idle_layer_labels,
                 implied_global_idle_label):
        self.qubit_labels = qubit_labels
        self.errcomp_type = errcomp_type
        self.implied_global_idle_label = implied_global_idle_label
        self.single_qubit_idle_layer_labels = singleq_idle_layer_labels
        self.implicit_idle_mode = implicit_idle_mode  # how to handle implied idles ("blanks") in circuits
        self._add_global_idle_to_all_layers = False
        self._add_padded_idle = False

        if implicit_idle_mode is None or implicit_idle_mode == "none":  # no noise on idles
            pass  # just use defaults above
        elif implicit_idle_mode == "add_global" and self.implied_global_idle_label is not None:
            self._add_global_idle_to_all_layers = True    # add global idle to all layers
        elif implicit_idle_mode == "pad_1Q" and self.single_qubit_idle_layer_labels is not None:
            self._add_padded_idle = True
        else:
            raise ValueError("Invalid `implicit_idle_mode`: '%s'" % str(implicit_idle_mode))

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        assert(all([len(k) == 1 for k in self.single_qubit_idle_layer_labels.keys()])), \
            "All keys of single_qubit_idle_layer_labels should be 1-tuples of a *single* sslbl!"
        state.update({'error_composition_mode': self.errcomp_type,
                      'implied_global_idle_label': (str(self.implied_global_idle_label)
                                                    if (self.implied_global_idle_label is not None) else None),
                      'single_qubit_idle_layer_labels': ({str(sslbls[0]): str(idle_lbl) for sslbls, idle_lbl
                                                          in self.single_qubit_idle_layer_labels.items()}
                                                         if self.single_qubit_idle_layer_labels is not None else None),
                      'implicit_idle_mode': self.implicit_idle_mode,
                      'qubit_labels': list(self.qubit_labels),
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        from pygsti.circuits.circuitparser import parse_label as _parse_label

        def _to_int(x):  # (same as in slowcircuitparser.py)
            return int(x) if x.isdigit() else x

        gi_label = _parse_label(state['implied_global_idle_label']) \
            if (state['implied_global_idle_label'] is not None) else None

        if state.get('single_qubit_idle_layer_labels', None) is not None:
            singleQ_idle_lbls = {(_to_int(k),): _parse_label(v)
                                 for k, v in state['single_qubit_idle_layer_labels'].items()}
        else:
            singleQ_idle_lbls = None

        qubit_labels = tuple(state['qubit_labels']) if ('qubit_labels' in state) else None

        return cls(state['error_composition_mode'], qubit_labels, state['implicit_idle_mode'],
                   singleQ_idle_lbls, gi_label)

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
        add_global_idle = self._add_global_idle_to_all_layers
        add_padded_idle = self._add_padded_idle

        #print("DB: CloudNoiseLayerLizard building gate %s for %s w/comp-type %s" %
        #      (('matrix' if dense else 'map'), str(oplabel), self.errcomp_type) )

        components = layerlbl.components
        if len(components) == 0:
            if add_global_idle:
                if self.errcomp_type == "gates":
                    return model.operation_blks['cloudnoise'][self.implied_global_idle_label]  # idle!
                elif self.errcomp_type == "errorgens":
                    return ExpErrorgen(model.operation_blks['cloudnoise'][self.implied_global_idle_label])
                else:
                    raise ValueError("Invalid errcomp_type in CloudNoiseLayerRules: %s" % str(self.errcomp_type))
            elif add_padded_idle:
                idle_factors = [model.operation_blks['cloudnoise'][self.single_qubit_idle_layer_labels[(sslbl,)]]
                                for sslbl in self.qubit_labels]
                if self.errcomp_type == "gates":
                    ret = Composed(idle_factors, evotype=model.evotype, state_space=model.state_space)
                elif self.errcomp_type == "errorgens":
                    ret = ExpErrorgen(Sum(idle_factors, state_space=model.state_space, evotype=model.evotype))
                else:
                    raise ValueError("Invalid errcomp_type in CloudNoiseLayerRules: %s" % str(self.errcomp_type))
                model._init_virtual_obj(ret)  # so ret's gpindices get set
                return ret
            else:
                #Perfect no-noise idle
                return Composed([], evotype=model.evotype, state_space=model.state_space)  # no need to init_virtual

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
            if add_global_idle:
                ops_to_compose.append(model.operation_blks['cloudnoise'][self.implied_global_idle_label])
            # Note: add_padded_idle handled within _layer_component_cloudnoises
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
            # Note: add_padded_idle handled within _layer_component_cloudnoises
            errorGens = [model.operation_blks['cloudnoise'][self.implied_global_idle_label]] if add_global_idle else []
            errorGens.extend(self._layer_component_cloudnoises(model, components, caches['op-cloudnoise']))
            if len(errorGens) > 0:
                if len(errorGens) > 1:
                    error = ExpErrorgen(Sum(errorGens, state_space=model.state_space, evotype=model.evotype))
                else:
                    error = ExpErrorgen(errorGens[0])
                ops_to_compose.append(error)
        else:
            raise ValueError("Invalid errcomp_type in CloudNoiseLayerRules: %s" % str(self.errcomp_type))

        ret = Composed(ops_to_compose, evotype=model.evotype, state_space=model.state_space) \
            if len(ops_to_compose) > 1 else ops_to_compose[0]
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
        if self._add_padded_idle:
            component_sslbls = [c.sslbls for c in complbl_list]
            if None not in component_sslbls:  # sslbls == None => label covers *all* labels, no padding needed
                present_sslbl_components = set(_itertools.chain(*[sslbls for sslbls in component_sslbls]))
                absent_sslbls = [sslbl for sslbl in self.qubit_labels if (sslbl not in present_sslbl_components)]
                factors = {sslbl: model.operation_blks['cloudnoise'][self.single_qubit_idle_layer_labels[(sslbl,)]]
                           for sslbl in absent_sslbls}  # key = *lowest* (and only) sslbl
            else:
                factors = {}

            for complbl in complbl_list:
                complbl_lowest_sslbl = sorted(complbl.sslbls)[0] if (complbl.sslbls is not None) else 0
                if complbl in cache:
                    factors[complbl_lowest_sslbl] = cache[complbl]
                elif complbl in model.operation_blks['cloudnoise']:
                    factors[complbl_lowest_sslbl] = model.operation_blks['cloudnoise'][complbl]
                else:
                    try:
                        factors[complbl_lowest_sslbl] = _opfactory.op_from_factories(
                            model.factories['cloudnoise'], complbl)
                    except KeyError: pass  # OK if cloudnoise doesn't exist (means no noise)

            ret = [factors[k] for k in sorted(factors.keys())]

        else:
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
