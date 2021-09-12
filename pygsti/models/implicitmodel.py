"""
Defines the ImplicitOpModel class and supporting functionality.
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
import uuid as _uuid

from pygsti.models import model as _mdl
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers import povms as _povm
from pygsti.modelmembers.modelmembergraph import ModelMemberGraph as _MMGraph
from pygsti.baseobjs.label import Label as _Label


class ImplicitOpModel(_mdl.OpModel):
    """
    A model that stores the building blocks for layer operations and build circuit-layer operations on-demand.

    An ImplicitOpModel represents a flexible QIP model whereby only the
    building blocks for layer operations are stored, and custom layer-lizard
    logic is used to construct layer operations from these blocks on an
    on-demand basis.

    Parameters
    ----------
    state_space : StateSpace
        The state space for this model.

    layer_rules : LayerRules
        The "layer rules" used for constructing operators for circuit
        layers.  This functionality is essential to using this model to
        simulate ciruits, and is typically supplied by derived classes.

    basis : Basis
        The basis used for the state space by dense operator representations.

    simulator : ForwardSimulator or {"auto", "matrix", "map"}
        The circuit simulator used to compute any
        requested probabilities, e.g. from :method:`probs` or

    evotype : {"densitymx", "statevec", "stabilizer", "svterm", "cterm"}
        The evolution type of this model, describing how states are
        represented, allowing compatibility checks with (super)operator
        objects.
    """
    def __init__(self,
                 state_space,
                 layer_rules,
                 basis="pp",
                 simulator="auto",
                 evotype="densitymx"):
        self.prep_blks = _collections.OrderedDict()
        self.povm_blks = _collections.OrderedDict()
        self.operation_blks = _collections.OrderedDict()
        self.instrument_blks = _collections.OrderedDict()
        self.factories = _collections.OrderedDict()

        super(ImplicitOpModel, self).__init__(state_space, basis, evotype, layer_rules, simulator)

    @property
    def _primitive_prep_label_dict(self):
        return self.prep_blks['layers']

    @property
    def _primitive_povm_label_dict(self):
        return self.povm_blks['layers']

    @property
    def _primitive_op_label_dict(self):
        # don't include 'implied' ops as primitive ops -- FUTURE - maybe should include empty layer ([])?
        return _collections.OrderedDict([(k, None) for k in self.operation_blks['layers']
                                         if not (k.name.startswith('(') and k.name.endswith(')'))])

    @property
    def _primitive_instrument_label_dict(self):
        return self.instrument_blks['layers']

    #Functions required for base class functionality

    def _iter_parameterized_objs(self):
        for dictlbl, objdict in _itertools.chain(self.prep_blks.items(),
                                                 self.povm_blks.items(),
                                                 self.operation_blks.items(),
                                                 self.instrument_blks.items(),
                                                 self.factories.items()):
            for lbl, obj in objdict.items():
                yield (_Label(dictlbl + ":" + lbl.name, lbl.sslbls), obj)

    def _init_copy(self, copy_into, memo):
        """
        Copies any "tricky" member of this model into `copy_into`, before
        deep copying everything else within a .copy() operation.
        """
        # Copy special base class members first
        super(ImplicitOpModel, self)._init_copy(copy_into, memo)

        # Copy our "tricky" members
        copy_into.prep_blks = _collections.OrderedDict([(lbl, prepdict.copy(copy_into, memo))
                                                       for lbl, prepdict in self.prep_blks.items()])
        copy_into.povm_blks = _collections.OrderedDict([(lbl, povmdict.copy(copy_into, memo))
                                                       for lbl, povmdict in self.povm_blks.items()])
        copy_into.operation_blks = _collections.OrderedDict([(lbl, opdict.copy(copy_into, memo))
                                                            for lbl, opdict in self.operation_blks.items()])
        copy_into.instrument_blks = _collections.OrderedDict([(lbl, idict.copy(copy_into, memo))
                                                             for lbl, idict in self.instrument_blks.items()])
        copy_into.factories = _collections.OrderedDict([(lbl, fdict.copy(copy_into, memo))
                                                       for lbl, fdict in self.factories.items()])

        copy_into._state_space = self.state_space.copy()  # needed by simplifier helper

    def __setstate__(self, state_dict):
        super().__setstate__(state_dict)
        if 'uuid' not in state_dict:
            self.uuid = _uuid.uuid4()  # create a new uuid

        if 'factories' not in state_dict:
            self.factories = _collections.OrderedDict()  # backward compatibility (temporary)

        #Additionally, must re-connect this model as the parent
        # of relevant OrderedDict-derived classes, which *don't*
        # preserve this information upon pickling so as to avoid
        # circular pickling...
        for prepdict in self.prep_blks.values():
            prepdict.parent = self
            for o in prepdict.values(): o.relink_parent(self)
        for povmdict in self.povm_blks.values():
            povmdict.parent = self
            for o in povmdict.values(): o.relink_parent(self)
        for opdict in self.operation_blks.values():
            opdict.parent = self
            for o in opdict.values(): o.relink_parent(self)
        for idict in self.instrument_blks.values():
            idict.parent = self
            for o in idict.values(): o.relink_parent(self)
        for fdict in self.factories.values():
            fdict.parent = self
            for o in fdict.values(): o.relink_parent(self)

    def compute_clifford_symplectic_reps(self, oplabel_filter=None):
        """
        Constructs a dictionary of the symplectic representations for all the Clifford gates in this model.

        Non-:class:`StaticCliffordOp` gates will be ignored and their entries omitted
        from the returned dictionary.

        Parameters
        ----------
        oplabel_filter : iterable, optional
            A list, tuple, or set of operation labels whose symplectic
            representations should be returned (if they exist).

        Returns
        -------
        dict
            keys are operation labels and/or just the root names of gates
            (without any state space indices/labels).  Values are
            `(symplectic_matrix, phase_vector)` tuples.
        """
        gfilter = set(oplabel_filter) if oplabel_filter is not None \
            else None

        srep_dict = {}

        for gl in self.primitive_op_labels:
            gate = self.operation_blks['layers'][gl]
            if (gfilter is not None) and (gl not in gfilter): continue

            if isinstance(gate, _op.EmbeddedOp):
                assert(isinstance(gate.embedded_op, _op.StaticCliffordOp)), \
                    "EmbeddedClifforGate contains a non-StaticCliffordOp!"
                lbl = gl.name  # strip state space labels off since this is a
                # symplectic rep for the *embedded* gate
                srep = (gate.embedded_op.smatrix, gate.embedded_op.svector)
            elif isinstance(gate, _op.StaticCliffordOp):
                lbl = gl.name
                srep = (gate.smatrix, gate.svector)
            else:
                lbl = srep = None

            if srep:
                if lbl in srep_dict:
                    assert(srep == srep_dict[lbl]), \
                        "Inconsistent symplectic reps for %s label!" % lbl
                else:
                    srep_dict[lbl] = srep

        return srep_dict

    def __str__(self):
        s = ""
        for dictlbl, d in self.prep_blks.items():
            for lbl, vec in d.items():
                s += "%s:%s = " % (str(dictlbl), str(lbl)) + str(vec) + "\n"
        s += "\n"
        for dictlbl, d in self.povm_blks.items():
            for lbl, povm in d.items():
                s += "%s:%s = " % (str(dictlbl), str(lbl)) + str(povm) + "\n"
        s += "\n"
        for dictlbl, d in self.operation_blks.items():
            for lbl, gate in d.items():
                s += "%s:%s = \n" % (str(dictlbl), str(lbl)) + str(gate) + "\n\n"
        for dictlbl, d in self.instrument_blks.items():
            for lbl, inst in d.items():
                s += "%s:%s = " % (str(dictlbl), str(lbl)) + str(inst) + "\n"
        s += "\n"
        for dictlbl, d in self.factories.items():
            for lbl, factory in d.items():
                s += "%s:%s = " % (str(dictlbl), str(lbl)) + str(factory) + "\n"
        s += "\n"

        return s

    def _default_primitive_povm_layer_lbl(self, sslbls):
        """
        Gets the default POVM label.

        This is often used when a circuit  is specified without an ending POVM layer.
        Returns `None` if there is no default and one *must* be specified.

        Parameters
        ----------
        sslbls : tuple or None
            The state space labels being measured, and for which a default POVM is desired.

        Returns
        -------
        Label or None
        """
        if len(self.primitive_povm_labels) == 1:
            povm_name = next(iter(self.primitive_povm_labels)).name
            if (self.state_space.num_tensor_product_blocks == 1
                and (self.state_space.tensor_product_block_labels(0) == sslbls
                     or sslbls == ('*',))):
                return _Label(povm_name)  # because sslbls == all of model's sslbls
            else:
                return _Label(povm_name, sslbls)
        else:
            return None

    def _effect_labels_for_povm(self, povm_lbl):
        """
        Gets the effect labels corresponding to the possible outcomes of POVM label `povm_lbl`.

        Parameters
        ----------
        povm_lbl : Label
            POVM label.

        Returns
        -------
        list
            A list of strings which label the POVM outcomes.
        """
        for povmdict in self.povm_blks.values():
            if povm_lbl in povmdict:
                return tuple(povmdict[povm_lbl].keys())
            if isinstance(povm_lbl, _Label) and povm_lbl.name in povmdict:
                return tuple(_povm.MarginalizedPOVM(povmdict[povm_lbl.name],
                                                    self.state_space, povm_lbl.sslbls).keys())

        raise KeyError("No POVM labeled %s!" % str(povm_lbl))

    def _member_labels_for_instrument(self, inst_lbl):
        """
        Gets the member labels corresponding to the possible outcomes of the instrument labeled by `inst_lbl`.

        Parameters
        ----------
        inst_lbl : Label
            Instrument label.

        Returns
        -------
        list
            A list of strings which label the instrument members.
        """
        for idict in self.instrument_blks.values():
            if inst_lbl in idict:
                return tuple(idict[inst_lbl].keys())
        raise KeyError("No instrument labeled %s!" % inst_lbl)

    def _reinit_opcaches(self):
        self._opcaches.clear()

        # Add expanded instrument and POVM operations to cache so these are accessible to circuit calcs
        simplified_effect_blks = _collections.OrderedDict()
        for povm_dict_lbl, povmdict in self.povm_blks.items():
            simplified_effect_blks['povm-' + povm_dict_lbl] = _collections.OrderedDict(
                [(k, e) for povm_lbl, povm in povmdict.items()
                 for k, e in povm.simplify_effects(povm_lbl).items()])

        simplified_op_blks = _collections.OrderedDict()
        for op_dict_lbl in self.operation_blks:
            simplified_op_blks['op-' + op_dict_lbl] = {}  # create *empty* caches corresponding to op categories
        for inst_dict_lbl, instdict in self.instrument_blks.items():
            if 'op-' + inst_dict_lbl not in simplified_op_blks:  # only create when needed
                simplified_op_blks['op-' + inst_dict_lbl] = _collections.OrderedDict()
            for inst_lbl, inst in instdict.items():
                for k, g in inst.simplify_operations(inst_lbl).items():
                    simplified_op_blks['op-' + inst_dict_lbl][k] = g

        #FUTURE: allow cache "cateogories"?  Now we just flatten the work we did above:
        self._opcaches.update(simplified_effect_blks)
        self._opcaches.update(simplified_op_blks)
        self._opcaches['complete-layers'] = {}  # used to hold final layers (of any type) if needed

    def create_modelmember_graph(self):
        self._clean_paramvec()  # Rebuild params to ensure accurate comparisons with MMGraphs

        root_dicts = {
            'prep_blks': self.prep_blks,
            'povm_blks': self.povm_blks,
            'operation_blks': self.operation_blks,
            'instrument_blks': self.instrument_blks,
            'factories': self.factories,
        }
        mm_dicts = {(root_str + "|" + k): mm_dict
                    for root_str, root_dict in root_dicts.items()
                    for k, mm_dict in root_dict.items()}
        return _MMGraph(mm_dicts)

    def _to_memoized_dict(self, memo):
        state = {'module': self.__class__.__module__,
                 'class': self.__class__.__name__,
                 'state_space': self.state_space._to_memoized_dict({}),
                 'basis': self.basis._to_memoized_dict({}),
                 'evotype': str(self.evotype),  # TODO or serialize?
                 'layer_rules': self._layer_rules._to_memoized_dict({}),
                 'simulator': self.sim._to_memoized_dict({})
                 }

        mmgraph = self.create_modelmember_graph()
        state['modelmembers'] = mmgraph.create_serialization_dict()
        return state

    @classmethod
    def _from_memoized_dict(cls, state, memo):
        from pygsti.io.metadir import _from_memoized_dict
        state_space = _from_memoized_dict(state['state_space'])
        layer_rules = _from_memoized_dict(state['layer_rules'])
        basis = _from_memoized_dict(state['basis'])
        modelmembers = _MMGraph.load_modelmembers_from_serialization_dict(state['modelmembers'])
        simulator = _from_memoized_dict(state['simulator'])

        mdl = cls(state_space, layer_rules, basis, simulator, state['evotype'])

        root_dicts = {
            'prep_blks': mdl.prep_blks,
            'povm_blks': mdl.povm_blks,
            'operation_blks': mdl.operation_blks,
            'instrument_blks': mdl.instrument_blks,
            'factories': mdl.factories,
        }
        for mm_key, mm_dict in modelmembers.items():
            root_key, sub_key = mm_key.split('|')
            root_dicts[root_key][sub_key].update(mm_dict)  # Note: sub_keys should already be created
        return mdl


def _init_spam_layers(model, prep_layers, povm_layers):
    """ Helper function for initializing the .prep_blks and .povm_blks elements of an implicit model"""
    # SPAM (same as for cloud noise model)
    if prep_layers is None:
        pass  # no prep layers
    elif isinstance(prep_layers, dict):
        for rhoname, layerop in prep_layers.items():
            model.prep_blks['layers'][_Label(rhoname)] = layerop
    elif isinstance(prep_layers, _op.LinearOperator):  # just a single layer op
        model.prep_blks['layers'][_Label('rho0')] = prep_layers
    else:  # assume prep_layers is an iterable of layers, e.g. isinstance(prep_layers, (list,tuple)):
        for i, layerop in enumerate(prep_layers):
            model.prep_blks['layers'][_Label("rho%d" % i)] = layerop

    if povm_layers is None:
        pass  # no povms
    elif isinstance(povm_layers, _povm.POVM):  # just a single povm - must precede 'dict' test!
        model.povm_blks['layers'][_Label('Mdefault')] = povm_layers
    elif isinstance(povm_layers, dict):
        for povmname, layerop in povm_layers.items():
            model.povm_blks['layers'][_Label(povmname)] = layerop
    else:  # assume povm_layers is an iterable of layers, e.g. isinstance(povm_layers, (list,tuple)):
        for i, layerop in enumerate(povm_layers):
            model.povm_blks['layers'][_Label("M%d" % i)] = layerop
