"""
Objects for specifying the noise to be added to models when they are created.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import warnings as _warnings
import collections as _collections
import itertools as _itertools
import numpy as _np

from pygsti.models.stencillabel import StencilLabel as _StencilLabel
from pygsti.tools import listtools as _lt
from pygsti.tools import optools as _ot
from pygsti.modelmembers import operations as _op
from pygsti.modelmembers.operations.lindbladcoefficients import LindbladCoefficientBlock as _LindbladCoefficientBlock
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.basis import BuiltinBasis as _BuiltinBasis
from pygsti.circuits.circuitparser import CircuitParser as _CircuitParser


class ModelNoise(object):
    """
    A base class for objects specifying noise (errors) that should be added to a quantum processor model.

    :class:`ModelNoise` objects serve as a lightweight and flexible way to specify the noise that should
    be included in a model prior to its creation.  Typically these objects, which can contain complex
    prescriptions for model noise, are constructed and then passed as input to a model construction routine.
    """
    pass


class OpModelNoise(ModelNoise):
    """
    Noise on a model containing individual gate/SPAM operations, e.g. an :class:`OpModel` object.

    This class is a base class that should not be instantiated directly.
    """

    @classmethod
    def cast(cls, obj):
        """
        Convert an object to an :class:`OpModelNoise` object if it isn't already.

        If a dictionary is given, it is assumed to contain per-operation error
        specifications.  If a list/tuple is given, it is assumed to contain multiple
        sub-specifications that should be composed together (by constructing a
        :class:`ComposedOpModelNoise` object).

        Parameters
        ----------
        obj : object
            The object to convert.

        Returns
        -------
        OpModelNoise
        """
        if obj is None:
            return OpModelPerOpNoise({})
        elif isinstance(obj, OpModelNoise):
            return obj
        elif isinstance(obj, (list, tuple)):  # assume obj == list of OpModelNoise objects
            return ComposedOpModelNoise([cls.cast(el) for el in obj])
        elif isinstance(obj, dict):  # assume obj = dict of per-op noise
            return OpModelPerOpNoise(obj)
        else:
            raise ValueError("Cannot convert type %s to an OpModelNoise object!" % str(type(obj)))

    def __init__(self):
        self._opkey_access_counters = {}
        super(OpModelNoise, self).__init__()

    def keys(self):
        """
        The operation labels for which this object specifies noise.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def __contains__(self, key):
        raise NotImplementedError("Derived classes should implement this!")

    def create_errorgen_stencil(self, opkey, evotype, state_space, num_target_labels=None):
        """
        Create an "error generator stencil" for the noise corresponding to an operation label.

        A stencil is one or more operator objects that have yet to be embedded on their
        final qudits and then composed.  The embedding and composing step is done later
        so that, if desired, the same errors can be used on multiple sets of target qudits
        (often this is done when a "independent" argument to a model-creation function is
        `False`).  An "error generator stencil" is a stencil whose operators are error
        generators, rather than error maps.

        Parameters
        ----------
        opkey : Label or StencilLabel
            The operation label to create the stencil for.

        evotype : str or Evotype
            The evolution type of to use when creating the stencil operators.

        state_space : StateSpace
            The state space to use when creating the stencil operators.  This can be a
            *local* state space, disparate from the actual processor's state space, or
            the entire state space that the stencil operations will ultimately be embedded
            into.  In the former case, `num_target_labels` should be left as `None`, indicating
            that `state_space` is the exact space being acted upon.  In the latter case,
            `num_target_labels` should specify the number of target labels within `state_space`
            that this stencil will be given when it is applied (embedded) into `state_space`.
            This requires that all the labels in `state_space` correspond to the same type and
            dimension space (e.g. all are qubit spaces).

        num_target_labels : int or None
            The number of labels within `state_space` that the `op_key` operation acts upon (this
            assumes that all the labels in `state_space` are similar, e.g., all qubits).  If `None`
            then it acts on the entire space given by `state_space`.

        Returns
        -------
        stencil : OrderedDict
            A dictionary with keys that are :class:`Label` or :class:`StencilLabel` objects and
            values that are :class:`LinearOperator` objects.  The stencil is applied by embedding
            each operator according to its key and then composing the results.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def apply_errorgen_stencil(self, stencil, evotype, state_space, target_labels=None, qudit_graph=None, copy=False):
        """
        Apply an error-generator stencil created by this object to a specific set of target labels.

        A stencil is applied by embedding each operator in the stencil according to its target
        state space labels (which may include stencil-label expansions) and then composing the
        results.

        Parameters
        ----------
        stencil : OrderedDict
            The stencil to apply, usually created by :method:`create_errorgen_stencil`.

        evotype : str or Evotype
            The evolution type of to use when creating the embedded and composed operators,
            which should match that of the stencil operators (the evotype used to create the stencil).

        state_space : StateSpace
            The state space to use when creating the composed and embedded operators.  This should
            be the total state space of the model that these noise operations will be inserted into.

        target_labels : tuple or None, optional
            The target labels that determine where on the qudit graph this stencil will be placed.  When a
            tuple, it should have length equal to the `num_target_labels` argument passed to
            :method:`create_errorgen_stencil`.  `None` indicates that the entire space is the "target"
            space of the stencil (e.g. a global idle, preparation, or measurement).

        qudit_graph : QubitGraph, optional
            The relevant qudit graph, usually from a processor specification, that contains adjacency and
            direction information used to resolve stencil state space labels into absolute labels within
            `state_space`.  If `None`, then an error will be raised if any direction or connectivity information
            is needed to resolve the state space labels.

        copy : bool, optional
            Whether the stencil operations should be copied before embedding and composing them to apply
            the stencil.  `True` can be used to make different applications of the stencil independent
            operations.

        Returns
        -------
        LinearOperator
        """
        raise NotImplementedError("Derived classes should implement this!")

    def create_errorgen(self, opkey, evotype, state_space, target_labels=None, qudit_graph=None):
        """
        Create an error generator object to implement the noise on a given model operation.

        Parameters
        ----------
        opkey : Label or StencilLabel
            The operation label to create the error generator for.

        evotype : str or Evotype
            The evolution type to use when creating the error generator.

        state_space : StateSpace
            The state space to use when creating the error generator.

        target_labels : tuple or None, optional
            The target state space labels for this operation.  Sometimes this
            is also contained in `opkey`, but not always, so it must be supplied
            as a separate argument.

        qudit_graph : QubitGraph, optional
            The relevant qudit graph, usually from a processor specification, that contains adjacency and
            direction information used to create more complex types of errors.  If `None`, then an error
            will be raised if graph information is needed.

        Returns
        -------
        LinearOperator
        """
        stencil = self.create_errorgen_stencil(opkey, evotype, state_space,
                                               len(target_labels) if (target_labels is not None) else None)
        return self.apply_errorgen_stencil(stencil, evotype, state_space, target_labels, qudit_graph)

    def create_errormap_stencil(self, opkey, evotype, state_space, num_target_labels=None):
        """
        Create an "error map stencil" for the noise corresponding to an operation label.

        A stencil is one or more operator objects that have yet to be embedded on their
        final qudits and then composed.  The embedding and composing step is done later
        so that, if desired, the same errors can be used on multiple sets of target qudits
        (often this is done when a "independent" argument to a model-creation function is
        `False`).  An "error map stencil" is a stencil whose operators are error maps
         rather than error generators.

        Parameters
        ----------
        opkey : Label or StencilLabel
            The operation label to create the stencil for.

        evotype : str or Evotype
            The evolution type of to use when creating the stencil operators.

        state_space : StateSpace
            The state space to use when creating the stencil operators.  This can be a
            *local* state space, disparate from the actual processor's state space, or
            the entire state space that the stencil operations will ultimately be embedded
            into.  In the former case, `num_target_labels` should be left as `None`, indicating
            that `state_space` is the exact space being acted upon.  In the latter case,
            `num_target_labels` should specify the number of target labels within `state_space`
            that this stencil will be given when it is applied (embedded) into `state_space`.
            This requires that all the labels in `state_space` correspond to the same type and
            dimension space (e.g. all are qubit spaces).

        num_target_labels : int or None
            The number of labels within `state_space` that the `op_key` operation acts upon (this
            assumes that all the labels in `state_space` are similar, e.g., all qubits).  If `None`
            then it acts on the entire space given by `state_space`.

        Returns
        -------
        stencil : OrderedDict
            A dictionary with keys that are :class:`Label` or :class:`StencilLabel` objects and
            values that are :class:`LinearOperator` objects.  The stencil is applied by embedding
            each operator according to its key and then composing the results.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def apply_errormap_stencil(self, stencil, evotype, state_space, target_labels=None, qudit_graph=None, copy=False):
        """
        Apply an error-map stencil created by this object to a specific set of target labels.

        A stencil is applied by embedding each operator in the stencil according to its target
        state space labels (which may include stencil-label expansions) and then composing the
        results.

        Parameters
        ----------
        stencil : OrderedDict
            The stencil to apply, usually created by :method:`create_errormap_stencil`.

        evotype : str or Evotype
            The evolution type of to use when creating the embedded and composed operators,
            which should match that of the stencil operators (the evotype used to create the stencil).

        state_space : StateSpace
            The state space to use when creating the composed and embedded operators.  This should
            be the total state space of the model that these noise operations will be inserted into.

        target_labels : tuple or None, optional
            The target labels that determine where on the qudit graph this stencil will be placed.  When a
            tuple, it should have length equal to the `num_target_labels` argument passed to
            :method:`create_errormap_stencil`.  `None` indicates that the entire space is the "target"
            space of the stencil (e.g. a global idle, preparation, or measurement).

        qudit_graph : QubitGraph, optional
            The relevant qudit graph, usually from a processor specification, that contains adjacency and
            direction information used to resolve stencil state space labels into absolute labels within
            `state_space`.  If `None`, then an error will be raised if any direction or connectivity information
            is needed to resolve the state space labels.

        copy : bool, optional
            Whether the stencil operations should be copied before embedding and composing them to apply
            the stencil.  `True` can be used to make different applications of the stencil independent
            operations.

        Returns
        -------
        LinearOperator
        """
        raise NotImplementedError("Derived classes should implement this!")

    def create_errormap(self, opkey, evotype, state_space, target_labels=None, qudit_graph=None):
        """
        Create an error map object to implement the noise on a given model operation.

        Parameters
        ----------
        opkey : Label or StencilLabel
            The operation label to create the error map for.

        evotype : str or Evotype
            The evolution type to use when creating the error map.

        state_space : StateSpace
            The state space to use when creating the error map.

        target_labels : tuple or None, optional
            The target state space labels for this operation.  Sometimes this
            is also contained in `opkey`, but not always, so it must be supplied
            as a separate argument.

        qudit_graph : QubitGraph, optional
            The relevant qudit graph, usually from a processor specification, that contains adjacency and
            direction information used to create more complex types of errors.  If `None`, then an error
            will be raised if graph information is needed.

        Returns
        -------
        LinearOperator
        """
        stencil = self.create_errormap_stencil(opkey, evotype, state_space,
                                               len(target_labels) if (target_labels is not None) else None)
        return self.apply_errormap_stencil(stencil, evotype, state_space, target_labels, qudit_graph)

    def reset_access_counters(self):
        """
        Resets the internal key-access counters to zero.

        These counters tally the number of times each operation key is accessed, and
        are used to identify model noise specification that are supplied by the user
        but never used.  See :method:`warn_about_zero_counters`.

        Returns
        -------
        None
        """
        self._opkey_access_counters = {k: 0 for k in self.keys()}

    def _increment_touch_count(self, opkey):
        if opkey in self._opkey_access_counters:
            self._opkey_access_counters[opkey] += 1

    def warn_about_zero_counters(self):
        """
        Issue a warning if any of the internal key-access counters are zero

        Used to catch noise specifications that are never utilized and that the
        caller/user should be warned about.

        Returns
        -------
        None
        """
        untouched_keys = [k for k, touch_cnt in self._opkey_access_counters.items() if touch_cnt == 0]
        if len(untouched_keys) > 0:
            _warnings.warn(("The following model-noise entries were unused: %s.  You may want to double check"
                            " that you've entered a valid noise specification.") % ", ".join(map(str, untouched_keys)))

    def compute_stencil_absolute_sslbls(self, stencil, state_space, target_labels=None, qudit_graph=None):
        """
        Computes the set of state space labels that would be utilized when applying a stencil.

        This function computes which state space labels are non-trivially acted upon by the
        operation that results from applying `stencil` to `target_labels`.

        Parameters
        ----------
        stencil : OrderedDict
            The stencil.  A dictionary with keys that are target state space labels (perhaps stencil labels)
            and values that are operations.  This function only cares about the keys of this dictionary.

        state_space : StateSpace
            The state space that would be given if/when applying `stencil`.  This should
            be the total state space of the model that the applied stencil would be inserted into.

        target_labels : tuple or None, optional
            The target labels that determine where on the qudit graph `stencil` will be placed.  `None`
            indicates that the entire space is the "target" space of the stencil.

        qudit_graph : QubitGraph, optional
            The relevant qudit graph that contains adjacency and direction information used to resolve stencil
            state space labels into absolute labels within `state_space`.  If `None`, then an error will be raised
            if any direction or connectivity information is needed to resolve the state space labels.

        Returns
        -------
        set
            A set (i.e. without any duplicates) of the state space labels that would be acted upon.
        """
        raise NotImplementedError("Derived classes should implement this!")


class OpModelPerOpNoise(OpModelNoise):
    """
    Model noise that is stored on a per-operation basis.

    Parameters
    ----------
    per_op_noise : dict
        A dictionary mapping operation labels (which will become the keys of this
        :class:`OpModelNoise` object) to either :class:`OpNoise` objects or to a
        nexted dictionary mapping absolute or stencil state space labels to
        :class:`OpNoise` objects.  In the former case, the :class:`OpNoise` object
        is assumed to apply to all the target labels of the operation.
    """

    def __init__(self, per_op_noise):
        # a dictionary mapping operation keys -> OpNoise objects
        #                                  OR -> {dict mapping sslbls -> OpNoise objects}
        self.per_op_noise = per_op_noise.copy()

        # Update any label-string format keys to actual Labels (convenience for users)
        cparser = _CircuitParser()
        cparser.lookup = None  # lookup - functionality removed as it wasn't used
        for k, v in per_op_noise.items():
            if isinstance(k, str) and ":" in k:  # then parse this to get a label, allowing, e.g. "Gx:0"
                lbls, _, _, _ = cparser.parse(k)
                assert (len(lbls) == 1), "Only single primitive-gate labels allowed as keys! (not %s)" % str(k)
                del self.per_op_noise[k]
                self.per_op_noise[lbls[0]] = v

        super(OpModelPerOpNoise, self).__init__()

    def keys(self):
        """
        The operation labels for which this object specifies noise.
        """
        return self.per_op_noise.keys()

    def __contains__(self, key):
        return key in self.per_op_noise

    def create_errorgen_stencil(self, opkey, evotype, state_space, num_target_labels=None):
        """
        See :method:`OpModelNoise.create_errorgen_stencil`.
        """
        if opkey not in self.per_op_noise: return {}  # an empty stencil
        opnoise = self.per_op_noise[opkey]
        errgens_to_embed_then_compose = _collections.OrderedDict()  # the "stencil" we return
        if isinstance(opnoise, dict):  # sslbls->OpNoise dict for potentially non-local noise
            opnoise_dict = opnoise
            for sslbls, opnoise in opnoise_dict.items():
                if sslbls is None:    # special behavior: `None` key => target labels
                    sslbls = tuple(['@{}'.format(i) for i in range(num_target_labels)])
                local_state_space = _StencilLabel.cast(sslbls).create_local_state_space(state_space)
                local_errorgen = opnoise.create_errorgen(evotype, local_state_space)

                if sslbls not in errgens_to_embed_then_compose:  # allow multiple sslbls => same *effetive* sslbls
                    errgens_to_embed_then_compose[sslbls] = local_errorgen
                elif isinstance(errgens_to_embed_then_compose[sslbls], _op.ComposedErrorgen):
                    errgens_to_embed_then_compose[sslbls].append(local_errorgen)
                else:
                    errgens_to_embed_then_compose[sslbls] = _op.ComposedErrorgen(
                        [errgens_to_embed_then_compose[sslbls], local_errorgen])

        else:  # assume opnoise is an OpNoise object
            local_errorgen = opnoise.create_errorgen(evotype, state_space)
            all_target_labels = None if (num_target_labels is None) else \
                tuple(['@{}'.format(i) for i in range(num_target_labels)])
            errgens_to_embed_then_compose[all_target_labels] = local_errorgen
        self._increment_touch_count(opkey)
        return errgens_to_embed_then_compose

    def apply_errorgen_stencil(self, stencil, evotype, state_space, target_labels=None, qudit_graph=None, copy=False):
        """
        See :method:`OpModelNoise.apply_errorgen_stencil`.
        """
        embedded_errgens = []
        for stencil_sslbls, local_errorgen in stencil.items():
            sslbls_list = self._map_stencil_sslbls(stencil_sslbls, qudit_graph, state_space, target_labels)
            if None in sslbls_list and stencil_sslbls is not None:
                # `None` in list signals a non-present direction => skip these terms
                sslbls_list = list(filter(lambda x: x is not None, sslbls_list))
            for sslbls in sslbls_list:
                op_to_embed = local_errorgen if (sslbls is None or state_space.is_entire_space(sslbls)) \
                    else _op.EmbeddedErrorgen(state_space, sslbls, local_errorgen)
                embedded_errgens.append(op_to_embed.copy() if copy else op_to_embed)

        if len(embedded_errgens) == 0:
            return None  # ==> no errorgen (could return an empty ComposedOp instead?)
        else:
            return _op.ComposedErrorgen(embedded_errgens, evotype, state_space) \
                if len(embedded_errgens) > 1 else embedded_errgens[0]

    def create_errormap_stencil(self, opkey, evotype, state_space, num_target_labels=None):
        """
        See :method:`OpModelNoise.create_errormap_stencil`.
        """
        if opkey not in self.per_op_noise: return {}  # an empty stencil
        opnoise = self.per_op_noise[opkey]
        errmaps_to_embed_then_compose = _collections.OrderedDict()  # the "stencil" we return
        if isinstance(opnoise, dict):  # sslbls->OpNoise dict for potentially non-local noise
            opnoise_dict = opnoise
            for sslbls, opnoise in opnoise_dict.items():
                if sslbls is None:    # special behavior: `None` key => target labels
                    sslbls = tuple(['@{}'.format(i) for i in range(num_target_labels)])
                local_state_space = _StencilLabel.cast(sslbls).create_local_state_space(state_space)
                local_errormap = opnoise.create_errormap(evotype, local_state_space)

                if sslbls not in errmaps_to_embed_then_compose:  # allow multiple sslbls => same *effetive* sslbls
                    errmaps_to_embed_then_compose[sslbls] = local_errormap
                elif isinstance(errmaps_to_embed_then_compose[sslbls], _op.ComposedOp):
                    errmaps_to_embed_then_compose[sslbls].append(local_errormap)
                else:
                    errmaps_to_embed_then_compose[sslbls] = _op.ComposedOp(
                        [errmaps_to_embed_then_compose[sslbls], local_errormap])

        else:  # assume opnoise is an OpNoise object
            all_target_labels = None if (num_target_labels is None) else \
                tuple(['@{}'.format(i) for i in range(num_target_labels)])
            local_state_space = _StencilLabel.cast(all_target_labels).create_local_state_space(state_space) \
                if (all_target_labels is not None) else state_space
            local_errormap = opnoise.create_errormap(evotype, local_state_space)
            errmaps_to_embed_then_compose[all_target_labels] = local_errormap
        self._increment_touch_count(opkey)
        return errmaps_to_embed_then_compose

    def apply_errormap_stencil(self, stencil, evotype, state_space, target_labels=None, qudit_graph=None, copy=False):
        """
        See :method:`OpModelNoise.apply_errormap_stencil`.
        """
        embedded_errmaps = []
        for stencil_sslbls, local_errormap in stencil.items():
            sslbls_list = self._map_stencil_sslbls(stencil_sslbls, qudit_graph, state_space, target_labels)
            if None in sslbls_list and stencil_sslbls is not None:
                # `None` in list signals a non-present direction => skip these terms
                sslbls_list = list(filter(lambda x: x is not None, sslbls_list))
            for sslbls in sslbls_list:
                op_to_embed = local_errormap if (sslbls is None or state_space.is_entire_space(sslbls)) \
                    else _op.EmbeddedOp(state_space, sslbls, local_errormap)
                embedded_errmaps.append(op_to_embed.copy() if copy else op_to_embed)

        if len(embedded_errmaps) == 0:
            return None  # ==> no errormap (could return an empty ComposedOp instead?)
        else:
            return _op.ComposedOp(embedded_errmaps, evotype, state_space) \
                if len(embedded_errmaps) > 1 else embedded_errmaps[0]

    def _map_stencil_sslbls(self, stencil_sslbls, qudit_graph, state_space, target_lbls):  # deals with graph directions
        stencil_sslbls = _StencilLabel.cast(stencil_sslbls)
        return stencil_sslbls.compute_absolute_sslbls(qudit_graph, state_space, target_lbls)

    def compute_stencil_absolute_sslbls(self, stencil, state_space, target_labels=None, qudit_graph=None):
        """
        Computes the set of state space labels that would be utilized when applying a stencil.

        This function computes which state space labels are non-trivially acted upon by the
        operation that results from applying `stencil` to `target_labels`.

        Parameters
        ----------
        stencil : OrderedDict
            The stencil.  A dictionary with keys that are target state space labels (perhaps stencil labels)
            and values that are operations.  This function only cares about the keys of this dictionary.

        state_space : StateSpace
            The state space that would be given if/when applying `stencil`.  This should
            be the total state space of the model that the applied stencil would be inserted into.

        target_labels : tuple or None, optional
            The target labels that determine where on the qudit graph `stencil` will be placed.  `None`
            indicates that the entire space is the "target" space of the stencil.

        qudit_graph : QubitGraph, optional
            The relevant qudit graph that contains adjacency and direction information used to resolve stencil
            state space labels into absolute labels within `state_space`.  If `None`, then an error will be raised
            if any direction or connectivity information is needed to resolve the state space labels.

        Returns
        -------
        set
            A set (i.e. without any duplicates) of the state space labels that would be acted upon.
        """
        stencil_lbls = set()
        for stencil_sslbls, local_errorgen in stencil.items():
            sslbls_list = _StencilLabel.cast(stencil_sslbls).compute_absolute_sslbls(
                qudit_graph, state_space, target_labels)
            for sslbls in sslbls_list:
                stencil_lbls.update(sslbls if (sslbls is not None) else {})
        return stencil_lbls

    def _key_to_str(self, key, prefix=""):
        opnoise = self.per_op_noise.get(key, None)
        if opnoise is None:
            return prefix + str(key) + ": <missing>"

        if isinstance(opnoise, dict):  # sslbls->OpNoise dict for potentially non-local noise
            opnoise_dict = opnoise; val_str = ''
            for sslbls, opnoise in opnoise_dict.items():
                val_str += prefix + "  " + str(sslbls) + ": " + str(opnoise) + '\n'
        else:
            val_str = prefix + "  " + str(opnoise) + '\n'
        return prefix + str(key) + ":\n" + val_str

    def __str__(self):
        return '\n'.join([self._key_to_str(k) for k in self.keys()])


class ComposedOpModelNoise(OpModelNoise):
    """
    Op-model noise that is specified simply as the composition of other op-model noise specifications.

    Parameters
    ----------
    opmodelnoises : iterable
        The sub- :class:`OpModelNoise` objects.
    """

    def __init__(self, opmodelnoises):
        self.opmodelnoises = tuple(opmodelnoises)  # elements == OpModelNoise objects
        # self.ensure_no_duplicates()  # not actually needed; we just compose errors
        super(ComposedOpModelNoise, self).__init__()

    def ensure_no_duplicates(self):
        """
        Raise an AssertionError if there are any duplicates among the composed noise specifications.

        Returns
        -------
        None
        """
        running_keys = set()
        for modelnoise in self.opmodelnoises:
            duplicate_keys = running_keys.intersection(modelnoise.keys())
            assert (len(duplicate_keys) == 0), \
                "Duplicate keys not allowed in model noise specifications: %s" % ','.join(duplicate_keys)
            running_keys = running_keys.union(modelnoise.keys())

    def keys(self):
        """
        The operation labels for which this object specifies noise.
        """
        # Use remove_duplicates rather than set(.) to preserve ordering (but this is slower!)
        return _lt.remove_duplicates(_itertools.chain(*[modelnoise.keys() for modelnoise in self.opmodelnoises]))

    def __contains__(self, key):
        return any([(key in modelnoise) for modelnoise in self.opmodelnoises])

    def create_errorgen_stencil(self, opkey, evotype, state_space, num_target_labels=None):
        """
        See :method:`OpModelNoise.create_errorgen_stencil`.
        """
        self._increment_touch_count(opkey)
        return tuple([modelnoise.create_errorgen_stencil(opkey, evotype, state_space, num_target_labels)
                      for modelnoise in self.opmodelnoises])

    def apply_errorgen_stencil(self, stencil, evotype, state_space, target_labels=None, qudit_graph=None, copy=False):
        """
        See :method:`OpModelNoise.apply_errorgen_stencil`.
        """
        noise_errgens = [modelnoise.apply_errorgen_stencil(s, evotype, state_space, target_labels, qudit_graph, copy)
                         for s, modelnoise in zip(stencil, self.opmodelnoises)]
        noise_errgens = list(filter(lambda x: x is not None, noise_errgens))
        return _op.ComposedErrorgen(noise_errgens) if len(noise_errgens) > 1 \
            else (noise_errgens[0] if len(noise_errgens) == 1 else None)

    def create_errormap_stencil(self, opkey, evotype, state_space, num_target_labels=None):
        """
        See :method:`OpModelNoise.create_errormap_stencil`.
        """
        self._increment_touch_count(opkey)
        return tuple([modelnoise.create_errormap_stencil(opkey, evotype, state_space, num_target_labels)
                      for modelnoise in self.opmodelnoises])

    def apply_errormap_stencil(self, stencil, evotype, state_space, target_labels=None, qudit_graph=None, copy=False):
        """
        See :method:`OpModelNoise.apply_errormap_stencil`.
        """
        noise_ops = [modelnoise.apply_errormap_stencil(s, evotype, state_space, target_labels, qudit_graph, copy)
                     for s, modelnoise in zip(stencil, self.opmodelnoises)]
        noise_ops = list(filter(lambda x: x is not None, noise_ops))
        return _op.ComposedOp(noise_ops) if len(noise_ops) > 1 \
            else (noise_ops[0] if len(noise_ops) == 1 else None)

    def compute_stencil_absolute_sslbls(self, stencil, state_space, target_labels=None, qudit_graph=None):
        """
        Computes the set of state space labels that would be utilized when applying a stencil.

        This function computes which state space labels are non-trivially acted upon by the
        operation that results from applying `stencil` to `target_labels`.

        Parameters
        ----------
        stencil : OrderedDict
            The stencil.  A dictionary with keys that are target state space labels (perhaps stencil labels)
            and values that are operations.  This function only cares about the keys of this dictionary.

        state_space : StateSpace
            The state space that would be given if/when applying `stencil`.  This should
            be the total state space of the model that the applied stencil would be inserted into.

        target_labels : tuple or None, optional
            The target labels that determine where on the qudit graph `stencil` will be placed.  `None`
            indicates that the entire space is the "target" space of the stencil.

        qudit_graph : QubitGraph, optional
            The relevant qudit graph that contains adjacency and direction information used to resolve stencil
            state space labels into absolute labels within `state_space`.  If `None`, then an error will be raised
            if any direction or connectivity information is needed to resolve the state space labels.

        Returns
        -------
        set
            A set (i.e. without any duplicates) of the state space labels that would be acted upon.
        """
        stencil_lbls = set()
        for sub_stencil, modelnoise in zip(stencil, self.opmodelnoises):  # stencil is a tuple of compontent stencils
            stencil_lbls.update(modelnoise.compute_stencil_absolute_sslbls(sub_stencil, state_space,
                                                                           target_labels, qudit_graph))
        return stencil_lbls

    def _key_to_str(self, key, prefix=''):
        val_str = ''
        for i, modelnoise in enumerate(self.opmodelnoises):
            if key in modelnoise:
                val_str += prefix + ("  [%d]:\n" % i) + modelnoise._key_to_str(key, prefix + "    ")
        if len(val_str) > 0:
            return prefix + str(key) + ":\n" + val_str
        else:
            return prefix + str(key) + ": <missing>"

    def __str__(self):
        return '\n'.join([self._key_to_str(k) for k in self.keys()])


class OpNoise(object):
    """
    Specification for a single noise operation.

    An :class:`OpNoise` object specifies a single noisy operation that acts on some state space
    (e.g. number of qubits).  This specification doesn't contain any information about embedding
    this noise operation somewhere within a larger space -- it just specifies a noise operation on
    a local space.  This removes significant complexity from :class:`OpNoise` objects, and provides
    upstream objects like :class:`OpModelNoise` a common interface for working with all types of
    noisy operations.
    """
    def __str__(self):
        return self.__class__.__name__ + "(" + ", ".join(["%s=%s" % (str(k), str(v))
                                                          for k, v in self.__dict__.items()]) + ")"


class DepolarizationNoise(OpNoise):
    """
    Depolarization noise.

    Parameters
    ----------
    depolarization_rate : float
        The uniform depolarization strength.

    parameterization : {"depolarize", "stochastic", or "lindblad"}
        Determines whether a :class:`DepolarizeOp`, :class:`StochasticNoiseOp`, or
        :class:`LindbladErrorgen` is used to represent the depolarization noise, respectively.
        When "depolarize" (the default), a DepolarizeOp is created with the strength given
        in `depolarization_strengths`. When "stochastic", the depolarization strength is split
        evenly among the stochastic channels of a StochasticOp. When "lindblad", the depolarization
        strength is split evenly among the coefficients of the stochastic error generators
        (which are exponentiated to form a LindbladErrorgen with the "depol" parameterization).
    """
    def __init__(self, depolarization_rate, parameterization='depolarize'):
        self.depolarization_rate = depolarization_rate
        self.parameterization = parameterization

        valid_depol_params = ['depolarize', 'stochastic', 'lindblad']
        assert (self.parameterization in valid_depol_params), \
            "The depolarization parameterization must be one of %s, not %s" \
            % (valid_depol_params, self.parameterization)

    def create_errorgen(self, evotype, state_space):
        """
        Create an error generator for this noise operation.

        Parameters
        ----------
        evotype : str or Evotype
            The evolution type for the returned operator.

        state_space : StateSpace
            The state space for the returned operator.

        Returns
        -------
        LinearOperator
        """
        if self.parameterization != 'lindblad':
            raise ValueError("Can only construct error generators for 'lindblad' parameterization")

        # LindbladErrorgen with "depol" or "diagonal" param
        basis_size = state_space.dim  # e.g. 4 for a single qubit
        basis = _BuiltinBasis('pp', basis_size)
        rate_per_pauli = self.depolarization_rate / (basis_size - 1)
        errdict = {('S', bl): rate_per_pauli for bl in basis.labels[1:]}
        return _op.LindbladErrorgen.from_elementary_errorgens(
            errdict, "D", basis, mx_basis='pp',
            truncate=False, evotype=evotype, state_space=state_space)

    def create_errormap(self, evotype, state_space):
        """
        Create an error map (operator or superoperator) for this noise operation.

        Parameters
        ----------
        evotype : str or Evotype
            The evolution type for the returned operator.

        state_space : StateSpace
            The state space for the returned operator.

        Returns
        -------
        LinearOperator
        """
        basis_size = state_space.dim  # e.g. 4 for a single qubit

        if self.parameterization == "depolarize":  # DepolarizeOp
            return _op.DepolarizeOp(state_space, basis="pp", evotype=evotype,
                                    initial_rate=self.depolarization_rate)

        elif self.parameterization == "stochastic":  # StochasticNoiseOp
            rate_per_pauli = self.depolarization_rate / (basis_size - 1)
            rates = [rate_per_pauli] * (basis_size - 1)
            return _op.StochasticNoiseOp(state_space, basis="pp", evotype=evotype, initial_rates=rates)

        elif self.parameterization == "lindblad":
            errgen = self.create_errorgen(evotype, state_space)
            return _op.ExpErrorgenOp(errgen)

        else:
            raise ValueError("Unknown parameterization %s for depolarizing error specification"
                             % self.parameterization)


class StochasticNoise(OpNoise):
    """
    Pauli stochastic noise.

    Parameters
    ----------
    error_probs : tuple
        The Pauli-stochastic rates for each of the non-trivial Paulis (a 3-tuple is expected for a
        1Q gate and a 15-tuple for a 2Q gate).

    parameterization : {"stochastic", or "lindblad"}
        Determines whether a :class:`StochasticNoiseOp` or :class:`LindbladErrorgen` is used to
        represent the stochastic noise, respectively. When `"stochastic"`, elements of `error_probs`
        are used as coefficients in a linear combination of stochastic channels (the default).
        When `"lindblad"`, the elements of `error_probs` are coefficients of stochastic error
        generators (which are exponentiated to form a LindbladErrorgen with "cptp" non-Hammiltonian
        parameterization).
    """
    def __init__(self, error_probs, parameterization='stochastic'):
        self.error_probs = error_probs
        self.parameterization = parameterization

        valid_sto_params = ['stochastic', 'lindblad']
        assert (self.parameterization in valid_sto_params), \
            "The stochastic parameterization must be one of %s, not %s" \
            % (valid_sto_params, self.parameterization)

    def create_errorgen(self, evotype, state_space):
        """
        Create an error generator for this noise operation.

        Parameters
        ----------
        evotype : str or Evotype
            The evolution type for the returned operator.

        state_space : StateSpace
            The state space for the returned operator.

        Returns
        -------
        LinearOperator
        """
        sto_rates = self.error_probs

        if self.parameterization != 'lindblad':
            raise ValueError("Cannot only construct error generators for 'lindblad' parameterization")

        basis_size = state_space.dim  # e.g. 4 for a single qubit
        basis = _BuiltinBasis('pp', basis_size)
        errdict = {('S', bl): rate for bl, rate in zip(basis.labels[1:], sto_rates)}
        return _op.LindbladErrorgen.from_elementary_errorgens(
            errdict, "S", basis, mx_basis='pp',
            truncate=False, evotype=evotype, state_space=state_space)

    def create_errormap(self, evotype, state_space):
        """
        Create an error map (operator or superoperator) for this noise operation.

        Parameters
        ----------
        evotype : str or Evotype
            The evolution type for the returned operator.

        state_space : StateSpace
            The state space for the returned operator.

        Returns
        -------
        LinearOperator
        """
        sto_rates = self.error_probs

        if self.parameterization == "stochastic":  # StochasticNoiseOp
            return _op.StochasticNoiseOp(state_space, basis="pp", evotype=evotype, initial_rates=sto_rates)

        elif self.parameterization == "lindblad":  # LindbladErrorgen with "cptp", "diagonal" parameterization
            errgen = self.create_errorgen(evotype, state_space)
            return _op.ExpErrorgenOp(errgen)
        else:
            raise ValueError("Unknown parameterization %s for stochastic error specification"
                             % self.parameterization)


class LindbladNoise(OpNoise):
    """
    Noise generated by exponentiating a Lindbladian error generator.

    The error generator is a Lindblad-form sum of elementary error generators
    corresponding to Hamiltonian and other (stochastic, correlation, etc.) type of errors.

    Parameters
    ----------
    error_coeffs : dict
        A dictionary of Lindblad-term coefficients. Keys are
        `(termType, basisLabel1, <basisLabel2>)` tuples, where `termType` can be
        `"H"` (Hamiltonian), `"S"` (Stochastic/other), or `"A"` (Affine).  Hamiltonian
        and Affine terms always have a single basis label (so key is a 2-tuple) whereas
        Stochastic/other tuples can have 1 or 2 basis labels depending on the
        parameterization type.   Tuples with 1 basis label indicate a stochastic
        (diagonal Lindblad) term, and are the only type of terms allowed when a parmeterization
        with `nonham_mode != "all"` is selected.  `"S"` terms with 2 basis specify
        "off-diagonal" non-Hamiltonian Lindblad terms.  Basis labels can be
        strings or integers.  Values are complex coefficients.

    parameterization : str or LindbladParameterization
        Determines the parameterization of the LindbladErrorgen objects used to represent this noise.
        When "auto" (the default), the parameterization is inferred from the types of error generators
        specified in the `error_coeffs` dictionary. When not "auto", the parameterization type is
        passed through to created :class:`LindbladErrorgen` objects.
    """

    @classmethod
    def from_basis_coefficients(cls, parameterization, lindblad_basis, state_space, ham_coefficients=None,
                                nonham_coefficients=None):
        """
        Create a :class:`LindbladNoise` object containing a complete basis of elementary terms.

        This method provides a convenient way to create a lindblad noise specification containing the
        complete set of terms in a Lindbladian based on a given "Lindblad basis" (often just a Pauli product
        basis).  This routine by default creates all the terms with zero coefficients, but coefficient vectors
        or matrices (usually obtained by projecting an arbitrary error generator onto the lindblad basis) can
        be specified via the `ham_coefficients` and `nonham_coefficients` arguments.

        Parameters
        ----------
        parameterization : str or LindbladParameterization
            The Lindblad parameterization, specifying what constitutes the "complete" set of
            Lindblad terms.  For example, `"H"` means that just Hamiltonian terms are included
            whereas `"CPTP"` includes all the terms in a standard Lindblad decomposition.

        lindblad_basis : str or Basis
            The basis used to construct the Lindblad terms.

        state_space : StateSpace
            The state space, used only to convert string-valued `lindblad_basis` names into a
            :class:`Basis` object.  If `lindblad_basis` is given as a :class:`Basis`, then this
            can be set to `None`.

        ham_coefficients : numpy.ndarray or None, optional
            A 1-dimensional array of coefficients giving the initial values of the Hamiltonian-term
            coefficients.  The length of this arrays should be one less than the size of `lindblad_basis`
            (since there's no Lindblad term for the identity element).

        nonham_coefficients : numpy.ndarray or None, optional
            A 1- or 2-dimensional array of coefficients for the "other" (non-Hamiltonian) terms.
            The shape of this array should be `(d,)`, `(2,d)`, or `(d,d)` depending on `parameterization`
            (e.g. for S, S+A, and CPTP parameterizations).

        Returns
        -------
        LindbladNoise
        """
        lindblad_basis = _Basis.cast(lindblad_basis, state_space)

        parameterization = _op.LindbladParameterization.cast(parameterization)
        ham_basis = lindblad_basis if parameterization.ham_params_allowed else None
        nonham_basis = lindblad_basis if parameterization.nonham_params_allowed else None

        if ham_coefficients is None and ham_basis is not None:
            ham_coefficients = _np.zeros(len(ham_basis) - 1, 'd')

        if nonham_coefficients is None and nonham_basis is not None:
            d = len(nonham_basis) - 1
            if parameterization.nonham_mode == 'all':
                nonham_coefficients = _np.zeros((d, d), complex)
            #REMOVE elif parameterization.nonham_mode == 'diag_affine':
            #REMOVE     nonham_coefficients = _np.zeros((2, d), 'd')
            else:
                nonham_coefficients = _np.zeros(d, 'd')

        # coeffs + bases => elementary errorgen dict
        elementary_errorgens = {}
        if ham_basis is not None:
            blk = _LindbladCoefficientBlock('ham', ham_basis, initial_block_data=ham_coefficients)
            elementary_errorgens.update(blk.elementary_errorgens)
        if nonham_basis is not None:
            blk = _LindbladCoefficientBlock('other' if (parameterization.nonham_mode == 'all') else 'other_diagonal',
                                            nonham_basis, initial_block_data=nonham_coefficients)
            elementary_errorgens.update(blk.elementary_errorgens)

        return cls(elementary_errorgens, parameterization)

    def __init__(self, error_coeffs, parameterization='auto'):
        self.error_coeffs = error_coeffs  # keys are LocalElementaryErrorgenLabel objects
        self.parameterization = parameterization

    def create_errorgen(self, evotype, state_space):
        """
        Create an error generator for this noise operation.

        Parameters
        ----------
        evotype : str or Evotype
            The evolution type for the returned operator.

        state_space : StateSpace
            The state space for the returned operator.

        Returns
        -------
        LinearOperator
        """
        # Build LindbladErrorgen directly to have control over which parameters are set (leads to lower param counts)
        basis_size = state_space.dim  # e.g. 4 for a single qubit
        basis = _BuiltinBasis('pp', basis_size)
        return _op.LindbladErrorgen.from_elementary_errorgens(
            self.error_coeffs, self.parameterization, basis, mx_basis='pp',
            truncate=False, evotype=evotype, state_space=state_space)

    def create_errormap(self, evotype, state_space):
        """
        Create an error map (operator or superoperator) for this noise operation.

        Parameters
        ----------
        evotype : str or Evotype
            The evolution type for the returned operator.

        state_space : StateSpace
            The state space for the returned operator.

        Returns
        -------
        LinearOperator
        """
        errgen = self.create_errorgen(evotype, state_space)
        return _op.ExpErrorgenOp(errgen)
