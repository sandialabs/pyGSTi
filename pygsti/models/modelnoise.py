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
from pygsti.baseobjs.basis import Basis as _Basis
from pygsti.baseobjs.basis import BuiltinBasis as _BuiltinBasis
from pygsti.circuits.circuitparser import CircuitParser as _CircuitParser


class ModelNoise(object):
    """ TODO: docstring -- lots of docstrings to do from here downward!"""
    pass


class OpModelNoise(ModelNoise):

    @classmethod
    def cast(cls, obj):
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
        raise NotImplementedError("Derived classes should implement this!")

    def __contains__(self, key):
        raise NotImplementedError("Derived classes should implement this!")

    def create_errorgen_stencil(self, opkey, evotype, state_space, target_labels=None):
        raise NotImplementedError("Derived classes should implement this!")

    def apply_errorgen_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        raise NotImplementedError("Derived classes should implement this!")

    def create_errorgen(self, opkey, evotype, state_space, target_labels=None, qubit_graph=None):
        stencil = self.create_errorgen_stencil(opkey, evotype, state_space, target_labels)
        return self.apply_errorgen_stencil(stencil, evotype, state_space, target_labels, qubit_graph)

    def create_errormap_stencil(self, opkey, evotype, state_space, target_labels=None):
        raise NotImplementedError("Derived classes should implement this!")

    def apply_errormap_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        raise NotImplementedError("Derived classes should implement this!")

    def create_errormap(self, opkey, evotype, state_space, target_labels=None, qubit_graph=None):
        stencil = self.create_errormap_stencil(opkey, evotype, state_space, target_labels)
        return self.apply_errormap_stencil(stencil, evotype, state_space, target_labels, qubit_graph)

    def reset_access_counters(self):
        self._opkey_access_counters = {k: 0 for k in self.keys()}

    def _increment_touch_count(self, opkey):
        if opkey in self._opkey_access_counters:
            self._opkey_access_counters[opkey] += 1

    def warn_about_zero_counters(self):
        untouched_keys = [k for k, touch_cnt in self._opkey_access_counters.items() if touch_cnt == 0]
        if len(untouched_keys) > 0:
            _warnings.warn(("The following model-noise entries were unused: %s.  You may want to double check"
                            " that you've entered a valid noise specification.") % ", ".join(map(str, untouched_keys)))

    def compute_stencil_absolute_sslbls(self, stencil, state_space, target_labels=None, qubit_graph=None):
        raise NotImplementedError("Derived classes should implement this!")


class OpModelPerOpNoise(OpModelNoise):

    def __init__(self, per_op_noise):
        # a dictionary mapping operation keys -> OpNoise objects
        #                                  OR -> {dict mapping sslbls -> OpNoise objects}
        self.per_op_noise = per_op_noise.copy()

        # Update any label-string format keys to actual Labels (convenience for users)
        cparser = _CircuitParser()
        cparser.lookup = None  # lookup - functionality removed as it wasn't used
        for k, v in per_op_noise.items():
            if isinstance(k, str) and ":" in k:  # then parse this to get a label, allowing, e.g. "Gx:0"
                lbls, _, _ = cparser.parse(k)
                assert (len(lbls) == 1), "Only single primitive-gate labels allowed as keys! (not %s)" % str(k)
                del self.per_op_noise[k]
                self.per_op_noise[lbls[0]] = v

        super(OpModelPerOpNoise, self).__init__()

    def keys(self):
        return self.per_op_noise.keys()

    def __contains__(self, key):
        return key in self.per_op_noise

    def create_errorgen_stencil(self, opkey, evotype, state_space, target_labels=None):
        if opkey not in self.per_op_noise: return {}  # an empty stencil
        opnoise = self.per_op_noise[opkey]
        errgens_to_embed_then_compose = _collections.OrderedDict()  # the "stencil" we return
        if isinstance(opnoise, dict):  # sslbls->OpNoise dict for potentially non-local noise
            opnoise_dict = opnoise
            for sslbls, opnoise in opnoise_dict.items():
                if sslbls is None:    # special behavior: `None` key => target labels
                    sslbls = tuple(['@{}'.format(i) for i in range(len(target_labels))])
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
            errgens_to_embed_then_compose[target_labels] = local_errorgen
        self._increment_touch_count(opkey)
        return errgens_to_embed_then_compose

    def apply_errorgen_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        embedded_errgens = []
        for stencil_sslbls, local_errorgen in stencil.items():
            sslbls_list = self._map_stencil_sslbls(stencil_sslbls, qubit_graph, state_space, target_labels)
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

    def create_errormap_stencil(self, opkey, evotype, state_space, target_labels=None):
        if opkey not in self.per_op_noise: return {}  # an empty stencil
        opnoise = self.per_op_noise[opkey]
        errmaps_to_embed_then_compose = _collections.OrderedDict()  # the "stencil" we return
        if isinstance(opnoise, dict):  # sslbls->OpNoise dict for potentially non-local noise
            opnoise_dict = opnoise
            for sslbls, opnoise in opnoise_dict.items():
                if sslbls is None:    # special behavior: `None` key => target labels
                    sslbls = tuple(['@{}'.format(i) for i in range(len(target_labels))])
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
            local_errormap = opnoise.create_errormap(evotype, state_space)
            errmaps_to_embed_then_compose[target_labels] = local_errormap
        self._increment_touch_count(opkey)
        return errmaps_to_embed_then_compose

    def apply_errormap_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        embedded_errmaps = []
        for stencil_sslbls, local_errormap in stencil.items():
            sslbls_list = self._map_stencil_sslbls(stencil_sslbls, qubit_graph, state_space, target_labels)
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

    def _map_stencil_sslbls(self, stencil_sslbls, qubit_graph, state_space, target_lbls):  # deals with graph directions
        stencil_sslbls = _StencilLabel.cast(stencil_sslbls)
        return stencil_sslbls.compute_absolute_sslbls(qubit_graph, state_space, target_lbls)

    def compute_stencil_absolute_sslbls(self, stencil, state_space, target_labels=None, qubit_graph=None):
        stencil_lbls = set()
        for stencil_sslbls, local_errorgen in stencil.items():
            sslbls_list = _StencilLabel.cast(stencil_sslbls).compute_absolute_sslbls(
                qubit_graph, state_space, target_labels)
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
    def __init__(self, opmodelnoises):
        self.opmodelnoises = tuple(opmodelnoises)  # elements == OpModelNoise objects
        # self.ensure_no_duplicates()  # not actually needed; we just compose errors
        super(ComposedOpModelNoise, self).__init__()

    def ensure_no_duplicates(self):
        running_keys = set()
        for modelnoise in self.opmodelnoises:
            duplicate_keys = running_keys.intersection(modelnoise.keys())
            assert (len(duplicate_keys) == 0), \
                "Duplicate keys not allowed in model noise specifications: %s" % ','.join(duplicate_keys)
            running_keys = running_keys.union(modelnoise.keys())

    def keys(self):
        # Use remove_duplicates rather than set(.) to preserve ordering (but this is slower!)
        return _lt.remove_duplicates(_itertools.chain(*[modelnoise.keys() for modelnoise in self.opmodelnoises]))

    def __contains__(self, key):
        return any([(key in modelnoise) for modelnoise in self.opmodelnoises])

    def create_errorgen_stencil(self, opkey, evotype, state_space, target_labels=None):
        self._increment_touch_count(opkey)
        return tuple([modelnoise.create_errorgen_stencil(opkey, evotype, state_space, target_labels)
                      for modelnoise in self.opmodelnoises])

    def apply_errorgen_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        noise_errgens = [modelnoise.apply_errorgen_stencil(s, evotype, state_space, target_labels, qubit_graph, copy)
                         for s, modelnoise in zip(stencil, self.opmodelnoises)]
        noise_errgens = list(filter(lambda x: x is not None, noise_errgens))
        return _op.ComposedErrorgen(noise_errgens) if len(noise_errgens) > 0 else None

    def create_errormap_stencil(self, opkey, evotype, state_space, target_labels=None):
        self._increment_touch_count(opkey)
        return tuple([modelnoise.create_errormap_stencil(opkey, evotype, state_space, target_labels)
                      for modelnoise in self.opmodelnoises])

    def apply_errormap_stencil(self, stencil, evotype, state_space, target_labels=None, qubit_graph=None, copy=False):
        noise_ops = [modelnoise.apply_errormap_stencil(s, evotype, state_space, target_labels, qubit_graph, copy)
                     for s, modelnoise in zip(stencil, self.opmodelnoises)]
        noise_ops = list(filter(lambda x: x is not None, noise_ops))
        return _op.ComposedOp(noise_ops) if len(noise_ops) > 0 else None

    def compute_stencil_absolute_sslbls(self, stencil, state_space, target_labels=None, qubit_graph=None):
        stencil_lbls = set()
        for sub_stencil, modelnoise in zip(stencil, self.opmodelnoises):  # stencil is a tuple of compontent stencils
            stencil_lbls.update(modelnoise.compute_stencil_absolute_sslbls(sub_stencil, state_space,
                                                                           target_labels, qubit_graph))
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
    def __str__(self):
        return self.__class__.__name__ + "(" + ", ".join(["%s=%s" % (str(k), str(v))
                                                          for k, v in self.__dict__.items()]) + ")"


class DepolarizationNoise(OpNoise):
    def __init__(self, depolarization_rate, parameterization='depolarize'):
        self.depolarization_rate = depolarization_rate
        self.parameterization = parameterization

        valid_depol_params = ['depolarize', 'stochastic', 'lindblad']
        assert (self.parameterization in valid_depol_params), \
            "The depolarization parameterization must be one of %s, not %s" \
            % (valid_depol_params, self.parameterization)

    def create_errorgen(self, evotype, state_space):
        if self.parameterization != 'lindblad':
            raise ValueError("Can only construct error generators for 'lindblad' parameterization")

        # LindbladErrorgen with "depol" or "diagonal" param
        basis_size = state_space.dim  # e.g. 4 for a single qubit
        basis = _BuiltinBasis('pp', basis_size)
        rate_per_pauli = self.depolarization_rate / (basis_size - 1)
        errdict = {('S', bl): rate_per_pauli for bl in basis.labels[1:]}
        return _op.LindbladErrorgen(errdict, "D", basis, mx_basis='pp',
                                    truncate=False, evotype=evotype, state_space=state_space)

    def create_errormap(self, evotype, state_space):
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
    def __init__(self, error_probs, parameterization='stochastic'):
        self.error_probs = error_probs
        self.parameterization = parameterization

        valid_sto_params = ['stochastic', 'lindblad']
        assert (self.parameterization in valid_sto_params), \
            "The stochastic parameterization must be one of %s, not %s" \
            % (valid_sto_params, self.parameterization)

    def create_errorgen(self, evotype, state_space):
        sto_rates = self.error_probs

        if self.parameterization != 'lindblad':
            raise ValueError("Cannot only construct error generators for 'lindblad' parameterization")

        basis_size = state_space.dim  # e.g. 4 for a single qubit
        basis = _BuiltinBasis('pp', basis_size)
        errdict = {('S', bl): rate for bl, rate in zip(basis.labels[1:], sto_rates)}
        return _op.LindbladErrorgen(errdict, "S", basis, mx_basis='pp',
                                    truncate=False, evotype=evotype, state_space=state_space)

    def create_errormap(self, evotype, state_space):
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
    @classmethod
    def from_basis_coefficients(cls, parameterization, lindblad_basis, state_space, ham_coefficients=None,
                                nonham_coefficients=None):
        """ TODO: docstring - None coefficients mean zeros"""
        dim = state_space.dim
        lindblad_basis = _Basis.cast(lindblad_basis, dim)

        parameterization = _op.LindbladParameterization.cast(parameterization)
        ham_basis = lindblad_basis if parameterization.ham_params_allowed else None
        nonham_basis = lindblad_basis if parameterization.nonham_params_allowed else None

        if ham_coefficients is None and ham_basis is not None:
            ham_coefficients = _np.zeros(len(ham_basis) - 1, 'd')
        if nonham_coefficients is None and nonham_basis is not None:
            d = len(ham_basis) - 1
            nonham_coefficients = _np.zeros((d, d), complex) if parameterization.nonham_mode == 'all' \
                else _np.zeros(d, 'd')

        # coeffs + bases => Ltermdict, basis
        Ltermdict, _ = _ot.projections_to_lindblad_terms(
            ham_coefficients, nonham_coefficients, ham_basis, nonham_basis, parameterization.nonham_mode)
        return cls(Ltermdict, parameterization)

    def __init__(self, error_coeffs, parameterization='auto'):
        self.error_coeffs = error_coeffs
        self.parameterization = parameterization

    def create_errorgen(self, evotype, state_space):
        # Build LindbladErrorgen directly to have control over which parameters are set (leads to lower param counts)
        basis_size = state_space.dim  # e.g. 4 for a single qubit
        basis = _BuiltinBasis('pp', basis_size)
        return _op.LindbladErrorgen(self.error_coeffs, self.parameterization, basis, mx_basis='pp',
                                    truncate=False, evotype=evotype, state_space=state_space)

    def create_errormap(self, evotype, state_space):
        errgen = self.create_errorgen(evotype, state_space)
        return _op.ExpErrorgenOp(errgen)
