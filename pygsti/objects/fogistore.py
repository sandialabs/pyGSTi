"""
Defines the FirstOrderGaugeInvariantStore class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import copy as _copy
import warnings as _warnings
from ..tools import matrixtools as _mt
from ..tools import fogitools as _fogit


class FirstOrderGaugeInvariantStore(object):
    """
    An object that computes and stores the first-order-gauge-invariant quantities of a model.

    Currently, it is only compatible with :class:`ExplicitOpModel` objects.
    """

    def __init__(self, gauge_action_matrices_by_op, errorgen_coefficient_labels_by_op,
                 elem_errorgen_labels, op_label_abbrevs=None, reduce_to_model_space=True,
                 dependent_fogi_action='drop', norm_order=None):
        """
        TODO: docstring
        """

        self.primitive_op_labels = tuple(gauge_action_matrices_by_op.keys())
        #self.gauge_action_for_op = gauge_action_matrices_by_op
        #errorgen_coefficient_labels_by_op

        self.allop_gauge_action, self.gauge_action_for_op, self.elem_errorgen_labels_by_op, self.gauge_linear_combos = \
            _fogit.construct_gauge_space_for_model(self.primitive_op_labels, gauge_action_matrices_by_op,
                                                   errorgen_coefficient_labels_by_op, elem_errorgen_labels,
                                                   reduce_to_model_space)

        self.errgen_space_op_elem_labels = tuple([(op_label, elem_lbl) for op_label in self.primitive_op_labels
                                                  for elem_lbl in self.elem_errorgen_labels_by_op[op_label]])

        if self.gauge_linear_combos is not None:
            self._gauge_space_dim = self.gauge_linear_combos.shape[1]
            self.gauge_elemgen_labels = [('G', str(i)) for i in range(self._gauge_space_dim)]
        else:
            self._gauge_space_dim = len(elem_errorgen_labels)
            self.gauge_elemgen_labels = elem_errorgen_labels

        (self.fogi_opsets, self.fogi_directions, self.fogi_r_factors, self.fogi_gaugespace_directions,
         self.dependent_dir_indices, self.op_errorgen_indices, self.fogi_labels, self.abbrev_fogi_labels) = \
            _fogit.construct_fogi_quantities(self.primitive_op_labels, self.gauge_action_for_op,
                                             self.elem_errorgen_labels_by_op, self.gauge_elemgen_labels,
                                             op_label_abbrevs, dependent_fogi_action, norm_order)
        self.norm_order = norm_order

        self.errorgen_space_labels = [(op_label, elem_lbl) for op_label in self.primitive_op_labels
                                      for elem_lbl in self.elem_errorgen_labels_by_op[op_label]]
        assert(len(self.errorgen_space_labels) == self.fogi_directions.shape[0])

        #fogv_directions = _mt.nice_nullspace(self.fogi_directions.T)  # can be dependent!
        fogv_directions = _mt.nullspace(self.fogi_directions.T)  # complement of fogi directions

        pinv_allop_gauge_action = _np.linalg.pinv(self.allop_gauge_action, rcond=1e-7)  # errgen-set -> gauge-gen space
        gauge_space_directions = _np.dot(pinv_allop_gauge_action, fogv_directions)  # in gauge-generator space
        self.gauge_space_directions = gauge_space_directions

        self.fogv_labels = ["%d gauge action" % i for i in range(gauge_space_directions.shape[1])]
        #self.fogv_labels = ["%s gauge action" % nm
        #                    for nm in _fogit.elem_vec_names(gauge_space_directions, gauge_elemgen_labels)]
        self.fogv_directions = fogv_directions
        # - directions in errorgen space that correspond to gauge transformations (to first order)

        # BELOW: an attempt to find nice FOGV directions - but we'd like all the vecs to be
        #  orthogonal and this seems to interfere with that, so we'll just leave the fogv dirs messy for now.
        #
        # # like to find LCs mix s.t.  dot(gauge_space_directions, mix) ~= identity, so use pinv
        # # then propagate this mixing to fogv_directions = dot(allop_gauge_action, mixed_gauge_space_directions)
        # mix = _np.linalg.pinv(gauge_space_directions)[:, 0:fogv_directions.shape[1]]  # use "full-rank" part of pinv
        # mixed_gauge_space_dirs = _np.dot(gauge_space_directions, mix)
        # 
        # #TODO - better mix matrix?
        # #print("gauge_space_directions shape = ",gauge_space_directions.shape)
        # #print("mixed_gauge_space_dirs = ");  _mt.print_mx(gauge_space_directions, width=6, prec=2)
        # #U, s, Vh = _np.linalg.svd(gauge_space_directions, full_matrices=True)
        # #inv_s = _np.array([1/x if abs(x) > 1e-4 else 0 for x in s])
        # #print("shapes = ",U.shape, s.shape, Vh.shape)
        # #print("s = ",s)
        # #print(_np.linalg.norm(_np.dot(U,_np.conjugate(U.T)) - _np.identity(U.shape[0])))
        # #_mt.print_mx(U, width=6, prec=2)
        # #print("U * Udag = ")
        # #_mt.print_mx(_np.dot(U,_np.conjugate(U.T)), width=6, prec=2)
        # #print(_np.linalg.norm(_np.dot(Vh,Vh.T) - _np.identity(Vh.shape[0])))
        # #full_mix = _np.dot(Vh.T, _np.dot(_np.diag(inv_s), U.T))  # _np.linalg.pinv(gauge_space_directions)
        # #full_mixed_gauge_space_dirs = _np.dot(gauge_space_directions, full_mix)
        # #print("full_mixed_gauge_space_dirs = ");  _mt.print_mx(full_mixed_gauge_space_dirs, width=6, prec=2)
        # 
        # self.fogv_labels = ["%s gauge action" % nm
        #                     for nm in _fogit.elem_vec_names(mixed_gauge_space_dirs, gauge_elemgen_labels)]
        # self.fogv_directions = _np.dot(self.allop_gauge_action, mixed_gauge_space_dirs)
        # # - directions in errorgen space that correspond to gauge transformations (to first order)

        #UNUSED - maybe useful just as a check? otherwise REMOVE
        #pinv_allop_gauge_action = _np.linalg.pinv(self.allop_gauge_action, rcond=1e-7)  # maps error -> gauge-gen space
        #gauge_space_directions = _np.dot(pinv_allop_gauge_action, self.fogv_directions)  # in gauge-generator space
        #assert(_np.linalg.matrix_rank(gauge_space_directions) <= self._gauge_space_dim)  # should be nearly full rank


        #Notes on error-gen vs gauge-gen space:
        # self.fogi_directions and self.fogv_directions are dual vectors in error-generator space,
        # i.e. with elements corresponding to the elementary error generators given by self.errorgen_space_labels.
        # self.fogi_gaugespace_directions contains, when applicable, a gauge-space direction that
        # correspondings to the FOGI quantity in self.fogi_directions.  Such a gauge-space direction
        # exists for relational FOGI quantities, where the FOGI quantity is constructed by taking the
        # *difference* of a gauge-space action (given by the gauge-space direction) on two operations
        # (or sets of operations).

        self.raw_fogi_labels = _fogit.op_elem_vec_names(self.fogi_directions,
                                                        self.errorgen_space_labels, op_label_abbrevs)

        # We must reduce X_gauge_action to the "in-model gauge space" before testing whether the computed vecs are FOGI:
        assert(_np.linalg.norm(_np.dot(self.allop_gauge_action.T, self.fogi_directions)) < 1e-8)

        #Check that pseudo-inverse was computed correctly (~ matrices are full rank)
        # fogi_coeffs = dot(fogi_directions.T, elem_errorgen_vec), where elem_errorgen_vec is filled from model params,
        #                 since fogi_directions columns are *dual* vectors in error-gen space.  Thus, to go in reverse:
        # elem_errogen_vec = dot(pinv_fogi_dirs_T, fogi_coeffs), where dot(fogi_directions.T, pinv_fogi_dirs_T) == I
        # (This will only be the case when fogi_vecs are linearly independent, so when dependent_indices == 'drop')
        self._dependent_fogi_action = dependent_fogi_action
        if dependent_fogi_action == 'drop':
            #assert(_mt.columns_are_orthogonal(self.fogi_directions))  # not true unless we construct them so...
            assert(_np.linalg.norm(_np.dot(self.fogi_directions.T, _np.linalg.pinv(self.fogi_directions.T))
                                   - _np.identity(self.fogi_directions.shape[1], 'd')) < 1e-6)

        # A similar relationship should always hold for the gauge directions, except for these we never
        #  keep linear dependencies
        assert(_mt.columns_are_orthogonal(self.fogv_directions))
        assert(_np.linalg.norm(_np.dot(self.fogv_directions.T, _np.linalg.pinv(self.fogv_directions.T))
                               - _np.identity(self.fogv_directions.shape[1], 'd')) < 1e-6)

    @property
    def errorgen_space_dim(self):
        return self.fogi_directions.shape[0]

    @property
    def gauge_space_dim(self):
        return self._gauge_space_dim

    @property
    def num_fogi_directions(self):
        return self.fogi_directions.shape[1]

    @property
    def num_fogv_directions(self):
        return self.fogv_directions.shape[1]

    def fogi_errorgen_direction_labels(self, typ='normal'):
        """ typ can be 'raw' or 'abbrev' too """
        if typ == 'normal': labels = self.fogi_labels
        elif typ == 'raw': labels = self.raw_fogi_labels
        elif typ == 'abrev': labels = self.abbrev_fogi_labels
        else: raise ValueError("Invalid `typ` argument: %s" % str(typ))
        return tuple(labels)

    def fogv_errorgen_direction_labels(self, typ='normal'):
        if typ == 'normal': labels = self.fogv_labels
        else: labels = [''] * len(self.fogv_labels)
        return tuple(labels)

    def errorgen_vec_to_fogi_components_array(self, errorgen_vec):
        fogi_coeffs = _np.dot(self.fogi_directions.T, errorgen_vec)
        assert(_np.linalg.norm(fogi_coeffs.imag) < 1e-8)
        return fogi_coeffs.real

    def errorgen_vec_to_fogv_components_array(self, errorgen_vec):
        fogv_coeffs = _np.dot(self.fogv_directions.T, errorgen_vec)
        assert(_np.linalg.norm(fogv_coeffs.imag) < 1e-8)
        return fogv_coeffs.real

    def opcoeffs_to_fogi_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogi_components_array(errorgen_vec)

    def opcoeffs_to_fogv_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogv_components_array(errorgen_vec)

    def opcoeffs_to_fogiv_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogi_components_array(errorgen_vec), \
            self.errorgen_vec_to_fogv_components_array(errorgen_vec)

    def fogi_components_array_to_errorgen_vec(self, fogi_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogi components to an errorgen-set vec when fogi directions are linearly-dependent!"
             "  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        return _np.dot(_np.linalg.pinv(self.fogi_directions.T, rcond=1e-7), fogi_components)

    def fogv_components_array_to_errorgen_vec(self, fogv_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogv components to an errorgen-set vec when fogi directions are linearly-dependent!"
             "  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        return _np.dot(_np.linalg.pinv(self.fogv_directions.T, rcond=1e-7), fogv_components)

    def fogiv_components_array_to_errorgen_vec(self, fogi_components, fogv_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogiv components to an errorgen-set vec when fogi directions are "
             "linearly-dependent!  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        return _np.dot(_np.linalg.pinv(
            _np.concatenate((self.fogi_directions, self.fogv_directions), axis=1).T,
            rcond=1e-7), _np.concatenate((fogi_components, fogv_components)))

    def errorgen_vec_to_opcoeffs(self, errorgen_vec):
        op_coeffs = {op_label: {} for op_label in self.primitive_op_labels}
        for (op_label, elem_lbl), coeff_value in zip(self.errgen_space_op_elem_labels, errorgen_vec):
            op_coeffs[op_label][elem_lbl] = coeff_value
        return op_coeffs

    def fogi_components_array_to_opcoeffs(self, fogi_components):
        return self.errorgen_vec_to_opcoeffs(self.fogi_components_array_to_errorgen_vec(fogi_components))

    def fogv_components_array_to_opcoeffs(self, fogv_components):
        return self.errorgen_vec_to_opcoeffs(self.fogv_components_array_to_errorgen_vec(fogv_components))

    def fogiv_components_array_to_opcoeffs(self, fogi_components, fogv_components):
        return self.errorgen_vec_to_opcoeffs(self.fogiv_components_array_to_errorgen_vec(
            fogi_components, fogv_components))

    def create_binned_fogi_infos(self, tol=1e-5):
        """
        Creates an 'info' dictionary for each FOGI quantity and places it within a
        nested dictionary structure by the operators involved, the types of error generators,
        and the qubits acted upon (a.k.a. the "target" qubits).
        TODO: docstring

        Returns
        -------
        dict
        """

        # Construct a dict of information for each elementary error-gen basis element (the basis for error-gen space)
        elemgen_info = {}
        for k, (op_label, eglabel) in enumerate(self.errgen_space_op_elem_labels):
            elemgen_info[k] = {
                'type': eglabel[0],
                'qubits': set([i for bel_lbl in eglabel[1:] for i, char in enumerate(bel_lbl) if char != 'I']),
                'op_label': op_label,
                'elemgen_label': eglabel,
            }

        bins = {}
        dependent_indices = set(self.dependent_dir_indices)  # indices of one set of linearly dep. fogi dirs
        for i in range(self.num_fogi_directions):
            fogi_dir = self.fogi_directions[:, i]
            label = self.fogi_labels[i]
            label_raw = self.raw_fogi_labels[i]
            label_abbrev = self.abbrev_fogi_labels[i]
            gauge_dir = self.fogi_gaugespace_directions[i]
            r_factor = self.fogi_r_factors[i]

            present_elgen_indices = _np.where(_np.abs(fogi_dir) > tol)[0]

            #Aggregate elemgen_info data for all elemgens that contribute to this FOGI qty (as determined by `tol`)
            ops_involved = set(); qubits_acted_upon = set(); types = set(); basismx = None
            for k in present_elgen_indices:
                k_info = elemgen_info[k]
                ops_involved.add(k_info['op_label'])
                qubits_acted_upon.update(k_info['qubits'])
                types.add(k_info['type'])

            #Create the "info" dictionary for this FOGI quantity
            info = {'op_set': ops_involved,
                    'types': types,
                    'qubits': qubits_acted_upon,
                    'fogi_index': i,
                    'label': label,
                    'label_raw': label_raw,
                    'label_abbrev': label_abbrev,
                    'dependent': bool(i in dependent_indices),
                    'gauge_dir': gauge_dir,
                    'fogi_dir': fogi_dir,
                    'r_factor': r_factor
                    }
            ops_involved = tuple(sorted(ops_involved))
            types = tuple(sorted(types))
            qubits_acted_upon = tuple(sorted(qubits_acted_upon))
            if ops_involved not in bins: bins[ops_involved] = {}
            if types not in bins[ops_involved]: bins[ops_involved][types] = {}
            if qubits_acted_upon not in bins[ops_involved][types]: bins[ops_involved][types][qubits_acted_upon] = []
            bins[ops_involved][types][qubits_acted_upon].append(info)

        return bins

    @classmethod
    def merge_binned_fogi_infos(cls, binned_fogi_infos, index_offsets):
        """
        Merge together multiple FOGI-info dictionaries created by :method:`create_binned_fogi_infos`.

        Parameters
        ----------
        binned_fogi_infos : list
            A list of FOGI-info dictionaries.

        index_offsets : list
            A list of length `len(binned_fogi_infos)` that gives the offset
            into an assumed-to-exist corresponding vector of components for
            all the FOGI infos.

        Returns
        -------
        dict
            The merged dictionary
        """
        def _merge_into(dest, src, offset, nlevels_to_merge, store_index):
            if nlevels_to_merge == 0:  # special last-level case where src and dest are *lists*
                for info in src:
                    new_info = _copy.deepcopy(info)
                    new_info['fogi_index'] += offset
                    new_info['store_index'] = store_index
                    dest.append(new_info)
            else:
                for k, d in src.items():
                    if k not in dest: dest[k] = {} if (nlevels_to_merge > 1) else []  # last level = list
                    _merge_into(dest[k], d, offset, nlevels_to_merge - 1, store_index)

        bins = {}
        nLevels = 3 # ops_involved, types, qubits_acted_upon
        for i, (sub_bins, offset) in enumerate(zip(binned_fogi_infos, index_offsets)):
            _merge_into(bins, sub_bins, offset, nLevels, i)
        return bins
