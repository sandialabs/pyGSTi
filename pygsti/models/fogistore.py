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
import scipy.sparse as _sps
import copy as _copy
import warnings as _warnings
import itertools as _itertools
import collections as _collections
from pygsti.baseobjs import Basis as _Basis
from pygsti.tools import matrixtools as _mt
from pygsti.tools import optools as _ot
from pygsti.tools import fogitools as _fogit


class FirstOrderGaugeInvariantStore(object):
    """
    An object that computes and stores the first-order-gauge-invariant quantities of a model.

    Currently, it is only compatible with :class:`ExplicitOpModel` objects.
    """

    def __init__(self, gauge_action_matrices_by_op, gauge_action_gauge_spaces_by_op, errorgen_coefficient_labels_by_op,
                 op_label_abbrevs=None, reduce_to_model_space=True,
                 dependent_fogi_action='drop', norm_order=None):
        """
        TODO: docstring
        """

        self.primitive_op_labels = tuple(gauge_action_matrices_by_op.keys())

        # Construct common gauge space by special way of intersecting the gauge spaces for all the ops
        # Note: the gauge_space of each op is constructed (see `setup_fogi`) so that the gauge action is
        #  zero on any elementary error generator not in the elementary-errorgen basis associated with
        #  gauge_space (gauge_space is the span of linear combos of a elemgen basis that was chosen to
        #  include all possible non-trivial (non-zero) gauge actions by the operator (gauge action is the
        #  *difference* K - UKU^dag in the Lindblad mapping under gauge transform exp(K),  L -> L + K - UKU^dag)
        common_gauge_space = None
        for op_label, gauge_space in gauge_action_gauge_spaces_by_op.items():
            #FOGI DEBUG print("DEBUG gauge space of ", op_label, "has dim", gauge_space.vectors.shape[1])
            if common_gauge_space is None:
                common_gauge_space = gauge_space
            else:
                common_gauge_space = common_gauge_space.intersection(gauge_space,
                                                                     free_on_unspecified_space=True)

        # column space of self.fogi_directions
        #FOGI DEBUG print("DEBUG common gauge space of has dim", common_gauge_space.vectors.shape[1])
        common_gauge_space.normalize()
        self.gauge_space = common_gauge_space

        # row space of self.fogi_directions - "errgen-set space" lookups
        # -- maybe make this into an "ErrgenSetSpace" object in FUTURE?
        self.elem_errorgen_labels_by_op = errorgen_coefficient_labels_by_op

        self.op_errorgen_indices = _fogit._create_op_errgen_indices_dict(self.primitive_op_labels,
                                                                         self.elem_errorgen_labels_by_op)
        self.errorgen_space_op_elem_labels = tuple([(op_label, elem_lbl) for op_label in self.primitive_op_labels
                                                    for elem_lbl in self.elem_errorgen_labels_by_op[op_label]])
        # above is same as flattened self.elem_errorgen_labels_by_op - labels final "row basis" of fogi dirs

        num_elem_errgens = sum([len(labels) for labels in self.elem_errorgen_labels_by_op.values()])
        allop_gauge_action = _sps.lil_matrix((num_elem_errgens, self.gauge_space.vectors.shape[1]), dtype=complex)

        # Now update (restrict) each op's gauge_action to use common gauge space
        # - need to write the vectors of the common (final) gauge space, w_i, as linear combos of
        #   the op's original gauge space vectors, v_i.
        # - ignore elemgens that are not in the op's orig_gauge_space's elemgen basis
        # W = V * alpha, and we want to find alpha.  W and V are the vectors in the op's elemgen basis
        #  (these could be seen as staring in the union of the common gauge's and op's elemgen
        #   bases - which would just be the common gauge's elemgen basis since it's strictly larger -
        #   restricted to the op's elemben basis)
        for op_label, orig_gauge_space in gauge_action_gauge_spaces_by_op.items():
            #FOGI DEBUG print("DEBUG: ", op_label, orig_gauge_space.vectors.shape, len(orig_gauge_space.elemgen_basis))
            gauge_action = gauge_action_matrices_by_op[op_label]  # a sparse matrix

            op_elemgen_lbls = orig_gauge_space.elemgen_basis.labels
            W = common_gauge_space.vectors[common_gauge_space.elemgen_basis.label_indices(op_elemgen_lbls), :]
            V = orig_gauge_space.vectors
            alpha = _np.dot(_np.linalg.pinv(V), W)  # make SPARSE compatible in future if space vectors are sparse
            alpha = _sps.csr_matrix(alpha)  # convert to dense -> CSR for now, as if we did sparse math above

            # update gauge action to use common gauge space
            sparse_gauge_action = gauge_action.dot(alpha)
            allop_gauge_action[self.op_errorgen_indices[op_label], :] = sparse_gauge_action[:, :]
            gauge_action_matrices_by_op[op_label] = sparse_gauge_action.toarray()  # make **DENSE** here
            # Hopefully matrices aren't too large after this reduction and dense matrices are ok,
            # otherwise we need to change downstream nullspace and pseduoinverse operations to be sparse compatible.

            #FUTURE: if update above creates zero-rows in gauge action matrix, maybe remove
            # these rows from the row basis, i.e. self.elem_errorgen_labels_by_op[op_label]

        self.gauge_action_for_op = gauge_action_matrices_by_op

        (indep_fogi_directions, indep_fogi_metadata, dep_fogi_directions, dep_fogi_metadata) = \
            _fogit.construct_fogi_quantities(self.primitive_op_labels, self.gauge_action_for_op,
                                             self.elem_errorgen_labels_by_op, self.op_errorgen_indices,
                                             self.gauge_space, op_label_abbrevs, dependent_fogi_action, norm_order)
        self.fogi_directions = _sps.hstack((indep_fogi_directions, dep_fogi_directions))
        self.fogi_metadata = indep_fogi_metadata + dep_fogi_metadata  # list concatenation
        self.dependent_dir_indices = _np.arange(len(indep_fogi_metadata), len(self.fogi_metadata))
        for j, meta in enumerate(self.fogi_metadata):
            meta['raw'] = _fogit.op_elem_vec_name(self.fogi_directions[:, j], self.errorgen_space_op_elem_labels,
                                                  op_label_abbrevs if (op_label_abbrevs is not None) else {})

        assert(len(self.errorgen_space_op_elem_labels) == self.fogi_directions.shape[0])

        # Note: currently PUNT on doing below with sparse math, as a sparse nullspace routine is unavailable

        #First order gauge *variant* directions (the complement of FOGI directions in errgen set space)
        # (directions in errorgen space that correspond to gauge transformations -- to first order)
        #fogv_directions = _mt.nice_nullspace(self.fogi_directions.T)  # can be dependent!
        self.fogv_directions = _mt.nullspace(self.fogi_directions.toarray().T)  # complement of fogi directions
        self.fogv_directions = _sps.csc_matrix(self.fogv_directions)  # as though we used sparse math above
        self.fogv_labels = ["%d gauge action" % i for i in range(self.fogv_directions.shape[1])]
        #self.fogv_labels = ["%s gauge action" % nm
        #                    for nm in _fogit.elem_vec_names(gauge_space_directions, gauge_elemgen_labels)]

        #Get gauge-space directions corresponding to the fogv directions
        # (pinv_allop_gauge_action takes errorgen-set -> gauge-gen space)
        self.allop_gauge_action = allop_gauge_action
        pinv_allop_gauge_action = _np.linalg.pinv(self.allop_gauge_action.toarray(), rcond=1e-7)
        gauge_space_directions = _np.dot(pinv_allop_gauge_action, self.fogv_directions)  # in gauge-generator space
        self.gauge_space_directions = gauge_space_directions

        #Notes on error-gen vs gauge-gen space:
        # self.fogi_directions and self.fogv_directions are dual vectors in error-generator space,
        # i.e. with elements corresponding to the elementary error generators given by
        # self.errorgen_space_op_elem_labels.
        # self.fogi_gaugespace_directions contains, when applicable, a gauge-space direction that
        # correspondings to the FOGI quantity in self.fogi_directions.  Such a gauge-space direction
        # exists for relational FOGI quantities, where the FOGI quantity is constructed by taking the
        # *difference* of a gauge-space action (given by the gauge-space direction) on two operations
        # (or sets of operations).

        #Store auxiliary info for later use
        self.norm_order = norm_order
        self._dependent_fogi_action = dependent_fogi_action

        #Assertions to check that everything looks good
        if True:
            fogi_dirs = self.fogi_directions.toarray()  # don't bother with sparse math yet
            fogv_dirs = self.fogv_directions.toarray()

            # We must reduce X_gauge_action to the "in-model gauge space" before testing if the computed vecs are FOGI:
            assert(_np.linalg.norm(_np.dot(self.allop_gauge_action.toarray().T, fogi_dirs)) < 1e-8)

            #Check that pseudo-inverse was computed correctly (~ matrices are full rank)
            # fogi_coeffs = dot(fogi_directions.T, elem_errorgen_vec), where elem_errorgen_vec is filled from model
            #                 params, since fogi_directions columns are *dual* vectors in error-gen space.  Thus,
            #                 to go in reverse:
            # elem_errogen_vec = dot(pinv_fogi_dirs_T, fogi_coeffs), where dot(fogi_directions.T, pinv_fogi_dirs_T) == I
            # (This will only be the case when fogi_vecs are linearly independent, so when dependent_indices == 'drop')

            if dependent_fogi_action == 'drop':
                #assert(_mt.columns_are_orthogonal(self.fogi_directions))  # not true unless we construct them so...
                assert(_np.linalg.norm(_np.dot(fogi_dirs.T, _np.linalg.pinv(fogi_dirs.T))
                                       - _np.identity(fogi_dirs.shape[1], 'd')) < 1e-6)

            # A similar relationship should always hold for the gauge directions, except for these we never
            #  keep linear dependencies
            assert(_mt.columns_are_orthogonal(fogv_dirs))
            assert(_np.linalg.norm(_np.dot(fogv_dirs.T, _np.linalg.pinv(fogv_dirs.T))
                                   - _np.identity(fogv_dirs.shape[1], 'd')) < 1e-6)

    def find_nice_fogiv_directions(self):
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
        pass

    @property
    def errorgen_space_dim(self):
        return self.fogi_directions.shape[0]

    @property
    def gauge_space_dim(self):
        return self.gauge_space.vectors.shape[1]

    @property
    def num_fogi_directions(self):
        return self.fogi_directions.shape[1]

    @property
    def num_fogv_directions(self):
        return self.fogv_directions.shape[1]

    def fogi_errorgen_direction_labels(self, typ='normal'):
        """ typ can be 'raw' or 'abbrev' too """
        if typ == 'normal': return tuple([meta['name'] for meta in self.fogi_metadata])
        elif typ == 'raw': return tuple([meta['raw'] for meta in self.fogi_metadata])
        elif typ == 'abrev': return tuple([meta['abbrev'] for meta in self.fogi_metadata])
        else: raise ValueError("Invalid `typ` argument: %s" % str(typ))

    def fogv_errorgen_direction_labels(self, typ='normal'):
        if typ == 'normal': labels = self.fogv_labels
        else: labels = [''] * len(self.fogv_labels)
        return tuple(labels)

    def errorgen_vec_to_fogi_components_array(self, errorgen_vec):
        fogi_coeffs = self.fogi_directions.transpose().dot(errorgen_vec)
        assert(_np.linalg.norm(fogi_coeffs.imag) < 1e-8)
        return fogi_coeffs.real

    def errorgen_vec_to_fogv_components_array(self, errorgen_vec):
        fogv_coeffs = self.fogv_directions.transpose().dot(errorgen_vec)
        assert(_np.linalg.norm(fogv_coeffs.imag) < 1e-8)
        return fogv_coeffs.real

    def opcoeffs_to_fogi_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errorgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogi_components_array(errorgen_vec)

    def opcoeffs_to_fogv_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errorgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogv_components_array(errorgen_vec)

    def opcoeffs_to_fogiv_components_array(self, op_coeffs):
        errorgen_vec = _np.zeros(self.errorgen_space_dim, 'd')
        for i, (op_label, elem_lbl) in enumerate(self.errorgen_space_op_elem_labels):
            errorgen_vec[i] += op_coeffs[op_label].get(elem_lbl, 0.0)
        return self.errorgen_vec_to_fogi_components_array(errorgen_vec), \
            self.errorgen_vec_to_fogv_components_array(errorgen_vec)

    def fogi_components_array_to_errorgen_vec(self, fogi_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogi components to an errorgen-set vec when fogi directions are linearly-dependent!"
             "  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        # DENSE - need to use sparse solve to enact sparse pinv on vector TODO
        return _np.dot(_np.linalg.pinv(self.fogi_directions.toarray().T, rcond=1e-7), fogi_components)

    def fogv_components_array_to_errorgen_vec(self, fogv_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogv components to an errorgen-set vec when fogi directions are linearly-dependent!"
             "  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        # DENSE - need to use sparse solve to enact sparse pinv on vector TODO
        return _np.dot(_np.linalg.pinv(self.fogv_directions.toarray().T, rcond=1e-7), fogv_components)

    def fogiv_components_array_to_errorgen_vec(self, fogi_components, fogv_components):
        assert(self._dependent_fogi_action == 'drop'), \
            ("Cannot convert *from* fogiv components to an errorgen-set vec when fogi directions are "
             "linearly-dependent!  (Set `dependent_fogi_action='drop'` to ensure directions are independent.)")
        # DENSE - need to use sparse solve to enact sparse pinv on vector TODO
        return _np.dot(_np.linalg.pinv(
            _np.concatenate((self.fogi_directions.toarray(), self.fogv_directions.toarray()), axis=1).T,
            rcond=1e-7), _np.concatenate((fogi_components, fogv_components)))

    def errorgen_vec_to_opcoeffs(self, errorgen_vec):
        op_coeffs = {op_label: {} for op_label in self.primitive_op_labels}
        for (op_label, elem_lbl), coeff_value in zip(self.errorgen_space_op_elem_labels, errorgen_vec):
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
        for k, (op_label, eglabel) in enumerate(self.errorgen_space_op_elem_labels):
            elemgen_info[k] = {
                'type': eglabel[0],
                'qubits': set([i for bel_lbl in eglabel[1:] for i, char in enumerate(bel_lbl) if char != 'I']),
                'op_label': op_label,
                'elemgen_label': eglabel,
            }

        bins = {}
        dependent_indices = set(self.dependent_dir_indices)  # indices of one set of linearly dep. fogi dirs
        for i, meta in enumerate(self.fogi_metadata):
            fogi_dir = self.fogi_directions[:, i].toarray()
            label = meta['name']
            label_raw = meta['raw']
            label_abbrev = meta['abbrev']
            gauge_dir = meta['gaugespace_dir']
            r_factor = meta['r']

            present_elgen_indices = _np.where(_np.abs(fogi_dir) > tol)[0]

            #Aggregate elemgen_info data for all elemgens that contribute to this FOGI qty (as determined by `tol`)
            ops_involved = set(); qubits_acted_upon = set(); types = set()  # basismx = None
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
        nLevels = 3  # ops_involved, types, qubits_acted_upon
        for i, (sub_bins, offset) in enumerate(zip(binned_fogi_infos, index_offsets)):
            _merge_into(bins, sub_bins, offset, nLevels, i)
        return bins
