"""
Utility functions for computing and working with first-order-gauge-invariant (FOGI) quantities.
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
import scipy.sparse.linalg as _spsl

from . import matrixtools as _mt
from . import optools as _ot


def first_order_gauge_action_matrix(clifford_superop_mx, target_sslbls, model_state_space,
                                    elemgen_gauge_basis, elemgen_row_basis):
    """
    Returns a matrix for computing the *offset* of a given gate's error generator due to a local gauge action.
    Note: clifford_superop must be in the *std* basis!
    TODO: docstring
    """

    #Utilize EmbeddedOp to perform superop embedding
    from pygsti.modelmembers.operations import EmbeddedOp as _EmbeddedOp, StaticArbitraryOp as _StaticArbitraryOp
    from pygsti.baseobjs.errorgenbasis import ExplicitElementaryErrorgenBasis as _ExplicitElementaryErrorgenBasis

    def _embed(mx, target_labels, state_space):
        if mx.shape[0] == state_space.dim:
            return mx  # no embedding needed
        else:
            dummy_op = _EmbeddedOp(state_space, target_labels,
                                   _StaticArbitraryOp(_np.identity(mx.shape[0], 'd'), 'densitymx_slow'))
            embeddedOp = _sps.identity(state_space.dim, mx.dtype, format='lil')
            scale = _np.sqrt(4**len(target_labels) / state_space.dim)  # is this correct??

            #fill in embedded_op contributions (always overwrites the diagonal
            # of finalOp where appropriate, so OK it starts as identity)
            for i, j, gi, gj in dummy_op._iter_matrix_elements('HilbertSchmidt'):
                embeddedOp[i, j] = mx[gi, gj] * scale
            #return embeddedOp.tocsr()  # Note: if sparse, assure that this is in CSR or CSC for fast products
            return embeddedOp.toarray()

    action_mx = _sps.lil_matrix((len(elemgen_row_basis), len(elemgen_gauge_basis)), dtype=clifford_superop_mx.dtype)
    nonzero_rows = set()
    nonzero_row_labels = {}
    TOL = 1e-12  # ~ machine precision

    for j, (gen_sslbls, gen) in enumerate(elemgen_gauge_basis.elemgen_supports_and_matrices):
        action_sslbls = tuple(sorted(set(gen_sslbls).union(target_sslbls)))  # (union) - joint support of ops
        action_space = model_state_space.create_subspace(action_sslbls)

        gen_expanded = _embed(gen, gen_sslbls, action_space)  # expand gen to shared action_space
        U_expanded = _embed(clifford_superop_mx, target_sslbls, action_space)  # expand to shared action_space
        if _sps.issparse(gen_expanded):
            conjugated_gen = U_expanded.dot(gen_expanded.dot(U_expanded.transpose().conjugate()))  # sparse matrices
        else:
            conjugated_gen = _np.dot(U_expanded, _np.dot(gen_expanded, _np.conjugate(U_expanded.T)))
        gauge_action_deriv = gen_expanded - conjugated_gen  # (on action_space)

        action_row_basis = elemgen_row_basis.create_subbasis(action_sslbls)  # spans all *possible* error generators
        # - a full basis for gauge_action_deriv
        #global_row_space.add_labels(row_space.labels)  # labels would need to contain sslbls too
        action_row_labels = action_row_basis.labels
        global_row_indices = elemgen_row_basis.label_indices(action_row_labels)

        # Note: can avoid this projection and conjugation math above if we know gen is Pauli action and U is clifford
        for i, row_label, (gen2_sslbls, gen2) in zip(global_row_indices, action_row_labels,
                                                     action_row_basis.elemgen_supports_and_matrices):
            #if not is_subset(gen2_sslbls, space):
            #    continue  # no overlap/component when gen2 is nontrivial (and assumed orthogonal to identity)
            #              # on a factor space where gauge_action_deriv is zero

            gen2_expanded = _embed(gen2, gen2_sslbls, action_space)  # embed gen2 into action_space
            if _sps.issparse(gen2_expanded):
                flat_gen2_expanded_conj = gen2_expanded.reshape((1, _np.product(gen2_expanded.shape))).conjugate()
                flat_gauge_action_deriv = gauge_action_deriv.reshape((_np.product(gauge_action_deriv.shape), 1))
                val = flat_gen2_expanded_conj.dot(flat_gauge_action_deriv)[0, 0]
            else:
                val = _np.vdot(gen2_expanded.flat, gauge_action_deriv.flat)

            assert(abs(val.imag) < TOL)  # all values should be real, I think...
            if abs(val) > TOL:
                if i not in nonzero_rows:
                    nonzero_rows.add(i)
                    nonzero_row_labels[i] = row_label
                action_mx[i, j] = val

        #TODO HERE: check that decomposition into components adds to entire gauge_action_deriv
        #  (checks "completeness" of row basis)

    #return action_mx

    #Remove all all-zero rows and cull these elements out of the row_basis.  Actually,
    # just construct a new matrix and basis
    nonzero_row_indices = list(sorted(nonzero_rows))
    labels = [nonzero_row_labels[i] for i in nonzero_row_indices]

    data = []; col_indices = []; rowptr = [0]  # build up a CSR matrix manually from nonzero rows
    for ii, i in enumerate(nonzero_row_indices):
        col_indices.extend(action_mx.rows[i])
        data.extend(action_mx.data[i])
        rowptr.append(len(data))
    culled_action_mx = _sps.csr_matrix((data, col_indices, rowptr),
                                       shape=(len(nonzero_rows), len(elemgen_gauge_basis)), dtype=action_mx.dtype)
    updated_row_basis = _ExplicitElementaryErrorgenBasis(elemgen_row_basis.state_space, labels)

    return culled_action_mx, updated_row_basis


def first_order_gauge_action_matrix_for_prep(prep_superket_vec, target_sslbls, model_state_space,
                                             elemgen_gauge_basis, elemgen_row_basis):
    """
    Returns a matrix for computing the *offset* of a given gate's error generator due to a local gauge action.
    Note: clifford_superop must be in the *std* basis!
    TODO: docstring
    """

    #Utilize EmbeddedOp to perform superop embedding
    from pygsti.modelmembers.operations import EmbeddedOp as _EmbeddedOp, StaticArbitraryOp as _StaticArbitraryOp
    from pygsti.baseobjs.errorgenbasis import ExplicitElementaryErrorgenBasis as _ExplicitElementaryErrorgenBasis

    def _embed(mx, target_labels, state_space):  # SAME as in fn above
        if mx.shape[0] == state_space.dim:
            return mx  # no embedding needed
        else:
            dummy_op = _EmbeddedOp(state_space, target_labels,
                                   _StaticArbitraryOp(_np.identity(mx.shape[0], 'd'), 'densitymx_slow'))
            embeddedOp = _sps.identity(state_space.dim, mx.dtype, format='lil')
            scale = _np.sqrt(4**len(target_labels) / state_space.dim)  # is this correct??

            #fill in embedded_op contributions (always overwrites the diagonal
            # of finalOp where appropriate, so OK it starts as identity)
            for i, j, gi, gj in dummy_op._iter_matrix_elements('HilbertSchmidt'):
                embeddedOp[i, j] = mx[gi, gj] * scale
            #return embeddedOp.tocsr()  # Note: if sparse, assure that this is in CSR or CSC for fast products
            return embeddedOp.toarray()

    element_action_mx = _sps.lil_matrix((prep_superket_vec.shape[0], len(elemgen_gauge_basis)),
                                        dtype=prep_superket_vec.dtype)
    for j, (gen_sslbls, gen) in enumerate(elemgen_gauge_basis.elemgen_supports_and_matrices):
        action_sslbls = tuple(sorted(set(gen_sslbls).union(target_sslbls)))  # (union) - joint support of ops
        action_space = model_state_space.create_subspace(action_sslbls)
        #Note: action_space is always (?) going to be the full model_state_space, since target_sslbls for a prep
        # should always be all the sslbls of the model (I think...)

        gen_expanded = _embed(gen, gen_sslbls, action_space)  # expand gen to shared action_space
        if _sps.issparse(gen_expanded):
            gauge_action_deriv = gen_expanded.dot(prep_superket_vec)  # sparse matrices
        else:
            gauge_action_deriv = _np.dot(gen_expanded, prep_superket_vec)
        element_action_mx[:, j] = gauge_action_deriv[:, None]

    #To identify set of vectors {v_i} such that {element_action_mx * v_i} span the range of element_action_mx,
    # we find the SVD of element_action_mx and use the columns of V:
    TOL = 1e-7
    U, s, Vh = _np.linalg.svd(element_action_mx.toarray(), full_matrices=False)  # DENSE - use sparse SVD here?
    n = _np.count_nonzero(s > TOL)
    relevant_basis = Vh[0:n, :].T.conjugate()

    for j in range(relevant_basis.shape[1]):  # normalize columns so largest element is +1.0
        i_max = _np.argmax(_np.abs(relevant_basis[:, j]))
        if abs(relevant_basis[i_max, j]) > 1e-6:
            relevant_basis[:, j] /= relevant_basis[i_max, j]
    relevant_basis = _mt.normalize_columns(relevant_basis)

    # "gauge action" matrix is just the identity on the *relevant* space of gauge transformations:
    action_mx_pre = _np.dot(relevant_basis, relevant_basis.T.conjugate())  # row basis == elemgen_gauge_basis
    action_mx = _sps.lil_matrix((len(elemgen_row_basis), len(elemgen_gauge_basis)),
                                dtype=prep_superket_vec.dtype)
    nonzero_rows = set()
    nonzero_row_labels = {}

    #Convert row-space to be over elemgen_row_basis instead of a elemgen_gauge_basis
    for i, glbl in enumerate(elemgen_gauge_basis.labels):
        new_i = elemgen_row_basis.label_index(glbl)
        if _np.linalg.norm(action_mx_pre[i, :]) > 1e-8:
            action_mx[new_i, :] = action_mx_pre[i, :]
            nonzero_rows.add(i)
            nonzero_row_labels[i] = glbl

    #Remove all all-zero rows and cull these elements out of the row_basis.  Actually,
    # just construct a new matrix and basis
    nonzero_row_indices = list(sorted(nonzero_rows))
    labels = [nonzero_row_labels[i] for i in nonzero_row_indices]

    data = []; col_indices = []; rowptr = [0]  # build up a CSR matrix manually from nonzero rows
    for ii, i in enumerate(nonzero_row_indices):
        col_indices.extend(action_mx.rows[i])
        data.extend(action_mx.data[i])
        rowptr.append(len(data))
    culled_action_mx = _sps.csr_matrix((data, col_indices, rowptr),
                                       shape=(len(nonzero_rows), len(elemgen_gauge_basis)), dtype=action_mx.dtype)
    updated_row_basis = _ExplicitElementaryErrorgenBasis(elemgen_row_basis.state_space, labels)

    return culled_action_mx, updated_row_basis


def first_order_gauge_action_matrix_for_povm(povm_superbra_vecs, target_sslbls, model_state_space,
                                             elemgen_gauge_basis, elemgen_row_basis):
    """
    Returns a matrix for computing the *offset* of a given gate's error generator due to a local gauge action.
    Note: clifford_superop must be in the *std* basis!
    TODO: docstring
    """

    #Utilize EmbeddedOp to perform superop embedding
    from pygsti.modelmembers.operations import EmbeddedOp as _EmbeddedOp, StaticArbitraryOp as _StaticArbitraryOp
    from pygsti.baseobjs.errorgenbasis import ExplicitElementaryErrorgenBasis as _ExplicitElementaryErrorgenBasis

    def _embed(mx, target_labels, state_space):  # SAME as in fn above
        if mx.shape[0] == state_space.dim:
            return mx  # no embedding needed
        else:
            dummy_op = _EmbeddedOp(state_space, target_labels,
                                   _StaticArbitraryOp(_np.identity(mx.shape[0], 'd'), 'densitymx_slow'))
            embeddedOp = _sps.identity(state_space.dim, mx.dtype, format='lil')
            scale = _np.sqrt(4**len(target_labels) / state_space.dim)  # is this correct??

            #fill in embedded_op contributions (always overwrites the diagonal
            # of finalOp where appropriate, so OK it starts as identity)
            for i, j, gi, gj in dummy_op._iter_matrix_elements('HilbertSchmidt'):
                embeddedOp[i, j] = mx[gi, gj] * scale
            #return embeddedOp.tocsr()  # Note: if sparse, assure that this is in CSR or CSC for fast products
            return embeddedOp.toarray()

    element_action_mx = _sps.lil_matrix((sum([v.shape[0] for v in povm_superbra_vecs]), len(elemgen_gauge_basis)),
                                        dtype=povm_superbra_vecs[0].dtype)
    for j, (gen_sslbls, gen) in enumerate(elemgen_gauge_basis.elemgen_supports_and_matrices):
        action_sslbls = tuple(sorted(set(gen_sslbls).union(target_sslbls)))  # (union) - joint support of ops
        action_space = model_state_space.create_subspace(action_sslbls)
        #Note: action_space is always (?) going to be the full model_state_space, since target_sslbls for a prep
        # should always be all the sslbls of the model (I think...)

        #Currently, this applies same vector to *all* povm effects - i.e. treats as a ComposedPOVM
        # Note: gauge acts on effects as: dot(v, gen_expanded.T.conj) = dot(gen_expanded.conj, v)
        gen_expanded = _embed(gen, gen_sslbls, action_space)  # expand gen to shared action_space
        if _sps.issparse(gen_expanded):
            gauge_action_deriv = _sps.vstack([gen_expanded.conjugate().dot(v) for v in povm_superbra_vecs])
        else:
            gauge_action_deriv = _np.concatenate([_np.dot(gen_expanded.conjugate(), v) for v in povm_superbra_vecs])
        element_action_mx[:, j] = gauge_action_deriv[:, None]

    #FROM HERE DOWN same as for prep vector (concat effects treated like one big prep vector)

    #To identify set of vectors {v_i} such that {element_action_mx * v_i} span the range of element_action_mx,
    # we find the SVD of element_action_mx and use the columns of V:
    TOL = 1e-7
    U, s, Vh = _np.linalg.svd(element_action_mx.toarray(), full_matrices=False)  # DENSE - use sparse SVD here?
    n = _np.count_nonzero(s > TOL)
    relevant_basis = Vh[0:n, :].T.conjugate()

    for j in range(relevant_basis.shape[1]):  # normalize columns so largest element is +1.0
        i_max = _np.argmax(_np.abs(relevant_basis[:, j]))
        if abs(relevant_basis[i_max, j]) > 1e-6:
            relevant_basis[:, j] /= relevant_basis[i_max, j]
    relevant_basis = _mt.normalize_columns(relevant_basis)

    # "gauge action" matrix is just the identity on the *relevant* space of gauge transformations:
    action_mx_pre = _np.dot(relevant_basis, relevant_basis.T.conjugate())  # row basis == elemgen_gauge_basis
    action_mx = _sps.lil_matrix((len(elemgen_row_basis), len(elemgen_gauge_basis)),
                                dtype=povm_superbra_vecs[0].dtype)
    nonzero_rows = set()
    nonzero_row_labels = {}

    #Convert row-space to be over elemgen_row_basis instead of a elemgen_gauge_basis
    for i, glbl in enumerate(elemgen_gauge_basis.labels):
        new_i = elemgen_row_basis.label_index(glbl)
        if _np.linalg.norm(action_mx_pre[i, :]) > 1e-8:
            action_mx[new_i, :] = action_mx_pre[i, :]
            nonzero_rows.add(i)
            nonzero_row_labels[i] = glbl

    #Remove all all-zero rows and cull these elements out of the row_basis.  Actually,
    # just construct a new matrix and basis
    nonzero_row_indices = list(sorted(nonzero_rows))
    labels = [nonzero_row_labels[i] for i in nonzero_row_indices]

    data = []; col_indices = []; rowptr = [0]  # build up a CSR matrix manually from nonzero rows
    for ii, i in enumerate(nonzero_row_indices):
        col_indices.extend(action_mx.rows[i])
        data.extend(action_mx.data[i])
        rowptr.append(len(data))
    culled_action_mx = _sps.csr_matrix((data, col_indices, rowptr),
                                       shape=(len(nonzero_rows), len(elemgen_gauge_basis)), dtype=action_mx.dtype)
    updated_row_basis = _ExplicitElementaryErrorgenBasis(elemgen_row_basis.state_space, labels)

    return culled_action_mx, updated_row_basis


def _create_op_errgen_indices_dict(primitive_op_labels, errorgen_coefficient_labels):
    op_errgen_indices = {}; off = 0  # tells us which indices of errgen-set space map to which ops
    for op_label in primitive_op_labels:
        num_coeffs = len(errorgen_coefficient_labels[op_label])
        op_errgen_indices[op_label] = slice(off, off + num_coeffs)
        off += num_coeffs
    return op_errgen_indices


def construct_fogi_quantities(primitive_op_labels, gauge_action_matrices,
                              errorgen_coefficient_labels, op_errgen_indices, gauge_space,
                              op_label_abbrevs=None, dependent_fogi_action='drop', norm_order=None):
    """ TODO: docstring """
    assert(dependent_fogi_action in ('drop', 'mark'))
    orthogonalize_relationals = True

    #Get lists of the present (existing within the model) labels for each operation
    if op_label_abbrevs is None: op_label_abbrevs = {}

    if op_errgen_indices is None:
        op_errgen_indices = _create_op_errgen_indices_dict(primitive_op_labels, errorgen_coefficient_labels)
    num_elem_errgens = sum([len(labels) for labels in errorgen_coefficient_labels.values()])

    #Step 1: construct FOGI quantities and reference frame for each op
    ccomms = {}

    fogi_dirs = _sps.csc_matrix((num_elem_errgens, 0), dtype=complex)  # dual vectors ("directions") in eg-set space
    fogi_meta = []  # elements correspond to matrix columns

    dep_fogi_dirs = _sps.csc_matrix((num_elem_errgens, 0), dtype=complex)  # dependent columns we still want to track
    dep_fogi_meta = []  # elements correspond to matrix columns

    def add_relational_fogi_dirs(dirs_to_add, gauge_vecs, gauge_dirs, initial_dirs, metadata,
                                 existing_opset, new_op_label, new_opset, norm_orders):
        """ Note: gauge_vecs and gauge_dirs are the same up to a normalization - maybe combine? """
        vecs_to_add, nrms = _mt.normalize_columns(dirs_to_add, ord=norm_orders, return_norms=True)  # f_hat_vec = f/nrm
        vector_L2_norm2s = _mt.column_norms(vecs_to_add)**2  # L2 norm squared
        vector_L2_norm2s[vector_L2_norm2s == 0.0] = 1.0  # avoid division of zero-column by zero
        dirs_to_add = _mt.scale_columns(vecs_to_add, 1 / vector_L2_norm2s)
        # above gives us *dir*-norm we want  # DUAL NORM
        # f_hat = f_hat_vec / L2^2 = f / (nrm * L2^2) = (1 / (nrm * L2^2)) * f

        resulting_dirs = _sps.hstack((initial_dirs, dirs_to_add))  # errgen-space NORMALIZED

        full_gauge_vecs = _np.dot(gauge_space.vectors, gauge_vecs)  # in gauge_space's basis
        gauge_names = elem_vec_names(full_gauge_vecs, gauge_space.elemgen_basis.labels)
        gauge_names_abbrev = elem_vec_names(full_gauge_vecs, gauge_space.elemgen_basis.labels, include_type=False)
        names = ["ga(%s)_%s - ga(%s)_%s" % (
            iname, "|".join([op_label_abbrevs.get(l, str(l)) for l in existing_opset]),
            iname, op_label_abbrevs.get(new_op_label, str(new_op_label))) for iname in gauge_names]
        abbrev_names = ["ga(%s)" % iname for iname in gauge_names_abbrev]

        for j, (name, name_abbrev, nrm, L2norm2) in enumerate(zip(names, abbrev_names, nrms, vector_L2_norm2s)):
            metadata.append({'name': name, 'abbrev': name_abbrev, 'r': 1 / (nrm * L2norm2),
                             'gaugespace_dir': gauge_dirs[:, j], 'opset': new_opset})
            # Note intersection_space is a subset of the *gauge-space*, and so its basis,
            # placed under gaugespace_dirs keys, is for gauge-space, not errorgen-space.

        return resulting_dirs

    def resolve_norm_order(vecs_to_normalize, label_lists, given_norm_order):
        """Turn user-supplied norm-order into an array of norm orders based, sometimes, on the vecs being normalized """
        if isinstance(given_norm_order, int):
            norm_order_array = _np.ones(local_fogi_dirs.shape[1], dtype=_np.int64) * given_norm_order
        elif given_norm_order == "auto":  # automaticaly choose norm order based on fogi direction composition
            lbl_lookup = {}; off = 0
            for label_list in label_lists:
                lbl_lookup.update({i + off: lbl for i, lbl in enumerate(label_list)})
                off += len(label_list)
            norm_order_array = []; TOL = 1e-8
            for j in range(vecs_to_normalize.shape[1]):
                lbl_types = set([lbl_lookup[i].errorgen_type for i, v in enumerate(vecs_to_normalize[:, j])
                                 if abs(v) > TOL])  # a set of the errorgen types contributing to the jth vec
                if lbl_types == set(['S']): norm_order_array.append(1)
                else: norm_order_array.append(2)
            norm_order_array = _np.array(norm_order_array, dtype=_np.int64)
        else:
            raise ValueError("Invalid norm_order: %s" % str(given_norm_order))
        return norm_order_array

    for op_label in primitive_op_labels:
        #FOGI DEBUG print("##", op_label)
        ga = gauge_action_matrices[op_label]
        # currently `ga` is a dense matrix, if SPARSE need to update nullspace and pinv math below

        #TODO: update this conditional to something more robust (same conditiona in explicitmodel.py too)
        if isinstance(op_label, str) and (op_label.startswith('rho') or op_label.startswith('M')):
            # Note: the "commutant" constructed in this way also includes irrelevant gauge directions,
            #  and for SPAM ops we know there are actually *no* local FOGI quantities and the entire
            #  "commutant" is irrelevant.  So below we perform similar math as for gates, but don't add
            #  any intrinsic FOGI quantities.  TODO - make this more general?
            commutant = _mt.nice_nullspace(ga)  # columns = *gauge* elem gen directions
            complement = _mt.nice_nullspace(commutant.T)  # complement of commutant - where op is faithful rep
            ccomms[(op_label,)] = complement
            #FOGI DEBUG print("  Skipping - SPAM, no intrinsic qtys")
            continue

        #Get commutant and communtant-complement spaces
        commutant = _mt.nice_nullspace(ga, orthogonalize=True)  # columns = *gauge* elem gen directions
        assert(_mt.columns_are_orthogonal(commutant))

        # Note: local/new_fogi_dirs are orthogonal but not necessarily normalized (so need to
        #  normalize to get inverse, but *don't* need pseudo-inverse).
        local_fogi_dirs = _mt.nice_nullspace(ga.T, orthogonalize=True)  # "conjugate space" to gauge action SPARSE?

        #NORMALIZE FOGI DIRS to have norm 1 - based on mapping between unit-variance
        # gaussian distribution of target-gateset perturbations in the usual errorgen-set-space
        # to the FOGI space.  The basis of the fogi directions is understood to be the basis
        # of errorgen-superops arising from *un-normalized* (traditional) Pauli matrices.
        ord_to_use = resolve_norm_order(local_fogi_dirs, [errorgen_coefficient_labels[op_label]], norm_order)
        local_fogi_vecs = _mt.normalize_columns(local_fogi_dirs, ord=ord_to_use)  # this gives us *vec*-norm we want
        vector_L2_norm2s = [_np.linalg.norm(local_fogi_vecs[:, j])**2 for j in range(local_fogi_vecs.shape[1])]
        local_fogi_dirs = local_fogi_vecs / _np.array(vector_L2_norm2s)[None, :]  # gives *dir*-norm we want # DUAL NORM
        #FOGI DEBUG print("  New intrinsic qtys = ", local_fogi_dirs.shape[1])
        #assert(_np.linalg.norm(local_fogi_dirs.imag) < 1e-6)  # ok for H+S but not for CPTP models

        assert(_mt.columns_are_orthogonal(local_fogi_dirs))  # Not for Cnot in 2Q_XYICNOT (check?)

        new_fogi_dirs = _sps.lil_matrix((num_elem_errgens, local_fogi_dirs.shape[1]), dtype=local_fogi_dirs.dtype)
        new_fogi_dirs[op_errgen_indices[op_label], :] = local_fogi_dirs  # "juice" this op
        fogi_dirs = _sps.hstack((fogi_dirs, new_fogi_dirs.tocsc()))

        #assert(_mt.columns_are_orthogonal(fogi_dirs))  # sparse version?

        #LABELS
        op_elemgen_labels = errorgen_coefficient_labels[op_label]
        errgen_names = elem_vec_names(local_fogi_vecs, op_elemgen_labels)
        errgen_names_abbrev = elem_vec_names(local_fogi_vecs, op_elemgen_labels, include_type=False)

        for egname, egname_abbrev in zip(errgen_names, errgen_names_abbrev):
            egname_with_op = "%s_%s" % ((("(%s)" % egname) if (' ' in egname) else egname),
                                        op_label_abbrevs.get(op_label, str(op_label)))
            fogi_meta.append({'name': egname_with_op, 'abbrev': egname_abbrev, 'r': 0,
                              'gaugespace_dir': None, 'opset': (op_label,)})

        complement = _mt.nice_nullspace(commutant.T,
                                        orthogonalize=True)  # complement of commutant - where op is faithful rep
        assert(_mt.columns_are_orthogonal(complement))
        ccomms[(op_label,)] = complement
        #gauge_action_for_op[op_label] = ga

        #print("Commutant:"); _mt.print_mx(commutant)
        #print("Names: ", errgen_names)
        #print("Complement:"); _mt.print_mx(complement)

    smaller_sets = [(op_label,) for op_label in primitive_op_labels]
    max_size = len(primitive_op_labels)
    for set_size in range(1, max_size):
        larger_sets = []
        num_indep_vecs_from_smaller_sets = fogi_dirs.shape[1]
        for op_label in primitive_op_labels:
            for existing_set in smaller_sets:
                if op_label in existing_set: continue
                new_set = tuple(sorted(existing_set + (op_label,)))
                if new_set in larger_sets: continue

                #FOGI DEBUG print("\n##", existing_set, "+", op_label)

                # Merge existing set + op_label => new set of larger size
                ccommA = ccomms.get(existing_set, None)  # Note: commutant-complements are in *gauge* space,
                ccommB = ccomms[(op_label,)]  # so they're all the same dimension.

                if ccommA is not None and ccommA.shape[1] > 0 and ccommB.shape[1] > 0:
                    # merging with an empty complement does nothing (no intersection, same ccomm)
                    intersection_space = _mt.intersection_space(ccommA, ccommB, use_nice_nullspace=True)
                    union_space = _mt.union_space(ccommA, ccommB)

                    #Don't orthogonalize these - instead orthogonalize fogi_dirs and find what these should be (below)
                    #intersection_space, _ = _np.linalg.qr(intersection_space)  # gram-schmidt orthogonalize cols
                    #assert(_mt.columns_are_orthogonal(intersection_space))  # Not always true

                    if intersection_space.shape[1] > 0:
                        #FOGI DEBUG print(" ==> intersection space dim = ", intersection_space.shape[1])
                        # Then each basis vector of the intersection space defines a gauge-invariant ("fogi")
                        # direction via the difference between that gauge direction's action on A and B:
                        gauge_action = _np.concatenate([gauge_action_matrices[ol] for ol in existing_set]
                                                       + [gauge_action_matrices[op_label]], axis=0)
                        n = sum([gauge_action_matrices[ol].shape[0] for ol in existing_set])  # boundary btwn A & B

                        # gauge trans: e => e + delta_e = e + dot(gauge_action, delta)
                        #   e = "errorgen-set space" vector; delta = "gauge space" vector
                        #   we want fogi_dir s.t. dot(fogi_dir.T, e) doesn't change when e transforms as above
                        # ==> we want fogi_dir s.t. dot(fogi_dir.T, gauge_action) == 0
                        #     (we want dot(fogi_dir.T, gauge_action, delta) == 0 for all delta)

                        # There exists subspace of errgen-set space = span({dot(gauge_action, delta) for all delta}).
                        # We call it "gauge-shift space" == gauge-orbit of target gateset, i.e. e=vec(0)  (within FOGI)
                        # We will construct {v_i} so each v_i is in the gauge-shift space or its orthogonal complement.
                        # We call components/coeffs of each v_i as FOGI or "gauge" based on where v_i lies, and
                        # call v_i a "FOGI direction" or "gauge direction"

                        # We find a set of fogi directions {f'_i} in the code that are potentially linearly dependent
                        # and non-orthogonal.  This is ok, as we can take the entire set and just define dot(f'_i.T, e)
                        # to be the *component* of e along f':
                        # => component_i := dot(f'_i.T, e)

                        # If we write e as as linear combination of fogi vectors:
                        # e = sum_i coeff_i * f_i   f_i are basis vecs, not nec. orthonormal, s.t. there exists
                        #  a dual basis f'_i s.t. dot(f'_i.T, f_i) = dirac_ij
                        # so if we take a subset of the {f'_i} that are linearly dependent we can define coefficients:
                        #  => coeff_i = dot(f'_i.T, e)  # for f' in a *basis* for the complement of gauge-shift space

                        #Note: construction method yields FOGI-space vectors - whether these vectors are "primal"
                        # or "dual" is a statement about *bases* for this FOGI space, i.e. a given basis has a dual
                        # basis.  The space is just the space - it's isomorphic to it's dual (it's a vector space).
                        # Observe:
                        # colspace(gauge_action) = gauge-shift space, and we can construct its orthogonal complement
                        # local_fogi_dir found by nullspace of gauge_action: dot(local_fogi.T, gauge_action) = 0
                        # Note: q in nullspace(gauge_action.T) => is q a valid fogi vector, i.e.
                        # dot(q.T, every_vec_in_gauge_shift_space) = 0 = dot(q.T, delta_e)
                        #                                              = dot(q.T, gauge_action, delta) for all delta
                        # So dot(gauge_action.T, q) = dot(q.T, gauge_action) = 0.  Thus

                        #Mathematically:
                        # let fogi_dir.T = int_vec.T * pinv(ga_A) - int_vec.T * pinv(ga_B) so that:
                        # dot(fogi_dir.T, gauge_action) = int_vec.T * (pinv(ga_A) - pinv(ga_B)) * gauge_action
                        #                               = (I - I) = 0
                        # (A,B are faithful reps of gauge on intersection space, so pinv(ga_A) * ga_A
                        # restricted to intersection space is I:   int_spc.T * pinv(ga_A) * ga_A * int_spc == I
                        # (when int_spc vecs are orthonormal) and so the above redues to I - I = 0

                        inv_diff_gauge_action = _np.concatenate((_np.linalg.pinv(gauge_action[0:n, :], rcond=1e-7),
                                                                 -_np.linalg.pinv(gauge_action[n:, :], rcond=1e-7)),
                                                                axis=1).T
                        #Equivalent:
                        #inv_diff_gauge_action = _np.concatenate((_np.linalg.pinv(gauge_action[0:n, :].T, rcond=1e-7),
                        #                                         -_np.linalg.pinv(gauge_action[n:, :].T, rcond=1e-7)),
                        #                                        axis=0)  # same as above, b/c T commutes w/pinv (?)

                        if orthogonalize_relationals:
                            # First, lets get a "good" basis for the intersection space - one that produces
                            # an orthogonal set of fogi directions.  Don't worry about normalization yet.
                            test_fogi_dirs = _np.dot(inv_diff_gauge_action, intersection_space)  # dot("M", epsilons)
                            Q, R = _np.linalg.qr(test_fogi_dirs)  # gram-schmidt orthogonalize cols
                            Q, R = _mt.sign_fix_qr(Q, R)  # removes sign ambiguity in QR decomp (simplifies comparisons)
                            # test_fogi_dirs = M * epsilons = Q * R
                            # => want orthogonalized dirs "Q" as new dirs: Q = M * epsilons * inv(R) = M * epsilons'
                            intersection_space = _np.dot(intersection_space, _np.linalg.inv(R))  # a "good" basis

                        # start w/normalizd epsilon vecs (normalize according to norm_order, then divide by L2-norm^2
                        # so that the resulting intersection-space vector, after action by "M", projects the component
                        # of the norm_order-normalized gauge-space vector)
                        ord_to_use = resolve_norm_order(intersection_space, [gauge_space.elemgen_basis.labels],
                                                        norm_order)
                        int_vecs = _mt.normalize_columns(intersection_space, ord=ord_to_use)
                        vector_L2_norm2s = [_np.linalg.norm(int_vecs[:, j])**2 for j in range(int_vecs.shape[1])]
                        intersection_space = int_vecs / _np.array(vector_L2_norm2s)[None, :]  # DUAL NORM

                        local_fogi_dirs = _np.dot(inv_diff_gauge_action, intersection_space)  # dot("M", epsilons)
                        #assert(_np.linalg.norm(local_fogi_dirs.imag) < 1e-6)  # ok for H+S but not for CPTP models
                        #Note: at this point `local_fogi_dirs` vectors are gauge-space-normalized, not numpy-norm-1
                        if orthogonalize_relationals:
                            assert(_mt.columns_are_orthogonal(local_fogi_dirs))  # true if we orthogonalize above

                        #NORMALIZATION:
                        # There are two normalizations relevant to relational fogi directions:
                        # 1) we normalize the would-be fogi vectors (those that would be prefixed by components in
                        #    a linear expansion if the fogi directions were an orthogonal basis) to 1 using
                        #    the `norm_order` norm.  The fogi directions are then normalized so
                        #    numpy.dot(dir, vec) = 1.0, i.e. so their L2 norm = 1/L2-norm2-of-norm_order-normalized-vec.
                        #    => set dir = vec / L2(vec)**2 so, if L2(vec)=x, then L2(dir)=1/x and dot(dir,vec) = 1.0
                        #    (Recall: The fogi component is defined as dot(fogi_dir.T, e)).
                        # 2) gauge vectors epsilon are also chosen to be normalized to 1 within gauge space.
                        #    Each epsilon in the intersection space gives rise via the "M" action
                        #    (inv_diff_gauge_action) to a fogi vector of norm 1/r so that
                        #    fogi_dir = r * dot(M, epsilon) = r * dot(inv_diff_gauge_action, int_vec)
                        #    We keep track of this 'r' value as a way of converting between the gauge-space-normalized
                        #    FOGI direction to the errgen-space-normalized version of it.
                        # The errgen-space-normalized fogi vector (in fogi_dirs) defines the "FOGI component",
                        # whereas the gauge-space-normalized version defines the "FOGI gauge angle"
                        # theta = dot(gauge_normd_fogi_dir.T, e) = dot( dot(M, epsilon).T, e) = dot(fogi_dir.T / r, e)
                        #       = component / r

                        norm_order_array = resolve_norm_order(
                            local_fogi_dirs,
                            [errorgen_coefficient_labels[ol] for ol in existing_set + (op_label,)],
                            norm_order)

                        assert(_np.linalg.norm(_np.dot(gauge_action.T, local_fogi_dirs)) < 1e-8)
                        # transpose => dot(local_fogi_dirs.T, gauge_action) = 0
                        # = int_spc.T * [ pinv_gA  -pinv_gB ] * [[ga] [gB]]
                        # = int_spc.T * (pinv_gA * gA - pinv_gB * gB) = 0 b/c int_vec lies in "support" of A & B,
                        #   i.e. int_spc.T * (pinv_gA * gA) * int_spc == I and similar with B, so get I - I = 0

                        new_fogi_dirs = _sps.lil_matrix((num_elem_errgens, local_fogi_dirs.shape[1]),
                                                        dtype=local_fogi_dirs.dtype); off = 0
                        for ol in existing_set + (op_label,):  # NOT new_set here b/c concat order above
                            n = len(errorgen_coefficient_labels[ol])
                            new_fogi_dirs[op_errgen_indices[ol], :] = local_fogi_dirs[off:off + n, :]; off += n
                        new_fogi_dirs = new_fogi_dirs.tocsc()

                        # figure out which directions are independent
                        indep_cols = _mt.independent_columns(new_fogi_dirs, fogi_dirs)
                        #FOGI DEBUG print(" ==> %d independent columns" % len(indep_cols))

                        if dependent_fogi_action == "drop":
                            dep_cols_to_add = []
                        elif dependent_fogi_action == "mark":
                            #Still add, as dependent fogi quantities, those that are independent of
                            # all the smaller-size op sets but dependent only on other sets of the current size.
                            smallset_indep_cols = _mt.independent_columns(
                                new_fogi_dirs, fogi_dirs[:, 0:num_indep_vecs_from_smaller_sets])
                            indep_cols_set = set(indep_cols)  # just for faster lookup
                            dep_cols_to_add = [i for i in smallset_indep_cols if i not in indep_cols_set]
                        else:
                            raise ValueError("Invalid `dependent_fogi_action` value: %s" % str(dependent_fogi_action))

                        # add new_fogi_dirs[:, indep_cols] to fogi_dirs w/meta data
                        fogi_dirs = add_relational_fogi_dirs(new_fogi_dirs[:, indep_cols],
                                                             _np.take(int_vecs, indep_cols, axis=1),
                                                             _np.take(intersection_space, indep_cols, axis=1),
                                                             fogi_dirs, fogi_meta, existing_set, op_label, new_set,
                                                             norm_order_array[indep_cols])

                        # add new_fogi_dirs[:, dep_cols_to_add] to dep_fogi_dirs w/meta data
                        dep_fogi_dirs = add_relational_fogi_dirs(new_fogi_dirs[:, dep_cols_to_add],
                                                                 _np.take(int_vecs, dep_cols_to_add, axis=1),
                                                                 _np.take(intersection_space, dep_cols_to_add, axis=1),
                                                                 dep_fogi_dirs, dep_fogi_meta, existing_set,
                                                                 op_label, new_set, norm_order_array[dep_cols_to_add])

                        #if dependent_fogi_action == "drop":  # we could construct these, but would make fogi qtys messy
                        #    assert(_mt.columns_are_orthogonal(fogi_dirs))

                        #print("Fogi vecs:\n"); _mt.print_mx(local_fogi_dirs)
                        #print("Ham Intersection names: ", intersection_names)

                    ccomms[new_set] = union_space
                    #print("Complement:\n"); _mt.print_mx(union_space)

                larger_sets.append(new_set)

        smaller_sets = larger_sets

    #big_gauge_action = _np.concatenate([other_gauge_action[ol] for ol in primitive_op_labels], axis=0)  # DEBUG
    #print("Fogi directions:\n"); _mt.print_mx(fogi_dirs, width=5, prec=1)
    #print("Names = \n", '\n'.join(["%d: %s" % (i, v) for i, v in enumerate(fogi_names)]))
    #print("Rank = ", _np.linalg.matrix_rank(fogi_dirs))

    #Convert to real matrices if possible (otherwise we can get pinv or nullspace being complex when it doesn't
    # need to be, and this causes, e.g. an attempt to set imaginary Hamiltonian coefficients of ops)
    if _spsl.norm(fogi_dirs.imag) < 1e-6:
        fogi_dirs = fogi_dirs.real
    if _spsl.norm(dep_fogi_dirs.imag) < 1e-6:
        dep_fogi_dirs = dep_fogi_dirs.real

    return (fogi_dirs, fogi_meta, dep_fogi_dirs, dep_fogi_meta)


#def create_fogi_dir_labels(fogi_opsets, fogi_dirs, fogi_rs, fogi_gaugespace_dirs, errorgen_coefficients):
#
#    fogi_names = []
#    fogi_abbrev_names = []
#
#    # Note: fogi_dirs is a 2D array, so .T to iterate over cols, whereas fogi_gaugespace_dirs
#    #  is a list of vectors, so just iterating is fine.
#    for opset, fogi_dir, fogi_epsilon in zip(fogi_opsets, fogi_dirs.T, fogi_gaugespace_dirs):
#
#        if len(opset) == 1:  # Intrinsic quantity
#            assert(fogi_epsilon is None)
#            op_elemgen_labels = errorgen_coefficient_labels[op_label]
#            errgen_name = elem_vec_name(fogi_dir, op_elemgen_labels)
#            errgen_names_abbrev = elem_vec_names(local_fogi_dirs, op_elemgen_labels, include_type=False)
#            fogi_names.extend(["%s_%s" % ((("(%s)" % egname) if (' ' in egname) else egname),
#                                          op_label_abbrevs.get(op_label, str(op_label)))
#                               for egname in errgen_names])
#            fogi_abbrev_names.extend(errgen_names_abbrev)
#
#                                    intersection_space_to_add = _np.take(intersection_space, rel_cols_to_add, axis=1)
#                        #intersection_space_to_add = _np.dot(gauge_linear_combos, indep_intersection_space) \
#                        #    if (gauge_linear_combos is not None) else intersection_space_to_add
#
#
#
#
#                        intersection_names = elem_vec_names(intersection_space_to_add, gauge_elemgen_labels)
#                        intersection_names_abbrev = elem_vec_names(intersection_space_to_add, gauge_elemgen_labels,
#                                                                   include_type=False)
#                        fogi_names.extend(["ga(%s)_%s - ga(%s)_%s" % (
#                            iname, "|".join([op_label_abbrevs.get(l, str(l)) for l in existing_set]),
#                            iname, op_label_abbrevs.get(op_label, str(op_label))) for iname in intersection_names])
#                        fogi_abbrev_names.extend(["ga(%s)" % iname for iname in intersection_names_abbrev])


def compute_maximum_relational_errors(primitive_op_labels, errorgen_coefficients, gauge_action_matrices,
                                      errorgen_coefficient_bases_by_op, gauge_basis, model_dim):
    """ TODO: docstring """

    gaugeSpaceDim = len(gauge_basis)
    errorgen_vec = {}
    for op_label in primitive_op_labels:
        errgen_dict = errorgen_coefficients[op_label]
        errorgen_vec[op_label] = _np.array([errgen_dict.get(eglbl, 0)
                                            for eglbl in errorgen_coefficient_bases_by_op[op_label].labels])

    def fix_gauge_using_op(op_label, allowed_gauge_directions, available_op_labels, running_best_gauge_vec,
                           best_gauge_vecs, debug, op_label_to_compute_max_for):
        if op_label is not None:  # initial iteration gives 'None' as op_label to kick things off
            ga = gauge_action_matrices[op_label]

            # get gauge directions that commute with gate:
            commutant = _mt.nullspace(ga)  # columns = *gauge* elem gen directions - these can't be fixed by this op
            assert(_mt.columns_are_orthonormal(commutant))
            #complement = _mt.nullspace(commutant.T)  # complement of commutant - where op is faithful rep

            # take intersection btwn allowed_gauge_directions and complement
            best_gauge_dir = - _np.dot(_np.linalg.pinv(ga), errorgen_vec[op_label])
            coeffs = _np.dot(_np.linalg.pinv(allowed_gauge_directions), best_gauge_dir)  # project onto Q (allowed dirs)
            projected_best_gauge_dir = _np.dot(allowed_gauge_directions, coeffs)

            # add projected vec to running "best_gauge_vector"
            running_best_gauge_vec = running_best_gauge_vec.copy()
            running_best_gauge_vec += projected_best_gauge_dir

            #update allowed gauge directions by taking intersection with commutant
            allowed_gauge_directions = _mt.intersection_space(allowed_gauge_directions, commutant,
                                                              use_nice_nullspace=False)
            assert(_mt.columns_are_orthogonal(allowed_gauge_directions))
            for i in range(allowed_gauge_directions.shape[1]):
                allowed_gauge_directions[:, i] /= _np.linalg.norm(allowed_gauge_directions[:, i])
            assert(_mt.columns_are_orthonormal(allowed_gauge_directions))

            available_op_labels.remove(op_label)

        if allowed_gauge_directions.shape[1] > 0:  # if there are still directions to fix, recurse
            assert(len(available_op_labels) > 0), "There are still unfixed gauge directions but we've run out of gates!"
            for oplbl in available_op_labels:
                fix_gauge_using_op(oplbl, allowed_gauge_directions, available_op_labels.copy(),
                                   running_best_gauge_vec, best_gauge_vecs, debug + [oplbl],
                                   op_label_to_compute_max_for)
        else:
            # we've entirely fixed the gauge - running_best_gauge_vec is actually a best gauge vector now.
            errgen_shift = _np.dot(gauge_action_matrices[op_label_to_compute_max_for], running_best_gauge_vec)
            #commutant = _mt.nullspace(ga)  # columns = *gauge* elem gen directions - these can't be fixed by this op
            #assert(_mt.columns_are_orthonormal(commutant))
            #complement = _mt.nullspace(commutant.T)  # complement of commutant - where op is faithful rep

            ga = gauge_action_matrices[op_label_to_compute_max_for]
            errgen_vec = errorgen_vec[op_label_to_compute_max_for] + errgen_shift
            errgen_vec = _np.dot(_np.dot(ga, _np.linalg.pinv(ga)), errgen_vec)

            #jangle = _mt.jamiolkowski_angle(_create_errgen_op(errgen_vec, gauge_basis_mxs))
            #FOGI DEBUG print("From ", debug, " jangle = ", jangle)
            best_gauge_vecs.append(running_best_gauge_vec)

    def _create_errgen_op(vec, list_of_mxs):
        return sum([c * mx for c, mx in zip(vec, list_of_mxs)])

    from ..baseobjs import Basis as _Basis
    ret = {}
    normalized_pauli_basis = _Basis.cast('pp', model_dim)
    scale = model_dim**(0.25)  # to change to standard pauli-product matrices
    gauge_basis_mxs = [mx * scale for mx in normalized_pauli_basis.elements[1:]]

    for op_label_to_compute_max_for in primitive_op_labels:

        #FOGI DEBUG print("Computing for", op_label_to_compute_max_for)
        running_gauge_vec = _np.zeros(gaugeSpaceDim, 'd')
        initial_allowed_gauge_directions = _np.identity(gaugeSpaceDim, 'd')
        resulting_best_gauge_vecs = []

        available_labels = set(primitive_op_labels)
        #available_labels.remove(op_label_to_compute_max_for)
        fix_gauge_using_op(None, initial_allowed_gauge_directions, available_labels,
                           running_gauge_vec, resulting_best_gauge_vecs, debug=[],
                           op_label_to_compute_max_for=op_label_to_compute_max_for)

        jamiol_angles = []
        ga = gauge_action_matrices[op_label_to_compute_max_for]
        projector = _np.dot(ga, _np.linalg.pinv(ga))
        for gauge_vec in resulting_best_gauge_vecs:
            errgen_shift = _np.dot(gauge_action_matrices[op_label_to_compute_max_for], gauge_vec)
            errgen_vec = errorgen_vec[op_label_to_compute_max_for] + errgen_shift
            errgen_vec = _np.dot(projector, errgen_vec)  # project onto non-local space
            jamiol_angles.append(_mt.jamiolkowski_angle(_create_errgen_op(errgen_vec, gauge_basis_mxs)))

        max_relational_jangle = max(jamiol_angles)
        #FOGI DEBUG print(max_relational_jangle)
        ret[op_label_to_compute_max_for] = max_relational_jangle
    return ret


#An alternative but inferior algorithm for constructing FOGI quantities: Keep around for checking/reference or REMOVE?
#def _compute_fogi_via_nullspaces(self, primitive_op_labels, ham_basis, other_basis, other_mode="all",
#                                 ham_gauge_linear_combos=None, other_gauge_linear_combos=None,
#                                 op_label_abbrevs=None, reduce_to_model_space=True):
#    num_ham_elem_errgens = (len(ham_basis) - 1)
#    num_other_elem_errgens = (len(other_basis) - 1)**2 if other_mode == "all" else (len(other_basis) - 1)
#    ham_elem_labels = [('H', bel) for bel in ham_basis.labels[1:]]
#    other_elem_labels = [('S', bel) for bel in other_basis.labels[1:]] if other_mode != "all" else \
#        [('S', bel1, bel2) for bel1 in other_basis.labels[1:] for bel2 in other_basis.labels[1:]]
#    assert(len(ham_elem_labels) == num_ham_elem_errgens)
#    assert(len(other_elem_labels) == num_other_elem_errgens)
#
#    #Get lists of the present (existing within the model) labels for each operation
#    ham_labels_for_op = {op_label: ham_elem_labels[:] for op_label in primitive_op_labels}  # COPY lists!
#    other_labels_for_op = {op_label: other_elem_labels[:] for op_label in primitive_op_labels}  # ditto
#    if reduce_to_model_space:
#        for op_label in primitive_op_labels:
#            op = self.operations[op_label]
#            lbls = op.errorgen_coefficient_labels()
#            present_ham_elem_lbls = set(filter(lambda lbl: lbl[0] == 'H', lbls))
#            present_other_elem_lbls = set(filter(lambda lbl: lbl[0] == 'S', lbls))
#
#            disallowed_ham_space_labels = set(ham_elem_labels) - present_ham_elem_lbls
#            disallowed_row_indices = [ham_elem_labels.index(disallowed_lbl)
#                                      for disallowed_lbl in disallowed_ham_space_labels]
#            for i in sorted(disallowed_row_indices, reverse=True):
#                del ham_labels_for_op[op_label][i]
#
#            disallowed_other_space_labels = set(other_elem_labels) - present_other_elem_lbls
#            disallowed_row_indices = [other_elem_labels.index(disallowed_lbl)
#                                      for disallowed_lbl in disallowed_other_space_labels]
#            for i in sorted(disallowed_row_indices, reverse=True):
#                del other_labels_for_op[op_label][i]
#
#    #Step 1: construct nullspaces associated with sets of operations
#    ham_nullspaces = {}
#    other_nullspaces = {}
#    max_size = len(primitive_op_labels)
#    for set_size in range(1, max_size + 1):
#        ham_nullspaces[set_size] = {}  # dict mapping operation-sets of `set_size` to nullspaces
#        other_nullspaces[set_size] = {}
#
#        for op_set in _itertools.combinations(primitive_op_labels, set_size):
#            #print(op_set)
#            ham_gauge_action_mxs = []
#            other_gauge_action_mxs = []
#            ham_rows_by_op = {}; h_off = 0
#            other_rows_by_op = {}; o_off = 0
#            for op_label in op_set:  # Note: "ga" stands for "gauge action" in variable names below
#                op = self.operations[op_label]
#                if isinstance(op, _op.LindbladOp):
#                    op_mx = op.unitary_postfactor.to_dense()
#                else:
#                    assert(False), "STOP - you probably don't want to do this!"
#                    op_mx = op.to_dense()
#                U = _bt.change_basis(op_mx, self.basis, 'std')
#                ham_ga = _gt.first_order_ham_gauge_action_matrix(U, ham_basis)
#                other_ga = _gt.first_order_other_gauge_action_matrix(U, other_basis, other_mode)
#
#                if ham_gauge_linear_combos is not None:
#                    ham_ga = _np.dot(ham_ga, ham_gauge_linear_combos)
#                if other_gauge_linear_combos is not None:
#                    other_ga = _np.dot(other_ga, other_gauge_linear_combos)
#
#                ham_gauge_action_mxs.append(ham_ga)
#                other_gauge_action_mxs.append(other_ga)
#                reduced_ham_nrows = len(ham_labels_for_op[op_label])  # ham_ga.shape[0] when unrestricted
#                reduced_other_nrows = len(other_labels_for_op[op_label])  # other_ga.shape[0] when unrestricted
#                ham_rows_by_op[op_label] = slice(h_off, h_off + reduced_ham_nrows); h_off += reduced_ham_nrows
#                other_rows_by_op[op_label] = slice(o_off, o_off + reduced_other_nrows); o_off += reduced_other_nrows
#                assert(ham_ga.shape[0] == num_ham_elem_errgens)
#                assert(other_ga.shape[0] == num_other_elem_errgens)
#
#            #Stack matrices to form "base" gauge action matrix for op_set
#            ham_ga_mx = _np.concatenate(ham_gauge_action_mxs, axis=0)
#            other_ga_mx = _np.concatenate(other_gauge_action_mxs, axis=0)
#
#            # Intersect gauge action with the space of elementary errorgens present in the model.
#            # We may need to eliminate some rows of X_ga matrices, and (only) keep linear combos
#            # of the columns that are zero on these rows.
#            present_ham_elem_lbls = set()
#            present_other_elem_lbls = set()
#            for op_label in op_set:
#                op = self.operations[op_label]
#                lbls = op.errorgen_coefficient_labels()  # length num_coeffs
#                present_ham_elem_lbls.update([(op_label, lbl) for lbl in lbls if lbl[0] == 'H'])
#                present_other_elem_lbls.update([(op_label, lbl) for lbl in lbls if lbl[0] == 'S'])
#
#            full_ham_elem_labels = [(op_label, elem_lbl) for op_label in op_set
#                                    for elem_lbl in ham_elem_labels]
#            assert(present_ham_elem_lbls.issubset(full_ham_elem_labels)), \
#                "The given space of hamiltonian elementary gauge-gens must encompass all those in model ops!"
#            disallowed_ham_space_labels = set(full_ham_elem_labels) - present_ham_elem_lbls
#            disallowed_row_indices = [full_ham_elem_labels.index(disallowed_lbl)
#                                      for disallowed_lbl in disallowed_ham_space_labels]
#
#            if reduce_to_model_space and len(disallowed_row_indices) > 0:
#                #disallowed_rows = _np.take(ham_ga_mx, disallowed_row_indices, axis=0)
#                #allowed_linear_combos = _mt.nice_nullspace(disallowed_rows, tol=1e-4)
#                #ham_ga_mx = _np.dot(ham_ga_mx, allowed_linear_combos)
#                ham_ga_mx = _np.delete(ham_ga_mx, disallowed_row_indices, axis=0)
#
#            full_other_elem_labels = [(op_label, elem_lbl) for op_label in op_set
#                                      for elem_lbl in other_elem_labels]
#            assert(present_other_elem_lbls.issubset(full_other_elem_labels)), \
#                "The given space of 'other' elementary gauge-gens must encompass all those in model ops!"
#            disallowed_other_space_labels = set(full_other_elem_labels) - present_other_elem_lbls
#            disallowed_row_indices = [full_other_elem_labels.index(disallowed_lbl)
#                                      for disallowed_lbl in disallowed_other_space_labels]
#
#            if reduce_to_model_space and len(disallowed_row_indices) > 0:
#                #disallowed_rows = _np.take(other_ga_mx, disallowed_row_indices, axis=0)
#                #allowed_linear_combos = _mt.nice_nullspace(disallowed_rows, tol=1e-4)
#                #other_ga_mx = _np.dot(other_ga_mx, allowed_linear_combos)
#                other_ga_mx = _np.delete(other_ga_mx, disallowed_row_indices, axis=0)
#
#            #Add all known (already tabulated) nullspace directions so that we avoid getting them again
#            # when we compute the nullspace of the gauge action matrix below.
#            for previous_size in range(1, set_size + 1):  # include current size!
#                for previous_op_set, (nullsp, previous_rows) in ham_nullspaces[previous_size].items():
#                    padded_nullsp = _np.zeros((ham_ga_mx.shape[0], nullsp.shape[1]), 'd')
#                    for op in previous_op_set:
#                        if op not in ham_rows_by_op: continue
#                        padded_nullsp[ham_rows_by_op[op], :] = nullsp[previous_rows[op], :]
#                    ham_ga_mx = _np.concatenate((ham_ga_mx, padded_nullsp), axis=1)
#
#                for previous_op_set, (nullsp, previous_rows) in other_nullspaces[previous_size].items():
#                    padded_nullsp = _np.zeros((other_ga_mx.shape[0], nullsp.shape[1]), other_ga_mx.dtype)
#                    for op in previous_op_set:
#                        if op not in other_rows_by_op: continue
#                        padded_nullsp[other_rows_by_op[op], :] = nullsp[previous_rows[op], :]
#                    other_ga_mx = _np.concatenate((other_ga_mx, padded_nullsp), axis=1)
#
#            #Finally, compute the nullspace of the resulting gauge-action + already-tallied matrix:
#            nullspace = _mt.nice_nullspace(ham_ga_mx.T)
#            ham_nullspaces[set_size][op_set] = (nullspace, ham_rows_by_op)
#            #DEBUG: print("  NULLSP DIM = ",nullspace.shape[1])
#            #DEBUG: labels = [(op_label, elem_lbl) for op_label in op_set
#            #DEBUG:           for elem_lbl in ham_labels_for_op[op_label]]
#            #DEBUG: print("\n".join(fogi_names(nullspace, labels, op_label_abbrevs)))
#
#            nullspace = _mt.nice_nullspace(other_ga_mx.T)
#            other_nullspaces[set_size][op_set] = (nullspace, other_rows_by_op)
#
#    # Step 2: convert these per-operation-set nullspaces into vectors over a single "full"
#    #  space of all the elementary error generators (as given by ham_basis, other_basis, & other_mode)
#
#    # Note: "full" designation is for space of all elementary error generators as given by their
#    #  supplied ham_basis, other_basis, and other_mode.
#
#    # Construct full-space vectors for each nullspace vector found by crawling through
#    #  the X_nullspaces dictionary and embedding values as needed.
#    ham_rows_by_op = {}; off = 0
#    for op_label in primitive_op_labels:
#        ham_rows_by_op[op_label] = slice(off, off + len(ham_labels_for_op[op_label]))
#        off += len(ham_labels_for_op[op_label])
#    full_ham_fogi_vecs = _np.empty((off, 0), 'd')
#    for size in range(1, max_size + 1):
#        for op_set, (nullsp, op_set_rows) in ham_nullspaces[size].items():
#            padded_nullsp = _np.zeros((full_ham_fogi_vecs.shape[0], nullsp.shape[1]), 'd')
#            for op in op_set:
#                padded_nullsp[ham_rows_by_op[op], :] = nullsp[op_set_rows[op], :]
#            full_ham_fogi_vecs = _np.concatenate((full_ham_fogi_vecs, padded_nullsp), axis=1)
#
#    other_rows_by_op = {}; off = 0
#    for op_label in primitive_op_labels:
#        other_rows_by_op[op_label] = slice(off, off + len(other_labels_for_op[op_label]))
#        off += len(other_labels_for_op[op_label])
#    full_other_fogi_vecs = _np.empty((off, 0), complex)
#    for size in range(1, max_size + 1):
#        for op_set, (nullsp, op_set_rows) in other_nullspaces[size].items():
#            padded_nullsp = _np.zeros((full_other_fogi_vecs.shape[0], nullsp.shape[1]), complex)
#            for op in op_set:
#                padded_nullsp[other_rows_by_op[op], :] = nullsp[op_set_rows[op], :]
#            full_other_fogi_vecs = _np.concatenate((full_other_fogi_vecs, padded_nullsp), axis=1)
#
#    assert(_np.linalg.matrix_rank(full_ham_fogi_vecs) == full_ham_fogi_vecs.shape[1])
#    assert(_np.linalg.matrix_rank(full_other_fogi_vecs) == full_other_fogi_vecs.shape[1])
#
#    # Returns the vectors of FOGI (first order gauge invariant) linear combos as well
#    # as lists of labels for the columns & rows, respectively.
#    return (full_ham_fogi_vecs, ham_labels_for_op), (full_other_fogi_vecs, other_labels_for_op)


def op_elem_vec_name(vec, elem_op_labels, op_label_abbrevs):
    name = ""
    for i, (op_lbl, elem_lbl) in enumerate(elem_op_labels):
        sslbls = elem_lbl.sslbls  # assume a *global* elem errogen label
        egtype = elem_lbl.errorgen_type
        bels = elem_lbl.basis_element_labels

        sslbls_str = ''.join(map(str, sslbls))
        val = vec[i][0, 0] if _sps.issparse(vec[i]) else vec[i]
        if abs(val) < 1e-6: continue
        sign = ' + ' if val > 0 else ' - '
        abs_val_str = '' if _np.isclose(abs(val), 1.0) else ("%g " % abs(val))  # was %.1g
        basis_str = ','.join(["%s:%s" % (bel, sslbls_str) for bel in bels])
        name += sign + abs_val_str + "%s(%s)_%s" % (egtype, basis_str,
                                                    op_label_abbrevs.get(op_lbl, str(op_lbl)))
    if name.startswith(' + '): name = name[3:]  # strip leading +
    if name.startswith(' - '): name = '-' + name[3:]  # strip leading spaces
    return name


def op_elem_vec_names(vecs, elem_op_labels, op_label_abbrevs):
    """ TODO: docstring """
    if op_label_abbrevs is None: op_label_abbrevs = {}
    return [op_elem_vec_name(vecs[:, j], elem_op_labels, op_label_abbrevs) for j in range(vecs.shape[1])]


def elem_vec_name(vec, elem_labels, include_type=True):
    """ TODO: docstring """
    name = ""
    for i, elem_lbl in enumerate(elem_labels):
        sslbls = elem_lbl.sslbls  # assume a *global* elem errogen label
        egtype = elem_lbl.errorgen_type
        bels = elem_lbl.basis_element_labels

        sslbls_str = ''.join(map(str, sslbls))
        val = vec[i]
        if abs(val) < 1e-6: continue
        sign = ' + ' if val > 0 else ' - '
        abs_val_str = '' if _np.isclose(abs(val), 1.0) else ("%g " % abs(val))  # was %.1g
        basis_str = ','.join(["%s:%s" % (bel, sslbls_str) for bel in bels])
        if include_type:
            name += sign + abs_val_str + "%s(%s)" % (egtype, basis_str)
        else:
            name += sign + abs_val_str + "%s" % basis_str  # 'H' or 'S'

    if name.startswith(' + '): name = name[3:]  # strip leading +
    if name.startswith(' - '): name = '-' + name[3:]  # strip leading spaces
    return name


def elem_vec_names(vecs, elem_labels, include_type=True):
    """ TODO: docstring """
    return [elem_vec_name(vecs[:, j], elem_labels, include_type) for j in range(vecs.shape[1])]
