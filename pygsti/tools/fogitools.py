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


from . import matrixtools as _mt
from . import optools as _ot


def first_order_ham_gauge_action_matrix(clifford_superop, dmbasis_ham):
    #Note: clifford_superop must be in the *std* basis!
    normalize = True  # I think ?
    ham_gens, _ = _ot.lindblad_error_generators(dmbasis_ham, None, normalize)

    ham_action_mx = _np.empty((len(ham_gens), len(ham_gens)), complex)
    for j, gen in enumerate(ham_gens):
        conjugated_gen = _np.dot(clifford_superop, _np.dot(gen, _np.conjugate(clifford_superop.T)))
        gauge_action_deriv = gen - conjugated_gen
        for i, gen2 in enumerate(ham_gens):
            val = _np.vdot(gen2.flat, gauge_action_deriv.flat)
            ham_action_mx[i, j] = val
    assert(_np.linalg.norm(ham_action_mx.imag) < 1e-6)
    ham_action_mx = ham_action_mx.real

    return ham_action_mx


def first_order_other_gauge_action_matrix(clifford_superop, dmbasis_other, other_mode="all"):
    #Note: clifford_superop must be in the *std* basis!
    normalize = True  # I think ?
    _, other_gens = _ot.lindblad_error_generators(None, dmbasis_other, normalize, other_mode)

    if other_mode == 'diagonal':
        all_other_gens = other_gens
    elif other_mode == 'all':
        all_other_gens = [other_gens[i][j] for i in range(len(other_gens)) for j in range(len(other_gens))]
    else:
        raise ValueError("Invalid `other_mode`: only 'diagonal' and 'all' modes are supported so far.")

    other_action_mx = _np.empty((len(all_other_gens), len(all_other_gens)), complex)
    for j, gen in enumerate(all_other_gens):
        conjugated_gen = _np.dot(clifford_superop, _np.dot(gen, _np.conjugate(clifford_superop.T)))
        gauge_action_deriv = gen - conjugated_gen
        for i, gen2 in enumerate(all_other_gens):
            val = _np.vdot(gen2.flat, gauge_action_deriv.flat)
            other_action_mx[i, j] = val
    #assert(_np.linalg.norm(other_action_mx.imag) < 1e-6)
    other_action_mx = _np.real_if_close(other_action_mx)  # .real

    return other_action_mx


def construct_gauge_space_for_model(primitive_op_labels, gauge_action_matrices,
                                    errorgen_coefficient_labels, elem_errgen_labels,
                                    reduce_to_model_space=True):
    """
    TODO: docstring
    note: elem_errgen_labels labels the basis elements of the (matrix) elements of gauge_action_matrices.
    We expect the values of gauge_action_matrices to be square matrices with dimension len(elem_errgen_labels).
    """

    # Intersect gauge action with the space of elementary errorgens present in the model.
    # We may need to eliminate some rows of X_ga matrices, and (only) keep linear combos
    # of the columns that are zero on these rows.
    ga_by_op = {}  # potentially different than gauge_action_matrices when reduce_to_model_space=True
    disallowed_rows_by_op = {}
    elgen_lbls_by_op = {}  # labels of the elementary error generators that correspond to the rows of ga_by_op
    all_elgen_lbls = _np.empty(len(elem_errgen_labels), dtype=object)  # as an array for convenience below
    all_elgen_lbls[:] = elem_errgen_labels

    for op_label in primitive_op_labels:
        present_elem_lbls = set(errorgen_coefficient_labels[op_label])
        assert(present_elem_lbls.issubset(elem_errgen_labels)), \
            "The given space of elementary error gens must encompass all those in the %s model op!" % str(op_label)

        disallowed_labels = set(elem_errgen_labels) - present_elem_lbls
        disallowed_row_indices = [elem_errgen_labels.index(disallowed_lbl) for disallowed_lbl in disallowed_labels]
        if reduce_to_model_space and len(disallowed_row_indices) > 0:
            ga_by_op[op_label] = _np.delete(gauge_action_matrices[op_label], disallowed_row_indices, axis=0)
            elgen_lbls_by_op[op_label] = _np.delete(all_elgen_lbls, disallowed_row_indices, axis=0)
            disallowed_rows_by_op[op_label] = _np.take(gauge_action_matrices[op_label],
                                                       disallowed_row_indices, axis=0)
            assert(set(list(elgen_lbls_by_op[op_label])) == present_elem_lbls)
        else:
            ga_by_op[op_label] = gauge_action_matrices[op_label]
            elgen_lbls_by_op[op_label] = all_elgen_lbls
            disallowed_rows_by_op[op_label] = _np.empty((0, gauge_action_matrices[op_label].shape[1]), 'd')

    allop_ga_mx = _np.concatenate([ga_by_op[op_label] for op_label in primitive_op_labels], axis=0)

    disallowed_rows = _np.concatenate([disallowed_rows_by_op[op_label] for op_label in primitive_op_labels], axis=0)
    if reduce_to_model_space and disallowed_rows.shape[0] > 0:
        allowed_gauge_linear_combos = _mt.nice_nullspace(disallowed_rows, tol=1e-4)
        allop_ga_mx = _np.dot(allop_ga_mx, allowed_gauge_linear_combos)
        ga_by_op = {k: _np.dot(v, allowed_gauge_linear_combos) for k, v in ga_by_op.items()}
    else:
        allowed_gauge_linear_combos = None

    return allop_ga_mx, ga_by_op, elgen_lbls_by_op, allowed_gauge_linear_combos


def construct_fogi_quantities(primitive_op_labels, gauge_action_matrices,
                              errorgen_coefficient_labels, gauge_elemgen_labels,
                              op_label_abbrevs=None, dependent_fogi_action='drop', norm_order=None):
    """ TODO: docstring """
    assert(dependent_fogi_action in ('drop', 'mark'))
    orthogonalize_relationals = True

    #typ = 'H' if (mode is None) else 'S'
    #if typ == 'H':
    #    elem_labels = [('H', bel) for bel in basis.labels[1:]]
    #else:
    #    elem_labels = [('S', bel) for bel in basis.labels[1:]] if mode != "all" else \
    #        [('S', bel1, bel2) for bel1 in basis.labels[1:] for bel2 in basis.labels[1:]]

    #Get lists of the present (existing within the model) labels for each operation
    if op_label_abbrevs is None: op_label_abbrevs = {}

    op_errgen_indices = {}; off = 0  # tells us which indices of errgen-set space map to which ops
    for op_label in primitive_op_labels:
        num_coeffs = len(errorgen_coefficient_labels[op_label])
        op_errgen_indices[op_label] = slice(off, off + num_coeffs)
        off += num_coeffs
    num_elem_errgens = off

    #Step 1: construct FOGI quantities and reference frame for each op
    ccomms = {}
    fogi_opsets = []
    fogi_dirs = _np.zeros((num_elem_errgens, 0), complex)  # columns = dual vectors ("directions") in error-gen space
    fogi_rs = _np.zeros(0, 'd')
    fogi_names = []
    fogi_abbrev_names = []
    fogi_gaugespace_dirs = []  # columns
    dependent_vec_indices = []

    for op_label in primitive_op_labels:
        #print("##", op_label)
        ga = gauge_action_matrices[op_label]

        #Get commutant and communtant-complement spaces
        commutant = _mt.nice_nullspace(ga)  # columns = *gauge* elem gen directions
        assert(_mt.columns_are_orthogonal(commutant))

        # Note: local/new_fogi_dirs are orthogonal but not necessarily normalized (so need to
        #  normalize to get inverse, but *don't* need pseudo-inverse).
        local_fogi_dirs = _mt.nice_nullspace(ga.T)  # "conjugate space" to gauge action

        #NORMALIZE FOGI DIRS to have norm 1 - based on mapping between unit-variance
        # gaussian distribution of target-gateset perturbations in the usual errorgen-set-space
        # to the FOGI space.  The basis of the fogi directions is understood to be the basis
        # of errorgen-superops arising from *un-normalized* (traditional) Pauli matrices.
        local_fogi_vecs = _mt.normalize_columns(local_fogi_dirs, ord=norm_order)  # this gives us *vec*-norm we want
        vector_L2_norm2s = [_np.linalg.norm(local_fogi_vecs[:, j])**2 for j in range(local_fogi_vecs.shape[1])]
        local_fogi_dirs = local_fogi_vecs / _np.array(vector_L2_norm2s)[None, :]  # gives us *dir*-norm we want  # DUAL NORM

        assert(_mt.columns_are_orthogonal(local_fogi_dirs))  # Not for Cnot in 2Q_XYICNOT (check?)

        new_fogi_dirs = _np.zeros((fogi_dirs.shape[0], local_fogi_dirs.shape[1]), local_fogi_dirs.dtype)
        new_fogi_dirs[op_errgen_indices[op_label], :] = local_fogi_dirs  # "juice" this op
        fogi_dirs = _np.concatenate((fogi_dirs, new_fogi_dirs), axis=1)
        fogi_rs = _np.concatenate((fogi_rs, _np.zeros(new_fogi_dirs.shape[1], 'd')))
        assert(_mt.columns_are_orthogonal(fogi_dirs))

        fogi_gaugespace_dirs.extend([None] * new_fogi_dirs.shape[1])  # local qtys don't have corresp. gauge dirs
        fogi_opsets.extend([(op_label,)] * new_fogi_dirs.shape[1])

        #LABELS
        op_elemgen_labels = errorgen_coefficient_labels[op_label]
        errgen_names = elem_vec_names(local_fogi_vecs, op_elemgen_labels)
        errgen_names_abbrev = elem_vec_names(local_fogi_vecs, op_elemgen_labels, include_type=False)
        fogi_names.extend(["%s_%s" % ((("(%s)" % egname) if (' ' in egname) else egname),
                                      op_label_abbrevs.get(op_label, str(op_label)))
                           for egname in errgen_names])
        fogi_abbrev_names.extend(errgen_names_abbrev)

        complement = _mt.nice_nullspace(commutant.T)  # complement of commutant - where op is faithful rep
        assert(_mt.columns_are_orthogonal(complement))
        ccomms[(op_label,)] = complement
        #gauge_action_for_op[op_label] = ga

        #print("Commutant:"); _mt.print_mx(commutant)
        #print("Names: ", errgen_names)
        #print("Complement:"); _mt.print_mx(complement)

    smaller_sets = [(op_label,) for op_label in primitive_op_labels]
    max_size = len(primitive_op_labels)
    num_indep_fogi_dirs = fogi_dirs.shape[1]
    for set_size in range(1, max_size):
        larger_sets = []
        num_indep_vecs_from_smaller_sets = num_indep_fogi_dirs
        num_vecs_from_smaller_sets = fogi_dirs.shape[1]
        for op_label in primitive_op_labels:
            for existing_set in smaller_sets:
                if op_label in existing_set: continue
                new_set = tuple(sorted(existing_set + (op_label,)))
                if new_set in larger_sets: continue

                # print("\n##", existing_set, "+", op_label)

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
                        # let fogi_dir.T = int_vec.T * pinv(ga_A) - int_vec.T * pinv(ga_B).T so that:
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
                            # test_fogi_dirs = M * epsilons = Q * R
                            # => want orthogonalized dirs "Q" as new dirs: Q = M * epsilons * inv(R) = M * epsilons'
                            intersection_space = _np.dot(intersection_space, _np.linalg.inv(R))  # a "good" basis

                        # start w/normalizd epsilon vecs (normalize according to norm_order, then divide by L2-norm^2
                        # so that the resulting intersection-space vector, after action by "M", projects the component
                        # of the norm_order-normalized gauge-space vector)
                        int_vecs = _mt.normalize_columns(intersection_space, ord=norm_order)
                        vector_L2_norm2s = [_np.linalg.norm(int_vecs[:, j])**2 for j in range(int_vecs.shape[1])]
                        intersection_space = int_vecs / _np.array(vector_L2_norm2s)[None, :]  # DUAL NORM

                        local_fogi_dirs = _np.dot(inv_diff_gauge_action, intersection_space)  # dot("M", epsilons)
                        #Note: at this point `local_fogi_dirs` vectors are gauge-space-normalized, not numpy-norm-1
                        if orthogonalize_relationals:
                            assert(_mt.columns_are_orthogonal(local_fogi_dirs))  # true if we orthogonalize above

                        #NORMALIZATION:
                        # There are two normalizations relevant to relational fogi directions:
                        # 1) we normalize the would-be fogi vectors (those that would be prefixed by components in
                        #    a linear expansion if the fogi directions were an ortogonal basis) to 1 using
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

                        assert(_np.linalg.norm(_np.dot(gauge_action.T, local_fogi_dirs)) < 1e-8)
                        # transpose => dot(local_fogi_dirs.T, gauge_action) = 0
                        # = int_spc.T * [ pinv_gA  -pinv_gB ] * [[ga] [gB]]
                        # = int_spc.T * (pinv_gA * gA - pinv_gB * gB) = 0 b/c int_vec lies in "support" of A & B,
                        #   i.e. int_spc.T * (pinv_gA * gA) * int_spc == I and similar with B, so get I - I = 0

                        #TODO REMOVE (DEBUG)
                        #except:
                        #    print(_np.linalg.norm(_np.dot(gauge_action.T, local_fogi_dirs)))
                        #    print(_np.linalg.norm(_np.dot(gauge_action.conj().T, local_fogi_dirs)))
                        #    print("Try conjugating intersection_space:")
                        #    conj_local_fogi_dirs = _np.dot(inv_diff_gauge_action, intersection_space.conj())
                        #    print(_np.linalg.norm(_np.dot(gauge_action.T, conj_local_fogi_dirs)))
                        #    print(_np.linalg.norm(_np.dot(gauge_action.conj().T, conj_local_fogi_dirs)))
                        #
                        #    print("Test restricted to intersection space:")
                        #    restricted_ga = _np.dot(gauge_action, intersection_space)
                        #    restricted_inv_diff_gauge_action = _np.concatenate((_np.linalg.pinv(restricted_ga[0:n, :], rcond=1e-7),
                        #                                                        -_np.linalg.pinv(restricted_ga[n:, :], rcond=1e-7)),
                        #                                                       axis=1).T
                        #    restricted_local_fogi_dirs = restricted_inv_diff_gauge_action
                        #    print(_np.linalg.norm(_np.dot(gauge_action.T, restricted_local_fogi_dirs)))
                        #    ga = restricted_ga[0:n, :]
                        #    inv_ga = _np.linalg.pinv(ga, rcond=1e-7)
                        #    print(_np.linalg.norm(_np.dot(inv_ga, ga) - _np.identity(ga.shape[1],'d')))
                        #
                        #    print("DEBUG")
                        #    gaA = gauge_action[0:n,:]
                        #    gaB = gauge_action[n:,:]
                        #    pinvA = _np.linalg.pinv(gaA, rcond=1e-7)
                        #    pinvB = _np.linalg.pinv(gaB, rcond=1e-7)
                        #    iA = _np.dot(pinvA, gaA)
                        #    iB = _np.dot(pinvB, gaB)
                        #    tA = _np.dot(intersection_space.T, iA)
                        #    tB = _np.dot(intersection_space.T, iB)
                        #    print(_np.linalg.norm(tA))
                        #    print(_np.linalg.norm(tB))
                        #
                        #    import bpdb; bpdb.set_trace()
                        #    print("prob")

                        new_fogi_dirs = _np.zeros((fogi_dirs.shape[0], local_fogi_dirs.shape[1]),
                                                  local_fogi_dirs.dtype); off = 0
                        for ol in existing_set + (op_label,):  # NOT new_set here b/c concat order below
                            n = len(errorgen_coefficient_labels[ol])
                            new_fogi_dirs[op_errgen_indices[ol], :] = local_fogi_dirs[off:off + n, :]; off += n

                        indep_cols = _mt.independent_columns(_np.concatenate((fogi_dirs, new_fogi_dirs), axis=1),
                                                             start_col=fogi_dirs.shape[1],
                                                             start_independent=num_indep_fogi_dirs)
                        rel_indep_cols = [c - fogi_dirs.shape[1] for c in indep_cols]
                        if dependent_fogi_action == "drop":
                            rel_cols_to_add = rel_indep_cols
                            rel_cols_to_mark = []
                        elif dependent_fogi_action == "mark":
                            smallset_indep_cols = _mt.independent_columns(
                                _np.concatenate((fogi_dirs[:, 0:num_vecs_from_smaller_sets], new_fogi_dirs), axis=1),
                                start_col=num_vecs_from_smaller_sets,
                                start_independent=num_indep_vecs_from_smaller_sets)
                            rel_indep_cols_set = set(rel_indep_cols)  # just for lookup speed
                            rel_cols_to_add = [c - num_vecs_from_smaller_sets for c in smallset_indep_cols]
                            rel_cols_to_mark = [c for c in rel_cols_to_add if c not in rel_indep_cols_set]

                        dependent_vec_indices.extend([c + fogi_dirs.shape[1] for c in rel_cols_to_mark])
                        num_indep_fogi_dirs += len(indep_cols)  # the total number of independent columns

                        vecs_to_add, nrms = _mt.normalize_columns(new_fogi_dirs[:, rel_cols_to_add], ord=norm_order,
                                                                  return_norms=True)  # f_hat_vec = f / nrm
                        vector_L2_norm2s = [_np.linalg.norm(vecs_to_add[:, j])**2 for j in range(vecs_to_add.shape[1])]
                        dirs_to_add = vecs_to_add / _np.array(vector_L2_norm2s)[None, :]  # gives us *dir*-norm we want  # DUAL NORM
                        # f_hat = f_hat_vec / L2^2 = f / (nrm * L2^2) = (1 / (nrm * L2^2)) * f

                        fogi_dirs = _np.concatenate((fogi_dirs, dirs_to_add), axis=1)  # errgen-space NORMALIZED
                        fogi_rs = _np.concatenate((fogi_rs, 1 / (nrms * _np.array(vector_L2_norm2s))))

                        fogi_opsets.extend([new_set] * dirs_to_add.shape[1])

                        #if dependent_fogi_action == "drop":  # we could construct these, but would make fogi qtys messy
                        #    assert(_mt.columns_are_orthogonal(fogi_dirs))

                        #OLD -REMOVE
                        #indep_cols = []  # debug = []
                        #if dependent_fogi_action == "drop":
                        #    for j in range(new_fogi_dirs.shape[1]):
                        #        test = _np.concatenate((fogi_dirs, new_fogi_dirs[:, j:j + 1]), axis=1)
                        #        if _np.linalg.matrix_rank(test, tol=1e-7) == num_indep_fogi_dirs + 1:
                        #            #assert(_mt.columns_are_orthogonal(test))  # there's no reason this needs to be true
                        #            indep_cols.append(j)
                        #            fogi_dirs = test
                        #            num_indep_fogi_dirs += 1
                        #            #debug.append("IND")
                        #        #else:
                        #        #    debug.append('-')
                        #elif dependent_fogi_action == "mark":
                        #    for j in range(new_fogi_dirs.shape[1]):
                        #        test = _np.concatenate((fogi_dirs[:, 0:num_vecs_from_smaller_sets],
                        #                                new_fogi_dirs[:, j:j + 1]), axis=1)
                        #        test2 = _np.concatenate((fogi_dirs, new_fogi_dirs[:, j:j + 1]), axis=1)
                        #        #U, s, Vh = _np.linalg.svd(test2)
                        #        if _np.linalg.matrix_rank(test2, tol=1e-7) == num_indep_fogi_dirs + 1:
                        #            # new vec is indep w/everything
                        #            indep_cols.append(j)
                        #            fogi_dirs = test2
                        #            num_indep_fogi_dirs += 1
                        #            #debug.append("IND")
                        #        elif _np.linalg.matrix_rank(test, tol=1e-7) == num_indep_vecs_from_smaller_sets + 1:
                        #            # new vec is indep w/fogi vecs from smaller sets, so dependency must just be
                        #            # among other vecs for this same size.  Which vecs we keep is arbitrary here,
                        #            # so keep this vec and "mark" it as a linearly dependent vec.
                        #            indep_cols.append(j)  # keep this vec
                        #            fogi_dirs = test2
                        #            dependent_vec_indices.append(fogi_dirs.shape[1] - 1)  # but mark it
                        #            #debug.append("DEP")
                        #        #else:
                        #        #    debug.append('-')
                        #        # else new vec is dependent on *smaller* size fogi vecs - omit it then
                        #print("DEBUG: ",debug)  # TODO: REMOVE this and 'debug' uses above

                        #intersection_space_to_add = _np.take(intersection_space, rel_cols_to_add, axis=1)
                        #intersection_space_to_add = _np.dot(gauge_linear_combos, indep_intersection_space) \
                        #    if (gauge_linear_combos is not None) else intersection_space_to_add

                        int_vecs_to_add = _np.take(int_vecs, rel_cols_to_add, axis=1)  # make labels from *vectors*
                        intersection_names = elem_vec_names(int_vecs_to_add, gauge_elemgen_labels)
                        intersection_names_abbrev = elem_vec_names(int_vecs_to_add, gauge_elemgen_labels,
                                                                   include_type=False)
                        fogi_names.extend(["ga(%s)_%s - ga(%s)_%s" % (
                            iname, "|".join([op_label_abbrevs.get(l, str(l)) for l in existing_set]),
                            iname, op_label_abbrevs.get(op_label, str(op_label))) for iname in intersection_names])
                        fogi_abbrev_names.extend(["ga(%s)" % iname for iname in intersection_names_abbrev])

                        fogi_gaugespace_dirs.extend([intersection_space[:, j] for j in rel_cols_to_add])
                        # Note intersection_space is a subset of the *gauge-space*, and so its basis,
                        # placed in fogi_gaugespace_dirs, is for gauge-space, not errorgen-space.

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

    return (fogi_opsets, fogi_dirs, fogi_rs, fogi_gaugespace_dirs, dependent_vec_indices, op_errgen_indices,
            fogi_names, fogi_abbrev_names)


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
                                      errorgen_coefficient_labels, gauge_elemgen_labels, model_dim):
    """ TODO: docstring """

    gaugeSpaceDim = len(gauge_elemgen_labels)
    errorgen_vec = {}
    for op_label in primitive_op_labels:
        errgen_dict = errorgen_coefficients[op_label]
        errorgen_vec[op_label] = _np.array([errgen_dict.get(eglbl, 0) for eglbl in errorgen_coefficient_labels])

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

            jangle = _mt.jamiolkowski_angle(_create_errgen_op(errgen_vec, gauge_basis_mxs))
            print("From ", debug, " jangle = ", jangle)
            best_gauge_vecs.append(running_best_gauge_vec)

    def _create_errgen_op(vec, list_of_mxs):
        return sum([c * mx for c, mx in zip(vec, list_of_mxs)])

    from ..objects import Basis as _Basis
    ret = {}
    normalized_pauli_basis = _Basis.cast('pp', model_dim)
    scale = model_dim**(0.25)  # to change to standard pauli-product matrices
    gauge_basis_mxs = [mx * scale for mx in normalized_pauli_basis.elements[1:]]

    for op_label_to_compute_max_for in primitive_op_labels:

        print("Computing for", op_label_to_compute_max_for)
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
        print(max_relational_jangle)
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
        val = vec[i]
        if abs(val) < 1e-6: continue
        sign = ' + ' if val > 0 else ' - '
        abs_val_str = '' if _np.isclose(abs(val), 1.0) else ("%g " % abs(val))  # was %.1g
        name += sign + abs_val_str + "%s(%s)_%s" % (elem_lbl[0], ','.join(elem_lbl[1:]),
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
        val = vec[i]
        if abs(val) < 1e-6: continue
        sign = ' + ' if val > 0 else ' - '
        abs_val_str = '' if _np.isclose(abs(val), 1.0) else ("%g " % abs(val))  # was %.1g
        if include_type:
            name += sign + abs_val_str + "%s(%s)" % (elem_lbl[0], ','.join(elem_lbl[1:]))
        else:
            name += sign + abs_val_str + "%s" % (','.join(elem_lbl[1:]))  # 'H' or 'S'

    if name.startswith(' + '): name = name[3:]  # strip leading +
    if name.startswith(' - '): name = '-' + name[3:]  # strip leading spaces
    return name


def elem_vec_names(vecs, elem_labels, include_type=True):
    """ TODO: docstring """
    return [elem_vec_name(vecs[:, j], elem_labels, include_type) for j in range(vecs.shape[1])]
