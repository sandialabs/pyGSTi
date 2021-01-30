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
                              op_label_abbrevs=None, dependent_fogi_action='drop'):
    """ TODO: docstring """
    assert(dependent_fogi_action in ('drop', 'mark'))
    #typ = 'H' if (mode is None) else 'S'
    #if typ == 'H':
    #    elem_labels = [('H', bel) for bel in basis.labels[1:]]
    #else:
    #    elem_labels = [('S', bel) for bel in basis.labels[1:]] if mode != "all" else \
    #        [('S', bel1, bel2) for bel1 in basis.labels[1:] for bel2 in basis.labels[1:]]

    #Get lists of the present (existing within the model) labels for each operation
    if op_label_abbrevs is None: op_label_abbrevs = {}

    op_errgen_indices = {}; off = 0
    for op_label in primitive_op_labels:
        num_coeffs = len(errorgen_coefficient_labels[op_label])
        op_errgen_indices[op_label] = slice(off, off + num_coeffs)
        off += num_coeffs
    num_elem_errgens = off

    #Step 1: construct FOGI quantities and reference frame for each op
    ccomms = {}
    fogi_dirs = _np.zeros((num_elem_errgens, 0), complex)  # columns = dual vectors ("directions") in error-gen space
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
        op_elemgen_labels = errorgen_coefficient_labels[op_label]

        # Note: local/new_fogi_dirs are orthogonal but not necessarily normalized (so need to
        #  normalize to get inverse, but *don't* need pseudo-inverse).
        local_fogi_dirs = _mt.nice_nullspace(ga.T)  # "conjugate space" to gauge action
        assert(_mt.columns_are_orthogonal(fogi_dirs))

        #NORMALIZE FOGI DIRS to have norm 1 -- is this useful?
        for j in range(local_fogi_dirs.shape[1]):  # normalize columns so largest element is +1.0
            mag = _np.linalg.norm(local_fogi_dirs[:, j])
            if mag > 1e-6: local_fogi_dirs[:, j] /= mag

        new_fogi_dirs = _np.zeros((fogi_dirs.shape[0], local_fogi_dirs.shape[1]), local_fogi_dirs.dtype)
        new_fogi_dirs[op_errgen_indices[op_label], :] = local_fogi_dirs  # "juice" this op
        fogi_dirs = _np.concatenate((fogi_dirs, new_fogi_dirs), axis=1)
        assert(_mt.columns_are_orthogonal(fogi_dirs))

        fogi_gaugespace_dirs.extend([None] * new_fogi_dirs.shape[1])  # local qtys don't have corresp. gauge dirs
        errgen_names = elem_vec_names(local_fogi_dirs, op_elemgen_labels)
        errgen_names_abbrev = elem_vec_names(local_fogi_dirs, op_elemgen_labels, include_type=False)
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
                    #assert(_mt.columns_are_orthogonal(intersection_space))  # Not always true

                    if intersection_space.shape[1] > 0:
                        # Then each basis vector of the intersection space defines a gauge-invariant ("fogi")
                        # direction via the difference between that gauge direction's action on A and B:
                        gauge_action = _np.concatenate([gauge_action_matrices[ol] for ol in existing_set]
                                                       + [gauge_action_matrices[op_label]], axis=0)
                        n = sum([gauge_action_matrices[ol].shape[0] for ol in existing_set])  # boundary btwn A & B

                        # gauge trans: e => e + delta_e = e + dot(gauge_action, delta);  e = "errorgen-set space" vector; delta = "gauge space" vector
                        #   we want fogi_dir s.t. dot(fogi_dir.T, e) doesn't change when e transforms as above
                        # ==> we want fogi_dir s.t. dot(fogi_dir.T, gauge_action) == 0   (we want dot(fogi_dir.T, gauge_action, delta) == 0 for all delta)

                        # e = sum_i coeff_i * f_i   f_i are basis vecs, not nec. orthonormal, s.t. there exists
                        #  a dual basis f'_i s.t. dot(f'_i.T, f_i) = dirac_ij
                        #  => coeff_i = dot(f'_i.T, e)

                        # exists subset & subspace of errgen-set space that = span({dot(gauge_action, delta) for all delta}) - call this "gauge-shift space"
                        #  == gauge-orbit of target gateset, i.e. e=vec(0)  (within FOGI framework)
                        # we will construct {f_i} so each f_i is in the gauge-shift space or its orthogonal complement.
                        #  => each coeff_i is FOGI or "gauge", and f_i or f'_i can be labelled as a "FOGI direction" or "gauge direction"

                        # colspace(gauge_action) = gauge-shift space, and we can construct its orthogonal complement
                        # local_fogi_dir found by nullspace of gauge_action: dot(local_fogi.T, gauge_action) = 0
                        # q in nullspace(gauge_action.T) - is q a valid f?  is q in orthog-complement?
                        # is dot(q.T, every_vec_in_gauge_shift_space) = 0 = dot(q.T, delta_e) = dot(q.T, gauge_action, delta) for all delta
                        # dot(gauge_action.T, q) = dot(q.T, gauge_action) = 0

                        # Normalizations s.t H-coeffs == J-gauge-angle, S-coeffs == prob. the error occurs; e.g. f_i = (SX + SY) / 2
                        # so need H-fogi vecs (f_i) to be normalized or ...

                        #Relational qtys: want dot(q.T, gauge_action) = 0

                        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxlet fogi_dir ~= ga_A(intersection_vec) - ga_B(intersection_vec) - but actually
                        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx let fogi_dir = pinv(ga_A).T * int_vec - pinv(ga_B).T * int_vec, so that:
                        # let fogi_dir.T = int_vec.T * pinv(ga_A) - int_vec.T * pinv(ga_B).T so that:
                        # dot(fogi_dir.T, gauge_action) = int_vec.T * (pinv(ga_A) - pinv(ga_B)) * gauge_action
                        #                               = (I - I) = 0
                        # (A,B are faithful reps of gauge on intersection space, so pinv(ga_A) * ga_A
                        # restricted to intersection space is I:   int_spc.T * pinv(ga_A) * ga_A * int_spc == I
                        # (when int_spc vecs are orthonormal) and so the above redues to I - I = 0

                        # HERE WIP: trying to understand how int_vec relates to error magnitudes/metrics:
                        # Note that the errorgen-space vector v := dot(ga_A - ga_B, int_vec) has the property
                        # that dot(fogi_dir.T, v) = int_vec.T * (pinv(ga_A) * gaA + pinv(ga_B) * gaB) * int_vec
                        # = 2 * norm2(int_vec)  (??)

                        # gauge_action maps gauge_space => errorgen_space
                        # pinv_gauge_action maps errorgen_space => gauge_space  (need pinv rather than just rescaling
                        #      & transpose b/c gauge action can contain linear dependencies, but similar)
                        # pinv_gauge_action.T maps gauge_

                        # Almost by definition:
                        # fogi_direction = nullspace(gauge_action.T) = vector s.g. dot(fogi_direction.T, gauge_action) = 0
                        # what gauge direction generates fogi_direction? or actually fogi_vector = fogi_direction / norm2(fogi_direction)?
                        # i.e. fogi_dir_A := pinv(ga_A).T * int_vec = dot(ga_A, X)  - what is X?
                        #  fogi_dir_A.T * ga_A = int_vec.T * pinv(ga_A) * ga_A = int_vec.T
                        # fogi_dir_A = renorm_cols(ga_A) * int_vec
                        # fogi_vec_A = renorm_cols(ga_A) * int_vec / norm2(fogi_dir_A)

                        inv_diff_gauge_action = _np.concatenate((_np.linalg.pinv(gauge_action[0:n, :], rcond=1e-7),
                                                                 -_np.linalg.pinv(gauge_action[n:, :], rcond=1e-7)),
                                                                axis=1).T
                        #inv_diff_gauge_action = _np.concatenate((_np.linalg.pinv(gauge_action[0:n, :].T, rcond=1e-7),
                        #                                         -_np.linalg.pinv(gauge_action[n:, :].T, rcond=1e-7)),
                        #                                        axis=0)  # same as above, b/c T commutes w/pinv (?)

                        #DEBUG REMOVE - this isn't always true - e.g. gauge action can be rank defficient
                        #assert(_mt.columns_are_orthogonal(gauge_action[0:n, :]))
                        #assert(_mt.columns_are_orthogonal(gauge_action[n:, :]))
                        #invA = _mt.pinv_of_matrix_with_orthogonal_columns(gauge_action[0:n, :])
                        #invB = _mt.pinv_of_matrix_with_orthogonal_columns(gauge_action[n:, :])
                        #inv_diff_gauge_action2 = _np.concatenate((invA, -invB), axis=1).T
                        #test = _mt.columns_are_orthogonal(inv_diff_gauge_action2)

                        local_fogi_dirs = _np.dot(inv_diff_gauge_action, intersection_space)
                        #assert(_mt.columns_are_orthogonal(local_fogi_dirs))  # NOT always true...

                        #NORMALIZE FOGI DIRS to have norm 1 -- is this useful? NO, this isn't what we want - let's REMOVE this...
                        #for j in range(local_fogi_dirs.shape[1]):  # normalize columns so largest element is +1.0
                        #    mag = _np.linalg.norm(local_fogi_dirs[:, j])
                        #    if mag > 1e-6:
                        #        local_fogi_dirs[:, j] /= mag
                        #        intersection_space[:, j] /= mag

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

                        indep_cols = []  # debug = []
                        if dependent_fogi_action == "drop":
                            for j in range(new_fogi_dirs.shape[1]):
                                test = _np.concatenate((fogi_dirs, new_fogi_dirs[:, j:j + 1]), axis=1)
                                if _np.linalg.matrix_rank(test, tol=1e-7) == num_indep_fogi_dirs + 1:
                                    #assert(_mt.columns_are_orthogonal(test))  # there's no reason this needs to be true
                                    indep_cols.append(j)
                                    fogi_dirs = test
                                    num_indep_fogi_dirs += 1
                                    #debug.append("IND")
                                #else:
                                #    debug.append('-')
                        elif dependent_fogi_action == "mark":
                            for j in range(new_fogi_dirs.shape[1]):
                                test = _np.concatenate((fogi_dirs[:, 0:num_vecs_from_smaller_sets],
                                                        new_fogi_dirs[:, j:j + 1]), axis=1)
                                test2 = _np.concatenate((fogi_dirs, new_fogi_dirs[:, j:j + 1]), axis=1)
                                #U, s, Vh = _np.linalg.svd(test2)
                                if _np.linalg.matrix_rank(test2, tol=1e-7) == num_indep_fogi_dirs + 1:
                                    # new vec is indep w/everything
                                    indep_cols.append(j)
                                    fogi_dirs = test2
                                    num_indep_fogi_dirs += 1
                                    #debug.append("IND")
                                elif _np.linalg.matrix_rank(test, tol=1e-7) == num_indep_vecs_from_smaller_sets + 1:
                                    # new vec is indep w/fogi vecs from smaller sets, so dependency must just be
                                    # among other vecs for this same size.  Which vecs we keep is arbitrary here,
                                    # so keep this vec and "mark" it as a linearly dependent vec.
                                    indep_cols.append(j)  # keep this vec
                                    fogi_dirs = test2
                                    dependent_vec_indices.append(fogi_dirs.shape[1] - 1)  # but mark it
                                    #debug.append("DEP")
                                #else:
                                #    debug.append('-')
                                # else new vec is dependent on *smaller* size fogi vecs - omit it then
                        #print("DEBUG: ",debug)  # TODO: REMOVE this and 'debug' uses above

                        indep_intersection_space = _np.take(intersection_space, indep_cols, axis=1)
                        #indep_intersection_space = _np.dot(gauge_linear_combos, indep_intersection_space) \
                        #    if (gauge_linear_combos is not None) else indep_intersection_space
                        intersection_names = elem_vec_names(indep_intersection_space, gauge_elemgen_labels)
                        intersection_names_abbrev = elem_vec_names(indep_intersection_space, gauge_elemgen_labels,
                                                                   include_type=False)
                        fogi_names.extend(["ga(%s)_%s - ga(%s)_%s" % (
                            iname, "|".join([op_label_abbrevs.get(l, str(l)) for l in existing_set]),
                            iname, op_label_abbrevs.get(op_label, str(op_label))) for iname in intersection_names])
                        fogi_abbrev_names.extend(["ga(%s)" % iname for iname in intersection_names_abbrev])
                        fogi_gaugespace_dirs.extend([intersection_space[:, j] for j in indep_cols])
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

    return (fogi_dirs, fogi_names, fogi_abbrev_names, dependent_vec_indices,
            fogi_gaugespace_dirs, op_errgen_indices)

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


def op_elem_vec_names(vecs, elem_op_labels, op_label_abbrevs):
    """ TODO: docstring """
    if op_label_abbrevs is None: op_label_abbrevs = {}
    vec_names = []
    for j in range(vecs.shape[1]):
        name = ""
        for i, (op_lbl, elem_lbl) in enumerate(elem_op_labels):
            val = vecs[i, j]
            if abs(val) < 1e-6: continue
            sign = ' + ' if val > 0 else ' - '
            abs_val_str = '' if _np.isclose(abs(val), 1.0) else ("%g " % abs(val))  # was %.1g
            name += sign + abs_val_str + "%s(%s)_%s" % (elem_lbl[0], ','.join(elem_lbl[1:]),
                                                        op_label_abbrevs.get(op_lbl, str(op_lbl)))
        if name.startswith(' + '): name = name[3:]  # strip leading +
        if name.startswith(' - '): name = '-' + name[3:]  # strip leading spaces
        vec_names.append(name)
    return vec_names


def elem_vec_names(vecs, elem_labels, include_type=True):
    """ TODO: docstring """
    vec_names = []
    for j in range(vecs.shape[1]):
        name = ""
        for i, elem_lbl in enumerate(elem_labels):
            val = vecs[i, j]
            if abs(val) < 1e-6: continue
            sign = ' + ' if val > 0 else ' - '
            abs_val_str = '' if _np.isclose(abs(val), 1.0) else ("%g " % abs(val))  # was %.1g
            if include_type:
                name += sign + abs_val_str + "%s(%s)" % (elem_lbl[0], ','.join(elem_lbl[1:]))
            else:
                name += sign + abs_val_str + "%s" % (','.join(elem_lbl[1:]))  # 'H' or 'S'

        if name.startswith(' + '): name = name[3:]  # strip leading +
        if name.startswith(' - '): name = '-' + name[3:]  # strip leading spaces
        vec_names.append(name)
    return vec_names
