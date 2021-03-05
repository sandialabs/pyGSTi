"""
Functions related to computation of the log-likelihood.
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
import warnings as _warnings
from .. import tools as _tools

#pos = lambda x: x**2
pos = abs


class WildcardBudget(object):
    """
    A fixed wildcard budget.

    Encapsulates a fixed amount of "wildcard budget" that allows each circuit
    an amount "slack" in its outcomes probabilities.  The way in which this
    slack is computed - or "distributed", though it need not necessarily sum to
    a fixed total - per circuit depends on each derived class's implementation
    of the :method:`circuit_budget` method.  Goodness-of-fit quantities such as
    the log-likelihood or chi2 can utilize a `WildcardBudget` object to compute
    a value that shifts the circuit outcome probabilities within their allowed
    slack (so `|p_used - p_actual| <= slack`) to achieve the best goodness of
    fit.  For example, see the `wildcard` argument of :function:`two_delta_logl_terms`.

    This is a base class, which must be inherited from in order to obtain a
    full functional wildcard budge (the `circuit_budget` method must be
    implemented and usually `__init__` should accept more customized args).

    Parameters
    ----------
    w_vec : numpy.array
        vector of wildcard budget components.
    """

    def __init__(self, w_vec):
        """
        Create a new WildcardBudget.

        Parameters
        ----------
        w_vec : numpy array
            The "wildcard vector" which stores the parameters of this budget
            which can be varied when trying to find an optimal budget (similar
            to the parameters of a :class:`Model`).
        """
        self.wildcard_vector = w_vec

    def to_vector(self):
        """
        Get the parameters of this wildcard budget.

        Returns
        -------
        numpy array
        """
        return self.wildcard_vector

    def from_vector(self, w_vec):
        """
        Set the parameters of this wildcard budge.

        Parameters
        ----------
        w_vec : numpy array
            A vector of parameter values.

        Returns
        -------
        None
        """
        self.wildcard_vector = w_vec

    @property
    def num_params(self):
        """
        The number of parameters of this wildcard budget.

        Returns
        -------
        int
        """
        return len(self.wildcard_vector)

    def circuit_budget(self, circuit):
        """
        Get the amount of wildcard budget, or "outcome-probability-slack" for `circuit`.

        Parameters
        ----------
        circuit : Circuit
            the circuit to get the budget for.

        Returns
        -------
        float
        """
        raise NotImplementedError("Derived classes must implement `circuit_budget`")

    def circuit_budgets(self, circuits, precomp=None):
        """
        Get the wildcard budgets for a list of circuits.

        Parameters
        ----------
        circuits : list
            The list of circuits to act on.

        precomp : numpy.ndarray, optional
            A precomputed quantity that speeds up the computation of circuit
            budgets.  Given by :method:`precompute_for_same_circuits`.

        Returns
        -------
        numpy.ndarray
        """
        # XXX is this supposed to do something?
        # circuit_budgets = [self.circuit_budget(circ) for circ in circuits]
        pass

    @property
    def description(self):
        """
        A dictionary of quantities describing this budget.

        Return the contents of this budget in a dictionary containing
        (description, value) pairs for each element name.

        Returns
        -------
        dict
        """
        raise NotImplementedError("Derived classes must implement `description`")

    #def compute_circuit_wildcard_budget(c, w_vec):
    #    #raise NotImplementedError("TODO!!!")
    #    #for now, assume w_vec is a length-1 vector
    #    return abs(w_vec[0]) * len(c)

    def precompute_for_same_circuits(self, circuits):
        """
        Compute a pre-computed quantity for speeding up circuit calculations.

        This value can be passed to `update_probs` or `circuit_budgets` whenever this
        same `circuits` list is passed to `update_probs` to speed things up.

        Parameters
        ----------
        circuits : list
            A list of :class:`Circuit` objects.

        Returns
        -------
        object
        """
        raise NotImplementedError("Derived classes must implement `precompute_for_same_circuits`")

    def slow_update_probs(self, probs_in, probs_out, freqs, layout, precomp=None):
        """
        Updates `probs_in` to `probs_out` by applying this wildcard budget.

        Update a set of circuit outcome probabilities, `probs_in`, into a
        corresponding set, `probs_out`, which uses the slack alloted to each
        outcome probability to match (as best as possible) the data frequencies
        in `freqs`.  In particular, it computes this best-match in a way that
        maximizes the likelihood between `probs_out` and `freqs`. This method is
        the core function of a :class:`WildcardBudget`.

        Parameters
        ----------
        probs_in : numpy array
            The input probabilities, usually computed by a :class:`Model`.

        probs_out : numpy array
            The output probabilities: `probs_in`, adjusted according to the
            slack allowed by this wildcard budget, in order to maximize
            `logl(probs_out, freqs)`.  Note that `probs_out` may be the same
            array as `probs_in` for in-place updating.

        freqs : numpy array
            An array of frequencies corresponding to each of the
            outcome probabilites in `probs_in` or `probs_out`.

        layout : CircuitOutcomeProbabilityArrayLayout
            The layout for `probs_in`, `probs_out`, and `freqs`, specifying how array
            indices correspond to circuit outcomes, as well as the list of circuits.

        precomp : numpy.ndarray, optional
            A precomputed quantity for speeding up this calculation.

        Returns
        -------
        None
        """

        #Special case where f_k=0, since ratio is ill-defined. One might think
        # we shouldn't don't bother wasting any TVD on these since the corresponding
        # p_k doesn't enter the likelihood. ( => treat these components as if f_k == q_k (ratio = 1))
        # BUT they *do* enter in poisson-picture logl...
        # so set freqs very small so ratio is large (and probably not chosen)

        for i, circ in enumerate(layout.circuits):
            elInds = layout.indices_for_index(i)
            qvec = probs_in[elInds]
            fvec = freqs[elInds]
            W = self.circuit_budget(circ)

            #print("Circuit %d: %s" % (i, circ))
            #print(" inds = ", elInds, "q = ", qvec, " f = ", fvec)

            updated_qvec = update_circuit_probs(qvec, fvec, W)
            _tools.matrixtools._fas(probs_out, (elInds,), updated_qvec)

        return

    def precompute_for_same_probs_freqs(self, probs_in, freqs, layout):
        tol = 1e-8  # for checking equality - same as in update_probs
        tvd_precomp = 0.5 * _np.abs(probs_in - freqs)
        A_precomp = _np.logical_and(probs_in > freqs + tol, freqs > 0)
        B_precomp = _np.logical_and(probs_in < freqs - tol, freqs > 0)
        C_precomp = _np.logical_and(freqs - tol <= probs_in, probs_in <= freqs + tol)  # freqs == probs
        D_precomp = _np.logical_and(~C_precomp, freqs == 0)  # probs_in != freqs and freqs == 0
        circuits = layout.circuits

        precomp_info = []

        for i, circ in enumerate(circuits):
            elInds = layout.indices_for_index(i)
            fvec = freqs[elInds]
            qvec = probs_in[elInds]

            initialTVD = sum(tvd_precomp[elInds])  # 0.5 * sum(_np.abs(qvec - fvec))

            A = A_precomp[elInds]
            B = B_precomp[elInds]
            C = C_precomp[elInds]
            D = D_precomp[elInds]
            sum_fA = _np.sum(fvec[A])
            sum_fB = _np.sum(fvec[B])
            sum_qA = _np.sum(qvec[A])
            sum_qB = _np.sum(qvec[B])
            sum_qC = _np.sum(qvec[C])
            sum_qD = _np.sum(qvec[D])

            min_qvec = _np.min(qvec)

            # sort(abs(qvec[A] / fvec[A] - 1.0)) but abs and 1.0 irrelevant since ratio is always > 1
            iA = sorted(zip(_np.nonzero(A)[0], qvec[A] / fvec[A]), key=lambda x: x[1])

            # sort(abs(1.0 - qvec[B] / fvec[B])) but abs and 1.0 irrelevant since ratio is always < 1
            iB = sorted(zip(_np.nonzero(B)[0], qvec[B] / fvec[B]), key=lambda x: -x[1])

            precomp_info.append((A, B, C, D, sum_fA, sum_fB, sum_qA, sum_qB, sum_qC, sum_qD,
                                 initialTVD, fvec, qvec, min_qvec, iA, iB))

        return precomp_info

    def update_probs(self, probs_in, probs_out, freqs, layout, precomp=None, probs_freqs_precomp=None,
                     return_deriv=False):
        """
        Updates `probs_in` to `probs_out` by applying this wildcard budget.

        Update a set of circuit outcome probabilities, `probs_in`, into a
        corresponding set, `probs_out`, which uses the slack alloted to each
        outcome probability to match (as best as possible) the data frequencies
        in `freqs`.  In particular, it computes this best-match in a way that
        maximizes the likelihood between `probs_out` and `freqs`. This method is
        the core function of a :class:`WildcardBudget`.

        Parameters
        ----------
        probs_in : numpy array
            The input probabilities, usually computed by a :class:`Model`.

        probs_out : numpy array
            The output probabilities: `probs_in`, adjusted according to the
            slack allowed by this wildcard budget, in order to maximize
            `logl(probs_out, freqs)`.  Note that `probs_out` may be the same
            array as `probs_in` for in-place updating.

        freqs : numpy array
            An array of frequencies corresponding to each of the
            outcome probabilites in `probs_in` or `probs_out`.

        layout : CircuitOutcomeProbabilityArrayLayout
            The layout for `probs_in`, `probs_out`, and `freqs`, specifying how array
            indices correspond to circuit outcomes, as well as the list of circuits.

        precomp : numpy.ndarray, optional
            A precomputed quantity for speeding up this calculation.

        probs_freqs_precomp : list, optional
            A precomputed list of quantities re-used when calling `update_probs`
            using the same `probs_in`, `freqs`, and `layout`.  Generate by calling
            :method:`precompute_for_same_probs_freqs`.

        return_deriv : bool, optional
            When True, returns the derivative of each updated probability with
            respect to its circuit budget as a numpy array.  Useful for internal
            methods.

        Returns
        -------
        None
        """

        #Note: special case where f_k=0, since ratio is ill-defined. One might think
        # we shouldn't don't bother wasting any TVD on these since the corresponding
        # p_k doesn't enter the likelihood. ( => treat these components as if f_k == q_k (ratio = 1))
        # BUT they *do* enter in poisson-picture logl...
        # so set freqs very small so ratio is large (and probably not chosen)
        tol = 1e-8  # for checking equality
        circuits = layout.circuits
        circuit_budgets = self.circuit_budgets(circuits, precomp)
        p_deriv = _np.empty(layout.num_elements, 'd')

        if probs_freqs_precomp is None:
            probs_freqs_precomp = self.precompute_for_same_probs_freqs(probs_in, freqs, layout)

        for i, (circ, W, info) in enumerate(zip(circuits, circuit_budgets, probs_freqs_precomp)):
            A, B, C, D, sum_fA, sum_fB, sum_qA, sum_qB, sum_qC, sum_qD, initialTVD, fvec, qvec, min_qvec, iA, iB = info

            elInds = layout.indices_for_index(i)

            if initialTVD <= W + tol:  # TVD is already "in-budget" for this circuit - can adjust to fvec exactly
                probs_out[elInds] = fvec  # _tools.matrixtools._fas(probs_out, (elInds,), fvec)
                continue

            if min_qvec < 0:
                #Stopgap solution when a probability is negative: use wcbudget to move as
                # much negative prob to zero as possible, while reducing all the positive
                # probs.  This seems reasonable but isn't provably the right thing to do!
                qvec = qvec.copy()  # make sure we don't mess with memory we shouldn't
                neg_inds = _np.where(qvec < 0)[0]; neg_sum = sum(qvec[neg_inds])
                pos_inds = _np.where(qvec > 0)[0]; pos_sum = sum(qvec[pos_inds])  # note: NOT >= (leave zeros alone)
                if -neg_sum > pos_sum:
                    raise NotImplementedError(("Wildcard budget cannot be applied when the model predicts more "
                                               "*negative* then positive probability! (%s predicts neg_sum=%.3g, "
                                               "pos_sum=%.3g)") % (circ.str, neg_sum, pos_sum))
                while _np.min(qvec) < 0 and not _np.isclose(W, 0):
                    add_to = _np.argmin(qvec)
                    subtract_from = _np.argmax(qvec)
                    amount = _np.min([qvec[subtract_from], -qvec[add_to], W])
                    qvec[add_to] += amount
                    qvec[subtract_from] -= amount
                    W -= amount

                #recompute A-D b/c we've updated qvec
                A = _np.logical_and(qvec > fvec, fvec > 0); sum_fA = sum(fvec[A]); sum_qA = sum(qvec[A])
                B = _np.logical_and(qvec < fvec, fvec > 0); sum_fB = sum(fvec[B]); sum_qB = sum(qvec[B])
                C = (qvec == fvec); sum_qC = sum(qvec[C])
                D = _np.logical_and(qvec != fvec, fvec == 0); sum_qD = sum(qvec[D])

            indices_moved_to_C = []; alist_ptr = blist_ptr = 0

            while alist_ptr < len(iA):  # and sum_fA > 0
                # if alist_ptr < len(iA):  # always true when sum_fA > 0
                jA, alphaA = iA[alist_ptr]
                betaA = (1.0 - alphaA * sum_fA - sum_qC) / sum_fB
                testA = min(alphaA - 1.0, 1.0 - betaA)

                #if blist_ptr < len(iB):  # also always true
                assert(sum_fB > 0)  # sum_fB should always be > 0 - otherwise there's nowhere to take probability from
                jB, betaB = iB[blist_ptr]
                alphaB = (1.0 - betaB * sum_fB - sum_qC) / sum_fA
                testB = min(alphaB - 1.0, 1.0 - betaB)

                #Note: pushedSD = 0.0
                if testA < testB:
                    j, alpha_break, beta_break = jA, alphaA, betaA
                    TVD_at_breakpt = 0.5 * (sum_qA - alpha_break * sum_fA
                                            + beta_break * sum_fB - sum_qB
                                            + sum_qD)  # - pushedSD)  # compute_tvd
                    if TVD_at_breakpt <= W + tol: break  # exit loop

                    # move j from A -> C
                    sum_qA -= qvec[j]; sum_qC += qvec[j]; sum_fA -= fvec[j]
                    alist_ptr += 1
                else:
                    j, alpha_break, beta_break = jB, alphaB, betaB
                    TVD_at_breakpt = 0.5 * (sum_qA - alpha_break * sum_fA
                                            + beta_break * sum_fB - sum_qB
                                            + sum_qD)  # - pushedSD)  # compute_tvd
                    if TVD_at_breakpt <= W + tol: break  # exit loop

                    # move j from B -> C
                    sum_qB -= qvec[j]; sum_qC += qvec[j]; sum_fB -= fvec[j]
                    blist_ptr += 1
                indices_moved_to_C.append(j)

            else:  # if we didn't break due to TVD being small enough, continue to process with empty A-list:

                while blist_ptr < len(iB):  # now sum_fA == 0 => alist_ptr is maxed out
                    assert(sum_fB > 0)  # otherwise there's nowhere to take probability from
                    j, beta_break = iB[blist_ptr]
                    pushedSD = 1.0 - beta_break * sum_fB - sum_qC  # just used for TVD calc below

                    TVD_at_breakpt = 0.5 * (sum_qA + beta_break * sum_fB - sum_qB
                                            + sum_qD - pushedSD)  # compute_tvd

                    if TVD_at_breakpt <= W + tol: break  # exit loop

                    # move j from B -> C
                    sum_qB -= qvec[j]; sum_qC += qvec[j]; sum_fB -= fvec[j]
                    blist_ptr += 1
                    indices_moved_to_C.append(j)
                else:
                    assert(False), "TVD should reach zero: qvec=%s, fvec=%s, W=%g" % (str(qvec), str(fvec), W)

            #Now A,B,C are fixed to what they need to be for our given W
            # test if len(A) > 0, make tol here *smaller* than that assigned to zero freqs above
            if sum_fA > tol:
                alpha = (sum_qA - sum_qB + sum_qD - 2 * W) / sum_fA if sum_fB == 0 else \
                    (sum_qA - sum_qB + sum_qD + 1.0 - sum_qC - 2 * W) / (2 * sum_fA)  # compute_alpha
                beta = _np.nan if sum_fB == 0 else (1.0 - alpha * sum_fA - sum_qC) / sum_fB  # beta_fn
                pushedSD = 0.0  # assume initially that we don't need to push any TVD into the "D" set

                dalpha_dW = -2 / sum_fA if sum_fB == 0 else -1 / sum_fA
                dbeta_dW = 0.0 if sum_fB == 0 else (- dalpha_dW * sum_fA) / sum_fB
                dpushedSD_dW = 0.0
            else:  # fall back to this when len(A) == 0
                beta = -(sum_qA - sum_qB + sum_qD + sum_qC - 1 - 2 * W) / (2 * sum_fB) if sum_fA == 0 else \
                    -(sum_qA - sum_qB + sum_qD - 1.0 + sum_qC - 2 * W) / (2 * sum_fB)
                # compute_beta (assumes pushedSD can be >0)
                #beta = -(sum_qA - sum_qB + sum_qD - 2 * W) / sum_fB  # assumes pushedSD == 0
                alpha = 0.0  # doesn't matter OLD: _alpha_fn(beta, A, B, C, qvec, fvec)
                pushedSD = 1 - beta * sum_fB - sum_qC

                dalpha_dW = 0.0
                dbeta_dW = 2 / sum_fB if sum_fA == 0 else 1 / sum_fB
                dpushedSD_dW = -dbeta_dW * sum_fB

            #compute_pvec
            pvec = fvec.copy()
            pvec[A] = alpha * fvec[A]
            pvec[B] = beta * fvec[B]
            pvec[C] = qvec[C]
            #indices_moved_to_C = [x[0] for x in sorted_indices_and_ratios[0:nMovedToC]]
            pvec[indices_moved_to_C] = qvec[indices_moved_to_C]

            pvec[D] = pushedSD * qvec[D] / sum_qD
            probs_out[elInds] = pvec  # _tools.matrixtools._fas(probs_out, (elInds,), pvec)

            assert(W > 0 or _np.linalg.norm(qvec - pvec) < 1e-6), "Probability shouldn't be updated when W=0!"

            #Check with other version (for debugging)
            #check_pvec = update_circuit_probs(qvec.copy(), fvec.copy(), W)
            #assert(_np.linalg.norm(check_pvec - pvec) < 1e-6)

            if return_deriv:
                p_deriv_wrt_W = _np.zeros(len(pvec), 'd')
                p_deriv_wrt_W[A] = dalpha_dW * fvec[A]
                p_deriv_wrt_W[B] = dbeta_dW * fvec[B]
                p_deriv_wrt_W[indices_moved_to_C] = 0.0
                p_deriv_wrt_W[D] = dpushedSD_dW * qvec[D] / sum_qD
                p_deriv[elInds] = p_deriv_wrt_W

        return p_deriv if return_deriv else None


class PrimitiveOpsWildcardBudget(WildcardBudget):
    """
    A wildcard budget containing one parameter per "primitive operation".

    A parameter's absolute value gives the amount of "slack", or
    "wildcard budget" that is allocated per that particular primitive
    operation.

    Primitive operations are the components of circuit layers, and so
    the wilcard budget for a circuit is just the sum of the (abs vals of)
    the parameters corresponding to each primitive operation in the circuit.

    Parameters
    ----------
    primitive_op_labels : iterable or dict
        A list of primitive-operation labels, e.g. `Label('Gx',(0,))`,
        which give all the possible primitive ops (components of circuit
        layers) that will appear in circuits.  Each one of these operations
        will be assigned it's own independent element in the wilcard-vector.
        A dictionary can be given whose keys are Labels and whose values are
        0-based parameter indices.  In the non-dictionary case, each label gets
        it's own parameter.  Dictionaries allow multiple labels to be associated
        with the *same* wildcard budget parameter,
        e.g. `{Label('Gx',(0,)): 0, Label('Gy',(0,)): 0}`.
        If `'SPAM'` is included as a primitive op, this value correspond to a
        uniform "SPAM budget" added to each circuit.

    start_budget : float or dict, optional
        An initial value to set all the parameters to (if a float), or a
        dictionary mapping primitive operation labels to initial values.
    """

    def __init__(self, primitive_op_labels, start_budget=0.0, idle_name=None):
        """
        Create a new PrimitiveOpsWildcardBudget.

        Parameters
        ----------
        primitive_op_labels : iterable or dict
            A list of primitive-operation labels, e.g. `Label('Gx',(0,))`,
            which give all the possible primitive ops (components of circuit
            layers) that will appear in circuits.  Each one of these operations
            will be assigned it's own independent element in the wilcard-vector.
            A dictionary can be given whose keys are Labels and whose values are
            0-based parameter indices.  In the non-dictionary case, each label gets
            it's own parameter.  Dictionaries allow multiple labels to be associated
            with the *same* wildcard budget parameter,
            e.g. `{Label('Gx',(0,)): 0, Label('Gy',(0,)): 0}`.
            If `'SPAM'` is included as a primitive op, this value correspond to a
            uniform "SPAM budget" added to each circuit.

        start_budget : float or dict, optional
            An initial value to set all the parameters to (if a float), or a
            dictionary mapping primitive operation labels to initial values.

        idle_name : str, optional
            The gate name to be used for the 1-qubit idle gate.  If not `None`, then
            circuit budgets are computed by considering layers of the circuit as being
            "padded" with `1-qubit` idles gates on any empty lines.

        """
        if isinstance(primitive_op_labels, dict):
            assert(set(primitive_op_labels.values()) == set(range(len(set(primitive_op_labels.values())))))
            self.primOpLookup = primitive_op_labels
        else:
            self.primOpLookup = {lbl: i for i, lbl in enumerate(primitive_op_labels)}

        if 'SPAM' in self.primOpLookup:
            self.spam_index = self.primOpLookup['SPAM']
        else:
            self.spam_index = None

        self._idlename = idle_name

        nParams = len(set(self.primOpLookup.values()))
        if isinstance(start_budget, dict):
            Wvec = _np.zeros(nParams, 'd')
            for op, val in start_budget.items:
                Wvec[self.primOpLookup[op]] = val
        else:
            Wvec = _np.array([start_budget] * nParams)
        super(PrimitiveOpsWildcardBudget, self).__init__(Wvec)

    def circuit_budget(self, circuit):
        """
        Get the amount of wildcard budget, or "outcome-probability-slack" for `circuit`.

        Parameters
        ----------
        circuit : Circuit
            the circuit to get the budget for.

        Returns
        -------
        float
        """
        def budget_for_label(lbl):
            if lbl in self.primOpLookup:  # Note: includes len(lbl.components) == 0 case of (global) idle
                return pos(Wvec[self.primOpLookup[lbl]])
            elif lbl.name in self.primOpLookup:
                return pos(Wvec[self.primOpLookup[lbl.name]])
            else:
                assert(not lbl.is_simple()), "Simple label %s must be a primitive op of this WEB!" % str(lbl)
                return sum([budget_for_label(component) for component in lbl.components])

        Wvec = self.wildcard_vector
        budget = 0 if (self.spam_index is None) else pos(Wvec[self.spam_index])
        layers = [circuit.layer_label(i) for i in range(circuit.depth)] if (self._idlename is None) \
            else [circuit.layer_label_with_idles(i, idle_gate_name=self._idlename) for i in range(circuit.depth)]
        for layer in layers:
            budget += budget_for_label(layer)
        return budget

    def circuit_budgets(self, circuits, precomp=None):
        """
        Get the wildcard budgets for a list of circuits.

        Parameters
        ----------
        circuits : list
            The list of circuits to act on.

        precomp : numpy.ndarray, optional
            A precomputed quantity that speeds up the computation of circuit
            budgets.  Given by :method:`precompute_for_same_circuits`.

        Returns
        -------
        numpy.ndarray
        """
        if precomp is None:
            circuit_budgets = _np.array([self.circuit_budget(circ) for circ in circuits])
        else:
            Wvec = _np.abs(self.wildcard_vector)
            circuit_budgets = _np.dot(precomp, Wvec)
        return circuit_budgets

    def precompute_for_same_circuits(self, circuits):
        """
        Compute a pre-computed quantity for speeding up circuit calculations.

        This value can be passed to `update_probs` or `circuit_budgets` whenever this
        same `circuits` list is passed to `update_probs` to speed things up.

        Parameters
        ----------
        circuits : list
            A list of :class:`Circuit` objects.

        Returns
        -------
        object
        """
        def budget_deriv_for_label(lbl):
            if lbl in self.primOpLookup:  # Note: includes len(lbl.components) == 0 case of (global) idle
                deriv = _np.zeros(len(self.wildcard_vector), 'd')
                deriv[self.primOpLookup[lbl]] = 1.0
                return deriv
            elif lbl.name in self.primOpLookup:
                deriv = _np.zeros(len(self.wildcard_vector), 'd')
                deriv[self.primOpLookup[lbl.name]] = 1.0
                return deriv
            else:
                assert(not lbl.is_simple()), "Simple label %s must be a primitive op of this WEB!" % str(lbl)
                return sum([budget_deriv_for_label(component) for component in lbl.components])

        circuit_budget_matrix = _np.zeros((len(circuits), len(self.wildcard_vector)), 'd')
        for i, circuit in enumerate(circuits):

            layers = [circuit.layer_label(i) for i in range(circuit.depth)] if (self._idlename is None) \
                else [circuit.layer_label_with_idles(i, idle_gate_name=self._idlename) for i in range(circuit.depth)]
            for layer in layers:
                circuit_budget_matrix[i, :] += budget_deriv_for_label(layer)

        if self.spam_index is not None:
            circuit_budget_matrix[:, self.spam_index] = 1.0

        return circuit_budget_matrix

    @property
    def description(self):
        """
        A dictionary of quantities describing this budget.

        Return the contents of this budget in a dictionary containing
        (description, value) pairs for each element name.

        Returns
        -------
        dict
            Keys are primitive op labels and values are (description_string, value) tuples.
        """
        wildcardDict = {}
        for lbl, index in self.primOpLookup.items():
            if lbl == "SPAM": continue  # treated separately below
            wildcardDict[lbl] = ('budget per each instance %s' % str(lbl), pos(self.wildcard_vector[index]))
        if self.spam_index is not None:
            wildcardDict['SPAM'] = ('uniform per-circuit SPAM budget', pos(self.wildcard_vector[self.spam_index]))
        return wildcardDict

    def budget_for(self, op_label):
        """
        Retrieve the budget amount correponding to primitive op `op_label`.

        This is just the absolute value of this wildcard budget's parameter
        that corresponds to `op_label`.

        Parameters
        ----------
        op_label : Label
            The operation label to extract a budget for.

        Returns
        -------
        float
        """
        return pos(self.wildcard_vector[self.primOpLookup[op_label]])

    def __str__(self):
        wildcardDict = {lbl: pos(self.wildcard_vector[index]) for lbl, index in self.primOpLookup.items()}
        return "Wildcard budget: " + str(wildcardDict)


#For these helper functions, see Robin's notes
def _compute_tvd(a, b, d, alpha, beta, pushedD, q, f):
    # TVD = 0.5 * (qA - alpha*SA + beta*SB - qB + qD - pushed_pd) = difference between p=[alpha|beta]*f and q
    # (no contrib from set C)
    pushed_pd = pushedD * q[d] / sum(q[d])  # vector that sums to pushedD and aligns with q[d]
    ret = 0.5 * (sum(q[a] - alpha * f[a]) + sum(beta * f[b] - q[b]) + sum(q[d] - pushed_pd))
    return ret


def _compute_alpha(a, b, c, d, tvd, q, f):
    # beta = (1-alpha*SA - qC)/SB
    # 2*tvd = qA - alpha*SA + [(1-alpha*SA - qC)/SB]*SB - qB + qD  (pushedSD == 0 b/c A is nonempty if we call this fn)
    # 2*tvd = qA - alpha(SA + SA) + (1-qC) - qB + qD
    # alpha = [ qA-qB+qD + (1-qC) - 2*tvd ] / 2*SA
    # But if SB == 0 then 2*tvd = qA - alpha*SA - qB + qD => alpha = (qA-qB+qD-2*tvd)/SA
    # Note: no need to deal with pushedSD > 0 since this only occurs when alpha is irrelevant.
    if sum(f[b]) == 0:
        return (sum(q[a]) - sum(q[b]) + sum(q[d]) - 2 * tvd) / sum(f[a])
    return (sum(q[a]) - sum(q[b]) + sum(q[d]) + 1.0 - sum(q[c]) - 2 * tvd) / (2 * sum(f[a]))


def _compute_beta(a, b, c, d, tvd, q, f):
    # alpha = (1-beta*SB - qC)/SA
    # 2*tvd = qA - [(1-beta*SB - qC)/SA]*SA + beta*SB - qB + qD  (assume pushedD == 0)
    # 2*tvd = qA - (1-qC) + beta(SB + SB) - qB + qD
    # beta = -[ qA-qB+qD - (1-qC) - 2*tvd ] / 2*SB
    # But if SA == 0 then some probability may be "pushed" into set D:
    # 2*tvd = qA + (beta*SB - qB) + (qD - pushed_pD) and pushed_pD = 1 - beta * SB - qC, so
    # 2*tvd = qA + (beta*SB - qB) + (qD - 1 + beta*SB + qC) = qA - qB + qD +qC -1 + 2*beta*SB
    #  => beta = -(qA-qB+qD+qC-1-2*tvd)/(2*SB)
    if sum(f[a]) == 0:
        return -(sum(q[a]) - sum(q[b]) + sum(q[d]) + sum(q[c]) - 1 - 2 * tvd) / (2 * sum(f[b]))
    return -(sum(q[a]) - sum(q[b]) + sum(q[d]) - 1.0 + sum(q[c]) - 2 * tvd) / (2 * sum(f[b]))


def _compute_pvec(alpha, beta, pushedD, a, b, c, d, q, f):
    p = f.copy()
    #print("Fill pvec alpha=%g, beta=%g" % (alpha,beta))
    #print("f = ",f, " a = ",a, "b=",b," c=",c)
    p[a] = alpha * f[a]
    p[b] = beta * f[b]
    p[c] = q[c]
    p[d] = pushedD * q[d] / sum(q[d])
    return p


def _alpha_fn(beta, a, b, c, q, f, empty_val=1.0):
    # Note: this function is for use before we shift "D" set of f == 0 probs, and assumes all probs in set D are 0
    if len(a) == 0: return empty_val
    return (1.0 - beta * sum(f[b]) - sum(q[c])) / sum(f[a])


def _beta_fn(alpha, a, b, c, q, f, empty_val=1.0):
    # Note: this function is for use before we shift "D" set of f == 0 probs, and assumes all probs in set D are 0
    # beta * SB = 1 - alpha * SA - qC   => 1 = alpha*SA + beta*SB + qC (probs sum to 1)
    # also though, beta must be > 0 so (alpha*SA + qC) < 1.0
    if len(b) == 0: return empty_val
    return (1.0 - alpha * sum(f[a]) - sum(q[c])) / sum(f[b])


def _pushedD_fn(beta, b, c, q, f):  # The sum of additional TVD that gets "pushed" into set D
    # 1 = alpha*SA + beta*SB + qC + pushedSD => 1 = beta*SB + qC + pushedSD (probs sum to 1)
    return 1 - beta * sum(f[b]) - sum(q[c])


def _get_nextalpha_breakpoint(remaining_ratios):

    j = None; best_test = 1e10  # sentinel
    for jj, dct in remaining_ratios.items():
        test = min(dct['alpha'] - 1.0 if (dct['alpha'] is not None) else 1e10,
                   1.0 - dct['beta'] if (dct['beta'] is not None) else 1e10)
        if test < best_test:
            best_test, j = test, jj

    best_dct = remaining_ratios[j]
    alpha = best_dct['alpha'] if (best_dct['alpha'] is not None) else 1.0
    beta = best_dct['beta'] if (best_dct['beta'] is not None) else 1.0

    return j, alpha, beta, best_dct['typ']

#OLD REMOVE
#def _get_minalpha_breakpoint(remaining_ratios, a, b, c, qvec, fvec, ratio_vec):
    #k, r = sorted([(kx, rx) for kx, rx in enumerate(ratio_vec)
    #               if kx in remaining_indices], key=lambda x: abs(1.0 - x[1]))[0]
    #pushedSD = 0.0
    #
    #if k in a:
    #    alpha_break = r
    #    beta_break = _beta_fn(alpha_break, a, b, c, qvec, fvec)
    #    #print("alpha-break = %g -> beta-break = %g" % (alpha_break,beta_break))
    #    AorBorC = "A"
    #elif k in b:
    #    beta_break = r
    #    if len(a) > 0:
    #        alpha_break = _alpha_fn(beta_break, a, b, c, qvec, fvec)
    #
    #        if sum(fvec[a]) * alpha_break <= sum(qvec[a]):  # check if alpha_break gets set too large
    #            pushedSD = 0.0  # ok - alpha_break doesn't push sum_fA beyond its "target", sum_qA
    #        else:
    #            # because there must be some weight in set-D (zero-freq but nonzero prob)
    #            # the alpha as computed above will push sum_fA too much - we instead just
    #            # push sum_fA right to sum_qA and dump the rest into pushedSD
    #            alpha_break = sum(qvec[a]) / sum(fvec[a])
    #            pushedSD = 1.0 - (alpha_break * sum(fvec[a]) + beta_break * sum(fvec[b]) + sum(qvec[c]))
    #
    #        #print("beta-break = %g -> alpha-break = %g" % (beta_break,alpha_break))
    #    else:  # need to push set D to compensate
    #        alpha_break = 0.0  # value doesn't matter
    #        pushedSD = _pushedD_fn(beta_break, b, c, qvec, fvec)
    #    AorBorC = "B"
    #else:
    #    alpha_break = beta_break = 1e100  # sentinel so it gets sorted at end
    #    AorBorC = "C"
    ##print("chksum = ", _chk_sum(alpha_break, beta_break))
    #return (k, alpha_break, beta_break, pushedSD, AorBorC)


def _chk_sum(alpha, beta, fvec, A, B, C):
    return alpha * sum(fvec[A]) + beta * sum(fvec[B]) + sum(fvec[C])


def update_circuit_probs(probs, freqs, circuit_budget):
    qvec = probs
    fvec = freqs
    W = circuit_budget
    debug = False

    base_tol = 1e-8  # for checking for equality of qvec and fvec
    tol = len(qvec) * base_tol  # for checking if TVD is zero (e.g. when W==0 and TVD_at_breakpt is 1e-17)
    initialTVD = 0.5 * sum(_np.abs(qvec - fvec))
    if initialTVD <= W + tol:  # TVD is already "in-budget" for this circuit - can adjust to fvec exactly
        return fvec

    #Note: must ensure that A,B,C,D are *disjoint*
    fvec_equals_qvec = _np.logical_and(fvec - base_tol <= qvec, qvec <= fvec + base_tol)  # fvec == qvec
    A = _np.where(_np.logical_and(qvec > fvec + base_tol, fvec > 0))[0]
    B = _np.where(_np.logical_and(qvec < fvec - base_tol, fvec > 0))[0]
    C = _np.where(fvec_equals_qvec)[0]
    D = _np.where(_np.logical_and(~fvec_equals_qvec, fvec == 0))[0]

    if debug:
        print(" budget = ", W, " A=", A, " B=", B, " C=", C, " D=", D)

    ratio_vec = qvec / _np.where(fvec > 0, fvec, 1.0)  # avoid divide-by-zero warning (on sets C & D)
    ratio_vec[C] = _np.inf  # below we work in order of ratios distance
    ratio_vec[D] = _np.inf  # from 1.0 - and we don't want exactly-1.0 ratios.

    if debug: print("  Ratio vec = ", ratio_vec)

    #OLD: remaining_indices = list(range(len(ratio_vec)))

    # Set A: q > f != 0 alpha > 1.0 => p = alpha*f gets closer to q and p increases => logl increases
    # Set B: q < f  beta < 1.0 => p = beta*f gets closer to q and p decreases => logl decreases
    # Set C: q = f  requires no change (ever!)
    # Set D: q > f = 0  p=0 initially can be added to (to reduce TVD) but other p's must
    #                    decrease then: adding moves p closer to q and p increases => logl stays same
    # See that working on Set A is preferable to Set D since logl increases in the former but
    #  stays the same in the latter.  Once set A is exhausted, however, we should increase set D
    #  proportionally balance out movements from set B

    ratios = {}
    for j in A:
        ratios[j] = {'alpha': ratio_vec[j],
                     'beta': _beta_fn(ratio_vec[j], A, B, C, qvec, fvec, None),
                     'typ': 'A'}
    for j in B:
        ratios[j] = {'beta': ratio_vec[j],
                     'alpha': _alpha_fn(ratio_vec[j], A, B, C, qvec, fvec, None),
                     'typ': 'B'}

    while len(ratios) > 0:
        # find best next element
        j, alpha0, beta0, typ = _get_nextalpha_breakpoint(ratios)

        if len(A) == 0:  # no frequencies left to increase => qvec via alpha, so dump into pushedSD0
            pushedSD0 = 1.0 - beta0 * sum(fvec[B]) - sum(qvec[C])
        else:
            pushedSD0 = 0.0

        # will keep getting smaller with each iteration
        TVD_at_breakpt = _compute_tvd(A, B, D, alpha0, beta0, pushedSD0, qvec, fvec)
        #Note: does't matter if we move j from A or B -> C before calling this, as alpha0 is set so results is
        #the same

        if debug: print("break: j=", j, " alpha=", alpha0, " beta=",
                        beta0, " typ=", typ, " TVD = ", TVD_at_breakpt)
        if TVD_at_breakpt <= W + tol:
            break  # exit loop

        # move j from A/B -> C
        if typ == 'A':
            Alst = list(A); del Alst[Alst.index(j)]; A = _np.array(Alst, int)
            Clst = list(C); Clst.append(j); C = _np.array(Clst, int)  # move A -> C
        else:  # typ == 'B'
            Blst = list(B); del Blst[Blst.index(j)]; B = _np.array(Blst, int)
            Clst = list(C); Clst.append(j); C = _np.array(Clst, int)  # move B -> C

        #update ratios
        del ratios[j]
        for dct in ratios.values():
            if dct['typ'] == 'A':
                dct['beta'] = _beta_fn(dct['alpha'], A, B, C, qvec, fvec, None)
            else:  # dct['typ'] == 'B':
                dct['alpha'] = _alpha_fn(dct['beta'], A, B, C, qvec, fvec, None)
    else:
        assert(False), "TVD should eventually reach zero: qvec=%s, fvec=%s, W=%g" % (str(qvec), str(fvec), W)

    #OLD REMOVE
    #while len(remaining_indices) > 0:
    #    assert(len(A) > 0 or len(B) > 0)  # then we can step `alpha` up and preserve the overall probability:
    #    j, alpha0, beta0, pushedSD0, AorBorC = _get_minalpha_breakpoint(remaining_indices, A, B, C,
    #                                                                    qvec, fvec, ratio_vec)
    #    remaining_indices.remove(j)
    #
    #    # will keep getting smaller with each iteration
    #    TVD_at_breakpt = _compute_tvd(A, B, D, alpha0, beta0, pushedSD0, qvec, fvec)
    #    #Note: does't matter if we move j from A or B -> C before calling this, as alpha0 is set so results is
    #    #the same
    #
    #    if debug: print("break: j=", j, " alpha=", alpha0, " beta=",
    #                    beta0, " A?=", AorBorC, " TVD = ", TVD_at_breakpt)
    #    if TVD_at_breakpt <= W + tol:
    #        break  # exit loop
    #
    #    #Move
    #    if AorBorC == "A":
    #        if debug:
    #            beta_chk1 = _beta_fn(alpha0, A, B, C, qvec, fvec)
    #        Alst = list(A); del Alst[Alst.index(j)]; A = _np.array(Alst, int)
    #        Clst = list(C); Clst.append(j); C = _np.array(Clst, int)  # move A -> C
    #        if debug:
    #            beta_chk2 = _beta_fn(alpha0, A, B, C, qvec, fvec)
    #            print("CHKA: ", alpha0, beta0, beta_chk1, beta_chk2)
    #
    #    elif AorBorC == "B":
    #        if debug:
    #            alpha_chk1 = _alpha_fn(beta0, A, B, C, qvec, fvec)
    #        Blst = list(B); del Blst[Blst.index(j)]; B = _np.array(Blst, int)
    #        Clst = list(C); Clst.append(j); C = _np.array(Clst, int)  # move B -> C
    #        if debug:
    #            alpha_chk2 = _alpha_fn(beta0, A, B, C, qvec, fvec)
    #            print("CHKB: ", alpha0, beta0, alpha_chk1, alpha_chk2)
    #
    #    else:
    #        pass
    #
    #    if debug: TVD_at_breakpt_chk = _compute_tvd(A, B, D, alpha0, beta0, pushedSD0, qvec, fvec)
    #    if debug: print(" --> A=", A, " B=", B, " C=", C, " chk = ", TVD_at_breakpt_chk)
    #
    #else:
    #    assert(False), "TVD should eventually reach zero: qvec=%s, fvec=%s, W=%g" % (str(qvec), str(fvec), W)

    #Now A,B,C are fixed to what they need to be for our given W
    if debug: print("Final A=", A, "B=", B, "C=", C, "W=", W, "qvec=", qvec, 'fvec=', fvec)
    if len(A) > 0:
        alpha = _compute_alpha(A, B, C, D, W, qvec, fvec)
        beta = _beta_fn(alpha, A, B, C, qvec, fvec, _np.nan)
        pushedSD = 0.0
    else:  # fall back to this when len(A) == 0
        beta = _compute_beta(A, B, C, D, W, qvec, fvec)
        alpha = 0.0  # doesn't matter OLD: _alpha_fn(beta, A, B, C, qvec, fvec)
        pushedSD = _pushedD_fn(beta, B, C, qvec, fvec)

    if debug:
        print("Computed final alpha,beta = ", alpha, beta)
        print("CHECK SUM = ", _chk_sum(alpha, beta, fvec, A, B, C))
        print("DB: probs_in = ", qvec)
    updated_qvec = _compute_pvec(alpha, beta, pushedSD, A, B, C, D, qvec, fvec)
    if debug:
        print("DB: probs_out = ", updated_qvec)
    #print("TVD = ",compute_tvd(A,B,alpha,beta_fn(alpha,A,B,C,fvec),qvec,fvec))
    compTVD = _compute_tvd(A, B, D, alpha, beta, pushedSD, qvec, fvec)
    #print("compare: ",W,compTVD)
    assert(abs(W - compTVD) < 1e-3), "TVD mismatch!"
    #assert(_np.isclose(W, compTVD)), "TVD mismatch!"

    return updated_qvec
