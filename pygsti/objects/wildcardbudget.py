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

    def description(self):
        """
        A dictionary of quantities describing this budget.

        Return the contents of this budget in a dictionary containing
        (description, value) pairs for each element name.

        Returns
        -------
        dict
        """
        raise NotImplementedError("Derived classes must implement `to_descriptive_dict`")

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
        circuit_budget_matrix = _np.zeros((len(circuits), len(self.wildcard_vector)), 'd')
        for i, circuit in enumerate(circuits):
            for layer in circuit:
                for component in layer.components:
                    circuit_budget_matrix[i, self.primOpLookup[component]] += 1.0
        return circuit_budget_matrix

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
            A precmputed quantity for speeding up this calculation.

        Returns
        -------
        None
        """

        #For these helper functions, see Robin's notes
        def compute_tvd(a, b, alpha, beta, q, f):
            # TVD = 0.5 * (qA - alpha*SA + beta*SB - qB)  - difference between p=[alpha|beta]*f and q
            # (no contrib from set C)
            ret = 0.5 * (sum(q[a] - alpha * f[a]) + sum(beta * f[b] - q[b]))
            return ret

        def compute_alpha(a, b, c, tvd, q, f):
            # beta = (1-alpha*SA - qC)/SB
            # 2*tvd = qA - alpha*SA + [(1-alpha*SA - qC)/SB]*SB - qB
            # 2*tvd = qA - alpha(SA + SA) + (1-qC) - qB
            # alpha = [ qA-qB + (1-qC) - 2*tvd ] / 2*SA
            # But if SB == 0 then 2*tvd = qA - alpha*SA - qB => alpha = (qA-qB-2*tvd)/SA
            if sum(f[b]) == 0:
                return (sum(q[a]) - sum(q[b]) - 2 * tvd) / sum(f[a])
            return (sum(q[a]) - sum(q[b]) + 1.0 - sum(q[c]) - 2 * tvd) / (2 * sum(f[a]))

        def compute_beta(a, b, c, tvd, q, f):
            # alpha = (1-beta*SB - qC)/SA
            # 2*tvd = qA - [(1-beta*SB - qC)/SA]*SA + beta*SB - qB
            # 2*tvd = qA - (1-qC) + beta(SB + SB) - qB
            # beta = -[ qA-qB - (1-qC) - 2*tvd ] / 2*SB
            # But if SA == 0 then 2*tvd = qA + beta*SB - qB => beta = -(qA-qB-2*tvd)/SB
            if sum(f[a]) == 0:
                return -(sum(q[a]) - sum(q[b]) - 2 * tvd) / sum(f[b])
            return -(sum(q[a]) - sum(q[b]) - 1.0 + sum(q[c]) - 2 * tvd) / (2 * sum(f[b]))

        def compute_pvec(alpha, beta, a, b, c, q, f):
            p = f.copy()
            #print("Fill pvec alpha=%g, beta=%g" % (alpha,beta))
            #print("f = ",f, " a = ",a, "b=",b," c=",c)
            p[a] = alpha * f[a]
            p[b] = beta * f[b]
            p[c] = q[c]
            return p

        def alpha_fn(beta, a, b, c, q, f):
            if len(a) == 0: return _np.nan  # this can be ok, but mark it
            return (1.0 - beta * sum(f[b]) - sum(q[c])) / sum(f[a])

        def beta_fn(alpha, a, b, c, q, f):
            # beta * SB = 1 - alpha * SA - qC   => 1 = alpha*SA + beta*SB + qC (probs sum to 1)
            # also though, beta must be > 0 so (alpha*SA + qC) < 1.0
            if len(b) == 0: return _np.nan  # this can be ok, but mark it
            return (1.0 - alpha * sum(f[a]) - sum(q[c])) / sum(f[b])

        def get_minalpha_breakpoint(remaining_indices, a, b, c, qvec):
            k, r = sorted([(kx, rx) for kx, rx in enumerate(ratio_vec)
                           if kx in remaining_indices], key=lambda x: abs(1.0 - x[1]))[0]
            if k in a:
                alpha_break = r
                beta_break = beta_fn(alpha_break, a, b, c, qvec, fvec)
                #print("alpha-break = %g -> beta-break = %g" % (alpha_break,beta_break))
                AorBorC = "a"
            elif k in b:
                beta_break = r
                alpha_break = alpha_fn(beta_break, a, b, c, qvec, fvec)
                #print("beta-break = %g -> alpha-break = %g" % (beta_break,alpha_break))
                AorBorC = "b"
            else:
                alpha_break = beta_break = 1e100  # sentinel so it gets sorted at end
                AorBorC = "c"
            if debug: print("chksum = ", chk_sum(alpha_break, beta_break))
            return (k, alpha_break, beta_break, AorBorC)

        def chk_sum(alpha, beta):
            return alpha * sum(fvec[A]) + beta * sum(fvec[B]) + sum(fvec[C])

        #Special case where f_k=0, since ratio is ill-defined. One might think
        # we shouldn't don't bother wasting any TVD on these since the corresponding
        # p_k doesn't enter the likelihood. ( => treat these components as if f_k == q_k (ratio = 1))
        # BUT they *do* enter in poisson-picture logl...
        # so set freqs very small so ratio is large (and probably not chosen)
        zero_inds = _np.where(freqs == 0.0)[0]
        if len(zero_inds) > 0:
            freqs = freqs.copy()  # copy for now instead of doing something more clever
            freqs[zero_inds] = 1e-8
            #freqs[zero_inds] = probs_in[zero_inds]  # OLD (use this if f_k=0 terms don't enter likelihood)

        for i, circ in enumerate(layout.circuits):
            elInds = layout.indices_for_index(i)
            qvec = probs_in[elInds]
            fvec = freqs[elInds]
            W = self.circuit_budget(circ)

            tol = 1e-6  # for instance, when W==0 and TVD_at_breakpt is 1e-17
            initialTVD = 0.5 * sum(_np.abs(qvec - fvec))
            if initialTVD <= W + tol:  # TVD is already "in-budget" for this circuit - can adjust to fvec exactly
                _tools.matrixtools._fas(probs_out, (elInds,), fvec)
                continue

            A = _np.where(qvec > fvec)[0]
            B = _np.where(qvec < fvec)[0]
            C = _np.where(qvec == fvec)[0]

            debug = False  # (i == 827)

            if debug:
                print("Circuit %d: %s" % (i, circ))
                print(" inds = ", elInds, "q = ", qvec, " f = ", fvec)
                print(" budget = ", W, " A=", A, " B=", B, " C=", C)

            #Note: need special case for fvec == 0
            ratio_vec = qvec / fvec  # TODO: replace with more complex condition:
            if debug: print("  Ratio vec = ", ratio_vec)

            remaining_indices = list(range(len(ratio_vec)))

            while len(remaining_indices) > 0:
                j, alpha0, beta0, AorBorC = get_minalpha_breakpoint(remaining_indices, A, B, C, qvec)
                remaining_indices.remove(j)

                # will keep getting smaller with each iteration
                TVD_at_breakpt = compute_tvd(A, B, alpha0, beta0, qvec, fvec)
                #Note: does't matter if we move j from A or B -> C before calling this, as alpha0 is set so results is
                #the same

                if debug: print("break: j=", j, " alpha=", alpha0, " beta=",
                                beta0, " A?=", AorBorC, " TVD = ", TVD_at_breakpt)
                if TVD_at_breakpt <= W + tol:
                    break  # exit loop

                #Move
                if AorBorC == "A":
                    if debug:
                        beta_chk1 = beta_fn(alpha0, A, B, C, qvec, fvec)
                    Alst = list(A); del Alst[Alst.index(j)]; A = _np.array(Alst, int)
                    Clst = list(C); Clst.append(j); C = _np.array(Clst, int)  # move A -> C
                    if debug:
                        beta_chk2 = beta_fn(alpha0, A, B, C, qvec, fvec)
                        print("CHKA: ", alpha0, beta0, beta_chk1, beta_chk2)

                elif AorBorC == "B":
                    if debug:
                        alpha_chk1 = alpha_fn(beta0, A, B, C, qvec, fvec)
                    Blst = list(B); del Blst[Blst.index(j)]; B = _np.array(Blst, int)
                    Clst = list(C); Clst.append(j); C = _np.array(Clst, int)  # move B -> C
                    if debug:
                        alpha_chk2 = alpha_fn(beta0, A, B, C, qvec, fvec)
                        print("CHKB: ", alpha0, beta0, alpha_chk1, alpha_chk2)

                else:
                    pass

                if debug: TVD_at_breakpt_chk = compute_tvd(A, B, alpha0, beta0, qvec, fvec)
                if debug: print(" --> A=", A, " B=", B, " C=", C, " chk = ", TVD_at_breakpt_chk)

            else:
                assert(False), "TVD should eventually reach zero (I think)!"

            #Now A,B,C are fixed to what they need to be for our given W
            if debug: print("Final A=", A, "B=", B, "C=", C, "W=", W, "qvec=", qvec, 'fvec=', fvec)
            if len(A) > 0:
                alpha = compute_alpha(A, B, C, W, qvec, fvec)
                beta = beta_fn(alpha, A, B, C, qvec, fvec)
                if debug and len(B) > 0:
                    abeta = compute_beta(A, B, C, W, qvec, fvec)
                    aalpha = alpha_fn(beta, A, B, C, qvec, fvec)
                    print("ALT final alpha,beta = ", aalpha, abeta)
            else:  # fall back to this when len(A) == 0
                beta = compute_beta(A, B, C, W, qvec, fvec)
                alpha = alpha_fn(beta, A, B, C, qvec, fvec)
            if debug:
                print("Computed final alpha,beta = ", alpha, beta)
                print("CHECK SUM = ", chk_sum(alpha, beta))
                print("DB: probs_in = ", probs_in[elInds])
            _tools.matrixtools._fas(probs_out, (elInds,), compute_pvec(alpha, beta, A, B, C, qvec, fvec))
            if debug:
                print("DB: probs_out = ", probs_out[elInds])
            #print("TVD = ",compute_tvd(A,B,alpha,beta_fn(alpha,A,B,C,fvec),qvec,fvec))
            compTVD = compute_tvd(A, B, alpha, beta, qvec, fvec)
            #print("compare: ",W,compTVD)
            assert(abs(W - compTVD) < 1e-3), "TVD mismatch!"
            #assert(_np.isclose(W, compTVD)), "TVD mismatch!"

        return

    def update_probs(self, probs_in, probs_out, freqs, layout, precomp=None):
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
            A precmputed quantity for speeding up this calculation.

        Returns
        -------
        None
        """

        #Special case where f_k=0, since ratio is ill-defined. One might think
        # we shouldn't don't bother wasting any TVD on these since the corresponding
        # p_k doesn't enter the likelihood. ( => treat these components as if f_k == q_k (ratio = 1))
        # BUT they *do* enter in poisson-picture logl...
        # so set freqs very small so ratio is large (and probably not chosen)
        MIN_FREQ = 1e-8
        MIN_FREQ_OVER_2 = MIN_FREQ / 2
        zero_inds = _np.where(freqs == 0.0)[0]
        if len(zero_inds) > 0:
            freqs = freqs.copy()  # copy for now instead of doing something more clever
            freqs[zero_inds] = MIN_FREQ
            #freqs[zero_inds] = probs_in[zero_inds]  # OLD (use this if f_k=0 terms don't enter likelihood)

        circuits = layout.circuits
        circuit_budgets = self.circuit_budgets(circuits, precomp)
        tvd_precomp = 0.5 * _np.abs(probs_in - freqs)
        A_precomp = (probs_in > freqs)
        B_precomp = (probs_in < freqs)
        C_precomp = (probs_in == freqs)

        tol = 1e-6  # for instance, when W==0 and TVD_at_breakpt is 1e-17

        for i, (circ, W) in enumerate(zip(circuits, circuit_budgets)):
            elInds = layout.indices_for_index(i)
            fvec = freqs[elInds]
            qvec = probs_in[elInds]

            if _np.min(qvec) < 0:
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

                if -neg_sum < W:  # then there's enough budget to pay for all of negatives
                    alpha = -neg_sum / len(neg_inds) * 1.0 / qvec[pos_inds]  # sum(qvec[pos])*alpha=-neg_sum*sum(ones/N)
                    qvec[pos_inds] *= 1.0 - alpha
                    qvec[neg_inds] = 0.0
                    W -= (-neg_sum)
                else:
                    alpha = W / len(neg_inds) * 1.0 / qvec[pos_inds]  # sum(qvec[pos])*alpha = W * sum(ones/N)
                    beta = -W / len(neg_inds) * 1.0 / qvec[neg_inds]  # sum(beta*qvec[neg]) = -W * sum(ones/N)
                    qvec[pos_inds] *= 1.0 - alpha
                    qvec[neg_inds] *= 1.0 - beta
                    W = 0

            initialTVD = sum(tvd_precomp[elInds])  # 0.5 * sum(_np.abs(qvec - fvec))
            if initialTVD <= W + tol:  # TVD is already "in-budget" for this circuit - can adjust to fvec exactly
                probs_out[elInds] = fvec  # _tools.matrixtools._fas(probs_out, (elInds,), fvec)
                continue

            A = A_precomp[elInds]
            B = B_precomp[elInds]
            C = C_precomp[elInds]
            sum_fA = sum(fvec[A])
            sum_fB = sum(fvec[B])
            sum_qA = sum(qvec[A])
            sum_qB = sum(qvec[B])
            sum_qC = sum(qvec[C])

            #Note: need special case for fvec == 0
            ratio_vec = qvec / fvec
            # remaining_indices = list(range(len(ratio_vec)))
            sorted_indices_and_ratios = sorted(
                [(kx, rx) for kx, rx in enumerate(ratio_vec)], key=lambda x: abs(1.0 - x[1]))
            nMovedToC = 0

            #print("Circuit ",i, " indices_and_ratios = ",sorted_indices_and_ratios)

            for j, ratio in sorted_indices_and_ratios:

                if ratio > 1.0:  # j in A
                    alpha_break = ratio
                    beta_break = _np.nan if sum_fB == 0.0 else (1.0 - alpha_break * sum_fA - sum_qC) / sum_fB  # beta_fn

                    TVD_at_breakpt = 0.5 * (sum_qA - alpha_break * sum_fA + beta_break * sum_fB - sum_qB)  # compute_tvd
                    #print("A TVD at ",alpha_break,beta_break,"=",TVD_at_breakpt, "(ratio = ",ratio,")")
                    if TVD_at_breakpt <= W + tol: break  # exit loop

                    # move j from A -> C
                    sum_qA -= qvec[j]; sum_qC += qvec[j]; sum_fA -= fvec[j]
                elif ratio < 1.0:  # j in B
                    beta_break = ratio
                    alpha_break = _np.nan if sum_fA == 0.0 else (
                        1.0 - beta_break * sum_fB - sum_qC) / sum_fA  # alpha_fn

                    TVD_at_breakpt = 0.5 * (sum_qA - alpha_break * sum_fA + beta_break * sum_fB - sum_qB)  # compute_tvd
                    #print("B TVD at ",alpha_break,beta_break,"=",TVD_at_breakpt, "(ratio = ",ratio,")")
                    if TVD_at_breakpt <= W + tol: break  # exit loop

                    # move j from B -> C
                    sum_qB -= qvec[j]; sum_qC += qvec[j]; sum_fB -= fvec[j]

                else:  # j in C
                    #print("C TVD at ",alpha_break,beta_break,"=",TVD_at_breakpt, "(ratio = ",ratio,")")
                    pass  # (no movement, nothing happens)

                nMovedToC += 1
            else:
                assert(False), "TVD should eventually reach zero (I think)!"

            #Now A,B,C are fixed to what they need to be for our given W
            # test if len(A) > 0, make tol here *smaller* than that assigned to zero freqs above
            if sum_fA > MIN_FREQ_OVER_2:
                alpha = (sum_qA - sum_qB - 2 * W) / sum_fA if sum_fB == 0 else \
                    (sum_qA - sum_qB + 1.0 - sum_qC - 2 * W) / (2 * sum_fA)  # compute_alpha
                beta = _np.nan if sum_fB == 0 else (1.0 - alpha * sum_fA - sum_qC) / sum_fB  # beta_fn
            else:  # fall back to this when len(A) == 0
                beta = -(sum_qA - sum_qB - 2 * W) / sum_fB if sum_fA == 0 else \
                    -(sum_qA - sum_qB - 1.0 + sum_qC - 2 * W) / (2 * sum_fB)  # compute_beta
                alpha = _np.nan if sum_fA == 0 else (1.0 - beta * sum_fB - sum_qC) / sum_fA  # alpha_fn

            #compute_pvec
            pvec = fvec.copy()
            pvec[A] = alpha * fvec[A]
            pvec[B] = beta * fvec[B]
            pvec[C] = qvec[C]
            indices_moved_to_C = [x[0] for x in sorted_indices_and_ratios[0:nMovedToC]]
            pvec[indices_moved_to_C] = qvec[indices_moved_to_C]
            probs_out[elInds] = pvec  # _tools.matrixtools._fas(probs_out, (elInds,), pvec)

        return


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
    primitive_op_labels : iterable
        A list of primitive-operation labels, e.g. `Label('Gx',(0,))`,
        which give all the possible primitive ops (components of circuit
        layers) that will appear in circuits.  Each one of these operations
        will be assigned it's own independent element in the wilcard-vector.

    add_spam : bool, optional
        Whether an additional "SPAM" budget should be included, which is
        simply a uniform budget added to each circuit.

    start_budget : float, optional
        An initial value to set all the parameters to.
    """

    def __init__(self, primitive_op_labels, add_spam=True, start_budget=0.0):
        """
        Create a new PrimitiveOpsWildcardBudget.

        Parameters
        ----------
        primitive_op_labels : iterable
            A list of primitive-operation labels, e.g. `Label('Gx',(0,))`,
            which give all the possible primitive ops (components of circuit
            layers) that will appear in circuits.  Each one of these operations
            will be assigned it's own independent element in the wilcard-vector.

        add_spam : bool, optional
            Whether an additional "SPAM" budget should be included, which is
            simply a uniform budget added to each circuit.

        start_budget : float, optional
            An initial value to set all the parameters to.
        """
        self.primOpLookup = {lbl: i for i, lbl in enumerate(primitive_op_labels)}
        nPrimOps = len(self.primOpLookup)
        if add_spam:
            nPrimOps += 1
            self.spam_index = nPrimOps - 1  # last element is SPAM
        else:
            self.spam_index = None

        Wvec = _np.array([start_budget] * nPrimOps)
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
        Wvec = self.wildcard_vector
        budget = 0 if (self.spam_index is None) else pos(Wvec[self.spam_index])
        for layer in circuit:
            for component in layer.components:
                budget += pos(Wvec[self.primOpLookup[component]])
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
            off = 0 if (self.spam_index is None) else Wvec[self.spam_index]
            circuit_budgets = _np.dot(precomp, Wvec) + off
        return circuit_budgets

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
        if self.spam_index is not None: wildcardDict['SPAM'] = pos(self.wildcard_vector[self.spam_index])
        return "Wildcard budget: " + str(wildcardDict)
