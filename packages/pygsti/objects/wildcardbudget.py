"""Functions related to computation of the log-likelihood."""
from __future__ import division, print_function, absolute_import, unicode_literals
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


class WildcardBudget(object):
    """
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
    """

    def __init__(self, Wvec):
        """
        Create a new WildcardBudget.

        Parameters
        ----------
        Wvec : numpy array
            The "wildcard vector" which stores the parameters of this budget
            which can be varied when trying to find an optimal budget (similar
            to the parameters of a :class:`Model`).
        """
        self.wildcard_vector = Wvec

    def to_vector(self):
        """
        Get the parameters of this wildcard budget.

        Returns
        -------
        numpy array
        """
        return self.wildcard_vector

    def from_vector(self, Wvec):
        """
        Set the parameters of this wildcard budge.

        Parameters
        ----------
        Wvec : numpy array
            A vector of parameter values.

        Returns
        -------
        None
        """
        self.wildcard_vector = Wvec

    def circuit_budget(self, circuit):
        """
        Get the amount of wildcard budget, or "outcome-probability-slack"
        for `circuit`.

        Parameters
        ----------
        circuit : Circuit

        Returns
        -------
        float
        """
        raise NotImplementedError("Derived classes must implement `circuit_budget`")

    #def compute_circuit_wildcard_budget(c, Wvec):
    #    #raise NotImplementedError("TODO!!!")
    #    #for now, assume Wvec is a length-1 vector
    #    return abs(Wvec[0]) * len(c)

    def update_probs(self, probs_in, probs_out, freqs, circuits, elIndices):
        """
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

        circuits : list
            A list of :class:`Circuit` objects giving the circuits that
            `probs_in` contains the outcome probabilities of.  Typically
            there are multiple outcomes per circuit, so `len(circuits)`
            is less than `len(probs_in)` - see `elIndices` below.

        elIndices : list or numpy array
            A list of the element indices corresponding to each circuit in
            `circuits`.  Thus, `probs_in[elIndices[i]]` must give the
            probabilities corresponding to `circuits[i]`, and `elIndices[i]`
            can be any valid index for a numpy array (an integer, a slice,
            or an integer-array).  Similarly, `freqs[elIndices[i]]` gives
            the corresponding frequencies.

        Returns
        -------
        None
        """

        #For these helper functions, see Robin's notes
        def computeTVD(A, B, alpha, beta, q, f):
            ret = 0.5 * (sum(q[A] - alpha * f[A]) + sum(beta * f[B] - q[B]))
            return ret

        def compute_alpha(A, B, C, TVD, q, f):
            # beta = (1-alpha*SA - SC)/SB
            # 2*TVD = qA - alpha*SA + [(1-alpha*SA - SC)/SB]*SB - qB
            # 2*TVD = qA - alpha(SA + SA) + (1-SC) - qB
            # alpha = [ qA-qB + (1-SC) - 2*TVD ] / 2*SA
            return (sum(q[A]) - sum(q[B]) + 1.0 - sum(f[C]) - 2 * TVD) / (2 * sum(f[A]))

        def compute_beta(A, B, C, TVD, q, f):
            # alpha = (1-beta*SB - SC)/SA
            # 2*TVD = qA - [(1-beta*SB - SC)/SA]*SA + beta*SB - qB
            # 2*TVD = qA - (1-SC) + beta(SB + SB) - qB
            # beta = -[ qA-qB - (1-SC) - 2*TVD ] / 2*SB
            return -(sum(q[A]) - sum(q[B]) - 1.0 + sum(f[C]) - 2 * TVD) / (2 * sum(f[B]))

        def compute_pvec(alpha, beta, A, B, C, q, f):
            p = f.copy()
            #print("Fill pvec alpha=%g, beta=%g" % (alpha,beta))
            #print("f = ",f, " A = ",A, "B=",B," C=",C)
            p[A] = alpha * f[A]
            p[B] = beta * f[B]
            p[C] = q[C]
            return p

        def alpha_fn(beta, A, B, C, f):
            if len(A) == 0: return _np.nan  # this can be ok, but mark it
            return (1.0 - beta * sum(f[B]) - sum(f[C])) / sum(f[A])

        def beta_fn(alpha, A, B, C, f):
            if len(B) == 0: return _np.nan  # this can be ok, but mark it
            return (1.0 - alpha * sum(f[A]) - sum(f[C])) / sum(f[B])

        #Special case where f_k=0 - then don't bother wasting any TVD on
        # these since the corresponding p_k doesn't enter the likelihood.
        # => treat these components as if f_k == q_k (ratio = 1)
        zero_inds = _np.where(freqs == 0.0)[0]
        if len(zero_inds) > 0:
            freqs = freqs.copy()  # copy for now instead of doing something more clever
            freqs[zero_inds] = probs_in[zero_inds]

        for i, circ in enumerate(circuits):
            elInds = elIndices[i]
            #outLbls = outcomes_lookup[i] # needed?
            qvec = probs_in[elInds]
            fvec = freqs[elInds]
            W = self.circuit_budget(circ)

            initialTVD = 0.5 * sum(_np.abs(qvec - fvec))
            if initialTVD <= W:  # TVD is already "in-budget" for this circuit - can adjust to fvec exactly
                _tools.matrixtools._fas(probs_out, (elInds,), fvec)
                continue

            A = _np.where(qvec > fvec)[0]
            B = _np.where(qvec < fvec)[0]
            C = _np.where(qvec == fvec)[0]

            #print("Circuit %d: %s" % (i,circ))
            #print(" inds = ",elInds, "q = ",qvec, " f = ",fvec)
            #print(" budget = ",W, " A=",A," B=",B," C=",C)

            #Note: need special case for fvec == 0
            ratio_vec = qvec / fvec  # TODO: replace with more complex condition:
            #print("  Ratio vec = ", ratio_vec)

            breaks = []
            for k, r in enumerate(ratio_vec):
                if k in A:
                    alpha_break = r
                    beta_break = beta_fn(alpha_break, A, B, C, fvec)
                    #print("alpha-break = %g -> beta-break = %g" % (alpha_break,beta_break))
                    AorB = True
                elif k in B:
                    beta_break = r
                    alpha_break = alpha_fn(beta_break, A, B, C, fvec)
                    #print("beta-break = %g -> alpha-break = %g" % (beta_break,alpha_break))
                    AorB = False
                breaks.append((k, alpha_break, beta_break, AorB))
            #print("Breaks = ",breaks)

            sorted_breaks = sorted(breaks, key=lambda x: x[1])
            for j, alpha0, beta0, AorB in sorted_breaks:
                # will keep getting smaller with each iteration
                TVD_at_breakpt = computeTVD(A, B, alpha0, beta0, qvec, fvec)
                #Note: does't matter if we move j from A or B -> C before calling this, as alpha0 is set so results is
                #the same

                #print("break: j=",j," alpha=",alpha0," beta=",beta0," A?=",AorB, " TVD = ",TVD_at_breakpt)
                tol = 1e-6  # for instance, when W==0 and TVD_at_breakpt is 1e-17
                if TVD_at_breakpt <= W + tol:
                    break  # exit loop

                #Move
                if AorB:  # A
                    Alst = list(A); del Alst[Alst.index(j)]; A = _np.array(Alst, int)
                    Clst = list(C); Clst.append(j); C = _np.array(Clst, int)  # move A -> C
                else:  # B
                    Blst = list(B); del Blst[Blst.index(j)]; B = _np.array(Blst, int)
                    Clst = list(C); Clst.append(j); C = _np.array(Clst, int)  # move B -> C
                    #B.remove(j); C.add(j) # move A -> C
                #print(" --> A=",A," B=",B," C=",C)
            else:
                assert(False), "TVD should eventually reach zero (I think)!"

            #Now A,B,C are fixed to what they need to be for our given W
            if len(A) > 0:
                alpha = compute_alpha(A, B, C, W, qvec, fvec)
                beta = beta_fn(alpha, A, B, C, fvec)
            else:  # fall back to this when len(A) == 0
                beta = compute_beta(A, B, C, W, qvec, fvec)
                alpha = alpha_fn(beta, A, B, C, fvec)
            _tools.matrixtools._fas(probs_out, (elInds,), compute_pvec(alpha, beta, A, B, C, qvec, fvec))
            #print("TVD = ",computeTVD(A,B,alpha,beta_fn(alpha,A,B,C,fvec),qvec,fvec))
            compTVD = computeTVD(A, B, alpha, beta, qvec, fvec)
            #print("compare: ",W,compTVD)
            assert(abs(W - compTVD) < 1e-3), "TVD mismatch!"
            #assert(_np.isclose(W, compTVD)), "TVD mismatch!"

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
    """

    def __init__(self, primitiveOpLabels, start_budget=0.01):
        """
        Create a new PrimitiveOpsWildcardBudget.

        Parameters
        ----------
        primitiveOpLabels : iterable
            A list of primitive-operation labels, e.g. `Label('Gx',(0,))`,
            which give all the possible primitive ops (components of circuit
            layers) that will appear in circuits.  Each one of these operations
            will be assigned it's own independent element in the wilcard-vector.

        start_budget : float
            A rough initial value to set all the parameters to.  Some slight
            offset "noise" is also applied to better seed optimization of these
            parameters later on - this just gives a rough order of magnitude.
        """
        self.primOpLookup = {lbl: i for i, lbl in enumerate(primitiveOpLabels)}
        nPrimOps = len(self.primOpLookup)
        Wvec = _np.array([start_budget] * nPrimOps) + start_budget / 10.0 * \
            _np.arange(nPrimOps)  # 2nd term to slightly offset initial values
        super(PrimitiveOpsWildcardBudget, self).__init__(Wvec)

    def circuit_budget(self, circuit):
        """
        Get the amount of wildcard budget, or "outcome-probability-slack"
        for `circuit`.

        Parameters
        ----------
        circuit : Circuit

        Returns
        -------
        float
        """
        Wvec = self.wildcard_vector
        budget = 0
        for layer in circuit:
            for component in layer.components:
                budget += abs(Wvec[self.primOpLookup[component]])
        return budget

    def get_op_budget(self, op_label):
        """
        Retrieve the budget amount correponding to primitive op `op_label`.

        This is just the absolute value of this wildcard budget's parameter
        that corresponds to `op_label`.

        Parameters
        ----------
        op_label : Label

        Returns
        -------
        float
        """
        return abs(self.wildcard_vector[self.primOpLookup[op_label]])

    def __str__(self):
        wildcardDict = {lbl: abs(self.wildcard_vector[index]) for lbl, index in self.primOpLookup.items()}
        return "Wildcard budget: " + str(wildcardDict)
