#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
""" Core Idle Tomography routines """

import numpy as _np
import itertools as _itertools
import time as _time
import collections as _collections
import warnings as _warnings

from ... import construction as _cnst
from ... import objects as _objs
from ... import tools as _tools
from ...objects.verbosityprinter import VerbosityPrinter as _VerbosityPrinter

from . import pauliobjs as _pobjs
from . import idttools as _idttools
from .idtresults import IdleTomographyResults as _IdleTomographyResults

# This module implements idle tomography, which deals only with
# many-qubit idle gates (on some number of qubits) and single-
# qubit gates (or tensor products of them) used to fiducials.
# As such, it is conventient to represent operations as native
# Python strings, where there is one I,X,Y, or Z letter per
# qubit.


def hamiltonian_jac_element(prep, error, observable):
    """
    Computes the Jacobian matrix element for a Hamiltonian error: how the
    expectation value of `observable` in state `prep` changes due to
    Hamiltonian `error`.

    Parameters
    ----------
    prep : NQPauliState
        The state that is prepared.

    error : NQPauliOp
        The type (as a pauli operator) of Hamiltonian error.

    observable : NQPauliOp
        The observable whose expectation value is measured.
        Note: giving a NQPauliState will be treated as an
        N-qubit Pauli operator whose sign is the product of
        the signs of the NQPauliState's per-qubit basis signs.

    Returns
    -------
    float
    """
    # dp/deps where p = eps * i * Tr(Obs (Err*rho - rho*Err)) = eps * i * ( Tr(Obs Err rho) - Tr(Obs rho Err))
    #                 = eps * i * Tr([Obs,Err] * rho)  so dp/deps just drops eps factor
    com = error.icommutator_over_2(observable)
    return 0 if (com is None) else com.statedot(prep)


def stochastic_outcome(prep, error, meas):
    """
    Computes the "expected" outcome when the stochastic error `error`
    occurs between preparing in `prep` and measuring in basis `meas`.

    Note: currently, the preparation and measurement bases must be the
    same (up to signs) or an AssertionError is raised. (If they're not,
    there isn't a single expected outcome).

    Parameters
    ----------
    prep : NQPauliState
        The state that is prepared.

    error : NQPauliOp
        The type (as a pauli operator) of Stochastic error.

    meas : NQPauliState
        The basis which is measured.  The 'signs' of the basis
        Paulis determine which state is measured as a '0' vs. a '1'.
        (essentially the POVM.)

    Returns
    -------
    NQOutcome
    """
    # We can consider each qubit separately, since Tr(A x B) = Tr(A)Tr(B).
    # If for the i-th qubit the prep basis is s1*P and the meas basis is s2*P
    #  (s1 and s2 are the signs -- either +1 or -1 -- and P is the common
    #  Pauli whose eigenstates 1. form the measurement basis and 2. contain the
    #  state prep), then we're trying to sell which of:
    # Tr( (I+s2*P) Err (I+s1*P) ) ['+' or '0' outcome] OR
    # Tr( (I-s2*P) Err (I+s1*P) ) ['-' or '1' outcome] is nonzero.
    # Combining these two via use of '+-' and expanding gives:
    # Tr( Err + s1* Err P +- s2* P Err +- s1*s2* P Err P )
    #   assuming Err != I so Tr(Err) = 0 and Tr(P Err P) = 0 (b/c P either
    #   commutes or anticommutes w/Err and P^2 == I) =>
    # Tr( s1* Err P +- s2* P Err )
    # if [Err,P] = 0, then the '+'/'0' branch is nonzero when s1==s2
    #   and the '-'/'1' branch is nonzero when s1!=s2
    # if {Err,P} = 0, then the opposite is true: '+'/'0' branch is nonzero
    #   when s1!=s2, etc.
    # Takeaway: if basis (P) commutes with Err then outcome is '0' if s1==s2, "1" otherwise ...
    outcome_str = ""
    for s1, P1, s2, P2, Err in zip(prep.signs, prep.rep, meas.signs, meas.rep, error.rep):
        assert(P1 == P2), "Stochastic outcomes must prep & measure along same bases!"
        P = P1  # ( = P2)
        if _pobjs._commute_parity(P, Err) == 1:  # commutes: [P,Err] == 0
            outcome_str += "0" if (s1 == s2) else "1"
        else:  # anticommutes: {P,Err} == 0
            outcome_str += "1" if (s1 == s2) else "0"

    return _pobjs.NQOutcome(outcome_str)


# Now we can define the functions that do the real work for stochastic tomography.

# StochasticMatrixElement() computes the derivative of the probability of "Outcome" with respect
# to the rate of "Error" if the N-qubit Pauli basis defined by "PrepMeas" is prepped and measured.
def stochastic_jac_element(prep, error, meas, outcome):
    """
    Computes the Jacobian matrix element for a Stochastic error: how the
    probability of `outcome` changes with respect to the rate of `error`
    when preparing state `prep` and measuring in basis `meas`.

    Parameters
    ----------
    prep : NQPauliState
        The state that is prepared.

    error : NQPauliOp
        The type (as a pauli operator) of Stochastic error.

    meas : NQPauliState
        The basis that is measured (essentially the POVM).

    outcome : NQOutcome
        The measurement outcome that is considered.

    Returns
    -------
    float
    """
    return 1 if (stochastic_outcome(prep, error, meas) == outcome) else 0


def affine_jac_element(prep, error, meas, outcome):
    """
    Computes the Jacobian matrix element for a Affine error: how the
    probability of `outcome` changes with respect to the rate of `error`
    when preparing state `prep` and measuring in basis `meas`.

    Note: Affine error maps leave qubits corresponging to I's in
    `error` alone.  An affine error is defined as replacing
    portions of the density matrix corresponding to *non-trivial*
    Pauli operators with those operators.

    Parameters
    ----------
    prep : NQPauliState
        The state that is prepared.

    error : NQPauliOp
        The type (as a pauli operator) of Affine error.

    meas : NQPauliState
        The basis that is measured (essentially the POVM).

    outcome : NQOutcome
        The measurement outcome that is considered.

    Returns
    -------
    float
    """
    # Note an error of 'ZI' does *not* mean the "ZI affine error":
    #   rho -> (Id[rho] + eps*AffZI[rho]) = rho + eps*ZI
    #   where ZI = diag(1,1,-1,-1), so this adds prob to 00 and 01 and removes from 10 and 11.
    # Instead it means the map AffZ x Id where AffZ : rho -> rho + eps Z and Id : rho -> rho.

    def _affhelper(prep_sign, prep_basis, error_pauli, meas_sign, meas_basis, outcome_bit):
        """
        Answers this question:
        If a qubit is prepped in state (prep_sign,prep_basis) & measured
        using POVM (meas_sign,meas_basis), and experiences an affine error given
        (at this qubit) by Pauli "error_pauli", then at what rate does that change probability of outcome "bit"?
        This is going to get multiplied over all qubits.  A zero indicates that the affine error is orthogonal
        to the measurement basis, which means the probability of *all* outcomes including this bit are unaffected.

        Returns 0, +1, or -1.
        """
        # Specifically, this computes Tr( (I+/-P) AffErr[ (I+/-P) ] ) where the two
        # P's represent the prep & measure bases (and can be different).  Here AffErr
        # outputs ErrP if ErrP != 'I', otherwise it's just the identity map (see above).
        #
        # Thus, when ErrP != 'I', we have Tr( (I+/-P) ErrP ) which equals 0 whenever
        # ErrP != P and +/-1 if ErrP == P.  The sign equals meas_sign when outcome_bit == "0",
        # and is reversed when it == "1".
        # When ErrP == 'I', we have Tr( (I+/-P) (I+/-P) ) = Tr( I + sign*I)
        #  = 1 where sign = prep_sign*meas_sign when outcome == "0" and -1 times
        #      this when == "1".
        #  = 0 otherwise

        assert(prep_basis in ("X", "Y", "Z"))  # 'I', for instance, is invalid
        assert(meas_basis in ("X", "Y", "Z"))  # 'I', for instance, is invalid
        assert(prep_basis == meas_basis)  # always true
        outsign = 1 if (outcome_bit == "0") else -1  # b/c we often just flip a sign when == "1"
        # i.e. the sign used in I+/-P for measuring is meas_sign * outsign

        if error_pauli == 'I':  # special case: no affine action on this space
            if prep_basis == meas_basis:
                return 1 if (prep_sign * meas_sign * outsign == 1) else 0
            else: return 1  # bases don't match

        if meas_basis != error_pauli:  # then they don't commute (b/c neither can be 'I')
            return 0  # so there's no change along this axis (see docstring)
        else:  # meas_basis == error_pauli != 'I'
            if outcome_bit == "0": return meas_sign
            else: return meas_sign * -1

    return _np.prod([_affhelper(s1, P1, Err, s2, P2, o) for s1, P1, s2, P2, Err, o
                     in zip(prep.signs, prep.rep, meas.signs, meas.rep,
                            error.rep, outcome.rep)])


def affine_jac_obs_element(prep, error, observable):
    """
    Computes the Jacobian matrix element for a Affine error: how the
    expectation value of `observable` changes with respect to the rate of
    `error` when preparing state `prep`.

    Note: Affine error maps leave qubits corresponging to I's in
    `error` alone.  An affine error is defined as replacing
    portions of the density matrix corresponding to *non-trivial*
    Pauli operators with those operators.

    Parameters
    ----------
    prep : NQPauliState
        The state that is prepared.

    error : NQPauliOp
        The type (as a pauli operator) of Affine error.

    observable : NQPauliOp
        The observable whose expectation value is measured.

    Returns
    -------
    float
    """
    # Computes the Jacobian element of Tr(observable * error * prep) with basis
    # convention given by `meas` (dictates sign of outcome).
    # (observable should be equal to meas when it's not equal to 'I', up to sign)

    # Note: as in affine_jac_element, 'I's in error mean that this affine error
    # doesn't act (acts as the identity) on that qubit.

    def _affhelper(prep_sign, prep_basis, error_pauli, obs_pauli):
        assert(prep_basis in ("X", "Y", "Z"))  # 'I', for instance, is invalid

        # want Tr(obs_pauli * AffErr[ I+/-P ] ).  There are several cases:
        # 1) if obs_pauli == 'I':
        #   - if error_pauli == 'I' (so AffErr = Id), Tr(I +/- P) == 1 always
        #   - if error_pauli != 'I', Tr(ErrP) == 0 since ErrP != 'I'
        # 2) if obs_pauli != 'I' (so Tr(obs_pauli) == 0)
        #   - if error_pauli == 'I', Tr(obs_pauli * (I +/- P)) = prep_sign if (obs_pauli == prep_basis) else 0
        #   - if error_pauli != 'I', Tr(obs_pauli * error_pauli) = 1 if (obs_pauli == error_pauli) else 0
        #      (and actually this counts at 2 instead of 1 b/c obs isn't normalized (I think?))

        if obs_pauli == 'I':
            return 1 if (error_pauli == 'I') else 0
        elif error_pauli == 'I':
            return prep_sign if (prep_basis == obs_pauli) else 0
        else:
            return 2 if (obs_pauli == error_pauli) else 0

    return _np.prod([_affhelper(s1, P1, Err, o) for s1, P1, Err, o
                     in zip(prep.signs, prep.rep, error.rep, observable.rep)])


# -----------------------------------------------------------------------------
# Experiment generation:
# -----------------------------------------------------------------------------

def idle_tomography_fidpairs(nqubits, maxweight=2, include_hamiltonian=True,
                             include_stochastic=True, include_affine=True,
                             ham_tmpl="auto",
                             preferred_prep_basis_signs=("+", "+", "+"),
                             preferred_meas_basis_signs=("+", "+", "+")):
    """
    Construct a list of Pauli-basis fiducial pairs for idle tomography.

    This function constructs the "standard" set of fiducial pairs used
    to generate idle tomography sequences which probe Hamiltonian,
    Stochastic, and/or Affine errors in an idle gate.

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    maxweight : int, optional
        The maximum weight of errors to consider.

    include_hamiltonian, include_stochastic, include_affine : bool, optional
        Whether to include fiducial pairs for finding Hamiltonian-, Stochastic-,
        and Affine-type errors.

    ham_tmpl : tuple, optional
        A tuple of length-`maxweight` Pauli strings (i.e. string w/letters "X",
        "Y", or "Z"), describing how to construct the fiducial pairs used to
        detect Hamiltonian errors.  The special (and default) value "auto"
        uses `("X","Y","Z")` and `("ZY","ZX","XZ","YZ","YX","XY")` for
        `maxweight` equal to 1 and 2, repectively, and will generate an error
        if `maxweight > 2`.

    preferred_prep_basis_signs, preferred_meas_basis_signs: tuple, optional
        A 3-tuple of "+" or "-" strings indicating which sign for preparing
        or measuring in the X, Y, and Z bases is preferable.  Usually one
        orientation if preferred because it's easier to achieve using the
        native model.

    Returns
    -------
    list
        a list of (prep,meas) 2-tuples of NQPauliState objects, each of
        length `nqubits`, representing the fiducial pairs.
    """
    fidpairs = []  # list of 2-tuples of NQPauliState objects to return

    #convert +'s and -'s to dictionaries of +/-1 used later:
    def conv(x): return 1 if x == "+" else -1
    base_prep_signs = {l: conv(s) for l, s in zip(('X', 'Y', 'Z'), preferred_prep_basis_signs)}
    base_meas_signs = {l: conv(s) for l, s in zip(('X', 'Y', 'Z'), preferred_meas_basis_signs)}
    #these dicts give the preferred sign for prepping or measuring along each 1Q axis.

    if include_stochastic:
        if include_affine:
            # in general there are 2^maxweight different permutations of +/- signs
            # in maxweight==1 case, need 2 of 2 permutations
            # in maxweight==2 case, need 3 of 4 permutations
            # higher maxweight?

            if maxweight == 1:
                flips = [(1,), (-1,)]  # consider both cases of not-flipping & flipping the preferred basis signs

            elif maxweight == 2:
                flips = [(1, 1),  # don't flip anything
                         (1, -1), (-1, 1)]  # flip 2nd or 1st pauli basis (weight = 2)
            else:
                raise NotImplementedError("No implementation for affine errors and maxweight > 2!")
                #need to do more work to figure out how to generalize this to maxweight > 2
        else:
            flips = [(1,) * maxweight]  # don't flip anything

        #Build up "template" of 2-tuples of NQPauliState objects acting on
        # maxweight qubits that should be tiled to full fiducial pairs.
        sto_tmpl_pairs = []
        for fliptup in flips:  # elements of flips must have length=maxweight

            # Create a set of "template" fiducial pairs using the current flips
            for basisLets in _itertools.product(('X', 'Y', 'Z'), repeat=maxweight):

                # flip base (preferred) basis signs as instructed by fliptup
                prep_signs = [f * base_prep_signs[l] for f, l in zip(fliptup, basisLets)]
                meas_signs = [f * base_meas_signs[l] for f, l in zip(fliptup, basisLets)]
                sto_tmpl_pairs.append((_pobjs.NQPauliState(''.join(basisLets), prep_signs),
                                       _pobjs.NQPauliState(''.join(basisLets), meas_signs)))

        fidpairs.extend(_idttools.tile_pauli_fidpairs(sto_tmpl_pairs, nqubits, maxweight))

    elif include_affine:
        raise ValueError("Cannot include affine sequences without also including stochastic ones!")

    if include_hamiltonian:

        nextPauli = {"X": "Y", "Y": "Z", "Z": "X"}
        prevPauli = {"X": "Z", "Y": "X", "Z": "Y"}
        def prev(expt): return ''.join([prevPauli[p] for p in expt])
        def next(expt): return ''.join([nextPauli[p] for p in expt])

        if ham_tmpl == "auto":
            if maxweight == 1: ham_tmpl = ("X", "Y", "Z")
            elif maxweight == 2: ham_tmpl = ("ZY", "ZX", "XZ", "YZ", "YX", "XY")
            else: raise ValueError("Must supply `ham_tmpl` when `maxweight > 2`!")
        ham_tmpl_pairs = []
        for tmplLets in ham_tmpl:  # "Lets" = "letters", i.e. 'X', 'Y', or 'Z'
            assert(len(tmplLets) == maxweight), \
                "Hamiltonian 'template' strings must have length == maxweight: len(%s) != %d!" % (tmplLets, maxweight)

            prepLets, measLets = prev(tmplLets), next(tmplLets)

            # basis sign doesn't matter for hamiltonian terms,
            #  so just use preferred signs
            prep_signs = [base_prep_signs[l] for l in prepLets]
            meas_signs = [base_meas_signs[l] for l in measLets]
            ham_tmpl_pairs.append((_pobjs.NQPauliState(prepLets, prep_signs),
                                   _pobjs.NQPauliState(measLets, meas_signs)))

        fidpairs.extend(_idttools.tile_pauli_fidpairs(ham_tmpl_pairs, nqubits, maxweight))

    return fidpairs


def preferred_signs_from_paulidict(pauli_basis_dict):
    """
    Infers what the preferred basis signs are based on the length of gate-name
    strings in `pauli_basis_dict` (shorter strings are preferred).

    Parameters
    ----------
    pauli_basis_dict : dict
        A dictionary w/keys like `"+X"` or `"-Y"` and values that
        are tuples of gate *names* (not labels, which include qubit or
        other state-space designations), e.g. `("Gx","Gx")`.

    Returns
    -------
    tuple
        A 3-tuple of elements in {"+", "-"}, exactly the format expected
        by `preferred_*_basis_signs` arguments of
        :function:`idle_tomography_fidpairs`.
    """
    preferred_signs = ()
    for let in ('X', 'Y', 'Z'):
        if "+" + let in pauli_basis_dict: plusKey = "+" + let
        elif let in pauli_basis_dict: plusKey = let
        else: plusKey = None

        if "-" + let in pauli_basis_dict: minusKey = '-' + let
        else: minusKey = None

        if minusKey and plusKey:
            if len(pauli_basis_dict[plusKey]) <= len(pauli_basis_dict[minusKey]):
                preferred_sign = '+'
            else:
                preferred_sign = '-'
        elif plusKey:
            preferred_sign = '+'
        elif minusKey:
            preferred_sign = '-'
        else:
            raise ValueError("No entry for %s-basis!" % let)

        preferred_signs += (preferred_sign,)

    return preferred_signs


def fidpairs_to_pauli_fidpairs(fidpairs_list, pauli_basis_dicts, nqubits):
    """
    Translate :class:`GatesString`-type fiducial pairs to
    :class:`NQPauliState`-type "Pauli fiducial pairs" using `pauli_basis_dicts`.

    Parameters
    ----------
    fidpairs_list : list
        A list whose elements are 2-tuples of :class:`Circuit` objects.

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    nqubits : int
        The number of qubits.  Needed because :class:`Circuit`
        objects don't contain this information.

    Returns
    -------
    list
        A list of 2-tuples of :class:`NQPauliState` objects.
    """

    #Example dicts:
    #prepDict = { 'X': ('Gy',), 'Y': ('Gx',)*3, 'Z': (),
    #         '-X': ('Gy',)*3, '-Y': ('Gx',), '-Z': ('Gx','Gx')}
    #measDict = { 'X': ('Gy',)*3, 'Y': ('Gx',), 'Z': (),
    #         '-X': ('Gy',), '-Y': ('Gx',)*3, '-Z': ('Gx','Gx')}
    prepDict, measDict = pauli_basis_dicts

    for k, v in prepDict.items():
        assert(k[-1] in ('X', 'Y', 'Z') and isinstance(v, tuple)), \
            "Invalid prep pauli dict format!"
    for k, v in measDict.items():
        assert(k[-1] in ('X', 'Y', 'Z') and isinstance(v, tuple)), \
            "Invalid measuse pauli dict format!"

    rev_prepDict = {v: k for k, v in prepDict.items()}
    rev_measDict = {v: k for k, v in measDict.items()}

    def convert(opstr, rev_pauli_dict):
        #Get gatenames_per_qubit (keys = sslbls, vals = lists of gatenames)
        #print("DB: Converting ",opstr)
        gatenames_per_qubit = _collections.defaultdict(list)
        for glbl in opstr:
            for c in glbl.components:  # in case of parallel labels
                assert(len(c.sslbls) == 1)
                assert(isinstance(c.sslbls[0], int))
                gatenames_per_qubit[c.sslbls[0]].append(c.name)
        #print("DB: gatenames_per_qubit =  ",gatenames_per_qubit)
        #print("DB: rev keys = ",list(rev_pauli_dict.keys()))

        #Check if list of gatenames equals a known basis prep/meas:
        letters = ""; signs = []
        for i in range(nqubits):
            basis = rev_pauli_dict.get(tuple(gatenames_per_qubit[i]), None)
            #print("DB:  Q%d: %s -> %s" % (i,str(gatenames_per_qubit[i]), str(basis)))
            assert(basis is not None)  # to indicate convert failed
            letters += basis[-1]  # last letter of basis should be 'X' 'Y' or 'Z'
            signs.append(-1 if (basis[0] == '-') else 1)

        #print("DB: SUCCESS: --> ",letters,signs)
        return _pobjs.NQPauliState(letters, signs)

    ret = []
    for prepStr, measStr in fidpairs_list:
        try:
            prepPauli = convert(prepStr, rev_prepDict)
            measPauli = convert(measStr, rev_measDict)
        except AssertionError:
            continue  # skip strings we can't convert
        ret.append((prepPauli, measPauli))

    return ret


def determine_paulidicts(model):
    """
    Intelligently determine preparation and measurement Pauli basis
    dictionaries from a :class:`Model`.

    The returned dictionaries are required for various parts of idle tomography,
    as they bridge the native model's gates to the "Pauli basis language"
    used in idle tomography.

    Parameters
    ----------
    model : Model
        The model which defines the available preparation, measurement, and
        operations.  It is assumed that `model`'s operation are expressed
        in a Pauli-product basis.

    Returns
    -------
    pauli_basis_dicts or None
        If successful, a `(prepDict,measureDict)` 2-tuple of Pauli basis
        dictionaries.  If unsuccessful, None.
    """
    #TODO: check that basis == "pp" or something similar?
    #Note: this routine just punts if model's operation labels are just strings.

    model._clean_paramvec()  # to ensure calls to obj.to_vector work below (setup model paramvec)

    #First, check that spam is prep/meas in Z basis (just check prep for now):
    try:
        prepLbls = list(model.preps.keys())
        prep = model.preps[prepLbls[0]]  # just take the first one (usually there's only one anyway)
    except AttributeError:  # HACK to work w/Implicit models
        prepLbls = list(model.prep_blks['layers'].keys())
        prep = model.prep_blks['layers'][prepLbls[0]]

    if isinstance(prep, _objs.ComputationalSPAMVec):
        if any([b != 0 for b in prep._zvals]): return None
    elif isinstance(prep, _objs.LindbladSPAMVec):
        if isinstance(prep.state_vec, _objs.ComputationalSPAMVec):
            if any([b != 0 for b in prep.state_vec._zvals]): return None
        if any([abs(v) > 1e-6 for v in prep.to_vector()]): return None
    else:
        nqubits = int(round(_np.log2(model.dim) / 2))
        cmp = _objs.ComputationalSPAMVec([0] * nqubits, model._evotype).todense()
        if _np.linalg.norm(prep.todense() - cmp) > 1e-6: return None

    def extract_action(g, cur_sslbls, ql):
        """ Note: assumes cur_sslbs is just a list of labels (of first "sector"
            of a real StateSpaceLabels struct) """
        if isinstance(g, _objs.ComposedOp):
            action = _np.identity(4, 'd')
            for fg in g.factorops:
                action = _np.dot(extract_action(fg, cur_sslbls, ql), action)
            return action

        if isinstance(g, _objs.EmbeddedOp):
            #Note: an embedded gate need not use the *same* state space labels as the model
            lbls = [cur_sslbls[g.state_space_labels.labels[0].index(locLbl)] for locLbl in g.targetLabels]
            # TODO: add to StateSpaceLabels functionality to make sure two are compatible, and to translate between
            # them, & make sub-labels?
            return extract_action(g.embedded_op, lbls, ql)

        # StaticDenseOp, LindbladDenseOp, other gates...
        if len(cur_sslbls) == 1 and cur_sslbls[0] == ql:
            mx = g.todense()
            assert(mx.shape == (4, 4))
            return mx
        else:
            mx = g.todense()
            if _np.linalg.norm(mx - _np.identity(g.dim, 'd')) < 1e-6:
                # acts as identity on some other space - this is ok
                return _np.identity(4, 'd')
            else:
                raise ValueError("LinearOperator acts nontrivially on a space other than that in its label!")

    #Get several standard 1-qubit pi/2 rotations in Pauli basis:
    pp = _objs.BuiltinBasis('pp', 4)
    Gx = _cnst.basis_build_operation([('Q0',)], "X(pi/2,Q0)", basis=pp, parameterization="static").todense()
    Gy = _cnst.basis_build_operation([('Q0',)], "Y(pi/2,Q0)", basis=pp, parameterization="static").todense()

    #try to find 1-qubit pi/2 rotations
    found = {}
    for gl in model.get_primitive_op_labels():
        if isinstance(model, _objs.ExplicitOpModel):
            gate = model.operations[gl]
        else:
            gate = model.operation_blks['layers'][gl]

        if gl.sslbls is None or len(gl.sslbls) != 1:
            continue  # skip gates that don't have 1Q-like labels
        qubit_label = gl.sslbls[0]  # the qubit this gate is supposed to act on
        try:
            assert(len(model.state_space_labels.labels) == 1), "Assumes a single state space sector"
            action_on_qubit = extract_action(gate,
                                             model.state_space_labels.labels[0],
                                             qubit_label)
        except ValueError:
            continue  # skip gates that we can't extract action from

        #See if we recognize this action
        # FUTURE: add more options for using other existing gates?
        if _np.linalg.norm(action_on_qubit - Gx) < 1e-6:
            found['Gx'] = gl.name
        elif _np.linalg.norm(action_on_qubit - Gy) < 1e-6:
            found['Gy'] = gl.name

    if 'Gx' in found and 'Gy' in found:
        Gxl = found['Gx']; Gyl = found['Gy']

        prepDict = {'X': (Gyl,), 'Y': (Gxl,) * 3, 'Z': (),
                    '-X': (Gyl,) * 3, '-Y': (Gxl,), '-Z': (Gxl, Gxl)}
        measDict = {'X': (Gyl,) * 3, 'Y': (Gxl,), 'Z': (),
                    '-X': (Gyl,), '-Y': (Gxl,) * 3, '-Z': (Gxl, Gxl)}
        return prepDict, measDict

    return None


def make_idle_tomography_list(nqubits, max_lenghts, pauli_basis_dicts, maxweight=2,
                              idle_string=((),), include_hamiltonian=True,
                              include_stochastic=True, include_affine=True,
                              ham_tmpl="auto", preferred_prep_basis_signs="auto",
                              preferred_meas_basis_signs="auto"):
    """
    Construct the list of experiments needed to perform idle tomography.

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    max_lenghts : list
        A list of maximum germ-power lengths. Each specifies a number many times
        to repeat the idle gate, and typically this is a list of the powers of
        2 preceded by zero, e.g. `[0,1,2,4,16]`.  The largest value in this
        list should be chosen to be the maximum number of idle gates you want to
        perform in a row (typically limited by performance or time constraints).

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    maxweight : int, optional
        The maximum weight of errors to consider.

    idle_string : Circuit-like, optional
        A Circuit or tuple of operation labels that represents the idle
        gate being characterized by idle tomography.

    include_hamiltonian, include_stochastic, include_affine : bool, optional
        Whether to include fiducial pairs for finding Hamiltonian-, Stochastic-,
        and Affine-type errors.

    ham_tmpl : tuple, optional
        A tuple of length-`maxweight` Pauli strings (i.e. string w/letters "X",
        "Y", or "Z"), describing how to construct the fiducial pairs used to
        detect Hamiltonian errors.  The special (and default) value "auto"
        uses `("X","Y","Z")` and `("ZY","ZX","XZ","YZ","YX","XY")` for
        `maxweight` equal to 1 and 2, repectively, and will generate an error
        if `maxweight > 2`.

    preferred_prep_basis_signs, preferred_meas_basis_signs: tuple, optional
        A 3-tuple of "+" or "-" strings indicating which sign for preparing
        or measuring in the X, Y, and Z bases is preferable.  Usually one
        orientation if preferred because it's easier to achieve using the
        native model.  Additionally, the special (and default) value "auto"
        may be used, in which case :function:`preferred_signs_from_paulidict`
        is used to choose preferred signs based on `pauli_basis_dicts`.

    Returns
    -------
    list
        A list of :class:`Circuit` objects.
    """

    prepDict, measDict = pauli_basis_dicts
    if preferred_prep_basis_signs == "auto":
        preferred_prep_basis_signs = preferred_signs_from_paulidict(prepDict)
    if preferred_meas_basis_signs == "auto":
        preferred_meas_basis_signs = preferred_signs_from_paulidict(measDict)

    GiStr = _objs.Circuit(idle_string, num_lines=nqubits)

    pauli_fidpairs = idle_tomography_fidpairs(
        nqubits, maxweight, include_hamiltonian, include_stochastic,
        include_affine, ham_tmpl, preferred_prep_basis_signs,
        preferred_meas_basis_signs)

    fidpairs = [(x.to_circuit(prepDict), y.to_circuit(measDict))
                for x, y in pauli_fidpairs]  # e.g. convert ("XY","ZX") to tuple of Circuits

    listOfExperiments = []
    for prepFid, measFid in fidpairs:  # list of fidpairs / configs (a prep/meas that gets I^L placed btwn it)
        for L in max_lenghts:
            listOfExperiments.append(prepFid + GiStr * L + measFid)

    return listOfExperiments


def make_idle_tomography_lists(nqubits, max_lenghts, pauli_basis_dicts, maxweight=2,
                               idle_string=((),), include_hamiltonian=True,
                               include_stochastic=True, include_affine=True,
                               ham_tmpl="auto", preferred_prep_basis_signs="auto",
                               preferred_meas_basis_signs="auto"):
    """
    Construct lists of experiments, one for each maximum-length value, needed
    to perform idle tomography.  This is potentiall useful for running GST on
    idle tomography data.

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    max_lenghts : list
        A list of maximum germ-power lengths. Each specifies a number many times
        to repeat the idle gate, and typically this is a list of the powers of
        2 preceded by zero, e.g. `[0,1,2,4,16]`.  The largest value in this
        list should be chosen to be the maximum number of idle gates you want to
        perform in a row (typically limited by performance or time constraints).

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    maxweight : int, optional
        The maximum weight of errors to consider.

    idle_string : Circuit-like, optional
        A Circuit or tuple of operation labels that represents the idle
        gate being characterized by idle tomography.

    include_hamiltonian, include_stochastic, include_affine : bool, optional
        Whether to include fiducial pairs for finding Hamiltonian-, Stochastic-,
        and Affine-type errors.

    ham_tmpl : tuple, optional
        A tuple of length-`maxweight` Pauli strings (i.e. string w/letters "X",
        "Y", or "Z"), describing how to construct the fiducial pairs used to
        detect Hamiltonian errors.  The special (and default) value "auto"
        uses `("X","Y","Z")` and `("ZY","ZX","XZ","YZ","YX","XY")` for
        `maxweight` equal to 1 and 2, repectively, and will generate an error
        if `maxweight > 2`.

    preferred_prep_basis_signs, preferred_meas_basis_signs: tuple, optional
        A 3-tuple of "+" or "-" strings indicating which sign for preparing
        or measuring in the X, Y, and Z bases is preferable.  Usually one
        orientation if preferred because it's easier to achieve using the
        native model.  Additionally, the special (and default) value "auto"
        may be used, in which case :function:`preferred_signs_from_paulidict`
        is used to choose preferred signs based on `pauli_basis_dicts`.

    Returns
    -------
    list
        A list of lists of :class:`Circuit` objects, one list per max-L value.
    """

    prepDict, measDict = pauli_basis_dicts
    if preferred_prep_basis_signs == "auto":
        preferred_prep_basis_signs = preferred_signs_from_paulidict(prepDict)
    if preferred_meas_basis_signs == "auto":
        preferred_meas_basis_signs = preferred_signs_from_paulidict(measDict)

    GiStr = _objs.Circuit(idle_string, num_lines=nqubits)

    pauli_fidpairs = idle_tomography_fidpairs(
        nqubits, maxweight, include_hamiltonian, include_stochastic,
        include_affine, ham_tmpl, preferred_prep_basis_signs,
        preferred_meas_basis_signs)

    fidpairs = [(x.to_circuit(prepDict), y.to_circuit(measDict))
                for x, y in pauli_fidpairs]  # e.g. convert ("XY","ZX") to tuple of Circuits

    listOfListsOfExperiments = []
    for L in max_lenghts:
        expsForThisL = []
        for prepFid, measFid in fidpairs:  # list of fidpairs / configs (a prep/meas that gets I^L placed btwn it)
            expsForThisL.append(prepFid + GiStr * L + measFid)
        listOfListsOfExperiments.append(expsForThisL)

    return listOfListsOfExperiments


# -----------------------------------------------------------------------------
# Running idle tomography
# -----------------------------------------------------------------------------

def get_obs_samebasis_err_rate(dataset, pauli_fidpair, pauli_basis_dicts, idle_string,
                               outcome, max_lenghts, fit_order=1):
    """
    Extract the observed error rate from a series of experiments which prepares
    and measures in the *same* Pauli basis and tracks a particular `outcome`.

    Parameters
    ----------
    dataset : DataSet
        The set of data counts (observations) to use.

    pauli_fidpair : tuple
        A `(prep,measure)` 2-tuple of :class:`NQPauliState` objects specifying
        the prepation state and measurement basis.

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    idle_string : Circuit
        The Circuit representing the idle operation being characterized.

    outcome : NQOutcome
        The outcome being tracked.

    max_lenghts : list
        A list of maximum germ-power lengths.  The seriese of sequences
        considered is `prepFiducial + idle_string^L + measFiducial`, where
        `L` ranges over the values in `max_lenghts`.

    fit_order : int, optional
        The polynomial order used to fit the observed data probabilities.

    Returns
    -------
    dict
        A dictionary of information about the fit, including the observed
        error rate and the data points that were fit.
    """
    # fit number of given outcome counts to a line
    pauli_prep, pauli_meas = pauli_fidpair

    prepDict, measDict = pauli_basis_dicts
    prepFid = pauli_prep.to_circuit(prepDict)
    measFid = pauli_meas.to_circuit(measDict)

    #Note on weights:
    # data point with frequency f and N samples should be weighted w/ sqrt(N)/sqrt(f*(1-f))
    # but in case f is 0 or 1 we use proxy f' by adding a dummy 0 and 1 count.
    def freq_and_weight(circuit, outcome):
        """Get the frequency, weight, and errobar for a ptic circuit"""
        cnts = dataset[circuit].counts  # a normal dict
        total = sum(cnts.values())
        f = cnts.get((outcome.rep,), 0) / total  # (py3 division) NOTE: outcomes are actually 1-tuples
        fp = (cnts.get((outcome.rep,), 0) + 1) / (total + 2)  # Note: can't == 1
        wt = _np.sqrt(total / abs(fp * (1.0 - fp)))  # abs to deal with non-CP data (simulated using termorder:1)
        err = _np.sqrt(abs(f * (1.0 - f)) / total)  # no need to use fp
        return f, wt, err

    #Get data to fit and weights to use in fitting
    data_to_fit = []; wts = []; errbars = []
    for L in max_lenghts:
        opstr = prepFid + idle_string * L + measFid
        f, wt, err = freq_and_weight(opstr, outcome)
        data_to_fit.append(f)
        wts.append(wt)
        errbars.append(err)

    #curvefit -> slope
    coeffs = _np.polyfit(max_lenghts, data_to_fit, fit_order, w=wts)  # when fit_order = 1 = line
    if fit_order == 1:
        slope = coeffs[0]
    elif fit_order == 2:
        #OLD: slope =  coeffs[1] # c2*x2 + c1*x + c0 ->deriv@x=0-> c1
        det = coeffs[1]**2 - 4 * coeffs[2] * coeffs[0]
        slope = -_np.sign(coeffs[0]) * _np.sqrt(det) if det >= 0 else coeffs[1]
    else: raise NotImplementedError("Only fit_order <= 2 are supported!")

    return {'rate': slope,
            'fit_order': fit_order,
            'fitCoeffs': coeffs,
            'data': data_to_fit,
            'errbars': errbars,
            'weights': wts}


def get_obs_diffbasis_err_rate(dataset, pauli_fidpair, pauli_basis_dicts,
                               idle_string, observable, max_lenghts, fit_order=1):
    """
    Extract the observed error rate from a series of experiments which prepares
    and measures in *different* Pauli basis and tracks the expectation value of
    a particular `observable`.

    Parameters
    ----------
    dataset : DataSet
        The set of data counts (observations) to use.

    pauli_fidpair : tuple
        A `(prep,measure)` 2-tuple of :class:`NQPauliState` objects specifying
        the prepation state and measurement basis.

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    idle_string : Circuit
        The Circuit representing the idle operation being characterized.

    observable : NQPauliOp
        The observable whose expectation value is being tracked.

    max_lenghts : list
        A list of maximum germ-power lengths.  The seriese of sequences
        considered is `prepFiducial + idle_string^L + measFiducial`, where
        `L` ranges over the values in `max_lenghts`.

    fit_order : int, optional
        The polynomial order used to fit the observed data probabilities.

    Returns
    -------
    dict
        A dictionary of information about the fit, including the observed
        error rate and the data points that were fit.
    """
    # fit expectation value of `observable` (trace over all I elements of it) to a line
    pauli_prep, pauli_meas = pauli_fidpair

    prepDict, measDict = pauli_basis_dicts
    prepFid = pauli_prep.to_circuit(prepDict)
    measFid = pauli_meas.to_circuit(measDict)

    #observable is always equal to pauli_meas (up to signs) with all but 1 or 2
    # (maxErrWt in general) of it's elements replaced with 'I', essentially just
    # telling us which 1 or 2 qubits to take the <Z> or <ZZ> expectation value of
    # (since the meas fiducial gets us in the right basis) -- i.e. the qubits to *not* trace over.
    obs_indices = [i for i, letter in enumerate(observable.rep) if letter != 'I']
    minus_sign = _np.prod([pauli_meas.signs[i] for i in obs_indices])

    def unsigned_exptn_and_weight(circuit, observed_indices):
        #compute expectation value of observable
        drow = dataset[circuit]  # dataset row
        total = drow.total

        # <Z> = 0 count - 1 count (if measFid sign is +1, otherwise reversed via minus_sign)
        if len(observed_indices) == 1:
            i = observed_indices[0]  # the qubit we care about
            cnt0 = cnt1 = 0
            for outcome, cnt in drow.counts.items():
                if outcome[0][i] == '0': cnt0 += cnt  # [0] b/c outcomes are actually 1-tuples
                else: cnt1 += cnt
            exptn = float(cnt0 - cnt1) / total
            fp = 0.5 + 0.5 * float(cnt0 - cnt1 + 1) / (total + 2)

        # <ZZ> = 00 count - 01 count - 10 count + 11 count (* minus_sign)
        elif len(observed_indices) == 2:
            i, j = observed_indices  # the qubits we care about
            cnt_even = cnt_odd = 0
            for outcome, cnt in drow.counts.items():
                if outcome[0][i] == outcome[0][j]: cnt_even += cnt
                else: cnt_odd += cnt
            exptn = float(cnt_even - cnt_odd) / total
            fp = 0.5 + 0.5 * float(cnt_even - cnt_odd + 1) / (total + 2)
        else:
            raise NotImplementedError("Expectation values of weight > 2 observables are not implemented!")

        wt = _np.sqrt(total) / _np.sqrt(fp * (1.0 - fp))
        f = 0.5 + 0.5 * exptn
        err = 2 * _np.sqrt(f * (1.0 - f) / total)  # factor of 2 b/c expectation is addition of 2 terms
        return exptn, wt, err

    #Get data to fit and weights to use in fitting
    data_to_fit = []; wts = []; errbars = []
    for L in max_lenghts:
        opstr = prepFid + idle_string * L + measFid
        exptn, wt, err = unsigned_exptn_and_weight(opstr, obs_indices)
        data_to_fit.append(minus_sign * exptn)
        wts.append(wt)
        errbars.append(err)

    #curvefit -> slope
    coeffs = _np.polyfit(max_lenghts, data_to_fit, fit_order, w=wts)  # when fit_order = 1 = line
    if fit_order == 1:
        slope = coeffs[0]
    elif fit_order == 2:
        #OLD: slope =  coeffs[1] # c2*x2 + c1*x + c0 ->deriv@x=0-> c1
        det = coeffs[1]**2 - 4 * coeffs[2] * coeffs[0]
        slope = -_np.sign(coeffs[0]) * _np.sqrt(det) if det >= 0 else coeffs[1]
        # c2*x2 + c1*x + c0 ->deriv@y=0-> 2*c2*x0 + c1;
        # x0=[-c1 +/- sqrt(c1^2 - 4c2*c0)] / 2*c2; take smaller root
        # but if determinant is < 0, fall back to x=0 slope
    else: raise NotImplementedError("Only fit_order <= 2 are supported!")

    return {'rate': slope,
            'fit_order': fit_order,
            'fitCoeffs': coeffs,
            'data': data_to_fit,
            'errbars': errbars,
            'weights': wts}


def do_idle_tomography(nqubits, dataset, max_lenghts, pauli_basis_dicts, maxweight=2,
                       idle_string=((),), include_hamiltonian="auto",
                       include_stochastic="auto", include_affine="auto",
                       advanced_options=None, verbosity=0, comm=None):
    """
    Analyze `dataset` using the idle tomography protocol to characterize
    `idle_string`.

    Parameters
    ----------
    nqubits : int
        The number of qubits.

    dataset : DataSet
        The set of data counts (observations) to use.

    max_lenghts : list
        A list of maximum germ-power lengths. Each specifies a number many times
        to repeat the idle gate, and typically this is a list of the powers of
        2 preceded by zero, e.g. `[0,1,2,4,16]`.  The largest value in this
        list should be chosen to be the maximum number of idle gates you want to
        perform in a row (typically limited by performance or time constraints).

    pauli_basis_dicts : tuple
        A `(prepPauliBasisDict,measPauliBasisDict)` tuple of dictionaries
        specifying the way to prepare and measure in Pauli bases.  See
        :function:`preferred_signs_from_paulidict` for details on each
        dictionary's format.

    maxweight : int, optional
        The maximum weight of errors to consider.

    idle_string : Circuit-like, optional
        A Circuit or tuple of operation labels that represents the idle
        gate being characterized by idle tomography.

    include_hamiltonian, include_stochastic, include_affine : {True,False,"auto"}
        Whether to extract Hamiltonian-, Stochastic-, and Affine-type
        intrinsic errors.  If "auto" is specified, then the corresponding
        error-type is extracted only if there is enough data to reliably
        infer them (i.e. enough data to construct "full rank" Jacobian
        matrices).

    advanced_options : dict, optional
        A dictionary of optional advanced arguments which influence the
        way idle tomography is performed.  Allowed keys are:

        - "jacobian mode": {"separate","together"} how to evaluate jacobians
        - "preferred_prep_basis_signs" : 3-tuple of "+"/"-" or default="auto"
        - "preferred_meas_basis_signs" : 3-tuple of "+"/"-" or default="auto"
        - "pauli_fidpairs": alternate list of pauli fiducial pairs to use
        - "fit order" : integer order for polynomial fits to data
        - "ham_tmpl" : see :function:`make_idle_tomography_list`

    verbosity : int, optional
        How much detail to send to stdout.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    Returns
    -------
    IdleTomographyResults
    """

    printer = _VerbosityPrinter.build_printer(verbosity, comm=comm)

    if advanced_options is None:
        advanced_options = {}

    prepDict, measDict = pauli_basis_dicts
    if nqubits == 1:  # special case where line-labels may be ('*',)
        if len(dataset) > 0:
            first_circuit = list(dataset.keys())[0]
            line_labels = first_circuit.line_labels
        else:
            line_labels = (0,)
        GiStr = _objs.Circuit(idle_string, line_labels=line_labels)
    else:
        GiStr = _objs.Circuit(idle_string, num_lines=nqubits)

    jacmode = advanced_options.get("jacobian mode", "separate")
    sto_aff_jac = None; sto_aff_obs_err_rates = None
    ham_aff_jac = None; ham_aff_obs_err_rates = None

    rankStr = "" if (comm is None) else "Rank%d: " % comm.Get_rank()

    preferred_prep_basis_signs = advanced_options.get('preferred_prep_basis_signs', 'auto')
    preferred_meas_basis_signs = advanced_options.get('preferred_meas_basis_signs', 'auto')
    if preferred_prep_basis_signs == "auto":
        preferred_prep_basis_signs = preferred_signs_from_paulidict(prepDict)
    if preferred_meas_basis_signs == "auto":
        preferred_meas_basis_signs = preferred_signs_from_paulidict(measDict)

    if 'pauli_fidpairs' in advanced_options:
        same_basis_fidpairs = []  # *all* qubits prep/meas in same basis
        diff_basis_fidpairs = []  # at least one doesn't
        for pauli_fidpair in advanced_options['pauli_fidpairs']:
            #pauli_fidpair is a (prep,meas) tuple of NQPauliState objects
            if pauli_fidpair[0].rep == pauli_fidpair[1].rep:  # don't care about sign
                same_basis_fidpairs.append(pauli_fidpair)
            else:
                diff_basis_fidpairs.append(pauli_fidpair)
        #print("DB: LENGTHS: same=",len(same_basis_fidpairs)," diff=",len(diff_basis_fidpairs))
    else:
        same_basis_fidpairs = None  # just for
        diff_basis_fidpairs = None  # safety

    errors = _idttools.allerrors(nqubits, maxweight)
    fit_order = advanced_options.get('fit order', 1)
    intrinsic_rates = {}
    pauli_fidpair_dict = {}
    observed_rate_infos = {}

    if include_stochastic in (True, "auto"):
        tStart = _time.time()
        if 'pauli_fidpairs' in advanced_options:
            pauli_fidpairs = same_basis_fidpairs
        else:
            pauli_fidpairs = idle_tomography_fidpairs(
                nqubits, maxweight, False, include_stochastic, include_affine,
                advanced_options.get('ham_tmpl', "auto"),
                preferred_prep_basis_signs, preferred_meas_basis_signs)
        #print("DB: %d same-basis pairs" % len(pauli_fidpairs))

        #divide up strings among ranks
        indxFidpairList = list(enumerate(pauli_fidpairs))
        my_FidpairList, _, _ = _tools.mpitools.distribute_indices(indxFidpairList, comm, False)

        my_J = []; my_obs_infos = []
        for i, (ifp, pauli_fidpair) in enumerate(my_FidpairList):
            #NOTE: pauli_fidpair is a 2-tuple of NQPauliState objects

            all_outcomes = _idttools.alloutcomes(pauli_fidpair[0], pauli_fidpair[1], maxweight)
            t0 = _time.time(); infos_for_this_fidpair = _collections.OrderedDict()
            for j, out in enumerate(all_outcomes):

                printer.log("  - outcome %d of %d" % (j, len(all_outcomes)), 2)

                #form jacobian rows as we get extrinsic error rates
                Jrow = [stochastic_jac_element(pauli_fidpair[0], err, pauli_fidpair[1], out)
                        for err in errors]
                if include_affine:
                    Jrow.extend([affine_jac_element(pauli_fidpair[0], err, pauli_fidpair[1], out)
                                 for err in errors])
                my_J.append(Jrow)

                info = get_obs_samebasis_err_rate(dataset, pauli_fidpair, pauli_basis_dicts, GiStr,
                                                  out, max_lenghts, fit_order)
                info['jacobian row'] = _np.array(Jrow)
                infos_for_this_fidpair[out] = info

            my_obs_infos.append(infos_for_this_fidpair)
            printer.log("%sStochastic fidpair %d of %d: %d outcomes analyzed [%.1fs]" %
                        (rankStr, i, len(my_FidpairList), len(all_outcomes), _time.time() - t0), 1)

        #Gather results
        info_list = [my_obs_infos] if (comm is None) else comm.gather(my_obs_infos, root=0)
        J_list = [my_J] if (comm is None) else comm.gather(my_J, root=0)

        if comm is None or comm.Get_rank() == 0:
            # pseudo-invert J to get "intrinsic" error rates (labeled by AllErrors(nqubits))
            # J*intr = obs
            J = _np.concatenate(J_list, axis=0)
            infos_by_fidpair = list(_itertools.chain(*info_list))  # flatten ~ concatenate

            obs_err_rates = _np.array([info['rate']
                                       for fidpair_infos in infos_by_fidpair
                                       for info in fidpair_infos.values()])

            if jacmode == "separate":
                rank = _np.linalg.matrix_rank(J)
                if rank < J.shape[1]:
                    #Rank defficiency - if affine is "auto", try with just stochastic
                    if include_affine == "auto":
                        J_sto = J[:, 0:len(errors)]
                        rank_sto = _np.linalg.matrix_rank(J_sto)
                        if rank_sto < len(errors):
                            if include_stochastic == "auto":
                                include_stochastic = False  # drop stochastic part
                            else:
                                _warnings.warn(("Idle tomography: stochastic-jacobian rank "
                                                "(%d) < #intrinsic rates (%d)") % (rank_sto, J_sto.shape[1]))
                        else:  # stochasic alone is OK - drop affine part
                            J = J_sto
                            include_affine = False  # for below processing

                    else:
                        if include_affine and include_stochastic == "auto":
                            raise ValueError(("Cannot set `include_stochastic`"
                                              " to 'auto' when `include_affine` is True"))
                        _warnings.warn(("Idle tomography: %s-jacobian rank "
                                        "(%d) < #intrinsic rates (%d)") % ("samebasis", rank, J.shape[1]))

                invJ = _np.linalg.pinv(J)
                intrinsic_stochastic_rates = _np.dot(invJ, obs_err_rates)

            if include_stochastic:  # "auto" could change to False in jac processing above
                if include_affine:
                    if jacmode == "separate":
                        Nrates = len(intrinsic_stochastic_rates)
                        intrinsic_rates['stochastic'] = intrinsic_stochastic_rates[0:Nrates // 2]
                        intrinsic_rates['affine'] = intrinsic_stochastic_rates[Nrates // 2:]
                    elif jacmode == "together":
                        sto_aff_jac = J
                        sto_aff_obs_err_rates = obs_err_rates
                    else: raise ValueError("Invalid `jacmode` == %s" % str(jacmode))
                    pauli_fidpair_dict['samebasis'] = pauli_fidpairs  # "key" to observed rates
                    observed_rate_infos['samebasis'] = infos_by_fidpair
                else:
                    if jacmode == "separate":
                        intrinsic_rates['stochastic'] = intrinsic_stochastic_rates
                    elif jacmode == "together":
                        sto_aff_jac = J
                        sto_aff_obs_err_rates = obs_err_rates
                    pauli_fidpair_dict['samebasis'] = pauli_fidpairs  # "key" to observed rates
                    observed_rate_infos['samebasis'] = infos_by_fidpair

            printer.log("Completed Stochastic/Affine in %.2fs" % (_time.time() - tStart), 1)

    elif include_affine:  # either True or "auto"
        raise ValueError("Cannot extract affine error rates without also extracting stochastic ones!")

    if include_hamiltonian in (True, "auto"):
        tStart = _time.time()
        if 'pauli_fidpairs' in advanced_options:
            pauli_fidpairs = diff_basis_fidpairs
        else:
            pauli_fidpairs = idle_tomography_fidpairs(
                nqubits, maxweight, include_hamiltonian, False, False,
                advanced_options.get('ham_tmpl', "auto"),
                preferred_prep_basis_signs, preferred_meas_basis_signs)
        #print("DB: %d diff-basis pairs" % len(pauli_fidpairs))

        #divide up fiducial pairs among ranks
        indxFidpairList = list(enumerate(pauli_fidpairs))
        my_FidpairList, _, _ = _tools.mpitools.distribute_indices(indxFidpairList, comm, False)

        my_J = []; my_obs_infos = []; my_Jaff = []
        for i, (ifp, pauli_fidpair) in enumerate(my_FidpairList):
            all_observables = _idttools.allobservables(pauli_fidpair[1], maxweight)

            t0 = _time.time(); infos_for_this_fidpair = _collections.OrderedDict()
            for j, obs in enumerate(all_observables):
                printer.log("  - observable %d of %d" % (j, len(all_observables)), 2)

                #form jacobian rows as we get extrinsic error rates
                Jrow = [hamiltonian_jac_element(pauli_fidpair[0], err, obs) for err in errors]
                my_J.append(Jrow)

                # J_ham * Hintrinsic + J_aff * Aintrinsic = observed_rates, and Aintrinsic is known
                #  -> need to find J_aff, the jacobian of *observable expectation vales* w/affine params.
                if include_affine:
                    Jaff_row = [affine_jac_obs_element(pauli_fidpair[0], err, obs)
                                for err in errors]
                    my_Jaff.append(Jaff_row)

                info = get_obs_diffbasis_err_rate(dataset, pauli_fidpair, pauli_basis_dicts, GiStr, obs,
                                                  max_lenghts, fit_order)
                info['jacobian row'] = _np.array(Jrow)
                if include_affine: info['affine jacobian row'] = _np.array(Jaff_row)
                infos_for_this_fidpair[obs] = info

            my_obs_infos.append(infos_for_this_fidpair)
            printer.log("%sHamiltonian fidpair %d of %d: %d observables analyzed [%.1fs]" %
                        (rankStr, i, len(my_FidpairList), len(all_observables), _time.time() - t0), 1)

        #Gather results
        info_list = [my_obs_infos] if (comm is None) else comm.gather(my_obs_infos, root=0)
        J_list = [my_J] if (comm is None) else comm.gather(my_J, root=0)
        if include_affine:
            Jaff_list = [my_Jaff] if (comm is None) else comm.gather(my_Jaff, root=0)

        if comm is None or comm.Get_rank() == 0:
            # pseudo-invert J to get "intrinsic" error rates (labeled by AllErrors(nqubits))
            # J*intr = obs
            J = _np.concatenate(J_list, axis=0)
            infos_by_fidpair = list(_itertools.chain(*info_list))  # flatten ~ concatenate

            obs_err_rates = _np.array([info['rate']
                                       for fidpair_infos in infos_by_fidpair
                                       for info in fidpair_infos.values()])

            if jacmode == "separate":
                if include_affine:
                    #'correct' observed rates due to known affine errors, i.e.:
                    # J_ham * Hintrinsic = observed_rates - J_aff * Aintrinsic
                    Jaff = _np.concatenate(Jaff_list, axis=0)
                    Aintrinsic = intrinsic_rates['affine']
                    corr = _np.dot(Jaff, Aintrinsic)
                    obs_err_rates -= corr

                rank = _np.linalg.matrix_rank(J)
                if rank < J.shape[1]:
                    if include_hamiltonian == "auto":
                        include_hamiltonian = False
                    else:
                        _warnings.warn(("Idle tomography: hamiltonian-jacobian rank "
                                        "(%d) < #intrinsic rates (%d)") % (rank, J.shape[1]))

                if include_hamiltonian:  # could have been changed "auto" -> False above
                    invJ = _np.linalg.pinv(J)
                    intrinsic_hamiltonian_rates = _np.dot(invJ, obs_err_rates)
                    intrinsic_rates['hamiltonian'] = intrinsic_hamiltonian_rates
            elif jacmode == "together":
                if include_affine:
                    Jaff = _np.concatenate(Jaff_list, axis=0)
                    ham_aff_jac = _np.concatenate((J, Jaff), axis=1)
                else:
                    ham_aff_jac = J
                ham_aff_obs_err_rates = obs_err_rates

            pauli_fidpair_dict['diffbasis'] = pauli_fidpairs  # give "key" to observed rates
            observed_rate_infos['diffbasis'] = infos_by_fidpair
            printer.log("Completed Hamiltonian in %.2fs" % (_time.time() - tStart), 1)

    if comm is None or comm.Get_rank() == 0:
        if jacmode == "together":
            #Compute all intrinsic rates now, by inverting one big jacobian
            Ne = len(errors)

            Nrows = 0
            if include_hamiltonian: Nrows += ham_aff_jac.shape[0]
            if include_stochastic: Nrows += sto_aff_jac.shape[0]

            Ncols = 0
            if include_hamiltonian: Ncols += Ne
            if include_stochastic: Ncols += Ne
            if include_affine: Ncols += Ne

            if include_hamiltonian:
                sto_col = Ne
                sto_row = ham_aff_jac.shape[0]
            else:
                sto_col = sto_row = 0

            Jbig = _np.zeros((Nrows, Ncols), 'd')

            obs_to_concat = []
            if include_hamiltonian:
                Jbig[0:sto_row, 0:Ne] = ham_aff_jac[:, 0:Ne]
                obs_to_concat.append(ham_aff_obs_err_rates)
                if include_affine:
                    Jbig[0:sto_row, 2 * Ne:3 * Ne] = ham_aff_jac[:, Ne:]
            if include_stochastic:
                Jbig[sto_row:, sto_col:] = sto_aff_jac
                obs_to_concat.append(sto_aff_obs_err_rates)

            while _np.linalg.matrix_rank(Jbig) < Jbig.shape[1]:
                if include_affine == "auto":  # then drop affine
                    include_affine = False
                    Jbig = Jbig[:, 0:sto_col + Ne]
                elif include_hamiltonian == "auto":  # then drop hamiltonian
                    include_hamiltonian = False
                    Jbig = Jbig[:, Ne:]; sto_col = 0
                elif include_stochastic == "auto":  # then drop stochastic
                    include_stochastic = False
                    Jbig = Jbig[:, 0:sto_col]
                else:  # nothing to drop... warn if anything is left
                    if include_hamiltonian or include_stochastic or include_affine:
                        rank = _np.linalg.matrix_rank(Jbig)
                        _warnings.warn(("Idle tomography: whole-jacobian rank "
                                        "(%d) < #intrinsic rates (%d)") % (rank, Jbig.shape[1]))
                    break
                if Jbig.shape[1] == 0: break  # nothing left (matrix_rank will fail!)

            if Jbig.shape[1] > 0:  # if there's any jacobian left
                invJ = _np.linalg.pinv(Jbig)
                obs_err_rates = _np.concatenate(obs_to_concat)
                all_intrinsic_rates = _np.dot(invJ, obs_err_rates)

            off = 0
            if include_hamiltonian:
                intrinsic_rates['hamiltonian'] = all_intrinsic_rates[off:off + len(errors)]
                off += len(errors)
            if include_stochastic:
                intrinsic_rates['stochastic'] = all_intrinsic_rates[off:off + len(errors)]
                off += len(errors)
            if include_affine:
                intrinsic_rates['affine'] = all_intrinsic_rates[off:off + len(errors)]

        return _IdleTomographyResults(dataset, max_lenghts, maxweight, fit_order, pauli_basis_dicts, GiStr,
                                      errors, intrinsic_rates, pauli_fidpair_dict, observed_rate_infos)
    else:  # no results on other ranks...
        return None
