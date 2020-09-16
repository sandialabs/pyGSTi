"""
Core GST algorithms
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
import scipy.optimize as _spo
import scipy.stats as _stats
import warnings as _warnings
import time as _time
import collections as _collections

from .. import optimize as _opt
from .. import tools as _tools
from .. import objects as _objs
from .. import construction as _pc
from ..objects import objectivefns as _objfns
from ..objects.profiler import DummyProfiler as _DummyProfiler
from ..objects.circuitlist import CircuitList as _CircuitList
from ..objects.resourceallocation import ResourceAllocation as _ResourceAllocation
from ..objects.termforwardsim import TermForwardSimulator as _TermFSim
from ..optimize.customlm import Optimizer as _Optimizer
from ..optimize.customlm import CustomLMOptimizer as _CustomLMOptimizer
_dummy_profiler = _DummyProfiler()


CUSTOMLM = True
FLOATSIZE = 8  # TODO: better way?
#from .track_allocations import AllocationTracker

#Note on where 4x4 or possibly other integral-qubit dimensions are needed:
# 1) Need to use Jamiol. Isomorphism to contract to CPTP or even gauge optimize to CPTP
#       since we need to know a Choi matrix basis to perform the Jamiol. isomorphism
# 2) Need pauilVector <=> matrix in contractToValidSpam
# 3) use Jamiol. Iso in print_model_info(...)

###################################################################################
#                 Linear Inversion GST (LGST)
###################################################################################


def run_lgst(dataset, prep_fiducials, effect_fiducials, target_model, op_labels=None, op_label_aliases=None,
             guess_model_for_gauge=None, svd_truncate_to=None, verbosity=0):
    """
    Performs Linear-inversion Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to generate the LGST estimates

    prep_fiducials : list of Circuits
        Fiducial Circuits used to construct a informationally complete
        effective preparation.

    effect_fiducials : list of Circuits
        Fiducial Circuits used to construct a informationally complete
        effective measurement.

    target_model : Model
        A model used to specify which operation labels should be estimated, a
        guess for which gauge these estimates should be returned in.

    op_labels : list, optional
        A list of which operation labels (or aliases) should be estimated.
        Overrides the operation labels in target_model.
        e.g. ['Gi','Gx','Gy','Gx2']

    op_label_aliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are circuits
        corresponding to what that operation label should be expanded into before querying
        the dataset.  Defaults to the empty dictionary (no aliases defined)
        e.g. op_label_aliases['Gx^3'] = pygsti.obj.Circuit(['Gx','Gx','Gx'])

    guess_model_for_gauge : Model, optional
        A model used to compute a gauge transformation that is applied to
        the LGST estimates before they are returned.  This gauge transformation
        is computed such that if the estimated gates matched the model given,
        then the operation matrices would match, i.e. the gauge would be the same as
        the model supplied.
        Defaults to target_model.

    svd_truncate_to : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `target_model`.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    Model
        A model containing all of the estimated labels (or aliases)
    """

    #Notes:
    # We compute,                                                                                                                                               # noqa
    # I_tilde = AB   (trunc,trunc), where trunc <= K = min(nRhoSpecs,nESpecs)                                                                                   # noqa
    # X_tilde = AXB  (trunc,trunc)                                                                                                                              # noqa
    # and  A, B for *target* model. (but target model may need dimension increase to get to trunc... and then A,B are rank deficient)                           # noqa
    # We would like to get X or it's gauge equivalent.                                                                                                          # noqa
    #  We do:       1)  (I^-1)*AXB ~= B^-1 X B := Xhat -- we solve Ii*A*B = identity for Ii                                                                     # noqa
    #               2) B * Xhat * B^-1 ==> X  (but what if B is non-invertible -- say rectangular) Want B*(something) ~ identity ??                             # noqa
    # for lower rank target models, want a gauge tranformation that brings Xhat => X of "increased dim" model                                                   # noqa
    # want "B^-1" such that B(gsDim,nRhoSpecs) "B^-1"(nRhoSpecs,gsDim) ~ Identity(gsDim)                                                                        # noqa
    #   Ub,sb,Vb = svd(B) so B = Ub*diag(sb)*Vb  where Ub = (gsDim,M), s = (M,M), Vb = (M,prepSpecs)                                                            # noqa
    #   if B^-1 := VbT*sb^-1*Ub^-1 then B*B^-1 = I(gsDim)                                                                                                       # noqa
    # similarly, can get want "A^-1" such that "A^-1"(gsDim,nESpecs) A(nESpecs,gsDim) ~ Identity(gsDim)                                                         # noqa
    # or do we want not Ii*A*B = I but B*Ii*A = I(gsDim), so something like Ii = (B^-1)(A^-1) using pseudoinversese above.                                      # noqa
    #   (but we can't do this, since we only have AB, not A and B separately)                                                                                   # noqa
    # A is (trunc, gsDim)                                                                                                                                       # noqa
    # B is (gsDim, trunc)                                                                                                                                       # noqa

    # With no svd truncation (but we always truncate; this is just for reference)
    # AXB     = (nESpecs, nRhoSpecs)
    # I (=AB) = (nESpecs, nRhoSpecs)
    # A       = (nESpecs, gsDim)
    # B       = (gsDim, nRhoSpecs)

    printer = _objs.VerbosityPrinter.create_printer(verbosity)
    if target_model is None:
        raise ValueError("Must specify a target model for LGST!")

    #printer.log('', 2)
    printer.log("--- LGST ---", 1)

    #Process input parameters
    if op_labels is not None:
        op_labelsToEstimate = op_labels
    else:
        op_labelsToEstimate = list(target_model.operations.keys()) + \
            list(target_model.instruments.keys())

    rhoLabelsToEstimate = list(target_model.preps.keys())
    povmLabelsToEstimate = list(target_model.povms.keys())

    if guess_model_for_gauge is None:
        guess_model_for_gauge = target_model

    # the dimensions of the LGST matrices, called (nESpecs, nRhoSpecs),
    # are determined by the number of outcomes obtained by compiling the
    # all prepStr * effectStr sequences:
    nRhoSpecs, nESpecs, povmLbls, povmLens = _lgst_matrix_dims(
        target_model, prep_fiducials, effect_fiducials)
    K = min(nRhoSpecs, nESpecs)

    #Create truncation projector -- just trims columns (Pj) or rows (Pjt) of a matrix.
    # note K = min(nRhoSpecs,nESpecs), and dot(Pjt,Pj) == identity(trunc)
    if svd_truncate_to is None: svd_truncate_to = target_model.dim
    trunc = svd_truncate_to if svd_truncate_to > 0 else K
    assert(trunc <= K)
    Pj = _np.zeros((K, trunc), 'd')  # shape = (K, trunc) projector with only trunc columns
    for i in range(trunc): Pj[i, i] = 1.0
    Pjt = _np.transpose(Pj)         # shape = (trunc, K)

    ABMat = _construct_ab(prep_fiducials, effect_fiducials, target_model, dataset, op_label_aliases)
    # shape = (nESpecs, nRhoSpecs)

    U, s, V = _np.linalg.svd(ABMat, full_matrices=False)
    printer.log("Singular values of I_tilde (truncating to first %d of %d) = " % (trunc, len(s)), 2)
    for sval in s: printer.log(sval, 2)
    printer.log('', 2)

    Ud, Vd = _np.transpose(_np.conjugate(U)), _np.transpose(_np.conjugate(V))  # Udagger, Vdagger
    # truncate ABMat => ABMat' (note diag(s) = Ud*ABMat*Vd), shape = (trunc, trunc)
    ABMat_p = _np.dot(Pjt, _np.dot(_np.diag(s), Pj))
    # U shape = (nESpecs, K)
    # V shape = (K, nRhoSpecs)
    # Ud shape = (K, nESpecs)
    # Vd shape = (nRhoSpecs, K)

    #print "DEBUG: dataset = ",dataset
    #print "DEBUG: ABmat = \n",ABMat
    #print "DEBUG: Evals(ABmat) = \n",_np.linalg.eigvals(ABMat)
    rankAB = _np.linalg.matrix_rank(ABMat_p)
    if rankAB < ABMat_p.shape[0]:
        raise ValueError("LGST AB matrix is rank %d < %d. Choose better prep_fiducials and/or effect_fiducials, "
                         "or decrease svd_truncate_to" % (rankAB, ABMat_p.shape[0]))

    invABMat_p = _np.dot(Pjt, _np.dot(_np.diag(1.0 / s), Pj))  # (trunc,trunc)
    # check inverse is correct (TODO: comment out later)
    assert(_np.linalg.norm(_np.linalg.inv(ABMat_p) - invABMat_p) < 1e-8)
    assert(len((_np.isnan(invABMat_p)).nonzero()[0]) == 0)

    if svd_truncate_to is None or svd_truncate_to == target_model.dim:  # use target sslbls and basis
        lgstModel = _objs.ExplicitOpModel(target_model.state_space_labels, target_model.basis)
    else:  # construct a default basis for the requested dimension
        # - just act on diagonal density mx
        dumb_basis = _objs.DirectSumBasis([_objs.BuiltinBasis('gm', 1)] * svd_truncate_to)
        lgstModel = _objs.ExplicitOpModel([('L%d' % i,) for i in range(svd_truncate_to)], dumb_basis)

    for opLabel in op_labelsToEstimate:
        Xs = _construct_x_matrix(prep_fiducials, effect_fiducials, target_model, (opLabel,),
                                 dataset, op_label_aliases)  # shape (nVariants, nESpecs, nRhoSpecs)

        X_ps = []
        for X in Xs:
            # shape (K,K) this should be close to rank "svd_truncate_to" (which is <= K) -- TODO: check this
            X2 = _np.dot(Ud, _np.dot(X, Vd))

            #if svd_truncate_to > 0:
            #    printer.log("LGST DEBUG: %s before trunc to first %d row and cols = \n" % (opLabel,svd_truncate_to), 3)
            #    if printer.verbosity >= 3:
            #        _tools.print_mx(X2)
            X_p = _np.dot(Pjt, _np.dot(X2, Pj))  # truncate X => X', shape (trunc, trunc)
            X_ps.append(X_p)

        if opLabel in target_model.instruments:
            #Note: we assume leading dim of X matches instrument element ordering
            lgstModel.instruments[opLabel] = _objs.Instrument(
                [(lbl, _np.dot(invABMat_p, X_ps[i]))
                 for i, lbl in enumerate(target_model.instruments[opLabel])])
        else:
            #Just a normal gae
            assert(len(X_ps) == 1); X_p = X_ps[0]  # shape (nESpecs, nRhoSpecs)
            lgstModel.operations[opLabel] = _objs.FullDenseOp(_np.dot(invABMat_p, X_p))  # shape (trunc,trunc)

        #print "DEBUG: X(%s) = \n" % opLabel,X
        #print "DEBUG: Evals(X) = \n",_np.linalg.eigvals(X)
        #print "DEBUG: %s = \n" % opLabel,lgstModel[ opLabel ]

    #Form POVMs
    for povmLabel in povmLabelsToEstimate:
        povm_effects = []
        for effectLabel in target_model.povms[povmLabel]:
            EVec = _np.zeros((1, nRhoSpecs))
            for i, rhostr in enumerate(prep_fiducials):
                circuit = rhostr + _objs.Circuit((povmLabel,), line_labels=rhostr.line_labels)
                if circuit not in dataset and len(target_model.povms) == 1:
                    # try without povmLabel since it will be the default
                    circuit = rhostr
                dsRow_fractions = dataset[circuit].fractions
                # outcome labels should just be effect labels (no instruments!)
                EVec[0, i] = dsRow_fractions[(effectLabel,)]
            EVec_p = _np.dot(_np.dot(EVec, Vd), Pj)  # truncate Evec => Evec', shape (1,trunc)
            povm_effects.append((effectLabel, _np.transpose(EVec_p)))
        lgstModel.povms[povmLabel] = _objs.UnconstrainedPOVM(povm_effects)
        # unconstrained POVM for now - wait until after guess gauge for TP-constraining)

    # Form rhoVecs
    for prepLabel in rhoLabelsToEstimate:
        rhoVec = _np.zeros((nESpecs, 1)); eoff = 0
        for i, (estr, povmLbl, povmLen) in enumerate(zip(effect_fiducials, povmLbls, povmLens)):
            circuit = _objs.Circuit((prepLabel,), line_labels=estr.line_labels) + estr
            if circuit not in dataset and len(target_model.preps) == 1:
                # try without prepLabel since it will be the default
                circuit = estr
            dsRow_fractions = dataset[circuit].fractions
            rhoVec[eoff:eoff + povmLen, 0] = [dsRow_fractions[(ol,)] for ol in target_model.povms[povmLbl]]
            eoff += povmLen
        rhoVec_p = _np.dot(Pjt, _np.dot(Ud, rhoVec))  # truncate rhoVec => rhoVec', shape (trunc, 1)
        rhoVec_p = _np.dot(invABMat_p, rhoVec_p)
        lgstModel.preps[prepLabel] = rhoVec_p

    # Perform "guess" gauge transformation by computing the "B" matrix
    #  assuming rhos, Es, and gates are those of a guesstimate of the model
    if guess_model_for_gauge is not None:
        guessTrunc = guess_model_for_gauge.dim  # the truncation to apply to it's B matrix
        # the dimension of the model for gauge guessing cannot exceed the dimension of the model being estimated
        assert(guessTrunc <= trunc)

        guessPj = _np.zeros((K, guessTrunc), 'd')  # shape = (K, guessTrunc) projector with only trunc columns
        for i in range(guessTrunc): guessPj[i, i] = 1.0
        # guessPjt = _np.transpose(guessPj)         # shape = (guessTrunc, K)

        AMat = _construct_a(effect_fiducials, guess_model_for_gauge)    # shape = (nESpecs, gsDim)
        # AMat_p = _np.dot( guessPjt, _np.dot(Ud, AMat)) #truncate Evec => Evec', shape (guessTrunc,gsDim) (square!)

        BMat = _construct_b(prep_fiducials, guess_model_for_gauge)  # shape = (gsDim, nRhoSpecs)
        BMat_p = _np.dot(_np.dot(BMat, Vd), guessPj)  # truncate Evec => Evec', shape (gsDim,guessTrunc) (square!)

        guess_ABMat = _np.dot(AMat, BMat)
        _, guess_s, _ = _np.linalg.svd(guess_ABMat, full_matrices=False)

        printer.log("Singular values of target I_tilde (truncating to first %d of %d) = "
                    % (guessTrunc, len(guess_s)), 2)
        for sval in guess_s: printer.log(sval, 2)
        printer.log('', 2)

        if guessTrunc < trunc:
            # if the dimension of the gauge-guess model is smaller than the matrices being estimated, pad B with
            # identity
            printer.log("LGST: Padding target B with sqrt of low singular values of I_tilde: \n", 2)
            printer.log(s[guessTrunc:trunc], 2)

            BMat_p_padded = _np.identity(trunc, 'd')
            BMat_p_padded[0:guessTrunc, 0:guessTrunc] = BMat_p
            for i in range(guessTrunc, trunc):
                BMat_p_padded[i, i] = _np.sqrt(s[i])  # set diagonal as sqrt of actual AB matrix's singular values
            ggEl = _objs.FullGaugeGroupElement(_np.linalg.inv(BMat_p_padded))
            lgstModel.transform_inplace(ggEl)
        else:
            ggEl = _objs.FullGaugeGroupElement(_np.linalg.inv(BMat_p))
            lgstModel.transform_inplace(ggEl)

        # Force lgstModel to have gates, preps, & effects parameterized in the same way as those in
        # guess_model_for_gauge, but we only know how to do this when the dimensions of the target and
        # created model match.  If they don't, it doesn't make sense to increase the target model
        # dimension, as this will generally not preserve its parameterization.
        if guessTrunc == trunc:
            for opLabel in op_labelsToEstimate:
                if opLabel in guess_model_for_gauge.operations:
                    new_op = guess_model_for_gauge.operations[opLabel].copy()
                    _objs.operation.optimize_operation(new_op, lgstModel.operations[opLabel])
                    lgstModel.operations[opLabel] = new_op

            for prepLabel in rhoLabelsToEstimate:
                if prepLabel in guess_model_for_gauge.preps:
                    new_vec = guess_model_for_gauge.preps[prepLabel].copy()
                    _objs.spamvec.optimize_spamvec(new_vec, lgstModel.preps[prepLabel])
                    lgstModel.preps[prepLabel] = new_vec

            for povmLabel in povmLabelsToEstimate:
                if povmLabel in guess_model_for_gauge.povms:
                    povm = guess_model_for_gauge.povms[povmLabel]
                    new_effects = []

                    if isinstance(povm, _objs.TPPOVM):  # preserve *identity* of guess
                        for effectLabel, EVec in povm.items():
                            if effectLabel == povm.complement_label: continue
                            new_vec = EVec.copy()
                            _objs.spamvec.optimize_spamvec(new_vec, lgstModel.povms[povmLabel][effectLabel])
                            new_effects.append((effectLabel, new_vec))

                        # Construct identity vector for complement effect vector
                        #  Pad with zeros if needed (ROBIN - is this correct?)
                        identity = povm[povm.complement_label].identity
                        Idim = identity.shape[0]
                        assert(Idim <= trunc)
                        if Idim < trunc:
                            padded_identityVec = _np.concatenate((identity, _np.zeros((trunc - Idim, 1), 'd')))
                        else:
                            padded_identityVec = identity
                        comp_effect = padded_identityVec - sum([v for k, v in new_effects])
                        new_effects.append((povm.complement_label, comp_effect))  # add complement
                        lgstModel.povms[povmLabel] = _objs.TPPOVM(new_effects)

                    else:  # just create an unconstrained POVM
                        for effectLabel, EVec in povm.items():
                            new_vec = EVec.copy()
                            _objs.spamvec.optimize_spamvec(new_vec, lgstModel.povms[povmLabel][effectLabel])
                            new_effects.append((effectLabel, new_vec))
                        lgstModel.povms[povmLabel] = _objs.UnconstrainedPOVM(new_effects)

            #Also convey default gauge group & simulator from guess_model_for_gauge
            lgstModel.default_gauge_group = \
                guess_model_for_gauge.default_gauge_group
            lgstModel.sim = guess_model_for_gauge.sim.copy()

        #inv_BMat_p = _np.dot(invABMat_p, AMat_p) # should be equal to inv(BMat_p) when trunc == gsDim ?? check??
        # # lgstModel had dim trunc, so after transform is has dim gsDim
        #lgstModel.transform_inplace( S=_np.dot(invABMat_p, AMat_p), Si=BMat_p )

    printer.log("Resulting model:\n", 3)
    printer.log(lgstModel, 3)
    #    for line in str(lgstModel).split('\n'):
    #       printer.log(line, 3)
    return lgstModel


def _lgst_matrix_dims(model, prep_fiducials, effect_fiducials):
    assert(model is not None), "LGST matrix construction requires a non-None Model!"
    nRhoSpecs = len(prep_fiducials)  # no instruments allowed in prep_fiducials
    povmLbls = [model.split_circuit(s, ('povm',))[2]  # povm_label
                for s in effect_fiducials]
    povmLens = ([len(model.povms[l]) for l in povmLbls])
    nESpecs = sum(povmLens)
    return nRhoSpecs, nESpecs, povmLbls, povmLens


def _construct_ab(prep_fiducials, effect_fiducials, model, dataset, op_label_aliases=None):
    nRhoSpecs, nESpecs, povmLbls, povmLens = _lgst_matrix_dims(
        model, prep_fiducials, effect_fiducials)

    AB = _np.empty((nESpecs, nRhoSpecs))
    eoff = 0
    for i, (estr, povmLen) in enumerate(zip(effect_fiducials, povmLens)):
        for j, rhostr in enumerate(prep_fiducials):
            opLabelString = rhostr + estr  # LEXICOGRAPHICAL VS MATRIX ORDER
            dsStr = opLabelString.replace_layers_with_aliases(op_label_aliases)
            expd_circuit_outcomes = opLabelString.expand_instruments_and_separate_povm(model)
            assert(len(expd_circuit_outcomes) == 1), "No instruments are allowed in LGST fiducials!"
            unique_key = next(iter(expd_circuit_outcomes.keys()))
            outcomes = expd_circuit_outcomes[unique_key]
            assert(len(outcomes) == povmLen)

            dsRow_fractions = dataset[dsStr].fractions
            AB[eoff:eoff + povmLen, j] = [dsRow_fractions.get(ol, 0.0) for ol in outcomes]
        eoff += povmLen

    return AB


def _construct_x_matrix(prep_fiducials, effect_fiducials, model, op_label_tuple, dataset, op_label_aliases=None):
    nRhoSpecs, nESpecs, povmLbls, povmLens = _lgst_matrix_dims(
        model, prep_fiducials, effect_fiducials)

    nVariants = 1
    for g in op_label_tuple:
        if g in model.instruments:
            nVariants *= len(model.instruments[g])

    X = _np.empty((nVariants, nESpecs, nRhoSpecs))  # multiple "X" matrix variants b/c of instruments

    eoff = 0  # effect-dimension offset
    for i, (estr, povmLen) in enumerate(zip(effect_fiducials, povmLens)):
        for j, rhostr in enumerate(prep_fiducials):
            opLabelString = rhostr + _objs.Circuit(op_label_tuple, line_labels=rhostr.line_labels) + estr
            dsStr = opLabelString.replace_layers_with_aliases(op_label_aliases)
            expd_circuit_outcomes = opLabelString.expand_instruments_and_separate_povm(model)
            dsRow_fractions = dataset[dsStr].fractions
            assert(len(expd_circuit_outcomes) == nVariants)

            for k, (sep_povm_c, outcomes) in enumerate(expd_circuit_outcomes.items()):
                assert(len(outcomes) == povmLen)
                X[k, eoff:eoff + povmLen, j] = [dsRow_fractions.get(ol, 0) for ol in outcomes]
        eoff += povmLen

    return X


def _construct_a(effect_fiducials, model):
    _, n, povmLbls, povmLens = _lgst_matrix_dims(
        model, [], effect_fiducials)

    dim = model.dim
    A = _np.empty((n, dim))
    # st = _np.empty(dim, 'd')

    basis_st = _np.zeros((dim, 1), 'd'); eoff = 0
    for k, (estr, povmLbl, povmLen) in enumerate(zip(effect_fiducials, povmLbls, povmLens)):
        #Build fiducial < E_k | := < EVec[ effectSpec[0] ] | Circuit(effectSpec[1:])
        #st = dot(Ek.T, Estr) = ( dot(Estr.T,Ek)  ).T
        #A[k,:] = st[0,:] # E_k == kth row of A
        for i in range(dim):  # propagate each basis initial state
            basis_st[i] = 1.0
            model.preps['rho_LGST_tmp'] = basis_st
            probs = model.probabilities(_objs.Circuit(('rho_LGST_tmp',), line_labels=estr.line_labels) + estr)
            A[eoff:eoff + povmLen, i] = [probs[(ol,)] for ol in model.povms[povmLbl]]  # CHECK will this work?
            del model.preps['rho_LGST_tmp']
            basis_st[i] = 0.0

        eoff += povmLen
    return A


def _construct_b(prep_fiducials, model):
    n = len(prep_fiducials)
    dim = model.dim
    B = _np.empty((dim, n))
    # st = _np.empty(dim, 'd')

    #Create POVM of vector units
    basis_Es = []
    for i in range(dim):  # propagate each basis initial state
        basis_E = _np.zeros((dim, 1), 'd')
        basis_E[i] = 1.0
        basis_Es.append(basis_E)
    model.povms['M_LGST_tmp_povm'] = _objs.UnconstrainedPOVM(
        [("E%d" % i, E) for i, E in enumerate(basis_Es)])

    for k, rhostr in enumerate(prep_fiducials):
        #Build fiducial | rho_k > := Circuit(prepSpec[0:-1]) | rhoVec[ prepSpec[-1] ] >
        # B[:,k] = st[:,0] # rho_k == kth column of B
        probs = model.probabilities(rhostr + _objs.Circuit(('M_LGST_tmp_povm',), line_labels=rhostr.line_labels))
        B[:, k] = [probs[("E%d" % i,)] for i in range(dim)]  # CHECK will this work?

    del model.povms['M_LGST_tmp_povm']
    return B


def _construct_target_ab(prep_fiducials, effect_fiducials, target_model):
    nRhoSpecs, nESpecs, povmLbls, povmLens = _lgst_matrix_dims(
        target_model, prep_fiducials, effect_fiducials)

    AB = _np.empty((nESpecs, nRhoSpecs))
    eoff = 0
    for i, (estr, povmLbl, povmLen) in enumerate(zip(effect_fiducials, povmLbls, povmLens)):
        for j, rhostr in enumerate(prep_fiducials):
            opLabelString = rhostr + estr  # LEXICOGRAPHICAL VS MATRIX ORDER
            probs = target_model.probabilities(opLabelString)
            AB[eoff:eoff + povmLen, j] = \
                [probs[(ol,)] for ol in target_model.povms[povmLbl]]
            # outcomes (keys of probs) should just be povm effect labels
            # since no instruments are allowed in fiducial strings.
        eoff += povmLen

    return AB


def gram_rank_and_eigenvalues(dataset, prep_fiducials, effect_fiducials, target_model):
    """
    Returns the rank and singular values of the Gram matrix for a dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to populate the Gram matrix

    prep_fiducials : list of Circuits
        Fiducial Circuits used to construct a informationally complete
        effective preparation.

    effect_fiducials : list of Circuits
        Fiducial Circuits used to construct a informationally complete
        effective measurement.

    target_model : Model
        A model used to make sense of circuit elements, and to compute the
        theoretical gram matrix eigenvalues (returned as `svalues_target`).

    Returns
    -------
    rank : int
        the rank of the Gram matrix
    svalues : numpy array
        the singular values of the Gram matrix
    svalues_target : numpy array
        the corresponding singular values of the Gram matrix
        generated by target_model.
    """
    if target_model is None: raise ValueError("Must supply `target_model`")
    ABMat = _construct_ab(prep_fiducials, effect_fiducials, target_model, dataset)
    _, s, _ = _np.linalg.svd(ABMat)

    ABMat_tgt = _construct_target_ab(prep_fiducials, effect_fiducials, target_model)
    _, s_tgt, _ = _np.linalg.svd(ABMat_tgt)

    return _np.linalg.matrix_rank(ABMat), s, s_tgt  # _np.linalg.eigvals(ABMat)


##################################################################################
#                 Long sequence GST
##################################################################################

def run_gst_fit_simple(dataset, start_model, circuits, optimizer, objective_function_builder,
                       resource_alloc, verbosity=0):
    """
    Performs core Gate Set Tomography function of model optimization.

    Optimizes the parameters of `start_model` by minimizing the objective function
    built by `objective_function_builder`.  Probabilities are computed by the model,
    and outcome counts are supplied by `dataset`.

    Parameters
    ----------
    dataset : DataSet
        The dataset to obtain counts from.

    start_model : Model
        The Model used as a starting point for the least-squares
        optimization.

    circuits : list of (tuples or Circuits)
        Each tuple contains operation labels and specifies a circuit whose
        probabilities are considered when trying to least-squares-fit the
        probabilities given in the dataset.
        e.g. [ (), ('Gx',), ('Gx','Gy') ]

    optimizer : Optimizer or dict
        The optimizer to use, or a dictionary of optimizer parameters
        from which a default optimizer can be built.

    objective_function_builder : ObjectiveFunctionBuilder
        Defines the objective function that is optimized.  Can also be anything
        readily converted to an objective function builder, e.g. `"logl"`.

    resource_alloc : ResourceAllocation
        A resource allocation object containing information about how to
        divide computation amongst multiple processors and any memory
        limits that should be imposed.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    result : OptimizerResult
        the result of the optimization
    model : Model
        the best-fit model.
    """
    optimizer = optimizer if isinstance(optimizer, _Optimizer) else _CustomLMOptimizer.cast(optimizer)
    objective_function_builder = _objs.ObjectiveFunctionBuilder.cast(objective_function_builder)
    array_types = optimizer.array_types + \
        objective_function_builder.compute_array_types(optimizer.called_objective_methods, start_model.sim)

    mdc_store = _objs.ModelDatasetCircuitsStore(start_model, dataset, circuits, resource_alloc,
                                                array_types=array_types, verbosity=verbosity)
    result, mdc_store = run_gst_fit(mdc_store, optimizer, objective_function_builder, verbosity)
    return result, mdc_store.model


def run_gst_fit(mdc_store, optimizer, objective_function_builder, verbosity=0):
    """
    Performs core Gate Set Tomography function of model optimization.

    Optimizes the model to the data within `mdc_store` by minimizing the objective function
    built by `objective_function_builder`.

    Parameters
    ----------
    mdc_store : ModelDatasetCircuitsStore
        An object holding a model, data set, and set of circuits.  This defines the model
        to be optimized, the data to fit to, and the circuits where predicted vs. observed
        comparisons should be made.  This object also contains additional information specific
        to the given model, data set, and circuit list, doubling as a cache for increased performance.
        This information is also specific to a particular resource allocation, which affects how
        cached values stored.

    optimizer : Optimizer or dict
        The optimizer to use, or a dictionary of optimizer parameters
        from which a default optimizer can be built.

    objective_function_builder : ObjectiveFunctionBuilder
        Defines the objective function that is optimized.  Can also be anything
        readily converted to an objective function builder, e.g. `"logl"`.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    result : OptimizerResult
        the result of the optimization
    objfn_store : MDCObjectiveFunction
        the objective function and store containing the best-fit model evaluated at the best-fit point.
    """
    optimizer = optimizer if isinstance(optimizer, _Optimizer) else _CustomLMOptimizer.cast(optimizer)
    comm = mdc_store.resource_alloc.comm
    profiler = mdc_store.resource_alloc.profiler
    printer = _objs.VerbosityPrinter.create_printer(verbosity, comm)

    tStart = _time.time()

    if comm is not None:
        #assume all models at least have same parameters - so just compare vecs
        v_cmp = comm.bcast(mdc_store.model.to_vector() if (comm.Get_rank() == 0) else None, root=0)
        if _np.linalg.norm(mdc_store.model.to_vector() - v_cmp) > 1e-6:
            raise ValueError("MPI ERROR: *different* MC2GST start models"
                             " given to different processors!")                   # pragma: no cover

    #MEM from ..objects.profiler import Profiler
    #MEM debug_prof = Profiler(comm)
    #MEM debug_prof.print_memory("run_gst_fit1", True)

    objective_function_builder = _objs.ObjectiveFunctionBuilder.cast(objective_function_builder)
    #MEM debug_prof.print_memory("run_gst_fit2", True)
    objective = objective_function_builder.build_from_store(mdc_store, printer)  # (objective is *also* a store)
    #MEM debug_prof.print_memory("run_gst_fit3", True)
    profiler.add_time("run_gst_fit: pre-opt", tStart)
    printer.log("--- %s GST ---" % objective.name, 1)

    #Step 3: solve least squares minimization problem
    if isinstance(objective.model.sim, _TermFSim):  # could have used mdc_store.model (it's the same model)
        opt_result = _do_term_runopt(objective, optimizer, printer)
    else:
        #Normal case of just a single "sub-iteration"
        opt_result = _do_runopt(objective, optimizer, printer)

    printer.log("Completed in %.1fs" % (_time.time() - tStart), 1)

    #if target_model is not None:
    #  target_vec = target_model.to_vector()
    #  targetErrVec = _objective_func(target_vec)
    #  return minErrVec, soln_gs, targetErrVec
    profiler.add_time("do_mc2gst: total time", tStart)
    #TODO: evTree.permute_computation_to_original(minErrVec) #Doesn't work b/c minErrVec is flattened
    # but maybe best to just remove minErrVec from return value since this isn't very useful
    # anyway?
    return opt_result, objective


def run_iterative_gst(dataset, start_model, circuit_lists,
                      optimizer, iteration_objfn_builders, final_objfn_builders,
                      resource_alloc, verbosity=0):
    """
    Performs Iterative Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to generate MLGST gate estimates

    start_model : Model
        The Model used as a starting point for the least-squares
        optimization.

    circuit_lists : list of lists of (tuples or Circuits)
        The i-th element is a list of the circuits to be used in the i-th iteration
        of the optimization.  Each element of these lists is a circuit, specifed as
        either a Circuit object or as a tuple of operation labels (but all must be specified
        using the same type).
        e.g. [ [ (), ('Gx',) ], [ (), ('Gx',), ('Gy',) ], [ (), ('Gx',), ('Gy',), ('Gx','Gy') ]  ]

    optimizer : Optimizer or dict
        The optimizer to use, or a dictionary of optimizer parameters
        from which a default optimizer can be built.

    iteration_objfn_builders : list
        List of ObjectiveFunctionBuilder objects defining which objective functions
        should be optimizized (successively) on each iteration.

    final_objfn_builders : list
        List of ObjectiveFunctionBuilder objects defining which objective functions
        should be optimizized (successively) on the final iteration.

    resource_alloc : ResourceAllocation
        A resource allocation object containing information about how to
        divide computation amongst multiple processors and any memory
        limits that should be imposed.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    models : list of Models
        list whose i-th element is the model corresponding to the results
        of the i-th iteration.
    optimums : list of OptimizerResults
        list whose i-th element is the final optimizer result from that iteration.
    final_store : MDSObjectiveFunction
        The final iteration's objective function / store, which encapsulated the final objective
        function evaluated at the best-fit point (an "evaluated" model-dataSet-circuits store).
    """
    resource_alloc = _ResourceAllocation.cast(resource_alloc)
    optimizer = optimizer if isinstance(optimizer, _Optimizer) else _CustomLMOptimizer.cast(optimizer)
    comm = resource_alloc.comm
    profiler = resource_alloc.profiler
    printer = _objs.VerbosityPrinter.create_printer(verbosity, comm)

    models = []; optimums = []
    mdl = start_model.copy(); nIters = len(circuit_lists)
    tStart = _time.time()
    tRef = tStart
    final_store = None

    iteration_objfn_builders = [_objs.ObjectiveFunctionBuilder.cast(ofb) for ofb in iteration_objfn_builders]
    final_objfn_builders = [_objs.ObjectiveFunctionBuilder.cast(ofb) for ofb in final_objfn_builders]

    def _max_array_types(artypes_list):  # get the maximum number of each array type and return as an array-types tuple
        max_cnts = {}
        for artypes in artypes_list:
            cnts = _collections.defaultdict(lambda: 0)
            for artype in artypes:
                cnts[artype] += 1
            for artype, cnt in cnts.items(): max_cnts[artype] = max(max_cnts.get(artype, 0), cnt)
        ret = ()
        for artype, cnt in max_cnts.items(): ret += (artype,) * cnt
        return ret

    with printer.progress_logging(1):
        for (i, circuitsToEstimate) in enumerate(circuit_lists):
            extraMessages = []
            if isinstance(circuitsToEstimate, _CircuitList) and circuitsToEstimate.name:
                extraMessages.append("(%s) " % circuitsToEstimate.name)

            printer.show_progress(i, nIters, verbose_messages=extraMessages,
                                  prefix="--- Iterative GST:", suffix=" %d circuits ---" % len(circuitsToEstimate))

            if circuitsToEstimate is None or len(circuitsToEstimate) == 0: continue

            mdl.basis = start_model.basis  # set basis in case of CPTP constraints (needed?)
            method_names = optimizer.called_objective_methods
            array_types = optimizer.array_types + \
                _max_array_types([builder.compute_array_types(method_names, mdl.sim)
                                  for builder in iteration_objfn_builders + final_objfn_builders])
            mdc_store = _objs.ModelDatasetCircuitsStore(mdl, dataset, circuitsToEstimate, resource_alloc,
                                                        array_types=array_types, verbosity=printer - 1)

            for j, obj_fn_builder in enumerate(iteration_objfn_builders):
                tNxt = _time.time()
                optimizer.fditer = optimizer.first_fditer if (i == 0 and j == 0) else 0
                opt_result, mdc_store = run_gst_fit(mdc_store, optimizer, obj_fn_builder, printer - 1)
                profiler.add_time('run_iterative_gst: iter %d %s-opt' % (i + 1, obj_fn_builder.name), tNxt)

            tNxt = _time.time()
            printer.log("Iteration %d took %.1fs\n" % (i + 1, tNxt - tRef), 2)
            tRef = tNxt

            if i == len(circuit_lists) - 1:  # the last iteration
                printer.log("Last iteration:", 2)

                for j, obj_fn_builder in enumerate(final_objfn_builders):
                    tNxt = _time.time()
                    mdl.basis = start_model.basis
                    opt_result, mdl = run_gst_fit(mdc_store, optimizer, obj_fn_builder, printer - 1)
                    profiler.add_time('run_iterative_gst: final %s opt' % obj_fn_builder.name, tNxt)

                tNxt = _time.time()
                printer.log("Final optimization took %.1fs\n" % (tNxt - tRef), 2)
                tRef = tNxt

                #send final cache back to caller to facilitate more operations on the final (model, circuits, dataset)
                final_store = mdc_store
                models.append(mdc_store.model)  # don't copy so `mdc_store.model` *is* the final model, `models[-1]`
            else:
                models.append(mdc_store.model.copy())

            optimums.append(opt_result)

    printer.log('Iterative GST Total Time: %.1fs' % (_time.time() - tStart))
    profiler.add_time('run_iterative_gst: total time', tStart)
    return models, optimums, final_store


def _do_runopt(objective, optimizer, printer):
    """
    Runs the core model-optimization step within a GST routine by optimizing
    `objective` using `optimizer`.

    This is factored out as a separate function because of the differences
    when running Taylor-term simtype calculations, which utilize this
    as a subroutine (see :function:`_do_term_runopt`).

    Parameters
    ----------
    objective : MDSObjectiveFunction
        A "model-dataset" objective function to optimize.

    optimizer : Optimizer
        The optimizer to use.

    printer : VerbosityPrinter
        An object for printing output.

    Returns
    -------
    OptimizerResult
    """

    mdl = objective.model
    resource_alloc = objective.resource_alloc
    profiler = resource_alloc.profiler

    #Perform actual optimization
    tm = _time.time()
    opt_result = optimizer.run(objective, resource_alloc.comm, profiler, printer)
    profiler.add_time("run_gst_fit: optimize", tm)

    if printer.verbosity > 0:
        #Don't compute num gauge params if it's expensive (>10% of mem limit) or unavailable
        if hasattr(mdl, 'num_elements'):
            memForNumGaugeParams = mdl.num_elements * (mdl.num_params + mdl.dim**2) \
                * _np.dtype('d').itemsize  # see Model._buildup_dpg (this is mem for dPG)

            if resource_alloc.mem_limit is None or 0.1 * resource_alloc.mem_limit < memForNumGaugeParams:
                try:
                    nModelParams = mdl.num_nongauge_params  # len(x0)
                except:  # numpy can throw a LinAlgError or sparse cases can throw a NotImplementedError
                    printer.warning("Could not obtain number of *non-gauge* parameters - using total params instead")
                    nModelParams = mdl.num_params
            else:
                printer.log("Finding num_nongauge_params is too expensive: using total params.")
                nModelParams = mdl.num_params  # just use total number of params
        else:
            nModelParams = mdl.num_params  # just use total number of params

        #Get number of maximal-model parameter ("dataset params") if needed for print messages
        # -> number of independent parameters in dataset (max. model # of params)
        tm = _time.time()
        nDataParams = objective.num_data_params()  # TODO - cache this somehow in term-based calcs...
        profiler.add_time("run_gst_fit: num data params", tm)

        chi2_k_qty = opt_result.chi2_k_distributed_qty  # total chi2 or 2*deltaLogL
        desc = objective.description
        # reject GST model if p-value < threshold (~0.05?)
        pvalue = 1.0 - _stats.chi2.cdf(chi2_k_qty, nDataParams - nModelParams)
        printer.log("%s = %g (%d data params - %d model params = expected mean of %g; p-value = %g)" %
                    (desc, chi2_k_qty, nDataParams, nModelParams, nDataParams - nModelParams, pvalue), 1)

    return opt_result


def _do_term_runopt(objective, optimizer, printer):
    """
    Runs the core model-optimization step for models using the
    Taylor-term (path integral) method of computing probabilities.

    This routine serves the same purpose as :function:`_do_runopt`, but
    is more complex because an appropriate "path set" must be found,
    requiring a loop of model optimizations with fixed path sets until
    a sufficient "good" path set is obtained.

    Parameters
    ----------
    objective : MDSObjectiveFunction
        A "model-dataset" objective function to optimize.

    optimizer : Optimizer
        The optimizer to use.

    printer : VerbosityPrinter
        An object for printing output.

    Returns
    -------
    OptimizerResult
    """

    mdl = objective.model
    fwdsim = mdl.sim

    #MEM from ..objects.profiler import Profiler
    #MEM debug_prof = Profiler(objective.resource_alloc.comm)
    #MEM debug_prof.print_memory("do_term_runopt1", True)

    #Pipe these parameters in from fwdsim, even though they're used to control the term-stage loop
    maxTermStages = fwdsim.max_term_stages
    pathFractionThreshold = fwdsim.path_fraction_threshold  # 0 when not using path-sets
    oob_check_interval = fwdsim.oob_check_interval

    #assume a path set has already been chosen, (one should have been chosen when layout was created)
    layout = objective.layout

    resource_alloc = objective.resource_alloc
    pathSet = layout.pathset(resource_alloc.comm)
    if pathSet:  # only some types of term "modes" (see fwdsim.mode) use path-sets
        pathFraction = pathSet.allowed_path_fraction

        objective.lsvec()  #ensure objective.probs initialized
        bSufficient = fwdsim.bulk_test_if_paths_are_sufficient(layout, objective.probs, objective.resource_alloc, verbosity=0)

        printer.log("Initial Term-stage model has %d failures and uses %.1f%% of allowed paths (ok=%s)." %
                    (pathSet.num_failures, 100 * pathFraction, str(bSufficient)))

        while not bSufficient:
            # Backtrack toward all zeros param vector (maybe backtrack along path optimizer took in future?)
            #  in hopes of finding a place to begin where paths are sufficient.
            mdl.from_vector(0.9 * mdl.to_vector())  # contract paramvector toward zero

            #Adapting the path set doesn't seem necessary (and takes time), but we could do this:
            #new_pathSet = mdl.sim.find_minimal_paths_set(layout, resource_alloc)  # `mdl.sim` instead of `fwdsim` to
            #mdl.sim.select_paths_set(layout, new_pathSet, resource_alloc)  # ensure paramvec is updated
            #pathFraction = pathSet.allowed_path_fraction
            #printer.log("  After adapting paths, num failures = %d, %.1f%% of allowed paths used." %
            #            (pathSet.num_failures, 100 * pathFraction))

            obj_val = objective.fn()  #ensure objective.probs initialized
            bSufficient = fwdsim.bulk_test_if_paths_are_sufficient(layout, objective.probs, objective.resource_alloc, verbosity=0)
            printer.log("Looking for initial model where paths are sufficient: |paramvec| = %g, obj=%g (ok=%s)"
                        % (_np.linalg.norm(mdl.to_vector()), obj_val, str(bSufficient)))
    else:
        pathFraction = 1.0  # b/c "all" paths are used, and > pathFractionThreshold, which should be 0

    opt_result = None
    #MEM debug_prof.print_memory("do_term_runopt2", True)
    for sub_iter in range(maxTermStages):

        bFinalIter = (sub_iter == maxTermStages - 1) or (pathFraction > pathFractionThreshold)
        optimizer.oob_check_interval = oob_check_interval
        # don't stop early on last iter - do as much as possible.
        optimizer.oob_action = "reject" if bFinalIter else "stop"
        opt_result = _do_runopt(objective, optimizer, printer)

        if not opt_result.optimizer_specific_qtys['msg'] == "Objective function out-of-bounds! STOP":
            if not bFinalIter:
                printer.log("Term-states Converged!")  # we're done! the path integrals used were sufficient.
            elif pathFraction > pathFractionThreshold:
                printer.log("Last term-stage used %.1f%% > %.0f%% of allowed paths, so exiting."
                            % (100 * pathFraction, 100 * pathFractionThreshold))
            else:
                printer.log("Max num of term-stages (%d) reached." % maxTermStages)
            break  # subiterations have "converged", i.e. there are no failures in prepping => enough paths kept

        else:
            # Try to get more paths if we can and use those regardless of whether there are failures
            #MEM debug_prof.print_memory("do_term_runopt3", True)
            pathSet = mdl.sim.find_minimal_paths_set(layout, resource_alloc)  # `mdl.sim` instead of `fwdsim` to
            #MEM debug_prof.print_memory("do_term_runopt4", True)
            mdl.sim.select_paths_set(layout, pathSet, resource_alloc)  # ensure paramvec is updated
            #MEM debug_prof.print_memory("do_term_runopt5", True)
            pathFraction = pathSet.allowed_path_fraction
            optimizer.init_munu = opt_result.optimizer_specific_qtys['mu'], opt_result.optimizer_specific_qtys['nu']
            printer.log("After adapting paths, num failures = %d, %.1f%% of allowed paths used." %
                        (pathSet.num_failures, 100 * pathFraction))

    return opt_result


###################################################################################
#                 Other Tools
###################################################################################


def find_closest_unitary_opmx(operation_mx):
    """
    Find the closest (in fidelity) unitary superoperator to `operation_mx`.

    Finds the closest operation matrix (by maximizing fidelity)
    to `operation_mx` that describes a unitary quantum gate.

    Parameters
    ----------
    operation_mx : numpy array
        The operation matrix to act on.

    Returns
    -------
    numpy array
        The resulting closest unitary operation matrix.
    """

    gate_JMx = _tools.jamiolkowski_iso(operation_mx, choi_mx_basis="std")
    # d = _np.sqrt(operation_mx.shape[0])
    # I = _np.identity(d)

    #def getu_1q(basisVec):  # 1 qubit version
    #    return _spl.expm( 1j * (basisVec[0]*_tools.sigmax + basisVec[1]*_tools.sigmay + basisVec[2]*_tools.sigmaz) )
    def _get_gate_mx_1q(basis_vec):  # 1 qubit version
        return _pc.single_qubit_gate(basis_vec[0],
                                     basis_vec[1],
                                     basis_vec[2])

    if operation_mx.shape[0] == 4:
        #bell = _np.transpose(_np.array( [[1,0,0,1]] )) / _np.sqrt(2)
        initialBasisVec = [0, 0, 0]  # start with I until we figure out how to extract target unitary
        #getU = getu_1q
        getGateMx = _get_gate_mx_1q
    # Note: seems like for 2 qubits bell = [1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ]/sqrt(4)
    # (4 zeros between 1's since state dimension is 4 ( == sqrt(gate dimension))
    else:
        raise ValueError("Can't get closest unitary for > 1 qubits yet -- need to generalize.")

    def _objective_func(basis_vec):
        operation_mx = getGateMx(basis_vec)
        JU = _tools.jamiolkowski_iso(operation_mx, choi_mx_basis="std")
        # OLD: but computes JU in Pauli basis (I think) -> wrong matrix to fidelity check with gate_JMx
        #U = getU(basis_vec)
        #vU = _np.dot( _np.kron(U,I), bell ) # "Choi vector" corresponding to unitary U
        #JU = _np.kron( vU, _np.transpose(_np.conjugate(vU))) # Choi matrix corresponding to U
        return -_tools.fidelity(gate_JMx, JU)

    # print_obj_func = _opt.create_objfn_printer(_objective_func)
    solution = _spo.minimize(_objective_func, initialBasisVec, options={'maxiter': 10000},
                             method='Nelder-Mead', callback=None, tol=1e-8)  # if verbosity > 2 else None
    operation_mx = getGateMx(solution.x)

    #print "DEBUG: Best fidelity = ",-solution.fun
    #print "DEBUG: Using vector = ", solution.x
    #print "DEBUG: Gate Mx = \n", operation_mx
    #print "DEBUG: Chi Mx = \n", _tools.jamiolkowski_iso( operation_mx)
    #return -solution.fun, operation_mx
    return operation_mx
