""" Core GST algorithms """
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

from .. import optimize as _opt
from .. import tools as _tools
from .. import objects as _objs
from .. import construction as _pc
from ..objects import objectivefns as _objfns
from ..objects.profiler import DummyProfiler as _DummyProfiler
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


def do_lgst(dataset, prepStrs, effectStrs, targetModel, opLabels=None, opLabelAliases=None,
            guessModelForGauge=None, svdTruncateTo=None, verbosity=0):
    """
    Performs Linear-inversion Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to generate the LGST estimates

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    targetModel : Model
        A model used to specify which operation labels should be estimated, a
        guess for which gauge these estimates should be returned in, and
        used to simplify operation sequences.

    opLabels : list, optional
        A list of which operation labels (or aliases) should be estimated.
        Overrides the operation labels in targetModel.
        e.g. ['Gi','Gx','Gy','Gx2']

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are circuits
        corresponding to what that operation label should be expanded into before querying
        the dataset.  Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = pygsti.obj.Circuit(['Gx','Gx','Gx'])

    guessModelForGauge : Model, optional
        A model used to compute a gauge transformation that is applied to
        the LGST estimates before they are returned.  This gauge transformation
        is computed such that if the estimated gates matched the model given,
        then the operation matrices would match, i.e. the gauge would be the same as
        the model supplied.
        Defaults to targetModel.

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. Zero means no truncation.
        Defaults to dimension of `targetModel`.

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

    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    if targetModel is None:
        raise ValueError("Must specify a target model for LGST!")

    #printer.log('', 2)
    printer.log("--- LGST ---", 1)

    #Process input parameters
    if opLabels is not None:
        opLabelsToEstimate = opLabels
    else:
        opLabelsToEstimate = list(targetModel.operations.keys()) + \
            list(targetModel.instruments.keys())

    rhoLabelsToEstimate = list(targetModel.preps.keys())
    povmLabelsToEstimate = list(targetModel.povms.keys())

    if guessModelForGauge is None:
        guessModelForGauge = targetModel

    # the dimensions of the LGST matrices, called (nESpecs, nRhoSpecs),
    # are determined by the number of outcomes obtained by compiling the
    # all prepStr * effectStr sequences:
    nRhoSpecs, nESpecs, povmLbls, povmLens = _lgst_matrix_dims(
        targetModel, prepStrs, effectStrs)
    K = min(nRhoSpecs, nESpecs)

    #Create truncation projector -- just trims columns (Pj) or rows (Pjt) of a matrix.
    # note K = min(nRhoSpecs,nESpecs), and dot(Pjt,Pj) == identity(trunc)
    if svdTruncateTo is None: svdTruncateTo = targetModel.dim
    trunc = svdTruncateTo if svdTruncateTo > 0 else K
    assert(trunc <= K)
    Pj = _np.zeros((K, trunc), 'd')  # shape = (K, trunc) projector with only trunc columns
    for i in range(trunc): Pj[i, i] = 1.0
    Pjt = _np.transpose(Pj)         # shape = (trunc, K)

    ABMat = _constructAB(prepStrs, effectStrs, targetModel, dataset, opLabelAliases)  # shape = (nESpecs, nRhoSpecs)

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
        raise ValueError("LGST AB matrix is rank %d < %d. Choose better prepStrs and/or effectStrs, "
                         "or decrease svdTruncateTo" % (rankAB, ABMat_p.shape[0]))

    invABMat_p = _np.dot(Pjt, _np.dot(_np.diag(1.0 / s), Pj))  # (trunc,trunc)
    # check inverse is correct (TODO: comment out later)
    assert(_np.linalg.norm(_np.linalg.inv(ABMat_p) - invABMat_p) < 1e-8)
    assert(len((_np.isnan(invABMat_p)).nonzero()[0]) == 0)

    if svdTruncateTo is None or svdTruncateTo == targetModel.dim:  # use target sslbls and basis
        lgstModel = _objs.ExplicitOpModel(targetModel.state_space_labels, targetModel.basis)
    else:  # construct a default basis for the requested dimension
        # - just act on diagonal density mx
        dumb_basis = _objs.DirectSumBasis([_objs.BuiltinBasis('gm', 1)] * svdTruncateTo)
        lgstModel = _objs.ExplicitOpModel([('L%d' % i,) for i in range(svdTruncateTo)], dumb_basis)

    for opLabel in opLabelsToEstimate:
        Xs = _constructXMatrix(prepStrs, effectStrs, targetModel, (opLabel,),
                               dataset, opLabelAliases)  # shape (nVariants, nESpecs, nRhoSpecs)

        X_ps = []
        for X in Xs:
            # shape (K,K) this should be close to rank "svdTruncateTo" (which is <= K) -- TODO: check this
            X2 = _np.dot(Ud, _np.dot(X, Vd))

            #if svdTruncateTo > 0:
            #    printer.log("LGST DEBUG: %s before trunc to first %d row and cols = \n" % (opLabel,svdTruncateTo), 3)
            #    if printer.verbosity >= 3:
            #        _tools.print_mx(X2)
            X_p = _np.dot(Pjt, _np.dot(X2, Pj))  # truncate X => X', shape (trunc, trunc)
            X_ps.append(X_p)

        if opLabel in targetModel.instruments:
            #Note: we assume leading dim of X matches instrument element ordering
            lgstModel.instruments[opLabel] = _objs.Instrument(
                [(lbl, _np.dot(invABMat_p, X_ps[i]))
                 for i, lbl in enumerate(targetModel.instruments[opLabel])])
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
        for effectLabel in targetModel.povms[povmLabel]:
            EVec = _np.zeros((1, nRhoSpecs))
            for i, rhostr in enumerate(prepStrs):
                circuit = rhostr + _objs.Circuit((povmLabel,), line_labels=rhostr.line_labels)
                if circuit not in dataset and len(targetModel.povms) == 1:
                    # try without povmLabel since it will be the default
                    circuit = rhostr
                dsRow = dataset[circuit]
                # outcome labels should just be effect labels (no instruments!)
                EVec[0, i] = dsRow.fraction((effectLabel,))
            EVec_p = _np.dot(_np.dot(EVec, Vd), Pj)  # truncate Evec => Evec', shape (1,trunc)
            povm_effects.append((effectLabel, _np.transpose(EVec_p)))
        lgstModel.povms[povmLabel] = _objs.UnconstrainedPOVM(povm_effects)
        # unconstrained POVM for now - wait until after guess gauge for TP-constraining)

    # Form rhoVecs
    for prepLabel in rhoLabelsToEstimate:
        rhoVec = _np.zeros((nESpecs, 1)); eoff = 0
        for i, (estr, povmLbl, povmLen) in enumerate(zip(effectStrs, povmLbls, povmLens)):
            circuit = _objs.Circuit((prepLabel,), line_labels=estr.line_labels) + estr
            if circuit not in dataset and len(targetModel.preps) == 1:
                # try without prepLabel since it will be the default
                circuit = estr
            dsRow = dataset[circuit]
            rhoVec[eoff:eoff + povmLen, 0] = [dsRow.fraction((ol,)) for ol in targetModel.povms[povmLbl]]
            eoff += povmLen
        rhoVec_p = _np.dot(Pjt, _np.dot(Ud, rhoVec))  # truncate rhoVec => rhoVec', shape (trunc, 1)
        rhoVec_p = _np.dot(invABMat_p, rhoVec_p)
        lgstModel.preps[prepLabel] = rhoVec_p

    # Perform "guess" gauge transformation by computing the "B" matrix
    #  assuming rhos, Es, and gates are those of a guesstimate of the model
    if guessModelForGauge is not None:
        guessTrunc = guessModelForGauge.get_dimension()  # the truncation to apply to it's B matrix
        # the dimension of the model for gauge guessing cannot exceed the dimension of the model being estimated
        assert(guessTrunc <= trunc)

        guessPj = _np.zeros((K, guessTrunc), 'd')  # shape = (K, guessTrunc) projector with only trunc columns
        for i in range(guessTrunc): guessPj[i, i] = 1.0
        # guessPjt = _np.transpose(guessPj)         # shape = (guessTrunc, K)

        AMat = _constructA(effectStrs, guessModelForGauge)    # shape = (nESpecs, gsDim)
        # AMat_p = _np.dot( guessPjt, _np.dot(Ud, AMat)) #truncate Evec => Evec', shape (guessTrunc,gsDim) (square!)

        BMat = _constructB(prepStrs, guessModelForGauge)  # shape = (gsDim, nRhoSpecs)
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
            lgstModel.transform(ggEl)
        else:
            ggEl = _objs.FullGaugeGroupElement(_np.linalg.inv(BMat_p))
            lgstModel.transform(ggEl)

        # Force lgstModel to have gates, preps, & effects parameterized in the same way as those in
        # guessModelForGauge, but we only know how to do this when the dimensions of the target and
        # created model match.  If they don't, it doesn't make sense to increase the target model
        # dimension, as this will generally not preserve its parameterization.
        if guessTrunc == trunc:
            for opLabel in opLabelsToEstimate:
                if opLabel in guessModelForGauge.operations:
                    new_op = guessModelForGauge.operations[opLabel].copy()
                    _objs.operation.optimize_operation(new_op, lgstModel.operations[opLabel])
                    lgstModel.operations[opLabel] = new_op

            for prepLabel in rhoLabelsToEstimate:
                if prepLabel in guessModelForGauge.preps:
                    new_vec = guessModelForGauge.preps[prepLabel].copy()
                    _objs.spamvec.optimize_spamvec(new_vec, lgstModel.preps[prepLabel])
                    lgstModel.preps[prepLabel] = new_vec

            for povmLabel in povmLabelsToEstimate:
                if povmLabel in guessModelForGauge.povms:
                    povm = guessModelForGauge.povms[povmLabel]
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

            #Also convey default gauge group & calc class from guessModelForGauge
            lgstModel.default_gauge_group = \
                guessModelForGauge.default_gauge_group
            lgstModel._calcClass = guessModelForGauge._calcClass

        #inv_BMat_p = _np.dot(invABMat_p, AMat_p) # should be equal to inv(BMat_p) when trunc == gsDim ?? check??
        # # lgstModel had dim trunc, so after transform is has dim gsDim
        #lgstModel.transform( S=_np.dot(invABMat_p, AMat_p), Si=BMat_p )

    printer.log("Resulting model:\n", 3)
    printer.log(lgstModel, 3)
    #    for line in str(lgstModel).split('\n'):
    #       printer.log(line, 3)
    return lgstModel


def _lgst_matrix_dims(mdl, prepStrs, effectStrs):
    assert(mdl is not None), "LGST matrix construction requires a non-None Model!"
    nRhoSpecs = len(prepStrs)  # no instruments allowed in prepStrs
    povmLbls = [mdl.split_circuit(s, ('povm',))[2]  # povm_label
                for s in effectStrs]
    povmLens = ([len(mdl.povms[l]) for l in povmLbls])
    nESpecs = sum(povmLens)
    return nRhoSpecs, nESpecs, povmLbls, povmLens


def _constructAB(prepStrs, effectStrs, model, dataset, opLabelAliases=None):
    nRhoSpecs, nESpecs, povmLbls, povmLens = _lgst_matrix_dims(
        model, prepStrs, effectStrs)

    AB = _np.empty((nESpecs, nRhoSpecs))
    eoff = 0
    for i, (estr, povmLen) in enumerate(zip(effectStrs, povmLens)):
        for j, rhostr in enumerate(prepStrs):
            opLabelString = rhostr + estr  # LEXICOGRAPHICAL VS MATRIX ORDER
            dsStr = opLabelString.replace_layers_with_aliases(opLabelAliases)
            raw_dict, outcomes = model.simplify_circuit(opLabelString)
            assert(len(raw_dict) == 1), "No instruments are allowed in LGST fiducials!"
            unique_key = list(raw_dict.keys())[0]
            assert(len(raw_dict[unique_key]) == povmLen)

            dsRow = dataset[dsStr]
            AB[eoff:eoff + povmLen, j] = [dsRow.fraction(ol) for ol in outcomes]
        eoff += povmLen

    return AB


def _constructXMatrix(prepStrs, effectStrs, model, opLabelTuple, dataset, opLabelAliases=None):
    nRhoSpecs, nESpecs, povmLbls, povmLens = _lgst_matrix_dims(
        model, prepStrs, effectStrs)

    nVariants = 1
    for g in opLabelTuple:
        if g in model.instruments:
            nVariants *= len(model.instruments[g])

    X = _np.empty((nVariants, nESpecs, nRhoSpecs))  # multiple "X" matrix variants b/c of instruments

    eoff = 0  # effect-dimension offset
    for i, (estr, povmLen) in enumerate(zip(effectStrs, povmLens)):
        for j, rhostr in enumerate(prepStrs):
            opLabelString = rhostr + _objs.Circuit(opLabelTuple, line_labels=rhostr.line_labels) + estr
            dsStr = opLabelString.replace_layers_with_aliases(opLabelAliases)
            raw_dict, outcomes = model.simplify_circuit(opLabelString)
            dsRow = dataset[dsStr]
            assert(len(raw_dict) == nVariants)

            ooff = 0  # outcome offset
            for k, (raw_str, spamtups) in enumerate(raw_dict.items()):
                assert(len(spamtups) == povmLen)
                X[k, eoff:eoff + povmLen, j] = [
                    dsRow.fraction(ol) for ol in outcomes[ooff:ooff + len(spamtups)]]
                ooff += len(spamtups)
        eoff += povmLen

    return X


def _constructA(effectStrs, mdl):
    _, n, povmLbls, povmLens = _lgst_matrix_dims(
        mdl, [], effectStrs)

    dim = mdl.get_dimension()
    A = _np.empty((n, dim))
    # st = _np.empty(dim, 'd')

    basis_st = _np.zeros((dim, 1), 'd'); eoff = 0
    for k, (estr, povmLbl, povmLen) in enumerate(zip(effectStrs, povmLbls, povmLens)):
        #Build fiducial < E_k | := < EVec[ effectSpec[0] ] | Circuit(effectSpec[1:])
        #st = dot(Ek.T, Estr) = ( dot(Estr.T,Ek)  ).T
        #A[k,:] = st[0,:] # E_k == kth row of A
        for i in range(dim):  # propagate each basis initial state
            basis_st[i] = 1.0
            mdl.preps['rho_LGST_tmp'] = basis_st
            probs = mdl.probs(_objs.Circuit(('rho_LGST_tmp',), line_labels=estr.line_labels) + estr)
            A[eoff:eoff + povmLen, i] = [probs[(ol,)] for ol in mdl.povms[povmLbl]]  # CHECK will this work?
            del mdl.preps['rho_LGST_tmp']
            basis_st[i] = 0.0

        eoff += povmLen
    return A


def _constructB(prepStrs, mdl):
    n = len(prepStrs)
    dim = mdl.get_dimension()
    B = _np.empty((dim, n))
    # st = _np.empty(dim, 'd')

    #Create POVM of vector units
    basis_Es = []
    for i in range(dim):  # propagate each basis initial state
        basis_E = _np.zeros((dim, 1), 'd')
        basis_E[i] = 1.0
        basis_Es.append(basis_E)
    mdl.povms['M_LGST_tmp_povm'] = _objs.UnconstrainedPOVM(
        [("E%d" % i, E) for i, E in enumerate(basis_Es)])

    for k, rhostr in enumerate(prepStrs):
        #Build fiducial | rho_k > := Circuit(prepSpec[0:-1]) | rhoVec[ prepSpec[-1] ] >
        # B[:,k] = st[:,0] # rho_k == kth column of B
        probs = mdl.probs(rhostr + _objs.Circuit(('M_LGST_tmp_povm',), line_labels=rhostr.line_labels))
        B[:, k] = [probs[("E%d" % i,)] for i in range(dim)]  # CHECK will this work?

    del mdl.povms['M_LGST_tmp_povm']
    return B


def _constructTargetAB(prepStrs, effectStrs, targetModel):
    nRhoSpecs, nESpecs, povmLbls, povmLens = _lgst_matrix_dims(
        targetModel, prepStrs, effectStrs)

    AB = _np.empty((nESpecs, nRhoSpecs))
    eoff = 0
    for i, (estr, povmLbl, povmLen) in enumerate(zip(effectStrs, povmLbls, povmLens)):
        for j, rhostr in enumerate(prepStrs):
            opLabelString = rhostr + estr  # LEXICOGRAPHICAL VS MATRIX ORDER
            probs = targetModel.probs(opLabelString)
            AB[eoff:eoff + povmLen, j] = \
                [probs[(ol,)] for ol in targetModel.povms[povmLbl]]
            # outcomes (keys of probs) should just be povm effect labels
            # since no instruments are allowed in fiducial strings.
        eoff += povmLen

    return AB


def gram_rank_and_evals(dataset, prepStrs, effectStrs, targetModel):
    """
    Returns the rank and singular values of the Gram matrix for a dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to populate the Gram matrix

    prepStrs, effectStrs : list
        Lists of preparation and measurement fiducial sequences.

    targetModel : Model
        A model used to make sense of operation sequence elements, and to compute the
        theoretical gram matrix eigenvalues (returned as `svalues_target`).

    Returns
    -------
    rank : int
        the rank of the Gram matrix
    svalues : numpy array
        the singular values of the Gram matrix
    svalues_target : numpy array
        the corresponding singular values of the Gram matrix
        generated by targetModel.
    """
    if targetModel is None: raise ValueError("Must supply `targetModel`")
    ABMat = _constructAB(prepStrs, effectStrs, targetModel, dataset)
    _, s, _ = _np.linalg.svd(ABMat)

    ABMat_tgt = _constructTargetAB(prepStrs, effectStrs, targetModel)
    _, s_tgt, _ = _np.linalg.svd(ABMat_tgt)

    return _np.linalg.matrix_rank(ABMat), s, s_tgt  # _np.linalg.eigvals(ABMat)


###################################################################################
#                 Extended Linear GST (ExLGST)
##################################################################################

#Given dataset D
# Chi2 statistic = sum_k (p_k-f_k)^2/ (N f_kt(1-f_kt) ) where f_kt ~ f_k with +1/+2 to avoid zero denom

def do_exlgst(dataset, startModel, circuitsToUseInEstimation, prepStrs,
              effectStrs, targetModel, guessModelForGauge=None,
              svdTruncateTo=None, maxiter=100000, maxfev=None, tol=1e-6,
              regularizeFactor=0, verbosity=0, comm=None, check_jacobian=False):
    """
    Performs Extended Linear-inversion Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to generate Extended-LGST estimates

    startModel : Model
        The Model used as a starting point for the least-squares
        optimization.

    circuitsToUseInEstimation : list of (tuples or Circuits)
        Each element of this list specifies a operation sequence that is
        estimated using LGST and used in the overall least-squares
        fit that determines the final "extended LGST" model.
        e.g. [ (), ('Gx',), ('Gx','Gy') ]

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    targetModel : Model
        A model used to provide a guess for gauge in which LGST estimates
        should be returned, and the ability to make sense of ("complile")
        operation sequences.

    guessModelForGauge : Model, optional
        A model used to compute a gauge transformation that is applied to
        the LGST estimates before they are returned.
        Defaults to targetModel.

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. 0 causes no truncation, and default is
        `targetModel.dim`.

    maxiter : int, optional
        Maximum number of iterations for the least squares optimization

    maxfev : int, optional
        Maximum number of function evaluations for the least squares optimization
        Defaults to maxiter

    tol : float, optional
        The tolerance for the least squares optimization.

    regularizeFactor : float, optional
        Multiplicative prefactor of L2-like regularization term that penalizes model entries
        which have absolute value greater than 1.  When set to 0, no regularization is applied.

    verbosity : int, optional
        How much detail to send to stdout.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    check_jacobian : bool, optional
        If True, compare the analytic jacobian with a forward finite difference jacobean
        and print warning messages if there is disagreement.  Defaults to False.

    Returns
    -------
    numpy array
        The minimum error vector v = f(x_min), where f(x)**2 is the function being minimized.
    Model
        The model containing all of the estimated labels.
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    if maxfev is None: maxfev = maxiter

    mdl = startModel.copy()
    op_dim = mdl.get_dimension()

    #In order to make sure the vectorized model size matches the
    # "dproduct" size, we *don't* want to parameterize any of the
    # SPAM vectors in the model -- these parameters are not
    # optimized in exLGST so don't paramterize them.  Note that this
    # assumes a "local" Model where parameters are attributed to
    # individual SPAM vecs and gates.
    for prepLabel, rhoVec in mdl.preps.items():
        mdl.preps[prepLabel] = _objs.StaticSPAMVec(rhoVec)
    for povmLabel, povm in mdl.povms.items():
        mdl.povms[povmLabel] = _objs.UnconstrainedPOVM(
            [(lbl, _objs.StaticSPAMVec(E, typ="effect"))
             for lbl, E in povm.items()])

    printer.log("--- eLGST (least squares) ---", 1)

    #TODO REMOVE - now Circuits are tuples + linelabels, so we need the entire object
    #convert list of Circuits to list of raw tuples since that's all we'll need
    #if len(circuitsToUseInEstimation) > 0 and isinstance(circuitsToUseInEstimation[0], _objs.Circuit):
    #    circuitsToUseInEstimation = [opstr.tup for opstr in circuitsToUseInEstimation]

    #Setup and solve a least-squares problem where each element of each
    # (lgst_estimated_process - process_estimate_using_current_model)  difference is a least-squares
    # term and the optimization is over the elements of the "current_model".  Note that:
    #   lgst_estimated_process = LGST estimate for a operation sequence in circuitsToUseInEstimation
    #   process_estimate_using_current_model = process mx you get from multiplying together the operation matrices of
    #                                          the current model

    #Step 1: get the lgst estimates for each of the "operation sequences to use in estimation" list
    evTree, _, _ = mdl.bulk_evaltree(circuitsToUseInEstimation)
    circuitsToUseInEstimation = evTree.generate_circuit_list(permute=False)
    # length of this list == that of raw "simplify" dict == dim of bulk_product, etc.

    opLabelAliases = {}
    for i, opStrTuple in enumerate(circuitsToUseInEstimation):
        opLabelAliases["Gestimator%d" % i] = _objs.Circuit(opStrTuple)  # Note: line labels don't matter in aliases

    lgstEstimates = do_lgst(dataset, prepStrs, effectStrs, targetModel, list(opLabelAliases.keys()),
                            opLabelAliases, guessModelForGauge, svdTruncateTo,
                            verbosity=0)  # override verbosity

    estimates = _np.empty((len(circuitsToUseInEstimation), op_dim, op_dim), 'd')
    for i in range(len(circuitsToUseInEstimation)):
        estimates[i] = lgstEstimates.operations["Gestimator%d" % i]

    maxCircuitLength = max([len(x) for x in circuitsToUseInEstimation])

    #Step 2: create objective function for least squares optimization
    if printer.verbosity < 3:

        if regularizeFactor == 0:
            def _objective_func(vectorGS):
                mdl.from_vector(vectorGS)
                prods = mdl.bulk_product(evTree, comm=comm)
                ret = (prods - estimates).flatten()
                #assert( len( (_np.isnan(ret)).nonzero()[0] ) == 0 )
                return ret
        else:
            def _objective_func(vectorGS):
                mdl.from_vector(vectorGS)
                prods = mdl.bulk_product(evTree, comm=comm)
                gsVecNorm = regularizeFactor * _np.array([max(0, absx - 1.0) for absx in map(abs, vectorGS)], 'd')
                ret = _np.concatenate(((prods - estimates).flatten(), gsVecNorm))
                #assert( len( (_np.isnan(ret)).nonzero()[0] ) == 0 )
                return ret

    else:
        def _objective_func(vectorGS):
            mdl.from_vector(vectorGS)
            prods = mdl.bulk_product(evTree, comm=comm)
            ret = (prods - estimates).flatten()

            #OLD (uncomment to check)
            #errvec = []
            #for (i,opStr) in enumerate(circuitsToUseInEstimation):
            #  term1 = lgstEstimates[ "Gestimator%d" % i ]
            #  term2 = mdl.product(opStr)
            #  if _np.linalg.norm(term2 - prods[i]) > 1e-6:
            #    print "term 2 = \n",term2
            #    print "prod = \n",prods[i]
            #    print "Check failed for product %d: %s : %g" % (i,str(opStr[0:10]),_np.linalg.norm(term2 - prods[i]))
            #  diff = (term2 - term1).flatten()
            #  errvec += list(diff)
            #ret_chk = _np.array(errvec)
            #if _np.linalg.norm( ret - ret_chk ) > 1e-6:
            #  raise ValueError("Check failed with diff = %g" % _np.linalg.norm( ret - ret_chk ))

            if regularizeFactor > 0:
                gsVecNorm = regularizeFactor * _np.array([max(0, absx - 1.0) for absx in map(abs, vectorGS)], 'd')
                ret = _np.concatenate((ret, gsVecNorm))

            retSq = sum(ret * ret)
            printer.log(
                ("%g: objfn vec in (%g,%g),  mdl in (%g,%g), maxLen = %d" %
                 (retSq, _np.min(ret), _np.max(ret), _np.min(vectorGS), _np.max(vectorGS), maxCircuitLength)),
                3)
            #assert( len( (_np.isnan(ret)).nonzero()[0] ) == 0 )
            return ret

    if printer.verbosity < 3:
        if regularizeFactor == 0:
            def _jacobian(vectorGS):
                mdl.from_vector(vectorGS)
                jac = mdl.bulk_dproduct(evTree, flat=True, comm=comm)
                # shape == nCircuits*nFlatOp, nDerivCols
                if check_jacobian: _opt.check_jac(_objective_func, vectorGS, jac, tol=1e-3, eps=1e-6, errType='abs')
                return jac
        else:
            def _jacobian(vectorGS):
                mdl.from_vector(vectorGS)
                gsVecGrad = _np.diag([(regularizeFactor * _np.sign(x) if abs(x) > 1.0 else 0.0) for x in vectorGS])
                jac = mdl.bulk_dproduct(evTree, flat=True, comm=comm)
                # shape == nCircuits*nFlatOp, nDerivCols
                jac = _np.concatenate((jac, gsVecGrad), axis=0)  # shape == nCircuits*nFlatOp+nDerivCols, nDerivCols
                if check_jacobian: _opt.check_jac(_objective_func, vectorGS, jac, tol=1e-3, eps=1e-6, errType='abs')
                return jac

    else:
        def _jacobian(vectorGS):
            mdl.from_vector(vectorGS)
            jac = mdl.bulk_dproduct(evTree, flat=True, comm=comm)
            # shape == nCircuits*nFlatOp, nDerivCols
            if regularizeFactor > 0:
                gsVecGrad = _np.diag([(regularizeFactor * _np.sign(x) if abs(x) > 1.0 else 0.0) for x in vectorGS])
                jac = _np.concatenate((jac, gsVecGrad), axis=0)

            if check_jacobian:
                errSum, errs, fd_jac = _opt.check_jac(_objective_func, vectorGS, jac, tol=1e-3, eps=1e-6, errType='abs')
                printer.log("Jacobian has error %g and %d of %d indices with error > tol" %
                            (errSum, len(errs), jac.shape[0]), 3)
                if len(errs) > 0:
                    i, j = errs[0][0:2]; maxabs = _np.max(_np.abs(jac))                        # pragma: no cover
                    printer.log(" ==> Worst index = %d,%d. Analytic jac = %g, Fwd Diff = %g"  # pragma: no cover
                                % (i, j, jac[i, j], fd_jac[i, j]), 3)                            # pragma: no cover
                    printer.log(" ==> max err = ", errs[0][2], 3)                             # pragma: no cover
                    printer.log(" ==> max err/max = ", max([x[2] / maxabs for x in errs]), 3)  # pragma: no cover

            return jac

    #def checked_jacobian(vectorGS):
    #  def obj_i(x, i): return _objective_func(x)[i]
    #  def jac_i(x, i): return (_jacobian(x))[i]
    #  y = _objective_func(vectorGS)
    #  jac = _jacobian(vectorGS); nJ = _np.linalg.norm(jac)
    #  for i in range(len(y)):
    #    err = _spo.check_grad(obj_i, jac_i, vectorGS, i)
    #    if err/nJ > 1e-6: print "Jacobian(%d) Error = %g (jac norm = %g)" % (i,err,nJ)
    #  return jac

    #Step 3: solve least squares minimization problem
    x0 = mdl.to_vector()
    opt_x, _, _, _, _ = \
        _spo.leastsq(_objective_func, x0, xtol=tol, ftol=tol, gtol=tol,
                     maxfev=maxfev * (len(x0) + 1), full_output=True, Dfun=_jacobian)
    full_minErrVec = _objective_func(opt_x)
    # don't include regularization terms
    minErrVec = full_minErrVec if regularizeFactor == 0 else full_minErrVec[0:-len(x0)]

    #DEBUG: check without using our jacobian
    #opt_x_chk, opt_jac_chk, info_chk, msg_chk, flag_chk = \
    #    _spo.leastsq( _objective_func, x0, xtol=tol, ftol=tol, gtol=tol,
    #             maxfev=maxfev*(len(x0)+1), full_output=True, epsfcn=1e-30)
    #minErrVec_chk = _objective_func(opt_x_chk)

    mdl.from_vector(opt_x)
    #mdl.log("ExLGST", { 'method': "leastsq", 'tol': tol,  'maxiter': maxiter } )

    printer.log(("Sum of minimum least squares error (w/out reg terms) = %g" % sum([x**2 for x in minErrVec])), 2)
    #try: print "   log(likelihood) = ", _tools.logl(mdl, dataset)
    #except: pass
    if targetModel.get_dimension() == mdl.get_dimension():
        printer.log("frobenius distance to target = %s" % mdl.frobeniusdist(targetModel), 2)

        #DEBUG
        #print "  Sum of minimum least squares error check = %g" % sum([x**2 for x in minErrVec_chk])
        #print "DEBUG : opt_x diff = ", _np.linalg.norm( opt_x - opt_x_chk )
        #print "DEBUG : opt_jac diff = ", _np.linalg.norm( opt_jac - opt_jac_chk )
        #print "DEBUG : flags (1,2,3,4=OK) = %d, check = %d" % (flag, flag_chk)

    #TODO: perhaps permute minErrVec using evTree to restore original circuit ordering
    #  but currently minErrVec isn't in such an intuitive format anyway (list of flattened gates)
    #  so maybe just drop minErrVec from return value entirely?
    return minErrVec, mdl


def do_iterative_exlgst(
        dataset, startModel, prepStrs, effectStrs, circuitSetsToUseInEstimation,
        targetModel, guessModelForGauge=None, svdTruncateTo=None, maxiter=100000,
        maxfev=None, tol=1e-6, regularizeFactor=0, returnErrorVec=False,
        returnAll=False, circuitSetLabels=None, verbosity=0, comm=None,
        check_jacobian=False):
    """
    Performs Iterated Extended Linear-inversion Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to generate Extended-LGST estimates

    startModel : Model
        The Model used as a starting point for the least-squares
        optimization.

    prepStrs,effectStrs : list of Circuits
        Fiducial Circuit lists used to construct a informationally complete
        preparation and measurement.

    circuitSetsToUseInEstimation : list of lists of (tuples or Circuits)
        The i-th element is a list of the operation sequences to be used in the i-th iteration
        of extended-LGST.  Each element of these lists is a operation sequence, specifed as
        either a Circuit object or as a tuple of operation labels (but all must be specified
        using the same type).
        e.g. [ [ (), ('Gx',) ], [ (), ('Gx',), ('Gy',) ], [ (), ('Gx',), ('Gy',), ('Gx','Gy') ]  ]

    targetModel : Model
        A model used to provide a guess for gauge in which LGST estimates
        should be returned, and the ability to make sense of ("complile")
        operation sequences.

    guessModelForGauge : Model, optional
        A model used to compute a gauge transformation that is applied to
        the LGST estimates before they are returned.
        Defaults to targetModel.

    svdTruncateTo : int, optional
        The Hilbert space dimension to truncate the operation matrices to using
        a SVD to keep only the largest svdToTruncateTo singular values of
        the I_tildle LGST matrix. 0 causes no truncation, and default is
        `targetModel.dim`.

    maxiter : int, optional
        Maximum number of iterations in each of the chi^2 optimizations

    maxfev : int, optional
        Maximum number of function evaluations for each of the chi^2 optimizations
        Defaults to maxiter

    tol : float, optional
        The tolerance for each of the chi^2 optimizations.

    regularizeFactor : float, optional
        Multiplicative prefactor of L2-like regularization term that penalizes model entries
        which have absolute value greater than 1.  When set to 0, no regularization is applied.

    returnErrorVec : bool, optional
        If True, return (errorVec, model), or (errorVecs, models) if
        returnAll == True, instead of just the model or models.

    returnAll : bool, optional
        If True return a list of models (and errorVecs if returnErrorVec == True),
        one per iteration, instead of the results from just the final iteration.

    circuitSetLabels : list of strings, optional
        An identification label for each of the operation sequence sets (used for displaying
        progress).  Must be the same length as circuitSetsToUseInEstimation.

    verbosity : int, optional
        How much detail to send to stdout.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    check_jacobian : boolean, optional
        If True, compare the analytic jacobian with a forward finite difference jacobean
        and print warning messages if there is disagreement.

    Returns
    -------
    model               if returnAll == False and returnErrorVec == False
    models              if returnAll == True  and returnErrorVec == False
    (errorVec, model)   if returnAll == False and returnErrorVec == True
    (errorVecs, models) if returnAll == True  and returnErrorVec == True
        where errorVec is a numpy array of minimum error values v = f(x_min), where f(x)**2 is
        the function being minimized, model is the Model containing the final estimated gates.
        In cases when returnAll == True, models and errorVecs are lists whose i-th elements are the
        errorVec and model corresponding to the results of the i-th iteration.
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

# Parameter to add later??
#    whenCannotEstimate : string
#        What to do when a operation sequence to be estimated by LGST cannot because there isn't enough data.
#        Allowed values are:
#          'stop'   - stop algorithm and report an error (Default)
#          'warn'   - skip string, print a warning to stdout, and proceed
#          'ignore' - skip string silently and proceed

    #TODO REMOVE - now Circuits are tuples + linelabels, so we need the entire object
    #convert lists of Circuits to lists of raw tuples since that's all we'll need
    #if len(circuitSetsToUseInEstimation) > 0 and \
    #   len(circuitSetsToUseInEstimation[0]) > 0 and \
    #   isinstance(circuitSetsToUseInEstimation[0][0], _objs.Circuit):
    #    circuitLists = [[opstr.tup for opstr in gsList] for gsList in circuitSetsToUseInEstimation]
    #else:
    circuitLists = circuitSetsToUseInEstimation

    #Run extended eLGST iteratively on given sets of estimatable strings
    elgstModels = []; minErrs = []  # for returnAll == True case
    elgstModel = startModel.copy(); nIters = len(circuitLists)

    with printer.progress_logging(1):
        for (i, stringsToEstimate) in enumerate(circuitLists):
            if stringsToEstimate is None or len(stringsToEstimate) == 0: continue

        #printer.log('', 2) #newline if we have more info to print
        extraMessages = ["(%s)" % circuitSetLabels[i]] if circuitSetLabels else []
        printer.show_progress(i, nIters, prefix='--- Iterative eLGST: ',
                              suffix='; %s operation sequences ---' % len(stringsToEstimate),
                              verboseMessages=extraMessages)

        minErr, elgstModel = do_exlgst(
            dataset, elgstModel, stringsToEstimate, prepStrs, effectStrs,
            targetModel, guessModelForGauge, svdTruncateTo, maxiter, maxfev,
            tol, regularizeFactor, printer - 2, comm, check_jacobian)

        if returnAll:
            elgstModels.append(elgstModel)
            minErrs.append(minErr)

    if returnErrorVec:
        return (minErrs, elgstModels) if returnAll else (minErr, elgstModel)
    else:
        return elgstModels if returnAll else elgstModel


###################################################################################
#                 Minimum-Chi2 GST (MC2GST)
##################################################################################

def do_mc2gst(dataset, startModel, circuitsToUse,
              maxiter=100000, maxfev=None, fditer=0, tol=1e-6, extra_lm_opts=None,
              cptp_penalty_factor=0, spam_penalty_factor=0,
              minProbClipForWeighting=1e-4,
              probClipInterval=(-1e6, 1e6), useFreqWeightedChiSq=False,
              regularizeFactor=0, verbosity=0, check=False,
              check_jacobian=False, circuitWeights=None,
              opLabelAliases=None, memLimit=None, comm=None,
              distributeMethod="deriv", profiler=None,
              evaltree_cache=None, time_dependent=False):
    """
    Performs Least-Squares Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The dataset to obtain counts from.

    startModel : Model
        The Model used as a starting point for the least-squares
        optimization.

    circuitsToUse : list of (tuples or Circuits)
        Each tuple contains operation labels and specifies a operation sequence whose
        probabilities are considered when trying to least-squares-fit the
        probabilities given in the dataset.
        e.g. [ (), ('Gx',), ('Gx','Gy') ]

    maxiter : int, optional
        Maximum number of iterations for the chi^2 optimization.

    maxfev : int, optional
        Maximum number of function evaluations for the chi^2 optimization.
        Defaults to maxiter.

    fditer : int, optional
        The number of iterations to perform with a finite-difference-Jacobian,
        as opposed to the more precise "analytic" Jacobian.  Useful if the
        optimization's starting point is special/singular.

    tol : float or dict, optional
        The tolerance for the chi^2 optimization.  If a dict, allowed keys are
        `'relx'`, `'relf'`, `'f'`, `'jac'`, and `'maxdx'`.  If a float, then
        `{'relx': 1e-8, 'relf': tol, 'f': 1.0, 'jac': tol, 'maxdx': 1.0 }` is used.

    extra_lm_opts : dict or None, optional
        Additional options for the Levenberg-Marquardt algorithm.

    cptp_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains CPTP penalty
        terms which penalize non-CPTP-ness of the gates being optimized.  This factor
        multiplies these CPTP penalty terms.

    spam_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains SPAM penalty
        terms which penalize non-positive-ness of the state preps being optimized.  This
        factor multiplies these SPAM penalty terms.

    minProbClipForWeighting : float, optional
        Sets the minimum and maximum probability p allowed in the chi^2 weights: N/(p*(1-p))
        by clipping probability p values to lie within the interval
        [ minProbClipForWeighting, 1-minProbClipForWeighting ].

    probClipInterval : 2-tuple or None, optional
       (min,max) values used to clip the probabilities predicted by models during MC2GST's
       least squares search for an optimal model (if not None).  if None, no clipping is performed.

    useFreqWeightedChiSq : bool, optional
        If True, objective function uses only an approximate chi^2 weighting:  N/(f*(1-f))
        where f is the frequency obtained from the dataset, instead of the true chi^2: N/(p*(1-p))
        where p is a predicted probability.  Defaults to False, and only should use
        True for backward compatibility.

    regularizeFactor : float, optional
        Multiplicative prefactor of L2-like regularization term that penalizes model entries
        which have absolute value greater than 1.  When set to 0, no regularization is applied.

    verbosity : int, optional
        How much detail to send to stdout.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    check_jacobian : boolean, optional
        If True, compare the analytic jacobian with a forward finite difference jacobean
        and print warning messages if there is disagreement.  Defaults to False.

    circuitWeights : numpy array, optional
        An array of length len(circuitsToUse).  Each element scales the
        least-squares term of the corresponding operation sequence in circuitsToUse.
        The default is no weight scaling at all.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are circuits
        corresponding to what that operation label should be expanded into before querying
        the dataset.  Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = pygsti.obj.Circuit(['Gx','Gx','Gx'])

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    distributeMethod : {"circuits", "deriv"}
        How to distribute calculation amongst processors (only has effect
        when comm is not None).  "circuits" will divide the list of
        circuits; "deriv" will divide the columns of the jacobian matrix.

    profiler : Profiler, optional
        A profiler object used for to track timing and memory usage.

    evaltree_cache : dict, optional
        A dictionary which server as a cache for the computed EvalTree used
        in this computation.  If an empty dictionary is supplied, it is filled
        with cached values to speed up subsequent executions of this function
        which use the *same* `startModel`, `circuitsToUse`, `memLimit`,
        `comm`, and `distributeMethod`.

    time_dependent : bool, optional
        Whether any timestamps in the data should be taken seriously and used
        to compare with a potentially time-dependent model.


    Returns
    -------
    errorVec : numpy array
        Minimum error values v = f(x_best), where f(x)**2 is the function being minimized
    model : Model
        Model containing the estimated gates.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if profiler is None: profiler = _dummy_profiler
    tStart = _time.time()
    mdl = startModel  # .copy()  # to allow caches in startModel to be retained
    if maxfev is None: maxfev = maxiter

    #printer.log('', 2)
    printer.log("--- Minimum Chi^2 GST ---", 1)

    if comm is not None:
        #assume all models at least have same parameters - so just compare vecs
        v_cmp = comm.bcast(mdl.to_vector() if (comm.Get_rank() == 0) else None, root=0)
        if _np.linalg.norm(mdl.to_vector() - v_cmp) > 1e-6:
            raise ValueError("MPI ERROR: *different* MC2GST start models"
                             " given to different processors!")                   # pragma: no cover

    #TODO REMOVE - now Circuits are tuples + linelabels, so we need the entire object
    #convert list of Circuits to list of raw tuples since that's all we'll need
    #if len(circuitsToUse) > 0 and \
    #        isinstance(circuitsToUse[0], _objs.Circuit):
    #    circuitsToUse = [opstr.tup for opstr in circuitsToUse]

    #Memory allocation
    ns = int(round(_np.sqrt(mdl.dim)))  # estimate avg number of spamtuples per string
    ng = len(circuitsToUse)
    ne = mdl.num_params()
    C = 1.0 / 1024.0**3

    #  Estimate & check persistent memory (from allocs directly below)
    persistentMem = 8 * (ng * (ns + ns * ne + 1 + 3 * ns))  # final results in bytes
    if memLimit is not None and memLimit < persistentMem:
        raise MemoryError("Memory limit ({} GB) is < memory required to hold final results "
                          "({} GB)".format(memLimit * C, persistentMem * C))

    #Create evaluation tree (split into subtrees if needed)
    tm = _time.time()
    if (memLimit is not None):
        curMem = _objs.profiler._get_max_mem_usage(comm)
        gthrMem = int(0.1 * (memLimit - persistentMem))
        mlim = memLimit - persistentMem - gthrMem - curMem
        printer.log("Memory limit = %.2fGB" % (memLimit * C))
        printer.log("Cur, Persist, Gather = %.2f, %.2f, %.2f GB" %
                    (curMem * C, persistentMem * C, gthrMem * C))
        assert mlim > 0, 'Not enough memory, exiting..'
    else: gthrMem = mlim = None

    if evaltree_cache and 'evTree' in evaltree_cache \
       and 'wrtBlkSize' in evaltree_cache:
        evTree = evaltree_cache['evTree']
        wrtBlkSize = evaltree_cache['wrtBlkSize']
        lookup = evaltree_cache['lookup']
        outcomes_lookup = evaltree_cache['outcomes_lookup']
    else:
        # Note: simplify_circuits doesn't support aliased dataset (yet)
        dstree = dataset if (opLabelAliases is None) else None
        evTree, wrtBlkSize, _, lookup, outcomes_lookup = mdl.bulk_evaltree_from_resources(
            circuitsToUse, comm, mlim, distributeMethod,
            ["bulk_fill_probs", "bulk_fill_dprobs"], dstree, printer - 1)

        #Fill cache dict if one was given
        if evaltree_cache is not None:
            evaltree_cache['evTree'] = evTree
            evaltree_cache['wrtBlkSize'] = wrtBlkSize
            evaltree_cache['lookup'] = lookup
            evaltree_cache['outcomes_lookup'] = outcomes_lookup

    profiler.add_time("do_mc2gst: pre-opt treegen", tStart)

    #Expand operation label aliases used in DataSet lookups
    dsCircuitsToUse = _tools.apply_aliases_to_circuit_list(circuitsToUse, opLabelAliases)

    #  Allocate peristent memory
    #  (must be AFTER possible operation sequence permutation by
    #   tree and initialization of dsCircuitsToUse)

    if evaltree_cache and 'cntVecMx' in evaltree_cache \
       and not useFreqWeightedChiSq:  # b/c we don't cache fweights
        cntVecMx = evaltree_cache['cntVecMx']
        N = evaltree_cache['totalCntVec']
        fweights = None
    else:
        KM = evTree.num_final_elements()  # shorthand for combined spam+circuit dimension
        N = _np.empty(KM, 'd')  # totalCntVec
        cntVecMx = _np.empty(KM, 'd')
        fweights = _np.empty(KM, 'd') if useFreqWeightedChiSq else None  # usually not used

        #NOTE on chi^2 expressions:
        #in general case:   chi^2 = sum (p_i-f_i)^2/p_i  (for i summed over outcomes)
        #in 2-outcome case: chi^2 = (p+ - f+)^2/p+ + (p- - f-)^2/p-
        #                         = (p - f)^2/p + (1-p - (1-f))^2/(1-p)
        #                         = (p - f)^2 * (1/p + 1/(1-p))
        #                         = (p - f)^2 * ( ((1-p) + p)/(p*(1-p)) )
        #                         = 1/(p*(1-p)) * (p - f)^2

        for (i, opStr) in enumerate(dsCircuitsToUse):
            cnts = dataset[opStr].counts
            N[lookup[i]] = sum(cnts.values())  # dataset[opStr].total
            cntVecMx[lookup[i]] = [cnts.get(x, 0) for x in outcomes_lookup[i]]
            if useFreqWeightedChiSq:
                wts = []
                for x in outcomes_lookup[i]:
                    Nx = dataset[opStr].total
                    f1 = dataset[opStr].fraction(x); f2 = (f1 + 1) / (Nx + 2)
                    wts.append(_np.sqrt(Nx / (f2 * (1 - f2))))
                fweights[lookup[i]] = wts

        if circuitWeights is not None:
            for i in range(len(circuitsToUse)):
                cntVecMx[lookup[i]] *= circuitWeights[i]  # dim KM (K = nSpamLabels, M = nCircuits )
                N[lookup[i]] *= circuitWeights[i]  # multiply N's by weights

        if evaltree_cache is not None:
            evaltree_cache['cntVecMx'] = cntVecMx
            evaltree_cache['totalCntVec'] = N

    if useFreqWeightedChiSq:
        assert(not time_dependent), "Cannot use frequency-weighted chi2 with `time_dependent` == True!"
        objective = _objfns.FreqWeightedChi2Function(
            mdl, evTree, lookup, circuitsToUse, opLabelAliases, regularizeFactor, cptp_penalty_factor,
            spam_penalty_factor, cntVecMx, N, fweights, minProbClipForWeighting,
            probClipInterval, wrtBlkSize, gthrMem, check, check_jacobian, comm, profiler, printer)
    else:
        if time_dependent:
            objective = _objfns.TimeDependentChi2Function(
                mdl, evTree, lookup, circuitsToUse, opLabelAliases, regularizeFactor, cptp_penalty_factor,
                spam_penalty_factor, dataset, dsCircuitsToUse, minProbClipForWeighting,
                probClipInterval, wrtBlkSize, gthrMem, check, check_jacobian, comm, profiler, printer)
        else:
            objective = _objfns.Chi2Function(
                mdl, evTree, lookup, circuitsToUse, opLabelAliases, regularizeFactor, cptp_penalty_factor,
                spam_penalty_factor, cntVecMx, N, minProbClipForWeighting, probClipInterval,
                wrtBlkSize, gthrMem, check, check_jacobian, comm, profiler, printer)

    #Get number of maximal-model parameter ("dataset params") if needed for print messages
    tm = _time.time()
    if printer.verbosity > 0:
        # number of independent parameters in dataset (max. model # of params)
        nDataParams = dataset.get_degrees_of_freedom(
            dsCircuitsToUse, aggregate_times=not time_dependent)
    else:
        nDataParams = 0  # because it's never used
    profiler.add_time("do_mc2gst: num data params", tm)

    #Step 3: solve least squares minimization problem

    if mdl.simtype in ("termgap", "termorder"):
        minErrVec = _do_term_runopt(evTree, mdl, objective, "chi2", maxiter, maxfev, tol, fditer,
                                    extra_lm_opts, comm, printer, profiler, nDataParams, memLimit)
    else:
        #Normal case of just a single "sub-iteration"
        minErrVec, _ = _do_runopt(mdl, objective, "chi2", maxiter, maxfev, tol, fditer, extra_lm_opts, comm,
                                  printer, profiler, nDataParams, memLimit)

    printer.log("Completed in %.1fs" % (_time.time() - tStart), 1)

    #if targetModel is not None:
    #  target_vec = targetModel.to_vector()
    #  targetErrVec = _objective_func(target_vec)
    #  return minErrVec, soln_gs, targetErrVec
    profiler.add_time("do_mc2gst: total time", tStart)
    #TODO: evTree.permute_computation_to_original(minErrVec) #Doesn't work b/c minErrVec is flattened
    # but maybe best to just remove minErrVec from return value since this isn't very useful
    # anyway?
    return minErrVec, mdl


def _do_runopt(mdl, objective, objective_name, maxiter, maxfev, tol, fditer, extra_lm_opts, comm,
               printer, profiler, nDataParams, memLimit, logL_upperbound=None):

    tm = _time.time()

    if extra_lm_opts is None: extra_lm_opts = {}
    objective_func = objective.fn
    jacobian = objective.jfn

    x0 = mdl.to_vector()
    if isinstance(tol, float): tol = {'relx': 1e-8, 'relf': tol, 'f': 1.0, 'jac': tol, 'maxdx': 1.0}
    if CUSTOMLM:
        opt_x, converged, msg, mu, nu = _opt.custom_leastsq(
            objective_func, jacobian, x0, f_norm2_tol=tol['f'],
            jac_norm_tol=tol['jac'], rel_ftol=tol['relf'], rel_xtol=tol['relx'],
            max_iter=maxiter, num_fd_iters=fditer, max_dx_scale=tol['maxdx'],
            comm=comm, verbosity=printer - 1, profiler=profiler, **extra_lm_opts)
        printer.log("Least squares message = %s" % msg, 2)
        assert(converged), "Failed to converge: %s" % msg
        opt_state = (msg, mu, nu)
    else:
        opt_x, _, _, msg, flag = \
            _spo.leastsq(objective_func, x0, xtol=tol['relx'], ftol=tol['relf'], gtol=tol['jac'],
                         maxfev=maxfev * (len(x0) + 1), full_output=True, Dfun=jacobian)  # pragma: no cover
        printer.log("Least squares message = %s; flag =%s" % (msg, flag), 2)            # pragma: no cover
        opt_state = (msg,)

    full_minErrVec = objective_func(opt_x)  # note: calls mdl.from_vector(opt_x,...) so don't need to call this again
    # don't include "extra" regularization terms
    minErrVec = full_minErrVec[0:-objective.ex] if (objective.ex > 0) else full_minErrVec
    sum_minErrVec = sum(minErrVec**2)  # total chi2 or (upperBoundLogL - logl), depending on objective_name

    profiler.add_time("do_mc2gst: leastsq", tm)

    tm = _time.time()

    if printer.verbosity > 0:
        #Don't compute num gauge params if it's expensive (>10% of mem limit) or unavailable
        if hasattr(mdl, 'num_elements'):
            memForNumGaugeParams = mdl.num_elements() * (mdl.num_params() + mdl.dim**2) \
                * FLOATSIZE  # see Model._buildup_dPG (this is mem for dPG)

            if memLimit is None or 0.1 * memLimit < memForNumGaugeParams:
                try:
                    nModelParams = mdl.num_nongauge_params()  # len(x0)
                except:  # numpy can throw a LinAlgError or sparse cases can throw a NotImplementedError
                    printer.warning("Could not obtain number of *non-gauge* parameters - using total params instead")
                    nModelParams = mdl.num_params()
            else:
                printer.log("Finding num_nongauge_params is too expensive: using total params.")
                nModelParams = mdl.num_params()  # just use total number of params
        else:
            nModelParams = mdl.num_params()  # just use total number of params

        if objective_name == "chi2":  # could use isinstance(objective, Chi2Function) here?
            totChi2 = sum_minErrVec
            # reject GST model if p-value < threshold (~0.05?)
            pvalue = 1.0 - _stats.chi2.cdf(totChi2, nDataParams - nModelParams)
            printer.log("Sum of Chi^2 = %g (%d data params - %d model params = expected mean of %g; p-value = %g)" %
                        (totChi2, nDataParams, nModelParams, nDataParams - nModelParams, pvalue), 1)
        else:
            assert(objective_name == "logl")  # assume only options are "chi2" or "logl"
            # reject GST if p-value < threshold (~0.05?)
            deltaLogL = sum_minErrVec

            if _np.isfinite(deltaLogL):
                pvalue = 1.0 - _stats.chi2.cdf(2 * deltaLogL, nDataParams - nModelParams)
                printer.log("  Maximum log(L) = %g below upper bound of %g" % (deltaLogL, logL_upperbound), 1)
                printer.log("    2*Delta(log(L)) = %g (%d data params - %d model params = expected mean of %g; "
                            "p-value = %g)" %
                            (2 * deltaLogL, nDataParams, nModelParams, nDataParams - nModelParams, pvalue), 1)
            else:
                printer.log("  **Warning** upper_bound_logL - logl = " + str(deltaLogL), 1)  # pragma: no cover

    if objective_name == "logl":
        deltaLogL = sum_minErrVec
        return (logL_upperbound - deltaLogL), opt_state
    else:
        return minErrVec, opt_state


def _do_term_runopt(evTree, mdl, objective, objective_name, maxiter, maxfev, tol, fditer,
                    extra_lm_opts, comm, printer, profiler, nDataParams, memLimit, logL_upperbound=None):
    """ TODO: docstring """

    fwdsim = mdl._fwdsim()

    #Pipe these parameters in from fwdsim, even though they're used to control the term-stage loop
    maxTermStages = fwdsim.max_term_stages
    pathFractionThreshold = fwdsim.path_fraction_threshold  # 0 when not using path-sets
    oob_check_interval = fwdsim.oob_check_interval
    if extra_lm_opts is None: extra_lm_opts = {}

    #assume a path set has already been chosen, as one should have been chosen
    # when evTree was created.
    pathSet = fwdsim.get_current_pathset(evTree, comm)
    if pathSet:  # only some types of term "modes" (see fwdsim.mode) use path-sets
        pathFraction = pathSet.get_allowed_path_fraction()
        printer.log("Initial Term-stage model has %d failures and uses %.1f%% of allowed paths." %
                    (pathSet.num_failures, 100 * pathFraction))
    else:
        pathFraction = 1.0  # b/c "all" paths are used, and > pathFractionThreshold, which should be 0

    minErrVec = None
    for sub_iter in range(maxTermStages):

        bFinalIter = (sub_iter == maxTermStages - 1) or (pathFraction > pathFractionThreshold)
        extra_lm_opts['oob_check_interval'] = oob_check_interval
        # don't stop early on last iter - do as much as possible.
        extra_lm_opts['oob_action'] = "reject" if bFinalIter else "stop"
        minErrVec, opt_state = _do_runopt(mdl, objective, objective_name, maxiter, maxfev, tol, fditer, extra_lm_opts,
                                          comm, printer, profiler, nDataParams, memLimit, logL_upperbound)

        if not opt_state[0] == "Objective function out-of-bounds! STOP":
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
            pathSet = mdl._fwdsim().find_minimal_paths_set(evTree, comm, memLimit)
            mdl._fwdsim().select_paths_set(pathSet, comm, memLimit)
            pathFraction = pathSet.get_allowed_path_fraction()
            extra_lm_opts['init_munu'] = (opt_state[1], opt_state[2])
            printer.log("After adapting paths, num failures = %d, %.1f%% of allowed paths used." %
                        (pathSet.num_failures, 100 * pathFraction))

    return minErrVec


def do_mc2gst_with_model_selection(
        dataset, startModel, dimDelta, circuitsToUse,
        maxiter=100000, maxfev=None, tol=1e-6,
        cptp_penalty_factor=0, spam_penalty_factor=0,
        minProbClipForWeighting=1e-4, probClipInterval=(-1e6, 1e6),
        useFreqWeightedChiSq=False, regularizeFactor=0, verbosity=0,
        check=False, check_jacobian=False, circuitWeights=None,
        opLabelAliases=None, memLimit=None, comm=None):
    """
    Performs Least-Squares Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The dataset to obtain counts from.

    startModel : Model
        The Model used as a starting point for the least-squares
        optimization.

    dimDelta : integer
        Amount by which to increment or decrement the dimension of
        current model (initially startModel) to obtain candidate
        alternative models for performing model selection.

    circuitsToUse : list of (tuples or Circuits)
        Each tuple contains operation labels and specifies a operation sequence whose
        probabilities are considered when trying to least-squares-fit the
        probabilities given in the dataset.
        e.g. [ (), ('Gx',), ('Gx','Gy') ]

    maxiter : int, optional
        Maximum number of iterations for the chi^2 optimization.

    maxfev : int, optional
        Maximum number of function evaluations for the chi^2 optimization.
        Defaults to maxiter.

    tol : float, optional
        The tolerance for the chi^2 optimization.  If a dict, allowed keys are
        `'relx'`, `'relf'`, `'f'`, and `'jac'`.  If a float, then
        `{'relx': 1e-8, 'relf': tol, 'f': 1.0, 'jac': tol }` is used.

    cptp_penalty_factor : float, optional
        If greater than zero, the optimization also contains CPTP penalty
        terms which penalize non-CPTP-ness of the gates being optimized.  This factor
        multiplies these CPTP penalty terms.

    spam_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains SPAM penalty
        terms which penalize non-positive-ness of the state preps being optimized.  This
        factor multiplies these SPAM penalty terms.

    minProbClipForWeighting : float, optional
        Sets the minimum and maximum probability p allowed in the chi^2 weights: N/(p*(1-p))
        by clipping probability p values to lie within the interval
        [ minProbClipForWeighting, 1-minProbClipForWeighting ].

    probClipInterval : 2-tuple or None, optional
       (min,max) values used to clip the probabilities predicted by models during MC2GST's
       least squares search for an optimal model (if not None).  if None, no clipping is performed.

    useFreqWeightedChiSq : bool, optional
        If True, objective function uses only an approximate chi^2 weighting:  N/(f*(1-f))
        where f is the frequency obtained from the dataset, instead of the true chi^2: N/(p*(1-p))
        where p is a predicted probability.  Defaults to False, and only should use
        True for backward compatibility.

    regularizeFactor : float, optional
        Multiplicative prefactor of L2-like regularization term that penalizes model entries
        which have absolute value greater than 1.  When set to 0, no regularization is applied.

    verbosity : int, optional
        How much detail to send to stdout.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.  Defaults to False.

    check_jacobian : boolean, optional
        If True, compare the analytic jacobian with a forward finite difference jacobean
        and print warning messages if there is disagreement.  Defaults to False.

    circuitWeights : numpy array, optional
        An array of length len(circuitsToUse).  Each element scales the
        least-squares term of the corresponding operation sequence in circuitsToUse.
        The default is no weight scaling at all.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are circuits
        corresponding to what that operation label should be expanded into before querying
        the dataset.  Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = pygsti.obj.Circuit(['Gx','Gx','Gx'])

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.


    Returns
    -------
    errorVec : numpy array
        Minimum error values v = f(x_best), where f(x)**2 is the function
        being minimized.
    model : Model
        Model containing the estimated gates.
    """
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    dim = startModel.get_dimension()
    nStrings = len(circuitsToUse)

    #Run do_mc2gst multiple times - one for the starting Model and one for the starting model
    # with increased or decreased dimension as per dimDelta
    #printer.log('', 2)
    printer.log("--- Minimum Chi^2 GST with model selection (starting dim = %d) ---" % dim, 1)

    #TODO REMOVE - now Circuits are tuples + linelabels, so we need the entire object
    #convert list of Circuits to list of raw tuples since that's all we'll need
    #if len(circuitsToUse) > 0 and isinstance(circuitsToUse[0], _objs.Circuit):
    #    circuitsToUse = [opstr.tup for opstr in circuitsToUse]
    extra_lm_opts = {}  # FUTURE: make these accessible to caller?
    minErr, mdl = do_mc2gst(dataset, startModel, circuitsToUse, maxiter,
                            maxfev, 0, tol, extra_lm_opts, cptp_penalty_factor, spam_penalty_factor,
                            minProbClipForWeighting, probClipInterval,
                            useFreqWeightedChiSq, regularizeFactor, printer - 1,
                            check, check_jacobian, circuitWeights, opLabelAliases,
                            memLimit, comm)
    chiSqBest = sum([x**2 for x in minErr])  # using only circuitsToUse
    nParamsBest = len(startModel.to_vector())
    origMDL = bestGS = mdl
    bestMinErr = minErr

    printer.log("Dim %d: chi^2 = %g, nCircuits=%d, nParams=%d (so expected mean = %d)" %
                (dim, chiSqBest, nStrings, nParamsBest, nStrings - nParamsBest))

    #Notes on Model selection test:
    # compare chi2 - 2*(nStrings-nParams) for each model -- select lower one
    # So compare:  chi2_A - 2*nStrings + 2*nParams_A <> chi2_B - 2*nStrings + 2*nParamsB
    #              chi2_A + 2*nParams_A <> chi2_B + 2*nParams_B
    #              chi2_A - chi2_B <> 2*(nParams_B - nParams_A)

    #try decreasing the dimension
    curDim = dim
    tryDecreasedDim = True
    curStartModel = origMDL
    while tryDecreasedDim:
        curDim -= dimDelta
        curStartModel = curStartModel.decrease_dimension(curDim)
        nParams = curStartModel.num_params()

        minErr, mdl = do_mc2gst(dataset, curStartModel, circuitsToUse, maxiter,
                                maxfev, 0, tol, extra_lm_opts, cptp_penalty_factor, spam_penalty_factor,
                                minProbClipForWeighting, probClipInterval,
                                useFreqWeightedChiSq, regularizeFactor, printer - 1,
                                check, check_jacobian, circuitWeights, opLabelAliases,
                                memLimit, comm)

        chiSq = sum([x**2 for x in minErr])  # using only circuitsToUse

        #Model selection test
        chi2diff = chiSq - chiSqBest
        paramDiff = nParams - nParamsBest
        if (chiSqBest - chiSq) > 2 * (nParams - nParamsBest):  # equivaletly: -chi2diff > 2*paramDiff
            bestGS, bestMinErr, chiSqBest, nParamsBest = mdl, minErr, chiSq, nParams
            msResult = "Selected"
        else:
            msResult = "Rejected"
            tryDecreasedDim = False

        printer.log("%s dim %d: chi^2 = %g (%+g w.r.t. expected mean of %d strings - %d params = %d) "
                    "(dChi^2=%d, 2*dParams=%d)" %
                    (msResult, curDim, chiSq, chiSq - (nStrings - nParams), nStrings, nParams, nStrings - nParams,
                     chi2diff, 2 * paramDiff))

    #try increasing the dimension
    curDim = dim
    tryIncreasedDim = bool(curDim == dim)  # if we didn't decrease the dimension
    curStartModel = origMDL

    while tryIncreasedDim:
        curDim += dimDelta
        curStartModel = curStartModel.increase_dimension(curDim)
        curStartModel = curStartModel.kick(0.01)  # give random kick here??
        nParams = curStartModel.num_params()
        if nParams > nStrings:
            #Future: do "MC2GST" for underconstrained nonlinear problems -- or just double up?
            tryIncreasedDim = False
            continue

        minErr, mdl = do_mc2gst(dataset, curStartModel, circuitsToUse, maxiter,
                                maxfev, 0, tol, extra_lm_opts, cptp_penalty_factor, spam_penalty_factor,
                                minProbClipForWeighting, probClipInterval,
                                useFreqWeightedChiSq, regularizeFactor, printer - 1,
                                check, check_jacobian, circuitWeights, opLabelAliases,
                                memLimit, comm)

        chiSq = sum([x**2 for x in minErr])  # using only circuitsToUse

        #Model selection test
        chi2diff = chiSq - chiSqBest
        paramDiff = nParams - nParamsBest
        if (chiSqBest - chiSq) > 2 * (nParams - nParamsBest):  # equivaletly: -chi2diff > 2*paramDiff
            bestGS, bestMinErr, chiSqBest, nParamsBest = mdl, minErr, chiSq, nParams
            msResult = "Selected"
        else:
            msResult = "Rejected"
            tryIncreasedDim = False

        printer.log("%s dim %d: chi^2 = %g (%+g w.r.t. expected mean of %d strings - %d params = %d) "
                    "(dChi^2=%d, 2*dParams=%d)" %
                    (msResult, curDim, chiSq, chiSq - (nStrings - nParams), nStrings, nParams, nStrings - nParams,
                     chi2diff, 2 * paramDiff))

    return bestMinErr, bestGS


def do_iterative_mc2gst(dataset, startModel, circuitSetsToUseInEstimation,
                        maxiter=100000, maxfev=None, fditer=0, tol=1e-6, extra_lm_opts=None,
                        cptp_penalty_factor=0, spam_penalty_factor=0,
                        minProbClipForWeighting=1e-4,
                        probClipInterval=(-1e6, 1e6), useFreqWeightedChiSq=False,
                        regularizeFactor=0, returnErrorVec=False,
                        returnAll=False, circuitSetLabels=None, verbosity=0,
                        check=False, check_jacobian=False,
                        circuitWeightsDict=None, opLabelAliases=None,
                        memLimit=None, profiler=None, comm=None,
                        distributeMethod="deriv", evaltree_cache=None, time_dependent=False):
    """
    Performs Iterative Minimum Chi^2 Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to generate MC2GST gate estimates

    startModel : Model
        The Model used as a starting point for the least-squares
        optimization.

    circuitSetsToUseInEstimation : list of lists of (tuples or Circuits)
        The i-th element is a list of the operation sequences to be used in the i-th iteration
        of MC2GST.  Each element of these lists is a operation sequence, specifed as
        either a Circuit object or as a tuple of operation labels (but all must be specified
        using the same type).
        e.g. [ [ (), ('Gx',) ], [ (), ('Gx',), ('Gy',) ], [ (), ('Gx',), ('Gy',), ('Gx','Gy') ]  ]

    maxiter : int, optional
        Maximum number of iterations for the chi^2 optimization.

    maxfev : int, optional
        Maximum number of function evaluations for the chi^2 optimization.

    fditer : int, optional
        The number of initial (step 0) iterations to perform with a
        finite-difference-Jacobian, as opposed to the more precise "analytic"
        Jacobian.  Useful if the optimization's starting point is special/singular.

    tol : float, optional
        The tolerance for the chi^2 optimization.  If a dict, allowed keys are
        `'relx'`, `'relf'`, `'f'`, and `'jac'`.  If a float, then
        `{'relx': 1e-8, 'relf': tol, 'f': 1.0, 'jac': tol }` is used.

    extra_lm_opts : dict or None, optional
        Additional options for the Levenberg-Marquardt algorithm.

    cptp_penalty_factor : float, optional
        If greater than zero, the optimization also contains CPTP penalty
        terms which penalize non-CPTP-ness of the gates being optimized.  This factor
        multiplies these CPTP penalty terms.

    spam_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains SPAM penalty
        terms which penalize non-positive-ness of the state preps being optimized.  This
        factor multiplies these SPAM penalty terms.

    minProbClipForWeighting : float, optional
        Sets the minimum and maximum probability p allowed in the chi^2 weights: N/(p*(1-p))
        by clipping probability p values to lie within the interval
        [ minProbClipForWeighting, 1-minProbClipForWeighting ].

    probClipInterval : 2-tuple or None, optional
       (min,max) values used to clip the probabilities predicted by models during MC2GST's
       least squares search for an optimal model (if not None).  if None, no clipping is performed.

    useFreqWeightedChiSq : bool, optional
        If True, objective function uses only an approximate chi^2 weighting:  N/(f*(1-f))
        where f is the frequency obtained from the dataset, instead of the true chi^2: N/(p*(1-p))
        where p is a predicted probability.  Defaults to False, and only should use
        True for backward compatibility.

    regularizeFactor : float, optional
        Multiplicative prefactor of L2-like regularization term that penalizes model entries
        which have absolute value greater than 1.  When set to 0, no regularization is applied.

    returnErrorVec : bool, optional
        If True, return (errorVec, model), or (errorVecs, models) if
        returnAll == True, instead of just the model or models.

    returnAll : bool, optional
        If True return a list of models (and errorVecs if returnErrorVec == True),
        one per iteration, instead of the results from just the final iteration.

    circuitSetLabels : list of strings, optional
        An identification label for each of the operation sequence sets (used for displaying
        progress).  Must be the same length as circuitSetsToUseInEstimation.

    verbosity : int, optional
        How much detail to send to stdout.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    check_jacobian : boolean, optional
        If True, compare the analytic jacobian with a forward finite difference jacobean
        and print warning messages if there is disagreement.

    circuitWeightsDict : dictionary, optional
        A dictionary with keys == operation sequences and values == multiplicative scaling
        factor for the corresponding operation sequence. The default is no weight scaling at all.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are circuits
        corresponding to what that operation label should be expanded into before querying
        the dataset.  Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = pygsti.obj.Circuit(['Gx','Gx','Gx'])

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    profiler : Profiler, optional
         A profiler object used for to track timing and memory usage.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    distributeMethod : {"circuits", "deriv"}
        How to distribute calculation amongst processors (only has effect
        when comm is not None).  "circuits" will divide the list of
        circuits; "deriv" will divide the columns of the jacobian matrix.

    evaltree_cache : dict, optional
        An empty dictionary which gets filled with the *final* computed EvalTree
        (and supporting info) used in this computation.

    time_dependent : bool, optional
        Whether any timestamps in the data should be taken seriously and used
        to compare with a potentially time-dependent model.


    Returns
    -------
    model               if returnAll == False and returnErrorVec == False
    models              if returnAll == True  and returnErrorVec == False
    (errorVec, model)   if returnAll == False and returnErrorVec == True
    (errorVecs, models) if returnAll == True  and returnErrorVec == True
        where errorVec is a numpy array of minimum error values v = f(x_min), where f(x)**2 is
        the function being minimized, model is the Model containing the final estimated gates.
        In cases when returnAll == True, models and errorVecs are lists whose i-th elements are the
        errorVec and model corresponding to the results of the i-th iteration.
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if profiler is None: profiler = _dummy_profiler

    #TODO REMOVE - now Circuits are tuples + linelabels, so we need the entire object
    #convert lists of Circuits to lists of raw tuples since that's all we'll need
    #if len(circuitSetsToUseInEstimation) > 0 and \
    #   len(circuitSetsToUseInEstimation[0]) > 0 and \
    #   isinstance(circuitSetsToUseInEstimation[0][0], _objs.Circuit):
    #    circuitLists = [[opstr.tup for opstr in gsList] for gsList in circuitSetsToUseInEstimation]
    #else:
    circuitLists = circuitSetsToUseInEstimation

    #Run MC2GST iteratively on given sets of estimatable strings
    lsgstModels = []; minErrs = []  # for returnAll == True case
    lsgstModel = startModel.copy(); nIters = len(circuitLists)
    tStart = _time.time()
    tRef = tStart

    with printer.progress_logging(1):
        for (i, stringsToEstimate) in enumerate(circuitLists):
            #printer.log('', 2)
            extraMessages = ["(%s)" % circuitSetLabels[i]] if circuitSetLabels else []
            printer.show_progress(i, nIters, verboseMessages=extraMessages, prefix="--- Iterative MC2GST:",
                                  suffix=" %d operation sequences ---" % len(stringsToEstimate))

            if stringsToEstimate is None or len(stringsToEstimate) == 0: continue

            if circuitWeightsDict is not None:
                circuitWeights = _np.ones(len(stringsToEstimate), 'd')
                for opstr, weight in circuitWeightsDict.items():
                    if opstr in stringsToEstimate:
                        circuitWeights[stringsToEstimate.index(opstr)] = weight
            else: circuitWeights = None
            lsgstModel.basis = startModel.basis
            num_fd = fditer if (i == 0) else 0

            evt_cache = {}  # get the eval tree that's created so we can reuse it
            minErr, lsgstModel = \
                do_mc2gst(dataset, lsgstModel, stringsToEstimate,
                          maxiter, maxfev, num_fd, tol, extra_lm_opts,
                          cptp_penalty_factor, spam_penalty_factor,
                          minProbClipForWeighting, probClipInterval,
                          useFreqWeightedChiSq, regularizeFactor,
                          printer - 1, check, check_jacobian,
                          circuitWeights, opLabelAliases, memLimit, comm,
                          distributeMethod, profiler, evt_cache, time_dependent)
            if returnAll:
                lsgstModels.append(lsgstModel)
                minErrs.append(minErr)

            if evaltree_cache is not None:
                evaltree_cache.update(evt_cache)  # final evaltree cache

            tNxt = _time.time()
            profiler.add_time('do_iterative_mc2gst: iter %d chi2-opt' % (i + 1), tRef)
            printer.log("    Iteration %d took %.1fs" % (i + 1, tNxt - tRef), 2)
            printer.log('', 2)  # extra newline
            tRef = tNxt

    printer.log('Iterative MC2GST Total Time: %.1fs' % (_time.time() - tStart))
    profiler.add_time('do_iterative_mc2gst: total time', tStart)

    if returnErrorVec:
        return (minErrs, lsgstModels) if returnAll else (minErr, lsgstModel)
    else:
        return lsgstModels if returnAll else lsgstModel


def do_iterative_mc2gst_with_model_selection(
        dataset, startModel, dimDelta, circuitSetsToUseInEstimation,
        maxiter=100000, maxfev=None, tol=1e-6,
        cptp_penalty_factor=0, spam_penalty_factor=0,
        minProbClipForWeighting=1e-4, probClipInterval=(-1e6, 1e6),
        useFreqWeightedChiSq=False, regularizeFactor=0, returnErrorVec=False,
        returnAll=False, circuitSetLabels=None, verbosity=0, check=False,
        check_jacobian=False, circuitWeightsDict=None,
        opLabelAliases=None, memLimit=None, comm=None):
    """
    Performs Iterative Minimum Chi^2 Gate Set Tomography on the dataset, and at
    each iteration tests the current model model against model models with
    an increased and/or decreased dimension (model selection).

    Parameters
    ----------
    dataset : DataSet
        The data used to generate MC2GST gate estimates

    startModel : Model
        The Model used as a starting point for the least-squares
        optimization.

    dimDelta : integer
        Amount by which to increment or decrement the dimension of the
        current model when performing model selection

    circuitSetsToUseInEstimation : list of lists of (tuples or Circuits)
        The i-th element lists the operation sequences to be used in the i-th iteration of MC2GST.
        Each element is a list of operation label tuples.
        e.g. [ [ (), ('Gx',) ], [ (), ('Gx',), ('Gy',) ], [ (), ('Gx',), ('Gy',), ('Gx','Gy') ]  ]

    maxiter : int, optional
        Maximum number of iterations for the chi^2 optimization.

    maxfev : int, optional
        Maximum number of function evaluations for the chi^2 optimization.
        Defaults to maxiter.

    tol : float, optional
        The tolerance for the chi^2 optimization.  If a dict, allowed keys are
        `'relx'`, `'relf'`, `'f'`, and `'jac'`.  If a float, then
        `{'relx': 1e-8, 'relf': tol, 'f': 1.0, 'jac': tol }` is used.

    cptp_penalty_factor : float, optional
        If greater than zero, the optimization also contains CPTP penalty
        terms which penalize non-CPTP-ness of the gates being optimized.  This factor
        multiplies these CPTP penalty terms.

    spam_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains SPAM penalty
        terms which penalize non-positive-ness of the state preps being optimized.  This
        factor multiplies these SPAM penalty terms.

    minProbClipForWeighting : float, optional
        Sets the minimum and maximum probability p allowed in the chi^2 weights: N/(p*(1-p))
        by clipping probability p values to lie within the interval
        [ minProbClipForWeighting, 1-minProbClipForWeighting ].

    probClipInterval : 2-tuple or None, optional
       (min,max) values used to clip the probabilities predicted by models during MC2GST's
       least squares search for an optimal model (if not None).  if None, no clipping is performed.

    useFreqWeightedChiSq : bool, optional
        If True, objective function uses only an approximate chi^2 weighting:  N/(f*(1-f))
        where f is the frequency obtained from the dataset, instead of the true chi^2: N/(p*(1-p))
        where p is a predicted probability.  Defaults to False, and only should use
        True for backward compatibility.

    regularizeFactor : float, optional
        Multiplicative prefactor of L2-like regularization term that penalizes model entries
        which have absolute value greater than 1.  When set to 0, no regularization is applied.
        Defaults to 0.

    returnErrorVec : bool, optional
        If True, return (errorVec, model), or (errorVecs, models) if
        returnAll == True, instead of just the model or models.

    returnAll : bool, optional
        If True return a list of models (and errorVecs if returnErrorVec == True),
        one per iteration, instead of the results from just the final iteration.

    circuitSetLabels : list of strings, optional
        An identification label for each of the operation sequence sets (used for displaying
        progress).  Must be the same length as circuitSetsToUseInEstimation.

    verbosity : int, optional
        How much detail to send to stdout.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    check_jacobian : boolean, optional
        If True, compare the analytic jacobian with a forward finite difference jacobean
        and print warning messages if there is disagreement.

    circuitWeightsDict : dictionary, optional
        A dictionary with keys == operation sequences and values == multiplicative scaling
        factor for the corresponding operation sequence. The default is no weight scaling at all.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are circuits
        corresponding to what that operation label should be expanded into before querying
        the dataset.  Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = pygsti.obj.Circuit(['Gx','Gx','Gx'])

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.


    Returns
    -------
    model               if returnAll == False and returnErrorVec == False
    models              if returnAll == True  and returnErrorVec == False
    (errorVec, model)   if returnAll == False and returnErrorVec == True
    (errorVecs, models) if returnAll == True  and returnErrorVec == True
        where errorVec is a numpy array of minimum error values v = f(x_min), where f(x)**2 is
        the function being minimized, model is the Model containing the final estimated gates.
        In cases when returnAll == True, models and errorVecs are lists whose i-th elements are the
        errorVec and model corresponding to the results of the i-th iteration.
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    #TODO REMOVE - now Circuits are tuples + linelabels, so we need the entire object
    #convert lists of Circuits to lists of raw tuples since that's all we'll need
    #if len(circuitSetsToUseInEstimation) > 0 and \
    #   len(circuitSetsToUseInEstimation[0]) > 0 and \
    #   isinstance(circuitSetsToUseInEstimation[0][0], _objs.Circuit):
    #    circuitLists = [[opstr.tup for opstr in gsList] for gsList in circuitSetsToUseInEstimation]
    #else:
    circuitLists = circuitSetsToUseInEstimation

    #Run MC2GST iteratively on given sets of estimatable strings
    lsgstModels = []; minErrs = []  # for returnAll == True case
    lsgstModel = startModel.copy(); nIters = len(circuitLists)
    with printer.progress_logging(1):
        for (i, stringsToEstimate) in enumerate(circuitLists):
            #printer.log('', 2)
            extraMessages = (["(%s) "] % circuitSetLabels[i]) if circuitSetLabels else []
            printer.show_progress(i, nIters, prefix="--- Iterative MC2GST:", suffix="%d operation sequences ---" %
                                  (len(stringsToEstimate)), verboseMessages=extraMessages)

            if stringsToEstimate is None or len(stringsToEstimate) == 0: continue

            if circuitWeightsDict is not None:
                circuitWeights = _np.ones(len(stringsToEstimate), 'd')
                for opstr, weight in circuitWeightsDict.items():
                    if opstr in stringsToEstimate:
                        circuitWeights[stringsToEstimate.index(opstr)] = weight
            else: circuitWeights = None

            minErr, lsgstModel = do_mc2gst_with_model_selection(
                dataset, lsgstModel, dimDelta, stringsToEstimate,
                maxiter, maxfev, tol,
                cptp_penalty_factor, spam_penalty_factor,
                minProbClipForWeighting, probClipInterval,
                useFreqWeightedChiSq, regularizeFactor, printer - 1,
                check, check_jacobian, circuitWeights,
                opLabelAliases, memLimit, comm)

            if returnAll:
                lsgstModels.append(lsgstModel)
                minErrs.append(minErr)

    if returnErrorVec:
        return (minErrs, lsgstModels) if returnAll else (minErr, lsgstModel)
    else:
        return lsgstModels if returnAll else lsgstModel


###################################################################################
#                 Maximum Likelihood Estimation GST (MLGST)
##################################################################################

def do_mlgst(dataset, startModel, circuitsToUse,
             maxiter=100000, maxfev=None, fditer=0, tol=1e-6, extra_lm_opts=None,
             cptp_penalty_factor=0, spam_penalty_factor=0,
             minProbClip=1e-4, probClipInterval=(-1e6, 1e6), radius=1e-4,
             poissonPicture=True, verbosity=0, check=False,
             circuitWeights=None, opLabelAliases=None,
             memLimit=None, comm=None,
             distributeMethod="deriv", profiler=None,
             evaltree_cache=None, time_dependent=False):
    """
    Performs Maximum Likelihood Estimation Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to generate MLGST gate estimates

    startModel : Model
        The Model used as a starting point for the maximum-likelihood estimation.

    maxiter : int, optional
        Maximum number of iterations for the logL optimization.

    maxfev : int, optional
        Maximum number of function evaluations for the logL optimization.
        Defaults to maxiter.

    fditer : int, optional
        The number of iterations to perform with a finite-difference-Jacobian,
        as opposed to the more precise "analytic" Jacobian.  Useful if the
        optimization's starting point is special/singular.

    tol : float or dict, optional
        The tolerance for the logL optimization.  If a dict, allowed keys are
        `'relx'`, `'relf'`, `'f'`, `'jac'`, and `'maxdx'`.  If a float, then
        `{'relx': 1e-8, 'relf': tol, 'f': 1.0, 'jac': tol, 'maxdx': 1.0 }` is used.

    extra_lm_opts : dict or None, optional
        Additional options for the Levenberg-Marquardt algorithm.

    cptp_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains CPTP penalty
        terms which penalize non-CPTP-ness of the gates being optimized.  This factor
        multiplies these CPTP penalty terms.

    spam_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains SPAM penalty
        terms which penalize non-positive-ness of the state preps being optimized.  This
        factor multiplies these SPAM penalty terms.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poissonPicture : boolean, optional
        Whether the Poisson-picture log-likelihood should be used.

    verbosity : int, optional
        How much detail to send to stdout.

    check : boolean, optional
      If True, perform extra checks within code to verify correctness.  Used
      for testing, and runs much slower when True.

    circuitWeights : numpy array, optional
      An array of length len(circuitsToUse).  Each element scales the
      log-likelihood term of the corresponding operation sequence in circuitsToUse.
      The default is no weight scaling at all.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are circuits
        corresponding to what that operation label should be expanded into before querying
        the dataset.  Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = pygsti.obj.Circuit(['Gx','Gx','Gx'])

    memLimit : int, optional
      A rough memory limit in bytes which restricts the amount of intermediate
      values that are computed and stored.

    comm : mpi4py.MPI.Comm, optional
      When not None, an MPI communicator for distributing the computation
      across multiple processors.

    distributeMethod : {"circuits", "deriv"}
      How to distribute calculation amongst processors (only has effect
      when comm is not None).  "circuits" will divide the list of
      circuits; "deriv" will divide the columns of the jacobian matrix.

    profiler : Profiler, optional
        A profiler object used for to track timing and memory usage.

    evaltree_cache : dict, optional
        A dictionary which server as a cache for the computed EvalTree used
        in this computation.  If an empty dictionary is supplied, it is filled
        with cached values to speed up subsequent executions of this function
        which use the *same* `startModel`, `circuitsToUse`, `memLimit`,
        `comm`, and `distributeMethod`.

    time_dependent : bool, optional
        Whether any timestamps in the data should be taken seriously and used
        to compare with a potentially time-dependent model.


    Returns
    -------
    maxLogL : float
        The maximum log-likelihood obtained.
    model : Model
        The model that maximized the log-likelihood.
    """
    return _do_mlgst_base(dataset, startModel, circuitsToUse, maxiter, maxfev,
                          fditer, tol, extra_lm_opts, cptp_penalty_factor, spam_penalty_factor, minProbClip,
                          probClipInterval, radius, poissonPicture, verbosity,
                          check, circuitWeights, opLabelAliases, memLimit,
                          comm, distributeMethod, profiler, evaltree_cache, None,
                          100, time_dependent)


def _do_mlgst_base(dataset, startModel, circuitsToUse,
                   maxiter=100000, maxfev=None, fditer=0, tol=1e-6, extra_lm_opts=None,
                   cptp_penalty_factor=0, spam_penalty_factor=0,
                   minProbClip=1e-4, probClipInterval=(-1e6, 1e6), radius=1e-4,
                   poissonPicture=True, verbosity=0, check=False,
                   circuitWeights=None, opLabelAliases=None,
                   memLimit=None, comm=None,
                   distributeMethod="deriv", profiler=None,
                   evaltree_cache=None, forcefn_grad=None, shiftFctr=100,
                   time_dependent=False):
    """
    Same args and behavior as do_mlgst, but with additional:

    Parameters
    ----------
    evaltree_cache : dict, optional
        A dictionary which server as a cache for the computed EvalTree used
        in this computation.  If an empty dictionary is supplied, it is filled
        with cached values to speed up subsequent executions of this function
        which use the *same* `startModel`, `circuitsToUse`, `memLimit`,
        `comm`, and `distributeMethod`.

    forcefn_grad : numpy array, optional
        An array of shape `(D,nParams)`, where `D` is the dimension of the
        (unspecified) forcing function and `nParams=startModel.num_params()`.
        This array gives the gradient of the forcing function with respect to
        each model parameter and is used for the computation of "linear
        response error bars".

    shiftFctr : float
        A tuning parameter used to ensure the positivity of the forcing term.
        This should be > 1, and the larger the value the more positive-shift
        is applied to keep the forcing term positive.  Thus, if you receive
        an "Inadequate forcing shift" error, make this value larger.
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if profiler is None: profiler = _dummy_profiler
    tStart = _time.time()
    mdl = startModel  # .copy()  # to allow caches in startModel to be retained

    if maxfev is None: maxfev = maxiter

    #printer.log('', 2)
    printer.log("--- MLGST ---", 1)

    if comm is not None:
        #assume all models at least have same parameters - so just compare vecs
        v_cmp = comm.bcast(mdl.to_vector() if (comm.Get_rank() == 0) else None, root=0)
        if _np.linalg.norm(mdl.to_vector() - v_cmp) > 1e-6:
            raise ValueError("MPI ERROR: *different* MC2GST start models"
                             " given to different processors!")                   # pragma: no cover

        if forcefn_grad is not None:
            forcefn_cmp = comm.bcast(forcefn_grad if (comm.Get_rank() == 0) else None, root=0)
            normdiff = _np.linalg.norm(forcefn_cmp - forcefn_grad)
            if normdiff > 1e-6:
                #printer.warning("forcefn_grad norm mismatch = ",normdiff) #only prints on rank0
                _warnings.warn("Rank %d: forcefn_grad norm mismatch = %g" %
                               (comm.Get_rank(), normdiff))  # pragma: no cover
            #assert(normdiff <= 1e-6)
            forcefn_grad = forcefn_cmp  # use broadcast value to make certain each proc has *exactly* the same input

    #Memory allocation
    ns = int(round(_np.sqrt(mdl.dim)))  # estimate avg number of spamtuples per string
    ng = len(circuitsToUse)
    ne = mdl.num_params()
    C = 1.0 / 1024.0**3

    #  Estimate & check persistent memory (from allocs directly below)
    persistentMem = 8 * (ng * (ns + ns * ne + 1 * ns))  # final results in bytes
    if memLimit is not None and memLimit < persistentMem:
        raise MemoryError("Memory limit ({} GB) is < memory required to hold final results "
                          "({} GB)".format(memLimit * C, persistentMem * C))

    #Get evaluation tree (split into subtrees if needed)
    if (memLimit is not None):
        curMem = _objs.profiler._get_max_mem_usage(comm)
        gthrMem = int(0.1 * (memLimit - persistentMem))
        mlim = memLimit - persistentMem - gthrMem - curMem
        assert mlim > 0, 'Not enough memory, exiting..'
        printer.log("Memory: limit = %.2fGB" % (memLimit * C)
                    + "(cur, persist, gthr = %.2f, %.2f, %.2f GB)"
                    % (curMem * C, persistentMem * C, gthrMem * C))
    else: gthrMem = mlim = None

    if evaltree_cache and 'evTree' in evaltree_cache \
            and 'wrtBlkSize' in evaltree_cache:
        #use cache dictionary to speed multiple calls which use
        # the same model, operation sequences, comm, memlim, etc.
        evTree = evaltree_cache['evTree']
        wrtBlkSize = evaltree_cache['wrtBlkSize']
        lookup = evaltree_cache['lookup']
        outcomes_lookup = evaltree_cache['outcomes_lookup']
    else:
        # Note: simplify_circuits doesn't support aliased dataset (yet)
        dstree = dataset if (opLabelAliases is None) else None
        evTree, wrtBlkSize, _, lookup, outcomes_lookup = mdl.bulk_evaltree_from_resources(
            circuitsToUse, comm, mlim, distributeMethod,
            ["bulk_fill_probs", "bulk_fill_dprobs"], dstree, printer - 1)

        #Fill cache dict if one was given
        if evaltree_cache is not None:
            evaltree_cache['evTree'] = evTree
            evaltree_cache['wrtBlkSize'] = wrtBlkSize
            evaltree_cache['lookup'] = lookup
            evaltree_cache['outcomes_lookup'] = outcomes_lookup

    #Expand operation label aliases used in DataSet lookups
    dsCircuitsToUse = _tools.apply_aliases_to_circuit_list(circuitsToUse, opLabelAliases)

    if evaltree_cache and 'cntVecMx' in evaltree_cache:
        cntVecMx = evaltree_cache['cntVecMx']
        totalCntVec = evaltree_cache['totalCntVec']
    else:
        KM = evTree.num_final_elements()  # shorthand for combined spam+circuit dimension
        cntVecMx = _np.empty(KM, 'd')
        totalCntVec = _np.empty(KM, 'd')
        for (i, opStr) in enumerate(dsCircuitsToUse):
            cnts = dataset[opStr].counts
            totalCntVec[lookup[i]] = sum(cnts.values())  # dataset[opStr].total
            cntVecMx[lookup[i]] = [cnts.get(x, 0) for x in outcomes_lookup[i]]
            # OLD: totalCntVec[ lookup[i] ] = dataset[opStr].total
            # OLD: cntVecMx[ lookup[i] ] = [ dataset[opStr][x] for x in outcomes_lookup[i] ]

        if circuitWeights is not None:
            #From this point downward, scaling cntVecMx, totalCntVec and
            # minusCntVecMx will scale the corresponding logL terms, as desired.
            for i in range(len(circuitsToUse)):
                cntVecMx[lookup[i]] *= circuitWeights[i]  # dim KM (K = nSpamLabels, M = nCircuits )
                totalCntVec[lookup[i]] *= circuitWeights[i]  # multiply N's by weights

        if evaltree_cache is not None:
            evaltree_cache['cntVecMx'] = cntVecMx
            evaltree_cache['totalCntVec'] = totalCntVec

    # The theoretical upper bound on the log(likelihood)
    logL_upperbound = _tools.logl_max(mdl, dataset, dsCircuitsToUse,
                                      poissonPicture, check, opLabelAliases, evaltree_cache)

    if time_dependent:
        objective = _objfns.TimeDependentLogLFunction(
            mdl, evTree, lookup, circuitsToUse, opLabelAliases, cptp_penalty_factor,
            spam_penalty_factor, dsCircuitsToUse, dataset, minProbClip, radius, probClipInterval,
            wrtBlkSize, gthrMem, forcefn_grad, poissonPicture, shiftFctr,
            check, comm, profiler, printer)

        #DEBUG TODO REMOVE (to use, also need to indent objective_func assignment below)
        #objective2 = _objfns.LogLFunction(mdl, evTree, lookup, circuitsToUse, opLabelAliases, cptp_penalty_factor,
        #                                  spam_penalty_factor, cntVecMx, totalCntVec, minProbClip, radius,
        #                                  probClipInterval,
        #                                  wrtBlkSize, gthrMem, forcefn_grad, poissonPicture, shiftFctr, check, comm,
        #                                  profiler, printer)
        #
        #def objective_func(v):
        #    ret1 = objective.fn(v)
        #    ret2 = objective2.fn(v)
        #    maxdiff = max(_np.abs(ret1-ret2).flatten())
        #    print("MAX DIFF = ",max(_np.abs(ret1-ret2).flatten()))
        #    if maxdiff > 1e-1: #not _np.allclose(ret1,ret2):
        #        print("DIFF TD-BASE: = ",ret1-ret2)
        #        #print("BASE = ",ret2)
        #        assert(False),"STOP"
        #    return ret1

    else:
        #Create a termgap-penalizable objective function in termgap case
        objective = _objfns.LogLFunction(mdl, evTree, lookup, circuitsToUse, opLabelAliases, cptp_penalty_factor,
                                         spam_penalty_factor, cntVecMx, totalCntVec, minProbClip, radius,
                                         probClipInterval, wrtBlkSize, gthrMem, forcefn_grad, poissonPicture,
                                         shiftFctr, check, comm, profiler, printer)

    profiler.add_time("do_mlgst: pre-opt", tStart)

    #Get number of maximal-model parameter ("dataset params") if needed for print messages
    tm = _time.time()
    if printer.verbosity > 0:
        # number of independent parameters in dataset (max. model # of params)
        nDataParams = dataset.get_degrees_of_freedom(
            dsCircuitsToUse, aggregate_times=not time_dependent)
    else:
        nDataParams = 0  # because it's never used

    profiler.add_time("do_mlgst: num data params", tm)

    if mdl.simtype in ("termgap", "termorder"):
        delta_logl = _do_term_runopt(evTree, mdl, objective, "logl", maxiter, maxfev, tol, fditer, extra_lm_opts, comm,
                                     printer, profiler, nDataParams, memLimit, logL_upperbound)  # updates mdl
    else:
        #Normal case of just a single "sub-iteration"
        #Run optimization (use leastsq)
        delta_logl, _ = _do_runopt(mdl, objective, "logl", maxiter, maxfev, tol, fditer, extra_lm_opts, comm,
                                   printer, profiler, nDataParams, memLimit, logL_upperbound)  # updates mdl

    printer.log("  Completed in %.1fs" % (_time.time() - tStart), 1)

    profiler.add_time("do_mlgst: post-opt", tm)
    profiler.add_time("do_mlgst: total time", tStart)
    return delta_logl, mdl


def do_iterative_mlgst(dataset, startModel, circuitSetsToUseInEstimation,
                       maxiter=100000, maxfev=None, fditer=0, tol=1e-6, extra_lm_opts=None,
                       cptp_penalty_factor=0, spam_penalty_factor=0,
                       minProbClip=1e-4, probClipInterval=(-1e6, 1e6), radius=1e-4,
                       poissonPicture=True, returnMaxLogL=False, returnAll=False,
                       circuitSetLabels=None, useFreqWeightedChiSq=False,
                       verbosity=0, check=False, circuitWeightsDict=None,
                       opLabelAliases=None, memLimit=None,
                       profiler=None, comm=None, distributeMethod="deriv",
                       alwaysPerformMLE=False, onlyPerformMLE=False, evaltree_cache=None,
                       time_dependent=False):
    """
    Performs Iterative Maximum Likelihood Estimation Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to generate MLGST gate estimates

    startModel : Model
        The Model used as a starting point for the least-squares
        optimization.

    circuitSetsToUseInEstimation : list of lists of (tuples or Circuits)
        The i-th element is a list of the operation sequences to be used in the i-th iteration
        of MLGST.  Each element of these lists is a operation sequence, specifed as
        either a Circuit object or as a tuple of operation labels (but all must be specified
        using the same type).
        e.g. [ [ (), ('Gx',) ], [ (), ('Gx',), ('Gy',) ], [ (), ('Gx',), ('Gy',), ('Gx','Gy') ]  ]

    maxiter : int, optional
        Maximum number of iterations for the logL optimization.

    maxfev : int, optional
        Maximum number of function evaluations for the logL optimization.

    fditer : int, optional
        The number of initial (step 0) iterations to perform with a
        finite-difference-Jacobian, as opposed to the more precise "analytic"
        Jacobian.  Useful if the optimization's starting point is special/singular.

    tol : float or dict, optional
        The tolerance for the logL optimization.  If a dict, allowed keys are
        `'relx'`, `'relf'`, `'f'`, `'jac'`, and `'maxdx'`.  If a float, then
        `{'relx': 1e-8, 'relf': tol, 'f': 1.0, 'jac': tol, 'maxdx': 1.0 }` is used.

    extra_lm_opts : dict or None, optional
        Additional options for the Levenberg-Marquardt algorithm.

    cptp_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains CPTP penalty
        terms which penalize non-CPTP-ness of the gates being optimized.  This factor
        multiplies these CPTP penalty terms.

    spam_penalty_factor : float, optional
        If greater than zero, the least squares optimization also contains SPAM penalty
        terms which penalize non-positive-ness of the state preps being optimized.  This
        factor multiplies these SPAM penalty terms.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined (which improves
        optimizer performance).

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by models during MLGST's
        search for an optimal model (if not None).  if None, no clipping is performed.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    poissonPicture : boolean, optional
        Whether the Poisson-picture log-likelihood should be used.

    returnAll : bool, optional
        If True return a list of models (and maxLogLs if returnMaxLogL == True),
        one per iteration, instead of the results from just the final iteration.

    circuitSetLabels : list of strings, optional
        An identification label for each of the operation sequence sets (used for displaying
        progress).  Must be the same length as circuitSetsToUseInEstimation.

    useFreqWeightedChiSq : bool, optional
        If True, chi-square objective function uses the approximate chi^2 weighting:  N/(f*(1-f))
        where f is the frequency obtained from the dataset, instead of the true chi^2: N/(p*(1-p))
        where p is a predicted probability.  Defaults to False, and only should use
        True for backward compatibility.

    returnAll : boolean, optional
        If True return a list of models
                        circuitSetLabels=None,

    verbosity : int, optional
        How much detail to send to stdout.

    check : boolean, optional
        If True, perform extra checks within code to verify correctness.  Used
        for testing, and runs much slower when True.

    circuitWeightsDict : dictionary, optional
        A dictionary with keys == operation sequences and values == multiplicative scaling
        factor for the corresponding operation sequence. The default is no weight scaling at all.

    opLabelAliases : dictionary, optional
        Dictionary whose keys are operation label "aliases" and whose values are circuits
        corresponding to what that operation label should be expanded into before querying
        the dataset.  Defaults to the empty dictionary (no aliases defined)
        e.g. opLabelAliases['Gx^3'] = pygsti.obj.Circuit(['Gx','Gx','Gx'])

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    profiler : Profiler, optional
         A profiler object used for to track timing and memory usage.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    distributeMethod : {"circuits", "deriv"}
        How to distribute calculation amongst processors (only has effect
        when comm is not None).  "circuits" will divide the list of
        circuits; "deriv" will divide the columns of the jacobian matrix.

    alwaysPerformMLE : bool, optional
        When True, perform a maximum-likelihood estimate after *every* iteration,
        not just the final one.  When False, chi2 minimization is used for all
        except the final iteration (for improved numerical stability).

    onlyPerformMLE : bool, optional
        When True, `alwaysPerformMLE` must also be true, and in this case only
        a ML optimization is performed for each iteration.

    evaltree_cache : dict, optional
        An empty dictionary which gets filled with the *final* computed EvalTree
        (and supporting info) used in this computation.

    time_dependent : bool, optional
        Whether any timestamps in the data should be taken seriously and used
        to compare with a potentially time-dependent model.


    Returns
    -------
    model               if returnAll == False and returnMaxLogL == False
    models              if returnAll == True  and returnMaxLogL == False
    (maxLogL, model)    if returnAll == False and returnMaxLogL == True
    (maxLogL, models)   if returnAll == True  and returnMaxLogL == True
        where maxLogL is the maximum log-likelihood, and model is the Model containing
        the final estimated gates.  In cases when returnAll == True, maxLogLs and models
        are lists whose i-th elements are the maxLogL and model corresponding to the results
        of the i-th iteration.
    """

    if onlyPerformMLE:
        assert(alwaysPerformMLE), "Must set `alwaysPerformMLE` to True whenever `onlyPerformMLE` is True."

    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)
    if profiler is None: profiler = _dummy_profiler

    #TODO REMOVE - now Circuits are tuples + linelabels, so we need the entire object
    #convert lists of Circuits to lists of raw tuples since that's all we'll need
    #if len(circuitSetsToUseInEstimation) > 0 and \
    #   len(circuitSetsToUseInEstimation[0]) > 0 and \
    #   isinstance(circuitSetsToUseInEstimation[0][0], _objs.Circuit):
    #    circuitLists = [[opstr.tup for opstr in gsList] for gsList in circuitSetsToUseInEstimation]
    #else:
    circuitLists = circuitSetsToUseInEstimation

    #Run extended MLGST iteratively on given sets of estimatable strings
    mleModels = []; maxLogLs = []  # for returnAll == True case
    mleModel = startModel.copy(); nIters = len(circuitLists)
    tStart = _time.time()
    tRef = tStart

    with printer.progress_logging(1):
        for (i, stringsToEstimate) in enumerate(circuitLists):
            #printer.log('', 2)
            extraMessages = [("(%s) " % circuitSetLabels[i])] if circuitSetLabels else []
            printer.show_progress(i, nIters, verboseMessages=extraMessages,
                                  prefix="--- Iterative MLGST:",
                                  suffix=" %d operation sequences ---" % len(stringsToEstimate))
            #print("DB: OPCACHE len = ",len(mleModel._opcache))

            if stringsToEstimate is None or len(stringsToEstimate) == 0: continue

            if circuitWeightsDict is not None:
                circuitWeights = _np.ones(len(stringsToEstimate), 'd')
                for opstr, weight in circuitWeightsDict.items():
                    if opstr in stringsToEstimate:
                        circuitWeights[stringsToEstimate.index(opstr)] = weight
            else: circuitWeights = None

            mleModel.basis = startModel.basis
            #set basis in case of CPTP constraints

            num_fd = fditer if (i == 0) else 0

            evt_cache = {}  # get the eval tree that's created so we can reuse it
            if not onlyPerformMLE:
                _, mleModel = do_mc2gst(dataset, mleModel, stringsToEstimate,
                                        maxiter, maxfev, num_fd, tol, extra_lm_opts, cptp_penalty_factor,
                                        spam_penalty_factor, minProbClip,
                                        probClipInterval, useFreqWeightedChiSq, 0, printer - 1, check,
                                        check, circuitWeights, opLabelAliases,
                                        memLimit, comm, distributeMethod, profiler, evt_cache,
                                        time_dependent)

            if alwaysPerformMLE:
                _, mleModel = do_mlgst(dataset, mleModel, stringsToEstimate,
                                       maxiter, maxfev, num_fd, tol, extra_lm_opts,
                                       cptp_penalty_factor, spam_penalty_factor,
                                       minProbClip, probClipInterval, radius,
                                       poissonPicture, printer - 1, check, circuitWeights,
                                       opLabelAliases, memLimit, comm, distributeMethod, profiler,
                                       evt_cache, time_dependent)

            tNxt = _time.time()
            profiler.add_time('do_iterative_mlgst: iter %d chi2-opt' % (i + 1), tRef)
            tRef2 = tNxt

            logL_ub = _tools.logl_max(mleModel, dataset, stringsToEstimate,
                                      poissonPicture, check, opLabelAliases, evt_cache)
            # get maxLogL from chi2 estimate
            maxLogL = _tools.logl(mleModel, dataset, stringsToEstimate, minProbClip, probClipInterval,
                                  radius, poissonPicture, check, opLabelAliases, evt_cache, comm)

            printer.log("2*Delta(log(L)) = %g" % (2 * (logL_ub - maxLogL)), 2)

            tNxt = _time.time()
            profiler.add_time('do_iterative_mlgst: iter %d logl-comp' % (i + 1), tRef2)
            printer.log("Iteration %d took %.1fs" % (i + 1, tNxt - tRef), 2)
            printer.log('', 2)  # extra newline
            tRef = tNxt

            if i == len(circuitLists) - 1 and not alwaysPerformMLE:  # on the last iteration, do ML
                printer.log("Switching to ML objective (last iteration)", 2)

                mleModel.basis = startModel.basis

                maxLogL_p, mleModel_p = do_mlgst(
                    dataset, mleModel, stringsToEstimate, maxiter, maxfev, 0, tol, extra_lm_opts,
                    cptp_penalty_factor, spam_penalty_factor, minProbClip, probClipInterval, radius,
                    poissonPicture, printer - 1, check, circuitWeights, opLabelAliases,
                    memLimit, comm, distributeMethod, profiler, evt_cache, time_dependent)

                printer.log("2*Delta(log(L)) = %g" % (2 * (logL_ub - maxLogL_p)), 2)

                if maxLogL_p > maxLogL:  # if do_mlgst improved the maximum log-likelihood
                    maxLogL = maxLogL_p
                    mleModel = mleModel_p
                else:
                    printer.warning("MLGST failed to improve logl: retaining chi2-objective estimate")

                tNxt = _time.time()
                profiler.add_time('do_iterative_mlgst: iter %d logl-opt' % (i + 1), tRef)
                printer.log("Final MLGST took %.1fs" % (tNxt - tRef), 2)
                printer.log('', 2)  # extra newline
                tRef = tNxt

                if evaltree_cache is not None:
                    evaltree_cache.update(evt_cache)  # final evaltree cache

            if returnAll:
                mleModels.append(mleModel)
                maxLogLs.append(maxLogL)

    printer.log('Iterative MLGST Total Time: %.1fs' % (_time.time() - tStart))
    profiler.add_time('do_iterative_mlgst: total time', tStart)

    if returnMaxLogL:
        return (maxLogL, mleModels) if returnAll else (maxLogL, mleModel)
    else:
        return mleModels if returnAll else mleModel

###################################################################################
#                 Other Tools
###################################################################################


def find_closest_unitary_opmx(operationMx):
    """
    Get the closest operation matrix (by maximizing fidelity)
      to operationMx that describes a unitary quantum gate.

    Parameters
    ----------
    operationMx : numpy array
        The operation matrix to act on.

    Returns
    -------
    numpy array
        The resulting closest unitary operation matrix.
    """

    gate_JMx = _tools.jamiolkowski_iso(operationMx, choiMxBasis="std")
    # d = _np.sqrt(operationMx.shape[0])
    # I = _np.identity(d)

    #def getu_1q(basisVec):  # 1 qubit version
    #    return _spl.expm( 1j * (basisVec[0]*_tools.sigmax + basisVec[1]*_tools.sigmay + basisVec[2]*_tools.sigmaz) )
    def _get_gate_mx_1q(basisVec):  # 1 qubit version
        return _pc.single_qubit_gate(basisVec[0],
                                     basisVec[1],
                                     basisVec[2])

    if operationMx.shape[0] == 4:
        #bell = _np.transpose(_np.array( [[1,0,0,1]] )) / _np.sqrt(2)
        initialBasisVec = [0, 0, 0]  # start with I until we figure out how to extract target unitary
        #getU = getu_1q
        getGateMx = _get_gate_mx_1q
    # Note: seems like for 2 qubits bell = [1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ]/sqrt(4)
    # (4 zeros between 1's since state dimension is 4 ( == sqrt(gate dimension))
    else:
        raise ValueError("Can't get closest unitary for > 1 qubits yet -- need to generalize.")

    def _objective_func(basisVec):
        operationMx = getGateMx(basisVec)
        JU = _tools.jamiolkowski_iso(operationMx, choiMxBasis="std")
        # OLD: but computes JU in Pauli basis (I think) -> wrong matrix to fidelity check with gate_JMx
        #U = getU(basisVec)
        #vU = _np.dot( _np.kron(U,I), bell ) # "Choi vector" corresponding to unitary U
        #JU = _np.kron( vU, _np.transpose(_np.conjugate(vU))) # Choi matrix corresponding to U
        return -_tools.fidelity(gate_JMx, JU)

    # print_obj_func = _opt.create_obj_func_printer(_objective_func)
    solution = _spo.minimize(_objective_func, initialBasisVec, options={'maxiter': 10000},
                             method='Nelder-Mead', callback=None, tol=1e-8)  # if verbosity > 2 else None
    operationMx = getGateMx(solution.x)

    #print "DEBUG: Best fidelity = ",-solution.fun
    #print "DEBUG: Using vector = ", solution.x
    #print "DEBUG: Gate Mx = \n", operationMx
    #print "DEBUG: Chi Mx = \n", _tools.jamiolkowski_iso( operationMx)
    #return -solution.fun, operationMx
    return operationMx
