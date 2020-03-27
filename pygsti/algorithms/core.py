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


##################################################################################
#                 Long sequence GST
##################################################################################

def do_gst_fit(dataset, startModel, circuitsToUse, optimizer, objective_function_builder,
               resource_alloc, cache, verbosity=0):
    """
    Performs Gate Set Tomography on the dataset by optimizing the objective function
    built by `objective_function_builder`.

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

    optimizer : Optimizer or dict
        The optimizer to use, or a dictionary of optimizer parameters
        from which a default optimizer can be built.

    objective_function_builder : ObjectiveFunctionBuilder
        Defines the objective function that is optimized.

    resource_alloc : ResourceAllocation
        A resource allocation object containing information about how to
        divide computation amongst multiple processors and any memory
        limits that should be imposed.

    cache : ComputationCache
        A persistent cache used to hold information specific to the
        given dataset, model, and circuit list.

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    result : OptimizerResult
        the result of the optimization
    model : Model
        the best-fit model.
    """
    resource_alloc = _objfns.ResourceAllocation.build_resource_allocation(resource_alloc)
    comm = resource_alloc.comm
    profiler = resource_alloc.profiler
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    tStart = _time.time()
    mdl = startModel  # .copy()  # to allow caches in startModel to be retained

    if comm is not None:
        #assume all models at least have same parameters - so just compare vecs
        v_cmp = comm.bcast(mdl.to_vector() if (comm.Get_rank() == 0) else None, root=0)
        if _np.linalg.norm(mdl.to_vector() - v_cmp) > 1e-6:
            raise ValueError("MPI ERROR: *different* MC2GST start models"
                             " given to different processors!")                   # pragma: no cover

    objective = objective_function_builder.build(mdl, dataset, circuitsToUse, resource_alloc, cache, printer)
    profiler.add_time("do_gst_fit: pre-opt", tStart)
    printer.log("--- %s GST ---" % objective.name, 1)

    #Step 3: solve least squares minimization problem
    if mdl.simtype in ("termgap", "termorder"):
        opt_result = _do_term_runopt(mdl, objective, optimizer, resource_alloc, printer)
    else:
        #Normal case of just a single "sub-iteration"
        opt_result = _do_runopt(mdl, objective, optimizer, resource_alloc, printer)

    printer.log("Completed in %.1fs" % (_time.time() - tStart), 1)

    #if targetModel is not None:
    #  target_vec = targetModel.to_vector()
    #  targetErrVec = _objective_func(target_vec)
    #  return minErrVec, soln_gs, targetErrVec
    profiler.add_time("do_mc2gst: total time", tStart)
    #TODO: evTree.permute_computation_to_original(minErrVec) #Doesn't work b/c minErrVec is flattened
    # but maybe best to just remove minErrVec from return value since this isn't very useful
    # anyway?
    return opt_result, mdl


def do_iterative_gst(dataset, startModel, circuitLists,
                     optimizer, iteration_objfn_builders, final_objfn_builders,
                     resource_alloc, verbosity=0):
    """
    Performs Iterative Gate Set Tomography on the dataset.

    Parameters
    ----------
    dataset : DataSet
        The data used to generate MLGST gate estimates

    startModel : Model
        The Model used as a starting point for the least-squares
        optimization.

    circuitLists : list of lists of (tuples or Circuits)
        The i-th element is a list of the operation sequences to be used in the i-th iteration
        of the optimization.  Each element of these lists is a circuit, specifed as
        either a Circuit object or as a tuple of operation labels (but all must be specified
        using the same type).
        e.g. [ [ (), ('Gx',) ], [ (), ('Gx',), ('Gy',) ], [ (), ('Gx',), ('Gy',), ('Gx','Gy') ]  ]

    optimizer : Optimizer or dict
        The optimizer to use, or a dictionary of optimizer parameters
        from which a default optimizer can be built.

    iteration_objfn_builders, final_objfn_builders: list
        Lists of ObjectiveFunctionBuilder objects defining which objective functions
        should be optimizized (successively) on each iteration, and the additional
        objective functions to optimize on the final iteration.

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
    final_cache : ComputationCache
        The final iteration's computation cache.
    """
    resource_alloc = _objfns.ResourceAllocation.build_resource_allocation(resource_alloc)
    comm = resource_alloc.comm
    profiler = resource_alloc.profiler
    printer = _objs.VerbosityPrinter.build_printer(verbosity, comm)

    models = []; optimums = []
    mdl = startModel.copy(); nIters = len(circuitLists)
    tStart = _time.time()
    tRef = tStart

    with printer.progress_logging(1):
        for (i, circuitsToEstimate) in enumerate(circuitLists):
            extraMessages = []
            if isinstance(circuitsToEstimate, _objfns.BulkCircuitList) and circuitsToEstimate.name:
                extraMessages.append("(%s) " % circuitsToEstimate.name)

            printer.show_progress(i, nIters, verboseMessages=extraMessages,
                                  prefix="--- Iterative GST:", suffix=" %d circuits ---" % len(circuitsToEstimate))

            if circuitsToEstimate is None or len(circuitsToEstimate) == 0: continue

            mdl.basis = startModel.basis  # set basis in case of CPTP constraints (needed?)
            cache = _objfns.ComputationCache()  # store objects for this particular model, dataset, and circuit list

            for j, obj_fn_builder in enumerate(iteration_objfn_builders):
                tNxt = _time.time()
                optimizer.fditer = optimizer.first_fditer if (i == 0 and j == 0) else 0
                opt_result, mdl = do_gst_fit(dataset, mdl, circuitsToEstimate, optimizer, obj_fn_builder,
                                             resource_alloc, cache, printer - 1)
                profiler.add_time('do_iterative_gst: iter %d %s-opt' % (i + 1, obj_fn_builder.name), tNxt)

            tNxt = _time.time()
            printer.log("Iteration %d took %.1fs\n" % (i + 1, tNxt - tRef), 2)
            tRef = tNxt

            if i == len(circuitLists) - 1:  # the last iteration
                printer.log("Last iteration:", 2)

                for j, obj_fn_builder in enumerate(final_objfn_builders):
                    tNxt = _time.time()
                    mdl.basis = startModel.basis
                    opt_result, mdl = do_gst_fit(dataset, mdl, circuitsToEstimate, optimizer, obj_fn_builder,
                                                 resource_alloc, cache, printer - 1)
                    profiler.add_time('do_iterative_gst: final %s opt' % obj_fn_builder.name, tNxt)

                tNxt = _time.time()
                printer.log("Final optimization took %.1fs\n" % (tNxt - tRef), 2)
                tRef = tNxt

                #send final cache back to caller to facilitate more operations on the final (model, circuits, dataset)
                final_cache = cache

            models.append(mdl.copy())
            optimums.append(opt_result)

    printer.log('Iterative GST Total Time: %.1fs' % (_time.time() - tStart))
    profiler.add_time('do_iterative_gst: total time', tStart)
    return models, optimums, final_cache


def _do_runopt(mdl, objective, optimizer, resource_alloc, printer):
    """ TODO: docstring """

    profiler = resource_alloc.profiler

    #Perform actual optimization
    tm = _time.time()
    opt_result = optimizer.run(objective, resource_alloc.comm, profiler, printer)
    profiler.add_time("do_gst_fit: optimize", tm)

    if printer.verbosity > 0:
        #Don't compute num gauge params if it's expensive (>10% of mem limit) or unavailable
        if hasattr(mdl, 'num_elements'):
            memForNumGaugeParams = mdl.num_elements() * (mdl.num_params() + mdl.dim**2) \
                * FLOATSIZE  # see Model._buildup_dpg (this is mem for dPG)

            if resource_alloc.memLimit is None or 0.1 * resource_alloc.memLimit < memForNumGaugeParams:
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

        #Get number of maximal-model parameter ("dataset params") if needed for print messages
        # -> number of independent parameters in dataset (max. model # of params)
        tm = _time.time()
        nDataParams = objective.get_num_data_params()  #TODO - cache this somehow in term-based calcs...
        profiler.add_time("do_gst_fit: num data params", tm)

        chi2_k_qty = opt_result.chi2_k_distributed_qty  # total chi2 or 2*deltaLogL
        desc = objective.description
        # reject GST model if p-value < threshold (~0.05?)
        pvalue = 1.0 - _stats.chi2.cdf(chi2_k_qty, nDataParams - nModelParams)
        printer.log("%s = %g (%d data params - %d model params = expected mean of %g; p-value = %g)" %
                    (desc, chi2_k_qty, nDataParams, nModelParams, nDataParams - nModelParams, pvalue), 1)

    return opt_result


def _do_term_runopt(mdl, objective, optimizer, resource_alloc, printer):
    """ TODO: docstring """

    fwdsim = mdl._fwdsim()

    #Pipe these parameters in from fwdsim, even though they're used to control the term-stage loop
    maxTermStages = fwdsim.max_term_stages
    pathFractionThreshold = fwdsim.path_fraction_threshold  # 0 when not using path-sets
    oob_check_interval = fwdsim.oob_check_interval

    #assume a path set has already been chosen, as one should have been chosen
    # when evTree was created.
    evTree = objective.evTree
    comm, memLimit = resource_alloc.comm, resource_alloc.memLimit
    pathSet = fwdsim.get_current_pathset(evTree, comm)
    if pathSet:  # only some types of term "modes" (see fwdsim.mode) use path-sets
        pathFraction = pathSet.get_allowed_path_fraction()
        printer.log("Initial Term-stage model has %d failures and uses %.1f%% of allowed paths." %
                    (pathSet.num_failures, 100 * pathFraction))
    else:
        pathFraction = 1.0  # b/c "all" paths are used, and > pathFractionThreshold, which should be 0

    opt_result = None
    for sub_iter in range(maxTermStages):

        bFinalIter = (sub_iter == maxTermStages - 1) or (pathFraction > pathFractionThreshold)
        optimizer.oob_check_interval = oob_check_interval
        # don't stop early on last iter - do as much as possible.
        optimizer.oob_action = "reject" if bFinalIter else "stop"
        opt_result = _do_runopt(mdl, objective, optimizer, resource_alloc, printer)

        if not opt_result.msg == "Objective function out-of-bounds! STOP":
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
            optimizer.init_munu = opt_result.optimizer_specific_qtys['mu'], opt_result.optimizer_specific_qtys['nu']
            printer.log("After adapting paths, num failures = %d, %.1f%% of allowed paths used." %
                        (pathSet.num_failures, 100 * pathFraction))

    return opt_result


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

    gate_JMx = _tools.jamiolkowski_iso(operationMx, choi_mx_basis="std")
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
        JU = _tools.jamiolkowski_iso(operationMx, choi_mx_basis="std")
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
