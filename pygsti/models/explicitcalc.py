"""
Defines the ExplicitOpModelCalc class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


import collections as _collections
import itertools as _itertools
import warnings as _warnings

import numpy as _np

from pygsti.baseobjs import basisconstructors as _bc
from pygsti.tools import matrixtools as _mt

# Tolerace for matrix_rank when finding rank of a *normalized* projection
# matrix.  This is a legitimate tolerace since a normalized projection matrix
# should have just 0 and 1 eigenvalues, and thus a tolerace << 1.0 will work
# well.
P_RANK_TOL = 1e-7


class ExplicitOpModelCalc(object):
    """
    Performs calculations with explicitly-represented objects.

    This class performs calculations with *simplified* objects (so don't
    need to worry abount POVMs or Instruments, just preps, ops, & effects),
    but, unlike forward simulators, these calculations require knowledge of *all*
    of the possible operations in each category (not just the ones in a given
    circuti).  As such, instances of `ExplicitOpModelCalc` are almost always
    associated with an instance of `ExplicitOpModel`.

    Parameters
    ----------
    dim : int
        The dimenstion of the Hilbert-Schmidt space upon which the
        various operators act.

    simplified_preps : dict
        Dictionary containing *all* the possible state preparations.

    simplified_ops : dict
        Dictionary containing *all* the possible layer operations.

    simplified_effects : dict
        Dictionary containing *all* the possible POVM effects.

    np : int
        The total number of parameters in all the operators (the
        number of parameters of the associated :class:`ExplicitOpModel`).
    """

    def __init__(self, dim, simplified_preps, simplified_ops, simplified_effects, np, interposer=None):
        """
        Initialize a new ExplicitOpModelCalc object.

        Parameters
        ----------
        dim : int
            The dimenstion of the Hilbert-Schmidt space upon which the
            various operators act.

        simplified_preps, simplified_ops, simplified_effects : dict
            Dictionaries containing *all* the possible state preparations,
            layer operations, and POVM effects, respectively.

        np : int
            The total number of parameters in all the operators (the
            number of parameters of the associated :class:`ExplicitOpModel`).

        interposer : ModelParamsInterposer, optional
            An interposer object that converts between "operator" and "model" parameter arrays.
        """
        self.dim = dim
        self.preps = simplified_preps
        self.operations = simplified_ops
        self.effects = simplified_effects
        self.Np = np
        self.interposer = interposer

    def all_objects(self):
        """
        An iterator over all the state preparation, POVM effect, and layer operations.
        """
        for lbl, obj in _itertools.chain(self.preps.items(),
                                         self.effects.items(),
                                         self.operations.items()):
            yield (lbl, obj)

    def copy(self):
        """
        Return a shallow copy of this ExplicitOpModelCalc

        Returns
        -------
        ExplicitOpModelCalc
        """
        return ExplicitOpModelCalc(self.dim, self.preps, self.operations, self.effects, self.Np, self.interposer)

    def frobeniusdist(self, other_calc, transform_mx=None,
                      item_weights=None, normalize=True):
        """
        Compute the weighted frobenius norm of the difference between this calc object and `other_calc`.

        Differences in each corresponding gate matrix and spam vector element are squared,
        weighted (using `item_weights` as applicable), then summed.  The value returned is
        the square root of this sum, or the square root of this sum divided by the number
        of summands if normalize == True.

        Parameters
        ----------
        other_calc : ForwardSimulator
            the other gate calculator to difference against.

        transform_mx : numpy array or tuple, optional
            if transform_mx is a numpy array, then for each operation matrix
            G we implicitly consider the transformed quantity
                G => inv(transform_mx) * G * transform_mx
            Similar transformations are applied for effect vectors.
            This transformation is applied only for the difference and does not
            alter the values stored in this model.

            if transform_mx is a tuple then it should consist of two numpy arrays,
            the first of which will be interpreted as transform_mx in the usual
            sense and the second of which will either be None or will be syntactically
            subtituted for inv(transform_mx).

        item_weights : dict, optional
            Dictionary of weighting factors for individual gates and spam
            operators. Weights are applied multiplicatively to the squared
            differences, i.e., (*before* the final square root is taken).  Keys
            can be gate, state preparation, POVM effect, or spam labels, as well
            as the two special labels `"gates"` and `"spam"` which apply to all
            of the gate or SPAM elements, respectively (but are overridden by
            specific element values).  Values are floating point numbers.
            By default, all weights are 1.0.

        normalize : bool, optional
            if True (the default), the sum of weighted squared-differences
            is divided by the weighted number of differences before the
            final square root is taken.  If False, the division is not performed.

        Returns
        -------
        float
        """
        if isinstance(transform_mx, tuple):
            T, Ti = transform_mx
        else:
            T = transform_mx
            Ti = None if T is None else _np.linalg.pinv(T)
        d = 0
        nSummands = 0.0
        if item_weights is None: item_weights = {}
        opWeight = item_weights.get('gates', 1.0)
        spamWeight = item_weights.get('spam', 1.0)

        if (T is None and Ti is None) or isinstance(transform_mx, _np.ndarray):
            # use the original implementation.
            for opLabel, gate in self.operations.items():
                wt = item_weights.get(opLabel, opWeight)
                d += wt * gate.frobeniusdist_squared(other_calc.operations[opLabel], T, Ti)
                nSummands += wt * (gate.dim)**2

            for lbl, rhoV in self.preps.items():
                wt = item_weights.get(lbl, spamWeight)
                d += wt * rhoV.frobeniusdist_squared(other_calc.preps[lbl], T, Ti)
                nSummands += wt * rhoV.dim

            for lbl, Evec in self.effects.items():
                wt = item_weights.get(lbl, spamWeight)
                d += wt * Evec.frobeniusdist_squared(other_calc.effects[lbl], T, Ti)
                nSummands += wt * Evec.dim
        else:
            # we're in the special case that I'm creating.
            for opLabel, gate in self.operations.items():
                wt = item_weights.get(opLabel, opWeight)
                gate_mx = gate.to_dense()
                other_mx = other_calc.operations[opLabel].to_dense()
                delta = gate_mx - other_mx
                if T is not None:
                    delta = delta @ T
                if Ti is not None:
                    delta = Ti @ delta
                val = _np.linalg.norm(delta.flatten())
                d += wt * val**2
                nSummands += wt * (gate.dim)**2

            for lbl, rhoV in self.preps.items():
                # keep the original implementation, for now.
                wt = item_weights.get(lbl, spamWeight)
                d += wt * rhoV.frobeniusdist_squared(other_calc.preps[lbl], None, None)
                nSummands += wt * rhoV.dim

            for lbl, Evec in self.effects.items():
                # keep the original implementation, for now.
                wt = item_weights.get(lbl, spamWeight)
                d += wt * Evec.frobeniusdist_squared(other_calc.effects[lbl], None, None)
                nSummands += wt * Evec.dimxw

        if normalize and nSummands > 0:
            return _np.sqrt(d / nSummands)
        else:
            return _np.sqrt(d)

    def residuals(self, other_calc, transform_mx=None, item_weights=None):
        """
        Compute the weighted residuals between two models/calcs.

        Residuals are the differences in corresponding operation matrix
        and spam vector elements.

        Parameters
        ----------
        other_calc : ForwardSimulator
            the other gate calculator to difference against.

        transform_mx : numpy array, optional
            if not None, transform this model by
            G => inv(transform_mx) * G * transform_mx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

        item_weights : dict, optional
            Dictionary of weighting factors for individual gates and spam
            operators. Weights applied such that they act multiplicatively on
            the *squared* differences, so that the residuals themselves are
            scaled by the square roots of these weights.  Keys can be gate, state
            preparation, POVM effect, or spam labels, as well as the two special
            labels `"gates"` and `"spam"` which apply to all of the gate or SPAM
            elements, respectively (but are overridden by specific element
            values).  Values are floating point numbers.  By default, all weights
            are 1.0.

        Returns
        -------
        residuals : numpy.ndarray
            A 1D array of residuals (differences w.r.t. other)
        nSummands : int
            The (weighted) number of elements accounted for by the residuals.
        """
        resids = []
        T = transform_mx
        nSummands = 0.0
        if item_weights is None: item_weights = {}
        sqrt_itemWeights = {k: _np.sqrt(v) for k, v in item_weights.items()}
        opWeight = sqrt_itemWeights.get('gates', 1.0)
        spamWeight = sqrt_itemWeights.get('spam', 1.0)
        Ti = None if T is None else _np.linalg.pinv(T)
        # ^ TODO: generalize inverse op (call T.inverse() if T were a "transform" object?)

        for opLabel, gate in self.operations.items():
            wt = sqrt_itemWeights.get(opLabel, opWeight)
            other_gate = other_calc.operations[opLabel]
            resid =  wt * gate.residuals(other_gate, T, Ti)
            resids.append(resid)
            nSummands += wt**2 * (gate.dim)**2

        for lbl, rhoV in self.preps.items():
            wt = sqrt_itemWeights.get(lbl, spamWeight)
            other_prep = other_calc.preps[lbl]
            resid = wt * rhoV.residuals(other_prep, T, Ti)
            resids.append(resid)
            nSummands += wt**2 * rhoV.dim

        for lbl, Evec in self.effects.items():
            wt = sqrt_itemWeights.get(lbl, spamWeight)
            other_effect = other_calc.effects[lbl]
            resid = wt * Evec.residuals(other_effect, T, Ti)
            resids.append(resid)

            nSummands += wt**2 * Evec.dim

        resids = [r.ravel() for r in resids]
        resids = _np.concatenate(resids)
        return resids, nSummands

    def jtracedist(self, other_calc, transform_mx=None, include_spam=True):
        """
        Compute the Jamiolkowski trace distance between two models/calcs.

        This is defined as the maximum of the trace distances
        between each corresponding gate, including spam gates.

        Parameters
        ----------
        other_calc : ForwardSimulator
            the other model to difference against.

        transform_mx : numpy array, optional
            if not None, transform this model by
            G => inv(transform_mx) * G * transform_mx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

        include_spam : bool, optional
            Whether to add to the max-trace-distance the frobenius distances
            between corresponding SPAM vectors.

        Returns
        -------
        float
        """
        T = transform_mx
        d = 0  # spam difference
        nSummands = 0  # for spam terms

        if T is not None:
            Ti = _np.linalg.inv(T)
            dists = [gate.jtracedist(other_calc.operations[lbl], T, Ti)
                     for lbl, gate in self.operations.items()]

            #Just use frobenius distance between spam vecs, since jtracedist
            # doesn't really make sense
            if include_spam:
                for lbl, rhoV in self.preps.items():
                    d += rhoV.frobeniusdist_squared(other_calc.preps[lbl], T, Ti)
                    nSummands += rhoV.dim

                for lbl, Evec in self.effects.items():
                    d += Evec.frobeniusdist_squared(other_calc.effects[lbl], T, Ti)
                    nSummands += Evec.dim

        else:
            dists = [gate.jtracedist(other_calc.operations[lbl])
                     for lbl, gate in self.operations.items()]

            #Just use frobenius distance between spam vecs, since jtracedist
            # doesn't really make sense
            if include_spam:
                for lbl, rhoV in self.preps.items():
                    d += rhoV.frobeniusdist_squared(other_calc.preps[lbl])
                    nSummands += rhoV.dim

                for lbl, Evec in self.effects.items():
                    d += Evec.frobeniusdist_squared(other_calc.effects[lbl])
                    nSummands += Evec.dim

        spamVal = _np.sqrt(d / nSummands) if (nSummands > 0) else 0
        return max(dists) + spamVal

    def diamonddist(self, other_calc, transform_mx=None, include_spam=True):
        """
        Compute the diamond-norm distance between two models/calcs.

        This is defined as the maximum of the diamond-norm distances between
        each corresponding gate, including spam gates.

        Parameters
        ----------
        other_calc : ForwardSimulator
            the other gate calculator to difference against.

        transform_mx : numpy array, optional
            if not None, transform this model by
            G => inv(transform_mx) * G * transform_mx, for each operation matrix G
            (and similar for rho and E vectors) before taking the difference.
            This transformation is applied only for the difference and does
            not alter the values stored in this model.

        include_spam : bool, optional
            Whether to add to the max-diamond-distance the frobenius distances
            between corresponding SPAM vectors.

        Returns
        -------
        float
        """
        T = transform_mx
        d = 0  # spam difference
        nSummands = 0  # for spam terms

        if T is not None:
            Ti = _np.linalg.inv(T)
            dists = [gate.diamonddist(other_calc.operations[lbl], T, Ti)
                     for lbl, gate in self.operations.items()]

            #Just use frobenius distance between spam vecs, since jtracedist
            # doesn't really make sense
            if include_spam:
                for lbl, rhoV in self.preps.items():
                    d += rhoV.frobeniusdist_squared(other_calc.preps[lbl], T, Ti)
                    nSummands += rhoV.dim

                for lbl, Evec in self.effects.items():
                    d += Evec.frobeniusdist_squared(other_calc.effects[lbl], T, Ti)
                    nSummands += Evec.dim

        else:
            dists = [gate.diamonddist(other_calc.operations[lbl])
                     for lbl, gate in self.operations.items()]

            #Just use frobenius distance between spam vecs, since jtracedist
            # doesn't really make sense
            if include_spam:
                for lbl, rhoV in self.preps.items():
                    d += rhoV.frobeniusdist_squared(other_calc.preps[lbl])
                    nSummands += rhoV.dim

                for lbl, Evec in self.effects.items():
                    d += Evec.frobeniusdist_squared(other_calc.effects[lbl])
                    nSummands += Evec.dim

        spamVal = _np.sqrt(d / nSummands) if (nSummands > 0) else 0
        return max(dists) + spamVal

    def deriv_wrt_params(self):
        """
        The element-wise derivative of all this calculator's operations.

        Constructs a matrix whose columns are the vectorized derivatives of all
        this calc object's (model's) raw matrix and vector *elements* (placed in
        a vector) with respect to each single model parameter.

        Returns
        -------
        numpy array
            2D array of derivatives.
        """
        num_els = sum([obj.hilbert_schmidt_size for _, obj in self.all_objects()])
        num_op_params = self.Np if (self.interposer is None) else self.interposer.num_op_params
        deriv = _np.zeros((num_els, num_op_params), 'd')

        eo = 0  # element offset
        for lbl, obj in self.all_objects():
            #Note: no overlaps possible b/c of independent *elements*
            deriv[eo:eo + obj.hilbert_schmidt_size, obj.gpindices] = obj.deriv_wrt_params()
            eo += obj.hilbert_schmidt_size

        if self.interposer is not None:
            deriv = _np.dot(deriv, self.interposer.deriv_op_params_wrt_model_params())

        return deriv

    def _buildup_dpg(self):
        """
        Helper function for building gauge/non-gauge projectors and
          for computing the number of gauge/non-gauge elements.
        Returns the `[ dP | dG ]` matrix, i.e. np.concatenate( (dP,dG), axis=1 )
        whose nullspace gives the gauge directions in parameter space.
        """

        from ..modelmembers.povms.complementeffect import ComplementPOVMEffect as _ComplementPOVMEffect
        # ** See comments at the beginning of nongauge_projector for explanation **

        on_space = 'minimal'
        try:
            self_operations = _collections.OrderedDict([(lbl, gate.to_dense(on_space=on_space))
                                                        for lbl, gate in self.operations.items()])
            self_preps = _collections.OrderedDict([(lbl, vec.to_dense(on_space=on_space)[:, None])
                                                   for lbl, vec in self.preps.items()])
            self_effects = _collections.OrderedDict([(lbl, vec.to_dense(on_space=on_space)[:, None])
                                                     for lbl, vec in self.effects.items()])
        except:
            raise NotImplementedError(("Cannot (yet) extract gauge/non-gauge "
                                       "parameters for Models with non-dense "
                                       "member representations"))

        bSkipEcs = True  # Whether we should artificially skip complement-type
        # effect vecs, which is historically what we've done, even though
        # this seems somewhat wrong.  Not skipping them will alter the
        # number of "gauge params" since a complement Evec has a *fixed*
        # identity from the perspective of the Model params (which are
        # *varied* in gauge optimization, even though it's not a POVMEffect
        # param, creating a weird inconsistency...) SKIP
        if bSkipEcs:
            newSelf = self.copy()
            newSelf.effects = self.effects.copy()  # b/c ForwardSimulator.__init__ doesn't copy members (for efficiency)
            for effectlbl, EVec in self.effects.items():
                if isinstance(EVec, _ComplementPOVMEffect):
                    del newSelf.effects[effectlbl]
            self = newSelf  # HACK!!! replacing self for remainder of this fn with version without Ecs

            #recompute effects in case we deleted any ComplementPOVMEffects
            self_effects = _collections.OrderedDict([(lbl, vec.to_dense(on_space=on_space)[:, None])
                                                     for lbl, vec in self.effects.items()])

        #Use a Model object to hold & then vectorize the derivatives wrt each gauge transform basis element (each ij)
        dim = self.dim
        nParams = self.Np

        nElements = sum([obj.hilbert_schmidt_size for _, obj in self.all_objects()])
        #nElements = sum([o.hilbert_schmidt_size for o in self_operations.values()]) + \
        #            sum([o.hilbert_schmidt_size for o in self_preps.values()]) + \
        #            sum([o.hilbert_schmidt_size for o in self_effects.values()])

        #This was considered as optional behavior, but better to just delete qtys from Model
        ##whether elements of the raw model matrices/SPAM vectors that are not
        ## parameterized at all should be ignored.   By ignoring changes to such
        ## elements, they are treated as not being a part of the "model"
        #bIgnoreUnparameterizedEls = True

        #Note: model object (mdlDeriv) must have all elements of gate
        # mxs and spam vectors as parameters (i.e. be "fully parameterized") in
        # order to match deriv_wrt_params call, which gives derivatives wrt
        # *all* elements of a model.

        mdlDeriv_ops = _collections.OrderedDict(
            [(label, _np.zeros((dim, dim), 'd')) for label in self_operations])
        mdlDeriv_preps = _collections.OrderedDict(
            [(label, _np.zeros((dim, 1), 'd')) for label in self_preps])
        mdlDeriv_effects = _collections.OrderedDict(
            [(label, _np.zeros((dim, 1), 'd')) for label in self_effects])

        dPG = _np.empty((nElements, nParams + dim**2), 'd')
        for i in range(dim):      # always range over all rows: this is the
            for j in range(dim):  # *generator* mx, not gauge mx itself
                unitMx = _bc.mut(i, j, dim)
                for lbl, rhoVec in self_preps.items():
                    mdlDeriv_preps[lbl] = _np.dot(unitMx, rhoVec)
                for lbl, EVec in self_effects.items():
                    mdlDeriv_effects[lbl] = -_np.dot(EVec.T, unitMx).T

                for lbl, gate in self_operations.items():
                    #if isinstance(gate,_op.DenseOperator):
                    mdlDeriv_ops[lbl] = _np.dot(unitMx, gate) - \
                        _np.dot(gate, unitMx)
                    #else:
                    #    #use acton... maybe throw error if dim is too large (maybe above?)
                    #    deriv = _np.zeros((dim,dim),'d')
                    #    uv = _np.zeros((dim,1),'d') # unit vec
                    #    for k in range(dim): #FUTURE: could optimize this by bookeeping and pulling this loop outward
                    #        uv[k] = 1.0; Guv = gate.acton(uv); uv[k] = 0.0 #get k-th col of operation matrix
                    #        # termA_mn = sum( U_mk*Gkn ) so U locks m=i,k=j => termA_in = 1.0*Gjn
                    #        # termB_mn = sum( Gmk*U_kn ) so U locks k=i,n=j => termB_mj = 1.0*Gmi
                    #        deriv[i,k] += Guv[j,0] # termA contrib
                    #        if k == i: # i-th col of operation matrix goes in deriv's j-th col
                    #            deriv[:,j] -= Guv[:,0] # termB contrib
                    #    mdlDeriv_ops[lbl] = deriv

                #Note: vectorize all the parameters in this full-
                # parameterization object, which gives a vector of length
                # equal to the number of model *elements*.
                to_vector = _np.concatenate(
                    [obj.ravel() for obj in _itertools.chain(
                        mdlDeriv_preps.values(), mdlDeriv_effects.values(),
                        mdlDeriv_ops.values())], axis=0)
                dPG[:, nParams + i * dim + j] = to_vector

        dPG[:, 0:nParams] = self.deriv_wrt_params()
        return dPG

    def nongauge_and_gauge_spaces(self, item_weights=None, non_gauge_mix_mx=None):
        nParams = self.Np
        dPG = self._buildup_dpg()

        #print("DB: shapes = ",dP.shape,dG.shape)
        nullsp = _mt.nullspace_qr(dPG)  # columns are nullspace basis vectors
        gauge_space = nullsp[0:nParams, :]  # take upper (gate-param-segment) of vectors for basis
        # of subspace intersection in gate-parameter space
        #Note: gauge_space is (nParams)x(nullSpaceDim==gaugeSpaceDim)

        # Build final non-gauge space by getting a mx of column vectors
        # orthogonal (w.r.t. some metric) to the cols of gauge_space:
        #     gauge_space^T @ metric @ nongauge_space = 0
        #       => nongauge_space = nullspace(gauge_space^T @ metric)
        #                         = nullspace(orthog_to^T) below

        if non_gauge_mix_mx is not None:
            msg = "You've set both non_gauge_mix_mx and item_weights, both of which"\
                + " set the gauge metric... You probably don't want to do this."
            assert(item_weights is None), msg

            # BEGIN GAUGE MIX ----------------------------------------
            # Let metric^T = I + mix_gauge_to_nongauge so
            # orthog_to = metric^T @ gauge_space
            #           = gauge_space + mix_gauge_to_nongauge @ gauge_space
            #           = gauge_space + nongauge_directions @ nongauge_mix_mx
            #
            # where the last line is motivated by the fact that we don't change the space we're
            # orthogonalizing with respect to if we just add gauge directions.
            nonGaugeDirections = _mt.nullspace_qr(gauge_space.T)

            #for each column of gen_dG, which is a gauge direction in model parameter space,
            # we add some amount of non-gauge direction, given by coefficients of the
            # numNonGaugeParams non-gauge directions.
            orthog_to = gauge_space + _np.dot(nonGaugeDirections, non_gauge_mix_mx)  # add non-gauge components in
            # dims: (nParams,n_gauge_params) + (nParams,n_non_gauge_params) * (n_non_gauge_params,n_gauge_params)
            # non_gauge_mix_mx is a (n_non_gauge_params,n_gauge_params) matrix whose i-th column specifies the
            #  coefficents to multipy each of the non-gauge directions by before adding them to the i-th
            #  direction to project out (i.e. what were the pure gauge directions).

        elif item_weights is not None:
            metric_diag = _np.ones(self.Np, 'd')
            opWeight = item_weights.get('gates', 1.0)
            spamWeight = item_weights.get('spam', 1.0)
            for lbl, gate in self.operations.items():
                metric_diag[gate.gpindices] = item_weights.get(lbl, opWeight)
            for lbl, vec in _itertools.chain(iter(self.preps.items()),
                                             iter(self.effects.items())):
                metric_diag[vec.gpindices] = item_weights.get(lbl, spamWeight)
            metric = _np.diag(metric_diag)
            #OLD: gen_ndG = _mt.nullspace(_np.dot(gen_dG.T,metric))
            orthog_to = _np.dot(metric.T, gauge_space)

        else:
            orthog_to = gauge_space

        #OLD: nongauge_space = _mt.nullspace(orthog_to.T) #cols are non-gauge directions
        nongauge_space = _mt.nullspace_qr(orthog_to.T)  # cols are non-gauge directions
        # print("DB: nullspace of gen_dG (shape = %s, rank=%d) = %s" \
        #       % (str(gen_dG.shape),_np.linalg.matrix_rank(gen_dG),str(gen_ndG.shape)))

        #REMOVE
        ## reduce gen_dG if it doesn't have full rank
        #u, s, vh = _np.linalg.svd(gen_dG, full_matrices=False)
        #rank = _np.count_nonzero(s > P_RANK_TOL)
        #if rank < gen_dG.shape[1]:
        #    gen_dG = u[:, 0:rank]

        assert(nongauge_space.shape[0] == gauge_space.shape[0] == nongauge_space.shape[1] + gauge_space.shape[1])
        return nongauge_space, gauge_space

    #UNUSED - just used for checking understanding of where the nonzero logL Hessian on gauge space comes from.
    def _gauge_orbit_curvature(self, item_weights=None, non_gauge_mix_mx=None):

        nParams = self.Np
        dPG = self._buildup_dpg()
        nongauge_space, gauge_space = self.nongauge_and_gauge_spaces(item_weights, non_gauge_mix_mx)

        #Fill Heps == hessian d2/d(eps_j)d(eps_i) exp(sum(eps_k unit_k)) G exp(-sum(eps_k unit_k)) for ops, etc.
        from ..modelmembers.povms.complementeffect import ComplementPOVMEffect as _ComplementPOVMEffect
        on_space = 'minimal'
        try:
            self_operations = _collections.OrderedDict([(lbl, gate.to_dense(on_space=on_space))
                                                        for lbl, gate in self.operations.items()])
            self_preps = _collections.OrderedDict([(lbl, vec.to_dense(on_space=on_space)[:, None])
                                                   for lbl, vec in self.preps.items()])
            self_effects = _collections.OrderedDict([(lbl, vec.to_dense(on_space=on_space)[:, None])
                                                     for lbl, vec in self.effects.items()])
        except:
            raise NotImplementedError(("Cannot (yet) extract gauge/non-gauge "
                                       "parameters for Models with non-dense "
                                       "member representations"))

        bSkipEcs = True  # see above
        if bSkipEcs:
            newSelf = self.copy()
            newSelf.effects = self.effects.copy()  # b/c ForwardSimulator.__init__ doesn't copy members (for efficiency)
            for effectlbl, EVec in self.effects.items():
                if isinstance(EVec, _ComplementPOVMEffect):
                    del newSelf.effects[effectlbl]
            self = newSelf  # HACK!!! replacing self for remainder of this fn with version without Ecs

            #recompute effects in case we deleted any ComplementPOVMEffects
            self_effects = _collections.OrderedDict([(lbl, vec.to_dense(on_space=on_space)[:, None])
                                                     for lbl, vec in self.effects.items()])

        #Use a Model object to hold & then vectorize the derivatives wrt each gauge transform basis element (each ij)
        dim = self.dim
        nParams = self.Np

        nElements = sum([obj.hilbert_schmidt_size for _, obj in self.all_objects()])
        mdlHess_ops = _collections.OrderedDict(
            [(label, _np.zeros((dim, dim), 'd')) for label in self_operations])
        mdlHess_preps = _collections.OrderedDict(
            [(label, _np.zeros((dim, 1), 'd')) for label in self_preps])
        mdlHess_effects = _collections.OrderedDict(
            [(label, _np.zeros((dim, 1), 'd')) for label in self_effects])

        Heps = _np.empty((nElements, dim**2, dim**2), 'd')  # dim**2 == num of gauge "units"
        for i1 in range(dim):      # always range over all rows: this is the
            for i2 in range(dim):      # always range over all rows: this is the
                unitMx_i = _bc.mut(i1, i2, dim)
                for j1 in range(dim):  # *generator* mx, not gauge mx itself
                    for j2 in range(dim):  # *generator* mx, not gauge mx itself
                        unitMx_j = _bc.mut(j1, j2, dim)
                        antiComm = (unitMx_i @ unitMx_j + unitMx_j @ unitMx_i)
                        for lbl, rhoVec in self_preps.items():
                            mdlHess_preps[lbl] = 0.5 * _np.dot(antiComm, rhoVec)
                        for lbl, EVec in self_effects.items():
                            mdlHess_effects[lbl] = 0.5 * _np.dot(EVec.T, antiComm).T
                        for lbl, gate in self_operations.items():
                            mdlHess_ops[lbl] = 0.5 * (antiComm @ gate + gate @ antiComm) \
                                - unitMx_i @ gate @ unitMx_j - unitMx_j @ gate @ unitMx_i

                        to_vector = _np.concatenate(
                            [obj.ravel() for obj in _itertools.chain(
                                mdlHess_preps.values(), mdlHess_effects.values(),
                                mdlHess_ops.values())], axis=0)
                        Heps[:, i1 * dim + i2, j1 * dim + j2] = to_vector

        paramspace_to_elspace = dPG[:, 0:nParams]
        nongauge_to_elspace = paramspace_to_elspace @ nongauge_space
        elspace_to_nongauge = _np.linalg.pinv(nongauge_to_elspace)
        physHeps = _np.einsum('kij,lk->lij', Heps, elspace_to_nongauge)

        T = _mt.nullspace_qr(dPG)[nParams:, :]  # cols = gauge directions in eps space (gauge -> eps)
        # coord change physHeps from eps => gauge = invT @ eps, so H => T.T @ H @ T
        Q = _np.einsum('ik,jkl,lm->jim', T.T, physHeps, T)  # ~ inv(T) @ physHeps @ T
        return Q

    def nongauge_projector(self, item_weights=None, non_gauge_mix_mx=None):
        """
        Constructs a projector onto the non-gauge parameter space.

        This is useful for isolating the gauge degrees of freedom from the non-gauge
        degrees of freedom.

        Parameters
        ----------
        item_weights : dict, optional
            Dictionary of weighting factors for individual gates and spam operators.
            Keys can be gate, state preparation, POVM effect, spam labels, or the
            special strings "gates" or "spam" whic represent the entire set of gate
            or SPAM operators, respectively.  Values are floating point numbers.
            These weights define the metric used to compute the non-gauge space,
            *orthogonal* the gauge space, that is projected onto.

        non_gauge_mix_mx : numpy array, optional
            An array of shape (n_non_gauge_params,n_gauge_params) specifying how to
            mix the non-gauge degrees of freedom into the gauge degrees of
            freedom that are projected out by the returned object.  This argument
            essentially sets the off-diagonal block of the metric used for
            orthogonality in the "gauge + non-gauge" space.  It is for advanced
            usage and typically left as None (the default).

        Returns
        -------
        numpy array
            The projection operator as a N x N matrix, where N is the number
            of parameters (obtained via num_params()).  This projector acts on
            parameter-space, and has rank equal to the number of non-gauge
            degrees of freedom.
        """

        # We want to divide the Model-space H (a Hilbert space, 56-dimensional in the 1Q, 3-gate, 2-vec case)
        # into the direct sum of gauge and non-gauge spaces, and find projectors onto each
        # sub-space (the non-gauge space in particular).
        #
        # Within the Model H-space lies a gauge-manifold of maximum chi2 (16-dimensional in 1Q case),
        #  where the gauge-invariant chi2 function is constant.  At a single point (Model) P on this manifold,
        #  chosen by (somewhat arbitrarily) fixing the gauge, we can talk about the tangent space
        #  at that point.  This tangent space is spanned by some basis (~16 elements for the 1Q case),
        #  which associate with the infinitesimal gauge transform ?generators? on the full space.
        #  The subspace of H spanned by the derivatives w.r.t gauge transformations at P (a Model) spans
        #  the gauge space, and the complement of this (in H) is the non-gauge space.
        #
        #  An element of the gauge group can be written gg = exp(-K), where K is a n x n matrix.  If K is
        #   hermitian then gg is unitary, but this need not be the case.  A gauge transform acts on a
        #   gatset via Model => gg^-1 G gg, i.e. G => exp(K) G exp(-K).  We care about the action of
        #   infinitesimal gauge tranformations (b/c the *derivative* vectors span the tangent space),
        #   which act as:
        #    G => (I+K) G (I-K) = G + [K,G] + ignored(K^2), where [K,G] := KG-GK
        #
        # To produce a projector onto the gauge-space, we compute the *column* vectors
        #  dG_ij = [K_ij,G], where K_ij is the i,j-th matrix unit (56x1 in the 1Q case, 16 such column vectors)
        #  and then form a projector in the standard way.
        #  (to project onto |v1>, |v2>, etc., form P = sum_i |v_i><v_i|)
        #
        #Typically nOpParams < len(dG_ij) and linear system is overconstrained
        #   and no solution is expected.  If no solution exists, simply ignore
        #
        # So we form P = sum_ij dG_ij * transpose(dG_ij) (56x56 in 1Q case)
        #              = dG * transpose(dG)              where dG_ij form the *columns* of dG (56x16 in 1Q case)
        # But dG_ij are not orthonormal, so really we need a slight modification,
        #  otherwise P'P' != P' as must be the case for a projector:
        #
        # P' = dG * (transpose(dG) * dG)^-1 * transpose(dG) (see wikipedia on projectors)
        #
        #    or equivalently (I think)
        #
        # P' = pseudo-inv(P)*P
        #
        #  since the pseudo-inv is defined by P*pseudo-inv(P) = I and so P'P' == P'
        #  and P' is our gauge-projector!

        # Note: In the general case of parameterized gates (i.e. non-fully parameterized gates), there are fewer
        #   gate parameters than the size of dG_ij ( < 56 in 1Q case).  In this general case, we want to know
        #   what (if any) change in gate parameters produces the change dG_ij of the operation matrix elements.
        #   That is, we solve dG_ij = derivWRTParams * dParams_ij for dParams_ij, where derivWRTParams is
        #   the derivative of the operation matrix elements with respect to the gate parameters (derivWRTParams
        #   is 56x(nOpParams) and dParams_ij is (nOpParams)x1 in the 1Q case) -- then substitute dG_ij
        #   with dParams_ij above to arrive at a (nOpParams)x(nOpParams) projector (compatible with
        #   hessian computed by model).
        #
        #   Actually, we want to know if what changes gate parameters
        #   produce changes in the span of all the dG_ij, so really we want the intersection of the space
        #   defined by the columns of derivWRTParams (the "gate-parameter range" space) and the dG_ij vectors.
        #
        #   This intersection is determined by nullspace( derivWRTParams | dG ), where the pipe denotes
        #     concatenating the matrices together.  Then, if x is a basis vector of the nullspace
        #     x[0:nOpParams] is the basis vector for the intersection space within the gate parameter space,
        #     that is, the analogue of dParams_ij in the single-dG_ij introduction above.
        #
        #   Still, we just substitue these dParams_ij vectors (as many as the nullspace dimension) for dG_ij
        #   above to get the general case projector.
        gen_ndG, _ = self.nongauge_and_gauge_spaces(item_weights, non_gauge_mix_mx)

        # ORIG WAY: use psuedo-inverse to normalize projector.  Ran into problems where
        #  default rcond == 1e-15 didn't work for 2-qubit case, but still more stable than inv method below
        P = _np.dot(gen_ndG, _np.transpose(gen_ndG))  # almost a projector, but cols of dG are not orthonormal
        Pp = _np.dot(_np.linalg.pinv(P, rcond=1e-7), P)  # make P into a true projector (onto gauge space)

        # ALT WAY: use inverse of dG^T*dG to normalize projector (see wikipedia on projectors, dG => A)
        #  This *should* give the same thing as above, but numerical differences indicate the pinv method
        #  is prefereable (so long as rcond=1e-7 is ok in general...)
        #  Check: P'*P' = (dG (dGT dG)^1 dGT)(dG (dGT dG)^-1 dGT) = (dG (dGT dG)^1 dGT) = P'
        #invGG = _np.linalg.inv(_np.dot(_np.transpose(gen_ndG), gen_ndG))
        #Pp_alt = _np.dot(gen_ndG, _np.dot(invGG, _np.transpose(gen_ndG))) # a true projector (onto gauge space)
        #print "Pp - Pp_alt norm diff = ", _np.linalg.norm(Pp_alt - Pp)

        #OLD: ret = _np.identity(nParams,'d') - Pp
        # Check ranks to make sure everything is consistent.  If either of these assertions fail,
        #  then pinv's rcond or some other numerical tolerances probably need adjustment.
        #print "Rank P = ",_np.linalg.matrix_rank(P)
        #print "Rank Pp = ",_np.linalg.matrix_rank(Pp, P_RANK_TOL)
        #print "Rank 1-Pp = ",_np.linalg.matrix_rank(_np.identity(nParams,'d') - Pp, P_RANK_TOL)
        #print " Evals(1-Pp) = \n","\n".join([ "%d: %g" % (i,ev) \
        #       for i,ev in enumerate(_np.sort(_np.linalg.eigvals(_np.identity(nParams,'d') - Pp))) ])

        try:
            rank_P = _np.linalg.matrix_rank(P, P_RANK_TOL)  # original un-normalized projector

            # Note: use P_RANK_TOL here even though projector is *un-normalized* since sometimes P will
            #  have eigenvalues 1e-17 and one or two 1e-11 that should still be "zero" but aren't when
            #  no tolerance is given.  Perhaps a more custom tolerance based on the singular values of P
            #  but different from numpy's default tolerance would be appropriate here.

            assert(rank_P == _np.linalg.matrix_rank(Pp, P_RANK_TOL))  # rank shouldn't change with normalization
            #assert( (nParams - rank_P) == _np.linalg.matrix_rank(ret, P_RANK_TOL) ) # dimension of orthogonal space
        except(_np.linalg.LinAlgError):
            _warnings.warn("Linear algebra error (probably a non-convergent"
                           "SVD) ignored during matric rank checks in "
                           "Model.nongauge_projector(...) ")

        return Pp
