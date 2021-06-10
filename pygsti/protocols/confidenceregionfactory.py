"""
Classes for constructing confidence regions
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
import copy as _copy
import itertools as _itertools
import warnings as _warnings

import numpy as _np
import scipy.stats as _stats

from pygsti import optimize as _opt
from pygsti import tools as _tools
from pygsti.models.explicitcalc import P_RANK_TOL
from ..baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter


# NON-MARKOVIAN ERROR BARS
#Connection with Robin's notes:
#
# Robins notes: pg 21 : want to set radius delta'(alpha,r2)
#   via: lambda(G) = lambda(G_mle) + delta'
#
# Connecting with pyGSTi Hessian (H) calculations:
#   lambda(G) = 2(maxLogL - logL(G)) ~ chi2_k   (as defined in notes)
#   lambda(G_mle) = 2(maxLogL - logL(G_mle))
#
#  expand logL around max:
#    logL(G_mle + dx) ~= logL(G_mle) - 1/2 dx*H*dx (no first order term)
#
#  Thus, delta'
#    delta' = lambda(G) - lambda(G_mle) = -2(log(G)-log(G_mle))
#           = dx*H*dx ==> delta' is just like C1 or Ck scaling factors
#                         used for computing normal confidence regions
#    (recall delta' is computed as the alpha-th quantile of a
#     non-central chi^2_{K',r2} where K' = #of model params and
#     r2 = lambda(G_mle) - (K-K'), where K = #max-model params (~#circuits)
#     is the difference between the expected (mean) lambda (=K-K') and what
#     we actually observe (=lambda(G_mle)).
#


class ConfidenceRegionFactory(object):
    """
    An object which is capable of generating confidence intervals/regions.

    Often times, it does so by holding the Hessian of a fit function with
    respect to a `Model`'s parameters and related projections of it onto the
    non-gauge space.

    Alternative (non-Hessian-based) means of computing confidence intervals
    are also available, such as by using so-called "linear reponse error bars".

    Parameters
    ----------
    parent : Estimate
        the parent estimate object, needed to resolve model and gate
        string list labels.

    model_lbl : str
        The key into the parent `Estimate`'s `.models` dictionary that
        gives the `Model` about which confidence regions will be
        constructed.

    circuit_list_lbl : str
        The key into the parent `Results`'s `.circuit_lists` dictionary
        that specifies which circuits should be or were included
        when computing fit functions (the log-likelihood or chi2).

    hessian : numpy array, optional
        A pre-computed num_params x num_params Hessian matrix, where num_params is
        the number of dimensions of model space, i.e. model.num_params.

    non_mark_radius_sq : float, optional
        The non-Markovian radius associated with the goodness of fit found
        at the point where `hessian` was computed.  This must be specified
        whenver `hessian` is, and should be left as `None` when `hessian`
        is not specified.
    """

    def __init__(self, parent, model_lbl, circuit_list_lbl,
                 hessian=None, non_mark_radius_sq=None):
        """
        Initializes a new ConfidenceRegionFactory.

        Parameters
        ----------
        parent : Estimate
            the parent estimate object, needed to resolve model and gate
            string list labels.

        model_lbl : str
            The key into the parent `Estimate`'s `.models` dictionary that
            gives the `Model` about which confidence regions will be
            constructed.

        circuit_list_lbl : str
            The key into the parent `Results`'s `.circuit_lists` dictionary
            that specifies which circuits should be or were included
            when computing fit functions (the log-likelihood or chi2).

        hessian : numpy array, optional
            A pre-computed num_params x num_params Hessian matrix, where num_params is
            the number of dimensions of model space, i.e. model.num_params.

        non_mark_radius_sq : float, optional
            The non-Markovian radius associated with the goodness of fit found
            at the point where `hessian` was computed.  This must be specified
            whenver `hessian` is, and should be left as `None` when `hessian`
            is not specified.
        """

        #May be specified (together) whey hessian has already been computed
        assert(hessian is None or non_mark_radius_sq is not None), \
            "'non_mark_radius_sq' must be non-None when 'hessian' is specified"
        self.hessian = hessian
        self.nonMarkRadiusSq = non_mark_radius_sq

        self.hessian_projection_parameters = _collections.OrderedDict()
        self.inv_hessian_projections = _collections.OrderedDict()
        self.linresponse_gstfit_params = None
        self.nNonGaugeParams = self.nGaugeParams = None

        self.model_lbl = model_lbl
        self.circuit_list_lbl = circuit_list_lbl
        self.set_parent(parent)

    def __getstate__(self):
        # don't pickle parent (will create circular reference)
        to_pickle = self.__dict__.copy()
        del to_pickle['parent']

        # *don't* pickle any Comm objects
        if self.linresponse_gstfit_params and "resource_alloc" in self.linresponse_gstfit_params:
            to_pickle['linresponse_gstfit_params'] = self.linresponse_gstfit_params.copy()
            del to_pickle['linresponse_gstfit_params']['resource_alloc']  # one *cannot* pickle Comm objects

        return to_pickle

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
        self.parent = None  # initialize to None upon unpickling

    def set_parent(self, parent):
        """
        Sets the parent Estimate object of this ConfidenceRegionFactory.

        This function is usually only needed internally to re-link a
        ConfidenceRegionFactory with its parent after be un-serialized
        from disk.

        Parameters
        ----------
        parent : Estimate
            The parent of this object.

        Returns
        -------
        None
        """
        self.parent = parent

    @property
    def has_hessian(self):
        """
        Returns whether or not the Hessian has already been computed.

        When True, :func:`project_hessian` can be used to project the
        Hessian for use in creating confidence intervals.  When False,
        either :func:`compute_hessian` can be called to compute the
        Hessian or slower methods must be used to estimate the necessary
        portion of the Hessian.  The result of this function is often used
        to decide whether or not to proceed with an error-bar computation.

        Returns
        -------
        bool
        """
        #return bool(self.invRegionQuadcForm is not None)
        return bool(self.hessian is not None)

    def can_construct_views(self):
        """
        Checks whether this factory has enough information to construct 'views' of itself.

        `ConfidenceRegionFactoryView` view objects are created using the
        :method:`view` method, which can in turn be used to construct
        confidence intervals.

        Returns
        -------
        bool
        """
        try:
            self.view(95)  # will raise assertion errors
            return True
        except:
            return False

    @property
    def model(self):
        """
        Retrieve the associated model.

        Returns
        -------
        Model
            the model marking the center location of this confidence region.
        """
        assert(self.parent is not None)  # Estimate
        return self.parent.models[self.model_lbl]

    def compute_hessian(self, comm=None, mem_limit=None, approximate=False):
        """
        Computes the Hessian for this factory.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.

        mem_limit : int, optional
            A rough memory limit in bytes which restricts the amount of intermediate
            values that are computed and stored.

        approximate : bool, optional
            Whether to compute the true Hessian or just an approximation of it.
            See :function:`logl_approximate_hessian`.  Setting to True can
            significantly reduce the run time.

        Returns
        -------
        numpy.ndarray
            The Hessian matrix (also stored internally)
        """
        assert(self.parent is not None)  # Estimate
        assert(self.parent.parent is not None)  # Results

        model = self.parent.models[self.model_lbl]
        circuit_list = self.parent.parent.circuit_lists[self.circuit_list_lbl]
        dataset = self.parent.parent.dataset

        #extract any parameters we can get from the Estimate
        parameters = self.parent.parameters
        obj = parameters.get('objective', 'logl')
        minProbClip = parameters.get('minProbClip', 1e-4)
        minProbClipForWeighting = parameters.get('minProbClipForWeighting', 1e-4)
        probClipInterval = parameters.get('probClipInterval', (-1e6, 1e6))
        radius = parameters.get('radius', 1e-4)
        cptp_penalty_factor = parameters.get('cptpPenaltyFactor', 0)
        spam_penalty_factor = parameters.get('spamPenaltyFactor', 0)
        useFreqWt = parameters.get('useFreqWeightedChiSq', False)
        aliases = parameters.get('opLabelAliases', None)
        if mem_limit is None:
            mem_limit = parameters.get('mem_limit', None)

        vb = 3 if mem_limit else 0  # only show details of hessian comp when there's a mem limit (a heuristic)

        assert(cptp_penalty_factor == 0), 'cptp_penalty_factor unsupported in hessian computation'
        assert(spam_penalty_factor == 0), 'spam_penalty_factor unsupported in hessian computation'
        assert(useFreqWt is False), 'useFreqWeightedChiSq unsupported in hessian computation'

        #Expand operation label aliases used in DataSet lookups
        ds_circuit_list = _tools.apply_aliases_to_circuits(circuit_list, aliases)

        nModelParams = model.num_nongauge_params
        nDataParams = dataset.degrees_of_freedom(ds_circuit_list)
        #number of independent parameters in dataset (max. model # of params)

        MIN_NON_MARK_RADIUS = 1e-8  # must be >= 0

        if obj == 'logl':
            hessian_fn = _tools.logl_approximate_hessian if approximate \
                else _tools.logl_hessian
            hessian = hessian_fn(model, dataset, circuit_list,
                                 minProbClip, probClipInterval, radius,
                                 comm=comm, mem_limit=mem_limit, verbosity=vb,
                                 op_label_aliases=aliases)

            nonMarkRadiusSq = max(2 * (_tools.logl_max(model, dataset)
                                       - _tools.logl(model, dataset,
                                                     op_label_aliases=aliases))
                                  - (nDataParams - nModelParams), MIN_NON_MARK_RADIUS)

        elif obj == 'chi2':
            chi2, hessian = [f(model, dataset, circuit_list,
                               minProbClipForWeighting,
                               probClipInterval, mem_limit=mem_limit,
                               op_label_aliases=aliases) for f in (_tools.chi2, _tools.chi2_hessian)]

            nonMarkRadiusSq = max(chi2 - (nDataParams - nModelParams), MIN_NON_MARK_RADIUS)
        else:
            raise ValueError("Invalid objective '%s'" % obj)

        self.hessian = hessian
        self.nonMarkRadiusSq = nonMarkRadiusSq
        return hessian

    def project_hessian(self, projection_type, label=None, tol=1e-7, maxiter=10000):
        """
        Projects the Hessian onto the non-gauge space.

        This is a necessary step before confidence regions/intervals can be
        computed via Hessian-based methods.

        Parameters
        ----------
        projection_type : string
            Specifies how (and whether) to project the given hessian matrix
            onto a non-gauge space.  Allowed values are:

            - 'std' -- standard projection onto the space perpendicular to the
              gauge space.
            - 'none' -- no projection is performed.  Useful if the supplied
              hessian has already been projected.
            - 'optimal gate CIs' -- a lengthier projection process in which a
              numerical optimization is performed to find the non-gauge space
              which minimizes the (average) size of the confidence intervals
              corresponding to gate (as opposed to SPAM vector) parameters.
            - 'intrinsic error' -- compute separately the intrinsic error
              in the gate and spam Model parameters and set weighting metric
              based on their ratio.

        label : str, optional
            The internal label to use for this projection.  If None, then
            `projection_type` is used, which is usually fine.

        tol : float, optional
            Tolerance for optimal Hessian projection.  Only used when
            `projection_type == 'optimal gate CIs'`.

        maxiter : int, optional
            Maximum iterations for optimal Hessian projection.  Only used when
            `projection_type == 'optimal gate CIs'`.

        Returns
        -------
        numpy.ndarray
            The *inverse* of the projected Hessian matrix (also stored internally)
        """
        assert(self.hessian is not None), "No hessian! Compute it using 'compute_hessian'"

        if label is None:
            label = projection_type

        model = self.parent.models[self.model_lbl]
        proj_non_gauge = model.compute_nongauge_projector()
        self.nNonGaugeParams = _np.linalg.matrix_rank(proj_non_gauge, P_RANK_TOL)
        self.nGaugeParams = model.num_params - self.nNonGaugeParams

        #Project Hessian onto non-gauge space
        if projection_type == 'none':
            projected_hessian = self.hessian
        elif projection_type == 'std':
            projected_hessian = _np.dot(proj_non_gauge, _np.dot(self.hessian, proj_non_gauge))
        elif projection_type == 'optimal gate CIs':
            projected_hessian = self._opt_projection_for_operation_cis("L-BFGS-B", maxiter, maxiter,
                                                                       tol, verbosity=3)  # verbosity for DEBUG
        elif projection_type == 'intrinsic error':
            projected_hessian = self._opt_projection_from_split(verbosity=3)  # verbosity for DEBUG
        else:
            raise ValueError("Invalid value of projection_type argument: %s" % projection_type)

        #Invert *non-gauge-part* of quadratic 'projected_hessian' by eigen-decomposing ->
        #   inverting the non-gauge eigenvalues -> re-constructing via eigenvectors.
        # (Note that Hessian & quadc form mxs are symmetric so eigenvalues == singular values)
        # regionQuadcForm = U * diag(evals) * U.dag (b/c U.dag == inv(U) )
        evals, U = _np.linalg.eigh(projected_hessian)
        Udag = _np.conjugate(_np.transpose(U))

        #invert only the non-gauge eigenvalues (those with ordering index >= n_gauge_params)
        orderInds = [el[0] for el in sorted(enumerate(evals), key=lambda x: abs(x[1]))
                     ]  # ordering index of each eigenvalue
        invEvals = _np.zeros(evals.shape, evals.dtype)
        for i in orderInds[self.nGaugeParams:]:
            invEvals[i] = 1.0 / evals[i]

        #re-construct "inverted" quadratic form
        inv_projected_hessian = _np.diag(invEvals)
        inv_projected_hessian = _np.dot(U, _np.dot(inv_projected_hessian, Udag))

        #save input args for copying object
        self.inv_hessian_projections[label] = inv_projected_hessian
        self.hessian_projection_parameters[label] = {
            'projection_type': projection_type,
            'tol': tol,
            'maxiter': maxiter
        }
        return inv_projected_hessian

    def enable_linear_response_errorbars(self, resource_alloc=None):
        """
        Stores the parameters needed to compute (on-demand) linear response error bars.

        In particular, this function sets up the parameters needed to perform the
        model optimizations needed to compute error bars on quantities.

        'linear response' mode obtains elements of the Hessian via the
        linear response of a "forcing term".  This requres a likelihood
        optimization for *every* computed error bar, but avoids pre-
        computation of the entire Hessian matrix, which can be
        prohibitively costly on large parameter spaces.

        Parameters
        ----------
        resoure_alloc : ResourceAllocation
            Allocation for running linear-response GST fits.

        Returns
        -------
        None
        """
        assert(self.parent is not None)  # Estimate
        assert(self.parent.parent is not None)  # Results

        circuits = self.parent.parent.circuit_lists[self.circuit_list_lbl]
        dataset = self.parent.parent.dataset

        parameters = self.parent.parameters
        builder = parameters['final_objfn_builder']
        opt = {'maxiter': 100, 'tol': 1e-10}  # don't let this run for too long

        self.linresponse_gstfit_params = {
            'dataset': dataset,
            'circuits': circuits,
            'optimizer': opt,
            'objective_function_builder': _copy.deepcopy(builder),
            'resource_alloc': resource_alloc,
        }

        #Count everything as non-gauge? TODO BETTER
        self.nNonGaugeParams = self.model.num_params
        self.nGaugeParams = 0

    def view(self, confidence_level, region_type='normal',
             hessian_projection_label=None):
        """
        Constructs a "view" of this ConfidenceRegionFactory for a particular type and confidence level.

        The returned view object can then be used to construct confidence intervals/regions.

        Parameters
        ----------
        confidence_level : float
            The confidence level as a percentage, i.e. between 0 and 100.

        region_type : {'normal', 'non-markovian'}
            The type of confidence regions.  `'normal'` constructs standard
            intervals based on the inverted Hessian matrix or linear-response
            optimizations.  `'non-markovian'` attempts to enlarge the intervals
            to account for the badness-of-fit at the current location.

        hessian_projection_label : str, optional
            A label specifying which Hessian projection to use (only useful
            when there are multiple).  These labels are either the
            `projection_type` values of :func:`project_hessian` or the
            custom `label` argument provided to that function.  If None,
            then the most recent (perhaps the only) projection is used.

        Returns
        -------
        ConfidenceRegionFactoryView
        """
        inv_hessian_projection = None
        linresponse_gstfit_params = None

        assert(self.parent is not None)  # Estimate
        model = self.parent.models[self.model_lbl]

        if self.hessian is not None:
            assert(len(self.inv_hessian_projections) > 0), \
                "No Hessian projections!  Use 'project_hessian' to create at least one."
            if hessian_projection_label is None:
                hessian_projection_label = list(self.inv_hessian_projections.keys())[-1]
            assert(hessian_projection_label in self.inv_hessian_projections.keys()), \
                "Hessian projection '%s' does not exist!" % hessian_projection_label
            inv_hessian_projection = self.inv_hessian_projections[hessian_projection_label]
        else:
            assert(self.linresponse_gstfit_params is not None), \
                "Must either compute & project a Hessian matrix or enable linear response parameters"
            assert(hessian_projection_label is None), \
                "Must set `hessian_projection_label` to None when using linear-response error bars"
            linresponse_gstfit_params = self.linresponse_gstfit_params

        #Compute the non-Markovian "radius" if required
        if region_type == "normal":
            nonMarkRadiusSq = 0.0
        elif region_type == "non-markovian":
            nonMarkRadiusSq = self.nonMarkRadiusSq
        else:
            raise ValueError("Invalid confidence region type: %s" % region_type)

        return ConfidenceRegionFactoryView(model, inv_hessian_projection, linresponse_gstfit_params,
                                           confidence_level, nonMarkRadiusSq,
                                           self.nNonGaugeParams, self.nGaugeParams)

        #TODO: where to move this?
        ##Check that number of gauge parameters reported by model is consistent with confidence region
        ## since the parameter number computed this way is used in chi2 or logl progress tables
        #Np_check =  model.num_nongauge_params
        #if(Np_check != cri.nNonGaugeParams):
        #    _warnings.warn("Number of non-gauge parameters in model and confidence region do "
        #                   + " not match.  This indicates an internal logic error.")

    def _opt_projection_for_operation_cis(self, method="L-BFGS-B", maxiter=10000,
                                          maxfev=10000, tol=1e-6, verbosity=0):
        printer = _VerbosityPrinter.create_printer(verbosity)
        model = self.parent.models[self.model_lbl]
        base_hessian = self.hessian
        level = 95  # or 50, or whatever - the scale factory doesn't matter for the optimization

        printer.log('', 3)
        printer.log("--- Hessian Projector Optimization for gate CIs (%s) ---" % method, 2, indent_offset=-1)

        def _objective_func(vector_m):
            matM = vector_m.reshape((self.nNonGaugeParams, self.nGaugeParams))
            proj_extra = model.compute_nongauge_projector(non_gauge_mix_mx=matM)
            projected_hessian_ex = _np.dot(proj_extra, _np.dot(base_hessian, proj_extra))

            sub_crf = ConfidenceRegionFactory(self.parent, self.model_lbl, self.circuit_list_lbl,
                                              projected_hessian_ex, 0.0)
            sub_crf.project_hessian('none')
            crfv = sub_crf.view(level)

            operationCIs = _np.concatenate([crfv.retrieve_profile_likelihood_confidence_intervals(gl).flatten()
                                            for gl in model.operations])
            return _np.sqrt(_np.sum(operationCIs**2))

        #Run Minimization Algorithm
        startM = _np.zeros((self.nNonGaugeParams, self.nGaugeParams), 'd')
        x0 = startM.flatten()
        print_obj_func = _opt.create_objfn_printer(_objective_func)
        minSol = _opt.minimize(_objective_func, x0,
                               method=method, maxiter=maxiter,
                               maxfev=maxfev, tol=tol,
                               callback=print_obj_func if verbosity > 2 else None)

        mixMx = minSol.x.reshape((self.nNonGaugeParams, self.nGaugeParams))
        proj_extra = model.compute_nongauge_projector(non_gauge_mix_mx=mixMx)
        projected_hessian_ex = _np.dot(proj_extra, _np.dot(base_hessian, proj_extra))

        printer.log('The resulting min sqrt(sum(operationCIs**2)): %g' % minSol.fun, 2)
        return projected_hessian_ex

    def _opt_projection_from_split(self, verbosity=0):
        printer = _VerbosityPrinter.create_printer(verbosity)
        model = self.parent.models[self.model_lbl]
        base_hessian = self.hessian
        level = 95  # or 50, or whatever - the scale factory doesn't matter for the optimization

        printer.log('', 3)
        printer.log("--- Hessian Projector Optimization from separate SPAM and Gate weighting ---", 2, indent_offset=-1)

        #get gate-intrinsic-error
        proj = model.compute_nongauge_projector(item_weights={'gates': 1.0, 'spam': 0.0})
        projected_hessian = _np.dot(proj, _np.dot(base_hessian, proj))
        sub_crf = ConfidenceRegionFactory(self.parent, self.model_lbl,
                                          self.circuit_list_lbl, projected_hessian, 0.0)
        sub_crf.project_hessian('none')
        crfv = sub_crf.view(level)
        operationCIs = _np.concatenate([crfv.retrieve_profile_likelihood_confidence_intervals(gl).flatten()
                                        for gl in model.operations])
        op_intrinsic_err = _np.sqrt(_np.mean(operationCIs**2))

        #get spam-intrinsic-error
        proj = model.compute_nongauge_projector(item_weights={'gates': 0.0, 'spam': 1.0})
        projected_hessian = _np.dot(proj, _np.dot(base_hessian, proj))
        sub_crf = ConfidenceRegionFactory(self.parent, self.model_lbl,
                                          self.circuit_list_lbl, projected_hessian, 0.0)
        sub_crf.project_hessian('none')
        crfv = sub_crf.view(level)
        spamCIs = _np.concatenate([crfv.retrieve_profile_likelihood_confidence_intervals(sl).flatten()
                                   for sl in _itertools.chain(iter(model.preps),
                                                              iter(model.povms))])
        spam_intrinsic_err = _np.sqrt(_np.mean(spamCIs**2))

        ratio = op_intrinsic_err / spam_intrinsic_err
        proj = model.compute_nongauge_projector(item_weights={'gates': 1.0, 'spam': ratio})
        projected_hessian = _np.dot(proj, _np.dot(base_hessian, proj))

        if printer.verbosity >= 2:
            #Create crfv here just to extract #'s for print stmts
            sub_crf = ConfidenceRegionFactory(self.parent, self.model_lbl,
                                              self.circuit_list_lbl, projected_hessian, 0.0)
            sub_crf.project_hessian('none')
            crfv = sub_crf.view(level)

            operationCIs = _np.concatenate([crfv.retrieve_profile_likelihood_confidence_intervals(gl).flatten()
                                            for gl in model.operations])
            spamCIs = _np.concatenate([crfv.retrieve_profile_likelihood_confidence_intervals(sl).flatten()
                                       for sl in _itertools.chain(iter(model.preps),
                                                                  iter(model.povms))])
            op_err = _np.sqrt(_np.mean(operationCIs**2))
            spam_err = _np.sqrt(_np.mean(spamCIs**2))
            printer.log('Resulting intrinsic errors: %g (gates), %g (spam)' %
                        (op_intrinsic_err, spam_intrinsic_err), 2)
            printer.log('Resulting sqrt(mean(operationCIs**2)): %g' % op_err, 2)
            printer.log('Resulting sqrt(mean(spamCIs**2)): %g' % spam_err, 2)

        return projected_hessian


class ConfidenceRegionFactoryView(object):
    """
    Encapsulates a lightweight "view" of a ConfidenceRegionFactory.

    A view object is principally defined by it's having a fixed confidence-level.
    Thus, a "view" is like a factory that generates confidence intervals for
    just a single confidence level.  As such, it is a useful object to pass
    around to routines which compute and display error bars, as these routines
    typically don't depend on what confidence-level is being used.

    Parameters
    ----------
    model : Model
        The model at the center of this confidence region.

    inv_projected_hessian : numpy.ndarray
        The computed inverse of the non-gauge-projected Hessian.

    mlgst_params : dict
        A dictionary of ML-GST parameters only used for linear-response
        error bars.

    confidence_level : float
        the confidence level (between 0 and 100) used in
        the computation of confidence regions/intervals.

    non_mark_radius_sq : float, optional
        When non-zero, "a non-Markovian error region" is constructed using
        this value as the squared "non-markovian radius". This specifies the
        portion of 2*(max-log-likelihood - model-log-likelihood) that we
        attribute to non-Markovian errors (typically the previous
        difference minus it's expected value, the difference in number of
        parameters between the maximal and model models).  If set to
        zero (the default), a standard and thereby statistically rigorous
        conficence region is created.  Non-zero values should only be
        supplied if you really know what you're doing.

    n_non_gauge_params : int
        The numbers of non-gauge parameters.  This could be computed from `model`
        but can be passed in to save compuational time.

    n_gauge_params : int
        The numbers of gauge parameters.  This could be computed from `model`
        but can be passed in to save compuational time.
    """

    def __init__(self, model, inv_projected_hessian, mlgst_params, confidence_level,
                 non_mark_radius_sq, n_non_gauge_params, n_gauge_params):
        """
        Creates a new ConfidenceRegionFactoryView.

        Usually this constructor is not called directly, and objects of
        this type are obtained by calling the :method:`view` method of
        a `ConfidenceRegionFactory` object.

        Parameters
        ----------
        model : Model
            The model at the center of this confidence region.

        inv_projected_hessian : numpy.ndarray
            The computed inverse of the non-gauge-projected Hessian.

        mlgst_params : dict
            A dictionary of ML-GST parameters only used for linear-response
            error bars.

        confidence_level : float
            the confidence level (between 0 and 100) used in
            the computation of confidence regions/intervals.

        non_mark_radius_sq : float, optional
            When non-zero, "a non-Markovian error region" is constructed using
            this value as the squared "non-markovian radius". This specifies the
            portion of 2*(max-log-likelihood - model-log-likelihood) that we
            attribute to non-Markovian errors (typically the previous
            difference minus it's expected value, the difference in number of
            parameters between the maximal and model models).  If set to
            zero (the default), a standard and thereby statistically rigorous
            conficence region is created.  Non-zero values should only be
            supplied if you really know what you're doing.

        n_non_gauge_params, n_gauge_params : int
            The numbers of non-gauge and gauge parameters, respectively.  These could be
            computed from `model` but they're passed in to save compuational time.
        """

        # Scale projected Hessian for desired confidence level => quadratic form for confidence region assume hessian
        # gives Fisher info, so asymptotically normal => confidence interval = +/- seScaleFctr * 1/sqrt(hessian) where
        # seScaleFctr gives the scaling factor for a normal distribution, i.e. integrating the std normal distribution
        # between -seScaleFctr and seScaleFctr == confidence_level/100 (as a percentage)
        assert(confidence_level > 0.0 and confidence_level < 100.0)
        if confidence_level < 1.0:
            _warnings.warn("You've specified a %f%% confidence interval, " % confidence_level
                           + "which is usually small.  Be sure to specify this"
                           + "number as a percentage in (0,100) and not a fraction in (0,1).")

        # Get constants C such that xT*Hessian*x = C gives contour for the desired confidence region.
        #  C1 == Single DOF case: constant for a single-DOF likelihood, (or a profile likelihood in our case)
        #  Ck == Total DOF case: constant for a region of the likelihood as a function of *all non-gauge* model
        #        parameters
        self.nonMarkRadiusSq = non_mark_radius_sq
        if non_mark_radius_sq == 0.0:  # use == to test for *exact* zero floating pt value as herald
            C1 = _stats.chi2.ppf(confidence_level / 100.0, 1)
            Ck = _stats.chi2.ppf(confidence_level / 100.0, n_non_gauge_params)

            # Alt. method to get C1: square the result of a single gaussian (normal distribution)
            #Note: scipy's ppf gives inverse of cdf, so want to know where cdf == the leftover probability on left side
            # std error scaling factor for desired confidence region
            seScaleFctr = -_stats.norm.ppf((1.0 - confidence_level / 100.0) / 2.0)
            assert(_np.isclose(C1, seScaleFctr**2))

            # save quadratic form Q s.t. xT*Q*x = 1 gives confidence region using C1, i.e. a
            #  region appropriate for generating 1-D confidence intervals.
            if inv_projected_hessian is not None:
                self.invRegionQuadcForm = inv_projected_hessian * C1
            else:
                self.invRegionQuadcForm = None

            self.intervalScaling = _np.sqrt(Ck / C1)  # multiplicative scaling required to convert intervals
            # to those obtained using a full (using Ck) confidence region.
            self.stdIntervalScaling = 1.0  # multiplicative scaling required to convert intervals
            # to *standard* (e.g. not non-Mark.) intervals.
            self.stdRegionScaling = self.intervalScaling  # multiplicative scaling required to convert intervals
            # to those obtained using a full *standard* confidence region.

        else:
            C1 = _stats.ncx2.ppf(confidence_level / 100.0, 1, non_mark_radius_sq)
            Ck = _stats.ncx2.ppf(confidence_level / 100.0, n_non_gauge_params, non_mark_radius_sq)

            # save quadratic form Q s.t. xT*Q*x = 1 gives confidence region using C1, i.e. a
            #  region appropriate for generating 1-D confidence intervals.
            if inv_projected_hessian is not None:
                self.invRegionQuadcForm = inv_projected_hessian * C1
                self.invRegionQuadcForm /= _np.sqrt(n_non_gauge_params)  # make a *worst case* non-mark. region...
            else:
                self.invRegionQuadcForm = None

            self.intervalScaling = _np.sqrt(Ck / C1)  # multiplicative scaling required to convert intervals
            # to those obtained using a full (using Ck) confidence region.

            stdC1 = _stats.chi2.ppf(confidence_level / 100.0, 1)
            stdCk = _stats.chi2.ppf(confidence_level / 100.0, n_non_gauge_params)
            self.stdIntervalScaling = _np.sqrt(stdC1 / C1)  # see above description
            self.stdRegionScaling = _np.sqrt(stdCk / C1)  # see above description

            _warnings.warn("Non-Markovian error bars are experimental and"
                           " cannot be interpreted as standard error bars."
                           " Proceed with caution!")

        #Store list of profile-likelihood confidence intervals
        #  which == sqrt(diagonal els) of invRegionQuadcForm
        if self.invRegionQuadcForm is not None:
            dim = self.invRegionQuadcForm.shape[0]
            self.profLCI = [_np.sqrt(abs(self.invRegionQuadcForm[k, k])) for k in range(dim)]
            self.profLCI = _np.array(self.profLCI, 'd')
        else:
            self.profLCI = None

        self.model = model
        self.level = confidence_level  # a percentage, i.e. btwn 0 and 100
        self.nNonGaugeParams = n_non_gauge_params
        self.nGaugeParams = n_gauge_params

        self.mlgst_params = mlgst_params
        self._C1 = C1  # save for linear response scaling
        self.mlgst_evaltree_cache = {}  # for _do_mlgst_base speedup

    def __getstate__(self):
        # *don't* pickle any Comm objects
        to_pickle = self.__dict__.copy()
        if self.mlgst_params and "comm" in self.mlgst_params:
            del self.mlgst_params['comm']  # one *cannot* pickle Comm objects
        return to_pickle

    @property
    def errorbar_type(self):
        """
        Return the type of error bars this view will generate, either "standard" or "non-markovian".

        Returns
        -------
        str
        """
        if self.nonMarkRadiusSq > 0:
            return "non-markovian"
        else:
            return "standard"

    def retrieve_profile_likelihood_confidence_intervals(self, label=None, component_label=None):
        """
        Retrieve the profile-likelihood confidence intervals for a specified model object (or all such intervals).

        Parameters
        ----------
        label : Label, optional
            If not None, can be either a gate or SPAM vector label
            to specify the confidence intervals corresponding to gate, rhoVec,
            or EVec parameters respectively.  If None, then intervals corresponding
            to all of the model's parameters are returned.

        component_label : Label, optional
            Labels an effect within a POVM or a member within an instrument.

        Returns
        -------
        numpy array
            One-dimensional array of (positive) interval half-widths which specify
            a symmetric confidence interval.
        """
        if self.profLCI is None:
            raise NotImplementedError("Profile-likelihood confidence intervals"
                                      "are not implemented for this type of confidence region")
        if label is None:
            return self.profLCI

        elif label in self.model.operations:
            return self.profLCI[self.model.operations[label].gpindices]

        elif label in self.model.preps:
            return self.profLCI[self.model.preps[label].gpindices]

        elif label in self.model.povms:
            if component_label is not None:
                return self.profLCI[self.model.povms[label][component_label].gpindices]
            return self.profLCI[self.model.povms[label].gpindices]

        elif label in self.model.instruments:
            if component_label is not None:
                return self.profLCI[self.model.instruments[label][component_label].gpindices]
            return self.profLCI[self.model.instruments[label].gpindices]

        else:
            raise ValueError(("Invalid item label (%s) for computing" % label)
                             + "profile likelihood confidence intervals")

    def compute_confidence_interval(self, fn_obj, eps=1e-7,
                                    return_fn_val=False, verbosity=0):
        """
        Compute the confidence interval for an arbitrary function.

        This "function", however, must be encapsulated as a
        `ModelFunction` object, which allows it to neatly specify
        what its dependencies are and allows it to compaute finite-
        different derivatives more efficiently.

        Parameters
        ----------
        fn_obj : ModelFunction
            An object representing the function to evaluate. The
            returned confidence interval is based on linearizing this function
            and propagating the model-space confidence region.

        eps : float, optional
            Step size used when taking finite-difference derivatives of fnOfOp.

        return_fn_val : bool, optional
            If True, return the value of fnOfOp along with it's confidence
            region half-widths.

        verbosity : int, optional
            Specifies level of detail in standard output.

        Returns
        -------
        df : float or numpy array
            Half-widths of confidence intervals for each of the elements
            in the float or array returned by fnOfOp.  Thus, shape of
            df matches that returned by fnOfOp.
        f0 : float or numpy array
            Only returned when return_fn_val == True. Value of fnOfOp
            at the gate specified by op_label.
        """

        nParams = self.model.num_params
        f0 = fn_obj.evaluate(self.model)  # function value at "base point"

        #Get finite difference derivative gradF that is shape (nParams, <shape of f0>)
        gradF = _create_empty_grad_f(f0, nParams)

        fn_dependencies = fn_obj.get_dependencies()
        if 'all' in fn_dependencies:
            fn_dependencies = ['all']  # no need to do anything else
        if 'spam' in fn_dependencies:
            fn_dependencies = [("prep", l) for l in self.model.preps.keys()] + \
                              [("povm", l) for l in self.model.povms.keys()]

        #elements of fn_dependencies are either 'all', 'spam', or
        # the "type:label" of a specific gate or spam vector.
        all_gpindices = []
        for dependency in fn_dependencies:
            mdl = self.model.copy()  # copy that will contain the "+eps" model

            if dependency == 'all':
                all_gpindices.extend(range(mdl.num_params))
            else:
                # copy objects because we add eps to them below
                typ, lbl = dependency
                if typ == "gate": modelObj = mdl.operations[lbl]
                elif typ == "prep": modelObj = mdl.preps[lbl]
                elif typ == "povm": modelObj = mdl.povms[lbl]
                elif typ == "instrument": modelObj = mdl.instruments[lbl]
                else: raise ValueError("Invalid dependency type: %s" % typ)
                all_gpindices.extend(modelObj.gpindices_as_array())

        vec0 = mdl.to_vector()
        all_gpindices = sorted(list(set(all_gpindices)))  # remove duplicates

        for igp in all_gpindices:  # iterate over "global" Model-parameter indices
            vec = vec0.copy(); vec[igp] += eps
            mdl.from_vector(vec)
            mdl.basis = self.model.basis  # we're still in the same basis (maybe needed by fn_obj)

            f = fn_obj.evaluate_nearby(mdl)
            if isinstance(f0, dict):  # special behavior for dict: process each item separately
                for ky in gradF:
                    gradF[ky][igp] = (f[ky] - f0[ky]) / eps
            else:
                assert(_np.linalg.norm(_np.imag(f - f0)) < 1e-12 or _np.iscomplexobj(gradF)
                       ), "gradF seems to be the wrong type!"
                gradF[igp] = _np.real_if_close(f - f0) / eps

        return self._compute_return_from_grad_f(gradF, f0, return_fn_val, verbosity)

    def _compute_return_from_grad_f(self, grad_f, f0, return_fn_val, verbosity):
        """ Just adds logic for special behavior when f0 is a dict """
        if isinstance(f0, dict):
            df_dict = {ky: self._compute_df_from_grad_f(
                grad_f[ky], f0[ky], False, verbosity)
                for ky in grad_f}
            return (df_dict, f0) if return_fn_val else df_dict
        else:
            return self._compute_df_from_grad_f(grad_f, f0, return_fn_val, verbosity)

    def _compute_df_from_grad_f(self, grad_f, f0, return_fn_val, verbosity):
        if self.invRegionQuadcForm is None:
            df = self._compute_df_from_grad_f_linresponse(
                grad_f, f0, verbosity)
        else:
            df = self._compute_df_from_grad_f_hessian(
                grad_f, f0, verbosity)
        return (df, f0) if return_fn_val else df

    def _compute_df_from_grad_f_linresponse(self, grad_f, f0, verbosity):
        from .. import algorithms as _alg
        assert(self.mlgst_params is not None)

        if isinstance(f0, complex) or (hasattr(f0, 'dtype') and f0.dtype == _np.dtype("complex")):
            raise NotImplementedError("Can't handle complex-valued functions yet")

        if hasattr(f0, 'shape') and len(f0.shape) > 2:
            raise ValueError("Unsupported number of dimensions returned by fnOfOp or fnOfModel: %d" % len(f0.shape))
            #May not be needed here, but gives uniformity with Hessian case

        #massage grad_f, which has shape (num_params,) + f0.shape
        # to that expected by _do_mlgst_base, which is
        # (flat_f0_size, num_params)
        if len(grad_f.shape) == 1:
            grad_f.shape = (1, grad_f.shape[0])
        else:
            flatDim = _np.prod(f0.shape)
            grad_f.shape = (grad_f.shape[0], flatDim)
            grad_f = _np.transpose(grad_f)  # now shape == (flatDim, num_params)
        assert(len(grad_f.shape) == 2)

        mlgst_args = self.mlgst_params.copy()
        mlgst_args['start_model'] = self.model

        penalties = mlgst_args['objective_function_builder'].penalties or {}
        penalties.update({'forcefn_grad': grad_f, 'shift_fctr': 100.0})
        mlgst_args['objective_function_builder'].penalties = penalties

        _, bestGS = _alg.core.run_gst_fit_simple(**mlgst_args)
        bestGS = _alg.gaugeopt_to_target(bestGS, self.model)  # maybe more params here?
        norms = _np.array([_np.dot(grad_f[i], grad_f[i]) for i in range(grad_f.shape[0])])
        delta2 = _np.abs(_np.dot(grad_f, bestGS.to_vector() - self.model.to_vector())
                         * _np.where(norms > 1e-10, 1.0 / norms, 0.0))
        delta2 *= self._C1  # scaling appropriate for confidence level
        delta = _np.sqrt(delta2)  # error^2 -> error

        if hasattr(f0, 'shape'):
            delta.shape = f0.shape  # reshape to un-flattened
        else:
            assert(isinstance(f0, float))
            delta = float(delta)

        return delta

    def _compute_df_from_grad_f_hessian(self, grad_f, f0, verbosity):
        """
        Internal function which computes error bars given an function value
        and gradient (using linear approx. to function)
        """

        #Compute df = sqrt( gradFu.dag * 1/D * gradFu )
        #  where regionQuadcForm = U * D * U.dag and gradFu = U.dag * grad_f
        #  so df = sqrt( grad_f.dag * U * 1/D * U.dag * grad_f )
        #        = sqrt( grad_f.dag * invRegionQuadcForm * grad_f )

        printer = _VerbosityPrinter.create_printer(verbosity)

        printer.log("grad_f = %s" % grad_f)

        if isinstance(f0, float) or isinstance(f0, int):
            gradFdag = _np.conjugate(_np.transpose(grad_f))

            #DEBUG
            #arg = _np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, grad_f))
            #print "HERE: taking sqrt(abs(%s))" % arg

            df = _np.sqrt(abs(_np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, grad_f))))
        elif isinstance(f0, complex):
            gradFdag = _np.transpose(grad_f)  # conjugate?
            df = _np.sqrt(abs(_np.dot(gradFdag.real, _np.dot(self.invRegionQuadcForm, grad_f.real)))) \
                + 1j * _np.sqrt(abs(_np.dot(gradFdag.imag, _np.dot(self.invRegionQuadcForm, grad_f.imag))))
        else:
            fDims = len(f0.shape)
            grad_f = _np.rollaxis(grad_f, 0, 1 + fDims)  # roll parameter axis to be the last index, preceded by f-shape
            df = _np.empty(f0.shape, f0.dtype)

            if f0.dtype == _np.dtype("complex"):  # real and imaginary parts separately
                if fDims == 0:  # same as float case above
                    gradFdag = _np.transpose(grad_f)  # conjugate?
                    df = _np.sqrt(abs(_np.dot(gradFdag.real, _np.dot(self.invRegionQuadcForm, grad_f.real)))) \
                        + 1j * _np.sqrt(abs(_np.dot(gradFdag.imag, _np.dot(self.invRegionQuadcForm, grad_f.imag))))
                elif fDims == 1:
                    for i in range(f0.shape[0]):
                        gradFdag = _np.transpose(grad_f[i])  # conjugate?
                        df[i] = (_np.sqrt(abs(_np.dot(gradFdag.real, _np.dot(self.invRegionQuadcForm, grad_f[i].real))))
                                 + 1j * _np.sqrt(abs(_np.dot(gradFdag.imag,
                                                             _np.dot(self.invRegionQuadcForm, grad_f[i].imag)))))
                elif fDims == 2:
                    for i in range(f0.shape[0]):
                        for j in range(f0.shape[1]):
                            gradFdag = _np.transpose(grad_f[i, j])  # conjugate?
                            df[i, j] = _np.sqrt(abs(_np.dot(
                                gradFdag.real,
                                _np.dot(self.invRegionQuadcForm, grad_f[i, j].real)))) \
                                + 1j * \
                                _np.sqrt(abs(_np.dot(gradFdag.imag, _np.dot(
                                    self.invRegionQuadcForm, grad_f[i, j].imag))))
                else:
                    raise ValueError("Unsupported number of dimensions returned by fnOfOp or fnOfModel: %d" % fDims)

            else:  # assume real -- so really don't need conjugate calls below
                if fDims == 0:  # same as float case above
                    gradFdag = _np.conjugate(_np.transpose(grad_f))

                    #DEBUG
                    #arg = _np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, grad_f))
                    #print "HERE2: taking sqrt(abs(%s))" % arg

                    df = _np.sqrt(abs(_np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, grad_f))))
                elif fDims == 1:
                    for i in range(f0.shape[0]):
                        gradFdag = _np.conjugate(_np.transpose(grad_f[i]))
                        df[i] = _np.sqrt(abs(_np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, grad_f[i]))))
                elif fDims == 2:
                    for i in range(f0.shape[0]):
                        for j in range(f0.shape[1]):
                            gradFdag = _np.conjugate(_np.transpose(grad_f[i, j]))
                            df[i, j] = _np.sqrt(abs(_np.dot(gradFdag, _np.dot(self.invRegionQuadcForm, grad_f[i, j]))))
                else:
                    raise ValueError("Unsupported number of dimensions returned by fnOfOp or fnOfModel: %d" % fDims)

        printer.log("df = %s" % df)

        return df

#Helper functions


def _create_empty_grad(val, num_params):
    """ Get finite difference derivative grad_f that is shape (num_params, <shape of val>) """
    if isinstance(val, float) or isinstance(val, int):
        gradVal = _np.zeros(num_params, 'd')
    elif isinstance(val, complex):
        gradVal = _np.zeros(num_params, 'complex')
    else:
        gradSize = (num_params,) + tuple(val.shape)
        gradVal = _np.zeros(gradSize, val.dtype)
    return gradVal  # gradient of value (empty)


def _create_empty_grad_f(f0, num_params):
    if isinstance(f0, dict):  # special behavior for dict: process each item separately
        gradF = {ky: _create_empty_grad(val, num_params) for ky, val in f0.items()}
    else:
        gradF = _create_empty_grad(f0, num_params)
    return gradF
