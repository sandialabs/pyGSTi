from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Functions for generating GST reports (PDF or HTML)."""

import warnings           as _warnings

from .. import tools      as _tools
from .. import objects    as _objs


def logl_confidence_region(gateset, dataset, confidenceLevel,
                           gatestring_list=None, probClipInterval=(-1e6,1e6),
                           minProbClip=1e-4, radius=1e-4, hessianProjection="std",
                           regionType="std", comm=None, memLimit=None,
                           cptp_penalty_factor=None, distributeMethod="deriv",
                           gateLabelAliases=None):

    """
    Constructs a ConfidenceRegion given a gateset and dataset using the log-likelihood Hessian.
    (Internally, this evaluates the log-likelihood Hessian.)

    Parameters
    ----------
    gateset : GateSet
        the gate set point estimate that maximizes the logl or minimizes
        the chi2, and marks the point in gateset-space where the Hessian
        has been evaluated.

    dataset : DataSet
        Probability data

    confidenceLevel : float
        If not None, then the confidence level (between 0 and 100) used in
        the computation of confidence regions/intervals. If None, no
        confidence regions or intervals are computed.

    gatestring_list : list of (tuples or GateStrings), optional
        Each element specifies a gate string to include in the log-likelihood
        sum.  Default value of None implies all the gate strings in dataset
        should be used.

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by gateset. Defaults to no clipping.

    minProbClip : float, optional
        The minimum probability treated normally in the evaluation of the log-likelihood.
        A penalty function replaces the true log-likelihood for probabilities that lie
        below this threshold so that the log-likelihood never becomes undefined.

    radius : float, optional
        Specifies the severity of rounding used to "patch" the zero-frequency
        terms of the log-likelihood.

    hessianProjection : string, optional
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
          in the gate and spam GateSet parameters and set weighting metric
          based on their ratio.
        - 'linear response' -- obtain elements of the Hessian via the
          linear response of a "forcing term".  This requres a likelihood
          optimization for *every* computed error bar, but avoids pre-
          computation of the entire Hessian matrix, which can be 
          prohibitively costly on large parameter spaces.

    regionType : {'std', 'non-markovian'}, optional
        The type of confidence region to create.  'std' creates a standard
        confidence region, while 'non-markovian' creates a region which
        attempts to account for the non-markovian-ness of the data.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    cptp_penalty_factor : float, optional
        The CPTP penalty factor used in MLGST when computing error bars via
        linear-response.

    distributeMethod : {"gatestrings", "deriv"}
        The distribute-method used in MLGST when computing error bars via
        linear-response.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')


    Returns
    -------
    ConfidenceRegion
    """
    if gatestring_list is None:
        gatestring_list = list(dataset.keys())

    if hessianProjection != "linear response":
        #Compute appropriate Hessian
        vb = 3 if memLimit else 0 #only show details of hessian comp when there's a mem limit (a heuristic)
        hessian = _tools.logl_hessian(gateset, dataset, gatestring_list,
                                      minProbClip, probClipInterval, radius,
                                      comm=comm, memLimit=memLimit, verbosity=vb,
                                      gateLabelAliases=gateLabelAliases)
        mlgst_args = None
    else: 
        hessian = None
        mlgst_args = {
            'dataset': dataset,
            'gateStringsToUse': gatestring_list,
            'maxiter': 10000, 'tol': 1e-10,
            'cptp_penalty_factor': cptp_penalty_factor,
            'minProbClip': minProbClip, 
            'probClipInterval': probClipInterval,
            'radius': radius,
            'poissonPicture': True, 'verbosity': 2, #NOTE: HARDCODED
            'memLimit': memLimit, 'comm': comm,
            'distributeMethod': distributeMethod, 'profiler': None
            }

    #Compute the non-Markovian "radius" if required
    if regionType == "std":
        nonMarkRadiusSq = 0.0
    elif regionType == "non-markovian":
        nGateStrings = len(gatestring_list)
        nModelParams = gateset.num_nongauge_params()
        nDataParams  = nGateStrings*(len(dataset.get_spam_labels())-1)
          #number of independent parameters in dataset (max. model # of params)

        MIN_NON_MARK_RADIUS = 1e-8 #must be >= 0
        nonMarkRadiusSq = max( 2*(_tools.logl_max(dataset)
                                  - _tools.logl(gateset, dataset,
                                                gateLabelAliases=gateLabelAliases)) \
                                   - (nDataParams-nModelParams),
                               MIN_NON_MARK_RADIUS )
    else:
        raise ValueError("Invalid confidence region type: %s" % regionType)


    cri = _objs.ConfidenceRegion(gateset, hessian, confidenceLevel,
                                 hessianProjection,
                                 nonMarkRadiusSq=nonMarkRadiusSq,
                                 linresponse_mlgst_params=mlgst_args)

    #Check that number of gauge parameters reported by gateset is consistent with confidence region
    # since the parameter number computed this way is used in chi2 or logl progress tables
    Np_check =  gateset.num_nongauge_params()
    if(Np_check != cri.nNonGaugeParams):
        _warnings.warn("Number of non-gauge parameters in gateset and confidence region do "
                       + " not match.  This indicates an internal logic error.")

    return cri


def chi2_confidence_region(gateset, dataset, confidenceLevel,
                           gatestring_list=None, probClipInterval=(-1e6,1e6),
                           minProbClipForWeighting=1e-4, hessianProjection="std",
                           regionType='std', comm=None, memLimit=None,
                           gateLabelAliases=None):

    """
    Constructs a ConfidenceRegion given a gateset and dataset using the Chi2 Hessian.
    (Internally, this evaluates the Chi2 Hessian.)

    Parameters
    ----------
    gateset : GateSet
        the gate set point estimate that maximizes the logl or minimizes
        the chi2, and marks the point in gateset-space where the Hessian
        has been evaluated.

    dataset : DataSet
        Probability data

    confidenceLevel : float
        If not None, then the confidence level (between 0 and 100) used in
        the computation of confidence regions/intervals. If None, no
        confidence regions or intervals are computed.

    gatestring_list : list of (tuples or GateStrings), optional
        Each element specifies a gate string to include in the log-likelihood
        sum.  Default value of None implies all the gate strings in dataset
        should be used.

    probClipInterval : 2-tuple or None, optional
        (min,max) values used to clip the probabilities predicted by gateset. Defaults to no clipping.

    minProbClipForWeighting : float, optional
        Sets the minimum and maximum probability p allowed in the chi^2 weights: N/(p*(1-p))
        by clipping probability p values to lie within the interval
        [ minProbClipForWeighting, 1-minProbClipForWeighting ].

    hessianProjection : string, optional
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
          in the gate and spam GateSet parameters and set weighting metric
          based on their ratio.
        - 'linear response' -- obtain elements of the Hessian via the
          linear response of a "forcing term".  This requres a likelihood
          optimization for *every* computed error bar, but avoids pre-
          computation of the entire Hessian matrix, which can be 
          prohibitively costly on large parameter spaces.


    regionType : {'std', 'non-markovian'}, optional
        The type of confidence region to create.  'std' creates a standard
        confidence region, while 'non-markovian' creates a region which
        attempts to account for the non-markovian-ness of the data.

    comm : mpi4py.MPI.Comm, optional
        When not None, an MPI communicator for distributing the computation
        across multiple processors.

    memLimit : int, optional
        A rough memory limit in bytes which restricts the amount of intermediate
        values that are computed and stored.

    gateLabelAliases : dictionary, optional
        Dictionary whose keys are gate label "aliases" and whose values are tuples
        corresponding to what that gate label should be expanded into before querying
        the dataset. Defaults to the empty dictionary (no aliases defined)
        e.g. gateLabelAliases['Gx^3'] = ('Gx','Gx','Gx')


    Returns
    -------
    ConfidenceRegion
    """
    if gatestring_list is None:
        gatestring_list = list(dataset.keys())

    if hessianProjection == "linear response":
        raise NotImplementedError("Linear response hessian projection is only"
                                  + " implemented for the logL-objective case")

    #Compute appropriate Hessian
    chi2, hessian = _tools.chi2(dataset, gateset, gatestring_list,
                                False, True, minProbClipForWeighting,
                                probClipInterval, memLimit=memLimit,
                                gateLabelAliases=gateLabelAliases)

    #Compute the non-Markovian "radius" if required
    if regionType == "std":
        nonMarkRadiusSq = 0.0
    elif regionType == "non-markovian":
        nGateStrings = len(gatestring_list)
        nModelParams = gateset.num_nongauge_params()
        nDataParams  = nGateStrings*(len(dataset.get_spam_labels())-1)
          #number of independent parameters in dataset (max. model # of params)

        MIN_NON_MARK_RADIUS = 1e-8 #must be >= 0
        nonMarkRadiusSq = max(chi2 - (nDataParams-nModelParams), MIN_NON_MARK_RADIUS)
    else:
        raise ValueError("Invalid confidence region type: %s" % regionType)


    cri = _objs.ConfidenceRegion(gateset, hessian, confidenceLevel,
                                 hessianProjection,
                                 nonMarkRadiusSq=nonMarkRadiusSq)

    #Check that number of gauge parameters reported by gateset is consistent with confidence region
    # since the parameter number computed this way is used in chi2 or logl progress tables
    Np_check =  gateset.num_nongauge_params()
    if(Np_check != cri.nNonGaugeParams):
        _warnings.warn("Number of non-gauge parameters in gateset and confidence region do "
                       + " not match.  This indicates an internal logic error.")

    return cri
