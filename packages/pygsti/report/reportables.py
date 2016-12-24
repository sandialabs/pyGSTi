from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Functions which compute named quantities for GateSets and Datasets.

Named quantities as well as their confidence-region error bars are
 computed by the functions in this module. These quantities are
 used primarily in reports, so we refer to these quantities as
 "reportables".
"""
import numpy as _np
from collections import OrderedDict as _OrderedDict

from .. import tools as _tools
from .. import algorithms as _alg

FINITE_DIFF_EPS = 1e-7

class ReportableQty(object):
    """
    Encapsulates a computed quantity and possibly its error bars,
    primarily for use in reports.
    """

    def __init__(self, value, errbar=None):
        """
        Initialize a new ReportableQty object, which
        is essentially a container for a value and error bars.

        Parameters
        ----------
        value : anything
           The value to store

        errbar : anything
           The error bar(s) to store
        """
        self.value = value
        self.errbar = errbar

    def get_value(self):
        """
        Returns the quantity's value
        """
        return self.value

    def get_err_bar(self):
        """
        Returns the quantity's error bar(s)
        """
        return self.errbar

    def get_value_and_err_bar(self):
        """
        Returns the quantity's value and error bar(s)
        """
        return self.value, self.errbar

    def __str__(self):
        if self.errbar is not None:
            return str(self.value) + " +/- " + str(self.errbar)
        else: return str(self.value)


def _projectToValidProb(p, tol=1e-9):
    if p < tol: return tol
    if p > 1-tol: return 1-tol
    return p


def _getGateQuantity(fnOfGate, gateset, gateLabel, eps, confidenceRegionInfo, verbosity=0):
    """ For constructing a ReportableQty from a function of a gate. """

    if confidenceRegionInfo is None: # No Error bars
        return ReportableQty(fnOfGate(gateset.gates[gateLabel]))

    # make sure the gateset we're given is the one used to generate the confidence region
    if(gateset.frobeniusdist(confidenceRegionInfo.get_gateset()) > 1e-6):
        raise ValueError("Prep quantity confidence region is being requested for " +
                         "a different gateset than the given confidenceRegionInfo")

    df, f0 = confidenceRegionInfo.get_gate_fn_confidence_interval(fnOfGate, gateLabel,
                                                              eps, returnFnVal=True,
                                                              verbosity=verbosity)
    return ReportableQty(f0,df)

def _getPrepQuantity(fnOfPrep, gateset, prepLabel, eps, confidenceRegionInfo, verbosity=0):
    """ For constructing a ReportableQty from a function of a state preparation. """

    if confidenceRegionInfo is None: # No Error bars
        return ReportableQty(fnOfPrep(gateset.preps[prepLabel]))

    # make sure the gateset we're given is the one used to generate the confidence region
    if(gateset.frobeniusdist(confidenceRegionInfo.get_gateset()) > 1e-6):
        raise ValueError("Gate quantity confidence region is being requested for " +
                         "a different gateset than the given confidenceRegionInfo")

    df, f0 = confidenceRegionInfo.get_prep_fn_confidence_interval(fnOfPrep, prepLabel,
                                                              eps, returnFnVal=True,
                                                              verbosity=verbosity)
    return ReportableQty(f0,df)


def _getEffectQuantity(fnOfEffect, gateset, effectLabel, eps, confidenceRegionInfo, verbosity=0):
    """ For constructing a ReportableQty from a function of a POVM effect. """

    if confidenceRegionInfo is None: # No Error bars
        return ReportableQty(fnOfEffect(gateset.effects[effectLabel]))

    # make sure the gateset we're given is the one used to generate the confidence region
    if(gateset.frobeniusdist(confidenceRegionInfo.get_gateset()) > 1e-6):
        raise ValueError("Effect quantity confidence region is being requested for " +
                         "a different gateset than the given confidenceRegionInfo")

    df, f0 = confidenceRegionInfo.get_effect_fn_confidence_interval(
        fnOfEffect, effectLabel, eps, returnFnVal=True, verbosity=verbosity)
    return ReportableQty(f0,df)


def _getGateSetQuantity(fnOfGateSet, gateset, eps, confidenceRegionInfo, verbosity=0):
    """ For constructing a ReportableQty from a function of a gate. """

    if confidenceRegionInfo is None: # No Error bars
        return ReportableQty(fnOfGateSet(gateset))

    # make sure the gateset we're given is the one used to generate the confidence region
    if(gateset.frobeniusdist(confidenceRegionInfo.get_gateset()) > 1e-6):
        raise ValueError("GateSet quantity confidence region is being requested for " +
                         "a different gateset than the given confidenceRegionInfo")

    df, f0 = confidenceRegionInfo.get_gateset_fn_confidence_interval(
        fnOfGateSet, eps, returnFnVal=True, verbosity=verbosity)

    return ReportableQty(f0,df)


def _getSpamQuantity(fnOfSpamVecs, gateset, eps, confidenceRegionInfo, verbosity=0):
    """ For constructing a ReportableQty from a function of a spam vectors."""

    if confidenceRegionInfo is None: # No Error bars
        return ReportableQty(fnOfSpamVecs(gateset.get_preps(), gateset.get_effects()))

    # make sure the gateset we're given is the one used to generate the confidence region
    if(gateset.frobeniusdist(confidenceRegionInfo.get_gateset()) > 1e-6):
        raise ValueError("Spam quantity confidence region is being requested for " +
                         "a different gateset than the given confidenceRegionInfo")

    df, f0 = confidenceRegionInfo.get_spam_fn_confidence_interval(fnOfSpamVecs,
                                                              eps, returnFnVal=True,
                                                              verbosity=verbosity)
    return ReportableQty(f0,df)



def compute_dataset_qty(qtyname, dataset, gatestrings=None):
    """
    Compute the named "Dataset" quantity.

    Parameters
    ----------
    qtyname : string
        Name of the quantity to compute.

    dataset : DataSet
        Data used to compute the quantity.

    gatestrings : list of tuples or GateString objects, optional
        A list of gatestrings used in the computation of certain quantities.
        If None, all the gatestrings in the dataset are used.

    Returns
    -------
    ReportableQty
        The quantity requested, or None if quantity could not be computed.
    """
    ret = compute_dataset_qtys( [qtyname], dataset, gatestrings )
    if qtyname is None: return ret
    elif qtyname in ret: return ret[qtyname]
    else: return None

def compute_dataset_qtys(qtynames, dataset, gatestrings=None):
    """
    Compute the named "Dataset" quantities.

    Parameters
    ----------
    qtynames : list of strings
        Names of the quantities to compute.

    dataset : DataSet
        Data used to compute the quantity.

    gatestrings : list of tuples or GateString objects, optional
        A list of gatestrings used in the computation of certain quantities.
        If None, all the gatestrings in the dataset are used.

    Returns
    -------
    dict
        Dictionary whose keys are the requested quantity names and values are
        ReportableQty objects.
    """

    ret = _OrderedDict()
    possible_qtys = [ ]

    #Quantities computed per gatestring
    per_gatestring_qtys = _OrderedDict( [('gate string', []), ('gate string index', []), ('gate string length', []), ('count total', [])] )
    spamLabels = dataset.get_spam_labels()
    for spl in spamLabels:
        per_gatestring_qtys['Exp prob(%s)' % spl] = []
        per_gatestring_qtys['Exp count(%s)' % spl] = []

    if any( [qtyname in per_gatestring_qtys for qtyname in qtynames ] ):
        if gatestrings is None: gatestrings = list(dataset.keys())
        for (i,gs) in enumerate(gatestrings):
            if gs in dataset: # skip gate strings given that are not in dataset
                dsRow = dataset[gs]
            else:
                #print "Warning: skipping gate string %s" % str(gs)
                continue

            N = dsRow.total()
            per_gatestring_qtys['gate string'].append(  ''.join(gs)  )
            per_gatestring_qtys['gate string index'].append( i )
            per_gatestring_qtys['gate string length'].append(  len(gs)  )
            per_gatestring_qtys['count total'].append(  N  )

            for spamLabel in spamLabels:
                pExp = _projectToValidProb( dsRow[spamLabel] / N, tol=1e-10 )
                per_gatestring_qtys['Exp prob(%s)' % spamLabel].append( pExp )
                per_gatestring_qtys['Exp count(%s)' % spamLabel].append( dsRow[spamLabel] )

        for qtyname in qtynames:
            if qtyname in per_gatestring_qtys:
                ret[qtyname] = ReportableQty(per_gatestring_qtys[qtyname])


    #Quantities computed per dataset
    qty = "max logl"; possible_qtys.append(qty)
    if qty in qtynames:
        ret[qty] = ReportableQty( _tools.logl_max(dataset))

    qty = "number of gate strings"; possible_qtys.append(qty)
    if qty in qtynames:
        ret[qty] = ReportableQty( len(dataset) )

    if qtynames[0] is None:
        return possible_qtys + list(per_gatestring_qtys.keys())
    return ret


def compute_gateset_qty(qtyname, gateset, confidenceRegionInfo=None):
    """
    Compute the named "GateSet" quantity.

    Parameters
    ----------
    qtyname : string
        Name of the quantity to compute.

    gateset : GateSet
        Gate set used to compute the quantity.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region used to compute the error bars
        contained in the returned quantity.  If None, then no error bars are
        computed.

    Returns
    -------
    ReportableQty
        The quantity requested, or None if quantity could not be computed.
    """
    ret = compute_gateset_qtys( [qtyname], gateset, confidenceRegionInfo)
    if qtyname is None: return ret
    elif qtyname in ret: return ret[qtyname]
    else: return None

def compute_gateset_qtys(qtynames, gateset, confidenceRegionInfo=None):
    """
    Compute the named "GateSet" quantities.

    Parameters
    ----------
    qtynames : list of strings
        Names of the quantities to compute.

    gateset : GateSet
        Gate set used to compute the quantities.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region used to compute the error bars
        contained in the returned quantities.  If None, then no error bars are
        computed.

    Returns
    -------
    dict
        Dictionary whose keys are the requested quantity names and values are
        ReportableQty objects.
    """
    ret = _OrderedDict()
    possible_qtys = [ ]
    eps = FINITE_DIFF_EPS
    mxBasis = gateset.get_basis_name()

    def choi_matrix(gate):
        return _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)

    def choi_evals(gate):
        choi = _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)
        choi_eigvals = _np.linalg.eigvals(choi)
        return _np.array(sorted(choi_eigvals))

    def choi_trace(gate):
        choi = _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)
        return _np.trace(choi)

    def decomp_angle(gate):
        decomp = _tools.decompose_gate_matrix(gate)
        return decomp.get('pi rotations',0)

    def decomp_decay_diag(gate):
        decomp = _tools.decompose_gate_matrix(gate)
        return decomp.get('decay of diagonal rotation terms',0)

    def decomp_decay_offdiag(gate):
        decomp = _tools.decompose_gate_matrix(gate)
        return decomp.get('decay of off diagonal rotation terms',0)

    def decomp_cu_angle(gate):
        closestUGateMx = _alg.find_closest_unitary_gatemx(gate)
        decomp = _tools.decompose_gate_matrix(closestUGateMx)
        return decomp.get('pi rotations',0)

    def decomp_cu_decay_diag(gate):
        closestUGateMx = _alg.find_closest_unitary_gatemx(gate)
        decomp = _tools.decompose_gate_matrix(closestUGateMx)
        return decomp.get('decay of diagonal rotation terms',0)

    def decomp_cu_decay_offdiag(gate):
        closestUGateMx = _alg.find_closest_unitary_gatemx(gate)
        decomp = _tools.decompose_gate_matrix(closestUGateMx)
        return decomp.get('decay of off diagonal rotation terms',0)

    def upper_bound_fidelity(gate):
        return _tools.fidelity_upper_bound(gate)[0]

    def closest_ujmx(gate):
        closestUGateMx = _alg.find_closest_unitary_gatemx(gate)
        return _tools.jamiolkowski_iso(closestUGateMx, mxBasis, mxBasis)

    def maximum_fidelity(gate):
        closestUGateMx = _alg.find_closest_unitary_gatemx(gate)
        closestUJMx = _tools.jamiolkowski_iso(closestUGateMx, mxBasis, mxBasis)
        choi = _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)
        return _tools.fidelity(closestUJMx, choi)

    def maximum_trace_dist(gate):
        closestUGateMx = _alg.find_closest_unitary_gatemx(gate)
        #closestUJMx = _tools.jamiolkowski_iso(closestUGateMx, mxBasis, mxBasis)
        _tools.jamiolkowski_iso(closestUGateMx, mxBasis, mxBasis)
        return _tools.jtracedist(gate, closestUGateMx)

    def spam_dotprods(rhoVecs, EVecs):
        ret = _np.empty( (len(rhoVecs), len(EVecs)), 'd')
        for i,rhoVec in enumerate(rhoVecs):
            for j,EVec in enumerate(EVecs):
                ret[i,j] = _np.dot(_np.transpose(EVec), rhoVec)
        return ret

    def angles_btwn_rotn_axes(gateset):
        gateLabels = list(gateset.gates.keys())
        angles_btwn_rotn_axes = _np.zeros( (len(gateLabels), len(gateLabels)), 'd' )

        for i,gl in enumerate(gateLabels):
            decomp = _tools.decompose_gate_matrix(gateset.gates[gl])
            rotnAngle = decomp.get('pi rotations','X')
            axisOfRotn = decomp.get('axis of rotation',None)

            for j,gl_other in enumerate(gateLabels[i+1:],start=i+1):
                decomp_other = _tools.decompose_gate_matrix(gateset.gates[gl_other])
                rotnAngle_other = decomp_other.get('pi rotations','X')

                if str(rotnAngle) == 'X' or abs(rotnAngle) < 1e-4 or \
                   str(rotnAngle_other) == 'X' or abs(rotnAngle_other) < 1e-4:
                    angles_btwn_rotn_axes[i,j] =  _np.nan
                else:
                    axisOfRotn_other = decomp_other.get('axis of rotation',None)
                    if axisOfRotn is not None and axisOfRotn_other is not None:
                        real_dot =  _np.clip( _np.real(_np.dot(axisOfRotn,axisOfRotn_other)), -1.0, 1.0)
                        angles_btwn_rotn_axes[i,j] = _np.arccos( real_dot ) / _np.pi
                    else:
                        angles_btwn_rotn_axes[i,j] = _np.nan

                angles_btwn_rotn_axes[j,i] = angles_btwn_rotn_axes[i,j]
        return angles_btwn_rotn_axes



    # Spam quantities (computed for all spam vectors at once):
    key = "Spam DotProds"; possible_qtys.append(key)
    if key in qtynames:
        ret[key] = _getSpamQuantity(spam_dotprods, gateset, eps, confidenceRegionInfo)

    key = "Gateset Axis Angles"; possible_qtys.append(key)
    if key in qtynames:
        ret[key] = _getGateSetQuantity(angles_btwn_rotn_axes, gateset, eps, confidenceRegionInfo)

    # Quantities computed per gate
    for (label,gate) in gateset.gates.items():

        #Gate quantities
        suffixes = ('eigenvalues', 'eigenvectors', 'choi eigenvalues', 'choi trace',
                    'choi matrix', 'decomposition')
        gate_qtys = _OrderedDict( [ ("%s %s" % (label,s), None) for s in suffixes ] )
        possible_qtys += list(gate_qtys.keys())

        if any( [qtyname in gate_qtys for qtyname in qtynames] ):
            #gate_evals,gate_evecs = _np.linalg.eig(gate)
            evalsQty = _getGateQuantity(_np.linalg.eigvals, gateset, label, eps, confidenceRegionInfo)
            choiQty = _getGateQuantity(choi_matrix, gateset, label, eps, confidenceRegionInfo)
            choiEvQty = _getGateQuantity(choi_evals, gateset, label, eps, confidenceRegionInfo)
            choiTrQty = _getGateQuantity(choi_trace, gateset, label, eps, confidenceRegionInfo)

            decompDict = _tools.decompose_gate_matrix(gate)
            if decompDict['isValid']:
                angleQty = _getGateQuantity(decomp_angle, gateset, label, eps, confidenceRegionInfo)
                diagQty = _getGateQuantity(decomp_decay_diag, gateset, label, eps, confidenceRegionInfo)
                offdiagQty = _getGateQuantity(decomp_decay_offdiag, gateset, label, eps, confidenceRegionInfo)
                errBarDict = { 'pi rotations': angleQty.get_err_bar(),
                               'decay of diagonal rotation terms': diagQty.get_err_bar(),
                               'decay of off diagonal rotation terms': offdiagQty.get_err_bar() }
                decompQty = ReportableQty(decompDict, errBarDict)
            else:
                decompQty = ReportableQty({})

            gate_qtys[ '%s eigenvalues' % label ]      = evalsQty
            #gate_qtys[ '%s eigenvectors' % label ]     = gate_evecs
            gate_qtys[ '%s choi matrix' % label ]      = choiQty
            gate_qtys[ '%s choi eigenvalues' % label ] = choiEvQty
            gate_qtys[ '%s choi trace' % label ]       = choiTrQty
            gate_qtys[ '%s decomposition' % label]     = decompQty

            for qtyname in qtynames:
                if qtyname in gate_qtys:
                    ret[qtyname] = gate_qtys[qtyname]


        #Closest unitary quantities
        suffixes = ('max fidelity with unitary',
                    'max trace dist with unitary',
                    'upper bound on fidelity with unitary',
                    'closest unitary choi matrix',
                    'closest unitary decomposition')
        closestU_qtys = _OrderedDict( [ ("%s %s" % (label,s), None) for s in suffixes ] )
        possible_qtys += list(closestU_qtys.keys())
        if any( [qtyname in closestU_qtys for qtyname in qtynames] ):
            ubFQty = _getGateQuantity(upper_bound_fidelity, gateset, label, eps, confidenceRegionInfo)
            closeUJMxQty = _getGateQuantity(closest_ujmx, gateset, label, eps, confidenceRegionInfo)
            maxFQty = _getGateQuantity(maximum_fidelity, gateset, label, eps, confidenceRegionInfo)
            maxJTDQty = _getGateQuantity(maximum_trace_dist, gateset, label, eps, confidenceRegionInfo)

            closestUGateMx = _alg.find_closest_unitary_gatemx(gate)
            decompDict = _tools.decompose_gate_matrix(closestUGateMx)
            if decompDict['isValid']:
                angleQty = _getGateQuantity(decomp_cu_angle, gateset, label, eps, confidenceRegionInfo)
                diagQty = _getGateQuantity(decomp_cu_decay_diag, gateset, label, eps, confidenceRegionInfo)
                offdiagQty = _getGateQuantity(decomp_cu_decay_offdiag, gateset, label, eps, confidenceRegionInfo)
                errBarDict = { 'pi rotations': angleQty.get_err_bar(),
                               'decay of diagonal rotation terms': diagQty.get_err_bar(),
                               'decay of off diagonal rotation terms': offdiagQty.get_err_bar() }
                decompQty = ReportableQty(decompDict, errBarDict)
            else:
                decompQty = ReportableQty({})

            closestU_qtys[ '%s max fidelity with unitary' % label ]                  = maxFQty
            closestU_qtys[ '%s max trace dist with unitary' % label ]                = maxJTDQty
            closestU_qtys[ '%s upper bound on fidelity with unitary' % label ]       = ubFQty
            closestU_qtys[ '%s closest unitary choi matrix' % label ]                = closeUJMxQty
            closestU_qtys[ '%s closest unitary decomposition' % label ]              = decompQty

            for qtyname in qtynames:
                if qtyname in closestU_qtys:
                    ret[qtyname] = closestU_qtys[qtyname]

    if qtynames[0] is None:
        return possible_qtys
    return ret


def compute_gateset_dataset_qty(qtyname, gateset, dataset, gatestrings=None):
    """
    Compute the named "GateSet & Dataset" quantity.

    Parameters
    ----------
    qtyname : string
        Name of the quantity to compute.

    gateset : GateSet
        Gate set used to compute the quantity.

    dataset : DataSet
        Data used to compute the quantity.

    gatestrings : list of tuples or GateString objects, optional
        A list of gatestrings used in the computation of certain quantities.
        If None, all the gatestrings in the dataset are used.

    Returns
    -------
    ReportableQty
        The quantity requested, or None if quantity could not be computed.
    """
    ret = compute_gateset_dataset_qtys( [qtyname], gateset, dataset, gatestrings )
    if qtyname is None: return ret
    elif qtyname in ret: return ret[qtyname]
    else: return None

def compute_gateset_dataset_qtys(qtynames, gateset, dataset, gatestrings=None):
    """
    Compute the named "GateSet & Dataset" quantities.

    Parameters
    ----------
    qtynames : list of strings
        Names of the quantities to compute.

    gateset : GateSet
        Gate set used to compute the quantities.

    dataset : DataSet
        Data used to compute the quantities.

    gatestrings : list of tuples or GateString objects, optional
        A list of gatestrings used in the computation of certain quantities.
        If None, all the gatestrings in the dataset are used.

    Returns
    -------
    dict
        Dictionary whose keys are the requested quantity names and values are
        ReportableQty objects.
    """

    #Note: no error bars computed for these quantities yet...

    ret = _OrderedDict()
    possible_qtys = [ ]

    #Quantities computed per gatestring
    per_gatestring_qtys = _OrderedDict() # OLD qtys: [('logl term diff', []), ('score', [])]
    for spl in gateset.get_spam_labels():
        per_gatestring_qtys['prob(%s) diff' % spl] = []
        per_gatestring_qtys['count(%s) diff' % spl] = []
        per_gatestring_qtys['Est prob(%s)' % spl] = []
        per_gatestring_qtys['Est count(%s)' % spl] = []
        per_gatestring_qtys['gatestring chi2(%s)' % spl] = []

    if any( [qtyname in per_gatestring_qtys for qtyname in qtynames ] ):
        if gatestrings is None: gatestrings = list(dataset.keys())
        for gs in gatestrings:
            if gs in dataset: # skip gate strings given that are not in dataset
                dsRow = dataset[gs]
            else: continue

            p = gateset.probs(gs)
            pExp = { }; N = dsRow.total()
            for spamLabel in p:
                p[spamLabel] = _projectToValidProb( p[spamLabel], tol=1e-10 )
                pExp[spamLabel] = _projectToValidProb( dsRow[spamLabel] / N, tol=1e-10 )

            #OLD
            #per_gatestring_qtys['logl term diff'].append(  _tools.logL_term(dsRow, pExp) - _tools.logL_term(dsRow, p)  )
            #per_gatestring_qtys['score'].append(  (_tools.logL_term(dsRow, pExp) - _tools.logL_term(dsRow, p)) / N  )

            for spamLabel in p:
                per_gatestring_qtys['prob(%s) diff' % spamLabel].append( abs(p[spamLabel] - pExp[spamLabel]) )
                per_gatestring_qtys['count(%s) diff' % spamLabel].append( int( round(p[spamLabel] * N) - dsRow[spamLabel]) )
                per_gatestring_qtys['Est prob(%s)' % spamLabel].append( p[spamLabel] )
                per_gatestring_qtys['Est count(%s)' % spamLabel].append( int(round(p[spamLabel] * N)) )
                per_gatestring_qtys['gatestring chi2(%s)' % spamLabel].append( _tools.chi2fn( N, p[spamLabel], pExp[spamLabel], 1e-4 ) )

        for qtyname in qtynames:
            if qtyname in per_gatestring_qtys:
                ret[qtyname] = ReportableQty( per_gatestring_qtys[qtyname] )

    #Quantities which take a single value for a given gateset and dataset
    qty = "logl"; possible_qtys.append(qty)
    if qty in qtynames:
        ret[qty] = ReportableQty( _tools.logl(gateset, dataset) )

    qty = "logl diff"; possible_qtys.append(qty)
    if qty in qtynames:
        ret[qty] = ReportableQty( _tools.logl_max(dataset) - _tools.logl(gateset, dataset) )

    qty = "chi2"; possible_qtys.append(qty)
    if qty in qtynames:
        ret[qty] = ReportableQty( _tools.chi2( dataset, gateset, minProbClipForWeighting=1e-4) )

    #Quantities which take a single value per spamlabel for a given gateset and dataset
    #for spl in gateset.get_spam_labels():
    #    qty = "chi2(%s)" % spl; possible_qtys.append(qty)
    #    if qty in qtynames:
    #        ret[qty] = _tools.chi2( dataset, gateset, minProbClipForWeighting=1e-4)

    if qtynames[0] is None:
        return possible_qtys + list(per_gatestring_qtys.keys())
    return ret


def compute_gateset_gateset_qty(qtyname, gateset1, gateset2,
                                confidenceRegionInfo=None):
    """
    Compute the named "GateSet vs. GateSet" quantity.

    Parameters
    ----------
    qtyname : string
        Name of the quantity to compute.

    gateset1 : GateSet
        First gate set used to compute the quantity.

    gateset2 : GateSet
        Second gate set used to compute the quantity.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region used to compute the error bars
        contained in the returned quantity.  If None, then no error bars are
        computed.

    Returns
    -------
    ReportableQty
        The quantity requested, or None if quantity could not be computed.
    """
    ret = compute_gateset_gateset_qtys( [qtyname], gateset1, gateset2, confidenceRegionInfo)
    if qtyname is None: return ret
    elif qtyname in ret: return ret[qtyname]
    else: return None

def compute_gateset_gateset_qtys(qtynames, gateset1, gateset2,
                                 confidenceRegionInfo=None):
    """
    Compute the named "GateSet vs. GateSet" quantities.

    Parameters
    ----------
    qtynames : list of strings
        Names of the quantities to compute.

    gateset1 : GateSet
        First gate set used to compute the quantities.

    gateset2 : GateSet
        Second gate set used to compute the quantities.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region used to compute the error bars
        contained in the returned quantities.  If None, then no error bars are
        computed.

    Returns
    -------
    dict
        Dictionary whose keys are the requested quantity names and values are
        ReportableQty objects.
    """
    ret = _OrderedDict()
    possible_qtys = [ ]
    eps = FINITE_DIFF_EPS

    for gateLabel in gateset1.gates:
        if gateLabel not in gateset2.gates:
            raise ValueError("%s gate is missing from second gateset - cannot compare gatesets", gateLabel)
    for gateLabel in gateset2.gates:
        if gateLabel not in gateset1.gates:
            raise ValueError("%s gate is missing from first gateset - cannot compare gatesets", gateLabel)

    mxBasis = gateset1.get_basis_name()
    if mxBasis != gateset2.get_basis_name():
        raise ValueError("Basis mismatch: %s != %s" %
                         (mxBasis, gateset2.get_basis_name()))

    ### per gate quantities
    #############################################
    for gateLabel in gateset1.gates:

        key = '%s fidelity' % gateLabel; possible_qtys.append(key)
        key2 = '%s infidelity' % gateLabel; possible_qtys.append(key)
        if key in qtynames or key2 in qtynames:

            def process_fidelity(gate):
                return _tools.process_fidelity(gate, gateset2.gates[gateLabel], mxBasis)
                  #vary elements of gateset1 (assume gateset2 is fixed)

            #print "DEBUG: fidelity(%s)" % gateLabel
            FQty = _getGateQuantity(process_fidelity, gateset1, gateLabel,
                                    eps, confidenceRegionInfo)

            InFQty = ReportableQty( 1.0-FQty.get_value(), FQty.get_err_bar() )
            if key in qtynames: ret[key] = FQty
            if key2 in qtynames: ret[key2] = InFQty

        key = '%s closest unitary fidelity' % gateLabel; possible_qtys.append(key)
        if key in qtynames:

            #Note: default 'gm' basis
            def closest_unitary_fidelity(gate): # assume vary gateset1, gateset2 fixed
                decomp1 = _tools.decompose_gate_matrix(gate)
                decomp2 = _tools.decompose_gate_matrix(gateset2.gates[gateLabel])

                if decomp1['isUnitary']:
                    closestUGateMx1 = gate
                else: closestUGateMx1 = _alg.find_closest_unitary_gatemx(gate)

                if decomp2['isUnitary']:
                    closestUGateMx2 = gateset2.gates[gateLabel]
                else: closestUGateMx2 = _alg.find_closest_unitary_gatemx(gateset2.gates[gateLabel])

                closeChoi1 = _tools.jamiolkowski_iso(closestUGateMx1)
                closeChoi2 = _tools.jamiolkowski_iso(closestUGateMx2)
                return _tools.fidelity(closeChoi1,closeChoi2)

            ret[key] = _getGateQuantity(closest_unitary_fidelity, gateset1, gateLabel, eps, confidenceRegionInfo)

        key = "%s Frobenius diff" % gateLabel; possible_qtys.append(key)
        if key in qtynames:
            def fro_diff(gate): # assume vary gateset1, gateset2 fixed
                return _tools.frobeniusdist(gate,gateset2.gates[gateLabel])
            #print "DEBUG: frodist(%s)" % gateLabel
            ret[key] = _getGateQuantity(fro_diff, gateset1, gateLabel, eps, confidenceRegionInfo)

        key = "%s Jamiolkowski trace dist" % gateLabel; possible_qtys.append(key)
        if key in qtynames:
            def jt_diff(gate): # assume vary gateset1, gateset2 fixed
                return _tools.jtracedist(gate,gateset2.gates[gateLabel], mxBasis)
            #print "DEBUG: jtdist(%s)" % gateLabel
            ret[key] = _getGateQuantity(jt_diff, gateset1, gateLabel, eps, confidenceRegionInfo)

        key = '%s diamond norm' % gateLabel; possible_qtys.append(key)
        if key in qtynames:

            def half_diamond_norm(gate):
                return 0.5 * _tools.diamonddist(gate, gateset2.gates[gateLabel], mxBasis)
                  #vary elements of gateset1 (assume gateset2 is fixed)

            try:
                ret[key] = _getGateQuantity(half_diamond_norm, gateset1, gateLabel,
                                            eps, confidenceRegionInfo)
            except ImportError: #if failed to import cvxpy (probably b/c it's not installed)
                ret[key] = ReportableQty(_np.nan) # report NAN for diamond norms

        key = '%s angle btwn rotn axes' % gateLabel; possible_qtys.append(key)
        if key in qtynames:

            def angle_btwn_axes(gate): #Note: default 'gm' basis
                decomp = _tools.decompose_gate_matrix(gate)
                decomp2 = _tools.decompose_gate_matrix(gateset2.gates[gateLabel])
                axisOfRotn = decomp.get('axis of rotation',None)
                rotnAngle = decomp.get('pi rotations','X')
                axisOfRotn2 = decomp2.get('axis of rotation',None)
                rotnAngle2 = decomp2.get('pi rotations','X')

                if rotnAngle == 'X' or abs(rotnAngle) < 1e-4 or \
                   rotnAngle2 == 'X' or abs(rotnAngle2) < 1e-4:
                    return _np.nan

                if axisOfRotn is None or axisOfRotn2 is None:
                    return _np.nan

                real_dot =  _np.clip( _np.real(_np.dot(axisOfRotn,axisOfRotn2)), -1.0, 1.0)
                return _np.arccos( abs(real_dot) ) / _np.pi
                  #Note: abs() allows axis to be off by 180 degrees -- if showing *angle* as
                  #      well, must flip sign of angle of rotation if you allow axis to
                  #      "reverse" by 180 degrees.

            ret[key] = _getGateQuantity(angle_btwn_axes, gateset1, gateLabel,
                                    eps, confidenceRegionInfo)

        key = '%s relative logTiG eigenvalues' % gateLabel; possible_qtys.append(key)
        if key in qtynames:
            def rel_eigvals(gate):
                rel_gate = _tools.error_generator(gate, gateset2.gates[gateLabel], "logTiG")
                return _np.linalg.eigvals(rel_gate)
                  #vary elements of gateset1 (assume gateset2 is fixed)

            ret[key] = _getGateQuantity(rel_eigvals, gateset1, gateLabel,
                                        eps, confidenceRegionInfo)

        key = '%s relative logG-logT eigenvalues' % gateLabel; possible_qtys.append(key)
        if key in qtynames:
            def rel_eigvals(gate):
                rel_gate = _tools.error_generator(gate, gateset2.gates[gateLabel], "logG-logT")
                return _np.linalg.eigvals(rel_gate)
                  #vary elements of gateset1 (assume gateset2 is fixed)

            ret[key] = _getGateQuantity(rel_eigvals, gateset1, gateLabel,
                                        eps, confidenceRegionInfo)


    ### per prep vector quantities
    #############################################
    for prepLabel in gateset1.get_prep_labels():

        key = '%s prep state fidelity' % prepLabel; possible_qtys.append(key)
        key2 = '%s prep state infidelity' % prepLabel; possible_qtys.append(key)
        if key in qtynames or key2 in qtynames:

            def fidelity(vec):
                rhoMx1 = _tools.vec_to_stdmx(vec, mxBasis)
                rhoMx2 = _tools.vec_to_stdmx(gateset2.preps[prepLabel], mxBasis)
                return _tools.fidelity(rhoMx1, rhoMx2)
                  #vary elements of gateset1 (assume gateset2 is fixed)

            FQty = _getPrepQuantity(fidelity, gateset1, prepLabel,
                                    eps, confidenceRegionInfo)

            InFQty = ReportableQty( 1.0-FQty.get_value(), FQty.get_err_bar() )
            if key in qtynames: ret[key] = FQty
            if key2 in qtynames: ret[key2] = InFQty

        key = "%s prep trace dist" % prepLabel; possible_qtys.append(key)
        if key in qtynames:
            def tr_diff(vec): # assume vary gateset1, gateset2 fixed
                rhoMx1 = _tools.vec_to_stdmx(vec, mxBasis)
                rhoMx2 = _tools.vec_to_stdmx(gateset2.preps[prepLabel], mxBasis)
                return _tools.tracedist(rhoMx1, rhoMx2)
            ret[key] = _getPrepQuantity(tr_diff, gateset1, prepLabel,
                                        eps, confidenceRegionInfo)


    ### per effect vector quantities
    #############################################
    for effectLabel in gateset1.get_effect_labels():

        key = '%s effect state fidelity' % effectLabel; possible_qtys.append(key)
        key2 = '%s effect state infidelity' % effectLabel; possible_qtys.append(key)
        if key in qtynames or key2 in qtynames:

            def fidelity(vec):
                EMx1 = _tools.vec_to_stdmx(vec, mxBasis)
                EMx2 = _tools.vec_to_stdmx(gateset2.effects[effectLabel], mxBasis)
                return _tools.fidelity(EMx1,EMx2)
                  #vary elements of gateset1 (assume gateset2 is fixed)

            FQty = _getEffectQuantity(fidelity, gateset1, effectLabel,
                                      eps, confidenceRegionInfo)

            InFQty = ReportableQty( 1.0-FQty.get_value(), FQty.get_err_bar() )
            if key in qtynames: ret[key] = FQty
            if key2 in qtynames: ret[key2] = InFQty

        key = "%s effect trace dist" % effectLabel; possible_qtys.append(key)
        if key in qtynames:
            def tr_diff(vec): # assume vary gateset1, gateset2 fixed
                EMx1 = _tools.vec_to_stdmx(vec, mxBasis)
                EMx2 = _tools.vec_to_stdmx(gateset2.effects[effectLabel], mxBasis)
                return _tools.tracedist(EMx1, EMx2)
            ret[key] = _getEffectQuantity(tr_diff, gateset1, effectLabel,
                                          eps, confidenceRegionInfo)


    ###  per gateset quantities
    #############################################
    key = "Gateset Frobenius diff"; possible_qtys.append(key)
    if key in qtynames: ret[key] = ReportableQty( gateset1.frobeniusdist(gateset2) )

    key = "Max Jamiolkowski trace dist"; possible_qtys.append(key)
    if key in qtynames: ret[key] = ReportableQty(
        max( [ _tools.jtracedist(gateset1.gates[l],gateset2.gates[l])
               for l in gateset1.gates ] ) )


    #Special case: when qtyname is None then return a list of all possible names that can be computed
    if qtynames[0] is None:
        return possible_qtys
    return ret
