""" 
Functions which compute named quantities for GateSets and Datasets.

Named quantities as well as their confidence-region error bars are
 computed by the functions in this module. These quantities are 
 used primarily in reports, so we refer to these quantities as
 "reportables".
"""
import numpy as _np
from collections import OrderedDict as _OrderedDict

import LikelihoodFunctions as _LF
import GateOps as _GateOps
import JamiolkowskiOps as _JOps
import MatrixOps as _MatrixOps
import AnalysisTools as _AT

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

    def getValue(self):
        """
        Returns the quantity's value
        """
        return self.value

    def getErrBar(self):
        """
        Returns the quantity's error bar(s)
        """
        return self.errbar

    def getValueAndErrBar(self):
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
        return ReportableQty(fnOfGate(gateset[gateLabel]))

    # make sure the gateset we're given is the one used to generate the confidence region
    if(gateset.diff_Frobenius(confidenceRegionInfo.getGateset()) > 1e-6):
        raise ValueError("Gate quantity confidence region is being requested for " +
                         "a different gateset than the given confidenceRegionInfo")

    df, f0 = confidenceRegionInfo.getGateFnConfidenceInterval(fnOfGate, gateLabel,
                                                              eps, returnFnVal=True,
                                                              verbosity=verbosity)
    return ReportableQty(f0,df)


def _getSpamQuantity(fnOfSpamVecs, gateset, eps, confidenceRegionInfo, verbosity=0):
    """ For constructing a ReportableQty from a function of a spam vectors."""

    if confidenceRegionInfo is None: # No Error bars
        return ReportableQty(fnOfSpamVecs(gateset.get_rhoVecs(), gateset.get_EVecs()))

    # make sure the gateset we're given is the one used to generate the confidence region
    if(gateset.diff_Frobenius(confidenceRegionInfo.getGateset()) > 1e-6):
        raise ValueError("Spam quantity confidence region is being requested for " +
                         "a different gateset than the given confidenceRegionInfo")

    df, f0 = confidenceRegionInfo.getSpamFnConfidenceInterval(fnOfSpamVecs, 
                                                              eps, returnFnVal=True,
                                                              verbosity=verbosity)
    return ReportableQty(f0,df)

                                   

def compute_DataSet_Quantity(qtyname, dataset, gatestrings):
    """
    Compute the named "Dataset" quantity.

    Parameters
    ----------
    qtyname : string
        Name of the quantity to compute.
        
    dataset : DataSet
        Data used to compute the quantity.

    gatestrings : list of tuples or GateString objects
        A list of gatestrings used in the computation of certain quantities.
        
    Returns
    -------
    ReportableQty
        The quantity requested, or None if quantity could not be computed.
    """
    ret = compute_DataSet_Quantities( [qtyname], dataset, gatestrings )
    if qtyname is None: return ret
    elif ret.has_key(qtyname): return ret[qtyname]
    else: return None

def compute_DataSet_Quantities(qtynames, dataset, gatestrings):
    """
    Compute the named "Dataset" quantities.

    Parameters
    ----------
    qtynames : list of strings
        Names of the quantities to compute.
        
    dataset : DataSet
        Data used to compute the quantity.

    gatestrings : list of tuples or GateString objects
        A list of gatestrings used in the computation of certain quantities.
        
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
    spamLabels = dataset.getSpamLabels()
    for spl in spamLabels:
        per_gatestring_qtys['Exp prob(%s)' % spl] = []
        per_gatestring_qtys['Exp count(%s)' % spl] = []

    if any( [qtyname in per_gatestring_qtys for qtyname in qtynames ] ):
        if gatestrings is None: gatestrings = dataset.keys()
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
    qty = "max logL"; possible_qtys.append(qty)
    if qty in qtynames:
        ret[qty] = ReportableQty( _LF.logL_max(dataset))

    qty = "number of gate strings"; possible_qtys.append(qty)
    if qty in qtynames:
        ret[qty] = ReportableQty( len(dataset) )

    if qtynames[0] is None:
        return possible_qtys + per_gatestring_qtys.keys()
    return ret


def compute_GateSet_Quantity(qtyname, gateset, confidenceRegionInfo=None):
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
    ret = compute_GateSet_Quantities( [qtyname], gateset, confidenceRegionInfo )
    if qtyname is None: return ret
    elif ret.has_key(qtyname): return ret[qtyname]
    else: return None

def compute_GateSet_Quantities(qtynames, gateset, confidenceRegionInfo=None):
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

    def choi_evals(gate):
        choi = _JOps.opWithJamiolkowskiIsomorphism(gate)
        choi_eigvals = _np.linalg.eigvals(choi)
        return _np.array(sorted(choi_eigvals))

    def choi_trace(gate):
        choi = _JOps.opWithJamiolkowskiIsomorphism(gate)
        return _np.trace(choi)

    def decomp_angle(gate):
        decomp = _GateOps.decomposeGateMatrix(gate)
        return decomp.get('pi rotations',0)

    def decomp_decay_diag(gate):
        decomp = _GateOps.decomposeGateMatrix(gate)
        return decomp.get('decay of diagonal rotation terms',0)

    def decomp_decay_offdiag(gate):
        decomp = _GateOps.decomposeGateMatrix(gate)
        return decomp.get('decay of off diagonal rotation terms',0)

    def decomp_cu_angle(gate):
        closestUGateMx = _GateOps.getClosestUnitaryGateMx(gate)
        decomp = _GateOps.decomposeGateMatrix(closestUGateMx)
        return decomp.get('pi rotations',0)

    def decomp_cu_decay_diag(gate):
        closestUGateMx = _GateOps.getClosestUnitaryGateMx(gate)
        decomp = _GateOps.decomposeGateMatrix(closestUGateMx)
        return decomp.get('decay of diagonal rotation terms',0)

    def decomp_cu_decay_offdiag(gate):
        closestUGateMx = _GateOps.getClosestUnitaryGateMx(gate)
        decomp = _GateOps.decomposeGateMatrix(closestUGateMx)
        return decomp.get('decay of off diagonal rotation terms',0)

    def upper_bound_fidelity(gate):
        ubF, ubGateMx = _GateOps.getFidelityUpperBound(gate)
        return ubF

    def closest_UJMx(gate):
        closestUGateMx = _GateOps.getClosestUnitaryGateMx(gate)
        return _JOps.opWithJamiolkowskiIsomorphism(closestUGateMx)
        
    def maximum_fidelity(gate):
        closestUGateMx = _GateOps.getClosestUnitaryGateMx(gate)
        closestUJMx = _JOps.opWithJamiolkowskiIsomorphism(closestUGateMx)
        choi = _JOps.opWithJamiolkowskiIsomorphism(gate)
        return _GateOps.Fidelity(closestUJMx, choi)

    def maximum_trace_dist(gate):
        closestUGateMx = _GateOps.getClosestUnitaryGateMx(gate)
        closestUJMx = _JOps.opWithJamiolkowskiIsomorphism(closestUGateMx)
        return _JOps.JTraceDistance(gate, closestUGateMx)

    def spam_dotprods(rhoVecs, EVecs):
        ret = _np.empty( (len(rhoVecs), len(EVecs)), 'd')
        for i,rhoVec in enumerate(rhoVecs):
            for j,EVec in enumerate(EVecs):
                ret[i,j] = _np.dot(_np.transpose(EVec), rhoVec)
        return ret

    # Spam quantities (computed for all spam vectors at once):
    key = "Spam DotProds"; possible_qtys.append(key)
    if key in qtynames:
        ret[key] = _getSpamQuantity(spam_dotprods, gateset, eps, confidenceRegionInfo)

    # Quantities computed per gate
    for (label,gate) in gateset.iteritems():

        #Gate quantities
        suffixes = ('eigenvalues', 'eigenvectors', 'choi eigenvalues', 'choi trace',
                    'choi matrix in pauli basis', 'decomposition')
        gate_qtys = _OrderedDict( [ ("%s %s" % (label,s), None) for s in suffixes ] )
        possible_qtys += gate_qtys.keys()

        if any( [qtyname in gate_qtys for qtyname in qtynames] ):
            #gate_evals,gate_evecs = _np.linalg.eig(gate)
            evalsQty = _getGateQuantity(_np.linalg.eigvals, gateset, label, eps, confidenceRegionInfo)
            choiQty = _getGateQuantity(_JOps.opWithJamiolkowskiIsomorphism, gateset, label, eps, confidenceRegionInfo) 
            choiEvQty = _getGateQuantity(choi_evals, gateset, label, eps, confidenceRegionInfo) 
            choiTrQty = _getGateQuantity(choi_trace, gateset, label, eps, confidenceRegionInfo) 

            decompDict = _GateOps.decomposeGateMatrix(gate)
            if decompDict['isValid']:
                angleQty = _getGateQuantity(decomp_angle, gateset, label, eps, confidenceRegionInfo) 
                diagQty = _getGateQuantity(decomp_decay_diag, gateset, label, eps, confidenceRegionInfo) 
                offdiagQty = _getGateQuantity(decomp_decay_offdiag, gateset, label, eps, confidenceRegionInfo) 
                errBarDict = { 'pi rotations': angleQty.getErrBar(), 
                               'decay of diagonal rotation terms': diagQty.getErrBar(),
                               'decay of off diagonal rotation terms': offdiagQty.getErrBar() }
                decompQty = ReportableQty(decompDict, errBarDict)
            else:
                decompQty = ReportableQty({})

            gate_qtys[ '%s eigenvalues' % label ]      = evalsQty
            #gate_qtys[ '%s eigenvectors' % label ]     = gate_evecs
            gate_qtys[ '%s choi matrix in pauli basis' % label ] = choiQty
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
                    'closest unitary choi matrix in pauli basis',
                    'closest unitary decomposition')
        closestU_qtys = _OrderedDict( [ ("%s %s" % (label,s), None) for s in suffixes ] )
        possible_qtys += closestU_qtys.keys()
        if any( [qtyname in closestU_qtys for qtyname in qtynames] ):
            ubFQty = _getGateQuantity(upper_bound_fidelity, gateset, label, eps, confidenceRegionInfo) 
            closeUJMxQty = _getGateQuantity(closest_UJMx, gateset, label, eps, confidenceRegionInfo) 
            maxFQty = _getGateQuantity(maximum_fidelity, gateset, label, eps, confidenceRegionInfo) 
            maxJTDQty = _getGateQuantity(maximum_trace_dist, gateset, label, eps, confidenceRegionInfo) 

            closestUGateMx = _GateOps.getClosestUnitaryGateMx(gate)
            decompDict = _GateOps.decomposeGateMatrix(closestUGateMx)
            if decompDict['isValid']:
                angleQty = _getGateQuantity(decomp_cu_angle, gateset, label, eps, confidenceRegionInfo) 
                diagQty = _getGateQuantity(decomp_cu_decay_diag, gateset, label, eps, confidenceRegionInfo) 
                offdiagQty = _getGateQuantity(decomp_cu_decay_offdiag, gateset, label, eps, confidenceRegionInfo) 
                errBarDict = { 'pi rotations': angleQty.getErrBar(), 
                               'decay of diagonal rotation terms': diagQty.getErrBar(),
                               'decay of off diagonal rotation terms': offdiagQty.getErrBar() }
                decompQty = ReportableQty(decompDict, errBarDict)
            else:
                decompQty = ReportableQty({})

            closestU_qtys[ '%s max fidelity with unitary' % label ]                  = maxFQty
            closestU_qtys[ '%s max trace dist with unitary' % label ]                = maxJTDQty
            closestU_qtys[ '%s upper bound on fidelity with unitary' % label ]       = ubFQty
            closestU_qtys[ '%s closest unitary choi matrix in pauli basis' % label ] = closeUJMxQty
            closestU_qtys[ '%s closest unitary decomposition' % label ]              = decompQty

            for qtyname in qtynames:
                if qtyname in closestU_qtys: 
                    ret[qtyname] = closestU_qtys[qtyname]

    if qtynames[0] is None:
        return possible_qtys
    return ret        

    
def compute_GateSet_DataSet_Quantity(qtyname, gateset, dataset, gatestrings):
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

    gatestrings : list of tuples or GateString objects
        A list of gatestrings used in the computation of certain quantities.
        
    Returns
    -------
    ReportableQty
        The quantity requested, or None if quantity could not be computed.
    """
    ret = compute_GateSet_DataSet_Quantities( [qtyname], gateset, dataset, gatestrings )
    if qtyname is None: return ret
    elif ret.has_key(qtyname): return ret[qtyname]
    else: return None

def compute_GateSet_DataSet_Quantities(qtynames, gateset, dataset, gatestrings):
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

    gatestrings : list of tuples or GateString objects
        A list of gatestrings used in the computation of certain quantities.
        
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
    per_gatestring_qtys = _OrderedDict() # OLD qtys: [('logL term diff', []), ('score', [])]
    for spl in gateset.SPAMs.keys(): 
        per_gatestring_qtys['prob(%s) diff' % spl] = []
        per_gatestring_qtys['count(%s) diff' % spl] = []
        per_gatestring_qtys['Est prob(%s)' % spl] = []
        per_gatestring_qtys['Est count(%s)' % spl] = []
        per_gatestring_qtys['gatestring chi2(%s)' % spl] = []

    if any( [qtyname in per_gatestring_qtys for qtyname in qtynames ] ):
        if gatestrings is None: gatestrings = dataset.keys()
        for (i,gs) in enumerate(gatestrings):
            if gs in dataset: # skip gate strings given that are not in dataset
                dsRow = dataset[gs]
            else: continue

            p = gateset.Probs(gs)  
            pExp = { }; N = dsRow.total()
            for spamLabel in p:
                p[spamLabel] = _projectToValidProb( p[spamLabel], tol=1e-10 )
                pExp[spamLabel] = _projectToValidProb( dsRow[spamLabel] / N, tol=1e-10 )
            
            #OLD
            #per_gatestring_qtys['logL term diff'].append(  _LF.logL_term(dsRow, pExp) - _LF.logL_term(dsRow, p)  )
            #per_gatestring_qtys['score'].append(  (_LF.logL_term(dsRow, pExp) - _LF.logL_term(dsRow, p)) / N  )

            for spamLabel in p:
                per_gatestring_qtys['prob(%s) diff' % spamLabel].append( abs(p[spamLabel] - pExp[spamLabel]) )
                per_gatestring_qtys['count(%s) diff' % spamLabel].append( int( round(p[spamLabel] * N) - dsRow[spamLabel]) )
                per_gatestring_qtys['Est prob(%s)' % spamLabel].append( p[spamLabel] )
                per_gatestring_qtys['Est count(%s)' % spamLabel].append( int(round(p[spamLabel] * N)) )
                per_gatestring_qtys['gatestring chi2(%s)' % spamLabel].append( _AT.ChiSqFunc( N, p[spamLabel], pExp[spamLabel], 1e-4 ) )
                        
        for qtyname in qtynames:
            if qtyname in per_gatestring_qtys:
                ret[qtyname] = ReportableQty( per_gatestring_qtys[qtyname] )

    #Quantities which take a single value for a given gateset and dataset
    qty = "logL"; possible_qtys.append(qty)
    if qty in qtynames:
        ret[qty] = ReportableQty( _LF.logL(gateset, dataset) )

    qty = "logL diff"; possible_qtys.append(qty)
    if qty in qtynames:
        ret[qty] = ReportableQty( _LF.logL_max(dataset) - _LF.logL(gateset, dataset) )

    qty = "chi2"; possible_qtys.append(qty)        
    if qty in qtynames:
        ret[qty] = ReportableQty( _AT.TotalChiSquared( dataset, gateset, minProbClipForWeighting=1e-4) )

    #Quantities which take a single value per spamlabel for a given gateset and dataset
    #for spl in gateset.SPAMs.keys(): 
    #    qty = "chi2(%s)" % spl; possible_qtys.append(qty)        
    #    if qty in qtynames:
    #        ret[qty] = _AT.TotalChiSquared( dataset, gateset, minProbClipForWeighting=1e-4)

    if qtynames[0] is None:
        return possible_qtys + per_gatestring_qtys.keys()
    return ret


def compute_GateSet_GateSet_Quantity(qtyname, gateset1, gateset2, confidenceRegionInfo=None):
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
    ret = compute_GateSet_GateSet_Quantities( [qtyname], gateset1, gateset2, confidenceRegionInfo)
    if qtyname is None: return ret
    elif ret.has_key(qtyname): return ret[qtyname]
    else: return None

def compute_GateSet_GateSet_Quantities(qtynames, gateset1, gateset2, confidenceRegionInfo=None):
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

    for gateLabel in gateset1:
        if gateLabel not in gateset2:
            raise ValueError("%s gate is missing from second gateset - cannot compare gatesets", gateLabel)
    for gateLabel in gateset2:
        if gateLabel not in gateset1:
            raise ValueError("%s gate is missing from first gateset - cannot compare gatesets", gateLabel)

    ### per gate quantities           
    #############################################
    for gateLabel in gateset1:

        key = '%s fidelity' % gateLabel; possible_qtys.append(key)
        key2 = '%s infidelity' % gateLabel; possible_qtys.append(key)
        if key in qtynames or key2 in qtynames:

            def process_fidelity(gate): #Note: default 'gm' basis
                return _JOps.ProcessFidelity(gate, gateset2[gateLabel]) #vary elements of gateset1 (assume gateset2 is fixed)

            #print "DEBUG: fidelity(%s)" % gateLabel
            FQty = _getGateQuantity(process_fidelity, gateset1, gateLabel,
                                    eps, confidenceRegionInfo) 

            InFQty = ReportableQty( 1.0-FQty.getValue(), FQty.getErrBar() )
            if key in qtynames: ret[key] = FQty
            if key2 in qtynames: ret[key2] = InFQty

        key = '%s closest unitary fidelity' % gateLabel; possible_qtys.append(key)
        if key in qtynames:
            
            #Note: default 'gm' basis
            def closest_unitary_fidelity(gate): # assume vary gateset1, gateset2 fixed
                decomp1 = _GateOps.decomposeGateMatrix(gate)
                decomp2 = _GateOps.decomposeGateMatrix(gateset2[gateLabel])

                if decomp1['isUnitary']:
                    closestUGateMx1 = gate
                else: closestUGateMx1 = _GateOps.getClosestUnitaryGateMx(gate)
    
                if decomp2['isUnitary']:
                    closestUGateMx2 = gateset2[gateLabel] 
                else: closestUGateMx2 = _GateOps.getClosestUnitaryGateMx(gateset2[gateLabel])
            
                closeChoi1 = _JOps.opWithJamiolkowskiIsomorphism(closestUGateMx1)
                closeChoi2 = _JOps.opWithJamiolkowskiIsomorphism(closestUGateMx2)
                return _GateOps.Fidelity(closeChoi1,closeChoi2)

            ret[key] = _getGateQuantity(closest_unitary_fidelity, gateset1, gateLabel, eps, confidenceRegionInfo) 

        key = "%s Frobenius diff" % gateLabel; possible_qtys.append(key)
        if key in qtynames: 
            def fro_diff(gate): # assume vary gateset1, gateset2 fixed
                return _MatrixOps.frobeniusNorm(gate-gateset2[gateLabel])
            #print "DEBUG: frodist(%s)" % gateLabel
            ret[key] = _getGateQuantity(fro_diff, gateset1, gateLabel, eps, confidenceRegionInfo) 

        key = "%s Jamiolkowski trace dist" % gateLabel; possible_qtys.append(key)
        if key in qtynames: 
            def jt_diff(gate): # assume vary gateset1, gateset2 fixed
                return _JOps.JTraceDistance(gate,gateset2[gateLabel]) #Note: default 'gm' basis
            #print "DEBUG: jtdist(%s)" % gateLabel
            ret[key] = _getGateQuantity(jt_diff, gateset1, gateLabel, eps, confidenceRegionInfo) 

        key = '%s diamond norm' % gateLabel; possible_qtys.append(key)
        if key in qtynames:

            def half_diamond_norm(gate):
                return 0.5 * _GateOps.DiamondNorm(gate, gateset2[gateLabel]) #Note: default 'gm' basis
                  #vary elements of gateset1 (assume gateset2 is fixed)

            try:
                ret[key] = _getGateQuantity(half_diamond_norm, gateset1, gateLabel,
                                            eps, confidenceRegionInfo) 
            except ImportError: #if failed to import cvxpy (probably b/c it's not installed)
                ret[key] = ReportableQty(_np.nan) # report NAN for diamond norms


    ###  per gateset quantities
    #############################################
    key = "Gateset Frobenius diff"; possible_qtys.append(key)
    if key in qtynames: ret[key] = ReportableQty( gateset1.diff_Frobenius(gateset2) )

    key = "Max Jamiolkowski trace dist"; possible_qtys.append(key)
    if key in qtynames: ret[key] = ReportableQty( max( [ _JOps.JTraceDistance(gateset1[l],gateset2[l]) for l in gateset1 ] ) )

 
    #Special case: when qtyname is None then return a list of all possible names that can be computed
    if qtynames[0] is None: 
        return possible_qtys
    return ret
