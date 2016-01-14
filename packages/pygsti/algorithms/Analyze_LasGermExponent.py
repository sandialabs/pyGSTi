""" End-to-end Analysis functions using germ^exp strings."""
import GST as _GST
import _Analyze_Base as _base
from MakeLists_LasGermExponent import *

def doLSGSTAnalysis(dataFilenameOrSet, targetGateFilenameOrSet,
                    rhoStrsListOrFilename, EStrsListOrFilename, germsListOrFilename, Lvalues,
                    gateLabelsInStrings=None, weightsDict=None, rhoEPairs=None, constrainToTP=False,
                    gaugeOptToCPTP=False, gaugeOptRatio=1e-4, advancedOptions={}):
    """
    Perform end-to-end LGST analysis using Ls and germs, with L as a germ exponent.

    Constructs gate strings by repeating the germ strings an integer number of
    times given by Lvalues.  The LGST estimate of the gates is computed,
    gauge optimized, and then and used as the seed for LSGST.

    LSGST is iterated len(Lvalues) times with successively larger sets of gate
    strings.  On the i-th iteration, the germs repeated Lvalues[i] times are 
    included in the growing set of strings used by LSGST.  

    Once computed, the LSGST gate set estimates are gauge optimized to the
    CPTP space (if gaugeOptToCPTP == True) and then to the target gate set
    (using gaugeOptRatio). A Results object is returned, which encapsulates the
    input and outputs of this GST analysis, and can to generate final end-user
    output such as reports and presentations.

    Parameters
    ----------
    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (in text format).

    targetGateFilenameOrSet : GateSet or string
        The target gate set, specified either directly or by the filename of a 
        gateset file (text format).

    rhoStrsListOrFilename : (list of GateStrings) or string
        The state preparation fiducial gate strings, specified either directly 
        or by the filename of a gate string list file (text format).

    EStrsListOrFilename : (list of GateStrings) or string or None
        The measurement fiducial gate strings, specified either directly 
        or by the filename of a gate string list file (text format).  If None,
        then use the same strings as specified by rhoStrsListOrFilename.

    germsListOrFilename : (list of GateStrings) or string
        The germ gate strings, specified either directly or by the filename of a
        gate string list file (text format).

    Lvalues : list of ints
        List of the integers, one per LGST iteration, which exponentiate each of
        germs.  The list of gate strings for the i-th LSGST iteration includes
        the germs exponentiated to the L-values *up to* and including the i-th one.

    gateLabelsInStrings : list or tuple
        A list or tuple of the gate labels to use when generating the sets of
        gate strings used in LSGST iterations.  If None, then the gate labels
        of the target gateset will be used.  This option is useful if you 
        only want to include a *subset* of the available gates in the LSGST
        strings (e.g. leaving out the identity gate).

    weightsDict : dict, optional
        A dictionary with keys == gate strings and values == multiplicative scaling 
        factor for the corresponding gate string. The default is no weight scaling at all.

    rhoEPairs : list of 2-tuples, optional
        Specifies a subset of all rhoStr,EStr string pairs to be used in this
        analysis.  Each element of rhoEPairs is a (iRhoStr, iEStr) 2-tuple of integers,
        which index a string within the state preparation and measurement fiducial
        strings respectively.

    constrainToTP : bool, optional
        Whether to constrain GST to trace-preserving gatesets.

    gaugeOptToCPTP : bool, optional
        If True, resulting gate sets are first optimized to CPTP and then to the target.
        If False, gate sets are only optimized to the target gate set.
        
    gaugeOptRatio : float, optional
        The ratio spamWeight/gateWeight used for gauge optimizing to the target gate set.

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of the
        objective function.   The 'verbosity' option is an integer specifying the level
        of detail printed to stdout during the GST calculation.
        
    Returns
    -------
    Results
    """
    return _base.doLongSequenceGST("chi2", dataFilenameOrSet, targetGateFilenameOrSet, rhoStrsListOrFilename,
                                   EStrsListOrFilename, germsListOrFilename, Lvalues, gateLabelsInStrings,
                                   weightsDict, make_lsgst_lists_asymmetric_fids, _GST.GateStringTools.repeat,
                                   rhoEPairs, constrainToTP, gaugeOptToCPTP, gaugeOptRatio, advancedOptions)




def doMLEAnalysis(dataFilenameOrSet, targetGateFilenameOrSet, 
                  rhoStrsListOrFilename, EStrsListOrFilename, germsListOrFilename, Lvalues,
                  gateLabelsInStrings=None, weightsDict=None, rhoEPairs=None, constrainToTP=False,
                  gaugeOptToCPTP=False, gaugeOptRatio=1e-4, advancedOptions={}):
    """
    Perform end-to-end maximum-likelihood  analysis using Ls and germs, with L as a maximum length.

    Constructs gate strings by repeating the germ strings an integer number of
    times given by Lvalues.  The LGST estimate of the gates is computed,
    gauge optimized, and then and used as the seed for MLEGST.

    MLEGST is iterated len(Lvalues) times with successively larger sets of gate
    strings.  On the i-th iteration, the germs repeated Lvalues[i] times are 
    included in the growing set of strings used by MLEGST.

    Once computed, the MLEGST gate set estimates are gauge optimized to the
    CPTP space (if gaugeOptToCPTP == True) and then to the target gate set
    (using gaugeOptRatio). A Results object is returned, which encapsulates the
    input and outputs of this GST analysis, and can to generate final end-user
    output such as reports and presentations.    


    Parameters
    ----------
    dataFilenameOrSet : DataSet or string
        The data set object to use for the analysis, specified either directly
        or by the filename of a dataset file (in text format).

    targetGateFilenameOrSet : GateSet or string
        The target gate set, specified either directly or by the filename of a 
        gateset file (text format).

    rhoStrsListOrFilename : (list of GateStrings) or string
        The state preparation fiducial gate strings, specified either directly 
        or by the filename of a gate string list file (text format).

    EStrsListOrFilename : (list of GateStrings) or string or None
        The measurement fiducial gate strings, specified either directly 
        or by the filename of a gate string list file (text format).  If None,
        then use the same strings as specified by rhoStrsListOrFilename.

    germsListOrFilename : (list of GateStrings) or string
        The germ gate strings, specified either directly or by the filename of a
        gate string list file (text format).

    Lvalues : list of ints
        List of the integers, one per LGST iteration, which exponentiate each of
        germs.  The list of gate strings for the i-th LSGST iteration includes
        the germs exponentiated to the L-values *up to* and including the i-th one.

    gateLabelsInStrings : list or tuple
        A list or tuple of the gate labels to use when generating the sets of
        gate strings used in LSGST iterations.  If None, then the gate labels
        of the target gateset will be used.  This option is useful if you 
        only want to include a *subset* of the available gates in the LSGST
        strings (e.g. leaving out the identity gate).

    weightsDict : dict, optional
        A dictionary with keys == gate strings and values == multiplicative scaling 
        factor for the corresponding gate string. The default is no weight scaling at all.

    rhoEPairs : list of 2-tuples, optional
        Specifies a subset of all rhoStr,EStr string pairs to be used in this
        analysis.  Each element of rhoEPairs is a (iRhoStr, iEStr) 2-tuple of integers,
        which index a string within the state preparation and measurement fiducial
        strings respectively.

    constrainToTP : bool, optional
        Whether to constrain GST to trace-preserving gatesets.

    gaugeOptToCPTP : bool, optional
        If True, resulting gate sets are first optimized to CPTP and then to the target.
        If False, gate sets are only optimized to the target gate set.
        
    gaugeOptRatio : float, optional
        The ratio spamWeight/gateWeight used for gauge optimizing to the target gate set.

    advancedOptions : dict, optional
        Specifies advanced options most of which deal with numerical details of the
        objective function.   The 'verbosity' option is an integer specifying the level
        of detail printed to stdout during the GST calculation.

        
    Returns
    -------
    Results
    """
    return _base.doLongSequenceGST("logL", dataFilenameOrSet, targetGateFilenameOrSet, rhoStrsListOrFilename,
                                   EStrsListOrFilename, germsListOrFilename, Lvalues, gateLabelsInStrings,
                                   weightsDict, make_lsgst_lists_asymmetric_fids, _GST.GateStringTools.repeat,
                                   rhoEPairs, constrainToTP, gaugeOptToCPTP, gaugeOptRatio, advancedOptions)
