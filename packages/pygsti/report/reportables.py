"""Functions which compute named quantities for GateSets and Datasets."""
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
"""
Named quantities as well as their confidence-region error bars are
 computed by the functions in this module. These quantities are
 used primarily in reports, so we refer to these quantities as
 "reportables".
"""
import numpy as _np
import scipy.linalg as _spl
import warnings as _warnings

from .. import tools as _tools
from .. import algorithms as _alg
from ..baseobjs import Basis as _Basis
from ..objects.reportableqty import ReportableQty as _ReportableQty
from ..objects import gatesetfunction as _gsf

try:
    import sys as _sys
    if _sys.version_info < (3, 0):
        #Attempt "safe" import of cvxpy so that pickle isn't messed up...
        import pickle as _pickle
        p = _pickle.Pickler.dispatch.copy()
        import cvxpy as _cvxpy
        _pickle.Pickler.dispatch = p
    else:
        import cvxpy as _cvxpy
except ImportError:
    _cvxpy = None
    
FINITE_DIFF_EPS = 1e-7

def _nullFn(*arg):
    return None

def _projectToValidProb(p, tol=1e-9):
    if p < tol: return tol
    if p > 1-tol: return 1-tol
    return p

def _make_reportable_qty_or_dict(f0, df=None, nonMarkovianEBs=False):
    """ Just adds special processing with f0 is a dict, where we 
        return a dict or ReportableQtys rather than a single
        ReportableQty of the dict.
    """
    if isinstance(f0,dict):
        #special processing for dict -> df is dict of error bars
        # and we return a dict of ReportableQtys
        if df:
            return { ky: _ReportableQty(f0[ky], df[ky], nonMarkovianEBs) for ky in f0 }
        else:
            return { ky: _ReportableQty(f0[ky], None, False) for ky in f0 }
    else:
        return _ReportableQty(f0, df, nonMarkovianEBs)

def evaluate(gatesetFn, cri=None, verbosity=0):
    """ 
    Evaluate a GateSetFunction object using confidence region information

    Parameters
    ----------
    gatesetFn : GateSetFunction
        The function to evaluate

    cri : ConfidenceRegionFactoryView, optional
        View for computing confidence intervals.

    verbosity : int, optional
        Amount of detail to print to stdout.

    Returns
    -------
    ReportableQty or dict
        If `gatesetFn` does returns a dict of ReportableQty objects, otherwise
        a single ReportableQty.
    """
    if gatesetFn is None: # so you can set fn to None when they're missing (e.g. diamond norm)
        return _ReportableQty(_np.nan)
    
    if cri:
        nmEBs = bool(cri.get_errobar_type() == "non-markovian")
        df, f0 =  cri.get_fn_confidence_interval(
            gatesetFn, returnFnVal=True,
            verbosity=verbosity)
        return _make_reportable_qty_or_dict(f0, df, nmEBs)
    else:
        return _make_reportable_qty_or_dict( gatesetFn.evaluate(gatesetFn.base_gateset) )


def spam_dotprods(rhoVecs, povms):
    """SPAM dot products (concatenates POVMS)"""
    nEVecs = sum(len(povm) for povm in povms)
    ret = _np.empty( (len(rhoVecs), nEVecs), 'd')
    for i,rhoVec in enumerate(rhoVecs):
        j = 0
        for povm in povms:
            for EVec in povm.values():
                ret[i,j] = _np.dot(_np.transpose(EVec), rhoVec); j += 1
    return ret
Spam_dotprods = _gsf.spamfn_factory(spam_dotprods) #init args == (gateset)


def choi_matrix(gate, mxBasis):
    """Choi matrix"""
    return _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)
Choi_matrix = _gsf.gatefn_factory(choi_matrix) # init args == (gateset, gateLabel)


def choi_evals(gate, mxBasis):
    """Choi matrix eigenvalues"""
    choi = _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)
    choi_eigvals = _np.linalg.eigvals(choi)
    return _np.array(sorted(choi_eigvals))
Choi_evals = _gsf.gatefn_factory(choi_evals) # init args == (gateset, gateLabel)


def choi_trace(gate, mxBasis):
    """Trace of the Choi matrix"""
    choi = _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)
    return _np.trace(choi)
Choi_trace = _gsf.gatefn_factory(choi_trace) # init args == (gateset, gateLabel)



class Gate_eigenvalues(_gsf.GateSetFunction):
    """Gate eigenvalues"""
    def __init__(self, gateset, gatelabel):
        self.gatelabel = gatelabel
        _gsf.GateSetFunction.__init__(self, gateset, ["gate:" + str(gatelabel)])
            
    def evaluate(self, gateset):
        """Evaluate at `gateset`"""
        evals,evecs = _np.linalg.eig(gateset.gates[self.gatelabel])
        
        ev_list = list(enumerate(evals))
        ev_list.sort(key=lambda tup:abs(tup[1]), reverse=True)
        indx,evals = zip(*ev_list)
        evecs = evecs[:,indx] #sort evecs according to evals

        self.G0 = gateset.gates[self.gatelabel]
        self.evals = _np.array(evals)
        self.evecs = evecs
        self.inv_evecs = _np.linalg.inv(evecs)

        return self.evals

    def evaluate_nearby(self, nearby_gateset):
        """Evaluate at a nearby gate set"""
        #avoid calling minweight_match again
        dMx = nearby_gateset.gates[self.gatelabel] - self.G0
        #evalsM = evals0 + Uinv * (M-M0) * U
        return _np.array( [ self.evals[k] + _np.dot(self.inv_evecs[k,:], _np.dot(dMx, self.evecs[:,k]))
                            for k in range(dMx.shape[0])] )
    # ref for eigenvalue derivatives: https://www.win.tue.nl/casa/meetings/seminar/previous/_abstract051019_files/Presentation.pdf


#def gate_eigenvalues(gate, mxBasis):
#    return _np.array(sorted(_np.linalg.eigvals(gate),
#                            key=lambda ev: abs(ev), reverse=True))
#Gate_eigenvalues = _gsf.gatefn_factory(gate_eigenvalues)
## init args == (gateset, gateLabel)


#Example....
#class Gatestring_eigenvalues(_gsf.GateSetFunction):
#    def __init__(self, gatesetA, gatesetB, gatestring):
#        self.gatestring = gatestring
#        self.B = gatesetB.product(gatestring)
#        self.evB = _np.linalg.eigvals(B)
#        self.gatestring = gatestring
#        _gsf.GateSetFunction.__init__(self, gatesetA, ["all"])
#            
#    def evaluate(self, gateset):
#        Mx = gateset.product(self.gatestring)
#        return _np.array(sorted(_np.linalg.eigvals(),
#                            key=lambda ev: abs(ev), reverse=True))
#
#        A = gateset.product(self.gatestring)
#        evA, evecsA = _np.linalg.eig(A)
#        self.A0, self.evA0, self.evecsA0, self.ievecsA0 = A, evA, evecsA, _np.linalg.inv(evecsA) #save for evaluate_nearby...
#        wts, self.pairs = _tools.minweight_match(evA, self.evB, lambda x,y: abs(x-y), return_pairs=True)
#        return _np.max(wts)
#
#    def evaluate_nearby(self, nearby_gateset):
#        #avoid calling minweight_match again
#        A = nearby_gateset.product(self.gatestring)
#        dA = A - self.A0
#        #evA = _np.linalg.eigvals(A)  # = self.evA0 + U * (A-A0) * Udag
#        evA = _np.array( [ self.evA0 + _np.dot(self.ievecsA0[k,:], _np.dot(dA, self.evecsA0[:,k])) for k in range(dA.shape[0])] )
#        return _np.max( [ abs(evA[i]-self.evB[j]) for i,j in self.pairs ] )


class Gatestring_eigenvalues(_gsf.GateSetFunction):
    """Gate sequence eigenvalues"""
    def __init__(self, gateset, gatestring):
        self.gatestring = gatestring
        _gsf.GateSetFunction.__init__(self, gateset, ["all"])
            
    def evaluate(self, gateset):
        """Evaluate at `gateset`"""
        Mx = gateset.product(self.gatestring)
        evals,evecs = _np.linalg.eig(Mx)
        
        ev_list = list(enumerate(evals))
        ev_list.sort(key=lambda tup:abs(tup[1]), reverse=True)
        indx,evals = zip(*ev_list)
        evecs = evecs[:,indx] #sort evecs according to evals

        self.Mx = Mx
        self.evals = _np.array(evals)
        self.evecs = evecs
        self.inv_evecs = _np.linalg.inv(evecs)

        return self.evals

    def evaluate_nearby(self, nearby_gateset):
        """Evaluate at nearby gate set"""
        #avoid calling minweight_match again
        Mx = nearby_gateset.product(self.gatestring)
        dMx = Mx - self.Mx
        #evalsM = evals0 + Uinv * (M-M0) * U
        return _np.array( [ self.evals[k] + _np.dot(self.inv_evecs[k,:], _np.dot(dMx, self.evecs[:,k]))
                            for k in range(dMx.shape[0])] )
    # ref for eigenvalue derivatives: https://www.win.tue.nl/casa/meetings/seminar/previous/_abstract051019_files/Presentation.pdf


#def gatestring_eigenvalues(gateset, gatestring):
#    return _np.array(sorted(_np.linalg.eigvals(gateset.product(gatestring)),
#                            key=lambda ev: abs(ev), reverse=True))
#Gatestring_eigenvalues = _gsf.gatesetfn_factory(gatestring_eigenvalues)
## init args == (gateset, gatestring)

  
def rel_gatestring_eigenvalues(gatesetA, gatesetB, gatestring):
    """Eigenvalues of dot(productB(gatestring)^-1, productA(gatestring))"""
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    rel_gate = _np.dot(_np.linalg.inv(B), A) # "relative gate" == target^{-1} * gate
    return _np.linalg.eigvals(rel_gate)
Rel_gatestring_eigenvalues = _gsf.gatesetfn_factory(rel_gatestring_eigenvalues)
# init args == (gatesetA, gatesetB, gatestring) 


def gatestring_fro_diff(gatesetA, gatesetB, gatestring):
    """ Frobenius distance btwn productA(gatestring) and productB(gatestring)"""
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return fro_diff(A,B,gatesetB.basis)
Gatestring_fro_diff = _gsf.gatesetfn_factory(gatestring_fro_diff)
# init args == (gatesetA, gatesetB, gatestring)

def gatestring_entanglement_infidelity(gatesetA, gatesetB, gatestring):
    """ Entanglement infidelity btwn productA(gatestring)
        and productB(gatestring)"""
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return entanglement_infidelity(A,B,gatesetB.basis)
Gatestring_entanglement_infidelity = _gsf.gatesetfn_factory(gatestring_entanglement_infidelity)
# init args == (gatesetA, gatesetB, gatestring)

def gatestring_avg_gate_infidelity(gatesetA, gatesetB, gatestring):
    """ Average gate infidelity between productA(gatestring) 
        and productB(gatestring)"""
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return avg_gate_infidelity(A,B,gatesetB.basis)
Gatestring_avg_gate_infidelity = _gsf.gatesetfn_factory(gatestring_avg_gate_infidelity)
  # init args == (gatesetA, gatesetB, gatestring)


def gatestring_jt_diff(gatesetA, gatesetB, gatestring):
    """ Jamiolkowski trace distance between productA(gatestring) 
        and productB(gatestring)"""
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return jt_diff(A, B, gatesetB.basis)
Gatestring_jt_diff = _gsf.gatesetfn_factory(gatestring_jt_diff)
# init args == (gatesetA, gatesetB, gatestring)

if _cvxpy:

    class Gatestring_half_diamond_norm(_gsf.GateSetFunction):
        """ 1/2 diamond norm of difference between productA(gatestring)
            and productB(gatestring)"""
        def __init__(self, gatesetA, gatesetB, gatestring):
            self.gatestring = gatestring
            self.B = gatesetB.product(gatestring)
            self.d = int(round(_np.sqrt(gatesetA.dim)))
            _gsf.GateSetFunction.__init__(self, gatesetA, ["all"])
                
        def evaluate(self, gateset):
            """Evaluate this function at `gateset`"""
            A = gateset.product(self.gatestring)
            dm, W = _tools.diamonddist(A, self.B, gateset.basis,
                                       return_x=True)
            self.W = W
            return 0.5*dm
    
        def evaluate_nearby(self, nearby_gateset):
            """Evaluate at a nearby gate set"""
            mxBasis = nearby_gateset.basis
            JAstd = self.d * _tools.fast_jamiolkowski_iso_std(
                nearby_gateset.product(self.gatestring), mxBasis)
            JBstd = self.d * _tools.fast_jamiolkowski_iso_std(self.B, mxBasis)
            Jt = (JBstd-JAstd).T
            return 0.5*_np.trace( Jt.real * self.W.real + Jt.imag * self.W.imag)

    #def gatestring_half_diamond_norm(gatesetA, gatesetB, gatestring):
    #    A = gatesetA.product(gatestring) # "gate"
    #    B = gatesetB.product(gatestring) # "target gate"
    #    return half_diamond_norm(A, B, gatesetB.basis)
    #Gatestring_half_diamond_norm = _gsf.gatesetfn_factory(gatestring_half_diamond_norm)
    #  # init args == (gatesetA, gatesetB, gatestring)

else:
    gatestring_half_diamond_norm = None
    Gatestring_half_diamond_norm = _nullFn


def gatestring_nonunitary_entanglement_infidelity(gatesetA, gatesetB, gatestring):
    """ Nonunitary entanglement infidelity between productA(gatestring) 
        and productB(gatestring)"""    
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return nonunitary_entanglement_infidelity(A,B,gatesetB.basis)
Gatestring_nonunitary_entanglement_infidelity = _gsf.gatesetfn_factory(gatestring_nonunitary_entanglement_infidelity)
  # init args == (gatesetA, gatesetB, gatestring)


def gatestring_nonunitary_avg_gate_infidelity(gatesetA, gatesetB, gatestring):
    """ Nonunitary average gate infidelity between productA(gatestring) 
        and productB(gatestring)"""    
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return nonunitary_avg_gate_infidelity(A,B,gatesetB.basis)
Gatestring_nonunitary_avg_gate_infidelity = _gsf.gatesetfn_factory(gatestring_nonunitary_avg_gate_infidelity)
  # init args == (gatesetA, gatesetB, gatestring)


def gatestring_eigenvalue_entanglement_infidelity(gatesetA, gatesetB, gatestring):
    """ Eigenvalue entanglement infidelity between productA(gatestring) 
        and productB(gatestring)"""    
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return eigenvalue_entanglement_infidelity(A,B,gatesetB.basis)
Gatestring_eigenvalue_entanglement_infidelity = _gsf.gatesetfn_factory(gatestring_eigenvalue_entanglement_infidelity)
  # init args == (gatesetA, gatesetB, gatestring)


def gatestring_eigenvalue_avg_gate_infidelity(gatesetA, gatesetB, gatestring):
    """ Eigenvalue average gate infidelity between productA(gatestring) 
        and productB(gatestring)"""    
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return eigenvalue_avg_gate_infidelity(A,B,gatesetB.basis)
Gatestring_eigenvalue_avg_gate_infidelity = _gsf.gatesetfn_factory(gatestring_eigenvalue_avg_gate_infidelity)
  # init args == (gatesetA, gatesetB, gatestring)

def gatestring_eigenvalue_nonunitary_entanglement_infidelity(gatesetA, gatesetB, gatestring):
    """ Eigenvalue nonunitary entanglement infidelity between 
        productA(gatestring) and productB(gatestring)"""    
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return eigenvalue_nonunitary_entanglement_infidelity(A,B,gatesetB.basis)
Gatestring_eigenvalue_nonunitary_entanglement_infidelity = _gsf.gatesetfn_factory(gatestring_eigenvalue_nonunitary_entanglement_infidelity)
  # init args == (gatesetA, gatesetB, gatestring)


def gatestring_eigenvalue_nonunitary_avg_gate_infidelity(gatesetA, gatesetB, gatestring):
    """ Eigenvalue nonunitary average gate infidelity between 
        productA(gatestring) and productB(gatestring)"""    
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return eigenvalue_nonunitary_avg_gate_infidelity(A,B,gatesetB.basis)
Gatestring_eigenvalue_nonunitary_avg_gate_infidelity = _gsf.gatesetfn_factory(gatestring_eigenvalue_nonunitary_avg_gate_infidelity)
  # init args == (gatesetA, gatesetB, gatestring)

  
def gatestring_eigenvalue_diamondnorm(gatesetA, gatesetB, gatestring):
    """ Eigenvalue diamond distance between 
        productA(gatestring) and productB(gatestring)"""    
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return eigenvalue_diamondnorm(A,B,gatesetB.basis)
Gatestring_eigenvalue_diamondnorm = _gsf.gatesetfn_factory(gatestring_eigenvalue_diamondnorm)
# init args == (gatesetA, gatesetB, gatestring)


def gatestring_eigenvalue_nonunitary_diamondnorm(gatesetA, gatesetB, gatestring):
    """ Eigenvalue nonunitary diamond distance between 
        productA(gatestring) and productB(gatestring)"""    
    A = gatesetA.product(gatestring) # "gate"
    B = gatesetB.product(gatestring) # "target gate"
    return eigenvalue_nonunitary_diamondnorm(A,B,gatesetB.basis)
Gatestring_eigenvalue_nonunitary_diamondnorm = _gsf.gatesetfn_factory(gatestring_eigenvalue_nonunitary_diamondnorm)
# init args == (gatesetA, gatesetB, gatestring)


def povm_entanglement_infidelity(gatesetA, gatesetB, povmlbl):
    """ 
    POVM entanglement infidelity between `gatesetA` and `gatesetB`, equal to 
    `1 - entanglement_fidelity(POVM_MAP)` where `POVM_MAP` is the extension
    of the POVM from the classical space of k-outcomes to the space of
    (diagonal) k by k density matrices.
    """
    return 1.0 - _tools.povm_fidelity(gatesetA, gatesetB, povmlbl)
POVM_entanglement_infidelity = _gsf.povmfn_factory(povm_entanglement_infidelity)
# init args == (gateset1, gatesetB, povmlbl)

def povm_jt_diff(gatesetA, gatesetB, povmlbl):
    """ 
    POVM Jamiolkowski trace distance between `gatesetA` and `gatesetB`, equal to
    `Jamiolkowski_trace_distance(POVM_MAP)` where `POVM_MAP` is the extension
    of the POVM from the classical space of k-outcomes to the space of
    (diagonal) k by k density matrices.
    """
    return _tools.povm_jtracedist(gatesetA, gatesetB, povmlbl)
POVM_jt_diff = _gsf.povmfn_factory(povm_jt_diff)
# init args == (gateset1, gatesetB, povmlbl)

if _cvxpy:

    def povm_half_diamond_norm(gatesetA, gatesetB, povmlbl):
        """ 
        Half the POVM diamond distance between `gatesetA` and `gatesetB`, equal
        to `half_diamond_dist(POVM_MAP)` where `POVM_MAP` is the extension
        of the POVM from the classical space of k-outcomes to the space of
        (diagonal) k by k density matrices.
        """
        return 0.5 * _tools.povm_diamonddist(gatesetA, gatesetB, povmlbl)
    POVM_half_diamond_norm = _gsf.povmfn_factory(povm_half_diamond_norm)
else:
    povm_half_diamond_norm = None
    POVM_half_diamond_norm = _nullFn



def decomposition(gate):
    """
    DEPRECATED: Decompose a 1Q `gate` into rotations about axes.
    """
    decompDict = _tools.decompose_gate_matrix(gate)
    if decompDict['isValid']:
        #angleQty   = decompDict.get('pi rotations',0)
        #diagQty    = decompDict.get('decay of diagonal rotation terms',0)
        #offdiagQty = decompDict.get('decay of off diagonal rotation terms',0)
        errBarDict = { 'pi rotations': None,
                       'decay of diagonal rotation terms': None,
                       'decay of off diagonal rotation terms': None }
        return _ReportableQty(decompDict, errBarDict)
    else:
        return _ReportableQty({})

def upper_bound_fidelity(gate, mxBasis):
    """ Upper bound on entanglement fidelity """
    return _tools.fidelity_upper_bound(gate)[0]
Upper_bound_fidelity = _gsf.gatefn_factory(upper_bound_fidelity)
# init args == (gateset, gateLabel)


def closest_ujmx(gate, mxBasis):
    """ Jamiolkowski state of closest unitary to `gate` """
    closestUGateMx = _alg.find_closest_unitary_gatemx(gate)
    return _tools.jamiolkowski_iso(closestUGateMx, mxBasis, mxBasis)
Closest_ujmx = _gsf.gatefn_factory(closest_ujmx)
# init args == (gateset, gateLabel)


def maximum_fidelity(gate, mxBasis):
    """ Fidelity between `gate` and its closest unitary"""
    closestUGateMx = _alg.find_closest_unitary_gatemx(gate)
    closestUJMx = _tools.jamiolkowski_iso(closestUGateMx, mxBasis, mxBasis)
    choi = _tools.jamiolkowski_iso(gate, mxBasis, mxBasis)
    return _tools.fidelity(closestUJMx, choi)
Maximum_fidelity = _gsf.gatefn_factory(maximum_fidelity)
# init args == (gateset, gateLabel)


def maximum_trace_dist(gate, mxBasis):
    """ Jamiolkowski trace distance between `gate` and its closest unitary"""
    closestUGateMx = _alg.find_closest_unitary_gatemx(gate)
    #closestUJMx = _tools.jamiolkowski_iso(closestUGateMx, mxBasis, mxBasis)
    _tools.jamiolkowski_iso(closestUGateMx, mxBasis, mxBasis)
    return _tools.jtracedist(gate, closestUGateMx)
Maximum_trace_dist = _gsf.gatefn_factory(maximum_trace_dist)
# init args == (gateset, gateLabel)


def angles_btwn_rotn_axes(gateset):
    """
    Array of angles between the rotation axes of the gates of `gateset`.
    
    Returns
    -------
    numpy.ndarray
        Of size `(nGates,nGate)` where `nGates=len(gateset.gates)`
    """
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
Angles_btwn_rotn_axes = _gsf.gatesetfn_factory(angles_btwn_rotn_axes)
# init args == (gateset)


def entanglement_fidelity(A, B, mxBasis):
    """Entanglement fidelity between A and B"""
    return _tools.entanglement_fidelity(A, B, mxBasis)
Entanglement_fidelity = _gsf.gatesfn_factory(entanglement_fidelity)
# init args == (gateset1, gateset2, gateLabel)


def entanglement_infidelity(A, B, mxBasis):
    """Entanglement infidelity between A and B"""
    return 1 - _tools.entanglement_fidelity(A, B, mxBasis)
Entanglement_infidelity = _gsf.gatesfn_factory(entanglement_infidelity)
# init args == (gateset1, gateset2, gateLabel)


def closest_unitary_fidelity(A, B, mxBasis): # assume vary gateset1, gateset2 fixed
    """Entanglement infidelity between closest unitaries to A and B"""
    decomp1 = _tools.decompose_gate_matrix(A)
    decomp2 = _tools.decompose_gate_matrix(B)

    if decomp1['isUnitary']:
        closestUGateMx1 = A
    else: closestUGateMx1 = _alg.find_closest_unitary_gatemx(A)

    if decomp2['isUnitary']:
        closestUGateMx2 = B
    else: closestUGateMx2 = _alg.find_closest_unitary_gatemx(A)

    closeChoi1 = _tools.jamiolkowski_iso(closestUGateMx1)
    closeChoi2 = _tools.jamiolkowski_iso(closestUGateMx2)
    return _tools.fidelity(closeChoi1, closeChoi2)
Closest_unitary_fidelity = _gsf.gatesfn_factory(closest_unitary_fidelity)
# init args == (gateset1, gateset2, gateLabel)


def fro_diff(A, B, mxBasis): # assume vary gateset1, gateset2 fixed
    """ Frobenius distance between A and B """
    return _tools.frobeniusdist(A, B)
Fro_diff = _gsf.gatesfn_factory(fro_diff)
# init args == (gateset1, gateset2, gateLabel)


def jt_diff(A, B, mxBasis): # assume vary gateset1, gateset2 fixed
    """ Jamiolkowski trace distance between A and B"""
    return _tools.jtracedist(A, B, mxBasis)
Jt_diff = _gsf.gatesfn_factory(jt_diff)
# init args == (gateset1, gateset2, gateLabel)


if _cvxpy:

    class Half_diamond_norm(_gsf.GateSetFunction):
        """Half the diamond distance bewteen `gatesetA.gates[gateLabel]` and
           `gatesetB.gates[gateLabel]` """
        def __init__(self, gatesetA, gatesetB, gatelabel):
            self.gatelabel = gatelabel
            self.B = gatesetB.gates[gatelabel]
            self.d = int(round(_np.sqrt(gatesetA.dim)))
            _gsf.GateSetFunction.__init__(self, gatesetA, ["gate:"+gatelabel])
                
        def evaluate(self, gateset):
            """Evaluate at `gatesetA = gateset` """
            gl = self.gatelabel
            dm, W = _tools.diamonddist(gateset.gates[gl], self.B, gateset.basis,
                                       return_x=True)
            self.W = W
            return 0.5*dm
    
        def evaluate_nearby(self, nearby_gateset):
            """Evaluates at a nearby gate set"""
            gl = self.gatelabel; mxBasis = nearby_gateset.basis
            JAstd = self.d * _tools.fast_jamiolkowski_iso_std(
                nearby_gateset.gates[gl], mxBasis)
            JBstd = self.d * _tools.fast_jamiolkowski_iso_std(self.B, mxBasis)
            Jt = (JBstd-JAstd).T
            return 0.5*_np.trace( Jt.real * self.W.real + Jt.imag * self.W.imag)

    #def half_diamond_norm(A, B, mxBasis):
    #    return 0.5 * _tools.diamonddist(A, B, mxBasis)
    #Half_diamond_norm = _gsf.gatesfn_factory(half_diamond_norm)
    ## init args == (gateset1, gateset2, gateLabel)

else:
    half_diamond_norm = None
    Half_diamond_norm = _nullFn


def std_unitarity(A,B, mxBasis):
    """ A gauge-invariant quantity that behaves like the unitarity """
    Lambda = _np.dot(A, _np.linalg.inv(B))
    return _tools.unitarity( Lambda, mxBasis )

def eigenvalue_unitarity(A,B):
    """ A gauge-invariant quantity that behaves like the unitarity """
    Lambda = _np.dot(A, _np.linalg.inv(B))
    d2 = Lambda.shape[0]
    lmb = _np.linalg.eigvals(Lambda)
    return (_np.real(_np.vdot(lmb,lmb)) - 1.0) / (d2 - 1.0)
    
def nonunitary_entanglement_infidelity(A, B, mxBasis):
    """ Returns (d^2 - 1)/d^2 * (1 - sqrt(U)), where U is the unitarity of A*B^{-1} """
    d2 = A.shape[0]; U = std_unitarity(A,B,mxBasis)
    return (d2-1.0)/d2 * (1.0 - _np.sqrt(U))
Nonunitary_entanglement_infidelity = _gsf.gatesfn_factory(nonunitary_entanglement_infidelity)
# init args == (gateset1, gateset2, gateLabel)


def nonunitary_avg_gate_infidelity(A, B, mxBasis):
    """ Returns (d - 1)/d * (1 - sqrt(U)), where U is the unitarity of A*B^{-1} """
    d2 = A.shape[0]; d = int(round(_np.sqrt(d2)))
    U = std_unitarity(A,B,mxBasis)
    return (d-1.0)/d * (1.0 - _np.sqrt(U))
Nonunitary_avg_gate_infidelity = _gsf.gatesfn_factory(nonunitary_avg_gate_infidelity)
# init args == (gateset1, gateset2, gateLabel)

def eigenvalue_nonunitary_entanglement_infidelity(A, B, mxBasis):
    """ Returns (d^2 - 1)/d^2 * (1 - sqrt(U)), where U is the eigenvalue-unitarity of A*B^{-1} """
    d2 = A.shape[0]; U = eigenvalue_unitarity(A,B)
    return (d2-1.0)/d2 * (1.0 - _np.sqrt(U))
Eigenvalue_nonunitary_entanglement_infidelity = _gsf.gatesfn_factory(eigenvalue_nonunitary_entanglement_infidelity)
# init args == (gateset1, gateset2, gateLabel)


def eigenvalue_nonunitary_avg_gate_infidelity(A, B, mxBasis):
    """ Returns (d - 1)/d * (1 - sqrt(U)), where U is the eigenvalue-unitarity of A*B^{-1} """
    d2 = A.shape[0]; d = int(round(_np.sqrt(d2)))
    U = eigenvalue_unitarity(A,B)
    return (d-1.0)/d * (1.0 - _np.sqrt(U))
Eigenvalue_nonunitary_avg_gate_infidelity = _gsf.gatesfn_factory(eigenvalue_nonunitary_avg_gate_infidelity)
# init args == (gateset1, gateset2, gateLabel)


def eigenvalue_entanglement_infidelity(A, B, mxBasis):
    """ Eigenvalue entanglement infidelity between A and B """
    d2 = A.shape[0]
    evA = _np.linalg.eigvals(A)
    evB = _np.linalg.eigvals(B)
    mlPl = _np.sum(_tools.minweight_match(evA,evB, lambda x,y: -_np.abs(_np.conjugate(y)*x),
                                 return_pairs=False))
    return 1.0 + mlPl/float(d2)
Eigenvalue_entanglement_infidelity = _gsf.gatesfn_factory(eigenvalue_entanglement_infidelity)
# init args == (gateset1, gateset2, gateLabel)


def eigenvalue_avg_gate_infidelity(A, B, mxBasis):
    """ Eigenvalue average gate infidelity between A and B """
    d2 = A.shape[0]; d = int(round(_np.sqrt(d2)))
    evA = _np.linalg.eigvals(A)
    evB = _np.linalg.eigvals(B)
    mlPl = _np.sum(_tools.minweight_match(evA,evB, lambda x,y: -_np.abs(_np.conjugate(y)*x),
                                 return_pairs=False))
    return (d2 + mlPl)/float(d*(d+1))
Eigenvalue_avg_gate_infidelity = _gsf.gatesfn_factory(eigenvalue_avg_gate_infidelity)
# init args == (gateset1, gateset2, gateLabel)


def eigenvalue_diamondnorm(A, B, mxBasis):
    """ Eigenvalue diamond distance between A and B """
    d2 = A.shape[0]
    evA = _np.linalg.eigvals(A)
    evB = _np.linalg.eigvals(B)
    return (d2-1.0)/d2 * _np.max(_tools.minweight_match(evA,evB, lambda x,y: abs(x-y),
                                                        return_pairs=False))
Eigenvalue_diamondnorm = _gsf.gatesfn_factory(eigenvalue_diamondnorm)
# init args == (gateset1, gateset2, gateLabel)


def eigenvalue_nonunitary_diamondnorm(A, B, mxBasis):
    """ Eigenvalue nonunitary diamond distance between A and B """
    d2 = A.shape[0]
    evA = _np.linalg.eigvals(A)
    evB = _np.linalg.eigvals(B)
    return (d2-1.0)/d2 * _np.max(_tools.minweight_match(evA,evB, lambda x,y: abs(abs(x)-abs(y)),
                                                        return_pairs=False))
Eigenvalue_nonunitary_diamondnorm = _gsf.gatesfn_factory(eigenvalue_nonunitary_diamondnorm)
# init args == (gateset1, gateset2, gateLabel)


def avg_gate_infidelity(A, B, mxBasis):
    """ Returns the average gate infidelity between A and B, where B is the "target" operation."""
    d = _np.sqrt(A.shape[0])
    return _tools.average_gate_infidelity(A,B,mxBasis)
Avg_gate_infidelity = _gsf.gatesfn_factory(avg_gate_infidelity)
# init args == (gateset1, gateset2, gateLabel)



def gateset_gateset_angles_btwn_axes(A, B, mxBasis): #Note: default 'gm' basis
    """ Angle between the rotation axes of A and B (1-qubit gates)"""
    decomp = _tools.decompose_gate_matrix(A)
    decomp2 = _tools.decompose_gate_matrix(B)
    axisOfRotn = decomp.get('axis of rotation', None)
    rotnAngle = decomp.get('pi rotations','X')
    axisOfRotn2 = decomp2.get('axis of rotation', None)
    rotnAngle2 = decomp2.get('pi rotations','X')

    if rotnAngle == 'X' or abs(rotnAngle) < 1e-4 or \
       rotnAngle2 == 'X' or abs(rotnAngle2) < 1e-4:
        return _np.nan

    if axisOfRotn is None or axisOfRotn2 is None:
        return _np.nan

    real_dot =  _np.clip( _np.real(_np.dot(axisOfRotn, axisOfRotn2)), -1.0, 1.0)
    return _np.arccos( abs(real_dot) ) / _np.pi
      #Note: abs() allows axis to be off by 180 degrees -- if showing *angle* as
      #      well, must flip sign of angle of rotation if you allow axis to
      #      "reverse" by 180 degrees.

Gateset_gateset_angles_btwn_axes = _gsf.gatesfn_factory(gateset_gateset_angles_btwn_axes)
# init args == (gateset1, gateset2, gateLabel)


def rel_eigvals(A, B, mxBasis):
    """ Eigenvalues of B^{-1} * A"""
    target_gate_inv = _np.linalg.inv(B)
    rel_gate = _np.dot(target_gate_inv, A)
    return _np.linalg.eigvals(rel_gate).astype("complex") #since they generally *can* be complex
Rel_eigvals = _gsf.gatesfn_factory(rel_eigvals)
# init args == (gateset1, gateset2, gateLabel)

def rel_logTiG_eigvals(A, B, mxBasis):
    """ Eigenvalues of log(B^{-1} * A)"""
    rel_gate = _tools.error_generator(A, B, mxBasis, "logTiG")
    return _np.linalg.eigvals(rel_gate).astype("complex") #since they generally *can* be complex
Rel_logTiG_eigvals = _gsf.gatesfn_factory(rel_logTiG_eigvals)
# init args == (gateset1, gateset2, gateLabel)

def rel_logGTi_eigvals(A, B, mxBasis):
    """ Eigenvalues of log(A * B^{-1})"""
    rel_gate = _tools.error_generator(A, B, mxBasis, "logGTi")
    return _np.linalg.eigvals(rel_gate).astype("complex") #since they generally *can* be complex
Rel_logGTi_eigvals = _gsf.gatesfn_factory(rel_logGTi_eigvals)
# init args == (gateset1, gateset2, gateLabel)

def rel_logGmlogT_eigvals(A, B, mxBasis):
    """ Eigenvalues of log(A) - log(B)"""
    rel_gate = _tools.error_generator(A, B, mxBasis, "logG-logT")
    return _np.linalg.eigvals(rel_gate).astype("complex") #since they generally *can* be complex
Rel_logGmlogT_eigvals = _gsf.gatesfn_factory(rel_logGmlogT_eigvals)
# init args == (gateset1, gateset2, gateLabel)


def rel_gate_eigenvalues(A, B, mxBasis):  #DUPLICATE of rel_eigvals TODO
    """ Eigenvalues of B^{-1} * A """
    rel_gate = _np.dot(_np.linalg.inv(B), A) # "relative gate" == target^{-1} * gate
    return _np.linalg.eigvals(rel_gate).astype("complex") #since they generally *can* be complex
Rel_gate_eigenvalues = _gsf.gatesfn_factory(rel_gate_eigenvalues)
# init args == (gateset1, gateset2, gateLabel)


def errgen_and_projections(errgen, mxBasis):
    """
    Project `errgen` on all of the standard sets of error generators.

    Returns
    -------
    dict
        Dictionary of 'error generator', '*X* projections', and 
        '*X* projection power' keys, where *X* is 'hamiltonian', 
        'stochastic', and 'affine'.
    """
    ret = {}
    egnorm = _np.linalg.norm(errgen.flatten())
    ret['error generator'] = errgen
    proj, scale = \
        _tools.std_errgen_projections( 
            errgen,"hamiltonian",mxBasis.name,mxBasis,return_scale_fctr=True)
        # mxBasis.name because projector dim is not the same as gate dim
    ret['hamiltonian projections'] = proj
    ret['hamiltonian projection power'] =  float(_np.sum(proj**2)/scale**2) / egnorm**2 \
                                           if (abs(scale) > 1e-8 and abs(egnorm) > 1e-8) else 0
      #sum of squared projections of normalized error generator onto normalized projectors
      
    proj, scale = \
        _tools.std_errgen_projections( 
            errgen,"stochastic",mxBasis.name,mxBasis,return_scale_fctr=True)
        # mxBasis.name because projector dim is not the same as gate dim
    ret['stochastic projections'] = proj
    ret['stochastic projection power'] =  float(_np.sum(proj**2)/scale**2) / egnorm**2 \
                                          if (abs(scale) > 1e-8 and abs(egnorm) > 1e-8) else 0
      #sum of squared projections of normalized error generator onto normalized projectors

    proj, scale = \
        _tools.std_errgen_projections( 
            errgen,"affine",mxBasis.name,mxBasis,return_scale_fctr=True)
        # mxBasis.name because projector dim is not the same as gate dim
    ret['affine projections'] = proj
    ret['affine projection power'] = float(_np.sum(proj**2)/scale**2) / egnorm**2 \
                                     if (abs(scale) > 1e-8 and abs(egnorm) > 1e-8) else 0
      #sum of squared projections of normalized error generator onto normalized projectors  
    return ret

def logTiG_and_projections(A, B, mxBasis):
    """
    Projections of `log(B^{-1}*A)`.  Returns a dictionary of quantities with
    keys 'error generator', '*X* projections', and '*X* projection power',
    where *X* is 'hamiltonian', 'stochastic', and 'affine'.
    """
    errgen = _tools.error_generator(A, B, mxBasis, "logTiG")
    return errgen_and_projections(errgen, mxBasis)
LogTiG_and_projections = _gsf.gatesfn_factory(logTiG_and_projections)
# init args == (gateset1, gateset2, gateLabel)

def logGTi_and_projections(A, B, mxBasis):
    """
    Projections of `log(A*B^{-1})`.  Returns a dictionary of quantities with
    keys 'error generator', '*X* projections', and '*X* projection power',
    where *X* is 'hamiltonian', 'stochastic', and 'affine'.
    """
    errgen = _tools.error_generator(A, B, mxBasis, "logGTi")
    return errgen_and_projections(errgen, mxBasis)
LogGTi_and_projections = _gsf.gatesfn_factory(logGTi_and_projections)
# init args == (gateset1, gateset2, gateLabel)

def logGmlogT_and_projections(A, B, mxBasis):
    """
    Projections of `log(A)-log(B)`.  Returns a dictionary of quantities with
    keys 'error generator', '*X* projections', and '*X* projection power',
    where *X* is 'hamiltonian', 'stochastic', and 'affine'.
    """
    errgen = _tools.error_generator(A, B, mxBasis, "logG-logT")
    return errgen_and_projections(errgen, mxBasis)
LogGmlogT_and_projections = _gsf.gatesfn_factory(logGmlogT_and_projections)
# init args == (gateset1, gateset2, gateLabel)

def robust_logGTi_and_projections(gatesetA, gatesetB, syntheticIdleStrs):
    """
    Projections of `log(A*B^{-1})` using a gauge-robust technique.
    Returns a dictionary of quantities with keys '*G* error generator',
    '*G* *X* projections', and '*G* *X* projection power',
    where *G* is a gate label and *X* is 'hamiltonian', 'stochastic', and
    'affine'.
    """
    ret = {}
    mxBasis = gatesetB.basis #target gateset is more likely to have a valid basis
    Id = _np.identity(gatesetA.dim, 'd')
    gateLabels = [gl for gl,gate in gatesetB.gates.items() if not _np.allclose(gate, Id)]
    idLabels = [gl for gl,gate in gatesetB.gates.items() if _np.allclose(gate, Id)]
    nGates = len(gateLabels)
    
    error_superops = []; ptype_counts = {}; ptype_scaleFctrs = {}
    error_labels = []
    for ptype in ("hamiltonian","stochastic","affine"):
        lindbladMxs = _tools.std_error_generators(gatesetA.dim, ptype,
                                                  mxBasis.name) # in std basis
        lindbladMxBasis = _Basis(mxBasis.name, int(round(_np.sqrt(gatesetA.dim))))
        
        lindbladMxs = lindbladMxs[1:] #skip [0] == Identity
        lbls = lindbladMxBasis.labels[1:]
        
        scaleFctr = _tools.std_scale_factor(gatesetA.dim, ptype)
        #if ptype == "hamiltonian": scaleFctr *= 2.0 #HACK (DEAL LATER)
        #if ptype == "affine": scaleFctr *= 0.5 #HACK
        ptype_counts[ptype] = len(lindbladMxs)
        ptype_scaleFctrs[ptype] = scaleFctr #UNUSED?
        error_superops.extend( [ _tools.change_basis(eg,"std",mxBasis) for eg in lindbladMxs ] )
        error_labels.extend( [ "%s(%s)" % (ptype[0],lbl) for lbl in lbls ] )
    nSuperOps = len(error_superops)
    assert(len(error_labels) == nSuperOps)

    #DEBUG
    #print("DB: %d gates (%s)" % (nGates, str(gateLabels)))
    #print("DB: %d superops; counts = " % nSuperOps, ptype_counts)
    #print("DB: factors = ",ptype_scaleFctrs)
    #for i,x in enumerate(error_superops):
    #    print("DB: Superop vec[%d] norm = %g" % (i,_np.linalg.norm(x)))
    #    print("DB: Choi Superop[%d] = " % i)
    #    _tools.print_mx(_tools.jamiolkowski_iso(x, mxBasis, mxBasis), width=4, prec=1)
    #    print("")


    def get_projection_vec(errgen):
        proj = []
        for ptype in ("hamiltonian","stochastic","affine"):
            proj.append( _tools.std_errgen_projections( 
                errgen,ptype,mxBasis.name,mxBasis)[1:] ) #skip [0] == Identity
        return _np.concatenate( proj )

    #def vec_to_projdict(vec):
    #    ret = {}
    #    off = 0 #current offset into vec
    #    for gl in gatesetA.gates.keys():
    #        ret['%s error generator' % gl] = _np.zeros((gatesetA.dim,gatesetA.dim),'d') # TODO: something here (just a placeholder now)
    #        if gl in gateLabels: # a non-identity gate
    #            for ptype in ("hamiltonian","stochastic","affine"):
    #                ret['%s %s projections' % (gl, ptype)] = vec[off:off+ptype_counts[ptype]]
    #                ret['%s %s projections power' % (gl, ptype)] = 0 # TODO - use scale factor... vec[off:off+ptype_counts[ptype]]
    #        else: # an identity gate - just put in zeros of now
    #            for ptype in ("hamiltonian","stochastic","affine"):
    #                ret['%s %s projections' % (gl, ptype)] = _np.zeros(ptype_counts[ptype], 'd')
    #                ret['%s %s projections power' % (gl, ptype)] = 0
    #            
    #    return ret
    
    def firstOrderNoise( gstr, errSupOp, glWithErr ):
        noise = _np.zeros((gatesetB.dim,gatesetB.dim), 'd')
        for n,gl in enumerate(gstr):
            if gl == glWithErr:
                noise += _np.dot( gatesetB.product(gstr[n+1:]),
                                  _np.dot(errSupOp, gatesetB.product(gstr[:n+1])) )
        #DEBUG
        #print("first order noise (%s,%s) Choi superop : " % (str(gstr),glWithErr))
        #_tools.print_mx( _tools.jamiolkowski_iso(noise, mxBasis, mxBasis) ,width=4,prec=1)
        
        return noise #_tools.jamiolkowski_iso(noise, mxBasis, mxBasis)

    def errorGeneratorJacobian(gstr):
        jac = _np.empty( (nSuperOps, nSuperOps*nGates), 'complex') #should be real, but we'll check

        for i,gl in enumerate(gateLabels):
            for k,errOnGate in enumerate(error_superops):
                noise = firstOrderNoise( gstr, errOnGate, gl)
                jac[:,i*nSuperOps+k] = [ _np.vdot(errOut.flatten(), noise.flatten()) for errOut in error_superops]

                #DEBUG CHECK
                check = [ _np.trace( _np.dot(
                    _tools.jamiolkowski_iso(errOut, mxBasis, mxBasis).conj().T,
                    _tools.jamiolkowski_iso(noise, mxBasis, mxBasis) ))*4 #for 1-qubit...
                          for errOut in error_superops]
                assert(_np.allclose( jac[:,i*nSuperOps+k], check))

        assert( _np.linalg.norm(jac.imag) < 1e-6 ), "error generator jacobian should be real!"
        return jac.real

    runningJac = None; runningY = None
    for s in syntheticIdleStrs:
        Sa = gatesetA.product(s)
        Sb = gatesetB.product(s)
        assert( _np.linalg.norm(Sb - _np.identity(gatesetB.dim,'d')) < 1e-6), \
            "Synthetic idle %s is not an idle!!" % str(s)
        SIerrgen = _tools.error_generator(Sa, Sb, mxBasis, "logGTi")
        SIproj = get_projection_vec(SIerrgen)
        jacSI = errorGeneratorJacobian(s)
        #print("DB jacobian for %s = \n" % str(s)); _tools.print_mx(jacSI, width=4, prec=1) #DEBUG
        if runningJac is None:
            runningJac = jacSI
            runningY = SIproj
        else:
            runningJac = _np.concatenate((runningJac,jacSI), axis=0)
            runningY = _np.concatenate( (runningY,SIproj), axis=0)
            
        rank = _np.linalg.matrix_rank(runningJac)

        print("DB: Added synthetic idle %s => rank=%d <?> %d (shape=%s; %s)" % (str(s),rank,nSuperOps*nGates,str(runningJac.shape),str(runningY.shape)))
        
        #if rank >= nSuperOps*nGates: #then we can extract error terms for the gates
        #    # J*vec_gateErrs = Y => vec_gateErrs = (J^T*J)^-1 J^T*Y (lin least squares)
        #    J,JT = runningJac, runningJac.T
        #    vec_gateErrs = _np.dot( _np.linalg.inv(_np.dot(JT,J)), _np.dot(JT,runningY))
        #    return vec_to_projdict(vec_gateErrs)
    #raise ValueError("Not enough synthetic idle sequences to extract gauge-robust error rates.")

    # J*vec_gateErrs = Y => U*s*Vt * vecErrRates = Y  => Vt*vecErrRates = s^{-1}*U^-1*Y
    # where shapes are: U = (M,K), s = (K,K), Vt = (K,N),
    #   so Uinv*Y = (K,) and s^{-1}*Uinv*Y = (K,), and rows of Vt specify the linear combos
    #   corresponding to values in s^{-1}*Uinv*Y that are != 0
    ret = {}
    RANK_TOL = 1e-8; COEFF_TOL = 1e-1
    U,s,Vt = _np.linalg.svd(runningJac)
    rank = _np.count_nonzero(s > RANK_TOL)
    vals = _np.dot( _np.diag(1.0/s[0:rank]), _np.dot(U[:,0:rank].conj().T, runningY))
    gate_error_labels = [ "%s.%s" % (gl,errLbl) for gl in gateLabels for errLbl in error_labels ]
    assert(len(gate_error_labels) == runningJac.shape[1])
    for combo,val in zip(Vt[0:rank,:],vals):
        combo_str = " + ".join([ "%.1f*%s" % (c,errLbl) for c,errLbl in zip(combo,gate_error_labels) if abs(c) > COEFF_TOL ])
        ret[combo_str] = val
    return ret
    

Robust_LogGTi_and_projections = _gsf.gatesetfn_factory(robust_logGTi_and_projections)
# init args == (gatesetA, gatesetB, syntheticIdleStrs)



def general_decomposition(gatesetA, gatesetB):
    """
    Decomposition of gates in `gatesetA` using those in `gatesetB` as their
    targets.  This function uses a generalized decomposition algorithm that
    can gates acting on a Hilbert space of any dimension.

    Returns
    -------
    dict
    """
    # B is target gateset usually but must be "gatsetB" b/c of decorator coding...
    decomp = {}
    gateLabels = list(gatesetA.gates.keys())  # gate labels
    mxBasis = gatesetB.basis # B is usually the target which has a well-defined basis
    
    for gl in gateLabels:
        gate = gatesetA.gates[gl]
        targetGate = gatesetB.gates[gl]
        gl = str(gl) # Label -> str for decomp-dict keys

        target_evals = _np.linalg.eigvals(targetGate)
        if _np.any(_np.isclose(target_evals,-1.0)):
            target_logG = _tools.unitary_superoperator_matrix_log(targetGate, mxBasis)        
            logG = _tools.approximate_matrix_log(gate, target_logG)
        else:
            logG = _tools.real_matrix_log(gate, "warn")
            if _np.linalg.norm(logG.imag) > 1e-6:
                _warnings.warn("Truncating imaginary logarithm!")
                logG = _np.real(logG)
                
        decomp[gl + ' log inexactness'] = _np.linalg.norm(_spl.expm(logG)-gate)
    
        hamProjs, hamGens = _tools.std_errgen_projections(
            logG, "hamiltonian", mxBasis.name, mxBasis, return_generators=True)
        norm = _np.linalg.norm(hamProjs)
        decomp[gl + ' axis'] = hamProjs / norm if (norm > 1e-15) else hamProjs
            
        decomp[gl + ' angle'] = norm * 2.0 / _np.pi
        # Units: hamProjs (and norm) are already in "Hamiltonian-coefficient" units,
        # (see 'std_scale_factor' fn), but because of convention the "angle" is equal
        # to *twice* this coefficient (e.g. a X(pi/2) rotn is exp( i pi/4 X ) ),
        # thus the factor of 2.0 above.
    
        basis_mxs = mxBasis.get_composite_matrices()
        scalings = [ ( _np.linalg.norm(hamGens[i]) / _np.linalg.norm(_tools.hamiltonian_to_lindbladian(mx))
                       if _np.linalg.norm(hamGens[i]) > 1e-10 else 0.0 )
                     for i,mx in enumerate(basis_mxs) ]
          #really want hamProjs[i] * lindbladian_to_hamiltonian(hamGens[i]) but fn doesn't exists (yet)
        hamMx = sum([s*c*bmx for s,c,bmx in zip(scalings,hamProjs,basis_mxs)])
        decomp[gl + ' hamiltonian eigenvalues'] = _np.array(_np.linalg.eigvals(hamMx))

    for gl in gateLabels:
        for gl_other in gateLabels:            
            rotnAngle = decomp[str(gl) + ' angle']
            rotnAngle_other = decomp[str(gl_other) + ' angle']

            if gl == gl_other or abs(rotnAngle) < 1e-4 or abs(rotnAngle_other) < 1e-4:
                decomp[str(gl) + "," + str(gl_other) + " axis angle"] = 10000.0 #sentinel for irrelevant angle
    
            real_dot = _np.clip(
                _np.real(_np.dot(decomp[str(gl) + ' axis'].flatten(),
                                 decomp[str(gl_other) + ' axis'].flatten())),
            -1.0, 1.0)
            angle = _np.arccos( real_dot ) / _np.pi
            decomp[str(gl) + "," + str(gl_other) + " axis angle"] = angle

    return decomp
General_decomposition = _gsf.gatesetfn_factory(general_decomposition)
# init args == (gatesetA, gatesetB)


def average_gateset_infidelity(gatesetA, gatesetB):
    """ Average gate set infidelity """
    # B is target gateset usually but must be "gatesetB" b/c of decorator coding...
    from ..extras.rb import theory as _rbtheory
    return _rbtheory.gateset_infidelity(gatesetA,gatesetB)
Average_gateset_infidelity = _gsf.gatesetfn_factory(average_gateset_infidelity)
# init args == (gatesetA, gatesetB)


def predicted_rb_number(gatesetA, gatesetB):
    """ 
    Prediction of RB number based on estimated (A) and target (B) gate sets
    """
    from ..extras.rb import theory as _rbtheory
    return _rbtheory.predicted_RB_number(gatesetA, gatesetB)
Predicted_rb_number = _gsf.gatesetfn_factory(predicted_rb_number)
# init args == (gatesetA, gatesetB)


def vec_fidelity(A, B, mxBasis):
    """ State fidelity between SPAM vectors A and B """
    rhoMx1 = _tools.vec_to_stdmx(A, mxBasis)
    rhoMx2 = _tools.vec_to_stdmx(B, mxBasis)
    return _tools.fidelity(rhoMx1, rhoMx2)
Vec_fidelity = _gsf.vecsfn_factory(vec_fidelity)
# init args == (gateset1, gateset2, label, typ)


def vec_infidelity(A, B, mxBasis):
    """ State infidelity fidelity between SPAM vectors A and B """
    rhoMx1 = _tools.vec_to_stdmx(A, mxBasis)
    rhoMx2 = _tools.vec_to_stdmx(B, mxBasis)
    return 1 - _tools.fidelity(rhoMx1, rhoMx2)
Vec_infidelity = _gsf.vecsfn_factory(vec_infidelity)
# init args == (gateset1, gateset2, label, typ)


def vec_tr_diff(A, B, mxBasis): # assume vary gateset1, gateset2 fixed
    """ Trace distance between SPAM vectors A and B """
    rhoMx1 = _tools.vec_to_stdmx(A, mxBasis)
    rhoMx2 = _tools.vec_to_stdmx(B, mxBasis)
    return _tools.tracedist(rhoMx1, rhoMx2)
Vec_tr_diff = _gsf.vecsfn_factory(vec_tr_diff)
# init args == (gateset1, gateset2, label, typ)

def vec_as_stdmx(vec, mxBasis):
    """ SPAM vectors as a standard density matrix """
    return _tools.vec_to_stdmx(vec, mxBasis)
Vec_as_stdmx = _gsf.vecfn_factory(vec_as_stdmx)
# init args == (gateset, label, typ)

def vec_as_stdmx_eigenvalues(vec, mxBasis):
    """ Eigenvalues of the density matrix corresponding to a SPAM vector """
    mx = _tools.vec_to_stdmx(vec, mxBasis)
    return _np.linalg.eigvals(mx)
Vec_as_stdmx_eigenvalues = _gsf.vecfn_factory(vec_as_stdmx_eigenvalues)
# init args == (gateset, label, typ)


def info_of_gatefn_by_name(name):
    """ 
    Returns a nice human-readable name and tooltip for a given gate-function
    abbreviation.

    Parameters
    ----------
    name : str
        An appreviation for a gate-function name.  Allowed values are:

        - "inf" :     entanglement infidelity
        - "agi" :     average gate infidelity
        - "trace" :   1/2 trace distance
        - "diamond" : 1/2 diamond norm distance
        - "nuinf" :   non-unitary entanglement infidelity
        - "nuagi" :   non-unitary entanglement infidelity
        - "evinf" :     eigenvalue entanglement infidelity
        - "evagi" :     eigenvalue average gate infidelity
        - "evnuinf" :   eigenvalue non-unitary entanglement infidelity
        - "evnuagi" :   eigenvalue non-unitary entanglement infidelity
        - "evdiamond" : eigenvalue 1/2 diamond norm distance
        - "evnudiamond" : eigenvalue non-unitary 1/2 diamond norm distance
        - "frob" :    frobenius distance

    Returns
    -------
    nicename : str
    tooltip : str
    """
    if name == "inf":
        niceName = "Entanglement|Infidelity"
        tooltip = "1.0 - <psi| 1 x Lambda(psi) |psi>"
    elif name == "agi":
        niceName = "Avg. Gate|Infidelity"
        tooltip = "d/(d+1) (entanglement infidelity)"
    elif name == "trace":
        niceName = "1/2 Trace|Distance"
        tooltip = "0.5 | Chi(A) - Chi(B) |_tr"
    elif name == "diamond":
        niceName =  "1/2 Diamond-Dist"
        tooltip = "0.5 sup | (1 x (A-B))(rho) |_tr"
    elif name == "nuinf":
        niceName = "Non-unitary|Ent. Infidelity"
        tooltip = "(d^2-1)/d^2 [1 - sqrt( unitarity(A B^-1) )]"
    elif name == "nuagi":
        niceName = "Non-unitary|Avg. Gate Infidelity"
        tooltip = "(d-1)/d [1 - sqrt( unitarity(A B^-1) )]"
    elif name == "evinf":
        niceName = "Eigenvalue|Ent. Infidelity"
        tooltip = "min_P 1 - (lambda P lambda^dag)/d^2  [P = permutation, lambda = eigenvalues]"
    elif name == "evagi":
        niceName = "Eigenvalue|Avg. Gate Infidelity"
        tooltip = "min_P (d^2 - lambda P lambda^dag)/d(d+1)  [P = permutation, lambda = eigenvalues]"
    elif name == "evnuinf":
        niceName = "Eigenvalue Non-U.|Ent. Infidelity"
        tooltip = "(d^2-1)/d^2 [1 - sqrt( eigenvalue_unitarity(A B^-1) )]"
    elif name == "evnuagi":
        niceName = "Eigenvalue Non-U.|Avg. Gate Infidelity"
        tooltip = "(d-1)/d [1 - sqrt( eigenvalue_unitarity(A B^-1) )]"
    elif name == "evdiamond":
        niceName = "Eigenvalue|1/2 Diamond-Dist"
        tooltip = "(d^2-1)/d^2 max_i { |a_i - b_i| } where (a_i,b_i) are corresponding eigenvalues of A and B."
    elif name == "evnudiamond":
        niceName = "Eigenvalue Non-U.|1/2 Diamond-Dist"
        tooltip = "(d^2-1)/d^2 max_i { | |a_i| - |b_i| | } where (a_i,b_i) are corresponding eigenvalues of A and B."
    elif name == "frob":
        niceName = "Frobenius|Distance"
        tooltip = "sqrt( sum( (A_ij - B_ij)^2 ) )"
    else: raise ValueError("Invalid name: %s" % name)
    return niceName, tooltip


def evaluate_gatefn_by_name(name, gateset, targetGateset, gateLabelOrString,
                            confidenceRegionInfo):
    """ 
    Evaluates that gate-function named by the abbreviation `name`.

    Parameters
    ----------
    name : str
        An appreviation for a gate-function name.  Allowed values are the
        same as those of :func:`info_of_gatefn_by_name`.

    gateset, targetGateSet : GateSet
        The gatesets to compare.  Only the element or product given by 
        `gateLabelOrString` is compared using the named gate-function.

    gateLabelOrString : str or GateString or tuple
        The gate label or sequence of labels to compare.  If a sequence
        of labels is given, then the "virtual gate" computed by taking the
        product of the specified gate matries is compared.

    confidenceRegionInfo : ConfidenceRegion, optional
        If not None, specifies a confidence-region  used to compute error
        intervals.    

    Returns
    -------
    ReportableQty
    """
    gl = gateLabelOrString
    b = bool(_tools.isstr(gl)) #whether this is a gate label or a string
    
    if name == "inf":
        fn = Entanglement_infidelity if b else \
             Gatestring_entanglement_infidelity
    elif name == "agi":
        fn = Avg_gate_infidelity if b else \
             Gatestring_avg_gate_infidelity
    elif name == "trace":
        fn = Jt_diff if b else \
             Gatestring_jt_diff
    elif name == "diamond":
        fn = Half_diamond_norm if b else \
             Gatestring_half_diamond_norm
    elif name == "nuinf":
        fn = Nonunitary_entanglement_infidelity if b else \
             Gatestring_nonunitary_entanglement_infidelity
    elif name == "nuagi":
        fn = Nonunitary_avg_gate_infidelity if b else \
             Gatestring_nonunitary_avg_gate_infidelity
    elif name == "evinf":
        fn = Eigenvalue_entanglement_infidelity if b else \
             Gatestring_eigenvalue_entanglement_infidelity
    elif name == "evagi":
        fn = Eigenvalue_avg_gate_infidelity if b else \
             Gatestring_eigenvalue_avg_gate_infidelity
    elif name == "evnuinf":
        fn = Eigenvalue_nonunitary_entanglement_infidelity if b else \
             Gatestring_eigenvalue_nonunitary_entanglement_infidelity
    elif name == "evnuagi":
        fn = Eigenvalue_nonunitary_avg_gate_infidelity if b else \
             Gatestring_eigenvalue_nonunitary_avg_gate_infidelity
    elif name == "evdiamond":
        fn = Eigenvalue_diamondnorm if b else \
             Gatestring_eigenvalue_diamondnorm
    elif name == "evnudiamond":
        fn = Eigenvalue_nonunitary_diamondnorm if b else \
             Gatestring_eigenvalue_nonunitary_diamondnorm
    elif name == "frob":
        fn = Fro_diff if b else \
             Gatestring_fro_diff

    return evaluate( fn(gateset, targetGateset, gl), confidenceRegionInfo)
