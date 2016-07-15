from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Defines an *experimental* gauge-invariant GateSet class and supporting functionality."""

import numpy as _np

from ..tools import matrixtools as _mt

#import evaltree as _evaltree
from . import gate as _gate
from . import gateset as _gateset

from .verbosityprinter import VerbosityPrinter

SMALL = 1e-10

def _isreal(a):
    return _np.isclose(a.imag, 0.0)

def _angle(x):
    return _np.angle(x) if abs(x) > SMALL else 0.0

class GaugeInvGateSet(object):  #(_collections.OrderedDict):
    """
    Encapsulates a set of gate, state preparation, and POVM effect operations
     in a gauge-invariant manner.

    A GaugeInvGateSet stores a gateset using a *minimal* gauge-invariant
    representation.  This class is *experimental* and not ready for use yet.
    """

    def __init__(self,items=[]):
        """
        Initialize a gauge-invariant gate set, possibly from a list of
          items (used primarily by pickle)
        """
        self.gate_dim = None
        self.E_params = None
        self.D_params = []
        self.B0_params = []
        self.gateLabels = []
        self.storedY0 = None

    def from_gateset(self, gateset, debug=None, verbosity=0, fix=1.0):
        """
        Initialize a gauge-invariant gate set from an existing (gauge-
         variant) GateSet.
        """

        printer = VerbosityPrinter.build_printer(verbosity)

        #This only works for fully-parameterized GateSets so far...
        #assert(gates == True and G0 == True and SPAM == True and SP0 == True)
        #assert(all([isinstance(g, _gate.FullyParameterizedGate)
        #            for g in gateset.values() ]))

        vb = verbosity #shorthand
        self.gate_dim = gateset.get_dimension()
        #self.evstruct = evstruct #eigenvalue structure: a string of "R"s and
        #                         # "C"s for real-pair and conjugate-pair.


        self.gateLabels = list(gateset.keys())
        #gl0 = self.gateLabels[0] #plays a special role

        self.D_params = [None]*len(self.gateLabels)
        Y = [None]*len(self.gateLabels) #List of eigenvector mxs

        #Get parameterization of gate eigenvalues (and get corresponding
        #  eigenvector matrices for later use)
        for i,gl in enumerate(self.gateLabels):
            self.D_params[i], Y[i] = _parameterize_real_gate_mx(gateset[gl],vb,debug)

        #DEBUG
        printer.log("DB: Y0 = %s" % _mt.mx_to_string(Y[0]), 4)
        printer.log("DB: Y0 evals = %s" % _np.linalg.eigvals(Y[0]), 4)
        printer.log("DB: invY0 = %s" %  _mt.mx_to_string(_np.linalg.inv(Y[0])), 4)
        printer.log("DB: rho = %s" % _mt.mx_to_string(gateset.preps[0]), 4)

        #Get parameterization of SPAM pair (assume just a single pair is
        # present for now)
        rho_tilde = _np.dot(_np.linalg.inv(Y[0]), gateset.preps[0])
        E_tilde = _np.dot( _np.transpose(gateset.effects[0]), Y[0] )
        delta0_diag, inv_delta0_diag = \
            _get_delta0_diag(rho_tilde, E_tilde, self.D_params[0], vb)
        delta0 = _np.diag( delta0_diag )
        scaledY0 = _np.dot( Y[0], delta0 )

        #remember the "gauge" of this gateset for going back to a
        # gauge-dependent gateset.
        self.storedY0 = scaledY0

        #DEBUG
        printer.log("DB: Scaled Y0 = %s" %    _mt.mx_to_string(scaledY0))
        printer.log("DB: delta0_diag = %s" %  _mt.mx_to_string(delta0_diag))
        printer.log("DB: rho_tilde = %s" %    _mt.mx_to_string(_np.dot( _np.diag(inv_delta0_diag), rho_tilde)))
        if debug is not None:
            printer.log("DB: BScaled Y0 = " %  _mt.mx_to_string( _np.dot(debug,scaledY0)))

        #Make sure scaling rho_tilde gives vector of all 1s
        #assert( _np.allclose( _np.dot( _np.diag(1.0/delta0_diag), rho_tilde),
        #                      _np.ones(rho_tilde.shape)))

        ET_tilde = _np.dot( _np.transpose(gateset.effects[0]), scaledY0 )
        self.E_params = _get_ET_params(ET_tilde, self.D_params[0])
        printer.log("DB: ET_tilde = %s" % _mt.mx_to_string(ET_tilde))
        if debug is not None:
            printer.log("DB: BScaled ET_tilde = " % _mt.mx_to_string( _np.dot(ET_tilde,debug)))

           #FUTURE: multiple effect-vectors allowed?

        #assume a SPAM pair is present and parameterized (for now) - so
        # parameterize B0j matrices all the same.
        self.B0_params = [None]*(len(self.gateLabels))
        for j,gl in enumerate(self.gateLabels[1:],start=1):
            invYjY0 = _np.dot( _np.linalg.inv(Y[j]), scaledY0 )
            printer.log("DB: invYjY0 = %s" % _mt.mx_to_string(invYjY0))
            self.B0_params[j] = _get_B_params(invYjY0, self.D_params[j],
                                              self.D_params[0], vb)


    def to_gateset(self,verbosity=0):
        """
        Create a gauge-variant GateSet from this gauge-invariant
        representation
        """
        printer = VerbosityPrinter.build_printer(verbosity)

        #We're free to assume some Y0 (~choosing a gauge).  We use the one
        # stored during from_gateset for now.  We *could* choose something
        # else if we wanted (or never stored one) -- it's cols just have to
        # have the correct conjugate-pair structure (given by D_params[0])
        Y0 = self.storedY0
        invY0 = _np.linalg.inv(Y0)

        printer.log(("Y0 = %s") % _mt.mx_to_string(Y0))
        printer.log("Y0 evals = %s" %  _np.linalg.eigvals(Y0))
        printer.log(("inv(Y0) = %s") %  _mt.mx_to_string(invY0))

        #Create the gate set
        gs = _gateset.GateSet()

        #Set E (FUTURE: multiple allowed?)
        ETilde = _get_ETilde_vector( self.E_params, self.D_params[0] )
        Evec = _np.dot(_np.transpose(invY0),ETilde)
        assert( all(_isreal(Evec)) ) # b/c of conjugacy structure
        gs.set_evec( _np.real(Evec) )

        #Set rho
        rhoTilde = _get_rhoTilde(ETilde, self.D_params[0])
        gs.set_rhovec( _np.dot(Y0,rhoTilde) )

        #Set initial gate
        D0 = _constructDj(self.D_params[0])
        mx = _np.dot( Y0, _np.dot(D0, invY0) )
        gs.set_gate(self.gateLabels[0], _gate.FullyParameterizedGate(mx))

        #Set remaining gates
        for i,gl in enumerate(self.gateLabels[1:],start=1):
            mx = _deparameterize_real_gate_mx(self.D_params[i],
                                              self.D_params[0],
                                              self.B0_params[i],
                                              invY0, verbosity)
            gs.set_gate(gl, _gate.FullyParameterizedGate(mx))

        #Set identity vector (store it upon creation?)

        #Return constructed gate set
        return gs


    def from_vector(self, v, gates=True,G0=True,SPAM=True,SP0=True):
        k = 0
        dim = self.gate_dim
        self.E_params = v[k:k+dim]; k += dim
        for i in range(len(self.D_params)):
            self.D_params[i] = v[k:k+dim]; k += dim

        assert(self.B0_params[0] is None) # just a placeholder
        for i in range(1, len(self.B0_params)):
            L = len(self.B0_params[i])
            self.B0_params[i] = v[k:k+L]; k += L
            #TODO: test length L against what it should be

    def to_vector(self, gates=True,G0=True,SPAM=True,SP0=True):
        #concat self.E_params, self.D_params, self.B0_params into one vector
        assert(len(self.E_params) == self.gate_dim)
        assert( all([len(Dp) == self.gate_dim for Dp in self.D_params]))
        to_concat = [self.E_params] + self.D_params + self.B0_params[1:]
        return _np.concatenate(to_concat)

    def num_params(self,gates=True,G0=True,SPAM=True,SP0=True):
        return len(self.to_vector())

    def num_nongauge_params(self,gates=True,G0=True,SPAM=True,SP0=True):
        return self.num_params(gates,G0,SPAM,SP0)

    def num_gauge_params(self,gates=True,G0=True,SPAM=True,SP0=True):
        return 0



def _parameterize_real_gate_mx(real_gate_mx, verbosity, debug):
    """
    Convert the possibly complex eigenvalues of a real matrix into
    the purely real gauge-inv-gateset parameters.
    """

    printer = VerbosityPrinter.build_printer(verbosity)

    #Get (complex) eigenvalues and eigenvectors
    evals,evecs = _np.linalg.eig(real_gate_mx)

    printer.log(" PRE -> Eigenvalues: " % _mt.mx_to_string(evals))
    printer.log(" PRE -> Eigenvecs: " % _mt.mx_to_string(evecs))
    evecs = (1.0+0j)*evecs #to convert evecs to complex (sometimes needed for all-real case)

    #find complex-conjugate (or real degenerate) eigenvalue pairs
    conjugate_pairs = []
    real_pairs = []
    for i,v in enumerate(evals):
        iUsed = [p[0] for p in conjugate_pairs]+[p[1] for p in conjugate_pairs]
        if i in iUsed: continue

        V = evecs[:,i].copy()

        #get all indices j where eval[j] is the conjugate of eval[i] (or degen)
        iClose = [ j for j,w in enumerate(evals[i+1:],start=i+1) \
                       if _np.isclose(v.conj(), w) and j not in iUsed ]
        if len(iClose) == 0: continue #nothing to pair with

        #if multiple "close" indices, choose best one based on evecs
        if _isreal(v):
            if all(_isreal(V)): #then look for another *real* evec
                iReal = [ j for j in iClose if all(_isreal(evecs[:,j])) ]
                if len(iReal) == 0:
                    #raise ValueError("Could not find real pair")
                    continue
                j = iReal[0] #just take first one

                #we treat all degen. eigenvals as conjugate-pairs,
                # so want conjugate-pair, not real eigenvectors:
                W = evecs[:,j].copy()
                evecs[:,i] = (V + 1j*W); evecs[:,j] = (V - 1j*W)
                evecs[:,i] /= _np.linalg.norm(evecs[:,i])
                evecs[:,j] /= _np.linalg.norm(evecs[:,j])

            else: #then look for complex conjugate evec
                iConj = [ j for j in iClose \
                              if all(_np.isclose(evecs[:,j],V.conj())) ]
                if len(iConj) == 0:
                    j = iClose[0]; W = evecs[:,j].copy()
                    printer.log(" AV chk = %s" % _np.linalg.norm( _np.dot(real_gate_mx,V) - v*V))
                    printer.log(" AVc chk= %s" % _np.linalg.norm( _np.dot(real_gate_mx,V.conj()) - v*V.conj()))
                    printer.log(" AW chk= %s"  % _np.linalg.norm( _np.dot(real_gate_mx,W) - v*W))
                    printer.log(" AWc chk= %s" % _np.linalg.norm( _np.dot(real_gate_mx,W.conj()) - v*W.conj()))
                    printer.log("Vnorm = %s"   % _np.linalg.norm(V))
                    printer.log("Wnorm = %s"   % _np.linalg.norm(W))
                    printer.log("nrm = %s"     % _np.linalg.norm(W.real/_np.linalg.norm(W.real)))
                    printer.log(" chk= %s"     %  _np.linalg.norm( W.real/_np.linalg.norm(W.real) - V))

                    printer.log("prod chk = (should be zero) %s" % \
                        _np.linalg.norm( _np.dot(evecs, _np.dot(_np.diag(evals),
                                         _np.linalg.inv(evecs))) - real_gate_mx))

                    evecs[:,i] = W.conj()
                    evecs[:,i] /= _np.linalg.norm(evecs[:,i])

                    printer.log("prod chk2 = (should be zero) %s" % \
                        _np.linalg.norm( _np.dot(evecs, _np.dot(_np.diag(evals),
                                         _np.linalg.inv(evecs))) - real_gate_mx))

                    printer.log("COMPLEX DEGEN PAIR", 5)
                    raise ValueError("Could not find conj pair")

                j = iConj[0] #just take first one

        else: #true complex conjugate pair of eigenvals
            iConj = [ j for j in iClose \
                          if all(_np.isclose(evecs[:,j],V.conj())) ]
            if len(iConj) == 0: raise ValueError("Could not find conj pair")
            j = iConj[0] #just take first one

        #At this point, j == other index to pair
        conjugate_pairs.append( (i,j) )


    #put remaining (non-conj-pair) indices ==> real_pairs
    remaining = [ i for i in range(len(evals)) if \
                      not any([ i in p for p in conjugate_pairs]) ]
    assert(all([ _isreal(evals[i]) for i in remaining ]))

    #sort conjugate pairs by real part then imaginary magnitude
    def cmp_conj_pairs(a, b):
        if evals[a[0]].real > evals[b[0]].real: return 1    # a > b
        elif evals[a[0]].real < evals[b[0]].real: return -1 # a < b
        elif abs(evals[a[0]].imag) > abs(evals[b[0]].imag):
            return 1 # a > b
        elif abs(evals[a[0]].imag) == abs(evals[b[0]].imag):
            return 0
        else:
            return -1
    conjugate_pairs.sort(cmp_conj_pairs, reverse=True)

    #sort remaining (real) eigenvalues
    remaining.sort(key=lambda i: evals[i].real, reverse=True)

    nLeft = len(remaining)
    real_pairs = [(remaining[i],remaining[i+1]) for i in range(0,nLeft-1,2)]

    printer.log("DB: conj pairs = %s" % conjugate_pairs)
    printer.log("DB: real pairs = %s" % real_pairs)

    #create array of eigenvalue parameters, keeping track of the
    # needed eigenvector permutation given the new eigenvalue ordering.
    eval_params = _np.empty( evals.shape, 'd' ) # Note: *not* complex
    permMx = _np.zeros(evecs.shape, 'd')
    k = 0 #running index of current eigenvalue after re-arrangement

    for i,j in conjugate_pairs:
        if evals[i].imag > evals[j].imag: # i => k, j => k+1
            permMx[i,k] = 1.0; permMx[j,k+1] = 1.0
        else: # j => k, i => k+1 (lower index always has greater imag part)
            permMx[i,k] = 1.0; permMx[j,k+1] = 1.0
        eval_params[k] = evals[i].real
        eval_params[k+1] = -abs(evals[i].imag) # neg => conj. pair
        k += 2

    for i,j in real_pairs:
        if evals[i] > evals[j]: # i => k, j => k+1
            permMx[i,k] = 1.0; permMx[j,k+1] = 1.0
        else: # j => k, i => k+1 (lower index always is greater value)
            permMx[i,k] = 1.0; permMx[j,k+1] = 1.0
        eval_params[k] = (evals[i].real + evals[j].real) / 2.0
        eval_params[k+1] = abs(evals[i].real -
                               evals[j].real) / 2.0 # pos => real pair
        k += 2

    if len(remaining) % 2 == 1: #if there's one un-paired (real) eval
        assert(_isreal(evals[remaining[-1]]))
        permMx[remaining[-1],k] = 1.0
        eval_params[k] = evals[remaining[-1]].real

    Y = _np.dot(evecs, permMx)
    printer.log("Parameterizing matrix: %s" % _mt.mx_to_string(real_gate_mx), 4)
    printer.log(" -> Eigenvalues: %s"       % _mt.mx_to_string(evals), 4)
    printer.log(" -> Eigenvectors: %s"      % _mt.mx_to_string(evecs), 4)
    printer.log(" -> Parameters: %s"        % _mt.mx_to_string(eval_params), 4)
    printer.log(" -> Evec Permutation Mx  = %s" % _mt.mx_to_string(permMx), 4)
    printer.log(" -> Y-Mx  = %s"            % _mt.mx_to_string(Y), 4)

    D = _constructDj(eval_params)
    Yi = _np.linalg.inv(Y)
    printer.log(" -> test Y*D*Yi: %s"  %  _mt.mx_to_string(_np.dot(Y,_np.dot(D,Yi))), 4)

    if debug is not None:
        B = debug
        Bi = _np.linalg.inv(debug)
        BY = _np.dot(B,Y)
        BYi = _np.linalg.inv(BY)

        printer.log(" -> debug matrix:%s" % _mt.mx_to_string(_np.dot(B,_np.dot(real_gate_mx,Bi))), 4)
        printer.log(" -> debug Y-Mx  = %s" % _mt.mx_to_string(BY), 4)
        printer.log(" -> debug test BY*D*BYi: %s" % _mt.mx_to_string(_np.dot(BY,_np.dot(D,BYi))), 4)
    printer.log('', 4)

    new_evecs = _np.dot(evecs, permMx)
    return eval_params, new_evecs


def _get_delta0_diag(rho_tilde, E_tilde, D0_params, verbosity):
    """ Trys to set rhoTilde[i] == 1.0 """

    #Note:
    # rho-tilde has form inv(Y0) * rho, where inv(Y0) has conjugate-paired
    # rows as determined by the eigenvalue parameters D0_params and rho
    # is a real column vector.  Thus, inv(Y0) * rho is a complex column
    # vector with the conjugate-pair structure given by D0_params.  Thus,
    # the diagonal of inv(delta0) (or just delta0 since it's a diag mx)
    # will have this same structure, to preserve the conjugate-pair
    # structure of rho-tilde and the B-matrices.
    rho_tilde = rho_tilde.flatten() #so we can index using a single index
    E_tilde = E_tilde.flatten() #so we can index using a single index
    inv_delta0_diag = _np.empty(len(rho_tilde),'complex')

    printer = VerbosityPrinter.build_printer(verbosity)

    printer.log("Finding delta0", 4)
    printer.log(" rho-tilde = %s" % _mt.mx_to_string(rho_tilde), 4)
    printer.log(" E-tilde = %s"   % _mt.mx_to_string(E_tilde), 4)
    printer.log(" D0-params = %s" % _mt.mx_to_string(D0_params), 4)

    #We compute the diagonal of inv(delta0) which makes
    # inv(delta0) * rho_tilde = delta0 * E_tilde (complex case)
    # inv(delta0) * |rho_tilde| = delta0 * |E_tilde| (real case), so
    #  |rho_tilde_element| = delta0**2 * |corresponding_E_tilde_element|
    #  delta0 = sqrt( |rho_tilde_element| / |corresponding_E_tilde_element|)
    # OLD inv(delta0) * inv(Y0) * rho = vector of ones
    for i in range(0,len(D0_params)-1,2):
        _, b = D0_params[i:i+2]
        if b <= 0:
            # complex-conj pair at index i,i+1, so:
            assert(_np.isclose(rho_tilde[i],rho_tilde[i+1].conj()))
            if abs(rho_tilde[i]) < SMALL: # and E_tilde[i] not small??
                printer.warning("(1) scaling near-zero rho_tilde element to 1.0!")
                inv_delta0_diag[i] = 1.0/SMALL
            else:
                inv_delta0_diag[i] = (1.0+0j) / rho_tilde[i] #complex division
            inv_delta0_diag[i+1] = inv_delta0_diag[i].conj()
        else:
            # real pair at index i,i+1, so:
            assert(_isreal(rho_tilde[i]) and _isreal(rho_tilde[i+1]))
            if abs(rho_tilde[i]) < SMALL:
                printer.warning("(2) scaling near-zero rho_tilde element to 1.0!")
                inv_delta0_diag[i] = 1.0/SMALL
            else:
                inv_delta0_diag[i] = 1.0 / rho_tilde[i].real

            if abs(rho_tilde[i+1]) < SMALL:
                printer.warning("(3) scaling near-zero rho_tilde element to 1.0!")
                inv_delta0_diag[i+1] = 1.0/SMALL
            else:
                inv_delta0_diag[i+1] = 1.0 / rho_tilde[i+1].real

    if len(D0_params) % 2 == 1: #then there's an un-paired real eigenvalue
        i = len(D0_params)-1
        assert(_isreal(rho_tilde[i]))
        if abs(rho_tilde[i]) < SMALL:
            printer.log("(4) scaling near-zero rho_tilde element to 1.0!")
            inv_delta0_diag[i] = 1.0/SMALL
        else:
            inv_delta0_diag[i] = 1.0 / rho_tilde[i].real

    delta0_diag = 1.0 / inv_delta0_diag # delta0_diag (could compute directly above?)
    return delta0_diag, inv_delta0_diag


def _get_delta0_diag_mark2(rho_tilde, E_tilde, D0_params, fix, verbosity):
    """ Trys to set rhoTilde[i] == ETilde[i] """

    #Note:
    # rho-tilde has form inv(Y0) * rho, where inv(Y0) has conjugate-paired
    # rows as determined by the eigenvalue parameters D0_params and rho
    # is a real column vector.  Thus, inv(Y0) * rho is a complex column
    # vector with the conjugate-pair structure given by D0_params.  Thus,
    # the diagonal of inv(delta0) (or just delta0 since it's a diag mx)
    # will have this same structure, to preserve the conjugate-pair
    # structure of rho-tilde and the B-matrices.
    rho_tilde = rho_tilde.flatten() #so we can index using a single index
    E_tilde = E_tilde.flatten() #so we can index using a single index
    inv_delta0_diag = _np.empty(len(rho_tilde),'complex')

    printer = VerbosityPrinter.build_printer(verbosity)

    printer.log("Finding delta0", 4)
    printer.log(" rho-tilde = %s" % _mt.mx_to_string(rho_tilde), 4)
    printer.log(" E-tilde = %s" % _mt.mx_to_string(E_tilde), 4)
    printer.log(" D0-params = %s" % _mt.mx_to_string(D0_params), 4)

    #We compute the diagonal of inv(delta0) which makes
    # inv(delta0) * rho_tilde = delta0 * E_tilde (complex case)
    # inv(delta0) * |rho_tilde| = delta0 * |E_tilde| (real case), so
    #  |rho_tilde_element| = delta0**2 * |corresponding_E_tilde_element|
    #  delta0 = sqrt( |rho_tilde_element| / |corresponding_E_tilde_element|)
    # OLD inv(delta0) * inv(Y0) * rho = vector of ones
    for i in range(0,len(D0_params)-1,2):
        _, b = D0_params[i:i+2]
        if b <= 0:
            # complex-conj pair at index i,i+1, so:
            assert(_np.isclose(rho_tilde[i],rho_tilde[i+1].conj()))
            if abs(rho_tilde[i]) < SMALL: # and E_tilde[i] not small??
                printer.warning("(1) scaling near-zero rho_tilde element to 1.0!")
                inv_delta0_diag[i] = 1.0/SMALL
                inv_delta0_diag[i] = 1.0/SMALL
            else:
                inv_delta0_diag[i] = _np.sqrt(E_tilde[i] / rho_tilde[i]) #complex division
            inv_delta0_diag[i+1] = inv_delta0_diag[i].conj()
        else:
            # real pair at index i,i+1, so:
            assert(_isreal(rho_tilde[i]) and _isreal(rho_tilde[i+1]))
            if abs(rho_tilde[i]) < SMALL:
                if abs(E_tilde[i]) < SMALL:
                    #both rho and E are small - ok: leave delta0 == 1.0
                    inv_delta0_diag[i] = fix #1.0 # set sign?
                else:
                    printer.warning("(2) scaling near-zero rho_tilde element to 1.0!")
                    inv_delta0_diag[i] = 1.0/SMALL
            elif abs(E_tilde[i]) < SMALL:
                inv_delta0_diag[i] = fix #1.0 # set sign?
            else:
                inv_delta0_diag[i] = _np.sqrt(abs(E_tilde[i].real / rho_tilde[i].real))
                if (E_tilde[i].real * rho_tilde[i].real < 0 and E_tilde[i] > 0) or \
                   (E_tilde[i].real * rho_tilde[i].real >= 0 and E_tilde[i] < 0):
                    inv_delta0_diag[i] *= -1   # E_tilde[i] should be negative if rho & E
                                               #  are different signs, positive otherwise

            if abs(rho_tilde[i+1]) < SMALL:
                if abs(E_tilde[i+1]) < SMALL:
                    #both rho and E are small - ok: leave delta0 == 1.0
                    inv_delta0_diag[i+1] = fix #1.0 # set sign?
                else:
                    printer.warning("(3) scaling near-zero rho_tilde element to 1.0!")
                    inv_delta0_diag[i+1] = 1.0/SMALL
            elif abs(E_tilde[i+1]) < SMALL:
                inv_delta0_diag[i+1] = fix #1.0 # set sign?
            else:
                inv_delta0_diag[i+1] = _np.sqrt(abs(E_tilde[i+1].real / rho_tilde[i+1].real))
                if (E_tilde[i+1].real * rho_tilde[i+1].real < 0 and E_tilde[i+1] > 0) or \
                   (E_tilde[i+1].real * rho_tilde[i+1].real >= 0 and E_tilde[i+1] < 0):
                    inv_delta0_diag[i+1] *= -1


    if len(D0_params) % 2 == 1: #then there's an un-paired real eigenvalue
        i = len(D0_params)-1
        assert(_isreal(rho_tilde[i]))
        if abs(rho_tilde[i]) < SMALL:
            printer.warning("(4) scaling near-zero rho_tilde element to 1.0!")
            inv_delta0_diag[i] = 1.0/SMALL
        else:
            raise NotImplementedError("TODO - like above")

    #delta0_diag = 1.0 / inv_delta0_diag # delta0_diag (could compute directly above?)
    delta0_diag = _np.array( [ 1.0/x if x != 0.0 else 0.0 for x in inv_delta0_diag ] )
    return delta0_diag, inv_delta0_diag


def _get_ET_params(ET_tilde, D0_params):
    # ET_tilde is a (complex) row-vector with conjugate-pair structure given
    # by D0_params
    ET_tilde = ET_tilde.flatten() #so we can index using a single index
    E_params = _np.empty(len(ET_tilde),'d') #Note: *not* complex

    for i in range(0,len(D0_params)-1,2):
        _, b = D0_params[i:i+2]
        if b <= 0:
            # complex-conj pair at index i,i+1, so:
            assert(_np.isclose(ET_tilde[i],ET_tilde[i+1].conj()))
            E_params[i] = ET_tilde[i].real
            E_params[i+1] = ET_tilde[i].imag # (can be pos or neg)
            #TODO: maybe we divide by abs(b) before setting E_params??
        else:
            # real pair at index i,i+1, so:
            #NAN? assert(_isreal(ET_tilde[i]) and _isreal(ET_tilde[i+1]))
            E_params[i]   = (ET_tilde[i].real + ET_tilde[i+1].real)/2.0
            E_params[i+1] = (ET_tilde[i].real - ET_tilde[i+1].real)/2.0
            #TODO: maybe we divide by abs(b) before setting E_params??

    if len(D0_params) % 2 == 1: #then there's an un-paired real eigenvalue
        i = len(D0_params)-1
        assert(_isreal(ET_tilde[i]))
        E_params[i] = ET_tilde[i].real

    return E_params


def _rowsum(ar, D_params): # for rows where b > 0
    s = 0
    for j in range(0,len(D_params)-1,2): #loop over col-pairs
        _, b = D_params[j:j+2] #cols
        if b <= 0:
            # block is [ [a, a.C], [b, b.C] ], so add a.r + a.i
            s += ar[j].real + ar[j].imag
        else:
            # block is [ [a, b], [c, d] ], so add a + b (normal)
            s += ar[j] + ar[j+1]
    return s


def _get_B_params(invYjY0, Dj_params, D0_params, verbosity):
    #Need to find inv(deltaj) to fix rows ~ such that sum( abs(el) ) == 1
    # B0j == inv(deltaj) * invYjY0  (j != 0)
    # where invYjY0 has special "two-sided" (rows+cols) conjugacy structure
    # given by Dj_params (rows) and D0_params (cols).  Diagonal els of
    # inv(deltaj) will have conjugacy-pair structure of Dj_params so overall
    # structure of B0j is the same as that of invYjY0.  After deltaj element
    # are found, conjugacy-pair structure is used to extract (real) parameters
    # of B0j.
    printer = VerbosityPrinter.build_printer(verbosity)

    assert(len(Dj_params) == len(D0_params))
    inv_deltaj_diag = _np.empty(invYjY0.shape[0],'complex')

    def abssum(ar):
        return sum(map(abs, ar))
    def anglesum(ar):
        return sum(_np.where( _np.absolute(ar) > SMALL, _np.angle(ar), 0) )
    def first_nonzero_angle(ar):
        for x in ar:
            if abs(x) > SMALL:
                return _np.angle(x)
        raise ValueError("No nonzero angles found!")

    #DEBUG
    printer.log("DB: invYjY0 (abs,angle) sums = %s" % [ (abssum(invYjY0[i,:]), anglesum(invYjY0[i,:])) for i in range(invYjY0.shape[0])])


    #Record whether the final colum corresponds to a real eigenvalue
    bRealLastCol = (len(D0_params) % 2 == 1) or (D0_params[-1] > 0)

    #get "sums" of invYjY0 rows => choose inv_deltaj_diag accordingly
    for i in range(0,len(Dj_params)-1,2):
        _, b1 = Dj_params[i:i+2] #rows
        if b1 <= 0:
            # sum(row i) should be the conjugate of sum(row i+1)
            abss = abssum(invYjY0[i,:])
            angs = anglesum(invYjY0[i,:])
            assert(_np.isclose( abssum(invYjY0[i+1,:]), abss ))
            #assert(_np.isclose( anglesum(invYjY0[i+1,:]), -angs )) #need mod 2pi...
            nnz = sum([ 1 if abs(x) > SMALL else 0 for x in invYjY0[i,:]]) #num nonzero

            if abss < 1e-10:
                printer.warning("*** Warning *** scaling near-zero B0j row abssum to 1.0!")
                inv_deltaj_diag[i] = 1 / SMALL
            else:
                inv_deltaj_diag[i] = 1.0 / abss   # to make sum of abs(el) == 1
            angle = -angs / nnz                   # to make sum of angles == 0
            mod = 2*_np.pi / nnz

            # can add/subtract "mod" angle and still sum zero, so add/subtract
            # in order to make angle of first nonzero row element close to zero
            fnza = first_nonzero_angle(invYjY0[i,:]) + angle
            #print "DB: first nz angle = ", fnza, " (mod = %g)" % mod
            mod_fctr = 0
            while abs(fnza + mod) < abs(fnza):
                fnza += mod; mod_fctr += 1
            while abs(fnza - mod) < abs(fnza):
                fnza -= mod; mod_fctr -= 1
            angle += mod_fctr * mod
            #print "DB: final fnza = ",fnza, " mod_fctr = ",mod_fctr
            #print "DB: angle = ",angle

            inv_deltaj_diag[i] *= _np.exp(1j*angle)
            inv_deltaj_diag[i+1] = inv_deltaj_diag[i].conj()
        else:
            # sum(row i) and sum(row i+1) should be real
            # scale so that abssum == 1 AND final column is positive (if it's a "real" col)
            abss1 = abssum(invYjY0[i,:])
            abss2 = abssum(invYjY0[i+1,:])

            sign1 = -1 if (bRealLastCol and invYjY0[i,-1] < -SMALL) else 1
            sign2 = -1 if (bRealLastCol and invYjY0[i+1,-1] < -SMALL) else 1

            if abss1 < SMALL:
                printer.warning("*** Warning *** scaling near-zero B0j row abssum to 1.0!")
                inv_deltaj_diag[i] = sign1 * 1/SMALL
            else:
                inv_deltaj_diag[i] = sign1 * 1.0 / abss1

            if abss2 < SMALL:
                printer.warning("*** Warning *** scaling near-zero B0j row abssum to 1.0!")
                inv_deltaj_diag[i+1] = sign2 * 1/SMALL
            else:
                inv_deltaj_diag[i+1] = sign2 * 1.0 / abss2

    if len(Dj_params) % 2 == 1: #then there's an un-paired eigenvalue row
        i = len(Dj_params)-1
        abss = abssum(invYjY0[i,:])
        sign = -1 if (bRealLastCol and invYjY0[i,-1] < 0) else 1

        if abss < 1e-10:
            printer.warning("*** Warning *** scaling near-zero B0j row abssum to 1.0!")
            inv_deltaj_diag[i] = sign * 1/SMALL
        else:
            inv_deltaj_diag[i] = sign * 1.0 / abss

    # Scale invYjY0
    B0j = _np.dot( _np.diag(inv_deltaj_diag), invYjY0 )

    #DEBUG
    printer.log("DB: Bj0 = %s" %  _mt.mx_to_string(B0j), 4)
    printer.log("DB: Bj0 (abs,angle) sums = %s" % [ (abssum(B0j[i,:]), anglesum(B0j[i,:])) for i in range(B0j.shape[0])], 4)

    # Extract (real) parameters from B0j
    L = len(Dj_params)-2 #starting index of final 2x2 block (if one *is* final)
    B0j_params = {} # a dictionary of lists, indexed by two "2x2 block" indices
    for i in range(0,len(Dj_params)-1,2): #loop over row-pairs
        _, b1 = Dj_params[i:i+2] #rows
        for j in range(0,len(D0_params)-1,2): #loop over col-pairs
            _, b2 = D0_params[j:j+2] #cols

            #Each 2x2 square of B0j contains 4 real parameters (after
            # accounting for structure) *except* if j == L, in which case
            # the deltaj-scaling has removed two of these, leaving only 2.
            if b1 <= 0:

                #NOTE: since both b2 cases do exacty the same thing (because
                # a and b lie in the same positions ([i,i] and [i,i+1]) )
                # there's no need to put the code below into the b2 if blocks.
                if j != L: # parameterize a.r, a.i, b.r, b.i
                    B0j_params[i//2,j//2] = \
                        [ B0j[i,j].real, B0j[i,j].imag,
                          B0j[i,j+1].real, B0j[i,j+1].imag ]
                else: # b == 1.0-sum so just parameterize a.r, a.i
                    B0j_params[i//2,j//2] = \
                        [ B0j[i,j].real, B0j[i,j].imag ]
                #TODO: maybe we divide by abs(b) before setting??

                if b2 <= 0:
                    # complex-conj pair rows & cols, so
                    # block is [ [a, b], [b.C, a.C] ]; parameterize a then b.
                    assert(_np.isclose(B0j[i,j], B0j[i+1,j+1].conj()))
                    assert(_np.isclose(B0j[i+1,j], B0j[i,j+1].conj()))
                else:
                    # complex-conj pair rows & real-pair cols, so
                    # block is [ [a, b], [a.C, b.C] ]; parameterize a then b.
                    assert(_np.isclose(B0j[i,j], B0j[i+1,j].conj()))
                    assert(_np.isclose(B0j[i,j+1], B0j[i+1,j+1].conj()))

            else:
                if b2 <= 0:
                    # real-pair rows & complex-conj pair cols,
                    # so block is [ [a, a.C], [b, b.C] ]
                    assert(_np.isclose(B0j[i,j], B0j[i,j+1].conj()))
                    assert(_np.isclose(B0j[i+1,j], B0j[i+1,j+1].conj()))

                    if j != L: # parameterize a.r, a.i, b.r, b.i
                        B0j_params[i//2,j//2] = \
                            [ B0j[i,j].real, B0j[i,j].imag,
                              B0j[i+1,j].real, B0j[i+1,j].imag ]
                    else: # 2|a| == 1-sum1, 2|b| == 1-sum2, so angle(a), angle(b)
                        B0j_params[i//2,j//2] = \
                            [ _angle(B0j[i,j]), _angle(B0j[i+1,j]) ]
                    #TODO: maybe we divide by abs(b) before setting??

                else:
                    # real-pair rows & cols, so block is
                    # [ [a, b], [c, d] ] (all real)
                    assert(_isreal(B0j[i,j]) and _isreal(B0j[i,j+1]) and
                           _isreal(B0j[i+1,j]) and _isreal(B0j[i+1,j+1]))

                    if j != L: # parameterize a, b, c, d
                        B0j_params[i//2,j//2] = \
                            [ B0j[i,j].real, B0j[i,j+1].real,
                              B0j[i+1,j].real, B0j[i+1,j+1].real ]
                    else: # |b| == 1.0-sum1, b>0, |d| == 1.0-sum2, d>0, so just a, c
                        B0j_params[i//2,j//2] = \
                            [ B0j[i,j].real, B0j[i+1,j].real ]
                    #TODO: maybe we divide by abs(b) before setting??

    if len(Dj_params) % 2 == 1: #then there's an un-paired eigenvalue row
        M = len(Dj_params)//2 # index of unpaired block in B0j_params
        i = len(Dj_params)-1
        for j in range(0,len(D0_params)-1,2): #loop over col-pairs (final row)
            _, b2 = D0_params[j:j+2] #cols
            # (note j can never equal L b/c unpaired col exists)

            if b2 <= 0:
                # single-real row & complex-conj pair cols,
                # so block is [a, a.C]; parameterize a
                assert(_np.isclose(B0j[i,j], B0j[i,j+1].conj()))
                B0j_params[M,j//2] = \
                    [ B0j[i,j].real, B0j[i,j].imag ]
                #TODO: maybe we divide by abs(b) before setting??

            else:
                # single-real row & real-pair cols, so block is
                # [a, b] (both real); parameterize a, b
                assert(_isreal(B0j[i,j]) and _isreal(B0j[i,j+1]))
                B0j_params[M,j//2] = \
                    [ B0j[i,j].real, B0j[i,j+1].real ]
                #TODO: maybe we divide by abs(b) before setting??

        j = len(D0_params)-1
        assert(j == i) #since D?_params are the same length
        for i in range(0,len(Dj_params)-1,2): #loop over row-pairs (final col)
            _, b1 = Dj_params[i:i+2] #cols
            # Note j is always the final column here - so the two parameters of
            # each block are set by 1.0-sum of the relevant column.  Thus, there
            # are *no* parameters to add for this final column.

            if b1 <= 0:
                # complex-conj pair rows & single-real col,
                # so block is [[a], [a.C]]; parameterize a
                # (note i can never equal j)
                assert(_np.isclose(B0j[i,j], B0j[i+1,j].conj()))
            else:
                # real-pair rows & single-real col, so block is
                # [[a], [b]] (both real); parameterize a, b
                assert(_isreal(B0j[i,j]) and _isreal(B0j[i+1,j]))

        #Now deal with final diagonal element (single real row & col). The
        # deltaj scaling has set this element to 1-sum as well, so no more
        # parameters are needed.
        #B0j_params[M,M] = [ ] # just so this list exists for loop below?
    else:
        M = None #there is no unpaired block, so no index for it!

    #Now collect B0j_params into a numpy array by concatenating blocks.
    # (This always puts the diagonal blocks into the same locations, so
    #  there's no ambiguity about the size of each segment)
    concat_params = []
    n2x2 = len(Dj_params)//2 # number of 2x2 blocks
    for k in range(n2x2):
        for l in range(n2x2):
            if l == (L+1)//2: assert(len(B0j_params[k,l]) == 2)
            else: assert(len(B0j_params[k,l]) == 4)
            concat_params.extend( B0j_params[k,l] )

    if len(Dj_params) % 2 == 1: #then there's an un-paired eigenvalue
        #Add final row
        for l in range(n2x2):
            assert(len(B0j_params[M,l]) == 2)
            concat_params.extend( B0j_params[M,l] )

        #Add final col (but no parameters!)
        #for k in range(n2x2):
        #    assert(len(B0j_params[k,M]) == 2)
        #    concat_params.extend( B0j_params[k,M] )

    assert( all(_np.isclose(_np.imag(concat_params), 0.0) ))
    return _np.array(concat_params, 'd')


def _constructDj(Dj_params):
    #Construct Dj
    Dj_diag = _np.empty( len(Dj_params), 'complex' )
    for i in range(0,len(Dj_params)-1,2):
        a,b = Dj_params[i:i+2]
        if b <= 0: # complex-conj pair
            Dj_diag[i]   = a-b*1j #a + abs(b)*1j
            Dj_diag[i+1] = a+b*1j #a - abs(b)*1j
        else: # real-pair
            Dj_diag[i]   = a+b
            Dj_diag[i+1] = a-b
    if len(Dj_params) % 2 == 1: # un-paired real eigenvalue
        Dj_diag[-1] = Dj_params[-1]
    return _np.diag(Dj_diag)

def _get_rhoTilde(ETilde, D0_params):
    """ For rhoTilde[i] == 1.0 assumption """
    return _np.ones( (len(D0_params),1), 'complex' )

def _get_rhoTilde_mark2(ETilde, D0_params):
    """ For rhoTilde[i] == ETilde[i] assumption """
    rho_tilde = _np.empty( (len(D0_params),1),'complex')
    for i in range(0,len(D0_params)-1,2):
        _, b = D0_params[i:i+2]
        if b <= 0:
            # complex-conj pair at index i,i+1, so:
            rho_tilde[i] = ETilde[i]
            rho_tilde[i+1] = ETilde[i+1]
        else:
            # real pair at index i,i+1, so:
            rho_tilde[i] = ETilde[i] if ETilde[i] >= 0 else -ETilde[i]
            rho_tilde[i+1] = ETilde[i+1] if ETilde[i+1] >= 0 else -ETilde[i+1]
            #TODO: maybe we divide by abs(b) before setting E_params??

    if len(D0_params) % 2 == 1: #then there's an un-paired real eigenvalue
        rho_tilde[len(D0_params)-1] = Etilde[len(D0_params)-1] if Etilde[len(D0_params)-1] >= 0 else -Etilde[len(D0_params)-1] #pylint: disable=undefined-variable

    return rho_tilde

def _deparameterize_real_gate_mx(Dj_params, D0_params, B0j_params,
                                 invY0, verbosity):
    printer = VerbosityPrinter.build_printer(verbosity)
    assert(len(Dj_params) == len(D0_params))

    def abssum(ar):
        return sum(map(abs,ar))
    def anglesum(ar):
        return sum(_np.where( _np.absolute(ar) > SMALL, _np.angle(ar), 0) )

    #def rowsum(ar, ar_end):
     #   full = list(ar) + list(ar_end)
      #  return _rowsum(full, D0_params)

    #Construct Dj
    Dj = _constructDj(Dj_params)

    #Construct Yj, inv(Yj)
    # B0j := inv(Yj)Y0, so deparameterize B0j then apply inv(Y0)
    B0j = _np.empty( (len(Dj_params),len(Dj_params)), 'complex' )
    L = len(Dj_params)-2 #starting index of final 2x2 block (if one *is* final)

    k = 0 #running index into B0j_params
    for i in range(0,len(Dj_params)-1,2): #loop over row-pairs
        _, b1 = Dj_params[i:i+2] #rows
        for j in range(0,len(D0_params)-1,2): #loop over col-pairs
            _, b2 = D0_params[j:j+2] #cols

            #Each 2x2 square of B0j is specified with 4 real parameters
            # *except* if j == L, in which case our deltaj-scaling has
            # removed two of these, leaving only 2.
            nP = 2 if (j == L) else 4
            params = B0j_params[k:k+nP]
            k += nP

            if b1 <= 0:
                #Get parameters of 2x2 block (which we call a and b below)
                if j == L: #just a parameterized
                    ar,ai = params
                    a = ar + ai*1j
                    abs_b = 1.0 - (abssum(B0j[i,0:L]) + abs(a))
                    angle_b = -1.0 * (anglesum(B0j[i,0:L]) + _angle(a))
                    b = abs_b * _np.exp(1j*angle_b)
                else: # a then b parameterized
                    ar,ai,br,bi = params
                    a = ar + ai*1j
                    b = br + bi*1j

                if b2 <= 0:
                    # complex-conj pair rows & cols, so
                    # block is [ [a, b], [b.C, a.C] ]
                    B0j[i,j:j+2]   = a       , b
                    B0j[i+1,j:j+2] = b.conj(), a.conj()
                else:
                    # complex-conj pair rows & real-pair cols, so
                    # block is [ [a, b], [a.C, b.C] ]
                    B0j[i,j:j+2]   = a       , b
                    B0j[i+1,j:j+2] = a.conj(), b.conj()

            else:
                if b2 <= 0:
                    # real-pair rows & complex-conj pair cols,
                    # so block is [ [a, a.C], [b, b.C] ]

                    #Get parameters of 2x2 block (a and b)
                    if j == L: #a.r+a.i == 1-sum1, b.r+b.i == 1-sum2 ; a.i, b.i
                        angle_a,angle_b = params
                        abs_a = 0.5 * (1.0 - abssum(B0j[i,0:L]))
                        abs_b = 0.5 * (1.0 - abssum(B0j[i+1,0:L]))
                        a = abs_a * _np.exp(1j*angle_a)
                        b = abs_b * _np.exp(1j*angle_b)

                    else: # a then b parameterized
                        ar,ai,br,bi = params
                        a = ar + ai*1j
                        b = br + bi*1j

                    B0j[i,j:j+2]   = a , a.conj()
                    B0j[i+1,j:j+2] = b , b.conj()
                    #TODO: maybe we divide by abs(b) before setting??

                else:
                    # real-pair rows & cols, so block is
                    # [ [a, b], [c, d] ] (all real)

                    #Get parameters of 2x2 block (a, b, c, d)
                    if j == L: # b == 1.0-sum1, d == 1.0-sum2, so just a, c
                        # real part of full row-sum == 1.0
                        a,c = params # AND we've scaled so b, d are positive
                        b = 1.0 - (abssum(B0j[i,0:L]) + abs(a))
                        d = 1.0 - (abssum(B0j[i+1,0:L]) + abs(c))
                    else: # a, b, c, d parameterized
                        a,b,c,d = params

                    B0j[i,j:j+2]   = a , b
                    B0j[i+1,j:j+2] = c , d
                    #TODO: maybe we divide by abs(b) before setting??

    if len(Dj_params) % 2 == 1: #then there's an un-paired eigenvalue row
        i = len(Dj_params)-1
        for j in range(0,len(D0_params)-1,2): #loop over col-pairs
            _, b2 = D0_params[j:j+2] #cols
            params = B0j_params[k:k+2]; k += 2 #always length 2 (j != L always)

            if b2 <= 0:
                # single-real row & complex-conj pair cols,
                # so block is [a, a.C]
                ar,ai = params
                B0j[i,j]   = a + ai*1j
                B0j[i,j+1] = a - ai*1j
                #TODO: maybe we divide by abs(b) before setting??

            else:
                # single-real row & real-pair cols, so block is
                # [a, b] (both real)
                a,b = params
                B0j[i,j]   = a
                B0j[i,j+1] = b
                #TODO: maybe we divide by abs(b) before setting??

        j = len(D0_params)-1
        assert(j == i) #since D?_params are the same length
        for i in range(0,len(Dj_params)-1,2): #loop over row-pairs (final col)
            _, b1 = Dj_params[i:i+2] #cols
            # No params (j == last col, always) - all determined by
            #  rows-wsum-to-one scaling

            if b1 <= 0:
                # complex-conj pair rows & single-real col,
                # so block is [[a], [a.C]]
                abs_a = 1.0 - abssum(B0j[i,0:j])
                angle_a = -anglesum(B0j[i,0:j])
                a = abs_a * _np.exp(1j*angle_a)
                B0j[i,j]   = a
                B0j[i+1,j] = a.conj()
                #TODO: maybe we divide by abs(b) before setting??

            else:
                # real-pair rows & single-real col, so block is
                # [[a], [b]] (both real -- AND positive b/c of scaling)
                a = 1.0 - abssum(B0j[i,0:j])
                b = 1.0 - abssum(B0j[i+1,0:j])
                B0j[i,j]   = a
                B0j[i+1,j] = b
                #TODO: maybe we divide by abs(b) before setting??

        #Now deal with final diagonal element (single real row & col). The
        # deltaj scaling has set the abs-sum of this row to be 1.0 and this
        # element to be positive, so set:
        B0j[j,j] = 1.0 - abssum(B0j[j,0:j])

    invYj = _np.dot( B0j, invY0) # b/c B0j = inv(Yj)Y0
    Yj = _np.linalg.inv(invYj)

    printer.log("De-parameterizing gate:", 4)
    printer.log("B0j = %s" % _mt.mx_to_string(B0j), 4)
    printer.log("Yj = %s" % _mt.mx_to_string(Yj), 4)
    printer.log("invYj = %s" % _mt.mx_to_string(invYj), 4)
    printer.log("Dj = %s" % _mt.mx_to_string(Dj))
    printer.log("mx = %s" % _mt.mx_to_string( _np.dot(Yj, _np.dot(Dj, invYj))), 4)

    #Construct gate
    mx = _np.dot(Yj, _np.dot(Dj, invYj))
    assert(all(_isreal(mx.flatten())))

    return _np.real(mx)


def _get_ETilde_vector( E_params, D0_params ):
    E_tilde = _np.empty( (len(E_params),1),'complex')

    for i in range(0,len(D0_params)-1,2):
        _, b = D0_params[i:i+2]
        if b <= 0:
            # complex-conj pair at index i,i+1, so:
            Er,Ei = E_params[i:i+2]
            E_tilde[i]   = Er + Ei*1j
            E_tilde[i+1] = Er - Ei*1j
            #TODO: maybe we divide by abs(b) before setting E_params??
        else:
            # real pair at index i,i+1, so:
            E1,E2 = E_params[i:i+2]
            E_tilde[i]   = E1 + E2
            E_tilde[i+1] = E1 - E2
            #TODO: maybe we divide by abs(b) before setting E_params??

    if len(D0_params) % 2 == 1: #then there's an un-paired real eigenvalue
        E_tilde[len(D0_params)-1] = E_params[len(D0_params)-1]

    return E_tilde
