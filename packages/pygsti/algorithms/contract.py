from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" GST contraction algorithms """

import numpy as _np
import warnings as _warnings

from .. import objects as _objs
from .. import tools as _tools
from .. import optimize as _opt

def contract(gateset, toWhat, dataset=None, maxiter=1000000, tol=0.01, useDirectCP=True, method="Nelder-Mead", verbosity=0):
    """
    Contract a GateSet to a specified space.

    All contraction operations except 'vSPAM' operate entirely on the gate
    matrices and leave state preparations and measurments alone, while 'vSPAM'
    operations only on SPAM.

    Parameters
    ----------
    gateset : GateSet
        The gateset to contract

    toWhat : string
        Specifies which space is the gateset is contracted to.
        Allowed values are:

        - 'TP'     -- All gates are manifestly trace-preserving maps.
        - 'CP'     -- All gates are manifestly completely-positive maps.
        - 'CPTP'   -- All gates are manifestly completely-positive and trace-preserving maps.
        - 'XP'     -- All gates are manifestly "experimentally-positive" maps.
        - 'XPTP'   -- All gates are manifestly "experimentally-positive" and trace-preserving maps.
        - 'vSPAM'  -- state preparation and measurement operations are valid.
        - 'nothing' -- no contraction is performed.

    dataset : DataSet, optional
        Dataset to use to determine whether a gateset is in the
        "experimentally-positive" (XP) space.  Required only when
        contracting to XP or XPTP.

    maxiter : int, optional
        Maximum number of iterations for iterative contraction routines.

    tol : float, optional
        Tolerance for iterative contraction routines.

    useDirectCP : bool, optional
        Whether to use a faster direct-contraction method for CP
        contraction.  This method essentially transforms to the
        Choi matrix, truncates any negative eigenvalues to zero,
        then transforms back to a gate matrix.

    method : string, optional
        The method used when contracting to XP and non-directly to CP
        (i.e. useDirectCP == False).

    verbosity : int, optional
        How much detail to send to stdout.

    Returns
    -------
    GateSet
        The contracted gateset
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    if toWhat == 'CPTP':
        if useDirectCP:
            distance,contractedGateset = _contractToCP_direct(gateset, printer, TPalso=True, maxiter=maxiter)
        else:
            distance,contractedGateset = _contractToTP(gateset,verbosity)
            distance,contractedGateset = _contractToCP(contractedGateset, printer, method, maxiter, tol)
    elif toWhat == 'XPTP':
        if dataset is None: raise ValueError("dataset must be given to contract to " + toWhat)
        distance,contractedGateset = _contractToTP(gateset,verbosity)
        distance,contractedGateset = _contractToXP(contractedGateset, dataset,verbosity, method, maxiter, tol)
    elif toWhat == 'CP':
        if useDirectCP:
            distance,contractedGateset = _contractToCP_direct(gateset, printer, TPalso=False, maxiter=maxiter)
        else:
            distance,contractedGateset = _contractToCP(gateset, printer, method, maxiter, tol)
    elif toWhat == 'TP':
        distance,contractedGateset = _contractToTP(gateset,verbosity)
    elif toWhat == 'XP':
        if dataset is None: raise ValueError("dataset must be given to contract to " + toWhat)
        distance,contractedGateset = _contractToXP(gateset,dataset,verbosity,method,maxiter,tol)
    elif toWhat == 'vSPAM':
        contractedGateset = _contractToValidSPAM(gateset, printer)
    elif toWhat == 'nothing':
        contractedGateset = gateset.copy()
    else: raise ValueError("Invalid contract argument: %s" % toWhat)

    return contractedGateset


#modifies gates only (not rhoVecs or EVecs = SPAM)
def _contractToXP(gateset,dataset,verbosity,method='Nelder-Mead',
                 maxiter=100000, tol=1e-10):

    CLIFF = 10000

    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    #printer.log('', 2)
    printer.log("--- Contract to XP ---", 1)
    gs = gateset.copy() #working copy that we keep overwriting with vectorized data

    def objective_func(vectorGS):
        gs.from_vector(vectorGS)
        forbiddenProbPenalty = _tools.forbidden_prob(gs,dataset)
        return (CLIFF + forbiddenProbPenalty if forbiddenProbPenalty > 1e-10 else 0) \
            + gs.frobeniusdist(gateset)

    bToStdout = (printer.verbosity > 2 and printer.filename is None)
    print_obj_func = _opt.create_obj_func_printer(objective_func) #only ever prints to stdout!
    if objective_func(gs.to_vector()) < 1e-8:
        printer.log('Already in XP - no contraction necessary', 1)
        return 0.0, gs

    optSol = _opt.minimize(objective_func,gs.to_vector(),
                          method=method, tol=tol, maxiter=maxiter,
                          callback = print_obj_func if bToStdout else None)

    gs.from_vector(optSol.x)
    #gs.log("Contract to XP", { 'method': method, 'tol': tol, 'maxiter': maxiter } )
    if optSol.fun >= CLIFF: _warnings.warn("Failed to contract gateset to XP")

    printer.log('The closest legal point found was distance: ' + str(optSol.fun), 1)

    return optSol.fun, gs

#modifies gates only (not rhoVecs or EVecs = SPAM)
def _contractToCP(gateset,verbosity,method='Nelder-Mead',
                 maxiter=100000, tol=1e-2):

    CLIFF = 10000
    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    #printer.log('', 2)
    printer.log("--- Contract to CP ---", 1)
    gs = gateset.copy() #working copy that we keep overwriting with vectorized data
    mxBasis = gs.get_basis_name()
    basisDim = gs.get_basis_dimension()

    def objective_func(vectorGS):
        gs.from_vector(vectorGS)
        gs.set_basis(mxBasis,basisDim) #set basis for jamiolkowski iso
        cpPenalty = _tools.sum_of_negative_choi_evals(gs) * 1000
        return (CLIFF + cpPenalty if cpPenalty > 1e-10 else 0) + gs.frobeniusdist(gateset)

    bToStdout = (printer.verbosity > 2 and printer.filename is None)
    print_obj_func = _opt.create_obj_func_printer(objective_func) #only ever prints to stdout!
    if objective_func(gs.to_vector()) < 1e-8:
        printer.log('Already in CP - no contraction necessary', 1)
        return 0.0, gs

    optSol = _opt.minimize(objective_func,gs.to_vector(),
                          method=method, tol=tol, maxiter=maxiter,
                          callback = print_obj_func if bToStdout else None)

    gs.from_vector(optSol.x)
    #gs.log("Contract to CP", { 'method': method, 'tol': tol, 'maxiter': maxiter } )
    if optSol.fun >= CLIFF: _warnings.warn("Failed to contract gateset to CP")

    printer.log('The closest legal point found was distance: ' + str(optSol.fun), 1)

    return optSol.fun, gs


#modifies gates only (not rhoVecs or EVecs = SPAM)
def _contractToCP_direct(gateset,verbosity,TPalso=False,maxiter=100000,tol=1e-8):

    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    gs = gateset.copy() #working copy that we keep overwriting with vectorized data
    mxBasis = gs.get_basis_name()
    #printer.log('', 1)
    printer.log(("--- Contract to %s (direct) ---" % ("CPTP" if TPalso else "CP")), 1)

    for (gateLabel,gate) in gateset.gates.items():
        new_gate = gate.copy()
        if(TPalso):
            for k in range(new_gate.shape[1]): new_gate[0,k] = 1.0 if k == 0 else 0.0

        Jmx = _tools.jamiolkowski_iso(new_gate,gateMxBasis=mxBasis,choiMxBasis="gm")
        evals,evecs = _np.linalg.eig(Jmx)

        if TPalso:
            assert( abs( sum(evals) - 1.0 ) < 1e-8 ) #check that Jmx always has trace == 1
        #if abs( sum(evals) - 1.0 ) >= 1e-8: #DEBUG
        #  print "WARNING: JMx given with evals = %s (sum = %s != 1)" % (evals,sum(evals))
        #  print "WARNING: JMx from: "; _tools.print_mx(new_gate)


        it = 0
        while min(evals) < -tol or abs( sum(evals) - 1.0 ) >= tol:

            #Project eigenvalues to being all positive
            new_evals = evals[:]

            #New projection code
            new_evals = [ max(ev.real,0) for ev in new_evals ]  #don't need .real in theory, but small im parts can snowball in practice
            total_shift = 1.0 - sum(new_evals)  #amount (usually/always < 0) needed to add to eigenvalues to make sum == 1
            sorted_evals_with_inds = sorted( enumerate(new_evals), key=lambda x: x[1] ) # (index,eval) tuples sorted by eval

            shift_left = total_shift
            evals_left = len(sorted_evals_with_inds)
            ideal_shift = shift_left / evals_left

            for (i, sorted_eval) in sorted_evals_with_inds: #loop over new_evals from smallest to largest (note all > 0)
                evals_left -= 1  #number of eigenvalue beyond current eval (in sorted order)
                if sorted_eval+ideal_shift >= 0:
                    new_evals[i] = sorted_eval + ideal_shift
                    shift_left -= ideal_shift
                elif evals_left > 0:
                    new_evals[i] = 0
                    shift_left += sorted_eval
                    ideal_shift = shift_left / evals_left #divide remaining shift evenly among remaining eigenvalues
                else: #last eigenvalue would be < 0 with ideal shift and can't set == 0 b/c all others must be zero too
                    new_evals[i] = 1.0 # so set what was the largest eigenvalue == 1.0

            #if abs( sum(new_evals) - 1.0 ) >= 1e-8:              #DEBUG
            #  print "DEBUG: sum(new_evals) == ",sum(new_evals)   #DEBUG
            #  print "DEBUG: new_evals == ",new_evals             #DEBUG
            #  print "DEBUG: orig evals == ",evals                #DEBUG
            assert( abs( sum(new_evals) - 1.0 ) < 1e-8 )


#  OLD projection code -- can runaway, and take many iters
#        while min(new_evals) < 0:
#          new_evals = [ max(ev.real,0) for ev in new_evals ]  #don't need .real in theory, but small im parts can snowball in practice
#          inds_to_shift = [i for (i,ev) in enumerate(new_evals) if ev > 0]
#          assert(len(inds_to_shift) > 0)
#
#          shift = (1.0 - sum(new_evals))/float(len(inds_to_shift))
#          for i in inds_to_shift: new_evals[i] += shift
#
#          if abs( sum(new_evals) - 1.0 ) >= 1e-8:              #DEBUG
#            print "DEBUG: sum(new_evals) == ",sum(new_evals)   #DEBUG
#            print "DEBUG: new_evals == ",new_evals             #DEBUG
#
#          assert( abs( sum(new_evals) - 1.0 ) < 1e-8 )

            new_Jmx = _np.dot(evecs, _np.dot( _np.diag(new_evals), _np.linalg.inv(evecs) ) )

            #Make trace preserving by zeroing out real parts of off diagonal blocks and imaginary parts
            #  within diagaonal 1x1 and 3x3 block (so really just the 3x3 block's off diag elements)
            #assert(new_Jmx.shape == (4,4)) #NOTE: only works for 1-qubit case so far
            kmax = new_Jmx.shape[0]
            for k in range(1,kmax):
                new_Jmx[0,k] = 1j*new_Jmx[0,k].imag
                new_Jmx[k,0] = 1j*new_Jmx[k,0].imag
            for i in range(1,kmax):
                for j in range(1,kmax):
                    new_Jmx[i,j] = new_Jmx[i,j].real

            evals,evecs = _np.linalg.eig(new_Jmx)

            #DEBUG
            #EVAL_TOL = 1e-10
            #if abs( sum(evals) - 1.0 ) >= 1e-8:
            #  print "DEBUG2: sum(evals) == ",sum(evals)
            #  print "DEBUG2: evals == ",evals
            #if min(evals) < -EVAL_TOL:
            #  print "DEBUG3: evals = ",evals

            assert( min(evals) >= -1e-10 and abs( sum(evals) - 1.0 ) < 1e-8) #Check that trace-trunc above didn't mess up positivity

            new_gate = _tools.jamiolkowski_iso_inv(new_Jmx,gateMxBasis=mxBasis,choiMxBasis="gm")

            #Old way of enforcing TP -- new way should be better since it's not iterative, but keep this around just in case.
            #  new_gate = _tools.jamiolkowski_iso_inv(new_Jmx)
            #
            #  if(TPalso):
            #    for k in range(new_gate.shape[1]):
            #      #if k == 0: assert( abs(new_gate[0,k] - 1.0) < 1e-8 )
            #      #else: assert( abs(new_gate[0,k]) < 1e-8 )
            #      new_gate[0,k] = 1.0 if k == 0 else 0.0
            #
            #  new_Jmx = _tools.jamiolkowski_iso(new_gate)
            #  evals,evecs = _np.linalg.eig(new_Jmx)

            it += 1
            if it > maxiter: break

        gs.gates[gateLabel] = _objs.FullyParameterizedGate( new_gate )

        if it > maxiter:
            printer.warning("Max iterations exceeded in contract_to_cp_direct")
        #else: print "contract_to_cp_direct success in %d iterations" % it  #DEBUG

        printer.log("Direct CP contraction of %s gate gives frobenius diff of %g" % \
              (gateLabel, _tools.frobeniusdist(gs.gates[gateLabel],gate)), 2)

    #gs.log("Choi-Truncate to %s" % ("CPTP" if TPalso else "CP"), { 'maxiter': maxiter } )
    distance = gs.frobeniusdist(gateset)
    printer.log(('The closest legal point found was distance: %s' % str(distance)), 1)

    if TPalso: #TP also constrains prep vectors
        gate_dim = gs.get_dimension()
        for rhoVec in list(gs.preps.values()):
            rhoVec[0,0] = 1.0 / gate_dim**0.25

    return distance, gs


#modifies gates only (not rhoVecs or EVecs = SPAM)
def _contractToTP(gateset,verbosity):
    printer = _objs.VerbosityPrinter.build_printer(verbosity)
    #printer.log('', 2)
    printer.log("--- Contract to TP ---", 1)
    gs = gateset.copy()
    for gate in list(gs.gates.values()):
        gate[0,0] = 1.0
        for k in range(1,gate.shape[1]): gate[0,k] = 0.0

    gate_dim = gs.get_dimension()
    for rhoVec in list(gs.preps.values()):
        rhoVec[0,0] = 1.0 / gate_dim**0.25

    distance = gs.frobeniusdist(gateset)
    printer.log(('Projected TP gateset was at distance: %g' % distance), 1)

    #gs.log("Contract to TP")
    return distance, gs


#modifies rhoVecs and EVecs (SPAM) only (not gates)
def _contractToValidSPAM(gateset, verbosity=0):
    """
    Contract the surface preparation and measurement operations of
    a GateSet to the space of valid quantum operations.

    Parameters
    --------------------
    gateset : GateSet
        The gateset to contract

    verbosity : int
        How much detail to send to stdout.

    Returns
    -------
    GateSet
        The contracted gateset
    """

    printer = _objs.VerbosityPrinter.build_printer(verbosity)

    TOL = 1e-9
    gs = gateset.copy()

    # ** assumption: only the first vector element of pauli vectors has nonzero trace
    dummyVec = _np.zeros( (gateset.get_dimension(),1), 'd'); dummyVec[0,0] = 1.0
    firstElTrace = _np.real( _tools.trace(_tools.ppvec_to_stdmx(dummyVec)))  # == sqrt(2)**nQubits
    diff = 0

    # rhoVec must be positive semidefinite and trace = 1
    for prepLabel,rhoVec in gs.preps.items():
        vec = rhoVec.copy()

        #Ensure trace == 1.0 (maybe later have this be optional)
        firstElTarget = 1.0 / firstElTrace    #TODO: make this function more robust
                                              # to multiple rhovecs -- can only use on ratio,
                                              # so maybe take average of ideal ratios for each rhoVec
                                              # and apply that?  The function works fine now for just one rhovec.
        if abs(firstElTarget - vec[0,0]) > TOL:
            r = firstElTarget / vec[0,0]
            vec *= r  #multiply rhovec by factor
            for ELabel,EVec in gs.effects.items():
                gs.effects[ELabel] = EVec / r

        mx = _tools.ppvec_to_stdmx(vec)

        #Ensure positive semidefinite
        lowEval = min( [ev.real for ev in _np.linalg.eigvals( mx ) ])
        while(lowEval < -TOL):
            idEl = vec[0,0] #only element with trace (even for multiple qubits) -- keep this constant and decrease others
            vec /= 1.00001; vec[0,0] = idEl
            lowEval = min( [ev.real for ev in _np.linalg.eigvals( _tools.ppvec_to_stdmx(vec) ) ])

        diff += _np.linalg.norm( gateset.preps[prepLabel] - vec )
        gs.preps[prepLabel] = vec

    # EVec must have eigenvals between 0 and 1 <==> positive semidefinite and trace <= 1
    for ELabel,EVec in gs.effects.items():
        evals,evecs = _np.linalg.eig( _tools.ppvec_to_stdmx(EVec) )
        if(min(evals) < 0.0 or max(evals) > 1.0):
            if all([ev > 1.0 for ev in evals]):
                evals[ evals.argmin() ] = 0.0 #at least one eigenvalue must be != 1.0
            if all([ev < 0.0 for ev in evals]):
                evals[ evals.argmax() ] = 1.0 #at least one eigenvalue must be != 0.0
            for (k,ev) in enumerate(evals):
                if ev < 0.0: evals[k] = 0.0
                if ev > 1.0: evals[k] = 1.0
            mx = _np.dot(evecs, _np.dot( _np.diag(evals), _np.linalg.inv(evecs) ) )
            vec = _tools.stdmx_to_ppvec(mx)
            diff += _np.linalg.norm( gateset.effects[ELabel] - vec )
            gs.effects[ELabel] = vec

    #gs.log("Contract to valid SPAM")
    #printer.log('', 2)
    printer.log("--- Contract to valid SPAM ---", 1)
    printer.log(("Sum of norm(deltaE) and norm(deltaRho) = %g" % diff), 1)
    for (prepLabel,rhoVec) in gateset.preps.items():
        printer.log("  %s: %s ==> %s " % (prepLabel, str(_np.transpose(rhoVec)),
                                   str(_np.transpose(gs.preps[prepLabel]))), 2)
    for (ELabel,EVec) in gateset.effects.items():
        printer.log("  %s: %s ==> %s " % (ELabel, str(_np.transpose(EVec)),
                                    str(_np.transpose(gs.effects[ELabel]))), 2)

    return gs #return contracted gateset
