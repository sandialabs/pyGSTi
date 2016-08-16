from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" A custom conjugate gradient descent algorithm """

import numpy as _np

try:
    from scipy.optimize import Result as _optResult #for earlier scipy versions
except:
    from scipy.optimize import OptimizeResult as _optResult #for later scipy versions


def fmax_cg(f, x0, maxiters=100, tol=1e-8, dfdx_and_bdflag = None, xopt=None):
    """
    Custom conjugate-gradient (CG) routine for maximizing a function.

    This function runs slower than scipy.optimize's 'CG' method, but doesn't
    give up or get stuck as easily, and so sometimes can be a better option.

    Parameters
    ----------
    fn : function
        The function to minimize.

    x0 : numpy array
        The starting point (argument to fn).

    maxiters : int, optional
        Maximum iterations.

    tol : float, optional
        Tolerace for convergence (compared to absolute difference in f)

    dfdx_and_bdflag : function, optional
        Function to compute jacobian of f as well as a boundary-flag.

    xopt : numpy array, optional
        Used for debugging, output can be printed relating current optimum
        relative xopt, assumed to be a known good optimum.

    Returns
    -------
    scipy.optimize.Result object
        Includes members 'x', 'fun', 'success', and 'message'.  **Note:** returns
        the negated maximum in 'fun' in order to conform to the return value of
        other minimization routines.
    """

    MIN_STEPSIZE = 1e-8
    FINITE_DIFF_STEP = 1e-4
    RESET = 5
    stepsize = 1e-6

    #if no dfdx specifed, use finite differences
    if dfdx_and_bdflag is None:
        dfdx_and_bdflag = lambda x: _finite_diff_dfdx_and_bdflag(f,x,FINITE_DIFF_STEP)

    step = 0
    x = x0; last_fx = f(x0); last_x = x0
    lastchange = 0
    lastgradnorm = 0.0 # Safer than relying on uninitialized variables
    lastgrad     = 0.0
    if last_fx is None: raise ValueError("fmax_cg was started out of bounds!")
    while( step < maxiters and ((stepsize>MIN_STEPSIZE) or (step%RESET !=1 and RESET>1))):

        grad,boundaryFlag = dfdx_and_bdflag(x)
        gradnorm = _np.dot(grad,grad)
        if step%RESET == 0: #reset change == gradient
            change = grad[:]
        else: # add gradient to change (conjugate gradient)
            #beta = gradnorm / lastgradnorm # Fletcher-Reeves
            beta = (gradnorm - _np.dot(grad,lastgrad))/lastgradnorm # Polak-Ribiere
            #beta = (gradnorm - _np.dot(grad,lastgrad))/(_np.dot(lastchange,grad-lastgrad)) #Hestenes-Stiefel
            change = grad + beta * lastchange

        for i in range(len(change)):
            if boundaryFlag[i]*change[i] > 0:
                change[i] = 0
                print("DEBUG: fmax Preventing motion along dim %s" % i)

        if max(abs(change)) == 0:
            print("Warning: Completely Boxed in!")
            fx = last_fx; x = last_x
            assert( abs(last_fx - f(last_x)) < 1e-6)
            break
            #i = list(abs(grad)).index(min(abs(grad)))
            #change[i] = -boundaryFlag[i] * 1.0 # could pick a random direction to move in?
            #gradnorm = 1.0  # punt...

        lastgrad = grad
        lastgradnorm = gradnorm

        multiplier = 1.0/max(abs(change))
        change *= multiplier

        # Now "change" has largest element 1.  Time to do a linear search to find optimal stepsize.
        # If the last step had crazy short length, reset stepsize
        if stepsize < MIN_STEPSIZE: stepsize = MIN_STEPSIZE;
        g = lambda s: f(x+s*change) # f along a line given by changedir.  Argument to function is stepsize.
        stepsize = _maximize1D(g,0,abs(stepsize),last_fx)  #find optimal stepsize along change direction

        predicted_difference = stepsize * _np.dot(grad,change)
        if xopt is not None: xopt_dot = _np.dot(change, xopt-x) / (_np.linalg.norm(change) * _np.linalg.norm(xopt-x))
        x += stepsize * change; fx = f(x)
        difference = fx - last_fx
        print("DEBUG: Max iter ", step, ": f=",fx,", dexpct=",predicted_difference-difference, \
            ", step=",stepsize,", xopt_dot=", xopt_dot if xopt is not None else "--", \
            ", chg_dot=",_np.dot(change,lastchange)/(_np.linalg.norm(change)*_np.linalg.norm(lastchange)+1e-6))

        if abs(difference) < tol: break #Convergence condition

        lastchange = change
        last_fx = fx
        last_x = x.copy()
        step += 1

    print("Finished Custom Contrained Newton CG Method")
    print(" iterations = %d" % step)
    print(" maximum f = %g" % fx)

    solution = _optResult()
    solution.x = x; solution.fun = -fx if fx is not None else None  # negate maximum to conform to other minimization routines
    if step < maxiters:
        solution.success = True
    else:
        solution.success = False
        solution.message = "Maximum iterations exceeded"
    return solution


# Minimize g(s), given (s1,g1=g(s1)) as a starting point and guess, s2 for minimum
def _maximize1D(g,s1,s2,g1):

    PHI = (1.0+_np.sqrt(5.0))/2 #golden ratio
    TOL = 1e-10; FRAC_TOL = 1e-6

    # Note (s1,g1) and s2 are given.  Start with bracket (s1,s2,s3)
    #s3 = s2*(1.0+PHI); g2 = g(s2); g3 = g(s3)
    s3 = s2 + PHI*(s2-s1); g2 = g(s2); g3 = g(s3)
    # s4,g4 = None,None
    s1_on_bd = s3_on_bd = False

    #print "DEBUG: BEGIN MAX 1D: s1,s3=", (s1,s3)

    assert(g1 is not None or g3 is not None)
    if g1 is None or g3 is None:
        if g1 is None: s1,g1 = _findBoundary(g,s3,s1); s1_on_bd = True
        if g3 is None: s3,g3 = _findBoundary(g,s1,s3); s3_on_bd = True
        s2 = s1+(s3-s1)/PHI; g2 = g(s2)


    while( (abs(s3-s1) > TOL) and (abs(s3-s1) > FRAC_TOL*(abs(s3)+abs(s1))) ):
        #print "DEBUG: Max1D iter: (",s1,",",g1,") (",s2,",",g2,") (",s3,",",g3,")"
        if g3>g2:
            if g2>=g1: # Expand to the right
                if s3_on_bd: print("** Returning on bd"); return s3 #can't expand any further to right
                s2,g2 = s3,g3
                s3 = s1 + (s3-s1)*PHI; g3 = g(s3)
                if g3 is None:
                    s3,g3 = _findBoundary(g,s2,s3)
                    s3_on_bd = True
            else:  # contract to the left.
                s3,g3 = s2,g2
                s2 = s1 + (s3-s1)/PHI; g2 = g(s2)
        else:
            if g2<=g1: # Expand to the left
                if s1_on_bd: print("** Returning on bd2"); return s1 #can't expand any further to left
                s2,g2 = s1,g1
                s1 = s3 - (s3-s1)*PHI; g1 = g(s1)
                if g1 is None:
                    s1,g1 = _findBoundary(g,s2,s1)
                    s1_on_bd = True

            else:  # Got it bracketed: now just narrow down bracket
                return _max_within_bracket(g,s1,g1,s2,g2,s3,g3)

    print("Warning: maximize_1d could not find bracket")

    ret = s2 if g2 is not None else s1 #return s2 if it evaluates to a valid point
    assert( g(ret) is not None )       #  otherwise return s1, since it should always be valid
    return ret


#TODO: g3 and g1 are unused!
def _max_within_bracket(g,s1,g1,s2,g2,s3,g3):
    TOL = 1e-10; FRAC_TOL = 1e-6
    assert( s2-s1 > TOL )  # a legit bracket must have s1 < s2
    while( (abs(s3-s1) > TOL) and (abs(s3-s1) > FRAC_TOL*(abs(s3)+abs(s1))) ):
        s4=s1+(s3-s2); g4 = g(s4)
        #print "DEBUG: Max in brk iter: (",s1,",",g1,") (",s2,",",g2,") (",s3,",",g3,") (",s4,",",g4,")"
        assert(g4 is not None) #assume function is defined at all points in bracket
        if s4 > s2:
            if( g4 > g2 ): # Drop x1 (move to x2), move x2 to x4.
                s1,g1 = s2,g2
                s2,g2 = s4,g4
            else: # Drop x3 (move to x4)
                s3,g3 = s4,g4
        else:
            if g4 > g2: # Drop x3 (move to x2), move x2 to x4
                s3,g3 = s2,g2
                s2,g2 = s4,g4
            else: # Drop x1 (move to x4)
                s1,g1 = s4,g4
    return s2


#find boundary of g (i.e. at the edge of where it is defined)
# g(s1) must be defined (not None) and g(s2) must == None (function undefined)
def _findBoundary(g,s1,s2):
    #print "DEBUG: finding bd fn"
    TOL = 1e-6
    while( abs(s1-s2) > TOL ): #just do binary search
        m = (s1+s2)/2.0; gm = g(m)
        if gm is None: s2 = m
        else: s1 = m
    return s1,g(s1)




#provide finite difference derivatives with boundary for a given function f.  Boundaries are
# determined by the function f returning a None value when it is not defined.
def _finite_diff_dfdx_and_bdflag(f,x,DELTA):
    x = x.copy() #make sure x is unaltered
    N = len(x)
    dfdx = _np.zeros(N) #complex?
    bd   = _np.zeros(N)
    for k in range(N):
        x[k] += DELTA;   fPlus  = f(x)
        x[k] -= 2*DELTA; fMinus = f(x)
        x[k] += DELTA
        if fPlus  is None: bd[k] = +1.0
        elif fMinus is None: bd[k] = -1.0
        else: dfdx[k] = (fPlus - fMinus)/(2*DELTA)
        #assert(fPlus is not None or fMinus is not None) #make sure we don't evaluate f somewhere it's completely undefined

    return dfdx,bd

#def f6(param):
#    '''Schaffer's F6 function'''
#    para = param*10
#    para = param[0:2]
#    num = (sin(sqrt((para[0] * para[0]) + (para[1] * para[1])))) * \
#        (sin(sqrt((para[0] * para[0]) + (para[1] * para[1])))) - 0.5
#    denom = (1.0 + 0.001 * ((para[0] * para[0]) + (para[1] * para[1]))) * \
#            (1.0 + 0.001 * ((para[0] * para[0]) + (para[1] * para[1])))
#    f6 =  0.5 - (num/denom)
#    errorf6 = 1 - f6
#    return f6, errorf6;
