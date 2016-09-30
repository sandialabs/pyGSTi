from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Custom implementation of the Levenberg-Marquardt Algorithm """

import time as _time
import numpy as _np
#from scipy.optimize import OptimizeResult as _optResult 

from ..tools import mpitools as _mpit

#constants
MACH_PRECISION = 1e-16


def custom_leastsq(obj_fn, jac_fn, x0, f_norm_tol=1e-6, jac_norm_tol=1e-6,
                   rel_tol=1e-6, max_iter=100, comm=None, verbosity=0, profiler=None):
    msg = ""
    converged = False
    x = x0
    f = obj_fn(x)
    norm_f = _np.linalg.norm(f)
    half_max_nu = 2**62 #what should this be??
    tau = 1e-3
    nu = 2
    my_cols_slice = None
    

    if not _np.isfinite(norm_f):
        msg = "Infinite norm of objective function at initial point!"

    for k in range(max_iter): #outer loop
        # assume x, f, fnorm hold valid values

        if len(msg) > 0: 
            break #exit outer loop if an exit-message has been set

        if norm_f < f_norm_tol:
            msg = "norm(objectivefn) is small"
            converged = True; break

        if verbosity > 0:
            print("--- Outer Iter %d: norm_f = %g" % (k,norm_f))
            
        if profiler: profiler.mem_check("custom_leastsq: begin outer iter *before de-alloc*")
        Jac = None; JTJ = None; JTf = None

        if profiler: profiler.mem_check("custom_leastsq: begin outer iter")
        Jac = jac_fn(x)
        if profiler: profiler.mem_check("custom_leastsq: after jacobian:" 
                                        + "shape=%s, GB=%.2f" % (str(Jac.shape),
                                                        Jac.nbytes/(1024.0**3)) )

        tm = _time.time()
        if my_cols_slice is None:
            my_cols_slice = _mpit.distribute_for_dot(Jac.shape[0], comm)
        JTJ = _mpit.mpidot(Jac.T,Jac,my_cols_slice,comm)   #_np.dot(Jac.T,Jac)
        JTf = _np.dot(Jac.T,f)
        if profiler: profiler.add_time("custom_leastsq: dotprods",tm)

        idiag = _np.diag_indices_from(JTJ)
        norm_JTf = _np.linalg.norm(JTf) #, ord='inf')
        norm_x = _np.linalg.norm(x)
        undampled_JTJ_diag = JTJ.diagonal().copy()

        if norm_JTf < jac_norm_tol:
            msg = "norm(jacobian) is small"
            converged = True; break

        mu = tau #* _np.max(undampled_JTJ_diag) # initial damping element

        #determing increment using adaptive damping
        while True:  #inner loop
            #JTJ[idiag] += mu # augment normal equations

            if profiler: profiler.mem_check("custom_leastsq: begin inner iter")
            JTJ[idiag] *= (1.0 + mu) # augment normal equations

            try:
                if profiler: profiler.mem_check("custom_leastsq: before linsolve")
                tm = _time.time()
                success = True
                dx = _np.linalg.solve(JTJ, -JTf) 
                if profiler: profiler.add_time("custom_leastsq: linsolve",tm)
            except _np.linalg.LinAlgError:
                success = False
            
            if profiler: profiler.mem_check("custom_leastsq: after linsolve")
            if success: #linear solve succeeded
                new_x = x + dx
                norm_dx = _np.linalg.norm(dx)

                #if verbosity > 1:
                #    print("--- Inner Loop: mu=%g, norm_dx=%g" % (mu,norm_dx))

                if norm_dx < rel_tol*norm_x: #use squared qtys instead (speed)?
                    msg = "relative change in x is small"
                    converged = True; break

                if norm_dx > (norm_x+rel_tol)/MACH_PRECISION:
                    msg = "(near-)singular linear system"; break
                
                new_f = obj_fn(new_x)
                if profiler: profiler.mem_check("custom_leastsq: after obj_fn")
                norm_new_f = _np.linalg.norm(new_f)
                if not _np.isfinite(norm_new_f): # avoid infinite loop...
                    msg = "Infinite norm of objective function!"; break

                dL = _np.dot(dx, mu*dx - JTf) # (JTJ + muI)*dx = -JTf ??
                dF = norm_f - norm_new_f

                #print("      (cont): norm_new_f=%g, dL=%g, dF=%g" % 
                #      (norm_new_f,dL,dF))
                if profiler: profiler.mem_check("custom_leastsq: before success")

                if dL > 0 and dF > 0:
                    # reduction in error: increment accepted!
                    #print("      Accepted!")
                    t = 1.0 - (2*dF/dL-1.0)**3
                    mu *= max(t,1.0/3.0)
                    nu = 2
                    x,f, norm_f = new_x, new_f, norm_new_f
                    break # exit inner loop normally

            
            # if this point is reached, either the linear solve failed
            # or the error did not reduce.  In either case, reject increment.
                
            #Increase damping (mu), then increase damping factor to 
            # accelerate further damping increases.
            mu *= nu
            if nu > half_max_nu : #watch for nu getting too large (&overflow)
                msg = "Stopping after nu overflow!"; break
            nu = 2*nu
            #print("      Rejected!  mu => mu*nu = %g, nu => 2*nu = %g"
            #      % (mu, nu))
            
            JTJ[idiag] = undampled_JTJ_diag #restore diagonal
        #end of inner loop
    #end of outer loop
    else:
        #if no break stmt hit, then we've exceeded maxIter
        msg = "Maximum iterations (%d) exceeded" % max_iter

    #JTJ[idiag] = undampled_JTJ_diag #restore diagonal
    print("DONE: %s" % msg)

    return x, converged, msg
    #solution = _optResult()
    #solution.x = x; solution.fun = f
    #solution.success = converged
    #solution.message = msg
    #return solution
