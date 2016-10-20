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
MACH_PRECISION = 1e-12


def custom_leastsq(obj_fn, jac_fn, x0, f_norm2_tol=1e-6, jac_norm_tol=1e-6,
                   rel_ftol=1e-6, rel_xtol=1e-6, max_iter=100, comm=None,
                   verbosity=0, profiler=None):
    msg = ""
    converged = False
    x = x0
    f = obj_fn(x)
    norm_f = _np.dot(f,f) # _np.linalg.norm(f)**2
    half_max_nu = 2**62 #what should this be??
    tau = 1e-3
    nu = 2
    mu = 0 #initialized on 1st iter
    my_cols_slice = None

    if comm is not None and comm.Get_rank() != 0:
        verbosity = 0 #Only print to stdout from root process

    if not _np.isfinite(norm_f):
        msg = "Infinite norm of objective function at initial point!"


    for k in range(max_iter): #outer loop
        # assume x, f, fnorm hold valid values

        if len(msg) > 0: 
            break #exit outer loop if an exit-message has been set

        if norm_f < f_norm2_tol:
            msg = "Sum of squares is at most %g" % f_norm2_tol
            converged = True; break

        if verbosity > 0:
            print("--- Outer Iter %d: norm_f = %g, mu=%g" % (k,norm_f,mu))
            
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
        norm_JTf = _np.linalg.norm(JTf,ord=_np.inf)
        norm_x = _np.dot(x,x) # _np.linalg.norm(x)**2
        undampled_JTJ_diag = JTJ.diagonal().copy()

        if norm_JTf < jac_norm_tol:
            msg = "norm(jacobian) is at most %g" % jac_norm_tol
            converged = True; break

        if k == 0:
            #mu = tau # initial damping element
            mu = tau * _np.max(undampled_JTJ_diag) # initial damping element

        #determing increment using adaptive damping
        while True:  #inner loop

            if profiler: profiler.mem_check("custom_leastsq: begin inner iter")
            JTJ[idiag] += mu # augment normal equations
            #JTJ[idiag] *= (1.0 + mu) # augment normal equations

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
                norm_dx = _np.dot(dx,dx) # _np.linalg.norm(dx)**2

                if verbosity > 1:
                    print("  - Inner Loop: mu=%g, norm_dx=%g" % (mu,norm_dx))

                if norm_dx < (rel_xtol**2)*norm_x:
                    msg = "Relative change in |x| is at most %g" % rel_xtol
                    converged = True; break

                if norm_dx > (norm_x+rel_xtol)/(MACH_PRECISION**2):
                    msg = "(near-)singular linear system"; break
                
                new_f = obj_fn(new_x)
                if profiler: profiler.mem_check("custom_leastsq: after obj_fn")
                norm_new_f = _np.dot(new_f,new_f) # _np.linalg.norm(new_f)**2
                if not _np.isfinite(norm_new_f): # avoid infinite loop...
                    msg = "Infinite norm of objective function!"; break

                dL = _np.dot(dx, mu*dx - JTf) # expected decrease in ||F||^2 from linear model
                dF = norm_f - norm_new_f      # actual decrease in ||F||^2

                if verbosity > 1:
                    print("      (cont): norm_new_f=%g, dL=%g, dF=%g, reldL=%g, reldF=%g" % 
                          (norm_new_f,dL,dF,dL/norm_f,dF/norm_f))

                if dL/norm_f < rel_ftol and dF/norm_f < rel_ftol and dF/dL < 2.0:
                    msg = "Both actual and predicted relative reductions in the" + \
                        " sum of squares are at most %g" % rel_ftol
                    converged = True; break

                if profiler: profiler.mem_check("custom_leastsq: before success")

                if dL > 0 and dF > 0:
                    # reduction in error: increment accepted!
                    t = 1.0 - (2*dF/dL-1.0)**3 # dF/dL == gain ratio
                    mu *= max(t,1.0/3.0)
                    nu = 2
                    x,f, norm_f = new_x, new_f, norm_new_f

                    if verbosity > 1:
                        print("      Accepted! gain ratio=%g  mu * %g => %g"
                              % (dF/dL,max(t,1.0/3.0),mu))

                    ##Check to see if we *would* switch to Q-N method in a hybrid algorithm
                    #new_Jac = jac_fn(new_x)
                    #new_JTf = _np.dot(new_Jac.T,new_f)
                    #print(" CHECK: %g < %g ?" % (_np.linalg.norm(new_JTf,
                    #    ord=_np.inf),0.02 * _np.linalg.norm(new_f)))

                    break # exit inner loop normally
            #else:
            #    print("LinSolve Failure!!")

            # if this point is reached, either the linear solve failed
            # or the error did not reduce.  In either case, reject increment.
                
            #Increase damping (mu), then increase damping factor to 
            # accelerate further damping increases.
            mu *= nu
            if nu > half_max_nu : #watch for nu getting too large (&overflow)
                msg = "Stopping after nu overflow!"; break
            nu = 2*nu
            if verbosity > 1:
                print("      Rejected!  mu => mu*nu = %g, nu => 2*nu = %g"
                      % (mu, nu))
            
            JTJ[idiag] = undampled_JTJ_diag #restore diagonal
        #end of inner loop
    #end of outer loop
    else:
        #if no break stmt hit, then we've exceeded maxIter
        msg = "Maximum iterations (%d) exceeded" % max_iter

    #JTJ[idiag] = undampled_JTJ_diag #restore diagonal
    return x, converged, msg
    #solution = _optResult()
    #solution.x = x; solution.fun = f
    #solution.success = converged
    #solution.message = msg
    #return solution



#Wikipedia-version of LM algorithm, testing mu and mu/nu damping params and taking 
# mu/nu => new_mu if acceptable...  This didn't seem to perform well, but maybe just
# needs some tweaking, so leaving it commented here for reference
#def custom_leastsq_wikip(obj_fn, jac_fn, x0, f_norm_tol=1e-6, jac_norm_tol=1e-6,
#                   rel_tol=1e-6, max_iter=100, comm=None, verbosity=0, profiler=None):
#    msg = ""
#    converged = False
#    x = x0
#    f = obj_fn(x)
#    norm_f = _np.linalg.norm(f)
#    tau = 1e-3 #initial mu
#    nu = 1.3
#    my_cols_slice = None
#    
#
#    if not _np.isfinite(norm_f):
#        msg = "Infinite norm of objective function at initial point!"
#
#    for k in range(max_iter): #outer loop
#        # assume x, f, fnorm hold valid values
#
#        if len(msg) > 0: 
#            break #exit outer loop if an exit-message has been set
#
#        if norm_f < f_norm_tol:
#            msg = "norm(objectivefn) is small"
#            converged = True; break
#
#        if verbosity > 0:
#            print("--- Outer Iter %d: norm_f = %g" % (k,norm_f))
#            
#        if profiler: profiler.mem_check("custom_leastsq: begin outer iter *before de-alloc*")
#        Jac = None; JTJ = None; JTf = None
#
#        if profiler: profiler.mem_check("custom_leastsq: begin outer iter")
#        Jac = jac_fn(x)
#        if profiler: profiler.mem_check("custom_leastsq: after jacobian:" 
#                                        + "shape=%s, GB=%.2f" % (str(Jac.shape),
#                                                        Jac.nbytes/(1024.0**3)) )
#
#        tm = _time.time()
#        if my_cols_slice is None:
#            my_cols_slice = _mpit.distribute_for_dot(Jac.shape[0], comm)
#        JTJ = _mpit.mpidot(Jac.T,Jac,my_cols_slice,comm)   #_np.dot(Jac.T,Jac)
#        JTf = _np.dot(Jac.T,f)
#        if profiler: profiler.add_time("custom_leastsq: dotprods",tm)
#
#        idiag = _np.diag_indices_from(JTJ)
#        norm_JTf = _np.linalg.norm(JTf) #, ord='inf')
#        norm_x = _np.linalg.norm(x)
#        undampled_JTJ_diag = JTJ.diagonal().copy()
#
#        if norm_JTf < jac_norm_tol:
#            msg = "norm(jacobian) is small"
#            converged = True; break
#
#        if k == 0:
#            mu = tau #* _np.max(undampled_JTJ_diag) # initial damping element
#        #mu = tau #* _np.max(undampled_JTJ_diag) # initial damping element
#
#        #determing increment using adaptive damping
#        while True:  #inner loop
#
#            ### Evaluate with mu' = mu / nu
#            mu = mu / nu
#            if profiler: profiler.mem_check("custom_leastsq: begin inner iter")
#            JTJ[idiag] *= (1.0 + mu) # augment normal equations
#            #JTJ[idiag] += mu # augment normal equations
#
#            try:
#                if profiler: profiler.mem_check("custom_leastsq: before linsolve")
#                tm = _time.time()
#                success = True
#                dx = _np.linalg.solve(JTJ, -JTf) 
#                if profiler: profiler.add_time("custom_leastsq: linsolve",tm)
#            except _np.linalg.LinAlgError:
#                success = False
#            
#            if profiler: profiler.mem_check("custom_leastsq: after linsolve")
#            if success: #linear solve succeeded
#                new_x = x + dx
#                norm_dx = _np.linalg.norm(dx)
#
#                #if verbosity > 1:
#                #    print("--- Inner Loop: mu=%g, norm_dx=%g" % (mu,norm_dx))
#
#                if norm_dx < rel_tol*norm_x: #use squared qtys instead (speed)?
#                    msg = "relative change in x is small"
#                    converged = True; break
#
#                if norm_dx > (norm_x+rel_tol)/MACH_PRECISION:
#                    msg = "(near-)singular linear system"; break
#                
#                new_f = obj_fn(new_x)
#                if profiler: profiler.mem_check("custom_leastsq: after obj_fn")
#                norm_new_f = _np.linalg.norm(new_f)
#                if not _np.isfinite(norm_new_f): # avoid infinite loop...
#                    msg = "Infinite norm of objective function!"; break
#
#                dF = norm_f - norm_new_f
#                if dF > 0: #accept step
#                    #print("      Accepted!")
#                    x,f, norm_f = new_x, new_f, norm_new_f
#                    nu = 1.3
#                    break # exit inner loop normally
#                else:
#                    mu *= nu #increase mu
#            else:
#                #Linear solve failed:
#                mu *= nu #increase mu
#                nu = 2*nu
#            
#            JTJ[idiag] = undampled_JTJ_diag #restore diagonal for next inner loop iter
#        #end of inner loop
#    #end of outer loop
#    else:
#        #if no break stmt hit, then we've exceeded maxIter
#        msg = "Maximum iterations (%d) exceeded" % max_iter
#
#    return x, converged, msg
