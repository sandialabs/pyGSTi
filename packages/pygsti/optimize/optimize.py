from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************
""" Optimization (minimization) functions """

import numpy as _np
import time as _time
import sys as _sys
import os as _os
import scipy.optimize as _spo

try:
    from scipy.optimize import Result as _optResult #for earlier scipy versions
except:
    from scipy.optimize import OptimizeResult as _optResult #for later scipy versions

from .customcg import fmax_cg


startTime = _time.time() #for reference point of obj func printer


def minimize(fn,x0, method='cg', callback=None,
             tol=1e-10, maxiter=1000000, maxfev=None,
             stopval=None, jac=None):
    """
    Minimizes the function fn starting at x0.

    This is a gateway function to all other minimization routines within this
    module, providing a common interface to many different minimization methods
    (including and extending beyond those available from scipy.optimize).

    Parameters
    ----------
    fn : function
        The function to minimize.

    x0 : numpy array
        The starting point (argument to fn).

    method : string, optional
        Which minimization method to use.  Allowed values are:
        "simplex" : uses fmin_simplex
        "supersimplex" : uses fmin_supersimplex
        "customcg" : uses fmax_cg (custom CG method)
        "brute" : uses scipy.optimize.brute
        "basinhopping" : uses scipy.optimize.basinhopping with L-BFGS-B
        "swarm" : uses fmin_particle_swarm
        "evolve" : uses fmin_evolutionary (which uses DEAP)
        < methods available from scipy.optimize.minimize >

    callback : function, optional
        A callback function to be called in order to track optimizer progress.
        Should have signature: myCallback(x, f=None, accepted=None).  Note that
        create_obj_func_printer(...) function can be used to create a callback.

    tol : float, optional
        Tolerance value used for all types of tolerances available in a given method.

    maxiter : int, optional
        Maximum iterations.

    maxfev : int, optional
        Maximum function evaluations; used only when available, and defaults to maxiter.

    stopval : float, optional
        For basinhopping method only.  When f <= stopval then basinhopping outer loop
        will terminate.  Useful when a bound on the minimum is known.

    jac : function
        Jacobian function.

    Returns
    -------
    scipy.optimize.Result object
        Includes members 'x', 'fun', 'success', and 'message'.
    """

    if maxfev is None: maxfev = maxiter

    #Run Minimization Algorithm
    if method == 'simplex':
        solution = fmin_simplex(fn, x0, slide=1.0, tol=tol, maxiter=maxiter)

    elif method == 'supersimplex':
        solution = fmin_supersimplex(fn, x0, outer_tol=1.0, inner_tol=tol,
                                     max_outer_iter=100, min_inner_maxiter=100, max_inner_maxiter=maxiter)

    elif method == 'customcg':
        def fn_to_max(x):
            f = fn(x); return -f if f is not None else None

        if jac is not None:
            def dfdx_and_bdflag(x):
                j = -jac(x)
                bd = _np.zeros(len(j)) #never say fn is on boundary, since this is an analytic derivative
                return j, bd
        else:
            dfdx_and_bdflag = None

        solution = fmax_cg(fn_to_max, x0, maxiter, tol, dfdx_and_bdflag, None ) #Note: even though we maximize, return value is negated to conform to min routines

    elif method == 'brute':
        ranges = [ (0.0,1.0) ] * len(x0); Ns = 4  #params for 'brute' algorithm
        xmin, _ = _spo.brute(fn, ranges, (), Ns) #jac=jac
        #print "DEBUG: Brute fmin = ",fmin
        solution = _spo.minimize(fn,xmin,method="Nelder-Mead", options={}, tol=tol, callback = callback, jac=jac)

    elif method == 'basinhopping':
        def basin_callback(x, f, accept):
            if callback is not None: callback(x,f=f,accepted=accept)
            if stopval is not None and f <= stopval:
                return True #signals basinhopping to stop
            return False
        solution = _spo.basinhopping(fn, x0, niter=maxiter, T=2.0, stepsize=1.0,
                                               callback=basin_callback, minimizer_kwargs={'method': "L-BFGS-B", 'jac': jac})

        #DEBUG -- follow with Nelder Mead to make sure basinhopping found a minimum. (It seems to)
        #print "DEBUG: running Nelder-Mead:"
        #opts = { 'maxfev': maxiter, 'maxiter': maxiter }
        #solution = _spo.minimize(fn, solution.x, options=opts, method="Nelder-Mead", tol=1e-8, callback=callback)
        #print "DEBUG: done: best f = ",solution.fun

        solution.success = True #basinhopping doesn't seem to set this...

    elif method == 'swarm':
        solution = fmin_particle_swarm(fn, x0, tol, maxiter, popsize=1000) #, callback = callback)

    elif method == 'evolve':
        solution = fmin_evolutionary(fn, x0, num_generations = maxiter, num_individuals = 500)

#    elif method == 'homebrew':
#      solution = fmin_homebrew(fn, x0, maxiter)

    else:
        #Set options for different algorithms
        opts = {'maxiter': maxiter, 'disp': False }
        if method == "BFGS": opts['gtol'] = tol  #gradient norm tolerance
        elif method == "L-BFGS-B": opts['gtol'] = opts['ftol'] = tol  #gradient norm and fractional y-tolerance
        elif method == "Nelder-Mead": opts['maxfev'] = maxfev  #max fn evals (note: ftol and xtol can also be set)

        if method in ("BFGS","CG","Newton-CG","L-BFGS-B","TNC","SLSQP","dogleg","trust-ncg"): #use jacobian
            solution = _spo.minimize(fn, x0, options=opts, method=method, tol=tol, callback=callback, jac=jac)
        else:
            solution = _spo.minimize(fn, x0, options=opts, method=method, tol=tol, callback=callback)

    return solution



def fmin_supersimplex(fn, x0, outer_tol, inner_tol, max_outer_iter, min_inner_maxiter, max_inner_maxiter):
    """
    Minimize a function using repeated applications of the simplex algorithm.

    By varying the maximum number of iterations and repeatedly calling scipy's
    Nelder-Mead simplex optimization, this function performs as a robust (but
    slow) minimization.

    Parameters
    ----------
    fn : function
        The function to minimize.

    x0 : numpy array
        The starting point (argument to fn).

    outer_tol : float
        Tolerance of outer loop

    inner_tol : float
        Tolerance fo inner loop

    max_outer_iter : int
        Maximum number of outer-loop iterations

    min_inner_maxiter : int
        Minimum number of inner-loop iterations

    max_inner_maxiter : int
        Maxium number of outer-loop iterations

    Returns
    -------
    scipy.optimize.Result object
        Includes members 'x', 'fun', 'success', and 'message'.
    """
    f_init = fn(x0)
    f_final = f_init - 10*outer_tol #prime the loop
    x_start = x0

    i = 1
    cnt_at_same_maxiter = 1
    inner_maxiter = min_inner_maxiter

    while ( f_init-f_final > outer_tol or inner_maxiter < max_inner_maxiter) and i < max_outer_iter:
        if f_init-f_final <= outer_tol and inner_maxiter < max_inner_maxiter:
            inner_maxiter *= 10; cnt_at_same_maxiter = 1
        if cnt_at_same_maxiter > 10 and inner_maxiter > min_inner_maxiter:
            inner_maxiter /= 10; cnt_at_same_maxiter = 1
        f_init = f_final

        print(">>> fmin_supersimplex: outer iteration %d (inner_maxiter = %d)" % (i,inner_maxiter))
        i += 1; cnt_at_same_maxiter += 1

        opts = {'maxiter': inner_maxiter, 'maxfev': inner_maxiter, 'disp': False }
        inner_solution = _spo.minimize(fn,x_start,options=opts, method='Nelder-Mead',callback=None, tol=inner_tol)

        if not inner_solution.success:
            print("WARNING: fmin_supersimplex inner loop failed (tol=%g, maxiter=%d): %s" \
                % (inner_tol,inner_maxiter,inner_solution.message))

        f_final = inner_solution.fun
        x_start = inner_solution.x
        print(">>> fmin_supersimplex: outer iteration %d gives min = %f" % (i,f_final))

    solution = _optResult()
    solution.x = inner_solution.x
    solution.fun = inner_solution.fun
    if i < max_outer_iter:
        solution.success = True
    else:
        solution.success = False
        solution.message = "Maximum iterations exceeded"
    return solution



def fmin_simplex(fn, x0, slide=1.0, tol=1e-8, maxiter=1000):
    """
    Minimizes a function using a custom simplex implmentation.

    This was used primarily to check scipy's Nelder-Mead method
    and runs much slower, so there's not much reason for using
    this method.

    Parameters
    ----------
    fn : function
        The function to minimize.

    x0 : numpy array
        The starting point (argument to fn).

    slide : float, optional
        Affects initial simplex point locations

    tol : float, optional
        Relative tolerance as a convergence criterion.

    maxiter : int, optional
        Maximum iterations.

    Returns
    -------
    scipy.optimize.Result object
        Includes members 'x', 'fun', 'success', and 'message'.
    """

    # Setup intial values
    n = len(x0)
    f = _np.zeros(n+1)
    x = _np.zeros((n+1,n))

    x[0] = x0

    # Setup intial X range
    for i in range(1,n+1):
        x[i] = x0
        x[i,i-1] = x0[i-1] + slide

    # Setup intial functions based on x's just defined
    for i in range(n+1):
        f[i] = fn(x[i])

    # Main Loop operation, loops infinitly until break condition
    counter = 0
    while True:
        low = _np.argmin(f)
        high = _np.argmax(f)
        counter += 1

        # Compute Migration
        d = (-(n+1)*x[high]+sum(x))/n

        # Break if value is close
        if _np.sqrt(_np.dot(d,d)/n)<tol or counter == maxiter:
            solution = _optResult()
            solution.x = x[low]; solution.fun = f[low]
            if counter < maxiter:
                solution.success = True
            else:
                solution.success = False
                solution.message = "Maximum iterations exceeded"
            return solution

        newX = x[high] + 2.0*d
        newF = fn(newX)

        if newF <= f[low]:
            # Bad news, new value is lower than any other point => replace high point with new values
            x[high] = newX
            f[high] = newF
            newX = x[high] + d
            newF = fn(newX)

            # Check if need to expand
            if newF <= f[low]:
                x[high] = newX
                f[high] = newF

        else:
            # Good news, new value is higher than lowest point

            # Check if need to contract
            if newF <= f[high]:
                x[high] = newX
                f[high] = newF
            else:
                # Contraction
                newX = x[high] + 0.5*d
                newF = fn(newX)
                if newF <= f[high]:
                    x[high] = newX
                    f[high] = newF
                else:
                    for i in range(len(x)):
                        if i!=low:
                            x[i] = (x[i]-x[low])
                            f[i] = fn(x[i])



#TODO err_crit is never used?
def fmin_particle_swarm(f, x0, err_crit, iter_max, popsize=100, c1=2, c2=2):
    """
    A simple implementation of the Particle Swarm Optimization Algorithm.
    Pradeep Gowda 2009-03-16

    Parameters
    ----------
    f : function
        The function to minimize.

    x0 : numpy array
        The starting point (argument to fn).

    err_crit : float
        Critical error (i.e. tolerance).  Stops when error < err_crit.

    iter_max : int
        Maximum iterations.

    popsize : int, optional
        Population size.  Larger populations are better at finding the global
        optimum but make the algorithm take longer to run.

    c1 : float, optional
        Coefficient describing a particle's affinity for it's (local) maximum.

    c2 : float, optional
        Coefficient describing a particle's affinity for the best maximum any
        particle has seen (the current global max).

    Returns
    -------
    scipy.optimize.Result object
        Includes members 'x', 'fun', 'success', and 'message'.
    """
    dimensions = len(x0)
    LARGE = 1e10

    class Particle:
        pass

    #initialize the particles
    particles = []
    for i in range(popsize):
        p = Particle()
        p.params = x0 + 2 * (_np.random.random(dimensions) - 0.5)
        p.best = p.params[:]
        p.fitness = LARGE # large == bad fitness
        p.v = _np.zeros(dimensions)
        particles.append(p)

    # let the first particle be the global best
    gbest = particles[0]; ibest = 0
    # bDoLocalFitnessOpt = False

    #DEBUG
    #if False:
    #    import pickle as _pickle
    #    bestGaugeMx = _pickle.load(open("bestGaugeMx.debug"))
    #    lbfgsbGaugeMx = _pickle.load(open("lbfgsbGaugeMx.debug"))
    #    cgGaugeMx = _pickle.load(open("cgGaugeMx.debug"))
    #    initialGaugeMx = x0.reshape( (4,4) )
    #
    #    #DEBUG: dump line cut to plot
    #    nPts = 100
    #    print "DEBUG: best offsets = \n", bestGaugeMx - initialGaugeMx
    #    print "DEBUG: lbfgs offsets = \n", lbfgsbGaugeMx - initialGaugeMx
    #    print "DEBUG: cg offsets = \n", cgGaugeMx - initialGaugeMx
    #
    #    print "# DEBUG plot"
    #    #fDebug = open("x0ToBest.dat","w")
    #    #fDebug = open("x0ToLBFGS.dat","w")
    #    fDebug = open("x0ToCG.dat","w")
    #    #fDebug = open("LBFGSToBest.dat","w")
    #    #fDebug = open("CGToBest.dat","w")
    #    #fDebug = open("CGToLBFGS.dat","w")
    #
    #    for i in range(nPts+1):
    #        alpha = float(i) / nPts
    #        #matM = (1.0-alpha) * initialGaugeMx + alpha*bestGaugeMx
    #        #matM = (1.0-alpha) * initialGaugeMx + alpha*lbfgsbGaugeMx
    #        matM = (1.0-alpha) * initialGaugeMx + alpha*cgGaugeMx
    #        #matM = (1.0-alpha) * lbfgsbGaugeMx + alpha*bestGaugeMx
    #        #matM = (1.0-alpha) * cgGaugeMx + alpha*bestGaugeMx
    #        #matM = (1.0-alpha) * cgGaugeMx + alpha*lbfgsbGaugeMx
    #        print >> fDebug, "%g %g" % (alpha, f(matM.flatten()))
    #    exit()
    #
    #
    #    fDebug = open("lineDataFromX0.dat","w")
    #    min_offset = -1; max_offset = 1
    #    for i in range(nPts+1):
    #        offset = min_offset + float(i)/nPts * (max_offset-min_offset)
    #        print >> fDebug, "%g" % offset,
    #
    #        for k in range(len(x0)):
    #            x = x0.copy(); x[k] += offset
    #            try:
    #                print >> fDebug, " %g" % f(x),
    #            except:
    #                print >> fDebug, " nan",
    #        print >> fDebug, ""
    #
    #    print >> fDebug, "#END DEBUG plot"
    #    exit()
    #END DEBUG


    #err = 1e10
    for iter_num in range(iter_max):
        w = 1.0 #- i/iter_max

        #bDoLocalFitnessOpt = bool(iter_num > 20 and abs(lastBest-gbest.fitness) < 0.001 and iter_num % 10 == 0)
        # lastBest = gbest.fitness
        # minDistToBest = 1e10; minV = 1e10; maxV = 0 #DEBUG

        for (ip,p) in enumerate(particles):
            fitness = f(p.params)

            #if bDoLocalFitnessOpt:
            #    opts = {'maxiter': 100, 'maxfev': 100, 'disp': False }
            #    local_soln = _spo.minimize(f,p.params,options=opts, method='L-BFGS-B',callback=None, tol=1e-2)
            #    p.params = local_soln.x
            #    fitness = local_soln.fun

            if fitness < p.fitness:  #low 'fitness' is good b/c we're minimizing
                p.fitness = fitness
                p.best = p.params

            if fitness < gbest.fitness:
                gbest = p; ibest = ip

            v = w*p.v + c1 * _np.random.random() * (p.best - p.params) \
                    + c2 * _np.random.random() * (gbest.params - p.params)
            p.params = p.params + v
            for (i,pv) in enumerate(p.params):
                p.params[i] = ((pv+1) % 2) - 1 #periodic b/c on box between -1 and 1

            #from .. import tools as tools_
            #matM = p.params.reshape( (4,4) )  #DEBUG
            #minDistToBest = min(minDistToBest, _tools.frobeniusdist(
            #                                    bestGaugeMx,matM)) #DEBUG
            #minV = min( _np.linalg.norm(v), minV)
            #maxV = max( _np.linalg.norm(v), maxV)

        #print "DB: min diff from best = ", minDistToBest #DEBUG
        #print "DB: min,max v = ", (minV,maxV)


        #if False: #bDoLocalFitnessOpt:
        #    opts = {'maxiter': 100, 'maxfev': 100, 'disp': False }
        #    print "initial fun = ",gbest.fitness,
        #    local_soln = _spo.minimize(f,gbest.params,options=opts, method='L-BFGS-B',callback=None, tol=1e-5)
        #    gbest.params = local_soln.x
        #    gbest.fitness = local_soln.fun
        #    print "  final fun = ",gbest.fitness

        print("Iter %d: global best = %g (index %d)" % (iter_num, gbest.fitness, ibest))

        #if err < err_crit:  break  #TODO: stopping condition

    ## Uncomment to print particles
    #for p in particles:
    #    print 'params: %s, fitness: %s, best: %s' % (p.params, p.fitness, p.best)

    solution = _optResult()
    solution.x = gbest.params; solution.fun = gbest.fitness
    solution.success = True
#    if iter_num < maxiter:
#        solution.success = True
#    else:
#        solution.success = False
#        solution.message = "Maximum iterations exceeded"
    return solution



def fmin_evolutionary(f, x0, num_generations, num_individuals):
    """
    Minimize a function using an evolutionary algorithm.

    Uses python's deap package to perform an evolutionary
    algorithm to find a function's global minimum.

    Parameters
    ----------
    fn : function
        The function to minimize.

    x0 : numpy array
        The starting point (argument to fn).

    num_generations : int
        The number of generations to carry out. (similar to the number
        of iterations)

    num_individuals : int
        The number of individuals in each generation.  More individuals
        make finding the global optimum more likely, but take longer
        to run.


    Returns
    -------
    scipy.optimize.Result object
        Includes members 'x', 'fun', 'success', and 'message'.
    """

    import deap as _deap
    import deap.creator as _creator
    import deap.base as _base
    import deap.tools as _tools
    numParams = len(x0)

    # Create the individual class
    _creator.create("FitnessMin", _base.Fitness, weights=(-1.0,))
    _creator.create("Individual", list, fitness=_creator.FitnessMin)

    # Create initialization functions
    toolbox = _base.Toolbox()
    toolbox.register("random", _np.random.random)
    toolbox.register("individual", _tools.initRepeat, _creator.Individual,
                     toolbox.random, n=numParams) # fn to init an individual from a list of numParams random numbers
    toolbox.register("population", _tools.initRepeat, list, toolbox.individual) # fn to create a population (still need to specify n)

    # Create operation functions
    def evaluate(individual):
        return f( _np.array(individual) ),  #note: must return a tuple

    toolbox.register("mate", _tools.cxTwoPoint)
    toolbox.register("mutate", _tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    toolbox.register("select", _tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Create the population
    pop = toolbox.population(n=num_individuals)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    PROB_TO_CROSS = 0.5
    PROB_TO_MUTATE = 0.2

    # Initialize statistics
    stats = _tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", _np.mean)
    stats.register("std", _np.std)
    stats.register("min", _np.min)
    stats.register("max", _np.max)
    logbook = _tools.Logbook()

    #Run algorithm
    for g in range(num_generations):
        record = stats.compile(pop)
        logbook.record(gen=g, **record)
        print("Gen %d: %s" % (g,record))

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if _np.random.random() < PROB_TO_CROSS:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation on the offspring
        for mutant in offspring:
            if _np.random.random() < PROB_TO_MUTATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    #get best individual and return params
    indx_min_fitness = _np.argmin( [ ind.fitness.values[0] for ind in pop ] )
    best_params = _np.array(pop[indx_min_fitness])

    solution = _optResult()
    solution.x = best_params; solution.fun = pop[indx_min_fitness].fitness.values[0]
    solution.success = True
    return solution




#def fmin_homebrew(f, x0, maxiter):
#    """
#    Cooked up by Erik, this algorithm is similar to basinhopping but with some tweaks.
#
#    Parameters
#    ----------
#    fn : function
#        The function to minimize.
#
#    x0 : numpy array
#        The starting point (argument to fn).
#
#    maxiter : int
#        The maximum number of iterations.
#
#    Returns
#    -------
#    scipy.optimize.Result object
#        Includes members 'x', 'fun', 'success', and 'message'.
#    """
#
#    STEP = 0.01
#    MAX_STEPS = int(2.0 / STEP) # allow a change of at most 2.0
#    MAX_DIR_TRIES = 1000
#    T = 1.0
#
#    global_best_params = cur_x0 = x0
#    global_best = cur_f = f(x0)
#    N = len(x0)
#    trial_x0 = x0.copy()
#
#    for it in range(maxiter):
#
#        #Minimize using L-BFGS-B
#        opts = {'maxiter': maxiter, 'maxfev': maxiter, 'disp': False }
#        soln = _spo.minimize(f,trial_x0,options=opts, method='L-BFGS-B',callback=None, tol=1e-8)
#
#        # Update global best
#        if soln.fun < global_best:
#            global_best_params = soln.x
#            global_best = soln.fun
#
#        #check if we accept the new minimum
#        if soln.fun < cur_f or _np.random.random() < _np.exp( -(soln.fun - cur_f)/T ):
#            cur_x0 = soln.x; cur_f = soln.fun
#            print "Iter %d: f=%g accepted -- global best = %g" % (it, cur_f, global_best)
#        else:
#            print "Iter %d: f=%g declined" % (it, cur_f)
#
#        trial_x0 = None; numTries = 0
#        while trial_x0 is None and numTries < MAX_DIR_TRIES:
#            #choose a random direction
#            direction = _np.random.random( N )
#            numTries += 1
#
#            #print "DB: test dir %d" % numTries #DEBUG
#
#            #kick solution along random direction until the value of f starts to get smaller again (if it ever does)
#            #  (this indicates we've gone over a maximum along this direction)
#            last_f = cur_f
#            for i in range(1,MAX_STEPS):
#                test_x = cur_x0 + i*STEP * direction
#                test_f = f(test_x)
#                #print "DB: test step=%f: f=%f" % (i*STEP, test_f)
#                if test_f < last_f:
#                    trial_x0 = test_x
#                    print "Found new direction in %d tries, new f(x0) = %g" % (numTries,test_f)
#                    break
#                last_f = test_f
#
#        if trial_x0 is None:
#            raise ValueError("Maximum number of direction tries exceeded")
#
#    solution = _optResult()
#    solution.x = global_best_params; solution.fun = global_best
#    solution.success = True
##    if it < maxiter:
##        solution.success = True
##    else:
##        solution.success = False
##        solution.message = "Maximum iterations exceeded"
#    return solution



def create_obj_func_printer(objFunc):
    """
    Create a callback function that prints the value of an objective function.

    Parameters
    ----------
    objFunc : function
        The objective function to print.

    Returns
    -------
    function
        A callback function which prints objFunc.
    """
    def print_obj_func(x,f=None,accepted=None): # Just print the objective function value (used to monitor convergence in a callback)
        if f is not None and accepted is not None:
            print("%5ds %22.10f %s" % (_time.time()-startTime, f, 'accepted' if accepted else 'not accepted'))
        else:
            print("%5ds %22.10f" % (_time.time()-startTime, objFunc(x)))
    return print_obj_func



def _fwd_diff_jacobian(f, x0, eps=1e-10):
    y0 = f(x0)
    M = len(y0)
    N = len(x0)
    jac = _np.empty( (M,N), 'd' )

    for j in range(N):
        xj = x0.copy(); xj[j] += eps
        yj = f(xj)
        jac[:,j] = (yj-y0)/eps # df_dxj

    return jac

def check_jac(f, x0, jacToCheck, eps=1e-10, tol=1e-6, errType='rel'):
    """
    Checks a jacobian function using finite differences.

    Parameters
    ----------
    f : function
        The function to check.

    x0 : numpy array
        The point at which to check the jacobian.

    jacToCheck : function
        A function which should compute the jacobian of f at x0.

    eps : float, optional
        Epsilon to use in finite difference calculations of jacobian.

    tol : float, optional
        The allowd tolerance on the relative differene between the
        values of the finite difference and jacToCheck jacobians
        if errType == 'rel' or the absolute difference if errType == 'abs'.

    Returns
    -------
    errSum : float
        The total error between the jacobians.
    errs : list
        List of (row,col,err) tuples giving the error for each row and column.
    ffd_jac : numpy array
        The computed forward-finite-difference jacobian.
    """
    orig_stdout = _sys.stdout
    devnull = open(_os.devnull, 'w')

    try:
        _sys.stdout = devnull #redirect stdout to null during the many f(x) calls
        fd_jac = _fwd_diff_jacobian(f,x0,eps)
    finally:
        _sys.stdout = orig_stdout
        devnull.close()

    assert(jacToCheck.shape == fd_jac.shape)
    M,N = jacToCheck.shape

    errSum = 0; errs = []
    if errType == 'rel':
        for i in range(M):
            for j in range(N):
                err = _np.abs(fd_jac[i,j]-jacToCheck[i,j]) / (_np.abs(fd_jac[i,j])+1e-10)
                if err > tol: errs.append( (i,j,err) )
                errSum += err

    elif errType == 'abs':
        for i in range(M):
            for j in range(N):
                err = _np.abs(fd_jac[i,j]-jacToCheck[i,j])
                if err > tol: 
                    errs.append( (i,j,err) )
                    #print("DEBUG JAC CHECK (%d,%d): %g vs %g (diff = %g)" %
                    # (i,j,fd_jac[i,j],jacToCheck[i,j],fd_jac[i,j]-jacToCheck[i,j]))
                errSum += err

    errs.sort(key=lambda x: -x[2])

    if len(errs) > 0:
        maxabs = _np.max(_np.abs(jacToCheck))
        max_err_ratio = _np.max([ x[2]/maxabs for x in errs ])
        if max_err_ratio > 0.01: 
            print("Warning: jacobian_check has max err/jac_max = %g (jac_max = %g)" % (max_err_ratio,maxabs))

    return errSum, errs, fd_jac
