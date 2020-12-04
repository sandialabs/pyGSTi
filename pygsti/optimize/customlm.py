"""
Custom implementation of the Levenberg-Marquardt Algorithm
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import time as _time
import numpy as _np
import scipy as _scipy
import signal as _signal
#from scipy.optimize import OptimizeResult as _optResult

from ..tools import mpitools as _mpit
from ..tools import sharedmemtools as _smt
from ..objects.verbosityprinter import VerbosityPrinter as _VerbosityPrinter

#Make sure SIGINT will generate a KeyboardInterrupt (even if we're launched in the background)
_signal.signal(_signal.SIGINT, _signal.default_int_handler)

#constants
_MACH_PRECISION = 1e-12
#MU_TOL1 = 1e10 # ??
#MU_TOL2 = 1e3  # ??


class OptimizerResult(object):
    """
    The result from an optimization.

    Parameters
    ----------
    objective_func : ObjectiveFunction
        The objective function that was optimized.

    opt_x : numpy.ndarray
        The optimal argument (x) value.  Often a vector of parameters.

    opt_f : numpy.ndarray
        the optimal objective function (f) value.  Often this is the least-squares
        vector of objective function values.

    opt_jtj : numpy.ndarray, optional
        the optimial `dot(transpose(J),J)` value, where `J`
        is the Jacobian matrix.  This may be useful for computing
        approximate error bars.

    opt_unpenalized_f : numpy.ndarray, optional
        the optimal objective function (f) value with any
        penalty terms removed.

    chi2_k_distributed_qty : float, optional
        a value that is supposed to be chi2_k distributed.

    optimizer_specific_qtys : dict, optional
        a dictionary of additional optimization parameters.
    """
    def __init__(self, objective_func, opt_x, opt_f=None, opt_jtj=None,
                 opt_unpenalized_f=None, chi2_k_distributed_qty=None,
                 optimizer_specific_qtys=None):
        self.objective_func = objective_func
        self.x = opt_x
        self.f = opt_f
        self.jtj = opt_jtj  # jacobian.T * jacobian
        self.f_no_penalties = opt_unpenalized_f
        self.optimizer_specific_qtys = optimizer_specific_qtys
        self.chi2_k_distributed_qty = chi2_k_distributed_qty


class Optimizer(object):
    """
    An optimizer.  Optimizes an objective function.
    """

    @classmethod
    def cast(cls, obj):
        """
        Cast `obj` to a :class:`Optimizer`.

        If `obj` is already an `Optimizer` it is just returned,
        otherwise this function tries to create a new object
        using `obj` as a dictionary of constructor arguments.

        Parameters
        ----------
        obj : Optimizer or dict
            The object to cast.

        Returns
        -------
        Optimizer
        """
        if isinstance(obj, cls):
            return obj
        else:
            return cls(**obj) if obj else cls()


class CustomLMOptimizer(Optimizer):
    """
    A Levenberg-Marquardt optimizer customized for GST-like problems.

    Parameters
    ----------
    maxiter : int, optional
        The maximum number of (outer) interations.

    maxfev : int, optional
        The maximum function evaluations.

    tol : float or dict, optional
        The tolerance, specified as a single float or as a dict
        with keys `{'relx', 'relf', 'jac', 'maxdx'}`.  A single
        float sets the `'relf'` and `'jac'` elemments and leaves
        the others at their default values.

    fditer : int optional
        Internally compute the Jacobian using a finite-difference method
        for the first `fditer` iterations.  This is useful when the initial
        point lies at a special or singular point where the analytic Jacobian
        is misleading.

    first_fditer : int, optional
        Number of finite-difference iterations applied to the first
        stage of the optimization (only).  Unused.

    damping_mode : {'identity', 'JTJ', 'invJTJ', 'adaptive'}
        How damping is applied.  `'identity'` means that the damping parameter mu
        multiplies the identity matrix.  `'JTJ'` means that mu multiplies the
        diagonal or singular values (depending on `scaling_mode`) of the JTJ
        (Fischer information and approx. hessaian) matrix, whereas `'invJTJ'`
        means mu multiplies the reciprocals of these values instead.  The
        `'adaptive'` mode adaptively chooses a damping strategy.

    damping_basis : {'diagonal_values', 'singular_values'}
        Whether the the diagonal or singular values of the JTJ matrix are used
        during damping.  If `'singular_values'` is selected, then a SVD of the
        Jacobian (J) matrix is performed and damping is performed in the basis
        of (right) singular vectors.  If `'diagonal_values'` is selected, the
        diagonal values of relevant matrices are used as a proxy for the the
        singular values (saving the cost of performing a SVD).

    damping_clip : tuple, optional
        A 2-tuple giving upper and lower bounds for the values that mu multiplies.
        If `damping_mode == "identity"` then this argument is ignored, as mu always
        multiplies a 1.0 on the diagonal if the identity matrix.  If None, then no
        clipping is applied.

    use_acceleration : bool, optional
        Whether to include a geodesic acceleration term as suggested in
        arXiv:1201.5885.  This is supposed to increase the rate of
        convergence with very little overhead.  In practice we've seen
        mixed results.

    uphill_step_threshold : float, optional
        Allows uphill steps when taking two consecutive steps in nearly
        the same direction.  The condition for accepting an uphill step
        is that `(uphill_step_threshold-beta)*new_objective < old_objective`,
        where `beta` is the cosine of the angle between successive steps.
        If `uphill_step_threshold == 0` then no uphill steps are allowed,
        otherwise it should take a value between 1.0 and 2.0, with 1.0 being
        the most permissive to uphill steps.

    init_munu : tuple, optional
        If not None, a (mu, nu) tuple of 2 floats giving the initial values
        for mu and nu.

    oob_check_interval : int, optional
        Every `oob_check_interval` outer iterations, the objective function
        (`obj_fn`) is called with a second argument 'oob_check', set to True.
        In this case, `obj_fn` can raise a ValueError exception to indicate
        that it is Out Of Bounds.  If `oob_check_interval` is 0 then this
        check is never performed; if 1 then it is always performed.

    oob_action : {"reject","stop"}
        What to do when the objective function indicates (by raising a ValueError
        as described above).  `"reject"` means the step is rejected but the
        optimization proceeds; `"stop"` means the optimization stops and returns
        as converged at the last known-in-bounds point.

    oob_check_mode : int, optional
        An advanced option, expert use only.  If 0 then the optimization is
        halted as soon as an *attempt* is made to evaluate the function out of bounds.
        If 1 then the optimization is halted only when a would-be *accepted* step
        is out of bounds.
    """
    def __init__(self, maxiter=100, maxfev=100, tol=1e-6, fditer=0, first_fditer=0, damping_mode="identity",
                 damping_basis="diagonal_values", damping_clip=None, use_acceleration=False,
                 uphill_step_threshold=0.0, init_munu="auto", oob_check_interval=0,
                 oob_action="reject", oob_check_mode=0):

        if isinstance(tol, float): tol = {'relx': 1e-8, 'relf': tol, 'f': 1.0, 'jac': tol, 'maxdx': 1.0}
        self.maxiter = maxiter
        self.maxfev = maxfev
        self.tol = tol
        self.fditer = fditer
        self.first_fditer = first_fditer
        self.damping_mode = damping_mode
        self.damping_basis = damping_basis
        self.damping_clip = damping_clip
        self.use_acceleration = use_acceleration
        self.uphill_step_threshold = uphill_step_threshold
        self.init_munu = init_munu
        self.oob_check_interval = oob_check_interval
        self.oob_action = oob_action
        self.oob_check_mode = oob_check_mode
        self.array_types = 3 * ('P',) + ('E', 'PP', 'EP')  # see custom_leastsq fn "-type"s
        self.called_objective_methods = ('lsvec', 'dlsvec')  # the objective function methods we use (for mem estimate)

    def run(self, objective, comm, profiler, printer):

        """
        Perform the optimization.

        Parameters
        ----------
        objective : ObjectiveFunction
            The objective function to optimize.

        comm : mpi4py.MPI.Comm
            MPI communicator to divide optimization among multiple processors.

        profiler : Profiler
            A profiler to track resource usage.

        printer : VerbosityPrinter
            printer to use for sending output to stdout.
        """
        objective_func = objective.lsvec
        jacobian = objective.dlsvec
        x0 = objective.model.to_vector()

        # Check memory limit can handle what custom_leastsq will "allocate"
        nEls = objective.layout.num_elements; nP = len(x0)
        objective.resource_alloc.check_can_allocate_memory(3 * nP + nEls + nEls * nP + nP * nP)  # see array_types above

        opt_x, converged, msg, mu, nu, norm_f, f, opt_jtj = custom_leastsq(
            objective_func, jacobian, x0,
            max_iter=self.maxiter,
            num_fd_iters=self.fditer,
            f_norm2_tol=self.tol.get('f', 1.0),
            jac_norm_tol=self.tol.get('jac', 1e-6),
            rel_ftol=self.tol.get('relf', 1e-6),
            rel_xtol=self.tol.get('relx', 1e-8),
            max_dx_scale=self.tol.get('maxdx', 1.0),
            damping_mode=self.damping_mode,
            damping_basis=self.damping_basis,
            damping_clip=self.damping_clip,
            use_acceleration=self.use_acceleration,
            uphill_step_threshold=self.uphill_step_threshold,
            init_munu=self.init_munu,
            oob_check_interval=self.oob_check_interval,
            oob_action=self.oob_action,
            oob_check_mode=self.oob_check_mode,
            resource_alloc=objective.resource_alloc,
            verbosity=printer - 1, profiler=profiler)

        printer.log("Least squares message = %s" % msg, 2)
        assert(converged), "Failed to converge: %s" % msg
        objective.model.from_vector(opt_x)  # needed apparently

        #REMOVE DEBUG TO CHECK SYNC (especially for shared mem)
        if comm:
            v_cmp = comm.bcast(objective.model.to_vector() if (comm.Get_rank() == 0) else None, root=0)
            v_matches_x = _np.isclose(_np.linalg.norm(objective.model.to_vector() - opt_x), 0.0)
            same_as_root = _np.isclose(_np.linalg.norm(objective.model.to_vector() - v_cmp), 0.0)
            if not (v_matches_x and same_as_root):
                raise ValueError("Rank %d CUSTOMLM ERROR: END model vector-matches-x=%s and vector-is-same-as-root=%s"
                                 % (comm.rank, str(v_matches_x), str(same_as_root)))
            comm.barrier()  # if we get past here, then *all* processors are OK
            if comm.rank == 0:
                print("OK - model vector == best_x and all vectors agree w/root proc's")

        unpenalized_f = f[0:-objective.ex] if (objective.ex > 0) else f
        unpenalized_normf = sum(unpenalized_f**2)  # objective function without penalty factors
        chi2k_qty = objective.chi2k_distributed_qty(norm_f)

        return OptimizerResult(objective, opt_x, norm_f, opt_jtj, unpenalized_normf, chi2k_qty,
                               {'msg': msg, 'mu': mu, 'nu': nu, 'fvec': f})

#Scipy version...
#            opt_x, _, _, msg, flag = \
#                _spo.leastsq(objective_func, x0, xtol=tol['relx'], ftol=tol['relf'], gtol=tol['jac'],
#                             maxfev=maxfev * (len(x0) + 1), full_output=True, Dfun=jacobian)  # pragma: no cover
#            printer.log("Least squares message = %s; flag =%s" % (msg, flag), 2)            # pragma: no cover
#            opt_state = (msg,)


def custom_leastsq(obj_fn, jac_fn, x0, f_norm2_tol=1e-6, jac_norm_tol=1e-6,
                   rel_ftol=1e-6, rel_xtol=1e-6, max_iter=100, num_fd_iters=0,
                   max_dx_scale=1.0, damping_mode="identity", damping_basis="diagonal_values",
                   damping_clip=None, use_acceleration=False, uphill_step_threshold=0.0,
                   init_munu="auto", oob_check_interval=0, oob_action="reject", oob_check_mode=0,
                   resource_alloc=None, verbosity=0, profiler=None):
    """
    An implementation of the Levenberg-Marquardt least-squares optimization algorithm customized for use within pyGSTi.

    This general purpose routine mimic to a large extent the interface used by
    `scipy.optimize.leastsq`, though it implements a newer (and more robust) version
    of the algorithm.

    Parameters
    ----------
    obj_fn : function
        The objective function.  Must accept and return 1D numpy ndarrays of
        length N and M respectively.  Same form as scipy.optimize.leastsq.

    jac_fn : function
        The jacobian function (not optional!).  Accepts a 1D array of length N
        and returns an array of shape (M,N).

    x0 : numpy.ndarray
        Initial evaluation point.

    f_norm2_tol : float, optional
        Tolerace for `F^2` where `F = `norm( sum(obj_fn(x)**2) )` is the
        least-squares residual.  If `F**2 < f_norm2_tol`, then mark converged.

    jac_norm_tol : float, optional
        Tolerance for jacobian norm, namely if `infn(dot(J.T,f)) < jac_norm_tol`
        then mark converged, where `infn` is the infinity-norm and
        `f = obj_fn(x)`.

    rel_ftol : float, optional
        Tolerance on the relative reduction in `F^2`, that is, if
        `d(F^2)/F^2 < rel_ftol` then mark converged.

    rel_xtol : float, optional
        Tolerance on the relative value of `|x|`, so that if
        `d(|x|)/|x| < rel_xtol` then mark converged.

    max_iter : int, optional
        The maximum number of (outer) interations.

    num_fd_iters : int optional
        Internally compute the Jacobian using a finite-difference method
        for the first `num_fd_iters` iterations.  This is useful when `x0`
        lies at a special or singular point where the analytic Jacobian is
        misleading.

    max_dx_scale : float, optional
        If not None, impose a limit on the magnitude of the step, so that
        `|dx|^2 < max_dx_scale^2 * len(dx)` (so elements of `dx` should be,
        roughly, less than `max_dx_scale`).

    damping_mode : {'identity', 'JTJ', 'invJTJ', 'adaptive'}
        How damping is applied.  `'identity'` means that the damping parameter mu
        multiplies the identity matrix.  `'JTJ'` means that mu multiplies the
        diagonal or singular values (depending on `scaling_mode`) of the JTJ
        (Fischer information and approx. hessaian) matrix, whereas `'invJTJ'`
        means mu multiplies the reciprocals of these values instead.  The
        `'adaptive'` mode adaptively chooses a damping strategy.

    damping_basis : {'diagonal_values', 'singular_values'}
        Whether the the diagonal or singular values of the JTJ matrix are used
        during damping.  If `'singular_values'` is selected, then a SVD of the
        Jacobian (J) matrix is performed and damping is performed in the basis
        of (right) singular vectors.  If `'diagonal_values'` is selected, the
        diagonal values of relevant matrices are used as a proxy for the the
        singular values (saving the cost of performing a SVD).

    damping_clip : tuple, optional
        A 2-tuple giving upper and lower bounds for the values that mu multiplies.
        If `damping_mode == "identity"` then this argument is ignored, as mu always
        multiplies a 1.0 on the diagonal if the identity matrix.  If None, then no
        clipping is applied.

    use_acceleration : bool, optional
        Whether to include a geodesic acceleration term as suggested in
        arXiv:1201.5885.  This is supposed to increase the rate of
        convergence with very little overhead.  In practice we've seen
        mixed results.

    uphill_step_threshold : float, optional
        Allows uphill steps when taking two consecutive steps in nearly
        the same direction.  The condition for accepting an uphill step
        is that `(uphill_step_threshold-beta)*new_objective < old_objective`,
        where `beta` is the cosine of the angle between successive steps.
        If `uphill_step_threshold == 0` then no uphill steps are allowed,
        otherwise it should take a value between 1.0 and 2.0, with 1.0 being
        the most permissive to uphill steps.

    init_munu : tuple, optional
        If not None, a (mu, nu) tuple of 2 floats giving the initial values
        for mu and nu.

    oob_check_interval : int, optional
        Every `oob_check_interval` outer iterations, the objective function
        (`obj_fn`) is called with a second argument 'oob_check', set to True.
        In this case, `obj_fn` can raise a ValueError exception to indicate
        that it is Out Of Bounds.  If `oob_check_interval` is 0 then this
        check is never performed; if 1 then it is always performed.

    oob_action : {"reject","stop"}
        What to do when the objective function indicates (by raising a ValueError
        as described above).  `"reject"` means the step is rejected but the
        optimization proceeds; `"stop"` means the optimization stops and returns
        as converged at the last known-in-bounds point.

    oob_check_mode : int, optional
        An advanced option, expert use only.  If 0 then the optimization is
        halted as soon as an *attempt* is made to evaluate the function out of bounds.
        If 1 then the optimization is halted only when a would-be *accepted* step
        is out of bounds.

    resource_alloc : ResourceAllocation, optional
        When not None, an resource allocation object used for distributing the computation
        across multiple processors.

    verbosity : int, optional
        Amount of detail to print to stdout.

    profiler : Profiler, optional
        A profiler object used for to track timing and memory usage.

    Returns
    -------
    x : numpy.ndarray
        The optimal solution.
    converged : bool
        Whether the solution converged.
    msg : str
        A message indicating why the solution converged (or didn't).
    """
    comm = resource_alloc.comm
    printer = _VerbosityPrinter.create_printer(verbosity, comm)
    shared_mem_leader = resource_alloc.is_host_leader

    # MEM from ..objects.profiler import Profiler
    # MEM debug_prof = Profiler(comm, True)
    # MEM profiler = debug_prof

    msg = ""
    converged = False
    x = x0
    f = obj_fn(x)  # 'E'-type array
    norm_f = _np.dot(f, f)  # _np.linalg.norm(f)**2
    half_max_nu = 2**62  # what should this be??
    tau = 1e-3
    alpha = 0.5  # for acceleration
    nu = 2
    mu = 1  # just a guess - initialized on 1st iter and only used if rejected
    my_mpidot_qtys = None
    JTJ = None
    JTJ_shm = None  # shared memory handle (used when sharing memory)

    # don't let any component change by more than ~max_dx_scale
    if max_dx_scale:
        max_norm_dx = (max_dx_scale**2) * x.size
    else: max_norm_dx = None

    if not _np.isfinite(norm_f):
        msg = "Infinite norm of objective function at initial point!"

    if x.size == 0:  # a model with 0 parameters - nothing to optimize
        msg = "No parameters to optimize"; converged = True

    # DB: from ..tools import matrixtools as _mt
    # DB: print("DB F0 (%s)=" % str(f.shape)); _mt.print_mx(f,prec=0,width=4)
    #num_fd_iters = 1000000 # DEBUG: use finite difference iterations instead
    # print("DEBUG: setting num_fd_iters == 0!");  num_fd_iters = 0 # DEBUG
    dx = None; last_accepted_dx = None
    min_norm_f = 1e100  # sentinel
    best_x = x.copy()  # the x-value corresponding to min_norm_f ('P'-type)

    spow = 0.0  # for damping_mode == 'adaptive'
    if damping_clip is not None:
        def dclip(ar): return _np.clip(ar, damping_clip[0], damping_clip[1])
    else:
        def dclip(ar): return ar

    if init_munu != "auto":
        mu, nu = init_munu
    best_x_state = (mu, nu, norm_f, f, spow, None)
    rawJTJ_scratch = None

    try:

        for k in range(max_iter):  # outer loop
            # assume x, f, fnorm hold valid values

            #t0 = _time.time() # REMOVE
            if len(msg) > 0:
                break  # exit outer loop if an exit-message has been set

            if norm_f < f_norm2_tol:
                if oob_check_interval <= 1:
                    msg = "Sum of squares is at most %g" % f_norm2_tol
                    converged = True; break
                else:
                    printer.log(("** Converged with out-of-bounds with check interval=%d, reverting to last "
                                 "know in-bounds point and setting interval=1 **") % oob_check_interval, 2)
                    oob_check_interval = 1
                    x[:] = best_x[:]
                    mu, nu, norm_f, f, spow, _ = best_x_state  # can't make use of saved JTJ yet - recompute on nxt iter
                    continue

            #printer.log("--- Outer Iter %d: norm_f = %g, mu=%g" % (k,norm_f,mu))

            if profiler: profiler.memory_check("custom_leastsq: begin outer iter *before de-alloc*")
            Jac = None; JTf = None  # JTJ = None  # we re-use JTJ now via mpidot's `out` argument.

            if profiler: profiler.memory_check("custom_leastsq: begin outer iter")
            if k >= num_fd_iters:
                Jac = jac_fn(x)  # 'EP'-type, but doesn't actually allocate any more mem (!)
            else:
                eps = 1e-7
                Jac = _np.empty((len(f), len(x)), 'd')  # 'EP'-type
                for i in range(len(x)):
                    x_plus_dx = x.copy()
                    x_plus_dx[i] += eps
                    Jac[:, i] = (obj_fn(x_plus_dx) - f) / eps

            #DEBUG shared mem REMOVE
            #printer.log("PT2: %.3fs" % (_time.time()-t0)) # REMOVE
            #comm.barrier()
            #print("DB(%d): Jnorm = " % comm.rank, _np.linalg.norm(Jac))
            #comm.barrier()
            #print("DB(%d): fnorm = " % comm.rank, _np.linalg.norm(f))
            #comm.barrier()
            #print("DB(%d): xnorm = " % comm.rank, _np.linalg.norm(x))
            #comm.barrier()
            #x_list = comm.gather(_np.linalg.norm(x), root=0)
            #Jac_list = comm.gather(_np.linalg.norm(Jac), root=0)
            ##f_list = comm.gather(_np.linalg.norm(f), root=0)
            #if comm.rank == 0:
            #    #if all([_np.isclose(x_list[0], _x) for _x in x_list]):
            #    #    pass #print("x OK")
            #    #else:
            #    #    print("x BAD!!!"); assert(False)
            #    if all([_np.isclose(Jac_list[0], _x) for _x in Jac_list]):
            #        pass #print("Jac OK")
            #    else:
            #        print("Jac BAD!!!"); assert(False)
            #    #if all([_np.isclose(f_list[0], _x) for _x in f_list]):
            #    #    pass #print("f OK")
            #    #else:
            #    #    print("f BAD!!!"); assert(False)


            #DEBUG: compare with analytic jacobian (need to uncomment num_fd_iters DEBUG line above too)
            #Jac_analytic = jac_fn(x)
            #if _np.linalg.norm(Jac_analytic-Jac) > 1e-6:
            #    print("JACDIFF = ",_np.linalg.norm(Jac_analytic-Jac)," per el=",
            #          _np.linalg.norm(Jac_analytic-Jac)/Jac.size," sz=",Jac.size)

            # DB: from ..tools import matrixtools as _mt
            # DB: print("DB JAC (%s)=" % str(Jac.shape)); _mt.print_mx(Jac,prec=0,width=4); assert(False)
            if profiler: profiler.memory_check("custom_leastsq: after jacobian:"
                                               + "shape=%s, GB=%.2f" % (str(Jac.shape),
                                                                        Jac.nbytes / (1024.0**3)))

            Jnorm = _np.linalg.norm(Jac)
            xnorm = _np.linalg.norm(x)
            printer.log("--- Outer Iter %d: norm_f = %g, mu=%g, |x|=%g, |J|=%g" % (k, norm_f, mu, xnorm, Jnorm))

            #assert(_np.isfinite(Jac).all()), "Non-finite Jacobian!" # NaNs tracking
            #assert(_np.isfinite(_np.linalg.norm(Jac))), "Finite Jacobian has inf norm!" # NaNs tracking

            tm = _time.time()
            if my_mpidot_qtys is None:
                my_mpidot_qtys = _mpit.distribute_for_dot(Jac.T.shape, Jac.shape, resource_alloc)
            #printer.log("PT3: %.3fs" % (_time.time()-t0)) # REMOVE
            JTJ, JTJ_shm = _mpit.mpidot(Jac.T, Jac, my_mpidot_qtys[0], my_mpidot_qtys[1],
                                        my_mpidot_qtys[2], resource_alloc, JTJ, JTJ_shm)  # _np.dot(Jac.T,Jac) 'PP'-type
            #DEBUG REMOVE
            #JTJ_list = comm.gather(_np.linalg.norm(JTJ), root=0)
            #if comm.rank == 0:
            #    if all([_np.isclose(JTJ_list[0], _x) for _x in JTJ_list]):
            #        pass #print("Jac OK")
            #    else:
            #        print("Jac BAD!!!"); assert(False)
            #JTJ = JTJ.copy(); shared_mem_leader = True

            JTf = _np.dot(Jac.T, f)  # 'P'-type
            if profiler: profiler.add_time("custom_leastsq: dotprods", tm)
            #assert(not _np.isnan(JTJ).any()), "NaN in JTJ!" # NaNs tracking
            #assert(not _np.isinf(JTJ).any()), "inf in JTJ! norm Jac = %g" % _np.linalg.norm(Jac) # NaNs tracking
            #assert(_np.isfinite(JTJ).all()), "Non-finite JTJ!" # NaNs tracking
            #assert(_np.isfinite(JTf).all()), "Non-finite JTf!" # NaNs tracking

            idiag = _np.diag_indices_from(JTJ)
            norm_JTf = _np.linalg.norm(JTf, ord=_np.inf)
            norm_x = _np.dot(x, x)  # _np.linalg.norm(x)**2
            undamped_JTJ_diag = JTJ.diagonal().copy()  # 'P'-type
            #max_JTJ_diag = JTJ.diagonal().copy()

            # FUTURE TODO: keep tallying allocated memory, i.e. array_types (stopped here)

            if damping_basis == "singular_values":
                Jac_U, Jac_s, Jac_Vh = _np.linalg.svd(Jac, full_matrices=False)
                Jac_V = _np.conjugate(Jac_Vh.T)

                #DEBUG
                #num_large_svals = _np.count_nonzero(Jac_s > _np.max(Jac_s) / 1e2)
                #Jac_Uproj = Jac_U[:,0:num_large_svals]
                #JTJ_evals, JTJ_U = _np.linalg.eig(JTJ)
                #printer.log("JTJ (dim=%d) eval min/max=%g, %g; %d large svals (of %d)" % (
                #    JTJ.shape[0], _np.min(_np.abs(JTJ_evals)), _np.max(_np.abs(JTJ_evals)),
                #                          num_large_svals, len(Jac_s)))

                #TODO REMOVE
                #JTJ_U, JTJ_s, JTJ_Vh = _np.linalg.svd(JTJ)
                #JTJ_s = _np.linalg.svd(JTJ, compute_uv=False)
                # JTJ = U * diag(JTJ_s) * Vh
                # (JTJ + mu*D) => U * (JTJ_s + mu*dvec) * Vh

            if norm_JTf < jac_norm_tol:
                if oob_check_interval <= 1:
                    msg = "norm(jacobian) is at most %g" % jac_norm_tol
                    converged = True; break
                else:
                    printer.log(("** Converged with out-of-bounds with check interval=%d, reverting to last "
                                 "know in-bounds point and setting interval=1 **") % oob_check_interval, 2)
                    oob_check_interval = 1
                    x[:] = best_x[:]
                    mu, nu, norm_f, f, spow, _ = best_x_state  # can't make use of saved JTJ yet - recompute on nxt iter
                    continue

            if k == 0:
                if init_munu == "auto":
                    if damping_mode == 'identity':
                        mu = tau * _np.max(undamped_JTJ_diag)  # initial damping element
                        #mu = min(mu, MU_TOL1)
                    else:
                        # initial multiplicative damping element
                        #mu = tau # initial damping element - but this seem to low, at least for termgap...
                        mu = min(1.0e5, _np.max(undamped_JTJ_diag) / norm_JTf)  # Erik's heuristic
                        #tries to avoid making mu so large that dx is tiny and we declare victory prematurely
                else:
                    mu, nu = init_munu
                rawJTJ_scratch = JTJ.copy()  # allocates the memory for a copy of JTJ so only update mem elsewhere
                best_x_state = mu, nu, norm_f, f, spow, rawJTJ_scratch  # update mu,nu,JTJ of initial "best state"
            else:
                #on all other iterations, update JTJ of best_x_state if best_x == x, i.e. if we've just evaluated
                # a previously accepted step that was deemed the best we've seen so far
                if _np.allclose(x, best_x):
                    rawJTJ_scratch[:, :] = JTJ[:, :]  # use pre-allocated memory
                    rawJTJ_scratch[idiag] = undamped_JTJ_diag  # no damping; the "raw" JTJ
                    best_x_state = best_x_state[0:5] + (rawJTJ_scratch,)  # update mu,nu,JTJ of initial "best state"

            #determing increment using adaptive damping
            while True:  # inner loop

                if profiler: profiler.memory_check("custom_leastsq: begin inner iter")
                #print("DB: Pre-damping JTJ diag = [",_np.min(_np.abs(JTJ[idiag])),_np.max(_np.abs(JTJ[idiag])),"]")

                if damping_mode == 'identity':
                    assert(damping_clip is None), "damping_clip cannot be used with damping_mode == 'identity'"
                    if damping_basis == "singular_values":
                        reg_Jac_s = Jac_s + mu
                        inv_JTJ = _np.dot(Jac_V, _np.dot(_np.diag(1 / reg_Jac_s**2), Jac_V.T))
                    else:
                        if shared_mem_leader:  # JTJ may be shared mem
                            JTJ[idiag] = undamped_JTJ_diag + mu  # augment normal equations
                        resource_alloc.host_comm_barrier()

                elif damping_mode == 'JTJ':
                    if damping_basis == "singular_values":
                        reg_Jac_s = Jac_s + mu * dclip(Jac_s)
                        inv_JTJ = _np.dot(Jac_V, _np.dot(_np.diag(1 / reg_Jac_s**2), Jac_V.T))
                    else:
                        if shared_mem_leader:  # JTJ may be shared mem
                            add_to_diag = mu * dclip(undamped_JTJ_diag)
                            JTJ[idiag] = undamped_JTJ_diag + add_to_diag
                        resource_alloc.host_comm_barrier()

                elif damping_mode == 'invJTJ':
                    if damping_basis == "singular_values":
                        reg_Jac_s = Jac_s + mu * dclip(1.0 / Jac_s)
                        inv_JTJ = _np.dot(Jac_V, _np.dot(_np.diag(1 / reg_Jac_s**2), Jac_V.T))
                    else:
                        if shared_mem_leader:  # JTJ may be shared mem
                            add_to_diag = mu * dclip(1.0 / undamped_JTJ_diag)
                            JTJ[idiag] = undamped_JTJ_diag + add_to_diag
                        resource_alloc.host_comm_barrier()

                elif damping_mode == 'adaptive':
                    if damping_basis == "singular_values":
                        reg_Jac_s_lst = [Jac_s + mu * dclip(Jac_s**(spow + 0.1)),
                                         Jac_s + mu * dclip(Jac_s**spow),
                                         Jac_s + mu * dclip(Jac_s**(spow - 0.1))]
                        inv_JTJ_lst = [_np.dot(Jac_V, _np.dot(_np.diag(1 / s**2), Jac_V.T)) for s in reg_Jac_s_lst]

                        #DEBUG TODO REMOVE
                        #JTJ_lst = [_np.dot(Jac_V.T, _np.dot(_np.diag(s**2), Jac_V)) for s in reg_Jac_s_lst]
                        #JTJ_evals2_lst = [_np.linalg.eigvals(jtj) for jtj in JTJ_lst]
                        #for ii,jtje in enumerate(JTJ_evals2_lst):
                        #    printer.log("%d: JTJ POST-damping (spow=%g) eval min/max=%g, %g"
                        #                % (ii, spow, _np.min(_np.abs(jtje)), _np.max(_np.abs(jtje))))
                    else:
                        add_to_diag_lst = [mu * dclip(undamped_JTJ_diag**(spow + 0.1)),
                                           mu * dclip(undamped_JTJ_diag**spow),
                                           mu * dclip(undamped_JTJ_diag**(spow - 0.1))]
                else:
                    raise ValueError("Invalid damping mode: %s" % damping_mode)

                #TODO REMOVE (DEBUG)
                #printer.log("JTJ (dim=%d, mu=%g) diag vals=" % (JTJ.shape[0], mu))
                #for ii,(ud,atd,dd) in enumerate(zip(undamped_JTJ_diag, add_to_diag, JTJ[idiag])):
                #    printer.log("%d: %g \t%g \t%g" % (ii,ud,atd,dd))
                #print("DB: Post-damping JTJ diag = [",_np.min(_np.abs(JTJ[idiag])),_np.max(_np.abs(JTJ[idiag])),"]")

                #assert(_np.isfinite(JTJ).all()), "Non-finite JTJ (inner)!" # NaNs tracking
                #assert(_np.isfinite(JTf).all()), "Non-finite JTf (inner)!" # NaNs tracking

                try:
                    if profiler: profiler.memory_check("custom_leastsq: before linsolve")
                    tm = _time.time()
                    success = True

                    if damping_basis == 'diagonal_values':
                        if damping_mode == 'adaptive':
                            dx_lst = []
                            for add_to_diag in add_to_diag_lst:
                                if shared_mem_leader:  # JTJ may be shared mem
                                    JTJ[idiag] = undamped_JTJ_diag + add_to_diag
                                resource_alloc.host_comm_barrier()
                                dx_lst.append(_scipy.linalg.solve(JTJ, -JTf, sym_pos=True))
                        else:
                            #dx = _np.linalg.solve(JTJ, -JTf)
                            #NEW scipy: dx = _scipy.linalg.solve(JTJ, -JTf, assume_a='pos') #or 'sym'
                            dx = _scipy.linalg.solve(JTJ, -JTf, sym_pos=True)

                    elif damping_basis == 'singular_values':
                        #Note: above solves JTJ*x = -JTf => x = inv_JTJ * (-JTf)
                        # but: J = U*s*Vh => JTJ = (VhT*s*UT)(U*s*Vh) = VhT*s^2*Vh, and inv_Vh = V b/c V is unitary
                        # so inv_JTJ = inv_Vh * 1/s^2 * inv_VhT = V * 1/s^2 * VT  = (N,K)*(K,K)*(K,N) if use psuedoinv
                        if damping_mode == 'adaptive':
                            dx_lst = [_np.dot(ijtj, -JTf) for ijtj in inv_JTJ_lst]  # special case
                        else:
                            dx = _np.dot(inv_JTJ, -JTf)
                    else:
                        raise ValueError("Invalid damping_basis = '%s'" % damping_basis)

                    if profiler: profiler.add_time("custom_leastsq: linsolve", tm)
                #except _np.linalg.LinAlgError:
                except _scipy.linalg.LinAlgError:
                    success = False

                if success and use_acceleration:  # Find acceleration term:
                    assert(damping_mode != 'adaptive'), "Cannot use acceleration in adaptive mode (yet)"
                    assert(damping_basis != 'singular_values'), "Cannot use acceleration w/singular-value basis (yet)"
                    df2_eps = 1.0
                    df2_dx = df2_eps * dx
                    try:
                        df2 = (obj_fn(x + df2_dx) + obj_fn(x - df2_dx) - 2 * f) / \
                            df2_eps**2  # 2nd deriv of f along dx direction
                        JTdf2 = _np.dot(Jac.T, df2)
                        dx2 = _scipy.linalg.solve(JTJ, -0.5 * JTdf2, sym_pos=True)  # Note: JTJ not init w/'adaptive'
                        dx1 = dx.copy()
                        dx += dx2  # add acceleration term to dx
                    except _scipy.linalg.LinAlgError:
                        print("WARNING - linear solve failed for acceleration term!")
                        # but ok to continue - just stick with first order term
                    except ValueError:
                        print("WARNING - value error during computation of acceleration term!")

                reject_msg = ""
                if profiler: profiler.memory_check("custom_leastsq: after linsolve")
                if success:  # linear solve succeeded
                    #dx = _hack_dx(obj_fn, x, dx, Jac, JTJ, JTf, f, norm_f)

                    if damping_mode != 'adaptive':
                        new_x = x + dx
                        norm_dx = _np.dot(dx, dx)  # _np.linalg.norm(dx)**2

                        #ensure dx isn't too large - don't let any component change by more than ~max_dx_scale
                        if max_norm_dx and norm_dx > max_norm_dx:
                            dx *= _np.sqrt(max_norm_dx / norm_dx)
                            new_x = x + dx
                            norm_dx = _np.dot(dx, dx)  # _np.linalg.norm(dx)**2
                    else:
                        new_x_lst = [x + dx for dx in dx_lst]
                        norm_dx_lst = [_np.dot(dx, dx) for dx in dx_lst]
                        norm_dx = norm_dx_lst[1]  # just use center value for printing & checks below

                        #ensure dx isn't too large - don't let any component change by more than ~max_dx_scale
                        if max_norm_dx:
                            for i, norm_dx in enumerate(norm_dx_lst):
                                if norm_dx > max_norm_dx:
                                    dx_lst[i] *= _np.sqrt(max_norm_dx / norm_dx)
                                    new_x_lst[i] = x + dx_lst[i]
                                    norm_dx_lst[i] = _np.dot(dx_lst[i], dx_lst[i])

                    printer.log("  - Inner Loop: mu=%g, norm_dx=%g" % (mu, norm_dx), 2)
                    #MEM if profiler: profiler.memory_check("custom_leastsq: mid inner loop")
                    #print("DB: new_x = ", new_x)

                    if norm_dx < (rel_xtol**2) * norm_x:  # and mu < MU_TOL2:
                        if oob_check_interval <= 1:
                            msg = "Relative change, |dx|/|x|, is at most %g" % rel_xtol
                            converged = True; break
                        else:
                            printer.log(("** Converged with out-of-bounds with check interval=%d, reverting to last "
                                         "know in-bounds point and setting interval=1 **") % oob_check_interval, 2)
                            oob_check_interval = 1
                            x[:] = best_x[:]
                            mu, nu, norm_f, f, spow, _ = best_x_state
                            break

                    if norm_dx > (norm_x + rel_xtol) / (_MACH_PRECISION**2):
                        msg = "(near-)singular linear system"; break

                    if oob_check_interval > 0 and oob_check_mode == 0:
                        assert(damping_mode != 'adaptive'), "Cannot use adaptive mode 0 with OOB support yet"
                        if k % oob_check_interval == 0:
                            #Check to see if objective function is out of bounds
                            try:
                                #print("DB: Trying |x| = ", _np.linalg.norm(new_x), " |x|^2=", _np.dot(new_x,new_x))
                                # MEM if profiler: profiler.memory_check("custom_leastsq: before oob_check obj_fn")
                                new_f = obj_fn(new_x, oob_check=True)
                                new_x_is_allowed = True
                                new_x_is_known_inbounds = True
                            except ValueError:  # Use this to mean - "not allowed, but don't stop"
                                MIN_STOP_ITER = 1  # the minimum iteration where an OOB objective stops the optimization
                                if oob_action == "reject" or k < MIN_STOP_ITER:
                                    new_x_is_allowed = False  # (and also not in bounds)
                                elif oob_action == "stop":
                                    if oob_check_interval == 1:
                                        msg = "Objective function out-of-bounds! STOP"
                                        converged = True; break
                                    else:  # reset to last know in-bounds point and not do oob check every step
                                        printer.log(
                                            ("** Hit out-of-bounds with check interval=%d, reverting to last "
                                             "know in-bounds point and setting interval=1 **") % oob_check_interval, 2)
                                        oob_check_interval = 1
                                        x[:] = best_x[:]
                                        mu, nu, norm_f, f, spow, _ = best_x_state  # can't make use of saved JTJ yet
                                        break  # restart next outer loop
                                else:
                                    raise ValueError("Invalid `oob_action`: '%s'" % oob_action)
                        else:  # don't check this time
                            new_f = obj_fn(new_x, oob_check=False)
                            new_x_is_allowed = True
                            new_x_is_known_inbounds = False
                    else:
                        #Just evaluate objective function normally; never check for in-bounds condition
                        if damping_mode == 'adaptive':
                            new_f_lst = [obj_fn(new_x) for new_x in new_x_lst]
                        else:
                            new_f = obj_fn(new_x)

                        new_x_is_allowed = True
                        new_x_is_known_inbounds = bool(oob_check_interval == 0)  # consider "in bounds" if not checking

                    if new_x_is_allowed:

                        # MEM if profiler: profiler.memory_check("custom_leastsq: after obj_fn")
                        if damping_mode == 'adaptive':
                            norm_new_f_lst = [_np.dot(new_f, new_f) for new_f in new_f_lst]
                            if any([not _np.isfinite(norm_new_f) for norm_new_f in norm_new_f_lst]):  # avoid inf loop
                                msg = "Infinite norm of objective function!"; break

                            #iMin = _np.argmin(norm_new_f_lst)  # pick lowest (best) objective
                            gain_ratio_lst = [(norm_f - nnf) / _np.dot(dx, mu * dx - JTf)
                                              for (nnf, dx) in zip(norm_new_f_lst, dx_lst)]
                            iMin = _np.argmax(gain_ratio_lst)  # pick highest (best) gain ratio
                            # but expected decrease is |f|^2 = grad(fTf) * dx = (grad(fT)*f + fT*grad(f)) * dx
                            #                                                 = (JT*f + fT*J) * dx
                            # <<more explanation>>
                            norm_new_f = norm_new_f_lst[iMin]
                            new_f = new_f_lst[iMin]
                            new_x = new_x_lst[iMin]
                            dx = dx_lst[iMin]
                            if iMin == 0: spow = min(1.0, spow + 0.1)
                            elif iMin == 2: spow = max(-1.0, spow - 0.1)
                            printer.log("ADAPTIVE damping => i=%d b/c fs=%s gains=%s => spow=%g" % (
                                iMin, str(norm_new_f_lst), str(gain_ratio_lst), spow))

                        else:
                            norm_new_f = _np.dot(new_f, new_f)  # _np.linalg.norm(new_f)**2
                            if not _np.isfinite(norm_new_f):  # avoid infinite loop...
                                msg = "Infinite norm of objective function!"; break

                        dL = _np.dot(dx, mu * dx - JTf)  # expected decrease in ||F||^2 from linear model
                        dF = norm_f - norm_new_f      # actual decrease in ||F||^2

                        #DEBUG - see if cos_phi < 0.001, say, might work as a convergence criterion
                        #if damping_basis == 'singular_values':
                        #    # projection of new_f onto solution tangent plane
                        #    new_f_proj = _np.dot(Jac_Uproj, _np.dot(Jac_Uproj.T, new_f))
                        #    # angle between residual vec and tangent plane
                        #    cos_phi = _np.sqrt(_np.dot(new_f_proj, new_f_proj) / norm_new_f)
                        #    #grad_f_norm = _np.linalg.norm(mu * dx - JTf)
                        #else:
                        #    cos_phi = 0

                        if dF <= 0 and uphill_step_threshold > 0:
                            beta = 0 if last_accepted_dx is None else \
                                _np.dot(dx, last_accepted_dx) / (_np.linalg.norm(dx)
                                                                 * _np.linalg.norm(last_accepted_dx))
                            uphill_ok = (uphill_step_threshold - beta) * norm_new_f < min(min_norm_f, norm_f)
                        else:
                            uphill_ok = False

                        if use_acceleration:
                            accel_ratio = 2 * _np.linalg.norm(dx2) / _np.linalg.norm(dx1)
                            printer.log("      (cont): norm_new_f=%g, dL=%g, dF=%g, reldL=%g, reldF=%g aC=%g" %
                                        (norm_new_f, dL, dF, dL / norm_f, dF / norm_f, accel_ratio), 2)

                        else:
                            printer.log("      (cont): norm_new_f=%g, dL=%g, dF=%g, reldL=%g, reldF=%g" %
                                        (norm_new_f, dL, dF, dL / norm_f, dF / norm_f), 2)
                            accel_ratio = 0.0

                        if dL / norm_f < rel_ftol and dF >= 0 and dF / norm_f < rel_ftol \
                           and dF / dL < 2.0 and accel_ratio <= alpha:
                            if oob_check_interval <= 1:  # (if 0 then no oob checking is done)
                                msg = "Both actual and predicted relative reductions in the" + \
                                    " sum of squares are at most %g" % rel_ftol
                                converged = True; break
                            else:
                                printer.log(("** Converged with out-of-bounds with check interval=%d, "
                                             "reverting to last know in-bounds point and setting "
                                             "interval=1 **") % oob_check_interval, 2)
                                oob_check_interval = 1
                                x[:] = best_x[:]
                                mu, nu, norm_f, f, spow, _ = best_x_state  # can't make use of saved JTJ yet
                                break

                        # MEM if profiler: profiler.memory_check("custom_leastsq: before success")

                        if (dL > 0 and dF > 0 and accel_ratio <= alpha) or uphill_ok:
                            #Check whether an otherwise acceptable solution is in-bounds
                            if oob_check_mode == 1 and oob_check_interval > 0 and k % oob_check_interval == 0:
                                #Check to see if objective function is out of bounds
                                try:
                                    #print("DB: Trying |x| = ", _np.linalg.norm(new_x), " |x|^2=", _np.dot(new_x,new_x))
                                    # MEM if profiler:
                                    # MEM    profiler.memory_check("custom_leastsq: before oob_check obj_fn mode 1")
                                    obj_fn(new_x, oob_check=True)  # don't actually need retrn val (same as new_f above)
                                    new_f_is_allowed = True
                                    new_x_is_known_inbounds = True
                                except ValueError:  # Use this to mean - "not allowed, but don't stop"
                                    MIN_STOP_ITER = 1  # the minimum iteration where an OOB objective can stops the opt.
                                    if oob_action == "reject" or k < MIN_STOP_ITER:
                                        new_f_is_allowed = False  # (and also not in bounds)
                                    elif oob_action == "stop":
                                        if oob_check_interval == 1:
                                            msg = "Objective function out-of-bounds! STOP"
                                            converged = True; break
                                        else:  # reset to last know in-bounds point and not do oob check every step
                                            printer.log(
                                                ("** Hit out-of-bounds with check interval=%d, reverting to last "
                                                 "know in-bounds point and setting interval=1 **") % oob_check_interval,
                                                2)
                                            oob_check_interval = 1
                                            x[:] = best_x[:]
                                            mu, nu, norm_f, f, spow, _ = best_x_state  # can't make use of saved JTJ yet
                                            break  # restart next outer loop
                                    else:
                                        raise ValueError("Invalid `oob_action`: '%s'" % oob_action)
                            else:
                                new_f_is_allowed = True

                            if new_f_is_allowed:
                                # reduction in error: increment accepted!
                                t = 1.0 - (2 * dF / dL - 1.0)**3  # dF/dL == gain ratio
                                # always reduce mu for accepted step when |dx| is small
                                mu_factor = max(t, 1.0 / 3.0) if norm_dx > 1e-8 else 0.3
                                mu *= mu_factor
                                nu = 2
                                x, f, norm_f = new_x, new_f, norm_new_f
                                printer.log("      Accepted%s! gain ratio=%g  mu * %g => %g"
                                            % (" UPHILL" if uphill_ok else "", dF / dL, mu_factor, mu), 2)
                                last_accepted_dx = dx.copy()
                                if new_x_is_known_inbounds and norm_f < min_norm_f:
                                    min_norm_f = norm_f
                                    best_x[:] = x[:]
                                    best_x_state = (mu, nu, norm_f, f, spow, None)
                                    #Note: we use rawJTJ=None above because the current `JTJ` was evaluated
                                    # at the *last* x-value -- we need to wait for the next outer loop
                                    # to compute the JTJ for this best_x_state

                                #assert(_np.isfinite(x).all()), "Non-finite x!" # NaNs tracking
                                #assert(_np.isfinite(f).all()), "Non-finite f!" # NaNs tracking

                                ##Check to see if we *would* switch to Q-N method in a hybrid algorithm
                                #new_Jac = jac_fn(new_x)
                                #new_JTf = _np.dot(new_Jac.T,new_f)
                                #print(" CHECK: %g < %g ?" % (_np.linalg.norm(new_JTf,
                                #    ord=_np.inf),0.02 * _np.linalg.norm(new_f)))

                                break  # exit inner loop normally
                            else:
                                reject_msg = " (out-of-bounds)"

                        #TEST TODO REMOVE - update Jac w/rank1 term given info from failed evaluation
                        #else:
                        #    if _np.sqrt(norm_dx) < 0.1:
                        #        print("DB: updating jac w/rank1")
                        #        delta_f = (new_f - f) - _np.dot(Jac,dx)  # df_actual - df_expected_from_Jac
                        #        Jac -= 0.1 * _np.outer(delta_f,dx) / norm_dx
                        #        Jnorm = _np.linalg.norm(Jac)
                        #        JTJ = _mpit.mpidot(Jac.T, Jac, my_cols_slice, comm)  # _np.dot(Jac.T,Jac)
                        #        JTf = _np.dot(Jac.T, f)
                        #        norm_JTf = _np.linalg.norm(JTf, ord=_np.inf)
                        #        undamped_JTJ_diag = JTJ.diagonal().copy()
                        #        mu *= 1/3.0 # so mu stays level when updating J
                        #        print("DB: new |J| = ",Jnorm)

                    else:
                        reject_msg = " (out-of-bounds)"

                else:
                    reject_msg = " (LinSolve Failure)"

                # if this point is reached, either the linear solve failed
                # or the error did not reduce.  In either case, reject increment.

                #Increase damping (mu), then increase damping factor to
                # accelerate further damping increases.
                mu *= nu
                if nu > half_max_nu:  # watch for nu getting too large (&overflow)
                    msg = "Stopping after nu overflow!"; break
                nu = 2 * nu
                printer.log("      Rejected%s!  mu => mu*nu = %g, nu => 2*nu = %g"
                            % (reject_msg, mu, nu), 2)

                #JTJ[idiag] = undamped_JTJ_diag  # restore diagonal - not needed anymore TODO REMOVE
            #end of inner loop

        #end of outer loop
        else:
            #if no break stmt hit, then we've exceeded max_iter
            msg = "Maximum iterations (%d) exceeded" % max_iter
            converged = True  # call result "converged" even in this case, but issue warning:
            printer.warning("Treating result as *converged* after maximum iterations (%d) were exceeded." % max_iter)

    except KeyboardInterrupt:
        if comm is not None:
            # ensure all procs agree on what best_x is (in case the interrupt occurred around x being updated)
            comm.Bcast(best_x, root=0)
            printer.log("Rank %d caught keyboard interrupt!  Returning the current solution as being *converged*."
                        % comm.Get_rank())
        else:
            printer.log("Caught keyboard interrupt!  Returning the current solution as being *converged*.")
        msg = "Keyboard interrupt!"
        converged = True

    if comm is not None:
        comm.barrier()  # Just to be safe, so procs stay synchronized

    _smt.cleanup_shared_ndarray(JTJ_shm)  # cleaup shared memory, if it was used

    #DEBUG REMOVE
    #x_list = comm.gather(_np.linalg.norm(best_x), root=0)
    #if comm.rank == 0:
    #    if all([_np.isclose(x_list[0], _x) for _x in x_list]):
    #        print("bestx OK")
    #    else: print("bestx BAD!!!")

    #JTJ[idiag] = undampled_JTJ_diag #restore diagonal
    mu, nu, norm_f, f, spow, rawJTJ = best_x_state
    return best_x, converged, msg, mu, nu, norm_f, f, rawJTJ
    #solution = _optResult()
    #solution.x = x; solution.fun = f
    #solution.success = converged
    #solution.message = msg
    #return solution


def _hack_dx(obj_fn, x, dx, jac, jtj, jtf, f, norm_f):
    #HACK1
    #if nRejects >= 2:
    #    dx = -(10.0**(1-nRejects))*x
    #    print("HACK - setting dx = -%gx!" % 10.0**(1-nRejects))
    #    return dx

    #HACK2
    if True:
        print("HACK2 - trying to find a good dx by iteratively stepping in each direction...")

        test_f = obj_fn(x + dx); cmp_normf = _np.dot(test_f, test_f)
        print("Compare with suggested step => ", cmp_normf)
        STEP = 0.0001

        #import bpdb; bpdb.set_trace()
        #gradient = -jtf
        test_dx = _np.zeros(len(dx), 'd')
        last_normf = norm_f
        for ii in range(len(dx)):

            #Try adding
            while True:
                test_dx[ii] += STEP
                test_f = obj_fn(x + test_dx); test_normf = _np.dot(test_f, test_f)
                if test_normf < last_normf:
                    last_normf = test_normf
                else:
                    test_dx[ii] -= STEP
                    break

            if test_dx[ii] == 0:  # then try subtracting
                while True:
                    test_dx[ii] -= STEP
                    test_f = obj_fn(x + test_dx); test_normf = _np.dot(test_f, test_f)
                    if test_normf < last_normf:
                        last_normf = test_normf
                    else:
                        test_dx[ii] += STEP
                        break

            if abs(test_dx[ii]) > 1e-6:
                test_prediction = norm_f + _np.dot(-2 * jtf, test_dx)
                tp2_f = f + _np.dot(jac, test_dx)
                test_prediction2 = _np.dot(tp2_f, tp2_f)
                cmp_dx = dx  # -jtf
                print(" -> Adjusting index ", ii, ":", x[ii], "+", test_dx[ii], " => ", last_normf, "(cmp w/dx: ",
                      cmp_dx[ii], test_prediction, test_prediction2, ") ",
                      "YES" if test_dx[ii] * cmp_dx[ii] > 0 else "NO")

        if _np.linalg.norm(test_dx) > 0 and last_normf < cmp_normf:
            print("FOUND HACK dx w/norm = ", _np.linalg.norm(test_dx))
            return test_dx
        else:
            print("KEEPING ORIGINAL dx")

    #HACK3
    if False:
        print("HACK3 - checking if there's a simple dx that is better...")
        test_f = obj_fn(x + dx); cmp_normf = _np.dot(test_f, test_f)
        orig_prediction = norm_f + _np.dot(2 * jtf, dx)
        Jdx = _np.dot(jac, dx)
        op2_f = f + Jdx
        orig_prediction2 = _np.dot(op2_f, op2_f)
        # main objective = fT*f = norm_f
        # at new x => (f+J*dx)T * (f+J*dx) = norm_f + JdxT*f + fT*Jdx
        #                                  = norm_f + 2*(fT*J)dx (b/c transpose of real# does nothing)
        #                                  = norm_f + 2*dxT*(JT*f)
        # prediction 2 also includes (J*dx)T * (J*dx) term = dxT * (jtj) * dx
        orig_prediction3 = orig_prediction + _np.dot(Jdx, Jdx)
        norm_dx = _np.linalg.norm(dx)
        print("Compare with suggested |dx| = ", norm_dx, " => ", cmp_normf,
              "(predicted: ", orig_prediction, orig_prediction2, orig_prediction3)
        STEP = norm_dx  # 0.0001

        #import bpdb; bpdb.set_trace()
        test_dx = _np.zeros(len(dx), 'd')
        best_ii = -1; best_normf = norm_f; best_dx = 0
        for ii in range(len(dx)):

            #Try adding a small amount
            test_dx[ii] = STEP
            test_f = obj_fn(x + test_dx); test_normf = _np.dot(test_f, test_f)
            if test_normf < best_normf:
                best_normf = test_normf
                best_dx = STEP
                best_ii = ii
            else:
                test_dx[ii] = -STEP
                test_f = obj_fn(x + test_dx); test_normf = _np.dot(test_f, test_f)
                if test_normf < best_normf:
                    best_normf = test_normf
                    best_dx = -STEP
                    best_ii = ii
            test_dx[ii] = 0

        test_dx[best_ii] = best_dx
        test_prediction = norm_f + _np.dot(2 * jtf, test_dx)
        tp2_f = f + _np.dot(jac, test_dx)
        test_prediction2 = _np.dot(tp2_f, tp2_f)

        jj = _np.argmax(_np.abs(dx))
        print("Best decrease = index", best_ii, ":", x[best_ii], '+', best_dx, "==>",
              best_normf, " (predictions: ", test_prediction, test_prediction2, ")")
        print(" compare with original dx[", best_ii, "]=", dx[best_ii],
              "YES" if test_dx[best_ii] * dx[best_ii] > 0 else "NO")
        print(" max of abs(dx) is index ", jj, ":", dx[jj], "yes" if jj == best_ii else "no")

        if _np.linalg.norm(test_dx) > 0 and best_normf < cmp_normf:
            print("FOUND HACK dx w/norm = ", _np.linalg.norm(test_dx))
            return test_dx
        else:
            print("KEEPING ORIGINAL dx")
    return dx


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
#        jac = None; jtj = None; jtf = None
#
#        if profiler: profiler.mem_check("custom_leastsq: begin outer iter")
#        jac = jac_fn(x)
#        if profiler: profiler.mem_check("custom_leastsq: after jacobian:"
#                                        + "shape=%s, GB=%.2f" % (str(jac.shape),
#                                                        jac.nbytes/(1024.0**3)) )
#
#        tm = _time.time()
#        if my_cols_slice is None:
#            my_cols_slice = _mpit.distribute_for_dot(jac.shape[0], comm)
#        jtj = _mpit.mpidot(jac.T,jac,my_cols_slice,comm)   #_np.dot(jac.T,jac)
#        jtf = _np.dot(jac.T,f)
#        if profiler: profiler.add_time("custom_leastsq: dotprods",tm)
#
#        idiag = _np.diag_indices_from(jtj)
#        norm_JTf = _np.linalg.norm(jtf) #, ord='inf')
#        norm_x = _np.linalg.norm(x)
#        undampled_JTJ_diag = jtj.diagonal().copy()
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
#            jtj[idiag] *= (1.0 + mu) # augment normal equations
#            #jtj[idiag] += mu # augment normal equations
#
#            try:
#                if profiler: profiler.mem_check("custom_leastsq: before linsolve")
#                tm = _time.time()
#                success = True
#                dx = _np.linalg.solve(jtj, -jtf)
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
#                if norm_dx > (norm_x+rel_tol)/_MACH_PRECISION:
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
#            jtj[idiag] = undampled_JTJ_diag #restore diagonal for next inner loop iter
#        #end of inner loop
#    #end of outer loop
#    else:
#        #if no break stmt hit, then we've exceeded max_iter
#        msg = "Maximum iterations (%d) exceeded" % max_iter
#
#    return x, converged, msg
