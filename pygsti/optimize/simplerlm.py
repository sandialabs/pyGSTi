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

import os as _os
import signal as _signal
import time as _time

import numpy as _np
import scipy as _scipy

from pygsti.optimize import arraysinterface as _ari
from pygsti.optimize.customsolve import custom_solve as _custom_solve
from pygsti.baseobjs.verbosityprinter import VerbosityPrinter as _VerbosityPrinter
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable

#Make sure SIGINT will generate a KeyboardInterrupt (even if we're launched in the background)
#This may be problematic for multithreaded parallelism above pyGSTi, e.g. Dask,
#so this can be turned off by setting the PYGSTI_NO_CUSTOMLM_SIGINT environment variable
if 'PYGSTI_NO_CUSTOMLM_SIGINT' not in _os.environ:
    _signal.signal(_signal.SIGINT, _signal.default_int_handler)

#constants
_MACH_PRECISION = 1e-12


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


class Optimizer(_NicelySerializable):
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

    def __init__(self):
        super().__init__()


class SimplerLMOptimizer(Optimizer):
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

    serial_solve_proc_threshold : int, optional
        When there are fewer than this many processors, the optimizer will solve linear
        systems serially, using SciPy on a single processor, rather than using a parallelized
        Gaussian Elimination (with partial pivoting) algorithm coded in Python. Since SciPy's
        implementation is more efficient, it's not worth using the parallel version until there
        are many processors to spread the work among.

    lsvec_mode : {'normal', 'percircuit'}
        Whether the terms used in the least-squares optimization are the "elements" as computed
        by the objective function's `.terms()` and `.lsvec()` methods (`'normal'` mode) or the
        "per-circuit quantities" computed by the objective function's `.percircuit()` and
        `.lsvec_percircuit()` methods (`'percircuit'` mode).
    """

    @classmethod
    def cast(cls, obj):
        if isinstance(obj, cls):
            return obj
        if obj:
            try:
                return cls(**obj)
            except:
                from pygsti.optimize.customlm import CustomLMOptimizer
                return CustomLMOptimizer(**obj)
        return cls()

    def __init__(self, maxiter=100, maxfev=100, tol=1e-6, fditer=0, first_fditer=0, init_munu="auto", oob_check_interval=0,
                 oob_action="reject", oob_check_mode=0, serial_solve_proc_threshold=100, lsvec_mode="normal"):

        super().__init__()
        if isinstance(tol, float): tol = {'relx': 1e-8, 'relf': tol, 'f': 1.0, 'jac': tol, 'maxdx': 1.0}
        self.maxiter = maxiter
        self.maxfev = maxfev
        self.tol = tol
        self.fditer = fditer
        self.first_fditer = first_fditer
        self.init_munu = init_munu
        self.oob_check_interval = oob_check_interval
        self.oob_action = oob_action
        self.oob_check_mode = oob_check_mode
        self.array_types = 3 * ('p',) + ('e', 'ep')  # see simplish_leastsq fn "-type"s  -need to add 'jtj' type
        self.called_objective_methods = ('lsvec', 'dlsvec')  # the objective function methods we use (for mem estimate)
        self.serial_solve_proc_threshold = serial_solve_proc_threshold
        self.lsvec_mode = lsvec_mode

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({
            'maximum_iterations': self.maxiter,
            'maximum_function_evaluations': self.maxfev,
            'tolerance': self.tol,
            'number_of_finite_difference_iterations': self.fditer,
            'number_of_first_stage_finite_difference_iterations': self.first_fditer,
            'initial_mu_and_nu': self.init_munu,
            'out_of_bounds_check_interval': self.oob_check_interval,
            'out_of_bounds_action': self.oob_action,
            'out_of_bounds_check_mode': self.oob_check_mode,
            'array_types': self.array_types,
            'called_objective_function_methods': self.called_objective_methods,
            'serial_solve_number_of_processors_threshold': self.serial_solve_proc_threshold,
            'lsvec_mode': self.lsvec_mode
        })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        return cls(maxiter=state['maximum_iterations'],
                   maxfev=state['maximum_function_evaluations'],
                   tol=state['tolerance'],
                   fditer=state['number_of_finite_difference_iterations'],
                   first_fditer=state['number_of_first_stage_finite_difference_iterations'],
                   init_munu=state['initial_mu_and_nu'],
                   oob_check_interval=state['out_of_bounds_check_interval'],
                   oob_action=state['out_of_bounds_action'],
                   oob_check_mode=state['out_of_bounds_check_mode'],
                   serial_solve_proc_threshold=state['serial_solve_number_of_processors_threshold'],
                   lsvec_mode=state.get('lsvec_mode', 'normal'))

    def run(self, objective, profiler, printer):

        """
        Perform the optimization.

        Parameters
        ----------
        objective : ObjectiveFunction
            The objective function to optimize.

        profiler : Profiler
            A profiler to track resource usage.

        printer : VerbosityPrinter
            printer to use for sending output to stdout.
        """
        nExtra = objective.ex  # number of additional "extra" elements

        if self.lsvec_mode == 'normal':
            objective_func = objective.lsvec
            jacobian = objective.dlsvec
            nEls = objective.layout.num_elements + nExtra  # 'e' for array types
        elif self.lsvec_mode == 'percircuit':
            objective_func = objective.lsvec_percircuit
            jacobian = objective.dlsvec_percircuit
            nEls = objective.layout.num_circuits + nExtra  # 'e' for array types
        else:
            raise ValueError("Invalid `lsvec_mode`: %s" % str(self.lsvec_mode))

        x0 = objective.model.to_vector()
        x_limits = objective.model.parameter_bounds
        # x_limits should be a (num_params, 2)-shaped array, holding on each row the (min, max) values for the
        #  corresponding parameter (element of the "x" vector) or `None`.  If `None`, then no limits are imposed.

        # Check memory limit can handle what simplish_leastsq will "allocate"
        nP = len(x0)  # 'p' for array types
        objective.resource_alloc.check_can_allocate_memory(3 * nP + nEls + nEls * nP + nP * nP)  # see array_types above

        from ..layouts.distlayout import DistributableCOPALayout as _DL
        if isinstance(objective.layout, _DL):
            ari = _ari.DistributedArraysInterface(objective.layout, self.lsvec_mode, nExtra)
        else:
            ari = _ari.UndistributedArraysInterface(nEls, nP)

        opt_x, converged, msg, mu, nu, norm_f, f, opt_jtj = simplish_leastsq(
            objective_func, jacobian, x0,
            max_iter=self.maxiter,
            num_fd_iters=self.fditer,
            f_norm2_tol=self.tol.get('f', 1.0),
            jac_norm_tol=self.tol.get('jac', 1e-6),
            rel_ftol=self.tol.get('relf', 1e-6),
            rel_xtol=self.tol.get('relx', 1e-8),
            max_dx_scale=self.tol.get('maxdx', 1.0),
            init_munu=self.init_munu,
            oob_check_interval=self.oob_check_interval,
            oob_action=self.oob_action,
            oob_check_mode=self.oob_check_mode,
            resource_alloc=objective.resource_alloc,
            arrays_interface=ari,
            serial_solve_proc_threshold=self.serial_solve_proc_threshold,
            x_limits=x_limits,
            verbosity=printer - 1, profiler=profiler)

        printer.log("Least squares message = %s" % msg, 2)
        assert(converged), "Failed to converge: %s" % msg
        current_v = objective.model.to_vector()
        if not _np.allclose(current_v, opt_x):  # ensure the last model evaluation was at opt_x
            objective_func(opt_x)
            #objective.model.from_vector(opt_x)  # performed within line above

        #DEBUG CHECK SYNC between procs (especially for shared mem) - could REMOVE
        # if objective.resource_alloc.comm is not None:
        #     comm = objective.resource_alloc.comm
        #     v_cmp = comm.bcast(objective.model.to_vector() if (comm.Get_rank() == 0) else None, root=0)
        #     v_matches_x = _np.allclose(objective.model.to_vector(), opt_x)
        #     same_as_root = _np.isclose(_np.linalg.norm(objective.model.to_vector() - v_cmp), 0.0)
        #     if not (v_matches_x and same_as_root):
        #         raise ValueError("Rank %d CUSTOMLM ERROR: END model vector-matches-x=%s and vector-is-same-as-root=%s"
        #                          % (comm.rank, str(v_matches_x), str(same_as_root)))
        #     comm.barrier()  # if we get past here, then *all* processors are OK
        #     if comm.rank == 0:
        #         print("OK - model vector == best_x and all vectors agree w/root proc's")

        unpenalized_f = f[0:-objective.ex] if (objective.ex > 0) else f
        unpenalized_normf = sum(unpenalized_f**2)  # objective function without penalty factors
        chi2k_qty = objective.chi2k_distributed_qty(norm_f)

        return OptimizerResult(objective, opt_x, norm_f, opt_jtj, unpenalized_normf, chi2k_qty,
                               {'msg': msg, 'mu': mu, 'nu': nu, 'fvec': f})



def damp_coeff_update(mu, nu, half_max_nu, reject_msg, printer):
    ############################################################################################
    #
    #   if this point is reached, either the linear solve failed
    #   or the error did not reduce.  In either case, reject increment.
    #
    ############################################################################################
    mu *= nu
    if nu > half_max_nu:  # watch for nu getting too large (&overflow)
        msg = "Stopping after nu overflow!"
    else:
        msg = ""
    nu = 2 * nu
    printer.log("      Rejected%s!  mu => mu*nu = %g, nu => 2*nu = %g" % (reject_msg, mu, nu), 2)
    return mu, nu, msg


def simplish_leastsq(
    obj_fn, jac_fn, x0, f_norm2_tol=1e-6, jac_norm_tol=1e-6,
    rel_ftol=1e-6, rel_xtol=1e-6, max_iter=100, num_fd_iters=0, max_dx_scale=1.0,
    init_munu="auto", oob_check_interval=0, oob_action="reject", oob_check_mode=0,
    resource_alloc=None, arrays_interface=None, serial_solve_proc_threshold=100,
    x_limits=None, verbosity=0, profiler=None
    ):
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

    arrays_interface : ArraysInterface
        An object that provides an interface for creating and manipulating data arrays.

    serial_solve_proc_threshold : int optional
        When there are fewer than this many processors, the optimizer will solve linear
        systems serially, using SciPy on a single processor, rather than using a parallelized
        Gaussian Elimination (with partial pivoting) algorithm coded in Python. Since SciPy's
        implementation is more efficient, it's not worth using the parallel version until there
        are many processors to spread the work among.

    x_limits : numpy.ndarray, optional
        A (num_params, 2)-shaped array, holding on each row the (min, max) values for the corresponding
        parameter (element of the "x" vector).  If `None`, then no limits are imposed.

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
    resource_alloc = _ResourceAllocation.cast(resource_alloc)
    comm = resource_alloc.comm
    printer = _VerbosityPrinter.create_printer(verbosity, comm)
    ari = arrays_interface  # shorthand

    msg = ""
    converged = False
    global_x = x0.copy()
    f = obj_fn(global_x)  # 'E'-type array
    norm_f = ari.norm2_f(f)
    half_max_nu = 2**62  # what should this be??
    tau = 1e-3
    nu = 2
    mu = 1  # just a guess - initialized on 1st iter and only used if rejected

    #Allocate potentially shared memory used in loop
    JTJ = ari.allocate_jtj()
    JTf = ari.allocate_jtf()
    x = ari.allocate_jtf()

    if num_fd_iters > 0:
        fdJac = ari.allocate_jac()

    ari.allscatter_x(global_x, x)

    if x_limits is not None:
        x_lower_limits = ari.allocate_jtf()
        x_upper_limits = ari.allocate_jtf()
        ari.allscatter_x(x_limits[:, 0], x_lower_limits)
        ari.allscatter_x(x_limits[:, 1], x_upper_limits)

    dx = ari.allocate_jtf()
    new_x = ari.allocate_jtf()
    global_new_x = global_x.copy()

    # don't let any component change by more than ~max_dx_scale
    if max_dx_scale:
        max_norm_dx = (max_dx_scale**2) * len(global_x)
    else: max_norm_dx = None

    if not _np.isfinite(norm_f):
        msg = "Infinite norm of objective function at initial point!"

    if len(global_x) == 0:  # a model with 0 parameters - nothing to optimize
        msg = "No parameters to optimize"
        converged = True

    last_accepted_dx = None
    min_norm_f = 1e100  # sentinel
    best_x = ari.allocate_jtf()
    best_x[:] = x[:]  # like x.copy() -the x-value corresponding to min_norm_f ('P'-type)

    if init_munu != "auto":
        mu, nu = init_munu
    best_x_state = (mu, nu, norm_f, f.copy(), None)  # need f.copy() b/c f is objfn mem
    rawJTJ_scratch = None
    jtj_buf = ari.allocate_jtj_shared_mem_buf()

    try:

        for k in range(max_iter):  # outer loop
            # assume global_x, x, f, fnorm hold valid values

            if len(msg) > 0:
                break  # exit outer loop if an exit-message has been set

            if norm_f < f_norm2_tol:
                if oob_check_interval <= 1:
                    msg = "Sum of squares is at most %g" % f_norm2_tol
                    converged = True
                    break
                else:
                    printer.log(("** Converged with out-of-bounds with check interval=%d, reverting to last know in-bounds point and setting interval=1 **") % oob_check_interval, 2)
                    oob_check_interval = 1
                    x[:] = best_x[:]
                    mu, nu, norm_f, f[:], _ = best_x_state
                    continue  # can't make use of saved JTJ yet - recompute on nxt iter

            if profiler: profiler.memory_check("simplish_leastsq: begin outer iter")

            # unnecessary b/c global_x is already valid: ari.allgather_x(x, global_x)
            if k >= num_fd_iters:
                Jac = jac_fn(global_x)  # 'EP'-type, but doesn't actually allocate any more mem (!)
            else:
                # Note: x holds only number of "fine"-division params - need to use global_x, and
                # Jac only holds a subset of the derivative and element columns and rows, respectively.
                f_fixed = f.copy()  # a static part of the distributed `f` resturned by obj_fn - MUST copy this.

                pslice = ari.jac_param_slice(only_if_leader=True)
                eps = 1e-7
                #Don't do this: for ii, i in enumerate(range(pslice.start, pslice.stop)): (must keep procs in sync)
                for i in range(len(global_x)):
                    x_plus_dx = global_x.copy()
                    x_plus_dx[i] += eps
                    fd = (obj_fn(x_plus_dx) - f_fixed) / eps
                    if pslice.start <= i < pslice.stop:
                        fdJac[:, i - pslice.start] = fd
                    #if comm is not None: comm.barrier()  # overkill for shared memory leader host barrier
                Jac = fdJac
                #DEBUG: compare with analytic jacobian (need to uncomment num_fd_iters DEBUG line above too)
                #Jac_analytic = jac_fn(x)
                #if _np.linalg.norm(Jac_analytic-Jac) > 1e-6:
                #    print("JACDIFF = ",_np.linalg.norm(Jac_analytic-Jac)," per el=",
                #          _np.linalg.norm(Jac_analytic-Jac)/Jac.size," sz=",Jac.size)


            if profiler: profiler.memory_check("simplish_leastsq: after jacobian:"
                                               + "shape=%s, GB=%.2f" % (str(Jac.shape),
                                                                        Jac.nbytes / (1024.0**3)))
            Jnorm = _np.sqrt(ari.norm2_jac(Jac))
            xnorm = _np.sqrt(ari.norm2_x(x))
            printer.log("--- Outer Iter %d: norm_f = %g, mu=%g, |x|=%g, |J|=%g" % (k, norm_f, mu, xnorm, Jnorm))

            #assert(_np.isfinite(Jac).all()), "Non-finite Jacobian!" # NaNs tracking
            #assert(_np.isfinite(_np.linalg.norm(Jac))), "Finite Jacobian has inf norm!" # NaNs tracking

            tm = _time.time()

            # Riley note: fill_JTJ is the first place where we try to access J as a dense matrix.
            ari.fill_jtj(Jac, JTJ, jtj_buf)
            ari.fill_jtf(Jac, f, JTf)  # 'P'-type

            if profiler: profiler.add_time("simplish_leastsq: dotprods", tm)
            #assert(not _np.isnan(JTJ).any()), "NaN in JTJ!" # NaNs tracking
            #assert(not _np.isinf(JTJ).any()), "inf in JTJ! norm Jac = %g" % _np.linalg.norm(Jac) # NaNs tracking
            #assert(_np.isfinite(JTJ).all()), "Non-finite JTJ!" # NaNs tracking
            #assert(_np.isfinite(JTf).all()), "Non-finite JTf!" # NaNs tracking

            idiag = ari.jtj_diag_indices(JTJ)
            norm_JTf = ari.infnorm_x(JTf)
            norm_x = ari.norm2_x(x)
            undamped_JTJ_diag = JTJ[idiag].copy()  # 'P'-type

            JTf *= -1.0
            minus_JTf = JTf  # use the same memory for -JTf below (shouldn't use JTf anymore)
            #Maybe just have a minus_JTf variable?

            if norm_JTf < jac_norm_tol:
                if oob_check_interval <= 1:
                    msg = "norm(jacobian) is at most %g" % jac_norm_tol
                    converged = True
                    break
                else:
                    printer.log(("** Converged with out-of-bounds with check interval=%d, reverting to last know in-bounds point and setting interval=1 **") % oob_check_interval, 2)
                    oob_check_interval = 1
                    x[:] = best_x[:]
                    mu, nu, norm_f, f[:], _ = best_x_state
                    continue  # can't make use of saved JTJ yet - recompute on nxt iter

            if k == 0:
                if init_munu == "auto":
                    mu = tau * ari.max_x(undamped_JTJ_diag)  # initial damping element
                else:
                    mu, nu = init_munu
                rawJTJ_scratch = JTJ.copy()  # allocates the memory for a copy of JTJ so only update mem elsewhere
                best_x_state = mu, nu, norm_f, f.copy(), rawJTJ_scratch  # update mu,nu,JTJ of initial best state
            else:
                # on all other iterations, update JTJ of best_x_state if best_x == x, i.e. if we've just evaluated
                # a previously accepted step that was deemed the best we've seen so far
                if _np.allclose(x, best_x):
                    rawJTJ_scratch[:, :] = JTJ[:, :]  # use pre-allocated memory
                    rawJTJ_scratch[idiag] = undamped_JTJ_diag  # no damping; the "raw" JTJ
                    best_x_state = best_x_state[0:4] + (rawJTJ_scratch,)  # update mu,nu,JTJ of initial "best state"

            #determing increment using adaptive damping
            while True:  # inner loop

                if profiler: profiler.memory_check("simplish_leastsq: begin inner iter")

                # ok if assume fine-param-proc.size == 1 (otherwise need to sync setting local JTJ)
                JTJ[idiag] = undamped_JTJ_diag + mu  # augment normal equations

                #assert(_np.isfinite(JTJ).all()), "Non-finite JTJ (inner)!" # NaNs tracking
                #assert(_np.isfinite(JTf).all()), "Non-finite JTf (inner)!" # NaNs tracking

                try:
                    if profiler: profiler.memory_check("simplish_leastsq: before linsolve")
                    tm = _time.time()
                    success = True
                    _custom_solve(JTJ, minus_JTf, dx, ari, resource_alloc, serial_solve_proc_threshold)
                    if profiler: profiler.add_time("simplish_leastsq: linsolve", tm)
                except _scipy.linalg.LinAlgError:
                    success = False

                """
                We have > 180 l.o.c. for handling success==True.
                These lines should be factored out into their own function.

                    The last 100 lines of this region are just for handling new_x_is_allowed == True.
                """

                reject_msg = ""
                if profiler: profiler.memory_check("simplish_leastsq: after linsolve")

                if not success:
                    # linear solve failed
                    reject_msg = " (LinSolve Failure)"
                    mu, nu, msg = damp_coeff_update(mu, nu, half_max_nu, reject_msg, printer)
                    if len(msg) > 0:
                        break
                    else:
                        continue

                new_x[:] = x + dx
                norm_dx = ari.norm2_x(dx)

                #ensure dx isn't too large - don't let any component change by more than ~max_dx_scale
                if max_norm_dx and norm_dx > max_norm_dx:
                    dx *= _np.sqrt(max_norm_dx / norm_dx)
                    new_x[:] = x + dx
                    norm_dx = ari.norm2_x(dx)

                #apply x limits (bounds)
                if x_limits is not None:
                    # Approach 1: project x into valid space by simply clipping out-of-bounds values
                    for i, (x_el, lower, upper) in enumerate(zip(x, x_lower_limits, x_upper_limits)):
                        if new_x[i] < lower:
                            new_x[i] = lower
                            dx[i] = lower - x_el
                        elif new_x[i] > upper:
                            new_x[i] = upper
                            dx[i] = upper - x_el
                    norm_dx = ari.norm2_x(dx)

                printer.log("  - Inner Loop: mu=%g, norm_dx=%g" % (mu, norm_dx), 2)

                if norm_dx < (rel_xtol**2) * norm_x:
                    if oob_check_interval <= 1:
                        msg = "Relative change, |dx|/|x|, is at most %g" % rel_xtol
                        converged = True
                        break
                    else:
                        printer.log(("** Converged with out-of-bounds with check interval=%d, reverting to last know in-bounds point and setting interval=1 **") % oob_check_interval, 2)
                        oob_check_interval = 1
                        x[:] = best_x[:]
                        mu, nu, norm_f, f[:], _ = best_x_state
                        break

                if norm_dx > (norm_x + rel_xtol) / (_MACH_PRECISION**2):
                    msg = "(near-)singular linear system"
                    break

                if oob_check_interval > 0 and oob_check_mode == 0:
                    if k % oob_check_interval == 0:
                        #Check to see if objective function is out of bounds

                        in_bounds = []
                        ari.allgather_x(new_x, global_new_x)
                        try:
                            new_f = obj_fn(global_new_x, oob_check=True)
                        except ValueError:  # Use this to mean - "not allowed, but don't stop"
                            in_bounds.append(False)
                        else:
                            in_bounds.append(True)

                        if any(in_bounds):  # In adaptive mode, proceed if *any* cases are in-bounds
                            new_x_is_allowed = True
                            new_x_is_known_inbounds = True
                        else:
                            MIN_STOP_ITER = 1  # the minimum iteration where an OOB objective stops the optimization
                            if oob_action == "reject" or k < MIN_STOP_ITER:
                                new_x_is_allowed = False  # (and also not in bounds)
                            elif oob_action == "stop":
                                if oob_check_interval == 1:
                                    msg = "Objective function out-of-bounds! STOP"
                                    converged = True
                                    break
                                else:  # reset to last know in-bounds point and not do oob check every step
                                    printer.log(("** Hit out-of-bounds with check interval=%d, reverting to last know in-bounds point and setting interval=1 **") % oob_check_interval, 2)
                                    oob_check_interval = 1
                                    x[:] = best_x[:]
                                    mu, nu, norm_f, f[:], _ = best_x_state  # can't make use of saved JTJ yet
                                    break  # restart next outer loop
                            else:
                                raise ValueError("Invalid `oob_action`: '%s'" % oob_action)
                    else:  # don't check this time
                        ari.allgather_x(new_x, global_new_x)
                        new_f = obj_fn(global_new_x, oob_check=False)

                        new_x_is_allowed = True
                        new_x_is_known_inbounds = False
                else:
                    #Just evaluate objective function normally; never check for in-bounds condition
                    ari.allgather_x(new_x, global_new_x)
                    new_f = obj_fn(global_new_x)

                    new_x_is_allowed = True
                    new_x_is_known_inbounds = bool(oob_check_interval == 0)  # consider "in bounds" if not checking

                if not new_x_is_allowed:
                    reject_msg = " (out-of-bounds)"
                    mu, nu, msg = damp_coeff_update(mu, nu, half_max_nu, reject_msg, printer)
                    if len(msg) > 0:
                        break
                    else:
                        continue

                norm_new_f = ari.norm2_f(new_f)
                if not _np.isfinite(norm_new_f):  # avoid infinite loop...
                    msg = "Infinite norm of objective function!"
                    break

                # dL = expected decrease in ||F||^2 from linear model
                dL = ari.dot_x(dx, mu * dx + minus_JTf)
                dF = norm_f - norm_new_f      # actual decrease in ||F||^2

                printer.log("      (cont): norm_new_f=%g, dL=%g, dF=%g, reldL=%g, reldF=%g" % (norm_new_f, dL, dF, dL / norm_f, dF / norm_f), 2)

                if dL / norm_f < rel_ftol and dF >= 0 and dF / norm_f < rel_ftol and dF / dL < 2.0:
                    if oob_check_interval <= 1:  # (if 0 then no oob checking is done)
                        msg = "Both actual and predicted relative reductions in the sum of squares are at most %g" % rel_ftol
                        converged = True
                        break
                    else:
                        printer.log(("** Converged with out-of-bounds with check interval=%d, reverting to last know in-bounds point and setting interval=1 **") % oob_check_interval, 2)
                        oob_check_interval = 1
                        x[:] = best_x[:]
                        mu, nu, norm_f, f[:], _ = best_x_state  # can't make use of saved JTJ yet
                        break

                if (dL <= 0 or dF <= 0):
                    reject_msg = " (out-of-bounds)"
                    mu, nu, msg = damp_coeff_update(mu, nu, half_max_nu, reject_msg, printer)
                    if len(msg) > 0:
                        break
                    else:
                        continue

                #Check whether an otherwise acceptable solution is in-bounds
                if oob_check_mode == 1 and oob_check_interval > 0 and k % oob_check_interval == 0:
                    #Check to see if objective function is out of bounds
                    try:
                        obj_fn(global_new_x, oob_check=True)  # don't actually need return val (== new_f)
                        new_f_is_allowed = True
                        new_x_is_known_inbounds = True
                    except ValueError:  # Use this to mean - "not allowed, but don't stop"
                        MIN_STOP_ITER = 1  # the minimum iteration where an OOB objective can stops the opt.
                        if oob_action == "reject" or k < MIN_STOP_ITER:
                            new_f_is_allowed = False  # (and also not in bounds)
                        elif oob_action == "stop":
                            if oob_check_interval == 1:
                                msg = "Objective function out-of-bounds! STOP"
                                converged = True
                                break
                            else:  # reset to last know in-bounds point and not do oob check every step
                                printer.log(("** Hit out-of-bounds with check interval=%d, reverting to last know in-bounds point and setting interval=1 **") % oob_check_interval, 2)
                                oob_check_interval = 1
                                x[:] = best_x[:]
                                mu, nu, norm_f, f[:], _ = best_x_state  # can't use of saved JTJ yet
                                break  # restart next outer loop
                        else:
                            raise ValueError("Invalid `oob_action`: '%s'" % oob_action)
                else:
                    new_f_is_allowed = True

                if not new_f_is_allowed:
                    reject_msg = " (out-of-bounds)"
                    mu, nu, msg = damp_coeff_update(mu, nu, half_max_nu, reject_msg, printer)
                    if len(msg) > 0:
                        break
                    else:
                        continue

                # reduction in error: increment accepted!
                t = 1.0 - (2 * dF / dL - 1.0)**3  # dF/dL == gain ratio
                # always reduce mu for accepted step when |dx| is small
                mu_factor = max(t, 1.0 / 3.0) if norm_dx > 1e-8 else 0.3
                mu *= mu_factor
                nu = 2
                x[:] = new_x[:]
                f[:] = new_f[:]
                norm_f = norm_new_f
                global_x[:] = global_new_x[:]
                printer.log("      Accepted%s! gain ratio=%g  mu * %g => %g" % ("", dF / dL, mu_factor, mu), 2)
                last_accepted_dx = dx.copy()
                if new_x_is_known_inbounds and norm_f < min_norm_f:
                    min_norm_f = norm_f
                    best_x[:] = x[:]
                    best_x_state = (mu, nu, norm_f, f.copy(), None)
                    #Note: we use rawJTJ=None above because the current `JTJ` was evaluated
                    # at the *last* x-value -- we need to wait for the next outer loop
                    # to compute the JTJ for this best_x_state

                #assert(_np.isfinite(x).all()), "Non-finite x!" # NaNs tracking
                #assert(_np.isfinite(f).all()), "Non-finite f!" # NaNs tracking

                break  # exit inner loop normally
                # ... ^ Do we really break if we hit the end of the loop?
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
        comm.barrier()  # Just to be safe, so procs stay synchronized and we don't free anything too soon

    ari.deallocate_jtj(JTJ)
    ari.deallocate_jtf(JTf)
    ari.deallocate_jtf(x)
    ari.deallocate_jtj_shared_mem_buf(jtj_buf)
    #ari.deallocate_x_for_jac(x_for_jac)

    if x_limits is not None:
        ari.deallocate_jtf(x_lower_limits)
        ari.deallocate_jtf(x_upper_limits)

    ari.deallocate_jtf(dx)
    ari.deallocate_jtf(new_x)

    if num_fd_iters > 0:
        ari.deallocate_jac(fdJac)

    ari.allgather_x(best_x, global_x)
    ari.deallocate_jtf(best_x)

    #JTJ[idiag] = undampled_JTJ_diag #restore diagonal
    mu, nu, norm_f, f[:], rawJTJ = best_x_state

    global_f = _np.empty(ari.global_num_elements(), 'd')
    ari.allgather_f(f, global_f)

    return global_x, converged, msg, mu, nu, norm_f, global_f, rawJTJ

