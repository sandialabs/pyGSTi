"""Public optimizer extension point and GST result compatibility container."""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable


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
        the optimal `dot(transpose(J),J)` value, where `J`
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
    An independently implementable optimizer.

    New implementations override :meth:`solve` and return an
    :class:`OptimizerOutcome`. Declare ``required_operations`` before a GST
    layout is constructed. The default :meth:`run` adapts the configured GST
    objective, invokes the solver, and commits the returned point exactly.
    Existing implementations overriding ``run`` remain supported.
    """

    required_operations = ('value',)
    array_types = ()
    problem_factory = None
    uses_problem_api = True
    supports_evaluation_domain = False

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
        if isinstance(obj, Optimizer):
            return obj
        if obj is None or isinstance(obj, dict):
            from .lm import LMOptimizer
            return LMOptimizer(**(obj or {}))
        raise TypeError("Expected an Optimizer instance, a dictionary, or None")

    def __init__(self, required_operations=None, array_types=None, problem_factory=None):
        super().__init__()
        from .gstproblem import objective_methods, replicated_array_types
        if required_operations is not None:
            self.required_operations = tuple(required_operations)
        if array_types is not None:
            self.array_types = tuple(array_types)
        if problem_factory is not None:
            self.problem_factory = problem_factory
        self.called_objective_methods = objective_methods(self.required_operations)
        if self.problem_factory is None:
            self.array_types = tuple(self.array_types) + replicated_array_types(self.required_operations)

    def solve(self, problem):
        """Minimize an OptimizationProblem and return an OptimizerOutcome.

        Solver-specific tolerances, iteration policies, linear algebra, array
        backends and restart state belong to the implementation. The problem
        supplies only the mathematical operations explicitly requested.
        """
        raise NotImplementedError("Implement solve(problem), or the legacy run method")

    def run(self, objective, profiler, printer, *, domain=None):
        """Fit a GST objective using the problem-facing solver extension point."""
        from .gstproblem import run_problem_optimizer
        return run_problem_optimizer(self, objective, profiler, printer, domain=domain)


