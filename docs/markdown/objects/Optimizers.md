---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Optimizers and numerical problems

An optimizer implementation can live outside pyGSTi. Subclass `Optimizer`, declare the numerical information it uses, and implement `solve(problem)`. GST constructs the configured objective, supplies a numerical problem, and installs the returned parameter vector in the fitted model.

The standard problem uses real NumPy parameter vectors. A custom problem factory can additionally expose a native tensor graph or a probability/loss composition through `problem.structure`. This lets an experimental solver choose its own backend and local mathematical model without adding a solver branch to pyGSTi.

## The numerical contract

The required operation is a real scalar `value(x)`. Derivatives and least-squares structure are optional, explicitly supplied capabilities.

| Operation | Meaning |
| --- | --- |
| `gradient(x)` | Gradient of the scalar objective |
| `residual(x)` | A representation satisfying $F(x)=r(x)^T r(x)$ |
| `residual_jacobian(x)` | Dense derivative of that residual representation |
| `residual_jvp(x, v)` | $J(x)v$ |
| `residual_vjp(x, w)` | $J(x)^T w$ |
| `hessian_vector_product(x, v)` | Scalar Hessian action supplied by the problem |
| `gauss_newton_vector_product(x, v)` | $2J(x)^T J(x)v$ |

The normalization is the full sum of squares. Consequently,

$$
\nabla F=2J^Tr,\qquad
\nabla^2F=2J^TJ+2\sum_i r_i\nabla^2r_i.
$$

An exact scalar Hessian action and a Gauss–Newton action are different capabilities. Missing scalar gradients or Hessians raise `NotImplementedError`; the API does not silently select finite differences or a curvature approximation. Residual products use explicit product callbacks when supplied, or an explicitly supplied dense Jacobian. Inspect `problem.capabilities` to see the supplied operations.

```{code-cell} ipython3
import numpy as np
from pygsti.optimize import OptimizationProblem, LMOptimizer

def residual(x):
    return np.array([x[0]**2 - 1, x[1] - 2])

problem = OptimizationProblem(
    initial_point=[1.2, 0.0, 7.0],
    value=lambda x: residual(x) @ residual(x),
    residual=residual,
    residual_jacobian=lambda x: np.array([[2*x[0], 0, 0], [0, 1, 0]]),
)
outcome = LMOptimizer(tol={'f': 1e-16, 'jac': 1e-10}).solve(problem)
np.testing.assert_allclose(outcome.x, [1, 2, 7], atol=1e-7)
assert problem.value(outcome.x) < 1e-14
```

The third coordinate is unidentifiable and remains unchanged. GST can also have gauge directions and redundant parameterizations; an optimizer must not assume a positive definite Hessian or a unique fitted vector.

`initial_point` and `bounds` return owned copies. Bounds are either `None` or a pair of lower/upper vectors; evaluation does not clip the solver's trial points automatically.

## Keep a local model while evaluating trials

All standard numerical outputs are owned arrays. Evaluating a rejected trial cannot overwrite an earlier residual or Jacobian. `linearize(x)` captures a point-bound residual model:

```{code-cell} ipython3
local = problem.linearize([1.2, 0, 7])
saved_residual = local.residual
problem.residual([3, 4, 7])
np.testing.assert_array_equal(local.residual, saved_residual)
np.testing.assert_allclose(local.jvp([1, 0, 0]), [2.4, 0])
np.testing.assert_allclose(local.vjp([1, 0]), [2.4, 0, 0])
```

A dense linearization copies its Jacobian once. A product-only linearization invokes the supplied products at its captured point; it does not construct a dense matrix by probing columns. `local.storage` distinguishes these representations. `local.to_dense()` requires a supplied dense Jacobian.

## Implement an external scalar optimizer

Here is a scalar first-order example using SciPy, which is already a pyGSTi dependency. Its stopping policy belongs to this implementation. A solver's `CONVERGED` status means its stated stopping criteria were met; it is not a certificate of a globally optimal GST fit.

```{code-cell} ipython3
from scipy.optimize import minimize
from pygsti.optimize import Optimizer, OptimizerOutcome, OptimizationStatus

class ExternalLBFGS(Optimizer):
    required_operations = ('value', 'gradient')

    def __init__(self, max_iterations=100):
        super().__init__()
        self.max_iterations = max_iterations

    def solve(self, problem):
        bounds = None if problem.bounds is None else list(zip(*problem.bounds))
        result = minimize(
            problem.value, problem.initial_point, jac=problem.gradient,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': self.max_iterations},
        )
        status = (OptimizationStatus.CONVERGED if result.success else
                  OptimizationStatus.ITERATION_LIMIT if result.status == 1 else
                  OptimizationStatus.FAILED)
        return OptimizerOutcome(
            result.x, status, str(result.message),
            diagnostics={'iterations': int(result.nit),
                         'gradient_inf': float(np.linalg.norm(result.jac, np.inf))},
        )

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state['max_iterations'] = self.max_iterations
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        return cls(state['max_iterations'])
```

For persistence, put the class in an importable module and implement its configuration serialization. Runtime closures, native graphs, and external solver handles are not automatically made persistent. Reconstruct the numerical problem on each run.

Pass this object to `GateSetTomography`, `StandardGST`, or the low-level fitting kernel:

```python
result, fitted_model = pygsti.algorithms.run_gst_fit_simple(
    dataset, initial_model, circuits,
    optimizer=ExternalLBFGS(max_iterations=200),
    objective_function_builder='logl',
    resource_alloc=None,
)
print(result.outcome.status, result.outcome.message)
```

The solver returns an `OptimizerOutcome`, containing a point, explicit termination status, message, diagnostics, and optional solver-specific restart state. The GST kernel returns the existing `OptimizerResult`, augmented with `outcome`. It recomputes configured scalar statistics at the exact returned point, including penalty-free statistics; it does not require a final residual or Jacobian from a scalar-only solver. A normal return installs that point with its stated status, including budget or failure statuses. If construction, solving, or final validation raises, the model parameters at entry to that optimizer invocation are restored. Earlier completed fitting stages remain committed. The canonical LM wrapper follows this per-invocation rollback rule too.

GST run logs show the termination status and reason. Protocol estimates retain one concise summary per circuit iteration in `estimate.parameters['optimization_status']`, containing status, message, and final objective. A summary describes the last optimizer/objective run for that iteration; `None` denotes a historical optimizer or checkpoint without this information. Arbitrary runtime diagnostics and restart objects remain on low-level results rather than being implicitly serialized into estimates.

## Supply a different numerical representation

`Optimizer(problem_factory=factory, ...)` calls `factory(context, required_operations)` after the legacy objective has been configured. The context provides `objective`, `model`, `dataset`, `circuits`, `resource_alloc`, `profiler`, `printer`, and an optional approximation `domain`.

The factory returns an `OptimizationProblem` representing the configured objective. It can attach an explicitly documented object to `problem.structure`, such as a native Torch residual function or the composition $F=h\circ p$ of a probability map and an outer loss. The solver and factory agree on that extension object; pyGSTi does not translate arbitrary loss functions between numerical backends.

Numerical operations and simulator storage planning are separate decisions:

```python
super().__init__(
    required_operations=('value', 'gradient'),
    problem_factory=native_factory,
    objective_methods=('fn',),  # legacy storage needed for final statistics
    array_types=(),            # declare additional solver/factory storage here
)
```

The default factory conservatively maps requested operations to legacy objective storage. A custom factory can override `objective_methods`; `fn` is always included for final statistics. For example, a native gradient need not allocate a legacy dense Jacobian. This hook still materializes the legacy objective's scalar/data buffers. `array_types` uses the existing layout memory vocabulary; uppercase dimensions describe global storage. Iterative fitting plans an envelope over all stages' optimizer requirements.

The default factory returns owned global arrays, gathering distributed rows and columns through the objective layout. Its derivative products use dense Jacobians. A product method alone does not make its implementation matrix-free. A native or operator factory must explicitly supply its own products and resource declarations to avoid that storage.

## Separate forward simulation from the statistical loss

`model.sim.probability_function(layout, backend=...)` returns a parameter-to-probability function. It exposes `jacobian`, `jvp`, and `vjp`, along with immutable circuit/outcome/index metadata in `outcomes`.

```python
layout = model.sim.create_layout(circuits, dataset, array_types=('e', 'ep'))
p = model.sim.probability_function(layout, backend='numpy')
probabilities = p(model.to_vector())
for row in p.outcomes:
    print(row.circuit, row.outcomes, probabilities[list(row.indices)])
```

The NumPy wrapper restores the model's entry vector after each call and returns owned arrays. Its derivatives are explicitly dense and require a layout with derivative storage. The Torch backend consumes a tensor directly, preserves its graph, gathers actual modelmember parameter indices, and restores the caller's requested outcome order after canonical instrument expansion:

```python
import torch
p_torch = model.sim.probability_function(layout, backend='torch', dtype=torch.float64)
x = torch.tensor(model.to_vector(), dtype=torch.float64, requires_grad=True)
probabilities = p_torch(x)
```

The current native Torch contract supports first derivatives of compatible real dense model representations on a serial layout. It rejects parameter interposers and complex dense representations explicitly. A retained graph does not guarantee correct higher derivatives through every modelmember. A factory must not advertise exact Hessian actions using an unsupported double-backward path.

Raw probability functions apply no statistical loss, probability clipping, or omitted-outcome completion. Instrument branches represent joint probabilities; normalizing each branch would change the statistical problem. A loss adapter must implement its configured clipping, missing-outcome, weighting, and penalty semantics or reject unsupported configurations. Rebuild the function after structural changes to a model or layout.

This separation supports genuinely different local models. A probability-space SCP method can retain a convex nonlinear outer loss:

$$
\min_s\; h\bigl(p(x)+Dp(x)s\bigr)+\lambda\|s\|^2,
\qquad \|s\|_\infty\leq\Delta.
$$

For a Poisson loss, this can retain the logarithm of affine probabilities and impose its positivity domain in a convex subproblem. Step acceptance still uses the actual nonlinear configured objective. A physical model parameterization does not make its affine probability approximation automatically physical.

## Collective execution and approximation domains

With an MPI communicator, the standard external-solver adapter calls `solve` only on rank zero. Other ranks serve the same standard numerical operations. Solver trials, random choices, and external-library branches therefore do not need to be replicated. Construction/validation failures are synchronized at phase boundaries, and recoverable oracle-boundary errors are communicated to the solver.

Factories and simulator oracles must enter their own internal collectives consistently. The adapter cannot repair an implementation that raises on one rank while other ranks are blocked inside an internal collective. The current root-driven service rejects backend-native `structure`; such an extension needs its own explicit distributed execution model. Outcomes and restart state must be MPI-serializable. Numerical callbacks captured from the service cannot be used after the solve ends.

A truncated path set describes a fixed approximate objective on a validity domain. `EvaluationDomainError` distinguishes an insufficient approximation from parameter bounds and unrelated programming exceptions. A solver opting into `supports_evaluation_domain = True` must own its accepted incumbent, decide how to reject invalid trials, and return `DOMAIN_LIMIT` at a valid incumbent when it requests an outer restart. The API never interprets the last evaluated valid point as accepted.

Changing a path set creates a new domain generation and invalidates old guarded linearizations. The final returned point is checked against the same domain. A guarded oracle containing collectives requires a globally consistent domain decision before those collectives are entered. Native structures need explicit native domain handling. Existing term-simulator behavior remains available through `LMOptimizer`.

## Request the tuned LM implementation

`LMOptimizer(**options)` is the canonical entry point for the existing tuned Levenberg–Marquardt method. Its GST wrapper preserves all `SimplerLMOptimizer` constructor options and distributed linear algebra. `SimplerLMOptimizer` and `CustomLMOptimizer`, and their historical import paths, remain available for compatibility.

Independent `LMOptimizer.solve(problem)` uses the supplied numerical residual representation. Per-circuit residual selection and legacy path-check controls are GST-specific and are rejected there when requested. Tolerances, damping choices, and line-search policy belong to the selected algorithm; there is no universal tolerance translation across LM, first-order methods, truncated CG, and SCP.
