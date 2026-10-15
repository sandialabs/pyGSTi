"""Backend-native, parameter-to-probability functions with explicit outcome order.

These functions contain no statistical loss or probability clipping. Reconstruct
one after changing model structure or its layout. NumPy calls are serialized and
collective for distributed layouts; Torch functions currently support serial
layouts only. Neither implementation promises concurrent use of a shared model.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CircuitOutcomes:
    """One circuit's observed outcomes and their indices in the flat output."""
    circuit: object
    outcomes: tuple
    indices: tuple


def _metadata(layout):
    global_layout = layout.global_layout
    metadata = []
    for indices, circuit, outcomes in global_layout.iter_unique_circuits():
        if isinstance(indices, slice):
            indices = range(*indices.indices(global_layout.num_elements))
        metadata.append(CircuitOutcomes(circuit, tuple(tuple(o) for o in outcomes),
                                        tuple(int(i) for i in indices)))
    return tuple(metadata)


class _NumPyProbabilityFunction:
    backend = 'numpy'
    derivative_storage = 'dense'

    def __init__(self, model, layout, dtype):
        self._model, self._layout = model, layout
        self._dtype = np.dtype(float if dtype is None else dtype)
        if self._dtype.kind != 'f':
            raise ValueError('NumPy probabilities require a real floating dtype')
        self.num_parameters = model.num_params
        self.num_outcomes = layout.global_layout.num_elements
        self.outcomes = _metadata(layout)

    def _point(self, x):
        x = np.asarray(x)
        if x.shape != (self.num_parameters,) or np.iscomplexobj(x) or not np.all(np.isfinite(x)):
            raise ValueError('Expected a finite real parameter vector of the model dimension')
        return x.copy()

    def _evaluate(self, x, derivative):
        x = self._point(x)
        layout = self._layout
        if derivative and not getattr(layout, '_param_dimensions', ()):
            raise ValueError("Derivative evaluation needs a layout created with array_types=('e', 'ep')")
        entry = self._model.to_vector().copy()
        array = None
        kind = 'ep' if derivative else 'e'
        try:
            self._model.from_vector(x)
            array = layout.allocate_local_array(kind, self._dtype)
            if derivative:
                self._model.sim.bulk_fill_dprobs(array, layout)
            else:
                self._model.sim.bulk_fill_probs(array, layout)
            result = np.array(layout.allgather_local_array(kind, array), copy=True)
            expected = ((self.num_outcomes, self.num_parameters) if derivative else (self.num_outcomes,))
            if result.shape != expected:
                raise ValueError('The layout derivative dimension does not match all model parameters')
            return result
        finally:
            try:
                if array is not None:
                    layout.free_local_array(array)
            finally:
                self._model.from_vector(entry)

    def __call__(self, x):
        return self._evaluate(x, False)

    def jacobian(self, x):
        """Materialize an owned global dense probability Jacobian."""
        return self._evaluate(x, True)

    def jvp(self, x, v):
        """Apply the dense Jacobian; this implementation is not matrix free."""
        v = self._point(v)
        return self.jacobian(x) @ v

    def vjp(self, x, w):
        w = np.asarray(w)
        if w.shape != (self.num_outcomes,) or np.iscomplexobj(w):
            raise ValueError('Expected one real cotangent per flat probability')
        return self.jacobian(x).T @ w


class _TorchProbabilityFunction:
    backend = 'torch'
    derivative_storage = 'autograd'

    def __init__(self, model, layout, dtype, device):
        import torch
        from .torchfwdsim import StatelessModelCircuitStore
        self._torch = torch
        self._dtype = torch.float64 if dtype is None else dtype
        if self._dtype not in (torch.float32, torch.float64):
            raise ValueError('Torch probabilities require float32 or float64')
        self._device = torch.device('cpu' if device is None else device)
        resource_alloc = getattr(layout, 'resource_alloc', None)
        if callable(resource_alloc):
            resource_alloc = resource_alloc()
        comm = getattr(resource_alloc, 'comm', None)
        if comm is not None and comm.Get_size() > 1:
            raise ValueError('Torch probability functions currently require a serial layout')
        if layout.num_elements != layout.global_layout.num_elements:
            raise ValueError('Torch probability functions require all outcomes on one process')
        self.num_parameters = model.num_params
        self.num_outcomes = layout.num_elements
        self.outcomes = _metadata(layout)
        # The internal store groups instrument branches. Build its contiguous
        # canonical order independently of the caller's layout, then restore
        # exactly the caller's observed outcome order with a differentiable gather.
        from pygsti.layouts.copalayout import CircuitOutcomeProbabilityArrayLayout
        circuits = [row.circuit for row in self.outcomes]
        canonical_elements = {}
        permutation = np.empty(self.num_outcomes, dtype=int)
        offset = 0
        for i, row in enumerate(self.outcomes):
            expanded = model.expand_instruments_and_separate_povm(row.circuit, row.outcomes)
            canonical = tuple(outcome for outcomes in expanded.values() for outcome in outcomes)
            if len(canonical) != len(row.outcomes) or set(canonical) != set(row.outcomes):
                raise ValueError('Observed outcomes do not match model instrument expansion')
            canonical_elements[i] = tuple((offset + j, outcome) for j, outcome in enumerate(canonical))
            positions = {outcome: offset + j for j, outcome in enumerate(canonical)}
            for index, outcome in zip(row.indices, row.outcomes):
                permutation[index] = positions[outcome]
            offset += len(canonical)
        canonical_layout = CircuitOutcomeProbabilityArrayLayout(
            circuits, circuits, {i: i for i in range(len(circuits))}, canonical_elements,
            [model.complete_circuit(circuit) for circuit in circuits])
        self._output_indices = torch.as_tensor(permutation, dtype=torch.long, device=self._device)
        self._store = StatelessModelCircuitStore(model, canonical_layout, self._dtype, self._device)
        indices = []
        for label, member in model._iter_parameterized_objs():
            gpindices = np.asarray(member.gpindices_as_array(), dtype=int)
            if (gpindices.shape != (member.num_params,) or np.any(gpindices < 0)
                    or np.any(gpindices >= self.num_parameters)):
                raise ValueError('Unsupported global parameter mapping for %s' % (label,))
            indices.append(torch.as_tensor(gpindices.copy(), dtype=torch.long, device=self._device))
        if len(indices) != len(self._store.param_metadata):
            raise ValueError('Model member metadata changed while creating probability function')
        self._indices = tuple(indices)

    def _point(self, x):
        torch = self._torch
        if not isinstance(x, torch.Tensor):
            raise TypeError('Torch probability functions consume a Torch tensor directly')
        if x.shape != (self.num_parameters,) or x.dtype != self._dtype or x.device != self._device:
            raise ValueError('Parameter tensor must match the function dimension, dtype, and device')
        return x

    def __call__(self, x):
        x = self._point(x)
        # Gather the actual modelmember maps. Shared and noncontiguous parameter
        # indices retain their correct chain rule through index_select.
        free_params = tuple(x.index_select(0, indices) for indices in self._indices)
        canonical = self._store.circuit_probs_from_free_params(*free_params)
        return canonical.index_select(0, self._output_indices)

    def jacobian(self, x):
        """Differentiate the native graph, materializing its dense Jacobian."""
        x = self._point(x)
        return self._torch.autograd.functional.jacobian(self, x, create_graph=True)

    def jvp(self, x, v):
        x, v = self._point(x), self._point(v)
        return self._torch.autograd.functional.jvp(self, x, v, create_graph=True)[1]

    def vjp(self, x, w):
        x = self._point(x)
        if not isinstance(w, self._torch.Tensor) or w.shape != (self.num_outcomes,):
            raise ValueError('Expected a Torch cotangent for each flat probability')
        return self._torch.autograd.functional.vjp(self, x, w, create_graph=True)[1]


def probability_function(model, layout, backend='numpy', dtype=None, device=None):
    """Create a raw probability map in the supplied layout's observed outcome order.

    The returned object supplies ``jacobian``, ``jvp``, and ``vjp``. Its immutable
    ``outcomes`` tuple identifies circuit outcomes and global output indices.
    NumPy derivatives gather a dense global Jacobian and require derivative
    storage in the supplied layout. Torch evaluation retains the input tensor's
    computation graph and gathers actual modelmember global parameter indices.
    No probability clipping, omitted-outcome completion, or loss is applied.
    """
    if backend == 'numpy':
        if device is not None:
            raise ValueError('device applies only to the Torch backend')
        return _NumPyProbabilityFunction(model, layout, dtype)
    if backend == 'torch':
        return _TorchProbabilityFunction(model, layout, dtype, device)
    raise ValueError("backend must be 'numpy' or 'torch'")
