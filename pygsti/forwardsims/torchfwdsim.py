#***************************************************************************************************
# Copyright 2024, 2026, National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
Defines a ForwardSimulator class called "TorchForwardSimulator" that can leverage the automatic
differentation and GPU-acceleration features of PyTorch.

This file also defines two helper classes: StatelessCircuit and StatelessModelCircuitStore.

See also: pyGSTi/modelmembers/torchable.py.
"""


from __future__ import annotations
from typing import Tuple, Optional, Dict, Any, Type, TYPE_CHECKING
from pygsti.modelmembers.torchable import Torchable
from pygsti.forwardsims.forwardsim import ForwardSimulator
from pygsti.forwardsims.torchfwdsim_plan import TorchEvalPlan

if TYPE_CHECKING:
    from pygsti.baseobjs.label import Label
    from pygsti.models import ExplicitOpModel
    from pygsti.circuits.circuit import SeparatePOVMCircuit
    from pygsti.layouts.copalayout import CircuitOutcomeProbabilityArrayLayout
    import torch

import warnings as warnings
import weakref as _weakref


try:
    import torch
    TORCH_ENABLED = True
    DEFAULT_REAL_TYPE = torch.float64

except ImportError:
    TORCH_ENABLED = False
    pass


class StatelessCircuit:
    """
    Helper data structure for specifying a quantum circuit (consisting of prep,
    applying a sequence of gates, and applying a POVM to the output of the last gate).
    """

    def __init__(self, spc: SeparatePOVMCircuit):
        self.prep_label = spc.circuit_without_povm[0]
        self.op_labels  = spc.circuit_without_povm[1:]
        self.povm_label = spc.povm_label
        self.effect_labels = tuple(spc.effect_labels)
        self.outcome_probs_dim = len(spc.effect_labels)
        self.effect_row_indices = None
        # ^ Set by StatelessModelCircuitStore.__init__ when effect_labels isn't the POVM's full
        #   effect-label sequence (torch_base POVM matrices stack rows in povm.keys() order).
        return


class StatelessModelCircuitStore:
    """
    A container for the information in a (layout, ExplicitOpModel) pair that's "stateless"
    in the sense of object-oriented programming:
    
        * A list of StatelessCircuits
        * Metadata for parameterized ModelMembers

    StatelessModelCircuitStores have instance functions to facilitate computation of differentiable
    circuit outcome probabilities.

    Design notes
    ------------
    Much of this functionality could be packed into the TorchForwardSimulator class.
    Keeping it separate from TorchForwardSimulator helps clarify that it uses none of
    the sophiciated machinery in TorchForwardSimulator's base class.
    """

    circuits         : list[StatelessCircuit]
    param_metadata   : list[tuple[Label, Type[Torchable], tuple[Any,...]]]
    free_param_sizes : list[int]
    params_dim       : int


    def __init__(self, model: ExplicitOpModel, layout: CircuitOutcomeProbabilityArrayLayout, dtype: torch.dtype, device: torch.device):
        from itertools import chain as _chain
        from pygsti.modelmembers.instruments.instrument import Instrument as _Instrument
        from pygsti.modelmembers.instruments.tpinstrument import TPInstrument as _TPInstrument

        self.dtype = dtype
        self.device = device
        circuits = []
        self.outcome_probs_dim = 0
        povm_effect_orders = {}
        # ^ povm label -> the POVM's full effect-label sequence, which is the row order that
        #   the POVM classes' torch_base implementations stack effects in (povm.keys() order).
        for _, circuit, outcomes in layout.iter_unique_circuits():
            expanded_circuits = model.expand_instruments_and_separate_povm(circuit, outcomes)
            # A circuit containing instrument layer(s) expands to one SeparatePOVMCircuit per
            # combination of instrument-member outcomes; each becomes its own StatelessCircuit.  The
            # layout lays this circuit's outcomes out as the concatenation of the per-SPC outcome tuples
            # in expansion order, so we guard that the layout's outcome order matches -- a dataset-ordered
            # layout could differ, which would silently mis-map probabilities to outcomes.
            if tuple(_chain(*expanded_circuits.values())) != tuple(outcomes):
                raise ValueError(
                    "TorchForwardSimulator requires a circuit's layout outcome order to match its "
                    "instrument-expansion order, but they differ (e.g. a dataset-ordered layout)."
                )
            for spc in expanded_circuits:
                c = StatelessCircuit(spc)
                # A dataset-built layout can observe a subset of the POVM's outcomes, or observe them
                # in a different order than povm.keys(); expand_instruments_and_separate_povm filters
                # and orders spc.effect_labels accordingly. Record the row indices needed to select
                # those effects from the full torch POVM matrix.
                if c.povm_label not in povm_effect_orders:
                    povm_effect_orders[c.povm_label] = model._effect_labels_for_povm(c.povm_label)
                full_order = povm_effect_orders[c.povm_label]
                if c.effect_labels != tuple(full_order):
                    c.effect_row_indices = [full_order.index(el) for el in c.effect_labels]
                circuits.append(c)
                self.outcome_probs_dim += c.outcome_probs_dim
        self.circuits = circuits
        self.povm_effect_counts = {lbl: len(order) for lbl, order in povm_effect_orders.items()}
        self.model_dim = model.dim
        self._eval_plan = None

        # We need to verify assumptions on what layout.iter_unique_circuits() returns.
        # Looking at the implementation of that function, the assumptions can be
        # framed in terms of the "layout._element_indicies" dict.
        eind = layout._element_indices
        assert isinstance(eind, dict)
        assert len(eind) > 0
        items = iter(eind.items())
        k_prev, v_prev = next(items)
        assert k_prev == 0
        assert v_prev.start == 0
        for k, v in items:
            assert k == k_prev + 1
            assert v.start == v_prev.stop
            k_prev = k
            v_prev = v
        assert self.outcome_probs_dim == v_prev.stop
        self.param_metadata = []
        self.instrument_expansions = {}
        # ^ maps an instrument's model label -> the tuple of its expanded per-member op labels, in
        #   obj.keys() order == the order Instrument.torch_base stacks members. simplify_operations
        #   is the same convention model.py itself uses to build these labels (e.g. when expanding
        #   instrument layers in a circuit), so this can't drift out of sync with it.
        for lbl, obj in model._iter_parameterized_objs():
            assert isinstance(obj, Torchable), \
                f"TorchForwardSimulator requires every parameterized model member to be Torchable; " \
                f"{lbl} is a {type(obj)}, which isn't."
            sld = obj.stateless_data(dtype, device)
            param_data = (lbl, type(obj), sld)
            self.param_metadata.append(param_data)
            if isinstance(obj, (_Instrument, _TPInstrument)):
                self.instrument_expansions[lbl] = tuple(obj.simplify_operations(lbl).keys())

        self.params_dim = 0
        self.free_param_sizes = []
        self.default_to_reverse_ad = False
        # ^ Those are set in get_free_params
        return
    
    def get_free_params(self, model: ExplicitOpModel, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor]:
        """
        Return a tuple of Tensors that encode the states of the provided model's ModelMembers
        (where "state" in meant the sense of  object-oriented programming).
        
        We compare the labels of the input model's ModelMembers to those of the model provided
        to StatelessModelCircuitStore.__init__(...). We raise an error if an inconsistency is detected.
        """
        free_params = []
        free_param_sizes = []
        prev_idx = 0
        for i, (lbl, obj) in enumerate(model._iter_parameterized_objs()):
            gpind = obj.gpindices_as_array()
            vec = obj.to_vector()
            vec_size = vec.size
            vec = torch.from_numpy(vec)
            if vec_size > 0:
                assert gpind[0] == prev_idx and gpind[-1] == prev_idx + vec_size - 1
                # ^ We should have gpind = (prev_idx, prev_idx + 1, ..., prev_idx + vec.size - 1).
                #   That assert checks a cheap necessary condition that this holds.
            # ^ A 0-parameter member (e.g. ComputationalBasisPOVM) has an empty gpindices array, so
            #   there is nothing to check; we still append its empty tensor below so that free_params
            #   stays aligned with param_metadata.
            prev_idx += vec_size
            if self.param_metadata[i][0] != lbl:
                message = """
                The structure of the model's parameterizered objects provided to `get_free_params`
                differs from that of the model passed to `__init__`.
                """
                raise ValueError(message)
            vec = vec.to(dtype=dtype, device=device)
            free_params.append(vec)
            free_param_sizes.append(vec_size)
        self.params_dim = prev_idx
        self.free_param_sizes = free_param_sizes
        self.default_to_reverse_ad = self.outcome_probs_dim < self.params_dim
        return tuple(free_params)

    def get_torch_bases(self, free_params: Tuple[torch.Tensor]) -> Dict[Label, torch.Tensor]:
        """
        Take data of the kind produced by get_free_params and format it in the way required by
        circuit_probs_from_torch_bases.

        Note
        ----
        If you want to use the returned dict to build a PyTorch Tensor that supports the 
        .backward() method, then you need to make sure that fp.requires_grad is True for all
        fp in free_params. This can be done by calling fp.requires_grad_(True) before calling
        this function.
        """
        assert len(free_params) == len(self.param_metadata)
         # ^ A sanity check that we're being called with the correct number of arguments.
        torch_bases = dict()
        for i, val in enumerate(free_params):

            label, type_handle, stateless_data = self.param_metadata[i]
            param_t = type_handle.torch_base(stateless_data, val)
            torch_bases[label] = param_t

            # An instrument's torch_base is a stacked (n_members, d^2, d^2) tensor.  Expose each member
            # under its expanded op label (Iz_plus, Iz_minus, ...) so circuit_probs_from_torch_bases can
            # look them up unchanged; slicing the stacked tensor keeps autograd flowing to the params.
            expansions = self.instrument_expansions.get(label)
            if expansions is not None:
                for j, expanded_lbl in enumerate(expansions):
                    torch_bases[expanded_lbl] = param_t[j]

        return torch_bases

    @property
    def eval_plan(self) -> TorchEvalPlan:
        """
        The batched :class:`TorchEvalPlan` for this store's circuits, built on first use.

        The plan depends only on the *structure* of the (model, layout) pair -- circuit contents,
        which modelmembers are parameterized, and how many parameters each has -- so one plan
        serves every parameter value a GST optimization visits.
        """
        if self._eval_plan is None:
            self._eval_plan = TorchEvalPlan(
                self.circuits, self.param_metadata, self.instrument_expansions,
                self.povm_effect_counts, self.outcome_probs_dim, self.dtype, self.device
            )
        return self._eval_plan

    def circuit_probs_from_torch_bases(self, torch_bases: Dict[Label, torch.Tensor]) -> torch.Tensor:
        """
        Compute the circuit outcome probabilities that result when all of this StatelessModelCircuitStore's
        StatelessCircuits are run with data in torch_bases.

        Return the results as a single (vectorized) torch Tensor.
        """
        return self.eval_plan.probs(torch_bases, self.model_dim)

    def circuit_jacobian_from_free_params(self, free_params: Tuple[torch.Tensor],
                                          probs_out=None) -> torch.Tensor:
        """
        The Jacobian of :meth:`circuit_probs_from_free_params` at `free_params`, as a dense
        ``(outcome_probs_dim, params_dim)`` Tensor.

        Unlike ``torch.func.jacfwd(circuit_probs_from_free_params)``, this does not differentiate
        through the circuit-evaluation loop. It differentiates each modelmember's
        ``torch_base`` map on its own (a small, per-member autodiff problem) and applies the chain
        rule with an explicitly-derived expression for the derivative of a circuit's outcome
        probabilities with respect to the dense superoperators. See
        :mod:`pygsti.forwardsims.torchfwdsim_plan`.
        """
        num_params = sum(int(fp.numel()) for fp in free_params)
        return self.eval_plan.jacobian(free_params, self.model_dim, num_params,
                                       probs_out=probs_out)


    def circuit_probs_from_free_params(self, *free_params: Tuple[torch.Tensor], enable_backward=False) -> torch.Tensor:
        """
        This is the basic function we expose to pytorch for automatic differentiation. It returns the circuit
        outcome probabilities resulting when the states of ModelMembers associated with this StatelessModelCircuitStore
        are set based on free_params. 

        If you want to call PyTorch's .backward() on the returned Tensor (or a function of that Tensor), then
        you should set enable_backward=True. Keep the default value of enable_backward=False in all other
        situations, including when using PyTorch's jacrev function.
        """
        if enable_backward:
            for fp in free_params:
                fp.requires_grad_(True)

        torch_bases = self.get_torch_bases(free_params) # type: ignore
        probs = self.circuit_probs_from_torch_bases(torch_bases)
        return probs


def get_device() -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return 'cuda:0'  # NVIDIA and AMD
    elif torch.mps.is_available() and torch.mps.device_count() > 0:
        return 'mps:0'  # Apple
    elif torch.xpu.is_available() and torch.xpu.device_count() > 0:
        return 'xpu:0'  # Intel
    elif hasattr(torch, "tpu") and torch.tpu.is_available() and torch.tpu.device_count() > 0:
        return 'tpu:0'  # Google TPU (Modern: TorchTPU)
    else:
        try:  # Google TPU (Legacy: PyTorch/XLA fallback)
            import torch_xla.core.xla_model as xm  # type: ignore
            if xm.xla_device() is not None:
                return 'xla:0'
        except (ImportError, RuntimeError):
            pass
    return 'cpu'


class TorchForwardSimulator(ForwardSimulator):
    """
    A forward simulator that leverages automatic differentiation in PyTorch.
    """

    ENABLED = TORCH_ENABLED
    model   : ExplicitOpModel | None
    device  : torch.device
    dtype   : torch.dtype

    def __init__(self, model : Optional[ExplicitOpModel] = None, use_gpu: Optional[bool] = None,
                 dtype: Optional[torch.dtype] = None, device: torch.device | str = 'auto'):
        """
        Parameters
        ----------
        model : ExplicitOpModel, optional
            The model this simulator will use to compute circuit outcome probabilities.

        use_gpu : bool, optional
            If True, require a GPU: run on the detected GPU, raising a ValueError if none is
            found. If False, run on the CPU even when a GPU is available. If None (the default),
            run on a detected CUDA or XPU device when present and otherwise on the CPU; an Apple
            MPS device is deliberately not used by default, since MPS only supports float32.
            Ignored (must be left as None) if `device` is given explicitly.

        dtype : torch.dtype, optional
            The real dtype used for free parameters and all baked constants. If None, defaults
            to `DEFAULT_REAL_TYPE` (`torch.float64`, matching the precision of pyGSTi's other
            forward simulators), except on an MPS device where float64 is unsupported and
            float32 is used instead.

        device : torch.device or str
            If 'auto', then the device is chosen from `use_gpu` as documented above. Otherwise,
            this must be a torch.device or a string that can be passed to the torch.device
            constructor. Examples of the latter include 'cuda:0' and 'cpu'.
        """
        if not self.ENABLED:
            raise RuntimeError('PyTorch could not be imported.')
        self.model = model

        if device == 'auto':
            device = get_device()
            if use_gpu and 'cpu' in device:
                raise ValueError('No GPU detected.')
            if use_gpu is None and 'mps' in device:
                device = 'cpu'   # we override per our discretion (MPS lacks float64 support)
            if use_gpu is False:
                device = 'cpu'   # user declines a detected GPU
            device = torch.device(device)
        elif use_gpu is not None:
            raise ValueError('Specify at most one of `device` and `use_gpu`.')

        if dtype is None:
            dtype = torch.float32 if 'mps' in str(device) else DEFAULT_REAL_TYPE
            # ^ float32 is the only real floating point dtype that MPS supports

        self.dtype  = dtype
        self.device = device
        self._store_cache = None
        super(ForwardSimulator, self).__init__(model)

    def _circuit_store(self, layout) -> StatelessModelCircuitStore:
        """
        The StatelessModelCircuitStore for `layout`, reused across calls.

        Building the store walks every circuit in Python and bakes each modelmember's stateless
        data; building its evaluation plan walks every gate application. Neither depends on the
        model's parameter *values*, so a GST optimization -- which calls this simulator once or
        twice per iteration against a fixed layout -- should pay for them once, not once per
        iteration. The cache holds weak references so it can't keep a layout or model alive, and
        is keyed by object identity: a structurally different layout or model is a different
        object, and `get_free_params` re-checks the modelmember labels on every call.
        """
        cached = self._store_cache
        if cached is not None:
            layout_ref, model_ref, store = cached
            if layout_ref() is layout and model_ref() is self.model:
                return store
        store = StatelessModelCircuitStore(self.model, layout, self.dtype, self.device)
        try:
            self._store_cache = (_weakref.ref(layout), _weakref.ref(self.model), store)
        except TypeError:  # pragma: no cover - a layout or model that can't be weak-referenced
            self._store_cache = None
        return store

    def _bulk_fill_probs(self, array_to_fill, layout, split_model = None) -> None:
        assert self.model is not None
        if split_model is None:
            smcs = self._circuit_store(layout)
            free_params = smcs.get_free_params(self.model, self.dtype, self.device)
            torch_bases = smcs.get_torch_bases(free_params)
        else:
            smcs, torch_bases = split_model

        probs = smcs.circuit_probs_from_torch_bases(torch_bases)
        array_to_fill[:smcs.outcome_probs_dim] = probs.cpu().detach().numpy().ravel()
        return

    def _bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill) -> None:
        assert self.model is not None
        smcs = self._circuit_store(layout)
        free_params = smcs.get_free_params(self.model, self.dtype, self.device)

        probs_out = None
        if pr_array_to_fill is not None:
            probs_out = torch.empty(smcs.outcome_probs_dim, dtype=self.dtype, device=self.device)

        J_val = smcs.circuit_jacobian_from_free_params(free_params, probs_out=probs_out)
        array_to_fill[:] = J_val.cpu().numpy()
        if probs_out is not None:
            pr_array_to_fill[:smcs.outcome_probs_dim] = probs_out.cpu().numpy().ravel()
        return
