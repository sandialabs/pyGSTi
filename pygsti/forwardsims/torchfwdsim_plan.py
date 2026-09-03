#***************************************************************************************************
# Copyright 2026, National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
"""
A batched evaluation plan for :class:`TorchForwardSimulator`.

Motivation
----------
The straightforward way to evaluate a list of circuits with PyTorch -- loop over circuits, loop
over layers, ``superket = superop @ superket`` -- issues one GPU kernel launch per gate
application. A GST circuit list for a six-level system has on the order of 10^5 gate applications,
so a single probability evaluation costs 10^5 launches of a matrix-vector product on a 36x36
matrix. That is entirely launch-bound: the GPU is idle almost all of the time. Differentiating
that loop with ``torch.func.jacfwd``/``jacrev`` keeps the launch count and multiplies the working
set by the number of parameters (or outcomes), which is worse still.

This module replaces that loop with a plan that is compiled once per (circuit list, model
structure) and then replayed cheaply. It rests on two observations.

**1. Circuits share prefixes and suffixes.** A GST circuit is
``prep_fiducial + germ^p + meas_fiducial``, so the list has enormous redundancy. We build a prefix
DAG over the op-label sequences and evaluate it breadth-first: all nodes at tree depth ``t`` that
apply the same gate are updated by a *single* dense matrix-matrix product. The number of kernel
launches drops from "one per gate application" to "one per (tree depth, gate label) pair" -- from
10^5 to 10^2 -- and the total flop count drops by the prefix-sharing factor. The same is done with
a suffix DAG, evaluated from the POVM backwards.

**2. The Jacobian is a sum of rank-one terms in the dense superoperators.** For a circuit
``c = (g_1, ..., g_L)`` with prep ``rho`` and POVM ``E``, write

    R_k = M_{g_k} ... M_{g_1} rho          (a prefix-DAG node state)
    L_k = E M_{g_L} ... M_{g_{k+1}}        (a suffix-DAG node state)

so that ``p_c = L_k M_{g_k} R_{k-1}`` for every k. Then

    d p_c[e] / d M_g [a,b] = sum_{k : g_k = g} L_k[e,a] R_{k-1}[b] =: A_g[(c,e), a, b]

and the chain rule through the *parameterization* gives the Jacobian block for gate ``g``:

    J[(c,e), theta_g] = A_g[(c,e), :, :] . (d M_g / d theta_g)

The second factor is a small per-modelmember object of shape ``(d^2, n_params_g)`` that we get
from ``torch.func.jacrev`` on ``Torchable.torch_base`` -- so PyTorch still does all the
differentiation of the parameterization, which is the whole point of the Torch simulator. The
first factor is built by one scatter-add of outer products, and the contraction is one large
matrix-matrix product. Both map perfectly onto a GPU.

The result is a Jacobian computed in a few hundred kernel launches instead of ~10^5 * n_params.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    import torch
    from pygsti.baseobjs.label import Label

try:
    import torch
except ImportError:
    pass

import numpy as _np


# Cap on the temporary that holds a batch of outer products while a scatter-add accumulates the
# "A" tensor.  Chosen so the temporary stays well inside a GPU's memory even for large d.
_OUTER_PRODUCT_CHUNK_BYTES = 1 << 28  # 256 MiB


class _SequenceDag:
    """
    A DAG over sequences of op labels: one node per distinct sequence that occurs as a prefix (or,
    for the reversed use, a suffix) of some circuit, plus one root per distinct "seed" key.

    Nodes are numbered so that a node's parent always has a smaller index and so that all nodes
    sharing a (depth, op-label) pair are *contiguous*. That lets the evaluation schedule write its
    results into a contiguous slice of the state tensor -- the source indices still need a gather,
    but the destination never does.
    """

    def __init__(self):
        self._children: List[dict] = []
        self._parent: List[int] = []
        self._op: List[Any] = []
        self._depth: List[int] = []
        self.root_keys: List[Any] = []
        self._finalized = False

    # -- construction -------------------------------------------------------------------------

    def root(self, key) -> int:
        """Index of the root node seeded by `key`, creating it if needed."""
        for i, k in enumerate(self.root_keys):
            if k == key:
                return i
        assert not self._children or self._depth[-1] == 0, \
            "all roots must be created before any interior node"
        idx = len(self._parent)
        self.root_keys.append(key)
        self._children.append({})
        self._parent.append(-1)
        self._op.append(None)
        self._depth.append(0)
        return idx

    def walk(self, root_idx: int, ops: Sequence) -> List[int]:
        """
        Insert `ops` under `root_idx` and return the node index after each step, i.e. a list of
        ``len(ops) + 1`` indices starting with `root_idx`.
        """
        cur = root_idx
        path = [cur]
        for op in ops:
            nxt = self._children[cur].get(op)
            if nxt is None:
                nxt = len(self._parent)
                self._children[cur][op] = nxt
                self._children.append({})
                self._parent.append(cur)
                self._op.append(op)
                self._depth.append(self._depth[cur] + 1)
            cur = nxt
            path.append(cur)
        return path

    # -- finalization -------------------------------------------------------------------------

    def finalize(self, op_ids: Dict[Any, int], device) -> _np.ndarray:
        """
        Renumber nodes into (depth, op-label) order and build the evaluation schedule.

        Returns the ``old index -> new index`` map so the caller can translate the node indices it
        collected during construction.
        """
        assert not self._finalized
        n = len(self._parent)
        depth = _np.asarray(self._depth)
        opid = _np.array([-1 if o is None else op_ids[o] for o in self._op])
        # lexsort by (depth, op id) -- stable, so ties keep insertion order.
        order = _np.lexsort((_np.arange(n), opid, depth))
        new_of_old = _np.empty(n, dtype=_np.int64)
        new_of_old[order] = _np.arange(n, dtype=_np.int64)

        parent = _np.asarray(self._parent)
        n_roots = len(self.root_keys)
        assert _np.all(order[:n_roots] == _np.arange(n_roots)), "roots must sort to the front"

        # schedule: contiguous runs of equal (depth, op id) among the non-root nodes
        self.schedule: List[Tuple[Any, int, int, "torch.Tensor"]] = []
        keys = list(zip(depth[order], opid[order]))
        inv_op = {v: k for k, v in op_ids.items()}
        start = n_roots
        while start < n:
            stop = start + 1
            while stop < n and keys[stop] == keys[start]:
                stop += 1
            src_old = parent[order[start:stop]]
            src_new = new_of_old[src_old]
            self.schedule.append((
                inv_op[keys[start][1]], start, stop,
                torch.as_tensor(src_new, dtype=torch.long, device=device),
            ))
            start = stop

        self.num_nodes = n
        self.num_roots = n_roots
        self._finalized = True
        # drop construction-only state
        self._children = []
        self._parent = []
        self._op = []
        self._depth = []
        return new_of_old


def _member_value_and_jacobian(type_handle, stateless_data, param_vec):
    """
    ``(torch_base(sd, v), d torch_base / d v)``. The Jacobian has shape ``out_shape + (n_v,)``;
    it is the *only* place autodiff is used, and it runs on a single modelmember's parameters, so
    it is cheap regardless of how many circuits are being simulated.
    """
    def f(v):
        return type_handle.torch_base(stateless_data, v)

    out = f(param_vec)
    n_v = param_vec.numel()
    if n_v == 0:
        return out, out.new_zeros(tuple(out.shape) + (0,))
    if out.numel() <= n_v:
        jac = torch.func.jacrev(f)(param_vec)
    else:
        jac = torch.func.jacfwd(f)(param_vec)
    return out, jac


class _PovmGroup:
    """
    The plan for the subset of circuits that share a single POVM label.

    Within a group every circuit produces the same number of "dense" outcome rows -- the POVM's
    full effect count -- even if the layout only asks for a subset of them. Keeping the dense
    width uniform is what lets the whole group be one set of rectangular tensors; the (rare)
    subsetting is applied by a row gather when results are written out.
    """

    def __init__(self, povm_label, n_eff: int):
        self.povm_label = povm_label
        self.n_eff = n_eff
        self.fwd = _SequenceDag()
        self.bwd = _SequenceDag()
        self._circ_end_fwd: List[int] = []   # prefix node for the whole circuit  -> R_L
        self._circ_end_bwd: List[int] = []   # suffix node for the whole circuit  -> L_0
        self._prep_of_circuit: List[Any] = []
        self._positions: Dict[Any, List[Tuple[int, int, int]]] = {}  # op -> [(circ, lnode, rnode)]
        self._dense_rows: List[int] = []
        self._out_rows: List[int] = []
        self.num_circuits = 0

    def add_circuit(self, prep_label, op_labels, effect_rows: Sequence[int], elem_offset: int):
        ic = self.num_circuits
        self.num_circuits += 1
        self._prep_of_circuit.append(prep_label)

        fwd_path = self.fwd.walk(self.fwd.root(prep_label), op_labels)
        bwd_path = self.bwd.walk(self.bwd.root(self.povm_label), tuple(reversed(op_labels)))
        # bwd_path[j] is the suffix consisting of the last j ops, i.e. L_{L-j}.
        self._circ_end_fwd.append(fwd_path[-1])
        self._circ_end_bwd.append(bwd_path[-1])

        n_ops = len(op_labels)
        for k in range(1, n_ops + 1):          # position k applies op_labels[k-1]
            self._positions.setdefault(op_labels[k - 1], []).append(
                (ic, bwd_path[n_ops - k], fwd_path[k - 1])
            )

        for j, e in enumerate(effect_rows):
            self._dense_rows.append(ic * self.n_eff + e)
            self._out_rows.append(elem_offset + j)

    def finalize(self, op_ids, device):
        fwd_map = self.fwd.finalize(op_ids, device)
        bwd_map = self.bwd.finalize(op_ids, device)

        def _t(a, dtype=torch.long):
            return torch.as_tensor(_np.asarray(a), dtype=dtype, device=device)

        self.circ_end_fwd = _t(fwd_map[_np.asarray(self._circ_end_fwd, dtype=_np.int64)])
        self.circ_end_bwd = _t(bwd_map[_np.asarray(self._circ_end_bwd, dtype=_np.int64)])

        self.positions = {}
        for op, plist in self._positions.items():
            arr = _np.asarray(plist, dtype=_np.int64)
            self.positions[op] = (_t(arr[:, 0]), _t(bwd_map[arr[:, 1]]), _t(fwd_map[arr[:, 2]]))

        # circuits grouped by prep label (almost always a single group)
        preps = {}
        for ic, p in enumerate(self._prep_of_circuit):
            preps.setdefault(p, []).append(ic)
        self.circuits_by_prep = {p: _t(v) for p, v in preps.items()}
        self.all_circuits_share_one_prep = len(preps) == 1

        dense = _np.asarray(self._dense_rows, dtype=_np.int64)
        out = _np.asarray(self._out_rows, dtype=_np.int64)
        # Fast path: the group's dense rows are exactly its output rows, contiguous from
        # out[0].  Then a whole column block can be written with a plain slice assignment.
        self.contiguous_out = (
            dense.size == self.num_circuits * self.n_eff
            and _np.array_equal(dense, _np.arange(dense.size))
            and _np.array_equal(out, out[0] + _np.arange(out.size))
        ) if dense.size else False
        self.out_slice = slice(int(out[0]), int(out[0]) + out.size) if self.contiguous_out else None
        self.dense_rows = _t(dense)
        self.out_rows = _t(out)
        self.n_dense = self.num_circuits * self.n_eff

        self._circ_end_fwd = self._circ_end_bwd = self._prep_of_circuit = None
        self._positions = self._dense_rows = self._out_rows = None

    # -- evaluation ---------------------------------------------------------------------------

    def forward_states(self, torch_bases, dim, dtype, device):
        """All prefix-DAG node states, shape ``(num_nodes, dim)``."""
        dag = self.fwd
        states = torch.empty((dag.num_nodes, dim), dtype=dtype, device=device)
        for i, prep_label in enumerate(dag.root_keys):
            states[i] = torch_bases[prep_label]
        for op, lo, hi, src in dag.schedule:
            # (m, dim) @ (dim, dim) computes  M @ r  for each row r, since (M r)^T = r^T M^T.
            states[lo:hi] = states.index_select(0, src) @ torch_bases[op].transpose(-2, -1)
        return states

    def backward_states(self, torch_bases, dim, dtype, device, povm_mat):
        """All suffix-DAG node states, shape ``(num_nodes, n_eff, dim)``."""
        dag = self.bwd
        n_eff = self.n_eff
        states = torch.empty((dag.num_nodes, n_eff, dim), dtype=dtype, device=device)
        states[0] = povm_mat
        for op, lo, hi, src in dag.schedule:
            sub = states.index_select(0, src)                      # (m, n_eff, dim)
            states[lo:hi] = (sub.reshape(-1, dim) @ torch_bases[op]).view(-1, n_eff, dim)
        return states


class TorchEvalPlan:
    """
    A compiled, batched evaluation plan for a list of :class:`StatelessCircuit` objects.

    Build it once per (circuit list, model structure); it is independent of the model's *parameter
    values*, so it can be replayed for every iteration of a GST optimization.
    """

    def __init__(self, circuits, param_metadata, instrument_expansions, povm_effect_counts,
                 outcome_probs_dim, dtype, device):
        self.dtype = dtype
        self.device = device
        self.outcome_probs_dim = outcome_probs_dim

        # A stable integer id per op label, used only to sort DAG nodes.
        op_ids: Dict[Any, int] = {}
        for c in circuits:
            for ol in c.op_labels:
                op_ids.setdefault(ol, len(op_ids))

        groups: Dict[Any, _PovmGroup] = {}
        elem_offset = 0
        for c in circuits:
            g = groups.get(c.povm_label)
            if g is None:
                g = groups[c.povm_label] = _PovmGroup(c.povm_label, povm_effect_counts[c.povm_label])
            rows = (c.effect_row_indices if c.effect_row_indices is not None
                    else range(g.n_eff))
            g.add_circuit(c.prep_label, tuple(c.op_labels), rows, elem_offset)
            elem_offset += c.outcome_probs_dim
        assert elem_offset == outcome_probs_dim
        for g in groups.values():
            g.finalize(op_ids, device)
        self.groups = list(groups.values())

        # Column block (into the parameter vector) of each parameterized modelmember, plus the
        # op labels its dense matrices appear under in a circuit.
        self.param_metadata = param_metadata
        self.instrument_expansions = instrument_expansions
        self.op_labels = list(op_ids)

    # -- diagnostics --------------------------------------------------------------------------

    def stats(self):
        gate_applications = sum(
            int(idx[0].numel()) for g in self.groups for idx in g.positions.values())
        return dict(
            num_groups=len(self.groups),
            num_circuits=sum(g.num_circuits for g in self.groups),
            gate_applications=gate_applications,
            prefix_nodes=sum(g.fwd.num_nodes for g in self.groups),
            suffix_nodes=sum(g.bwd.num_nodes for g in self.groups),
            forward_matmuls=sum(len(g.fwd.schedule) for g in self.groups),
            backward_matmuls=sum(len(g.bwd.schedule) for g in self.groups),
        )

    # -- probabilities ------------------------------------------------------------------------

    def probs(self, torch_bases, dim, out=None):
        """Outcome probabilities for every circuit, in layout element order."""
        if out is None:
            out = torch.empty(self.outcome_probs_dim, dtype=self.dtype, device=self.device)
        for g in self.groups:
            fwd = g.forward_states(torch_bases, dim, self.dtype, self.device)
            povm_mat = torch_bases[g.povm_label]
            dense = (fwd.index_select(0, g.circ_end_fwd) @ povm_mat.transpose(-2, -1)).reshape(-1)
            if g.contiguous_out:
                out[g.out_slice] = dense
            else:
                out[g.out_rows] = dense[g.dense_rows]
        return out

    # -- Jacobian -----------------------------------------------------------------------------

    def jacobian(self, free_params, dim, num_params, out=None, probs_out=None):
        """
        The ``(num_elements, num_params)`` Jacobian of the outcome probabilities.

        `free_params` is the per-modelmember tuple produced by
        ``StatelessModelCircuitStore.get_free_params``. The dense superoperators and their
        derivatives with respect to the modelmember's own parameters are computed here (by
        autodiff, per modelmember); everything downstream is explicit linear algebra.
        """
        dtype, device = self.dtype, self.device
        torch_bases: Dict[Any, "torch.Tensor"] = {}
        derivs: Dict[Any, Tuple[Any, "torch.Tensor", slice]] = {}
        col = 0
        for i, val in enumerate(free_params):
            label, type_handle, sld = self.param_metadata[i]
            base, jac = _member_value_and_jacobian(type_handle, sld, val)
            torch_bases[label] = base
            cols = slice(col, col + val.numel())
            col += val.numel()
            derivs[label] = (base, jac, cols)
            expansions = self.instrument_expansions.get(label)
            if expansions is not None:
                for j, expanded in enumerate(expansions):
                    torch_bases[expanded] = base[j]
                    derivs[expanded] = (base[j], jac[j], cols)
        assert col == num_params, (col, num_params)

        if out is None:
            out = torch.zeros((self.outcome_probs_dim, num_params), dtype=dtype, device=device)
        else:
            out.zero_()

        # Which member-label a circuit op label draws its columns from (identity, except for
        # instrument members, several of which share one member's column block).
        member_of_op = {}
        for label, _, _ in self.param_metadata:
            expansions = self.instrument_expansions.get(label)
            if expansions is None:
                member_of_op.setdefault(label, label)
            else:
                for expanded in expansions:
                    member_of_op[expanded] = label

        for g in self.groups:
            self._jacobian_group(g, torch_bases, derivs, member_of_op, dim, out, probs_out)
        return out

    def _jacobian_group(self, g, torch_bases, derivs, member_of_op, dim, out, probs_out):
        dtype, device = self.dtype, self.device
        n_eff, n_c, n_dense = g.n_eff, g.num_circuits, g.n_dense
        d2 = dim * dim

        povm_mat = torch_bases[g.povm_label]
        fwd = g.forward_states(torch_bases, dim, dtype, device)
        bwd = g.backward_states(torch_bases, dim, dtype, device, povm_mat)

        r_end = fwd.index_select(0, g.circ_end_fwd)          # (n_c, dim)      R_L
        l_end = bwd.index_select(0, g.circ_end_bwd)          # (n_c, n_eff, d) L_0

        if probs_out is not None:
            dense = (r_end @ povm_mat.transpose(-2, -1)).reshape(-1)
            if g.contiguous_out:
                probs_out[g.out_slice] = dense
            else:
                probs_out[g.out_rows] = dense[g.dense_rows]

        def write(cols, block):
            """Write a (n_dense, n_cols) block into the right rows/columns of `out`."""
            if g.contiguous_out:
                out[g.out_slice, cols] = block
            else:
                out[g.out_rows, cols] = block.index_select(0, g.dense_rows)

        # --- gate blocks: A_g . (dM_g / dtheta) ------------------------------------------------
        # Instrument members share a column block, so accumulate per member label first.
        by_member: Dict[Any, List[Any]] = {}
        for op in g.positions:
            by_member.setdefault(member_of_op[op], []).append(op)

        eff_arange = torch.arange(n_eff, dtype=torch.long, device=device)
        for member, ops in by_member.items():
            cols = derivs[member][2]
            n_v = cols.stop - cols.start
            if n_v == 0:
                continue
            block = None
            for op in ops:
                ci, lnode, rnode = g.positions[op]
                a_mat = self._accumulate_a(g, fwd, bwd, ci, lnode, rnode, dim, eff_arange)
                contrib = a_mat @ derivs[op][1].reshape(d2, n_v)
                block = contrib if block is None else block + contrib
            write(cols, block)

        # --- prep block: L_0 . (d rho / d theta) -----------------------------------------------
        for prep_label, circ_idx in g.circuits_by_prep.items():
            _, jac, cols = derivs[prep_label]
            n_v = cols.stop - cols.start
            if n_v == 0:
                continue
            if g.all_circuits_share_one_prep:
                block = l_end.reshape(-1, dim) @ jac.reshape(dim, n_v)
                write(cols, block)
            else:
                sub = l_end.index_select(0, circ_idx).reshape(-1, dim) @ jac.reshape(dim, n_v)
                rows = (circ_idx[:, None] * n_eff + eff_arange).reshape(-1)
                dense_block = torch.zeros((n_dense, n_v), dtype=dtype, device=device)
                dense_block[rows] = sub
                write(cols, dense_block)

        # --- POVM block: R_L contracted with (d E / d theta) ------------------------------------
        _, jac, cols = derivs[g.povm_label]
        n_v = cols.stop - cols.start
        if n_v > 0:
            w = jac.reshape(n_eff, dim, n_v)
            block = torch.einsum('ca,ean->cen', r_end, w).reshape(n_dense, n_v)
            write(cols, block)

    def _accumulate_a(self, g, fwd, bwd, ci, lnode, rnode, dim, eff_arange):
        """
        ``A[(c,e), a, b] = sum_{positions p of this gate in circuit c} L_p[e,a] R_p[b]``,
        returned flattened to ``(n_dense, dim*dim)``.
        """
        dtype, device = self.dtype, self.device
        n_eff, n_dense = g.n_eff, g.n_dense
        d2 = dim * dim
        a_mat = torch.zeros((n_dense, d2), dtype=dtype, device=device)

        n_pos = int(ci.numel())
        per_pos_bytes = n_eff * d2 * torch.finfo(dtype).bits // 8
        chunk = max(1, min(n_pos, _OUTER_PRODUCT_CHUNK_BYTES // max(per_pos_bytes, 1)))
        for lo in range(0, n_pos, chunk):
            hi = min(lo + chunk, n_pos)
            l_sub = bwd.index_select(0, lnode[lo:hi])            # (m, n_eff, dim)
            r_sub = fwd.index_select(0, rnode[lo:hi])            # (m, dim)
            outer = (l_sub.unsqueeze(-1) * r_sub[:, None, None, :]).reshape(-1, d2)
            rows = (ci[lo:hi, None] * n_eff + eff_arange).reshape(-1)
            a_mat.index_add_(0, rows, outer)
        return a_mat
