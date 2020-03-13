""" Defines the OplessModel class"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
import collections as _collections

from .model import Model as _Model
from .evaltree import EvalTree as _EvalTree
from .labeldicts import OutcomeLabelDict as _OutcomeLabelDict
from .circuit import Circuit as _Circuit
from .polynomial import Polynomial as _Polynomial
from ..tools import slicetools as _slct

from .opcalc import compact_deriv as _compact_deriv, float_product as prod, \
    safe_bulk_eval_compact_polys as _safe_bulk_eval_compact_polys


class OplessModelTree(_EvalTree):
    def __init__(self, circuit_list, lookup, outcome_lookup, cache=None):
        _EvalTree.__init__(self, circuit_list)
        self.element_indices = lookup
        self.outcomes = outcome_lookup
        self.num_final_strs = len(circuit_list)  # circuits
        max_el_index = -1
        for elIndices in lookup.values():
            max_i = elIndices.stop - 1 if isinstance(elIndices, slice) else max(elIndices)
            max_el_index = max(max_el_index, max_i)
        self.num_final_els = max_el_index + 1
        self.cache = cache


class OplessModel(_Model):
    """
    TODO docstring
    """

    def __init__(self, state_space_labels):
        """
        Creates a new Model.  Rarely used except from derived classes
        `__init__` functions.

        Parameters
        ----------
        state_space_labels : StateSpaceLabels or list or tuple
            The decomposition (with labels) of (pure) state-space this model
            acts upon.  Regardless of whether the model contains operators or
            superoperators, this argument describes the Hilbert space dimension
            and imposed structure.  If a list or tuple is given, it must be
            of a from that can be passed to `StateSpaceLabels.__init__`.
        """
        _Model.__init__(self, state_space_labels)

        #Setting things the rest of pyGSTi expects but probably shouldn't...
        self.simtype = "opless"
        self.basis = None
        self.dim = 0

    def get_dimension(self):
        return self.dim

    def get_num_outcomes(self, circuit):  # needed for sparse data detection
        raise NotImplementedError("Derived classes should implement this!")

    def probs(self, circuit, clip_to=None, cache=None):
        """
        Construct a dictionary containing the probabilities of every spam label
        given a operation sequence.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels specifying the operation sequence.

        clip_to : 2-tuple, optional
           (min,max) to clip probabilities to if not None.

        Returns
        -------
        probs : dictionary
            A dictionary such that
            probs[SL] = pr(SL,circuit,clip_to)
            for each spam label (string) SL.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def dprobs(self, circuit, return_pr=False, clip_to=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given operation sequence.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels specifying the operation sequence.

        return_pr : bool, optional
          when set to True, additionally return the probabilities.

        clip_to : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when return_pr == True.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that
            dprobs[SL] = dpr(SL,circuit,gates,G0,SPAM,SP0,return_pr,clip_to)
            for each spam label (string) SL.
        """
        eps = 1e-7
        orig_pvec = self.to_vector()
        Np = self.num_params()
        probs0 = self.probs(circuit, clip_to, None)

        deriv = {k: _np.empty(Np, 'd') for k in probs0.keys()}
        for i in range(Np):
            p_plus_dp = orig_pvec.copy()
            p_plus_dp[i] += eps
            self.from_vector(p_plus_dp)
            probs1 = self.probs(circuit, clip_to, None)
            for k, p0 in probs0.items():
                deriv[k][i] = (probs1[k] - p0) / eps
        self.from_vector(orig_pvec)

        if return_pr:
            return {k: (p0, deriv[k]) for k in probs0.keys()}
        else:
            return deriv

    def bulk_evaltree_from_resources(self, circuit_list, comm=None, mem_limit=None,
                                     distribute_method="default", subcalls=[],
                                     dataset=None, verbosity=0):
        #TODO: choose these based on resources, and enable split trees
        minSubtrees = 0
        numSubtreeComms = 1
        maxTreeSize = None
        evTree = self.bulk_evaltree(circuit_list, minSubtrees, maxTreeSize,
                                    numSubtreeComms, dataset, verbosity)
        return evTree, 0, 0, evTree.element_indices, evTree.outcomes

    def bulk_evaltree(self, circuit_list, min_subtrees=None, max_tree_size=None,
                      num_subtree_comms=1, dataset=None, verbosity=0):
        raise NotImplementedError("Derived classes should implement this!")

    def bulk_probs(self, circuit_list, clip_to=None, check=False,
                   comm=None, mem_limit=None, dataset=None, smartc=None):
        evalTree, _, _, elIndices, outcomes = self.bulk_evaltree_from_resources(circuit_list, comm, mem_limit, "default",
                                                                                [], dataset)
        vp = _np.empty(evalTree.num_final_elements(), 'd')
        self.bulk_fill_probs(vp, evalTree, clip_to, check, comm)

        ret = _collections.OrderedDict()
        for i, opstr in enumerate(evalTree):
            elInds = _slct.indices(elIndices[i]) \
                if isinstance(elIndices[i], slice) else elIndices[i]
            ret[opstr] = _OutcomeLabelDict(
                [(outLbl, vp[ei]) for ei, outLbl in zip(elInds, outcomes[i])])
        return ret

    def bulk_dprobs(self, circuit_list, return_pr=False, clip_to=None,
                    check=False, comm=None, wrt_block_size=None, dataset=None):
        memLimit = None
        evalTree, _, _, elIndices, outcomes = self.bulk_evaltree_from_resources(circuit_list, comm, memLimit,
                                                                                "default", [], dataset)
        nElements = evalTree.num_final_elements()
        nDerivCols = self.num_params()

        vdp = _np.empty((nElements, nDerivCols), 'd')
        vp = _np.empty(nElements, 'd') if return_pr else None

        self.bulk_fill_dprobs(vdp, evalTree,
                              vp, clip_to, check, comm,
                              None, wrt_block_size)

        ret = _collections.OrderedDict()
        for i, opstr in enumerate(evalTree):
            elInds = _slct.indices(elIndices[i]) \
                if isinstance(elIndices[i], slice) else elIndices[i]
            if return_pr:
                ret[opstr] = _OutcomeLabelDict(
                    [(outLbl, (vdp[ei], vp[ei])) for ei, outLbl in zip(elInds, outcomes[i])])
            else:
                ret[opstr] = _OutcomeLabelDict(
                    [(outLbl, vdp[ei]) for ei, outLbl in zip(elInds, outcomes[i])])
        return ret

    def bulk_fill_probs(self, mx_to_fill, eval_tree, clip_to=None, check=False, comm=None):
        if False and eval_tree.cache:  # TEST (disabled)
            cpolys = eval_tree.cache
            ps = _safe_bulk_eval_compact_polys(cpolys[0], cpolys[1], self._paramvec, (eval_tree.num_final_elements(),))
            assert(_np.linalg.norm(_np.imag(ps)) < 1e-6)
            ps = _np.real(ps)
            if clip_to is not None: ps = _np.clip(ps, clip_to[0], clip_to[1])
            mx_to_fill[:] = ps
        else:
            for i, c in enumerate(eval_tree):
                cache = eval_tree.cache[i] if eval_tree.cache else None
                probs = self.probs(c, clip_to, cache)
                elInds = _slct.indices(eval_tree.element_indices[i]) \
                    if isinstance(eval_tree.element_indices[i], slice) else eval_tree.element_indices[i]
                for k, outcome in zip(elInds, eval_tree.outcomes[i]):
                    mx_to_fill[k] = probs[outcome]

    def bulk_fill_dprobs(self, mx_to_fill, eval_tree, pr_mx_to_fill=None, clip_to=None,
                         check=False, comm=None, wrt_block_size=None,
                         profiler=None, gather_mem_limit=None):

        Np = self.num_params()
        p = self.to_vector()

        if False and eval_tree.cache:  # TEST (disabled)
            cpolys = eval_tree.cache
            if pr_mx_to_fill is not None:
                ps = _safe_bulk_eval_compact_polys(cpolys[0], cpolys[1], p, (eval_tree.num_final_elements(),))
                assert(_np.linalg.norm(_np.imag(ps)) < 1e-6)
                ps = _np.real(ps)
                if clip_to is not None: ps = _np.clip(ps, clip_to[0], clip_to[1])
                pr_mx_to_fill[:] = ps
            dpolys = _compact_deriv(cpolys[0], cpolys[1], list(range(Np)))
            dps = _safe_bulk_eval_compact_polys(dpolys[0], dpolys[1], p, (eval_tree.num_final_elements(), Np))
            mx_to_fill[:, :] = dps
        else:
            # eps = 1e-6
            for i, c in enumerate(eval_tree):
                cache = eval_tree.cache[i] if eval_tree.cache else None
                probs0 = self.probs(c, clip_to, cache)
                dprobs0 = self.dprobs(c, False, clip_to, cache)
                elInds = _slct.indices(eval_tree.element_indices[i]) \
                    if isinstance(eval_tree.element_indices[i], slice) else eval_tree.element_indices[i]
                for k, outcome in zip(elInds, eval_tree.outcomes[i]):
                    if pr_mx_to_fill is not None:
                        pr_mx_to_fill[k] = probs0[outcome]
                    mx_to_fill[k, :] = dprobs0[outcome]

                    #Do this to fill mx_to_fill instead of calling dprobs above as it's a little faster for finite diff?
                    #for j in range(np):
                    #    p_plus_dp = p.copy()
                    #    p_plus_dp[j] += eps
                    #    self.from_vector(p_plus_dp)
                    #    probs1 = self.probs(c,clip_to,cache)
                    #    mx_to_fill[k,j] = (probs1[outcome]-probs0[outcome]) / eps
                    #self.from_vector(p)

    def __str__(self):
        raise "Derived classes should implement OplessModel.__str__ !!"


class SuccessFailModel(OplessModel):
    def __init__(self, state_space_labels, use_cache=False):
        OplessModel.__init__(self, state_space_labels)
        self.use_cache = use_cache

    def get_num_outcomes(self, circuit):  # needed for sparse data detection
        return 2

    def _success_prob(self, circuit, cache):
        raise NotImplementedError("Derived classes should implement this!")

    def _success_dprob(self, circuit, cache):
        raise NotImplementedError("Derived classes should implement this!")

    #FUTURE?: def _fill_circuit_probs(self, array_to_fill, outcomes, circuit, clip_to):
    def probs(self, circuit, clip_to=None, cache=None):
        """
        Construct a dictionary containing the probabilities of every spam label
        given a operation sequence.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels specifying the operation sequence.

        clip_to : 2-tuple, optional
           (min,max) to clip probabilities to if not None.

        Returns
        -------
        probs : dictionary
            A dictionary such that
            probs[outcome] = pr(outcome,circuit,clip_to).
        """
        sp = self._success_prob(circuit, cache)
        if clip_to is not None: sp = _np.clip(sp, clip_to[0], clip_to[1])
        return _OutcomeLabelDict([('success', sp), ('fail', 1 - sp)])

    def dprobs(self, circuit, return_pr=False, clip_to=None, cache=None):
        """
        Construct a dictionary containing the probability derivatives of every
        spam label for a given operation sequence.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
          The sequence of operation labels specifying the operation sequence.

        return_pr : bool, optional
          when set to True, additionally return the probabilities.

        clip_to : 2-tuple, optional
           (min,max) to clip returned probability to if not None.
           Only relevant when return_pr == True.

        Returns
        -------
        dprobs : dictionary
            A dictionary such that
            dprobs[SL] = dpr(SL,circuit,gates,G0,SPAM,SP0,return_pr,clip_to)
            for each spam label (string) SL.
        """
        try:
            dsp = self._success_dprob(circuit, cache)
        except NotImplementedError:
            return OplessModel.dprobs(self, circuit, return_pr, clip_to)

        if return_pr:
            sp = self._success_prob(circuit, cache)
            if clip_to is not None: sp = _np.clip(sp, clip_to[0], clip_to[1])
            return {('success',): (sp, dsp), ('fail',): (1 - sp, -dsp)}
        else:
            return {('success',): dsp, ('fail',): -dsp}

    def poly_probs(self, circuit):
        """
        Same as probs(...) but return polynomials.
        """
        sp = self._success_prob_poly(circuit)
        return _OutcomeLabelDict([('success', sp), ('fail', _Polynomial({(): 1.0}) - sp)])

    def simplify_circuits(self, circuits, dataset=None):
        rawdict = None  # TODO - is this needed?
        lookup = {i: slice(2 * i, 2 * i + 2, 1) for i in range(len(circuits))}
        outcome_lookup = {i: (('success',), ('fail',)) for i in range(len(circuits))}

        return rawdict, lookup, outcome_lookup, 2 * len(circuits)

    def bulk_evaltree(self, circuit_list, min_subtrees=None, max_tree_size=None,
                      num_subtree_comms=1, dataset=None, verbosity=0):
        lookup = {i: slice(2 * i, 2 * i + 2, 1) for i in range(len(circuit_list))}
        outcome_lookup = {i: (('success',), ('fail',)) for i in range(len(circuit_list))}

        if self.use_cache == "poly":
            #Do precomputation here
            polys = []
            for i, circuit in enumerate(circuit_list):
                print("Generating probs for circuit %d of %d" % (i + 1, len(circuit_list)))
                probs = self.poly_probs(circuit)
                polys.append(probs['success'])
                polys.append(probs['fail'])
            compact_polys = compact_poly_list(polys)
            cache = compact_polys
        elif self.use_cache is True:
            cache = [self._circuit_cache(circuit) for circuit in circuit_list]
        else:
            cache = None

        return OplessModelTree(circuit_list, lookup, outcome_lookup, cache)

#TODO: move this to polynomial.py??


def compact_poly_list(list_of_polys):
    """Create a single vtape,ctape pair from a list of normal Polynomals """
    tapes = [p.compact() for p in list_of_polys]
    vtape = _np.concatenate([t[0] for t in tapes])
    ctape = _np.concatenate([t[1] for t in tapes])
    return vtape, ctape


class ErrorRatesModel(SuccessFailModel):

    def __init__(self, error_rates, n_qubits, state_space_labels=None, alias_dict={}, idlename='Gi'):
        """
        todo
        """
        if state_space_labels is None:
            state_space_labels = ['Q%d' % i for i in range(n_qubits)]
        else:
            assert(len(state_space_labels) == n_qubits)

        SuccessFailModel.__init__(self, state_space_labels, use_cache=True)

        gate_error_rate_keys = (list(error_rates['gates'].keys()))
        readout_error_rate_keys = (list(error_rates['readout'].keys()))

        # if gate_error_rate_keys[0] in state_space_labels:
        #     self._gateind = True
        # else:
        #     self._gateind = False

        self._idlename = idlename
        self._alias_dict = alias_dict.copy()
        self._gate_error_rate_indices = {k: i for i, k in enumerate(gate_error_rate_keys)}
        self._readout_error_rate_indices = {k: i + len(gate_error_rate_keys)
                                            for i, k in enumerate(readout_error_rate_keys)}
        self._paramvec = _np.concatenate(
            (_np.array([_np.sqrt(error_rates['gates'][k]) for k in gate_error_rate_keys], 'd'),
             _np.array([_np.sqrt(error_rates['readout'][k]) for k in readout_error_rate_keys], 'd'))
        )

    def __str__(self):
        s = "Error Rates model with error rates: \n" + \
            "\n".join(["%s = %g" % (k, self._paramvec[i]**2) for k, i in self._gate_error_rate_indices.items()]) + \
            "\n" + \
            "\n".join(["%s = %g" % (k, self._paramvec[i]**2) for k, i in self._readout_error_rate_indices.items()])
        return s

    def to_dict(self):
        error_rate_dict = {'gates': {}, 'readout': {}}
        error_rate_dict['gates'] = {k: self._paramvec[i]**2 for k, i in self._gate_error_rate_indices.items()}
        error_rate_dict['readout'] = {k: self._paramvec[i]**2 for k, i in self._readout_error_rate_indices.items()}
        asdict = {'error_rates': error_rate_dict, 'alias_dict': self._alias_dict.copy()}
        return asdict

    def _circuit_cache(self, circuit):
        if not isinstance(circuit, _Circuit):
            circuit = _Circuit.fromtup(circuit)

        depth = circuit.depth()
        width = circuit.width()
        g_inds = self._gate_error_rate_indices
        r_inds = self._readout_error_rate_indices

        # if self._gateind:
        #     inds_to_mult_by_layer = []
        #     for i in range(depth):

        #         layer = circuit.get_layer(i)
        #         inds_to_mult = []
        #         usedQs = []

        #         for gate in layer:
        #             if len(gate.qubits) > 1:
        #                 usedQs += list(gate.qubits)
        #                 inds_to_mult.append(g_inds[frozenset(gate.qubits)])

        #         for q in circuit.line_labels:
        #             if q not in usedQs:
        #                 inds_to_mult.append(g_inds[q])

        #         inds_to_mult_by_layer.append(_np.array(inds_to_mult, int))

        # else:
        layers_with_idles = [circuit.get_layer_with_idles(i, idle_gate_name=self._idlename) for i in range(depth)]
        inds_to_mult_by_layer = [_np.array([g_inds[self._alias_dict.get(str(gate), str(gate))] for gate in layer], int)
                                 for layer in layers_with_idles]

        # Bit-flip readout error as a pre-measurement depolarizing channel.
        inds_to_mult = [r_inds[q] for q in circuit.line_labels]
        inds_to_mult_by_layer.append(_np.array(inds_to_mult, int))

        # The scaling constant such that lambda = 1 - alpha * epsilon where lambda is the diagonal of a depolarizing
        # channel with entanglement infidelity of epsilon.
        alpha = 4**width / (4**width - 1)

        return (width, depth, alpha, 1 / 2**width, inds_to_mult_by_layer)


class TwirledLayersModel(ErrorRatesModel):

    def __init__(self, error_rates, n_qubits, state_space_labels=None, alias_dict={}, idlename='Gi'):
        """
        todo
        """
        ErrorRatesModel.__init__(self, error_rates, n_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idlename=idlename)

    def _success_prob(self, circuit, cache):
        """
        todo
        """
        pvec = self._paramvec**2
        if cache is None:
            cache = self._circuit_cache(circuit)

        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = cache
        # The success probability for all the operations (the entanglment fidelity for the gates)
        sp = 1.0 - pvec

        # The depolarizing constant for the full sequence of twirled layers.
        lambda_all_layers = 1.0
        for inds_to_mult in inds_to_mult_by_layer[:-1]:
            lambda_all_layers *= 1 - alpha * (1 - prod(sp[inds_to_mult]))
        # lambda_all_layers = prod([(1 - alpha * (1 - prod(sp[inds_to_mult])))
        #                           for inds_to_mult in inds_to_mult_by_layer[:-1]])

        # The readout success probability.
        successprob_readout = prod(sp[inds_to_mult_by_layer[-1]])
        # THe success probability of the circuit.
        successprob_circuit = lambda_all_layers * (successprob_readout - one_over_2_width) + one_over_2_width

        return successprob_circuit

    def _success_dprob(self, circuit, cache):
        pvec = self._paramvec**2
        dpvec_dparams = 2 * self._paramvec

        if cache is None:
            cache = self._circuit_cache(circuit)

        # p = product_layers(1 - alpha * (1 - prod_[inds4layer](1 - param))) * \
        #     (prod_[inds4LASTlayer](1 - param) - 1 / 2**width)
        # Note: indices cannot be repeated in a layer, i.e. either a given index appears one or zero times in inds4layer

        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = cache
        sp = 1.0 - pvec
        deriv = _np.zeros(len(pvec), 'd')

        nLayers = len(inds_to_mult_by_layer)
        lambda_per_layer = _np.empty(nLayers, 'd')
        for i, inds_to_mult in enumerate(inds_to_mult_by_layer[:-1]):
            lambda_per_layer[i] = 1 - alpha * (1 - prod(sp[inds_to_mult]))

        successprob_readout = prod(sp[inds_to_mult_by_layer[-1]])
        lambda_per_layer[nLayers - 1] = successprob_readout - one_over_2_width
        lambda_all_layers = prod(lambda_per_layer)  # includes readout factor as last layer

        #All layers except last
        for i, inds_to_mult in enumerate(inds_to_mult_by_layer[:-1]):
            lambda_all_but_current_layer = lambda_all_layers / lambda_per_layer[i]
            # for each such ind, when we take deriv wrt this index, we need to differentiate this layer, etc.
            for ind in inds_to_mult:
                deriv[ind] += lambda_all_but_current_layer * alpha * \
                    (prod(sp[inds_to_mult]) / sp[ind]) * -1.0  # what if sp[ind] == 0?

        #Last layer
        lambda_all_but_current_layer = lambda_all_layers / lambda_per_layer[-1]
        for ind in inds_to_mult_by_layer[-1]:
            deriv[ind] += lambda_all_but_current_layer * (successprob_readout / sp[ind]) * -1.0  # what if sp[ind] == 0?

        return deriv * dpvec_dparams


class TwirledGatesModel(ErrorRatesModel):

    def __init__(self, error_rates, n_qubits, state_space_labels=None, alias_dict={}, idlename='Gi'):
        """
        todo
        """
        ErrorRatesModel.__init__(self, error_rates, n_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idlename=idlename)

    def _circuit_cache(self, circuit):
        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = super()._circuit_cache(circuit)
        all_inds_to_mult = _np.concatenate(inds_to_mult_by_layer[:-1])
        readout_inds_to_mult = inds_to_mult_by_layer[-1]
        all_inds_to_mult_cnt = _np.zeros(self.num_params(), int)
        for i in all_inds_to_mult:
            all_inds_to_mult_cnt[i] += 1
        return width, depth, alpha, one_over_2_width, all_inds_to_mult, readout_inds_to_mult, all_inds_to_mult_cnt

    def _success_prob(self, circuit, cache):
        """
        todo
        """
        pvec = self._paramvec**2
        if cache is None:
            cache = self._circuit_cache(circuit)

        width, depth, alpha, one_over_2_width, all_inds_to_mult, readout_inds_to_mult, all_inds_to_mult_cnt = cache
        # The success probability for all the operations (the entanglment fidelity for the gates)
        sp = 1.0 - pvec

        # The 'lambda' for all gates (+ readout, which isn't used).
        lambda_ops = 1.0 - alpha * pvec

        # The depolarizing constant for the full sequence of twirled gates.
        lambda_all_layers = prod(lambda_ops[all_inds_to_mult])
        # The readout success probability.
        successprob_readout = prod(sp[readout_inds_to_mult])
        # THe success probability of the circuit.
        successprob_circuit = lambda_all_layers * (successprob_readout - one_over_2_width) + one_over_2_width

        return successprob_circuit

    def _success_dprob(self, circuit, cache):
        """
        todo
        """
        pvec = self._paramvec**2
        dpvec_dparams = 2 * self._paramvec

        if cache is None:
            cache = self._circuit_cache(circuit)

        width, depth, alpha, one_over_2_width, all_inds_to_mult, readout_inds_to_mult, all_inds_to_mult_cnt = cache
        sp = 1.0 - pvec
        lambda_ops = 1.0 - alpha * pvec
        deriv = _np.zeros(len(pvec), 'd')

        # The depolarizing constant for the full sequence of twirled gates.
        lambda_all_layers = prod(lambda_ops[all_inds_to_mult])
        for i, n in enumerate(all_inds_to_mult_cnt):
            deriv[i] = n * lambda_all_layers / lambda_ops[i] * -alpha  # -alpha = d(lambda_ops/dparam)

        # The readout success probability.
        readout_deriv = _np.zeros(len(pvec), 'd')
        successprob_readout = prod(sp[readout_inds_to_mult])
        for ind in readout_inds_to_mult:
            readout_deriv[ind] = (successprob_readout / sp[ind]) * -1.0  # what if sp[ind] == 0?

        # The success probability of the circuit.
        #successprob_circuit = lambda_all_layers * (successprob_readout - one_over_2_width) + one_over_2_width

        # product rule
        return (deriv * (successprob_readout - one_over_2_width) + lambda_all_layers * readout_deriv) * dpvec_dparams


class AnyErrorCausesFailureModel(ErrorRatesModel):

    def __init__(self, error_rates, n_qubits, state_space_labels=None, alias_dict={}, idlename='Gi'):
        """
        todo
        """
        ErrorRatesModel.__init__(self, error_rates, n_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idlename=idlename)

    def _circuit_cache(self, circuit):
        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = super()._circuit_cache(circuit)
        all_inds_to_mult = _np.concatenate(inds_to_mult_by_layer)
        all_inds_to_mult_cnt = _np.zeros(self.num_params(), int)
        for i in all_inds_to_mult:
            all_inds_to_mult_cnt[i] += 1
        return all_inds_to_mult, all_inds_to_mult_cnt

    def _success_prob(self, circuit, cache):
        """
        todo
        """
        pvec = self._paramvec**2

        if cache is None:
            cache = self._circuit_cache(circuit)

        all_inds_to_mult, all_inds_to_mult_cnt = cache
        # The success probability for all the operations (the entanglment fidelity for the gates)
        sp = 1.0 - pvec

        # The probability that every operation succeeds.
        successprob_circuit = prod(sp[all_inds_to_mult])

        return successprob_circuit

    def _success_dprob(self, circuit, cache):
        """
        todo
        """
        pvec = self._paramvec**2
        dpvec_dparams = 2 * self._paramvec

        if cache is None:
            cache = self._circuit_cache(circuit)

        all_inds_to_mult, all_inds_to_mult_cnt = cache
        sp = 1.0 - pvec
        successprob_circuit = prod(sp[all_inds_to_mult])
        deriv = _np.zeros(len(pvec), 'd')
        for i, n in enumerate(all_inds_to_mult_cnt):
            deriv[i] = n * successprob_circuit / sp[i] * -1.0

        return deriv * dpvec_dparams


class AnyErrorCausesRandomOutputModel(ErrorRatesModel):

    def __init__(self, error_rates, n_qubits, state_space_labels=None, alias_dict={}, idlename='Gi'):
        """
        todo
        """
        ErrorRatesModel.__init__(self, error_rates, n_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idlename=idlename)

    def _circuit_cache(self, circuit):
        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = super()._circuit_cache(circuit)
        all_inds_to_mult = _np.concatenate(inds_to_mult_by_layer)
        all_inds_to_mult_cnt = _np.zeros(self.num_params(), int)
        for i in all_inds_to_mult:
            all_inds_to_mult_cnt[i] += 1
        return one_over_2_width, all_inds_to_mult, all_inds_to_mult_cnt

    def _success_prob(self, circuit, cache):
        """
        todo
        """
        pvec = self._paramvec**2
        if cache is None:
            cache = self._circuit_cache(circuit)

        one_over_2_width, all_inds_to_mult, all_inds_to_mult_cnt = cache
        # The success probability for all the operations (the entanglment fidelity for the gates)
        sp = 1.0 - pvec

        # The probability that every operation succeeds.
        successprob_all_ops = prod(sp[all_inds_to_mult])
        # The circuit succeeds if all ops succeed, and has a random outcome otherwise.
        successprob_circuit = successprob_all_ops + (1 - successprob_all_ops) * one_over_2_width

        return successprob_circuit

    def _success_dprob(self, circuit, cache):
        """
        todo
        """
        pvec = self._paramvec**2
        dpvec_dparams = 2 * self._paramvec

        if cache is None:
            cache = self._circuit_cache(circuit)

        one_over_2_width, all_inds_to_mult, all_inds_to_mult_cnt = cache
        sp = 1.0 - pvec

        successprob_all_ops = prod(sp[all_inds_to_mult])
        deriv = _np.zeros(len(pvec), 'd')
        for i, n in enumerate(all_inds_to_mult_cnt):
            deriv[i] = n * successprob_all_ops / sp[i] * -1.0

        # The circuit succeeds if all ops succeed, and has a random outcome otherwise.
        # successprob_circuit = successprob_all_ops + (1 - successprob_all_ops) / 2**width
        # = const + (1-1/2**width)*successprobs_all_ops
        deriv *= (1.0 - one_over_2_width)
        return deriv * dpvec_dparams

    # def ORIGINAL_success_prob(self, circuit, cache):
    #     """
    #     todo
    #     """
    #     if not isinstance(circuit, _Circuit):
    #         circuit = _Circuit.fromtup(circuit)

    #     depth = circuit.depth()
    #     width = circuit.width()
    #     pvec = self._paramvec
    #     g_inds = self._gate_error_rate_indices
    #     r_inds = self._readout_error_rate_indices

    #     if self.model_type in ('FE', 'FiE+U'):

    #         two_q_gates = []
    #         for i in range(depth):
    #             layer = circuit.get_layer(i)
    #             two_q_gates += [q.qubits for q in layer if len(q.qubits) > 1]

    #         sp = 1
    #         oneqs = {q: depth for q in circuit.line_labels}

    #         for qs in two_q_gates:
    #             sp = sp * (1 - pvec[g_inds[frozenset(qs)]])
    #             oneqs[qs[0]] += -1
    #             oneqs[qs[1]] += -1

    #         sp = sp * _np.prod([(1 - pvec[g_inds[q]])**oneqs[q]
    #                             * (1 - pvec[r_inds[q]]) for q in circuit.line_labels])

    #         if self.model_type == 'FiE+U':
    #             sp = sp + (1 - sp) * (1 / 2**width)

    #         return sp

    #     if self.model_type == 'GlobalDep':

    #         p = 1
    #         for i in range(depth):

    #             layer = circuit.get_layer(i)
    #             sp_layer = 1
    #             usedQs = []

    #             for gate in layer:
    #                 if len(gate.qubits) > 1:
    #                     usedQs += list(gate.qubits)
    #                     sp_layer = sp_layer * (1 - pvec[g_inds[frozenset(gate.qubits)]])

    #             for q in circuit.line_labels:
    #                 if q not in usedQs:
    #                     sp_layer = sp_layer * (1 - pvec[g_inds[q]])

    #             p_layer = 1 - 4**width * (1 - sp_layer) / (4**width - 1)
    #             p = p * p_layer

    #         # Bit-flip readout error as a pre-measurement depolarizing channel.
    #         sp_layer = _np.prod([(1 - 3 * pvec[r_inds[q]] / 2) for q in circuit.line_labels])
    #         p_layer = 1 - 4**width * (1 - sp_layer) / (4**width - 1)
    #         p = p * p_layer
    #         sp = p + (1 - p) * (1 / 2**width)

    #         return sp
