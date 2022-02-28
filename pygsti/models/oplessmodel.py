"""
Defines the OplessModel class
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np

from pygsti.models.model import Model as _Model
from pygsti.baseobjs.label import Label as _Label
from pygsti.baseobjs.opcalc import float_product as prod
from pygsti.baseobjs.statespace import StateSpace as _StateSpace
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.forwardsims.successfailfwdsim import SuccessFailForwardSimulator as _SuccessFailForwardSimulator
from pygsti.baseobjs.resourceallocation import ResourceAllocation as _ResourceAllocation
from pygsti.tools import slicetools as _slct


class OplessModel(_Model):
    """
    A model that does *not* have independent component operations.

    :class:`OplessModel`-derived classes often implement coarser models that
    predict the success or outcome probability of a circuit based on simple
    properties of the circuit and not detailed gate-level modeling.

    Parameters
    ----------
    state_space : StateSpace
        The state space of this model.
    """

    def __init__(self, state_space):
        _Model.__init__(self, state_space)

        #Setting things the rest of pyGSTi expects but probably shouldn't...
        self.basis = None

    @property
    def dim(self):
        return 0

    @property
    def parameter_bounds(self):
        """ Upper and lower bounds on the values of each parameter, utilized by optimization routines """
        # Note: this just replicates the base class version (in `Model`) but is needed to have setter method.
        return self._param_bounds

    @parameter_bounds.setter
    def parameter_bounds(self, val):  # (opless models can have their bounds set directly)
        """ Upper and lower bounds on the values of each parameter, utilized by optimization routines """
        if val is not None:
            assert(val.shape == (self.num_params, 2)), \
                "`parameter_bounds` can only be set to None or a (num_params, 2)-shaped array!"
        self._param_bounds = val
        self.dirty = True

    def circuit_outcomes(self, circuit):  # needed for sparse data detection
        """
        Get all the possible outcome labels produced by simulating this circuit.

        Parameters
        ----------
        circuit : Circuit
            Circuit to get outcomes of.

        Returns
        -------
        tuple
        """
        raise NotImplementedError("Derived classes should implement this!")

    def probabilities(self, circuit, outcomes=None, time=None):
        """
        Construct a dictionary containing the outcome probabilities of `circuit`.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        outcomes : list or tuple
            A sequence of outcomes, which can themselves be either tuples
            (to include intermediate measurements) or simple strings, e.g. `'010'`.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        Returns
        -------
        probs : OutcomeLabelDict
            A dictionary with keys equal to outcome labels and
            values equal to probabilities.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def bulk_probabilities(self, circuits, clip_to=None, comm=None, mem_limit=None, smartc=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : (list of Circuits) or CircuitOutcomeProbabilityArrayLayout
            When a list, each element specifies a circuit to compute outcome probabilities for.
            A :class:`CircuitOutcomeProbabilityArrayLayout` specifies the circuits along with
            an internal memory layout that reduces the time required by this function and can
            restrict the computed probabilities to those corresponding to only certain outcomes.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of evalTree (if it is split).

        mem_limit : int, optional
            A rough memory limit in bytes which is used to determine processor
            allocation.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(outcome, p)` tuples, where `outcome` is a tuple of labels
            and `p` is the corresponding probability.
        """
        raise NotImplementedError("Derived classes should implement this!")

    def __str__(self):
        raise NotImplementedError("Derived classes should implement OplessModel.__str__ !!")


class SuccessFailModel(OplessModel):
    """
    An op-less model that always outputs 2 (success & failure) probabilities for each circuit.

    Parameters
    ----------
    state_space : StateSpace
        The state space of this model.

    use_cache : bool, optional
        Whether a cache should be used to increase performance.
    """
    def __init__(self, state_space, use_cache=False):
        OplessModel.__init__(self, state_space)
        self.use_cache = use_cache
        self._sim = _SuccessFailForwardSimulator(self)

    def _to_nice_serialization(self):
        state = super()._to_nice_serialization()
        state.update({'use_cache': self.use_cache
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):
        state_space = _StateSpace.from_nice_serialization(state['state_space'])
        return cls(state_space, state['use_cache'])

    @property
    def sim(self):
        """ Forward simulator for this model """
        return self._sim

    def _post_copy(self, copy_into, memo):
        """
        Called after all other copying is done, to perform "linking" between
        the new model (`copy_into`) and its members.
        """
        copy_into.sim.model = copy_into  # set copy's `.model` link (just linking so no need to use memo)

    def circuit_outcomes(self, circuit):  # needed for sparse data detection
        """
        Get all the possible outcome labels produced by simulating this circuit.

        Parameters
        ----------
        circuit : Circuit
            Circuit to get outcomes of.

        Returns
        -------
        tuple
        """
        return (('success',), ('fail',))

    def _success_prob(self, circuit, cache):
        raise NotImplementedError("Derived classes should implement this!")

    def _success_dprob(self, circuit, param_slice, cache):
        """ Derived classes can override this.  Default implemntation is to use finite difference. """
        eps = 1e-7
        orig_pvec = self.to_vector()
        wrtIndices = _slct.indices(param_slice) if (param_slice is not None) else list(range(self.num_params))
        sp0 = self._success_prob(circuit, cache)

        deriv = _np.empty(len(wrtIndices), 'd')
        for i in wrtIndices:
            p_plus_dp = orig_pvec.copy()
            p_plus_dp[i] += eps
            self.from_vector(p_plus_dp)
            sp1 = self._success_prob(circuit, cache)
            deriv[i] = (sp1 - sp0) / eps
        self.from_vector(orig_pvec)
        return deriv

    def probabilities(self, circuit, outcomes=None, time=None):
        """
        Construct a dictionary containing the outcome probabilities of `circuit`.

        Parameters
        ----------
        circuit : Circuit or tuple of operation labels
            The sequence of operation labels specifying the circuit.

        outcomes : list or tuple
            A sequence of outcomes, which can themselves be either tuples
            (to include intermediate measurements) or simple strings, e.g. `'010'`.

        time : float, optional
            The *start* time at which `circuit` is evaluated.

        Returns
        -------
        probs : OutcomeLabelDict
            A dictionary with keys equal to outcome labels and
            values equal to probabilities.
        """
        return self.sim.probs(circuit, outcomes, time)

    def bulk_probabilities(self, circuits, clip_to=None, comm=None, mem_limit=None, smartc=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : (list of Circuits) or CircuitOutcomeProbabilityArrayLayout
            When a list, each element specifies a circuit to compute outcome probabilities for.
            A :class:`CircuitOutcomeProbabilityArrayLayout` specifies the circuits along with
            an internal memory layout that reduces the time required by this function and can
            restrict the computed probabilities to those corresponding to only certain outcomes.

        clip_to : 2-tuple, optional
            (min,max) to clip return value if not None.

        comm : mpi4py.MPI.Comm, optional
            When not None, an MPI communicator for distributing the computation
            across multiple processors.  Distribution is performed over
            subtrees of evalTree (if it is split).

        mem_limit : int, optional
            A rough memory limit in bytes which is used to determine processor
            allocation.

        smartc : SmartCache, optional
            A cache object to cache & use previously cached values inside this
            function.

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[opstr]` is an ordered dictionary of
            `(outcome, p)` tuples, where `outcome` is a tuple of labels
            and `p` is the corresponding probability.
        """
        resource_alloc = _ResourceAllocation(comm, mem_limit)
        return self.sim.bulk_probs(circuits, clip_to, resource_alloc, smartc)


class ErrorRatesModel(SuccessFailModel):

    """
    A success-fail model based on per-gate error rates.

    Parameters
    ----------
    error_rates : dict
        A dictionary with "gates" and "readout" keys, each of which corresponds to a
        dictionary of error-rates for gates or SPAM elements, respectively.

    num_qubits : int
        The number of qubits in the model.

    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    alias_dict : dict, optional
        An alias dictionary mapping the gate labels in circuits to the
        keys of a (nested) `error_rates` dictionary.  This allows, for instance,
        many gates' error rates to be set by the same model parameter.

    idle_name : str, optional
        The gate name to be used for the 1-qubit idle gate (this should be
        set in `error_rates` to add idle errors.
    """
    def __init__(self, error_rates, num_qubits, state_space_labels=None, alias_dict={}, idle_name='Gi'):
        if state_space_labels is None:
            state_space_labels = ['Q%d' % i for i in range(num_qubits)]
        else:
            assert(len(state_space_labels) == num_qubits)

        SuccessFailModel.__init__(self, state_space_labels, use_cache=True)

        gate_error_rate_keys = (list(error_rates['gates'].keys()))
        readout_error_rate_keys = (list(error_rates['readout'].keys()))

        # if gate_error_rate_keys[0] in state_space_labels:
        #     self._gateind = True
        # else:
        #     self._gateind = False

        self._idlename = idle_name
        self._alias_dict = alias_dict.copy()
        self._gate_error_rate_indices = {k: i for i, k in enumerate(gate_error_rate_keys)}
        self._readout_error_rate_indices = {k: i + len(gate_error_rate_keys)
                                            for i, k in enumerate(readout_error_rate_keys)}
        self._paramvec = _np.concatenate(
            (_np.array([_np.sqrt(error_rates['gates'][k]) for k in gate_error_rate_keys], 'd'),
             _np.array([_np.sqrt(error_rates['readout'][k]) for k in readout_error_rate_keys], 'd'))
        )

    @property
    def primitive_op_labels(self):
        #So primitive op wildcard budget can work with ErrorRatesModel
        return tuple(self._gate_error_rate_indices.keys())

    @property
    def primitive_instrument_labels(self):
        #So primitive op wildcard budget can work with ErrorRatesModel
        return tuple()  # no support for instruments yet

    def __str__(self):
        s = "Error Rates model with error rates: \n" + \
            "\n".join(["%s = %g" % (k, self._paramvec[i]**2) for k, i in self._gate_error_rate_indices.items()]) + \
            "\n" + \
            "\n".join(["%s = %g" % (k, self._paramvec[i]**2) for k, i in self._readout_error_rate_indices.items()])
        return s

    def to_dict(self):
        """
        Convert this model to a dictionary (for debugging or easy printing).
        """
        error_rate_dict = {'gates': {}, 'readout': {}}
        error_rate_dict['gates'] = {k: self._paramvec[i]**2 for k, i in self._gate_error_rate_indices.items()}
        error_rate_dict['readout'] = {k: self._paramvec[i]**2 for k, i in self._readout_error_rate_indices.items()}
        asdict = {'error_rates': error_rate_dict, 'alias_dict': self._alias_dict.copy()}
        return asdict

    def _circuit_cache(self, circuit):
        if not isinstance(circuit, _Circuit):
            circuit = _Circuit.from_tuple(circuit)

        depth = circuit.depth
        width = circuit.width
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

        #         inds_to_mult_by_layer.append(_np.array(inds_to_mult, _np.int64))

        # else:

        def indices_for_label(lbl):
            """ Returns a list of the parameter indices corresponding to `lbl` """
            if self._alias_dict.get(lbl, lbl) in g_inds:
                return [g_inds[self._alias_dict.get(lbl, lbl)]]
            elif self._alias_dict.get(lbl.name, lbl.name) in g_inds:  # allow, e.g. "Gx" to work for Gx:0, Gx:1, etc.
                return [g_inds[self._alias_dict.get(lbl.name, lbl.name)]]
            # allows for a model where a gate with arguments has same error rate for all arg values
            elif self._alias_dict.get(_Label(lbl.name, lbl.qubits), _Label(lbl.name, lbl.qubits)) in g_inds:
                return [g_inds[self._alias_dict.get(_Label(lbl.name, lbl.qubits))]] 
            else:
                indices = []
                assert(not lbl.is_simple()), "Cannot find error rate for label: %s" % str(lbl)
                for component in lbl:
                    indices.extend(indices_for_label(component))
                return indices

        if self._idlename is not None:
            layers_with_idles = [circuit.layer_label_with_idles(i, idle_gate_name=self._idlename) for i in range(depth)]
            inds_to_mult_by_layer = [_np.array(indices_for_label(layer), _np.int64) for layer in layers_with_idles]
        else:
            inds_to_mult_by_layer = [_np.array(indices_for_label(circuit.layer_label(i)), _np.int64)
                                     for i in range(depth)]

        # Bit-flip readout error as a pre-measurement depolarizing channel.
        inds_to_mult = [r_inds[q] for q in circuit.line_labels]
        inds_to_mult_by_layer.append(_np.array(inds_to_mult, _np.int64))

        # The scaling constant such that lambda = 1 - alpha * epsilon where lambda is the diagonal of a depolarizing
        # channel with entanglement infidelity of epsilon.
        alpha = 4**width / (4**width - 1)

        return (width, depth, alpha, 1 / 2**width, inds_to_mult_by_layer)


class TwirledLayersModel(ErrorRatesModel):
    """
    A model where twirled-layer error rates are computed and multiplied together to compute success probabilities.

    In this model, the success probability of a circuit is the product of
    `1.0 - alpha * pfail` terms, one per layer of the circuit (including idles).
    The `pfail` of a circuit layer is given as `1 - prod(1 - error_rate_i)`, where
    `i` ranges over the gates in the layer.  `alpha` is the constant `4^w / (4^w - 1)`
    where `w` is the circuit width.

    Parameters
    ----------
    error_rates : dict
        A dictionary with "gates" and "readout" keys, each of which corresponds to a
        dictionary of error-rates for gates or SPAM elements, respectively.

    num_qubits : int
        The number of qubits in the model.

    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    alias_dict : dict, optional
        An alias dictionary mapping the gate labels in circuits to the
        keys of a (nested) `error_rates` dictionary.  This allows, for instance,
        many gates' error rates to be set by the same model parameter.

    idle_name : str, optional
        The gate name to be used for the 1-qubit idle gate (this should be
        set in `error_rates` to add idle errors.
    """
    def __init__(self, error_rates, num_qubits, state_space_labels=None, alias_dict={}, idle_name='Gi'):
        ErrorRatesModel.__init__(self, error_rates, num_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idle_name=idle_name)

    def _success_prob(self, circuit, cache):
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

    def _success_dprob(self, circuit, param_slice, cache):
        assert(param_slice is None or _slct.length(param_slice) == len(self._paramvec)), \
            "No support for derivatives with respect to a subset of model parameters yet!"
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
    """
    A model where twirled-gate error rates are computed and multiplied together to compute success probabilities.

    In this model, the success probability of a circuit is the product of
    `1.0 - alpha * pfail` terms, one per gate of the circuit (including idles).
    The `pfail` of a gate is given as `1 - error_rate`, and `alpha` is the constant
    `4^w / (4^w - 1)` where `w` is the circuit width.

    Parameters
    ----------
    error_rates : dict
        A dictionary with "gates" and "readout" keys, each of which corresponds to a
        dictionary of error-rates for gates or SPAM elements, respectively.

    num_qubits : int
        The number of qubits in the model.

    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    alias_dict : dict, optional
        An alias dictionary mapping the gate labels in circuits to the
        keys of a (nested) `error_rates` dictionary.  This allows, for instance,
        many gates' error rates to be set by the same model parameter.

    idle_name : str, optional
        The gate name to be used for the 1-qubit idle gate (this should be
        set in `error_rates` to add idle errors.
    """
    def __init__(self, error_rates, num_qubits, state_space_labels=None, alias_dict={}, idle_name='Gi'):
        """
        todo
        """
        ErrorRatesModel.__init__(self, error_rates, num_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idle_name=idle_name)

    def _circuit_cache(self, circuit):
        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = super()._circuit_cache(circuit)
        all_inds_to_mult = _np.concatenate(inds_to_mult_by_layer[:-1])
        readout_inds_to_mult = inds_to_mult_by_layer[-1]
        all_inds_to_mult_cnt = _np.zeros(self.num_params, _np.int64)
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

    def _success_dprob(self, circuit, param_slice, cache):
        """
        todo
        """
        assert(param_slice is None or _slct.length(param_slice) == len(self._paramvec)), \
            "No support for derivatives with respect to a subset of model parameters yet!"
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
    """
    A model where any gate failure causes a circuit failure.

    Specifically, the success probability of a circuit is give by
    `1 - prod(1 - error_rate_i)` where `i` ranges over all the gates in the circuit.
    That is, a circuit success probability is just the product of all its gate
    success probabilities. In this pessimistic model, any gate failure causes the
    circuit to fail.

    Parameters
    ----------
    error_rates : dict
        A dictionary with "gates" and "readout" keys, each of which corresponds to a
        dictionary of error-rates for gates or SPAM elements, respectively.

    num_qubits : int
        The number of qubits in the model.

    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    alias_dict : dict, optional
        An alias dictionary mapping the gate labels in circuits to the
        keys of a (nested) `error_rates` dictionary.  This allows, for instance,
        many gates' error rates to be set by the same model parameter.

    idle_name : str, optional
        The gate name to be used for the 1-qubit idle gate (this should be
        set in `error_rates` to add idle errors.
    """
    def __init__(self, error_rates, num_qubits, state_space_labels=None, alias_dict={}, idle_name='Gi'):
        ErrorRatesModel.__init__(self, error_rates, num_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idle_name=idle_name)

    def _circuit_cache(self, circuit):
        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = super()._circuit_cache(circuit)
        all_inds_to_mult = _np.concatenate(inds_to_mult_by_layer)
        all_inds_to_mult_cnt = _np.zeros(self.num_params, _np.int64)
        for i in all_inds_to_mult:
            all_inds_to_mult_cnt[i] += 1
        return all_inds_to_mult, all_inds_to_mult_cnt

    def _success_prob(self, circuit, cache):
        pvec = self._paramvec**2

        if cache is None:
            cache = self._circuit_cache(circuit)

        all_inds_to_mult, all_inds_to_mult_cnt = cache
        # The success probability for all the operations (the entanglment fidelity for the gates)
        sp = 1.0 - pvec

        # The probability that every operation succeeds.
        successprob_circuit = prod(sp[all_inds_to_mult])

        return successprob_circuit

    def _success_dprob(self, circuit, param_slice, cache):
        assert(param_slice is None or _slct.length(param_slice) == len(self._paramvec)), \
            "No support for derivatives with respect to a subset of model parameters yet!"
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
    """
    A model where any gate error causes a random circuit output.

    Specifically, the success probability of a circuit is give by
    `all_ok + (1 - all_ok) * 1 / 2^circuit_width` where `all_ok` is the
    probability that all the gates succeed:
    `all_ok = 1 - prod(1 - error_rate_i)` with `i` ranging over all the
    gates in the circuit.  In this model, any gate failure causes the
    circuit to produce a random output.

    Parameters
    ----------
    error_rates : dict
        A dictionary with "gates" and "readout" keys, each of which corresponds to a
        dictionary of error-rates for gates or SPAM elements, respectively.

    num_qubits : int
        The number of qubits in the model.

    state_space_labels : StateSpaceLabels or list or tuple
        The decomposition (with labels) of (pure) state-space this model
        acts upon.  Regardless of whether the model contains operators or
        superoperators, this argument describes the Hilbert space dimension
        and imposed structure.  If a list or tuple is given, it must be
        of a from that can be passed to `StateSpaceLabels.__init__`.

    alias_dict : dict, optional
        An alias dictionary mapping the gate labels in circuits to the
        keys of a (nested) `error_rates` dictionary.  This allows, for instance,
        many gates' error rates to be set by the same model parameter.

    idle_name : str, optional
        The gate name to be used for the 1-qubit idle gate (this should be
        set in `error_rates` to add idle errors.
    """
    def __init__(self, error_rates, num_qubits, state_space_labels=None, alias_dict={}, idle_name='Gi'):
        ErrorRatesModel.__init__(self, error_rates, num_qubits, state_space_labels=state_space_labels,
                                 alias_dict=alias_dict, idle_name=idle_name)

    def _circuit_cache(self, circuit):
        width, depth, alpha, one_over_2_width, inds_to_mult_by_layer = super()._circuit_cache(circuit)
        all_inds_to_mult = _np.concatenate(inds_to_mult_by_layer)
        all_inds_to_mult_cnt = _np.zeros(self.num_params, _np.int64)
        for i in all_inds_to_mult:
            all_inds_to_mult_cnt[i] += 1
        return one_over_2_width, all_inds_to_mult, all_inds_to_mult_cnt

    def _success_prob(self, circuit, cache):
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

    def _success_dprob(self, circuit, param_slice, cache):
        """
        todo
        """
        assert(param_slice is None or _slct.length(param_slice) == len(self._paramvec)), \
            "No support for derivatives with respect to a subset of model parameters yet!"
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
