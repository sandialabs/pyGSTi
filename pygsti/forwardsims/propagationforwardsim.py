"""
Defines the PropagationForwardSimulator class.
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
from pygsti.forwardsims import ForwardSimulator as _ForwardSimulator
from pygsti.errorgenpropagation import ErrorGeneratorPropagator
from pygsti.baseobjs import Label
import pygsti.tools.errgenproptools as _eprop
import scipy.linalg as _spl


class SimplePropagationForwardSimulator(_ForwardSimulator):
    """
    Forward simulator class which utilizes the error generator propagation
    framework as a backend for approximate simulation of noisy clifford circuit
    under both general markovian and non-markovian noise.
    """

    def __init__(self, approximate=True, bch_order=1, taylor_order=1, cumulant_order=2, model=None, ideal_model=None, truncation_threshold = 1e-14):
        """
        Instantiate an instance of `PropagationForwardSimulator`.

        Parameters
        ----------
        approximate : bool, optional (default True)
            A flag for specifying the handling of post-propagation error generators.
            When True a series of approximations are applied to the set of propagated
            error generator layers coming out of the error generator propagation framework
            as part of computing the final probabilities. First an approximate end-of-circuit
            error generator is constructed using some order of the BCH approximation.
            Second, the effective end-of-circuit error generator's corrections to the 
            ideal output probability distribution are computed using a taylor series approximation.
            When False we instead exponentiate and compose the dense representations for the
            propagated error generators numerically to produce the final output probability
            distribution.

            Note: When simulating models non-Markovian parameters (in the form of
            temporally correlated fluctuating parameters described by a covariance function)
            there is also an approximation being applied to average over the stochastic processes
            for the non-Markovian parameters in the form of a cumulant expansion at some order. 
            This cumulant expansion approximation is applied in both cases to construct the
            non-Markovian generator layers independent of the value for approximate.

        bch_order : int, optional (default 1)
            Order of the BCH approximation to use in forming the effective end-of-circuit
            error generator. Value ignored when `approximate`=False.

        taylor_order : int, optional (default 1)
            Order of the Taylor series approximation to use in computing approximate corrections
            to the ideal probability distribtions for the noisy Clifford circuit being simulated.
            Value ignored when `approximate`=False.

        cumulant_order : int, optional (default 2)
            Order of the cumulant expansion to use in computing the approximate non-Markovian
            error generator layers that arise when averaging over the stochastic processes
            associated with non-Markovian noise parameters for the model. Note this is only
            used when there is a covariance function parameterizing such non-Markovian parameters
            in the model.

        model : `Model`, optional (default None)
            Optional parent model associated with this simulator.

        ideal_model : `Model`, optional (default None)
            Optional model corresponding to the ideal target model. Currently used when performing
            non-approximate simulation. TODO: figure out how to grab this a different way.

        truncation_threshold : float, optional (default 1e-14) #TODO: Make these thresholds individually specifable.
            Threshold below which any error generators with magnitudes below this value
            are truncated during the BCH, taylor series, or cumulant expansion approximations.
        """
        self.approximate = approximate
        self.bch_order = bch_order
        self.taylor_order = taylor_order
        self.cumulant_order = cumulant_order
        self.truncation_threshold = truncation_threshold

        if model is not None:
            self.errorgen_propagator = ErrorGeneratorPropagator(model)
        else:
            self.errorgen_propagator = None

        if ideal_model is not None:
            self.ideal_model = ideal_model
            self.ideal_model.sim = 'matrix'
        else:
            self.ideal_model = model.copy()
            self.ideal_model.sim = 'matrix'

        super().__init__(model)

    def bulk_probs(self, circuits, clip=False, resource_alloc=None, smartc=None, precomp_ideals=None, precomp_errorgen_phases=None):
        """
        Construct a dictionary containing the probabilities for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        clip : bool, optional (default False)
            If true clip the probabilities to the interval [0,1].

        precomp_ideals : list of numpy ndarrays
            A precomputed list of the ideal (noise-free) process matrices for each circuit.
        
        precomp_errorgen_phases: list of lists of dicts, optional (default None)
            An optionally precomputed list of error generator phase maps for each circuit.
            Each of this is a map from input error generators to the phases accrued on that
            error generator as a result of propagation through a circuit. Each of these is
            one of the lists of dictionaries returned by 
            `ErrorGeneratorPropagator.errorgen_transform_phases`. 

        Returns
        -------
        probs : dictionary
            A dictionary such that `probs[circuit]` is an ordered dictionary of
            outcome probabilities whose keys are outcome labels.
        """

        if self.errorgen_propagator is None:
            self.errorgen_propagator = ErrorGeneratorPropagator(self.model)

        #TODO: This is currently assuming all circuits have the same outcome labels
        #need to generalize this.
        circuit_outcomes = self.model.circuit_outcomes(circuits[0])
        prob_dict = dict()

        if precomp_ideals is None:
            precomp_ideals = [None]*len(circuits)
        if precomp_errorgen_phases is None:
            precomp_errorgen_phases = [None]*len(circuits)

        if self.approximate:
            for ckt in circuits:
                propagated_errorgen_layer = self.errorgen_propagator.propagate_errorgens_bch(ckt, bch_order=self.bch_order, include_spam=True,
                                                                                             truncation_threshold=self.truncation_threshold)
                approx_probs = _eprop.approximate_stabilizer_probabilities(propagated_errorgen_layer, ckt, order=self.taylor_order, 
                                                                           truncation_threshold=self.truncation_threshold)
                prob_dict[ckt] = {olbl: prob for olbl, prob in zip(circuit_outcomes, approx_probs)}

        else:
            #get the ideal state prep and povm: #TODO: Infer the correct preps and POVMs from circuits. 
            ideal_prep = self.ideal_model.circuit_layer_operator(Label('rho0'), typ='prep').copy()
            ideal_meas = self.ideal_model.circuit_layer_operator(Label('Mdefault'), typ='povm').copy()
            dense_prep = ideal_prep.to_dense()
            dense_prep.reshape((len(dense_prep),1))
            #print(f'{dense_prep=}')
            dense_effects = [effect.to_dense() for effect in ideal_meas.values()]
            for effect in dense_effects:
                effect.reshape((1,len(effect)))
                #print(f'{effect=}')
            for ckt, ideal_channel, errorgen_phases in zip(circuits, precomp_ideals, precomp_errorgen_phases):
                #get the eoc error channel, and the process matrix for the ideal circuit:
                if self.model.covariance_function is None:
                    eoc_channel = self.errorgen_propagator.eoc_error_channel(ckt, include_spam=True, use_bch=True,
                                                                            bch_kwargs={'bch_order':self.bch_order,
                                                                                        'truncation_threshold':self.truncation_threshold})
                else:
                    #propagate_errorgens_nonmarkovian returns a list of list of 
                    propagated_error_generators = self.errorgen_propagator.propagate_errorgens_nonmarkovian(ckt, include_spam=True, errorgen_phase_corrections=errorgen_phases)
                    
                    #loop though the propagated error generator layers and construct their error generators.
                    #Then exponentiate
                    non_empty_errorgen_layers = [err_gen_layer for err_gen_layer in propagated_error_generators if err_gen_layer]
                    errorgen_layer_arrays = self.errorgen_propagator.errorgen_layer_dicts_to_errorgens(non_empty_errorgen_layers, mx_basis=self.model.basis)

                    exp_error_generators = []
                    for err_gen_layer in errorgen_layer_arrays:
                        exp_error_generators.append(_spl.expm(err_gen_layer))
                    #Next take the product of these exponentiated error generators.
                    #These are in circuit ordering, so reverse for matmul.
                    #exp_error_generators.reverse()
                    
                    #print(f'{eoc_channel=}')
                    if ideal_channel is None:
                        ideal_channel = self.ideal_model.sim.product(ckt)
                    #print(f'{ideal_channel=}')
                    #print(f'{ideal_channel@dense_prep=}')
                #calculate the probabilities.
                prob_vec = _np.zeros(len(ideal_meas))
                propagated_state = dense_prep.copy()
                for operation in [ideal_channel] + exp_error_generators:
                    propagated_state = operation@propagated_state

                for j, effect in enumerate(dense_effects):
                    prob_vec[j] = effect@propagated_state
                
                prob_dict[ckt] = {olbl: prob for olbl, prob in zip(circuit_outcomes, prob_vec)}
        
        return prob_dict


    def bulk_dprobs(self, circuits, clip=False):
        """
        Construct a dictionary containing the probability derivatives for an entire list of circuits.

        Parameters
        ----------
        circuits : list of Circuits
            The list of circuits.  May also be a :class:`CircuitOutcomeProbabilityArrayLayout`
            object containing pre-computed quantities that make this function run faster.

        clip : bool, optional (default False)
            If true clip the probabilities to the interval [0,1].

        Returns
        -------
        dprobs : dictionary
            A dictionary such that `dprobs[circuit]` is an ordered dictionary of
            derivative arrays (one element per differentiated parameter) whose
            keys are outcome labels
        """

        if self.errorgen_propagator is None:
            self.errorgen_propagator = ErrorGeneratorPropagator(self.model)

        precomp_ideals = [self.ideal_model.sim.product(ckt) for ckt in circuits]
        precomp_errorgen_phases = [self.errorgen_propagator.errorgen_transform_phases(ckt, include_spam=True) for ckt in circuits]

        probs = self.bulk_probs(circuits, precomp_ideals=precomp_ideals, precomp_errorgen_phases=precomp_errorgen_phases)
        orig_vec = self.model.to_vector().copy()

        circuit_outcomes = self.model.circuit_outcomes(circuits[0])

        #initialize a dprobs array:
        dprobs= {ckt: {lbl: _np.empty(self.model.num_params, dtype= _np.double) for lbl in circuit_outcomes} for ckt in circuits}

        eps = 1e-7
        for i in range(self.model.num_params):
            vec = orig_vec.copy()
            vec[i] += eps
            self.model.from_vector(vec, close=True)
            probs2 = self.bulk_probs(circuits, precomp_ideals= precomp_ideals, precomp_errorgen_phases=precomp_errorgen_phases)
            
            #need to parse this and construct the corresponding entries of the dprobs dict.
        
            for ckt in circuits:
                for lbl in circuit_outcomes:
                    dprobs[ckt][lbl][i] = (probs2[ckt][lbl] - probs[ckt][lbl]) / eps
            
        #restore the model to it's original value
        self.model.from_vector(orig_vec)

        return dprobs
    
    #TODO: Can reduce the number of phase precomputations even more, this should only be needed once per set of circuits.
    #TODO: make this more efficient and less duct tapey.
    def bulk_fill_probs(self, array_to_fill, layout, precomp_ideals=None, precomp_errorgen_phases=None):
        if self.errorgen_propagator is None:
            self.errorgen_propagator = ErrorGeneratorPropagator(self.model)

        probs = self.bulk_probs(layout._unique_circuits, precomp_ideals=precomp_ideals, precomp_errorgen_phases=precomp_errorgen_phases)

        for element_indices, circuit, _ in layout.iter_unique_circuits():
            array_to_fill[element_indices] = _np.fromiter(probs[circuit].values(), dtype=_np.double)

    #TODO: Make this less duct tapey
    def bulk_fill_dprobs(self, array_to_fill, layout, pr_array_to_fill=None):
        if self.errorgen_propagator is None:
            self.errorgen_propagator = ErrorGeneratorPropagator(self.model)
        
        precomp_ideals = [self.ideal_model.sim.product(ckt) for _, ckt, _ in layout.iter_unique_circuits()]
        precomp_errorgen_phases = [self.errorgen_propagator.errorgen_transform_phases(ckt, include_spam=True) for _, ckt, _ in layout.iter_unique_circuits()]

        if pr_array_to_fill is not None:
            self.bulk_fill_probs(pr_array_to_fill, layout, precomp_ideals=precomp_ideals, precomp_errorgen_phases=precomp_errorgen_phases)

        probs = _np.empty(len(layout), 'd')
        self.bulk_fill_probs(probs, layout, precomp_ideals=precomp_ideals, precomp_errorgen_phases=precomp_errorgen_phases)
        
        eps= 1e-7
        probs2 = _np.empty(len(layout), 'd')
        orig_vec = self.model.to_vector().copy()
        for i in range(self.model.num_params):
            vec = orig_vec.copy() 
            vec[i] += eps
            self.model.from_vector(vec, close=True)
            self.bulk_fill_probs(probs2, layout, precomp_ideals=precomp_ideals, precomp_errorgen_phases=precomp_errorgen_phases)
            array_to_fill[:, i] = (probs2 - probs) / eps
        self.model.from_vector(orig_vec, close=True)