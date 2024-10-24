import stim
import numpy as _np
import scipy.linalg as _spl
from .localstimerrorgen import LocalStimErrorgenLabel as _LSE
from numpy import abs,zeros, complex128
from numpy.linalg import multi_dot
from scipy.linalg import expm
from pygsti.tools.internalgates import standard_gatenames_stim_conversions
from .utilserrorgenpropagation import *
import copy as _copy
from pygsti.baseobjs import Label, ExplicitElementaryErrorgenBasis as _ExplicitElementaryErrorgenBasis
import pygsti.tools.errgenproptools as _eprop
import pygsti.tools.basistools as _bt

class ErrorGeneratorPropagator:

    def __init__(self, model, multi_gate_dict=None, bch_order=1,
                 bch_layerwise=False, nonmarkovian=False, multi_gate=False):
        self.model = model
        self.bch_order = bch_order
        self.bch_layerwise = bch_layerwise

    def eoc_error_channel(self, circuit, multi_gate_dict=None, include_spam=True, use_bch=False,
                          bch_kwargs=None, mx_basis='pp'):
        """
        Propagate all of the error generators for each circuit layer to the end of the circuit
        and return the result of exponentiating these error generators, and if necessary taking
        their products, to return the end of circuit error channel.

        Parameters
        ----------
        circuit : `Circuit`
            Circuit to construct a set of post gate error generators for.

        multi_gate_dict : dict, optional (default None)
            An optional dictionary mapping between gate name aliases and their
            standard name counterparts.

        include_spam : bool, optional (default True)
            If True then we include in the propagation the error generators associated
            with state preparation and measurement.

        use_bch : bool, optional (default False)
            If True use the BCH approximation as part of the propagation algorithm.

        bch_kwarg : dict, optional (default None)
            Only used is `use_bch` is True, this dictionary contains a set of
            BCH-specific kwargs which are passed to `propagate_errorgens_bch`.

        mx_basis : Basis or str, optional (default 'pp')
            Either a `Basis` object, or a string which can be cast to a `Basis`, specifying the
            basis in which to return the process matrix for the error channel.

        Returns
        -------
        eoc_error_channel : numpy.ndarray
            A numpy array corresponding to the end-of-circuit error channel resulting
            from the propagated error generators. This is 
        """

        if use_bch:
            raise NotImplementedError('Still under development.')
            propagated_error_generators = self.propagate_errorgens_bch(circuit, multi_gate_dict=multi_gate_dict,
                                                                       *bch_kwargs)
            
        else:
            propagated_error_generators = self.propagate_errorgens(circuit, multi_gate_dict, include_spam)
            #loop though the propagated error generator layers and construct their error generators.
            #Then exponentiate
            exp_error_generators = []
            for err_gen_layer_list in propagated_error_generators:
                if err_gen_layer_list: #if not empty. Should be length one if not empty.
                    #Keep the error generator in the standard basis until after the end-of-circuit
                    #channel is constructed so we can reduce the overhead of changing basis.
                    exp_error_generators.append(_spl.expm(self.errorgen_layer_dict_to_errorgen(err_gen_layer_list[0], mx_basis='pp')))
            #Next take the product of these exponentiated error generators.
            #These are in circuit ordering, so reverse for matmul.
            exp_error_generators.reverse()
            #print(exp_error_generators)
            eoc_error_channel = _np.linalg.multi_dot(exp_error_generators)
            #eoc_error_channel = _bt.change_basis(eoc_error_channel, from_basis='std', to_basis='pp')

        return eoc_error_channel
    
    def averaged_eoc_error_channel(self, circuit, multi_gate_dict=None, include_spam=True, mx_basis='pp'):
        """
        Propagate all of the error generators for each circuit layer to the end of the circuit,
        then apply a second order cumulant expansion to approximate the average of the end of circuit
        error channel over the values error generator rates that are stochastic processes.

        Parameters
        ----------
        circuit : `Circuit`
            Circuit to construct a set of post gate error generators for.

        multi_gate_dict : dict, optional (default None)
            An optional dictionary mapping between gate name aliases and their
            standard name counterparts.

        include_spam : bool, optional (default True)
            If True then we include in the propagation the error generators associated
            with state preparation and measurement.

        mx_basis : Basis or str, optional (default 'pp')
            Either a `Basis` object, or a string which can be cast to a `Basis`, specifying the
            basis in which to return the process matrix for the error channel.

        Returns
        -------
        avg_eoc_error_channel : numpy.ndarray
            A numpy array corresponding to the end-of-circuit error channel resulting
            from the propagated error generators and averaging over the stochastic processes
            for the error generator rates using a second order cumulant approximation.
        """

        #propagate_errorgens_nonmarkovian returns a list of list of 
        propagated_error_generators = self.propagate_errorgens_nonmarkovian(circuit, multi_gate_dict, include_spam)
        
        #construct the nonmarkovian propagators
        for i in range(len(propagated_error_generators)):
            for j in range(i+1):
                if i==j:
                    #<L_s> term:

                    prop_contrib = amam
                else:
                    pass
        
        
        
        
        
        #loop though the propagated error generator layers and construct their error generators.
        #Then exponentiate
        exp_error_generators = []
        for err_gen_layer_list in propagated_error_generators:
            if err_gen_layer_list: #if not empty. Should be length one if not empty.
                #Keep the error generator in the standard basis until after the end-of-circuit
                #channel is constructed so we can reduce the overhead of changing basis.
                exp_error_generators.append(_spl.expm(self.errorgen_layer_dict_to_errorgen(err_gen_layer_list[0], mx_basis='std')))
        #Next take the product of these exponentiated error generators.
        #These are in circuit ordering, so reverse for matmul.
        exp_error_generators.reverse()
        eoc_error_channel = _np.linalg.multi_dot(exp_error_generators)
        eoc_error_channel = _bt.change_basis(eoc_error_channel, from_basis='std', to_basis='pp')

        return eoc_error_channel


    def propagate_errorgens(self, circuit, multi_gate_dict=None, include_spam=True):
        """
        Propagate all of the error generators for each circuit layer to the end without
        any recombinations or averaging.

        Parameters
        ----------
        circuit : `Circuit`
            Circuit to construct a set of post gate error generators for.

        multi_gate_dict : dict, optional (default None)
            An optional dictionary mapping between gate name aliases and their
            standard name counterparts.

        include_spam : bool, optional (default True)
            If True then we include in the propagation the error generators associated
            with state preparation and measurement.

        Returns
        -------
        propagated_errorgen_layers : list of lists of dictionaries
            A list of lists of dictionaries, each corresponding to the result of propagating
            an error generator layer through to the end of the circuit.
        """
        #TODO: Check for proper handling of empty circuit and length 1 circuits.

        #start by converting the input circuit into a list of stim Tableaus with the 
        #first element dropped.
        stim_layers = self.construct_stim_layers(circuit, multi_gate_dict, drop_first_layer = not include_spam)
        
        #We next want to construct a new set of Tableaus corresponding to the cumulative products
        #of each of the circuit layers with those that follow. These Tableaus correspond to the
        #clifford operations each error generator will be propagated through in order to reach the
        #end of the circuit.
        propagation_layers = self.construct_propagation_layers(stim_layers)

        #Next we take the input circuit and construct a list of dictionaries, each corresponding
        #to the error generators for a particular gate layer.
        #TODO: Add proper inferencing for number of qubits:
        assert circuit.line_labels is not None and circuit.line_labels != ('*',)
        errorgen_layers = self.construct_errorgen_layers(circuit, len(circuit.line_labels), include_spam)

        #propagate the errorgen_layers through the propagation_layers to get a list
        #of end of circuit error generator dictionaries.
        propagated_errorgen_layers = self._propagate_errorgen_layers(errorgen_layers, propagation_layers, include_spam)

        return propagated_errorgen_layers
        

    def propagate_errorgens_bch(self, circuit, bch_order=1, bch_layerwise=False, multi_gate_dict=None):
        """
        Propagate all of the error generators for each circuit to the end,
        performing approximation/recombination either along the way (layerwise)
        or at the very end using the BCH approximation.

        Parameters
        ----------
        circuit : `Circuit`
            Circuit to construct a set of post gate error generators for.

        bch_order : int, optional (default 1)
            Order of the BCH approximation to use. When bch_layerwise is True
            this can take the values of either 1 or 2. Otherwise only
            a value of 1 is currently implemented.

        bch_layerwise : bool, optional (default False)
            If True perform the BCH approximation incrementally, performing the 
            approximate recombination layer-by-layer during the course of error
            generator propagation. If False (the default) then the BCH approximation
            is only applied at the very end after all of the error generators have
            been propagated to the end.
 
        multi_gate_dict : dict, optional (default None)
            An optional dictionary mapping between gate name aliases and their
            standard name counterparts.
        """

        msg = 'When bch_layerwise is True this can take the values of either 1 or 2.'\
              +' Otherwise only a value of 1 is currently implemented.'
        if not bch_layerwise:
            assert bch_order==1, msg
        else:
            msg1 = 'When bch_layerwise is False only bch_order values of 1 and 2 are currently'\
                   + ' supported.'
            assert bch_order==1 or bch_order==2, msg1

        #if not doing layerwise BCH then we can re-use `propagate_errorgens` fully.
        if not bch_layerwise:
            propagated_errorgen_layers = self.propagate_errorgens(circuit, multi_gate_dict)
        #otherwise we need to do the error generator layer propagation slightly
        #differently.
        else:
            #start by converting the input circuit into a list of stim Tableaus with the 
            #first element dropped.
            stim_layers = self.construct_stim_layers(circuit, multi_gate_dict, drop_first_layer=True)

            #We next want to construct a new set of Tableaus corresponding to the cumulative products
            #of each of the circuit layers with those that follow. These Tableaus correspond to the
            #clifford operations each error generator will be propagated through in order to reach the
            #end of the circuit.
            propagation_layers = self.construct_propagation_layers(stim_layers)

            #Next we take the input circuit and construct a list of dictionaries, each corresponding
            #to the error generators for a particular gate layer.
            #TODO: Add proper inferencing for number of qubits:
            assert circuit.line_labels is not None and circuit.line_labels != ('*',)
            errorgen_layers = self.construct_errorgen_layers(circuit, len(circuit.line_labels))

            #propagate the errorgen_layers through the propagation_layers to get a list
            #of end of circuit error generator dictionaries.
            propagated_errorgen_layers = self._propagate_errorgen_layers_bch(errorgen_layers, propagation_layers)

        return propagated_errorgen_layers

        
    def propagate_errorgens_nonmarkovian(self, circuit, multi_gate_dict=None, include_spam=True):
        """
        Propagate all of the error generators for each circuit layer to the end without
        any recombinations or averaging. This version also only track the overall modifier/weighting
        factor picked up by each of the final error generators over the course of the optimization,
        with the actual rates introduced in subsequent stages.

        Parameters
        ----------
        circuit : `Circuit`
            Circuit to construct a set of post gate error generators for.

        multi_gate_dict : dict, optional (default None)
            An optional dictionary mapping between gate name aliases and their
            standard name counterparts.

        include_spam : bool, optional (default True)
            If True then we include in the propagation the error generators associated
            with state preparation and measurement.

        Returns
        -------
        propagated_errorgen_layers : list of lists of dictionaries
            A list of lists of dictionaries, each corresponding to the result of propagating
            an error generator layer through to the end of the circuit.

        """

        #TODO: Check for proper handling of empty circuit and length 1 circuits.

        #start by converting the input circuit into a list of stim Tableaus with the 
        #first element dropped.
        stim_layers = self.construct_stim_layers(circuit, multi_gate_dict, drop_first_layer = not include_spam)
        
        #We next want to construct a new set of Tableaus corresponding to the cumulative products
        #of each of the circuit layers with those that follow. These Tableaus correspond to the
        #clifford operations each error generator will be propagated through in order to reach the
        #end of the circuit.
        propagation_layers = self.construct_propagation_layers(stim_layers)

        #Next we take the input circuit and construct a list of dictionaries, each corresponding
        #to the error generators for a particular gate layer.
        #TODO: Add proper inferencing for number of qubits:
        assert circuit.line_labels is not None and circuit.line_labels != ('*',)
        errorgen_layers = self.construct_errorgen_layers(circuit, len(circuit.line_labels), include_spam, 
                                                         include_circuit_time=True, fixed_rate=1)

        #propagate the errorgen_layers through the propagation_layers to get a list
        #of end of circuit error generator dictionaries.
        propagated_errorgen_layers = self._propagate_errorgen_layers(errorgen_layers, propagation_layers, include_spam)

        #in the context of doing propagation for nonmarkovianity we won't be using BCH, so do a partial flattening
        #of this data structure.
        propagated_errorgen_layers = [errorgen_layers[0] for errorgen_layers in propagated_errorgen_layers]

        return propagated_errorgen_layers



    def propagate_errorgens_analytic(self, circuit):
        pass

    def construct_stim_layers(self, circuit, multi_gate_dict=None, drop_first_layer=True):
        """
        Converts a `Circuit` to a list of stim Tableau objects corresponding to each
        gate layer.

        TODO: Move to a tools module? Locality of behavior considerations.

        Parameters
        ----------
        circuit : `Circuit`
            Circuit to convert.
        multi_gate_dict : dict, optional (default None)
            If specified this augments the standard dictionary for conversion between
            pygsti gate labels and stim (found in `pygsti.tools.internalgates.standard_gatenames_stim_conversions`)
            with additional entries corresponding to aliases for the entries of the standard dictionary.
            This is presently used in the context of non-Markovian applications where tracking
            circuit time for gate labels is required.
        drop_first_layer : bool, optional (default True)
            If True the first Tableau for the first gate layer is dropped in the returned output.
            This default setting is what is primarily used in the context of error generator 
            propagation.

        Returns
        -------
        stim_layers : list of `stim.Tableau`
            A list of `stim.Tableau` objects, each corresponding to the ideal Clifford operation
            for each layer of the input pygsti `Circuit`, with the first layer optionally dropped.
        """

        stim_dict=standard_gatenames_stim_conversions()
        if multi_gate_dict is not None:
            for key in multi_gate_dict:
                stim_dict[key]=stim_dict[multi_gate_dict[key]]
        stim_layers=circuit.convert_to_stim_tableau_layers(gate_name_conversions=stim_dict)
        if drop_first_layer and len(stim_layers)>0:
            stim_layers = stim_layers[1:]
        return stim_layers
    
    def construct_propagation_layers(self, stim_layers):
        """
        Construct a list of stim Tableau objects corresponding to the Clifford
        operation each error generator will be propagated through. This corresponds
        to a list of cumulative products of the ideal operations, but in reverse.
        I.e. the initial entry corresponds to the product (in matrix multiplication order)
        of all elements of `stim_layers`, the second entry is the product of the elements of
        `stim_layers[1:]`, then `stim_layers[2:]` and so on until the last entry which is
        `stim_layers[-1]`.

        Parameters
        ----------
        stim_layers : list of stim.Tableau
            The list of stim.Tableau objects corresponding to a set of ideal Clifford
            operation for each circuit layer through which we will be propagating error
            generators.

        Returns
        -------
        propagation_layers : list of `stim.Tableau`
            A list of `stim.Tableau` objects, each corresponding to a cumulative product of 
            ideal Clifford operations for a set of circuit layers, each corresponding to a layer
            of operations which we will be propagating error generators through. 
        """
        if len(stim_layers) > 1:
            propagation_layers = [0]*len(stim_layers)
            #if propagation_layers is empty that means that stim_layers was empty
            #final propagation layer is the final stim layer for the circuit
            propagation_layers[-1] = stim_layers[-1]
            for layer_idx in reversed(range(len(stim_layers)-1)):
                propagation_layers[layer_idx] = propagation_layers[layer_idx+1]*stim_layers[layer_idx]
        elif len(stim_layers) == 1:
            propagation_layers = stim_layers
        else:
            propagation_layers = []
        return propagation_layers
    
    def construct_errorgen_layers(self, circuit, num_qubits, include_spam=True, include_circuit_time=False, fixed_rate=None):
        """
        Construct a nested list of lists of dictionaries corresponding to the error generators for each circuit layer.
        This is currently (as implemented) only well defined for `ExplicitOpModels` where each layer corresponds
        to a single 'gate'. This should also in principle work for crosstalk-free `ImplicitOpModels`, but is not
        configured to do so just yet. The entries of the top-level list correspond to circuit layers, while the entries
        of the second level (i.e. the dictionaries at each layer) correspond to different orders of the BCH approximation.

        Parameters
        ----------
        circuit : `Circuit`
            Circuit to construct the error generator layers for.
        
        num_qubits : int
            Total number of qubits, used for padding out error generator coefficient labels.

        include_spam : bool, optional (default True)
            If True then include the error generators for state preparation and measurement.
        
        include_circuit_time : bool, optional (default False)
            If True then include as part of the error generator coefficient labels the circuit
            time from which that error generator arose.
        
        fixed_rate : float, optional (default None)
            If specified this rate is used for all of the error generator coefficients, overriding the
            value currently found in the model.
        Returns
        -------
        List of lists of dictionaries, each one containing the error generator coefficients and rates for a circuit layer,
        with the error generator coefficients now represented using LocalStimErrorgenLabel.

        """
        #If including spam then start by completing the circuit (i.e. adding in the explicit SPAM labels).
        if include_spam:
            circuit = self.model.complete_circuit(circuit)

        #TODO: Infer the number of qubits from the model and/or the circuit somehow.
        #Pull out the error generator dictionaries for each operation (may need to generalize this for implicit models):
        model_error_generator_dict = dict() #key will be a label and value the lindblad error generator dictionary.
        for op_lbl, op in self.model.operations.items():
            #TODO add assertion that the operation is a lindblad error generator type modelmember.
            model_error_generator_dict[op_lbl] = op.errorgen_coefficients()
        #add in the error generators for the prep and measurement if needed.
        if include_spam:
            for prep_lbl, prep in self.model.preps.items():
                model_error_generator_dict[prep_lbl] = prep.errorgen_coefficients()
            for povm_lbl, povm in self.model.povms.items():
                model_error_generator_dict[povm_lbl] = povm.errorgen_coefficients()

        #TODO: Generalize circuit time to not be in one-to-one correspondence with the layer index.
        error_gen_dicts_by_layer = []
        for j in range(len(circuit)):
            circuit_layer = circuit[j] # get the layer
            #can probably relax this if we detect that the model is a crosstalk free model.
            assert isinstance(circuit_layer, Label), 'Correct support for parallel gates is still under development.'
            errorgen_layer = dict()
            layer_errorgen_coeff_dict = model_error_generator_dict[circuit_layer] #get the errors for the gate
            for errgen_coeff_lbl, rate in layer_errorgen_coeff_dict.items(): #for an error in the accompanying error dictionary 
                #TODO: Can probably replace this function call with `padded_basis_element_labels` method of `GlobalElementaryErrorgenLabel`
                paulis = _eprop.errgen_coeff_label_to_stim_pauli_strs(errgen_coeff_lbl, num_qubits)
                if include_circuit_time:
                    #TODO: Refactor the fixed rate stuff to reduce the number of if statement evaluations.
                    errorgen_layer[_LSE(errgen_coeff_lbl.errorgen_type, paulis, circuit_time=j)] = rate if fixed_rate is None else fixed_rate
                else:
                    errorgen_layer[_LSE(errgen_coeff_lbl.errorgen_type, paulis)] = rate if fixed_rate is None else fixed_rate
            error_gen_dicts_by_layer.append([errorgen_layer])
        return error_gen_dicts_by_layer
    
    def _propagate_errorgen_layers(self, errorgen_layers, propagation_layers, include_spam=True):
        """
        Propagates the error generator layers through each of the corresponding propagation layers
        (i.e. the clifford operations for the remainder of the circuit). This results in a list of 
        lists of dictionaries, where each sublist corresponds to an order of the BCH approximation 
        (when not using the BCH approximation this list will be length 1), and the dictionaries
        correspond to end of circuit error generators and rates.

        Parameters
        ----------
        errorgen_layers : list of lists of dicts
            Each sublist corresponds to a circuit layer, with these sublists containing dictionaries 
            of the error generator coefficients and rates for a circuit layer. Each dictionary corresponds
            to a different order of the BCH approximation (when not using the BCH approximation this list will
            be length 1).  The error generator coefficients are represented using LocalStimErrorgenLabel.

        propagation_layers : list of `stim.Tableau`
            A list of `stim.Tableau` objects, each corresponding to a cumulative product of 
            ideal Clifford operations for a set of circuit layers, each corresponding to a layer
            of operations which we will be propagating error generators through. 

        include_spam : bool, optional (default True)
            If True then include the error generators for state preparation and measurement
            are included in errogen_layers, and the state preparation error generator should
            be propagated through (the measurement one is simply appended at the end).
        
        Returns
        -------
        fully_propagated_layers : list of lists of dicts
            A list of list of dicts with the same structure as errorgen_layers corresponding
            to the results of having propagated each of the error generator layers through
            the circuit to the end.
        """

        #the stopping index in errorgen_layers will depend on whether the measurement error
        #generator is included or not.
        if include_spam:
            stopping_idx = len(errorgen_layers)-2
        else:
            stopping_idx = len(errorgen_layers)-1

        fully_propagated_layers = []    
        for i in range(stopping_idx):
            err_layer = errorgen_layers[i]
            prop_layer = propagation_layers[i]
            new_err_layer = []
            #err_layer should be length 1 if using this method
            for bch_level_list in err_layer: 
                new_error_dict=dict()
                #iterate through dictionary of error generator coefficients and propagate each one.
                for errgen_coeff_lbl in bch_level_list:
                    propagated_error_gen = errgen_coeff_lbl.propagate_error_gen_tableau(prop_layer, bch_level_list[errgen_coeff_lbl])
                    new_error_dict[propagated_error_gen[0]] = propagated_error_gen[1]
                new_err_layer.append(new_error_dict)        
            fully_propagated_layers.append(new_err_layer)
        #add the final layers which didn't require actual propagation (since they were already at the end).
        fully_propagated_layers.extend(errorgen_layers[stopping_idx:])
        return fully_propagated_layers
    
    #TODO: Add an option to return the results with the different BCH order combined.
    def _propagate_errorgen_layers_bch(self, errorgen_layers, propagation_layers, bch_order=1):
        """
        Propagates the error generator layers through each of the corresponding propagation layers
        (i.e. the clifford operations for the remainder of the circuit). In this version we 
        perform a layerwise application of the BCH approximation following each propagation to
        recombine the propaged error generator layer with the layer proceeding it before each
        successive propagation step.

        Parameters
        ----------
        errorgen_layers : list of lists of dicts
            Each sublist corresponds to a circuit layer, with these sublists containing dictionaries 
            of the error generator coefficients and rates for a circuit layer. Each dictionary corresponds
            to a different order of the BCH approximation (when not using the BCH approximation this list will
            be length 1).  The error generator coefficients are represented using LocalStimErrorgenLabel.

        propagation_layers : list of `stim.Tableau`
            A list of `stim.Tableau` objects, each corresponding to a cumulative product of 
            ideal Clifford operations for a set of circuit layers, each corresponding to a layer
            of operations which we will be propagating error generators through. 

        bch_order : int, optional (default 1)
            Order of the BCH approximation to use.
        
        Returns
        -------

        """

        #Add temporary errors when trying to do BCH beyond 1st order while the details of the 2nd order
        #approximation's implementation are sorted out.
        if bch_order != 1:
            msg = 'The implementation of the 2nd order BCH approx is still under development. For now only 1st order is supported.'
            raise NotImplementedError(msg)
        assert all([len(layer)==1 for layer in errorgen_layers]), msg

        fully_propagated_layers=[]
        #initialize a variable as temporary storage of the result
        #of performing BCH on pairwise between a propagater errorgen
        #layer and an unpropagated layer for layerwise BCH.
        if len(errorgen_layers)>0:
            combined_err_layer = errorgen_layers[0]

        for i in range(len(errorgen_layers)-1):
            #err_layer = errorgen_layers[i]
            prop_layer = propagation_layers[i]
            new_err_layer = []
            #err_layer should be length 1 if using this method
            for bch_level_dict in combined_err_layer: 
                new_error_dict = dict()
                #iterate through dictionary of error generator coefficients and propagate each one.
                for errgen_coeff_lbl in bch_level_dict:
                    propagated_error_gen = errgen_coeff_lbl.propagate_error_gen_tableau(prop_layer, bch_level_dict[errgen_coeff_lbl])
                    new_error_dict[propagated_error_gen[0]] = propagated_error_gen[1]
                new_err_layer.append(new_error_dict)
            #next use BCH to combine new_err_layer with the now adjacent layer of errorgen_layers[i+1]
            combined_err_layer = _eprop.bch_approximation(new_err_layer, errorgen_layers[i+1], bch_order=1)

        fully_propagated_layers.append(combined_err_layer)
        return fully_propagated_layers
    
    def errorgen_layer_dict_to_errorgen(self, errorgen_layer, mx_basis='pp'):
        """
        Helper method for converting from an error generator dictionary in the format
        utilized in the `errorgenpropagation` module into a numpy array.

        Parameters
        ----------
        errorgen_layer : dict
            A dictionary containing the error generator coefficients and rates for a circuit layer,
            with the error generator coefficients labels represented using `LocalStimErrorgenLabel`.

        mx_basis : Basis or str, optional (default 'pp')
            Either a `Basis` object, or a string which can be cast to a `Basis`, specifying the
            basis in which to return the error generator.
        Returns
        -------
        errorgen : numpy.ndarray
            Error generator corresponding to input `errorgen_layer` dictionary as a numpy array.
        """

        #Use the keys of errorgen_layer to construct a new `ExplicitErrorgenBasis` with
        #the elements necessary for the construction of the error generator matrix.

        #Construct a list of new errorgen coefficients by looping through the keys of errorgen_layer
        #and converting them to LocalElementaryErrorgenLabels.
        local_errorgen_coeffs = [coeff_lbl.to_local_eel() for coeff_lbl in errorgen_layer.keys()]

        errorgen_basis = _ExplicitElementaryErrorgenBasis(self.model.state_space, local_errorgen_coeffs, basis_1q='PP')
        
        #Stack the arrays and then use broadcasting to weight them according to the rates
        elemgen_matrices_array = _np.stack(errorgen_basis.elemgen_matrices, axis=-1)
        weighted_elemgen_matrices_array = _np.fromiter(errorgen_layer.values(), dtype=_np.double)*elemgen_matrices_array

        #The error generator is then just the sum of weighted_elemgen_matrices_array along the third axis.
        errorgen = _np.sum(weighted_elemgen_matrices_array, axis = 2)

        #finally need to change from the standard basis (which is what the error generator is currently in)
        #to the pauli basis.
        errorgen = _bt.change_basis(errorgen, from_basis='std', to_basis=mx_basis)

        return errorgen



def ErrorPropagatorAnalytic(circ,errorModel,ErrorLayerDef=False,startingErrors=None):
    stim_layers=circ.convert_to_stim_tableau_layers()
    
    if startingErrors is None:
        stim_layers.pop(0)

    propagation_layers=[]
    while len(stim_layers)>0:
        top_layer=stim_layers.pop(0)
        for layer in stim_layers:
            top_layer = layer*top_layer
        propagation_layers.append(top_layer)
    
    if not ErrorLayerDef:
        errorLayers=buildErrorlayers(circ,errorModel,len(circ.line_labels))
    else:
        errorLayers=[[_copy.deepcopy(eg) for eg in errorModel] for i in range(circ.depth)]
    
    if not startingErrors is None:
        errorLayers.insert(0,startingErrors)
    
    fully_propagated_layers=[]
    for (idx,layer) in enumerate(errorLayers):
        new_error_dict=dict()
        if idx <len(errorLayers)-1:

            for error in layer:    
                propagated_error_gen=error.propagate_error_gen_tableau(propagation_layers[idx],1.)
                new_error_dict[error]=propagated_error_gen   
        else:
            for error in layer:
                new_error_dict[error]=(error,1)
        fully_propagated_layers.append(new_error_dict)

    return fully_propagated_layers
    
def InverseErrorMap(errorMap):
    InvertedMap=dict()
    for layer_no,layer in enumerate(errorMap):
        for key in layer:
            if layer[key][0] in InvertedMap:
                errgen=_copy.copy(key)
                errgen.label=layer_no
                InvertedMap[layer[key][0]].append(tuple([errgen,layer[key][1]**(-1)]))
            else:
                errgen=_copy.copy(key)
                errgen.label=layer_no
                InvertedMap[layer[key][0]]=[tuple([errgen,layer[key][1]**(-1)])]
    return InvertedMap

def InvertedNumericMap(errorMap,errorValues):
    numeric_map=dict()
    for layer_no,layer in enumerate(errorMap):
        for key in layer:
            if layer[key][0] in numeric_map and key in errorValues[layer_no]:
                numeric_map[layer[key][0]]+=errorValues[layer_no][key]*layer[key][1]**(-1)
            elif key in errorValues[layer_no]:
                numeric_map[layer[key][0]]=errorValues[layer_no][key]*layer[key][1]**(-1)
            else:
                continue
    return numeric_map


def ErrorPropagator(circ,errorModel,multi_gate_dict=None,bch_order=1,bch_layerwise=False,
                    nonmarkovian=False,multi_gate=False,error_layer_def=False):
    if multi_gate and multi_gate_dict is None:
        multi_gate_dict=dict()
    stim_dict=standard_gatenames_stim_conversions()
    if multi_gate:
        for key in multi_gate_dict:
            stim_dict[key]=stim_dict[multi_gate_dict[key]]
    stim_layers=circ.convert_to_stim_tableau_layers(gate_name_conversions=stim_dict)
    stim_layers.pop(0)  #Immediatly toss the first layer because it is not important,

    propagation_layers=[]
    if not bch_layerwise or nonmarkovian:
        while len(stim_layers) != 0:
            top_layer=stim_layers.pop(0)
            for layer in stim_layers:
                top_layer = layer*top_layer
            propagation_layers.append(top_layer)
    else:
        propagation_layers = stim_layers

    if not error_layer_def:
        errorLayers=buildErrorlayers(circ,errorModel, len(circ.line_labels))
    else:
        errorLayers=[[[_copy.deepcopy(eg) for eg in errorModel]] for i in range(circ.depth)]

    num_error_layers=len(errorLayers)
    
    fully_propagated_layers=[]
    for _ in range(0,num_error_layers-1):
        err_layer=errorLayers.pop(0)
        layer=propagation_layers.pop(0)
        new_error_layer=[]
        for err_order in err_layer:
            new_error_dict=dict()
            for key in err_order:
                propagated_error_gen=key.propagate_error_gen_tableau(layer,err_order[key])
                new_error_dict[propagated_error_gen[0]]=propagated_error_gen[1]
            new_error_layer.append(new_error_dict)
        if bch_layerwise and not nonmarkovian:
            following_layer = errorLayers.pop(0)
            new_errors=BCH_Handler(err_layer,following_layer,bch_order) #This should be new_error_layer as the first arg.
            errorLayers.insert(new_errors,0)
        else:
            fully_propagated_layers.append(new_error_layer)

    fully_propagated_layers.append(errorLayers.pop(0))


    if bch_layerwise and not nonmarkovian:
        final_error=dict()
        for order in errorLayers[0]:
            for error in order:
                if error in final_error:
                    final_error[error]=final_error[error]+order[error]
                else:
                    final_error[error]=order[error]
        return final_error
    
    elif not bch_layerwise and not nonmarkovian:
        simplified_EOC_errors=dict()
        if bch_order == 1:
            for layer in fully_propagated_layers:
                for order in layer:
                    for error in order:
                        if error in simplified_EOC_errors:
                            simplified_EOC_errors[error]=simplified_EOC_errors[error]+order[error]
                        else:
                            simplified_EOC_errors[error]=order[error]

        else:
            Exception("Higher propagated through Errors are not Implemented Yet")
        return simplified_EOC_errors
    
    else:
        return fully_propagated_layers



def buildErrorlayers(circ,errorDict,qubits):
    ErrorGens=[]
    #For the jth layer of each circuit
    for j in range(circ.depth):
        l = circ.layer(j) # get the layer
        errorLayer=dict()
        for _, g in enumerate(l): # for gate in layer l
            gErrorDict = errorDict[g.name] #get the errors for the gate
            p1=qubits*'I' # make some paulis why?
            p2=qubits*'I'
            for errs in gErrorDict: #for an error in the accompanying error dictionary 
                errType=errs[0]
                paulis=[]
                for ind,el in enumerate(g): #enumerate the gate ind =0 is name ind = 1 is first qubit ind = 2 is second qubit
                    if ind !=0:  #if the gate element of concern is not the name
                        p1=p1[:el] + errs[1][ind-1] +p1[(el+1):]
                
                paulis.append(stim.PauliString(p1))
                if errType in "CA":
                    for ind,el in enumerate(g):
                        if ind !=0:
                            p2=p2[:el] + errs[2][ind-1] +p2[(el+1):]
                    paulis.append(stim.PauliString(p2))     
                errorLayer[_LSE(errType,paulis)]=gErrorDict[errs]
        ErrorGens.append([errorLayer])
    return ErrorGens
'''

Inputs:
_______
err_layer (list of dictionaries)
following_layer (list of dictionaries)
bch_order:

'''
def BCH_Handler(err_layer,following_layer,bch_order):         
    new_errors=[]
    for curr_order in range(0,bch_order):
        working_order=dict()
        #add first order terms into new layer
        if curr_order == 0:
            for error_key in err_layer[curr_order]:
                working_order[error_key]=err_layer[curr_order][error_key]
            for error_key in following_layer[curr_order]:
                working_order[error_key]=following_layer[curr_order[error_key]] 
            new_errors.append(working_order)

        elif curr_order ==1:
            working_order={}
            for error1 in err_layer[curr_order-1]:
                for error2 in following_layer[curr_order-1]:
                    errorlist = commute_errors(error1,error2,BCHweight=1/2*err_layer[error1]*following_layer[error2])
                    for error_tuple in errorlist:
                        working_order[error_tuple[0]]=error_tuple[1]
            if len(err_layer)==2:
                for error_key in err_layer[1]:
                    working_order[error_key]=err_layer[1][error_key]
            if len(following_layer)==2:
                for error_key in following_layer[1]:
                    working_order[error_key]=following_layer[1][error_key]
            new_errors.append(working_order)

        else:
            Exception("Higher Orders are not Implemented Yet")
    return new_errors

# There's a factor of a half missing in here. 
def nm_propagators(corr, Elist,qubits):
    Kms = []
    for idm in range(len(Elist)):
        Am=zeros([4**qubits,4**qubits],dtype=complex128)
        for key in Elist[idm][0]:
            Am += key.toWeightedErrorBasisMatrix()
            # This assumes that Elist is in reverse chronological order
        partials = []
        for idn in range(idm, len(Elist)):
            An=zeros([4**qubits,4**qubits],dtype=complex128)
            for key2 in Elist[idn][0]:
                An = key2.toWeightedErrorBasisMatrix()
            partials += [corr[idm,idn] * Am @ An]
        partials[0] = partials[0]/2
        Kms += [sum(partials,0)]
    return Kms

def averaged_evolution(corr, Elist,qubits):
    Kms = nm_propagators(corr, Elist,qubits)
    return multi_dot([expm(Km) for Km in Kms])

def error_stitcher(first_error,second_error):
    link_dict=second_error.pop(0)
    new_errors=[]
    for layer in first_error:
        new_layer=dict()
        for key in layer:
            if layer[key][0] in link_dict:
                new_error=link_dict[layer[key][0]]
                new_layer[key]=(new_error[0],new_error[1]*layer[key][1])
            elif layer[key][0].errorgen_type =='Z':
                new_layer[key]=layer[key]
            else:
                continue
            
        new_errors.append(new_layer)
    for layer in second_error:
        new_errors.append(layer)
    return new_errors