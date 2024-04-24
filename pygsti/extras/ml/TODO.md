TO DO:
    - Model biased readout error:
        - ~~Easy: give the network the target bitstring as a tensor.~~
        - Harder: individual networks learn errors for each qubit, networks get the target bitstring for that qubit. Add up the failure rates (or predict multiplied success probability and do one minus the result.)
        - Harder: predict an error rate vector for the measurements (UPDATE: create_input_data to pre-pad the P and S matrices)
            - Would it be faster to learn all of the measurement error vectors first and then send them off?
    - Explore the data: see how networks trained on subsets of the data work.
    - Verify that we are writing down the Paulis in the correct order (i.e., should an XY error on qubits 0 and 1 be "XY" or "YX"? This has implications for interfacing with Ashe's code, for the layer snipper, and screening Z-type errors. Looks like pyGSTi does not reverse index Paulis?)
    - ~~Verify that the circuit encoding code works. It's odd because of all the transpositions!~~
    - Write unit tests for ml.newtools.up_to_weight_k_error_gens_from_qubit_graph
    - ~~Change the LocalDense to use a relu activation function in its final layer.~~
    - See if you are learning the true error vectors. It would be cool if you were!
    - Test out encoding idle qubits for the experimental data.

QUESTIONS:
    - It is possible that training on good/very good circuits leads to better predcitons on okay circuits. Rationale: you learn the real error vectors this way, because the first-order approximation is very good.
    - What happens if IBM runs circuits simultaneously (so the circuit doesn't actually store all the information needed to contextualize its errors?)
        - This would appear as poor performance on few-qubit circuits.
    - Do we need a mask to stop the network from trying to create error vectors for the padded part of a circuit? 