TO DO:
    - Verify that we are writing down the Paulis in the correct order (i.e., should an XY error on qubits 0 and 1 be "XY" or "YX"? This has implications for interfacing with Ashe's code and for the layer snipper. Looks like pyGSTi does not reverse index Paulis?)
    - Write unit tests for ml.newtools.up_to_weight_k_error_gens_from_qubit_graph
    - Change the LocalDense to use a relu activation function in its final layer
    - See if you are learning the true error vectors. It would be cool if you were!
    - It is possible that training on good/very good circuits leads to better predcitons on okay circuits. Rationale: you learn the real error vectors this way, because the first-order approximation is very good.