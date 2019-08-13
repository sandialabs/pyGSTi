""" Functions for interfacing pyGSTi with ibmqx5 """
from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

import numpy as _np
from ...objects import processorspec as _pspec

def qubits():

    return ['Q' + str(x) for x in range(14)]

def make_processor_spec(one_q_gate_names, construct_clifford_compilations = {'paulieq' : ('1Qcliffords',),
                        'absolute': ('paulis', '1Qcliffords')}, verbosity=0):
    """
    todo

    """
    total_qubits = 14
    gate_names = ['Gcnot'] + one_q_gate_names
    qubit_labels = qubits()
    cnot_edge_list = get_twoQgate_edgelist()
    availability = {'Gcnot':cnot_edge_list}
    pspec = _pspec.ProcessorSpec(total_qubits, gate_names, availability=availability,
                                     construct_clifford_compilations=construct_clifford_compilations,
                                     verbosity=verbosity, qubit_labels=qubit_labels)
    return pspec

def get_twoQgate_edgelist(subset=None):
    """
    The edgelist for the CNOT gates in IBMQX5. If subset is None this is
    all the CNOTs in the device; otherwise it only includes the qubits in
    the subset (qubits are labelled 'Qi' for i = 0,1,2...).
    """
    cnot_edge_list = [('Q1','Q0'),
                      ('Q1','Q2'),
                      ('Q2','Q3'),
                      ('Q4','Q3'),
                      ('Q5','Q4'),
                      ('Q5','Q6'),
                      ('Q6','Q8'),
                      ('Q7','Q8'),
                      ('Q9','Q8'),
                      ('Q9','Q10'),
                      ('Q11','Q10'),
                      ('Q11','Q12'),
                      ('Q13','Q12'),
                      ('Q13','Q1'),
                      ('Q12','Q2'),
                      ('Q11','Q3'),
                      ('Q4','Q10'),
                      ('Q5','Q9')]
    
    if subset is None:    
        return cnot_edge_list
    
    else:
        subset_cnot_edge_list = []
        for cnot_edge in cnot_edge_list:
            if cnot_edge[0] in subset and cnot_edge[1] in subset:
                subset_cnot_edge_list.append(cnot_edge)
        
        return subset_cnot_edge_list