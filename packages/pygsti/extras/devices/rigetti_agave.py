""" Functions for interfacing pyGSTi with RQC Agave """
from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# #***************************************************************************************************

# import numpy as _np
# from ...objects import processorspec as _pspec

qubits = ['Q'+str(x) for x in range(8)]

twoQgate = 'Gcphase'

edgelist = [('Q0','Q1'),
             ('Q1','Q2'),
             ('Q2','Q3'),
             ('Q3','Q4'),
             ('Q4','Q5'),
             ('Q5','Q6'),
             ('Q6','Q7'),
             ('Q7','Q0'),
             ]

spec_format = 'rigetti'

# def qubits():

#     return ['Q'+str(x) for x in range(8)]

# def make_processor_spec(one_q_gate_names, construct_clifford_compilations = {'paulieq' : ('1Qcliffords',), 
#                         'absolute': ('paulis','1Qcliffords')}, verbosity=0):
#     total_qubits = 8
#     gate_names = ['Gcphase'] + one_q_gate_names
#     qubit_labels = qubits()
#     cphase_edge_list = get_twoQgate_edgelist()
#     availability = {'Gcphase':cphase_edge_list}
#     pspec = _pspec.ProcessorSpec(total_qubits, gate_names, availability=availability,
#                                      construct_clifford_compilations=construct_clifford_compilations,
#                                      verbosity=verbosity, qubit_labels=qubit_labels)
#     return pspec

# def get_twoQgate_edgelist(subset=None):
#     """
#     The edgelist for the CPHASE gates in Agave. If subset is None this is
#     all the CPHASE gates in the device; otherwise it only includes the qubits in
#     the subset (qubits are labelled 'Qi' for i = 0,1,2...).
#     """
#     edge_list = [('Q0','Q1'),
#                  ('Q1','Q2'),
#                  ('Q2','Q3'),
#                  ('Q3','Q4'),
#                  ('Q4','Q5'),
#                  ('Q5','Q6'),
#                  ('Q6','Q7'),
#                  ('Q7','Q0'),
#                  ]
    
#     if subset is None:    
#         return edge_list
    
#     else:
#         subset_edge_list = []
#         for edge in edge_list:
#             if edge[0] in subset and edge[1] in subset:
#                 subset_edge_list.append(edge)
        
#         return subset_edge_list
    
# def get_splitting(n,startnode=0):
#     """
#     Splits the qubits into 8/n dijoint sets each consisting of n 
#     connected qubits, with set starting at qubit Q+str(startnode)
#     """
#     if n == 2:
#         qubit_sets = [('Q'+str((startnode+i) % 8),'Q'+str((startnode+i+1) % 8)) for i in [0,2,4,6]] 
#     elif n == 4:
#         qubit_sets = [tuple(['Q'+str((startnode+q) % 8) for q in range(4)]), 
#                       tuple(['Q'+str((startnode+4+q) % 8) for q in range(4)])]                  
#     elif n == 8:
#         qubit_sets = [tuple(['Q'+str(i) for i in range(8)])] 
#     else:
#         raise ValueError("This value is not allowed!")
#         return 
    
#     return qubit_sets