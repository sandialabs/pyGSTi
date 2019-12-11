""" Functions for interfacing pyGSTi with RQC Aspen-4 and Aspen-6 """
# flake8: noqa  # When this functionality is ready to merge for real, remove this line
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

qubits = ['Q' + str(x) for x in [0, 1, 2, 3, 4, 5, 6, 7]] + ['Q' + str(x) for x in [10, 11, 12, 13, 14, 15, 16, 17]]

twoQgate = 'Gcphase'

edgelist = [  # Ring 1
    ('Q0', 'Q1'),
    ('Q1', 'Q0'),
    ('Q1', 'Q2'),
    ('Q2', 'Q1'),
    ('Q2', 'Q3'),
    ('Q3', 'Q2'),
    ('Q3', 'Q4'),
    ('Q4', 'Q3'),
    ('Q4', 'Q5'),
    ('Q5', 'Q4'),
    ('Q5', 'Q6'),
    ('Q6', 'Q5'),
    ('Q6', 'Q7'),
    ('Q7', 'Q6'),
    ('Q0', 'Q7'),
    ('Q7', 'Q0'),
    # Ring 2
    ('Q10', 'Q11'),
    ('Q11', 'Q10'),
    ('Q11', 'Q12'),
    ('Q12', 'Q11'),
    ('Q12', 'Q13'),
    ('Q13', 'Q12'),
    ('Q13', 'Q14'),
    ('Q14', 'Q13'),
    ('Q14', 'Q15'),
    ('Q15', 'Q14'),
    ('Q15', 'Q16'),
    ('Q16', 'Q15'),
    ('Q16', 'Q17'),
    ('Q17', 'Q16'),
    ('Q10', 'Q17'),
    ('Q17', 'Q10'),
    # Connection
    ('Q1', 'Q16'),
    ('Q16', 'Q1'),
    ('Q2', 'Q15'),
    ('Q15', 'Q2')]

spec_format = 'rigetti'

# def qubits(version):

#     if version == 6:
#         return ['Q'+str(x) for x in [10,11,12,13,14,15,16,17]]
#     elif version == 4:
#         return ['Q'+str(x) for x in [0,1,2,3,4,5,6,7]] + ['Q'+str(x) for x in [10,11,12,13,14,15,16,17]]
#     elif version == 4.13:
#         return ['Q'+str(x) for x in [0,1,2,3,6,7]] + ['Q'+str(x) for x in [10,11,12,14,15,16,17]]

#     else:
#         raise ValueError("Unknown version!")

# def make_processor_spec(one_q_gate_names, version, construct_clifford_compilations = {'paulieq' : ('1Qcliffords',),
#                         'absolute': ('paulis','1Qcliffords')}, verbosity=0):

#     gate_names = ['Gcphase'] + one_q_gate_names
#     if version == 4:
#         total_qubits = 16
#     elif version == 4.13:
#         total_qubits = 13
#     elif version == 6:
#         total_qubits = 8
#     else:
#         raise ValueError("Unknown version!")

#     qubit_labels = qubits(version)

#     cphase_edge_list = get_twoQgate_edgelist(version)
#     availability = {'Gcphase':cphase_edge_list}
#     pspec = _pspec.ProcessorSpec(total_qubits, gate_names, availability=availability,
#                                      construct_clifford_compilations=construct_clifford_compilations,
#                                      verbosity=verbosity, qubit_labels=qubit_labels)
#     return pspec

# def get_twoQgate_edgelist(version, subset=None):
#     """
#     The edgelist for the CPHASE gates in Agave. If subset is None this is
#     all the CPHASE gates in the device; otherwise it only includes the qubits in
#     the subset (qubits are labelled 'Qi' for i = 0,1,2...).
#     """
#     if version == 4:
#         edge_list = [# Ring 1
#                      ('Q0','Q1'),
#                      ('Q1','Q2'),
#                      ('Q2','Q3'),
#                      ('Q3','Q4'),
#                      ('Q4','Q5'),
#                      ('Q5','Q6'),
#                      ('Q6','Q7'),
#                      ('Q0','Q7'),
#                      # Ring 2
#                      ('Q10','Q11'),
#                      ('Q11','Q12'),
#                      ('Q12','Q13'),
#                      ('Q13','Q14'),
#                      ('Q14','Q15'),
#                      ('Q15','Q16'),
#                      ('Q16','Q17'),
#                      ('Q10','Q17'),
#                      # Connection
#                      ('Q1','Q16'),
#                      ('Q2','Q15'),
#                      ]

#     if version == 4.13:
#         edge_list = [# broken ring 1
#                      ('Q0','Q1'),
#                      ('Q1','Q2'),
#                      ('Q2','Q3'),
#                      ('Q6','Q7'),
#                      ('Q0','Q7'),
#                      # broken ring 2
#                      ('Q10','Q11'),
#                      ('Q11','Q12'),
#                      ('Q14','Q15'),
#                      ('Q15','Q16'),
#                      ('Q16','Q17'),
#                      ('Q10','Q17'),
#                      # partial connection
#                      ('Q2','Q15'),
#                      ]

#     elif version == 6:
#         edge_list = [('Q10','Q11'),
#                      ('Q11','Q12'),
#                      ('Q12','Q13'),
#                      ('Q13','Q14'),
#                      ('Q14','Q15'),
#                      ('Q15','Q16'),
#                      ('Q16','Q17'),
#                      ('Q10','Q17'),
#                      ]

#     else:
#         raise ValueError("Unknown version!")

#     if subset is None:
#         return edge_list

#     else:
#         subset_edge_list = []
#         for edge in edge_list:
#             if edge[0] in subset and edge[1] in subset:
#                 subset_edge_list.append(edge)

#         return subset_edge_list

# def get_splitting(version, n):
#     """
#     Splits the qubits into 16/n dijoint sets each consisting of n
#     connected qubits, with set starting at qubit Q+str(startnode)
#     """
#     if version == 4:
#         if n == 2:
#             qubit_sets = [('Q0','Q1'),('Q2','Q3'),('Q4','Q5'),('Q6','Q7'),('Q10','Q11'),('Q12','Q13'),('Q14','Q15'),('Q16','Q17')]
#         elif n == 4:
#             qubit_sets = [('Q0','Q1','Q2','Q3'),('Q4','Q5','Q6','Q7'),('Q10','Q11','Q12','Q13'),('Q14','Q15','Q16','Q17')]
#         elif n == 8:
#             qubit_sets = [('Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7'),('Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17')]
#         elif n == 16:
#             qubit_sets = [('Q0','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17')]
#         else:
#             raise ValueError("This value is not allowed!")

#     elif version == 6:
#         if n == 2:
#             qubit_sets = [('Q10','Q11'),('Q12','Q13'),('Q14','Q15'),('Q16','Q17')]
#         elif n == 4:
#             qubit_sets = [('Q10','Q11','Q12','Q13'),('Q14','Q15','Q16','Q17')]
#         elif n == 8:
#             qubit_sets = [('Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17')]
#         else:
#             raise ValueError("This value is not allowed!")

#     else:
#         raise ValueError("Unknown version!")

#     return qubit_sets

# def get_all_connected_sets(version, n):
#     """

#     """

#     pspec = make_processor_spec(['Gxpi2', 'Gxmpi2', 'Gxpi','Gzpi2', 'Gzmpi2', 'Gzpi', 'Gypi'], version,
#                                  construct_clifford_compilations = {})
#     import itertools as _iter
#     connectedqubits = []
#     for combo in _iter.combinations(pspec.qubit_labels, n):
#         if pspec.qubitgraph.subgraph(list(combo)).are_glob_connected(combo):
#             connectedqubits.append(combo)

#     # if version == 4:
#     #     assert(n == 2), "Only implemented for n = 2"
#     #     if n == 2:
#     #         qubit_sets = get_twoQgate_edgelist(version)

#     # elif version == 6:
#     #     if n < 8: qubit_sets = [tuple(['Q1'+str((q + i) % 8) for i in range(n)]) for q in range(8)]
#     #     if n == 8: qubit_sets = [tuple(['Q1'+str(i) for i in range(8)])]
#     # else:
#     #     raise ValueError("Unknown version!")

#     return connectedqubits
