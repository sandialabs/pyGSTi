""" Specification of IBM Q Rueschlikon (aka ibmqx5) """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

qubits = ['Q' + str(x) for x in range(16)]

two_qubit_gate = 'Gcnot'

edgelist = [('Q1', 'Q0'),
            ('Q1', 'Q2'),
            ('Q2', 'Q3'),
            ('Q3', 'Q4'),
            ('Q3', 'Q14'),
            ('Q5', 'Q4'),
            ('Q6', 'Q5'),
            ('Q6', 'Q7'),
            ('Q6', 'Q11'),
            ('Q7', 'Q10'),
            ('Q8', 'Q7'),
            ('Q9', 'Q8'),
            ('Q9', 'Q10'),
            ('Q11', 'Q10'),
            ('Q12', 'Q5'),
            ('Q12', 'Q11'),
            ('Q12', 'Q13'),
            ('Q13', 'Q4'),
            ('Q13', 'Q14'),
            ('Q15', 'Q0'),
            ('Q15', 'Q2'),
            ('Q15', 'Q14')]

spec_format = 'ibmq_v2018'
