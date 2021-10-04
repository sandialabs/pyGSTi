""" Specification of IBM Q Cambridge """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

qubits = ['Q' + str(x) for x in range(28)]

two_qubit_gate = 'Gcnot'

edgelist = [('Q0', 'Q1'),
            ('Q1', 'Q0'),
            ('Q1', 'Q2'),
            ('Q2', 'Q1'),
            ('Q2', 'Q3'),
            ('Q3', 'Q2'),
            ('Q3', 'Q4'),
            ('Q4', 'Q3'),
            ('Q4', 'Q6'),
            ('Q6', 'Q4'),
            ('Q0', 'Q5'),
            ('Q5', 'Q0'),
            ('Q6', 'Q13'),
            ('Q13', 'Q6'),
            ('Q5', 'Q9'),
            ('Q9', 'Q5'),
            ('Q7', 'Q8'),
            ('Q8', 'Q7'),
            ('Q8', 'Q9'),
            ('Q9', 'Q8'),
            ('Q9', 'Q10'),
            ('Q10', 'Q9'),
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
            ('Q7', 'Q16'),
            ('Q16', 'Q7'),
            ('Q11', 'Q17'),
            ('Q17', 'Q11'),
            ('Q15', 'Q18'),
            ('Q18', 'Q15'),
            ('Q16', 'Q19'),
            ('Q19', 'Q16'),
            ('Q17', 'Q23'),
            ('Q23', 'Q17'),
            ('Q18', 'Q27'),
            ('Q27', 'Q18'),
            ('Q19', 'Q20'),
            ('Q20', 'Q19'),
            ('Q20', 'Q21'),
            ('Q21', 'Q20'),
            ('Q21', 'Q22'),
            ('Q22', 'Q21'),
            ('Q22', 'Q23'),
            ('Q23', 'Q22'),
            ('Q23', 'Q24'),
            ('Q24', 'Q23'),
            ('Q24', 'Q25'),
            ('Q25', 'Q24'),
            ('Q25', 'Q26'),
            ('Q26', 'Q25'),
            ('Q26', 'Q27'),
            ('Q27', 'Q26'),
            ]

spec_format = 'ibmq_v2019'
