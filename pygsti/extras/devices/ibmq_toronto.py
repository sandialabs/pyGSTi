""" Specification of IBM Q Toronto """
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

edgelist = [
    # 1st row of connections
    ('Q0', 'Q1'), ('Q1', 'Q0'),
    ('Q1', 'Q4'), ('Q4', 'Q1'),
    ('Q4', 'Q7'), ('Q7', 'Q4'),
    ('Q7', 'Q10'), ('Q10', 'Q7'),
    ('Q10', 'Q12'), ('Q12', 'Q10'),
    ('Q12', 'Q15'), ('Q15', 'Q12'),
    ('Q15', 'Q18'), ('Q18', 'Q15'),
    ('Q18', 'Q21'), ('Q21', 'Q18'),
    ('Q21', 'Q23'), ('Q23', 'Q21'),
    # 2nd row of connections
    ('Q3', 'Q5'), ('Q5', 'Q3'),
    ('Q5', 'Q8'), ('Q8', 'Q5'),
    ('Q8', 'Q11'), ('Q11', 'Q8'),         
    ('Q11', 'Q14'), ('Q14', 'Q11'),     
    ('Q14', 'Q16'), ('Q16', 'Q14'),     
    ('Q16', 'Q19'), ('Q19', 'Q16'),     
    ('Q19', 'Q22'), ('Q22', 'Q19'),  
    ('Q22', 'Q25'), ('Q25', 'Q22'),  
    ('Q25', 'Q26'), ('Q26', 'Q25'),           
    # 1st column of connections
    ('Q1', 'Q2'), ('Q2', 'Q1'),
    ('Q2', 'Q3'), ('Q3', 'Q2'),
    # 2nd column of connections   
    ('Q6', 'Q7'), ('Q7', 'Q6'), 
    ('Q8', 'Q9'), ('Q9', 'Q8'), 
    # 3rd column of connections  
    ('Q12', 'Q13'), ('Q13', 'Q12'),                
    ('Q13', 'Q14'), ('Q14', 'Q13'),       
    # 4th column of connections           
    ('Q17', 'Q18'), ('Q18', 'Q17'), 
    ('Q19', 'Q20'), ('Q20', 'Q19')
]

spec_format = 'ibmq_v2019'