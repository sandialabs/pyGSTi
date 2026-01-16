""" Specification of Rigetti Aspen 6 """
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

qubits = ['Q' + str(x) for x in [10, 11, 12, 13, 14, 15, 16, 17]]

two_qubit_gate = 'Gcphase'

edgelist = [('Q10', 'Q11'),
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
            ]
