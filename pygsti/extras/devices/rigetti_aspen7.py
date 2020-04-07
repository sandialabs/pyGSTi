""" Specification of Rigetti Aspen 7 """
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

qubits = ['Q' + str(x) for x in range(8)] + ['Q1' + str(x) for x in range(8)]  \
    + ['Q2' + str(x) for x in range(8)] + ['Q3' + str(x) for x in range(8)]

twoQgate = 'Gcphase'

edgelist = [('Q' + str(i), 'Q' + str((i + 1) % 8)) for i in range(8)] \
    + [('Q' + str((i + 1) % 8), 'Q' + str(i)) for i in range(8)] \
    + [('Q1', 'Q16'), ('Q16', 'Q1'), ('Q2', 'Q15'), ('Q15', 'Q2')] \
    + [('Q1' + str(i), 'Q1' + str((i + 1) % 8)) for i in range(8)] \
    + [('Q1' + str((i + 1) % 8), 'Q1' + str(i)) for i in range(8)] \
    + [('Q11', 'Q26'), ('Q26', 'Q11'), ('Q12', 'Q25'), ('Q25', 'Q12')] \
    + [('Q2' + str(i), 'Q2' + str((i + 1) % 8)) for i in range(8)] \
    + [('Q2' + str((i + 1) % 8), 'Q2' + str(i)) for i in range(8)] \
    + [('Q21', 'Q36'), ('Q36', 'Q21'), ('Q22', 'Q35'), ('Q35', 'Q22')] \
    + [('Q3' + str(i), 'Q3' + str((i + 1) % 8)) for i in range(8)] \
    + [('Q3' + str((i + 1) % 8), 'Q3' + str(i)) for i in range(8)]

spec_format = 'rigetti'
