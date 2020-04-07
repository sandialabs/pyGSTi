""" Circuit list for bulk computation """
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from pygsti.objects.circuitstructure import LsGermsStructure as _LsGermsStructure


# XXX should this be refactored into any related circuit/circuit-structure module?
class BulkCircuitList(list):
    def __init__(self, circuit_list_or_structure, op_label_aliases=None, circuit_weights=None, name=None):

        #validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
        if isinstance(circuit_list_or_structure, (list, tuple)):
            self.circuits_to_use = circuit_list_or_structure
            self.circuit_structure = _LsGermsStructure([], [], [], [], None)  # create a dummy circuit structure
            self.circuit_structure.add_unindexed(circuit_list_or_structure)   # which => "no circuit structure"
        else:
            self.circuit_structure = circuit_list_or_structure
            self.circuits_to_use = self.circuit_structure.allstrs

        self.op_label_aliases = op_label_aliases
        self.circuit_weights = circuit_weights
        self.name = name  # an optional name for this circuit list
        self[:] = self.circuits_to_use  # maybe get rid of self.circuits_to_use in the future...
