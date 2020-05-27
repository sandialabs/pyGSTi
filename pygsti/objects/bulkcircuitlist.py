"""
Circuit list for bulk computation
"""
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************
from .circuitstructure import LsGermsStructure as _LsGermsStructure
from .circuit import Circuit as _Circuit


class BulkCircuitList(list):
    """
    A list of :class:`Circuit` objects and associated metadata.

    Parameters
    ----------
    circuit_list_or_structure : list or CircuitStructure
        The list of circuits that constitutes the primary data held by this object.
        If this list is obtained by providing a :class:`CircuitStructure`, this
        object will hold the additional structure information as well.

    op_label_aliases : dict, optional
        Dictionary of circuit meta-data whose keys are operation label "aliases"
        and whose values are circuits corresponding to what that operation label
        should be expanded into before querying the dataset.  Defaults to the
        empty dictionary (no aliases defined).  e.g. op_label_aliases['Gx^3'] =
        pygsti.obj.Circuit(['Gx','Gx','Gx'])

    circuit_weights : numpy.ndarray, optional
        If not None, an array of per-circuit weights (of length equal to the number of
        circuits) that are typically used to multiply the counts extracted for each circuit.

    name : str, optional
        An optional name for this list, used for status messages.
    """
    def __init__(self, circuit_list_or_structure, op_label_aliases=None, circuit_weights=None, name=None):
        """
        Create a BulkCircuitList.

        Parameters
        ----------
        circuit_list_or_structure : list or CircuitStructure
            The list of circuits that constitutes the primary data held by this object.
            If this list is obtained by providing a :class:`CircuitStructure`, this
            object will hold the additional structure information as well.

        op_label_aliases : dict, optional
            Dictionary of circuit meta-data whose keys are operation label "aliases"
            and whose values are circuits corresponding to what that operation label
            should be expanded into before querying the dataset.  Defaults to the
            empty dictionary (no aliases defined).  e.g. op_label_aliases['Gx^3'] =
            pygsti.obj.Circuit(['Gx','Gx','Gx'])

        circuit_weights : numpy.ndarray, optional
            If not None, an array of per-circuit weights (of length equal to the number of
            circuits) that are typically used to multiply the counts extracted for each circuit.

        name : str, optional
            An optional name for this list, used for status messages.
        """
        #validStructTypes = (_objs.LsGermsStructure, _objs.LsGermsSerialStructure)
        if isinstance(circuit_list_or_structure, (list, tuple)):
            circuit_list_or_structure = list(map(_Circuit.cast, circuit_list_or_structure))
            self.circuits_to_use = circuit_list_or_structure
            self.circuit_structure = _LsGermsStructure([], [], [], [], None)  # create a dummy circuit structure
            self.circuit_structure.add_unindexed(circuit_list_or_structure)   # which => "no circuit structure"
        else:  # assume a circuit structure
            self.circuit_structure = circuit_list_or_structure
            self.circuits_to_use = self.circuit_structure.allstrs

        self.op_label_aliases = op_label_aliases
        self.circuit_weights = circuit_weights
        self.name = name  # an optional name for this circuit list
        self[:] = self.circuits_to_use  # maybe get rid of self.circuits_to_use in the future...
