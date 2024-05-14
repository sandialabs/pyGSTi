"""
Defines the CircuitList class, for holding meta-data alongside a list or tuple of Circuits.
"""
# ***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
# ***************************************************************************************************
import copy as _copy
import uuid as _uuid
import numpy as _np

from pygsti.baseobjs.nicelyserializable import NicelySerializable as _NicelySerializable
from pygsti.circuits.circuit import Circuit as _Circuit
from pygsti.tools import listtools as _lt


class CircuitList(_NicelySerializable):
    """
    A unmutable list (a tuple) of :class:`Circuit` objects and associated metadata.

    Parameters
    ----------
    circuits : list
        The list of circuits that constitutes the primary data held by this object.

    op_label_aliases : dict, optional
        Dictionary of circuit meta-data whose keys are operation label "aliases"
        and whose values are circuits corresponding to what that operation label
        should be expanded into before querying the dataset.  Defaults to the
        empty dictionary (no aliases defined).  e.g. op_label_aliases['Gx^3'] =
        pygsti.baseobjs.Circuit(['Gx','Gx','Gx'])

    circuit_weights : numpy.ndarray, optional
        If not None, an array of per-circuit weights (of length equal to the number of
        circuits) that are typically used to multiply the counts extracted for each circuit.

    name : str, optional
        An optional name for this list, used for status messages.
    """

    @classmethod
    def cast(cls, circuits):
        """
        Convert (if needed) an object into a :class:`CircuitList`.

        Parameters
        ----------
        circuits : list or CircuitList
            The object to convert.

        Returns
        -------
        CircuitList
        """
        if isinstance(circuits, CircuitList):
            return circuits
        return cls(circuits)

    def __init__(self, circuits, op_label_aliases=None, circuit_rules=None, circuit_weights=None, name=None):
        """
        Create a CircuitList.

        Parameters
        ----------
        circuits : list
            The list of circuits that constitutes the primary data held by this object.

        op_label_aliases : dict, optional
            Dictionary of circuit meta-data whose keys are operation label "aliases"
            and whose values are circuits corresponding to what that operation label
            should be expanded into before querying the dataset.  Defaults to the
            empty dictionary (no aliases defined).  e.g. op_label_aliases['Gx^3'] =
            pygsti.baseobjs.Circuit(['Gx','Gx','Gx'])

        circuit_rules : list, optional
            A list of `(find,replace)` 2-tuples which specify circuit-label replacement
            rules.  Both `find` and `replace` are tuples of operation labels (or `Circuit` objects).

        circuit_weights : numpy.ndarray, optional
            If not None, an array of per-circuit weights (of length equal to the number of
            circuits) that are typically used to multiply the counts extracted for each circuit.

        name : str, optional
            An optional name for this list, used for status messages.
        """
        super().__init__()
        self._circuits = tuple(map(_Circuit.cast, circuits))  # *static* container - can't add/append
        self.op_label_aliases = op_label_aliases
        self.circuit_rules = circuit_rules
        self.circuit_weights = circuit_weights
        self.name = name  # an optional name for this circuit list
        self.uuid = _uuid.uuid4()  # like a persistent id(), useful for peristent (file) caches

    def _to_nice_serialization(self):  # memo holds already serialized objects
        from pygsti.io.writers import convert_circuits_to_strings as _convert_circuits_to_strings
        state = super()._to_nice_serialization()
        state.update({'name': self.name,
                      'op_label_aliases': _convert_circuits_to_strings(self.op_label_aliases),
                      'circuit_rules': _convert_circuits_to_strings(self.circuit_rules),
                      'circuits': [c.str for c in self._circuits],
                      'circuit_weights': list(self.circuit_weights) if (self.circuit_weights is not None) else None,
                      'uuid': str(self.uuid)
                      })
        return state

    @classmethod
    def _from_nice_serialization(cls, state):  # memo holds already de-serialized objects
        from pygsti.io.readers import convert_strings_to_circuits as _convert_strings_to_circuits
        from pygsti.io import stdinput as _stdinput
        std = _stdinput.StdInputParser()
        circuits = [std.parse_circuit(s, create_subcircuits=_Circuit.default_expand_subcircuits)
                    for s in state['circuits']]
        circuit_weights = _np.array(state['circuit_weights'], 'd') if (state['circuit_weights'] is not None) else None
        op_label_aliases = _convert_strings_to_circuits(state['op_label_aliases'])
        circuit_rules = _convert_strings_to_circuits(state['circuit_rules'])
        ret = cls(circuits, op_label_aliases, circuit_rules, circuit_weights, state['name'])
        ret.uuid = _uuid.UUID(state['uuid'])
        return ret

    # Mimic list / tuple
    def __len__(self):
        return len(self._circuits)

    def __getitem__(self, index):
        return self._circuits[index]

    def __iter__(self):
        yield from self._circuits

    def apply_aliases(self):
        """
        Applies any operation-label aliases to this circuit list.

        Returns
        -------
        list
            A list of :class:`Circuit` objects.
        """
        return _lt.apply_aliases_to_circuits(self._circuits, self.op_label_aliases)

    def truncate(self, circuits_to_keep):
        """
        Builds a new circuit list containing only a given subset.

        This can be safer then just creating a new :class:`CircuitList`
        because it preserves the aliases, etc., of this list.

        Parameters
        ----------
        circuits_to_keep : list or set
            The circuits to retain in the returned circuit list.

        Returns
        -------
        CircuitList
        """
        if isinstance(circuits_to_keep, set):
            new_circuits = list(filter(lambda c: c in circuits_to_keep, self._circuits))
        else:
            current_circuits = set(self._circuits)
            new_circuits = list(filter(lambda c: c in current_circuits, circuits_to_keep))
        return CircuitList(new_circuits, self.op_label_aliases)  # don't transfer weights or name

    def truncate_to_dataset(self, dataset):
        """
        Builds a new circuit list containing only those elements in `dataset`.

        Parameters
        ----------
        dataset : DataSet
            The dataset to check.  Aliases are applied to the circuits in
            this circuit list before they are tested.

        Returns
        -------
        CircuitList
        """
        ret = _copy.deepcopy(self)  # so this works on derived classes too (e.g. PlaquetteGridCircuitStructure)
        if dataset is None: return ret

        ret.circuit_weights = ret.name = None  # don't transfer weights or name
        dataset_circuits = set(dataset.keys())
        circuits_in_dataset = [c for c, aliased_c in zip(self._circuits, self.apply_aliases())
                               if aliased_c in dataset_circuits]
        return ret.truncate(circuits_in_dataset)  # uses truncate method, potentially of derived class

    def __hash__(self):
        if self.uuid is not None:
            return hash(self.uuid)
        else:
            raise TypeError('Use digest hash')

    def __eq__(self, other):
        #Compare with non-CircuitLists as lists
        if isinstance(other, CircuitList):
            return self.uuid == other.uuid
        else:
            return self._circuits == other

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)
        if 'uuid' not in state_dict:  # backward compatibility
            self.uuid = _uuid.uuid4()  # create a new uuid
