"""
The KrausOperatorInterface class and supporting functionality.
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


class KrausOperatorInterface(object):
    """
    Adds an interface for extracting the Kraus operator(s) of an operation (quantum map).
    """

    @classmethod
    def from_kraus_operators(cls, kraus_operators, basis='pp', evotype="default", state_space=None):
        """
        Create an operation by specifying its Kraus operators.

        Parameters
        ----------
        kraus_operators : list
            A list of numpy arrays, each of which specifyies a Kraus operator.

        basis : str or Basis, optional
            The basis in which the created operator's superoperator representation is in.

        evotype : Evotype or str, optional
            The evolution type.  The special value `"default"` is equivalent
            to specifying the value of `pygsti.evotypes.Evotype.default_evotype`.

        state_space : StateSpace, optional
            The state space for this operation.  If `None` a default state space
            with the appropriate number of qubits is used.
        """
        raise NotImplementedError("Derived classes must implement the from_kraus_operators method!")

    def __init__(self):
        pass

    @property
    def kraus_operators(self):
        """A list of this operation's Kraus operators as numpy arrays."""
        raise NotImplementedError("Derived classes must implement the kraus_operators property!")

    def set_kraus_operators(self, kraus_operators):
        """
        Set the parameters of this operation by specifying its Kraus operators.

        Parameters
        ----------
        kraus_operators : list
            A list of numpy arrays, each of which specifyies a Kraus operator.

        Returns
        -------
        None
        """
        raise NotImplementedError("Derived classes must implement the set_kraus_operators method!")

    @property
    def num_kraus_operators(self):
        """The number of Kraus operators in the Kraus decomposition of this operation."""
        return len(self.kraus_operators)
