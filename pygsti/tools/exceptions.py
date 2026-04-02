"""
Defines GST exception classes
"""
#***************************************************************************************************
# Copyright 2015, 2019, 2025 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************


class GSTRuntimeError(Exception):
    """
    Gate Set Tomography run-time exception class.
    """
    pass


class GSTValueError(Exception):
    """
    Gate Set Tomography value error exception class.
    """
    pass


class MissingDependencyWarning(UserWarning):
    """
    Inform the user that we're missing an optional dependency they
    PROBABLY want, but isn't strictly required.
    """
    pass


class DeprecatedPositionalArgumentsWarning(UserWarning):
    """
    Inform the user that they're using positional arguments
    that should be specified as keyword arguments.
    """
    pass


class NumericalDomainWarning(UserWarning):
    """
    Inform the user that some mathematical function is being applied on
    an input that's slightly outside of its usual domain. E.g., we're
    computing the fidelity between states (x, y) where trace(x) < 1.
    """
    pass


class pyGSTiDeprecationWarning(UserWarning, DeprecationWarning):
    """
    A helper class so users (and pyGSTi developers) can distinguish
    between deprecation warnings raised by us versus by other
    libraries.
    """
    pass


class ForwardSimulatorSuitabilityWarning(UserWarning):
    """
    Inform the user that they should consider using a different
    forward simulator class in a given context.
    """
    pass


class ImplicitlyDoneEditingCircuitWarning(UserWarning):
    """
    Inform the user that a Circuit.__hash__ has been called
    on a Circuit with Circuit.editable == True. This is often
    triggered when performing a check like 
    `if c in dict_of_circuits: ...`.
    """
    pass


class PrepareThyself(UserWarning):
    """ 
    Indicate to the user that there's good reason to expect that what
    they're about to do will fail.
    """
    pass


class UnknownGaugeSpaceDimension(UserWarning):
    """ 
    Inform the user that we weren't sure of the dimension of gauge
    space in current model parameterization. We use some kind of a
    default value in these cases instead.
    """
    pass


class CVXPYFailure(UserWarning):
    """ 
    Numerical solvers dispatched by CVXPY failed when trying
    to solve a pyGSTi-constructed problem. We have fallback
    behavior in these cases.
    """
    pass


class UntouchedModelNoiseKey(UserWarning):
    """ 
    Alert the user that they may have incorrectly
    constructed an OpModelNoise object.
    """
    pass


class OverparameterizationWarning(UserWarning):
    """ 
    Signal that the maximal model has fewer parameters than
    the current model.
    """
    pass


class UnnamedReportWarning(UserWarning):
    """ 
    Signal that we're generating a name for a report automatically
    and randomly, since a user didn't provide a name.
    """
    pass


class StolenResourceWarning(UserWarning):
    """
    Suppose we have two types with a parent-child relationship, Foo and Bar.
    In the event of a call sequence like

        B = Bar()        # sets B.parent = None
        F = Foo(child=B) # sets F.child = B and updates B.parent = F
        G = make_foo(B)  # sets G.child = B and udpates B.parent = G,

    the `make_foo` function should raise a StolenResourceWarning if it
    changes the value of `id(F.child)`.
    """
    pass
