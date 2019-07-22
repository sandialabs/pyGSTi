""" Defines the 'parameterized' decorator of decorators """
from __future__ import division, print_function, absolute_import, unicode_literals
#***************************************************************************************************
# Copyright 2015, 2019 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
# Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights
# in this software.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 or in the LICENSE file in the root pyGSTi directory.
#***************************************************************************************************

from functools import wraps


def parameterized(dec):
    """
    Used to create decorator functions that take arguments.  Functions
    decorated with this function (which should be decorators themselves
    but can have more than the standard single function argument), get
    morphed into a standard decorator function.
    """
    @wraps(dec)
    def _decorated_dec(*args, **kwargs):  # new function that replaces dec, and returns a *standard* decorator function
        @wraps(_decorated_dec)
        def _standard_decorator(f):  # std decorator (function that replaces f) that calls dec with more args
            return dec(f, *args, **kwargs)
        return _standard_decorator
    return _decorated_dec  # function this replaces the action of dec
