""" Defines the 'parameterized' decorator of decorators """
from __future__ import division, print_function, absolute_import, unicode_literals
#*****************************************************************
#    pyGSTi 0.9:  Copyright 2015 Sandia Corporation
#    This Software is released under the GPL license detailed
#    in the file "license.txt" in the top-level pyGSTi directory
#*****************************************************************

from functools import wraps

def parameterized(dec):
    """
    Used to create decorator functions that take arguments.  Functions
    decorated with this function (which should be decorators themselves
    but can have more than the standard single function argument), get
    morphed into a standard decorator function.
    """
    @wraps(dec)
    def _decorated_dec(*args, **kwargs): # new function that replaces dec, and returns a *standard* decorator function
        @wraps(_decorated_dec)
        def _standard_decorator(f): #std decorator (function that replaces f) that calls dec with more args
            return dec(f, *args, **kwargs)
        return _standard_decorator
    return _decorated_dec # function this replaces the action of dec
