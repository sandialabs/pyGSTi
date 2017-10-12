from functools import wraps

def parameterized(dec):
    """
    Used to create decorator functions that take arguments.  Functions
    decorated with this function (which should be decorators themselves
    but can have more than the standard single function argument), get
    morphed into a standard decorator function.
    """
    @wraps(dec)
    def decorated_dec(*args, **kwargs): # new function that replaces dec, and returns a *standard* decorator function
        @wraps(decorated_dec)
        def standard_decorator(f): #std decorator (function that replaces f) that calls dec with more args
            return dec(f, *args, **kwargs)
        return standard_decorator
    return decorated_dec # function this replaces the action of dec
