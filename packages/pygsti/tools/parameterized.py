from functools import wraps

def parameterized(dec):
    @wraps(dec)
    def layer(*args, **kwargs):
        @wraps(layer)
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer
