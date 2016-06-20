from __future__ import print_function
import time, sys, os

# allows creation of decorators that take arguments
def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

@parametrized
def benchmark(function, filename):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result    = function(*args, **kwargs)
        endTime   = time.time()
        totalTime = endTime - startTime

        # Log the amount of time the function took to finish
        info = 'The function %s in the module %s took %s seconds to run' % \
            (function.__name__, function.__module__, totalTime)
        # filename exists here no matter what
        with open(filename, 'a') as output:
            output.write(info + '\n')

        return result
    return wrapper
