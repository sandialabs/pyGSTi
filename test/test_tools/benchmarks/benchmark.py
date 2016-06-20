from __future__ import print_function
import time, sys, os

# # allows creation of decorators that take arguments
# def parametrized(dec):
#     def layer(*args, **kwargs):
#         def repl(f):
#             return dec(f, *args, **kwargs)
#         return repl
#     return layer
#
# @parametrized

def benchmark(function):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result    = function(*args, **kwargs)
        endTime   = time.time()
        totalTime = endTime - startTime

        # Log the amount of time the function took to finish
        #info = 'The Benchmark took %s seconds\n' % (totalTime)

        #with open(filename, 'a') as output:
        #    output.write(info)

        return result, totalTime
    return wrapper
