from __future__ import print_function
import time, sys, os

def benchmark(function):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result    = function(*args, **kwargs)
        endTime   = time.time()
        totalTime = endTime - startTime

        return result, totalTime
    return wrapper
