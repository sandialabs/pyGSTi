from __future__ import print_function
import os, sys

show_message = lambda message : print('\n\n%s\n\n%s%s\n\n%s\n\n' % ('#' * 80, ' ' * 30, message, '#' * 80))

def tool(function):
    def wrapper(*args, **kwargs):
        owd = os.getcwd() # Handle moving between directories
        os.chdir('..')
        function(*args, **kwargs)
        os.chdir(owd)
    return wrapper

def get_args(rawArgs):
    args      = [[arg for arg in rawArgs[1:] if not arg.startswith('--')]] # create args
    optionals = [arg for arg in rawArgs[1:] if arg.startswith('--')]
    kwargs    = {}
    # create kwargs
    for optional in optionals:
        kv = optional[2:].split('=') # remove prepending '--' and seperate into key : value
        kwargs[kv[0]] = kv[1]

    return args, kwargs
