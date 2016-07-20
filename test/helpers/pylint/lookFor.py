from .helpers           import get_pylint_output, write_output
from ..automation_tools import read_yaml
import os, sys

# https://docs.pylint.org/features.html#general-options

def find(items, filename):
    enabled   = ','.join(items)
    print('Generating %s in all of pygsti. This should take less than a minute' % enabled)
    config    = read_yaml('pylint_config.yml')
    commands  = [config['pylint-version'], 
                 '--disable=all',
                 '--enable=%s' % enabled,
                 '--rcfile=%s' % config['config-file'],
                 '--reports=n'] + config['packages']
    output = get_pylint_output(commands, filename) # implicitly puts to screen/saves to file

def look_for(args):
    if len(args) == 0:
        print('Please supply a filename and list of things to check for. (see https://docs.pylint.org/features.html#general-options)')
        sys.exit(1)
    # If only one argument is supplied, assume it is both the filename and the itemname
    elif len(sys.argv) == 2:
        sys.argv.append(sys.argv[1])

    find(sys.argv[2:], sys.argv[1])
