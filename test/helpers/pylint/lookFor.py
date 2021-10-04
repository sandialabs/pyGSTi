import sys

from .helpers import get_pylint_output
from ..automation_tools import read_json


# https://docs.pylint.org/features.html#general-options

def find(items, filename, coreonly):
    enabled   = ','.join(items)
    print('Generating %s in all of pygsti%s. This should take less than a minute' %
          (enabled, " (core only)" if coreonly else ""))
    config    = read_json('config/pylint_config.json')
    commands  = [config['pylint-version'],
                 '--disable=all',
                 '--enable=%s' % enabled,
                 '--rcfile=%s' % config['config-file'],
                 '--reports=n'] + (['pygsti'] if coreonly else config['packages'])
    output = get_pylint_output(commands, filename) # implicitly puts to screen/saves to file

def look_for(args, coreonly=True):
    if len(args) == 0:
        print('Please supply a filename and list of things to check for. (see https://docs.pylint.org/features.html#general-options)')
        sys.exit(1)
    # If only one argument is supplied, assume it is both the filename and the itemname
    elif len(sys.argv) == 2:
        sys.argv.append(sys.argv[1])

    find(sys.argv[2:], sys.argv[1], coreonly)
