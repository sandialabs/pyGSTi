#!/usr/bin/env python3
from helpers  import get_output, write_output
from readyaml import read_yaml
import sys

# https://docs.pylint.org/features.html#general-options

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print('Please supply a filename and list of things to check for. (see https://docs.pylint.org/features.html#general-options)')
        sys.exit(1)
    elif len(sys.argv) == 2:
        sys.argv.append(sys.argv[1])

    enabled   = ','.join(sys.argv[2:])
    print('Generating %s in all of pygsti. This should take less than a minute' % enabled)

    config    = read_yaml('config.yml')
    commands  = [config['pylint-version'], 
                 '--disable=all',
                 '--enable=%s' % enabled,
                 '--rcfile=%s' % config['config-file'],
                 '--reports=n'] + config['packages']
    
    output = get_output(commands)
    print('\n'.join(output))
    write_output(output, 'output/%s.out' % sys.argv[1])
