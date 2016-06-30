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
    print('Generating %s in all of pygsti. This might take a few minutes' % enabled)
    commands  = ['pylint3', '--disable=all',
                            '--enable=%s' % enabled,
                            '--rcfile=.lint.conf',
                            '--reports=n',
                            '../packages/pygsti']
    output = get_output(commands)
    print('\n'.join(output))
    write_output(output, 'output/%s.out' % sys.argv[1])
