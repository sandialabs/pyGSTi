#!/usr/bin/env python3
from helpers  import get_output, write_output
from readyaml import read_yaml
import sys

if __name__ == "__main__":

    enabled   = ','.join(sys.argv[1:])
    print('Generating %s in all of pygsti. This might take a few minutes' % enabled)
    commands  = ['pylint3', '--disable=all',
                            '--enable=%s' % enabled,
                            '--rcfile=.lint.conf',
                            '--reports=n',
                            '../packages/pygsti']
    output = get_output(commands)
    print('\n'.join(output))
    write_output(output, 'output/specific.out')
