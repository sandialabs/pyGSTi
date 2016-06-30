#!/usr/bin/env python3
from helpers  import get_output, write_output
from readyaml import read_yaml

if __name__ == "__main__":
    print('Generating errors in all of pygsti. This might take a few minutes')
    blacklist = read_yaml('config.yml')['blacklisted-errors']
    commands  = ['pylint3', '--disable=W,R,C,I,%s' % ','.join(blacklist),
                            '--rcfile=.lint.conf',
                            '--reports=n',
                            '../packages/pygsti']
    output = get_output(commands)
    print('\n'.join(output))
    write_output(output, 'output/errors.out')
