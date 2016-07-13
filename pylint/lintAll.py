#!/usr/bin/env python3
from helpers  import get_output, write_output
from readyaml import read_yaml

def lint_all():
    config = read_yaml('config.yml')

    print('Linting all of pygsti. This might take a few minutes')
 
    blacklisted_warnings    = config['blacklisted-warnings']
    blacklisted_errors      = config['blacklisted-errors']
    whitelisted_conventions = config['whitelisted-conventions']
    whitelisted_refactors   = config['whitelisted-refactors']

    blacklist = blacklisted_warnings + blacklisted_errors
    whitelist = whitelisted_conventions + whitelisted_refactors 

    commands  = ['pylint3', '--disable=R,%s' % ','.join(blacklist),
                            '--enable=%s'  % ','.join(whitelist),
                            '--rcfile=.lint.conf',
                            '../packages/pygsti']

    output = get_output(commands)
    print('\n'.join(output))
    write_output(output, 'output/all.out')
    return output

if __name__ == "__main__":
    lint_all()
