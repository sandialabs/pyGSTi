#!/usr/bin/env python3
from helpers            import get_pylint_output
from ..automation_tools import read_yaml

def find_errors():
    print('Generating errors in all of pygsti. This takes around 30 seconds')
    config    = read_yaml('pylint_config.yml')
    blacklist = config['blacklisted-errors']
    commands  = [config['pylint-version'], 
                 '--disable=W,R,C,I,%s' % ','.join(blacklist),
                 '--rcfile=%s' % config['config-file'],
                 '--reports=n'] + config['packages']
    output = get_pylint_output(commands, 'errors')
