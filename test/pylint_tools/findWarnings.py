#!/usr/bin/env python3
from helpers            import get_output, write_output
from ..automation_tools import read_yaml

if __name__ == "__main__":
    print('Generating warnings in all of pygsti. This takes around 30 seconds')
    config    = read_yaml('config.yml')
    blacklist = config['blacklisted-warnings']
    commands  = [config['pylint-version'], 
                 '--disable=R,C,E,%s' % ','.join(blacklist),
                 '--rcfile=%s' % config['config-file'],
                 '--reports=n'] + config['packages']
    output = get_output(commands)
    print('\n'.join(output))
    write_output(output, 'output/warnings.out')
