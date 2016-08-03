from .helpers           import get_pylint_output
from ..automation_tools import read_json

def find_errors(coreonly=True):
    print('Generating errors in all of pygsti. This takes around 30 seconds')
    config    = read_json('config/pylint_config.json')
    blacklist = config['blacklisted-errors']
    commands  = [config['pylint-version'],
                 '--disable=W,R,C,I,%s' % ','.join(blacklist),
                 '--rcfile=%s' % config['config-file']] + (['pygsti'] if coreonly else config['packages'])
    output = get_pylint_output(commands, 'errors')
    return output
