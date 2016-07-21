from .helpers           import get_pylint_output, write_output
from ..automation_tools import read_yaml

def lint_all():
    config = read_yaml('pylint_config.yml')

    print('Linting all of pygsti. This takes around thirty seconds')
    print('  (Report can be found in pylint/output/all.out)')

    blacklisted_warnings    = config['blacklisted-warnings']
    blacklisted_errors      = config['blacklisted-errors']
    whitelisted_refactors   = config['whitelisted-refactors']

    blacklist = blacklisted_warnings + blacklisted_errors
    whitelist = whitelisted_refactors

    commands  = [config['pylint-version'],
                '--disable=R,%s' % ','.join(blacklist),
                '--enable=%s'  % ','.join(whitelist),
                '--rcfile=.lint.conf'] + config['packages']

    output = get_pylint_output(commands, 'all')
    return output
