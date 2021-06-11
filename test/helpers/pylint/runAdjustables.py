from .helpers import get_pylint_output
from ..automation_tools import read_json


# A function that lets us adjust the value of the adjustable when linting
def build_commands(adjustable, setting, value):
    config    = read_json('config/pylint_config.json')
    commands  = [config['pylint-version'],
                 '--enable=%s' % adjustable,
                 '--disable=all',
                 '--rcfile=%s' % config['config-file'],
                 '--reports=n'] + config['packages']
    return commands

def run_adjustables(desiredLength=20):
    # The wanted size of an output file (ex: too-many-arguments.txt)
    adjustables = read_json('config/pylint_config.yml')['adjustables']

    print('Beginning to lint for adjustable refactoring issues')
    print('Many adjustments indicate a more serious problem, while few/none indicate normal code')

    for adjustable in adjustables:

        print('    Linting for %s' % adjustable)

        default               = adjustables[adjustable]
        setting, defaultvalue = default.rsplit('=')
        defaultvalue          = int(defaultvalue)

        adjust_commands       = lambda value : build_commands(adjustable, setting, value)

        currentvalue          = defaultvalue
        output                = get_pylint_output(adjust_commands(currentvalue), adjustable)

        # Adjust the value (ex: max arguments/function) until a properly sized file is generated
        while(len(output) > desiredLength):
            print('      Adjusting the value of %s. Was %s, is now %s' % (setting, currentvalue, currentvalue + defaultvalue))
            currentvalue += defaultvalue
            output        = get_pylint_output(adjust_commands(currentvalue), adjustable) # implicit write

        print('    Finished linting for %s, the final value of %s was %s\n' % (adjustable, setting, currentvalue))

    print('Linting Complete')
