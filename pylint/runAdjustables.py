#!/usr/bin/env python3
from readyaml   import read_yaml
from helpers    import get_output, write_output
import sys

# A function that lets us adjust the value of the adjustable when linting
def build_commands(adjustable, setting, value):
    config    = read_yaml('config.yml')
    commands  = [config['pylint-version'], 
		 '--enable=%s' % adjustable,
                 '--disable=all',
                 '--rcfile=%s' % config['config-file'],
                 '--reports=n'] + config['packages']
    return commands

if __name__ == "__main__":

    # The wanted size of an output file (ex: too-many-arguments.txt)
    desiredLength= 20
    adjustables = read_yaml('config.yml')['adjustables']

    print('Beginning to lint for adjustable refactoring issues')
    print('Many adjustments indicate a more serious problem, while few/none indicate normal code')

    for adjustable in adjustables:

        print('    Linting for %s' % adjustable)

        default               = adjustables[adjustable]
        setting, defaultvalue = default.rsplit('=')
        defaultvalue          = int(defaultvalue)

        adjust_commands       = lambda value : build_commands(adjustable, setting, value)

        currentvalue          = defaultvalue
        output                = get_output(adjust_commands(currentvalue))
         
        # Adjust the value (ex: max arguments/function) until a properly sized file is generated
        while(len(output) > desiredLength):
            print('      Adjusting the value of %s. Was %s, is now %s' % (setting, currentvalue, currentvalue + defaultvalue))
            currentvalue += defaultvalue 
            output        = get_output(adjust_commands(currentvalue))
 
        print('    Finished linting for %s, the final value of %s was %s\n' % (adjustable, setting, currentvalue))
        # Once the file is satisfactory, write the thing
        write_output(output, 'output/%s.txt' % adjustable)

    print('Linting Complete')
    
    
