#!/usr/bin/env python3
from readyaml   import read_yaml
import subprocess
import sys

# A wrapper for subprocess
def get_output(commands):
    try:
        output = subprocess.check_output(commands)
        return output.decode('utf-8').splitlines()
    except subprocess.CalledProcessError as e:
        return e.output.decode('utf-8').splitlines()

def write_output(output, filename):
    with open(filename, 'w') as outputfile:
        outputfile.write("\n".join(output))

# A function that lets us adjust the value of the adjustable when linting
def build_commands(adjustable, setting, value, package='packages/pygsti'):
	return ['pylint3', '--disable=all',
				   '--enable=%s' % adjustable,
				   '--%s=%s'     % (setting, value),
				   '--reports=n',
				   '../%s' % package]

if __name__ == "__main__":

    # The wanted size of an output file (ex: too-many-arguments.txt)
    desiredLength= 20
    package      = 'packages/pygsti' if len(sys.argv) == 1 else sys.argv[1]

    adjustables = read_yaml('config.yml')['adjustables']

    for adjustable in adjustables:

        default = adjustables[adjustable]
        setting, defaultvalue = default.rsplit('=')

        # Wrapper around build_commands for adjusting the value of the adjustable's setting :)
        adjust_commands = lambda value : build_commands(adjustable, setting, value, package)

        currentvalue = defaultvalue
        output = get_output(adjust_commands(currentvalue))
         
        # Adjust the value (ex: max arguments/function) until a properly sized file is generated
        while(len(output) > desiredLength):
            currentvalue += defaultvalue
            output = get_output(adjust_commands(currentvalue))
 
        # Once the file is satisfactory, write the thing
        write_output(output, 'output/%s.txt' % adjustable)
    
    
