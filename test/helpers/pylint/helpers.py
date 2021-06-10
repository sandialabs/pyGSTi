import subprocess

from ..automation_tools import directory


def get_pylint_output(commands, filename):
    # pyGSTi directory
    with directory('..'):
        output = get_output(commands)
    print('\n'.join(output))
    # test/output/pylint
    with directory('output/pylint'):
        write_output(output, '%s.out' % filename)
    return output

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
