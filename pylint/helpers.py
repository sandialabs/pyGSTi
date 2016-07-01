import subprocess

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

