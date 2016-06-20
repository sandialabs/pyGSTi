from __future__ import print_function
from runModule  import run_module
from benchmarks import benchmark
from helpers    import tool, get_args
import os, sys

def benchmark_template(fullpath, command):
    directory, name = fullpath.rsplit('/', 1)
    @benchmark
    def template():
        command(fullpath)
    _, time = template()
    return time

def benchmark_module(moduleName):
    return benchmark_template(moduleName, run_module)

def benchmark_file(filename):
    run_file = lambda filename : os.system('python %s' % filename)
    return benchmark_template(filename, run_file)

@tool
def run_benchmarks(names, output=None):
    # build modulenames and filenames
    _, moduleNames, _ = os.walk(os.getcwd()).next()
    fileNames = {}
    for subdir, dirs, files in os.walk(os.getcwd()):
        for filename in files:
            if filename.endswith('.py') and filename.startswith('test'):
                fileNames[filename] = subdir + os.sep + filename

    benchDict = {}

    for name in names:
        if name in moduleNames:
            benchDict[name] = benchmark_module(os.getcwd() + '/' + name)
        elif name in fileNames:
            # send the full filepath to benchmark_file()
            benchDict[name] = benchmark_file(fileNames[name])
        else:
            print('%s is neither a valid modulename, nor a valid filename' % name)

    if output != None:
        with open(output, 'w') as benchfile:
            for key in benchDict:
                info = '%s | %s\n' % (key.ljust(20), benchDict[key])
                benchfile.write(info)
    return benchDict

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    run_benchmarks(*args, **kwargs)
