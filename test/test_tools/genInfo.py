from __future__   import print_function
from helpers      import *
from runBenchmark import benchmark_coverage
import os, sys

@tool
def gen_info(names, output=None, package=''):
    # build modulenames and filenames
    fileNames   = get_file_names()
    moduleNames = get_module_names()

    infoDict = {}

    for name in names:
        if name in moduleNames:
            infoDict[name] = benchmark_coverage(os.getcwd() + '/' + name)
        elif name in fileNames:
            # send the full filepath to benchmark_file()
            infoDict[name] = benchmark_coverage(fileNames[name])
        else:
            print('%s is neither a valid modulename, nor a valid filename' % name)

    infoDict = { key : ('%s%% coverage | %s seconds' % infoDict[key]) for key in infoDict }

    if output != None:
        write_formatted_table(output, infoDict.items())

    return infoDict

if __name__ == "__main__":
    args, kwargs = get_args(sys.argv)
    gen_info(*args, **kwargs)
