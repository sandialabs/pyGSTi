from ..automation_tools import directory, get_files
from shutil import copyfile
import os


def update_if_higher(dictionary, key, value):
    if key in dictionary:
        if dictionary[key] < value:
            print(dictionary[key], value)
            dictionary[key] = value
    else:
        dictionary[key] = value

# Generate dict of form: { modulename : percentage } for percentages below threshold
def find_uncovered(packageDict, threshold=90):
    uncoveredDict = {}
    for filename, fileDict in packageDict.items():
        for caseDict in fileDict.values():
            for testinfo, _ in caseDict.values():
                for modulename, moduleinfo in testinfo.items():
                    update_if_higher(uncoveredDict, modulename, moduleinfo[0])
    uncoveredDict = { key : value for (key, value) in uncoveredDict.items() if value < threshold }
    return uncoveredDict


# Generate dict of form: { modulename : percentage } for percentages below threshold
def find_uncovered_lines(packageDict):
    uncoveredLineDict = {}
    for filename, fileDict in packageDict.items():
        for caseDict in fileDict.values():
            for testinfo, _ in caseDict.values():
                for modulename, moduleinfo in testinfo.items():
                    if modulename not in uncoveredLineDict:
                        uncoveredLineDict[modulename] = set(moduleinfo[1])
                    else:
                        uncoveredLineDict[modulename] = uncoveredLineDict[modulename].intersection(set(moduleinfo[1]))
    return uncoveredLineDict


def annotate_uncovered(packageName, infoDict):
    uncoveredLineDict = find_uncovered_lines(infoDict[packageName])
    with directory('../packages/pygsti/%s' % packageName):
        files = [filename for filename in get_files(os.getcwd()) if filename in uncoveredLineDict]
        print(files)
        for filename in files:
            copyfile(filename, filename + '.bak')
            with open(filename, 'r') as source:
                content = source.read().splitlines()

            uncoveredLines = uncoveredLineDict[filename]
            for i, line in enumerate(content):
                content[i].replace('#!*uncovered*!', '') # Remove old annotations
                if (i+1) in uncoveredLines:
                    content[i] += '#!*uncovered*!'
            print('Annotating %s' % filename)
            with open(filename, 'w') as source:
                source.write('\n'.join(content))
