from ..automation_tools import directory, get_files
from shutil import copyfile
import os

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

    def do_annotate(line):
        return ('\\' not in line and 
                line.replace(' ', '') != '' and
                'pylint' not in line)

    uncoveredLineDict = find_uncovered_lines(infoDict[packageName])
    with directory('../packages/pygsti/%s' % packageName):
        files = [filename for filename in get_files(os.getcwd()) if filename in uncoveredLineDict]
        for filename in files:
            copyfile(filename, filename + '.bak')
            with open(filename, 'r') as source:
                content = source.read().splitlines()

            uncoveredLines = uncoveredLineDict[filename]
            for i, line in enumerate(content):
                content[i] = content[i].replace('# uncovered!', '') # Remove old annotations
                if (i+1) in uncoveredLines and do_annotate(content[i]):
                    content[i] = content[i].ljust(100) + ' # uncovered!'
            print('Annotating %s' % filename)
            with open(filename, 'w') as source:
                source.write('\n'.join(content))
